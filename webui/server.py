#!/usr/bin/env python3
"""Sol-H3 Studio: a small web front end for NVIDIA Research's Sol-H3-Spark pipeline (unofficial).

- One job at a time (FIFO queue). Models stay loaded between jobs with the same task, orientation and length
  (one Pipeline session). A different combination closes the session and loads the models again.
- Tasks: t2va (text), fl2va (first and/or last frame), ref2va (reference images, videos, audio).
- Output: 1344x768 (landscape) or 768x1344 (portrait), 24 fps, 3-15 s, with audio.

Run it with cwd = the patched Sol-H3-Spark package (models/minimax_h3/Sol-H3-Spark in the patched Sana checkout):
it imports runtime.config and runtime.pipeline from there. Standard library only; this process needs no torch.
bin/sol-h3-studio does all of this for you.

Environment:
  SOL_H3_SPARK_RUNTIME_ROOT  required. Uploads and session outputs go under it (the Qwen container sees only it).
  SOLH3_HOME                 default ~/sol-h3. Only used for the default manifest path.
  SOLH3_PATHS                manifest path with {task}; default $SOLH3_HOME/paths-{task}.json
  SOLH3_STATE_DIR            jobs.db (every clip; jobs.json from before 0.2 is imported once) and
                             load-times.json; default ~/.local/state/sol-h3-studio
  SOLH3_PUBLISH_DIR          optional: also copy each finished clip (and its poster) here
  SOLH3_HOST, SOLH3_PORT     default 127.0.0.1 and 8190. 0.0.0.0 serves the whole LAN; there is no login.
  SOLH3_PRELOAD              1 (default): at start, warm up the settings of the last finished clip. 0: off.
  SOLH3_LAUNCHER_PID         set by bin/sol-h3-studio: the server stops (closing its pipeline) when that process dies.

SOL_H3_SECONDS and SOL_H3_ORIENTATION are infer.py defaults. The studio passes length and format with every job, so
the server removes both from its environment at start; otherwise a 5 s job would render at SOL_H3_SECONDS.
"""
import email.parser
import email.policy
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

TASKS = ("t2va", "fl2va", "ref2va")


def fail(message):
    print(f"sol-h3-studio: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


# infer.py defaults (see the docstring); the studio sets length and format per job
IGNORED_ENV = {name: os.environ.pop(name) for name in ("SOL_H3_SECONDS", "SOL_H3_ORIENTATION") if name in os.environ}

PKG = Path.cwd()
sys.path.insert(0, str(PKG))
try:
    from runtime.config import load_paths, normalize_case  # noqa: E402
    from runtime.pipeline import Pipeline  # noqa: E402
    from runtime.geometry import ORIENTATIONS  # noqa: E402  (exists only in the patched package)
except ImportError as error:
    fail(f"cannot import the patched Sol-H3-Spark runtime from {PKG} ({error}). Start the server with the "
         "working directory set to models/minimax_h3/Sol-H3-Spark inside the patched Sana checkout.")

if not os.environ.get("SOL_H3_SPARK_RUNTIME_ROOT"):
    fail("SOL_H3_SPARK_RUNTIME_ROOT is not set. Point it at your runtime directory "
         "(the Qwen container mounts it; see bin/env.example).")
RUNTIME = Path(os.environ["SOL_H3_SPARK_RUNTIME_ROOT"]).expanduser()
if not RUNTIME.is_absolute():
    fail(f"SOL_H3_SPARK_RUNTIME_ROOT must be an absolute path, not {RUNTIME}")

HOME = Path(os.environ.get("SOLH3_HOME") or Path.home() / "sol-h3").expanduser()
PATHS_TEMPLATE = os.environ.get("SOLH3_PATHS") or str(HOME / "paths-{task}.json")
if "{task}" not in PATHS_TEMPLATE:
    fail(f"SOLH3_PATHS must contain {{task}}, e.g. {HOME}/paths-{{task}}.json (got {PATHS_TEMPLATE})")
PATHS = {t: Path(PATHS_TEMPLATE.format(task=t)).expanduser() for t in TASKS}
STATE_DIR = Path(os.environ.get("SOLH3_STATE_DIR") or Path.home() / ".local/state/sol-h3-studio").expanduser()
DB_FILE = STATE_DIR / "jobs.db"
OLD_STATE_FILE = STATE_DIR / "jobs.json"         # before 0.2: the newest 200 jobs; imported once
LOAD_TIMES = STATE_DIR / "load-times.json"
PUBLISH = Path(os.environ["SOLH3_PUBLISH_DIR"]).expanduser() if os.environ.get("SOLH3_PUBLISH_DIR") else None
HOST = os.environ.get("SOLH3_HOST") or "127.0.0.1"
PORT = int(os.environ.get("SOLH3_PORT") or "8190")
PRELOAD = os.environ.get("SOLH3_PRELOAD", "1") == "1"

INPUTS = RUNTIME / "inputs" / "webui"
SESSIONS = RUNTIME / "outputs" / "webui"
for d in (INPUTS, SESSIONS, STATE_DIR) + ((PUBLISH,) if PUBLISH else ()):
    d.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()          # guards jobs / session
wake = threading.Event()
jobs = []                        # every job, oldest first (also in jobs.db)
PAGE = 24                        # clips per gallery page (/api/state carries the newest page, /api/jobs older ones)
session = {"pipeline": None, "key": None, "state": "idle", "since": time.time()}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


db = None                        # sqlite3 connection to DB_FILE (open_db); writes are serialized by db_lock
db_lock = threading.Lock()


def as_time(value):
    """A job's 'created' as a finite float (0 for anything else, e.g. from a hand-edited jobs.json)."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return 0.0
    return t if math.isfinite(t) else 0.0


def open_db():
    """Open (or create) jobs.db and return every job, oldest first, with strictly increasing 'created' (the
    gallery's paging cursor). Imports jobs.json from before 0.2 once; it stays as it was (no longer written)."""
    global db
    db = sqlite3.connect(DB_FILE, check_same_thread=False, isolation_level=None)    # autocommit
    db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, data TEXT NOT NULL)")
    # user_version 1: jobs.json has been imported (or there was none). Set in the import's own transaction, so a
    # failed or interrupted import is tried again at the next start.
    if db.execute("PRAGMA user_version").fetchone()[0] == 0:
        try:
            old = json.loads(OLD_STATE_FILE.read_text()) if OLD_STATE_FILE.exists() else []
            old = [j for j in old if isinstance(j, dict) and isinstance(j.get("id"), str)]
            for j in old:
                j["created"] = as_time(j.get("created"))
            rows = [(j["id"], j["created"], json.dumps(j)) for j in old]
            with db_lock:
                db.execute("BEGIN")
                try:
                    db.executemany("INSERT OR IGNORE INTO jobs VALUES (?, ?, ?)", rows)
                    db.execute("PRAGMA user_version = 1")
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
            if old:
                log(f"imported {len(old)} jobs from {OLD_STATE_FILE} into {DB_FILE}")
        except (OSError, ValueError, TypeError, sqlite3.Error) as error:
            log("could not import", OLD_STATE_FILE, f"({error}); trying again at the next start")
    loaded = [json.loads(row[0]) for row in db.execute("SELECT data FROM jobs ORDER BY created, rowid")]
    nudged, last = [], None
    for j in loaded:
        j["created"] = as_time(j.get("created"))
        if last is not None and j["created"] <= last:      # equal or out-of-order times: 1 ms apart, in load order
            j["created"] = last + 0.001
            nudged.append(j)
        last = j["created"]
    for j in nudged:
        save_job(j)
    return loaded


def save_job(job):
    """Write one job to jobs.db (its current state)."""
    try:
        with db_lock:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)",
                       (job["id"], job["created"], json.dumps(dict(job))))   # dict(): a consistent copy
    except sqlite3.Error:
        traceback.print_exc()


def ready_tasks():
    """Tasks whose path manifest exists (install/70-manifests.sh writes them)."""
    return [t for t in TASKS if PATHS[t].is_file()]


def load_eta(key):
    """Last measured cold start for this (task, orientation, seconds), else for any setting, else 4 min."""
    try:
        times = json.loads(LOAD_TIMES.read_text())
    except (OSError, ValueError):
        return 240
    return times.get("/".join(map(str, key))) or times.get("last") or 240


def warm_time(key):
    """Last measured clip time for this setting with the models already loaded, else None."""
    try:
        return json.loads(LOAD_TIMES.read_text()).get("warm/" + "/".join(map(str, key)))
    except (OSError, ValueError):
        return None


def remember_load(key, seconds, warm=False):
    """Record a cold start (load + one clip), or with warm=True a clip in a loaded session."""
    try:
        times = json.loads(LOAD_TIMES.read_text()) if LOAD_TIMES.exists() else {}
        if warm:
            times["warm/" + "/".join(map(str, key))] = seconds
        else:
            times["/".join(map(str, key))] = times["last"] = seconds
        LOAD_TIMES.write_text(json.dumps(times, indent=1))
    except (OSError, ValueError):
        traceback.print_exc()


def new_pipeline(key, root):
    task, orientation, seconds = key
    return Pipeline(load_paths(PATHS[task], task=task), root, task=task,
                    orientation=orientation, seconds=None if seconds == 5 else seconds)   # 5 s = frozen route


def close_session(reason):
    p = session["pipeline"]
    if p is not None:
        log("closing session", session["key"], "-", reason)
        session["state"] = "unloading"
        try:
            p.close()
        except Exception:
            traceback.print_exc()
    session.update(pipeline=None, key=None, state="idle", since=time.time(), loading_key=None, preloading=False,
                   takeover_until=None, load_eta=None, warm_s=None)


def run_job(job):
    key = (job["task"], job["orientation"], job.get("seconds", 5))
    case = {"case_id": job["case_id"], "prompt": job["prompt"], "seed": job["seed"], "task": job["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if job.get(k):
            case[k] = job[k]
    case = normalize_case(case)
    if session["key"] != key:
        close_session(f"switching to {key}")
        job["status"] = "loading models + generating"
        job["phase_since"] = time.time()
        job["load_eta"] = load_eta(key)
        session.update(state="loading", since=time.time(), loading_key=key)
        root = SESSIONS / f"{time.strftime('%Y%m%d-%H%M%S')}-{key[0]}-{key[1]}-{key[2]}s"
        p = new_pipeline(key, root)
        session["pipeline"] = p
        t0 = time.time()
        try:
            row = p.start(case, deliver=True)     # the loading pass renders this job (no throwaway warm-up)
        except BaseException:
            close_session("model load failed")
            raise
        job["cold_start_s"] = round(time.time() - t0, 1)   # load + this clip
        job["includes_model_load"] = True
        remember_load(key, job["cold_start_s"])
        session.update(key=key, state="ready", since=time.time(), loading_key=None)
    else:
        job["status"] = "generating"
        job["phase_since"] = time.time()
        session["state"] = "busy"
        t0 = time.time()
        try:
            row = session["pipeline"].generate(case)
        except BaseException:
            close_session("request failed (worker died)")
            raise
        session["state"] = "ready"
        remember_load(key, round(time.time() - t0, 1), warm=True)
    src = Path(row["output"])
    make_poster(src)
    job.update(output=str(src), e2e_s=round(row["e2e_s"], 1), stage1_s=round(row.get("stage1_s", 0), 1),
               stage2_s=round(row.get("stage2_s", 0), 1))
    if PUBLISH:
        dst = PUBLISH / (f"{time.strftime('%Y%m%d-%H%M%S')}_{job['task']}_{job['orientation']}_"
                         f"{job.get('seconds', 5)}s_seed{job['seed']}_{job['id'][:6]}.mp4")
        try:
            shutil.copy2(src, dst)
            if src.with_suffix(".jpg").is_file():
                shutil.copy2(src.with_suffix(".jpg"), dst.with_suffix(".jpg"))
            job["published"] = str(dst)
        except OSError as error:
            log("could not copy the clip to SOLH3_PUBLISH_DIR:", error)


def make_poster(mp4):
    """Poster frame (~1 s in) for the gallery; best effort."""
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(mp4), "-frames:v", "1",
                        "-vf", "scale=640:-2", "-q:v", "4", str(Path(mp4).with_suffix(".jpg"))],
                       timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def clip_file(job):
    """A finished job's MP4: the copy in its session directory, or the SOLH3_PUBLISH_DIR copy once the session
    directory has been pruned."""
    for key in ("output", "published"):
        if job and job.get(key) and Path(job[key]).is_file():
            return Path(job[key])
    return None


preload = {"pending": None}      # (key, case) to warm up when the app starts and nothing is queued


def preload_candidate():
    """The last finished clip's settings, as a warm-up case (its prompt and inputs, seed 999).
    None when there is no such clip, its manifest is gone or its uploaded inputs are gone."""
    last = next((j for j in reversed(jobs) if j.get("status") == "done"), None)
    if last is None or not PATHS.get(last["task"], Path("/nonexistent")).is_file():
        return None
    case = {"case_id": "preload", "prompt": last["prompt"], "seed": 999, "task": last["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if last.get(k):
            case[k] = last[k]
    media = [case.get("first_frame"), case.get("last_frame")] + [r.get("path") for r in case.get("references", [])]
    if any(m and not Path(m).is_file() for m in media):
        return None
    try:
        case = normalize_case(case)
    except Exception:
        return None
    return (last["task"], last["orientation"], last.get("seconds", 5)), case


def run_preload(key, case):
    """Load and compile the models for `key` before anyone asks (a normal warm-up pass, output discarded)."""
    log("preloading", key)
    root = SESSIONS / f"{time.strftime('%Y%m%d-%H%M%S')}-preload-{key[0]}-{key[1]}-{key[2]}s"
    try:
        p = new_pipeline(key, root)
    except Exception as e:
        log("preload skipped:", f"{type(e).__name__}: {e}"[:200])
        return
    # Until takeover_until, a job for these same settings stops the warm-up and loads the models itself: its loading
    # pass renders it. Started t s into a warm-up that takes L, it is done at about t + L (plus ~5 s to stop the
    # warm-up); waiting is done at L + W, W being a clip with the models loaded. So taking over wins while t < W - 5.
    # Before a warm clip of this setting has been measured: half the expected load time (5 s 16:9 t2va: L ~105-113 s,
    # W ~65 s).
    # load_eta and warm_s let the page say when a clip submitted now would be ready (both ways)
    warm, eta = warm_time(key), load_eta(key)
    session.update(pipeline=p, state="loading", since=time.time(), loading_key=key, preloading=True,
                   takeover_until=time.time() + (warm - 5 if warm else eta / 2), load_eta=eta, warm_s=warm)
    t0 = time.time()
    try:
        p.start(case)
    except BaseException as e:
        log("preload stopped:", f"{type(e).__name__}: {e}"[:200])
        session["preloading"] = False
        close_session("preload stopped")
        return
    with lock:                             # atomic with abort_preload: either it is aborted or it is ready
        aborted = not session.get("preloading")
        if not aborted:
            session.update(key=key, state="ready", since=time.time(), loading_key=None, preloading=False,
                           takeover_until=None, load_eta=None, warm_s=None)
    if aborted:                            # aborted just as it finished: its workers got SIGTERM
        close_session("preload stopped")
        return
    remember_load(key, round(time.time() - t0, 1))     # a warm-up pass is a cold start too (load + one clip)
    log(f"preload ready in {time.time() - t0:.0f} s")


def abort_preload(reason):
    """Stop a warm-up that is no longer wanted: SIGTERM the worker process groups (children of this server).
    Call it without holding `lock`."""
    with lock:
        if not session.get("preloading"):
            return False
        session["preloading"] = False
    log("aborting preload:", reason)
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            comm = (d / "comm").read_text().strip()
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me and comm != "ffmpeg":
            try:
                os.killpg(int(d.name), signal.SIGTERM)
            except OSError:
                pass
    return True


def worker_loop():
    while True:
        wake.wait(5)
        wake.clear()
        pending, preload["pending"] = preload["pending"], None
        if pending and session["pipeline"] is None and not any(j["status"] == "queued" for j in jobs):
            run_preload(*pending)
        while True:
            with lock:
                job = next((j for j in jobs if j["status"] == "queued"), None)
                if job is None:
                    break
                job["status"] = "starting"
                job["started"] = time.time()
            try:
                try:
                    run_job(job)
                except Exception as e:   # worker died -> session was closed; rebuild it and retry this job once
                    traceback.print_exc()
                    job["first_error"] = f"{type(e).__name__}: {e}"[:600]
                    job["status"] = "retrying after a worker error"
                    log("retrying job", job["id"], "after:", job["first_error"][:200])
                    run_job(job)
                job["status"] = "done"
            except BaseException as e:
                job["status"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"[:600]
                traceback.print_exc()
            job["finished"] = time.time()
            job["wall_s"] = round(job["finished"] - job["started"], 1)
            save_job(job)


# ------------------------------------------------------------------------------------------ HTTP
PAGE_FILE = Path(__file__).resolve().parent / "index.html"   # read per request: page edits need no restart


def parse_multipart(ctype, body):
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fn = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if fn:
            if data:
                files.setdefault(name, []).append((fn, data))
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def save_upload(job_dir, fn, data, allowed):
    ext = Path(fn).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"unsupported file type {ext} (allowed: {', '.join(sorted(allowed))})")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(fn).stem)[:40] or "file"
    p = job_dir / f"{safe}{ext}"
    i = 1
    while p.exists():
        p = job_dir / f"{safe}_{i}{ext}"
        i += 1
    p.write_bytes(data)
    return str(p)


# The form's file inputs ("zones") and where a job keeps their files
ZONES = ("first_frame", "last_frame", "ref_images", "ref_videos", "ref_audio")
REF_TYPE = {"ref_images": "image", "ref_videos": "video", "ref_audio": "audio"}


def zone_files(job, zone):
    """The input files of `job` for one form zone, in order (reference videos: the normalized copy the run used)."""
    if zone in REF_TYPE:
        return [r["path"] for r in job.get("references") or [] if r.get("type") == REF_TYPE[zone]]
    return [job[zone]] if job.get(zone) else []


def reuse_input(job_dir, path):
    """Put an earlier job's input file into this job's directory: a hardlink (no extra space; deleting either job
    later leaves the other's file alone), or a copy across filesystems."""
    src = Path(path)
    if not src.is_file():
        raise ValueError(f"{src.name} from the remixed clip is gone; attach it again")
    dst = job_dir / src.name
    i = 1
    while dst.exists():
        dst = job_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


REF_VIDEO_AREA = 672 * 384   # the draft's pixel area (the patch also prepares reference videos at this canvas)


def prepare_reference_video(path):
    """Normalize an uploaded reference video: downscale (never upscale) to ~672x384 area, 24 fps, <=15 s, keep
    audio, reject <2 s. The token-cost fix itself is in the patch (reference videos are resized to the draft
    canvas in stage1 and in the Qwen worker); this keeps uploads small and decoding cheap."""
    src = Path(path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height:format=duration", "-of", "json", str(src)],
                           capture_output=True, text=True, timeout=60)
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"{src.name}: no video stream")
    dur = float((info.get("format") or {}).get("duration") or 0)
    if dur and dur < 2.0:
        raise ValueError(f"{src.name}: reference videos must be at least 2 s (H3 limit)")
    out = src.with_name(src.stem + "_ref.mp4")
    vf = (f"scale='trunc(iw*min(1,sqrt({REF_VIDEO_AREA}/(iw*ih)))/2)*2':"
          f"'trunc(ih*min(1,sqrt({REF_VIDEO_AREA}/(iw*ih)))/2)*2':flags=lanczos,fps=24")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", "15", "-vf", vf, "-c:v", "libx264",
                        "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(out)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.is_file():
        raise ValueError(f"{src.name}: could not read this video ({r.stderr.strip()[:200]})")
    return str(out)


IMG = {".png", ".jpg", ".jpeg", ".webp"}
AUD = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VID = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
DURATIONS = (3, 4, 5, 6, 7, 8, 10, 12, 15)   # seconds; 5 = NVIDIA's frozen route, others = sol-h3-studio extension
MAX_UPLOAD = 200 * 1024 * 1024


class H(BaseHTTPRequestHandler):
    server_version = "SolH3Studio/1.0"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (extra or {}):
            self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send(200, PAGE_FILE.read_text(), "text/html; charset=utf-8")
        if path == "/health":
            return self.send(200, {"ok": True})
        if path == "/api/state":
            # the session, every queued or running job (spark-switchboard's busy probe reads these) and the newest
            # page of finished ones; older clips come from /api/jobs
            ready = ready_tasks()
            with lock:
                active = [j for j in jobs if j["status"] not in ("done", "failed")]
                finished = [j for j in jobs if j["status"] in ("done", "failed")]
                window = sorted(active + finished[-PAGE:], key=lambda j: j["created"], reverse=True)
                state = {"now": time.time(), "tasks_ready": ready,
                         "session": {k: session.get(k) for k in ("key", "state", "since", "loading_key", "preloading",
                                                                 "takeover_until", "load_eta", "warm_s")},
                         "jobs": [dict(j) for j in window],
                         "done": sum(j["status"] == "done" for j in finished), "more": len(finished) > PAGE}
            return self.send(200, state)
        if path == "/api/jobs":            # older clips: ?before=<created>&limit=<n>, newest first
            q = parse_qs(urlsplit(self.path).query)
            try:
                before = float(q.get("before", ["inf"])[0])
                limit = max(1, min(100, int(q.get("limit", [PAGE])[0])))
            except ValueError:
                return self.send(400, {"error": "before must be a number and limit an integer"})
            with lock:
                older = [j for j in jobs if j["created"] < before]
                page = [dict(j) for j in older[::-1][:limit]]
            return self.send(200, {"jobs": page, "more": len(older) > limit})
        m = re.fullmatch(r"/poster/([0-9a-f]{32})", path)
        if m:
            mp4 = clip_file(next((j for j in jobs if j["id"] == m.group(1)), None))
            poster = mp4.with_suffix(".jpg") if mp4 else None
            if poster and not poster.exists():
                make_poster(mp4)
            if not poster or not poster.exists():
                return self.send(404, {"error": "no poster"})
            return self.send(200, poster.read_bytes(), "image/jpeg", {"Cache-Control": "max-age=86400"})
        m = re.fullmatch(r"/api/inputs/([0-9a-f]{32})", path)
        if m:                      # a clip's input files, for Remix: what can be reused and what is gone
            job = next((j for j in jobs if j["id"] == m.group(1)), None)
            if job is None:
                return self.send(404, {"error": "no such clip"})
            out, missing = {}, []
            for zone in ZONES:
                for i, f in enumerate(zone_files(job, zone)):
                    if Path(f).is_file():
                        out.setdefault(zone, []).append({"name": Path(f).name, "index": i,
                                                         "url": f"/input/{job['id']}/{zone}/{i}"})
                    else:
                        missing.append(Path(f).name)
            return self.send(200, {"task": job["task"], "inputs": out, "missing": missing})
        m = re.fullmatch(r"/input/([0-9a-f]{32})/([a-z_]+)/(\d+)", path)
        if m:
            job = next((j for j in jobs if j["id"] == m.group(1)), None)
            files = zone_files(job, m.group(2)) if job and m.group(2) in ZONES else []
            f = Path(files[int(m.group(3))]) if int(m.group(3)) < len(files) else None
            if not f or not f.is_file():
                return self.send(404, {"error": "no such input"})
            return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        m = re.fullmatch(r"/video/([0-9a-f]{32})", path)
        if m:
            mp4 = clip_file(next((j for j in jobs if j["id"] == m.group(1)), None))
            if not mp4:
                return self.send(404, {"error": "no such video"})
            return self.send_file(mp4)
        self.send(404, {"error": "not found"})

    def send_file(self, p, ctype="video/mp4"):
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and (m := re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())) and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Disposition", f'inline; filename="{p.name}"')
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def same_origin(self):
        """Refuse cross-site POSTs (a page on another site submitting a form to this server).
        Browsers send Origin on such requests; same-page fetches send the page's own origin."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        allowed = {self.headers.get("Host"), self.headers.get("X-Forwarded-Host")}
        return urlsplit(origin).netloc in allowed

    def do_POST(self):
        if not self.same_origin():
            return self.send(403, {"error": "cross-origin request refused"})
        if self.path == "/api/unload":
            with lock:
                busy = any(j["status"] not in ("done", "failed", "queued") for j in jobs)
            if busy:
                return self.send(409, {"error": "a job is running"})
            if abort_preload("unload requested"):
                return self.send(200, {"ok": True})
            close_session("unload requested")
            return self.send(200, {"ok": True})
        if self.path != "/api/generate":
            return self.send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        if n > MAX_UPLOAD:
            return self.send(413, {"error": "upload too large (200 MB max)"})
        job_dir = None
        try:
            fields, files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(n))
            task = fields.get("task", "t2va")
            orientation = fields.get("orientation", "landscape")
            if task not in PATHS or orientation not in ORIENTATIONS:
                raise ValueError("bad task/orientation")
            if not PATHS[task].is_file():
                raise ValueError(f"no path manifest for {task} at {PATHS[task]} (run install/70-manifests.sh)")
            prompt = fields.get("prompt", "").strip()
            if not prompt:
                raise ValueError("prompt is empty")
            raw = (fields.get("seed") or "").strip()
            seconds = int(fields.get("seconds") or 5)
            if seconds not in DURATIONS:
                raise ValueError(f"duration must be one of {DURATIONS} seconds")
            seed = int(raw) if raw else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF   # empty -> random
            # Remix: "keep" names files of clip "remix_from" to reuse, as zone:index in the order of /api/inputs
            src = None
            if fields.get("remix_from"):
                src = next((j for j in jobs if j["id"] == fields["remix_from"]), None)
                if src is None:
                    raise ValueError("the clip to remix is no longer in the list; attach the files again")
            keep = {}
            for item in (fields.get("keep") or "").split(","):
                zone, _, index = item.partition(":")
                if src and zone in ZONES and index.isdigit():
                    keep.setdefault(zone, {})[int(index)] = None          # ordered, no duplicates
            jid = uuid.uuid4().hex
            job_dir = INPUTS / jid
            job_dir.mkdir(parents=True)

            def kept(zone):
                """The kept files of this zone, put into the new job's directory, in the order the form sent."""
                available = zone_files(src, zone) if src else []
                if any(i >= len(available) for i in keep.get(zone, {})):
                    raise ValueError("the remixed clip has fewer files than the form kept; attach them again")
                return [reuse_input(job_dir, available[i]) for i in keep.get(zone, {})]
            job = {"id": jid, "case_id": f"web_{time.strftime('%H%M%S')}_{jid[:6]}", "task": task,
                   "orientation": orientation, "seconds": seconds, "prompt": prompt, "seed": seed, "status": "queued",
                   "created": time.time()}
            if task == "fl2va":
                for key in ("first_frame", "last_frame"):
                    if files.get(key):
                        fn, data = files[key][0]
                        job[key] = save_upload(job_dir, fn, data, IMG)
                    elif kept_file := kept(key):         # one file per frame: a new upload replaces it
                        job[key] = kept_file[0]
                if not (job.get("first_frame") or job.get("last_frame")):
                    raise ValueError("fl2va needs a first frame, a last frame, or both")
            if task == "ref2va":
                # order = prompt numbering: <Picture N> images, then <Video N> videos (a video's own soundtrack is
                # added by Sol-H3 as an audio reference just before it), then standalone <Audio N> files
                # per kind: kept files first, then new uploads (the page shows them in that order); kept reference
                # videos are the normalized copies already
                refs = [{"type": "image", "path": p} for p in kept("ref_images")]
                refs += [{"type": "image", "path": save_upload(job_dir, fn, d, IMG)} for fn, d in files.get("ref_images", [])]
                refs += [{"type": "video", "path": p} for p in kept("ref_videos")]
                refs += [{"type": "video", "path": prepare_reference_video(save_upload(job_dir, fn, d, VID))}
                         for fn, d in files.get("ref_videos", [])]
                refs += [{"type": "audio", "path": p} for p in kept("ref_audio")]
                refs += [{"type": "audio", "path": save_upload(job_dir, fn, d, AUD)} for fn, d in files.get("ref_audio", [])]
                counts = {t: sum(r["type"] == t for r in refs) for t in ("image", "video", "audio")}
                if not (counts["image"] or counts["video"]):
                    raise ValueError("ref2va needs at least one reference image or video")
                if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3 or len(refs) > 12:
                    raise ValueError("H3 limits: up to 9 images, 3 videos, 3 audio files, 12 references in total")
                job["references"] = refs
            # validate now so errors come back immediately
            probe = {k: job[k] for k in ("case_id", "prompt", "seed", "task")}
            for k in ("first_frame", "last_frame", "references"):
                if k in job:
                    probe[k] = job[k]
            normalize_case(probe)
        except Exception as e:
            if job_dir is not None:
                shutil.rmtree(job_dir, ignore_errors=True)     # a refused request leaves nothing behind
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})
        with lock:
            # 'created' orders the history and is the paging cursor: set it here, strictly increasing, so a request
            # that took longer (a reference video being normalized) can't be appended after a later one
            job["created"] = max(time.time(), jobs[-1]["created"] + 0.001 if jobs else 0)
            jobs.append(job)
            # a job already queued for the warm-up's settings runs first, so it keeps the warm-up
            waited_for = any(j["status"] == "queued" and (j["task"], j["orientation"], j.get("seconds", 5))
                             == session.get("loading_key") for j in jobs)
        save_job(job)
        if session.get("preloading"):
            if session.get("loading_key") != (job["task"], job["orientation"], job["seconds"]):
                if not waited_for:
                    abort_preload("a job for another mode/format/length arrived")
            elif time.time() < (session.get("takeover_until") or 0):
                abort_preload("a job for the same settings arrived early in the warm-up; it loads the models itself")
        wake.set()
        self.send(200, {"id": jid})


def stop_with_launcher():
    """bin/sol-h3-studio starts the server in its own session and sets SOLH3_LAUNCHER_PID. If the launcher dies
    without stopping the server (SIGKILL), Linux sends this process SIGTERM (PR_SET_PDEATHSIG), so the pipeline is
    still closed."""
    launcher = os.environ.get("SOLH3_LAUNCHER_PID")
    if not launcher or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:     # 1 = PR_SET_PDEATHSIG
            return
    except (OSError, AttributeError):
        return
    if str(os.getppid()) != launcher:        # the launcher died before the prctl
        log("the launcher is gone; stopping")
        os.kill(os.getpid(), signal.SIGTERM)


def main():
    try:
        jobs.extend(open_db())
    except sqlite3.Error as error:
        fail(f"cannot open {DB_FILE}: {error}")
    for j in jobs:
        if j.get("status") not in ("done", "failed"):
            j["status"] = "failed"
            j["error"] = "server restarted"
            save_job(j)
    missing = [t for t in TASKS if t not in ready_tasks()]
    if missing:
        log("no path manifest for", ", ".join(missing), f"(looked for {PATHS_TEMPLATE})")
    if PRELOAD:
        preload["pending"] = preload_candidate()
        if preload["pending"]:
            log("will preload", preload["pending"][0])
            wake.set()
    threading.Thread(target=worker_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), H)
    srv.daemon_threads = True

    def stop(*_):
        log("shutting down")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)    # a closed terminal or SSH session: close the pipeline, as for SIGTERM
    stop_with_launcher()
    for name, value in IGNORED_ENV.items():
        log(f"ignoring {name}={value}: it is an infer.py default; the studio sets length and format per job")
    log(f"Sol-H3 Studio on http://{HOST}:{PORT}/ (package {PKG}, state {STATE_DIR})")
    try:
        srv.serve_forever()
    finally:
        close_session("server exit")
        log("bye")


if __name__ == "__main__":
    main()
