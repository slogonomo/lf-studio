#!/usr/bin/env python3
"""Sol-H3 Studio: a small web front end for NVIDIA Research's Sol-H3-Spark pipeline (unofficial).

- One job at a time (FIFO queue). Models stay loaded between jobs with the same task, orientation and length
  (one Pipeline session). A different combination closes the session and loads the models again.
- Tasks: t2va (text), fl2va (first and/or last frame), ref2va (reference images, videos, audio).
- Output: 1344x768 (landscape) or 768x1344 (portrait), 24 fps, 3-15 s, with audio.
- POST /api/cancel {"id": ...}: a queued clip leaves the queue at once; a running one is stopped (its pipeline
  workers get SIGTERM, so the next clip loads the models again). Either way it stays in the history with the status
  "cancelled" and its input files, for Remix and Delete. 409 for a clip that has finished (or is being saved).
  A studio before 0.3 shows cancelled clips as failed ("server restarted") and keeps them so.
- GET /still/<id>: the picture a queued or running clip's card shows behind its timer: a keyframe, a reference image,
  or the first frame of a reference video, as a JPEG at most 640 px wide (made on the first request, kept with the
  clip's input files; an image ffmpeg can't read goes out as it is). 404 for t2va.

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
  SOLH3_TRASH_DAYS           days a deleted clip stays in $SOL_H3_SPARK_RUNTIME_ROOT/trash before it is removed
                             for good (default 7)
  SOLH3_LAUNCHER_PID         set by bin/sol-h3-studio: the server stops (closing its pipeline) when that process dies.

SOL_H3_STAGE2_AUDIO (frozen or joint) and SOL_H3_PROMPT_TOKENS (plain or minimax) are patch settings (see
docs/PATCH.md), checked at start: another value stops the server with a message. So does a value that the applied patch
is too old to give (it lacks the setting): run install/10-sana.sh.
SOL_H3_SECONDS and SOL_H3_ORIENTATION are infer.py defaults. The studio passes length and format with every job, so
the server removes both from its environment at start; otherwise a 5 s job would render at SOL_H3_SECONDS.
"""
import contextlib
import email.parser
import email.policy
import errno
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

# the patch's settings (SOL_H3_STAGE2_AUDIO, patch 0.3+; SOL_H3_PROMPT_TOKENS, 0.4+): a bad value stops the studio here,
# not every clip. An older patch lacks one or both and does what the last value of each row asks; any other value stops
# the studio too, since that patch would ignore it.
import runtime.config  # noqa: E402
for setting, name, older in (("SOL_H3_STAGE2_AUDIO", "stage2_audio_mode", "joint"),
                             ("SOL_H3_PROMPT_TOKENS", "prompt_tokens_mode", "plain")):
    setting_check, value = getattr(runtime.config, name, None), os.environ.get(setting, "")
    if setting_check is None and value.strip().lower() not in ("", older):
        fail(f"{setting}={value} needs a newer patch than the one applied in {PKG}; run install/10-sana.sh")
    try:
        setting_check and setting_check()
    except ValueError as error:
        fail(str(error))

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
TRASH = RUNTIME / "trash"        # deleted clips, one directory per job, until SOLH3_TRASH_DAYS have passed
try:
    TRASH_DAYS = float(os.environ.get("SOLH3_TRASH_DAYS") or 7)
    assert math.isfinite(TRASH_DAYS) and TRASH_DAYS >= 0
except (ValueError, AssertionError):
    fail(f"SOLH3_TRASH_DAYS must be a number of days, 0 or more (got {os.environ.get('SOLH3_TRASH_DAYS')!r})")
for d in (INPUTS, SESSIONS, STATE_DIR) + ((PUBLISH,) if PUBLISH else ()):
    d.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()          # guards jobs / session
close_lock = threading.Lock()    # one pipeline close at a time (close_session)
wake = threading.Event()
stopping = threading.Event()     # set when the server shuts down: the worker starts nothing new
jobs = []                        # every job, oldest first (also in jobs.db)
PAGE = 24                        # clips per gallery page (/api/state carries the newest page, /api/jobs older ones)
FINISHED = ("done", "failed", "cancelled")       # a job's final statuses; any other is queued or running
session = {"pipeline": None, "key": None, "state": "idle", "since": time.time()}
work = {"job": None}             # the job whose pipeline call (start or generate) is running (pipeline_call)


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
            usable = [j for j in old if isinstance(j, dict) and re.fullmatch(r"[0-9a-f]{32}", str(j.get("id")))
                      and j.get("task") in TASKS and isinstance(j.get("prompt"), str)]
            if len(usable) < len(old):
                log(f"{OLD_STATE_FILE}: skipping {len(old) - len(usable)} entries without a valid id, task or prompt")
            old = usable
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
        j.pop("moving", None)
        j["created"] = as_time(j.get("created"))
        if last is not None and j["created"] <= last:      # equal or out-of-order times: 1 ms apart, in load order
            j["created"] = last + 0.001
            nudged.append(j)
        last = j["created"]
    for j in nudged:
        save_job(j)
    return loaded


def save_job(job, strict=False):
    """Write one job to jobs.db (its current state). strict: raise sqlite3.Error instead of only logging it."""
    try:
        with db_lock:
            data = dict(job)                           # a consistent copy (the worker updates jobs without lock)
            data.pop("moving", None)                   # only while this process moves its files
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)", (job["id"], job["created"], json.dumps(data)))
    except sqlite3.Error:
        traceback.print_exc()
        if strict:
            raise


def warm_up_wanted():
    """A queued job waits for the warm-up's settings, so the warm-up is kept for it. Call it holding `lock`."""
    return any(j["status"] == "queued" and (j["task"], j["orientation"], j.get("seconds", 5)) == session.get("loading_key")
               for j in jobs)


def find_job(jid):
    """The job with this id, unless it has been deleted."""
    return next((j for j in jobs if j["id"] == jid and not j.get("deleted")), None)


def in_sessions(case):
    """Whether `case` is a clip's own directory, <SESSIONS>/<session>/<case_id> (paths compared resolved: the
    pipeline records resolved paths, the runtime root may be reached through a symlink)."""
    try:
        return case.parent.parent.resolve() == SESSIONS.resolve()
    except OSError:
        return False


def under(path, root):
    """Whether `path` lies in `root` (both resolved)."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except OSError:
        return False


def trash_paths(job):
    """What deleting a job moves to the trash: the clip's own directory in its session (MP4, poster, intermediate
    files; for clips made before per-clip directories, the MP4 and its poster), the directories of runs that did not
    finish (case_dirs), its input directory, and its SOLH3_PUBLISH_DIR copy with poster. Nothing another clip still
    uses (an old history can list the same file twice). Call with `lock` held."""
    found = [Path(d) for d in job.get("case_dirs") or [] if in_sessions(Path(d))]
    if job.get("output"):
        out = Path(job["output"])
        case = out.parent.parent                   # <session>/<case_id>/stage2/<clip>.mp4
        found += [case] if case.name == job.get("case_id") and in_sessions(case) else [out, out.with_suffix(".jpg")]
    found.append(INPUTS / job["id"])
    if job.get("published"):
        found += [Path(job["published"]), Path(job["published"]).with_suffix(".jpg")]
    others = [str(Path(j[k]).with_suffix(x)) for j in jobs if j is not job and not j.get("deleted")
              for k in ("output", "published") if j.get(k) for x in (".mp4", ".jpg")]
    unique = []
    for f in found:
        shared = any(o == str(f) or o.startswith(str(f) + "/") for o in others)
        if f.exists() and f not in unique and not shared:
            unique.append(f)
    return unique


def safe_move(src, dst):
    """Move a file or directory without ever leaving a partial copy at `dst`: a rename on the same filesystem; across
    filesystems (the publish folder can be a network share) a copy under a temporary name next to `dst`, renamed into
    place, then the source removed. If a file's source can't be removed, the copy is removed again and the move
    fails; for a directory removed only in part the move counts as done and it is logged. `dst` must not exist."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} is already there")
    try:
        os.rename(src, dst)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    tmp = dst.with_name(f".{dst.name}.part")
    if tmp.exists():                                   # from a copy that was interrupted
        shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
    try:
        if src.is_dir() and not src.is_symlink():
            shutil.copytree(src, tmp, symlinks=True)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
        raise
    if src.is_dir() and not src.is_symlink():
        try:
            shutil.rmtree(src)
        except OSError as error:
            log(f"moved {src} to {dst} but could not remove all of the original: {error}")
        return
    try:
        src.unlink()
    except OSError:
        dst.unlink(missing_ok=True)                    # not moved after all: one copy, where it was
        raise


moves_done = threading.Condition()    # Delete/Undo file moves in progress (moves_running); shutdown waits for them
moves_running = 0


class MoveError(OSError):
    """A move to or from the trash failed; `left` maps originals to what is still in the trash after rolling back."""
    def __init__(self, error, left):
        super().__init__(str(error))
        self.left = left


def trash_targets(jid, paths):
    """Where each of `paths` goes in TRASH/<jid>/: {original: target}, names not taken yet."""
    dest, targets = TRASH / jid, {}
    for i, src in enumerate(paths):
        target = dest / f"{i}-{src.name}"
        n = 1
        while target.exists():                         # left over from an earlier failed attempt
            target = dest / f"{i}-{n}-{src.name}"
            n += 1
        targets[str(src)] = str(target)
    return targets


def move_all(targets):
    """Move every original to its target. If one fails, move back what was moved and raise MoveError with what
    could not go back."""
    moved = {}
    try:
        for src, target in targets.items():
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            safe_move(src, target)
            moved[src] = target
    except OSError as error:
        left = {}
        for src, target in moved.items():
            try:
                safe_move(target, src)
            except OSError:
                traceback.print_exc()
                left[src] = target
        raise MoveError(error, left) from error


def delete_clip(job, paths):
    """Move a clip's files to the trash and mark it deleted. What is about to move is saved first (job['trashing']),
    so a restart after a crash in the middle puts the files back (recover_moves). Returns the HTTP reply."""
    jid, targets = job["id"], trash_targets(job["id"], paths)
    with lock:
        job["trashing"] = targets
    try:
        save_job(job, strict=True)
    except sqlite3.Error as error:
        with lock:
            job.pop("trashing", None)
        return 500, {"error": f"could not record the delete ({error}); nothing was moved"}
    try:
        move_all(targets)
    except MoveError as error:
        with lock:
            job.pop("trashing", None)
            if error.left:
                job["trashing"] = error.left          # still in the trash: put back at the next start
        save_job(job)
        with contextlib.suppress(OSError):
            (TRASH / jid).rmdir()
        return 500, {"error": f"could not move the clip to the trash: {error}"}
    with lock:
        job.pop("trashing", None)
        job.update(deleted=time.time(), trashed=targets)
    try:
        save_job(job, strict=True)
    except sqlite3.Error as error:                     # not recorded: put it all back
        left, _, _ = restore_files(targets, set(targets))
        with lock:
            job.pop("deleted", None)
            job.pop("trashed", None)
        return 500, {"error": f"could not record the delete ({error}); the clip is back"
                     + (f", except {len(left)} file(s) in {TRASH / jid}" if left else "")}
    log("deleted", job["case_id"], "to", TRASH / jid)
    return 200, {"ok": True}


def needed_files(job, trashed):
    """The trashed originals a clip can't do without: everything in the runtime root, and its MP4, which is the
    publish copy when the clip has no other (a clip from before per-clip directories, or one whose session
    directory was pruned). The other publish copies are best effort."""
    own = job.get("output")
    needed = {k for k in trashed if under(k, RUNTIME)} | ({own} if own else set())
    has_own = own and (own in trashed or any(own.startswith(k + "/") for k in trashed) or Path(own).is_file())
    if not has_own and job.get("published"):
        needed |= {job["published"]}
    return needed


def undelete_clip(job):
    """Move a deleted clip's files back (restore_files) and unmark it. Returns the HTTP reply."""
    trashed = dict(job.get("trashed") or {})
    left, copy_lost, error = restore_files(trashed, needed_files(job, trashed))
    with lock:
        if error:
            job["trashed"] = left                      # what is still there, for another try
        else:
            job.pop("deleted", None)
            job.pop("trashed", None)
            if copy_lost:
                job.pop("published", None)
    if not error:
        with contextlib.suppress(OSError):
            (TRASH / job["id"]).rmdir()                # empty unless a publish copy stayed
    try:
        save_job(job, strict=True)
    except sqlite3.Error as db_error:
        return 500, {"error": f"the files were moved back but the history could not be updated ({db_error}); "
                              "it may show as deleted again after a restart"}
    if error:
        return 500, {"error": f"could not restore the clip: {error}"}
    log("restored", job["case_id"])
    return 200, {"ok": True, "job": dict(job)}


def recover_moves():
    """At start: put back the files of a delete that did not finish (job['trashing'], e.g. the server was killed)."""
    for job in [j for j in jobs if j.get("trashing")]:
        left, _, error = restore_files(job["trashing"], set(job["trashing"]))
        if error:
            log("could not put back the files of an unfinished delete:", error)
            job["trashing"] = left
        else:
            job.pop("trashing", None)
            log("put back the files of an unfinished delete of", job.get("case_id"))
        save_job(job)


def trashed_path(job, path):
    """Where `path` (an input of `job`) is now: under TRASH/<id>/ while the job is deleted, else where it was."""
    for src, target in (job.get("trashed") or {}).items():
        if path == src or path.startswith(src + "/"):
            return target + path[len(src):]
    return path


def restore_files(trashed, needed):
    """Move trashed files back. `needed`: the originals the clip can't do without (its directory, inputs, and an old
    clip's only MP4, even if that is in the publish folder); any other is a SOLH3_PUBLISH_DIR copy, best effort.
    Folders are created again only inside the runtime root, never on a share that is not mounted. Returns (what is
    still in the trash, whether a publish copy stayed there, the error that stopped a needed file or None). Runs
    without `lock`."""
    left, copy_lost = dict(trashed), False
    for src, target in trashed.items():
        if not Path(target).exists():                  # nothing to bring back
            del left[src]
            continue
        try:
            if Path(src).exists():
                raise FileExistsError(f"{src} is already there")
            if under(src, RUNTIME):
                Path(src).parent.mkdir(parents=True, exist_ok=True)
            elif not Path(src).parent.is_dir():
                raise FileNotFoundError(f"{Path(src).parent} is not there (an unmounted share?)")
            safe_move(target, src)
            del left[src]
        except OSError as error:
            if src in needed:
                return left, copy_lost, error
            log("could not put the publish copy back, it stays in the trash:", error)
            del left[src]                              # given up on; purge_trash sweeps it with the trash folder
            copy_lost = True
    return left, copy_lost, None


def purge_trash():
    """Remove clips deleted more than TRASH_DAYS ago, their files and their history entry, and anything else in
    TRASH older than that (left over from a failed delete or restore)."""
    cutoff = time.time() - TRASH_DAYS * 86400
    with lock:
        old = [j for j in jobs if j.get("deleted") and j["deleted"] < cutoff and not j.get("moving")]
        for j in old:
            jobs.remove(j)
        keep = {j["id"] for j in jobs if j.get("deleted") or j.get("moving") or j.get("trashing")}
    for j in old:
        shutil.rmtree(TRASH / j["id"], ignore_errors=True)
        try:
            with db_lock:
                db.execute("DELETE FROM jobs WHERE id = ?", (j["id"],))
        except sqlite3.Error:
            traceback.print_exc()
    swept = 0
    for d in TRASH.iterdir() if TRASH.is_dir() else []:
        with contextlib.suppress(OSError):
            if d.name not in keep and d.stat().st_mtime < cutoff:
                shutil.rmtree(d) if d.is_dir() else d.unlink()
                swept += 1
    if old or swept:
        log(f"emptied the trash: {len(old)} clip(s) deleted more than {TRASH_DAYS:g} days ago"
            + (f", {swept} leftover(s)" if swept else ""))


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


def session_root(key, kind=""):
    """A new session directory: start time and settings, -2, -3, ... after it when another session started in the
    same second (a clip cancelled right away, then the next one; the pipeline refuses an existing directory)."""
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}{key[0]}-{key[1]}-{key[2]}s"
    root, n = SESSIONS / base, 2
    while root.exists():
        root, n = SESSIONS / f"{base}-{n}", n + 1
    return root


def new_pipeline(key, root):
    task, orientation, seconds = key
    return Pipeline(load_paths(PATHS[task], task=task), root, task=task,
                    orientation=orientation, seconds=None if seconds == 5 else seconds)   # 5 s = frozen route


def close_session(reason, p=None):
    """Close the session's pipeline, or `p` (a pipeline of this thread's that may no longer be the session's).
    One close at a time: a second caller waits for the first, then finds nothing left to close. The session is reset
    only if it still holds the closed pipeline (or none), never a newer one. Call it without holding `lock`."""
    with close_lock:
        with lock:
            target = p or session["pipeline"]
            mine = target is not None and target is session["pipeline"]
            if mine:
                session["state"] = "unloading"
                session["since"] = time.time()
            key = session["key"] if mine else None
        if target is not None:
            log("closing session", key or "", "-", reason)
            try:
                target.close()
            except Exception:
                traceback.print_exc()
        with lock:
            if session["pipeline"] is None or session["pipeline"] is target:
                session.update(pipeline=None, key=None, state="idle", since=time.time(), loading_key=None,
                               preloading=False, takeover_until=None, load_eta=None, warm_s=None)


class Cancelled(Exception):
    """The job was cancelled (POST /api/cancel) after it had started."""


def pipeline_call(job, call, failure):
    """Run `call`, the pipeline's start() or generate() for `job`, so that Cancel can stop it: while it runs,
    work["job"] is the job, and stop_job SIGTERMs the workers of a cancelled one. If it fails, or Cancel came first,
    the session is closed (its workers may have been signalled) and it raises; Cancelled when Cancel came first, even
    if the clip finished just then: Cancel has answered that it is cancelled. Otherwise the result is the job's, and
    from here on Cancel answers 409 (job["saving"]). `failure`: the reason logged when it failed by itself."""
    with lock:
        work["job"] = job
    try:
        row = call()
        with lock:                         # atomic with Cancel: either it is cancelled or its clip is kept
            cancelled = job.get("cancel_requested")
            if not cancelled:
                job["saving"] = True
                work["job"] = None
        if cancelled:
            raise Cancelled("cancelled as it finished")
        return row
    except BaseException:
        close_session("clip cancelled" if job.get("cancel_requested") else failure)   # stop_job still signals
        raise                                                                        # workers started meanwhile
    finally:
        with lock:
            work["job"] = None


def run_job(job):
    key = (job["task"], job["orientation"], job.get("seconds", 5))
    case = {"case_id": job["case_id"], "prompt": job["prompt"], "seed": job["seed"], "task": job["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if job.get(k):
            case[k] = job[k]
    case = normalize_case(case)
    if session["key"] != key:
        close_session(f"switching to {key}")
        if job.get("cancel_requested"):    # cancelled while the old models were closed: nothing of its own ran
            raise Cancelled("cancelled before it started")
        job["status"] = "loading models + generating"
        job["phase_since"] = time.time()
        job["load_eta"] = load_eta(key)
        session.update(state="loading", since=time.time(), loading_key=key)
        try:
            root = session_root(key)
            with lock:
                if stopping.is_set():
                    raise RuntimeError("the studio is shutting down")
            p = new_pipeline(key, root)
        except BaseException:
            close_session("could not start the pipeline")      # no pipeline: back to idle, not "loading"
            raise
        session["pipeline"] = p
        job.setdefault("case_dirs", []).append(str(Path(p.root) / job["case_id"]))   # for Delete, even if it fails
        t0 = time.time()
        # the loading pass renders this job (no throwaway warm-up)
        row = pipeline_call(job, lambda: p.start(case, deliver=True), "model load failed")
        job["cold_start_s"] = round(time.time() - t0, 1)   # load + this clip
        job["includes_model_load"] = True
        remember_load(key, job["cold_start_s"])
        session.update(key=key, state="ready", since=time.time(), loading_key=None)
    else:
        if job.get("cancel_requested"):    # nothing ran: the models stay loaded
            raise Cancelled("cancelled before it started")
        job["status"] = "generating"
        job["phase_since"] = time.time()
        job.setdefault("case_dirs", []).append(str(Path(session["pipeline"].root) / job["case_id"]))
        session["state"] = "busy"
        t0 = time.time()
        row = pipeline_call(job, lambda: session["pipeline"].generate(case), "request failed (worker died)")
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
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "1", "-i", str(mp4), "-frames:v", "1",
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


preload = {"pending": None}      # (key, case) to warm up when nothing is queued: at start, or "Load selected now"


def warmup_case(job):
    """A finished clip as a warm-up case: its prompt and inputs, seed 999. None when its manifest or inputs are gone."""
    if not PATHS.get(job.get("task"), Path("/nonexistent")).is_file():
        return None
    case = {"case_id": "preload", "prompt": job["prompt"], "seed": 999, "task": job["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if job.get(k):
            case[k] = job[k]
    media = [case.get("first_frame"), case.get("last_frame")] + [r.get("path") for r in case.get("references", [])]
    if any(m and not Path(m).is_file() for m in media):
        return None
    try:
        return normalize_case(case)
    except Exception:
        return None


def preload_candidate():
    """The last finished clip's settings and warm-up case, or None (see warmup_case)."""
    last = next((j for j in reversed(jobs) if j.get("status") == "done" and not j.get("deleted")), None)
    case = warmup_case(last) if last else None
    return ((last["task"], last["orientation"], last.get("seconds", 5)), case) if case else None


WARMUP_INPUTS = INPUTS / "_warmup"   # the running warm-up's own links to its input files


def own_inputs(case):
    """The warm-up case with hardlinks of its input files in WARMUP_INPUTS (like Remix), so deleting the clip they
    came from can't take them away while the models load. None if they are gone already."""
    shutil.rmtree(WARMUP_INPUTS, ignore_errors=True)     # the previous warm-up's, no longer read
    out = dict(case)
    try:
        WARMUP_INPUTS.mkdir(parents=True)
        for k in ("first_frame", "last_frame"):
            if case.get(k):
                out[k] = reuse_input(WARMUP_INPUTS, case[k])
        if case.get("references"):
            out["references"] = [dict(r, path=reuse_input(WARMUP_INPUTS, r["path"])) for r in case["references"]]
    except (OSError, ValueError) as error:
        log("warm-up skipped, its input files are gone:", error)
        return None
    return out


WARMUP_PROMPT = ("integrated_multimodal_description: A quiet lake at sunrise, mist over the water, one continuous wide "
                 "shot.\noverall_soundscape: Soft wind and water.\nnon_diegetic_music: None.")


def mode_warmup_case(task):
    """A warm-up case for a mode ("Load selected now"): the newest of its last few finished clips whose inputs still
    exist; for t2va without one, a built-in prompt. fl2va and ref2va need real inputs, so without such a clip: None."""
    recent = [j for j in reversed(jobs) if j.get("task") == task and j.get("status") == "done"
              and not j.get("deleted") and not j.get("moving")]
    for j in recent[:5]:
        case = warmup_case(j)
        if case:
            return case
    if task == "t2va" and PATHS["t2va"].is_file():
        return normalize_case({"case_id": "preload", "prompt": WARMUP_PROMPT, "seed": 999, "task": "t2va"})
    return None


def run_preload(key, case):
    """Load and compile the models for `key` before anyone asks (a normal warm-up pass, output discarded)."""
    log("preloading", key)
    root = session_root(key, "preload-")
    try:
        p = new_pipeline(key, root)
    except Exception as e:
        log("preload skipped:", f"{type(e).__name__}: {e}"[:200])
        with lock:
            session.update(preloading=False, loading_key=None, state="idle" if session["pipeline"] is None else
                           session["state"])
        return
    # Until takeover_until, a job for these same settings stops the warm-up and loads the models itself: its loading
    # pass renders it. Started t s into a warm-up that takes L, it is done at about t + L (plus ~5 s to stop the
    # warm-up); waiting is done at L + W, W being a clip with the models loaded. So taking over wins while t < W - 5.
    # Before a warm clip of this setting has been measured: half the expected load time (5 s 16:9 t2va: L ~105-113 s,
    # W ~65 s).
    # load_eta and warm_s let the page say when a clip submitted now would be ready (both ways)
    warm, eta = warm_time(key), load_eta(key)
    with lock:                             # the worker set preloading when it took the request; a Generate or an
        stopped = not session.get("preloading") or stopping.is_set()    # Unload since then has cleared it
        if not stopped:
            session.update(pipeline=p, state="loading", since=time.time(), loading_key=key,
                           takeover_until=time.time() + (warm - 5 if warm else eta / 2), load_eta=eta, warm_s=warm)
    if stopped:
        close_session("warm-up no longer wanted", p)
        return
    t0 = time.time()
    try:
        p.start(case)
    except BaseException as e:
        log("preload stopped:", f"{type(e).__name__}: {e}"[:200])
        session["preloading"] = False
        close_session("preload stopped", p)
        return
    with lock:                             # atomic with abort_preload: either it is aborted or it is ready
        aborted = not session.get("preloading")
        if not aborted:
            session.update(key=key, state="ready", since=time.time(), loading_key=None, preloading=False,
                           takeover_until=None, load_eta=None, warm_s=None)
    if aborted:                            # aborted just as it finished: its workers got SIGTERM
        close_session("preload stopped", p)
        return
    remember_load(key, round(time.time() - t0, 1))     # a warm-up pass is a cold start too (load + one clip)
    log(f"preload ready in {time.time() - t0:.0f} s")


def signal_workers(skip=()):
    """SIGTERM the pipeline's worker process groups: children of this server that lead their own process group (the
    runtime starts each worker with start_new_session; ffmpeg and ffprobe run in the server's group). Groups in
    `skip` are left alone. Returns the groups signalled."""
    me, sent = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            ppid, pgrp = map(int, (d / "stat").read_text().rsplit(")", 1)[1].split()[1:3])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me and pgrp == int(d.name):
            try:
                os.killpg(pgrp, signal.SIGTERM)
                sent.append(pgrp)
            except OSError:
                pass
    return sent


def abort_preload(reason):
    """Stop a warm-up that is no longer wanted: SIGTERM the worker process groups (children of this server).
    Call it without holding `lock`."""
    with lock:
        if not session.get("preloading"):
            return False
        session["preloading"] = False
    log("aborting preload:", reason)
    signal_workers()
    return True


def stop_job(job):
    """Stop a running job that has been cancelled (its own thread, started by /api/cancel): while its pipeline call
    runs (work["job"]), SIGTERM the worker process groups as abort_preload does, also those started after the first
    signal (the loading pass starts Stage2 once Stage1 is loaded). Each group once: qwen_python.sh stops its
    container in a TERM trap, which a second signal would cut short. The signal and the check are under `lock`, so
    they never reach the next job's workers. Ends when the job has finished."""
    sent = set()
    while not stopping.is_set():
        with lock:
            if job["status"] in FINISHED:
                return
            if work["job"] is job:
                new = signal_workers(sent)
                if new:
                    log(f"cancel: sent SIGTERM to the worker process group(s) {new} of", job["case_id"])
                sent.update(new)
        time.sleep(0.2)


def worker_loop():
    purged = time.time()                           # main() purged at start
    while not stopping.is_set():
        wake.wait(5)
        wake.clear()
        if stopping.is_set():
            break
        if time.time() - purged > 86400:
            purged = time.time()
            purge_trash()
        # a requested warm-up (at start, or "Load selected now"): session["switching_to"] shows it until it starts.
        # A clip submitted, an Unload or a newer request meanwhile cancels or replaces it (they clear or change
        # switching_to), also while the old models are being closed for it, which takes ~40 s.
        with lock:
            pending, preload["pending"] = preload["pending"], None
        if pending:
            if session["pipeline"] is not None and session["key"] != pending[0]:
                close_session(f"loading {pending[0]} as asked")
            with lock:
                go = (session.get("switching_to") == pending[0] and not preload["pending"] and not stopping.is_set()
                      and session["pipeline"] is None and not any(j["status"] == "queued" for j in jobs))
                if session.get("switching_to") == pending[0] and not preload["pending"]:
                    session["switching_to"] = None
                # the case is chosen again now (the clip it came from may have been deleted during the close), and
                # the warm-up gets its own links to the input files
                case = own_inputs(mode_warmup_case(pending[0][0]) or {}) if go else None
                if go and case and case.get("prompt"):
                    session.update(preloading=True, loading_key=pending[0], state="loading", since=time.time())
                else:
                    go = False
            if go:
                run_preload(pending[0], case)
                shutil.rmtree(WARMUP_INPUTS, ignore_errors=True)   # read by the loading pass only
        while True:
            with lock:
                job = None if stopping.is_set() else next((j for j in jobs if j["status"] == "queued"), None)
                if job is None:
                    break
                job["status"] = "starting"
                job["started"] = time.time()
            try:
                try:
                    run_job(job)
                except Exception as e:   # worker died -> session was closed; rebuild it and retry this job once
                    if job.get("cancel_requested"):            # cancelled: no retry (run_job closed the session)
                        raise
                    traceback.print_exc()
                    job["first_error"] = f"{type(e).__name__}: {e}"[:600]
                    job["status"] = "retrying after a worker error"
                    log("retrying job", job["id"], "after:", job["first_error"][:200])
                    run_job(job)
                job["status"] = "done"
            except BaseException as e:
                if job.get("cancel_requested"):
                    job["status"] = "cancelled"
                    log("cancelled job", job["id"], f"({type(e).__name__}: {e})"[:200])
                else:
                    job["status"] = "failed"
                    job["error"] = f"{type(e).__name__}: {e}"[:600]
                    traceback.print_exc()
            for k in ("saving", "cancel_requested"):     # only while it runs; the status says the rest
                job.pop(k, None)
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


STILL = "@still.jpg"     # a clip's card picture, in the clip's input directory (uploads never have an @)
STILL_PART = ".still-*.part"   # one being made, in INPUTS itself; one left by a server stopped part-way goes at start
PICTURE_CACHE = "private, max-age=86400"   # /still and /poster: they never change; the user's own pictures, so only
                                           # the browser keeps them, not a shared cache in front of the studio


def card_picture(job):
    """The picture a queued or running clip's card shows behind its timer (#30): fl2va its first frame, else its last;
    ref2va its first reference image, else the first frame of its first reference video. It is a small JPEG, at most
    640 px wide (the card shows it blurred, and an upload can be a photo of many MB), made on the first request and
    kept in the clip's input directory (INPUTS/<id>), so Delete, Undo and the trash purge move it with the inputs.
    An image goes out as it is if ffmpeg can't make the JPEG, or if the clip has no input directory of its own (an
    old history). None for t2va, when the file is gone, or when a video's frame can't be read. While ffmpeg makes the
    JPEG, it is a temporary file in INPUTS itself, not in the clip's directory, which a Delete can move meanwhile (the
    JPEG is then not kept, and the moved image is not served). Never writes elsewhere."""
    src, video = None, False
    if job.get("task") == "fl2va":
        src = job.get("first_frame") or job.get("last_frame")
    elif job.get("task") == "ref2va":
        refs = job.get("references") or []
        ref = next((r for kind in ("image", "video") for r in refs if r.get("type") == kind), None)
        src, video = (ref["path"], ref["type"] == "video") if ref else (None, False)
    if not src or not Path(src).is_file():
        return None
    original = None if video else Path(src)
    still = INPUTS / job["id"] / STILL
    if still.is_file():
        return still
    if not still.parent.is_dir():                      # no input directory of its own (an old history): no JPEG
        return original
    tmp = INPUTS / STILL_PART.replace("*", uuid.uuid4().hex)    # two requests at once: each its own, the last wins
    try:            # -nostdin: runs at once (a page asks for every card's picture) could leave a terminal without echo
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-frames:v", "1", "-vf",
                        "scale='min(640,iw)':-2", "-q:v", "4", "-f", "mjpeg", str(tmp)],
                       capture_output=True, timeout=60, check=True)
        os.replace(tmp, still)                          # FileNotFoundError if Delete has moved the directory
    except (OSError, subprocess.SubprocessError):
        return original if original and original.is_file() else None
    finally:
        tmp.unlink(missing_ok=True)
    return still


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
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-t", "15", "-vf", vf,
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                        str(out)],
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
                active = [j for j in jobs if j["status"] not in FINISHED]
                finished = [j for j in jobs if j["status"] in FINISHED and not j.get("deleted")]
                window = sorted(active + finished[-PAGE:], key=lambda j: j["created"], reverse=True)
                warmable = [t for t in ready if mode_warmup_case(t)]     # modes "Load selected now" can warm up
                state = {"now": time.time(), "tasks_ready": ready, "trash_days": TRASH_DAYS, "warmable": warmable,
                         "session": {k: session.get(k) for k in ("key", "state", "since", "loading_key", "preloading",
                                                                 "takeover_until", "load_eta", "warm_s",
                                                                 "switching_to")},
                         "jobs": [dict(j) for j in window],
                         "done": sum(j["status"] == "done" for j in finished), "more": len(finished) > PAGE,
                         "deleted": {j["id"]: j["created"] for j in jobs if j.get("deleted")}}   # in the trash
            return self.send(200, state)
        if path == "/api/jobs":            # older clips: ?before=<created>&limit=<n>, newest first
            q = parse_qs(urlsplit(self.path).query)
            try:
                before = float(q.get("before", ["inf"])[0])
                limit = max(1, min(100, int(q.get("limit", [PAGE])[0])))
            except ValueError:
                return self.send(400, {"error": "before must be a number and limit an integer"})
            with lock:
                older = [j for j in jobs if j["created"] < before and not j.get("deleted")]
                page = [dict(j) for j in older[::-1][:limit]]
            return self.send(200, {"jobs": page, "more": len(older) > limit})
        m = re.fullmatch(r"/poster/([0-9a-f]{32})", path)
        if m:
            mp4 = clip_file(find_job(m.group(1)))
            poster = mp4.with_suffix(".jpg") if mp4 else None
            if poster and not poster.exists():
                make_poster(mp4)
            if not poster or not poster.exists():
                return self.send(404, {"error": "no poster"})
            return self.send(200, poster.read_bytes(), "image/jpeg", {"Cache-Control": PICTURE_CACHE})
        m = re.fullmatch(r"/api/inputs/([0-9a-f]{32})", path)
        if m:                      # a clip's input files, for Remix: what can be reused and what is gone
            job = find_job(m.group(1))
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
            job = find_job(m.group(1))
            files = zone_files(job, m.group(2)) if job and m.group(2) in ZONES else []
            f = Path(files[int(m.group(3))]) if int(m.group(3)) < len(files) else None
            if not f or not f.is_file():
                return self.send(404, {"error": "no such input"})
            return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        m = re.fullmatch(r"/still/([0-9a-f]{32})", path)
        if m:                      # the picture behind a queued or running clip's timer (card_picture)
            with lock:
                job = find_job(m.group(1))
                job = None if job is None or job.get("moving") else dict(job)    # not while Delete moves its files
            f = card_picture(job) if job else None
            if not f:
                return self.send(404, {"error": "no picture"})
            return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream",
                                  {"Cache-Control": PICTURE_CACHE})
        m = re.fullmatch(r"/video/([0-9a-f]{32})", path)
        if m:
            mp4 = clip_file(find_job(m.group(1)))
            if not mp4:
                return self.send(404, {"error": "no such video"})
            return self.send_file(mp4)
        self.send(404, {"error": "not found"})

    def send_file(self, p, ctype="video/mp4", extra=None):
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
        for k, v in (extra or {}).items():
            self.send_header(k, v)
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
                busy = any(j["status"] not in FINISHED + ("queued",) for j in jobs)
                if not busy:                       # a requested load ("Load selected now") is cancelled too
                    preload["pending"] = None
                    session["switching_to"] = None
            if busy:
                return self.send(409, {"error": "a job is running"})
            if abort_preload("unload requested"):
                return self.send(200, {"ok": True})
            close_session("unload requested")
            return self.send(200, {"ok": True})
        if self.path == "/api/load":       # "Load selected now": warm up these settings before the prompt is ready
            try:
                want = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                key = (want["task"], want["orientation"], int(want["seconds"]))
                if key[0] not in PATHS or key[1] not in ORIENTATIONS or key[2] not in DURATIONS:
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                return self.send(400, {"error": "expected JSON {task, orientation, seconds}"})
            if not PATHS[key[0]].is_file():
                return self.send(400, {"error": f"no path manifest for {key[0]} (run install/70-manifests.sh)"})
            with lock:
                busy = any(j["status"] not in FINISHED for j in jobs)
                case = mode_warmup_case(key[0])
                switching = session.get("switching_to")
                already = (switching == key or not switching and (
                    key == session.get("key") and session.get("state") in ("ready", "busy")
                    or key == session.get("loading_key") and session.get("preloading")))
                if not busy and not already and case:
                    preload["pending"] = (key, case)      # replaces an earlier request that hasn't started
                    session["switching_to"] = key
            if busy:
                return self.send(409, {"error": "a clip is queued or running; the models change for it anyway"})
            if already:
                return self.send(200, {"ok": True, "already": True})
            if case is None:
                return self.send(409, {"error": f"no earlier {key[0]} clip with its files to warm up with"})
            abort_preload(f"loading {key} as asked")            # a warm-up for other settings
            wake.set()
            log("load requested:", key)
            return self.send(200, {"ok": True})
        if self.path == "/api/cancel":
            try:
                jid = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}").get("id")
            except (ValueError, AttributeError):
                return self.send(400, {"error": "expected JSON {\"id\": ...}"})
            # Under the lock, atomic with the worker taking a queued job and with pipeline_call accepting a finished
            # clip: a queued job is cancelled here and now; a running one is marked, and stop_job stops its workers.
            # Its input files stay (Remix), like a failed clip's.
            queued = stop = drop = False
            with lock:
                job = find_job(jid)
                if job is None:
                    reply = 404, {"error": "no such clip"}
                elif job["status"] in FINISHED:
                    reply = 409, {"error": {"done": "the clip has already finished",
                                            "failed": "the clip has already failed",
                                            "cancelled": "the clip is already cancelled"}[job["status"]]}
                elif job.get("saving"):
                    reply = 409, {"error": "the clip has just finished and is being saved"}
                else:
                    queued = job["status"] == "queued"
                    if queued:
                        job.update(status="cancelled", finished=time.time())
                        # the warm-up may have been kept only for this clip (/api/generate): if another clip waits
                        # now and none for the warm-up's settings, stop it as a clip for other settings would have
                        drop = (session.get("preloading") and not warm_up_wanted()
                                and any(j["status"] == "queued" for j in jobs))
                    else:
                        stop = not job.get("cancel_requested")      # a second Cancel: already being stopped
                        job["cancel_requested"] = True
                    reply = 200, {"ok": True, "job": dict(job)}
            if reply[0] == 200:
                save_job(job)        # a running one too: if the server dies before it stops, it stays cancelled
                if queued:
                    log("cancelled queued job", job["id"])
                    if drop:
                        abort_preload("the clip it was kept for was cancelled")
                        wake.set()
                elif stop:
                    log("cancelling job", job["id"], "while", reply[1]["job"]["status"])
                    threading.Thread(target=stop_job, args=(job,), daemon=True).start()
            return self.send(*reply)
        if self.path in ("/api/delete", "/api/undelete"):
            try:
                jid = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}").get("id")
            except (ValueError, AttributeError):
                return self.send(400, {"error": "expected JSON {\"id\": ...}"})
            # Under the lock: check and mark the clip as moving (Remix, warm-ups, the trash purge and another Delete
            # or Undo leave it alone meanwhile). The files move without the lock: a copy to or from a share can take
            # a while. Shutdown waits for moves in progress.
            global moves_running
            with lock:
                job = next((j for j in jobs if j["id"] == jid), None)
                if job is None:
                    reply = 404, {"error": "no such clip"}
                elif stopping.is_set():
                    reply = 503, {"error": "the studio is shutting down"}
                elif job.get("moving"):
                    reply = 409, {"error": "the clip is being moved; try again in a moment"}
                elif self.path == "/api/undelete":
                    reply = None if job.get("deleted") else (200, {"ok": True, "job": dict(job)})
                elif job["status"] not in FINISHED:
                    reply = 409, {"error": "the clip is still queued or running"}
                elif job.get("deleted"):
                    reply = 200, {"ok": True}
                else:
                    reply, paths = None, trash_paths(job)
                if reply is None:
                    job["moving"] = True
                    with moves_done:
                        moves_running += 1
            if reply is None:
                try:
                    reply = delete_clip(job, paths) if self.path == "/api/delete" else undelete_clip(job)
                except Exception as error:                     # never leave the clip marked as moving
                    traceback.print_exc()
                    reply = 500, {"error": f"{type(error).__name__}: {error}"}
                finally:
                    with lock:
                        job.pop("moving", None)
                    with moves_done:
                        moves_running -= 1
                        moves_done.notify_all()
            return self.send(*reply)
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
                src = next((j for j in jobs if j["id"] == fields["remix_from"]), None)   # deleted too: its files
                                                                                        # are in the trash
                if src is None:
                    raise ValueError("the clip to remix is no longer in the list; attach the files again")
                if src.get("moving"):
                    raise ValueError("the clip to remix is being deleted or restored right now; try again")
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
                available = [trashed_path(src, f) for f in zone_files(src, zone)] if src else []
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
            preload["pending"] = None              # a clip decides what is loaded next: a requested warm-up that
            session["switching_to"] = None         # hasn't started is dropped (the clip loads its own settings)
            # a job already queued for the warm-up's settings runs first, so it keeps the warm-up
            waited_for = warm_up_wanted()
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
        if j.get("status") not in FINISHED:            # it was queued or running when the server stopped
            if j.pop("cancel_requested", None):        # Cancel had answered: it stays cancelled
                j["status"] = "cancelled"
            else:
                j["status"] = "failed"
                j["error"] = "server restarted"
            save_job(j)
    recover_moves()
    purge_trash()
    for part in INPUTS.glob(STILL_PART):              # a card picture being made when the server stopped
        with contextlib.suppress(OSError):
            part.unlink()
    missing = [t for t in TASKS if t not in ready_tasks()]
    if missing:
        log("no path manifest for", ", ".join(missing), f"(looked for {PATHS_TEMPLATE})")
    if PRELOAD:
        preload["pending"] = preload_candidate()
        if preload["pending"]:
            session["switching_to"] = preload["pending"][0]
            log("will preload", preload["pending"][0])
            wake.set()
    threading.Thread(target=worker_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), H)
    srv.daemon_threads = True

    def stop(*_):
        log("shutting down")
        stopping.set()
        wake.set()
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
        stopping.set()
        with moves_done:                               # a Delete or Undo moving files finishes first (up to 60 s)
            if not moves_done.wait_for(lambda: moves_running == 0, timeout=60):
                log("a Delete or Undo is still moving files; the next start puts back an unfinished delete")
        close_session("server exit")
        log("bye")


if __name__ == "__main__":
    main()
