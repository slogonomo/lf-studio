#!/usr/bin/env python3
"""Stand-in for jobs/h3turbo_job.py, the experimental LightX2V Turbo's GPU job (one clip per process): no torch, no
GPU, no weights.

    python -I -B tests/stub/h3turbo_job.py --job <case>/turbo/job.json [--parent PID]

The server loads and runs it as it loads and runs the real job (LFSTUDIO_H3TURBO_SCRIPT; tests/stub/studio.py sets
that). Loaded: the standard-library names webui/server.py uses, with the real job's interface and values (MODEL,
JOB_VERSION, PRESETS, CANVAS, MINIMAX_PROMPT_TOKENS, RECEIPT, studio_settings with the real settings, defaults and
messages, min_free_for with the real job's memory model, parse_job with its lengths, output_name with the frames it
keeps, worker_command, worker_environment, parse_line, write_json, check_install; tests/test_server.py TurboSeam
compares them with the real job's). Run: in its own process group, its output in worker.log. It reads job.json (version
1), refuses one whose conditioning or LoRA file is missing (exit 2, as the real job), prints the real job's progress
lines ("[lf-studio] phase loading", "step 1/N" ... "step N/N", "decoding", "saving"; N = the job's steps) over
STUB_H3TURBO_S seconds, then writes in paths.output clip_<W>x<H>_<frames>f.mp4 (an ffmpeg testsrc2 pattern with a tone,
at the job's canvas, the frames the real job keeps for its length and keyframes, 24 fps, AAC), stand-ins for audio.pt
and latents.safetensors, and receipt.json. SIGTERM (Cancel) ends it at once: a "cancelled" receipt, an error line, exit
status 143. With --parent, as the real job, it is killed when that process (the server) ends.
  STUB_H3TURBO_S       seconds the job takes (default 2)
  STUB_H3TURBO_FAIL    1: fail at the first step, as a job that ran out of memory ("[lf-studio] error ...", exit 1)
  STUB_H3TURBO_ABORT   1: stop at the first step as the real job does when MemAvailable falls below its
                       abort_below_gib (a receipt "aborted", exit status 3)
  STUB_H3TURBO_TERM    "ignore": SIGTERM does nothing, like the real job's handler while a long GPU call runs
  STUB_H3TURBO_OUTPUT  "none": exit 0 without the MP4; "bad": an MP4 that ffmpeg can't read
  STUB_H3TURBO_LOG     a file each run appends a JSON line to ({"pid", "job", "argv"}), so a test can find a job and
                       its command line
Standard library only, so it runs under -I with any Python 3.10+.
"""
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TAG = "[lf-studio]"
MODEL = "lightx2v-turbo"
JOB_VERSION = 1
RECEIPT = "receipt.json"
FPS, AUDIO_RATE = 24, 32000
# the real job's tables (jobs/h3turbo_job.py)
CANVAS = {("576p", "landscape"): (1024, 576), ("576p", "portrait"): (576, 1024),
          ("768p", "landscape"): (1344, 768), ("768p", "portrait"): (768, 1344)}
QWEN_CANVASES = ("draft", "canvas")
MINIMAX_PROMPT_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>",
                         "<|caption_end|>")
PRESETS = {
    4: {"file": "minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors", "bytes": 1383677808,
        "sha256": "97b775707bb5732e01dba0cf3833d3fe2947c953859d4d05ea3bd74894c43d2b", "shift": [6, 3]},
    8: {"file": "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors", "bytes": 1383677808,
        "sha256": "fb41185df94bf01f9ac1aeedc2f07e5f748da3b237d18a16f5b3f5960839fdab", "shift": [6, 3]}}
OFFERED_STEPS = (4,)
MIN_FREE_GIB = 100
GUARD_GIB = 5
MIN_SECONDS, MAX_SECONDS = 5, 15
ABORT_BELOW_GIB = 6
USE_GIB, USE_SPREAD_GIB, FLOOR_ROWS = (77.8, 0.186), 1.6, (2, 1000)       # the real job's memory model
JOB_KEYS = {"version", "task", "orientation", "resolution", "seconds", "seed", "steps", "lora", "shift",
            "conditioning", "qwen_canvas", "first_frame", "last_frame", "audio_noise", "crf", "x264_preset", "load",
            "abort_below_gib", "paths", "meta"}
REQUIRED = ("task", "orientation", "resolution", "seconds", "seed", "steps", "lora", "shift", "conditioning",
            "qwen_canvas", "paths")
EXIT_FAILED, EXIT_REFUSED, EXIT_ABORTED, EXIT_CANCELLED = 1, 2, 3, 143


class JobError(ValueError):
    """The job or its inputs are refused (exit status 2)."""


class Cancelled(BaseException):
    pass


class RunAborted(Exception):
    pass


class OutOfMemoryError(RuntimeError):
    """As torch's: the real job's likely failure."""


def frames_for_seconds(seconds):
    n = max(5, round(float(seconds) * FPS))
    return n + (5 - n % 17) % 17


def studio_frames(draft_frames):
    return ((int(draft_frames) - 1) // 8) * 8 + 1


def kept_frames(frames, keyframes):
    """As the real job's: every frame made with a last keyframe, else the studio's length."""
    return int(frames) if "last" in keyframes else studio_frames(frames)


def output_name(job):
    width, height = CANVAS[(job["resolution"], job["orientation"])]
    return f"clip_{width}x{height}_{kept_frames(frames_for_seconds(job['seconds']), job['keyframes'])}f.mp4"


def min_free_for(resolution, seconds, floor):
    """As the real job's: the floor, or more where its memory model asks for it (768p from 12 s at 100; every clip
    with a floor below 90)."""
    if floor <= 0:
        return floor
    width, height = CANVAS[(resolution, "landscape")]
    frames, per_frame = frames_for_seconds(seconds), (height // 32) * (width // 32)
    keyframes, text = FLOOR_ROWS
    rows = text + keyframes * per_frame + 2 * int(round(frames / FPS * 40)) + ((frames - 5) // 17 * 5 + 2) * per_frame
    return max(floor, math.ceil(USE_GIB[0] + USE_GIB[1] * rows / 1000 + ABORT_BELOW_GIB + USE_SPREAD_GIB))


def studio_settings(environ):
    """As the real job's: the studio's LFSTUDIO_H3TURBO_* settings with their defaults: (settings, problems)."""
    home = Path(environ.get("LFSTUDIO_HOME") or Path.home() / "sol-h3").expanduser()
    checkpoints = Path(environ.get("LFSTUDIO_CHECKPOINTS") or home / "checkpoints").expanduser()
    settings, problems = {}, []
    for key, name, default in (("python", "LFSTUDIO_H3TURBO_PYTHON", home / "envs" / "regen2k" / "bin" / "python"),
                               ("base_model", "LFSTUDIO_H3TURBO_BASE", checkpoints / "MiniMax-H3"),
                               ("loras", "LFSTUDIO_H3TURBO_LORAS", checkpoints / "Minimax-h3-Turbo")):
        settings[key] = Path(environ.get(name) or default).expanduser()
        if not settings[key].is_absolute():
            problems.append(f"{name}={settings[key]}: must be an absolute path")
    settings["qwen_canvas"] = environ.get("LFSTUDIO_H3TURBO_QWEN_CANVAS") or "draft"
    if settings["qwen_canvas"] not in QWEN_CANVASES:
        problems.append(f"LFSTUDIO_H3TURBO_QWEN_CANVAS={settings['qwen_canvas']}: must be draft or canvas")
    for key, name, default in (("min_free_gib", "LFSTUDIO_H3TURBO_MIN_FREE_GIB", MIN_FREE_GIB),
                               ("guard_gib", "LFSTUDIO_H3TURBO_GUARD_GIB", GUARD_GIB)):
        raw = environ.get(name) or str(default)
        try:
            settings[key] = float(raw)
            if not math.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError
        except ValueError:
            problems.append(f"{name}={raw}: must be a number of GiB, 0 or more")
            settings[key] = float(default)
    return settings, problems


def check_install(settings, env=False):
    """As the real job's, cut down: the base model's folder and the LoRA files of OFFERED_STEPS are there (names
    only). Returns (report, problems)."""
    report, problems = {"base_model": str(settings["base_model"]), "loras": {}}, []
    if not Path(settings["base_model"]).is_dir():
        problems.append(f"paths.base_model: no directory {settings['base_model']}")
    for steps in OFFERED_STEPS:
        path = Path(settings["loras"]) / PRESETS[steps]["file"]
        report["loras"][steps] = {"file": str(path), "state": "ok" if path.is_file() else "missing"}
        if not path.is_file():
            problems.append(f"no {path} (the {steps}-step LoRA; install/66-h3turbo.sh fetches it)")
    return report, problems


def _path(value, name, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise JobError(f"{name}: an absolute path is needed, got {value!r}")
    return Path(value)


def parse_job(data, job_dir):
    """job.json's object -> the job, as the real job checks it (cut down: the fields, their kinds and choices)."""
    if not isinstance(data, dict):
        raise JobError("job.json must hold a JSON object")
    unknown = sorted(set(data) - JOB_KEYS)
    if unknown:
        raise JobError(f"unknown field(s): {', '.join(unknown)}")
    if data.get("version", JOB_VERSION) != JOB_VERSION:
        raise JobError(f"version: {JOB_VERSION} is the only one, got {data.get('version')!r}")
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise JobError(f"missing field(s): {', '.join(missing)}")
    if data["task"] not in ("t2va", "fl2va") or data["orientation"] not in ("landscape", "portrait"):
        raise JobError(f"task/orientation: {data['task']!r}, {data['orientation']!r}")
    if (data["resolution"], data["orientation"]) not in CANVAS or data["qwen_canvas"] not in QWEN_CANVASES:
        raise JobError(f"resolution/qwen_canvas: {data['resolution']!r}, {data['qwen_canvas']!r}")
    if type(data["steps"]) is not int or type(data["seconds"]) is not int or type(data["seed"]) is not int:
        raise JobError("steps, seconds and seed: whole numbers")
    if not MIN_SECONDS <= data["seconds"] <= MAX_SECONDS:
        raise JobError(f"seconds: a whole number from {MIN_SECONDS} to {MAX_SECONDS}, got {data['seconds']!r}")
    job = dict(data, lora=_path(data["lora"], "lora"), conditioning=_path(data["conditioning"], "conditioning"),
               first_frame=_path(data.get("first_frame"), "first_frame", required=False),
               last_frame=_path(data.get("last_frame"), "last_frame", required=False))
    if data["task"] == "t2va" and (job["first_frame"] or job["last_frame"]):
        raise JobError("t2va takes no first_frame or last_frame (fl2va does)")
    if data["task"] == "fl2va" and not (job["first_frame"] or job["last_frame"]):
        raise JobError("fl2va needs a first_frame, a last_frame or both")
    job["keyframes"] = [anchor for anchor in ("first", "last") if job[f"{anchor}_frame"]]
    paths = data["paths"]
    if not isinstance(paths, dict) or set(paths) - {"base_model", "output"}:
        raise JobError("paths: a JSON object with base_model and output")
    job["base_model"] = _path(paths.get("base_model"), "paths.base_model")
    job["output"] = _path(paths.get("output"), "paths.output", required=False) or Path(job_dir)
    return job


def worker_command(python, script, job_path, parent=None):
    return [str(python), "-I", "-B", str(script), "--job", str(job_path)] + \
        (["--parent", str(int(parent))] if parent is not None else [])


def worker_environment(environ):
    """As the real job's: offline, no token."""
    env = {k: v for k, v in environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def parse_line(line):
    line = line.rstrip("\r\n")
    for kind in ("phase", "error"):
        prefix = f"{TAG} {kind} "
        if line.startswith(prefix) and line[len(prefix):].strip():
            return kind, line[len(prefix):].strip()
    return None


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def cancel(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    raise Cancelled("stopped by SIGTERM")


def say(kind, text):
    print(f"{TAG} {kind} {text}", flush=True)


def tie_to_parent(parent):
    """As the real job's: SIGKILL when `parent` ends; False if it has ended already."""
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:     # 1 = PR_SET_PDEATHSIG
        return True
    return os.getppid() == parent


def write_mp4(path, width, height, frames, output):
    partial = path.with_name(f".{path.name}.partial.mp4")
    if output == "bad":
        partial.write_bytes(b"not an mp4\n" * 100)
    else:
        seconds = frames / FPS
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=s={width}x{height}:r={FPS}", "-f", "lavfi", "-i",
                        f"sine=frequency=330:sample_rate={AUDIO_RATE}:duration={seconds}", "-frames:v", str(frames),
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2",
                        str(partial)], check=True, timeout=120)
    os.replace(partial, path)


def main(argv):
    if "--parent" in argv and not tie_to_parent(int(argv[argv.index("--parent") + 1])):
        say("error", "the studio's server is gone; not starting")
        return EXIT_CANCELLED
    ignore = os.environ.get("STUB_H3TURBO_TERM") == "ignore"
    signal.signal(signal.SIGTERM, signal.SIG_IGN if ignore else cancel)
    job_path = Path(argv[argv.index("--job") + 1]).resolve()
    if os.environ.get("STUB_H3TURBO_LOG"):
        with open(os.environ["STUB_H3TURBO_LOG"], "a") as log:
            log.write(json.dumps({"pid": os.getpid(), "job": str(job_path), "argv": argv[1:]}) + "\n")
    receipt = {"state": "running", "model": MODEL, "pid": os.getpid(), "job_file": str(job_path), "times_s": {},
               "stub": True}
    started, status, out = time.time(), 0, job_path.parent
    try:
        job = parse_job(json.loads(job_path.read_text()), job_path.parent)
        out = job["output"]
        receipt["job"] = json.loads(job_path.read_text())
        for name in ("conditioning", "lora"):
            if not job[name].is_file():
                raise JobError(f"no {name} file {job[name]}")
        receipt["lora"] = {"file": job["lora"].name, "scale": 0.0625}
        write_json(out / RECEIPT, receipt)
        phases = ["loading"] + [f"step {i}/{job['steps']}" for i in range(1, job["steps"] + 1)] + \
            ["decoding", "saving"]
        pause = float(os.environ.get("STUB_H3TURBO_S", "2")) / len(phases)
        for name in phases:
            say("phase", name)
            if name.startswith("step"):
                if os.environ.get("STUB_H3TURBO_FAIL") == "1":
                    raise OutOfMemoryError("CUDA out of memory (STUB_H3TURBO_FAIL)")
                if os.environ.get("STUB_H3TURBO_ABORT") == "1":
                    raise RunAborted("MemAvailable 5.2 GiB, below abort_below_gib 6 (STUB_H3TURBO_ABORT)")
            t0, key = time.time(), name.split()[0]
            time.sleep(pause)
            receipt["times_s"][key] = round(receipt["times_s"].get(key, 0) + time.time() - t0, 2)
        receipt["load_s"] = receipt["times_s"]["loading"]
        width, height = CANVAS[(job["resolution"], job["orientation"])]
        frames = kept_frames(frames_for_seconds(job["seconds"]), job["keyframes"])
        name = output_name(job)
        if os.environ.get("STUB_H3TURBO_OUTPUT") != "none":
            write_mp4(out / name, width, height, frames, os.environ.get("STUB_H3TURBO_OUTPUT"))
        for stand_in in ("audio.pt", "latents.safetensors"):
            (out / stand_in).write_bytes(b"stub " + stand_in.encode())
        receipt.update(state="done", output={"file": name, "size": [width, height], "frames": frames,
                                             "samples": int(frames / FPS * AUDIO_RATE)})
    except Cancelled as error:
        receipt.update(state="cancelled", error=str(error))
        status = EXIT_CANCELLED
    except RunAborted as error:
        receipt.update(state="aborted", error=str(error))
        status = EXIT_ABORTED
    except JobError as error:
        receipt.update(state="refused", error=str(error))
        status = EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - as the real job: recorded, and the exit status says it failed
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        status = EXIT_FAILED
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        receipt.update(wall_s=round(time.time() - started, 2), exit_status=status, peak_allocated_gib=77.4,
                       lowest_mem_available_gib={"all": 35.2})
        if out.is_dir():
            write_json(out / RECEIPT, receipt)
    if status:
        say("error", receipt["error"])
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
