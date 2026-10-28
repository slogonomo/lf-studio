#!/usr/bin/env python3
"""Stand-in for upscale/regen2k_job.py, the experimental Upscale's GPU job: no torch, no GPU, no weights.

    python -I -B tests/stub/upscale_job.py --job <dir>/job.json

The server runs it as it runs the real job (LFSTUDIO_UPSCALE_SCRIPT; tests/stub/studio.py sets that): in its own
process group, with its output in worker.log. It reads job.json, prints the real job's progress lines
("[lf-studio] phase encoding", "loading", "refine 1/N" ... "refine N/N", "decoding", "saving"; N = the job's steps)
spread over STUB_UPSCALE_S seconds, then writes paths.output/video_noaudio.mp4 (a small ffmpeg testsrc2 pattern,
121 frames at 24 fps, no audio, as the real job's MP4 has none) and receipt.json. SIGTERM (Cancel) ends it at once,
like the real job: a "cancelled" receipt, an error line, exit status 143.
  STUB_UPSCALE_S     seconds the job takes (default 2)
  STUB_UPSCALE_FAIL  1: fail after loading, as a job that ran out of memory ("[lf-studio] error ...", exit 1)
  STUB_UPSCALE_LOG   a file each run appends a JSON line to ({"pid", "job"}), so a test can find a job it left
                     running (the server killed with SIGKILL)
Standard library only, so it runs under -I with any Python 3.10+.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TAG = "[lf-studio]"
SIZE = {"1080p": (384, 216), "2k": (512, 288), "4k": (768, 432)}      # small; the server names the file by the setting


class Cancelled(BaseException):
    pass


class OutOfMemoryError(RuntimeError):
    """As torch's: the real job's likely failure."""


def cancel(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    raise Cancelled("stopped by SIGTERM")


def say(kind, text):
    print(f"{TAG} {kind} {text}", flush=True)


def write_json(path, data):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def main(argv):
    signal.signal(signal.SIGTERM, cancel)
    job_path = Path(argv[argv.index("--job") + 1]).resolve()
    job = json.loads(job_path.read_text())
    out = Path(job["paths"].get("output") or job_path.parent)
    if os.environ.get("STUB_UPSCALE_LOG"):
        with open(os.environ["STUB_UPSCALE_LOG"], "a") as log:
            log.write(json.dumps({"pid": os.getpid(), "job": str(job_path)}) + "\n")
    receipt = {"state": "running", "pid": os.getpid(), "job": job, "times_s": {}, "stub": True}
    started, status = time.time(), 0
    phases = ["encoding", "loading"] + [f"refine {i}/{job['steps']}" for i in range(1, job["steps"] + 1)] + \
        ["decoding", "saving"]
    pause = float(os.environ.get("STUB_UPSCALE_S", "2")) / len(phases)
    try:
        write_json(out / "receipt.json", receipt)
        for name in phases:
            say("phase", name)
            if name.startswith("refine") and os.environ.get("STUB_UPSCALE_FAIL") == "1":
                raise OutOfMemoryError("CUDA out of memory (STUB_UPSCALE_FAIL)")
            t0, key = time.time(), name.split()[0]                # refine k/N: one time for all steps
            time.sleep(pause)
            receipt["times_s"][key] = round(receipt["times_s"].get(key, 0) + time.time() - t0, 2)
        width, height = SIZE[job["resolution"]]
        partial = out / ".video_noaudio.mp4.partial.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=s={width}x{height}:r=24", "-frames:v", "121", "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(partial)], check=True, timeout=120)
        os.replace(partial, out / "video_noaudio.mp4")
        receipt.update(state="done", output={"file": "video_noaudio.mp4", "size": [width, height], "frames": 121})
    except Cancelled as error:
        receipt.update(state="cancelled", error=str(error))
        status = 143
    except Exception as error:  # noqa: BLE001 - as the real job: recorded, and the exit status says it failed
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        status = 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        receipt.update(wall_s=round(time.time() - started, 2), exit_status=status, peak_allocated_gib=91.0,
                       lowest_mem_available_gib={"all": 15.0})
        write_json(out / "receipt.json", receipt)
    if status:
        say("error", receipt["error"])
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
