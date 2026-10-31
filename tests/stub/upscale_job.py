#!/usr/bin/env python3
"""Stand-in for upscale/regen2k_job.py, the experimental Upscale's GPU job: no torch, no GPU, no weights.

    python -I -B tests/stub/upscale_job.py --job <dir>/job.json [--parent PID]

The server runs it as it runs the real job (LFSTUDIO_UPSCALE_SCRIPT; tests/stub/studio.py sets that): in its own
process group, with its output in worker.log. It reads job.json, prints the real job's progress lines
("[lf-studio] phase encoding", "loading", "refine 1/N" ... "refine N/N", "decoding", "saving"; N = the job's steps)
spread over STUB_UPSCALE_S seconds, then writes paths.output/video_noaudio.mp4 (a small ffmpeg testsrc2 pattern with
the clip's frames, at 24 fps, no audio, as the real job's MP4 has none; upright when the clip's MP4,
<case>/stage2/refined_*.mp4, is, as the real job takes the orientation from it) and receipt.json, with the real job's
"frames" (frame_case), "rows" (case_rows), "forward" (forward_for) and "lean_chunks" (lean_chunks_for) and, for a
job.json with a "pin", its "pin": per keyframe its picture checked by its sha256 as
the real job checks it (PIN_WHY: gone, changed; a last end of a clip that keeps fewer frames than made: past_end),
else held at 33.0 dB (STUB_UPSCALE_PIN_DB sets others: "last=25.4", below the 28 needed, is not held: far). SIGTERM
(Cancel) ends it at once, like the real job between two GPU calls: a "cancelled" receipt, an error line, exit status
143. With --parent, as the real job, it is killed when that process (the server) ends. Before anything else it checks
job.json's "case" as the real job does (the real job's own case_inputs, from upscale/regen2k_job.py: one
stage2/refined_*.mp4, the conditioning, the audio for audio clip), that its MP4 has the frames at 24 fps the real job
takes (check_final's FRAME_CASES: a length the studio makes; the stub clips' size is their own), that the real job
offers its size for the clip's length (offer), that a conditioning.json beside the conditioning is t2va's (check_text)
and its "pin" (the real job's parse_pin), and job.json's "min_free_gib" against MemAvailable as the real job does
(check_start_memory; read from LFSTUDIO_TEST_MEMINFO, the stub studio's file, and not at all without it); else, as the
real job, a "refused" error line and exit status 2.
  STUB_UPSCALE_S          seconds the job takes (default 2)
  STUB_UPSCALE_FAIL       1: fail after loading, as a job that ran out of memory ("[lf-studio] error ...", exit 1)
  STUB_UPSCALE_LOG        a file each run appends a JSON line to ({"pid", "job", "argv"}), so a test can find a
                          job and its command line
  STUB_UPSCALE_TERM       "ignore": SIGTERM does nothing, like the real job's handler while a long GPU call runs
  STUB_UPSCALE_OUTPUT     "none": exit 0 without video_noaudio.mp4; "bad": a video_noaudio.mp4 that is not an MP4
                          (the server's mux fails)
  STUB_UPSCALE_PIN_DB     each end's agreement in dB, as "first=31.2,last=25.4" (default 33.0)
Standard library only, so it runs under -I with any Python 3.10+.
"""
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TAG = "[lf-studio]"
REAL_JOB = Path(__file__).resolve().parents[2] / "upscale" / "regen2k_job.py"
SIZE = {"1080p": (384, 216), "2k": (512, 288), "4k": (768, 432)}      # small; the server names the file by the setting


def orientation(case):
    """The clip's orientation from its MP4's size, as the real job's check_final reads it (landscape when unknown)."""
    finals = sorted(Path(case).glob("stage2/refined_*.mp4"))
    if not finals:
        return "landscape"
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(finals[0])], capture_output=True, text=True, timeout=60)
    try:
        width, height = map(int, probe.stdout.split(",")[:2])
    except ValueError:
        return "landscape"
    return "portrait" if height > width else "landscape"


class Cancelled(BaseException):
    pass


class OutOfMemoryError(RuntimeError):
    """As torch's: the real job's likely failure."""


def real_job():
    spec = importlib.util.spec_from_file_location("regen2k_job", REAL_JOB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_case(job):
    """The real job's checks of its input folder and pin (the docstring): (None, the MP4's frames), or (why it is
    refused, None)."""
    regen = real_job()
    try:
        files = regen.case_inputs({"case": Path(job["case"]), "audio": job["audio"]})
        if "pin" in job:
            regen.parse_pin(job["pin"])
    except regen.JobError as error:
        return str(error), None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_frames,r_frame_rate", "-of", "json", str(files["final"])],
                           capture_output=True, text=True, timeout=60)
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        found = (stream.get("nb_frames"), stream.get("r_frame_rate"))
    except (ValueError, KeyError, IndexError):
        found = None
    if found is None or found[0] not in {str(frames) for frames in regen.FRAME_CASES} or found[1] != regen.FPS:
        counts = sorted(regen.FRAME_CASES)
        return (f"{files['final'].name} is {found}: not a clip the Upscale takes (at 24 fps, with the frames of a "
                f"length the studio makes: {', '.join(map(str, counts[:-1]))} or {counts[-1]}); it never cuts a clip"), \
            None
    offered = regen.offer(int(found[0]), job["resolution"])
    if "why" in offered:
        return offered["why"], None
    receipt = files["conditioning"].with_suffix(".json")
    if receipt.is_file() and json.loads(receipt.read_text()).get("task") not in (None, "t2va"):
        return f"the conditioning is {json.loads(receipt.read_text())['task']}'s: it needs a text-only encode", None
    if os.environ.get("LFSTUDIO_TEST_MEMINFO"):          # never the machine's own memory
        try:
            regen.check_start_memory({"min_free_gib": job.get("min_free_gib"), "resolution": job["resolution"],
                                      "frames": regen.frame_case(int(found[0]))}, os.environ["LFSTUDIO_TEST_MEMINFO"])
        except regen.JobError as error:
            return str(error), None
    return None, int(found[0])


def check_pin(pin, frames):
    """The receipt's "pin" (the real job's shape, check_pin's records): each keyframe's picture by its sha256, then
    the dB of STUB_UPSCALE_PIN_DB (33.0 by default) against the real job's 28."""
    regen = real_job()
    pin = regen.parse_pin(pin)
    dbs = dict(item.split("=") for item in os.environ.get("STUB_UPSCALE_PIN_DB", "").split(",") if "=" in item)
    record = {"asked": [k["anchor"] for k in pin["keyframes"]]}
    for keyframe in pin["keyframes"]:
        anchor = keyframe["anchor"]
        frame = 0 if anchor == "first" else frames["kept"] - 1
        result = record[anchor] = {"used": False, "agreement_db": None, "frame": frame, "fit": keyframe["fit"],
                                   "box": None, "size": None, "why": None}
        try:
            data = Path(keyframe["image"]).read_bytes()
        except OSError:
            result["why"] = regen.PIN_WHY["gone"]
            continue
        if hashlib.sha256(data).hexdigest() != keyframe["sha256"]:
            result["why"] = regen.PIN_WHY["changed"]
            continue
        result["agreement_db"] = db = float(dbs.get(anchor, 33.0))
        if anchor == "last" and frames["kept"] != frames["made"]:
            result["why"] = regen.PIN_WHY["past_end"].format(n=frames["made"] - frames["kept"])
        elif db < pin["check"]["min_db"]:
            result["why"] = regen.PIN_WHY["far"].format(frame=frame, db=db, min=pin["check"]["min_db"])
        else:
            result["used"] = True
    record.update(min_db=pin["check"]["min_db"], noise_aug=pin["noise_aug"], encode_seed=pin["encode_seed"],
                  noise_seed=pin["noise_seed"])
    return record


def cancel(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    raise Cancelled("stopped by SIGTERM")


def say(kind, text):
    print(f"{TAG} {kind} {text}", flush=True)


def write_json(path, data):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def tie_to_parent(parent):
    """As the real job's: SIGKILL when `parent` ends; False if it has ended already."""
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:     # 1 = PR_SET_PDEATHSIG
        return True
    return os.getppid() == parent


def main(argv):
    if "--parent" in argv and not tie_to_parent(int(argv[argv.index("--parent") + 1])):
        say("error", "the studio's server is gone; not starting")
        return 143
    ignore = os.environ.get("STUB_UPSCALE_TERM") == "ignore"
    signal.signal(signal.SIGTERM, signal.SIG_IGN if ignore else cancel)
    job_path = Path(argv[argv.index("--job") + 1]).resolve()
    job = json.loads(job_path.read_text())
    out = Path(job["paths"].get("output") or job_path.parent)
    if os.environ.get("STUB_UPSCALE_LOG"):
        with open(os.environ["STUB_UPSCALE_LOG"], "a") as log:
            log.write(json.dumps({"pid": os.getpid(), "job": str(job_path), "argv": argv[1:]}) + "\n")
    receipt = {"state": "running", "pid": os.getpid(), "job": job, "orientation": orientation(job["case"]),
               "times_s": {}, "stub": True}
    refused, kept = check_case(job)
    if refused:
        receipt.update(state="refused", error=refused, exit_status=2)
        write_json(out / "receipt.json", receipt)
        say("error", refused)
        return 2
    regen = real_job()
    receipt["frames"] = frames = regen.frame_case(kept)
    receipt.update(rows=regen.case_rows(frames, job["resolution"], len((job.get("pin") or {}).get("keyframes", ()))),
                   forward=regen.forward_for(frames, job["resolution"]),
                   lean_chunks=regen.lean_chunks_for(frames, job["resolution"]))
    if "pin" in job:
        receipt["pin"] = check_pin(job["pin"], frames)
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
        width, height = SIZE[job["resolution"]][::1 if receipt["orientation"] == "landscape" else -1]
        partial = out / ".video_noaudio.mp4.partial.mp4"
        output = os.environ.get("STUB_UPSCALE_OUTPUT")
        if output == "bad":
            partial.write_bytes(b"not an mp4\n" * 100)
        elif output != "none":
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"testsrc2=s={width}x{height}:r=24", "-frames:v", str(kept), "-an", "-c:v", "libx264",
                            "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(partial)], check=True, timeout=120)
        if output != "none":
            os.replace(partial, out / "video_noaudio.mp4")
        receipt.update(state="done", output={"file": "video_noaudio.mp4", "size": [width, height], "frames": kept})
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
