"""Stub of runtime.pipeline.Pipeline: no models, no GPU.

Loading and generating are `sleep` processes in their own sessions, like the real workers, so the server can stop
them (a stopped warm-up, a cancelled clip). A pipeline whose worker was stopped by a signal is broken like a real one
whose workers died: it fails until it is closed and a new one is started. Each clip is a small ffmpeg test pattern
with a tone, with the requested format and length (seconds x 24 + 1 frames, as the real clips), next to stand-ins
for the real clip's working files the Upscale reads (qwen/conditioning.pt and stage1/stage1_direct_tensors.pt).
  STUB_LOAD_S    seconds the loading pass waits before its clip (default 3)
  STUB_CLIP_S    seconds per clip once loaded (default 2)
  STUB_CLOSE_S   seconds close() takes (default 0; the real pipeline needs ~40 s)
  STUB_FINISH_S  seconds a clip takes to come back after its worker ended (default 0), with a FINISHING file in its
                 request directory meanwhile: no worker runs then, so a signal can't stop it (a clip finishing just
                 as it is cancelled)
  STUB_LATE_S    seconds into the loading pass when a second worker starts (default: none). It runs to the end of
                 the pass, which waits for it even if the first one failed, as the real pass waits for the workers
                 it started (Stage2, started once Stage1 is loaded): a worker that starts after a Cancel
  Each pipeline writes STARTED into its root when start() begins and CLOSED when close() ends, so a test can check
  that no started pipeline was left open. Like the real pipeline, a request makes its directory when it begins.
  A prompt containing STUB_FAIL makes the request fail as if a worker died; STUB_FAIL_ONCE does that only the first
  time a clip runs (a .failed-once file next to the session directories remembers it), so the server's retry runs.
  reshape() (another length or orientation, models kept) is a `sleep` of STUB_RESHAPE_S (default 0.2) and appends a
  line to RESHAPES in the root; STUB_RESHAPE_FAIL=1 makes it fail like a dead worker. Like the real pipeline, close()
  first marks the pipeline as closing, reshape() and generate() refuse a closing one, and a reshape that a close
  overlaps fails.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

SIZE = {"landscape": (336, 192), "portrait": (192, 336)}


class Pipeline:
    def __init__(self, paths, output_root, *, task="t2va", orientation=None, seconds=None, **_):
        self.task = task
        self.orientation = orientation or "landscape"
        self.size = SIZE[self.orientation]
        self.seconds = seconds or 5
        self.root = Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = self.broken = self.closing = False
        (self.root / "results.json").write_text(json.dumps({"status": "STARTING", "task": task}))

    def _work(self, seconds, check=True):
        if check and self.broken:
            raise RuntimeError("a worker of this pipeline died; close it and start a new one")
        p = subprocess.Popen(["sleep", str(seconds)], start_new_session=True)
        if p.wait() != 0:
            self.broken = p.returncode < 0                           # stopped by a signal (SIGTERM from the server)
            raise RuntimeError(f"Worker exited with code {p.returncode}")

    def _clip(self, case, request):
        once = self.root.parent / f".failed-once-{case['case_id']}"
        if "STUB_FAIL" in case["prompt"] and not ("STUB_FAIL_ONCE" in case["prompt"] and once.exists()):
            (request / "stage1").mkdir(parents=True, exist_ok=True)       # a run that died part-way leaves files
            once.touch()
            raise RuntimeError("Worker exited with code 1 (STUB_FAIL)")
        finish = float(os.environ.get("STUB_FINISH_S", "0"))
        if finish:
            (request / "FINISHING").touch()
            time.sleep(finish)
        for name in ("qwen/conditioning.pt", "stage1/stage1_direct_tensors.pt"):
            (request / name).parent.mkdir(parents=True, exist_ok=True)       # a real clip's working files (the
            (request / name).write_bytes(b"stub " + name.encode())          # Upscale reads them), as stand-ins
        w, h = self.size
        frames = self.seconds * 24 + 1                   # as the real clips: 121 frames at 5 s
        out = request / "stage2" / f"refined_{w}x{h}_{frames}f.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        # (by the pattern's length, half a frame short of the next frame: no -frames:v, which a test's ffmpeg on PATH
        # takes for the poster's)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=s={w}x{h}:r=24:d={(frames - 0.5) / 24:.4f}", "-f", "lavfi",
                        "-i", f"sine=frequency=440:duration={frames / 24:.4f}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
                       check=True, timeout=120)
        return str(out)

    def _row(self, case, request, started):
        output = self._clip(case, request)
        return {"case_id": case["case_id"], "status": "PASS", "output": output, "e2e_s": time.time() - started,
                "stage1_s": 0.0, "stage2_s": 0.0}

    def start(self, case, deliver=False):
        (self.root / "STARTED").touch()
        started = time.time()
        request = self.root / (case["case_id"] if deliver else "warmup")
        request.mkdir()
        load, late, errors = float(os.environ.get("STUB_LOAD_S", "3")), os.environ.get("STUB_LATE_S"), []

        def second():
            time.sleep(float(late))
            try:
                self._work(max(0.0, load - float(late)), check=False)
            except RuntimeError as error:
                errors.append(error)
        helper = threading.Thread(target=second) if late else None
        if helper:
            helper.start()
        try:
            self._work(load)
        finally:
            if helper:
                helper.join()
        if errors:
            raise errors[0]
        media = [case.get("first_frame"), case.get("last_frame")] + [r["path"] for r in case.get("references", [])]
        for m in filter(None, media):                    # the real loading pass reads its inputs (Qwen, Stage1)
            Path(m).read_bytes()
        self.started = True
        return self._row(case, request, started)

    def generate(self, case):
        if not self.started:
            raise RuntimeError("start() first")
        if self.closing:
            raise RuntimeError("The pipeline is closing")
        started = time.time()
        request = self.root / case["case_id"]
        request.mkdir()
        self._work(float(os.environ.get("STUB_CLIP_S", "2")))
        return self._row(case, request, started)

    def reshape(self, *, orientation=None, seconds=None):
        if self.closing:
            raise RuntimeError("The pipeline is closing")
        before = {"orientation": self.orientation, "seconds": self.seconds}
        after = {"orientation": orientation or "landscape", "seconds": seconds or 5}
        if after == before:
            return {"from": before, "to": after, "changed": False, "reshape_s": 0.0}
        if os.environ.get("STUB_RESHAPE_FAIL") == "1":
            self.broken = True
            raise RuntimeError("Worker exited with code 1 (STUB_RESHAPE_FAIL)")
        started = time.time()
        self._work(float(os.environ.get("STUB_RESHAPE_S", "0.2")))
        if self.closing:                       # close() began meanwhile (Unload): no change on a closing pipeline
            raise RuntimeError("The pipeline is closing")
        self.orientation, self.seconds = after["orientation"], after["seconds"]
        self.size = SIZE[self.orientation]
        with (self.root / "RESHAPES").open("a") as log:
            log.write(json.dumps({"from": before, "to": after}) + "\n")
        return {"from": before, "to": after, "changed": True, "reshape_s": time.time() - started}

    def close(self):
        self.closing = True
        time.sleep(float(os.environ.get("STUB_CLOSE_S", "0")))
        self.started = False
        (self.root / "CLOSED").touch()
