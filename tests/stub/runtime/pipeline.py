"""Stub of runtime.pipeline.Pipeline: no models, no GPU.

Loading and generating are `sleep` processes in their own sessions, like the real workers, so the server can stop
them (a stopped warm-up). Each clip is a small ffmpeg test pattern with a tone, with the requested format and length.
  STUB_LOAD_S   seconds the loading pass waits before its clip (default 3)
  STUB_CLIP_S   seconds per clip once loaded (default 2)
  STUB_CLOSE_S  seconds close() takes (default 0; the real pipeline needs ~40 s)
  Each pipeline writes STARTED into its root when start() begins and CLOSED when close() ends, so a test can check
  that no started pipeline was left open.
  A prompt containing STUB_FAIL makes the request fail as if a worker died.
"""
import json
import os
import subprocess
import time
from pathlib import Path

SIZE = {"landscape": (336, 192), "portrait": (192, 336)}


class Pipeline:
    def __init__(self, paths, output_root, *, task="t2va", orientation=None, seconds=None, **_):
        self.task = task
        self.size = SIZE[orientation or "landscape"]
        self.seconds = seconds or 5
        self.root = Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = False
        (self.root / "results.json").write_text(json.dumps({"status": "STARTING", "task": task}))

    @staticmethod
    def _work(seconds):
        p = subprocess.Popen(["sleep", str(seconds)], start_new_session=True)
        if p.wait() != 0:
            raise RuntimeError(f"Worker exited with code {p.returncode}")

    def _clip(self, case, request):
        if "STUB_FAIL" in case["prompt"]:
            (request / "stage1").mkdir(parents=True, exist_ok=True)       # a run that died part-way leaves files
            raise RuntimeError("Worker exited with code 1 (STUB_FAIL)")
        w, h = self.size
        out = request / "stage2" / f"refined_{w}x{h}_{self.seconds * 24 + 1}f.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=24:d={self.seconds}",
                        "-f", "lavfi", "-i", f"sine=frequency=440:duration={self.seconds}", "-shortest",
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
        self._work(float(os.environ.get("STUB_LOAD_S", "3")))
        media = [case.get("first_frame"), case.get("last_frame")] + [r["path"] for r in case.get("references", [])]
        for m in filter(None, media):                    # the real loading pass reads its inputs (Qwen, Stage1)
            Path(m).read_bytes()
        self.started = True
        return self._row(case, self.root / (case["case_id"] if deliver else "warmup"), started)

    def generate(self, case):
        if not self.started:
            raise RuntimeError("start() first")
        started = time.time()
        self._work(float(os.environ.get("STUB_CLIP_S", "2")))
        return self._row(case, self.root / case["case_id"], started)

    def close(self):
        time.sleep(float(os.environ.get("STUB_CLOSE_S", "0")))
        self.started = False
        (self.root / "CLOSED").touch()
