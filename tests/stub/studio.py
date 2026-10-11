#!/usr/bin/env python3
"""Run webui/server.py against the stub runtime in tests/stub/runtime: no models, no GPU, small real MP4 clips.

usage: tests/stub/studio.py [--port N] [--dir DIR] [--load-s S] [--clip-s S] [--no-preload]

Everything (runtime root, state, manifests) goes under --dir, default a new temp directory, so it never touches
a real install. Ctrl-C stops it. Open the printed URL, or point tests/ui_test.py at it.
The tests import StubStudio from here.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1] / "webui" / "server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubStudio:
    def __init__(self, root, port=None, load_s=3, clip_s=2, preload=True, env=None):
        self.root = Path(root)
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.runtime, self.state_dir = self.root / "runtime", self.root / "state"
        for d in (self.runtime, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)
        for task in ("t2va", "fl2va", "ref2va"):
            (self.root / f"paths-{task}.json").write_text("{}")
        self.env = dict(os.environ, SOL_H3_SPARK_RUNTIME_ROOT=str(self.runtime),
                        SOLH3_PATHS=str(self.root / "paths-{task}.json"), SOLH3_STATE_DIR=str(self.state_dir),
                        SOLH3_HOST="127.0.0.1", SOLH3_PORT=str(self.port), SOLH3_PRELOAD="1" if preload else "0",
                        STUB_LOAD_S=str(load_s), STUB_CLIP_S=str(clip_s), PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("SOLH3_PUBLISH_DIR", None)
        self.env.update(env or {})
        self.proc = None
        self.log = self.root / "server.log"

    def start(self, timeout=15):
        if getattr(self, "_logf", None):
            self._logf.close()
        self._logf = self.log.open("a")
        self.proc = subprocess.Popen([sys.executable, str(SERVER)], cwd=HERE, env=self.env,
                                     stdout=self._logf, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._logf.close()
                raise RuntimeError(f"stub studio exited: {self.log.read_text()[-2000:]}")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1)
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("stub studio did not answer /health")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc:
            self._logf.close()

    # ---- HTTP helpers
    def get(self, path, headers=None):
        with urllib.request.urlopen(urllib.request.Request(self.url + path, headers=headers or {}), timeout=10) as r:
            return r.status, dict(r.headers), r.read()

    def state(self):
        return json.loads(self.get("/api/state")[2])

    def post(self, path, fields=None, files=None, json_body=None):
        """POST form fields and files ({name: [(filename, bytes), ...]}) as multipart, or a JSON body.
        Returns (status, parsed JSON)."""
        if json_body is not None:
            body, ctype = json.dumps(json_body).encode(), "application/json"
        else:
            b = "stubboundary7MA4YWxk"
            parts = []
            for k, v in (fields or {}).items():
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
            for k, items in (files or {}).items():
                for fn, data in items:
                    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
                                 f'Content-Type: application/octet-stream\r\n\r\n'.encode() + data + b"\r\n")
            body, ctype = b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"
        req = urllib.request.Request(self.url + path, body, {"Content-Type": ctype}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def generate(self, task="t2va", orientation="landscape", seconds=5, prompt="a test clip", seed=1, files=None,
                 **fields):
        return self.post("/api/generate", dict(task=task, orientation=orientation, seconds=seconds, prompt=prompt,
                                               seed=seed, **fields), files)

    def all_jobs(self):
        """Every job the gallery can reach, by id: /api/state, then /api/jobs page by page."""
        st = self.state()
        jobs = {j["id"]: j for j in st["jobs"]}
        before = min((j["created"] for j in st["jobs"]), default=float("inf"))
        while st.get("more"):
            page = json.loads(self.get(f"/api/jobs?before={before}&limit=100")[2])
            jobs.update((j["id"], j) for j in page["jobs"] if j["id"] not in jobs)
            if not page["more"] or not page["jobs"]:
                break
            before = page["jobs"][-1]["created"]
        return jobs

    def job(self, jid):
        jobs = {j["id"]: j for j in self.state()["jobs"]}
        return jobs.get(jid) or self.all_jobs().get(jid)

    def wait(self, jids, timeout=60):
        """Wait until every job in jids is done, failed or cancelled; returns them in order."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            jobs = {j["id"]: j for j in self.state()["jobs"]}
            if any(i not in jobs for i in jids):
                jobs = self.all_jobs()               # older than the newest page
            if all(jobs.get(i, {}).get("status") in ("done", "failed", "cancelled") for i in jids):
                return [jobs[i] for i in jids]
            time.sleep(0.2)
        raise TimeoutError(f"jobs not finished after {timeout} s: {jids}")

    def wait_for(self, predicate, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = self.state()
            if predicate(s):
                return s
            time.sleep(0.1)
        raise TimeoutError("condition not reached")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=18299)
    ap.add_argument("--dir", help="keep runtime and state here (default: a new temp directory)")
    ap.add_argument("--load-s", type=float, default=3)
    ap.add_argument("--clip-s", type=float, default=2)
    ap.add_argument("--no-preload", action="store_true")
    a = ap.parse_args()
    root = Path(a.dir or tempfile.mkdtemp(prefix="sol-h3-stub-"))
    s = StubStudio(root, a.port, a.load_s, a.clip_s, not a.no_preload).start()
    print(f"stub studio on {s.url}/  (files under {root}, log {s.log})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # stop the server too, as on Ctrl-C
    try:
        s.proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()


if __name__ == "__main__":
    main()
