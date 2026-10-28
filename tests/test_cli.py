"""bin/lf-cli, the client for scripts (#29), run as a subprocess against the stub studio (tests/stub) and against small
fake servers that answer as spark-switchboard or a studio shutting down would. No models, no GPU.

python3 -m unittest tests.test_cli -v      (or with the rest: python3 -m unittest discover -s tests)

The stubs and fake servers of this file and of test_mcp.py listen on ports 18820-18839.
"""
import ast
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BIN = REPO / "bin" / "lf-cli"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "stub"))
from studio import StubStudio  # noqa: E402
from test_server import mp4, png  # noqa: E402

if not shutil.which("ffmpeg"):
    raise unittest.SkipTest("ffmpeg not installed")

PORTS = range(18820, 18840)
WAV = b"RIFF0000WAVEfmt "          # the stub never reads audio files


def free_port(taken=()):
    """A port in 18820-18839 that nothing listens on."""
    for port in PORTS:
        if port in taken:
            continue
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # as the servers do: TIME_WAIT is fine
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port in 18820-18839")


def load_cli():
    """bin/lf-cli as a module (it has no .py name), without writing bytecode next to it."""
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        loader = importlib.machinery.SourceFileLoader("lf_cli", str(BIN))
        module = importlib.util.module_from_spec(importlib.util.spec_from_loader("lf_cli", loader))
        loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = before


def client_env(url=None, **extra):
    """This process's environment without LFSTUDIO_* settings (or their old names), plus the given ones."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LFSTUDIO_", "SOLH3_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if url:
        env["LFSTUDIO_URL"] = url
    env.update(extra)
    return env


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


class Fake:
    """A local HTTP server for the client to talk to: answer(method, path, headers, body) gives (status, headers,
    body), or None to close the connection without an answer; each request is recorded as (method, path, headers,
    body)."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                fake.serve(self)

            do_POST = do_HEAD = do_GET

        self.server = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def serve(self, h):
        body = h.rfile.read(int(h.headers.get("Content-Length") or 0))
        self.requests.append((h.command, h.path, dict(h.headers), body))
        answer = self.answer(h.command, h.path, h.headers, body)
        if answer is None:
            h.close_connection = True
            return
        status, headers, data = answer
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        h.send_response(status)
        for k, v in headers.items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)

    def paths(self):
        return [p for _, p, _, _ in self.requests]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


JSON = {"Content-Type": "application/json"}


def switchboard_error(code, message, retry=None):
    """spark-switchboard's 503 for a request it can't pass on (adapters/base.py)."""
    headers = dict(JSON, **({"Retry-After": str(retry)} if retry else {}))
    return 503, headers, {"error": {"type": "service_unavailable", "code": code, "message": message},
                          "switchboard": {"service": "lf-studio", "state": "stopped", "retry_after_s": retry}}


class CliTest(unittest.TestCase):
    """One fresh stub studio per test (with studio_env's settings), and a working directory for the client."""
    load_s, clip_s = 1, 0.5
    studio_env = {}

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-cli-"))
        self.work = self.dir / "work"
        self.work.mkdir()
        self.studio = StubStudio(self.dir / "studio", port=free_port(), load_s=self.load_s, clip_s=self.clip_s,
                                 preload=False, env=self.studio_env).start()

    def tearDown(self):
        self.studio.stop()
        if hasattr(self, "_outcome") and not getattr(self._outcome, "success", True):
            print(f"\n--- server log ({self.studio.log})\n" + self.studio.log.read_text()[-3000:])
        shutil.rmtree(self.dir, ignore_errors=True)

    def sol(self, *args, url=None, env=None, stdin=None, timeout=120, cwd=None):
        """Run bin/lf-cli with these arguments in self.work (or cwd): the CompletedProcess (text)."""
        return subprocess.run([sys.executable, str(BIN), *map(str, args)], input=stdin, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd or self.work,
                              env=client_env(url or self.studio.url, **(env or {})))

    def ok(self, *args, **kw):
        p = self.sol(*args, **kw)
        self.assertEqual(p.returncode, 0, f"stdout: {p.stdout}\nstderr: {p.stderr}")
        return p

    def frame(self, name="f.png", width=42, height=24):
        (self.work / name).write_bytes(png(width, height))
        return self.work / name


class Generate(CliTest):
    def test_wait_and_save(self):
        """9.2 #1: generate --wait --out saves the MP4; stdout is its absolute path and nothing else."""
        p = self.ok("generate", "--prompt", "a test clip", "--seconds", 3, "--wait", "--out", "c.mp4")
        out = self.work / "c.mp4"
        self.assertEqual(p.stdout, f"{out}\n")
        self.assertAlmostEqual(duration(out), 3.04, delta=0.1)
        self.assertRegex(p.stderr, r"queued [0-9a-f]{8} \(t2va, landscape, 3 s, seed \d+\), next in line")
        self.assertRegex(p.stderr, r"done in 0:0\d")
        self.assertEqual([f.name for f in self.work.iterdir()], ["c.mp4"])     # no .part left behind

    def test_prompt_on_stdin_and_json(self):
        """9.2 #2: the prompt from stdin, a first frame makes fl2va, --json gives the record with path."""
        self.frame()
        p = self.ok("generate", "--first-frame", "f.png", "--prompt-file", "-", "--json", "--wait",
                    stdin="the prompt\nfrom stdin\n")
        rec = json.loads(p.stdout)
        self.assertEqual((rec["task"], rec["state"], rec["prompt"]), ("fl2va", "done", "the prompt\nfrom stdin"))
        self.assertTrue(Path(rec["path"]).is_file())
        self.assertEqual(Path(rec["path"]).parent, self.work)
        self.assertEqual(rec["bytes"], Path(rec["path"]).stat().st_size)
        self.assertEqual(rec["labels"], [{"label": "<Picture 1>", "zone": "first_frame", "index": 0, "name": "f.png"}])

    def test_queue_then_wait_into_a_directory(self):
        """9.2 #3: without --wait stdout is the id; wait --out <dir>/ saves under the clip's own name (and the
        poster next to it)."""
        p = self.ok("generate", "--prompt", "x", "--seed", 7)
        jid = p.stdout.strip()
        self.assertRegex(jid, r"^[0-9a-f]{32}$")
        p = self.ok("wait", jid, "--out", "clips/", "--poster")
        out = self.work / "clips" / f"lf-studio_t2va_5s_seed7_{jid[:6]}.mp4"
        self.assertEqual(p.stdout.strip(), str(out))
        self.assertTrue(out.is_file())
        self.assertEqual(out.with_suffix(".jpg").read_bytes()[:2], b"\xff\xd8")
        p = self.ok("fetch", jid, "--out", "again.mp4", "--json")             # an existing clip, from the page too
        self.assertEqual(json.loads(p.stdout)["path"], str(self.work / "again.mp4"))
        self.assertEqual((self.work / "again.mp4").read_bytes(), out.read_bytes())

    def test_unwritable_place(self):
        """generate --wait checks that it can save the clip before it submits; a save that fails all the same says
        the clip is done and how to fetch it."""
        if os.geteuid() == 0:
            self.skipTest("root can write anywhere")
        ro = self.work / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        self.addCleanup(lambda: ro.exists() and ro.chmod(0o755))
        for args, cwd in ((["--wait"], ro), (["--out", "ro/new/c.mp4"], None), (["--out", "ro"], None)):
            p = self.sol("generate", "--prompt", "x", *args, cwd=cwd)
            self.assertEqual(p.returncode, 2, (args, p.stderr))
            self.assertIn(f"can't save the clip there: no permission to write to {ro}; nothing was submitted",
                          p.stderr)
        self.assertEqual(self.studio.state()["jobs"], [])
        jid = self.ok("generate", "--prompt", "x").stdout.strip()
        self.studio.wait([jid], timeout=30)
        p = self.sol("fetch", jid, "--out", "ro/c.mp4")                       # fetch doesn't check first
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn(f"The clip is done on the studio: save it with lf-cli fetch {jid} --out <a directory you can "
                      "write to>", p.stderr)

    def test_explicit_out_is_replaced(self):
        """A person typed --out: an existing file there is replaced."""
        (self.work / "c.mp4").write_bytes(b"old")
        self.ok("generate", "--prompt", "x", "--out", "c.mp4")
        self.assertEqual((self.work / "c.mp4").read_bytes()[4:8], b"ftyp")

    def test_failure(self):
        """9.2 #5: a failed clip is exit 1, with the studio's error on stderr."""
        p = self.sol("generate", "--prompt", "this one will STUB_FAIL", "--wait")
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("failed: RuntimeError: Worker exited with code 1 (STUB_FAIL)", p.stderr)
        self.assertEqual(p.stdout, "")
        p = self.sol("generate", "--prompt", "STUB_FAIL again", "--wait", "--json")
        self.assertEqual(p.returncode, 1)
        answer = json.loads(p.stdout)
        self.assertEqual(answer["job"]["state"], "failed")
        self.assertIn("STUB_FAIL", answer["error"])

    def test_refused_here_or_by_the_studio(self):
        """9.2 #6: local checks are exit 2 before anything is sent; a refusal by the studio is exit 2 with its
        message."""
        (self.work / "a.gif").write_bytes(b"GIF89a")
        self.frame()
        for args, words in [
                (["--first-frame", "a.gif"], "unsupported file type .gif"),
                (["--first-frame", "missing.png"], "no such file: missing.png"),
                (["--seconds", 9], re.compile(r"invalid choice: '?9'?")),     # Python 3.14 quotes the value
                (["--task", "t2va", "--first-frame", "f.png"], "t2va takes no files"),
                (["--first-frame", "f.png", "--ref", "f.png"], "can't go together"),
                (["--task", "ref2va", "--first-frame", "f.png"], "--first-frame is used only by fl2va"),
                (["--ref", "x.wav"], "no such file"),
                (["--task", "ref2va"], "ref2va needs at least one reference image or video")]:
            p = self.sol("generate", "--prompt", "x", *args)
            self.assertEqual(p.returncode, 2, (args, p.stderr))
            if isinstance(words, re.Pattern):
                self.assertRegex(p.stderr, words)
            else:
                self.assertIn(words, p.stderr)
        (self.work / "x.wav").write_bytes(WAV)
        p = self.sol("generate", "--prompt", "x", "--ref", "x.wav")
        self.assertEqual(p.returncode, 2)
        self.assertIn("ref2va needs at least one reference image or video", p.stderr)
        p = self.sol("generate", "--prompt", os.fsdecode(b"x \xff"))              # argv bytes that are not UTF-8
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("the prompt is not valid text at character 3", p.stderr)
        self.assertEqual(self.studio.state()["jobs"], [])                        # none of these reached the studio
        p = self.sol("generate", "--prompt", "x", "--seed", -1)
        self.assertEqual(p.returncode, 2)
        self.assertIn("seed must be a nonnegative 63-bit integer", p.stderr)
        p = self.sol("generate", "--prompt", "x", "--seed", -1, "--json")
        self.assertEqual(json.loads(p.stdout)["http_status"], 400)

    def test_ref2va_numbering(self):
        """References go to ref_images, ref_videos and ref_audio by extension, in order; the record's labels show
        the numbering (a video with sound also takes an <Audio N> before its <Video N>)."""
        self.frame("face.png")
        (self.work / "walk.mp4").write_bytes(mp4())
        (self.work / "voice.wav").write_bytes(WAV)
        p = self.ok("generate", "--prompt", "x", "--ref", "voice.wav", "--ref", "walk.mp4", "--ref", "face.png",
                    "--wait", "--json")
        rec = json.loads(p.stdout)
        self.assertEqual(rec["task"], "ref2va")
        self.assertEqual([(x["label"], x["zone"]) for x in rec["labels"]],
                         [("<Picture 1>", "ref_images"), ("<Audio 1>", "ref_videos"), ("<Video 1>", "ref_videos"),
                          ("<Audio 2>", "ref_audio")])

    def test_keyframe_picks_the_orientation(self):
        """DECISIONS 12: without --orientation, a portrait keyframe makes a portrait clip; a frame of another shape
        than the clip gets a warning."""
        self.frame("tall.png", 24, 42)
        p = self.ok("generate", "--prompt", "x", "--first-frame", "tall.png", "--json")
        self.assertEqual(json.loads(p.stdout)["orientation"], "portrait")
        self.assertIn("orientation: portrait, from the shape of --first-frame (24x42)", p.stderr)
        self.assertNotIn("warning", p.stderr)
        p = self.ok("generate", "--prompt", "x", "--last-frame", "tall.png", "--orientation", "landscape", "--json")
        self.assertEqual(json.loads(p.stdout)["orientation"], "landscape")
        self.assertIn("warning: --last-frame is 24x42, but a landscape clip is 1344x768", p.stderr)
        self.frame("square.png", 30, 30)
        p = self.ok("generate", "--prompt", "x", "--first-frame", "square.png", "--json", "--quiet")
        self.assertEqual(json.loads(p.stdout)["orientation"], "landscape")
        self.assertIn("warning: --first-frame is 30x30", p.stderr)                # warnings even with --quiet

    def test_status(self):
        """status with an id: one line; without: the queue and the newest clips."""
        jid = self.ok("generate", "--prompt", "x", "--seed", 3, "--wait", "--json").stdout
        jid = json.loads(jid)["id"]
        p = self.ok("status", jid)
        self.assertRegex(p.stdout, rf"^{jid}  done       t2va, landscape, 5 s, seed 3  done in 0:0\d\n$")
        rec = json.loads(self.ok("status", jid, "--json").stdout)
        self.assertEqual((rec["state"], rec["urls"]["video"]), ("done", f"/video/{jid}"))
        listing = json.loads(self.ok("status", "--json").stdout)["jobs"]
        self.assertEqual([j["id"] for j in listing], [jid])
        p = self.sol("status", "f" * 32)
        self.assertEqual(p.returncode, 2)
        self.assertIn("no clip with id", p.stderr)
        p = self.sol("status", "abc")
        self.assertEqual(p.returncode, 2)
        self.assertIn("not a clip id", p.stderr)

    def test_url_files(self):
        """DECISIONS 16: the client downloads URL files itself; the type comes from the path's extension when it is
        a known one, else from Content-Type, else it refuses."""
        frame = png(42, 24)
        files = {"/pics/frame.png": ("image/png", frame), "/noext": ("image/jpg", b"\xff\xd8\xff\xd9"),
                 "/blob": ("application/octet-stream", frame), "/clip.png?x=1": ("text/plain", frame)}

        def answer(method, path, headers, body):
            if path in files:
                return 200, {"Content-Type": files[path][0]}, files[path][1]
            return 404, JSON, {"error": "not found"}
        fake = Fake(answer)
        self.addCleanup(fake.stop)
        p = self.ok("generate", "--prompt", "x", "--first-frame", fake.url + "/pics/frame.png", "--last-frame",
                    fake.url + "/noext", "--json")
        rec = json.loads(p.stdout)
        self.assertEqual([x["name"] for x in rec["labels"]], ["frame.png", "noext.jpg"])
        p = self.ok("generate", "--prompt", "x", "--first-frame", fake.url + "/clip.png?x=1", "--json")
        self.assertEqual(json.loads(p.stdout)["labels"][0]["name"], "clip.png")
        p = self.sol("generate", "--prompt", "x", "--first-frame", fake.url + "/blob")
        self.assertEqual(p.returncode, 2)
        self.assertIn("can't tell what kind of file", p.stderr)
        p = self.sol("generate", "--prompt", "x", "--first-frame", fake.url + "/gone.png")
        self.assertEqual(p.returncode, 2)
        self.assertIn("could not download", p.stderr)
        self.assertIn("HTTP 404", p.stderr)
        self.assertEqual(len(self.studio.state()["jobs"]), 2)

    def test_studio_address(self):
        """9.2 #8: LFSTUDIO_HOST=0.0.0.0 and LFSTUDIO_PORT find a local studio; LFSTUDIO_URL beats them; --url beats
        both, before or after the command."""
        dead = f"http://127.0.0.1:{free_port({self.studio.port})}"
        port = str(self.studio.port)
        env = {k: v for k, v in client_env().items()}
        for extra, args in [({"LFSTUDIO_HOST": "0.0.0.0", "LFSTUDIO_PORT": port}, ["status"]),
                            ({"LFSTUDIO_HOST": "", "LFSTUDIO_PORT": port}, ["status"]),
                            ({"LFSTUDIO_URL": self.studio.url, "LFSTUDIO_HOST": "127.0.0.1", "LFSTUDIO_PORT": "1"},
                             ["status"]),
                            ({"LFSTUDIO_URL": dead}, ["--url", self.studio.url, "status"]),
                            ({"LFSTUDIO_URL": dead}, ["status", "--url", self.studio.url + "/"])]:
            p = subprocess.run([sys.executable, str(BIN), *args], capture_output=True, text=True, timeout=30,
                               env=dict(env, **extra))
            self.assertEqual(p.returncode, 0, (extra, args, p.stderr))
            self.assertEqual(p.stdout, "no clips yet\n")
        p = subprocess.run([sys.executable, str(BIN), "status"], capture_output=True, text=True, timeout=30,
                           env=dict(env, LFSTUDIO_URL=dead))
        self.assertEqual(p.returncode, 3)

    def test_unreachable(self):
        """9.2 #7: nothing listening: exit 3, and the message names the address."""
        dead = f"http://127.0.0.1:{free_port({self.studio.port})}"
        for args in (["generate", "--prompt", "x"], ["status"], ["wait", "0" * 32], ["fetch", "0" * 32]):
            p = self.sol(*args, url=dead)
            self.assertEqual(p.returncode, 3, (args, p.stderr))
            self.assertIn(f"can't reach the studio at {dead} (connection refused)", p.stderr)
        p = self.sol("status", url="ftp://somewhere")
        self.assertEqual(p.returncode, 2)


class ApiOff(CliTest):
    """A studio with LFSTUDIO_API=0 (#58): generate (refused before anything is queued: the form carries
    X-LFStudio-Client), wait, status ID and fetch say so, naming the setting (exit 2); status reads the page's
    /api/state and still works."""
    studio_env = {"LFSTUDIO_API": "0"}

    def test_refused_with_the_setting_named(self):
        off = load_cli().API_OFF
        self.assertIn("LFSTUDIO_API=0", off)
        self.assertIn("LFSTUDIO_API=1", off)
        self.frame()
        for args in (["--prompt", "x"], ["--prompt", "x", "--wait"], ["--first-frame", "f.png", "--prompt", "x"]):
            p = self.sol("generate", *args)
            self.assertEqual((p.returncode, p.stdout), (2, ""), p.stderr)
            self.assertTrue(p.stderr.endswith(f"\nlf-cli: {off}\n" if "f.png" in args else f"lf-cli: {off}\n"),
                            p.stderr)
        self.assertEqual(self.studio.state()["jobs"], [])                         # nothing was queued
        p = self.sol("generate", "--prompt", "x", "--json")
        self.assertEqual(json.loads(p.stdout), {"error": off, "http_status": 403, "id": None})
        jid = self.studio.generate()[1]["id"]                                     # a clip made in the page
        self.studio.wait([jid], timeout=30)
        for args in (["wait", jid], ["wait", jid, "--timeout", 5], ["status", jid], ["fetch", jid],
                     ["fetch", jid, "--out", "c.mp4"]):
            p = self.sol(*args)
            self.assertEqual((p.returncode, p.stdout), (2, ""), (args, p.stderr))
            self.assertEqual(p.stderr, f"lf-cli: {off}\n", args)
        self.assertEqual(list(self.work.iterdir()), [self.work / "f.png"])       # nothing saved
        p = self.sol("--json", "status", jid)
        self.assertEqual(json.loads(p.stdout), {"error": off, "http_status": 403, "id": jid})
        p = self.ok("status")
        self.assertTrue(p.stdout.startswith(f"{jid}  done"), p.stdout)


class Waiting(CliTest):
    """Clips that take a while (the models load for 6 s)."""
    load_s, clip_s = 6, 0.5

    def test_timeout_cancel_and_wait(self):
        """9.2 #4: --timeout runs out: exit 4 with the id on stdout; cancel; wait then reports it cancelled (exit 1)."""
        t = time.time()
        p = self.sol("generate", "--prompt", "x", "--wait", "--timeout", 1)
        self.assertEqual(p.returncode, 4, p.stderr)
        self.assertLess(time.time() - t, 5)
        jid = p.stdout.strip()
        self.assertRegex(jid, r"^[0-9a-f]{32}$")
        self.assertIn(f"Keep waiting with: lf-cli wait {jid}", p.stderr)
        p = self.ok("cancel", jid)
        self.assertEqual(p.stdout, f"cancelling {jid}: it stops once the pipeline has closed (about 40 s)\n")
        p = self.sol("wait", jid)
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"clip {jid} was cancelled", p.stderr)
        p = self.sol("cancel", jid)
        self.assertEqual(p.returncode, 2)
        self.assertIn("the clip is already cancelled", p.stderr)

    def test_cancel_queued(self):
        self.studio.generate(prompt="first")                                     # keeps the pipeline busy
        jid = self.ok("generate", "--prompt", "second").stdout.strip()
        self.assertEqual(self.ok("cancel", jid).stdout, f"cancelled {jid}\n")
        self.assertEqual(json.loads(self.ok("status", jid, "--json").stdout)["state"], "cancelled")

    def test_interrupt_leaves_the_clip_running(self):
        """Ctrl-C while waiting: exit 130, a hint how to pick it up again, and the clip still finishes."""
        proc = subprocess.Popen([sys.executable, str(BIN), "generate", "--prompt", "x", "--wait"], cwd=self.work,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=client_env(self.studio.url))
        deadline = time.time() + 10
        while time.time() < deadline and not self.studio.state()["jobs"]:
            time.sleep(0.1)
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 130, err)
        jid = self.studio.state()["jobs"][0]["id"]
        self.assertIn(f"Get it later with: lf-cli wait {jid}", err)
        self.assertEqual(self.studio.wait([jid], timeout=30)[0]["status"], "done")
        self.assertEqual(list(self.work.iterdir()), [])


class Switchboard(unittest.TestCase):
    """9.2 #9 and DECISIONS 13, against fake servers that answer as spark-switchboard does."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-cli-"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def fake(self, answer):
        fake = Fake(answer)
        self.addCleanup(fake.stop)
        return fake

    def sol(self, fake, *args):
        return subprocess.run([sys.executable, str(BIN), *map(str, args)], capture_output=True, text=True,
                              timeout=60, cwd=self.dir, env=client_env(fake.url))

    def test_stopped_studio(self):
        """The studio is stopped: status, wait, cancel and fetch don't start it (exit 3, fetch asks for --wake);
        generate is held (X-Switchboard-Wait: 900) and switchboard's 503 is exit 3 with its message."""
        jid = "0123456789abcdef0123456789abcdef"

        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "stopped", "paused": None, "message": "LF Studio is not running"}
            if path.startswith("/video/"):
                return 200, {"Content-Type": "video/mp4", "Content-Disposition": 'inline; filename="woken.mp4"'}, \
                    b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
            if path == "/api/jobs/" + jid:
                return 200, JSON, {"id": jid, "state": "done", "task": "t2va", "seconds": 5, "seed": 1}
            if path == "/api/generate":
                return switchboard_error("waiting", "LF Studio will start once memory is free: TensorFold is "
                                                    "busy", retry=120)
            return switchboard_error("stopped", "LF Studio is not running")
        fake = self.fake(answer)
        for args, words in [(["status"], "the studio is stopped"), (["status", jid], "the studio is stopped"),
                            (["wait", jid], f"Get it with: lf-cli fetch {jid} --wake"),
                            (["cancel", jid], "no clip is queued or running there"),
                            (["fetch", jid],
                             "fetching would start it (it needs most of the machine's memory). Add --wake.")]:
            p = self.sol(fake, *args)
            self.assertEqual(p.returncode, 3, (args, p.stderr))
            self.assertIn(words, p.stderr)
        self.assertEqual(set(fake.paths()), {"/__switchboard/status"})            # nothing else was asked for
        p = self.sol(fake, "generate", "--prompt", "x")
        self.assertEqual(p.returncode, 3)
        self.assertIn("LF Studio will start once memory is free: TensorFold is busy The clip was NOT submitted; "
                      "try again in about 120 s.", p.stderr)
        method, path, headers, body = fake.requests[-1]
        self.assertEqual((method, path, headers["X-Switchboard-Wait"]), ("POST", "/api/generate", "900"))
        self.assertEqual(headers["Accept"], "application/json")
        self.assertTrue(headers["User-Agent"].startswith("lf-cli/0.5.0"))
        self.assertEqual(headers["X-LFStudio-Client"], headers["User-Agent"])    # #58: a studio with the API off
        self.assertNotIn("Authorization", headers)                               # DECISIONS 2: no credentials
        p = self.sol(fake, "fetch", jid, "--wake")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), str(self.dir / "woken.mp4"))
        method, path, headers, body = [r for r in fake.requests if r[1].startswith("/video/")][0]
        self.assertEqual((path, headers["X-Switchboard-Wait"]), (f"/video/{jid}", "900"))

    def test_stopped_while_waiting(self):
        """DECISIONS 13: switchboard stops the studio during a wait (most likely finished: fetch --wake), versus the
        studio's own shutdown 503 with the record (it was still running: failed after a restart)."""
        jid = "0123456789abcdef0123456789abcdef"
        running = {"id": jid, "state": "running", "phase": "generating", "task": "t2va", "seconds": 5, "seed": 1}
        polls = []

        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "ready", "paused": None}
            polls.append(path)
            if len(polls) == 1:
                return 202, JSON, running
            return switchboard_error("stopped", "LF Studio is not running")
        p = self.sol(self.fake(answer), "wait", jid)
        self.assertEqual(p.returncode, 3)
        self.assertIn(f"the clip most likely finished first. Get it with: lf-cli fetch {jid} --wake", p.stderr)
        self.assertTrue(all(p.startswith(f"/api/jobs/{jid}/video?wait=") for p in polls))

        def shutdown(method, path, headers, body):
            if path == "/__switchboard/status":
                return 404, JSON, {"error": "not found"}                        # no switchboard
            return 503, JSON, {"error": "the studio is shutting down; ask again once it is back", "job": running}
        p = self.sol(self.fake(shutdown), "wait", jid, "--json")
        self.assertEqual(p.returncode, 3)
        self.assertIn(f"the studio stopped while clip {jid} was generating; after a restart it shows as failed",
                      p.stderr)
        self.assertEqual(json.loads(p.stdout)["job"]["phase"], "generating")

    def test_upscale_while_stopped(self):
        """An upscale is held like a submit (X-Switchboard-Wait, --hold), with only the choices given; switchboard's
        503 means nothing was queued."""
        jid = "0123456789abcdef0123456789abcdef"

        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "stopped", "paused": None, "message": "LF Studio is not running"}
            return switchboard_error("waiting", "LF Studio will start once memory is free", retry=120)
        fake = self.fake(answer)
        p = self.sol(fake, "upscale", jid, "--resolution", "4k", "--hold", 30)
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertIn("LF Studio will start once memory is free The upscale was NOT queued; try again in about 120 s.",
                      p.stderr)
        method, path, headers, body = fake.requests[-1]
        self.assertEqual((method, path, headers["X-Switchboard-Wait"], headers["Content-Type"]),
                         ("POST", "/api/upscale", "30", "application/json"))
        self.assertEqual(json.loads(body), {"id": jid, "resolution": "4k"})

    def test_loading_is_retried(self):
        """A switchboard 503 "loading" while waiting is retried (5 s later), then the MP4 arrives."""
        jid = "0123456789abcdef0123456789abcdef"
        polls = []

        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "starting", "paused": None}
            if path.startswith(f"/api/jobs/{jid}/video"):
                polls.append(time.time())
                if len(polls) == 1:
                    return switchboard_error("loading", "LF Studio is starting", retry=10)
                return 200, {"Content-Type": "video/mp4"}, b"\x00\x00\x00\x18ftypmp42"
            return 200, JSON, {"id": jid, "state": "done", "task": "t2va", "seconds": 5, "seed": 1, "wall_s": 61}
        p = self.sol(self.fake(answer), "wait", jid, "--out", "x.mp4")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertGreaterEqual(polls[1] - polls[0], 4.5)
        self.assertEqual((self.dir / "x.mp4").read_bytes(), b"\x00\x00\x00\x18ftypmp42")
        self.assertIn("done in 1:01", p.stderr)


class Upscale(CliTest):
    """lf-cli upscale: the experimental Upscale (the stub studio has it on, with a stand-in job of about 1 s)."""

    def clip(self, **kw):
        jid = self.studio.generate(**kw)[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        return jid

    def test_wait_and_save(self):
        """--wait --out <dir>/ saves it under the studio's name for it: the clip's, with the resolution and the
        upscale's own id after it."""
        jid = self.clip(seed=7)
        p = self.ok("upscale", jid, "--resolution", "1080p", "--strength", "faithful", "--source", "draft",
                    "--audio", "silence", "--seed", 99, "--out", "up/", "--poster")
        up = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        out = self.work / "up" / f"lf-studio_t2va_5s_seed7_{jid[:6]}_1080p_{up['id'][:6]}.mp4"
        self.assertEqual(p.stdout, f"{out}\n")
        self.assertEqual(out.read_bytes()[4:8], b"ftyp")
        self.assertEqual(out.with_suffix(".jpg").read_bytes()[:2], b"\xff\xd8")
        self.assertRegex(p.stderr, rf"queued [0-9a-f]{{8}} \(1080p upscale of {jid[:8]}, faithful, t2va, 5 s, seed 7\), "
                                   r"next in line")
        job = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual({k: job[k] for k in ("parent", "resolution", "strength", "source", "audio", "refine_seed",
                                              "status")},
                         {"parent": jid, "resolution": "1080p", "strength": "faithful", "source": "draft",
                          "audio": "silence", "refine_seed": 99, "status": "done"})
        p = self.ok("upscale", jid, "--json")                                  # the studio's defaults; no wait
        rec = json.loads(p.stdout)
        self.assertEqual((rec["kind"], rec["parent"], rec["state"], rec["resolution"], rec["strength"], rec["audio"]),
                         ("upscale", jid, "queued", "2k", "balanced", "clip"))
        p = self.ok("wait", rec["id"], "--json")
        self.assertEqual(Path(json.loads(p.stdout)["path"]).name,
                         f"lf-studio_t2va_5s_seed7_{jid[:6]}_2k_{rec['id'][:6]}.mp4")
        lines = self.ok("status").stdout.splitlines()
        self.assertTrue(lines[0].startswith(f"{rec['id']}  done       2k upscale of {jid[:8]}, balanced"), lines)

    def test_refusals(self):
        """The studio's refusals are exit 2 with its words; argparse checks the choices."""
        ten = self.clip(seconds=10)
        p = self.sol("upscale", ten, "--json")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(json.loads(p.stdout), {"error": "Upscale needs a 5 s 16:9 clip (NVIDIA's upscaler takes "
                                                         "1344x768, 121 frames)", "http_status": 400, "id": ten})
        p = self.sol("upscale", "f" * 32)
        self.assertEqual((p.returncode, p.stderr), (2, "lf-cli: no such clip\n"))
        for args in (["upscale", ten, "--resolution", "8k"], ["upscale", ten, "--strength", "nvidia"],
                     ["upscale", "nonsense"], ["upscale", ten, "--seed", -1], ["upscale", ten, "--hold", 9999]):
            p = self.sol(*args)
            self.assertEqual(p.returncode, 2, (args, p.stderr))
        self.assertEqual([j.get("kind") for j in self.studio.state()["jobs"]], [None])     # nothing was queued
        if os.geteuid() != 0:
            ro = self.work / "ro"
            ro.mkdir()
            ro.chmod(0o555)
            self.addCleanup(lambda: ro.exists() and ro.chmod(0o755))
            p = self.sol("upscale", self.clip(seed=3), "--out", "ro/")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(f"can't save the upscale there: no permission to write to {ro}; nothing was queued",
                          p.stderr)


class UpscaleRunning(CliTest):
    """An upscale that takes a while (its stand-in job: 30 s): --timeout, status, cancel and wait."""
    studio_env = {"STUB_UPSCALE_S": "30"}

    def test_timeout_status_cancel(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid], timeout=60)
        p = self.sol("upscale", jid, "--resolution", "4k", "--wait", "--timeout", 3)
        self.assertEqual(p.returncode, 4, p.stderr)
        uid = p.stdout.strip()
        self.assertRegex(uid, r"^[0-9a-f]{32}$")
        self.assertIn(f"Keep waiting with: lf-cli wait {uid}", p.stderr)
        self.assertRegex(p.stderr, r"\n(closing the studio's models for the upscale|upscaling(: [a-z /0-9]+)?)\n")
        rec = json.loads(self.ok("status", uid, "--json").stdout)
        self.assertEqual((rec["state"], rec["phase"]), ("running", "upscaling"))
        self.assertRegex(self.ok("status", uid).stdout, rf"^{uid}  running    4k upscale of {jid[:8]}, balanced, t2va, "
                                                        r"5 s, seed \d+  upscaling: [a-z /0-9]+, 0:\d\d since it started\n$")
        self.assertEqual(self.ok("cancel", uid).stdout,
                         f"cancelling upscale {uid}: its job stops within about 20 s\n")
        p = self.sol("wait", uid)
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"upscale {uid} was cancelled", p.stderr)


class UpscaleOff(CliTest):
    studio_env = {"LFSTUDIO_UPSCALE": "0"}

    def test_off(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid], timeout=60)
        p = self.sol("upscale", jid)
        self.assertEqual((p.returncode, p.stdout), (2, ""), p.stderr)
        self.assertEqual(p.stderr, "lf-cli: the experimental Upscale is off on this studio (LFSTUDIO_UPSCALE=1 turns "
                                   "it on)\n")


class Guide(unittest.TestCase):
    def test_guide(self):
        """9.2 #10: guide prints the prompt guide; docs/API.md carries the same text between its markers."""
        cli = load_cli()
        p = subprocess.run([sys.executable, str(BIN), "guide"], capture_output=True, timeout=30,
                           env=client_env())
        self.assertEqual(p.returncode, 0)
        self.assertEqual(p.stdout.decode("utf-8"), cli.GUIDE)
        self.assertIn("**4. fl2va: the keyframe line comes first**", cli.GUIDE)
        self.assertIn("stretched or cropped to fit", cli.GUIDE)                  # DECISIONS 19
        text = (REPO / "docs" / "API.md").read_text()
        m = re.search(r"<!-- prompt-guide:start -->(.*)<!-- prompt-guide:end -->", text, re.S)
        self.assertIsNotNone(m, "docs/API.md has no prompt-guide markers")
        self.assertEqual(m.group(1).strip(), cli.GUIDE.strip())

    def test_examples_follow_the_rules(self):
        """The guide's examples keep its own rules, since agents copy them: the ref2va example's
        detailed_description runs 350-500 words (MiniMax: a single shot does not justify a shorter one); the static
        shots of the examples take no amplitude or speed, as the rule now says."""
        guide = load_cli().GUIDE
        ref2va = guide[guide.index("\nref2va, one portrait image"):]
        words = len(ref2va[ref2va.index("detailed_description:"):ref2va.index("overall_soundscape:")].split()) - 1
        self.assertTrue(350 <= words <= 500, words)
        self.assertIn("A static shot takes neither", guide)


class SwitchboardIntegration(unittest.TestCase):
    def test_passive_paths(self):
        """The integration file's passive_paths (switchboard matches them with fnmatch on the path): the reads the
        client polls never start a stopped studio (DECISIONS 10), downloads and POSTs still do."""
        import fnmatch
        text = (REPO / "integrations" / "spark-switchboard" / "lf-studio.toml").read_text()
        m = re.search(r"^passive_paths = (\[.*\])", text, re.M)
        self.assertIsNotNone(m)
        patterns = json.loads(m.group(1))
        jid = "0123456789abcdef0123456789abcdef"
        for path in ("/health", "/api/state", "/api/jobs", f"/api/jobs/{jid}", f"/api/jobs/{jid}/video",
                     f"/poster/{jid}"):
            self.assertTrue(any(fnmatch.fnmatch(path, p) for p in patterns), path)
        for path in (f"/video/{jid}", f"/input/{jid}/first_frame/0", f"/api/inputs/{jid}", f"/still/{jid}",
                     "/api/generate", "/api/cancel", "/api/upscale", "/"):
            self.assertFalse(any(fnmatch.fnmatch(path, p) for p in patterns), path)


class PassThrough(CliTest):
    def test_launcher_runs_the_client(self):
        """9.2 #11: bin/lf-studio status|guide|... runs bin/lf-cli with the env file's settings."""
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid])
        envfile = self.dir / "env"
        envfile.write_text(f"LFSTUDIO_HOST=0.0.0.0\nLFSTUDIO_PORT={self.studio.port}\n")
        p = subprocess.run(["bash", str(REPO / "bin" / "lf-studio"), "status", jid], capture_output=True,
                           text=True, timeout=30, cwd=self.work, env=client_env(LFSTUDIO_ENV=str(envfile)))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(p.stdout.startswith(f"{jid}  done"), p.stdout)
        p = subprocess.run(["bash", str(REPO / "bin" / "lf-studio"), "nonsense"], capture_output=True, text=True,
                           timeout=30, env=client_env(LFSTUDIO_ENV=str(envfile)))
        self.assertEqual(p.returncode, 2)


class Trickle:
    """A server that sends an image's headers, then 100 bytes a second (for up to 30 s), with no Content-Length, or
    chunked."""

    def __init__(self, chunked=False):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", free_port()))
        self.sock.listen(4)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}/frame.png"
        self.chunked, self.closed_early = chunked, threading.Event()
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            conn.recv(65536)
            head = "Transfer-Encoding: chunked\r\n" if self.chunked else "Connection: close\r\n"
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n{head}\r\n".encode())
            for _ in range(30):
                piece = b"x" * 100
                try:
                    conn.sendall(b"64\r\n" + piece + b"\r\n" if self.chunked else piece)
                except OSError:
                    self.closed_early.set()          # the client gave up
                    return
                time.sleep(1)

    def stop(self):
        self.sock.close()


class Downloads(unittest.TestCase):
    """An input given as a URL takes at most DOWNLOAD_S (120 s), also from a server that trickles."""

    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def test_trickle_stops_at_the_deadline(self):
        self.cli.DOWNLOAD_S = 3
        self.addCleanup(setattr, self.cli, "DOWNLOAD_S", 120)
        for chunked in (False, True):
            server = Trickle(chunked)
            self.addCleanup(server.stop)
            t = time.time()
            with self.assertRaises(self.cli.Fail) as caught:
                self.cli.download_input(server.url, "--first-frame", 10 ** 9)
            self.assertLess(time.time() - t, 6, chunked)
            self.assertEqual(caught.exception.message, f"--first-frame: downloading {server.url} took longer than 3 s")

    def test_check_stops_a_download(self):
        """The MCP server passes its call's check: a cancelled call stops downloading at once."""
        class Cancelled(Exception):
            pass

        def check():
            if time.time() - t > 1:
                raise Cancelled()
        server = Trickle()
        self.addCleanup(server.stop)
        t = time.time()
        with self.assertRaises(Cancelled):
            self.cli.download_input(server.url, "first_frame", 10 ** 9, check)
        self.assertLess(time.time() - t, 3)
        self.assertTrue(server.closed_early.wait(5))


class ImageSize(unittest.TestCase):
    """DECISIONS 12: the keyframe's size from its header (PNG, JPEG with EXIF rotation, WebP)."""

    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    @staticmethod
    def tiff(orientation, order="II"):
        """EXIF's TIFF data with a Make tag, then Orientation."""
        e = "<" if order == "II" else ">"
        tiff = order.encode() + struct.pack(e + "HI", 42, 8) + struct.pack(e + "H", 2)
        tiff += struct.pack(e + "HHI", 0x010F, 2, 1) + b"\x00" * 4                     # Make, before Orientation
        return tiff + struct.pack(e + "HHIH", 0x0112, 3, 1, orientation) + b"\x00\x00" + struct.pack(e + "I", 0)

    def jpeg(self, width, height, orientation=None, order="II", sof=0xC0):
        exif = b""
        if orientation:
            tiff = self.tiff(orientation, order)
            exif = b"\xff\xe1" + struct.pack(">H", len(tiff) + 8) + b"Exif\x00\x00" + tiff
        jfif = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        frame = bytes([0xFF, sof]) + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
        return b"\xff\xd8" + jfif + exif + frame + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x00" * 10

    def test_formats(self):
        size = self.cli.image_size
        self.assertEqual(size(png(42, 24)), (42, 24))
        self.assertEqual(size(self.jpeg(1344, 768)), (1344, 768))
        self.assertEqual(size(self.jpeg(1344, 768, sof=0xC2)), (1344, 768))         # progressive
        for orientation, shown in ((1, (1344, 768)), (3, (1344, 768)), (6, (768, 1344)), (8, (768, 1344))):
            for order in ("II", "MM"):
                self.assertEqual(size(self.jpeg(1344, 768, orientation, order)), shown, (orientation, order))
        vp8 = b"RIFF\x00\x00\x00\x00WEBPVP8 \x00\x00\x00\x00" + b"\x10\x02\x00\x9d\x01\x2a" + struct.pack("<HH",
                                                                                                         640, 360)
        self.assertEqual(size(vp8), (640, 360))
        bits = (639) | (359 << 14)
        vp8l = b"RIFF\x00\x00\x00\x00WEBPVP8L\x00\x00\x00\x00\x2f" + struct.pack("<I", bits)
        self.assertEqual(size(vp8l), (640, 360))
        vp8x = b"RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x00\x00\x00\x00" + (639).to_bytes(3, "little") + \
            (359).to_bytes(3, "little")
        self.assertEqual(size(vp8x), (640, 360))
        for junk in (b"", b"GIF89a", b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF\x00\x00\x00\x00WEBPVP8 "):
            self.assertIsNone(size(junk))

    def test_exif_in_png_and_webp(self):
        """The pipeline applies EXIF rotation to every format (PIL's exif_transpose), so a PNG's eXIf chunk and an
        extended WebP's EXIF chunk count too. libwebp reads that chunk only when the VP8X EXIF flag is set."""
        size = self.cli.image_size

        def png_chunk(name, payload):
            return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))
        plain = png(1344, 768)
        end = plain.rindex(b"IEND") - 4
        for orientation, shown in ((1, (1344, 768)), (3, (1344, 768)), (6, (768, 1344)), (8, (768, 1344))):
            for order in ("II", "MM"):
                tagged = plain[:end] + png_chunk(b"eXIf", self.tiff(orientation, order)) + plain[end:]
                self.assertEqual(size(tagged), shown, ("png", orientation, order))

        def webp(orientation, flag=True, header=b""):
            vp8x = b"VP8X" + struct.pack("<I", 10) + bytes([0x08 if flag else 0]) + b"\x00" * 3 + \
                (1343).to_bytes(3, "little") + (767).to_bytes(3, "little")
            payload = header + self.tiff(orientation)
            exif = b"EXIF" + struct.pack("<I", len(payload)) + payload + b"\x00" * (len(payload) & 1)
            vp8 = b"VP8 " + struct.pack("<I", 10) + b"\x10\x02\x00\x9d\x01\x2a" + struct.pack("<HH", 1344, 768)
            body = b"WEBP" + vp8x + vp8 + exif
            return b"RIFF" + struct.pack("<I", len(body)) + body
        self.assertEqual(size(webp(6)), (768, 1344))
        self.assertEqual(size(webp(8, header=b"Exif\x00\x00")), (768, 1344))
        self.assertEqual(size(webp(3)), (1344, 768))
        self.assertEqual(size(webp(6, flag=False)), (1344, 768))

    def test_files_from_ffmpeg(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-img-"))
        self.addCleanup(shutil.rmtree, d, True)
        for name, extra in (("a.png", []), ("a.jpg", []), ("a.webp", []), ("b.webp", ["-lossless", "1"])):
            p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=40x24",
                                "-frames:v", "1", *extra, str(d / name)], capture_output=True)
            if p.returncode != 0:
                continue                         # an ffmpeg without that encoder (libwebp)
            self.assertEqual(self.cli.image_size((d / name).read_bytes()), (40, 24), name)


class Source(unittest.TestCase):
    """9.2 #12: bin/lf-cli runs on Python 3.9 with the standard library alone, as a lone copied file."""

    def test_python39_and_stdlib_only(self):
        source = BIN.read_text()
        tree = ast.parse(source, feature_version=(3, 9))
        modules = set()
        for node in ast.walk(tree):
            self.assertNotIsInstance(node, getattr(ast, "Match", ()), "no match statements (3.10+)")
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                modules.add(node.module.split(".")[0])
            elif isinstance(node, ast.arg):
                self.assertIsNone(node.annotation, "no annotations")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertIsNone(node.returns, "no annotations")
            self.assertNotIsInstance(node, ast.AnnAssign, "no annotations")
        self.assertEqual(modules - set(sys.stdlib_module_names), set())
        # standard library calls that need 3.10 or later
        for pattern in (r"strict=True", r"\bpairwise\b", r"\.bit_count\(", r"\banext\(", r"\baiter\(",
                        r"file_digest", r"\btomllib\b", r"datetime\.UTC", r"ExceptionGroup", r"except\*",
                        r"\bTaskGroup\b", r"root_dir=", r"\.walk\(", r"kw_only", r"slots=True"):
            self.assertIsNone(re.search(pattern, source), pattern)
        self.assertTrue(source.startswith("#!/usr/bin/env python3\n"))
        self.assertTrue(os.access(BIN, os.X_OK))
        self.assertNotIn("/home/", source)

    def test_no_python312_fstrings(self):
        """ast.parse(feature_version=(3, 9)) lets 3.12 f-strings through: inside an f-string's braces, 3.9 takes no
        string in the same quotes and no backslash."""
        if not hasattr(tokenize, "FSTRING_START"):
            self.skipTest("needs the Python 3.12 tokenizer")
        quotes = []                                      # the quote character of each f-string we are inside
        for tok in tokenize.generate_tokens(io.StringIO(BIN.read_text()).readline):
            if tok.type == tokenize.FSTRING_END:
                quotes.pop()
                continue
            if quotes and tok.type != tokenize.FSTRING_MIDDLE:
                self.assertNotIn("\\", tok.string, f"line {tok.start[0]}")
                if tok.type in (tokenize.STRING, tokenize.FSTRING_START):
                    self.assertNotIn(quotes[-1], tok.string.lstrip("rRbBuUfF")[:1], f"line {tok.start[0]}")
            if tok.type == tokenize.FSTRING_START:
                quotes.append(tok.string[-1])

    def test_runs_as_a_lone_copy(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-copy-"))
        self.addCleanup(shutil.rmtree, d, True)
        shutil.copy(BIN, d / "lf-cli")
        p = subprocess.run([sys.executable, "-I", str(d / "lf-cli"), "--version"], capture_output=True, text=True,
                           cwd=d, timeout=30, env=client_env())
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "lf-cli 0.5.0"))
        self.assertEqual(sorted(x.name for x in d.iterdir()), ["lf-cli"])


if __name__ == "__main__":
    unittest.main()
