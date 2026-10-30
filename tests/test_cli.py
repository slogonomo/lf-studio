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
    """This process's environment without LFSTUDIO_* settings (and SOLH3_*, which stop it), plus the given ones."""
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
    models = None                  # LFSTUDIO_MODELS (the stub studio's default: sol-h3 alone, the model choice off)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-cli-"))
        self.work = self.dir / "work"
        self.work.mkdir()
        self.studio = StubStudio(self.dir / "studio", port=free_port(), load_s=self.load_s, clip_s=self.clip_s,
                                 preload=False, env=self.studio_env, models=self.models).start()

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
        p = self.ok("generate", "--prompt", "a test clip", "--seconds", 6, "--wait", "--out", "c.mp4")
        out = self.work / "c.mp4"
        self.assertEqual(p.stdout, f"{out}\n")
        self.assertAlmostEqual(duration(out), 6.375, delta=0.1)                # 153 frames, as a 6 s clip has
        self.assertRegex(p.stderr, r"queued [0-9a-f]{8} \(t2va, landscape, 6 s, seed \d+\), next in line")
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
                (["--seconds", 3], re.compile(r"invalid choice: '?3'? \(choose from '?5'?, '?6'?, '?7'?, '?8'?, "
                                              r"'?10'?, '?12'?, '?15'?\)")),   # 3 and 4 s went after 0.5.0
                (["--seconds", 4], re.compile(r"invalid choice: '?4'?")),
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
        self.assertTrue(headers["User-Agent"].startswith("lf-cli/0.5.2"))
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

    def test_upscale_pin(self):
        """--pin goes only to a studio that takes it (/api/state "upscale_pin"): an older studio gets the request
        without it; a stopped one (switchboard answers /api/state for it) gets it, and again without it when the
        studio answers that it does not know the key; other refusals are not sent again."""
        jid, uid = "0123456789abcdef0123456789abcdef", "f" * 32
        for name, sb_state, state, answers, bodies in (
                ("takes it", "ready", {"upscale": True, "upscale_pin": True}, [None], [{"pin": "off"}]),
                ("older", "ready", {"upscale": True, "upscale_pin": None}, [None], [{}]),
                ("upscale off", "ready", {"upscale": False}, [None], [{}]),
                ("stopped, takes it", "stopped", None, [None], [{"pin": "off"}]),
                ("stopped, older", "stopped", None, ['unknown key "pin" (known: id, resolution, strength, audio, seed)',
                                                     None], [{"pin": "off"}, {}]),
                ("stopped, refused", "stopped", None, ["no such clip"], [{"pin": "off"}])):
            with self.subTest(name):
                posts = []

                def answer(method, path, headers, body):
                    if path == "/__switchboard/status":
                        return 200, JSON, {"state": sb_state, "paused": None}
                    if path == "/api/state":
                        return (200, JSON, state) if state else switchboard_error("stopped", "not running")
                    posts.append(json.loads(body))
                    error = answers[len(posts) - 1]
                    if error:
                        return 400, JSON, {"error": error}
                    return 200, JSON, {"id": uid, "job": {"id": uid, "kind": "upscale", "state": "queued"}}
                p = self.sol(self.fake(answer), "upscale", jid, "--pin", "off")
                self.assertEqual(p.returncode, 2 if name == "stopped, refused" else 0, p.stderr)
                self.assertEqual(posts, [dict({"id": jid}, **b) for b in bodies])
                if name == "stopped, refused":
                    self.assertIn("no such clip", p.stderr)
                else:
                    self.assertEqual(p.stdout, f"{uid}\n")

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
        p = self.ok("upscale", jid, "--resolution", "1080p", "--strength", "faithful", "--audio", "silence",
                    "--seed", 99, "--out", "up/", "--poster")
        up = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        out = self.work / "up" / f"lf-studio_t2va_5s_seed7_{jid[:6]}_1080p_{up['id'][:6]}.mp4"
        self.assertEqual(p.stdout, f"{out}\n")
        self.assertEqual(out.read_bytes()[4:8], b"ftyp")
        self.assertEqual(out.with_suffix(".jpg").read_bytes()[:2], b"\xff\xd8")
        self.assertRegex(p.stderr, rf"queued [0-9a-f]{{8}} \(1080p upscale of {jid[:8]}, faithful, t2va, 5 s, seed 7\), "
                                   r"next in line")
        job = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual({k: job[k] for k in ("parent", "resolution", "strength", "audio", "refine_seed", "status")},
                         {"parent": jid, "resolution": "1080p", "strength": "faithful", "audio": "silence",
                          "refine_seed": 99, "status": "done"})
        p = self.ok("upscale", jid, "--json")                                  # the studio's defaults; no wait
        rec = json.loads(p.stdout)
        self.assertEqual((rec["kind"], rec["parent"], rec["state"], rec["resolution"], rec["strength"], rec["audio"]),
                         ("upscale", jid, "queued", "2k", "balanced", "clip"))
        p = self.ok("wait", rec["id"], "--json")
        self.assertEqual(Path(json.loads(p.stdout)["path"]).name,
                         f"lf-studio_t2va_5s_seed7_{jid[:6]}_2k_{rec['id'][:6]}.mp4")
        lines = self.ok("status").stdout.splitlines()
        self.assertTrue(lines[0].startswith(f"{rec['id']}  done       2k upscale of {jid[:8]}, balanced"), lines)

    def test_a_9x16_clip(self):
        """A 5 s 9:16 clip upscales as a 16:9 one does (the studio decides; the help says so, and which sizes take
        which lengths), upright."""
        jid = self.clip(orientation="portrait", seed=8)
        p = self.ok("upscale", jid, "--wait", "--out", "up/")
        up = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual((up["orientation"], Path(up["output"]).name), ("portrait", "regen_1440x2560_121f.mp4"))
        self.assertEqual(Path(p.stdout.strip()).name, f"lf-studio_t2va_5s_seed8_{jid[:6]}_2k_{up['id'][:6]}.mp4")
        self.assertIn("upscale a finished clip, 16:9 or 9:16, whole, to 1080p, 2K (up to 10 s) or 4K (5 s)",
                      " ".join(self.ok("--help").stdout.split()))
        self.assertIn("default 2k (2560x1440, or 1440x2560 for 9:16); a 12 or 15 s clip needs 1080p",
                      " ".join(self.ok("upscale", "--help").stdout.split()))
        usage = " ".join(load_cli().__doc__.split())
        self.assertIn("a finished clip (Text, Keyframes or Reference; Sol-H3 or LightX2V Turbo), 16:9 or 9:16, whole, "
                      "to 1080p (any length), 2K (up to 10 s) or 4K (5 s), on a studio", usage)
        self.assertNotIn("5 s clip", usage)
        self.assertNotIn("untested", usage)                                  # 4K has run (16:9, docs/PERFORMANCE.md)

    def test_refusals(self):
        """The studio's refusals are exit 2 with its words (a size the Upscale does not offer for the clip's length:
        its reason, which names the sizes that take it; 2K is the studio's default); argparse checks the choices."""
        ten, twelve = self.clip(seconds=10), self.clip(seconds=12, seed=2)
        p = self.sol("upscale", ten, "--resolution", "4k", "--json")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(json.loads(p.stdout), {"error": "4K for 10 s clips comes later: 1080p and 2K take them",
                                                "http_status": 400, "id": ten})
        p = self.sol("upscale", twelve)
        self.assertEqual((p.returncode, p.stderr), (2, "lf-cli: 2K for 12 s clips comes later: 1080p takes them\n"))
        p = self.sol("upscale", "f" * 32)
        self.assertEqual((p.returncode, p.stderr), (2, "lf-cli: no such clip\n"))
        for args in (["upscale", ten, "--resolution", "8k"], ["upscale", ten, "--strength", "nvidia"],
                     ["upscale", "nonsense"], ["upscale", ten, "--seed", -1], ["upscale", ten, "--hold", 9999],
                     ["upscale", ten, "--source", "final"],                  # no --source: the clip's MP4 only
                     ["upscale", ten, "--pin", "on"]):
            p = self.sol(*args)
            self.assertEqual(p.returncode, 2, (args, p.stderr))
        self.assertEqual([j.get("kind") for j in self.studio.state()["jobs"]], [None, None])     # nothing was queued
        if os.geteuid() != 0:
            ro = self.work / "ro"
            ro.mkdir()
            ro.chmod(0o555)
            self.addCleanup(lambda: ro.exists() and ro.chmod(0o755))
            p = self.sol("upscale", self.clip(seed=3), "--out", "ro/")
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn(f"can't save the upscale there: no permission to write to {ro}; nothing was queued",
                          p.stderr)


class UpscaleEveryClip(CliTest):
    """A Keyframes clip upscales as a Text clip does (its text encoded first, on the studio); status names the model of
    an upscale's clip when it is not Sol-H3."""

    def test_a_keyframes_clip(self):
        line = "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
        jid = self.studio.generate(task="fl2va", prompt=line + "\nA red kite rises.", seed=5,
                                   files={"first_frame": [("f.png", png())]})[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        p = self.ok("upscale", jid, "--resolution", "1080p", "--wait", "--out", "up/")
        up = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual(Path(p.stdout.strip()).name, f"lf-studio_fl2va_5s_seed5_{jid[:6]}_1080p_{up['id'][:6]}.mp4")
        self.assertEqual((up["text"]["source"], up["text"]["prompt"]), ("encode", "A red kite rises."))
        self.assertTrue(self.ok("status", up["id"]).stdout.startswith(
            f"{up['id']}  done       1080p upscale of {jid[:8]}, balanced, fl2va, 5 s, seed 5"))

    def test_keep_my_frames(self):
        """--pin: off sends no end, auto (also the studio's default) its first frame (Stage 2 held it); status says
        what became of it."""
        line = "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
        jid = self.studio.generate(task="fl2va", prompt=line + "\nA red kite rises.", seed=6,
                                   files={"first_frame": [("f.png", png())]})[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        p = self.ok("upscale", jid, "--resolution", "1080p", "--pin", "off", "--json")
        rec = json.loads(p.stdout)
        self.assertEqual(rec["pin"], {"asked": "off", "ends": [], "why": {}})
        self.ok("wait", rec["id"], "--json")
        p = self.ok("upscale", jid, "--resolution", "1080p", "--pin", "auto", "--wait", "--out", "up/")
        up = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual((up["pin"]["asked"], up["pin"]["ends"], up["pin"]["result"]["first"]["used"]),
                         ("auto", ["first"], True))
        self.assertTrue(self.ok("status", up["id"]).stdout.startswith(
            f"{up['id']}  done       1080p upscale of {jid[:8]}, balanced, fl2va, 5 s, seed 6, first frame kept  "))
        self.assertNotIn("kept", self.ok("status", rec["id"]).stdout)
        self.assertIn("[--pin auto|off]", load_cli().__doc__)

    def test_the_model_of_its_clip(self):
        cli = load_cli()
        rec = {"kind": "upscale", "resolution": "2k", "parent": "a" * 32, "strength": "balanced", "task": "fl2va",
               "seconds": 5, "seed": 3}
        self.assertEqual(cli.clip_desc(dict(rec, parent_model="lightx2v-turbo", parent_opts={"resolution": "576p"})),
                         "2k upscale of aaaaaaaa (lightx2v-turbo 576p), balanced, fl2va, 5 s, seed 3")
        self.assertEqual(cli.clip_desc(dict(rec, parent_model="sol-h3")), cli.clip_desc(rec))    # as before
        kept, far = ({"used": True, "agreement_db": 33.1, "why": None},
                     {"used": False, "agreement_db": 25.43, "why": "the clip's frame 123 is 25.4 dB from your picture "
                                                                    "(28 needed)"})
        for result, words in (({"first": kept}, "first frame kept"), ({"first": kept, "last": kept},
                                                                       "first and last frames kept"),
                              ({"first": kept, "last": far}, "first frame kept, last frame not kept (25.4 dB)"),
                              ({"last": {"used": False, "agreement_db": None, "why": "your picture is no longer on "
                                                                                     "disk"}},
                               "last frame not kept: your picture is no longer on disk")):
            self.assertEqual(cli.clip_desc(dict(rec, pin={"asked": "auto", "ends": list(result), "result": result})),
                             f"{cli.clip_desc(rec)}, {words}")
        self.assertEqual(cli.clip_desc(dict(rec, pin={"asked": "off", "ends": [], "why": {}})), cli.clip_desc(rec))


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


class UpscaleTimes(unittest.TestCase):
    """The upscale times in lf-cli (its usage, the MCP instructions and the upscale tool) and docs/API.md agree with
    the Upscale's own table (upscale/regen2k_job.py OFFER_ETA_S: measured at 5 s and at some longer lengths, fitted
    for the others; docs/PERFORMANCE.md), which /api/state upscale_offers carries and the page's upEta reads, with the
    page's ~8 min more for the standard load; the page has no times of its own."""

    def model(self):
        page = (REPO / "webui" / "index.html").read_text()
        standard = int(re.search(r"j\.load==='standard'\?(\d+):0", page).group(1))
        self.assertIn("return e[0]+(j.steps||2)*e[1]+(j.load==='standard'?", page)        # nothing else counts
        spec = importlib.util.spec_from_file_location("regen2k_job_times", REPO / "upscale" / "regen2k_job.py")
        regen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regen)

        def eta(seconds, res, steps, load="fast"):
            base, step = regen.OFFER_ETA_S[seconds][res]
            return base + steps * step + (standard if load == "standard" else 0)
        return page, regen, eta, standard

    def test_the_range_and_the_standard_load(self):
        _, regen, eta, standard = self.model()
        shortest = min(eta(5, res, 2) for res in ("1080p", "2k", "4k"))                  # faithful or balanced
        longest = max(eta(5, res, 3) for res in ("1080p", "2k", "4k"))                   # full
        self.assertEqual((round(shortest / 60), round(longest / 3600, 1), round(standard / 60)), (10, 1.6, 8))
        cli = load_cli()
        api = " ".join((REPO / "docs" / "API.md").read_text().split())
        usage = " ".join(cli.__doc__.split())
        self.assertIn("about 10 min (1080p, 5 s) to 1.6 h (4K, full) in all, 4K over an hour, with a ~22 s load", usage)
        self.assertIn("~8 min more with LFSTUDIO_UPSCALE_LOAD=standard", usage)
        self.assertIn("which takes about 10 min (a 5 s clip at 1080p) to 1.6 h or more", cli.INSTRUCTIONS)
        self.assertIn("An upscale takes about 10 min (1080p, 5 s) to 1.6 h (4K, full) in all", api)
        self.assertIn("~8 min more on a studio with `LFSTUDIO_UPSCALE_LOAD=standard`", api)
        self.assertIn("An upscale takes about 10 min (a 5 s clip at 1080p) to 1.6 h, far longer than any `wait_s`", api)
        # every length offered: the shortest is a 5 s clip at 1080p, which each text names, the longest 4K full
        everything = [(eta(seconds, res, 2), seconds, res) for seconds, sized in regen.OFFER_ETA_S.items()
                      for res in sized]
        self.assertEqual(min(everything)[1:], (5, "1080p"))
        self.assertEqual((round(min(everything)[0] / 60), round(max(eta(*row[1:], 3) for row in everything) / 3600, 1)),
                         (10, 1.6))
        self.assertNotIn("min at 3 s", api)
        for text in (usage, cli.INSTRUCTIONS, api):
            self.assertNotRegex(text, r"about 10 min to")
        tool = next(t for t in cli.TOOLS if t["name"] == "upscale")["description"]
        self.assertIn("about 10 min at 1080p, 19 min at 2K and 1.1 h at 4K with 2 steps for a 5 s clip", tool)
        self.assertEqual([round(eta(5, res, 2) / 60) for res in ("1080p", "2k")], [10, 19])     # 615 s, 1165 s
        self.assertEqual(round(eta(5, "4k", 2) / 3600, 1), 1.1)                                 # 3995 s
        # longer clips, measured (2 steps, the fast load): 1080p at 10 and 15 s, 2K at 10 s
        self.assertEqual([round(eta(s, res, 2) / 60) for s, res in ((10, "1080p"), (15, "1080p"), (10, "2k"))],
                         [24, 44, 54])
        longer = "1080p 24 min at 10 s, 44 min at 15 s; 2K 54 min at 10 s"
        for text in (usage, tool):
            self.assertIn(longer, text)
            self.assertNotIn("estimate", text)
        self.assertIn(f"longer clips take longer: {longer})", usage)
        self.assertEqual(sorted(regen.UPSCALE_RAN), sorted((s, res) for s, sized in regen.OFFER_ETA_S.items()
                                                           for res in sized))     # every length: run, or covered
        for text in (usage, cli.INSTRUCTIONS, tool, api):                   # 4K has run end to end (16:9)
            self.assertNotIn("untested", text)
            self.assertNotIn("not yet run end to end", text)

    def test_the_popover_has_no_times_of_its_own(self):
        """The popover's sizes have no times in the page's markup, nor the page a table of them: openUpscale writes
        each size's time from /api/state upscale_offers (tests/test_page.py checks them)."""
        page, _, _, _ = self.model()
        for res in ("1080p", "2k", "4k"):
            label = re.search(rf'<label for="up-{res}"[^>]*>[^<]*<small([^>]*)>([^<]*)</small>', page)
            self.assertEqual(label.groups(), ("", ""), res)
        self.assertNotRegex(page, r"base:\d+,step:\d+")


def video_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return tuple(int(x) for x in out.strip().split(","))


TURBO = "lightx2v-turbo"


class Models(CliTest):
    """The model choice (LFSTUDIO_MODELS=sol-h3,lightx2v-turbo): lf-cli models, generate --model with the model's
    options, checked against the studio's own table before anything is sent, and LightX2V Turbo's clips in status
    and as files."""
    models = "sol-h3,lightx2v-turbo"

    def test_models(self):
        """lf-cli models: each model, what it makes now, its options, what not yet and why, its time per clip (an
        estimate until a clip has been measured on the studio); --json is the studio's table."""
        p = self.ok("models")
        lines = p.stdout.splitlines()
        self.assertTrue(lines[0].startswith("sol-h3: Sol-H3 (the default), NVIDIA's Sol two-stage pipeline"), lines)
        self.assertIn("  makes t2va, fl2va or ref2va (fl2va keyframes: first, last or first+last), 5, 6, 7, 8, 10, 12 "
                      "or 15 s, landscape or portrait", lines)
        for line in ("  makes t2va or fl2va (fl2va keyframes: first, last or first+last), 5, 6, 7, 8, 10, 12 or 15 s, "
                     "landscape or portrait, resolution 576p or 768p, steps 4",
                     "  option resolution: 576p (the default; 1024x576, 9:16 576x1024), 768p (1344x768, 9:16 768x1344)",
                     "  option steps: 4 (the default)",
                     "  not yet: ref2va: LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not "
                     "Reference (ref2va): use Sol-H3",
                     "  time per clip, resolution 576p, steps 4: 5 s about 3:20, 6 s about 4:10, 7 s about 4:40, 8 s "
                     "about 5:10, 10 s about 6:40, 12 s about 8:20, 15 s about 11:00 (estimates until one is measured "
                     "here)",
                     "  time per clip, resolution 768p, steps 4: 5 s about 5:50, 6 s about 7:50, 7 s about 8:50, 8 s "
                     "about 10:00, 10 s about 13:40, 12 s about 17:50, 15 s about 24:00 (estimates until one is "
                     "measured here)"):
            self.assertIn(line, lines)
        self.assertTrue(any(x.startswith(f"{TURBO}: LightX2V Turbo (experimental), ") for x in lines), lines)
        self.assertEqual([x for x in lines if x.startswith("  not yet: ")],
                         ["  not yet: ref2va: LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, "
                          "not Reference (ref2va): use Sol-H3"])
        table = json.loads(self.ok("models", "--json").stdout)
        self.assertTrue(table["model_choice"])
        self.assertEqual(table["models"], self.studio.state()["models"])
        jid = json.loads(self.ok("generate", "--prompt", "x", "--model", TURBO, "--wait", "--json").stdout)["id"]
        self.assertRegex(self.ok("models").stdout, r"  time per clip, resolution 576p, steps 4: 5 s about 0:0\d "
                                                   r"\(measured here\), 6 s about 4:10, .*, 15 s about 11:00 "
                                                   r"\(estimates where not measured here\)\n")
        self.assertRegex(self.ok("status", jid).stdout, rf"^{jid}  done       t2va, landscape, 5 s, seed \d+, {TURBO} "
                                                        r"576p 4 steps  done in 0:0\d\n$")

    def test_generate_and_save(self):
        """--model lightx2v-turbo --resolution 768p: the clip's record has the model and its options, the MP4 is
        1344x768 and is saved under the studio's name for it (file_name); 9:16 at 576p is 576x1024. --model sol-h3
        makes a Sol-H3 clip as without it, under its name as before."""
        p = self.ok("generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p", "--seed", 3, "--wait",
                    "--json")
        rec = json.loads(p.stdout)
        self.assertEqual((rec["state"], rec["model"], rec["model_opts"]["resolution"], rec["model_opts"]["steps"]),
                         ("done", TURBO, "768p", 4))
        path = Path(rec["path"])
        self.assertEqual(path.name, f"lf-studio_{TURBO}-768p_t2va_5s_seed3_{rec['id'][:6]}.mp4")
        self.assertEqual(path.name, rec["file_name"])
        self.assertEqual(video_size(path), (1344, 768))
        self.assertIn(f"(t2va, landscape, 5 s, seed 3, {TURBO} 768p 4 steps), next in line", p.stderr)
        self.assertRegex(p.stderr, r"done in 0:0\d")
        self.frame("first.png", 48, 84)                                          # 9:16, from the frame's shape
        p = self.ok("generate", "--prompt", "x", "--model", TURBO, "--option", "steps=4", "--first-frame",
                    "first.png", "--out", "clips/", "--json")
        rec = json.loads(p.stdout)
        self.assertEqual((rec["task"], rec["orientation"], rec["model_opts"]["resolution"], rec["model_opts"]["size"]),
                         ("fl2va", "portrait", "576p", [576, 1024]))
        self.assertEqual(video_size(rec["path"]), (576, 1024))
        self.assertEqual(Path(rec["path"]).parent, self.work / "clips")
        rec = json.loads(self.ok("generate", "--prompt", "x", "--model", "sol-h3", "--wait", "--json").stdout)
        self.assertEqual((rec["state"], rec["model"]), ("done", "sol-h3"))
        self.assertNotIn("model_opts", rec)
        self.assertEqual(Path(rec["path"]).name, f"lf-studio_t2va_5s_seed{rec['seed']}_{rec['id'][:6]}.mp4")

    def test_refused_before_anything_is_sent(self):
        """What the model doesn't make (yet), an option it doesn't have or a value it lacks, a model the studio
        doesn't have, a core field as an option: exit 2 with the reason, and nothing reaches the studio. What it makes
        since its GPU validation (6 s, a last frame alone) and its GPU runs of 15 s and 768p 12 s is taken."""
        self.frame("last.png", 84, 48)
        self.frame("face.png")
        for args, words in [
                (["--model", TURBO, "--ref", "face.png"], "LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) "
                                                          "clips for now, not Reference (ref2va): use Sol-H3"),
                (["--model", TURBO, "--resolution", "1080p"], "resolution must be 576p or 768p for lightx2v-turbo"),
                (["--model", TURBO, "--steps", 8], "steps must be 4 for lightx2v-turbo"),
                (["--model", TURBO, "--option", "foo=1"], "foo is not an option of lightx2v-turbo; it takes "
                                                          "resolution, steps"),
                (["--model", TURBO, "--option", "seconds=5"], "--option seconds: seconds is a field of every clip"),
                (["--model", TURBO, "--option", "resolution"], "--option takes NAME=VALUE"),
                (["--model", TURBO, "--resolution", "768p", "--option", "resolution=576p"],
                 "resolution is given twice"),
                (["--resolution", "576p"], "resolution is not an option of sol-h3; it takes none"),
                (["--model", "fast"], "this studio has no model fast; it has sol-h3 (Sol-H3), lightx2v-turbo "
                                      "(LightX2V Turbo) (lf-cli models)"),
                (["--model", "LightX2V"], "--model must be a model id such as sol-h3 or lightx2v-turbo")]:
            p = self.sol("generate", "--prompt", "x", *args)
            self.assertEqual(p.returncode, 2, (args, p.stderr))
            self.assertIn(words, p.stderr, args)
        self.assertEqual(self.studio.state()["jobs"], [])
        p = self.sol("generate", "--prompt", "x", "--model", TURBO, "--ref", "face.png", "--json")
        self.assertEqual(json.loads(p.stdout)["error"], "LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips "
                                                        "for now, not Reference (ref2va): use Sol-H3; nothing was "
                                                        "submitted")
        rec = json.loads(self.ok("generate", "--prompt", "x", "--model", TURBO, "--seconds", 6, "--last-frame",
                                 "last.png", "--wait", "--json").stdout)    # what the GPU validation cleared
        self.assertEqual((rec["state"], rec["task"], rec["seconds"], rec["model"]), ("done", "fl2va", 6, TURBO))
        for seconds in (12, 15):
            rec = json.loads(self.ok("generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p",
                                     "--seconds", seconds, "--wait", "--json").stdout)
            self.assertEqual((rec["state"], rec["seconds"], rec["model_opts"]["resolution"]), ("done", seconds, "768p"))

    def test_values_made_apart(self):
        """A studio whose table has values only in different rows (LightX2V Turbo's before its 15 s and 768p 12 s ran
        on the GPU: 576p up to 12 s, 768p up to 10 s; a fake studio serves it): a clip no row has in full is refused
        before anything is sent, one a row has is sent."""
        state = self.studio.state()
        turbo = next(m for m in state["models"] if m["id"] == TURBO)
        row = turbo["combinations"][0]
        turbo["combinations"] = [dict(row, seconds=[5, 6, 7, 8, 10, 12], resolution=["576p"]),
                                 dict(row, seconds=[5, 6, 7, 8, 10], resolution=["768p"])]

        def answer(method, path, headers, body):
            if path == "/api/state":
                return 200, JSON, state
            return 500, JSON, {"error": "not expected"}
        fake = Fake(answer)
        self.addCleanup(fake.stop)
        p = self.sol("generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p", "--seconds", 12,
                     url=fake.url)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("LightX2V Turbo does not make clips with task t2va, seconds 12, orientation landscape, "
                      "resolution 768p, steps 4 together yet", p.stderr)
        self.assertEqual(fake.paths(), ["/api/state"])
        p = self.sol("generate", "--prompt", "x", "--model", TURBO, "--resolution", "576p", "--seconds", 12,
                     url=fake.url)
        self.assertEqual(fake.paths()[-1], "/api/generate", p.stderr)

    def test_keyframe_shape_of_the_model(self):
        """The shape warning names the clip's size at the model's resolution."""
        self.frame("tall.png", 24, 42)
        p = self.ok("generate", "--prompt", "x", "--model", TURBO, "--first-frame", "tall.png", "--orientation",
                    "landscape")
        self.assertIn("warning: --first-frame is 24x42, but a landscape clip is 1024x576", p.stderr)
        p = self.ok("generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p", "--first-frame",
                    "tall.png", "--orientation", "landscape")
        self.assertIn("warning: --first-frame is 24x42, but a landscape clip is 1344x768", p.stderr)


class ModelsRunning(CliTest):
    """A LightX2V Turbo clip while it runs: status follows its steps, cancel says how it stops."""
    models = "sol-h3,lightx2v-turbo"
    studio_env = {"STUB_H3TURBO_S": "30"}

    def test_status_and_cancel(self):
        jid = self.ok("generate", "--prompt", "x", "--model", TURBO).stdout.strip()
        seen, deadline = set(), time.time() + 60
        while time.time() < deadline and not any(s.startswith("generating: step") for s in seen):
            line = self.ok("status", jid).stdout
            self.assertRegex(line, rf"^{jid}  running    t2va, landscape, 5 s, seed \d+, {TURBO} 576p 4 steps  ")
            seen.add(line.split("4 steps  ", 1)[1].split(",")[0].strip())
            time.sleep(0.2)
        self.assertIn("generating: step 1/4", seen)
        self.assertLessEqual(seen - {"closing the sol-h3 models first", "encoding the prompt",
                                     f"loading {TURBO}", "generating: step 1/4", "generating: step 2/4"}, set(), seen)
        p = self.ok("cancel", jid)
        self.assertEqual(p.stdout.strip(), f"cancelling {jid} ({TURBO}): its job stops within about 20 s")
        p = self.sol("wait", jid)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn(f"clip {jid} was cancelled", p.stderr)


class ModelsOff(CliTest):
    """A studio with the model choice off (as by default): one model, Sol-H3; another model or an option is refused
    before anything is sent; --model sol-h3 makes the clip as without it."""

    def test_off(self):
        p = self.ok("models")
        self.assertEqual(p.stdout, "This studio has no model choice: every clip is Sol-H3 (sol-h3), NVIDIA's two-stage "
                                   "pipeline (its LFSTUDIO_MODELS lists sol-h3 alone, or it is an older version).\n")
        self.assertEqual(json.loads(self.ok("models", "--json").stdout), {"model_choice": False, "models": []})
        for args, what in ((["--model", TURBO], f"a {TURBO} clip"), (["--resolution", "576p"], "a clip with options"),
                           (["--model", "sol-h3", "--steps", 4], "a clip with options")):
            p = self.sol("generate", "--prompt", "x", *args)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("this studio has no model choice: it makes Sol-H3 (sol-h3) clips only, without options (its "
                          f"LFSTUDIO_MODELS lists sol-h3 alone, or it is an older version), so it can't make {what}; "
                          "leave out --model and the options", p.stderr)
        self.assertEqual(self.studio.state()["jobs"], [])
        rec = json.loads(self.ok("generate", "--prompt", "x", "--model", "sol-h3", "--wait", "--json").stdout)
        self.assertEqual(rec["state"], "done")
        self.assertNotIn("model", rec)                                   # the records as before the model choice
        envfile = self.dir / "env"                                       # bin/lf-studio models runs the client
        envfile.write_text(f"LFSTUDIO_HOST=127.0.0.1\nLFSTUDIO_PORT={self.studio.port}\n")
        p = subprocess.run(["bash", str(REPO / "bin" / "lf-studio"), "models"], capture_output=True, text=True,
                           timeout=30, cwd=self.work, env=client_env(LFSTUDIO_ENV=str(envfile)))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(p.stdout.startswith("This studio has no model choice"), p.stdout)


class ModelsOldStudio(unittest.TestCase):
    """An older studio (no "models" in /api/state; its form ignores fields it doesn't know) and one that
    spark-switchboard has stopped (/api/state is passive: it can't be asked first): never a silent Sol-H3 clip."""
    JID = "0123456789abcdef0123456789abcdef"

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

    def test_an_older_studio(self):
        def answer(method, path, headers, body):
            if path == "/api/state":
                return 200, JSON, {"now": 1, "jobs": [], "session": {}, "upscale": False}
            return 500, JSON, {"error": "not expected"}
        fake = self.fake(answer)
        p = self.sol(fake, "generate", "--prompt", "x", "--model", TURBO)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn(f"this studio has no model choice: it makes Sol-H3 (sol-h3) clips only", p.stderr)
        self.assertEqual([(m, p) for m, p, _, _ in fake.requests], [("GET", "/api/state")])
        self.assertEqual(fake.requests[0][2]["X-Switchboard-Wait"], "0")            # not held while it starts
        p = self.sol(fake, "models")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("This studio has no model choice", p.stdout)
        p = self.sol(fake, "generate", "--prompt", "x")                             # no model: no look first
        self.assertEqual(p.returncode, 3)
        self.assertEqual(fake.requests[-1][:2], ("POST", "/api/generate"))
        self.assertEqual([p for _, p, _, _ in fake.requests].count("/api/state"), 2)

    def test_stopped_by_switchboard(self):
        """switchboard has the studio stopped: the submit starts it, and an answer without the model (an older
        studio, or the model choice off) is cancelled at once (exit 2); one with it goes on as usual."""
        has_model = []

        def answer(method, path, headers, body):
            if path == "/api/state":
                return switchboard_error("stopped", "LF Studio is not running")
            if path == "/api/generate":
                job = {"id": self.JID, "state": "queued", "ahead": 0, "task": "t2va", "orientation": "landscape",
                       "seconds": 5, "seed": 1}
                if has_model:
                    job.update(model=TURBO, model_opts={"resolution": "768p", "steps": 4})
                return 200, JSON, {"id": self.JID, "job": job}
            if path == "/api/cancel":
                return 200, JSON, {"ok": True, "job": {"id": self.JID, "status": "cancelled"}}
            return 500, JSON, {"error": "not expected"}
        fake = self.fake(answer)
        p = self.sol(fake, "generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p", "--json")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn(f"so it can't make a {TURBO} clip", p.stderr)
        self.assertIn(f"spark-switchboard had the studio stopped, so its models could not be read before the submit; "
                      f"the clip it queued instead ({self.JID}, a Sol-H3 clip) is cancelled", p.stderr)
        self.assertEqual(json.loads(p.stdout)["id"], self.JID)
        self.assertEqual([(m, p) for m, p, _, _ in fake.requests],
                         [("GET", "/api/state"), ("POST", "/api/generate"), ("POST", "/api/cancel")])
        form = fake.requests[1][3]
        self.assertIn(b'name="model"\r\n\r\nlightx2v-turbo\r\n', form)
        self.assertIn(b'name="resolution"\r\n\r\n768p\r\n', form)
        self.assertEqual(json.loads(fake.requests[2][3]), {"id": self.JID})
        has_model.append(True)
        p = self.sol(fake, "generate", "--prompt", "x", "--model", TURBO, "--resolution", "768p")
        self.assertEqual((p.returncode, p.stdout.strip()), (0, self.JID), p.stderr)
        self.assertEqual(fake.requests[-1][:2], ("POST", "/api/generate"))


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

    def test_as_v051(self):
        """The prompt guide is v0.5.1's (from git): it is Sol-H3's, and lf-cli guide and the MCP prompt_guide say the
        same on every studio, the model choice on or off. The models tool and docs/API.md's Models say the rest."""
        try:
            r = subprocess.run(["git", "-C", str(REPO), "show", "v0.5.1:bin/lf-cli"], capture_output=True, text=True,
                               timeout=60)
        except (OSError, subprocess.SubprocessError):
            r = None
        if r is None or r.returncode:
            self.skipTest("v0.5.1 is not in this copy's git history")
        old = next(ast.literal_eval(n.value) for n in ast.parse(r.stdout).body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "GUIDE")
        self.assertEqual(load_cli().GUIDE, old)

    def test_lengths_offered(self):
        """3 and 4 s went after 0.5.0: the guide, the usage, generate --help and the MCP instructions name 5-15 s."""
        cli = load_cli()
        self.assertEqual(cli.DURATIONS, (5, 6, 7, 8, 10, 12, 15))
        self.assertIn("- `seconds`: 5, 6, 7, 8, 10, 12 or 15.\n", cli.GUIDE)
        self.assertIn("[--seconds 5|6|7|8|10|12|15]", cli.__doc__)
        self.assertIn(": 5-15 s,", cli.INSTRUCTIONS)
        self.assertNotRegex(cli.INSTRUCTIONS, r"\b3 s\b")
        p = subprocess.run([sys.executable, str(BIN), "generate", "--help"], capture_output=True, text=True,
                           timeout=30, env=client_env())
        self.assertEqual(p.returncode, 0)
        self.assertIn("--seconds S 5, 6, 7, 8, 10, 12 or 15 (default 5)", " ".join(p.stdout.split()))

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
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "lf-cli 0.5.2"))
        self.assertEqual(sorted(x.name for x in d.iterdir()), ["lf-cli"])


if __name__ == "__main__":
    unittest.main()
