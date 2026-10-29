"""webui/server.py end to end against the stub runtime (tests/stub): no models, no GPU, a few seconds per test.

python3 -m unittest discover -s tests -v      (needs python3 >= 3.10 and ffmpeg; the server uses the standard
library only)
"""
import ast
import base64
import concurrent.futures
import contextlib
import errno
import hashlib
import importlib.util
import io
import itertools
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import types
import unittest
import urllib.error
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio, free_port  # noqa: E402

if not shutil.which("ffmpeg"):
    raise unittest.SkipTest("ffmpeg not installed")


def png(width=32, height=24):
    rows = b"".join(b"\x00" + bytes((x * 8) % 256 for x in range(width) for _ in range(3)) for _ in range(height))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def finished_job(jid="0" * 32, task="t2va", orientation="landscape", seconds=5, **extra):
    return dict({"id": jid, "case_id": f"web_seed_{jid[:6]}", "task": task, "orientation": orientation,
                 "seconds": seconds, "prompt": "an earlier clip", "seed": 7, "status": "done",
                 "created": time.time() - 3600}, **extra)


# the refusal of a length that went after 0.5.0 (3 and 4 s), by /api/generate (after "ValueError: ") and /api/load
GONE = "duration must be 5, 6, 7, 8, 10, 12 or 15 seconds (3 and 4 s are no longer offered)"


class StudioTest(unittest.TestCase):
    """One fresh stub studio per test. Subclasses set the timings; seed() writes state before start."""
    load_s, clip_s, preload = 2, 1, False

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.studio = StubStudio(self.dir, load_s=self.load_s, clip_s=self.clip_s, preload=self.preload)
        self.seed()
        self.studio.start()

    def tearDown(self):
        self.studio.stop()
        if hasattr(self, "_outcome") and not getattr(self._outcome, "success", True):
            print(f"\n--- server log ({self.studio.log})\n" + self.studio.log.read_text()[-3000:])
        shutil.rmtree(self.dir, ignore_errors=True)

    def seed(self):
        pass

    def write_state(self, jobs=None, load_times=None):
        if jobs is not None:
            (self.studio.state_dir / "jobs.json").write_text(json.dumps(jobs))
        if load_times is not None:
            (self.studio.state_dir / "load-times.json").write_text(json.dumps(load_times))

    def generate(self, **kw):
        code, body = self.studio.generate(**kw)
        self.assertEqual(code, 200, body)
        return body["id"]


def children(pid):
    """The processes `pid` started (Linux /proc)."""
    found = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]) == pid:
                    found.append(int(d.name))
            except (OSError, ValueError):
                continue
    return found


class StubStart(unittest.TestCase):
    """tests/stub/studio.py: a stub studio counts as started only once its own server serves the port. Another studio
    on that port answers /health too, and everything the tests or a reviewer then sent went to it."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-stub-"))
        self.first = StubStudio(self.dir / "first", preload=False).start()
        self.others = []

    def tearDown(self):
        for s in self.others + [self.first]:
            s.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def other(self, name, port):
        self.others.append(StubStudio(self.dir / name, port=port, preload=False))
        return self.others[-1]

    def test_a_port_in_use(self):
        second = self.other("second", self.first.port)
        t = time.time()
        with self.assertRaisesRegex(RuntimeError, f"port {self.first.port} is in use"):
            second.start()
        self.assertLess(time.time() - t, 1, "fails at once")
        self.assertIsNone(second.proc)                                           # no server was started
        # taken between that check and the server's bind: the server exits, and the first studio's answers don't count
        with mock.patch("studio.port_taken", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Address already in use"):
                second.start()
            self.assertIsNotNone(second.proc.poll())
            # a restart: the log still has the start line of the run before, which doesn't count either
            self.first.stop()
            third = self.other("third", self.first.port).start()
            with self.assertRaisesRegex(RuntimeError, "Address already in use"):
                self.first.start()
        self.assertEqual(self.first.log.read_text().count("LF Studio on"), 1)
        self.assertEqual(third.state()["jobs"], [])                              # nothing went to the third studio

    def command(self, port, name):
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "stub" / "studio.py"),
                                 "--port", str(port), "--dir", str(self.dir / name), "--no-preload"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_the_command(self):
        """`python3 tests/stub/studio.py --port P` exits non-zero at once if P is in use, and when its server stops
        by itself; stopped (SIGTERM, Ctrl-C), it exits with 0."""
        cmd = self.command(self.first.port, "taken")
        out, err = cmd.communicate(timeout=20)
        self.assertEqual((cmd.returncode, out), (1, ""))
        self.assertIn(f"stub studio not started: port {self.first.port} is in use", err)
        for stop, code in (("server", 1), ("command", 0)):
            port = free_port()
            cmd = self.command(port, f"run-{stop}")
            self.assertIn(f"stub studio on http://127.0.0.1:{port}/", cmd.stdout.readline())
            server, = children(cmd.pid)
            os.kill(server if stop == "server" else cmd.pid, signal.SIGKILL if stop == "server" else signal.SIGTERM)
            out, err = cmd.communicate(timeout=30)
            self.assertEqual(cmd.returncode, code, err)
            if stop == "server":
                self.assertIn("the server stopped by itself (exit code -9)", err)
            self.assertFalse(Path(f"/proc/{server}").exists(), "the server was waited for")


class Basics(StudioTest):
    def test_clip_cold_then_warm_and_served(self):
        a = self.generate()
        b = self.generate(seed=2)
        ja, jb = self.studio.wait([a, b])
        self.assertEqual((ja["status"], jb["status"]), ("done", "done"))
        self.assertTrue(ja.get("includes_model_load"))
        self.assertFalse(jb.get("includes_model_load"))
        code, headers, data = self.studio.get(f"/video/{a}", {"Range": "bytes=0-99"})
        self.assertEqual((code, len(data), headers["Content-Type"]), (206, 100, "video/mp4"))
        code, headers, _ = self.studio.get(f"/poster/{a}")
        self.assertEqual((code, headers["Content-Type"]), (200, "image/jpeg"))
        times = json.loads((self.studio.state_dir / "load-times.json").read_text())
        self.assertIn("t2va/landscape/5", times)
        self.assertIn("warm/t2va/landscape/5", times)

    def test_fl2va_upload_kept_per_job(self):
        code, body = self.studio.generate(task="fl2va")
        self.assertEqual(code, 400)
        self.assertIn("first frame", body["error"])
        jid = self.generate(task="fl2va", files={"first_frame": [("my frame.png", png())]})
        job = self.studio.wait([jid])[0]
        self.assertEqual(job["status"], "done")
        self.assertEqual(Path(job["first_frame"]).parent, self.studio.runtime / "inputs" / "webui" / jid)
        self.assertEqual(Path(job["first_frame"]).name, "my_frame.png")

    def test_worker_failure_retried_then_failed(self):
        jid = self.generate(prompt="STUB_FAIL please")
        job = self.studio.wait([jid])[0]
        self.assertEqual(job["status"], "failed")
        self.assertIn("STUB_FAIL", job["first_error"])
        self.assertIn("STUB_FAIL", job["error"])

    def test_cross_origin_post_refused(self):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.studio.url + "/api/unload", b"", {"Origin": "http://evil.example"},
                                     method="POST")
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        self.assertEqual(e.exception.code, 403)


class HostCheck(StudioTest):
    """#29: a request for a host name the studio does not answer to gets 421, whatever the method: a web page on a
    name of its own that resolves to the studio's address (DNS rebinding) gets nothing. IP addresses, localhost, names
    without a dot and local suffixes pass; LFSTUDIO_ALLOWED_HOSTS adds names, or turns the check off with *."""
    load_s, clip_s = 1, 0.5

    def ask(self, host, method="GET", path="/api/state", **headers):
        body = json.dumps({"id": "f" * 32}).encode() if method == "POST" else None
        code, _, data = self.studio.request(method, path, body, dict(headers, Host=host))
        return code, data

    def refused(self, host, method="GET", path="/api/state", **headers):
        code, data = self.ask(host, method, path, **headers)
        self.assertEqual(code, 421, (host, method, path, data))
        if method != "HEAD":
            self.assertIn(b"add it to LFSTUDIO_ALLOWED_HOSTS", data)
        return json.loads(data or b"{}").get("error")

    def test_unknown_names_refused_for_every_method(self):
        self.assertEqual(self.refused("Evil.Example:8190"),
                         "unknown host name evil.example; add it to LFSTUDIO_ALLOWED_HOSTS")
        self.refused("evil.example", "HEAD")
        self.refused("evil.example", "GET", "/")
        self.refused("evil.example", "GET", f"/video/{'f' * 32}")
        self.refused("evil.example", "POST", "/api/cancel")
        self.refused("studio.lan.evil.example", "POST", "/api/unload")
        self.refused("ts.net")                                               # a suffix, not a name of its own
        b = "hostcheck7MA4"
        form = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\na clip\r\n--{b}--\r\n').encode()
        code, _, data = self.studio.request("POST", "/api/generate", form,
                                            {"Host": "evil.example", "Content-Type": f"multipart/form-data; boundary={b}"})
        self.assertEqual(code, 421, data)
        self.assertEqual(self.studio.state()["jobs"], [])                     # nothing queued
        self.refused(f"127.0.0.1:{self.studio.port}", **{"X-Forwarded-Host": "evil.example"})
        self.refused(f"127.0.0.1:{self.studio.port}", **{"X-Forwarded-Host": "spark:8190, evil.example"})

    def test_addresses_local_names_and_the_busy_probe_pass(self):
        port = self.studio.port
        for host in (f"127.0.0.1:{port}", "127.0.0.1", f"[::1]:{port}", "[::1]", f"localhost:{port}", "LOCALHOST",
                     "localhost.", "10.1.2.3:8190", "100.101.102.103", "[fd7a:115c:a1e0::1]:8190", "spark:8190",
                     "spark-box", "spark.tail1234.ts.net:8190", "studio.local", "box.lan", "box.home",
                     "box.home.arpa", "gpu.internal:8190"):
            self.assertEqual(self.ask(host)[0], 200, host)
        self.assertEqual(self.ask("spark:8190", "POST", "/api/cancel")[0], 404)         # past the check: no such clip
        self.assertEqual(self.ask(f"127.0.0.1:{port}", **{"X-Forwarded-Host": "spark.tail1234.ts.net:8190"})[0], 200)

    def test_allowed_hosts_setting(self):
        self.studio.stop()
        self.studio.env["LFSTUDIO_ALLOWED_HOSTS"] = " Studio.Example.com:8443 , other.example"
        self.studio.start()
        for host in ("studio.example.com", "STUDIO.example.com:8190", "other.example:80"):
            self.assertEqual(self.ask(host)[0], 200, host)
        self.assertEqual(self.ask(f"127.0.0.1:{self.studio.port}", **{"X-Forwarded-Host": "other.example"})[0], 200)
        self.refused("evil.example")
        self.refused("example.com")
        self.studio.stop()
        self.studio.env["LFSTUDIO_ALLOWED_HOSTS"] = "*"
        self.studio.start()
        self.assertEqual(self.ask("evil.example")[0], 200)
        self.assertEqual(self.ask("evil.example", "POST", "/api/cancel")[0], 404)


def mp4(seconds=3):
    """A small test video with sound (reference videos must be at least 2 s)."""
    out = Path(tempfile.mkdtemp(prefix="lf-studio-mp4-")) / "ref.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=320x180:r=24:d={seconds}", "-f",
                    "lavfi", "-i", f"sine=duration={seconds}", "-shortest", "-pix_fmt", "yuv420p", str(out)], check=True)
    data = out.read_bytes()
    shutil.rmtree(out.parent)
    return data


class Remix(StudioTest):
    """#1: Remix reuses the clip's input files (hardlinked into the new job's directory), in their order."""

    def test_fl2va_inputs_reused(self):
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png(40))]})
        src = self.studio.wait([a])[0]
        code, body = self.studio.post("/api/generate", dict(task="fl2va", orientation="landscape", seconds=5,
                                                              prompt="again", seed=3, remix_from=a,
                                                              keep="first_frame:0,last_frame:0"))
        self.assertEqual(code, 200, body)
        job = self.studio.wait([body["id"]])[0]
        self.assertEqual(job["status"], "done", job.get("error"))
        for key in ("first_frame", "last_frame"):
            new, old = Path(job[key]), Path(src[key])
            self.assertEqual(new.parent.name, body["id"])                      # in the new job's own directory
            self.assertEqual(new.read_bytes(), old.read_bytes())
            self.assertEqual(new.stat().st_ino, old.stat().st_ino)             # a hardlink, not a copy

    def test_ref2va_order_and_normalized_video_kept(self):
        a = self.generate(task="ref2va", files={"ref_images": [("one.png", png()), ("two.png", png(48))],
                                                 "ref_videos": [("clip.mp4", mp4())],
                                                 "ref_audio": [("voice.wav", b"RIFF0000WAVEfmt ")]})
        src = self.studio.wait([a])[0]
        self.assertEqual(src["status"], "done", src.get("error"))
        code, inputs = self.studio.post("/api/generate", dict(task="ref2va", prompt="again", seed=4, remix_from=a,
                                                                keep="ref_images:0,ref_images:1,ref_videos:0,"
                                                                     "ref_audio:0"))
        self.assertEqual(code, 200, inputs)
        job = self.studio.wait([inputs["id"]])[0]
        names = [(r["type"], Path(r["path"]).name) for r in job["references"]]
        self.assertEqual(names, [("image", "one.png"), ("image", "two.png"), ("video", "clip_ref.mp4"),
                                 ("audio", "voice.wav")])                       # same order, video not re-normalized

    def test_reference_zone_partly_missing_kept_one_by_one_and_new_files_after(self):
        a = self.generate(task="ref2va", files={"ref_images": [("one.png", png()), ("two.png", png(48)),
                                                                ("three.png", png(56))]})
        src = self.studio.wait([a])[0]
        Path(src["references"][1]["path"]).unlink()                            # two.png is gone
        listing = json.loads(self.studio.get(f"/api/inputs/{a}")[2])
        self.assertEqual([(i["name"], i["index"]) for i in listing["inputs"]["ref_images"]],
                         [("one.png", 0), ("three.png", 2)])
        self.assertEqual(listing["missing"], ["two.png"])
        jid = self.generate(task="ref2va", remix_from=a, keep="ref_images:2,ref_images:0",
                            files={"ref_images": [("new.png", png(20))]})
        job = self.studio.wait([jid])[0]
        self.assertEqual([Path(r["path"]).name for r in job["references"]], ["three.png", "one.png", "new.png"])
        code, body = self.studio.generate(task="ref2va", remix_from=a, keep="ref_images:7")
        self.assertEqual(code, 400)
        self.assertIn("fewer files", body["error"])

    def test_new_files_replace_a_kept_zone_and_unlisted_zones_are_dropped(self):
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png())]})
        self.studio.wait([a])
        jid = self.generate(task="fl2va", remix_from=a, keep="first_frame:0",
                            files={"first_frame": [("new.png", png(20))]})
        job = self.studio.wait([jid])[0]
        self.assertEqual(Path(job["first_frame"]).name, "new.png")
        self.assertNotIn("last_frame", job)                                     # not in keep: not reused

    def test_inputs_listing_serving_and_missing(self):
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png())]})
        src = self.studio.wait([a])[0]
        listing = json.loads(self.studio.get(f"/api/inputs/{a}")[2])
        self.assertEqual(listing["missing"], [])
        self.assertEqual([i["name"] for i in listing["inputs"]["first_frame"]], ["first.png"])
        code, headers, data = self.studio.get(listing["inputs"]["first_frame"][0]["url"])
        self.assertEqual((code, headers["Content-Type"], data), (200, "image/png", Path(src["first_frame"]).read_bytes()))
        Path(src["last_frame"]).unlink()
        listing = json.loads(self.studio.get(f"/api/inputs/{a}")[2])
        self.assertEqual((listing["missing"], "last_frame" in listing["inputs"]), (["last.png"], False))
        code, body = self.studio.generate(task="fl2va", remix_from=a, keep="first_frame:0,last_frame:0")
        self.assertEqual(code, 400)                                         # a listed file vanished since
        self.assertIn("last.png", body["error"])
        self.assertEqual(len(list((self.studio.runtime / "inputs" / "webui").iterdir())), 1)   # refused: no dir left
        for path in (f"/input/{a}/last_frame/0", f"/input/{a}/first_frame/1", f"/input/{a}/nonsense/0",
                     f"/api/inputs/{'f' * 32}"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(path)
            self.assertEqual(e.exception.code, 404, path)

    def test_unknown_remix_source(self):
        code, body = self.studio.generate(task="fl2va", remix_from="a" * 32, keep="first_frame:0")
        self.assertEqual(code, 400)
        self.assertIn("no longer in the list", body["error"])


STILL = "@still.jpg"                         # a clip's card picture in its input directory (server.py)
PICTURE_CACHE = "private, max-age=86400"     # /still and /poster


def photo(size):
    """A large PNG, like a phone photo (ffmpeg's test pattern: about 1.2 MB at 4032x3024)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}", "-frames:v", "1",
                           "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True, check=True).stdout


def jpeg_size(data):
    """(width, height) from a JPEG's frame header."""
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    i = 2
    while i + 9 <= len(data):
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise ValueError("no frame header")


class CardPicture(StudioTest):
    """#30: GET /still/<id>, the picture behind a queued or running clip's timer: its first frame (else its last), its
    first reference image, else the first frame of its first reference video, as a JPEG at most 640 px wide, kept
    with the clip's input files."""
    load_s = 3

    def still(self, path):
        try:
            code, headers, data = self.studio.get(path if path.startswith("/") else f"/still/{path}")
            return code, headers["Content-Type"], data
        except urllib.error.HTTPError as e:
            return e.code, None, None

    def picture(self, jid):
        """/still/<id> as (status, type, JPEG size)."""
        code, ctype, data = self.still(jid)
        return code, ctype, jpeg_size(data)

    def test_keyframes_and_reference_images(self):
        first, last, one, video = png(), png(40), png(48), mp4()
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", first)], "last_frame": [("last.png", last)]})
        b = self.generate(task="fl2va", seed=2, files={"last_frame": [("last.png", last)]})
        self.assertEqual(self.studio.job(b)["status"], "queued")
        self.assertEqual(self.picture(b), (200, "image/jpeg", (40, 24)))       # no first frame: the last
        self.assertEqual(self.picture(a), (200, "image/jpeg", (32, 24)))       # small ones keep their size
        c = self.generate(task="ref2va", seed=3, files={"ref_images": [("one.png", one), ("two.png", png(56))],
                                                       "ref_videos": [("clip.mp4", video)]})
        self.assertEqual(self.picture(c), (200, "image/jpeg", (48, 24)))       # an image before a video
        t = self.generate(seed=4)
        self.assertEqual(self.still(t)[0], 404)                                  # t2va: no picture
        data = self.still(a)[2]
        self.assertEqual([j["status"] for j in self.studio.wait([a, b, c, t])], ["done"] * 4)
        self.assertEqual(self.still(a), (200, "image/jpeg", data))               # and still once it is done
        inputs = self.studio.runtime / "inputs" / "webui"
        self.assertEqual(sorted(p.name for p in (inputs / a).iterdir()), [STILL, "first.png", "last.png"])
        self.assertEqual(((inputs / a / "first.png").read_bytes(), (inputs / b / "last.png").read_bytes()),
                         (first, last))                                          # the uploads as they were

    def test_a_photo_goes_out_small_and_cached(self):
        """An upload can be a phone photo of many MB, and the card shows it blurred: a JPEG at most 640 px wide, which
        the browser keeps (a clip's pictures never change); the poster too."""
        wide, tall = photo("4032x3024"), photo("3024x4032")
        a = self.generate(task="fl2va", files={"first_frame": [("wide.png", wide)]})
        b = self.generate(task="ref2va", seed=2, files={"ref_images": [("tall.png", tall)]})
        for jid, size, upload in ((a, (640, 480), wide), (b, (640, 854), tall)):
            code, headers, data = self.studio.get(f"/still/{jid}")
            self.assertEqual((code, headers["Content-Type"], jpeg_size(data)), (200, "image/jpeg", size))
            self.assertEqual(headers["Cache-Control"], PICTURE_CACHE)
            self.assertLess(len(data), len(upload) / 10)
        self.studio.wait([a, b])
        self.assertEqual(self.studio.get(f"/poster/{a}")[1]["Cache-Control"], PICTURE_CACHE)

    def test_reference_video_frame_kept_with_the_inputs(self):
        a = self.generate(task="ref2va", files={"ref_videos": [("clip.mp4", mp4())],
                                                 "ref_audio": [("voice.wav", b"RIFF0000WAVEfmt ")]})
        still = self.studio.runtime / "inputs" / "webui" / a / STILL
        self.assertFalse(still.exists())                                         # made on the first request
        with concurrent.futures.ThreadPoolExecutor(4) as pool:                   # several tabs at once
            got = list(pool.map(self.still, [a] * 4))
        code, ctype, data = got[0]
        self.assertEqual((code, ctype, data[:2], data[-2:]), (200, "image/jpeg", b"\xff\xd8", b"\xff\xd9"))
        self.assertEqual(got, [got[0]] * 4)
        self.assertEqual(sorted(p.name for p in still.parent.iterdir()), [STILL, "clip.mp4", "clip_ref.mp4", "voice.wav"])
        inode = still.stat().st_ino
        self.assertEqual(self.still(a)[2], data)
        self.assertEqual(still.stat().st_ino, inode)                             # made once
        self.assertEqual(self.studio.wait([a])[0]["status"], "done")
        listing = json.loads(self.studio.get(f"/api/inputs/{a}")[2])            # not an input for Remix
        self.assertEqual([i["name"] for zone in listing["inputs"].values() for i in zone], ["clip_ref.mp4", "voice.wav"])
        # Delete moves it to the trash with the inputs, Undo brings it back
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": a})[0], 200)
        self.assertEqual(self.still(a)[0], 404)
        self.assertFalse(still.parent.exists())
        self.assertEqual(len(list((self.studio.runtime / "trash" / a).rglob(STILL))), 1)
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": a})[0], 200)
        self.assertEqual(still.stat().st_ino, inode)                             # moved back, not made again
        self.assertEqual(self.still(a)[2], data)
        # a Remix reuses the video and gets a picture of its own
        b = self.generate(task="ref2va", remix_from=a, keep="ref_videos:0", seed=2)
        self.assertEqual(self.still(b)[:2], (200, "image/jpeg"))
        self.assertNotEqual((self.studio.runtime / "inputs" / "webui" / b / STILL).stat().st_ino, inode)
        self.studio.wait([b])

    def test_only_ids(self):
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", png())]})
        self.assertEqual(self.still(f"/still/{a}?v=1")[0], 200)
        for path in (f"/still/{'f' * 32}", f"/still/{a.upper()}", f"/still/{a}/", f"/still/{a}/0", f"/still/{a[:31]}",
                     "/still/", "/still/../../../etc/passwd", f"/still/..%2f{a}", f"/still/{a}%00"):
            self.assertEqual(self.still(path)[0], 404, path)
        self.studio.wait([a])


class CardPictureOldClips(StudioTest):
    """#30: clips from the history. The JPEG goes into the clip's own input directory. A clip without one (an old
    history) has no reference video frame, and its image goes out as it is; one whose file is gone has no picture;
    nothing is written next to their files."""

    def seed(self):
        self.outside = self.dir / "elsewhere"
        self.outside.mkdir()
        (self.outside / "clip_ref.mp4").write_bytes(mp4())
        (self.outside / "first.png").write_bytes(png())
        own = self.studio.runtime / "inputs" / "webui" / ("c" * 32)
        own.mkdir(parents=True)
        (own / "clip_ref.mp4").write_bytes(mp4())
        self.write_state([finished_job("a" * 32, created=1, task="ref2va",
                                       references=[{"type": "video", "path": str(self.outside / "clip_ref.mp4")}]),
                          finished_job("b" * 32, created=2, task="fl2va", first_frame=str(self.outside / "gone.png")),
                          finished_job("c" * 32, created=3, task="ref2va",
                                       references=[{"type": "video", "path": str(own / "clip_ref.mp4")}]),
                          finished_job("e" * 32, created=4, task="fl2va", first_frame=str(self.outside / "first.png"))])

    def test_no_picture_and_nothing_written(self):
        code, headers, _ = self.studio.get(f"/still/{'c' * 32}")
        self.assertEqual((code, headers["Content-Type"]), (200, "image/jpeg"))
        self.assertTrue((self.studio.runtime / "inputs" / "webui" / ("c" * 32) / STILL).is_file())
        for jid in ("a" * 32, "b" * 32):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(f"/still/{jid}")
            self.assertEqual(e.exception.code, 404, jid)
        code, headers, data = self.studio.get(f"/still/{'e' * 32}")
        self.assertEqual((code, headers["Content-Type"], data), (200, "image/png", png()))
        self.assertEqual(sorted(p.name for p in self.outside.iterdir()), ["clip_ref.mp4", "first.png"])
        for jid in ("a" * 32, "e" * 32):
            self.assertFalse((self.studio.runtime / "inputs" / "webui" / jid).exists())


class CardPictureWithoutJpeg(StudioTest):
    """#30: when ffmpeg can't make the card picture, an image goes out as it is (a reference video has none), and
    nothing is left behind."""

    def seed(self):
        bindir = self.dir / "bin"
        bindir.mkdir()
        (bindir / "ffmpeg").write_text('#!/bin/sh\n'
                                       'case " $* " in *"min(640,iw)"*) echo "cannot read it" >&2; exit 1;; esac\n'
                                       f'exec {shutil.which("ffmpeg")} "$@"\n')
        (bindir / "ffmpeg").chmod(0o755)
        self.studio.env.update(PATH=f"{bindir}:{os.environ['PATH']}")

    def test_the_image_as_it_is(self):
        first = png()
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", first)]})
        b = self.generate(task="ref2va", seed=2, files={"ref_videos": [("clip.mp4", mp4())]})
        for _ in range(2):                                                       # tried again, not remembered
            code, headers, data = self.studio.get(f"/still/{a}")
            self.assertEqual((code, headers["Content-Type"], data), (200, "image/png", first))
            self.assertEqual(headers["Cache-Control"], PICTURE_CACHE)
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(f"/still/{b}")
            self.assertEqual(e.exception.code, 404)
        inputs = self.studio.runtime / "inputs" / "webui"
        self.assertEqual(list(inputs.glob(f"*/{STILL}")) + list(inputs.glob(".still-*")), [])
        self.studio.wait([a, b])


class CardPictureDeleteRace(StudioTest):
    """#30: a Delete while a card picture is being made, from a reference video or an image. The JPEG is written under
    a temporary name outside the clip's input directory, so nothing of it goes to the trash with the inputs or comes
    back with Undo, and the image, now in the trash, is not served; one left by a server stopped part-way is removed
    at the next start."""

    def seed(self):
        self.inputs = self.studio.runtime / "inputs" / "webui"
        self.clips = {"c" * 32: "clip_ref.mp4", "d" * 32: "first.png"}
        for jid, name in self.clips.items():
            (self.inputs / jid).mkdir(parents=True)
            (self.inputs / jid / name).write_bytes(mp4() if name.endswith(".mp4") else png())
        (self.inputs / f".still-{'0' * 32}.part").write_bytes(b"\xff\xd8")     # from a server stopped part-way
        self.write_state([finished_job("c" * 32, created=1, task="ref2va", references=[
                              {"type": "video", "path": str(self.inputs / ("c" * 32) / "clip_ref.mp4")}]),
                          finished_job("d" * 32, created=2, task="fl2va",
                                       first_frame=str(self.inputs / ("d" * 32) / "first.png"))])
        # ffmpeg making the picture writes it, then waits until the test says go
        self.hold = self.dir / "hold"
        self.hold.mkdir()
        bindir = self.dir / "bin"
        bindir.mkdir()
        (bindir / "ffmpeg").write_text('#!/bin/sh\n'
                                       'case " $* " in *" mjpeg "*)\n'
                                       f'  {shutil.which("ffmpeg")} "$@" || exit $?\n'
                                       '  touch "$STILL_HOLD/written"; i=0\n'
                                       '  while [ ! -e "$STILL_HOLD/go" ] && [ $i -lt 400 ]; do\n'
                                       '    sleep 0.05; i=$((i+1))\n'
                                       '  done\n'
                                       '  exit 0;;\n'
                                       'esac\n'
                                       f'exec {shutil.which("ffmpeg")} "$@"\n')
        (bindir / "ffmpeg").chmod(0o755)
        self.studio.env.update(PATH=f"{bindir}:{os.environ['PATH']}", STILL_HOLD=str(self.hold))

    def still(self, jid):
        try:
            return self.studio.get(f"/still/{jid}")[0]
        except urllib.error.HTTPError as e:
            return e.code

    def files(self, d):
        return sorted(p.relative_to(d).name for p in d.rglob("*") if not p.is_dir())

    def test_delete_while_the_picture_is_made(self):
        self.assertEqual(list(self.inputs.glob(".still-*")), [])                # the stopped server's is gone
        for jid, name in self.clips.items():
            with self.subTest(name):
                for f in ("written", "go"):
                    (self.hold / f).unlink(missing_ok=True)
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    got = pool.submit(self.still, jid)
                    deadline = time.time() + 20
                    while not (self.hold / "written").exists():
                        self.assertLess(time.time(), deadline, "ffmpeg did not write the picture")
                        time.sleep(0.05)
                    self.assertEqual(self.studio.post("/api/delete", json_body={"id": jid})[0], 200)
                    (self.hold / "go").touch()
                    self.assertEqual(got.result(timeout=20), 404)                # deleted meanwhile: no picture
                self.assertEqual(self.files(self.studio.runtime / "trash" / jid), [name])
                self.assertEqual(list(self.inputs.glob(".still-*")), [])
                self.assertEqual(self.studio.post("/api/undelete", json_body={"id": jid})[0], 200)
                self.assertEqual(self.files(self.inputs / jid), [name])
                self.assertEqual(self.still(jid), 200)                          # made now, as usual
                self.assertEqual(self.files(self.inputs / jid), [STILL, name])
                self.assertEqual(list(self.inputs.glob(".still-*")), [])


class FfmpegAndTheTerminal(StudioTest):
    """The server's ffmpeg runs leave a terminal on its stdin alone (python3 tests/stub/studio.py, or the tests, run in
    one). ffmpeg puts its stdin terminal into raw mode, no echo, while it runs and restores it as it found it at exit,
    so runs at once (a page asking for several cards' pictures) could leave the terminal raw."""

    def seed(self):
        self.master, self.tty = os.openpty()
        self.studio.stdin = self.tty
        self.calls = self.dir / "ffmpeg-calls"
        bindir = self.dir / "bin"
        bindir.mkdir()
        (bindir / "ffmpeg").write_text('#!/bin/sh\n'
                                       'if [ -t 0 ]; then echo "tty $*"; else echo "no-tty $*"; fi >>"$FFMPEG_CALLS"\n'
                                       f'exec {shutil.which("ffmpeg")} "$@"\n')
        (bindir / "ffmpeg").chmod(0o755)
        self.studio.env.update(PATH=f"{bindir}:{os.environ['PATH']}", FFMPEG_CALLS=str(self.calls))

    def tearDown(self):
        super().tearDown()
        os.close(self.master)
        os.close(self.tty)

    def test_terminal_left_alone(self):
        ids = [self.generate(task="ref2va", seed=i, files={"ref_videos": [("clip.mp4", mp4())]}) for i in (1, 2)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:                  # each card's picture, several tabs
            codes = list(pool.map(lambda jid: self.studio.get(f"/still/{jid}")[0], ids * 4))
        self.assertEqual(codes, [200] * 8)
        self.assertEqual([j["status"] for j in self.studio.wait(ids)], ["done"] * 2)
        calls = self.calls.read_text().splitlines()
        for made, by in (("reference video", "libx264 -crf"), ("card picture", "mjpeg"), ("clip", "testsrc2"),
                         ("poster", "-ss 1")):
            self.assertTrue(any(by in c for c in calls), f"no ffmpeg run for the {made}")
        self.assertTrue(all(c.startswith("tty ") for c in calls))               # each one had the terminal
        self.assertEqual([c for c in calls if "-nostdin" not in c.split()], [])  # and was told not to use it
        lflag = termios.tcgetattr(self.tty)[3]
        self.assertEqual(lflag & (termios.ECHO | termios.ICANON), termios.ECHO | termios.ICANON)


PAGE = 24                                   # clips per gallery page (webui/server.py)


def history(n, start=1_700_000_000):
    """n finished jobs, oldest first, one minute apart."""
    return [finished_job(f"{i:032x}", created=start + 60 * i, seed=i) for i in range(n)]


class History(StudioTest):
    """#5: every job is kept (jobs.db); /api/state carries the active jobs and the newest page, /api/jobs the rest."""

    def seed(self):
        self.write_state(history(230))                 # more than the old 200-job cap

    def test_import_and_pages(self):
        self.assertTrue((self.studio.state_dir / "jobs.db").is_file())
        self.assertEqual(len(json.loads((self.studio.state_dir / "jobs.json").read_text())), 230)   # left as it was
        st = self.studio.state()
        self.assertEqual((len(st["jobs"]), st["done"], st["more"]), (24, 230, True))
        seen = [j["id"] for j in st["jobs"]]
        self.assertEqual(seen[0], f"{229:032x}")        # newest first
        before = min(j["created"] for j in st["jobs"])
        while True:
            page = json.loads(self.studio.get(f"/api/jobs?before={before}&limit=50")[2])
            seen += [j["id"] for j in page["jobs"]]
            if not page["more"]:
                break
            before = page["jobs"][-1]["created"]
        self.assertEqual(seen, [f"{i:032x}" for i in reversed(range(230))])   # every job once, in order
        for q in ("before=x", "limit=y"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(f"/api/jobs?{q}")
            self.assertEqual(e.exception.code, 400)

    def test_active_jobs_always_in_state_and_kept_across_restarts(self):
        jid = self.generate()
        st = self.studio.wait_for(lambda st: any(j["id"] == jid and j["status"] != "queued" for j in st["jobs"]))
        self.assertEqual(len(st["jobs"]), 25)           # the running job + the newest 24 finished
        self.studio.wait([jid])
        self.studio.stop()
        self.studio.start()
        st = self.studio.state()
        self.assertEqual(st["jobs"][0]["id"], jid)
        self.assertEqual((st["jobs"][0]["status"], st["done"]), ("done", 231))

    def test_running_job_marked_failed_after_restart(self):
        jid = self.generate(seconds=8)
        self.studio.wait_for(lambda st: any(j["id"] == jid and j["status"] != "queued" for j in st["jobs"]))
        self.studio.proc.kill()                         # no clean shutdown
        self.studio.proc.wait()
        self.studio.start()
        job = self.studio.job(jid)
        self.assertEqual((job["status"], job["error"]), ("failed", "server restarted"))


class HistoryImport(StudioTest):
    """#5: the jobs.json import copes with odd entries, is retried after a failure and happens only once."""

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text("[{ not json")

    def test_failed_import_retried_then_done_once(self):
        self.assertIn("could not import", self.studio.log.read_text())
        self.assertEqual(self.studio.state()["done"], 0)
        jid = self.generate()                                                 # a clip made in this run is kept
        self.studio.wait([jid])
        self.studio.stop()
        odd = [finished_job("1" * 32, created=50), finished_job("2" * 32, created="abc"),
               finished_job("3" * 32, created=None), finished_job("4" * 32, created=50),
               dict(finished_job("5" * 32), created=float("nan"))]
        odd[-1].pop("created")
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(odd + [{"no": "id"}]))
        self.studio.start()
        self.assertIn("imported 5 jobs", self.studio.log.read_text())
        jobs = sorted(self.studio.all_jobs().values(), key=lambda j: j["created"])
        self.assertEqual(len(jobs), 6)
        created = [j["created"] for j in jobs]
        self.assertTrue(all(a < b for a, b in zip(created, created[1:])), created)   # strictly increasing
        self.studio.stop()
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(history(3)))   # changed later: not imported
        self.studio.start()
        self.assertEqual(self.studio.state()["done"], 6)


class HistoryReimport(StudioTest):
    def seed(self):
        self.write_state(history(3))

    def test_documented_reimport(self):
        self.assertEqual(self.studio.state()["done"], 3)
        self.studio.stop()
        self.write_state(history(5))                                          # 2 clips made under 0.1 meanwhile
        subprocess.run([sys.executable, "-c", 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); '
                        'c.execute("PRAGMA user_version = 0"); c.commit()', str(self.studio.state_dir / "jobs.db")],
                       check=True)                                           # the command in TROUBLESHOOTING.md
        self.studio.start()
        self.assertEqual(self.studio.state()["done"], 5)


class HistoryOrder(StudioTest):
    """#5: a request that takes longer (a reference video being normalized) is ordered by when it was added, so
    paging (cursor = 'created') reaches every clip."""
    load_s, clip_s = 1, 0.1

    def test_slow_upload_does_not_hide_a_clip(self):
        import threading
        big = Path(tempfile.mkdtemp(prefix="lf-studio-big-")) / "big.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=24:d=12",
                        "-f", "lavfi", "-i", "sine=duration=12", "-shortest", "-pix_fmt", "yuv420p", str(big)],
                       check=True)
        slow = {}
        t = threading.Thread(target=lambda: slow.update(r=self.studio.generate(
            task="ref2va", files={"ref_videos": [("big.mp4", big.read_bytes())]})))
        t.start()
        time.sleep(0.4)
        ids = [self.generate(seed=i) for i in range(24)]                     # posted while the video is normalized
        t.join()
        shutil.rmtree(big.parent)
        self.assertEqual(slow["r"][0], 200, slow["r"])
        ids.append(slow["r"][1]["id"])
        self.studio.wait(ids, timeout=120)
        self.assertEqual(set(self.studio.all_jobs()), set(ids))
        jobs = sorted(self.studio.all_jobs().values(), key=lambda j: j["created"])
        self.assertEqual(jobs[-1]["id"], slow["r"][1]["id"])                 # added last, so newest


class Delete(StudioTest):
    """#2: Delete moves a clip's files to the trash and hides it; undelete puts everything back; old trash is purged."""
    env = {}

    def setUp(self):
        self.publish = Path(tempfile.mkdtemp(prefix="lf-studio-publish-"))
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.publish, ignore_errors=True)

    def seed(self):
        self.studio.env.update(LFSTUDIO_PUBLISH_DIR=str(self.publish), **self.env)

    def delete(self, jid, path="/api/delete"):
        return self.studio.post(path, json_body={"id": jid})

    def test_delete_and_undelete_fl2va(self):
        jid = self.generate(task="fl2va", files={"first_frame": [("first.png", png())]})
        job = self.studio.wait([jid])[0]
        clip_dir = Path(job["output"]).parent.parent
        paths = [clip_dir, Path(job["first_frame"]).parent, Path(job["published"]),
                 Path(job["published"]).with_suffix(".jpg")]
        self.assertTrue(all(p.exists() for p in paths), paths)
        self.assertEqual(self.delete(jid), (200, {"ok": True}))
        self.assertTrue(not any(p.exists() for p in paths))
        self.assertEqual(len(list((self.studio.runtime / "trash" / jid).iterdir())), 4)
        st = self.studio.state()
        self.assertEqual((st["jobs"], st["done"]), ([], 0))
        self.assertEqual(json.loads(self.studio.get("/api/jobs")[2])["jobs"], [])
        for url in (f"/video/{jid}", f"/poster/{jid}", f"/api/inputs/{jid}"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(url)
            self.assertEqual(e.exception.code, 404, url)
        code, body = self.studio.generate(task="fl2va", remix_from=jid, keep="first_frame")
        self.assertEqual(code, 400)
        code, body = self.delete(jid, "/api/undelete")
        self.assertEqual((code, body["job"]["id"], "deleted" in body["job"]), (200, jid, False))
        self.assertTrue(all(p.exists() for p in paths))
        self.assertFalse((self.studio.runtime / "trash" / jid).exists())
        self.assertEqual(self.studio.get(f"/video/{jid}")[0], 200)
        self.assertEqual(self.studio.state()["done"], 1)

    def test_deleted_state_survives_restart(self):
        jid = self.generate()
        self.studio.wait([jid])
        self.delete(jid)
        self.studio.stop()
        self.studio.start()
        self.assertEqual(self.studio.state()["jobs"], [])
        self.assertEqual(self.delete(jid, "/api/undelete")[0], 200)
        self.assertEqual(self.studio.state()["done"], 1)

    def test_refusals(self):
        self.assertEqual(self.delete("f" * 32)[0], 404)
        self.assertEqual(self.studio.post("/api/delete", json_body=None, fields={"id": "x"})[0], 400)
        jid = self.generate(seconds=8)
        self.studio.wait_for(lambda st: any(j["id"] == jid for j in st["jobs"]))
        self.assertEqual(self.delete(jid)[0], 409)                         # queued or running
        self.studio.wait([jid])
        self.assertEqual(self.delete(jid)[0], 200)
        self.assertEqual(self.delete(jid)[0], 200)                         # twice is fine

    def test_old_layout_clip_moves_just_its_files(self):
        old = self.publish / "20260923-113128_t2va_portrait_seed11_4f969b.mp4"
        old.write_bytes(b"mp4")
        old.with_suffix(".jpg").write_bytes(b"jpg")
        self.studio.stop()
        (self.studio.state_dir / "jobs.db").unlink()
        self.write_state([finished_job("a" * 32, output=str(old))])
        self.studio.start()
        self.assertEqual(self.delete("a" * 32)[0], 200)
        self.assertEqual(sorted(p.name for p in (self.studio.runtime / "trash" / ("a" * 32)).iterdir()),
                         ["0-" + old.name, "1-" + old.with_suffix(".jpg").name])
        self.assertTrue(self.publish.is_dir())                            # the folder itself stays

    def test_refused_request_leaves_no_input_directory(self):
        before = set((self.studio.runtime / "inputs" / "webui").iterdir())
        self.assertEqual(self.studio.generate(task="fl2va", files={"first_frame": [("x.txt", b"no")]})[0], 400)
        self.assertEqual(set((self.studio.runtime / "inputs" / "webui").iterdir()), before)


class DeleteEdgeCases(Delete):
    """#2, from review: failures half-way, a symlinked runtime root, failed runs, remix after delete."""
    test_delete_and_undelete_fl2va = test_deleted_state_survives_restart = test_refusals = None
    test_old_layout_clip_moves_just_its_files = test_refused_request_leaves_no_input_directory = None

    def clip(self, **kw):
        jid = self.generate(**kw)
        return self.studio.wait([jid])[0]

    def test_undelete_with_the_publish_folder_read_only(self):
        job = self.clip()
        self.delete(job["id"])
        self.publish.chmod(0o555)
        try:
            code, body = self.delete(job["id"], "/api/undelete")
        finally:
            self.publish.chmod(0o755)
        self.assertEqual(code, 200, body)
        self.assertNotIn("published", body["job"])                         # restored without its share copy
        self.assertEqual(self.studio.get(f"/video/{job['id']}")[0], 200)
        self.assertEqual(len(list((self.studio.runtime / "trash" / job["id"]).iterdir())), 2)   # mp4 + jpg wait

    def test_failed_delete_moves_everything_back(self):
        job = self.clip()
        self.publish.chmod(0o555)                                           # the share copy can't be moved
        try:
            code, body = self.delete(job["id"])
        finally:
            self.publish.chmod(0o755)
        self.assertEqual(code, 500, body)
        self.assertTrue(Path(job["output"]).exists() and Path(job["published"]).exists())
        self.assertFalse((self.studio.runtime / "trash" / job["id"]).exists())
        self.assertEqual(self.studio.state()["done"], 1)

    def test_failed_run_directory_goes_too(self):
        job = self.clip(prompt="STUB_FAIL")
        self.assertEqual(job["status"], "failed")
        dirs = [Path(d) for d in job["case_dirs"]]
        self.assertTrue(dirs and all(d.is_dir() for d in dirs))
        self.assertEqual(self.delete(job["id"])[0], 200)
        self.assertFalse(any(d.exists() for d in dirs))

    def test_remix_after_the_source_was_deleted(self):
        job = self.clip(task="fl2va", files={"first_frame": [("first.png", png())]})
        self.delete(job["id"])
        jid = self.generate(task="fl2va", remix_from=job["id"], keep="first_frame:0")   # form still had its files
        self.assertEqual(self.studio.wait([jid])[0]["status"], "done")
        self.assertEqual(self.delete(job["id"], "/api/undelete")[0], 200)  # and the source can still come back

    def test_leftovers_in_the_trash_are_swept(self):
        old = self.studio.runtime / "trash" / "leftover"
        old.mkdir(parents=True)
        os.utime(old, (1, 1))
        self.studio.stop()
        self.studio.start()
        self.assertFalse(old.exists())


class ReleaseReview(StudioTest):
    """From the whole-release review: features meeting each other, and the upgrade of real 0.1 data."""
    load_s, clip_s, preload = 2, 0.5, True

    def setUp(self):
        self.share = Path(tempfile.mkdtemp(prefix="lf-studio-share-", dir="/dev/shm"))     # another filesystem
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.share, ignore_errors=True)

    def seed(self):
        self.studio.env.update(STUB_CLOSE_S="2", LFSTUDIO_PUBLISH_DIR=str(self.share))
        self.old = self.share / "20260923-113128_t2va_portrait_seed11_4f969b.mp4"   # a 0.1 clip: its only copy
        self.old.write_bytes(os.urandom(600_000))
        self.old.with_suffix(".jpg").write_bytes(b"jpg")
        self.write_state([finished_job("a" * 32, output=str(self.old), created=1),
                          finished_job("b" * 32, output=str(self.old), created=2),       # listed twice
                          finished_job("c" * 32, task=None, created=3), {"id": "../../x", "task": "t2va"},
                          finished_job("d" * 32, created=4)])

    def test_import_skips_bad_entries_and_state_works(self):
        self.assertIn("skipping 2 entries", self.studio.log.read_text())
        self.assertEqual(self.studio.state()["done"], 3)

    def test_shared_file_stays_for_the_other_entry(self):
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": "a" * 32})[0], 200)
        self.assertTrue(self.old.exists())                                    # b still uses it
        self.assertEqual(self.studio.get(f"/video/{'b' * 32}")[0], 200)

    def test_only_copy_on_an_unmounted_share_waits_for_it(self):
        self.studio.post("/api/delete", json_body={"id": "b" * 32})          # b first: a still lists the file
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": "a" * 32})[0], 200)
        self.assertFalse(self.old.exists())
        hidden = self.share.with_name(self.share.name + ".off")
        self.share.rename(hidden)                                             # the share is not mounted
        try:
            code, body = self.studio.post("/api/undelete", json_body={"id": "a" * 32})
            self.assertEqual(code, 500, body)
            self.assertFalse(self.share.exists())                             # not created on the local disk
        finally:
            hidden.rename(self.share)
        code, body = self.studio.post("/api/undelete", json_body={"id": "a" * 32})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.old.stat().st_size, 600_000)

    def test_failed_copy_back_leaves_no_partial_file(self):
        for jid in ("b" * 32, "a" * 32):
            self.studio.post("/api/delete", json_body={"id": jid})
        subprocess.run(["prlimit", f"--pid={self.studio.proc.pid}", "--fsize=300000:unlimited"], check=True)
        code, body = self.studio.post("/api/undelete", json_body={"id": "a" * 32})
        self.assertEqual(code, 500, body)
        self.assertFalse(self.old.exists())                                   # no truncated file in the share
        self.assertEqual([p.name for p in self.share.iterdir() if p.name.startswith(".")], [])
        subprocess.run(["prlimit", f"--pid={self.studio.proc.pid}", "--fsize=unlimited:unlimited"], check=True)
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": "a" * 32})[0], 200)
        self.assertEqual(self.old.stat().st_size, 600_000)

    def warm_up_source(self):
        jid = self.generate(task="fl2va", seconds=6, files={"first_frame": [("f.png", png())]})
        self.studio.wait([jid])
        self.studio.wait_for(lambda st: st["session"]["state"] == "ready", timeout=30)
        return jid

    def test_delete_during_a_requested_warm_up(self):
        src = self.warm_up_source()
        self.studio.wait_for(lambda st: st["session"]["state"] == "ready", timeout=30)
        self.assertEqual(self.studio.post("/api/unload")[0], 200)     # else the loaded fl2va session only changes
        self.studio.post("/api/load", json_body={"task": "fl2va", "orientation": "portrait", "seconds": 5})   # shape
        self.studio.wait_for(lambda st: st["session"].get("preloading") and st["session"]["state"] == "loading"
                             and st["session"].get("loading_key") == ["fl2va", "portrait", 5], timeout=20)
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": src})[0], 200)   # its inputs move away
        st = self.studio.wait_for(lambda st: st["session"]["state"] in ("ready", "idle")
                                  and not st["session"].get("switching_to"), timeout=30)
        self.assertEqual(st["session"]["key"], ["fl2va", "portrait", 5])     # the warm-up had its own links
        self.assertNotIn("preload stopped", self.studio.log.read_text().split("load requested")[1])


class FixReview(ReleaseReview):
    """From the review of the release fixes: a pruned clip's only copy, shutdown and crashes during a move, and a copy
    whose source can't be removed."""
    test_import_skips_bad_entries_and_state_works = test_shared_file_stays_for_the_other_entry = None
    test_only_copy_on_an_unmounted_share_waits_for_it = test_failed_copy_back_leaves_no_partial_file = None
    test_delete_during_a_requested_warm_up = None

    def clip(self):
        jid = self.generate(seconds=6)
        return self.studio.wait([jid])[0]

    def test_pruned_clip_waits_for_its_publish_copy(self):
        job = self.clip()
        shutil.rmtree(Path(job["output"]).parent.parent.parent)               # session directories pruned
        self.assertEqual(self.studio.get(f"/video/{job['id']}")[0], 200)     # plays from the publish copy
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": job["id"]})[0], 200)
        off = self.share.with_name(self.share.name + ".off")
        self.share.rename(off)
        try:
            self.assertEqual(self.studio.post("/api/undelete", json_body={"id": job["id"]})[0], 500)
        finally:
            off.rename(self.share)
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": job["id"]})[0], 200)
        self.assertEqual(self.studio.get(f"/video/{job['id']}")[0], 200)

    def big_publish_copy(self, job, size=1 << 30):
        with open(job["published"], "r+b") as f:                              # a large clip, sparse
            f.truncate(size)

    def test_shutdown_waits_for_a_move(self):
        job = self.clip()
        self.big_publish_copy(job)
        t = __import__("threading").Thread(target=lambda: self.studio.post("/api/delete", json_body={"id": job["id"]}))
        t.start()
        time.sleep(0.15)
        self.studio.stop()                                                     # SIGTERM during the copy
        t.join()
        self.studio.start()
        self.assertNotIn(job["id"], [j["id"] for j in self.studio.state()["jobs"]])   # the delete finished
        self.assertFalse(Path(job["published"]).exists())
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": job["id"]})[0], 200)
        self.assertEqual(Path(job["published"]).stat().st_size, 1 << 30)

    def test_crash_during_a_delete_is_put_back(self):
        job = self.clip()
        self.big_publish_copy(job, 4 << 30)

        def delete():
            with __import__("contextlib").suppress(OSError):                  # the server dies under it
                self.studio.post("/api/delete", json_body={"id": job["id"]})
        t = __import__("threading").Thread(target=delete)
        t.start()
        time.sleep(0.3)
        self.studio.proc.kill()                                                # no clean shutdown
        self.studio.proc.wait()
        t.join()
        self.studio.start()
        self.assertIn("put back the files of an unfinished delete", self.studio.log.read_text())
        self.assertEqual(self.studio.get(f"/video/{job['id']}")[0], 200)
        self.assertTrue(Path(job["output"]).exists() and Path(job["published"]).exists())
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": job["id"]})[0], 200)   # works again

    def test_copy_whose_source_cant_be_removed_is_undone(self):
        job = self.clip()
        os.chmod(self.share, 0o555)                                            # copied out, but not removable
        try:
            code, body = self.studio.post("/api/delete", json_body={"id": job["id"]})
        finally:
            os.chmod(self.share, 0o755)
        self.assertEqual(code, 500, body)
        self.assertTrue(Path(job["output"]).exists() and Path(job["published"]).exists())
        trash = self.studio.runtime / "trash" / job["id"]
        self.assertFalse(trash.exists() and any(trash.rglob("*.mp4")))         # no second copy left behind


class SymlinkedRuntime(StudioTest):
    def setUp(self):
        self.real = Path(tempfile.mkdtemp(prefix="lf-studio-real-"))
        link = Path(tempfile.mkdtemp(prefix="lf-studio-link-")) / "root"
        link.symlink_to(self.real)
        self.link = link
        self.dir = link
        self.studio = StubStudio(link, load_s=1, clip_s=0.2, preload=False)
        self.studio.start()

    def tearDown(self):
        self.studio.stop()
        shutil.rmtree(self.real, ignore_errors=True)
        shutil.rmtree(self.link.parent, ignore_errors=True)

    def test_clip_directory_moved(self):
        jid = self.generate()
        case = Path(self.studio.wait([jid])[0]["output"]).parent.parent
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": jid})[0], 200)
        self.assertFalse(case.exists())
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": jid})[0], 200)
        self.assertTrue(case.is_dir())


class BadTrashDays(unittest.TestCase):
    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-bad-"))
        try:
            studio = StubStudio(d, env={"LFSTUDIO_TRASH_DAYS": "inf"})
            with self.assertRaises(RuntimeError) as e:
                studio.start()
            self.assertIn("LFSTUDIO_TRASH_DAYS", str(e.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class BadStage2Audio(unittest.TestCase):
    """SOL_H3_STAGE2_AUDIO (a patch setting) is checked at start, not when every clip fails."""

    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-bad-"))
        try:
            with self.assertRaises(RuntimeError) as e:
                StubStudio(d / "a", env={"SOL_H3_STAGE2_AUDIO": "noised"}).start()
            self.assertIn("SOL_H3_STAGE2_AUDIO must be one of", str(e.exception))
            studio = StubStudio(d / "b", env={"SOL_H3_STAGE2_AUDIO": "JOINT"}).start()
            studio.stop()
            with self.assertRaises(RuntimeError) as e:                      # the same for SOL_H3_STATIC_SHAPES
                StubStudio(d / "c", env={"SOL_H3_STATIC_SHAPES": "yes"}).start()
            self.assertIn("SOL_H3_STATIC_SHAPES must be 1 or 0", str(e.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class BadStage2Keyframe(unittest.TestCase):
    """SOL_H3_STAGE2_KEYFRAME (a patch setting, #57; pin by default on the DGX Spark) is checked at start too: a bad
    value stops the studio instead of failing every Stage-2 load."""

    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-bad-"))
        try:
            for i, value in enumerate(("first", "on", "1")):
                with self.assertRaises(RuntimeError) as e:
                    StubStudio(d / f"a{i}", env={"SOL_H3_STAGE2_KEYFRAME": value}).start()
                self.assertIn(f"SOL_H3_STAGE2_KEYFRAME must be one of ('off', 'pin', 'draft'), not {value!r}",
                              str(e.exception))
            for i, value in enumerate(("PIN", "draft", " ", "off")):       # empty is the default (pin), as unset
                studio = StubStudio(d / f"b{i}", env={"SOL_H3_STAGE2_KEYFRAME": value}).start()
                studio.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class BadDynamicTextLength(unittest.TestCase):
    """SOL_H3_DYNAMIC_TEXT_LENGTH (a patch setting, #41; on by default since 0.4) is checked at start too: a bad value
    stops the studio instead of failing every Stage-1 load once the transformer is in memory."""

    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-bad-"))
        try:
            for i, value in enumerate(("on", "2", "true")):
                with self.assertRaises(RuntimeError) as e:
                    StubStudio(d / f"a{i}", env={"SOL_H3_DYNAMIC_TEXT_LENGTH": value}).start()
                self.assertIn(f"SOL_H3_DYNAMIC_TEXT_LENGTH must be 0 or 1, not {value!r}", str(e.exception))
            for i, value in enumerate(("0", " ", "1")):                    # the opt-out, empty (the default), set
                studio = StubStudio(d / f"b{i}", env={"SOL_H3_DYNAMIC_TEXT_LENGTH": value}).start()
                studio.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)


def older_patch_package(directory, version):
    """The stub runtime as an older patch has it. 0.5's (0.5.0 and 0.5.1) has every setting but SOL_H3_STAGE2_KEYFRAME
    (it never pins the keyframe, as off does). 0.3's has SOL_H3_STAGE2_AUDIO but not SOL_H3_PROMPT_TOKENS,
    SOL_H3_STATIC_SHAPES or SOL_H3_DYNAMIC_TEXT_LENGTH; 0.2's has none of them. Such a patch tokenizes as plain does
    and compiles as SOL_H3_STATIC_SHAPES=0 and SOL_H3_DYNAMIC_TEXT_LENGTH=0 do, and 0.2's refines as joint does."""
    package = Path(directory) / f"package-{version}"
    shutil.copytree(Path(__file__).resolve().parent / "stub" / "runtime", package / "runtime",
                    ignore=shutil.ignore_patterns("__pycache__"))
    missing = {"0.5": "stage2_keyframe_mode",
               "0.3": "prompt_tokens_mode, compile_dynamic, dynamic_text_length, stage2_keyframe_mode",
               "0.2": "stage2_audio_mode, prompt_tokens_mode, compile_dynamic, dynamic_text_length, "
                      "stage2_keyframe_mode"}[version]
    with (package / "runtime" / "config.py").open("a") as f:
        f.write(f"\ndel {missing}   # the patch of {version} has no such setting\n")
    return package


# What an older patch is asked for, and how the studio and the launcher answer: None = starts; otherwise the start of
# the refusal, which ends "needs a newer patch than the one applied in <package>; run install/10-sana.sh" (and, for
# the keyframe pin's default, " (or set SOL_H3_STAGE2_KEYFRAME=off)": KEYFRAME_DEFAULT_REFUSAL). An empty value is
# the default, as unset. On the DGX Spark (these cases) SOL_H3_STAGE2_KEYFRAME's default is pin since 0.5.2.
KEYFRAME_DEFAULT_REFUSAL = "SOL_H3_STAGE2_KEYFRAME=pin (the default since 0.5.2)"
OLDER_PATCH_CASES = (
    ("0.3", {}, "SOL_H3_PROMPT_TOKENS=minimax (the default since 0.4)"),        # a git pull without install/10-sana.sh
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "minimax"}, "SOL_H3_PROMPT_TOKENS=minimax"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain"}, "SOL_H3_STATIC_SHAPES=1 (the default since 0.4)"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "1"}, "SOL_H3_STATIC_SHAPES=1"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain", "SOL_H3_STATIC_SHAPES": " 0"},
     "SOL_H3_DYNAMIC_TEXT_LENGTH=1 (the default since 0.4)"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "1"},
     "SOL_H3_DYNAMIC_TEXT_LENGTH=1"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain", "SOL_H3_STATIC_SHAPES": " 0", "SOL_H3_DYNAMIC_TEXT_LENGTH": " 0"},
     KEYFRAME_DEFAULT_REFUSAL),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain", "SOL_H3_STATIC_SHAPES": " 0", "SOL_H3_DYNAMIC_TEXT_LENGTH": " 0",
             "SOL_H3_STAGE2_KEYFRAME": "off"}, None),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0"},
     "SOL_H3_STAGE2_AUDIO=frozen (the default since 0.3)"),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0",
             "SOL_H3_STAGE2_AUDIO": "frozen"}, "SOL_H3_STAGE2_AUDIO=frozen"),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0",
             "SOL_H3_STAGE2_AUDIO": "JOINT", "SOL_H3_STAGE2_KEYFRAME": "OFF"}, None),
    ("0.5", {}, KEYFRAME_DEFAULT_REFUSAL),                 # a git pull without install/10-sana.sh: 0.5 never pins
    ("0.5", {"SOL_H3_STAGE2_KEYFRAME": " Off"}, None),
    ("0.5", {"SOL_H3_STAGE2_KEYFRAME": "pin"}, "SOL_H3_STAGE2_KEYFRAME=pin"),
    ("0.5", {"SOL_H3_STAGE2_KEYFRAME": "Draft"}, "SOL_H3_STAGE2_KEYFRAME=Draft"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0",
             "SOL_H3_STAGE2_KEYFRAME": "pin"}, "SOL_H3_STAGE2_KEYFRAME=pin"),
)


def older_patch_settings(settings):
    return {"SOL_H3_PROMPT_TOKENS": "", "SOL_H3_STAGE2_AUDIO": "", "SOL_H3_STATIC_SHAPES": "",
            "SOL_H3_DYNAMIC_TEXT_LENGTH": "", "SOL_H3_STAGE2_KEYFRAME": "", **settings}


class BadPromptTokens(unittest.TestCase):
    """SOL_H3_PROMPT_TOKENS (a patch setting, #38) is checked at start like SOL_H3_STAGE2_AUDIO."""

    def test_older_patch(self):
        """A checkout with an older patch would ignore the setting: the studio refuses any value but what that
        patch does anyway, the default (minimax) included. Without this, a git pull without install/10-sana.sh
        started fine and every clip came out plain (#38)."""
        d = Path(tempfile.mkdtemp(prefix="lf-studio-old-"))
        try:
            packages = {version: older_patch_package(d, version) for version in ("0.5", "0.3", "0.2")}
            for i, (version, settings, refused) in enumerate(OLDER_PATCH_CASES):
                package = packages[version]
                studio = StubStudio(d / f"case-{i}", env=older_patch_settings(settings), package=package)
                if refused is None:
                    studio.start()
                    studio.stop()
                    continue
                with self.assertRaises(RuntimeError) as e:
                    studio.start()
                self.assertIn(f"{refused} needs a newer patch than the one applied in {package.resolve()}; "
                              "run install/10-sana.sh", str(e.exception))
                opt_out = "; run install/10-sana.sh (or set SOL_H3_STAGE2_KEYFRAME=off)"
                if refused == KEYFRAME_DEFAULT_REFUSAL:                 # the default: off starts too
                    self.assertIn(opt_out, str(e.exception))
                else:
                    self.assertNotIn("(or set", str(e.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-bad-"))
        try:
            with self.assertRaises(RuntimeError) as e:
                StubStudio(d / "a", env={"SOL_H3_PROMPT_TOKENS": "special"}).start()
            self.assertIn("SOL_H3_PROMPT_TOKENS must be one of", str(e.exception))
            for i, value in enumerate(("MiniMax", "plain", "")):
                studio = StubStudio(d / f"b{i}", env={"SOL_H3_PROMPT_TOKENS": value}).start()
                studio.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class LauncherSettingsCheck(unittest.TestCase):
    """bin/lf-studio --check runs the patch's settings checks (SOL_H3_STAGE2_AUDIO, SOL_H3_PROMPT_TOKENS,
    SOL_H3_STATIC_SHAPES, SOL_H3_DYNAMIC_TEXT_LENGTH, SOL_H3_STAGE2_KEYFRAME) in the package, here the stub runtime, and
    refuses a bad value with its message, or a value an older patch can't give (#38)."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-check-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        (self.d / "sana/models/minimax_h3").mkdir(parents=True)
        (self.d / "sana/models/minimax_h3/Sol-H3-Spark").symlink_to(Path(__file__).resolve().parent / "stub")
        for task in ("t2va", "fl2va", "ref2va"):
            (self.d / f"paths-{task}.json").write_text("{}")

    def check(self, launcher=None, **settings):
        """bin/lf-studio --check (or `launcher`, a copy elsewhere) with `settings`: (exit status, its output)."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_"))}
        env.update(LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_SANA=str(self.d / "sana"),
                   LFSTUDIO_PYTHON=sys.executable, LFSTUDIO_PATHS=str(self.d / "paths-{task}.json"),
                   SOL_H3_SPARK_RUNTIME_ROOT=str(self.d / "runtime"), LFSTUDIO_STATE_DIR=str(self.d / "state"),
                   LFSTUDIO_HOST="127.0.0.1", LFSTUDIO_PORT=str(free_port()),
                   LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0", PYTHONDONTWRITEBYTECODE="1")
        env.update(settings)
        launcher = launcher or Path(__file__).resolve().parents[1] / "bin/lf-studio"
        run = subprocess.run(["bash", str(launcher), "--check"], env=env, capture_output=True, text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    def test_settings(self):
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(self.check(SOL_H3_PROMPT_TOKENS="Plain", SOL_H3_STAGE2_AUDIO="joint")[0], 0)
        for name, value in (("SOL_H3_PROMPT_TOKENS", "special"), ("SOL_H3_STAGE2_AUDIO", "noised")):
            code, out = self.check(**{name: value})
            self.assertEqual(code, 1, out)
            self.assertIn(f"REFUSED - {name} must be one of", out)
        self.assertEqual(self.check(SOL_H3_STATIC_SHAPES="0")[0], 0)
        code, out = self.check(SOL_H3_STATIC_SHAPES="yes")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - SOL_H3_STATIC_SHAPES must be 1 or 0", out)
        for value in ("0", "1"):
            self.assertEqual(self.check(SOL_H3_DYNAMIC_TEXT_LENGTH=value)[0], 0, value)
        for value in ("on", "true"):
            code, out = self.check(SOL_H3_DYNAMIC_TEXT_LENGTH=value)
            self.assertEqual(code, 1, out)
            self.assertIn(f"REFUSED - SOL_H3_DYNAMIC_TEXT_LENGTH must be 0 or 1, not {value!r}", out)
        for value in ("off", "Pin", "draft", " "):                         # " ": the default, pin
            self.assertEqual(self.check(SOL_H3_STAGE2_KEYFRAME=value)[0], 0, value)
        code, out = self.check(SOL_H3_STAGE2_KEYFRAME="first")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - SOL_H3_STAGE2_KEYFRAME must be one of ('off', 'pin', 'draft'), not 'first'", out)

    def test_api_setting(self):
        """LFSTUDIO_API (#58): 1, 0, or unset or empty (on); any other value is refused with the server's words."""
        for value in ("1", "0", ""):
            self.assertEqual(self.check(LFSTUDIO_API=value)[0], 0, value)
        for value in ("2", "on", "off", "true"):
            code, out = self.check(LFSTUDIO_API=value)
            self.assertEqual(code, 1, out)
            self.assertIn(f"REFUSED - LFSTUDIO_API={value}: must be 1 (on, the default) or 0 (off) (see "
                          "bin/env.example)", out)
        envfile = self.d / "env"                                     # from the settings file, as the others
        envfile.write_text("LFSTUDIO_API=yes\n")
        code, out = self.check(LFSTUDIO_ENV=str(envfile))
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - LFSTUDIO_API=yes: must be", out)

    def test_older_patch(self):
        """A checkout with an older patch (no settings checks, or one of them) passes only when every setting, set or
        the default, asks for what that patch does anyway."""
        for i, (version, settings, refused) in enumerate(OLDER_PATCH_CASES):
            sana = self.d / f"sana-{i}"
            (sana / "models/minimax_h3").mkdir(parents=True)
            (sana / "models/minimax_h3/Sol-H3-Spark").symlink_to(older_patch_package(sana, version))
            code, out = self.check(LFSTUDIO_SANA=str(sana), **older_patch_settings(settings))
            if refused is None:
                self.assertEqual(code, 0, out)
                continue
            self.assertEqual(code, 1, out)
            self.assertIn(f"REFUSED - {refused} needs a newer patch than the one applied in", out)
            self.assertIn("run install/10-sana.sh", out)
            self.assertNotIn("(see bin/env.example)", out)                  # exit 4 (a newer patch), not 3
            if refused == KEYFRAME_DEFAULT_REFUSAL:
                self.assertIn("; run install/10-sana.sh (or set SOL_H3_STAGE2_KEYFRAME=off)", out)
            else:
                self.assertNotIn("(or set", out)


class TrashPurge(Delete):
    env = {"LFSTUDIO_TRASH_DAYS": "0.00001"}                                  # under a second

    def test_purged_at_start(self):
        jid = self.generate()
        self.studio.wait([jid])
        self.delete(jid)
        time.sleep(1.2)
        self.studio.stop()
        self.studio.start()
        self.assertFalse((self.studio.runtime / "trash" / jid).exists())
        self.assertEqual(self.delete(jid, "/api/undelete")[0], 404)        # gone from the history too
        self.assertIn("emptied the trash", self.studio.log.read_text())

    test_delete_and_undelete_fl2va = test_deleted_state_survives_restart = test_refusals = None
    test_old_layout_clip_moves_just_its_files = test_refused_request_leaves_no_input_directory = None


class LoadSelected(StudioTest):
    """#3: POST /api/load warms up chosen settings on request: it replaces what is loaded or warming up, never runs
    while clips are queued or running, and needs an earlier clip of the mode for fl2va/ref2va. With LFSTUDIO_RESHAPE=0,
    so that another length or format of the loaded mode closes the session too (ReshapeLoadSelected: the default)."""
    load_s, clip_s, preload = 2, 1, True

    def seed(self):
        self.studio.env["LFSTUDIO_RESHAPE"] = "0"
        self.write_state([finished_job()])

    def load(self, task="t2va", orientation="portrait", seconds=10):
        return self.studio.post("/api/load", json_body={"task": task, "orientation": orientation, "seconds": seconds})

    def ready(self, key):
        return self.studio.wait_for(lambda st: st["session"]["key"] == key and st["session"]["state"] == "ready")

    def test_switch_loaded_settings(self):
        self.ready(["t2va", "landscape", 5])                                 # the start-up warm-up
        self.assertEqual(self.load(), (200, {"ok": True}))
        st = self.ready(["t2va", "portrait", 10])
        self.assertEqual(st["jobs"][0]["id"], "0" * 32)                      # no job was created for it
        log = self.studio.log.read_text()
        self.assertIn("closing session ('t2va', 'landscape', 5)", log)
        self.assertIn("preloading ('t2va', 'portrait', 10)", log)
        self.assertEqual(self.load()[1].get("already"), True)               # same settings again: nothing to do
        jid = self.generate(orientation="portrait", seconds=10)
        self.assertFalse(self.studio.wait([jid])[0].get("includes_model_load"))   # it used the loaded models

    def test_replaces_a_warm_up_for_other_settings(self):
        self.studio.wait_for(lambda st: st["session"].get("preloading"))
        self.assertEqual(self.load()[0], 200)
        self.ready(["t2va", "portrait", 10])
        self.assertIn("aborting preload", self.studio.log.read_text())

    def test_refused_while_busy_and_for_modes_without_a_clip(self):
        self.ready(["t2va", "landscape", 5])
        st = self.studio.state()
        self.assertEqual(st["warmable"], ["t2va"])
        code, body = self.load(task="fl2va")
        self.assertEqual(code, 409)
        self.assertIn("no earlier fl2va clip", body["error"])
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va"})[0], 400)
        for seconds, error in ((3, GONE), (4, GONE), (9, "duration must be 5, 6, 7, 8, 10, 12 or 15 seconds")):
            self.assertEqual(self.load(seconds=seconds), (400, {"error": error}))     # 3 and 4 s went after 0.5.0
        jid = self.generate(task="fl2va", seconds=8, files={"first_frame": [("f.png", png())]})
        self.studio.wait_for(lambda st: any(j["id"] == jid and j["status"] != "queued" for j in st["jobs"]))
        self.assertEqual(self.load()[0], 409)                                # a clip is running
        self.studio.wait([jid])
        self.assertIn("fl2va", self.studio.state()["warmable"])
        self.assertEqual(self.load(task="fl2va", orientation="landscape", seconds=5)[0], 200)
        self.ready(["fl2va", "landscape", 5])


class LoadDuringClose(StudioTest):
    """#3, from review: while the old models close for a requested load (~40 s for real, 3 s here), a clip, an
    Unload or a newer request takes precedence, and no pipeline is ever left open."""
    load_s, clip_s, preload = 2, 1, True

    def seed(self):
        self.studio.env.update(STUB_CLOSE_S="3", LFSTUDIO_RESHAPE="0")     # length/format changes close the session too
        self.write_state([finished_job()])

    def load(self, orientation="portrait", seconds=10):
        return self.studio.post("/api/load", json_body={"task": "t2va", "orientation": orientation, "seconds": seconds})

    def closing(self):
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 5]
                             and st["session"]["state"] == "ready", timeout=30)
        self.assertEqual(self.load()[0], 200)
        st = self.studio.wait_for(lambda st: st["session"]["state"] == "unloading")
        self.assertEqual(st["session"]["switching_to"], ["t2va", "portrait", 10])

    def open_pipelines(self):
        roots = (self.studio.runtime / "outputs" / "webui").iterdir()
        return [r.name for r in roots if (r / "STARTED").exists() and not (r / "CLOSED").exists()]

    def no_warm_up_for_the_request(self):
        self.assertNotIn("preloading ('t2va', 'portrait', 10)", self.studio.log.read_text())

    def test_same_settings_clip_loads_its_own(self):
        self.closing()
        job = self.studio.wait([self.generate(orientation="portrait", seconds=10)], timeout=60)[0]
        self.assertTrue(job["includes_model_load"])
        self.no_warm_up_for_the_request()
        self.assertEqual(len(self.open_pipelines()), 1)                     # the clip's own session

    def test_other_settings_clip_cancels_the_request(self):
        self.closing()
        job = self.studio.wait([self.generate(seconds=6)], timeout=60)[0]
        self.assertEqual((job["status"], job["includes_model_load"]), ("done", True))
        self.no_warm_up_for_the_request()
        self.assertEqual(self.studio.state()["session"]["key"], ["t2va", "landscape", 6])

    def test_unload_cancels_the_request(self):
        self.closing()
        self.assertEqual(self.studio.post("/api/unload")[0], 200)            # waits for the close in progress
        time.sleep(1)
        st = self.studio.state()["session"]
        self.assertEqual((st["state"], st["key"], st["switching_to"]), ("idle", None, None))
        self.no_warm_up_for_the_request()
        self.assertEqual(self.open_pipelines(), [])

    def test_newer_request_wins(self):
        self.closing()
        self.assertEqual(self.load(orientation="landscape", seconds=6)[0], 200)
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 6]
                             and st["session"]["state"] == "ready", timeout=30)
        self.no_warm_up_for_the_request()
        self.assertEqual(len(self.open_pipelines()), 1)

    def test_back_to_the_settings_being_closed(self):
        self.closing()
        self.assertEqual(self.load(orientation="landscape", seconds=5), (200, {"ok": True}))
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 5]
                             and st["session"]["state"] == "ready", timeout=30)
        self.assertEqual(len(self.open_pipelines()), 1)

    def test_refusals(self):
        self.studio.wait_for(lambda st: st["session"]["state"] == "ready", timeout=30)
        bad = self.studio.post("/api/load", json_body=None, fields=None)
        self.assertEqual(bad[0], 400)
        req = self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait", "seconds": 1e999})
        self.assertEqual(req[0], 400)
        (self.studio.root / "paths-ref2va.json").unlink()
        code, body = self.studio.post("/api/load", json_body={"task": "ref2va", "orientation": "portrait", "seconds": 5})
        self.assertEqual(code, 400)
        self.assertIn("manifest", body["error"])
        a = self.generate(seconds=8)
        b = self.generate(seconds=8, seed=2)                                  # queued behind a
        self.studio.wait_for(lambda st: [j["status"] for j in st["jobs"] if j["id"] in (a, b)].count("queued") == 1)
        self.assertEqual(self.load()[0], 409)


class Reshape(StudioTest):
    """#45: a clip with the loaded task but another length or orientation keeps the models loaded: the session changes
    its shape (Pipeline.reshape), then renders the clip. Another task closes the session and loads the models again."""
    load_s, clip_s = 2, 1

    def sessions(self):
        return sorted(d.name for d in (self.studio.runtime / "outputs" / "webui").iterdir() if d.is_dir())

    def reshapes(self, session):
        log = self.studio.runtime / "outputs" / "webui" / session / "RESHAPES"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def test_same_task_keeps_the_models(self):
        self.assertTrue(self.studio.state()["reshape"])
        plan = [("landscape", 5), ("portrait", 10), ("landscape", 5), ("landscape", 6)]
        jobs = [self.studio.wait([self.generate(orientation=o, seconds=s, seed=n)], timeout=60)[0]
                for n, (o, s) in enumerate(plan)]
        self.assertEqual([j["status"] for j in jobs], ["done"] * 4)
        self.assertTrue(jobs[0]["includes_model_load"])
        for job, (orientation, seconds), before in zip(jobs[1:], plan[1:], plan):
            self.assertNotIn("includes_model_load", job)
            self.assertEqual(job["reshaped_from"], ["t2va", *before])
            w, h = (336, 192) if orientation == "landscape" else (192, 336)
            self.assertEqual(Path(job["output"]).name, f"refined_{w}x{h}_{seconds * 24 + 1}f.mp4")   # its own shape
        [session] = self.sessions()                                          # one pipeline for all four
        self.assertEqual([r["to"]["seconds"] for r in self.reshapes(session)], [10, 5, 6])
        st = self.studio.state()["session"]
        self.assertEqual((st["key"], st["state"]), (["t2va", "landscape", 6], "ready"))
        log = self.studio.log.read_text()
        self.assertNotIn("closing session", log)
        self.assertIn("reshaping session ('t2va', 'landscape', 5) -> ('t2va', 'portrait', 10)", log)
        again = self.studio.wait([self.generate(seconds=6, seed=9)], timeout=30)[0]
        self.assertNotIn("reshaped_from", again)                             # the same settings: a warm clip

    def test_another_task_reloads(self):
        self.studio.wait([self.generate()], timeout=30)
        job = self.studio.wait([self.generate(task="fl2va", orientation="portrait", seconds=6,
                                              files={"first_frame": [("f.png", png())]})], timeout=60)[0]
        self.assertEqual((job["status"], job["includes_model_load"]), ("done", True))
        self.assertNotIn("reshaped_from", job)
        self.assertEqual(len(self.sessions()), 2)
        self.assertIn("closing session ('t2va', 'landscape', 5)", self.studio.log.read_text())

    def test_a_failed_change_closes_the_session_and_the_retry_loads(self):
        self.studio.stop()
        self.studio.env["STUB_RESHAPE_FAIL"] = "1"
        self.studio.start()
        self.studio.wait([self.generate()], timeout=30)
        job = self.studio.wait([self.generate(seconds=8)], timeout=60)[0]
        self.assertEqual((job["status"], job["includes_model_load"]), ("done", True))
        self.assertIn("STUB_RESHAPE_FAIL", job["first_error"])
        self.assertNotIn("reshaped_from", job)                                # the retry loaded the models
        self.assertNotIn("reshape_s", job)
        self.assertEqual(len(self.sessions()), 2)
        self.assertIn("closing session ('t2va', 'landscape', 5) - length/format change failed",
                      self.studio.log.read_text())

    def test_cancel_during_the_change(self):
        self.studio.stop()
        self.studio.env["STUB_RESHAPE_S"] = "6"
        self.studio.start()
        self.studio.wait([self.generate()], timeout=30)
        b = self.generate(orientation="portrait", seed=2)
        self.studio.wait_for(lambda st: any(j["id"] == b and j.get("reshaped_from") for j in st["jobs"]))
        deadline = time.time() + 10
        while not worker_groups(self.studio.proc.pid) and time.time() < deadline:
            time.sleep(0.05)
        t0 = time.time()
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": b})[0], 200)
        job = self.studio.wait([b], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertLess(time.time() - t0, 4, "stopped right away, not after the change")
        self.assertEqual(self.studio.state()["session"]["state"], "idle")    # its workers were stopped
        c = self.studio.wait([self.generate(orientation="portrait", seed=3)], timeout=30)[0]
        self.assertEqual((c["status"], c["includes_model_load"]), ("done", True))


class ReshapeDuringUnload(StudioTest):
    """From the review: a clip at another length queued while Unload closes the session must not change the shape of
    the pipeline being closed (on the real runtime that could start a resident prompt encoder next to the next load);
    it waits for the close and loads its own models."""
    load_s, clip_s = 2, 1

    def seed(self):
        self.studio.env["STUB_CLOSE_S"] = "3"

    def test_unload_then_another_length(self):
        import threading
        self.studio.wait([self.generate()], timeout=30)
        unload = threading.Thread(target=self.studio.post, args=("/api/unload",))
        unload.start()
        self.studio.wait_for(lambda st: st["session"]["state"] == "unloading", timeout=10)
        job = self.studio.wait([self.generate(seconds=8, seed=2)], timeout=60)[0]
        unload.join(30)
        self.assertEqual((job["status"], job.get("includes_model_load")), ("done", True))
        self.assertNotIn("reshaped_from", job)
        self.assertNotIn("first_error", job)                                 # it never touched the closing pipeline
        self.assertNotIn("reshaping session", self.studio.log.read_text())
        roots = sorted((self.studio.runtime / "outputs" / "webui").iterdir())
        self.assertEqual([(r / "STARTED").exists() and not (r / "CLOSED").exists() for r in roots], [False, True])
        self.assertEqual(self.studio.state()["session"]["key"], ["t2va", "landscape", 8])


class ReshapeOff(StudioTest):
    """LFSTUDIO_RESHAPE=0: any other task, length or orientation closes the session and loads the models again."""

    def seed(self):
        self.studio.env["LFSTUDIO_RESHAPE"] = "0"

    def test_a_new_length_reloads(self):
        self.assertFalse(self.studio.state()["reshape"])
        self.studio.wait([self.generate()], timeout=30)
        job = self.studio.wait([self.generate(seconds=8)], timeout=60)[0]
        self.assertEqual((job["status"], job["includes_model_load"]), ("done", True))
        self.assertNotIn("reshaped_from", job)
        self.assertIn("closing session ('t2va', 'landscape', 5) - switching to ('t2va', 'landscape', 8)",
                      self.studio.log.read_text())


class ReshapeLoadSelected(StudioTest):
    """With the task loaded, "Load selected now" for another length or orientation changes the session's shape:
    no close, no warm-up clip."""
    load_s, clip_s, preload = 2, 1, True

    def seed(self):
        self.studio.env["STUB_RESHAPE_S"] = "2"
        self.write_state([finished_job()])

    def test_changes_the_shape_only(self):
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 5]
                             and st["session"]["state"] == "ready", timeout=30)
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait",
                                                                  "seconds": 10}), (200, {"ok": True}))
        st = self.studio.wait_for(lambda st: st["session"]["state"] == "reshaping", timeout=10)   # not "loading"
        self.assertEqual(st["session"]["loading_key"], ["t2va", "portrait", 10])
        st = self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "portrait", 10]
                                  and st["session"]["state"] == "ready", timeout=30)
        self.assertIsNone(st["session"]["switching_to"])
        log = self.studio.log.read_text()
        self.assertIn("reshaping session ('t2va', 'landscape', 5) -> ('t2va', 'portrait', 10) - as asked", log)
        self.assertNotIn("preloading ('t2va', 'portrait', 10)", log)
        self.assertNotIn("closing session", log)
        job = self.studio.wait([self.generate(orientation="portrait", seconds=10)], timeout=30)[0]
        self.assertEqual(job["status"], "done")
        self.assertNotIn("includes_model_load", job)
        self.assertNotIn("reshaped_from", job)                               # already that shape
        self.assertEqual(len(list((self.studio.runtime / "outputs" / "webui").iterdir())), 1)


class ReshapeLoadSelectedUnload(StudioTest):
    """Unload while "Load selected now" changes the shape: the change fails on the closing pipeline. The studio says
    so in one line, with no traceback and no second close; Unload's close ends the session."""
    load_s, clip_s, preload = 2, 1, True

    def seed(self):
        self.studio.env.update(STUB_RESHAPE_S="1", STUB_CLOSE_S="3")     # the change fails while the close runs
        self.write_state([finished_job()])

    def test_unload_during_the_change(self):
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 5]
                             and st["session"]["state"] == "ready", timeout=30)
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait",
                                                                  "seconds": 10})[0], 200)
        self.studio.wait_for(lambda st: st["session"]["state"] == "reshaping", timeout=10)
        self.assertEqual(self.studio.post("/api/unload")[0], 200)              # answers once the close is done
        st = self.studio.state()["session"]
        self.assertEqual((st["key"], st["state"], st.get("loading_key")), (None, "idle", None))
        log = self.studio.log.read_text()
        self.assertIn("length/format change stopped: the session was closed meanwhile - RuntimeError: The pipeline "
                      "is closing", log)
        self.assertNotIn("Traceback", log)
        self.assertEqual(log.count("closing session"), 1)                      # Unload's
        [root] = (self.studio.runtime / "outputs" / "webui").iterdir()
        self.assertTrue((root / "CLOSED").exists())
        self.assertFalse((root / "RESHAPES").exists())


class ReshapeWarmUp(StudioTest):
    """The warm-up's takeover rule counts a clip of the warm-up's task as one for the same models: early in the warm-up
    it stops the warm-up and loads them itself, later it waits and changes the shape. A queued one keeps the warm-up
    alive when a clip for another task arrives."""
    load_s, clip_s, preload = 4, 1, True
    times = {"t2va/landscape/5": 4}                                    # no warm time: window = half the load, 2 s

    def seed(self):
        self.write_state([finished_job()], self.times)

    def warming(self):
        return self.studio.wait_for(lambda s: s["session"].get("preloading"))["session"]

    def test_early_takes_over(self):
        self.warming()
        job = self.studio.wait([self.generate(orientation="portrait")])[0]
        self.assertEqual((job["status"], job.get("includes_model_load")), ("done", True))
        self.assertIn("aborting preload: a job for the same models arrived early", self.studio.log.read_text())


class ReshapeWarmUpLate(ReshapeWarmUp):
    times = {"t2va/landscape/5": 4, "warm/t2va/landscape/5": 6}       # window = 6 - 5 = 1 s

    def test_early_takes_over(self):
        pass

    def test_late_waits_and_changes_the_shape(self):
        s = self.warming()
        time.sleep(max(0, s["takeover_until"] - s["since"]) + 0.8)
        waiting = self.generate(orientation="portrait", seconds=6)            # the warm-up's task: it waits
        other = self.generate(task="fl2va", files={"first_frame": [("f.png", png())]})   # kept for `waiting`
        jw, jo = self.studio.wait([waiting, other], timeout=60)
        self.assertNotIn("aborting preload", self.studio.log.read_text())
        self.assertEqual((jw["status"], jw.get("includes_model_load")), ("done", None))
        self.assertEqual(jw["reshaped_from"], ["t2va", "landscape", 5])
        self.assertEqual((jo["status"], jo["includes_model_load"]), ("done", True))


class PipelineNotMade(StudioTest):
    def test_session_goes_back_to_idle(self):
        """A pipeline that can't be created (here: its session directory) fails the clip; the session is idle again,
        not "loading" forever."""
        sessions = self.studio.runtime / "outputs" / "webui"
        sessions.mkdir(parents=True, exist_ok=True)
        sessions.chmod(0o500)
        try:
            job = self.studio.wait([self.generate()])[0]
            self.assertEqual(job["status"], "failed")
            s = self.studio.state()["session"]
            self.assertEqual((s["state"], s.get("loading_key")), ("idle", None))
        finally:
            sessions.chmod(0o700)
        self.assertEqual(self.studio.wait([self.generate()])[0]["status"], "done")


class WarmUp(StudioTest):
    """The start-up warm-up and the takeover rule (docs/PERFORMANCE.md): a same-settings job submitted before
    takeover_until stops the warm-up and loads the models itself; later it waits; other settings stop it. With
    LFSTUDIO_RESHAPE=0, so that another length or format counts as other settings (ReshapeWarmUp: the default)."""
    load_s, clip_s, preload = 4, 1, True
    times = {"t2va/landscape/5": 4}

    def seed(self):
        self.studio.env["LFSTUDIO_RESHAPE"] = "0"
        self.write_state([finished_job()], self.times)

    def warming(self):
        return self.studio.wait_for(lambda s: s["session"].get("preloading"))["session"]

    def test_without_warm_time_window_is_half_the_load(self):
        s = self.warming()
        self.assertAlmostEqual(s["takeover_until"] - s["since"], 2, delta=0.5)
        job = self.studio.wait([self.generate()])[0]
        self.assertTrue(job.get("includes_model_load"), "early same-settings job should take over")
        self.assertIn("aborting preload", self.studio.log.read_text())

    def test_session_shows_the_estimates_while_warming(self):
        s = self.warming()
        self.assertEqual((s["load_eta"], s["warm_s"]), (4, self.times.get("warm/t2va/landscape/5")))
        s = self.studio.wait_for(lambda s: s["session"]["state"] == "ready", timeout=20)["session"]
        self.assertEqual((s["load_eta"], s["warm_s"], s["takeover_until"]), (None, None, None))

    def test_other_settings_stop_the_warm_up(self):
        self.warming()
        job = self.studio.wait([self.generate(orientation="portrait")])[0]
        self.assertEqual(job["status"], "done")
        self.assertTrue(job.get("includes_model_load"))
        self.assertIn("aborting preload", self.studio.log.read_text())


class WarmUpSkipsDeleted(StudioTest):
    preload = True

    def seed(self):
        self.write_state([finished_job("1" * 32, orientation="portrait", created=1),
                          finished_job("2" * 32, created=2, deleted=3, trashed={})])

    def test_warm_up_uses_the_newest_clip_not_deleted(self):
        s = self.studio.wait_for(lambda s: s["session"].get("loading_key"))["session"]
        self.assertEqual(s["loading_key"], ["t2va", "portrait", 5])


class OldLengths(StudioTest):
    """3 and 4 s went after 0.5.0: clips made at those lengths stay in the history. They list, play, download and give
    their files to Remix as before; the remix itself takes a length offered. The Upscale refuses them (5 s only)."""

    def seed(self):
        self.old = self.dir / "old"
        self.old.mkdir()
        for seconds in (3, 4):
            (self.old / f"{seconds}.mp4").write_bytes(mp4(seconds))
        (self.old / "first.png").write_bytes(png())
        self.write_state([finished_job("3" * 32, seconds=3, output=str(self.old / "3.mp4"), created=1),
                          finished_job("4" * 32, task="fl2va", orientation="portrait", seconds=4, created=2,
                                       first_frame=str(self.old / "first.png"), output=str(self.old / "4.mp4"))])

    def test_listed_played_and_remixed(self):
        self.assertEqual({j["id"]: j["seconds"] for j in self.studio.state()["jobs"]}, {"3" * 32: 3, "4" * 32: 4})
        for jid, seconds in (("3" * 32, 3), ("4" * 32, 4)):
            code, headers, data = self.studio.get(f"/video/{jid}")
            self.assertEqual((code, data), (200, (self.old / f"{seconds}.mp4").read_bytes()))
            self.assertIn(f"_{seconds}s_seed7_", headers["Content-Disposition"])
            self.assertEqual(self.studio.get(f"/poster/{jid}")[0], 200)
            rec = json.loads(self.studio.get(f"/api/jobs/{jid}")[2])
            self.assertEqual((rec["state"], rec["seconds"]), ("done", seconds))
        listing = json.loads(self.studio.get(f"/api/inputs/{'4' * 32}")[2])
        self.assertEqual(([i["name"] for i in listing["inputs"]["first_frame"]], listing["missing"]), (["first.png"], []))
        code, body = self.studio.generate(task="fl2va", orientation="portrait", seconds=4, remix_from="4" * 32,
                                          keep="first_frame:0")
        self.assertEqual((code, body), (400, {"error": f"ValueError: {GONE}"}))
        jid = self.generate(task="fl2va", orientation="portrait", seconds=5, remix_from="4" * 32, keep="first_frame:0")
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual((job["status"], job["seconds"]), ("done", 5), job.get("error"))
        self.assertEqual(Path(job["first_frame"]).stat().st_ino, (self.old / "first.png").stat().st_ino)
        self.assertEqual(self.studio.job("4" * 32)["seconds"], 4)          # the old clip is as it was
        # the experimental Upscale (on in the stub studio) takes 5 s clips only: the old 3 s t2va clip is refused
        self.assertEqual(self.studio.job("3" * 32)["upscalable"], {"ok": False, "why": UPSCALE_LENGTH})
        self.assertEqual(self.studio.upscale("3" * 32), (400, {"error": UPSCALE_LENGTH}))


class OldLengthWarmUp(StudioTest):
    """A 0.5.0 state started by this server: the start-up warm-up after a 4 s clip (the last finished) loads 5 s, as
    its Remix on the page does, and a 3 s clip that was queued or running when 0.5.0 stopped reads failed ("server
    restarted") as any other, keeps its length and is not run or warmed up for."""
    preload = True

    def seed(self):
        self.write_state([finished_job("1" * 32, seconds=10, created=1), finished_job("2" * 32, seconds=4, created=2),
                          dict(finished_job("3" * 32, seconds=3, created=3), status="queued"),
                          dict(finished_job("5" * 32, seconds=3, created=4), status="generating", started=4)])

    def test_warm_up_at_5_s(self):
        s = self.studio.wait_for(lambda s: s["session"].get("loading_key"))
        self.assertEqual(s["session"]["loading_key"], ["t2va", "landscape", 5])
        jobs = {j["id"]: j for j in s["jobs"]}
        for jid in ("3" * 32, "5" * 32):
            self.assertEqual((jobs[jid]["status"], jobs[jid].get("error"), jobs[jid]["seconds"]),
                             ("failed", "server restarted", 3))
        self.studio.wait_for(lambda s: s["session"]["key"] == ["t2va", "landscape", 5] and s["session"]["state"] == "ready")
        self.assertIn("preloading ('t2va', 'landscape', 5)", self.studio.log.read_text())
        sessions = [d.name for d in (self.studio.runtime / "outputs" / "webui").iterdir() if d.is_dir()]
        self.assertTrue(sessions)
        self.assertEqual([n for n in sessions if not re.search(r"-t2va-landscape-5s(-\d+)?$", n)], [])   # no 3 s one


class WarmUpWithWarmTime(WarmUp):
    times = {"t2va/landscape/5": 4, "warm/t2va/landscape/5": 6}     # window = 6 - 5 = 1 s

    def test_without_warm_time_window_is_half_the_load(self):
        pass

    def test_late_same_settings_job_waits(self):
        s = self.warming()
        self.assertAlmostEqual(s["takeover_until"] - s["since"], 1, delta=0.5)
        time.sleep(max(0, s["takeover_until"] - s["since"]) + 0.8)
        job = self.studio.wait([self.generate()])[0]
        self.assertEqual(job["status"], "done")
        self.assertFalse(job.get("includes_model_load"), "a late same-settings job should wait for the warm-up")
        self.assertNotIn("aborting preload", self.studio.log.read_text())

    def test_cancelling_the_clip_the_warm_up_was_kept_for_stops_it(self):
        s = self.warming()
        time.sleep(max(0, s["takeover_until"] - s["since"]) + 0.5)
        waiting = self.generate()                                  # same settings, late: waits for the warm-up
        other = self.generate(orientation="portrait")              # other settings: the warm-up is kept for `waiting`
        self.assertNotIn("aborting preload", self.studio.log.read_text())
        code, body = self.studio.post("/api/cancel", json_body={"id": waiting})
        self.assertEqual((code, body["job"]["status"]), (200, "cancelled"))
        self.assertIn("aborting preload: the clip it was kept for was cancelled", self.studio.log.read_text())
        job = self.studio.wait([other])[0]
        self.assertEqual(job["status"], "done")
        self.assertTrue(job.get("includes_model_load"))


def worker_groups(pid):
    """The live processes `pid` started in a process group of their own: a stub studio's pipeline workers."""
    found = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                state, ppid, pgrp = (d / "stat").read_text().rsplit(")", 1)[1].split()[:3]
            except (OSError, ValueError):
                continue
            if int(ppid) == pid and int(pgrp) == int(d.name) and state != "Z":
                found.append(int(d.name))
    return found


def idle_values():
    """spark-switchboard's busy probe: the statuses it counts as idle (integrations/spark-switchboard/)."""
    toml = (Path(__file__).resolve().parents[1] / "integrations/spark-switchboard/lf-studio.toml").read_text()
    return json.loads(re.search(r"^idle_values\s*=\s*(\[[^\]]*\])", toml, re.M).group(1))


class CancelTest(StudioTest):
    def cancel(self, jid):
        return self.studio.post("/api/cancel", json_body={"id": jid})

    def until(self, jid, predicate, timeout=30):
        """Wait until the job with this id satisfies predicate; returns it."""
        st = self.studio.wait_for(lambda st: any(j["id"] == jid and predicate(j) for j in st["jobs"]), timeout)
        return next(j for j in st["jobs"] if j["id"] == jid)

    def running(self, jid, status):
        """Wait until the job is in this status with a worker running; returns the worker process groups."""
        self.until(jid, lambda j: j["status"] == status)
        deadline = time.time() + 10
        while not (groups := worker_groups(self.studio.proc.pid)) and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(groups, "a worker runs")
        return groups

    def open_pipelines(self):
        roots = (self.studio.runtime / "outputs" / "webui").iterdir()
        return [r.name for r in roots if (r / "STARTED").exists() and not (r / "CLOSED").exists()]

    def stopped(self, jid, since, within=3, retries=0):
        """The job ends cancelled soon after Cancel, without an error or another retry, its workers gone and the
        session closed; returns it."""
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled", job)
        self.assertLess(time.time() - since, within, "stopped right away, not after the work")
        for key in ("error", "output", "published", "cancel_requested", "saving"):
            self.assertNotIn(key, job)
        self.assertEqual(worker_groups(self.studio.proc.pid), [])
        self.assertEqual(self.open_pipelines(), [])
        log = self.studio.log.read_text()
        self.assertIn("clip cancelled", log)
        self.assertEqual(log.count(f"retrying job {jid}"), retries)
        session = self.studio.state()["session"]
        self.assertEqual((session["key"], session["state"]), (None, "idle"))
        return job


class Cancel(CancelTest):
    """#20: Cancel takes a queued clip out of the queue, or stops a running one (its workers get SIGTERM, the next
    clip loads the models again); either way the card stays as cancelled, for Remix and Delete."""
    load_s, clip_s = 6, 5

    def test_queued_never_runs_and_the_queue_carries_on(self):
        a = self.generate()
        b = self.generate(seed=2, task="fl2va", files={"first_frame": [("f.png", png())]})
        c = self.generate(seed=3)
        self.running(a, "loading models + generating")
        code, body = self.cancel(b)
        self.assertEqual((code, body["job"]["status"]), (200, "cancelled"))
        self.assertEqual(self.cancel(b), (409, {"error": "the clip is already cancelled"}))
        ja, jb, jc = self.studio.wait([a, b, c], timeout=60)
        self.assertEqual([j["status"] for j in (ja, jb, jc)], ["done", "cancelled", "done"])
        self.assertFalse(jc.get("includes_model_load"))                     # the session was left alone
        self.assertNotIn("started", jb)
        self.assertNotIn("case_dirs", jb)
        self.assertTrue(Path(jb["first_frame"]).is_file())                  # its input stays, for Remix
        self.assertNotIn("cancelling job", self.studio.log.read_text())     # nothing to stop

    def test_while_loading(self):
        self.studio.stop()
        self.studio.env["STUB_LATE_S"] = "2"            # the loading pass starts a second worker 2 s in
        self.studio.start()
        a = self.generate(task="fl2va", files={"first_frame": [("f.png", png())]})
        self.running(a, "loading models + generating")
        t0 = time.time()
        code, body = self.cancel(a)
        self.assertEqual((code, body["job"]["cancel_requested"]), (200, True))
        job = self.stopped(a, t0, within=4.5)            # the pass waits for the second worker (2 s in, 6 s if
                                                         # it ran): it is stopped as well
        self.assertEqual(self.studio.log.read_text().count("cancel: sent SIGTERM"), 2)
        self.assertTrue(Path(job["first_frame"]).is_file())
        b = self.studio.wait([self.generate(seed=2)], timeout=30)[0]
        self.assertEqual((b["status"], b["includes_model_load"]), ("done", True))

    def test_while_the_old_models_close(self):
        self.studio.stop()
        self.studio.env["STUB_CLOSE_S"] = "3"
        self.studio.start()
        self.studio.wait([self.generate()], timeout=30)
        b = self.generate(task="fl2va", seed=2, files={"first_frame": [("f.png", png())]})   # another mode: the
        self.studio.wait_for(lambda st: st["session"]["state"] == "unloading")              # loaded session closes
        self.assertEqual(self.studio.job(b)["status"], "starting")
        self.assertEqual(self.cancel(b)[0], 200)
        job = self.studio.wait([b], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("case_dirs", job)                       # nothing of its own ran: no session for it
        sessions = (self.studio.runtime / "outputs" / "webui").iterdir()
        self.assertEqual([d.name for d in sessions if "fl2va" in d.name], [])
        self.assertNotIn("cancel: sent SIGTERM", self.studio.log.read_text())
        self.assertEqual(self.studio.state()["session"]["state"], "idle")

    def test_next_clip_loads_in_the_same_second(self):
        time.sleep(1 - time.time() % 1 + 0.05)          # early in a second: both sessions start within it
        a = self.generate()
        b = self.generate(seed=2)                        # same settings, queued
        self.running(a, "loading models + generating")
        self.assertEqual(self.cancel(a)[0], 200)
        ja, jb = self.studio.wait([a, b], timeout=30)
        self.assertEqual((ja["status"], jb["status"], jb.get("first_error")), ("cancelled", "done", None))
        self.assertEqual(len(list((self.studio.runtime / "outputs" / "webui").iterdir())), 2)   # its own directory

    def test_while_generating(self):
        a = self.studio.wait([self.generate()], timeout=30)[0]
        self.assertEqual(a["status"], "done")
        b = self.generate(seed=2)
        self.running(b, "generating")
        t0 = time.time()
        self.assertEqual(self.cancel(b)[0], 200)
        self.assertEqual(self.cancel(b)[0], 200)                            # twice: it is being stopped already
        job = self.stopped(b, t0)
        self.assertNotIn("includes_model_load", job)
        c = self.studio.wait([self.generate(seed=3)], timeout=30)[0]
        self.assertEqual((c["status"], c["includes_model_load"]), ("done", True))   # the models load again
        self.assertEqual(self.studio.get(f"/video/{a['id']}")[0], 200)

    def test_during_the_retry_after_a_worker_error(self):
        a = self.generate(prompt="STUB_FAIL_ONCE")
        self.until(a, lambda j: j.get("first_error"), timeout=30)
        self.running(a, "loading models + generating")
        t0 = time.time()
        self.assertEqual(self.cancel(a)[0], 200)
        job = self.stopped(a, t0, retries=1)
        self.assertIn("STUB_FAIL", job["first_error"])                      # the first error stays on record
        b = self.studio.wait([self.generate(seed=2)], timeout=30)[0]
        self.assertEqual(b["status"], "done")

    def test_remix_of_cancelled_clips(self):
        frames = {"first_frame": [("first.png", png())], "last_frame": [("last.png", png(40))]}
        a = self.generate(task="fl2va", files=frames)                       # cancelled while loading
        b = self.generate(task="fl2va", seed=2, files=frames)               # cancelled while queued
        self.running(a, "loading models + generating")
        self.assertEqual(self.cancel(b)[0], 200)
        self.assertEqual(self.cancel(a)[0], 200)
        self.studio.wait([a, b])
        for src in (a, b):
            listing = json.loads(self.studio.get(f"/api/inputs/{src}")[2])
            self.assertEqual(listing["missing"], [])
            self.assertEqual({z: [i["name"] for i in v] for z, v in listing["inputs"].items()},
                             {"first_frame": ["first.png"], "last_frame": ["last.png"]})
        old = self.studio.job(a)
        jid = self.generate(task="fl2va", seconds=6, remix_from=a, keep="first_frame:0,last_frame:0")
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual(job["status"], "done", job.get("error"))
        for key in ("first_frame", "last_frame"):
            self.assertEqual(Path(job[key]).stat().st_ino, Path(old[key]).stat().st_ino)   # reused, not uploaded

    def test_delete_and_undo(self):
        self.studio.wait([self.generate()], timeout=30)
        b = self.generate(task="fl2va", seed=2, files={"first_frame": [("f.png", png())]})
        self.running(b, "loading models + generating")                      # another mode: a new session
        self.assertEqual(self.cancel(b)[0], 200)
        job = self.studio.wait([b])[0]
        paths = [Path(d) for d in job["case_dirs"]] + [Path(job["first_frame"]).parent]
        self.assertTrue(all(p.exists() for p in paths), paths)             # the run's directory and its inputs
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": b}), (200, {"ok": True}))
        self.assertFalse(any(p.exists() for p in paths))
        self.assertNotIn(b, [j["id"] for j in self.studio.state()["jobs"]])
        self.assertEqual(self.cancel(b)[0], 404)                             # deleted
        code, body = self.studio.post("/api/undelete", json_body={"id": b})
        self.assertEqual((code, body["job"]["status"]), (200, "cancelled"))
        self.assertTrue(all(p.exists() for p in paths))


class CancelRefusals(CancelTest):
    load_s, clip_s = 1, 0.5

    def test_finished_unknown_and_bad_requests(self):
        done = self.generate()
        failed = self.generate(prompt="STUB_FAIL", seed=2)
        self.studio.wait([done, failed], timeout=30)
        self.assertEqual(self.cancel(done), (409, {"error": "the clip has already finished"}))
        self.assertEqual(self.cancel(failed), (409, {"error": "the clip has already failed"}))
        self.assertEqual(self.cancel("f" * 32)[0], 404)
        self.assertEqual(self.studio.post("/api/cancel", fields={"id": done})[0], 400)
        self.assertEqual(self.studio.post("/api/cancel", json_body=[done])[0], 400)
        self.assertEqual(self.studio.job(done)["status"], "done")


class CancelState(CancelTest):
    """#20: a cancelled clip is finished everywhere: the poll's lists, the busy checks, switchboard's busy probe and
    the start-up check (it stays cancelled after a restart, even one in the middle of stopping it)."""
    load_s, clip_s = 4, 4

    def seed(self):
        self.write_state(history(PAGE))

    def test_listed_as_finished_and_not_busy(self):
        a = self.generate()
        b = self.generate(seed=2)
        self.running(a, "loading models + generating")
        self.assertEqual(self.cancel(b)[0], 200)
        st = self.studio.state()
        self.assertEqual((len(st["jobs"]), st["more"]), (PAGE + 1, True))   # a + the newest page, b among them
        self.assertIn(b, [j["id"] for j in st["jobs"]])
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait",
                                                                  "seconds": 5})[0], 409)   # a still runs
        self.assertEqual(self.cancel(a)[0], 200)
        self.studio.wait([a, b])
        st = self.studio.state()
        self.assertEqual((len(st["jobs"]), st["done"]), (PAGE, PAGE))       # finished; not counted as made
        self.assertTrue(all(j["status"] in idle_values() for j in st["jobs"]), "switchboard sees it idle")
        self.assertIn(a, self.studio.all_jobs())
        self.assertEqual(self.studio.post("/api/unload")[0], 200)
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait",
                                                                  "seconds": 5})[0], 200)

    def test_stays_cancelled_after_a_restart(self):
        self.studio.stop()
        self.studio.env["STUB_CLOSE_S"] = "20"                              # the stopped session closes slowly
        self.studio.start()
        a = self.generate()
        b = self.generate(seed=2)
        self.running(a, "loading models + generating")
        self.assertEqual(self.cancel(b)[0], 200)
        self.assertEqual(self.cancel(a)[0], 200)
        self.studio.wait_for(lambda st: st["session"]["state"] == "unloading")   # a's session is closing
        self.assertEqual(self.studio.job(a)["status"], "loading models + generating")
        self.studio.proc.kill()                                             # no clean shutdown
        self.studio.proc.wait()
        self.studio.start()
        for jid in (a, b):
            job = self.studio.job(jid)
            self.assertEqual(job["status"], "cancelled", job)
            self.assertNotIn("error", job)
            self.assertNotIn("cancel_requested", job)
        self.studio.stop()                                                  # and again, cleanly
        self.studio.start()
        self.assertEqual([self.studio.job(j)["status"] for j in (a, b)], ["cancelled", "cancelled"])


class CancelRace(CancelTest):
    """#20: a clip that finishes just as Cancel arrives: one of the two wins, never both. Cancel first: the clip is
    cancelled even though it was made (no output, session closed). The clip first: Cancel answers 409 while it is
    saved (poster, publish copy), and it is done."""
    load_s, clip_s = 1, 1

    def setUp(self):
        self.publish = Path(tempfile.mkdtemp(prefix="lf-studio-publish-"))
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.publish, ignore_errors=True)

    def seed(self):
        # ffmpeg making a poster (after the clip is accepted) waits 2 s, with a marker file meanwhile
        bindir = self.dir / "bin"
        bindir.mkdir()
        (bindir / "ffmpeg").write_text('#!/bin/sh\n'
                                       'case " $* " in *" -frames:v "*) touch "$POSTER_MARK"; sleep 2;; esac\n'
                                       f'exec {shutil.which("ffmpeg")} "$@"\n')
        (bindir / "ffmpeg").chmod(0o755)
        self.poster = self.dir / "poster-started"
        self.studio.env.update(PATH=f"{bindir}:{os.environ['PATH']}", POSTER_MARK=str(self.poster),
                               STUB_FINISH_S="2", LFSTUDIO_PUBLISH_DIR=str(self.publish))

    def finishing(self, jid):
        """Wait until the stub's worker has ended and the clip is about to come back."""
        case = self.studio.job(jid)["case_id"]
        deadline = time.time() + 20
        while not list((self.studio.runtime / "outputs" / "webui").glob(f"*/{case}/FINISHING")):
            self.assertLess(time.time(), deadline, "the clip did not reach its end")
            time.sleep(0.05)

    def test_cancel_first(self):
        a = self.generate()
        self.finishing(a)
        self.assertEqual(worker_groups(self.studio.proc.pid), [])           # nothing left to signal
        t0 = time.time()
        self.assertEqual(self.cancel(a)[0], 200)
        job = self.stopped(a, t0, within=5)                                 # the clip's last 2 s ran out
        self.assertTrue(list(Path(job["case_dirs"][0]).rglob("*.mp4")))    # it was made, but not kept
        self.assertFalse(self.poster.exists() or any(self.publish.iterdir()))
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": a})[0], 200)
        self.assertFalse(Path(job["case_dirs"][0]).exists())
        b = self.studio.wait([self.generate(seed=2)], timeout=30)[0]
        self.assertEqual((b["status"], b["includes_model_load"]), ("done", True))

    def test_clip_first(self):
        a = self.generate()
        deadline = time.time() + 20
        while not self.poster.exists():
            self.assertLess(time.time(), deadline, "no poster made")
            time.sleep(0.05)
        self.assertEqual(self.cancel(a), (409, {"error": "the clip has just finished and is being saved"}))
        job = self.studio.wait([a])[0]
        self.assertEqual(job["status"], "done")
        self.assertNotIn("saving", job)
        self.assertTrue(Path(job["output"]).is_file() and Path(job["published"]).is_file())
        self.assertEqual(self.studio.get(f"/video/{a}")[0], 200)
        self.assertEqual(self.cancel(a)[0], 409)
        self.assertNotIn("closing session", self.studio.log.read_text())    # the models stay loaded
        self.poster.unlink()
        b = self.studio.wait([self.generate(seed=2)], timeout=30)[0]
        self.assertEqual((b["status"], b.get("includes_model_load")), ("done", None))


# ------------------------------------------------------------------------------------------ the experimental Upscale
UPSCALE_LENGTH = "Upscale takes 5 s clips for now: other lengths not yet (NVIDIA's pipeline is fixed at 121 frames)"
NOTHING_LEFT = "nothing is left of the prompt without its keyframe line: the Upscale needs text"
TEXT_CLIP = {"ok": True, "clip_audio": True}     # a Sol-H3 Text clip's upscalable, as before every clip was taken
FILES_GONE = "the clip's working files (its session directory) are gone, so it can't be upscaled"
PARENT_BUSY = "an upscale of this clip is queued or running; cancel it first, or wait for it"


def probe(path):
    """A media file's streams by type (ffprobe): {"video": {...}, "audio": {...}}."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,"
                          "nb_frames,r_frame_rate,sample_rate,channels,duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def alive(pid):
    """Whether process `pid` runs (not a zombie)."""
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


class UpscaleTest(CancelTest):
    """The experimental Upscale (LFSTUDIO_UPSCALE=1, which tests/stub/studio.py sets, with tests/stub/upscale_job.py
    standing in for upscale/regen2k_job.py and MemAvailable read from a file it writes). `env` is set before start."""
    load_s, clip_s = 1, 0.5
    env = {}

    def seed(self):
        self.studio.env.update(self.env)

    def clip(self, **kw):
        """A finished clip: generate and wait; returns its record."""
        job = self.studio.wait([self.generate(**kw)], timeout=60)[0]
        self.assertEqual(job["status"], "done", job.get("error"))
        return job

    def upscale(self, jid, **fields):
        code, body = self.studio.upscale(jid, **fields)
        self.assertEqual(code, 200, body)
        self.assertEqual((body["job"]["kind"], body["job"]["parent"]), ("upscale", jid))
        return body["id"]

    def done(self, uid, timeout=60):
        job = self.studio.wait([uid], timeout=timeout)[0]
        self.assertEqual(job["status"], "done", job.get("error"))
        return job

    def regen(self, job):
        """The upscale's own directory, <session>-upscale-<resolution>/<case_id>/regen."""
        return Path(job["case_dirs"][0]) / "regen"

    def upscale_dirs(self):
        return [d.name for d in (self.studio.runtime / "outputs" / "webui").iterdir() if "-upscale-" in d.name]

    def job_pids(self):
        """The pids of the stand-in upscale jobs this test's studio started (STUB_UPSCALE_LOG)."""
        log = self.dir / "upscale-jobs.log"
        return [json.loads(line)["pid"] for line in log.read_text().splitlines()] if log.exists() else []

    def upscaling(self, uid):
        """Wait until the upscale's job runs; returns its pid."""
        self.until(uid, lambda j: j["status"] == "upscaling" and j.get("upscale_phase") not in (None, "starting"))
        deadline = time.time() + 10
        while not self.job_pids() and time.time() < deadline:
            time.sleep(0.05)
        return self.job_pids()[-1]

    def setUp(self):
        self.env = dict(self.env)
        super().setUp()

    def tearDown(self):
        for pid in self.job_pids():                # a job left running by a test that killed its server
            with contextlib.suppress(OSError):
                if alive(pid) and b"upscale_job.py" in (Path("/proc") / str(pid) / "cmdline").read_bytes():
                    os.killpg(pid, signal.SIGTERM)
                    deadline = time.time() + 10
                    while alive(pid) and time.time() < deadline:       # before its directory is removed
                        time.sleep(0.05)
        super().tearDown()


class Upscale(UpscaleTest):
    """An upscale of a finished 5 s 16:9 t2va clip: a job of its own, which closes the loaded models, runs the job in a
    session directory of its own and copies the clip's sound into the result."""

    def seed(self):
        self.studio.env["STUB_UPSCALE_LOG"] = str(self.dir / "upscale-jobs.log")

    def test_done_with_the_clips_sound(self):
        parent = self.clip()
        session_root = Path(parent["output"]).parents[2]
        st = self.studio.state()
        self.assertIs(st["upscale"], True)
        self.assertEqual(st["upscale_defaults"], {"resolution": "2k", "strength": "balanced",
                                                  "audio": "clip"})            # the page's choices start there
        self.assertEqual(st["upscale_load"], "fast")                         # the page's times: LFSTUDIO_UPSCALE_LOAD
        self.assertEqual(st["jobs"][0]["upscalable"], TEXT_CLIP)
        uid = self.upscale(parent["id"])
        job = self.done(uid)
        self.assertEqual({k: job[k] for k in ("task", "orientation", "seconds", "prompt", "seed")},
                         {k: parent[k] for k in ("task", "orientation", "seconds", "prompt", "seed")})
        self.assertEqual({k: job[k] for k in ("kind", "parent", "resolution", "strength", "steps", "start_sigma",
                                              "audio", "refine_seed")},
                         {"kind": "upscale", "parent": parent["id"], "resolution": "2k", "strength": "balanced",
                          "steps": 2, "start_sigma": 0.7, "audio": "clip", "refine_seed": 1})
        self.assertNotIn("source", job)                                     # from the clip's MP4, always
        self.assertRegex(job["case_id"], rf"^up_\d{{6}}_{uid[:6]}$")
        case = Path(job["case_dirs"][0])
        self.assertEqual((case.name, case.parent.parent), (job["case_id"], self.studio.runtime / "outputs" / "webui"))
        self.assertRegex(case.parent.name, r"^\d{8}-\d{6}-upscale-2k$")
        out = Path(job["output"])
        self.assertEqual(out, case / "regen" / "regen_2560x1440_121f.mp4")
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()),          # no copy without sound left
                         ["job.json", "receipt.json", "regen_2560x1440_121f.jpg", "regen_2560x1440_121f.mp4",
                          "worker.log"])
        mine, its = probe(out), probe(parent["output"])
        self.assertEqual((mine["video"]["nb_frames"], mine["video"]["r_frame_rate"]), ("121", "24/1"))
        for key in ("codec_name", "sample_rate", "channels", "nb_frames", "duration"):
            self.assertEqual(mine["audio"][key], its["audio"][key], key)       # the clip's AAC, copied as it is
        self.assertEqual(json.loads((out.parent / "receipt.json").read_text())["state"], "done")
        self.assertIn("[lf-studio] phase refine 2/2", (out.parent / "worker.log").read_text())
        self.assertEqual(set(job["phases_s"]), {"encoding", "loading", "refine", "decoding", "saving"})
        self.assertEqual((job["peak_gib"], job["low_mem_gib"]), (91.0, 15.0))       # from the job's receipt
        self.assertNotIn("upscale_phase", job)
        code, headers, _ = self.studio.get(f"/video/{uid}", {"Range": "bytes=0-9"})
        self.assertEqual((code, headers["Content-Disposition"]),
                         (206, f'inline; filename="lf-studio_t2va_5s_seed1_{parent["id"][:6]}_2k_{uid[:6]}.mp4"'))
        self.assertEqual(self.studio.get(f"/poster/{uid}")[1]["Content-Type"], "image/jpeg")
        for path in (f"/still/{uid}", f"/input/{uid}/first_frame/0"):            # an upscale has no inputs
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(path)
            self.assertEqual(e.exception.code, 404, path)
        inputs = json.loads(self.studio.get(f"/api/inputs/{uid}")[2])
        self.assertEqual((inputs["inputs"], inputs["missing"]), ({}, []))
        # the loaded models were closed for it (the job ran in its own process, nothing of the studio's was open)
        self.assertTrue((session_root / "CLOSED").exists())
        self.assertEqual(self.open_pipelines(), [])
        self.assertIn("closing session ('t2va', 'landscape', 5) - an upscale needs the memory",
                      self.studio.log.read_text())
        st = self.studio.state()
        self.assertEqual((st["session"]["state"], st["session"]["key"], st["done"]), ("idle", None, 2))
        up = next(j for j in st["jobs"] if j["id"] == uid)
        self.assertEqual(up["upscalable"], {"ok": False, "why": "an upscale can't be upscaled again"})
        self.assertFalse(alive(self.job_pids()[0]))
        self.assertTrue(self.clip(seed=2)["includes_model_load"])              # the next clip loads them again
        self.assertIn(f"WARNING: LFSTUDIO_TEST_MEMINFO={self.studio.meminfo}, a test hook", self.studio.log.read_text())
        argv = json.loads((self.dir / "upscale-jobs.log").read_text().splitlines()[0])["argv"]
        self.assertEqual(argv[-4:], ["--job", str(self.regen(job) / "job.json"), "--parent", str(self.studio.proc.pid)])

    def test_job_json_and_request_fields(self):
        parent = self.clip()
        uid = self.upscale(parent["id"], resolution="1080p", strength="full", audio="silence", seed=99)
        job = self.done(uid)
        self.assertEqual(Path(job["output"]).name, "regen_1920x1080_121f.mp4")
        self.assertEqual((job["seed"], job["refine_seed"], job["steps"], job["start_sigma"], job["load"]),
                         (1, 99, 3, None, "fast"))                      # LFSTUDIO_UPSCALE_LOAD's default
        data = json.loads((self.regen(job) / "job.json").read_text())
        root = self.dir / "regen2k"                     # tests/stub/studio.py's LFSTUDIO_UPSCALE_PKG and so on
        self.assertEqual(data, {"version": 1, "case": str(self.regen(job).parent / "input"),
                                "resolution": "1080p", "steps": 3, "start_sigma": None, "seed": 99,
                                "audio": "silence", "load": "fast", "preread": False,
                                "paths": {"package": str(root / "package"), "weights": str(root / "weights"),
                                          "base_model": str(root / "base"), "output": str(self.regen(job))},
                                "meta": {"studio_job": uid, "parent": parent["id"], "strength": "full"}})
        uid = self.upscale(parent["id"], resolution="4k", strength="faithful", seed=None)     # null: the default
        job = self.done(uid)
        self.assertEqual(Path(job["output"]).name, "regen_3840x2160_121f.mp4")
        self.assertRegex(Path(job["case_dirs"][0]).parent.name, r"-upscale-4k$")
        self.assertEqual((job["steps"], job["start_sigma"], job["refine_seed"], job["audio"]), (2, 0.5, 1, "clip"))
        self.assertEqual(len(self.upscale_dirs()), 2)

    def test_a_9x16_clip(self):
        """A 5 s 9:16 t2va clip at each resolution: the output upright (the job takes the orientation from the clip's
        MP4; job.json has no field for it), named by its size."""
        parent = self.clip(orientation="portrait")
        self.assertEqual(self.studio.job(parent["id"])["upscalable"], TEXT_CLIP)
        for resolution, name in (("2k", "regen_1440x2560_121f.mp4"), ("1080p", "regen_1080x1920_121f.mp4"),
                                 ("4k", "regen_2160x3840_121f.mp4")):
            job = self.done(self.upscale(parent["id"], resolution=resolution))
            self.assertEqual((job["orientation"], Path(job["output"]).name), ("portrait", name), resolution)
            video = probe(job["output"])["video"]
            self.assertLess(video["width"], video["height"])                       # the stand-in job's, upright
            self.assertEqual(video["nb_frames"], "121")
            data = json.loads((self.regen(job) / "job.json").read_text())
            self.assertNotIn("orientation", data)
            self.assertEqual(data["resolution"], resolution)
            self.assertEqual(json.loads((self.regen(job) / "receipt.json").read_text())["orientation"], "portrait")
        code, headers, _ = self.studio.get(f"/video/{job['id']}", {"Range": "bytes=0-9"})
        self.assertEqual(headers["Content-Disposition"],
                         f'inline; filename="lf-studio_t2va_5s_seed1_{parent["id"][:6]}_4k_{job["id"][:6]}.mp4"')

    def test_progress(self):
        """The status while it runs, and the job's own phase lines as upscale_phase."""
        self.studio.stop()
        self.studio.env.update(STUB_UPSCALE_S="6", STUB_CLOSE_S="1")          # 1 s per phase of the job
        self.studio.start()
        uid = self.upscale(self.clip()["id"])
        seen, deadline = [], time.time() + 30
        while time.time() < deadline:
            rec = json.loads(self.studio.get(f"/api/jobs/{uid}")[2])
            step = (rec["status"], rec["phase"], rec.get("upscale_phase"))
            if step not in seen:
                seen.append(step)
            if rec["state"] == "done":
                break
            time.sleep(0.03)
        self.assertIn(("freeing memory", "unloading", None), seen)
        phases = [p for status, phase, p in seen if status == "upscaling" and phase == "upscaling"]
        self.assertIn(phases[0], ("starting", "encoding"))                  # starting: until the job's first line
        self.assertEqual([p for p in phases if p not in ("starting", "adding the sound")],
                         ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])
        self.assertEqual(seen[-1], ("done", None, None))


class UpscaleSettings(UpscaleTest):
    """LFSTUDIO_UPSCALE_STRENGTH and _AUDIO are a request's defaults; _LOAD, _PREREAD and the paths go into job.json
    (and _LOAD into the upscale's record)."""
    env = {"LFSTUDIO_UPSCALE_STRENGTH": "full", "LFSTUDIO_UPSCALE_AUDIO": "silence",
           "LFSTUDIO_UPSCALE_LOAD": "standard", "LFSTUDIO_UPSCALE_PREREAD": "1",
           "LFSTUDIO_UPSCALE_PKG": "/opt/regen2k/package", "LFSTUDIO_UPSCALE_WEIGHTS": "/opt/regen2k/weights",
           "LFSTUDIO_UPSCALE_BASE": "/opt/minimax-h3", "SPARK_DIRECT_IO": "0"}       # no O_DIRECT: standard anyway

    def test_defaults(self):
        parent = self.clip()
        self.assertEqual(self.studio.state()["upscale_defaults"],
                         {"resolution": "2k", "strength": "full", "audio": "silence"})
        self.assertEqual(self.studio.state()["upscale_load"], "standard")
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual((job["strength"], job["steps"], job["start_sigma"], job["audio"]),
                         ("full", 3, None, "silence"))
        data = json.loads((self.regen(job) / "job.json").read_text())
        self.assertEqual((data["load"], data["preread"], data["audio"], data["paths"]["package"],
                          data["paths"]["weights"], data["paths"]["base_model"]),
                         ("standard", True, "silence", "/opt/regen2k/package", "/opt/regen2k/weights",
                          "/opt/minimax-h3"))
        self.assertEqual(job["load"], "standard")
        self.assertIn("default strength full, audio silence; standard load", self.studio.log.read_text())
        self.assertNotIn("O_DIRECT", self.studio.log.read_text())


class UpscaleWithoutODirect(UpscaleTest):
    """SPARK_FAST_SAFETENSORS=0 or SPARK_DIRECT_IO=0 (the patch's switches) also reach the Upscale's job: with the fast
    load the start log says that it reads every file through mmap then."""
    env = {"SPARK_FAST_SAFETENSORS": "0"}

    def test_the_start_log_says_so(self):
        log = self.studio.log.read_text()
        self.assertIn("audio clip; fast load\n", log)
        self.assertIn("WARNING: SPARK_FAST_SAFETENSORS or SPARK_DIRECT_IO is not 1 (the patch's switches), and the "
                      "Upscale's job inherits them: its fast load reads no weight file with O_DIRECT, all of them "
                      "through mmap, which takes minutes", log)


class UpscaleRefusals(UpscaleTest):
    load_s, clip_s = 1, 0.3

    def test_bad_requests(self):
        jid = self.clip()["id"]
        self.assertEqual(self.studio.upscale("f" * 32), (404, {"error": "no such clip"}))
        bad = (({}, "id must be the id of a finished clip"), ({"id": 5}, "id must be the id of a finished clip"),
               ({"id": jid, "resolution": "2K"}, "resolution must be one of 1080p, 2k, 4k"),
               ({"id": jid, "resolution": "768p"}, "resolution must be one of 1080p, 2k, 4k"),
               ({"id": jid, "strength": "nvidia"}, "strength must be one of faithful, balanced, full"),
               ({"id": jid, "strength": ["full"]}, "strength must be one of faithful, balanced, full"),
               ({"id": jid, "audio": "both"}, "audio must be one of clip, silence"),
               ({"id": jid, "steps": 3}, 'unknown key "steps" (known: id, resolution, strength, audio, seed)'))
        bad += tuple(({"id": jid, "source": source}, 'unknown key "source" (known: id, resolution, strength, audio, '
                      'seed)') for source in ("draft", "final", None))          # no source to pick: the clip's MP4
        bad += tuple(({"id": jid, "seed": seed}, "seed must be a nonnegative 63-bit integer, or null for the clip's")
                     for seed in (-1, 2 ** 63, "5", True, 1.5))
        for body, error in bad:
            self.assertEqual(self.studio.post("/api/upscale", json_body=body), (400, {"error": error}), body)
        expected = 'expected JSON {"id": ..., "resolution": ..., "strength": ..., "audio": ..., "seed": ...}'
        for raw_body in (b"{", b"[1]", b'"x"', b"\xff"):
            code, _, answer = self.studio.request("POST", "/api/upscale", raw_body,
                                                  {"Content-Type": "application/json"})
            self.assertEqual((code, json.loads(answer)), (400, {"error": expected}), raw_body)
        self.assertEqual(self.studio.post("/api/upscale", fields={"id": jid})[0], 400)     # a form is not JSON
        self.assertEqual([j["id"] for j in self.studio.state()["jobs"]], [jid])           # nothing queued

    def test_clips_it_cannot_upscale(self):
        """Other lengths than 5 s (not yet, every task), a Keyframes clip whose prompt is nothing but its keyframe line
        (no text without it), a failed clip."""
        ten = self.clip(seconds=10, seed=2)
        six = self.clip(orientation="portrait", seconds=6, seed=6)     # 3 s went after 0.5.0: see OldLengths
        fl2va = self.clip(task="fl2va", seed=3, seconds=8, files={"first_frame": [("f.png", png())]})
        line = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
                "referenced.")
        bare = self.clip(task="fl2va", seed=7, prompt=line, files={"first_frame": [("f.png", png())]})
        failed = self.studio.wait([self.generate(prompt="STUB_FAIL", seed=4)])[0]
        for job, code, why in ((ten, 400, UPSCALE_LENGTH), (six, 400, UPSCALE_LENGTH), (fl2va, 400, UPSCALE_LENGTH),
                               (bare, 400, NOTHING_LEFT), (failed, 409, "the clip failed")):
            self.assertEqual(self.studio.upscale(job["id"]), (code, {"error": why}), job["id"])
        found = {j["id"]: j.get("upscalable") for j in self.studio.state()["jobs"]}
        self.assertEqual((found[ten["id"]], found[six["id"]], found[fl2va["id"]], found[bare["id"]],
                          found[failed["id"]]),
                         ({"ok": False, "why": UPSCALE_LENGTH}, {"ok": False, "why": UPSCALE_LENGTH},
                          {"ok": False, "why": UPSCALE_LENGTH}, {"ok": False, "why": NOTHING_LEFT}, None))
        uid = self.upscale(self.clip(seed=5)["id"])
        self.assertEqual(self.studio.upscale(uid), (400, {"error": "an upscale can't be upscaled again"}))  # queued
        self.done(uid)
        self.assertEqual(self.studio.upscale(uid), (400, {"error": "an upscale can't be upscaled again"}))

    def test_missing_files(self):
        parent = self.clip()
        case = Path(parent["output"]).parents[1]
        upscalable = lambda: self.studio.job(parent["id"])["upscalable"]        # noqa: E731
        self.assertEqual(upscalable(), TEXT_CLIP)
        (case / "stage1" / "stage1_direct_tensors.pt").unlink()
        self.assertEqual(upscalable(), dict(TEXT_CLIP, clip_audio=False))
        no_audio = (409, {"error": "the clip has no Stage-1 audio (stage1/stage1_direct_tensors.pt); ask for audio "
                                   "silence"})
        self.assertEqual(self.studio.upscale(parent["id"]), no_audio)          # audio clip is the default
        self.assertEqual(self.done(self.upscale(parent["id"], audio="silence"))["audio"], "silence")
        (case / "qwen" / "conditioning.pt").unlink()
        self.assertEqual(upscalable(), {"ok": False, "why": FILES_GONE})
        self.assertEqual(self.studio.upscale(parent["id"], audio="silence"), (409, {"error": FILES_GONE}))


class UpscaleUnfinishedParents(UpscaleTest):
    load_s, clip_s = 3, 3

    def test_queued_running_and_cancelled(self):
        a, b = self.generate(), self.generate(seed=2)
        self.until(a, lambda j: j["status"] == "loading models + generating")
        self.assertEqual(self.studio.upscale(a), (409, {"error": "the clip is still being made"}))
        self.assertEqual(self.studio.upscale(b), (409, {"error": "the clip is still queued"}))
        self.assertEqual(self.cancel(b)[0], 200)
        self.assertEqual(self.studio.upscale(b), (409, {"error": "the clip was cancelled"}))
        self.assertEqual(self.cancel(a)[0], 200)
        self.studio.wait([a, b])


class UpscaleQueue(UpscaleTest):
    """An upscale waits while any clip is queued (it closes the models they use); it runs once none is."""
    load_s, clip_s = 2, 1.5

    def test_clips_first(self):
        parent = self.clip()
        a = self.generate(seed=2)                                  # in the loaded session
        self.until(a, lambda j: j["status"] == "generating")
        uid = self.upscale(parent["id"])
        b = self.generate(seed=3)                                  # queued after the upscale: runs before it
        c = self.generate(seed=4, orientation="portrait")
        ja, jb, jc, up = self.studio.wait([a, b, c, uid], timeout=60)
        self.assertEqual([j["status"] for j in (ja, jb, jc, up)], ["done"] * 4)
        self.assertLess(jc["finished"], up["started"])
        self.assertFalse(jb.get("includes_model_load") or jc.get("includes_model_load"))   # the session stayed
        self.assertTrue(Path(jc["output"]).parents[2].joinpath("CLOSED").exists())        # until the upscale
        self.assertEqual(self.open_pipelines(), [])


class UpscaleCancel(UpscaleTest):
    """Cancel takes a queued upscale out of the queue, or stops its job (SIGTERM to its process group, SIGKILL 20 s
    later if it is still there); the card stays, cancelled, and Delete moves its directory."""
    load_s, clip_s = 1, 3
    env = {"STUB_UPSCALE_S": "30"}

    def seed(self):
        self.studio.env.update(self.env, STUB_UPSCALE_LOG=str(self.dir / "upscale-jobs.log"))

    def test_queued(self):
        parent = self.clip()
        a = self.generate(seed=2)
        self.until(a, lambda j: j["status"] == "generating")
        uid = self.upscale(parent["id"])
        code, body = self.cancel(uid)
        self.assertEqual((code, body["job"]["status"]), (200, "cancelled"))
        self.assertEqual(self.cancel(uid), (409, {"error": "the clip is already cancelled"}))
        self.assertEqual(self.studio.wait([a])[0]["status"], "done")
        job = self.studio.job(uid)
        self.assertEqual(job["status"], "cancelled")
        for key in ("started", "case_dirs", "error", "output"):
            self.assertNotIn(key, job)
        self.assertEqual((self.upscale_dirs(), self.job_pids()), ([], []))
        self.assertNotIn("an upscale needs the memory", self.studio.log.read_text())   # the models stayed

    def test_running(self):
        parent = self.clip()
        uid = self.upscale(parent["id"])
        pid = self.upscaling(uid)
        self.assertEqual(worker_groups(self.studio.proc.pid), [pid])        # the job, in a process group of its own
        st = self.studio.state()                                            # busy for spark-switchboard and the page
        self.assertNotIn(next(j for j in st["jobs"] if j["id"] == uid)["status"], idle_values())
        self.assertEqual(self.studio.post("/api/unload"), (409, {"error": "a job is running"}))
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "landscape",
                                                                  "seconds": 5})[0], 409)
        t0 = time.time()
        code, body = self.cancel(uid)
        self.assertEqual((code, body["job"]["cancel_requested"]), (200, True))
        job = self.studio.wait([uid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertLess(time.time() - t0, 5, "stopped right away")
        for key in ("error", "output", "published", "cancel_requested", "saving"):
            self.assertNotIn(key, job)
        self.assertFalse(alive(pid))
        self.assertEqual(worker_groups(self.studio.proc.pid), [])
        receipt = json.loads((self.regen(job) / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("cancelled", 143))
        st = self.studio.state()
        self.assertTrue(all(j["status"] in idle_values() for j in st["jobs"]))
        self.assertIn(uid, [j["id"] for j in st["jobs"]])                   # the card stays
        case = Path(job["case_dirs"][0])
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": uid}), (200, {"ok": True}))
        self.assertFalse(case.exists())
        self.assertTrue(Path(parent["output"]).is_file())

    def test_while_the_models_close(self):
        self.studio.stop()
        self.studio.env["STUB_CLOSE_S"] = "3"
        self.studio.start()
        parent = self.clip()
        uid = self.upscale(parent["id"])
        self.until(uid, lambda j: j["status"] == "freeing memory")
        self.assertEqual(self.cancel(uid)[0], 200)
        job = self.studio.wait([uid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("case_dirs", job)                                  # nothing of its own ran
        self.assertEqual((self.upscale_dirs(), self.job_pids()), ([], []))

    def test_a_job_that_ignores_sigterm(self):
        """The real job's SIGTERM handler runs only once the GPU call under way returns (minutes at 4K): SIGKILL
        follows 20 s after the SIGTERM."""
        self.studio.stop()
        self.studio.env["STUB_UPSCALE_TERM"] = "ignore"
        self.studio.start()
        uid = self.upscale(self.clip()["id"])
        pid = self.upscaling(uid)
        t0 = time.time()
        self.assertEqual(self.cancel(uid)[0], 200)
        time.sleep(3)
        self.assertTrue(alive(pid))                                         # it ignored the SIGTERM
        self.assertEqual(self.studio.job(uid)["status"], "upscaling")
        job = self.studio.wait([uid], timeout=40)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertGreater(time.time() - t0, 19)
        self.assertFalse(alive(pid))
        self.assertIn(f"cancel: the upscale's job {pid} is still there 20 s after SIGTERM: SIGKILL",
                      self.studio.log.read_text())
        self.assertEqual(worker_groups(self.studio.proc.pid), [])

    def test_restart_while_running(self):
        """A server killed hard takes its job with it (the job's --parent), so a studio started again never runs clips
        next to an unguarded upscale."""
        parent = self.clip()
        uid = self.upscale(parent["id"])
        pid = self.upscaling(uid)
        self.studio.proc.kill()                                             # no clean shutdown
        self.studio.proc.wait()
        deadline = time.time() + 10
        while alive(pid) and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid), "the upscale's job outlived the server")
        self.studio.start()
        job = self.studio.job(uid)
        self.assertEqual((job["status"], job["error"]), ("failed", "server restarted"))
        self.assertEqual(self.studio.upscale(uid)[0], 400)

    def test_stop_ends_the_job(self):
        uid = self.upscale(self.clip()["id"])
        pid = self.upscaling(uid)
        self.studio.stop()                                                  # SIGTERM: the server stops its job
        self.assertFalse(alive(pid))
        self.assertIn("stopping the running upscale", self.studio.log.read_text())
        self.studio.start()
        self.assertEqual(self.studio.job(uid)["status"], "failed")


class UpscaleDelete(UpscaleTest):
    """Delete and Undo of an upscale move only its own files (its directory and publish copy), and the clip's only
    the clip's; a clip can't be deleted while an upscale of it is queued or running."""
    env = {"STUB_UPSCALE_S": "1.5"}

    def setUp(self):
        self.publish = Path(tempfile.mkdtemp(prefix="lf-studio-publish-"))
        self.env = dict(self.env, LFSTUDIO_PUBLISH_DIR=str(self.publish))
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.publish, ignore_errors=True)

    def delete(self, jid, path="/api/delete"):
        return self.studio.post(path, json_body={"id": jid})

    def test_each_moves_only_its_own(self):
        parent = self.clip()
        uid = self.upscale(parent["id"])
        self.assertEqual(self.delete(parent["id"]), (409, {"error": PARENT_BUSY}))     # queued or running
        up = self.done(uid)
        pub = Path(parent["published"])
        self.assertEqual(Path(up["published"]), self.publish / f"{pub.stem}_2k.mp4")      # next to the clip's
        clip_files = [Path(parent["output"]).parents[1], pub, pub.with_suffix(".jpg")]
        up_files = [Path(up["case_dirs"][0]), Path(up["published"]), Path(up["published"]).with_suffix(".jpg")]
        self.assertTrue(all(p.exists() for p in clip_files + up_files))
        self.assertEqual(self.delete(uid), (200, {"ok": True}))
        self.assertEqual([p.exists() for p in clip_files + up_files], [True] * 3 + [False] * 3)
        self.assertEqual(len(list((self.studio.runtime / "trash" / uid).iterdir())), 3)
        self.assertEqual(self.studio.get(f"/video/{parent['id']}")[0], 200)
        self.assertEqual(self.delete(uid, "/api/undelete")[0], 200)
        self.assertTrue(all(p.exists() for p in clip_files + up_files))
        self.assertEqual(self.delete(parent["id"]), (200, {"ok": True}))               # done now: allowed
        self.assertEqual([p.exists() for p in clip_files + up_files], [False] * 3 + [True] * 3)
        code, headers, data = self.studio.get(f"/video/{uid}")                           # still playable
        self.assertEqual((code, data[4:8]), (200, b"ftyp"))
        self.assertEqual(self.studio.upscale(parent["id"]), (404, {"error": "no such clip"}))
        self.assertEqual(self.delete(parent["id"], "/api/undelete")[0], 200)
        second = self.done(self.upscale(parent["id"]))
        self.assertEqual(Path(second["published"]).name, f"{pub.stem}_2k-2.mp4")       # a second one beside it

    def test_clip_without_publish_copy(self):
        """A clip with no publish copy of its own (made before the setting): the upscale's is named as the clip's
        would be."""
        self.studio.stop()
        self.studio.env.pop("LFSTUDIO_PUBLISH_DIR")
        self.studio.start()
        parent = self.clip()
        self.studio.stop()
        self.studio.env["LFSTUDIO_PUBLISH_DIR"] = str(self.publish)
        self.studio.start()
        up = self.done(self.upscale(parent["id"], resolution="4k"))
        self.assertRegex(Path(up["published"]).name,
                         rf"^\d{{8}}-\d{{6}}_t2va_landscape_5s_seed1_{parent['id'][:6]}_4k\.mp4$")


class UpscaleWarmUp(UpscaleTest):
    """Upscales load none of the studio's models: the warm-up at start skips them, and one is done again after an
    upscale."""
    preload, load_s = True, 2

    def seed(self):
        self.write_state([finished_job("1" * 32, created=1),
                          finished_job("2" * 32, orientation="portrait", created=2),
                          finished_job("3" * 32, created=3, kind="upscale", parent="1" * 32, resolution="2k")])

    def test_warm_up_skips_upscales(self):
        s = self.studio.wait_for(lambda s: s["session"].get("loading_key"))["session"]
        self.assertEqual(s["loading_key"], ["t2va", "portrait", 5])          # the last clip's, not the upscale's
        self.assertEqual(self.studio.wait_for(lambda s: s["session"]["state"] == "ready")["session"]["key"],
                         ["t2va", "portrait", 5])


class UpscaleWarmUpAfter(UpscaleTest):
    preload, load_s = True, 2

    def test_warm_up_again_and_an_upscale_stops_it(self):
        parent = self.clip()
        self.done(self.upscale(parent["id"]))
        s = self.studio.wait_for(lambda s: s["session"].get("preloading"))["session"]
        self.assertEqual(s["loading_key"], ["t2va", "landscape", 5])
        self.assertIn("again after the upscale", self.studio.log.read_text())
        self.done(self.upscale(parent["id"]))                                # during that warm-up
        self.assertIn("aborting preload: an upscale arrived, which needs the memory", self.studio.log.read_text())


class UpscaleMemory(UpscaleTest):
    """The memory floor before the job (LFSTUDIO_UPSCALE_MIN_FREE_GIB, default 110) and the guard while it runs
    (LFSTUDIO_UPSCALE_GUARD_GIB, default 5), on the stub studio's MemAvailable file."""
    env = {"STUB_UPSCALE_S": "30"}

    def seed(self):
        self.studio.env.update(self.env, STUB_UPSCALE_LOG=str(self.dir / "upscale-jobs.log"))

    def test_floor(self):
        parent = self.clip()
        self.studio.set_memory(50)
        job = self.studio.wait([self.upscale(parent["id"])], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "UpscaleFailed: not enough free memory for an upscale: 50.0 GiB available with the models closed, 110 GiB "
            "needed (LFSTUDIO_UPSCALE_MIN_FREE_GIB); close other apps")))
        self.assertNotIn("case_dirs", job)
        self.assertEqual((self.upscale_dirs(), self.job_pids()), ([], []))
        self.assertNotIn("retrying", self.studio.log.read_text())

    def test_floor_at_4k(self):
        """4K needs 5 GiB more (its job used ~105 GiB against ~101 at 2K): 112 GiB is enough for 2K, not for 4K."""
        parent = self.clip()
        self.studio.set_memory(112)
        job = self.studio.wait([self.upscale(parent["id"], resolution="4k")], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "UpscaleFailed: not enough free memory for an upscale: 112.0 GiB available with the models closed, 115 GiB "
            "needed (LFSTUDIO_UPSCALE_MIN_FREE_GIB plus 5 GiB at 4K); close other apps")))
        self.assertEqual((self.upscale_dirs(), self.job_pids()), ([], []))
        self.assertTrue(self.upscaling(self.upscale(parent["id"], resolution="2k")))

    def test_guard(self):
        uid = self.upscale(self.clip()["id"])
        pid = self.upscaling(uid)
        self.studio.set_memory(1)
        t0 = time.time()
        job = self.studio.wait([uid], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "UpscaleFailed: stopped: MemAvailable was below 5 GiB for 3 s (1.0 GiB; LFSTUDIO_UPSCALE_GUARD_GIB)")))
        self.assertLess(time.time() - t0, 8)
        self.assertFalse(alive(pid))
        # SIGKILL at once, not SIGTERM: the job had no time for its receipt
        self.assertEqual(json.loads((self.regen(job) / "receipt.json").read_text())["state"], "running")
        self.assertIn("upscale: MemAvailable 1.0 GiB, below 5 GiB for 3 s: stopping its job (SIGKILL)",
                      self.studio.log.read_text())

    def test_guard_on_a_job_that_ignores_sigterm(self):
        self.studio.stop()
        self.studio.env["STUB_UPSCALE_TERM"] = "ignore"
        self.studio.start()
        uid = self.upscale(self.clip()["id"])
        pid = self.upscaling(uid)
        self.studio.set_memory(1)
        t0 = time.time()
        self.assertEqual(self.studio.wait([uid], timeout=30)[0]["status"], "failed")
        self.assertLess(time.time() - t0, 6)                               # 3 s below the floor, then at once
        self.assertFalse(alive(pid))


class UpscaleMemoryChecksOff(UpscaleTest):
    env = {"LFSTUDIO_UPSCALE_MIN_FREE_GIB": "0", "LFSTUDIO_UPSCALE_GUARD_GIB": "0", "STUB_UPSCALE_S": "4"}

    def test_zero_turns_them_off(self):
        parent = self.clip()
        self.studio.set_memory(1)
        self.assertEqual(self.done(self.upscale(parent["id"]))["status"], "done")
        self.assertEqual(self.done(self.upscale(parent["id"], resolution="4k"))["status"], "done")   # no 4K extra


class UpscaleFailure(UpscaleTest):
    env = {"STUB_UPSCALE_FAIL": "1"}

    def test_failed_with_the_jobs_error_and_no_retry(self):
        parent = self.clip()
        job = self.studio.wait([self.upscale(parent["id"])], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]),
                         ("failed", "UpscaleFailed: OutOfMemoryError: CUDA out of memory (STUB_UPSCALE_FAIL)"))
        self.assertNotIn("first_error", job)
        self.assertNotIn("output", job)
        self.assertEqual(job["upscale_phase"], "refine 1/2")                # where it failed
        self.assertNotIn("retrying job", self.studio.log.read_text())
        self.assertIn("[lf-studio] error OutOfMemoryError", (self.regen(job) / "worker.log").read_text())
        self.assertEqual(self.studio.job(parent["id"])["upscalable"]["ok"], True)
        self.assertNotIn("upscalable", self.studio.job(job["id"]))          # only finished clips have it
        case = Path(job["case_dirs"][0])
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": job["id"]}), (200, {"ok": True}))
        self.assertFalse(case.exists())

    def test_no_output(self):
        self.studio.stop()
        self.studio.env.update(STUB_UPSCALE_FAIL="0", STUB_UPSCALE_OUTPUT="none")
        self.studio.start()
        job = self.studio.wait([self.upscale(self.clip()["id"])], timeout=30)[0]
        log = self.regen(job) / "worker.log"
        self.assertEqual((job["status"], job["error"]),
                         ("failed", f"UpscaleFailed: the upscale job wrote no video_noaudio.mp4; see {log}"))
        self.assertNotIn("output", job)

    def test_the_sound_cannot_be_added(self):
        self.studio.stop()
        self.studio.env.update(STUB_UPSCALE_FAIL="0", STUB_UPSCALE_OUTPUT="bad")
        self.studio.start()
        job = self.studio.wait([self.upscale(self.clip()["id"])], timeout=30)[0]
        self.assertEqual(job["status"], "failed")
        self.assertTrue(job["error"].startswith("UpscaleFailed: could not add the clip's sound (ffmpeg: "), job["error"])
        self.assertNotIn("output", job)
        self.assertEqual(sorted(p.name for p in self.regen(job).iterdir()),         # no half-made MP4 left
                         ["job.json", "receipt.json", "video_noaudio.mp4", "worker.log"])

    def test_no_python(self):
        python = self.dir / "regen2k" / "no-python"
        self.studio.env["LFSTUDIO_UPSCALE_PYTHON"] = str(python)
        self.studio.stop()
        self.studio.start()
        job = self.studio.wait([self.upscale(self.clip()["id"])], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", "UpscaleFailed: cannot start the upscale job with "
                                                                   f"LFSTUDIO_UPSCALE_PYTHON={python} (No such file "
                                                                   "or directory)"))


class UpscaleRunTimeChecks(UpscaleTest):
    """An upscale checks its clip again when it starts: files removed while it waited fail it, without running its
    job."""
    load_s, clip_s = 1, 4

    def seed(self):
        self.studio.env.update(STUB_UPSCALE_LOG=str(self.dir / "upscale-jobs.log"))

    def queued_behind_a_clip(self, parent, **fields):
        a = self.generate(seed=7)
        self.until(a, lambda j: j["status"] in ("generating", "loading models + generating"))
        return a, self.upscale(parent["id"], **fields)

    def test_files_gone_meanwhile(self):
        parent = self.clip()
        case = Path(parent["output"]).parents[1]
        a, uid = self.queued_behind_a_clip(parent)                            # audio clip, the default
        (case / "stage1" / "stage1_direct_tensors.pt").unlink()
        job = self.studio.wait([a, uid], timeout=60)[1]
        self.assertEqual((job["status"], job["error"]), ("failed", "UpscaleFailed: the clip's Stage-1 audio is gone"))
        a, uid = self.queued_behind_a_clip(parent, audio="silence")             # no Stage-1 audio any more
        (case / "qwen" / "conditioning.pt").unlink()
        job = self.studio.wait([a, uid], timeout=60)[1]
        self.assertEqual((job["status"], job["error"]),
                         ("failed", f"UpscaleFailed: the clip can't be upscaled now: {FILES_GONE}"))
        self.assertNotIn("case_dirs", job)
        self.assertEqual((self.upscale_dirs(), self.job_pids()), ([], []))


class UpscaleQueueFull(UpscaleTest):
    """LFSTUDIO_MAX_QUEUED counts queued upscales with the clips: a request while the queue is full gets 429."""
    load_s, clip_s = 1, 6
    env = {"LFSTUDIO_MAX_QUEUED": "1"}

    def test_full(self):
        parent = self.clip()
        a = self.generate(seed=2)
        self.until(a, lambda j: j["status"] == "generating")              # the running clip does not count
        uid = self.upscale(parent["id"])                                    # waits: the queue is full now
        full = "the queue is full (1 clips waiting); try again when one has finished"
        code, headers, answer = self.studio.request("POST", "/api/upscale", json.dumps({"id": parent["id"]}).encode(),
                                                    {"Content-Type": "application/json"})
        self.assertEqual((code, headers["Retry-After"], json.loads(answer)), (429, "60", {"error": full}))
        self.assertEqual(self.studio.generate(seed=3), (429, {"error": full}))
        self.assertEqual(len(self.studio.state()["jobs"]), 3)
        self.assertEqual([j["status"] for j in self.studio.wait([a, uid], timeout=60)], ["done", "done"])


class UpscaleWhileMoving(UpscaleTest):
    """A request for a clip whose Delete is moving its files gets 409 (the move is held up by a lock on jobs.db
    that this test takes, under the server's 5 s busy timeout)."""

    def test_being_moved(self):
        import sqlite3
        import threading
        parent = self.clip()
        db = sqlite3.connect(self.studio.state_dir / "jobs.db", isolation_level=None, timeout=1)
        self.addCleanup(db.close)
        answers = []
        db.execute("BEGIN EXCLUSIVE")
        try:
            delete = threading.Thread(target=lambda: answers.append(
                self.studio.post("/api/delete", json_body={"id": parent["id"]})))
            delete.start()
            moving = "the clip is being moved; try again in a moment"
            self.studio.wait_for(lambda st: any(j["id"] == parent["id"] and (j.get("upscalable") or {}).get("why")
                                                == moving for j in st["jobs"]), timeout=3)
            self.assertEqual(self.studio.upscale(parent["id"]), (409, {"error": moving}))
        finally:
            db.execute("ROLLBACK")
        delete.join(timeout=15)
        self.assertEqual(answers, [(200, {"ok": True})])
        self.assertEqual([j["id"] for j in self.studio.state()["jobs"] if j.get("kind") == "upscale"], [])


def load_rewrites():
    """bench/h3turbo_encode.py, for its text rewrites (text_only_prompt, ref_text_only_prompt), as the server loads
    it."""
    spec = importlib.util.spec_from_file_location("rewrites_under_test", REPO_DIR / "bench" / "h3turbo_encode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEYFRAME_PROMPT = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
                   "referenced.\n\nintegrated_multimodal_description: [Shot 1] The kitchen shown in <Picture 1>; a "
                   "kettle steams.\n\noverall_soundscape: A kettle whistles.\n\nnon_diegetic_music: N/A")
REFERENCE_PROMPT = ("subject_definitions:\n<Subject 1> is the woman in <Picture 1>, with short red hair.\n<Audio 1> is "
                    "the voice-timbre reference for <Subject 1> (S1).\n\nsummary:\n[reference generation + audio "
                    "reference] <Subject 1> greets the camera.\n\nretention_analysis:\n<Subject 1>: fully_preserved."
                    "\n\ndetailed_description:\n[Shot 1] A medium shot frames <Subject 1> (S1) on a snowy street. "
                    "She says in the warm voice referenced from <Audio 1>, <d>[English] Welcome back.</d>\n\n"
                    "overall_soundscape:\nSoft wind.\n\nnon_diegetic_music:\nN/A")


def inode(path):
    return os.stat(path).st_ino


class EveryClipParts:
    """What UpscaleEveryClip and UpscaleEncode share (a mixin: no tests of its own): the logs of the stand-ins' runs,
    and an upscale's statuses."""

    def seed(self):
        self.studio.env.update(STUB_UPSCALE_LOG=str(self.dir / "upscale-jobs.log"),
                               STUB_QWEN_LOG=str(self.dir / "encodes.log"), **self.env)

    def encodes(self):
        log = self.dir / "encodes.log"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def statuses(self, uid, timeout=60):
        """The (status, upscale_phase) an upscale goes through, polled until it has finished; its record."""
        seen, deadline = [], time.time() + timeout
        while time.time() < deadline:
            rec = json.loads(self.studio.get(f"/api/jobs/{uid}")[2])
            step = (rec["status"], rec.get("upscale_phase"))
            if step not in seen:
                seen.append(step)
            if rec["state"] not in ("queued", "running"):
                return seen, rec
            time.sleep(0.02)
        self.fail(f"upscale {uid} did not finish: {seen}")


class UpscaleEveryClip(EveryClipParts, UpscaleTest):
    """The Upscale takes every finished 5 s clip: a Sol-H3 Keyframes clip (any keyframes) and a Reference clip get a
    conditioning of text alone made for the upscale (the prompt encode, --upscale, of the prompt without its keyframe
    line or its references), after the models close and before the job, under the same Cancel. Every upscale reads an
    input folder of its own (job.json "case"): the clip's files linked, source.json."""
    env = {"STUB_UPSCALE_S": "1"}

    def test_a_keyframes_clip(self):
        """A first frame: upscalable with its text encoded; the statuses in order; the encode's arguments; the input
        folder (the clip's MP4 and audio linked, the encode's qwen/, source.json) and job.json; the record's text."""
        self.studio.stop()                 # the models' close takes a second, so "freeing memory" is seen
        self.studio.env.update(STUB_QWEN_S="1.5", STUB_UPSCALE_S="2", STUB_CLOSE_S="1")
        self.studio.start()
        parent = self.clip(task="fl2va", prompt=KEYFRAME_PROMPT, seed=3, files={"first_frame": [("f.png", png())]})
        want = load_rewrites().text_only_prompt(KEYFRAME_PROMPT)
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "text": "encode"})
        self.assertEqual(self.studio.state()["session"]["state"], "ready")
        uid = self.upscale(parent["id"], resolution="1080p")
        seen, rec = self.statuses(uid)
        self.assertEqual(rec["state"], "done", rec.get("error"))
        steps = [st for st, _ in seen if st not in ("queued", "starting")]          # (those two may pass unseen)
        self.assertEqual([st for i, st in enumerate(steps) if st not in steps[:i]],
                         ["freeing memory", "encoding the prompt", "upscaling", "done"])
        self.assertIn(("encoding the prompt", "prompt"), seen)
        job = self.studio.job(uid)
        self.assertEqual(job["text"], {"source": "encode", "prompt": want["prompt"], "dropped": want["dropped"],
                                       "notes": want["notes"], "rows": job["text"]["rows"],
                                       "encode_s": job["text"]["encode_s"]})
        self.assertNotIn("Picture", job["text"]["prompt"])
        self.assertGreater(job["text"]["rows"], 0)
        self.assertEqual((job["parent_model"], job["video"]), ("sol-h3", "clip"))
        self.assertNotIn("parent_opts", job)
        case = Path(job["case_dirs"][0])
        view = case / "input"
        encode, = self.encodes()
        self.assertEqual(encode["argv"], ["--case", str(view), "--prompt-file", str(case / "prompt-text-only.txt"),
                                          "--orientation", "landscape", "--seed", "3", "--case-id", job["case_id"],
                                          "--package", str(self.studio.package), "--upscale"])
        self.assertEqual((case / "prompt-text-only.txt").read_text(encoding="utf-8"), want["prompt"])
        self.assertEqual(sorted(p.name for p in case.iterdir()),
                         ["encode.log", "input", "prompt-text-only.txt", "regen"])
        self.assertEqual(sorted(str(p.relative_to(view)) for p in view.rglob("*") if p.is_file()),
                         ["encode.json", "qwen-worker/worker.log", "qwen/conditioning.json", "qwen/conditioning.pt",
                          "source.json", f"stage1/stage1_direct_tensors.pt", f"stage2/{Path(parent['output']).name}"])
        parent_case = Path(parent["output"]).parents[1]
        self.assertEqual(inode(view / "stage2" / Path(parent["output"]).name), inode(parent["output"]))
        self.assertEqual(inode(view / "stage1" / "stage1_direct_tensors.pt"),
                         inode(parent_case / "stage1" / "stage1_direct_tensors.pt"))
        cond = json.loads((view / "qwen" / "conditioning.json").read_text())
        self.assertEqual((cond["task"], cond["prompt"]), ("t2va", want["prompt"]))
        self.assertEqual(json.loads((case / "regen" / "job.json").read_text())["case"], str(view))
        source = json.loads((view / "source.json").read_text())
        self.assertEqual({k: source[k] for k in ("parent", "model", "options", "task", "orientation", "seconds",
                                                 "video")},
                         {"parent": parent["id"], "model": "sol-h3", "options": None, "task": "fl2va",
                          "orientation": "landscape", "seconds": 5, "video": "clip"})
        self.assertEqual(source["text"]["prompt"], want["prompt"])
        self.assertEqual(source["files"][f"stage2/{Path(parent['output']).name}"],
                         {"from": parent["output"], "how": "link",
                          "sha256": hashlib.sha256(Path(parent["output"]).read_bytes()).hexdigest()})
        self.assertEqual(source["files"]["qwen/conditioning.pt"]["how"], "encode")
        self.assertEqual(source["files"]["qwen/conditioning.pt"]["sha256"],
                         hashlib.sha256((view / "qwen" / "conditioning.pt").read_bytes()).hexdigest())
        self.assertIn("encoding the text of", self.studio.log.read_text())
        # the mux took the clip's own sound; Delete and Undo move the input folder with the rest
        self.assertEqual(probe(job["output"])["audio"]["duration"], probe(parent["output"])["audio"]["duration"])
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": uid}), (200, {"ok": True}))
        self.assertFalse(case.exists())
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": uid})[0], 200)
        self.assertTrue((view / "source.json").is_file())

    def test_a_prompt_with_crlf_line_breaks(self):
        """The page's form sends a prompt's line breaks as CRLF (a browser's multipart), and the studio keeps them;
        the prompt encode reads its prompt file with LF ones. The text the upscale records, writes and checks the
        receipt against has LF line breaks, so the upscale finishes."""
        prompt = KEYFRAME_PROMPT.replace("\n", "\r\n")
        parent = self.clip(task="fl2va", prompt=prompt, seed=8, files={"first_frame": [("f.png", png())]})
        self.assertEqual(self.studio.job(parent["id"])["prompt"], prompt)                 # kept as it came
        self.assertEqual(self.studio.job(parent["id"])["upscalable"]["text"], "encode")
        job = self.done(self.upscale(parent["id"]))
        want = load_rewrites().text_only_prompt(KEYFRAME_PROMPT)["prompt"]
        case = Path(job["case_dirs"][0])
        self.assertEqual((job["text"]["prompt"], (case / "prompt-text-only.txt").read_bytes().decode()), (want, want))
        self.assertNotIn("\r", job["text"]["prompt"])
        self.assertEqual(json.loads((case / "input" / "qwen" / "conditioning.json").read_text())["prompt"], want)
        self.assertEqual(json.loads((case / "input" / "source.json").read_text())["text"]["prompt"], want)

    def test_the_clip_deleted_afterwards(self):
        """The input folder's files are links: Delete of the clip, and its trash emptied, leave the upscale playable
        and its input folder whole."""
        parent = self.clip(task="fl2va", prompt=KEYFRAME_PROMPT, seed=9, files={"first_frame": [("f.png", png())]})
        job = self.done(self.upscale(parent["id"]))
        mp4 = Path(job["case_dirs"][0]) / "input" / "stage2" / Path(parent["output"]).name
        self.assertEqual(os.stat(mp4).st_nlink, 2)
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": parent["id"]}), (200, {"ok": True}))
        self.assertFalse(Path(parent["output"]).exists())
        trash = self.studio.runtime / "trash" / parent["id"]
        self.assertTrue(trash.is_dir())
        shutil.rmtree(trash)                                                     # as the trash is emptied later
        rec = json.loads(self.studio.get(f"/api/jobs/{job['id']}")[2])
        self.assertEqual(rec["state"], "done")
        status, _, body = self.studio.get(rec["urls"]["video"])
        self.assertEqual((status, body), (200, Path(job["output"]).read_bytes()))
        self.assertEqual((os.stat(mp4).st_nlink, probe(mp4)["video"]["nb_frames"]), (1, "121"))
        self.assertTrue((mp4.parents[1] / "stage1" / "stage1_direct_tensors.pt").is_file())

    def test_no_t2va_manifest(self):
        """The upscale's encode is a t2va one: without the t2va path manifest a clip whose text must be encoded is
        refused with the reason (a Text clip, which needs no encode, is not)."""
        kf = self.clip(task="fl2va", prompt=KEYFRAME_PROMPT, seed=10, files={"first_frame": [("f.png", png())]})
        text = self.clip(seed=11)
        (self.studio.root / "paths-t2va.json").unlink()
        why = ("the Upscale encodes this clip's prompt as text alone first, which needs the t2va path manifest (run "
               "install/70-manifests.sh)")
        self.assertEqual(self.studio.job(kf["id"])["upscalable"], {"ok": False, "why": why})
        self.assertEqual(self.studio.upscale(kf["id"]), (409, {"error": why}))
        self.assertEqual(self.studio.job(text["id"])["upscalable"], {"ok": True, "clip_audio": True})

    def test_every_set_of_keyframes(self):
        """A last frame alone and first + last: the same text-only route; the encode never gets a keyframe."""
        line = "For the target video, at 5.00 seconds into the target video, <Picture 1> is fully referenced."
        for files, seed in (({"last_frame": [("l.png", png())]}, 4),
                            ({"first_frame": [("f.png", png())], "last_frame": [("l.png", png())]}, 5)):
            parent = self.clip(task="fl2va", prompt=line + "\nA red kite rises.", seed=seed, files=files)
            self.assertEqual(self.studio.job(parent["id"])["upscalable"]["text"], "encode")
            job = self.done(self.upscale(parent["id"]))
            self.assertEqual(job["text"]["prompt"], "A red kite rises.")
        self.assertEqual(len(self.encodes()), 2)
        self.assertFalse(any(a in r["argv"] for r in self.encodes() for a in ("--first", "--last")))

    def test_a_reference_clip(self):
        """Its prompt as t2va's three fields (ref_text_only_prompt): no labels, the dialogue as it was."""
        parent = self.clip(task="ref2va", prompt=REFERENCE_PROMPT, seed=6, orientation="portrait",
                           files={"ref_images": [("r.png", png())]})
        want = load_rewrites().ref_text_only_prompt(REFERENCE_PROMPT)
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "text": "encode"})
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual({k: job["text"].get(k) for k in ("source", "prompt", "dropped", "notes")},
                         {"source": "encode", "prompt": want["prompt"], "dropped": want["dropped"], "notes": []})
        text = job["text"]["prompt"]
        self.assertTrue(text.startswith("integrated_multimodal_description: [Shot 1] A medium shot frames the woman "
                                        "(S1) on a snowy street. She says, <d>[English] Welcome back.</d>"), text)
        self.assertIsNone(re.search(r"<(?:Picture|Audio|Video|Subject) \d+>", text))
        case = Path(job["case_dirs"][0])
        self.assertEqual((case / "prompt-text-only.txt").read_text(encoding="utf-8"), text)
        encode, = self.encodes()
        self.assertEqual(encode["argv"][encode["argv"].index("--orientation") + 1], "portrait")
        self.assertEqual(json.loads((case / "input" / "qwen" / "conditioning.json").read_text())["prompt"], text)

    def test_nothing_left_of_a_reference_prompt(self):
        parent = self.clip(task="ref2va", prompt="<Picture 1>", files={"ref_images": [("r.png", png())]})
        why = "nothing is left of the prompt without its references: the Upscale needs text"
        self.assertEqual(self.studio.job(parent["id"])["upscalable"], {"ok": False, "why": why})
        self.assertEqual(self.studio.upscale(parent["id"]), (400, {"error": why}))

    def test_reference_prompts_with_chains_of_definitions(self):
        """A chain of 1000 subject definitions (<Subject 1> is <Subject 2>, ...), which a recursive rewrite could
        not resolve, and one that doubles at each link: /api/state, /api/jobs, the record and /api/upscale answer
        (they ask for the rewrite of every finished clip), the first upscalable, the second refused as too long."""
        def prompt(n, ref):
            lines = [f"<Subject {i}> is {ref.format(i + 1)}" for i in range(1, n)] + [f"<Subject {n}> is a dog"]
            return "subject_definitions:\n" + "\n".join(lines) + "\n\ndetailed_description: <Subject 1> runs."
        long = self.clip(task="ref2va", prompt=prompt(1000, "<Subject {}>"), files={"ref_images": [("r.png", png())]})
        wide = self.clip(task="ref2va", prompt=prompt(40, "<Subject {0}> and <Subject {0}>"),
                         files={"ref_images": [("r.png", png())]})
        too_long = {"ok": False, "why": "the prompt is too long as text alone: with its subjects written out it has "
                                        "more than 32768 characters"}
        for jobs in (self.studio.state()["jobs"], json.loads(self.studio.get("/api/jobs")[2])["jobs"]):
            found = {j["id"]: j.get("upscalable") for j in jobs}
            self.assertEqual((found[long["id"]], found[wide["id"]]),
                             ({"ok": True, "clip_audio": True, "text": "encode"}, too_long))
        self.assertEqual(self.studio.upscale(wide["id"]), (400, {"error": too_long["why"]}))
        job = self.done(self.upscale(long["id"]))
        self.assertEqual(job["text"]["prompt"], "integrated_multimodal_description: A dog runs.")
        self.assertNotIn("Traceback", self.studio.log.read_text())

    def test_a_text_clip_reads_its_own_folder_too(self):
        """A Sol-H3 Text clip: no encode, its own qwen/ linked into the input folder."""
        parent = self.clip()
        job = self.done(self.upscale(parent["id"]))
        view = Path(job["case_dirs"][0]) / "input"
        parent_case = Path(parent["output"]).parents[1]
        self.assertEqual(inode(view / "qwen" / "conditioning.pt"), inode(parent_case / "qwen" / "conditioning.pt"))
        self.assertEqual((job["text"], self.encodes()), ({"source": "clip"}, []))
        self.assertEqual(json.loads((view / "source.json").read_text())["files"]["qwen/conditioning.pt"]["how"], "link")


class UpscaleEncode(EveryClipParts, UpscaleTest):
    """The upscale's prompt encode: Cancel stops it at once (and its Qwen worker), a failed encode or one whose receipt
    does not match fails the upscale with its reason, and the memory floor is checked again after it."""
    env = {"STUB_UPSCALE_S": "1", "STUB_QWEN_S": "30"}

    def keyframes_clip(self, seed=3):
        return self.clip(task="fl2va", prompt=KEYFRAME_PROMPT, seed=seed, files={"first_frame": [("f.png", png())]})

    def encode_pid(self, uid):
        self.until(uid, lambda j: j["status"] == "encoding the prompt")
        deadline = time.time() + 10
        while not self.encodes() and time.time() < deadline:
            time.sleep(0.05)
        return self.encodes()[-1]["pid"]

    def test_cancel_during_the_encode(self):
        self.studio.stop()
        self.studio.env["STUB_QWEN_WORKER_S"] = "60"
        self.studio.start()
        uid = self.upscale(self.keyframes_clip()["id"])
        pid = self.encode_pid(uid)
        self.assertEqual(worker_groups(self.studio.proc.pid), [pid])            # in a process group of its own
        rec = json.loads(self.studio.get(f"/api/jobs/{uid}")[2])
        self.assertEqual((rec["state"], rec["phase"], rec["upscale_phase"]), ("running", "encoding", "prompt"))
        self.assertNotIn(rec["status"], idle_values())                           # busy for spark-switchboard
        view = Path(rec["case_dirs"][0]) / "input"
        deadline = time.time() + 10
        while not (view / "qwen-worker" / "worker.pid").is_file() and time.time() < deadline:
            time.sleep(0.05)
        worker = int((view / "qwen-worker" / "worker.pid").read_text())
        self.addCleanup(lambda: alive(worker) and os.killpg(worker, signal.SIGKILL))
        t0 = time.time()
        self.assertEqual(self.cancel(uid)[0], 200)
        job = self.studio.wait([uid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertLess(time.time() - t0, 5, "stopped right away")
        self.assertFalse(alive(pid))
        gone_soon(self, worker)                                                  # it closed its worker
        self.assertEqual((self.job_pids(), worker_groups(self.studio.proc.pid)), ([], []))   # no job started
        self.assertFalse((Path(job["case_dirs"][0]) / "regen" / "job.json").exists())
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": uid}), (200, {"ok": True}))

    def test_an_encode_that_ignores_sigterm(self):
        """SIGKILL 20 s after the SIGTERM, and the server stops the Qwen worker it left."""
        self.studio.stop()
        self.studio.env.update(STUB_QWEN_WORKER_S="90", STUB_QWEN_TERM="ignore")
        self.studio.start()
        uid = self.upscale(self.keyframes_clip()["id"])
        pid = self.encode_pid(uid)
        view = Path(self.studio.job(uid)["case_dirs"][0]) / "input"
        deadline = time.time() + 10
        while not (view / "qwen-worker" / "worker.pid").is_file() and time.time() < deadline:
            time.sleep(0.05)
        worker = int((view / "qwen-worker" / "worker.pid").read_text())
        self.addCleanup(lambda: alive(worker) and os.killpg(worker, signal.SIGKILL))
        self.assertEqual(self.cancel(uid)[0], 200)
        self.assertEqual(self.studio.wait([uid], timeout=40)[0]["status"], "cancelled")
        self.assertFalse(alive(pid))
        gone_soon(self, worker)
        log = self.studio.log.read_text()
        self.assertIn(f"cancel: the upscale's prompt encode {pid} is still there 20 s after SIGTERM: SIGKILL", log)
        self.assertIn("stopping the Qwen worker the prompt encode left", log)

    def test_failures(self):
        parent = self.keyframes_clip()
        self.studio.stop()
        for hook, why in (("1", "UpscaleFailed: the prompt encode failed: RuntimeError: Worker exited with code 1 "
                                "(STUB_QWEN_FAIL)"),
                          ("receipt", "UpscaleFailed: the prompt encode failed: the qwen/ encode does not match: "
                                      "['prompt']"),
                          # an encode that ends well (exit 0), and the server's own checks of what it wrote
                          ("no-conditioning", "UpscaleFailed: the prompt encode wrote no qwen/conditioning.pt and "
                                              ".json; see {log}"),
                          ("other-prompt", "UpscaleFailed: the prompt encode's conditioning is not of the text asked "
                                           "for: its prompt differs (task t2va); see {log}"),
                          ("other-task", "UpscaleFailed: the prompt encode's conditioning is not of the text asked "
                                         "for: its task differs (task fl2va); see {log}")):
            self.studio.env.update(STUB_QWEN_S="0.2", STUB_QWEN_FAIL=hook)
            self.studio.start()
            job = self.studio.wait([self.upscale(parent["id"])], timeout=30)[0]
            log = Path(job["case_dirs"][0]) / "encode.log"
            self.assertEqual((job["status"], job["error"]), ("failed", why.format(log=log)))
            self.assertEqual(self.job_pids(), [])
            self.studio.stop()

    def test_the_floor_again_after_the_encode(self):
        parent = self.keyframes_clip()
        self.studio.stop()
        self.studio.env["STUB_QWEN_S"] = "2"
        self.studio.start()
        uid = self.upscale(parent["id"])
        self.encode_pid(uid)
        self.studio.set_memory(60)                       # something else took memory while the encode ran
        job = self.studio.wait([uid], timeout=30)[0]
        self.assertEqual(job["status"], "failed")
        self.assertIn("not enough free memory for an upscale: 60.0 GiB available", job["error"])
        self.assertEqual(self.job_pids(), [])


class UpscaleOff(StudioTest):
    """LFSTUDIO_UPSCALE=0 (the default outside the tests): /api/upscale answers 404, nothing has "upscalable", and the
    other upscale settings are not read."""

    def seed(self):
        self.studio.env.update(LFSTUDIO_UPSCALE="0", LFSTUDIO_UPSCALE_STRENGTH="nonsense")

    def test_off(self):
        jid = self.generate()
        self.studio.wait([jid])
        st = self.studio.state()
        self.assertIs(st["upscale"], False)
        self.assertIsNone(st["upscale_defaults"])
        self.assertIsNone(st["upscale_load"])
        self.assertNotIn("upscalable", st["jobs"][0])
        self.assertEqual(self.studio.upscale(jid), (404, {"error": "the experimental Upscale is off on this studio "
                                                                   "(LFSTUDIO_UPSCALE=1 turns it on)"}))
        older = json.loads(self.studio.get(f"/api/jobs?before={time.time() + 60}")[2])["jobs"]
        self.assertNotIn("upscalable", older[0])
        self.assertNotIn("upscalable", json.loads(self.studio.get(f"/api/jobs/{jid}")[2]))
        self.assertNotIn("Upscale", self.studio.log.read_text())
        self.assertEqual([j["id"] for j in self.studio.state()["jobs"]], [jid])


class BadUpscaleSettings(unittest.TestCase):
    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, d, True)
        for i, (env, message) in enumerate((
                ({"LFSTUDIO_UPSCALE": "2"}, "LFSTUDIO_UPSCALE=2: must be 0 (off, the default) or 1 (on) (see "
                                            "bin/env.example)"),
                ({"LFSTUDIO_UPSCALE": "on"}, "LFSTUDIO_UPSCALE=on: must be 0 (off, the default) or 1 (on)"),
                ({"LFSTUDIO_UPSCALE_STRENGTH": "nvidia"},
                 "LFSTUDIO_UPSCALE_STRENGTH=nvidia: must be faithful, balanced or full (see bin/env.example)"),
                ({"LFSTUDIO_UPSCALE_AUDIO": "both"}, "LFSTUDIO_UPSCALE_AUDIO=both: must be clip or silence"),
                ({"LFSTUDIO_UPSCALE_PREREAD": "yes"}, "LFSTUDIO_UPSCALE_PREREAD=yes: must be 0 or 1"),
                ({"LFSTUDIO_UPSCALE_LOAD": "quick"}, "LFSTUDIO_UPSCALE_LOAD=quick: must be fast or standard"),
                ({"LFSTUDIO_UPSCALE_PREREAD": "1"}, "LFSTUDIO_UPSCALE_PREREAD=1: only with "
                                                    "LFSTUDIO_UPSCALE_LOAD=standard (the fast load reads the weight "
                                                    "files past the page cache) (see bin/env.example)"),
                ({"LFSTUDIO_UPSCALE_MIN_FREE_GIB": "-1"}, "LFSTUDIO_UPSCALE_MIN_FREE_GIB=-1: must be a number of GiB, "
                                                          "0 or more"),
                ({"LFSTUDIO_UPSCALE_GUARD_GIB": "nan"}, "LFSTUDIO_UPSCALE_GUARD_GIB=nan: must be a number of GiB"),
                ({"LFSTUDIO_UPSCALE_PKG": "relative/package"},
                 "LFSTUDIO_UPSCALE_PKG=relative/package: must be an absolute path"),
                ({"LFSTUDIO_UPSCALE_AUDIO": "both", "LFSTUDIO_UPSCALE_PREREAD": "2"},
                 "LFSTUDIO_UPSCALE_AUDIO=both: must be clip or silence; LFSTUDIO_UPSCALE_PREREAD=2: must be 0 or 1 "
                 "(see bin/env.example)"))):
            with self.assertRaises(RuntimeError) as e:
                StubStudio(d / str(i), env=env).start(timeout=10)
            self.assertIn(message, str(e.exception))
        studio = StubStudio(d / "empty", env={"LFSTUDIO_UPSCALE": ""}).start()      # empty: the default, off
        try:
            self.assertIs(studio.state()["upscale"], False)
        finally:
            studio.stop()

    def test_the_prompt_encode_and_its_rewrites(self):
        """With the Upscale on, the prompt encode it runs for a Keyframes or Reference clip's text
        (LFSTUDIO_H3TURBO_ENCODE_SCRIPT, with LightX2V Turbo off too) must be there, and the repo's
        bench/h3turbo_encode.py, whose rewrites the server imports, readable (here a copy of the server whose copy of
        it is broken)."""
        d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, d, True)
        with self.assertRaises(RuntimeError) as e:
            StubStudio(d / "a", env={"LFSTUDIO_H3TURBO_ENCODE_SCRIPT": str(d / "none.py")}).start(timeout=10)
        self.assertIn(f"LFSTUDIO_UPSCALE=1: no prompt encode at {d / 'none.py'}", str(e.exception))
        repo = d / "repo"
        for rel in ("webui/server.py", "upscale/regen2k_job.py"):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_DIR / rel, repo / rel)
        (repo / "bench").mkdir()
        (repo / "bench" / "h3turbo_encode.py").write_text("def broken(:\n")
        with self.assertRaises(RuntimeError) as e:
            StubStudio(d / "b", server=repo / "webui" / "server.py").start(timeout=10)
        self.assertIn(f"LFSTUDIO_UPSCALE=1: cannot read {repo / 'bench' / 'h3turbo_encode.py'} (", str(e.exception))
        studio = StubStudio(d / "c", server=repo / "webui" / "server.py", env={"LFSTUDIO_UPSCALE": "0"}).start()
        try:                                                    # with the Upscale off, nothing reads it
            self.assertIs(studio.state()["upscale"], False)
        finally:
            studio.stop()


class LauncherUpscale(LauncherSettingsCheck):
    """bin/lf-studio with LFSTUDIO_UPSCALE: the value, and with 1 the settings (as the server reads them), the
    Regen-2K Python, NVIDIA's package and the weights, checked by upscale/regen2k_job.py's own code; --kill ends an
    upscale's job that the server left behind."""
    test_settings = test_api_setting = test_older_patch = None

    def regen2k(self):
        """A stand-in Regen-2K install (NVIDIA's package and the weights, as empty files): its settings."""
        root = self.d / "regen2k"
        (root / "package" / "open_h3").mkdir(parents=True)
        (root / "package" / "open_h3" / "pipeline.py").write_text("")
        for name in ("lora/open_h3_regenerate_2k_lora.safetensors", "lora/sharp_curve_branches.safetensors",
                     "lora/adaln_t_table.bin",
                     "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"):
            (root / "weights" / name).parent.mkdir(parents=True, exist_ok=True)
            (root / "weights" / name).write_bytes(b"")
        for part in ("transformer", "vae", "audio_vae"):
            (root / "base" / part).mkdir(parents=True)
            for name in ("config.json", "diffusion_pytorch_model.safetensors"):
                (root / "base" / part / name).write_bytes(b"")
        return {"LFSTUDIO_UPSCALE": "1", "LFSTUDIO_UPSCALE_PYTHON": sys.executable,
                "LFSTUDIO_UPSCALE_PKG": str(root / "package"), "LFSTUDIO_UPSCALE_WEIGHTS": str(root / "weights"),
                "LFSTUDIO_UPSCALE_BASE": str(root / "base")}

    def refused(self, why, **settings):
        code, out = self.check(**settings)
        self.assertEqual(code, 1, out)
        self.assertIn(f"REFUSED - {why}", out)

    def test_the_switch(self):
        for value in ("0", ""):
            code, out = self.check(LFSTUDIO_UPSCALE=value, LFSTUDIO_UPSCALE_STRENGTH="not read")
            self.assertEqual(code, 0, out)
            self.assertNotIn("Upscale", out)
        for value in ("2", "on", "true"):
            self.refused(f"LFSTUDIO_UPSCALE={value}: must be 0 (off, the default) or 1 (on) (see bin/env.example)",
                         LFSTUDIO_UPSCALE=value)

    def test_paths_and_settings(self):
        settings = self.regen2k()
        code, out = self.check(**settings)
        self.assertEqual(code, 0, out)
        self.assertIn("the experimental Upscale is on (LFSTUDIO_UPSCALE=1): its environment, package and weights "
                      "are there", out)
        upscale = "the experimental Upscale (LFSTUDIO_UPSCALE=1): "
        self.refused("LFSTUDIO_UPSCALE_STRENGTH=max: must be faithful, balanced or full (see bin/env.example)",
                     **settings, LFSTUDIO_UPSCALE_STRENGTH="max")
        self.refused("LFSTUDIO_UPSCALE_GUARD_GIB=-5: must be a number of GiB, 0 or more (see bin/env.example)",
                     **settings, LFSTUDIO_UPSCALE_GUARD_GIB="-5")
        self.refused("LFSTUDIO_UPSCALE_LOAD=mmap: must be fast or standard (see bin/env.example)",
                     **settings, LFSTUDIO_UPSCALE_LOAD="mmap")
        self.refused("LFSTUDIO_UPSCALE_PREREAD=1: only with LFSTUDIO_UPSCALE_LOAD=standard",
                     **settings, LFSTUDIO_UPSCALE_PREREAD="1")
        code, out = self.check(**settings, LFSTUDIO_UPSCALE_LOAD="standard", LFSTUDIO_UPSCALE_PREREAD="1")
        self.assertEqual(code, 0, out)
        self.refused("LFSTUDIO_UPSCALE_WEIGHTS=weights: must be an absolute path",
                     **dict(settings, LFSTUDIO_UPSCALE_WEIGHTS="weights"))
        self.refused(f"{upscale}no Python at LFSTUDIO_UPSCALE_PYTHON={self.d / 'nothing'} (the "
                     "Open-H3-Regenerate-2K environment); set it up, or set LFSTUDIO_UPSCALE=0",
                     **dict(settings, LFSTUDIO_UPSCALE_PYTHON=str(self.d / "nothing")))
        self.refused(f"{upscale}no prompt encode at {self.d / 'none.py'} (it encodes the text of Keyframes and "
                     "Reference clips); set it up, or set LFSTUDIO_UPSCALE=0",
                     **settings, LFSTUDIO_H3TURBO_ENCODE_SCRIPT=str(self.d / "none.py"))
        lora = self.d / "regen2k" / "weights" / "lora" / "open_h3_regenerate_2k_lora.safetensors"
        lora.unlink()
        self.refused(f"{upscale}missing weights: {lora}; set it up", **settings)
        (self.d / "regen2k" / "package" / "open_h3" / "pipeline.py").unlink()
        self.refused(f"{upscale}LFSTUDIO_UPSCALE_PKG: no open_h3/pipeline.py in {self.d / 'regen2k' / 'package'}",
                     **settings)

    def test_the_fast_load_needs_its_loader(self):
        """A checkout without upscale/fastload.py (here a copy of the launcher, the server, the job and the prompt
        encode) is refused with the fast load, the default, and passes with the standard load, which does without
        it."""
        repo = self.d / "repo"
        for rel in ("bin/lf-studio", "webui/server.py", "upscale/regen2k_job.py", "bench/h3turbo_encode.py"):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__).resolve().parents[1] / rel, repo / rel)
        settings = self.regen2k()
        self.refused(f"the experimental Upscale (LFSTUDIO_UPSCALE=1): no {repo / 'upscale' / 'fastload.py'}, the fast "
                     "load's loader (part of LF Studio; the standard load, LFSTUDIO_UPSCALE_LOAD=standard, does "
                     "without it)", launcher=repo / "bin/lf-studio", **settings)
        code, out = self.check(repo / "bin/lf-studio", **settings, LFSTUDIO_UPSCALE_LOAD="standard")
        self.assertEqual(code, 0, out)

    def test_a_note_for_other_nvidia_files(self):
        """NVIDIA's files the job's 9:16 route is pinned to (NVIDIA_FILES, by sha256): other or missing ones get a note
        at start, not a refusal (the job refuses only a 9:16 clip with them); the pinned ones none. Here a copy of the
        launcher, the server and the job with the stand-in files pinned."""
        settings = self.regen2k()
        package = Path(settings["LFSTUDIO_UPSCALE_PKG"])
        upscale_on = ("the experimental Upscale is on (LFSTUDIO_UPSCALE=1): its environment, package and weights are "
                      "there")
        note = ("note: NVIDIA's package (LFSTUDIO_UPSCALE_PKG) has other files than the Upscale's 9:16 route was "
                "written for ({}; see docs/TROUBLESHOOTING.md): upscales of 9:16 clips will be refused (16:9 clips "
                "are not checked against these files)")
        code, out = self.check(**settings)
        self.assertEqual(code, 0, out)
        self.assertIn(upscale_on, out)
        self.assertIn(note.format("open_h3/pipeline.py changed, open_h3/upscaler.py missing, "
                                  "open_h3/upscaler_node.py missing"), out)
        repo = self.d / "repo"
        for rel in ("bin/lf-studio", "webui/server.py", "upscale/regen2k_job.py", "upscale/fastload.py",
                    "bench/h3turbo_encode.py"):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__).resolve().parents[1] / rel, repo / rel)
        job = (repo / "upscale/regen2k_job.py").read_text()
        pins = dict(re.findall(r'"(open_h3/\w+\.py)": "([0-9a-f]{64})"', job))
        self.assertEqual(sorted(pins), ["open_h3/pipeline.py", "open_h3/upscaler.py", "open_h3/upscaler_node.py"])
        for rel, digest in pins.items():
            (package / rel).write_text(f"# a stand-in of {rel}\n")
            job = job.replace(digest, hashlib.sha256((package / rel).read_bytes()).hexdigest())
        (repo / "upscale/regen2k_job.py").write_text(job)
        code, out = self.check(repo / "bin/lf-studio", **settings)
        self.assertEqual(code, 0, out)
        self.assertIn(upscale_on, out)
        self.assertNotIn("note: NVIDIA's package", out)
        (package / "open_h3" / "upscaler_node.py").write_text("# another version\n")
        code, out = self.check(repo / "bin/lf-studio", **settings)
        self.assertEqual(code, 0, out)
        self.assertIn(note.format("open_h3/upscaler_node.py changed"), out)

    def test_defaults(self):
        """The defaults under LFSTUDIO_HOME and LFSTUDIO_CHECKPOINTS, as in bin/env.example."""
        home, upscale = self.d / "home", "the experimental Upscale (LFSTUDIO_UPSCALE=1): "
        self.refused(f"{upscale}no Python at LFSTUDIO_UPSCALE_PYTHON={home / 'envs' / 'regen2k' / 'bin' / 'python'}",
                     LFSTUDIO_UPSCALE="1", LFSTUDIO_HOME=str(home))
        self.refused(f"{upscale}LFSTUDIO_UPSCALE_PKG: no open_h3/pipeline.py in {home / 'Open-H3-Regenerate-2K'}",
                     LFSTUDIO_UPSCALE="1", LFSTUDIO_HOME=str(home), LFSTUDIO_UPSCALE_PYTHON=sys.executable)
        settings = self.regen2k()
        self.refused(f"{upscale}missing weights: {home / 'checkpoints' / 'Open-H3-Regenerate-2K' / 'lora'}",
                     LFSTUDIO_UPSCALE="1", LFSTUDIO_HOME=str(home), LFSTUDIO_UPSCALE_PYTHON=sys.executable,
                     LFSTUDIO_UPSCALE_PKG=settings["LFSTUDIO_UPSCALE_PKG"])

    def test_kill_ends_a_left_over_job(self):
        """--kill (and the launcher's cleanup) find an upscale's job by its command line: upscale/regen2k_job.py with
        a job.json under this runtime root's web-UI outputs, not one elsewhere."""
        script = Path(__file__).resolve().parents[1] / "upscale" / "regen2k_job.py"
        runtime = self.d / "runtime"
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)", str(script), "--job"]
        job = runtime / "outputs" / "webui" / "20261028-000000-upscale-2k" / "up_x" / "regen" / "job.json"
        mine = subprocess.Popen(sleeper + [str(job)], start_new_session=True)
        other = subprocess.Popen(sleeper + [str(self.d / "bench" / "job.json")], start_new_session=True)
        self.addCleanup(lambda: [p.kill() or p.wait() for p in (mine, other) if p.poll() is None])
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_"))}
        # a Sana checkout that is not there: the pattern for the pipeline's workers names only this temp directory
        env.update(LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_SANA=str(self.d / "no-sana"),
                   LFSTUDIO_PYTHON=sys.executable, SOL_H3_SPARK_RUNTIME_ROOT=str(runtime),
                   LFSTUDIO_STATE_DIR=str(self.d / "state"), LFSTUDIO_SERVER=str(self.d / "no-server.py"),
                   LFSTUDIO_TEST_SKIP_DOCKER="1")
        run = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "bin/lf-studio"), "--kill"], env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(mine.wait(timeout=10), -signal.SIGKILL)
        self.assertIsNone(other.poll())


# ------------------------------------------------------------------------------------------ the model choice
TURBO, BOTH = "lightx2v-turbo", "sol-h3,lightx2v-turbo"
LORA4 = "minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors"
NOT_REF2VA = ("LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not Reference (ref2va): use "
              "Sol-H3")
# the registry's estimates (the job's estimate_s, to 10 s): its eta_s until a clip has been measured
TURBO_ETA = {"576p/4/5": 200, "576p/4/6": 250, "576p/4/7": 280, "576p/4/8": 310, "576p/4/10": 400, "576p/4/12": 500,
             "576p/4/15": 660, "768p/4/5": 350, "768p/4/6": 470, "768p/4/7": 530, "768p/4/8": 600, "768p/4/10": 820,
             "768p/4/12": 1070, "768p/4/15": 1440}
REPO_DIR = Path(__file__).resolve().parents[1]


def release_server(ref, directory):
    """An earlier release's server (git show: webui/server.py, the page it serves and the Upscale's job it loads) under
    `directory`: its server.py, or None where git or the release is not there (a copy of the repo without history)."""
    for rel in ("webui/server.py", "webui/index.html", "upscale/regen2k_job.py"):
        try:
            r = subprocess.run(["git", "-C", str(REPO_DIR), "show", f"{ref}:{rel}"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode:
            return None
        (directory / rel).parent.mkdir(parents=True, exist_ok=True)
        (directory / rel).write_bytes(r.stdout)
    return directory / "webui" / "server.py"


def server_function(name, **names):
    """A function of webui/server.py compiled on its own from the server's source, with `names` as its globals (the
    server can't be imported without a studio's settings): for what a stub studio can't reach on cue."""
    tree = ast.parse((REPO_DIR / "webui" / "server.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(names)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "webui/server.py", "exec"), namespace)
    return namespace[name]


class ServerParts(unittest.TestCase):
    """Parts of the server on their own (server_function), with stand-ins for what they use."""

    def test_stop_job_between_gpu_processes(self):
        """stop_job after it signalled a GPU process that has ended, while the job's pipeline call has not returned
        (work["process"] None: an upscale's mux, the receipt read, a LightX2V Turbo clip between its steps): no
        AttributeError, and its SIGKILL went once, to that process only."""
        for stop_s, kills in ((3600, []), (-1, [(4242, signal.SIGKILL)])):
            with self.subTest(GPU_JOB_STOP_S=stop_s):
                job, calls, killed = {"status": "running", "case_id": "web_x"}, [], []
                work = {"job": job, "process": types.SimpleNamespace(pid=4242, poll=lambda: None),
                        "labels": ("the job", "the running job")}

                def signal_workers(sent):
                    calls.append(set(sent))
                    if len(calls) == 1:
                        return [4242]                      # the GPU process gets its SIGTERM
                    if len(calls) == 2:
                        work["process"] = None             # it has ended; the pipeline call goes on
                    else:
                        job["status"] = "cancelled"
                    return []
                stop_job = server_function(
                    "stop_job", lock=threading.Lock(), stopping=threading.Event(), work=work, time=time,
                    FINISHED=("done", "failed", "cancelled"), signal_workers=signal_workers, log=lambda *a: None,
                    GPU_JOB_STOP_S=stop_s, contextlib=contextlib, signal=signal,
                    os=types.SimpleNamespace(killpg=lambda *a: killed.append(a)))
                stop_job(job)                              # returns once the job has finished
                self.assertEqual((len(calls), killed), (3, kills))

    def test_registry_problems(self):
        """registry_problems refuses an option the requests can't carry (only strings and integers: json_form,
        check_model, lf-cli) and values or a default not of its type, besides the tables' own problems."""
        end = {"task": ("t2va", "fl2va"), "seconds": (5, 6)}

        def problems(*options, other=()):
            def model(opts):
                dims = {o["name"]: tuple(c["value"] for c in o["choices"]) for o in opts}
                return {"options": opts, "offers": (dict(end, **dims),), "not_yet": ()}
            models = {"sol-h3": model(()), "turbo": model(options), **({"other": model(other)} if other else {})}
            return server_function("registry_problems", MODELS=models, END_STATE=end,
                                   OPTION_KINDS={"string": str, "integer": int})()
        size = {"name": "resolution", "type": "string", "default": "576p",
                "choices": ({"value": "576p"}, {"value": "768p"})}
        steps = {"name": "steps", "type": "integer", "default": 4, "choices": ({"value": 4}, {"value": 8})}
        self.assertEqual(problems(size, steps), [])
        self.assertEqual(problems(dict(size, type="number", default=0.5, choices=({"value": 0.5},))),
                         ["turbo: option resolution has type 'number', not string or integer"])
        self.assertEqual(problems(dict(steps, choices=({"value": 4}, {"value": "8"}))),
                         ["turbo: option steps: '8' is not of its type, integer"])
        self.assertEqual(problems(dict(steps, default=True, choices=({"value": True}, {"value": 4}))),
                         ["turbo: option steps: True is not of its type, integer",
                          "turbo: option steps: True is not of its type, integer"])
        self.assertEqual(problems(dict(size, default="1080p")),
                         ["turbo: option resolution's default '1080p' is not one of its choices"])
        self.assertEqual(problems(steps, other=(dict(steps, type="string", default="4", choices=({"value": "4"},)),)),
                         ["other: option steps is of type string, another model's of type integer"])

    def test_registry_upscale_entries(self):
        """A model's "upscale" entry may only name the studio's tasks and lengths; the registry's own pass."""
        end = {"task": ("t2va",), "seconds": (5,)}

        def problems(upscale):
            models = {"sol-h3": {"options": (), "offers": (dict(end),), "not_yet": (), "upscale": upscale}}
            return server_function("registry_problems", MODELS=models, END_STATE=end, OPTION_KINDS={},
                                   TASKS=("t2va", "fl2va", "ref2va"), DURATIONS=(5, 6))()
        self.assertEqual(problems({"tasks": ("t2va", "ref2va"), "seconds": (5,)}), [])
        self.assertEqual(problems({"tasks": ("i2v",), "seconds": (5, 3)}),
                         ["sol-h3: the Upscale's task i2v is not one of ('t2va', 'fl2va', 'ref2va')",
                          "sol-h3: the Upscale's length 3 is not one of (5, 6)"])
        registry = server_value("MODELS", SOL_H3="sol-h3", TURBO=TURBO, TASKS=("t2va", "fl2va", "ref2va"),
                                FRAME_SETS=("first", "last", "first+last"), END_STATE={},
                                BACKENDS={"spark": 0, "rtx5090": 0}, DURATIONS=(5, 6, 7, 8, 10, 12, 15))
        self.assertEqual({mid: m["upscale"] for mid, m in registry.items()},
                         {"sol-h3": {"tasks": ("t2va", "fl2va", "ref2va"), "seconds": (5,)},
                          TURBO: {"tasks": ("t2va", "fl2va"), "seconds": (5,)}})

    def test_link_or_copy(self):
        """The input folder's files: hardlinks, or copies where a link can't be made (another filesystem)."""
        tmp = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "a").write_bytes(b"clip")
        link_or_copy = server_function("link_or_copy", os=os, shutil=shutil)
        self.assertEqual(link_or_copy(tmp / "a", tmp / "b"), "link")
        self.assertEqual(inode(tmp / "a"), inode(tmp / "b"))

        def no_link(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        link_or_copy = server_function("link_or_copy", os=types.SimpleNamespace(link=no_link), shutil=shutil)
        self.assertEqual(link_or_copy(tmp / "a", tmp / "c"), "copy")
        self.assertEqual(((tmp / "c").read_bytes(), inode(tmp / "c") != inode(tmp / "a")), (b"clip", True))

    def test_the_stretch_and_the_cut(self):
        """prepare_video_command on a 124-frame 576p clip with sound: 1344x768, 121 frames at 24 fps, its sound cut to
        about the 121 frames (copied); a 768p clip is cut but not scaled."""
        tmp = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        src = tmp / "clip_1024x576_124f.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1024x576:r=24",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=32000:duration=5.1667", "-frames:v",
                        "124", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        str(src)], check=True, timeout=120)
        regen = types.SimpleNamespace(FINAL={"frames": 121, "fps": "24/1"})
        command = server_function("prepare_video_command", regen=regen)
        out = tmp / "refined_1344x768_121f.mp4"
        subprocess.run(command(src, out, (1344, 768), "stretch+trim"), check=True, timeout=120)
        streams = probe(out)
        self.assertEqual((streams["video"]["width"], streams["video"]["height"], streams["video"]["nb_frames"],
                          streams["video"]["r_frame_rate"]), (1344, 768, "121", "24/1"))
        self.assertEqual((streams["audio"]["codec_name"], streams["audio"]["sample_rate"]), ("aac", "32000"))
        self.assertLess(abs(float(streams["audio"]["duration"]) - 121 / 24), 0.07)
        self.assertNotIn("-vf", command(src, out, (1344, 768), "trim"))
        self.assertIn("scale=768:1344:flags=lanczos", command(src, out, (768, 1344), "stretch"))
        # a 576p clip of 121 frames (Text, or a first frame only), stretched alone: its sound is copied as it is, every
        # AAC packet (-t 121/24 or -frames:v 121 would drop the last one)
        src = tmp / "clip_1024x576_121f.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1024x576:r=24",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=32000", "-frames:v", "121",
                        "-af", "atrim=end_sample=161333", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
                        "yuv420p", "-c:a", "aac", str(src)], check=True, timeout=120)
        self.assertFalse({"-t", "-frames:v"} & set(command(src, out, (1344, 768), "stretch")))
        subprocess.run(command(src, out, (1344, 768), "stretch"), check=True, timeout=120)
        streams, clip = probe(out), probe(src)
        self.assertEqual((streams["video"]["width"], streams["video"]["nb_frames"]), (1344, "121"))
        self.assertEqual([streams["audio"][k] for k in ("nb_frames", "duration")],
                         [clip["audio"][k] for k in ("nb_frames", "duration")])

    def test_model_refusal_of_values_made_apart(self):
        """A table of two rows (LightX2V Turbo's before its 15 s and 768p 12 s ran on the GPU: 576p up to 12 s, 768p up
        to 10 s, 15 s not yet): a value no row has gets its reason, values each made in another row that they are not
        made together, and a clip one row has in full is offered."""
        row = {"task": ("t2va", "fl2va"), "orientation": ("landscape", "portrait"), "steps": (4,)}
        models = {"turbo": {"name": "LightX2V Turbo", "offers": (
            dict(row, seconds=(5, 6, 7, 8, 10, 12), resolution=("576p",)),
            dict(row, seconds=(5, 6, 7, 8, 10), resolution=("768p",))),
            "not_yet": ({"seconds": (15,), "why": "not 15 s"},)}}
        offered = server_function("offered", MODELS=models)
        refusal = server_function("model_refusal", MODELS=models, offered=offered)
        clip = lambda seconds, resolution: dict(task="t2va", seconds=seconds, orientation="landscape",  # noqa: E731
                                                resolution=resolution, steps=4)
        self.assertEqual([refusal("turbo", clip(s, r)) for s, r in ((12, "576p"), (10, "768p"), (15, "576p"),
                                                                     (12, "768p"))],
                         [None, None, "not 15 s", "LightX2V Turbo does not make clips with task t2va, seconds 12, "
                                                  "orientation landscape, resolution 768p, steps 4 together yet"])


def release_client(ref, directory):
    """An earlier release's bin/lf-cli (git show) under `directory`, or None where git or the release is not there."""
    try:
        r = subprocess.run(["git", "-C", str(REPO_DIR), "show", f"{ref}:bin/lf-cli"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lf-cli").write_bytes(r.stdout)
    return directory / "lf-cli"


def cli(url, *args, client=None, stderr=False):
    """bin/lf-cli (or `client`, another copy) with these arguments against `url`: (exit status, stdout), with
    `stderr` (exit status, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LFSTUDIO_", "SOLH3_"))}
    env.update(LFSTUDIO_URL=url, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, str(client or REPO_DIR / "bin" / "lf-cli"), *args], env=env,
                         capture_output=True, text=True, timeout=120)
    return (run.returncode, run.stdout, run.stderr) if stderr else (run.returncode, run.stdout)


def normalized(value, *places, ids=()):
    """`value` (JSON-able, or text) with what two studios and two runs can't share made alike: `places` (a studio's
    root and address, an output directory: replaced by their order), clip ids (those in the text and `ids`, also by
    their first 6 or 8 characters), case ids, the time stamps in names, times and durations (any number with a
    decimal point, m:ss)."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    for i, place in enumerate(places):
        text = text.replace(str(place), f"<place {i}>")
    for jid in set(re.findall(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", text)) | set(ids):
        text = re.sub(rf"(?<![0-9a-f]){jid[:6]}[0-9a-f]{{0,26}}(?![0-9a-f])", "<id>", text)
    text = re.sub(r"\bweb_\d{6}_<id>", "<case>", text)
    text = re.sub(r"\b\d{8}-\d{6}\b", "<stamp>", text)
    text = re.sub(r"\b\d+\.\d+\b", "<n>", text)
    return re.sub(r"\b\d+:\d\d\b", "<m:ss>", text)


class ModelChoiceOff(unittest.TestCase):
    """With LFSTUDIO_MODELS unset (sol-h3 alone, the default) the studio is v0.5.1's: side by side with v0.5.1's own
    server (from git), the same /api/state, records and API records, the same answers to bad requests and the same
    lf-cli output; the page as it is, without the models' table; LightX2V Turbo's settings are not read. What is new:
    a request that names a model, which v0.5.1's form ignored and its JSON refused, makes the clip with "model":
    "sol-h3" (its record as before) and is refused for a model that is off or unknown."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.studios = {}
        old = release_server("v0.5.1", self.d / "v0.5.1")
        for name, server in (("now", None), ("v0.5.1", old)):
            if name == "now" or server:
                s = StubStudio(self.d / name, load_s=1, clip_s=0.5, preload=False, server=server)
                s.env.update(LFSTUDIO_H3TURBO_MIN_FREE_GIB="nonsense", LFSTUDIO_H3TURBO_QWEN_CANVAS="nonsense",
                             LFSTUDIO_H3TURBO_SCRIPT=str(self.d / "no-such-job.py"))     # not read while it is off
                self.studios[name] = s.start()
                self.addCleanup(s.stop)
        self.now = self.studios["now"]

    def each(self, fn):
        """fn(studio) on each studio: {name: result}."""
        return {name: fn(s) for name, s in self.studios.items()}

    def same(self, results, what):
        if len(results) == 2:
            self.assertEqual(results["now"], results["v0.5.1"], what)
        return results["now"]

    def test_as_v051(self):
        if len(self.studios) < 2:
            self.skipTest("v0.5.1 is not in this copy's git history; the rest is checked without it")
        def start_log(s):
            text = s.log.read_text().split("LF Studio on")[0].replace(str(s.root), "ROOT")
            return re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", text, flags=re.M)
        self.same(self.each(start_log), "the start log")
        out = self.d / "out"
        here = lambda s, value: normalized(value, s.root, s.url, out / s.root.name)            # noqa: E731
        st = self.same(self.each(lambda s: here(s, s.state())), "/api/state")
        self.assertNotIn('"models"', st)
        self.assertIn('"session"', st)
        clips = self.each(lambda s: cli(s.url, "generate", "--prompt", "a test clip", "--seed", "1", "--json", "-q",
                                        "--out", str(out / s.root.name)))
        records = {name: json.loads(text) for name, (code, text) in clips.items()}
        self.same({name: (clips[name][0], here(self.studios[name], rec)) for name, rec in records.items()},
                  "lf-cli generate --json")
        rec = records["now"]
        self.assertTrue(Path(rec["path"]).is_file())
        self.assertNotIn("model", rec)
        self.same(self.each(lambda s: here(s, s.state())), "/api/state with the clip")
        api = self.same(self.each(lambda s: here(s, json.loads(s.get(f"/api/jobs/{s.state()['jobs'][0]['id']}")[2]))),
                        "a clip's API record")
        self.assertNotIn("file_name", api)
        for args in (("status",), ("status", "--json"), ("status", "ID"), ("status", "ID", "--json")):
            self.same(self.each(lambda s: here(s, cli(s.url, *[s.state()["jobs"][0]["id"] if a == "ID" else a
                                                               for a in args], stderr=True))),
                      f"lf-cli {' '.join(args)}")
        bad = ({"prompt": "x", "seconds": 4}, {"prompt": "x", "task": "x"}, {"prompt": "  "},
               {"prompt": "x", "first_frame": {"name": "f.png", "data": "!!!!"}})
        for body in bad:
            self.same(self.each(lambda s: s.post("/api/generate", json_body=body)), body)
        self.same(self.each(lambda s: s.post("/api/load", json_body={"task": "t2va"})), "/api/load")
        # the form ignores fields it does not know, as before: a clip of sol-h3, its record as before
        jids = self.each(lambda s: s.generate(seed=2, resolution="768p", steps="8")[1]["id"])
        done = self.each(lambda s: s.wait([jids[s.root.name]])[0])
        self.same({name: here(self.studios[name], job) for name, job in done.items()},
                  "a clip whose form had other fields")
        self.same(self.each(lambda s: here(s, s.state())), "/api/state with both clips")

    def test_v051s_client(self):
        """v0.5.1's lf-cli (from git) and this one print the same against a studio with the model choice off: a
        generate that waits and saves the clip, status and the status of a clip (their usage lines differ: this one
        lists the model's flags too). The generate's progress lines of the clip while it runs are compared apart:
        which a client prints depends on what its first look finds (queued, running or done), not on the client."""
        old = release_client("v0.5.1", self.d / "v0.5.1-client")
        if not old:
            self.skipTest("v0.5.1 is not in this copy's git history")
        s, out = self.now, self.d / "out"
        clients = {"now": None, "v0.5.1": old}
        s.wait([s.generate(seed=3)[1]["id"]])                                   # the models loaded for both

        def each(*args):
            ids = [j["id"] for j in s.state()["jobs"]]
            return {name: normalized(cli(s.url, *args, client=client, stderr=True), s.root, s.url, out, ids=ids)
                    for name, client in clients.items()}
        made = {name: cli(s.url, "generate", "--prompt", "a test clip", "--seed", "4", "--wait", "--out",
                          str(out / f"{name}.mp4"), client=client, stderr=True) for name, client in clients.items()}
        self.assertEqual(made["now"][0], 0, made["now"])
        running = ["starting", "generating", "saving the clip"]          # in the order a clip goes through them
        for name, (code, stdout, stderr) in made.items():
            seen = [line for line in stderr.splitlines() if line in running]
            self.assertEqual(seen, sorted(set(seen), key=running.index), (name, stderr))
            made[name] = (code, stdout, "".join(line for line in stderr.splitlines(True)
                                                if line.rstrip("\n") not in running))
        ids = [j["id"] for j in s.state()["jobs"]]
        self.same({name: normalized(json.dumps(answer).replace(f"{name}.mp4", "<file>"), s.root, s.url, out, ids=ids)
                   for name, answer in made.items()}, "lf-cli generate --wait --out")
        self.same(each("status"), "lf-cli status")
        for job in s.state()["jobs"]:
            self.same(each("status", job["id"]), "lf-cli status ID")
            self.same(each("status", job["id"], "--json"), "lf-cli status ID --json")

    def test_model_named(self):
        s = self.now
        self.assertEqual(s.get("/")[2].decode(), (REPO_DIR / "webui" / "index.html").read_text())   # no lf-models
        for answer in (s.post("/api/generate", json_body={"prompt": "x", "model": TURBO}),
                       s.generate(model=TURBO)):
            self.assertEqual(answer, (400, {"error": f"ValueError: {TURBO} is off on this studio (LFSTUDIO_MODELS)"}))
        self.assertEqual(s.generate(model="flux"), (400, {"error": "ValueError: model must be one of sol-h3"}))
        code, body = s.post("/api/generate", json_body={"prompt": "x", "resolution": "576p"})   # an option: unknown
        self.assertEqual(code, 400)
        self.assertIn('unknown key "resolution"', body["error"])
        self.assertEqual(s.post("/api/load", json_body={"task": "t2va", "orientation": "landscape", "seconds": 5,
                                                        "model": TURBO}),
                         (400, {"error": f"{TURBO} is off on this studio (LFSTUDIO_MODELS)"}))
        for model in (["x"], {"a": 1}, 5):
            self.assertEqual(s.post("/api/load", json_body={"task": "t2va", "orientation": "landscape", "seconds": 5,
                                                            "model": model}),
                             (400, {"error": "model must be a string"}))
        self.assertEqual(s.state()["jobs"], [])
        a = s.post("/api/generate", json_body={"prompt": "x", "model": "sol-h3", "seed": 3})[1]["id"]
        b = s.generate(model="sol-h3", seed=4)[1]["id"]
        c = s.generate(model="", seed=5)[1]["id"]                      # empty: the default
        for job in s.wait([a, b, c]):
            self.assertEqual(job["status"], "done")
            self.assertFalse({"model", "model_opts", "file_name"} & set(job))
        self.assertNotIn("LightX2V Turbo", s.log.read_text())


def qwen_worker(test, jid):
    """The pid of the stand-in Qwen worker (STUB_QWEN_WORKER_S) of LightX2V Turbo clip `jid`'s prompt encode, once it
    runs; killed at the test's end if it is still there."""
    deadline = time.time() + 20
    while time.time() < deadline:
        dirs = test.studio.job(jid).get("case_dirs") or []
        pid_file = Path(dirs[0]) / "qwen-worker" / "worker.pid" if dirs else None
        if pid_file and pid_file.is_file() and pid_file.read_text().endswith("\n"):
            pid = int(pid_file.read_text())
            test.addCleanup(lambda: alive(pid) and os.killpg(pid, signal.SIGKILL))
            return pid
        time.sleep(0.05)
    test.fail(f"no stand-in Qwen worker for {jid}")


def gone_soon(test, pid, within=5):
    deadline = time.time() + within
    while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    test.assertFalse(alive(pid), f"process {pid} is still there")


class TurboTest(UpscaleTest):
    """The experimental LightX2V Turbo on (LFSTUDIO_MODELS=sol-h3,lightx2v-turbo), with tests/stub/h3turbo_job.py
    as its job and tests/stub/h3turbo_encode.py as its prompt encode (tests/stub/studio.py sets their hooks and
    MemAvailable). `env` is set before start; the stand-ins log each run to turbo-jobs.log and encodes.log."""
    env = {}

    def seed(self):
        self.studio.env.update(LFSTUDIO_MODELS=BOTH, STUB_H3TURBO_LOG=str(self.dir / "turbo-jobs.log"),
                               STUB_QWEN_LOG=str(self.dir / "encodes.log"), STUB_UPSCALE_LOG=str(self.dir /
                                                                                                  "upscale-jobs.log"))
        self.studio.env.update(self.env)

    def restart(self, **env):
        self.studio.stop()
        self.studio.env.update(env)
        self.studio.start()

    def turbo(self, **kw):
        code, body = self.studio.turbo(**kw)
        self.assertEqual(code, 200, body)
        self.assertEqual(body["job"]["model"], TURBO)
        return body["id"]

    def runs(self, name):
        """The runs a stand-in logged: turbo-jobs.log or encodes.log."""
        log = self.dir / name
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def turbo_dirs(self):
        return sorted(d.name for d in (self.studio.runtime / "outputs" / "webui").iterdir() if f"-{TURBO}-" in d.name)

    def waiting_for(self, jid, name):
        """Wait until the stand-in `name` (turbo-jobs.log, encodes.log) runs for the clip; returns its pid."""
        deadline = time.time() + 20
        while time.time() < deadline:
            pids = [r["pid"] for r in self.runs(name) if jid[:6] in json.dumps(r)]
            if pids and alive(pids[-1]):
                return pids[-1]
            time.sleep(0.05)
        self.fail(f"no {name} run for {jid}")

    def tearDown(self):
        for pid in [r["pid"] for name in ("turbo-jobs.log", "encodes.log") for r in self.runs(name)]:
            with contextlib.suppress(OSError):         # left running by a test that killed its server
                if alive(pid) and b"h3turbo_" in (Path("/proc") / str(pid) / "cmdline").read_bytes():
                    os.killpg(pid, signal.SIGKILL)
                    deadline = time.time() + 10
                    while alive(pid) and time.time() < deadline:
                        time.sleep(0.05)
        super().tearDown()


class TurboRegistry(TurboTest):
    """The models' table in /api/state ("models"), in the page (its lf-models script) and in the records."""

    def test_table(self):
        st = self.studio.state()
        sol, turbo = st["models"]
        self.assertEqual([sol["id"], sol["name"], sol["runs"], sol["default"], sol["experimental"]],
                         ["sol-h3", "Sol-H3", "session", True, False])
        self.assertEqual([turbo["id"], turbo["name"], turbo["runs"], turbo["default"], turbo["experimental"]],
                         [TURBO, "LightX2V Turbo", "job", False, True])
        offered = lambda caps: {dim: {v["value"]: v.get("why", True) for v in values}           # noqa: E731
                                for dim, values in caps.items()}
        self.assertEqual(offered(sol["capabilities"]), {
            "task": {"t2va": True, "fl2va": True, "ref2va": True},
            "frames": {"first": True, "last": True, "first+last": True},
            "seconds": {s: True for s in (5, 6, 7, 8, 10, 12, 15)},
            "orientation": {"landscape": True, "portrait": True}})
        self.assertEqual((sol["options"], sol["eta_s"]), ([], {}))
        self.assertEqual(offered(turbo["capabilities"]), {
            "task": {"t2va": True, "fl2va": True, "ref2va": NOT_REF2VA},
            "frames": {"first": True, "last": True, "first+last": True},
            "seconds": {s: True for s in (5, 6, 7, 8, 10, 12, 15)},
            "orientation": {"landscape": True, "portrait": True}})
        self.assertEqual(turbo["options"], [
            {"name": "resolution", "label": "Size", "type": "string", "default": "576p",
             "choices": [{"value": "576p", "size": {"landscape": [1024, 576], "portrait": [576, 1024]},
                          "offered": True},
                         {"value": "768p", "size": {"landscape": [1344, 768], "portrait": [768, 1344]},
                          "offered": True}]},
            {"name": "steps", "label": "Steps", "type": "integer", "default": 4,
             "choices": [{"value": 4, "offered": True}]}])
        self.assertEqual(turbo["combinations"], [
            {"task": ["t2va", "fl2va"], "frames": ["first", "last", "first+last"], "seconds": [5, 6, 7, 8, 10, 12, 15],
             "orientation": ["landscape", "portrait"], "resolution": ["576p", "768p"], "steps": [4]}])
        self.assertEqual(turbo["eta_s"], {key: {"s": s, "measured": False} for key, s in TURBO_ETA.items()})
        self.assertIsNone(st["session"]["model"])
        page = self.studio.get("/")[2].decode()
        table = re.search(r'<meta name="lf-studio-backend" content="spark">\n'
                          r'<script type="application/json" id="lf-models">(.*?)</script>', page).group(1)
        self.assertEqual(json.loads(table), st["models"])
        self.assertNotIn("<", table)

    def seed(self):
        super().seed()
        self.write_state([finished_job("a" * 32, created=1)])                   # a clip from before the model choice

    def test_records(self):
        """Every clip's record has its model (sol-h3 for one without, as all clips before the model choice) and the
        name it downloads as; an upscale has no model of its own."""
        st = self.studio.state()
        self.assertEqual((st["jobs"][0]["model"], st["jobs"][0]["file_name"]),
                         ("sol-h3", "lf-studio_t2va_5s_seed7_aaaaaa.mp4"))
        parent = self.clip()
        self.assertEqual(self.studio.state()["session"]["model"], "sol-h3")
        uid = self.done(self.upscale(parent["id"]))["id"]
        rec = json.loads(self.studio.get(f"/api/jobs/{uid}")[2])
        self.assertNotIn("model", rec)
        self.assertEqual(rec["file_name"], f"lf-studio_t2va_5s_seed1_{parent['id'][:6]}_2k_{uid[:6]}.mp4")
        older = json.loads(self.studio.get(f"/api/jobs?before={time.time() + 60}&limit=100")[2])["jobs"]
        self.assertEqual({j["id"]: j.get("model") for j in older},
                         {"a" * 32: "sol-h3", parent["id"]: "sol-h3", uid: None})
        self.assertTrue(all("file_name" in j for j in older))


class TurboRun(TurboTest):
    """A LightX2V Turbo clip: the Sol-H3 session closed, the prompt encoded, its job run, its files and its record."""

    def test_t2va_after_a_sol_clip(self):
        sol = self.clip()
        session_root = Path(sol["output"]).parents[2]
        jid = self.turbo(seed=9)
        job = self.done(jid)
        self.assertTrue((session_root / "CLOSED").exists())                   # Sol-H3's models were closed for it
        self.assertIn("closing session ('t2va', 'landscape', 5) - LightX2V Turbo needs the memory",
                      self.studio.log.read_text())
        self.assertEqual((job["model"], job["model_opts"]), (TURBO, {
            "resolution": "576p", "steps": 4, "size": [1024, 576], "lora": LORA4, "qwen_canvas": "draft",
            "image_tokens": 0}))
        case = Path(job["case_dirs"][0])
        self.assertEqual((case.name, case.parent.parent), (job["case_id"], self.studio.runtime / "outputs" / "webui"))
        self.assertRegex(case.parent.name, rf"^\d{{8}}-\d{{6}}-{TURBO}-576p-t2va-landscape-5s$")
        out = Path(job["output"])
        self.assertEqual(out, case / "turbo" / "clip_1024x576_121f.mp4")
        self.assertEqual(sorted(p.name for p in case.iterdir()),
                         ["encode.json", "encode.log", "prompt.txt", "qwen", "qwen-worker", "turbo"])  # no qwen-text
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()),
                         ["audio.pt", "clip_1024x576_121f.jpg", "clip_1024x576_121f.mp4", "job.json",
                          "latents.safetensors", "receipt.json", "worker.log"])
        streams = probe(out)
        self.assertEqual((streams["video"]["width"], streams["video"]["height"], streams["video"]["nb_frames"],
                          streams["video"]["r_frame_rate"]), (1024, 576, "121", "24/1"))
        self.assertEqual(streams["audio"]["codec_name"], "aac")
        self.assertEqual((case / "prompt.txt").read_text(), "a test clip")
        data = json.loads((case / "turbo" / "job.json").read_text())
        self.assertEqual(data, {
            "version": 1, "task": "t2va", "orientation": "landscape", "resolution": "576p", "seconds": 5, "seed": 9,
            "steps": 4, "lora": str(self.studio.h3turbo / "loras" / LORA4), "shift": [6, 3],
            "conditioning": str(case / "qwen" / "conditioning.pt"), "qwen_canvas": "draft", "first_frame": None,
            "last_frame": None, "audio_noise": "seed", "crf": 19, "x264_preset": "veryfast", "load": "fast",
            "abort_below_gib": 6, "paths": {"base_model": str(self.studio.h3turbo / "base"),
                                            "output": str(case / "turbo")}, "meta": {"studio_job": jid}})
        encode, = self.runs("encodes.log")
        self.assertEqual(encode["argv"], ["--case", str(case), "--prompt-file", str(case / "prompt.txt"),
                                          "--orientation", "landscape", "--resolution", "576p", "--canvas", "draft",
                                          "--seed", "9", "--case-id", job["case_id"], "--package",
                                          str(self.studio.package)])
        run, = self.runs("turbo-jobs.log")
        self.assertEqual(run["argv"], ["--job", str(case / "turbo" / "job.json"), "--parent",
                                       str(self.studio.proc.pid)])
        self.assertEqual(json.loads((case / "turbo" / "receipt.json").read_text())["state"], "done")
        self.assertEqual(set(job["phases_s"]), {"loading", "step", "decoding", "saving"})       # the job's receipt
        self.assertEqual((job["peak_gib"], job["low_mem_gib"], job["includes_model_load"]), (77.4, 35.2, True))
        self.assertGreater(job["load_s"], 0)
        self.assertGreaterEqual(job["e2e_s"], job["encode_s"])
        for key in ("job_phase", "text_only", "first_error", "saving", "cancel_requested"):
            self.assertNotIn(key, job)
        code, headers, _ = self.studio.get(f"/video/{jid}", {"Range": "bytes=0-9"})
        self.assertEqual((code, headers["Content-Disposition"]),
                         (206, f'inline; filename="lf-studio_{TURBO}-576p_t2va_5s_seed9_{jid[:6]}.mp4"'))
        self.assertEqual(job["file_name"], f"lf-studio_{TURBO}-576p_t2va_5s_seed9_{jid[:6]}.mp4")
        self.assertEqual(self.studio.get(f"/poster/{jid}")[1]["Content-Type"], "image/jpeg")
        times = json.loads((self.studio.state_dir / "load-times.json").read_text())
        self.assertEqual(times[f"job/{TURBO}/576p/4/5"], job["e2e_s"])
        self.assertEqual(times["last"], times["t2va/landscape/5"])            # Sol-H3's cold start, untouched
        self.assertEqual(self.studio.state()["models"][1]["eta_s"]["576p/4/5"], {"s": job["e2e_s"], "measured": True})
        self.assertFalse(alive(run["pid"]))
        self.assertEqual(worker_groups(self.studio.proc.pid), [])
        st = self.studio.state()
        self.assertEqual((st["session"]["state"], st["session"]["key"], st["session"]["model"]), ("idle", None, None))
        self.assertTrue(self.clip(seed=2)["includes_model_load"])              # Sol-H3 loads its models again

    def test_a_last_frame_alone_and_the_lengths(self):
        """A last frame alone at 6 s: its job keeps every frame it makes (158), so the clip ends on the frame the last
        keyframe is anchored to; a 12 s t2va clip keeps the studio's 289 of the 294 made, a 15 s one 361 of the 362
        made (its job raises diffusers' 15.0 s for it), also at 768p. Each length's time is remembered under its own
        key."""
        last = self.done(self.turbo(task="fl2va", seconds=6, files={"last_frame": [("b.png", png())]}))
        t2va = self.done(self.turbo(seconds=12, seed=2))
        long = self.done(self.turbo(seconds=15, resolution="768p", seed=3))
        for job, size, frames in ((last, "1024x576", 158), (t2va, "1024x576", 289), (long, "1344x768", 361)):
            self.assertEqual((job["status"], Path(job["output"]).name), ("done", f"clip_{size}_{frames}f.mp4"))
            self.assertEqual(probe(job["output"])["video"]["nb_frames"], str(frames))
        data = json.loads((Path(last["output"]).parent / "job.json").read_text())
        self.assertEqual((data["task"], data["seconds"], data["first_frame"]), ("fl2va", 6, None))
        self.assertEqual(data["last_frame"], last["last_frame"])
        self.assertRegex(Path(last["case_dirs"][0]).parent.name, rf"-{TURBO}-576p-fl2va-landscape-6s$")
        encode = next(r for r in self.runs("encodes.log") if last["case_id"] in r["argv"])
        self.assertEqual(encode["argv"][encode["argv"].index("--last") + 1], last["last_frame"])
        self.assertNotIn("--first", encode["argv"])
        times = json.loads((self.studio.state_dir / "load-times.json").read_text())
        self.assertEqual((times[f"job/{TURBO}/576p/4/6"], times[f"job/{TURBO}/576p/4/12"],
                          times[f"job/{TURBO}/768p/4/15"]), (last["e2e_s"], t2va["e2e_s"], long["e2e_s"]))

    def test_fl2va_first_and_last_portrait_768p(self):
        prompt = ("How the reference pictures align with the target video - Picture 1 aligns with the 0.00-second "
                  "mark; Picture 2 aligns with the 5.00-second mark.\nA dancer shown in <Picture 1> turns slowly.")
        jid = self.turbo(task="fl2va", orientation="portrait", resolution="768p", prompt=prompt,
                         files={"first_frame": [("a.png", png(24, 32))], "last_frame": [("b.png", png(24, 32))]})
        job = self.done(jid)
        self.assertEqual(job["model_opts"], {"resolution": "768p", "steps": 4, "size": [768, 1344], "lora": LORA4,
                                             "qwen_canvas": "draft", "image_tokens": 252})
        case = Path(job["case_dirs"][0])
        self.assertRegex(case.parent.name, rf"-{TURBO}-768p-fl2va-portrait-5s$")
        self.assertEqual(Path(job["output"]).name, "clip_768x1344_124f.mp4")     # every frame made: it ends on the last
        video = probe(job["output"])["video"]
        self.assertEqual((video["width"], video["height"], video["nb_frames"]), (768, 1344, "124"))
        data = json.loads((case / "turbo" / "job.json").read_text())
        self.assertEqual((data["first_frame"], data["last_frame"], data["resolution"], data["orientation"]),
                         (job["first_frame"], job["last_frame"], "768p", "portrait"))
        argv = self.runs("encodes.log")[0]["argv"]
        self.assertEqual(argv[argv.index("--resolution") + 1], "768p")
        self.assertEqual(argv[-4:], ["--first", job["first_frame"], "--last", job["last_frame"]])
        # the prompt without its keyframe references, for a later Upscale (the encode's own rewrite)
        self.assertEqual(job["text_only"], {"prompt": "A dancer turns slowly.",
                                            "dropped": ["alignment line", "1 x '<Picture N>'"]})
        self.assertEqual((case / "qwen-text" / "prompt.txt").read_text(), "A dancer turns slowly.")
        self.assertTrue((case / "qwen-text" / "conditioning.pt").is_file())
        self.assertEqual(job["labels"][0]["label"], "<Picture 1>")
        self.assertEqual(job["file_name"], f"lf-studio_{TURBO}-768p_fl2va_5s_seed1_{jid[:6]}.mp4")
        self.assertEqual(self.studio.get(f"/still/{jid}")[1]["Content-Type"], "image/jpeg")
        self.assertEqual(self.done(self.turbo(task="fl2va", seed=2, files={"first_frame": [("a.png", png())]}))[
            "model_opts"]["image_tokens"], 252)                                   # a first frame alone

    def test_fl2va_prompt_of_keyframe_lines_alone(self):
        """A prompt that is nothing but its keyframe line (the guide's first-frame line, the page's alignLine()) has no
        text-only prompt: the encode refuses that part before it starts anything, and the server encodes the clip
        again without it (--no-text-only), so the clip is made; its text_only is null (no later Upscale)."""
        guide = "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
        page = ("How the reference pictures align with the target video - Picture 1 aligns with the 0.00-second mark; "
                "Picture 2 aligns with the 5.00-second mark.")
        clips = ((guide, {"first_frame": [("a.png", png())]}),
                 (page, {"first_frame": [("a.png", png())], "last_frame": [("b.png", png())]}))
        for seed, (prompt, files) in enumerate(clips, 1):
            with self.subTest(prompt=prompt):
                jid = self.turbo(task="fl2va", prompt=prompt, files=files, seed=seed)
                job = self.done(jid)
                self.assertIsNone(job["text_only"])
                self.assertIsNone(json.loads(self.studio.get(f"/api/jobs/{jid}")[2])["text_only"])
                case = Path(job["case_dirs"][0])
                self.assertFalse((case / "qwen-text").exists())
                self.assertTrue((case / "qwen" / "conditioning.pt").is_file())
                first, again = [r["argv"] for r in self.runs("encodes.log") if job["case_id"] in r["argv"]]
                self.assertNotIn("--no-text-only", first)
                self.assertEqual(again, first + ["--no-text-only"])
                self.assertIn(f"{job['case_id']}: the prompt encode refused the text-only input (nothing is left of "
                              "the prompt without its keyframe references (--no-text-only skips the text-only "
                              "encode)): encoding the clip without it", self.studio.log.read_text())
        self.assertEqual(len(self.runs("turbo-jobs.log")), 2)

    def test_progress(self):
        """The status while it runs: freeing memory, encoding the prompt, loading (the job's own load), then
        generating from its first step, with its phase lines as job_phase."""
        self.restart(STUB_H3TURBO_S="6", STUB_QWEN_S="1", STUB_CLOSE_S="1")
        self.clip()
        jid = self.turbo()
        seen, deadline = [], time.time() + 40
        while time.time() < deadline:
            rec = json.loads(self.studio.get(f"/api/jobs/{jid}")[2])
            step = (rec["status"], rec["phase"], rec.get("job_phase"))
            if step not in seen:
                seen.append(step)
            if rec["state"] == "done":
                break
            time.sleep(0.03)
        self.assertIn(("freeing memory", "unloading", None), seen)
        self.assertIn(("encoding the prompt", "encoding", None), seen)
        self.assertIn(("loading models + generating", "loading", "loading"), seen)
        steps = [p for status, phase, p in seen if status == "generating"]
        self.assertEqual([p for i, p in enumerate(steps) if p and p not in steps[:i]],
                         ["step 1/4", "step 2/4", "step 3/4", "step 4/4", "decoding", "saving"])
        # the API's "saving" (Cancel answers 409) only once the job has written its clip, not after the encode
        self.assertGreater([phase for _, phase, p in seen].index("saving"), [p for _, _, p in seen].index("saving"))
        self.assertEqual(seen[-1], ("done", None, None))

    def test_canvas_setting(self):
        """LFSTUDIO_H3TURBO_QWEN_CANVAS=canvas: the encode sees the keyframes at the clip's canvas."""
        self.restart(LFSTUDIO_H3TURBO_QWEN_CANVAS="canvas")
        job = self.done(self.turbo(task="fl2va", files={"first_frame": [("a.png", png())]}))
        self.assertEqual((job["model_opts"]["qwen_canvas"], job["model_opts"]["image_tokens"]), ("canvas", 576))
        argv = self.runs("encodes.log")[0]["argv"]
        self.assertEqual(argv[argv.index("--canvas") + 1], "canvas")
        self.assertEqual(json.loads((Path(job["case_dirs"][0]) / "turbo" / "job.json").read_text())["qwen_canvas"],
                         "canvas")
        self.assertIn("Qwen canvas canvas, memory floor 100 GiB, guard 5 GiB", self.studio.log.read_text())


class TurboNames(TurboTest):
    """File names: Sol-H3's as before, byte for byte (the download and the publish copy); a LightX2V Turbo clip's
    with its model and options."""

    def seed(self):
        super().seed()
        self.publish = self.dir / "publish"
        self.studio.env["LFSTUDIO_PUBLISH_DIR"] = str(self.publish)

    def test_names(self):
        sol, turbo = self.clip(), self.done(self.turbo(resolution="768p", seed=3))
        for job, name, stem in ((sol, f"lf-studio_t2va_5s_seed1_{sol['id'][:6]}.mp4",
                                 rf"t2va_landscape_5s_seed1_{sol['id'][:6]}"),
                                (turbo, f"lf-studio_{TURBO}-768p_t2va_5s_seed3_{turbo['id'][:6]}.mp4",
                                 rf"t2va_landscape_5s_{TURBO}-768p_seed3_{turbo['id'][:6]}")):
            headers = self.studio.get(f"/video/{job['id']}", {"Range": "bytes=0-9"})[1]
            self.assertEqual(headers["Content-Disposition"], f'inline; filename="{name}"')
            self.assertEqual(job["file_name"], name)
            self.assertRegex(Path(job["published"]).name, rf"^\d{{8}}-\d{{6}}_{stem}\.mp4$")
            self.assertTrue(Path(job["published"]).with_suffix(".jpg").is_file())


class TurboRefusals(TurboTest):
    """What LightX2V Turbo does not make, a model that is unknown, an option of another model and a value an option
    does not have: 400 with the reason, the same from the form and JSON, nothing left behind."""

    def inputs(self):
        return sorted(p.name for p in (self.studio.runtime / "inputs" / "webui").iterdir())

    def test_refused(self):
        frame = lambda zone: {zone: [("f.png", png())]}                                         # noqa: E731
        b64 = lambda: base64.b64encode(png()).decode()                                           # noqa: E731
        cases = (   # (form fields, form files, JSON body, error)
            ({"model": "flux"}, None, {"model": "flux"}, "model must be one of sol-h3, lightx2v-turbo"),
            ({"model": TURBO, "task": "ref2va"}, frame("ref_images"),
             {"model": TURBO, "ref_images": [{"name": "r.png", "data": b64()}]}, NOT_REF2VA),
            ({"model": TURBO, "seconds": "9"}, None, {"model": TURBO, "seconds": 9},
             "duration must be 5, 6, 7, 8, 10, 12 or 15 seconds"),
            ({"model": TURBO, "resolution": "1080p"}, None, {"model": TURBO, "resolution": "1080p"},
             "resolution must be 576p or 768p for lightx2v-turbo"),
            ({"model": TURBO, "steps": "8"}, None, {"model": TURBO, "steps": 8}, "steps must be 4 for lightx2v-turbo"),
            ({"resolution": "576p"}, None, {"resolution": "576p"},
             "resolution is an option of lightx2v-turbo; sol-h3 takes none"),
            ({"model": "sol-h3", "steps": "4"}, None, {"model": "sol-h3", "steps": 4},
             "steps is an option of lightx2v-turbo; sol-h3 takes none"))
        for form, files, body, error in cases:
            by_form = self.studio.post("/api/generate", dict({"prompt": "x"}, **form), files)
            by_json = self.studio.post("/api/generate", json_body=dict({"prompt": "x"}, **body))
            self.assertEqual(by_form, (400, {"error": f"ValueError: {error}"}), form)
            self.assertEqual(by_json, by_form, body)
        self.assertEqual(self.studio.post("/api/generate", {"prompt": "x", "model": TURBO, "steps": "four"}),
                         (400, {"error": "ValueError: steps must be 4 for lightx2v-turbo"}))
        for body, error in (({"model": 1}, "model must be a string"), ({"model": TURBO, "resolution": 576},
                                                                       "resolution must be a string"),
                            ({"model": TURBO, "steps": "4"}, "steps must be an integer"),
                            ({"model": TURBO, "steps": True}, "steps must be an integer"),
                            ({"model": TURBO, "width": 1}, 'unknown key "width"')):
            code, answer = self.studio.post("/api/generate", json_body=dict({"prompt": "x"}, **body))
            self.assertEqual(code, 400, body)
            self.assertIn(error, answer["error"])
        self.assertIn("(known: prompt, task, orientation, seconds, seed, remix_from, keep, first_frame, last_frame, "
                      "ref_images, ref_videos, ref_audio, model, resolution, steps)", answer["error"])
        self.assertEqual((self.studio.state()["jobs"], self.inputs()), ([], []))
        # taken: no model, an empty one or null (sol-h3), and LightX2V Turbo's own clips with its options
        a = self.generate(model="")
        b = self.studio.post("/api/generate", json_body={"prompt": "x", "model": None, "resolution": None})[1]["id"]
        c = self.turbo(resolution="768p", steps="4")
        d = self.studio.post("/api/generate", json_body={"prompt": "x", "model": TURBO, "steps": 4,
                                                         "resolution": "576p"})[1]["id"]
        jobs = self.studio.wait([a, b, c, d], timeout=60)
        self.assertEqual([(j["status"], j["model"]) for j in jobs],
                         [("done", "sol-h3"), ("done", "sol-h3"), ("done", TURBO), ("done", TURBO)])

    def test_remix(self):
        """A Remix of a first + last clip as LightX2V Turbo keeps both frames, or the last one alone; a Remix of a
        Reference clip as LightX2V Turbo is refused."""
        src = self.clip(task="fl2va", files={"first_frame": [("a.png", png())], "last_frame": [("b.png", png())]})
        code, body = self.studio.turbo(task="fl2va", remix_from=src["id"], keep="first_frame:0,last_frame:0")
        self.assertEqual(code, 200, body)
        self.assertEqual(self.done(body["id"])["model"], TURBO)
        code, body = self.studio.turbo(task="fl2va", remix_from=src["id"], keep="last_frame:0")
        self.assertEqual(code, 200, body)
        last = self.done(body["id"])
        self.assertEqual((last["model"], last["status"], bool(last.get("last_frame")), last.get("first_frame")),
                         (TURBO, "done", True, None))
        self.assertEqual(Path(last["output"]).name, "clip_1024x576_124f.mp4")      # it ends on the last frame
        ref = self.clip(task="ref2va", files={"ref_images": [("r.png", png())]}, seed=4)
        self.assertEqual(self.studio.turbo(task="ref2va", remix_from=ref["id"], keep="ref_images:0"),
                         (400, {"error": f"ValueError: {NOT_REF2VA}"}))
        self.assertEqual(len(self.inputs()), 4)                 # the clips' and the Remixes'; none for the refused


class TurboPlainTokens(TurboTest):
    """With SOL_H3_PROMPT_TOKENS=plain, a LightX2V Turbo prompt with MiniMax's tokens is refused at once (its job
    refuses such a conditioning); Sol-H3 takes it as before."""
    env = {"SOL_H3_PROMPT_TOKENS": "plain"}

    def test_refused(self):
        prompt = "A man says <d>hello</d>."
        self.assertEqual(self.studio.turbo(prompt=prompt), (400, {"error": (
            "ValueError: lightx2v-turbo needs SOL_H3_PROMPT_TOKENS=minimax for a prompt with MiniMax's tokens (<d>, "
            "</d>): plain splits them into text tokens H3 never saw")}))
        self.generate(prompt=prompt)
        self.turbo(prompt="A man says hello.")


class TurboDialogue(TurboTest):
    """With SOL_H3_PROMPT_TOKENS unset (minimax, the default), a LightX2V Turbo prompt with MiniMax's dialogue tokens
    runs, and its encode gets it as written (TurboPlainTokens: refused with plain)."""

    def seed(self):
        super().seed()
        self.studio.env.pop("SOL_H3_PROMPT_TOKENS", None)

    def test_runs(self):
        prompt = "A man looks up and says, <d>[English] Hello there.</d>"
        job = self.done(self.turbo(prompt=prompt))
        case = Path(job["case_dirs"][0])
        self.assertEqual((case / "prompt.txt").read_text(), prompt)
        self.assertEqual(json.loads((case / "qwen" / "conditioning.json").read_text())["prompt"], prompt)
        encode, = self.runs("encodes.log")
        self.assertEqual(encode["argv"][encode["argv"].index("--prompt-file") + 1], str(case / "prompt.txt"))


class TurboQueue(TurboTest):
    """One queue for every clip, in order (strict FIFO across models), upscales after them; no warm-up after a LightX2V
    Turbo clip."""
    preload = True

    def test_fifo(self):
        parent = self.clip()
        a = self.generate(seed=2, seconds=6)                                    # runs while the others are queued
        self.until(a, lambda j: j["status"] != "queued")
        up = self.upscale(parent["id"])
        b, c, d = self.turbo(seed=3), self.generate(seed=4), self.turbo(seed=5)
        jobs = {j["id"]: j for j in self.studio.wait([a, up, b, c, d], timeout=90)}
        self.assertTrue(all(j["status"] == "done" for j in jobs.values()), jobs)
        order = sorted(jobs, key=lambda jid: jobs[jid]["started"])
        self.assertEqual(order, [a, b, c, d, up])
        self.assertTrue(jobs[c]["includes_model_load"])                         # a cold start after the turbo clip
        log = self.studio.log.read_text()
        self.assertEqual(log.count("LightX2V Turbo needs the memory"), 2)       # both Sol-H3 sessions closed
        time.sleep(3)                                                            # no warm-up after the turbo clip
        self.assertEqual(self.studio.state()["session"]["state"], "idle")
        self.assertNotIn("will preload", log.split("queued upscale")[-1].split("LightX2V Turbo:")[-1])


class TurboWarmUpAtStart(TurboTest):
    """None at start when the last finished clip is a LightX2V Turbo one."""
    preload = True

    def seed(self):
        super().seed()
        self.write_state([finished_job("1" * 32, created=1),
                          finished_job("2" * 32, created=2, model=TURBO,
                                       model_opts={"resolution": "576p", "steps": 4})])

    def test_none(self):
        time.sleep(2)
        self.assertNotIn("will preload", self.studio.log.read_text())
        self.assertEqual(self.studio.state()["session"]["state"], "idle")


class TurboStopsAWarmUp(TurboTest):
    """A LightX2V Turbo clip stops a warm-up (it closes the models anyway), unless a Sol-H3 clip queued before it
    waits for the warm-up's models."""
    preload, load_s = True, 4

    def seed(self):
        super().seed()
        self.write_state([finished_job("1" * 32, created=1)],
                         load_times={"warm/t2va/landscape/5": 1, "t2va/landscape/5": 6, "last": 6})   # no takeover

    def test_stopped(self):
        self.studio.wait_for(lambda s: s["session"].get("preloading"))
        self.done(self.turbo())
        self.assertIn("aborting preload: a LightX2V Turbo clip arrived, which needs the memory",
                      self.studio.log.read_text())

    def test_kept_for_a_sol_clip_before_it(self):
        self.studio.wait_for(lambda s: s["session"].get("preloading"))
        a = self.generate()                                                       # the warm-up's settings
        b = self.turbo(seed=2)
        jobs = self.studio.wait([a, b], timeout=60)
        self.assertEqual([j["status"] for j in jobs], ["done", "done"])
        self.assertNotIn("aborting preload", self.studio.log.read_text())
        self.assertNotIn("includes_model_load", jobs[0])                          # a warm clip in the warmed models


class TurboCancel(TurboTest):
    """Cancel takes a queued LightX2V Turbo clip out of the queue, or stops it while the Sol-H3 models close, while its
    prompt is encoded and while its job runs (SIGTERM to the process group of the step under way, SIGKILL 20 s later):
    Cancel answers 200 from the encode to the job's last step (one pipeline_call), the card stays, cancelled."""
    env = {"STUB_H3TURBO_S": "30"}

    def test_queued(self):
        a = self.generate()
        self.until(a, lambda j: j["status"] != "queued")
        jid = self.turbo()
        self.assertEqual(self.cancel(jid)[1]["job"]["status"], "cancelled")
        self.assertEqual(self.studio.wait([a])[0]["status"], "done")
        job = self.studio.job(jid)
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("case_dirs", job)
        self.assertEqual((self.turbo_dirs(), self.runs("encodes.log")), ([], []))
        self.assertNotIn("LightX2V Turbo needs the memory", self.studio.log.read_text())     # the models stayed

    def test_while_the_models_close(self):
        self.restart(STUB_CLOSE_S="3")
        self.clip()
        jid = self.turbo()
        self.until(jid, lambda j: j["status"] == "freeing memory")
        self.assertEqual(self.cancel(jid)[0], 200)
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertNotIn("case_dirs", job)
        self.assertEqual((self.turbo_dirs(), self.runs("encodes.log")), ([], []))

    def test_during_the_encode(self):
        self.restart(STUB_QWEN_S="30")
        jid = self.turbo()
        pid = self.waiting_for(jid, "encodes.log")
        self.until(jid, lambda j: j["status"] == "encoding the prompt")
        self.assertEqual(worker_groups(self.studio.proc.pid), [pid])             # in a process group of its own
        rec = json.loads(self.studio.get(f"/api/jobs/{jid}")[2])
        self.assertEqual((rec["state"], rec["phase"]), ("running", "encoding"))
        self.assertNotIn(rec["status"], idle_values())                            # busy for spark-switchboard
        t0 = time.time()
        self.assertEqual(self.cancel(jid)[0], 200)
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertLess(time.time() - t0, 5, "stopped right away")
        self.assertFalse(alive(pid))
        self.assertEqual((self.runs("turbo-jobs.log"), worker_groups(self.studio.proc.pid)), ([], []))
        case = Path(job["case_dirs"][0])
        self.assertTrue(case.is_dir())
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": jid}), (200, {"ok": True}))
        self.assertFalse(case.exists())

    def test_during_the_job(self):
        """After the encode, while the job loads and steps, the clip is not "saving": Cancel answers 200 (the encode
        and the job are one pipeline_call), and the job ends at once."""
        jid = self.turbo()
        pid = self.waiting_for(jid, "turbo-jobs.log")
        self.until(jid, lambda j: (j.get("job_phase") or "").startswith("step"))
        self.assertEqual(worker_groups(self.studio.proc.pid), [pid])
        rec = json.loads(self.studio.get(f"/api/jobs/{jid}")[2])
        self.assertEqual((rec["phase"], rec["status"]), ("generating", "generating"))
        self.assertEqual(self.studio.post("/api/unload"), (409, {"error": "a job is running"}))
        t0 = time.time()
        code, body = self.cancel(jid)
        self.assertEqual((code, body["job"]["cancel_requested"]), (200, True))
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertLess(time.time() - t0, 5)
        for key in ("error", "output", "published", "cancel_requested", "saving"):
            self.assertNotIn(key, job)
        self.assertTrue(job["job_phase"].startswith("step"), job["job_phase"])      # where it stopped (docs/API.md)
        self.assertFalse(alive(pid))
        receipt = json.loads((Path(job["case_dirs"][0]) / "turbo" / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("cancelled", 143))
        self.assertTrue(all(j["status"] in idle_values() for j in self.studio.state()["jobs"]))
        self.assertTrue(self.clip(seed=2)["includes_model_load"])               # the queue carries on

    def test_a_job_that_ignores_sigterm(self):
        self.restart(STUB_H3TURBO_TERM="ignore")
        jid = self.turbo()
        pid = self.waiting_for(jid, "turbo-jobs.log")
        self.until(jid, lambda j: (j.get("job_phase") or "").startswith("step"))
        t0 = time.time()
        self.assertEqual(self.cancel(jid)[0], 200)
        time.sleep(3)
        self.assertTrue(alive(pid))
        job = self.studio.wait([jid], timeout=40)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertGreater(time.time() - t0, 19)
        self.assertFalse(alive(pid))
        self.assertIn(f"cancel: the LightX2V Turbo job {pid} is still there 20 s after SIGTERM: SIGKILL",
                      self.studio.log.read_text())

    def test_an_encode_that_ignores_sigterm(self):
        """An encode that does not end on SIGTERM gets SIGKILL 20 s later, and the server stops the transient Qwen
        worker it could not close; one that ends on SIGTERM closes its worker itself."""
        self.restart(STUB_QWEN_S="60", STUB_QWEN_WORKER_S="90", STUB_QWEN_TERM="ignore")
        jid = self.turbo()
        pid = self.waiting_for(jid, "encodes.log")
        worker = qwen_worker(self, jid)
        self.assertEqual(self.cancel(jid)[0], 200)
        job = self.studio.wait([jid], timeout=40)[0]
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(alive(pid))
        gone_soon(self, worker)
        log = self.studio.log.read_text()
        self.assertIn(f"cancel: the LightX2V Turbo prompt encode {pid} is still there 20 s after SIGTERM: SIGKILL", log)
        self.assertIn(f"stopping the Qwen worker the prompt encode left (process group {worker})", log)
        self.restart(STUB_QWEN_TERM="")
        jid = self.turbo(seed=2)
        self.waiting_for(jid, "encodes.log")
        worker = qwen_worker(self, jid)
        self.assertEqual(self.cancel(jid)[0], 200)
        self.assertEqual(self.studio.wait([jid], timeout=30)[0]["status"], "cancelled")
        self.assertFalse(alive(worker))
        self.assertEqual(self.studio.log.read_text().count("stopping the Qwen worker the prompt encode left"), 1)

    def test_restart_while_running(self):
        """A server killed hard takes the job with it (--parent); the clip reads failed after the restart."""
        jid = self.turbo()
        pid = self.waiting_for(jid, "turbo-jobs.log")
        self.studio.proc.kill()
        self.studio.proc.wait()
        deadline = time.time() + 10
        while alive(pid) and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid), "the job outlived the server")
        self.studio.start()
        job = self.studio.job(jid)
        self.assertEqual((job["status"], job["error"]), ("failed", "server restarted"))

    def test_stop_ends_the_job_and_the_encode(self):
        jid = self.turbo()
        pid = self.waiting_for(jid, "turbo-jobs.log")
        self.studio.stop()                                                        # SIGTERM: the server stops its job
        self.assertFalse(alive(pid))
        self.assertIn("stopping the running LightX2V Turbo clip", self.studio.log.read_text())
        self.restart(STUB_QWEN_S="30")
        self.assertEqual(self.studio.job(jid)["status"], "failed")
        jid = self.turbo(seed=2)
        pid = self.waiting_for(jid, "encodes.log")
        self.studio.stop()
        self.assertFalse(alive(pid))
        self.assertIn("stopping the running LightX2V Turbo prompt encode", self.studio.log.read_text())


class TurboDelete(TurboTest):
    """Delete and Undo of a LightX2V Turbo clip move its own directory and inputs; a failed one's directory goes too."""

    def test_delete_and_undo(self):
        jid = self.turbo(task="fl2va", files={"first_frame": [("a.png", png())]})
        job = self.done(jid)
        case, inputs = Path(job["case_dirs"][0]), self.studio.runtime / "inputs" / "webui" / jid
        self.assertEqual(Path(job["output"]).parents[1], case)
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": jid}), (200, {"ok": True}))
        self.assertFalse(case.exists() or inputs.exists())
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": jid})[0], 200)
        self.assertTrue(Path(job["output"]).is_file() and inputs.is_dir())
        self.restart(STUB_H3TURBO_FAIL="1")
        failed = self.studio.wait([self.turbo(seed=2)], timeout=30)[0]
        self.assertEqual(failed["status"], "failed")
        case = Path(failed["case_dirs"][0])
        self.assertTrue(case.is_dir())
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": failed["id"]}), (200, {"ok": True}))
        self.assertFalse(case.exists())


class TurboMemory(TurboTest):
    """The memory floor before a clip (LFSTUDIO_H3TURBO_MIN_FREE_GIB, default 100), the guard while its job runs
    (LFSTUDIO_H3TURBO_GUARD_GIB, default 5) and the job's own stop (exit 3), on the stub studio's MemAvailable file."""
    env = {"STUB_H3TURBO_S": "30"}

    def test_floor(self):
        self.studio.set_memory(50)
        job = self.studio.wait([self.turbo()], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: not enough free memory for a LightX2V Turbo clip: 50.0 GiB available with the models "
            "closed, 100 GiB needed (LFSTUDIO_H3TURBO_MIN_FREE_GIB); close other apps")))
        self.assertNotIn("case_dirs", job)
        self.assertEqual((self.turbo_dirs(), self.runs("encodes.log")), ([], []))
        self.assertNotIn("retrying", self.studio.log.read_text())
        self.studio.set_memory(101)
        self.assertTrue(self.waiting_for(self.turbo(seed=2), "turbo-jobs.log"))

    def test_floor_per_size_and_length(self):
        """The setting is a minimum: a clip needs at least what its job's min_free_for says one of its size and length
        takes (90 GiB at 576p 5 s, 100 at 768p 10 s; above the default 100, 103 at 768p 12 s and 107 at 768p 15 s), so
        a lower setting does not lower that; at or above it, the setting is the floor."""
        for seconds, available, needed in ((15, 105, 107), (12, 102, 103)):
            self.studio.set_memory(available)
            job = self.studio.wait([self.turbo(resolution="768p", seconds=seconds, seed=seconds)], timeout=30)[0]
            self.assertEqual((job["status"], job["error"]), ("failed", (
                f"ModelJobFailed: not enough free memory for a LightX2V Turbo clip: {available:.1f} GiB available "
                f"with the models closed, {needed} GiB needed (what a 768p {seconds} s clip needs, more than "
                "LFSTUDIO_H3TURBO_MIN_FREE_GIB=100); close other apps")))
        self.restart(LFSTUDIO_H3TURBO_MIN_FREE_GIB="80")
        self.studio.set_memory(85)
        job = self.studio.wait([self.turbo()], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: not enough free memory for a LightX2V Turbo clip: 85.0 GiB available with the models "
            "closed, 90 GiB needed (what a 576p 5 s clip needs, more than LFSTUDIO_H3TURBO_MIN_FREE_GIB=80); close "
            "other apps")))
        self.studio.set_memory(95)
        job = self.studio.wait([self.turbo(resolution="768p", seconds=10, seed=2)], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: not enough free memory for a LightX2V Turbo clip: 95.0 GiB available with the models "
            "closed, 100 GiB needed (what a 768p 10 s clip needs, more than LFSTUDIO_H3TURBO_MIN_FREE_GIB=80); "
            "close other apps")))
        self.assertEqual(self.runs("encodes.log"), [])
        self.assertTrue(self.waiting_for(self.turbo(seed=3), "turbo-jobs.log"))            # 576p 5 s from 95 GiB

    def test_guard(self):
        jid = self.turbo()
        pid = self.waiting_for(jid, "turbo-jobs.log")
        self.studio.set_memory(1)
        t0 = time.time()
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: stopped: MemAvailable was below 5 GiB for 3 s (1.0 GiB; LFSTUDIO_H3TURBO_GUARD_GIB)")))
        self.assertLess(time.time() - t0, 8)
        self.assertFalse(alive(pid))
        self.assertIn("LightX2V Turbo: MemAvailable 1.0 GiB, below 5 GiB for 3 s: stopping its job (SIGKILL)",
                      self.studio.log.read_text())

    def test_guard_during_the_encode(self):
        """The guard stops the prompt encode (SIGKILL to its group), and the server stops the transient Qwen worker it
        leaves (a group of its own, a grandchild of the server), which would otherwise hold its memory until it next
        reads its stdin."""
        self.restart(STUB_QWEN_S="30", STUB_QWEN_WORKER_S="60")
        jid = self.turbo()
        self.waiting_for(jid, "encodes.log")
        worker = qwen_worker(self, jid)
        self.studio.set_memory(1)
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: stopped: MemAvailable was below 5 GiB for 3 s (1.0 GiB; LFSTUDIO_H3TURBO_GUARD_GIB)")))
        gone_soon(self, worker)
        self.assertIn(f"stopping the Qwen worker the prompt encode left (process group {worker})",
                      self.studio.log.read_text())
        self.assertEqual(self.runs("turbo-jobs.log"), [])

    def test_the_jobs_own_stop(self):
        self.restart(STUB_H3TURBO_ABORT="1")
        job = self.studio.wait([self.turbo()], timeout=30)[0]
        self.assertEqual((job["status"], job["error"]), ("failed", (
            "ModelJobFailed: stopped by its job: MemAvailable fell below 6 GiB (MemAvailable 5.2 GiB, below "
            "abort_below_gib 6 (STUB_H3TURBO_ABORT))")))

    def test_checks_off(self):
        self.restart(LFSTUDIO_H3TURBO_MIN_FREE_GIB="0", LFSTUDIO_H3TURBO_GUARD_GIB="0", STUB_H3TURBO_S="4")
        self.studio.set_memory(1)
        self.assertEqual(self.done(self.turbo())["status"], "done")


class TurboFailure(TurboTest):
    """A step that fails fails the clip with its own error, without a retry; the next Sol-H3 clip loads as usual."""

    def failed(self, **env):
        self.restart(**env)
        job = self.studio.wait([self.turbo()], timeout=30)[0]
        self.assertEqual(job["status"], "failed")
        self.assertNotIn("first_error", job)
        return job

    def test_the_job(self):
        job = self.failed(STUB_H3TURBO_FAIL="1")
        self.assertEqual(job["error"], "ModelJobFailed: OutOfMemoryError: CUDA out of memory (STUB_H3TURBO_FAIL)")
        self.assertNotIn("retrying", self.studio.log.read_text())
        self.assertTrue(self.clip(seed=2)["includes_model_load"])

    def test_the_encode(self):
        job = self.failed(STUB_QWEN_FAIL="1")
        self.assertEqual(job["error"], "ModelJobFailed: the prompt encode failed: RuntimeError: Worker exited with "
                                       "code 1 (STUB_QWEN_FAIL)")
        self.assertEqual(self.runs("turbo-jobs.log"), [])

    def test_outputs(self):
        log = lambda job: Path(job["case_dirs"][0]) / "turbo" / "worker.log"                     # noqa: E731
        job = self.failed(STUB_H3TURBO_OUTPUT="none")
        self.assertEqual(job["error"], f"ModelJobFailed: the LightX2V Turbo job wrote no clip_1024x576_121f.mp4; see "
                                       f"{log(job)}")
        job = self.failed(STUB_H3TURBO_OUTPUT="bad")
        self.assertEqual(job["error"], f"ModelJobFailed: the LightX2V Turbo job's clip_1024x576_121f.mp4 can't be "
                                       f"read; see {log(job)}")

    def test_no_python_and_no_lora(self):
        job = self.failed(LFSTUDIO_H3TURBO_PYTHON=str(self.dir / "nothing"))
        self.assertTrue(job["error"].startswith(f"ModelJobFailed: cannot start the LightX2V Turbo job with "
                                                f"LFSTUDIO_H3TURBO_PYTHON={self.dir / 'nothing'} ("), job["error"])
        lora = self.studio.h3turbo / "loras" / LORA4
        lora.unlink()
        job = self.failed(LFSTUDIO_H3TURBO_PYTHON=sys.executable)
        self.assertEqual(job["error"], f"ModelJobFailed: no lora file {lora}")           # the job's refusal


class TurboWithoutUpscale(TurboTest):
    """With the Upscale off (its job not loaded) a LightX2V Turbo clip runs, its memory floor and guard work."""
    env = {"LFSTUDIO_UPSCALE": "0"}

    def test_runs(self):
        self.assertIs(self.studio.state()["upscale"], False)
        job = self.done(self.turbo())
        self.assertNotIn("upscalable", job)
        self.studio.set_memory(50)
        self.assertIn("50.0 GiB available", self.studio.wait([self.turbo(seed=2)], timeout=30)[0]["error"])


class TurboUpscale(TurboTest):
    """The Upscale takes every 5 s LightX2V Turbo clip, Text and Keyframes, at either size, 16:9 or 9:16: its input
    folder links the clip's 768p MP4 and stretches a 576p one to 768p (LANCZOS; 121 frames), cuts a clip with a last
    keyframe (124 frames) to its first 121, links its audio.pt as the job's audio, and its qwen/ (Text) or qwen-text/
    (Keyframes: made with the clip) as its text; a Keyframes clip without qwen-text/ gets the upscale's encode. The
    result has the clip's sound, cut to the frames kept."""
    env = {"STUB_UPSCALE_S": "1"}

    def view(self, job):
        return Path(job["case_dirs"][0]) / "input"

    def input_mp4(self, job):
        mp4, = (self.view(job) / "stage2").glob("refined_*.mp4")
        return mp4

    def test_text_576p_stretched(self):
        parent = self.done(self.turbo(seed=4))
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "video": "stretch"})
        job = self.done(self.upscale(parent["id"], resolution="1080p"))
        self.assertEqual((job["parent_model"], job["parent_opts"]["resolution"], job["video"], job["text"]),
                         (TURBO, "576p", "stretch", {"source": "clip"}))
        mp4 = self.input_mp4(job)
        self.assertEqual(mp4.name, "refined_1344x768_121f.mp4")
        streams = probe(mp4)
        self.assertEqual((streams["video"]["width"], streams["video"]["height"], streams["video"]["nb_frames"],
                          streams["video"]["r_frame_rate"]), (1344, 768, "121", "24/1"))
        parent_case = Path(parent["output"]).parents[1]
        view = self.view(job)
        self.assertEqual(inode(view / "stage1" / "stage1_direct_tensors.pt"), inode(parent_case / "turbo" / "audio.pt"))
        self.assertEqual(inode(view / "qwen" / "conditioning.pt"), inode(parent_case / "qwen" / "conditioning.pt"))
        source = json.loads((view / "source.json").read_text())
        self.assertEqual((source["model"], source["options"]["resolution"], source["video"]),
                         (TURBO, "576p", "stretch"))
        self.assertEqual(source["files"]["stage2/refined_1344x768_121f.mp4"]["how"], "stretch")
        self.assertEqual(source["files"]["stage1/stage1_direct_tensors.pt"],
                         {"from": str(parent_case / "turbo" / "audio.pt"), "how": "link",
                          "sha256": hashlib.sha256((parent_case / "turbo" / "audio.pt").read_bytes()).hexdigest()})
        self.assertEqual(self.runs("encodes.log")[1:], [])                      # the clip's own encode only
        out = probe(job["output"])
        self.assertEqual(Path(job["output"]).name, "regen_1920x1080_121f.mp4")
        for key in ("codec_name", "sample_rate", "channels", "nb_frames", "duration"):   # its sound as it is: only
            self.assertEqual(out["audio"][key], probe(parent["output"])["audio"][key], key)  # stretched, not cut
        self.assertEqual(json.loads((Path(job["case_dirs"][0]) / "regen" / "job.json").read_text())["case"], str(view))

    def test_text_768p_portrait_linked(self):
        parent = self.done(self.turbo(orientation="portrait", resolution="768p"))
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True})
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual(inode(self.input_mp4(job)), inode(parent["output"]))
        self.assertEqual(self.input_mp4(job).name, "refined_768x1344_121f.mp4")      # named as the job finds it
        self.assertEqual(Path(job["output"]).name, "regen_1440x2560_121f.mp4")
        self.assertEqual(json.loads((self.view(job) / "source.json").read_text())["files"][
            "stage2/refined_768x1344_121f.mp4"]["how"], "link")
        for key in ("nb_frames", "duration"):                                    # its sound, copied as it is
            self.assertEqual(probe(job["output"])["audio"][key], probe(parent["output"])["audio"][key])

    def test_keyframes_with_the_text_made_with_the_clip(self):
        """A first frame at 576p: qwen-text/ is its text (no encode), stretched as a Text clip."""
        prompt = ("For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
                  "\nA kite.")
        parent = self.done(self.turbo(task="fl2va", prompt=prompt, files={"first_frame": [("a.png", png())]}))
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "text": "generation", "video": "stretch"})
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual(job["text"], {"source": "generation", "prompt": parent["text_only"]["prompt"],
                                       "dropped": parent["text_only"]["dropped"]})
        parent_case = Path(parent["output"]).parents[1]
        self.assertEqual(inode(self.view(job) / "qwen" / "conditioning.pt"),
                         inode(parent_case / "qwen-text" / "conditioning.pt"))
        self.assertEqual(json.loads((self.view(job) / "qwen" / "conditioning.json").read_text())["task"], "t2va")
        self.assertEqual(len(self.runs("encodes.log")), 1)

    def test_a_last_keyframe_cut_to_121_frames(self):
        """124 frames at 5 s, its last keyframe on the last: the input is its first 121 (768p: cut; 576p: stretched
        and cut), its sound cut with it, and the upscale has 121 frames."""
        line = ("How the reference pictures align with the target video - Picture 1 aligns with the 0.00-second mark; "
                "Picture 2 aligns with the 5.00-second mark.\nA dancer turns slowly.")
        both = {"first_frame": [("a.png", png())], "last_frame": [("b.png", png())]}
        for resolution, video in (("768p", "trim"), ("576p", "stretch+trim")):
            with self.subTest(resolution):
                parent = self.done(self.turbo(task="fl2va", prompt=line, resolution=resolution, files=both,
                                              seed=7 if resolution == "768p" else 8))
                self.assertEqual(probe(parent["output"])["video"]["nb_frames"], "124")
                self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                                 {"ok": True, "clip_audio": True, "text": "generation", "video": video})
                job = self.done(self.upscale(parent["id"]))
                self.assertEqual(job["video"], video)
                mp4 = probe(self.input_mp4(job))
                self.assertEqual((mp4["video"]["width"], mp4["video"]["height"], mp4["video"]["nb_frames"]),
                                 (1344, 768, "121"))
                self.assertLess(float(mp4["audio"]["duration"]), 5.1)
                self.assertGreater(float(probe(parent["output"])["audio"]["duration"]), 5.15)
                out = probe(job["output"])
                self.assertEqual(out["video"]["nb_frames"], "121")
                self.assertEqual(out["audio"]["duration"], mp4["audio"]["duration"])     # the cut sound, copied
                self.assertEqual(json.loads((self.view(job) / "source.json").read_text())["files"][
                    "stage2/refined_1344x768_121f.mp4"]["how"], video)

    def test_keyframes_without_qwen_text_get_the_upscales_encode(self):
        prompt = ("For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
                  "\nA kite.")
        parent = self.done(self.turbo(task="fl2va", prompt=prompt, resolution="768p",
                                      files={"first_frame": [("a.png", png())]}))
        shutil.rmtree(Path(parent["output"]).parents[1] / "qwen-text")          # an older clip, say
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "text": "encode"})
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual((job["text"]["source"], job["text"]["prompt"]), ("encode", "A kite."))
        run = self.runs("encodes.log")[-1]
        self.assertEqual(run["argv"][-1], "--upscale")
        self.assertEqual(run["argv"][run["argv"].index("--case") + 1], str(self.view(job)))

    def test_nothing_left_of_the_prompt(self):
        """A prompt of its keyframe line alone: the clip is made (without qwen-text/), its Upscale refused."""
        line = "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
        parent = self.done(self.turbo(task="fl2va", prompt=line, files={"first_frame": [("a.png", png())]}))
        self.assertIsNone(parent["text_only"])
        self.assertEqual(self.studio.job(parent["id"])["upscalable"], {"ok": False, "why": NOTHING_LEFT})
        self.assertEqual(self.studio.upscale(parent["id"]), (400, {"error": NOTHING_LEFT}))

    def test_other_lengths_not_yet_and_the_audio(self):
        parent = self.done(self.turbo(seconds=6))
        self.assertEqual(self.studio.upscale(parent["id"]), (400, {"error": UPSCALE_LENGTH}))
        parent = self.done(self.turbo(seed=3))
        (Path(parent["output"]).parent / "audio.pt").unlink()
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": False, "video": "stretch"})
        self.assertEqual(self.studio.upscale(parent["id"]),
                         (409, {"error": "the clip has no audio (turbo/audio.pt); ask for audio silence"}))
        job = self.done(self.upscale(parent["id"], audio="silence"))
        self.assertFalse((self.view(job) / "stage1").exists())

    def test_with_the_model_turned_off(self):
        """LightX2V Turbo off (LFSTUDIO_MODELS unset): a clip it made earlier stays upscalable; the Upscale needs
        nothing of the model's own."""
        parent = self.done(self.turbo())
        self.restart(LFSTUDIO_MODELS="")
        self.assertNotIn("models", self.studio.state())
        self.assertEqual(self.studio.job(parent["id"])["upscalable"],
                         {"ok": True, "clip_audio": True, "video": "stretch"})
        job = self.done(self.upscale(parent["id"]))
        self.assertEqual((job["parent_model"], job["video"]), (TURBO, "stretch"))


class TurboLoad(TurboTest):
    """POST /api/load takes a model: sol-h3 (the default) as before; LightX2V Turbo loads its model with each clip."""

    def test_load(self):
        want = {"task": "t2va", "orientation": "landscape", "seconds": 5}
        self.assertEqual(self.studio.post("/api/load", json_body=dict(want, model=TURBO)), (409, {
            "error": "LightX2V Turbo loads its model with each clip; there is nothing to load ahead"}))
        self.assertEqual(self.studio.post("/api/load", json_body=dict(want, model="flux")),
                         (400, {"error": "model must be one of sol-h3, lightx2v-turbo"}))
        self.assertEqual(self.studio.post("/api/load", json_body=dict(want, model="sol-h3")), (200, {"ok": True}))
        self.assertEqual(self.studio.wait_for(lambda s: s["session"]["state"] == "ready")["session"]["model"],
                         "sol-h3")


class TurboSettings(unittest.TestCase):
    """LFSTUDIO_MODELS and, with lightx2v-turbo, LightX2V Turbo's settings: a bad value stops the server at start."""

    def test_refused_at_start(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        self.addCleanup(shutil.rmtree, d, True)
        for i, (env, message) in enumerate((
                ({"LFSTUDIO_MODELS": "sol-h3,flux"}, "LFSTUDIO_MODELS=sol-h3,flux: unknown model flux (known: sol-h3, "
                                                     "lightx2v-turbo) (see bin/env.example)"),
                ({"LFSTUDIO_MODELS": "lightx2v-turbo"}, "LFSTUDIO_MODELS=lightx2v-turbo: must include sol-h3, the "
                                                        "studio's own model (see bin/env.example)"),
                ({"LFSTUDIO_MODELS": BOTH, "LFSTUDIO_H3TURBO_MIN_FREE_GIB": "-1"},
                 "LFSTUDIO_H3TURBO_MIN_FREE_GIB=-1: must be a number of GiB, 0 or more (see bin/env.example)"),
                ({"LFSTUDIO_MODELS": BOTH, "LFSTUDIO_H3TURBO_QWEN_CANVAS": "full", "LFSTUDIO_H3TURBO_BASE": "rel"},
                 "LFSTUDIO_H3TURBO_BASE=rel: must be an absolute path; LFSTUDIO_H3TURBO_QWEN_CANVAS=full: must be "
                 "draft or canvas (see bin/env.example)"),
                ({"LFSTUDIO_MODELS": BOTH, "LFSTUDIO_H3TURBO_SCRIPT": str(d / "no-job.py")},
                 f"LFSTUDIO_MODELS has lightx2v-turbo: cannot read its job {d / 'no-job.py'}"),
                ({"LFSTUDIO_MODELS": BOTH, "LFSTUDIO_H3TURBO_ENCODE_SCRIPT": str(d / "no-encode.py")},
                 f"LFSTUDIO_MODELS has lightx2v-turbo: no prompt encode at {d / 'no-encode.py'}"))):
            with self.assertRaises(RuntimeError) as e:
                StubStudio(d / str(i), env=env).start(timeout=10)
            self.assertIn(message, str(e.exception))
        for i, value in enumerate(("", " sol-h3 ", "sol-h3,,", " lightx2v-turbo , sol-h3")):
            studio = StubStudio(d / f"ok{i}", env={"LFSTUDIO_MODELS": value}).start()
            try:
                self.assertEqual("models" in studio.state(), "lightx2v-turbo" in value, value)
            finally:
                studio.stop()


class LauncherModels(LauncherSettingsCheck):
    """bin/lf-studio --check with LFSTUDIO_MODELS (in the server's words), and with lightx2v-turbo its settings, its
    Python, its prompt encode and its job's check_install (here the stand-ins'); --kill ends a left-over job or
    encode."""
    test_settings = test_api_setting = test_older_patch = None

    def turbo(self):
        root = self.d / "h3turbo"
        (root / "base").mkdir(parents=True)
        (root / "loras").mkdir()
        (root / "loras" / LORA4).write_bytes(b"")
        stub = Path(__file__).resolve().parent / "stub"
        return {"LFSTUDIO_MODELS": BOTH, "LFSTUDIO_H3TURBO_SCRIPT": str(stub / "h3turbo_job.py"),
                "LFSTUDIO_H3TURBO_ENCODE_SCRIPT": str(stub / "h3turbo_encode.py"),
                "LFSTUDIO_H3TURBO_PYTHON": sys.executable, "LFSTUDIO_H3TURBO_BASE": str(root / "base"),
                "LFSTUDIO_H3TURBO_LORAS": str(root / "loras")}

    def refused(self, why, **settings):
        code, out = self.check(**settings)
        self.assertEqual(code, 1, out)
        self.assertIn(f"REFUSED - {why}", out)

    def test_the_setting(self):
        for value in ("", "sol-h3", " sol-h3 ,"):
            code, out = self.check(LFSTUDIO_MODELS=value, LFSTUDIO_H3TURBO_MIN_FREE_GIB="not read")
            self.assertEqual(code, 0, out)
            self.assertNotIn("LightX2V", out)
        self.refused("LFSTUDIO_MODELS=sol-h3,flux: unknown model flux (known: sol-h3, lightx2v-turbo) (see "
                     "bin/env.example)", LFSTUDIO_MODELS="sol-h3,flux")
        self.refused("LFSTUDIO_MODELS=lightx2v-turbo: must include sol-h3, the studio's own model (see "
                     "bin/env.example)", LFSTUDIO_MODELS="lightx2v-turbo")

    def test_turbo(self):
        settings = self.turbo()
        code, out = self.check(**settings)
        self.assertEqual(code, 0, out)
        self.assertIn("the experimental LightX2V Turbo is on (LFSTUDIO_MODELS): its environment, base model and LoRA "
                      "are there", out)
        turbo = "the experimental LightX2V Turbo (LFSTUDIO_MODELS): "
        self.refused("LFSTUDIO_H3TURBO_GUARD_GIB=x: must be a number of GiB, 0 or more (see bin/env.example)",
                     **settings, LFSTUDIO_H3TURBO_GUARD_GIB="x")
        self.refused(f"{turbo}no Python at LFSTUDIO_H3TURBO_PYTHON={self.d / 'nothing'} (the environment of its job); "
                     "set it up, or set LFSTUDIO_MODELS=sol-h3",
                     **dict(settings, LFSTUDIO_H3TURBO_PYTHON=str(self.d / "nothing")))
        self.refused(f"{turbo}no prompt encode at {self.d / 'no-encode.py'}",
                     **dict(settings, LFSTUDIO_H3TURBO_ENCODE_SCRIPT=str(self.d / "no-encode.py")))
        self.refused(f"LFSTUDIO_MODELS has lightx2v-turbo: cannot read its job {self.d / 'no-job.py'} (no such file)",
                     **dict(settings, LFSTUDIO_H3TURBO_SCRIPT=str(self.d / "no-job.py")))
        lora = self.d / "h3turbo" / "loras" / LORA4
        lora.unlink()
        self.refused(f"{turbo}no {lora} (the 4-step LoRA; install/66-h3turbo.sh fetches it); set it up", **settings)

    def test_kill_ends_left_over_jobs(self):
        """--kill (and the launcher's cleanup) find LightX2V Turbo's job and prompt encode by their scripts' names and
        command lines (jobs/h3turbo_job.py --job, bench/h3turbo_encode.py --case under this runtime root's web-UI
        outputs), not one elsewhere, nor another script of bench/ or jobs/ on a studio case."""
        repo, runtime = Path(__file__).resolve().parents[1], self.d / "runtime"
        case = runtime / "outputs" / "webui" / f"20261028-000000-{TURBO}-576p-t2va-landscape-5s" / "web_x"
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        procs = [subprocess.Popen(sleeper + [str(repo / rel), flag, str(path)], start_new_session=True)
                 for rel, flag, path in (("jobs/h3turbo_job.py", "--job", case / "turbo" / "job.json"),
                                         ("bench/h3turbo_encode.py", "--case", case),
                                         ("jobs/h3turbo_job.py", "--job", self.d / "bench" / "job.json"),
                                         ("bench/research_probe.py", "--case", case),
                                         ("jobs/other_job.py", "--job", case / "turbo" / "job.json"),
                                         ("bench/h3turbo_encode.py", "--job", case))]
        self.addCleanup(lambda: [p.kill() or p.wait() for p in procs if p.poll() is None])
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_"))}
        env.update(LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_SANA=str(self.d / "no-sana"),
                   LFSTUDIO_PYTHON=sys.executable, SOL_H3_SPARK_RUNTIME_ROOT=str(runtime),
                   LFSTUDIO_STATE_DIR=str(self.d / "state"), LFSTUDIO_SERVER=str(self.d / "no-server.py"),
                   LFSTUDIO_TEST_SKIP_DOCKER="1")
        run = subprocess.run(["bash", str(repo / "bin/lf-studio"), "--kill"], env=env, capture_output=True, text=True,
                             timeout=60)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual([procs[0].wait(timeout=10), procs[1].wait(timeout=10)], [-signal.SIGKILL] * 2)
        self.assertEqual([p.poll() for p in procs[2:]], [None] * 4)


class TurboSeam(unittest.TestCase):
    """Where LightX2V Turbo's job plugs in: the job the server loads (LFSTUDIO_H3TURBO_SCRIPT, default
    jobs/h3turbo_job.py) has the names and tables the server uses, the registry's model id and the sizes of its
    canvases, takes every clip the registry offers, and gives the registry's estimates (its estimate_s); the stand-in
    the stub studio runs has the same names and gives the same values (its tables, settings, lengths, the frames it
    keeps and the memory floor)."""
    NAMES = ("MODEL", "JOB_VERSION", "PRESETS", "RECEIPT", "MINIMAX_PROMPT_TOKENS", "CANVAS", "studio_settings",
             "min_free_for", "parse_job", "output_name", "worker_command", "worker_environment", "parse_line",
             "write_json", "check_install")
    REAL = REPO_DIR / "jobs" / "h3turbo_job.py"
    STUB = Path(__file__).resolve().parent / "stub" / "h3turbo_job.py"

    @staticmethod
    def load(path):
        spec = importlib.util.spec_from_file_location(f"h3turbo_job_under_test_{path.parent.name}", path)
        job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job)
        return job

    def check(self, path):
        job = self.load(path)
        for name in self.NAMES:
            self.assertTrue(hasattr(job, name), f"{path} has no {name}")
        self.assertEqual((job.MODEL, job.JOB_VERSION), (TURBO, 1))
        self.assertIn(4, job.PRESETS)
        self.assertEqual({k: list(v) for k, v in job.CANVAS.items()},
                         {("576p", "landscape"): [1024, 576], ("576p", "portrait"): [576, 1024],
                          ("768p", "landscape"): [1344, 768], ("768p", "portrait"): [768, 1344]})
        settings, problems = job.studio_settings({"LFSTUDIO_HOME": "/opt/lf"})
        self.assertEqual(problems, [])
        self.assertEqual({k: settings[k] for k in ("python", "base_model", "loras", "qwen_canvas", "min_free_gib",
                                                   "guard_gib")},
                         {"python": Path("/opt/lf/envs/regen2k/bin/python"),
                          "base_model": Path("/opt/lf/checkpoints/MiniMax-H3"),
                          "loras": Path("/opt/lf/checkpoints/Minimax-h3-Turbo"), "qwen_canvas": "draft",
                          "min_free_gib": 100.0, "guard_gib": 5.0})
        self.assertEqual(job.worker_command("/py", "/job.py", "/c/job.json", 7),
                         ["/py", "-I", "-B", "/job.py", "--job", "/c/job.json", "--parent", "7"])
        self.assertEqual(job.parse_line("[lf-studio] phase step 2/4"), ("phase", "step 2/4"))
        self.assertNotIn("HF_TOKEN", job.worker_environment({"HF_TOKEN": "x", "PATH": "/bin"}))
        return job

    def registry(self):
        return server_value("MODELS", SOL_H3="sol-h3", TURBO=TURBO, END_STATE={}, BACKENDS={},
                            FRAME_SETS=server_value("FRAME_SETS"), DURATIONS=server_value("DURATIONS"),
                            TASKS=server_value("TASKS"))[TURBO]

    def job_data(self, **changes):
        """A job.json as the server's turbo_job_data writes it."""
        data = {"version": 1, "task": "fl2va", "orientation": "landscape", "resolution": "576p", "seconds": 5,
                "seed": 1, "steps": 4, "lora": "/l/" + LORA4, "shift": [6, 3], "conditioning": "/c/qwen/c.pt",
                "qwen_canvas": "draft", "first_frame": "/c/a.png", "last_frame": None, "audio_noise": "seed",
                "crf": 19, "x264_preset": "veryfast", "load": "fast", "abort_below_gib": 6,
                "paths": {"base_model": "/b", "output": "/c/turbo"}, "meta": {"studio_job": "x"}}
        data.update(changes)
        return data

    def test_the_stand_in(self):
        self.check(self.STUB)

    def test_the_job(self):
        job = self.check(self.REAL)
        model = self.registry()
        # every clip the registry offers is a job.json the job takes; its estimates are the job's
        for row in model["offers"]:
            for task, frames, seconds, orientation, resolution, steps in itertools.product(
                    row["task"], row["frames"], row["seconds"], row["orientation"], row["resolution"], row["steps"]):
                if task == "t2va" and frames != "first":
                    continue
                anchors = frames.split("+") if task == "fl2va" else []
                parsed = job.parse_job(self.job_data(
                    task=task, seconds=seconds, orientation=orientation, resolution=resolution, steps=steps,
                    first_frame="/c/a.png" if "first" in anchors else None,
                    last_frame="/c/b.png" if "last" in anchors else None), "/c/turbo")
                self.assertEqual(parsed["keyframes"], anchors)
        estimates = {f"{r}/{s}/{sec}": int(round(job.estimate_s(r, sec, s), -1))
                     for row in model["offers"] for r in row["resolution"] for s in row["steps"]
                     for sec in row["seconds"]}
        self.assertEqual(model["estimates_s"], estimates)

    def test_the_stand_in_is_the_job(self):
        """The stand-in's tables and answers are the real job's: what the stub studio tests is what runs."""
        real, stub = self.load(self.REAL), self.load(self.STUB)
        for name in ("MODEL", "JOB_VERSION", "RECEIPT", "PRESETS", "OFFERED_STEPS", "CANVAS", "QWEN_CANVASES",
                     "MINIMAX_PROMPT_TOKENS", "JOB_KEYS", "REQUIRED", "MIN_FREE_GIB", "GUARD_GIB", "MIN_SECONDS",
                     "MAX_SECONDS", "ABORT_BELOW_GIB", "USE_GIB", "USE_SPREAD_GIB", "FLOOR_ROWS"):
            self.assertEqual(getattr(stub, name), getattr(real, name), name)
        for environ in ({}, {"LFSTUDIO_H3TURBO_QWEN_CANVAS": "full", "LFSTUDIO_H3TURBO_MIN_FREE_GIB": "-1",
                             "LFSTUDIO_H3TURBO_BASE": "rel"}):
            self.assertEqual(stub.studio_settings(environ), real.studio_settings(environ))
        for resolution, orientation, seconds in itertools.product(("576p", "768p"), ("landscape", "portrait"),
                                                                  range(5, 16)):
            for floor in (0, 100, 110):
                self.assertEqual(stub.min_free_for(resolution, seconds, floor),
                                 real.min_free_for(resolution, seconds, floor), (resolution, seconds, floor))
            for first, last in ((None, None), ("/c/a.png", None), (None, "/c/b.png"), ("/c/a.png", "/c/b.png")):
                data = self.job_data(task="fl2va" if first or last else "t2va", resolution=resolution,
                                     orientation=orientation, seconds=seconds, first_frame=first, last_frame=last)
                self.assertEqual(stub.output_name(stub.parse_job(data, "/c/turbo")),
                                 real.output_name(real.parse_job(data, "/c/turbo")))
        for seconds in (4, 16):
            for job in (stub, real):
                with self.assertRaisesRegex(ValueError, "seconds: a whole number from 5 to 15"):
                    job.parse_job(self.job_data(seconds=seconds), "/c/turbo")


def server_value(name, **names):
    """The value of a module-level assignment of webui/server.py, evaluated from its source with `names`."""
    source = (REPO_DIR / "webui" / "server.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and len(n.targets) == 1
                and getattr(n.targets[0], "id", None) == name)
    return eval(ast.get_source_segment(source, node.value), dict(names))    # noqa: S307 - the repo's own source


class TurboEncodeSeam(TurboTest):
    """Where LightX2V Turbo's prompt encode (LFSTUDIO_H3TURBO_ENCODE_SCRIPT, default bench/h3turbo_encode.py) plugs in:
    it takes the arguments encode_prompt gives it (as the stand-in logged them on a stub studio), refuses a prompt of
    keyframe lines alone with a line ENCODE_LINE reads that names NO_TEXT_ONLY and takes that flag, and writes the files
    and keys the server reads (qwen/conditioning.pt, encode.json image_tokens, qwen-text/prompt.txt and text-only.json
    dropped), here through its own encode() with a stand-in of runtime.pipeline.Worker. Standard library: no Qwen, no
    GPU."""
    helper = REPO_DIR / "bench" / "h3turbo_encode.py"

    def load_helper(self):
        spec = importlib.util.spec_from_file_location("h3turbo_encode_under_test", self.helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_prompt_encode(self):
        prompt = ("How the reference pictures align with the target video - Picture 1 aligns with the 0.00-second "
                  "mark; Picture 2 aligns with the 5.00-second mark.\nA dancer shown in <Picture 1> turns slowly.")
        made = [self.done(self.turbo(task="fl2va", prompt=prompt, resolution="768p", orientation="portrait",
                                     files={"first_frame": [("a.png", png(24, 32))],
                                            "last_frame": [("b.png", png(24, 32))]})),
                self.done(self.turbo(seed=2))]
        helper = self.load_helper()
        encode_line = server_value("ENCODE_LINE", re=re)
        no_text_only = server_value("NO_TEXT_ONLY")
        environ = {"SOL_H3_SPARK_RUNTIME_ROOT": str(self.studio.runtime),
                   "LFSTUDIO_PATHS": str(self.studio.root / "paths-{task}.json")}
        for job in made:
            with self.subTest(task=job["task"]):
                argv = next(r["argv"] for r in self.runs("encodes.log") if job["case_id"] in r["argv"])
                case = Path(argv[argv.index("--case") + 1]).with_name(job["case_id"] + "-seam")
                argv[argv.index("--case") + 1] = str(case)              # the clip's own case has its files already
                plan = helper.make_plan(helper.parse_args(argv), environ, helper.load_job_module())
                self.assertEqual((plan["task"], plan["orientation"], plan["resolution"], plan["canvas_mode"],
                                  plan["seed"], plan["case_id"], plan["prompt"]),
                                 (job["task"], job["orientation"], job["model_opts"]["resolution"],
                                  job["model_opts"]["qwen_canvas"], job["seed"], job["case_id"], job["prompt"]))
                self.assertEqual(plan["keyframes"], ["first", "last"] if job["task"] == "fl2va" else [])
                plan["prompt_tokens"] = "minimax"                       # runtime_side's, from the patch's runtime
                case_spec = {"case_id": plan["case_id"], "prompt": plan["prompt"], "seed": plan["seed"],
                             "task": plan["task"], **{f"{a}_frame": f for a, f in plan["frames"].items()}}
                with contextlib.redirect_stderr(io.StringIO()):         # its progress lines
                    summary = helper.encode(plan, self.worker(plan), {}, {}, case_spec)
                # what encode_prompt reads
                self.assertTrue((case / "qwen" / "conditioning.pt").is_file())
                written = json.loads((case / "encode.json").read_text())
                self.assertEqual((written["image_tokens"], summary["image_tokens"]),
                                 (job["model_opts"]["image_tokens"],) * 2)
                if job["task"] == "fl2va":
                    self.assertEqual((case / "qwen-text" / "prompt.txt").read_text(encoding="utf-8"),
                                     "A dancer turns slowly.")
                    dropped = json.loads((case / "qwen-text" / "text-only.json").read_text())["dropped"]
                    self.assertIsInstance(dropped, list)
                    self.assertIn("alignment line", dropped)
                else:
                    self.assertFalse((case / "qwen-text").exists())
        # a prompt of the keyframe line alone: refused (exit 2) with a line the server reads, naming the flag that
        # skips the text-only encode; with that flag it is planned
        argv = next(r["argv"] for r in self.runs("encodes.log") if made[0]["case_id"] in r["argv"])
        alone = self.dir / "alone.txt"
        alone.write_text("For the target video, at 0.00 seconds into the target video, <Picture 1> is fully "
                         "referenced.", encoding="utf-8")
        argv[argv.index("--prompt-file") + 1] = str(alone)
        argv[argv.index("--case") + 1] += "-alone"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = helper.main(argv, environ)
        self.assertEqual(code, 2)
        m = encode_line.match(printed.getvalue().strip())
        self.assertTrue(m and m.group(1) == "refused" and no_text_only in m.group(2), printed.getvalue())
        plan = helper.make_plan(helper.parse_args(argv + [no_text_only]), environ, helper.load_job_module())
        self.assertNotIn("text_only", plan)

    def test_the_upscales_encode(self):
        """The Upscale's prompt encode of a Sol-H3 Keyframes clip: the real encode takes the arguments run_upscale gives
        it (--upscale, the input folder, the text-only prompt file), plans a t2va encode of exactly that text, and
        writes what encode_upscale_text reads (qwen/conditioning.pt and .json: t2va, that prompt; encode.json's
        rows)."""
        parent = self.clip(task="fl2va", prompt=KEYFRAME_PROMPT, seed=3, files={"first_frame": [("f.png", png())]})
        job = self.done(self.upscale(parent["id"]))
        helper = self.load_helper()
        environ = {"SOL_H3_SPARK_RUNTIME_ROOT": str(self.studio.runtime),
                   "LFSTUDIO_PATHS": str(self.studio.root / "paths-{task}.json")}
        argv = self.runs("encodes.log")[-1]["argv"]
        case = Path(argv[argv.index("--case") + 1])
        self.assertEqual(case, Path(job["case_dirs"][0]) / "input")
        argv[argv.index("--case") + 1] = str(case.with_name("input-seam"))      # the upscale's folder has its files
        plan = helper.make_plan(helper.parse_args(argv), environ, helper.load_job_module())
        self.assertEqual((plan["task"], plan["use"], plan["keyframes"], plan["prompt"], plan["case_id"], plan["seed"]),
                         ("t2va", "upscale", [], job["text"]["prompt"], job["case_id"], 3))
        plan["prompt_tokens"] = "minimax"
        spec = {"case_id": plan["case_id"], "prompt": plan["prompt"], "seed": plan["seed"], "task": "t2va"}
        with contextlib.redirect_stderr(io.StringIO()):
            summary = helper.encode(plan, self.worker(plan), {}, {}, spec)
        seam = case.with_name("input-seam")
        receipt = json.loads((seam / "qwen" / "conditioning.json").read_text())
        self.assertEqual((receipt["task"], receipt["prompt"]), ("t2va", job["text"]["prompt"]))
        self.assertTrue((seam / "qwen" / "conditioning.pt").is_file())
        self.assertEqual(json.loads((seam / "encode.json").read_text())["encodes"]["qwen"]["rows"],
                         summary["encodes"]["qwen"]["rows"])
        self.assertNotIn("for_job", summary)

    @staticmethod
    def worker(plan):
        """A stand-in of runtime.pipeline.Worker for the helper's encode(): each call writes a conditioning and the
        receipt the real worker writes, with the rows of its keyframes at the plan's canvas."""
        width, height = plan["qwen_canvas"]

        class Worker:
            def __init__(self, name, root, paths, config):
                Path(root).mkdir()

            def call(self, case, output_root):
                out = Path(output_root)
                out.mkdir()
                frames = [a for a in ("first", "last") if case.get(f"{a}_frame")]
                text_tokens = len(case["prompt"].split()) + 10
                rows = text_tokens + len(frames) * (plan["image_tokens"] + 2)
                payload = f"stand-in {case['task']} {rows}".encode()
                (out / "conditioning.pt").write_bytes(payload)
                receipt = {"status": "PASS", "task": case["task"], "prompt": case["prompt"],
                           "prompt_tokens": "minimax", "shape": [1, rows, 5120], "text_tokens": text_tokens,
                           "payload_sha256": hashlib.sha256(payload).hexdigest(),
                           "prepared_media": [{"type": "image", "anchor": a, "shape": [height, width, 3]}
                                              for a in frames],
                           "conditioning_path": str(out / "conditioning.pt"), "request_wall_s": 0.1}
                (out / "conditioning.json").write_text(json.dumps(receipt))
                return receipt

            def close(self):
                pass
        return Worker

if __name__ == "__main__":
    unittest.main()
