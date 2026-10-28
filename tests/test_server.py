"""webui/server.py end to end against the stub runtime (tests/stub): no models, no GPU, a few seconds per test.

python3 -m unittest discover -s tests -v      (needs python3 >= 3.10 and ffmpeg; the server uses the standard
library only)
"""
import concurrent.futures
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
import time
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
    """The stub runtime as an older patch has it. 0.3's has SOL_H3_STAGE2_AUDIO but not SOL_H3_PROMPT_TOKENS,
    SOL_H3_STATIC_SHAPES or SOL_H3_DYNAMIC_TEXT_LENGTH; 0.2's has none of them. Such a patch tokenizes as plain does
    and compiles as SOL_H3_STATIC_SHAPES=0 and SOL_H3_DYNAMIC_TEXT_LENGTH=0 do, and 0.2's refines as joint does."""
    package = Path(directory) / f"package-{version}"
    shutil.copytree(Path(__file__).resolve().parent / "stub" / "runtime", package / "runtime",
                    ignore=shutil.ignore_patterns("__pycache__"))
    missing = {"0.3": "prompt_tokens_mode, compile_dynamic, dynamic_text_length",
               "0.2": "stage2_audio_mode, prompt_tokens_mode, compile_dynamic, dynamic_text_length"}[version]
    with (package / "runtime" / "config.py").open("a") as f:
        f.write(f"\ndel {missing}   # the patch of {version} has no such setting\n")
    return package


# What an older patch is asked for, and how the studio and the launcher answer: None = starts; otherwise the start of
# the refusal, which ends "needs a newer patch than the one applied in <package>; run install/10-sana.sh". An empty
# value is the default, as unset.
OLDER_PATCH_CASES = (
    ("0.3", {}, "SOL_H3_PROMPT_TOKENS=minimax (the default since 0.4)"),        # a git pull without install/10-sana.sh
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "minimax"}, "SOL_H3_PROMPT_TOKENS=minimax"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain"}, "SOL_H3_STATIC_SHAPES=1 (the default since 0.4)"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "1"}, "SOL_H3_STATIC_SHAPES=1"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain", "SOL_H3_STATIC_SHAPES": " 0"},
     "SOL_H3_DYNAMIC_TEXT_LENGTH=1 (the default since 0.4)"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "1"},
     "SOL_H3_DYNAMIC_TEXT_LENGTH=1"),
    ("0.3", {"SOL_H3_PROMPT_TOKENS": " Plain", "SOL_H3_STATIC_SHAPES": " 0", "SOL_H3_DYNAMIC_TEXT_LENGTH": " 0"}, None),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0"},
     "SOL_H3_STAGE2_AUDIO=frozen (the default since 0.3)"),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0",
             "SOL_H3_STAGE2_AUDIO": "frozen"}, "SOL_H3_STAGE2_AUDIO=frozen"),
    ("0.2", {"SOL_H3_PROMPT_TOKENS": "plain", "SOL_H3_STATIC_SHAPES": "0", "SOL_H3_DYNAMIC_TEXT_LENGTH": "0",
             "SOL_H3_STAGE2_AUDIO": "JOINT"}, None),
)


def older_patch_settings(settings):
    return {"SOL_H3_PROMPT_TOKENS": "", "SOL_H3_STAGE2_AUDIO": "", "SOL_H3_STATIC_SHAPES": "",
            "SOL_H3_DYNAMIC_TEXT_LENGTH": "", **settings}


class BadPromptTokens(unittest.TestCase):
    """SOL_H3_PROMPT_TOKENS (a patch setting, #38) is checked at start like SOL_H3_STAGE2_AUDIO."""

    def test_older_patch(self):
        """A checkout with an older patch would ignore the setting: the studio refuses any value but what that
        patch does anyway, the default (minimax) included. Without this, a git pull without install/10-sana.sh
        started fine and every clip came out plain (#38)."""
        d = Path(tempfile.mkdtemp(prefix="lf-studio-old-"))
        try:
            packages = {version: older_patch_package(d, version) for version in ("0.3", "0.2")}
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
    SOL_H3_STATIC_SHAPES, SOL_H3_DYNAMIC_TEXT_LENGTH) in the package, here the stub runtime, and refuses a bad value
    with its message, or a value an older patch can't give (#38)."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-check-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        (self.d / "sana/models/minimax_h3").mkdir(parents=True)
        (self.d / "sana/models/minimax_h3/Sol-H3-Spark").symlink_to(Path(__file__).resolve().parent / "stub")
        for task in ("t2va", "fl2va", "ref2va"):
            (self.d / f"paths-{task}.json").write_text("{}")

    def check(self, **settings):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_"))}
        env.update(LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_SANA=str(self.d / "sana"),
                   LFSTUDIO_PYTHON=sys.executable, LFSTUDIO_PATHS=str(self.d / "paths-{task}.json"),
                   SOL_H3_SPARK_RUNTIME_ROOT=str(self.d / "runtime"), LFSTUDIO_STATE_DIR=str(self.d / "state"),
                   LFSTUDIO_HOST="127.0.0.1", LFSTUDIO_PORT=str(free_port()),
                   LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0", PYTHONDONTWRITEBYTECODE="1")
        env.update(settings)
        run = subprocess.run(["bash", str(Path(__file__).resolve().parents[1] / "bin/lf-studio"), "--check"],
                             env=env, capture_output=True, text=True, timeout=120)
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
    their files to Remix as before; the remix itself takes a length offered."""

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


class OldLengthWarmUp(StudioTest):
    """The start-up warm-up after a 3 s clip (the last finished, up to 0.5.0) loads 5 s, as its Remix on the page does."""
    preload = True

    def seed(self):
        self.write_state([finished_job("1" * 32, seconds=10, created=1), finished_job("2" * 32, seconds=3, created=2)])

    def test_warm_up_at_5_s(self):
        s = self.studio.wait_for(lambda s: s["session"].get("loading_key"))["session"]
        self.assertEqual(s["loading_key"], ["t2va", "landscape", 5])
        self.studio.wait_for(lambda s: s["session"]["key"] == ["t2va", "landscape", 5] and s["session"]["state"] == "ready")
        self.assertIn("preloading ('t2va', 'landscape', 5)", self.studio.log.read_text())


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


if __name__ == "__main__":
    unittest.main()
