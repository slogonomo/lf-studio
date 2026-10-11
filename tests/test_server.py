"""webui/server.py end to end against the stub runtime (tests/stub): no models, no GPU, a few seconds per test.

python3 -m unittest discover -s tests -v      (needs python3 >= 3.10 and ffmpeg; the server uses the standard
library only)
"""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio  # noqa: E402

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


class StudioTest(unittest.TestCase):
    """One fresh stub studio per test. Subclasses set the timings; seed() writes state before start."""
    load_s, clip_s, preload = 2, 1, False

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="sol-h3-test-"))
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


def mp4(seconds=3):
    """A small test video with sound (reference videos must be at least 2 s)."""
    out = Path(tempfile.mkdtemp(prefix="sol-h3-mp4-")) / "ref.mp4"
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
                                                              keep="first_frame,last_frame"))
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
                                                                keep="ref_images,ref_videos,ref_audio"))
        self.assertEqual(code, 200, inputs)
        job = self.studio.wait([inputs["id"]])[0]
        names = [(r["type"], Path(r["path"]).name) for r in job["references"]]
        self.assertEqual(names, [("image", "one.png"), ("image", "two.png"), ("video", "clip_ref.mp4"),
                                 ("audio", "voice.wav")])                       # same order, video not re-normalized

    def test_new_files_replace_a_kept_zone_and_unlisted_zones_are_dropped(self):
        a = self.generate(task="fl2va", files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png())]})
        self.studio.wait([a])
        jid = self.generate(task="fl2va", remix_from=a, keep="first_frame",
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
        code, body = self.studio.generate(task="fl2va", remix_from=a, keep="first_frame,last_frame")
        self.assertEqual(code, 400)
        self.assertIn("last.png", body["error"])
        for path in (f"/input/{a}/last_frame/0", f"/input/{a}/first_frame/1", f"/input/{a}/nonsense/0",
                     f"/api/inputs/{'f' * 32}"):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.studio.get(path)
            self.assertEqual(e.exception.code, 404, path)

    def test_unknown_remix_source(self):
        code, body = self.studio.generate(task="fl2va", remix_from="a" * 32, keep="first_frame")
        self.assertEqual(code, 400)
        self.assertIn("no longer in the list", body["error"])


class WarmUp(StudioTest):
    """The start-up warm-up and the takeover rule (docs/PERFORMANCE.md): a same-settings job submitted before
    takeover_until stops the warm-up and loads the models itself; later it waits; other settings stop it."""
    load_s, clip_s, preload = 4, 1, True
    times = {"t2va/landscape/5": 4}

    def seed(self):
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


if __name__ == "__main__":
    unittest.main()
