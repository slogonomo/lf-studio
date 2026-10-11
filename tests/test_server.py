"""webui/server.py end to end against the stub runtime (tests/stub): no models, no GPU, a few seconds per test.

python3 -m unittest discover -s tests -v      (needs python3 >= 3.10 and ffmpeg; the server uses the standard
library only)
"""
import json
import os
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


class HistoryOrder(StudioTest):
    """#5: a request that takes longer (a reference video being normalized) is ordered by when it was added, so
    paging (cursor = 'created') reaches every clip."""
    load_s, clip_s = 1, 0.1

    def test_slow_upload_does_not_hide_a_clip(self):
        import threading
        big = Path(tempfile.mkdtemp(prefix="sol-h3-big-")) / "big.mp4"
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
        self.publish = Path(tempfile.mkdtemp(prefix="sol-h3-publish-"))
        super().setUp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.publish, ignore_errors=True)

    def seed(self):
        self.studio.env.update(SOLH3_PUBLISH_DIR=str(self.publish), **self.env)

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


class SymlinkedRuntime(StudioTest):
    def setUp(self):
        self.real = Path(tempfile.mkdtemp(prefix="sol-h3-real-"))
        link = Path(tempfile.mkdtemp(prefix="sol-h3-link-")) / "root"
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
        d = Path(tempfile.mkdtemp(prefix="sol-h3-bad-"))
        try:
            studio = StubStudio(d, env={"SOLH3_TRASH_DAYS": "inf"})
            with self.assertRaises(RuntimeError) as e:
                studio.start()
            self.assertIn("SOLH3_TRASH_DAYS", str(e.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TrashPurge(Delete):
    env = {"SOLH3_TRASH_DAYS": "0.00001"}                                  # under a second

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
    while clips are queued or running, and needs an earlier clip of the mode for fl2va/ref2va."""
    load_s, clip_s, preload = 2, 1, True

    def seed(self):
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
        self.studio.env["STUB_CLOSE_S"] = "3"
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
        job = self.studio.wait([self.generate(seconds=3)], timeout=60)[0]
        self.assertEqual((job["status"], job["includes_model_load"]), ("done", True))
        self.no_warm_up_for_the_request()
        self.assertEqual(self.studio.state()["session"]["key"], ["t2va", "landscape", 3])

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
        self.assertEqual(self.load(orientation="landscape", seconds=3)[0], 200)
        self.studio.wait_for(lambda st: st["session"]["key"] == ["t2va", "landscape", 3]
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


class WarmUpSkipsDeleted(StudioTest):
    preload = True

    def seed(self):
        self.write_state([finished_job("1" * 32, orientation="portrait", created=1),
                          finished_job("2" * 32, created=2, deleted=3, trashed={})])

    def test_warm_up_uses_the_newest_clip_not_deleted(self):
        s = self.studio.wait_for(lambda s: s["session"].get("loading_key"))["session"]
        self.assertEqual(s["loading_key"], ["t2va", "portrait", 5])


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
