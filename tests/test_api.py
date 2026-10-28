"""The API for agents and scripts (#29) against the stub runtime (tests/stub): GET /api/jobs/<id>[?wait=], its /video,
POST /api/generate with JSON, and the request limits. No models, no GPU, a few seconds per test.

python3 -m unittest tests.test_api -v      (or with the rest: python3 -m unittest discover -s tests)
"""
import base64
import concurrent.futures
import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from test_server import StudioTest, png  # noqa: E402  (StudioTest has no tests of its own)

if not shutil.which("ffmpeg"):
    raise unittest.SkipTest("ffmpeg not installed")

WAV = b"RIFF0000WAVEfmt "          # the stub never reads audio files


def video(seconds=3, sound=True):
    """A small test video, with a sound track or without (reference videos must be at least 2 s)."""
    d = Path(tempfile.mkdtemp(prefix="lf-studio-api-"))
    audio = ["-f", "lavfi", "-i", f"sine=duration={seconds}", "-shortest"] if sound else ["-an"]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s=320x180:r=24:d={seconds}", *audio, "-pix_fmt", "yuv420p", str(d / "v.mp4")],
                   check=True)
    data = (d / "v.mp4").read_bytes()
    shutil.rmtree(d)
    return data


def b64(data):
    return base64.b64encode(data).decode()


def raw(port, head, timeout=10):
    """Send `head` (request line and headers, no body) as it is; returns (status, JSON body) of the answer."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall(head.encode())
        answer = b""
        while chunk := s.recv(65536):
            answer += chunk
    status = int(answer.split(b" ", 2)[1])
    return status, json.loads(answer.split(b"\r\n\r\n", 1)[1] or b"{}")


class ApiTest(StudioTest):
    load_s, clip_s = 1, 0.5

    def get_job(self, jid, query="", path=""):
        code, headers, data = self.studio.request("GET", f"/api/jobs/{jid}{path}{query}", timeout=60)
        return code, json.loads(data) if headers.get("Content-Type") == "application/json" else data

    def submit(self, body):
        return self.studio.post("/api/generate", json_body=body)

    def submit_ok(self, body):
        code, answer = self.submit(body)
        self.assertEqual(code, 200, answer)
        return answer["id"]

    def wait_state(self, jid, predicate, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            code, rec = self.get_job(jid)
            if code == 200 and predicate(rec):
                return rec
            time.sleep(0.05)
        self.fail(f"{jid} never reached the state wanted (last: {rec})")

    def inputs(self):
        return list((self.studio.runtime / "inputs" / "webui").iterdir())


class JobRecord(ApiTest):
    """GET /api/jobs/<id>: one clip as an API record, at once."""
    load_s, clip_s = 3, 1

    def test_unknown_malformed_and_deleted(self):
        self.assertEqual(self.get_job("f" * 32), (404, {"error": "no such clip"}))
        self.assertEqual(self.get_job("F" * 32), (404, {"error": "not found"}))
        self.assertEqual(self.get_job("abc"), (404, {"error": "not found"}))
        jid = self.generate()
        self.studio.wait([jid], timeout=30)
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": jid}), (200, {"ok": True}))
        self.assertEqual(self.get_job(jid), (404, {"error": "no such clip"}))
        self.assertEqual(self.get_job(jid, "?wait=5", "/video"), (404, {"error": "no such clip"}))

    def test_queued_running_done(self):
        a, b, c = self.generate(), self.generate(seed=2), self.generate(seed=3)
        rec = self.wait_state(a, lambda r: r["phase"] == "loading")
        self.assertEqual((rec["state"], rec["status"]), ("running", "loading models + generating"))
        self.assertGreaterEqual(rec["elapsed_s"], 0)
        self.assertNotIn("ahead", rec)
        for jid, ahead in ((b, 0), (c, 1)):
            code, rec = self.get_job(jid)
            self.assertEqual((code, rec["state"], rec["phase"], rec["ahead"]), (200, "queued", None, ahead))
            self.assertNotIn("elapsed_s", rec)
        rec = self.wait_state(b, lambda r: r["phase"] == "generating")      # the models are loaded: a warm clip
        self.assertEqual(rec["status"], "generating")
        rec = self.wait_state(c, lambda r: r["state"] == "done")
        self.assertEqual(rec["phase"], None)
        self.assertEqual(rec["urls"], {"video": f"/video/{c}", "poster": f"/poster/{c}"})
        code, headers, data = self.studio.get(rec["urls"]["video"])
        self.assertEqual((code, headers["Content-Type"], data[4:8]), (200, "video/mp4", b"ftyp"))
        code, headers, _ = self.studio.get(rec["urls"]["poster"])
        self.assertEqual((code, headers["Content-Type"]), (200, "image/jpeg"))
        Path(rec["output"]).unlink()                                         # the MP4 is gone: no urls
        self.assertNotIn("urls", self.get_job(c)[1])
        st = self.studio.state()                                             # /api/state's records are as stored
        self.assertFalse(any("state" in j or "phase" in j for j in st["jobs"]))

    def test_video_file_name(self):
        jid = self.generate(seed=1)
        self.studio.wait([jid], timeout=30)
        code, headers, _ = self.studio.get(f"/video/{jid}", {"Range": "bytes=0-9"})
        self.assertEqual((code, headers["Content-Disposition"]),
                         (206, f'inline; filename="lf-studio_t2va_5s_seed1_{jid[:6]}.mp4"'))


class Wait(ApiTest):
    """GET /api/jobs/<id>?wait=S: answers as soon as the clip is settled, or after S."""
    load_s, clip_s = 2, 1

    def test_returns_when_it_settles(self):
        jid = self.generate()
        self.wait_state(jid, lambda r: r["state"] == "running")
        t0 = time.time()
        code, rec = self.get_job(jid, "?wait=20")
        took = time.time() - t0
        self.assertEqual((code, rec["state"], rec["status"]), (200, "done", "done"))
        self.assertLess(took, self.load_s + self.clip_s + 2)
        self.assertIn("finished", rec)
        self.assertIn("wall_s", rec)
        self.assertNotIn("saving", rec)
        t0 = time.time()
        self.assertEqual(self.get_job(jid, "?wait=100000")[1]["state"], "done")   # settled: at once
        self.assertLess(time.time() - t0, 1)

    def test_bad_and_immediate_waits(self):
        jid = self.generate()
        for bad in ("abc", "-1", "nan", "inf", "-inf", "1e999", ""):
            self.assertEqual(self.get_job(jid, f"?wait={bad}"),
                             (400, {"error": "wait must be a number of seconds, 0 or more"}), bad)
            self.assertEqual(self.get_job(jid, f"?wait={bad}", "/video")[0], 400, bad)
        for query in ("", "?wait=0", "?other=1"):
            t0 = time.time()
            code, rec = self.get_job(jid, query)
            self.assertEqual(code, 200)
            self.assertIn(rec["state"], ("queued", "running"))
            self.assertLess(time.time() - t0, 0.5)

    def test_failed(self):
        jid = self.generate(prompt="STUB_FAIL")
        code, rec = self.get_job(jid, "?wait=30")
        self.assertEqual((code, rec["state"]), (200, "failed"))
        self.assertIn("STUB_FAIL", rec["error"])

    def test_cancel_of_a_queued_clip_ends_the_wait(self):
        a, b = self.generate(), self.generate(seed=2)
        out = {}
        waiter = threading.Thread(target=lambda: out.update(r=self.get_job(b, "?wait=30"), t=time.time()))
        waiter.start()
        time.sleep(0.3)
        t0 = time.time()
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": b})[0], 200)
        waiter.join(10)
        self.assertLess(out["t"] - t0, 1)
        self.assertEqual((out["r"][0], out["r"][1]["state"]), (200, "cancelled"))
        self.studio.wait([a], timeout=30)

    def test_many_waiters(self):
        jid = self.generate()
        with concurrent.futures.ThreadPoolExecutor(20) as pool:
            waits = [pool.submit(self.get_job, jid, "?wait=30") for _ in range(20)]
            time.sleep(0.5)
            t0 = time.time()
            self.assertEqual(self.studio.get("/api/state")[0], 200)
            self.assertLess(time.time() - t0, 1)
            self.assertFalse(any(w.done() for w in waits))
            answers = [w.result(timeout=40) for w in waits]
        self.assertEqual({(code, rec["state"]) for code, rec in answers}, {(200, "done")})


class WaitLong(ApiTest):
    load_s, clip_s = 6, 1

    def seed(self):
        self.studio.env["STUB_CLOSE_S"] = "2"

    def test_wait_runs_out(self):
        jid = self.generate()
        t0 = time.time()
        code, rec = self.get_job(jid, "?wait=1")
        self.assertTrue(1 <= time.time() - t0 < 2.5, time.time() - t0)
        self.assertEqual((code, rec["state"]), (200, "running"))
        code, rec = self.get_job(jid, "?wait=1", "/video")
        self.assertEqual((code, rec["state"]), (202, "running"))

    def test_cancel_of_a_running_clip_ends_the_wait_after_the_close(self):
        jid = self.generate()
        self.wait_state(jid, lambda r: r["phase"] == "loading")
        out = {}
        waiter = threading.Thread(target=lambda: out.update(r=self.get_job(jid, "?wait=30"), t=time.time()))
        waiter.start()
        time.sleep(0.3)
        t0 = time.time()
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": jid})[0], 200)
        self.assertEqual(self.get_job(jid)[1]["phase"], "cancelling")
        waiter.join(20)
        self.assertGreaterEqual(out["t"] - t0, 1.8)                         # the pipeline's close (2 s) came first
        self.assertEqual((out["r"][0], out["r"][1]["state"]), (200, "cancelled"))
        code, answer = self.get_job(jid, "?wait=5", "/video")
        self.assertEqual((code, answer["error"], answer["job"]["state"]), (409, "the clip was cancelled", "cancelled"))


class Shutdown(ApiTest):
    """A wait that the studio's shutdown interrupts gets 503 with the record, before the process ends; after the
    restart the clip reads failed."""
    load_s, clip_s = 8, 1

    def seed(self):
        self.studio.env["STUB_CLOSE_S"] = "1"

    def test_waiter_told_before_exit(self):
        jid = self.generate()
        self.wait_state(jid, lambda r: r["phase"] == "loading")
        out = {}
        waiter = threading.Thread(target=lambda: out.update(r=self.get_job(jid, "?wait=60")))
        video = threading.Thread(target=lambda: out.update(v=self.get_job(jid, "?wait=60", "/video")))
        waiter.start()
        video.start()
        time.sleep(0.5)
        self.studio.proc.send_signal(signal.SIGTERM)
        waiter.join(15)
        video.join(15)
        for key in ("r", "v"):
            code, answer = out[key]
            self.assertEqual((code, answer["error"]), (503, "the studio is shutting down; ask again once it is back"))
            self.assertEqual((answer["job"]["id"], answer["job"]["state"]), (jid, "running"))
        self.studio.proc.wait(20)
        self.studio.stop()
        self.studio.start()
        code, rec = self.get_job(jid, "?wait=10")
        self.assertEqual((code, rec["state"], rec["error"]), (200, "failed", "server restarted"))


class VideoWait(ApiTest):
    """GET /api/jobs/<id>/video?wait=S: the MP4 on the same connection once the clip is done."""
    load_s, clip_s = 2, 1

    def test_streams_when_done(self):
        jid = self.generate(seed=5)
        t0 = time.time()
        code, rec = self.get_job(jid, "", "/video")
        self.assertEqual(code, 202)                                            # no wait: at once
        self.assertIn(rec["state"], ("queued", "running"))
        self.assertLess(time.time() - t0, 0.5)
        code, headers, data = self.studio.request("GET", f"/api/jobs/{jid}/video?wait=30", timeout=40)
        self.assertEqual((code, headers["Content-Type"]), (200, "video/mp4"))
        self.assertEqual(headers["Content-Disposition"], f'inline; filename="lf-studio_t2va_5s_seed5_{jid[:6]}.mp4"')
        self.assertEqual(data, self.studio.get(f"/video/{jid}")[2])
        code, headers, data = self.studio.request("GET", f"/api/jobs/{jid}/video", headers={"Range": "bytes=0-99"})
        self.assertEqual((code, len(data), headers["Content-Range"].split("/")[0]), (206, 100, "bytes 0-99"))
        code, headers, data = self.studio.request("HEAD", f"/api/jobs/{jid}/video?wait=5")
        self.assertEqual((code, data, headers["Content-Type"]), (200, b"", "video/mp4"))
        Path(self.studio.job(jid)["output"]).unlink()
        code, answer = self.get_job(jid, "?wait=5", "/video")
        self.assertEqual((code, answer["error"], answer["job"]["state"]), (404, "no such video", "done"))

    def test_failed(self):
        jid = self.generate(prompt="STUB_FAIL")
        code, answer = self.get_job(jid, "?wait=30", "/video")
        self.assertEqual((code, answer["job"]["state"]), (409, "failed"))
        self.assertTrue(answer["error"].startswith("the clip failed: RuntimeError"), answer["error"])


class JsonGenerate(ApiTest):
    """POST /api/generate with a JSON body: the form's fields, files as base64, the same checks."""

    def test_minimal_t2va(self):
        code, answer = self.submit({"prompt": "a quiet lake"})
        self.assertEqual(code, 200, answer)
        job = answer["job"]
        self.assertEqual((job["id"], job["state"], job["task"], job["orientation"], job["seconds"]),
                         (answer["id"], "queued", "t2va", "landscape", 5))
        self.assertIsInstance(job["seed"], int)
        self.assertNotIn("labels", job)
        code, rec = self.get_job(answer["id"], "?wait=30")
        self.assertEqual(rec["state"], "done")
        jid = self.submit_ok({"prompt": "x", "task": "t2va", "orientation": "portrait", "seconds": 3, "seed": 0,
                              "first_frame": None, "ref_images": None})          # null = left out
        job = self.studio.wait([jid], timeout=30)[0]
        self.assertEqual((job["status"], job["orientation"], job["seconds"], job["seed"]), ("done", "portrait", 3, 0))

    def test_fl2va_object_and_data_uri(self):
        frame = png(40, 30)
        a = self.submit_ok({"prompt": "x", "task": "fl2va", "first_frame": {"name": "kitchen.png", "data": b64(frame)}})
        b = self.submit_ok({"prompt": "x", "first_frame": "data:image/png;base64," + b64(frame),
                            "last_frame": {"name": "end shot.jpg", "data": b64(png())}})
        c = self.submit_ok({"prompt": "x", "last_frame": "data:image/jpeg;name=x;base64," + b64(png())})
        ja, jb, jc = self.studio.wait([a, b, c], timeout=40)
        self.assertEqual([j["status"] for j in (ja, jb, jc)], ["done"] * 3)
        self.assertEqual(Path(ja["first_frame"]).name, "kitchen.png")
        self.assertEqual(Path(ja["first_frame"]).read_bytes(), frame)
        self.assertEqual(ja["labels"], [{"label": "<Picture 1>", "zone": "first_frame", "index": 0,
                                         "name": "kitchen.png"}])
        self.assertEqual(jb["task"], "fl2va")
        self.assertEqual([(lb["label"], lb["zone"], lb["name"]) for lb in jb["labels"]],
                         [("<Picture 1>", "first_frame", "first_frame.png"), ("<Picture 2>", "last_frame", "end_shot.jpg")])
        self.assertEqual(jc["labels"], [{"label": "<Picture 1>", "zone": "last_frame", "index": 0, "name": "last_frame.jpg"}])
        self.assertEqual(self.get_job(c)[1]["labels"], jc["labels"])

    def test_wrapped_base64(self):
        data = base64.encodebytes(png(64, 48)).decode()                          # 76 characters a line, as base64(1)
        self.assertIn("\n", data.strip())
        body = '{"prompt": "x", "first_frame": {"name": "f.png", "data": "' + data + '"}}'   # raw line breaks
        code, _, answer = self.studio.request("POST", "/api/generate", body.encode(),
                                              {"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(code, 200, answer)
        job = self.studio.wait([json.loads(answer)["id"]], timeout=30)[0]
        self.assertEqual(Path(job["first_frame"]).read_bytes(), png(64, 48))

    def test_task_inferred_and_refused(self):
        a = self.submit_ok({"prompt": "x", "first_frame": {"name": "f.png", "data": b64(png())}})
        b = self.submit_ok({"prompt": "x", "ref_images": [{"name": "r.png", "data": b64(png())}], "ref_audio": []})
        self.assertEqual([self.get_job(j)[1]["task"] for j in (a, b)], ["fl2va", "ref2va"])
        f = {"name": "f.png", "data": b64(png())}
        for body, error in (
                ({"task": "t2va", "first_frame": f}, "t2va takes no files"),
                ({"task": "fl2va", "ref_images": [f]}, "ref_images is used only by ref2va"),
                ({"task": "ref2va", "first_frame": f, "ref_images": [f]}, "first_frame is used only by fl2va"),
                ({"first_frame": f, "ref_videos": [f]}, "first_frame and ref_videos can't go together"),
                ({"task": "fl2va", "first_frame": "https://example.com/f.png"}, "first_frame: the studio does not "
                 "fetch URLs; send the file as base64, or use bin/lf-cli, which downloads URLs itself"),
                ({"ref_images": [f, {"url": "http://example.com/r.png"}]}, "ref_images[1]: the studio does not fetch"),
                ({"keep": ["first_frame:0"]}, "keep needs remix_from")):
            code, answer = self.submit(dict(body, prompt="x"))
            self.assertEqual(code, 400, body)
            self.assertIn(error, answer["error"])
        self.assertEqual(len(self.inputs()), 2)                                  # refused: nothing saved

    def test_ref2va_labels(self):
        a = self.submit_ok({"prompt": "x", "ref_images": [{"name": "face.png", "data": b64(png())}],
                            "ref_videos": [{"name": "talk.mp4", "data": b64(video())}],
                            "ref_audio": [{"name": "voice.wav", "data": b64(WAV)}]})
        job = self.studio.wait([a], timeout=40)[0]
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["labels"], [
            {"label": "<Picture 1>", "zone": "ref_images", "index": 0, "name": "face.png"},
            {"label": "<Audio 1>", "zone": "ref_videos", "index": 0, "name": "talk_ref.mp4", "sound_of": "<Video 1>"},
            {"label": "<Video 1>", "zone": "ref_videos", "index": 0, "name": "talk_ref.mp4"},
            {"label": "<Audio 2>", "zone": "ref_audio", "index": 0, "name": "voice.wav"}])
        b = self.submit_ok({"prompt": "x", "ref_videos": [{"name": "silent.mp4", "data": b64(video(sound=False))},
                                                          {"name": "loud.mp4", "data": b64(video())}],
                            "ref_audio": ["data:audio/wav;base64," + b64(WAV)]})
        labels = self.studio.wait([b], timeout=40)[0]["labels"]
        self.assertEqual([(lb["label"], lb["index"], lb.get("sound_of")) for lb in labels],
                         [("<Video 1>", 0, None), ("<Audio 1>", 1, "<Video 2>"), ("<Video 2>", 1, None),
                          ("<Audio 2>", 0, None)])
        listing = json.loads(self.studio.get(f"/api/inputs/{b}")[2])["inputs"]        # zone and index as listed
        self.assertEqual([i["name"] for i in listing["ref_videos"]], ["silent_ref.mp4", "loud_ref.mp4"])
        self.assertEqual(listing["ref_audio"][0]["name"], labels[3]["name"])

    def test_remix_with_kept_files(self):
        a = self.submit_ok({"prompt": "x", "first_frame": {"name": "first.png", "data": b64(png())},
                            "last_frame": {"name": "last.png", "data": b64(png(40))}})
        src = self.studio.wait([a], timeout=30)[0]
        b = self.submit_ok({"prompt": "again", "remix_from": a, "keep": ["last_frame:0"]})   # task from what is kept
        job = self.studio.wait([b], timeout=30)[0]
        self.assertEqual((job["status"], job["task"]), ("done", "fl2va"))
        self.assertEqual(Path(job["last_frame"]).stat().st_ino, Path(src["last_frame"]).stat().st_ino)
        self.assertEqual(job["labels"], [{"label": "<Picture 1>", "zone": "last_frame", "index": 0, "name": "last.png"}])

    def test_strict(self):
        f = {"name": "f.png", "data": b64(png())}
        for body, error in (
                (b"{not json", "the body is not valid JSON"),
                (b"\xff\xfe\x00", "the body is not valid JSON"),
                (b'["prompt"]', "the body must be a JSON object"),
                ({"prompt": "x", "frst_frame": f}, 'unknown key "frst_frame"'),
                ({"prompt": "x", "seconds": "5"}, "seconds must be an integer"),
                ({"prompt": "x", "seconds": 5.5}, "seconds must be an integer"),
                ({"prompt": "x", "seconds": 5.0}, "seconds must be an integer"),
                ({"prompt": "x", "seed": 1.5}, "seed must be an integer or null"),
                ({"prompt": "x", "seed": True}, "seed must be an integer or null"),
                ({"prompt": 3}, "prompt must be a string"),
                ({"prompt": "x", "task": ["fl2va"]}, "task must be a string"),
                ({"prompt": "x", "first_frame": {"name": "f.png", "data": "!!!!"}}, "first_frame: the data is not valid base64"),
                ({"prompt": "x", "first_frame": {"name": "f.png", "data": "iVBORw0KGgo"}}, "not valid base64"),
                ({"prompt": "x", "first_frame": {"name": "f.png", "data": ""}}, "first_frame is empty"),
                ({"prompt": "x", "first_frame": {"data": b64(png())}}, "first_frame: name must be a file name"),
                ({"prompt": "x", "first_frame": {"name": "f.png", "data": b64(png()), "type": "image"}},
                 'first_frame: unknown key "type"'),
                ({"prompt": "x", "first_frame": [f]}, "first_frame must be one file"),
                ({"prompt": "x", "first_frame": b64(png())}, "first_frame: a file is"),
                ({"prompt": "x", "first_frame": "data:;base64,abc"}, "first_frame: unsupported file type (none)"),
                ({"prompt": "x", "first_frame": "data:image/png,abc"}, "first_frame: a file is"),   # not base64
                ({"prompt": "x", "ref_images": f}, "ref_images must be a list"),
                ({"prompt": "x", "remix_from": "f" * 32, "keep": "first_frame:0"}, "keep must be a list"),
                ({"prompt": "x", "remix_from": "f" * 32, "keep": ["frame:0"]}, "keep must be a list")):
            raw_body = body if isinstance(body, bytes) else json.dumps(body).encode()
            code, _, answer = self.studio.request("POST", "/api/generate", raw_body, {"Content-Type": "application/json"})
            self.assertEqual(code, 400, body)
            message = json.loads(answer)["error"]
            self.assertTrue(message.startswith("ValueError: "), message)
            self.assertIn(error, message, body)
        self.assertEqual(self.inputs(), [])
        self.assertEqual(self.studio.state()["jobs"], [])

    def test_prompt_with_half_an_emoji(self):
        """JSON can escape a lone UTF-16 surrogate (an emoji cut in half by a length limit), which the form can't
        send: refused at once, not queued to fail on the GPU. A whole emoji (a surrogate pair) is fine."""
        error = "ValueError: prompt is not valid text (an unpaired UTF-16 surrogate, e.g. an emoji cut in half)"
        ctype = {"Content-Type": "application/json"}
        for body in (b'{"prompt": "a quiet lake at dawn \\ud83c", "seed": 7}', b'{"prompt": "\\udf05 dawn"}'):
            code, _, answer = self.studio.request("POST", "/api/generate", body, ctype)
            self.assertEqual((code, json.loads(answer)["error"]), (400, error), body)
        self.assertEqual(self.studio.state()["jobs"], [])
        code, _, answer = self.studio.request("POST", "/api/generate", b'{"prompt": "dawn \\ud83c\\udf05"}', ctype)
        self.assertEqual(code, 200, answer)
        self.assertEqual(self.studio.job(json.loads(answer)["id"])["prompt"], "dawn \U0001F305")

    def test_form_and_json_give_the_same_errors(self):
        frames = lambda name: {"first_frame": [(name, png())]}
        images = [(f"r{i}.png", png()) for i in range(10)]
        cases = (   # (form fields, form files, JSON body)
            ({"seconds": "9"}, None, {"seconds": 9}),
            ({"prompt": "  "}, None, {"prompt": "  "}),
            ({"task": "x"}, None, {"task": "x"}),
            ({"task": "fl2va"}, frames("f.gif"), {"task": "fl2va", "first_frame": {"name": "f.gif", "data": b64(png())}}),
            ({"task": "ref2va"}, {"ref_images": images},
             {"task": "ref2va", "ref_images": [{"name": n, "data": b64(d)} for n, d in images]}),
            ({"task": "fl2va"}, None, {"task": "fl2va"}),
            ({"seed": "-1"}, None, {"seed": -1}),
            ({"seed": str(2**63)}, None, {"seed": 2**63}),
            ({"prompt": "x" * 32769}, None, {"prompt": "x" * 32769}))
        for form, files, body in cases:
            code, by_form = self.studio.post("/api/generate", dict({"prompt": "x"}, **form), files)
            code2, by_json = self.submit(dict({"prompt": "x"}, **body))
            self.assertEqual((code, code2), (400, 400), (by_form, by_json))
            self.assertEqual(by_form["error"], by_json["error"])
        self.assertEqual(by_form["error"], "ValueError: prompt is too long (32769 characters; at most 32768)")
        self.assertEqual(self.inputs(), [])
        jid = self.submit_ok({"prompt": "x" * 32768, "seconds": 3})
        self.assertEqual(len(self.studio.job(jid)["prompt"]), 32768)

    def test_form_answer_and_task_inferred(self):
        code, answer = self.studio.generate()
        self.assertEqual((code, answer["job"]["id"], answer["job"]["state"]), (200, answer["id"], "queued"))
        tasks = []
        for files in (None, {"first_frame": [("f.png", png())]}, {"ref_images": [("r.png", png())]}):
            code, answer = self.studio.post("/api/generate", {"prompt": "x"}, files)     # no task field
            self.assertEqual(code, 200, answer)
            tasks.append(answer["job"]["task"])
        self.assertEqual(tasks, ["t2va", "fl2va", "ref2va"])
        code, answer = self.studio.post("/api/generate", {"prompt": "x"}, {"last_frame": [("f.png", png())],
                                                                          "ref_audio": [("a.wav", WAV)]})
        self.assertEqual(code, 400)
        self.assertIn("last_frame and ref_audio can't go together", answer["error"])
        code, answer = self.studio.post("/api/generate", {"prompt": "x", "task": "t2va"},
                                        {"first_frame": [("f.png", png())]})      # the form's own leniency stays
        self.assertEqual((code, answer["job"]["task"]), (200, "t2va"))


class Limits(ApiTest):
    """Content-Length and Content-Type of POST /api/generate, and the small JSON POSTs."""

    def test_generate_body(self):
        port = self.studio.port
        post = "POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        self.assertEqual(raw(port, post + "\r\n"),
                         (411, {"error": "Content-Length required (chunked uploads are not supported)"}))
        self.assertEqual(raw(port, post + "Transfer-Encoding: chunked\r\n\r\n")[0], 411)
        for bad in ("abc", "-5", "1.5"):
            self.assertEqual(raw(port, post + f"Content-Length: {bad}\r\n\r\n"), (400, {"error": "bad Content-Length"}))
        t0 = time.time()
        self.assertEqual(raw(port, post + "Content-Length: 300000000\r\n\r\n"),
                         (413, {"error": "upload too large (200 MB max)"}))
        self.assertLess(time.time() - t0, 2)
        code, _, answer = self.studio.request("POST", "/api/generate", b"prompt=x", {"Content-Type": "text/plain"})
        self.assertEqual((code, json.loads(answer)), (415, {"error": "send multipart/form-data or application/json"}))
        code, _, answer = self.studio.request("POST", "/api/generate", b"prompt=x",
                                              {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(code, 415)
        self.assertEqual(self.studio.state()["jobs"], [])

    def test_small_bodies(self):
        for path in ("/api/cancel", "/api/delete", "/api/undelete", "/api/load", "/api/unload", "/api/upscale"):
            head = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            self.assertEqual(raw(self.studio.port, head + f"Content-Length: {64 * 1024 + 1}\r\n\r\n"),
                             (413, {"error": "request body too large (64 KiB max)"}), path)
            self.assertEqual(raw(self.studio.port, head + "Content-Length: x\r\n\r\n")[0], 400, path)
        body = json.dumps({"id": "f" * 32}).encode()
        body += b" " * (64 * 1024 - len(body))                                 # at the limit: read as usual
        code, _, answer = self.studio.request("POST", "/api/cancel", body, {"Content-Type": "application/json"})
        self.assertEqual((code, json.loads(answer)), (404, {"error": "no such clip"}))
        code, _, _ = self.studio.request("POST", "/api/unload", None)            # Content-Length: 0
        self.assertEqual(code, 200)
        self.assertEqual(raw(self.studio.port, "POST /api/unload HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")[0], 200)


class QueueCap(ApiTest):
    """LFSTUDIO_MAX_QUEUED: a submit while that many clips wait gets 429, the page's form included."""
    load_s, clip_s = 6, 1

    def seed(self):
        self.studio.env["LFSTUDIO_MAX_QUEUED"] = "2"

    def test_full_queue(self):
        a = self.generate()
        self.wait_state(a, lambda r: r["state"] == "running")                  # the running clip does not count
        b, c = self.generate(seed=2), self.generate(seed=3)
        full = "the queue is full (2 clips waiting); try again when one has finished"
        code, headers, answer = self.studio.request("POST", "/api/generate", json.dumps({"prompt": "x"}).encode(),
                                                    {"Content-Type": "application/json"})
        self.assertEqual((code, headers["Retry-After"], json.loads(answer)), (429, "60", {"error": full}))
        code, answer = self.studio.generate(task="fl2va", files={"first_frame": [("f.png", png())]})
        self.assertEqual((code, answer), (429, {"error": full}))
        self.assertEqual(len(self.inputs()), 3)                                  # a, b and c's: nothing saved
        self.assertEqual(len(self.studio.state()["jobs"]), 3)
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": b})[0], 200)
        d = self.submit_ok({"prompt": "x", "seed": 4})
        self.assertEqual(self.get_job(d)[1]["ahead"], 1)
        self.studio.wait([a, c, d], timeout=60)


class QueueCapDefault(ApiTest):
    load_s, clip_s = 10, 1

    def seed(self):
        self.studio.env.pop("LFSTUDIO_MAX_QUEUED")       # the stub's own 100: some older tests queue more than 20

    def test_twenty_wait(self):
        a = self.generate()
        self.wait_state(a, lambda r: r["state"] == "running")
        waiting = [self.submit_ok({"prompt": "x", "seed": i}) for i in range(20)]
        self.assertEqual(self.get_job(waiting[-1])[1]["ahead"], 19)
        self.assertEqual(self.submit({"prompt": "x"}),
                         (429, {"error": "the queue is full (20 clips waiting); try again when one has finished"}))


class UpscaleRecord(ApiTest):
    """An upscale (the experimental Upscale, on in the stub studio) through the API: its record while queued and
    running, ?wait=, /video, and "upscalable" on a finished clip's record."""
    load_s, clip_s = 2, 2

    def seed(self):
        self.studio.env["STUB_UPSCALE_S"] = "2"

    def test_record_wait_and_video(self):
        jid = self.generate()
        rec = self.get_job(jid, "?wait=30")[1]
        self.assertEqual(rec["upscalable"], {"ok": True, "draft": True, "clip_audio": True})
        code, body = self.studio.upscale(jid, resolution="1080p")
        self.assertEqual(code, 200, body)
        uid = body["id"]
        self.assertEqual((body["job"]["kind"], body["job"]["parent"], body["job"]["resolution"]),
                         ("upscale", jid, "1080p"))
        rec = self.wait_state(uid, lambda r: r["phase"] == "upscaling")
        self.assertEqual((rec["state"], rec["status"]), ("running", "upscaling"))
        self.assertIn(rec["upscale_phase"], ("starting", "encoding", "loading", "refine 1/2", "refine 2/2",
                                             "decoding", "saving", "adding the sound"))
        self.assertGreaterEqual(rec["elapsed_s"], 0)
        self.assertNotIn("upscalable", rec)
        self.assertEqual(self.get_job(uid, "?wait=0.5", "/video")[0], 202)
        code, rec = self.get_job(uid, "?wait=30")
        self.assertEqual((code, rec["state"], rec["phase"]), (200, "done", None))
        self.assertEqual(rec["urls"], {"video": f"/video/{uid}", "poster": f"/poster/{uid}"})
        self.assertEqual(rec["upscalable"], {"ok": False, "why": "an upscale can't be upscaled again"})
        code, headers, data = self.studio.request("GET", f"/api/jobs/{uid}/video?wait=5")
        self.assertEqual((code, headers["Content-Type"], data[4:8]), (200, "video/mp4", b"ftyp"))
        self.assertEqual(headers["Content-Disposition"],
                         f'inline; filename="lf-studio_t2va_5s_seed1_{jid[:6]}_1080p_{uid[:6]}.mp4"')

    def test_clips_run_first(self):
        """ahead counts the jobs that run before: every clip queued runs before an upscale, even one queued later."""
        parent = self.generate()
        self.get_job(parent, "?wait=30")
        a = self.generate(seed=2)
        self.wait_state(a, lambda r: r["phase"] == "generating")
        uid = self.studio.upscale(parent)[1]["id"]
        b, c = self.generate(seed=3), self.generate(seed=4)
        second = self.studio.upscale(parent, resolution="4k")[1]["id"]
        self.assertEqual([self.get_job(j)[1]["ahead"] for j in (uid, b, c, second)], [2, 0, 1, 3])
        rec = self.wait_state(uid, lambda r: r["state"] == "running", timeout=60)
        self.assertEqual(self.get_job(c, "?wait=1")[1]["state"], "done")   # the clips were done before it began
        self.assertEqual(self.get_job(second)[1]["ahead"], 0)
        self.assertIn(rec["phase"], ("starting", "unloading", "upscaling"))
        self.studio.wait([second], timeout=60)


class UpscaleFailedRecord(ApiTest):
    def seed(self):
        self.studio.env["STUB_UPSCALE_FAIL"] = "1"

    def test_failed(self):
        jid = self.generate()
        self.get_job(jid, "?wait=30")
        uid = self.studio.upscale(jid)[1]["id"]
        code, body = self.get_job(uid, "?wait=30", "/video")
        self.assertEqual(code, 409)
        self.assertEqual(body["error"],
                         "the clip failed: UpscaleFailed: OutOfMemoryError: CUDA out of memory (STUB_UPSCALE_FAIL)")
        self.assertEqual((body["job"]["state"], body["job"]["upscale_phase"]), ("failed", "refine 1/2"))


API_OFF = {"error": "the API for programs is off on this studio (LFSTUDIO_API=0)"}
PAGE = Path(__file__).resolve().parents[1] / "webui" / "index.html"


class ApiOn(ApiTest):
    """LFSTUDIO_API unset: the API for programs is on, as in 0.4 (#58), also for a submit marked as a program's."""

    def test_on_by_default(self):
        self.assertIs(self.studio.state()["api"], True)
        jid = self.submit_ok({"prompt": "x", "seconds": 3})
        code, _, answer = self.studio.request("POST", "/api/generate", json.dumps({"prompt": "y"}).encode(),
                                              {"Content-Type": "application/json", "X-LFStudio-Client": "lf-cli/0.5.0"})
        self.assertEqual(code, 200, answer)
        self.assertEqual(self.get_job(jid)[0], 200)
        self.studio.wait([jid, json.loads(answer)["id"]], timeout=30)


class ApiOff(ApiTest):
    """LFSTUDIO_API=0 (#58): JSON submits, submits marked by a program (X-LFStudio-Client, as bin/lf-cli sends) and
    GET/HEAD /api/jobs/<id>[/video] answer 403 with a message naming the setting; everything the page uses works."""

    def seed(self):
        self.studio.env["LFSTUDIO_API"] = "0"

    def assert_off(self, method, path, body=None, headers=None):
        code, h, data = self.studio.request(method, path, body, headers)
        self.assertEqual((code, h["Content-Type"]), (403, "application/json"), path)
        self.assertEqual(json.loads(data) if method != "HEAD" else data, API_OFF if method != "HEAD" else b"", path)

    def test_refused(self):
        self.assertIs(self.studio.state()["api"], False)
        self.assertIn("the API for programs is off (LFSTUDIO_API=0)", self.studio.log.read_text())
        self.assert_off("POST", "/api/generate", json.dumps({"prompt": "x"}).encode(),
                        {"Content-Type": "application/json"})
        self.assert_off("POST", "/api/generate", json.dumps({"prompt": "x", "first_frame": {
            "name": "f.png", "data": b64(png())}}).encode(), {"Content-Type": "application/json; charset=utf-8"})
        b = "apioffboundary"
        form = (f'--{b}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nx\r\n--{b}--\r\n').encode()
        self.assert_off("POST", "/api/generate", form, {"Content-Type": f"multipart/form-data; boundary={b}",
                                                        "X-LFStudio-Client": "lf-cli/0.5.0"})
        self.assertEqual(self.studio.state()["jobs"], [])                     # nothing queued, no files kept
        self.assertEqual(self.inputs(), [])
        jid = self.generate()                                                 # the page's form
        self.studio.wait([jid], timeout=30)
        for path in (f"/api/jobs/{jid}", f"/api/jobs/{jid}?wait=5", f"/api/jobs/{jid}/video",
                     f"/api/jobs/{jid}/video?wait=5", f"/api/jobs/{'f' * 32}", f"/api/jobs/{'f' * 32}/video?wait=1"):
            for method in ("GET", "HEAD"):
                self.assert_off(method, path)
        self.assertEqual(self.get_job("abc"), (404, {"error": "not found"}))  # not an id: not the API's path

    def test_the_page_works(self):
        """Every endpoint the page calls (webui/index.html), none of them /api/jobs/<id>. (/api/upscale, the
        experimental Upscale's: test_upscale_is_the_pages.)"""
        page = PAGE.read_text()
        self.assertNotIn("/api/jobs/", page)
        self.assertEqual(sorted(set(re.findall(r"/api/[a-z]+", page))),
                         ["/api/cancel", "/api/delete", "/api/generate", "/api/inputs", "/api/jobs", "/api/load",
                          "/api/state", "/api/undelete", "/api/unload", "/api/upscale"])
        for path in ("/", "/index.html", "/health"):
            self.assertEqual(self.studio.get(path)[0], 200, path)
        a = self.generate(task="fl2va", files={"first_frame": [("f.png", png())]})
        code, answer = self.studio.post("/api/generate", {"prompt": "remix", "task": "fl2va", "remix_from": a,
                                                          "keep": "first_frame:0"})      # Remix: the form as well
        self.assertEqual(code, 200, answer)
        b = answer["id"]
        self.studio.wait([a, b], timeout=30)
        st = self.studio.state()
        self.assertEqual({j["id"] for j in st["jobs"]}, {a, b})
        older = json.loads(self.studio.get(f"/api/jobs?before={st['jobs'][0]['created']}&limit=24")[2])
        self.assertEqual([j["id"] for j in older["jobs"]], [a])
        for path in (f"/video/{a}", f"/poster/{a}", f"/still/{a}", f"/input/{a}/first_frame/0"):
            self.assertEqual(self.studio.get(path)[0], 200, path)
        inputs = json.loads(self.studio.get(f"/api/inputs/{a}")[2])
        self.assertEqual(inputs["inputs"]["first_frame"][0]["url"], f"/input/{a}/first_frame/0")
        self.assertEqual(self.studio.post("/api/load", json_body={"task": "fl2va", "orientation": "landscape",
                                                                  "seconds": 5})[0], 200)
        self.assertEqual(self.studio.post("/api/unload")[0], 200)
        self.assertEqual(self.studio.post("/api/delete", json_body={"id": b}), (200, {"ok": True}))
        self.assertEqual(self.studio.post("/api/undelete", json_body={"id": b})[0], 200)
        c = self.generate(seed=3)
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": c})[0], 200)
        self.assertEqual(self.studio.wait([c], timeout=30)[0]["status"], "cancelled")

    def test_upscale_is_the_pages(self):
        """POST /api/upscale is not part of the API for programs (the page's Upscale button uses it), like Cancel."""
        jid = self.generate()
        self.studio.wait([jid], timeout=30)
        self.assertIs(self.studio.state()["jobs"][0]["upscalable"]["ok"], True)
        code, body = self.studio.upscale(jid)
        self.assertEqual(code, 200, body)
        self.assertEqual(self.studio.wait([body["id"]], timeout=30)[0]["status"], "done")
        self.assert_off("GET", f"/api/jobs/{body['id']}")


class BadApiSetting(unittest.TestCase):
    def test_refused_at_start(self):
        from studio import StubStudio
        d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        try:
            for value in ("2", "on", "true", "off", " 1"):
                studio = StubStudio(d, env={"LFSTUDIO_API": value})
                with self.assertRaises(RuntimeError) as e:
                    studio.start(timeout=10)
                self.assertIn(f"LFSTUDIO_API={value}: must be 1 (on, the default) or 0 (off) (see bin/env.example)",
                              str(e.exception))
            studio = StubStudio(d, env={"LFSTUDIO_API": ""}).start()          # empty: the default
            try:
                self.assertIs(studio.state()["api"], True)
            finally:
                studio.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class BadMaxQueued(unittest.TestCase):
    def test_refused_at_start(self):
        from studio import StubStudio
        d = Path(tempfile.mkdtemp(prefix="lf-studio-test-"))
        try:
            for value in ("0", "x", "-3", "2.5"):
                studio = StubStudio(d, env={"LFSTUDIO_MAX_QUEUED": value})
                with self.assertRaises(RuntimeError) as e:
                    studio.start(timeout=10)
                self.assertIn("LFSTUDIO_MAX_QUEUED must be a whole number, 1 or more", str(e.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
