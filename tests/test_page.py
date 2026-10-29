"""The page (webui/index.html) in a real browser against the stub studio: no models, no GPU.

Needs Playwright with WebKit (pip install playwright && playwright install webkit) and ffmpeg; skipped otherwise.
python3 -m unittest tests.test_page -v      (or with the rest: python3 -m unittest discover -s tests)
"""
import base64
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio  # noqa: E402
from test_server import (BOTH, NOT_REF2VA, TURBO, finished_job, history, mp4, png,  # noqa: E402
                         release_server)

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipUnless(sync_playwright and shutil.which("ffmpeg"), "needs playwright (webkit) and ffmpeg")
class PageTest(unittest.TestCase):
    load_s, clip_s, preload = 2, 1, False
    upscale = True                     # the stub studio has the experimental Upscale on (OneRowUpscaleOff: off)
    models = None                      # LFSTUDIO_MODELS: the model choice off, as by default (ModelPage: on)

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.webkit.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-page-"))
        self.studio = StubStudio(self.dir, load_s=self.load_s, clip_s=self.clip_s, preload=self.preload,
                                 models=self.models)
        self.seed()
        self.studio.start()
        self.ctx = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.page = self.ctx.new_page()
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(self.studio.url + "/")
        self.page.wait_for_timeout(500)

    def tearDown(self):
        self.ctx.close()
        self.studio.stop()
        self.assertEqual(self.errors, [], "errors in the page")
        shutil.rmtree(self.dir, ignore_errors=True)

    def seed(self):
        pass

    def pick(self, name, value):
        self.page.click(f'input[name="{name}"][value="{value}"] + label')

    def go_text(self):
        return self.page.inner_text("#go")


class WarmUpLabel(PageTest):
    """#4: during a warm-up for the selected settings the button says when the clip would be ready. With
    LFSTUDIO_RESHAPE=0, so that another format counts as other settings (ReshapeLabel: the default)."""
    load_s, preload = 20, True

    def seed(self):
        self.studio.env["LFSTUDIO_RESHAPE"] = "0"
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([finished_job()]))
        (self.studio.state_dir / "load-times.json").write_text(
            '{"t2va/landscape/5": 20, "warm/t2va/landscape/5": 12}')        # takeover window 12 - 5 = 7 s

    def test_label_early_then_late_then_other_settings(self):
        s = self.studio.wait_for(lambda s: s["session"].get("preloading"))["session"]
        self.assertEqual((s["load_eta"], s["warm_s"]), (20, 12))
        self.page.wait_for_timeout(1200)                                     # next poll and render
        early = self.go_text()
        self.assertIn("replaces the warm-up", early)
        self.assertRegex(early, r"ready in ~25 s")                           # load + ~5 s to stop the warm-up
        self.assertNotIn("stops", early)
        time.sleep(max(0, s["takeover_until"] - time.time()) + 1.5)
        late = self.go_text()
        self.assertIn("after the warm-up", late)
        left = int(re.search(r"ready in ~(\d+) s", late).group(1))          # rest of the warm-up + a warm clip
        self.assertTrue(20 <= left <= 26, late)
        self.pick("orientation", "portrait")
        self.assertIn("stops the warm-up, loads the models first", self.go_text())
        self.pick("orientation", "landscape")
        self.assertIn("after the warm-up", self.go_text())


class Durations(PageTest):
    """#26: a duration is rounded to whole seconds before it is split into minutes and seconds: 2:00, never 1:60."""
    load_s = 30

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([
            finished_job("a" * 32, e2e_s=119.6), finished_job("b" * 32, e2e_s=179.5, includes_model_load=True),
            finished_job("c" * 32, e2e_s=89.6, cold_start_s=239.7)]))
        (self.studio.state_dir / "load-times.json").write_text('{"t2va/landscape/5": 119.6}')

    def test_formatters_at_the_boundaries(self):
        # fmtS (made in, model load, ETAs, ready in, last time) rounds; clock (running timers) counts up and floors
        want = {0: ("0 s", "0:00"), 0.4: ("0 s", "0:00"), 0.5: ("1 s", "0:00"), 59.4: ("59 s", "0:59"),
                59.5: ("60 s", "0:59"), 59.6: ("60 s", "0:59"), 89.4: ("89 s", "1:29"), 89.5: ("1:30 min", "1:29"),
                119.4: ("1:59 min", "1:59"), 119.5: ("2:00 min", "1:59"), 119.6: ("2:00 min", "1:59"),
                120: ("2:00 min", "2:00"), 3599.4: ("59:59 min", "59:59"), 3599.6: ("1:00:00", "59:59"),
                3600: ("1:00:00", "1:00:00"), 4300.4: ("1:11:40", "1:11:40"),     # an upscale: hours, as its timer
                -3: ("0 s", "0:00"), -0.4: ("0 s", "0:00")}
        got = self.page.evaluate(f"{json.dumps(list(want))}.map(s => [fmtS(s), clock(s)])")
        self.assertEqual(dict(zip(want, map(tuple, got))), want)
        # none, not a number or endless: 0, not "NaN" or "Infinity"
        odd = self.page.evaluate("[null, undefined, NaN, 'x', Infinity].map(s => [fmtS(s), clock(s)])")
        self.assertEqual(odd, [["0 s", "0:00"]] * 5)

    def test_cards(self):
        when = lambda j: self.page.inner_text(f'article[data-id="{j * 32}"] .when').split(" · ", 1)[1]
        self.page.wait_for_selector(f'article[data-id="{"c" * 32}"]')
        self.assertEqual([when(j) for j in "abc"], ["made in 2:00 min", "3:00 min incl. model load",
                                                    "made in 1:30 min · +4:00 min model load"])
        jid = self.studio.generate(seed=3)[1]["id"]                          # loads 30 s; the last load took 119.6 s
        card = f'article[data-id="{jid}"]'
        self.page.wait_for_selector(f'{card}[data-status="loading models + generating"]')
        self.page.wait_for_function(f"document.querySelector('{card} .ph2 .small').textContent.includes('last time')")
        self.assertEqual(self.page.inner_text(f"{card} .ph2 .small"), "Loading models, then your clip · last time 2:00 min")
        self.assertRegex(self.page.inner_text(f"{card} .ph2 .big"), r"^0:0\d$")


class RemixInputs(PageTest):
    """#1: Remix puts the clip's files back into the form; Generate reuses them without uploading."""

    def test_remix_fl2va_restores_frames(self):
        a = self.studio.generate(task="fl2va", prompt="frames", seed=5,
                                 files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png())]})[1]["id"]
        self.studio.wait([a])
        self.page.wait_for_timeout(2500)                                    # the card appears with the next poll
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        self.page.wait_for_function("document.querySelectorAll('#fl .thumbs img.kept').length === 2")
        self.assertTrue(self.page.is_visible("#fl"))
        self.assertEqual(self.page.inner_text("#err"), "")
        srcs = self.page.eval_on_selector_all("#fl .thumbs img.kept", "els => els.map(e => e.naturalWidth)")
        self.assertTrue(all(w > 0 for w in srcs), "thumbnails load from /input/")
        self.page.click("#go")
        jobs = self.studio.wait_for(lambda s: len(s["jobs"]) == 2)["jobs"]
        new = next(j for j in jobs if j["id"] != a)
        self.assertEqual([Path(new[k]).name for k in ("first_frame", "last_frame")], ["first.png", "last.png"])
        self.assertEqual(Path(new["first_frame"]).parent.name, new["id"])
        # x clears the zone: nothing kept, so the next Generate is refused as having no frame
        self.page.click("#fl .drop:first-of-type .clear")
        self.page.click("#fl .drops > .drop:last-of-type .clear")
        self.page.click("#go")
        self.page.wait_for_function("document.getElementById('err').textContent.includes('first frame')")

    def test_remix_ref2va_keeps_order_remove_one_and_add_one(self):
        a = self.studio.generate(task="ref2va", prompt="refs", seed=6,
                                 files={"ref_images": [("one.png", png()), ("two.png", png(40)), ("three.png", png(48))],
                                        "ref_videos": [("clip.mp4", mp4())]})[1]["id"]
        self.assertEqual(self.studio.wait([a])[0]["status"], "done")
        self.page.wait_for_timeout(2500)
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        self.page.wait_for_function("document.querySelectorAll('#ref .kept').length === 4")
        self.page.click('#ref .keptw:nth-child(2) .x')                        # remove two.png only
        self.assertEqual(self.page.eval_on_selector_all("#ref .kept", "els => els.map(e => e.title.split(' ')[0])"),
                         ["one.png", "three.png", "clip_ref.mp4"])
        self.page.set_input_files('input[name="ref_images"]', files=[{"name": "four.png", "mimeType": "image/png",
                                                                       "buffer": png(30)}])
        self.page.click("#go")
        jobs = self.studio.wait_for(lambda s: len(s["jobs"]) == 2)["jobs"]
        new = self.studio.wait([next(j["id"] for j in jobs if j["id"] != a)])[0]
        self.assertEqual([(r["type"], Path(r["path"]).name) for r in new["references"]],
                         [("image", "one.png"), ("image", "three.png"), ("image", "four.png"),
                          ("video", "clip_ref.mp4")])

    def test_remix_reports_missing_files(self):
        a = self.studio.generate(task="fl2va", prompt="frames", seed=5,
                                 files={"first_frame": [("first.png", png())], "last_frame": [("last.png", png())]})[1]["id"]
        job = self.studio.wait([a])[0]
        Path(job["last_frame"]).unlink()
        self.page.wait_for_timeout(2500)
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        self.page.wait_for_function("document.getElementById('err').textContent.includes('last.png')")
        self.assertEqual(self.page.eval_on_selector_all("#fl .thumbs img.kept", "els => els.length"), 1)

    def test_remix_t2va_clears_kept_files(self):
        a = self.studio.generate(task="fl2va", files={"first_frame": [("first.png", png())]})[1]["id"]
        b = self.studio.generate(task="t2va", seed=9)[1]["id"]
        self.studio.wait([a, b])
        self.page.wait_for_timeout(2500)
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        self.page.wait_for_function("document.querySelectorAll('#fl .thumbs img.kept').length === 1")
        self.page.click(f'article[data-id="{b}"] .reuse-all')
        self.page.wait_for_timeout(300)
        self.assertEqual(self.page.eval_on_selector_all(".thumbs .kept", "els => els.length"), 0)
        self.assertEqual(self.page.evaluate("remixFrom"), None)


class Paging(PageTest):
    """#5: the gallery shows the newest page and loads older clips as you scroll."""

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(history(60)))

    def cards(self):
        return self.page.eval_on_selector_all("#jobs .job", "els => els.length")

    def test_scroll_loads_older_pages(self):
        self.assertEqual(self.cards(), 24)
        self.assertEqual(self.page.inner_text("#count"), "60 clips")
        for want in (48, 60):
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.page.wait_for_function(f"document.querySelectorAll('#jobs .job').length === {want}")
        self.page.wait_for_function("document.getElementById('more').hidden")
        created = self.page.eval_on_selector_all("#jobs .job", "els => els.map(e => +e.dataset.created)")
        self.assertEqual(created, sorted(created, reverse=True))                 # newest first, no duplicates
        self.assertEqual(len(set(created)), 60)
        # a new clip goes on top, and nothing is lost or repeated
        new = self.studio.generate()[1]["id"]
        self.studio.wait([new])
        self.page.wait_for_function("document.querySelectorAll('#jobs .job').length === 61")
        self.page.wait_for_function("document.getElementById('count').textContent === '61 clips'")
        ids = self.page.eval_on_selector_all("#jobs .job", "els => els.map(e => e.dataset.id)")
        self.assertEqual((ids[0], len(set(ids))), (new, 61))

    def test_failing_older_pages_are_not_retried_in_a_loop(self):
        calls = []
        self.page.route("**/api/jobs*", lambda route: (calls.append(1), route.abort()))
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.page.wait_for_timeout(2000)
        self.assertLessEqual(len(calls), 2)
        self.assertTrue(self.page.is_visible("#more"))                         # still there to try again


class MissedPolls(PageTest):
    """#5: after a gap in polling, clips that finished meanwhile are updated and the ones in between appear."""
    load_s, clip_s = 1, 0.1

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(history(30)))

    def test_gap_is_filled(self):
        first = self.studio.generate(seed=1)[1]["id"]
        self.page.wait_for_selector(f'article[data-id="{first}"]')
        self.page.route("**/api/state", lambda route: route.abort())          # the tab sleeps
        ids = [first] + [self.studio.generate(seed=i)[1]["id"] for i in range(2, 32)]
        self.studio.wait(ids, timeout=120)
        self.page.unroute("**/api/state")
        self.page.wait_for_function("document.querySelectorAll('#jobs .job').length >= 55")
        self.page.wait_for_function(f"document.querySelector('article[data-id=\"{first}\"]').dataset.status === 'done'")
        shown = set(self.page.eval_on_selector_all("#jobs .job", "els => els.map(e => e.dataset.id)"))
        self.assertTrue(set(ids) <= shown, len(set(ids) - shown))


class DeleteUndo(PageTest):
    """#2: Delete removes the card at once; Undo in the toast brings it back with its video."""

    def test_delete_then_undo(self):
        a = self.studio.generate(prompt="to delete")[1]["id"]
        self.studio.wait([a])
        card = f'article[data-id="{a}"]'
        self.page.wait_for_selector(f"{card} video")
        self.assertIn("trash", self.page.get_attribute(f"{card} .del", "title"))
        self.page.click(f"{card} .del")
        self.page.wait_for_selector(card, state="detached")
        self.page.wait_for_selector("#undo", state="visible")
        self.page.wait_for_timeout(2500)                                    # a poll must not bring it back
        self.assertEqual(self.page.query_selector(card), None)
        self.assertEqual(self.studio.state()["jobs"], [])
        self.page.click("#undo button")
        self.page.wait_for_selector(f"{card} video")
        self.page.wait_for_function("document.getElementById('count').textContent === '1 clip'")
        self.assertEqual(self.studio.get(f"/video/{a}")[0], 200)

    def test_stale_poll_and_other_tabs(self):
        a, b = (self.studio.generate(prompt=f"clip {i}", seed=i)[1]["id"] for i in (1, 2))
        self.studio.wait([a, b])
        self.page.wait_for_selector(f'article[data-id="{b}"] video')
        held = []
        # polls answered by the server before the delete (still listing a) reach the page only after it
        self.page.route("**/api/state", lambda route: held.append((route, route.fetch())))
        self.page.wait_for_timeout(2500)
        self.page.click(f'article[data-id="{a}"] .del')
        self.page.wait_for_selector(f'article[data-id="{a}"]', state="detached")
        self.assertTrue(held and any(a.encode() in r.body() for _, r in held))
        for route, response in held:
            route.fulfill(response=response)
        self.page.unroute("**/api/state")
        self.page.wait_for_timeout(2500)
        self.assertIsNone(self.page.query_selector(f'article[data-id="{a}"]'))
        other = self.ctx.new_page()                                            # another tab
        other.goto(self.studio.url + "/")
        other.wait_for_selector(f'article[data-id="{b}"] video')
        self.page.click(f'article[data-id="{b}"] .del')
        other.wait_for_selector(f'article[data-id="{b}"]', state="detached")
        self.page.wait_for_selector("#nojobs", state="visible")               # nothing left: the empty message
        self.assertIn("2 clips deleted", self.page.inner_text("#undo"))
        self.page.click("#undo button")
        self.page.wait_for_selector(f'article[data-id="{a}"] video')
        self.page.wait_for_selector(f'article[data-id="{b}"] video')
        other.wait_for_selector(f'article[data-id="{b}"]')                    # back in the other tab too
        self.studio.post("/api/delete", json_body={"id": a})                  # deleted by hand (curl) ...
        self.page.wait_for_selector(f'article[data-id="{a}"]', state="detached")
        self.studio.post("/api/undelete", json_body={"id": a})                # ... and restored by hand
        self.page.wait_for_selector(f'article[data-id="{a}"]')

    def test_no_delete_while_running(self):
        a = self.studio.generate(prompt="running", seconds=8)[1]["id"]
        self.page.wait_for_selector(f'article[data-id="{a}"]')
        self.assertFalse(self.page.is_visible(f'article[data-id="{a}"] .del'))
        self.studio.wait([a])
        self.page.wait_for_selector(f'article[data-id="{a}"] .del', state="visible")


class LoadSelectedNow(PageTest):
    """#3: the link under Generate loads the selected settings on request. With LFSTUDIO_RESHAPE=0: by default another
    format or length of the loaded mode keeps the models, and the page offers no link for it (ReshapeLabel)."""
    load_s, preload = 6, True

    def seed(self):
        self.studio.env.update(STUB_CLOSE_S="2", LFSTUDIO_RESHAPE="0")
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([finished_job()]))

    def test_link_switches_the_warm_up(self):
        self.studio.wait_for(lambda s: s["session"].get("preloading"))
        self.page.wait_for_timeout(1200)
        self.assertFalse(self.page.is_visible("#reloadnote"))               # selection = what's warming up
        self.pick("orientation", "portrait")
        self.page.wait_for_selector("#loadnow", state="visible")
        self.page.wait_for_timeout(2500)                                     # a selection change alone loads nothing
        self.assertEqual(self.studio.state()["session"]["loading_key"], ["t2va", "landscape", 5])
        self.assertNotIn("load requested", self.studio.log.read_text())
        self.assertIn("Warming up: Text · 16:9 · 5 s. Selected: Text · 9:16 · 5 s.", self.page.inner_text("#reloadnote"))
        self.page.click("#loadnow")
        self.page.wait_for_function("document.getElementById('loaded').textContent.includes('warming up Text · 9:16')")
        self.page.wait_for_selector("#reloadnote", state="hidden")
        self.assertIn("ready in", self.go_text())                            # the warm-up label (#4) applies

    def test_switch_shows_in_the_status(self):
        self.studio.wait_for(lambda s: s["session"]["state"] == "ready", timeout=30)
        self.pick("seconds", "6")
        self.page.click("#loadnow")
        self.page.wait_for_function("document.getElementById('loaded').textContent.startsWith('switching to Text · 16:9 · 6 s')")
        self.assertFalse(self.page.is_visible("#loadnow"))
        self.assertIn("loads these models first", self.go_text())
        self.page.wait_for_function("document.getElementById('loaded').textContent.includes('warming up')")

    def test_no_link_while_a_clip_runs(self):
        self.studio.wait_for(lambda s: s["session"]["state"] == "ready", timeout=30)
        self.studio.generate(seconds=8)
        self.pick("seconds", "10")
        self.page.wait_for_timeout(2500)
        self.assertIn("after the queue", self.page.inner_text("#reloadnote"))
        self.assertFalse(self.page.is_visible("#loadnow"))


class ReshapeLabel(PageTest):
    """#45: another format or length of the loaded mode keeps the models: the button says so, without a reload note
    or a "Load selected now" link; another mode still reloads. LFSTUDIO_RESHAPE=0 (ReshapeLabelOff) reloads for both."""
    reshape = True

    def seed(self):
        self.studio.env.update(LFSTUDIO_RESHAPE="1" if self.reshape else "0", STUB_RESHAPE_S="3")

    def test_labels(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid])
        self.studio.wait_for(lambda s: s["session"]["state"] == "ready")
        self.page.wait_for_timeout(1200)                                     # next poll and render
        self.pick("orientation", "portrait")
        self.pick("seconds", "10")
        if self.reshape:
            self.assertIn("same models · a bit over", self.go_text())     # its first clip compiles
            self.assertNotIn("load", self.go_text())
            self.assertFalse(self.page.is_visible("#reloadnote"))
            self.assertNotIn("reload", self.page.get_attribute("#go", "class") or "")
            self.assertIn("keeps the models loaded", self.page.inner_text(".foot"))
        else:
            self.assertIn("reloads the models first", self.go_text())
            self.assertTrue(self.page.is_visible("#loadnow"))
            self.assertIn("Changing any of them reloads", self.page.inner_text(".foot"))
        self.pick("task", "fl2va")                                           # another mode reloads either way
        self.assertIn("reloads the models first", self.go_text())
        if self.reshape:                                                     # a requested change of length/format
            self.studio.post("/api/load", json_body={"task": "t2va", "orientation": "portrait", "seconds": 10})
            self.page.wait_for_function(
                "document.getElementById('loaded').textContent.startsWith('changing to Text · 9:16 · 10 s')")


class ReshapeLabelOff(ReshapeLabel):
    reshape = False


UI_TEST = Path(__file__).resolve().parent / "ui_test.py"


@unittest.skipUnless(sync_playwright and shutil.which("ffmpeg"), "needs playwright (webkit) and ffmpeg")
class UiTestScript(unittest.TestCase):
    """tests/ui_test.py, the general check of the page that AGENTS.md runs against any studio, passes against a stub
    studio with a t2va session loaded, with LFSTUDIO_RESHAPE=1 (the default since 0.4: another format or length keeps
    the models, another mode reloads them) and with 0 (any change reloads), and with the API for programs off
    (LFSTUDIO_API=0, #58), which the page does not use. An fl2va clip before it makes fl2va a mode 'Load selected now'
    can warm up."""

    def run_ui_test(self, reshape, api="1", models=None):
        root = Path(tempfile.mkdtemp(prefix="lf-studio-uitest-"))
        self.addCleanup(shutil.rmtree, root, True)
        studio = StubStudio(root, load_s=1, clip_s=0.5, preload=False,
                            env={"LFSTUDIO_RESHAPE": reshape, "LFSTUDIO_API": api}, models=models)
        studio.start()
        self.addCleanup(studio.stop)
        first = [studio.turbo(seed=3)[1]["id"]] if models else []      # a LightX2V Turbo clip before the session
        ids = first + [studio.generate(task="fl2va", files={"first_frame": [("first.png", png())]})[1]["id"],
                       studio.generate()[1]["id"]]
        self.assertEqual([job["status"] for job in studio.wait(ids)], ["done"] * len(ids))
        state = studio.state()
        self.assertEqual((state["reshape"], state["session"]["key"], sorted(state["warmable"]), state["api"]),
                         (reshape == "1", ["t2va", "landscape", 5], ["fl2va", "t2va"], api == "1"))
        result = subprocess.run([sys.executable, str(UI_TEST), studio.url + "/", "--shots", str(root / "shots")],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        self.assertIn("ALL PASS", result.stdout)
        return result.stdout

    def test_reshape_on(self):
        out = self.run_ui_test("1")
        for check in ("SKIP model choice checks (one model on this studio)",
                      "PASS Generate keeps the models when the format differs",
                      "PASS Generate keeps the models when the length differs",
                      "PASS Generate warns about a reload when the mode differs"):
            self.assertIn(check, out)

    def test_reshape_off(self):
        self.assertIn("PASS Generate warns about a reload when the format differs", self.run_ui_test("0"))

    def test_api_off(self):
        self.assertIn("PASS Generate keeps the models when the format differs", self.run_ui_test("1", api="0"))

    def test_with_the_model_choice(self):
        """LightX2V Turbo on (LFSTUDIO_MODELS), with one of its clips in the gallery before the two Sol-H3 clips: the
        Model row's checks, its clip's tag and its Upscale button's reason, and the rest as before (Sol-H3, the
        default, is picked in a new browser)."""
        out = self.run_ui_test("1", models=BOTH)
        for check in (f"PASS fields unchanged after model -> {TURBO}", "PASS fields unchanged after model -> sol-h3",
                      f"PASS model {TURBO}: what it doesn't make yet is greyed out with the reason (1)",   # Reference
                      "PASS a disabled Upscale button says why (2 disabled)",     # its clip and the fl2va one
                      "PASS clips of another model show it in a tag (1: ['LightX2V Turbo 576p'])",
                      "PASS Generate keeps the models when the format differs"):
            self.assertIn(check, out)

    def test_with_an_upscale(self):
        """An upscale of the t2va clip in the gallery (the experimental Upscale, which the stub studio has on). It
        closed the models, so the Generate checks run without a session."""
        root = Path(tempfile.mkdtemp(prefix="lf-studio-uitest-"))
        self.addCleanup(shutil.rmtree, root, True)
        studio = StubStudio(root, load_s=1, clip_s=0.5, preload=False)
        studio.start()
        self.addCleanup(studio.stop)
        jid = studio.generate()[1]["id"]
        studio.wait([jid])
        uid = studio.upscale(jid)[1]["id"]
        self.assertEqual(studio.wait([uid])[0]["status"], "done")
        result = subprocess.run([sys.executable, str(UI_TEST), studio.url + "/", "--shots", str(root / "shots")],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        self.assertIn("PASS Upscale only on finished clips with it on (on), no Remix or Upscale on an upscale "
                      "(1 upscales)", result.stdout)


class ReleaseReviewPage(PageTest):
    """From the whole-release review: Undo vs a poll that still lists the clip as deleted, clips restored elsewhere
    from an older page, and the keyboard."""

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(history(30)))

    def test_undo_survives_an_older_deleted_list(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid])
        card = f'article[data-id="{jid}"]'
        self.page.wait_for_selector(card)
        self.page.click(f"{card} .del")
        self.page.wait_for_timeout(2500)                                     # a poll lists it as deleted
        held = []
        self.page.route("**/api/state", lambda route: held.append((route, route.fetch())))
        self.page.wait_for_timeout(2200)                                     # one more, answered before the Undo
        self.page.click("#undo button")
        self.page.wait_for_selector(card)
        for route, response in held:
            route.fulfill(response=response)
        self.page.unroute("**/api/state")
        self.page.wait_for_timeout(2500)
        self.assertIsNotNone(self.page.query_selector(card))

    def test_restored_by_hand_from_an_older_page(self):
        old = f"{3:032x}"
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.page.wait_for_selector(f'article[data-id="{old}"]')
        self.studio.post("/api/delete", json_body={"id": old})
        self.page.wait_for_selector(f'article[data-id="{old}"]', state="detached")
        self.studio.post("/api/undelete", json_body={"id": old})
        self.page.wait_for_selector(f'article[data-id="{old}"]')

    def test_deleted_here_then_restored_by_hand(self):
        old = f"{3:032x}"
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.page.wait_for_selector(f'article[data-id="{old}"]')
        self.page.click(f'article[data-id="{old}"] .del')
        self.page.wait_for_selector(f'article[data-id="{old}"]', state="detached")
        self.page.wait_for_timeout(2500)
        self.studio.post("/api/undelete", json_body={"id": old})
        self.page.wait_for_selector(f'article[data-id="{old}"]')

    def test_failed_undo_offers_retry(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid])
        card = f'article[data-id="{jid}"]'
        self.page.wait_for_selector(card)
        self.page.click(f"{card} .del")
        self.page.route("**/api/undelete", lambda route: route.fulfill(status=500, json={"error": "share not there"}))
        self.page.click("#undo button")
        self.page.wait_for_function("document.querySelector('#undo button').textContent === 'Retry'")
        self.assertIn("share not there", self.page.inner_text("#undo"))
        self.page.unroute("**/api/undelete")
        self.page.click("#undo button")
        self.page.wait_for_selector(card)

    def test_keyboard(self):
        chooser = []
        self.page.on("filechooser", lambda fc: chooser.append(fc))
        a = self.studio.generate(task="fl2va", files={"first_frame": [("f.png", png())]})[1]["id"]
        self.studio.wait([a])
        self.page.wait_for_selector(f'article[data-id="{a}"] .reuse-all')
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        self.page.wait_for_selector("#fl .keptw .x")
        self.page.focus("#fl .keptw .x")
        self.page.keyboard.press("Enter")                                    # removes the file, no file picker
        self.page.wait_for_timeout(500)
        self.assertEqual((self.page.query_selector("#fl .keptw"), chooser), (None, []))
        self.page.focus(f'article[data-id="{a}"] .del')
        self.page.keyboard.press("Enter")
        self.page.wait_for_selector("#undo", state="visible")
        self.assertEqual(self.page.evaluate("document.activeElement === document.querySelector('#undo button')"), True)
        self.page.wait_for_selector("#undo", state="hidden", timeout=15000)   # toast gone: focus back in the gallery
        self.assertEqual(self.page.evaluate("!!document.activeElement.closest('article.job')"), True)


class OldLengths(PageTest):
    """3 and 4 s went after 0.5.0: the page offers 5-15 s, a length kept from 0.5.0 reads 5 s, and clips of 3 and 4 s
    made before still list and play, and Remix brings them back at 5 s without an error. Their Upscale button is
    disabled with the reason (5 s clips only)."""

    def seed(self):
        old = self.dir / "old"
        old.mkdir()
        (old / "3.mp4").write_bytes(mp4(3))
        (old / "4.mp4").write_bytes(mp4(4))
        (old / "first.png").write_bytes(png())
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([
            finished_job("3" * 32, seconds=3, output=str(old / "3.mp4"), created=1, prompt="three seconds"),
            finished_job("4" * 32, task="fl2va", seconds=4, output=str(old / "4.mp4"), created=2,
                         first_frame=str(old / "first.png"), prompt="four seconds")]))

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def test_lengths_offered(self):
        offered = self.page.eval_on_selector_all('input[name="seconds"]', "els => els.map(e => e.value)")
        self.assertEqual(offered, ["5", "6", "7", "8", "10", "12", "15"])
        self.assertEqual(self.page.eval_on_selector_all(".len label", "els => els.map(e => e.textContent)"), offered)
        self.page.evaluate("localStorage.setItem('lfstudio.seconds', '3')")       # kept by a 0.5.0 page
        other = self.ctx.new_page()                                           # (a reload here would cancel a poll)
        other.goto(self.studio.url + "/")
        self.assertEqual(other.evaluate("val('seconds')"), "5")
        self.assertEqual(other.inner_text("#lenout"), "~60 s per clip")
        other.close()

    def test_old_clips_list_play_and_remix_at_5_s(self):
        for jid, seconds in (("3" * 32, 3), ("4" * 32, 4)):
            card = self.card(jid)
            self.page.wait_for_selector(f"{card} video")
            self.assertEqual(self.page.eval_on_selector_all(f"{card} .tags .tag", "els => els.map(e => e.textContent)")[2],
                             f"{seconds} s")
            self.assertEqual(self.page.get_attribute(f"{card} video", "src"), f"/video/{jid}")
            self.assertIn(f"_{seconds}s_", self.page.get_attribute(f"{card} a.act", "download"))
            self.assertEqual(self.page.evaluate("id => fetch('/video/' + id).then(r => r.status)", jid), 200)
        up = f"{self.card('3' * 32)} .act.up"                                   # the Upscale takes 5 s clips only
        self.page.wait_for_selector(f'{up}[aria-disabled="true"]', state="attached")
        self.assertEqual(self.page.get_attribute(up, "title"),
                         "Upscale takes 5 s clips only, for now (NVIDIA's pipeline is fixed at 121 frames)")
        self.pick("seconds", "10")
        self.page.click(f"{self.card('3' * 32)} .reuse-all")                   # t2va, 3 s
        self.page.wait_for_function("document.getElementById('toast').textContent"
                                    " === 'Settings loaded at 5 s (3 s is no longer offered)'")
        self.assertEqual(self.page.evaluate("[val('task'), val('seconds'), store.get('seconds')]"), ["t2va", "5", "5"])
        self.assertEqual(self.page.input_value("#prompt"), "three seconds")
        self.assertEqual(self.page.inner_text("#err"), "")
        self.page.click(f"{self.card('4' * 32)} .reuse-all")                   # fl2va, 4 s, with its first frame
        self.page.wait_for_function("document.querySelectorAll('#fl .thumbs img.kept').length === 1")
        self.page.wait_for_function("document.getElementById('toast').textContent"
                                    " === 'Settings and files loaded at 5 s (4 s is no longer offered)'")
        self.assertEqual(self.page.evaluate("[val('task'), val('seconds')]"), ["fl2va", "5"])
        self.assertEqual(self.page.inner_text("#err"), "")
        self.page.click("#go")
        jobs = self.studio.wait_for(lambda s: len(s["jobs"]) == 3)["jobs"]
        new = self.studio.wait([next(j["id"] for j in jobs if j["id"] not in ("3" * 32, "4" * 32))])[0]
        self.assertEqual((new["status"], new["task"], new["seconds"], new["prompt"]), ("done", "fl2va", 5, "four seconds"))
        self.assertEqual(Path(new["first_frame"]).name, "first.png")
        self.assertEqual(self.page.inner_text("#err"), "")


class RemixSeed(PageTest):
    """#14: Remix loads the clip's seed and leaves "New random seed every run" as it is."""

    def remix(self, seed):
        a = self.studio.generate(prompt="the source clip", seed=seed)[1]["id"]
        self.studio.wait([a])
        self.page.wait_for_selector(f'article[data-id="{a}"] .reuse-all')
        self.page.click(f'article[data-id="{a}"] .reuse-all')
        return a

    def generate_from_page(self, a):
        self.page.click("#go")
        jobs = self.studio.wait_for(lambda s: len(s["jobs"]) == 2)["jobs"]
        return next(j for j in jobs if j["id"] != a)

    def test_ticked_stays_ticked_and_draws_a_new_seed(self):
        self.page.check("#autoroll")
        a = self.remix(777)
        self.page.wait_for_function("document.getElementById('toast').textContent === 'Settings loaded'")  # 5 s: as is
        self.assertTrue(self.page.is_checked("#autoroll"))
        self.assertEqual(self.page.input_value("#seed"), "777")
        new = self.generate_from_page(a)                                    # Remix, then Generate in the same tab
        self.assertNotEqual(int(new["seed"]), 777)
        self.assertEqual(new["prompt"], "the source clip")
        other = self.ctx.new_page()                                         # remembered as ticked: a new tab
        other.goto(self.studio.url + "/")                                   # (a reload here would cancel a poll)
        self.assertTrue(other.is_checked("#autoroll"))
        other.close()

    def test_unticked_uses_the_clips_seed(self):
        self.page.fill("#seed", "5")
        a = self.remix(777)
        self.assertFalse(self.page.is_checked("#autoroll"))
        self.assertEqual(self.page.input_value("#seed"), "777")
        self.assertEqual(int(self.generate_from_page(a)["seed"]), 777)


class RemixScroll(PageTest):
    """#25: beside the gallery (a sticky column) the form is in view, so Remix leaves the page where it is; in one
    column (the form above the gallery) Remix brings the form's top into view, and a keyboard press moves focus there."""

    def seed(self):
        jobs = history(30)                  # cards are newest first: card n is jobs[29 - n]
        jobs[21].update(task="fl2va", first_frame="gone/first.png", last_frame="gone/last.png")    # card 8
        # card 11: Reference, with its files; three wide pictures take two rows in the zone on a phone
        inputs = self.studio.runtime / "inputs" / "webui" / jobs[18]["id"]
        inputs.mkdir(parents=True)
        files = [(f"{c}.png", picture(c, "320x180"), "image") for c in ("red", "green", "blue")]
        files += [("clip.mp4", mp4(), "video"), ("voice.wav", b"RIFF0000WAVEfmt ", "audio")]
        for name, data, _ in files:
            (inputs / name).write_bytes(data)
        jobs[18].update(task="ref2va", references=[{"type": kind, "path": str(inputs / name)} for name, _, kind in files])
        (self.studio.state_dir / "jobs.json").write_text(json.dumps(jobs))

    def one_column(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.wait_for_timeout(300)
        self.assertEqual(self.page.evaluate("getComputedStyle(document.getElementById('f')).position"), "static")

    def remix_button(self, n):
        """Scroll the nth card to the middle of the screen; its Remix button and the clip's seed."""
        self.page.evaluate(f"document.querySelectorAll('#jobs .job')[{n}].scrollIntoView({{block: 'center'}})")
        self.page.wait_for_timeout(300)
        card = self.page.locator("#jobs .job").nth(n)
        return card.locator(".reuse-all"), card.locator(".seedv").inner_text()

    def settle(self):
        """scrollY once a (smooth) scroll has stopped."""
        self.page.wait_for_timeout(300)
        last = None
        for _ in range(30):
            y = self.page.evaluate("scrollY")
            if y == last:
                return y
            last = y
            self.page.wait_for_timeout(150)
        return y

    def focused(self, selector):
        return self.page.evaluate(f"document.activeElement === document.querySelector('{selector}')")

    def form_in_view(self):
        """The form's top is at the top of the screen (14 px, as when sticky) and Generate is on screen."""
        top, go, height = self.page.evaluate("""() => [document.getElementById('f').getBoundingClientRect().top,
            document.getElementById('go').getBoundingClientRect().bottom, innerHeight]""")
        self.assertAlmostEqual(top, 14, delta=2)
        self.assertLessEqual(go, height, "Generate on screen")

    def test_two_columns_the_page_stays(self):
        self.assertEqual(self.page.evaluate("getComputedStyle(document.getElementById('f')).position"), "sticky")
        button, seed = self.remix_button(12)
        y = self.settle()
        self.assertGreater(y, 1000)
        button.click()
        self.assertEqual(self.settle(), y, "Remix scrolled the page")
        self.assertEqual(self.page.input_value("#seed"), seed)
        # keyboard: the page stays and focus stays on the card's Remix
        button, seed = self.remix_button(16)
        y = self.settle()
        button.focus()
        self.page.keyboard.press("Enter")
        self.assertEqual(self.settle(), y, "Remix scrolled the page")
        self.assertEqual(self.page.input_value("#seed"), seed)
        self.assertTrue(self.page.evaluate("document.activeElement.classList.contains('reuse-all')"))

    def test_one_column_brings_the_form_into_view(self):
        self.one_column()
        button, seed = self.remix_button(6)
        self.assertGreater(self.settle(), 1500)
        button.click()
        self.settle()
        self.form_in_view()
        self.assertEqual(self.page.input_value("#seed"), seed)
        self.assertFalse(self.focused("#prompt"), "a tap does not focus the prompt (no on-screen keyboard)")
        # keyboard: focus goes to the prompt with the view, so the next Tab does not jump back to the gallery
        button, seed = self.remix_button(9)
        button.focus()
        self.page.keyboard.press("Enter")
        self.settle()
        self.form_in_view()
        self.assertEqual(self.page.input_value("#seed"), seed)
        self.assertTrue(self.focused("#prompt"))

    def test_one_column_the_form_changes_height(self):
        """Remix changes the form's height above the gallery: it clears an error, and a Keyframes card shows the drop
        zones and then (its files are gone) an error. WebKit's scroll anchoring kept the card in place, which
        cancelled the scroll (the form's top ended thousands of px above the screen)."""
        self.one_column()
        button, seed = self.remix_button(7)
        self.page.evaluate("document.getElementById('err').textContent = 'Could not reach the server'")
        self.assertGreater(self.settle(), 1500)
        button.click()
        self.settle()
        self.form_in_view()
        self.assertEqual((self.page.input_value("#seed"), self.page.inner_text("#err")), (seed, ""))
        # keyboard, Keyframes card: the drop zones appear at once, the error when /api/inputs answers
        button, seed = self.remix_button(8)
        self.assertTrue(self.page.is_hidden("#fl"))
        self.assertGreater(self.settle(), 1500)
        button.focus()
        self.page.keyboard.press("Enter")
        self.page.wait_for_function("document.getElementById('err').textContent.includes('Gone from the server')")
        self.settle()
        self.assertTrue(self.page.is_visible("#fl"))
        top, ptop, pbottom, height = self.page.evaluate("""() => [document.getElementById('f').getBoundingClientRect().top,
            ...['top', 'bottom'].map(k => document.getElementById('prompt').getBoundingClientRect()[k]), innerHeight]""")
        self.assertAlmostEqual(top, 14, delta=2)
        self.assertTrue(self.focused("#prompt"))
        self.assertTrue(0 <= ptop and pbottom <= height, "the focused prompt is on screen")
        self.assertEqual(self.page.input_value("#seed"), seed)

    def prompt_on_screen(self):
        """The prompt is focused and wholly on screen."""
        ptop, pbottom, height = self.page.evaluate("""() => [...['top', 'bottom'].map(k =>
            document.getElementById('prompt').getBoundingClientRect()[k]), innerHeight]""")
        self.assertTrue(self.focused("#prompt"))
        self.assertTrue(0 <= ptop and pbottom <= height, f"the focused prompt is on screen: {ptop}-{pbottom} of {height}")

    def test_one_column_keyboard_scrolls_on_to_the_prompt(self):
        """Keyframes and Reference show drop zones above the prompt (Reference pushes it ~350 px down): a keyboard Remix
        scrolls on until the focused prompt is on screen, also when the clip's pictures then take a second row. A tap
        still brings the form's top into view."""
        self.one_column()
        button, seed = self.remix_button(11)                                   # Reference, with its files
        self.assertGreater(self.settle(), 1500)
        button.focus()
        self.page.keyboard.press("Enter")
        self.page.wait_for_function("""() => [...document.querySelectorAll('#ref .thumbs img.kept')].filter(i =>
            i.complete && i.naturalWidth).length === 3""")
        self.settle()
        self.assertGreater(self.page.eval_on_selector("#ref .drop", "e => e.offsetHeight"), 140, "two rows of pictures")
        self.assertEqual(self.page.input_value("#seed"), seed)
        self.prompt_on_screen()
        top = self.page.evaluate("document.getElementById('f').getBoundingClientRect().top")
        self.assertLess(top, -100, "scrolled past the form's top")
        # a tap: the form's top, the prompt not focused
        self.page.evaluate("document.activeElement.blur()")
        button, seed = self.remix_button(11)
        self.assertGreater(self.settle(), 1500)
        button.click()
        self.settle()
        self.assertAlmostEqual(self.page.evaluate("document.getElementById('f').getBoundingClientRect().top"), 14, delta=2)
        self.assertFalse(self.focused("#prompt"))
        # Keyframes on a shorter screen (about what Safari leaves of a 390x844 phone)
        self.page.set_viewport_size({"width": 390, "height": 664})
        button, seed = self.remix_button(8)
        self.assertGreater(self.settle(), 1500)
        self.page.evaluate("document.getElementById('err').textContent = ''")
        button.focus()
        self.page.keyboard.press("Enter")
        self.page.wait_for_function("document.getElementById('err').textContent.includes('Gone from the server')")
        self.settle()
        self.assertTrue(self.page.is_visible("#fl"))
        self.assertEqual(self.page.input_value("#seed"), seed)
        self.prompt_on_screen()

    def test_one_column_reduced_motion_jumps(self):
        self.page.emulate_media(reduced_motion="reduce")
        self.one_column()
        self.remix_button(6)
        top = self.page.evaluate("""() => { document.querySelectorAll('#jobs .job')[6].querySelector('.reuse-all').click();
            return document.getElementById('f').getBoundingClientRect().top; }""")     # at once, not smoothly
        self.assertAlmostEqual(top, 14, delta=2)


class SeedText(PageTest):
    """#15: the seed is plain text next to the time; Remix, Download, Upscale and Delete share one row (a 10 s clip:
    its Upscale shows, disabled; the stub studio has the experimental Upscale on)."""

    def test_seed_in_the_info_line_and_one_row_of_buttons(self):
        ok = self.studio.generate(prompt="a long prompt " * 30, seed=2034940401, seconds=10)[1]["id"]
        bad = self.studio.generate(prompt="STUB_FAIL", seed=2147483646)[1]["id"]
        self.studio.wait([ok, bad])           # the page was open: the cards were built while queued, then updated
        self.page.wait_for_selector(f'article[data-id="{ok}"] video')
        self.page.wait_for_selector(f'article[data-id="{bad}"][data-status="failed"]')
        for width in (1280, 390, 320):        # 320: the narrowest phone card (266 px, #19)
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.wait_for_timeout(300)
            up = ["Upscale"] if self.upscale else []
            for jid, seed, buttons in ((ok, "2034940401", ["Remix", "Download", *up, "Delete"]),
                                       (bad, "2147483646", ["Remix", "Delete"])):
                card = f'article[data-id="{jid}"]'
                self.assertTrue(self.page.inner_text(f"{card} .when"))
                self.assertTrue(self.page.inner_text(f"{card} .info").endswith(f" · seed {seed}"))
                self.assertIsNone(self.page.query_selector(f"{card} .reuse-seed"))
                row = self.page.eval_on_selector_all(
                    f"{card} .acts > *", "els => els.filter(e => e.offsetWidth).map(e => "
                                         "[e.textContent.trim(), Math.round(e.getBoundingClientRect().top)])")
                self.assertEqual([name for name, _ in row], buttons, width)
                self.assertEqual(len({top for _, top in row}), 1, f"one row at {width} px: {row}")

    def selected(self):
        return self.page.evaluate("getSelection().toString()")

    def finished(self, **kw):
        a = self.studio.generate(seed=2034940401, **kw)[1]["id"]
        self.studio.wait([a])
        card = f'article[data-id="{a}"]'
        self.page.wait_for_selector(f"{card} video")
        return card

    def wait_selected(self, text):
        self.page.wait_for_function("t => getSelection().toString() === t", arg=text, timeout=3000)

    def test_a_click_or_tab_selects_the_seed_and_polls_keep_it(self):
        card = self.finished()
        self.page.evaluate(f"window.seedNode = document.querySelector('{card} .seedv').firstChild")
        self.page.click(f"{card} .seedv")
        self.wait_selected("2034940401")
        self.assertFalse(self.page.evaluate("document.activeElement.classList.contains('seedv')"))  # mouse: no focus
        self.page.wait_for_timeout(2500)                                    # a poll and a render later
        self.assertEqual(self.selected(), "2034940401")
        self.assertTrue(self.page.evaluate(f"document.querySelector('{card} .seedv').firstChild === window.seedNode"))
        self.page.evaluate("getSelection().removeAllRanges()")
        self.page.focus(f"{card} .reuse-all")
        self.page.keyboard.press("Shift+Tab")                               # keyboard: the seed is one Tab stop
        self.assertTrue(self.page.evaluate("document.activeElement.classList.contains('seedv')"))
        self.wait_selected("2034940401")

    def select_prompt_to_seed(self, card):
        return self.page.evaluate(f"""() => {{ const c = document.querySelector('{card}'), r = document.createRange();
            r.setStart(c.querySelector('.p').firstChild, 0); r.setEnd(c.querySelector('.seedv').firstChild, 10);
            getSelection().removeAllRanges(); getSelection().addRange(r); return getSelection().toString(); }}""")

    def test_a_selection_of_your_own_stays(self):
        card = self.finished(prompt="a red fox runs through snow")
        whole = self.select_prompt_to_seed(card)
        self.assertTrue(whole.startswith("a red fox") and whole.endswith("seed 2034940401"), whole)
        self.page.click(f"{card} .seedv", button="right")                  # context menu: Copy copies all of it
        self.page.keyboard.press("Escape")
        self.page.wait_for_timeout(300)
        self.assertEqual(self.selected(), whole)
        self.page.focus(f"{card} .reuse-all")
        self.page.keyboard.press("Shift+Tab")                               # the same with the seed focused
        self.assertEqual(self.select_prompt_to_seed(card), whole)
        self.page.click(f"{card} .seedv", button="right")
        self.page.keyboard.press("Escape")
        self.page.wait_for_timeout(300)
        self.assertEqual(self.selected(), whole)
        self.page.evaluate(f"""() => {{ const r = document.createRange();         // a caret at the prompt's start
            r.setStart(document.querySelector('{card} .p').firstChild, 0); r.collapse(true);
            getSelection().removeAllRanges(); getSelection().addRange(r); }}""")
        self.page.click(f"{card} .seedv", modifiers=["Shift"])             # Shift+click extends to the seed
        self.wait_selected(whole)

    def test_the_mouse_leaves_focus_and_dragging_to_the_browser(self):
        card = self.finished()
        other = self.studio.generate(prompt="another clip")[1]["id"]
        self.studio.wait([other])
        self.page.wait_for_selector(f'article[data-id="{other}"] video')
        cards = self.page.eval_on_selector_all(".job", "els => els.length")
        self.page.focus(f'article[data-id="{other}"] .del')
        self.page.click(f"{card} .seedv")                                   # focus leaves Delete, as on any text
        self.page.keyboard.press("Space")
        self.page.wait_for_timeout(500)
        self.assertEqual(self.page.eval_on_selector_all(".job", "els => els.length"), cards)
        self.assertFalse(self.page.is_visible("#undo"))
        self.page.fill("#seed", "")
        self.page.click(f"{card} .seedv")
        self.page.drag_and_drop(f"{card} .seedv", "#seed")                  # the selected number dragged into the form
        self.assertEqual(self.page.input_value("#seed"), "2034940401")

    def test_after_a_click_keyboard_focus_moves_on_as_from_text(self):
        card = self.finished()
        self.page.click(f"{card} .seedv")
        self.page.keyboard.press("Shift+Tab")                               # not back onto the clicked seed
        self.assertFalse(self.page.evaluate("document.activeElement.classList.contains('seedv')"))
        self.assertEqual(self.page.get_attribute(f"{card} .seedv", "tabindex"), "0")   # a Tab stop again
        self.page.click(f"{card} .seedv")
        self.page.click(f"{card} .when")                                    # a press elsewhere, then Tab
        self.page.keyboard.press("Tab")
        self.assertTrue(self.page.evaluate("document.activeElement.classList.contains('seedv')"))
        self.wait_selected("2034940401")

    def test_a_touch_tap_neither_focuses_the_seed_nor_drops_a_selection(self):
        # Chromium (touch through CDP): a tap's mouse events come when the finger lifts, here 80 ms later
        try:
            browser = self.pw.chromium.launch()
        except Exception:
            self.skipTest("needs Playwright's Chromium")
        try:
            card = self.finished(prompt="a red fox runs through snow")
            ctx = browser.new_context(has_touch=True, viewport={"width": 1280, "height": 900})
            page = ctx.new_page()
            page.goto(self.studio.url + "/")
            page.wait_for_selector(f"{card} video")
            box = page.locator(f"{card} .seedv").bounding_box()
            point = {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}
            whole = page.evaluate(f"""() => {{ const c = document.querySelector('{card}'), r = document.createRange();
                r.setStart(c.querySelector('.p').firstChild, 0); r.setEnd(c.querySelector('.seedv').firstChild, 10);
                getSelection().removeAllRanges(); getSelection().addRange(r); return getSelection().toString(); }}""")
            cdp = ctx.new_cdp_session(page)
            cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [point]})
            page.wait_for_timeout(80)
            cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            page.wait_for_timeout(300)
            self.assertFalse(page.evaluate("document.activeElement.classList.contains('seedv')"))
            self.assertEqual(page.evaluate("getSelection().toString()"), whole)
        finally:
            browser.close()

    def test_the_seed_is_readable(self):
        card = self.finished()
        for scheme in ("light", "dark"):
            self.page.emulate_media(color_scheme=scheme)
            # both colours through a canvas (any CSS colour format; the text blended over the card), then WCAG contrast
            ratio = self.page.evaluate(f"""() => {{
                const x = document.createElement('canvas').getContext('2d', {{willReadFrequently: true}});
                const lum = () => {{ const [r, g, b] = [...x.getImageData(0, 0, 1, 1).data].slice(0, 3).map(v => {{
                    v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }});
                    return 0.2126 * r + 0.7152 * g + 0.0722 * b; }};
                const paint = c => {{ x.fillStyle = c; x.fillRect(0, 0, 1, 1); }};
                paint(getComputedStyle(document.querySelector('{card}')).backgroundColor); const bg = lum();
                paint(getComputedStyle(document.querySelector('{card} .seedv')).color); const fg = lum();
                return (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05); }}""")
            self.assertGreaterEqual(ratio, 4.5, f"seed contrast in {scheme} mode")


class PhoneWidth(PageTest):
    """#19: no sideways scroll on a phone, and the card buttons still share one row: in cards too narrow for their
    names, Upscale and Delete show their icons only (with their names as aria-label)."""

    def test_no_sideways_scroll_and_one_row_of_buttons(self):
        ok = self.studio.generate(prompt="a long prompt " * 30, seed=2034940401)[1]["id"]
        bad = self.studio.generate(prompt="STUB_FAIL", seed=2147483646)[1]["id"]
        self.studio.wait([ok, bad])
        self.page.wait_for_selector(f'article[data-id="{ok}"] video')
        self.page.wait_for_selector(f'article[data-id="{bad}"][data-status="failed"]')
        for width in (300, 320, 340, 360, 375, 414, 600):
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.wait_for_timeout(300)
            scroll, client, wide = self.page.evaluate("""() => {
                const cw = document.documentElement.clientWidth;
                const wide = [...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > cw + 0.5)
                    .map(e => e.tagName + (e.id ? '#' + e.id : '') + '.' + e.className);
                return [document.documentElement.scrollWidth, cw, wide]; }""")
            self.assertLessEqual(scroll, client, f"sideways scroll at {width} px: {wide}")
            self.assertEqual(wide, [], f"wider than the screen at {width} px")
            up = ["Upscale"] if self.upscale else []
            for jid, buttons in ((ok, ["Remix", "Download", *up, "Delete"]), (bad, ["Remix", "Delete"])):
                row = self.page.eval_on_selector_all(
                    f'article[data-id="{jid}"] .acts > *', "els => els.filter(e => e.offsetWidth).map(e => "
                                                           "[e.textContent.trim(), Math.round(e.getBoundingClientRect().top), "
                                                           "e.innerText.trim(), e.getAttribute('aria-label')])")
                self.assertEqual([name for name, *_ in row], buttons, width)
                self.assertEqual(len({top for _, top, *_ in row}), 1, f"one row at {width} px: {row}")
                tight = self.page.evaluate("document.getElementById('jobs').classList.contains('tight')")
                for name, _, shown, label in row:
                    if name in ("Upscale", "Delete"):        # the name as text, or as aria-label beside an icon
                        self.assertEqual((shown, label), ("" if tight else name, name), width)
                if not self.upscale:                         # without Upscale the names fit even at 300 px
                    self.assertFalse(tight, width)
                elif width <= 340 or width == 600:           # no room for the names on a small phone; room at 600
                    self.assertEqual(tight, width <= 340, width)

    def test_a_long_file_name_in_an_error_wraps(self):
        """The form panel scrolls on its own, so check it too: an error naming a long upload must wrap."""
        self.page.set_viewport_size({"width": 320, "height": 900})
        self.pick("task", "ref2va")
        self.page.fill("#prompt", "refs")
        name = "PXL_20261011_145012345_short_clip_from_the_trip.mp4"           # nowhere to break the saved name
        self.page.set_input_files('input[name="ref_videos"]', files=[{"name": name, "mimeType": "video/mp4",
                                                                       "buffer": mp4(1)}])
        self.page.click("#go")
        self.page.wait_for_function("document.getElementById('err').textContent.includes('at least 2 s')")
        widths = self.page.evaluate("""() => [...document.querySelectorAll('#f, #err')].map(e => [e.id, e.scrollWidth,
                                                                                          e.clientWidth])""")
        for el, scroll, client in widths:
            self.assertLessEqual(scroll, client, f"#{el} scrolls sideways: {widths}")


class OneRowUpscaleOff(PageTest):
    """The one row of card buttons with the experimental Upscale off, as every studio runs by default: Remix,
    Download and Delete, with their names, at every width (#15, #19)."""
    upscale = False

    def seed(self):
        self.studio.env["LFSTUDIO_UPSCALE"] = "0"

    test_seed_in_the_info_line_and_one_row_of_buttons = SeedText.test_seed_in_the_info_line_and_one_row_of_buttons
    test_no_sideways_scroll_and_one_row_of_buttons = PhoneWidth.test_no_sideways_scroll_and_one_row_of_buttons


class CancelCards(PageTest):
    """#20: Cancel on queued and running cards only; the card stays as Cancelled (neutral, not an error), with Remix
    and Delete, and Remix restores everything."""
    load_s, clip_s = 6, 6

    def seed(self):
        self.studio.env["STUB_CLOSE_S"] = "3"                                 # a stopped clip shows "Cancelling" 3 s
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([
            finished_job("1" * 32, created=1), finished_job("2" * 32, created=2, status="failed", error="boom")]))

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def status(self, jid, status):
        self.page.wait_for_selector(f'{self.card(jid)}[data-status="{status}"]')

    def visible(self, jid):
        """The card's buttons on screen, by name."""
        return self.page.eval_on_selector_all(f"{self.card(jid)} .acts > *",
                                              "els => els.filter(e => e.offsetWidth).map(e => e.textContent.trim())")

    def test_the_cancel_toast_uses_the_width_of_a_phone(self):
        self.page.set_viewport_size({"width": 320, "height": 800})
        a = self.studio.generate(prompt="running", seconds=8)[1]["id"]
        self.status(a, "loading models + generating")
        self.page.click(f"{self.card(a)} .cancel")
        self.page.wait_for_function("document.getElementById('toast').textContent.includes('models load again')")
        width = self.page.eval_on_selector("#toast", "e => e.getBoundingClientRect().width")
        self.assertGreater(width, 0.75 * 320, "a long toast is not squeezed to half the screen")
        self.assertLessEqual(width, 320 - 30)

    def test_focus_stays_on_the_card_when_a_poll_hides_cancel(self):
        """Cancel focused, then the clip finishes, or another tab cancels it: focus moves to Delete or Remix,
        not to the page."""
        a = self.studio.generate(prompt="finishes by itself", seconds=6)[1]["id"]
        self.status(a, "loading models + generating")
        self.page.focus(f"{self.card(a)} .cancel")
        self.status(a, "done")
        self.assertEqual(self.page.evaluate("document.activeElement.textContent.trim()"), "Delete")
        b = self.studio.generate(prompt="cancelled elsewhere", seed=3, seconds=6)[1]["id"]
        self.status(b, "generating")
        self.page.focus(f"{self.card(b)} .cancel")
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": b})[0], 200)     # another tab
        self.page.wait_for_function(f"document.querySelector('{self.card(b)} .cancel').hidden")
        self.assertEqual(self.page.evaluate("document.activeElement.textContent.trim()"), "Remix")

    def test_cancel_queued_and_running_cards(self):
        a = self.studio.generate(prompt="running", seconds=8)[1]["id"]
        b = self.studio.generate(prompt="queued", seed=2)[1]["id"]
        self.status(a, "loading models + generating")
        self.status(b, "queued")
        self.assertEqual(self.visible(a), ["Remix", "Cancel"])
        self.assertEqual(self.visible(b), ["Remix", "Cancel"])
        self.assertEqual(self.visible("1" * 32), ["Remix", "Download", "Upscale", "Delete"])   # finished: no Cancel
        self.assertEqual(self.visible("2" * 32), ["Remix", "Delete"])
        self.assertIn("models load again", self.page.get_attribute(f"{self.card(a)} .cancel", "title"))
        self.assertIn("out of the queue", self.page.get_attribute(f"{self.card(b)} .cancel", "title"))
        self.page.click(f"{self.card(b)} .cancel")                             # the mouse
        self.status(b, "cancelled")
        self.assertEqual(self.studio.job(b)["status"], "cancelled")
        self.assertEqual(self.visible(b), ["Remix", "Delete"])
        texts = [self.page.inner_text(f"{self.card(b)} {part}") for part in (".st", ".ph2 .big", ".errtxt")]
        self.assertEqual(texts, ["Cancelled", "Cancelled", ""])
        self.pick("seconds", "8")                                              # a's settings: no reload
        self.assertNotIn("load", self.go_text())
        self.page.focus(f"{self.card(a)} .cancel")                             # the keyboard
        self.page.keyboard.press("Enter")
        self.page.wait_for_selector(f"{self.card(a)} .cancel", state="hidden")
        self.assertTrue(self.page.evaluate(f"document.activeElement === document.querySelector("
                                           f"'{self.card(a)} .reuse-all')"), "focus moves to the card's Remix")
        self.assertEqual(self.page.inner_text(f"{self.card(a)} .st"), "Cancelling")   # while its session closes
        self.assertIn("loads the models first", self.go_text())               # the next clip loads them again
        self.status(a, "cancelled")
        self.assertEqual(self.visible(a), ["Remix", "Delete"])
        self.assertIsNone(self.page.query_selector(f"{self.card(a)} video"))   # nothing to play or download
        self.page.wait_for_function("document.getElementById('count').textContent === '1 clip'")   # none running
        # neutral, in both themes: the status in the muted colour of the other tags (not a failed clip's red or a
        # running one's orange), the card bordered like a finished clip
        for scheme in ("light", "dark"):
            self.page.emulate_media(color_scheme=scheme)
            look = self.page.evaluate(f"""() => {{ const c = s => getComputedStyle(document.querySelector(s));
                return [c('{self.card(a)} .st').color, c('{self.card(a)} .tag').color,
                        c('{self.card(a)}').borderColor, c('{self.card("1" * 32)}').borderColor]; }}""")
            self.assertEqual(look[0], look[1], scheme)
            self.assertEqual(look[2], look[3], scheme)

    def test_an_older_poll_does_not_bring_the_button_back(self):
        self.studio.generate(prompt="running", seconds=8)
        b = self.studio.generate(prompt="queued", seed=2)[1]["id"]
        self.status(b, "queued")
        held = []
        self.page.route("**/api/state", lambda route: held.append((route, route.fetch())))
        self.page.wait_for_timeout(2500)                                       # polls answered before the Cancel
        stale = list(held)
        self.assertTrue(stale and all(b"queued" in r.body() for _, r in stale))
        self.page.click(f"{self.card(b)} .cancel")
        self.status(b, "cancelled")
        for route, response in stale:
            route.fulfill(response=response)
        self.page.wait_for_timeout(500)
        self.assertEqual(self.page.get_attribute(self.card(b), "data-status"), "cancelled")
        self.assertEqual(self.visible(b), ["Remix", "Delete"])
        for route, response in held[len(stale):]:
            route.fulfill(response=response)
        self.page.unroute("**/api/state")

    def test_remix_restores_settings_and_files(self):
        frames = {"first_frame": [("first.png", png())], "last_frame": [("last.png", png(40))]}
        a = self.studio.generate(task="fl2va", orientation="portrait", seconds=8, prompt="the cancelled one", seed=1234,
                                 files=frames)[1]["id"]
        self.status(a, "loading models + generating")
        self.page.click(f"{self.card(a)} .cancel")
        self.status(a, "cancelled")
        self.pick("task", "t2va")
        self.pick("orientation", "landscape")
        self.pick("seconds", "5")
        self.page.fill("#prompt", "something else")
        self.page.click(f"{self.card(a)} .reuse-all")
        self.page.wait_for_function("document.querySelectorAll('#fl .thumbs img.kept').length === 2")
        form = self.page.evaluate("""() => ['task', 'orientation', 'seconds'].map(n =>
            document.querySelector(`input[name=${n}]:checked`).value)""")
        self.assertEqual(form, ["fl2va", "portrait", "8"])
        self.assertEqual((self.page.input_value("#prompt"), self.page.input_value("#seed")),
                         ("the cancelled one", "1234"))
        self.assertEqual(self.page.inner_text("#err"), "")
        self.page.click("#go")
        new = self.studio.wait_for(lambda s: len([j for j in s["jobs"] if j["prompt"] == "the cancelled one"]) == 2)
        job = next(j for j in new["jobs"] if j["prompt"] == "the cancelled one" and j["id"] != a)
        self.assertEqual([Path(job[k]).name for k in ("first_frame", "last_frame")], ["first.png", "last.png"])
        self.assertEqual((job["orientation"], job["seconds"], job["seed"]), ("portrait", 8, 1234))


class DoubleClicks(PageTest):
    """A double-click whose first click changes what is under the pointer: the second one (detail 2) must not reach
    the Remix, Cancel or Delete that is there now, nor, after Undo, anything else. The running clip loads for 30 s, so
    the others stay queued."""
    load_s, clip_s = 30, 6

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([
            finished_job("1" * 32, created=1), finished_job("2" * 32, created=2, status="failed", error="boom"),
            finished_job("3" * 32, created=3)]))

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def status(self, jid, status):
        self.page.wait_for_selector(f'{self.card(jid)}[data-status="{status}"]')

    def center(self, selector):
        box = self.page.locator(selector).bounding_box()
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def under(self, x, y):
        """The card button at a point of the screen: [card id, class], or None."""
        return self.page.evaluate("""([x, y]) => { const b = document.elementFromPoint(x, y)?.closest('.act');
            return b ? [b.closest('.job').dataset.id, b.className] : null; }""", [x, y])

    def second_click(self):
        """The second click of a double-click, where the pointer is (the first was a plain click there)."""
        self.page.mouse.down(click_count=2); self.page.mouse.up(click_count=2)

    def test_a_double_click_on_cancel_of_a_queued_clip_does_not_delete_it(self):
        """The answer shows Delete right where Cancel was; the double-click's second click must not delete."""
        self.studio.generate(prompt="running", seconds=8)                     # keeps the next one queued
        b = self.studio.generate(prompt="queued", seed=2)[1]["id"]
        self.status(b, "queued")
        self.page.mouse.move(*self.center(f"{self.card(b)} .cancel"))
        self.page.mouse.down(); self.page.mouse.up()
        self.page.wait_for_timeout(120)
        self.second_click()
        self.page.wait_for_timeout(600)
        self.assertNotIn(b, self.studio.state()["deleted"], "the second click deleted the cancelled clip")
        self.status(b, "cancelled")
        self.assertEqual(self.studio.job(b)["status"], "cancelled")
        self.assertFalse(self.page.eval_on_selector("#undo", "e => e.classList.contains('show')"))

    def test_a_double_click_on_delete_does_not_cancel_the_next_card(self):
        """A cancelled clip's card comes right before the older running or queued one (newest first), and both have
        their last button at the right end: Delete takes the cancelled card away, the next card moves up, and its
        Cancel is under the pointer for the double-click's second click."""
        a = self.studio.generate(prompt="neighbour 1", seconds=8)[1]["id"]
        self.status(a, "loading models + generating")
        for neighbour, status, seed in ((a, "loading models + generating", 2), (None, "queued", 4)):
            if neighbour is None:                                               # queued behind a
                neighbour = self.studio.generate(prompt="neighbour 2", seed=seed - 1)[1]["id"]
            gone = self.studio.generate(prompt=f"cancelled {seed}", seed=seed)[1]["id"]
            self.assertEqual(self.studio.post("/api/cancel", json_body={"id": gone})[0], 200)
            self.status(neighbour, status)
            self.status(gone, "cancelled")
            x, y = self.center(f"{self.card(gone)} .del")
            self.page.mouse.move(x, y)
            self.page.mouse.down(); self.page.mouse.up()
            self.page.wait_for_selector(self.card(gone), state="detached")
            self.assertEqual(self.under(x, y), [neighbour, "act cancel"], "the next card's Cancel moved up")
            self.second_click()
            self.page.wait_for_timeout(600)
            self.assertIn(gone, self.studio.state()["deleted"])
            self.assertEqual(self.studio.job(neighbour)["status"], status, f"the second click cancelled the {status} clip")
            self.assertFalse(self.studio.job(neighbour).get("cancel_requested"))
            self.assertEqual(self.page.get_attribute(self.card(neighbour), "data-status"), status)
            self.page.wait_for_selector("#undo.show")                            # Undo is offered, no Cancel toast
            self.assertFalse(self.page.eval_on_selector("#toast", "e => e.classList.contains('show')"))

    def test_a_double_click_on_undo_does_not_cancel_the_card_under_it(self):
        """Undo hides at once, so the double-click's second click goes to what is under the toast: on a phone, the
        Cancel of the queued card below the restored one. Remix there ignores it too (it would replace the draft)."""
        self.page.set_viewport_size({"width": 320, "height": 640})
        self.studio.generate(prompt="running", seconds=8)
        c = self.studio.generate(prompt="queued 1", seed=2)[1]["id"]
        d = self.studio.generate(prompt="queued 2", seed=3)[1]["id"]
        self.assertEqual(self.studio.post("/api/cancel", json_body={"id": d})[0], 200)
        self.status(c, "queued")
        self.status(d, "cancelled")
        # where Undo shows: Delete a finished clip and bring it back
        self.page.evaluate(f"document.querySelector('{self.card('1' * 32)} .del').scrollIntoView({{block: 'center'}})")
        self.page.click(f"{self.card('1' * 32)} .del")
        self.page.wait_for_selector("#undo.show")
        self.page.wait_for_timeout(300)                                         # its transition
        undo = self.page.locator("#undo button").bounding_box()
        self.page.click("#undo button")
        self.page.wait_for_selector(f"{self.card('1' * 32)}")
        self.page.wait_for_selector("#undo", state="hidden")
        # c's Cancel under the Undo button, d's Delete above it on screen
        cancel = self.page.locator(f"{self.card(c)} .cancel").bounding_box()
        x = (max(undo["x"], cancel["x"]) + min(undo["x"] + undo["width"], cancel["x"] + cancel["width"])) / 2
        y = undo["y"] + undo["height"] / 2
        self.assertTrue(undo["x"] < x < undo["x"] + undo["width"] and cancel["x"] < x < cancel["x"] + cancel["width"],
                        "the Undo button is over the Cancel buttons' column at 320 px")
        self.page.evaluate(f"scrollBy(0, {cancel['y'] + cancel['height'] / 2 - y})")
        self.page.wait_for_timeout(300)
        self.assertEqual(self.under(x, y), [c, "act cancel"])
        self.page.mouse.click(*self.center(f"{self.card(d)} .del"))           # on screen, above c
        self.page.wait_for_selector("#undo.show")
        self.page.wait_for_timeout(300)
        self.page.mouse.move(x, y)
        self.page.mouse.down(); self.page.mouse.up()
        self.page.wait_for_selector(self.card(d))                               # back, above c again
        self.assertEqual(self.under(x, y), [c, "act cancel"], "c's Cancel is under the pointer again")
        self.second_click()
        self.page.wait_for_function("document.getElementById('toast').textContent === 'Clip restored'")
        self.page.wait_for_timeout(600)
        self.assertEqual(self.studio.job(c)["status"], "queued", "the second click took c out of the queue")
        self.assertNotIn(d, self.studio.state()["deleted"])
        self.assertEqual(self.page.inner_text("#toast"), "Clip restored")
        # Remix: the rest of a double-click leaves the draft alone
        self.page.fill("#prompt", "my draft")
        self.page.evaluate(f"document.querySelector('{self.card(c)} .reuse-all').scrollIntoView({{block: 'center'}})")
        x, y = self.center(f"{self.card(c)} .reuse-all")
        self.assertEqual(self.under(x, y), [c, "act pri reuse-all"])
        self.page.mouse.move(x, y)
        self.second_click()
        self.page.wait_for_timeout(300)
        self.assertEqual(self.page.input_value("#prompt"), "my draft")

    def test_a_double_click_on_undo_does_not_reach_the_form(self):
        """In one column the form is above the gallery, and Undo shows for 10 s: scrolled up while it shows, Generate,
        the "New random seed every run" toggle or a prompt chip can be under it, or the video of a card above the
        deleted one. The rest of the double-click on Undo must not queue the draft, flip the toggle, add to the prompt
        or play the clip (WebKit plays a video on a click). At 390 px the cards are in one column too."""
        self.page.set_viewport_size({"width": 390, "height": 664})
        self.page.fill("#prompt", "my draft")
        form = ("[prompt.value, seed.value, autoroll.checked, "
                "[...document.querySelectorAll('video')].filter(v => !v.paused).length]")
        undo = None
        for target in ("#go", ".toggle", "#chips button", f"{self.card('3' * 32)} video"):
            with self.subTest(target=target):
                jobs = {j["id"] for j in self.studio.state()["jobs"]}
                self.page.click(f"{self.card('1' * 32)} .del")
                self.page.wait_for_selector("#undo.show")
                self.page.wait_for_timeout(300)                                 # its transition
                undo = undo or self.page.locator("#undo button").bounding_box()
                # scroll the target (the first one that shares columns with the Undo button) under the Undo button
                box = self.page.evaluate("""([sel, l, r]) => {
                    document.querySelectorAll('[data-target]').forEach(e => e.removeAttribute('data-target'));
                    const e = [...document.querySelectorAll(sel)].find(e => {
                        const b = e.getBoundingClientRect(); return b.left < r && b.right > l; });
                    e.setAttribute('data-target', ''); const b = e.getBoundingClientRect();
                    return {x: b.x, y: b.y, width: b.width, height: b.height}; }""",
                    [target, undo["x"], undo["x"] + undo["width"]])
                x = (max(undo["x"], box["x"]) + min(undo["x"] + undo["width"], box["x"] + box["width"])) / 2
                y = undo["y"] + undo["height"] / 2
                self.page.evaluate(f"scrollBy(0, {box['y'] + box['height'] / 2 - y})")
                self.page.wait_for_timeout(300)
                before = self.page.evaluate(form)
                self.page.evaluate("document.getElementById('toast').textContent = ''")
                self.page.mouse.move(x, y)
                self.page.mouse.down(); self.page.mouse.up()                   # the clip comes back below
                self.assertTrue(self.page.evaluate(f"!!document.elementFromPoint({x}, {y}).closest('[data-target]')"),
                                f"{target} is under the pointer once Undo hides")
                self.second_click()
                self.page.wait_for_selector(self.card("1" * 32))
                self.page.wait_for_timeout(600)
                self.assertEqual({j["id"] for j in self.studio.state()["jobs"]}, jobs,
                                 "the clip is back, and the second click queued nothing")
                self.assertEqual(self.page.evaluate(form), before, "the second click changed the form or played a clip")
                self.assertEqual(self.page.inner_text("#toast"), "Clip restored")


def picture(color, size="64x36"):
    """A PNG of one colour (ffmpeg)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s={size}", "-frames:v", "1",
                           "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True, check=True).stdout


class QueuePicture(PageTest):
    """#30: a queued or running card shows the clip's first picture behind its timer (clearer while queued, fainter
    while it runs); the card is not rebuilt for it, and the picture goes when the clip is done, failed or cancelled."""
    load_s, clip_s = 5, 5

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def status(self, jid, status):
        self.page.wait_for_selector(f'{self.card(jid)}[data-status="{status}"]', timeout=30000)

    def shown(self, jid):
        """The card's picture, once loaded: [src, opacity, blur in px], or None if the card has none on screen."""
        self.page.wait_for_function(f"""() => {{ const i = document.querySelector('{self.card(jid)} .qimg');
            return !i || i.complete; }}""")
        return self.page.evaluate(f"""() => {{ const i = document.querySelector('{self.card(jid)} .qimg');
            if (!i || !i.offsetWidth || !i.naturalWidth) return null; const s = getComputedStyle(i);
            return [new URL(i.src).pathname, +s.opacity, parseFloat(s.filter.match(/blur\\(([\\d.]+)px/)[1])]; }}""")

    def test_queued_and_running_cards(self):
        stills = []
        self.page.on("request", lambda r: stills.append(r.url.rsplit("/", 1)[1]) if "/still/" in r.url else None)
        a = self.studio.generate(task="fl2va", prompt="first frame", files={"first_frame": [("a.png", picture("red"))]})[1]["id"]
        b = self.studio.generate(task="ref2va", prompt="a video only", seed=2, files={"ref_videos": [("v.mp4", mp4())]})[1]["id"]
        c = self.studio.generate(prompt="text", seed=3)[1]["id"]
        d = self.studio.generate(task="fl2va", prompt="last frame", seed=4,
                                 files={"last_frame": [("d.png", picture("blue"))]})[1]["id"]
        self.status(a, "loading models + generating")
        self.status(b, "queued")
        running, queued = self.shown(a), self.shown(b)
        self.assertEqual((running[0], queued[0], self.shown(d)[0]), (f"/still/{a}", f"/still/{b}", f"/still/{d}"))
        self.assertIsNone(self.shown(c))                                         # t2va: no picture
        self.assertIsNone(self.page.query_selector(f"{self.card(c)} .qimg"))
        self.assertLess(running[1], queued[1])                                   # fainter and more blurred while it runs
        self.assertGreater(running[2], queued[2])
        # a layer of its own: WebKit blurs it once, not again on every frame of the running card's shimmer
        layer = f"getComputedStyle(document.querySelector('{self.card(a)} .qimg')).willChange"
        self.assertEqual(self.page.evaluate(layer), "transform")
        for width in (300, 320, 375, 420):                                       # phones: nothing wider than the screen
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.wait_for_timeout(200)
            wide = self.page.evaluate("""() => [...document.querySelectorAll('body *')].filter(e =>
                e.getBoundingClientRect().right > document.documentElement.clientWidth + 0.5).map(e => e.className)""")
            self.assertEqual(wide, [], f"wider than the screen at {width} px")
            self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), width)
        self.page.set_viewport_size({"width": 1280, "height": 900})
        # polls change the card's attributes only: the same picture, now in the running style
        self.page.evaluate(f"window.mark = document.querySelector('{self.card(b)} .qimg')")
        self.page.wait_for_selector(f'{self.card(b)}[data-status="loading models + generating"]', timeout=30000)
        self.page.wait_for_timeout(1000)                                         # its transition
        self.assertEqual(self.shown(b)[1:], running[1:])
        self.assertTrue(self.page.evaluate(f"window.mark === document.querySelector('{self.card(b)} .qimg')"))
        self.status(b, "done")                                                   # done: the video with its poster
        self.page.wait_for_selector(f"{self.card(b)} video")
        self.assertIsNone(self.page.query_selector(f"{self.card(b)} .qimg"))
        self.assertTrue(self.page.get_attribute(f"{self.card(b)} video", "poster").endswith(f"/poster/{b}"))
        self.assertEqual(sorted(stills), sorted([a, b, d]))                      # each asked for once
        self.studio.wait([a, b, c, d])
        stills.clear()
        self.page.reload()                                                       # finished cards ask for none
        self.page.wait_for_selector(f"{self.card(d)} video")
        self.page.wait_for_timeout(500)
        self.assertEqual(stills, [])

    def test_cancelled_and_failed_cards_drop_it(self):
        a = self.studio.generate(task="fl2va", prompt="STUB_FAIL", files={"first_frame": [("a.png", picture("red"))]})[1]["id"]
        b = self.studio.generate(task="fl2va", prompt="cancelled", seed=2,
                                 files={"first_frame": [("b.png", picture("blue"))]})[1]["id"]
        self.status(b, "queued")
        self.assertIsNotNone(self.shown(b))
        self.page.click(f"{self.card(b)} .cancel")
        self.status(b, "cancelled")
        self.assertIsNone(self.shown(b))
        self.assertEqual(self.page.inner_text(f"{self.card(b)} .ph2 .big"), "Cancelled")
        self.assertIsNotNone(self.shown(a))
        self.studio.wait([a], timeout=60)
        self.status(a, "failed")
        self.assertIsNone(self.shown(a))

    def contrast(self, jid, part):
        """WCAG contrast of the card's placeholder text (.big or .small) against what is behind it, the worst pixel:
        the text made transparent, a screenshot of its box decoded in the page."""
        sel = f"{self.card(jid)} .ph2 {part}"
        lum = """const lum = ([r, g, b]) => 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b),
                       f = v => (v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;"""
        text = self.page.evaluate(f"""() => {{ {lum} const x = document.createElement('canvas').getContext('2d');
            x.fillStyle = getComputedStyle(document.querySelector('{sel}')).color; x.fillRect(0, 0, 1, 1);
            return lum(x.getImageData(0, 0, 1, 1).data); }}""")
        self.page.evaluate(f"""() => document.querySelectorAll('{self.card(jid)} .ph2 > div').forEach(e =>
            e.style.cssText = 'color: transparent; text-shadow: none')""")
        shot = base64.b64encode(self.page.locator(sel).screenshot()).decode()
        self.page.evaluate(f"""() => document.querySelectorAll('{self.card(jid)} .ph2 > div').forEach(e =>
            e.style.cssText = '')""")
        back = self.page.evaluate(f"""async b64 => {{ {lum} const im = new Image(); im.src = 'data:image/png;base64,' + b64;
            await im.decode(); const x = document.createElement('canvas').getContext('2d');
            x.canvas.width = im.width; x.canvas.height = im.height; x.drawImage(im, 0, 0);
            const d = x.getImageData(0, 0, im.width, im.height).data; let worst = 0;
            for (let i = 0; i < d.length; i += 4) worst = Math.max(worst, lum([d[i], d[i + 1], d[i + 2]]));
            return worst; }}""", shot)
        return (max(text, back) + 0.05) / (min(text, back) + 0.05)

    def test_text_readable_over_a_white_picture(self):
        """The worst case for the light text: a white picture, in both themes, queued and running."""
        self.page.emulate_media(reduced_motion="reduce")                         # no shimmer or transition: a still
        white = {"first_frame": [("white.png", picture("white", "640x360"))]}
        a = self.studio.generate(task="fl2va", prompt="running", files=white)[1]["id"]
        b = self.studio.generate(task="fl2va", prompt="queued", seed=2, files=white)[1]["id"]
        self.status(a, "loading models + generating")
        self.status(b, "queued")
        self.assertIsNotNone(self.shown(a) and self.shown(b))
        for scheme in ("light", "dark"):
            self.page.emulate_media(color_scheme=scheme)
            for jid in (a, b):
                for part, least in ((".big", 4.5), (".small", 4.5)):
                    ratio = self.contrast(jid, part)
                    self.assertGreaterEqual(ratio, least, f"{part} on the {scheme} {jid == a and 'running' or 'queued'} card")


class UpscaleCards:
    """For the page tests of the experimental Upscale (the stub studio has it on, with tests/stub/upscale_job.py as
    its job)."""
    load_s, clip_s = 1, 0.5

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def visible(self, jid):
        return self.page.eval_on_selector_all(f"{self.card(jid)} .acts > *",
                                              "els => els.filter(e => e.offsetWidth).map(e => e.textContent.trim())")

    def done_clip(self, **kw):
        jid = self.studio.generate(**kw)[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        self.page.wait_for_selector(f"{self.card(jid)} video")
        return jid

    def open_popover(self, jid):
        self.page.click(f"{self.card(jid)} .act.up")
        self.page.wait_for_selector("#uppop", state="visible")

    def checked(self, name):
        return self.page.eval_on_selector(f'#uppop input[name="{name}"]:checked', "e => e.value")


class UpscalePage(UpscaleCards, PageTest):
    """The button on finished clips (disabled, with the reason, on the ones it can't take), the popover and its
    request, and the upscale's own card: tags, the player, Download and Delete, without Remix."""

    def test_button_popover_and_card(self):
        ok = self.done_clip(prompt="the clip to upscale", seed=31)
        ten = self.done_clip(prompt="too long", seconds=10, seed=32)
        self.page.wait_for_timeout(2200)                                     # a poll with upscalable
        up = f"{self.card(ok)} .act.up"
        self.assertEqual((self.page.get_attribute(up, "aria-disabled"), self.page.get_attribute(up, "title")),
                         ("false", "Upscale to 1080p, 2K or 4K (experimental)"))
        self.assertEqual(self.visible(ok), ["Remix", "Download", "Upscale", "Delete"])
        why = self.page.get_attribute(f"{self.card(ten)} .act.up", "title")
        self.assertEqual((self.page.get_attribute(f"{self.card(ten)} .act.up", "aria-disabled"), why),
                         ("true", "Upscale takes 5 s clips only, for now (NVIDIA's pipeline is fixed at 121 frames)"))
        self.page.click(f"{self.card(ten)} .act.up", force=True)             # aria-disabled: the reason, also on a
                                                                               # phone
        self.page.wait_for_function("t => document.getElementById('toast').textContent === t", arg=why)
        self.assertTrue(self.page.is_hidden("#uppop"))
        # the popover: the server's defaults, the expected times, one line on the models; Escape closes it
        self.open_popover(ok)
        self.assertEqual([self.checked(n) for n in ("up_res", "up_strength")], ["2k", "balanced"])
        self.assertIsNone(self.page.query_selector('#uppop input[name="up_source"]'))      # one source: the clip's MP4
        text = self.page.inner_text("#uppop")
        for words in ("1080p\n~10 min", "2K\n~19 min", "4K\n~1.1 h", "in all, with a ~22 s load",
                      "Faithful", "Start noise 0.5, 2 steps", "Balanced", "Full re-generate", "redraws faces",
                      "Unloads the models while it runs"):
            self.assertIn(words, text)
        self.assertNotIn("draft", text)
        self.assertNotIn("9:16", text)                                       # the note is for 9:16 clips
        self.assertNotIn("untested", text)                                   # 4K has run (16:9, docs/PERFORMANCE.md)
        self.assertEqual([self.page.get_attribute(f'#uppop label[for="up-{r}"]', "title") for r in ("1080p", "2k", "4k")],
                         ["1920×1080", "2560×1440", "3840×2160"])
        # the progress bar's expected seconds (docs/PERFORMANCE.md): the fast load, ~8 min more with the standard load
        # (the record's "load"); a record of an older studio's upscale from the draft counts as any other
        self.assertEqual(self.page.evaluate("""[{resolution: '2k', steps: 2, load: 'fast'},
            {resolution: '2k', steps: 3}, {resolution: '1080p', steps: 2},
            {resolution: '2k', steps: 2, load: 'standard'}, {resolution: '4k', steps: 2},
            {resolution: '1080p', steps: 2, source: 'draft'}].map(upEta)"""), [1165, 1545, 615, 1665, 3995, 615])
        # its times are upEta's for 2 steps with the studio's load (/api/state upscale_load), and follow it each time
        # it opens (one evaluate: no poll in between)
        self.assertEqual(self.page.evaluate("""['1080p', '2k', '4k'].flatMap(r => ['fast', 'standard'].map(load =>
            upEta({resolution: r, steps: 2, load})))"""), [615, 1115, 1165, 1665, 3995, 4495])
        self.page.keyboard.press("Escape")
        self.assertTrue(self.page.is_hidden("#uppop"))
        times = self.page.evaluate("""id => {
            const btn = document.querySelector(`article[data-id="${id}"] .act.up`), seen = [];
            for (const load of ['standard', 'fast', 'standard', undefined]) {
              state.upscale_load = load; openUpscale(id, btn);
              seen.push([...document.querySelectorAll('#uppop small[data-standard]')].map(e => e.textContent));
              closeUpscale(false);
            }
            state.upscale_load = 'fast'; return seen;}""", ok)
        fast = ["in all, with a ~22 s load", "~10 min", "~19 min", "~1.1 h"]
        standard = ["in all, with a ~7.5 min load", "~19 min", "~28 min", "~1.2 h"]
        self.assertEqual(times, [standard, fast, standard, fast])
        self.page.focus(up)                                                   # the keyboard: Enter opens it, focus
        self.page.keyboard.press("Enter")                                     # on the resolution; Escape returns
        self.page.wait_for_selector("#uppop", state="visible")
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "up-2k")
        self.page.keyboard.press("Escape")
        self.assertTrue(self.page.evaluate(f"document.activeElement === document.querySelector('{up}')"))
        self.page.click(f"{up} svg")                                          # the button toggles it, its icon too
        self.page.wait_for_selector("#uppop", state="visible")
        self.page.click(f"{up} svg")
        self.page.wait_for_selector("#uppop", state="hidden")
        self.open_popover(ok)                                                 # a click elsewhere closes it
        self.page.click("#jobs .job .p")
        self.page.wait_for_selector("#uppop", state="hidden")
        # 1080p, faithful: a card of its own at the top, with the clip's prompt
        self.open_popover(ok)
        self.page.click('#uppop label[for="up-1080p"]')
        self.page.click('#uppop input[name="up_strength"][value="faithful"]')
        with self.page.expect_request("**/api/upscale") as request:
            self.page.click("#upgo")
        self.assertEqual(json.loads(request.value.post_data), {"id": ok, "resolution": "1080p", "strength": "faithful"})
        self.page.wait_for_function("document.getElementById('toast').textContent.startsWith('1080p upscale queued')")
        self.assertTrue(self.page.is_hidden("#uppop"))
        job = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual({k: job[k] for k in ("parent", "resolution", "strength", "audio")},
                         {"parent": ok, "resolution": "1080p", "strength": "faithful", "audio": "clip"})
        uid, card = job["id"], self.card(job["id"])
        self.assertEqual(self.page.get_attribute("#jobs .job", "data-id"), uid)          # the newest: at the top
        self.assertEqual(self.page.get_attribute(card, "data-kind"), "upscale")
        tags = self.page.eval_on_selector_all(f"{card} .tags .tag", "els => els.map(e => e.textContent)")
        self.assertEqual(tags, ["1080p upscale", "Text", "5 s"])
        self.assertEqual(self.page.inner_text(f"{card} .p"), "the clip to upscale")
        self.assertIn("Faithful", self.page.inner_text(f"{card} .when"))
        self.assertIsNone(self.page.query_selector(f"{card} .reuse-all"))   # no Remix
        self.assertIsNone(self.page.query_selector(f"{card} .act.up"))      # an upscale is not upscaled again
        self.page.wait_for_selector(f"{card} video", timeout=30000)
        self.assertEqual(self.visible(uid), ["Download", "Delete"])
        self.assertEqual(self.page.get_attribute(f"{card} a.act", "download"),
                         f"lf-studio_t2va_5s_seed31_{ok[:6]}_1080p_{uid[:6]}.mp4")
        self.assertEqual(self.page.inner_text(f"{card} .st"), "Done")
        self.assertIn("Move this upscale to the trash", self.page.get_attribute(f"{card} .del", "title"))
        self.page.click(f"{card} .del")                                     # its own Undo toast
        self.page.wait_for_selector(card, state="detached")
        self.page.wait_for_function("document.querySelector('#undo span').textContent === 'Upscale deleted'")

    def test_without_the_clips_stage1_audio_and_a_refusal(self):
        """No Stage-1 audio: the request asks for silence. A refusal shows in the popover, which stays open."""
        ok = self.done_clip(prompt="files gone", seed=51)
        case = Path(self.studio.job(ok)["output"]).parents[1]
        (case / "stage1" / "stage1_direct_tensors.pt").unlink()
        self.page.wait_for_timeout(2200)                                     # a poll with the new upscalable
        self.open_popover(ok)
        self.page.route("**/api/upscale", lambda route: route.fulfill(
            status=409, content_type="application/json", body=json.dumps({"error": "the clip is busy, try later"})))
        self.page.click("#upgo")
        self.page.wait_for_function("document.getElementById('uperr').textContent === 'the clip is busy, try later'")
        self.assertTrue(self.page.is_visible("#uppop"))
        self.assertFalse(self.page.is_disabled("#upgo"))
        self.page.unroute("**/api/upscale")
        self.page.click("#upgo")
        self.page.wait_for_selector("#uppop", state="hidden")
        job = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        self.assertEqual(job["audio"], "silence")
        self.studio.wait([job["id"]], timeout=30)


class UpscalePortrait(UpscaleCards, PageTest):
    """A 5 s 9:16 clip: the button on, the popover's sizes upright with its note, and the upscale's card upright."""

    def test_a_9x16_clip(self):
        tall = self.done_clip(prompt="an upright one", orientation="portrait", seed=61)
        self.page.wait_for_timeout(2200)                                     # a poll with upscalable
        self.assertEqual(self.page.get_attribute(f"{self.card(tall)} .act.up", "aria-disabled"), "false")
        self.open_popover(tall)
        self.assertEqual([self.page.get_attribute(f'#uppop label[for="up-{r}"]', "title") for r in ("1080p", "2k", "4k")],
                         ["1080×1920", "1440×2560",
                          "2160×3840 · the studio's 4K upscale has not been run with a 9:16 clip yet"])
        self.assertTrue(self.page.is_visible("#upportrait"))
        text = self.page.inner_text("#uppop")
        for words in ("9:16 comes out upright (2K: 1440×2560). NVIDIA trained its refiner on 16:9 video",
                      "1080p\n~10 min", "2K\n~19 min", "4K\n~1.1 h"):        # the same times: the same tokens
            self.assertIn(words, text)
        with self.page.expect_request("**/api/upscale") as request:
            self.page.click("#upgo")
        self.assertEqual(json.loads(request.value.post_data), {"id": tall, "resolution": "2k", "strength": "balanced"})
        self.page.wait_for_selector("#uppop", state="hidden")
        job = next(j for j in self.studio.state()["jobs"] if j.get("kind") == "upscale")
        card = self.card(job["id"])
        self.page.wait_for_selector(f"{card} video", timeout=30000)
        self.assertEqual(self.page.eval_on_selector_all(f"{card} .tags .tag", "els => els.map(e => e.textContent)"),
                         ["2K upscale", "Text", "5 s"])
        self.assertEqual(self.page.eval_on_selector(f"{card} .media", "e => e.style.aspectRatio"), "4 / 5")  # as its clip's
        self.assertEqual(self.page.get_attribute(f"{card} a.act", "download"),
                         f"lf-studio_t2va_5s_seed61_{tall[:6]}_2k_{job['id'][:6]}.mp4")
        self.open_popover(tall)                                               # and back on a 16:9 clip: no note
        self.page.keyboard.press("Escape")
        wide = self.done_clip(prompt="a wide one", seed=62)
        self.page.wait_for_timeout(2200)
        self.open_popover(wide)
        self.assertTrue(self.page.is_hidden("#upportrait"))
        self.assertEqual(self.page.get_attribute('#uppop label[for="up-2k"]', "title"), "2560×1440")


class UpscaleRunning(UpscaleCards, PageTest):
    """While an upscale runs (its stand-in job takes 60 s): its phase and timer, the status says why no models are
    loaded, Generate loads them first; Cancel stops it and leaves Delete."""

    def seed(self):
        self.studio.env["STUB_UPSCALE_S"] = "60"

    def test_running_upscale_and_cancel(self):
        ok = self.done_clip(prompt="a long one", seed=41)
        uid = self.studio.upscale(ok, resolution="4k")[1]["id"]
        card = self.card(uid)
        self.page.wait_for_function(f"(document.querySelector('{card} .ph2 .small') || {{}}).textContent "
                                    "=== 'Upscaling · refine 1/2'", timeout=30000)
        self.assertEqual(self.page.inner_text(f"{card} .st"), "Upscaling")
        self.assertEqual(self.page.eval_on_selector_all(f"{card} .tags .tag", "els => els.map(e => e.textContent)"),
                         ["4K upscale", "Text", "5 s"])
        self.assertRegex(self.page.inner_text(f"{card} .ph2 .big"), r"^0:\d\d$")
        self.assertEqual(self.visible(uid), ["Cancel"])
        self.assertEqual(self.page.get_attribute(f"{card} .cancel", "title"), "Stop this upscale")
        self.assertEqual(self.page.inner_text("#loaded"), "closed for an upscale")
        self.assertIn("loads the models first", self.go_text())
        self.page.focus(f"{card} .cancel")
        self.page.keyboard.press("Enter")
        self.page.wait_for_selector(f'{card}[data-status="cancelled"]', timeout=20000)
        self.assertEqual(self.visible(uid), ["Delete"])
        self.assertEqual(self.page.inner_text(f"{card} .ph2 .big"), "Cancelled")
        self.assertEqual(self.studio.job(uid)["status"], "cancelled")


class UpscaleDefaults(UpscaleCards, PageTest):
    """The popover starts at the server's defaults (/api/state upscale_defaults: LFSTUDIO_UPSCALE_STRENGTH) and shows
    the times with its load (upscale_load: LFSTUDIO_UPSCALE_LOAD=standard, ~8 min more)."""

    def seed(self):
        self.studio.env.update(LFSTUDIO_UPSCALE_STRENGTH="full", LFSTUDIO_UPSCALE_LOAD="standard")

    def test_the_servers_strength_and_load(self):
        ok = self.done_clip(prompt="defaults")
        self.open_popover(ok)
        self.assertEqual([self.checked(n) for n in ("up_res", "up_strength")], ["2k", "full"])
        text = self.page.inner_text("#uppop")
        for words in ("1080p\n~19 min", "2K\n~28 min", "4K\n~1.2 h", "in all, with a ~7.5 min load"):
            self.assertIn(words, text)


class UpscalePageOff(PageTest):
    """LFSTUDIO_UPSCALE=0: no Upscale button anywhere."""
    load_s, clip_s = 1, 0.5

    def seed(self):
        self.studio.env["LFSTUDIO_UPSCALE"] = "0"

    def test_no_button(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid], timeout=60)
        self.page.wait_for_selector(f'article[data-id="{jid}"] video')
        self.page.wait_for_timeout(2200)
        self.assertEqual(self.page.eval_on_selector_all(f'article[data-id="{jid}"] .acts > *',
                                                        "els => els.filter(e => e.offsetWidth).map(e => "
                                                        "e.textContent.trim())"), ["Remix", "Download", "Delete"])


def form_fields(request):
    """The text fields of a multipart POST the page sent, in their order: [(name, value)]; the names of its file
    fields (sent empty or not) as ("<file>", name)."""
    boundary = request.headers["content-type"].split("boundary=", 1)[1].strip('"').encode()
    out = []
    for part in request.post_data_buffer.split(b"--" + boundary)[1:-1]:
        head, _, body = part.partition(b"\r\n\r\n")
        m = re.search(rb'name="([^"]*)"(; filename=)?', head)
        if m:
            out.append(("<file>", m.group(1).decode()) if m.group(2) else (m.group(1).decode(), body[:-2].decode()))
    return out


_TABLE = []


def models_table():
    """The models' table a studio with LightX2V Turbo on serves (/api/state "models"), read once from a stub studio."""
    if not _TABLE:
        root = Path(tempfile.mkdtemp(prefix="lf-studio-page-"))
        try:
            studio = StubStudio(root, preload=False, models=BOTH).start()
            try:
                _TABLE.append(studio.state()["models"])
            finally:
                studio.stop()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return _TABLE[0]


# what LightX2V Turbo did not make before its GPU validation, as a table of a later model may hold such reasons
# (not_yet_table)
NOT_LAST = "LightX2V Turbo needs a first frame for now (a last frame alone: use Sol-H3)"
NOT_LONGER = "LightX2V Turbo makes 5 s clips for now"


def not_yet_table():
    """The models' table with LightX2V Turbo as it was before its GPU validation cleared its lengths and a last frame
    alone: 5 s and a first frame (or both) only, with their reasons, and its times at 5 s only. The page must still grey
    out such picks for a model that has them: the tests put this table into /api/state."""
    table = copy.deepcopy(models_table())
    turbo = next(m for m in table if m["id"] == TURBO)
    for c in turbo["capabilities"]["frames"]:
        if c["value"] == "last":
            c.update(offered=False, why=NOT_LAST)
    for c in turbo["capabilities"]["seconds"]:
        if c["value"] != 5:
            c.update(offered=False, why=NOT_LONGER)
    for row in turbo["combinations"]:
        row.update(frames=["first", "first+last"], seconds=[5])
    turbo["eta_s"] = {k: v for k, v in turbo["eta_s"].items() if k.endswith("/5")}
    return table


class ModelPage(PageTest):
    """For the page tests of the model choice: LightX2V Turbo on (LFSTUDIO_MODELS=sol-h3,lightx2v-turbo), with
    tests/stub/h3turbo_job.py and tests/stub/h3turbo_encode.py standing in for its job and prompt encode."""
    load_s, clip_s, models = 1, 0.5, BOTH

    def model(self, mid):
        self.page.click(f'label[for="m-{mid}"]')

    def size(self, value):
        self.page.click(f'label[for="mo-{TURBO}-resolution-{value}"]')

    def text(self, sel):
        """An element's text as it reads, its lines joined by spaces."""
        return " ".join(self.page.inner_text(sel).split())

    def greyed(self):
        """What the form shows as not offered: {name=value: the reason (its label's title)}."""
        return dict(self.page.eval_on_selector_all('#f input[aria-disabled="true"]',
                                                   "els => els.map(e => [e.name.replace(/^opt:[^:]*:/, '') + '=' + "
                                                   "e.value, e.nextElementSibling.title])"))

    def fields(self):
        """The form's own fields: mode, format, length, prompt and seed."""
        return self.page.evaluate("['task', 'orientation', 'seconds'].map(val).concat([prompt.value, seed.value])")

    def checked(self, name):
        return self.page.evaluate(f"val('{name}')")

    def sizes(self):
        """The Size row's choices as they read (hidden: None)."""
        return None if self.page.is_hidden(f'#mo-{TURBO}-resolution-576p + label') else \
            self.page.eval_on_selector_all(
                f'#mopts input[name="opt:{TURBO}:resolution"]',
                "els => els.map(e => e.nextElementSibling.innerText.split(/\\s+/).join(' '))")

    def card(self, jid):
        return f'article[data-id="{jid}"]'

    def tags(self, jid):
        return self.page.eval_on_selector_all(f"{self.card(jid)} .tags .tag", "els => els.map(e => e.textContent)")


class ModelRow(ModelPage):
    """The Model row at the top of the form, built from the studio's table: Sol-H3 (the default) and the experimental
    LightX2V Turbo with its Size row (576p, the default, or 768p). What LightX2V Turbo doesn't make yet is greyed out
    with the reason, and no field changes another: a pick it can't make disables Generate, which says why. The header,
    the times and Generate's label follow the model; the choices persist; Generate sends the model and its options."""

    def test_the_row_and_what_each_model_offers(self):
        self.assertTrue(self.page.is_visible("#msec"))
        self.assertEqual(self.page.eval_on_selector_all("#mseg input", "els => els.map(e => [e.value, e.checked])"),
                         [["sol-h3", True], [TURBO, False]])
        self.assertEqual(self.page.eval_on_selector_all("#mseg label", "els => els.map(e => e.innerText.split(/\\s+/)"
                                                                       ".join(' '))"),
                         ["Sol-H3 ~60 s", "LightX2V Turbo experimental · ~3:20 min"])
        self.assertEqual(self.page.get_attribute("#msec .lbl", "title"),
                         "The same seed makes a different clip in another model")
        self.assertEqual(self.text("#mabout"), "NVIDIA's Sol two-stage pipeline: a MiniMax-H3 draft at 384p refined "
                                               "by LTX-2.5")
        self.assertEqual((self.greyed(), self.sizes()), ({}, None))          # Sol-H3 makes everything; no options
        self.assertEqual(self.text(".brand p"), "MiniMax H3 draft → LTX-2.5 refine · 1344×768 video with sound, on "
                                                "one DGX Spark")
        # Keyframes, 9:16, 10 s, with a prompt: LightX2V Turbo makes it too (its GPU validation), and picking it
        # changes none of it
        self.pick("task", "fl2va")
        self.pick("orientation", "portrait")
        self.pick("seconds", "10")
        self.page.fill("#prompt", "keep me")
        before = self.fields()
        self.model(TURBO)
        self.assertEqual(self.fields(), before)
        self.assertEqual(self.greyed(), {"task=ref2va": NOT_REF2VA})             # Reference is all it lacks
        self.assertFalse(self.page.is_disabled("#go"))
        self.assertIsNone(self.page.get_attribute("#go", "title"))
        self.assertEqual((self.text("#go"), self.text("#err")), ("Generate · LightX2V Turbo · ~6:40 min est.", ""))
        self.assertEqual((self.text("#lenout"), self.page.is_hidden("#reloadnote")),
                         ("~6:40 min per clip, estimated", True))
        self.assertEqual(self.sizes(), ["576p 576×1024 · ~6:40 min", "768p 768×1344 · ~13:40 min"])
        self.assertEqual(self.text("#mabout"), "The full MiniMax-H3 in one stage with LightX2V's turbo LoRA: keeps "
                                               "the look of your keyframe")
        self.assertEqual(self.text("#fl .lbl small"), "one or both")            # a last frame alone too
        self.assertEqual(self.text("#loadrule"), "Each LightX2V Turbo clip loads its model itself, and closes the "
                                                 "Sol-H3 models first if they are loaded.")
        self.assertEqual(self.text(".brand p"), "LightX2V Turbo (experimental) · 576×1024 video with sound, on one "
                                                "DGX Spark")
        # every length has its estimate, per size
        for seconds, s576, s768 in (("5", "3:20", "5:50"), ("15", "11:00", "24:00")):
            self.pick("seconds", seconds)
            self.assertEqual(self.sizes(), [f"576p 576×1024 · ~{s576} min", f"768p 768×1344 · ~{s768} min"])
        self.pick("seconds", "5")
        self.assertEqual((self.text("#go"), self.text("#lenout")),
                         ("Generate · LightX2V Turbo · ~3:20 min est.", "~3:20 min per clip, estimated"))
        self.assertEqual(self.checked(f"opt:{TURBO}:resolution"), "576p")
        self.size("768p")
        self.assertEqual((self.text("#go"), self.text("#res")),
                         ("Generate · LightX2V Turbo · ~5:50 min est.", "768×1344"))
        self.pick("orientation", "landscape")
        self.assertEqual((self.text("#res"), self.sizes()), ("1344×768", ["576p 1024×576 · ~3:20 min",
                                                                          "768p 1344×768 · ~5:50 min"]))
        self.assertEqual(self.fields(), ["fl2va", "landscape", "5", "keep me", "42"])
        # Reference: greyed, picked anyway (it stays picked: no field changes another), Generate says why
        self.page.click('input[name="task"][value="ref2va"] + label', force=True)
        self.assertEqual((self.checked("task"), self.text("#err"), self.page.is_disabled("#go")),
                         ("ref2va", NOT_REF2VA, True))
        self.assertEqual(self.page.get_attribute('input[name="task"][value="ref2va"] + label', "title"), NOT_REF2VA)
        self.pick("task", "fl2va")
        self.assertEqual((self.text("#err"), self.page.is_disabled("#go")), ("", False))
        # Keyframes: a last frame alone runs (its GPU validation), and so do both
        self.page.set_input_files('input[name="last_frame"]', files=[{"name": "last.png", "mimeType": "image/png",
                                                                       "buffer": png()}])
        self.page.wait_for_timeout(300)
        self.assertEqual((self.text("#err"), self.page.is_disabled("#go")), ("", False))
        self.page.set_input_files('input[name="first_frame"]', files=[{"name": "first.png", "mimeType": "image/png",
                                                                        "buffer": png()}])
        self.page.wait_for_timeout(300)
        self.assertEqual((self.text("#err"), self.page.is_disabled("#go")), ("", False))
        # kept by this browser: the model and its size (each field on its own)
        self.page.reload()
        self.page.wait_for_timeout(500)
        self.assertEqual((self.checked("model"), self.checked(f"opt:{TURBO}:resolution"), self.text("#res")),
                         (TURBO, "768p", "1344×768"))
        self.assertEqual(self.page.evaluate("[localStorage['lfstudio.model'], "
                                            f"localStorage['lfstudio.opt.{TURBO}.resolution']]"), [TURBO, "768p"])
        self.assertEqual(self.fields(), ["fl2va", "landscape", "5", "keep me", "42"])
        # back to Sol-H3: nothing greyed, no Size row, its header and times; the Model row has both times at 10 s
        self.model("sol-h3")
        self.pick("seconds", "10")
        self.assertEqual((self.greyed(), self.sizes(), self.text("#err"), self.page.is_disabled("#go")),
                         ({}, None, "", False))
        self.assertEqual(self.text(".brand p"), "MiniMax H3 draft → LTX-2.5 refine · 1344×768 video with sound, on "
                                                "one DGX Spark")
        self.assertEqual((self.text("#go"), self.text("#lenout")), ("Generate · loads the models first",
                                                                    "~2:05 min per clip"))
        self.assertEqual(self.text("#fl .lbl small"), "one or both")
        self.assertEqual(self.page.eval_on_selector_all("#mseg label", "els => els.map(e => e.innerText.split(/\\s+/)"
                                                                       ".join(' '))"),
                         ["Sol-H3 ~2:05 min", "LightX2V Turbo experimental · ~13:40 min"])    # its 768p, at 10 s
        self.assertTrue(self.page.evaluate("document.getElementById('loadrule').textContent.startsWith('If the mode')"))
        self.assertEqual(self.page.evaluate("localStorage['lfstudio.model']"), "sol-h3")

    def test_what_a_model_does_not_make_yet(self):
        """A model's table with lengths and keyframes it doesn't make yet (LightX2V Turbo's before its GPU validation,
        put into /api/state: not_yet_table): those are greyed out with the reason, no field changes another, a pick it
        can't make disables Generate, which says why, and no time shows where the table has none."""
        table = not_yet_table()

        def with_table(route):
            response = route.fetch()
            route.fulfill(response=response, json=dict(response.json(), models=table))

        self.page.route("**/api/state", with_table)
        self.page.wait_for_function("MODELS.some(m => (m.capabilities.seconds || []).some(c => !c.offered))")
        # a pick LightX2V Turbo doesn't make yet: Keyframes, 9:16, 10 s, with a prompt; picking it changes none of it
        self.pick("task", "fl2va")
        self.pick("orientation", "portrait")
        self.pick("seconds", "10")
        self.page.fill("#prompt", "keep me")
        before = self.fields()
        self.model(TURBO)
        self.assertEqual(self.fields(), before)
        self.assertEqual(self.greyed(), {"task=ref2va": NOT_REF2VA,
                                         **{f"seconds={s}": NOT_LONGER for s in (6, 7, 8, 10, 12, 15)}})
        self.assertTrue(self.page.is_disabled("#go"))
        self.assertEqual(self.text("#go"), "Generate · not with LightX2V Turbo yet")
        self.assertEqual((self.text("#err"), self.page.get_attribute("#go", "title")), (NOT_LONGER, NOT_LONGER))
        self.assertEqual((self.text("#lenout"), self.page.is_hidden("#reloadnote")), ("", True))
        self.assertEqual(self.sizes(), ["576p 576×1024", "768p 768×1344"])       # no time at 10 s
        self.assertEqual(self.text("#mabout"), "The full MiniMax-H3 in one stage with LightX2V's turbo LoRA: keeps "
                                               "the look of your keyframe")
        self.assertEqual(self.text("#fl .lbl small"), "first or both")
        self.assertEqual(self.text("#loadrule"), "Each LightX2V Turbo clip loads its model itself, and closes the "
                                                 "Sol-H3 models first if they are loaded.")
        # a length it makes: Generate with its estimated time; the header names it with its size
        self.pick("seconds", "5")
        self.assertFalse(self.page.is_disabled("#go"))
        self.assertIsNone(self.page.get_attribute("#go", "title"))
        self.assertEqual((self.text("#go"), self.text("#err")), ("Generate · LightX2V Turbo · ~3:20 min est.", ""))
        self.assertEqual(self.text("#lenout"), "~3:20 min per clip, estimated")
        self.assertEqual(self.text(".brand p"), "LightX2V Turbo (experimental) · 576×1024 video with sound, on one "
                                                "DGX Spark")
        self.assertEqual(self.sizes(), ["576p 576×1024 · ~3:20 min", "768p 768×1344 · ~5:50 min"])
        self.assertEqual(self.checked(f"opt:{TURBO}:resolution"), "576p")
        self.size("768p")
        self.assertEqual((self.text("#go"), self.text("#res")),
                         ("Generate · LightX2V Turbo · ~5:50 min est.", "768×1344"))
        self.pick("orientation", "landscape")
        self.assertEqual((self.text("#res"), self.sizes()), ("1344×768", ["576p 1024×576 · ~3:20 min",
                                                                          "768p 1344×768 · ~5:50 min"]))
        self.assertEqual(self.fields(), ["fl2va", "landscape", "5", "keep me", "42"])
        # Reference: greyed, picked anyway (it stays picked: no field changes another), Generate says why
        self.page.click('input[name="task"][value="ref2va"] + label', force=True)
        self.assertEqual((self.checked("task"), self.text("#err"), self.page.is_disabled("#go")),
                         ("ref2va", NOT_REF2VA, True))
        self.assertEqual(self.page.get_attribute('input[name="task"][value="ref2va"] + label', "title"), NOT_REF2VA)
        self.pick("task", "fl2va")
        self.assertEqual((self.text("#err"), self.page.is_disabled("#go")), ("", False))
        # Keyframes: a last frame alone can't run yet; with a first frame it can
        self.page.set_input_files('input[name="last_frame"]', files=[{"name": "last.png", "mimeType": "image/png",
                                                                       "buffer": png()}])
        self.page.wait_for_function("t => document.getElementById('err').textContent === t", arg=NOT_LAST)
        self.assertTrue(self.page.is_disabled("#go"))
        self.page.set_input_files('input[name="first_frame"]', files=[{"name": "first.png", "mimeType": "image/png",
                                                                        "buffer": png()}])
        self.page.wait_for_function("document.getElementById('err').textContent === ''")
        self.assertFalse(self.page.is_disabled("#go"))
        # back to Sol-H3: a length LightX2V Turbo lacks is fine there, and it shows no time for it
        self.model("sol-h3")
        self.pick("seconds", "10")
        self.assertEqual((self.greyed(), self.sizes(), self.text("#err"), self.page.is_disabled("#go")),
                         ({}, None, "", False))
        self.assertEqual(self.text("#fl .lbl small"), "one or both")
        self.assertEqual(self.page.eval_on_selector_all("#mseg label", "els => els.map(e => e.innerText.split(/\\s+/)"
                                                                       ".join(' '))"),
                         ["Sol-H3 ~2:05 min", "LightX2V Turbo experimental"])       # no LightX2V Turbo time at 10 s

    def test_generate_sends_the_model_and_its_options(self):
        self.model(TURBO)
        self.size("768p")
        self.page.fill("#prompt", "a turbo clip")
        with self.page.expect_request("**/api/generate") as request:
            self.page.click("#go")
        sent = form_fields(request.value)
        self.assertEqual([f for f in sent if f[0] != "<file>"],
                         [("model", TURBO), ("task", "t2va"), ("orientation", "landscape"), ("seconds", "5"),
                          ("prompt", "a turbo clip"), ("seed", "42"), ("resolution", "768p"), ("steps", "4")])
        jid = self.studio.wait_for(lambda s: s["jobs"])["jobs"][0]["id"]
        job = self.studio.wait([jid])[0]
        self.assertEqual((job["status"], job["model"], job["model_opts"]["resolution"], job["model_opts"]["size"]),
                         ("done", TURBO, "768p", [1344, 768]))
        self.model("sol-h3")
        self.page.fill("#prompt", "a sol clip")
        with self.page.expect_request("**/api/generate") as request:
            self.page.click("#go")
        self.assertEqual([f for f in form_fields(request.value) if f[0] != "<file>"],
                         [("model", "sol-h3"), ("task", "t2va"), ("orientation", "landscape"), ("seconds", "5"),
                          ("prompt", "a sol clip"), ("seed", "42")])
        jobs = self.studio.wait_for(lambda s: len(s["jobs"]) == 2)["jobs"]
        sol = self.studio.wait([next(j["id"] for j in jobs if j["id"] != jid)])[0]
        self.assertEqual(sol["status"], "done")
        self.assertNotIn("model_opts", sol)

    def test_ctrl_enter_on_a_pick_it_cant_make(self):
        """Ctrl+Enter submits the form without Generate: a pick the model can't make is not sent either."""
        self.model(TURBO)
        self.page.click('input[name="task"][value="ref2va"] + label', force=True)
        self.page.fill("#prompt", "a reference clip")
        self.page.evaluate("document.getElementById('err').textContent = ''")
        requests = []
        self.page.on("request", lambda r: requests.append(r.url) if "/api/generate" in r.url else None)
        self.page.focus("#prompt")
        self.page.keyboard.press("Control+Enter")
        self.page.wait_for_function("t => document.getElementById('err').textContent === t", arg=NOT_REF2VA)
        self.page.wait_for_timeout(300)
        self.assertEqual(requests, [])


class ModelCards(ModelPage):
    """A LightX2V Turbo clip's card: its model in a tag of its own (none on Sol-H3's cards, which look as before), its
    Upscale button disabled with the reason, the server's file name for Download; Remix brings each clip's model and
    options back."""

    def test_cards_and_remix(self):
        sol = self.studio.generate(seed=21, prompt="a sol clip")[1]["id"]
        turbo = self.studio.turbo(seed=22, prompt="a turbo clip", orientation="portrait", resolution="768p")[1]["id"]
        self.assertEqual([j["status"] for j in self.studio.wait([sol, turbo])], ["done", "done"])
        for jid in (sol, turbo):
            self.page.wait_for_selector(f"{self.card(jid)} video")
        self.page.wait_for_timeout(2200)                                     # a poll with upscalable
        self.assertEqual(self.tags(sol), ["Text", "16:9", "5 s"])
        self.assertEqual(self.tags(turbo), ["Text", "9:16", "5 s", "LightX2V Turbo 768p"])
        up = f"{self.card(turbo)} .act.up"
        why = "Upscale of LightX2V Turbo clips comes in a later version"
        self.assertEqual((self.page.get_attribute(up, "aria-disabled"), self.page.get_attribute(up, "title")),
                         ("true", why))
        self.page.click(up, force=True)                                      # the reason, also on a phone
        self.page.wait_for_function("t => document.getElementById('toast').textContent === t", arg=why)
        self.assertTrue(self.page.is_hidden("#uppop"))
        self.assertEqual(self.page.get_attribute(f"{self.card(sol)} .act.up", "aria-disabled"), "false")
        self.assertEqual(self.page.get_attribute(f"{self.card(turbo)} a.act", "download"),
                         f"lf-studio_{TURBO}-768p_t2va_5s_seed22_{turbo[:6]}.mp4")
        self.assertEqual(self.page.get_attribute(f"{self.card(sol)} a.act", "download"), "lf-studio_t2va_5s_seed21.mp4")
        self.assertIn("incl. model load", self.page.inner_text(f"{self.card(turbo)} .when"))
        # Remix: the LightX2V Turbo clip's model, size and format come back, and are kept
        self.page.click(f"{self.card(turbo)} .reuse-all")
        self.page.wait_for_function("document.getElementById('toast').textContent === 'Settings loaded'")
        self.assertEqual([self.checked(n) for n in ("model", f"opt:{TURBO}:resolution", "task", "orientation",
                                                    "seconds")], [TURBO, "768p", "t2va", "portrait", "5"])
        took = self.page.evaluate(f"fmtS({self.studio.job(turbo)['e2e_s']})")       # measured: no longer estimated
        self.assertEqual((self.text("#res"), self.text("#go")), ("768×1344", f"Generate · LightX2V Turbo · ~{took}"))
        self.assertEqual((self.page.input_value("#prompt"), self.page.input_value("#seed")), ("a turbo clip", "22"))
        self.assertEqual(self.page.evaluate("[localStorage['lfstudio.model'], "
                                            f"localStorage['lfstudio.opt.{TURBO}.resolution']]"), [TURBO, "768p"])
        # and the Sol-H3 clip's: Sol-H3, no Size row, its header
        self.page.click(f"{self.card(sol)} .reuse-all")
        self.page.wait_for_function("document.getElementById('toast').textContent === 'Settings loaded'")
        self.assertEqual([self.checked(n) for n in ("model", "orientation")], ["sol-h3", "landscape"])
        self.assertEqual((self.sizes(), self.text("#res"), self.text("#pipe")),
                         (None, "1344×768", "MiniMax H3 draft → LTX-2.5 refine"))
        self.assertEqual(self.page.evaluate("localStorage['lfstudio.model']"), "sol-h3")
        self.assertEqual(self.page.evaluate(f"localStorage['lfstudio.opt.{TURBO}.resolution']"), "768p")  # its own


class ModelRunning(ModelPage):
    """A LightX2V Turbo clip after a Sol-H3 clip, its stand-ins slowed down: Generate's label says it closes the Sol-H3
    models first; the card and the status line say where the clip is (closing Sol-H3, encoding the prompt, loading,
    each step, decoding, saving), Unload is hidden, and the clip's time is used once measured. Cancel stops one."""

    def seed(self):
        self.studio.env.update(STUB_H3TURBO_S="21", STUB_QWEN_S="3", STUB_CLOSE_S="3")

    def watch(self, jid, until):
        """The distinct [card's line, its status tag, the status line, the status dot, Unload hidden] seen while the
        clip runs, until until(its record) holds."""
        seen, deadline = [], time.time() + 90
        while time.time() < deadline:
            row = self.page.evaluate("""id => {const c = document.querySelector(`article[data-id="${id}"]`);
                const small = c && c.querySelector('.ph2 .small');
                return [small ? small.textContent : null, c ? c.querySelector('.st').textContent : null,
                        document.getElementById('loaded').textContent, document.getElementById('status').dataset.s,
                        document.getElementById('unload').hidden];}""", jid)
            if not seen or seen[-1] != row:
                seen.append(row)
            if until(self.studio.job(jid)):
                return seen
            self.page.wait_for_timeout(150)
        self.fail(f"the clip did not get there: {seen}")

    def test_progress(self):
        sol = self.studio.generate(seed=31)[1]["id"]
        self.assertEqual(self.studio.wait([sol])[0]["status"], "done")
        self.page.wait_for_function("document.getElementById('loaded').textContent.endsWith('ready')")
        self.assertEqual(self.text("#loaded"), "Sol-H3 · Text · 16:9 · 5 s · ready")
        self.model(TURBO)
        self.assertEqual(self.text("#go"), "Generate · closes Sol-H3 first · ~4:00 min est.")
        self.page.fill("#prompt", "a slow one")
        self.page.click("#go")
        jid = next(j["id"] for j in self.studio.wait_for(lambda s: len(s["jobs"]) == 2)["jobs"] if j["id"] != sol)
        seen = self.watch(jid, lambda j: j["status"] == "done")
        status = "LightX2V Turbo · Text · 16:9 · 5 s · 576p · "
        want = [("Freeing memory · closing the Sol-H3 models", "Freeing memory", status + "closing Sol-H3"),
                ("Encoding the prompt", "Encoding", status + "encoding the prompt"),
                ("Loading LightX2V Turbo", "Loading model", status + "loading"),
                *[(f"Generating · step {i}/4", "Generating", status + f"step {i}/4") for i in (1, 2, 3, 4)],
                ("Decoding", "Generating", status + "decoding"), ("Saving", "Generating", status + "saving")]
        # each (card, its tag, the status line) with the status dot busy and no Unload, in this order
        got = [(small, st, loaded) for small, st, loaded, dot, unload in seen if small and dot == "busy" and unload]
        self.assertEqual([w for w in want if w not in got], [], seen)
        self.assertEqual(sorted(want, key=got.index), want, seen)
        self.page.wait_for_selector(f"{self.card(jid)} video")
        self.assertEqual(self.text("#loaded"), "not loaded")
        self.assertRegex(self.text(f"{self.card(jid)} .when"), r" · \d\d s incl. model load$")
        measured = self.studio.job(jid)["e2e_s"]
        self.page.wait_for_timeout(2200)                                     # a poll with the measured time
        self.assertNotIn("est.", self.text("#go"))
        self.assertEqual(self.text("#go"), "Generate · LightX2V Turbo · ~"
                         + self.page.evaluate(f"fmtS({measured})"))

    def test_meanwhile_and_cancel(self):
        jid = self.studio.turbo(seed=32, task="fl2va", orientation="portrait", resolution="768p",
                                files={"first_frame": [("first.png", png())]})[1]["id"]
        card = self.card(jid)
        self.page.wait_for_function("id => (document.querySelector(`article[data-id='${id}'] .ph2 .small`) || "
                                    "{textContent: ''}).textContent.startsWith('Generating · step')", arg=jid,
                                    timeout=60000)
        self.assertRegex(self.text("#loaded"), r"^LightX2V Turbo · Keyframes · 9:16 · 5 s · 768p · step \d/4$")
        self.assertEqual(self.tags(jid), ["Keyframes", "9:16", "5 s", "LightX2V Turbo 768p"])
        # meanwhile nothing will be loaded after it: neither model has models to close first
        self.model(TURBO)
        self.assertEqual(self.text("#go"), "Generate · LightX2V Turbo · ~3:20 min est.")
        self.model("sol-h3")
        self.assertEqual(self.text("#go"), "Generate · loads the models first")
        # a Sol-H3 clip queued after it: a LightX2V Turbo clip after that one closes its models first
        sol = self.studio.generate(seed=33)[1]["id"]
        self.page.wait_for_selector(f'{self.card(sol)}[data-status="queued"]')
        self.assertEqual(self.text(f"{self.card(sol)} .ph2 .small"), "next up")
        self.model(TURBO)
        self.assertEqual(self.text("#go"), "Generate · closes Sol-H3 first · ~4:00 min est.")
        self.assertEqual(self.page.get_attribute(f"{card} .cancel", "title"), "Stop this clip")
        self.page.click(f"{card} .cancel")
        self.page.wait_for_function("document.getElementById('toast').textContent === 'Stopping the clip'")
        self.page.wait_for_selector(f'{card}[data-status="cancelled"]', timeout=30000)
        self.assertEqual(self.text(f"{card} .ph2 .big"), "Cancelled")
        self.assertEqual(self.studio.wait([sol])[0]["status"], "done")        # the Sol-H3 clip after it runs
        self.page.wait_for_function("document.getElementById('loaded').textContent === "
                                    "'Sol-H3 · Text · 16:9 · 5 s · ready'")


class ModelOffRecord(PageTest):
    """A LightX2V Turbo clip made while the model choice was on, on a studio where it is off now: no Model row; its
    card still names its model, and Remix brings it back as Sol-H3 and says so."""
    load_s, clip_s = 1, 0.5

    def seed(self):
        (self.studio.state_dir / "jobs.json").write_text(json.dumps([
            finished_job("d" * 32, seed=8, model=TURBO, model_opts={"resolution": "576p", "steps": 4,
                                                                   "size": [1024, 576]})]))

    def test_card_and_remix(self):
        card = f'article[data-id="{"d" * 32}"]'
        self.page.wait_for_selector(card)
        self.assertTrue(self.page.is_hidden("#msec"))
        self.assertEqual(self.page.eval_on_selector_all(f"{card} .tags .tag", "els => els.map(e => e.textContent)"),
                         ["Text", "16:9", "5 s", f"{TURBO} 576p"])
        self.page.click(f"{card} .reuse-all")
        self.page.wait_for_function("t => document.getElementById('toast').textContent === t",
                                    arg=f"Settings loaded as Sol-H3 ({TURBO} is off on this studio)")
        self.assertEqual(self.page.evaluate("val('model')"), "")
        self.assertIn("loads the models first", self.go_text())


# what the page shows and sends, for ModelChoiceOffPage: the form's and the header's texts, Generate, the titles and
# aria-disabled values shown, the form's text fields and what this browser keeps
LOOK = """() => {const shown = e => e.getClientRects().length > 0, f = document.getElementById('f');
  return {form: f.innerText, header: document.querySelector('.top').innerText,
          go: document.getElementById('go').outerHTML, prompt: document.getElementById('prompt').placeholder,
          titles: [...f.querySelectorAll('[title]')].filter(shown).map(e => e.title),
          greyed: [...f.querySelectorAll('[aria-disabled]')].filter(shown).length,
          sent: [...new FormData(f).entries()].map(([k, v]) => typeof v === 'string' ? [k, v] : [k, v.name]),
          err: document.getElementById('err').textContent,
          toast: document.getElementById('toast').textContent,
          kept: Object.fromEntries(Object.keys(localStorage).sort().map(k => [k, localStorage.getItem(k)]))};}"""
GALLERY = """() => [...document.querySelectorAll('#jobs .job')].map(c => ({
  text: c.innerText, status: c.dataset.status, kind: c.dataset.kind || null,
  buttons: [...c.querySelectorAll('.acts > *')].filter(e => e.offsetWidth).map(e => [e.textContent.trim(), e.title,
            e.getAttribute('aria-disabled'), e.getAttribute('download')])}))"""


@unittest.skipUnless(sync_playwright and shutil.which("ffmpeg"), "needs playwright (webkit) and ffmpeg")
class ModelChoiceOffPage(unittest.TestCase):
    """With the model choice off (LFSTUDIO_MODELS unset, the default) the page is v0.5.1's: side by side with v0.5.1's
    own server and page (from git; skipped in a copy without that history), seeded with the same clips, in the same
    browser: the same pixels (light and dark, 1280 px and a phone), and after the same clicks (every mode, format and
    length, a chip, a Remix of a 9:16 10 s clip and of a Text one, a refused Generate, a Generate and its clip) the same
    texts, buttons and titles, the same request and the same choices kept in localStorage."""
    T0 = 1_760_000_000

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.webkit.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-page-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        old = release_server("v0.5.1", self.d / "v0.5.1-src")
        if not old:
            self.skipTest("v0.5.1 is not in this copy's git history")
        jobs = [finished_job("a" * 32, created=self.T0, seed=11, prompt="a wide one", e2e_s=61.2),
                finished_job("b" * 32, orientation="portrait", seconds=10, created=self.T0 + 60, seed=12,
                             prompt="an upright one", e2e_s=130.4, cold_start_s=101.2),
                finished_job("c" * 32, created=self.T0 + 120, seed=13, status="failed", prompt="a failed one",
                             error="RuntimeError: the worker died")]
        self.studios = {}
        for name, server in (("now", None), ("v0.5.1", old)):
            s = StubStudio(self.d / name, load_s=1, clip_s=0.5, preload=False, server=server)
            (s.state_dir / "jobs.json").write_text(json.dumps(jobs))
            self.studios[name] = s.start()
            self.addCleanup(s.stop)

    def page(self, studio, **context):
        ctx = self.browser.new_context(**({"viewport": {"width": 1280, "height": 900}} | context))
        self.addCleanup(ctx.close)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        self.addCleanup(lambda: self.assertEqual(errors, [], "errors in the page"))
        page.goto(studio.url + "/")
        page.wait_for_timeout(2500)                                          # two polls
        return page

    def both(self, fn):
        """fn(studio) on each: {name: result}; then the same, or the difference shown."""
        got = {name: fn(s) for name, s in self.studios.items()}
        self.assertEqual(got["now"], got["v0.5.1"])
        return got["now"]

    def test_the_same_pixels(self):
        for context in ({"color_scheme": "light"}, {"color_scheme": "dark"},
                        {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "device_scale_factor": 2,
                         "color_scheme": "dark"}):
            shots = {}
            for name, s in self.studios.items():
                page = self.page(s, **context)
                shots[name] = page.screenshot(full_page=True, animations="disabled")
                (self.d / f"{name}-{len(shots)}.png").write_bytes(shots[name])
            self.assertTrue(shots["now"] == shots["v0.5.1"], f"the page looks different ({context})")

    def test_the_same_behaviour(self):
        def walk(s):
            page, seen = self.page(s), []
            look = lambda what: seen.append((what, page.evaluate(LOOK)))     # noqa: E731
            look("loaded")
            seen.append(("gallery", page.evaluate(GALLERY)))
            for name, value in (("task", "fl2va"), ("orientation", "portrait"), ("seconds", "10"), ("task", "ref2va"),
                                ("seconds", "15"), ("task", "t2va"), ("orientation", "landscape"), ("seconds", "5")):
                page.click(f'input[name="{name}"][value="{value}"] + label')
                look(f"{name}={value}")
            seen.append(("the keyframes label's attributes",            # no data-all with the model choice off
                         page.evaluate("[...document.querySelector('#fl .lbl small').attributes].map(a => a.name)")))
            page.fill("#prompt", "")
            page.click("#chips .chip:nth-child(2)")
            look("a chip")
            for jid in ("b" * 32, "a" * 32):                                 # 9:16 at 10 s, then 16:9 at 5 s
                page.click(f'article[data-id="{jid}"] .reuse-all')
                page.wait_for_timeout(300)
                look(f"remix {jid[0]}")
            page.fill("#prompt", "  ")
            page.click("#go")
            look("an empty prompt")
            page.fill("#prompt", "a new clip")
            with page.expect_request("**/api/generate") as request:
                page.click("#go")
            sent = form_fields(request.value)
            jid = s.wait_for(lambda st: len(st["jobs"]) == 4)["jobs"][0]["id"]
            s.wait([jid])
            page.wait_for_selector(f'article[data-id="{jid}"] video')
            page.wait_for_timeout(2200)
            look("its clip")
            new = page.evaluate(GALLERY)[0]
            new["text"] = [line for line in new["text"].split("\n") if "seed" not in line]   # its time: left out
            page.reload()
            page.wait_for_timeout(1500)
            look("reloaded")
            return seen, sent, new

        seen, sent, new = self.both(walk)
        self.assertEqual([f for f in sent if f[0] != "<file>"], [("task", "t2va"), ("orientation", "landscape"),
                                                                ("seconds", "5"), ("prompt", "a new clip"),
                                                                ("seed", "11")])
        self.assertEqual([b[0] for b in new["buttons"]], ["Remix", "Download", "Upscale", "Delete"])
        self.assertEqual(len(seen), 17)


class MachineTexts(PageTest):
    """LFSTUDIO_BACKEND (docs/RTX5090.md): the header and the note on times name the machine, and the clip-time
    estimates follow its table. Unset = the DGX Spark, as before. tests/test_backend.py checks the table and the
    page's applyMachine without a browser."""
    backend = None

    def seed(self):
        self.studio.env.pop("LFSTUDIO_BACKEND", None)
        if self.backend:
            self.studio.env["LFSTUDIO_BACKEND"] = self.backend
            self.studio.env["LFSTUDIO_UPSCALE"] = "0"           # the experimental Upscale is the DGX Spark's only

    def test_texts(self):
        m = self.page.evaluate("MACHINES")[self.backend or "spark"]
        self.assertEqual(self.page.inner_text("#machine"), m["name"])
        self.assertEqual(self.page.inner_text("#timesnote"), "".join(m["times"]))
        self.pick("seconds", "5")
        self.assertIn(f"~{m['eta']['5']} s per clip", self.page.inner_text("#lenout"))

    def test_the_model_rows_times(self):
        """With the model choice on, Sol-H3's entry in the Model row shows the machine's time per clip, and LightX2V
        Turbo's its own (the studio's). LightX2V Turbo is the DGX Spark's only, so the table a Spark studio serves is
        put into /api/state here: the row is built from the first poll that has it."""
        table = models_table()

        def with_models(route):
            response = route.fetch()
            route.fulfill(response=response, json=dict(response.json(), models=table))

        self.page.route("**/api/state", with_models)
        self.page.wait_for_selector("#msec", state="visible")
        m = self.page.evaluate("MACHINES")[self.backend or "spark"]
        small = lambda mid: self.page.inner_text(f'label[for="m-{mid}"] small')           # noqa: E731
        for seconds in ("5", "10"):
            self.pick("seconds", seconds)
            self.assertEqual(small("sol-h3"), "~" + self.page.evaluate(f"fmtS({m['eta'][seconds]})"))
            self.assertEqual(small(TURBO), "experimental · ~" + {"5": "3:20", "10": "6:40"}[seconds] + " min")
        self.assertEqual(self.page.inner_text("#machine"), m["name"])


class MachineTextsRtx5090(MachineTexts):
    backend = "rtx5090"


if __name__ == "__main__":
    unittest.main()
