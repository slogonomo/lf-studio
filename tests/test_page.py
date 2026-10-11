"""The page (webui/index.html) in a real browser against the stub studio: no models, no GPU.

Needs Playwright with WebKit (pip install playwright && playwright install webkit) and ffmpeg; skipped otherwise.
python3 -m unittest tests.test_page -v      (or with the rest: python3 -m unittest discover -s tests)
"""
import json
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio  # noqa: E402
from test_server import finished_job, history, mp4, png  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipUnless(sync_playwright and shutil.which("ffmpeg"), "needs playwright (webkit) and ffmpeg")
class PageTest(unittest.TestCase):
    load_s, clip_s, preload = 2, 1, False

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.webkit.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="sol-h3-page-"))
        self.studio = StubStudio(self.dir, load_s=self.load_s, clip_s=self.clip_s, preload=self.preload)
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
    """#4: during a warm-up for the selected settings the button says when the clip would be ready."""
    load_s, preload = 20, True

    def seed(self):
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
    """#3: the link under Generate loads the selected settings on request."""
    load_s, preload = 6, True

    def seed(self):
        self.studio.env["STUB_CLOSE_S"] = "2"
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
        self.pick("seconds", "3")
        self.page.click("#loadnow")
        self.page.wait_for_function("document.getElementById('loaded').textContent.startsWith('switching to Text · 16:9 · 3 s')")
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


class SeedText(PageTest):
    """#15: the seed is plain text next to the time; Remix, Download and Delete share one row."""

    def test_seed_in_the_info_line_and_one_row_of_buttons(self):
        ok = self.studio.generate(prompt="a long prompt " * 30, seed=2034940401, seconds=10)[1]["id"]
        bad = self.studio.generate(prompt="STUB_FAIL", seed=2147483646)[1]["id"]
        self.studio.wait([ok, bad])           # the page was open: the cards were built while queued, then updated
        self.page.wait_for_selector(f'article[data-id="{ok}"] video')
        self.page.wait_for_selector(f'article[data-id="{bad}"][data-status="failed"]')
        for width in (1280, 390, 320):        # 320: the narrowest phone card (266 px, #19)
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.wait_for_timeout(300)
            for jid, seed, buttons in ((ok, "2034940401", ["Remix", "Download", "Delete"]),
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
    """#19: no sideways scroll on a phone, and the card buttons still share one row."""

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
            for jid, buttons in ((ok, ["Remix", "Download", "Delete"]), (bad, ["Remix", "Delete"])):
                row = self.page.eval_on_selector_all(
                    f'article[data-id="{jid}"] .acts > *', "els => els.filter(e => e.offsetWidth).map(e => "
                                                           "[e.textContent.trim(), Math.round(e.getBoundingClientRect().top)])")
                self.assertEqual([name for name, _ in row], buttons, width)
                self.assertEqual(len({top for _, top in row}), 1, f"one row at {width} px: {row}")

if __name__ == "__main__":
    unittest.main()
