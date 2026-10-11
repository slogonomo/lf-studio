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
from test_server import finished_job, png  # noqa: E402

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
        self.page.wait_for_timeout(800)
        jobs = self.studio.state()["jobs"]
        new = next(j for j in jobs if j["id"] != a)
        self.assertEqual([Path(new[k]).name for k in ("first_frame", "last_frame")], ["first.png", "last.png"])
        self.assertEqual(Path(new["first_frame"]).parent.name, new["id"])
        # x clears the zone: nothing kept, so the next Generate is refused as having no frame
        self.page.click("#fl .drop:first-of-type .clear")
        self.page.click("#fl .drops > .drop:last-of-type .clear")
        self.page.click("#go")
        self.page.wait_for_function("document.getElementById('err').textContent.includes('first frame')")

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


if __name__ == "__main__":
    unittest.main()
