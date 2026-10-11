#!/usr/bin/env python3
"""Browser test of the Sol-H3 Studio page (Playwright; WebKit by default, close to Safari).

usage: ui_test.py [URL] [--browser webkit|chromium] [--shots DIR] [--job]

Default URL http://127.0.0.1:8190/. Without --job nothing is submitted, so no GPU work starts; with --job one real
t2va clip is generated with the currently loaded (or default) settings and the test checks that a playing video and
the form survive the polling while it runs (takes minutes). Fixture images and videos are generated in a temp
directory under the screenshot directory and removed at exit (PNG written by this script; MP4 by ffmpeg if installed). Screenshots go to --shots or a new temp
directory. Do not commit screenshots that show generated clips.
Needs: pip install playwright && playwright install webkit (or chromium).
"""
import atexit
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

from playwright.sync_api import sync_playwright

ARGS = sys.argv[1:]
URL = next((a for a in ARGS if a.startswith("http")), "http://127.0.0.1:8190/")
RUN_JOB = "--job" in ARGS
BROWSER = ARGS[ARGS.index("--browser") + 1] if "--browser" in ARGS else "webkit"
SHOTS = Path(ARGS[ARGS.index("--shots") + 1]) if "--shots" in ARGS else Path(tempfile.mkdtemp(prefix="sol-h3-ui-"))
SHOTS.mkdir(parents=True, exist_ok=True)
FIX = Path(tempfile.mkdtemp(prefix="fixtures-", dir=SHOTS))   # removed at exit
atexit.register(shutil.rmtree, FIX, True)
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    ok &= bool(cond)


def skip(msg):
    print("SKIP " + msg, flush=True)


def write_png(path, width=96, height=64):
    """A small gradient PNG, standard library only."""
    rows = b"".join(b"\x00" + bytes(c for x in range(width) for c in (x * 255 // width, y * 255 // height, 128))
                    for y in range(height))
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return path


def write_mp4(path):
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=3",
                        "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
        return path
    except (OSError, subprocess.SubprocessError):
        return None


pick = lambda pg, name, value: pg.click(f'input[name="{name}"][value="{value}"] + label')
chosen = lambda pg, name: pg.evaluate(f"document.querySelector('input[name={name}]:checked').value")

img = write_png(FIX / "fixture.png")
vid = write_mp4(FIX / "fixture.mp4")
print(f"url {URL} | browser {BROWSER} | screenshots {SHOTS}")

with sync_playwright() as p:
    b = getattr(p, BROWSER).launch()
    for scheme in ("dark", "light"):
        ctx = b.new_context(viewport={"width": 1440, "height": 1000}, color_scheme=scheme)
        pg = ctx.new_page()
        pg.goto(URL)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(SHOTS / f"ui_{scheme}.png"))
        ctx.close()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, color_scheme="dark", is_mobile=True,
                        device_scale_factor=2)
    pg = ctx.new_page()
    pg.goto(URL)
    pg.wait_for_timeout(2000)
    check(pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "phone width: no horizontal scroll")
    pg.screenshot(path=str(SHOTS / "ui_phone.png"), full_page=False)
    ctx.close()

    ctx = b.new_context(viewport={"width": 1400, "height": 1000})
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL)
    pg.wait_for_timeout(2500)
    check(pg.title() == "Sol-H3 Studio", f"page title ({pg.title()!r})")
    loaded = pg.text_content("#loaded")
    print("loaded shows:", loaded)
    check("·" in loaded or "not loaded" in loaded, "loaded mode/format/length is shown")

    # the header resolution follows the format; fields are independent of each other
    pick(pg, "orientation", "landscape")
    check(pg.text_content("#res") == "1344×768", "header shows 1344×768 for 16:9")
    pick(pg, "orientation", "portrait")
    check(pg.text_content("#res") == "768×1344", "header shows 768×1344 for 9:16")
    pick(pg, "seconds", "5")
    pg.fill("#prompt", "my own prompt")
    for t in ("fl2va", "ref2va", "t2va"):
        pick(pg, "task", t)
        check(chosen(pg, "orientation") == "portrait" and chosen(pg, "seconds") == "5",
              f"format/length unchanged after mode -> {t}")
        vis = {s: pg.is_visible(s) for s in ("#fl", "#ref")}
        check(vis == {"#fl": t == "fl2va", "#ref": t == "ref2va"}, f"only the {t} inputs are shown {vis}")
    check(pg.input_value("#prompt") == "my own prompt", "prompt untouched by mode changes")
    key = pg.evaluate("state.session.key")
    print("session key:", key)
    if key:
        pick(pg, "task", key[0]); pick(pg, "orientation", key[1]); pick(pg, "seconds", str(key[2] or 5))
        go = pg.text_content("#go")
        check("load" not in go.lower() and not pg.is_visible("#reloadnote"),
              "no reload warning when everything matches what's loaded")
        pick(pg, "orientation", "portrait" if key[1] == "landscape" else "landscape")
        go = pg.text_content("#go")
        check("reloads" in go.lower() and pg.is_visible("#reloadnote") and pg.text_content("#reloadtext"),
              "Generate warns about a reload when the format differs")
        busy = pg.evaluate("state.jobs.some(active)")
        check(pg.is_visible("#loadnow") != busy, "'Load selected now' offered only while nothing is queued or running")
        pick(pg, "orientation", key[1])
    else:
        skip("reload-warning checks (no session loaded)")
    pg.reload()
    pg.wait_for_timeout(2000)
    check(pg.input_value("#prompt") == "my own prompt", "choices persist across a page reload")

    # examples and section chips
    pick(pg, "task", "ref2va")
    pg.fill("#prompt", "")
    pg.click("#chips .chip.ex")
    check(pg.input_value("#prompt").startswith("subject_definitions:"), "Example fills the H3 reference format")
    pick(pg, "task", "fl2va"); pick(pg, "seconds", "10")
    pg.fill("#prompt", "")
    pg.click("#chips .chip.ex")
    check("10.00-second mark" in pg.input_value("#prompt"), "fl2va example aligns the last frame with the length")
    pg.fill("#prompt", "x")
    pg.click("#prompt")
    pg.keyboard.press("End")
    pg.click("#chips .chip:nth-child(3)")
    check("integrated_multimodal_description:" in pg.input_value("#prompt"), "section chip inserts a header")

    # drop zones: previews appear, clear works
    pick(pg, "task", "ref2va")
    pg.set_input_files("input[name=ref_images]", str(img))
    check(pg.evaluate("document.querySelector('input[name=ref_images]').closest('.drop')"
                      ".querySelectorAll('.thumbs img').length") == 1, "image preview shown")
    pg.click("input[name=ref_images] ~ .clear", force=True)
    check(pg.evaluate("document.querySelector('input[name=ref_images]').files.length") == 0, "clear removes the file")
    if vid:
        pg.set_input_files("input[name=ref_videos]", str(vid))
        check(pg.evaluate("document.querySelector('input[name=ref_videos]').closest('.drop')"
                          ".querySelectorAll('.thumbs video').length") == 1, "video preview shown")
        pg.screenshot(path=str(SHOTS / "ui_ref.png"))
        pg.click("input[name=ref_videos] ~ .clear", force=True)
    else:
        skip("video preview (ffmpeg not found)")

    # seed
    s0 = pg.input_value("#seed")
    pg.click("#roll")
    s1 = pg.input_value("#seed")
    check(s1 != s0 and s1.isdigit(), f"Roll changes the seed ({s0} -> {s1})")

    # finished clips: players with posters; a playing video survives polling
    n_done = pg.evaluate("document.querySelectorAll('.media video').length")
    t0 = 0.0
    if n_done:
        check(pg.evaluate("[...document.querySelectorAll('.media video')].every(v => v.poster.includes('/poster/'))"),
              "players use poster frames")
        pg.evaluate("""() => { const v=document.querySelector('.media video'); v.muted=true; v.playbackRate=0.1;
                               v.__mark='keep'; v.play(); document.getElementById('f').__mark='keep'; }""")
        for _ in range(12):
            pg.wait_for_timeout(1000)
            t0 = pg.evaluate("[...document.querySelectorAll('video')].find(x=>x.__mark==='keep').currentTime")
            if t0 > 0.05:
                break
        check(t0 > 0.05, f"a finished clip plays (t={t0:.2f}s at 0.1x)")
        pg.wait_for_timeout(4500)          # two polls
        alive = pg.evaluate("(() => { const v=[...document.querySelectorAll('video')].find(x=>x.__mark==='keep');"
                            " return !!v && v.isConnected && !v.paused && document.getElementById('f').__mark==='keep'; })()")
        check(alive, "polling keeps the playing video and the form (no rebuild)")
    else:
        skip("player checks (no finished clips on this server yet)")

    if RUN_JOB:
        if not n_done:
            skip("--job needs a finished clip to watch; generate one first")
        else:
            run_key = key or ["t2va", "landscape", 5]
            pick(pg, "task", run_key[0]); pick(pg, "orientation", run_key[1]); pick(pg, "seconds", str(run_key[2] or 5))
            pg.check("#autoroll")
            pg.fill("#prompt", "integrated_multimodal_description: A static wide shot of a harbor at dawn; a crane "
                               "slowly turns. overall_soundscape: Gulls, water lapping, a metallic creak.")
            pg.click("#go")
            pg.wait_for_timeout(1500)
            used = pg.input_value("#seed")
            print("seed submitted:", used, "| error:", pg.text_content("#err"))
            start, last_t, stalls, st = time.time(), -1.0, 0, {}
            while time.time() - start < 900:
                pg.wait_for_timeout(5000)
                st = pg.evaluate("""() => { const v=[...document.querySelectorAll('video')].find(x=>x.__mark==='keep');
                      const top=document.querySelector('.job');
                      return {alive: !!v && v.isConnected, paused: v ? v.paused : null, t: v ? v.currentTime : -1,
                              form: document.getElementById('f').__mark==='keep', status: top.dataset.status,
                              seed: top.querySelector('.seedv').textContent}; }""")
                if not st["alive"] or st["paused"] or not st["form"] or st["t"] <= last_t:
                    stalls += 1
                last_t = st["t"]
                if st["status"] in ("done", "failed"):
                    break
            check(stalls == 0, "the playing video and the form survived the whole job")
            check(st.get("status") == "done", f"job finished: {st.get('status')}")
            check(st.get("seed") == used, f"the card shows the seed used ({st.get('seed')})")
    check(not errs, f"no page errors {errs[:2]}")
    pg.screenshot(path=str(SHOTS / "ui_end.png"), full_page=False)
    b.close()
print("screenshots:", SHOTS)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
