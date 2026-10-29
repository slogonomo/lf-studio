#!/usr/bin/env python3
"""Browser test of the LF Studio page (Playwright; WebKit by default, close to Safari).

usage: ui_test.py [URL] [--browser webkit|chromium] [--shots DIR] [--job]

Default URL http://127.0.0.1:8190/. Without --job nothing is submitted, so no GPU work starts; with --job one real
t2va clip is generated with the currently loaded (or default) settings and the test checks that a playing video and
the form survive the polling while it runs (takes minutes). Fixture images and videos are generated in a temp
directory under the screenshot directory and removed at exit (PNG written by this script; MP4 by ffmpeg if installed). Screenshots go to --shots or a new temp
directory. Do not commit screenshots that show generated clips.
Needs: pip install playwright && playwright install webkit (or chromium).
"""
import atexit
import re
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
SHOTS = Path(ARGS[ARGS.index("--shots") + 1]) if "--shots" in ARGS else Path(tempfile.mkdtemp(prefix="lf-studio-ui-"))
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
    for width in (320, 390):                       # 320: the narrowest common phone (#19)
        ctx = b.new_context(viewport={"width": width, "height": 844}, color_scheme="dark", is_mobile=True,
                            device_scale_factor=2)
        pg = ctx.new_page()
        pg.goto(URL)
        pg.wait_for_timeout(2000)
        check(pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"),
              f"phone width {width} px: no horizontal scroll")
        rows = pg.evaluate("""() => [...document.querySelectorAll('.job')].map(c => new Set([...c.querySelectorAll('.acts > *')]
                                .filter(e => e.offsetWidth).map(e => Math.round(e.getBoundingClientRect().top))).size)""")
        check(all(n <= 1 for n in rows), f"phone width {width} px: card buttons on one row ({len(rows)} cards)")
        pg.screenshot(path=str(SHOTS / f"ui_phone_{width}.png"), full_page=False)
        ctx.close()

    ctx = b.new_context(viewport={"width": 1400, "height": 1000})
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL)
    pg.wait_for_timeout(2500)
    check(pg.title() == "LF Studio", f"page title ({pg.title()!r})")
    loaded = pg.text_content("#loaded")
    print("loaded shows:", loaded)
    check("·" in loaded or "not loaded" in loaded, "loaded mode/format/length is shown")

    # the model choice (LFSTUDIO_MODELS with more than one model): picking a model changes no other field; what the
    # selected model doesn't make yet is greyed out with the reason as its title; a pick it can't make disables
    # Generate, and #err says why. It ends on the model it started with.
    if pg.is_visible("#msec"):
        models = pg.evaluate("[...document.querySelectorAll('#mseg input')].map(i => i.value)")
        first = chosen(pg, "model")
        fields = lambda: pg.evaluate("['task', 'orientation', 'seconds'].map(val).concat([prompt.value])")  # noqa: E731
        before = fields()
        print("models:", models, "| picked:", first)
        for m in models + [first]:
            pg.click(f'label[for="m-{m}"]')
            check(fields() == before, f"fields unchanged after model -> {m}")
            greyed = pg.evaluate("[...document.querySelectorAll('#f input[aria-disabled=true]')]"
                                 ".map(i => i.nextElementSibling.title)")
            check(all(greyed), f"model {m}: what it doesn't make yet is greyed out with the reason ({len(greyed)})")
            why = pg.evaluate("cantRun()")
            check(pg.is_disabled("#go") == bool(why) and (not why or pg.text_content("#err") == why),
                  f"model {m}: Generate {'is off and says why' if why else 'is on'} ({why or 'it makes this pick'})")
    else:
        skip("model choice checks (one model on this studio)")

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
    reshape = pg.evaluate("!!state.reshape")    # LFSTUDIO_RESHAPE=1 (default since 0.4): same mode, no reload
    print("session key:", key, "| reshape:", reshape)
    if key:
        pick(pg, "task", key[0]); pick(pg, "orientation", key[1]); pick(pg, "seconds", str(key[2] or 5))
        go = pg.text_content("#go")
        check("load" not in go.lower() and not pg.is_visible("#reloadnote"),
              "no reload warning when everything matches what's loaded")
        other_orientation = "portrait" if key[1] == "landscape" else "landscape"
        other_seconds = "10" if (key[2] or 5) != 10 else "5"
        if reshape:
            # another format or length keeps the models loaded: no reload warning and no 'Load selected now'
            for name, value, what in (("orientation", other_orientation, "format"),
                                      ("seconds", other_seconds, "length")):
                pick(pg, name, value)
                go = pg.text_content("#go")
                check("same models" in go and "load" not in go.lower() and not pg.is_visible("#reloadnote")
                      and not pg.is_visible("#loadnow"), f"Generate keeps the models when the {what} differs")
                pick(pg, name, key[1] if name == "orientation" else str(key[2] or 5))
            # another mode reloads them
            other_task = next((t for t in pg.evaluate("state.warmable || []") if t != key[0]),
                              next(t for t in ("t2va", "fl2va", "ref2va") if t != key[0]))
            pick(pg, "task", other_task)
            what = "the mode"
        else:
            pick(pg, "orientation", other_orientation)
            other_task, what = key[0], "the format"
        go = pg.text_content("#go")
        check("reloads" in go.lower() and pg.is_visible("#reloadnote") and pg.text_content("#reloadtext"),
              f"Generate warns about a reload when {what} differs")
        busy = pg.evaluate("state.jobs.some(active)")
        offered = not busy and pg.evaluate(
            f"(state.warmable || []).includes('{other_task}') && "
            f"(!Array.isArray(state.tasks_ready) || state.tasks_ready.includes('{other_task}'))")
        check(pg.is_visible("#loadnow") == offered,
              "'Load selected now' offered only while nothing is queued or running, for a mode it can warm up")
        pick(pg, "task", key[0]); pick(pg, "orientation", key[1])
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

    # card buttons: Delete on finished clips (done, failed, cancelled), Cancel only on queued or running ones
    cards = pg.evaluate("[...document.querySelectorAll('.job')].map(c => [c.dataset.status, "
                        "!c.querySelector('.cancel').hidden, !c.querySelector('.del').hidden])")
    finished = ("done", "failed", "cancelled")
    check(all(dl and not cancel for s, cancel, dl in cards if s in finished)
          and not any(dl for s, cancel, dl in cards if s not in finished),
          f"Cancel only on queued or running cards, Delete only on finished ones ({len(cards)} cards)")
    # the experimental Upscale (LFSTUDIO_UPSCALE=1): its button only on finished clips, and never on an upscale's own
    # card, which has no Remix either; none at all with it off
    on = pg.evaluate("!!state.upscale")
    ups = pg.evaluate("[...document.querySelectorAll('.job')].map(c => [c.dataset.status, c.dataset.kind || '', "
                      "!!c.querySelector('.act.up:not([hidden])'), !!c.querySelector('.reuse-all')])")
    check(all(not up or (on and s == "done" and not kind) for s, kind, up, remix in ups)
          and all(not remix for s, kind, up, remix in ups if kind == "upscale"),
          f"Upscale only on finished clips with it on ({'on' if on else 'off'}), no Remix or Upscale on an upscale "
          f"({sum(kind == 'upscale' for _, kind, _, _ in ups)} upscales)")
    # a disabled Upscale button says why (a clip it can't take yet, as one longer than 5 s); clips of another model
    # than the studio's own name it in a tag of their own
    reasons = pg.evaluate("[...document.querySelectorAll('.job .act.up[aria-disabled=true]:not([hidden])')]"
                          ".map(b => b.title)")
    check(all(reasons), f"a disabled Upscale button says why ({len(reasons)} disabled)")
    tags = pg.evaluate("[...document.querySelectorAll('.job .tag.model')].map(t => t.textContent)")
    check(all(tags), f"clips of another model show it in a tag ({len(tags)}: {sorted(set(tags))})")
    # durations are rounded to whole seconds before they are split (#26): "2:00 min", never "1:60 min"
    times = pg.evaluate("[...document.querySelectorAll('#go, #lenout, #loadtimer, .job .when, .job .ph2')]"
                        ".map(e => e.textContent)")
    bad = [t for t in times if re.search(r"\d:[6-9]\d\b", t)]
    check(not bad, f"no time on the page shows 60 seconds or more ({len(times)} labels) {bad[:2]}")
    pics = pg.evaluate("[...document.querySelectorAll('.qimg')].filter(i => i.offsetWidth)"
                       ".map(i => i.closest('.job').dataset.status)")
    check(not any(s in finished for s in pics), f"a clip's picture only on queued or running cards ({len(pics)} shown)")

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
                if st["status"] in finished:
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
