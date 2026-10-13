"""The replay bench's quality step, on the CPU after a run: what changed against the baseline, and how it looks.

  - SSIM and PSNR (ffmpeg, every frame) of each arm against the same draft's baseline, at the draft canvas size
    (672x384, or 384x672 for portrait; lanczos). Also against the `preview` arm when the run has it (how close the
    arm stays to the Stage-1 draft), the first run of each process against the warm run of the same draft and arm
    (the noise floor), and the studio's own clip against the baseline. These numbers measure change, not quality: a
    lower score can look as good or better.
  - LPIPS (optional, --lpips PYTHON): the same pairs, every 2nd frame, AlexNet, CPU (lpips_score.py).
  - Contact sheets per draft: 3 frames per arm, the middle frame's centre at full size, and with a mouth crop
    (`add --mouth X:Y:W:H`) a strip of that crop at 5 times. Side-by-side MP4s: baseline left, arm right, with the
    arm's audio.
  - SyncNet (optional, --syncnet CMD) on every dialogue draft: replay.syncnet.
Writes <run>/quality/: quality.csv, quality.md, sheets/, syncnet/.
"""
import csv
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from . import media, syncnet
from .report import load_rows, ok, warm

FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def picks(rows, plan):
    """{draft: {arm: row}}: each arm's first warm PASS run; and the first-in-process runs."""
    chosen, first = {}, []
    for r in sorted(rows, key=lambda r: r.get("repeat", 0)):
        if warm(r):
            chosen.setdefault(r["draft"], {}).setdefault(r["arm"], r)
        elif ok(r) and r.get("request_index") == 0:
            first.append(r)
    order = [a for a, _ in plan.get("arm_specs", [])]
    return ({d: dict(sorted(arms.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
             for d, arms in chosen.items()}, first)


def pairs(plan, chosen, first):
    """(draft, reference label, reference mp4, test label, test mp4) to compare."""
    out = []
    for name, arms in chosen.items():
        base = arms.get("baseline")
        if base:
            for arm, r in arms.items():
                if arm != "baseline":
                    out.append((name, "baseline", base["output"], arm, r["output"]))
            d = plan["drafts"][name]
            if d.get("studio"):
                out.append((name, "baseline", base["output"], "studio clip", str(Path(d["dir"]) / d["studio"]["mp4"])))
        prev = arms.get("preview")
        if prev:
            for arm, r in arms.items():
                if arm != "preview" and r["settings"]["output"] == prev["settings"]["output"]:
                    out.append((name, "preview", prev["output"], arm, r["output"]))
    for r in first:
        twin = chosen.get(r["draft"], {}).get(r["arm"])
        if twin:
            out.append((r["draft"], f"{r['arm']} warm", twin["output"], f"{r['arm']} first in process", r["output"]))
    return out


def lpips_pair(python, ref, test, size):
    script = Path(__file__).with_name("lpips_score.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    r = subprocess.run([python, str(script), "--size", f"{size[0]}x{size[1]}", ref, test],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"lpips_score.py failed: {r.stderr.strip()[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def label(text):
    font = next((f for f in FONTS if Path(f).is_file()), None)
    if font is None:
        return ""
    safe = "".join(c for c in text if c.isalnum() or c in " ._-x")
    return (f",drawtext=fontfile={font}:text='{safe}':x=8:y=8:fontsize=20:fontcolor=white:box=1:"
            "boxcolor=black@0.6")


def ffmpeg(*args):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", media.FFMPEG_THREADS, *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.strip()[-400:]}")


def sheets(name, arms, sheet_dir, mouth=None):
    """Contact sheets and side-by-sides for one draft (ffmpeg only)."""
    sheet_dir.mkdir(parents=True, exist_ok=True)
    tmp = sheet_dir / f".{name}"
    tmp.mkdir(exist_ok=True)
    items = list(arms.items())
    first = media.probe(items[0][1]["output"])
    n = first["frames"]
    portrait = first["height"] > first["width"]
    small = (256, 448) if portrait else (448, 256)
    common = (384, 672) if portrait else (672, 384)
    full = (first["width"], first["height"])
    marks = [int(n * f) for f in (0.1, 0.5, 0.9)]
    made, parts = [], {"overview": [], "detail": [], "mouth": []}
    for arm, r in items:
        src, tag = r["output"], f"{arm} {r.get('stage2_s') or 0:.1f} s"
        sel = "+".join(f"eq(n\\,{m})" for m in marks)
        o = tmp / f"{arm}_overview.png"
        ffmpeg("-i", src, "-vf", f"select='{sel}',scale={small[0]}:{small[1]}:flags=lanczos,tile=3x1{label(tag)}",
               "-frames:v", "1", str(o))
        parts["overview"].append(o)
        d = tmp / f"{arm}_detail.png"
        cw, ch = full[0] // 2, full[1] // 2
        ffmpeg("-i", src, "-vf", f"select='eq(n\\,{marks[1]})',scale={full[0]}:{full[1]}:flags=lanczos,"
               f"crop={cw}:{ch}:{cw // 2}:{ch // 2}{label(tag)}", "-frames:v", "1", str(d))
        parts["detail"].append(d)
        if mouth:
            x, y, w, h = mouth.split(":")
            ms = "+".join(f"eq(n\\,{int(n * f)})" for f in (0.2, 0.35, 0.5, 0.65, 0.8))
            m = tmp / f"{arm}_mouth.png"
            ffmpeg("-i", src, "-vf", f"select='{ms}',scale={full[0]}:{full[1]}:flags=lanczos,crop={w}:{h}:{x}:{y},"
                   f"tile=5x1{label(tag)}", "-frames:v", "1", str(m))
            parts["mouth"].append(m)
        if arm != "baseline" and "baseline" in arms:
            side = sheet_dir / f"{name}_{arm}_vs_baseline.mp4"
            ffmpeg("-i", arms["baseline"]["output"], "-i", src, "-filter_complex",
                   f"[0:v]scale={common[0]}:{common[1]}:flags=lanczos{label('baseline')}[a];"
                   f"[1:v]scale={common[0]}:{common[1]}:flags=lanczos{label(arm)}[b];[a][b]hstack=inputs=2[v]",
                   "-map", "[v]", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-c:a", "copy", "-movflags", "+faststart", str(side))
            made.append(side)
    for kind, files in parts.items():
        if not files:
            continue
        target = sheet_dir / f"{name}_{kind}.jpg"
        if len(files) == 1:
            ffmpeg("-i", str(files[0]), "-frames:v", "1", "-q:v", "3", str(target))
        else:
            inputs = [x for f in files for x in ("-i", str(f))]
            ffmpeg(*inputs, "-filter_complex", "".join(f"[{i}:v]" for i in range(len(files)))
                   + f"vstack=inputs={len(files)}", "-frames:v", "1", "-q:v", "3", str(target))
        made.append(target)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return made


FAILURES = (RuntimeError, OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError)


def check_tools(lpips_python=None, syncnet_command=None):
    """Refuse up front a --lpips or --syncnet that cannot run (before minutes of SSIM and sheets)."""
    problems = []
    if lpips_python and not shutil.which(lpips_python):
        problems.append(f"--lpips {lpips_python}: not an executable")
    if syncnet_command is not None:
        try:
            words = shlex.split(syncnet_command)
        except ValueError as error:
            words = None
            problems.append(f"--syncnet: {error}")
        if words == []:
            problems.append("--syncnet is empty")
        elif words and not shutil.which(words[0]):
            problems.append(f"--syncnet {words[0]}: not an executable")
    if problems:
        raise ValueError("; ".join(problems))


def quality(out, lpips_python=None, syncnet_command=None, log=print):
    """Returns (quality dir, complete): complete is False when a comparison, a sheet or the SyncNet table failed or
    is incomplete, None when nothing needed SyncNet and nothing failed. quality.md is written in every case."""
    check_tools(lpips_python, syncnet_command)
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    rows = load_rows(out)
    chosen, first = picks(rows, plan)
    qdir = out / "quality"
    qdir.mkdir(exist_ok=True)
    results, errors = [], []
    for name, ref_label, ref, test_label, test in pairs(plan, chosen, first):
        row = {"draft": name, "reference": ref_label, "test": test_label, "size": "", "psnr_db": None, "ssim": None,
               "lpips_mean": None, "lpips_max": None, "error": "", "reference_file": ref, "test_file": test}
        try:
            size = media.common_size(ref)
            row["size"] = f"{size[0]}x{size[1]}"
            row["psnr_db"], row["ssim"] = media.psnr_ssim(ref, test, size)
            if lpips_python:
                lp = lpips_pair(lpips_python, ref, test, size)
                row.update(lpips_mean=round(lp["mean"], 4), lpips_max=round(lp["max"], 4))
        except FAILURES as error:
            row["error"] = f"{type(error).__name__}: {error}"[:300]
            errors.append(f"{name} {test_label} vs {ref_label}: {row['error']}")
        results.append(row)
        log(f"[quality] {name}: {test_label} vs {ref_label}: PSNR {row['psnr_db']} dB, SSIM {row['ssim']}"
            + (f", LPIPS {row['lpips_mean']}" if lpips_python else "") + (f" ({row['error']})" if row["error"] else ""))
    with open(qdir / "quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]) if results else ["draft"])
        w.writeheader()
        w.writerows(results)
    made = []
    for name, arms in chosen.items():
        try:
            made += sheets(name, arms, qdir / "sheets", plan["drafts"][name].get("mouth"))
        except FAILURES as error:
            errors.append(f"{name} sheets: {type(error).__name__}: {error}"[:300])
    lines = [f"# Quality checks for {out.name}", "",
             "SSIM/PSNR/LPIPS measure how much the picture changed against the reference, not quality.", "",
             "| draft | test | vs | size | PSNR dB | SSIM | LPIPS mean (max) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lp = "" if r["lpips_mean"] is None else f"{r['lpips_mean']:.4f} ({r['lpips_max']:.4f})"
        psnr = "" if r["psnr_db"] is None else ("identical" if r["psnr_db"] == float("inf") else f"{r['psnr_db']:.2f}")
        ssim = "" if r["ssim"] is None else f"{r['ssim']:.4f}"
        if r["error"]:
            psnr, ssim, lp = "failed", "", r["error"][:80]
        lines.append(f"| {r['draft']} | {r['test']} | {r['reference']} | {r['size']} | {psnr} | {ssim} | {lp} |")
    lines += ["", f"Sheets and side-by-sides: {len(made)} files in quality/sheets/."]
    dialogue = {n: arms for n, arms in chosen.items() if plan["drafts"][n].get("dialogue")}
    complete = None
    if syncnet_command and dialogue:
        try:
            _, complete = syncnet.score(syncnet_command, qdir / "syncnet",
                                        {n: {a: r["output"] for a, r in arms.items()} for n, arms in dialogue.items()},
                                        plan["drafts"])
            lines += ["", "## SyncNet (each face over its own speaking turns)", "",
                      (qdir / "syncnet" / "syncnet.md").read_text()]
        except FAILURES as error:
            complete = False
            errors.append(f"SyncNet: {type(error).__name__}: {error}"[:300])
    elif dialogue:
        lines += ["", f"SyncNet not run (no --syncnet): {len(dialogue)} dialogue draft(s)."]
    if errors:
        lines += ["", "## Failed", ""] + [f"- {e}" for e in errors]
        complete = False
    (qdir / "quality.md").write_text("\n".join(lines) + "\n")
    return qdir, complete
