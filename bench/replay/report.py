"""The replay bench's report: output hashes, decoded-frame MD5s, the "same as" columns and summary.md. CPU only.

Reads <run>/plan.json and <run>/runs.jsonl, writes <run>/runs.csv, <run>/hashes.json (so a second report is quick)
and <run>/summary.md.

Columns that compare outputs (on decoded video frames and audio samples, so container bytes do not matter):
  same_as_baseline  yes/no against the same draft's first warm baseline run ("ref" on that run itself)
  same_as_studio    baseline runs only: the output SHA-256 against the studio's clip of that draft. Only warm runs
                    are compared with a warm studio clip: the first run in a process differs (see plan.py).
"""
import csv
import json
import re
import statistics
from pathlib import Path

from . import media

PHASES = {"upscale_s": "h3_latent_upscale_s", "adapter_s": "h3_ltx_adapter_s", "audio_enc_s": "audio_vae_encode_s",
          "state_prep_s": "multimodal_state_prepare_s", "denoise_s": "joint_video_audio_stage2_s",
          "decode_s": "official_video_vae_decode_s", "mux_s": "h264_encode_mux_s", "verify_s": "output_verify_s"}
CSV_FIELDS = ["group", "draft", "arm", "repeat", "request_index", "first_in_process", "status", "stage2_s", "wall_s",
              *PHASES, "other_s", "step_s", "peak_alloc_gib", "peak_reserved_gib", "min_memavail_gib", "output_size",
              "sigmas", "steps", "tiles", "relaxed_checks", "stage2_audio", "stage2_keyframe", "patch_sha256",
              "stage2_code_sha256", "output", "output_sha256", "video_md5", "audio_md5", "same_as_baseline",
              "same_as_studio", "started", "error"]


def load_rows(out):
    path = Path(out) / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def ok(row):
    return row.get("status") == "PASS" and row.get("output")


def warm(row):
    return ok(row) and row.get("request_index") not in (None, 0)


def hash_outputs(out, rows, plan):
    """sha256 + decoded-frame MD5s of every PASS output and of each draft's studio clip, cached in hashes.json."""
    path = Path(out) / "hashes.json"
    cache = json.loads(path.read_text()) if path.is_file() else {}
    files = [r["output"] for r in rows if ok(r)]
    for d in plan["drafts"].values():
        if d.get("studio"):
            files.append(str(Path(d["dir"]) / d["studio"]["mp4"]))
    for f in files:
        if f in cache or not Path(f).is_file():
            continue
        cache[f] = {"sha256": media.sha256(f), "video_md5": media.stream_md5(f, "v"), "audio_md5": media.stream_md5(f, "a")}
        path.write_text(json.dumps(cache, indent=1))
    return cache


def compare(rows, plan, hashes):
    """Adds video_md5/audio_md5/same_as_baseline/same_as_studio to each row. Returns per-draft facts."""
    facts = {}
    for name, d in plan["drafts"].items():
        base = [r for r in rows if r["draft"] == name and r["arm"] == "baseline" and warm(r)]
        base.sort(key=lambda r: r["repeat"])
        ref = hashes.get(base[0]["output"]) if base else None
        studio = d.get("studio")
        studio_hash = hashes.get(str(Path(d["dir"]) / studio["mp4"])) if studio else None
        facts[name] = {"ref": base[0] if base else None, "studio": studio, "studio_hash": studio_hash}
    for r in rows:
        h = hashes.get(r.get("output") or "", {})
        r["video_md5"], r["audio_md5"] = h.get("video_md5"), h.get("audio_md5")
        f = facts.get(r["draft"], {})
        ref = f.get("ref")
        if not ok(r) or ref is None:
            r["same_as_baseline"] = ""
        elif r is ref:
            r["same_as_baseline"] = "ref"
        else:
            rh = hashes.get(ref["output"], {})
            same = (h.get("video_md5"), h.get("audio_md5")) == (rh.get("video_md5"), rh.get("audio_md5"))
            r["same_as_baseline"] = "yes" if same else "no"
        r["same_as_studio"] = studio_verdict(r, f.get("studio"), f.get("studio_hash"), h) if r["arm"] == "baseline" else ""
    return facts


def studio_verdict(row, studio, studio_hash, row_hash):
    if not ok(row) or not studio or not studio_hash:
        return ""
    if row.get("stage2_audio") != studio.get("stage2_audio"):
        return f"n/a: Stage-2 audio {row.get('stage2_audio')} here, {studio.get('stage2_audio')} in the studio"
    here, there = row.get("stage2_keyframe") or "off", studio.get("stage2_keyframe") or "off"
    if here != there:
        return f"n/a: keyframe {here} here, {there} in the studio"
    if not warm(row) or not studio.get("warm"):
        return ("n/a: first run in this process" if not warm(row) else
                "n/a: the studio clip was the first in its process")
    if row_hash.get("sha256") == studio_hash.get("sha256"):
        return "yes"
    same_frames = (row_hash.get("video_md5"), row_hash.get("audio_md5")) == (studio_hash.get("video_md5"),
                                                                              studio_hash.get("audio_md5"))
    return "same frames, different file" if same_frames else "no"


def flat(row):
    phases = row.get("phases_s") or {}
    out = {k: row.get(k) for k in CSV_FIELDS if k in row}
    named = {}
    for col, key in PHASES.items():
        value = phases.get(key)
        named[col] = round(value, 3) if isinstance(value, (int, float)) else None
    out.update(named)
    total = row.get("stage2_s")
    if isinstance(total, (int, float)):
        out["stage2_s"] = round(total, 3)
        out["other_s"] = round(total - sum(v for v in named.values() if v is not None), 3)
    s = row.get("settings") or {}
    out["output_size"] = "x".join(str(v) for v in s.get("output", []))
    out["sigmas"] = "/".join(f"{x:g}" for x in s.get("sigmas", []))
    out["steps"] = s.get("steps")
    t = s.get("tiles") or {}
    out["tiles"] = " ".join(f"{k[0]}{v[0]}/{v[1]}" for k, v in t.items())
    out["relaxed_checks"] = len(row.get("relaxes") or [])
    out["step_s"] = " ".join(str(x) for x in row.get("step_s") or [])
    out["first_in_process"] = row.get("request_index") == 0 if row.get("request_index") is not None else ""
    return out


def write_csv(out, rows):
    with open(Path(out) / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(flat(r))


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def fmt(x, digits=1):
    return "" if x is None else f"{x:.{digits}f}"


def memguard_lows(out):
    lows = {}
    for log in sorted((Path(out) / "logs").glob("*.log")):
        hits = re.findall(r"MEMGUARD: exit=(-?\d+) min_mem_avail_gib=(\d+)", log.read_text(errors="replace"))
        killed = "MEMGUARD: MemAvailable" in log.read_text(errors="replace")
        if hits:
            lows[log.stem] = {"exit": int(hits[-1][0]), "min_gib": int(hits[-1][1]), "killed": killed}
    return lows


def gpu_clocks(out):
    path = Path(out) / "gpu_clocks.csv"
    if not path.is_file():
        return None
    sm, temp, power, reasons, busy = [], [], [], {}, 0
    for line in path.read_text(errors="replace").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10 or not p[1].endswith("MHz"):
            continue
        try:
            util = float(p[5].split()[0])
        except ValueError:
            continue
        if util < 50:
            continue
        busy += 1
        sm.append(float(p[1].split()[0]))
        temp.append(float(p[3]))
        try:
            power.append(float(p[4].split()[0]))
        except ValueError:
            pass
        for name, value in zip(("active", "sw_thermal", "hw_slowdown", "sw_power_cap"), p[6:10]):
            if value not in ("0x0000000000000000", "Not Active"):
                reasons[f"{name} {value}"] = reasons.get(f"{name} {value}", 0) + 1
    if not sm:
        return None
    q = lambda xs: f"{min(xs):.0f} / {statistics.median(xs):.0f} / {max(xs):.0f}" if xs else "-"
    return (f"{busy} samples at >= 50% utilization (1 Hz): SM clock {q(sm)} MHz (min / median / max), temperature "
            f"{q(temp)} C, power {q(power)} W; throttle reasons (samples): "
            + (", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) or "none"))


def summarize(out, plan, rows, facts):
    pi = plan.get("package_info") or {}
    procs = [json.loads(line) for line in (Path(out) / "processes.jsonl").read_text().splitlines()
             if line.strip()] if (Path(out) / "processes.jsonl").is_file() else []
    lines = [f"# Stage-2 replay {Path(out).name}", ""]
    lines.append(f"Package `{plan['package']}`; patch {(pi.get('patch_sha256') or 'unknown')[:12]} (same as the "
                 f"checkout's: {pi.get('patch_same_as_repo')}); Stage-2 code {(pi.get('stage2_code_sha256') or '')[:12]}; "
                 f"compile cache: {(plan.get('compile_cache') or {}).get('note', 'not recorded')}.")
    if procs:
        lines.append("Processes: " + "; ".join(f"{p['group']} loaded in {p['load_s']} s ({p['stage2_audio']} audio, "
                                                f"keyframe {p.get('stage2_keyframe', 'off')}, torch {p.get('torch')})"
                                                for p in procs) + ".")
    lows = memguard_lows(out)
    if lows:
        lines.append("Memory guard: " + "; ".join(f"{k} exit {v['exit']}, lowest MemAvailable {v['min_gib']} GiB"
                                                  + (" (KILLED by the guard)" if v["killed"] else "")
                                                  for k, v in lows.items()) + ".")
    clocks = gpu_clocks(out)
    if clocks:
        lines.append(f"GPU: {clocks}.")
    lines += ["", "## Timing (warm runs, medians; the first run of each process is listed below, not here)", "",
              "| draft | arm | output | steps | n | Stage 2 s | denoise s | per step s | decode s | mux s | "
              "Δ vs baseline s (%) | peak alloc GiB | lowest avail GiB | same as baseline | same as studio |",
              "|" + "---|" * 15]
    order = [a for a, _ in plan.get("arm_specs", [])]
    for name in plan["drafts"]:
        base_med = median([r["stage2_s"] for r in rows if r["draft"] == name and r["arm"] == "baseline" and warm(r)])
        for arm in order:
            rs = [r for r in rows if r["draft"] == name and r["arm"] == arm]
            if not rs:
                continue
            w = [r for r in rs if warm(r)]
            bad = [r for r in rs if r.get("status") != "PASS"]
            tot = median([r["stage2_s"] for r in w])
            steps = [x for r in w for x in r.get("step_s") or []]
            ph = lambda key: median([(r.get("phases_s") or {}).get(key) for r in w])
            delta = ""
            if tot is not None and base_med and arm != "baseline":
                delta = f"{tot - base_med:+.1f} ({100 * (tot - base_med) / base_med:+.0f}%)"
            same = sorted({r.get("same_as_baseline") for r in w if r.get("same_as_baseline")})
            studio = sorted({r.get("same_as_studio") for r in w if r.get("same_as_studio")})
            s = rs[0].get("settings") or {}
            cells = [name, arm, "x".join(str(v) for v in s.get("output", [])), str(s.get("steps", "")), str(len(w)),
                     fmt(tot), fmt(ph("joint_video_audio_stage2_s")), fmt(median(steps), 2),
                     fmt(ph("official_video_vae_decode_s")), fmt(ph("h264_encode_mux_s"), 2), delta,
                     fmt(median([r.get("peak_alloc_gib") for r in w])),
                     fmt(min([r["min_memavail_gib"] for r in w if r.get("min_memavail_gib") is not None], default=None)),
                     "/".join(same), "/".join(studio)]
            if bad:
                cells[-1] = (cells[-1] + " " if cells[-1] else "") + "; ".join(
                    f"{r['status']} r{r['repeat']}: {(r.get('error') or '')[:120]}" for r in bad)
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Baseline by format and length (median of each draft's warm median)", "",
              "| format | length | drafts | warm runs | Stage 2 s |", "|---|---|---|---|---|"]
    by_fmt = {}
    for name, d in plan["drafts"].items():
        w = [r["stage2_s"] for r in rows if r["draft"] == name and r["arm"] == "baseline" and warm(r)]
        if w:
            by_fmt.setdefault((d["orientation"], d["seconds"]), []).append((name, statistics.median(w), len(w)))
    for (orientation, seconds), items in sorted(by_fmt.items()):
        lines.append(f"| {orientation} | {seconds} s | {', '.join(n for n, _, _ in items)} | "
                     f"{sum(k for _, _, k in items)} | {statistics.median(m for _, m, _ in items):.1f} |")
    lines += ["", "## Bit-exact checks (decoded video frames and audio samples)", ""]
    for name in plan["drafts"]:
        for arm in order:
            w = [r for r in rows if r["draft"] == name and r["arm"] == arm and warm(r)]
            if len(w) >= 2:
                same = len({(r.get("video_md5"), r.get("audio_md5")) for r in w}) == 1
                lines.append(f"- {name} {arm}: {len(w)} warm runs, "
                             f"{'all identical' if same else 'NOT identical'}")
    for r in rows:
        if ok(r) and r.get("request_index") == 0:
            twin = [x for x in rows if x["draft"] == r["draft"] and x["arm"] == r["arm"] and warm(x)]
            verdict = ("no warm run of the same draft and arm to compare" if not twin else
                       "identical to the warm run" if (twin[0].get("video_md5"), twin[0].get("audio_md5")) ==
                       (r.get("video_md5"), r.get("audio_md5")) else
                       "differs from the warm run (the noise floor: PSNR/SSIM in quality/)")
            lines.append(f"- first run of {r['group']} ({r['draft']} {r['arm']}, Stage 2 {fmt(r.get('stage2_s'))} s): "
                         f"{verdict}")
    for name, f in facts.items():
        studio = f.get("studio")
        if not studio:
            lines.append(f"- {name}: no studio clip to compare")
            continue
        verdicts = sorted({r.get("same_as_studio") for r in rows if r["draft"] == name and r["arm"] == "baseline"
                           and warm(r) and r.get("same_as_studio")})
        lines.append(f"- {name} baseline vs the studio's clip ({'warm' if studio.get('warm') else 'first in its process'}"
                     f", Stage 2 {fmt(studio.get('stage2_s'))} s in the studio): {', '.join(verdicts) or 'no baseline run'}")
    audio = {}
    for r in rows:
        if ok(r) and r.get("audio_md5"):
            audio.setdefault(r["draft"], set()).add(r["audio_md5"])
    for name, values in audio.items():
        lines.append(f"- {name}: audio {'identical in every run' if len(values) == 1 else 'DIFFERS between runs'}"
                     " (Stage 2 muxes the draft's own PCM)")
    relaxed = {}
    for g in plan["groups"]:
        for arm, s in g["arms"].items():
            if s["relaxes"]:
                relaxed[arm] = s["relaxes"]
    if relaxed:
        lines += ["", "## Runtime checks the arms relax (all others stay in force)", ""]
        for arm, items in relaxed.items():
            lines.append(f"- {arm}: " + "; ".join(items))
    failed = [r for r in rows if r.get("status") not in ("PASS", "SKIPPED")]
    skipped = [r for r in rows if r.get("status") == "SKIPPED"]
    planned = sum(len(g["runs"]) for g in plan["groups"])
    lines += ["", f"{sum(1 for r in rows if ok(r))} of {planned} planned runs passed, {len(failed)} failed, "
              f"{len(skipped)} skipped, {planned - len(rows)} not reached."]
    return "\n".join(lines) + "\n"


def report(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    rows = load_rows(out)
    hashes = hash_outputs(out, rows, plan)
    facts = compare(rows, plan, hashes)
    pi = plan.get("package_info") or {}
    for r in rows:
        r.update(patch_sha256=pi.get("patch_sha256"), stage2_code_sha256=pi.get("stage2_code_sha256"))
    write_csv(out, rows)
    text = summarize(out, plan, rows, facts)
    (out / "summary.md").write_text(text)
    planned = sum(len(g["runs"]) for g in plan["groups"])
    all_ok = len(rows) == planned and all(ok(r) for r in rows)
    return text, all_ok
