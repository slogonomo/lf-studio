"""SyncNet lip-sync scores for the replay bench's dialogue drafts: runs an external scorer, builds the table.

The scorer is an optional dependency and is not part of this repository. The original SyncNet code (MIT) is at
https://github.com/joonson/syncnet_python; its download_model.sh fetches the SyncNet and face-detector weights.
Any scorer works that takes this command line and writes this JSON:

  SCORER VIDEO... --json OUT.json [--turns 'R:0-1.2,L:2.4-4.7']
  OUT.json: [{"file": VIDEO, "turns": [{"speaker", "t0", "t1"}, ...],
              "faces": {FACE: null | {"det_rate": 0-1, "speaking": {"offset", "lse_d", "lse_c"} | null}}}, ...]

LSE-C (confidence, higher = better sync) and LSE-D (distance, lower = better) are SyncNet's usual scores, taken over
each face's own speaking turns. A turn is speaker:start-end in seconds; the speaker names the face.

Turns are a property of the draft, not of the arm: Stage 2 muxes the draft's own audio into every output, so every
arm of a draft has the same speech at the same times. They are stored with the draft (`add --turns`). A draft
without turns gets them once, from the scorer's own detection on the baseline output, and all arms are then scored
with those, so the turn boundaries do not depend on the video being scored.

The table is built from a manifest of the files passed to the scorer (file -> draft, arm), not parsed back out of
file names: that parse is what failed with an IndexError for arm names like A3_480 in the first bench.
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

from .drafts import parse_turns


def turns_spec(turns):
    return ",".join(f"{t['speaker']}:{round(float(t['t0']), 2):g}-{round(float(t['t1']), 2):g}" for t in turns)


def run_scorer(command, files, json_out, turns=None, log=None):
    cmd = shlex.split(command) + [str(f) for f in files] + ["--json", str(json_out)]
    if turns:
        cmd += ["--turns", turns]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    Path(json_out).unlink(missing_ok=True)              # never read an earlier run's results
    with open(log or os.devnull, "a") as stream:
        r = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0 or not Path(json_out).is_file():
        raise RuntimeError(f"SyncNet scorer failed (exit {r.returncode}); see {log}")
    return json.loads(Path(json_out).read_text())


def table(results, manifest):
    """rows [{draft, arm, face, offset, lse_d, lse_c, det_rate, status}] for every (draft, arm) in the manifest.
    manifest: {file path as passed to the scorer: (draft, arm)}. A pair the scorer gave nothing for is 'missing'."""
    by_file = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        by_file[str(r.get("file"))] = r
        by_file[os.path.realpath(str(r.get("file")))] = r
    rows = []
    for path, (draft, arm) in manifest.items():
        r = by_file.get(str(path)) or by_file.get(os.path.realpath(str(path)))
        if r is None:
            error = Path(path).parent / "error.txt"
            why = error.read_text().strip()[:200] if error.is_file() else "no result from the scorer"
            rows.append({"draft": draft, "arm": arm, "face": "", "status": f"missing: {why}"})
            continue
        faces = r.get("faces") if isinstance(r.get("faces"), dict) else {}
        if not faces:
            rows.append({"draft": draft, "arm": arm, "face": "", "status": "missing: no face in the result"})
        for face in sorted(faces):
            f = faces[face]
            if not f:
                rows.append({"draft": draft, "arm": arm, "face": face, "status": "face not found"})
                continue
            s = f.get("speaking")
            if not s:
                rows.append({"draft": draft, "arm": arm, "face": face, "det_rate": f.get("det_rate"),
                             "status": "no speaking turn for this face"})
                continue
            rows.append({"draft": draft, "arm": arm, "face": face, "offset": s.get("offset"),
                         "lse_d": s.get("lse_d"), "lse_c": s.get("lse_c"), "det_rate": f.get("det_rate"),
                         "status": "ok"})
    return rows


def complete(rows, manifest):
    """True when every (draft, arm) in the manifest has at least one scored face."""
    scored = {(r["draft"], r["arm"]) for r in rows if r["status"] == "ok"}
    return all(pair in scored for pair in manifest.values())


def markdown(rows, manifest):
    num = lambda x, d=2: "" if x is None else (f"{x:+d}" if isinstance(x, int) else f"{x:.{d}f}")
    lines = ["| draft | arm | face | offset (frames) | LSE-D (lower = better) | LSE-C (higher = better) | "
             "face found in | status |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        det = r.get("det_rate")
        lines.append(f"| {r['draft']} | {r['arm']} | {r.get('face', '')} | {num(r.get('offset'))} | "
                     f"{num(r.get('lse_d'))} | {num(r.get('lse_c'))} | "
                     f"{'' if det is None else f'{100 * det:.0f}% of frames'} | {r['status']} |")
    means = {}
    for r in rows:
        if r["status"] == "ok":
            means.setdefault(r["arm"], []).append(r)
    if means:
        lines += ["", "| arm | faces | mean LSE-C | mean LSE-D |", "|---|---|---|---|"]
        for arm, rs in means.items():
            lines.append(f"| {arm} | {len(rs)} | {sum(r['lse_c'] for r in rs) / len(rs):.2f} | "
                         f"{sum(r['lse_d'] for r in rs) / len(rs):.2f} |")
    lines.append("")
    lines.append("Complete: every dialogue draft and arm has a score." if complete(rows, manifest) else
                 "INCOMPLETE: some draft and arm pairs have no score (status column).")
    return "\n".join(lines) + "\n"


def score(command, work, picks, drafts):
    """picks: {draft: {arm: mp4}} for the dialogue drafts. Writes work/<draft>.json, syncnet.md/.json; returns
    (rows, complete)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "scorer.log"
    manifest, results = {}, []
    for name, arms in picks.items():
        d = drafts[name]
        folder = work / name
        folder.mkdir(exist_ok=True)
        (folder / "error.txt").unlink(missing_ok=True)
        files = {}
        for arm, mp4 in arms.items():
            link = folder / f"{name}__{arm}.mp4"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(mp4)
            files[arm] = link
            manifest[str(link)] = (name, arm)
        spec = d.get("turns")
        source = "stored with the draft"
        try:
            if not spec:
                if "baseline" not in files:
                    raise RuntimeError("no stored turns and no baseline output to detect them on")
                detected = run_scorer(command, [files["baseline"]], folder / "turns-detect.json", log=log)
                spec = turns_spec(detected[0].get("turns") or [])
                parse_turns(spec)
                source = "detected on the baseline output"
            (folder / "turns.txt").write_text(f"{spec}\n{source}\n")
            results += run_scorer(command, list(files.values()), work / f"{name}.json", turns=spec, log=log)
        except (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError) as error:
            # this draft stays unscored ("missing", with the reason); the others go on
            (folder / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
    rows = table(results, manifest)
    (work / "syncnet.json").write_text(json.dumps({"rows": rows, "manifest": manifest}, indent=1))
    (work / "syncnet.md").write_text(markdown(rows, manifest))
    return rows, complete(rows, manifest)
