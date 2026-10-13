"""The replay bench's run plan. Standard library only; nothing here starts a GPU process.

Runs are grouped into processes by format, length and canvas: Inductor compiles one static shape per process, as in
the studio's Stage-2 worker, and the shared LTX module reads the frame count at import. Each process loads Stage 2
once (~17 s), then runs its list. The first run of a process is extra and flagged `first_in_process`: it differs
from the warm runs that follow (measured 2026-10-12: PSNR 40.2 dB, SSIM 0.985 at 5 s), so bit-exact checks compare
warm runs only, and the first run is reported as the noise floor.
"""
import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import arms as armlib
from .media import sha256

LOAD_S = 17          # Stage-2 load per process, measured 16.4-16.8 s on 2026-10-12
FIRST_RUN_EXTRA_S = {768: 7, 480: 80, 384: 75}   # first run of a process: +6-7 s with a warm cache, +69-80 s new shape
STAGE2_PATH_KEYS = ("stage2_python", "ltx_root", "transformer", "refiner_lora", "output_video_vae", "audio_vae",
                    "adapter_dir", "h3_upscaler_source", "h3_upscaler_checkpoint", "prompt_cache")
# the package files Stage 2 runs (their combined hash tells two installs apart even when the patch is the same)
STAGE2_FILES = ("runtime/stage2.py", "runtime/geometry.py", "runtime/config.py", "runtime/fastload.py",
                "runtime/latent_transfer.py", "runtime/prompt_cache.py", "runtime/stage2_ops/*.py",
                "runtime/stage2_ops/h3_ltx_adapter/**/*.py", "../super_acceleration/stage2/*.py")


def parse_arms(text, repeats):
    """'baseline:2,preview' -> [('baseline', 2), ('preview', repeats)] (warm repeats per draft)."""
    out = []
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        name, _, n = part.partition(":")
        if name not in armlib.ARMS:
            raise ValueError(f"unknown arm {name!r}; bench/replay.sh arms lists them")
        try:
            count = int(n) if n else int(repeats)
        except ValueError:
            raise ValueError(f"{part}: the repeat count after ':' must be a number") from None
        if count < 1:
            raise ValueError(f"{part}: at least 1 repeat")
        if name in [o[0] for o in out]:
            raise ValueError(f"arm {name} given twice")
        out.append((name, count))
    if not out:
        raise ValueError("no arms")
    return out


def native_geometry(package, geometry, orientation, seconds):
    """The runtime's own Stage-2 geometry for a format and length: configs/default.json through orient_recipe and
    the clip-length rule, as runtime/config.py load_recipe derives it, then runtime.geometry.from_recipe."""
    recipe = json.loads((Path(package) / "configs" / "default.json").read_text())
    recipe = geometry.orient_recipe(recipe, orientation)
    if seconds != 5:
        frames = geometry.frame_geometry(geometry.draft_frames_for_seconds(seconds))
        recipe["stage1"]["frames"], recipe["output"]["frames"] = frames["draft_frames"], frames["output_frames"]
    return geometry.from_recipe(recipe), tuple(recipe["stage2"]["sigmas"])


def jsonable(geo):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in geo.items()}


def git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def stage2_code_sha256(package):
    digest, names = hashlib.sha256(), []
    root = Path(package)
    for pattern in STAGE2_FILES:
        for path in sorted(glob.glob(str(root / pattern), recursive=True)):
            rel = os.path.relpath(path, root)
            names.append(rel)
            digest.update(rel.encode() + b"\0" + Path(path).read_bytes() + b"\0")
    return digest.hexdigest(), len(names)


def package_info(package, sana, repo):
    """Which patch the installed package carries (install/10-sana.sh keeps a copy of the applied patch in the
    checkout's git directory), and a hash of the Stage-2 code it runs."""
    info = {"package": str(package), "sana": str(sana)}
    code, count = stage2_code_sha256(package)
    info.update(stage2_code_sha256=code, stage2_code_files=count)
    repo_patch = Path(repo) / "patches" / "sol-h3-studio.patch"
    info["repo_patch_sha256"] = media_sha(repo_patch)
    gitdir = git("-C", str(sana), "rev-parse", "--absolute-git-dir")
    info["sana_head"] = git("-C", str(sana), "rev-parse", "HEAD")
    copy = Path(gitdir) / "sol-h3-studio-applied.patch" if gitdir else None
    if copy is not None and copy.is_file():
        info["patch_sha256"] = media_sha(copy)
        base = copy.with_name(copy.name + ".base")
        info["patch_base"] = base.read_text().strip() if base.is_file() else None
        info["patch_still_applied"] = git("-C", str(sana), "apply", "--reverse", "--check", str(copy)) is not None
        info["patch_same_as_repo"] = info["patch_sha256"] == info["repo_patch_sha256"]
    else:
        info.update(patch_sha256=None, patch_base=None, patch_still_applied=None, patch_same_as_repo=None)
    return info


def media_sha(path):
    return sha256(path) if Path(path).is_file() else None


def stage2_paths(paths_template, task):
    """The Stage-2 entries of a task's path manifest (SOLH3_PATHS, {task} = t2va/fl2va/ref2va)."""
    manifest = Path(paths_template.replace("{task}", task)).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"no path manifest {manifest} for {task} (install/70-manifests.sh)")
    values = json.loads(manifest.read_text())
    resolved = {}
    for key in STAGE2_PATH_KEYS:
        value = values.get(key)
        if not value:
            raise ValueError(f"{manifest} has no {key}")
        path = Path(value).expanduser()
        resolved[key] = str(path if path.is_absolute() else manifest.parent / path)
    python = Path(resolved["stage2_python"])
    if not (python.is_file() and os.access(python, os.X_OK)):
        raise ValueError(f"{manifest}: stage2_python {python} is not an executable")
    return str(manifest), resolved


def build(drafts, arm_specs, *, package, geometry, paths_template, out, label="", stop_at=None, info=None):
    """The plan: one group per (format, length, canvas), each with its run list."""
    keys, groups = [], {}
    for d in drafts:
        fmt = (d["orientation"], d["seconds"])
        if fmt not in keys:
            keys.append(fmt)
    canvases = []
    for name, _ in arm_specs:
        c = armlib.ARMS[name].canvas
        if c not in canvases:
            canvases.append(c)
    canvases.sort(key=lambda c: (c != 768, -c))
    order = 0
    for canvas in canvases:
        group_arms = [(n, k) for n, k in arm_specs if armlib.ARMS[n].canvas == canvas]
        for orientation, seconds in keys:
            members = [d for d in drafts if (d["orientation"], d["seconds"]) == (orientation, seconds)]
            native, recipe_sigmas = native_geometry(package, geometry, orientation, seconds)
            for d in members:
                if list(native["h3"]) != list(d["h3_shape"]):
                    raise ValueError(f"draft {d['name']}: H3 shape {d['h3_shape']} is not the runtime's "
                                     f"{list(native['h3'])} for {orientation} {seconds} s")
            tasks = []
            for d in members:
                if d["task"] not in tasks:
                    tasks.append(d["task"])
            manifest, s2paths = stage2_paths(paths_template, tasks[0])
            for t in tasks[1:]:
                if stage2_paths(paths_template, t)[1] != s2paths:
                    raise ValueError(f"the {tasks[0]} and {t} manifests name different Stage-2 files: "
                                     "replay those drafts separately")
            geo = armlib.canvas_geometry(native, canvas)
            order += 1
            gid = f"{order}-{orientation}-{seconds}s-{geo['output_width']}x{geo['output_height']}"
            arm_settings = {n: armlib.settings(n, recipe_sigmas, native) for n, _ in group_arms}
            runs = []

            def add_run(d, arm_name, repeat, first=False):
                # the first run of a process also compiles (or loads) its shapes: FIRST_RUN_EXTRA_S, so that
                # --stop-at counts it
                est = armlib.ARMS[arm_name].estimate_s(native["output_frames"]) + (FIRST_RUN_EXTRA_S[canvas]
                                                                                   if first else 0)
                runs.append({"draft": d["name"], "arm": arm_name, "repeat": repeat, "first_in_process": first,
                             "dir": f"runs/{gid}/{d['name']}/{arm_name}/r{repeat}", "est_s": round(est, 1)})
            add_run(members[0], group_arms[0][0], 0, first=True)
            for rep in range(1, max(k for _, k in group_arms) + 1):
                for d in members:
                    for n, k in group_arms:
                        if rep <= k:
                            add_run(d, n, rep)
            est = LOAD_S + sum(r["est_s"] for r in runs)
            groups[gid] = {"id": gid, "orientation": orientation, "seconds": seconds, "canvas": canvas,
                           "task": tasks[0], "tasks": tasks, "paths_manifest": manifest,
                           "stage2_python": s2paths["stage2_python"], "geometry": jsonable(geo),
                           "native_geometry": jsonable(native), "recipe_sigmas": list(recipe_sigmas),
                           "arms": arm_settings, "drafts": [d["name"] for d in members], "runs": runs,
                           "est_s": round(est)}
    return {"schema": 1, "label": label, "out": str(out), "package": str(package), "stop_at": stop_at,
            "package_info": info or {}, "drafts": {d["name"]: d for d in drafts},
            "arm_specs": [list(a) for a in arm_specs], "groups": list(groups.values()),
            "est_s": sum(g["est_s"] for g in groups.values())}


def shape(t):
    return "(" + ",".join(str(x) for x in t) + ")"


def describe(plan):
    """The --dry printout: package, drafts, then per process its shapes, arms and runs."""
    lines = []
    pi = plan.get("package_info") or {}
    lines.append(f"package {plan['package']}")
    if pi:
        patch = pi.get("patch_sha256")
        lines.append(f"  patch {patch[:12] if patch else 'unknown (no applied copy in the checkout)'}"
                     + (f", same as this checkout's patches/sol-h3-studio.patch: "
                        f"{'yes' if pi.get('patch_same_as_repo') else 'no'}" if patch else "")
                     + f"; Stage-2 code {pi.get('stage2_code_sha256', '')[:12]} ({pi.get('stage2_code_files')} files)")
    lines.append(f"drafts ({len(plan['drafts'])}):")
    from .drafts import describe as describe_draft
    for d in plan["drafts"].values():
        lines.append("  " + describe_draft(d))
    for i, g in enumerate(plan["groups"], 1):
        geo = g["geometry"]
        lines.append(f"process {i}/{len(plan['groups'])} [{g['id']}]: {g['orientation']} {g['seconds']} s, output "
                     f"{geo['output_width']}x{geo['output_height']}x{geo['output_frames']}, manifest of {g['task']}, "
                     f"~{g['est_s'] / 60:.1f} min (estimated)")
        lines.append(f"  shapes: H3 {shape(geo['h3'])} -> upscaled {shape(geo['h3_upscaled'])} -> adapter "
                     f"{shape(geo['adapter_output'])} -> refiner {shape(geo['refiner_input'])}, "
                     f"{armlib.video_tokens(geo)} video tokens")
        for name, s in g["arms"].items():
            t = s["tiles"]
            tiles = "default" if s["tiles_default"] else ""
            tiles += f" h {t['height'][0]}/{t['height'][1]} w {t['width'][0]}/{t['width'][1]}"
            lines.append(f"  arm {name}: {s['desc']}")
            lines.append(f"      sigmas {'/'.join(f'{x:g}' for x in s['sigmas'])} ({s['steps']} steps), "
                         f"upscale x{s['upscale_scale']:g}, decoder tiles {tiles.strip()}"
                         + (f", profiles step {s['profile_step']}" if s["profile_step"] is not None else ""))
            for r in s["relaxes"]:
                lines.append(f"      relaxes: {r}")
        for r in g["runs"]:
            lines.append(f"  run {r['draft']} {r['arm']} r{r['repeat']}"
                         + (" (first in process: the noise floor; includes the first-run compile)"
                            if r["first_in_process"] else "")
                         + f" ~{r['est_s']:.0f} s")
    n = sum(len(g["runs"]) for g in plan["groups"])
    lines.append(f"total: {n} runs in {len(plan['groups'])} processes, ~{plan['est_s'] / 60:.0f} min of GPU time "
                 "(estimated from the 2026-10-12 bench)")
    return "\n".join(lines)
