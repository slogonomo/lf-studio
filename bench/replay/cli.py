"""bench/replay.sh's commands (standard library only). Run it through bench/replay.sh, which reads the settings file.

The GPU work happens in replay/stage2.py processes that `run` starts under bench/memguard-run.sh; everything else here
runs on the CPU.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import arms as armlib
from . import drafts as draftlib
from . import plan as planlib
from . import quality as qualitylib
from . import report as reportlib

BENCH = Path(__file__).resolve().parent.parent
# runtime/pipeline.py worker_environment("stage2"): the studio's Stage-2 worker settings (stage2.py checks them)
STUDIO_WORKER_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                     "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CLOCKS_QUERY = ("timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,"
                "clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,"
                "clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_power_cap")


def setting(name, default=None):
    value = os.environ.get(name)
    if value is None and default is None:
        sys.exit(f"replay: {name} is not set; run this through bench/replay.sh")
    return value if value is not None else default


def runtime_root():
    return Path(setting("SOL_H3_SPARK_RUNTIME_ROOT"))


def mem_avail_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def descendants(pid):
    children = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    out, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            out.append(child)
            todo.append(child)
    return out


def stop_tree(pid):
    """SIGTERM to every process group under pid (memguard puts the GPU process in its own session), SIGKILL 15 s on."""
    own = os.getpgrp()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        groups = set()
        for p in [pid] + descendants(pid):
            try:
                groups.add(os.getpgid(p))
            except OSError:
                pass
        for g in groups - {own}:
            try:
                os.killpg(g, sig)
            except OSError:
                pass
        if sig == signal.SIGTERM:
            for _ in range(15):
                if not [p for p in [pid] + descendants(pid) if Path(f"/proc/{p}").exists()]:
                    return
                time.sleep(1)


def copy_compile_cache(out):
    """A scratch copy of the Stage-2 compile cache, so a run never writes the studio's (runtime/config.py
    compile_cache_root: SOL_H3_COMPILE_CACHE, else <runtime root>/cache/compile)."""
    source = os.environ.get("SOL_H3_COMPILE_CACHE") or str(runtime_root() / "cache" / "compile")
    if source == "session":
        return "session", {"source": "session", "note": "per-session (SOL_H3_COMPILE_CACHE=session): cold compiles"}
    target = out / "compile"
    src = Path(source) / "stage2"
    if src.is_dir():
        shutil.copytree(src, target / "stage2", symlinks=True)
        files = sum(1 for p in (target / "stage2").rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in (target / "stage2").rglob("*") if p.is_file())
        note = f"copy of {src} ({files} files, {size / 2**30:.2f} GiB)"
    else:
        (target / "stage2").mkdir(parents=True)
        note = f"{src} does not exist: started empty, every shape compiles on its first run"
    return str(target), {"source": str(src), "copy": str(target), "note": note}


def group_env(cache):
    env = dict(os.environ)
    for key in ("SOL_H3_SECONDS", "SOL_H3_ORIENTATION", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(key, None)
    env.update(STUDIO_WORKER_ENV)
    env.setdefault("MIMALLOC_PURGE_DELAY", "0")
    if cache != "session":
        env.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    env["SOL_H3_COMPILE_CACHE"] = cache
    env["PYTHONPATH"] = os.pathsep.join([setting("SOLH3_PKG"), str(BENCH)])
    return env


def cmd_add(a):
    store = draftlib.store_dir(runtime_root())
    store.mkdir(parents=True, exist_ok=True)
    geometry = draftlib.load_geometry(setting("SOLH3_PKG"))
    info = draftlib.add(store, a.name, a.case_dir, geometry, dialogue=a.dialogue, turns=a.turns, mouth=a.mouth,
                        note=a.note or "", force=a.force)
    print("added " + draftlib.describe(info))
    print(f"  in {store / a.name}")
    if info["studio"] and not info["studio"]["sha256_matches_receipt"]:
        print("  note: the studio clip's SHA-256 is not the one in its stage2_result.json (converted or edited?)")
    return 0


def cmd_list(a):
    store = draftlib.store_dir(runtime_root())
    items = draftlib.all_drafts(store)
    print(f"{len(items)} draft(s) in {store}")
    for d in items:
        print("  " + draftlib.describe(d))
    return 0


def cmd_arms(a):
    for name, arm in armlib.ARMS.items():
        canvas = "the draft's 2x canvas" if armlib.CANVASES[arm.canvas] is None else \
            f"{armlib.CANVASES[arm.canvas][1]}x{armlib.CANVASES[arm.canvas][0]}"
        print(f"{name:20s} {canvas:22s} ~{arm.est_s:4.1f} s at 5 s  {arm.desc}")
    return 0


def make_plan(a, out):
    store = draftlib.store_dir(runtime_root())
    names = [n for n in (a.drafts or "").split(",") if n] or [d["name"] for d in draftlib.all_drafts(store)]
    if not names:
        raise SystemExit(f"replay: no drafts in {store}; add some with bench/replay.sh add NAME CASE_DIR")
    selected = [draftlib.load(store, n) for n in names]
    package = setting("SOLH3_PKG")
    geometry = draftlib.load_geometry(package)
    info = planlib.package_info(package, setting("SOLH3_SANA"), setting("SOLH3_REPO"))
    return planlib.build(selected, planlib.parse_arms(a.arms or "baseline", a.repeats), package=package,
                         geometry=geometry, paths_template=setting("SOLH3_PATHS"), out=out, label=a.label or "",
                         stop_at=a.stop_at, info=info)


def cmd_run(a):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = runtime_root() / "outputs" / "bench" / (f"replay-{stamp}" + (f"-{a.label}" if a.label else ""))
    try:
        plan = make_plan(a, out)
    except (ValueError, FileNotFoundError) as error:
        print(f"replay: {error}", file=sys.stderr)
        return 2
    print(planlib.describe(plan))
    if a.dry:
        print("dry run: nothing was started or written")
        return 0
    need = float(setting("SOLH3_MIN_FREE_GIB", "110"))
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    print(f"run directory {out}")
    cache, plan["compile_cache"] = copy_compile_cache(out)
    print(f"compile cache: {plan['compile_cache']['note']}")
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=1))
    env = group_env(cache)
    current, stopping = {"pid": None}, {"signal": None}

    def on_signal(signum, _frame):
        stopping["signal"] = signum
        if current["pid"]:
            stop_tree(current["pid"])
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    clocks = clocks_file = None
    if shutil.which("nvidia-smi") and os.environ.get("CUDA_VISIBLE_DEVICES") != "":   # 1 Hz clocks, temperature
        clocks_file = open(out / "gpu_clocks.csv", "w")
        clocks = subprocess.Popen(["nvidia-smi", f"--query-gpu={CLOCKS_QUERY}", "--format=csv", "-l", "1"],
                                  stdout=clocks_file, stderr=subprocess.DEVNULL)
    exits = {}
    try:
        for g in plan["groups"]:
            if stopping["signal"]:
                break
            if a.stop_at and time.time() + planlib.LOAD_S + g["runs"][0]["est_s"] > a.stop_at:
                print(f"[replay] {g['id']}: not started, it would pass --stop-at")
                continue
            waited = 0
            while need and mem_avail_gib() < need and waited < 180 and not stopping["signal"]:
                print(f"[replay] MemAvailable {mem_avail_gib():.0f} GiB < {need:.0f}: waiting")
                time.sleep(10)
                waited += 10
            if need and mem_avail_gib() < need:
                print(f"[replay] MemAvailable stayed under {need:.0f} GiB: no further process started")
                break
            cmd = [str(BENCH / "memguard-run.sh"), str(out / "logs" / f"{g['id']}.log"), "--"]
            if a.stop_at:
                cmd += ["timeout", "-k", "15", str(int(max(60, a.stop_at + 180 - time.time())))]
            cmd += [g["stage2_python"], "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                    "--module", "replay.stage2", "--plan", str(plan_path), "--group", g["id"]]
            print(f"[replay] process {g['id']}: {len(g['runs'])} runs, ~{g['est_s'] / 60:.1f} min; log "
                  f"{out / 'logs' / (g['id'] + '.log')}", flush=True)
            p = subprocess.Popen(cmd, cwd=plan["package"], env=env, start_new_session=True)
            current["pid"] = p.pid
            exits[g["id"]] = p.wait()
            current["pid"] = None
            print(f"[replay] process {g['id']}: exit {exits[g['id']]}", flush=True)
    finally:
        if clocks is not None:
            clocks.terminate()
            clocks.wait()
            clocks_file.close()
        if not a.keep_cache and cache != "session":
            shutil.rmtree(out / "compile", ignore_errors=True)
    text, all_ok = reportlib.report(out)
    print(text)
    print(f"summary: {out / 'summary.md'}; rows: {out / 'runs.csv'}. CPU quality checks: "
          f"bench/replay.sh quality {out}")
    if stopping["signal"]:
        return 128 + stopping["signal"]
    return 0 if all_ok and all(v == 0 for v in exits.values()) else 1


def cmd_report(a):
    text, all_ok = reportlib.report(Path(a.run_dir))
    print(text)
    return 0 if all_ok else 1


def cmd_quality(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    qdir, complete = qualitylib.quality(Path(a.run_dir), lpips_python=a.lpips, syncnet_command=a.syncnet)
    print((qdir / "quality.md").read_text())
    print(f"written to {qdir}")
    return 0 if complete in (None, True) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench/replay.sh", description="Stage-2 replay bench (docs/PERFORMANCE.md)")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add", help="copy a studio clip's draft into the draft store")
    p.add_argument("name")
    p.add_argument("case_dir", help="<runtime root>/outputs/webui/<session>/<case>")
    p.add_argument("--dialogue", action="store_true", help="score it with SyncNet in the quality step")
    p.add_argument("--turns", help="speaking turns, e.g. 'R:0-1.2,L:2.4-4.7' (implies --dialogue)")
    p.add_argument("--mouth", help="mouth crop X:Y:W:H in output pixels, for the mouth strip")
    p.add_argument("--note")
    p.add_argument("--force", action="store_true", help="replace a draft of that name")
    p.set_defaults(fn=cmd_add)
    sub.add_parser("list", help="the drafts in the store").set_defaults(fn=cmd_list)
    sub.add_parser("arms", help="the arms and what each changes").set_defaults(fn=cmd_arms)
    p = sub.add_parser("run", help="replay Stage 2 (GPU); --dry prints the plan only")
    p.add_argument("--dry", action="store_true", help="print processes, shapes, arms and runs; start nothing")
    p.add_argument("--drafts", help="comma-separated draft names (default: all in the store)")
    p.add_argument("--arms", help="comma-separated ARM or ARM:REPEATS (default: baseline)")
    p.add_argument("--repeats", type=int, default=2, help="warm runs per draft and arm (default 2)")
    p.add_argument("--label", help="appended to the run directory's name")
    p.add_argument("--stop-at", type=float, help="epoch seconds: start no run that would end after it")
    p.add_argument("--keep-cache", action="store_true", help="keep the run's compile-cache copy")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report", help="hashes, comparisons and summary.md for a run (again)")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("quality", help="CPU: SSIM/PSNR, sheets, side-by-sides; LPIPS and SyncNet if given")
    p.add_argument("run_dir")
    p.add_argument("--lpips", metavar="PYTHON", help="a Python with torch and lpips (CPU)")
    p.add_argument("--syncnet", metavar="CMD", help="a SyncNet scorer command (docs/PERFORMANCE.md)")
    p.set_defaults(fn=cmd_quality)
    a = ap.parse_args(argv)
    if getattr(a, "label", None) and not draftlib.NAME.fullmatch(a.label):
        ap.error("--label is letters, digits, _ and -")
    try:
        return a.fn(a)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"replay: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
