"""One replay process on the GPU: Stage 2 for one group of the plan (format, length, canvas).

Started by `bench/replay.sh run` under the memory guard, with the Stage-2 python from the path manifest, under
torchrun with one rank, the working directory set to the package and the same worker settings as the studio's
Stage-2 worker (checked against runtime/pipeline.py worker_environment before anything loads). Not for direct use.

Nothing in the installed package is edited. The process builds the runtime's own `runtime.stage2.Session`, as the
studio's worker does, and replays each saved draft through `Session.prepare` and `Session.run`. Arms change the
live session for one run (replay.arms) and are undone after it. Two things apply to every run, the baseline too:
CUDA-event timers around each call of the refiner's transformer (per-step times; no computation changes), and a
0.5 s sampler of MemAvailable. Torch is imported only in main(), so the helpers here can be tested on the CPU.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path

from . import arms as armlib

STUDIO_WORKER_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                      "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
                      "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
                      "PYTORCH_CUDA_ALLOC_CONF", "MIMALLOC_PURGE_DELAY", "TRITON_CACHE_AUTOTUNING")


def env_differences(expected, environ):
    """Worker settings the studio's Stage-2 worker would have but this process lacks (PYTHONPATH aside: the bench
    adds its own directory after the package)."""
    return {k: (v, environ.get(k)) for k, v in expected.items()
            if k != "PYTHONPATH" and k in STUDIO_WORKER_KEYS and environ.get(k) != v}


def attention_contract(steps):
    """The runtime's `_checked_attention_stats` (runtime/stage2.py) for `steps` steps instead of 3: one dense and
    47 Sol calls per step, nothing through the dense guard or the Hunyuan path, the Triton Sol backend."""
    def _checked_attention_stats(model):
        stats = model.sol_attention.stats()
        stats.update(selected_backend=model.sol_backend, architecture="sm121-triton")
        kernel = stats.get("kernel", {})
        if (stats.get("completed_steps") != steps or stats.get("dense_calls") != steps
                or stats.get("sol_calls") != armlib.SOL_LAYERS * steps
                or (steps and kernel.get("kernel_calls") != armlib.SOL_LAYERS * steps)
                or kernel.get("dense_guard_calls", 0) != 0 or kernel.get("hunyuan_calls", 0) != 0
                or model.sol_backend != "triton"):
            raise RuntimeError(f"{steps}-step Triton Sol contract failed: {stats}")
        return stats
    _checked_attention_stats.replay_steps = steps
    return _checked_attention_stats


def skip_denoise(model, video_state, audio_state, video_context, audio_context):
    """denoise_multimodal for a preview: no step runs; the attention counters start from zero and are checked."""
    model.sol_attention.begin_denoise()
    model._checked_attention_stats()
    return video_state, audio_state


class GpuRun:
    """The run object an arm function gets on the GPU (see replay.arms). Every change is recorded and undone by
    undo(), so the next run starts from the runtime's own state, with its own checks."""

    def __init__(self, session, torch, compat, recipe_sigmas):
        self.session, self.models, self.cls = session, session.models, type(session.models)
        self.torch, self.compat = torch, compat
        self.recipe_steps = len(recipe_sigmas) - 1
        self.steps, self.denoise, self.profile = self.recipe_steps, True, None
        self._undo = []

    def _set(self, obj, name, value):
        had = name in vars(obj)
        old = vars(obj).get(name)
        setattr(obj, name, value)
        self._undo.append(lambda: setattr(obj, name, old) if had else delattr(obj, name))

    def sigmas(self, values):
        values = tuple(float(v) for v in values)
        self._set(self.models, "stage2_sigmas", values)
        self._set(self.models, "sigmas", self.torch.tensor(values, device=self.models.device,
                                                           dtype=self.torch.float32))
        self.steps = len(values) - 1

    def no_denoise(self):
        self.sigmas((0.0,))
        self.denoise, self.steps = False, 0
        self._set(self.cls, "denoise_multimodal", skip_denoise)

    def tiles(self, short=None, long=None):
        geo = self.session.geometry
        tiles = armlib.default_tiles(geo)
        for side, value in (("short", short), ("long", long)):
            if value is not None:
                tiles[side] = tuple(int(v) for v in value)
        hw = armlib.tiles_hw(tiles, geo)
        dim = self.compat.DimensionSizeConfig
        config = self.compat.TileSizeConfig(**{k: dim(tile_size=v[0], overlap=v[1]) for k, v in hw.items()})
        config.validate(self.compat.tiling_scale_factors_for_vae(str(self.session.paths["output_video_vae"])),
                        self.compat.VideoPixelShape(batch=1, frames=geo["output_frames"], height=geo["output_height"],
                                                    width=geo["output_width"], fps=24))
        self._set(self.models, "tiling_config", config)

    def profile_step(self, index):
        self.profile = int(index)

    def apply(self, arm_name):
        armlib.ARMS[arm_name].fn(self)
        if self.steps != self.recipe_steps:
            self._set(self.cls, "_checked_attention_stats", attention_contract(self.steps))

    def actual(self):
        """What the refiner holds now (read back before undo)."""
        tiles = self.models.tiling_config
        return {"sigmas": [float(x) for x in self.models.stage2_sigmas],
                "tiles": asdict(tiles) if is_dataclass(tiles) else repr(tiles),
                "checks": "relaxed" if "replay_steps" in vars(getattr(self.cls, "_checked_attention_stats"))
                else "runtime's own"}

    def undo(self):
        while self._undo:
            self._undo.pop()()


def verify(receipt, expected):
    """Problems if the receipt does not show the arm's settings (an arm that silently stopped working)."""
    result = receipt.get("result") or {}
    att, info = result.get("attention") or {}, result.get("output_info") or {}
    problems = []
    if att.get("completed_steps") != expected["steps"]:
        problems.append(f"completed_steps {att.get('completed_steps')}, expected {expected['steps']}")
    if att.get("sol_calls") != armlib.SOL_LAYERS * expected["steps"]:
        problems.append(f"sol_calls {att.get('sol_calls')}, expected {armlib.SOL_LAYERS * expected['steps']}")
    if [info.get("width"), info.get("height")] != list(expected["output"]):
        problems.append(f"output {info.get('width')}x{info.get('height')}, expected "
                        f"{expected['output'][0]}x{expected['output'][1]}")
    if info.get("frames") != expected["frames"]:
        problems.append(f"output frames {info.get('frames')}, expected {expected['frames']}")
    tiling = (result.get("official_vae_tiling") or {}).get("config")
    want = {k: {"tile_size": v[0], "overlap": v[1]} for k, v in expected["tiles"].items()}
    if tiling is not None and tiling != want:
        problems.append(f"decoder tiles {tiling}, expected {want}")
    return problems


class MemSampler:
    """Lowest MemAvailable (GiB) while a run is in flight (unified memory: torch's peaks miss the rest)."""

    def __init__(self, period=0.5):
        self.low, self.period, self._stop = float("inf"), period, threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            self.low = min(self.low, int(line.split()[1]) / 1048576)
            except OSError:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()


class StepTimer:
    """Wraps the refiner's transformer call: a CUDA event pair per step (one call per denoise step), and the
    profiler around one step when an arm asks for it."""

    def __init__(self, inner, torch):
        self.inner, self.torch = inner, torch
        self.events, self.profile_index, self.profile_dir = [], None, None

    def __call__(self, *args, **kwargs):
        torch, index, prof = self.torch, len(self.events), None
        if self.profile_index == index:
            torch.cuda.synchronize()
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.start()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.inner(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        if prof is not None:
            torch.cuda.synchronize()
            prof.stop()
            save_profile(prof, Path(self.profile_dir))
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def seconds(self):
        return [round(s.elapsed_time(e) / 1000, 3) for s, e in self.events]


def save_profile(prof, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    events = prof.key_averages()
    rows = []
    for e in events:
        dev = getattr(e, "self_device_time_total", None)
        rows.append({"name": e.key, "count": e.count,
                     "self_device_us": float(dev if dev is not None else getattr(e, "self_cuda_time_total", 0)),
                     "self_cpu_us": float(e.self_cpu_time_total)})
    rows.sort(key=lambda r: -r["self_device_us"])
    with open(out_dir / "profile_kernels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "count", "self_device_us", "self_cpu_us"])
        w.writeheader()
        w.writerows(rows)
    for key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            (out_dir / "profile_top.txt").write_text(events.table(sort_by=key, row_limit=60,
                                                                  max_name_column_width=110))
            break
        except Exception:
            continue
    try:
        prof.export_chrome_trace(str(out_dir / "profile_trace.json"))
    except Exception as error:
        print(f"[replay] chrome trace not written: {error}", flush=True)


def install_canvas(canvas, s2, geometry_module):
    """Before the Session exists: let the output canvas differ from 2x the draft (the arm's `relaxes` list says
    which runtime checks this bypasses). Returns a dict that receives the draft's own (native) geometry."""
    state = {}
    original_from_recipe = geometry_module.from_recipe

    def from_recipe(config=None):
        native = original_from_recipe(config)            # the runtime's checks on the draft's own 2x route first
        state["native"] = native
        return armlib.canvas_geometry(native, canvas)
    geometry_module.from_recipe = from_recipe

    original_modules = s2.public_refiner_modules

    def public_refiner_modules(ltx_root, geometry=None):
        base, compat = original_modules(ltx_root, state.get("native"))   # its 1344x768 contract, unchanged
        latent = tuple(geometry["refiner_input"])
        base.WIDTH, base.HEIGHT = geometry["output_width"], geometry["output_height"]
        base.TARGET_LATENT_SHAPE = latent
        base.EXPECTED_VIDEO_TOKENS = latent[2] * latent[3] * latent[4]
        return base, compat
    s2.public_refiner_modules = public_refiner_modules

    def upscale_comfy_normalized_h3(model, normalized, target_shape):
        if tuple(target_shape) == tuple(normalized.shape):         # 384p: the draft is already this size
            return normalized.clone()
        # as runtime/stage2_ops/h3_upscale.py, with the upscaler's scale input set to this canvas's ratio
        mean, std = model.comfy_latents_mean, model.comfy_latents_std
        samples = normalized.to(device=mean.device, dtype=mean.dtype, copy=True)
        scale = min(target_shape[3:]) / min(normalized.shape[3:])
        output = model((samples - mean) / std, scale=scale, target_size=tuple(target_shape[2:]), enable_chunking=True)
        return (output * std + mean).to(dtype=normalized.dtype)
    s2.upscale_comfy_normalized_h3 = upscale_comfy_normalized_h3
    return state


def preflight(s2, models_module, geometry_module, recipe_sigmas):
    """Fail before 40 GB load if the package no longer has what the arms change (instead of a silent no-op)."""
    missing = [f"runtime.stage2.{n}" for n in ("Session", "public_refiner_modules", "upscale_comfy_normalized_h3")
               if not hasattr(s2, n)]
    missing += ["runtime.geometry.from_recipe"] if not hasattr(geometry_module, "from_recipe") else []
    if missing:
        raise RuntimeError(f"the installed package lacks {missing}: bench/replay needs updating")
    if tuple(models_module.SIGMAS) != tuple(recipe_sigmas):
        raise RuntimeError(f"runtime.stage2_ops.models.SIGMAS {models_module.SIGMAS} differ from the recipe's "
                           f"{recipe_sigmas}: bench/replay needs updating")


def session_preflight(session):
    models, cls = session.models, type(session.models)
    wanted = [("class", n) for n in ("_checked_attention_stats", "denoise_multimodal", "run_diagnostic")]
    wanted += [("refiner", n) for n in ("stage2_sigmas", "sigmas", "tiling_config", "wrapped_transformer",
                                        "sol_attention", "sol_backend", "device")]
    missing = [n for where, n in wanted if not hasattr(cls if where == "class" else models, n)]
    if "_checked_attention_stats" not in vars(cls):
        missing.append("SparkRefiner._checked_attention_stats (expected on the runtime's own refiner class)")
    compat = sys.modules.get("official_compat_h3_refiner_diagnostic")
    for n in ("TileSizeConfig", "DimensionSizeConfig", "tiling_scale_factors_for_vae", "VideoPixelShape"):
        if compat is None or not hasattr(compat, n):
            missing.append(f"official_compat_h3_refiner_diagnostic.{n}")
    if missing:
        raise RuntimeError(f"the Stage-2 session lacks {missing}: bench/replay needs updating")
    return compat


def append_jsonl(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--group", required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    group = next(g for g in plan["groups"] if g["id"] == args.group)
    out = Path(plan["out"])
    package = plan["package"]
    sys.path.insert(0, package)
    os.chdir(package)
    from runtime.pipeline import worker_environment
    from runtime.config import load_paths, load_recipe
    paths = load_paths(group["paths_manifest"], task=group["task"])
    differences = env_differences(worker_environment("stage2", paths), os.environ)
    if differences:
        raise RuntimeError(f"this process's settings differ from the studio's Stage-2 worker: {differences}")
    import torch
    from runtime import geometry as geometry_module
    from runtime import stage2 as s2
    from runtime.stage2_ops import models as models_module
    preflight(s2, models_module, geometry_module, group["recipe_sigmas"])
    if armlib.CANVASES[group["canvas"]] is not None:
        install_canvas(group["canvas"], s2, geometry_module)
    config = load_recipe(group["task"], orientation=group["orientation"],
                         seconds=None if group["seconds"] == 5 else group["seconds"])
    started = time.time()
    session = s2.Session(paths, str(out / "work" / group["id"]), config)
    load_s = time.time() - started
    geometry = {k: (list(v) if isinstance(v, tuple) else v) for k, v in session.geometry.items()}
    if geometry != group["geometry"]:
        raise RuntimeError(f"the session's geometry {geometry} is not the plan's {group['geometry']}")
    compat = session_preflight(session)
    timer = StepTimer(session.models.wrapped_transformer, torch)
    session.models.wrapped_transformer = timer
    append_jsonl(out / "processes.jsonl", {"group": group["id"], "pid": os.getpid(), "load_s": round(load_s, 2),
                                           "stage2_audio": session.stage2_audio, "torch": torch.__version__,
                                           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))})
    print(f"[replay] {group['id']}: Stage 2 ready in {load_s:.1f} s ({session.stage2_audio} audio)", flush=True)
    failures = consecutive = 0
    for run in group["runs"]:
        draft = plan["drafts"][run["draft"]]
        expected = dict(group["arms"][run["arm"]], frames=group["geometry"]["output_frames"])
        row = {"group": group["id"], "draft": run["draft"], "arm": run["arm"], "repeat": run["repeat"],
               "planned_first_in_process": run["first_in_process"], "task": draft["task"],
               "orientation": group["orientation"], "seconds": group["seconds"], "canvas": group["canvas"],
               "settings": {k: expected[k] for k in ("output", "sigmas", "steps", "tiles", "upscale_scale",
                                                      "profile_step", "video_tokens")},
               "relaxes": expected["relaxes"], "stage2_audio": session.stage2_audio,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "dir": run["dir"]}
        if plan.get("stop_at") and time.time() + run["est_s"] > float(plan["stop_at"]):
            row.update(status="SKIPPED", error="would end after --stop-at")
            append_jsonl(out / "runs.jsonl", row)
            print(f"[replay] skip {run['draft']} {run['arm']} r{run['repeat']}: --stop-at", flush=True)
            continue
        run_dir = out / run["dir"]
        gpu_run = GpuRun(session, torch, compat, group["recipe_sigmas"])
        receipt = {}
        try:
            session.release_idle_cache(qwen_resident=False)       # as the studio does before each request
            gpu_run.apply(run["arm"])
            timer.events, timer.profile_index = [], gpu_run.profile
            timer.profile_dir = str(run_dir / "profile")
            torch.cuda.reset_peak_memory_stats()
            wall = time.time()
            with MemSampler() as mem:
                session.prepare(draft["request_id"], str(run_dir))
                receipt = session.run(str(Path(draft["dir"]) / "stage1"), str(run_dir), draft["request_id"])
            wall = time.time() - wall
            torch.cuda.synchronize()
            row["actual"] = gpu_run.actual()
            problems = verify(receipt, expected)
            result = receipt.get("result") or {}
            row.update(status="PASS" if not problems else "FAIL", error="; ".join(problems) or None,
                       request_index=receipt.get("request_index"),
                       first_in_process=receipt.get("request_index") == 0,
                       stage2_s=receipt.get("stage2_request_s"), wall_s=round(wall, 3),
                       phases_s=result.get("phases_s"), step_s=timer.seconds(),
                       peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                       min_memavail_gib=round(mem.low, 1), output=receipt.get("output"),
                       output_sha256=receipt.get("output_sha256"),
                       attention={k: (result.get("attention") or {}).get(k)
                                  for k in ("completed_steps", "dense_calls", "sol_calls", "stage2_taus")},
                       output_info=result.get("output_info"))
            consecutive = 0 if not problems else consecutive + 1
            failures += bool(problems)
        except Exception as error:
            traceback.print_exc()
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}"[:1000],
                       request_index=receipt.get("request_index"))
            failures += 1
            consecutive += 1
            session.failed = False          # Session.run's finally cleared the request; carry on with the next run
            session.current_capture = session.payload = session.prepared = None
        finally:
            gpu_run.undo()
            timer.profile_index = None
        append_jsonl(out / "runs.jsonl", row)
        print(f"[replay] {run['draft']} {run['arm']} r{run['repeat']}: {row['status']} Stage 2 "
              f"{row.get('stage2_s') and round(row['stage2_s'], 2)} s, steps {row.get('step_s')}, "
              f"min MemAvailable {row.get('min_memavail_gib')} GiB {row.get('error') or ''}", flush=True)
        if consecutive >= 2:
            print("[replay] two failures in a row: this process stops", flush=True)
            break
    session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
