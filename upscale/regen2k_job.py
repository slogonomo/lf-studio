#!/usr/bin/env python3
"""Experimental Upscale: one finished LF Studio clip to 1080p, 2K (2560x1440) or 4K with NVIDIA's
Open-H3-Regenerate-2K. One upscale per process, run by the studio's server in the Regen-2K environment:

    $LFSTUDIO_UPSCALE_PYTHON -I -B upscale/regen2k_job.py --job <dir>/job.json [--check]

NVIDIA's, not edited and imported from the package dir named in job.json: everything that computes the picture.
That is `open_h3` from NVlabs/Sana, branch sol-engine, models/Open-H3-Regenerate-2K at 6e4a440 (Apache-2.0), with
the weights of Efficient-Large-Model/Open-H3-Regenerate-2K at 5809c33: the latent upscaler, the merged LoRA and the
Sharp curve branches, the sigma schedules (SCHEDULES, shifted_grid), the refine loop (Pipeline.refine), the released
H3 VAE (encode_video, Pipeline.decode) and the MP4 writer (encode_mp4). The job runs NVIDIA's standard forward (BF16
weights, dense cuDNN attention) and the released VAE; never sol_engine, MXFP8, HyperFlow or the fast VAE.

What this wrapper changes, and why:
- The LoRA merge, for memory on GB10. NVIDIA's adapters.StageMerger keeps a pinned host copy of every merged weight
  and the LoRA in fp32 on the GPU, so it can switch between stages. On GB10 host and GPU share one 121 GiB memory,
  next to the 62 GiB DiT, and the load runs out of it. OneShotMerger merges once, in place, with NVIDIA's arithmetic
  (bf16(W.float() + B @ A), 512 rows at a time) and keeps nothing; on the GPU its weights were bit-identical to
  StageMerger's. Only the refine stage of the standard forward is used, so there is nothing to switch back to.
- The load, for time and memory on GB10 (job.json "load"). NVIDIA's code loads through diffusers' from_pretrained
  and safetensors' load_file, which read the files through mmap: on GB10 the page faults come in at ~0.2 GB/s (455 s
  for the job's load, below), every page read stays in the page cache (the GPU's memory too), and diffusers builds
  each model in host memory before Pipeline moves it to the GPU. "fast" (the default) installs the bench wrapper's
  measured setup before anything loads, the encodes too: safetensors.torch.load_file and open_h3.adapters.load_file
  become fastload.load_file(filename, device="cuda") (upscale/fastload.py, a copy of LF Studio's runtime/fastload.py:
  O_DIRECT reads into a few reused windows, each tensor sliced out of one and copied straight to the GPU, never a host
  copy of a file), and from_pretrained of diffusers' MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3 and
  AutoencoderKLMiniMaxH3Audio gets device_map={"": "cuda"} unless the call names one, so each tensor goes to the GPU
  as it is read. Where O_DIRECT can't be used, fastload falls back to safetensors' own load_file, to the GPU as well
  (the receipt counts the files and the fallbacks). Measured with the bench wrapper (a 768p draft, 3 steps): loaded in
  21 s with the 312 LoRA weights merged in 2.8 s, the native and refined latents bit-identical to a standard load's,
  ~1.9 GB of anonymous memory in the process, 40.0 GiB MemAvailable after the load. "standard" is NVIDIA's own
  loading, the fallback (~7.5 min). The latent upscaler's checkpoint (0.7 GB) loads through NVIDIA's node either way.
- The text: the clip's own prompt conditioning (qwen/conditioning.pt; t2va: prompt_embeds [1, L, 5120] bf16, every
  text_token_tag 1) instead of loading the 32B text encoder (66.7 GB). It is the same layer-50 hidden state, from
  LF Studio's NVFP4 encoder: close to NVIDIA's BF16 one, not bit-identical.
- The source. "final": the clip's MP4 (stage2/refined_*.mp4) through NVIDIA's encode_video at 768p. "draft": the
  Stage-1 latent (stage1/h3_normalized_native.pt, 384p), first upscaled x2 to 768p by the same latent upscaler.
- The audio the refine attends to. "silence" is NVIDIA's. "clip" is the clip's own Stage-1 audio
  (stage1/stage1_direct_tensors.pt["audio"]) through the audio VAE, encoded as encode_silence encodes its zeros.
- The strength: start_sigma re-noises from there on NVIDIA's shifted_grid; null keeps NVIDIA's schedule for the
  steps (3 steps: 0.909375, which regenerates faces: a sharper, different-looking person).
- Memory, for GB10 again. The DiT is freed before the decode (a job is the only work in its process; at 4K the
  decode's pixels don't fit next to it). At 4K every transformer block runs lean_forward: diffusers' block forward
  with its row-wise parts over the sequence in 16 chunks and q/k/v written straight into SDPA's layout (on GB10
  bit-identical to the stock forward, with 2.5x less activation memory). The stock forward with only its FFN in 8
  chunks ran out of memory at 4K. Once the weights are in GPU memory their page cache is dropped (on GB10 it takes
  the same memory; with the fast load only the upscaler's file and fallbacks are in it). With "preread" (the standard
  load only, off unless asked for), 8 threads read the weight files into the page cache before the load; it did not
  shorten the one load measured with it. job.json refuses it with the fast load, which reads past the page cache.
- The output has no audio (video_noaudio.mp4, NVIDIA's encode_mp4, CRF 10): the server muxes the clip's audio in.

Scope: t2va clips, 16:9, 5 s (1344x768, 121 frames), the shapes NVIDIA's pipeline is fixed at. fl2va and ref2va need
a text-only encode of the prompt (their conditioning interleaves the pictures), portrait and 10/15 s need changes
to NVIDIA's fixed shapes: follow-ups.

Measured on a DGX Spark (GB10), one 5 s clip, standard forward and VAE, cuDNN attention. This job, run by a staging
studio (2K, balanced: 2 steps from 0.7, source final, the clip's audio, the standard load): encode_video 185 s, load
455 s, steps of 379 and 380 s, decode 181 s, MP4 17 s, 1620.7 s from the request to the MP4; peak allocated 91.08
GiB, lowest MemAvailable 18.32 GiB (34.13 GiB during the load), 4.91 GiB allocated before the decode. The bench
wrapper it is derived from, standard load: load 445-539 s; per refine step 144 s at 1080p and 378 s at 2K; decode
104 / 180 s; MP4 8 / 16 s; peak allocated 81 / 91 GiB; lowest MemAvailable 28 / 15 GiB. 4K, with the lean blocks
(and a load through device_map): steps of 1687 and 1701 s, decode 399 s, peak allocated 94.5 GiB, lowest
MemAvailable 13.8 GiB. The bench with the fast load: above (768p). Not run on the GPU yet: this job with the fast
load (its default), and its 4K path.

Output, in paths.output (default: the directory of job.json): video_noaudio.mp4 and receipt.json (state done, failed or
cancelled; times per stage, steps, sigmas, peak GPU memory, lowest MemAvailable, the inputs' sha256, versions, the load:
its mode, and for fast the files and how many fell back to mmap). Progress on stdout: "[lf-studio] phase <name>" for
encoding, loading, refine k/n, decoding and saving, in that order (encoding only for the final source or the clip's
audio), and "[lf-studio] error <text>" last on failure.
Exit status: 0 done, 1 failed, 2 the job or its inputs were refused (nothing loaded), 143 stopped by SIGTERM.
--check validates job.json and the inputs, prints the plan and exits (0 or 2) without torch and without writing.
--parent PID (the server passes its own): the job is killed when that process ends (PR_SET_PDEATHSIG), so it never
outlives the studio that guards its memory; if that process is gone already, it exits at once (143).

job.json: {"version": 1, "case": DIR, "source": "final"|"draft", "resolution": "1080p"|"2k"|"4k", "steps": 1-4,
"start_sigma": 0<x<1 | null (steps 2 or 3), "seed": INT, "audio": "clip"|"silence", "load": "fast"|"standard" ("fast"),
"preread": BOOL (false; true only with load "standard"), "vae_batch": INT (26), "paths": {"package": DIR,
"weights": DIR, "base_model": DIR, "output": DIR (optional)}, "meta": {...} (optional, copied into the receipt)}.
Paths are absolute.

Everything outside run() and the torch helpers is standard library only, so the server (its settings, the command
line, the progress lines, the mux), bin/lf-studio --check (studio_settings, check_weights) and tests/test_upscale.py
import this file without torch.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TAG = "[lf-studio]"
JOB_VERSION = 1
SOURCES = ("final", "draft")
AUDIO = ("clip", "silence")
# output sizes (width, height); NVIDIA's pipeline.RESOLUTIONS computes 1080p and 4K at whole latent rows and keeps
# the centre rows. Its 768p (no upscale) is not offered.
SIZES = {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}
NVIDIA_STEPS = (2, 3)            # steps with a schedule of their own (sigmas.SCHEDULES); others need start_sigma
MAX_STEPS = 4
# sequence chunks of lean_forward, which every transformer block runs at these sizes (the stock forward elsewhere).
# 4K: the setup of the one 4K run that finished on GB10; the stock forward with its FFN in 8 chunks ran out of memory.
LEAN_CHUNKS = {"4k": 16}
VAE_BATCH = 26                   # NVIDIA's default
VAE_SEED = 42                    # NVIDIA's encode_video default (the posterior sample of the final MP4)
# how the weights load (job.json "load", LFSTUDIO_UPSCALE_LOAD; the docstring): "fast" with upscale/fastload.py
# straight to the GPU, "standard" NVIDIA's own loading through mmap (the fallback)
LOADS = ("fast", "standard")
DEFAULT_LOAD = "fast"
FASTLOAD = Path(__file__).resolve().parent / "fastload.py"
DIRECT_LOAD = ("MiniMaxH3Transformer3DModel", "AutoencoderKLMiniMaxH3", "AutoencoderKLMiniMaxH3Audio")   # diffusers'
# Provisional, until the quality measurements settle them: strength name -> (start_sigma, steps); None = NVIDIA's
# schedule for the steps. "full" is NVIDIA's release setting (3 steps from 0.909375: SCHEDULES[3], not
# shifted_grid(0.909375, 3)), much sharper but it regenerates faces. The studio's request field and
# LFSTUDIO_UPSCALE_STRENGTH take these names.
STRENGTHS = {"faithful": (0.5, 2), "balanced": (0.7, 2), "full": (None, 3)}
DEFAULT_STRENGTH = "balanced"
DEFAULT_AUDIO = "clip"
# The studio's memory floor before a job starts, and its guard while one runs (GiB of MemAvailable). Measured at 2K:
# MemAvailable 117.7 GiB on the idle box, lowest 15 GiB during the job, so the job takes ~103 GiB; 110 (the launcher's
# own floor for the studio) keeps the guard's 5 GiB clear. At 4K (the bench run with the lean blocks) the lowest was
# 13.8 GiB of 117.3, so ~104 GiB.
MIN_FREE_GIB = 110
GUARD_GIB = 5

# what this wrapper was written against (checked against the package's own names at run time)
NVIDIA_COMMIT = "6e4a44002149418a547434b91d7280a8bbdcace3"
WEIGHTS_REVISION = "5809c33730c18b6b1c827b40974855a4288641b3"
LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
CURVE_FILE = "lora/sharp_curve_branches.safetensors"
CURVE_TABLE = "lora/adaln_t_table.bin"
UPSCALER_FILE = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
BASE_PARTS = ("transformer", "vae", "audio_vae")
NATIVE_SHAPE = (24, 37, 48, 84)                  # 1344x768, 121 frames, normalized H3 latent
DRAFT_SHAPE = (1, 24, 37, 24, 42)                # h3_normalized_native.pt at 5 s 16:9 (672x384)
AUDIO_SAMPLES = round(124 / 24 * 32000)          # encode_silence's length: 124 frames at 32 kHz
TEXT_WIDTH = 5120
FINAL = {"width": 1344, "height": 768, "frames": 121, "fps": "24/1"}

OUTPUT_VIDEO = "video_noaudio.mp4"
RECEIPT = "receipt.json"
JOB_KEYS = {"version", "case", "source", "resolution", "steps", "start_sigma", "seed", "audio", "load", "preread",
            "vae_batch", "paths", "meta"}
REQUIRED = ("case", "source", "resolution", "steps", "start_sigma", "seed", "audio", "paths")
PATH_KEYS = ("package", "weights", "base_model", "output")
EXIT_FAILED, EXIT_REFUSED, EXIT_CANCELLED = 1, 2, 128 + signal.SIGTERM


class JobError(ValueError):
    """The job or its inputs are refused (exit status 2)."""


class Cancelled(BaseException):
    """SIGTERM (the studio's Cancel): a BaseException, so no `except Exception` on the way swallows it."""


# ---------------------------------------------------------------- job.json (standard library only)

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _path(value, name, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or "\0" in value or not os.path.isabs(value):
        raise JobError(f"{name}: an absolute path is needed, got {value!r}")
    return Path(value)


def _choice(data, key, choices):
    value = data[key]
    if not isinstance(value, str) or value not in choices:
        raise JobError(f"{key}: one of {', '.join(choices)}, got {value!r}")
    return value


def parse_job(data, job_dir):
    """job.json's object -> the job: a dict with Path values for the directories. Raises JobError naming the field.
    Checks values only; case_inputs and check_weights look at the files."""
    if not isinstance(data, dict):
        raise JobError("job.json must hold a JSON object")
    unknown = sorted(set(data) - JOB_KEYS)
    if unknown:
        raise JobError(f"unknown field(s): {', '.join(unknown)}")
    version = data.get("version", JOB_VERSION)
    if not _is_int(version) or version != JOB_VERSION:
        raise JobError(f"version: {JOB_VERSION} is the only one, got {version!r}")
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise JobError(f"missing field(s): {', '.join(missing)}")
    job = {"version": JOB_VERSION, "case": _path(data["case"], "case")}
    for key, choices in (("source", SOURCES), ("resolution", tuple(SIZES)), ("audio", AUDIO)):
        job[key] = _choice(data, key, choices)
    steps = data["steps"]
    if not _is_int(steps) or not 1 <= steps <= MAX_STEPS:
        raise JobError(f"steps: a whole number from 1 to {MAX_STEPS}, got {steps!r}")
    sigma = data["start_sigma"]
    if sigma is None:
        if steps not in NVIDIA_STEPS:
            raise JobError(f"start_sigma: NVIDIA's schedules have {' or '.join(map(str, NVIDIA_STEPS))} steps; "
                           f"{steps} steps need a start_sigma")
    elif isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) \
            or not 0 < sigma < 1:
        raise JobError(f"start_sigma: a number between 0 and 1 (exclusive) or null, got {sigma!r}")
    else:
        sigma = float(sigma)
    seed = data["seed"]
    if not _is_int(seed) or not 0 <= seed < 2 ** 63:
        raise JobError(f"seed: a whole number from 0 to 2**63-1, got {seed!r}")
    load = data.get("load", DEFAULT_LOAD)
    if not isinstance(load, str) or load not in LOADS:
        raise JobError(f"load: one of {', '.join(LOADS)}, got {load!r}")
    preread = data.get("preread", False)
    if not isinstance(preread, bool):
        raise JobError(f"preread: true or false, got {preread!r}")
    if preread and load == "fast":
        raise JobError('preread: only with load "standard" (the fast load reads the weight files past the page cache)')
    vae_batch = data.get("vae_batch", VAE_BATCH)
    if not _is_int(vae_batch) or not 1 <= vae_batch <= 256:
        raise JobError(f"vae_batch: a whole number from 1 to 256, got {vae_batch!r}")
    paths = data["paths"]
    if not isinstance(paths, dict):
        raise JobError("paths: a JSON object with package, weights, base_model and output")
    unknown = sorted(set(paths) - set(PATH_KEYS))
    if unknown:
        raise JobError(f"paths: unknown field(s): {', '.join(unknown)}")
    for key in PATH_KEYS:
        job[key] = _path(paths.get(key), f"paths.{key}", required=key != "output")
    job["output"] = job["output"] or Path(job_dir)
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise JobError("meta: a JSON object")
    job.update(steps=steps, start_sigma=sigma, seed=seed, load=load, preread=preread, vae_batch=vae_batch, meta=meta)
    return job


def load_job(path):
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            raw = handle.read((1 << 20) + 1)
    except OSError as error:
        raise JobError(f"cannot read {path}: {error.strerror or error}") from None
    if len(raw) > 1 << 20:
        raise JobError(f"{path} is larger than 1 MiB")
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise JobError(f"{path} is not JSON: {error}") from None
    return parse_job(data, path.parent)


def job_record(job):
    """The job as JSON for the receipt."""
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in job.items()}


# ---------------------------------------------------------------- the studio's settings (standard library only)

def studio_settings(environ):
    """The studio's LFSTUDIO_UPSCALE_* settings (bin/env.example) from an environment, with their defaults:
    (settings, problems). webui/server.py stops with the problems and bin/lf-studio --check refuses with them, so
    both say the same. LFSTUDIO_UPSCALE itself (0 or 1) is checked by each of them."""
    home = Path(environ.get("LFSTUDIO_HOME") or Path.home() / "sol-h3").expanduser()
    checkpoints = Path(environ.get("LFSTUDIO_CHECKPOINTS") or home / "checkpoints").expanduser()
    settings, problems = {}, []
    for key, name, default in (("python", "LFSTUDIO_UPSCALE_PYTHON", home / "envs" / "regen2k" / "bin" / "python"),
                               ("package", "LFSTUDIO_UPSCALE_PKG", home / "Open-H3-Regenerate-2K"),
                               ("weights", "LFSTUDIO_UPSCALE_WEIGHTS", checkpoints / "Open-H3-Regenerate-2K"),
                               ("base_model", "LFSTUDIO_UPSCALE_BASE", checkpoints / "MiniMax-H3")):
        settings[key] = Path(environ.get(name) or default).expanduser()
        if not settings[key].is_absolute():
            problems.append(f"{name}={settings[key]}: must be an absolute path")
    for key, name, default, choices in (("strength", "LFSTUDIO_UPSCALE_STRENGTH", DEFAULT_STRENGTH, tuple(STRENGTHS)),
                                        ("audio", "LFSTUDIO_UPSCALE_AUDIO", DEFAULT_AUDIO, AUDIO),
                                        ("load", "LFSTUDIO_UPSCALE_LOAD", DEFAULT_LOAD, LOADS),
                                        ("preread", "LFSTUDIO_UPSCALE_PREREAD", "0", ("0", "1"))):
        settings[key] = environ.get(name) or default
        if settings[key] not in choices:
            problems.append(f"{name}={settings[key]}: must be {', '.join(choices[:-1])} or {choices[-1]}")
    settings["preread"] = settings["preread"] == "1"
    if settings["preread"] and settings["load"] == "fast":         # parse_job's refusal, said at start
        problems.append("LFSTUDIO_UPSCALE_PREREAD=1: only with LFSTUDIO_UPSCALE_LOAD=standard (the fast load reads "
                        "the weight files past the page cache)")
    for key, name, default in (("min_free_gib", "LFSTUDIO_UPSCALE_MIN_FREE_GIB", MIN_FREE_GIB),
                               ("guard_gib", "LFSTUDIO_UPSCALE_GUARD_GIB", GUARD_GIB)):
        raw = environ.get(name) or str(default)
        try:
            settings[key] = float(raw)
            if not math.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError
        except ValueError:
            problems.append(f"{name}={raw}: must be a number of GiB, 0 or more")
            settings[key] = float(default)
    if problems:
        problems[-1] += " (see bin/env.example)"
    return settings, problems


# ---------------------------------------------------------------- inputs (standard library only)

def case_inputs(job):
    """The parent clip's files this job reads: {name: Path}. The final MP4 is always one (the shape check)."""
    case = job["case"]
    if not case.is_dir():
        raise JobError(f"no clip directory {case}")
    finals = sorted((case / "stage2").glob("refined_*.mp4"))
    if len(finals) != 1:
        raise JobError(f"expected one stage2/refined_*.mp4 in {case}, found {len(finals)}")
    files = {"conditioning": case / "qwen" / "conditioning.pt", "final": finals[0]}
    if job["source"] == "draft":
        files["draft"] = case / "stage1" / "h3_normalized_native.pt"
    if job["audio"] == "clip":
        files["pcm"] = case / "stage1" / "stage1_direct_tensors.pt"
    for name, path in files.items():
        if not path.is_file():
            what = {"conditioning": "the prompt conditioning", "draft": "the Stage-1 latent (source draft)",
                    "pcm": "the Stage-1 audio (audio clip)"}.get(name, name)
            raise JobError(f"the clip has no {what}: {path} is missing")
    return files


def probe_command(mp4, ffprobe="ffprobe"):
    return [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames,r_frame_rate",
            "-of", "json", str(mp4)]


def check_final(mp4, ffprobe=None):
    """The clip's MP4 must be 1344x768, 121 frames at 24 fps: a 5 s 16:9 clip, the shape NVIDIA's pipeline takes."""
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        raise JobError("ffprobe is not installed")
    result = subprocess.run(probe_command(mp4, ffprobe), capture_output=True, text=True)
    try:
        streams = json.loads(result.stdout)["streams"] if result.returncode == 0 else None
    except (ValueError, KeyError):
        streams = None
    if streams is None:
        raise JobError(f"ffprobe cannot read {mp4}: {result.stderr.strip()[:300]}")
    video = [s for s in streams if s.get("codec_type") == "video"]
    if len(video) != 1:
        raise JobError(f"{Path(mp4).name} has {len(video)} video streams, not one")
    found = tuple(video[0].get(key) for key in ("width", "height", "nb_frames", "r_frame_rate"))
    if found != (FINAL["width"], FINAL["height"], str(FINAL["frames"]), FINAL["fps"]):
        raise JobError(f"{Path(mp4).name} is {found[0]}x{found[1]}, {found[2]} frames at {found[3]} fps: not a 5 s "
                       "16:9 clip (1344x768, 121 frames at 24 fps)")


def weight_paths(job, upscaler_file=UPSCALER_FILE):
    """The weight files the job needs: {name: Path}."""
    weights, base = job["weights"], job["base_model"]
    files = {"lora": weights / LORA_FILE, "curve": weights / CURVE_FILE, "curve_table": weights / CURVE_TABLE,
             "upscaler": weights / upscaler_file}
    files.update({part: base / part / "config.json" for part in BASE_PARTS})
    return files


def check_weights(job, upscaler_file=UPSCALER_FILE, fastload=FASTLOAD):
    """NVIDIA's package and the weights are where job.json says, and with the fast load (the default when `job` has
    no "load", as bin/lf-studio's check passes it) its loader, upscale/fastload.py, next to this file. The upscaler
    must be a .safetensors file: NVIDIA's node loads anything else with torch.load(weights_only=False), which runs
    code from the file."""
    if not upscaler_file.endswith(".safetensors"):
        raise JobError(f"refusing the latent upscaler {upscaler_file}: only .safetensors files are loaded")
    if job.get("load", DEFAULT_LOAD) == "fast" and not Path(fastload).is_file():
        raise JobError(f"no {fastload}, the fast load's loader (part of LF Studio; the standard load, "
                       "LFSTUDIO_UPSCALE_LOAD=standard, does without it)")
    if not (job["package"] / "open_h3" / "pipeline.py").is_file():
        raise JobError(f"paths.package: no open_h3/pipeline.py in {job['package']}")
    for path in weight_paths(job, upscaler_file).values():
        if not path.is_file():
            raise JobError(f"missing weights: {path}")
    for part in BASE_PARTS:
        if not any((job["base_model"] / part).glob("*.safetensors")):
            raise JobError(f"missing weights: no .safetensors in {job['base_model'] / part}")
    if not job["output"].is_dir():
        raise JobError(f"paths.output: no directory {job['output']}")


def check_text(cond):
    """The conditioning must be t2va's: every row text (tag 1), prompt_embeds [1, rows, 5120]. Works on anything
    with .shape and .flatten().tolist(), so it can be checked without torch. Returns the number of rows."""
    if not isinstance(cond, dict) or "prompt_embeds" not in cond or "text_token_tags" not in cond:
        raise JobError("the conditioning has no prompt_embeds and text_token_tags")
    task = cond.get("task")
    if task not in (None, "t2va"):
        raise JobError(f"the conditioning is {task}'s: only t2va clips can be upscaled for now "
                       "(fl2va/ref2va need a text-only encode of the prompt)")
    tags = [int(tag) for tag in cond["text_token_tags"].flatten().tolist()]
    if not tags:
        raise JobError("the conditioning has no rows")
    other = sum(tag != 1 for tag in tags)
    if other:
        raise JobError(f"the conditioning has {other} non-text rows (fl2va/ref2va): it needs a text-only encode")
    shape = tuple(cond["prompt_embeds"].shape)
    if shape != (1, len(tags), TEXT_WIDTH):
        raise JobError(f"prompt_embeds {shape} do not match {len(tags)} text rows of {TEXT_WIDTH}")
    return len(tags)


def check_draft_shape(shape):
    if tuple(shape) != DRAFT_SHAPE:
        raise JobError(f"the Stage-1 latent is {tuple(shape)}, not a 5 s 16:9 one {DRAFT_SHAPE}")


def check_pcm_shape(shape):
    shape = tuple(shape)
    if len(shape) != 3 or shape[:2] != (1, 2) or not 0 < shape[2] <= AUDIO_SAMPLES:
        raise JobError(f"the Stage-1 audio is {shape}, not stereo PCM of at most {AUDIO_SAMPLES} samples")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan(job, schedules, shifted_grid):
    """What the job will do, with NVIDIA's sigma functions (open_h3.sigmas, standard library only)."""
    if job["start_sigma"] is None:
        if job["steps"] not in schedules:
            raise JobError(f"NVIDIA's package has no {job['steps']}-step schedule")
        sigmas = list(schedules[job["steps"]])
    else:
        sigmas = list(shifted_grid(job["start_sigma"], job["steps"]))
    return {"size": list(SIZES[job["resolution"]]), "frames": FINAL["frames"], "steps": len(sigmas) - 1,
            "sigmas": sigmas, "lean_chunks": LEAN_CHUNKS.get(job["resolution"])}


# ---------------------------------------------------------------- weights in the page cache (standard library only)

def preread_files(job):
    """The big files the load faults in through mmap, LoRA first (the reference wrapper's list)."""
    files = sorted((job["weights"] / "lora").glob("*.safetensors"))
    for part in ("audio_vae", "transformer"):
        files += sorted((job["base_model"] / part).glob("*.safetensors"))
    return files


def loaded_files(job):
    """Every weight file the pipeline reads: their page cache is dropped once the weights are in GPU memory."""
    files = [p for part in BASE_PARTS for p in sorted((job["base_model"] / part).glob("*.safetensors"))]
    for part in ("lora", "latent_upsampler"):
        files += sorted((job["weights"] / part).glob("*.safetensors"))
    return files


def preread(paths, workers=8, chunk=16 << 20):
    """Read the files into the page cache with parallel sequential readers. Returns the bytes read."""
    def read(path):
        total, view = 0, memoryview(bytearray(chunk))
        with open(path, "rb", buffering=0) as handle:
            while True:
                n = handle.readinto(view)
                if not n:
                    return total
                total += n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(read, paths))


def drop_from_page_cache(paths):
    """POSIX_FADV_DONTNEED on each file (its clean, unmapped cached pages). On GB10 the page cache takes the same
    memory as the GPU's. Returns how many files were advised."""
    done = 0
    for path in paths:
        try:
            fd = os.open(os.path.realpath(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            done += 1
        finally:
            os.close(fd)
    return done


# ---------------------------------------------------------------- processes and lines (standard library only)

def worker_command(python, script, job_path, parent=None):
    """How the server runs a job: -I (no user site, no PYTHONPATH, no script dir) and -B (no .pyc files); with
    `parent` (the server's pid) the job ends when the server does (--parent)."""
    return [str(python), "-I", "-B", str(script), "--job", str(job_path)] + \
        (["--parent", str(int(parent))] if parent is not None else [])


# the Hub must never be asked (a missed local path would download with the user's token); NVIDIA's own loaders
# pass local_files_only=True everywhere, this is the second fence
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_TOKEN_PATH": "/nonexistent", "PYTHONNOUSERSITE": "1"}
REMOVED = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "H3_ATTN_BACKEND", "PYTHONPATH")
TUNING = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",     # fragmentation over a ~90 GiB peak
          "MIMALLOC_PURGE_DELAY": "0"}       # torch's aarch64 mimalloc: give freed CPU memory back (it is GPU memory)


def worker_environment(environ):
    """The environment of a job: the offline settings (forced), no token, no H3_ATTN_BACKEND (NVIDIA's standard
    forward picks cuDNN itself), the allocator settings unless set, and the distro's ffmpeg for imageio-ffmpeg when
    there is one."""
    env = {k: v for k, v in environ.items() if k not in REMOVED}
    env.update(OFFLINE)
    for key, value in TUNING.items():
        env.setdefault(key, value)
    if "IMAGEIO_FFMPEG_EXE" not in env and os.path.isfile("/usr/bin/ffmpeg"):
        env["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    return env


def mux_command(silent, parent_mp4, out, ffmpeg="ffmpeg"):
    """The server's step after a job: the job's silent video with the parent clip's audio, both copied. Both are the
    clip's length (121 frames at 24 fps), as in the parent clip. No -shortest: the AAC stream ends a few ms before
    the last frame, and with -shortest that frame was dropped (120 frames)."""
    return [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(silent), "-i", str(parent_mp4), "-map", "0:v:0",
            "-map", "1:a:0", "-c", "copy", "-movflags", "+faststart", str(out)]


def one_line(text, limit=500):
    return " ".join(str(text).split())[:limit]


def phase_line(name):
    return f"{TAG} phase {name}"


def error_line(text):
    return f"{TAG} error {one_line(text)}"


def parse_line(line):
    """A line of the job's output -> ("phase", name), ("error", text) or None."""
    line = line.rstrip("\r\n")
    for kind in ("phase", "error"):
        prefix = f"{TAG} {kind} "
        if line.startswith(prefix) and line[len(prefix):].strip():
            return kind, line[len(prefix):].strip()
    return None


def say(message):
    print(f"{TAG} {time.strftime('%H:%M:%S')} {message}", flush=True)


def write_json(path, data):
    """Atomically: a reader never sees half a receipt."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- memory (standard library only)

def meminfo(path="/proc/meminfo"):
    """MemAvailable, MemFree and Cached in GiB."""
    values = {}
    with open(path) as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "MemFree", "Cached"):
                values[key] = int(rest.split()[0]) / 1048576
    return values


class MemTrace(threading.Thread):
    """The lowest MemAvailable per label, sampled every 0.5 s."""

    def __init__(self, path="/proc/meminfo", interval=0.5):
        super().__init__(daemon=True, name="memtrace")
        self.path, self.interval, self.label, self.low = path, interval, "start", {}
        self.halt = threading.Event()

    def sample(self):
        label = self.label                 # once, before the read: a value read before mark() stays under its label
        available = meminfo(self.path).get("MemAvailable")
        if available is not None:
            self.low[label] = round(min(self.low.get(label, available), available), 2)
        return available

    def mark(self, label):
        self.label = label
        return self.sample()

    def run(self):
        while not self.halt.is_set():
            try:
                self.sample()
            except (OSError, ValueError, IndexError):
                pass
            self.halt.wait(self.interval)

    def stop(self):
        self.halt.set()
        if self.is_alive():
            self.join(timeout=2)


# ---------------------------------------------------------------- with torch (imported inside, on the GPU)

class OneShotMerger:
    """Replaces NVIDIA's adapters.StageMerger (same constructor): its merge arithmetic (StageMerger._merge:
    bf16(W.float() + scale * B @ A), 512 rows at a time, fp32 addmm_), done once in place at construction. No pinned
    pristine copies, no fp32 LoRA kept on the GPU. The refine stage of the standard forward only."""

    def __init__(self, model, stages, device, resident=False):
        import torch
        if set(stages) != {"refine"} or resident:
            raise RuntimeError("OneShotMerger: the refine stage of the standard forward only")
        groups = {}
        for pairs, scale in stages["refine"]:
            if not scale:
                continue
            for name, (a, b) in pairs.items():
                weight = model.get_submodule(name).weight
                groups.setdefault(id(weight), (weight, []))[1].append((a, b, float(scale)))
        start = time.perf_counter()
        with torch.no_grad():
            for weight, items in groups.values():
                on_device = [(a.to(device, torch.float32), b.to(device, torch.float32), scale) for a, b, scale in items]
                for offset in range(0, len(weight), 512):
                    n = min(512, len(weight) - offset)
                    merged = weight[offset:offset + n].float()
                    for a, b, scale in on_device:
                        merged.addmm_(b[offset:offset + n], a, beta=1., alpha=scale)
                    weight[offset:offset + n].copy_(merged.to(weight.dtype))
                del on_device
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.merged, self.seconds = len(groups), time.perf_counter() - start

    def merge(self, stage):
        if stage != "refine":
            raise RuntimeError(f"OneShotMerger has no stage {stage!r}")

    def attach_quantized(self, layers):
        raise RuntimeError("OneShotMerger: no MXFP8 layers (standard forward only)")


def lean_forward(block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None, *, chunks, rotary):
    """diffusers' MiniMaxH3TransformerBlock.forward (diffusers 0.40.0) with less memory, for 4K on GB10. The stock
    forward keeps several full-sequence tensors alive at once (the modulations gathered per row, q/k/v and the
    attention backend's copies of them, the attention output, the FFN's 2 x 28,672-wide hidden); at 4K (~302k rows)
    that does not fit next to the DiT. Here everything but the attention runs over the sequence in `chunks` pieces
    (it is row-wise), q/k/v are written straight into (1, heads, rows, head_dim) buffers that SDPA takes as they are,
    and only q/k/v, the attention output and the block output are full-sized. The same operations in the same order
    per row (only the GEMMs' row counts differ); on GB10 its output was bit-identical to the stock forward's at 4,096
    and 37,910 rows with cuDNN attention, with 2.5x less activation memory. `rotary`: diffusers' _apply_rotary_emb."""
    import torch
    import torch.nn.functional as F
    attn = block.attn
    if attention_mask is not None or hidden_states.shape[0] != 1 or getattr(attn, "fused_projections", False):
        raise RuntimeError("lean_forward: one unmasked sequence and separate q/k/v projections only")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
    rows = hidden_states.shape[1]
    step = -(-rows // chunks)
    heads = attn.heads
    q = torch.empty((1, heads, rows, attn.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    for a in range(0, rows, step):
        b = min(rows, a + step)
        index = adaln_indices[a:b]
        x = block.norm1(hidden_states[:, a:b])
        x = x * (1.0 + scale_msa.index_select(0, index)) + shift_msa.index_select(0, index)
        qa = attn.norm_q(attn.to_q(x).unflatten(-1, (heads, -1)))
        ka = attn.norm_k(attn.to_k(x).unflatten(-1, (heads, -1)))
        va = attn.to_v(x).unflatten(-1, (heads, -1))
        if rotary_emb is not None:
            cos, sin = rotary_emb
            qa = rotary(qa, cos[a:b], sin[a:b])
            ka = rotary(ka, cos[a:b], sin[a:b])
        q[:, :, a:b] = qa.transpose(1, 2)
        k[:, :, a:b] = ka.transpose(1, 2)
        v[:, :, a:b] = va.transpose(1, 2)
        del x, qa, ka, va
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    del q, k, v
    result = torch.empty_like(hidden_states)
    for a in range(0, rows, step):
        b = min(rows, a + step)
        index = adaln_indices[a:b]
        o = out[:, :, a:b].transpose(1, 2).flatten(2, 3).type_as(hidden_states)
        h = hidden_states[:, a:b] + gate_msa.index_select(0, index) * attn.to_out[1](attn.to_out[0](o))
        y = block.norm2(h) * (1.0 + scale_mlp.index_select(0, index)) + shift_mlp.index_select(0, index)
        result[:, a:b] = h + gate_mlp.index_select(0, index) * block.ff(y)
        del o, h, y
    return result


def install_lean_blocks(model, chunks):
    """Every transformer block of the DiT runs lean_forward, with SDPA pinned to cuDNN (the standard forward's own
    attention, _native_cudnn). Returns the number of blocks."""
    import types
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def forward(self, *args, **kwargs):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return lean_forward(self, *args, chunks=chunks, rotary=_apply_rotary_emb, **kwargs)
    for block in model.transformer_blocks:
        block.forward = types.MethodType(forward, block)
    return len(model.transformer_blocks)


def text_input(cond, torch):
    """LF Studio's conditioning.pt -> NVIDIA's text dict (encode_prompt's format)."""
    check_text(cond)
    embeds = cond["prompt_embeds"][0].to(torch.bfloat16)
    if not bool(torch.isfinite(embeds).all()):
        raise JobError("the conditioning has values that are not finite")
    return {"prompt_embeds": embeds, "text_token_tags": cond["text_token_tags"].flatten().long()}


def draft_input(saved, torch):
    """h3_normalized_native.pt -> the normalized (24, 37, 24, 42) Stage-1 latent, fp32."""
    if not isinstance(saved, dict) or "h3_normalized" not in saved:
        raise JobError("the Stage-1 latent file has no h3_normalized")
    check_draft_shape(saved["h3_normalized"].shape)
    latent = saved["h3_normalized"][0].float()
    if not bool(torch.isfinite(latent).all()):
        raise JobError("the Stage-1 latent has values that are not finite")
    return latent


def pcm_input(saved, torch):
    """stage1_direct_tensors.pt -> the clip's Stage-1 audio, (2, samples) fp32 at 32 kHz."""
    if not isinstance(saved, dict) or "audio" not in saved:
        raise JobError("the Stage-1 tensors have no audio")
    check_pcm_shape(saved["audio"].shape)
    pcm = saved["audio"][0].float()
    if not bool(torch.isfinite(pcm).all()):
        raise JobError("the Stage-1 audio has values that are not finite")
    return pcm


def encode_audio(base_model, pcm, torch):
    """NVIDIA's pipeline.encode_silence with the clip's audio in place of the zeros: (2, samples) PCM padded with
    silence to its length, each channel through the audio VAE, normalized, bf16 -> (2, 32, 207)."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        base_model, subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    with torch.inference_mode():
        x = torch.zeros(2, 1, AUDIO_SAMPLES, device="cuda")
        x[:, 0, :pcm.shape[1]] = pcm.cuda()
        audio = audio_vae.encode(x, return_dict=False)[0].mode().float().cpu()
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        audio = ((audio - mean) / std).to(torch.bfloat16)
    del audio_vae
    torch.cuda.empty_cache()
    return audio


def check_package(adapters, pipeline, resolution):
    """NVIDIA's package is the one this wrapper was written against: its file names and shapes."""
    found = {"LORA_FILE": adapters.LORA_FILE, "CURVE_FILE": adapters.CURVE_FILE, "CURVE_TABLE": adapters.CURVE_TABLE,
             "UPSCALER_FILE": pipeline.UPSCALER_FILE, "NATIVE_SHAPE": tuple(pipeline.NATIVE_SHAPE)}
    wanted = {"LORA_FILE": LORA_FILE, "CURVE_FILE": CURVE_FILE, "CURVE_TABLE": CURVE_TABLE,
              "UPSCALER_FILE": UPSCALER_FILE, "NATIVE_SHAPE": NATIVE_SHAPE}
    height, width, keep = pipeline.RESOLUTIONS[resolution]
    found["size"], wanted["size"] = (width, keep), SIZES[resolution]
    differ = {k: found[k] for k in wanted if found[k] != wanted[k]}
    if differ or not hasattr(adapters, "StageMerger"):
        raise JobError(f"NVIDIA's package is not the one this wrapper was written for ({NVIDIA_COMMIT[:7]}): {differ}")


def versions(torch):
    from importlib import metadata
    found = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
             "cudnn": torch.backends.cudnn.version(), "nvidia_package": NVIDIA_COMMIT,
             "weights_revision": WEIGHTS_REVISION}
    for name in ("diffusers", "safetensors", "transformers"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    if torch.cuda.is_available():
        found["gpu"] = torch.cuda.get_device_name()
        found["capability"] = list(torch.cuda.get_device_capability())
    return found


def load_fastload(path=FASTLOAD):
    """upscale/fastload.py as a module of its own, from this file's directory: the job runs with -I, so that directory
    is not on sys.path, and nothing on it (NVIDIA's package is) can stand in for it. Standard library on import."""
    spec = importlib.util.spec_from_file_location("lf_studio_fastload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_fast_load(adapters, fastload, safetensors_torch, diffusers):
    """The fast load (the docstring), the bench wrapper's: safetensors.torch.load_file (diffusers calls it by
    attribute) and NVIDIA's adapters.load_file (imported by name) load every tensor straight to the GPU through
    fastload.load_file, whatever device the caller asks for, so no host copy of a file is made; from_pretrained of
    the DIRECT_LOAD classes gets device_map={"": "cuda"} unless the call names one, so diffusers keeps the tensors
    where they are. Call it before anything loads. Returns the receipt's "load", whose counts grow as files load."""
    stock = safetensors_torch.load_file
    record = {"mode": "fast", "o_direct": bool(fastload.ENABLED and fastload.DIRECT), "files": 0, "fallbacks": 0}
    # diffusers can load from threads (HF_ENABLE_PARALLEL_LOADING, off by default): the counts under a lock
    inside, counting = threading.local(), threading.Lock()

    def count(key):
        with counting:
            record[key] += 1

    def load_file(filename, device="cpu"):
        if getattr(inside, "active", False):
            # fastload.load_file's own fallback (no O_DIRECT for this file, or SPARK_FAST_SAFETENSORS=0) calls
            # safetensors.torch.load_file as it is now, this function: the stock loader, to the GPU too
            count("fallbacks")
            return stock(filename, device="cuda")
        inside.active = True
        try:
            count("files")
            return fastload.load_file(filename, device="cuda")
        finally:
            inside.active = False
    safetensors_torch.load_file = adapters.load_file = load_file
    for name in DIRECT_LOAD:
        cls = getattr(diffusers, name)
        original = cls.from_pretrained.__func__

        def direct(klass, *args, _original=original, **kwargs):
            kwargs.setdefault("device_map", {"": "cuda"})
            return _original(klass, *args, **kwargs)
        cls.from_pretrained = classmethod(direct)
    return record


def run(job, files, receipt, trace):
    """The upscale on the GPU. Fills the receipt as it goes."""
    import torch
    sys.path.insert(0, str(job["package"]))           # -I drops PYTHONPATH and the script dir: name it explicitly
    from open_h3 import adapters, pipeline, upscaler
    check_package(adapters, pipeline, job["resolution"])
    receipt["versions"] = versions(torch)
    torch.cuda.set_device(0)
    times, peaks = receipt["times_s"], receipt["peak_gib"]
    base, weights, out = job["base_model"], job["weights"], job["output"]

    def phase(name):
        receipt["phase"] = name
        print(phase_line(name), flush=True)

    def timed(label, fn):
        trace.mark(label)
        receipt["stage"] = label
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times[label] = round(time.perf_counter() - start, 2)
        peaks[label] = {"allocated": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                        "reserved": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
        say(f"{label} {times[label]:.1f} s, peak allocated {peaks[label]['allocated']:.1f} GiB, MemAvailable "
            f"{meminfo().get('MemAvailable', 0):.1f} GiB")
        return result

    # inputs on the CPU, before any model
    cond = torch.load(files["conditioning"], map_location="cpu", weights_only=True)
    text = text_input(cond, torch)
    receipt["prompt"] = cond.get("prompt") if isinstance(cond.get("prompt"), str) else None
    receipt["text_rows"] = int(text["text_token_tags"].numel())
    del cond
    draft = pcm = native = clip_audio = None
    if job["source"] == "draft":
        draft = draft_input(torch.load(files["draft"], map_location="cpu", weights_only=True), torch)
    if job["audio"] == "clip":
        pcm = pcm_input(torch.load(files["pcm"], map_location="cpu", weights_only=True), torch)

    if job["load"] == "fast":                         # before any model loads, the encodes too
        import diffusers
        import safetensors.torch
        receipt["load"] = install_fast_load(adapters, load_fastload(), safetensors.torch, diffusers)
        say(f"fast load: upscale/fastload.py to the GPU (O_DIRECT {'on' if receipt['load']['o_direct'] else 'off'}), "
            "from_pretrained with device_map cuda")
    else:
        receipt["load"] = {"mode": "standard"}

    # the final MP4 and the clip's audio through the VAEs, loaded, used and freed before the DiT
    if job["source"] == "final" or pcm is not None:
        phase("encoding")
        if job["source"] == "final":
            native = timed("encode_video", lambda: pipeline.encode_video(base, files["final"], "768p", seed=VAE_SEED))
        if pcm is not None:
            clip_audio = timed("encode_audio", lambda: encode_audio(base, pcm, torch))
        gc.collect()
        torch.cuda.empty_cache()
        drop_from_page_cache(p for part in ("vae", "audio_vae") for p in (base / part).glob("*.safetensors"))

    phase("loading")
    if job["preread"]:
        trace.mark("preread")
        start, paths = time.perf_counter(), preread_files(job)
        receipt["preread"] = {"files": len(paths), "gb": round(preread(paths) / 1e9, 2)}
        times["preread"] = round(time.perf_counter() - start, 2)
        say(f"pre-read {receipt['preread']['gb']} GB in {times['preread']:.0f} s")
    adapters.StageMerger = OneShotMerger
    pipe = timed("load", lambda: pipeline.Pipeline(
        weights, base, fwd="standard", steps=job["steps"], start_sigma=job["start_sigma"],
        resolution=job["resolution"], vae="standard", vae_batch=job["vae_batch"]))
    pipe.node.load_model("upscaler", pipe.device, "bf16")    # the node reads its checkpoint on first use: now
    drop_from_page_cache(loaded_files(job))
    # what the load left in the allocator's cache, the fast load's fp32 LoRA (~11.6 GiB, on the GPU from the start)
    # above all: back to the system before the refine (the bench did gc.collect here)
    gc.collect()
    torch.cuda.empty_cache()
    if job["load"] == "fast":
        say(f"fast load: {receipt['load']['files']} files, {receipt['load']['fallbacks']} of them through mmap")
    receipt["merge"] = {"weights": pipe.merger.merged, "seconds": round(pipe.merger.seconds, 2)}
    receipt["lean_chunks"] = LEAN_CHUNKS.get(job["resolution"])
    if receipt["lean_chunks"]:
        install_lean_blocks(pipe.model, receipt["lean_chunks"])
    if clip_audio is not None:
        if tuple(clip_audio.shape) != tuple(pipe.audio.shape):
            raise RuntimeError(f"the clip's audio latent is {tuple(clip_audio.shape)}, NVIDIA's silence "
                               f"{tuple(pipe.audio.shape)}")
        pipe.audio = clip_audio.to(pipe.device)
    if draft is not None:
        native = timed("draft_x2", lambda: upscaler.upscale(pipe.node, draft.to(pipe.device), pipeline.NATIVE_SHAPE))
    if tuple(native.shape) != NATIVE_SHAPE:
        raise RuntimeError(f"the 768p latent is {tuple(native.shape)}, not {NATIVE_SHAPE}")
    low = timed("upscale", lambda: pipe.upscale(native.to(pipe.device)))
    timed("merge_refine", lambda: pipe.set_stage("refine"))  # the merge is done; this turns the Sharp curves on
    receipt["sigmas"] = [float(s) for s in pipe.sigmas]
    receipt["steps"] = steps = len(pipe.sigmas) - 1

    marks, nvidia_on_step = [], pipe.on_step

    def on_step(stage, index):                        # NVIDIA calls it as each step starts
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        trace.mark(f"refine {index + 1}/{steps}")
        phase(f"refine {index + 1}/{steps}")
        nvidia_on_step(stage, index)
    pipe.on_step = on_step

    def refine():
        result = pipe.refine(low, text, job["seed"])
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        return result
    refined = timed("refine", refine)
    receipt["steps_s"] = [round(b - a, 2) for a, b in zip(marks, marks[1:])]

    phase("decoding")
    pipe.model = None                                 # the DiT, ~62 GiB: nothing runs on it any more
    getattr(pipe.node, "MODEL_CACHE", {}).clear()     # the latent upscaler
    del low, native, draft
    gc.collect()
    torch.cuda.empty_cache()
    receipt["allocated_before_decode_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    frames = timed("vae_decode", lambda: pipe.decode(refined))
    pipe.vae = None
    del refined
    gc.collect()
    torch.cuda.empty_cache()

    phase("saving")
    partial = out / f".{OUTPUT_VIDEO}.partial.mp4"
    timed("mp4", lambda: pipeline.encode_mp4(frames, partial))
    os.replace(partial, out / OUTPUT_VIDEO)
    receipt["output"] = {"file": OUTPUT_VIDEO, "size": [int(frames.shape[2]), int(frames.shape[1])],
                         "frames": int(frames.shape[0]), "bytes": (out / OUTPUT_VIDEO).stat().st_size}


# ---------------------------------------------------------------- main

def prepare(job_path, receipt):
    """job.json and the inputs, checked without torch -> (job, files). Fills the receipt's job as soon as it parses."""
    job = load_job(job_path)
    receipt["job"] = job_record(job)
    files = case_inputs(job)
    check_final(files["final"])
    check_weights(job)
    return job, files


def check(job_path):
    """--check: validate, print the plan (sigmas from the package's own open_h3.sigmas), write nothing."""
    try:
        job, files = prepare(job_path, {})
        sys.path.insert(0, str(job["package"]))
        try:
            from open_h3.sigmas import SCHEDULES, shifted_grid
        except ImportError as error:
            raise JobError(f"paths.package: cannot import open_h3.sigmas: {error}") from None
        planned = plan(job, SCHEDULES, shifted_grid)
    except JobError as error:
        print(error_line(error), flush=True)
        return EXIT_REFUSED
    print(json.dumps({"ok": True, "job": job_record(job), "inputs": {k: str(v) for k, v in files.items()},
                      **planned}, indent=1), flush=True)
    return 0


def apply_environment(environ):
    """worker_environment on this process's own environment, before torch or the Hub libraries are imported."""
    wanted = worker_environment(environ)
    for key in [key for key in environ if key not in wanted]:
        del environ[key]
    environ.update(wanted)


def cancel(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)     # a second SIGTERM ends the process at once
    raise Cancelled("stopped by SIGTERM")


PR_SET_PDEATHSIG = 1


def tie_to_parent(parent):
    """SIGKILL this process when `parent` (the studio's server, which started it) ends: then nothing guards the
    job's memory any more, and a studio started again would run clips next to it. Python's SIGTERM handler waits for
    the GPU call under way, which can take minutes, so SIGKILL. Returns False when `parent` is gone already (this
    process was handed to another parent before the prctl), True otherwise, also where prctl is not available."""
    if not sys.platform.startswith("linux"):
        return True
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
            return True
    except (OSError, AttributeError):
        return True
    return os.getppid() == parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", required=True, help="job.json written by the studio's server")
    parser.add_argument("--check", action="store_true",
                        help="validate the job and its inputs, print the plan and exit; no torch, no files written")
    parser.add_argument("--parent", type=int, metavar="PID",
                        help="the studio server's pid: end with it (the server passes its own)")
    args = parser.parse_args(argv)
    job_path = Path(os.path.abspath(args.job))
    if args.check:
        return check(job_path)
    if args.parent is not None and not tie_to_parent(args.parent):
        print(error_line(f"the studio's server (pid {args.parent}) is gone; not starting"), flush=True)
        return EXIT_CANCELLED

    signal.signal(signal.SIGTERM, cancel)
    apply_environment(os.environ)
    started = time.time()
    receipt = {"state": "running", "job_file": str(job_path), "pid": os.getpid(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)), "times_s": {}, "peak_gib": {}}
    out, trace, status = job_path.parent, MemTrace(), 0
    try:
        job, files = prepare(job_path, receipt)
        out = job["output"]
        receipt["inputs"] = {name: str(path) for name, path in files.items()}
        receipt["inputs_sha256"] = {name: sha256(path) for name, path in files.items()}
        write_json(out / RECEIPT, receipt)
        trace.start()
        run(job, files, receipt, trace)
        receipt["state"] = "done"
    except (Cancelled, KeyboardInterrupt) as error:
        receipt.update(state="cancelled", error=str(error) or "interrupted")
        status = EXIT_CANCELLED
    except JobError as error:
        receipt.update(state="failed", error=str(error))
        status = EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - recorded in the receipt; the exit status says it failed
        receipt.update(state="failed", error=one_line(f"{type(error).__name__}: {error}", 2000),
                       traceback=traceback.format_exc().splitlines()[-30:])
        traceback.print_exc()
        status = EXIT_FAILED
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        trace.stop()
        low = dict(trace.low)
        if low:
            low["all"] = min(low.values())
        peaks = [p["allocated"] for p in receipt["peak_gib"].values()]
        receipt.update(lowest_mem_available_gib=low, peak_allocated_gib=max(peaks) if peaks else None,
                       wall_s=round(time.time() - started, 1), exit_status=status)
        if not status:
            receipt.pop("stage", None)
            receipt.pop("phase", None)
        try:
            write_json(out / RECEIPT, receipt)
        except OSError as error:
            print(f"{TAG} cannot write {out / RECEIPT}: {error}", file=sys.stderr, flush=True)
    if status:
        print(error_line(receipt["error"]), flush=True)
    else:
        say(f"done in {receipt['wall_s'] / 60:.1f} min: {out / OUTPUT_VIDEO}")
    return status


if __name__ == "__main__":
    sys.exit(main())
