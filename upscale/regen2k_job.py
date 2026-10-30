#!/usr/bin/env python3
"""Experimental Upscale: one finished LF Studio clip to 1080p, 2K (2560x1440, or 1440x2560 for a 9:16 clip) or 4K with
NVIDIA's Open-H3-Regenerate-2K. One upscale per process, run by the studio's server in the Regen-2K environment:

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
  (the receipt counts the files and the fallbacks). Measured with the bench wrapper (3 steps at 768p from a clip's
  Stage-1 latent, a route of the bench's that the job does not take): loaded in 21 s with the 312 LoRA weights merged
  in 2.8 s, the native and refined latents bit-identical to a standard load's, ~1.9 GB of anonymous memory in the
  process, 40.0 GiB MemAvailable after the load; in this job, below. "standard" is NVIDIA's own loading, the fallback
  (~7.5 min). The latent upscaler's checkpoint (0.7 GB) loads through NVIDIA's node either way.
- The text: a prompt conditioning of LF Studio's (the case's qwen/conditioning.pt, text alone: t2va, prompt_embeds
  [1, L, 5120] bf16, every text_token_tag 1; the clip's own, or one made without its pictures, below) instead of
  loading the 32B text encoder (66.7 GB). It is the same layer-50 hidden state, from
  LF Studio's NVFP4 encoder: close to NVIDIA's BF16 one, not bit-identical.
- The audio the refine attends to. "silence" is NVIDIA's. "clip" is the clip's own audio (the case's
  stage1/stage1_direct_tensors.pt["audio"]: Sol-H3's Stage-1 audio, or a LightX2V Turbo clip's audio.pt) through the
  audio VAE, encoded as encode_silence encodes its zeros.
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
- Portrait (9:16), which NVIDIA's code is not written for (its sizes are 16:9). A 16:9 clip runs NVIDIA's calls as
  before. For a 9:16 clip, run() adds portrait entries to NVIDIA's module dicts (PORTRAIT_ENTRIES: INPUT_SIZES and
  NATIVE_SHAPES "768p-portrait", RESOLUTIONS "<resolution>-portrait", its landscape entries with width and height
  swapped; NVIDIA reads them at call time, no file is edited), so encode_video takes the clip at 768x1344 (with
  "768p" it would cover-crop it to 1344x768) and Pipeline targets the portrait grid. NVIDIA's Pipeline.upscale is not
  used for it: its test for the step through the 2K grid (pipeline.py:226, the target's latent rows above the
  landscape 2K grid's 90) holds for every portrait target, which it would stretch through the landscape grid and back
  with no error. upscale_route runs NVIDIA's upscaler as Pipeline.upscale does, with that 2K grid turned to portrait
  (160x90 latents): at 4K only. NVIDIA's decode keeps every row of a portrait output; the job keeps the centre
  columns (keep_columns: 1088 -> 1080 at 1080p, 2176 -> 2160 at 4K), as that decode keeps the centre rows of a
  landscape one. check_package pins NVIDIA's pipeline.py, upscaler.py and upscaler_node.py (sha256) for this, and
  refuses a portrait key NVIDIA has itself; bin/lf-studio notes at start when the package's files are other ones.
- The clip's length. NVIDIA's encode_video reads 121 frames and adds 3 copies of the last one to reach the 124 the
  VAE takes at 5 s (37 latent frames), and its decode keeps 121. H3 makes 124 frames at 5 s, and a LightX2V Turbo clip
  with a last keyframe keeps them all: it ends on the frame that keyframe is anchored to. Such a clip goes through
  encode_clip and decode_released instead, NVIDIA's code with the frames as parameters (FRAME_CASES: 124 read, no
  copies; the same 37 latent frames, rows and refine), and comes out with all 124 (not run on the GPU yet). A
  121-frame clip runs NVIDIA's own encode_video and decode, as before. check_package pins NVIDIA's pipeline.py
  (sha256) when the copies run.
- Keep my frames (job.json "pin"; experimental, not run on the GPU through the studio yet). NVIDIA's refine is text to
  video: it redraws the clip's first and last frames like any other, away from the pictures they were made from.
  With a pin, the clip's keyframes are held as H3 holds a keyframe, the bench wrapper's K1 refine for either end or
  both (refine_pinned): each picture fitted to the refine's grid as the clip's model fitted it to its canvas
  (keyframe_image: H3's cover-crop as a box in the picture, one LANCZOS step from it), encoded on encode_clip's VAE
  before it is freed as H3 encodes a keyframe, and put in front of the generated rows as R condition rows where H3
  puts them (diffusers' build_packed_sequence: the first at frame 0's rotary time, the last at the last frame made's,
  205 after the text at 5 s), at t = 0.999 in every step and never stepped. Only where the clip shows the picture
  already: before anything loads, check_pin compares each picture with the clip's own frame (0, or its last) at
  672x384, luma PSNR, and an end below 28 dB, one whose picture is gone or changed (sha256), or a last end the clip
  stops short of (a clip that keeps fewer frames than H3 made) is not held; the receipt says why, and the job never
  refuses for it. With no end held the refine is NVIDIA's.

The input: job.json "case", a folder in the shape of a Sol-H3 clip's own: stage2/refined_*.mp4 (the clip's MP4,
through NVIDIA's encode_video at 768p: 1344x768, or 768x1344 for 9:16), qwen/conditioning.pt (text alone: t2va) and,
for audio clip, stage1/stage1_direct_tensors.pt ({"audio": (1, 2, samples)} PCM at 32 kHz). The studio's server builds
it for every upscale (webui/server.py build_input: the upscale's input/ folder) from any finished 5 s clip: Sol-H3's
Text, Keyframes and Reference clips and LightX2V Turbo's Text and Keyframes clips, 16:9 or 9:16. It links the clip's
MP4 (a LightX2V Turbo clip at 576p stretched to 768p; one with a last keyframe whole, all 124 frames, or at 4K cut
to its first 121), its audio (a LightX2V Turbo clip's turbo/audio.pt) and a conditioning of text alone: the clip's
own for a Text clip, a LightX2V Turbo Keyframes clip's qwen-text/, or an encode of the prompt without its keyframe
line or its references for the others (bench/h3turbo_encode.py --upscale). This job takes that folder as it took a
Sol-H3 Text clip's own: 5 s (121 frames at 24 fps, the length NVIDIA's pipeline is fixed at, or the 124 H3 makes:
FRAME_CASES), 16:9 or 9:16, a t2va conditioning; its output has the clip's frames. The other lengths the studio offers
(6 to 15 s) change NVIDIA's fixed length: a follow-up.

Measured on a DGX Spark (GB10), 2026-10-28: this job run by a staging studio with nothing else loaded, 5 s clips,
balanced (2 steps from 0.7), the clip's audio, standard forward and VAE, cuDNN attention, one run each. A 16:9 clip with
the fast load (the job as it was just before it took 9:16; its calls into NVIDIA's code for 16:9 are unchanged,
tests/test_upscale.py), 1080p / 2K / 4K: encode_video 131 / 129 / 133 s, load 21.5 / 21.6 / 21.7 s (the LoRA merge 2.8 s
of it), steps of 151+149 / 379+380 / 1684+1688 s (at 4K every block ran lean_forward, 16 chunks), decode 108 / 182 /
396 s, MP4 3.5 / 15 / 12.7 s; 574.2 / 1122.6 / 3958.1 s from the job's start to the MP4; peak allocated 81.29 / 91.08
/ 94.46 GiB, lowest MemAvailable 27.4 / 15.7 / 11.1 GiB. At 2K the video is the same, bit for bit, as with the standard
load (the same clip and seed), which took 1620.8 s: encode_video 185 s, load 455 s (the merge 74 s of it), the refine
and decode as above. A 9:16 clip with the fast load at 1080p: steps of 143+144 s, 553.9 s, peak 81.26 GiB, as 16:9 (the
same number of tokens); at 2K, cleanly, on 2026-10-29 (a Keyframes clip, through the studio): steps of 376+376 s,
1117.4 s with a 9.1 s encode of its text, peak 91.09 GiB, lowest MemAvailable 16.8 GiB, as 16:9; at 4K not yet. That
day the studio also took that clip, a Sol-H3 Reference clip, a LightX2V Turbo 576p Text clip (stretched) and a 768p
one with a last keyframe (cut to 121 frames) through this job to 1080p: 553.6-566.8 s each, as a Text clip. The
bench wrapper it is derived from: docs/PERFORMANCE.md.

Output, in paths.output (default: the directory of job.json): video_noaudio.mp4 and receipt.json (state done, failed or
cancelled; the clip's orientation and "frames" (from its MP4: kept, made by H3, pad and T; frame_case), times per
stage, the latent grids the upscaler ran to, steps, sigmas, peak GPU memory, lowest MemAvailable, the inputs' sha256,
versions, the load: its mode, and for fast the files and how many fell back to mmap; with a pin its "pin": "asked"
(the ends job.json names), per end "used", "agreement_db", "frame", "fit", "box", "size" and "why" (check_pin's) and,
when held, "latent_shape" and "latent_sha256"; "grid", "check_grid", "min_db", "noise_aug", "encode_seed",
"noise_seed", "native_sha256" (the clip's latent) and, with an end held, "encode_s" (the pictures' encode), "layout"
(check_pin_layout's facts), "timesteps" per step, "condition_sha256" and "refined_sha256"). Progress on stdout:
"[lf-studio] phase <name>" for encoding, loading, refine k/n, decoding and saving, in that order, and
"[lf-studio] error <text>" last on failure.
Exit status: 0 done, 1 failed, 2 the job or its inputs were refused (nothing loaded), 143 stopped by SIGTERM.
--check validates job.json and the inputs (the pin's check too: it needs Pillow and numpy), prints the plan and exits
(0 or 2) without torch and without writing.
--parent PID (the server passes its own): the job is killed when that process ends (PR_SET_PDEATHSIG), so it never
outlives the studio that guards its memory; if that process is gone already, it exits at once (143).

job.json: {"version": 1, "case": DIR, "resolution": "1080p"|"2k"|"4k", "steps": 1-4,
"start_sigma": 0<x<1 | null (steps 2 or 3), "seed": INT, "audio": "clip"|"silence", "load": "fast"|"standard" ("fast"),
"preread": BOOL (false; true only with load "standard"), "vae_batch": INT (26), "paths": {"package": DIR,
"weights": DIR, "base_model": DIR, "output": DIR (optional)}, "meta": {...} (optional, copied into the receipt),
"pin": {"keyframes": [{"anchor": "first"|"last", "image": FILE, "sha256": HEX (the picture's, as the clip was made
from it), "fit": "as is"|"stretch"|"cover-crop" (how the clip's model put it on its canvas), "canvas": [W, H] (the
clip's)}, ...] (one or both ends, first before last), "check": {"grid": "384p", "min_db": 28}, "noise_aug": 0.999,
"encode_seed": 42, "noise_seed": 42} (optional; the values shown are the defaults)}. Paths are absolute. Any other
field is refused, "source" too: the job has no draft source any more (the clip's Stage-1 latent, which redrew the
video too much), it always reads the clip's MP4. The orientation is not a field either: the clip's MP4 has it (16:9
or 9:16), and the output is that way round (2k: 2560x1440 or 1440x2560).

Everything outside run() and the torch helpers is standard library only, so the server (its settings, the command
line, the progress lines, the mux), bin/lf-studio --check (studio_settings, check_weights, nvidia_files) and
tests/test_upscale.py import this file without torch.
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
AUDIO = ("clip", "silence")
# The clips this job takes, per orientation (the clip's MP4 decides): the MP4's size (width, height), the size name
# NVIDIA's encode_video gets, the normalized latent it gives (24, T, H/16, W/16), the key of NVIDIA's
# pipeline.RESOLUTIONS per resolution, and the output (width, height). NVIDIA's RESOLUTIONS computes 1080p and 4K at
# whole latent patches (1088 / 2176) and keeps the centre rows (landscape) or, here, columns (portrait). Its 768p (no
# upscale) is not offered. Portrait: NVIDIA's landscape shapes turned, with the entries PORTRAIT_ENTRIES adds.
SHAPES = {
    "landscape": {"input": (1344, 768), "size": "768p", "native": (24, 37, 48, 84),
                  "resolutions": {"1080p": "1080p", "2k": "2k", "4k": "4k"},
                  "sizes": {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}},
    "portrait": {"input": (768, 1344), "size": "768p-portrait", "native": (24, 37, 84, 48),
                 "resolutions": {"1080p": "1080p-portrait", "2k": "2k-portrait", "4k": "4k-portrait"},
                 "sizes": {"1080p": (1080, 1920), "2k": (1440, 2560), "4k": (2160, 3840)}}}
SIZES = SHAPES["landscape"]["sizes"]         # the resolutions (job.json, the studio's choices), 16:9
# NVIDIA's pipeline module dicts: its own landscape entries this wrapper was written against (check_package), and
# the portrait ones run() adds for a 9:16 clip, the same with width and height swapped. RESOLUTIONS is (computed
# height, width, kept rows): a portrait output keeps every row, and keep_columns keeps the centre columns.
NVIDIA_ENTRIES = {"INPUT_SIZES": {"768p": (1344, 768)}, "NATIVE_SHAPES": {"768p": (24, 37, 48, 84)},
                  "RESOLUTIONS": {"1080p": (1088, 1920, 1080), "2k": (1440, 2560, 1440), "4k": (2176, 3840, 2160)}}
PORTRAIT_ENTRIES = {"INPUT_SIZES": {"768p-portrait": (768, 1344)}, "NATIVE_SHAPES": {"768p-portrait": (24, 37, 84, 48)},
                    "RESOLUTIONS": {"1080p-portrait": (1920, 1088, 1920), "2k-portrait": (2560, 1440, 2560),
                                    "4k-portrait": (3840, 2176, 3840)}}
UPSCALED_SHAPE = (24, 37, 90, 160)           # NVIDIA's pipeline.UPSCALED_SHAPE: the 2K grid 4K is upscaled through
# NVIDIA's files the portrait route builds on (it adds to pipeline.py's dicts, stands in for its Pipeline.upscale and
# crops after its decode, and calls upscaler.upscale, whose node, upscaler_node.py, sizes the portrait grid and aligns
# it to 32 px): their sha256 at NVIDIA_COMMIT, checked for a 9:16 clip (check_package) and by bin/lf-studio at start
NVIDIA_FILES = {"open_h3/pipeline.py": "361729e81b9d15c8af4b1336e4bcbc36a4f20ddfe0e911e65e1d5c434a0f04ab",
                "open_h3/upscaler.py": "e54959b609c5732711fd8e87b5e6cbe4e80977379d988df175afa1c7745f4521",
                "open_h3/upscaler_node.py": "f5945b500ddb38afee23c377034bf15394ba64748511a7e9608953040c4be437"}
NVIDIA_STEPS = (2, 3)            # steps with a schedule of their own (sigmas.SCHEDULES); others need start_sigma
MAX_STEPS = 4
# sequence chunks of lean_forward, which every transformer block runs at these sizes (the stock forward elsewhere).
# 4K: the setup of the one 4K run that finished on GB10; the stock forward with its FFN in 8 chunks ran out of memory.
LEAN_CHUNKS = {"4k": 16}
VAE_BATCH = 26                   # NVIDIA's default
VAE_SEED = 42                    # NVIDIA's encode_video default (the posterior sample of the final MP4)
PIXEL_MEAN, PIXEL_STD = (.485, .456, .406), (.229, .224, .225)    # the video VAE's pixels (ImageNet's), as NVIDIA's
# how the weights load (job.json "load", LFSTUDIO_UPSCALE_LOAD; the docstring): "fast" with upscale/fastload.py
# straight to the GPU, "standard" NVIDIA's own loading through mmap (the fallback)
LOADS = ("fast", "standard")
DEFAULT_LOAD = "fast"
FASTLOAD = Path(__file__).resolve().parent / "fastload.py"
# the patch's switches for its loader (bin/env.example), which the job inherits from the studio and fastload reads at
# import: either one anything but 1 (unset is 1) and it reads nothing with O_DIRECT, every file goes through mmap
FASTLOAD_SWITCHES = ("SPARK_FAST_SAFETENSORS", "SPARK_DIRECT_IO")
DIRECT_LOAD = ("MiniMaxH3Transformer3DModel", "AutoencoderKLMiniMaxH3", "AutoencoderKLMiniMaxH3Audio")   # diffusers'
# Provisional, until the quality measurements settle them: strength name -> (start_sigma, steps); None = NVIDIA's
# schedule for the steps. "full" is NVIDIA's release setting (3 steps from 0.909375: SCHEDULES[3], not
# shifted_grid(0.909375, 3)), much sharper but it regenerates faces. The studio's request field and
# LFSTUDIO_UPSCALE_STRENGTH take these names.
STRENGTHS = {"faithful": (0.5, 2), "balanced": (0.7, 2), "full": (None, 3)}
DEFAULT_STRENGTH = "balanced"
DEFAULT_AUDIO = "clip"
# The studio's memory floor before a job starts, and its guard while one runs (GiB of MemAvailable). Measured in this
# job's runs (fast load, a 16:9 clip; MemAvailable at its start -> its lowest): 116.7 -> 15.7 GiB at 2K, so ~101 GiB
# used, and 116.4 -> 11.1 GiB at 4K (the lean blocks), so ~105 GiB (the bench's 4K run: 13.8 GiB of 117.3, ~104).
# 110 (the launcher's own floor for the studio; 117.7 GiB on the idle box) keeps the guard's 5 GiB clear at 2K, not
# at 4K: a 4K job that starts with just 110 GiB free would come down to ~4.7 GiB, below the guard, which stops it if
# that lasts 3 s. So 4K needs FLOOR_4K_EXTRA_GIB more than the floor (min_free_for): 115 GiB by default.
MIN_FREE_GIB = 110
FLOOR_4K_EXTRA_GIB = 5
GUARD_GIB = 5

# what this wrapper was written against (checked against the package's own names at run time)
NVIDIA_COMMIT = "6e4a44002149418a547434b91d7280a8bbdcace3"
WEIGHTS_REVISION = "5809c33730c18b6b1c827b40974855a4288641b3"
LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
CURVE_FILE = "lora/sharp_curve_branches.safetensors"
CURVE_TABLE = "lora/adaln_t_table.bin"
UPSCALER_FILE = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
BASE_PARTS = ("transformer", "vae", "audio_vae")
NATIVE_SHAPE = SHAPES["landscape"]["native"]     # 1344x768, 5 s (37 latent frames), normalized (pipeline.NATIVE_SHAPE)
AUDIO_SAMPLES = round(124 / 24 * 32000)          # encode_silence's length: 124 frames at 32 kHz
TEXT_WIDTH = 5120
FPS = "24/1"                                     # the clip's MP4 and the output
NVIDIA_FRAMES = 121              # NVIDIA's 5 s: encode_video reads at most 121 frames, its decode keeps 121
# The clips this job takes, by the frames their MP4 keeps (5 s): how many copies of its last frame the encode adds to
# reach the frames H3 makes (made_frames: 124 at 5 s, 37 latent frames either way). 121: NVIDIA's own encode, 3
# copies of frame 120. 124: every frame made, none added (a LightX2V Turbo clip with a last keyframe, which ends on the
# frame that keyframe is anchored to).
FRAME_CASES = {121: {"pad": 3}, 124: {"pad": 0}}
FINAL = {"frames": NVIDIA_FRAMES, "fps": FPS}    # NVIDIA's 5 s clip, the length the server cuts a longer one to
# Keep my frames (job.json "pin", the docstring): the clip's keyframes held in the refine as H3 holds them, each end
# only where the clip shows its picture already (check_pin): luma PSNR of the clip's frame against the picture, fitted
# as the clip's model fitted it, at the check's grid, at least min_db
ANCHORS = ("first", "last")                       # H3's packed order (diffusers' keyframe_anchors)
FITS = ("as is", "stretch", "cover-crop")         # how the clip's model put the picture on its canvas: H3's rule, in
#                                                   the words of jobs/h3turbo_job.py keyframe_fit and its receipts
PIN_CHECK_GRIDS = {"384p": {"landscape": (672, 384), "portrait": (384, 672)}}
PIN_MIN_DB = 28
PIN_NOISE_AUG = 0.999                             # H3's keyframe_noise_aug: the keyframe rows held at t = 0.999
PIN_SEED = 42                                     # H3's keyframe_encode_seed; the rows' noise (the bench wrapper's)
LUMA = (0.2126, 0.7152, 0.0722)                   # BT.709
PIN_WHY = {"gone": "your picture is no longer on disk",
           "changed": "your picture changed since the clip was made",
           "unreadable": "your picture does not open",
           "far": "the clip's frame {frame} is {db:.1f} dB from your picture ({min:g} needed)",
           "past_end": "the clip stops {n} frames before your last picture"}
PIN_KEYS = {"keyframes", "check", "noise_aug", "encode_seed", "noise_seed"}
KEYFRAME_KEYS = ("anchor", "image", "sha256", "fit", "canvas")

OUTPUT_VIDEO = "video_noaudio.mp4"
RECEIPT = "receipt.json"
JOB_KEYS = {"version", "case", "resolution", "steps", "start_sigma", "seed", "audio", "load", "preread", "vae_batch",
            "paths", "meta", "pin"}
REQUIRED = ("case", "resolution", "steps", "start_sigma", "seed", "audio", "paths")
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
    for key, choices in (("resolution", tuple(SIZES)), ("audio", AUDIO)):
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
    pin = parse_pin(data["pin"]) if "pin" in data else None
    job.update(steps=steps, start_sigma=sigma, seed=seed, load=load, preread=preread, vae_batch=vae_batch, meta=meta,
               pin=pin)
    return job


def _seed(value, name):
    if not _is_int(value) or not 0 <= value < 2 ** 63:
        raise JobError(f"{name}: a whole number from 0 to 2**63-1, got {value!r}")
    return value


def parse_pin(value):
    """job.json's "pin" -> {"keyframes": [{"anchor", "image" (an absolute path), "sha256", "fit", "canvas" (w, h)}]
    in H3's packed order (first before last, one each), "check": {"grid", "min_db"}, "noise_aug", "encode_seed",
    "noise_seed"}. Raises JobError naming the field. "canvas" is the clip's canvas, where its model fitted the
    picture; "sha256" the picture's as the clip was made from it."""
    if not isinstance(value, dict):
        raise JobError("pin: a JSON object with keyframes")
    unknown = sorted(set(value) - PIN_KEYS)
    if unknown:
        raise JobError(f"pin: unknown field(s): {', '.join(unknown)}")
    keyframes = value.get("keyframes")
    if not isinstance(keyframes, list) or not 1 <= len(keyframes) <= len(ANCHORS):
        raise JobError(f"pin.keyframes: a list of 1 or {len(ANCHORS)} keyframes, got {one_line(repr(keyframes), 80)}")
    parsed = []
    for index, keyframe in enumerate(keyframes):
        name = f"pin.keyframes[{index}]"
        if not isinstance(keyframe, dict):
            raise JobError(f"{name}: a JSON object with {', '.join(KEYFRAME_KEYS)}")
        unknown = sorted(set(keyframe) - set(KEYFRAME_KEYS))
        missing = [key for key in KEYFRAME_KEYS if key not in keyframe]
        if unknown or missing:
            raise JobError(f"{name}: " + "; ".join(([f"unknown field(s): {', '.join(unknown)}"] if unknown else []) +
                                                   ([f"missing field(s): {', '.join(missing)}"] if missing else [])))
        anchor, digest, fit, canvas = (keyframe[key] for key in ("anchor", "sha256", "fit", "canvas"))
        if not isinstance(anchor, str) or anchor not in ANCHORS:
            raise JobError(f"{name}.anchor: one of {', '.join(ANCHORS)}, got {anchor!r}")
        image = _path(keyframe["image"], f"{name}.image")
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise JobError(f"{name}.sha256: 64 lowercase hex digits, got {one_line(repr(digest), 80)}")
        if not isinstance(fit, str) or fit not in FITS:
            raise JobError(f"{name}.fit: one of {', '.join(FITS)}, got {fit!r}")
        if not isinstance(canvas, list) or len(canvas) != 2 or not all(_is_int(v) and 0 < v <= 16384 for v in canvas):
            raise JobError(f"{name}.canvas: [width, height] in pixels, got {one_line(repr(canvas), 80)}")
        parsed.append({"anchor": anchor, "image": str(image), "sha256": digest, "fit": fit, "canvas": tuple(canvas)})
    anchors = [keyframe["anchor"] for keyframe in parsed]
    if anchors != [anchor for anchor in ANCHORS if anchor in anchors]:
        raise JobError(f"pin.keyframes: one per end, first before last, got {', '.join(anchors)}")
    check = value.get("check", {})
    if not isinstance(check, dict) or set(check) - {"grid", "min_db"}:
        raise JobError(f"pin.check: a JSON object with grid and min_db, got {one_line(repr(check), 80)}")
    grid, min_db = check.get("grid", "384p"), check.get("min_db", PIN_MIN_DB)
    if not isinstance(grid, str) or grid not in PIN_CHECK_GRIDS:
        raise JobError(f"pin.check.grid: one of {', '.join(PIN_CHECK_GRIDS)}, got {grid!r}")
    if isinstance(min_db, bool) or not isinstance(min_db, (int, float)) or not math.isfinite(min_db) \
            or not 0 <= min_db <= 99:
        raise JobError(f"pin.check.min_db: a number of dB from 0 to 99, got {min_db!r}")
    noise_aug = value.get("noise_aug", PIN_NOISE_AUG)
    if isinstance(noise_aug, bool) or not isinstance(noise_aug, (int, float)) or not math.isfinite(noise_aug) \
            or not 0 < noise_aug < 1:
        raise JobError(f"pin.noise_aug: a number between 0 and 1 (exclusive), got {noise_aug!r}")
    return {"keyframes": parsed, "check": {"grid": grid, "min_db": min_db}, "noise_aug": float(noise_aug),
            "encode_seed": _seed(value.get("encode_seed", PIN_SEED), "pin.encode_seed"),
            "noise_seed": _seed(value.get("noise_seed", PIN_SEED), "pin.noise_seed")}


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
    both say the same. LFSTUDIO_UPSCALE itself (0 or 1) is checked by each of them. "o_direct" is whether the fast
    load's reads can use O_DIRECT: the patch's FASTLOAD_SWITCHES, which the job inherits, are both 1 (or unset)."""
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
    settings["o_direct"] = all(environ.get(name, "1") == "1" for name in FASTLOAD_SWITCHES)    # as fastload reads them
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


def min_free_for(resolution, floor):
    """The MemAvailable a job of this resolution needs to start: the floor (LFSTUDIO_UPSCALE_MIN_FREE_GIB), plus
    FLOOR_4K_EXTRA_GIB at 4K (see MIN_FREE_GIB). A floor of 0 (no check) stays 0."""
    return floor + FLOOR_4K_EXTRA_GIB if floor > 0 and resolution == "4k" else floor


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
    if job["audio"] == "clip":
        files["pcm"] = case / "stage1" / "stage1_direct_tensors.pt"
    for name, path in files.items():
        if not path.is_file():
            what = {"conditioning": "the prompt conditioning", "pcm": "the Stage-1 audio (audio clip)"}.get(name, name)
            raise JobError(f"the clip has no {what}: {path} is missing")
    return files


def probe_command(mp4, ffprobe="ffprobe"):
    return [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames,r_frame_rate",
            "-of", "json", str(mp4)]


def clip_read_command(mp4, size, kept, pad, ffmpeg="ffmpeg"):
    """How encode_clip reads the clip's MP4 at `size` (w, h): NVIDIA's encode_video command (pipeline.py:57-61; for
    121 frames and 3 copies the same list) with -frames:v `kept`. With pad 0 (a clip of every frame made) without its
    tpad, which holds the last frame of a source that is too short: so a short MP4 reads short and fails the length
    check."""
    width, height = size
    scale = f"fps=24,scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,crop={width}:{height}"
    return [str(ffmpeg), "-v", "error", "-i", str(mp4), "-vf",
            scale + (",tpad=stop_mode=clone:stop_duration=6" if pad else ""), "-frames:v", str(kept), "-an", "-f",
            "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def made_frames(kept):
    """The frames H3 makes for a clip that keeps `kept`: the VAE's 17n+5 at or above it (121 -> 124)."""
    return kept + (5 - kept) % 17


def latent_frames(frames):
    """The VAE's latent frames for 17n+5 pixel frames: 5n+2 (124 -> 37)."""
    if frames % 17 != 5:
        raise ValueError(f"{frames} frames are not 17n+5")
    return (frames - 5) // 17 * 5 + 2


def frame_case(kept):
    """A clip's frames (FRAME_CASES): kept (its MP4's, and the output's), made (by H3), pad (the copies of its last
    frame the encode adds) and T (the latent frames)."""
    pad = FRAME_CASES[kept]["pad"]
    return {"kept": kept, "made": made_frames(kept), "pad": pad, "T": latent_frames(kept + pad)}


def check_final(mp4, ffprobe=None):
    """The clip's MP4 must be a 5 s clip at 24 fps, 16:9 (1344x768) or 9:16 (768x1344), of 121 frames (the length
    NVIDIA's pipeline is fixed at) or 124 (every frame H3 makes): FRAME_CASES. Returns (its orientation, "landscape"
    or "portrait", and its frames)."""
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
    orientation = next((name for name, shape in SHAPES.items() if shape["input"] == found[:2]), None)
    if orientation is None or found[2] not in {str(frames) for frames in FRAME_CASES} or found[3] != FPS:
        raise JobError(f"{Path(mp4).name} is {found[0]}x{found[1]}, {found[2]} frames at {found[3]} fps: not a 5 s "
                       f"16:9 or 9:16 clip (1344x768 or 768x1344, {' or '.join(map(str, FRAME_CASES))} frames at "
                       "24 fps)")
    return orientation, int(found[2])


def output_size(orientation, resolution):
    """The output's (width, height) for a clip of that orientation at that resolution."""
    return SHAPES[orientation]["sizes"][resolution]


def computed_grid(orientation, resolution):
    """The (width, height) the refine computes at: NVIDIA's RESOLUTIONS entry the job uses (with PORTRAIT_ENTRIES for
    9:16): 1920x1088 at 1080p, 2560x1440 at 2K, 3840x2176 at 4K, turned for 9:16. The pin's keyframes are put on it."""
    height, width, _ = {**NVIDIA_ENTRIES["RESOLUTIONS"], **PORTRAIT_ENTRIES["RESOLUTIONS"]}[
        SHAPES[orientation]["resolutions"][resolution]]
    return width, height


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
        raise JobError(f"the conditioning is {task}'s, not text alone (t2va): the studio gives fl2va and ref2va clips "
                       "a text-only encode of their prompt (qwen-text/, or the upscale's own encode)")
    tags = [int(tag) for tag in cond["text_token_tags"].flatten().tolist()]
    if not tags:
        raise JobError("the conditioning has no rows")
    other = sum(tag != 1 for tag in tags)
    if other:
        raise JobError(f"the conditioning has {other} non-text rows (fl2va/ref2va): it needs a text-only encode of "
                       "the prompt")
    shape = tuple(cond["prompt_embeds"].shape)
    if shape != (1, len(tags), TEXT_WIDTH):
        raise JobError(f"prompt_embeds {shape} do not match {len(tags)} text rows of {TEXT_WIDTH}")
    return len(tags)


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


def nvidia_files(package):
    """NVIDIA_FILES as they are in NVIDIA's package dir `package`: {path: sha256, or None for a file that is not
    there}. check_package compares them with the pins for a 9:16 clip, bin/lf-studio at start."""
    package = Path(package)
    return {rel: sha256(package / rel) if (package / rel).is_file() else None for rel in NVIDIA_FILES}


def plan(job, schedules, shifted_grid):
    """What the job will do, with NVIDIA's sigma functions (open_h3.sigmas, standard library only). `job` has the
    clip's orientation and frames (prepare)."""
    if job["start_sigma"] is None:
        if job["steps"] not in schedules:
            raise JobError(f"NVIDIA's package has no {job['steps']}-step schedule")
        sigmas = list(schedules[job["steps"]])
    else:
        sigmas = list(shifted_grid(job["start_sigma"], job["steps"]))
    planned = {"orientation": job["orientation"], "size": list(output_size(job["orientation"], job["resolution"])),
               "frames": job["frames"], "steps": len(sigmas) - 1, "sigmas": sigmas,
               "lean_chunks": LEAN_CHUNKS.get(job["resolution"])}
    if job.get("pin_check"):                      # check_pin's answer per end
        planned["pin"] = {anchor: {key: record[key] for key in ("used", "agreement_db", "frame", "why")}
                          for anchor, record in job["pin_check"].items()}
    return planned


def upscale_route(native, target, upscaled=UPSCALED_SHAPE):
    """The latent grids NVIDIA's Pipeline.upscale (pipeline.py:220-228) runs the upscaler to, from a native latent of
    shape `native` to `target` ((24, T, H, W) each; `upscaled`: NVIDIA's UPSCALED_SHAPE, the landscape 2K grid), with
    that 2K grid turned to the clip's orientation. Standard library only. NVIDIA's code goes through the 2K grid first
    when the target has more latent rows than it (target[2] > 90): for 16:9 that is 4K only, and this gives the same
    route. A 9:16 target has more rows than the landscape 2K grid at every resolution (120, 160, 240), so NVIDIA's
    code would stretch it through that grid and back; here its short side, the columns, is compared with the
    portrait 2K grid's (90): 4K only again, through (160, 90). [] when native is the target already."""
    native, target, upscaled = tuple(native), tuple(target), tuple(upscaled)
    if (native[2] > native[3]) != (target[2] > target[3]):
        raise ValueError(f"the latent {native} and the target {target} differ in orientation")
    if native == target:
        return []
    if target[2] > target[3]:                                       # portrait: the 2K grid turned, columns compared
        grid = upscaled[:2] + (upscaled[3], upscaled[2])
        through = target[3] > grid[3]
    else:                                                           # landscape: NVIDIA's own test
        grid, through = upscaled, target[2] > upscaled[2]
    return ([grid] if through else []) + [target]


# ---------------------------------------------------------------- the pin's check (Pillow and numpy, imported inside)

def cover_crop(size, canvas):
    """H3's cover-crop of a picture of `size` (w, h) onto `canvas` (w, h), as diffusers 0.40's MiniMaxH3ResizeStep
    fits a keyframe after the first (and jobs/h3turbo_job.py keyframe_fit records it): the size it is resized to,
    rounded, that covers the canvas, and the crop box in it, centred at (resized - canvas) // 2."""
    (w, h), (cw, ch) = size, canvas
    scale = max(cw / w, ch / h)
    rw, rh = max(cw, round(w * scale)), max(ch, round(h * scale))
    left, top = max(0, (rw - cw) // 2), max(0, (rh - ch) // 2)
    return (rw, rh), (left, top, left + cw, top + ch)


def fit_box(size, canvas, fit):
    """The part of a picture of `size` (after its EXIF turn) that the clip's model put on its `canvas` with `fit`, as
    a box (left, top, right, bottom) in the picture: None for "as is" and "stretch" (all of it); for "cover-crop"
    cover_crop's crop mapped back from the resized picture."""
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}")
    if fit != "cover-crop":
        return None
    (w, h), ((rw, rh), (left, top, right, bottom)) = size, cover_crop(size, canvas)
    return left * w / rw, top * h / rh, right * w / rw, bottom * h / rh


def keyframe_image(image, grid, fit, canvas):
    """A keyframe (an RGB PIL image, open_keyframe) at `grid` (w, h): the part the clip's model put on its canvas
    (fit_box) resized to the grid in one LANCZOS step from the picture, so no detail is lost at the canvas's size.
    With no box (as is, stretch) the whole picture stretched, as the bench wrapper's encode_keyframe and H3's own
    stretch do it; a picture of the grid's size as it is."""
    from PIL import Image
    box = fit_box(image.size, canvas, fit)
    if box is None and image.size == tuple(grid):
        return image
    return image.resize(tuple(grid), Image.Resampling.LANCZOS, box=box)


def keyframe_data(keyframe):
    """A keyframe's file as it is on disk: (its bytes, None) when its sha256 is job.json's, else (None, why)."""
    try:
        data = Path(keyframe["image"]).read_bytes()
    except OSError:
        return None, PIN_WHY["gone"]
    if hashlib.sha256(data).hexdigest() != keyframe["sha256"]:
        return None, PIN_WHY["changed"]
    return data, None


def open_keyframe(data):
    """A picture's bytes as the studio's models read it: EXIF orientation applied, then RGB (as jobs/h3turbo_job.py
    load_rgb and the bench wrapper's encode_keyframe)."""
    import io
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def frame_command(mp4, size, index, ffmpeg="ffmpeg"):
    """ffmpeg for frame `index` of the clip's MP4 at `size` (w, h), scaled as NVIDIA's encode_video scales it
    (cover, bilinear, centre crop): the clip's side of the pin's check."""
    width, height = size
    return [str(ffmpeg), "-nostdin", "-v", "error", "-i", str(mp4), "-vf",
            f"fps=24,select=eq(n\\,{int(index)}),scale={width}:{height}:force_original_aspect_ratio=increase:"
            f"flags=bilinear,crop={width}:{height}", "-frames:v", "1", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1"]


def clip_frame(mp4, size, index, ffmpeg="ffmpeg"):
    """Frame `index` of the clip's MP4 at `size` (frame_command): (h, w, 3) uint8."""
    import numpy as np
    width, height = size
    result = subprocess.run(frame_command(mp4, size, index, ffmpeg), capture_output=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0 or len(result.stdout) != width * height * 3:
        raise JobError(f"cannot read frame {index} of {Path(mp4).name} at {width}x{height}: "
                       f"{one_line(result.stderr.decode(errors='replace'), 300)}")
    return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)


def luma_psnr(a, b):
    """PSNR of the luma (BT.709 weights) of two (h, w, 3) uint8 pictures, in dB to two places; 99.0 when equal."""
    import numpy as np
    weights = np.array(LUMA)
    mse = float(np.mean((a.astype(np.float64) @ weights - b.astype(np.float64) @ weights) ** 2))
    return 99.0 if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2)


def check_pin(job, final, ffmpeg=None):
    """Which ends of job.json's pin the refine holds, checked before anything loads (in --check too); it never
    refuses, an end it does not keep is not used and says why. Per keyframe: its picture as on disk, by the sha256
    the clip was made from (else gone or changed); opened, EXIF applied (else unreadable); fitted as the clip's model
    fitted it (keyframe_image) at the check's grid, against the clip's own frame there (frame 0 for "first", its last
    for "last"): used at min_db or more (else far). A "last" end needs the clip to end on the frame H3 anchors it to,
    every frame made kept (else past_end: a Sol-H3 clip, cut to 8n+1 frames). -> {anchor: {"used", "agreement_db",
    "frame", "fit", "box" (fit_box, in the picture), "size" (the picture's, turned), "why"}}."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise JobError("the pin's check needs Pillow and numpy (the Upscale's environment has both)") from None
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise JobError("ffmpeg is not installed")
    pin, frames = job["pin"], job["frames"]
    grid, min_db = PIN_CHECK_GRIDS[pin["check"]["grid"]][job["orientation"]], pin["check"]["min_db"]
    results = {}
    for keyframe in pin["keyframes"]:
        anchor = keyframe["anchor"]
        frame = 0 if anchor == "first" else frames["kept"] - 1
        result = results[anchor] = {"used": False, "agreement_db": None, "frame": frame, "fit": keyframe["fit"],
                                    "box": None, "size": None, "why": None}
        data, result["why"] = keyframe_data(keyframe)
        if data is None:
            continue
        try:
            image = open_keyframe(data)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            result["why"] = PIN_WHY["unreadable"]
            continue
        box = fit_box(image.size, keyframe["canvas"], keyframe["fit"])
        result.update(size=list(image.size), box=None if box is None else [round(x, 2) for x in box])
        fitted = np.asarray(keyframe_image(image, grid, keyframe["fit"], keyframe["canvas"]))
        result["agreement_db"] = db = luma_psnr(clip_frame(final, grid, frame, ffmpeg), fitted)
        if anchor == "last" and frames["kept"] != frames["made"]:
            result["why"] = PIN_WHY["past_end"].format(n=frames["made"] - frames["kept"])
        elif db < min_db:
            result["why"] = PIN_WHY["far"].format(frame=frame, db=db, min=min_db)
        else:
            result["used"] = True
    return results


def pin_record(job):
    """The receipt's "pin": the ends asked for, each end's check (check_pin's dicts themselves, so what run() adds
    to them shows), the grids and the values the refine uses."""
    pin = job["pin"]
    return {"asked": [keyframe["anchor"] for keyframe in pin["keyframes"]], **job["pin_check"],
            "grid": list(computed_grid(job["orientation"], job["resolution"])),
            "check_grid": list(PIN_CHECK_GRIDS[pin["check"]["grid"]][job["orientation"]]),
            "min_db": pin["check"]["min_db"], "noise_aug": pin["noise_aug"], "encode_seed": pin["encode_seed"],
            "noise_seed": pin["noise_seed"]}


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
    clip's length (the frames it keeps at 24 fps, 121 or 124), as in the parent clip. No -shortest: the AAC stream ends
    a few ms before the last frame, and with -shortest that frame was dropped (120 of 121 frames)."""
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


def load_video_vae(base_model, torch, device="cuda"):
    """The FP32 H3 video VAE as NVIDIA's encode_video loads it, tiled (256 px, 64 px overlap)."""
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    return vae


def encode_clip(base_model, mp4, size, kept, pad, seed, torch, keyframes=(), encode_seed=None, device="cuda",
                vae_loader=None, ffmpeg=None):
    """NVIDIA's encode_video (pipeline.py:46-83) with the clip's frames as parameters: its MP4 read at `size` (w, h)
    as NVIDIA reads it (clip_read_command), `kept` frames, `pad` copies of the last one (NVIDIA's 3, to the VAE's
    124), the FP32 VAE's posterior sampled under `seed`, normalized -> (24, T, h/16, w/16) fp32 on the CPU,
    T = latent_frames(kept + pad). With 121 frames and 3 copies it is NVIDIA's encode_video. The pin's `keyframes`,
    [(anchor, (H, W, 3) uint8 pixels)], go through the same VAE before it is freed, each as H3 encodes a keyframe
    (diffusers' encode_vae_condition: its posterior sampled under `encode_seed`, rounded through fp16, normalized) ->
    (1, 24, 1, H/16, W/16). Returns (the clip's latent, {anchor: latent}, the keyframes' seconds)."""
    import numpy as np
    width, height = size
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run(clip_read_command(mp4, size, kept, pad, ffmpeg), capture_output=True, check=True).stdout
    if len(raw) != kept * height * width * 3:
        raise ValueError(f"could not decode {mp4} into {kept} frames at {width}x{height}")
    pixels = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(kept, height, width, 3).copy())
    pixels = pixels.permute(3, 0, 1, 2)[None].to(device).float() / 255.0
    if pad:
        pixels = torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
    vae = (vae_loader or load_video_vae)(base_model, torch, device)
    with torch.inference_mode():
        pixel_mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)
        posterior = vae.encode((pixels - pixel_mean) / pixel_std, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(seed))
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 24, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, 24, 1, 1, 1)
        latents = ((latents - mean) / std)[0].float().cpu()
    del posterior, pixels
    start, encoded = time.perf_counter(), {}
    if keyframes:
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        for anchor, frame in keyframes:
            frame = torch.from_numpy(frame).permute(2, 0, 1)[None, :, None].to(device)
            with torch.inference_mode():
                encoded[anchor] = encode_vae_condition(vae, frame, PIXEL_MEAN, PIXEL_STD, encode_seed=encode_seed)
    seconds = time.perf_counter() - start
    del vae
    torch.cuda.empty_cache()
    if latents.shape[1] != latent_frames(kept + pad):
        raise ValueError(f"expected {latent_frames(kept + pad)} latent frames, got {tuple(latents.shape)}")
    return latents, encoded, seconds


def decode_released(vae, latents, kept, torch, device="cuda"):
    """NVIDIA's Pipeline._decode_released (pipeline.py:268-279), keeping `kept` frames where it keeps 121:
    normalized (24, T, H, W) -> (kept, 16H, 16W, 3) uint8."""
    from diffusers.models.attention_dispatch import attention_backend
    dtype = next(vae.parameters()).dtype
    latents = latents[None].to(device).float()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 24, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, 24, 1, 1, 1)
    pixel_mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)
    with torch.inference_mode(), attention_backend("native"):
        pixels = vae.decode((latents * std + mean).to(dtype), return_dict=False)[0].float()
        return ((pixels * pixel_std + pixel_mean).clamp(0, 1)[0, :, :kept].permute(1, 2, 3, 0) * 255).round().byte()


def decode(pipe, latents, kept, torch, device="cuda"):
    """The refined latent to frames: NVIDIA's Pipeline.decode for a 121-frame clip; else its released-VAE decode with
    `kept` frames (decode_released) and its centre rows (pipeline.py:265-266)."""
    if kept == NVIDIA_FRAMES:
        return pipe.decode(latents)
    if pipe.fast_vae is not None:
        raise RuntimeError("the fast VAE keeps 121 frames: this job decodes with the released one")
    frames = decode_released(pipe.vae, latents, kept, torch, device)
    top = (frames.shape[1] - pipe.keep_rows) // 2
    return frames[:, top:top + pipe.keep_rows]


def tensor_sha256(tensor):
    """sha256 of a tensor's bytes, as the bench wrapper records its latents (so a receipt compares with its)."""
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def pin_pixels(job, grid):
    """The pictures of the ends check_pin kept, read again (one gone or changed since is not used: its record says
    why) and fitted to the refine's computed grid (keyframe_image): [(anchor, (h, w, 3) uint8)] in packed order."""
    import numpy as np
    pixels = []
    for keyframe in job["pin"]["keyframes"]:
        record = job["pin_check"][keyframe["anchor"]]
        if not record["used"]:
            continue
        data, why = keyframe_data(keyframe)
        if data is None:
            record.update(used=False, why=why)
            continue
        image = keyframe_image(open_keyframe(data), grid, keyframe["fit"], keyframe["canvas"])
        pixels.append((keyframe["anchor"], np.asarray(image).copy()))
    return pixels


def pin_layout(text_token_tags, geometry, anchors):
    """The packed layout of a pinned refine: diffusers' build_packed_sequence with `anchors` (H3's fl2va layout:
    [text | R keyframe rows per anchor, in packed order | audio | video]), and NVIDIA's t2va layout
    (build_t2va_layout) its positions are checked against. No anchors: the t2va layout itself. -> (layout, t2va)."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from open_h3.geometry import H3_AUDIO_TAG, H3_VIDEO_TAG, MiniMaxH3PackedLayout, build_t2va_layout
    tags = torch.as_tensor(text_token_tags, dtype=torch.long).flatten().cpu()
    t2va = build_t2va_layout(tags, geometry)
    if not anchors:
        return t2va, t2va
    layout = MiniMaxH3PackedLayout(*MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=tags, num_latent_frames=geometry.num_latent_frames, latent_height=geometry.latent_height,
        latent_width=geometry.latent_width, num_audio_latents=geometry.num_audio_latents,
        patch_size=geometry.patch_size, audio_channels=geometry.audio_channels, audio_tag=H3_AUDIO_TAG,
        video_tag=H3_VIDEO_TAG, keyframe_anchors=tuple(anchors)))
    rows = len(anchors) * geometry.rows_per_video_frame
    if layout.sequence_length != t2va.sequence_length + rows or layout.num_condition_video_rows != rows:
        raise RuntimeError(f"pin layout: {layout.sequence_length} rows, {layout.num_condition_video_rows} condition "
                           f"rows, t2va {t2va.sequence_length}, {len(anchors)} x R {geometry.rows_per_video_frame}")
    return layout, t2va


def check_pin_layout(layout, t2va, geometry, anchors, made):
    """The keyframe rows sit where H3 puts them (the bench wrapper's check_layout, per block): block i, the condition
    rows video[iR:(i+1)R], has generated frame 0's (h, w) grid, the video tag and one rotary time: frame 0's (the
    text's rows, L) for "first", L + (5/3)(made - 1) for "last" (the last frame made: 205 after the text at 5 s;
    diffusers sums it with numpy, so within 1e-3); every generated, audio and text row has its t2va position. Returns
    the facts for the receipt; RuntimeError when one fails."""
    import torch
    rows = geometry.rows_per_video_frame
    n = len(anchors) * rows
    if latent_frames(made) != geometry.num_latent_frames:
        raise RuntimeError(f"{made} frames made are not {geometry.num_latent_frames} latent frames")
    positions, video, tags = layout.position_ids.cpu(), layout.video_indices.cpu(), layout.token_tags.cpu()
    frame0 = positions[video[n:n + rows]]
    same = {"r_per_anchor": layout.num_condition_video_rows == n,
            "video_as_t2va": torch.equal(positions[video[n:]], t2va.position_ids[t2va.video_indices]),
            "audio_as_t2va": torch.equal(positions[layout.audio_indices.cpu()], t2va.position_ids[t2va.audio_indices]),
            "text_as_t2va": torch.equal(positions[layout.text_indices.cpu()], t2va.position_ids[t2va.text_indices])}
    blocks = []
    for index, anchor in enumerate(anchors):
        block = video[index * rows:(index + 1) * rows]
        where = positions[block]
        time = float(frame0[0, 0]) + (0.0 if anchor == "first" else 5 / 3 * (made - 1))
        same.update({f"{anchor}_at_frame0_grid": torch.equal(where[:, 1:], frame0[:, 1:]),
                     f"{anchor}_tag_video": bool((tags[block] == 0).all()),
                     f"{anchor}_time": bool((where[:, 0] - time).abs().max() <= 1e-3)})
        blocks.append({"anchor": anchor, "time": round(float(where[0, 0] - frame0[0, 0]), 4)})
    facts = {"anchors": list(anchors), "rows_per_frame": rows, "sequence_rows": layout.sequence_length,
             "t2va_rows": t2va.sequence_length, "condition_rows": layout.num_condition_video_rows, "made": made,
             "blocks": blocks, **same}
    if not all(same.values()):
        raise RuntimeError(f"pin layout check failed: {same}")
    return facts


def refine_pinned(pipe, low, text, seed, keyframes, made, noise_aug=PIN_NOISE_AUG, noise_seed=PIN_SEED):
    """NVIDIA's Pipeline.refine (pipeline.py:230-260, its forward_padded call with cp = 1) with the pin's keyframes
    held as H3 holds a keyframe: the bench wrapper's K1 refine, for either end or both. `keyframes`: [(anchor, latent
    (1, 24, 1, H, W))] in packed order. Each block of R rows is scale_noise(z, noise_aug, eps), eps drawn per block in
    that order from one CPU generator under noise_seed (H3's order), patchified and put in front of the generated
    rows (n = blocks x R rows, pin_layout); the model sees them in every step at t = max(1 - sigma, noise_aug), and
    only the rows after them are stepped (velocity[:, n:]): they are never written. With no keyframes the same loop
    is NVIDIA's refine (n = 0, forward_padded's timestep plan); run() calls NVIDIA's own then. Returns (the refined
    latent (24, T, H, W), facts: check_pin_layout's, the timesteps of each step, the condition rows' sha256)."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from open_h3.geometry import MiniMaxH3Geometry, build_row_timestep_plan, pack_audio_latents, unpatchify_video_rows
    device = pipe.device
    geometry = MiniMaxH3Geometry(num_latent_frames=low.shape[1], latent_height=low.shape[2],
                                 latent_width=low.shape[3], num_audio_latents=pipe.audio.shape[-1])
    rows = geometry.rows_per_video_frame
    anchors = tuple(anchor for anchor, _ in keyframes)
    layout, t2va = pin_layout(text["text_token_tags"], geometry, anchors)
    facts = check_pin_layout(layout, t2va, geometry, anchors, made)
    n = len(anchors) * rows
    if keyframes:
        want = (1, geometry.video_latent_channels, 1, geometry.latent_height, geometry.latent_width)
        generator = torch.Generator().manual_seed(noise_seed)
        t = torch.tensor(noise_aug, dtype=torch.float32)
        blocks = []
        for anchor, latent in keyframes:
            if tuple(latent.shape) != want:
                raise ValueError(f"the {anchor} keyframe's latent is {tuple(latent.shape)}, the refine's {want}")
            clean = latent.detach().float().cpu()
            eps = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
            noised = t * clean + (1.0 - t) * eps                  # MiniMaxH3Scheduler.scale_noise(clean, t, eps)
            blocks.append(patchify_video_latents(noised, geometry.patch_size).reshape(1, rows, -1))
        condition = torch.cat(blocks, 1).to(device)
        facts.update(noise_aug=noise_aug, noise_seed=noise_seed, condition_sha256=tensor_sha256(condition))
    else:
        condition, noise_aug = None, 1.0                          # exactly forward_padded's plan
    layout = layout.to(device)
    audio = pack_audio_latents(pipe.audio[None]).to(device)
    prompt = text["prompt_embeds"][None].to(device)
    low = patchify_video_latents(low[None], geometry.patch_size).reshape(
        1, geometry.num_video_rows, geometry.video_row_dim).to(device, torch.float32)
    for block in pipe.model.transformer_blocks:                   # as forward_padded (cp = 1: no padding)
        block.attn.processor.valid_rows = layout.token_tags.numel()
    facts["timesteps"] = []
    with torch.inference_mode():
        noise = torch.randn(low.shape, device=device, dtype=torch.float32,
                            generator=torch.Generator(device=device).manual_seed(seed))      # NVIDIA's draw
        sigmas = pipe.sigmas
        state = (1 - sigmas[0]) * low + sigmas[0] * noise
        for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pipe.on_step("refine", index)
            times, indices = build_row_timestep_plan(layout, video_sigma=sigma, audio_sigma=0.,
                                                     keyframe_noise_aug=noise_aug)
            facts["timesteps"].append([round(float(x), 6) for x in times.tolist()])
            video = state if condition is None else torch.cat([condition, state], 1)
            velocity, _ = pipe.model(
                hidden_states=video.to(torch.bfloat16), audio_hidden_states=audio, encoder_hidden_states=prompt,
                timestep=times, timestep_indices=indices, token_tags=layout.token_tags,
                position_ids=layout.position_ids, video_indices=layout.video_indices,
                audio_indices=layout.audio_indices, text_indices=layout.text_indices, attention_kwargs={},
                return_dict=False)
            state = state + (sigma - following) * velocity[:, n:].float()
    return unpatchify_video_rows(state, geometry)[0], facts


def check_package(adapters, pipeline, resolution, orientation="landscape", own=False):
    """NVIDIA's package is the one this wrapper was written against: its file names and shapes. For a 9:16 clip also
    the files the portrait route builds on (NVIDIA_FILES, by sha256), the landscape entries its portrait ones are
    made from, and no portrait key of NVIDIA's own in the dicts it adds to. With `own` (the run uses encode_clip or
    decode_released, copies of NVIDIA's code) also its pipeline.py, by sha256."""
    found = {"LORA_FILE": adapters.LORA_FILE, "CURVE_FILE": adapters.CURVE_FILE, "CURVE_TABLE": adapters.CURVE_TABLE,
             "UPSCALER_FILE": pipeline.UPSCALER_FILE, "NATIVE_SHAPE": tuple(pipeline.NATIVE_SHAPE)}
    wanted = {"LORA_FILE": LORA_FILE, "CURVE_FILE": CURVE_FILE, "CURVE_TABLE": CURVE_TABLE,
              "UPSCALER_FILE": UPSCALER_FILE, "NATIVE_SHAPE": NATIVE_SHAPE}
    height, width, keep = pipeline.RESOLUTIONS[resolution]
    found["size"], wanted["size"] = (width, keep), SIZES[resolution]
    if own and orientation != "portrait":
        rel = "open_h3/pipeline.py"
        found[rel] = nvidia_files(Path(pipeline.__file__).resolve().parent.parent)[rel]
        wanted[rel] = NVIDIA_FILES[rel]
    if orientation == "portrait":
        found.update(nvidia_files(Path(pipeline.__file__).resolve().parent.parent))
        wanted.update(NVIDIA_FILES)
        found["UPSCALED_SHAPE"], wanted["UPSCALED_SHAPE"] = tuple(pipeline.UPSCALED_SHAPE), UPSCALED_SHAPE
        for name, entries in NVIDIA_ENTRIES.items():
            table = getattr(pipeline, name)
            found[name] = {key: tuple(table[key]) if key in table else None for key in entries}
            found[name].update({key: tuple(table[key]) for key in PORTRAIT_ENTRIES[name] if key in table})
            wanted[name] = entries
    differ = {k: found[k] for k in wanted if found[k] != wanted[k]}
    if differ or not hasattr(adapters, "StageMerger"):
        raise JobError(f"NVIDIA's package is not the one this wrapper was written for ({NVIDIA_COMMIT[:7]}): {differ}")


def add_portrait_entries(pipeline):
    """PORTRAIT_ENTRIES into NVIDIA's module dicts (INPUT_SIZES, NATIVE_SHAPES, RESOLUTIONS), which its encode_video
    and Pipeline read at call time. check_package has made sure that the keys are not NVIDIA's own."""
    for name, entries in PORTRAIT_ENTRIES.items():
        getattr(pipeline, name).update(entries)


def keep_columns(frames, width):
    """The centre `width` columns of decoded (frames, rows, columns, 3) frames, as a view: NVIDIA's decode keeps the
    centre rows of a landscape output computed at whole latent patches (pipeline.py:265-266); a portrait output is
    computed 1088 / 2176 columns wide at 1080p and 4K."""
    left = (frames.shape[2] - width) // 2
    return frames.narrow(2, left, width)


def run_route(upscaler, node, native, route, device):
    """NVIDIA's upscaler (open_h3.upscaler.upscale) to each grid of `route` in turn (upscale_route), as its
    Pipeline.upscale calls it."""
    if not route:
        return native.float().to(device)
    for grid in route:
        native = upscaler.upscale(node, native, grid)
    return native.to(device)


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
    """The upscale on the GPU. Fills the receipt as it goes. A 16:9 clip of 121 frames runs NVIDIA's calls; a 9:16
    one adds its portrait entries to NVIDIA's dicts, and runs upscale_route and keep_columns; a clip of 124 frames
    or one with an end of the pin kept (check_pin) goes through encode_clip, and with 124 frames decode_released; the
    pin through refine_pinned (the docstring)."""
    import torch
    sys.path.insert(0, str(job["package"]))           # -I drops PYTHONPATH and the script dir: name it explicitly
    from open_h3 import adapters, pipeline
    shape = SHAPES[job["orientation"]]
    portrait = job["orientation"] == "portrait"
    kept, pad = job["frames"]["kept"], job["frames"]["pad"]
    pin = receipt.get("pin")                          # with an end the check kept: the keyframes' rows in the refine
    used = [anchor for anchor in ANCHORS if pin and pin.get(anchor, {}).get("used")]
    own = kept != NVIDIA_FRAMES or bool(used)         # encode_clip (and decode_released), NVIDIA's with the frames
    check_package(adapters, pipeline, job["resolution"], job["orientation"], own=own)
    if portrait:
        from open_h3 import upscaler
        add_portrait_entries(pipeline)
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
    pcm = clip_audio = None
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

    # the final MP4 (and the pin's pictures) and the clip's audio through the VAEs, loaded, used and freed before
    # the DiT
    phase("encoding")
    keyframes = {}
    if pin:
        for anchor in pin["asked"]:
            end = pin[anchor]
            say(f"pin {anchor}: frame {end['frame']} " + (f"{end['agreement_db']} dB from the picture, "
                                                          if end["agreement_db"] is not None else "") +
                ("held in the refine" if end["used"] else f"not held: {end['why']}"))
    if own:
        size = pipeline.INPUT_SIZES[shape["size"]]   # what NVIDIA's encode_video reads the clip at
        pixels = pin_pixels(job, tuple(pin["grid"])) if used else []
        native, keyframes, seconds = timed("encode_video", lambda: encode_clip(
            base, files["final"], size, kept, pad, VAE_SEED, torch, keyframes=pixels,
            encode_seed=job["pin"]["encode_seed"] if pixels else None))
        if pixels:
            pin["encode_s"] = round(seconds, 2)
            for anchor, latent in keyframes.items():
                pin[anchor].update(latent_shape=list(latent.shape), latent_sha256=tensor_sha256(latent))
            say(f"pin: {', '.join(keyframes)} at {pin['grid'][0]}x{pin['grid'][1]} encoded in {seconds:.1f} s")
    else:
        native = timed("encode_video", lambda: pipeline.encode_video(base, files["final"], shape["size"],
                                                                     seed=VAE_SEED))
    if pin:
        pin["native_sha256"] = tensor_sha256(native)
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
        resolution=shape["resolutions"][job["resolution"]], vae="standard", vae_batch=job["vae_batch"]))
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
    if tuple(native.shape) != shape["native"]:
        raise RuntimeError(f"the 768p latent is {tuple(native.shape)}, not {shape['native']}")
    route = upscale_route(native.shape, pipe.target_shape, pipeline.UPSCALED_SHAPE)
    receipt["upscale_route"] = [list(grid[2:]) for grid in route]      # NVIDIA's own for 16:9 (the same grids)
    if portrait:
        low = timed("upscale", lambda: run_route(upscaler, pipe.node, native.to(pipe.device), route, pipe.device))
    else:
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
        if keyframes:                                 # the ends the check kept, in packed order
            result, facts = refine_pinned(pipe, low, text, job["seed"], list(keyframes.items()),
                                          job["frames"]["made"], job["pin"]["noise_aug"], job["pin"]["noise_seed"])
            pin.update(timesteps=facts.pop("timesteps"), condition_sha256=facts.pop("condition_sha256"),
                       layout={k: v for k, v in facts.items() if k not in ("noise_aug", "noise_seed")})
        else:
            result = pipe.refine(low, text, job["seed"])
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        return result
    refined = timed("refine", refine)
    receipt["steps_s"] = [round(b - a, 2) for a, b in zip(marks, marks[1:])]
    if keyframes:
        pin["refined_sha256"] = tensor_sha256(refined)
        say(f"pin: refined with {', '.join(keyframes)} held ({pin['layout']['sequence_rows']} rows, "
            f"{pin['layout']['condition_rows']} of them the keyframes'), timesteps {pin['timesteps']}")

    phase("decoding")
    pipe.model = None                                 # the DiT, ~62 GiB: nothing runs on it any more
    getattr(pipe.node, "MODEL_CACHE", {}).clear()     # the latent upscaler
    del low, native
    gc.collect()
    torch.cuda.empty_cache()
    receipt["allocated_before_decode_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    frames = timed("vae_decode", lambda: decode(pipe, refined, kept, torch))
    width, height = shape["sizes"][job["resolution"]]
    if portrait:
        frames = keep_columns(frames, width)
    if tuple(frames.shape) != (kept, height, width, 3):
        raise RuntimeError(f"the decoded video is {tuple(frames.shape)}, not {kept} frames of {width}x{height}")
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
    """job.json and the inputs, checked without torch -> (job, files), job["orientation"] and job["frames"] from the
    clip's MP4, and with a pin job["pin_check"] (check_pin). Fills the receipt's job as soon as it parses."""
    job = load_job(job_path)
    receipt["job"] = job_record(job)
    files = case_inputs(job)
    orientation, kept = check_final(files["final"])         # from the MP4 (not in job.json)
    job["orientation"] = receipt["orientation"] = orientation
    job["frames"] = receipt["frames"] = frame_case(kept)
    check_weights(job)
    if job["pin"]:
        job["pin_check"] = check_pin(job, files["final"])
        receipt["pin"] = pin_record(job)
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
