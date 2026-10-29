#!/usr/bin/env python3
"""Experimental LightX2V Turbo (the studio's model id: lightx2v-turbo, MODEL): one clip from the full MiniMax-H3
transformer in one stage, with a LightX2V fl2v turbo LoRA, at 576p or 768p, 16:9 or 9:16, 5 to 15 s. One clip per
process, run by the studio's server in the Regen-2K environment:

    $LFSTUDIO_H3TURBO_PYTHON -I -B jobs/h3turbo_job.py --job <case>/turbo/job.json [--parent PID]
    $LFSTUDIO_H3TURBO_PYTHON -I -B jobs/h3turbo_job.py --job <case>/turbo/job.json --check
    CUDA_VISIBLE_DEVICES= $LFSTUDIO_H3TURBO_PYTHON -I -B jobs/h3turbo_job.py --check-install

A port of the research probe that measured this route on a DGX Spark (2026-10-28; 17 clips at 5 s), cut down to one
job per process, with the probe's numerics: the same pipeline calls, sigma rule, LoRA arithmetic, generator use and
audio-noise rule, so the same conditioning, image and seed give the probe's latents, bit for bit, with the same
attention backend (on the CPU with small random-weight models: the same latents and MP4). What runs: diffusers
0.40.0's modular MiniMax-H3 blocks, stock, with two changes of wiring only:
- no text encoder: the `text_encoder` block is removed and the clip's qwen/conditioning.pt (LF Studio's NVFP4 Qwen
  encode: prompt_embeds [1, L, 5120] bf16 and text_token_tags [L], each keyframe a run of tag-0 rows) is passed in
  its place;
- the decode blocks run as a second pipeline call, so the latents are saved and the DiT dropped in between.
Everything else is the stock path. t2va (no keyframe) passes no image, and diffusers' auto blocks run their text-only
workflow. fl2va passes the first frame as `image` and the last as `last_image` (EXIF transpose, then RGB). diffusers
(MiniMaxH3ResizeStep) stretches the first keyframe given (the first frame, or a lone last frame) onto the canvas with
LANCZOS; with both, the last is cover-cropped (scaled with LANCZOS to the rounded size that covers the canvas, then
cut at (size - canvas) // 2), as the studio's Qwen worker does at its own canvas. Each keyframe is VAE-encoded with
its posterior sampled under seed 42 and rounded through fp16, noised to t = 0.999 and held there as extra rows in
front of the generated video rows. One CPU torch.Generator per clip, seeded `seed` (the reference's,
inference_minimax_h3.py:1152), draws the keyframe noise first, then the video noise. With audio_noise "seed" (the
default) the audio noise comes from a generator of its own, seeded (seed + 0x9E3779B9) mod 2^63, so a 576p clip and
a 768p re-render of one seed share their audio noise; "reference" draws it from the clip's generator after the video
noise, as diffusers does. No CFG (the checkpoint is guidance-distilled): one forward per step.

The turbo LoRA. Sigmas: scheduler.set_shift(shift[0]) and audio_scheduler.set_shift(shift[1]), num_inference_steps =
steps + 1 (the H3 scheduler counts grid points, terminal 0 included; LightX2V's reference does the same); the sigmas
the scheduler used are checked against the closed form s*q/(1+(s-1)q), q = (N-i)/N, and the forwards against the
steps: a mismatch fails the clip. Scale: alpha / rank, alpha from the file's __metadata__ (8 for v1.2 and later: 8 /
128 = 0.0625). Applied UNFUSED, as a bf16 branch on each of the 312 target Linear modules through forward hooks:
PEFT's LoRA Linear arithmetic (peft/tuners/lora/layer.py, result + lora_B(lora_A(x)) * scaling, the adapter in the
base layer's dtype, which is what the reference's add_adapter gives), with the scale and the add done in place (the
same bf16 roundings, one output-sized tensor fewer alive). A bf16 merge would keep only 4-20 % of these alpha-8
deltas. The base weights are never written. A file without an alpha in its metadata (4step_v0.1) and LightX2V's
*_comfyui_* files are refused, and so is a file whose size does not match its own header (a cut download), or one of
PRESETS whose size or step count differs.

Loading (job.json "load"). "fast" (the default) is LF Studio's fast load, as the Upscale's job installs it
(upscale/regen2k_job.py, without NVIDIA's adapters): safetensors.torch.load_file becomes
upscale/fastload.py's load_file(device="cuda") (O_DIRECT windows, each tensor copied straight to the GPU, no host copy
and no page cache; loaded by its path, not copied here) and from_pretrained of the three model classes gets
device_map {"": "cuda"} unless the call names one; where O_DIRECT cannot be used it falls back to safetensors' own
load_file, to the GPU too (the receipt counts the files and the fallbacks). The probe loaded 18 files in 15.7 s this
way, none through the fallback. "standard" is diffusers' own loading. The transformer is bf16 (62 GB), the VAEs fp32.

Memory (GB10: one 121 GiB memory for CPU and GPU). A thread samples MemAvailable every 0.5 s; under
abort_below_gib (6 by default) the clip is aborted at the next transformer block, VAE decoder call or phase boundary
(receipt state "aborted", exit status 3); the studio's own guard is the backstop. After the denoise the latents are
saved and the DiT is dropped (its last use, as the Upscale's job does) before the decode. Measured with the probe
(5 s, 4 steps, the model loaded, MemAvailable 108.5 GiB at its start): 576p steps of 27.4-28.9 s, decode 44.4 s, 160 s
in all (165 s with a last frame), peak allocated 77.1-77.5 GiB, lowest MemAvailable 35.0-35.4 GiB; 768p steps of
57.2-58.4 s, 320 s, peak 80.1-80.2 GiB, lowest 31.0-32.1 GiB.

Lengths. The frames made are 24 fps snapped up to 17k+5 (frames_for_seconds: 124 at 5 s, 362 at 15 s), as the
studio's Sol-H3 draws them. diffusers 0.40 refuses more than 15.0 s of them (MiniMaxH3ModularPipeline.max_duration,
checked on the frames made), and the studio's 15 s makes 362, 15.083 s: the same 362 frames the studio's Sol-H3 draws
with the same MiniMax-H3 transformer for its own 15 s clips (the patch's runtime/geometry.py takes up to 15.2 s). For
such a clip, and only for it, the job gives its pipeline instance a class of its own whose max_duration is the clip's
length (allow_length; diffusers' files, its class and every other check unchanged), and the receipt notes it. 14 s
(345 frames, 14.375 s) is the longest that stock diffusers 0.40 takes. The MP4 keeps the studio's length, the largest
8m+1 frames of the 17k+5 made (studio_frames: 121 of 124 at 5 s, 361 of 362 at 15 s), with frames / 24 * 32000
samples (161,333 at 5 s, of the decoded 165,600), as the studio's Sol clips do; a clip with a last keyframe keeps every
frame made (kept_frames: 124 at 5 s, 5.17 s; 345 of 345 at 14 s either way), because the last keyframe is anchored
to the last frame made: so its MP4 ends on it, with no frame dropped in between and nothing re-timed. Its audio is cut
to the same length (165,333 samples at 5 s); where the decoded audio (round(frames / 24 * 40) latents of 800 samples)
is shorter than that, at 6 and 15 s by 266 samples (8 ms), its last samples are mirrored onto the end.

Output, in paths.output (default: the directory of job.json): clip_<W>x<H>_<frames>f.mp4 (24 fps, H.264 yuv420p at
job.json's crf / x264_preset, the studio's 19 / veryfast by default, AAC stereo 32 kHz; the frames and samples kept,
above), audio.pt ({"audio": (1, 2, samples) float32}: the decoded PCM before AAC, trimmed as the MP4 is),
latents.safetensors ("video" (1, 24, T, H/16, W/16) and "audio" (2, 32, A) float32: every generated frame, normalized)
and receipt.json (state running, done, failed, refused, aborted or cancelled; the model (MODEL), the job, the
geometry and the layout rows, the sigmas, the LoRA's header facts, a raised max_duration, times per phase (torch's
and diffusers' imports too: with the load they are what a clip per process costs; encode_1, encode_2 for each
keyframe's VAE encode), peak GPU memory, lowest
MemAvailable, the inputs' sha256, the latents' sha256 per tensor, how each keyframe was fitted to the canvas (images:
"stretch" with its ratio, or "cover-crop" with the crop box, and the sha256 of the pixels diffusers encoded; a note
when one is stretched by more than FIT_RANGE allows or mostly cut away), frame 0 against the first keyframe as fitted
(f0_vs_keyframe, PSNR and MAE), the last frame made, where a last keyframe is anchored (frame 123 of 124 at 5 s),
against it (last_vs_keyframe) and the MP4's last frame against it (last_kept_vs_keyframe: the same frame, since such a
clip keeps every frame made), versions, the load). Progress on stdout: "[lf-studio] phase <name>" for loading, step
k/N, decoding and saving, in that order, and "[lf-studio] error <text>" last on failure.
Exit status: 0 done, 1 failed, 2 the job or its inputs were refused (nothing loaded), 3 aborted for memory, 143
stopped by SIGTERM.
--check validates job.json and its inputs without torch (the LoRA's sha256 against PRESETS too), prints the plan and
exits (0 or 2), writing nothing. --check-install checks the environment (diffusers 0.40.0 and the blocks this job
calls), the base model and the LoRA files of PRESETS where the LFSTUDIO_H3TURBO_* settings put them (studio_settings),
without the GPU and without reading weights; the server reads the same settings with studio_settings (bin/env.example
has them). --parent PID (the server passes its own): the job is killed when that process ends (PR_SET_PDEATHSIG); if
that process is gone already, it exits at once (143).

job.json: {"version": 1, "task": "t2va"|"fl2va", "orientation": "landscape"|"portrait", "resolution": "576p"|"768p",
"seconds": 5-15, "seed": INT, "steps": INT, "lora": FILE, "shift": [VIDEO, AUDIO], "conditioning": FILE (its
conditioning.json next to it), "qwen_canvas": "draft"|"canvas", "first_frame": FILE|null, "last_frame": FILE|null,
"audio_noise": "seed"|"reference" ("seed"), "crf": 0-51 (19), "x264_preset": NAME ("veryfast"), "load":
"fast"|"standard" ("fast"), "abort_below_gib": GiB (6), "paths": {"base_model": DIR, "output": DIR (optional)},
"meta": {...} (optional, copied into the receipt)}. Paths are absolute; any other field is refused. t2va takes no
frames; fl2va a first frame, first and last, or a last frame alone. qwen_canvas says where Qwen saw the keyframes:
"draft" the studio's 384p draft canvas (672x384 or 384x672: 252 image tokens per keyframe; the studio's default),
"canvas" the generation canvas (576 tokens at 576p, 1008 at 768p); the conditioning must match it.

The studio's model of a clip (the GPU validation's receipts on one DGX Spark, 2026-10-29: 15 clips, 576p at 5, 10
and 14 s, 768p at 5 and 10 s, 4 steps; the rest estimated from them): MemAvailable at its lowest is the start minus
(USE_GIB[0] + USE_GIB[1] per 1000 rows of the packed sequence, layout_rows), within USE_SPREAD_GIB of every clip;
min_free_for is the floor a clip of a size and length needs to start (the setting's, or more where that would not keep
the job above its abort_below_gib: long 768p clips). A step takes STEP_S[0] N + STEP_S[1] N^2 seconds (N: rows /
1000), the decode DECODE_S per frame and megapixel; estimate_s adds the prompt encode, the process (imports, load,
saving) and the poster: what the studio's registry shows until a clip has been measured on the machine.

Everything outside run() and the torch helpers is standard library only, so the server, bin/lf-studio --check and
tests/test_h3turbo_job.py import this file without torch.
"""
import argparse
import contextlib
import functools
import gc
import glob
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

TAG = "[lf-studio]"
MODEL = "lightx2v-turbo"         # the studio's id of this model (webui/server.py MODELS), in every receipt
JOB_VERSION = 1
REPO = Path(__file__).resolve().parents[1]
FASTLOAD = REPO / "upscale" / "fastload.py"     # the Upscale's copy of the patch's runtime/fastload.py, by its path
DEVICE = "cuda"
MEMINFO = "/proc/meminfo"
FPS = 24
AUDIO_RATE = 32000
TEXT_WIDTH = 5120
TASKS = ("t2va", "fl2va")
RESOLUTIONS = ("576p", "768p")
ORIENTATIONS = ("landscape", "portrait")
# (width, height) per resolution and orientation, all multiples of 32: 768p is the 768p turbo LoRAs' training canvas
# and H3's default (resolve_canvas_size(16, 9, 32, 768, 768 * 1344)); 576p is NVIDIA's draft size
CANVAS = {("576p", "landscape"): (1024, 576), ("576p", "portrait"): (576, 1024),
          ("768p", "landscape"): (1344, 768), ("768p", "portrait"): (768, 1344)}
# where Qwen sees the keyframes (job.json "qwen_canvas"): "draft" is the studio's Stage-1 draft canvas
# (runtime/qwen.py draft_canvas: the recipe's stage1 width x height), "canvas" the generation canvas (CANVAS)
QWEN_CANVASES = ("draft", "canvas")
DRAFT_CANVAS = {"landscape": (672, 384), "portrait": (384, 672)}
# a keyframe stretched by a ratio outside this range, or cover-cropped to less than its first value, gets a receipt
# note (the research probe refused a first frame of the other orientation; the studio decides what it allows)
FIT_RANGE = (0.8, 1.25)
MIN_SECONDS, MAX_SECONDS = 5, 15             # whole seconds; the studio offers 5, 6, 7, 8, 10, 12 and 15
STOCK_MAX_DURATION_S = 15.0      # diffusers 0.40: MiniMaxH3ModularPipeline.max_duration, of the 17k+5 frames made
AUDIO_NOISES = ("seed", "reference")
AUDIO_SEED_OFFSET = 0x9E3779B9   # audio_noise "seed": the audio generator's seed is (seed + this) mod 2^63
LOADS = ("fast", "standard")
X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
CRF, X264_PRESET = 19, "veryfast"        # the studio's final MP4 (Sol-H3-Spark stage2.py)
ABORT_BELOW_GIB = 6
MAX_STEPS = 64
ATTENTION_BACKEND = "_native_cudnn"      # the probe's, and NVIDIA's standard forward's (dense cuDNN attention)
DIFFUSERS_VERSION = "0.40.0"
POPPED_BLOCKS = ("text_encoder", "decode")
DIRECT_LOAD = ("MiniMaxH3Transformer3DModel", "AutoencoderKLMiniMaxH3", "AutoencoderKLMiniMaxH3Audio")
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")     # inference_minimax_h3.py:79-86
LORA_MODULES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
LORA_A, LORA_B = ".lora_A.default.weight", ".lora_B.default.weight"                    # inference_minimax_h3.py:87-88
# MiniMax's 7 prompt tokens (Sol-H3-Spark qwen_ops/prompt_tokens.py, MiniMax's tokenizer_config.json): the studio's
# "plain" prompt-token mode split them into text tokens H3 never saw
MINIMAX_PROMPT_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>",
                         "<|caption_end|>")
# the turbo LoRAs (lightx2v/Minimax-h3-Turbo at LORA_REVISION, the diffusers-format files): steps -> file, bytes,
# sha256 and the (video, audio) shift LightX2V documents for the 768p files. The studio offers OFFERED_STEPS.
LORA_REPO = "lightx2v/Minimax-h3-Turbo"
LORA_REVISION = "30f485ed874aa44c8183348ad361c5eabad318da"
PRESETS = {
    4: {"file": "minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors", "bytes": 1383677808,
        "sha256": "97b775707bb5732e01dba0cf3833d3fe2947c953859d4d05ea3bd74894c43d2b", "shift": [6, 3]},
    8: {"file": "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors", "bytes": 1383677808,
        "sha256": "fb41185df94bf01f9ac1aeedc2f07e5f748da3b237d18a16f5b3f5960839fdab", "shift": [6, 3]}}
OFFERED_STEPS = (4,)
# the studio's settings for this model (studio_settings): its memory floor before a clip starts and its guard while
# one runs, GiB of MemAvailable (the probe started at 108.5 and came down to 31.0 at 768p)
MIN_FREE_GIB = 100
GUARD_GIB = 5
# the studio's model of a clip on a DGX Spark (the docstring; the GPU validation's receipts): the MemAvailable a clip
# takes at its lowest, start - (77.8 + 0.186 per 1000 rows), within 1.6 GiB of every one of the 15 clips (21.7k-75.1k
# rows); a step, 0.848 N + 0.016 N^2 seconds (N = rows / 1000; within 1 s, 21.7k-75.1k rows); the decode, 0.67 s per
# frame and megapixel (0.64-0.69); the rest of the process (imports, load, LoRA, saving), 23 s (21.8-26.4); the prompt
# encode, 12 s (9-18 s); 2 s for the poster and the record (an allowance). The floor's rows: two keyframes and 1000
# conditioning rows (the validation's: 156-1441); the estimate's: one keyframe and 500.
USE_GIB = (77.8, 0.186)
USE_SPREAD_GIB = 1.6
STEP_S = (0.848, 0.016)
DECODE_S = 0.67
PROCESS_S, ENCODE_S, POSTER_S = 23, 12, 2
FLOOR_ROWS, ESTIMATE_ROWS = (2, 1000), (1, 500)       # (keyframes, conditioning rows)
BASE_PARTS = ("transformer", "vae", "audio_vae", "scheduler", "audio_scheduler")
WEIGHT_PARTS = ("transformer", "vae", "audio_vae")
AUDIO_FILE = "audio.pt"
LATENTS_FILE = "latents.safetensors"
RECEIPT = "receipt.json"
JOB_KEYS = {"version", "task", "orientation", "resolution", "seconds", "seed", "steps", "lora", "shift",
            "conditioning", "qwen_canvas", "first_frame", "last_frame", "audio_noise", "crf", "x264_preset", "load",
            "abort_below_gib", "paths", "meta"}
REQUIRED = ("task", "orientation", "resolution", "seconds", "seed", "steps", "lora", "shift", "conditioning",
            "qwen_canvas", "paths")
PATH_KEYS = ("base_model", "output")
EXIT_FAILED, EXIT_REFUSED, EXIT_ABORTED, EXIT_CANCELLED = 1, 2, 3, 128 + signal.SIGTERM


class JobError(ValueError):
    """The job or its inputs are refused (exit status 2)."""


class RunAborted(Exception):
    """MemAvailable fell under abort_below_gib during the clip (exit status 3)."""


class Cancelled(BaseException):
    """SIGTERM (the studio's Cancel): a BaseException, so no `except Exception` on the way swallows it."""


# ---------------------------------------------------------------- geometry and schedules (standard library only)

def frames_for_seconds(seconds):
    """The frames made: 24 fps snapped up to 17n+5 (the reference's rule, and the patch's draft_frames_for_seconds):
    5 s -> 124."""
    n = max(5, round(float(seconds) * FPS))
    return n + (5 - n % 17) % 17


def studio_frames(draft_frames):
    """The studio's length for that many made (the patch's geometry.output_frames_for): 124 -> 121."""
    return ((int(draft_frames) - 1) // 8) * 8 + 1


def kept_frames(frames, keyframes):
    """The frames the MP4 keeps of the `frames` made: the studio's length (studio_frames), or every frame made for a
    clip with a last keyframe, which is anchored to the last frame made (124 of 124 at 5 s): so the clip ends on it."""
    return int(frames) if "last" in keyframes else studio_frames(frames)


def max_duration_for(frames):
    """The pipeline's max_duration a clip of `frames` needs: None within diffusers' stock STOCK_MAX_DURATION_S, else
    its own length (362 frames at 15 s: 15.083 s; allow_length)."""
    duration = frames / FPS
    return None if duration <= STOCK_MAX_DURATION_S else duration


def latent_frames(frames):
    return (frames - 5) // 17 * 5 + 2


def audio_latents(frames):
    return int(round(frames / FPS * 40))


def kept_samples(frames):
    """The audio samples kept for that many frames (the patch's frame_geometry audio_samples): 121 -> 161,333."""
    return int(frames / FPS * AUDIO_RATE)


def expected_sigmas(steps, shift):
    """LightX2V's grid: q_i = (N - i) / N, sigma = s q / (1 + (s - 1) q), i = 0..N (terminal 0 included)."""
    out = []
    for i in range(steps + 1):
        q = (steps - i) / steps
        out.append(shift * q / (1 + (shift - 1) * q))
    return out


def layout_rows(text_rows, width, height, frames, keyframes=1):
    """Rows of the packed sequence [text | keyframe rows | audio | video] (diffusers before_denoise.py), text_rows
    being every row of the conditioning (its vision rows too)."""
    per_frame = (height // 32) * (width // 32)
    cond = keyframes * per_frame
    audio = 2 * audio_latents(frames)
    video = latent_frames(frames) * per_frame
    return {"text": text_rows, "condition": cond, "audio": audio, "video": video,
            "sequence": text_rows + cond + audio + video, "per_frame": per_frame}


def qwen_canvas(orientation, resolution, mode):
    """(width, height) at which Qwen saw the keyframes for job.json's qwen_canvas `mode`."""
    return DRAFT_CANVAS[orientation] if mode == "draft" else CANVAS[(resolution, orientation)]


def image_tokens(width, height):
    """Qwen's image tokens for a keyframe at (width, height): one per 32x32 pixels (252 at the 672x384 draft)."""
    return (width // 32) * (height // 32)


def keyframe_fit(size, canvas, first):
    """How diffusers 0.40 (MiniMaxH3ResizeStep) puts a keyframe of `size` (w, h) onto `canvas` (w, h): the first
    keyframe given (`first`: the first frame, or a lone last frame) is stretched, the one after it cover-cropped
    (LANCZOS to the rounded size that covers the canvas, cut at (size - canvas) // 2); one of the canvas's size is
    taken as it is. Returns the receipt's record: "fit", and the stretch ratio, or the resized size, the crop box in
    it and the share of it kept."""
    (w, h), (cw, ch) = size, canvas
    if (w, h) == (cw, ch):
        return {"fit": "as is"}
    if first:
        return {"fit": "stretch", "stretch": round((cw / ch) / (w / h), 5)}
    scale = max(cw / w, ch / h)
    rw, rh = max(cw, round(w * scale)), max(ch, round(h * scale))
    left, top = max(0, (rw - cw) // 2), max(0, (rh - ch) // 2)
    return {"fit": "cover-crop", "resized": [rw, rh], "crop": [left, top, left + cw, top + ch],
            "kept": round(cw * ch / (rw * rh), 4)}


def fit_note(anchor, size, canvas, fit):
    """A receipt note for a keyframe whose shape is far from the canvas's (outside FIT_RANGE): stretched, its shapes
    are distorted; cover-cropped, much of it is cut away. None otherwise."""
    if fit["fit"] == "stretch" and not FIT_RANGE[0] <= fit["stretch"] <= FIT_RANGE[1]:
        return (f"the {anchor} frame ({size[0]}x{size[1]}) is stretched by {fit['stretch']:g} onto the "
                f"{canvas[0]}x{canvas[1]} canvas: its shapes are distorted (another orientation than the clip's?)")
    if fit["fit"] == "cover-crop" and fit["kept"] < FIT_RANGE[0]:
        return (f"the {anchor} frame ({size[0]}x{size[1]}) is cover-cropped to the {canvas[0]}x{canvas[1]} canvas: "
                f"{round(100 * (1 - fit['kept']))} % of it is cut away")
    return None


def geometry(job):
    """What a job's settings give: canvas, frames made and kept (kept_frames), latent sizes, samples kept, the
    pipeline's max_duration it needs (null: stock), sigmas."""
    width, height = CANVAS[(job["resolution"], job["orientation"])]
    frames = frames_for_seconds(job["seconds"])
    keep = kept_frames(frames, job["keyframes"])
    qw, qh = qwen_canvas(job["orientation"], job["resolution"], job["qwen_canvas"])
    return {"width": width, "height": height, "frames": frames, "latent_frames": latent_frames(frames),
            "audio_latents": audio_latents(frames), "keep_frames": keep, "keep_samples": kept_samples(keep),
            "max_duration_s": max_duration_for(frames),
            "num_inference_steps": job["steps"] + 1,
            "sigmas": expected_sigmas(job["steps"], float(job["shift"][0])),
            "audio_sigmas": expected_sigmas(job["steps"], float(job["shift"][1])),
            "qwen_canvas": [qw, qh], "image_tokens": image_tokens(qw, qh)}


def output_name(job):
    g = geometry(job)
    return f"clip_{g['width']}x{g['height']}_{g['keep_frames']}f.mp4"


def sequence_rows(resolution, seconds, keyframes, text_rows):
    """The rows of a clip's packed sequence at `resolution` (either orientation) and `seconds`, with `keyframes`
    keyframes and `text_rows` conditioning rows (layout_rows)."""
    width, height = CANVAS[(resolution, "landscape")]
    return layout_rows(text_rows, width, height, frames_for_seconds(seconds), keyframes)["sequence"]


def min_free_for(resolution, seconds, floor):
    """The MemAvailable a clip of this size and length needs to start: the studio's floor
    (LFSTUDIO_H3TURBO_MIN_FREE_GIB, MIN_FREE_GIB by default), or more where the model of the docstring says the job
    would come down below its abort_below_gib (ABORT_BELOW_GIB) plus USE_SPREAD_GIB from it: 768p from 12 s (103 GiB at
    12 s, 107 at 15 s). A floor of 0 (no check) stays 0."""
    if floor <= 0:
        return floor
    use = USE_GIB[0] + USE_GIB[1] * sequence_rows(resolution, seconds, *FLOOR_ROWS) / 1000
    return max(floor, math.ceil(use + ABORT_BELOW_GIB + USE_SPREAD_GIB))


def estimate_s(resolution, seconds, steps):
    """Seconds a clip takes in the studio, from its prompt encode to its poster (the server's e2e_s), by the model of
    the docstring: an estimate (the GPU validation measured 576p at 5, 10 and 14 s and 768p at 5 and 10 s)."""
    n = sequence_rows(resolution, seconds, *ESTIMATE_ROWS) / 1000
    width, height = CANVAS[(resolution, "landscape")]
    decode = DECODE_S * frames_for_seconds(seconds) * width * height / 1e6
    return ENCODE_S + PROCESS_S + steps * (STEP_S[0] * n + STEP_S[1] * n * n) + decode + POSTER_S


# ---------------------------------------------------------------- job.json (standard library only)

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _path(value, name, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or "\0" in value or not os.path.isabs(value):
        raise JobError(f"{name}: an absolute path is needed, got {value!r}")
    return Path(value)


def _choice(data, key, choices, default=None):
    value = data.get(key, default)
    if not isinstance(value, str) or value not in choices:
        raise JobError(f"{key}: one of {', '.join(choices)}, got {value!r}")
    return value


def parse_job(data, job_dir):
    """job.json's object -> the job: a dict with Path values for the files and directories. Raises JobError naming
    the field. Checks values only; check_inputs looks at the files."""
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
    job = {"version": JOB_VERSION}
    for key, choices, default in (("task", TASKS, None), ("orientation", ORIENTATIONS, None),
                                  ("resolution", RESOLUTIONS, None), ("qwen_canvas", QWEN_CANVASES, None),
                                  ("audio_noise", AUDIO_NOISES, "seed"), ("x264_preset", X264_PRESETS, X264_PRESET),
                                  ("load", LOADS, "fast")):
        job[key] = _choice(data, key, choices, default)
    seconds = data["seconds"]
    if not _is_int(seconds) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise JobError(f"seconds: a whole number from {MIN_SECONDS} to {MAX_SECONDS}, got {seconds!r}")
    seed = data["seed"]
    if not _is_int(seed) or not 0 <= seed < 2 ** 63:
        raise JobError(f"seed: a whole number from 0 to 2**63-1, got {seed!r}")
    steps = data["steps"]
    if not _is_int(steps) or not 1 <= steps <= MAX_STEPS:
        raise JobError(f"steps: a whole number from 1 to {MAX_STEPS}, got {steps!r}")
    shift = data["shift"]
    if not isinstance(shift, list) or len(shift) != 2 or not all(_is_number(s) and 0 < s <= 1000 for s in shift):
        raise JobError(f"shift: [video, audio], two numbers above 0, got {shift!r}")
    crf = data.get("crf", CRF)
    if not _is_int(crf) or not 0 <= crf <= 51:
        raise JobError(f"crf: a whole number from 0 to 51, got {crf!r}")
    abort = data.get("abort_below_gib", ABORT_BELOW_GIB)
    if not _is_number(abort) or not 0 <= abort <= 100:
        raise JobError(f"abort_below_gib: a number of GiB from 0 to 100, got {abort!r}")
    job.update(seconds=seconds, seed=seed, steps=steps, shift=[float(s) for s in shift], crf=crf,
               abort_below_gib=float(abort), lora=_path(data["lora"], "lora"),
               conditioning=_path(data["conditioning"], "conditioning"),
               first_frame=_path(data.get("first_frame"), "first_frame", required=False),
               last_frame=_path(data.get("last_frame"), "last_frame", required=False))
    if job["task"] == "t2va" and (job["first_frame"] or job["last_frame"]):
        raise JobError("t2va takes no first_frame or last_frame (fl2va does)")
    if job["task"] == "fl2va" and not (job["first_frame"] or job["last_frame"]):
        raise JobError("fl2va needs a first_frame, a last_frame or both")
    job["keyframes"] = [anchor for anchor in ("first", "last") if job[f"{anchor}_frame"]]
    paths = data["paths"]
    if not isinstance(paths, dict):
        raise JobError("paths: a JSON object with base_model and output")
    unknown = sorted(set(paths) - set(PATH_KEYS))
    if unknown:
        raise JobError(f"paths: unknown field(s): {', '.join(unknown)}")
    job["base_model"] = _path(paths.get("base_model"), "paths.base_model")
    job["output"] = _path(paths.get("output"), "paths.output", required=False) or Path(job_dir)
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise JobError("meta: a JSON object")
    job["meta"] = meta
    if job["audio_noise"] == "seed":
        job["audio_noise_seed"] = (seed + AUDIO_SEED_OFFSET) % 2 ** 63
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
    """The studio's LFSTUDIO_H3TURBO_* settings from an environment, with their defaults: (settings, problems)."""
    home = Path(environ.get("LFSTUDIO_HOME") or Path.home() / "sol-h3").expanduser()
    checkpoints = Path(environ.get("LFSTUDIO_CHECKPOINTS") or home / "checkpoints").expanduser()
    settings, problems = {}, []
    for key, name, default in (("python", "LFSTUDIO_H3TURBO_PYTHON", home / "envs" / "regen2k" / "bin" / "python"),
                               ("base_model", "LFSTUDIO_H3TURBO_BASE", checkpoints / "MiniMax-H3"),
                               ("loras", "LFSTUDIO_H3TURBO_LORAS", checkpoints / "Minimax-h3-Turbo")):
        settings[key] = Path(environ.get(name) or default).expanduser()
        if not settings[key].is_absolute():
            problems.append(f"{name}={settings[key]}: must be an absolute path")
    settings["qwen_canvas"] = environ.get("LFSTUDIO_H3TURBO_QWEN_CANVAS") or "draft"
    if settings["qwen_canvas"] not in QWEN_CANVASES:
        problems.append(f"LFSTUDIO_H3TURBO_QWEN_CANVAS={settings['qwen_canvas']}: must be draft or canvas")
    for key, name, default in (("min_free_gib", "LFSTUDIO_H3TURBO_MIN_FREE_GIB", MIN_FREE_GIB),
                               ("guard_gib", "LFSTUDIO_H3TURBO_GUARD_GIB", GUARD_GIB)):
        raw = environ.get(name) or str(default)
        try:
            settings[key] = float(raw)
            if not math.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError
        except ValueError:
            problems.append(f"{name}={raw}: must be a number of GiB, 0 or more")
            settings[key] = float(default)
    return settings, problems


# ---------------------------------------------------------------- LoRA files (standard library only)

def safetensors_header(path):
    """(tensors, metadata, header length) from a safetensors file's header; JobError if it is not one."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            size = struct.unpack("<Q", head)[0] if len(head) == 8 else 0
            if not 0 < size < 100_000_000:
                raise JobError(f"{path}: not a safetensors file (header length {size})")
            raw = handle.read(size)
    except OSError as error:
        raise JobError(f"cannot read {path}: {error.strerror or error}") from None
    try:
        header = json.loads(raw)
    except ValueError:
        raise JobError(f"{path}: not a safetensors file (its header is not JSON)") from None
    if not isinstance(header, dict):
        raise JobError(f"{path}: not a safetensors file")
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata, size


def lora_info(path):
    """Checks a diffusers-format turbo LoRA file by its header: PEFT keys `<module>.lora_A.default.weight` /
    `.lora_B.default.weight` on the six target suffixes, A/B paired, one rank, an alpha in __metadata__, and the
    file's size what its header says (8 + header + data). Returns the modules, rank, alpha, scale and dtypes."""
    header, metadata, header_size = safetensors_header(path)
    pairs, ranks, end = {}, set(), 0
    for key, info in header.items():
        name = key[len("transformer."):] if key.startswith("transformer.") else key
        if name.endswith(LORA_A):
            module, side = name[:-len(LORA_A)], "A"
        elif name.endswith(LORA_B):
            module, side = name[:-len(LORA_B)], "B"
        else:
            raise JobError(f"{Path(path).name}: unexpected tensor {key} (not a diffusers-format LoRA; the *_comfyui_* "
                           "files are not for this job)")
        if not module.endswith(LORA_TARGETS):
            raise JobError(f"{Path(path).name}: {key} is not on a target module")
        if not isinstance(info, dict) or len(info.get("shape", ())) != 2:
            raise JobError(f"{Path(path).name}: {key} is not a matrix")
        pairs.setdefault(module, {})[side] = (tuple(info["shape"]), info.get("dtype"))
        end = max(end, int(info.get("data_offsets", (0, 0))[1]))
    if not pairs:
        raise JobError(f"{Path(path).name}: no LoRA tensors")
    for module, sides in pairs.items():
        if set(sides) != {"A", "B"}:
            raise JobError(f"{Path(path).name}: {module} has no lora_{'B' if 'A' in sides else 'A'}")
        if sides["A"][0][0] != sides["B"][0][1]:
            raise JobError(f"{Path(path).name}: {module} has A rank {sides['A'][0][0]} but B rank {sides['B'][0][1]}")
        ranks.add(sides["A"][0][0])
    if len(ranks) != 1:
        raise JobError(f"{Path(path).name}: expected one rank, found {sorted(ranks)}")
    rank = ranks.pop()
    try:
        alpha = float(metadata["alpha"])
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise JobError(f"{Path(path).name}: no alpha in the file's metadata (the 4step_v0.1 file has none; this job "
                       "takes the files with one)") from None
    size = os.path.getsize(path)
    if size != 8 + header_size + end:
        raise JobError(f"{Path(path).name} is {size} bytes, its header says {8 + header_size + end}: a cut or "
                       "damaged file")
    return {"file": Path(path).name, "bytes": size, "modules": sorted(pairs), "tensors": len(header), "rank": rank,
            "alpha": alpha, "scale": alpha / rank, "dtypes": sorted({str(s[1]) for p in pairs.values()
                                                                     for s in p.values()}),
            "shapes": {module: [list(sides["A"][0]), list(sides["B"][0])] for module, sides in pairs.items()},
            "metadata": metadata}


def preset_for(path):
    """(steps, PRESETS entry) of a LoRA file by its name, or (None, None) for another file."""
    name = Path(path).name
    for steps, preset in PRESETS.items():
        if preset["file"] == name:
            return steps, preset
    return None, None


def expected_lora(config, rank=128):
    """The 312 target modules of the H3 transformer and their LoRA shapes ([A], [B]) from its config.json."""
    hidden, inner = config["hidden_size"], config["num_attention_heads"] * config["attention_head_dim"]
    ffn = config["ffn_dim"]
    fans = {"attn.to_q": (hidden, inner), "attn.to_k": (hidden, inner), "attn.to_v": (hidden, inner),
            "attn.to_out.0": (inner, hidden), "ff.net.0.proj": (hidden, 2 * ffn), "ff.net.2": (ffn, hidden)}
    out = {}
    for prefix, count in (("transformer_blocks", config["num_layers"]),
                          ("token_refiner.refiner_blocks", config["num_refiner_layers"])):
        for index in range(count):
            for sub, (fan_in, fan_out) in fans.items():
                out[f"{prefix}.{index}.{sub}"] = [[rank, fan_in], [fan_out, rank]]
    return out


def check_lora(job):
    """The job's LoRA: its header rules (lora_info), and for a file of PRESETS its size and step count. Returns
    (the receipt's LoRA record, notes)."""
    if not job["lora"].is_file():
        raise JobError(f"no LoRA file {job['lora']}")
    info = lora_info(job["lora"])
    steps, preset = preset_for(job["lora"])
    notes = []
    if preset is not None:
        if info["bytes"] != preset["bytes"]:
            raise JobError(f"{info['file']} is {info['bytes']} bytes, not {preset['bytes']}")
        if steps != job["steps"]:
            raise JobError(f"{info['file']} is the {steps}-step LoRA, the job has {job['steps']} steps")
        if job["shift"] != [float(s) for s in preset["shift"]]:
            notes.append(f"shift {job['shift']}, not the {preset['shift']} LightX2V gives for {info['file']}")
    else:
        notes.append(f"{info['file']} is not one of PRESETS (its sha256 is not checked)")
    width, height = CANVAS[(job["resolution"], job["orientation"])]
    if "_768p" in info["file"] and (width, height) != (1344, 768):
        notes.append(f"off the LoRA's documented 1344x768 training canvas: {width}x{height}")
    record = {k: v for k, v in info.items() if k not in ("modules", "shapes")}
    record.update(module_count=len(info["modules"]), preset=steps)
    return record, notes


# ---------------------------------------------------------------- inputs (standard library only)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise JobError(f"cannot read {path}: {error}") from None


def check_base(base):
    """The base model's parts this job loads: their configs and weight files (names only, nothing is read). Returns
    the transformer's config. A t2va-only download has vae/config.json without weights."""
    if not base.is_dir():
        raise JobError(f"paths.base_model: no directory {base}")
    for part in BASE_PARTS:
        if not (base / part).is_dir():
            raise JobError(f"the base model {base} has no {part}/")
    for part in WEIGHT_PARTS:
        if not (base / part / "config.json").is_file():
            raise JobError(f"the base model has no {part}/config.json")
        if not any((base / part).glob("*.safetensors")):
            raise JobError(f"the base model has no weights in {part}/" + (
                " (a t2va-only download has vae/config.json only: run install/20-weights.sh with fl2va in "
                "LFSTUDIO_TASKS)" if part == "vae" else ""))
    for part in ("scheduler", "audio_scheduler"):
        if not (base / part / "scheduler_config.json").is_file():
            raise JobError(f"the base model has no {part}/scheduler_config.json")
    config = read_json(base / "transformer" / "config.json")
    if not isinstance(config, dict) or config.get("_class_name") != "MiniMaxH3Transformer3DModel":
        raise JobError(f"{base / 'transformer' / 'config.json'} is not a MiniMax-H3 transformer's config")
    return config


def check_conditioning_meta(meta, job, payload_sha256):
    """The Qwen encode's receipt (conditioning.json next to conditioning.pt) against the job, without torch: its
    task and keyframes, its payload, its shape, the canvas Qwen saw the keyframes at (prepared_media), the vision
    rows (shape - text_tokens) and the prompt-token rule. Returns (the receipt's conditioning record, notes)."""
    if not isinstance(meta, dict):
        raise JobError("conditioning.json must hold a JSON object")
    if meta.get("status", "PASS") != "PASS":
        raise JobError(f"the conditioning's encode did not pass: {meta.get('status')}")
    if meta.get("task") != job["task"]:
        raise JobError(f"the conditioning is for {meta.get('task')}, the job is {job['task']}")
    if meta.get("payload_sha256") and meta["payload_sha256"] != payload_sha256:
        raise JobError("conditioning.pt does not match its conditioning.json payload_sha256")
    shape = meta.get("shape")
    if not (isinstance(shape, list) and len(shape) == 3 and shape[0] == 1 and shape[2] == TEXT_WIDTH
            and _is_int(shape[1]) and shape[1] > 0):
        raise JobError(f"the conditioning's shape is {shape}, not [1, rows, {TEXT_WIDTH}]")
    prompt = meta.get("prompt")
    if not isinstance(prompt, str):
        raise JobError("conditioning.json has no prompt")
    notes = []
    spec = meta.get("input_spec") or {}
    for anchor in ("first", "last"):
        theirs, mine = spec.get(f"{anchor}_frame"), job[f"{anchor}_frame"]
        if bool(theirs) != bool(mine):
            raise JobError(f"the conditioning was made {'with' if theirs else 'without'} a {anchor} frame, the job "
                           f"has {'one' if mine else 'none'}")
        if theirs and mine and os.path.isfile(theirs) and str(mine) != theirs and sha256(theirs) != sha256(mine):
            notes.append(f"the {anchor} frame differs from the image the conditioning was made with")
    qw, qh = qwen_canvas(job["orientation"], job["resolution"], job["qwen_canvas"])
    tokens = image_tokens(qw, qh)
    for item in meta.get("prepared_media") or []:
        if item.get("type") == "image" and list(item.get("shape") or [])[:2] != [qh, qw]:
            raise JobError(f"Qwen saw the {item.get('anchor')} frame at {item.get('shape')} (height, width), not "
                           f"at the {qw}x{qh} of qwen_canvas {job['qwen_canvas']!r}")
    rows = shape[1]
    record = {"rows": rows, "text_tokens": meta.get("text_tokens"), "task": meta["task"],
              "prompt_tokens": meta.get("prompt_tokens", "plain"), "prompt_chars": len(prompt),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "qwen_canvas": [qw, qh],
              "image_tokens_per_keyframe": tokens if job["task"] == "fl2va" else 0}
    if _is_int(meta.get("text_tokens")):
        vision = rows - meta["text_tokens"]
        want = len(job["keyframes"]) * (tokens + 2)
        if vision != want:
            raise JobError(f"the conditioning has {vision} vision rows; {len(job['keyframes'])} keyframe(s) seen at "
                           f"{qw}x{qh} (qwen_canvas {job['qwen_canvas']!r}) give {want}")
    if record["prompt_tokens"] != "minimax":
        found = [token for token in MINIMAX_PROMPT_TOKENS if token in prompt]
        if found:
            raise JobError(f"the conditioning was encoded in prompt-token mode {record['prompt_tokens']!r} and its "
                           f"prompt has MiniMax's tokens {found}, which that mode split into text tokens H3 never saw "
                           "(SOL_H3_PROMPT_TOKENS=minimax encodes them as H3 was trained)")
    return record, notes


def check_inputs(job, receipt):
    """Every file the job reads, checked without torch; fills the receipt's inputs, conditioning, LoRA, geometry and
    notes. Raises JobError."""
    if job["load"] == "fast" and not FASTLOAD.is_file():
        raise JobError(f"no {FASTLOAD}, the fast load's loader (part of LF Studio)")
    if not job["output"].is_dir():
        raise JobError(f"paths.output: no directory {job['output']}")
    files = {"conditioning": job["conditioning"], "conditioning_json": job["conditioning"].with_suffix(".json")}
    for anchor in job["keyframes"]:
        files[f"{anchor}_frame"] = job[f"{anchor}_frame"]
    for name, path in files.items():
        if not path.is_file():
            raise JobError(f"no {name.replace('_', ' ')} file {path}")
    receipt["inputs"] = {name: str(path) for name, path in files.items()}
    receipt["inputs_sha256"] = {name: sha256(path) for name, path in files.items()}
    meta = read_json(files["conditioning_json"])
    receipt["conditioning"], notes = check_conditioning_meta(meta, job, receipt["inputs_sha256"]["conditioning"])
    receipt["lora"], lora_notes = check_lora(job)
    config = check_base(job["base_model"])
    receipt["transformer_config_sha256"] = sha256(job["base_model"] / "transformer" / "config.json")
    receipt["geometry"] = geometry(job)
    receipt["layout"] = layout_rows(receipt["conditioning"]["rows"], receipt["geometry"]["width"],
                                    receipt["geometry"]["height"], receipt["geometry"]["frames"],
                                    len(job["keyframes"]))
    receipt["notes"] = notes + lora_notes
    return config


def prepare(job_path, receipt):
    """job.json and the inputs, checked without torch -> the job. Fills the receipt's job as soon as it parses."""
    job = load_job(job_path)
    receipt["job"] = job_record(job)
    check_inputs(job, receipt)
    return job


def check_install(settings, env=False):
    """What the studio needs for LightX2V Turbo, without the GPU and without reading weights: the base model's parts and
    configs, the LoRA files of PRESETS (header, size, modules and shapes against the transformer's config; the files
    of OFFERED_STEPS are required) and the upscale/fastload.py loader; with `env` also this Python's packages
    (diffusers DIFFUSERS_VERSION and the blocks run() calls; imports torch). Returns (report, problems)."""
    report, problems = {"base_model": str(settings["base_model"]), "loras": {}}, []
    config = None
    try:
        config = check_base(settings["base_model"])
    except JobError as error:
        problems.append(str(error))
    if not FASTLOAD.is_file():
        problems.append(f"no {FASTLOAD} (the fast load's loader)")
    for steps, preset in sorted(PRESETS.items()):
        path = settings["loras"] / preset["file"]
        entry = report["loras"][steps] = {"file": str(path), "offered": steps in OFFERED_STEPS}
        if not path.is_file():
            entry["state"] = "missing"
            if steps in OFFERED_STEPS:
                problems.append(f"no {path} (the {steps}-step LoRA; install/66-h3turbo.sh fetches it)")
            continue
        try:
            info = lora_info(path)
            if info["bytes"] != preset["bytes"]:
                raise JobError(f"{path} is {info['bytes']} bytes, not {preset['bytes']}")
            if config is not None and info["shapes"] != expected_lora(config, info["rank"]):
                raise JobError(f"{path}: its modules or shapes do not fit the base model's transformer")
            entry.update(state="ok", rank=info["rank"], alpha=info["alpha"], modules=len(info["modules"]))
        except JobError as error:
            entry["state"] = "bad"
            problems.append(str(error))
    if env:
        report["env"] = check_env(problems)
    return report, problems


def check_env(problems):
    """This Python's packages for run(): diffusers DIFFUSERS_VERSION, its MiniMax-H3 classes and the blocks run()
    pops. Appends to problems; returns the versions found."""
    from importlib import metadata
    found = {}
    for name in ("torch", "diffusers", "safetensors", "numpy", "pillow", "imageio-ffmpeg", "peft"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    if found["diffusers"] != DIFFUSERS_VERSION:
        problems.append(f"diffusers {found['diffusers']} in {sys.executable}, this job is written for "
                        f"{DIFFUSERS_VERSION}")
        return found
    for name in ("torch", "safetensors", "numpy", "pillow"):
        if found[name] is None:
            problems.append(f"no {name} in {sys.executable}")
    try:
        import diffusers
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
        from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3DecodeStep  # noqa: F401
        for name in DIRECT_LOAD + ("MiniMaxH3Scheduler",):
            getattr(diffusers, name)
        blocks = MiniMaxH3Blocks()
        absent = [name for name in POPPED_BLOCKS if name not in blocks.sub_blocks]
        if absent:
            problems.append(f"diffusers' MiniMaxH3Blocks has no {absent} block(s)")
    except Exception as error:  # noqa: BLE001 - reported
        problems.append(f"diffusers' MiniMax-H3 pipeline: {type(error).__name__}: {error}"[:500])
    return found


# ---------------------------------------------------------------- processes and lines (standard library only)

def worker_command(python, script, job_path, parent=None):
    """How the server runs a job: -I (no user site, no PYTHONPATH, no script dir) and -B (no .pyc files); with
    `parent` (the server's pid) the job ends when the server does (--parent)."""
    return [str(python), "-I", "-B", str(script), "--job", str(job_path)] + \
        (["--parent", str(int(parent))] if parent is not None else [])


# the Hub must never be asked (a missed local path would download with the user's token); every load passes
# local_files_only=True, this is the second fence. The same as upscale/regen2k_job.py's (a test checks it).
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_TOKEN_PATH": "/nonexistent", "PYTHONNOUSERSITE": "1"}
REMOVED = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "H3_ATTN_BACKEND", "PYTHONPATH")
TUNING = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",     # fragmentation over a ~80 GiB peak
          "MIMALLOC_PURGE_DELAY": "0"}       # torch's aarch64 mimalloc: give freed CPU memory back (it is GPU memory)


def worker_environment(environ):
    """The environment of a job: the offline settings (forced), no token, no H3_ATTN_BACKEND, the allocator settings
    unless set, and the distro's ffmpeg for imageio-ffmpeg when there is one."""
    env = {k: v for k, v in environ.items() if k not in REMOVED}
    env.update(OFFLINE)
    for key, value in TUNING.items():
        env.setdefault(key, value)
    if "IMAGEIO_FFMPEG_EXE" not in env and os.path.isfile("/usr/bin/ffmpeg"):
        env["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    return env


def apply_environment(environ):
    """worker_environment on this process's own environment, before torch or the Hub libraries are imported."""
    wanted = worker_environment(environ)
    for key in [key for key in environ if key not in wanted]:
        del environ[key]
    environ.update(wanted)


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

def meminfo(path=None):
    """MemAvailable, MemFree and Cached in GiB."""
    values = {}
    with open(path or MEMINFO) as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "MemFree", "Cached"):
                values[key] = int(rest.split()[0]) / 1048576
    return values


class MemTrace(threading.Thread):
    """The lowest MemAvailable per label, sampled every 0.5 s (and at every mark); sets `abort` under the floor."""

    def __init__(self, floor_gib, path=None, interval=0.5):
        super().__init__(daemon=True, name="memtrace")
        self.path = path or MEMINFO
        self.floor, self.interval, self.label, self.low = floor_gib, interval, "start", {}
        self.halt, self.abort, self.reason = threading.Event(), threading.Event(), None
        self.lock = threading.Lock()

    def sample(self):
        label = self.label                 # once, before the read: a value read before mark() stays under its label
        available = meminfo(self.path).get("MemAvailable")
        if available is not None:
            with self.lock:
                self.low[label] = round(min(self.low.get(label, available), available), 2)
                if available < self.floor and not self.abort.is_set():
                    self.reason = f"MemAvailable {available:.2f} GiB < {self.floor:g} GiB during {label}"
                    self.abort.set()
        return available

    def mark(self, label):
        self.label = label
        return self.sample()

    def check(self):
        if self.abort.is_set():
            raise RunAborted(self.reason)

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


class Recorder:
    """Times, peak allocated GPU memory and lowest MemAvailable per phase of the clip (phases nest: a parent's peak
    covers its children's, though each child resets the allocator's peak counter). A phase starts with the memory
    check, so an abort lands at the next phase boundary at the latest."""

    def __init__(self, torch, device, trace, receipt):
        self.torch, self.cuda, self.trace = torch, device.type == "cuda", trace
        self.times = receipt.setdefault("times_s", {})
        self.peaks = receipt.setdefault("peak_gib", {})
        self.lows = receipt.setdefault("lowest_mem_available_gib", {})
        self.stack, self.step = [], 0
        self.receipt = receipt

    def _peak(self):
        return self.torch.cuda.max_memory_allocated() / 2 ** 30 if self.cuda else 0.0

    @contextlib.contextmanager
    def phase(self, label):
        self.trace.check()
        if self.cuda:
            self.torch.cuda.synchronize()
            if self.stack:
                self.stack[-1]["peak"] = max(self.stack[-1]["peak"], self._peak())
            self.torch.cuda.reset_peak_memory_stats()
        frame = {"label": label, "peak": 0.0, "low": None}
        self.stack.append(frame)
        self.receipt["stage"] = label
        self.trace.mark(label)
        start = time.perf_counter()
        cancelled = False
        try:
            yield frame
        except Cancelled:
            cancelled = True                  # SIGTERM: no wait for queued GPU work
            raise
        finally:
            try:
                if self.cuda and not cancelled:
                    self.torch.cuda.synchronize()
            finally:
                self.times[label] = round(time.perf_counter() - start, 3)
                peak = max(frame["peak"], self._peak())
                self.stack.pop()
                lows = [x for x in (self.trace.low.get(label), frame["low"]) if x is not None]
                low = min(lows) if lows else None
                if self.cuda:
                    self.peaks[label] = round(peak, 3)
                if low is not None:
                    self.lows[label] = low
                if self.stack:
                    parent = self.stack[-1]
                    parent["peak"] = max(parent["peak"], peak)
                    if low is not None:
                        parent["low"] = low if parent["low"] is None else min(parent["low"], low)
                    self.trace.label = parent["label"]


# ---------------------------------------------------------------- with torch, numpy, PIL (imported inside)

def versions(torch):
    from importlib import metadata
    found = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
             "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    for name in ("diffusers", "transformers", "safetensors", "peft", "accelerate", "numpy", "pillow", "av",
                 "imageio-ffmpeg"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    if torch.cuda.is_available():
        found["gpu"] = torch.cuda.get_device_name()
        found["capability"] = list(torch.cuda.get_device_capability())
    found["job_sha256"] = sha256(__file__)
    found["fastload_sha256"] = sha256(FASTLOAD) if FASTLOAD.is_file() else None
    return found


def load_fastload(path=FASTLOAD):
    """upscale/fastload.py as a module of its own, by its path (the job runs with -I: no script dir on sys.path)."""
    spec = importlib.util.spec_from_file_location("lf_studio_fastload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_fast_load(fastload, safetensors_torch, diffusers):
    """The fast load (the docstring): safetensors.torch.load_file (diffusers calls it by attribute) loads every
    tensor straight to the GPU through fastload.load_file, whatever device the caller asks for, and from_pretrained
    of the DIRECT_LOAD classes gets device_map={"": "cuda"} unless the call names one. Call it before anything loads.
    Returns the receipt's "load", whose counts grow as files load."""
    stock = safetensors_torch.load_file
    record = {"mode": "fast", "o_direct": bool(fastload.ENABLED and fastload.DIRECT), "files": 0, "fallbacks": 0}
    inside, counting = threading.local(), threading.Lock()

    def count(key):
        with counting:
            record[key] += 1

    def load_file(filename, device="cpu"):
        if getattr(inside, "active", False):
            count("fallbacks")                  # fastload's own fallback: the stock loader, to the GPU too
            return stock(filename, device="cuda")
        inside.active = True
        try:
            count("files")
            return fastload.load_file(filename, device="cuda")
        finally:
            inside.active = False
    safetensors_torch.load_file = load_file
    for name in DIRECT_LOAD:
        cls = getattr(diffusers, name)
        original = cls.from_pretrained.__func__

        def direct(klass, *args, _original=original, **kwargs):
            kwargs.setdefault("device_map", {"": "cuda"})
            return _original(klass, *args, **kwargs)
        cls.from_pretrained = classmethod(direct)
    return record


def drop_from_page_cache(paths):
    """POSIX_FADV_DONTNEED on each file: on GB10 the page cache takes the GPU's memory too."""
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


def load_lora_pairs(path, device, torch, fastload=None):
    """The LoRA file's tensors on `device`, as {module: (A [r, in], B [out, r])} (bf16, the file's dtype)."""
    if fastload is not None:
        tensors = fastload.load_file(str(path), device=str(device))
    else:
        from safetensors.torch import load_file
        tensors = load_file(str(path), device=str(device))
    pairs = {}
    for key, tensor in tensors.items():
        name = key[len("transformer."):] if key.startswith("transformer.") else key
        if name.endswith(LORA_A):
            pairs.setdefault(name[:-len(LORA_A)], [None, None])[0] = tensor
        elif name.endswith(LORA_B):
            pairs.setdefault(name[:-len(LORA_B)], [None, None])[1] = tensor
    return {name: (a, b) for name, (a, b) in pairs.items()}


def lora_targets(model, torch):
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Linear) and name.endswith(LORA_TARGETS))


class LoraAdapter:
    """A turbo LoRA as an unfused branch on each target Linear: y = W x + B (A x) * scale.

    PEFT's LoRA Linear forward (result + lora_B(lora_A(dropout(x))) * scaling, dropout the identity, x cast to the
    adapter's dtype) with the adapter in the base layer's dtype (bf16), which is what the reference's
    transformer.add_adapter gives. The base weights are never written, so removing the hooks restores the model
    exactly. The file's modules must be exactly the model's target modules, as the reference requires."""

    def __init__(self, model, pairs, scale, torch):
        import torch.nn.functional as F
        self.handles, self.modules = [], sorted(pairs)
        targets = lora_targets(model, torch)
        if self.modules != targets:
            missing, extra = sorted(set(targets) - set(pairs)), sorted(set(pairs) - set(targets))
            raise RuntimeError(f"LoRA modules do not match the transformer's targets: missing {missing[:3]} "
                               f"({len(missing)}), unexpected {extra[:3]} ({len(extra)})")
        for name in self.modules:                 # every pair checked before any hook goes on
            module = model.get_submodule(name)
            a, b = pairs[name]
            if a is None or b is None or tuple(a.shape) != (a.shape[0], module.in_features) or \
                    tuple(b.shape) != (module.out_features, a.shape[0]):
                raise RuntimeError(f"LoRA {name}: A {None if a is None else tuple(a.shape)}, B "
                                   f"{None if b is None else tuple(b.shape)} do not fit {module}")
        for name in self.modules:
            module = model.get_submodule(name)
            a, b = pairs[name]

            def hook(mod, args, output, _a=a, _b=b):
                # PEFT's `result + lora_B(lora_A(x)) * scaling`, with the scale and the add in place: the same bf16
                # mul and add (the same roundings), one output-sized tensor fewer alive (2.1 GiB for ff.net.0.proj
                # at 768p). `output` is the Linear's own fresh result, nothing else holds it.
                x = args[0]
                if x.dtype != _a.dtype:
                    x = x.to(_a.dtype)
                delta = F.linear(F.linear(x, _a), _b)
                delta.mul_(scale)
                return output.add_(delta)
            self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_rgb(path):
    """As the studio's Qwen worker loads it (Sol-H3-Spark qwen_ops/media.py): EXIF transpose, then RGB. Returns the
    image and whether the EXIF orientation turned it."""
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        orientation = image.getexif().get(0x0112, 1)
        return ImageOps.exif_transpose(image).convert("RGB"), orientation not in (None, 1)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - the system ffmpeg is the fallback
        return "ffmpeg"


def write_wav(path, audio):
    """(2, samples) float in [-1, 1] -> 16-bit stereo WAV at 32 kHz (diffusers' encode_video conversion: clip,
    * 32767, to int16)."""
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(AUDIO_RATE)
        out.writeframes(np.ascontiguousarray(pcm.T).tobytes())


def write_mp4(frames, audio, path, crf, preset):
    """frames (F, H, W, 3) uint8, audio (2, samples) float -> H.264 yuv420p at CRF crf and x264 preset `preset` with
    the audio muxed in as AAC stereo, 24 fps. The WAV goes to a temporary dir next to the output, not $TMPDIR.
    Returns the ffmpeg used and its video settings."""
    path = Path(path)
    partial = path.with_name(f".{path.name}.partial.mp4")
    with tempfile.TemporaryDirectory(prefix=".h3turbo-", dir=str(path.parent)) as tmp:
        wav = Path(tmp) / "audio.wav"
        write_wav(wav, audio)
        height, width = frames.shape[1:3]
        command = [ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0", "-i", str(wav),
                   "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(partial)]
        subprocess.run(command, input=frames.tobytes(), check=True)
    os.replace(partial, path)
    return {"ffmpeg": command[0], "video": f"libx264 -preset {preset} -crf {crf} -pix_fmt yuv420p"}


def adherence(frame, keyframe):
    """A decoded frame against the keyframe the VAE encoded (same canvas): PSNR (dB) and mean absolute error."""
    import numpy as np
    a, b = np.asarray(frame).astype(np.float64), np.asarray(keyframe).astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    return {"psnr_db": round(10 * math.log10(255 ** 2 / mse), 3) if mse > 0 else None,
            "mae": round(float(np.abs(a - b).mean()), 3)}


def tensor_sha256(tensor):
    """sha256 of a CPU tensor's bytes, in order (what two latents files are compared by)."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _on(model, device):
    """The fast load puts the weights on the GPU already (device_map): no second move."""
    if next(model.parameters()).device.type != device.type:
        model = model.to(device)
    return model


def _quiet(block):
    """No tqdm bars from the denoise loop (the log is a file)."""
    if hasattr(block, "set_progress_bar_config"):
        block.set_progress_bar_config(disable=True)
    for child in getattr(block, "sub_blocks", {}).values():
        _quiet(child)


def allow_length(pipe, frames):
    """A clip longer than diffusers' stock max_duration (the studio's 15 s: 362 frames, 15.083 s; the docstring's
    Lengths): this pipeline instance gets a class of its own, a subclass of its class whose max_duration is the clip's
    length, so diffusers' own setup step takes the frames, and every other check of it stays. diffusers' files, its
    class and other instances are unchanged. Returns the receipt's record, or None for a clip within stock (the
    pipeline untouched)."""
    duration = max_duration_for(frames)
    stock = pipe.max_duration
    if duration is None or duration <= stock:
        return None
    base = type(pipe)
    pipe.__class__ = type(f"LFStudio{base.__name__}", (base,), {
        "__module__": __name__, "max_duration": property(lambda self: duration,
                                                         doc=f"the clip's {frames} frames (LF Studio's job)")})
    return {"stock_s": stock, "clip_s": round(duration, 4), "frames": frames}


def pad_audio(audio, samples):
    """(2, n) decoded audio cut or padded to `samples`: where the decode is shorter (a clip that keeps every frame made,
    at 6 and 15 s: 266 samples), its last samples mirrored onto the end, which keeps it continuous. Returns (audio, the
    samples added)."""
    import numpy as np
    short = samples - audio.shape[1]
    if short <= 0:
        return audio[:, :samples], 0
    return np.pad(audio, ((0, 0), (0, short)), mode="reflect"), short


def empty_cache(torch, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def check_conditioning(cond, job, torch):
    """The conditioning tensors (as Sol-H3-Spark's stage1 checks them): prompt_embeds [1, L, 5120] bf16 and finite,
    text_token_tags [L] in {0, 1}; t2va every tag 1, fl2va one vision block (a run of tag 0) per keyframe, each of
    the image tokens qwen_canvas gives (+2 for its start and end). Returns what the receipt records."""
    if not isinstance(cond, dict) or "prompt_embeds" not in cond or "text_token_tags" not in cond:
        raise JobError("the conditioning has no prompt_embeds and text_token_tags")
    if cond.get("task") not in (None, job["task"]):
        raise JobError(f"the conditioning is for {cond.get('task')}, the job is {job['task']}")
    embeds, tags = cond["prompt_embeds"], cond["text_token_tags"]
    if tags.ndim == 2 and tags.shape[0] == 1:
        tags = tags[0]
    if (embeds.dtype != torch.bfloat16 or embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[2] != TEXT_WIDTH
            or tags.ndim != 1 or tags.shape[0] != embeds.shape[1]):
        raise JobError(f"conditioning shapes: prompt_embeds {tuple(embeds.shape)} {embeds.dtype}, tags "
                       f"{tuple(tags.shape)}; want [1, L, {TEXT_WIDTH}] bfloat16 and [L]")
    values = tags.tolist()
    if not set(values) <= {0, 1}:
        raise JobError(f"text_token_tags hold {sorted(set(values))}, want 0 and 1")
    if not bool(torch.isfinite(embeds).all()):
        raise JobError("prompt_embeds has values that are not finite")
    blocks, previous = [], 1
    for index, value in enumerate(values):
        if value == 0 and previous == 1:
            blocks.append([index, 0])
        if value == 0:
            blocks[-1][1] += 1
        previous = value
    if len(blocks) != len(job["keyframes"]):
        raise JobError(f"the conditioning has {len(blocks)} vision blocks for {len(job['keyframes'])} keyframe(s)")
    want = image_tokens(*qwen_canvas(job["orientation"], job["resolution"], job["qwen_canvas"]))
    for start, rows in blocks:
        if rows - 2 != want:
            raise JobError(f"a vision block of the conditioning has {rows - 2} image tokens, qwen_canvas "
                           f"{job['qwen_canvas']!r} gives {want}")
    return {"rows": len(values), "text_rows": values.count(1), "vision_rows": values.count(0),
            "vision_blocks": [{"start": start, "rows": rows, "image_tokens": rows - 2} for start, rows in blocks]}


def generate_kwargs(job, images, prompt_embeds, tags, torch):
    """The arguments of the generation call, as the probe made them: the canvas, the frames made, the conditioning,
    steps + 1 grid points and one CPU generator seeded `seed`; t2va no image, fl2va `image` (the first frame) and/or
    `last_image`; with audio_noise "seed" the audio noise drawn from a generator of its own."""
    g = geometry(job)
    kwargs = {"height": g["height"], "width": g["width"], "num_frames": g["frames"], "prompt_embeds": prompt_embeds,
              "text_token_tags": tags, "num_inference_steps": g["num_inference_steps"],
              "generator": torch.Generator().manual_seed(job["seed"])}
    named = dict(zip(job["keyframes"], images))
    if "first" in named:
        kwargs["image"] = named["first"]
    if "last" in named:
        kwargs["last_image"] = named["last"]
    if job["audio_noise"] == "seed":
        # the same audio noise at every resolution for one seed, from a seed of its own: a generator seeded `seed`
        # would repeat the first values of the keyframe-conditioning noise (CPU randn's first values do not depend
        # on the tensor's size)
        rows = torch.randn((2 * g["audio_latents"], 32), generator=torch.Generator().manual_seed(
            job["audio_noise_seed"]), dtype=torch.float32)
        kwargs["audio_latents"] = rows.reshape(2, g["audio_latents"], 32).permute(0, 2, 1)
    return kwargs


def load_models(job, device, recorder, receipt, phase):
    """The base model on `device` and the two pipelines that share it: `gen` (resize, keyframe VAE encode, layout,
    noise, sigmas, the denoise loop, unpack; no text encoder, no decode) and `dec` (video and audio decode). Every
    transformer forward is timed as a step and prints its phase line; the transformer's blocks and the VAEs' encoder
    and decoders check the memory floor first."""
    import torch
    start = time.perf_counter()
    import diffusers
    from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler,
                           MiniMaxH3Transformer3DModel)
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3DecodeStep
    receipt["times_s"]["import_diffusers"] = round(time.perf_counter() - start, 3)
    diffusers.utils.logging.disable_progress_bar()
    base, fastload = job["base_model"], None
    if device.type == "cuda" and job["load"] == "fast":
        import safetensors.torch
        fastload = load_fastload()
        receipt["load"] = install_fast_load(fastload, safetensors.torch, diffusers)
    else:
        receipt["load"] = {"mode": "standard" if device.type == "cuda" else "standard (cpu)"}
    steps = job["steps"]

    def check_abort(*_):
        recorder.trace.check()

    calls = {}

    def timed(label, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if label == "step":
                recorder.step += 1
                name = f"step_{recorder.step}"
                phase(f"step {recorder.step}/{steps}")
            elif label == "encode":               # once per keyframe: encode_1, encode_2
                calls[label] = calls.get(label, 0) + 1
                name = f"{label}_{calls[label]}"
            else:
                name = label
            with recorder.phase(name):
                return function(*args, **kwargs)
        return wrapper

    with recorder.phase("load"):
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(base, subfolder="transformer",
                                                                  torch_dtype=torch.bfloat16, local_files_only=True)
        transformer = _on(transformer, device).eval()
        transformer.requires_grad_(False)
        with recorder.phase("load_vae"):
            kwargs = {"torch_dtype": torch.float32, "local_files_only": True}
            vae = AutoencoderKLMiniMaxH3.from_pretrained(base, subfolder="vae", **kwargs)
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(base, subfolder="audio_vae", **kwargs)
            vae, audio_vae = _on(vae, device).eval(), _on(audio_vae, device).eval()
            vae.requires_grad_(False)
            audio_vae.requires_grad_(False)
        scheduler = MiniMaxH3Scheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True)
        audio_scheduler = MiniMaxH3Scheduler.from_pretrained(base, subfolder="audio_scheduler", local_files_only=True)
    if device.type == "cuda":
        receipt["page_cache_dropped_files"] = drop_from_page_cache(
            p for part in WEIGHT_PARTS for p in glob.glob(str(base / part / "*.safetensors")))
    if fastload is not None:
        say(f"fast load: {receipt['load']['files']} files, {receipt['load']['fallbacks']} through the stock loader")
    # timing and abort hooks (functools.wraps keeps the signature the denoiser reads the forward's arguments by)
    transformer.forward = timed("step", transformer.forward)
    for block in list(transformer.transformer_blocks) + list(transformer.token_refiner.refiner_blocks):
        block.register_forward_pre_hook(check_abort)
    vae.encode = timed("encode", vae.encode)
    vae.decode = timed("decode_video", vae.decode)
    audio_vae.decode = timed("decode_audio", audio_vae.decode)
    for module in (vae.encoder, vae.decoder, audio_vae.decoder):
        module.register_forward_pre_hook(check_abort)
    blocks = MiniMaxH3Blocks()
    for name in POPPED_BLOCKS:                    # the studio's conditioning.pt instead; the decode is `dec`
        blocks.sub_blocks.pop(name)
    gen = blocks.init_pipeline()                  # no path: nothing is read from modular_model_index.json
    raised = allow_length(gen, geometry(job)["frames"])  # 15 s only: diffusers' max_duration for this instance
    if raised:
        receipt["max_duration"] = raised
        receipt.setdefault("notes", []).append(
            f"{raised['frames']} frames ({raised['clip_s']:g} s): diffusers {DIFFUSERS_VERSION}'s max_duration "
            f"({raised['stock_s']:g} s) raised to the clip's length for this pipeline instance")
    gen.update_components(transformer=transformer, vae=vae, audio_vae=audio_vae, scheduler=scheduler,
                          audio_scheduler=audio_scheduler)
    dec = MiniMaxH3DecodeStep().init_pipeline()
    dec.update_components(vae=vae, audio_vae=audio_vae)
    for pipe in (gen, dec):
        _quiet(pipe._blocks)
    if gen._execution_device.type != device.type:
        raise RuntimeError(f"the pipeline runs on {gen._execution_device}, not {device}")
    receipt["device"] = str(gen._execution_device)
    if device.type == "cuda":
        transformer.set_attention_backend(ATTENTION_BACKEND)
        receipt["allocated_after_load_gib"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    receipt["attention_backend"] = ATTENTION_BACKEND if device.type == "cuda" else "native"
    receipt["mem_available_after_load_gib"] = round(meminfo(recorder.trace.path).get("MemAvailable", 0), 2)
    return {"gen": gen, "dec": dec, "transformer": transformer, "fastload": fastload}


def run(job, receipt, trace):
    """The clip on the GPU (DEVICE). Fills the receipt as it goes (the docstring)."""
    start = time.perf_counter()
    import numpy as np
    import torch
    receipt["times_s"]["import_torch"] = round(time.perf_counter() - start, 3)
    device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    else:
        torch.backends.mkldnn.enabled = False     # aarch64 oneDNN fails on the audio VAE's long depthwise convs
    receipt["versions"] = versions(torch)
    recorder = Recorder(torch, device, trace, receipt)
    g = geometry(job)
    out = job["output"]

    def phase(name):
        receipt["phase"] = name
        print(phase_line(name), flush=True)

    # the inputs on the CPU, checked before anything loads
    with recorder.phase("prepare"):
        cond = torch.load(job["conditioning"], map_location="cpu", weights_only=True)
        receipt["conditioning"].update(check_conditioning(cond, job, torch))
        prompt_embeds, tags = cond["prompt_embeds"], cond["text_token_tags"].flatten().long()
        del cond
        images = []
        receipt["images"] = {}
        canvas = (g["width"], g["height"])
        for anchor in job["keyframes"]:
            image, turned = load_rgb(job[f"{anchor}_frame"])
            fit = keyframe_fit(image.size, canvas, first=not images)
            images.append(image)
            receipt["images"][anchor] = {"size": list(image.size), "exif_transposed": turned, **fit}
            note = fit_note(anchor, image.size, canvas, fit)
            if note:
                receipt.setdefault("notes", []).append(note)
                say(note)

    phase("loading")
    models = load_models(job, device, recorder, receipt, phase)
    transformer, gen = models["transformer"], models["gen"]
    with recorder.phase("lora_attach"):
        pairs = load_lora_pairs(job["lora"], device, torch, models["fastload"])
        adapter = LoraAdapter(transformer, pairs, receipt["lora"]["scale"], torch)
        del pairs
    receipt["lora"].update(applied="unfused bf16 branch (forward hooks)", modules_hooked=len(adapter.modules))
    receipt["load_s"] = receipt["times_s"].get("load")
    gen.scheduler.set_shift(job["shift"][0])
    gen.audio_scheduler.set_shift(job["shift"][1])
    kwargs = generate_kwargs(job, images, prompt_embeds.to(device, torch.bfloat16), tags, torch)
    del prompt_embeds
    receipt["generator"] = "torch.Generator() (CPU), seeded seed, as LightX2V's reference"
    if "audio_noise_seed" in job:
        receipt["audio_noise_seed"] = job["audio_noise_seed"]
    recorder.step = 0
    with recorder.phase("generate"), torch.inference_mode():
        state = gen(**kwargs)
    del kwargs
    latents, audio_lat = state.get("latents"), state.get("audio_latents")
    keyframes = state.get("keyframes") or []
    for anchor, keyframe in zip(job["keyframes"], keyframes):     # the pixels diffusers encoded, as it fitted them
        receipt["images"][anchor]["prepared_sha256"] = hashlib.sha256(np.asarray(keyframe).tobytes()).hexdigest()
    receipt["layout_actual"] = {
        "sequence": int(state.get("position_ids").shape[0]), "text": int(tags.numel()),
        "condition": int(state.get("num_condition_video_rows") or 0),
        "video_rows": int(state.get("video_indices").numel()), "audio_rows": int(state.get("audio_indices").numel()),
        "frames": int(state.get("num_frames")), "latent_frames": int(state.get("num_latent_frames")),
        "canvas": [int(state.get("width")), int(state.get("height"))]}
    if job["keyframes"]:
        receipt["layout_actual"]["condition_t"] = float(state.get("position_ids")[tags.numel(), 0])
    del state
    sigmas = [float(x) for x in gen.scheduler.sigmas.tolist()]
    audio_sigmas = [float(x) for x in gen.audio_scheduler.sigmas.tolist()]
    receipt.update(sigmas=sigmas, audio_sigmas=audio_sigmas, forwards=recorder.step)
    receipt["sigmas_match_expected"] = (
        len(sigmas) == len(g["sigmas"]) and len(audio_sigmas) == len(g["audio_sigmas"])
        and all(abs(a - b) < 2e-6 for a, b in zip(sigmas, g["sigmas"]))
        and all(abs(a - b) < 2e-6 for a, b in zip(audio_sigmas, g["audio_sigmas"])))
    if not receipt["sigmas_match_expected"] or recorder.step != job["steps"]:
        raise RuntimeError(f"the clip used sigmas {sigmas} / {audio_sigmas} and {recorder.step} forwards, expected "
                           f"{g['sigmas']} / {g['audio_sigmas']} and {job['steps']}")
    with recorder.phase("save_latents"):
        from safetensors.torch import save_file
        saved = {"video": latents.detach().float().cpu().contiguous(),
                 "audio": audio_lat.detach().float().cpu().contiguous()}
        partial = out / f".{LATENTS_FILE}.partial"
        save_file(saved, str(partial), metadata={"seed": str(job["seed"]), "task": job["task"],
                                                 "canvas": f"{g['width']}x{g['height']}", "frames": str(g["frames"])})
        os.replace(partial, out / LATENTS_FILE)
        receipt["latent_shapes"] = {name: list(t.shape) for name, t in saved.items()}
        receipt["latents_sha256"] = {name: tensor_sha256(t) for name, t in saved.items()}
        del saved
    # the DiT's last use was the denoise: dropped before the decode, as the Upscale's job does
    dec = models["dec"]
    with recorder.phase("drop_transformer"):
        adapter.remove()
        gen.update_components(transformer=None)
        del adapter, transformer, models
        empty_cache(torch, device)
        receipt["mem_available_before_decode_gib"] = round(meminfo(trace.path).get("MemAvailable", 0), 2)
    phase("decoding")
    with recorder.phase("decode"), torch.inference_mode():
        result = dec(latents=latents, audio_latents=audio_lat, output_type="pt",
                     output=["videos", "audio", "sampling_rate"])
        videos = result["videos"][0]
        receipt["decoded"] = {"frames": int(videos.shape[0]), "samples": int(result["audio"].shape[-1])}
        frames = (videos[:g["keep_frames"]] * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu() \
            .numpy()
        last = None
        if "last" in job["keyframes"]:          # the last frame made, where the last keyframe is anchored (kept)
            last = (videos[-1:] * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]
        audio, padded = pad_audio(result["audio"][0].float().cpu().numpy(), g["keep_samples"])
        rate = int(result["sampling_rate"])
        del result, videos, latents, audio_lat
    empty_cache(torch, device)
    if rate != AUDIO_RATE:
        raise RuntimeError(f"the audio is at {rate} Hz, not {AUDIO_RATE}")
    if frames.shape[0] != g["keep_frames"] or audio.shape != (2, g["keep_samples"]):
        raise RuntimeError(f"decoded {receipt['decoded']}: fewer than the {g['keep_frames']} frames and "
                           f"{g['keep_samples']} samples kept")
    receipt["decoded"].update(kept_frames=int(frames.shape[0]), kept_samples=int(audio.shape[1]),
                              size=[int(frames.shape[2]), int(frames.shape[1])])
    if padded:
        receipt["decoded"]["padded_samples"] = padded
    named = dict(zip(job["keyframes"], keyframes))
    if "first" in named:
        receipt["f0_vs_keyframe"] = adherence(frames[0], named["first"])
    if "last" in named:
        receipt["last_vs_keyframe"] = {"frame": receipt["decoded"]["frames"] - 1, **adherence(last, named["last"])}
        receipt["last_kept_vs_keyframe"] = {"frame": int(frames.shape[0]) - 1,
                                            **adherence(frames[-1], named["last"])}
    phase("saving")
    with recorder.phase("save_audio"):
        partial = out / f".{AUDIO_FILE}.partial"
        torch.save({"audio": torch.from_numpy(np.array(audio, dtype=np.float32))[None]}, str(partial))
        os.replace(partial, out / AUDIO_FILE)
    with recorder.phase("mux"):
        name = output_name(job)
        receipt["ffmpeg"] = write_mp4(frames, audio, out / name, job["crf"], job["x264_preset"])
    receipt["output"] = {"file": name, "size": [g["width"], g["height"]], "frames": int(frames.shape[0]),
                         "samples": int(audio.shape[1])}
    receipt["outputs_bytes"] = {path: (out / path).stat().st_size for path in (name, AUDIO_FILE, LATENTS_FILE)}


# ---------------------------------------------------------------- main

def check(job_path):
    """--check: validate the job and its inputs (and the LoRA's sha256 for a file of PRESETS), print the plan, write
    nothing. No torch."""
    receipt = {}
    try:
        job = prepare(job_path, receipt)
        steps, preset = preset_for(job["lora"])
        if preset is not None:
            receipt["lora"]["sha256"] = sha256(job["lora"])
            drop_from_page_cache([job["lora"]])          # 1.38 GB just read: on GB10 the page cache is GPU memory
            if receipt["lora"]["sha256"] != preset["sha256"]:
                raise JobError(f"{job['lora'].name}: sha256 {receipt['lora']['sha256']}, not {preset['sha256']} "
                               f"({LORA_REPO} at {LORA_REVISION[:8]})")
    except JobError as error:
        print(error_line(error), flush=True)
        return EXIT_REFUSED
    planned = {"ok": True, "output": output_name(job)}
    planned.update({key: receipt[key] for key in ("job", "inputs", "geometry", "layout", "conditioning", "lora",
                                                  "notes")})
    print(json.dumps(planned, indent=1), flush=True)
    return 0


def check_install_main():
    """--check-install: check_install with this Python's packages, on the settings of this environment."""
    settings, problems = studio_settings(os.environ)
    report, found = check_install(settings, env=True)
    problems += found
    print(json.dumps({"ok": not problems, "problems": problems, **report}, indent=1, default=str), flush=True)
    if problems:
        print(error_line("; ".join(problems)), flush=True)
    return EXIT_REFUSED if problems else 0


def cancel(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)     # a second SIGTERM ends the process at once
    raise Cancelled("stopped by SIGTERM")


PR_SET_PDEATHSIG = 1


def tie_to_parent(parent):
    """SIGKILL this process when `parent` (the studio's server, which started it) ends: then nothing guards the
    job's memory any more. Python's SIGTERM handler waits for the GPU call under way, so SIGKILL. Returns False when
    `parent` is gone already (this process was handed to another parent before the prctl), True otherwise, also where
    prctl is not available."""
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
    parser.add_argument("--job", help="job.json written by the studio's server")
    parser.add_argument("--check", action="store_true",
                        help="validate the job and its inputs, print the plan and exit; no torch, no files written")
    parser.add_argument("--check-install", action="store_true",
                        help="check this environment, the base model and the LoRA files of PRESETS (no GPU) and exit")
    parser.add_argument("--parent", type=int, metavar="PID",
                        help="the studio server's pid: end with it (the server passes its own)")
    args = parser.parse_args(argv)
    if args.check_install:
        if args.job or args.check or args.parent is not None:
            parser.error("--check-install takes no other option")
        return check_install_main()
    if not args.job:
        parser.error("--job is required (or --check-install)")
    job_path = Path(os.path.abspath(args.job))
    if args.check:
        return check(job_path)
    if args.parent is not None and not tie_to_parent(args.parent):
        print(error_line(f"the studio's server (pid {args.parent}) is gone; not starting"), flush=True)
        return EXIT_CANCELLED

    signal.signal(signal.SIGTERM, cancel)
    apply_environment(os.environ)
    started = time.time()
    receipt = {"state": "running", "model": MODEL, "job_file": str(job_path), "pid": os.getpid(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)), "times_s": {},
               "peak_gib": {}, "lowest_mem_available_gib": {}}
    out, trace, status = job_path.parent, MemTrace(ABORT_BELOW_GIB), 0
    try:
        job = prepare(job_path, receipt)
        out = job["output"]
        write_json(out / RECEIPT, receipt)
        trace.floor = job["abort_below_gib"]
        trace.start()
        run(job, receipt, trace)
        receipt["state"] = "done"
    except (Cancelled, KeyboardInterrupt) as error:
        receipt.update(state="cancelled", error=str(error) or "interrupted")
        status = EXIT_CANCELLED
    except RunAborted as error:
        receipt.update(state="aborted", error=str(error))
        status = EXIT_ABORTED
    except JobError as error:
        receipt.update(state="refused", error=str(error))
        status = EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - recorded in the receipt; the exit status says it failed
        receipt.update(state="failed", error=one_line(f"{type(error).__name__}: {error}", 2000),
                       traceback=traceback.format_exc().splitlines()[-30:])
        traceback.print_exc()
        status = EXIT_FAILED
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        trace.stop()
        lows = receipt.setdefault("lowest_mem_available_gib", {})
        if trace.low:
            lows["all"] = min(list(trace.low.values()) + [v for k, v in lows.items() if k != "all"])
        peaks = list(receipt.get("peak_gib", {}).values())
        receipt.update(peak_allocated_gib=max(peaks) if peaks else None, wall_s=round(time.time() - started, 1),
                       exit_status=status)
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
        say(f"done in {receipt['wall_s'] / 60:.1f} min: {out / receipt['output']['file']}")
    return status


if __name__ == "__main__":
    sys.exit(main())
