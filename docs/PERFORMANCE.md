# Performance: Sol-H3-Spark on one DGX Spark

Measured on one DGX Spark (GB10, 128 GB unified memory, ~121 GiB usable), DGX OS (Ubuntu 24.04), driver
580.178, CUDA 13.0, internal NVMe. Stage 1 runs in a torch 2.12+cu130 environment, Stage 2 in torch 2.13+cu132
(`install/freeze/`). Nothing else was running. Model weights are not part of this repo.

## Results

| | Before | After | Before = |
|---|---|---|---|
| t2va 16:9, 5 s: launch to first finished MP4 | 282 s | **99 s** | NVIDIA's flow plus this patch's persistent compile cache (without it: ~359 s) |
| t2va 16:9, 15 s, first clip | 681 s | **238 s** | the pipeline before these changes: NVIDIA's flow with only this patch's length change (NVIDIA supports 5 s only) |
| ref2va with a 5 s reference video, each clip | 400 s | **133 s** | the same pipeline with NVIDIA's reference-video canvas (H3's native 768p) instead of this repo's draft canvas |

The VAE fix alone, measured with this repo's loading changes (overlap, deliver-first, O_DIRECT, the 8-call
attention check, the persistent caches) already in on both sides:

| | Before | After |
|---|---|---|
| fl2va (first + last frame), 5 s, first clip | 194 s | **115 s** |
| ref2va, image + video reference, 5 s, first clip | 254 s | **178 s** |

Notes on the "Before" figures:

- t2va 5 s: NVIDIA's flow with a per-session compile cache measured 304 s of startup plus a 55 s clip, ~359 s.
  The persistent compile cache (part of this patch, item 7 in [PATCH.md](PATCH.md)) brought the startup to 227 s,
  hence the 282 s baseline.
- t2va 15 s: 490 s of startup plus a 191 s clip, measured before the loading changes and before the persistent
  compile cache existed. The 238 s is with a warm compile cache; the very first 15 s run, which compiles the 15 s
  shapes, took 357 s.
- fl2va and ref2va: no stock NVIDIA first-clip figure was measured. An earlier web-UI run of the same ref2va
  setting, before the loading changes, took ~553 s (421 s loading plus a 132 s clip); it already had this repo's
  reference-video canvas, so it is not an NVIDIA figure either.

The 99 s is the `infer.py --deliver-first` bench (the script `bench/first-clip.sh` was ported from). The web UI's
own first clip after a start measured 104-113 s for the same setting (t2va 16:9 5 s, 2026-10-10; an earlier
load-time record had ~121 s). The difference to the bench was not broken down.

Warm clips (models loaded): 5 s 54-66 s · 10 s 123 s · 15 s 191 s. NVIDIA lists 56 s for a warm 5 s 16:9 t2va clip;
on this box the warm t2va mean was 65.9 s for 16:9 and 56.6 s for 9:16. 5 s 16:9 is NVIDIA's route; 9:16 and the
other lengths (the studio offers 5-15 s) are this repo's extensions ([PATCH.md](PATCH.md)).

## How it was measured

- Metric: wall time from launching the pipeline to the first finished MP4 of a new session (t2va 16:9, 5 s,
  seed 42). Warm clips: one session per length. `bench/first-clip.sh` and `bench/duration.sh` are ports of the
  scripts that took these measurements (same steps and memory guard; paths and settings now come from the env
  file).
- MemAvailable sampled every 0.25 s; a guard (`bench/memguard-run.sh`) stops the run below 5 GiB.
- Repeat runs of one configuration differed by up to ~7 s, so changes under ~5 s are noise.
- Stage 1 output is bit-identical to upstream (audio MD5 equal in every run). Stage 2 is not deterministic run
  to run, even upstream: PSNR 43.3-43.6 dB upstream vs upstream, 42.2-43.4 dB upstream vs this repo (measured
  before 0.3, with Stage 2 as upstream, today's `SOL_H3_STAGE2_AUDIO=joint`). With the default frozen audio, Stage 2
  differs from upstream by design: about 35 dB, 3/255 on average
  ([PATCH.md](PATCH.md#15-stage-2-audio-held-fixed-lip-sync)). The timings here don't change with it.
- All 4,349 LTX transformer and 3,320 LoRA tensors read by the new loader were compared bit for bit with
  safetensors (earlier, also 856 adapter tensors). The H3 shards and the VAEs were not compared tensor by tensor.
- The numbers were measured with a different prompt of similar length. `bench/prompts/bench.jsonl` ships NVIDIA's
  "mountain-lake" example prompt (seed 42) instead. The first run of a new prompt length compiles it once
  (see "Known costs").

## Cold start, step by step (t2va 16:9, 5 s)

| Change | First clip |
|---|---|
| Baseline: NVIDIA's flow plus the persistent compile cache | 282 s |
| Clone mmap tensors into RAM before the GPU copy | 282 s (no gain) |
| Loading pass renders the requested clip; stages overlap | 177 s |
| + read-ahead of all weights into the page cache | 195 s (worse, removed) |
| O_DIRECT weight reads + 8-call attention self-check | 127 s |
| Triton autotune results kept on disk | 124 s |
| O_DIRECT for the 5.3 GB LoRA | 110 s |
| Audio VAE loaded during the first denoise | 106-110 s |
| Read-ahead of the next O_DIRECT window | 108 s (within noise) |
| Inductor autotune-cache fix | **99 s** |

## What made the difference

**Render the requested clip while loading (282 → 177 s).** Upstream loads the models by rendering a throwaway
warm-up clip (55 s), then renders yours. Here the loading pass renders yours. The prompt encoder runs while
Stage 1 starts, and Stage 2 loads while Stage 1 compiles. Stage 2 waits until Stage 1 has converted its
transformer from BF16 to FP8, Stage 1's memory peak. The first attempt at the overlap fell to 2.4 GiB free and was
stopped by the guard; with the wait, the lowest point was 18 GiB.

**Read weights with O_DIRECT (177 → 127 s, with the next item).** Loading safetensors through mmap page faults
ran at ~1.3-2 GB/s with the default 128 KB read-ahead. The same NVMe streams 11.8 GB/s with large O_DIRECT reads.
`runtime/fastload.py` reads each file into 256 MiB aligned windows with 4 parallel `preadv` calls and copies
tensors to the GPU from there. H3 shards: 33 → 15 s. LTX transformer (42 GB): ~40 → 11 s, then 8.3 s with
forward-only read-ahead of the next window. It also keeps weights out of the page cache, which on GB10 is memory
the GPU can't use. Copying to the GPU straight from a file-backed mmap is itself slow on GB10 (~1 GiB/s vs
~18 GiB/s from ordinary RAM, warm cache). Fixing only that copy gained nothing here: cold page faults were the
limit.

**Shorter first-run attention check.** On the first request upstream validates 200 cuDNN block-sparse attention
calls against the Triton kernel (~30 s of synchronization). This repo checks the first 8.
`SOL_H3_BSA_CHECK_CALLS=200` restores the full check.

**O_DIRECT for the 5.3 GB LoRA (124 → 110 s).** The Stage 1 transformer is resident 6 s sooner.

**Stop re-tuning kernels every session (→ 99 s).** Triton autotune results now persist
(`TRITON_CACHE_AUTOTUNING=1`, 127 → 124 s). Separately, with torch 2.13, 11 of 61 Stage 2 kernels loaded from
the FX graph cache re-ran coordinate-descent tuning in every new process (3-4 s). Their cached results had been
pruned to one config, and `CachingAutotuner.recheck_autotune_cache` hashed that one-config list instead of the
candidate list the result was saved under, so the lookup always missed. After a miss, the patch reads the entry
through the kernel's own `AutotuneCache`. Result: 11/11 hits, Stage 2's first render 45-49 → 41 s.

**Audio VAE off the critical path.** Loaded on a side thread during the first denoise: audio decode in the first
clip 5.8 → 0.4 s. Stage 1 and Stage 2 now finish loading together at ~62 s.

**Video VAE built on the GPU (fl2va/ref2va: 70 s → 2.9 s).** FastVideo's loader built the 10 GB video VAE on
the CPU, randomly initialising ~5B parameters on one thread, then moved it to the GPU and loaded the real
weights. Building it under `torch.device("cuda")` and loading with O_DIRECT takes 2.9 s. First clip: fl2va
194 → 115 s, ref2va 254 → 178 s.

**Reference videos at the draft size (ref2va: 400 → 133 s per clip).** Sol-H3 resized reference videos to H3's
native 768p canvas: 6,714 context tokens, 228 s draft denoise, 6 GiB free. Shrinking the input didn't help
because it was upscaled again. Reference videos are now prepared at the draft canvas (short side 384, at most
672×384), as Sol-H3 already does for reference images: 2,178 tokens, 54 s denoise, 19 GiB free. Output looked
comparable side by side.

## Memory

- PyTorch's aarch64 wheels allocate CPU tensors with a bundled mimalloc that, by default, keeps freed memory while
  the process is idle. On unified memory that is GPU memory too. `MIMALLOC_PURGE_DELAY=0` (set for both workers):
  RSS after freeing 3.55 → 0.47 GiB; fl2va lowest free memory 9.3 → 19.8 GiB.
- Clips over 10 s load the prompt encoder per request. With it resident, a 15 s refine fell to 4 GiB free.
  Keeping it resident saved nothing (192.9 vs 190.8 s per clip).
- Lowest MemAvailable: 15-18 GiB for a 5 s first clip (upstream flow: 13 GiB); 12 GiB for 15 s. Measured before
  0.4, when the resident prompt encoder loaded after the first clip. It now loads during the first clip's refine,
  so the next clip doesn't wait for it. On a separate test studio on the same DGX Spark (2026-10-13) that took the
  lowest point of a load from 14.0-16.3 to 10.3-12.1 GiB (t2va 5 s, fl2va 10 s, a warm-up), about what the next warm
  clip reaches (11.8-12.2 GiB). [PATCH.md](PATCH.md) item 4 has the numbers.
- Since 0.5.2 a Keyframes session also holds the LTX-2.5 VAE's encoder for the keyframe pin (608 MiB). On a test
  studio (2026-10-29) a clip's lowest MemAvailable was 11.3 GiB with it against 11.9 GiB without at fl2va 16:9 10 s
  (the session's first clip) and 18.5 against 19.5 GiB at 9:16 15 s; 5 s Keyframes clips went down to 12.2-12.4 GiB.
  The resident prompt encoder still loaded during the first clip. The 110 GiB start floor and the 4 GiB guard stay
  ([PATCH.md](PATCH.md#19-stage-2-keyframe-pin)).
- Read-ahead into the page cache made the cold start slower (177 → 195 s): 54 GB of Stage 2 read-ahead competed
  with Stage 1's compile for disk and memory.

## Reliability: prompt-length recompiles

Stage 1's transformer compiled new graphs for each new prompt length (in text tokens). FastVideo caps recompiles at 16;
the 4th distinct prompt length in a session hit the limit, killed the worker and forced a full reload. The limit is
raised (1,024; 65,536 accumulated). Tested: 7 distinct prompt lengths in one session, no failure. Since 0.4
`SOL_H3_DYNAMIC_TEXT_LENGTH=1` (the default) stops the per-length graphs of the body blocks; only the two text-refiner
regions still compile once per new length. A session can also change its clip length and format, and every graph is
static, apart from the dimensions that setting marks dynamic. With `SOL_H3_DYNAMIC_TEXT_LENGTH=0` each pair of prompt
length and clip shape then compiles its own graphs, 4-5 per region (inferred), and the limit covers about 200-250 such
pairs in one session (inferred); past it the worker fails and the clip reloads the models.

## Changing length or format

Before 0.4 a clip with another length or format closed the loaded session and loaded every model again, like a
change of mode. Measured cost of such a switch: the first clip after it, minus a warm clip of the new setting
(seconds, on one DGX Spark, from the studio's worker logs and `results.json`; issue #45):

| | 5 → 15 s, t2va 16:9 | 15 → 5 s, t2va 16:9 | 9:16 5 s → 16:9 10 s, fl2va | mode only, 16:9 10 s |
|---|---|---|---|---|
| Close the old session | 15 | 5 | 7 | 9 |
| Start Stage 1 and the prompt encoder | 15 | 14 | 15 | 14 |
| Stage 1 weights: read, LoRA merge, FP8 | 21 | 22 | 20 | 23 |
| Stage 1 first denoise, extra over a warm clip | ~35 | ~16 | 22 | 12 |
| Stage 2 first run, extra over a warm clip | ~9 | ~6 | 5 | 7 |
| **Total extra** | **~81** | **~62** | **~65** | **~63** |

Now the session keeps its models for another length or format of the same mode and changes its shape in place
([PATCH.md](PATCH.md#18-reshape-another-length-or-orientation-without-a-reload)). The first two rows and the weights go
away; the first clip at a shape the session has not run compiles its graphs (from the compile cache when that shape ran
before): Stage 2's, and with `SOL_H3_DYNAMIC_TEXT_LENGTH=0` Stage 1's body blocks too. Measured in one run before 0.4's
defaults (`SOL_H3_DYNAMIC_TEXT_LENGTH` off; t2va and fl2va, one prompt; first clip after the change minus a warm clip of
that shape): 5 → 10 s 14.4 s, 5 → 15 s 27.0 s (both compiled Stage 1 for this prompt length from scratch), 15 → 5 s (a
shape the session had run) none, fl2va 16:9 → 9:16 0.2 s. Reloads in the same run cost 37.5-47.1 s over a warm clip,
plus 4-9 s to close the previous session. In the staging check of 0.4 (t2va, the defaults, a warm compile cache, one
run): 5 → 10 s 7.1 s and 10 → 15 s 10.1 s, of which Stage 1 1.8 s and 2.7 s, with no Stage-1 compile; 10 → 5 s (a shape
the session had run) none. Above 10 s the prompt encoder still loads per request; after a change back to 10 s or less
the resident encoder loads again in the background if MemAvailable is at least the minimum of its early load (32 GiB up
to 5 s, 34 GiB above; [PATCH.md](PATCH.md) item 4), else after the next clip, so the first clip after such a change may
wait for it. A change of mode still reloads.
`LFSTUDIO_RESHAPE=0` reloads for any change.

## Known costs, not fixed

- A new prompt length on a warm session added 9-10 s before 0.4 (Stage 1 17.6 → 26.2 s): the body blocks' graphs were
  tied to the length through FastVideo's H3 sparse-attention metadata and tiling (`total_seq_length`, the tile counts)
  and to the denoise step. Since 0.4 `SOL_H3_DYNAMIC_TEXT_LENGTH=1` (the default;
  [PATCH.md, 17](PATCH.md#17-prompt-lengths-without-recompiles)) compiles them per session (2 graphs: the first denoise
  step and the rest), not per length. What is left is the two text-refiner regions, which compile once per new length
  with static compiles (0.4, `SOL_H3_STATIC_SHAPES=1`). In the staging check of 0.4 (a separate test studio on the same
  DGX Spark, 2026-10-14, 19 clips in 5 sessions, both settings on) a new prompt length in t2va 5 s cost 1.9 s in Stage 1
  over the same prompt repeated later when its text-refiner graph was new to the compile cache, and 0.5 s when it was
  not; no body block compiled for it. Every pair of clips that should match was bit-identical, and so were 5 clips
  against renders made before these settings. Reusing a prompt, or editing it without changing its token count, avoids
  even that. An earlier run with dynamic compiles (`SOL_H3_STATIC_SHAPES=0`; 2026-10-13, 19 clips each way): a new
  length cost 0.0-0.8 s in t2va and fl2va with the setting, 2.3-9.7 s without; known-length clips +0.3 to +0.9 %; every
  warm clip bit-identical. The text refiner still compiled once more at a session's second length (2.6 s in that ref2va
  clip). In ref2va the dense body used to become length-dynamic at a session's second length, so its third new length
  cost nothing even without the setting (0.0 s in that run). With static compiles and `SOL_H3_DYNAMIC_TEXT_LENGTH=0` it
  compiles per new length again, as in t2va and fl2va (inferred).
- Cold vs warm overhead (~55 s), measured before the last fix: Stage 2 first compile 15-20 s, Stage 1 LoRA merge
  + FP8 ~7 s, Stage 1 first-denoise compile ~11 s, audio VAE 6 s.
- The numbers above assume the compile cache is warm from an earlier run of the same mode, format and length.
  The first run of a configuration fills it: t2va 9:16 5 s 111 s, fl2va 9:16 114 s, ref2va 9:16 176 s,
  t2va 16:9 10 s 264 s, t2va 16:9 15 s 357 s.

## Switches

| Variable | Default | Effect |
|---|---|---|
| `SPARK_FAST_SAFETENSORS` | 1 | 0 = upstream safetensors loading (also in the experimental Upscale's fast load) |
| `SPARK_DIRECT_IO` / `SPARK_DIRECT_AHEAD` | 1 / 1 | O_DIRECT reads / read-ahead of the next window (also in the Upscale's fast load) |
| `SOL_H3_BSA_CHECK_CALLS` | 8 | 200 = upstream's full attention self-check |
| `SOL_H3_COMPILE_CACHE` | `$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` | `session` = upstream per-session cache |
| `SPARK_AUTOTUNE_CACHE_FIX` | 1 | 0 = upstream Inductor behaviour |
| `SOL_H3_VAE_ON_DEVICE` | 1 | 0 = build VAEs on the CPU as upstream |
| `SOL_H3_AUDIO_VAE_PREFETCH` | 1 | 0 = load the audio VAE when first needed |
| `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` | 241 | longer clips load the prompt encoder per request |
| `SOL_H3_STATIC_SHAPES` | 1 | 0 = upstream `torch.compile(dynamic=None)` |
| `SOL_H3_RESIDENT_QWEN_EARLY` | 1 | 0 = the resident prompt encoder loads after the first clip, not during it |
| `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` | 32 up to 5 s, 34 above | MemAvailable needed to load it during the first clip, or right after a change back to 10 s or less |
| `SOL_H3_DYNAMIC_TEXT_LENGTH` | 1 | 0 = Stage 1's body blocks compile per prompt length, not once per session |
| `SOL_H3_STAGE2_KEYFRAME` | pin | Stage 2 holds a Keyframes clip's first frame: frame 0 at 30-38 dB against the image instead of 17-23 dB, Stage 2 time unchanged (37.4 s against 37.5 s warm at 5 s, 83.6 s against 83.8 s at 10 s), 608 MiB more resident in Keyframes sessions (the encoder's weights) and a clip's lowest MemAvailable 0.6-1.0 GiB lower (one DGX Spark, 2026-10-29; on one RTX 5090 frame 0 at 32-36 dB instead of 18-21 dB, Stage 2 15.0-15.4 s against 14.9-15.2 s warm at 5 s, the same GPU peak, 608 MiB more RAM; [PATCH.md](PATCH.md#19-stage-2-keyframe-pin)). `off` = upstream: Stage 2 repaints it |
| `LFSTUDIO_PRELOAD` (web UI) | 1 | 0 = no warm-up of the last settings at launch |
| `LFSTUDIO_RESHAPE` (web UI) | 1 | 0 = another length or format closes the session and loads the models again |

The warm-up at launch (`LFSTUDIO_PRELOAD`) is a cold start like any other: 102-112 s for t2va 16:9 5 s with a filled
compile cache. A clip with the same settings submitted while it runs used to wait for it and then render warm
(186 s from submit to clip when submitted at launch). Now such a clip takes the warm-up over: it stops the warm-up
and loads the models itself (112 s from submit to clip, measured 2026-10-10). It does so while the warm-up has run
less than a warm clip of that setting takes, minus ~5 s for stopping it; the studio records warm clip times, and
before one is recorded it uses half the expected load time. Later in the warm-up, waiting for it is the faster
choice. Either way nothing is lost by submitting early, and the Generate button shows which applies and when the
clip would be ready ("replaces the warm-up" or "after the warm-up", from the recorded load and clip times). A clip
for other settings stops the warm-up too, unless a queued clip is waiting for it. "Load selected now" starts the
same kind of warm-up for the selected settings on request (with the prompt and inputs of the newest clip of that
mode, or a built-in prompt for Text), so the same rules apply to it. If other models are loaded, they are closed
first (~40 s; the status shows "switching to ..."); a clip or Unload in that time cancels the request, and the clip
loads its own settings right after the close. For another length or format of the loaded mode the models stay:
the session only changes its shape, without a warm-up clip (the status shows "changing to ..."), and the page offers
no "Load selected now" for it. A clip of the warm-up's mode with another length or format counts as one for the same
models: early in the warm-up it takes the warm-up over, later it waits for it and the session then changes its
shape.

## Experimental Upscale (Open-H3-Regenerate-2K)

Measured on one DGX Spark (GB10) on 2026-10-28 unless marked, 5 s clips, NVIDIA's standard forward (BF16, dense
attention through cuDNN, about 90 TFLOPS) and its released VAE, the LoRA merged once in place (bit-identical to
NVIDIA's merge, checked). The numbers come from four kinds of runs, kept apart here:

- the studio's own job (`upscale/regen2k_job.py`) in a staging studio with nothing else loaded, with the fast load (its
  default) and, earlier, with the standard load: t2va clips, and on 2026-10-29 Keyframes, Reference and LightX2V Turbo
  clips (below);
- the bench wrapper the job was made from (the same NVIDIA code and changes, the studio stopped), with the standard
  load;
- that bench wrapper with the fast load;
- research runs: the job on LightX2V Turbo clips in folders built by hand, and the prompt encode on its own ("Measured
  apart", below).

The first tables have one run per resolution and orientation; the runs through the studio on 2026-10-29 add more.
Every run here is of a 5 s clip: clips of other lengths ran on 2026-10-30 ([Longer clips](#longer-clips), below).

**The job itself, fast load** (staging studio, with the job as it was just before it took 9:16 clips: its calls into
NVIDIA's code for a 16:9 clip are unchanged since, which `tests/test_upscale.py` checks; one 16:9 clip (1344×768) to
each resolution, one after the other, the same clip and seed; `balanced`: 2 steps from 0.7, audio `clip`; the stages
from the job's `receipt.json`, the total from the studio's record):

| Stage | 1080p (1920×1080) | 2K (2560×1440) | 4K (3840×2160) |
|---|---|---|---|
| Encoding the final clip (`encode_video`) | 131 s | 129 s | 133 s |
| Encoding the clip's audio | 0.6 s | 0.6 s | 0.6 s |
| Loading the models (fast; the LoRA merge 2.8 s of it) | 21.5 s | 21.6 s | 21.7 s |
| The latent upscaler | 3.1 s | 4.5 s | 14.8 s |
| Refine, per step | 151 s, 149 s | 379 s, 380 s | 1684 s, 1688 s (the lean forward, below) |
| VAE decode | 108 s | 182 s | 396 s |
| MP4 (x264) | 3.5 s | 15 s | 12.7 s |
| **From the job's start to the MP4** | **574.2 s (9:34)** | **1122.6 s (18:43)** | **3958.1 s (65:58)** |
| Peak GPU memory allocated (in the refine) | 81.29 GiB | 91.08 GiB | 94.46 GiB |
| Lowest MemAvailable: in the refine; in the load | 27.4; 37.1 GiB | 15.7; 37.2 GiB | 11.1; 36.9 GiB |

The job's start is when the studio takes it from the queue (`started` in its record; the request came 5.4-6.4 s
before), its end the MP4 with the clip's sound (`wall_s`). No studio models were loaded, so there was nothing to close
first. 4.9-5.0 GiB were allocated before the decode (the DiT freed). The two loads give the same result: the 2K run's
video stream is MD5-identical to that of the standard-load run below (the same clip and seed).

**9:16, the job with the fast load** (the same staging studio and settings, later that day, with the job as it takes
9:16 clips; one 9:16 clip (768×1344), another prompt and seed). The job takes it through NVIDIA's code with the shapes
turned: at 2K its upscaler goes straight to the portrait grid (latents 48×84 → 90×160, pixels 1440×2560; `upscale_route`
`[[160, 90]]` in the receipt), at 1080p to `[[120, 68]]`. The outputs were 1440×2560 and 1080×1920, 121 frames, with the
clip's sound.

| Stage | 1080p (1080×1920) | 2K (1440×2560): **CPU work alongside, not a clean measurement** |
|---|---|---|
| Encoding the final clip | 128 s | 133 s |
| Loading the models (fast) | 21.9 s | 25.9 s |
| The latent upscaler | 3.0 s | 5.1 s |
| Refine, per step | 143 s, 144 s | 413 s, 395 s |
| VAE decode | 105 s | 188 s |
| MP4 (x264) | 3.4 s | 15.8 s |
| **From the job's start to the MP4** | **553.9 s (9:14)** | **1186.6 s (19:47)** |
| Peak GPU memory allocated | 81.26 GiB | 91.06 GiB |
| Lowest MemAvailable (in the refine) | 28.5 GiB | 9.9 GiB |

The 2K run did not have the machine to itself, from its start: MemAvailable was 111.9 GiB when the job started
(112.4 GiB before the studio was up), against 116.1-116.9 GiB at the start of the job's other runs here, and its load
took 25.9 s against 21.5-21.9 s; during its refine CPU test suites (a browser test run and a CPU model-plan check) ran
on the same machine. So none of its times or memory figures are clean measurements, and nothing here is estimated from
them. The 1080p run had the machine to itself and came out as the 16:9 one did, within the spread of the runs so far
(143-144 s per step against 149-151 s, and 144 s in the bench; peak 81.26 against 81.29 GiB; lowest MemAvailable 28.5
against 27.4 GiB). That fits the counts: the latent grids are the 16:9 ones turned, so the refine has the same number of
tokens at each resolution (133,200 video rows at 2K, for example), the decode the same number of 256 px tiles and the
upscaler the same scale factors. A clean 9:16 run at 2K came the next day, through the studio (below: 1117.4 s with a
9 s encode of its text, against 1122.6 s for 16:9). The page and the client show the 16:9 times for a 9:16 clip. **9:16
at 4K has not run**; there the upscaler goes through the 2K grid (turned), as for 16:9.

**Every 5 s clip: Keyframes, Reference and LightX2V Turbo clips.** The studio now gives the job every 5 s clip
through an input folder of its own (docs/API.md, Upscale): the clip's MP4 (a LightX2V Turbo clip at 576p stretched to
768p; one with a last keyframe was cut to 121 frames in the first runs below, and since #85 keeps all 124 at 1080p and
2K),
its audio, and a prompt conditioning of text alone (for Sol-H3 Keyframes and Reference clips encoded first). The job's
work is a Text clip's work: the same latent frames (37 for 121 or 124 frames), sizes and steps.

Through the studio (a staging studio of this version on the same DGX Spark, 2026-10-29, driven by `lf-cli`; one clip of
each kind, made in that studio just before; nothing else loaded, so no models to close, and MemAvailable
116.3-116.7 GiB at each job's start; the fast load, `balanced`, the clip's audio; the time from the job's start to the
MP4 with the clip's sound, the encode included; the stages from the job's `receipt.json`):

| The clip | What the studio does with it | To | From the job's start to the MP4 | Refine, per step | Peak allocated | Lowest MemAvailable |
|---|---|---|---|---|---|---|
| Sol-H3 Keyframes, a first frame, 9:16 | its text encoded first (9.2 s) | 1080p (1080×1920) | **566.8 s (9:27)** | 143.9 s, 142.7 s | 81.29 GiB | 29.2 GiB |
| Sol-H3 Reference, one picture, 16:9 | its text encoded first (9.2 s) | 1080p | **562.9 s (9:23)** | 143.6 s, 142.4 s | 81.30 GiB | 28.8 GiB |
| LightX2V Turbo Text, 576p 16:9 | stretched to 768p; its own text | 1080p | **554.4 s (9:14)** | 143.4 s, 142.4 s | 81.29 GiB | 28.4 GiB |
| LightX2V Turbo Keyframes, 768p 16:9, first and last keyframes | cut from 124 frames to 121; its `qwen-text/` | 1080p | **553.6 s (9:14)** | 142.6 s, 142.1 s | 81.27 GiB | 28.0 GiB |
| The Sol-H3 Keyframes clip again | its text encoded again (9.1 s) | 2K (1440×2560) | **1117.4 s (18:37)** | 375.9 s, 375.9 s | 91.09 GiB | 16.8 GiB |

The other stages, at 1080p (2K): encoding the final clip 128.3-128.6 s (128.3 s), its audio 0.6-0.7 s, the load
21.7-21.9 s (21.8 s), the latent upscaler 2.9-3.0 s (4.3 s), the VAE decode 105.1-105.3 s (181.3 s), the MP4 3.2-3.6 s
(14.2 s).

- **Times:** each came 1-4 % under the Sol-H3 Text clip's time at its resolution (574.2 s at 1080p, 1122.6 s at 2K,
  the first table) plus its encode, so the Text clip's times hold for these kinds. The 2K run is the clean 9:16 run at
  2K that the table above lacks: 376 s per step against 379-380 s for 16:9, the same peak, its lowest MemAvailable
  16.8 GiB against 15.7 GiB.
- **The encode** (status `encoding the prompt`) took 9.1-9.2 s in all. Its receipts: task t2va, the text asked for, no
  vision rows (195 and 232 rows, the text's tokens alone). The 2K upscale's encode of the Keyframes clip's text wrote
  the same `conditioning.pt` as the 1080p one's, byte for byte, and so did a LightX2V Turbo Keyframes clip's
  `qwen-text/` of the same prompt (encoded with that clip in the studio's queue:
  [LightX2V Turbo](#experimental-lightx2v-turbo)). So such a clip's `qwen-text/` is what an encode at upscale time
  would give.
- **The sound:** each upscale has the clip's own: every AAC packet of a linked or stretched clip (5.04 s), and for the
  clip cut to 121 frames 5.02 s of its 5.17 s, cut with the picture. That cut was that day's route: it left out the
  clip's last 3 frames, the last of them the one anchored to the last keyframe. Since #85 such a clip keeps all 124 at
  1080p and 2K, with its 5.17 s of sound (measured the next day, below), and is cut to 121 only at 4K.
- **Cancel** about 2 s into the encode (an upscale of the Reference clip) stopped it in 5.4 s, with no encode or job
  process, Qwen container or GPU memory left, and no job started.
- **Not run through the studio:** LightX2V Turbo's 768p Text clips (linked as they are), its Keyframes clips at 768p
  without a last keyframe (linked, as those with one, which have run: below), one whose text is encoded at upscale time
  (a Keyframes clip without `qwen-text/`), its 576p clips at 2K (run at 1080p only; at 2K apart, in research runs:
  below), and 4K of any clip but a 16:9 Text clip (no 9:16 clip and no 124-frame clip has run at 4K). Expected: a Text
  clip's times, plus the encode where there is one.

**Keep my frames** (`pin`). The pinned refine adds one block of condition rows per end held, as MiniMax-H3 holds a
keyframe: +2,040 rows at 1080p and +3,600 at 2K per end (+2.7 %; 78,130 and 137,410 rows with one end, 80,170 and
141,010 with both, for a prompt of that length), and encodes each picture on the VAE the clip's encode already loaded.
Before the job had it, research runs held a first frame the same way (the job run by a research wrapper that adds the
block; LightX2V Turbo 576p clips, 2026-10-28, one clip at 1080p and two at 2K, the same machine, with CPU work
alongside), against the same clips unpinned:

| Clip | Steps pinned | Steps unpinned | Per step | Peak allocated |
|---|---|---|---|---|
| 1080p | 149.9, 149.4 s | 151.3, 148.9 s | within the noise | 81.66 GiB (+0.37) |
| 2K, the first clip | 395.2, 394.7 s | 386.6, 389.5 s | +7 s | 91.74 GiB (+0.65) |
| 2K, the second clip | 406.3, 395.2 s | 376.2, 374.7 s | +25 s (+20 s in its second step) | 91.73 GiB (+0.65) |

The second 2K clip's first step ran with more CPU work alongside (its lowest MemAvailable 11.7 GiB, 16.7 GiB in its
second step), but its second step was still 20 s longer. Their picture encode took 8.7 s (1080p) and 10.5-10.6 s (2K),
with a VAE load and two diagnostic encodes. Measured on the CPU (2026-10-29, three LightX2V Turbo clips): the job's
check before anything loads (the pictures' sha256, one frame of the clip per end) took 0.18 s for one end and
0.27-0.29 s for both.

**124 frames and Keep my frames through the studio** (a staging studio of this version on the same DGX Spark, the
night of 2026-10-29 to 30, driven by `lf-cli`; clips made in the staging studios of 2026-10-28 and 29; nothing else
loaded, and MemAvailable 110 GiB or more with the page cache dropped at each job's start; the fast load, `balanced`,
the clip's audio; each LightX2V Turbo clip with `pin` `off` and `auto` at the same seed; the time from the job's
start to the MP4 with the clip's sound, the encode included):

| The clip | To | `pin` | From the job's start to the MP4 | Refine, per step | Peak allocated | Lowest MemAvailable |
|---|---|---|---|---|---|---|
| LightX2V Turbo Keyframes, 768p 16:9, first and last keyframes (124 frames, linked) | 1080p | off | **560.1 s (9:20)** | 144.9 s, 142.2 s | 81.27 GiB | 29.1 GiB |
| The same | 1080p | both ends held | **590.4 s (9:50)** | 155.9 s, 155.4 s | 81.98 GiB | 28.2 GiB |
| The same | 2K | off | **1118.9 s (18:39)** | 378.0 s, 377.0 s | 91.06 GiB | 16.0 GiB |
| The same | 2K | both ends held | **1202.1 s (20:02)** | 414.7 s, 413.5 s | 92.32 GiB | 14.1 GiB |
| LightX2V Turbo Keyframes, 768p 16:9, a dialogue, the same picture first and last (124 frames) | 1080p | off | **558.2 s (9:18)** | 143.4 s, 142.5 s | 81.30 GiB | 29.0 GiB |
| The same | 1080p | both ends held | **592.6 s (9:53)** | 156.5 s, 157.2 s | 82.02 GiB | 28.2 GiB |
| LightX2V Turbo Keyframes, 768p 9:16, first and last keyframes (124 frames) | 1080p (1080×1920) | off | **558.5 s (9:18)** | 143.5 s, 143.0 s | 81.29 GiB | 29.0 GiB |
| The same | 1080p | both ends held | **591.0 s (9:51)** | 156.1 s, 155.6 s | 82.00 GiB | 27.3 GiB |
| LightX2V Turbo Keyframes, 576p 16:9, a last keyframe alone (124 frames, stretched) | 1080p | off | **557.8 s (9:18)** | 142.9 s, 142.0 s | 81.27 GiB | 27.5 GiB |
| The same | 1080p | its last frame held | **573.8 s (9:34)** | 149.4 s, 148.8 s | 81.64 GiB | 27.8 GiB |
| LightX2V Turbo Keyframes, 576p 9:16, a first frame (121 frames, stretched) | 1080p (1080×1920) | off | **558.8 s (9:19)** | 143.4 s, 142.6 s | 81.29 GiB | 28.8 GiB |
| The same | 1080p | its first frame held | **575.4 s (9:35)** | 149.9 s, 149.2 s | 81.66 GiB | 27.7 GiB |
| The Sol-H3 Keyframes clip of the table above (9:16, a first frame; its text encoded first, 15.6 s) | 1080p | its first frame held | **590.0 s (9:50)** | 150.2 s, 149.6 s | 81.66 GiB | 28.0 GiB |

- **124 frames:** every upscale of a clip with a last keyframe had all 124 frames and the clip's whole sound (5.167 s,
  every AAC packet), in a Text clip's time (557.8-560.1 s at 1080p against 574.2 s, 1118.9 s at 2K against 1122.6 s:
  the same 37 latent frames). Against the 768p clip's upscale cut to 121 frames on 2026-10-29 (553.6 s, the same seed
  and text), frames 0-119 were a mean 40.3 dB apart, and the last frame, the one anchored to the last keyframe, was
  25.9 dB from that picture, where the cut's last (frame 120) was 23.2 dB.
- **Keep my frames, the picture:** every end sent was held (the job's check: 31.2-37.9 dB at 384p). Against the same
  clip upscaled with `pin` `off`, the frame held came 3.2-5.3 dB closer to the picture (luma PSNR at the output size;
  2.5-5.5 dB on a face), at either end and at 1080p and 2K (+5.0 and +4.7 dB at 2K), with no flash at that frame and
  the motion near it kept (the run's checks). The Sol-H3 clip is compared with its upscale of 2026-10-29 (the table
  above): +4.7 dB, +4.2 dB on the face. The dialogue clip gained less, +2.3 and +2.4 dB at its ends (+3.0 and +3.1 dB
  on the face): it is a static shot, and its plain upscale was already closer to the picture (30.0-30.1 dB at either
  end, against 24.1-27.0 dB for the others). Its lip sync held (SyncNet LSE-C 8.94, 8.82 unpinned, offset 0). The
  research runs with a first frame at 2K had brought frame 0 only about 1 dB closer (the face 3-5 dB); with these runs
  2K is offered as 1080p is.
- **Keep my frames, the cost:** +16.0 and +16.6 s with one end, +30.3 to +34.4 s with both at 1080p, +83.2 s with both
  at 2K: the pictures' encode (3.0-3.1 s per end at 1080p, 10.5 s for both at 2K) and longer steps (149-150 s with one
  end and 155-157 s with both, against 142-145 s; at 2K 414 s against 377 s). Peak allocated +0.4 GiB with one end and
  +0.7 GiB with both at 1080p, +1.3 GiB at 2K; lowest MemAvailable 27.3 GiB or more at 1080p, 14.1 GiB at 2K with both
  ends against 16.0 GiB unpinned. At 2K this was one pinned refine, the route the cascade now replaces
  ([below](#keep-my-frames-at-2k-the-cascade)).
- **The job alone** (the same night, on input folders built for it, not through the studio): with a first frame its
  pinned refine gave the research wrapper's refined latent bit for bit (the same clip, picture and seed); a 16:9 clip's
  first frame came 4.2 dB closer (6.4 dB on the face) for +14 s; with `faithful` a first frame came only 1.2 dB closer,
  so Keep my frames stays with `balanced`. On the dialogue clip with its first frame alone, frame 0 came out 0.05 dB
  from the run with both ends: the last end's block does not weaken the first.

Measured apart:

| | Measured | How |
|---|---|---|
| LightX2V Turbo 576p Keyframes clips (a first frame), stretched to 768p | **1080p 558-576 s (~9.5 min), 2K 1107 s (16:9) and 1136 s (9:16) (~19 min)**; peak allocated 81.3 / 91.1 GiB; lowest MemAvailable 28.0-28.8 / 15.8-16.6 GiB | research runs on one DGX Spark (2026-10-28): this job with the fast load, `balanced`, the clip's audio, on folders built by hand much as the studio now builds them, with two differences: the audio was decoded from the clip's AAC track (not its `audio.pt`), and the text alone was a Sol-H3 Text clip's own conditioning, made by the studio for each prompt without its picture references (not `bench/h3turbo_encode.py --upscale`); the LANCZOS stretch as now; four clips at 1080p (one 16:9, three 9:16), two at 2K; from the job's start to its MP4 without sound (the studio adds the mux, a few seconds, and closing its models, ~40 s) |
| The upscale's prompt encode (Sol-H3 Keyframes and Reference clips; a LightX2V Turbo Keyframes clip without its own text) | 9.1-18 s in all, 1.1-1.8 s of it the encode; ~15 GiB while its Qwen worker lives | the times: ten runs of the same helper (`bench/h3turbo_encode.py`) for LightX2V Turbo clips on the GPU, 2026-10-28; the memory (15.1-15.4 GiB): the receipts of the studio's own Qwen encodes on the same machine |
| The stretch of a 576p clip to 768p | 0.55-0.92 s | two clips, on the CPU |

So the page shows the Text clip's times for every 5 s clip (and a longer clip its length's times, below), plus
15 s in its progress bar for an upscale whose text is encoded first, and its popover says of the kinds not run through
the studio yet (above) that this kind of clip has not been upscaled on the GPU through the studio yet. It goes by
route (the model, where the text comes from, what is done to the MP4), not by size: a route run at one size counts at
the others (so a 576p clip at 2K has no note), and a 768p Keyframes clip with a first frame alone counts as one with
both. A clip of a length that has not run at that size, nor a longer one there, would get the length's note in its
place, that its times are estimates (none does now). With Keep my frames the progress bar adds 16 s per end at 1080p
(measured; on a clip longer than 5 s a little short, below); at 2K it goes through the cascade, whose time is the
job's model of it ([below](#keep-my-frames-at-2k-the-cascade)). The 2K rows of the table above are the single pinned
refine at 2K, the route the cascade replaces.

**The job itself, standard load** (2026-10-28, before the fast load's runs, the same staging studio and 16:9 clip, an
earlier version of the job too; 2K, `balanced`, audio `clip`):

| Stage | Time |
|---|---|
| Encoding the final clip (`encode_video`, its VAE loaded through mmap) | 185 s |
| Encoding the clip's audio | 4 s |
| Loading the models (standard: NVIDIA's loaders through mmap) | 455 s |
| The LoRA merge (in the load) | 74 s |
| Refine, per step | 379 s, 380 s |
| VAE decode | 181 s |
| MP4 (x264) | 17 s |
| **From the job's start to the MP4** | **1620.8 s (27 min)** |

Peak GPU memory allocated 91.08 GiB (in the refine), 4.91 GiB before the decode (the DiT freed); lowest MemAvailable
18.32 GiB (in the refine; 34.13 GiB during the load). Against the fast load's 2K run: 433 s more for the load, 56 s
more for the encode, 498 s more in all; the refine and the decode took the same time.

**The bench wrapper, standard load.** One run per resolution; 1080p and 2K with 3 steps (`full`), 4K with 2
(`balanced`):

| | 768p (no upscale; not offered) | 1080p | 2K (2560×1440) | 4K (3840×2160), see below |
|---|---|---|---|---|
| Loading the models (once per job) | 445 s | 445 s | 445 s | 501 s (another loader) |
| Encoding the final clip | 183 s | 183 s | 183 s | 188 s |
| Refine, per step | 48 s | 144 s | 378 s | 1687 s, 1701 s |
| VAE decode | 49 s | 104 s | 180 s | 399 s |
| MP4 (x264) | 5 s | 8 s | 16 s | 13 s |
| Peak GPU memory allocated | 75 GiB | 81 GiB | 91 GiB | 94.5 GiB |
| Lowest MemAvailable during the job | 38 GiB | 28 GiB | 15 GiB | 13.8 GiB |

**The bench wrapper, fast load** (the setup that `LFSTUDIO_UPSCALE_LOAD=fast` copies into the job; one run at 768p from
a clip's Stage-1 latent, a route of the bench's that the studio does not take, 3 steps). NVIDIA's loaders fault the
weight files in through mmap at about 0.2 GB/s on GB10, and what they read stays in the page cache, the same memory as
the GPU's. The fast load replaces, before anything loads,
`safetensors.torch.load_file` and NVIDIA's `adapters.load_file` with LF Studio's O_DIRECT loader (`upscale/fastload.py`,
a copy of the patch's `runtime/fastload.py`), which copies every tensor from its read window straight to the GPU, and
has diffusers build the DiT and the VAEs on the GPU (`device_map`). Measured: the models loaded in 21 s, the 312 LoRA
weights merged in 2.8 s, the native and refined latents bit-identical to a run with the standard load, ~1.9 GB of
anonymous memory in the process and 40.0 GiB MemAvailable right after the load. Earlier variants that made host copies
of the shards loaded in 23-108 s but kept 10-17 GiB resident (26-32 GiB MemAvailable after the load, against ~42 with
the standard load) and once tripped the memory guard; earlier 2K runs from the final clip with `device_map` and such a
fast load ran without trouble, and encoded the final clip in 135-139 s (three runs), against 183-191 s with the standard
load. The run above started from the Stage-1 latent, so the two encodes did not run under the fast load; the job's
runs above did (128-133 s).

- What the page shows: its progress bar, counted from the job's start, expects the fast-load job's runs above plus
  ~40 s for closing the studio's models, which a studio usually has loaded: `base` (everything but the refine) plus
  `step` per refine step, 315 s + 150 s at 1080p, 405 s + 380 s at 2K and 625 s + 1685 s at 4K (`base`: each 16:9
  run's total less its two steps, plus 40 s; `step`: the mean of its two steps; rounded to 5 s). With the standard
  load 500 s more: its run above took 498 s more in all than the fast load's 2K run, 433 s of it the load and 56 s the
  encode (its VAE loaded through mmap; the bench's encodes too: 183-191 s, against 135-139 s with a fast load), and
  neither depends on the resolution. In all, with 2 steps (`faithful`, `balanced`) about 10 min at 1080p, 19 min at 2K
  and 1.1 h at 4K; with 3 (`full`) about 13 min, 26 min and 1.6 h. The popover shows the 2-step times with the
  studio's load (`/api/state` `upscale_load`; with the standard load ~19 min, ~28 min and ~1.2 h), the same for a
  9:16 clip and for every other 5 s clip (15 s more in the progress bar when its text is encoded first). These times
  come from the job's table, per length (`OFFER_ETA_S` in `upscale/regen2k_job.py`; `/api/state` `upscale_offers`
  carries it to the page, and `lf-cli`'s texts are checked against it): a longer clip gets its length's row
  (below). A run with the preread (`LFSTUDIO_UPSCALE_PREREAD=1`, the standard load only: 8 parallel readers fill the
  page cache first) loaded in 539 s, against 445 s without; the files were cached already, so this says little about
  a cold start.
- 4K: the stock block forward with only its FFN in 8 sequence chunks ran out of memory in the first step (the bench's
  guard stopped it at 3 GiB of MemAvailable). At 4K (and on longer clips from 10 s at 1080p and 6 s at 2K, below) the
  job runs every block with a memory-lean forward (everything but the attention over the sequence in 16 chunks, q/k/v
  straight into SDPA's layout; bit-identical to the stock forward on GB10, with 2.5x less activation memory; `forward`
  `lean` and `lean_chunks` 16 in the receipt). **The studio's 4K path has run end to
  end once**, with a 16:9 clip (the first table): 66 min from the job's start to the MP4, 56 min of it the two refine
  steps, with the most memory of the three (94.46 GiB allocated at the peak, MemAvailable down to 11.1 GiB in the first
  step; the studio's guard stops a job below 5 GiB). The bench's 4K run (the same forward, the weights loaded with
  `device_map` straight to the GPU, as the fast load does) took about 67 min after its load; its attention at 302,534
  rows gave the right result on GB10 (checked against fp32). 9:16 at 4K has not run.
- The studio needs 110 GiB free to start an upscale, 115 GiB at 4K (`LFSTUDIO_UPSCALE_MIN_FREE_GIB`, plus 5 GiB at
  4K; 117.7 GiB were free on the idle box), and kills it below 5 GiB for 3 s (`LFSTUDIO_UPSCALE_GUARD_GIB`). The job's
  16:9 runs above took ~101 GiB at 2K (MemAvailable 116.7 → 15.7 GiB) and ~105 GiB at 4K (116.4 → 11.1 GiB): a 4K
  upscale that started with just 110 GiB free would come down to ~4.7 GiB, below the guard. A longer clip needs what
  the memory model below gives where that is more (at 110 GiB only 2K at 15 s does: 115 GiB, 116 with Keep my frames,
  an estimate; the others 109 at most, 2K at 10 s, whose run came down from 116.5 to 16.7 GiB), and the job checks the
  floor again as it starts (job.json `min_free_gib`).
- Quality, so far: at NVIDIA's settings (start noise 0.909, 3 steps: `full`) the result is much sharper but has a
  different-looking face. Lower start noise (0.7 and 0.5 with 2 steps) and refining
  against the clip's own audio instead of NVIDIA's silence are being measured; the defaults (`balanced`, `clip`)
  are provisional until then.
- Not tried: NVIDIA's `sol_engine` forward (Sol-Attn sparse attention, MXFP8; NVIDIA lists ~150 GB for it) and its
  fast VAE decoder.

### Longer clips

The job takes a clip of any length the studio makes whole, in one pass of NVIDIA's refine, as it takes a 5 s one, at
1080p and 2K (every length) or 4K (5 s only). Its tables are in `upscale/regen2k_job.py`: `FRAME_CASES`,
`OFFER_ETA_S`, `UPSCALE_RAN` and the memory model (`MODEL_GIB`, `ROW_BYTES`, `OVERHEAD_GIB`, `need_gib`). **6, 10 and
15 s at 1080p and 10 s at 2K have run on the GPU** (below); 7, 8 and 12 s at 1080p and 6, 7 and 8 s at 2K have not,
but a longer clip at their size has, and their times and memory follow the job's model of these runs, which held
(steps within 1.4 %, the peak within 0.1 GiB). **2K at 12 and 15 s have not run**, nor has anything as long: their
times and memory are the model's, carried past every run, with a leaner forward of their own
([below](#2k-at-12-and-15-s-not-run-yet)).

**What each length gives the refine.** H3 makes 17n+5 frames (124 at 5 s), with 5n+2 latent frames (T) and 40 audio
latents a second (A). Sol-H3's clips, and LightX2V Turbo's without a last keyframe, keep the largest 8m+1 of them (121
at 5 s); the job adds NVIDIA's 3 copies of the last frame, as NVIDIA's encode does at 5 s, so the VAE gives the T of
the frames made. A LightX2V Turbo clip with a last keyframe keeps every frame made, and gets no copies. The refine's
sequence has T × 2,040 video rows at 1080p (3,600 at 2K, 8,160 at 4K), 2 × A audio rows and the prompt's rows (about
200; the table counts 200). The output has the clip's frames, and its whole sound.

| Length | Frames kept (made) | T | A | Rows at 1080p | Rows at 2K |
|---|---|---|---|---|---|
| 5 s | 121 or 124 (124) | 37 | 207 | 76,094 | 133,814 |
| 6 s | 153 or 158 (158) | 47 | 263 | 96,606 | 169,926 |
| 7 s | 169 or 175 (175) | 52 | 292 | 106,864 | 187,984 |
| 8 s | 185 or 192 (192) | 57 | 320 | 117,120 | 206,040 |
| 10 s | 241 or 243 (243) | 72 | 405 | 147,890 | 260,210 |
| 12 s | 289 or 294 (294) | 87 | 490 | 178,660 | 314,380 |
| 15 s | 361 or 362 (362) | 107 | 603 | 219,686 | 386,606 |

At 4K a 5 s clip has 302,534 rows. A clip of any other count of frames (Sol-H3's 3 and 4 s clips, made up to 0.5.0,
73 and 105; a 14 s research clip, 345) is refused, never cut.

**Measured** (a staging studio of this version on one DGX Spark, 2026-10-30, driven by `lf-cli`; one clip each, made
in the studios before, the 6 s one in that window; nothing else loaded, MemAvailable 116.3-116.6 GiB with the page cache
dropped at each job's start; the fast load, `balanced`, the clip's audio; the time from the job's start to the MP4
with the clip's sound, the encode of its text included; the receipt's rows, with the prompt's real ones). In brackets
the job's model at the run's own rows and start: the step model, the peak model, and the lowest MemAvailable by the
overhead before and after its refit to these runs (below):

| The clip | To | Frames, forward, rows | From the job's start to the MP4 | Refine, per step | Peak allocated | Lowest MemAvailable |
|---|---|---|---|---|---|---|
| Sol-H3 Text, 6 s 16:9, a static two-shot with three lines | 1080p | 153, stock, 96,632 | **759.7 s (12:40)** | 213.1, 211.8 s (215) | 84.78 GiB (84.76) | 23.5 GiB (22.3 / 23.5) |
| Sol-H3 Keyframes, 10 s 16:9, a static two-shot with three lines (its text encoded first, 14.8 s) | 1080p | 241, lean, 148,077 | **1423.0 s (23:43)** | 454.6, 450.5 s (455) | 81.15 GiB (81.13) | 29.0 GiB (24.4 / 29.2) |
| Sol-H3 Keyframes, 15 s 9:16, four lines (encoded first, 16.9 s) | 1080p (1080×1920) | 361, lean, 220,258 | **2587.0 s (43:07)** | 926.8, 930.3 s (937) | 87.37 GiB (87.33) | 20.4 GiB (17.9 / 20.7) |
| The 10 s Keyframes clip | 2K | 241, lean, 260,397 | **3188.1 s (53:08)** | 1261.2, 1260.8 s (1279) | 90.83 GiB (90.78) | 16.7 GiB (14.7 / 16.3) |
| LightX2V Turbo Keyframes, 576p 16:9 10 s, one picture first and last, stretched; both ends kept | 1080p | 243, lean, 152,050 | **1459.2 s (24:19)** | 474.4, 473.6 s (477) | 81.54 GiB (81.47) | 28.8 GiB (24.0 / 28.7) |

- **Every one came out whole:** the clip's frames at its size, its whole sound (every AAC packet), the last second
  clean (no spike in the frame-to-frame change over the last 24 frames, no drift from the clip and the detail kept to
  the end), and its lip sync (SyncNet LSE-C over the whole clip, at the face the clip syncs best, the offset kept):
  3.93 against the clip's 3.66 (6 s), 2.39 and 2.21 against 2.12 (10 s at 1080p and 2K), 2.56 against 2.24 (15 s) and
  3.03 against 3.09 (the LightX2V Turbo clip). The validation's lip sync gate was two-sided, so the 15 s upscale,
  0.32 above its clip, counted against it there; syncing better than the clip is not a failure. By eye the last seconds
  of the 10 and 15 s Sol-H3 clips' upscales at 1080p are sharper than the clip to the last frame, with the same faces,
  no drift or softening.
- **Times:** the steps came within 1.4 % of the step model (at their own rows), so it holds past 5 s, and the stages
  scale with the clip's frames: encoding the final clip 160 / 239 / 350 s at 6 / 10 / 15 s (245 s for the 243-frame
  clip), the VAE decode 135 / 210 / 314 s at 1080p (362 s at 2K for 10 s), the load 21.7-22.1 s. Each run less its two
  steps and its encode, plus ~40 s for closing the studio's models, gives the page's `base`: 375 s at 6 s, 543 s at
  10 s and 753 s at 15 s at 1080p, 691 s at 2K for 10 s (the estimates before the runs: 372, 535, 763 and 708 s).
- **Memory:** the peak model held within 0.1 GiB; the lowest MemAvailable came 1.2-4.9 GiB higher than the model said
  before the runs (its overhead was too high for the lean forward), and the model's overhead is now refit to them
  (below). The lowest of all was 16.7 GiB, at 2K for 10 s.
- **The 5 s routes, bit for bit.** In the same windows a 768p LightX2V Turbo clip with both ends kept (124 frames) and
  a 576p 9:16 one with its first frame kept (121, stretched) went to 1080p again, through this version's job: the
  receipts' tensors (the clip's latent, the keyframes' latents, the condition rows, the refined latent), timesteps,
  rows, sigmas and frames, and the MP4s' decoded frames and sound packets, are those of their runs of 2026-10-29/30
  (the table above), bit for bit, in 591.1 and 578.0 s (590.4 and 575.4 s).

**Time.** A refine step takes 6.457e-4 × S + 1.638e-8 × S² seconds for S rows: a fit to the 5 s steps of the bench
and the job (within 5 % of them), and within 1.4 % of the long clips' steps above. Most of a long step is the attention
(the S² term). The page's `base` + `step` per length (`OFFER_ETA_S`): measured where a run is (at 5 s the 16:9 runs
above, rounded to 5 s; the long clips' table above); elsewhere the step by the step model and the base by a straight
line in the frames H3 makes, fitted per size to the measured bases (1080p 86.1 s + 1.851 s a frame, within 8 s of its
four; 2K 107.0 s + 2.403 s a frame). With the fast load (the standard load adds ~8 min):

| Length | 1080p: base + step | 2 steps / 3 steps | 2K: base + step | 2 steps / 3 steps |
|---|---|---|---|---|
| 5 s | 315 + 150 s (measured) | 10 / 13 min | 405 + 380 s (measured) | 19 / 26 min |
| 6 s | 375 + 212 s (measured) | 13 / 17 min | 487 + 583 s | 28 / 37 min |
| 7 s | 410 + 256 s | 15 / 20 min | 528 + 700 s | 32 / 44 min |
| 8 s | 441 + 300 s | 17 / 22 min | 568 + 828 s | 37 / 51 min |
| 10 s | 543 + 453 s (measured) | 24 / 32 min | 691 + 1261 s (measured) | 54 / 75 min |
| 12 s | 630 + 638 s | 32 / 42 min | 814 + 1822 s (not run) | 74 / 105 min |
| 15 s | 753 + 929 s (measured) | 44 / 59 min | 977 + 2698 s (not run) | 106 / 151 min |

4K: 625 + 1685 s at 5 s (measured), 67 / 95 min; not offered for other lengths. 2K at 12 and 15 s are fitted as
the others are, past the rows any step has run at (314,380 and 386,606, against 302,534 at 4K for 5 s): the attention,
about 90 % of those steps, has not run on GB10 at that length, so they are estimates.

**Memory.** GPU memory allocated at the refine's peak = 68.4 GiB (the DiT, the upscaler and the video VAE kept for
the decode) + S × 181,800 bytes with the stock block forward, or S × 92,300 bytes with the lean one: fitted to the 5 s
runs (81.3 / 91.1 / 94.4 GiB at 1080p / 2K / 4K), within 0.1 GiB of every long clip's run; S × 70,796 bytes with
lean2, an estimate until it runs ([below](#2k-at-12-and-15-s-not-run-yet)). MemAvailable falls further
than that (the allocator's reserve and the process) by a straight line in the clip's own rows (video and audio), per
forward: stock 3.5 GiB + 5.0 GiB per 100,000 rows, lean 2.1 GiB + 2.8 GiB per 100,000 (lean2 the same line, an
estimate). Fitted to the runs' start less
their peak and lowest: stock 5.3 GiB at 45,124 rows (a 73-frame clip, an old 3 s one, run the same day), 6.2-8.1 at
75,894 (5 s at 1080p, with the runs through the studio of 2026-10-29 and that night above), 8.3 at 96,406 (6 s),
8.7-10.1 at 133,614 (5 s at 2K, the same); lean 6.0-6.4 at 147,690 (10 s at 1080p, with both ends kept too), 8.5 at
219,486 (15 s), 9.0 at 260,010 (10 s at 2K), 10.8 at 302,334 (5 s at 4K). From each run's own start of 2026-10-30 the
model's lowest MemAvailable is within 0.5 GiB of the measured one (`tests/test_upscale.py`), and from the earlier 5 s
runs' up to 0.8 GiB higher and up to 1.4 GiB lower. Before the refit it had a step at 80,000 and 140,000 rows (7, 9.5
and 11 GiB), too high for the lean forward by up to 4.9 GiB. The floor to start (`min_free_for`) is allocated + that
overhead + the guard's 5 GiB + 3 GiB of margin, and never below `LFSTUDIO_UPSCALE_MIN_FREE_GIB` (110 GiB; 115 at 4K).
Every 5 s clip keeps the floor it had, whatever the setting.

**The forward, by rows.** Where the clip's own rows (video and audio: not the prompt's or Keep my frames') pass
140,000, the job runs every block with the lean forward (16 chunks), as at 4K: at 1080p from 10 s, at 2K from 6 s.
Where they pass 302,334, the most of any case that has run (4K at 5 s), it runs lean2: at 2K for 12 and 15 s only.
Every case that has run keeps the forward it ran with on the GPU: stock at 1080p and 2K at 5 s (also with both ends
kept, about 141,000 rows at 2K), lean at 4K and on the long clips' runs (`tests/test_upscale.py` checks it). The
receipt has the refine's `rows` and its `forward`. The table gives the model's figures from 116.5 GiB free without
Keep my frames, the measured ones in brackets. With both ends kept the model's peak is up to 1.2 GiB higher and its
lowest MemAvailable as much lower: 18.2 GiB at 1080p for 8 s and 15.7 GiB at 2K for 10 s (9.2 GiB at 2K for 15 s, not
run), the lowest of each size past 5 s (14.0 GiB at 2K for 5 s, measured 14.1).

| Size and length | Forward | Peak allocated | Lowest MemAvailable (no ends kept) | Floor to start |
|---|---|---|---|---|
| 1080p 5 s (ran) | stock | 81.3 GiB (81.26-81.30) | 27.9 GiB (27.4-29.2) | 110 GiB |
| 1080p 6 s (ran) | stock | 84.8 GiB (84.78) | 23.4 GiB (23.5) | 110 GiB |
| 1080p 7 s | stock | 86.5 GiB | 21.2 GiB | 110 GiB |
| 1080p 8 s | stock | 88.2 GiB | 18.9 GiB | 110 GiB |
| 1080p 10 s (ran) | lean | 81.1 GiB (81.15) | 29.2 GiB (29.0) | 110 GiB |
| 1080p 12 s | lean | 83.8 GiB | 25.6 GiB | 110 GiB |
| 1080p 15 s (ran, 9:16) | lean | 87.3 GiB (87.37) | 21.0 GiB (20.4) | 110 GiB |
| 2K 5 s (ran) | stock | 91.1 GiB (91.06-91.09) | 15.3 GiB (15.7-16.8) | 110 GiB |
| 2K 6 s | lean | 83.0 GiB | 26.6 GiB | 110 GiB |
| 2K 7 s | lean | 84.6 GiB | 24.6 GiB | 110 GiB |
| 2K 8 s | lean | 86.1 GiB | 22.5 GiB | 110 GiB |
| 2K 10 s (ran) | lean | 90.8 GiB (90.83) | 16.4 GiB (16.7) | 110 GiB |
| 2K 12 s (not run) | lean2 | 89.1 GiB | 16.5 GiB | 110 GiB |
| 2K 15 s (not run) | lean2 | 93.9 GiB | 9.7 GiB | 115 GiB (116 with Keep my frames) |
| 4K 5 s (ran) | lean | 94.4 GiB (94.46) | 11.5 GiB (11.1) | 115 GiB |

Up to 12 s the most a clip needs by the model is 109 GiB (2K at 10 and 12 s, with Keep my frames or without), so each
starts at the default 110 GiB, as a 5 s clip does (before the refit: 111 GiB at 2K for 10 s with Keep my frames). 2K at
15 s needs 115 GiB, 116 with an end or both kept: its own floor, above the default. The long runs above started at
116.3-116.6 GiB free (117.7 on the idle box), which leaves it 0.3-1.6 GiB; a studio with less free refuses it with the
reason, before anything starts.

**4K past 5 s** is not offered: past the memory measured here, it takes the clip in pieces (windows of the refine),
which comes later.

**Keep my frames on longer clips.** Each end adds 2,040 rows at 1080p and 3,600 at 2K, which cost more in a longer
clip's steps. It has run once on a long clip, with the lean forward: both ends of the LightX2V Turbo 10 s clip at 1080p
(the table above). Its steps took 474 s, where the step model gives 454 s for the same clip without them, so both ends
added about 45 s with the pictures' encode (6.1 s): an estimate, as that clip was not upscaled without them. By the
step model both ends add about 2.2 min of steps at 2K for 10 s (26 s and 74 s at 5 s, where the runs above measured
30-34 s and 83 s in all). At 1080p the page adds the 5 s seconds per end (`UP_PIN_S`), so its time for a long clip
with Keep my frames is short by about 13 s for 10 s; at 2K the cascade's time counts the rows at each length (below).
The popover says that Keep my frames has not run where its route has not run at that length and size.

**Keep my frames' held ends at 10 s** (the run above, measured on the CPU from its MP4, its clip's and the pictures,
fitted as the receipt says: the first stretched, the last cover-cropped): both ends held, and as close to the pictures
as the 5 s pinned runs' best, with no flash at either end.

| | Frame 0 | Frame 242 (the last) |
|---|---|---|
| The upscale against the picture, luma PSNR at 1080p (SSIM) | 31.89 dB (0.832) | 32.01 dB (0.831) |
| The clip against the picture, at 1080p | 36.06 dB | 35.66 dB |
| The upscale against the picture, at 384p (the job's check grid) | 32.88 dB | 33.00 dB |
| The clip against the picture, at 384p (the job's own check: 36.69 and 36.97) | 37.75 dB | 37.16 dB |
| The face (a 192 px box at 1080p): upscale / clip | 30.39 / 31.87 dB | 29.59 / 31.59 dB |

- The 5 s pinned runs' ends came 27.4-31.9 dB from their pictures at 1080p (28.8-33.0 dB at 384p), so the 10 s ends
  are at the top of that range.
- The upscale gives up 3.6-4.2 dB to its clip at 1080p (4.2-4.9 dB at 384p), as the 5 s run built the same way (one
  picture at both ends, stretched and cover-cropped, a still ending) gave up 4.9-5.1 dB at 384p; the other 5 s runs
  gave up less. This clip's ends match the picture better than any of theirs. Every upscale comes out 2-4 luma levels
  darker than its picture (this one the least, -2.0 and -1.9), with no shift in place (the low-pass PSNR is about
  35.5-36 dB on every 16:9 run), and adds fine detail of its own.
- No flash: the PSNR to the picture is flat into frame 0 (31.894, 31.876 and 31.863 dB at frames 0-2) and rises
  smoothly into frame 242 (31.871, 31.967 and 32.014 at frames 240-242), and the change from frame to frame stays
  within the 5 s runs' rules at both ends. The clip almost freezes on its last frames, where the upscale keeps a faint
  shimmer, below its own average a few frames earlier, as a 5 s run did.
- Not measured: how much the pin gains at 10 s, as this clip was not upscaled without it (at 5 s the same clips
  without it were 3.2-5.3 dB further from the pictures at 1080p). The comparison across runs also mixes different clips
  and pictures.

**What has run where.** `UPSCALE_RAN` in the job lists the lengths and sizes upscaled on the GPU through the studio
(5 s at 1080p, 2K and 4K; 6, 10 and 15 s at 1080p and 10 s at 2K) and those a longer run at their size covers (7, 8
and 12 s at 1080p; 6, 7 and 8 s at 2K), one line each: every length and size offered but 2K at 12 and 15 s, where
the page shows "Clips of this length have not been upscaled on the GPU through the studio yet: the times are
estimates". A run that lands adds its line; a run that fails takes its line out, and the page then shows that note
there too. Keep my frames has its own lines (the server's `PIN_ROUTES` `"ran"`: a size alone at 5 s, `"1080p/10"` for
both ends of a LightX2V Turbo 10 s clip; no 2K line since the cascade replaced the single pinned refine that ran at
2K), and its popover note where its route has not run at the clip's length and size.

#### 2K at 12 and 15 s: not run yet

By the lean forward's model these would not fit, or only past anything measured: 2K at 12 s would allocate 95.4 GiB
and come down to about 10 GiB free, and 2K at 15 s would allocate 101.6 GiB and need 123 GiB free to start. So the job
runs lean2 there (`install_lean2`), a 21,504-byte-a-row saving at the peak (6.3 GiB at 12 s, 7.7 GiB at 15 s):

- diffusers' MiniMax-H3 transformer projects the video rows with `proj_in`, which is fp32 in the checkpoint, keeps
  that fp32 output (5,376 × 4 bytes a row) in a local through its whole block loop, and puts a bf16 copy of it into the
  sequence. In the lean forward's itemized peak (92,288 bytes a row at 4K, against 92,278 measured) those embeddings
  are 21,504 bytes a video row.
- lean2 casts `proj_in`'s output to bf16 as it comes out, the cast diffusers' forward does itself, and empties it once
  the first block starts (the sequence has it; nothing reads it again). The cast alone would still keep a bf16 copy,
  10,752 bytes a row. No file of diffusers or NVIDIA changes: a hook on `proj_in` and one on the first block.
- On the CPU it gives the lean forward's result **bit for bit** (`tests/test_upscale.py` `Lean2Tests`: a tiny
  random-weight MiniMax-H3 transformer of diffusers' own class, loaded as the job loads the real one, through
  NVIDIA's `Pipeline.refine` and the job's pinned refine with both ends; the stock blocks with lean2's embeddings
  give the stock result bit for bit too).
- Its peak (`ROW_BYTES` `"lean2"`, 70,796 bytes a row) and overhead (lean's line) are estimates until it runs: 89.1
  and 93.9 GiB at 12 and 15 s; the lowest MemAvailable from 116.5 GiB free about 16.5 and 9.7 GiB (9.2 with both
  ends at 15 s), against the guard's 5 GiB. At 15 s that is below the lowest of any clean run
  (11.1 GiB, at 4K for 5 s).
- **What has never run on GB10:** attention over 314,380 and 386,606 rows (cuDNN's SDPA; the longest so far is
  302,534 rows, at 4K for 5 s, where q, k and v had 2.17e9 elements each, past 2³¹; 2.77e9 at 2K for 15 s), the
  memory model past 302,334 rows of a clip's own, and the refiner on a 12 or 15 s clip at 2K (NVIDIA's page shows a
  15 s clip regenerated at 2K in one run; its released code is 5 s only).
- **The decode** stays NVIDIA's at every length. With the DiT freed it takes about 3 × the fp32 pixels of the clip
  (its `.float()`, `*std+mean`, `clamp` and `*255` over the whole clip) above the ~5 GiB left: 35.0 GiB allocated at 2K
  for 10 s (3.0 ×, its lowest MemAvailable then 75.2 GiB), 30.3 GiB at 1080p for 15 s, 13.6 GiB at 5 s. That gives
  about 41 GiB at 2K for 12 s and 50 GiB for 15 s, far under the refine's peak, so no streamed decode is needed there
  (by the same rule 4K at 10 s would need about 73 GiB).

The page shows its note on these ("Clips of this length have not been upscaled on the GPU through the studio yet: the
times are estimates"): their offers' `"ran"` is false until a run lands. A 2K 15 s run that holds the model would
bring 12 s in by the offer rule, a longer clip at the same size having run.

### Keep my frames at 2K: the cascade

**Why.** A pinned refine at 2K holds the picture far less than one at 1080p. In research runs (2026-10-28, a research
wrapper on this job, LightX2V Turbo 576p clips with a first frame; luma PSNR of frame 0 against the picture at a
288 px short side) the pin brought frame 0 from 25.5 to 29.6 dB at 1080p (+4.1), but only from 25.2 to 26.4 and from
27.1 to 28.5 dB at 2K (+1.2, +1.4); through the studio, both ends of a clip without a face came 4.7-5.0 dB closer in
one pinned refine at 2K (above), so how much it falls short varies by clip. Yet the 1080p pinned frame 0 alone,
resized to 2K (LANCZOS), was closer to the picture at full size (27.85 dB) than the unpinned 2K (23.98) and the pinned
2K (24.97), and the band above 1080p held almost none of the 2K output's error. So the picture can be made at 1080p,
where the pin holds, and a short pass at 2K only has to add detail.

**The route** (`upscale/regen2k_job.py` `route_for`, `run`; the receipt's `"route": "cascade"` and its `"cascade"`).
Keep my frames at 2K with an end held runs:

1. the pinned refine at 1080p, exactly as a 1080p upscale with Keep my frames runs it: NVIDIA's pipeline built at
   1080p with the strength's steps (Balanced: 2 from start noise 0.7), NVIDIA's upscale from the clip's 768p latent,
   the pictures at the 1080p grid, the upscale's seed;
2. NVIDIA's latent upscaler from that refined 1080p latent to the 2K grid (`open_h3.upscaler.upscale`, the call its
   own 4K route makes from 2K), with the pipeline's target and kept rows set as NVIDIA's sets them at 2K;
3. one step at 2K from start noise 0.4 (NVIDIA's `shifted_grid(0.4, 1)`: [0.4, 0]), with the same pictures at the 2K
   grid (encoded on the same VAE) held as before, NVIDIA's noise under the same seed at the 2K shape;
4. the 2K decode and the MP4, as without the pin.

Each refine takes the block forward of its own rows (`forward_for`: at 6-8 s the 1080p refine is stock and the 2K
step lean, at 10 s both lean, at 12 and 15 s the 1080p refine lean and the 2K step `lean2`, as 2K without the box
there). With no end held (the job's check) the 2K upscale is NVIDIA's refine at 2K, as without
the box. 1080p (with Keep my frames or without), the unpinned 2K and 4K are as before; there is no request field: the
server sends the pin as at 1080p and the job takes the cascade by itself.

**Research runs** (2026-10-29, one DGX Spark: a research runner around this job, which took the 2K step from a saved
1080p pinned result of the same clip, made in a run of its own; 5 s LightX2V Turbo 576p clips with a first frame,
the fast load, `balanced` at 1080p, the clip's audio, seed 4242; scored against the same clip's unpinned 2K, luma
PSNR at the output size; fine detail: band energy above 0.4 Nyquist against the unpinned 2K's; flash: the step from
frame 0 to 1 and 5 against the unpinned clip's, at most 1.5; motion: the mean change over frames 1-16 against the
unpinned clip's):

| Arm | Frame 0 | Face | Region | Fine detail (frame / face / region) | Flash (1 / 5) | Motion |
|---|---|---|---|---|---|---|
| 9:16 clip, 1 step from 0.4 with the pin | +2.03 dB | +3.45 dB | +2.45 dB | ×2.48 / 2.69 / 2.88 | 0.72 / 0.72 | ×1.14 |
| 16:9 clip, the same | +2.13 dB | +4.83 dB | +3.09 dB | ×2.22 / 1.94 / 1.71 | 0.89 / 0.75 | ×1.24 |
| 9:16 clip, 2 steps from 0.5 with the pin | +1.84 dB | +3.48 dB | +2.20 dB | ×2.67 / 3.06 / 2.94 | 0.59 / 0.59 | ×1.12 |
| 9:16 clip, 1 step from 0.4, the 2K step without the pin | +1.42 dB | +1.98 dB | +1.60 dB | ×2.64 / 2.92 / 3.07 | 0.74 / 0.72 | ×1.14 |

- Against the single pinned refine at 2K of the same clips (research runs of 2026-10-28), the first two came 1.04 and
  0.90 dB closer at frame 0 and 2.30 and 2.50 dB in the region, the face about the same (+0.30, -0.17 dB).
- Short of the +3 dB at frame 0 the research set itself, but closer than either 2K route, sharper, with no flash and
  the motion kept: so it replaces the single pinned refine at 2K. Two steps from 0.5 gave no more, and the 2K step
  without the pin less, so the route is one step from 0.4 with the pin.
- Their 2K step took 412-416 s with one end (one 2K step, 5 s, stock), NVIDIA's upscaler from 1080p 5.4-5.6 s, the
  pictures' encode at 2K 11 s (with a VAE load of its own there), peak allocated 91.73-91.74 GiB (as the single
  pinned 2K's 91.74), lowest MemAvailable 10.7-15.3 GiB from 111-115 GiB at their start. Their 1080p half ran apart
  (the 16:9 clip's in 604.5 s, steps of 158 s, 81.66 GiB).
- In the product job it is one run: by the job's model (below) 1129 s with one end and 1169 s with both at 5 s,
  closing the studio's models included (18.8 and 19.5 min), against 1165 s for the unpinned 2K and 1249 s by the
  same model for the single pinned pass (measured 1202 s with both ends through the studio). The research's 13-15 min
  2K arms leave out the 1080p half's two steps.

**Not run yet:** the cascade through the studio, at any length (the page says so at 2K: `PIN_ROUTES` has no 2K line);
a product run of a research clip, to compare with its arm (the same recipe: the same seeds, sigmas and noise; it can
differ where the research's input was built by hand, and where its picture was stretched to the grid and the clip's
model cover-cropped it); a 9:16 clip with both ends, a Sol-H3 clip's first frame, and clips longer than 5 s.

**Time** (`cascade_eta_s`, `/api/state` `upscale_offers` `cascade_eta_s`; the job's model, not measured): 2K's base
(`OFFER_ETA_S`), two steps at 1080p and one at 2K of the clip's length (`OFFER_ETA_S`: measured, or the step model),
each with the pictures' condition rows by the step model, NVIDIA's upscaler from 1080p to 2K (5 s at 5 s, more per
latent frame) and each picture's encode at both grids (3 s at 1080p, 5 s at 2K):

| Length | One end | Both ends | 2K without Keep my frames |
|---|---|---|---|
| 5 s | 1129 s (19 min) | 1169 s (19 min) | 1165 s (19 min) |
| 6 s | 1547 s (26 min) | 1594 s (27 min) | 1653 s (28 min) |
| 7 s | 1797 s (30 min) | 1847 s (31 min) | 1928 s (32 min) |
| 8 s | 2057 s (34 min) | 2111 s (35 min) | 2224 s (37 min) |
| 10 s | 2931 s (49 min) | 2996 s (50 min) | 3213 s (54 min) |
| 12 s | 3998 s (67 min) | 4073 s (68 min) | 4458 s (74 min) |
| 15 s | 5636 s (94 min) | 5725 s (95 min) | 6373 s (106 min) |

At 12 and 15 s both rows rest on 2K's fitted step there, which has not run
([2K at 12 and 15 s](#2k-at-12-and-15-s-not-run-yet)).

**Memory.** The peak is the larger refine's, the 2K step's (the same rows as the single pinned refine at 2K); the
latents kept between the refines are small (the refined 1080p latent: 29 MB at 5 s). The floor to start
(`route_need_gib`) is the larger of the two refines' needs: at 6-8 s the 1080p refine (the stock forward) needs more
than the 2K step (lean): 102-107 GiB against 99-103; at 12 and 15 s the 2K step (`lean2`) needs more: 109 and 116 GiB
against 100 and 104. Every length but 15 s starts at the default 110 GiB; at 15 s it needs 116, as 2K with Keep my
frames needs there in one pass (an estimate until `lean2` runs).

## Experimental LightX2V Turbo

Measured on one DGX Spark (GB10) on 2026-10-28/29: the studio's own job, `jobs/h3turbo_job.py`, run by hand with the
studio stopped, one clip per process as the studio runs it (its imports and the fast load in every run), 4 steps with
LightX2V's 4-step v1.3 LoRA, the Qwen conditioning at the 384p draft canvas unless marked. Most prompt conditionings
came from `bench/h3turbo-encode.sh`, the prompt encode the studio runs; the 10 and 14 s runs and two 5 s runs took a
studio clip's own conditioning. 15 runs, MemAvailable 116.2-116.9 GiB at each start. These are the job's own times;
in the studio a clip also waits for the Sol-H3 models to close when they are loaded (6.8 and 7.0 s through the
studio's queue, below; up to 40 s when a worker doesn't answer), and has its poster made. The last three rows are a
later session on the same machine (2026-10-29, after the staging run of the studio's queue below): one Text run each, by
hand in the same way, on the conditioning of a 576p Text clip of that staging studio (no keyframe, so it holds for every
size and length).

| Clip | Rows | Step (mean) | Decode | The job's process | Lowest MemAvailable | Peak allocated |
|---|---|---|---|---|---|---|
| 576p 5 s t2va, 16:9 and 9:16 (2 runs) | 21.7-21.9k | 25.9-26.8 s | 47.1-47.2 s | 172.6-176.5 s | 33.0-34.8 GiB | 77.0 GiB |
| 576p 5 s, a first frame (3 runs, one at canvas tokens) | 22.8-23.1k | 28.0-28.5 s | 46.8-49.3 s | 181.4-189.7 s | 33.7-35.7 GiB | 77.1-77.2 GiB |
| 576p 5 s, a last frame alone | 22.7k | 27.1 s | 47.1 s | 178.3 s | 36.2 GiB | 77.1 GiB |
| 576p 5 s, first and last (4 runs) | 23.5k | 28.5 s | 46.8-47.1 s | 184.2-184.7 s | 33.7-35.1 GiB | 77.3 GiB |
| 768p 5 s t2va | 37.9k | 54.6 s | 87.3 s | 327.5 s | 32.2 GiB | 79.7 GiB |
| 768p 5 s, a first frame at canvas tokens | 40.2k | 59.9 s | 87.0 s | 350.3 s | 32.1 GiB | 80.1 GiB |
| 576p 10 s, a first frame | 43.5k | 67.0 s | 95.2 s | 386.5 s (6.4 min) | 31.5 GiB | 80.6 GiB |
| 576p 14 s, a first frame | 61.2k | 112.0 s | 134.4 s | 605.8 s (10.1 min) | 27.4 GiB | 83.6 GiB |
| 768p 10 s, a first frame | 75.1k | 153.6 s | 173.7 s | 812.5 s (13.5 min) | 24.5 GiB | 86.0 GiB |
| 576p 15 s t2va (from 115.8 GiB) | 63.0k | 117.4 s | 139.7 s | 631.9 s (10.5 min) | 27.3 GiB | 83.9 GiB |
| 768p 12 s t2va (from 116.7 GiB) | 88.8k | 199.2 s | 215.7 s | 1035.7 s (17.3 min) | 22.5 GiB | 88.3 GiB |
| 768p 15 s t2va (from 112.0 GiB; other CPU work ran alongside) | 109.2k | 284.1 s | 263.1 s | 1425.0 s (23.8 min) | 14.4 GiB | 91.7 GiB |

- **Rows** are the packed sequence the transformer attends over: the conditioning, the keyframes' rows (576 each at
  576p, 1008 at 768p), the audio and the video latents (576 or 1008 per latent frame: 37 at 5 s, 72 at 10 s, 102 at
  14 s, 107 at 15 s). A step grows with them a little faster than linearly: 0.848 N + 0.016 N² seconds for N thousand
  rows fits every run of the first 15 within 1 s, and the three longer ones within 1.2 % (0.5-2.4 s). The decode took
  0.64-0.71 s per frame and megapixel.
- **The process around the steps and the decode** took 21.8-26.4 s per clip (22.6-25.5 s in the three longer runs):
  torch and Diffusers imported (4.2-4.5 s the first time in each environment, 1.6-1.9 s after), the fast load
  17.6-19.4 s for 18 files, the LoRA's 312 hooks 0.26-0.36 s, the latents saved, the MP4 written (0.4-1.4 s). The
  prompt encode, a transient Qwen worker of its own, took 9.1-17.8 s (ready after 6.5-11.8 s, then 1-5 s per
  encode).
- **Memory.** The model takes ~72 GiB once loaded; MemAvailable came down 80-98 GiB at its lowest, in the steps, and
  linearly with the rows: start − (77.8 + 0.186 GiB per thousand rows), within 1.6 GiB of every run (the three longer
  runs, past the 75.1k rows the line was fitted on, came 0.05-1.0 GiB under it: 88.5, 94.3 and 97.6 GiB). The decode
  runs after the transformer is dropped. The studio needs 100 GiB free to start a clip
  (`LFSTUDIO_H3TURBO_MIN_FREE_GIB`). The job's `min_free_for` takes the same line, plus the 1.6 GiB spread, as the
  least a clip of a size and length needs to stay above the job's own 6 GiB stop (89.8 GiB at 576p 5 s, 99.6 at 768p
  10 s, at the rows of two keyframes and a long prompt); the setting is a minimum on top of it. A 768p clip of 12 s or
  more needs more than 100 GiB by that line, so it needs 103 GiB at 12 s and 107 at 15 s. Started at those floors, the
  use measured in the runs (taken to the floor's rows) would leave 11.1 GiB at 576p 15 s, 8.2 at 768p 12 s, 8.8 at
  768p 15 s and 7.8 at 768p 10 s, above the job's own 6 GiB stop and the studio's 5 GiB guard.
- **The same as the probe it was ported from:** given the probe's conditioning, image and seed, the job's latents
  and MP4 were identical to the probe's, bit for bit, also in the studio's own environment; and the prompt encode's
  conditioning was identical, byte for byte, to the studio's own encode of the same clip, for a Keyframes and a Text
  clip.
- **Keyframes.** Frame 0 matched the first frame at 29.1-35.6 dB PSNR (against the frame as it was fitted to the
  canvas). The Qwen conditioning at the clip's canvas (576 or 1008 image tokens per keyframe instead of 252) gave the
  same frame 0 as the draft canvas's runs of the same clips (29.56 against 29.60 dB at 576p; 35.6 against 35.6 at
  768p, against the research probe's run), with steps 1-3 % longer than the research probe's runs of the same clips
  (as their extra rows predict), so the draft canvas stays the default (`LFSTUDIO_H3TURBO_QWEN_CANVAS`). The prompt
  guide's first + last line and the page's gave the same results (within 0.2 dB); a first frame alone and a last frame
  alone ran with the guide's lines for them, which the page now writes too. A last frame alone and both frames
  worked; the model anchors the last frame to the last frame it makes (frame 123 of 124 at 5 s: 30.9-34.3 dB against
  the last image), and the studio's usual length, 121 frames, ended 3 frames before it (18.4-26.4 dB). So a clip with
  a last frame keeps every frame made now ([API.md](API.md#models-experimental-lightx2v-turbo)). Through the studio's
  queue (below) such clips, a last frame alone and first + last, came out at 124 frames, their MP4's last frame the
  frame the last keyframe is anchored to (30.9 and 31.0 dB against it).
- **Lengths.** 10, 12, 14 and 15 s ran (576p at 10, 14 and 15 s, 768p at 10, 12 and 15 s). In the first session the
  job refused 15 s up front: the studio's 15 s is 362 frames, 15.08 s, past Diffusers 0.40's 15.0 s `max_duration`; 14
  s (345 frames) is the longest that Diffusers takes as it is. The job now raises that limit for a 15 s clip alone, as
  Sol-H3 draws the same 362 frames with the same transformer (stock Diffusers refusing them is checked on the CPU with
  a small random-weight model); the later 15 s runs, at 576p and 768p, made their 362 frames that way and kept the
  studio's 361 (289 at 768p 12 s). One run each.
- **Not measured:** 6, 7 and 8 s, 12 s at 576p, Keyframes clips at 768p past 10 s and at 576p past 14 s, and the
  mirrored end of the sound at 6 and 15 s (8 ms). The studio's estimates (`eta_s`) come from the runs above: the
  prompt encode (12 s), the process (23 s), 4 steps by the rows, the decode at 0.67 s per frame and megapixel, and 2 s
  for the poster: 200 s for 5 s at 576p and 350 s at 768p, 400 and 820 s for 10 s, 500 s for 12 s at 576p and 1070 s
  at 768p, 660 and 1440 s for 15 s; within 8 % of every run of the first session (the job's process, plus the measured
  or the estimated prompt encode, plus the poster's 2 s), and within 2.2 % of the three longer ones (660 s against
  646 s at 576p 15 s, 1070 against 1050 at 768p 12 s, 1440 against 1439 at 768p 15 s).
- **The studio's queue** (a staging studio on the same DGX Spark, 2026-10-29, with this version's LightX2V Turbo before
  the Upscale took its clips: a second studio with its own port and state, driven by `lf-cli`, one clip each). From the
  prompt encode to the poster: 191-204 s for 5 s at 576p (Text, a first frame, first + last, a last frame alone;
  estimate 200 s), 338 s at 768p (350 s), and 399 s for a 10 s clip with a first frame at 576p (400 s), queued after a
  Sol-H3 clip, whose models it closed first: 7.0 s from the clip's start to its prompt encode, the close and the
  memory check (`wall_s` less `e2e_s`; with nothing loaded the two are equal). Given the same prompt, keyframes and seed
  as the first session's runs by hand, each of these six clips' latents were identical to theirs, bit for bit, and so
  were the prompt encode's tensors (the conditioning files differ only in the keyframe paths they record). A Cancel
  during the prompt encode stopped the clip in 8 s, one at the job's first step in 2 s, with no process, Qwen container
  or GPU memory left behind; the refusals (Reference, and the Upscale of such a clip, which that version did not offer
  yet) came back with their reasons. In the Upscale's staging run later that day (this version), a 576p Text clip took
  188 s and a 768p clip with first and last keyframes 363 s (350 s), its 124 frames kept; the first closed a Sol-H3
  Reference session in 6.8 s.

## Replay bench: Stage 2 on saved drafts

`bench/duration.sh` and `bench/first-clip.sh` time whole sessions. `bench/replay.sh` replays one stage, Stage 2, on
fixed inputs: the Stage-1 drafts of clips the studio made. It answers two questions about a speed change: how much
faster, and whether the picture or the sound changed. Each change is an *arm*. `baseline` is today's recipe,
unchanged.

### How to run it

```bash
# 1. Copy drafts out of the studio: a clip's case folder is <runtime root>/outputs/webui/<session>/<case>.
#    Copies go to <runtime root>/outputs/bench/replay-drafts/. The trash would otherwise delete them.
bench/replay.sh add talk5 <case folder> --turns 'R:0-2.1'          # a dialogue draft: speaker:start-end in s
bench/replay.sh add quiet5 <case folder>
bench/replay.sh add scene10 <case folder> --turns 'R:0-1,L:4-6.3,R:6.3-7.5' --mouth 600:300:400:200
bench/replay.sh list
bench/replay.sh arms                                                  # what each arm changes

# 2. Plan and check without the GPU: processes, shapes, arms, runs, the runtime checks each arm relaxes, minutes.
bench/replay.sh run --dry --arms baseline:2,steps2_from_0725:1,decode_untiled:1,preview:1

# 3. Replay (GPU: stop the studio first). Then the CPU checks, which can run while the studio is back.
bench/replay.sh run --arms baseline:2,steps2_from_0725:1,decode_untiled:1,preview:1 --label steps
bench/replay.sh quality <run directory> [--lpips <python>] [--syncnet '<scorer command>']
```

`run` reads the same settings file as `bin/lf-studio`. Like `bench/duration.sh`, it refuses to start below
`LFSTUDIO_MIN_FREE_GIB` (110) and runs each GPU process under `bench/memguard-run.sh`. It copies the Stage-2 compile
cache into the run directory and uses the copy, so a run never writes the studio's cache. It deletes the copy at the
end (`--keep-cache` keeps it). It does not stop or start the studio and does not use sudo. Everything goes to
`<runtime root>/outputs/bench/replay-<time>[-label]/`: `plan.json`, `runs.jsonl` and `runs.csv` (one row per run),
`summary.md`, `logs/`, `gpu_clocks.csv`, and each run's MP4 and Stage-2 receipts under `runs/`.

`--arms` takes `ARM` or `ARM:N`, with N warm runs per draft (`--repeats`, default 2). Runs are grouped into one
process per format, length and output size: Inductor compiles one shape per process, as in the studio's Stage-2
worker. Each process loads Stage 2 once (16.4-16.8 s, measured 2026-10-12) and runs its list. The first run of each
process is an extra run, marked "first in process". `--stop-at EPOCH` starts no run that would end after that time,
by the estimates `--dry` prints; the first run of a process includes its compile (+6-7 s for a shape in the cache,
+69-80 s for a new one, measured 2026-10-12).

### Choosing drafts

Draft selection is up to you; nothing is hard-coded. A useful set:

- a 5 s t2va clip with on-screen dialogue (SyncNet needs a visible speaking face);
- a 5 s t2va clip without dialogue whose baseline already looks good (a draft that looks poor unchanged makes any
  change hard to judge);
- a 10 s fl2va dialogue clip with two people in one static shot (the scorer below expects that framing);
- a 9:16 clip and a ref2va clip, when you have them.

Clips that were warm runs in the studio (not the first clip after loading) can be checked bit for bit against
the replay. `add` records which kind each clip was.

### What each check measures

| Check | What it is | Where |
|---|---|---|
| Stage 2 s | Stage 2's own clock in its receipt (`stage2_request_s`): from the run's start to the finished MP4 | `runs.csv` |
| Phases | upscale, adapter, denoise, decode, H.264 mux, from the receipt's `phases_s` | `runs.csv` |
| Per-step s | CUDA-event time of each call of the refiner's transformer (one per denoise step) | `step_s` |
| Memory | torch peak allocated and reserved; lowest MemAvailable, sampled every 0.5 s during the run | `runs.csv` |
| Settings | each run's sigmas, steps, output size, decoder tiles and upscale factor; the runtime checks it relaxes | `runs.jsonl` |
| Arm took effect | the receipt shows the arm's step count, Sol-call count, output size, frames and decoder tiles, the refiner held the arm's sigmas, `profile_step` wrote its profile, and a keyframe arm's mode, VAE encodes and frame-0 record are in the receipt; otherwise the run fails | `status` |
| Patch | SHA-256 of the patch applied to the package (the copy `install/10-sana.sh` keeps) and of the Stage-2 code it runs | `runs.csv` |
| Output | SHA-256 of the MP4, and MD5 of its decoded video frames and of its decoded audio | `runs.csv` |
| Same as baseline | decoded video and audio equal to the same draft's first warm baseline run | `runs.csv` |
| Same as studio | baseline runs: the MP4 equal to the studio's clip of that draft, warm run against warm clip (n/a when the Stage-2 audio or keyframe mode differs from the clip's) | `runs.csv` |
| Repeats | whether the warm runs of each draft and arm are identical; the first run of each process against the warm one | `summary.md` |
| GPU clocks | SM clock, temperature, power and throttle reasons from `nvidia-smi` at 1 Hz | `summary.md` |
| SSIM, PSNR | ffmpeg, every frame, at the draft canvas (672×384 or 384×672): each arm against the baseline, against the `preview` arm when the run has it (how close it stays to the draft), the first run against the warm run (the noise floor), and the studio clip against the baseline | `quality/` |
| LPIPS | optional: the same pairs, AlexNet, every 2nd frame, CPU | `quality/` |
| Sheets | per draft: 3 frames per arm, the middle frame's centre at full size, a mouth strip with `--mouth`; side-by-side MP4s, baseline left, arm right, with the arm's audio | `quality/sheets/` |
| SyncNet | optional, dialogue drafts: LSE-C (higher is better) and LSE-D (lower is better) per face over that speaker's turns | `quality/syncnet/` |

SSIM, PSNR and LPIPS measure how much the picture changed, not its quality: a lower score can look as good or
better. The sheets and side-by-sides are for looking.

Stage 2 muxes the draft's own audio into every output, so every arm of a draft has the same audio (the report
checks this). For the same reason the speaking turns belong to the draft: `add --turns` stores them. A dialogue
draft without turns gets them once, detected by the scorer on the baseline output, and every arm is scored with
those. Not every scorer can detect them: the one used on 2026-10-12 expects two speakers and fails on a clip with
one. Give such drafts their turns with `add --turns`.

### Arms

Each arm is one named function in `bench/replay/arms.py`. On the GPU it changes the live Stage-2 session for one run
and is undone after it, so the next run starts from the runtime's own state. Nothing in the installed package is
edited.

| Arm | Change | Runtime checks it relaxes |
|---|---|---|
| `baseline` | none | none |
| `steps2_from_0725`, `steps2_from_0625` | 2-step refine (sigmas 0.725/0.421875/0, or LTX-2's 0.625/0.4/0) | the attention-call count, checked for 2 steps instead of 3 |
| `preview` | no refine: the draft through the upscaler, the adapter and the LTX decoder | denoise skipped; the attention-call count, checked for 0 steps |
| `decode_untiled`, `decode_long768` | decoder tiles: one per frame, or the short side untiled and the long side 768/64 | none (the tiles pass the decoder's own validation) |
| `profile_step` | the baseline with torch.profiler around denoise step 1 (timings perturbed) | none |
| `canvas480`, `canvas480_untiled`, `canvas480_steps2`, `preview480` | refine at 832×480 with the H3 upscaler at ×1.25 (`preview480`: no refine) | the 2×-draft output rule, the upscaler's ×2, the refiner modules' 1344×768 size; the attention-call count for `canvas480_steps2` (2 steps); for `preview480` the denoise (skipped) and the attention-call count (0 steps) |
| `canvas384`, `preview384` | refine at 672×384, the draft's own size, without the upscaler (`preview384`: no refine) | the same three, with the upscaler skipped; for `preview384` the denoise (skipped) and the attention-call count (0 steps) |
| `keyframe_off`, `keyframe_pin`, `keyframe_draft`, `keyframe_pin_steps2` | `SOL_H3_STAGE2_KEYFRAME` for the run ([PATCH.md](PATCH.md#19-stage-2-keyframe-pin)): no pin (as before 0.5.2), the first frame held at latent frame 0 (the default since 0.5.2), or the draft's own frame 0; `keyframe_pin_steps2` with the 2-step refine. Only fl2va drafts with a first frame are pinned; the others run as `off` | none; for `keyframe_pin_steps2` the attention-call count (2 steps) |

All other runtime checks stay in force: the capture and payload checks, the shape and dtype checks between the
upscaler, the adapter and the refiner, the Sol backend check, the writer's frame-count check and the output check.
`--dry` prints the exact list per arm.

Arms are runtime patches in the bench, not hooks in the patch. That keeps the studio's runtime free of bench-only
code, and the bench works with any installed 0.3 or later patch without reinstalling. Such patches could break
silently when the package changes, so the bench checks three times. Before loading: its worker settings against
the studio's (`runtime/pipeline.py`), and that the runtime modules still have the functions it replaces and the
recipe's sigmas. Right after loading (Stage 2 loads in ~17 s): that the refiner has every attribute and method an
arm changes. After each run: that the receipt and the refiner show the arm's settings (see "Arm took effect").
A deeper change, such as an FP8 Stage 2, may need a hook in the patch.

The keyframe arms are the exception: they set the patch's own keyframe mode on the session, so they need a patch with
`SOL_H3_STAGE2_KEYFRAME` (checked before loading). Stage 2 pins only in an fl2va session, as in the studio, and a
process loads Stage 2 for its first draft's task. So with a keyframe arm, runs are grouped by task, one process each.
Without one, the bench's own `SOL_H3_STAGE2_KEYFRAME` applies to every run, and fl2va drafts get processes of their own
unless it is `off` (also when it is unset: the default is `pin` since 0.5.2); `--dry` says so. Otherwise whether an
fl2va draft is pinned would depend on which draft its process starts with. `keyframe_pin` builds the image encoder once
in an fl2va process started with `SOL_H3_STAGE2_KEYFRAME=off` or `draft` and keeps it; with the default (`pin`) the
encoder is loaded from the start, as in a studio session. `add` copies an fl2va clip's first and last frame with its
draft and points the copy's `capture.json` at them, so a pin still finds them once the studio's trash is emptied (a
draft added before this names the studio's own file).

### Reference numbers

Measured 2026-10-12 on one DGX Spark with the bench this one was ported from (the same Stage-2 session code and
the same arms under other names), on one 5 s t2va draft without dialogue and four 10 s fl2va dialogue drafts.
Medians of warm runs.

| | 5 s, 1344×768 | 10 s, 1344×768 |
|---|---|---|
| `baseline` Stage 2 | 35.6 s (denoise 24.6, decode 7.2) | 75.4 s (denoise 51.1, decode 17.3) |
| `steps2_from_0725` | 27.5 s | 58.9 s |
| `decode_untiled` | 34.5 s | 72.8 s (one draft) |
| `canvas480` / `canvas480_steps2` | 14.4 s / 11.2 s | 29.2 s / 23.2 s |
| `preview` | 11.0 s | 24.3 s (one draft) |

- Warm repeats were within 1.5% of each other (one arm 5%) and bit-identical (decoded-frame MD5).
- The first run in each process differed from the warm runs: PSNR 40.2 dB and SSIM 0.985 at 5 s, at 672×384. So
  bit-exact checks compare warm runs, and that difference is the noise floor. First-run kernel autotuning is a
  likely cause (inferred, not measured).
- The first run of a new output size compiles: +69-80 s. At 1344×768 with the copied cache: +6-7 s.
- The SM clock median was 2385-2463 MHz per process group, with no thermal throttling in a 37-minute window.

The draft without dialogue in that run was judged poor even unchanged, which is why the list above asks for one
that looks good. Some arms change the content: in that run the 3-step refine at 832×480 changed objects and faces,
and at 672×384 objects. They are for measurement, not presets.

### Optional quality tools (CPU)

- **LPIPS:** a Python with the CPU build of torch, numpy and `lpips` (`pip install lpips`, BSD-2-Clause). Its first
  use downloads torchvision's AlexNet weights. Pass it with `--lpips <python>`.
- **SyncNet:** not included. The original SyncNet code (MIT) is at
  [joonson/syncnet_python](https://github.com/joonson/syncnet_python); its `download_model.sh` fetches the SyncNet
  and face-detector weights. Pass a scorer command with `--syncnet`. It must accept
  `VIDEO... --json OUT [--turns 'R:0-1.2,L:2.4-4.7']` and write, per video, `file`, `turns` and `faces` with
  `det_rate` and `speaking` = `offset`, `lse_d`, `lse_c` (the format is in `bench/replay/syncnet.py`). The table is
  complete when every dialogue draft and arm has a score; `quality` exits with 1 when it is not.

Not done yet: Stage-1 arms (they need a scratch runtime root and compile cache for Stage 1, see
`runtime/config.py` `compile_cache_root`).
