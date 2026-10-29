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

Measured on one DGX Spark (GB10) on 2026-10-28, 5 s t2va clips, NVIDIA's standard forward (BF16, dense attention
through cuDNN, about 90 TFLOPS) and its released VAE, the LoRA merged once in place (bit-identical to NVIDIA's merge,
checked). The numbers come from three kinds of runs, kept apart here: the studio's own job (`upscale/regen2k_job.py`)
in a staging studio with nothing else loaded, with the fast load (its default) and, earlier, with the standard load;
the bench wrapper the job was made from (the same NVIDIA code and changes, the studio stopped), with the standard load;
and that bench wrapper with the fast load. Each has one run per resolution and orientation it was measured at.

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
upscaler the same scale factors. The page and the client show the 16:9 times for a 9:16 clip. **9:16 at 4K has not
run**; there the upscaler goes through the 2K grid (turned), as for 16:9.

**The job itself, standard load** (earlier that day, the same staging studio and 16:9 clip, an earlier version of the
job too; 2K, `balanced`, audio `clip`):

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
  9:16 clip. A run with the preread (`LFSTUDIO_UPSCALE_PREREAD=1`, the standard load only: 8 parallel readers fill the
  page cache first) loaded in 539 s, against 445 s without; the files were cached already, so this says little about
  a cold start.
- 4K: the stock block forward with only its FFN in 8 sequence chunks ran out of memory in the first step (the bench's
  guard stopped it at 3 GiB of MemAvailable). At 4K the job runs every block with a memory-lean forward (everything but
  the attention over the sequence in 16 chunks, q/k/v straight into SDPA's layout; bit-identical to the stock forward
  on GB10, with 2.5x less activation memory; `lean_chunks` 16 in the receipt). **The studio's 4K path has run end to
  end once**, with a 16:9 clip (the first table): 66 min from the job's start to the MP4, 56 min of it the two refine
  steps, with the most memory of the three (94.46 GiB allocated at the peak, MemAvailable down to 11.1 GiB in the first
  step; the studio's guard stops a job below 5 GiB). The bench's 4K run (the same forward, the weights loaded with
  `device_map` straight to the GPU, as the fast load does) took about 67 min after its load; its attention at 302,534
  rows gave the right result on GB10 (checked against fp32). 9:16 at 4K has not run.
- The studio needs 110 GiB free to start an upscale, 115 GiB at 4K (`LFSTUDIO_UPSCALE_MIN_FREE_GIB`, plus 5 GiB at
  4K; 117.7 GiB were free on the idle box), and kills it below 5 GiB for 3 s (`LFSTUDIO_UPSCALE_GUARD_GIB`). The job's
  16:9 runs above took ~101 GiB at 2K (MemAvailable 116.7 → 15.7 GiB) and ~105 GiB at 4K (116.4 → 11.1 GiB): a 4K
  upscale that started with just 110 GiB free would come down to ~4.7 GiB, below the guard.
- Quality, so far: at NVIDIA's settings (start noise 0.909, 3 steps: `full`) the result is much sharper but has a
  different-looking face. Lower start noise (0.7 and 0.5 with 2 steps) and refining
  against the clip's own audio instead of NVIDIA's silence are being measured; the defaults (`balanced`, `clip`)
  are provisional until then.
- Not tried: NVIDIA's `sol_engine` forward (Sol-Attn sparse attention, MXFP8; NVIDIA lists ~150 GB for it) and its
  fast VAE decoder.

## Experimental LightX2V Turbo

Measured on one DGX Spark (GB10) on 2026-10-28/29: the studio's own job, `jobs/h3turbo_job.py`, run by hand with the
studio stopped, one clip per process as the studio runs it (its imports and the fast load in every run), 4 steps with
LightX2V's 4-step v1.3 LoRA, the Qwen conditioning at the 384p draft canvas unless marked. Most prompt conditionings
came from `bench/h3turbo-encode.sh`, the prompt encode the studio runs; the 10 and 14 s runs and two 5 s runs took a
studio clip's own conditioning. 15 runs, MemAvailable 116.2-116.9 GiB at each start. These are the job's own times;
in the studio a clip also waits for the Sol-H3 models to close when they are loaded (about 40 s), and has its poster
made.

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

- **Rows** are the packed sequence the transformer attends over: the conditioning, the keyframes' rows (576 each at
  576p, 1008 at 768p), the audio and the video latents (576 or 1008 per latent frame: 37 at 5 s, 72 at 10 s, 102 at
  14 s). A step grows with them a little faster than linearly: 0.848 N + 0.016 N² seconds for N thousand rows fits
  every run within 1 s.
- **The process around the steps and the decode** took 21.8-26.4 s per clip: torch and Diffusers imported (4.2-4.5 s the
  first time in each environment, 1.6-1.7 s after), the fast load 17.6-18.1 s for 18 files, the LoRA's 312 hooks
  0.26-0.36 s, the latents saved, the MP4 written (0.4-1.0 s). The prompt encode, a transient Qwen worker of its own,
  took 9.1-17.8 s (ready after 6.5-11.8 s, then 1-5 s per encode).
- **Memory.** The model takes ~72 GiB once loaded; MemAvailable came down 80-92 GiB at its lowest, in the steps, and
  linearly with the rows: start − (77.8 + 0.186 GiB per thousand rows), within 1.6 GiB of every run. The decode runs
  after the transformer is dropped. The studio needs 100 GiB free to start a clip (`LFSTUDIO_H3TURBO_MIN_FREE_GIB`;
  768p 10 s, the most, would come down to 7.8 GiB from there, above the job's own 6 GiB stop and the studio's 5 GiB
  guard). The job's `min_free_for` takes the same line, plus the 1.6 GiB spread, as the least a clip of a size and
  length needs (89.8 GiB at 576p 5 s, 99.6 at 768p 10 s); the setting is a minimum on top of it. By that line a 768p
  clip of 12 s or more would need more than 100 GiB (103 at 12 s, 107 at 15 s): an extrapolation past the largest run
  (75.1k rows; 91.7k and 112.1k), which the studio does not offer yet.
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
  a last frame keeps every frame made now ([API.md](API.md#models-experimental-lightx2v-turbo)); that MP4 (124 frames
  at 5 s, its sound cut to match) has so far come only from the CPU test with a small random-weight model, not from a
  GPU run.
- **Lengths.** 10 and 14 s ran (576p and 768p at 10 s, 576p at 14 s). 15 s did not run: the job refused it up front
  (the studio's 15 s is 362 frames, 15.08 s, past Diffusers 0.40's 15.0 s `max_duration`); 14 s (345 frames) is the
  longest that Diffusers takes as it is. The job now raises that limit for a 15 s clip alone, as Sol-H3 draws the same
  362 frames with the same transformer; that it works, and that stock Diffusers refuses those frames, is checked on the
  CPU with a small random-weight model only. The studio does not offer 15 s, nor 768p 12 s, until each has run on the
  GPU.
- **Not measured:** 6, 7, 8 and 12 s, 768p beyond 10 s, 15 s at either size, the mirrored end of the sound at 6 and
  15 s (8 ms), and the studio's queue around the job. The studio's estimates (`eta_s`) come from the
  runs above: the prompt encode (12 s), the process (23 s), 4 steps by the rows, the decode at 0.67 s per frame and
  megapixel, and 2 s for the poster: 200 s for 5 s at 576p and 350 s at 768p, 400 and 820 s for 10 s, 500 s for 12 s
  at 576p (by the same model, 660 and 1440 s for 15 s, not offered); within 8 % of every run above.

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
| Arm took effect | the receipt shows the arm's step count, Sol-call count, output size, frames and decoder tiles, the refiner held the arm's sigmas, and `profile_step` wrote its profile; otherwise the run fails | `status` |
| Patch | SHA-256 of the patch applied to the package (the copy `install/10-sana.sh` keeps) and of the Stage-2 code it runs | `runs.csv` |
| Output | SHA-256 of the MP4, and MD5 of its decoded video frames and of its decoded audio | `runs.csv` |
| Same as baseline | decoded video and audio equal to the same draft's first warm baseline run | `runs.csv` |
| Same as studio | baseline runs: the MP4 equal to the studio's clip of that draft, warm run against warm clip | `runs.csv` |
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
