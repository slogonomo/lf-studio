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

Warm clips (models loaded): 3 s 36 s · 5 s 54-66 s · 10 s 123 s · 15 s 191 s. NVIDIA lists 56 s for a warm
5 s 16:9 t2va clip; on this box the warm t2va mean was 65.9 s for 16:9 and 56.6 s for 9:16. 5 s 16:9 is NVIDIA's
route; 9:16 and 3-15 s are this repo's extensions ([PATCH.md](PATCH.md)).

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
- Lowest MemAvailable: 15-18 GiB for a 5 s first clip (upstream flow: 13 GiB); 12 GiB for 15 s.
- Read-ahead into the page cache made the cold start slower (177 → 195 s): 54 GB of Stage 2 read-ahead competed
  with Stage 1's compile for disk and memory.

## Reliability: prompt-length recompiles

Stage 1's transformer is compiled with static shapes, so each new prompt length (in text tokens) compiles new
graphs. FastVideo caps recompiles at 16; the 4th distinct prompt length in a session hit the limit, killed the
worker and forced a full reload. The limit is raised (1,024; 65,536 accumulated). Tested: 7 distinct prompt
lengths in one session, no failure.

## Decode and encode at the end of Stage 2

Stage 2 ends with the VAE decode and the H.264/AAC encode of the MP4. In the Stage-2 receipts of warm t2va 16:9
clips (studio and bench runs, medians of 46, 12 and 3 clips) the decoder took 7.1 s at 5 s, 17.4 s at 10 s and
25.4 s at 15 s. The writer added 1.50 s, 2.76 s and 2.13 s on top (`h264_encode_mux_s`).

That time is not the x264 encode. In a lane run on 2026-10-13, libx264's thread was busy 0.85 s per 10 s clip
(3-3.5 ms per frame) and waited 12 s for frames. Most of the writer's time is about 0.18 s per 16-frame view on the
decoder's thread: LTX-2's colour conversion on the GPU and the copy of the 25 MB I420 view to the CPU. The figure
holds in 232 earlier receipts, at every request index. Which part dominates was not measured. Writing 25 MB into
fresh pages costs 9.3 ms on this CPU (1.4 ms into pages in use), so the CPU side alone does not explain it.

Letting the decoder run one chunk ahead of the encoder ([PATCH.md](PATCH.md#16-decode-ahead-of-the-h264-encoder))
gave the same MP4 but saved no time (+0.3 s at 10 s, -0.3 s at 15 s, within noise), because it only reordered that
work.

## Known costs, not fixed

- A new prompt length on a warm session adds 9-10 s (Stage 1 17.6 → 26.2 s). Reusing a prompt, or editing it
  without changing its token count, avoids it. The failing guards are in FastVideo's H3 sparse-attention metadata
  and tiling (`total_seq_length`), so a fix needs that dimension made dynamic.
- Cold vs warm overhead (~55 s), measured before the last fix: Stage 2 first compile 15-20 s, Stage 1 LoRA merge
  + FP8 ~7 s, Stage 1 first-denoise compile ~11 s, audio VAE 6 s.
- The numbers above assume the compile cache is warm from an earlier run of the same mode, format and length.
  The first run of a configuration fills it: t2va 9:16 5 s 111 s, fl2va 9:16 114 s, ref2va 9:16 176 s,
  t2va 16:9 10 s 264 s, t2va 16:9 15 s 357 s.

## Switches

| Variable | Default | Effect |
|---|---|---|
| `SPARK_FAST_SAFETENSORS` | 1 | 0 = upstream safetensors loading |
| `SPARK_DIRECT_IO` / `SPARK_DIRECT_AHEAD` | 1 / 1 | O_DIRECT reads / read-ahead of the next window |
| `SOL_H3_BSA_CHECK_CALLS` | 8 | 200 = upstream's full attention self-check |
| `SOL_H3_COMPILE_CACHE` | `$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` | `session` = upstream per-session cache |
| `SPARK_AUTOTUNE_CACHE_FIX` | 1 | 0 = upstream Inductor behaviour |
| `SOL_H3_VAE_ON_DEVICE` | 1 | 0 = build VAEs on the CPU as upstream |
| `SOL_H3_AUDIO_VAE_PREFETCH` | 1 | 0 = load the audio VAE when first needed |
| `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` | 241 | longer clips load the prompt encoder per request |
| `SOL_H3_DECODE_AHEAD` | 1 | 0 = the VAE decoder waits for the H.264 encoder (LTX-2's order) |
| `SOLH3_PRELOAD` (web UI) | 1 | 0 = no warm-up of the last settings at launch |

The warm-up at launch (`SOLH3_PRELOAD`) is a cold start like any other: 102-112 s for t2va 16:9 5 s with a filled
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
loads its own settings right after the close.
