# The patch: what it changes in Sol-H3-Spark

`patches/sol-h3-studio.patch` changes NVIDIA's Sol-H3-Spark package in the
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark) repository,
branch `sol-engine`. It is a plain `git diff`: 23 files, 18 modified and 5 new.

- Base: `6c2f582ba9681dfe352ae4ca23b6f0030cf120cf` (`patches/BASE_COMMIT`). Applies cleanly; 93 CPU tests pass
  (2 skipped).
- Also checked on `sol-engine` HEAD `670482d8a857d578ac8a2ea89b052d0fb47badba` (2026-09-29): applies cleanly,
  69 CPU tests passed with the patch of that date. The upstream files the patch touches are identical in both commits.
- 17 of the 18 modified files are in `models/minimax_h3/Sol-H3-Spark`. One is a shared Sana module:
  `models/minimax_h3/super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py`.
- Every modified file carries a `Modified by sol-h3-studio` comment at its first change. Other changed lines are
  tagged `sol-h3-studio:`. The 5 new files say they are part of sol-h3-studio.

## Apply, test, revert

```bash
git clone --filter=blob:none -b sol-engine https://github.com/NVlabs/Sana
cd Sana
git checkout --detach "$(cat /path/to/sol-h3-studio/patches/BASE_COMMIT)"
git apply --check /path/to/sol-h3-studio/patches/sol-h3-studio.patch
git apply /path/to/sol-h3-studio/patches/sol-h3-studio.patch

# CPU tests (no GPU, no weights), with the Stage1 python
cd models/minimax_h3/Sol-H3-Spark
CUDA_VISIBLE_DEVICES= ~/sol-h3/envs/stage1/bin/python -B -m unittest discover -s tests # Ran 93 tests ... OK (skipped=2)

# undo the whole patch
git -C ../../.. apply -R /path/to/sol-h3-studio/patches/sol-h3-studio.patch
```

`install/10-sana.sh` does the clone, the apply and the tests. Most features below also have an environment
switch that restores NVIDIA's behaviour without removing the patch. Set switches in the settings file
(`~/.config/sol-h3-studio/env`) or the environment of whatever starts the pipeline. The pipeline passes the
environment to its Stage1 and Stage2 workers. `SOL_H3_ORIENTATION` and `SOL_H3_SECONDS` are only defaults for
`infer.py`: the web server removes both from its environment at start, because it sets format and length per job.

## Deviations from NVIDIA's validated recipe

NVIDIA validated one route: 5 s, 1344×768 (16:9), with the checks described in Sol-H3-Spark's
`docs/validation.md`. The patch goes beyond that in these places:

| Deviation | Default | How to stay on NVIDIA's route |
|---|---|---|
| Portrait 768×1344 output. Not quality-validated: the distilled LoRAs, the H3 upscaler and the H3-to-LTX adapter were trained and validated on landscape. | off | don't select 9:16 |
| Clip lengths other than 5 s (3-15 s). NVIDIA validated 5 s only. | off | keep 5 s |
| 3 s clips are below the 4-15 s that MiniMax lists for H3. | off | use 4 s or more |
| The first-request cuDNN BSA self-check compares 8 calls with the Triton kernel instead of all 200. | on | `SOL_H3_BSA_CHECK_CALLS=200` |
| Orientation and clip length are applied after the frozen-recipe hash check. `configs/default.json` is still checked against `FROZEN_RECIPE_SHA256` first, then portrait and length are derived from it, the same way upstream derives its Ref2VA options. `results.json` records the derived recipe. | off | landscape, 5 s |
| Model loading overlaps (see below), so stages load at the same time instead of one after another. | on | no switch; `git apply -R` |
| Reference videos (ref2va) are prepared at the draft canvas, not H3's native 768p canvas. NVIDIA lists video references as GPU-unvalidated. | on | no switch; `git apply -R` |
| Stage 2 refines the video against the H3 audio held clean and fixed, instead of a re-noised copy that it regenerates and then discards (NVIDIA's `STAGE2_CONTRACT.md` steps 4 and 7). See [15](#15-stage-2-audio-held-fixed-lip-sync). | on | `SOL_H3_STAGE2_AUDIO=joint` |

Everything here was run on one DGX Spark. It was not run on other hardware.

## Changes by feature

Numbers are from one DGX Spark (GB10, driver 580.178, CUDA 13.0); see [PERFORMANCE.md](PERFORMANCE.md) for how
they were measured. "First clip" = launch to the first finished MP4 of a new session, t2va 16:9 5 s.

### 1. Portrait 9:16

- **Purpose.** 768×1344 output. The canvas, latents, upscaler target and decoder tiles are transposed. The Stage2
  token count is unchanged (16,128), so speed and memory are expected to match landscape.
- **Files.** `infer.py`, `runtime/config.py`, `runtime/geometry.py` (new), `runtime/qwen.py`,
  `runtime/qwen_ops/media.py`, `runtime/stage1.py`, `runtime/stage1_ops/capture.py`, `runtime/stage2.py`,
  `runtime/stage2_ops/{models,h3_upscale,writer_chunking}.py`, `runtime/latent_transfer.py`, `tests/test_geometry.py`.
- **Switch.** `infer.py --orientation portrait`, `Pipeline(..., orientation="portrait")` or
  `SOL_H3_ORIENTATION=portrait` (an `infer.py` default; the web server ignores it). Default landscape.
- **Measured.** Warm t2va means on the same box: 56.6 s for 9:16 (2 clips) and 65.9 s for 16:9 (3 clips, with
  different prompts).
- **Deviation.** Not quality-validated (see the table above).

### 2. Clip length 3-15 s

- **Purpose.** Lengths other than the frozen 5 s. The draft length is 24 fps × seconds, snapped up to H3's
  17k+5 frame grid. The output is the largest 8m+1 frame count that fits in the draft, at most 65,536 Stage2 tokens.

  | Selected | Draft frames | Output frames | Output length | Prompt encoder |
  |---|---|---|---|---|
  | 3 s | 73 | 73 | 3.04 s | resident |
  | 4 s | 107 | 105 | 4.38 s | resident |
  | 5 s (frozen route) | 124 | 121 | 5.04 s | resident |
  | 6 s | 158 | 153 | 6.38 s | resident |
  | 7 s | 175 | 169 | 7.04 s | resident |
  | 8 s | 192 | 185 | 7.71 s | resident |
  | 10 s | 243 | 241 | 10.04 s | resident |
  | 12 s | 294 | 289 | 12.04 s | loaded per request |
  | 15 s | 362 | 361 | 15.04 s | loaded per request |

- **Files.** `runtime/config.py`, `runtime/geometry.py`, `runtime/latent_transfer.py` (payload cap 4 → 32 MiB;
  a 15 s latent plus audio is ~9 MB), `runtime/stage1.py` (FastVideo's 5.0-15.0 s duration bounds widened to
  3.0-15.1 s; 15.1 admits the 362-frame grid point), `runtime/stage1_ops/capture.py`, `runtime/stage2.py`,
  `runtime/stage2_ops/{models,writer_chunking}.py`, and the shared
  `super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py`, whose frame-profile check allowed only
  124→121 and 243→241.
- **Switch.** `infer.py --seconds N`, `Pipeline(..., seconds=N)` or `SOL_H3_SECONDS=N` (an `infer.py` default; the
  web server removes it from its environment, since a 5 s job is sent as "no length" = the frozen route). Unset =
  frozen 5 s route.
- **Measured.** Warm clips: 3 s 35.8 s, 5 s 54-66 s, 10 s 123.2 s, 15 s 190.8 s.
- **Deviation.** Not validated by NVIDIA; 3 s is below MiniMax's stated 4-15 s.

### 3. Prompt encoder per request for clips over 10 s

- **Purpose.** With the Qwen prompt encoder resident next to Stage1 and Stage2, a 15 s refine dropped to 4 GiB
  MemAvailable and the memory guard stopped it. Sessions whose output exceeds 241 frames load the encoder for
  each request and close it again.
- **Files.** `runtime/pipeline.py`.
- **Switch.** `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` (default 241).
- **Measured.** A 15 s session with the encoder resident: 192.9 s per clip vs 190.8 s, lowest MemAvailable
  7.3 GiB. No gain, so the default stays.

### 4. Overlapped loading and deliver-first

- **Purpose.** Upstream loads by rendering a throwaway warm-up clip (seed 999), then renders the first request,
  loading each stage after the previous one. Now:
  - `Pipeline.start(case, deliver=True)` makes the loading pass render the real first clip and returns its row
    (marked `includes_model_load`). `infer.py --deliver-first` uses it; the web UI always does.
  - The warm-up prompt encoder runs while Stage1 starts. Stage2 starts once Stage1 logs
    `Loaded deferred module transformer`, after Stage1's BF16 load and FP8 conversion, its memory peak.
  - The resident prompt encoder loads in the background after the first clip.
  - If Stage1 dies before that point, Stage2 is not started, and a failed load joins the half-started workers
    so `close()` can stop them.
- **Files.** `runtime/pipeline.py`, `infer.py`, `tests/test_pipeline.py`.
- **Switch.** `deliver` is opt-in (`infer.py` without `--deliver-first` keeps the separate warm-up). The overlap
  itself has no switch.
- **Measured.** First clip 282 → 177 s, lowest MemAvailable 18 GiB. The first overlap attempt fell to 2.4 GiB
  MemAvailable and was stopped by the memory guard; the version in the patch waits for the Stage1 marker.

### 5. O_DIRECT weight loading (`runtime/fastload.py`)

- **Purpose.** On GB10, loading safetensors through mmap page faults ran at ~1.3-2 GB/s with the default
  128 KiB read-ahead. The same NVMe streams 11.8 GB/s with large O_DIRECT reads. A host-to-GPU copy straight from
  a file-backed mmap is also slow (~1 GiB/s vs ~18 GiB/s from anonymous memory, warm cache).
  `fastload` reads each file into 256 MiB aligned windows with 4 parallel `preadv` calls, copies tensors to the GPU
  from there, and reads the next window ahead while the current one is consumed. It falls back to mmap per file.
  Pages read this way stay out of the page cache, which on GB10 competes with GPU memory.
  Stage1 routes FastVideo's `safe_open`, its LoRA `load_file` and its component loader through it; Stage2 routes
  `safetensors.safe_open`.
- **Files.** `runtime/fastload.py` (new), `runtime/stage1.py`, `runtime/stage2.py`.
- **Switches.** `SPARK_FAST_SAFETENSORS=0` (upstream loading), `SPARK_DIRECT_IO=0` (no O_DIRECT, mmap + copy),
  `SPARK_DIRECT_AHEAD=0` (no read-ahead of the next window).
- **Measured.** Together with item 6: first clip 177 → 127 s; H3 shards 33 → 15 s; LTX transformer (42 GB)
  ~40 → 11 s. O_DIRECT for the 5.3 GB LoRA: 124 → 110 s. Read-ahead: LTX 42 GB to GPU 10.7 → 8.3 s, LoRA
  1.9 → 1.35 s (first clip within noise). All 4,349 LTX and 3,320 LoRA tensors were compared bit for bit with
  safetensors.
- **Removed.** A page-cache read-ahead of whole files was tried and made the first clip slower (177 → 195 s).
  The patch does not contain it.

### 6. Shorter first-request BSA self-check

- **Purpose.** On the first request upstream compares all 200 cuDNN block-sparse attention calls with the
  original Triton kernel. Each comparison adds a reference pass and two host syncs, ~30 s per cold start. The
  patch checks the first 8 calls (layers 0-7 of the first update, the same kernel, real data).
- **Files.** `runtime/stage1_ops/vsa.py`, `tests/test_stage1_contracts.py`.
- **Switch.** `SOL_H3_BSA_CHECK_CALLS` (default 8, range 1-200; 200 = upstream).
- **Measured.** Part of 177 → 127 s (item 5).
- **Deviation.** Fewer calls are validated (see the table).

### 7. Persistent compile caches

- **Purpose.** Upstream compiles Stage1 into the default Inductor cache (under `/tmp`, gone after a reboot) and
  Stage2 into a per-session directory, so every cold start recompiled, including Stage2's max-autotune (Stage2
  warm-up 121 → 57 s with the cache). The caches now persist, and Triton keeps its autotune results on disk
  (`TRITON_CACHE_AUTOTUNING=1` for both workers).
- **Files.** `runtime/config.py` (`compile_cache_root`), `runtime/pipeline.py`, `runtime/stage2.py`,
  `tests/test_pipeline.py`.
- **Switch.** `SOL_H3_COMPILE_CACHE`: a directory, or `session` for upstream's behaviour. Default
  `$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile`, or `~/.cache/sol-h3-studio/compile` if the runtime root is unset.
  Stage1 uses `<root>/stage1/{inductor,triton}`, Stage2 `<root>/stage2/{inductor,triton}`.
- **Measured.** Startup 304 → 227 s (this was in place before the 282 s baseline). Triton autotune results
  on disk: 127 → 124 s.

### 8. Inductor autotune-cache fix (torch 2.13)

- **Purpose.** Kernels loaded from the FX graph cache are pickled `CachingAutotuner`s. For 11 of 61 Stage2
  kernels the pickled results were already pruned to one config, and torch 2.13's
  `CachingAutotuner.recheck_autotune_cache` hashed that one-config list instead of the candidate list the result
  was saved under. The lookup always missed, so every new process re-ran coordinate-descent tuning (3-4 s) and
  saved the same result again. After an upstream miss the patch reads the entry through the kernel's own
  `AutotuneCache` (same key, same configs hash that the save used) and then does what upstream does on a hit.
- **Files.** `runtime/stage2.py` (monkeypatch, installed in the Stage2 worker).
- **Switches.** `SPARK_AUTOTUNE_CACHE_FIX=0` disables it; `SPARK_DEBUG_AUTOTUNE=1` prints each fallback hit.
- **Measured.** 11/11 hits, no re-tuning; Stage2 first render 45-49 → 41 s; first clip 103-104 → 99.3/99.4 s.
  Specific to torch 2.13's Inductor; re-check it when the Stage2 torch changes.

### 9. VAEs built on the GPU

- **Purpose.** FastVideo's VAE loader built the 10 GB H3 video VAE on the CPU (random init of every parameter
  with `OMP_NUM_THREADS=1`), moved it to the GPU, then loaded the real weights. `VAELoader` and
  `AudioDecoderLoader` now run under `torch.device("cuda")` and read files with O_DIRECT. Every weight is still
  loaded strictly from the checkpoint.
- **Files.** `runtime/stage1.py`.
- **Switch.** `SOL_H3_VAE_ON_DEVICE=0` restores CPU construction.
- **Measured.** Video VAE 70 → 2.9 s. First clip, with items 4-8 already in on both sides: fl2va 194 → 115 s,
  ref2va (image + video reference) 254 → 178 s.

### 10. mimalloc purge for the workers

- **Purpose.** PyTorch's aarch64 wheels allocate CPU tensors with a bundled mimalloc that by default keeps freed
  memory while the process idles. On unified memory that memory is GPU memory too. The pipeline sets
  `MIMALLOC_PURGE_DELAY=0` for its workers unless you set it yourself.
- **Files.** `runtime/pipeline.py`.
- **Switch.** Set `MIMALLOC_PURGE_DELAY` to another value to override.
- **Measured.** RSS after freeing 3 GB of CPU tensors 3.55 → 0.47 GiB; fl2va lowest MemAvailable 9.3 → 19.8 GiB.

### 11. Audio VAE loaded during the first denoise

- **Purpose.** The small audio VAE loaded lazily after the denoise, on the critical path. It now loads on a side
  thread once the Stage1 transformer is resident (loading both at once slowed the transformer ~4 s), and audio
  decoding waits for that thread.
- **Files.** `runtime/stage1.py`.
- **Switch.** `SOL_H3_AUDIO_VAE_PREFETCH=0`.
- **Measured.** Audio decode in the first clip 5.8 → 0.4 s; first clip 106-110 s (within run-to-run noise).

### 12. Reference videos at the draft canvas (ref2va)

- **Purpose.** Sol-H3-Spark prepared reference *images* at the 672×384 draft budget but resized reference
  *videos* to H3's native 768p canvas, which made the context and the draft much larger. Shrinking the upload
  did not help, because it was scaled up again. Videos are now prepared at the draft canvas (short side 384, area at most 672×384) in both the Qwen worker and Stage1.
- **Files.** `runtime/qwen_ops/media.py`, `runtime/qwen.py`, `runtime/stage1.py`, `tests/test_qwen_media.py`.
- **Switch.** None.
- **Measured.** ref2va with a 5 s reference video: context 6,714 → 2,178 tokens, draft denoise 228 → 54 s,
  per clip 400 → 133 s, lowest MemAvailable 6 → 19 GiB. Output looked comparable in one side-by-side comparison.

### 13. Recompile limit

- **Purpose.** The H3 draft DiT is compiled with static shapes, so each new prompt length (in text tokens)
  compiles new graphs. FastVideo caps Dynamo recompiles at 16; in a long-running session the 4th distinct prompt
  length failed with "Forward execution thread failed", the worker died and the next job reloaded everything.
  The limits are raised (`recompile_limit` 1,024, `accumulated_recompile_limit` 65,536).
- **Files.** `runtime/stage1.py`.
- **Switch.** None.
- **Measured.** 7 distinct prompt lengths in one session, no failure. Each new length still costs 9-10 s once
  (see [PERFORMANCE.md](PERFORMANCE.md), known costs).

### 14. Load-timing lines

The Stage2 worker prints `[sol-h3-studio] HH:MM:SS ...` lines while it builds its models (upscaler, transformer,
audio encoder, decoder). They go to that worker's `worker.log`. No switch.

### 15. Stage-2 audio held fixed (lip sync)

NVIDIA's Stage 2 encodes the H3 audio with the LTX-2.5 audio VAE and noises it to the first Stage-2 sigma (0.909),
like the video. It denoises audio and video together for three updates under the generic Stage-2 prompt. Then it
discards that audio and muxes the original H3 PCM (its `STAGE2_CONTRACT.md`, steps 4 and 7). The video attends to the
audio while it is refined, so the mouths are redrawn to the regenerated speech. About 9% of the H3 signal survives
the noising. The regenerated speech keeps the rhythm but not the sounds: Whisper hears it as English-sounding babble,
with a phoneme error rate at chance level against the heard track. It sometimes also adds speech in pauses. In the
MP4 the lips then look close to the dialogue but approximated, and they sometimes move while nobody speaks.

The patch holds the H3 audio clean and fixed during the refine, as LTX-2's own audio-to-video pipeline does
(`a2vid_two_stage.py`: `frozen=True, noise_scale=0.0`). `SparkRefiner.prepare_multimodal_states` first builds the
upstream states, so the random draws and the video noise are the same. It then replaces the audio state with its
clean latent, an all-zero denoise mask and `frozen=True`, so the transformer is told sigma 0 for the audio. A
`SOL_H3_STAGE2_AUDIO` value other than `frozen` or `joint` (any case) fails when the pipeline is created, before any
worker loads; the studio refuses to start with it. An installed LTX-2 without frozen states fails when Stage 2 starts. The Stage-2 receipts
(`stage2_result.json`) record the mode as `stage2_audio`. Stage 2 takes the same time.

Measured on 8 dialogue clips: two people, fl2va 10 s, 8 seeds, one scene. Each clip's Stage 2 was run both ways
from the same saved Stage-1 capture:
- the upstream run reproduced the studio's output exactly;
- the frozen run is what the patched studio renders, bit for bit when both are warm requests.

SyncNet confidence over each speaker's lines (LSE-C, higher is better; the draft was decoded for the first 4 clips
only):

| speaker | Stage-1 draft (4 clips) | upstream Stage 2 | frozen audio |
|---|---|---|---|
| man, facing the camera | 8.2 | 4.4 | 7.8 |
| woman, near profile | 6.5 | 3.5 | 5.2 |

Over 14 clips, the upstream finals' timing within a line wandered by 3-4 frames against about one in the drafts.
The silent mouthing seen in one clip is gone with frozen audio. The frames differ by about 3/255 on average, faces
and identity are unchanged, and the image is 5-11% sharper.

- **Files.** `runtime/stage2.py`, `runtime/config.py` (the setting), `runtime/pipeline.py` (checks it first),
  `tests/test_stage2_contract.py`, `tests/test_pipeline.py`.
- **Switch.** `SOL_H3_STAGE2_AUDIO=joint` restores upstream's behaviour (older clips then reproduce from their seed).

### 16. Decode ahead of the H.264 encoder

- **Purpose.** Stage 2 ends by streaming LTX-2's tiled VAE decoder into LTX-2's `encode_video` (libx264 `veryfast`,
  CRF 19, 8 threads). Issue #43 assumed that the GPU decoder and the CPU encoder took turns, for two reasons:
  - `encode_video` hands frames to its encoder thread through a one-slot queue;
  - the decoder yields chunk k only after it has decoded group k+1, although chunk k's frames are final after
    group k (group k+1 only adds into the overlap, and chunk k does not include it).

  This change lets the queue hold one decoder chunk (8 views, about 200 MB of I420 frames at 1344×768). It also
  makes the decoder yield each chunk before it decodes the next group. After the last decoded frame, only the last
  chunk (41 frames at 10 s, 57 at 15 s), the flush, the AAC track and the mux remain. Measured, the first reason did
  not hold: the encoder was faster than the colour conversion of each view, so it never held the decoder back, and
  the change saved no time (see "Measured").
- **Same file.** The same frames reach the same encoder, in the same order, with the same settings. The decoder
  runs on the same thread and CUDA stream, with the same tiles, decoder calls, random draws and blending.
  `runtime/stage2_ops/decode_ahead.py` wraps LTX-2 and copies none of its code:
  - for one `encode_video` call, LTX-2's encode module builds its queue with room for 8 views instead of 1. A
    clip of one temporal tile (5 s and shorter) keeps LTX-2's queue: its one chunk follows the whole decode anyway;
  - between two temporal groups the decoder gets an empty marker group that spans their overlap. LTX-2's own
    generator blends it like any group and so yields the previous chunk one group earlier. The marker's empty
    chunks are dropped. Adding the marker's zeros can only turn -0.0 into +0.0 in the overlap, and `to_rgb`
    (x + 1) removes that.

  Each hook first checks the source hashes of the LTX-2 code it relies on (LTX-2 `d151147`): `encode_video` and
  `_encode_chunks_threaded` for the queue; `ConvVideoDecoder.tiled_decode`, `decode_video`,
  `_accumulate_temporal_group_into_buffer` and `ltx_core.tiling.group_tiles_by_temporal_slice` for the decoder.
  With another LTX-2 version, Stage 2 keeps LTX-2's own order, and the receipt says so (`eager_decoder: false`,
  `encoder_queue.hooked: false` with the reason). Each hook swaps one module global (the encode module's `Queue`,
  the decoder module's `group_tiles_by_temporal_slice`) for its first use only, puts it back on that use and again
  at the end of the call, also after an error.
- **Receipts** (`stage2_result.json`). `decode_ahead`; `timing_notes`, which says what the writer's timings mean;
  `decode_done_to_mp4_s`, from the GPU finishing the last temporal group's decoder calls to the finished MP4. That
  moment comes from CUDA events recorded after each group, in both modes, so the number compares across them.
  `writer_chunking` adds:
  - `decode_done_monotonic_ns` (that moment), `frames_queued_monotonic_ns` (the decoder iterator ended: every frame
    converted and queued), and `cuda_peak_allocated_bytes` / `cuda_peak_reserved_bytes` of decode and encode;
  - `encoder_queue`, when hooked: `decoder_blocked_s` (the decoder waited for room for a frame), `end_marker_wait_s`
    (the wait to queue the end-of-stream marker, i.e. for the encoder to take the last views), `encoder_waited_s`
    (the encoder waited for frames), `encoder_busy_s` (the encoder thread's time on frames, without the flush) and
    `max_queued_views`.

  `h264_encode_mux_s` keeps its formula, decode-and-mux time minus decoder time. It is now the part of the encode
  that is not hidden behind the decode. The encode itself is expected to take as long as before (inferred;
  `encoder_busy_s` records it with decode-ahead on). `e2e_s` and `stage2_request_s` still end at the finished MP4.
- **Memory.** The queue holds at most one chunk of CPU frames, about 200 MB (inferred). On the GPU a marker buffer
  (one overlap, about 0.15 GB) is held while the next group decodes. Measured CUDA peak of decode + encode: +0.17 GiB
  at 10 s and +0.15 GiB at 15 s, none at 5 s. Fewer old chunks are held during the later decodes, but that did not
  lower the peak at 15 s.
- **Files.** `runtime/stage2_ops/decode_ahead.py` (new), `runtime/stage2.py`, `runtime/stage2_ops/writer_chunking.py`,
  `runtime/config.py`, `runtime/pipeline.py`, `tests/test_decode_ahead.py` (new).
- **Switch.** `SOL_H3_DECODE_AHEAD`: `1` (default) or `0`, which restores LTX-2's order and leaves `encode_video`'s
  module untouched (only the decoder's timing events remain). Any other value is refused when a pipeline is made,
  and by `bin/sol-h3-studio --check` and the server at start, as for `SOL_H3_STAGE2_AUDIO`.
- **Measured: the same MP4, no time saved.** One lane run of the patch on one DGX Spark (2026-10-13), the same
  warm t2va 16:9 clips with `SOL_H3_DECODE_AHEAD=0` and `1` (medians, 1 warm clip at 5 s, 2 at 10 and 15 s):

  | | 5 s off | 5 s on | 10 s off | 10 s on | 15 s off | 15 s on |
  |---|---|---|---|---|---|---|
  | `decode_stream_and_h264_mux_s` | 7.69 | 8.44 | 20.39 | 20.70 | 31.09 | 30.81 |
  | `official_video_vae_decode_s` | 7.10 | 6.95 | 17.25 | 17.72 | 26.69 | 26.36 |
  | `h264_encode_mux_s` | 0.58 | 1.49 | 3.14 | 2.98 | 4.40 | 4.45 |
  | `decode_done_to_mp4_s` | 0.60 | 1.51 | 1.99 | 0.68 | 2.06 | 0.97 |
  | CUDA peak of decode + encode (GiB) | 39.79 | 39.79 | 40.66 | 40.83 | 41.43 | 41.58 |

  - All warm MP4s of a length were the same file, off and on.
  - The time from the last decode to the MP4 fell, but decode and encode together took the same time, within run
    to run noise (+0.3 s at 10 s, -0.3 s at 15 s). Stage 2 and e2e moved with the denoise.
  - The 5 s off clip was an outlier: 232 earlier receipts give about 1.5 s of `h264_encode_mux_s` at 5 s.
  - The reason: libx264 was not what held the decoder back. Its thread was busy 0.85 s per 10 s clip (3-3.5 ms per
    frame) and waited 12 s for frames, and the queue never held more than one view. The writer's time outside the
    decoder is about 0.18 s per 16-frame view (in 232 earlier receipts too): LTX-2's colour conversion and the copy
    of the 25 MB view to the CPU. Those run on the decoder's thread, so decode-ahead only reordered them.
  - Lowest MemAvailable over a phase: 8.8 GiB off, 8.3 GiB on.
- **Tests.** `tests/test_decode_ahead.py` uses stand-ins, independent models of LTX-2's behaviour: a decoder
  generator, a module with `encode_video`'s one-slot queue to an encoder thread, and a tiled decoder with LTX-2's
  yield and blend rule. It checks:
  - the same frames in the same order, and that the decoder runs one chunk ahead and no further;
  - that errors from either side reach the caller and undo the hooks, and that both hooks are one-shot;
  - the same chunks (values, shapes, strides, random draws) one group earlier, with the clock alone too;
  - the version checks, including an encode module with another bounded queue, which is left alone;
  - the setting, the Stage-2 glue (off leaves the encode module alone; a single tile keeps LTX-2's queue).

  With `SOL_H3_TEST_LTX_ROOT` set and the Stage-2 Python, it also runs LTX-2's own code on the CPU, with its real
  hashes: the tiled decoder (equal chunks, complementary and weighted blending) and `encode_video` with PyAV and
  libx264 (byte-identical MP4).

### 17. Tests

`tests/test_geometry.py` (new) covers the derived portrait geometry. `tests/test_pipeline.py` adds
deliver-first, the background prompt encoder, the new `Pipeline` arguments and the compile-cache location (it
also points the cache at a temp directory, so running the tests writes nothing under `$HOME`).
`tests/test_qwen_media.py` and `tests/test_stage1_contracts.py` follow the new signatures and the configurable
BSA check. `tests/test_stage2_contract.py` adds the Stage-2 audio policy (frozen and joint states, the setting,
older LTX-2), `tests/test_pipeline.py` its check before any worker starts. `tests/test_decode_ahead.py` (new) covers
decoding ahead of the encoder ([16](#16-decode-ahead-of-the-h264-encoder)); 2 of its tests run only with
`SOL_H3_TEST_LTX_ROOT` set (and the Stage-2 Python). 93 tests in total, 2 skipped otherwise (upstream: 65).

## All switches

| Variable | Default | Effect |
|---|---|---|
| `SOL_H3_ORIENTATION` | landscape | `infer.py` default for `--orientation` (the web server ignores it) |
| `SOL_H3_SECONDS` | unset (5 s) | `infer.py` default for `--seconds` (the web server ignores it) |
| `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` | 241 | longer outputs load the prompt encoder per request |
| `SPARK_FAST_SAFETENSORS` | 1 | 0 = upstream safetensors loading |
| `SPARK_DIRECT_IO` | 1 | 0 = no O_DIRECT reads |
| `SPARK_DIRECT_AHEAD` | 1 | 0 = no read-ahead of the next O_DIRECT window |
| `SOL_H3_BSA_CHECK_CALLS` | 8 | 200 = upstream's full cuDNN BSA self-check |
| `SOL_H3_COMPILE_CACHE` | `$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` | a directory, or `session` = upstream per-session caches |
| `SPARK_AUTOTUNE_CACHE_FIX` | 1 | 0 = upstream Inductor behaviour |
| `SPARK_DEBUG_AUTOTUNE` | 0 | 1 = print autotune-cache fallback hits |
| `SOL_H3_VAE_ON_DEVICE` | 1 | 0 = build the VAEs on the CPU as upstream |
| `SOL_H3_AUDIO_VAE_PREFETCH` | 1 | 0 = load the audio VAE when first needed |
| `SOL_H3_STAGE2_AUDIO` | frozen | `joint` = upstream: Stage 2 re-noises and regenerates the audio it refines the video against |
| `SOL_H3_DECODE_AHEAD` | 1 | 0 = LTX-2's order: the decoder waits for the H.264 encoder ([16](#16-decode-ahead-of-the-h264-encoder)); other values are refused |
| `MIMALLOC_PURGE_DELAY` | 0 (workers) | any value you set wins |

The `SPARK_*` names are kept from the patch's first version so existing setups keep working.

## License of the changes

The files the patch modifies are Apache-2.0 NVlabs/Sana code; `runtime/qwen_ops/media.py` also contains
FastVideo-derived Apache-2.0 code (see Sol-H3-Spark's `THIRD_PARTY_NOTICES.md`). The changes and the new files are
Apache-2.0, like the rest of this repository.
