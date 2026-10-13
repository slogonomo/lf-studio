# The patch: what it changes in Sol-H3-Spark

`patches/sol-h3-studio.patch` changes NVIDIA's Sol-H3-Spark package in the
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark) repository,
branch `sol-engine`. It is a plain `git diff`: 23 files, 19 modified and 4 new.

- Base: `6c2f582ba9681dfe352ae4ca23b6f0030cf120cf` (`patches/BASE_COMMIT`). Applies cleanly; 98 CPU tests pass.
- Also checked on `sol-engine` HEAD `670482d8a857d578ac8a2ea89b052d0fb47badba` (2026-09-29): applies cleanly,
  69 CPU tests passed with the patch of that date. The upstream files the patch touches are identical in both commits.
- 18 of the 19 modified files are in `models/minimax_h3/Sol-H3-Spark`. One is a shared Sana module:
  `models/minimax_h3/super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py`.
- Every modified file carries a `Modified by sol-h3-studio` comment at its first change. Other changed lines are
  tagged `sol-h3-studio:`. The 4 new files say they are part of sol-h3-studio.

## Apply, test, revert

```bash
git clone --filter=blob:none -b sol-engine https://github.com/NVlabs/Sana
cd Sana
git checkout --detach "$(cat /path/to/sol-h3-studio/patches/BASE_COMMIT)"
git apply --check /path/to/sol-h3-studio/patches/sol-h3-studio.patch
git apply /path/to/sol-h3-studio/patches/sol-h3-studio.patch

# CPU tests (no GPU, no weights), with the Stage1 python
cd models/minimax_h3/Sol-H3-Spark
CUDA_VISIBLE_DEVICES= ~/sol-h3/envs/stage1/bin/python -B -m unittest discover -s tests   # Ran 98 tests ... OK

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
| A running session changes its clip length and orientation in place (reshape) instead of loading the models again. It gave the same Stage-1 latents as a fresh session at every shape checked, and the same MP4 at 5 s, 10 s and 9:16; at 15 s the MP4 differed through Stage 2's run-time kernel tuning (see [16](#16-reshape-another-length-or-orientation-without-a-reload)). | on (studio) | `SOLH3_RESHAPE=0` (studio setting) |
| Stage 1's regions and Stage 2's blocks compile with `torch.compile(dynamic=False)`: every graph is static, also after a shape change. This changes Stage 1's graphs from the first request on: upstream compiles the regions for updates 2-4 with a dynamic size even then (inferred from the code). The output was the same with and without it in one GPU check (3 clips). See [16](#16-reshape-another-length-or-orientation-without-a-reload). | on | `SOL_H3_STATIC_SHAPES=0` |

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
  MemAvailable and the memory guard stopped it. Clips whose output exceeds 241 frames load the encoder for each
  request and close it again. A session that changes to such a length closes its resident encoder first, and loads
  it again when it changes back ([16](#16-reshape-another-length-or-orientation-without-a-reload)).
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
  patch checks the first 8 calls (layers 0-7 of the first update, the same kernel, real data). A session that
  changes its clip shape checks again on the first request at each new shape
  ([16](#16-reshape-another-length-or-orientation-without-a-reload)).
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
- **Switches.** `SPARK_AUTOTUNE_CACHE_FIX=0` disables it; `SPARK_DEBUG_AUTOTUNE=1` prints each fallback hit, each
  kernel tuned at run time and each tuning result stored.
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

### 16. Reshape: another length or orientation without a reload

- **Purpose.** A session was built for one task, length and orientation. Another length or orientation closed it
  and loaded every model again: the first clip after such a switch took about 60-80 s longer than a warm clip
  (from the studio's worker logs on one DGX Spark; issue #45). Nothing loaded depends on the shape.
  `Pipeline.reshape(orientation=..., seconds=...)` gives the running workers the recipe of the new shape, for the
  same task (worker command `reshape`):
  - Stage 1: height, width and frame count of the next request (FastVideo takes them per request), the latent
    capture's shape check and PCM length, and the ref2va reference-video canvas. The 8-call cuDNN BSA self-check
    runs again on the first request at a shape the worker has not checked; the receipt lists `bsa_checked_shapes`.
  - Stage 2: the shared module's frame and shape globals, set to the values a fresh worker for that shape has
    (`bind_geometry`; a test runs the module's own import code for every length), the latent tools, the decoder
    tiles (validated against the new pixel shape) and the spatial route. The next request runs under the
    compile warm-up timeout, as a fresh worker's first request does.
  - Prompt encoder: the keyframe canvas. Above `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` the resident encoder closes and
    each request loads its own, as before. Back below, the resident encoder loads again in the background if
    MemAvailable is at least `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` (default 32 GiB); otherwise the next clip loads its
    own encoder and the resident one loads after that clip.
  - If a worker fails the command, the reshape fails, and the pipeline refuses requests until it is closed. A
    pipeline that is closing refuses a reshape, and starts no resident encoder after its close began.
- **Static compiles.** Graphs for a new shape compile on its first request. Stage 1's regions and Stage 2's blocks
  now compile with `dynamic=False` (Stage 1 sets `torch._dynamo.config.automatic_dynamic_shapes = False` in its
  worker, which does the same for every compile there). With upstream's `dynamic=None`, Dynamo compiles a graph
  statically the first time and, when a size changes, may compile it again with that size dynamic, which can use
  other kernels (a CPU test shows it for a small function with the Stage 1 torch, 2.12). That happens already in a
  fresh worker's first request: the AdaLN lookup has 1 timestep row in the first update and 2 in the next three, so
  Stage 1's regions for updates 2-4 get a dynamic size (inferred from the code). It happens again after a prompt
  length or clip shape change. With `dynamic=False` every graph is static, and each shape gets the graph a fresh
  worker compiles for it, so a worker's output does not depend on what it ran before. Consequences:
  - Stage 1's graphs differ from 0.4's for every clip, not only after a reshape. In one GPU check the Stage-1
    latents and the MP4 streams were the same with `SOL_H3_STATIC_SHAPES=0` and `1` (3 clips at 5 s 16:9, two
    prompt lengths, fresh sessions).
  - Each new prompt length also compiles the two text-refiner regions statically (about 1-2 s, inferred); before,
    their dynamic graph served later lengths.
  - The compile-cache keys don't include the setting (read in torch's source), but the graphs change: each prompt
    length and shape that Stage 1 compiled before compiles once more after the upgrade (inferred).
- **Files.** `runtime/pipeline.py`, `runtime/worker.py`, `runtime/config.py`, `runtime/stage1.py`,
  `runtime/stage1_ops/{capture,vsa}.py`, `runtime/stage2.py`, `runtime/stage2_ops/models.py`, `runtime/qwen.py`,
  `tests/test_reshape.py` (new).
- **Switches.** `SOL_H3_STATIC_SHAPES=0` restores `dynamic=None`. The studio's `SOLH3_RESHAPE=0` closes the session
  and loads the models again for any change, as before. `infer.py` does not reshape.
- **Measured** (one GPU run on one DGX Spark, 22 clips, one seed, short prompts: a reshaped session against fresh
  sessions at each shape, each a first and a warm clip):
  - Stage-1 latents (capture sha256) were the same as in a fresh session at every shape: t2va 5, 10 and 15 s 16:9,
    fl2va 5 s 9:16. The MP4 streams of warm clips were the same at 5 s, 10 s and fl2va 9:16, and a fresh-against-fresh
    control at 5 s matched too. The first clip of a process, or the first after a change to a shape that process had
    not compiled, differs slightly from later clips in fresh sessions as well, and it matched the fresh first clip
    where checked (10 s).
  - At 15 s the warm MP4s' video differed (audio the same). Both workers ran the same compiled Stage-2 graph. Three
    Stage-2 reduction kernels of that shape had no tuning result stored yet, so each process tuned them by timing at
    run time, which can pick another block size and shifts the numerics; only the fresh worker stored its result
    (read from the compile cache afterwards). `SPARK_DEBUG_AUTOTUNE=1` prints such tunings.
  - First clip after a change, minus a warm clip at that shape: 5 → 10 s 14.4 s, 5 → 15 s 27.0 s (both compiled
    Stage 1 for this prompt length from scratch), 15 → 5 s (a shape the session had run) none, fl2va 16:9 → 9:16
    0.2 s. Reloads in the same run: 37.5-47.1 s, plus 4-9 s of closing.
  - Lowest MemAvailable 8.1 GiB, at 10 s, in the reshaped and in a fresh session alike. Stage 1's reserved memory
    after the 15 s clips was back at its 5 s level (37.6 GiB). The BSA self-check ran 8 calls at each new shape.
- **Deviation.** One worker renders several shapes; NVIDIA ran one shape per process (see the table above).

### 17. Tests

`tests/test_geometry.py` (new) covers the derived portrait geometry. `tests/test_pipeline.py` adds
deliver-first, the background prompt encoder, the new `Pipeline` arguments and the compile-cache location (it
also points the cache at a temp directory, so running the tests writes nothing under `$HOME`).
`tests/test_qwen_media.py` and `tests/test_stage1_contracts.py` follow the new signatures and the configurable
BSA check. `tests/test_stage2_contract.py` adds the Stage-2 audio policy (frozen and joint states, the setting,
older LTX-2), `tests/test_pipeline.py` its check before any worker starts. `tests/test_reshape.py` (new) covers
reshape: stand-in workers check that Stage 1, Stage 2 and the prompt encoder each see the shape of every request;
the Stage 2 globals are compared with the shared module's own import code for every length and orientation; it also
covers the capture's shape check and PCM trim, the BSA self-check per shape, the keyframe canvas, the static
compiles, a pipeline that is closing, failed a reshape or is short of memory for the resident encoder, and the
`SPARK_DEBUG_AUTOTUNE` lines. 98 tests in total (upstream: 65).

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
| `SPARK_DEBUG_AUTOTUNE` | 0 | 1 = print autotune-cache fallback hits, kernels tuned at run time and stored results |
| `SOL_H3_VAE_ON_DEVICE` | 1 | 0 = build the VAEs on the CPU as upstream |
| `SOL_H3_AUDIO_VAE_PREFETCH` | 1 | 0 = load the audio VAE when first needed |
| `SOL_H3_STAGE2_AUDIO` | frozen | `joint` = upstream: Stage 2 re-noises and regenerates the audio it refines the video against |
| `SOL_H3_STATIC_SHAPES` | 1 | 0 = upstream `torch.compile(dynamic=None)`: after a size change a graph may get dynamic sizes |
| `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` | 32 | MemAvailable needed to load the resident prompt encoder right after a reshape back to 10 s or less; below it, the next clip loads its own encoder first |
| `SOLH3_RESHAPE` (studio) | 1 | 0 = another length or orientation closes the session and loads the models again |
| `MIMALLOC_PURGE_DELAY` | 0 (workers) | any value you set wins |

The `SPARK_*` names are kept from the patch's first version so existing setups keep working.

## License of the changes

The files the patch modifies are Apache-2.0 NVlabs/Sana code; `runtime/qwen_ops/media.py` also contains
FastVideo-derived Apache-2.0 code (see Sol-H3-Spark's `THIRD_PARTY_NOTICES.md`). The changes and the new files are
Apache-2.0, like the rest of this repository.
