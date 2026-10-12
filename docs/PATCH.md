# The patch: what it changes in Sol-H3-Spark

`patches/sol-h3-studio.patch` changes NVIDIA's Sol-H3-Spark package in the
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark) repository,
branch `sol-engine`. It is a plain `git diff`: 23 files, 18 modified and 5 new.

- Base: `6c2f582ba9681dfe352ae4ca23b6f0030cf120cf` (`patches/BASE_COMMIT`). Applies cleanly; 87 CPU tests pass.
- Also checked on `sol-engine` HEAD `670482d8a857d578ac8a2ea89b052d0fb47badba` (2026-09-29): applies cleanly,
  69 CPU tests passed with the patch of that date. The upstream files the patch touches are identical in both commits.
  The current patch also applies (`git apply --check`) to `670482d` and to `62da34b` (2026-10-11), where those files
  are unchanged too.
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

# CPU tests (no GPU, no weights), with the Stage1 python. SOL_H3_SPARK_COMFY_ROOT (the pinned ComfyUI) lets the
# prompt-token tests read its Qwen tokenizer files; without it 5 tests are skipped.
cd models/minimax_h3/Sol-H3-Spark
SOL_H3_SPARK_COMFY_ROOT=~/sol-h3/sources/ComfyUI CUDA_VISIBLE_DEVICES= ~/sol-h3/envs/stage1/bin/python -B \
    -m unittest discover -s tests   # Ran 87 tests ... OK (skipped=1)

# the skipped test (ComfyUI's own MiniMaxH3Tokenizer) needs ComfyUI's packages: run the prompt-token tests in the
# Qwen image, on the CPU (no --gpus)
COMFY=~/sol-h3/sources/ComfyUI
docker run --rm --network=none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= \
    -e SOL_H3_SPARK_COMFY_ROOT="$COMFY" -v "$COMFY:$COMFY:ro" -v "$PWD:$PWD:ro" -w "$PWD" \
    --entrypoint python3 sol-h3-spark-qwen -B -m unittest tests.test_prompt_tokens   # Ran 12 tests ... OK

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
| The prompt encoder tokenizes `<d>`, `</d>` and 5 more tags as single tokens, as MiniMax's tokenizer does, instead of splitting them into text as the pinned ComfyUI does. Not yet compared on clips. See [16](#16-minimaxs-prompt-tokens). | off | `SOL_H3_PROMPT_TOKENS=plain` (the default) |

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

### 16. MiniMax's prompt tokens

- **Purpose.** H3 was trained with the tokenizer files of MiniMax's H3 repository. Their `tokenizer_config.json` adds
  7 special tokens after Qwen2.5's vocabulary: `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
  `<|caption_start|>` and `<|caption_end|>`, ids 151669-151675. Sol-H3-Spark's prompt encoder tokenizes with
  ComfyUI v0.30.0, which has only Qwen2.5's tokens. It splits every dialogue tag into ordinary text tokens, which
  also merge with the characters next to them: `<d>[English] Welcome to the garden.</d>` becomes `Ġ< d >[ … .</ d >`
  (20 tokens for the line "The woman (S1) says: …") instead of MiniMax's `Ġ <d> [ … . </d>` (19).
  - `minimax`: the Qwen worker adds the 7 tokens to ComfyUI's tokenizer after the text encoder loads, with the call
    ComfyUI itself uses since 924743af (first released in v0.34.0). It fails unless they get ids 151669-151675 and
    each tokenizes as one id. A ComfyUI that already has them is left as it is.
  - `plain`: the pinned tokenizer as before. It refuses a ComfyUI that already has the tokens, since that could no
    longer give the plain ids.
  - A prompt without any of the 7 tokens gets the same ids in both modes. `<cutoff>` and `<scenetrans>` from MiniMax's
    prompt guides are not tokens in either tokenizer.
  - H3's text encoder is stock Qwen3-VL-32B, where these 7 embedding rows are untrained, like the unused rows after
    them. To H3's transformer they are the fixed markers it saw around every dialogue line in training.
- **How the setting reaches the worker.** The Qwen worker runs in Docker with a fixed environment (`qwen_python.sh`),
  so the host reads `SOL_H3_PROMPT_TOKENS` in `runtime/config.py` and passes the mode to the worker in the recipe, as
  `stage1.prompt_tokens`. A value other than `plain` or `minimax` (any case) fails when the pipeline is created,
  before any worker loads; the studio and `bin/sol-h3-studio --check` refuse to start with it.
- **Receipts.** `results.json` has the mode in its recipe. Each request's Qwen receipt (`qwen/conditioning.json`)
  records `prompt_tokens` (the mode), `text_tokens` (the text positions in the conditioning) and `minimax_tokens` (how
  many of them are the 7 tokens). The Qwen worker's log has a `[sol-h3-studio] prompt tokens:` line.
- **Caches.** No cache keeps a prompt encoding between requests, so none needs the mode in its key. Stage 1 takes
  only the fresh Qwen receipt of its own request. `fixed-prompt.pt` (`runtime/prompt_cache.py`) holds Stage 2's
  fixed prompt, encoded by Gemma; it never sees the user's prompt. A dialogue prompt has fewer tokens with `minimax`,
  so its first clip in that mode may compile new Stage-1 graphs (see 13).
- **Files.** `runtime/qwen_ops/prompt_tokens.py` (new), `runtime/qwen.py`, `runtime/config.py`, `runtime/pipeline.py`,
  `tests/test_prompt_tokens.py` (new), `tests/test_pipeline.py`, `tests/test_qwen_media.py`.
- **Switch.** `SOL_H3_PROMPT_TOKENS=plain|minimax`, default `plain`. `Pipeline(..., prompt_tokens=...)` overrides it.
- **Measured.** On the CPU only, in the `sol-h3-spark-qwen` image (transformers 5.17.0), with the pinned ComfyUI's own
  `MiniMaxH3Tokenizer`, compared with the tokenizer files of MiniMax's H3 repository:
  - 40 texts: prompt examples and instructions from MiniMax's two prompt guides, lines of notes about the guides,
    and the studio page's 3 format hints. 30 contain one of the 7 tokens (29 of them `<d>`). They get MiniMax's ids
    for 40 of 40 with `minimax` and for 10 of 40 with `plain`: the 10 without any of the 7 tokens, which get the same
    ids in both modes.
  - The 6 prompts of the A/B test planned for issue #38 (t2va, ref2va and the 10 s fl2va scene of #21): MiniMax's
    ids for 6 of 6 with `minimax`, 1 of 6 with `plain` (the one without dialogue).
  - Not yet measured on clips. `minimax` becomes the default only if an A/B test of lip sync, speech and picture on
    the same seeds holds up (issue #38).
- **Deviation.** With `minimax`, prompts reach H3 as MiniMax's tokenizer gives them, not as NVIDIA's route does (see
  the table).

### 17. Tests

`tests/test_geometry.py` (new) covers the derived portrait geometry. `tests/test_pipeline.py` adds
deliver-first, the background prompt encoder, the new `Pipeline` arguments and the compile-cache location (it
also points the cache at a temp directory, so running the tests writes nothing under `$HOME`).
`tests/test_qwen_media.py` and `tests/test_stage1_contracts.py` follow the new signatures and the configurable
BSA check. `tests/test_stage2_contract.py` adds the Stage-2 audio policy (frozen and joint states, the setting,
older LTX-2), `tests/test_pipeline.py` its check before any worker starts.
`tests/test_prompt_tokens.py` (new) covers the prompt-token setting and its recipe entry, the token checks with a
stand-in tokenizer, and the Qwen worker's session applying the recipe's mode to the CLIP's tokenizer (with stand-ins
for ComfyUI, torch and the checkpoint). With `SOL_H3_SPARK_COMFY_ROOT` set it also loads the pinned ComfyUI's Qwen
tokenizer files: the 7 ids, the 19 ids of the example above, and unchanged ids for prompts without the tokens. Its
test of ComfyUI's own `MiniMaxH3Tokenizer` needs ComfyUI's packages: it is skipped in the Stage1 environment and runs
in the Qwen image ([Apply, test, revert](#apply-test-revert)). `tests/test_pipeline.py` checks the setting before
any worker starts and that the setting, or the `Pipeline` argument, reaches every worker; `tests/test_qwen_media.py`
checks the receipt fields.
87 tests in total (upstream: 65).

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
| `SOL_H3_PROMPT_TOKENS` | plain | `minimax` = MiniMax's 7 prompt tokens (`<d>`, `</d>`, ...) are single tokens, as in H3's training |
| `MIMALLOC_PURGE_DELAY` | 0 (workers) | any value you set wins |

The `SPARK_*` names are kept from the patch's first version so existing setups keep working.

## License of the changes

The files the patch modifies are Apache-2.0 NVlabs/Sana code; `runtime/qwen_ops/media.py` also contains
FastVideo-derived Apache-2.0 code (see Sol-H3-Spark's `THIRD_PARTY_NOTICES.md`). The changes and the new files are
Apache-2.0, like the rest of this repository.
