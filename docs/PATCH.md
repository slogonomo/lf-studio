# The patch: what it changes in Sol-H3-Spark

`patches/sol-h3-studio.patch` changes NVIDIA's Sol-H3-Spark package in the
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark) repository,
branch `sol-engine`. It is a plain `git diff`: 27 files, 20 modified and 7 new.

- Base: `6c2f582ba9681dfe352ae4ca23b6f0030cf120cf` (`patches/BASE_COMMIT`). Applies cleanly; 133 CPU tests pass.
- Also checked on `sol-engine` HEAD `670482d8a857d578ac8a2ea89b052d0fb47badba` (2026-09-29): applies cleanly,
  69 CPU tests passed with the patch of that date. The upstream files the patch touches are identical in both commits.
  The current patch also applies (`git apply --check`) to `670482d` and to `62da34b` (2026-10-11), where those files
  are unchanged too.
- 19 of the 20 modified files are in `models/minimax_h3/Sol-H3-Spark`. One is a shared Sana module:
  `models/minimax_h3/super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py`.
- Every modified file carries a `Modified by sol-h3-studio` comment at its first change. Other changed lines are
  tagged `sol-h3-studio:`. The 7 new files say they are part of sol-h3-studio.

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
    -m unittest discover -s tests   # Ran 133 tests ... OK (skipped=1)

# the skipped test (ComfyUI's own MiniMaxH3Tokenizer) needs ComfyUI's packages: run the prompt-token tests in the
# Qwen image, on the CPU (no --gpus)
COMFY=~/sol-h3/sources/ComfyUI
docker run --rm --network=none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= \
    -e SOL_H3_SPARK_COMFY_ROOT="$COMFY" -v "$COMFY:$COMFY:ro" -v "$PWD:$PWD:ro" -w "$PWD" \
    --entrypoint python3 sol-h3-spark-qwen -B -m unittest tests.test_prompt_tokens   # Ran 12 tests ... OK

# undo the whole patch
git -C ../../.. apply -R /path/to/sol-h3-studio/patches/sol-h3-studio.patch
```

`install/10-sana.sh` does the clone, the apply and the tests (without the `SOL_H3_*` and `SPARK_*` switches of the
settings file, since the tests check the defaults; the `SOL_H3_SPARK_*` paths stay). Most features below also have
an environment switch that restores NVIDIA's behaviour without removing the patch. Set switches in the settings file
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
| The prompt encoder tokenizes `<d>`, `</d>` and 5 more tags as single tokens, as MiniMax's tokenizer does, instead of splitting them into text as the pinned ComfyUI does. Compared on clips in an A/B test; see [16](#16-minimaxs-prompt-tokens). | on (since 0.4) | `SOL_H3_PROMPT_TOKENS=plain` |
| A running session changes its clip length and orientation in place (reshape) instead of loading the models again. It gave the same Stage-1 latents as a fresh session at every shape checked, and the same MP4 wherever two fresh sessions agree (5 s, 10 s, 9:16). At 15 s Stage 2 differed between any two processes, reshaped or not, because three kernels were tuned again in each. 0.4 stops this re-tuning ([8](#8-inductor-autotune-cache-fix-torch-213), issue #51). Whether 15 s output then matches is not shown yet: the 15 s comparison runs again in staging ([18](#18-reshape-another-length-or-orientation-without-a-reload)). | on (studio) | `SOLH3_RESHAPE=0` (studio setting) |
| Stage 1's regions and Stage 2's blocks compile with `torch.compile(dynamic=False)`: every graph is static, also after a shape change. This changes Stage 1's graphs from the first request on: upstream compiles the regions for updates 2-4 with a dynamic size even then (inferred from the code). The output was the same with and without it in one GPU check (3 clips). See [18](#18-reshape-another-length-or-orientation-without-a-reload). | on | `SOL_H3_STATIC_SHAPES=0` |

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
  it again when it changes back ([18](#18-reshape-another-length-or-orientation-without-a-reload)).
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
  - If Stage1 dies before that point, Stage2 is not started, and a failed load joins the half-started workers
    so `close()` can stop them.
  - The resident prompt encoder loads in the background while the first clip refines, so the next clip doesn't
    wait for it. It starts once Stage1's first request has returned and Stage2 is built: both load peaks are over.
    Stage2 releases its idle cache first, as it does before every later clip. If MemAvailable is then below
    a minimum (32 GiB up to 5 s, 34 GiB for longer clips), the encoder loads after the first clip, as before.
    `results.json` records the reading and where the encoder started (`resident_qwen`). The warm-up encoder still
    closes before Stage2 starts. Sessions over 241 output frames load the encoder per request, as before (item 3).
    A clip that waits for an encoder still loading may be interrupted (Ctrl-C in `infer.py`): the encoder stays
    registered, and `close()` waits for it and closes it.
- **Files.** `runtime/pipeline.py`, `infer.py`, `tests/test_pipeline.py`.
- **Switch.** `deliver` is opt-in (`infer.py` without `--deliver-first` keeps the separate warm-up). The overlap
  itself has no switch. `SOL_H3_RESIDENT_QWEN_EARLY=0` loads the resident encoder after the first clip, as before
  0.4. `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` sets one minimum for every length instead (a value that is not a number
  of GiB, such as `32GiB`, stops the pipeline when it is created, with the setting's name). The same minimum, for the
  new length, decides whether the resident encoder loads again right after a session changes back to 10 s or less
  ([18](#18-reshape-another-length-or-orientation-without-a-reload)).
- **Measured.** First clip 282 → 177 s, lowest MemAvailable 18 GiB. The first overlap attempt fell to 2.4 GiB
  MemAvailable and was stopped by the memory guard; the version in the patch waits for the Stage1 marker.
- **Resident encoder during the refine (0.4).** Before 0.4 it loaded after the first clip, and the next clip
  started its prompt encoding 3-11 s late in 17 of 128 warm web-UI clips (studio receipts, 2026-09-23 to 10-12).
  Measured in the cold-start bench traces before 0.4 (MemAvailable every 0.25 s; t2va 16:9 at 5, 10 and 15 s, fl2va
  5 s):
  - Through the first clip's Stage2, denoise and decode alike, MemAvailable stayed at 20-23 GiB (fl2va 26 GiB).
    At 5 s it was already that low when Stage2 became ready, before the refine.
  - The release after the MP4 then freed 12-14 GiB, at every length.
  - Loading the encoder took 17-19 GiB of MemAvailable, more than its 14.6 GiB of weights.
  - A warm refine with the encoder resident took 4.1-5.7 GiB at 5 s (down to 11.0-12.7 GiB) and ~8 GiB at 15 s.

  So the encoder can't just start during the refine: that would leave ~3-5 GiB, at the memory guard's 5 GiB
  (inferred). With Stage2's release first, the first refine gets about as low as a warm clip with the encoder
  resident. Measured on a separate test studio on the same DGX Spark (its own checkout, runtime root and clip
  history), 2026-10-13, one run each: `SOL_H3_RESIDENT_QWEN_EARLY=0` → the default (the same branch); MemAvailable
  every 0.5 s.

  | Load | Next clip's encoding starts after its request | Next clip | MemAvailable at the check | Lowest in the load | Lowest in the next clip |
  |---|---|---|---|---|---|
  | t2va 16:9 5 s | 11.1 → 0.16 s | 74.2 → 54.6 s | 34.2 GiB | 14.0 → 10.3 GiB | 12.6 → 11.9 GiB |
  | fl2va 16:9 10 s | 6.1 → 0.13 s | 124.2 → 118.4 s | 37.0 GiB | 15.4 → 10.3 GiB | 11.2 → 12.2 GiB |
  | warm-up, t2va 16:9 5 s | 6.2 → 0.16 s | 61.4 → 55.3 s | 35.2 GiB | 16.3 → 12.1 GiB | 12.1 → 11.8 GiB |

  - The load's lowest point is now in the first refine, 23-27 GiB below the reading at the check (the encoder plus a
    first refine). In two of the three loads it was 1.5-2 GiB below the next clip's. The memory guard never fired.
  - From the end of Stage1's first request to the MP4, the part of the first clip the change acts on, it took
    0.7-3.0 s longer: the refine while the encoder loads. One whole first clip was 12.6 s slower, 9.6 s of it
    before Stage1 returned, where both runs run the same code.
  - The next clips came out identical for t2va: the encoder's output, Stage1's tensors, decoded audio and video.
    For fl2va the encoder's output was identical, but Stage1 differed. Each run had compiled that prompt length's
    Stage1 graphs anew, and autotuning picked another reduction kernel in one of them. A second fl2va run of the
    default, started from the other run's final compile cache, matched it bit for bit in both clips: the encoder's
    output, Stage1's tensors, decoded audio and video.

  The minimum depends on the length, since a longer first refine needs more. Measured, the encoder plus a first
  refine took 23.1-23.8 GiB at t2va 5 s and 26.7 GiB at fl2va 10 s. So the minimum is 32 GiB up to 121 output
  frames (5 s) and 34 GiB above: at the minimum, ~8-9 GiB would be left at 5 s and ~7 GiB at 10 s, above the memory
  guard's 5 GiB. After the release, the bench traces had 34-39 GiB in every setting and the test studio 34.2-37.0 GiB,
  so the check passes in a normal session. It is there for a box with less free memory. `results.json` records the
  minimum that applied (`resident_qwen.min_gib`, `min_from`).
  With a cold compile cache (the first run of a format and length), Stage2's first refine also compiles and
  benchmarks kernels (max-autotune) while the encoder loads. That first clip is then likely slower than it would be
  without the load, and the kernel choices saved in the cache are timed under it (inferred, not measured).

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
  ([18](#18-reshape-another-length-or-orientation-without-a-reload)).
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
- **A fallback hit is a hit (0.4, issue #51).** Up to 0.3 the fallback labelled the kernel's cache state
  "hit (spark fallback)". Torch skips its run-time R0_BLOCK halving (`_dynamic_scale_rblock`) only when the state is
  exactly "hit". Where that halving applies, a kernel read through the fallback got smaller-block candidates added and
  was benchmarked by timing. A smaller-block winner then also went through coordinate descent, because Triton's
  `Config` drops its `found_by_coordesc` mark when it is copied (read in torch's and Triton's code, shown on the
  CPU). The winner overwrote the stored result. The next process read that result and could pick another config,
  which changes a reduction's order and so the output. Seen at 15 s for three of Stage 2's audio-branch reductions
  (`SPARK_DEBUG_AUTOTUNE=1` lines, two GPU runs). Not at 5 s or 10 s, where the halving rule adds no candidates for
  these kernels.
  - Now the fallback sets the cache info that torch's own hit sets (`autotune_cache_state` `"hit"` and
    `best_config`), plus `spark_fallback: True`. Torch uses the stored result as it is: no halving, no benchmark, no
    coordinate descent, nothing stored. The marker keeps the fallback visible: `SPARK_DEBUG_AUTOTUNE=1` prints a
    `fallback hit` line for each, and a tuning line would name the state `hit (spark fallback)`.
  - Nothing needs clearing after the upgrade. The first 15 s clip after it uses the result the last process before
    the upgrade stored, and stores nothing new; every later process reads the same result. So that clip settles the
    configs used from then on (inferred). The same holds for any other shape whose kernels were re-tuned this way.
    A kernel without a stored result is still tuned by timing once, on the first clip that needs it, as before.
  - Not part of the patch (proposals 2-3 of the issue): deleting the stored results first so that one deliberate run
    tunes them, and a pre-tune pass per shape.
  - 5 s and 10 s are expected to give the same output as before: their kernels read through the fallback already
    ran the stored config (inferred).
- **Files.** `runtime/stage2.py` (monkeypatch, installed in the Stage2 worker), `tests/test_stage2_contract.py`,
  `tests/test_reshape.py` (the debug lines).
- **Switches.** `SPARK_AUTOTUNE_CACHE_FIX=0` disables it; `SPARK_DEBUG_AUTOTUNE=1` prints each fallback hit, each
  kernel tuned at run time and each tuning result stored.
- **Measured.** 11/11 hits, no re-tuning; Stage2 first render 45-49 → 41 s; first clip 103-104 → 99.3/99.4 s.
  Specific to torch 2.13's Inductor; re-check it when the Stage2 torch changes. The 0.4 change is checked on the CPU
  only, with a stand-in `CachingAutotuner` that follows torch 2.13's halving, benchmark and coordinate-descent rules
  and uses torch's own cache check (19). The GPU check (two fresh 15 s sessions and a reshaped one with the same MP4
  streams, no `tuned at run time` line after the first 15 s process) runs in staging.

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
  `SOL_H3_DYNAMIC_TEXT_LENGTH=1` ([17](#17-prompt-lengths-without-recompiles-experimental)) stops the per-length
  graphs of the body blocks.
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
  - `minimax` (the default since 0.4): the Qwen worker adds the 7 tokens to ComfyUI's tokenizer after the text
    encoder loads, with the call ComfyUI itself uses since 924743af (first released in v0.34.0). It fails unless they
    get ids 151669-151675 and each tokenizes as one id. A ComfyUI that already has them is left as it is.
  - `plain`: the pinned tokenizer, as up to 0.3. It refuses a ComfyUI that already has the tokens, since that could no
    longer give the plain ids.
  - A prompt without any of the 7 tokens gets the same ids in both modes. `<cutoff>` and `<scenetrans>` from MiniMax's
    prompt guides are not tokens in either tokenizer.
  - H3's text encoder is stock Qwen3-VL-32B, where these 7 embedding rows are untrained, like the unused rows after
    them. To H3's transformer they are the fixed markers it saw around every dialogue line in training.
- **How the setting reaches the worker.** The Qwen worker runs in Docker with a fixed environment (`qwen_python.sh`),
  so the host reads `SOL_H3_PROMPT_TOKENS` in `runtime/config.py` and passes the mode to the worker in the recipe, as
  `stage1.prompt_tokens`. Every Qwen worker gets that recipe: the warm-up one, the resident one (also when it starts
  during the first clip, see 4, or again after a reshape, see
  [18](#18-reshape-another-length-or-orientation-without-a-reload)) and those loaded per request. A value other than
  `plain` or `minimax` (any case) fails when the pipeline is created, before any worker loads; the studio and
  `bin/sol-h3-studio --check` refuse to start with it. They also refuse `minimax`, set or by default, when the Sana
  checkout still carries an older patch, which would ignore the setting and tokenize as `plain`:
  "SOL_H3_PROMPT_TOKENS=minimax (the default since 0.4) needs a newer patch than the one applied in ...; run
  install/10-sana.sh". So after a `git pull`, run `install/10-sana.sh`. (Likewise `SOL_H3_STAGE2_AUDIO=frozen` with a
  patch older than 0.3, which refines as `joint`.)
- **Older clips.** Up to 0.3 every prompt was encoded `plain`. A dialogue prompt now encodes differently, so a
  dialogue clip made with 0.3 doesn't come back from its seed unless `SOL_H3_PROMPT_TOKENS=plain`. A prompt without
  any of the 7 tokens gets the same ids in both modes, and its clips came out bit-identical (measured; see below).
- **Receipts.** `results.json` has the mode in its recipe. Each request's Qwen receipt (`qwen/conditioning.json`)
  records `prompt_tokens` (the mode), `text_tokens` (the text positions in the conditioning) and `minimax_tokens` (how
  many of them are the 7 tokens). The Qwen worker's log has a `[sol-h3-studio] prompt tokens:` line.
- **Caches.** No cache keeps a prompt encoding between requests, so none needs the mode in its key. Stage 1 takes
  only the fresh Qwen receipt of its own request. `fixed-prompt.pt` (`runtime/prompt_cache.py`) holds Stage 2's
  fixed prompt, encoded by Gemma; it never sees the user's prompt. A dialogue prompt has fewer tokens with `minimax`,
  so its first clip in that mode may compile new Stage-1 graphs (see 13).
- **Files.** `runtime/qwen_ops/prompt_tokens.py` (new), `runtime/qwen.py`, `runtime/config.py`, `runtime/pipeline.py`,
  `tests/test_prompt_tokens.py` (new), `tests/test_pipeline.py`, `tests/test_qwen_media.py`.
- **Switch.** `SOL_H3_PROMPT_TOKENS=minimax|plain`, default `minimax` (`plain` up to 0.3).
  `Pipeline(..., prompt_tokens=...)` overrides it.
- **Measured.** On the CPU only, in the `sol-h3-spark-qwen` image (transformers 5.17.0), with the pinned ComfyUI's own
  `MiniMaxH3Tokenizer`, compared with the tokenizer files of MiniMax's H3 repository:
  - 40 texts: prompt examples and instructions from MiniMax's two prompt guides, lines of notes about the guides,
    and the studio page's 3 format hints. 30 contain one of the 7 tokens (29 of them `<d>`). They get MiniMax's ids
    for 40 of 40 with `minimax` and for 10 of 40 with `plain`: the 10 without any of the 7 tokens, which get the same
    ids in both modes.
  - The 6 prompts first planned for the A/B test of issue #38 (t2va, ref2va and the 10 s fl2va scene of #21):
    MiniMax's ids for 6 of 6 with `minimax`, 1 of 6 with `plain` (the one without dialogue).
- **Measured on clips.** The A/B test of issue #38 ran on the GPU in two windows (2026-10-12/13, one DGX Spark, a
  separate test studio). Each prompt was rendered in both modes with the same 4 seeds: t2va 5 s with one English
  speaker, the same line in Spanish, an off-screen voiceover and no dialogue; ref2va 5 s with a voice reference; two
  fl2va 10 s two-person scenes and variants of one of them. Measured with Whisper (words, language), SyncNet (LSE-C on
  the speaking face) and a look at every clip. Details in [PR #40](https://github.com/slogonomo/sol-h3-studio/pull/40).
  - The 8 clips whose prompts have none of the 7 tokens (5 s without dialogue, and the scene with its lines in
    quotes) are bit-identical between the modes, decoded audio and video.
  - Word error rate, language and lip sync tie within `plain`'s spread in every group of the second window. In the
    first window's two scenes no scripted word was lost and language and lip sync tied too, but `minimax` fell
    outside `plain`'s spread on the bad side on 8 of the 38 measures taken, each from one or two clips or from the
    burst below. Over 52 seed pairs of speaking faces in the final clips of both windows, `minimax` scored higher on
    25 and lower on 27.
  - Many clips with `<d>` lines started with a short voiced burst at 0 s, before the first word. With `minimax` and a
    first line that starts with a capital letter it was nearly gone: 1 of 24 clips (6 prompts), against 18 of 20 with
    `plain` (5 of those prompts). With a lowercase first word it stayed in both modes (16 of 16 `minimax` clips, two
    scenes). So start each `<d>` line with a capital letter, as MiniMax's guide does (issue #53). Lines in quotes
    instead of `<d>` tags had no burst (0 of 12 clips).
  - In the voiceover the lips should stay closed. They moved with the line in 3 of 4 `minimax` clips against 1 of 4
    `plain`, and in 3 of 4 with the line in quotes (identical in both modes).
  - Nothing here is significant at n = 4.
- **Deviation.** With `minimax`, prompts reach H3 as MiniMax's tokenizer gives them, not as NVIDIA's route does (see
  the table).

### 17. Prompt lengths without recompiles (experimental)

- **Purpose.** A warm session compiles new Stage-1 graphs for every prompt length it has not seen (13). From the
  studio's receipts, that added 7-14 s to Stage 1 of 23% of warm clips (issue #41). With
  `SOL_H3_DYNAMIC_TEXT_LENGTH=1`, the 50 body blocks of the H3 DiT compile 2 graphs per session (the first denoise
  step and the rest: the AdaLN table has fewer rows in the first step) and none per new prompt length.
- **Why they recompiled.** Dynamo specializes every int it reads from a global variable. FastVideo hands its
  sparse-attention metadata to the attention through a global forward context. So each body graph was tied to
  `total_seq_length` (the length check in FastVideo's `tile()`), to `num_prefix_tiles` (it changes every 64 text
  tokens) and to `current_timestep`. The patch's opaque attention op took the tile counts and the step as ints; the
  step made one graph per denoise step. `tile()` also branches on its tile buffer, which made one more graph for each
  request's first block. Measured on the CPU, with a small model built from FastVideo's own H3 block and
  attention-metadata classes and a stand-in for the attention kernel: 5 body graphs for every new length. The two
  text-refiner blocks compile at the first length and once more at the second; then PyTorch's automatic dynamic
  shapes cover any length.
- **What changes.** In `runtime/stage1_ops/regional.py`, with the setting on:
  - the opaque attention op reads the tile counts and the step from the forward context when it runs;
  - the length check and the tile-buffer bookkeeping of `tile()` run before each block, outside its graph. The graph
    keeps the scatter of the packed rows into the tile buffer and the gather back, as before;
  - the packed-row dimension of the block inputs and of the tile metadata is marked dynamic. The attention gets its
    row count from its own query instead of the block's int argument (the same value on one GPU).

  The text refiner and FastVideo are not changed. One GPU only (no sequence parallelism).
- **Ref2VA.** The dense route reads no sparse-attention metadata, so its body graphs already become length-dynamic
  at a session's second length (automatic dynamic shapes). There the setting saves that one recompile, but it also
  compiles the first length's body for a symbolic length (measured below: the same output). The optional Sol route
  (`ref_stage1_attn=sol`) ignores the setting; its receipt says `per_length (setting ignored)`.
- **Measured** on a separate test studio on the same DGX Spark, 2026-10-13, with this setting's patch before its
  merge into 0.4 (prompts tokenized `plain`, 16): a studio with the setting off and then one with it on, the same 19
  clips (5 s, 16:9) in the same order. A t2va, an fl2va (keyframe) and a ref2va session each had 2 new prompt
  lengths and repeated every prompt later. Both studios started from the same copy of the compile cache.
  - New prompt length, Stage 1 against the same prompt repeated later: with the setting 0.0-0.8 s in t2va and fl2va,
    and 0.0 s in ref2va at its third length. ref2va's second length cost 2.6 s: the text refiner recompiles there,
    with or without the setting. Without the setting: 2.3-9.7 s in t2va and fl2va, and 5.9 s and 0.0 s in ref2va
    (its dense body is already length-dynamic from the second length).
  - Graphs: with the setting, 2 body graphs at each session's first clip and none for any new length. Without it,
    5 body graphs for every new length in t2va and fl2va.
  - Known-length clips, median Stage 1 (3-4 clips per task): t2va 18.15 → 18.2 s (+0.3 %), fl2va 21.2 → 21.4 s
    (+0.9 %), ref2va 22.0 → 22.1 s (+0.5 %).
  - Output: every warm clip (16) was bit-identical with and without the setting (Stage-1 `payload_sha256` and
    `native_normalized_sha256`, MP4 video and audio streams). Without the setting, 4 of the clips also matched the
    same prompts and seeds from an earlier session of the #38 A/B.
- **Why the output stays the same.** In the body graph, Inductor generates only data movement, casts, the FP8
  activation scaling (a maximum and elementwise steps) and the residual adds. The norms, RoPE and SwiGLU are
  FastVideo's own Triton kernels, and the matmuls are cuBLAS calls of the same sizes. None of these results depends
  on how a kernel is tuned for a symbolic length (inferred).
- **A session's first clip, with or without the setting.** In the same run, with the setting on, the first fl2va
  clip's Stage-1 output differed from its own repeat later in the session (payload hash). Its MP4 against the
  setting-off run's first fl2va clip: SSIM 0.88 at 672×384. The other first clips matched their repeats at Stage 1
  (payload and native hashes) and in audio. The Stage-1 difference comes from the text refiner, which the setting
  doesn't change. It compiles a graph for the session's first length, then a dynamic one at the second length (13),
  and the repeat ran on the dynamic one. Inductor picks between two configurations of a reduction by timing it when
  it first compiles a kernel, and keeps the choice in the compile cache. Here the refiner's residual-add + RMSNorm
  kernel for that first length (the same source in both runs) got the looped configuration without the setting and
  the persistent one with it. Only the looped one rounds like the dynamic graph. So the first clip of a session at a
  length new to the compile cache can differ from a later re-render of the same prompt and seed, with or without the
  setting. Separately, in both runs the MP4 video of every session's first clip differed from its repeats, also where
  Stage 1 and the audio matched. That difference comes after Stage 1 (perhaps from Stage 2's first run in a session;
  inferred, not examined). Later clips matched their repeats in video too.
- **With static compiles** (`SOL_H3_STATIC_SHAPES=1`, the default;
  [18](#18-reshape-another-length-or-orientation-without-a-reload)). The Stage1 worker then turns Dynamo's automatic
  dynamic shapes off. A dimension marked dynamic stays dynamic without them (Dynamo's builder honours an explicit
  mark), so the body blocks keep their 2 graphs per session. What relied on automatic dynamic shapes compiles per
  length instead: the text refiner from a session's second length on, and, without this setting, the dense Ref2VA body
  (inferred). CPU tests show the graph counts of the body blocks and the text refiner with a stand-in model
  (`tests/test_regional_length.py`, `tests/test_reshape.py`), and the receipt's `text_regions` says which applies. The
  measurements above ran with dynamic compiles, before static compiles were in. The GPU check runs again in staging
  with `SOL_H3_DYNAMIC_TEXT_LENGTH=1`.
- **Files.** `runtime/stage1_ops/regional.py`, `runtime/stage1.py` (comment; the Sol route's receipt),
  `tests/test_regional_length.py` (new).
- **Switch.** `SOL_H3_DYNAMIC_TEXT_LENGTH`: `0` (default) as before, `1` on. Another value stops Stage 1 when it
  loads.

### 18. Reshape: another length or orientation without a reload

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
    MemAvailable is at least the minimum of its early load in item 4, for the new length (32 GiB up to 5 s, 34 GiB
    above, or `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB`). Otherwise the next clip loads its own encoder and the resident
    one loads after that clip, unless another reshape before that clip starts it. There is never more than one
    resident encoder. `results.json` records the reading with the reshape. Every encoder started after a
    reshape gets the session's prompt-token mode (16), even if `SOL_H3_PROMPT_TOKENS` changed meanwhile, and a
    resident one refuses a recipe with another mode.
  - If a worker fails the command, the reshape fails, and the pipeline refuses requests until it is closed. A
    pipeline that is closing refuses a reshape, and starts no resident encoder after its close began. A reshape
    that a close overlaps (Unload while it waits for a resident encoder that is still loading) fails, and the close
    waits for that encoder and closes it. A close that overlaps a reshape closing the resident encoder (a change to
    over 10 s) returns only once that encoder has shut down.
  - Before the first clip only the recipe changes, and the first clip follows the new length's rules for the
    resident encoder (item 3, item 4).
- **Static compiles.** Graphs for a new shape compile on its first request. Stage 1's regions and Stage 2's blocks
  now compile with `dynamic=False` (Stage 1 sets `torch._dynamo.config.automatic_dynamic_shapes = False` in its
  worker, which does the same for every compile there). With upstream's `dynamic=None`, Dynamo compiles a graph
  statically the first time and, when a size changes, may compile it again with that size dynamic, which can use
  other kernels (a CPU test shows it for a small function with the Stage 1 torch, 2.12). That happens already in a
  fresh worker's first request: the AdaLN lookup has 1 timestep row in the first update and 2 in the next three, so
  Stage 1's regions for updates 2-4 get a dynamic size (inferred from the code). It happens again after a prompt
  length or clip shape change. With `dynamic=False` every graph is static, and each shape gets the graph a fresh
  worker compiles for it, so a worker's output does not depend on what it ran before. Consequences:
  - Stage 1's graphs differ from upstream's and 0.3's for every clip, not only after a reshape. In one GPU check the
    Stage-1 latents and the MP4 streams were the same with `SOL_H3_STATIC_SHAPES=0` and `1` (3 clips at 5 s 16:9,
    two prompt lengths, fresh sessions).
  - Each new prompt length also compiles the two text-refiner regions statically (about 1-2 s, inferred); before,
    their dynamic graph served later lengths.
  - The compile-cache keys don't include the setting (read in torch's source), but the graphs change: each prompt
    length and shape that Stage 1 compiled before compiles once more after the upgrade (inferred).
  - With `SOL_H3_DYNAMIC_TEXT_LENGTH=1` (17): the dimensions it marks dynamic stay dynamic, because Dynamo honours an
    explicit mark without automatic dynamic shapes. Its body blocks keep their 2 graphs per session; the text-refiner
    regions, which it leaves alone, compile once per new length, as above. CPU tests show both graph counts with a
    stand-in model and the Stage 1 torch. The two settings have not run together on the GPU; that check runs again
    in staging with `SOL_H3_DYNAMIC_TEXT_LENGTH=1`.
- **Files.** `runtime/pipeline.py`, `runtime/worker.py`, `runtime/config.py`, `runtime/stage1.py`,
  `runtime/stage1_ops/{capture,vsa,regional}.py`, `runtime/stage2.py`, `runtime/stage2_ops/models.py`,
  `runtime/qwen.py`, `tests/test_reshape.py` (new), `tests/test_regional_length.py`.
- **Switches.** `SOL_H3_STATIC_SHAPES=0` restores `dynamic=None`. Another value (empty is the default) fails when the
  pipeline is created; the studio and `bin/sol-h3-studio --check` refuse to start with it. They also refuse `1`, set
  or by default, when the Sana checkout carries an older patch, which compiles as `0`. The studio's
  `SOLH3_RESHAPE=0` closes the session and loads the models again for any change, as before. `infer.py` does not
  reshape.
- **Measured** (one GPU run on one DGX Spark, 22 clips, one seed, short prompts: a reshaped session against fresh
  sessions at each shape, each a first and a warm clip; with this change's patch before its merge into 0.4, so
  prompts tokenized `plain` and the resident encoder loaded after the first clip):
  - Stage-1 latents (capture sha256) were the same as in a fresh session at every shape: t2va 5, 10 and 15 s 16:9,
    fl2va 5 s 9:16. The MP4 streams of warm clips were the same at 5 s, 10 s and fl2va 9:16, and a fresh-against-fresh
    control at 5 s matched too. The first clip of a process, or the first after a change to a shape that process had
    not compiled, differs slightly from later clips in fresh sessions as well, and it matched the fresh first clip
    where checked (10 s).
  - At 15 s the warm MP4s' video differed (audio and Stage 1 the same). A second run, with
    `SPARK_DEBUG_AUTOTUNE=1`, showed why. Two fresh 15 s sessions differed too: Stage 2 at 15 s wasn't reproducible
    across processes, reshaped or not. Both runs used the same compiled Stage-2 graph. Three reduction kernels of
    the 15 s audio branch read their stored tuning result through item 8's fallback, were tuned again at run time
    anyway, and stored a new winner each time. So reshape matched a fresh session at every shape where two fresh
    sessions matched each other (5 s, 10 s, fl2va 9:16). Item 8's fallback now marks its reads as hits, so these
    kernels are no longer tuned again (issue #51). The 15 s comparison runs again in staging.
  - First clip after a change, minus a warm clip at that shape: 5 → 10 s 14.4 s, 5 → 15 s 27.0 s (both compiled
    Stage 1 for this prompt length from scratch), 15 → 5 s (a shape the session had run) none, fl2va 16:9 → 9:16
    0.2 s. Reloads in the same run: 37.5-47.1 s, plus 4-9 s of closing.
  - Lowest MemAvailable 8.1 GiB, at 10 s, in the reshaped and in a fresh session alike. Stage 1's reserved memory
    after the 15 s clips was back at its 5 s level (37.6 GiB). The BSA self-check ran 8 calls at each new shape.
- **Deviation.** One worker renders several shapes; NVIDIA ran one shape per process (see the table above).

### 19. Tests

`tests/test_geometry.py` (new) covers the derived portrait geometry. `tests/test_pipeline.py` adds
deliver-first, when the resident prompt encoder starts, the new `Pipeline` arguments and the compile-cache
location (it also points the cache at a temp directory, so running the tests writes nothing under `$HOME`).
`tests/test_qwen_media.py` and `tests/test_stage1_contracts.py` follow the new signatures and the configurable
BSA check. `tests/test_pipeline.py` also checks that a malformed `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` fails with its
name before any worker starts. `tests/test_stage2_contract.py` adds the Stage-2 audio policy (frozen and joint
states, the setting, older LTX-2), `tests/test_pipeline.py` its check before any worker starts.
`tests/test_stage2_contract.py` also covers item 8's fallback with a stand-in `CachingAutotuner` that follows torch
2.13's rules for halving, benchmarking and coordinate descent, with torch's own recheck and cache check and Triton's
`Config`. A fallback hit gets the cache info of torch's own hit plus `spark_fallback`, and runs as stored. With the
label used before issue #51, or without the fix, the kernel is halved, benchmarked, tuned again and stored again.
`tests/test_prompt_tokens.py` (new) covers the prompt-token setting (default `minimax`) and its recipe entry, the token
checks with a stand-in tokenizer, and the Qwen worker's session applying the recipe's mode to the CLIP's tokenizer (with
stand-ins for ComfyUI, torch and the checkpoint). With `SOL_H3_SPARK_COMFY_ROOT` set it also loads the pinned ComfyUI's
Qwen tokenizer files: the 7 ids, the 19 ids of the example above, and unchanged ids for prompts without the tokens. Its
test of ComfyUI's own `MiniMaxH3Tokenizer` needs ComfyUI's packages: it is skipped in the Stage1 environment and runs in
the Qwen image ([Apply, test, revert](#apply-test-revert)). `tests/test_pipeline.py` checks the setting before any
worker starts and that the default, the setting or the `Pipeline` argument reaches every worker, the resident
prompt encoder too, whether it starts during the first clip or after it; `tests/test_qwen_media.py` checks the
receipt fields.
`tests/test_regional_length.py` (new) checks `SOL_H3_DYNAMIC_TEXT_LENGTH` on the CPU. On a small stand-in model: the
graphs it compiles over several prompt lengths (also with static compiles), outputs equal to the uncompiled model,
the tile-buffer rules and the FastVideo signatures it relies on. On FastVideo's own H3 blocks (skipped when FastVideo
can't be imported): what `install()` and `install_dense()` wrap with the setting on and off.
`tests/test_reshape.py` (new) covers reshape: stand-in workers check that Stage 1, Stage 2 and the prompt encoder each
see the shape of every request; the Stage 2 globals are compared with the shared module's own import code for every
length and orientation. It also covers the capture's shape check and PCM trim, the BSA self-check per shape, the
keyframe canvas, the static compiles (with the torch behaviour they rely on, a marked dimension included), a
pipeline that is closing or failed a reshape, a close during a reshape that waits for a loading encoder, a close
while a reshape closes the encoder, a wait for a loading encoder that is interrupted, the resident encoder's restart
(the minimum of item 4 for the new length, the session's prompt-token mode whether given to `Pipeline` or read from
`SOL_H3_PROMPT_TOKENS`, two reshapes without a clip between them, never two resident encoders), a reshape before the
first clip, and the `SPARK_DEBUG_AUTOTUNE` lines.
Settings: `PipelineReshapeTests` clears the 8 settings it reads (`SOL_H3_RESIDENT_QWEN_EARLY`,
`SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB`, `SOL_H3_RESIDENT_QWEN_MAX_FRAMES`, `SOL_H3_PROMPT_TOKENS`,
`SOL_H3_STATIC_SHAPES`, `SOL_H3_DYNAMIC_TEXT_LENGTH`, `SOL_H3_ORIENTATION`, `SOL_H3_SECONDS`), and `QwenReshapeTests`
clears `SOL_H3_ORIENTATION` and `SOL_H3_SECONDS`. `UserSettingsTests` runs both classes again with a user's own value
of each of the 8. `tests/test_pipeline.py` clears the three `SOL_H3_RESIDENT_QWEN_*` settings in its tests of where
the resident encoder starts. Other tests still read `SOL_H3_ORIENTATION` and `SOL_H3_SECONDS`, so
`install/10-sana.sh` runs the suite without the settings file's switches. 133 tests in total (upstream: 65).

## All switches

| Variable | Default | Effect |
|---|---|---|
| `SOL_H3_ORIENTATION` | landscape | `infer.py` default for `--orientation` (the web server ignores it) |
| `SOL_H3_SECONDS` | unset (5 s) | `infer.py` default for `--seconds` (the web server ignores it) |
| `SOL_H3_RESIDENT_QWEN_MAX_FRAMES` | 241 | longer outputs load the prompt encoder per request |
| `SOL_H3_RESIDENT_QWEN_EARLY` | 1 | 0 = the resident prompt encoder loads after the first clip, not during it |
| `SOL_H3_RESIDENT_QWEN_EARLY_MIN_GIB` | 32 up to 5 s, 34 above | MemAvailable needed to load it during the first clip, or right after a reshape back to 10 s or less (else the next clip loads its own first) |
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
| `SOL_H3_PROMPT_TOKENS` | minimax | `plain` = as up to 0.3: the pinned ComfyUI splits MiniMax's 7 prompt tokens (`<d>`, `</d>`, ...) into text tokens; `minimax` reads each as one token, as in H3's training |
| `SOL_H3_DYNAMIC_TEXT_LENGTH` | 0 | 1 = Stage 1's body blocks compile per session, not per prompt length (experimental, [17](#17-prompt-lengths-without-recompiles-experimental)) |
| `SOL_H3_STATIC_SHAPES` | 1 | 0 = upstream `torch.compile(dynamic=None)`: after a size change a graph may get dynamic sizes ([18](#18-reshape-another-length-or-orientation-without-a-reload)) |
| `SOLH3_RESHAPE` (studio) | 1 | 0 = another length or orientation closes the session and loads the models again |
| `MIMALLOC_PURGE_DELAY` | 0 (workers) | any value you set wins |

The `SPARK_*` names are kept from the patch's first version so existing setups keep working.

## License of the changes

The files the patch modifies are Apache-2.0 NVlabs/Sana code; `runtime/qwen_ops/media.py` also contains
FastVideo-derived Apache-2.0 code (see Sol-H3-Spark's `THIRD_PARTY_NOTICES.md`). The changes and the new files are
Apache-2.0, like the rest of this repository.
