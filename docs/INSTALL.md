# Install

Step by step on one DGX Spark. Times and sizes were measured on a DGX Spark; yours will differ, mostly in the
download. A dash means not measured.

Status: these steps were first run by hand on one DGX Spark (DGX OS = Ubuntu 24.04 aarch64, driver 580.178.04,
CUDA 13.0); the scripts in `install/` automate them.

<!-- validation-status -->
Validation of the install scripts of 0.1-0.2.1 on a DGX Spark (2026-10-09/10), into an empty `SOLH3_HOME` with an
empty pip cache: steps 00 to 90 ran end to end, and the 69 CPU tests of the patch of that date passed with both
freshly built Pythons, whose pinned packages match the environments
the measurements in [PERFORMANCE.md](PERFORMANCE.md) came from. Step 80 built a prompt cache bit-identical to the
one the measurements used. On that install the first clip (t2va, 16:9, 5 s) took 213 s, including the one-time
kernel compile; the next clip 65 s; after a restart, a cold start with the filled compile cache 104 s. That run
reused weights already on the box (step 20 only verified them) and Docker's build cache for NVIDIA's base image.
0.3 changed `install/10-sana.sh` and the patch (now 74 tests); its update path was checked in scratch clones, not by
a fresh install on hardware.
<!-- /validation-status -->

NVIDIA notes that a clean installation of Sol-H3-Spark has not been validated upstream; the pins here are the
versions that worked on that box.

## What you need

- One DGX Spark (GB10). The pipeline needs nearly all of its memory while it runs: stop other GPU apps first.
- DGX OS with CUDA 13 at `/usr/local/cuda`, `python3.12` with `python3.12-venv`, `git`, `curl`, `ffmpeg`
  (`sudo apt install python3.12-venv ffmpeg`), Docker with the NVIDIA runtime (DGX OS has it) and your user in
  the `docker` group.
- ~300 GB free disk: ~224 GB of weights (all three modes plus the one-time offline weights), ~14 GB of Python
  environments and sources, ~21 GB for the Docker image, ~6 GB of pip downloads in `~/.cache/pip` (safe to delete
  afterwards), plus room for outputs and caches. If you reuse weights you
  already have (`SOLH3_CHECKPOINTS`, see below), ~60 GB.
- A Hugging Face account that has accepted the terms of the gated
  [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) repo, and a read token for it. You log in once,
  before step 00 (see [Steps](#steps)). Read the model terms in the [README](../README.md#model-terms) first.

## Where things go

Everything goes under `$SOLH3_HOME` (default `~/sol-h3`). To use other places, copy `bin/env.example` to
`~/.config/sol-h3-studio/env` and edit it before you start; every script reads it. Variables set in your shell
win over the file.

```text
~/sol-h3/
  Sana/                  NVlabs/Sana sol-engine + patches/sol-h3-studio.patch
  checkpoints/           model weights (SOLH3_CHECKPOINTS; point it at an existing directory to reuse weights)
  sources/               pinned FastVideo, flash-attention, ComfyUI, LTX-2, H3 upscaler
  envs/{download,stage1,stage2}
  dependencies/{fa4,offline-context}
  runtime/               SOL_H3_SPARK_RUNTIME_ROOT: uploads, outputs, compile cache, trash (deleted clips)
  paths-{t2va,fl2va,ref2va}.json, existing-paths.json, fixed-prompt.pt, logs/
  applied.patch          up to 0.2: the patch install/10-sana.sh applied last (now kept in the checkout's git directory)
```

Every script is idempotent (it skips work that is done; `FORCE=1` redoes it) and accepts `--dry-run`, which
prints the commands instead of running them.

## Steps

```bash
git clone https://github.com/slogonomo/sol-h3-studio ~/src/sol-h3-studio
cd ~/src/sol-h3-studio
```

Log in to Hugging Face once, before step 00 (00 checks the login, 20 downloads with it). The login lives in
`envs/download`, the venv that `install/20-weights.sh` uses for NVIDIA's downloader (paths for the default
`SOLH3_HOME`):

```bash
python3.12 -m venv ~/sol-h3/envs/download
~/sol-h3/envs/download/bin/pip install huggingface-hub==1.29.0    # the version NVIDIA's requirements/download.txt pins
~/sol-h3/envs/download/bin/hf auth login
```

Instead of `hf auth login` you can `export HF_TOKEN=<your read token>` in the shell that runs the scripts.

To install fewer than all three modes, set `SOLH3_TASKS` (for example `SOLH3_TASKS="t2va"`) before step 20, for
the whole install: export it, or put it in `~/.config/sol-h3-studio/env`. Steps 20 and 70 both read it; 70 fails
for a mode whose weights were not downloaded.

| Step | What it does | Time / size measured |
|---|---|---|
| `install/00-check.sh` | Checks aarch64, GB10, driver, CUDA 13, python3.12, git, curl, ffmpeg, docker, disk, HF access to LTX-2.5. Changes nothing. | - |
| `install/10-sana.sh` | Clones NVlabs/Sana `sol-engine` at `patches/BASE_COMMIT`, applies the patch (and keeps a copy for [updates](#updating)), runs the 89 CPU tests once the Stage1 env exists. | - |
| `install/20-weights.sh` | Downloads the pinned weights for t2va, fl2va and ref2va with NVIDIA's `download_checkpoints.py --include-offline --verify-sha256`. `SOLH3_TASKS` limits it (see above). Resumable. | ~48 min at ~80 MB/s, ~224 GB |
| `install/30-sources.sh` | Fetches the pinned source trees with `prepare.py --fetch-sources`. It ends with "Missing required inputs", which is expected; the script checks the revisions itself. | ~12 s, ~0.5-0.8 GB |
| `install/40-stage1.sh` | Stage1 venv (torch 2.12+cu130, FastVideo, fastvideo-kernel built for SM121, cuDNN 9.20) and the isolated FA4 target. | 5-22 min (5.5 min on a fast connection, 22 min in the first install), ~7.6 GB with FA4 |
| `install/50-stage2.sh` | Stage2 venv (torch 2.13+cu132, LTX-2, ltx-kernels built for SM120/121), the torchaudio CUDA-check fix, the offline-context target. | ~6 min, ~6.2 GB |
| `install/60-qwen-image.sh` | Docker: NVIDIA's `Dockerfile.qwen` as `sol-h3-spark-qwen:base`, then this repo's `qwen-image/` layer as `sol-h3-spark-qwen`. | ~20 min (mostly pulling NGC PyTorch 25.11), ~21 GB |
| `install/70-manifests.sh` | `prepare.py` once per mode in `SOLH3_TASKS`, writing `paths-<task>.json` with `qwen_python` in the patched checkout. Also applies NVIDIA's FastVideo loader patch. | - |
| `install/80-prompt-cache.sh` | One-time Stage2 prompt cache (`fixed-prompt.pt`) with NVIDIA's `runtime.cache_builder`. Reads ~37 GB of INT8 weights and uses the GPU for ~20 s. | 16.7 s, 12.6 MB |
| `install/90-env.sh` | Writes `~/.config/sol-h3-studio/env` and runs `bin/sol-h3-studio --check`. | - |

Run them in order. Don't run 40 and 50 at the same time: two `nvcc` builds at once can run the box out of memory.
After 40, run `install/10-sana.sh` again to run the CPU tests with the new env.

After step 80 the two offline weight files (~37 GB, INT8 Gemma and INT8 connector) are only needed to rebuild the
prompt cache; the script prints them so you can delete them. `20-weights.sh` would download them again.

## Start

```bash
~/src/sol-h3-studio/bin/sol-h3-studio
```

It checks the setup, refuses to start below 110 GiB MemAvailable (`SOLH3_MIN_FREE_GIB`), starts the web server and
prints `>>> STATUS:` lines. Open <http://localhost:8190>. The server listens on 127.0.0.1; from a laptop use
NVIDIA Sync ([nvidia-sync.md](nvidia-sync.md)) or an SSH tunnel:

```bash
ssh -L 8190:127.0.0.1:8190 you@spark-xxxx.local
```

To reach it from the LAN directly, set `SOLH3_HOST=0.0.0.0`. There is no login, so do that only on a network you
trust.

The first clip of each new mode, format and length loads the models (5 s clips: 104-113 s for t2va in the studio,
99 s in the `infer.py` bench; ~115 s for fl2va and ~180 s for ref2va with a video reference, both measured with
`infer.py`). The very first run of a configuration also compiles its kernels (see
[PERFORMANCE.md](PERFORMANCE.md#known-costs-not-fixed)); after that the compile cache in
`$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` makes cold starts faster. Ctrl-C stops the server, its workers and the
Qwen container.

Other ways to run it: as an NVIDIA Sync custom app ([nvidia-sync.md](nvidia-sync.md)), as a systemd user service
(`integrations/systemd/sol-h3-studio.service`), or on demand with
[spark-switchboard](https://github.com/slogonomo/spark-switchboard): copy
`integrations/spark-switchboard/sol-h3-studio.toml` into switchboard's `services/` directory. It sets the pidfile
(`SOLH3_PIDFILE`) through which switchboard finds the studio again after its own restart, and `--kill` as the
command that removes leftovers.

## Updating

```bash
cd ~/src/sol-h3-studio && git pull
install/10-sana.sh
```

Then restart the studio right away and reload any open studio page (also on other devices). The server reads the
page on every request, so between the pull and the restart a reloaded page and the running server don't match; and
a page from before an update can show newer job states wrongly (a 0.2 page shows cancelled clips as still running).

`install/10-sana.sh` keeps a copy of the patch it applied with the checkout (`sol-h3-studio-applied.patch` in its
git directory, so each clone and git worktree has its own; versions up to 0.2 kept one in `$SOLH3_HOME/applied.patch`).
When the repo brings a new patch, it finds the earlier patch the checkout carries (that copy, the 0.2 location, or
an earlier version of `patches/sol-h3-studio.patch` from this repo's history), checks that the checkout holds it and
nothing else, reverses it, checks out the new `patches/BASE_COMMIT` if that changed, and applies the new patch. If the checkout
has other local changes it lists them and stops without changing anything; move them away, or move `$SOLH3_SANA`
away and run the script again for a fresh clone.

If spark-switchboard runs the studio, bring your copy in switchboard's `services/` directory (whatever you named it)
up to date when `integrations/spark-switchboard/sol-h3-studio.toml` changes: for 0.3, add `"cancelled"` to its
`idle_values` (the busy probe must count the new status as idle), or replace its contents with the new file and
keep its file name and your path edits. Then `spark-switchboard check`, and restart switchboard.

A checkout you patched by hand ([PATCH.md](PATCH.md#apply-test-revert)) with a released version of the patch is
found the same way. Any other patch: reverse it with the file you applied (`git -C "$SOLH3_SANA" apply -R <that
patch>`), or re-clone as above.

The compile cache is keyed on code and versions, so it stays valid; delete `$SOL_H3_SPARK_RUNTIME_ROOT/cache` to
start clean.

From 0.1 to 0.2: the first start imports the clip history (`jobs.json`) into `jobs.db` in the same directory
(see "Where the clip history is" in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#while-it-runs)). Back up
`$SOLH3_STATE_DIR` first if you want a copy.

## Uninstall

Stop the studio, then remove `~/sol-h3` (or your `$SOLH3_HOME`), `~/.config/sol-h3-studio`,
`~/.local/state/sol-h3-studio` and the images: `docker rmi sol-h3-spark-qwen sol-h3-spark-qwen:base`.
