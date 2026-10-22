# Install

Step by step on one DGX Spark. Times and sizes were measured on a DGX Spark; yours will differ, mostly in the
download. A dash means not measured. For a workstation with one RTX 5090, see [RTX 5090](#rtx-5090) below: the same
scripts with `LFSTUDIO_BACKEND=rtx5090`.

Status: these steps were first run by hand on one DGX Spark (DGX OS = Ubuntu 24.04 aarch64, driver 580.178.04,
CUDA 13.0); the scripts in `install/` automate them.

<!-- validation-status -->
Validation of the install scripts of 0.1-0.2.1 on a DGX Spark (2026-10-09/10), into an empty `LFSTUDIO_HOME` with an
empty pip cache: steps 00 to 90 ran end to end, and the 69 CPU tests of the patch of that date passed with both
freshly built Pythons, whose pinned packages match the environments
the measurements in [PERFORMANCE.md](PERFORMANCE.md) came from. Step 80 built a prompt cache bit-identical to the
one the measurements used. On that install the first clip (t2va, 16:9, 5 s) took 213 s, including the one-time
kernel compile; the next clip 65 s; after a restart, a cold start with the filled compile cache 104 s. That run
reused weights already on the box (step 20 only verified them) and Docker's build cache for NVIDIA's base image.
0.3 changed `install/10-sana.sh` and the patch (74 tests then). 0.4 changes the patch again (135 tests) and how
`install/10-sana.sh` runs its tests (without the settings file's switches). Their update paths were checked in
scratch clones, not by a fresh install on hardware.
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
  already have (`LFSTUDIO_CHECKPOINTS`, see below), ~60 GB.
- A Hugging Face account that has accepted the terms of the gated
  [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) repo, and a read token for it. You log in once,
  before step 00 (see [Steps](#steps)). Read the model terms in the [README](../README.md#model-terms) first.

## Where things go

Everything goes under `$LFSTUDIO_HOME` (default `~/sol-h3`). To use other places, copy `bin/env.example` to
`~/.config/lf-studio/env` and edit it before you start; every script reads it. Variables set in your shell
win over the file.

```text
~/sol-h3/
  Sana/                  NVlabs/Sana sol-engine + patches/lf-studio.patch
  checkpoints/           model weights (LFSTUDIO_CHECKPOINTS; point it at an existing directory to reuse weights)
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
git clone https://github.com/slogonomo/lf-studio ~/src/lf-studio
cd ~/src/lf-studio
```

Log in to Hugging Face once, before step 00 (00 checks the login, 20 downloads with it). The login lives in
`envs/download`, the venv that `install/20-weights.sh` uses for NVIDIA's downloader (paths for the default
`LFSTUDIO_HOME`):

```bash
python3.12 -m venv ~/sol-h3/envs/download
~/sol-h3/envs/download/bin/pip install huggingface-hub==1.29.0    # the version NVIDIA's requirements/download.txt pins
~/sol-h3/envs/download/bin/hf auth login
```

Instead of `hf auth login` you can `export HF_TOKEN=<your read token>` in the shell that runs the scripts.

To install fewer than all three modes, set `LFSTUDIO_TASKS` (for example `LFSTUDIO_TASKS="t2va"`) before step 20, for
the whole install: export it, or put it in `~/.config/lf-studio/env`. Steps 20 and 70 both read it; 70 fails
for a mode whose weights were not downloaded.

| Step | What it does | Time / size measured |
|---|---|---|
| `install/00-check.sh` | Checks aarch64, GB10, driver, CUDA 13, python3.12, git, curl, ffmpeg, docker, disk, HF access to LTX-2.5. Changes nothing. | - |
| `install/05-python.sh` | RTX 5090 only; on a DGX Spark it does nothing (the install uses the system's `python3.12`). | - |
| `install/10-sana.sh` | Clones NVlabs/Sana `sol-engine` at `patches/BASE_COMMIT`, applies the patch (and keeps a copy for [updates](#updating)), runs the 135 CPU tests once the Stage1 env exists. | - |
| `install/20-weights.sh` | Downloads the pinned weights for t2va, fl2va and ref2va with NVIDIA's `download_checkpoints.py --include-offline --verify-sha256`. `LFSTUDIO_TASKS` limits it (see above). Resumable. | ~48 min at ~80 MB/s, ~224 GB |
| `install/30-sources.sh` | Fetches the pinned source trees with `prepare.py --fetch-sources`. It ends with "Missing required inputs", which is expected; the script checks the revisions itself. | ~12 s, ~0.5-0.8 GB |
| `install/40-stage1.sh` | Stage1 venv (torch 2.12+cu130, FastVideo, fastvideo-kernel built for SM121, cuDNN 9.20) and the isolated FA4 target. | 5-22 min (5.5 min on a fast connection, 22 min in the first install), ~7.6 GB with FA4 |
| `install/50-stage2.sh` | Stage2 venv (torch 2.13+cu132, LTX-2, ltx-kernels built for SM120/121), the torchaudio CUDA-check fix, the offline-context target. | ~6 min, ~6.2 GB |
| `install/60-qwen-image.sh` | Docker: NVIDIA's `Dockerfile.qwen` as `sol-h3-spark-qwen:base`, then this repo's `qwen-image/` layer as `sol-h3-spark-qwen`. | ~20 min (mostly pulling NGC PyTorch 25.11), ~21 GB |
| `install/70-manifests.sh` | `prepare.py` once per mode in `LFSTUDIO_TASKS`, writing `paths-<task>.json` with `qwen_python` in the patched checkout. Also applies NVIDIA's FastVideo loader patch. | - |
| `install/80-prompt-cache.sh` | One-time Stage2 prompt cache (`fixed-prompt.pt`) with NVIDIA's `runtime.cache_builder`. Reads ~37 GB of INT8 weights and uses the GPU for ~20 s. | 16.7 s, 12.6 MB |
| `install/90-env.sh` | Writes `~/.config/lf-studio/env` and runs `bin/lf-studio --check`. | - |

Run them in order. Don't run 40 and 50 at the same time: two `nvcc` builds at once can run the box out of memory.
After 40, run `install/10-sana.sh` again to run the CPU tests with the new env.

After step 80 the two offline weight files (~37 GB, INT8 Gemma and INT8 connector) are only needed to rebuild the
prompt cache; the script prints them so you can delete them. `20-weights.sh` would download them again.

## Start

```bash
~/src/lf-studio/bin/lf-studio
```

It checks the setup, refuses to start below 110 GiB MemAvailable (`LFSTUDIO_MIN_FREE_GIB`), starts the web server and
prints `>>> STATUS:` lines. Open <http://localhost:8190>. The server listens on 127.0.0.1; from a laptop use
NVIDIA Sync ([nvidia-sync.md](nvidia-sync.md)) or an SSH tunnel:

```bash
ssh -L 8190:127.0.0.1:8190 you@spark-xxxx.local
```

To reach it from the LAN directly, set `LFSTUDIO_HOST=0.0.0.0`. There is no login, so do that only on a network you
trust.

Scripts and agents use the same port ([API.md](API.md)). On the Spark, `bin/lf-studio generate ...` (and
`wait`, `status`, `cancel`, `fetch`, `guide`, `mcp`) runs the client with this env file's settings. On another
machine the client is one file (Python 3.9 or later), pointed at the studio with `LFSTUDIO_URL`:

```bash
mkdir -p ~/.local/bin && curl -fsSLo ~/.local/bin/lf-cli \
  https://raw.githubusercontent.com/slogonomo/lf-studio/main/bin/lf-cli && chmod +x ~/.local/bin/lf-cli
LFSTUDIO_URL=http://<spark-host>:8190 lf-cli status
```

The first clip of each new mode loads the models (5 s clips: 104-113 s for t2va in the studio, 99 s in the
`infer.py` bench; ~115 s for fl2va and ~180 s for ref2va with a video reference, both measured with `infer.py`).
Another format or length of the loaded mode keeps them loaded; its first clip compiles for the new shape (up to 10 s
more than a warm clip in the staging check of 0.4, up to 27 s in an earlier run with `SOL_H3_DYNAMIC_TEXT_LENGTH` off;
[PERFORMANCE.md](PERFORMANCE.md#changing-length-or-format)). With `LFSTUDIO_RESHAPE=0` any change loads the models. The
very first run of a configuration also compiles its kernels (see
[PERFORMANCE.md](PERFORMANCE.md#known-costs-not-fixed)); after that the compile cache in
`$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` makes cold starts faster. Ctrl-C stops the server, its workers and the Qwen
container.

Other ways to run it: as an NVIDIA Sync custom app ([nvidia-sync.md](nvidia-sync.md)), as a systemd user service
(`integrations/systemd/lf-studio.service`), or on demand with
[spark-switchboard](https://github.com/slogonomo/spark-switchboard): copy
`integrations/spark-switchboard/lf-studio.toml` into switchboard's `services/` directory. It sets the pidfile
(`LFSTUDIO_PIDFILE`) through which switchboard finds the studio again after its own restart, `--kill` as the
command that removes leftovers, and the paths whose requests never start a stopped studio (`passive_paths`: the
health check, and the state, clip records, waits and posters that scripts read).

## RTX 5090

The same studio on a workstation with one GeForce RTX 5090 (32 GB), x86_64 Linux, with NVIDIA's
[Sol-H3-RTX5090](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-RTX5090) package and its own
patch, `patches/lf-studio-rtx5090.patch` against `patches/BASE_COMMIT_RTX5090`
([RTX5090.md](RTX5090.md)). No Docker, no CUDA toolkit and no system Python are needed: the Qwen prompt encoder runs
natively in the Stage1 env, `nvcc` comes from pip, and both Pythons come from [uv](https://github.com/astral-sh/uv).
The install steps use no sudo.

What you need:

- The NVIDIA driver 580 or newer, and `git`, `curl`, `tar`, `sha256sum`, `ffmpeg` and `gcc` from your distribution
  (installed with its package manager, with sudo, before step 00).
- 64 GB of RAM. Disk, for all three modes: ~220 GB of weights, ~180 GB of files derived from them next to the
  checkpoints (the LTX block folder 42 GB, one H3 snapshot per mode 20-24 GB each, ~66 GB of scratch while a snapshot is
  built), and ~40 GB under `~/sol-h3` ([RTX5090.md](RTX5090.md)).
- A DGX Spark's `fixed-prompt.pt` (the prompt cache, below): it can't be built on the RTX 5090.
- A Hugging Face login whose account accepted the Lightricks/LTX-2.5 terms. The login step above uses
  `python3.12`, which the RTX 5090 path doesn't need: after step 05, make the download venv with uv's Python instead,
  `~/sol-h3/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12 -m venv ~/sol-h3/envs/download`, then the same
  `pip install` and `hf auth login`; or `export HF_TOKEN=<your read token>`. Not needed if you copy the weights from
  a Spark (below).

Set `LFSTUDIO_BACKEND=rtx5090` for the whole install: `export LFSTUDIO_BACKEND=rtx5090` in the shell that runs the steps
(or put it in `~/.config/lf-studio/env` before step 00). `install/00-check.sh` says so if the machine looks like an
RTX 5090 and it is not set, and the other steps refuse to run with a backend that doesn't match the machine.
`install/90-env.sh` writes it into a new settings file; an existing settings file is kept as it is, so add the line
there yourself (or rewrite the file with `FORCE=1 install/90-env.sh`). The steps, in this order (80 comes before 70
here):

| Step | On the RTX 5090 | Time / size measured |
|---|---|---|
| `install/00-check.sh` | Checks x86_64, the RTX 5090, driver 580 or newer, git, curl, tar, sha256sum, ffmpeg, gcc, disk, HF access. | - |
| `install/05-python.sh` | uv 0.13.1 (the release binary, SHA-256 pinned) into `~/sol-h3/bin`, CPython 3.12.15 and 3.13.16 into `~/sol-h3/python`. | under 1 min, ~270 MB (~230 MB of it the Pythons) |
| `install/10-sana.sh` | Sana at `patches/BASE_COMMIT_RTX5090` (`801f040`), `patches/lf-studio-rtx5090.patch`, its 201 CPU tests. | - |
| `install/20-weights.sh` | The same weights and revisions as the Spark's, without the offline weights (the prompt cache comes from a Spark). | - (or a copy, below) |
| `install/30-sources.sh` | The same pinned sources (FastVideo, flash-attention, ComfyUI v0.30.0, LTX-2, H3 upscaler). | ~1 min |
| `install/40-stage1.sh` | Stage1 venv with Python 3.12.15: torch 2.12+cu130, FastVideo, ComfyUI's requirements, comfy-kitchen (the native Qwen encoder), no fastvideo-kernel; the FA4 target. | ~25 min, most of it downloads at ~9 MB/s; ~7.7 GB |
| `install/50-stage2.sh` | Stage2 venv with Python 3.13.16: torch 2.13+cu132, LTX-2, ltx-kernels built for SM120; a GCC 14 toolchain for that build if the system's gcc is newer than 15. | ~10 min, the ltx-kernels build ~3 min of it; ~5.9 GB, toolchain ~1.2 GB |
| `install/60-qwen-image.sh` | Nothing: says so and exits. | - |
| `install/80-prompt-cache.sh` | Takes the prompt cache from a DGX Spark and checks it (see below). | seconds, 12.6 MB |
| `install/70-manifests.sh` | `prepare.py` per mode, `--python-qwen` = the Stage1 python. Needs the prompt cache. | - |
| `install/90-env.sh` | Writes the settings file with `LFSTUDIO_BACKEND=rtx5090`, then `bin/lf-studio --check`. | - |

Measured on a workstation with one RTX 5090, 64 GB of RAM and one NVMe disk, Arch Linux, driver 615.78.08,
2026-10-16.

**Weights from a DGX Spark.** The checkpoints are the same files and revisions on both machines, so instead of
step 20 you can copy a Spark's `$LFSTUDIO_CHECKPOINTS` (or point `LFSTUDIO_CHECKPOINTS` at a shared copy) and then run
step 20, which verifies them and downloads only what is missing. Copying the 220 GB (205 GiB) from a Spark over 1 GbE
took 47 min. With `LFSTUDIO_CHECKPOINTS` pointing at a shared copy, the derived files (`derived/`, `scratch/`) are written
next to it.

**The prompt cache.** NVIDIA's cache builder requires an SM121 GPU (a DGX Spark), so step 80 cannot build the cache
on the RTX 5090. Copy `fixed-prompt.pt` from a Spark install (`~/sol-h3/fixed-prompt.pt` there; its step 80 builds
it), then:

```bash
LFSTUDIO_PROMPT_CACHE_FROM=<the copy> install/80-prompt-cache.sh
```

It checks the file with the package's own recipe check (`runtime/prompt_cache.py`: prompt, recipe fingerprint, shapes,
dtypes, finite values; on the CPU) and copies it to `$LFSTUDIO_HOME/fixed-prompt.pt`. A file that fails the check is
refused, and so is a missing cache. The cache builder and the recipe check are byte-identical in both packages.

**ltx-kernels and GCC.** `nvcc` 13.2 accepts host compilers up to GCC 15, and `nvcc` 13.4 rejects torch's CUDA 13.2
headers. With a newer system gcc (Arch ships GCC 16), step 50 installs micromamba 2.9.0 (static binary, SHA-256
pinned) into `~/sol-h3/bin` and GCC 14 from conda-forge into `~/sol-h3/toolchains/gcc14`, and builds with it
([TROUBLESHOOTING.md](TROUBLESHOOTING.md#loading-fails)). With GCC 15 or older no toolchain is installed.

**Start.** `bin/lf-studio` as on the Spark. It skips the Docker checks, and refuses to start below
`LFSTUDIO_MIN_FREE_GIB` of free RAM (default 42 GiB on the RTX 5090) or `LFSTUDIO_MIN_FREE_VRAM_GIB` of free GPU memory
(default 25 GiB, from `nvidia-smi`), measured on every mode ([RTX5090.md](RTX5090.md)); 0 turns
a check off. The page says "on one RTX 5090", with clip times measured there at 3, 5, 10 and 15 s (other lengths
interpolated).

Under `~/sol-h3` the RTX 5090 install also has `bin/` (uv, uvx, micromamba), `python/` (uv's CPythons),
`toolchains/` (the GCC 14 toolchain, only with a newer system gcc), `cache/downloads/` (the checked downloads) and
`cache/pip`, `cache/uv` (pip's cache, ~6.3 GB of wheels after steps 40 and 50, and uv's, a few KB; `PIP_CACHE_DIR`,
`UV_CACHE_DIR`), so nothing is written outside it and a rerun does not download the wheels again. Keep `toolchains/gcc14`: the ltx-kernels extensions carry an RPATH to its `lib/` and
load its libstdc++ at run time. `FORCE=1` keeps it; `FORCE=1 FORCE_TOOLCHAIN=1 install/50-stage2.sh` installs it
again (the old one is moved to `cache/` first) and rebuilds the extensions against it.

**An install made by hand.** Steps 40 and 50 count as done when the venv has the marker `.lf-studio-done` (until 0.6
also 0.4's `.sol-h3-studio-done`). A venv without it whose `pip freeze` matches
`install/freeze/stage1-rtx5090-freeze.txt` or `stage2-rtx5090-freeze.txt` exactly (one built by hand to the same recipe)
is adopted (the lines of Pillow, Gradio and MoviePy don't count): nothing is installed or rebuilt, apart from the
Pillow step that every run of 40 and 50 ends with (see [From 0.4 to 0.5](#from-04-to-05)), and the marker is written
once the step's import check passes. A venv that
differs in any line is built as usual (pip installs into it again, and step 50 rebuilds ltx-kernels); if you know such a
venv is right (for example one with an extra tool such as py-spy), mark it done yourself:
`touch ~/sol-h3/envs/stage1/.lf-studio-done`.

## Updating

```bash
cd ~/src/lf-studio && git pull
install/10-sana.sh
```

Then restart the studio right away and reload any open studio page (also on other devices). The server reads the
page on every request, so between the pull and the restart a reloaded page and the running server don't match; and
a page from before an update can show newer job states wrongly (a 0.2 page shows cancelled clips as still running).

`install/10-sana.sh` keeps a copy of the patch it applied with the checkout (`lf-studio-applied.patch` in its git
directory, so each clone and git worktree has its own; 0.4 called it `sol-h3-studio-applied.patch`, which it still finds
until 0.6; versions up to 0.2 kept one in `$LFSTUDIO_HOME/applied.patch`). When the repo brings a new patch, it finds
the earlier patch the checkout carries (that copy, the 0.2 location, or an earlier version of `patches/lf-studio.patch`
from this repo's history, also under its 0.4 name `patches/sol-h3-studio.patch`), checks that the checkout holds it and
nothing else, reverses it, checks out the new `patches/BASE_COMMIT` if that changed, and applies the new patch. If the
checkout has other local changes it lists them and stops without changing anything; move them away, or move
`$LFSTUDIO_SANA` away and run the script again for a fresh clone.

If spark-switchboard runs the studio, bring your copy in switchboard's `services/` directory (whatever you named it) up
to date when `integrations/spark-switchboard/lf-studio.toml` changes: for 0.3, add `"cancelled"` to its `idle_values`
(the busy probe must count the new status as idle); for 0.4, set its `passive_paths` to `["/health", "/api/state",
"/api/jobs", "/api/jobs/*", "/poster/*"]` (without it, a script that reads a clip's state or waits for it starts a
stopped studio); for 0.5, see below. Or replace its contents with the new file, keep its file name, and set your
checkout's path in `command` and `kill_command` (the file assumes `~/src/lf-studio`). Then `spark-switchboard check`,
and restart switchboard. A config made from switchboard's own example (in its `examples/services/`) needs the same
`passive_paths` while the example lists only `"/health"`.

A checkout you patched by hand ([PATCH.md](PATCH.md#apply-test-revert)) with a released version of the patch is
found the same way. Any other patch: reverse it with the file you applied (`git -C "$LFSTUDIO_SANA" apply -R <that
patch>`), or re-clone as above.

The compile cache is keyed on code and versions, so it stays valid; delete `$SOL_H3_SPARK_RUNTIME_ROOT/cache` to
start clean.

On the RTX 5090 the same: `install/10-sana.sh` (with `LFSTUDIO_BACKEND=rtx5090` in the settings file) finds the earlier
version of `patches/lf-studio-rtx5090.patch`, reverses it, moves to a new `patches/BASE_COMMIT_RTX5090` if that
changed, and applies the new one. A new patch file is all an update of the 5090 patch needs.

### From 0.4 to 0.5

0.5 renames the project to LF Studio: the launcher is `bin/lf-studio`, the client and MCP server `bin/lf-cli` (`lf-cli
mcp`), the settings `LFSTUDIO_*` (`SOLH3_*` before), the settings file `~/.config/lf-studio/env`, the state
`~/.local/state/lf-studio`. NVIDIA's names stay: the data home `~/sol-h3`, the `SOL_H3_*` switches, the Qwen image and
its containers.

Run `install/10-sana.sh` after the pull: the patch is now `patches/lf-studio.patch` (`patches/lf-studio-rtx5090.patch`
on the RTX 5090), its changes marked "lf-studio". The script replaces the 0.4 patch in the checkout as for any update
and keeps its copy as `lf-studio-applied.patch`.

Then run `install/40-stage1.sh --pillow-only` and `install/50-stage2.sh --pillow-only`, and on a DGX Spark also
`install/60-qwen-image.sh`, for Pillow 13.0.0 (#69). The image-decoder advisories of 2026-10-22 affect every earlier Pillow
([GHSA-v9pv-5gc8-w2m6](https://github.com/advisories/GHSA-v9pv-5gc8-w2m6), CVE-2026-96639, and others), and the runtime
decodes the keyframes and reference images clients upload in the Stage 1 env and, on a DGX Spark, in the Qwen image.
With `--pillow-only`, 40 and 50 change nothing else in the env that is there. Without it, an env that lacks the done
marker (one built before the markers, or by hand) would be built into again.
- 40 uninstalls Gradio, gradio-client and MoviePy (they capped Pillow below 12, and nothing in the studio or the
  runtime imports them) and installs Pillow 13.0.0;
- 50 installs Pillow 13.0.0;
- 60 rebuilds the studio's layer of the Qwen image on the base it keeps.

Together they take a few minutes. Afterwards `pip check` in the Stage 1 env also lists FastVideo's Gradio and MoviePy
as missing, next to what it listed before (flashinfer-python; on the RTX 5090 also fastvideo-kernel); that is expected. A new install gets Pillow 13.0.0 and no Gradio or MoviePy from the start.

New and optional: `LFSTUDIO_API`, on (`1`) by default as in 0.4; `0` turns the API for programs off and leaves the
studio to its page ([API.md](API.md#turning-the-api-off)).

Nothing else needs to change for a working install. Until 0.6:

- `bin/sol-h3-studio` and `bin/sol-h3` run the new files with the same arguments (after a note on stderr), so NVIDIA
  Sync apps, systemd units, spark-switchboard services, scripts and MCP registrations (`sol-h3 mcp`) keep working. A
  copy of the 0.4 `bin/sol-h3` on another machine works with the new studio as it is; downloaded again from its 0.4
  address, it is only the shim, which says where to get `bin/lf-cli`.
- Every setting is also read by its 0.4 name, from the settings file and from the environment, with one warning that
  lists the old names used. Where both names are set, the new one wins; the environment still wins over the file.
- `~/.config/sol-h3-studio/env` is used while `~/.config/lf-studio/env` does not exist, `~/.local/state/sol-h3-studio`
  while it has the clip history (`jobs.db`) and `~/.local/state/lf-studio` does not, and `~/.cache/sol-h3-studio` (the
  compile cache when the runtime root is unset) and the MCP server's `./sol-h3-clips` while the new one is missing or
  empty, each with a note naming both. Nothing is moved; a new install uses the new places.
- Venvs that 0.4 marked as built (`.sol-h3-studio-done`) count as built, and the page takes over the settings it kept in
  the browser (`solh3.*`).
- `integrations/spark-switchboard/sol-h3-studio.toml` and `integrations/systemd/sol-h3-studio.service` stay for this
  release as they were in 0.4, marked as deprecated.

Change these when convenient, before 0.6:

- The settings' names, in the settings file and wherever else you set them (agent configs, scripts): `SOLH3_X` becomes
  `LFSTUDIO_X`. For the file: `sed -i 's/\bSOLH3_/LFSTUDIO_/g' <your settings file>`.
- MCP: register the server under its new name, for example `claude mcp add --scope user --transport stdio lf-studio -e
  LFSTUDIO_URL=http://<spark-host>:8190 -- python3 /path/to/lf-cli mcp` ([API.md](API.md#mcp-server-lf-cli-mcp)), and
  remove the old entry (`sol-h3`; `sol_h3` in Hermes). On other machines, download `bin/lf-cli`
  ([API.md](API.md#cli-binlf-cli)).
- spark-switchboard, with the studio stopped (Stop in switchboard; one left running keeps its pid in the 0.4 pidfile,
  so the restarted switchboard would not find it and would never stop it): replace your service file's contents with
  `integrations/spark-switchboard/lf-studio.toml` (the command `bin/lf-studio`, `LFSTUDIO_*` in `env`, the pidfile
  under `~/.local/state/lf-studio`) and set your checkout's path in `command` and `kill_command`. The file assumes
  `~/src/lf-studio`; a checkout cloned before 0.5 is still at `~/src/sol-h3-studio` unless you moved it, so the
  command becomes `~/src/sol-h3-studio/bin/lf-studio`. The file name is the service name, so a new name also changes
  it for `delegate.sh` ([nvidia-sync.md](nvidia-sync.md)). Then `spark-switchboard check`, and restart switchboard.
- NVIDIA Sync: the command `bash <checkout>/bin/lf-studio`. systemd: `systemctl --user disable --now sol-h3-studio`,
  then set up `integrations/systemd/lf-studio.service` as its comment says, with your checkout's path in `ExecStart`
  (it assumes `~/src/lf-studio`).
- Optional: move the settings file and the state to the new places, with the studio stopped, and leave a link at each
  old place for anything that still names it (the pidfile of a 0.4 service file). `mv -T` takes the place of an empty
  new directory (the new spark-switchboard file leaves an empty `~/.local/state/lf-studio` behind) and refuses one that
  holds anything, changing nothing; if it refuses, look at what the new directory holds (a `launcher.pid` left by a
  killed launcher can go) before you run the line again:

```bash
mv -T ~/.config/sol-h3-studio ~/.config/lf-studio && ln -s ~/.config/lf-studio ~/.config/sol-h3-studio
mv -T ~/.local/state/sol-h3-studio ~/.local/state/lf-studio && ln -s ~/.local/state/lf-studio ~/.local/state/sol-h3-studio
```

Leave `~/sol-h3` where it is: its venvs hold absolute paths. The repository is now github.com/slogonomo/lf-studio;
GitHub forwards the old address, and `git remote set-url origin https://github.com/slogonomo/lf-studio` updates a
checkout.

Back to 0.4 (DGX Spark), with the studio stopped: `git checkout v0.4.0`, then `install/10-sana.sh`. Until 0.6, 0.5 keeps
its copy of the patch also under 0.4's name (`sol-h3-studio-applied.patch`), so 0.4's script replaces the 0.5 patch with
its own; it calls it "an earlier version". Then start the studio as before. Back to 0.5: `git checkout main`,
`install/10-sana.sh`.

### Earlier versions

From 0.3 to 0.4: the patch changed, so run `install/10-sana.sh` after the pull. Until then the studio refuses to start
with "SOL_H3_PROMPT_TOKENS=minimax (the default since 0.4) needs a newer patch than the one applied in ...; run
install/10-sana.sh": the old patch would ignore the new setting. With `minimax` the prompt encoder reads the dialogue
tags `<d>` and `</d>` as single tokens, as MiniMax's tokenizer does. A dialogue prompt therefore encodes differently
from 0.3, and a 0.3 dialogue clip doesn't come back from its seed unless you set `SOL_H3_PROMPT_TOKENS=plain`. Prompts
without these tags encode as before ([PATCH.md](PATCH.md#16-minimaxs-prompt-tokens)). Stage 1 now compiles its
graphs statically ([PATCH.md](PATCH.md#18-reshape-another-length-or-orientation-without-a-reload)), apart from the
prompt length in its body blocks (below), so the first clips after the update compile Stage 1 once more (inferred).
Also new in 0.4:

- A new prompt length no longer recompiles Stage 1's body blocks (`SOL_H3_DYNAMIC_TEXT_LENGTH=1`, the default;
  [PATCH.md](PATCH.md#17-prompt-lengths-without-recompiles)). Before 0.4 a prompt with a token count new to the session
  usually took 7-14 s longer in Stage 1 than a repeat (the studio's receipts). In the staging check of 0.4 two new
  lengths in t2va 5 s took 1.9 s and 0.5 s longer, with bit-identical output. `SOL_H3_DYNAMIC_TEXT_LENGTH=0` compiles
  per prompt length, as before.
- The studio answers only to IP addresses, `localhost`, names without a dot and names ending in `.ts.net`, `.local`,
  `.lan`, `.home`, `.home.arpa` or `.internal`. If you open it by any other name (a router's name such as
  `spark.fritz.box` or `spark.localdomain`, a domain of your own, or a reverse proxy that sends such a name in
  `X-Forwarded-Host`), add that name to `LFSTUDIO_ALLOWED_HOSTS` in the env file before you restart. Otherwise every
  request gets 421 `unknown host name` ([API.md](API.md#access)).
- At most `LFSTUDIO_MAX_QUEUED` clips (default 20) wait in the queue; a script that queues more gets 429 until one has
  started.
- A clip of the loaded mode with another length or format keeps the models loaded and changes the session's shape
  (`LFSTUDIO_RESHAPE=1`, the default). `LFSTUDIO_RESHAPE=0` loads the models again for it, as before.

From 0.1 to 0.2: the first start imports the clip history (`jobs.json`) into `jobs.db` in the same directory
(see "Where the clip history is" in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#while-it-runs)). Back up
`$LFSTUDIO_STATE_DIR` first if you want a copy.

## Uninstall

Stop the studio, then remove `~/sol-h3` (or your `$LFSTUDIO_HOME`), `~/.config/lf-studio`,
`~/.local/state/lf-studio` (an install from before 0.5 may still use `~/.config/sol-h3-studio` and
`~/.local/state/sol-h3-studio`, or links there: remove those too) and the images: `docker rmi sol-h3-spark-qwen
sol-h3-spark-qwen:base` (the RTX 5090 install has no images, and keeps uv's cache in `~/sol-h3/cache/uv`). The RTX
5090 install also writes files derived from the weights next to `$LFSTUDIO_CHECKPOINTS`: `derived/` (the LTX block
folder and the H3 snapshots, ~110 GB) and `scratch/` (empty between starts); remove them too, also when the
checkpoints are a shared copy that stays.
