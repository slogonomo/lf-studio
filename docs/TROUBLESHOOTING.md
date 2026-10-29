# Troubleshooting

Logs:

- Launcher: `~/.local/state/lf-studio/launcher.log` (`$LFSTUDIO_STATE_DIR`; an install made with 0.4 keeps
  `~/.local/state/sol-h3-studio` until 0.6, while the new directory has no clip history).
- Web server: `~/.local/state/lf-studio/server.log`.
- Pipeline workers: `$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/<session>/{stage1,stage2,qwen,warmup-qwen}-worker*/worker.log`.
  A failed clip's card shows the error; "Worker exited with code N; see .../worker.log" points at the right file.

## The server won't start

**`lf-studio: settings under their 0.4 names, which work until 0.6: SOLH3_PORT, ...`** and **`lf-studio: using
~/.config/sol-h3-studio/env: ~/.config/lf-studio/env does not exist`.** Not errors: an install made with 0.4 (or a
spark-switchboard service file of 0.4, or an agent's config) uses the names the studio had up to 0.4, and they still
work. "(LFSTUDIO_X wins)" after a name means both names are set and the old one is ignored. Rename them before 0.6
([INSTALL.md](INSTALL.md#from-04-to-05)). `bin/sol-h3-studio` and `bin/sol-h3` say "renamed" the same way.

**`REFUSED - LF Studio needs ~110 GiB free memory, only N GiB available`.** Another app holds the memory.
The launcher lists the biggest users (GPU processes, top RSS, containers). Stop them and start again.
`LFSTUDIO_MIN_FREE_GIB=0` skips the check; the pipeline still needs nearly the whole box, so it will fail or hit the
memory guard later.

**`REFUSED - port 8190 is already in use`.** Another copy runs, or another app uses the port
([spark-switchboard](https://github.com/slogonomo/spark-switchboard) owns 8190 when it manages the studio; from
NVIDIA Sync, use switchboard's `delegate.sh` then, see [nvidia-sync.md](nvidia-sync.md)). Stop it or set
`LFSTUDIO_PORT`.

**`REFUSED - SOL_H3_PROMPT_TOKENS=minimax (the default since 0.4) needs a newer patch than the one applied in ...`**
(or the same for `SOL_H3_STAGE2_AUDIO`, `SOL_H3_STATIC_SHAPES` or `SOL_H3_DYNAMIC_TEXT_LENGTH`). The repo was
updated, but the Sana checkout still carries the patch of an older release, which would ignore that setting. Run
`install/10-sana.sh`, then start again ([updating](INSTALL.md#updating)).

**`cannot talk to docker as <user>`, or `permission denied while trying to connect to the Docker daemon socket`.**
Add yourself to the group (`sudo usermod -aG docker $USER`) and log in again. A login session started before
that change keeps its old groups. NVIDIA Sync keeps one SSH login open for a long time, so this happens there
even after you were added. The launcher falls back to `sg docker` when the socket is not writable and exports
that fallback to the Qwen worker; that works as soon as `/etc/group` lists you. `install/00-check.sh` says which
case you are in.

**`REFUSED - LFSTUDIO_BACKEND=...: must be spark or rtx5090`.** The settings file or the environment has another
value. Leave it unset on a DGX Spark; set `rtx5090` on a workstation with one RTX 5090 ([RTX5090.md](RTX5090.md)).

**`REFUSED - ... carries the patch for Sol-H3-RTX5090, not Sol-H3-Spark: set LFSTUDIO_BACKEND=rtx5090`** (or the other
way round). The Sana checkout in `LFSTUDIO_SANA` was patched for the other machine. Set `LFSTUDIO_BACKEND` to the one
the message names, or point `LFSTUDIO_SANA` at the right checkout. `no Sol-H3-RTX5090 package at ...` means the checkout
is older than NVIDIA's RTX 5090 package (a DGX Spark's, at `patches/BASE_COMMIT`): run `install/10-sana.sh` with
`LFSTUDIO_BACKEND=rtx5090` into its own `LFSTUDIO_SANA`.

**RTX 5090: `REFUSED - LF Studio needs ~N GiB free GPU memory, only M GiB free`.** Other apps hold GPU memory
(`nvidia-smi` lists them; so does the launcher). Close them, or set `LFSTUDIO_MIN_FREE_VRAM_GIB` lower (0 turns the
check off) at your own risk: every mode needed at most 24.8 GiB ([RTX5090.md](RTX5090.md)); below
that a clip can fail with a CUDA out-of-memory error.

**`>>> STATUS: ERROR: the server exited ...`** or **`... did not answer ... within 60 s`.** The launcher shows
the last lines of `server.log`; the full log is in `$LFSTUDIO_STATE_DIR`.

**`docker image sol-h3-spark-qwen not found`.** Run `install/60-qwen-image.sh`. Both tags are needed:
`sol-h3-spark-qwen:base` (NVIDIA's `Dockerfile.qwen`) and `sol-h3-spark-qwen` (the torchaudio stub layer on top).

**`no path manifests`** or a mode shows "The server has no path manifest". Run `install/70-manifests.sh`
(`LFSTUDIO_TASKS` selects modes). The manifests must point `qwen_python` at the patched checkout.

## Loading fails

**CUDA out of memory while a worker starts, with plenty of MemAvailable.** Symptoms:
`cuDevicePrimaryCtxRetain` returns `CUDA_ERROR_OUT_OF_MEMORY`, and `sudo dmesg` shows `NV_ERR_NO_MEMORY` (in
`kgrctxAllocMainCtxBuffer`). On GB10, creating a CUDA context needs *free* memory; page cache counts as
available but not as free. Seen on a DGX Spark when an LLM server started right after this pipeline had run
(MemFree ~3 GiB, page cache ~114 GiB, MemAvailable 117 GiB); the same applies to this pipeline's own workers when
the page cache is full. Drop the page cache, then start again:

```bash
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches
```

The launcher prints a note when MemFree is under 20 GiB. Large file copies or downloads right before a start
fill the page cache. Under [spark-switchboard](https://github.com/slogonomo/spark-switchboard), its free-memory
floor and reclaim command can do this before each start.

**`Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA version 13.2
whereas TorchAudio has CUDA version 13.0.`** (Stage2 worker). There is no aarch64 torchaudio wheel for cu132, so
the Stage2 env uses the cu130 build, and torchaudio's import check refuses any CUDA minor mismatch.
`install/50-stage2.sh` relaxes that check to the major version (backup `utils.py.orig` next to it). If you built
the env at `$LFSTUDIO_HOME/envs/stage2` yourself, run `install/50-stage2.sh --torchaudio-fix-only`: it makes only this
edit and installs nothing (plain `install/50-stage2.sh` would reinstall the whole env into that venv). Or edit
`envs/stage2/lib/python3.12/site-packages/torchaudio/_extension/utils.py` by hand: change
`if ta_version != t_version:` to `if ta_version.split('.')[0] != t_version.split('.')[0]:`.

**`ptxas fatal: Unsupported .version 9.4; current version is '9.2'`** while building ltx-kernels.
`nvidia-cuda-nvcc==13.2.*` pulls unpinned `nvidia-nvvm` / `nvidia-cuda-crt` 13.4. `install/50-stage2.sh` pins
both to 13.2. In an existing env:
`envs/stage2/bin/pip install "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" "nvidia-cuda-nvcc==13.2.*"`, then
rebuild ltx-kernels (`FORCE=1 install/50-stage2.sh`).

**RTX 5090: ltx-kernels does not build with the system's GCC.** nvcc 13.2 accepts host compilers up to GCC 15. With
GCC 16 (Arch Linux) its host-compiler check refuses the build, and with `-allow-unsupported-compiler` it fails in
GCC 16's own headers (`/usr/include/c++/16/bits/shared_ptr_base.h(2118): error: type name is not allowed`,
`identifier "__builtin_is_virtual_base_of" is undefined`). A newer nvcc is no way out: nvcc 13.4 rejects torch's
CUDA 13.2 headers (`CUDA compiler and CUDA toolkit headers are incompatible`). With `LFSTUDIO_BACKEND=rtx5090`,
`install/50-stage2.sh` therefore builds with GCC 14 from conda-forge (micromamba, in `$LFSTUDIO_HOME/toolchains/gcc14`)
whenever the system's gcc is newer than 15, through `CC`, `CXX` and `NVCC_CCBIN`. It also sets `CFLAGS=-w
CXXFLAGS=-w`: glibc 2.44 defines `_POSIX_C_SOURCE` differently from Python's `pyconfig.h`, and the extension builds
with `-Werror` (`error: '_POSIX_C_SOURCE' redefined [-Werror]`); these are warnings only. Found on a workstation with
one RTX 5090 (Arch Linux, GCC 16, glibc 2.44). To build by hand, use the command `install/50-stage2.sh --dry-run`
prints. Keep the toolchain afterwards: the built extensions carry an RPATH to `toolchains/gcc14/lib` and load its
libstdc++ at run time. `FORCE=1` keeps it; `FORCE=1 FORCE_TOOLCHAIN=1 install/50-stage2.sh` installs it again (the
old one is moved to `$LFSTUDIO_HOME/cache` first) and rebuilds the extensions against it.

**`torchaudio is a stub in the sol-h3-spark-qwen image`.** Something in the Qwen container tried to use audio.
The prompt encoder never should; report it with the worker log.

**A reference file is "not found" inside the Qwen container.** The container sees only
`SOL_H3_SPARK_RUNTIME_ROOT`, the package, `SOL_H3_SPARK_QWEN_WEIGHTS_ROOT` and `SOL_H3_SPARK_COMFY_ROOT`. The web
UI stores uploads under the runtime root. For `infer.py`, keep input media there too; symlinks pointing outside
are not visible.

## While it runs

**`>>> STATUS: ERROR: LF Studio is running out of memory (MemAvailable < 4 GiB for 3 s) - stopping it`.** The
launcher's memory guard stopped the app before GB10's unified memory ran out, which can hang the box. Usually
another app grew next to it. `launcher.log` lists the memory users at that moment. `LFSTUDIO_GUARD_GIB` changes the
floor; 0 turns the guard off (not recommended).

**The first clip with a new prompt takes longer than the next one.** Stage 1 is compiled for static shapes, and a new
prompt length (in text tokens) compiles new graphs once. Since 0.4 (`SOL_H3_DYNAMIC_TEXT_LENGTH=1`, the default) that is
only the two text-refiner regions: in the staging check of 0.4, two new lengths in t2va 5 s took 1.9 s and 0.5 s longer
in Stage 1; the body blocks compile per session, not per length
([PATCH.md](PATCH.md#17-prompt-lengths-without-recompiles)). With `SOL_H3_DYNAMIC_TEXT_LENGTH=0` they compile per length
again: 9-10 s (measured before 0.4; about 1-2 s more with 0.4's static compiles, inferred). Reusing a prompt, or editing
it without changing its token count, avoids it. Before the patch the 4th distinct length in a session crashed the worker
("Forward execution thread failed", Dynamo recompile limit); the patch raises the limit.

**The first run of a mode, format or length is much slower.** Its kernels are compiled once into
`$SOL_H3_SPARK_RUNTIME_ROOT/cache/compile` (measured first runs: t2va 9:16 5 s 111 s, t2va 16:9 10 s 264 s,
t2va 16:9 15 s 357 s). Later cold starts reuse the cache. If you suspect a broken cache, move that directory away;
`SOL_H3_COMPILE_CACHE=session` runs with upstream's per-session caches for comparison.

**A clip shows "Recovered after: ...".** A worker died during the job; the server closed the session, reloaded
the models and ran the job again once. If it happens again, the card shows "Failed" with the error.

**What Cancel does.** On a queued clip it takes the clip out of the queue; nothing has run. On a running clip
("Loading models", "Generating", also during the retry after a worker error) the server sends SIGTERM to the
pipeline's workers, as when it stops an unwanted warm-up, and closes the session: the runtime can't stop between its
steps and keep the models, so the next clip loads them first, like the first clip of a mode (a 5 s t2va clip then
takes 104-113 s instead of 54-66 s with the models loaded; see [INSTALL.md](INSTALL.md#start) and
[PERFORMANCE.md](PERFORMANCE.md)). The card shows "Cancelling" until the workers are gone, then "Cancelled", with the
clip's input files kept for Remix, and Delete. A clip that finishes just as you press Cancel is either kept (Cancel
then says it has just finished) or cancelled, never both. A clip whose Cancel was answered stays cancelled even if
the studio stops before the clip has stopped.

**Cancelled clips after going back to a studio before 0.3.** An older studio doesn't know the status "cancelled": at
start it marks those clips as failed with "server restarted" and saves that, so they stay failed after updating
again.

**spark-switchboard never stops the studio for being idle after a Cancel.** Its busy probe must count "cancelled"
as idle (`idle_values` in `integrations/spark-switchboard/lf-studio.toml`, since 0.3). With a copy from before,
switchboard sees the studio as busy while a cancelled clip is among the newest 24. That includes a config made from
switchboard's own example (in its `examples/services/`, before spark-switchboard's matching change). Update
the existing copy in switchboard's `services/` directory (whatever you named it): add `"cancelled"` to its
`idle_values`, or replace its contents with the new file and keep its file name and your path edits. Then
`spark-switchboard check`, and restart switchboard ([INSTALL.md](INSTALL.md#updating)). Until then, deleting the
cancelled clip ends the block at once.

**Switching mode takes minutes.** Each mode is one loaded session. Another mode closes it and loads the models again;
the Generate button says so before you click. Another format or length in the same mode keeps the models loaded: the
session changes its shape, and the first clip at a shape it has not run yet compiles for it (measured 0-10 s more than a
warm clip in the staging check of 0.4; 0-27 s in an earlier run with `SOL_H3_DYNAMIC_TEXT_LENGTH` off, against 43-54 s
for a reload in the same run). With `LFSTUDIO_RESHAPE=0` any change of mode, format or length reloads, as before 0.4.

**Clips in the gallery after pruning old session directories.** Each clip lives in its session directory under
`$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/`. If you prune those, clips that also have a copy in
`LFSTUDIO_PUBLISH_DIR` keep playing from that copy; others show as missing.

**Where the clip history is.** `$LFSTUDIO_STATE_DIR/jobs.db` (SQLite) holds every clip; the gallery shows the newest
24 and loads older ones as you scroll. Before 0.2 the history was `jobs.json` (the newest 200 clips; the gallery
showed 50). 0.2 imports it once and leaves it as it was, so going back to 0.1 shows the history as of that import.
Clips made under 0.1 after that are not imported by themselves when you return to 0.2; to add them, stop the studio and
mark the import as not done (entries `jobs.db` already has are kept):

```bash
python3 -c 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.execute("PRAGMA user_version = 0"); c.commit()' \
    "$LFSTUDIO_STATE_DIR/jobs.db"
```

Clips from before per-clip session directories may have their only file in `LFSTUDIO_PUBLISH_DIR`; Delete moves that
file to the trash like any other (Undo puts it back).

**Getting a deleted clip back after the Undo toast is gone.** Delete moves the clip's directory, its input files
and its `LFSTUDIO_PUBLISH_DIR` copy to `$SOL_H3_SPARK_RUNTIME_ROOT/trash/<clip id>/` and keeps it in the history,
hidden, for `LFSTUDIO_TRASH_DAYS` (default 7). Until then it can be restored with everything moved back where it was
(the id is the directory name):

```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"id": "<clip id>"}' http://127.0.0.1:8190/api/undelete
```

If the publish folder can't be written when a clip is restored (an unmounted share), the clip comes back without
its publish copy, which stays in the trash until it is emptied; when that copy is the clip's only MP4 (an old clip,
or one whose session directory was pruned), Undo answers with an error instead and can be tried again later. Files
are copied to and from another filesystem under a temporary name, so a share never holds a partial MP4. Shutdown
waits up to 60 s for a Delete or Undo that is moving files, and a delete cut short (the server killed) is put back
at the next start. Anything else left in the trash (from a delete or
restore that failed half-way) is removed after the same number of days.

## Agents and scripts

The API, the CLI (`bin/lf-cli`) and the MCP server are in [API.md](API.md); its [Errors](API.md#errors) table lists
every status code.

**411 `Content-Length required (chunked uploads are not supported)`.** The client streamed the upload without a
length (chunked), which the studio can't read. Send the body with a `Content-Length`: curl's `-F` and
`--data-binary @file` (also `@-`, from a pipe) do; `-T -` and a `Transfer-Encoding: chunked` header don't.

**413 `upload too large (200 MB max)`, or the connection closes during an upload.** The request is over 200 MB. The
studio answers before it reads the body, so a client still sending may see a broken connection instead of the
answer. With JSON, base64 makes files a third larger, so they can total about 150 MB. `bin/lf-cli` checks the size
before it uploads.

**403 `the API for programs is off on this studio (LFSTUDIO_API=0)`**, or `bin/lf-cli` saying so. The studio's
settings turn the API off: JSON submits, submits from `bin/lf-cli` and `/api/jobs/<id>` are refused, and the page
works as usual. Set `LFSTUDIO_API=1` (or remove the line) in the env file and restart the studio to turn it on
([API.md](API.md#turning-the-api-off)).

**421 `unknown host name <name>; add it to LFSTUDIO_ALLOWED_HOSTS`.** You reach the studio by a name it doesn't
answer to by default, such as a domain of your own. Add the name to `LFSTUDIO_ALLOWED_HOSTS` in the env file and
restart the studio ([API.md](API.md#access) says what the check is for).

**429 `the queue is full`.** `LFSTUDIO_MAX_QUEUED` clips (default 20) are waiting. Wait until one has started
(`Retry-After: 60`), or raise the setting.

**400 `the studio does not fetch URLs`.** A JSON submit named a file by its URL. Send the file itself as base64, or
use `bin/lf-cli` (CLI or MCP), which downloads URLs on its own machine and uploads the files.

**400 `duration must be 5, 6, 7, 8, 10, 12 or 15 seconds (3 and 4 s are no longer offered)`**, or `lf-cli` refusing
`--seconds 3` (`invalid choice`). The studio offered 3 and 4 s up to 0.5.0. Ask for one of the lengths offered (5, 6,
7, 8, 10, 12 or 15 s); clips made at 3 or 4 s stay in the history, and their Remix on the page loads 5 s and says so.

**503 from spark-switchboard** (`{"error": {"code": ...}, "switchboard": ...}`; ask with `Accept:
application/json` to get JSON). A POST that got it was not passed on, so no clip was made, and sending it again is
safe:

- `waiting`: the studio starts once another app's work or quiet period is over. Send it again after `Retry-After`,
  or let switchboard hold the request longer with `X-Switchboard-Wait: <seconds>` (up to 3600; `bin/lf-cli` sends
  900).
- `no_memory`: something switchboard doesn't manage holds the memory. Stop it, or see the dashboard.
- `stopped`, on a status read or a wait (`/api/state`, `/api/jobs/...`, `/poster/...`): the studio is stopped, and
  those reads never start it. A clip that was running when you last looked has most likely finished: switchboard
  stops the studio only when no clip is queued or running.
- `paused`, `failed`: as the message says; switchboard's dashboard has more.

**"spark-switchboard stopped the studio while waiting for clip <id>; the clip most likely finished first".** The
studio was stopped between two polls of a wait, usually because another app wanted the memory right after the clip
ended. Get the clip with `lf-cli fetch <id> --wake` (MCP: `fetch` with `wake: true`), or `GET /video/<id>`; that
starts the studio again (about 110 GiB). If instead the studio itself stopped while the clip was still running
(`the studio is shutting down ...`, or "the studio stopped while clip <id> was ..."), the clip reads `failed` with
`server restarted` after the restart: make it again.

**A script's status check or wait starts the studio (and pushes another app out).** Your copy of the integration
file in switchboard's `services/` directory is older than this version: add the API's paths to its
`passive_paths` ([INSTALL.md](INSTALL.md#updating)). That includes a config made from switchboard's own example
(in its `examples/services/`) while its `passive_paths` is only `["/health"]`; this repo's
`integrations/spark-switchboard/lf-studio.toml` has the full list. `bin/lf-cli` asks switchboard first and never
starts a stopped studio for `status`, `wait` or `fetch` (only `fetch --wake` does).

**An MCP call says the clip is still running, or the host stops the call.** Each host limits how long a tool call
may run. Set `LFSTUDIO_MCP_WAIT_S` below that limit and let the agent call `wait` with the id
([API.md](API.md#waiting-values-per-client)). A stopped call never cancels the clip.

**MCP: `out must be inside LFSTUDIO_OUT_DIR ... or the working directory`.** The MCP server saves only there, unless
its environment has `LFSTUDIO_MCP_OUT_ANYWHERE=1`. Point `LFSTUDIO_OUT_DIR` at the folder you want.

## Upscale (experimental)

The Upscale is off unless `LFSTUDIO_UPSCALE=1` ([INSTALL.md](INSTALL.md#experimental-upscale)). Each upscale has a
directory of its own, `$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/<session>-upscale-<resolution>/<case>/regen/`, with
`job.json` (what the studio asked for), `worker.log` (the job's output) and `receipt.json` (its times, sigmas, peak
memory and lowest MemAvailable per phase). A failed upscale's card shows the job's own error line.

**No Upscale button on the cards.** The studio runs with the Upscale off (the default), or with
`LFSTUDIO_BACKEND=rtx5090`, where it does not run. Set `LFSTUDIO_UPSCALE=1` and restart.

**The Upscale button is greyed out.** It says why when you point at it, or click or tap it: for now it takes finished
5 s text clips only, 16:9 or 9:16 (`Upscale takes 5 s clips only, for now ...`, `Upscale takes text-to-video (t2va)
clips only, for now`), and it reads the clip's working files, which clips made before per-clip folders, or whose
session folder was removed, no longer have (`the clip's working files (its session directory) are gone`).

**`REFUSED - the experimental Upscale (LFSTUDIO_UPSCALE=1): no Python at LFSTUDIO_UPSCALE_PYTHON=...`** (or
`missing weights: ...`, or `LFSTUDIO_UPSCALE_PKG: no open_h3/pipeline.py in ...`). The launcher checks the Upscale's
environment, NVIDIA's package and the weights at start. Run `install/65-regen2k.sh`, or set `LFSTUDIO_UPSCALE=0`.
`LFSTUDIO_UPSCALE=1: the experimental Upscale runs on the DGX Spark only` means `LFSTUDIO_BACKEND=rtx5090`: the
release needs ~100 GiB of GPU memory.

**`not enough free memory for an upscale: N GiB available with the models closed, 110 GiB needed`** (115 GiB at
4K). The studio closed its own models, and something else still holds the memory (another app, a container). Close it
and try again; `LFSTUDIO_UPSCALE_MIN_FREE_GIB` changes the floor, and 4K always needs 5 GiB more than it (measured in
the job: ~101 GiB used at 2K, ~105 GiB at 4K).

**`stopped: MemAvailable was below 5 GiB for 3 s`.** The Upscale's own memory guard killed the job (SIGKILL, so its
`receipt.json` still says `running`) before the machine ran out (on GB10's unified memory that can hang the box), and
before the launcher's guard would have stopped the whole studio. Usually another app grew meanwhile; at 4K it can be the
job itself (in its one 4K run so far, a 16:9 clip, MemAvailable came down from 116.4 to 11.1 GiB, so ~105 GiB used: a 4K
upscale that starts with just the floor's 110 GiB free comes down below 5 GiB;
[PERFORMANCE.md](PERFORMANCE.md#experimental-upscale-open-h3-regenerate-2k)). At 4K, ~111 GiB or more free at the start
avoids it. `worker.log` shows how far it got. `LFSTUDIO_UPSCALE_GUARD_GIB` changes the floor; 0 turns it off.

**It sits at "Upscaling · loading its models" for minutes.** With the fast load (`LFSTUDIO_UPSCALE_LOAD=fast`, the
default) the 62 GB DiT and 14 GB of LoRA weights should load in under a minute (21.5-21.9 s in the studio's job, its
clean runs); "reading the clip" before it takes ~2 min from the final clip, and the refine and decode after it ~7-22 min
at 1080p and 2K and ~63-91 min at 4K, with 2 or 3 steps
([PERFORMANCE.md](PERFORMANCE.md#experimental-upscale-open-h3-regenerate-2k)). Minutes at loading mean the standard load
(`LFSTUDIO_UPSCALE_LOAD=standard`, NVIDIA's loaders through mmap: ~7.5 min, 455 s measured), or files the fast load
could not read with O_DIRECT and read through mmap instead: `worker.log` in the upscale's `regen/` folder says
`fast load: N files, M of them through mmap`, and its `receipt.json` has the same as `"load"`. All of them through mmap,
and `O_DIRECT off` in `worker.log`, mean `SPARK_FAST_SAFETENSORS=0` or `SPARK_DIRECT_IO=0` in the studio's settings: the
job inherits the patch's switches (the studio's start log warns about it).

**An upscale fails while it loads its models.** With the fast load, the default, try `LFSTUDIO_UPSCALE_LOAD=standard`
(NVIDIA's own loading, ~7.5 min) and restart the studio; the fast load has run in the studio's job only a few times so
far, on one DGX Spark, so its `worker.log` is worth a report. `LFSTUDIO_UPSCALE_PREREAD=1` at start is refused with the
fast load (`only with LFSTUDIO_UPSCALE_LOAD=standard`): it fills the page cache, which the fast load reads past. An
upscale waits while any clip is queued, also one queued after it, since it closes the studio's models; the next clip
loads them again, and with `LFSTUDIO_PRELOAD=1` the studio warms them up again once the upscale is done.

**An upscale of a 9:16 clip fails at once with `NVIDIA's package is not the one this wrapper was written for
(6e4a440): {'open_h3/pipeline.py': ...}`.** The job takes a 9:16 clip through NVIDIA's code with the shapes turned,
and does so only with the NVIDIA files it was written against (`open_h3/pipeline.py`, `open_h3/upscaler.py` and
`open_h3/upscaler_node.py`, by SHA-256, `None` for a file that is not there; also the 16:9 sizes in them, and no 9:16
entry of NVIDIA's own). The studio's start log says so too, as a note (`NVIDIA's package (LFSTUDIO_UPSCALE_PKG) has
other files than the Upscale's 9:16 route was written for (open_h3/pipeline.py changed, ...)`). `LFSTUDIO_UPSCALE_PKG`
names another version of NVIDIA's package, or its files were changed: point it at the extract of the pinned commit
that `install/65-regen2k.sh` makes (`$LFSTUDIO_HOME/sources/sana-6e4a440/models/Open-H3-Regenerate-2K`, its default
link). 16:9 clips are not checked against these files.

**Cancel of an upscale shows "Cancelling" for up to ~20 s.** The job stops once the GPU call under way returns; if
that takes longer than 20 s, the studio kills it. Either way the next job starts only after it is gone.

**The person in the upscale looks different.** "Full re-generate" (NVIDIA's setting, start noise 0.909) redraws
faces. Use "Balanced" or "Faithful".

**Delete of a clip says `an upscale of this clip is queued or running; cancel it first, or wait for it`.** The
upscale reads the clip's files.

**`install/65-regen2k.sh`: `... exists but this script did not make it`.** A venv at `envs/regen2k` that the step did
not build is never touched; move it away. `the packages of install/freeze/regen2k-lock.txt did not install`: a
network error or a package that changed on the index (every one is checked against its SHA-256); run the step again
and it goes on. `does not have NVIDIA's SHA-256 (kept as ....bad)`: a weight file that is not NVIDIA's; the bad file is
kept for a look, and a run again downloads it anew. `ignored the Range request`: something between you and Hugging Face
(a proxy, `HF_ENDPOINT`) does not pass on byte ranges, which the AdaLN table needs.

## LightX2V Turbo (experimental)

LightX2V Turbo, the second model, is off unless `LFSTUDIO_MODELS` lists it
([INSTALL.md](INSTALL.md#experimental-lightx2v-turbo)). Each of its clips has a directory of its own,
`$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/<session>-lightx2v-turbo-<resolution>-<task>-<format>-<seconds>s/<case>/`,
with `prompt.txt` and `encode.log` (the prompt encode's output), `qwen/` (and for Keyframes `qwen-text/`, the prompt
without its keyframe references), and `turbo/` with `job.json` (what the studio asked for), `worker.log` (the job's
output) and `receipt.json` (its times, peak memory and lowest MemAvailable). A failed clip's card shows the error.

**No Model row on the page.** The studio runs with `LFSTUDIO_MODELS` unset or `sol-h3` alone (the default): the page
is then as it was. Set `LFSTUDIO_MODELS=sol-h3,lightx2v-turbo` and restart.

**The studio won't start: `LFSTUDIO_MODELS=...: unknown model ... (known: sol-h3, lightx2v-turbo)`,
`... must include sol-h3, the studio's own model`, or `the experimental LightX2V Turbo (lightx2v-turbo) runs on the
DGX Spark only, not with LFSTUDIO_BACKEND=rtx5090`.** The setting is a comma list of model ids with `sol-h3` in it;
LightX2V Turbo needs ~100 GiB of GPU memory, so not on the RTX 5090. `bin/lf-studio --check` says the same.

**`REFUSED - the experimental LightX2V Turbo (LFSTUDIO_MODELS): no Python at LFSTUDIO_H3TURBO_PYTHON=...`** (or `no
.../minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors (the 4-step LoRA; install/66-h3turbo.sh fetches it)`, `the
base model has no weights in vae/ ...`, `no prompt encode at ...`). The launcher checks its environment, base model
and LoRA files at start with its job's own code. Run `install/66-h3turbo.sh`, or take `lightx2v-turbo` out of
`LFSTUDIO_MODELS`. `cannot read its job .../jobs/h3turbo_job.py (no such file)`: this copy of LF Studio has no
LightX2V Turbo job; update it.

**`install/66-h3turbo.sh`: `the base model ... has no VAE weights in vae/ (a t2va-only download has vae/config.json
only)`.** Step 20 downloaded MiniMax-H3 for t2va alone, which leaves out the VAE's weights; LightX2V Turbo decodes with
them. Run `install/20-weights.sh` with `fl2va` in `LFSTUDIO_TASKS` (the default has it), then 66 again. `does not have
the pinned size ... and SHA-256 ... (kept as ....partial.bad)`: the downloaded LoRA file is not the pinned one; it is
kept for a look, and a run again downloads it anew. `the download of ... broke off`: run the step again; it goes on
from the `.partial` file.

**400 `lightx2v-turbo is off on this studio (LFSTUDIO_MODELS)`**, or `lf-cli` saying `this studio has no model
choice: it makes Sol-H3 (sol-h3) clips only`. The studio has LightX2V Turbo off, or is a version without the model
choice (`lf-cli models` says which models it has). `lf-cli` and the MCP server check this before they submit, so no
Sol-H3 clip is made in its place.

**400 `LightX2V Turbo makes 5 s clips for now`, `... makes Text (t2va) and Keyframes (fl2va) clips for now, not
Reference (ref2va): use Sol-H3`, `... needs a first frame for now (a last frame alone: use Sol-H3)`,
`resolution must be 576p or 768p for lightx2v-turbo`, `steps must be 4 for lightx2v-turbo`.** What LightX2V Turbo
makes is a table in the studio, and these are not in it yet; the page greys them out with the same reason. Use
Sol-H3 for them. `resolution is an option of lightx2v-turbo; sol-h3 takes none`: the option goes with
`"model": "lightx2v-turbo"` only.

**400 `lightx2v-turbo needs SOL_H3_PROMPT_TOKENS=minimax for a prompt with MiniMax's tokens (<d>, ...)`.** The
studio's settings tokenize prompts as plain text (`SOL_H3_PROMPT_TOKENS=plain`), which splits `<d>` and the other
tokens into text that H3 never saw; LightX2V Turbo's job refuses such a conditioning, so the studio says it at once.
Remove the setting (minimax is the default), or write the prompt without those tokens.

**`not enough free memory for a LightX2V Turbo clip: N GiB available with the models closed, 100 GiB needed
(LFSTUDIO_H3TURBO_MIN_FREE_GIB); close other apps`.** The studio closed the Sol-H3 models, and something else still
holds the memory. Close it and try again. In the research probe the job came down to 35 GiB of MemAvailable at its
lowest at 576p and 31 GiB at 768p, from 108.5 GiB at its start (5 s, one DGX Spark).

**`stopped: MemAvailable was below 5 GiB for 3 s (...; LFSTUDIO_H3TURBO_GUARD_GIB)`**, or `stopped by its job:
MemAvailable fell below 6 GiB`. The studio's guard, or the job's own (exit status 3), stopped it before the machine
ran out of memory. Usually another app grew meanwhile; `turbo/worker.log` shows how far it got.

**`the prompt encode failed: ...`** (or `refused the clip`). The prompt encode runs a Qwen worker of its own, in the
studio's Qwen container (Docker), as the Sol-H3 models do; `encode.log` in the clip's directory has its output. A
container that won't start fails the clip, with no retry: see [Loading fails](#loading-fails).

**A queue of Sol-H3 and LightX2V Turbo clips is slow.** Clips run in order, and each change of model costs time: a
LightX2V Turbo clip closes the Sol-H3 models first (about 40 s), and the next Sol-H3 clip loads them again (about
1-2 min). Nothing is warmed up after a LightX2V Turbo clip. Queue clips of one model together.

**The Upscale button of a LightX2V Turbo clip is greyed out** (`Upscale of LightX2V Turbo clips comes in a later
version`). Not yet.

## Stopping

Ctrl-C, SIGTERM or SIGHUP: the server closes its pipeline (up to ~40 s; the launcher waits up to
`LFSTUDIO_STOP_WAIT_S`, default 90 s), then the launcher stops leftover workers and `sol-h3-qwen-*` containers.
The server runs in its own session, so a closed terminal or SSH session reaches only the launcher, which then stops
everything in that order; if the launcher itself is killed hard, the server gets SIGTERM and still closes its
pipeline. If something was killed hard anyway, `bin/lf-studio --kill` removes the server, the workers and the
Qwen containers.
