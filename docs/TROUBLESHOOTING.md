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
(or the same for `SOL_H3_STAGE2_AUDIO`, `SOL_H3_STATIC_SHAPES`, `SOL_H3_DYNAMIC_TEXT_LENGTH` or
`SOL_H3_STAGE2_KEYFRAME`). The repo was updated, but the Sana checkout still carries the patch of an older release,
which would ignore that setting. Run `install/10-sana.sh`, then start again ([updating](INSTALL.md#updating)).

**`REFUSED - SOL_H3_STAGE2_KEYFRAME=pin (the default since 0.5.2) needs a newer patch than the one applied in ...; run
install/10-sana.sh (or set SOL_H3_STAGE2_KEYFRAME=off)`.** The studio holds a Keyframes clip's first frame in Stage 2 by
default since 0.5.2, on the DGX Spark and on the RTX 5090 (the keyframe pin,
[PATCH.md](PATCH.md#19-stage-2-keyframe-pin)). The Sana checkout still carries an older patch, which never pins. Run
`install/10-sana.sh`, then start again. To start without the pin until then, set `SOL_H3_STAGE2_KEYFRAME=off` in the
settings file: Keyframes clips then come out as before.

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

**A Keyframes clip doesn't keep my image: another face, or its details redrawn.** The draft starts on your first
frame, but in NVIDIA's route Stage 2, which refines it to 1344×768, refines every frame from 91 % noise, without the
image and under a fixed generic prompt, so it repaints the first frame too (issue #57; NVIDIA's README guarantees
neither the keyframes nor the identity). Measured on 6 clips, frame 0 against the image at the draft's size: 23-34 dB
PSNR in the draft, 16-23 dB after Stage 2. It can also draw people into the first frame of an empty scene. This is
fixed by default since 0.5.2, on the DGX Spark and on the RTX 5090: Stage 2 holds the uploaded first frame (the
keyframe pin, [PATCH.md](PATCH.md#19-stage-2-keyframe-pin)). Measured at the output size, frame 0 then matches the
image at 30-38 dB instead of 17-23 dB on one DGX Spark, and at 32-36 dB instead of 18-21 dB on one RTX 5090, and in
some clips the next frames keep the face better too. Only the first frame is held: later frames are still refined, so
a clip with a lot of motion can still drift, and a last frame, Text and Reference clips are not changed. Check that
`SOL_H3_STAGE2_KEYFRAME` is not `off` in the settings file.

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
directory of its own, `$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/<session>-upscale-<resolution>/<case>/`: `regen/`,
with `job.json` (what the studio asked for), `worker.log` (the job's output) and `receipt.json` (its times, sigmas,
peak memory and lowest MemAvailable per phase, the clip's `frames`, the refine's `rows` and its block `forward`; with
Keep my frames each end's check, `pin`); `input/`, what the job read (the clip's MP4, audio and text, linked or made
for it, the pictures of the ends it was to keep, `keyframe/`, and `source.json`, which says from where and how); for a
Keyframes or Reference clip also `prompt-text-only.txt` (the text it was refined against) and `encode.log`. A failed
upscale's card shows the job's own error line.

**No Upscale button on the cards.** The studio runs with the Upscale off (the default), or with
`LFSTUDIO_BACKEND=rtx5090`, where it does not run. Set `LFSTUDIO_UPSCALE=1` and restart.

**The Upscale button is greyed out.** It says why when you point at it, or click or tap it. It takes every finished
clip of a length the studio makes, 5 to 15 s, 16:9 or 9:16, of every mode and model. Sol-H3's 3 and 4 s clips, made
up to 0.5.0, are not taken (`Upscale takes clips of the lengths the studio makes, 5 to 15 s, not 3 s`), nor a clip
whose MP4 has other frames than its length's (`Upscale can't take this clip: its MP4 has 145 frames, not the frames of
a 6 s clip`): the Upscale never cuts a clip to fit. It reads the clip's
working files, which clips made before per-clip folders, or whose session folder was removed, no longer have (`the
clip's working files (its session directory) are gone`). A Keyframes clip whose prompt is only its keyframe line (or a
Reference prompt of references alone) has no text to refine against (`nothing is left of the prompt without its
keyframe line: the Upscale needs text`): make the clip again with a few words of what happens in it. A Reference
prompt whose subjects, written out where they are used, would pass 32768 characters (`the prompt is too long as text
alone ...`) can't be encoded as text either: define each subject in its own words rather than through others. `the
Upscale can't write this clip's prompt as text alone` means the rewrite itself failed: the studio's log has the error.

**A size is greyed out in the popover, or the Upscale is refused for one.** The sizes depend on the clip's length:
1080p takes every length, 2K clips up to 10 s, 4K 5 s clips. A size the clip's length has not got is greyed out,
`later` under its name, with the reason as its title (`2K for 12 s clips comes later: 1080p takes them`, `4K for
10 s clips comes later: 1080p and 2K take them`). When that is the studio's
default size, the popover starts at the first size the clip has, with that reason as its first note. A whole clip at
those sizes would not fit in memory, or only past anything measured: they need the clip in pieces, which comes later
([PERFORMANCE.md](PERFORMANCE.md#longer-clips)). `lf-cli upscale`, the MCP tool and `POST /api/upscale` ask
for 2K unless told otherwise, so a 12 or 15 s clip needs `--resolution 1080p` (`resolution` `1080p`); the other sizes
are refused with the same reason (400; `lf-cli` exits 2).

**The popover says `Clips of this length have not been upscaled on the GPU through the studio yet: the times are
estimates`.** Every length and size offered has run on the GPU (5 s at every size; 6, 10 and 15 s at 1080p and 10 s
at 2K), or a longer clip at that size has, so this note shows only where a run failed and its line was taken out
(`UPSCALE_RAN` in `upscale/regen2k_job.py`). Such a clip is upscaled all the same, whole, and its times and memory
floor come from the job's model of the runs ([PERFORMANCE.md](PERFORMANCE.md#longer-clips)): it can take longer or
shorter than shown. Its `regen/receipt.json` has its real times, memory and rows, and is worth a report.

**An upscale of a Keyframes or Reference clip sits at "Encoding the prompt" or fails there.** Such a clip's prompt is
encoded first without its pictures (a transient Qwen worker, 9-18 s, 9.1-9.2 s in the Upscale's GPU runs, ~15 GiB), the
same encode LightX2V Turbo uses; its log is `encode.log` in the upscale's folder, the text it encoded
`prompt-text-only.txt`, its summary `input/encode.json` and its Qwen worker's log `input/qwen-worker/`. `the prompt
encode failed: ...` names the encode's own error (a Qwen container that did not start, for example: the same causes as a
LightX2V Turbo clip's encode, below). `the prompt encode's conditioning is not of the text asked for: its prompt
differs` (or `task`) means its receipt, `input/qwen/conditioning.json`, is of another text than `prompt-text-only.txt`:
compare the two. A Cancel stops it at once; a Qwen worker it leaves behind is stopped by the studio. The encode needs
the t2va path manifest: without it the Upscale button of such a clip says so (`... which needs the t2va path manifest
(run install/70-manifests.sh)`). A Reference prompt is rewritten as text alone (its subjects as their definitions, its
media labels out); if the result reads oddly, `prompt-text-only.txt` shows what the upscale used.

**The upscale of a LightX2V Turbo clip is shorter than the clip.** A LightX2V Turbo clip with a last keyframe has
every frame its model made (124 at 5 s, 5.17 s; 243 at 10 s), ending on that keyframe. The upscale keeps all of them,
with the clip's whole sound, but for a 5 s clip at 4K, which has not run with 124 frames yet: there it is the first 121
frames (5.04 s) and its sound is cut with them (the popover says so at 4K), so it ends 3 frames before your last
keyframe. A 576p clip is stretched to 768p first, as the popover says too.

**The card says "first frame not kept" or "last frame not kept".** Keep my frames (the popover's box, `pin`) holds an
end only where the clip already shows your picture. The job checks each end it is sent before anything loads, and the
card says why it did not hold one (`regen/receipt.json` has each end's check under `pin`):
- `(25.4 dB from your picture)`: the clip's own frame (its first, or its last) matches your picture at less than the
  28 dB the job needs (luma PSNR at 384p). The clip itself drifted from the picture there, and the Upscale does not
  pull it back: that end is refined as without the box.
- `your picture does not open`: the file is the one the clip was made from, but the job's Pillow can't read it.
- `the clip stops N frames before your last picture`: the clip keeps fewer frames than its model made, so its last
  frame is not the one your last keyframe was anchored to.
- `your picture is no longer on disk` or `your picture changed since the clip was made`: the picture in the clip's
  input files (`$SOL_H3_SPARK_RUNTIME_ROOT/inputs/webui/<clip id>/`) was removed, or replaced by another file, just
  as the upscale started. The studio compares it with the sha256 the clip was made from (the LightX2V Turbo receipt's,
  or Sol-H3's capture) at the request and at the start, and leaves such an end out (below); the job checks once more.

An upscale with the box that fails at once with `cannot read frame N of refined_...: ...` could not read the clip's own
frame for that check: ffmpeg's error follows, and `input/stage2/` has the MP4 it read.

**The popover has no Keep my frames box, or it is greyed out.** It is greyed out at 4K (`Not at 4K yet`) and with
Faithful or Full (`With Balanced only, for now`). A Text or Reference clip has no box. Under the box, or in its place,
the popover says why an end is not offered (`Last frame not kept: ...`), and an upscale's record has the same in
`pin.why`. A Sol-H3 clip never offers its last frame (it stops a few frames before it: 3 at 5 s), and offers its first
frame only when Stage 2 held it, so a clip made before 0.5.2, or with `SOL_H3_STAGE2_KEYFRAME=off`, says so. `your
picture is no longer on disk` or `your picture changed since the clip was made`: as above (the popover never reads the
pictures, so a picture changed in place shows there only once a request has hashed it, and that request leaves the end
out). `the clip has no record of how it used your picture` or `the studio can't tell how your picture was fitted to
the clip`: the clip's own record (a LightX2V Turbo clip's `turbo/receipt.json`, Sol-H3's `stage2/stage2_result.json`)
does not say what the job needs. Where Keep my frames has not run on the GPU through the studio at that size or length
(one end at 2K: a LightX2V Turbo clip's first or last frame alone, a Sol-H3 clip's first frame; every clip longer than
5 s but a LightX2V Turbo clip's both ends at 1080p for 10 s), the popover says `Keeping your frames has not been run
on the GPU through the studio yet: its times are estimates`: it is offered all the same, and the time it adds there is
an estimate (60 s for one end at 2K). On a longer clip it adds the 5 s seconds per end, a little short: both ends at
1080p for 10 s add about 45 s, not 32 ([PERFORMANCE.md](PERFORMANCE.md#longer-clips)). On a 12 or 15 s clip it is
offered at 1080p only, the one size those lengths have.

**`REFUSED - the experimental Upscale (LFSTUDIO_UPSCALE=1): no Python at LFSTUDIO_UPSCALE_PYTHON=...`** (or
`missing weights: ...`, or `LFSTUDIO_UPSCALE_PKG: no open_h3/pipeline.py in ...`). The launcher checks the Upscale's
environment, NVIDIA's package and the weights at start. Run `install/65-regen2k.sh`, or set `LFSTUDIO_UPSCALE=0`.
`LFSTUDIO_UPSCALE=1: the experimental Upscale runs on the DGX Spark only` means `LFSTUDIO_BACKEND=rtx5090`: the
release needs ~100 GiB of GPU memory.

**`not enough free memory for an upscale: N GiB available with the models closed, 110 GiB needed`** (115 GiB at 4K).
The studio closed its own models, and something else still holds the memory (another app, a container). Close it and
try again; `LFSTUDIO_UPSCALE_MIN_FREE_GIB` changes the floor, and 4K always needs 5 GiB more than it (measured in the
job: ~101 GiB used at 2K, ~105 GiB at 4K). A clip longer than 5 s needs what its length and size need by the memory
model where that is more than the setting: none at the default 110 GiB (2K at 10 s needs the most, 109). With a lower
setting it says so: `109 GiB needed (what a 2K 10 s upscale needs, more than LFSTUDIO_UPSCALE_MIN_FREE_GIB=100)`; the
setting does not take it below that. 0 turns the check off.

**`not enough free memory to start: N GiB available, 110 GiB needed for a 2K 10 s upscale`** (the job's own error, on
the card and in `worker.log`). The job checks the studio's floor again as it starts (job.json `min_free_gib`, less
0.5 GiB), and refuses before it loads anything (exit status 2): the memory went down between the studio's check and
the job's start. Close what took it, and try again.

**`stopped: MemAvailable was below 5 GiB for 3 s`.** The Upscale's own memory guard killed the job (SIGKILL, so its
`receipt.json` still says `running`) before the machine ran out (on GB10's unified memory that can hang the box), and
before the launcher's guard would have stopped the whole studio. Usually another app grew meanwhile; at 4K it can be the
job itself (in its one 4K run so far, a 16:9 clip, MemAvailable came down from 116.4 to 11.1 GiB, so ~105 GiB used: a 4K
upscale that starts with just the floor's 110 GiB free comes down below 5 GiB;
[PERFORMANCE.md](PERFORMANCE.md#experimental-upscale-open-h3-regenerate-2k)). At 4K, ~111 GiB or more free at the start
avoids it. At 2K it comes down to about 16 GiB, and to 14.1 GiB with Keep my frames holding both ends (measured through
the studio). A 10 s clip at 2K came down to 16.7 GiB from 116.5 GiB free (its one run, through the studio), and the
longer clips at 1080p to 20.4 GiB or more (15 s); the memory model expects 15.7 GiB at 2K for 10 s with both ends kept,
and 18.2 GiB or more at 1080p (8 s with both ends), from a start at ~116.5 GiB free. `worker.log` shows how far it
got. `LFSTUDIO_UPSCALE_GUARD_GIB` changes the
floor; 0 turns it off.

**It sits at "Upscaling · loading its models" for minutes.** With the fast load (`LFSTUDIO_UPSCALE_LOAD=fast`, the
default) the 62 GB DiT and 14 GB of LoRA weights should load in under a minute (21.5-21.9 s in the studio's job, its
clean runs); for a 5 s clip "reading the clip" before it takes ~2 min from the final clip, and the refine and decode
after it ~7-22 min at 1080p and 2K and ~63-91 min at 4K, with 2 or 3 steps
([PERFORMANCE.md](PERFORMANCE.md#experimental-upscale-open-h3-regenerate-2k)); a longer clip takes longer at each
([PERFORMANCE.md](PERFORMANCE.md#longer-clips)). Minutes at loading mean the standard load
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
(6e4a440): {'open_h3/pipeline.py': ...}`** (a 16:9 clip of 124 frames, of another length than 5 s, or one that keeps
your frames, too). The job takes a 9:16 clip through NVIDIA's code with the shapes turned, and does so only with the
NVIDIA files it was written against (`open_h3/pipeline.py`, `open_h3/upscaler.py` and `open_h3/upscaler_node.py`, by
SHA-256, `None` for a file that is not there; also the 16:9 sizes in them, and no 9:16 entry of NVIDIA's own). A clip
of any other length than 5 s needs all three as well: NVIDIA's upscaler keeps its latent frames, where NVIDIA's target
has a 5 s clip's. A 16:9 clip of 124 frames (a LightX2V Turbo clip with a last keyframe) or with an end held goes
through the job's copies of NVIDIA's encode and decode, which need `open_h3/pipeline.py` as it was written against.
The studio's start log says so as well, as a note that names the clips refused (`NVIDIA's package
(LFSTUDIO_UPSCALE_PKG) has other files than the Upscale's 9:16 route was written for (open_h3/pipeline.py changed,
...)`). `LFSTUDIO_UPSCALE_PKG` names another version of NVIDIA's package, or its files were changed: point it at the
extract of the pinned commit that `install/65-regen2k.sh` makes
(`$LFSTUDIO_HOME/sources/sana-6e4a440/models/Open-H3-Regenerate-2K`, its default link). A 16:9 clip of 121 frames with
no end held is not checked against these files. A clip of another length whose upscaled latent comes out with other
latent frames fails before the refine (`the upscaled latent is ..., not ...`), never cut.

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
without its keyframe references; none for a prompt that is nothing but its keyframe line, which is encoded again
without it, as the server log says), and `turbo/` with `job.json` (what the studio asked for), `worker.log` (the job's
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

**400 `LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not Reference (ref2va): use Sol-H3`,
`resolution must be 576p or 768p for lightx2v-turbo`, `steps must be 4 for lightx2v-turbo`.** What LightX2V Turbo
makes is a table in the studio, and these are not in it yet; the page greys them out with the same reason. Use
Sol-H3 for them. `resolution is an option of lightx2v-turbo; sol-h3 takes none`: the option goes with
`"model": "lightx2v-turbo"` only. (Before its GPU validation it also refused 6-12 s and a last frame alone, `LightX2V
Turbo makes 5 s clips for now` and `... needs a first frame for now`, and until 15 s and 768p 12 s had run on the GPU
(one run each, on one DGX Spark, 2026-10-29), `LightX2V Turbo makes clips of up to 12 s for now (10 s at 768p), not
15 s: use Sol-H3` and `LightX2V Turbo does not make clips with ..., seconds 12, ..., resolution 768p, ... together
yet`; it makes all those now.)

**A LightX2V Turbo clip with a last frame is 3 frames longer** (124 frames at 5 s, `clip_..._124f.mp4`, where its
other clips and Sol-H3's have 121). The model anchors the last frame to the last frame it makes, so such a clip keeps
every frame made and ends on your last frame; the studio's usual length would end 3 frames (0.125 s) before it. At 6 to
12 s it is 2 to 7 frames longer, at 15 s one (362), with its sound as long.

**400 `lightx2v-turbo needs SOL_H3_PROMPT_TOKENS=minimax for a prompt with MiniMax's tokens (<d>, ...)`.** The
studio's settings tokenize prompts as plain text (`SOL_H3_PROMPT_TOKENS=plain`), which splits `<d>` and the other
tokens into text that H3 never saw; LightX2V Turbo's job refuses such a conditioning, so the studio says it at once.
Remove the setting (minimax is the default), or write the prompt without those tokens.

**`not enough free memory for a LightX2V Turbo clip: N GiB available with the models closed, 100 GiB needed
(LFSTUDIO_H3TURBO_MIN_FREE_GIB); close other apps`** (or, with a lower setting, `90 GiB needed (what a 576p 5 s clip
needs, more than LFSTUDIO_H3TURBO_MIN_FREE_GIB=80)`). The studio closed the Sol-H3 models, and something else still
holds the memory. Close it and try again. On one DGX Spark the job took 80-98 GiB of MemAvailable at its lowest (from
about 117 GiB: 24.5 GiB left at 768p 10 s, 22.5 at 768p 12 s; 768p 15 s, the most it used, took 97.6 GiB). A 768p clip
of 12 s or more needs more than the default 100 GiB: 103 GiB at 12 s (`103 GiB needed (what a 768p 12 s clip needs,
more than LFSTUDIO_H3TURBO_MIN_FREE_GIB=100)`) and 107 at 15 s. The setting is a minimum: below what the job's memory
model says a clip of its size and length needs to stay above the job's own 6 GiB stop (90 GiB at 576p 5 s, 100 at
768p 10 s), the clip would be stopped part way, so a lower value does not take it below that; 0 turns the check off
([PERFORMANCE.md](PERFORMANCE.md#experimental-lightx2v-turbo)).

**`stopped: MemAvailable was below 5 GiB for 3 s (...; LFSTUDIO_H3TURBO_GUARD_GIB)`**, or `stopped by its job:
MemAvailable fell below 6 GiB`. The studio's guard, or the job's own (exit status 3), stopped it before the machine
ran out of memory. Usually another app grew meanwhile; `turbo/worker.log` shows how far it got.

**`the prompt encode failed: ...`** (or `refused the clip`). The prompt encode runs a Qwen worker of its own, in the
studio's Qwen container (Docker), as the Sol-H3 models do; `encode.log` in the clip's directory has its output. A
container that won't start fails the clip, with no retry: see [Loading fails](#loading-fails). When the guard or a
Cancel had to kill the encode, the studio also stops the Qwen worker it left (the log says `stopping the Qwen worker
the prompt encode left`).

**A queue of Sol-H3 and LightX2V Turbo clips is slow.** Clips run in order, and each change of model costs time: a
LightX2V Turbo clip closes the Sol-H3 models first (about 7 s, up to 40 s when a worker doesn't answer), and the
next Sol-H3 clip loads them again (about 1-2 min). Nothing is warmed up after a LightX2V Turbo clip. Queue clips of one
model together.

**The Upscale button of a LightX2V Turbo clip is greyed out.** A clip of either size and any length can be upscaled
(also with the model turned off since): 1080p takes every length, 2K up to 10 s, 4K 5 s. A Keyframes clip whose prompt
was only its keyframe line has no text to refine against (its `qwen-text/` is missing for the same reason). See
[Upscale](#upscale-experimental).

## Stopping

Ctrl-C, SIGTERM or SIGHUP: the server closes its pipeline (up to ~40 s; the launcher waits up to
`LFSTUDIO_STOP_WAIT_S`, default 90 s), then the launcher stops leftover workers and `sol-h3-qwen-*` containers.
The server runs in its own session, so a closed terminal or SSH session reaches only the launcher, which then stops
everything in that order; if the launcher itself is killed hard, the server gets SIGTERM and still closes its
pipeline. If something was killed hard anyway, `bin/lf-studio --kill` removes the server, the workers and the
Qwen containers.
