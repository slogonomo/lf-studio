# Troubleshooting

Logs:

- Launcher: `~/.local/state/sol-h3-studio/launcher.log` (`$SOLH3_STATE_DIR`).
- Web server: `~/.local/state/sol-h3-studio/server.log`.
- Pipeline workers: `$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/<session>/{stage1,stage2,qwen,warmup-qwen}-worker*/worker.log`.
  A failed clip's card shows the error; "Worker exited with code N; see .../worker.log" points at the right file.

## The server won't start

**`REFUSED - Sol-H3 Studio needs ~110 GiB free memory, only N GiB available`.** Another app holds the memory.
The launcher lists the biggest users (GPU processes, top RSS, containers). Stop them and start again.
`SOLH3_MIN_FREE_GIB=0` skips the check; the pipeline still needs nearly the whole box, so it will fail or hit the
memory guard later.

**`REFUSED - port 8190 is already in use`.** Another copy runs, or another app uses the port
([spark-switchboard](https://github.com/slogonomo/spark-switchboard) owns 8190 when it manages the studio; from
NVIDIA Sync, use switchboard's `delegate.sh` then, see [nvidia-sync.md](nvidia-sync.md)). Stop it or set
`SOLH3_PORT`.

**`cannot talk to docker as <user>`, or `permission denied while trying to connect to the Docker daemon socket`.**
Add yourself to the group (`sudo usermod -aG docker $USER`) and log in again. A login session started before
that change keeps its old groups. NVIDIA Sync keeps one SSH login open for a long time, so this happens there
even after you were added. The launcher falls back to `sg docker` when the socket is not writable and exports
that fallback to the Qwen worker; that works as soon as `/etc/group` lists you. `install/00-check.sh` says which
case you are in.

**`>>> STATUS: ERROR: the server exited ...`** or **`... did not answer ... within 60 s`.** The launcher shows
the last lines of `server.log`; the full log is in `$SOLH3_STATE_DIR`.

**`docker image sol-h3-spark-qwen not found`.** Run `install/60-qwen-image.sh`. Both tags are needed:
`sol-h3-spark-qwen:base` (NVIDIA's `Dockerfile.qwen`) and `sol-h3-spark-qwen` (the torchaudio stub layer on top).

**`no path manifests`** or a mode shows "The server has no path manifest". Run `install/70-manifests.sh`
(`SOLH3_TASKS` selects modes). The manifests must point `qwen_python` at the patched checkout.

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
the env at `$SOLH3_HOME/envs/stage2` yourself, run `install/50-stage2.sh --torchaudio-fix-only`: it makes only this
edit and installs nothing (plain `install/50-stage2.sh` would reinstall the whole env into that venv). Or edit
`envs/stage2/lib/python3.12/site-packages/torchaudio/_extension/utils.py` by hand: change
`if ta_version != t_version:` to `if ta_version.split('.')[0] != t_version.split('.')[0]:`.

**`ptxas fatal: Unsupported .version 9.4; current version is '9.2'`** while building ltx-kernels.
`nvidia-cuda-nvcc==13.2.*` pulls unpinned `nvidia-nvvm` / `nvidia-cuda-crt` 13.4. `install/50-stage2.sh` pins
both to 13.2. In an existing env:
`envs/stage2/bin/pip install "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" "nvidia-cuda-nvcc==13.2.*"`, then
rebuild ltx-kernels (`FORCE=1 install/50-stage2.sh`).

**`torchaudio is a stub in the sol-h3-spark-qwen image`.** Something in the Qwen container tried to use audio.
The prompt encoder never should; report it with the worker log.

**A reference file is "not found" inside the Qwen container.** The container sees only
`SOL_H3_SPARK_RUNTIME_ROOT`, the package, `SOL_H3_SPARK_QWEN_WEIGHTS_ROOT` and `SOL_H3_SPARK_COMFY_ROOT`. The web
UI stores uploads under the runtime root. For `infer.py`, keep input media there too; symlinks pointing outside
are not visible.

## While it runs

**`>>> STATUS: ERROR: Sol-H3 Studio is running out of memory (MemAvailable < 4 GiB for 3 s) - stopping it`.** The
launcher's memory guard stopped the app before GB10's unified memory ran out, which can hang the box. Usually
another app grew next to it. `launcher.log` lists the memory users at that moment. `SOLH3_GUARD_GIB` changes the
floor; 0 turns the guard off (not recommended).

**The first clip with a new prompt takes 9-10 s longer than the next one.** Stage 1 is compiled for static
shapes, and each new prompt length (in text tokens) compiles new graphs once. Reusing a prompt, or editing it
without changing its token count, avoids it. Before the patch the 4th distinct length in a session crashed the
worker ("Forward execution thread failed", Dynamo recompile limit); the patch raises the limit.

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
as idle (`idle_values` in `integrations/spark-switchboard/sol-h3-studio.toml`, since 0.3). With a copy from before,
switchboard sees the studio as busy while a cancelled clip is among the newest 24. That includes a config made from
switchboard's own example (`examples/services/sol-h3-studio.toml` before spark-switchboard's matching change). Add
`"cancelled"` to `idle_values` in switchboard's `services/` copy, or copy the file there again, and restart
switchboard. Until then, deleting the cancelled clip ends the block at once.

**Switching mode, format or length takes minutes.** Each combination is one loaded session. Changing any of them
closes the session and loads the models again; the Generate button says so before you click.

**Clips in the gallery after pruning old session directories.** Each clip lives in its session directory under
`$SOL_H3_SPARK_RUNTIME_ROOT/outputs/webui/`. If you prune those, clips that also have a copy in
`SOLH3_PUBLISH_DIR` keep playing from that copy; others show as missing.

**Where the clip history is.** `$SOLH3_STATE_DIR/jobs.db` (SQLite) holds every clip; the gallery shows the newest
24 and loads older ones as you scroll. Before 0.2 the history was `jobs.json` (the newest 200 clips; the gallery
showed 50). 0.2 imports it once and leaves it as it was, so going back to 0.1 shows the history as of that import.
Clips made under 0.1 after that are not imported by themselves when you return to 0.2; to add them, stop the studio and
mark the import as not done (entries `jobs.db` already has are kept):

```bash
python3 -c 'import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); c.execute("PRAGMA user_version = 0"); c.commit()' \
    "$SOLH3_STATE_DIR/jobs.db"
```

Clips from before per-clip session directories may have their only file in `SOLH3_PUBLISH_DIR`; Delete moves that
file to the trash like any other (Undo puts it back).

**Getting a deleted clip back after the Undo toast is gone.** Delete moves the clip's directory, its input files
and its `SOLH3_PUBLISH_DIR` copy to `$SOL_H3_SPARK_RUNTIME_ROOT/trash/<clip id>/` and keeps it in the history,
hidden, for `SOLH3_TRASH_DAYS` (default 7). Until then it can be restored with everything moved back where it was
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

## Stopping

Ctrl-C, SIGTERM or SIGHUP: the server closes its pipeline (up to ~40 s; the launcher waits up to
`SOLH3_STOP_WAIT_S`, default 90 s), then the launcher stops leftover workers and `sol-h3-qwen-*` containers.
The server runs in its own session, so a closed terminal or SSH session reaches only the launcher, which then stops
everything in that order; if the launcher itself is killed hard, the server gets SIGTERM and still closes its
pipeline. If something was killed hard anyway, `bin/sol-h3-studio --kill` removes the server, the workers and the
Qwen containers.
