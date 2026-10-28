# Working on LF Studio

Notes for anyone changing this repo, people and coding agents alike. `CLAUDE.md` points here.

## Layout

- `webui/server.py`: the studio server. Python standard library only. `bin/lf-studio` runs it with the Stage1
  Python and the working directory set to the patched Sol-H3-Spark package, from which it imports `runtime.*`.
- `webui/index.html`: the whole page in one file: no framework, no build step, no requests to other sites. The
  server reads it on every request, so edits show up on reload.
- `bin/lf-cli`: the client for agents and scripts, a CLI and an MCP server (`lf-cli mcp`) in one file. People copy
  it to other machines, so it stays standard-library only, Python 3.9 syntax, and reads no other file of this repo.
  `docs/API.md` documents it and the HTTP API.
- `upscale/regen2k_job.py`: the experimental Upscale's GPU job, one upscale per process, run in its own env with
  NVIDIA's Open-H3-Regenerate-2K (its docstring says what is NVIDIA's and what it changes). Standard library outside its
  torch functions, so the server and the tests import it without torch. `upscale/fastload.py`, its loader, is a copy of
  the patch's `runtime/fastload.py`: change the patch's file and copy it (`tests/test_upscale.py` checks that they are
  the same, byte for byte). The server runs it as a job of its own (`POST /api/upscale`, `run_upscale`; off unless
  `LFSTUDIO_UPSCALE=1`), and reads the `LFSTUDIO_UPSCALE_*` settings with its `studio_settings`, as `bin/lf-studio
  --check` does. The page's Upscale button and popover, `lf-cli upscale` and the MCP tool `upscale` use the endpoint;
  `install/65-regen2k.sh` (optional) builds its environment from the hashed lock `install/freeze/regen2k-lock.txt` and
  fetches its weights (`install/regen2k_weights.py`).
- `bin/lf-studio` (launcher), `bin/env.example` (every setting, with comments), `install/` (00-90),
  `patches/lf-studio.patch` against NVlabs/Sana at `patches/BASE_COMMIT` ([docs/PATCH.md](docs/PATCH.md)),
  `integrations/`, `bench/`, `docs/`.
- The project was called Sol-H3 Studio up to 0.4. Until 0.6 its old names keep a 0.4 install working
  ([docs/INSTALL.md](docs/INSTALL.md#from-04-to-05)): `bin/sol-h3-studio` and `bin/sol-h3` are shims, every setting
  is also read as `SOLH3_*`, and the old places of the settings file, the state, the compile cache and the clips are
  used while the new ones hold nothing yet (decided by content: the state by its `jobs.db`). That code is marked
  "until 0.6".
- The RTX 5090 backend (`LFSTUDIO_BACKEND=rtx5090`, [docs/RTX5090.md](docs/RTX5090.md)):
  `patches/lf-studio-rtx5090.patch` against `patches/BASE_COMMIT_RTX5090` for NVIDIA's Sol-H3-RTX5090 package,
  `install/05-python.sh` (uv and its Pythons), and per-backend branches in the launcher, `install/env.sh` and the
  install scripts. Without the setting everything is the DGX Spark's, as before; keep it so.

## Tests

None of these needs the GPU:

```bash
python3 -m unittest discover -s tests -v       # server, client and page against the stub runtime (needs ffmpeg)
python3 tests/stub/studio.py                   # a stub studio on http://127.0.0.1:18299/ to click through
python3 tests/ui_test.py http://127.0.0.1:18299/   # a general check of the page against any studio
```

`tests/test_server.py` drives the server over HTTP, and `tests/test_api.py` its API for programs (JSON submit,
`/api/jobs/<id>?wait=`, labels, limits, and the API turned off with `LFSTUDIO_API=0`). `tests/test_cli.py` and
`tests/test_mcp.py` run `bin/lf-cli` as a CLI and as an MCP server against the stub and against small fake
spark-switchboard servers (ports 18820-18839), and check that it stays a lone Python 3.9 file. `tests/test_page.py`
drives the page in WebKit, also running `tests/ui_test.py` against a stub studio with a session loaded
(`LFSTUDIO_RESHAPE` 1 and 0, and `LFSTUDIO_API=0`), and is skipped without Playwright (`pip install playwright &&
playwright install webkit`). One touch test uses Chromium (`playwright install chromium`) and is skipped without it.

`tests/stub/runtime/` stands in for the patched `runtime` package: loading and generating are `sleep` processes
(`STUB_LOAD_S`, `STUB_CLIP_S`), each clip is a small ffmpeg test pattern in the requested format and length, and a
prompt containing `STUB_FAIL` fails like a dead worker. Its docstring lists the other hooks (a slow close, a clip
finishing as it is cancelled, a worker that starts late, a failure only on the first run). `tests/stub/studio.py`
gives each run its own runtime root, state directory and port under a temp directory; the tests use its
`StubStudio` class. It counts as started only once its own server serves the port, and stops with an error (a
non-zero exit status) if the port is in use or its server stops by itself, so nothing goes to another studio on that
port. Add a test there for every server change.

The stub studio has the experimental Upscale on (`LFSTUDIO_UPSCALE=1`), with `tests/stub/upscale_job.py` standing in
for `upscale/regen2k_job.py` (`LFSTUDIO_UPSCALE_SCRIPT`): it prints the job's phase lines over `STUB_UPSCALE_S`, ends
at once on SIGTERM (or ignores it with `STUB_UPSCALE_TERM=ignore`), dies with the server (`--parent`), writes a small
121-frame MP4 without sound and a receipt (none, or one ffmpeg can't read, with `STUB_UPSCALE_OUTPUT`), and fails
with `STUB_UPSCALE_FAIL=1`. The stub runtime writes stand-ins for a clip's working files (`qwen/conditioning.pt`, the
Stage-1 latent and audio) for it to find, and the server reads MemAvailable from a file the stub studio writes
(`LFSTUDIO_TEST_MEMINFO`; `StubStudio.set_memory`), so the memory floor and guard are tested without the machine's
memory. The `Upscale*` classes in `tests/test_server.py` cover the job type (queue and its limit, Cancel and its
SIGKILL, Delete and Undo, a request during a Delete's move, restart, warm-up, memory, failure, the checks at run
time, the settings, the launcher's checks and `--kill`), `tests/test_api.py` its API record, the `Upscale*` classes
in `tests/test_cli.py` and `tests/test_mcp.py` the client, and those in `tests/test_page.py` the button (disabled
with the reason on clips it can't take; in cards too narrow for the names, Upscale and Delete show icons, so the
buttons keep one row; `OneRowUpscaleOff` checks that row with the Upscale off), the popover, the upscale's card and
`tests/ui_test.py` with an upscale in the gallery.

`tests/test_backend.py` covers `LFSTUDIO_BACKEND` on the CPU: the launcher's `--check` and `--kill` with no backend
(the Spark: its package, the Docker refusal, the 110 GiB floor), with `rtx5090` (a copy of the stub runtime as
`Sol-H3-RTX5090` with a stand-in `runtime/offload.py`; no Docker; the RAM floor; the GPU memory floor driven by a fake
`nvidia-smi` on PATH) and with a wrong or mismatched value; the server's `backend` in `/api/state` and its page; the
page's per-backend texts without a browser (its `MACHINES` table, and its `applyMachine` in node when node is
installed); and every install script's `--dry-run` under both backends, with fake `uname`, `nvidia-smi`, `gcc`,
`docker` and `sg` on PATH and everything in a temp directory; on the RTX 5090 also the caches under `$LFSTUDIO_HOME`,
the GCC 14 toolchain kept on `FORCE=1`, the adoption of a venv whose fake `pip freeze` is the reference freeze, and
the dated prompt-cache backup. Add a test there for every backend change. It also covers `install/65-regen2k.sh` (its
dry run, and runs with a fake uv: a failed step stops it and leaves the venv unmarked, #74) and
`install/regen2k_weights.py` (with a stand-in `huggingface_hub` and a fake Hugging Face for the range reads, which
must never send a token, also through a redirect).

`tests/test_old_names.py` covers the names up to 0.4 that work until 0.6: the old setting names in the environment
and the settings file (and that the four `OLD_SETTINGS` lists agree), the old places, the shims, the launcher's check
for a running copy, a launcher run with the 0.4 spark-switchboard service file's settings, the deprecated integration
files, the install steps and the page's old localStorage keys. It goes with them in 0.6.

`tests/test_replay.py` covers the Stage-2 replay bench (`bench/replay.sh`, `bench/replay/`) on the CPU: the draft
store, every arm, the `--dry` plan, the GPU run object with stand-ins for torch, a `run` with a stand-in Stage-2
process, stopping it under `bench/memguard-run.sh`, the report, the quality step and the SyncNet table. It takes
`runtime/geometry.py` from the patch itself.

`tests/test_upscale.py` covers `upscale/regen2k_job.py` on the CPU (and that `upscale/fastload.py` is the patch's
`runtime/fastload.py`): job.json, the clip's and the weights' files (a 5 s 16:9 MP4 made by ffmpeg), the text check on
stand-in tensors, the command lines (the mux run with ffmpeg) and the environment, the progress lines, the memory trace,
`--check`, the receipt when a job is done, fails, is refused or gets SIGTERM, `--parent` (the job dies with the server),
and `run()` with a stand-in torch and a stand-in of NVIDIA's `open_h3`, with both loads (fast and standard: stand-ins
for safetensors, diffusers and `fastload` record every load, which the fast load must have replaced before the first
one). Its torch tests (the merger, the lean block forward against a stock-style block, the tensor checks) run where
torch is importable (CPU is enough) and are skipped otherwise.

The patch has its own 135 CPU tests, run in the patched checkout ([docs/PATCH.md](docs/PATCH.md#apply-test-revert)).
The RTX 5090 patch has 201 (`install/10-sana.sh` with `LFSTUDIO_BACKEND=rtx5090` runs them).

On real hardware: the pipeline needs nearly the whole DGX Spark (~110 GiB free), so run one GPU app at a time. A second
studio next to an installed one needs its own env file (`LFSTUDIO_ENV=<file> bin/lf-studio`) with its own
`LFSTUDIO_PORT`, `LFSTUDIO_STATE_DIR` and `SOL_H3_SPARK_RUNTIME_ROOT`, and `LFSTUDIO_PUBLISH_DIR` empty. Otherwise it
shares the other studio's clip history and files.

## Conventions

- The server stays standard-library only, the page a single file.
- A new setting goes into `bin/env.example` with a comment, and into the server's docstring. It gets no 0.4 name: the
  `OLD_SETTINGS` lists (launcher, `install/env.sh`, server, client) stay as they are until they go in 0.6.
- A change to an endpoint, or to what `bin/lf-cli` does, goes into `docs/API.md`. The prompt guide lives twice, in
  `docs/API.md` (between its `prompt-guide` markers) and as `GUIDE` in `bin/lf-cli`; `tests/test_cli.py` checks
  that they are the same.
- Keep the docs in step with the behaviour: `README.md` and `docs/*.md` describe what the studio does, with
  measured numbers where there are any (say where and how they were measured).
- Never commit generated clips or screenshots that show them (see the model terms in the README), model weights,
  run state, local settings, or personal details (home paths, host names, addresses).

## Branches and releases

1. Each issue gets a branch from the current `release/x.y` branch and a PR into it that names the issue
   ("Part of #N"). PRs into `release/x.y` are squash-merged.
2. The release branch is tested as a whole, on a separate studio as above.
3. A "Release x.y.0" PR merges it into `main` with a merge commit; its description has a `Fixes #N` line per issue,
   which closes them. Then `git tag -a vx.y.0` and `gh release create vx.y.0 --generate-notes`.
4. Installations follow `main` (`git pull`, then `install/10-sana.sh` if the patch changed, then restart).
