# Working on Sol-H3 Studio

Notes for anyone changing this repo, people and coding agents alike. `CLAUDE.md` points here.

## Layout

- `webui/server.py`: the studio server. Python standard library only. `bin/sol-h3-studio` runs it with the Stage1
  Python and the working directory set to the patched Sol-H3-Spark package, from which it imports `runtime.*`.
- `webui/index.html`: the whole page in one file: no framework, no build step, no requests to other sites. The
  server reads it on every request, so edits show up on reload.
- `bin/sol-h3`: the client for agents and scripts, a CLI and an MCP server (`sol-h3 mcp`) in one file. People copy
  it to other machines, so it stays standard-library only, Python 3.9 syntax, and reads no other file of this repo.
  `docs/API.md` documents it and the HTTP API.
- `bin/sol-h3-studio` (launcher), `bin/env.example` (every setting, with comments), `install/` (00-90),
  `patches/sol-h3-studio.patch` against NVlabs/Sana at `patches/BASE_COMMIT` ([docs/PATCH.md](docs/PATCH.md)),
  `integrations/`, `bench/`, `docs/`.
- The RTX 5090 backend (`SOLH3_BACKEND=rtx5090`, [docs/RTX5090.md](docs/RTX5090.md)):
  `patches/sol-h3-studio-rtx5090.patch` against `patches/BASE_COMMIT_RTX5090` for NVIDIA's Sol-H3-RTX5090 package,
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
`/api/jobs/<id>?wait=`, labels, limits). `tests/test_cli.py` and `tests/test_mcp.py` run `bin/sol-h3` as a CLI and
as an MCP server against the stub and against small fake spark-switchboard servers (ports 18820-18839), and check
that it stays a lone Python 3.9 file. `tests/test_page.py` drives the page in WebKit, also running
`tests/ui_test.py` against a stub studio with a session loaded (`SOLH3_RESHAPE` 1 and 0), and is skipped without
Playwright (`pip install playwright && playwright install webkit`). One touch test uses Chromium
(`playwright install chromium`) and is skipped without it.

`tests/stub/runtime/` stands in for the patched `runtime` package: loading and generating are `sleep` processes
(`STUB_LOAD_S`, `STUB_CLIP_S`), each clip is a small ffmpeg test pattern in the requested format and length, and a
prompt containing `STUB_FAIL` fails like a dead worker. Its docstring lists the other hooks (a slow close, a clip
finishing as it is cancelled, a worker that starts late, a failure only on the first run). `tests/stub/studio.py`
gives each run its own runtime root, state directory and port under a temp directory; the tests use its
`StubStudio` class. It counts as started only once its own server serves the port, and stops with an error (a
non-zero exit status) if the port is in use or its server stops by itself, so nothing goes to another studio on that
port. Add a test there for every server change.

`tests/test_backend.py` covers `SOLH3_BACKEND` on the CPU: the launcher's `--check` and `--kill` with no backend
(the Spark: its package, the Docker refusal, the 110 GiB floor), with `rtx5090` (a copy of the stub runtime as
`Sol-H3-RTX5090` with a stand-in `runtime/offload.py`; no Docker; the RAM floor; the GPU memory floor driven by a fake
`nvidia-smi` on PATH) and with a wrong or mismatched value; the server's `backend` in `/api/state` and its page; the
page's per-backend texts without a browser (its `MACHINES` table, and its `applyMachine` in node when node is
installed); and every install script's `--dry-run` under both backends, with fake `uname`, `nvidia-smi`, `gcc`,
`docker` and `sg` on PATH and everything in a temp directory; on the RTX 5090 also the caches under `$SOLH3_HOME`,
the GCC 14 toolchain kept on `FORCE=1`, the adoption of a venv whose fake `pip freeze` is the reference freeze, and
the dated prompt-cache backup. Add a test there for every backend change.

`tests/test_replay.py` covers the Stage-2 replay bench (`bench/replay.sh`, `bench/replay/`) on the CPU: the draft
store, every arm, the `--dry` plan, the GPU run object with stand-ins for torch, a `run` with a stand-in Stage-2
process, stopping it under `bench/memguard-run.sh`, the report, the quality step and the SyncNet table. It takes
`runtime/geometry.py` from the patch itself.

The patch has its own 135 CPU tests, run in the patched checkout ([docs/PATCH.md](docs/PATCH.md#apply-test-revert)).
The RTX 5090 patch has 198 (`install/10-sana.sh` with `SOLH3_BACKEND=rtx5090` runs them).

On real hardware: the pipeline needs nearly the whole DGX Spark (~110 GiB free), so run one GPU app at a time. A
second studio next to an installed one needs its own env file (`SOLH3_ENV=<file> bin/sol-h3-studio`) with its own
`SOLH3_PORT`, `SOLH3_STATE_DIR` and `SOL_H3_SPARK_RUNTIME_ROOT`, and `SOLH3_PUBLISH_DIR` empty. Otherwise it shares
the other studio's clip history and files.

## Conventions

- The server stays standard-library only, the page a single file.
- A new setting goes into `bin/env.example` with a comment, and into the server's docstring.
- A change to an endpoint, or to what `bin/sol-h3` does, goes into `docs/API.md`. The prompt guide lives twice, in
  `docs/API.md` (between its `prompt-guide` markers) and as `GUIDE` in `bin/sol-h3`; `tests/test_cli.py` checks
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
