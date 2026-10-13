# Working on Sol-H3 Studio

Notes for anyone changing this repo, people and coding agents alike. `CLAUDE.md` points here.

## Layout

- `webui/server.py`: the studio server. Python standard library only. `bin/sol-h3-studio` runs it with the Stage1
  Python and the working directory set to the patched Sol-H3-Spark package, from which it imports `runtime.*`.
- `webui/index.html`: the whole page in one file: no framework, no build step, no requests to other sites. The
  server reads it on every request, so edits show up on reload.
- `bin/sol-h3-studio` (launcher), `bin/env.example` (every setting, with comments), `install/` (00-90),
  `patches/sol-h3-studio.patch` against NVlabs/Sana at `patches/BASE_COMMIT` ([docs/PATCH.md](docs/PATCH.md)),
  `integrations/`, `bench/`, `docs/`.

## Tests

None of these needs the GPU:

```bash
python3 -m unittest discover -s tests -v       # server and page against the stub runtime (needs ffmpeg)
python3 tests/stub/studio.py                   # a stub studio on http://127.0.0.1:18299/ to click through
python3 tests/ui_test.py http://127.0.0.1:18299/   # a general check of the page against any studio
```

`tests/test_server.py` drives the server over HTTP; `tests/test_page.py` drives the page in WebKit and is skipped
without Playwright (`pip install playwright && playwright install webkit`). One touch test uses Chromium
(`playwright install chromium`) and is skipped without it.

`tests/stub/runtime/` stands in for the patched `runtime` package: loading and generating are `sleep` processes
(`STUB_LOAD_S`, `STUB_CLIP_S`), each clip is a small ffmpeg test pattern in the requested format and length, and a
prompt containing `STUB_FAIL` fails like a dead worker. Its docstring lists the other hooks (a slow close, a clip
finishing as it is cancelled, a worker that starts late, a failure only on the first run). `tests/stub/studio.py`
gives each run its own runtime root, state directory and port under a temp directory; the tests use its
`StubStudio` class. It counts as started only once its own server serves the port, and stops with an error (a
non-zero exit status) if the port is in use or its server stops by itself, so nothing goes to another studio on that
port. Add a test there for every server change.

The patch has its own 97 CPU tests, run in the patched checkout ([docs/PATCH.md](docs/PATCH.md#apply-test-revert)).

On real hardware: the pipeline needs nearly the whole DGX Spark (~110 GiB free), so run one GPU app at a time. A
second studio next to an installed one needs its own env file (`SOLH3_ENV=<file> bin/sol-h3-studio`) with its own
`SOLH3_PORT`, `SOLH3_STATE_DIR` and `SOL_H3_SPARK_RUNTIME_ROOT`, and `SOLH3_PUBLISH_DIR` empty. Otherwise it shares
the other studio's clip history and files.

## Conventions

- The server stays standard-library only, the page a single file.
- A new setting goes into `bin/env.example` with a comment, and into the server's docstring.
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
