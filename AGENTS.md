# Working on LF Studio

Notes for anyone changing this repo, people and coding agents alike. `CLAUDE.md` points here.

## Commands

- All CPU tests (needs ffmpeg; about 30 min): `python3 -m unittest discover -s tests`
- One module or class: `python3 -m unittest tests.test_server.ServerParts`. Run the ones your change touches.
- Page tests in WebKit (skipped without Playwright): `pip install playwright && playwright install webkit`
- Torch parts of `tests/test_upscale.py` and `tests/test_h3turbo_job.py`: run them with the Regen-2K Python and
  `CUDA_VISIBLE_DEVICES=` (each file's docstring has the command).
- A stub studio to click through: `python3 tests/stub/studio.py` (http://127.0.0.1:18299/), then
  `python3 tests/ui_test.py http://127.0.0.1:18299/`
- The patch's own tests: [docs/PATCH.md](docs/PATCH.md#apply-test-revert).

## Conventions

- `webui/server.py` is standard library only. `webui/index.html` is one file: no framework, no build step, no requests
  to other sites.
- `bin/lf-cli` stays one standard-library file in Python 3.9 syntax that reads no other file of the repo. People copy
  it to other machines.
- `upscale/regen2k_job.py` and `jobs/h3turbo_job.py` use the standard library outside their torch functions, because
  the server and the tests import them without torch.
- `upscale/fastload.py` is a copy of the patch's `runtime/fastload.py`. Change the patch's file, then copy it.
- The prompt guide lives twice: in `docs/API.md` between its `prompt-guide` markers, and as `GUIDE` in `bin/lf-cli`.
- A new setting goes into `bin/env.example` with a comment, and into the server's docstring.
- A change to an endpoint or to `bin/lf-cli` goes into `docs/API.md`.
- Add a test against the stub studio (`tests/stub/`) for every server change, and one in `tests/test_backend.py` for
  every backend change.
- The docs give measured numbers and say where they were measured.
- When a GPU run lands, update:
  - `UPSCALE_RAN`, `OFFER_ETA_S` and the memory model in `upscale/regen2k_job.py`;
  - `PIN_ROUTES` `"ran"` in `webui/server.py`;
  - `docs/PERFORMANCE.md`.
  Without a line there, the page notes that a route has not run on the GPU.
- Never commit generated clips (or screenshots that show them), model weights, run state, local settings, or
  personal details such as home paths, host names or addresses.

## Gotchas

- `PIN_ROUTES` keeps one route per line, even past 120 characters, because the tests rewrite it line by line.
- The project was Sol-H3 Studio up to 0.4. Only the stop reads the old names (`SOLH3_*`, `sol-h3-studio` paths):
  `install/old-names.sh`, `install/env.sh` and `old_names` in `bin/lf-cli`. `SOL_H3_*` and `COMFY_SOLH3_*` are
  other projects' names and are fine. The patches' compile cache still falls back to `~/.cache/sol-h3-studio` when
  the runtime root is unset (changing that is a patch change).
- The RTX 5090 backend only applies with `LFSTUDIO_BACKEND=rtx5090`. Without that setting, every path must stay the
  DGX Spark's.
- On a DGX Spark the pipeline needs about 110 GiB free, so run one GPU app at a time.
- A second studio next to an installed one needs its own env file (`LFSTUDIO_ENV=<file> bin/lf-studio`) with its own
  `LFSTUDIO_PORT`, `LFSTUDIO_STATE_DIR` and `SOL_H3_SPARK_RUNTIME_ROOT`, and an empty `LFSTUDIO_PUBLISH_DIR`.
  Otherwise it shares the other studio's clips.

## Branches and releases

1. Each issue gets a branch from the current `release/x.y` and a PR into it that says "Part of #N". These PRs are
   squash-merged.
2. The release branch is tested as a whole, on a separate studio.
3. A "Release x.y.0" PR merges it into `main` with a merge commit and one `Fixes #N` line per issue. Then
   `git tag -a vx.y.0` and `gh release create vx.y.0 --generate-notes`.
4. Installations follow `main`: `git pull`, then `install/10-sana.sh` if the patch changed, then a restart.
