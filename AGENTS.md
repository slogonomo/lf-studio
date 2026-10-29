# Working on LF Studio

Notes for anyone changing this repo, people and coding agents alike. `CLAUDE.md` points here.

## Layout

- `webui/server.py`: the studio server. Python standard library only. `bin/lf-studio` runs it with the Stage1
  Python and the working directory set to the patched Sol-H3-Spark package, from which it imports `runtime.*`.
- `webui/index.html`: the whole page in one file: no framework, no build step, no requests to other sites. The
  server reads it on every request, so edits show up on reload.
- `bin/lf-cli`: the client for agents and scripts, a CLI and an MCP server (`lf-cli mcp`) in one file. People copy
  it to other machines, so it stays standard-library only, Python 3.9 syntax, and reads no other file of this repo.
  `docs/API.md` documents it and the HTTP API. It knows one model, `sol-h3`, the default: everything else about the
  models (which there are, what each makes, their options and sizes) it reads from the studio's table (`/api/state`
  `models`) before a generate that names another model or an option, so a model added to the studio needs no client
  change (`--model`, `--option NAME=VALUE`, the MCP `model` and `options` have no enum).
- `upscale/regen2k_job.py`: the experimental Upscale's GPU job, one upscale per process, run in its own env with
  NVIDIA's Open-H3-Regenerate-2K (its docstring says what is NVIDIA's and what it changes). Standard library outside its
  torch functions, so the server and the tests import it without torch. `upscale/fastload.py`, its loader, is a copy of
  the patch's `runtime/fastload.py`: change the patch's file and copy it (`tests/test_upscale.py` checks that they are
  the same, byte for byte). The server runs it as a job of its own (`POST /api/upscale`, `run_upscale`; off unless
  `LFSTUDIO_UPSCALE=1`), and reads the `LFSTUDIO_UPSCALE_*` settings with its `studio_settings`, as `bin/lf-studio
  --check` does. It takes every finished 5 s clip of the models' `"upscale"` tables in `MODELS` through an input folder
  the server builds per upscale (`upscale_sources`, `build_input`: the clip's MP4, audio and text alone, linked, a
  LightX2V Turbo clip's MP4 stretched or cut to 121 frames, `source.json`), job.json `"case"`; a Keyframes or Reference
  clip's text alone is encoded first by `bench/h3turbo_encode.py --upscale` (`encode_upscale_text`), whose
  `text_only_prompt` and `ref_text_only_prompt` write it (`upscale_text`). The page's Upscale button and popover,
  `lf-cli upscale` and the MCP tool `upscale` use the endpoint; `install/65-regen2k.sh` (optional) builds its
  environment from the hashed lock `install/freeze/regen2k-lock.txt` and fetches its weights
  (`install/regen2k_weights.py`); `--env-only` builds the environment alone.
- `install/66-h3turbo.sh` (optional) sets up the experimental LightX2V Turbo model: it checks the base model (refusing
  one without the VAE's weights), uses the Upscale's environment (or runs `65-regen2k.sh --env-only`; it never touches
  a venv itself), fetches LightX2V's turbo LoRA files at a pinned revision (`install/h3turbo_weights.py`, standard
  library, pinned sizes and SHA-256, no token) and runs the job's `--check-install`.
- `jobs/h3turbo_job.py`: the experimental LightX2V Turbo's GPU job (model id `lightx2v-turbo`, its `MODEL`: the full
  MiniMax-H3 in one stage with a LightX2V turbo LoRA, one clip per process in the Regen-2K environment; its docstring
  is the spec). Standard library outside its torch functions: the server loads it (`LFSTUDIO_H3TURBO_SCRIPT`) for its
  settings (`studio_settings`, as `bin/lf-studio --check` does), presets, memory floor per size and length
  (`min_free_for`), `job.json` check and file names, and runs it per clip (`run_model_job`). Its `estimate_s` is the
  source of the registry's `estimates_s` (a test keeps them equal). `bench/h3turbo_encode.py` is its prompt encode,
  which the server runs before it (`encode_prompt`, `LFSTUDIO_H3TURBO_ENCODE_SCRIPT`; a transient Qwen worker);
  `bench/h3turbo-encode.sh` runs it by hand with the studio's settings.
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

The stub studio has the experimental Upscale on (`LFSTUDIO_UPSCALE=1`), with `tests/stub/upscale_job.py` standing in for
`upscale/regen2k_job.py` (`LFSTUDIO_UPSCALE_SCRIPT`): it prints the job's phase lines over `STUB_UPSCALE_S`, ends at
once on SIGTERM (or ignores it with `STUB_UPSCALE_TERM=ignore`), dies with the server (`--parent`), writes a small
121-frame MP4 without sound and a receipt (none, or one ffmpeg can't read, with `STUB_UPSCALE_OUTPUT`), and fails with
`STUB_UPSCALE_FAIL=1`; it refuses (exit 2) an input folder the real job would (the real job's `case_inputs`, 121 frames,
a conditioning that is not t2va's). The stub runtime writes stand-ins for a clip's working files (`qwen/conditioning.pt`
and the Stage-1 audio) for it to find, and the server reads MemAvailable from a file the stub studio writes
(`LFSTUDIO_TEST_MEMINFO`; `StubStudio.set_memory`), so the memory floor and guard are tested without the machine's
memory. The `Upscale*` classes in `tests/test_server.py` cover the job type (queue and its limit, Cancel and its
SIGKILL, Delete and Undo, a request during a Delete's move, restart, warm-up, memory, failure, the checks at run time,
the settings, the launcher's checks and `--kill`; `UpscaleEveryClip` and `UpscaleEncode` every Sol-H3 clip kind:
Keyframes with each set of keyframes and Reference with their text encoded first, the input folder and source.json, the
statuses in order, Cancel during the encode, a failed encode, the memory floor again after it), `TurboUpscale` LightX2V
Turbo's clips (576p stretched, 768p linked, a last keyframe cut to 121 frames, qwen-text/ or an encode, the model turned
off), `tests/test_api.py` its API record, the `Upscale*` classes in `tests/test_cli.py` and `tests/test_mcp.py` the
client, and those in `tests/test_page.py` the button (disabled with the reason on clips it can't take;
`UpscaleEveryClipPage` and `ModelUpscale` the notes per clip kind, the encode's phase and time, the text an upscale used
and its clip's model; in cards too narrow for the names, Upscale and Delete show icons, so the buttons keep one row;
`OneRowUpscaleOff` checks that row with the Upscale off), the popover, the upscale's card and `tests/ui_test.py` with an
upscale in the gallery.

The experimental model choice (`LFSTUDIO_MODELS`, the registry `MODELS` in the server) is off in the stub studio, as by
default; `StubStudio(models="sol-h3,lightx2v-turbo")` (or `tests/stub/studio.py --models ...`) turns LightX2V Turbo on.
`tests/stub/h3turbo_job.py` stands in for its job, `jobs/h3turbo_job.py` (`LFSTUDIO_H3TURBO_SCRIPT`): the server loads
it (the real job's model id, settings, presets, memory floor, lengths and job.json check) and runs it; it prints the
job's phase lines over `STUB_H3TURBO_S`, writes an MP4 at the clip's canvas with the frames the real job keeps (every
frame made when the clip has a last frame) and a receipt, ends at once on SIGTERM (or ignores it), dies with the server
(`--parent`), and fails, stops for memory (exit 3) or writes no or a bad MP4 on its `STUB_H3TURBO_*` hooks.
`tests/stub/h3turbo_encode.py` stands in for its prompt encode (`LFSTUDIO_H3TURBO_ENCODE_SCRIPT`, by default
`bench/h3turbo_encode.py`), with `STUB_QWEN_*` hooks. The `Turbo*` classes in `tests/test_server.py` cover the registry
in `/api/state` and the page, the refusals, a clip's run (a last frame alone, the lengths), files, record and names, the
queue across models, the warm-ups, Cancel in each step, restart, Delete, memory (the floor per size and length too),
failures, the settings and the launcher's checks; `TurboSeam` checks the job's interface, the real job's and the
stand-in's, that the real job takes every clip the registry offers and gives its estimates, and that the stand-in's
tables and answers are the real job's; and `TurboEncodeSeam` the real prompt encode's (the arguments the server gives
it, its refusal line, the files and keys the server reads); `ModelChoiceOff` runs v0.5.1's server from git next to this
one with the model choice off and compares what they answer, value by value, and v0.5.1's `lf-cli` with this one;
`ServerParts` compiles parts of the server (`stop_job`, `registry_problems`) from its source with stand-ins, for what a
stub studio can't reach on cue. `tests/test_api.py` has the API records (`Model*`). The `Model*` classes in
`tests/test_page.py` cover the page: the Model row and the Size row, what LightX2V Turbo doesn't make yet greyed out
with the reason (Reference and 15 s, and 768p at 12 s, a size and length it makes apart only; with `not_yet_table`, its
table before its GPU validation put into `/api/state`, also lengths and a last frame alone: no field changes another,
Generate is disabled and says why), the header, the times per size and length and Generate's label per model, the
request, the choices kept, the card's model tag, its Upscale button and download name, Remix, a clip's progress and the
status line, Cancel, and a clip of a model that is off now; `ModelChoiceOffPage` loads this page and v0.5.1's (from git)
side by side with the model choice off and compares their pixels, texts, requests and kept choices after the same
clicks; `MachineTexts` checks Sol-H3's times in the Model row per backend; `KeyframeLine` the fl2va keyframe line that
the chip, the Example and the placeholder write for the keyframes attached (the prompt guide's line for one keyframe,
the page's two-picture line for both). `tests/ui_test.py` checks the Model row when a studio has one (`UiTestScript`
runs it with a LightX2V Turbo clip). The client: `Models`, `ModelsRunning`, `ModelsOff` and `ModelsOldStudio` in
`tests/test_cli.py` (`lf-cli models`, `generate --model` with the model's options and its files, every refusal before
anything is sent, the shape warning at the model's size, status and cancel while a clip runs, a studio with the model
choice off, an older studio and one that spark-switchboard has stopped: fake servers), and `Models` and `ModelsOff` in
`tests/test_mcp.py` (the `models` tool, `generate` with a model, the repeat key with the model and its options, the
refusals; `FakeStudio` has the stopped studio).

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
must never send a token, also through a redirect), and `install/66-h3turbo.sh` (`InstallH3Turbo`: its dry run, the
base model without VAE weights refused, the Upscale's environment used as it is, built by `65-regen2k.sh --env-only`
with a fake uv, or one of your own; a failed step stops it) and `install/h3turbo_weights.py` (`H3TurboWeights`: the
pinned files, kept, resumed by Range or downloaded and checked, from a fake Hugging Face that redirects to another
host name, with no token ever sent). None of them downloads anything.

`tests/test_old_names.py` covers the names up to 0.4 that work until 0.6: the old setting names in the environment
and the settings file (and that the four `OLD_SETTINGS` lists agree), the old places, the shims, the launcher's check
for a running copy, a launcher run with the 0.4 spark-switchboard service file's settings, the deprecated integration
files, the install steps and the page's old localStorage keys. It goes with them in 0.6.

`tests/test_replay.py` covers the Stage-2 replay bench (`bench/replay.sh`, `bench/replay/`) on the CPU: the draft
store, every arm, the `--dry` plan, the GPU run object with stand-ins for torch, a `run` with a stand-in Stage-2
process, stopping it under `bench/memguard-run.sh`, the report, the quality step and the SyncNet table. It takes
`runtime/geometry.py` from the patch itself.

`tests/test_upscale.py` covers `upscale/regen2k_job.py` on the CPU (and that `upscale/fastload.py` is the patch's
`runtime/fastload.py`): job.json, the clip's and the weights' files (5 s 16:9 and 9:16 MP4s made by ffmpeg), the shape
table (with every orientation the patches make), the portrait entries for NVIDIA's dicts, the upscale route and
`check_package`, the text check on stand-in tensors, the command lines (the mux run with ffmpeg) and the environment,
the progress lines, the memory trace, `--check`, the receipt when a job is done, fails, is refused or gets SIGTERM,
`--parent` (the job dies with the server), and `run()` with a stand-in torch and a stand-in of NVIDIA's `open_h3`, with
both loads (fast and standard: stand-ins for safetensors, diffusers and `fastload` record every load, which the fast
load must have replaced before the first one): a 16:9 clip makes the same calls into NVIDIA's code as before the job
took 9:16, a 9:16 one its portrait route (the same with either load). With `LFSTUDIO_UPSCALE_PKG` naming NVIDIA's real
package it also checks the files the job pins and the entries it was written against (read, not imported). Its torch
tests (the merger, the lean block forward against a stock-style block, the tensor checks, the column crop) run where
torch is importable (CPU is enough) and are skipped otherwise.

`tests/test_h3turbo_job.py` covers `jobs/h3turbo_job.py` on the CPU: job.json and its refusals, the geometry (every
length against the patch's `runtime/geometry.py`, the frames a clip with a last frame keeps, the layout rows against the
research probe's GPU receipts), the studio's model of a clip (memory floor per size and length, estimates) against the
GPU validation's receipts, how a keyframe is fitted to the canvas, the sigmas, the LoRA header rules on sparse stand-in
files, the conditioning checks, the settings and `--check-install`, the command line, the environment and the progress
lines against `upscale/regen2k_job.py`'s, the memory trace, the fast load with stand-in loaders, `--check`, the receipt
when a job is done, fails, is refused, is aborted for memory or gets SIGTERM, and `--parent`. Its torch parts (the LoRA
hooks against PEFT, the conditioning tensors, and `run()` itself on a tiny random-weight MiniMax-H3 at the real 576p
canvases: t2va, a first frame, first + last and a last frame alone (the MP4 ends on the frame the last keyframe is
anchored to), the memory abort and SIGTERM; on a small canvas 15 s, past diffusers' stock 15.0 s, with the limit raised
for that clip and refused without it, and 14 s as stock) need torch and diffusers 0.40.0 and are skipped otherwise; run
them with the Regen-2K Python, CUDA hidden (the command is in the file's docstring; about 2.5 min, ~6.5 GB of memory: on
a DGX Spark where a GPU job runs, `systemd-run --user --scope -p MemoryMax=14G` keeps them from taking its memory).
`tests/test_h3turbo_encode.py` covers `bench/h3turbo_encode.py` and its wrapper, standard library only: the text-only
rewrite on every alignment-line form the guide and the page write, the Upscale's rewrite of a Reference prompt
(`ref_text_only_prompt`: the guide's and the page's examples and the other forms) and its `--upscale` encode, the prompt
as the server takes it, the refusals, `--check`, and the encodes against a stand-in of the patch's `runtime` package
kept in the test file itself (`TurboEncodeSeam` in `tests/test_server.py` runs it with the server's arguments, the
Upscale's too).

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
