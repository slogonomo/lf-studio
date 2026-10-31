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
  --check` does. It takes every finished clip of the models' `"upscale"` tables in `MODELS` (every length the studio
  makes; not Sol-H3's old 3 and 4 s clips), whole, at the sizes the job offers for the clip's length (its `offer_row`
  and `offers_table`, which `/api/state` `upscale_offers` carries to the page; a size not offered is refused with the
  reason) and with the job's memory floor for its case (`min_free_for`, job.json `min_free_gib`), through an input
  folder the server builds per upscale (`upscale_sources`, `build_input`: the clip's MP4, audio and text alone, linked,
  a LightX2V Turbo clip's MP4 stretched, or a 5 s one with a last keyframe cut to 121 frames at 4K (`UPSCALE_KEEP_ALL`:
  it keeps its 124 at 1080p and 2K), the pictures of the ends Keep my frames sends (`clip_keyframes`, `PIN_ROUTES`,
  job.json `"pin"`), `source.json`), job.json `"case"`; a Keyframes or Reference clip's text alone is encoded first by
  `bench/h3turbo_encode.py --upscale` (`encode_upscale_text`), whose `text_only_prompt` and `ref_text_only_prompt` write
  it (`upscale_text`). The page's Upscale button and popover, `lf-cli upscale` and the MCP tool `upscale` use the
  endpoint; `install/65-regen2k.sh` (optional) builds its environment from the hashed lock
  `install/freeze/regen2k-lock.txt` and fetches its weights (`install/regen2k_weights.py`); `--env-only` builds the
  environment alone. What has run on the GPU through the studio is listed three times: per length and size in the job's
  `UPSCALE_RAN` (`upscale_offers` `"ran"`; 6, 10 and 15 s at 1080p and 10 s at 2K ran, and the other lengths offered are
  there by the offer rule, a longer clip at their size having run, but 2K at 12 and 15 s, which has not run and runs the
  `lean2` forward past every case that has, `LEAN2_ROWS`; one line each), per clip route (at 5 s) in the page's
  `UP_RAN`, per Keep my frames route, size and length in `PIN_ROUTES`' `"ran"` (a size alone at 5 s,
  `"<size>/<seconds>"` at another length, `"1080p/10"` so far; one line per route, so a route that fails its GPU run
  goes out with its line; `PIN_SIZES` holds the sizes; no 2K line since Keep my frames at 2K went through the cascade,
  below). The rest shows the page's "not run on the GPU" note. Keep my frames at 2K is the job's cascade (`route_for`,
  `CASCADE_FROM`, `CASCADE_SIGMA`: the pinned refine at 1080p as a 1080p upscale runs it, NVIDIA's latent upscaler to
  the 2K grid, one step from 0.4 at 2K with the pin; receipt `"route"` and `"cascade"`): no request field, the server
  sends the pin as at 1080p (the record's `"route"`, upscalable's `pin_routes` `"route"`); its time is
  `cascade_eta_s` (`upscale_offers` `"cascade_eta_s"`, which the page's `upEta` and the 2K label take with the box
  ticked), its floor the larger refine's (`route_need_gib`).
  `OFFER_ETA_S` has the measured times where a run is and the others fitted to them, the memory model (`peak_gib`,
  `overhead_gib`, `need_gib`) is fitted to the runs (each 2026-10-30 run's lowest MemAvailable within 0.5 GiB,
  `tests/test_upscale.py`; `lean2`'s `ROW_BYTES` and `OVERHEAD_GIB` are estimates until it runs), and the docs say
  which lengths ran and which follow the model
  ([docs/PERFORMANCE.md](docs/PERFORMANCE.md#longer-clips), the README): change them when a run lands. The page's
  `UP_PIN_S` holds the seconds Keep my frames adds at 5 s at 1080p; the page adds them at every length (a little short
  on a longer clip, as the docs say). Change it when a 5 s run lands, not for a longer one. At 2K the time is the
  job's model of the cascade (`STEP_MODEL`, `CASCADE_UPSCALE_S`, `PIN_ENCODE_S`): change them when its runs land.
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
- The project was called Sol-H3 Studio up to 0.4. 0.6 no longer reads its old names. Where 0.5 still read one, a
  stop names each one in use with its new name and exits before anything starts
  ([docs/INSTALL.md](docs/INSTALL.md#from-05-to-06)): `install/old-names.sh` for the launcher (every mode) and
  `install/env.sh` (the install steps and `bench/*.sh`; also the venvs' 0.4 done marker and the checkout's 0.4 copy of
  the applied patch), and `old_names` in `bin/lf-cli` (its settings, its old name `sol-h3`, the MCP server's
  `./sol-h3-clips`). Nothing else may read an old name; the stop goes in a later release. The patches' compile cache
  still falls back to `~/.cache/sol-h3-studio` when the runtime root is unset (changing that is a patch change).
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
(`STUB_LOAD_S`, `STUB_CLIP_S`), each clip is a small ffmpeg test pattern in the requested format and length (the frames
the patch keeps: 121 at 5 s, 153 at 6 s, 241 at 10 s), and a prompt containing `STUB_FAIL` fails like a dead worker. Its
docstring lists the other hooks (a slow close, a clip finishing as it is cancelled, a worker that starts late, a failure
only on the first run). `tests/stub/studio.py` gives each run its own runtime root, state directory and port under a
temp directory; the tests use its `StubStudio` class. It counts as started only once its own server serves the port, and
stops with an error (a non-zero exit status) if the port is in use or its server stops by itself, so nothing goes to
another studio on that port. Add a test there for every server change.

The stub studio has the experimental Upscale on (`LFSTUDIO_UPSCALE=1`), with `tests/stub/upscale_job.py` standing in for
`upscale/regen2k_job.py` (`LFSTUDIO_UPSCALE_SCRIPT`): it prints the job's phase lines over `STUB_UPSCALE_S`, ends at
once on SIGTERM (or ignores it with `STUB_UPSCALE_TERM=ignore`), dies with the server (`--parent`), writes a small MP4
without sound with the clip's frames and a receipt (none, or one ffmpeg can't read, with `STUB_UPSCALE_OUTPUT`) with the
real job's `frames`, `route`, `rows` and `forward` and, with a pin, its `pin` (each picture checked by its sha256, then
33 dB, or what `STUB_UPSCALE_PIN_DB` says; at 2K with an end held the cascade's receipt and 3 steps), and fails with
`STUB_UPSCALE_FAIL=1`; it refuses (exit 2) an input folder the real job would (the real job's `case_inputs` and
`parse_pin`, a count of frames it has no case for, a size it does not offer for the clip's length, a conditioning that
is not t2va's, and MemAvailable from `LFSTUDIO_TEST_MEMINFO` below job.json's `min_free_gib`). The stub runtime writes
stand-ins for a clip's working files (`qwen/conditioning.pt` and the Stage-1 audio) for it to find, and the server reads
MemAvailable from a file the stub studio writes (`LFSTUDIO_TEST_MEMINFO`; `StubStudio.set_memory`), so the memory floor
and guard are tested without the machine's memory. The stub runtime writes `stage2/stage2_result.json` with the Stage-2
pin's record (`stage2_keyframe`, the capture's `input_sha256`; `STUB_STAGE2_KEYFRAME=absent` for a clip made before
0.5.2), and the stand-in LightX2V Turbo job its receipt's `geometry`, `images` and `inputs_sha256`, which Keep my frames
reads. The `Upscale*` classes in `tests/test_server.py` cover the job type (queue and its limit, Cancel and its SIGKILL,
Delete and Undo, a request during a Delete's move, restart, warm-up, memory, failure, the checks at run time, the
settings, the launcher's checks and `--kill`; `Upscale` and `UpscaleRefusals` clips of every length, whole, at the sizes
offered for it (`upscale_offers`, the output's frames and whole sound) and a size refused with the reason;
`UpscaleOddRecords` records refused, never cut (Sol-H3's old 3 and 4 s clips, an MP4 of frames the job has no case for,
none in its name, or another length's); `UpscaleMemory` and `UpscaleMemoryLowSetting` the floor per case (job.json
`min_free_gib`; with Keep my frames too, at 2K its cascade's larger refine); `UpscaleEveryClip` and
`UpscaleEncode` every Sol-H3 clip kind: Keyframes with each set of keyframes (also a prompt with the page's CRLF line
breaks) and Reference with their text encoded first (also prompts whose subject definitions chain or are too long as
text alone), the input folder and source.json, the statuses in order, the clip deleted afterwards (the upscale still
plays), no t2va manifest, Cancel during the encode, a failed encode, the server's own checks of its receipt
(`STUB_QWEN_FAIL` `no-conditioning`, `other-prompt`, `other-task`), the memory floor again after it;
`BadUpscaleSettings` the start's refusals, the prompt encode and its rewrites too; `UpscaleEveryClip` also Keep my
frames on Sol-H3's clips: the first frame where Stage 2 held it, the reasons where not, never the last, its length's
frames short of it; long clips of every kind), `TurboUpscale` LightX2V Turbo's clips (576p stretched, its sound as it
is; 768p linked, a last keyframe's 124 frames whole at 1080p and 2K with the clip's whole sound and cut to 121 at 4K,
every frame of a long one, qwen-text/ or an encode, the model turned off), `TurboPin` Keep my frames on its Keyframes
clips (each set of ends, job.json `pin`, the links, source.json, the record's result, where no end is sent, a picture
gone or changed before and after the request, the clip's receipt, also one of another shape; at 2K the cascade: the
record's `route` before and after, job.json as at 1080p, the stand-in's cascade receipt and 3 steps, `whole` where no
end is held), `ServerParts` also the
receipts' records (`keyframe_record`, also of another shape: never an error), the picture hash cache (`image_state`:
never a hash for `/api/state`), `PIN_ROUTES` as it is and changed by one line (a route out, a route's `"ran"` sizes and
lengths, 2K out; no 2K `"ran"`, the cascade's `"route"` at 2K), the seam between the models' `"upscale"` tables and the
job's frame cases and offers (every model, task, length and set of keyframes; the stub runtime's frames too),
`upscale_frames`, `sol_last`, `offer_refusal` and the stand-in job's refusals, `tests/test_api.py` its API record, the
`Upscale*` classes in `tests/test_cli.py` and `tests/test_mcp.py` the client (a size refused for a clip's length with
the studio's reason; `UpscaleTimes` the texts' times against the job's `OFFER_ETA_S`, and Keep my frames' at 2K against
its `cascade_eta_s`; also `--pin` and the MCP `pin`; that pin goes only to a studio with
`upscale_pin`, and again without it for a stopped studio that does not know it, against fake servers: `Switchboard` in
`tests/test_cli.py`, `FakeStudio` in `tests/test_mcp.py`), and those in `tests/test_page.py` the button (disabled with
the reason on clips it can't take; `UpscaleEveryClipPage` and `ModelUpscale` the notes per clip kind, the note that a
route has not run on the GPU through the studio (on the routes the page's `UP_RAN` leaves out), and the one that a
length has not (its offer's `"ran"`), a long clip's times and the sizes its length does not have greyed with the reason
(and the popover of a 12 s clip, at 2K with its "not run" note and at 1080p without; `UpscaleOldClipPage` a 3 s
clip's Upscale disabled with the reason), the encode's phase and time, the text an upscale used and its clip's model;
in cards too narrow for the names, Upscale and Delete show icons, so the buttons keep one row; `OneRowUpscaleOff`
checks that row with the Upscale off), the popover
(`ModelUpscale` its notes per size and Keep my frames: the box per ends, off with the reason at 4K and other strengths,
at 2K as at 1080p with its line that it is made at 1080p first there, its note only at a size where its route has not
run, `pin` in the request, the card's result; `UpscaleEveryClipPage` 2K's time with the box ticked and not, the
cascade's; `UpscalePage` `upEta` with the record's route), the
upscale's card and `tests/ui_test.py` with an upscale in the gallery (and the popover of a Keyframes clip, with its Keep
my frames box).

The experimental model choice (`LFSTUDIO_MODELS`, the registry `MODELS` in the server) is off in the stub studio, as by
default; `StubStudio(models="sol-h3,lightx2v-turbo")` (or `tests/stub/studio.py --models ...`) turns LightX2V Turbo on.
`tests/stub/h3turbo_job.py` stands in for its job, `jobs/h3turbo_job.py` (`LFSTUDIO_H3TURBO_SCRIPT`): the server loads
it (the real job's model id, settings, presets, memory floor, lengths and job.json check) and runs it; it prints the
job's phase lines over `STUB_H3TURBO_S`, writes an MP4 at the clip's canvas with the frames the real job keeps (every
frame made when the clip has a last frame) and a receipt, ends at once on SIGTERM (or ignores it), dies with the server
(`--parent`), and fails, stops for memory (exit 3) or writes no MP4, a bad one or one of the wrong size on its
`STUB_H3TURBO_*` hooks. `tests/stub/h3turbo_encode.py` stands in for its prompt encode
(`LFSTUDIO_H3TURBO_ENCODE_SCRIPT`, by default `bench/h3turbo_encode.py`), with `STUB_QWEN_*` hooks. The `Turbo*` classes
in `tests/test_server.py` cover the registry in `/api/state` and the page, the refusals, a clip's run (a last frame
alone, the lengths, dialogue tokens), files, record and names, the queue across models, the warm-ups, Cancel in each
step, restart, Delete, memory (the floor per size and length too), failures, the settings and the launcher's checks;
`TurboSeam` checks the job's interface, the real job's and the stand-in's, that the real job takes every clip the
registry offers and gives its estimates, and that the stand-in's tables and answers are the real job's; and
`TurboEncodeSeam` the real prompt encode's (the arguments the server gives it, its refusal line, the files and keys the
server reads); `ModelChoiceOff` runs v0.5.1's server from git next to this one with the model choice off and
compares what they answer, value by value, and v0.5.1's `lf-cli` with this one;
`ServerParts` compiles parts of the server (`stop_job`, `registry_problems`, `model_refusal` on a table of two rows,
`link_or_copy` across filesystems, `prepare_video_command` on ffmpeg-made clips) from its source with stand-ins, for
what a stub studio can't reach on cue. `tests/test_api.py` has the API records
(`Model*`). The `Model*` classes in `tests/test_page.py` cover the page: the Model row and the Size row, what LightX2V
Turbo doesn't make yet greyed out with the reason (Reference; with `not_yet_table`, its table before its GPU validation
put into `/api/state`, also lengths and a last frame alone; with `apart_table`, its table before 15 s and 768p 12 s ran
on the GPU, 15 s and 768p at 12 s, a size and length it made apart only: no field changes another, Generate is disabled
and says why), the header, the times per size and length and Generate's label per model, the request, the choices kept,
the card's model tag, its Upscale button and download name, Remix, a clip's progress and the status line, Cancel, and a
clip of a model that is off now; `ModelChoiceOffPage` loads this page and v0.5.1's (from git) side by side with the
model choice off and compares their pixels, texts, requests and kept choices after the same
clicks; `MachineTexts` checks Sol-H3's times in the Model row per backend; `KeyframeLine` the fl2va keyframe line that
the chip, the Example and the placeholder write for the keyframes attached (the prompt guide's line for one keyframe,
the page's two-picture line for both). `tests/ui_test.py` checks the Model row when a studio has one (`UiTestScript`
runs it with a LightX2V Turbo clip). The client: `Models`, `ModelsRunning`, `ModelsOff` and `ModelsOldStudio` in
`tests/test_cli.py` (`lf-cli models`, `generate --model` with the model's options and its files, every refusal before
anything is sent, also of values a table makes apart only (a fake server), the shape warning at the model's size, status
and cancel while a clip runs, a studio with the model choice off, an older studio and one that spark-switchboard has
stopped: fake servers), and `Models` and `ModelsOff` in `tests/test_mcp.py` (the `models` tool, `generate` with a model,
the repeat key with the model and its options, the refusals; `FakeStudio` has the stopped studio).

`tests/test_backend.py` covers `LFSTUDIO_BACKEND` on the CPU: the launcher's `--check` and `--kill` with no backend
(the Spark: its package, the Docker refusal, the 110 GiB floor), with `rtx5090` (a copy of the stub runtime as
`Sol-H3-RTX5090` with a stand-in `runtime/offload.py`; no Docker; the RAM floor; the GPU memory floor driven by a fake
`nvidia-smi` on PATH) and with a wrong or mismatched value; `SOL_H3_STAGE2_KEYFRAME`'s default on both backends in
the launcher's and the server's start checks (pin, refused with an older patch; off set explicitly starts); the
server's `backend` in `/api/state` and its page; the
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

`tests/test_old_names.py` covers the stop for the names up to 0.4: the launcher in every mode (start, `--check`,
`--kill`, the client commands) with an old setting in the environment and in the settings file (file and line named,
as a key and in a value, also one that nothing sets: no parse error, the old places still in the same message), the
file that `SOLH3_ENV` names, the old settings and state folders (and when they don't count: a setting, a new folder
with content, a link left at the old place), a settings file named in the old folder after its move, a run through a
link named `sol-h3-studio`; the install steps, `bench/replay.sh` and
`bench/first-clip.sh` (`SOLH3_DRY_RUN=1` alone does not run a step, the venvs' 0.4 done markers, a checkout with only
the 0.4 copy of the applied patch and the rename it asks for); `lf-cli` and `lf-cli mcp` (exit 2, nothing on stdout,
no connection, the `sol-h3` name, `./sol-h3-clips`); names that only look alike (`SOL_H3_*`, `COMFY_SOLH3_*`, NVIDIA's
own `SOLH3_*`) pass; the upgrade section's table has every old setting; the files of the old names are gone and no
other code names an old name. It goes with the stop.

`tests/test_replay.py` covers the Stage-2 replay bench (`bench/replay.sh`, `bench/replay/`) on the CPU: the draft
store, every arm, the `--dry` plan, the GPU run object with stand-ins for torch, a `run` with a stand-in Stage-2
process, stopping it under `bench/memguard-run.sh`, the report, the quality step and the SyncNet table. It takes
`runtime/geometry.py` from the patch itself.

`tests/test_upscale.py` covers `upscale/regen2k_job.py` on the CPU (and that `upscale/fastload.py` is the patch's
`runtime/fastload.py`): job.json (its `pin` and `min_free_gib` too), the clip's and the weights' files (16:9 and 9:16
MP4s made by ffmpeg of every count of frames the studio's lengths keep, and the counts, sizes and rates it refuses),
the frames of each length (`FRAME_CASES`, derived from both patches' `runtime/geometry.py` and `jobs/h3turbo_job.py`
at every length: the 3 copies of the last frame, the audio of A × 800 samples), what is offered per length and size
(`OFFER_ETA_S`, `offers_table`, `offer`: 1080p and 2K every length, 4K at 5 s, the reasons; the measured times and the
others fitted to them; `UPSCALE_RAN`, one line each, 2K at 12 and 15 s not in it; the cascade's time, `cascade_eta_s`,
by the step model), the block forward by rows (`forward_for`, `lean_chunks_for`: every case that ran keeps its forward,
`lean2` only past them) and the memory model and floor (`case_rows`, `peak_gib`, `overhead_gib`, `need_gib`,
`route_need_gib`, `min_free_for`; against the job's runs of every length; at 2K 15 s a floor of 115 GiB, 116 with a pin,
and none above 116), the route per size, pin and length (`route_for`, `plan`: only a pinned 2K cascades), the shape
table (with every orientation the patches make), the portrait entries for NVIDIA's dicts, the upscale route and
`check_package` (every pinned file for a 9:16 clip and for a clip of another length), the text check on stand-in
tensors, the command lines (NVIDIA's read of the clip, the others without its tpad: a clip cut off after its header
reads short; the mux run with ffmpeg) and the environment, the progress lines, the memory trace, `--check` (of every
count of frames, at every size: a size its length has not got refused with the reason), the receipt when a job is done,
fails, is refused, starts with too little memory or gets SIGTERM, `--parent` (the job dies with the server), v0.5.2's
job from git refusing a long clip and `min_free_gib` (`OlderJobTests`: so job.json stays version 1), and `run()` with a
stand-in torch and a stand-in of NVIDIA's `open_h3`, with both loads (fast and standard: stand-ins for safetensors,
diffusers and `fastload` record every load, which the fast load must have replaced before the first one): a 16:9 clip
makes the same calls into NVIDIA's code as before the job took 9:16, a 9:16 one its portrait route (the same with either
load), a clip of 124 frames and clips of other lengths the job's copies of NVIDIA's encode and decode (their audio and
silence, the lean blocks by their rows, NVIDIA's upscaler.py and upscaler_node.py pinned, an upscaled latent of another
T failing before the refine), every 5 s route the block forward it ran on the GPU, 2K at 12 and 15 s the lean blocks and
`install_lean2`, and a pin the pinned refine with the ends its check kept (a 10 s clip's too); at 2K the cascade
(`test_the_cascade`, 16:9 and 9:16, 5, 6, 10, 12 and 15 s): its 1080p refine call for call the 1080p pinned run's, then
NVIDIA's upscaler to the 2K grid and one step at [0.4, 0] with the pictures at 2K, the forward per stage (`set_forward`:
at 12 and 15 s lean at 1080p, then lean2 for the 2K step), the phase lines, the receipt's `route` and `cascade`. The
pin's fit (`cover_crop`, `fit_box`) against the LightX2V Turbo receipts' records, "as is" too, and the grids are
standard library. With `LFSTUDIO_UPSCALE_PKG` naming NVIDIA's real package it also checks the files the job pins, the
entries it was written against and its encode command (read, not imported). Its torch tests (the merger, the lean block
forward against a stock-style block, the tensor checks, the column crop) run where torch is importable (CPU is enough)
and are skipped otherwise. With Pillow and numpy (the Regen-2K environment) the pin's check runs on ffmpeg-made clips
(each reason an end is not kept, the EXIF turn, the fit at the grid, `--check`); with diffusers and imageio-ffmpeg too
and NVIDIA's package, the job's copies of NVIDIA's encode and decode against NVIDIA's own run on the CPU with a stand-in
VAE (bit for bit at 121 frames; other lengths too), diffusers' VAE chunks and H3's frame rules (`DiffusersRuleTests`:
every case's T and A), `encode_audio` with a small audio VAE of diffusers' against NVIDIA's silence, NVIDIA's t2va
layout of every frame case, the pin's layout at 1080p and 2K (the last block 205 after the text at 5 s, and at every
length) and the pinned refine with a stand-in transformer: without a keyframe NVIDIA's `Pipeline.refine` bit for bit,
with a first frame the bench wrapper's K1 refine (copied into the test) bit for bit, also at the cascade's 2K sigmas
(one step from 0.4, NVIDIA's noise at the latent's shape: the research runner's 2K step), and with both ends (at 5 s and
10 s) the first block unchanged, the condition rows never written and only the rows after them stepped; and `lean2`
(`Lean2Tests`) on a tiny random-weight MiniMax-H3 transformer of diffusers' own class, through NVIDIA's
`Pipeline.refine` and the pinned refine: the lean blocks' result bit for bit, the video embeds bf16 and emptied once the
first block runs, and diffusers' forward reading them only where it scatters them (its source). Run those with the
Regen-2K Python, CUDA hidden (the command is in the file's docstring).

`tests/test_h3turbo_job.py` covers `jobs/h3turbo_job.py` on the CPU: job.json and its refusals, the geometry (every
length against the patch's `runtime/geometry.py`, the frames a clip with a last frame keeps, the layout rows against the
research probe's GPU receipts), the studio's model of a clip (memory floor per size and length, estimates) against the
GPU validation's receipts and the later runs of 576p 15 s and 768p 12 and 15 s, how a keyframe is fitted to the canvas,
the sigmas, the LoRA header rules on sparse stand-in files, the conditioning checks, the settings and `--check-install`,
the command line, the environment and the progress lines against `upscale/regen2k_job.py`'s, the memory trace, the fast
load with stand-in loaders, `--check`, the receipt when a job is done, fails, is refused, is aborted for memory or gets
SIGTERM, and `--parent`. Its torch parts (the LoRA hooks against PEFT, the conditioning tensors, and `run()` itself on a
tiny random-weight MiniMax-H3 at the real 576p canvases: t2va, a first frame, first + last and a last frame alone (the
MP4 ends on the frame the last keyframe is anchored to), the memory abort and SIGTERM; on a small canvas 15 s, past
diffusers' stock 15.0 s, with the limit raised for that clip and refused without it, and 14 s as stock) need torch and
diffusers 0.40.0 and are skipped otherwise; run them with the Regen-2K Python, CUDA hidden (the command is in the file's
docstring; about 2.5 min, ~6.5 GB of memory: on a DGX Spark where a GPU job runs, `systemd-run --user --scope -p
MemoryMax=14G` keeps them from taking its memory). `tests/test_h3turbo_encode.py` covers `bench/h3turbo_encode.py` and
its wrapper, standard library only: the text-only rewrite on every alignment-line form the guide and the page write, the
Upscale's rewrite of a Reference prompt (`ref_text_only_prompt`: the guide's and the page's examples, the other forms of
a definition, one through another subject, a chain of 1000 of them, one too long as text alone, a reference opening a
sentence) and its `--upscale` encode, the prompt as the server takes it, the refusals, `--check`, and the encodes
against a stand-in of the patch's `runtime` package kept in the test file itself (`TurboEncodeSeam` in
`tests/test_server.py` runs it with the server's arguments, the Upscale's too).

The patch has its own 156 CPU tests, run in the patched checkout ([docs/PATCH.md](docs/PATCH.md#apply-test-revert)).
The RTX 5090 patch has 223 (`install/10-sana.sh` with `LFSTUDIO_BACKEND=rtx5090` runs them).

On real hardware: the pipeline needs nearly the whole DGX Spark (~110 GiB free), so run one GPU app at a time. A second
studio next to an installed one needs its own env file (`LFSTUDIO_ENV=<file> bin/lf-studio`) with its own
`LFSTUDIO_PORT`, `LFSTUDIO_STATE_DIR` and `SOL_H3_SPARK_RUNTIME_ROOT`, and `LFSTUDIO_PUBLISH_DIR` empty. Otherwise it
shares the other studio's clip history and files.

## Conventions

- The server stays standard-library only, the page a single file.
- A new setting goes into `bin/env.example` with a comment, and into the server's docstring. It gets no 0.4 name: the
  `OLD_SETTINGS` lists of the stop (`install/old-names.sh`, `bin/lf-cli`) stay as they are.
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
