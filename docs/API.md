# API for agents and scripts

The studio's page talks to its server over HTTP, and programs can use the same API: queue a clip with its input
files, wait for it, and download the MP4. This page documents that API and two clients built on it:

- **`bin/lf-cli`**, a command line. It is one Python file (standard library only, Python 3.9 or later), so it can be
  copied to any machine that reaches the studio.
- **`lf-cli mcp`**, the same file as an MCP server: agent hosts such as Claude Code, Hermes or Codex get tools to
  make clips, wait for them and save them.

H3 takes its prompt in a structured format, and nothing in the studio rewrites a prompt. The
[prompt guide](#prompt-guide) at the end says how to write one; it is also `lf-cli guide` and the MCP tool
`prompt_guide`.

Times on this page were measured on one DGX Spark ([PERFORMANCE.md](PERFORMANCE.md)); a studio on a workstation
with one RTX 5090 (`LFSTUDIO_BACKEND=rtx5090`, [RTX5090.md](RTX5090.md)) has the same API. Every curl, CLI and
MCP-registration example was run against the stub studio (`tests/stub/studio.py`), which makes small test clips
without a GPU.

Contents: [Quick start](#quick-start) · [Access](#access) · [Turning the API off](#turning-the-api-off) ·
[Through spark-switchboard](#through-spark-switchboard) · [Making a clip](#making-a-clip-post-apigenerate) ·
[Waiting](#waiting-get-apijobsid) · [Getting the clip](#getting-the-clip) · [Cancel](#cancel) ·
[Other endpoints](#other-endpoints) · [Errors](#errors) · [CLI](#cli-binlf-cli) ·
[MCP server](#mcp-server-lf-cli-mcp) · [Prompt guide](#prompt-guide)

## Quick start

Each example makes a 5 s clip from the first frame `kitchen.png` and the prompt in `prompt.txt` (an fl2va prompt,
as in the [prompt guide](#prompt-guide)), and saves it as `clip.mp4`. The studio's address is
`http://<spark-host>:8190` from another machine (through spark-switchboard, or a studio started with
`LFSTUDIO_HOST=0.0.0.0`), or `http://127.0.0.1:8190` on the Spark itself.

**The CLI.** Get the file once (`~/.local/bin` must be on your `PATH`), then:

```bash
mkdir -p ~/.local/bin && curl -fsSLo ~/.local/bin/lf-cli \
  https://raw.githubusercontent.com/slogonomo/lf-studio/main/bin/lf-cli && chmod +x ~/.local/bin/lf-cli
export LFSTUDIO_URL=http://<spark-host>:8190
lf-cli generate --first-frame kitchen.png --prompt-file prompt.txt --seconds 5 --out clip.mp4
```

It shows progress on stderr and prints the saved file's absolute path on stdout. On the Spark,
`bin/lf-studio generate ...` runs the same client with the studio's env file, so it needs no `LFSTUDIO_URL`.

**curl with a form upload** (the page's own format). One command submits the clip, waits for it and saves it:

```bash
S=http://<spark-host>:8190
answer=$(curl -sS --fail-with-body -H 'Accept: application/json' -H 'X-Switchboard-Wait: 900' \
    -F task=fl2va -F seconds=5 -F 'prompt=<prompt.txt' -F first_frame=@kitchen.png "$S/api/generate") \
  && id=$(printf '%s' "$answer" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])') \
  && while true; do
       code=$(curl -sS --fail-with-body -H 'Accept: application/json' -o clip.part -w '%{http_code}' \
         "$S/api/jobs/$id/video?wait=50")
       case "$?:$code" in
         0:200) mv clip.part clip.mp4 && echo "saved clip.mp4"; break ;;
         0:202) python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["state"], r["phase"] or "")' clip.part ;;
         *:200) echo "the download broke off; run the loop again"; break ;;
         *:000) echo "no answer from $S"; break ;;
         *) echo "HTTP $code: $(cat clip.part)"; break ;;
       esac
     done \
  || echo "not queued: $answer"
```

**curl with JSON** (files as base64). The same, with the body built in the shell; python3 does the JSON quoting of
the prompt:

```bash
S=http://<spark-host>:8190
{ printf '{"task": "fl2va", "seconds": 5, "prompt": %s, "first_frame": {"name": "kitchen.png", "data": "' \
    "$(python3 -c 'import json, sys; print(json.dumps(open(sys.argv[1], encoding="utf-8").read()))' prompt.txt)"
  base64 -w0 kitchen.png                  # macOS: base64 -i kitchen.png
  printf '"}}'
} > body.json \
  && answer=$(curl -sS --fail-with-body -H 'Accept: application/json' -H 'X-Switchboard-Wait: 900' \
    -H 'Content-Type: application/json' --data-binary @body.json "$S/api/generate") \
  && id=$(printf '%s' "$answer" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])') \
  && while true; do
       code=$(curl -sS --fail-with-body -H 'Accept: application/json' -o clip.part -w '%{http_code}' \
         "$S/api/jobs/$id/video?wait=50")
       case "$?:$code" in
         0:200) mv clip.part clip.mp4 && echo "saved clip.mp4"; break ;;
         0:202) python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["state"], r["phase"] or "")' clip.part ;;
         *:200) echo "the download broke off; run the loop again"; break ;;
         *:000) echo "no answer from $S"; break ;;
         *) echo "HTTP $code: $(cat clip.part)"; break ;;
       esac
     done \
  || echo "not queued: $answer"
```

What the curl commands rely on:

- `--fail-with-body` (curl 7.76 or later) makes a refusal fail the command and still shows the studio's message.
  Plain `-f` would hide the message.
- `Accept: application/json` gets spark-switchboard's answers as JSON too. `X-Switchboard-Wait: 900` lets
  switchboard hold the upload for up to 15 min while it frees the memory and starts the studio, instead of
  refusing it after 2 min ([Through spark-switchboard](#through-spark-switchboard)).
- `-F 'prompt=<prompt.txt'` sends the file's text as the field. `-F prompt=@prompt.txt` would send it as a file,
  which the studio does not read as the prompt.
- Each wait asks for at most 50 s, so a proxy that cuts requests at 60 s does no harm. 202 means "still queued or
  running, ask again"; 200 is the MP4 itself; 409 means the clip failed or was cancelled, with the reason.
- `-o clip.part`, then `mv`: a refusal or a broken download never ends up as `clip.mp4`.

**An agent host over MCP**: register `lf-cli mcp` ([MCP server](#mcp-server-lf-cli-mcp)), for example in Claude
Code:

```bash
claude mcp add --scope user --transport stdio lf-studio -e LFSTUDIO_URL=http://<spark-host>:8190 \
  -e LFSTUDIO_MCP_WAIT_S=600 -e LFSTUDIO_MCP_MAX_WAIT_S=1800 -- python3 /path/to/lf-cli mcp
```

## Access

- **There is no login.** The studio listens on 127.0.0.1 by default. With `LFSTUDIO_HOST=0.0.0.0`, or behind
  spark-switchboard on 8190, **anyone who can reach the port** can:
  - make, cancel, delete and restore clips;
  - load and unload the models;
  - read every prompt, input file and clip, and the job records, which contain absolute paths on the Spark.

  That is fine on a tailnet or LAN that only you use; Tailscale ACLs can limit who reaches the Spark. Never expose
  the port to the internet (a router port forward, Tailscale Funnel).
- **The Origin rule** refuses a POST whose `Origin` header names another site (403 `cross-origin request
  refused`). It stops other web sites from posting to the studio through your browser. Programs send no `Origin`,
  so it does not stop them.
- **The host check** answers only requests for a host name the studio knows: IP addresses, `localhost`, names
  without a dot (Tailscale MagicDNS short names), names ending in `.ts.net`, `.local`, `.lan`, `.home`,
  `.home.arpa` or `.internal`, and the names in `LFSTUDIO_ALLOWED_HOSTS` ([bin/env.example](../bin/env.example)).
  Any other name in `Host` or `X-Forwarded-Host` gets 421 `unknown host name <name>; add it to
  LFSTUDIO_ALLOWED_HOSTS`. This stops DNS rebinding, where a web page uses a name of its own that resolves to your
  studio's address. It does not stop anyone who reaches the port directly. If you reach the studio by a domain of
  your own, add that name to `LFSTUDIO_ALLOWED_HOSTS`.
- **Plain HTTP:** requests and clips cross the network unencrypted unless the network encrypts them. Tailscale
  does, through WireGuard.
- An access token for the API is possible future work; nothing sends or checks one today. `LFSTUDIO_API=0` turns
  the API for programs off ([below](#turning-the-api-off)), but that is not access control either.

## Turning the API off

`LFSTUDIO_API=0` in the studio's settings file, and a restart, leave the studio to its page. `1`, the default, keeps
the API on, as in 0.4. Any other value stops `bin/lf-studio` (also `--check`) and the server at start:
`LFSTUDIO_API=<value>: must be 1 (on, the default) or 0 (off)`. With the API off, these answer 403 `{"error": "the
API for programs is off on this studio (LFSTUDIO_API=0)"}`:

- `POST /api/generate` with a JSON body, and with the page's form when the request has an `X-LFStudio-Client`
  header (any value). `bin/lf-cli` sends its form with one (`X-LFStudio-Client: lf-cli/0.5.2`), so the studio
  refuses its `generate` before anything is queued. The 403 comes after the checks of the body's length (411, 413)
  and type (415), and before the queue's (429).
- `GET` and `HEAD` of `/api/jobs/<id>` and `/api/jobs/<id>/video`, with or without `wait`, for any id.

Everything the page uses works as before: its form submit (Remix too), `/api/state` (with `"api": false`),
`/api/jobs?before=`, `/video/<id>`, `/poster/<id>`, `/still/<id>`, `/api/inputs/<id>`, `/input/...`, `/health` and
the POSTs to cancel, delete, undelete, load and unload. The CLI's `generate`, `wait`, `status <id>` and `fetch`,
and the MCP tools `generate`, `wait`, `status` with an id and `fetch`, fail with a message that names the setting
(CLI exit 2; for MCP a tool error with the same text): `the studio's API for programs is off (LFSTUDIO_API=0 in the
studio's settings): it takes no clips, waits, clip status reads or downloads from lf-cli and agents. ...`. `status`
without an id, which reads `/api/state`, and `cancel` still work. A 0.4 copy of `bin/sol-h3` sends no
`X-LFStudio-Client`: its `generate` still queues the clip, and only its wait gets the 403.

**This is not access control.** The page's own form has taken scripted submits since the first release, and it
still does: anyone who can reach the studio can queue clips with a multipart POST without that header, and read
clips and prompts through the page's endpoints, with the API on or off. To keep others out, bind the studio to an
address that only trusted machines reach ([Access](#access)). An access token is left to a later issue.

## Through spark-switchboard

With [spark-switchboard](https://github.com/slogonomo/spark-switchboard), port 8190 belongs to switchboard. It
starts the studio when something needs it and stops it when the memory is wanted elsewhere or the studio has been
unused for 30 min (`integrations/spark-switchboard/lf-studio.toml`). What a program's request does there:

| Request on port 8190 | Studio stopped | Studio starting | Studio running | Counts as use |
|---|---|---|---|---|
| any POST (`/api/generate`, `/api/cancel`, ...) | starts it, held until it is ready | held | passed on | yes |
| GET of `/health`, `/api/state`, `/api/jobs`, `/api/jobs/<id>`, `/api/jobs/<id>/video` or `/poster/<id>` (the `passive_paths`) | 503 `stopped`; it stays stopped | held while switchboard knows how long the start takes, else 503 | passed on | no |
| any other GET from a program (`/video/<id>`, `/input/...`, `/api/inputs/<id>`, `/still/<id>`) | starts it, held until it is ready | held | passed on | no |
| `HEAD` | 503; it stays stopped | held as above | passed on | no |
| `GET /__switchboard/status` | switchboard's own JSON; never starts anything | the same | the same | no |

"Held" means switchboard keeps the request open, by default for up to `hold_s` (120 s), and then answers it.
The `passive_paths` line needs the integration file of this version (`/api/state`, `/api/jobs`, `/api/jobs/*` and
`/poster/*` are new in it); with an older copy in switchboard's `services/` directory, those GETs start a stopped
studio. [INSTALL.md](INSTALL.md#updating) says how to update the copy.

How a client behaves well:

1. **Send `Accept: application/json`.** switchboard then answers in JSON as well:
   `{"error": {"type": "service_unavailable", "code": "<code>", "message": "<text>"}, "switchboard": {"service",
   "state", "retry_after_s"}}`, status 503 (502 for `upstream_down`), with a `Retry-After` header (the expected
   wait, 5-600 s; 60 while switchboard is paused; none for `stopped`). The codes are `loading`, `waiting`,
   `no_memory`, `failed`, `stopping`, `stopped`, `paused` and `upstream_down`. The studio's own errors are
   `{"error": "<text>"}`: if `error` is an object, the answer came from switchboard. Without the header, a GET gets
   switchboard's answer as plain text.
2. **Send a `User-Agent` that names your program.** switchboard's log and dashboard show it for each start ("wanted:
   GET /api/state (curl/8.5.0 from ...)"). `bin/lf-cli` sends `lf-cli/0.5.2`.
3. **Submit with `X-Switchboard-Wait: <seconds>`** (0-3600). switchboard then holds the POST for that long while
   another app finishes its work and the studio starts, instead of answering 503 after `hold_s`. Your HTTP timeout
   must be longer. A 503 from switchboard means the POST was **not** passed on: no clip was made, and sending it again
   is safe (after `Retry-After`). The CLI holds up to 900 s (`--hold`).
4. **Wait with `GET /api/jobs/<id>?wait=` or `/api/jobs/<id>/video?wait=`** in a loop. switchboard sets no read
   timeout (a 646 s GET went through on a test switchboard); other proxies may cut requests at 60 s, so keep `wait`
   at 50 or less behind one. Waiting does not count as use; a queued or running clip keeps the studio busy through
   switchboard's busy probe (`/api/state`), and switchboard never stops a busy studio.
5. **Download in the same step as the wait.** Once the clip is done and nothing else is queued, switchboard may stop
   the studio as soon as another app wants the memory: on a test switchboard with the stub studio it stopped it
   1.7 s after the clip had finished (with another app waiting, and the studio's quiet period after the last POST,
   300 s in the integration file, already over). `GET /api/jobs/<id>/video?wait=` sends the MP4 on the same
   connection the moment the clip is done, with no new request in between. The CLI and the MCP server download this
   way.
6. **Before fetching an older clip, read `GET /__switchboard/status`**, which never starts anything. If its
   `state` is `stopped`, `stopping` or `failed`, or `paused` is true, a GET of `/video/<id>` would start the studio,
   which takes most of the machine's memory from other apps (about 110 GiB on a DGX Spark) and preloads the
   models. Do that only on purpose. A studio without switchboard answers that path with 404 `{"error": "not
   found"}`: no switchboard, and the request cost nothing.
7. **Always send `Content-Length`.** A chunked upload arrives empty, and the studio answers 411.

**Agents whose own model runs on the same Spark** (an LLM served there through switchboard): the studio needs
about 110 GiB, so the model and the studio can't run at the same time. Every LLM call of the agent while the clip
renders waits for the studio (and fails after the model's own hold), so the agent can't think while it waits. One
tool call must cover the submit, the wait and the download: MCP `generate` with a long `wait_s`, and the host's
tool timeout above it ([Waiting values per client](#waiting-values-per-client)).

## Making a clip: `POST /api/generate`

One endpoint, two encodings, the same checks. The body is either the page's form (`multipart/form-data`) or a
JSON object (`application/json`); anything else gets 415. It needs a `Content-Length` (411 without one, as for a
chunked upload) of at most 200 MB (209,715,200 bytes; 413 above, answered before the body is read, so a client
still sending may see the connection close instead). Base64 adds a third, so JSON files can total about 150 MB.

### Form fields

| Field | Value | Default |
|---|---|---|
| `prompt` | the prompt, at most 32,768 characters | required |
| `task` | `t2va` (text only), `fl2va` (first and/or last frame), `ref2va` (references) | from the files: frames → fl2va, references → ref2va, none → t2va |
| `orientation` | `landscape` (1344x768) or `portrait` (768x1344) | `landscape` |
| `seconds` | 5, 6, 7, 8, 10, 12 or 15 (3 and 4 s, offered up to 0.5.0, get 400; clips made at those lengths stay in the history as they are) | 5 |
| `seed` | an integer, 0 to 2^63 - 1 | empty: a random one |
| `first_frame`, `last_frame` | one image file each (fl2va) | |
| `ref_images`, `ref_videos`, `ref_audio` | files, repeated, in order (ref2va) | |
| `remix_from`, `keep` | reuse files of an earlier clip ([Remix](#remix)) | |

File types go by the file name's extension: images `.png .jpg .jpeg .webp`, videos `.mp4 .mov .m4v .webm .mkv`,
audio `.wav .mp3 .flac .m4a .ogg`. The form is lenient, as the page needs it: unknown fields are ignored, empty file
parts dropped, and with a `task` given, files that task does not use are ignored. Without a `task`, frames together
with references are refused.

### JSON body

```json
{
  "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: ...",
  "task": "fl2va",
  "orientation": "landscape",
  "seconds": 5,
  "seed": 1234,
  "first_frame": {"name": "kitchen.png", "data": "iVBORw0KGgo..."}
}
```

The keys are the form's: `prompt`, `task`, `orientation`, `seconds` (an integer), `seed` (an integer, or `null`
for a random one), `first_frame` and `last_frame` (one file each), `ref_images`, `ref_videos` and `ref_audio`
(lists of files, in order), `remix_from` and `keep` (a list of `"zone:index"` strings). `null` counts as left
out, for every key. A file is either

- `{"name": "kitchen.png", "data": "<base64>"}`: the name's extension gives the type; or
- `"data:image/png;base64,<base64>"`: the type gives the extension, and the file is named after its key
  (`first_frame.png`, `last_frame.jpg`, `image.webp`, `video.mp4`, `audio.wav`).

Base64 may be wrapped across lines (GNU `base64` without `-w0`). The studio does not fetch URLs: a file given as
`"https://..."` or `{"url": ...}` gets 400 `the studio does not fetch URLs; send the file as base64, or use
bin/lf-cli, which downloads URLs itself`. The CLI and the MCP server download URLs on their own machine and upload
the files.

JSON is stricter than the form. Each of these gets 400 before any file is decoded:

- the body is not valid JSON, or not an object;
- an unknown key (`unknown key "frst_frame" (known: ...)`);
- a wrong type (`seconds must be an integer`, `seed must be an integer or null`, `prompt must be a string`,
  `ref_images must be a list`, `first_frame must be one file`, `keep must be a list of "zone:index" strings`);
- a malformed file, base64 that does not decode, or a file of 0 bytes (`first_frame is empty`);
- files the task does not use (`t2va takes no files`, `first_frame is used only by fl2va`,
  `ref_videos is used only by ref2va`), and frames together with references;
- `keep` without `remix_from`.

### Limits and checks

Before the fields are read: 411, 400 (`bad Content-Length`) or 413 for the body's length, 415 for its type, 403 for
JSON and for a form with `X-LFStudio-Client` on a studio with the API off ([Turning the API
off](#turning-the-api-off)), and 429 while `LFSTUDIO_MAX_QUEUED` clips (default 20) are waiting in the queue (the
running clip does not count), with `Retry-After: 60`: `the queue is full (20 clips waiting); try again when one has
finished`.

Then the form and JSON go through the same checks, with the same messages, and a refused request leaves nothing
behind:

1. Task and orientation (`bad task/orientation`), the task's path manifest (`no path manifest for ...`), the
   prompt (`prompt is empty`, `prompt is too long (N characters; at most 32768)`, and `prompt is not valid text
   ...` for a JSON prompt holding half of a UTF-16 surrogate pair, such as `"\ud83c"` from an emoji cut in two),
   the length (`duration must be 5, 6, 7, 8, 10, 12 or 15 seconds`, for 3 or 4 s followed by `(3 and 4 s are no
   longer offered)`) and the seed (`seed must be a nonnegative 63-bit integer`).
2. The files. fl2va needs a first frame, a last frame, or both. ref2va takes up to 9 images, 3 videos and 3 audio
   files, 12 in total (`H3 limits: ...`), and at least one image or video. A wrong type gets `unsupported file type
   .gif (allowed: ...)`. Reference videos are converted on arrival (at most 672x384, 24 fps, their first 15 s, sound
   kept); one shorter than 2 s gets `reference videos must be at least 2 s (H3 limit)`.
3. The pipeline's own check of the request.

A refusal is 400 `{"error": "<type>: <message>"}`, for example `{"error": "ValueError: prompt is empty"}`.

**Keyframes** should have the clip's shape (1344x768 or 768x1344, or the same aspect ratio): the pipeline stretches
or crops a keyframe of another shape to fit. The studio does not check it; the CLI and the MCP server pick the
orientation from the first keyframe when none is given, and warn about a keyframe more than 5% off the clip's
aspect ratio.

### The answer

200 `{"id": "<32 hex characters>", "job": <the API record>}` ([Waiting](#waiting-get-apijobsid)), with `state`
`queued`, the seed that will be used, and for fl2va and ref2va the `labels`.

### Remix

`remix_from` names an earlier clip, and `keep` lists which of its input files to use again, as `zone:index`
(`first_frame:0`, `ref_images:1`; the form joins them with commas, JSON takes a list). The indexes are those of
`GET /api/inputs/<id>` ([Other endpoints](#other-endpoints)). Kept files come first in each zone, then new uploads;
a new frame replaces a kept one. Kept files count when the task is inferred. Errors: `the clip to remix is no longer
in the list; attach the files again`, `the remixed clip has fewer files than the form kept; attach them again`.

## Waiting: `GET /api/jobs/<id>`

```
GET /api/jobs/<id>?wait=50
```

Answers one clip as an **API record** (below). With `wait` (seconds, at most 300; leaving it out means 0) the
answer comes as soon as the clip has finished, or when `wait` runs out with the clip still queued or running
(still 200). A finished clip answers at once.

- 400 `wait must be a number of seconds, 0 or more`.
- 403 `the API for programs is off on this studio (LFSTUDIO_API=0)` on a studio with the API off, for any id
  ([Turning the API off](#turning-the-api-off)).
- 404 `no such clip` for an unknown or deleted clip; `not found` for an id that is not 32 hex characters.
- 503 `{"error": "the studio is shutting down; ask again once it is back", "job": <record>}` if the studio stops
  during the wait. After its restart, a clip that was queued or running reads `failed` with `server restarted`
  (or `cancelled`, if a Cancel had been answered).

### The API record

The clip as stored (the keys `/api/state` shows), plus these keys, worked out for the answer:

| Key | Value |
|---|---|
| `state` | `queued`, `running`, `done`, `failed` or `cancelled`. Use this; `status` is a phrase for people (`loading models + generating`, ...). `done`, `failed` and `cancelled` are final. |
| `phase` | `null` unless running; then `starting`, `loading` (the models, then this clip), `generating`, `retrying` (after a worker error), `saving`, `cancelling` or `running`; an upscale: `unloading`, then `upscaling` ([Upscale](#upscale-experimental-post-apiupscale)) |
| `ahead` | queued only: the jobs that run before it (every queued clip runs before a queued upscale) |
| `upscalable` | done, with the experimental Upscale on: whether it can be upscaled ([Upscale](#upscale-experimental-post-apiupscale)) |
| `elapsed_s` | running only: seconds since it started |
| `urls` | done, while its MP4 exists: `{"video": "/video/<id>", "poster": "/poster/<id>"}`, relative to the studio's address |
| `labels` | fl2va and ref2va: what each input file is called in the prompt (below) |

Some stored keys: `id`, `task`, `orientation`, `seconds`, `seed`, `prompt`, `created`, `started` and `finished`
(Unix times), `wall_s` (seconds from start to end), `error` (failed), `load_eta` (while the models load: the last
measured load plus clip for these settings, in seconds), `includes_model_load`. A clip of the loaded task with another
length or orientation keeps the models loaded and changes the session's shape first: it has `reshaped_from` (the
session's `[task, orientation, seconds]` before) and, once the change is done, `reshape_s` (the seconds it took). It
runs with `phase` `generating` from the start; its first clip at a shape the session hasn't run takes longer (below).
A clip that loads the models has neither key. Paths in the record (`first_frame`, `references`, `output`,
`published`, `case_dirs`) are paths on the Spark, for information only.

A done fl2va clip, abbreviated:

```json
{
  "id": "5cfd57b04762460aa3d232bd20ec42ee", "task": "fl2va", "orientation": "landscape", "seconds": 5,
  "seed": 193195278, "prompt": "For the target video, ...", "status": "done",
  "created": 1791793422.12, "started": 1791793422.14, "finished": 1791793426.47, "wall_s": 4.3,
  "labels": [{"label": "<Picture 1>", "zone": "first_frame", "index": 0, "name": "kitchen.png"}],
  "state": "done", "phase": null,
  "urls": {"video": "/video/5cfd57b04762460aa3d232bd20ec42ee", "poster": "/poster/5cfd57b04762460aa3d232bd20ec42ee"}
}
```

While it runs: `"status": "loading models + generating", "state": "running", "phase": "loading", "elapsed_s":
42.1, "load_eta": 113`. Queued: `"state": "queued", "phase": null, "ahead": 1`.

### Labels

`labels` lists `{"label", "zone", "index", "name"}` for each input file, with `sound_of` for a video's sound track,
in the numbering the pipeline uses: `zone` and `index` as in `/api/inputs/<id>`, `name` the stored file name.

- fl2va: the first frame is `<Picture 1>`, the last frame the next picture (`<Picture 1>` when it is alone).
- ref2va, in the order of the references (images, then videos, then audio; kept files first in each):
  each image is the next `<Picture N>`; each video the next `<Video N>`, and **a video with a sound track also takes
  the next `<Audio N>` just before it** (`"sound_of": "<Video N>"`); each audio file is the next `<Audio N>`.

An image, a video with sound and a WAV give `<Picture 1>`, `<Audio 1>` (the video's sound), `<Video 1>` and
`<Audio 2>`. Write the prompt with those numbers.

### How long a clip takes

One clip renders at a time. Measured on one DGX Spark ([PERFORMANCE.md](PERFORMANCE.md)):

| | Time |
|---|---|
| t2va with the models loaded (the same task as the clip before, at an orientation and length the session has run) | 5 s: 54-66 s · 10 s: 123 s · 15 s: 191 s |
| ref2va with a 5 s reference video, 5 s | 133 s per clip |
| The first clip at an orientation or length the session hasn't run, same task (the models stay loaded; Stage 2 compiles for the new shape, from the compile cache when that shape ran before) | a clip with the models loaded plus 0.2-10 s (5 → 10 s: 7 s, 10 → 15 s: 10 s in the staging check of 0.4, compile cache warm; fl2va 16:9 → 9:16: 0.2 s in an earlier run; one run each, [PERFORMANCE.md](PERFORMANCE.md#changing-length-or-format)). With `SOL_H3_DYNAMIC_TEXT_LENGTH=0` Stage 1 compiles for it too: 5 → 10 s: 14 s, 5 → 15 s: 27 s when it compiled from scratch |
| t2va, the first clip after a change of task (the models load first; also after any change with `LFSTUDIO_RESHAPE=0`) | 5 s: 104-113 s in the studio, 99 s with `infer.py` · 15 s: 238 s with `infer.py` |
| The first clip of fl2va (first and last frame) / ref2va (an image and a video), 5 s | 115 s / 178 s with `infer.py` |
| Cancel of a running clip, until the pipeline has closed | about 40 s; the next clip then loads the models again |

Add the clips queued before it. The first run of a new setting after an install also compiles its kernels once
(t2va 16:9 15 s: 357 s). The MP4s in one studio's publish folder were 3.2 MB on average and at most 7.7 MB (100
clips).

## Getting the clip

- **`GET /api/jobs/<id>/video?wait=S`** waits like `/api/jobs/<id>?wait=S`, then:
  - 200 with the MP4, the same as `/video/<id>` (below), as soon as the clip is done;
  - 202 with the API record while the clip is still queued or running when `wait` runs out: ask again;
  - 409 `{"error": "the clip failed: <error>", "job": <record>}`, or `the clip was cancelled`;
  - 404 `no such clip`, or `{"error": "no such video", "job": <record>}` for a done clip whose MP4 is gone;
  - 400, 403 and 503 as for `/api/jobs/<id>`.

  Behind switchboard this path never starts the studio, and it sends the MP4 the moment the clip is done.
- **`GET /video/<id>`**: the MP4 (`video/mp4`) with `Content-Disposition: inline;
  filename="lf-studio_<task>_<seconds>s_seed<seed>_<id6>.mp4"`. `Range` requests get 206. 404 `no such video`. Behind
  switchboard it starts a stopped studio.
- **`GET /poster/<id>`**: a JPEG of the frame at about 1 s, 640 pixels wide. 404 `no poster`. Behind switchboard it
  never starts the studio (with this version's integration file).

## Cancel

```bash
curl -sS --fail-with-body -H 'Accept: application/json' -H 'Content-Type: application/json' \
  -d "{\"id\": \"$id\"}" "$S/api/cancel"
```

- 200 `{"ok": true, "job": <stored record>}`. A queued clip is `cancelled` at once. A running one gets
  `"cancel_requested": true` and stays running (`phase` `cancelling`) until its pipeline has closed, about 40 s on
  the Spark, then reads `cancelled`. The next clip loads the models again. Cancelling it again meanwhile is 200 as
  well.
- 409 `the clip has already finished`, `the clip has already failed`, `the clip is already cancelled`, or `the clip
  has just finished and is being saved`.
- 404 `no such clip`; 400 `expected JSON {"id": ...}`.

A cancelled clip keeps its input files for Remix. Stopping a wait (Ctrl-C, a closed connection, a cancelled MCP
call) never cancels the clip.

## Upscale (experimental): `POST /api/upscale`

Only on a DGX Spark studio with `LFSTUDIO_UPSCALE=1` (off by default; `bin/env.example`). It upscales a finished
clip, from its MP4, to 1080p, 2K (2560×1440) or 4K with NVIDIA's Open-H3-Regenerate-2K, run as a job of its own
(`upscale/regen2k_job.py`, in an environment of its own). For now it takes **5 s text clips (t2va), 16:9 or 9:16**
only: NVIDIA's pipeline is fixed at 121 frames, and fl2va and ref2va clips need a text-only encode of their prompt
first. NVIDIA's code is written for 16:9 (1344×768): the job takes a 9:16 clip (768×1344) through it with the shapes
turned, without changing NVIDIA's files, and the result is upright (1080×1920, 1440×2560 or 2160×3840). 9:16 has run
on the GPU at 1080p (as fast as 16:9) and 2K, not yet at 4K, and NVIDIA trained its refiner on 16:9 video. The other
lengths the studio offers (6 to 15 s), fl2va and ref2va are follow-ups.

```bash
curl -sS --fail-with-body -H 'Accept: application/json' -H 'Content-Type: application/json' \
  -d "{\"id\": \"$id\", \"resolution\": \"2k\"}" "$S/api/upscale"
```

| Key | Value |
|---|---|
| `id` | the finished clip (required) |
| `resolution` | `1080p`, `2k` (default) or `4k` (the slowest: ~66 min in its one run so far, a 16:9 clip; not yet run with a 9:16 one): 1920×1080, 2560×1440 or 3840×2160 for a 16:9 clip, 1080×1920, 1440×2560 or 2160×3840 for a 9:16 one (`orientation` in the record: the clip's) |
| `strength` | how much it may redraw: `faithful` (start noise 0.5, 2 steps), `balanced` (0.7, 2 steps) or `full` (NVIDIA's release setting, 3 steps from 0.909: sharpest, but it redraws faces). Default `LFSTUDIO_UPSCALE_STRENGTH` (`balanced`). Provisional, until measured |
| `audio` | what the picture is refined against: `clip` (the clip's own Stage-1 audio) or `silence` (NVIDIA's). Default `LFSTUDIO_UPSCALE_AUDIO` (`clip`). The result gets the clip's sound either way, copied |
| `seed` | the refine's seed; default (or `null`) the clip's |

- 200 `{"id", "job": <API record>}`: queued as a job of its own (`"kind": "upscale"`, `"parent": "<clip id>"`; `task`,
  `orientation`, `seconds`, `prompt` and `seed` are the clip's; `load` is how its job loads the weights, `fast` or
  `standard`, from `LFSTUDIO_UPSCALE_LOAD`). It waits while any clip is queued, even one queued after it, since it
  closes the studio's models. It runs as `status` `freeing memory` (`phase` `unloading`: the models
  close) and then `upscaling` (`phase` `upscaling`, with `upscale_phase` from the job: `encoding`, `loading`,
  `refine 1/2`, ..., `decoding`, `saving`, `adding the sound`). The next clip loads the models again; with
  `LFSTUDIO_PRELOAD=1` (the default) the studio warms up the last clip's settings again as soon as the upscale is
  done and nothing is queued.
- 400 for a bad field (`resolution must be one of 1080p, 2k, 4k`, `unknown key "steps" (...)`, ...), `an upscale can't
  be upscaled again`, `Upscale takes text-to-video (t2va) clips only, for now` or `Upscale takes 5 s clips only, for
  now (NVIDIA's pipeline is fixed at 121 frames)`. Any key not in the table is unknown, `source` too: test builds of
  the Upscale took `"source": "draft"` (the clip's Stage-1 latent), which redrew the video too much and is gone, so a
  request with `source` (even `"final"` or `null`) gets 400 `unknown key "source" (known: id, resolution, strength,
  audio, seed)` and nothing is queued.
- 404 `no such clip`; with the Upscale off, `the experimental Upscale is off on this studio (LFSTUDIO_UPSCALE=1 turns
  it on)`.
- 409 for a clip that is not done (`the clip is still queued`, `the clip failed`, ...), is being deleted or restored,
  or whose files are gone (its session directory; for `audio` `clip` its Stage-1 audio). 429 when the queue is full.

From a program: `lf-cli upscale` ([CLI](#cli-binlf-cli)) or the MCP tool `upscale` ([Tools](#tools)). Behind
spark-switchboard the POST starts a stopped studio, as a submit does; send `X-Switchboard-Wait` with it.

Cancel, Delete, Undo, `/api/jobs/<id>[?wait=]`, `/video/<id>` and `/poster/<id>` work on an upscale as on a clip;
Cancel stops its job within about 20 s (SIGTERM, then SIGKILL if the GPU call under way has not returned). It
downloads as `lf-studio_<task>_5s_seed<seed>_<clip id6>_<resolution>_<upscale id6>.mp4`, and its
`LFSTUDIO_PUBLISH_DIR` copy is named as the clip's with `_<resolution>` after it (`-2`, ... for a second one). Its
files are in a directory of their own, so Delete moves only them; a clip can't be deleted while an upscale of it is
queued or running (409). It fails, without a retry, below `LFSTUDIO_UPSCALE_MIN_FREE_GIB` of free memory once the
models are closed (default 110; 5 GiB more at 4K), or when the memory guard kills it (`LFSTUDIO_UPSCALE_GUARD_GIB`,
5 GiB for 3 s); `error` says why (the job's own error line when it has one). A finished upscale also has `phases_s`,
`peak_gib` and `low_mem_gib` from its job.

With the Upscale on, `/api/state` has `"upscale": true`, `"upscale_defaults"`, what a request without those keys
gets (`{"resolution": "2k", "strength": <LFSTUDIO_UPSCALE_STRENGTH>, "audio": <LFSTUDIO_UPSCALE_AUDIO>}`; `null`
with the Upscale off), `"upscale_load"`, how its jobs load their weights (`fast` or
`standard`, `LFSTUDIO_UPSCALE_LOAD`; the page's popover shows its times with it; `null` with the Upscale off), and
every finished job in `/api/state`, `/api/jobs` and the API record has `upscalable`: `{"ok": true, "clip_audio": <has
its Stage-1 audio>}` or `{"ok": false, "why": "<the 400 or 409 text above>"}`.
This endpoint is the page's, like Cancel: it works with `LFSTUDIO_API=0`.

## Other endpoints

| Endpoint | Answer |
|---|---|
| `GET /api/state` | What the page shows: `{"now", "tasks_ready", "trash_days", "warmable", "reshape", "backend", "api", "upscale", "upscale_defaults", "upscale_load", "session", "jobs", "done", "more", "deleted"}`. `upscale` is true when the experimental Upscale is on (`LFSTUDIO_UPSCALE=1`); then finished jobs have `upscalable`, `upscale_defaults` holds the choices the page's Upscale starts with and `upscale_load` its load, `fast` or `standard` (else both `null`). `api` is false when the API for programs is off (`LFSTUDIO_API=0`, [Turning the API off](#turning-the-api-off)). `backend` is the machine, `spark` (the default) or `rtx5090` (`LFSTUDIO_BACKEND`); the page names the machine and picks its clip-time estimates by it. `reshape` is true when another length or orientation of the loaded task keeps the models loaded (`LFSTUDIO_RESHAPE=1` and a patch that can). `session.state` reads `reshaping` while a load request changes the loaded session's shape (the page: "changing to ..."). `jobs` holds every queued or running clip and the newest 24 finished ones, newest first, as stored (without the API record's keys). switchboard's busy probe reads it. |
| `GET /api/jobs?before=<created>&limit=<n>` | Older clips, newest first: `{"jobs": [...], "more": true/false}`. `before` is a `created` time (default: now), `limit` 1-100 (default 24). Records as stored. |
| `GET /api/inputs/<id>` | A clip's input files, for Remix: `{"task", "inputs": {"<zone>": [{"name", "index", "url"}]}, "missing": [...]}` |
| `GET /input/<id>/<zone>/<index>` | One input file (reference videos as converted). 404 `no such input`. |
| `GET /still/<id>` | The picture a queued or running clip's card shows (a keyframe, a reference image, or a reference video's first frame). 404 `no picture` for t2va. |
| `GET /health` | `{"ok": true}` |

Used by the page; stable, but not designed for agents. Each takes a JSON body of at most 64 KiB (413 above):

| Endpoint | Body | Does |
|---|---|---|
| `POST /api/delete` | `{"id"}` | moves a finished clip and its files to the trash for `LFSTUDIO_TRASH_DAYS` (default 7); 409 while it is queued or running, or while an upscale of it is |
| `POST /api/undelete` | `{"id"}` | brings it back |
| `POST /api/load` | `{"task", "orientation", "seconds"}` | loads the models for those settings ahead of a clip ("Load selected now"). With that task loaded, it only changes the session's length and orientation, without a warm-up clip (`reshape` true). 400 for a length not offered, with a submit's message; 409 while a clip is queued or running |
| `POST /api/unload` | none | frees the models; 409 while a clip runs |

## Errors

The studio answers errors as `{"error": "<text>"}` (some add `"job"`); spark-switchboard as
`{"error": {"type", "code", "message"}, "switchboard": {...}}` ([Through spark-switchboard](#through-spark-switchboard)).

| Status | From | Meaning |
|---|---|---|
| 200 | studio | done; for `/api/jobs/<id>?wait=`, also a clip still running when `wait` ran out (read `state`) |
| 202 | studio | `/api/jobs/<id>/video`: the clip is still queued or running; ask again |
| 206 | studio | part of a file (`Range`) |
| 400 | studio | a refused request: bad fields, files or JSON, a bad `wait`, a bad `Content-Length` |
| 403 | studio | `cross-origin request refused`: a POST from a page of another site |
| 403 | studio | `the API for programs is off on this studio (LFSTUDIO_API=0)`: a JSON submit, a form submit with `X-LFStudio-Client` (`bin/lf-cli`), or `/api/jobs/<id>[/video]` on a studio with the API off ([Turning the API off](#turning-the-api-off)) |
| 404 | studio | no such clip, video, poster, input or path (`not found`); on `/__switchboard/status`: no switchboard |
| 409 | studio | the clip failed or was cancelled (`/api/jobs/<id>/video`); Cancel of a finished clip; Delete of a running one |
| 411 | studio | `Content-Length required (chunked uploads are not supported)` |
| 413 | studio | over 200 MB (`/api/generate`) or 64 KiB (the other POSTs) |
| 415 | studio | `/api/generate` with a body that is neither form nor JSON |
| 416 | studio | a `Range` outside the file |
| 421 | studio | `unknown host name <name>; add it to LFSTUDIO_ALLOWED_HOSTS` ([Access](#access)) |
| 429 | studio | the queue is full; `Retry-After: 60` |
| 502 | switchboard | `upstream_down`: the studio stopped answering |
| 503 | switchboard | not running: `loading`, `waiting`, `no_memory`, `failed`, `stopping`, `stopped`, `paused`; `Retry-After`. A POST that got it was not passed on. |
| 503 | studio | `the studio is shutting down ...` with the clip's record (`/api/jobs/<id>`) |

## CLI: `bin/lf-cli`

`bin/lf-cli` is one Python file: standard library only, Python 3.9 or later, and it reads no other file of this
repo. On another machine, get it as in the [Quick start](#quick-start) (or copy it from a checkout). On the Spark,
`bin/lf-studio <command> ...` runs it with the settings of the studio's env file. Up to 0.4 it was `bin/sol-h3`: until
0.6 that name runs `bin/lf-cli` (with a note on stderr), and a 0.4 copy on another machine works with this studio. A
new download from the 0.4 address gets only that shim, which says where to get `bin/lf-cli`.

```
lf-cli generate (--prompt TEXT | --prompt-file FILE|-) [--task t2va|fl2va|ref2va]
       [--orientation landscape|portrait] [--seconds 5|6|7|8|10|12|15] [--seed N]
       [--first-frame FILE|URL] [--last-frame FILE|URL] [--ref FILE|URL]...
       [--wait] [--out PATH] [--poster] [--timeout S] [--hold S]
lf-cli wait ID [--out PATH] [--poster] [--timeout S]
lf-cli status [ID]
lf-cli cancel ID
lf-cli fetch ID [--out PATH] [--poster] [--wake]
lf-cli upscale ID [--resolution 1080p|2k|4k] [--strength faithful|balanced|full]
       [--audio clip|silence] [--seed N] [--wait] [--out PATH] [--poster] [--timeout S] [--hold S]
lf-cli guide                  # the prompt guide
lf-cli mcp                    # the MCP server on stdin/stdout
common options, before or after the command: --url URL  --json  --quiet
```

**The studio** is `--url`, else `$LFSTUDIO_URL`, else `http://$LFSTUDIO_HOST:$LFSTUDIO_PORT` (`0.0.0.0` or empty counts
as `127.0.0.1`; a missing port is 8190), else `http://127.0.0.1:8190`. Behind switchboard, use its port (8190).

**`generate`** checks everything it can before it uploads: the task against the files (without `--task`: frames →
fl2va, `--ref` → ref2va, none → t2va), types, H3's counts, and a total of 199 MB. `--ref` files go to images, videos
or audio by their extension, numbered in the order given per kind. A file may be a path or an http(s) URL; URLs are
downloaded by the CLI (so tailnet addresses work), at most 120 s each, typed by the URL path's extension or else
by `Content-Type`. Without `--orientation`, the first keyframe's shape picks it (stderr says so), and a keyframe
more than 5% off the clip's aspect ratio gets a warning. The shape is the one shown: an EXIF rotation in a JPEG,
PNG or WebP counts, as the pipeline applies it. The upload sends `X-Switchboard-Wait: <--hold>` (default
900 s).

**Line ends.** `--prompt-file FILE` reads the file with universal newlines, so CRLF becomes LF, while the page's form
sends a multi-line prompt with CRLF and the studio keeps it: the same prompt can tokenize differently through the CLI
and through the page. To render a page clip again, pass its prompt as the page sent it (its API record's `prompt`),
for example with `--prompt` (`--prompt-file -` also keeps CRLF, on Linux).

- Without `--wait`, `--out` or `--poster`, it prints the clip's id and exits.
- With any of them it waits, then saves the MP4: into the current directory under the clip's own name
  (`lf-studio_<task>_<s>s_seed<seed>_<id6>.mp4`), into a directory given as `--out` (an existing one, or a path ending
  in `/`), or as the `--out` file, which it replaces. `--poster` also saves the poster as `<name>.jpg`. It checks
  that it can write there before it submits (exit 2 if not).
- `--timeout S` stops waiting after S seconds (exit 4, the id on stdout); the clip keeps running. Ctrl-C while
  waiting also leaves the clip running (exit 130) and says how to pick it up: `lf-cli wait <id> --out ...`.

**`wait ID`** waits for a clip and saves it as above. **`fetch ID`** saves a finished clip (also one made in the
page) without waiting; a clip still running is exit 4. **`status`** prints one line per clip: the queued and
running ones, oldest first, then the newest 5 finished ones; `status ID` one clip. **`cancel ID`** cancels it.

**`upscale ID`** (experimental, on a studio with `LFSTUDIO_UPSCALE=1`) queues an upscale of a finished 5 s t2va clip,
16:9 or 9:16 ([Upscale](#upscale-experimental-post-apiupscale)), and works as `generate` from there: without
`--wait`, `--out` or `--poster` it prints the upscale's own id; with them it waits and saves the MP4, by default as
`lf-studio_<task>_5s_seed<seed>_<clip id6>_<resolution>_<upscale id6>.mp4`. It sends only the choices given, so the
studio's defaults hold for the rest (2K, `LFSTUDIO_UPSCALE_STRENGTH`, `LFSTUDIO_UPSCALE_AUDIO`, the
clip's seed), and holds at switchboard as `generate` does (`--hold`). An upscale takes about 10 min to 1.6 h in all (4K
over an hour), with a ~22 s load (measured in the studio's job; ~8 min more on a studio with
`LFSTUDIO_UPSCALE_LOAD=standard`, the fallback), so `--timeout` and a later `wait` are the usual way.
`status` shows it as `2k upscale of <clip id8>, balanced, ...`, `cancel` stops it within about 20 s, and the studio's
refusals (`Upscale takes 5 s clips only, for now ...`, the Upscale off) are exit 2.

**Waiting** is a loop of `GET /api/jobs/<id>/video?wait=15`, so the MP4 comes on the connection that saw the clip
finish. It shows one progress line per change on stderr (`queued 5cfd57b0 (fl2va, landscape, 5 s, seed 1234),
next in line`, `loading models, then generating (about 1:53 in all)`, `generating`, `done in 1:05`). Connection
errors and switchboard's `loading`, `waiting` and `upstream_down` answers are retried every 5 s for up to 120 s.
The MP4 is written to a `.part` file, checked against `Content-Length` and then renamed.

**With spark-switchboard**, `status`, `wait`, `cancel` and `fetch` read `/__switchboard/status` first and never
start a stopped studio. `fetch` then refuses (exit 3, "fetching would start it (it needs most of the machine's
memory). Add --wake."); `fetch --wake` starts it, holding up to 900 s. If switchboard stops the studio while `wait`
or `generate --wait` waits, the clip most likely finished first; the CLI says so and how to get it (`lf-cli fetch
<id> --wake`).

**With the API off** on the studio (`LFSTUDIO_API=0`), `generate` (before anything is queued), `wait`, `status <id>`
and `fetch` exit 2 with a message that names the setting ([Turning the API off](#turning-the-api-off)).

**Output:** stdout carries the result only: the saved MP4's absolute path, the id (generate without waiting), or
the status lines. Everything else goes to stderr; `--quiet` drops the progress lines. With `--json`, stdout is one
JSON object: the API record (plus `path`, `bytes` and `poster_path` when files were saved), `{"jobs": [...]}` for
`status` without an id, or on failure `{"error", "http_status", "id"}` (plus `job` when known).

| Exit code | Meaning |
|---|---|
| 0 | done and saved; queued (no wait); or status, cancel, guide succeeded |
| 1 | the clip failed or was cancelled (the studio's error on stderr) |
| 2 | a usage error, a local file check, or the studio refused the request (400, 403, 404, 409, 411, 413, 415, 421) |
| 3 | the studio can't be reached or isn't available (connection error, switchboard 502/503, studio 503 or 5xx, 429), or switchboard has it stopped and `--wake` was not given |
| 4 | `--timeout` ran out (the id is on stdout), or `fetch` of a clip still running |
| 130 | interrupted; the clip keeps running |

More examples:

```bash
lf-cli generate --prompt-file t2va.txt --seconds 8 --seed 7 --wait            # t2va, saved here
lf-cli generate --ref face.png --ref voice.wav --prompt-file ref.txt --out clips/   # ref2va, into clips/
id=$(lf-cli generate --prompt-file t2va.txt)                                   # queue only
lf-cli status "$id"
lf-cli wait "$id" --out clip.mp4 --poster --timeout 600
lf-cli --json fetch "$id" --out clips/
up=$(lf-cli upscale "$id" --resolution 2k --strength faithful)                # experimental: queue an upscale
lf-cli wait "$up" --out clips/                                                 # ~30 min later: ..._2k_<id6>.mp4
```

## MCP server: `lf-cli mcp`

`lf-cli mcp` speaks MCP over stdin and stdout (newline-delimited JSON-RPC), both the `initialize` handshake
(protocol versions 2024-11-05 to 2025-11-25) and the 2026-07-28 revision. It runs on the agent's machine: it reads
the agent's files there, uploads them, and saves the MP4 there. The studio itself has no MCP endpoint.

### Settings

From the MCP process's environment (the host's server config):

| Variable | Default | Meaning |
|---|---|---|
| `LFSTUDIO_URL` | as the CLI | the studio, `http://<spark-host>:8190` |
| `LFSTUDIO_OUT_DIR` | `./lf-clips` (from the process's working directory; Claude Code starts it in the project directory) | where MP4s go when `out` is not given |
| `LFSTUDIO_MCP_WAIT_S` | 50 | how long `generate` and `wait` wait when the call gives no `wait_s` |
| `LFSTUDIO_MCP_MAX_WAIT_S` | 1800 | the cap on any `wait_s` |
| `LFSTUDIO_MCP_OUT_ANYWHERE` | unset | `1` lets `out` point outside `LFSTUDIO_OUT_DIR` and the working directory |

Until 0.6 each also works by its 0.4 name (`SOLH3_URL` and so on; the new name wins), with one warning on stderr, and
with `LFSTUDIO_OUT_DIR` unset, `./sol-h3-clips` is the default while it exists and `./lf-clips` is missing or empty. A
host registered with `sol-h3 mcp` keeps working through `bin/sol-h3` until 0.6; the server now calls itself
`lf-studio` (`serverInfo`).

### Tools

| Tool | Arguments | Does |
|---|---|---|
| `generate` | `prompt` (required), `task`, `orientation`, `seconds`, `seed`, `first_frame`, `last_frame`, `references` (paths on the agent's machine or http(s) URLs), `wait_s`, `out`, `poster` (default true), `again` | queues a clip, waits up to `wait_s` and, when it is done, saves the MP4 |
| `wait` | `id` (required), `wait_s`, `out`, `poster` | waits for a clip and saves it |
| `status` | `id` | one clip's record; without an id, the clips this session submitted, then the queue and the newest clips. Never waits, downloads or starts a stopped studio. |
| `cancel` | `id` (required) | cancels a queued or running clip |
| `fetch` | `id` (required), `out`, `poster`, `wake` (default false) | saves a finished clip (also one made in the page); refuses on a studio switchboard has stopped unless `wake` is true |
| `upscale` | `id` (required), `resolution`, `strength`, `audio`, `seed`, `wait_s`, `out`, `poster` | experimental, on a studio with `LFSTUDIO_UPSCALE=1`: queues an upscale of a finished 5 s t2va clip, 16:9 or 9:16 ([Upscale](#upscale-experimental-post-apiupscale)), waits up to `wait_s` and saves the MP4 when it is done; else `wait` with the id it returns. Each call queues another upscale |
| `prompt_guide` | none | the [prompt guide](#prompt-guide) |

**Results.** A done clip gives a text block (`Clip <id> is done: saved <absolute path> (3.2 MB, 5 s fl2va
landscape, seed 1234).`, the labels, and the record as JSON), a `resource_link` to `<studio>/video/<id>`, and the
poster as an image (unless `poster: false`). The `resource_link` goes only to clients on protocol 2025-06-18 or
later: the official Python SDK 1.8.1 and 1.9.4 (which agree on 2024-11-05 and 2025-03-26) reject a result holding
one. The URL is in the JSON as well. `structuredContent` is the record plus `path`, `url`, `poster_url`, `bytes`
and `poster_path`. The poster is also saved next to the MP4 as `.jpg`. A clip still running is not an error: `Clip
<id> is generating, 1:10 since it started. Call wait with id "<id>" to keep waiting.` A failed clip, a refusal and
an unreachable studio are `isError` results with the message. A cancelled clip is not an error.

**How it behaves:**

- **The submit** holds at switchboard for what is left of the call's budget after a margin for the upload (15 s,
  plus 1 s per 2 MB, plus 15 s per reference video), so `generate` still answers within `wait_s`. The upload runs
  on its own: a call that is cancelled or times out still submits the clip, and `status` without an id shows it.
- **An identical `generate` to the same MCP server process within 30 minutes** (the same arguments and the same
  file contents) returns the earlier clip instead of making another: "Already submitted as <id> ...". Change the
  seed, or pass `again: true`, for another take. A clip that failed, was cancelled or was deleted is made again; a
  studio that is starting, stopped or unreachable keeps the earlier clip. The server remembers these calls in
  memory only: after a host restart or in a new host session, the same call makes a new clip, so look for the
  earlier one first with `status` (no id: the newest clips) or its id.
- **Cancelling a tool call** (Esc in the host, a host timeout) never cancels or removes a clip; only the `cancel`
  tool does.
- **`out`** must be an existing directory or a name ending in `.mp4`, inside `LFSTUDIO_OUT_DIR` or the working
  directory (unless `LFSTUDIO_MCP_OUT_ANYWHERE=1`). An existing file is never replaced: the clip is saved as
  `<name>-1.mp4`, `-2`, ... instead.
- **Relative paths** in `out`, `first_frame`, `last_frame` and `references` start from the MCP server's working
  directory, not from `LFSTUDIO_OUT_DIR`. That is the project directory in Claude Code; other hosts may start the
  server anywhere, so agents there should pass absolute paths. A missing file's error names the absolute path it
  looked for.
- **Before it submits**, `generate` checks that it can write where the clip goes (`out`, else `LFSTUDIO_OUT_DIR`, or
  the directory above it that exists). A host that starts the server in a directory it can't write to (`/`, say)
  gets a refusal naming `LFSTUDIO_OUT_DIR`, not a rendered clip that can't be saved. If saving fails all the same, the
  error says the clip is done on the studio and that `fetch` with its id and a writable `out` saves it.
- **Progress:** when the host asks for it, `notifications/progress` every 10 s while a call waits, with the phase as
  the message.
- **spark-switchboard:** as the CLI; a stopped studio makes `wait` and `fetch` answer that `fetch` with `wake: true`
  would start it.
- **`upscale`** answers `Queued upscale <id> of clip <clip id> (2k, balanced).`, then as `generate`.
  An upscale takes about 10 min to 1.6 h, far longer than any `wait_s`: the usual result says it is still running, and a
  later `wait` saves it.

### Waiting values per client

A tool call may run only as long as the host lets it. Set `LFSTUDIO_MCP_WAIT_S` below that limit; `wait_s` in a call
can ask for more, up to `LFSTUDIO_MCP_MAX_WAIT_S`. The limits as the hosts document them (2026-10):

| Host | Limit per tool call | `LFSTUDIO_MCP_WAIT_S` |
|---|---|---|
| Claude Code (stdio) | none; a call is stopped after 30 min without a result or progress, and the server sends progress every 10 s. A call still running after 2 min moves to the background, and its result arrives later. In `claude -p` (2.1.296) a 129 s `generate` stayed in the foreground and returned the clip as usual (measured 2026-10-12). | 600 |
| Hermes | `timeout`, 300 s by default | 280, or 1700 with `timeout: 1800` when Hermes's model runs on the same Spark |
| Codex CLI | `tool_timeout_sec`, 60 s by default | 50, or raise `tool_timeout_sec` |
| Other hosts on the TypeScript SDK 1.x | 60 s | 50 (the default) |

### Registration

**Claude Code** (the name comes before the `-e` options):

```bash
claude mcp add --scope user --transport stdio lf-studio -e LFSTUDIO_URL=http://<spark-host>:8190 \
  -e LFSTUDIO_MCP_WAIT_S=600 -e LFSTUDIO_MCP_MAX_WAIT_S=1800 -- python3 /path/to/lf-cli mcp
claude mcp list          # lf-studio: python3 /path/to/lf-cli mcp - ✔ Connected
```

**Hermes** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  lf_studio:
    command: "python3"
    args: ["/path/to/lf-cli", "mcp"]
    env: {LFSTUDIO_URL: "http://<spark-host>:8190", LFSTUDIO_OUT_DIR: "/path/for/clips", LFSTUDIO_MCP_WAIT_S: "280"}
    timeout: 300
```

If Hermes's own model runs on the same Spark, use `LFSTUDIO_MCP_WAIT_S: "1700"`, `LFSTUDIO_MCP_MAX_WAIT_S: "1800"` and
`timeout: 1800`, so that one call covers the submit, the wait and the download
([Through spark-switchboard](#through-spark-switchboard)). Set `LFSTUDIO_OUT_DIR`: Hermes may start the server
outside your project directory, and `out` must lie inside `LFSTUDIO_OUT_DIR` or the server's working directory. A
relative `out` or input path still starts from that working directory, so tell the agent to pass absolute paths.

**Codex:**

```bash
codex mcp add lf-studio --env LFSTUDIO_URL=http://<spark-host>:8190 -- python3 /path/to/lf-cli mcp
```

This keeps `LFSTUDIO_MCP_WAIT_S` at 50, under Codex's 60 s. To wait longer per call, add
`--env LFSTUDIO_MCP_WAIT_S=580` to the command, and `tool_timeout_sec = 600` to the `[mcp_servers.lf-studio]` table it
writes to `~/.codex/config.toml`.

**Any other host:** the command `python3 /path/to/lf-cli mcp`, with `LFSTUDIO_URL` in its environment, and
`LFSTUDIO_OUT_DIR` too unless the host starts the server in a directory where clips should go.

## Prompt guide

The same text is `lf-cli guide` and the MCP tool `prompt_guide`; a test keeps the copies identical.

<!-- prompt-guide:start -->

LF Studio gives your prompt to MiniMax-H3 unchanged; nothing rewrites it. H3 was trained on a structured
format, and MiniMax calls that format critical to the quality of the result, so write it yourself as below.
Plain prose is accepted. The prompt drives the draft picture and **all of the sound**. The refiner that makes the
final 1344x768 video uses a fixed prompt of its own.

**1. Task, length, format**

- `t2va`: text only. `fl2va`: a first frame, a last frame, or both (images).
  - A keyframe should have the clip's shape (1344x768 or 768x1344, or the same aspect ratio): one of another
    shape is stretched or cropped to fit. Without an orientation, `lf-cli` takes it from the first keyframe's
    shape.
- `ref2va`:
  - up to 9 images, 3 videos (2-15 s each) and 3 audio files, 12 files in total;
  - at least one image or video.
- `seconds`: 5, 6, 7, 8, 10, 12 or 15.
  - H3 was trained on 4-15 s; NVIDIA validated 5 s.
  - Every time in the prompt must fall inside the clip.
- `orientation`: landscape (1344x768) or portrait (768x1344; portrait is not quality-validated).

**2. t2va and fl2va: three fields**

Write these three fields in this order, each as `name: text`, with a blank line between them:

- `integrated_multimodal_description:` everything seen and heard, along the timeline.
  - Start with `[Shot 1]`, the style (live-action, cinematic, 2D-animated, 3D CG, claymation, ...) and the
    opening composition. Shot 1 has no timestamp.
  - Each later shot is `[Shot N] At MM:SS.mmm, the camera cuts to ...`, with cut times increasing and inside the
    clip.
  - A moving camera (push in, pan right, truck left, tracking shot, ...) can take an amplitude and a speed in
    plain words when they matter: "the camera pushes in with small amplitude at slow speed". Leave them out for a
    medium amplitude and a normal speed. A static shot takes neither: "the camera holds a static shot as ...".
- `overall_soundscape:` 1-4 sentences of ambience, physical and non-verbal sounds. No dialogue here. `N/A` only
  for total silence.
- `non_diegetic_music:` 1-3 sentences on the score: instruments, tempo, dynamics, no mood words. `N/A` if there is
  none.

**3. Dialogue, voice-over, on-screen text**

- **Speakers:**
  - Give each one a stable ID, `(S1)` or `(S2)`, or `(S1,S2)` when they speak together.
  - Describe the voice (age, timbre, on or off screen) **outside** the tags.
- **Inside `<d>...</d>`** goes only `[Language]` and the exact words, untranslated:
  `the fisherman with a low, gravelly voice (S1) says: <d>[English] The tide is turning.</d>`
- **Capital first letter:** start the words in each `<d>` with a capital letter, because a lowercase line that
  opens the clip can add a stray syllable at 0 s (measured in the A/B of issue #38; see issue #53).
- **Voice-over:** `says in an off-screen voiceover: <d>[English] ...</d> while his lips remain completely closed`.
- **Cuts and the clip's end** (MiniMax's rule; its guides give no example with these tokens, nor say whether they
  go inside `<d>`):
  - A line of dialogue or lyrics that runs across a cut takes `<scenetrans>` at the connecting point in both
    parts, and a sentence saying that the audio continues across the cut: "continues seamlessly across the cut",
    "continues uninterrupted into the next shot", "carries over from the previous shot" or "remains audible across
    the transition". MiniMax's t2va example has such a sentence: `[Shot 2] At 00:05.000, the camera cuts to a
    close-up of steam rising from the sliced bread while the baker's final words carry over from the previous
    shot.`
  - Speech cut off by the end of the clip takes `<cutoff>`.
- **Text visible in the picture** goes in double quotes, verbatim.
- **Language:** write the sections in English. Dialogue and visible text stay in their own language.

**4. fl2va: the keyframe line comes first**

Start with exactly one of these lines, then a blank line, then the three fields.
- `S.SS` is the clip length with two decimals (8 s → `8.00`).
- `N` is the shot the frame belongs to: the last shot, for a last frame.

- First frame only:
  `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
- First and last frame:
  `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.`
- Last frame only (it is `<Picture 1>`):
  `How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.`

Describe how the scene gets from one frame to the other. Don't describe the frames themselves again.

**5. ref2va: six sections and labels**

Sections, in this order, each name on its own line with its text below: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.

- **Labels:**
  - `<Picture N>` numbers the images in the order you send them, and `<Video N>` the videos.
  - A reference video **with a sound track also takes an `<Audio N>` number, just before its own `<Video N>`**.
    Audio files are numbered after the videos' sound tracks.
  - Example: image, video with sound, WAV → `<Picture 1>`, `<Audio 1>` (the video's sound), `<Video 1>`,
    `<Audio 2>` (the WAV).
  - The job's `labels` list shows the numbering the studio used.
- **`<Subject N>`** names visible content you define in `subject_definitions` and reuse. A label keeps one meaning
  in every section.
- **`summary`** starts with a task tag, several joined with ` + `: `[reference generation]`, `[keyframe
  completion]`, `[video editing]`, `[video continuation]`, `[audio reuse]`, `[audio reference]`.
- **`retention_analysis`** has one line per label:
  - for pictures and videos: `fully_preserved`, `partially_preserved`, `attribute_transfer` or `weak_reference`;
  - for audio: `fully_copy`, `partially_copy`, `reference` or `weak_reference`.
- **`detailed_description`**:
  - one or two style sentences, then the shots as in section 2;
  - a speaking subject is written `<Subject 1> (S1)`;
  - generation tasks run 350-500 words.

**6. Examples**

t2va, 5 s:

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames an elderly fisherman in a yellow raincoat mending a net on a wooden pier at dawn. The camera pushes in with small amplitude at slow speed as the fisherman with a low, gravelly voice (S1) looks out to sea and says: <d>[English] The tide is turning.</d> Gulls circle above the grey water behind him.

overall_soundscape: Small waves slap against the pier posts and gulls call in the distance. Wet rope creaks as he pulls the net across his knees.

non_diegetic_music: N/A
```

fl2va, first frame only, 5 s:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the scene opens on the kitchen shown in <Picture 1>, keeping the woman's appearance, the red kettle and the morning window light. The camera holds a static shot as she lifts the kettle and pours steaming water into a white mug, then wraps both hands around it while steam drifts across the window.

overall_soundscape: Water pours into the mug with a rising pitch, then the kettle clinks back onto the stove. Quiet room tone and faint birdsong come through the window.

non_diegetic_music: N/A
```

fl2va, first and last frame, 8 s:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot begins in the composition of Picture 1, with the small sailboat moored and its sail furled. The camera trucks left with small amplitude at slow speed as the sailor unties the bow line, hoists the sail and pushes off; the boat drifts away from the dock and settles into the position, sail shape and framing of Picture 2 at the end of the shot.

overall_soundscape: Rope runs through a pulley and canvas flaps as the sail fills. Water laps against the hull while a light breeze moves across the harbor.

non_diegetic_music: N/A
```

fl2va, last frame only, 6 s:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot begins on an empty ceramic plate on a wooden café table as a hand enters from the left carrying a slice of cake. The camera pushes in with small amplitude at slow speed while the hand sets the slice down, places a fork beside it and withdraws, and the table settles into the exact arrangement, lighting and framing of <Picture 1>.

overall_soundscape: The plate clinks softly on the table and the fork taps the ceramic rim. Low café chatter continues in the background.

non_diegetic_music: A light solo piano at a moderate tempo that fades out at the end.
```

ref2va, one portrait image then one voice recording (WAV), 5 s. The labels are `<Picture 1>` and `<Audio 1>`:

```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>, with short red hair, round glasses and a green wool coat.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] <Subject 1> stands on a snowy city street at night and greets the camera in the voice timbre of <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - her face, short red hair, round glasses and green wool coat are retained.
<Audio 1>: reference - her voice follows the timbre of <Audio 1> without copying the original signal.

detailed_description:
The target video is a live-action, cinematic night scene on a quiet city street in winter, lit by warm streetlights and the cool glow of shop windows, with a shallow depth of field and soft, slightly desaturated colors.
[Shot 1] A static medium shot frames <Subject 1> (S1), the woman with short red hair, round glasses and a green wool coat, standing on a snow-covered sidewalk under a single streetlight, centered in the frame from the waist up. Behind her the street falls out of focus: a row of closed shops with lit windows, parked cars under a thin layer of snow, and the red tail lights of a car passing slowly from left to right. A thin wisp of steam rises from a street grate behind her and drifts across the lit windows. Large snowflakes drift down through the cone of warm light above her, catching the light as they fall, and a few settle on her red hair, on the rims of her glasses and on the shoulders of her coat. The coat is buttoned to the collar, its wool texture visible in the light, and fresh snow lies on the pavement around her boots. She stands still at first, her hands loose at her sides, then lifts her gaze from the ground to the lens, and her face softens into a warm, slightly tired smile. She takes a small breath that clouds in the cold air and says in the clear, warm voice referenced from <Audio 1>, <d>[English] Welcome back. It has been a long winter.</d> Her delivery is calm and close, with the smile audible in it. As she finishes speaking, she tucks her hands into her coat pockets, shifts her weight onto one foot and tilts her head slightly to the side, still looking into the lens. Her breath keeps rising in small white clouds that drift up through the falling snow and fade in the light. The lenses of her glasses catch a brief reflection of the streetlight as she moves her head. The camera stays still; only the snow, her breath and the distant traffic change across the shot, which ends on her steady, smiling gaze with the snow still falling around her.

overall_soundscape:
Soft wind and the hush of falling snow, with distant traffic on wet pavement.

non_diegetic_music:
N/A
```

Add a reference video with sound after the image, and the labels become `<Picture 1>`, `<Audio 1>` (the video's
sound), `<Video 1>` and `<Audio 2>` (the WAV). The prompt must then use those numbers.

**7. More**

- MiniMax's guides: [base modes](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
  and [reference mode](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
- Their agent skill [`h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)
  rewrites free-form ideas into this format.

<!-- prompt-guide:end -->
