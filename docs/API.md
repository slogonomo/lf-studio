# API for agents and scripts

The studio's page talks to its server over HTTP, and programs can use the same API: queue a clip with its input
files, wait for it, and download the MP4. This page documents that API and two clients built on it:

- **`bin/sol-h3`**, a command line. It is one Python file (standard library only, Python 3.9 or later), so it can be
  copied to any machine that reaches the studio.
- **`sol-h3 mcp`**, the same file as an MCP server: agent hosts such as Claude Code, Hermes or Codex get tools to
  make clips, wait for them and save them.

H3 takes its prompt in a structured format, and nothing in the studio rewrites a prompt. The
[prompt guide](#prompt-guide) at the end says how to write one; it is also `sol-h3 guide` and the MCP tool
`prompt_guide`.

Times on this page were measured on one DGX Spark ([PERFORMANCE.md](PERFORMANCE.md)). Every curl, CLI and
MCP-registration example was run against the stub studio (`tests/stub/studio.py`), which makes small test clips
without a GPU.

Contents: [Quick start](#quick-start) · [Access](#access) · [Through spark-switchboard](#through-spark-switchboard) ·
[Making a clip](#making-a-clip-post-apigenerate) · [Waiting](#waiting-get-apijobsid) ·
[Getting the clip](#getting-the-clip) · [Cancel](#cancel) · [Other endpoints](#other-endpoints) ·
[Errors](#errors) · [CLI](#cli-binsol-h3) · [MCP server](#mcp-server-sol-h3-mcp) · [Prompt guide](#prompt-guide)

## Quick start

Each example makes a 5 s clip from the first frame `kitchen.png` and the prompt in `prompt.txt` (an fl2va prompt,
as in the [prompt guide](#prompt-guide)), and saves it as `clip.mp4`. The studio's address is
`http://<spark-host>:8190` from another machine (through spark-switchboard, or a studio started with
`SOLH3_HOST=0.0.0.0`), or `http://127.0.0.1:8190` on the Spark itself.

**The CLI.** Get the file once (`~/.local/bin` must be on your `PATH`), then:

```bash
mkdir -p ~/.local/bin && curl -fsSLo ~/.local/bin/sol-h3 \
  https://raw.githubusercontent.com/slogonomo/sol-h3-studio/main/bin/sol-h3 && chmod +x ~/.local/bin/sol-h3
export SOLH3_URL=http://<spark-host>:8190
sol-h3 generate --first-frame kitchen.png --prompt-file prompt.txt --seconds 5 --out clip.mp4
```

It shows progress on stderr and prints the saved file's absolute path on stdout. On the Spark,
`bin/sol-h3-studio generate ...` runs the same client with the studio's env file, so it needs no `SOLH3_URL`.

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

**An agent host over MCP**: register `sol-h3 mcp` ([MCP server](#mcp-server-sol-h3-mcp)), for example in Claude
Code:

```bash
claude mcp add --scope user --transport stdio sol-h3 -e SOLH3_URL=http://<spark-host>:8190 \
  -e SOLH3_MCP_WAIT_S=600 -e SOLH3_MCP_MAX_WAIT_S=1800 -- python3 /path/to/sol-h3 mcp
```

## Access

- **There is no login.** The studio listens on 127.0.0.1 by default. With `SOLH3_HOST=0.0.0.0`, or behind
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
  `.home.arpa` or `.internal`, and the names in `SOLH3_ALLOWED_HOSTS` ([bin/env.example](../bin/env.example)).
  Any other name in `Host` or `X-Forwarded-Host` gets 421 `unknown host name <name>; add it to
  SOLH3_ALLOWED_HOSTS`. This stops DNS rebinding, where a web page uses a name of its own that resolves to your
  studio's address. It does not stop anyone who reaches the port directly. If you reach the studio by a domain of
  your own, add that name to `SOLH3_ALLOWED_HOSTS`.
- **Plain HTTP:** requests and clips cross the network unencrypted unless the network encrypts them. Tailscale
  does, through WireGuard.
- An access token for the API is possible future work; nothing sends or checks one today.

## Through spark-switchboard

With [spark-switchboard](https://github.com/slogonomo/spark-switchboard), port 8190 belongs to switchboard. It
starts the studio when something needs it and stops it when the memory is wanted elsewhere or the studio has been
unused for 30 min (`integrations/spark-switchboard/sol-h3-studio.toml`). What a program's request does there:

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
   GET /api/state (curl/8.5.0 from ...)"). `bin/sol-h3` sends `sol-h3/0.4.0`.
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
   which takes about 110 GiB of the Spark's memory from other apps (and preloads the models). Do that only on
   purpose. A studio without switchboard answers that path with 404 `{"error": "not found"}`: no switchboard, and
   the request cost nothing.
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
| `seconds` | 3, 4, 5, 6, 7, 8, 10, 12 or 15 | 5 |
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
bin/sol-h3, which downloads URLs itself`. The CLI and the MCP server download URLs on their own machine and upload
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

Before the fields are read: 411, 400 (`bad Content-Length`) or 413 for the body's length, 415 for its type, and
429 while `SOLH3_MAX_QUEUED` clips (default 20) are waiting in the queue (the running clip does not count), with
`Retry-After: 60`: `the queue is full (20 clips waiting); try again when one has finished`.

Then the form and JSON go through the same checks, with the same messages, and a refused request leaves nothing
behind:

1. Task and orientation (`bad task/orientation`), the task's path manifest (`no path manifest for ...`), the
   prompt (`prompt is empty`, `prompt is too long (N characters; at most 32768)`, and `prompt is not valid text
   ...` for a JSON prompt holding half of a UTF-16 surrogate pair, such as `"\ud83c"` from an emoji cut in two),
   the length (`duration must be one of (3, 4, 5, 6, 7, 8, 10, 12, 15) seconds`) and the seed (`seed must be a
   nonnegative 63-bit integer`).
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
- 404 `no such clip` for an unknown or deleted clip; `not found` for an id that is not 32 hex characters.
- 503 `{"error": "the studio is shutting down; ask again once it is back", "job": <record>}` if the studio stops
  during the wait. After its restart, a clip that was queued or running reads `failed` with `server restarted`
  (or `cancelled`, if a Cancel had been answered).

### The API record

The clip as stored (the keys `/api/state` shows), plus these keys, worked out for the answer:

| Key | Value |
|---|---|
| `state` | `queued`, `running`, `done`, `failed` or `cancelled`. Use this; `status` is a phrase for people (`loading models + generating`, ...). `done`, `failed` and `cancelled` are final. |
| `phase` | `null` unless running; then `starting`, `loading` (the models, then this clip), `generating`, `retrying` (after a worker error), `saving`, `cancelling` or `running` |
| `ahead` | queued only: the clips queued before it |
| `elapsed_s` | running only: seconds since it started |
| `urls` | done, while its MP4 exists: `{"video": "/video/<id>", "poster": "/poster/<id>"}`, relative to the studio's address |
| `labels` | fl2va and ref2va: what each input file is called in the prompt (below) |

Some stored keys: `id`, `task`, `orientation`, `seconds`, `seed`, `prompt`, `created`, `started` and `finished`
(Unix times), `wall_s` (seconds from start to end), `error` (failed), `load_eta` (while the models load: the last
measured load plus clip for these settings, in seconds), `includes_model_load`. Paths in the record
(`first_frame`, `references`, `output`, `published`, `case_dirs`) are paths on the Spark, for information only.

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
| t2va with the models loaded (the same task, orientation and length as the clip before) | 3 s: 36 s · 5 s: 54-66 s · 10 s: 123 s · 15 s: 191 s |
| ref2va with a 5 s reference video, 5 s | 133 s per clip |
| t2va, the first clip after a change of task, orientation or length (the models load first) | 5 s: 104-113 s in the studio, 99 s with `infer.py` · 15 s: 238 s with `infer.py` |
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
  - 400 and 503 as for `/api/jobs/<id>`.

  Behind switchboard this path never starts the studio, and it sends the MP4 the moment the clip is done.
- **`GET /video/<id>`**: the MP4 (`video/mp4`) with `Content-Disposition: inline;
  filename="sol-h3_<task>_<seconds>s_seed<seed>_<id6>.mp4"`. `Range` requests get 206. 404 `no such video`. Behind
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

## Other endpoints

| Endpoint | Answer |
|---|---|
| `GET /api/state` | What the page shows: `{"now", "tasks_ready", "trash_days", "warmable", "session", "jobs", "done", "more", "deleted"}`. `jobs` holds every queued or running clip and the newest 24 finished ones, newest first, as stored (without the API record's keys). switchboard's busy probe reads it. |
| `GET /api/jobs?before=<created>&limit=<n>` | Older clips, newest first: `{"jobs": [...], "more": true/false}`. `before` is a `created` time (default: now), `limit` 1-100 (default 24). Records as stored. |
| `GET /api/inputs/<id>` | A clip's input files, for Remix: `{"task", "inputs": {"<zone>": [{"name", "index", "url"}]}, "missing": [...]}` |
| `GET /input/<id>/<zone>/<index>` | One input file (reference videos as converted). 404 `no such input`. |
| `GET /still/<id>` | The picture a queued or running clip's card shows (a keyframe, a reference image, or a reference video's first frame). 404 `no picture` for t2va. |
| `GET /health` | `{"ok": true}` |

Used by the page; stable, but not designed for agents. Each takes a JSON body of at most 64 KiB (413 above):

| Endpoint | Body | Does |
|---|---|---|
| `POST /api/delete` | `{"id"}` | moves a finished clip and its files to the trash for `SOLH3_TRASH_DAYS` (default 7); 409 while it is queued or running |
| `POST /api/undelete` | `{"id"}` | brings it back |
| `POST /api/load` | `{"task", "orientation", "seconds"}` | loads the models for those settings ahead of a clip ("Load selected now"); 409 while a clip is queued or running |
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
| 404 | studio | no such clip, video, poster, input or path (`not found`); on `/__switchboard/status`: no switchboard |
| 409 | studio | the clip failed or was cancelled (`/api/jobs/<id>/video`); Cancel of a finished clip; Delete of a running one |
| 411 | studio | `Content-Length required (chunked uploads are not supported)` |
| 413 | studio | over 200 MB (`/api/generate`) or 64 KiB (the other POSTs) |
| 415 | studio | `/api/generate` with a body that is neither form nor JSON |
| 416 | studio | a `Range` outside the file |
| 421 | studio | `unknown host name <name>; add it to SOLH3_ALLOWED_HOSTS` ([Access](#access)) |
| 429 | studio | the queue is full; `Retry-After: 60` |
| 502 | switchboard | `upstream_down`: the studio stopped answering |
| 503 | switchboard | not running: `loading`, `waiting`, `no_memory`, `failed`, `stopping`, `stopped`, `paused`; `Retry-After`. A POST that got it was not passed on. |
| 503 | studio | `the studio is shutting down ...` with the clip's record (`/api/jobs/<id>`) |

## CLI: `bin/sol-h3`

`bin/sol-h3` is one Python file: standard library only, Python 3.9 or later, and it reads no other file of this
repo. On another machine, get it as in the [Quick start](#quick-start) (or copy it from a checkout). On the Spark,
`bin/sol-h3-studio <command> ...` runs it with the settings of the studio's env file.

```
sol-h3 generate (--prompt TEXT | --prompt-file FILE|-) [--task t2va|fl2va|ref2va]
       [--orientation landscape|portrait] [--seconds 3|4|5|6|7|8|10|12|15] [--seed N]
       [--first-frame FILE|URL] [--last-frame FILE|URL] [--ref FILE|URL]...
       [--wait] [--out PATH] [--poster] [--timeout S] [--hold S]
sol-h3 wait ID [--out PATH] [--poster] [--timeout S]
sol-h3 status [ID]
sol-h3 cancel ID
sol-h3 fetch ID [--out PATH] [--poster] [--wake]
sol-h3 guide                  # the prompt guide
sol-h3 mcp                    # the MCP server on stdin/stdout
common options, before or after the command: --url URL  --json  --quiet
```

**The studio** is `--url`, else `$SOLH3_URL`, else `http://$SOLH3_HOST:$SOLH3_PORT` (`0.0.0.0` or empty counts
as `127.0.0.1`; a missing port is 8190), else `http://127.0.0.1:8190`. Behind switchboard, use its port (8190).

**`generate`** checks everything it can before it uploads: the task against the files (without `--task`: frames →
fl2va, `--ref` → ref2va, none → t2va), types, H3's counts, and a total of 199 MB. `--ref` files go to images, videos
or audio by their extension, numbered in the order given per kind. A file may be a path or an http(s) URL; URLs are
downloaded by the CLI (so tailnet addresses work), at most 120 s each, typed by the URL path's extension or else
by `Content-Type`. Without `--orientation`, the first keyframe's shape picks it (stderr says so), and a keyframe
more than 5% off the clip's aspect ratio gets a warning. The upload sends `X-Switchboard-Wait: <--hold>` (default
900 s).

- Without `--wait`, `--out` or `--poster`, it prints the clip's id and exits.
- With any of them it waits, then saves the MP4: into the current directory under the clip's own name
  (`sol-h3_<task>_<s>s_seed<seed>_<id6>.mp4`), into a directory given as `--out` (an existing one, or a path ending
  in `/`), or as the `--out` file, which it replaces. `--poster` also saves the poster as `<name>.jpg`. It checks
  that it can write there before it submits (exit 2 if not).
- `--timeout S` stops waiting after S seconds (exit 4, the id on stdout); the clip keeps running. Ctrl-C while
  waiting also leaves the clip running (exit 130) and says how to pick it up: `sol-h3 wait <id> --out ...`.

**`wait ID`** waits for a clip and saves it as above. **`fetch ID`** saves a finished clip (also one made in the
page) without waiting; a clip still running is exit 4. **`status`** prints one line per clip: the queued and
running ones, oldest first, then the newest 5 finished ones; `status ID` one clip. **`cancel ID`** cancels it.

**Waiting** is a loop of `GET /api/jobs/<id>/video?wait=15`, so the MP4 comes on the connection that saw the clip
finish. It shows one progress line per change on stderr (`queued 5cfd57b0 (fl2va, landscape, 5 s, seed 1234),
next in line`, `loading models, then generating (about 1:53 in all)`, `generating`, `done in 1:05`). Connection
errors and switchboard's `loading`, `waiting` and `upstream_down` answers are retried every 5 s for up to 120 s.
The MP4 is written to a `.part` file, checked against `Content-Length` and then renamed.

**With spark-switchboard**, `status`, `wait`, `cancel` and `fetch` read `/__switchboard/status` first and never
start a stopped studio. `fetch` then refuses (exit 3, "fetching would start it (it needs about 110 GiB). Add
--wake."); `fetch --wake` starts it, holding up to 900 s. If switchboard stops the studio while `wait` or `generate
--wait` waits, the clip most likely finished first; the CLI says so and how to get it (`sol-h3 fetch <id> --wake`).

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
sol-h3 generate --prompt-file t2va.txt --seconds 8 --seed 7 --wait            # t2va, saved here
sol-h3 generate --ref face.png --ref voice.wav --prompt-file ref.txt --out clips/   # ref2va, into clips/
id=$(sol-h3 generate --prompt-file t2va.txt)                                   # queue only
sol-h3 status "$id"
sol-h3 wait "$id" --out clip.mp4 --poster --timeout 600
sol-h3 --json fetch "$id" --out clips/
```

## MCP server: `sol-h3 mcp`

`sol-h3 mcp` speaks MCP over stdin and stdout (newline-delimited JSON-RPC), both the `initialize` handshake
(protocol versions 2024-11-05 to 2025-11-25) and the 2026-07-28 revision. It runs on the agent's machine: it reads
the agent's files there, uploads them, and saves the MP4 there. The studio itself has no MCP endpoint.

### Settings

From the MCP process's environment (the host's server config):

| Variable | Default | Meaning |
|---|---|---|
| `SOLH3_URL` | as the CLI | the studio, `http://<spark-host>:8190` |
| `SOLH3_OUT_DIR` | `./sol-h3-clips` (from the process's working directory; Claude Code starts it in the project directory) | where MP4s go when `out` is not given |
| `SOLH3_MCP_WAIT_S` | 50 | how long `generate` and `wait` wait when the call gives no `wait_s` |
| `SOLH3_MCP_MAX_WAIT_S` | 1800 | the cap on any `wait_s` |
| `SOLH3_MCP_OUT_ANYWHERE` | unset | `1` lets `out` point outside `SOLH3_OUT_DIR` and the working directory |

### Tools

| Tool | Arguments | Does |
|---|---|---|
| `generate` | `prompt` (required), `task`, `orientation`, `seconds`, `seed`, `first_frame`, `last_frame`, `references` (paths on the agent's machine or http(s) URLs), `wait_s`, `out`, `poster` (default true), `again` | queues a clip, waits up to `wait_s` and, when it is done, saves the MP4 |
| `wait` | `id` (required), `wait_s`, `out`, `poster` | waits for a clip and saves it |
| `status` | `id` | one clip's record; without an id, the clips this session submitted, then the queue and the newest clips. Never waits, downloads or starts a stopped studio. |
| `cancel` | `id` (required) | cancels a queued or running clip |
| `fetch` | `id` (required), `out`, `poster`, `wake` (default false) | saves a finished clip (also one made in the page); refuses on a studio switchboard has stopped unless `wake` is true |
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
- **`out`** must be an existing directory or a name ending in `.mp4`, inside `SOLH3_OUT_DIR` or the working
  directory (unless `SOLH3_MCP_OUT_ANYWHERE=1`). An existing file is never replaced: the clip is saved as
  `<name>-1.mp4`, `-2`, ... instead.
- **Relative paths** in `out`, `first_frame`, `last_frame` and `references` start from the MCP server's working
  directory, not from `SOLH3_OUT_DIR`. That is the project directory in Claude Code; other hosts may start the
  server anywhere, so agents there should pass absolute paths. A missing file's error names the absolute path it
  looked for.
- **Before it submits**, `generate` checks that it can write where the clip goes (`out`, else `SOLH3_OUT_DIR`, or
  the directory above it that exists). A host that starts the server in a directory it can't write to (`/`, say)
  gets a refusal naming `SOLH3_OUT_DIR`, not a rendered clip that can't be saved. If saving fails all the same, the
  error says the clip is done on the studio and that `fetch` with its id and a writable `out` saves it.
- **Progress:** when the host asks for it, `notifications/progress` every 10 s while a call waits, with the phase as
  the message.
- **spark-switchboard:** as the CLI; a stopped studio makes `wait` and `fetch` answer that `fetch` with `wake: true`
  would start it.

### Waiting values per client

A tool call may run only as long as the host lets it. Set `SOLH3_MCP_WAIT_S` below that limit; `wait_s` in a call
can ask for more, up to `SOLH3_MCP_MAX_WAIT_S`. The limits as the hosts document them (2026-10):

| Host | Limit per tool call | `SOLH3_MCP_WAIT_S` |
|---|---|---|
| Claude Code (stdio) | none; a call is stopped after 30 min without a result or progress, and the server sends progress every 10 s. A call still running after 2 min moves to the background, and its result arrives later. | 600 |
| Hermes | `timeout`, 300 s by default | 280, or 1700 with `timeout: 1800` when Hermes's model runs on the same Spark |
| Codex CLI | `tool_timeout_sec`, 60 s by default | 50, or raise `tool_timeout_sec` |
| Other hosts on the TypeScript SDK 1.x | 60 s | 50 (the default) |

### Registration

**Claude Code** (the name comes before the `-e` options):

```bash
claude mcp add --scope user --transport stdio sol-h3 -e SOLH3_URL=http://<spark-host>:8190 \
  -e SOLH3_MCP_WAIT_S=600 -e SOLH3_MCP_MAX_WAIT_S=1800 -- python3 /path/to/sol-h3 mcp
claude mcp list          # sol-h3: python3 /path/to/sol-h3 mcp - ✔ Connected
```

**Hermes** (`~/.hermes/config.yaml`):

```yaml
mcp_servers:
  sol_h3:
    command: "python3"
    args: ["/path/to/sol-h3", "mcp"]
    env: {SOLH3_URL: "http://<spark-host>:8190", SOLH3_OUT_DIR: "/path/for/clips", SOLH3_MCP_WAIT_S: "280"}
    timeout: 300
```

If Hermes's own model runs on the same Spark, use `SOLH3_MCP_WAIT_S: "1700"`, `SOLH3_MCP_MAX_WAIT_S: "1800"` and
`timeout: 1800`, so that one call covers the submit, the wait and the download
([Through spark-switchboard](#through-spark-switchboard)). Set `SOLH3_OUT_DIR`: Hermes may start the server
outside your project directory, and `out` must lie inside `SOLH3_OUT_DIR` or the server's working directory. A
relative `out` or input path still starts from that working directory, so tell the agent to pass absolute paths.

**Codex:**

```bash
codex mcp add sol-h3 --env SOLH3_URL=http://<spark-host>:8190 -- python3 /path/to/sol-h3 mcp
```

This keeps `SOLH3_MCP_WAIT_S` at 50, under Codex's 60 s. To wait longer per call, add
`--env SOLH3_MCP_WAIT_S=580` to the command, and `tool_timeout_sec = 600` to the `[mcp_servers.sol-h3]` table it
writes to `~/.codex/config.toml`.

**Any other host:** the command `python3 /path/to/sol-h3 mcp`, with `SOLH3_URL` in its environment, and
`SOLH3_OUT_DIR` too unless the host starts the server in a directory where clips should go.

## Prompt guide

The same text is `sol-h3 guide` and the MCP tool `prompt_guide`; a test keeps the copies identical.

<!-- prompt-guide:start -->

Sol-H3 Studio gives your prompt to MiniMax-H3 unchanged; nothing rewrites it. H3 was trained on a structured
format, and MiniMax calls that format critical to the quality of the result, so write it yourself as below.
Plain prose is accepted. The prompt drives the draft picture and **all of the sound**. The refiner that makes the
final 1344x768 video uses a fixed prompt of its own.

**1. Task, length, format**

- `t2va`: text only. `fl2va`: a first frame, a last frame, or both (images).
  - A keyframe should have the clip's shape (1344x768 or 768x1344, or the same aspect ratio): one of another
    shape is stretched or cropped to fit. Without an orientation, `sol-h3` takes it from the first keyframe's
    shape.
- `ref2va`:
  - up to 9 images, 3 videos (2-15 s each) and 3 audio files, 12 files in total;
  - at least one image or video.
- `seconds`: 3, 4, 5, 6, 7, 8, 10, 12 or 15.
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
  - Camera moves (push in, pan right, tracking shot, static shot, ...) take an amplitude and a speed in plain
    words: "the camera pushes in with small amplitude at slow speed".
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
- **Voice-over:** `says in an off-screen voiceover: <d>[English] ...</d> while his lips remain completely closed`.
- **Cuts and the clip's end:** speech that runs across a cut takes `<scenetrans>`; speech cut off by the end of
  the clip takes `<cutoff>`.
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
The target video is a live-action, cinematic night scene with warm streetlights and falling snow.
[Shot 1] A static medium shot frames <Subject 1> (S1), the woman with short red hair, round glasses and a green wool coat, standing under a streetlight as snowflakes drift past. She smiles into the lens and says in the clear voice referenced from <Audio 1>, <d>[English] Welcome back. It has been a long winter.</d> She tucks her hands into her coat pockets as her breath clouds in the cold air.

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
