#!/usr/bin/env python3
"""LF Studio: a small web front end for NVIDIA Research's Sol-H3-Spark pipeline (unofficial), and for its
Sol-H3-RTX5090 variant on a workstation with one RTX 5090 (LFSTUDIO_BACKEND=rtx5090; docs/RTX5090.md).

- One job at a time (FIFO queue). Models stay loaded between jobs with the same task (one Pipeline session). A
  clip with another length or orientation changes the loaded session's shape in place (Pipeline.reshape: Stage 1
  and Stage 2 keep their models; only the prompt encoder closes or loads again when a clip crosses 10 s; the first
  clip at a shape the session has not run compiles for it). Another task closes the session and loads the models
  again. LFSTUDIO_RESHAPE=0 closes and reloads for any change, as before 0.4.
- Tasks: t2va (text), fl2va (first and/or last frame), ref2va (reference images, videos, audio).
- Output: 1344x768 (landscape) or 768x1344 (portrait), 24 fps, 5, 6, 7, 8, 10, 12 or 15 s, with audio. 3 and 4 s,
  offered up to 0.5.0, get 400 (length_refusal), also from /api/load. Clips made at those lengths stay in the history:
  they play and download as before, Remix reuses their files, and the start-up warm-up after one loads 5 s.
- POST /api/cancel {"id": ...}: a queued clip leaves the queue at once; a running one is stopped (its pipeline
  workers get SIGTERM, so the next clip loads the models again). Either way it stays in the history with the status
  "cancelled" and its input files, for Remix and Delete. 409 for a clip that has finished (or is being saved).
  A studio before 0.3 shows cancelled clips as failed ("server restarted") and keeps them so.
- GET /still/<id>: the picture a queued or running clip's card shows behind its timer: a keyframe, a reference image,
  or the first frame of a reference video, as a JPEG at most 640 px wide (made on the first request, kept with the
  clip's input files; an image ffmpeg can't read goes out as it is). 404 for t2va.
- Experimental, with LFSTUDIO_UPSCALE=1: POST /api/upscale {"id", "resolution", "strength", "audio", "seed", "pin"}
  upscales a finished clip ("id") of a task, model and length the models' "upscale" tables have (MODELS: Sol-H3's
  t2va, fl2va and ref2va clips, LightX2V Turbo's t2va and fl2va clips at 576p or 768p; every length the studio makes,
  5 to 15 s, not Sol-H3's 3 and 4 s clips made up to 0.5.0), 16:9 or 9:16, from its MP4, whole, in one pass of NVIDIA's
  refine, to 1080p, 2k (2560x1440, or 1440x2560 for 9:16; the default) or 4k with NVIDIA's Open-H3-Regenerate-2K
  (upscale/regen2k_job.py, run in its own environment; it upscales a 9:16 clip through NVIDIA's code with the shapes
  turned, see its docstring), at the sizes the job offers for the clip's length (its offer_row: 1080p every length,
  2K up to 10 s, 4K 5 s clips; /api/state "upscale_offers"; 6, 10 and 15 s at 1080p and 10 s at 2K have run on the
  GPU, the other lengths follow the job's model of those runs). strength: faithful, balanced or full (upscale/regen2k_job.py STRENGTHS; default
  LFSTUDIO_UPSCALE_STRENGTH). audio: clip or silence, what the picture is refined against (default
  LFSTUDIO_UPSCALE_AUDIO). seed: the refine's (default the clip's). pin: auto (the default) or off, Keep my frames
  (experimental; PIN_ROUTES "ran": where it has run on the GPU): with auto, the clip's keyframes are held in the refine
  where its route offers it (PIN_ROUTES: LightX2V Turbo's Keyframes clips, either end or both, and Sol-H3's first frame
  on a clip made with the Stage-2 pin; 1080p and 2K, balanced) and the clip shows each picture already (the job's own
  check, before anything loads: 28 dB at 384p, the sha256 the clip was made from); off, or no end held: NVIDIA's
  refine. The upscale is a job of its own (kind "upscale", parent = the clip; task, format, length, prompt and seed
  copied from it; "load" = LFSTUDIO_UPSCALE_LOAD; "parent_model" and "parent_opts", the clip's model and options;
  "video", what is done to its MP4 (upscale_sources); "frames", the frames it keeps (its MP4's: 121 at 5 s, 241 at
  10 s; 121 of a 5 s clip's 124 at 4K); for a clip with keyframes "pin":
  {"asked": auto or off, "ends" (held if the job's check passes), "why" ({end: why not sent}) and, once done, "result"
  ({end: {"used", "agreement_db", "why"}}, the job's receipt)};
  "text", what it is refined against: {"source": "clip"}, the clip's own conditioning, or {"source": "generation" (a
  LightX2V Turbo Keyframes clip's qwen-text/, made with it) | "encode", "prompt", "dropped", "notes"}, the clip's prompt
  without its pictures (upscale_text), with "rows" and "encode_s" once encoded) in the same queue, where it waits while
  any clip is queued. It frees the memory first (status "freeing memory": the session is closed), refuses to start
  below its floor of MemAvailable (the job's min_free_for: LFSTUDIO_UPSCALE_MIN_FREE_GIB for a 5 s clip, 5 GiB more at
  4K; a longer clip what its case needs where that is more, 111 GiB at 2K 10 s with a pin; job.json "min_free_gib",
  which the job checks again as it starts), then, in one pipeline_call so that Cancel stops each step:
  builds its input folder (build_input: <case>/input/, the clip's MP4, audio and text alone, linked; a LightX2V Turbo
  clip at 576p stretched to 768p, every frame; one with a last keyframe (every frame made: 124 at 5 s) whole, a 5 s
  one cut to its first 121 at 4K; the pictures of the ends sent, linked into keyframe/; source.json), for a clip
  without a conditioning of text
  alone encodes one (status "encoding the prompt", upscale_phase "prompt": the prompt encode,
  LFSTUDIO_H3TURBO_ENCODE_SCRIPT --upscale, a transient Qwen worker, as a process group of its own; the memory floor
  again after it) and runs the job on that folder in a process group of its own (status "upscaling",
  "upscale_phase" from the job's "[lf-studio] phase" lines); it stops the job when MemAvailable stays below
  LFSTUDIO_UPSCALE_GUARD_GIB for 3 s. The clip's sound is copied in from the input folder's MP4 (ffmpeg, no re-encode;
  whole, or for a clip cut to 121 frames at 4K, cut with them). Its files go into a session directory of its own,
  <session>-upscale-<resolution>/<case_id>/ (input/, regen/; with an encode also prompt-text-only.txt and encode.log),
  so Delete and Undo move only them; a clip can't be deleted while an upscale of it is queued or running. No retry
  after a failure.
  Answers: 200 {"id", "job"}, 400 for a bad field (any other key too, "source" among them: there is no draft source any
  more), a clip it does not take (a length no model's table has, UPSCALE_LENGTH; an MP4 whose frames are not its
  length's, UPSCALE_FRAMES; a Keyframes or Reference clip with nothing left of its prompt without its pictures) or a
  size the job does not offer for its length (its reason, which names the sizes that take it: "2K for 12 s clips comes
  later: 1080p takes them"), 404 for an unknown clip (and for everything with the Upscale off), 409 for a clip
  that is not done, being moved, or without the files needed (its session directory; for audio clip its audio: Sol-H3's
  Stage-1 audio, LightX2V Turbo's turbo/audio.pt), 429 when the queue is full. /api/state has "upscale" (on or off),
  "upscale_defaults" (null with it off, else what a request without those keys gets: {"resolution", "strength",
  "audio"}, for the page's choices) and "upscale_load" (null with it off, else LFSTUDIO_UPSCALE_LOAD, for the page's
  expected times), "upscale_pin" (null with it off, else true: the studio takes "pin") and "upscale_offers" (null with
  it off, else UPSCALE_OFFERS: per length, {"<seconds>": {resolution: {"eta_s": [seconds before the refine, per refine
  step], "ran" (that length and size, or a longer clip at that size, upscaled on the GPU through the studio; else the
  times are estimates)} or
  {"why": why not offered}}}); with it on, every finished job in /api/state, /api/jobs and the API record has
  "upscalable": {"ok": true, "clip_audio": bool} (with "text": "generation" or "encode" when it is not refined against
  the clip's own conditioning; "video": "stretch" when its MP4 is not taken as it is at 1080p and 2K; "frames":
  {resolution: frames kept} where they differ between sizes (124, and 121 at 4K); "pin": the ends it can hold [first,
  last or both], with "pin_routes": {resolution: {"strengths", "ran" (on the GPU through the studio, at the clip's
  length)}} at the sizes offered for its length;
  "pin_why": {end: why not} for an end it has but can't hold) or {"ok": false, "why": "..."}.
- The model of a clip ("model" in POST /api/generate, the form and JSON): sol-h3 (Sol-H3, NVIDIA's two-stage
  pipeline above; the default: a request without a model and a record without one mean sol-h3) or, experimental and on
  the DGX Spark only with LFSTUDIO_MODELS=sol-h3,lightx2v-turbo, lightx2v-turbo (LightX2V Turbo: the full MiniMax-H3 in
  one stage with LightX2V's turbo LoRA). MODELS is the registry: per model its name, how it runs, its options
  (lightx2v-turbo: "resolution" 576p, the default, or 768p; "steps" 4) and its capability table: the rows of what it
  makes now, and why not yet for the rest of what every model is meant to make (END_STATE, Sol-H3's tasks, keyframes,
  lengths and formats). lightx2v-turbo makes t2va, and fl2va with a first frame, a last frame or both, at 5, 6, 7, 8,
  10, 12 or 15 s, 576p or 768p, 16:9 or 9:16, now (not ref2va). Its clip with a last frame keeps every frame its job
  makes, so that it ends on the frame the last keyframe is anchored to (124 frames at 5 s, 5.17 s; the studio's 121
  otherwise). A 15 s clip is 362 frames made, past diffusers 0.40's 15.0 s: its job raises that limit for that clip
  (jobs/h3turbo_job.py, "Lengths"). A model that is off or unknown, an option of another model, a value of an option it
  does not have, or a clip outside its table gets 400 with the reason; so does a prompt with MiniMax's tokens for
  lightx2v-turbo with SOL_H3_PROMPT_TOKENS=plain. With more than one model on, /api/state has "models" (public_models:
  the enabled models' tables, options and expected times) and "session" "model", the page gets the same table in a
  <script id="lf-models">, and the records in /api/state, /api/jobs and the API have "model" and "file_name" (the name
  /video/<id> downloads as).
  POST /api/load takes "model": a LightX2V Turbo clip loads its model itself (409). A LightX2V Turbo clip is a job of
  its own in the same queue, in order with every other clip (run_model_job): it closes the Sol-H3 session and stops a
  warm-up (status "freeing memory"), refuses to start below its memory floor (its job's min_free_for: the larger of
  LFSTUDIO_H3TURBO_MIN_FREE_GIB and what a clip of its size and length needs to stay above the job's own stop: 90-100
  GiB, 103 at 768p 12 s and 107 at 768p 15 s; 0 = no check), encodes the prompt with a transient Qwen worker
  (LFSTUDIO_H3TURBO_ENCODE_SCRIPT, status "encoding the prompt"; for fl2va also the prompt without its keyframe
  references, "text_only", for a later Upscale: null when nothing is left of it, a prompt of keyframe lines alone), then
  runs its job (LFSTUDIO_H3TURBO_SCRIPT with LFSTUDIO_H3TURBO_PYTHON; status "loading models + generating", "generating"
  from its first step, "job_phase" from its "[lf-studio] phase" lines) in a process group of its own, both under one
  pipeline_call, so Cancel stops either; LFSTUDIO_H3TURBO_GUARD_GIB guards it. No retry, no warm-up afterwards (nor at
  start after one). Its record has "model", "model_opts" (the options, the size, the LoRA, the Qwen canvas and its image
  tokens) and the job's times and memory; its files go into
  <session>-lightx2v-turbo-<resolution>-<task>-<format>-<seconds>s/<case_id>/ (qwen/, qwen-text/, turbo/), its name has
  the model and the options: lf-studio_lightx2v-turbo-576p_<task>_<seconds>s_seed<seed>_<id6>.mp4. It can be
  upscaled (above), also with the model turned off since. With LFSTUDIO_MODELS unset (sol-h3 only) everything is as
  before: no "models", no new keys in the records or the session, the page as it is; "model": "sol-h3" is taken,
  another model is refused.

For agents and scripts (#29; docs/API.md):
- POST /api/generate takes the page's multipart form or a JSON object (json_form): the same fields, files as base64
  ({"name", "data"} or a data: URI); the studio never fetches URLs. Without a task, it follows from the files (also
  for the form). Answer: {"id", "job"}, job being the API record (api_record). 411 without Content-Length (chunked),
  413 over 200 MB, 415 for another Content-Type, 429 (Retry-After: 60) while LFSTUDIO_MAX_QUEUED clips are waiting.
  fl2va and ref2va records keep "labels": the <Picture N>, <Video N> or <Audio N> each file is in the prompt.
- GET /api/jobs/<id>[?wait=S]: one clip as an API record: the stored record plus state (queued, running, done, failed,
  cancelled), phase, ahead, elapsed_s and urls. With wait (seconds, at most 300) it answers when the clip has
  finished (and is saved), or when S has passed. 404 "no such clip"; 503 with the record if the studio shuts down
  meanwhile (after the restart the clip reads failed, "server restarted").
- GET /api/jobs/<id>/video[?wait=S]: the same wait, then the MP4 as /video/<id> sends it; 202 with the record if the
  clip is still queued or running when S has passed, 409 with it for a failed or cancelled clip.
- /video/<id> downloads as lf-studio_<task>_<seconds>s_seed<seed>_<id6>.mp4.
- The small JSON POSTs (/api/cancel, /api/delete, /api/undelete, /api/load, /api/unload, /api/upscale) answer 413
  above 64 KiB.
- LFSTUDIO_API=0 turns this API off (#58): a JSON submit, a submit with an X-LFStudio-Client header (bin/lf-cli sends
  its form with one) and GET or HEAD /api/jobs/<id>[/video] answer 403 (API_OFF). The page uses none of them: its form
  submit, /api/state ("api": false), /api/jobs?before=, the files and the small POSTs work as before. Not access
  control: a script can still send the page's form.

Run it with cwd = the patched Sol-H3-Spark package (models/minimax_h3/Sol-H3-Spark in the patched Sana checkout), or
Sol-H3-RTX5090 with LFSTUDIO_BACKEND=rtx5090: it imports runtime.config and runtime.pipeline from there. Both packages
have the same runtime API, so the server does the same on both. Standard library only; this process needs no torch.
bin/lf-studio does all of this for you.

Environment:
  LFSTUDIO_BACKEND           spark (default) or rtx5090: the machine. It changes only what /api/state reports
                             ("backend") and so what the page says (the machine's name, its clip-time estimates);
                             any other value stops the server. bin/lf-studio picks the package and checks by it.
  SOL_H3_SPARK_RUNTIME_ROOT  required. Uploads and session outputs go under it (the Qwen container sees only it).
  LFSTUDIO_HOME              default ~/sol-h3. Only used for the default manifest path.
  LFSTUDIO_PATHS             manifest path with {task}; default $LFSTUDIO_HOME/paths-{task}.json
  LFSTUDIO_STATE_DIR         jobs.db (every clip; jobs.json from before 0.2 is imported once) and
                             load-times.json; default ~/.local/state/lf-studio
  LFSTUDIO_PUBLISH_DIR       optional: also copy each finished clip (and its poster) here
  LFSTUDIO_HOST, LFSTUDIO_PORT
                             default 127.0.0.1 and 8190. 0.0.0.0 serves the whole LAN; there is no login.
  LFSTUDIO_ALLOWED_HOSTS     more host names the studio answers to: a comma list (ports are ignored), * = any.
                             Always answered: IP addresses, localhost, names without a dot (Tailscale MagicDNS) and
                             names ending in .ts.net, .local, .lan, .home, .home.arpa or .internal. A request for any
                             other name (Host, or X-Forwarded-Host when present) gets 421, so a web page can't reach
                             the studio by DNS rebinding (a name of its own that resolves to the studio's address).
  LFSTUDIO_PRELOAD           1 (default): at start, warm up the settings of the last finished clip; with the
                             experimental Upscale on, also after each upscale (which closes the models) when nothing
                             is queued. 0: off.
  LFSTUDIO_RESHAPE           1 (default): a clip with the loaded task but another length or orientation keeps the
                             models loaded and changes the session's shape. 0: close the session and load the models
                             again, as before 0.4. A patch without Pipeline.reshape always reloads.
  LFSTUDIO_TRASH_DAYS        days a deleted clip stays in $SOL_H3_SPARK_RUNTIME_ROOT/trash before it is removed
                             for good (default 7)
  LFSTUDIO_MAX_QUEUED        clips that may wait in the queue (default 20; the running one does not count). Another
                             submit gets 429 until one has started.
  LFSTUDIO_API               1 (default, as in 0.4): the API for programs above is on. 0: off, the page only (403 for
                             what it does not use). Any other value stops the server.
  LFSTUDIO_LAUNCHER_PID      set by bin/lf-studio: the server stops (closing its pipeline) when that process dies.
  LFSTUDIO_UPSCALE           0 (default): off. 1: the experimental Upscale above, on the DGX Spark only. Any other
                             value, or 1 with LFSTUDIO_BACKEND=rtx5090, stops the server; so does a bad value of the
                             settings below while it is on (upscale/regen2k_job.py studio_settings, which
                             bin/lf-studio --check uses too).
  LFSTUDIO_UPSCALE_PYTHON    the Python of the Open-H3-Regenerate-2K environment (default
                             $LFSTUDIO_HOME/envs/regen2k/bin/python)
  LFSTUDIO_UPSCALE_PKG       NVIDIA's package, models/Open-H3-Regenerate-2K of NVlabs/Sana (default
                             $LFSTUDIO_HOME/Open-H3-Regenerate-2K)
  LFSTUDIO_UPSCALE_WEIGHTS   its weights (default $LFSTUDIO_CHECKPOINTS/Open-H3-Regenerate-2K; LFSTUDIO_CHECKPOINTS
                             defaults to $LFSTUDIO_HOME/checkpoints)
  LFSTUDIO_UPSCALE_BASE      the MiniMax-H3 base model (default $LFSTUDIO_CHECKPOINTS/MiniMax-H3)
  LFSTUDIO_UPSCALE_STRENGTH  how much an upscale may redraw, when the request does not say: faithful (start noise
                             0.5, 2 steps), balanced (0.7, 2 steps; the default) or full (NVIDIA's 3 steps from 0.909,
                             which redraws faces). Provisional, until measured.
  LFSTUDIO_UPSCALE_AUDIO     what the picture is refined against, when the request does not say: clip (the clip's own
                             Stage-1 audio; the default, an experiment) or silence (NVIDIA's). The clip's sound is
                             added to the result either way.
  LFSTUDIO_UPSCALE_LOAD      how the job loads its weights: fast (the default: upscale/fastload.py, O_DIRECT reads
                             straight to the GPU; 21.5-21.9 s in the job's clean runs) or standard
                             (NVIDIA's own loading through mmap, ~7.5 min; the fallback). The job inherits the patch's
                             SPARK_FAST_SAFETENSORS and SPARK_DIRECT_IO: either one off and the fast load reads every
                             file through mmap (the start log warns).
  LFSTUDIO_UPSCALE_PREREAD   0 (default): off. 1: the job reads its weights into the page cache in parallel before
                             it loads them (not shown to shorten the load yet). Needs LFSTUDIO_UPSCALE_LOAD=standard.
  LFSTUDIO_UPSCALE_MIN_FREE_GIB  MemAvailable an upscale of a 5 s clip needs once the session is closed (default 110;
                             5 more at 4K); a longer clip needs more where its case does (regen2k_job.min_free_for;
                             0 = no check)
  LFSTUDIO_UPSCALE_GUARD_GIB stop an upscale when MemAvailable stays below this for 3 s (default 5; 0 = no guard)
  LFSTUDIO_MODELS            sol-h3 (default): the models a clip may have, a comma list with sol-h3 in it.
                             sol-h3,lightx2v-turbo: also the experimental LightX2V Turbo above, on the DGX Spark only.
                             An unknown model, a list without sol-h3, or lightx2v-turbo with LFSTUDIO_BACKEND=rtx5090
                             stops the server; so does a bad value of the settings below while it is on (its job's
                             studio_settings, which bin/lf-studio --check uses too). They are not read while it is off.
  LFSTUDIO_H3TURBO_PYTHON    the Python of LightX2V Turbo's job (default $LFSTUDIO_HOME/envs/regen2k/bin/python, the
                             Upscale's environment)
  LFSTUDIO_H3TURBO_BASE      the MiniMax-H3 base model (default $LFSTUDIO_CHECKPOINTS/MiniMax-H3)
  LFSTUDIO_H3TURBO_LORAS     the folder of LightX2V's turbo LoRA files (default
                             $LFSTUDIO_CHECKPOINTS/Minimax-h3-Turbo)
  LFSTUDIO_H3TURBO_QWEN_CANVAS  where the prompt encoder sees the keyframes: draft (the default: the 384p draft canvas,
                             252 image tokens per keyframe) or canvas (the clip's own canvas: 576 or 1008 tokens,
                             steps 1-3 % longer; the GPU validation found its frame 0 the same as draft's)
  LFSTUDIO_H3TURBO_MIN_FREE_GIB  MemAvailable a LightX2V Turbo clip needs once the Sol-H3 session is closed (default
                             100; 0 = no check). A minimum: a clip needs at least what its job's min_free_for says one
                             of its size and length takes to stay above the job's 6 GiB stop (90 GiB at 576p 5 s, 100
                             at 768p 10 s, 103 at 768p 12 s and 107 at 768p 15 s): a value below that does not lower
                             that clip's floor, only 0 turns the check off.
  LFSTUDIO_H3TURBO_GUARD_GIB stop its job when MemAvailable stays below this for 3 s (default 5; 0 = no guard)
  Test hooks: LFSTUDIO_UPSCALE_SCRIPT (the job script, default upscale/regen2k_job.py), LFSTUDIO_H3TURBO_SCRIPT
  (LightX2V Turbo's job, which the server loads, for its model id, settings, presets, memory floor and job.json check,
  and runs; default jobs/h3turbo_job.py), LFSTUDIO_H3TURBO_ENCODE_SCRIPT (its prompt encode, run with this server's
  Python in the package; default bench/h3turbo_encode.py) and LFSTUDIO_TEST_MEMINFO (a file read for MemAvailable
  instead of /proc/meminfo; with the Upscale or LightX2V Turbo on, the start log warns when it is set).
Until 0.6 these settings may also have their names up to 0.4 (SOLH3_HOST for LFSTUDIO_HOST, and so on; not
LFSTUDIO_LAUNCHER_PID, LFSTUDIO_MODELS or the LFSTUDIO_UPSCALE and LFSTUDIO_H3TURBO ones, which are new): an old name
counts where the new one is not set, with one warning. The default state directory is
~/.local/state/sol-h3-studio while that has jobs.db and ~/.local/state/lf-studio does not.

SOL_H3_STAGE2_AUDIO (frozen or joint), SOL_H3_PROMPT_TOKENS (minimax or plain), SOL_H3_STATIC_SHAPES (1 or 0),
SOL_H3_DYNAMIC_TEXT_LENGTH (1 or 0; 1, the default since 0.4, compiles Stage 1's body blocks once per session for any
prompt length, 0 once per prompt length) and SOL_H3_STAGE2_KEYFRAME (pin, off or draft; pin, the default since 0.5.2
on both backends, holds a Keyframes clip's uploaded first frame in Stage 2, #57; off, as before, lets Stage 2 repaint
it) are patch settings (see docs/PATCH.md), checked at start: another value stops the server with a message, here and
not in every Stage-1 load. So does a value, set or the default, that the applied patch is too old to give. 0.3's
patch has none of SOL_H3_PROMPT_TOKENS, SOL_H3_STATIC_SHAPES and SOL_H3_DYNAMIC_TEXT_LENGTH: it tokenizes as plain and
compiles as SOL_H3_STATIC_SHAPES=0 and SOL_H3_DYNAMIC_TEXT_LENGTH=0 do, so the defaults stop the server until
install/10-sana.sh has applied the current patch. Patches before 0.5.2 (DGX Spark or RTX 5090) never pin, so the
default stops the server too, until install/10-sana.sh (or SOL_H3_STAGE2_KEYFRAME=off).
SOL_H3_SECONDS and SOL_H3_ORIENTATION are infer.py defaults. The studio passes length and format with every job, so
the server removes both from its environment at start; otherwise a 5 s job would render at SOL_H3_SECONDS.
"""
import base64
import binascii
import contextlib
import email.parser
import email.policy
import errno
import functools
import hashlib
import importlib.util
import ipaddress
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

TASKS = ("t2va", "fl2va", "ref2va")
# seconds; 5 = NVIDIA's frozen route, others = lf-studio extension. 3 and 4 s went after 0.5.0 (3 s clips never came
# out right; the patch still takes 3-15 s, for infer.py and the bench).
DURATIONS = (5, 6, 7, 8, 10, 12, 15)


def fail(message):
    print(f"lf-studio: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


# Until 0.6: the settings' names up to 0.4, SOLH3_X for LFSTUDIO_X (the docstring)
OLD_SETTINGS = ("BACKEND", "HOME", "PATHS", "STATE_DIR", "PUBLISH_DIR", "HOST", "PORT", "ALLOWED_HOSTS", "PRELOAD",
                "RESHAPE", "TRASH_DAYS", "MAX_QUEUED")


def old_settings():
    """Until 0.6: an old name counts where its new name is not set (the new one wins), with one warning on stderr
    that lists the old names used."""
    used = []
    for name in OLD_SETTINGS:
        old, new = "SOLH3_" + name, "LFSTUDIO_" + name
        if old not in os.environ:
            continue
        value = os.environ.pop(old)
        if new in os.environ:
            used.append(f"{old} ({new} wins)")
        else:
            os.environ[new] = value
            used.append(old)
    if used:
        print(f"lf-studio: settings under their 0.4 names, which work until 0.6: {', '.join(used)}. Rename them "
              "LFSTUDIO_*", file=sys.stderr, flush=True)


old_settings()

# infer.py defaults (see the docstring); the studio sets length and format per job
IGNORED_ENV = {name: os.environ.pop(name) for name in ("SOL_H3_SECONDS", "SOL_H3_ORIENTATION") if name in os.environ}

# the machine (docs/RTX5090.md): only the page's texts and /api/state depend on it
BACKENDS = {"spark": "Sol-H3-Spark", "rtx5090": "Sol-H3-RTX5090"}
BACKEND = os.environ.get("LFSTUDIO_BACKEND", "").strip() or "spark"
if BACKEND not in BACKENDS:
    fail(f"LFSTUDIO_BACKEND={BACKEND}: must be spark or rtx5090 (see bin/env.example)")

PKG = Path.cwd()
sys.path.insert(0, str(PKG))
try:
    from runtime.config import load_paths, normalize_case  # noqa: E402
    from runtime.pipeline import Pipeline  # noqa: E402
    from runtime.geometry import ORIENTATIONS  # noqa: E402  (exists only in the patched package)
except ImportError as error:
    fail(f"cannot import the patched {BACKENDS[BACKEND]} runtime from {PKG} ({error}). Start the server with the "
         f"working directory set to models/minimax_h3/{BACKENDS[BACKEND]} inside the patched Sana checkout.")

# the patch's settings: (name, its check in runtime/config.py, default, what a patch without the setting does, the
# version that added it). A bad value stops the studio here, not every clip. An older patch lacks some of them and
# would ignore them: any other value, set or the default, stops the studio too. SOL_H3_STAGE2_KEYFRAME is pin by
# default on both backends (#57); with an older patch the default stops the studio like the 0.4 ones, and the message
# names off, the way to start without the pin.
import runtime.config  # noqa: E402
for setting, name, default, older, since in (("SOL_H3_STAGE2_AUDIO", "stage2_audio_mode", "frozen", "joint", "0.3"),
                                             ("SOL_H3_PROMPT_TOKENS", "prompt_tokens_mode", "minimax", "plain", "0.4"),
                                             ("SOL_H3_STATIC_SHAPES", "compile_dynamic", "1", "0", "0.4"),
                                             ("SOL_H3_DYNAMIC_TEXT_LENGTH", "dynamic_text_length", "1", "0", "0.4"),
                                             ("SOL_H3_STAGE2_KEYFRAME", "stage2_keyframe_mode", "pin", "off", "0.5.2")):
    setting_check, value = getattr(runtime.config, name, None), os.environ.get(setting, "")
    if setting_check is None and (value.strip().lower() or default) != older:
        asked = f"{setting}={value}" if value.strip() else f"{setting}={default} (the default since {since})"
        opt_out = f" (or set {setting}=off)" if setting == "SOL_H3_STAGE2_KEYFRAME" and not value.strip() else ""
        fail(f"{asked} needs a newer patch than the one applied in {PKG}; run install/10-sana.sh{opt_out}")
    try:
        setting_check and setting_check()
    except ValueError as error:
        fail(str(error))
# the RTX 5090 patch also checks its own folders, switches and numbers (a list of problems; none = fine)
problems = getattr(runtime.config, "check_settings", list)()
if problems:
    fail("; ".join(problems))

if not os.environ.get("SOL_H3_SPARK_RUNTIME_ROOT"):
    fail("SOL_H3_SPARK_RUNTIME_ROOT is not set. Point it at your runtime directory "
         "(the Qwen container mounts it; see bin/env.example).")
RUNTIME = Path(os.environ["SOL_H3_SPARK_RUNTIME_ROOT"]).expanduser()
if not RUNTIME.is_absolute():
    fail(f"SOL_H3_SPARK_RUNTIME_ROOT must be an absolute path, not {RUNTIME}")

HOME = Path(os.environ.get("LFSTUDIO_HOME") or Path.home() / "sol-h3").expanduser()
PATHS_TEMPLATE = os.environ.get("LFSTUDIO_PATHS") or str(HOME / "paths-{task}.json")
if "{task}" not in PATHS_TEMPLATE:
    fail(f"LFSTUDIO_PATHS must contain {{task}}, e.g. {HOME}/paths-{{task}}.json (got {PATHS_TEMPLATE})")
PATHS = {t: Path(PATHS_TEMPLATE.format(task=t)).expanduser() for t in TASKS}
STATE_DIR = Path(os.environ.get("LFSTUDIO_STATE_DIR") or Path.home() / ".local/state/lf-studio").expanduser()
OLD_STATE_DIR = Path.home() / ".local/state/sol-h3-studio"   # until 0.6: 0.4's default, while it has the clip history
if (not os.environ.get("LFSTUDIO_STATE_DIR") and not (STATE_DIR / "jobs.db").exists()
        and (OLD_STATE_DIR / "jobs.db").exists()):
    print(f"lf-studio: using {OLD_STATE_DIR}: {STATE_DIR} has no jobs.db (the old place works until 0.6)",
          file=sys.stderr, flush=True)
    STATE_DIR = OLD_STATE_DIR
DB_FILE = STATE_DIR / "jobs.db"
OLD_STATE_FILE = STATE_DIR / "jobs.json"         # before 0.2: the newest 200 jobs; imported once
LOAD_TIMES = STATE_DIR / "load-times.json"
PUBLISH = Path(os.environ["LFSTUDIO_PUBLISH_DIR"]).expanduser() if os.environ.get("LFSTUDIO_PUBLISH_DIR") else None
HOST = os.environ.get("LFSTUDIO_HOST") or "127.0.0.1"
PORT = int(os.environ.get("LFSTUDIO_PORT") or "8190")
PRELOAD = os.environ.get("LFSTUDIO_PRELOAD", "1") == "1"
RESHAPE = os.environ.get("LFSTUDIO_RESHAPE", "1") == "1" and hasattr(Pipeline, "reshape")   # an older patch reloads


def host_name(value):
    """The host name in a Host header value: lower case, without the port or a final dot ([v6]:port -> v6)."""
    value = value.strip().lower()
    if value.startswith("["):
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:                        # name:port (a bare IPv6 address has more colons)
        value = value.split(":", 1)[0]
    return value.rstrip(".")


ALLOWED_HOSTS = {host_name(h) for h in (os.environ.get("LFSTUDIO_ALLOWED_HOSTS") or "").split(",") if h.strip()}
LOCAL_SUFFIXES = (".ts.net", ".local", ".lan", ".home", ".home.arpa", ".internal")


def known_host(name):
    """Whether the studio answers to this host name (host_name): see LFSTUDIO_ALLOWED_HOSTS in the docstring. A name
    without a dot also covers a request without Host (HTTP/1.0), which no browser sends."""
    if "*" in ALLOWED_HOSTS or name in ALLOWED_HOSTS or name == "localhost" or "." not in name:
        return True
    if name.endswith(LOCAL_SUFFIXES):
        return True
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False

INPUTS = RUNTIME / "inputs" / "webui"
SESSIONS = RUNTIME / "outputs" / "webui"
TRASH = RUNTIME / "trash"        # deleted clips, one directory per job, until LFSTUDIO_TRASH_DAYS have passed
try:
    TRASH_DAYS = float(os.environ.get("LFSTUDIO_TRASH_DAYS") or 7)
    assert math.isfinite(TRASH_DAYS) and TRASH_DAYS >= 0
except (ValueError, AssertionError):
    fail(f"LFSTUDIO_TRASH_DAYS must be a number of days, 0 or more (got {os.environ.get('LFSTUDIO_TRASH_DAYS')!r})")
try:
    MAX_QUEUED = int(os.environ.get("LFSTUDIO_MAX_QUEUED") or 20)
    assert MAX_QUEUED >= 1
except (ValueError, AssertionError):
    fail(f"LFSTUDIO_MAX_QUEUED must be a whole number, 1 or more (got {os.environ.get('LFSTUDIO_MAX_QUEUED')!r})")
# the API for programs (#58; the docstring). Off: a JSON submit, a program's form submit (API_CLIENT) and
# /api/jobs/<id>[/video] get 403 API_OFF. bin/lf-studio checks the value with the same words.
API_SETTING = os.environ.get("LFSTUDIO_API") or "1"
if API_SETTING not in ("0", "1"):
    fail(f"LFSTUDIO_API={API_SETTING}: must be 1 (on, the default) or 0 (off) (see bin/env.example)")
API = API_SETTING == "1"
API_OFF = {"error": "the API for programs is off on this studio (LFSTUDIO_API=0)"}
API_CLIENT = "X-LFStudio-Client"    # a program's form submit (bin/lf-cli): refused with the API off, as JSON is
# the experimental Upscale (the docstring): off unless LFSTUDIO_UPSCALE=1. bin/lf-studio checks the same, in the same
# words; the other LFSTUDIO_UPSCALE_* settings are read and checked by upscale/regen2k_job.py's studio_settings.
UPSCALE_SETTING = os.environ.get("LFSTUDIO_UPSCALE") or "0"
if UPSCALE_SETTING not in ("0", "1"):
    fail(f"LFSTUDIO_UPSCALE={UPSCALE_SETTING}: must be 0 (off, the default) or 1 (on) (see bin/env.example)")
UPSCALE = UPSCALE_SETTING == "1"
UPSCALE_JOB = Path(__file__).resolve().parents[1] / "upscale" / "regen2k_job.py"
regen = None                     # upscale/regen2k_job.py (standard library only), with the Upscale on
UPSCALE_CONF = {}                # its studio_settings: python, package, weights, base_model, strength, audio, ...
UPSCALE_SCRIPT = Path(os.environ.get("LFSTUDIO_UPSCALE_SCRIPT") or UPSCALE_JOB).expanduser()     # test hook
MEMINFO = os.environ.get("LFSTUDIO_TEST_MEMINFO") or "/proc/meminfo"                             # test hook
if UPSCALE:
    if BACKEND != "spark":
        fail(f"LFSTUDIO_UPSCALE=1: the experimental Upscale runs on the DGX Spark only, not with "
             f"LFSTUDIO_BACKEND={BACKEND} (Open-H3-Regenerate-2K needs ~100 GiB of GPU memory)")
    try:
        spec = importlib.util.spec_from_file_location("lf_studio_regen2k_job", UPSCALE_JOB)
        regen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regen)
    except (OSError, ImportError, SyntaxError) as error:
        fail(f"LFSTUDIO_UPSCALE=1: cannot read {UPSCALE_JOB} ({error})")
    UPSCALE_CONF, problems = regen.studio_settings(os.environ)
    if problems:
        fail("; ".join(problems))

# ------------------------------------------------------------------------------------------ the models (the docstring)
# MODELS is the registry the server checks requests against and that the page, the API and the clients read
# (public_models). Each model's capability table: "offers", the rows of what it makes now (a clip is offered when one
# row has each of its values: its task, for fl2va its keyframes, its length, its format and the model's options), and
# "not_yet", why it does not make the rest of END_STATE (what every model is meant to make in the end: Sol-H3's) or of
# its options' choices yet. Offering more is a change of this data, after a GPU validation; the page and the clients
# follow it. A model that "runs" a "job" is a GPU job of its own per clip (run_model_job); "setting" names its
# settings, LFSTUDIO_<setting>_*. "upscale": the finished clips of the model the experimental Upscale takes, by task and
# length (upscalable; every format the job has a shape for): every clip each model makes, of every length (not Sol-H3's
# old 3 and 4 s ones); which sizes a length gets is the job's table (upscale/regen2k_job.py offer_row, /api/state
# upscale_offers).
SOL_H3, TURBO = "sol-h3", "lightx2v-turbo"
DEFAULT_MODEL = SOL_H3           # a request without a model, and every record without one (all clips before 0.6)
FRAME_SETS = ("first", "last", "first+last")      # an fl2va clip's keyframes
END_STATE = {"task": TASKS, "frames": FRAME_SETS, "seconds": DURATIONS, "orientation": tuple(ORIENTATIONS)}
MODELS = {
    SOL_H3: {
        "name": "Sol-H3", "about": "NVIDIA's Sol two-stage pipeline: a MiniMax-H3 draft at 384p refined by LTX-2.5",
        "runs": "session", "experimental": False, "backends": tuple(BACKENDS), "options": (),
        "offers": (dict(END_STATE),), "not_yet": (),
        "upscale": {"tasks": TASKS, "seconds": DURATIONS}},
    TURBO: {
        "name": "LightX2V Turbo", "setting": "H3TURBO",
        "about": "the full MiniMax-H3 in one stage with LightX2V's turbo LoRA: keeps the look of your keyframe",
        "runs": "job", "experimental": True, "backends": ("spark",),
        # "tag": the option is in the clip's file name always (else only when it is not the default), with "unit"
        "options": (
            {"name": "resolution", "label": "Size", "type": "string", "default": "576p", "tag": "always",
             "choices": ({"value": "576p", "size": {"landscape": [1024, 576], "portrait": [576, 1024]}},
                         {"value": "768p", "size": {"landscape": [1344, 768], "portrait": [768, 1344]}})},
            {"name": "steps", "label": "Steps", "type": "integer", "default": 4, "unit": "step",
             "choices": ({"value": 4},)}),
        # The GPU validation on a DGX Spark (2026-10-28/29) ran t2va, a first frame, a last frame alone and first +
        # last, at 576p (5, 10 and 14 s) and 768p (5 and 10 s); GPU runs on 2026-10-29 (t2va, one clip each, the job on
        # a conditioning of the studio's encode) of 576p 15 s and 768p 12 and 15 s came within its model of a clip (the
        # job's min_free_for and estimate_s), which the other lengths follow. 15 s is the studio's 362 frames, past
        # diffusers 0.40's stock 15.0 s: the job raises that for its clip (its docstring, "Lengths").
        "offers": ({"task": ("t2va", "fl2va"), "frames": FRAME_SETS, "seconds": DURATIONS,
                    "orientation": ("landscape", "portrait"), "resolution": ("576p", "768p"), "steps": (4,)},),
        "not_yet": (
            {"task": ("ref2va",), "why": "LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not "
                                        "Reference (ref2va): use Sol-H3"},),
        # seconds a clip takes, by its options and length, until one has been measured here (remember_job_time): the
        # job's estimate_s (the GPU validation's receipts: the prompt encode, the process and its load, 4 steps by the
        # rows of the clip's sequence, the decode by its frames and pixels; measured at 576p 5, 10, 14 and 15 s and
        # 768p 5, 10, 12 and 15 s, the rest estimated), to 10 s; +~7 s when the Sol-H3 session must close first.
        # tests/test_server.py TurboSeam checks them against the job, for every clip offered.
        "estimates_s": {"576p/4/5": 200, "576p/4/6": 250, "576p/4/7": 280, "576p/4/8": 310, "576p/4/10": 400,
                        "576p/4/12": 500, "576p/4/15": 660, "768p/4/5": 350, "768p/4/6": 470, "768p/4/7": 530,
                        "768p/4/8": 600, "768p/4/10": 820, "768p/4/12": 1070, "768p/4/15": 1440},
        # both sizes: a 576p clip is stretched to 768p for the Upscale's job (every frame), and one with a last
        # keyframe (every frame made: 124 at 5 s, 243 at 10 s) taken whole; a 5 s one is cut to its first 121 at 4K
        # (upscale_sources, build_input)
        "upscale": {"tasks": ("t2va", "fl2va"), "seconds": DURATIONS}},
}


def models_setting(value, backend):
    """LFSTUDIO_MODELS -> (the enabled models in the registry's order, problem or None). bin/lf-studio --check says
    the same in the same words."""
    asked = [m.strip() for m in value.split(",") if m.strip()] or [DEFAULT_MODEL]
    unknown = [m for m in asked if m not in MODELS]
    if unknown:
        return (), f"LFSTUDIO_MODELS={value}: unknown model {unknown[0]} (known: {', '.join(MODELS)})"
    if DEFAULT_MODEL not in asked:
        return (), f"LFSTUDIO_MODELS={value}: must include {DEFAULT_MODEL}, the studio's own model"
    for m in asked:
        if backend not in MODELS[m]["backends"]:
            return (), (f"LFSTUDIO_MODELS={value}: the experimental {MODELS[m]['name']} ({m}) runs on the DGX Spark "
                        f"only, not with LFSTUDIO_BACKEND={backend} (it needs ~100 GiB of GPU memory); set "
                        f"LFSTUDIO_MODELS={DEFAULT_MODEL}")
    return tuple(m for m in MODELS if m in asked), None


OPTION_KINDS = {"string": str, "integer": int}     # an option's "type" -> the type of its values (JSON's)


def registry_problems():
    """What is wrong with MODELS' tables (a bug; the server stops): an option of a type the requests can't carry (the
    form and JSON carry strings and integers: json_form, check_model, lf-cli), with a value or default not of its
    type or a default not among its choices, or of another type than an option of the same name of another model; a
    row without a dimension of the model or with a value it can't have, or a value not offered without a reason; an
    Upscale entry with a task or a length the studio does not have."""
    found, kinds = [], {}
    for mid, m in MODELS.items():
        for o in m["options"]:
            kind = OPTION_KINDS.get(o.get("type"))
            if kind is None:
                found.append(f"{mid}: option {o['name']} has type {o.get('type')!r}, not "
                             f"{' or '.join(OPTION_KINDS)}")
                continue
            values = [c["value"] for c in o["choices"]]
            found += [f"{mid}: option {o['name']}: {v!r} is not of its type, {o['type']}"
                      for v in values + [o["default"]] if type(v) is not kind]
            if o["default"] not in values:
                found.append(f"{mid}: option {o['name']}'s default {o['default']!r} is not one of its choices")
            if kinds.setdefault(o["name"], o["type"]) != o["type"]:
                found.append(f"{mid}: option {o['name']} is of type {o['type']}, another model's of type "
                             f"{kinds[o['name']]}")
        dims = dict(END_STATE, **{o["name"]: tuple(c["value"] for c in o["choices"]) for o in m["options"]})
        for row in m["offers"]:
            if set(row) != set(dims):
                found.append(f"{mid}: a row has {sorted(row)}, not {sorted(dims)}")
                continue
            found += [f"{mid}: {d} {v} is not one of {dims[d]}" for d in dims for v in row[d] if v not in dims[d]]
        for d, values in dims.items():
            for v in values:
                offered = any(v in row.get(d, ()) for row in m["offers"])
                reasons = [n for n in m["not_yet"] if v in n.get(d, ())]
                if offered == bool(reasons):
                    found.append(f"{mid}: {d} {v} is {'offered and has' if offered else 'not offered and has no'} "
                                 "reason not yet")
        up = m.get("upscale", {"tasks": (), "seconds": ()})
        found += [f"{mid}: the Upscale's task {t} is not one of {TASKS}" for t in up["tasks"] if t not in TASKS]
        found += [f"{mid}: the Upscale's length {v} is not one of {DURATIONS}" for v in up["seconds"]
                  if v not in DURATIONS]
    return found


problems = registry_problems()
if problems:
    fail("the model registry (MODELS): " + "; ".join(problems))
ENABLED, problem = models_setting(os.environ.get("LFSTUDIO_MODELS") or DEFAULT_MODEL, BACKEND)
if problem:
    fail(f"{problem} (see bin/env.example)")
MULTI = len(ENABLED) > 1         # the model choice is on: /api/state "models", "model" in the records, the page's table
# the request fields of the enabled models' options (POST /api/generate), and their JSON types
OPTION_TYPES = {o["name"]: o["type"] for mid in ENABLED for o in MODELS[mid]["options"]}
# LightX2V Turbo (the docstring): its job, jobs/h3turbo_job.py, and its prompt encode, bench/h3turbo_encode.py;
# LFSTUDIO_H3TURBO_SCRIPT and LFSTUDIO_H3TURBO_ENCODE_SCRIPT (test hooks: tests/stub/h3turbo_job.py and
# tests/stub/h3turbo_encode.py) put others in their place. The job is loaded as the Upscale's is, for its
# standard-library part: MODEL (this registry's id), JOB_VERSION, PRESETS (steps -> LoRA file and shifts), RECEIPT,
# MINIMAX_PROMPT_TOKENS, studio_settings, min_free_for (the memory floor per size and length), parse_job,
# output_name, worker_command, worker_environment, parse_line and write_json (bin/lf-studio --check also calls its
# check_install).
REPO_ROOT = Path(__file__).resolve().parents[1]
H3TURBO_SCRIPT = Path(os.environ.get("LFSTUDIO_H3TURBO_SCRIPT") or REPO_ROOT / "jobs" / "h3turbo_job.py").expanduser()
H3TURBO_ENCODE = Path(os.environ.get("LFSTUDIO_H3TURBO_ENCODE_SCRIPT")
                      or REPO_ROOT / "bench" / "h3turbo_encode.py").expanduser()
H3TURBO_ABORT_GIB = 6            # job.json abort_below_gib: the job stops itself below this MemAvailable (exit 3)
JOB_MODULES = {}                 # an enabled job model -> its job's module
MODEL_CONF = {}                  # -> its studio_settings: python, base_model, loras, qwen_canvas, min_free_gib, ...
if TURBO in ENABLED:
    try:
        spec = importlib.util.spec_from_file_location("lf_studio_h3turbo_job", H3TURBO_SCRIPT)
        JOB_MODULES[TURBO] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(JOB_MODULES[TURBO])
    except (OSError, ImportError, SyntaxError) as error:
        fail(f"LFSTUDIO_MODELS has {TURBO}: cannot read its job {H3TURBO_SCRIPT} ({error})")
    if getattr(JOB_MODULES[TURBO], "MODEL", None) != TURBO:
        fail(f"{H3TURBO_SCRIPT} is the job of {getattr(JOB_MODULES[TURBO], 'MODEL', None)!r}, not of {TURBO}")
    MODEL_CONF[TURBO], problems = JOB_MODULES[TURBO].studio_settings(os.environ)
    if problems:
        fail("; ".join(problems) + " (see bin/env.example)")
    missing = [c["value"] for o in MODELS[TURBO]["options"] if o["name"] == "steps" for c in o["choices"]
               if c["value"] not in JOB_MODULES[TURBO].PRESETS]
    if missing:
        fail(f"{H3TURBO_SCRIPT} has no LoRA preset for {', '.join(map(str, missing))} steps, which MODELS offers")
    if not H3TURBO_ENCODE.is_file():
        fail(f"LFSTUDIO_MODELS has {TURBO}: no prompt encode at {H3TURBO_ENCODE}")
# The Upscale's text of a Keyframes or Reference clip (upscale_text): the prompt encode's own rewrites, text_only_prompt
# and ref_text_only_prompt (standard library on import), always from this repo's bench/h3turbo_encode.py; the encode
# itself runs LFSTUDIO_H3TURBO_ENCODE_SCRIPT (--upscale), with LightX2V Turbo on or off.
TEXT_REWRITES, REWRITES_FILE = None, REPO_ROOT / "bench" / "h3turbo_encode.py"
if UPSCALE:
    try:
        spec = importlib.util.spec_from_file_location("lf_studio_text_rewrites", REWRITES_FILE)
        TEXT_REWRITES = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TEXT_REWRITES)
    except (OSError, ImportError, SyntaxError) as error:
        fail(f"LFSTUDIO_UPSCALE=1: cannot read {REWRITES_FILE} ({error})")
    if not H3TURBO_ENCODE.is_file():
        fail(f"LFSTUDIO_UPSCALE=1: no prompt encode at {H3TURBO_ENCODE}")
for d in (INPUTS, SESSIONS, STATE_DIR) + ((PUBLISH,) if PUBLISH else ()):
    d.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()          # guards jobs / session
close_lock = threading.Lock()    # one pipeline close at a time (close_session)
wake = threading.Event()
stopping = threading.Event()     # set when the server shuts down: the worker starts nothing new
jobs = []                        # every job, oldest first (also in jobs.db)
PAGE = 24                        # clips per gallery page (/api/state carries the newest page, /api/jobs older ones)
FINISHED = ("done", "failed", "cancelled")       # a job's final statuses; any other is queued or running
session = {"pipeline": None, "key": None, "state": "idle", "since": time.time()}
work = {"job": None,             # the job whose pipeline call (start or generate) is running (pipeline_call)
        "process": None,         # a running GPU job's process (gpu_process: an upscale's job, a LightX2V Turbo
                                 # clip's prompt encode or its job), stopped at shutdown
        "labels": None}          # its names in the log: (for Cancel, for shutdown)
# Notified after every saved change of a job (save_job) and at shutdown: GET /api/jobs/<id>?wait= waits on it. Not
# `lock`, which the worker does not hold while it changes a job. `waiting` (guarded by it): the requests waiting.
changed = threading.Condition()
waiting = 0


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


db = None                        # sqlite3 connection to DB_FILE (open_db); writes are serialized by db_lock
db_lock = threading.Lock()


def as_time(value):
    """A job's 'created' as a finite float (0 for anything else, e.g. from a hand-edited jobs.json)."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return 0.0
    return t if math.isfinite(t) else 0.0


def open_db():
    """Open (or create) jobs.db and return every job, oldest first, with strictly increasing 'created' (the
    gallery's paging cursor). Imports jobs.json from before 0.2 once; it stays as it was (no longer written)."""
    global db
    db = sqlite3.connect(DB_FILE, check_same_thread=False, isolation_level=None)    # autocommit
    db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL NOT NULL, data TEXT NOT NULL)")
    # user_version 1: jobs.json has been imported (or there was none). Set in the import's own transaction, so a
    # failed or interrupted import is tried again at the next start.
    if db.execute("PRAGMA user_version").fetchone()[0] == 0:
        try:
            old = json.loads(OLD_STATE_FILE.read_text()) if OLD_STATE_FILE.exists() else []
            usable = [j for j in old if isinstance(j, dict) and re.fullmatch(r"[0-9a-f]{32}", str(j.get("id")))
                      and j.get("task") in TASKS and isinstance(j.get("prompt"), str)]
            if len(usable) < len(old):
                log(f"{OLD_STATE_FILE}: skipping {len(old) - len(usable)} entries without a valid id, task or prompt")
            old = usable
            for j in old:
                j["created"] = as_time(j.get("created"))
            rows = [(j["id"], j["created"], json.dumps(j)) for j in old]
            with db_lock:
                db.execute("BEGIN")
                try:
                    db.executemany("INSERT OR IGNORE INTO jobs VALUES (?, ?, ?)", rows)
                    db.execute("PRAGMA user_version = 1")
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
            if old:
                log(f"imported {len(old)} jobs from {OLD_STATE_FILE} into {DB_FILE}")
        except (OSError, ValueError, TypeError, sqlite3.Error) as error:
            log("could not import", OLD_STATE_FILE, f"({error}); trying again at the next start")
    loaded = [json.loads(row[0]) for row in db.execute("SELECT data FROM jobs ORDER BY created, rowid")]
    nudged, last = [], None
    for j in loaded:
        j.pop("moving", None)
        j["created"] = as_time(j.get("created"))
        if last is not None and j["created"] <= last:      # equal or out-of-order times: 1 ms apart, in load order
            j["created"] = last + 0.001
            nudged.append(j)
        last = j["created"]
    for j in nudged:
        save_job(j)
    return loaded


def save_job(job, strict=False):
    """Write one job to jobs.db (its current state). strict: raise sqlite3.Error instead of only logging it."""
    try:
        with db_lock:
            data = dict(job)                           # a consistent copy (the worker updates jobs without lock)
            data.pop("moving", None)                   # only while this process moves its files
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)", (job["id"], job["created"], json.dumps(data)))
    except sqlite3.Error:
        traceback.print_exc()
        if strict:
            raise
    finally:
        with changed:                                  # wakes GET /api/jobs/<id>?wait=, also after a failed write
            changed.notify_all()


def same_models(a, b):
    """Whether the models loaded for settings `a` (task, orientation, seconds) serve a clip with settings `b`: the same
    settings, or with LFSTUDIO_RESHAPE the same task (the session changes length and orientation in place)."""
    return a is not None and b is not None and (tuple(a) == tuple(b) or RESHAPE and a[0] == b[0])


def is_upscale(job):
    """Whether a job is an upscale of a clip (POST /api/upscale), not a clip of its own."""
    return job.get("kind") == "upscale"


def job_model(job):
    """A clip's model: its "model", or sol-h3 for a clip without one (every clip before the model choice)."""
    return job.get("model") or DEFAULT_MODEL


def warm_up_wanted():
    """A queued clip waits for the warm-up's models, so the warm-up is kept for it (an upscale or a clip of a job model
    closes them anyway). Call it holding `lock`."""
    return any(j["status"] == "queued" and not is_upscale(j) and job_model(j) == SOL_H3
               and same_models(session.get("loading_key"), (j["task"], j["orientation"], j.get("seconds", 5)))
               for j in jobs)


def find_job(jid):
    """The job with this id, unless it has been deleted."""
    return next((j for j in jobs if j["id"] == jid and not j.get("deleted")), None)


def in_sessions(case):
    """Whether `case` is a clip's own directory, <SESSIONS>/<session>/<case_id> (paths compared resolved: the
    pipeline records resolved paths, the runtime root may be reached through a symlink)."""
    try:
        return case.parent.parent.resolve() == SESSIONS.resolve()
    except OSError:
        return False


def under(path, root):
    """Whether `path` lies in `root` (both resolved)."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except OSError:
        return False


def trash_paths(job):
    """What deleting a job moves to the trash: the clip's own directory in its session (MP4, poster, intermediate
    files; for clips made before per-clip directories, the MP4 and its poster), the directories of runs that did not
    finish (case_dirs), its input directory, and its LFSTUDIO_PUBLISH_DIR copy with poster. Nothing another clip still
    uses (an old history can list the same file twice). Call with `lock` held."""
    found = [Path(d) for d in job.get("case_dirs") or [] if in_sessions(Path(d))]
    if job.get("output"):
        out = Path(job["output"])
        case = out.parent.parent                   # <session>/<case_id>/stage2/<clip>.mp4
        found += [case] if case.name == job.get("case_id") and in_sessions(case) else [out, out.with_suffix(".jpg")]
    found.append(INPUTS / job["id"])
    if job.get("published"):
        found += [Path(job["published"]), Path(job["published"]).with_suffix(".jpg")]
    others = [str(Path(j[k]).with_suffix(x)) for j in jobs if j is not job and not j.get("deleted")
              for k in ("output", "published") if j.get(k) for x in (".mp4", ".jpg")]
    unique = []
    for f in found:
        shared = any(o == str(f) or o.startswith(str(f) + "/") for o in others)
        if f.exists() and f not in unique and not shared:
            unique.append(f)
    return unique


def safe_move(src, dst):
    """Move a file or directory without ever leaving a partial copy at `dst`: a rename on the same filesystem; across
    filesystems (the publish folder can be a network share) a copy under a temporary name next to `dst`, renamed into
    place, then the source removed. If a file's source can't be removed, the copy is removed again and the move
    fails; for a directory removed only in part the move counts as done and it is logged. `dst` must not exist."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} is already there")
    try:
        os.rename(src, dst)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    tmp = dst.with_name(f".{dst.name}.part")
    if tmp.exists():                                   # from a copy that was interrupted
        shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
    try:
        if src.is_dir() and not src.is_symlink():
            shutil.copytree(src, tmp, symlinks=True)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
        raise
    if src.is_dir() and not src.is_symlink():
        try:
            shutil.rmtree(src)
        except OSError as error:
            log(f"moved {src} to {dst} but could not remove all of the original: {error}")
        return
    try:
        src.unlink()
    except OSError:
        dst.unlink(missing_ok=True)                    # not moved after all: one copy, where it was
        raise


moves_done = threading.Condition()    # Delete/Undo file moves in progress (moves_running); shutdown waits for them
moves_running = 0


class MoveError(OSError):
    """A move to or from the trash failed; `left` maps originals to what is still in the trash after rolling back."""
    def __init__(self, error, left):
        super().__init__(str(error))
        self.left = left


def trash_targets(jid, paths):
    """Where each of `paths` goes in TRASH/<jid>/: {original: target}, names not taken yet."""
    dest, targets = TRASH / jid, {}
    for i, src in enumerate(paths):
        target = dest / f"{i}-{src.name}"
        n = 1
        while target.exists():                         # left over from an earlier failed attempt
            target = dest / f"{i}-{n}-{src.name}"
            n += 1
        targets[str(src)] = str(target)
    return targets


def move_all(targets):
    """Move every original to its target. If one fails, move back what was moved and raise MoveError with what
    could not go back."""
    moved = {}
    try:
        for src, target in targets.items():
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            safe_move(src, target)
            moved[src] = target
    except OSError as error:
        left = {}
        for src, target in moved.items():
            try:
                safe_move(target, src)
            except OSError:
                traceback.print_exc()
                left[src] = target
        raise MoveError(error, left) from error


def delete_clip(job, paths):
    """Move a clip's files to the trash and mark it deleted. What is about to move is saved first (job['trashing']),
    so a restart after a crash in the middle puts the files back (recover_moves). Returns the HTTP reply."""
    jid, targets = job["id"], trash_targets(job["id"], paths)
    with lock:
        job["trashing"] = targets
    try:
        save_job(job, strict=True)
    except sqlite3.Error as error:
        with lock:
            job.pop("trashing", None)
        return 500, {"error": f"could not record the delete ({error}); nothing was moved"}
    try:
        move_all(targets)
    except MoveError as error:
        with lock:
            job.pop("trashing", None)
            if error.left:
                job["trashing"] = error.left          # still in the trash: put back at the next start
        save_job(job)
        with contextlib.suppress(OSError):
            (TRASH / jid).rmdir()
        return 500, {"error": f"could not move the clip to the trash: {error}"}
    with lock:
        job.pop("trashing", None)
        job.update(deleted=time.time(), trashed=targets)
    try:
        save_job(job, strict=True)
    except sqlite3.Error as error:                     # not recorded: put it all back
        left, _, _ = restore_files(targets, set(targets))
        with lock:
            job.pop("deleted", None)
            job.pop("trashed", None)
        return 500, {"error": f"could not record the delete ({error}); the clip is back"
                     + (f", except {len(left)} file(s) in {TRASH / jid}" if left else "")}
    log("deleted", job["case_id"], "to", TRASH / jid)
    return 200, {"ok": True}


def needed_files(job, trashed):
    """The trashed originals a clip can't do without: everything in the runtime root, and its MP4, which is the
    publish copy when the clip has no other (a clip from before per-clip directories, or one whose session
    directory was pruned). The other publish copies are best effort."""
    own = job.get("output")
    needed = {k for k in trashed if under(k, RUNTIME)} | ({own} if own else set())
    has_own = own and (own in trashed or any(own.startswith(k + "/") for k in trashed) or Path(own).is_file())
    if not has_own and job.get("published"):
        needed |= {job["published"]}
    return needed


def undelete_clip(job):
    """Move a deleted clip's files back (restore_files) and unmark it. Returns the HTTP reply."""
    trashed = dict(job.get("trashed") or {})
    left, copy_lost, error = restore_files(trashed, needed_files(job, trashed))
    with lock:
        if error:
            job["trashed"] = left                      # what is still there, for another try
        else:
            job.pop("deleted", None)
            job.pop("trashed", None)
            if copy_lost:
                job.pop("published", None)
    if not error:
        with contextlib.suppress(OSError):
            (TRASH / job["id"]).rmdir()                # empty unless a publish copy stayed
    try:
        save_job(job, strict=True)
    except sqlite3.Error as db_error:
        return 500, {"error": f"the files were moved back but the history could not be updated ({db_error}); "
                              "it may show as deleted again after a restart"}
    if error:
        return 500, {"error": f"could not restore the clip: {error}"}
    log("restored", job["case_id"])
    return 200, {"ok": True, "job": dict(job)}


def recover_moves():
    """At start: put back the files of a delete that did not finish (job['trashing'], e.g. the server was killed)."""
    for job in [j for j in jobs if j.get("trashing")]:
        left, _, error = restore_files(job["trashing"], set(job["trashing"]))
        if error:
            log("could not put back the files of an unfinished delete:", error)
            job["trashing"] = left
        else:
            job.pop("trashing", None)
            log("put back the files of an unfinished delete of", job.get("case_id"))
        save_job(job)


def trashed_path(job, path):
    """Where `path` (an input of `job`) is now: under TRASH/<id>/ while the job is deleted, else where it was."""
    for src, target in (job.get("trashed") or {}).items():
        if path == src or path.startswith(src + "/"):
            return target + path[len(src):]
    return path


def restore_files(trashed, needed):
    """Move trashed files back. `needed`: the originals the clip can't do without (its directory, inputs, and an old
    clip's only MP4, even if that is in the publish folder); any other is a LFSTUDIO_PUBLISH_DIR copy, best effort.
    Folders are created again only inside the runtime root, never on a share that is not mounted. Returns (what is
    still in the trash, whether a publish copy stayed there, the error that stopped a needed file or None). Runs
    without `lock`."""
    left, copy_lost = dict(trashed), False
    for src, target in trashed.items():
        if not Path(target).exists():                  # nothing to bring back
            del left[src]
            continue
        try:
            if Path(src).exists():
                raise FileExistsError(f"{src} is already there")
            if under(src, RUNTIME):
                Path(src).parent.mkdir(parents=True, exist_ok=True)
            elif not Path(src).parent.is_dir():
                raise FileNotFoundError(f"{Path(src).parent} is not there (an unmounted share?)")
            safe_move(target, src)
            del left[src]
        except OSError as error:
            if src in needed:
                return left, copy_lost, error
            log("could not put the publish copy back, it stays in the trash:", error)
            del left[src]                              # given up on; purge_trash sweeps it with the trash folder
            copy_lost = True
    return left, copy_lost, None


def purge_trash():
    """Remove clips deleted more than TRASH_DAYS ago, their files and their history entry, and anything else in
    TRASH older than that (left over from a failed delete or restore)."""
    cutoff = time.time() - TRASH_DAYS * 86400
    with lock:
        old = [j for j in jobs if j.get("deleted") and j["deleted"] < cutoff and not j.get("moving")]
        for j in old:
            jobs.remove(j)
        keep = {j["id"] for j in jobs if j.get("deleted") or j.get("moving") or j.get("trashing")}
    for j in old:
        shutil.rmtree(TRASH / j["id"], ignore_errors=True)
        try:
            with db_lock:
                db.execute("DELETE FROM jobs WHERE id = ?", (j["id"],))
        except sqlite3.Error:
            traceback.print_exc()
    swept = 0
    for d in TRASH.iterdir() if TRASH.is_dir() else []:
        with contextlib.suppress(OSError):
            if d.name not in keep and d.stat().st_mtime < cutoff:
                shutil.rmtree(d) if d.is_dir() else d.unlink()
                swept += 1
    if old or swept:
        log(f"emptied the trash: {len(old)} clip(s) deleted more than {TRASH_DAYS:g} days ago"
            + (f", {swept} leftover(s)" if swept else ""))


def ready_tasks():
    """Tasks whose path manifest exists (install/70-manifests.sh writes them)."""
    return [t for t in TASKS if PATHS[t].is_file()]


def load_eta(key):
    """Last measured cold start for this (task, orientation, seconds), else for any setting, else 4 min."""
    try:
        times = json.loads(LOAD_TIMES.read_text())
    except (OSError, ValueError):
        return 240
    return times.get("/".join(map(str, key))) or times.get("last") or 240


def warm_time(key):
    """Last measured clip time for this setting with the models already loaded, else None."""
    try:
        return json.loads(LOAD_TIMES.read_text()).get("warm/" + "/".join(map(str, key)))
    except (OSError, ValueError):
        return None


def remember_load(key, seconds, warm=False):
    """Record a cold start (load + one clip), or with warm=True a clip in a loaded session."""
    try:
        times = json.loads(LOAD_TIMES.read_text()) if LOAD_TIMES.exists() else {}
        if warm:
            times["warm/" + "/".join(map(str, key))] = seconds
        else:
            times["/".join(map(str, key))] = times["last"] = seconds
        LOAD_TIMES.write_text(json.dumps(times, indent=1))
    except (OSError, ValueError):
        traceback.print_exc()


def session_root(key, kind=""):
    """A new session directory: start time and settings, -2, -3, ... after it when another session started in the
    same second (a clip cancelled right away, then the next one; the pipeline refuses an existing directory)."""
    return new_session_dir(f"{kind}{key[0]}-{key[1]}-{key[2]}s")


def new_session_dir(name):
    """SESSIONS/<start time>-<name>, with -2, -3, ... after it when that is taken (not created)."""
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    root, n = SESSIONS / base, 2
    while root.exists():
        root, n = SESSIONS / f"{base}-{n}", n + 1
    return root


def new_pipeline(key, root):
    task, orientation, seconds = key
    return Pipeline(load_paths(PATHS[task], task=task), root, task=task,
                    orientation=orientation, seconds=None if seconds == 5 else seconds)   # 5 s = frozen route


def reshape_args(key):
    """Pipeline.reshape's arguments for settings `key`, as new_pipeline passes them (5 s = the frozen route)."""
    return {"orientation": key[1], "seconds": None if key[2] == 5 else key[2]}


def loaded_pipeline():
    """The session's pipeline, or None while it is being closed (Unload, a failed clip): a closing pipeline is not
    loaded any more, so nothing reshapes it or renders with it. Call it without holding `lock`."""
    with lock:
        p = session["pipeline"]
        return p if p is not None and session.get("state") != "unloading" and not close_lock.locked() else None


def reshape_session(key, reason):
    """Give the loaded session the length and orientation of `key` (same task), without a clip ("Load selected now").
    A failure closes the session, unless it failed because the session is being closed already (Unload meanwhile):
    then one line says so, and that close finishes the job. Call it without holding `lock`."""
    p, old = session["pipeline"], session["key"]
    log("reshaping session", old, "->", key, "-", reason)
    session.update(state="reshaping", since=time.time(), loading_key=key)
    try:
        p.reshape(**reshape_args(key))
    except Exception as e:
        with lock:
            closed = getattr(p, "closing", False) or session["pipeline"] is not p
        if closed:
            log("length/format change stopped: the session was closed meanwhile -", f"{type(e).__name__}: {e}")
            return
        traceback.print_exc()
        close_session("length/format change failed")
        return
    with lock:
        if session["pipeline"] is p:
            session.update(key=key, state="ready", since=time.time(), loading_key=None)


def close_session(reason, p=None):
    """Close the session's pipeline, or `p` (a pipeline of this thread's that may no longer be the session's).
    One close at a time: a second caller waits for the first, then finds nothing left to close. The session is reset
    only if it still holds the closed pipeline (or none), never a newer one. Call it without holding `lock`."""
    with close_lock:
        with lock:
            target = p or session["pipeline"]
            mine = target is not None and target is session["pipeline"]
            if mine:
                session["state"] = "unloading"
                session["since"] = time.time()
            key = session["key"] if mine else None
        if target is not None:
            log("closing session", key or "", "-", reason)
            try:
                target.close()
            except Exception:
                traceback.print_exc()
        with lock:
            if session["pipeline"] is None or session["pipeline"] is target:
                session.update(pipeline=None, key=None, state="idle", since=time.time(), loading_key=None,
                               preloading=False, takeover_until=None, load_eta=None, warm_s=None)


class Cancelled(Exception):
    """The job was cancelled (POST /api/cancel) after it had started."""


def pipeline_call(job, call, failure):
    """Run `call`, the pipeline's start() or generate() for `job`, so that Cancel can stop it: while it runs,
    work["job"] is the job, and stop_job SIGTERMs the workers of a cancelled one. If it fails, or Cancel came first,
    the session is closed (its workers may have been signalled) and it raises; Cancelled when Cancel came first, even
    if the clip finished just then: Cancel has answered that it is cancelled. Otherwise the result is the job's, and
    from here on Cancel answers 409 (job["saving"]). `failure`: the reason logged when it failed by itself."""
    with lock:
        work["job"] = job
    try:
        row = call()
        with lock:                         # atomic with Cancel: either it is cancelled or its clip is kept
            cancelled = job.get("cancel_requested")
            if not cancelled:
                job["saving"] = True
                work["job"] = None
        if cancelled:
            raise Cancelled("cancelled as it finished")
        return row
    except BaseException:
        close_session("clip cancelled" if job.get("cancel_requested") else failure)   # stop_job still signals
        raise                                                                        # workers started meanwhile
    finally:
        with lock:
            work["job"] = None


def run_job(job):
    key = (job["task"], job["orientation"], job.get("seconds", 5))
    case = {"case_id": job["case_id"], "prompt": job["prompt"], "seed": job["seed"], "task": job["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if job.get(k):
            case[k] = job[k]
    case = normalize_case(case)
    loaded = loaded_pipeline()             # None while an Unload or a failed clip closes the session
    if loaded is not None and session["key"] != key and same_models(session["key"], key):
        # same task, another length or orientation: the loaded session changes its shape, then renders the clip
        if job.get("cancel_requested"):    # nothing ran: the models stay loaded
            raise Cancelled("cancelled before it started")
        p, old = loaded, session["key"]
        job["status"] = "generating"
        job["phase_since"] = time.time()
        job["reshaped_from"] = list(old)
        job.setdefault("case_dirs", []).append(str(Path(p.root) / job["case_id"]))
        session["state"] = "busy"
        log("reshaping session", old, "->", key)

        def reshape_and_generate():
            record = p.reshape(**reshape_args(key))
            job["reshape_s"] = round(record.get("reshape_s", 0.0), 2)
            session["key"] = key           # the workers have the new shape; a failure from here closes the session
            return p.generate(case)
        row = pipeline_call(job, reshape_and_generate, "length/format change failed")
        session["state"] = "ready"         # not a warm time: the first clip at a shape compiles for it
    elif session["key"] != key or loaded is None:
        job.pop("reshaped_from", None)     # a retry after a failed length/format change loads the models
        job.pop("reshape_s", None)
        close_session(f"switching to {key}")   # (waits for a close in progress)
        if job.get("cancel_requested"):    # cancelled while the old models were closed: nothing of its own ran
            raise Cancelled("cancelled before it started")
        job["status"] = "loading models + generating"
        job["phase_since"] = time.time()
        job["load_eta"] = load_eta(key)
        session.update(state="loading", since=time.time(), loading_key=key)
        try:
            root = session_root(key)
            with lock:
                if stopping.is_set():
                    raise RuntimeError("the studio is shutting down")
            p = new_pipeline(key, root)
        except BaseException:
            close_session("could not start the pipeline")      # no pipeline: back to idle, not "loading"
            raise
        session["pipeline"] = p
        job.setdefault("case_dirs", []).append(str(Path(p.root) / job["case_id"]))   # for Delete, even if it fails
        t0 = time.time()
        # the loading pass renders this job (no throwaway warm-up)
        row = pipeline_call(job, lambda: p.start(case, deliver=True), "model load failed")
        job["cold_start_s"] = round(time.time() - t0, 1)   # load + this clip
        job["includes_model_load"] = True
        remember_load(key, job["cold_start_s"])
        session.update(key=key, state="ready", since=time.time(), loading_key=None)
    else:
        if job.get("cancel_requested"):    # nothing ran: the models stay loaded
            raise Cancelled("cancelled before it started")
        job["status"] = "generating"
        job["phase_since"] = time.time()
        job.setdefault("case_dirs", []).append(str(Path(session["pipeline"].root) / job["case_id"]))
        session["state"] = "busy"
        t0 = time.time()
        row = pipeline_call(job, lambda: session["pipeline"].generate(case), "request failed (worker died)")
        session["state"] = "ready"
        remember_load(key, round(time.time() - t0, 1), warm=True)
    src = Path(row["output"])
    make_poster(src)
    job.update(output=str(src), e2e_s=round(row["e2e_s"], 1), stage1_s=round(row.get("stage1_s", 0), 1),
               stage2_s=round(row.get("stage2_s", 0), 1))
    publish(job, src, publish_stem(job))


def model_tag(job):
    """What a clip's file names say of its model: "" for sol-h3 (its names are as before the model choice), else the
    model and its options, those marked "tag" always, the others when not the default: lightx2v-turbo-576p,
    lightx2v-turbo-768p-8step."""
    model = job_model(job)
    if model == SOL_H3 or is_upscale(job):
        return ""
    opts, parts = job.get("model_opts") or {}, [model]
    for option in MODELS.get(model, {}).get("options", ()):
        value = opts.get(option["name"])
        if value is not None and (option.get("tag") == "always" or value != option["default"]):
            parts.append(f"{value}{option.get('unit', '')}")
    return "-".join(parts)


def publish_stem(job):
    """The name of a clip's LFSTUDIO_PUBLISH_DIR copy, without .mp4: now, its settings, its model (model_tag), its
    seed and its id's start."""
    tag = model_tag(job)
    return (f"{time.strftime('%Y%m%d-%H%M%S')}_{job['task']}_{job['orientation']}_{job.get('seconds', 5)}s_"
            f"{tag + '_' if tag else ''}seed{job['seed']}_{job['id'][:6]}")


def publish(job, src, stem):
    """With LFSTUDIO_PUBLISH_DIR: copy a finished job's MP4 and poster there as <stem>.mp4 (<stem>-2.mp4, ... when
    that is taken: a second upscale of a clip) and record it in job["published"]. Best effort."""
    if not PUBLISH:
        return
    try:                                   # (a share can fail any call, the look for a free name too)
        dst, n = PUBLISH / f"{stem}.mp4", 2
        while dst.exists():
            dst, n = PUBLISH / f"{stem}-{n}.mp4", n + 1
        shutil.copy2(src, dst)
        if src.with_suffix(".jpg").is_file():
            shutil.copy2(src.with_suffix(".jpg"), dst.with_suffix(".jpg"))
        job["published"] = str(dst)
    except OSError as error:
        log("could not copy the clip to LFSTUDIO_PUBLISH_DIR:", error)


def make_poster(mp4):
    """Poster frame (~1 s in) for the gallery; best effort."""
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "1", "-i", str(mp4), "-frames:v", "1",
                        "-vf", "scale=640:-2", "-q:v", "4", str(Path(mp4).with_suffix(".jpg"))],
                       timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def clip_file(job):
    """A finished job's MP4: the copy in its session directory, or the LFSTUDIO_PUBLISH_DIR copy once the session
    directory has been pruned."""
    for key in ("output", "published"):
        if job and job.get(key) and Path(job[key]).is_file():
            return Path(job[key])
    return None


def clip_name(job):
    """The file name a clip downloads as (/video/<id>): what it is, and its id's start. An upscale's is its clip's
    (the parent's id) with the resolution and its own id's start after it: ..._<clip id6>_2k_<upscale id6>.mp4, so
    two upscales of a clip at one resolution (another strength, seed or audio) don't share a name."""
    if is_upscale(job):
        return (f"lf-studio_{job['task']}_{job.get('seconds', 5)}s_seed{job['seed']}_{job['parent'][:6]}_"
                f"{job['resolution']}_{job['id'][:6]}.mp4")
    tag = model_tag(job)                   # a clip of another model than sol-h3: the model and its options first
    return (f"lf-studio_{tag + '_' if tag else ''}{job['task']}_{job.get('seconds', 5)}s_seed{job['seed']}_"
            f"{job['id'][:6]}.mp4")


def settled(job):
    """Whether a job has its final status (done, failed or cancelled) and is saved with it."""
    return job["status"] in FINISHED and not job.get("saving")


PHASES = {"starting": "starting", "loading models + generating": "loading", "generating": "generating",
          "retrying after a worker error": "retrying", "freeing memory": "unloading", "upscaling": "upscaling",
          "encoding the prompt": "encoding"}


def runs_before(a, b):
    """Whether queued job `a` runs before queued job `b`: clips in order, then upscales in order (next_job)."""
    return (is_upscale(a), a["created"]) < (is_upscale(b), b["created"])


def api_record(job):
    """A job as the API shows it (#29): the stored record plus
      state      queued, running, done, failed or cancelled: stable codes for programs (status is for people);
      phase      running only (else None): cancelling, saving, starting, loading (the models, with its clip),
                 generating, retrying (after a worker error), unloading (an upscale or a clip of a job model freeing
                 the memory), encoding (a LightX2V Turbo clip's prompt), upscaling (its job runs; upscale_phase says
                 where) or running; a LightX2V Turbo clip's job_phase says where its job is;
      ahead      queued only: the jobs that run before it (every clip queued runs before an upscale);
      elapsed_s  running only: seconds since it started;
      urls       done, with its MP4 still there: {"video", "poster"}, paths relative to the studio's address;
      upscalable done, with the Upscale on: as in /api/state;
      model, file_name  with more than one model on (with_model_fields)."""
    with lock:
        rec = dict(job)
        ahead = sum(j["status"] == "queued" and j is not job and runs_before(j, job) for j in jobs)
    status = rec["status"]
    rec["state"] = status if settled(rec) else "queued" if status == "queued" else "running"
    rec["phase"] = None
    if rec["state"] == "queued":
        rec["ahead"] = ahead
    elif rec["state"] == "running":
        rec["phase"] = ("cancelling" if rec.get("cancel_requested") else "saving" if rec.get("saving")
                        else PHASES.get(status, "running"))
        if rec.get("started"):
            rec["elapsed_s"] = round(time.time() - rec["started"], 1)
    elif rec["state"] == "done" and clip_file(rec):
        rec["urls"] = {"video": f"/video/{rec['id']}", "poster": f"/poster/{rec['id']}"}
    with_upscalable([rec])
    with_model_fields([rec])
    return rec


def with_model_fields(records):
    """With more than one model on (MULTI), give each record among `records` (copies) "model" (every clip has one:
    sol-h3 for a record without it; an upscale has none) and "file_name" (what /video/<id> downloads it as:
    clip_name). Nothing with sol-h3 alone, so the records are as before the model choice."""
    for rec in records if MULTI else ():
        if not is_upscale(rec):
            rec.setdefault("model", SOL_H3)
        rec["file_name"] = clip_name(rec)
    return records


preload = {"pending": None}      # (key, case) to warm up when nothing is queued: at start, or "Load selected now"


def warmup_case(job):
    """A finished clip as a warm-up case: its prompt and inputs, seed 999. None when its manifest or inputs are gone."""
    if not PATHS.get(job.get("task"), Path("/nonexistent")).is_file():
        return None
    case = {"case_id": "preload", "prompt": job["prompt"], "seed": 999, "task": job["task"]}
    for k in ("first_frame", "last_frame", "references"):
        if job.get(k):
            case[k] = job[k]
    media = [case.get("first_frame"), case.get("last_frame")] + [r.get("path") for r in case.get("references", [])]
    if any(m and not Path(m).is_file() for m in media):
        return None
    try:
        return normalize_case(case)
    except Exception:
        return None


def preload_candidate():
    """The last finished clip's settings and warm-up case, or None (see warmup_case). Upscales don't count: they
    load none of the studio's models. None when that clip is of a job model (LightX2V Turbo): whoever made it is
    making those, and a warm-up of Sol-H3 would only be closed again. A clip of a length no longer offered (3 or 4 s,
    up to 0.5.0) warms up 5 s, the length its Remix on the page picks."""
    last = next((j for j in reversed(jobs) if j.get("status") == "done" and not j.get("deleted")
                 and not is_upscale(j)), None)
    case = warmup_case(last) if last and job_model(last) == SOL_H3 else None
    if not case:
        return None
    seconds = last.get("seconds", 5)
    return (last["task"], last["orientation"], seconds if seconds in DURATIONS else 5), case


WARMUP_INPUTS = INPUTS / "_warmup"   # the running warm-up's own links to its input files


def own_inputs(case):
    """The warm-up case with hardlinks of its input files in WARMUP_INPUTS (like Remix), so deleting the clip they
    came from can't take them away while the models load. None if they are gone already."""
    shutil.rmtree(WARMUP_INPUTS, ignore_errors=True)     # the previous warm-up's, no longer read
    out = dict(case)
    try:
        WARMUP_INPUTS.mkdir(parents=True)
        for k in ("first_frame", "last_frame"):
            if case.get(k):
                out[k] = reuse_input(WARMUP_INPUTS, case[k])
        if case.get("references"):
            out["references"] = [dict(r, path=reuse_input(WARMUP_INPUTS, r["path"])) for r in case["references"]]
    except (OSError, ValueError) as error:
        log("warm-up skipped, its input files are gone:", error)
        return None
    return out


WARMUP_PROMPT = ("integrated_multimodal_description: A quiet lake at sunrise, mist over the water, one continuous wide "
                 "shot.\noverall_soundscape: Soft wind and water.\nnon_diegetic_music: None.")


def mode_warmup_case(task):
    """A warm-up case for a mode ("Load selected now"): the newest of its last few finished clips whose inputs still
    exist; for t2va without one, a built-in prompt. fl2va and ref2va need real inputs, so without such a clip: None."""
    recent = [j for j in reversed(jobs) if j.get("task") == task and j.get("status") == "done"
              and not j.get("deleted") and not j.get("moving") and not is_upscale(j)]
    for j in recent[:5]:
        case = warmup_case(j)
        if case:
            return case
    if task == "t2va" and PATHS["t2va"].is_file():
        return normalize_case({"case_id": "preload", "prompt": WARMUP_PROMPT, "seed": 999, "task": "t2va"})
    return None


def run_preload(key, case):
    """Load and compile the models for `key` before anyone asks (a normal warm-up pass, output discarded)."""
    log("preloading", key)
    root = session_root(key, "preload-")
    try:
        p = new_pipeline(key, root)
    except Exception as e:
        log("preload skipped:", f"{type(e).__name__}: {e}"[:200])
        with lock:
            session.update(preloading=False, loading_key=None, state="idle" if session["pipeline"] is None else
                           session["state"])
        return
    # Until takeover_until, a job for these same settings stops the warm-up and loads the models itself: its loading
    # pass renders it. Started t s into a warm-up that takes L, it is done at about t + L (plus ~5 s to stop the
    # warm-up); waiting is done at L + W, W being a clip with the models loaded. So taking over wins while t < W - 5.
    # Before a warm clip of this setting has been measured: half the expected load time (5 s 16:9 t2va: L ~105-113 s,
    # W ~65 s).
    # load_eta and warm_s let the page say when a clip submitted now would be ready (both ways)
    warm, eta = warm_time(key), load_eta(key)
    with lock:                             # the worker set preloading when it took the request; a Generate or an
        stopped = not session.get("preloading") or stopping.is_set()    # Unload since then has cleared it
        if not stopped:
            session.update(pipeline=p, state="loading", since=time.time(), loading_key=key,
                           takeover_until=time.time() + (warm - 5 if warm else eta / 2), load_eta=eta, warm_s=warm)
    if stopped:
        close_session("warm-up no longer wanted", p)
        return
    t0 = time.time()
    try:
        p.start(case)
    except BaseException as e:
        log("preload stopped:", f"{type(e).__name__}: {e}"[:200])
        session["preloading"] = False
        close_session("preload stopped", p)
        return
    with lock:                             # atomic with abort_preload: either it is aborted or it is ready
        aborted = not session.get("preloading")
        if not aborted:
            session.update(key=key, state="ready", since=time.time(), loading_key=None, preloading=False,
                           takeover_until=None, load_eta=None, warm_s=None)
    if aborted:                            # aborted just as it finished: its workers got SIGTERM
        close_session("preload stopped", p)
        return
    remember_load(key, round(time.time() - t0, 1))     # a warm-up pass is a cold start too (load + one clip)
    log(f"preload ready in {time.time() - t0:.0f} s")


def signal_workers(skip=()):
    """SIGTERM the pipeline's worker process groups: children of this server that lead their own process group (the
    runtime starts each worker with start_new_session; ffmpeg and ffprobe run in the server's group). Groups in
    `skip` are left alone. Returns the groups signalled."""
    me, sent = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            ppid, pgrp = map(int, (d / "stat").read_text().rsplit(")", 1)[1].split()[1:3])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me and pgrp == int(d.name):
            try:
                os.killpg(pgrp, signal.SIGTERM)
                sent.append(pgrp)
            except OSError:
                pass
    return sent


def abort_preload(reason):
    """Stop a warm-up that is no longer wanted: SIGTERM the worker process groups (children of this server).
    Call it without holding `lock`."""
    with lock:
        if not session.get("preloading"):
            return False
        session["preloading"] = False
    log("aborting preload:", reason)
    signal_workers()
    return True


def stop_job(job):
    """Stop a running job that has been cancelled (its own thread, started by /api/cancel): while its pipeline call
    runs (work["job"]), SIGTERM the worker process groups as abort_preload does, also those started after the first
    signal (the loading pass starts Stage2 once Stage1 is loaded). Each group once: qwen_python.sh stops its
    container in a TERM trap, which a second signal would cut short. A GPU job's process (work["process"]: an
    upscale's job, a LightX2V Turbo clip's prompt encode or its job) gets SIGKILL if it is still there GPU_JOB_STOP_S
    after its SIGTERM: a job's handler runs only once the GPU call under way returns, which can take minutes. The
    signals and the checks are under `lock`, so they never reach the next job's workers. Ends when the job has
    finished."""
    sent, termed = set(), None             # termed: (pid, when) of the GPU job's process signalled
    while not stopping.is_set():
        with lock:
            if job["status"] in FINISHED:
                return
            if work["job"] is job:
                new = signal_workers(sent)
                if new:
                    log(f"cancel: sent SIGTERM to the worker process group(s) {new} of", job["case_id"])
                sent.update(new)
                proc = work["process"]
                if proc is not None and proc.pid in sent and (termed is None or termed[0] != proc.pid):
                    termed = (proc.pid, time.time())
                # proc is None between a job's GPU processes (after one, before its pipeline call returns)
                if (termed and proc is not None and termed[0] == proc.pid and time.time() - termed[1] > GPU_JOB_STOP_S
                        and proc.poll() is None):
                    log(f"cancel: {work['labels'][0]} {proc.pid} is still there {GPU_JOB_STOP_S} s after SIGTERM: "
                        "SIGKILL")
                    with contextlib.suppress(OSError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    termed = (proc.pid, float("inf"))      # once
        time.sleep(0.2)


# ------------------------------------------------------------------------------------------ the experimental Upscale
UPSCALE_REQUEST = ("id", "resolution", "strength", "audio", "seed", "pin")    # POST /api/upscale's keys
FREE_WAIT_S = 5                  # how long a memory floor waits for a closed session's memory to come back
GPU_JOB_STOP_S = 20              # Cancel and shutdown: SIGKILL a GPU job's process this long after its SIGTERM
UPSCALE_FREE_WAIT_S, UPSCALE_STOP_S = FREE_WAIT_S, GPU_JOB_STOP_S      # their names up to 0.5.1
UPSCALE_LENGTH = "Upscale takes clips of the lengths the studio makes, 5 to 15 s, not {} s"
# a clip whose MP4 (its name: _<W>x<H>_<frames>f.mp4) has frames the job has no case for (its FRAME_CASES), or those
# of another length than its record's: refused, never cut
UPSCALE_FRAMES = "Upscale can't take this clip: its MP4 has {} frames, not the frames of a {} s clip"
# a guard: every orientation the studio makes (runtime.geometry's ORIENTATIONS) is one of the job's SHAPES
# (tests/test_upscale.py checks it), so only an orientation added later gets this
UPSCALE_SHAPE = "Upscale needs a 16:9 or 9:16 clip (1344x768 or 768x1344)"
# A 5 s clip that keeps every frame its model made (LightX2V Turbo's with a last keyframe: 124, ending on the frame
# that keyframe is anchored to) is upscaled whole at these sizes; at 4K it is cut to its first 121 (NVIDIA's length)
# until a 124-frame 4K run (upscale_frames). Every other clip is taken whole, at the sizes the job offers for its
# length (upscale/regen2k_job.py offer_row: 4K takes 5 s clips only), or refused.
UPSCALE_KEEP_ALL = ("1080p", "2k")
UPSCALE_NAMES = {"1080p": "1080p", "2k": "2K", "4k": "4K"}
# /api/state "upscale_offers" (null with the Upscale off): what the job offers per length and size, its times and
# whether that length and size has run on the GPU through the studio, or a longer clip at that size has
# (upscale/regen2k_job.py offers_table: {"<seconds>": {size: {"eta_s": [base, step], "ran"} | {"why"}}}), for the
# lengths the models' "upscale" tables take. The page reads its popover's sizes, times and notes and an upscale's
# expected time there, by the clip's length (the job's OFFER_ETA_S: measured, or from the job's model of the runs). A
# request for a size not offered is refused with why.
UPSCALE_OFFERS = ({seconds: row for seconds, row in regen.offers_table().items()
                   if int(seconds) in {v for m in MODELS.values() for v in m.get("upscale", {}).get("seconds", ())}}
                  if UPSCALE else None)
# Keep my frames (POST /api/upscale "pin", the docstring): the clip's keyframes held in the refine where the clip
# shows them already (the job's check_pin). PIN_ROUTES says where it is offered, by the clip's model and the ends it
# can hold (clip_keyframes): the sizes (of those the job offers for the clip's length), the strengths, and where it has
# run on the GPU through the studio ("ran", 2026-10-29/30; the page notes the others): a size alone is a 5 s clip at
# that size, "<size>/<seconds>" a clip of that length ("1080p/10": both ends of a LightX2V Turbo 10 s clip, 2026-10-30).
# One line per route, so a route that fails its GPU run goes with its line. Not at 4K (never run there); Balanced only
# (Faithful held a first frame only ~1 dB closer). On a long clip the pin runs with the job's lean blocks where its
# forward is lean (1080p from 10 s, 2K from 6 s: regen.lean_chunks_for), as it ran at 1080p 10 s; at the other
# lengths and sizes the note says it has not run.
PIN = ("auto", "off")
PIN_SIZES = ("1080p", "2k")      # ("1080p",) takes 2K out
PIN_ROUTES = {
    (TURBO, "first"): {"sizes": PIN_SIZES, "strengths": ("balanced",), "ran": ("1080p",)},
    (TURBO, "last"): {"sizes": PIN_SIZES, "strengths": ("balanced",), "ran": ("1080p",)},
    (TURBO, "first+last"): {"sizes": PIN_SIZES, "strengths": ("balanced",), "ran": ("1080p", "2k", "1080p/10")},
    (SOL_H3, "first"): {"sizes": PIN_SIZES, "strengths": ("balanced",), "ran": ("1080p",)},
}
# why an end of a Keyframes clip is not held, besides the job's own reasons (regen.PIN_WHY: the picture gone or
# changed, the clip stopping short of the last one)
PIN_SOL_LAST = "Sol-H3 clips stop {} before your last picture, so the Upscale can't hold it"   # sol_last: 3 frames
PIN_BEFORE = "the clip was made before 0.5.2, which did not hold your first frame"
PIN_SETTING = "the clip was made with SOL_H3_STAGE2_KEYFRAME={}, which did not hold your first frame"
PIN_NO_RECORD = "the clip has no record of how it used your picture"
PIN_FIT = "the studio can't tell how your picture was fitted to the clip"
PIN_NOT_YET = "not for this clip's keyframes yet"
PIN_SIZE = "not at {} yet"
PIN_STRENGTH = "with strength {} only, for now"


class JobFailed(RuntimeError):
    """A GPU job (gpu_process) could not start, or failed: the message says why (the job's own error line, when it
    has one)."""


class UpscaleFailed(JobFailed):
    """An upscale could not start, or its job failed."""


class ModelJobFailed(JobFailed):
    """A clip of a job model (LightX2V Turbo) could not start, or a step of it failed."""


def mem_available_gib():
    """MemAvailable in GiB (read from LFSTUDIO_TEST_MEMINFO in the tests), or None when it can't be read. The
    server's own reading, with the Upscale off too."""
    try:
        with open(MEMINFO) as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key == "MemAvailable":
                    return int(rest.split()[0]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return None


MP4_SIZE = re.compile(r"_(\d+)x(\d+)_(\d+)f\.mp4$")     # a clip's <name>_<W>x<H>_<frames>f.mp4 (either model)


def upscale_frames(clip, resolution=None):
    """The frames an upscale keeps of a clip's `clip` frames at `resolution` (None: at UPSCALE_KEEP_ALL's sizes): all
    of them where the job takes them (its FRAME_CASES: a length the studio makes), but NVIDIA's 121 of a 5 s clip
    that keeps every frame made (124) at a size outside UPSCALE_KEEP_ALL (4K); None for a count the job has no case
    for (never cut)."""
    case = regen.FRAME_CASES.get(clip)
    if case is None:
        return None
    if case["seconds"] == 5 and case["pad"] == 0 and resolution is not None and resolution not in UPSCALE_KEEP_ALL:
        return regen.NVIDIA_FRAMES
    return clip


def upscale_sources(job, resolution=None, hash_now=False):
    """What an upscale of a finished clip at `resolution` (None: at 1080p and 2K) reads, from its record and its files
    (build_input puts them into the upscale's input folder): {"case", "final" (its MP4), "frames" ({"clip" (its MP4's,
    by its name; None for a name without them), "made" (by H3), "kept" (upscale_frames; None off the job's cases)}),
    "video" (what build_input does to the MP4: "clip", taken as it is, or for a LightX2V Turbo clip "stretch" from 576p
    to 768p, every frame, "trim" to its first 121 frames (at 4K, a 5 s clip with a last keyframe, which has 124), or
    "stretch+trim"), "pcm" (its audio for audio clip, which may be missing:
    Sol-H3's Stage-1 audio or LightX2V Turbo's turbo/audio.pt), "text" (where its conditioning of text alone comes
    from: "clip", its own (t2va); "generation", the one a LightX2V Turbo Keyframes clip made with itself, qwen-text/;
    "encode", one made for the upscale), "conditioning" (that file, None for "encode"), "keyframes" and "pin_why"
    (clip_keyframes; hash_now: its pictures hashed)}, or None when its own directory or its MP4 is gone (a clip from
    before per-clip directories, or one whose session directory was pruned). The case is the MP4's grandparent for
    both models: <case>/stage2/refined_*.mp4 (Sol-H3), <case>/turbo/clip_*.mp4 (LightX2V Turbo)."""
    if not job.get("output"):
        return None
    final = Path(job["output"])
    case = final.parent.parent
    if case.name != job.get("case_id") or not in_sessions(case) or not final.is_file():
        return None
    turbo = job_model(job) == TURBO
    m = MP4_SIZE.search(final.name)
    size, frames = ((int(m.group(1)), int(m.group(2))), int(m.group(3))) if m else (None, None)
    kept, video = upscale_frames(frames, resolution), []
    if turbo:                              # Sol-H3's MP4 is the job's input as it is (the job checks it)
        if size != regen.SHAPES[job["orientation"]]["input"]:
            video.append("stretch")
        if kept is not None and kept != frames:
            video.append("trim")
    else:
        kept = frames if kept else None    # whole: Sol-H3's counts are never cut
    keyframes, why = clip_keyframes(job, final, hash_now)
    sources = {"case": case, "final": final, "video": "+".join(video) or "clip",
               "frames": {"clip": frames, "made": regen.made_frames(frames) if frames else None, "kept": kept},
               "keyframes": keyframes, "pin_why": why,
               "pcm": case / ("turbo/audio.pt" if turbo else "stage1/stage1_direct_tensors.pt")}
    if job.get("task") == "t2va":
        sources.update(text="clip", conditioning=case / "qwen" / "conditioning.pt")
    elif turbo and (case / "qwen-text" / "conditioning.pt").is_file():
        sources.update(text="generation", conditioning=case / "qwen-text" / "conditioning.pt")
    else:                                  # Sol-H3 Keyframes and Reference clips, and a Turbo one without qwen-text/
        sources.update(text="encode", conditioning=None)
    return sources


def read_record(path):
    """A clip's own JSON record (Sol-H3's stage2/stage2_result.json, a LightX2V Turbo clip's turbo/receipt.json) as a
    dict, or None."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError):      # RecursionError: JSON nested too deep
        return None
    return data if isinstance(data, dict) else None


def _dict(value):
    """A record's nested value where a dict is wanted, or an empty one."""
    return value if isinstance(value, dict) else {}


def _count(value):
    """A number of frames, or None."""
    return value if type(value) is int and value > 0 else None


def _size_pair(value):
    """[width, height] in pixels, or None."""
    if isinstance(value, list) and len(value) == 2 and all(type(v) is int and 0 < v <= 16384 for v in value):
        return value
    return None


def _sha(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


@functools.lru_cache(maxsize=256)
def keyframe_record(model, orientation, path, mtime_ns, ends):
    """What a Keyframes clip's own record (`path`, read when mtime_ns is not None; cached by it: with_upscalable asks
    for every finished clip on every /api/state) says of its keyframes `ends`, as JSON: [{end: {"sha256" (of the
    picture it was made from), "fit" (how its model put it on its canvas: the job's FITS), "canvas", "size" (the
    picture's, turned by its EXIF, or None)}}, {end: why not}].
    Sol-H3: its first frame where Stage 2 held it (stage2_keyframe "pin", 0.5.2's default), stretched onto the canvas,
    the sha256 from its capture; never its last, which the clip stops a few frames before (3 at 5 s: Stage 2 keeps the
    largest 8n+1 of the frames H3 makes, 121 of 124, and H3 anchors the last keyframe to the last of them):
    PIN_SOL_LAST, which clip_keyframes words by the clip's frames (sol_last).
    LightX2V Turbo: its receipt's geometry, the fit of each picture (jobs/h3turbo_job.py keyframe_fit: a cover-crop must
    be the one the job's cover_crop computes for its size and canvas) and their sha256; a last keyframe only where the
    clip kept every frame made, so that it ends on that frame (it does since the model came).
    A record of another shape (hand-edited, or not the studio's) gives its reasons, never an error: with_upscalable
    asks on every /api/state, /api/jobs and record."""
    record = read_record(path) if mtime_ns is not None else None
    found, why = {}, {}
    try:
        for end in ends:
            if model == SOL_H3 and end == "last":
                why[end] = PIN_SOL_LAST
            elif record is None:
                why[end] = PIN_NO_RECORD
            elif model == SOL_H3:
                mode = record.get("stage2_keyframe")
                sha = _sha(_dict(_dict(record.get("capture")).get("input_sha256")).get("first_frame"))
                if mode is None:
                    why[end] = PIN_BEFORE
                elif mode != "pin":
                    why[end] = PIN_SETTING.format(record.get("stage2_keyframe_setting") or mode)
                elif sha is None:
                    why[end] = PIN_NO_RECORD
                else:
                    found[end] = {"sha256": sha, "fit": "stretch",
                                  "canvas": list(regen.SHAPES[orientation]["input"]), "size": None}
            else:
                image, geometry = _dict(_dict(record.get("images")).get(end)), _dict(record.get("geometry"))
                sha = _sha(_dict(record.get("inputs_sha256")).get(f"{end}_frame"))
                canvas = _size_pair([geometry.get("width"), geometry.get("height")])
                size, fit = _size_pair(image.get("size")), image.get("fit")
                made, kept = _count(geometry.get("frames")), _count(geometry.get("keep_frames"))
                if (sha is None or canvas is None or fit not in regen.FITS
                        or end == "last" and (made is None or kept is None or kept > made)):
                    why[end] = PIN_NO_RECORD
                elif end == "last" and kept < made:
                    why[end] = regen.PIN_WHY["past_end"].format(n=made - kept)
                elif fit == "cover-crop" and (size is None or [list(v) for v in regen.cover_crop(size, canvas)]
                                              != [image.get("resized"), image.get("crop")]):
                    why[end] = PIN_FIT
                else:
                    found[end] = {"sha256": sha, "fit": fit, "canvas": canvas, "size": size}
    except Exception as error:      # never a reason for /api/state, /api/jobs or a record to fail (with_upscalable)
        log(f"Keep my frames can't read {path}: {type(error).__name__}: {str(error)[:200]}")
        found, why = {}, {end: PIN_SOL_LAST if model == SOL_H3 and end == "last" else PIN_NO_RECORD for end in ends}
    return json.dumps([found, why])


IMAGE_SHA = {}                   # (path, size, mtime_ns) -> the sha256 a submit or a run found (image_state)
IMAGE_SHA_MAX = 1024
image_lock = threading.Lock()


def image_state(path, want, hash_now=False):
    """None while the picture at `path` can be the one a clip was made from (sha256 `want`), else why not (the job's
    PIN_WHY: gone, changed). Without hash_now (/api/state, with_upscalable) it never reads the picture: it is there
    and not empty, and the sha256 found for that path, size and mtime (IMAGE_SHA), if any, is the clip's. hash_now (a
    submit, a run) hashes a picture not seen at that size and mtime; the job checks the sha256 again itself."""
    try:
        info = Path(path).stat()
    except (OSError, TypeError, ValueError):
        return regen.PIN_WHY["gone"]
    if not Path(path).is_file() or not info.st_size:
        return regen.PIN_WHY["gone"]
    key = (str(path), info.st_size, info.st_mtime_ns)
    with image_lock:
        digest = IMAGE_SHA.get(key)
    if digest is None and hash_now:
        try:
            digest = file_sha256(path)
        except OSError:
            return regen.PIN_WHY["gone"]
        with image_lock:
            IMAGE_SHA[key] = digest
            while len(IMAGE_SHA) > IMAGE_SHA_MAX:
                del IMAGE_SHA[next(iter(IMAGE_SHA))]          # the oldest
    return None if digest in (None, want) else regen.PIN_WHY["changed"]


def sol_last(final):
    """Why a Sol-H3 clip's last frame is not held (PIN_SOL_LAST), by the frames its MP4 (`final`) stops before the last
    H3 made, where its last keyframe is: 3 at 5 s, 7 at 8 s ("a few" for an MP4 whose name has no count); none for an
    MP4 that keeps every frame made (73 at 3 s, a length the Upscale does not take): not yet, then (no route holds a
    Sol-H3 clip's last frame)."""
    m = MP4_SIZE.search(Path(final).name)
    n = regen.made_frames(int(m.group(3))) - int(m.group(3)) if m and int(m.group(3)) > 0 else None
    if n == 0:
        return PIN_NOT_YET
    return PIN_SOL_LAST.format("a few frames" if n is None else f"{n} frame{'s' * (n != 1)}")


def clip_keyframes(job, final, hash_now=False):
    """The ends of a finished Keyframes clip (its MP4: `final`) the Upscale can hold, Keep my frames: ({end: {"from"
    (the picture it was made from, the record's first_frame or last_frame), "sha256", "fit", "canvas", "size"}},
    {end: why not}), first before last; ({}, {}) for a Text or Reference clip. Its own record decides
    (keyframe_record), then the picture: still on disk and still the one it was made from (image_state; hashed only
    with hash_now). A Sol-H3 clip's last frame: why not by its length (sol_last)."""
    ends = tuple(end for end in regen.ANCHORS if job.get(f"{end}_frame"))
    model = job_model(job)
    if job.get("task") != "fl2va" or not ends or model not in (SOL_H3, TURBO):
        return {}, {}
    # Sol-H3: <case>/stage2/stage2_result.json; LightX2V Turbo: its job's receipt.json beside its MP4, turbo/
    path = final.parent.parent / "stage2" / "stage2_result.json" if model == SOL_H3 else final.parent / "receipt.json"
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    found, why = json.loads(keyframe_record(model, job.get("orientation"), str(path), mtime_ns, ends))
    if why.get("last") == PIN_SOL_LAST:
        why["last"] = sol_last(final)
    for end in list(found):
        problem = image_state(job[f"{end}_frame"], found[end]["sha256"], hash_now)
        if problem:
            why[end] = problem
            del found[end]
        else:
            found[end]["from"] = job[f"{end}_frame"]
    return found, {end: why[end] for end in ends if end in why}


def pin_offer(job, sources):
    """The ends a clip can hold and their route (PIN_ROUTES, by its model and those ends), with why not for the
    others: (ends, route or None, {end: why})."""
    ends, why = list(sources["keyframes"]), dict(sources["pin_why"])
    route = PIN_ROUTES.get((job_model(job), "+".join(ends))) if ends else None
    if ends and route is None:
        why.update(dict.fromkeys(ends, PIN_NOT_YET))
        ends = []
    return ends, route, {end: why[end] for end in regen.ANCHORS if end in why}


def pin_ran(route, size, seconds):
    """Whether a Keep my frames route has run on the GPU through the studio at this size for a clip of this length
    (PIN_ROUTES "ran": a size alone at 5 s, "<size>/<seconds>" at another length)."""
    return (size if seconds == 5 else f"{size}/{seconds}") in route["ran"]


def pin_plan(parent, sources, resolution, strength, asked):
    """An upscale's Keep my frames (its record's "pin"), None for a clip without keyframes: {"asked" (auto or off),
    "ends" (the ends job.json names: with auto, every end the clip can hold where its route has this size and
    strength), "why" ({end: why not})}."""
    ends, route, why = pin_offer(parent, sources)
    if not ends and not why:
        return None
    reason = (None if not ends else PIN_SIZE.format(UPSCALE_NAMES[resolution]) if resolution not in route["sizes"]
              else PIN_STRENGTH.format(" or ".join(route["strengths"])) if strength not in route["strengths"]
              else None)
    if reason:
        why.update(dict.fromkeys(ends, reason))
        ends = []
    return {"asked": asked, "ends": ends if asked == "auto" else [],
            "why": {end: why[end] for end in regen.ANCHORS if end in why}}


AUDIO_FILE = {SOL_H3: "Stage-1 audio (stage1/stage1_direct_tensors.pt)", TURBO: "audio (turbo/audio.pt)"}
UPSCALE_NO_T2VA = ("the Upscale encodes this clip's prompt as text alone first, which needs the t2va path manifest "
                   "(run install/70-manifests.sh)")
NOTHING_LEFT = {"fl2va": "nothing is left of the prompt without its keyframe line: the Upscale needs text",
                "ref2va": "nothing is left of the prompt without its references: the Upscale needs text"}
UNREADABLE = "the Upscale can't write this clip's prompt as text alone (the studio's log has why)"


@functools.lru_cache(maxsize=256)
def text_rewrite(task, prompt):
    """upscale_text's rewrite, as JSON (cached: with_upscalable asks for it on every /api/state, and a prompt never
    changes). Its line breaks are LF: a prompt from the page's form has CRLF ones (a browser's multipart), which
    text_only_prompt keeps, and the prompt encode reads its --prompt-file with LF ones (Python's universal newlines),
    so its receipt's prompt is the text with LF; the record, prompt-text-only.txt and the receipt then agree."""
    rewrite = TEXT_REWRITES.text_only_prompt if task == "fl2va" else TEXT_REWRITES.ref_text_only_prompt
    try:
        result = rewrite(prompt)
    except TEXT_REWRITES.TooLong as error:
        return json.dumps({"error": str(error)})
    except ValueError:
        return json.dumps({"error": NOTHING_LEFT[task]})
    except Exception as error:      # never a reason for /api/state, /api/jobs or a record to fail (with_upscalable)
        log(f"the Upscale's rewrite of a {task} prompt failed: {type(error).__name__}: {str(error)[:200]}")
        return json.dumps({"error": UNREADABLE})
    result["prompt"] = re.sub(r"\r\n?", "\n", result["prompt"])
    return json.dumps({k: result[k] for k in ("prompt", "dropped", "notes")})


def upscale_text(job):
    """A Keyframes (fl2va) or Reference (ref2va) clip's prompt as text alone, what the Upscale's encode reads in place
    of the pictures (bench/h3turbo_encode.py: text_only_prompt drops the keyframe line and the picture references,
    ref_text_only_prompt writes a Reference prompt as t2va's three fields): {"prompt", "dropped", "notes"}, or
    {"error"} when nothing is left, when it would be longer than the encode takes, or when the rewrite fails; None
    for a t2va clip."""
    if job.get("task") not in NOTHING_LEFT:
        return None
    return json.loads(text_rewrite(job["task"], job.get("prompt") or ""))


def text_record(parent, source):
    """An upscale's record of its text (submit_upscale): {"source": upscalable's "text"} and, for a text of the clip's
    prompt without its pictures, "prompt", "dropped" and "notes" (a LightX2V Turbo Keyframes clip's: what its encode
    recorded as text_only)."""
    record = {"source": source}
    if source == "generation" and isinstance(parent.get("text_only"), dict):
        record.update(prompt=parent["text_only"].get("prompt"), dropped=parent["text_only"].get("dropped"))
    elif source != "clip":
        record.update(upscale_text(parent) or {})
    return record


def upscalable(job):
    """Whether a job can be upscaled now: (200, {"clip_audio"[, "text"][, "video"][, "frames"][, "pin", "pin_routes"]
    [, "pin_why"]}) or (the HTTP status, why not). clip_audio: whether its audio is there for audio clip; text and
    video: upscale_sources' at 1080p and 2K, each only when it is not "clip" (so a Sol-H3 Text clip's answer is as
    before the Upscale took every clip); frames: {resolution: the frames kept} where they differ between the sizes (a
    5 s clip with a last keyframe: 124, but 121 at 4K); pin: the ends it can hold (pin_offer) and pin_routes, where:
    {resolution: {"strengths", "ran" (at the clip's length)}} at the sizes its route and its length have;
    pin_why: {end: why not} for the others (the pictures are not hashed here: image_state). Which sizes a clip's
    length gets, with their times, is not here: /api/state "upscale_offers" (upscale/regen2k_job.py offers_table), by
    length. It takes every finished clip of the models' "upscale" tables (Sol-H3's Text, Keyframes and Reference
    clips, LightX2V Turbo's Text and Keyframes clips at either size), of every length the studio makes (not Sol-H3's
    old 3 and 4 s ones: UPSCALE_LENGTH), 16:9 or 9:16 (the job upscales 9:16 through NVIDIA's code with the shapes turned), whole: not one whose
    MP4 has frames the job has no case for, or not those of its length (UPSCALE_FRAMES), nor a Keyframes or Reference
    clip with nothing left of its prompt without its pictures (or a text alone too long), nor one whose text must be
    encoded while the studio has no t2va path manifest (the encode's). Looks at the clip's files: call it without
    `lock`, with a copy of the job. Never raises for a record of another shape (with_upscalable)."""
    if is_upscale(job):
        return 400, "an upscale can't be upscaled again"
    model = job_model(job)
    spec = MODELS.get(model, {})
    name = spec.get("name", model)
    if "upscale" not in spec:
        return 400, f"Upscale does not take {name} clips"
    takes = spec["upscale"]
    if job.get("task") not in takes["tasks"]:
        return 400, f"Upscale does not take {name}'s {job.get('task')} clips"
    seconds = job.get("seconds", 5)
    if seconds not in takes["seconds"]:
        return 400, UPSCALE_LENGTH.format(seconds)
    if job.get("orientation") not in regen.SHAPES:
        return 400, UPSCALE_SHAPE
    text = upscale_text(job)
    if text and "error" in text:
        return 400, text["error"]
    if job.get("status") != "done":
        return 409, {"queued": "the clip is still queued", "failed": "the clip failed",
                     "cancelled": "the clip was cancelled"}.get(job.get("status"), "the clip is still being made")
    if job.get("moving"):
        return 409, "the clip is being moved; try again in a moment"
    sources = upscale_sources(job)
    if sources is None or sources["conditioning"] is not None and not sources["conditioning"].is_file():
        return 409, "the clip's working files (its session directory) are gone, so it can't be upscaled"
    clip = sources["frames"]["clip"]
    if clip not in regen.FRAME_CASES or regen.FRAME_CASES[clip]["seconds"] != seconds:    # never cut: refused
        return 400, UPSCALE_FRAMES.format("an unknown number of" if clip is None else clip, seconds)
    if sources["text"] == "encode" and "t2va" not in ready_tasks():     # the encode is a t2va one: its manifest
        return 409, UPSCALE_NO_T2VA
    answer = {"clip_audio": sources["pcm"].is_file()}
    answer.update({key: sources[key] for key in ("text", "video") if sources[key] != "clip"})     # what is new only
    kept = {size: upscale_frames(clip, size) for size in regen.SIZES}
    if len(set(kept.values())) > 1:
        answer["frames"] = kept
    ends, route, why = pin_offer(job, sources)
    if ends:
        offered = [size for size in regen.SIZES if size in route["sizes"]
                   and "why" not in regen.offer_row(seconds, size)]
        answer["pin"] = ends
        answer["pin_routes"] = {size: {"strengths": list(route["strengths"]), "ran": pin_ran(route, size, seconds)}
                                for size in offered}
    if why:
        answer["pin_why"] = why
    return 200, answer


def with_upscalable(records):
    """With the Upscale on, give each finished job among `records` (copies) "upscalable": {"ok": true,
    "clip_audio"} or {"ok": false, "why"} (/api/state, /api/jobs, api_record). Call it without `lock`."""
    for rec in records if UPSCALE else ():
        if rec.get("status") == "done":
            code, info = upscalable(rec)
            rec["upscalable"] = dict(ok=True, **info) if code == 200 else {"ok": False, "why": info}
    return records


def upscale_defaults():
    """What a POST /api/upscale without resolution, strength or audio gets (/api/state "upscale_defaults", so the
    page offers the same), or None with the Upscale off."""
    if not UPSCALE:
        return None
    return {"resolution": "2k", "strength": UPSCALE_CONF["strength"], "audio": UPSCALE_CONF["audio"]}


def upscale_request(raw):
    """POST /api/upscale's JSON body -> (id, resolution, strength, audio, seed or None, pin). null is the same as
    leaving a key out. Raises ValueError with what the client is told: any other key is unknown, "source" too (the
    draft source is gone; an upscale starts from the clip's MP4). pin (Keep my frames): auto, the default, or off; a
    clip without keyframes takes either."""
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise ValueError('expected JSON {"id": ..., "resolution": ..., "strength": ..., "audio": ..., "seed": ..., '
                         '"pin": ...}')
    for key in body:
        if key not in UPSCALE_REQUEST:
            raise ValueError(f'unknown key "{key}" (known: {", ".join(UPSCALE_REQUEST)})')
    body = {k: v for k, v in body.items() if v is not None}
    if not isinstance(body.get("id"), str) or not body["id"]:
        raise ValueError("id must be the id of a finished clip")
    picked = []
    defaults = upscale_defaults()
    for key, choices in (("resolution", tuple(regen.SIZES)), ("strength", tuple(regen.STRENGTHS)),
                         ("audio", regen.AUDIO)):
        value = body.get(key, defaults[key])
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{key} must be one of {', '.join(choices)}")
        picked.append(value)
    seed = body.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed must be a nonnegative 63-bit integer, or null for the clip's")
    pin = body.get("pin", "auto")
    if not isinstance(pin, str) or pin not in PIN:
        raise ValueError("pin must be auto or off")
    return (body["id"], *picked, seed, pin)


def offer_refusal(offer, strength):
    """Why the job's offer (upscale/regen2k_job.py offer: of a clip's frames kept at a size) does not take this
    request: the size not offered for the clip's length (its reason, naming the sizes that are), or a strength it does
    not have; None when it takes it."""
    if "why" in offer:
        return offer["why"]
    if strength not in offer["strengths"]:
        return f"strength {strength} is not offered for this clip at this size: {', '.join(offer['strengths'])}"
    return None


def submit_upscale(raw):
    """POST /api/upscale (the docstring): check the request and the clip, then queue the upscale as a job of its own.
    Returns the reply."""
    if not UPSCALE:
        return 404, {"error": "the experimental Upscale is off on this studio (LFSTUDIO_UPSCALE=1 turns it on)"}
    try:
        jid, resolution, strength, audio, seed, pin = upscale_request(raw)
    except ValueError as error:
        return 400, {"error": str(error)}
    with lock:
        parent = find_job(jid)
        parent = dict(parent) if parent else None
    if parent is None:
        return 404, {"error": "no such clip"}
    code, info = upscalable(parent)
    if code != 200:
        return code, {"error": info}
    if audio == "clip" and not info["clip_audio"]:
        return 409, {"error": f"the clip has no {AUDIO_FILE[job_model(parent)]}; ask for audio silence"}
    sources = upscale_sources(parent, resolution, hash_now=pin == "auto")     # at its size; its pictures hashed
    if sources is None:
        return 409, {"error": "the clip's working files (its session directory) are gone, so it can't be upscaled"}
    why = offer_refusal(regen.offer(sources["frames"]["kept"], resolution), strength)    # its length at this size
    if why:
        return 400, {"error": why}
    sigma, steps = regen.STRENGTHS[strength]
    uid = uuid.uuid4().hex
    job = {"id": uid, "kind": "upscale", "parent": jid, "case_id": f"up_{time.strftime('%H%M%S')}_{uid[:6]}",
           "task": parent["task"], "orientation": parent["orientation"], "seconds": parent.get("seconds", 5),
           "prompt": parent["prompt"], "seed": parent["seed"], "resolution": resolution, "strength": strength,
           "steps": steps, "start_sigma": sigma, "audio": audio,
           "refine_seed": parent["seed"] if seed is None else seed, "load": UPSCALE_CONF["load"],
           "parent_model": job_model(parent), "text": text_record(parent, info.get("text", "clip")),
           "video": sources["video"], "frames": sources["frames"]["kept"],
           "status": "queued", "created": time.time()}
    if parent.get("model_opts"):
        job["parent_opts"] = dict(parent["model_opts"])
    plan = pin_plan(parent, sources, resolution, strength, pin)
    if plan:
        job["pin"] = plan
    with lock:
        full = queue_full()
        now = find_job(jid)
        gone = now is None or now.get("moving")       # deleted, or a Delete or Undo began meanwhile
        if not (full or gone):
            job["created"] = max(time.time(), jobs[-1]["created"] + 0.001 if jobs else 0)
            jobs.append(job)
            preload["pending"] = None              # a requested warm-up that hasn't started: the upscale closes the
            session["switching_to"] = None         # models anyway
            waited_for = warm_up_wanted()          # a clip queued for the warm-up's models runs first: keep it
    if full:
        return full
    if gone:
        return 409, {"error": "the clip is being deleted or moved; try again in a moment"}
    save_job(job)
    if session.get("preloading") and not waited_for:
        abort_preload("an upscale arrived, which needs the memory")
    log("queued upscale", job["case_id"], "of", parent["case_id"],
        f"({resolution}, {strength}, audio {audio}, seed {job['refine_seed']}"
        + (f", keeping its {' and '.join(plan['ends'])} frame{'s' * (len(plan['ends']) > 1)}"
           if plan and plan["ends"] else "") + ")")
    wake.set()
    return 200, {"id": uid, "job": api_record(job)}


def enough_memory(floor):
    """MemAvailable (GiB) as soon as it reaches `floor`, else the most seen within FREE_WAIT_S (a closed
    session's memory comes back as its workers end); None when it can't be read. No wait for a floor of 0."""
    best, deadline = None, time.time() + FREE_WAIT_S
    while floor > 0:
        available = mem_available_gib()
        if available is not None:
            best = max(best or 0.0, available)
        if best is not None and best >= floor or time.time() >= deadline or stopping.is_set():
            break
        time.sleep(0.5)
    return best


UPSCALE_INPUT = "input"         # the upscale's input folder in its case: job.json "case"
UPSCALE_ENCODE_LABELS = ("the upscale's prompt encode", "the running upscale's prompt encode")
VIDEO_PREP_S = 600               # the longest a stretch or trim of a clip's MP4 may take (~1 s on a DGX Spark)


def link_or_copy(src, dst):
    """A file into the upscale's input folder: a hardlink (no extra space; the clip's Delete later leaves it), or a
    copy across filesystems. Returns which."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_video_command(src, dst, size, how, frames):
    """ffmpeg for a LightX2V Turbo clip's MP4 the job can't take as it is (upscale_sources' "video"): stretched to the
    768p canvas (LANCZOS, as the Upscale's GPU validation of 576p clips did: never cover-cropped, so it stays on the
    keyframe's grid), every frame, and/or cut to its first `frames` (at 4K, 121 of a clip with a last keyframe, which
    has 124: the frames it keeps after 121 run up to that keyframe), H.264 CRF 12. Its sound is copied for the mux: as
    it is, every packet, when the clip is only stretched (no frame limit or -t then: either one ends the copied sound
    a packet early), cut to the frames' length when it is cut."""
    scale = ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos"] if "stretch" in how else []
    cut = (["-frames:v", str(frames)], ["-t", f"{frames / 24:.6f}"]) if "trim" in how else ([], [])
    return (["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?"] + cut[0]
            + scale + ["-r", "24", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                       "-c:a", "copy"] + cut[1] + ["-movflags", "+faststart", str(dst)])


def build_input(job, parent, sources, case):
    """The upscale's input folder, <case>/input/ (job.json "case"), in the shape of a Sol-H3 clip's own, so the job
    reads every clip as it reads a Sol-H3 Text clip (its case_inputs): stage2/refined_*.mp4 (the clip's MP4, linked,
    all its frames; for a LightX2V Turbo clip at 576p, or a 5 s one at 4K with a last keyframe,
    prepare_video_command's 1344x768 or 768x1344 with the frames kept: all of them, or 121 of 124 at 4K),
    stage1/stage1_direct_tensors.pt (its audio, linked: Sol-H3's
    Stage-1 audio, or LightX2V Turbo's audio.pt, the PCM of the frames it made), qwen/ (its conditioning of text alone,
    linked: the clip's own or its qwen-text/; none yet when the text is encoded for the upscale, encode_upscale_text)
    and keyframe/<end>.<ext> (the picture of each end job["pin"] sends to the job, linked; one gone meanwhile is not
    sent, with its reason). Records it all in source.json (write_source). Returns (the folder, its MP4, the files so
    far)."""
    view = case / UPSCALE_INPUT
    (view / "stage2").mkdir(parents=True)
    files = {}
    if sources["video"] == "clip":     # named as the job finds it: a LightX2V Turbo clip's clip_* as refined_*
        name = sources["final"].name
        mp4 = view / "stage2" / (name if name.startswith("refined_") else "refined_" + name.partition("_")[2])
        how = link_or_copy(sources["final"], mp4)
    else:
        width, height = regen.SHAPES[parent["orientation"]]["input"]
        kept = sources["frames"]["kept"]
        mp4 = view / "stage2" / f"refined_{width}x{height}_{kept}f.mp4"
        partial = mp4.with_name(f".{mp4.stem}.part.mp4")
        try:
            r = subprocess.run(prepare_video_command(sources["final"], partial, (width, height), sources["video"],
                                                     kept), capture_output=True, text=True, timeout=VIDEO_PREP_S)
        except (OSError, subprocess.SubprocessError) as error:
            raise UpscaleFailed(f"could not prepare the clip's MP4 ({sources['video']}): {error}") from None
        if r.returncode != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            raise UpscaleFailed(f"could not prepare the clip's MP4 ({sources['video']}; ffmpeg: "
                                f"{regen.one_line(r.stderr)[-300:]})")
        os.replace(partial, mp4)
        how = sources["video"]
    files[f"stage2/{mp4.name}"] = {"from": str(sources["final"]), "how": how}
    if sources["pcm"].is_file():
        (view / "stage1").mkdir()
        pcm = view / "stage1" / "stage1_direct_tensors.pt"
        files[f"stage1/{pcm.name}"] = {"from": str(sources["pcm"]), "how": link_or_copy(sources["pcm"], pcm)}
    if sources["conditioning"] is not None:
        (view / "qwen").mkdir()
        for name in ("conditioning.pt", "conditioning.json"):
            src = sources["conditioning"].parent / name
            if src.is_file():
                files[f"qwen/{name}"] = {"from": str(src), "how": link_or_copy(src, view / "qwen" / name)}
    pin = job.get("pin") or {}
    for end in list(pin.get("ends", ())):
        src = Path(sources["keyframes"][end]["from"])
        (view / "keyframe").mkdir(exist_ok=True)
        dst = view / "keyframe" / f"{end}{src.suffix.lower()}"
        try:
            files[f"keyframe/{dst.name}"] = {"from": str(src), "how": link_or_copy(src, dst)}
        except OSError:                    # deleted since the run's check: not sent
            pin["ends"].remove(end)
            pin["why"][end] = regen.PIN_WHY["gone"]
    write_source(job, parent, view, files, sources)
    return view, mp4, files


def write_source(job, parent, view, files, sources):
    """<input>/source.json: where the upscale's input comes from: the clip (id, model and its options, task,
    orientation, length), what was done to its MP4 and its frames (upscale_sources'), its text (the upscale's record
    of it), per file where from, how (link, copy, stretch, trim, encode) and its sha256, and per keyframe sent its
    record (the sha256 the clip was made from, its size, fit and canvas; same_as_clip: the link's sha256 is it)."""
    for rel, entry in files.items():
        if "sha256" not in entry and (view / rel).is_file():
            entry["sha256"] = file_sha256(view / rel)
    keyframes = {}
    for rel, entry in files.items():
        if rel.startswith("keyframe/"):
            end = Path(rel).stem
            record = sources["keyframes"][end]
            keyframes[end] = {"from": entry["from"], "how": entry["how"], "sha256": record["sha256"],
                              "size": record["size"], "fit": record["fit"], "canvas": record["canvas"],
                              "same_as_clip": entry.get("sha256") == record["sha256"]}
    regen.write_json(view / "source.json", {
        "parent": parent["id"], "case_id": parent.get("case_id"), "model": job_model(parent),
        "options": parent.get("model_opts"), "task": parent.get("task"), "orientation": parent.get("orientation"),
        "seconds": parent.get("seconds", 5), "video": job.get("video"), "frames": sources["frames"],
        "text": job.get("text"), "files": files, "keyframes": keyframes})


def encode_upscale_text(job, parent, case, view):
    """The upscale's conditioning of text alone, for a clip that has none (upscale_sources' "encode": a Sol-H3
    Keyframes or Reference clip, a LightX2V Turbo Keyframes clip without qwen-text/): the prompt encode
    (LFSTUDIO_H3TURBO_ENCODE_SCRIPT, bench/h3turbo_encode.py --upscale) of the clip's prompt without its pictures
    (job["text"]["prompt"], <case>/prompt-text-only.txt), a t2va encode with a transient Qwen worker into
    <input>/qwen/, as a GPU process of its own (gpu_process: Cancel and shutdown stop it, and it stops its worker;
    after an encode that ended badly, stop_encode_worker). Its log is <case>/encode.log. Then its receipt: t2va, of
    that prompt. Returns its summary (encode.json)."""
    text, prompt_file, log_path = job["text"]["prompt"], case / "prompt-text-only.txt", case / "encode.log"
    prompt_file.write_text(text, encoding="utf-8")
    command = [sys.executable, "-I", "-B", str(H3TURBO_ENCODE), "--case", str(view), "--prompt-file", str(prompt_file),
               "--orientation", parent["orientation"], "--seed", str(parent["seed"]), "--case-id", job["case_id"],
               "--package", str(PKG), "--upscale"]
    code, seen = gpu_process(command, dict(os.environ, LFSTUDIO_PATHS=PATHS_TEMPLATE), PKG, log_path, encode_line,
                             None, UPSCALE_ENCODE_LABELS, (UPSCALE_CONF["guard_gib"], "the upscale's prompt encode"),
                             UpscaleFailed, f"cannot start the prompt encode {H3TURBO_ENCODE}",
                             f"encoding the text of {parent['case_id']} for upscale {job['case_id']} (pid {{pid}})")
    if code not in (0, 2):                 # killed (the guard, a SIGKILL after Cancel or shutdown) or failed; a refusal
        stop_encode_worker(view)           # (2) started nothing
    what = "refused the text" if code == 2 else "failed"
    if seen.get("error"):
        seen["error"] = f"the prompt encode {what}: {seen['error']}"
    job_ended(code, seen, UpscaleFailed, "LFSTUDIO_UPSCALE_GUARD_GIB", UPSCALE_CONF["guard_gib"],
              f"the prompt encode {what} (exit code {code}); see {log_path}")
    try:
        receipt = json.loads((view / "qwen" / "conditioning.json").read_text())
    except (OSError, ValueError):
        receipt = None
    if not (view / "qwen" / "conditioning.pt").is_file() or not isinstance(receipt, dict):
        raise UpscaleFailed(f"the prompt encode wrote no qwen/conditioning.pt and .json; see {log_path}")
    differs = [key for key, want in (("task", "t2va"), ("prompt", text)) if receipt.get(key) != want]
    if differs:
        raise UpscaleFailed(f"the prompt encode's conditioning is not of the text asked for: its "
                            f"{' and '.join(differs)} differ{'s' * (len(differs) == 1)} (task {receipt.get('task')}); "
                            f"see {log_path}")
    try:
        summary = json.loads((view / "encode.json").read_text())
    except (OSError, ValueError) as error:
        log("no readable encode.json from the upscale's prompt encode:", error)
        summary = {}
    return summary


def upscale_floor(job, floor):
    """Raise UpscaleFailed when MemAvailable stays below the upscale's floor (regen.min_free_for: its case and size;
    enough_memory: it waits up to FREE_WAIT_S for closed workers' memory to come back)."""
    available = enough_memory(floor)
    if available is None and floor > 0:
        log(f"upscale: cannot read MemAvailable from {MEMINFO}; starting without the memory check")
    if available is not None and available < floor:
        setting = UPSCALE_CONF["min_free_gib"]
        what = ("LFSTUDIO_UPSCALE_MIN_FREE_GIB" if floor == setting
                else f"LFSTUDIO_UPSCALE_MIN_FREE_GIB plus {regen.FLOOR_4K_EXTRA_GIB:g} GiB at 4K"
                if floor == setting + regen.FLOOR_4K_EXTRA_GIB and job["resolution"] == "4k"
                else f"what a {UPSCALE_NAMES[job['resolution']]} {job.get('seconds', 5)} s upscale needs, more than "
                     f"LFSTUDIO_UPSCALE_MIN_FREE_GIB={setting:g}")
        raise UpscaleFailed(f"not enough free memory for an upscale: {available:.1f} GiB available with the models "
                            f"closed, {floor:g} GiB needed ({what}); close other apps")


def run_upscale(job):
    """An upscale (the docstring): free the memory and check it, then in a session directory of its own and under one
    pipeline_call (so that Cancel stops each step): its input folder (build_input: the clip's files linked, a LightX2V
    Turbo clip's MP4 stretched or cut, the pictures of the ends it keeps), for a clip without a conditioning of text
    alone the prompt encode (encode_upscale_text, status "encoding the prompt", then the memory check again: the Qwen
    worker's memory comes back), job.json ("case": the input folder; "pin" with the ends sent) and the job
    (upscale_process); then the poster, the record (with each end's result, from the job's receipt) and the publish
    copy. The clip, the job's offer of its size and its pictures are checked again first: an end whose picture is gone
    or changed since the request is not sent, with its reason. The memory floor is the job's for the clip's case, size
    and pin (regen.min_free_for), which job.json carries as "min_free_gib" (the job checks it again as it starts).
    Raises when it fails or is cancelled; no retry."""
    if job.get("cancel_requested"):
        raise Cancelled("cancelled before it started")
    job["status"] = "freeing memory"
    job["phase_since"] = time.time()
    abort_preload("an upscale needs the memory")
    close_session("an upscale needs the memory")
    if job.get("cancel_requested"):        # cancelled while the models were closed: nothing of its own ran
        raise Cancelled("cancelled before it started")
    with lock:
        parent = find_job(job["parent"])
        parent = dict(parent) if parent else None
    if parent is None:
        raise UpscaleFailed("the clip to upscale is gone")
    code, info = upscalable(parent)
    if code != 200:
        raise UpscaleFailed(f"the clip can't be upscaled now: {info}")
    if job["audio"] == "clip" and not info["clip_audio"]:
        raise UpscaleFailed(f"the clip's {AUDIO_FILE[job_model(parent)].split(' (')[0]} is gone")
    sources = upscale_sources(parent, job["resolution"], hash_now=bool((job.get("pin") or {}).get("ends")))
    why = offer_refusal(regen.offer(sources["frames"]["kept"], job["resolution"]), job["strength"])
    if why:
        raise UpscaleFailed(f"the clip can't be upscaled now: {why}")
    if (job.get("text") or {}).get("source") != sources["text"]:      # e.g. its qwen-text/ went since the request
        job["text"] = text_record(parent, sources["text"])
    job["video"], job["frames"] = sources["video"], sources["frames"]["kept"]
    if "pin" in job:                       # the ends sent at the request that can still be held
        plan = pin_plan(parent, sources, job["resolution"], job["strength"], job["pin"]["asked"]) or \
            {"ends": [], "why": {}}
        job["pin"] = {"asked": job["pin"]["asked"], "ends": [e for e in plan["ends"] if e in job["pin"]["ends"]],
                      "why": plan["why"]}
    floor = regen.min_free_for(job["resolution"], UPSCALE_CONF["min_free_gib"], sources["frames"]["kept"],
                               len((job.get("pin") or {}).get("ends", ())))      # by its case, size and pin
    upscale_floor(job, floor)              # before anything of its own: a start that can't make it starts nothing
    case = new_session_dir(f"upscale-{job['resolution']}") / job["case_id"]
    out = case / "regen"
    with lock:
        if stopping.is_set():
            raise UpscaleFailed("the studio is shutting down")
    out.mkdir(parents=True)
    job.setdefault("case_dirs", []).append(str(case))      # for Delete, even if it fails
    job_path = out / "job.json"
    t0 = time.time()

    def prepare_and_run():
        view, mp4, files = build_input(job, parent, sources, case)
        if job.get("cancel_requested"):
            raise Cancelled("cancelled while its input was prepared")
        if sources["text"] == "encode":
            job.update(status="encoding the prompt", phase_since=time.time(), upscale_phase="prompt")
            summary = encode_upscale_text(job, parent, case, view)
            job["text"].update(rows=(summary.get("encodes") or {}).get("qwen", {}).get("rows"),
                               encode_s=round(time.time() - job["phase_since"], 1))
            for name in ("conditioning.pt", "conditioning.json"):
                files[f"qwen/{name}"] = {"from": "encode", "how": "encode"}
            write_source(job, parent, view, files, sources)
            if job.get("cancel_requested"):    # Cancel came as the encode ended: the job is not started
                raise Cancelled("cancelled after the prompt encode")
            upscale_floor(job, floor)          # the Qwen worker's ~15 GiB back
        data = {"version": regen.JOB_VERSION, "case": str(view),
                "resolution": job["resolution"], "steps": job["steps"], "start_sigma": job["start_sigma"],
                "seed": job["refine_seed"], "audio": job["audio"], "load": job.get("load", UPSCALE_CONF["load"]),
                "preread": UPSCALE_CONF["preread"], "min_free_gib": floor,      # the job checks it again itself
                "paths": {"package": str(UPSCALE_CONF["package"]), "weights": str(UPSCALE_CONF["weights"]),
                          "base_model": str(UPSCALE_CONF["base_model"]), "output": str(out)},
                "meta": {"studio_job": job["id"], "parent": job["parent"], "strength": job["strength"]}}
        links = {Path(rel).stem: view / rel for rel in files if rel.startswith("keyframe/")}
        if (job.get("pin") or {}).get("ends"):        # the job's own check decides which ends it holds
            data["pin"] = {"keyframes": [
                {"anchor": end, "image": str(links[end]), "sha256": sources["keyframes"][end]["sha256"],
                 "fit": sources["keyframes"][end]["fit"], "canvas": sources["keyframes"][end]["canvas"]}
                for end in job["pin"]["ends"]]}
        regen.parse_job(data, out)         # the job's own check of job.json: a JobError (ValueError) is a bug here
        regen.write_json(job_path, data)
        job.update(status="upscaling", phase_since=time.time(), upscale_phase="starting")
        return upscale_process(job, out, job_path, mp4)

    final = pipeline_call(job, prepare_and_run, "upscale failed")
    make_poster(final)
    job.pop("upscale_phase", None)
    job.update(output=str(final), e2e_s=round(time.time() - t0, 1))
    try:
        receipt = json.loads((out / regen.RECEIPT).read_text())
        job.update(phases_s=receipt.get("times_s"), peak_gib=receipt.get("peak_allocated_gib"),
                   low_mem_gib=(receipt.get("lowest_mem_available_gib") or {}).get("all"))
        pinned = receipt.get("pin")
        if job.get("pin") and isinstance(pinned, dict):      # each end's result: held or not, and why
            job["pin"]["result"] = {end: {key: pinned[end].get(key) for key in ("used", "agreement_db", "why")}
                                    for end in regen.ANCHORS if isinstance(pinned.get(end), dict)}
    except (OSError, ValueError, AttributeError) as error:
        log("no readable receipt from the upscale:", error)
    stem = Path(parent["published"]).stem if parent.get("published") else publish_stem(parent)
    publish(job, final, f"{stem}_{job['resolution']}")      # next to the clip's own copy


UPSCALE_LABELS = ("the upscale's job", "the running upscale")      # in the log: for Cancel, for shutdown


def upscale_process(job, out, job_path, sound):
    """The upscale's job (LFSTUDIO_UPSCALE_SCRIPT: upscale/regen2k_job.py) with the Regen-2K Python (gpu_process:
    its progress as job["upscale_phase"], the guard LFSTUDIO_UPSCALE_GUARD_GIB). The job ends when this process does
    (--parent: PR_SET_PDEATHSIG), so a server killed hard does not leave it running unguarded. Then the clip's sound
    goes in, copied from `sound`, the input folder's MP4 (the clip's own, or its prepared copy, whose sound is the
    clip's as it is, whole, or cut to the 121 frames kept for a 5 s clip that is cut at 4K). The MP4 is named by its
    size and the frames kept: regen_<W>x<H>_<frames>f.mp4 (121 for a 5 s clip, 241 for a 10 s one). Returns it; raises
    with the job's own error line."""
    log_path = out / "worker.log"

    def on_phase(name):
        job["upscale_phase"] = name[:80]
    code, seen = gpu_process(
        regen.worker_command(UPSCALE_CONF["python"], UPSCALE_SCRIPT, job_path, os.getpid()),
        regen.worker_environment(os.environ), out, log_path, regen.parse_line, on_phase, UPSCALE_LABELS,
        (UPSCALE_CONF["guard_gib"], "upscale"), UpscaleFailed,
        f"cannot start the upscale job with LFSTUDIO_UPSCALE_PYTHON={UPSCALE_CONF['python']}",
        f"upscaling {job['case_id']} (job pid {{pid}}): {job['resolution']}, {job['strength']}, audio {job['audio']}")
    job_ended(code, seen, UpscaleFailed, "LFSTUDIO_UPSCALE_GUARD_GIB", UPSCALE_CONF["guard_gib"],
              f"the upscale job exited with code {code}; see {log_path}")
    silent = out / regen.OUTPUT_VIDEO
    if not silent.is_file():
        raise UpscaleFailed(f"the upscale job wrote no {silent.name}; see {log_path}")
    width, height = regen.output_size(job.get("orientation", "landscape"), job["resolution"])
    final = out / f"regen_{width}x{height}_{job.get('frames') or regen.NVIDIA_FRAMES}f.mp4"
    partial = out / f".{final.stem}.part.mp4"
    job["upscale_phase"] = "adding the sound"
    try:
        mux = subprocess.run(regen.mux_command(silent, sound, partial), capture_output=True, text=True,
                             timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        raise UpscaleFailed(f"could not add the clip's sound: {error}") from None
    if mux.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise UpscaleFailed(f"could not add the clip's sound (ffmpeg: {regen.one_line(mux.stderr)[-300:]})")
    os.replace(partial, final)
    silent.unlink()                        # the job's MP4 without sound (CRF 10, large): the muxed one replaces it
    return final


def gpu_process(command, env, cwd, log_path, parse_line, on_phase, labels, guard, failure, cannot_start, announce):
    """Run one GPU process of a job (an upscale's job, a LightX2V Turbo clip's prompt encode or its job) to its end:
    in a process group of its own (stop_job's SIGTERM reaches it as it reaches the pipeline's workers; it is
    work["process"], so stop_job and stop_gpu_job send SIGKILL if it outlives the SIGTERM by GPU_JOB_STOP_S), its
    output in log_path. A thread follows that log (job_tail: parse_line's phases go to on_phase, its errors to
    seen["error"]), another stops the process when memory runs out (memory_guard; guard = (floor GiB, who, for the
    log)). `labels`: its names in the log, for Cancel and for shutdown; `announce`: the log line once it runs ({pid}).
    Returns (its exit status, seen); raises `failure` with cannot_start when it can't be started."""
    with open(log_path, "ab") as log_file:
        try:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log_file,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            raise failure(f"{cannot_start} ({error.strerror or error})") from None
    with lock:                             # atomic with stop_gpu_job: it stops this process, or it is stopped here
        work["process"], work["labels"] = proc, labels
        late = stopping.is_set()
    if late:                               # the server began to shut down while it started
        log(f"the studio is shutting down: stopping {labels[0]} it just started")
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    log(announce.format(pid=proc.pid))
    halt, seen = threading.Event(), {}
    threads = [threading.Thread(target=job_tail, args=(log_path, parse_line, on_phase, halt, seen), daemon=True),
               threading.Thread(target=memory_guard, args=(proc, halt, seen, *guard), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        code = proc.wait()
    finally:
        halt.set()
        for thread in threads:
            thread.join(timeout=15)
        with lock:
            work["process"] = work["labels"] = None
    return code, seen


def job_ended(code, seen, failure, guard_setting, guard_gib, otherwise):
    """Raise `failure` for a GPU process that did not end well (gpu_process's exit status and seen): stopped by the
    guard, by the studio's shutdown, or with its own error line (else `otherwise`)."""
    if "guard" in seen:
        raise failure(f"stopped: MemAvailable was below {guard_gib:g} GiB for 3 s ({seen['guard']:.1f} GiB; "
                      f"{guard_setting})")
    if code and stopping.is_set():
        raise failure("the studio was stopped")
    if code:
        raise failure(seen.get("error") or otherwise)


def job_tail(path, parse_line, on_phase, halt, seen):
    """Follow a GPU job's log: for each line, parse_line gives ("phase", name), which goes to on_phase, or ("error",
    text), the last of which is seen["error"]. Reads what is left once more after `halt` is set."""
    pos, rest, chunk = 0, b"", 1 << 20
    while True:
        last = halt.is_set()
        try:
            with open(path, "rb") as log_file:
                log_file.seek(pos)
                data = log_file.read(chunk)
        except OSError:
            data = b""
        pos += len(data)
        *lines, rest = (rest + data).split(b"\n")
        rest = rest[-4096:]                # a line without an end (a progress bar) can't grow without bound
        for line in lines:
            found = parse_line(line.rsplit(b"\r", 1)[-1].decode("utf-8", "replace"))
            if found and found[0] == "phase":
                on_phase and on_phase(found[1])
            elif found:
                seen["error"] = found[1]
        if len(data) == chunk:             # more to read now
            continue
        if last:
            return
        halt.wait(0.25)


def memory_guard(proc, halt, seen, floor, who):
    """Stop a GPU job's process when MemAvailable stays below `floor` GiB for 3 s (its setting, e.g.
    LFSTUDIO_UPSCALE_GUARD_GIB): SIGKILL to its process group. On the DGX Spark's unified memory a job that takes the
    last few GiB can hang the machine, and this must fire before the launcher's guard (below 4 GiB for 3 s), which
    would stop the whole studio. So not SIGTERM, as bench/memguard-run.sh sends to a process without a handler: the
    job's handler runs only once the GPU call under way returns, and the launcher's seconds keep counting meanwhile.
    Sets seen["guard"] (the MemAvailable it saw), the reason the server records (the job's receipt stays at
    "running"); ends when `halt` is set. `who` names the job in the log."""
    low = 0
    while floor > 0 and not halt.wait(1):
        available = mem_available_gib()
        low = low + 1 if available is not None and available < floor else 0
        if low < 3:
            continue
        seen["guard"] = available
        log(f"{who}: MemAvailable {available:.1f} GiB, below {floor:g} GiB for 3 s: stopping its job (SIGKILL)")
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        return


def stop_gpu_job():
    """At shutdown: SIGTERM a running GPU job's process (its process group; SIGKILL if it is still there
    GPU_JOB_STOP_S later), so that it does not go on without the studio. (A job's --parent ends it if the server dies
    first, and bin/lf-studio's cleanup stops one left behind too: WORKER_RE.)"""
    with lock:
        proc, labels = work["process"], work["labels"]
    if proc is None or proc.poll() is not None:
        return
    log(f"stopping {labels[1]}")
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=GPU_JOB_STOP_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


# ------------------------------------------------------------------------------------------ clips of a job model
TURBO_LABELS = ("the LightX2V Turbo job", "the running LightX2V Turbo clip")
ENCODE_LABELS = ("the LightX2V Turbo prompt encode", "the running LightX2V Turbo prompt encode")
ENCODE_LINE = re.compile(r"\[h3turbo-encode\] (failed|refused): (.*)")
NO_TEXT_ONLY = "--no-text-only"     # the encode's flag that skips the text-only input; its refusal names it


def encode_line(line):
    """A line of the prompt encode's output -> ("error", text) for its failure or refusal, else None."""
    m = ENCODE_LINE.match(line.strip())
    return ("error", m.group(2).strip()) if m else None


def model_options(model, opts, orientation):
    """A clip's model_opts at submit: its options (check_model) and, from the option that has sizes, its size."""
    out = dict(opts)
    for option in MODELS[model]["options"]:
        choice = next((c for c in option["choices"] if c["value"] == opts.get(option["name"])), {})
        if "size" in choice:
            out["size"] = list(choice["size"][orientation])
    return out


def eta_key(model, opts, seconds):
    """The key of a clip's expected time: its options' values in the registry's order, then its length (576p/4/5)."""
    return "/".join([str(opts.get(o["name"])) for o in MODELS[model]["options"]] + [str(seconds)])


def read_times():
    try:
        times = json.loads(LOAD_TIMES.read_text())
        return times if isinstance(times, dict) else {}
    except (OSError, ValueError):
        return {}


def remember_job_time(job, seconds):
    """Record how long a clip of a job model took, from its prompt encode to its saved MP4 (the Sol-H3 close before
    it not counted), as job/<model>/<eta_key> in load-times.json. Never "last": that is Sol-H3's cold start."""
    try:
        times = read_times()
        times[f"job/{job['model']}/{eta_key(job['model'], job['model_opts'], job['seconds'])}"] = seconds
        LOAD_TIMES.write_text(json.dumps(times, indent=1))
    except (OSError, ValueError):
        traceback.print_exc()


def model_eta(model):
    """A job model's expected seconds per clip: {eta_key: {"s", "measured"}}, its last measured time where there is
    one (remember_job_time), else the registry's estimate. {} for sol-h3 (the page has its own times)."""
    times, out = read_times(), {}
    for key, estimate in MODELS[model].get("estimates_s", {}).items():
        measured = times.get(f"job/{model}/{key}")
        out[key] = {"s": measured, "measured": True} if measured else {"s": estimate, "measured": False}
    prefix = f"job/{model}/"
    for name, value in times.items():
        if name.startswith(prefix) and name[len(prefix):] not in out:
            out[name[len(prefix):]] = {"s": value, "measured": True}
    return out


def encode_prompt(job, case, conf):
    """A LightX2V Turbo clip's prompt conditioning: its prompt encode (LFSTUDIO_H3TURBO_ENCODE_SCRIPT, by default
    bench/h3turbo_encode.py) with this server's Python in the patched package, as a GPU process of its
    own (gpu_process: Cancel and shutdown stop it, and it stops its transient Qwen worker): the prompt and the
    keyframes, seen at the LFSTUDIO_H3TURBO_QWEN_CANVAS canvas, into <case>/qwen/, and for fl2va the prompt without its
    keyframe references into <case>/qwen-text/ (the input of a later Upscale; job["text_only"]). A prompt that is
    nothing but its keyframe references has no such text: the encode refuses it, naming --no-text-only, and runs again
    with it (text_only null). After an encode that ended badly, stop_encode_worker. Its log is <case>/encode.log.
    Returns (the conditioning, the encode's summary: encode.json)."""
    opts, prompt_file, log_path = job["model_opts"], case / "prompt.txt", case / "encode.log"
    prompt_file.write_text(job["prompt"], encoding="utf-8")
    command = [sys.executable, "-I", "-B", str(H3TURBO_ENCODE), "--case", str(case), "--prompt-file", str(prompt_file),
               "--orientation", job["orientation"], "--resolution", opts["resolution"], "--canvas", conf["qwen_canvas"],
               "--seed", str(job["seed"]), "--case-id", job["case_id"], "--package", str(PKG)]
    for anchor in ("first", "last"):
        if job.get(f"{anchor}_frame"):
            command += [f"--{anchor}", job[f"{anchor}_frame"]]
    env = dict(os.environ, LFSTUDIO_PATHS=PATHS_TEMPLATE)

    def encode(command):
        code, seen = gpu_process(command, env, PKG, log_path, encode_line, None, ENCODE_LABELS,
                                 (conf["guard_gib"], "LightX2V Turbo's prompt encode"), ModelJobFailed,
                                 f"cannot start the prompt encode {H3TURBO_ENCODE}",
                                 f"encoding the prompt of {job['case_id']} (pid {{pid}}): Qwen canvas "
                                 f"{conf['qwen_canvas']}")
        if code not in (0, 2):             # killed (the guard, a SIGKILL after Cancel or shutdown) or failed; a
            stop_encode_worker(case)       # refusal (2) started nothing
        return code, seen

    code, seen = encode(command)
    text_only = job["task"] == "fl2va"
    if code == 2 and text_only and NO_TEXT_ONLY in (seen.get("error") or ""):
        # The prompt is nothing but its keyframe references (the guide's first-frame line, the page's alignLine()):
        # nothing is left for the text-only encode, which only a later Upscale reads, and the encode refused before
        # it started anything. The clip itself needs no text-only input: encode it without one (text_only null).
        log(f"{job['case_id']}: the prompt encode refused the text-only input ({seen['error']}): encoding the clip "
            "without it")
        if job.get("cancel_requested"):
            raise Cancelled("cancelled during the prompt encode")
        text_only = False
        code, seen = encode(command + [NO_TEXT_ONLY])
    what = "refused the clip" if code == 2 else "failed"
    if seen.get("error"):
        seen["error"] = f"the prompt encode {what}: {seen['error']}"
    job_ended(code, seen, ModelJobFailed, "LFSTUDIO_H3TURBO_GUARD_GIB", conf["guard_gib"],
              f"the prompt encode {what} (exit code {code}); see {log_path}")
    conditioning = case / "qwen" / "conditioning.pt"
    if not conditioning.is_file():
        raise ModelJobFailed(f"the prompt encode wrote no {conditioning.relative_to(case)}; see {log_path}")
    try:
        summary = json.loads((case / "encode.json").read_text())
    except (OSError, ValueError) as error:
        log("no readable encode.json from the prompt encode:", error)
        summary = {}
    if (case / "qwen-text" / "prompt.txt").is_file():
        try:
            dropped = json.loads((case / "qwen-text" / "text-only.json").read_text()).get("dropped")
        except (OSError, ValueError, AttributeError):
            dropped = None
        job["text_only"] = {"prompt": (case / "qwen-text" / "prompt.txt").read_text(encoding="utf-8"),
                            "dropped": dropped}
    elif job["task"] == "fl2va" and not text_only:
        job["text_only"] = None            # no text without the keyframe references: it can't be upscaled later
    return conditioning, summary


def stop_encode_worker(case):
    """After a prompt encode that ended badly: stop the transient Qwen worker it may have left. runtime.pipeline.Worker
    starts that worker (qwen_python.sh, its Docker container) in a session of its own as the encode's child, so it is
    not in the encode's process group (the guard's and Cancel's SIGKILL) and not a child of this server
    (signal_workers). An encode killed that way can't close it, and it lives on, with Qwen's ~17 GiB, until it next
    reads its stdin: after its start (6-13 s) or the encode under way, while the next clip may load already. Found by
    its command line (runtime.worker --module qwen --options <case>/qwen-worker/options.json, the case as given or
    resolved, as bin/lf-studio's cleanup finds the pipeline's workers), its group gets SIGTERM once: qwen_python.sh
    stops its container in a TERM trap, which a second signal would cut short. An encode that closed its worker
    itself leaves nothing to find. Returns the groups signalled."""
    wanted = {str(root / "qwen-worker" / "options.json") for root in (case, case.resolve())}
    sent = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
            pgrp = int((d / "stat").read_text().rsplit(")", 1)[1].split()[2])
        except (OSError, ValueError, IndexError):
            continue
        after = dict(zip(argv, argv[1:]))   # each argument -> the one after it
        if (pgrp == int(d.name) and "runtime.worker" in argv and after.get("--module") == "qwen"
                and after.get("--options") in wanted):
            log(f"stopping the Qwen worker the prompt encode left (process group {pgrp})")
            with contextlib.suppress(OSError):
                os.killpg(pgrp, signal.SIGTERM)
                sent.append(pgrp)
    return sent


def turbo_job_data(job, out, conf, module, conditioning):
    """A LightX2V Turbo clip's job.json (version 1 of jobs/h3turbo_job.py): its settings, the LoRA and shifts of its
    steps (the job's PRESETS), the conditioning, its keyframes, the base model and where its files go."""
    opts = job["model_opts"]
    preset = module.PRESETS[opts["steps"]]
    return {"version": module.JOB_VERSION, "task": job["task"], "orientation": job["orientation"],
            "resolution": opts["resolution"], "seconds": job["seconds"], "seed": job["seed"], "steps": opts["steps"],
            "lora": str(conf["loras"] / preset["file"]), "shift": list(preset["shift"]),
            "conditioning": str(conditioning), "qwen_canvas": conf["qwen_canvas"],
            "first_frame": job.get("first_frame"), "last_frame": job.get("last_frame"), "audio_noise": "seed",
            "crf": 19, "x264_preset": "veryfast", "load": "fast", "abort_below_gib": H3TURBO_ABORT_GIB,
            "paths": {"base_model": str(conf["base_model"]), "output": str(out)}, "meta": {"studio_job": job["id"]}}


def video_size(mp4):
    """An MP4's (width, height) (ffprobe), or None when ffprobe can't read it."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(mp4)], capture_output=True, text=True, timeout=60)
        width, height = map(int, r.stdout.strip().split(",")[:2])
        return width, height
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def run_model_job(job):
    """A clip of a job model, LightX2V Turbo (the docstring): close the Sol-H3 session and check the memory floor, then
    in one pipeline_call (so that Cancel stops either step and answers 409 only once the clip is being saved) its
    prompt encode (encode_prompt) and its job (gpu_process) in <session>/<case_id>/turbo/, then the poster, the
    record, its time and the publish copy. Raises when it fails or is cancelled; no retry, and no warm-up after it."""
    model = job_model(job)
    spec, conf, module = MODELS[model], MODEL_CONF[model], JOB_MODULES[model]
    name, setting, opts = spec["name"], f"LFSTUDIO_{spec['setting']}", job["model_opts"]
    if job.get("cancel_requested"):
        raise Cancelled("cancelled before it started")
    job["status"] = "freeing memory"
    job["phase_since"] = time.time()
    abort_preload(f"{name} needs the memory")
    close_session(f"{name} needs the memory")
    if job.get("cancel_requested"):        # cancelled while the models were closed: nothing of its own ran
        raise Cancelled("cancelled before it started")
    floor = module.min_free_for(opts["resolution"], job["seconds"], conf["min_free_gib"])
    available = enough_memory(floor)
    if available is None and floor > 0:
        log(f"{name}: cannot read MemAvailable from {MEMINFO}; starting without the memory check")
    if available is not None and available < floor:
        # the setting is a minimum: a clip needs at least what its job's model of a clip says for its size and length
        why = f"{setting}_MIN_FREE_GIB" if floor == conf["min_free_gib"] else \
            (f"what a {opts['resolution']} {job['seconds']} s clip needs, more than {setting}_MIN_FREE_GIB="
             f"{conf['min_free_gib']:g}")
        raise ModelJobFailed(f"not enough free memory for a {name} clip: {available:.1f} GiB available with the "
                             f"models closed, {floor:g} GiB needed ({why}); close other apps")
    case = new_session_dir(f"{model}-{opts['resolution']}-{job['task']}-{job['orientation']}-{job['seconds']}s") \
        / job["case_id"]
    out = case / "turbo"
    with lock:
        if stopping.is_set():
            raise ModelJobFailed("the studio is shutting down")
    out.mkdir(parents=True)
    job.setdefault("case_dirs", []).append(str(case))      # for Delete, even if it fails
    opts.update(lora=module.PRESETS[opts["steps"]]["file"], qwen_canvas=conf["qwen_canvas"])
    t0 = time.time()

    def on_phase(phase):
        job["job_phase"] = phase[:80]
        if phase.startswith("step") and job["status"] == "loading models + generating":
            job.update(status="generating", phase_since=time.time())

    def encode_and_run():
        if job.get("cancel_requested"):    # cancelled during the memory check: nothing of its own ran yet
            raise Cancelled("cancelled before it started")
        job.update(status="encoding the prompt", phase_since=time.time())
        conditioning, summary = encode_prompt(job, case, conf)
        job["encode_s"] = round(time.time() - t0, 1)
        if isinstance(summary.get("image_tokens"), int):
            opts["image_tokens"] = summary["image_tokens"]
        if job.get("cancel_requested"):    # Cancel came as the encode ended: the job is not started
            raise Cancelled("cancelled after the prompt encode")
        data = turbo_job_data(job, out, conf, module, conditioning)
        parsed = module.parse_job(data, out)   # the job's own check of job.json: a JobError (ValueError) is a bug here
        job_path = out / "job.json"
        module.write_json(job_path, data)
        job.update(status="loading models + generating", phase_since=time.time(), job_phase="starting")
        log_path = out / "worker.log"
        code, seen = gpu_process(
            module.worker_command(conf["python"], H3TURBO_SCRIPT, job_path, os.getpid()),
            module.worker_environment(os.environ), out, log_path, module.parse_line, on_phase, TURBO_LABELS,
            (conf["guard_gib"], name), ModelJobFailed,
            f"cannot start the {name} job with {setting}_PYTHON={conf['python']}",
            f"{name}: {job['case_id']} (job pid {{pid}}): {job['task']}, {job['orientation']}, {job['seconds']} s, "
            f"{opts['resolution']}, {opts['steps']} steps")
        if code == 3 and "guard" not in seen and not stopping.is_set():
            raise ModelJobFailed(f"stopped by its job: MemAvailable fell below {H3TURBO_ABORT_GIB:g} GiB"
                                 + (f" ({seen['error']})" if seen.get("error") else ""))
        job_ended(code, seen, ModelJobFailed, f"{setting}_GUARD_GIB", conf["guard_gib"],
                  f"the {name} job exited with code {code}; see {log_path}")
        try:
            receipt = json.loads((out / module.RECEIPT).read_text())
        except (OSError, ValueError) as error:
            log(f"no readable receipt from the {name} job:", error)
            receipt = {}
        output = receipt.get("output") if isinstance(receipt.get("output"), dict) else {}
        mp4 = out / (output.get("file") or module.output_name(parsed))
        if not mp4.is_file():
            raise ModelJobFailed(f"the {name} job wrote no {mp4.name}; see {log_path}")
        size = video_size(mp4)
        if size is None:
            raise ModelJobFailed(f"the {name} job's {mp4.name} can't be read; see {log_path}")
        if opts.get("size") and list(size) != opts["size"]:
            raise ModelJobFailed(f"the {name} job's {mp4.name} is {size[0]}x{size[1]}, not "
                                 f"{opts['size'][0]}x{opts['size'][1]}")
        return mp4, receipt

    mp4, receipt = pipeline_call(job, encode_and_run, f"{name} failed")
    make_poster(mp4)
    job.pop("job_phase", None)
    job.update(output=str(mp4), e2e_s=round(time.time() - t0, 1), includes_model_load=True)
    lows = receipt.get("lowest_mem_available_gib")
    job.update(phases_s=receipt.get("times_s"), load_s=receipt.get("load_s"),
               peak_gib=receipt.get("peak_allocated_gib"),
               low_mem_gib=lows.get("all") if isinstance(lows, dict) else None)
    remember_job_time(job, job["e2e_s"])
    publish(job, mp4, publish_stem(job))


def warm_up_after_upscale():
    """After an upscale none of the studio's models are loaded. With LFSTUDIO_PRELOAD=1 and nothing queued, warm up
    the last clip's settings again, as at start."""
    if not PRELOAD:
        return
    with lock:
        busy = (stopping.is_set() or preload["pending"] or session["pipeline"] is not None
                or any(j["status"] == "queued" for j in jobs))
        pending = None if busy else preload_candidate()
        if pending:
            preload["pending"] = pending
            session["switching_to"] = pending[0]
    if pending:
        log("will preload", pending[0], "again after the upscale")
        wake.set()


def next_job():
    """The job to run next: the first clip queued, else the first upscale queued. An upscale closes the models, so it
    waits while any clip is queued (runs_before). Call it holding `lock`."""
    return min((j for j in jobs if j["status"] == "queued"), key=lambda j: (is_upscale(j), j["created"]),
               default=None)


def worker_loop():
    purged = time.time()                           # main() purged at start
    while not stopping.is_set():
        wake.wait(5)
        wake.clear()
        if stopping.is_set():
            break
        if time.time() - purged > 86400:
            purged = time.time()
            purge_trash()
        # a requested warm-up (at start, or "Load selected now"): session["switching_to"] shows it until it starts.
        # A clip submitted, an Unload or a newer request meanwhile cancels or replaces it (they clear or change
        # switching_to), also while the old models are being closed for it, which takes ~40 s.
        with lock:
            pending, preload["pending"] = preload["pending"], None
        if pending:
            if session["pipeline"] is not None and session["key"] != pending[0]:
                if loaded_pipeline() is not None and same_models(session["key"], pending[0]):
                    reshape_session(pending[0], "as asked")         # keeps the models: no warm-up pass
                else:
                    close_session(f"loading {pending[0]} as asked")
            with lock:
                go = (session.get("switching_to") == pending[0] and not preload["pending"] and not stopping.is_set()
                      and session["pipeline"] is None and not any(j["status"] == "queued" for j in jobs))
                if session.get("switching_to") == pending[0] and not preload["pending"]:
                    session["switching_to"] = None
                # the case is chosen again now (the clip it came from may have been deleted during the close), and
                # the warm-up gets its own links to the input files
                case = own_inputs(mode_warmup_case(pending[0][0]) or {}) if go else None
                if go and case and case.get("prompt"):
                    session.update(preloading=True, loading_key=pending[0], state="loading", since=time.time())
                else:
                    go = False
            if go:
                run_preload(pending[0], case)
                shutil.rmtree(WARMUP_INPUTS, ignore_errors=True)   # read by the loading pass only
        while True:
            with lock:
                job = None if stopping.is_set() else next_job()
                if job is None:
                    break
                job["status"] = "starting"
                job["started"] = time.time()
            try:
                if is_upscale(job):
                    run_upscale(job)             # no retry: a failed upscale most likely ran out of memory
                elif job_model(job) != SOL_H3:
                    run_model_job(job)           # no retry either: memory or a bug won't fix itself
                else:
                    try:
                        run_job(job)
                    except Exception as e:   # worker died -> session was closed; rebuild it and retry this job once
                        if job.get("cancel_requested"):        # cancelled: no retry (run_job closed the session)
                            raise
                        traceback.print_exc()
                        job["first_error"] = f"{type(e).__name__}: {e}"[:600]
                        job["status"] = "retrying after a worker error"
                        log("retrying job", job["id"], "after:", job["first_error"][:200])
                        run_job(job)
                final = "done"
            except BaseException as e:
                if job.get("cancel_requested"):
                    final = "cancelled"
                    log("cancelled job", job["id"], f"({type(e).__name__}: {e})"[:200])
                else:
                    final = "failed"
                    job["error"] = f"{type(e).__name__}: {e}"[:600]
                    traceback.print_exc()
            job["finished"] = time.time()
            job["wall_s"] = round(job["finished"] - job["started"], 1)
            job["status"] = final           # after finished and wall_s: whoever sees a final status sees them too
            for k in ("saving", "cancel_requested"):     # only while it runs; after the status, so Cancel keeps
                job.pop(k, None)                         # answering 409 meanwhile
            save_job(job)                   # wakes GET /api/jobs/<id>?wait=
            if is_upscale(job):
                warm_up_after_upscale()     # (none after a clip of a job model: it is what the user is making)


# ------------------------------------------------------------------------------------------ HTTP
PAGE_FILE = Path(__file__).resolve().parent / "index.html"   # read per request: page edits need no restart
PAGE_BACKEND = '<meta name="lf-studio-backend" content="spark">'   # the page's texts follow it (and /api/state)


def page_html():
    """The page, naming this studio's backend in its lf-studio-backend meta tag, so its first paint already has the
    machine's texts, and with more than one model on the model registry after it (public_models, as in /api/state)
    in <script type="application/json" id="lf-models">. On the Spark with sol-h3 alone it goes out unchanged."""
    meta = PAGE_BACKEND.replace('"spark"', f'"{BACKEND}"')
    page = PAGE_FILE.read_text().replace(PAGE_BACKEND, meta, 1)
    if MULTI:                              # "<" escaped, so no text in it can end the script element
        table = json.dumps(public_models()).replace("<", "\\u003c")
        page = page.replace(meta, f'{meta}\n<script type="application/json" id="lf-models">{table}</script>', 1)
    return page


def parse_multipart(ctype, body):
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        fn = part.get_filename()
        data = part.get_payload(decode=True) or b""
        if fn:
            if data:
                files.setdefault(name, []).append((fn, data))
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def save_upload(job_dir, fn, data, allowed):
    ext = Path(fn).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"unsupported file type {ext} (allowed: {', '.join(sorted(allowed))})")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(fn).stem)[:40] or "file"
    p = job_dir / f"{safe}{ext}"
    i = 1
    while p.exists():
        p = job_dir / f"{safe}_{i}{ext}"
        i += 1
    p.write_bytes(data)
    return str(p)


# The form's file inputs ("zones") and where a job keeps their files
ZONES = ("first_frame", "last_frame", "ref_images", "ref_videos", "ref_audio")
REF_TYPE = {"ref_images": "image", "ref_videos": "video", "ref_audio": "audio"}


def zone_files(job, zone):
    """The input files of `job` for one form zone, in order (reference videos: the normalized copy the run used).
    Nothing for an upscale: its only input is its clip."""
    if is_upscale(job):
        return []
    if zone in REF_TYPE:
        return [r["path"] for r in job.get("references") or [] if r.get("type") == REF_TYPE[zone]]
    return [job[zone]] if job.get(zone) else []


STILL = "@still.jpg"     # a clip's card picture, in the clip's input directory (uploads never have an @)
STILL_PART = ".still-*.part"   # one being made, in INPUTS itself; one left by a server stopped part-way goes at start
PICTURE_CACHE = "private, max-age=86400"   # /still and /poster: they never change; the user's own pictures, so only
                                           # the browser keeps them, not a shared cache in front of the studio


def card_picture(job):
    """The picture a queued or running clip's card shows behind its timer (#30): fl2va its first frame, else its last;
    ref2va its first reference image, else the first frame of its first reference video. It is a small JPEG, at most
    640 px wide (the card shows it blurred, and an upload can be a photo of many MB), made on the first request and
    kept in the clip's input directory (INPUTS/<id>), so Delete, Undo and the trash purge move it with the inputs.
    An image goes out as it is if ffmpeg can't make the JPEG, or if the clip has no input directory of its own (an
    old history). None for t2va, when the file is gone, or when a video's frame can't be read. While ffmpeg makes the
    JPEG, it is a temporary file in INPUTS itself, not in the clip's directory, which a Delete can move meanwhile (the
    JPEG is then not kept, and the moved image is not served). Never writes elsewhere. None for an upscale."""
    src, video = None, False
    if is_upscale(job):
        return None
    if job.get("task") == "fl2va":
        src = job.get("first_frame") or job.get("last_frame")
    elif job.get("task") == "ref2va":
        refs = job.get("references") or []
        ref = next((r for kind in ("image", "video") for r in refs if r.get("type") == kind), None)
        src, video = (ref["path"], ref["type"] == "video") if ref else (None, False)
    if not src or not Path(src).is_file():
        return None
    original = None if video else Path(src)
    still = INPUTS / job["id"] / STILL
    if still.is_file():
        return still
    if not still.parent.is_dir():                      # no input directory of its own (an old history): no JPEG
        return original
    tmp = INPUTS / STILL_PART.replace("*", uuid.uuid4().hex)    # two requests at once: each its own, the last wins
    try:            # -nostdin: runs at once (a page asks for every card's picture) could leave a terminal without echo
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-frames:v", "1", "-vf",
                        "scale='min(640,iw)':-2", "-q:v", "4", "-f", "mjpeg", str(tmp)],
                       capture_output=True, timeout=60, check=True)
        os.replace(tmp, still)                          # FileNotFoundError if Delete has moved the directory
    except (OSError, subprocess.SubprocessError):
        return original if original and original.is_file() else None
    finally:
        tmp.unlink(missing_ok=True)
    return still


def reuse_input(job_dir, path):
    """Put an earlier job's input file into this job's directory: a hardlink (no extra space; deleting either job
    later leaves the other's file alone), or a copy across filesystems."""
    src = Path(path)
    if not src.is_file():
        raise ValueError(f"{src.name} from the remixed clip is gone; attach it again")
    dst = job_dir / src.name
    i = 1
    while dst.exists():
        dst = job_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


REF_VIDEO_AREA = 672 * 384   # the draft's pixel area (the patch also prepares reference videos at this canvas)


def prepare_reference_video(path):
    """Normalize an uploaded reference video: downscale (never upscale) to ~672x384 area, 24 fps, <=15 s, keep
    audio, reject <2 s. The token-cost fix itself is in the patch (reference videos are resized to the draft
    canvas in stage1 and in the Qwen worker); this keeps uploads small and decoding cheap."""
    src = Path(path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height:format=duration", "-of", "json", str(src)],
                           capture_output=True, text=True, timeout=60)
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"{src.name}: no video stream")
    dur = float((info.get("format") or {}).get("duration") or 0)
    if dur and dur < 2.0:
        raise ValueError(f"{src.name}: reference videos must be at least 2 s (H3 limit)")
    out = src.with_name(src.stem + "_ref.mp4")
    vf = (f"scale='trunc(iw*min(1,sqrt({REF_VIDEO_AREA}/(iw*ih)))/2)*2':"
          f"'trunc(ih*min(1,sqrt({REF_VIDEO_AREA}/(iw*ih)))/2)*2':flags=lanczos,fps=24")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-t", "15", "-vf", vf,
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                        str(out)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.is_file():
        raise ValueError(f"{src.name}: could not read this video ({r.stderr.strip()[:200]})")
    return str(out)


def has_sound(path):
    """Whether a video has a sound track (ffprobe; one that can't be read counts as silent)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def job_labels(job):
    """What each input file of an fl2va or ref2va job is called in its prompt (#29), in the pipeline's numbering:
    [{"label", "zone", "index", "name", "sound_of"?}], zone and index as /api/inputs/<id> lists the file. fl2va: the
    first frame is <Picture 1>, the last frame the next picture. ref2va, in the order of its references (images,
    then videos, then audio): an image is the next <Picture N>; a video the next <Video N>, and if it has sound, its
    sound track takes the next <Audio N> just before it (sound_of names the video); an audio file the next <Audio N>.
    Reference videos are probed as stored (the normalized copy the pipeline reads). [] for t2va and for an upscale."""
    labels, count, index = [], {"Picture": 0, "Video": 0, "Audio": 0}, {}
    if is_upscale(job):
        return labels

    def add(kind, zone, path, **extra):
        count[kind] += 1
        labels.append(dict(label=f"<{kind} {count[kind]}>", zone=zone, index=index.get(zone, 0),
                           name=Path(path).name, **extra))

    if job["task"] == "fl2va":
        for zone in ("first_frame", "last_frame"):
            if job.get(zone):
                add("Picture", zone, job[zone])
    elif job["task"] == "ref2va":
        zones = {kind: zone for zone, kind in REF_TYPE.items()}
        for ref in job.get("references") or []:
            zone = zones[ref["type"]]
            if ref["type"] == "image":
                add("Picture", zone, ref["path"])
            elif ref["type"] == "video":
                if has_sound(ref["path"]):
                    add("Audio", zone, ref["path"], sound_of=f"<Video {count['Video'] + 1}>")
                add("Video", zone, ref["path"])
            else:
                add("Audio", zone, ref["path"])
            index[zone] = index.get(zone, 0) + 1
    return labels


IMG = {".png", ".jpg", ".jpeg", ".webp"}
AUD = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
VID = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
MAX_UPLOAD = 200 * 1024 * 1024
SMALL_BODY = 64 * 1024           # the JSON bodies of /api/cancel, /api/delete, /api/undelete, /api/load, /api/unload
PROMPT_MAX = 32768               # characters
REF_MAX = {"image": 9, "video": 3, "audio": 3}    # H3's limits for ref2va
REF_TOTAL = 12
WAIT_MAX_S = 300                 # the longest wait of GET /api/jobs/<id>?wait=
# a JSON data: URI's type -> the file extension that decides how it is used, as a form upload's name does
MIME_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp",
            "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm", "video/x-matroska": ".mkv",
            "video/x-m4v": ".m4v", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
            "audio/vnd.wave": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
            "audio/x-flac": ".flac", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/ogg": ".ogg"}
FRAMES = ("first_frame", "last_frame")
REFS = ("ref_images", "ref_videos", "ref_audio")


def length_refusal(seconds):
    """What a submit with a length not offered is told: the lengths that are, and for 3 or 4 s that they went."""
    allowed = ", ".join(map(str, DURATIONS[:-1])) + f" or {DURATIONS[-1]}"
    gone = " (3 and 4 s are no longer offered)" if seconds in (3, 4) else ""
    return f"duration must be {allowed} seconds{gone}"


def check_fields(fields):
    """The checks of a submit's plain fields, the same for the form and JSON: task and orientation, the task's path
    manifest, the prompt, the length and the seed. Returns (task, orientation, prompt, seconds, seed), seed None
    for a random one. Raises ValueError with what the client is told."""
    task = fields.get("task", "t2va")
    orientation = fields.get("orientation", "landscape")
    if task not in PATHS or orientation not in ORIENTATIONS:
        raise ValueError("bad task/orientation")
    if not PATHS[task].is_file():
        raise ValueError(f"no path manifest for {task} at {PATHS[task]} (run install/70-manifests.sh)")
    prompt = fields.get("prompt", "").strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if len(prompt) > PROMPT_MAX:
        raise ValueError(f"prompt is too long ({len(prompt)} characters; at most {PROMPT_MAX})")
    try:
        prompt.encode("utf-8")      # JSON can hold a lone surrogate ("\ud83c"); the tokenizer and the worker fail on it
    except UnicodeEncodeError:
        raise ValueError("prompt is not valid text (an unpaired UTF-16 surrogate, e.g. an emoji cut in half)") from None
    raw = (fields.get("seed") or "").strip()
    seconds = int(fields.get("seconds") or 5)
    if seconds not in DURATIONS:
        raise ValueError(length_refusal(seconds))
    seed = int(raw) if raw else None
    if seed is not None and not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    return task, orientation, prompt, seconds, seed


def one_of(values):
    """"a", "a or b", "a, b or c"."""
    values = [str(v) for v in values]
    return values[0] if len(values) == 1 else f"{', '.join(values[:-1])} or {values[-1]}"


def offered(model, dim, value):
    """Whether `model` makes clips with this value of a dimension (task, frames: fl2va's keyframes, seconds,
    orientation, or one of its options) in a row of its table: {"value", "offered": true}, or {"value", "offered":
    false, "why"} with its reason not yet."""
    m = MODELS[model]
    if any(value in row.get(dim, ()) for row in m["offers"]):
        return {"value": value, "offered": True}
    why = next((n["why"] for n in m["not_yet"] if value in n.get(dim, ())), f"{m['name']} does not make these clips")
    return {"value": value, "offered": False, "why": why}


def model_refusal(model, values):
    """Why `model` does not make a clip with these values ({dim: value} for offered()), or None: the reason of the
    first value it does not make, in the order given; else, when no row of its table has all of them, that it does not
    make them together."""
    for dim, value in values.items():
        found = offered(model, dim, value)
        if not found["offered"]:
            return found["why"]
    if any(all(value in row[dim] for dim, value in values.items()) for row in MODELS[model]["offers"]):
        return None
    asked = ", ".join(f"{dim} {value}" for dim, value in values.items())
    return f"{MODELS[model]['name']} does not make clips with {asked} together yet"


def public_models():
    """The enabled models, as /api/state "models" and the page's lf-models table give them, in the registry's order:
    [{"id", "name", "about", "runs" (session or job), "experimental", "default", "capabilities", "options",
    "combinations", "eta_s"}]. capabilities: for each of task, frames (fl2va's keyframes: first, last, first+last),
    seconds and orientation, every value of the end state (Sol-H3's) as offered(); options: name, label, type
    (string or integer: its JSON type), default, and the choices as offered(), with a "size" per orientation where the
    choice sets the clip's size; combinations: the rows of the table, one of which a clip must fit in full; eta_s:
    model_eta."""
    out = []
    for mid in ENABLED:
        m = MODELS[mid]
        out.append({"id": mid, "name": m["name"], "about": m["about"], "runs": m["runs"],
                    "experimental": m["experimental"], "default": mid == DEFAULT_MODEL,
                    "capabilities": {dim: [offered(mid, dim, v) for v in values] for dim, values in END_STATE.items()},
                    "options": [dict({k: o[k] for k in ("name", "label", "type", "default")},
                                     choices=[dict(c, **offered(mid, o["name"], c["value"])) for c in o["choices"]])
                                for o in m["options"]],
                    "combinations": [{dim: list(values) for dim, values in row.items()} for row in m["offers"]],
                    "eta_s": model_eta(mid)})
    return out


def check_model(fields, task, orientation, seconds):
    """The model of a submit and its options, the same for the form and JSON (json_form checks the JSON types
    first): "model" (none or empty: sol-h3) one of the enabled ones; each of its options one of its choices (its
    default when not given), and no option of another model; then its table for the task, length, format and options
    (the keyframes are checked in submit, once the files are known). Returns (model, options). Raises ValueError with
    what the client is told."""
    model = fields.get("model") or DEFAULT_MODEL
    if model not in ENABLED:
        if model in MODELS:
            raise ValueError(f"{model} is off on this studio (LFSTUDIO_MODELS)")
        raise ValueError(f"model must be one of {', '.join(ENABLED)}")
    own = {o["name"]: o for o in MODELS[model]["options"]}
    for name in OPTION_TYPES:
        if name not in own and fields.get(name) not in (None, ""):
            owners = [mid for mid in ENABLED if any(o["name"] == name for o in MODELS[mid]["options"])]
            raise ValueError(f"{name} is an option of {', '.join(owners)}; {model} takes "
                             + (", ".join(own) if own else "none"))
    opts = {}
    for name, option in own.items():
        value, choices = fields.get(name), [c["value"] for c in option["choices"]]
        value = option["default"] if value in (None, "") else value
        if option["type"] == "integer" and isinstance(value, str) and re.fullmatch(r"\s*[0-9]{1,9}\s*", value):
            value = int(value)
        if value not in choices:
            raise ValueError(f"{name} must be {one_of(choices)} for {model}")
        opts[name] = value
    why = model_refusal(model, dict(task=task, seconds=seconds, orientation=orientation, **opts))
    if why:
        raise ValueError(why)
    return model, opts


def check_counts(images, videos, audio):
    """H3's limits on ref2va references (the counts of one request)."""
    if (images > REF_MAX["image"] or videos > REF_MAX["video"] or audio > REF_MAX["audio"]
            or images + videos + audio > REF_TOTAL):
        raise ValueError(f"H3 limits: up to {REF_MAX['image']} images, {REF_MAX['video']} videos, "
                         f"{REF_MAX['audio']} audio files, {REF_TOTAL} references in total")


def infer_task(zones):
    """The task a request without one asks for, from the zones it has files in (new or kept): frames make fl2va,
    references ref2va, none t2va. Frames and references together are refused."""
    frames, refs = [z for z in FRAMES if z in zones], [z for z in REFS if z in zones]
    if frames and refs:
        raise ValueError(f"{frames[0]} and {refs[0]} can't go together: fl2va takes frames, ref2va references")
    return "fl2va" if frames else "ref2va" if refs else "t2va"


def kept_zones(fields):
    """The zones a form's "keep" names files in (only with remix_from, as submit reads it)."""
    if not fields.get("remix_from"):
        return set()
    return {item.partition(":")[0] for item in (fields.get("keep") or "").split(",")} & set(ZONES)


JSON_KEYS = ("prompt", "task", "orientation", "seconds", "seed", "remix_from", "keep") + ZONES + ("model",) + \
    tuple(OPTION_TYPES)          # the options of the enabled models (none with sol-h3 alone)
NO_URLS = "the studio does not fetch URLs; send the file as base64, or use bin/lf-cli, which downloads URLs itself"
DATA_URI = re.compile(r"data:([^;,]*)((?:;[^;,]*)*);base64,", re.I)
SPACES = str.maketrans("", "", " \t\n\r\f\v")


def json_file(label, value, stem):
    """One file of a JSON submit, checked but not decoded yet: (name, base64 text). `stem` names a data: URI's
    file."""
    form = 'a file is {"name": "frame.png", "data": "<base64>"} or a "data:<type>;base64,<base64>" URI'
    if isinstance(value, dict):
        if "url" in value:
            raise ValueError(f"{label}: {NO_URLS}")
        for key in value:
            if key not in ("name", "data"):
                raise ValueError(f'{label}: unknown key "{key}" ({form})')
        if not isinstance(value.get("name"), str) or not value["name"].strip():
            raise ValueError(f"{label}: name must be a file name, such as frame.png ({form})")
        if not isinstance(value.get("data"), str):
            raise ValueError(f"{label}: data must be a base64 string ({form})")
        return value["name"].strip(), value["data"]
    if isinstance(value, str):
        if re.match(r"\s*https?://", value, re.I):
            raise ValueError(f"{label}: {NO_URLS}")
        m = DATA_URI.match(value[:300])
        if m:
            mime = m.group(1).strip().lower()
            ext = MIME_EXT.get(mime) or mimetypes.guess_extension(mime) or ""
            if not ext:
                raise ValueError(f"{label}: unsupported file type {mime or '(none)'} in the data: URI")
            return stem + ext, value[m.end():]
    raise ValueError(f"{label}: {form}")


def b64_file(label, text):
    """Decode a JSON file's base64 (ASCII whitespace ignored, so wrapped base64 works)."""
    try:
        data = base64.b64decode(text.translate(SPACES), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{label}: the data is not valid base64") from None
    if not data:
        raise ValueError(f"{label} is empty")
    return data


def json_form(raw):
    """POST /api/generate with a JSON body (#29) as the form's (fields, files), for submit(). Keys: prompt, task,
    orientation, seconds, seed (an integer, or null for a random one), first_frame and last_frame (one file each),
    ref_images, ref_videos and ref_audio (lists of files), remix_from and keep (a list of "zone:index"), model (a
    string) and the enabled models' options (OPTION_TYPES: resolution, a string, and steps, an integer, with
    lightx2v-turbo on). null is the same as leaving a key out. Stricter than the form: an unknown key, a wrong type, a
    malformed file, files the task doesn't use, frames together with references, and keep without remix_from are
    refused (ValueError), all before any file is decoded; so is what check_model refuses."""
    try:
        body = json.loads(raw, strict=False)    # strict=False: wrapped base64 (raw line breaks in a string) is fine
    except ValueError as error:                 # also bytes that are not UTF-8
        raise ValueError(f"the body is not valid JSON ({error})") from None
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    for key in body:
        if key not in JSON_KEYS:
            raise ValueError(f'unknown key "{key}" (known: {", ".join(JSON_KEYS)})')
    body = {k: v for k, v in body.items() if v is not None}
    fields = {}
    for key in ("prompt", "task", "orientation", "remix_from", "model") + tuple(
            k for k, kind in OPTION_TYPES.items() if kind == "string"):
        if key in body:
            if not isinstance(body[key], str):
                raise ValueError(f"{key} must be a string")
            fields[key] = body[key]
    for key in (k for k, kind in OPTION_TYPES.items() if kind == "integer"):
        if key in body:
            if type(body[key]) is not int:
                raise ValueError(f"{key} must be an integer")
            fields[key] = str(body[key])
    if "seconds" in body:
        if type(body["seconds"]) is not int:
            raise ValueError("seconds must be an integer")
        fields["seconds"] = str(body["seconds"])
    if "seed" in body:
        if type(body["seed"]) is not int:
            raise ValueError("seed must be an integer or null")
        fields["seed"] = str(body["seed"])
    if "keep" in body:
        keep = body["keep"]
        if not isinstance(keep, list) or not all(isinstance(k, str) and re.fullmatch(
                rf"({'|'.join(ZONES)}):\d+", k) for k in keep):
            raise ValueError('keep must be a list of "zone:index" strings')
        if keep and "remix_from" not in body:
            raise ValueError("keep needs remix_from (the clip whose files it keeps)")
        fields["keep"] = ",".join(keep)
    pending = {}                                # zone -> [(label, name, base64 text)]
    stems = {"first_frame": "first_frame", "last_frame": "last_frame", "ref_images": "image",
             "ref_videos": "video", "ref_audio": "audio"}
    for zone in ZONES:
        if zone not in body:
            continue
        if zone in FRAMES:
            if isinstance(body[zone], list):
                raise ValueError(f"{zone} must be one file")
            items = [(zone, body[zone])]
        else:
            if not isinstance(body[zone], list):
                raise ValueError(f"{zone} must be a list")
            items = [(f"{zone}[{i}]", v) for i, v in enumerate(body[zone])]
        if items:
            pending[zone] = [(label, *json_file(label, v, stems[zone])) for label, v in items]
    zones = set(pending) | kept_zones(fields)
    fields["task"] = fields.get("task") or infer_task(zones)
    task, orientation, _, seconds, _ = check_fields(fields)
    check_model(fields, task, orientation, seconds)
    for zone in ZONES:                          # in the form's order, so the first one named is the first sent
        if zone in zones:
            if task == "t2va":
                raise ValueError("t2va takes no files")
            if (task == "fl2va") != (zone in FRAMES):
                raise ValueError(f"{zone} is used only by {'fl2va' if zone in FRAMES else 'ref2va'}")
    check_counts(*(len(pending.get(z, [])) for z in REFS))
    files = {zone: [(name, b64_file(label, text)) for label, name, text in items] for zone, items in pending.items()}
    return fields, files


def queue_full():
    """The 429 reply while LFSTUDIO_MAX_QUEUED clips are waiting, else None. Call it holding `lock`."""
    if sum(j["status"] == "queued" for j in jobs) < MAX_QUEUED:
        return None
    return (429, {"error": f"the queue is full ({MAX_QUEUED} clips waiting); try again when one has finished"},
            "application/json", {"Retry-After": "60"})


def submit(fields, files):
    """Queue a clip: the form's fields and its files ({zone: [(name, bytes), ...]}), from the page's multipart form
    or from JSON (json_form). Checks everything first and saves the files; a refused request leaves nothing behind.
    Returns the reply: 200 {"id", "job"} (job: api_record), 400 {"error": "<ExcType>: <message>"}, or 429."""
    job_dir = None
    try:
        task, orientation, prompt, seconds, seed = check_fields(fields)
        model, opts = check_model(fields, task, orientation, seconds)
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF        # empty -> random
        # Remix: "keep" names files of clip "remix_from" to reuse, as zone:index in the order of /api/inputs
        src = None
        if fields.get("remix_from"):
            src = next((j for j in jobs if j["id"] == fields["remix_from"]), None)   # deleted too: its files
                                                                                    # are in the trash
            if src is None:
                raise ValueError("the clip to remix is no longer in the list; attach the files again")
            if src.get("moving"):
                raise ValueError("the clip to remix is being deleted or restored right now; try again")
        keep = {}
        for item in (fields.get("keep") or "").split(","):
            zone, _, index = item.partition(":")
            if src and zone in ZONES and index.isdigit():
                keep.setdefault(zone, {})[int(index)] = None          # ordered, no duplicates
        jid = uuid.uuid4().hex
        job_dir = INPUTS / jid
        job_dir.mkdir(parents=True)

        def kept(zone):
            """The kept files of this zone, put into the new job's directory, in the order the form sent."""
            available = [trashed_path(src, f) for f in zone_files(src, zone)] if src else []
            if any(i >= len(available) for i in keep.get(zone, {})):
                raise ValueError("the remixed clip has fewer files than the form kept; attach them again")
            return [reuse_input(job_dir, available[i]) for i in keep.get(zone, {})]
        job = {"id": jid, "case_id": f"web_{time.strftime('%H%M%S')}_{jid[:6]}", "task": task,
               "orientation": orientation, "seconds": seconds, "prompt": prompt, "seed": seed, "status": "queued",
               "created": time.time()}
        if model != SOL_H3:                # a sol-h3 clip's record stays as before the model choice
            job.update(model=model, model_opts=model_options(model, opts, orientation))
        if task == "fl2va":
            for key in FRAMES:
                if files.get(key):
                    fn, data = files[key][0]
                    job[key] = save_upload(job_dir, fn, data, IMG)
                elif kept_file := kept(key):         # one file per frame: a new upload replaces it
                    job[key] = kept_file[0]
            if not (job.get("first_frame") or job.get("last_frame")):
                raise ValueError("fl2va needs a first frame, a last frame, or both")
        if task == "ref2va":
            # order = prompt numbering: <Picture N> images, then <Video N> videos (a video's own soundtrack is
            # added by Sol-H3 as an audio reference just before it), then standalone <Audio N> files
            # per kind: kept files first, then new uploads (the page shows them in that order); kept reference
            # videos are the normalized copies already
            refs = [{"type": "image", "path": p} for p in kept("ref_images")]
            refs += [{"type": "image", "path": save_upload(job_dir, fn, d, IMG)} for fn, d in files.get("ref_images", [])]
            refs += [{"type": "video", "path": p} for p in kept("ref_videos")]
            refs += [{"type": "video", "path": prepare_reference_video(save_upload(job_dir, fn, d, VID))}
                     for fn, d in files.get("ref_videos", [])]
            refs += [{"type": "audio", "path": p} for p in kept("ref_audio")]
            refs += [{"type": "audio", "path": save_upload(job_dir, fn, d, AUD)} for fn, d in files.get("ref_audio", [])]
            counts = {t: sum(r["type"] == t for r in refs) for t in ("image", "video", "audio")}
            if not (counts["image"] or counts["video"]):
                raise ValueError("ref2va needs at least one reference image or video")
            check_counts(counts["image"], counts["video"], counts["audio"])
            job["references"] = refs
        if task == "fl2va":                # the keyframes against the model's table
            frames = "+".join(anchor for anchor in ("first", "last") if job.get(f"{anchor}_frame"))
            why = model_refusal(model, dict(task=task, frames=frames, seconds=seconds, orientation=orientation, **opts))
            if why:
                raise ValueError(why)
        # a job model's conditioning with MiniMax's prompt tokens split into plain text (SOL_H3_PROMPT_TOKENS=plain,
        # or a patch without the setting) is refused by its job: said now, not after the memory was freed for it
        tokens = [t for t in getattr(JOB_MODULES.get(model), "MINIMAX_PROMPT_TOKENS", ()) if t in prompt]
        mode = getattr(runtime.config, "prompt_tokens_mode", lambda: "plain")() if tokens else None
        if tokens and mode != "minimax":
            raise ValueError(f"{model} needs SOL_H3_PROMPT_TOKENS=minimax for a prompt with MiniMax's tokens "
                             f"({', '.join(tokens)}): {mode} splits them into text tokens H3 never saw")
        # validate now so errors come back immediately
        probe = {k: job[k] for k in ("case_id", "prompt", "seed", "task")}
        for k in ("first_frame", "last_frame", "references"):
            if k in job:
                probe[k] = job[k]
        normalize_case(probe)
        if task != "t2va":
            job["labels"] = job_labels(job)        # never part of the case: run_job builds it from fixed keys
    except Exception as e:
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)     # a refused request leaves nothing behind
        return 400, {"error": f"{type(e).__name__}: {e}"}
    with lock:
        full = queue_full()             # checked again: other requests may have filled the queue meanwhile
        if not full:
            # 'created' orders the history and is the paging cursor: set it here, strictly increasing, so a request
            # that took longer (a reference video being normalized) can't be appended after a later one
            job["created"] = max(time.time(), jobs[-1]["created"] + 0.001 if jobs else 0)
            jobs.append(job)
            preload["pending"] = None              # a clip decides what is loaded next: a requested warm-up that
            session["switching_to"] = None         # hasn't started is dropped (the clip loads its own settings)
            # a job already queued for the warm-up's settings runs first, so it keeps the warm-up
            waited_for = warm_up_wanted()
    if full:
        shutil.rmtree(job_dir, ignore_errors=True)
        return full
    save_job(job)
    if session.get("preloading"):
        if job_model(job) != SOL_H3:       # it closes the warm-up's models anyway: kept only for a clip before it
            if not waited_for:
                abort_preload(f"a {MODELS[job_model(job)]['name']} clip arrived, which needs the memory")
        elif not same_models(session.get("loading_key"), (job["task"], job["orientation"], job["seconds"])):
            if not waited_for:
                abort_preload("a job for another mode/format/length arrived")
        elif time.time() < (session.get("takeover_until") or 0):
            abort_preload("a job for the same models arrived early in the warm-up; it loads the models itself")
    wake.set()
    return 200, {"id": jid, "job": api_record(job)}


class H(BaseHTTPRequestHandler):
    server_version = "LFStudio/0.5"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (extra or {}):
            self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def unknown_host(self):
        """Answer 421 when the request names a host the studio does not answer to (known_host), in Host or in
        X-Forwarded-Host (spark-switchboard passes the public one there). Returns whether it answered."""
        for header in ("Host", "X-Forwarded-Host"):
            for value in self.headers.get_all(header) or []:
                for name in map(host_name, value.split(",")):        # X-Forwarded-Host: one per proxy
                    if not known_host(name):
                        n = self.headers.get("Content-Length", "")
                        if n.isdigit() and int(n) <= SMALL_BODY:  # a small body is read, so the answer arrives
                            self.rfile.read(int(n))
                        self.send(421, {"error": f"unknown host name {name[:200]}; add it to LFSTUDIO_ALLOWED_HOSTS"})
                        return True
        return False

    def do_GET(self):
        if self.unknown_host():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send(200, page_html(), "text/html; charset=utf-8")
        if path == "/health":
            return self.send(200, {"ok": True})
        if path == "/api/state":
            # the session, every queued or running job (spark-switchboard's busy probe reads these) and the newest
            # page of finished ones; older clips come from /api/jobs
            ready = ready_tasks()
            with lock:
                active = [j for j in jobs if j["status"] not in FINISHED]
                finished = [j for j in jobs if j["status"] in FINISHED and not j.get("deleted")]
                window = sorted(active + finished[-PAGE:], key=lambda j: j["created"], reverse=True)
                warmable = [t for t in ready if mode_warmup_case(t)]     # modes "Load selected now" can warm up
                state = {"now": time.time(), "tasks_ready": ready, "trash_days": TRASH_DAYS, "warmable": warmable,
                         "reshape": RESHAPE, "backend": BACKEND, "api": API, "upscale": UPSCALE,
                         "upscale_defaults": upscale_defaults(),
                         "upscale_load": UPSCALE_CONF["load"] if UPSCALE else None,
                         "upscale_pin": True if UPSCALE else None, "upscale_offers": UPSCALE_OFFERS,
                         "session": {k: session.get(k) for k in ("key", "state", "since", "loading_key", "preloading",
                                                                 "takeover_until", "load_eta", "warm_s",
                                                                 "switching_to")},
                         "jobs": [dict(j) for j in window],
                         "done": sum(j["status"] == "done" for j in finished), "more": len(finished) > PAGE,
                         "deleted": {j["id"]: j["created"] for j in jobs if j.get("deleted")}}   # in the trash
                if MULTI:                  # the model choice: the registry, and the loaded session's model
                    state["session"]["model"] = SOL_H3 if session.get("key") or session.get("loading_key") else None
            if MULTI:
                state["models"] = public_models()                       # (reads load-times.json)
            with_upscalable(state["jobs"])                              # (reads files: without the lock)
            with_model_fields(state["jobs"])
            return self.send(200, state)
        if path == "/api/jobs":            # older clips: ?before=<created>&limit=<n>, newest first
            q = parse_qs(urlsplit(self.path).query)
            try:
                before = float(q.get("before", ["inf"])[0])
                limit = max(1, min(100, int(q.get("limit", [PAGE])[0])))
            except ValueError:
                return self.send(400, {"error": "before must be a number and limit an integer"})
            with lock:
                older = [j for j in jobs if j["created"] < before and not j.get("deleted")]
                page = [dict(j) for j in older[::-1][:limit]]
            return self.send(200, {"jobs": with_model_fields(with_upscalable(page)), "more": len(older) > limit})
        m = re.fullmatch(r"/poster/([0-9a-f]{32})", path)
        if m:
            mp4 = clip_file(find_job(m.group(1)))
            poster = mp4.with_suffix(".jpg") if mp4 else None
            if poster and not poster.exists():
                make_poster(mp4)
            if not poster or not poster.exists():
                return self.send(404, {"error": "no poster"})
            return self.send(200, poster.read_bytes(), "image/jpeg", {"Cache-Control": PICTURE_CACHE})
        m = re.fullmatch(r"/api/inputs/([0-9a-f]{32})", path)
        if m:                      # a clip's input files, for Remix: what can be reused and what is gone
            job = find_job(m.group(1))
            if job is None:
                return self.send(404, {"error": "no such clip"})
            out, missing = {}, []
            for zone in ZONES:
                for i, f in enumerate(zone_files(job, zone)):
                    if Path(f).is_file():
                        out.setdefault(zone, []).append({"name": Path(f).name, "index": i,
                                                         "url": f"/input/{job['id']}/{zone}/{i}"})
                    else:
                        missing.append(Path(f).name)
            return self.send(200, {"task": job["task"], "inputs": out, "missing": missing})
        m = re.fullmatch(r"/input/([0-9a-f]{32})/([a-z_]+)/(\d+)", path)
        if m:
            job = find_job(m.group(1))
            files = zone_files(job, m.group(2)) if job and m.group(2) in ZONES else []
            f = Path(files[int(m.group(3))]) if int(m.group(3)) < len(files) else None
            if not f or not f.is_file():
                return self.send(404, {"error": "no such input"})
            return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        m = re.fullmatch(r"/still/([0-9a-f]{32})", path)
        if m:                      # the picture behind a queued or running clip's timer (card_picture)
            with lock:
                job = find_job(m.group(1))
                job = None if job is None or job.get("moving") else dict(job)    # not while Delete moves its files
            f = card_picture(job) if job else None
            if not f:
                return self.send(404, {"error": "no picture"})
            return self.send_file(f, mimetypes.guess_type(f.name)[0] or "application/octet-stream",
                                  {"Cache-Control": PICTURE_CACHE})
        m = re.fullmatch(r"/video/([0-9a-f]{32})", path)
        if m:
            job = find_job(m.group(1))
            mp4 = clip_file(job)
            if not mp4:
                return self.send(404, {"error": "no such video"})
            return self.send_file(mp4, name=clip_name(job))
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})(/video)?", path)
        if m:
            if not API:                    # LFSTUDIO_API=0: the page never asks for these
                return self.send(403, API_OFF)
            return self.job_reply(m.group(1), video=bool(m.group(2)))
        self.send(404, {"error": "not found"})

    def job_reply(self, jid, video):
        """GET /api/jobs/<id>[?wait=S] and /api/jobs/<id>/video[?wait=S] (#29): wait up to S seconds (at most
        WAIT_MAX_S; none by default) for the clip to be settled, then answer its API record, or for /video the MP4
        once it is done (202 with the record while it is still queued or running, 409 with it if it failed or was
        cancelled). 503 with the record when the studio shuts down first."""
        global waiting
        q = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        try:
            wait = float(q.get("wait", ["0"])[0])
            if not (math.isfinite(wait) and wait >= 0):
                raise ValueError
        except ValueError:
            return self.send(400, {"error": "wait must be a number of seconds, 0 or more"})
        with lock:
            job = find_job(jid)
        if job is None:
            return self.send(404, {"error": "no such clip"})
        with changed:
            waiting += 1
        try:
            with changed:
                changed.wait_for(lambda: settled(job) or stopping.is_set(), timeout=min(wait, WAIT_MAX_S))
            rec = api_record(job)
            if stopping.is_set() and not settled(rec):
                return self.send(503, {"error": "the studio is shutting down; ask again once it is back", "job": rec})
            if not video:
                return self.send(200, rec)
            if rec["state"] in ("failed", "cancelled"):
                why = f"the clip failed: {rec.get('error')}" if rec["state"] == "failed" else "the clip was cancelled"
                return self.send(409, {"error": why, "job": rec})
            if rec["state"] != "done":
                return self.send(202, rec)
        finally:
            with changed:
                waiting -= 1
                changed.notify_all()                   # shutdown waits for these answers (main)
        mp4 = clip_file(job)
        if not mp4:
            return self.send(404, {"error": "no such video", "job": rec})
        return self.send_file(mp4, name=clip_name(job))

    def send_file(self, p, ctype="video/mp4", extra=None, name=None):
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng and (m := re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())) and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Disposition", f'inline; filename="{name or p.name}"')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def same_origin(self):
        """Refuse cross-site POSTs (a page on another site submitting a form to this server).
        Browsers send Origin on such requests; same-page fetches send the page's own origin."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        allowed = {self.headers.get("Host"), self.headers.get("X-Forwarded-Host")}
        return urlsplit(origin).netloc in allowed

    def body_length(self, limit, too_large, required=False):
        """The request's Content-Length, checked before any of the body is read. None after answering 411 (none
        given, when `required`: a chunked upload arrives empty), 400 (not a number) or 413 (over `limit`; the body
        is not read, so a client still sending it may see the connection close instead). A missing one counts as
        0 when not required."""
        value = self.headers.get("Content-Length")
        if value is None:
            if not required:
                return 0
            self.send(411, {"error": "Content-Length required (chunked uploads are not supported)"})
            return None
        if not re.fullmatch(r"[0-9]+", value.strip()):
            self.send(400, {"error": "bad Content-Length"})
            return None
        if int(value) > limit:
            self.send(413, {"error": too_large})
            return None
        return int(value)

    def do_POST(self):
        if self.unknown_host():
            return
        if not self.same_origin():
            return self.send(403, {"error": "cross-origin request refused"})
        if self.path in ("/api/unload", "/api/load", "/api/cancel", "/api/delete", "/api/undelete", "/api/upscale"):
            n = self.body_length(SMALL_BODY, f"request body too large ({SMALL_BODY // 1024} KiB max)")
            if n is None:
                return
            small = self.rfile.read(n)                 # JSON, {"id": ...} or {task, orientation, seconds}
        if self.path == "/api/upscale":    # the experimental Upscale (the docstring); 404 when it is off
            return self.send(*submit_upscale(small))
        if self.path == "/api/unload":
            with lock:
                busy = any(j["status"] not in FINISHED + ("queued",) for j in jobs)
                if not busy:                       # a requested load ("Load selected now") is cancelled too
                    preload["pending"] = None
                    session["switching_to"] = None
            if busy:
                return self.send(409, {"error": "a job is running"})
            if abort_preload("unload requested"):
                return self.send(200, {"ok": True})
            close_session("unload requested")
            return self.send(200, {"ok": True})
        if self.path == "/api/load":       # "Load selected now": warm up these settings before the prompt is ready
            try:
                want = json.loads(small or b"{}")
                key = (want["task"], want["orientation"], int(want["seconds"]))
                if key[0] not in PATHS or key[1] not in ORIENTATIONS:
                    raise ValueError
                model = want.get("model") or DEFAULT_MODEL          # sol-h3 when not given, as before
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                return self.send(400, {"error": "expected JSON {task, orientation, seconds}"})
            if not isinstance(model, str):
                return self.send(400, {"error": "model must be a string"})
            if model != SOL_H3:
                if model not in ENABLED:
                    return self.send(400, {"error": f"{model} is off on this studio (LFSTUDIO_MODELS)"
                                           if model in MODELS else f"model must be one of {', '.join(ENABLED)}"})
                return self.send(409, {"error": f"{MODELS[model]['name']} loads its model with each clip; there is "
                                                "nothing to load ahead"})
            if key[2] not in DURATIONS:
                return self.send(400, {"error": length_refusal(key[2])})
            if not PATHS[key[0]].is_file():
                return self.send(400, {"error": f"no path manifest for {key[0]} (run install/70-manifests.sh)"})
            with lock:
                busy = any(j["status"] not in FINISHED for j in jobs)
                case = mode_warmup_case(key[0])
                switching = session.get("switching_to")
                already = (switching == key or not switching and (
                    key == session.get("key") and session.get("state") in ("ready", "busy")
                    or key == session.get("loading_key") and session.get("preloading")))
                if not busy and not already and case:
                    preload["pending"] = (key, case)      # replaces an earlier request that hasn't started
                    session["switching_to"] = key
            if busy:
                return self.send(409, {"error": "a clip is queued or running; the models change for it anyway"})
            if already:
                return self.send(200, {"ok": True, "already": True})
            if case is None:
                return self.send(409, {"error": f"no earlier {key[0]} clip with its files to warm up with"})
            abort_preload(f"loading {key} as asked")            # a warm-up for other settings
            wake.set()
            log("load requested:", key)
            return self.send(200, {"ok": True})
        if self.path == "/api/cancel":
            try:
                jid = json.loads(small or b"{}").get("id")
            except (ValueError, AttributeError):
                return self.send(400, {"error": "expected JSON {\"id\": ...}"})
            # Under the lock, atomic with the worker taking a queued job and with pipeline_call accepting a finished
            # clip: a queued job is cancelled here and now; a running one is marked, and stop_job stops its workers.
            # Its input files stay (Remix), like a failed clip's.
            queued = stop = drop = False
            with lock:
                job = find_job(jid)
                if job is None:
                    reply = 404, {"error": "no such clip"}
                elif job["status"] in FINISHED:
                    reply = 409, {"error": {"done": "the clip has already finished",
                                            "failed": "the clip has already failed",
                                            "cancelled": "the clip is already cancelled"}[job["status"]]}
                elif job.get("saving"):
                    reply = 409, {"error": "the clip has just finished and is being saved"}
                else:
                    queued = job["status"] == "queued"
                    if queued:
                        job.update(status="cancelled", finished=time.time())
                        # the warm-up may have been kept only for this clip (/api/generate): if another clip waits
                        # now and none for the warm-up's settings, stop it as a clip for other settings would have
                        drop = (session.get("preloading") and not warm_up_wanted()
                                and any(j["status"] == "queued" for j in jobs))
                    else:
                        stop = not job.get("cancel_requested")      # a second Cancel: already being stopped
                        job["cancel_requested"] = True
                    reply = 200, {"ok": True, "job": dict(job)}
            if reply[0] == 200:
                save_job(job)        # a running one too: if the server dies before it stops, it stays cancelled
                if queued:
                    log("cancelled queued job", job["id"])
                    if drop:
                        abort_preload("the clip it was kept for was cancelled")
                        wake.set()
                elif stop:
                    log("cancelling job", job["id"], "while", reply[1]["job"]["status"])
                    threading.Thread(target=stop_job, args=(job,), daemon=True).start()
            return self.send(*reply)
        if self.path in ("/api/delete", "/api/undelete"):
            try:
                jid = json.loads(small or b"{}").get("id")
            except (ValueError, AttributeError):
                return self.send(400, {"error": "expected JSON {\"id\": ...}"})
            # Under the lock: check and mark the clip as moving (Remix, warm-ups, the trash purge and another Delete
            # or Undo leave it alone meanwhile). The files move without the lock: a copy to or from a share can take
            # a while. Shutdown waits for moves in progress.
            global moves_running
            with lock:
                job = next((j for j in jobs if j["id"] == jid), None)
                if job is None:
                    reply = 404, {"error": "no such clip"}
                elif stopping.is_set():
                    reply = 503, {"error": "the studio is shutting down"}
                elif job.get("moving"):
                    reply = 409, {"error": "the clip is being moved; try again in a moment"}
                elif self.path == "/api/undelete":
                    reply = None if job.get("deleted") else (200, {"ok": True, "job": dict(job)})
                elif job["status"] not in FINISHED:
                    reply = 409, {"error": "the clip is still queued or running"}
                elif self.path == "/api/delete" and any(is_upscale(j) and j["parent"] == jid and j["status"] not in
                                                        FINISHED for j in jobs):    # it reads the clip's files
                    reply = 409, {"error": "an upscale of this clip is queued or running; cancel it first, or wait "
                                           "for it"}
                elif job.get("deleted"):
                    reply = 200, {"ok": True}
                else:
                    reply, paths = None, trash_paths(job)
                if reply is None:
                    job["moving"] = True
                    with moves_done:
                        moves_running += 1
            if reply is None:
                try:
                    reply = delete_clip(job, paths) if self.path == "/api/delete" else undelete_clip(job)
                except Exception as error:                     # never leave the clip marked as moving
                    traceback.print_exc()
                    reply = 500, {"error": f"{type(error).__name__}: {error}"}
                finally:
                    with lock:
                        job.pop("moving", None)
                    with moves_done:
                        moves_running -= 1
                        moves_done.notify_all()
            return self.send(*reply)
        if self.path != "/api/generate":
            return self.send(404, {"error": "not found"})
        n = self.body_length(MAX_UPLOAD, "upload too large (200 MB max)", required=True)
        if n is None:
            return
        ctype = self.headers.get("Content-Type", "")
        kind = ctype.split(";", 1)[0].strip().lower()
        raw = self.rfile.read(n)                       # read first, so the client gets the answer below
        if kind not in ("multipart/form-data", "application/json"):
            return self.send(415, {"error": "send multipart/form-data or application/json"})
        if not API and (kind == "application/json" or self.headers.get(API_CLIENT) is not None):
            return self.send(403, API_OFF)             # LFSTUDIO_API=0: only the page's own form
        with lock:
            full = queue_full()
        if full:
            return self.send(*full)
        try:
            if kind == "application/json":
                fields, files = json_form(raw)
            else:
                fields, files = parse_multipart(ctype, raw)
                if not fields.get("task"):             # the page always sends one; a script may not
                    fields["task"] = infer_task({z for z in ZONES if files.get(z)} | kept_zones(fields))
        except Exception as e:
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})
        del raw
        self.send(*submit(fields, files))


def stop_with_launcher():
    """bin/lf-studio starts the server in its own session and sets LFSTUDIO_LAUNCHER_PID. If the launcher dies
    without stopping the server (SIGKILL), Linux sends this process SIGTERM (PR_SET_PDEATHSIG), so the pipeline is
    still closed."""
    launcher = os.environ.get("LFSTUDIO_LAUNCHER_PID")
    if not launcher or not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:     # 1 = PR_SET_PDEATHSIG
            return
    except (OSError, AttributeError):
        return
    if str(os.getppid()) != launcher:        # the launcher died before the prctl
        log("the launcher is gone; stopping")
        os.kill(os.getpid(), signal.SIGTERM)


def main():
    try:
        jobs.extend(open_db())
    except sqlite3.Error as error:
        fail(f"cannot open {DB_FILE}: {error}")
    for j in jobs:
        if j.get("status") not in FINISHED:            # it was queued or running when the server stopped
            if j.pop("cancel_requested", None):        # Cancel had answered: it stays cancelled
                j["status"] = "cancelled"
            else:
                j["status"] = "failed"
                j["error"] = "server restarted"
            save_job(j)
    recover_moves()
    purge_trash()
    for part in INPUTS.glob(STILL_PART):              # a card picture being made when the server stopped
        with contextlib.suppress(OSError):
            part.unlink()
    missing = [t for t in TASKS if t not in ready_tasks()]
    if missing:
        log("no path manifest for", ", ".join(missing), f"(looked for {PATHS_TEMPLATE})")
    if PRELOAD:
        preload["pending"] = preload_candidate()
        if preload["pending"]:
            session["switching_to"] = preload["pending"][0]
            log("will preload", preload["pending"][0])
            wake.set()
    threading.Thread(target=worker_loop, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), H)
    srv.daemon_threads = True

    def stop(*_):
        log("shutting down")
        stopping.set()
        wake.set()

        def wake_waiters_and_stop():                   # not in the signal handler itself (it may hold `changed`)
            with changed:                              # GET /api/jobs/<id>?wait= answers 503 now
                changed.notify_all()
            srv.shutdown()
        threading.Thread(target=wake_waiters_and_stop, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)    # a closed terminal or SSH session: close the pipeline, as for SIGTERM
    stop_with_launcher()
    for name, value in IGNORED_ENV.items():
        log(f"ignoring {name}={value}: it is an infer.py default; the studio sets length and format per job")
    if not API:
        log("the API for programs is off (LFSTUDIO_API=0): JSON submits, submits from bin/lf-cli and "
            "/api/jobs/<id> answer 403; the page works as usual")
    if UPSCALE:
        log(f"the experimental Upscale is on (LFSTUDIO_UPSCALE=1): {UPSCALE_SCRIPT} with {UPSCALE_CONF['python']}; "
            f"default strength {UPSCALE_CONF['strength']}, audio {UPSCALE_CONF['audio']}; {UPSCALE_CONF['load']} load")
        if UPSCALE_CONF["load"] == "fast" and not UPSCALE_CONF["o_direct"]:
            log(f"WARNING: {' or '.join(regen.FASTLOAD_SWITCHES)} is not 1 (the patch's switches), and the Upscale's "
                "job inherits them: its fast load reads no weight file with O_DIRECT, all of them through mmap, which "
                "takes minutes")
    if TURBO in ENABLED:
        conf = MODEL_CONF[TURBO]
        log(f"the experimental LightX2V Turbo is on (LFSTUDIO_MODELS): {H3TURBO_SCRIPT} with {conf['python']}, its "
            f"prompt encode {H3TURBO_ENCODE}; Qwen canvas {conf['qwen_canvas']}, memory floor {conf['min_free_gib']:g} "
            f"GiB, guard {conf['guard_gib']:g} GiB")
    users = [name for name, on in (("the Upscale's", UPSCALE), ("LightX2V Turbo's", TURBO in ENABLED)) if on]
    if users and MEMINFO != "/proc/meminfo":
        floors = "memory floors and guards" if len(users) > 1 else "memory floor and guard"
        log(f"WARNING: LFSTUDIO_TEST_MEMINFO={MEMINFO}, a test hook: {' and '.join(users)} {floors} read MemAvailable "
            "from that file, not from this machine; unset it outside the tests")
    log(f"LF Studio on http://{HOST}:{PORT}/ (package {PKG}, state {STATE_DIR})")
    try:
        srv.serve_forever()
    finally:
        stopping.set()
        with moves_done:                               # a Delete or Undo moving files finishes first (up to 60 s)
            if not moves_done.wait_for(lambda: moves_running == 0, timeout=60):
                log("a Delete or Undo is still moving files; the next start puts back an unfinished delete")
        with changed:                                  # the waiting requests' 503s go out before the process ends
            changed.notify_all()
            changed.wait_for(lambda: waiting == 0, timeout=5)
        stop_gpu_job()
        close_session("server exit")
        log("bye")


if __name__ == "__main__":
    main()
