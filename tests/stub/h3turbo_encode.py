#!/usr/bin/env python3
"""Stand-in for the experimental LightX2V Turbo's prompt encode (bench/h3turbo_encode.py: a transient Qwen worker
of the patched runtime): no Docker, no GPU, no Qwen.

    python -I -B tests/stub/h3turbo_encode.py --case DIR --prompt-file FILE --orientation O [--first IMG]
        [--last IMG] [--resolution 576p|768p] [--canvas draft|canvas] [--seed N] [--case-id ID] [--no-text-only]
        [--package DIR]

The server runs it as it runs the real one (LFSTUDIO_H3TURBO_ENCODE_SCRIPT; tests/stub/studio.py sets that), with its
own Python, in a process group of its own, its output in <case>/encode.log. It takes the real one's arguments and
writes what the real one writes in --case: qwen/conditioning.{pt,json} (the clip's: a stand-in payload, and a
receipt with the task, the prompt, the rows, and the canvas the keyframes were seen at in prepared_media), for fl2va
(unless --no-text-only) qwen-text/conditioning.{pt,json}, qwen-text/prompt.txt and qwen-text/text-only.json (the
prompt without its keyframe references: alignment lines and <Picture N> go, a simpler rewrite than the real one's),
qwen-worker/ (a log) and encode.json (the canvas, the image tokens, the times). It takes STUB_QWEN_S seconds;
SIGTERM (Cancel) ends it at once with exit status 143, as the real one closes its worker and exits. Exit status: 0
done, 1 failed ("[h3turbo-encode] failed: ..."), 2 refused ("[h3turbo-encode] refused: ..."), 143 SIGTERM.
  STUB_QWEN_S       seconds the encode takes (default 0.5)
  STUB_QWEN_FAIL    1: fail as a Qwen worker that died (exit 1)
  STUB_QWEN_TERM    "ignore": SIGTERM does nothing (a worker that does not answer)
  STUB_QWEN_LOG     a file each run appends a JSON line to ({"pid", "argv"})
  STUB_QWEN_WORKER_S  start a stand-in of its transient Qwen worker, as runtime.pipeline.Worker starts the real one:
                    a child in a session of its own whose command line names runtime.worker --module qwen --options
                    <case>/qwen-worker/options.json (the case resolved), its pid in qwen-worker/worker.pid. Like the
                    real one, it does not read its stdin while it starts (these seconds), then ends at EOF: when this
                    process closes it (at its end, on SIGTERM, on a failure) or is gone (killed)
Standard library only, so it runs under -I with any Python 3.10+.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

TAG = "[h3turbo-encode]"
CANVAS = {("576p", "landscape"): (1024, 576), ("576p", "portrait"): (576, 1024),
          ("768p", "landscape"): (1344, 768), ("768p", "portrait"): (768, 1344)}
DRAFT_CANVAS = {"landscape": (672, 384), "portrait": (384, 672)}
PICTURE = re.compile(r"\s*(?:shown in |with the composition of |(?:in|of|from) )?<Picture \d+>")
OPENINGS = ("For the target video", "How the reference pictures align")
WORKER = ("import signal, sys, time\n"
          "signal.signal(signal.SIGTERM, signal.SIG_DFL)   # not STUB_QWEN_TERM's SIG_IGN, which a child inherits\n"
          "time.sleep(float(sys.argv[1]))     # starting: its stdin is not read yet\n"
          "sys.stdin.read()                   # then it ends at EOF\n")


def text_only_prompt(prompt):
    """The prompt without its keyframe references (the real one's rules, simpler): {"prompt", "dropped"}."""
    kept, dropped, inline = [], [], 0
    for line in prompt.splitlines():
        if line.lstrip().startswith(OPENINGS):
            dropped.append("alignment line")
            continue
        line, n = PICTURE.subn("", line)
        inline += n
        kept.append(line)
    text = "\n".join(kept).strip()
    if not text:
        raise ValueError("nothing is left of the prompt without its keyframe references")
    return {"prompt": text, "dropped": dropped + ([f"{inline} x '<Picture N>'"] if inline else [])}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=1) + "\n")


def encode(out, task, prompt, frames, size, seed):
    """One encode's stand-in files in `out`: conditioning.pt and its receipt; returns its rows."""
    out.mkdir(parents=True)
    tokens = (size[0] // 32) * (size[1] // 32)
    text_tokens = len(prompt.split()) + 10
    rows = text_tokens + len(frames) * (tokens + 2)
    payload = f"stub conditioning {task} {rows}".encode()
    (out / "conditioning.pt").write_bytes(payload)
    write_json(out / "conditioning.json", {
        "status": "PASS", "task": task, "prompt": prompt, "seed": seed, "shape": [1, rows, 5120],
        "text_tokens": text_tokens, "prompt_tokens": "minimax", "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "input_spec": {f"{anchor}_frame": path for anchor, path in frames.items()},
        "prepared_media": [{"type": "image", "anchor": anchor, "shape": [size[1], size[0], 3]} for anchor in frames],
        "request_wall_s": 0.1, "stub": True})
    return rows


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--prompt-file", required=True, type=Path)
    p.add_argument("--orientation", required=True, choices=("landscape", "portrait"))
    p.add_argument("--first", type=Path)
    p.add_argument("--last", type=Path)
    p.add_argument("--resolution", default="576p", choices=("576p", "768p"))
    p.add_argument("--canvas", default="draft", choices=("draft", "canvas"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-id")
    p.add_argument("--no-text-only", action="store_true")
    p.add_argument("--package", type=Path)
    args = p.parse_args(argv[1:])
    signal.signal(signal.SIGTERM, signal.SIG_IGN if os.environ.get("STUB_QWEN_TERM") == "ignore"
                  else lambda signum, frame: sys.exit(128 + signum))
    if os.environ.get("STUB_QWEN_LOG"):
        with open(os.environ["STUB_QWEN_LOG"], "a") as log:
            log.write(json.dumps({"pid": os.getpid(), "argv": argv[1:]}) + "\n")
    case = args.case
    for sub in ("qwen", "qwen-text", "qwen-worker"):
        if (case / sub).exists():
            print(f"{TAG} refused: {case / sub} exists: use a new --case", flush=True)
            return 2
    prompt = args.prompt_file.read_text(encoding="utf-8")
    frames = {anchor: str(path) for anchor, path in (("first", args.first), ("last", args.last)) if path}
    for path in frames.values():
        if not Path(path).is_file():
            print(f"{TAG} refused: {path}: no such file", flush=True)
            return 2
    task = "fl2va" if frames else "t2va"
    size = DRAFT_CANVAS[args.orientation] if args.canvas == "draft" else CANVAS[(args.resolution, args.orientation)]
    text_only = None
    if task == "fl2va" and not args.no_text_only:
        try:
            text_only = text_only_prompt(prompt)
        except ValueError as error:
            print(f"{TAG} refused: {error} (--no-text-only skips the text-only encode)", flush=True)
            return 2
    start = time.monotonic()
    (case / "qwen-worker").mkdir(parents=True)
    (case / "qwen-worker" / "worker.log").write_text("stub Qwen worker\n")
    print(f"{TAG} {time.strftime('%H:%M:%S')} starting the Qwen worker (canvas {size[0]}x{size[1]})", file=sys.stderr,
          flush=True)
    worker = None
    if os.environ.get("STUB_QWEN_WORKER_S"):
        worker = subprocess.Popen(
            [sys.executable, "-c", WORKER, os.environ["STUB_QWEN_WORKER_S"], "runtime.worker", "--module", "qwen",
             "--options", str(case.resolve() / "qwen-worker" / "options.json")],
            stdin=subprocess.PIPE, start_new_session=True)
        (case / "qwen-worker" / "worker.pid").write_text(f"{worker.pid}\n")
    try:
        time.sleep(float(os.environ.get("STUB_QWEN_S", "0.5")))
        if os.environ.get("STUB_QWEN_FAIL") == "1":
            print(f"{TAG} failed: RuntimeError: Worker exited with code 1 (STUB_QWEN_FAIL)", flush=True)
            return 1
    finally:
        if worker is not None:             # the real one's worker.close(): the close op, then TERM to its group
            worker.stdin.close()
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait()
    summary = {"case": str(case), "case_id": args.case_id or case.name, "seed": args.seed, "task": task,
               "orientation": args.orientation, "resolution": args.resolution, "canvas_mode": args.canvas,
               "qwen_canvas": list(size), "image_tokens": (size[0] // 32) * (size[1] // 32) if frames else 0,
               "keyframes": list(frames), "frames": frames, "package": str(args.package), "ready_s": 0.1,
               "encodes": {"qwen": {"rows": encode(case / "qwen", task, prompt, frames, size, args.seed)}}}
    if text_only:
        rows = encode(case / "qwen-text", "t2va", text_only["prompt"], {}, size, args.seed)
        (case / "qwen-text" / "prompt.txt").write_text(text_only["prompt"], encoding="utf-8")
        write_json(case / "qwen-text" / "text-only.json", {"dropped": text_only["dropped"],
                                                            "chars": len(text_only["prompt"])})
        summary["encodes"]["qwen-text"] = {"rows": rows}
        summary["text_only"] = {"dropped": text_only["dropped"], "chars": len(text_only["prompt"])}
    summary["total_s"] = round(time.monotonic() - start, 2)
    write_json(case / "encode.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
