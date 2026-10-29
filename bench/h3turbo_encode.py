#!/usr/bin/env python3
"""H3 Turbo (experimental), Phase 0: one clip's prompt conditioning, encoded as the studio's server will encode it for
an H3 Turbo clip (its encode_prompt), for jobs/h3turbo_job.py to read. GPU: a transient Qwen worker (~15 GiB while it
lives); run it with the studio and other GPU apps stopped. bench/h3turbo-encode.sh runs it with the studio's settings
and the Stage-1 Python, in the patched package, as bin/lf-studio runs the server:

    bench/h3turbo-encode.sh --case DIR --prompt-file FILE|--prompt-from-case DIR --orientation portrait
        [--first IMG] [--last IMG] [--resolution 576p|768p] [--canvas draft|canvas] [--seed N] [--case-id ID]
        [--no-text-only] [--check]

--prompt-from-case takes an earlier clip's prompt as it was encoded (its qwen/conditioning.json), e.g. to encode it
again at the clip's canvas.

How: the patch's own runtime.pipeline.Worker("qwen", <case>/qwen-worker, paths, config), started the way
Pipeline.start() starts its transient Qwen (the NVFP4 encoder in Docker through qwen_python.sh, JSON lines through
runtime/worker.py), with paths from the studio's manifest for the task (LFSTUDIO_PATHS) and config from
runtime.config.load_recipe(task, orientation=...) (the frozen-recipe check, the SOL_H3_PROMPT_TOKENS mode). The
canvas Qwen sees the keyframes at is the recipe's stage1 height and width (runtime/qwen.py draft_canvas, read when the
worker starts): --canvas draft keeps the 384p draft canvas every probe number was measured with (672x384 or 384x672:
252 image tokens per keyframe), --canvas canvas sets it to the generation canvas of --resolution (576 tokens at 576p,
1008 at 768p). The recipe goes to this Qwen worker only, never to a Stage 1. One worker, then:
1. the clip's encode: t2va without frames, fl2va with --first and/or --last (a last frame alone is not offered by the
   studio yet: the patch's Qwen stretches the first existing keyframe, diffusers cover-crops a lone last image);
2. for fl2va, unless --no-text-only, a t2va encode of text_only_prompt(prompt), the Upscale's text-only input;
then the worker closes (also on SIGTERM).

text_only_prompt drops what refers to the keyframes: the alignment line in each form the studio writes (the prompt
guide's first-frame line, its first + last line and last-only line with an em dash, the page's alignLine() with a
hyphen; on a line of its own or in front of the text), the inline references the research probe's rewrite took out
(" shown in <Picture N>", "with the composition of <Picture N>", " in/of/from <Picture N>"), and any <Picture N> left
after them. The guide's own examples name the pictures without angle brackets too ("the composition of Picture 1"):
those go the same way.

Output in --case (a new or empty directory under $SOL_H3_SPARK_RUNTIME_ROOT, which is all the Qwen container sees;
the frames must be under it too): qwen/conditioning.{pt,json} (the clip's), for fl2va qwen-text/conditioning.{pt,json},
qwen-text/prompt.txt and qwen-text/text-only.json (what was dropped), qwen-worker/ (the worker's log) and encode.json
(the canvas, the rows, the times, and the job.json fields that go with it). Each encode's receipt is checked: its task,
prompt, shape, payload sha256, the canvas of its images and its vision rows (rows - text_tokens = keyframes x (image
tokens + 2)). --check validates everything, prints the plan and starts nothing, writes nothing.
Exit status: 0 done, 1 an encode failed or its receipt does not match, 2 refused (nothing started), 143 SIGTERM.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOB_SCRIPT = REPO / "jobs" / "h3turbo_job.py"
TAG = "[h3turbo-encode]"
EXIT_FAILED, EXIT_REFUSED = 1, 2
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")

# ---------------------------------------------------------------- the text-only prompt (standard library only)

# the alignment line, as the prompt guide (bin/lf-cli GUIDE, docs/API.md) and the page (webui/index.html alignLine)
# write it: first frame; first + last and last only (an em dash in the guide, a hyphen on the page; "Picture N" with
# or without angle brackets, "(from [Shot N])" or "(from Shot N)" or nothing)
PICTURE = r"(?:<Picture \d+>|\bPicture \d+\b)"
ALIGNMENT = (
    re.compile(r"For the target video, at [\d.]+ seconds into the target video, " + PICTURE +
               r"(?: \(from \[?Shot \d+\]?\))? is fully referenced\.?"),
    re.compile(r"How the reference pictures align with the target video\s*[—–:-]\s*.*?\bmark"
               r"(?: of the target video)?\.(?=\s|$)"))
OPENINGS = ("For the target video", "How the reference pictures align")
# the inline references, in this order (the research probe's rules), then anything left
INLINE = ((re.compile(r" shown in " + PICTURE), "", " shown in <Picture N>"),
          (re.compile(r"with the composition of " + PICTURE), "with this composition",
           "with the composition of <Picture N> -> with this composition"),
          (re.compile(r" (?:in|of|from) " + PICTURE), "", " in/of/from <Picture N>"),
          (re.compile(PICTURE), "", "<Picture N>"))


def _tidy(line):
    """A line a reference was cut out of: no double spaces, no space before punctuation, nothing trailing."""
    body = line.rstrip("\r\n")
    ending = line[len(body):]
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r" +([,.;:!?)])", r"\1", body)
    return body.rstrip() + ending


def text_only_prompt(prompt):
    """An fl2va prompt without its keyframe references, for a t2va (text-only) encode: {"prompt": text, "dropped":
    [what went]}. Lines keep their own line endings, and lines with nothing to drop stay as they are; a dropped
    alignment line takes the blank line after it. ValueError when nothing is left."""
    dropped, counts, kept, after_alignment = [], {}, [], False
    for line in prompt.splitlines(keepends=True):
        if after_alignment and not line.strip():
            after_alignment = False
            continue                                    # the blank line after a dropped alignment line
        after_alignment = False
        body, aligned = line, 0
        for pattern in ALIGNMENT:
            body, n = pattern.subn("", body)
            aligned += n
        if body.lstrip().startswith(OPENINGS):          # an alignment line in a form of its own: all of it
            body, aligned = "", aligned + 1
        dropped += ["alignment line"] * aligned
        if aligned and not body.strip():
            after_alignment = True
            continue
        if aligned:
            body = body.lstrip()
        edited = bool(aligned)
        for pattern, replacement, label in INLINE:
            body, n = pattern.subn(replacement, body)
            if n:
                counts[label] = counts.get(label, 0) + n
                edited = True
        kept.append(_tidy(body) if edited else line)
    text = "".join(kept).strip()
    if not text:
        raise ValueError("nothing is left of the prompt without its keyframe references")
    return {"prompt": text, "dropped": dropped + [f"{n} x '{label}'" for label, n in counts.items()]}


# ---------------------------------------------------------------- the plan (standard library only)

class Refused(ValueError):
    """Bad arguments or inputs: nothing was started (exit status 2)."""


class Failed(RuntimeError):
    """An encode failed or its receipt does not match (exit status 1)."""


def say(message):
    print(f"{TAG} {time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def load_job_module(path=JOB_SCRIPT):
    """jobs/h3turbo_job.py by its path, for its canvas and token rules (standard library on import)."""
    spec = importlib.util.spec_from_file_location("h3turbo_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", required=True, type=Path,
                        help="the clip's directory under $SOL_H3_SPARK_RUNTIME_ROOT (made if needed)")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--prompt-file", type=Path, help="the prompt, as the studio would send it (UTF-8)")
    text.add_argument("--prompt", help="the prompt itself")
    text.add_argument("--prompt-from-case", type=Path, metavar="CASE_DIR",
                      help="an earlier clip's prompt, exactly as it was encoded: CASE_DIR/qwen/conditioning.json")
    parser.add_argument("--orientation", required=True, choices=("landscape", "portrait"))
    parser.add_argument("--first", type=Path, help="the first frame (fl2va)")
    parser.add_argument("--last", type=Path, help="the last frame (fl2va)")
    parser.add_argument("--resolution", default="576p", choices=("576p", "768p"),
                        help="the clip's size: the canvas of --canvas canvas (576p)")
    parser.add_argument("--canvas", default="draft", choices=("draft", "canvas"),
                        help="where Qwen sees the keyframes: the studio's 384p draft (default) or the clip's canvas")
    parser.add_argument("--seed", type=int, default=42, help="recorded in conditioning.json (42)")
    parser.add_argument("--case-id", help="the case id Qwen records (default: the --case directory's name)")
    parser.add_argument("--no-text-only", action="store_true", help="fl2va: no text-only encode")
    parser.add_argument("--package", type=Path, default=Path.cwd(),
                        help="the patched Sol-H3-Spark package (default: the working directory)")
    parser.add_argument("--check", action="store_true", help="validate, print the plan; start nothing, write nothing")
    return parser.parse_args(argv)


def _under(path, root, what):
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise Refused(f"{what} {path} is not under SOL_H3_SPARK_RUNTIME_ROOT {root} (the Qwen container sees only "
                      "that)") from None
    return path


def make_plan(args, environ, job):
    """The arguments and the environment checked -> the plan (no runtime import). Raises Refused."""
    raw = environ.get("SOL_H3_SPARK_RUNTIME_ROOT", "")
    if not raw or not os.path.isabs(os.path.expanduser(raw)):
        raise Refused("SOL_H3_SPARK_RUNTIME_ROOT must be set to an absolute path (bench/h3turbo-encode.sh sets the "
                      "studio's)")
    runtime = Path(raw).expanduser().resolve()
    case = _under(Path(os.path.abspath(args.case)), runtime, "--case")
    for sub in ("qwen", "qwen-text", "qwen-worker"):
        if (case / sub).exists():
            raise Refused(f"{case / sub} exists: use a new --case")
    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise Refused(f"cannot read --prompt-file {args.prompt_file}: {error}") from None
    elif args.prompt_from_case is not None:
        receipt = args.prompt_from_case / "qwen" / "conditioning.json"
        try:
            prompt = json.loads(receipt.read_text(encoding="utf-8")).get("prompt")
        except (OSError, ValueError, AttributeError) as error:
            raise Refused(f"cannot read the prompt of --prompt-from-case: {receipt}: {error}") from None
        if not isinstance(prompt, str):
            raise Refused(f"{receipt} has no prompt")
    else:
        prompt = args.prompt
    if not prompt or not prompt.strip():
        raise Refused("the prompt is empty")
    frames = {}
    for anchor, path in (("first", args.first), ("last", args.last)):
        if path is not None:
            if not path.is_file():
                raise Refused(f"--{anchor} {path}: no such file")
            frames[anchor] = _under(Path(os.path.abspath(path)), runtime, f"--{anchor}")
    task = "fl2va" if frames else "t2va"
    case_id = args.case_id or case.name
    if not CASE_ID_RE.match(case_id) or case_id in ("warmup", "stage1-worker", "stage2-worker", "qwen-worker",
                                                     "warmup-qwen-worker"):
        raise Refused(f"case id {case_id!r}: letters, digits, '-' and '_' (not a reserved name); give --case-id")
    if not 0 <= args.seed < 2 ** 63:
        raise Refused(f"--seed {args.seed}: 0 to 2**63-1")
    width, height = job.qwen_canvas(args.orientation, args.resolution, args.canvas)
    template = environ.get("LFSTUDIO_PATHS") or str(
        Path(environ.get("LFSTUDIO_HOME") or Path.home() / "sol-h3").expanduser() / "paths-{task}.json")
    if "{task}" not in template:
        raise Refused(f"LFSTUDIO_PATHS must contain {{task}} (got {template})")
    manifest = Path(template.format(task=task)).expanduser()
    if not manifest.is_file():
        raise Refused(f"no path manifest for {task} at {manifest} (install/70-manifests.sh)")
    package = args.package.resolve()
    if not (package / "runtime" / "pipeline.py").is_file():
        raise Refused(f"--package {package}: no runtime/pipeline.py (the patched Sol-H3-Spark package)")
    plan = {"case": str(case), "case_id": case_id, "seed": args.seed, "task": task, "orientation": args.orientation,
            "resolution": args.resolution, "canvas_mode": args.canvas, "qwen_canvas": [width, height],
            "image_tokens": job.image_tokens(width, height) if frames else 0, "keyframes": list(frames),
            "frames": {anchor: str(path) for anchor, path in frames.items()}, "manifest": str(manifest),
            "package": str(package), "runtime_root": str(runtime), "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "notes": []}
    if frames and "first" not in frames:
        plan["notes"].append("a last frame alone: the studio does not offer it yet (Qwen stretches it, diffusers "
                             "cover-crops it)")
    if task == "fl2va" and not args.no_text_only:
        try:
            plan["text_only"] = text_only_prompt(prompt)
        except ValueError as error:
            raise Refused(f"{error} (--no-text-only skips the text-only encode)") from None
    return plan


def runtime_side(plan):
    """The patch's host-side objects, imported as the server imports them, and the recipe of the plan (no worker
    started): (Worker, paths, config, normalized case). Raises Refused."""
    package = plan["package"]
    os.chdir(package)
    sys.path.insert(0, package)
    try:
        from runtime.config import load_paths, load_recipe, normalize_case, prompt_tokens_mode
        from runtime.pipeline import Worker
    except ImportError as error:
        raise Refused(f"cannot import the patched runtime from {package}: {error}") from None
    try:
        mode = prompt_tokens_mode()
        config = load_recipe(plan["task"], orientation=plan["orientation"])
        paths = load_paths(plan["manifest"], task=plan["task"])
        case = {"case_id": plan["case_id"], "prompt": plan["prompt"], "seed": plan["seed"], "task": plan["task"]}
        case.update({f"{anchor}_frame": path for anchor, path in plan["frames"].items()})
        case = normalize_case(case)
    except (OSError, ValueError) as error:
        raise Refused(str(error)) from None
    job = load_job_module()
    found = [token for token in job.MINIMAX_PROMPT_TOKENS if token in plan["prompt"]]
    if mode != "minimax" and found:
        raise Refused(f"SOL_H3_PROMPT_TOKENS={mode} would split MiniMax's tokens {found} into text tokens H3 never "
                      "saw; jobs/h3turbo_job.py refuses such a conditioning")
    if plan["canvas_mode"] == "canvas":
        config["stage1"]["height"], config["stage1"]["width"] = plan["qwen_canvas"][1], plan["qwen_canvas"][0]
    plan["prompt_tokens"] = mode
    return Worker, paths, config, case


def check_receipt(receipt, out, plan, task, prompt, keyframes):
    """An encode's receipt (conditioning.json) against what was asked -> its rows. Raises Failed."""
    width, height = plan["qwen_canvas"]
    shape = receipt.get("shape")
    rows = shape[1] if isinstance(shape, list) and len(shape) == 3 else None
    payload = out / "conditioning.pt"
    digest = hashlib.sha256(payload.read_bytes()).hexdigest() if payload.is_file() else None
    images = [m for m in receipt.get("prepared_media") or [] if m.get("type") == "image"]
    problems = [what for what, ok in (
        ("status", receipt.get("status") == "PASS"), ("task", receipt.get("task") == task),
        ("prompt", receipt.get("prompt") == prompt), ("prompt_tokens", receipt.get("prompt_tokens") ==
                                                      plan["prompt_tokens"]),
        ("shape", rows is not None and shape == [1, rows, 5120]), ("payload_sha256", digest is not None and
                                                                   digest == receipt.get("payload_sha256")),
        ("images", [m.get("anchor") for m in images] == keyframes),
        ("canvas", all(list(m.get("shape") or [])[:2] == [height, width] for m in images))) if not ok]
    vision = rows - receipt["text_tokens"] if rows is not None and isinstance(receipt.get("text_tokens"), int) else None
    want = len(keyframes) * (plan["image_tokens"] + 2) if keyframes else 0
    if vision is not None and vision != want:
        problems.append(f"vision rows {vision}, want {want}")
    if problems:
        raise Failed(f"the {out.name}/ encode does not match: {problems}")
    return {"rows": rows, "text_tokens": receipt.get("text_tokens"), "vision_rows": vision,
            "payload_sha256": digest, "request_wall_s": receipt.get("request_wall_s")}


def encode(plan, Worker, paths, config, case):
    """The worker, the encodes, the receipts checked; writes encode.json. Returns the summary."""
    root = Path(plan["case"])
    root.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in plan.items() if k not in ("prompt", "text_only")}
    start = time.monotonic()
    say(f"starting the Qwen worker in {root / 'qwen-worker'} (canvas {plan['qwen_canvas'][0]}x"
        f"{plan['qwen_canvas'][1]})")
    worker = Worker("qwen", root / "qwen-worker", paths, config)     # blocks until ready.json
    summary["ready_s"] = round(time.monotonic() - start, 2)
    summary["encodes"] = {}
    try:
        receipt = worker.call(case=case, output_root=str(root / "qwen"))
        summary["encodes"]["qwen"] = check_receipt(receipt, root / "qwen", plan, plan["task"], plan["prompt"],
                                                   plan["keyframes"])
        say(f"qwen/: {summary['encodes']['qwen']['rows']} rows")
        if "text_only" in plan:
            text = plan["text_only"]["prompt"]
            receipt = worker.call(case={"case_id": plan["case_id"], "seed": plan["seed"], "task": "t2va",
                                        "prompt": text}, output_root=str(root / "qwen-text"))
            (root / "qwen-text" / "prompt.txt").write_text(text, encoding="utf-8")
            (root / "qwen-text" / "text-only.json").write_text(json.dumps(
                {"dropped": plan["text_only"]["dropped"], "chars": len(text),
                 "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1) + "\n")
            summary["encodes"]["qwen-text"] = check_receipt(receipt, root / "qwen-text", plan, "t2va", text, [])
            summary["text_only"] = {"dropped": plan["text_only"]["dropped"], "chars": len(text)}
            say(f"qwen-text/: {summary['encodes']['qwen-text']['rows']} rows")
    finally:
        worker.close()                                          # the close op, then TERM/KILL of its group
    summary["total_s"] = round(time.monotonic() - start, 2)
    summary["for_job"] = {"task": plan["task"], "orientation": plan["orientation"], "resolution": plan["resolution"],
                          "conditioning": str(root / "qwen" / "conditioning.pt"), "qwen_canvas": plan["canvas_mode"],
                          "first_frame": plan["frames"].get("first"), "last_frame": plan["frames"].get("last")}
    (root / "encode.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    args = parse_args(argv)
    try:
        plan = make_plan(args, environ, load_job_module())
        Worker, paths, config, case = runtime_side(plan)
    except Refused as error:
        print(f"{TAG} refused: {error}", flush=True)
        return EXIT_REFUSED
    if args.check:
        shown = {k: v for k, v in plan.items() if k != "prompt"}
        shown["recipe_stage1"] = {k: config["stage1"].get(k) for k in ("height", "width", "prompt_tokens")}
        print(json.dumps({"ok": True, **shown}, indent=1), flush=True)
        return 0
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))   # so `finally` closes the worker
    try:
        summary = encode(plan, Worker, paths, config, case)
    except Failed as error:
        print(f"{TAG} failed: {error}", flush=True)
        return EXIT_FAILED
    except Exception as error:  # noqa: BLE001 - the worker's own failure (its log is in qwen-worker/)
        print(f"{TAG} failed: {type(error).__name__}: {error}", flush=True)
        return EXIT_FAILED
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
