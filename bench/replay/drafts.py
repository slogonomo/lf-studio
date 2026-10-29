"""The replay bench's draft store. Standard library only.

A draft is the Stage-1 output of one studio clip: `stage1/capture.json` and `stage1/stage1_direct_tensors.pt` in its
case folder (`<runtime root>/outputs/webui/<session>/<case>`). Stage 2 reads nothing else, apart from an fl2va clip's
first frame with SOL_H3_STAGE2_KEYFRAME=pin. `add` copies them, the clip's keyframes (to `inputs/`, with the copy's
capture.json pointing there) and the studio's final MP4 and Stage-2 receipt for comparison, into
`<runtime root>/outputs/bench/replay-drafts/<name>/`. A copy, because deleting a clip in the studio moves its folder
and its input files to the trash, and emptying the trash removes them.

draft.json records what the bench needs: task, format, length (from the capture's tensor shapes, with the package's
own geometry rules), seed, the studio clip's hash and whether it was a warm run, and for dialogue drafts the
speaking turns that the SyncNet table uses.
"""
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

from . import media

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
TURNS = re.compile(r"[A-Za-z]:\d+(\.\d+)?-\d+(\.\d+)?(,[A-Za-z]:\d+(\.\d+)?-\d+(\.\d+)?)*")
MOUTH = re.compile(r"\d+:\d+:\d+:\d+")


def load_geometry(package):
    """The installed package's runtime/geometry.py (it imports nothing), loaded on its own."""
    path = Path(package) / "runtime" / "geometry.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: not a patched Sol-H3-Spark package (install/10-sana.sh)")
    spec = importlib.util.spec_from_file_location("_replay_runtime_geometry", path)
    module = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # write nothing into the installed package
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = keep
    return module


def infer_format(capture, geometry):
    """(seconds, orientation, frame geometry) of a capture, from its H3 latent and audio shapes. Seconds is the
    whole-second length the studio asked for (5 = the frozen 124 -> 121 route)."""
    h3, audio = capture["h3_normalized"]["shape"], capture["audio"]["shape"]
    if len(h3) != 5 or len(audio) != 3:
        raise ValueError(f"unexpected capture shapes {h3} {audio}")
    orientation = "portrait" if h3[3] > h3[4] else "landscape"
    for seconds in range(int(geometry.MIN_SECONDS), int(geometry.MAX_SECONDS) + 1):
        frames = geometry.frame_geometry(geometry.draft_frames_for_seconds(seconds))
        if frames["h3_latent_frames"] == h3[2] and frames["audio_samples"] == audio[2]:
            return seconds, orientation, frames
    raise ValueError(f"no clip length gives H3 latent frames {h3[2]} and {audio[2]} audio samples")


def parse_turns(spec):
    """'R:0-1.2,L:2.4-4.7' -> [{speaker, t0, t1}] (the format the SyncNet scorer takes)."""
    if not TURNS.fullmatch(spec or ""):
        raise ValueError(f"turns must look like 'R:0-1.2,L:2.4-4.7' (speaker:start-end in seconds), not {spec!r}")
    turns = []
    for part in spec.split(","):
        speaker, span = part.split(":")
        t0, t1 = (float(x) for x in span.split("-"))
        if t1 <= t0:
            raise ValueError(f"turn {part}: the end must come after the start")
        turns.append({"speaker": speaker, "t0": t0, "t1": t1})
    return turns


def store_dir(runtime_root):
    return Path(runtime_root) / "outputs" / "bench" / "replay-drafts"


def add(store, name, case_dir, geometry, *, dialogue=False, turns=None, mouth=None, note="", force=False):
    """Copy one studio clip's draft into the store. Returns its draft.json content."""
    if not NAME.fullmatch(name):
        raise ValueError("a draft name is letters, digits, _ and -, at most 64 characters")
    case = Path(case_dir).expanduser().resolve(strict=True)
    capture_path = case / "stage1" / "capture.json"
    payload_path = case / "stage1" / "stage1_direct_tensors.pt"
    if not capture_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError(f"{case} has no stage1/capture.json and stage1_direct_tensors.pt: pick a clip's "
                                "case folder, <runtime root>/outputs/webui/<session>/<case>")
    capture = json.loads(capture_path.read_text())
    if capture.get("status") != "PASS" or capture.get("latent_only_transfer") is not True:
        raise ValueError(f"{capture_path} is not a finished latent-only Stage-1 capture")
    if media.sha256(payload_path) != capture.get("payload_sha256"):
        raise ValueError(f"{payload_path} does not match the payload_sha256 in its capture.json")
    if turns is not None:
        parse_turns(turns)
    if mouth is not None and not MOUTH.fullmatch(mouth):
        raise ValueError("mouth crop must be X:Y:W:H in output pixels")
    seconds, orientation, frames = infer_format(capture, geometry)
    target = Path(store) / name
    if target.exists():
        if not force:
            raise FileExistsError(f"draft {name} exists ({target}); --force replaces it")
        shutil.rmtree(target)
    tmp = target.with_name(f".{name}.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "stage1").mkdir(parents=True)
    inputs = copy_keyframes(capture, tmp, target)
    if inputs:              # the copy's capture.json names the copied keyframes (the rest of it is the studio's)
        (tmp / "stage1" / "capture.json").write_text(json.dumps(capture, indent=2, ensure_ascii=False) + "\n")
    else:
        shutil.copy2(capture_path, tmp / "stage1" / "capture.json")
    shutil.copy2(payload_path, tmp / "stage1" / "stage1_direct_tensors.pt")
    studio = None
    receipt_path = case / "stage2" / "stage2_result.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        mp4 = Path(receipt.get("output") or "")
        if receipt.get("status") == "PASS" and mp4.name and (case / "stage2" / mp4.name).is_file():
            (tmp / "studio").mkdir()
            shutil.copy2(receipt_path, tmp / "studio" / "stage2_result.json")
            shutil.copy2(case / "stage2" / mp4.name, tmp / "studio" / mp4.name)
            sha = media.sha256(tmp / "studio" / mp4.name)
            index = receipt.get("request_index")
            studio = {"mp4": f"studio/{mp4.name}", "sha256": sha, "sha256_matches_receipt": sha == receipt.get("output_sha256"),
                      "request_index": index, "warm": isinstance(index, int) and index >= 1,
                      "stage2_audio": receipt.get("stage2_audio") or "joint (before 0.3)",
                      "stage2_keyframe": receipt.get("stage2_keyframe") or "off",
                      "stage2_s": receipt.get("stage2_request_s"),
                      "phases_s": (receipt.get("result") or {}).get("phases_s")}
    info = {"name": name, "added": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(case),
            "task": capture["task"], "orientation": orientation, "seconds": seconds,
            "draft_frames": frames["draft_frames"], "output_frames": frames["output_frames"],
            "h3_shape": capture["h3_normalized"]["shape"], "audio_shape": capture["audio"]["shape"],
            "request_id": capture.get("request_id", capture.get("case_id")), "seed": capture["seed"],
            "payload_sha256": capture["payload_sha256"], "prompt": capture.get("prompt", ""),
            "dialogue": bool(dialogue or turns), "turns": turns, "mouth": mouth, "note": note, "studio": studio,
            "inputs": inputs}
    (tmp / "draft.json").write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    tmp.rename(target)
    return info


def copy_keyframes(capture, tmp, target):
    """Copy an fl2va clip's first and last frame into the draft (tmp, renamed to target when done) and point the
    capture at the copies. Returns {key: {source, file, sha256}}; a keyframe that is gone is recorded as missing and
    keeps its path (a pin of it then fails, naming that path). The capture's input_sha256 (patches since #57), when
    present, must match the copy."""
    spec, recorded, inputs = capture.get("input_spec") or {}, capture.get("input_sha256") or {}, {}
    for key in ("first_frame", "last_frame"):
        source = spec.get(key)
        if not source:
            continue
        path = Path(source)
        if not path.is_file():
            inputs[key] = {"source": source, "missing": True}
            continue
        (tmp / "inputs").mkdir(exist_ok=True)
        name = f"{key}{path.suffix.lower()}"
        shutil.copy2(path, tmp / "inputs" / name)
        sha = media.sha256(tmp / "inputs" / name)
        if recorded.get(key) not in (None, sha):
            raise ValueError(f"{path} is not the {key.replace('_', ' ')} Stage 1 read (its sha256 differs from the "
                             "capture's input_sha256)")
        inputs[key] = {"source": source, "file": f"inputs/{name}", "sha256": sha}
        spec[key] = str(target / "inputs" / name)
    return inputs


def load(store, name):
    path = Path(store) / name / "draft.json"
    if not path.is_file():
        raise FileNotFoundError(f"no draft {name} in {store} (bench/replay.sh add NAME CASE_DIR)")
    info = json.loads(path.read_text())
    info["dir"] = str(path.parent)
    return info


def all_drafts(store):
    store = Path(store)
    return [load(store, p.parent.name) for p in sorted(store.glob("*/draft.json"))] if store.is_dir() else []


def describe(info):
    studio = info.get("studio")
    clip = ("no studio clip" if not studio else
            f"studio clip {'warm' if studio['warm'] else 'first in its process'}, {studio['stage2_audio']} audio")
    talk = f"dialogue, turns {info['turns']}" if info.get("turns") else ("dialogue, turns not set" if info.get("dialogue")
                                                                         else "no dialogue")
    keys = info.get("inputs") or {}
    frames = "".join(f", {key.replace('_', ' ')}{' MISSING' if value.get('missing') else ''}"
                     for key, value in keys.items())
    return (f"{info['name']}: {info['task']} {info['orientation']} {info['seconds']} s "
            f"({info['draft_frames']} -> {info['output_frames']} frames), seed {info['seed']}{frames}, {talk}; {clip}")
