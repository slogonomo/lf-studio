"""Stub of runtime.config: the same request rules as the patched runtime, no models."""
import json
import re
from pathlib import Path

TASKS = ("t2va", "fl2va", "ref2va")
STAGE2_AUDIO_MODES = ("frozen", "joint")


def stage2_audio_mode():
    """The patch's SOL_H3_STAGE2_AUDIO check (runtime/config.py), same rules."""
    import os
    mode = os.environ.get("SOL_H3_STAGE2_AUDIO", "").strip().lower() or "frozen"
    if mode not in STAGE2_AUDIO_MODES:
        raise ValueError(f"SOL_H3_STAGE2_AUDIO must be one of {STAGE2_AUDIO_MODES}, "
                         f"not {os.environ.get('SOL_H3_STAGE2_AUDIO')!r}")
    return mode


def compile_dynamic():
    """The patch's SOL_H3_STATIC_SHAPES check (runtime/config.py), same rules."""
    import os
    raw = os.environ.get("SOL_H3_STATIC_SHAPES", "")
    value = raw.strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"SOL_H3_STATIC_SHAPES must be 1 or 0, not {raw!r}")
    return False if value == "1" else None


def dynamic_text_length(environ=None):
    """The patch's SOL_H3_DYNAMIC_TEXT_LENGTH check (runtime/config.py), same rules."""
    import os
    raw = (os.environ if environ is None else environ).get("SOL_H3_DYNAMIC_TEXT_LENGTH", "")
    value = raw.strip() or "1"
    if value not in ("0", "1"):
        raise ValueError(f"SOL_H3_DYNAMIC_TEXT_LENGTH must be 0 or 1, not {raw!r}")
    return value == "1"


STAGE2_KEYFRAME_MODES = ("off", "pin", "draft")


def stage2_keyframe_mode():
    """The patch's SOL_H3_STAGE2_KEYFRAME check (runtime/config.py), same rules."""
    import os
    raw = os.environ.get("SOL_H3_STAGE2_KEYFRAME", "")
    mode = raw.strip().lower() or "off"
    if mode not in STAGE2_KEYFRAME_MODES:
        raise ValueError(f"SOL_H3_STAGE2_KEYFRAME must be one of {STAGE2_KEYFRAME_MODES}, not {raw!r}")
    return mode


PROMPT_TOKEN_MODES = ("plain", "minimax")


def prompt_tokens_mode(value=None):
    """The patch's SOL_H3_PROMPT_TOKENS check (runtime/config.py), same rules."""
    import os
    raw = os.environ.get("SOL_H3_PROMPT_TOKENS", "") if value is None else value
    mode = str(raw).strip().lower() or "minimax"
    if mode not in PROMPT_TOKEN_MODES:
        raise ValueError(f"SOL_H3_PROMPT_TOKENS must be one of {PROMPT_TOKEN_MODES}, not {raw!r}")
    return mode


def load_paths(filename, *, task="t2va"):
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    return json.loads(Path(filename).read_text() or "{}")


def normalize_case(case, *, base_dir=None, default_task="t2va"):
    if not isinstance(case, dict):
        raise ValueError("each case must be a JSON object")
    name, prompt = case.get("case_id"), case.get("prompt")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("case_id must be a simple file-safe name")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    seed = case.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    task = case.get("task", default_task)
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")

    def media(value):
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"media input must be a file: {path}")
        return str(path)

    result = {"case_id": name, "prompt": prompt, "seed": seed, "task": task}
    first, last, refs = case.get("first_frame"), case.get("last_frame"), case.get("references", [])
    if task == "t2va":
        if first is not None or last is not None or refs:
            raise ValueError("T2VA does not accept frame or reference inputs; select fl2va or ref2va")
    elif task == "fl2va":
        if (first is None and last is None) or refs:
            raise ValueError("FL2VA requires first_frame, last_frame, or both, and no references")
        for key in ("first_frame", "last_frame"):
            if case.get(key) is not None:
                result[key] = media(case[key])
    else:
        if first is not None or last is not None or not refs:
            raise ValueError("Ref2VA requires references, not first_frame/last_frame")
        result["references"] = [{"type": r["type"], "path": media(r["path"])} for r in refs]
    return result
