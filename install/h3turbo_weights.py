#!/usr/bin/env python3
"""The LoRA step of install/66-h3turbo.sh (the experimental LightX2V Turbo): LightX2V's turbo LoRA files for MiniMax-H3
at a pinned revision of lightx2v/Minimax-h3-Turbo, in the folder its job reads them from (LFSTUDIO_H3TURBO_LORAS):

    minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors     4 steps: what the studio runs, 1.38 GB
    minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors     8 steps: not offered yet, 1.38 GB

    python3 -I -B install/h3turbo_weights.py --loras DIR

The revision and each file's size and SHA-256 are pinned here (the Hugging Face listing of that revision, and the
files the research probe ran with); nothing is looked up at install time. A file already in place with its size and
SHA-256 is kept (also through a link). Else it is downloaded from HF_ENDPOINT (default https://huggingface.co) into
<name>.partial, resumed with a Range request when a .partial is there from an earlier run, and renamed into place only
once its size and SHA-256 match. A file in the way that fails the check is kept as <name>.bad, a download that fails it
as <name>.partial.bad, and the step fails. The repository is public, so no token is sent, ever: no Authorization
header is set (urllib would pass one on through Hugging Face's redirect to its CDN) and HF_TOKEN is not read.
Standard library only. Exit status 0 when every file is in place, 1 otherwise (the last line says why).
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "lightx2v/Minimax-h3-Turbo"
REVISION = "30f485ed874aa44c8183348ad361c5eabad318da"
FILES = {
    "minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors":
        (1383677808, "97b775707bb5732e01dba0cf3833d3fe2947c953859d4d05ea3bd74894c43d2b"),
    "minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors":
        (1383677808, "fb41185df94bf01f9ac1aeedc2f07e5f748da3b237d18a16f5b3f5960839fdab"),
}
CHUNK = 8 << 20
MARGIN = 1 << 30                 # free space to keep beyond the files


class Failed(Exception):
    pass


def say(text):
    print(f">>> {text}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has(path, size, digest):
    """Whether `path` (through any links) is a file of this size and SHA-256."""
    path = Path(path)
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def set_aside(path, suffix=".bad"):
    """A file that failed its check, renamed <name><suffix> (a link is only removed). Returns the new name or None."""
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return None
    if not path.exists():
        return None
    bad = path.with_name(path.name + suffix)
    os.replace(path, bad)
    return bad


def download(url, part, size):
    """`url` into `part`, from where an earlier run stopped (Range) when the server takes that, else from the start.
    No credentials are sent."""
    start = part.stat().st_size if part.is_file() else 0
    if start > size:
        start = 0
    headers = {"User-Agent": "lf-studio-install", "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if start and response.status != 206:
                say(f"the server sent the whole file, not the rest from byte {start}: starting over")
                start = 0
            got = start
            with open(part, "ab" if start else "wb") as handle:
                for chunk in iter(lambda: response.read(CHUNK), b""):
                    got += len(chunk)
                    if got > size:
                        raise Failed(f"{url} is larger than the pinned {size} bytes")
                    handle.write(chunk)
    except urllib.error.HTTPError as error:
        if error.code == 416 and start == size:          # the .partial is complete already
            return
        raise Failed(f"could not download {url}: HTTP {error.code}") from None
    except (urllib.error.URLError, OSError) as error:
        why = error.reason if isinstance(error, urllib.error.URLError) else error
        raise Failed(f"the download of {url} broke off ({why}); run this again to go on from where it stopped") \
            from None


def lora_file(loras, name, size, digest, endpoint):
    dest = loras / name
    if has(dest, size, digest):
        say(f"have {name} (size and SHA-256 checked)")
        return
    if dest.exists() or dest.is_symlink():
        moved = set_aside(dest)
        say(f"{dest} is not the pinned file" + (f": kept as {moved}" if moved else ": its link removed"))
    part = dest.with_name(dest.name + ".partial")
    need = size - (part.stat().st_size if part.is_file() else 0) + MARGIN
    free = shutil.disk_usage(loras).free
    if free < need:
        raise Failed(f"{loras} has {free / 1e9:.1f} GB free; {name} needs {need / 1e9:.1f} GB more")
    url = f"{endpoint.rstrip('/')}/{REPO}/resolve/{REVISION}/{name}"
    say(f"downloading {name} ({size / 1e9:.2f} GB, {REPO} at {REVISION[:8]}) into {loras}")
    download(url, part, size)
    if not has(part, size, digest):
        moved = set_aside(part)
        raise Failed(f"{name} does not have the pinned size {size} and SHA-256 {digest} (kept as {moved})")
    os.replace(part, dest)
    say(f"downloaded {name} (size and SHA-256 checked)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--loras", required=True, type=Path, help="the LoRA folder (LFSTUDIO_H3TURBO_LORAS)")
    args = ap.parse_args(argv)
    loras = Path(os.path.abspath(args.loras))
    try:
        loras.mkdir(parents=True, exist_ok=True)
        for name, (size, digest) in FILES.items():
            lora_file(loras, name, size, digest, os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    except (Failed, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    say(f"the LoRA files are in {loras}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
