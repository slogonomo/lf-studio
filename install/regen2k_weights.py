#!/usr/bin/env python3
"""The weights step of install/65-regen2k.sh (the experimental Upscale): NVIDIA's Open-H3-Regenerate-2K weights at
the pinned revision, in a weights directory as upscale/regen2k_job.py reads it (LFSTUDIO_UPSCALE_WEIGHTS):

    lora/open_h3_regenerate_2k_lora.safetensors        the merged LoRA, 12.5 GB
    lora/sharp_curve_branches.safetensors              the Sharp curve branches, 1.2 GB
    lora/adaln_t_table.bin                             Sharp's 1025x8 AdaLN table, 32,800 bytes
    latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors     0.7 GB

    python -I -B install/regen2k_weights.py --weights DIR [--upscaler-copy FILE]

A file that is already in place with NVIDIA's SHA-256 (download.py, EXPECTED) is kept. Else, a copy with that SHA-256
is linked, not copied: the file in the Hugging Face cache at the pinned revision (HF_HUB_CACHE or HF_HOME, read only),
or for the latent upscaler the studio's own H3-upscaler checkpoint (--upscaler-copy; the same bytes). Else it is
downloaded into the weights directory itself (huggingface_hub with local_dir: nothing is written into the cache) and
checked. The release does not carry the AdaLN table (it is derived from the base model): it is read by HTTP range
from one tensor of Comfy-Org/MiniMax-H3 at a pinned revision, as NVIDIA's download.py (fetch_curve) does, and
checked against its SHA-256. A file that fails its check is kept as <name>.bad, and the step fails.

The release files need huggingface_hub (the Regen-2K environment's Python runs this), which uses HF_ENDPOINT and
HF_TOKEN as it always does. The AdaLN table needs the standard library only; its range reads go to HF_ENDPOINT and
never carry a token: the repository is public, and urllib would send an Authorization header on to the host that
Hugging Face redirects to (its CDN). Exit status 0 when every file is in place, 1 otherwise (the last line says why).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

RELEASE = {"repo": "Efficient-Large-Model/Open-H3-Regenerate-2K",
           "revision": "5809c33730c18b6b1c827b40974855a4288641b3"}
# NVIDIA's download.py (Sana sol-engine 6e4a440, models/Open-H3-Regenerate-2K), EXPECTED and CURVE
FILES = {
    "lora/open_h3_regenerate_2k_lora.safetensors": "8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04",
    "lora/sharp_curve_branches.safetensors": "26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e",
    "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors":
        "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
}
UPSCALER = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
TABLE = "lora/adaln_t_table.bin"
CURVE = {"repo": "Comfy-Org/MiniMax-H3", "revision": "e5eb578a89295337b8ff433a035929ce0279e0b6",
         "file": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "tensor": "adaln_t_table",
         "dtype": "F32", "shape": [1025, 8], "bytes": 32800,
         "sha256": "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"}
MAX_HEADER = 64 << 20            # a safetensors header larger than this is not one


class Failed(Exception):
    pass


def say(text):
    print(f">>> {text}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has(path, digest):
    """Whether `path` (through any links) is a file with this SHA-256."""
    return Path(path).is_file() and sha256(path) == digest


def set_aside(path):
    """A file that failed its check: renamed <name>.bad (a link is only removed). Returns the new name or None."""
    path = Path(path)
    if path.is_symlink() or not path.exists():
        if path.is_symlink():
            path.unlink()
        return None
    bad = path.with_name(path.name + ".bad")
    os.replace(path, bad)
    return bad


def link(dest, target):
    """dest -> target (a symbolic link, absolute), replacing a link that is there; a real file there is set aside."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        moved = set_aside(dest)
        if moved:
            say(f"{dest} is not the expected file: kept as {moved}")
    tmp = dest.with_name(f".{dest.name}.link")
    if tmp.is_symlink():
        tmp.unlink()
    os.symlink(os.path.abspath(target), tmp)
    os.replace(tmp, dest)


def from_cache(name):
    """The file at the pinned revision in the Hugging Face cache, or None (read only: local_files_only)."""
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(RELEASE["repo"], name, revision=RELEASE["revision"], local_files_only=True))
    except Exception:                                  # not cached (LocalEntryNotFoundError), or no cache at all
        return None


def download(name, weights):
    """The file into the weights directory itself (local_dir: not into the Hugging Face cache)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(RELEASE["repo"], name, revision=RELEASE["revision"], local_dir=weights))


def release_file(weights, name, digest, upscaler_copy=None):
    dest = weights / name
    if has(dest, digest):
        say(f"have {name} (SHA-256 checked)")
        return
    for source in (from_cache(name), Path(upscaler_copy) if upscaler_copy and name == UPSCALER else None):
        if source is not None and has(source, digest):
            link(dest, source)
            say(f"linked {name} -> {source} (SHA-256 checked)")
            return
    if dest.exists() or dest.is_symlink():
        moved = set_aside(dest)
        if moved:
            say(f"{dest} is not the expected file: kept as {moved}")
    say(f"downloading {name} ({RELEASE['repo']} at {RELEASE['revision'][:7]}) into {weights}")
    got = download(name, weights)
    if not has(got, digest):
        moved = set_aside(got)
        raise Failed(f"{name} does not have NVIDIA's SHA-256 {digest} (kept as {moved})")
    say(f"downloaded {name} (SHA-256 checked)")


def read_range(url, start, length):
    """`length` bytes of `url` from `start`, by an HTTP Range request (never the whole file). No credentials: the
    file is public, and a redirect would take any header along to the next host."""
    headers = {"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity",
               "User-Agent": "lf-studio-install"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if response.status != 206:
            raise Failed(f"{url} ignored the Range request (HTTP {response.status}); refusing to download the whole "
                         "checkpoint")
        data = response.read(length + 1)
    if len(data) != length:
        raise Failed(f"{url}: {len(data)} bytes in a range of {length}")
    return data


def fetch_table(dest, endpoint):
    """Sharp's AdaLN table: one tensor of a Comfy-Org checkpoint, read by range (NVIDIA's download.py fetch_curve),
    checked, then written to dest (through a .part file)."""
    url = f"{endpoint.rstrip('/')}/{CURVE['repo']}/resolve/{CURVE['revision']}/{CURVE['file']}"
    size, = struct.unpack("<Q", read_range(url, 0, 8))
    if not 0 < size <= MAX_HEADER:
        raise Failed(f"{url} does not start with a safetensors header")
    try:
        header = json.loads(read_range(url, 8, size))
        tensor = header[CURVE["tensor"]]
        start, end = tensor["data_offsets"]
    except (ValueError, KeyError, TypeError) as error:
        raise Failed(f"{url}: no {CURVE['tensor']} in its header ({error})") from None
    if tensor.get("dtype") != CURVE["dtype"] or tensor.get("shape") != CURVE["shape"] or end - start != CURVE["bytes"]:
        raise Failed(f"{url}: {CURVE['tensor']} is {tensor.get('dtype')} {tensor.get('shape')}, not "
                     f"{CURVE['dtype']} {CURVE['shape']}")
    data = read_range(url, 8 + size + start, end - start)
    if hashlib.sha256(data).hexdigest() != CURVE["sha256"]:
        raise Failed(f"{CURVE['tensor']} from {url} does not have the SHA-256 {CURVE['sha256']}; nothing was written")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.part")
    part.write_bytes(data)
    os.replace(part, dest)


def table(weights, endpoint):
    dest = weights / TABLE
    if has(dest, CURVE["sha256"]):
        say(f"have {TABLE} (SHA-256 checked)")
        return
    if dest.exists() or dest.is_symlink():
        moved = set_aside(dest)
        if moved:
            say(f"{dest} is not the expected table: kept as {moved}")
    say(f"reading {TABLE} by range from {CURVE['repo']} at {CURVE['revision'][:7]} ({CURVE['file']})")
    fetch_table(dest, endpoint)
    say(f"wrote {TABLE} (SHA-256 checked)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True, type=Path, help="the weights directory (LFSTUDIO_UPSCALE_WEIGHTS)")
    ap.add_argument("--upscaler-copy", help="the studio's H3-upscaler checkpoint, linked when it has the same bytes")
    args = ap.parse_args(argv)
    weights = Path(os.path.abspath(args.weights))
    try:
        weights.mkdir(parents=True, exist_ok=True)
        for name, digest in FILES.items():
            release_file(weights, name, digest, args.upscaler_copy)
        table(weights, os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    except (Failed, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    except Exception as error:                         # huggingface_hub's own errors (network, 404, ...)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    say(f"the weights are in {weights}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
