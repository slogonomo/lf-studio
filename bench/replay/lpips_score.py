"""LPIPS (AlexNet) between two clips on the CPU: lpips_score.py --size WxH [--every N] REF TEST -> JSON on stdout.

Run with a Python that has torch (the CPU build is enough), numpy and lpips (`pip install lpips`, BSD-2-Clause; its
first use downloads torchvision's AlexNet weights into TORCH_HOME). `bench/replay.sh quality RUN --lpips PYTHON`
calls it once per pair. Both clips are scaled to WxH with ffmpeg (lanczos); every Nth frame is compared, in batches
of 8, with 4 CPU threads. LPIPS measures how different two pictures look (0 = identical), not their quality.
"""
import argparse
import json
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def frames(path, width, height, np):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", path, "-vf",
                          f"scale={width}:{height}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", required=True, help="WxH, e.g. 672x384")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("ref")
    ap.add_argument("test")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))
    import numpy as np
    import torch
    import lpips
    torch.set_num_threads(4)
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    a, b = frames(args.ref, width, height, np), frames(args.test, width, height, np)
    picked = list(range(0, min(len(a), len(b)), max(1, args.every)))
    values = []
    with torch.no_grad():
        for i in range(0, len(picked), 8):
            idx = picked[i:i + 8]
            ta = torch.from_numpy(a[idx].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1
            tb = torch.from_numpy(b[idx].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1
            values += net(ta, tb).flatten().tolist()
    json.dump({"mean": float(np.mean(values)), "max": float(np.max(values)), "frames": len(values)}, sys.stdout)
    print()


if __name__ == "__main__":
    main()
