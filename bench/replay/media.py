"""Hashes and ffmpeg helpers for the replay bench. Standard library plus the ffmpeg/ffprobe commands; CPU only."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

FFMPEG_THREADS = "2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    """{width, height, frames, fps, duration_s, audio} of a video file (frames counted from the packets)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,nb_read_packets:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    num, den = (int(x) for x in video.get("r_frame_rate", "0/1").split("/"))
    return {"width": int(video["width"]), "height": int(video["height"]),
            "frames": int(video.get("nb_read_packets") or 0), "fps": num / den if den else 0.0,
            "duration_s": float(info.get("format", {}).get("duration") or 0.0),
            "audio": any(s.get("codec_type") == "audio" for s in info.get("streams", []))}


def stream_md5(path, kind):
    """MD5 of the decoded video frames (kind "v") or audio samples ("a"), as ffmpeg's md5 muxer prints it.
    Two files with the same value decode to the same pictures (or sound), whatever their container bytes."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", FFMPEG_THREADS, "-i", str(path),
                        "-map", f"0:{kind}:0", "-f", "md5", "-"], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg md5 of {path} ({kind}): {r.stderr.strip()[-300:]}")
    return r.stdout.strip().removeprefix("MD5=")


def common_size(path):
    """The size all comparisons use: 672x384, or 384x672 for a portrait clip (the draft canvas)."""
    info = probe(path)
    return (384, 672) if info["height"] > info["width"] else (672, 384)


def psnr_ssim(ref, test, size):
    """PSNR (average, dB) and SSIM (All) of `test` against `ref` over every frame, both scaled to `size` (lanczos,
    yuv420p). These measure how much the picture changed, not its quality."""
    w, h = size
    graph = (f"[0:v]scale={w}:{h}:flags=lanczos,format=yuv420p,split[t1][t2];"
             f"[1:v]scale={w}:{h}:flags=lanczos,format=yuv420p,split[r1][r2];[t1][r1]psnr[p];[t2][r2]ssim[s]")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-threads", FFMPEG_THREADS, "-i", str(test),
                        "-i", str(ref), "-filter_complex", graph, "-map", "[p]", "-f", "null", "-",
                        "-map", "[s]", "-f", "null", "-"], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg psnr/ssim: {r.stderr.strip()[-300:]}")
    p = re.search(r"PSNR .* average:([\d.]+|inf)", r.stderr)
    s = re.search(r"SSIM .* All:([\d.]+)", r.stderr)
    psnr = None if p is None else (float("inf") if p.group(1) == "inf" else float(p.group(1)))
    return psnr, (float(s.group(1)) if s else None)
