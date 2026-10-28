"""lf-studio: faster safetensors -> GPU weight loading on DGX Spark (GB10).

Part of lf-studio (Apache-2.0); not part of the upstream Sol-H3-Spark package.

On GB10 a host->GPU copy whose source is a file-backed mmap runs at ~1 GiB/s, versus ~18 GiB/s from
anonymous RAM (measured on a DGX Spark with a warm page cache; any loader that copies straight from an
mmap to the GPU pays this on GB10).
`safetensors.safe_open(..., device="cuda").get_tensor()` builds the tensor on the mmap and copies it to the GPU.
This wrapper opens the file on CPU and reads it with O_DIRECT (below); if that is unavailable it clones each
tensor into anonymous RAM (one tensor at a time, transient), then copies it to the device. Byte-identical
weights. Disable with SPARK_FAST_SAFETENSORS=0.
"""
# O_DIRECT reads (SPARK_DIRECT_IO, default 1). The NVMe streams ~11.8 GB/s with large direct reads, but the loaders
# pull weights through mmap page faults with the default 128 KiB read-ahead (~1.3-2 GB/s), and whatever they read
# also fills the page cache, which on GB10 competes with GPU memory. _DirectReader reads each safetensors file in
# big aligned chunks (parallel preadv, O_DIRECT) into a few reusable anonymous windows; get_tensor() slices a window
# and copies to the GPU. Falls back to the mmap path per file if O_DIRECT is unavailable or access is too random.
# SPARK_DIRECT_AHEAD=0 turns off the forward read-ahead of the next window.

from __future__ import annotations

import json
import mmap
import os
import struct
from concurrent.futures import ThreadPoolExecutor

ENABLED = os.environ.get("SPARK_FAST_SAFETENSORS", "1") == "1"
DIRECT = os.environ.get("SPARK_DIRECT_IO", "1") == "1"
_ALIGN = 4096
_WINDOW = 256 * 1024 * 1024
_READERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="spark-direct")
_DTYPES = None


def _torch_dtypes():
    global _DTYPES
    if _DTYPES is None:
        import torch
        _DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
                   "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
                   "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
    return _DTYPES


_AHEAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spark-direct-ahead")


def _ceil(n):
    return -(-n // _ALIGN) * _ALIGN


class _DirectReader:
    """Chunked O_DIRECT reads of one safetensors file into up to 3 reusable page-aligned windows (LRU).
    While the caller copies tensors out of one window, the next window is read in the background."""

    def __init__(self, path, windows=3):
        with open(path, "rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        header.pop("__metadata__", None)
        self.base = 8 + size
        self.meta = {name: (info["dtype"], info["shape"], *info["data_offsets"]) for name, info in header.items()}
        self.file_size = os.path.getsize(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.windows = []            # {"buf", "start", "end", "future"}; most recently used first
        self.max_windows = windows
        self.bytes_read = 0
        self.gave_up = False
        self.ahead = os.environ.get("SPARK_DIRECT_AHEAD", "1") == "1"
        self.max_start = -1          # streaming frontier: read ahead only while access moves forward
        self.front = ()              # windows to keep when a backward jump needs a window of its own

    def close(self):
        for window in self.windows:
            self._settle(window)          # never close the fd under a background read
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.windows = []

    def _read_into(self, buf, begin, length):
        view = memoryview(buf)
        step = max(16 * 1024 * 1024, _ceil(-(-length // 4)))
        def part(offset):
            done, want = 0, min(step, length - offset)
            while done < want:
                got = os.preadv(self.fd, [view[offset + done:offset + want]], begin + offset + done)
                if got <= 0:
                    break
                done += got
            return done
        return sum(_READERS.map(part, range(0, length, step)))

    def _settle(self, window):
        future, window["future"] = window["future"], None
        if future is not None:
            try:
                total = future.result()
            except OSError:
                total = 0
            window["end"] = window["start"] + total
            self.bytes_read += total

    def _plan(self, start, end):
        begin = start // _ALIGN * _ALIGN
        return begin, min(max(_WINDOW, _ceil(end - begin)), _ceil(self.file_size - begin))

    def _buffer(self, length, keep=()):
        buf = None
        if len(self.windows) >= self.max_windows:
            victim = next((w for w in reversed(self.windows) if w not in keep), None)
            if victim is not None:
                self._settle(victim)
                self.windows.remove(victim)
                buf = victim["buf"]
        if buf is None or len(buf) < length:
            buf = mmap.mmap(-1, max(length, _WINDOW))
        return buf

    def _fill(self, start, end, keep=()):
        begin, length = self._plan(start, end)
        buf = self._buffer(length, keep)
        window = {"buf": buf, "start": begin, "end": begin, "future": None}
        total = self._read_into(buf, begin, length)
        window["end"] = begin + total
        self.bytes_read += total
        self.windows.insert(0, window)
        if self.bytes_read > 2 * self.file_size + 2 * _WINDOW:    # random access pattern: stop, use mmap
            self.gave_up = True
        return window

    def _read_ahead(self, window):
        begin = window["end"]
        if (not self.ahead or begin >= self.file_size or begin % _ALIGN
                or any(w["start"] == begin for w in self.windows)):
            return
        begin, length = self._plan(begin, begin + 1)
        buf = self._buffer(length, keep=(window,))
        ahead = {"buf": buf, "start": begin, "end": begin + length, "future": None}   # planned until settled
        ahead["future"] = _AHEAD.submit(self._read_into, buf, begin, length)
        self.windows.insert(1 if self.windows and self.windows[0] is window else 0, ahead)

    def _covering(self, start, end):
        for window in self.windows:
            if window["start"] <= start and end <= window["end"]:
                self._settle(window)
                if end <= window["end"]:
                    return window
        return None

    def tensor(self, name):
        import torch
        dtype, shape, lo, hi = self.meta[name]
        start, end = self.base + lo, self.base + hi
        target = _torch_dtypes()[dtype]
        forward = start >= self.max_start
        self.max_start = max(self.max_start, start)
        window = self._covering(start, end)
        if window is None and forward:
            # straddling two consecutive windows (the second one read ahead): join the two parts, no re-read
            first = next((w for w in self.windows if w["start"] <= start < w["end"] and w["future"] is None), None)
            second = next((w for w in self.windows if first is not None and w["start"] == first["end"]), None)
            if second is not None:
                self._settle(second)
                if end <= second["end"]:
                    joined = torch.empty(end - start, dtype=torch.uint8)
                    cut = first["end"] - start
                    joined[:cut] = torch.frombuffer(first["buf"], dtype=torch.uint8, count=cut, offset=start - first["start"])
                    joined[cut:] = torch.frombuffer(second["buf"], dtype=torch.uint8, count=end - first["end"], offset=0)
                    self.windows.remove(second)
                    self.windows.insert(0, second)
                    self._read_ahead(second)
                    self.front = (second,) + tuple(w for w in self.windows if w["start"] == second["end"])
                    return joined.view(target).reshape(shape)
        if window is None:
            window = self._fill(start, end, keep=() if forward else self.front)
            if not (window["start"] <= start and end <= window["end"]):
                raise OSError(f"short O_DIRECT read for {name}")
        else:
            self.windows.remove(window)
            self.windows.insert(0, window)
        if forward:
            self._read_ahead(window)
            self.front = (window,) + tuple(w for w in self.windows if w["start"] == window["end"])
        raw = torch.frombuffer(window["buf"], dtype=torch.uint8, count=end - start, offset=start - window["start"])
        if (start - window["start"]) % target.itemsize:
            raw = raw.clone()
        return raw.view(target).reshape(shape)


class _FastHandle:
    def __init__(self, inner, device, path=None):
        self._inner, self._device = inner, device
        self._reader = None
        if DIRECT and path is not None:
            try:
                self._reader = _DirectReader(path)
            except (OSError, ValueError, KeyError):
                self._reader = None

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        return self._inner.__exit__(*exc)

    def get_tensor(self, name):
        reader = self._reader
        if reader is not None and not reader.gave_up and name in reader.meta:
            try:
                return reader.tensor(name).to(self._device)   # O_DIRECT window -> GPU (sync copy)
            except OSError:
                reader.gave_up = True
        tensor = self._inner.get_tensor(name)          # CPU view on the mmap
        return tensor.clone().to(self._device)          # anon RAM -> fast H2D

    def __getattr__(self, name):                        # keys, metadata, get_slice, offset_keys, ...
        return getattr(self._inner, name)


def _wrap(original):
    if getattr(original, "_spark_fast", False):
        return original

    def safe_open(filename, framework="pt", device="cpu", *args, **kwargs):
        target = str(device)
        if not ENABLED or framework not in ("pt", "torch") or target == "cpu":
            return original(filename, framework=framework, device=device, *args, **kwargs)
        return _FastHandle(original(filename, framework=framework, device="cpu", *args, **kwargs), target, filename)

    safe_open._spark_fast = True
    safe_open.__wrapped__ = original
    return safe_open


def load_file(filename, device="cpu"):
    """Drop-in for safetensors.torch.load_file: O_DIRECT chunked read, tensors cloned out of the read window
    (anonymous RAM, so a later .to("cuda") is a fast copy); mmap fallback."""
    from safetensors.torch import load_file as original
    if not (ENABLED and DIRECT):
        return original(filename, device=device)
    try:
        reader = _DirectReader(filename)
    except (OSError, ValueError, KeyError):
        return original(filename, device=device)
    try:
        out = {}
        for name in sorted(reader.meta, key=lambda n: reader.meta[n][2]):     # file order
            tensor = reader.tensor(name)
            out[name] = tensor.clone() if str(device) == "cpu" else tensor.to(device)
        return out
    except OSError:
        return original(filename, device=device)
    finally:
        reader.close()


def install(*modules):
    """Replace `safe_open` in safetensors and in modules that imported it by name."""
    if not ENABLED:
        return []
    import safetensors
    patched = []
    safetensors.safe_open = _wrap(safetensors.safe_open)
    patched.append("safetensors.safe_open")
    for module in modules:
        if hasattr(module, "safe_open"):
            module.safe_open = _wrap(module.safe_open)
            patched.append(f"{module.__name__}.safe_open")
    return patched
