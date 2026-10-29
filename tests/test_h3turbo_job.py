"""jobs/h3turbo_job.py (the experimental H3 Turbo's GPU job) on the CPU: job.json and its refusals, the geometry (the
frames, latents and samples of every length against the patch's runtime/geometry.py, and the layout rows against
those the probe it was ported from recorded on the GPU), the sigmas, the LoRA header rules on stand-in files with the
real files' key names and shapes (no data: sparse files), the conditioning checks, the studio's settings and
check_install, the command line, the environment and the progress lines against the Upscale job's
(upscale/regen2k_job.py), the memory trace, the generation call's arguments per task, --check and --check-install,
main's receipt when the GPU part finishes, fails, is refused, is aborted for memory or gets SIGTERM (a stand-in
run()), and --parent (the job dies with the server). No GPU, no weights.

Where torch is importable (CPU is enough; skipped otherwise): the LoRA hooks against PEFT's arithmetic on tiny
layers, and against PEFT itself on a tiny MiniMax-H3 transformer when peft and diffusers are there; the tensor checks
of the conditioning; and with diffusers 0.40.0 and ffmpeg, run() itself on a tiny random-weight MiniMax-H3 at the
real 576p canvases: t2va (no image), a first frame (with its condition rows computed independently: the stretched
keyframe, the VAE posterior under seed 42, its noise, the draw order), first + last, a last frame alone, the memory
abort and SIGTERM. Run those with the Regen-2K environment's Python, CUDA hidden:
    CUDA_VISIBLE_DEVICES= $LFSTUDIO_HOME/envs/regen2k/bin/python -I -m unittest discover -s tests -p test_h3turbo_job.py
"""
import contextlib
import functools
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "jobs" / "h3turbo_job.py"
REGEN = REPO / "upscale" / "regen2k_job.py"
PKG_REL = "models/minimax_h3/Sol-H3-Spark"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job_module = load(SCRIPT, "h3turbo_job")
regen = load(REGEN, "regen2k_job")
JobError = job_module.JobError

try:
    import torch
except ImportError:
    torch = None
try:
    import diffusers
    if diffusers.__version__ != job_module.DIFFUSERS_VERSION:
        diffusers = None
except ImportError:
    diffusers = None
FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

# the released MiniMax-H3 transformer's config.json, the numbers the LoRA shapes follow
H3_CONFIG = {"_class_name": "MiniMaxH3Transformer3DModel", "num_attention_heads": 56, "attention_head_dim": 128,
             "hidden_size": 5376, "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336, "in_channels": 24,
             "audio_in_channels": 32, "patch_size": [1, 2, 2], "text_dim": 5120}
GUIDE_FIRST = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
               "referenced.")
PROMPT = GUIDE_FIRST + "\n\nintegrated_multimodal_description: [Shot 1] Live-action, a test.\n\noverall_soundscape: " \
    "Wind."


def patch_new_file(rel, patch="lf-studio.patch"):
    """A file that patches/<patch> adds, as text."""
    text = (REPO / "patches" / patch).read_text()
    for section in re.split(r"(?m)^(?=diff --git )", text):
        if section.startswith(f"diff --git a/{rel} b/{rel}\n") and "\nnew file mode" in section:
            body = section.split("\n+++ ", 1)[1].split("\n", 1)[1]
            return "".join(line[1:] + "\n" for line in body.splitlines() if line.startswith("+"))
    raise AssertionError(f"{rel} is not a new file in the patch")


def fake_lora(path, config=None, rank=128, alpha="8", size=None, edit=None, metadata=None):
    """A safetensors file with a turbo LoRA's header (the real files' key names, BF16 shapes and data offsets for the
    transformer `config`) and no data: the file is extended, sparse, to the size its header gives. `size`: the header
    is padded with spaces to make the file that size (as the real files are). `edit(header)` changes the header."""
    header, offset = {}, 0
    for module, (a, b) in sorted(job_module.expected_lora(config or H3_CONFIG, rank).items()):
        for side, shape in (("A", a), ("B", b)):
            nbytes = shape[0] * shape[1] * 2
            header[f"{module}.lora_{side}.default.weight"] = {"dtype": "BF16", "shape": shape,
                                                              "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
    if edit:
        edit(header)
        offset = max([v["data_offsets"][1] for v in header.values()] + [0])
    meta = {"format": "pt", "key_format": "minimax-h3-diffusers", "floating_dtype": "bfloat16"}
    if alpha is not None:
        meta["alpha"] = alpha
    meta.update(metadata or {})
    raw = json.dumps({"__metadata__": meta, **header}, separators=(",", ":")).encode()     # as safetensors writes it
    pad = (size - 8 - offset - len(raw)) if size else (-len(raw)) % 8
    assert pad >= 0, f"the header is {-pad} bytes too long for {size}"
    raw += b" " * pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)
    return Path(path)


def make_base(root, config=None, vae_weights=True):
    """A base model directory as check_base reads it: configs, schedulers and placeholder weight files."""
    root = Path(root)
    for part in job_module.BASE_PARTS:
        (root / part).mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "config.json").write_text(json.dumps(config or H3_CONFIG))
    for part in ("vae", "audio_vae"):
        (root / part / "config.json").write_text("{}")
    for part in job_module.WEIGHT_PARTS:
        if part != "vae" or vae_weights:
            (root / part / "diffusion_pytorch_model.safetensors").write_bytes(b"w" * 100)
    for part in ("scheduler", "audio_scheduler"):
        (root / part / "scheduler_config.json").write_text('{"_class_name": "MiniMaxH3Scheduler"}')
    return root


def conditioning_meta(payload, task="fl2va", frames=None, canvas=(384, 672), text_rows=242, prompt=PROMPT,
                      prompt_tokens="minimax"):
    """The Qwen worker's receipt (conditioning.json) of an encode with `frames` ({anchor: path}) seen at `canvas`
    (width, height)."""
    frames = frames or {}
    tokens = (canvas[0] // 32) * (canvas[1] // 32)
    rows = text_rows + len(frames) * (tokens + 2)
    meta = {"status": "PASS", "task": task, "input_conditioned": task != "t2va",
            "input_spec": {"task": task, "first_frame": frames.get("first"), "last_frame": frames.get("last"),
                           "references": []},
            "prepared_media": [{"type": "image", "path": path, "anchor": anchor, "shape": [canvas[1], canvas[0], 3],
                                "has_audio": False} for anchor, path in frames.items()],
            "prompt": prompt, "case_id": "web_000000_abcdef", "seed": 42,
            "payload_sha256": hashlib.sha256(payload).hexdigest(), "shape": [1, rows, 5120],
            "dtype": "torch.bfloat16", "text_tokens": text_rows, "minimax_tokens": 0}
    if prompt_tokens is not None:
        meta["prompt_tokens"] = prompt_tokens
    return meta


class Fixture(unittest.TestCase):
    """A clip's directory as the studio will leave it before the job (qwen/conditioning.pt and .json for its first
    frame, at the draft canvas), the frames, a base model (configs only), a LoRA (header only), an output directory
    and a valid job.json for them: fl2va, 9:16, 576p, 5 s, 4 steps."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = make_base(self.tmp / "MiniMax-H3")
        (self.tmp / "loras").mkdir()
        self.lora = fake_lora(self.tmp / "loras" / "turbo-test.safetensors")     # 1.38 GB, sparse: a header only
        self.inputs = self.tmp / "runtime" / "inputs" / "webui" / ("f" * 32)
        self.inputs.mkdir(parents=True)
        self.first = self.inputs / "first.png"
        self.first.write_bytes(b"first frame")
        self.last = self.inputs / "last.png"
        self.last.write_bytes(b"last frame")
        self.case = self.tmp / "runtime" / "outputs" / "webui" / "20261028-010203-h3turbo-576p-fl2va" / "web_abcdef"
        self.out = self.case / "turbo"
        self.out.mkdir(parents=True)
        self.write_conditioning()

    def write_conditioning(self, payload=b"conditioning", **meta):
        """qwen/conditioning.pt (stand-in bytes) and its conditioning.json (conditioning_meta's, with `meta`)."""
        (self.case / "qwen").mkdir(exist_ok=True)
        (self.case / "qwen" / "conditioning.pt").write_bytes(payload)
        meta.setdefault("frames", {"first": str(self.first)})
        data = conditioning_meta(payload, **meta)
        (self.case / "qwen" / "conditioning.json").write_text(json.dumps(data))
        return data

    def job(self, **changes):
        data = {"version": 1, "task": "fl2va", "orientation": "portrait", "resolution": "576p", "seconds": 5,
                "seed": 1234, "steps": 4, "lora": str(self.lora), "shift": [6, 3],
                "conditioning": str(self.case / "qwen" / "conditioning.pt"), "qwen_canvas": "draft",
                "first_frame": str(self.first), "last_frame": None, "audio_noise": "seed", "crf": 19,
                "x264_preset": "veryfast", "load": "fast", "abort_below_gib": 6,
                "paths": {"base_model": str(self.base), "output": str(self.out)},
                "meta": {"studio_job": "f" * 32}}
        data.update(changes)
        return {k: v for k, v in data.items() if v is not DROP}

    def write_job(self, data=None, **changes):
        path = self.out / "job.json"
        path.write_text(json.dumps(self.job(**changes) if data is None else data))
        return path

    def parsed(self, **changes):
        return job_module.parse_job(self.job(**changes), "/jobs/x")

    def script(self, *args, check=False, env=None):
        environ = dict(os.environ, CUDA_VISIBLE_DEVICES="", **(env or {}))
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True,
                              timeout=300, check=check, env=environ)


DROP = object()


class JobTests(Fixture):
    def test_a_valid_job_and_its_defaults(self):
        job = self.parsed(audio_noise=DROP, crf=DROP, x264_preset=DROP, load=DROP, abort_below_gib=DROP,
                          version=DROP, meta=DROP, last_frame=DROP)
        self.assertEqual((job["task"], job["orientation"], job["resolution"], job["seconds"], job["steps"]),
                         ("fl2va", "portrait", "576p", 5, 4))
        self.assertEqual((job["audio_noise"], job["crf"], job["x264_preset"], job["load"], job["abort_below_gib"]),
                         ("seed", 19, "veryfast", "fast", 6.0))
        self.assertEqual((job["shift"], job["keyframes"], job["meta"], job["last_frame"]), ([6.0, 3.0], ["first"], {},
                                                                                         None))
        self.assertEqual((job["lora"], job["first_frame"], job["base_model"], job["output"]),
                         (self.lora, self.first, self.base, self.out))
        self.assertEqual(job["audio_noise_seed"], 1234 + 0x9E3779B9)
        self.assertEqual(json.loads(json.dumps(job_module.job_record(job)))["lora"], str(self.lora))
        data = self.job()
        del data["paths"]["output"]
        self.assertEqual(job_module.parse_job(data, "/jobs/x")["output"], Path("/jobs/x"))

    def test_tasks_and_their_frames(self):
        self.assertEqual(self.parsed(task="t2va", first_frame=None)["keyframes"], [])
        self.assertEqual(self.parsed(last_frame=str(self.last))["keyframes"], ["first", "last"])
        self.assertEqual(self.parsed(first_frame=None, last_frame=str(self.last))["keyframes"], ["last"])
        for changes, message in (({"task": "t2va"}, "t2va takes no first_frame"),
                                 ({"task": "t2va", "first_frame": None, "last_frame": str(self.last)},
                                  "t2va takes no"),
                                 ({"first_frame": None}, "fl2va needs a first_frame, a last_frame or both"),
                                 ({"task": "ref2va"}, "task: one of t2va, fl2va")):
            with self.assertRaisesRegex(JobError, message):
                self.parsed(**changes)

    def test_lengths(self):
        """Whole seconds from 5; 15 s is above diffusers' 15.0 s once snapped to 17n+5 frames (362)."""
        for seconds in range(5, 15):
            self.assertEqual(self.parsed(seconds=seconds)["seconds"], seconds)
        with self.assertRaisesRegex(JobError, r"seconds: 15 s makes 362 frames \(15\.08 s\), above the 15 s"):
            self.parsed(seconds=15)
        for bad in (4, 5.0, "5", True, None, 16):
            with self.assertRaisesRegex(JobError, "^seconds: "):
                self.parsed(seconds=bad)

    def test_bad_values_are_refused_by_name(self):
        for key, bad in (("orientation", "square"), ("resolution", "1080p"), ("resolution", 576),
                         ("qwen_canvas", "generation"), ("audio_noise", "silence"), ("x264_preset", "placebo"),
                         ("load", "mmap"), ("seed", -1), ("seed", 2 ** 63), ("seed", 1.5), ("steps", 0),
                         ("steps", 65), ("steps", True), ("shift", [6]), ("shift", [6, 0]), ("shift", "6/3"),
                         ("shift", [6, float("nan")]), ("crf", 52), ("crf", 19.5), ("abort_below_gib", -1),
                         ("abort_below_gib", "6"), ("lora", "loras/x.safetensors"), ("conditioning", ""),
                         ("first_frame", "first.png"), ("last_frame", 3), ("meta", [])):
            with self.subTest(key=key, value=bad):
                with self.assertRaisesRegex(JobError, f"^{key}: "):
                    self.parsed(**{key: bad})

    def test_unknown_and_missing_fields(self):
        with self.assertRaisesRegex(JobError, "unknown field.*: lora_alpha, plan"):
            self.parsed(plan="x", lora_alpha=8)
        with self.assertRaisesRegex(JobError, "missing field.*: seed, qwen_canvas"):
            self.parsed(seed=DROP, qwen_canvas=DROP)
        with self.assertRaisesRegex(JobError, "version: 1 is the only one"):
            self.parsed(version=2)
        with self.assertRaisesRegex(JobError, "paths: unknown field.*: package"):
            self.parsed(paths={"base_model": str(self.base), "package": "/x"})
        with self.assertRaisesRegex(JobError, "paths.base_model: an absolute path"):
            self.parsed(paths={"output": str(self.out)})
        with self.assertRaisesRegex(JobError, "must hold a JSON object"):
            job_module.parse_job([], "/")

    def test_the_audio_seed_wraps(self):
        self.assertEqual(self.parsed(seed=2 ** 63 - 1)["audio_noise_seed"], (2 ** 63 - 1 + 0x9E3779B9) % 2 ** 63)
        self.assertNotIn("audio_noise_seed", self.parsed(audio_noise="reference"))

    def test_load_job(self):
        path = self.write_job()
        self.assertEqual(job_module.load_job(path)["output"], self.out)
        path.write_text("{")
        with self.assertRaisesRegex(JobError, "is not JSON"):
            job_module.load_job(path)
        path.write_text(" " * (1 << 20) + "{}")
        with self.assertRaisesRegex(JobError, "larger than 1 MiB"):
            job_module.load_job(path)
        with self.assertRaisesRegex(JobError, "cannot read"):
            job_module.load_job(self.out / "missing.json")


class GeometryTests(unittest.TestCase):
    # the probe's GPU receipts, 5 s at 4 or 8 steps: (resolution, orientation, conditioning rows, keyframes) ->
    # its layout (condition, audio, video, sequence), latent grid (H/16, W/16) and its video rows (condition + video)
    PROBE = [(("576p", "portrait", 496, 1), (576, 414, 21312, 22798), (64, 36), 21888),
             (("576p", "landscape", 685, 1), (576, 414, 21312, 22987), (36, 64), 21888),
             (("768p", "portrait", 496, 1), (1008, 414, 37296, 39214), (84, 48), 38304),
             (("768p", "landscape", 685, 1), (1008, 414, 37296, 39403), (48, 84), 38304),
             (("576p", "landscape", 578, 2), (1152, 414, 21312, 23456), (36, 64), 22464),
             (("576p", "portrait", 578, 2), (1152, 414, 21312, 23456), (64, 36), 22464)]

    def test_the_canvases(self):
        self.assertEqual(job_module.CANVAS, {("576p", "landscape"): (1024, 576), ("576p", "portrait"): (576, 1024),
                                             ("768p", "landscape"): (1344, 768), ("768p", "portrait"): (768, 1344)})
        for (resolution, orientation), (width, height) in job_module.CANVAS.items():
            self.assertEqual((width % 32, height % 32), (0, 0))
            self.assertEqual(job_module.qwen_canvas(orientation, resolution, "canvas"), (width, height))
            self.assertEqual(job_module.image_tokens(width, height), {"576p": 576, "768p": 1008}[resolution])
            draft = job_module.qwen_canvas(orientation, resolution, "draft")
            self.assertEqual(job_module.image_tokens(*draft), 252)
            self.assertEqual(draft[0] > draft[1], orientation == "landscape")

    def test_the_probe_layouts(self):
        for (resolution, orientation, text, keyframes), (cond, audio, video, sequence), grid, rows in self.PROBE:
            width, height = job_module.CANVAS[(resolution, orientation)]
            layout = job_module.layout_rows(text, width, height, 124, keyframes)
            self.assertEqual((layout["condition"], layout["audio"], layout["video"], layout["sequence"]),
                             (cond, audio, video, sequence))
            self.assertEqual(layout["condition"] + layout["video"], rows)
            self.assertEqual((height // 16, width // 16), grid)
        self.assertEqual(job_module.layout_rows(300, 1024, 576, 124, 0)["sequence"], 300 + 414 + 21312)   # t2va

    def test_every_length_against_the_patch(self):
        """frames made and kept, samples kept: the patch's runtime/geometry.py (draft_frames_for_seconds,
        output_frames_for, frame_geometry), for every length the job takes."""
        namespace = {}
        exec(compile(patch_new_file(f"{PKG_REL}/runtime/geometry.py"), "geometry.py", "exec"), namespace)
        for seconds in range(5, 15):
            frames = job_module.frames_for_seconds(seconds)
            self.assertEqual(frames, namespace["draft_frames_for_seconds"](seconds))
            facts = namespace["frame_geometry"](frames)
            keep = job_module.studio_frames(frames)
            self.assertEqual((keep, job_module.kept_samples(keep)), (facts["output_frames"], facts["audio_samples"]))
            self.assertLessEqual(job_module.kept_samples(keep), job_module.audio_latents(frames) * 800)
        self.assertEqual([job_module.frames_for_seconds(s) for s in (5, 6, 8, 10, 12, 15)],
                         [124, 158, 192, 243, 294, 362])
        self.assertEqual([job_module.studio_frames(job_module.frames_for_seconds(s)) for s in (5, 6, 8, 10, 12)],
                         [121, 153, 185, 241, 289])
        self.assertEqual([job_module.latent_frames(job_module.frames_for_seconds(s)) for s in (5, 6, 8, 10, 12)],
                         [37, 47, 57, 72, 87])
        self.assertEqual((job_module.audio_latents(124), job_module.kept_samples(121), 207 * 800), (207, 161333,
                                                                                                    165600))

    def test_geometry_of_a_job(self):
        job = {"resolution": "768p", "orientation": "landscape", "seconds": 5, "steps": 4, "shift": [6.0, 3.0],
               "qwen_canvas": "canvas"}
        g = job_module.geometry(job)
        self.assertEqual((g["width"], g["height"], g["frames"], g["keep_frames"], g["keep_samples"],
                          g["latent_frames"], g["audio_latents"], g["num_inference_steps"], g["qwen_canvas"],
                          g["image_tokens"]), (1344, 768, 124, 121, 161333, 37, 207, 5, [1344, 768], 1008))
        self.assertEqual(job_module.output_name(job), "clip_1344x768_121f.mp4")
        self.assertEqual(job_module.output_name(dict(job, resolution="576p", orientation="portrait", seconds=10)),
                         "clip_576x1024_241f.mp4")


class SigmaTests(unittest.TestCase):
    def test_closed_form(self):
        self.assertEqual(job_module.expected_sigmas(4, 6.0), [1.0, 18 / 19, 6 / 7, 2 / 3, 0.0])
        self.assertEqual(job_module.expected_sigmas(4, 3.0), [1.0, 0.9, 0.75, 0.5, 0.0])

    def test_the_probe_runs(self):
        """The sigmas the scheduler used in the probe's GPU runs (float32), within 2e-6 (run()'s tolerance)."""
        for steps, shift, used in ((4, 6, [1.0, 0.9473684430122375, 0.8571428656578064, 0.6666666865348816, 0.0]),
                                   (4, 3, [1.0, 0.8999999761581421, 0.75, 0.5, 0.0]),
                                   (4, 12, [1.0, 0.972973, 0.923077, 0.8, 0.0]),
                                   (8, 6, [1.0, 0.976744, 0.947368, 0.909091, 0.857143, 0.782609, 0.666667,
                                           0.461538, 0.0])):
            want = job_module.expected_sigmas(steps, float(shift))
            self.assertEqual(len(want), steps + 1)
            self.assertTrue(all(abs(a - b) < 2e-6 for a, b in zip(want, used)), (steps, shift, want))


class LoraHeaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-lora-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_turbo_file(self):
        """312 target modules (6 per block of the 50 transformer and 2 refiner blocks), rank 128, alpha 8 from the
        metadata: scale 0.0625."""
        path = fake_lora(self.tmp / "a.safetensors")
        info = job_module.lora_info(path)
        self.assertEqual((len(info["modules"]), info["tensors"], info["rank"], info["alpha"], info["scale"]),
                         (312, 624, 128, 8.0, 0.0625))
        self.assertEqual(info["dtypes"], ["BF16"])
        self.assertEqual(info["shapes"], job_module.expected_lora(H3_CONFIG))
        self.assertEqual(info["shapes"]["transformer_blocks.0.ff.net.0.proj"], [[128, 5376], [28672, 128]])
        self.assertEqual(info["bytes"], path.stat().st_size)
        self.assertEqual(job_module.lora_info(fake_lora(self.tmp / "b.safetensors", alpha="128"))["scale"], 1.0)

    def test_the_real_files_size(self):
        """The real files' header (compact JSON, these keys and metadata) and data make the size of PRESETS,
        1,383,677,808 bytes."""
        for steps, preset in job_module.PRESETS.items():
            path = fake_lora(self.tmp / preset["file"])
            self.assertEqual(job_module.lora_info(path)["bytes"], preset["bytes"])
            self.assertEqual(job_module.preset_for(path), (steps, preset))
        self.assertEqual(job_module.preset_for(self.tmp / "other.safetensors"), (None, None))

    def test_a_transformer_prefix_is_taken(self):
        def prefix(header):
            for key in list(header):
                header["transformer." + key] = header.pop(key)
        self.assertEqual(job_module.lora_info(fake_lora(self.tmp / "p.safetensors", edit=prefix))["tensors"], 624)

    def test_refusals(self):
        def drop_b(header):
            del header["transformer_blocks.3.attn.to_k.lora_B.default.weight"]

        def other_rank(header):
            header["transformer_blocks.0.attn.to_q.lora_A.default.weight"]["shape"] = [64, 5376]

        def all_rank_64(header):
            for key in [k for k in header if k.startswith("transformer_blocks.1.")]:
                shape = header[key]["shape"]
                header[key]["shape"] = [64, shape[1]] if ".lora_A." in key else [shape[0], 64]

        def comfyui(header):
            header["diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight"] = {
                "dtype": "BF16", "shape": [3, 3], "data_offsets": [0, 18]}

        def not_a_target(header):
            for side in "AB":
                header[f"transformer_blocks.0.norm1.linear.lora_{side}.default.weight"] = {
                    "dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 2]}

        def vector(header):
            header["transformer_blocks.0.attn.to_q.lora_A.default.weight"]["shape"] = [128]
        for name, kwargs, message in (
                ("v0.1", {"alpha": None}, "no alpha in the file's metadata"),
                ("alpha0", {"alpha": "0"}, "no alpha"),
                ("unpaired", {"edit": drop_b}, "transformer_blocks.3.attn.to_k has no lora_B"),
                ("rank", {"edit": other_rank}, "has A rank 64 but B rank 128"),
                ("ranks", {"edit": all_rank_64}, r"expected one rank, found \[64, 128\]"),
                ("comfyui", {"edit": comfyui}, "unexpected tensor .* the \\*_comfyui_\\* files"),
                ("target", {"edit": not_a_target}, "not on a target module"),
                ("vector", {"edit": vector}, "is not a matrix")):
            with self.subTest(name):
                with self.assertRaisesRegex(JobError, message):
                    job_module.lora_info(fake_lora(self.tmp / f"{name}.safetensors", **kwargs))

    def test_a_cut_file_and_other_files(self):
        path = fake_lora(self.tmp / "cut.safetensors")
        with open(path, "r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
        with self.assertRaisesRegex(JobError, "a cut or damaged file"):
            job_module.lora_info(path)
        (self.tmp / "text.safetensors").write_bytes(b"\xff" * 64)
        with self.assertRaisesRegex(JobError, "not a safetensors file"):
            job_module.lora_info(self.tmp / "text.safetensors")
        (self.tmp / "json.safetensors").write_bytes(struct.pack("<Q", 4) + b"nope")
        with self.assertRaisesRegex(JobError, "its header is not JSON"):
            job_module.lora_info(self.tmp / "json.safetensors")
        (self.tmp / "empty.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
        with self.assertRaisesRegex(JobError, "no LoRA tensors"):
            job_module.lora_info(self.tmp / "empty.safetensors")
        with self.assertRaisesRegex(JobError, "cannot read"):
            job_module.lora_info(self.tmp / "missing.safetensors")


class CheckLoraTests(Fixture):
    def test_a_file_of_presets(self):
        preset = job_module.PRESETS[4]
        path = fake_lora(self.tmp / "loras" / preset["file"], size=preset["bytes"])
        record, notes = job_module.check_lora(self.parsed(lora=str(path)))
        self.assertEqual((record["preset"], record["module_count"], record["scale"]), (4, 312, 0.0625))
        self.assertNotIn("modules", record)
        self.assertEqual(notes, ["off the LoRA's documented 1344x768 training canvas: 576x1024"])
        record, notes = job_module.check_lora(self.parsed(lora=str(path), resolution="768p", orientation="landscape",
                                                          shift=[5, 3]))
        self.assertEqual(notes, [f"shift [5.0, 3.0], not the [6, 3] LightX2V gives for {preset['file']}"])
        with self.assertRaisesRegex(JobError, "is the 4-step LoRA, the job has 8 steps"):
            job_module.check_lora(self.parsed(lora=str(path), steps=8))

    def test_a_file_of_presets_with_another_size(self):
        preset = job_module.PRESETS[8]
        path = fake_lora(self.tmp / "loras" / preset["file"], size=preset["bytes"] + 16)
        with self.assertRaisesRegex(JobError, f"is {preset['bytes'] + 16} bytes, not {preset['bytes']}"):
            job_module.check_lora(self.parsed(lora=str(path), steps=8))

    def test_another_file(self):
        record, notes = job_module.check_lora(self.parsed())
        self.assertIsNone(record["preset"])
        self.assertEqual(notes, ["turbo-test.safetensors is not one of PRESETS (its sha256 is not checked)"])
        with self.assertRaisesRegex(JobError, "no LoRA file"):
            job_module.check_lora(self.parsed(lora=str(self.tmp / "missing.safetensors")))


class ConditioningMetaTests(Fixture):
    def check(self, meta=None, payload=b"conditioning", **changes):
        meta = meta or conditioning_meta(payload, frames={"first": str(self.first)})
        return job_module.check_conditioning_meta(meta, self.parsed(**changes), hashlib.sha256(payload).hexdigest())

    def test_a_first_frame_at_the_draft_canvas(self):
        record, notes = self.check()
        self.assertEqual((record["rows"], record["text_tokens"], record["qwen_canvas"],
                          record["image_tokens_per_keyframe"], record["prompt_tokens"]),
                         (496, 242, [384, 672], 252, "minimax"))
        self.assertEqual(notes, [])

    def test_canvas_tokens(self):
        """qwen_canvas "canvas": Qwen saw the keyframe at the clip's canvas, 576 tokens at 576p, 1008 at 768p."""
        for resolution, orientation, canvas in (("576p", "portrait", (576, 1024)), ("768p", "landscape",
                                                                                    (1344, 768))):
            meta = conditioning_meta(b"conditioning", frames={"first": str(self.first)}, canvas=canvas)
            record, _ = self.check(meta, qwen_canvas="canvas", resolution=resolution, orientation=orientation)
            self.assertEqual(record["image_tokens_per_keyframe"], (canvas[0] // 32) * (canvas[1] // 32))
            with self.assertRaisesRegex(JobError, "Qwen saw the first frame at .* not at the 384x672 of qwen_canvas "
                                                  "'draft'"):
                self.check(meta, resolution=resolution, orientation="portrait")

    def test_vision_rows(self):
        meta = conditioning_meta(b"conditioning", frames={"first": str(self.first)})
        meta["text_tokens"] += 1
        with self.assertRaisesRegex(JobError, "253 vision rows; 1 keyframe.* give 254"):
            self.check(meta)
        frames = {"first": str(self.first), "last": str(self.last)}
        record, _ = self.check(conditioning_meta(b"conditioning", frames=frames), last_frame=str(self.last))
        self.assertEqual(record["rows"], 242 + 2 * 254)
        t2va = conditioning_meta(b"conditioning", task="t2va", frames={})
        record, _ = self.check(t2va, task="t2va", first_frame=None)
        self.assertEqual((record["rows"], record["image_tokens_per_keyframe"]), (242, 0))
        del t2va["text_tokens"]                       # an older receipt: run() checks the tags instead
        self.check(t2va, task="t2va", first_frame=None)

    def test_refusals(self):
        first = {"first": str(self.first)}
        for meta_changes, job_changes, message in (
                ({"task": "t2va"}, {}, "the conditioning is for t2va, the job is fl2va"),
                ({"status": "FAIL"}, {}, "did not pass"),
                ({"payload_sha256": "0" * 64}, {}, "does not match its conditioning.json payload_sha256"),
                ({"shape": [1, 496, 4096]}, {}, r"shape is \[1, 496, 4096\]"),
                ({"prompt": None}, {}, "has no prompt"),
                ({}, {"last_frame": "/x/last.png"}, "made without a last frame, the job has one"),
                ({"input_spec": {"first_frame": None, "last_frame": str(self.last)}}, {},
                 "made without a first frame")):
            with self.subTest(message):
                meta = conditioning_meta(b"conditioning", frames=first)
                meta.update(meta_changes)
                with self.assertRaisesRegex(JobError, message):
                    self.check(meta, **job_changes)
        with self.assertRaisesRegex(JobError, "JSON object"):
            self.check(meta=["x"])

    def test_the_prompt_token_rule(self):
        """A conditioning of the "plain" prompt-token mode (or from before the setting) whose prompt has MiniMax's
        tokens is refused; without them its ids are minimax mode's; minimax mode takes them."""
        speech = PROMPT + " He says: <d>[English] Hello.</d>"
        for mode in ("plain", None):
            meta = conditioning_meta(b"conditioning", frames={"first": str(self.first)}, prompt=speech,
                                     prompt_tokens=mode)
            with self.assertRaisesRegex(JobError, r"prompt-token mode 'plain' .* \['<d>', '</d>'\]"):
                self.check(meta)
            meta["prompt"] = PROMPT
            self.assertEqual(self.check(meta)[0]["prompt_tokens"], "plain")
        self.check(conditioning_meta(b"conditioning", frames={"first": str(self.first)}, prompt=speech))

    def test_another_image_is_a_note(self):
        other = self.inputs / "other.png"
        other.write_bytes(b"another first frame")
        _, notes = self.check(conditioning_meta(b"conditioning", frames={"first": str(other)}))
        self.assertEqual(notes, ["the first frame differs from the image the conditioning was made with"])
        shutil.copy(self.first, other)
        self.assertEqual(self.check(conditioning_meta(b"conditioning", frames={"first": str(other)}))[1], [])


class InputTests(Fixture):
    def prepare(self, **changes):
        receipt = {}
        job = job_module.prepare(self.write_job(**changes), receipt)
        return job, receipt

    def test_what_a_job_reads(self):
        job, receipt = self.prepare()
        self.assertEqual(set(receipt["inputs"]), {"conditioning", "conditioning_json", "first_frame"})
        self.assertEqual(receipt["inputs_sha256"]["first_frame"], hashlib.sha256(b"first frame").hexdigest())
        self.assertEqual(receipt["layout"], {"text": 496, "condition": 576, "audio": 414, "video": 21312,
                                             "sequence": 22798, "per_frame": 576})
        self.assertEqual((receipt["geometry"]["keep_frames"], receipt["geometry"]["keep_samples"]), (121, 161333))
        self.assertEqual(receipt["job"]["first_frame"], str(self.first))
        self.assertEqual(receipt["lora"]["module_count"], 312)
        self.assertEqual(len(receipt["transformer_config_sha256"]), 64)
        self.assertIn("is not one of PRESETS", receipt["notes"][-1])

    def test_missing_files(self):
        for path, message in ((self.case / "qwen" / "conditioning.json", "no conditioning json file"),
                              (self.first, "no first frame file"),
                              (self.case / "qwen" / "conditioning.pt", "no conditioning file")):
            with self.subTest(message):
                data = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(JobError, message):
                    self.prepare()
                path.write_bytes(data)
        with self.assertRaisesRegex(JobError, "paths.output: no directory"):
            self.prepare(paths={"base_model": str(self.base), "output": str(self.tmp / "nowhere")})
        with mock.patch.object(job_module, "FASTLOAD", self.tmp / "fastload.py"):
            with self.assertRaisesRegex(JobError, "fast load's loader"):
                self.prepare()
            self.prepare(load="standard")

    def test_the_base_model(self):
        base = make_base(self.tmp / "t2va-only", vae_weights=False)
        with self.assertRaisesRegex(JobError, r"no weights in vae/ \(a t2va-only download .* "
                                              "install/20-weights.sh with fl2va in LFSTUDIO_TASKS"):
            job_module.check_base(base)
        (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
        self.assertEqual(job_module.check_base(base)["num_layers"], 50)
        (base / "audio_scheduler" / "scheduler_config.json").unlink()
        with self.assertRaisesRegex(JobError, "no audio_scheduler/scheduler_config.json"):
            job_module.check_base(base)
        shutil.rmtree(base / "audio_scheduler")
        with self.assertRaisesRegex(JobError, "has no audio_scheduler/"):
            job_module.check_base(base)
        (self.base / "transformer" / "config.json").write_text('{"_class_name": "WanTransformer3DModel"}')
        with self.assertRaisesRegex(JobError, "is not a MiniMax-H3 transformer's config"):
            job_module.check_base(self.base)
        with self.assertRaisesRegex(JobError, "no directory"):
            job_module.check_base(self.tmp / "nowhere")


class StudioSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings, problems = job_module.studio_settings({"LFSTUDIO_HOME": "/srv/lf"})
        self.assertEqual(problems, [])
        self.assertEqual(settings, {"python": Path("/srv/lf/envs/regen2k/bin/python"),
                                    "base_model": Path("/srv/lf/checkpoints/MiniMax-H3"),
                                    "loras": Path("/srv/lf/checkpoints/Minimax-h3-Turbo"), "qwen_canvas": "draft",
                                    "min_free_gib": 100.0, "guard_gib": 5.0})
        settings, _ = job_module.studio_settings({"LFSTUDIO_HOME": "/srv/lf", "LFSTUDIO_CHECKPOINTS": "/ck"})
        self.assertEqual((settings["base_model"], settings["loras"]), (Path("/ck/MiniMax-H3"),
                                                                       Path("/ck/Minimax-h3-Turbo")))

    def test_values(self):
        settings, problems = job_module.studio_settings({
            "LFSTUDIO_HOME": "/srv/lf", "LFSTUDIO_H3TURBO_PYTHON": "/env/bin/python", "LFSTUDIO_H3TURBO_BASE": "/b",
            "LFSTUDIO_H3TURBO_LORAS": "/l", "LFSTUDIO_H3TURBO_QWEN_CANVAS": "canvas",
            "LFSTUDIO_H3TURBO_MIN_FREE_GIB": "90", "LFSTUDIO_H3TURBO_GUARD_GIB": "0"})
        self.assertEqual(problems, [])
        self.assertEqual((settings["python"], settings["base_model"], settings["loras"], settings["qwen_canvas"],
                          settings["min_free_gib"], settings["guard_gib"]),
                         (Path("/env/bin/python"), Path("/b"), Path("/l"), "canvas", 90.0, 0.0))
        settings, problems = job_module.studio_settings({
            "LFSTUDIO_HOME": "/srv/lf", "LFSTUDIO_H3TURBO_BASE": "rel/base", "LFSTUDIO_H3TURBO_QWEN_CANVAS": "full",
            "LFSTUDIO_H3TURBO_MIN_FREE_GIB": "-1", "LFSTUDIO_H3TURBO_GUARD_GIB": "x"})
        self.assertEqual(len(problems), 4, problems)
        self.assertEqual((settings["min_free_gib"], settings["guard_gib"]), (100.0, 5.0))


class CheckInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-install-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = make_base(self.tmp / "MiniMax-H3")
        self.loras = self.tmp / "Minimax-h3-Turbo"
        self.loras.mkdir()
        self.settings = {"base_model": self.base, "loras": self.loras}

    def preset(self, steps, **kwargs):
        preset = job_module.PRESETS[steps]
        return fake_lora(self.loras / preset["file"], size=kwargs.pop("size", preset["bytes"]), **kwargs)

    def test_the_offered_steps_are_required(self):
        report, problems = job_module.check_install(self.settings)
        self.assertEqual(len(problems), 1, problems)
        self.assertRegex(problems[0], "minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors \\(the 4-step LoRA; "
                                      "install/66-h3turbo.sh fetches it\\)")
        self.preset(4)
        report, problems = job_module.check_install(self.settings)
        self.assertEqual(problems, [])
        self.assertEqual((report["loras"][4]["state"], report["loras"][4]["modules"], report["loras"][4]["alpha"]),
                         ("ok", 312, 8.0))
        self.assertEqual((report["loras"][8]["state"], report["loras"][8]["offered"]), ("missing", False))
        self.preset(8)
        self.assertEqual(job_module.check_install(self.settings)[0]["loras"][8]["state"], "ok")
        self.assertNotIn("env", report)

    def test_bad_files(self):
        self.preset(4, size=job_module.PRESETS[4]["bytes"] + 8)
        _, problems = job_module.check_install(self.settings)
        self.assertRegex(problems[0], "is 1383677816 bytes, not 1383677808")
        self.preset(4, alpha=None)
        self.assertRegex(job_module.check_install(self.settings)[1][0], "no alpha")
        self.preset(4)
        (self.base / "transformer" / "config.json").write_text(json.dumps(dict(H3_CONFIG, num_layers=48)))
        self.assertRegex(job_module.check_install(self.settings)[1][0], "do not fit the base model's transformer")
        (self.base / "vae" / "diffusion_pytorch_model.safetensors").unlink()
        self.assertRegex(job_module.check_install(self.settings)[1][0], "no weights in vae/")
        with mock.patch.object(job_module, "FASTLOAD", self.tmp / "fastload.py"):
            self.assertIn("(the fast load's loader)", " ".join(job_module.check_install(self.settings)[1]))

    @unittest.skipUnless(torch is not None and diffusers is not None, "needs torch and diffusers 0.40.0")
    def test_the_environment(self):
        self.preset(4)
        report, problems = job_module.check_install(self.settings, env=True)
        self.assertEqual(problems, [])
        self.assertEqual(report["env"]["diffusers"], job_module.DIFFUSERS_VERSION)

    def test_another_diffusers(self):
        from importlib import metadata
        problems = []
        with mock.patch.object(metadata, "version", lambda name: "0.39.0" if name == "diffusers" else "1.0"):
            found = job_module.check_env(problems)
        self.assertEqual(found["diffusers"], "0.39.0")
        self.assertRegex(problems[0], "diffusers 0.39.0 in .*, this job is written for 0.40.0")


class LineAndCommandTests(unittest.TestCase):
    """The helpers copied from upscale/regen2k_job.py behave as regen's (a jobs/common.py comes with a third job)."""

    def test_progress_lines(self):
        for name in ("loading", "step 1/4", "step 4/4", "decoding", "saving"):
            self.assertEqual(job_module.parse_line(job_module.phase_line(name) + "\n"), ("phase", name))
            self.assertEqual(job_module.phase_line(name), regen.phase_line(name))
        for text in ("RuntimeError: out of\nmemory\n  at step 2", "x" * 5000):
            self.assertEqual(job_module.error_line(text), regen.error_line(text))
        for line in ("[lf-studio] 01:02:03 phase loading", "[lf-studio] phase ", "phase loading", "",
                     "[lf-studio] phaseloading", "[lf-studio] error out of memory", "[lf-studio] phase step 2/4"):
            self.assertEqual(job_module.parse_line(line), regen.parse_line(line), line)

    def test_worker_command(self):
        self.assertEqual(job_module.worker_command(Path("/env/bin/python"), SCRIPT, Path("/c/turbo/job.json"), 4321),
                         ["/env/bin/python", "-I", "-B", str(SCRIPT), "--job", "/c/turbo/job.json", "--parent",
                          "4321"])
        for parent in (None, 4321):
            self.assertEqual(job_module.worker_command("/p", SCRIPT, "/j", parent),
                             regen.worker_command("/p", SCRIPT, "/j", parent))

    def test_worker_environment(self):
        self.assertEqual((job_module.OFFLINE, job_module.REMOVED, job_module.TUNING),
                         (regen.OFFLINE, regen.REMOVED, regen.TUNING))
        for environ in ({"HF_TOKEN": "hf_x", "HUGGING_FACE_HUB_TOKEN": "hf_y", "PATH": "/bin", "HF_HUB_OFFLINE": "0",
                         "H3_ATTN_BACKEND": "_flash_3_hub", "PYTHONPATH": "/x",
                         "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"}, {}, {"IMAGEIO_FFMPEG_EXE": "/opt/ff"}):
            env = job_module.worker_environment(environ)
            self.assertEqual(env, regen.worker_environment(environ))
            self.assertNotIn("HF_TOKEN", env)
            self.assertEqual((env["HF_HUB_OFFLINE"], env["HF_TOKEN_PATH"]), ("1", "/nonexistent"))
        environ = {"HF_TOKEN": "hf_x", "KEEP": "1"}
        job_module.apply_environment(environ)
        self.assertEqual((environ.get("HF_TOKEN"), environ["KEEP"], environ["HF_HUB_OFFLINE"]), (None, "1", "1"))

    def test_write_json_and_meminfo(self):
        tmp = Path(tempfile.mkdtemp(prefix="h3turbo-json-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        job_module.write_json(tmp / "a.json", {"state": "done", "n": [1, 2]})
        regen.write_json(tmp / "b.json", {"state": "done", "n": [1, 2]})
        self.assertEqual((tmp / "a.json").read_bytes(), (tmp / "b.json").read_bytes())
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), ["a.json", "b.json"])
        (tmp / "meminfo").write_text("MemTotal: 127535648 kB\nMemFree: 1048576 kB\nMemAvailable: 41943040 kB\n"
                                     "Cached: 2097152 kB\n")
        self.assertEqual(job_module.meminfo(tmp / "meminfo"), regen.meminfo(tmp / "meminfo"))
        self.assertEqual(job_module.meminfo(tmp / "meminfo"), {"MemFree": 1.0, "MemAvailable": 40.0, "Cached": 2.0})


class MemTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-mem-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / "meminfo"
        self.set(40)

    def set(self, gib):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(f"MemTotal:       127535648 kB\nMemAvailable:   {int(gib * 1048576)} kB\n")
        os.replace(tmp, self.path)

    def test_lowest_per_label_and_the_abort(self):
        trace = job_module.MemTrace(6, self.path, interval=0.01)
        trace.start()
        self.addCleanup(trace.stop)
        trace.mark("load")
        trace.check()
        self.set(15)
        deadline = time.time() + 5
        while trace.low.get("load", 99) > 15 and time.time() < deadline:
            time.sleep(0.02)
        trace.mark("step_2")
        self.set(5.5)
        deadline = time.time() + 5
        while not trace.abort.is_set() and time.time() < deadline:
            time.sleep(0.02)
        with self.assertRaisesRegex(job_module.RunAborted, "MemAvailable 5.50 GiB < 6 GiB during step_2"):
            trace.check()
        self.set(30)
        trace.stop()
        self.assertEqual((trace.low["load"], trace.low["step_2"]), (15.0, 5.5))
        self.assertTrue(trace.abort.is_set())            # it stays set: the clip is aborted


class StandInGenerator:
    def __init__(self):
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class StandInRows:
    def __init__(self, shape, seed):
        self.shape, self.seed, self.ops = shape, seed, []

    def reshape(self, *shape):
        self.ops.append(("reshape", shape))
        return self

    def permute(self, *dims):
        self.ops.append(("permute", dims))
        return self


class GenerateKwargsTests(Fixture):
    """The generation call's arguments, as the probe made them, with a stand-in torch: no image for t2va, `image`
    and/or `last_image` for fl2va, steps + 1 grid points, the clip's own CPU generator, and the audio noise."""

    def kwargs(self, images, **changes):
        fake = types.SimpleNamespace(Generator=StandInGenerator, float32="float32",
                                     randn=lambda shape, generator, dtype: StandInRows(shape, generator.seed))
        return job_module.generate_kwargs(self.parsed(**changes), images, "embeds", "tags", fake)

    def test_fl2va_first_frame(self):
        kwargs = self.kwargs(["FIRST"])
        self.assertEqual(sorted(kwargs), ["audio_latents", "generator", "height", "image", "num_frames",
                                          "num_inference_steps", "prompt_embeds", "text_token_tags", "width"])
        self.assertEqual((kwargs["image"], kwargs["height"], kwargs["width"], kwargs["num_frames"],
                          kwargs["num_inference_steps"], kwargs["prompt_embeds"], kwargs["text_token_tags"]),
                         ("FIRST", 1024, 576, 124, 5, "embeds", "tags"))
        self.assertEqual(kwargs["generator"].seed, 1234)
        rows = kwargs["audio_latents"]
        self.assertEqual((rows.shape, rows.seed, rows.ops), ((414, 32), 1234 + 0x9E3779B9,
                                                             [("reshape", (2, 207, 32)), ("permute", (0, 2, 1))]))

    def test_the_other_keyframes_and_t2va(self):
        both = self.kwargs(["FIRST", "LAST"], last_frame=str(self.last))
        self.assertEqual((both["image"], both["last_image"]), ("FIRST", "LAST"))
        last = self.kwargs(["LAST"], first_frame=None, last_frame=str(self.last))
        self.assertEqual((last.get("image"), last["last_image"]), (None, "LAST"))
        t2va = self.kwargs([], task="t2va", first_frame=None, resolution="768p", orientation="landscape",
                           seconds=10, steps=8, audio_noise="reference")
        self.assertNotIn("image", t2va)
        self.assertNotIn("last_image", t2va)
        self.assertNotIn("audio_latents", t2va)                # drawn by diffusers from the clip's generator
        self.assertEqual((t2va["width"], t2va["height"], t2va["num_frames"], t2va["num_inference_steps"]),
                         (1344, 768, 243, 9))


class ScriptTests(Fixture):
    def test_importing_it_needs_no_torch(self):
        probe = (f"import importlib.util, sys; spec = importlib.util.spec_from_file_location('j', {str(SCRIPT)!r}); "
                 "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); f = m.load_fastload(); "
                 "print(f.__file__, sorted({k.split('.')[0] for k in sys.modules} & {'torch', 'numpy', 'diffusers', "
                 "'PIL', 'safetensors', 'fastload'}))")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), f"{REPO / 'upscale' / 'fastload.py'} []")   # the Upscale's, by path
        help_text = self.script("--help", check=True).stdout
        self.assertIn("--check-install", help_text)
        result = self.script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("--job is required", result.stderr)

    def test_check_prints_the_plan_and_writes_nothing(self):
        result = self.script("--job", str(self.write_job()), "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        planned = json.loads(result.stdout)
        self.assertEqual((planned["ok"], planned["output"]), (True, "clip_576x1024_121f.mp4"))
        self.assertEqual(planned["layout"]["sequence"], 22798)
        self.assertEqual((planned["geometry"]["frames"], planned["geometry"]["keep_frames"],
                          planned["geometry"]["keep_samples"]), (124, 121, 161333))
        self.assertEqual(planned["geometry"]["sigmas"], job_module.expected_sigmas(4, 6.0))
        self.assertEqual(planned["conditioning"]["image_tokens_per_keyframe"], 252)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])

    def test_check_hashes_a_file_of_presets(self):
        """--check compares the sha256 of a file of PRESETS (here a small stand-in) with the pinned one."""
        small = {"hidden_size": 64, "num_attention_heads": 2, "attention_head_dim": 32, "ffn_dim": 128,
                 "num_layers": 2, "num_refiner_layers": 1}
        path = fake_lora(self.tmp / "loras" / "small-preset.safetensors", config=small, rank=4)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stdout = io.StringIO()
        for pinned, status in ((digest, 0), ("0" * 64, 2)):
            presets = {4: {"file": path.name, "bytes": path.stat().st_size, "sha256": pinned, "shift": [6, 3]}}
            with mock.patch.dict(job_module.PRESETS, presets, clear=True), contextlib.redirect_stdout(stdout):
                self.assertEqual(job_module.main(["--job", str(self.write_job(lora=str(path))), "--check"]), status)
        self.assertEqual(json.loads(stdout.getvalue().split("\n[lf-studio]")[0])["lora"]["sha256"], digest)
        self.assertRegex(stdout.getvalue().splitlines()[-1], f"^\\[lf-studio\\] error small-preset.safetensors: "
                                                             f"sha256 {digest}, not 0{{64}}")

    def test_check_refuses(self):
        result = self.script("--job", str(self.write_job(qwen_canvas="canvas")), "--check")
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout.splitlines()[-1], r"^\[lf-studio\] error Qwen saw the first frame at "
                                                         r"\[672, 384, 3\] \(height, width\), not at the 576x1024")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])

    def test_a_refused_job_leaves_a_refused_receipt(self):
        result = self.script("--job", str(self.write_job(seconds=15)))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1])[0], "error")
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("refused", 2))
        self.assertIn("362 frames", receipt["error"])
        (self.case / "qwen" / "conditioning.json").unlink()
        self.assertEqual(self.script("--job", str(self.write_job())).returncode, 2)
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertIn("no conditioning json file", receipt["error"])
        self.assertEqual(receipt["job"]["meta"], {"studio_job": "f" * 32})

    def test_a_run_without_torch_or_a_gpu_fails_with_a_receipt(self):
        """Without torch, or with torch and no GPU (CUDA hidden here), the run fails before any model loads."""
        result = self.script("--job", str(self.write_job()))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("failed", 1))
        self.assertRegex(receipt["error"], "^(ModuleNotFoundError|RuntimeError|AssertionError): ")
        self.assertEqual(receipt["inputs_sha256"]["conditioning"], hashlib.sha256(b"conditioning").hexdigest())
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1]), ("error", receipt["error"][:500]))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json", "receipt.json"])

    def test_check_install(self):
        env = {"LFSTUDIO_H3TURBO_BASE": str(self.base), "LFSTUDIO_H3TURBO_LORAS": str(self.tmp / "loras")}
        result = self.script("--check-install", env=env)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        report = json.loads(result.stdout.split("\n[lf-studio] error")[0])
        self.assertFalse(report["ok"])
        self.assertEqual(report["base_model"], str(self.base))
        self.assertIn("the 4-step LoRA", report["problems"][0])
        self.assertIn("env", report)
        self.assertEqual(self.script("--check-install", "--job", "x").returncode, 2)


class MainTests(Fixture):
    """main() around a stand-in run(): the receipt for done, failed, refused, aborted and SIGTERM."""

    def main(self, run, **changes):
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        stdout = io.StringIO()
        with mock.patch.object(job_module, "run", run), mock.patch.object(job_module, "apply_environment") as env, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = job_module.main(["--job", str(self.write_job(**changes))])
        env.assert_called_once_with(os.environ)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        receipt = json.loads((self.out / "receipt.json").read_text())
        last = job_module.parse_line(stdout.getvalue().splitlines()[-1])
        self.assertEqual(last, ("error", receipt["error"][:500]) if status else None)
        self.assertEqual(receipt["exit_status"], status)
        return status, receipt

    def test_done(self):
        def run(job, receipt, trace):
            self.assertEqual(json.loads((self.out / "receipt.json").read_text())["state"], "running")
            self.assertEqual(trace.floor, 4.5)
            receipt["peak_gib"].update(load=72.0, step_1=77.1, decode=75.9)
            receipt["lowest_mem_available_gib"].update(step_1=35.2)
            receipt.update(stage="mux", phase="saving", output={"file": "clip_576x1024_121f.mp4"})
            trace.mark("step_1")
        status, receipt = self.main(run, abort_below_gib=4.5)
        self.assertEqual((status, receipt["state"]), (0, "done"))
        self.assertEqual(receipt["peak_allocated_gib"], 77.1)
        self.assertLessEqual(receipt["lowest_mem_available_gib"]["all"], 35.2)
        self.assertNotIn("stage", receipt)
        self.assertNotIn("phase", receipt)
        self.assertEqual(receipt["job"]["meta"]["studio_job"], "f" * 32)

    def test_failed(self):
        def run(job, receipt, trace):
            receipt.update(stage="decode_video", phase="decoding")
            raise RuntimeError("CUDA out of memory.\nTried to allocate 35.00 GiB")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"]), (1, "failed"))
        self.assertEqual(receipt["error"], "RuntimeError: CUDA out of memory. Tried to allocate 35.00 GiB")
        self.assertEqual((receipt["stage"], receipt["phase"]), ("decode_video", "decoding"))
        self.assertTrue(any(line.startswith("RuntimeError: CUDA out of memory.") for line in receipt["traceback"]))

    def test_refused_inside_the_run(self):
        def run(job, receipt, trace):
            raise JobError("a vision block of the conditioning has 576 image tokens, qwen_canvas 'draft' gives 252")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"]), (2, "refused"))

    def test_aborted_for_memory(self):
        def run(job, receipt, trace):
            receipt.update(stage="step_2", phase="step 2/4")
            raise job_module.RunAborted("MemAvailable 5.10 GiB < 6 GiB during step_2")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"], receipt["phase"]), (3, "aborted", "step 2/4"))
        self.assertEqual(receipt["error"], "MemAvailable 5.10 GiB < 6 GiB during step_2")

    def test_sigterm_is_a_cancel(self):
        def run(job, receipt, trace):
            receipt.update(stage="step_1", phase="step 1/4")
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(10)
            self.fail("SIGTERM did not stop the run")
        started = time.time()
        status, receipt = self.main(run)
        self.assertLess(time.time() - started, 5)
        self.assertEqual((status, receipt["state"], receipt["error"]), (143, "cancelled", "stopped by SIGTERM"))
        self.assertEqual(receipt["phase"], "step 1/4")


def gone(pid):
    """Whether a process has ended (a zombie that its new parent has not reaped yet counts as ended)."""
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


@unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux's")
class ParentTests(Fixture):
    """--parent: the job is killed when the studio's server ends, and does not start when it is gone already. In
    child processes only: tie_to_parent in this one would tie the test runner to its own parent."""

    def test_killed_with_its_parent(self):
        child = (f"import importlib.util, os, sys, time; spec = importlib.util.spec_from_file_location('j', "
                 f"{str(SCRIPT)!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                 "print(m.tie_to_parent(int(sys.argv[1])), flush=True); time.sleep(60)")
        middle = ("import subprocess, sys; p = subprocess.Popen([sys.executable, '-I', '-c', sys.argv[1], "
                  "str(__import__('os').getpid())], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True); "
                  "print(p.pid, p.stdout.readline().strip(), flush=True)")
        result = subprocess.run([sys.executable, "-I", "-c", middle, child], capture_output=True, text=True,
                                timeout=60, check=True)                      # the middle process ends here
        pid, tied = result.stdout.split()
        self.assertEqual(tied, "True")
        deadline = time.time() + 10
        while not gone(int(pid)) and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(gone(int(pid)), "the job outlived its parent")

    def test_a_parent_gone_already(self):
        result = self.script("--job", str(self.write_job()), "--parent", str(os.getpid() + 1))
        self.assertEqual(result.returncode, 143, result.stdout + result.stderr)
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1]),
                         ("error", f"the studio's server (pid {os.getpid() + 1}) is gone; not starting"))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])        # nothing written


# ---------------------------------------------------------------- with torch (CPU)

@unittest.skipUnless(torch, "needs torch (CPU is enough)")
class LoraArithmeticTests(unittest.TestCase):
    """The hooks against PEFT's LoRA Linear arithmetic (result + lora_B(lora_A(x)) * scaling, the adapter in the
    layer's bf16), bit for bit, on tiny layers named as the transformer's."""

    def model(self):
        torch.manual_seed(0)
        model = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(16, 24)
        block.ff = torch.nn.Module()
        block.ff.net = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Identity(), torch.nn.Linear(24, 16)])
        block.ff.net[0].proj = torch.nn.Linear(16, 48)
        model.transformer_blocks = torch.nn.ModuleList([block])
        model.norm = torch.nn.Linear(16, 16)          # not a target
        return model.to(torch.bfloat16)

    def pairs(self, model, rank=4, seed=1):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for name in job_module.lora_targets(model, torch):
            module = model.get_submodule(name)
            out[name] = ((torch.randn(rank, module.in_features, generator=g) * 0.5).to(torch.bfloat16),
                         (torch.randn(module.out_features, rank, generator=g) * 0.5).to(torch.bfloat16))
        return out

    def test_peft_arithmetic_bit_for_bit(self):
        model = self.model()
        self.assertEqual(job_module.lora_targets(model, torch), ["transformer_blocks.0.attn.to_q",
                                                                 "transformer_blocks.0.ff.net.0.proj",
                                                                 "transformer_blocks.0.ff.net.2"])
        pairs = self.pairs(model)
        x = (torch.randn(5, 7, 16) * 2).to(torch.bfloat16)
        F = torch.nn.functional
        with torch.no_grad():
            for name, (a, b) in pairs.items():
                module = model.get_submodule(name)
                inputs = x if module.in_features == 16 else (torch.randn(5, 7, 24) * 2).to(torch.bfloat16)
                base = module(inputs)
                want = base + F.linear(F.linear(inputs.to(a.dtype), a), b) * 0.0625      # PEFT's expression
                adapter = job_module.LoraAdapter(model, pairs, 0.0625, torch)
                got = module(inputs)
                adapter.remove()
                self.assertTrue(torch.equal(got, want), name)
                self.assertTrue(torch.equal(module(inputs), base))                       # removed exactly

    def test_the_input_is_not_touched_and_files_that_do_not_fit(self):
        model = self.model()
        pairs = self.pairs(model)
        adapter = job_module.LoraAdapter(model, pairs, 1.0, torch)
        module = model.get_submodule("transformer_blocks.0.ff.net.0.proj")
        x = (torch.randn(3, 16) * 2).to(torch.bfloat16)
        keep = x.clone()
        with torch.no_grad():
            first, second = module(x), module(x)
        adapter.remove()
        self.assertTrue(torch.equal(x, keep) and torch.equal(first, second))
        before = module(x)
        broken = dict(pairs)
        broken.pop("transformer_blocks.0.attn.to_q")
        with self.assertRaisesRegex(RuntimeError, "do not match the transformer's targets"):
            job_module.LoraAdapter(model, broken, 1.0, torch)
        a, b = pairs["transformer_blocks.0.ff.net.2"]
        broken = dict(pairs, **{"transformer_blocks.0.ff.net.2": (a[:, :-1], b)})
        with self.assertRaisesRegex(RuntimeError, "do not fit"):
            job_module.LoraAdapter(model, broken, 1.0, torch)
        self.assertTrue(torch.equal(module(x), before))                                  # nothing left attached

    def test_the_file_loads_as_pairs(self):
        from safetensors.torch import save_file
        tmp = Path(tempfile.mkdtemp(prefix="h3turbo-pairs-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        model = self.model()
        pairs = self.pairs(model)
        tensors = {}
        for name, (a, b) in pairs.items():
            tensors[f"transformer.{name}.lora_A.default.weight"] = a
            tensors[f"{name}.lora_B.default.weight"] = b
        save_file(tensors, str(tmp / "l.safetensors"), metadata={"alpha": "8"})
        loaded = job_module.load_lora_pairs(tmp / "l.safetensors", torch.device("cpu"), torch)
        fast = job_module.load_lora_pairs(tmp / "l.safetensors", torch.device("cpu"), torch, job_module.load_fastload())
        self.assertEqual(sorted(loaded), sorted(pairs))
        for name, (a, b) in pairs.items():
            self.assertTrue(torch.equal(loaded[name][0], a) and torch.equal(loaded[name][1], b))
            self.assertTrue(torch.equal(fast[name][0], a) and torch.equal(fast[name][1], b))


@unittest.skipUnless(torch, "needs torch (CPU is enough)")
class ConditioningTensorTests(Fixture):
    def cond(self, tags, task="fl2va", width=5120, dtype=None):
        rows = len(tags)
        return {"task": task, "prompt_embeds": torch.zeros(1, rows, width, dtype=dtype or torch.bfloat16),
                "text_token_tags": torch.tensor(tags, dtype=torch.int64)}

    def test_blocks_and_tokens(self):
        one = [1] * 6 + [0] * 254 + [1] * 30
        summary = job_module.check_conditioning(self.cond(one), self.parsed(), torch)
        self.assertEqual(summary, {"rows": 290, "text_rows": 36, "vision_rows": 254,
                                   "vision_blocks": [{"start": 6, "rows": 254, "image_tokens": 252}]})
        two = [1] * 6 + [0] * 254 + [1] * 6 + [0] * 254 + [1] * 30
        summary = job_module.check_conditioning(self.cond(two), self.parsed(last_frame=str(self.last)), torch)
        self.assertEqual([b["start"] for b in summary["vision_blocks"]], [6, 266])
        canvas = [1] * 6 + [0] * 578 + [1] * 30
        job_module.check_conditioning(self.cond(canvas), self.parsed(qwen_canvas="canvas"), torch)
        t2va = self.cond([1] * 40, task="t2va")
        self.assertEqual(job_module.check_conditioning(t2va, self.parsed(task="t2va", first_frame=None), torch)
                         ["vision_rows"], 0)
        batched = self.cond(one)
        batched["text_token_tags"] = batched["text_token_tags"][None]
        self.assertEqual(job_module.check_conditioning(batched, self.parsed(), torch)["rows"], 290)

    def test_refusals(self):
        one = [1] * 6 + [0] * 254 + [1] * 30
        bad_nan = self.cond(one)
        bad_nan["prompt_embeds"][0, 3, 7] = float("nan")
        for cond, changes, message in (
                (self.cond(one), {"qwen_canvas": "canvas"}, "has 252 image tokens, qwen_canvas 'canvas' gives 576"),
                (self.cond(one), {"last_frame": "/x/last.png"}, "1 vision blocks for 2 keyframe"),
                (self.cond(one), {"task": "t2va", "first_frame": None}, "is for fl2va, the job is t2va"),
                (self.cond([1] * 6 + [2] * 254), {}, r"text_token_tags hold \[1, 2\]"),
                (self.cond(one, width=4096), {}, "conditioning shapes"),
                (self.cond(one, dtype=torch.float32), {}, "conditioning shapes"),
                (bad_nan, {}, "not finite"),
                ({"prompt_embeds": None}, {}, "no prompt_embeds and text_token_tags")):
            with self.subTest(message):
                with self.assertRaisesRegex(JobError, message):
                    job_module.check_conditioning(cond, self.parsed(**changes), torch)


TINY_TRANSFORMER = dict(num_attention_heads=2, attention_head_dim=32, hidden_size=64, num_layers=2,
                        num_refiner_layers=1, ffn_dim=128, text_dim=5120, freq_dim=32, time_embed_hidden_dim=64,
                        time_embed_dim=32, rope_freq_dim=4)
TINY_VAE = dict(block_out_channels=(8, 8, 8, 8), spatial_downsample_factors=(2, 2, 2, 2),
                temporal_downsample_factors=(1, 2, 2, 1), norm_num_groups=4, decoder_num_layers=1,
                decoder_num_attention_heads=1, decoder_attention_head_dim=24, decoder_num_register_tokens=1,
                decoder_ffn_mult=1, latents_mean=tuple(0.05 * i for i in range(24)),
                latents_std=tuple(1.0 + 0.02 * i for i in range(24)))
TINY_AUDIO_VAE = dict(encoder_dim=2, latent_dim=32, latent_channels=32, num_attention_heads=1, decoder_dim=256,
                      resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1,),), latents_mean=[0.1] * 32,
                      latents_std=[1.5] * 32)
LABEL_ROWS, PROMPT_ROWS = 6, 24      # "<Picture N>: " and the prompt's rows around a keyframe's vision block


def make_tiny_base(root):
    """A tiny random-weight MiniMax-H3 base model directory in the diffusers layout (the real classes and
    schedulers, small configs)."""
    from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler,
                           MiniMaxH3Transformer3DModel)
    root = Path(root)
    torch.manual_seed(1234)
    MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER).save_pretrained(root / "transformer")
    AutoencoderKLMiniMaxH3(**TINY_VAE).save_pretrained(root / "vae")
    AutoencoderKLMiniMaxH3Audio(**TINY_AUDIO_VAE).save_pretrained(root / "audio_vae")
    MiniMaxH3Scheduler(shift=12.0).save_pretrained(root / "scheduler")
    MiniMaxH3Scheduler(shift=3.0).save_pretrained(root / "audio_scheduler")
    return root


def make_image(path, size, seed=0):
    """A structured RGB PNG (gradients and blocks) of size (width, height)."""
    import numpy as np
    from PIL import Image
    width, height = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack([(x * 255 // max(1, width - 1)), (y * 255 // max(1, height - 1)),
                      ((x // 64 + y // 64) % 2) * 200 + 20], axis=-1).astype(np.uint8)
    for _ in range(6):
        x0, y0 = rng.integers(0, width - 100), rng.integers(0, height - 100)
        image[y0:y0 + 90, x0:x0 + 90] = rng.integers(0, 255, 3)
    Image.fromarray(image).save(path)
    return Path(path)


def make_tiny_lora(path, model, rank=4, alpha="8", seed=9, std=0.3):
    """A LoRA file in the turbo files' format for the tiny transformer."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name in job_module.lora_targets(model, torch):
        module = model.get_submodule(name)
        tensors[f"{name}.lora_A.default.weight"] = (torch.randn(rank, module.in_features, generator=g) * std).to(
            torch.bfloat16)
        tensors[f"{name}.lora_B.default.weight"] = (torch.randn(module.out_features, rank, generator=g) * std).to(
            torch.bfloat16)
    save_file(tensors, str(path), metadata={"format": "pt", "key_format": "minimax-h3-diffusers", "alpha": alpha,
                                            "floating_dtype": "bfloat16"})
    return Path(path)


def packed_inputs(seed=0):
    """A valid packed fl2va input for the tiny transformer (layout from diffusers' own builder)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    g = torch.Generator().manual_seed(seed)
    tags = torch.tensor([1] * 6 + [0] * 10 + [1] * 8)
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, 7, 4, 4, 10, (1, 2, 2), 2, 2, 0, ("first",))
    timestep = torch.tensor([0.2, 0.5, 0.999])
    return dict(hidden_states=torch.randn(1, video_indices.numel(), 96, generator=g),
                audio_hidden_states=torch.randn(1, audio_indices.numel(), 32, generator=g),
                encoder_hidden_states=(torch.randn(1, tags.numel(), 5120, generator=g) * 4).to(torch.bfloat16),
                timestep=timestep, timestep_indices=torch.randint(0, 3, (position_ids.shape[0],), generator=g),
                token_tags=token_tags, position_ids=position_ids, video_indices=video_indices,
                audio_indices=audio_indices, text_indices=text_indices, return_dict=False)


@unittest.skipUnless(torch is not None and diffusers is not None, "needs torch and diffusers 0.40.0")
class PeftTests(unittest.TestCase):
    """The hooks on a tiny MiniMax-H3 transformer against PEFT itself (the reference's transformer.add_adapter,
    load_state_dict, set_adapters; unfused): the same output bit for bit, and the model exactly as before once the
    hooks are off."""

    @unittest.skipUnless(importlib.util.find_spec("peft"), "needs peft")
    def test_matches_peft_add_adapter(self):
        import copy
        from diffusers import MiniMaxH3Transformer3DModel
        from peft import LoraConfig
        from safetensors.torch import load_file
        tmp = Path(tempfile.mkdtemp(prefix="h3turbo-peft-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        torch.manual_seed(1234)
        MiniMaxH3Transformer3DModel(**TINY_TRANSFORMER).save_pretrained(tmp / "transformer")
        model = MiniMaxH3Transformer3DModel.from_pretrained(tmp / "transformer", torch_dtype=torch.bfloat16).eval()
        model.requires_grad_(False)
        lora = make_tiny_lora(tmp / "l.safetensors", model, rank=4, alpha="8", seed=1, std=0.5)
        inputs = packed_inputs()
        with torch.no_grad():
            base = model(**inputs)
        reference = copy.deepcopy(model)
        reference.add_adapter(LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                         target_modules=list(job_module.LORA_TARGETS), use_rslora=False))
        self.assertFalse(reference.load_state_dict(load_file(str(lora)), strict=False).unexpected_keys)
        reference.set_adapters("default", weights=1.0)
        with torch.no_grad():
            want = reference(**inputs)
        info = job_module.lora_info(lora)
        adapter = job_module.LoraAdapter(model, job_module.load_lora_pairs(lora, torch.device("cpu"), torch),
                                         info["scale"], torch)
        with torch.no_grad():
            got = model(**inputs)
        for a, b in zip(got, want):
            self.assertTrue(torch.equal(a, b), f"max diff {(a.float() - b.float()).abs().max()}")
        self.assertFalse(torch.equal(got[0], base[0]))
        adapter.remove()
        with torch.no_grad():
            for a, b in zip(model(**inputs), base):
                self.assertTrue(torch.equal(a, b))


@contextlib.contextmanager
def patched_forward(callback):
    """Calls callback(call_index, kwargs) before every transformer forward (patched on the class before the job loads
    the model; functools.wraps keeps the signature the denoiser reads the arguments by)."""
    from diffusers import MiniMaxH3Transformer3DModel
    original = MiniMaxH3Transformer3DModel.forward
    calls = []

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        callback(len(calls), kwargs)
        calls.append(1)
        return original(self, *args, **kwargs)
    MiniMaxH3Transformer3DModel.forward = forward
    try:
        yield calls
    finally:
        MiniMaxH3Transformer3DModel.forward = original


def patchify(latents):
    b, c, f, h, w = latents.shape
    return latents.reshape(b, c, f, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, c * 4)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


@unittest.skipUnless(torch is not None and diffusers is not None and FFMPEG,
                     "needs torch, diffusers 0.40.0 and ffmpeg (CPU is enough)")
class RunTests(unittest.TestCase):
    """run() through main() on the CPU, with a tiny random-weight MiniMax-H3 at the real 576p canvases."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-run-"))
        cls.base = make_tiny_base(cls.tmp / "base")
        cls.portrait = make_image(cls.tmp / "portrait.png", (824, 1472), seed=2)
        cls.land = make_image(cls.tmp / "land.png", (1456, 816), seed=5)
        cls.land_last = make_image(cls.tmp / "land-last.png", (1456, 816), seed=6)
        from diffusers import MiniMaxH3Transformer3DModel
        model = MiniMaxH3Transformer3DModel.from_pretrained(cls.base, subfolder="transformer")
        cls.lora = make_tiny_lora(cls.tmp / "lora.safetensors", model)
        cls.meminfo = cls.tmp / "meminfo"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.set_memory(100)

    def set_memory(self, gib):
        tmp = self.meminfo.with_suffix(".tmp")
        tmp.write_text(f"MemTotal: 127535648 kB\nMemAvailable: {int(gib * 1048576)} kB\n")
        os.replace(tmp, self.meminfo)

    def make_case(self, name, task, frames, orientation, canvas="draft", seed=0):
        """qwen/conditioning.pt in the Qwen worker's format (its keys, dtypes and row layout: a label, a vision block
        per keyframe, the prompt) and its conditioning.json."""
        case = self.tmp / name
        (case / "qwen").mkdir(parents=True)
        width, height = job_module.qwen_canvas(orientation, "576p", canvas)
        tokens = job_module.image_tokens(width, height)
        tags = []
        for _ in frames:
            tags += [1] * LABEL_ROWS + [0] * (tokens + 2)
        tags += [1] * PROMPT_ROWS
        g = torch.Generator().manual_seed(seed)
        spec = {"task": task, "first_frame": frames.get("first"), "last_frame": frames.get("last"), "references": []}
        payload = {"prompt": PROMPT, "task": task, "input_spec": spec,
                   "prompt_embeds": (torch.randn(1, len(tags), 5120, generator=g) * 4).to(torch.bfloat16),
                   "text_token_tags": torch.tensor(tags, dtype=torch.int64)}
        torch.save(payload, case / "qwen" / "conditioning.pt")
        meta = conditioning_meta((case / "qwen" / "conditioning.pt").read_bytes(), task=task, frames=frames,
                                 canvas=(width, height), text_rows=tags.count(1))
        (case / "qwen" / "conditioning.json").write_text(json.dumps(meta))
        return case

    def run_job(self, name, task="fl2va", frames=None, orientation="portrait", callback=None, **changes):
        """main() on a job for a new case; returns (status, receipt, phases printed, output dir, forwards)."""
        frames = {} if frames is None else frames
        case = self.make_case(name, task, {k: str(v) for k, v in frames.items()}, orientation,
                              canvas=changes.get("qwen_canvas", "draft"))
        out = case / "turbo"
        out.mkdir()
        job = {"version": 1, "task": task, "orientation": orientation, "resolution": "576p", "seconds": 5,
               "seed": 11, "steps": 4, "lora": str(self.lora), "shift": [6, 3],
               "conditioning": str(case / "qwen" / "conditioning.pt"), "qwen_canvas": "draft",
               "first_frame": str(frames["first"]) if "first" in frames else None,
               "last_frame": str(frames["last"]) if "last" in frames else None,
               "paths": {"base_model": str(self.base), "output": str(out)}}
        job.update(changes)
        (out / "job.json").write_text(json.dumps(job))
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        stdout = io.StringIO()
        with mock.patch.object(job_module, "DEVICE", "cpu"), mock.patch.object(job_module, "MEMINFO",
                                                                                str(self.meminfo)), \
                mock.patch.object(job_module, "apply_environment"), \
                patched_forward(callback or (lambda index, kwargs: None)) as calls, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = job_module.main(["--job", str(out / "job.json")])
        phases = [p[1] for p in map(job_module.parse_line, stdout.getvalue().splitlines()) if p and p[0] == "phase"]
        receipt = json.loads((out / "receipt.json").read_text())
        return status, receipt, phases, out, len(calls)

    def assert_outputs(self, receipt, out, size):
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(
            ["job.json", "receipt.json", "audio.pt", "latents.safetensors", f"clip_{size[0]}x{size[1]}_121f.mp4"]))
        streams = ffprobe(out / receipt["output"]["file"])
        video, audio = streams["video"], streams["audio"]
        self.assertEqual((video["codec_name"], video["width"], video["height"], video["pix_fmt"],
                          video["r_frame_rate"], int(video["nb_read_frames"])), ("h264", *size, "yuv420p", "24/1", 121))
        self.assertEqual((audio["codec_name"], audio["channels"], int(audio["sample_rate"])), ("aac", 2, 32000))
        self.assertAlmostEqual(float(audio["duration"]), 161333 / 32000, delta=0.05)
        pcm = torch.load(out / "audio.pt", map_location="cpu", weights_only=True)
        self.assertEqual((tuple(pcm["audio"].shape), pcm["audio"].dtype), ((1, 2, 161333), torch.float32))
        self.assertLess((out / "audio.pt").stat().st_size, 2 * 161333 * 4 + 4096)   # the trimmed PCM, not all of it
        from safetensors.torch import load_file
        latents = load_file(str(out / "latents.safetensors"))
        self.assertEqual((tuple(latents["video"].shape), tuple(latents["audio"].shape)),
                         ((1, 24, 37, size[1] // 16, size[0] // 16), (2, 32, 207)))
        for name, tensor in latents.items():
            self.assertEqual(receipt["latents_sha256"][name], job_module.tensor_sha256(tensor))
        self.assertEqual(receipt["decoded"], {"frames": 124, "samples": 165600, "kept_frames": 121,
                                              "kept_samples": 161333, "size": list(size)})
        self.assertEqual((receipt["state"], receipt["exit_status"], receipt["forwards"]), ("done", 0, 4))
        self.assertTrue(receipt["sigmas_match_expected"])
        self.assertEqual(receipt["layout_actual"]["sequence"], receipt["layout"]["sequence"])
        self.assertEqual(receipt["layout_actual"]["condition"], receipt["layout"]["condition"])
        self.assertEqual(receipt["lora"]["applied"], "unfused bf16 branch (forward hooks)")
        self.assertEqual((receipt["lora"]["scale"], receipt["lora"]["modules_hooked"]), (2.0, 18))
        for phase in ["prepare", "load", "load_vae", "lora_attach", "generate", "save_latents", "drop_transformer",
                      "decode", "decode_video", "decode_audio", "save_audio", "mux"] + [f"step_{i}" for i in
                                                                                         range(1, 5)]:
            self.assertIn(phase, receipt["times_s"], phase)
            self.assertIn(phase, receipt["lowest_mem_available_gib"], phase)
        self.assertNotIn("step_5", receipt["times_s"])
        self.assertEqual(receipt["lowest_mem_available_gib"]["all"], 100.0)
        self.assertEqual(receipt["load"], {"mode": "standard (cpu)"})
        self.assertEqual(receipt["attention_backend"], "native")
        self.assertEqual(receipt["versions"]["diffusers"], "0.40.0")

    def test_a_first_frame(self):
        """9:16, the first frame: the outputs, the receipt, the phases, and the condition rows computed
        independently (the stock fl2va path): the first frame stretched to the canvas with LANCZOS, ImageNet
        normalization, the VAE posterior sampled under seed 42 and rounded through fp16, normalized, noised to
        t = 0.999 with the clip generator's first draw, in front of the generated rows at the rotary time of target
        frame 0, pinned and never written; then the video noise; the audio noise from seed + 0x9E3779B9."""
        import numpy as np
        from PIL import Image
        seen = []

        def grab(index, kwargs):
            seen.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()})
        status, receipt, phases, out, forwards = self.run_job("first", frames={"first": self.portrait},
                                                              callback=grab)
        self.assertEqual(status, 0, receipt.get("traceback") or receipt.get("error"))
        self.assert_outputs(receipt, out, (576, 1024))
        self.assertEqual(phases, ["loading", "step 1/4", "step 2/4", "step 3/4", "step 4/4", "decoding", "saving"])
        self.assertEqual(receipt["images"]["first"], {"size": [824, 1472], "exif_transposed": False,
                                                      "stretch": 1.00485})
        self.assertIn("psnr_db", receipt["f0_vs_keyframe"])
        self.assertNotIn("last_vs_keyframe", receipt)
        self.assertEqual(receipt["audio_noise_seed"], 11 + 0x9E3779B9)
        text = receipt["conditioning"]["rows"]
        self.assertEqual(receipt["layout"], job_module.layout_rows(text, 576, 1024, 124, 1))
        n_cond = 576
        first = seen[0]
        vi, pos = first["video_indices"], first["position_ids"]
        rows0 = first["hidden_states"][0]
        self.assertEqual(int(vi[0]), text)
        self.assertTrue(bool((pos[vi[:n_cond], 0] == text).all()))
        self.assertTrue(torch.equal(pos[vi[:n_cond], 1:], pos[vi[n_cond:2 * n_cond], 1:]))
        self.assertEqual(receipt["layout_actual"]["condition_t"], float(text))
        for call in seen:
            self.assertTrue(torch.equal(call["hidden_states"][0][:n_cond], rows0[:n_cond]))
            t = call["timestep"][call["timestep_indices"][call["video_indices"][:n_cond]]]
            self.assertTrue(bool((t == torch.tensor(0.999)).all()))
        target_t = [float(c["timestep"][c["timestep_indices"][c["video_indices"][n_cond]]]) for c in seen]
        self.assertEqual([round(x, 4) for x in target_t], [round(1 - s, 4) for s in
                                                           job_module.expected_sigmas(4, 6)[:-1]])
        from diffusers import AutoencoderKLMiniMaxH3
        vae = AutoencoderKLMiniMaxH3.from_pretrained(self.base, subfolder="vae", torch_dtype=torch.float32).eval()
        keyframe = Image.open(self.portrait).convert("RGB").resize((576, 1024), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.array(keyframe)).permute(2, 0, 1)[None, :, None].float() / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        with torch.no_grad():
            posterior = vae.encode((pixels - mean) / std, return_dict=False)[0]
        z = posterior.sample(generator=torch.Generator().manual_seed(42)).to(torch.float16).float()
        z = (z - torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)) / torch.tensor(
            vae.config.latents_std).view(1, -1, 1, 1, 1)
        g = torch.Generator().manual_seed(11)
        t = torch.tensor(0.999)
        cond = t * z + (1.0 - t) * torch.randn(z.shape, generator=g)
        self.assertTrue(torch.allclose(patchify(cond), rows0[:n_cond], atol=1e-6, rtol=0),
                        f"max diff {(patchify(cond) - rows0[:n_cond]).abs().max()}")
        self.assertTrue(torch.equal(patchify(torch.randn((1, 24, 37, 64, 36), generator=g)), rows0[n_cond:]))
        audio = torch.randn((414, 32), generator=torch.Generator().manual_seed(11 + 0x9E3779B9))
        self.assertTrue(torch.equal(audio, first["audio_hidden_states"][0]))

    def test_t2va(self):
        """No keyframe: no image passed, diffusers' text-only workflow: no condition rows, the video noise is the
        clip generator's first draw."""
        seen = []

        def grab(index, kwargs):
            if index == 0:
                seen.append({k: v.detach().clone() for k, v in kwargs.items() if torch.is_tensor(v)})
        status, receipt, phases, out, forwards = self.run_job("t2va", task="t2va", orientation="landscape",
                                                              callback=grab)
        self.assertEqual(status, 0, receipt.get("traceback") or receipt.get("error"))
        self.assert_outputs(receipt, out, (1024, 576))
        self.assertEqual(receipt["layout"]["condition"], 0)
        self.assertEqual(receipt["layout_actual"]["condition"], 0)
        self.assertEqual(receipt["layout_actual"]["video_rows"], 21312)
        self.assertNotIn("condition_t", receipt["layout_actual"])
        self.assertNotIn("f0_vs_keyframe", receipt)
        self.assertEqual(receipt["conditioning"]["vision_rows"], 0)
        g = torch.Generator().manual_seed(11)
        self.assertTrue(torch.equal(patchify(torch.randn((1, 24, 37, 36, 64), generator=g)),
                                    seen[0]["hidden_states"][0]))

    def test_first_and_last(self):
        """16:9, first and last frame, audio noise "reference": two keyframes' condition rows, the last frame made
        against the last keyframe, the audio noise drawn from the clip's generator after the video noise."""
        seen = []

        def grab(index, kwargs):
            if index == 0:
                seen.append({k: v.detach().clone() for k, v in kwargs.items() if torch.is_tensor(v)})
        status, receipt, phases, out, forwards = self.run_job(
            "both", frames={"first": self.land, "last": self.land_last}, orientation="landscape", callback=grab,
            audio_noise="reference")
        self.assertEqual(status, 0, receipt.get("traceback") or receipt.get("error"))
        self.assert_outputs(receipt, out, (1024, 576))
        self.assertEqual(receipt["layout_actual"]["condition"], 2 * 576)
        self.assertEqual(receipt["layout_actual"]["video_rows"], 2 * 576 + 21312)
        self.assertEqual(receipt["last_vs_keyframe"]["frame"], 123)
        self.assertIn("psnr_db", receipt["f0_vs_keyframe"])
        self.assertNotIn("audio_noise_seed", receipt)
        g = torch.Generator().manual_seed(11)
        for _ in range(2):                                                  # one draw per keyframe
            torch.randn((1, 24, 1, 36, 64), generator=g)
        torch.randn((1, 24, 37, 36, 64), generator=g)                       # the video noise
        self.assertTrue(torch.equal(torch.randn((414, 32), generator=g), seen[0]["audio_hidden_states"][0]))

    def test_a_last_frame_alone(self):
        status, receipt, phases, out, forwards = self.run_job("last", frames={"last": self.portrait})
        self.assertEqual(status, 0, receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(receipt["layout_actual"]["condition"], 576)
        self.assertEqual(receipt["last_vs_keyframe"]["frame"], 123)
        self.assertNotIn("f0_vs_keyframe", receipt)
        self.assertEqual(receipt["job"]["keyframes"], ["last"])

    def test_aborted_for_memory(self):
        def drop(index, kwargs):
            if index == 1:
                self.set_memory(1)                          # 1 GiB < the 6 GiB floor
        status, receipt, phases, out, forwards = self.run_job("abort", frames={"first": self.portrait}, callback=drop)
        self.assertEqual((status, receipt["state"]), (3, "aborted"))
        self.assertLessEqual(forwards, 3)
        self.assertRegex(receipt["error"], r"MemAvailable 1\.00 GiB < 6 GiB during step_2")
        self.assertEqual(receipt["phase"], "step 2/4")
        self.assertEqual(receipt["lowest_mem_available_gib"]["all"], 1.0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["job.json", "receipt.json"])

    def test_sigterm(self):
        def stop(index, kwargs):
            if index == 1:
                os.kill(os.getpid(), signal.SIGTERM)
        status, receipt, phases, out, forwards = self.run_job("term", frames={"first": self.portrait}, callback=stop)
        self.assertEqual((status, receipt["state"], receipt["error"]), (143, "cancelled", "stopped by SIGTERM"))
        self.assertEqual(phases[-1], "step 2/4")
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["job.json", "receipt.json"])

    def test_a_conditioning_of_another_canvas_is_refused_before_loading(self):
        status, receipt, phases, out, forwards = self.run_job("canvas", frames={"first": self.portrait},
                                                              qwen_canvas="canvas")
        self.assertEqual(status, 0, receipt.get("error"))                       # made at the canvas: it runs
        case = self.make_case("canvas-draft", "fl2va", {"first": str(self.portrait)}, "portrait")
        job = json.loads((out / "job.json").read_text())
        job.update(conditioning=str(case / "qwen" / "conditioning.pt"))
        (out / "job.json").write_text(json.dumps(job))
        with mock.patch.object(job_module, "DEVICE", "cpu"), mock.patch.object(job_module, "apply_environment"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(job_module.main(["--job", str(out / "job.json")]), 2)
        receipt = json.loads((out / "receipt.json").read_text())
        self.assertEqual(receipt["state"], "refused")
        self.assertNotIn("load", receipt["times_s"])


if __name__ == "__main__":
    unittest.main()
