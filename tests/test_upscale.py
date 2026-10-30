"""upscale/regen2k_job.py (the experimental Upscale's GPU job) on the CPU: job.json (its pin too), the clip's and the
weights' files (5 s MP4s made by ffmpeg, 16:9 and 9:16, 121 and 124 frames, and the lengths and sizes it refuses), the
frames of a 5 s clip, the pin's fit against the LightX2V Turbo receipts' records and its grids, the shape table, the
portrait entries for NVIDIA's dicts, the upscale route and the column crop, check_package, the text check on a
stand-in, the command lines (NVIDIA's read of the clip among them) and the environment, the progress lines, the memory
trace, the script's --check and its refusals, main's receipt when the GPU part finishes, fails or gets SIGTERM (a
stand-in run()), and run() itself with a stand-in torch and a stand-in of NVIDIA's open_h3 that records its calls: for
16:9 the same calls as before the job took 9:16 clips, for 9:16 the portrait route, with both loads (fast and standard;
stand-ins for safetensors, diffusers and upscale/fastload.py record every load), a clip of 124 frames through the job's
copies of NVIDIA's encode and decode, and the pin's ends through the pinned refine (stand-ins for its parts). No GPU,
no weights: NVIDIA's package is a stand-in (open_h3/sigmas.py, and for run() adapters, pipeline and upscaler), the
weights are small files. upscale/fastload.py is checked against the patch's runtime/fastload.py. With NVIDIA's real
package where LFSTUDIO_UPSCALE_PKG (or its default) names it, its pinned files, its entries and its encode command are
checked too (read, not imported); skipped without it.

The parts that need torch (the merger, the lean block forward, the tensor checks, the column crop) run on the CPU when
torch is importable and are skipped otherwise. python3 -m unittest tests.test_upscale -v   (needs ffmpeg)
With Pillow and numpy (the Regen-2K environment) the pin's check runs on ffmpeg-made clips. With torch, numpy,
diffusers and imageio-ffmpeg (the Regen-2K environment) and NVIDIA's package (imported, run on the CPU) where
LFSTUDIO_UPSCALE_PKG names it: the job's copies of NVIDIA's encode_video and decode (encode_clip, decode_released) with
a stand-in VAE against NVIDIA's own, the pin's layout, and the pinned refine with a stand-in transformer against
NVIDIA's Pipeline.refine and the bench wrapper's K1 refine (copied here). Run them with CUDA hidden and
LFSTUDIO_UPSCALE_PKG set:
    CUDA_VISIBLE_DEVICES= $LFSTUDIO_HOME/envs/regen2k/bin/python -I -m unittest discover -s tests -p test_upscale.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "upscale" / "regen2k_job.py"
FASTLOAD = REPO / "upscale" / "fastload.py"
PATCH_FASTLOAD = "models/minimax_h3/Sol-H3-Spark/runtime/fastload.py"

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    raise unittest.SkipTest("ffmpeg not installed")


def load_wrapper():
    spec = importlib.util.spec_from_file_location("regen2k_job", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job_module = load_wrapper()
JobError = job_module.JobError

try:
    import torch
except ImportError:
    torch = None

# a stand-in for NVIDIA's open_h3.sigmas with values the real one never gives, so the plan shows whose they are
STAND_IN_SIGMAS = '''SCHEDULES = {3: [0.9, 0.8, 0.6, 0.0], 2: [0.75, 0.5, 0.0]}


def shifted_grid(start, steps):
    return [start] + [round(start / (i + 1), 6) for i in range(1, steps)] + [0.0]
'''


def make_mp4(path, width=1344, height=768, frames=121, fps=24, audio=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s={width}x{height}:r={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=32000:duration={frames / fps:.4f}"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-ac", "2"] if audio else ["-an"]
    subprocess.run(cmd + [str(path)], check=True)
    return path


class FakeTensor:
    """Just enough of a tensor for check_text: shape, flatten(), tolist()."""

    def __init__(self, shape, values=None):
        self.shape, self.values = tuple(shape), values

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)


def conditioning(rows=41, tags=None, task="t2va", width=5120):
    cond = {"prompt_embeds": FakeTensor((1, rows, width)), "text_token_tags": FakeTensor((rows,), tags or [1] * rows)}
    if task is not None:
        cond["task"] = task
    return cond


class Fixture(unittest.TestCase):
    """A clip directory as the studio leaves it, NVIDIA's package (stand-in), the weights (empty files), an output
    directory and a valid job.json for them."""

    @classmethod
    def setUpClass(cls):
        cls.media = Path(tempfile.mkdtemp(prefix="upscale-media-"))
        cls.final = make_mp4(cls.media / "final.mp4")
        cls.portrait = make_mp4(cls.media / "portrait.mp4", width=768, height=1344)
        cls.final124 = make_mp4(cls.media / "final124.mp4", frames=124)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.media, ignore_errors=True)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="upscale-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.case = self.tmp / "outputs" / "webui" / "20261028-010203-t2va-landscape-5s" / "web_010203_abcdef"
        for rel, data in (("qwen/conditioning.pt", b"conditioning"), ("stage1/stage1_direct_tensors.pt", b"pcm")):
            (self.case / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.case / rel).write_bytes(data)
        (self.case / "stage2").mkdir()
        shutil.copy(self.final, self.case / "stage2" / "refined_1344x768_121f.mp4")
        self.package = self.tmp / "Open-H3-Regenerate-2K"
        (self.package / "open_h3").mkdir(parents=True)
        (self.package / "open_h3" / "__init__.py").write_text("")
        (self.package / "open_h3" / "pipeline.py").write_text("raise ImportError('a stand-in: no pipeline')\n")
        (self.package / "open_h3" / "sigmas.py").write_text(STAND_IN_SIGMAS)
        self.weights = self.tmp / "weights"
        self.base = self.tmp / "MiniMax-H3"
        for rel in (job_module.LORA_FILE, job_module.CURVE_FILE, job_module.CURVE_TABLE, job_module.UPSCALER_FILE):
            (self.weights / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.weights / rel).write_bytes(b"w" * 1000)
        for part in job_module.BASE_PARTS:
            (self.base / part).mkdir(parents=True)
            (self.base / part / "config.json").write_text("{}")
            (self.base / part / "diffusion_pytorch_model.safetensors").write_bytes(b"b" * 3000)
        self.out = self.tmp / "outputs" / "webui" / "20261028-020304-upscale-2k" / "up_020304_abcdef" / "regen"
        self.out.mkdir(parents=True)

    def job(self, **changes):
        data = {"version": 1, "case": str(self.case), "resolution": "2k", "steps": 2,
                "start_sigma": 0.7, "seed": 1234, "audio": "clip",
                "paths": {"package": str(self.package), "weights": str(self.weights), "base_model": str(self.base),
                          "output": str(self.out)},
                "meta": {"id": "f" * 32, "parent": "a" * 32, "strength": "balanced"}}
        data.update(changes)
        return {k: v for k, v in data.items() if v is not DROP}

    def write_job(self, data=None, **changes):
        path = self.out / "job.json"
        path.write_text(json.dumps(self.job(**changes) if data is None else data))
        return path

    def script(self, *args, check=False):
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True,
                              timeout=120, check=check)


DROP = object()


def patch_new_file(rel, patch="lf-studio.patch"):
    """A file that patches/<patch> adds: (its content, the abbreviated blob hash of its index line)."""
    text = (REPO / "patches" / patch).read_text()
    for section in re.split(r"(?m)^(?=diff --git )", text):
        if section.startswith(f"diff --git a/{rel} b/{rel}\n") and "\nnew file mode" in section:
            blob = re.search(r"(?m)^index 0+\.\.([0-9a-f]+)$", section).group(1)
            body = section.split("\n+++ ", 1)[1].split("\n", 1)[1]
            return "".join(line[1:] + "\n" for line in body.splitlines() if line.startswith("+")), blob
    raise AssertionError(f"{rel} is not a new file in the patch")


class FastloadCopyTests(unittest.TestCase):
    def test_the_patch_file_byte_for_byte(self):
        """upscale/fastload.py is runtime/fastload.py as patches/lf-studio.patch adds it, so the two never drift:
        the same text as the patch's lines, and the same git blob as the patch's index line names."""
        data = FASTLOAD.read_bytes()
        text, blob = patch_new_file(PATCH_FASTLOAD)
        self.assertEqual(data, text.encode())
        self.assertTrue(hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest().startswith(blob), blob)


class JobTests(Fixture):
    def test_a_valid_job_and_its_defaults(self):
        job = job_module.parse_job(self.job(load=DROP, preread=DROP, vae_batch=DROP, version=DROP, meta=DROP),
                                   "/jobs/x")
        self.assertEqual((job["resolution"], job["audio"], job["steps"], job["seed"], job["load"]),
                         ("2k", "clip", 2, 1234, "fast"))
        self.assertNotIn("source", job)
        self.assertEqual((job["start_sigma"], job["preread"], job["vae_batch"], job["meta"]), (0.7, False, 26, {}))
        self.assertEqual((job["case"], job["package"], job["output"]), (self.case, self.package, self.out))
        self.assertIsInstance(job["start_sigma"], float)
        record = job_module.job_record(job)
        self.assertEqual(json.loads(json.dumps(record))["case"], str(self.case))
        data = self.job()
        del data["paths"]["output"]
        self.assertEqual(job_module.parse_job(data, "/jobs/x")["output"], Path("/jobs/x"))

    def test_nvidia_schedule_only_for_its_step_counts(self):
        for steps in (2, 3):
            self.assertIsNone(job_module.parse_job(self.job(steps=steps, start_sigma=None), "/")["start_sigma"])
        for steps in (1, 4):
            with self.assertRaisesRegex(JobError, "start_sigma"):
                job_module.parse_job(self.job(steps=steps, start_sigma=None), "/")
            self.assertEqual(job_module.parse_job(self.job(steps=steps, start_sigma=0.5), "/")["steps"], steps)

    def test_the_strengths_are_valid_jobs(self):
        self.assertIn(job_module.DEFAULT_STRENGTH, job_module.STRENGTHS)
        self.assertIn(job_module.DEFAULT_AUDIO, job_module.AUDIO)
        self.assertEqual(job_module.STRENGTHS[job_module.DEFAULT_STRENGTH], (0.7, 2))
        self.assertEqual(job_module.STRENGTHS, {"faithful": (0.5, 2), "balanced": (0.7, 2), "full": (None, 3)})
        for sigma, steps in job_module.STRENGTHS.values():
            job_module.parse_job(self.job(start_sigma=sigma, steps=steps), "/")

    def test_bad_values_are_refused_by_name(self):
        nan = float("nan")
        for field, value in (("resolution", "768p"), ("resolution", "2K"),
                             ("resolution", ["2k"]), ("audio", "both"), ("steps", 0), ("steps", 5), ("steps", "2"),
                             ("steps", True), ("steps", 2.0), ("start_sigma", 0), ("start_sigma", 1),
                             ("start_sigma", -0.1), ("start_sigma", "0.7"), ("start_sigma", True),
                             ("start_sigma", nan), ("seed", -1), ("seed", 2 ** 63), ("seed", 1.5), ("seed", True),
                             ("seed", None), ("preread", 1), ("preread", "yes"), ("load", "quick"),
                             ("load", "Fast"), ("load", None), ("load", True), ("vae_batch", 0),
                             ("vae_batch", 257), ("vae_batch", "26"), ("version", 2), ("version", True),
                             ("meta", []), ("case", "relative/dir"), ("case", ""), ("case", 7), ("paths", []),
                             ("paths", None)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(JobError, field):
                    job_module.parse_job(self.job(**{field: value}), "/")

    def test_no_source(self):
        """job.json has no "source" any more (the draft, from the clip's Stage-1 latent, is gone): any value of it is
        an unknown field."""
        for value in ("final", "draft", None):
            with self.subTest(value=value), self.assertRaisesRegex(JobError, r"^unknown field\(s\): source$"):
                job_module.parse_job(self.job(source=value), "/")
        self.assertNotIn("source", job_module.JOB_KEYS)

    def test_load_and_preread(self):
        """load: fast (the default) or standard; preread only with standard (the fast load reads past the page
        cache), refused with fast, also when fast is the default."""
        for load in ("fast", "standard"):
            self.assertEqual(job_module.parse_job(self.job(load=load), "/")["load"], load)
        job = job_module.parse_job(self.job(load="standard", preread=True), "/")
        self.assertEqual((job["load"], job["preread"]), ("standard", True))
        for changes in ({"load": "fast", "preread": True}, {"load": DROP, "preread": True}):
            with self.subTest(**{k: str(v) for k, v in changes.items()}):
                with self.assertRaisesRegex(JobError, '^preread: only with load "standard"'):
                    job_module.parse_job(self.job(**changes), "/")
        self.assertFalse(job_module.parse_job(self.job(load="fast", preread=False), "/")["preread"])

    def test_paths(self):
        for key in ("package", "weights", "base_model", "output"):
            data = self.job()
            data["paths"][key] = "not/absolute"
            with self.assertRaisesRegex(JobError, f"paths.{key}"):
                job_module.parse_job(data, "/")
        for key in ("package", "weights", "base_model"):
            data = self.job()
            del data["paths"][key]
            with self.assertRaisesRegex(JobError, f"paths.{key}"):
                job_module.parse_job(data, "/")
        data = self.job()
        data["paths"]["text"] = "/x"
        with self.assertRaisesRegex(JobError, "paths: unknown field.*text"):
            job_module.parse_job(data, "/")

    def test_unknown_and_missing_fields(self):
        with self.assertRaisesRegex(JobError, "unknown field.*start_sigms"):
            job_module.parse_job(self.job(start_sigms=0.7), "/")
        for field in job_module.REQUIRED:
            with self.subTest(field=field), self.assertRaisesRegex(JobError, f"missing field.*{field}"):
                job_module.parse_job(self.job(**{field: DROP}), "/")
        with self.assertRaisesRegex(JobError, "JSON object"):
            job_module.parse_job([], "/")

    def test_load_job(self):
        path = self.write_job()
        self.assertEqual(job_module.load_job(path)["case"], self.case)
        path.write_text("{not json")
        with self.assertRaisesRegex(JobError, "not JSON"):
            job_module.load_job(path)
        path.write_text(json.dumps({"meta": {"x": "y" * (1 << 20)}}))
        with self.assertRaisesRegex(JobError, "larger than 1 MiB"):
            job_module.load_job(path)
        with self.assertRaisesRegex(JobError, "cannot read"):
            job_module.load_job(self.out / "missing.json")


SHA = "0123456789abcdef" * 4


def pin(*ends, **changes):
    """job.json's "pin" for the ends named (first, last), with `changes`."""
    data = {"keyframes": [{"anchor": end, "image": f"/up/input/keyframe/{end}.png", "sha256": SHA,
                           "fit": "stretch" if index == 0 else "cover-crop", "canvas": [1024, 576]}
                          for index, end in enumerate(ends or ("first",))]}
    data.update(changes)
    return {k: v for k, v in data.items() if v is not DROP}


class PinJobTests(Fixture):
    """job.json's "pin" (parse_pin): its keyframes, the check and the refine's values, and every refusal by name."""

    def test_a_pin_and_its_defaults(self):
        job = job_module.parse_job(self.job(pin=pin("first", "last")), "/")
        self.assertEqual(job["pin"], {
            "keyframes": [{"anchor": "first", "image": "/up/input/keyframe/first.png", "sha256": SHA, "fit": "stretch",
                           "canvas": (1024, 576)},
                          {"anchor": "last", "image": "/up/input/keyframe/last.png", "sha256": SHA,
                           "fit": "cover-crop", "canvas": (1024, 576)}],
            "check": {"grid": "384p", "min_db": 28}, "noise_aug": 0.999, "encode_seed": 42, "noise_seed": 42})
        self.assertIsNone(job_module.parse_job(self.job(), "/")["pin"])
        self.assertEqual(json.loads(json.dumps(job_module.job_record(job)))["pin"]["keyframes"][1]["canvas"],
                         [1024, 576])
        job = job_module.parse_job(self.job(pin=pin("last", check={"min_db": 30.5}, noise_aug=0.99, encode_seed=7,
                                                    noise_seed=0)), "/")["pin"]
        self.assertEqual((job["keyframes"][0]["anchor"], job["check"], job["noise_aug"], job["encode_seed"],
                          job["noise_seed"]), ("last", {"grid": "384p", "min_db": 30.5}, 0.99, 7, 0))
        for fit in job_module.FITS:
            data = pin("first")
            data["keyframes"][0]["fit"] = fit
            self.assertEqual(job_module.parse_pin(data)["keyframes"][0]["fit"], fit)
        self.assertEqual(job_module.FITS, ("as is", "stretch", "cover-crop"))
        self.assertIn("pin", job_module.JOB_KEYS)
        self.assertEqual(job_module.JOB_VERSION, 1)             # an older job refuses the key: no new version

    def test_refusals_by_name(self):
        def keyframe(index=0, **changes):
            data = pin("first", "last")
            data["keyframes"][index].update(changes)
            data["keyframes"][index] = {k: v for k, v in data["keyframes"][index].items() if v is not DROP}
            return data
        nan = float("nan")
        for value, message in (
                ([], r"^pin: a JSON object"), ({}, r"^pin\.keyframes: a list of 1 or 2"),
                (pin("first", colour="red"), r"^pin: unknown field\(s\): colour$"),
                (pin("first", keyframes=[]), r"^pin\.keyframes: a list of 1 or 2"),
                (pin("first", keyframes="first"), r"^pin\.keyframes: a list"),
                (pin("first", "last", "first"), r"^pin\.keyframes: a list of 1 or 2"),
                (pin("last", "first"), r"^pin\.keyframes: one per end, first before last, got last, first$"),
                (pin("first", "first"), r"^pin\.keyframes: one per end"),
                (pin("first", keyframes=["first"]), r"^pin\.keyframes\[0\]: a JSON object"),
                (keyframe(anchor="middle"), r"^pin\.keyframes\[0\]\.anchor: one of first, last"),
                (keyframe(1, anchor=None), r"^pin\.keyframes\[1\]\.anchor"),
                (keyframe(image="input/keyframe/first.png"), r"^pin\.keyframes\[0\]\.image: an absolute path"),
                (keyframe(image=""), r"^pin\.keyframes\[0\]\.image"),
                (keyframe(sha256=SHA.upper()), r"^pin\.keyframes\[0\]\.sha256: 64 lowercase hex"),
                (keyframe(sha256=SHA[:63]), r"^pin\.keyframes\[0\]\.sha256"),
                (keyframe(sha256=None), r"^pin\.keyframes\[0\]\.sha256"),
                (keyframe(1, fit="cover"), r"^pin\.keyframes\[1\]\.fit: one of as is, stretch, cover-crop, got "
                                           r"'cover'$"),
                (keyframe(fit="Stretch"), r"^pin\.keyframes\[0\]\.fit"),
                (keyframe(canvas=[1024]), r"^pin\.keyframes\[0\]\.canvas: \[width, height\]"),
                (keyframe(canvas=[1024, 0]), r"^pin\.keyframes\[0\]\.canvas"),
                (keyframe(canvas=[1024.0, 576]), r"^pin\.keyframes\[0\]\.canvas"),
                (keyframe(canvas=[True, 576]), r"^pin\.keyframes\[0\]\.canvas"),
                (keyframe(canvas="1024x576"), r"^pin\.keyframes\[0\]\.canvas"),
                (keyframe(canvas=DROP), r"^pin\.keyframes\[0\]: missing field\(s\): canvas$"),
                (keyframe(1, frame=123), r"^pin\.keyframes\[1\]: unknown field\(s\): frame$"),
                (pin("first", check={"grid": "768p"}), r"^pin\.check\.grid: one of 384p, got '768p'$"),
                (pin("first", check={"min_db": -1}), r"^pin\.check\.min_db: a number of dB from 0 to 99"),
                (pin("first", check={"min_db": 100}), r"^pin\.check\.min_db"),
                (pin("first", check={"min_db": nan}), r"^pin\.check\.min_db"),
                (pin("first", check={"min_db": "28"}), r"^pin\.check\.min_db"),
                (pin("first", check={"min_db": True}), r"^pin\.check\.min_db"),
                (pin("first", check={"grid": "384p", "frames": 1}), r"^pin\.check: a JSON object with grid and "
                                                                    r"min_db"),
                (pin("first", check=[]), r"^pin\.check: a JSON object"),
                (pin("first", noise_aug=1), r"^pin\.noise_aug: a number between 0 and 1"),
                (pin("first", noise_aug=0), r"^pin\.noise_aug"), (pin("first", noise_aug=True), r"^pin\.noise_aug"),
                (pin("first", noise_aug="0.999"), r"^pin\.noise_aug"),
                (pin("first", encode_seed=-1), r"^pin\.encode_seed: a whole number from 0 to 2\*\*63-1"),
                (pin("first", noise_seed=2 ** 63), r"^pin\.noise_seed"),
                (pin("first", noise_seed=1.0), r"^pin\.noise_seed"), (pin("first", encode_seed=None),
                                                                       r"^pin\.encode_seed")):
            with self.subTest(value=value), self.assertRaisesRegex(JobError, message):
                job_module.parse_job(self.job(pin=value), "/")


def fit_record(size, canvas, first):
    """The fit jobs/h3turbo_job.py's keyframe_fit records (diffusers 0.40's rule) for a picture of `size`."""
    spec = importlib.util.spec_from_file_location("h3turbo_job", REPO / "jobs" / "h3turbo_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.keyframe_fit(size, canvas, first)


# The keyframes of LightX2V Turbo Keyframes clips as their receipts recorded them (turbo/receipt.json "images" and
# "geometry"; from GPU runs of 2026-10-29: the 576p 16:9 clips with first and last, last alone and a first frame alone,
# a 576p 9:16 clip, and a 768p 16:9 clip with first and last)
TURBO_RECEIPTS = {
    "576p first and last": ((1024, 576), {
        "first": {"size": [1344, 768], "fit": "stretch", "stretch": 1.01587},
        "last": {"size": [1344, 768], "fit": "cover-crop", "resized": [1024, 585], "crop": [0, 4, 1024, 580],
                 "kept": 0.9846}}),
    "576p last alone": ((1024, 576), {"last": {"size": [1344, 768], "fit": "stretch", "stretch": 1.01587}}),
    "576p 10 s first": ((1024, 576), {"first": {"size": [1456, 816], "fit": "stretch", "stretch": 0.99634}}),
    "576p 9:16 first": ((576, 1024), {"first": {"size": [1648, 2944], "fit": "stretch", "stretch": 1.00485}}),
    "768p first and last": ((1344, 768), {"first": {"size": [1344, 768], "fit": "as is"},
                                          "last": {"size": [1344, 768], "fit": "as is"}})}


class PinFitTests(unittest.TestCase):
    """How the pin puts a picture on the refine's grid: H3's fit at the clip's canvas (cover_crop, fit_box) against
    the LightX2V Turbo job's records, and the computed grids. Standard library."""

    def test_the_turbo_receipts(self):
        """Every keyframe a Turbo receipt recorded: H3's rule gives its fit, and for a cover-crop its resized size and
        crop; the box is None for "as is" and "stretch" (the whole picture) and the crop mapped into the picture for
        "cover-crop"."""
        for name, (canvas, images) in TURBO_RECEIPTS.items():
            for index, (anchor, record) in enumerate(images.items()):
                with self.subTest(clip=name, anchor=anchor):
                    size = tuple(record["size"])
                    self.assertEqual(fit_record(size, canvas, first=index == 0),
                                     {k: v for k, v in record.items() if k != "size"})
                    box = job_module.fit_box(size, canvas, record["fit"])
                    if record["fit"] != "cover-crop":
                        self.assertIsNone(box)
                        continue
                    resized, crop = job_module.cover_crop(size, canvas)
                    self.assertEqual((list(resized), list(crop)), (record["resized"], record["crop"]))
                    (rw, rh), (left, top, right, bottom) = record["resized"], record["crop"]
                    self.assertEqual(box, (left * size[0] / rw, top * size[1] / rh, right * size[0] / rw,
                                           bottom * size[1] / rh))
        box = job_module.fit_box((1344, 768), (1024, 576), "cover-crop")
        self.assertEqual([round(v, 2) for v in box], [0.0, 5.25, 1344.0, 761.44])

    def test_cover_crop_is_the_turbo_jobs(self):
        """cover_crop is keyframe_fit's cover-crop (diffusers' rounding: round, centred at (size - canvas) // 2) for
        pictures of many shapes on both orientations' canvases."""
        for canvas in ((1024, 576), (1344, 768), (576, 1024), (768, 1344)):
            for size in ((1344, 768), (1344, 800), (800, 1344), (4032, 3024), (3024, 4032), (1000, 1000), (1023, 577),
                         (640, 360), (5000, 1200), (1080, 1920), (1920, 1080), (1025, 576)):
                if size == canvas:
                    continue
                with self.subTest(canvas=canvas, size=size):
                    record = fit_record(size, canvas, first=False)
                    resized, crop = job_module.cover_crop(size, canvas)
                    self.assertEqual((list(resized), list(crop)), (record["resized"], record["crop"]))
                    self.assertEqual((crop[2] - crop[0], crop[3] - crop[1]), canvas)
                    left, top, right, bottom = job_module.fit_box(size, canvas, "cover-crop")
                    self.assertTrue(0 <= left < right <= size[0] and 0 <= top < bottom <= size[1])
        with self.assertRaisesRegex(ValueError, "unknown fit 'cover'"):
            job_module.fit_box((1344, 768), (1024, 576), "cover")

    def test_the_computed_grids(self):
        """The pictures go on the grid the refine computes at: NVIDIA's RESOLUTIONS (1088 / 2176 rows for 1080p and
        4K), turned for 9:16; the check's grid is 672x384 either way round."""
        for orientation, grids in (("landscape", {"1080p": (1920, 1088), "2k": (2560, 1440), "4k": (3840, 2176)}),
                                   ("portrait", {"1080p": (1088, 1920), "2k": (1440, 2560), "4k": (2176, 3840)})):
            for resolution, grid in grids.items():
                self.assertEqual(job_module.computed_grid(orientation, resolution), grid, (orientation, resolution))
                target = (LANDSCAPE_TARGETS if orientation == "landscape" else PORTRAIT_TARGETS)[resolution]
                self.assertEqual((grid[1] // 16, grid[0] // 16), target[2:])      # the refine's latent grid
        self.assertEqual(job_module.PIN_CHECK_GRIDS, {"384p": {"landscape": (672, 384), "portrait": (384, 672)}})
        self.assertEqual((job_module.PIN_MIN_DB, job_module.PIN_NOISE_AUG, job_module.PIN_SEED), (28, 0.999, 42))


def nvidia_read_command(source, path, width, height, ffmpeg):
    """The ffmpeg command NVIDIA's encode_video builds (its subprocess.run list, from its source): `path` read at
    width x height."""
    import ast
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "encode_video")
    call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "subprocess.run")
    names = {"imageio_ffmpeg": types.SimpleNamespace(get_ffmpeg_exe=lambda: ffmpeg), "str": str, "path": path,
             "width": width, "height": height}
    return eval(compile(ast.Expression(call.args[0]), "pipeline.py", "eval"), names)


class FrameTests(unittest.TestCase):
    """The frames of a 5 s clip (FRAME_CASES) and how the job reads its MP4 (clip_read_command), without torch."""

    def test_the_cases(self):
        self.assertEqual(job_module.FRAME_CASES, {121: {"pad": 3}, 124: {"pad": 0}})
        self.assertEqual((job_module.NVIDIA_FRAMES, job_module.FPS), (121, "24/1"))
        self.assertEqual(job_module.frame_case(121), {"kept": 121, "made": 124, "pad": 3, "T": 37})
        self.assertEqual(job_module.frame_case(124), {"kept": 124, "made": 124, "pad": 0, "T": 37})
        for kept in job_module.FRAME_CASES:                 # the shape NVIDIA's pipeline is written for, either way
            case = job_module.frame_case(kept)
            self.assertEqual((case["kept"] + case["pad"], case["T"]), (124, job_module.NATIVE_SHAPE[1]))
        with self.assertRaises(KeyError):
            job_module.frame_case(122)

    def test_h3s_frames(self):
        """H3 makes 17n+5 frames (diffusers' align_num_frames), 5n+2 latent frames (video_latent_num_frames)."""
        for kept, made in ((121, 124), (124, 124), (122, 124), (125, 141), (73, 73), (241, 243), (361, 362),
                           (153, 158)):
            self.assertEqual(job_module.made_frames(kept), made, kept)
        for frames, latents in ((124, 37), (73, 22), (158, 47), (243, 72), (362, 107), (5, 2)):
            self.assertEqual(job_module.latent_frames(frames), latents, frames)
        with self.assertRaisesRegex(ValueError, "121 frames are not 17n"):
            job_module.latent_frames(121)

    def test_the_read_is_nvidias(self):
        """121 frames with 3 copies: NVIDIA's encode_video command (pipeline.py:57-61), item for item; 124 with none:
        the same without tpad, so a short MP4 reads short (and fails the length check) instead of being padded."""
        nvidia = ["FFMPEG", "-v", "error", "-i", "/case/stage2/clip.mp4", "-vf",
                  "fps=24,scale=1344:768:force_original_aspect_ratio=increase:flags=bilinear,crop=1344:768,"
                  "tpad=stop_mode=clone:stop_duration=6", "-frames:v", "121", "-an", "-f", "rawvideo", "-pix_fmt",
                  "rgb24", "pipe:1"]
        self.assertEqual(job_module.clip_read_command(Path("/case/stage2/clip.mp4"), (1344, 768), 121, 3, "FFMPEG"),
                         nvidia)
        whole = job_module.clip_read_command("/case/stage2/clip.mp4", (1344, 768), 124, 0, "FFMPEG")
        self.assertEqual(whole, nvidia[:6] + [nvidia[6].replace(",tpad=stop_mode=clone:stop_duration=6", ""),
                                              "-frames:v", "124"] + nvidia[9:])
        self.assertNotIn("tpad", " ".join(whole))
        portrait = job_module.clip_read_command("/c.mp4", (768, 1344), 121, 3, "FFMPEG")
        self.assertIn("scale=768:1344:force_original_aspect_ratio=increase:flags=bilinear,crop=768:1344,", portrait[6])
        package = nvidia_package()
        if package is None:
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")
        source = (package / "open_h3" / "pipeline.py").read_text()
        for size in ((1344, 768), (768, 1344)):
            self.assertEqual(job_module.clip_read_command("/case/clip.mp4", size, 121, 3, "FFMPEG"),
                             nvidia_read_command(source, "/case/clip.mp4", *size, "FFMPEG"))
        # the rest of its encode_video, which encode_clip copies with the frames as parameters
        for line in ("if len(raw) != 121 * height * width * 3:",
                     "pixels = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(121, height, width, 3).copy())",
                     "pixels = pixels.permute(3, 0, 1, 2)[None].cuda().float() / 255.0",
                     "pixels = torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, 3, -1, -1)], dim=2)",
                     "latents = posterior.sample(generator=torch.Generator().manual_seed(seed))",
                     "latents = ((latents - mean) / std)[0].float().cpu()",
                     "frames = self.fast_vae.decode_frames(latents)[:121] if self.fast_vae is not None else "
                     "self._decode_released(latents)",
                     "return ((pixels * pixel_std + pixel_mean).clamp(0, 1)[0, :, :121].permute(1, 2, 3, 0) * 255)"
                     ".round().byte()"):
            self.assertIn(line, source)


class StudioSettingsTests(unittest.TestCase):
    """studio_settings: the LFSTUDIO_UPSCALE_* settings as the server and bin/lf-studio --check read them."""

    def test_defaults(self):
        settings, problems = job_module.studio_settings({"LFSTUDIO_HOME": "/srv/lf"})
        self.assertEqual(problems, [])
        self.assertEqual(settings, {"python": Path("/srv/lf/envs/regen2k/bin/python"),
                                    "package": Path("/srv/lf/Open-H3-Regenerate-2K"),
                                    "weights": Path("/srv/lf/checkpoints/Open-H3-Regenerate-2K"),
                                    "base_model": Path("/srv/lf/checkpoints/MiniMax-H3"), "strength": "balanced",
                                    "audio": "clip", "load": "fast", "preread": False, "o_direct": True,
                                    "min_free_gib": 110.0, "guard_gib": 5.0})
        settings, _ = job_module.studio_settings({"LFSTUDIO_HOME": "/srv/lf", "LFSTUDIO_CHECKPOINTS": "/data/ckpt",
                                                  "LFSTUDIO_UPSCALE_STRENGTH": ""})      # empty: the default
        self.assertEqual((settings["weights"], settings["base_model"], settings["strength"]),
                         (Path("/data/ckpt/Open-H3-Regenerate-2K"), Path("/data/ckpt/MiniMax-H3"), "balanced"))
        with mock.patch.dict(os.environ, {"HOME": "/srv/someone"}):
            self.assertEqual(job_module.studio_settings({"LFSTUDIO_UPSCALE_PKG": "~/regen"})[0]["package"],
                             Path("/srv/someone/regen"))

    def test_values(self):
        settings, problems = job_module.studio_settings({
            "LFSTUDIO_UPSCALE_STRENGTH": "full", "LFSTUDIO_UPSCALE_AUDIO": "silence",
            "LFSTUDIO_UPSCALE_LOAD": "standard", "LFSTUDIO_UPSCALE_PREREAD": "1", "LFSTUDIO_UPSCALE_MIN_FREE_GIB": "0",
            "LFSTUDIO_UPSCALE_GUARD_GIB": "2.5"})
        self.assertEqual(problems, [])
        self.assertEqual([settings[k] for k in ("strength", "audio", "load", "preread", "min_free_gib", "guard_gib")],
                         ["full", "silence", "standard", True, 0.0, 2.5])
        settings, problems = job_module.studio_settings({"LFSTUDIO_UPSCALE_LOAD": "fast",
                                                         "LFSTUDIO_UPSCALE_PREREAD": ""})       # empty: the default
        self.assertEqual((settings["load"], settings["preread"], problems), ("fast", False, []))
        _, problems = job_module.studio_settings({
            "LFSTUDIO_UPSCALE_PYTHON": "python3", "LFSTUDIO_UPSCALE_STRENGTH": "light",
            "LFSTUDIO_UPSCALE_AUDIO": "Clip", "LFSTUDIO_UPSCALE_LOAD": "quick", "LFSTUDIO_UPSCALE_PREREAD": "yes",
            "LFSTUDIO_UPSCALE_MIN_FREE_GIB": "-1", "LFSTUDIO_UPSCALE_GUARD_GIB": "inf"})
        self.assertEqual(problems, [
            "LFSTUDIO_UPSCALE_PYTHON=python3: must be an absolute path",
            "LFSTUDIO_UPSCALE_STRENGTH=light: must be faithful, balanced or full",
            "LFSTUDIO_UPSCALE_AUDIO=Clip: must be clip or silence",
            "LFSTUDIO_UPSCALE_LOAD=quick: must be fast or standard", "LFSTUDIO_UPSCALE_PREREAD=yes: must be 0 or 1",
            "LFSTUDIO_UPSCALE_MIN_FREE_GIB=-1: must be a number of GiB, 0 or more",
            "LFSTUDIO_UPSCALE_GUARD_GIB=inf: must be a number of GiB, 0 or more (see bin/env.example)"])

    def test_min_free_for(self):
        """4K needs 5 GiB more than the floor (its job used ~105 GiB, 2K ~101); a floor of 0 stays off."""
        self.assertEqual([job_module.min_free_for(res, 110.0) for res in ("1080p", "2k", "4k")], [110.0, 110.0, 115.0])
        self.assertEqual(job_module.min_free_for("4k", 0.0), 0.0)
        self.assertEqual(job_module.min_free_for("4k", 100.0), 105.0)

    def test_the_patch_switches_for_o_direct(self):
        """SPARK_FAST_SAFETENSORS and SPARK_DIRECT_IO, which the job inherits: anything but 1 turns fastload's O_DIRECT
        reads off, as fastload itself reads them (the server warns at start); not a problem, any load."""
        self.assertEqual(job_module.FASTLOAD_SWITCHES, ("SPARK_FAST_SAFETENSORS", "SPARK_DIRECT_IO"))
        for env, on in (({"SPARK_FAST_SAFETENSORS": "1", "SPARK_DIRECT_IO": "1"}, True),
                        ({"SPARK_FAST_SAFETENSORS": "0"}, False), ({"SPARK_DIRECT_IO": "0"}, False),
                        ({"SPARK_DIRECT_IO": ""}, False), ({"SPARK_DIRECT_AHEAD": "0"}, True)):
            for load in ("fast", "standard"):
                settings, problems = job_module.studio_settings({**env, "LFSTUDIO_UPSCALE_LOAD": load})
                self.assertEqual((settings["o_direct"], problems), (on, []), (env, load))
            with mock.patch.dict(os.environ, env):                      # what the job's fastload makes of them
                for name in job_module.FASTLOAD_SWITCHES:               # unset in `env`: the default, 1
                    os.environ[name] = env.get(name, "1")
                fastload = job_module.load_fastload()
                self.assertEqual(fastload.ENABLED and fastload.DIRECT, on, env)

    def test_preread_needs_the_standard_load(self):
        """parse_job refuses preread with the fast load; the settings say so at start, for the default load too."""
        for load in ({"LFSTUDIO_UPSCALE_LOAD": "fast"}, {}):
            _, problems = job_module.studio_settings({**load, "LFSTUDIO_UPSCALE_PREREAD": "1"})
            self.assertEqual(problems, ["LFSTUDIO_UPSCALE_PREREAD=1: only with LFSTUDIO_UPSCALE_LOAD=standard (the "
                                        "fast load reads the weight files past the page cache) (see bin/env.example)"])


class TextTests(unittest.TestCase):
    def test_t2va_conditioning(self):
        self.assertEqual(job_module.check_text(conditioning(41)), 41)
        self.assertEqual(job_module.check_text(conditioning(7, task=None)), 7)

    def test_fl2va_and_ref2va_conditioning_is_refused(self):
        tags = [1] * 6 + [0] * 254 + [1] * 369
        with self.assertRaisesRegex(JobError, "254 non-text rows"):
            job_module.check_text(conditioning(len(tags), tags=tags, task=None))
        for task in ("fl2va", "ref2va"):
            with self.assertRaisesRegex(JobError, f"{task}'s.*text-only"):
                job_module.check_text(conditioning(41, task=task))

    def test_shapes_and_keys(self):
        cond = conditioning(41)
        for shape in ((1, 42, 5120), (41, 5120), (1, 41, 4096), (2, 41, 5120)):
            cond["prompt_embeds"] = FakeTensor(shape)
            with self.subTest(shape=shape), self.assertRaisesRegex(JobError, "do not match"):
                job_module.check_text(cond)
        with self.assertRaisesRegex(JobError, "no rows"):
            job_module.check_text(conditioning(0, task=None))
        for key in ("prompt_embeds", "text_token_tags"):
            cond = conditioning(41)
            del cond[key]
            with self.assertRaisesRegex(JobError, "no prompt_embeds"):
                job_module.check_text(cond)
        with self.assertRaisesRegex(JobError, "no prompt_embeds"):
            job_module.check_text(None)

    def test_audio_shapes(self):
        self.assertEqual(job_module.AUDIO_SAMPLES, 165333)
        for samples in (161333, 165333, 1):
            job_module.check_pcm_shape((1, 2, samples))
        for shape in ((1, 2, 165334), (1, 2, 0), (2, 161333), (1, 1, 161333), (1, 2, 161333, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(JobError, "not stereo PCM"):
                job_module.check_pcm_shape(shape)


def nvidia_route(native, target, upscaled=(24, 37, 90, 160)):
    """NVIDIA's Pipeline.upscale (pipeline.py:223-228) as the grids it runs the upscaler to: through UPSCALED_SHAPE
    first when the target has more latent rows."""
    if tuple(native) == tuple(target):
        return []
    return ([tuple(upscaled)] if target[2] > upscaled[2] else []) + [tuple(target)]


def nvidia_package():
    """NVIDIA's real package where the studio's settings find it (LFSTUDIO_UPSCALE_PKG, else its default), or None."""
    package = job_module.studio_settings(os.environ)[0]["package"]
    return package if (package / "open_h3" / "pipeline.py").is_file() else None


LANDSCAPE_TARGETS = {"1080p": (24, 37, 68, 120), "2k": (24, 37, 90, 160), "4k": (24, 37, 136, 240)}
PORTRAIT_TARGETS = {"1080p": (24, 37, 120, 68), "2k": (24, 37, 160, 90), "4k": (24, 37, 240, 136)}


class ShapeTests(unittest.TestCase):
    """The shape table, the portrait entries for NVIDIA's dicts, the upscale route and the column crop, without
    torch."""

    def test_the_table(self):
        land, port = job_module.SHAPES["landscape"], job_module.SHAPES["portrait"]
        self.assertEqual(set(job_module.SHAPES), {"landscape", "portrait"})
        self.assertEqual(job_module.SIZES, land["sizes"])
        self.assertEqual(job_module.NATIVE_SHAPE, land["native"])
        for shape in (land, port):
            width, height = shape["input"]
            self.assertEqual(shape["native"], (24, 37, height // 16, width // 16))     # NVIDIA's NATIVE_SHAPES
            self.assertEqual(set(shape["sizes"]), {"1080p", "2k", "4k"})
            self.assertEqual(set(shape["resolutions"]), set(shape["sizes"]))
        self.assertEqual((land["input"], port["input"]), ((1344, 768), (768, 1344)))
        self.assertEqual({r: size[::-1] for r, size in land["sizes"].items()}, port["sizes"])
        self.assertEqual((land["size"], land["resolutions"]), ("768p", {r: r for r in job_module.SIZES}))
        self.assertEqual(port["sizes"], {"1080p": (1080, 1920), "2k": (1440, 2560), "4k": (2160, 3840)})
        self.assertEqual(job_module.output_size("portrait", "2k"), (1440, 2560))
        self.assertEqual(job_module.output_size("landscape", "4k"), (3840, 2160))

    def test_every_orientation_the_studio_makes(self):
        """The orientations of both patches (and the stub runtime's) are the job's SHAPES, so the server's
        UPSCALE_SHAPE refusal is a guard for an orientation added later: such a change fails here first."""
        import ast
        sources = {name: patch_new_file(f"models/minimax_h3/{package}/runtime/geometry.py", patch)[0]
                   for name, package, patch in (("spark", "Sol-H3-Spark", "lf-studio.patch"),
                                                ("rtx5090", "Sol-H3-RTX5090", "lf-studio-rtx5090.patch"))}
        sources["stub"] = (REPO / "tests" / "stub" / "runtime" / "geometry.py").read_text()
        for name, source in sources.items():
            found = [ast.literal_eval(node.value) for node in ast.parse(source).body if isinstance(node, ast.Assign)
                     and [getattr(t, "id", None) for t in node.targets] == ["ORIENTATIONS"]]
            self.assertEqual([set(value) for value in found], [set(job_module.SHAPES)], name)

    def test_the_portrait_entries_are_nvidias_turned(self):
        """INPUT_SIZES and NATIVE_SHAPES for encode_video, RESOLUTIONS for Pipeline: NVIDIA's landscape entries with
        width and height swapped, every row kept; after the column crop each gives the table's size."""
        land, port = job_module.SHAPES["landscape"], job_module.SHAPES["portrait"]
        nvidia, entries = job_module.NVIDIA_ENTRIES, job_module.PORTRAIT_ENTRIES
        self.assertEqual(set(entries), {"INPUT_SIZES", "NATIVE_SHAPES", "RESOLUTIONS"})
        self.assertEqual((nvidia["INPUT_SIZES"], nvidia["NATIVE_SHAPES"]),
                         ({"768p": land["input"]}, {"768p": land["native"]}))
        self.assertEqual((entries["INPUT_SIZES"], entries["NATIVE_SHAPES"]),
                         ({port["size"]: port["input"]}, {port["size"]: port["native"]}))
        self.assertEqual(set(entries["RESOLUTIONS"]), set(port["resolutions"].values()))
        for resolution, (width, height) in land["sizes"].items():
            computed, nvidia_width, keep = nvidia["RESOLUTIONS"][resolution]
            self.assertEqual((nvidia_width, keep, computed % 16), (width, height, 0))
            rows, columns, kept = entries["RESOLUTIONS"][port["resolutions"][resolution]]
            self.assertEqual((rows, columns, kept), (width, computed, width))
            self.assertEqual((24, 37, rows // 16, columns // 16), PORTRAIT_TARGETS[resolution])  # Pipeline's target
            self.assertEqual(port["sizes"][resolution], (min(columns, height), kept))     # after keep_columns

    def test_the_upscale_route(self):
        route = job_module.upscale_route
        land, port = job_module.NATIVE_SHAPE, job_module.SHAPES["portrait"]["native"]
        for resolution, target in LANDSCAPE_TARGETS.items():                    # 16:9: NVIDIA's own route
            self.assertEqual(route(land, target), nvidia_route(land, target), resolution)
        self.assertEqual(route(land, LANDSCAPE_TARGETS["4k"]), [(24, 37, 90, 160), (24, 37, 136, 240)])
        # 9:16: one pass to 1080p and 2K, through the 2K grid turned (160x90 latents) to 4K
        self.assertEqual(route(port, PORTRAIT_TARGETS["1080p"]), [(24, 37, 120, 68)])
        self.assertEqual(route(port, PORTRAIT_TARGETS["2k"]), [(24, 37, 160, 90)])
        self.assertEqual(route(port, PORTRAIT_TARGETS["4k"]), [(24, 37, 160, 90), (24, 37, 240, 136)])
        # NVIDIA's own test (pipeline.py:226) would send each of them through the landscape 2K grid and back
        for target in PORTRAIT_TARGETS.values():
            self.assertEqual(nvidia_route(port, target)[0], (24, 37, 90, 160))
        # no grid of a route has the other orientation
        for native, targets in ((land, LANDSCAPE_TARGETS), (port, PORTRAIT_TARGETS)):
            for target in targets.values():
                for grid in route(native, target):
                    self.assertEqual(grid[2] > grid[3], native[2] > native[3], (native, target, grid))
        self.assertEqual((route(land, land), route(port, port)), ([], []))
        for native, target in ((land, PORTRAIT_TARGETS["2k"]), (port, LANDSCAPE_TARGETS["1080p"])):
            with self.assertRaisesRegex(ValueError, "differ in orientation"):
                route(native, target)

    def test_the_column_crop(self):
        """The centre columns, as NVIDIA's decode keeps the centre rows ((rows - kept) // 2 on each side)."""
        for computed, width, left in ((1088, 1080, 4), (1440, 1440, 0), (2176, 2160, 8)):
            NARROWED.clear()
            frames = job_module.keep_columns(StandInTensor(121, 1920, computed, 3), width)
            self.assertEqual(frames.shape, (121, 1920, width, 3))
            self.assertEqual(NARROWED, [((121, 1920, computed, 3), 2, left, width)])


class PackageTests(unittest.TestCase):
    """check_package on stand-ins of NVIDIA's modules: the names and shapes for every clip, and for a 9:16 clip the
    pinned files, the entries its portrait ones are made from and no portrait key of NVIDIA's own."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="upscale-package-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "open_h3").mkdir()
        pins = {}
        for rel in job_module.NVIDIA_FILES:
            (self.tmp / rel).write_text(f"# {rel}\n")
            pins[rel] = hashlib.sha256((self.tmp / rel).read_bytes()).hexdigest()
        patcher = mock.patch.dict(job_module.NVIDIA_FILES, pins)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapters = types.SimpleNamespace(LORA_FILE=job_module.LORA_FILE, CURVE_FILE=job_module.CURVE_FILE,
                                              CURVE_TABLE=job_module.CURVE_TABLE, StageMerger=object)
        self.pipeline = types.SimpleNamespace(
            __file__=str(self.tmp / "open_h3" / "pipeline.py"), UPSCALER_FILE=job_module.UPSCALER_FILE,
            NATIVE_SHAPE=(24, 37, 48, 84), UPSCALED_SHAPE=(24, 37, 90, 160),
            INPUT_SIZES={"768p": (1344, 768), "576p": (1024, 576)},
            NATIVE_SHAPES={"768p": (24, 37, 48, 84), "576p": (24, 37, 36, 64)},
            RESOLUTIONS={"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080), "2k": (1440, 2560, 1440),
                         "4k": (2176, 3840, 2160)})

    def refused(self, resolution="2k", orientation="portrait"):
        with self.assertRaisesRegex(JobError, "not the one this wrapper was written for") as caught:
            job_module.check_package(self.adapters, self.pipeline, resolution, orientation)
        return str(caught.exception)

    def test_both_orientations(self):
        for resolution in job_module.SIZES:
            for orientation in job_module.SHAPES:
                job_module.check_package(self.adapters, self.pipeline, resolution, orientation)
        self.adapters.LORA_FILE += ".v2"
        for orientation in job_module.SHAPES:
            self.assertIn("LORA_FILE", self.refused(orientation=orientation))

    def test_landscape_does_not_read_the_pinned_files(self):
        (self.tmp / "open_h3" / "pipeline.py").write_text("# another version\n")
        self.pipeline.UPSCALED_SHAPE = (24, 37, 80, 160)
        self.pipeline.RESOLUTIONS["2k-portrait"] = (1, 2, 3)
        job_module.check_package(self.adapters, self.pipeline, "2k")
        job_module.check_package(self.adapters, self.pipeline, "1080p", "landscape")

    def test_portrait_refusals(self):
        (self.tmp / "open_h3" / "upscaler.py").write_text("# another version\n")
        self.assertIn("open_h3/upscaler.py", self.refused())
        (self.tmp / "open_h3" / "upscaler.py").write_text("# open_h3/upscaler.py\n")
        job_module.check_package(self.adapters, self.pipeline, "2k", "portrait")
        for name, change in (("UPSCALED_SHAPE", lambda p: setattr(p, "UPSCALED_SHAPE", (24, 37, 96, 160))),
                             ("RESOLUTIONS", lambda p: p.RESOLUTIONS.update({"1080p": (1080, 1920, 1080)})),
                             ("RESOLUTIONS", lambda p: p.RESOLUTIONS.update({"4k-portrait": (3840, 2176, 3840)})),
                             ("INPUT_SIZES", lambda p: p.INPUT_SIZES.pop("768p")),
                             ("NATIVE_SHAPES", lambda p: p.NATIVE_SHAPES.update({"768p-portrait": (24, 37, 84, 48)}))):
            with self.subTest(name=name):
                saved = {k: getattr(self.pipeline, k) for k in ("UPSCALED_SHAPE", "INPUT_SIZES", "NATIVE_SHAPES",
                                                                 "RESOLUTIONS")}
                for key in ("INPUT_SIZES", "NATIVE_SHAPES", "RESOLUTIONS"):
                    setattr(self.pipeline, key, dict(saved[key]))
                change(self.pipeline)
                self.assertIn(name, self.refused())             # NVIDIA's own portrait key too: not overwritten
                for key, value in saved.items():
                    setattr(self.pipeline, key, value)

    def test_a_pinned_file_that_is_not_there(self):
        """nvidia_files gives None for a file that is not there, so a 9:16 clip is refused (JobError, exit 2) with
        its name rather than failing on FileNotFoundError."""
        self.assertEqual(job_module.nvidia_files(self.tmp), job_module.NVIDIA_FILES)
        self.assertEqual(set(job_module.NVIDIA_FILES), {"open_h3/pipeline.py", "open_h3/upscaler.py",
                                                        "open_h3/upscaler_node.py"})
        (self.tmp / "open_h3" / "upscaler_node.py").unlink()
        self.assertIsNone(job_module.nvidia_files(str(self.tmp))["open_h3/upscaler_node.py"])
        self.assertIn("'open_h3/upscaler_node.py': None", self.refused())
        job_module.check_package(self.adapters, self.pipeline, "2k", "landscape")        # 16:9 does not check them

    def test_add_portrait_entries(self):
        job_module.add_portrait_entries(self.pipeline)
        self.assertEqual(self.pipeline.INPUT_SIZES, {"768p": (1344, 768), "576p": (1024, 576),
                                                     "768p-portrait": (768, 1344)})
        self.assertEqual(self.pipeline.NATIVE_SHAPES["768p-portrait"], (24, 37, 84, 48))
        self.assertEqual(self.pipeline.RESOLUTIONS["2k"], (1440, 2560, 1440))
        self.assertEqual({k: v for k, v in self.pipeline.RESOLUTIONS.items() if k.endswith("-portrait")},
                         job_module.PORTRAIT_ENTRIES["RESOLUTIONS"])


class NvidiaPackageTests(unittest.TestCase):
    """Against NVIDIA's real package where the studio's settings find it (LFSTUDIO_UPSCALE_PKG; skipped without it):
    the pinned files and the entries the job was written against, read from its pipeline.py without importing it."""

    def setUp(self):
        self.package = nvidia_package()
        if self.package is None:
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")

    def test_the_pinned_files(self):
        for rel, digest in job_module.NVIDIA_FILES.items():
            self.assertEqual(job_module.sha256(self.package / rel), digest, rel)
        self.assertEqual(job_module.nvidia_files(self.package), job_module.NVIDIA_FILES)     # bin/lf-studio's note
        # the node's sizes the portrait grid from the target's latent size and aligns it to 32 px
        source = (self.package / "open_h3" / "upscaler.py").read_text()
        self.assertIn('"width": target_shape[3] * 16, "height": target_shape[2] * 16}', source)
        self.assertIn("align=32", source)

    def test_the_entries_and_the_route(self):
        import ast
        source = (self.package / "open_h3" / "pipeline.py").read_text()
        found = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source).body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("NATIVE_SHAPE", "INPUT_SIZES", "UPSCALED_SHAPE", "RESOLUTIONS")}
        self.assertEqual((found["NATIVE_SHAPE"], found["UPSCALED_SHAPE"]),
                         (job_module.NATIVE_SHAPE, job_module.UPSCALED_SHAPE))
        for name, table in (("INPUT_SIZES", found["INPUT_SIZES"]), ("RESOLUTIONS", found["RESOLUTIONS"])):
            for key, value in job_module.NVIDIA_ENTRIES[name].items():
                self.assertEqual(table[key], value, (name, key))
            self.assertFalse(set(table) & set(job_module.PORTRAIT_ENTRIES[name]), name)
        self.assertIn("NATIVE_SHAPES = {name: (24, 37, h // 16, w // 16) for name, (w, h) in INPUT_SIZES.items()}",
                      source)
        # the route upscale_route mirrors (and the test it turns for 9:16)
        self.assertIn("        if target[2] > UPSCALED_SHAPE[2]:\n            native = upscaler.upscale(self.node, "
                      "native, UPSCALED_SHAPE)\n        return upscaler.upscale(self.node, native, target)"
                      ".to(self.device)\n", source)


class InputTests(Fixture):
    def test_the_files_each_job_reads(self):
        files = job_module.case_inputs(job_module.parse_job(self.job(audio="silence"), "/"))
        self.assertEqual(set(files), {"conditioning", "final"})
        self.assertEqual(files["final"], self.case / "stage2" / "refined_1344x768_121f.mp4")
        files = job_module.case_inputs(job_module.parse_job(self.job(audio="clip"), "/"))
        self.assertEqual(set(files), {"conditioning", "final", "pcm"})
        self.assertEqual(files["pcm"], self.case / "stage1" / "stage1_direct_tensors.pt")

    def test_missing_files_are_refused(self):
        for rel, changes, message in (("stage1/stage1_direct_tensors.pt", {"audio": "clip"}, "Stage-1 audio"),
                                      ("qwen/conditioning.pt", {}, "prompt conditioning")):
            with self.subTest(rel=rel):
                moved = self.tmp / "moved"
                os.replace(self.case / rel, moved)
                try:
                    with self.assertRaisesRegex(JobError, message):
                        job_module.case_inputs(job_module.parse_job(self.job(**changes), "/"))
                finally:
                    os.replace(moved, self.case / rel)
        shutil.copy(self.final, self.case / "stage2" / "refined_1344x768_121f_b.mp4")
        with self.assertRaisesRegex(JobError, "found 2"):
            job_module.case_inputs(job_module.parse_job(self.job(), "/"))
        with self.assertRaisesRegex(JobError, "no clip directory"):
            job_module.case_inputs(job_module.parse_job(self.job(case=str(self.tmp / "nope")), "/"))

    def test_5s_clips_of_either_orientation_pass(self):
        """1344x768 (16:9) and 768x1344 (9:16), 121 frames at 24 fps (NVIDIA's length) or 124 (every frame H3 makes:
        a LightX2V Turbo clip with a last keyframe); check_final says which. Any other size, length or rate is
        refused with what it is."""
        self.assertEqual(job_module.check_final(self.case / "stage2" / "refined_1344x768_121f.mp4"),
                         ("landscape", 121))
        self.assertEqual(job_module.check_final(self.portrait), ("portrait", 121))
        self.assertEqual(job_module.check_final(self.final124), ("landscape", 124))
        self.assertEqual(job_module.check_final(make_mp4(self.tmp / "whole9x16.mp4", width=768, height=1344,
                                                         frames=124, audio=False)), ("portrait", 124))
        for name, kwargs, message in (("ten.mp4", {"frames": 241}, "1344x768, 241 frames"),
                                      ("ten9x16.mp4", {"frames": 241, "width": 768, "height": 1344},
                                       "768x1344, 241 frames"),
                                      ("three.mp4", {"frames": 73, "width": 768, "height": 1344}, "73 frames"),
                                      ("122.mp4", {"frames": 122}, "122 frames"),
                                      ("120.mp4", {"frames": 120}, "120 frames"),
                                      ("fps.mp4", {"fps": 25}, "25/1"),
                                      ("fps124.mp4", {"fps": 25, "frames": 124}, "124 frames at 25/1"),
                                      ("576p.mp4", {"width": 1024, "height": 576}, "1024x576"),
                                      ("576p124.mp4", {"width": 1024, "height": 576, "frames": 124}, "1024x576"),
                                      ("1080p9x16.mp4", {"width": 1080, "height": 1920}, "1080x1920")):
            with self.subTest(name=name), self.assertRaisesRegex(
                    JobError, message + r".*: not a 5 s 16:9 or 9:16 clip \(1344x768 or 768x1344, 121 or 124 frames "
                                        r"at 24 fps\)$"):
                job_module.check_final(make_mp4(self.tmp / name, audio=False, **kwargs))
        (self.tmp / "text.mp4").write_text("not a video")
        with self.assertRaisesRegex(JobError, "ffprobe cannot read"):
            job_module.check_final(self.tmp / "text.mp4")
        self.assertEqual(job_module.probe_command("/x.mp4", "ffprobe")[-1], "/x.mp4")

    def test_weights(self):
        job = job_module.parse_job(self.job(), "/")
        job_module.check_weights(job)
        with self.assertRaisesRegex(JobError, "only .safetensors"):
            job_module.check_weights(job, "latent_upsampler/minimax_h3_latent_upscaler.pt")
        for path, message in ((self.weights / job_module.LORA_FILE, "missing weights"),
                              (self.weights / job_module.CURVE_TABLE, "missing weights"),
                              (self.base / "vae" / "diffusion_pytorch_model.safetensors", "no .safetensors in"),
                              (self.base / "transformer" / "config.json", "missing weights"),
                              (self.package / "open_h3" / "pipeline.py", "paths.package")):
            with self.subTest(path=path.name):
                moved = self.tmp / "moved"
                os.replace(path, moved)
                try:
                    with self.assertRaisesRegex(JobError, message):
                        job_module.check_weights(job)
                finally:
                    os.replace(moved, path)
        with self.assertRaisesRegex(JobError, "paths.output"):
            job_module.check_weights(job_module.parse_job(self.job(paths={**self.job()["paths"],
                                                                          "output": str(self.tmp / "no")}), "/"))

    def test_the_fast_load_needs_its_loader(self):
        """upscale/fastload.py next to the job, with the fast load (also when the job has no "load", as bin/lf-studio
        once passed it); the standard load does without it."""
        missing = self.tmp / "upscale" / "fastload.py"
        for job in (job_module.parse_job(self.job(), "/"), {k: v for k, v in job_module.parse_job(self.job(), "/")
                                                             .items() if k != "load"}):
            job_module.check_weights(job)                               # the real one, next to the job
            with self.assertRaisesRegex(JobError, rf"^no {re.escape(str(missing))}, the fast load's loader \(part of "
                                        r"LF Studio; the standard load, LFSTUDIO_UPSCALE_LOAD=standard, does without "
                                        r"it\)$"):
                job_module.check_weights(job, fastload=missing)
        job_module.check_weights(job_module.parse_job(self.job(load="standard"), "/"), fastload=missing)

    def test_weight_lists_and_the_page_cache(self):
        job = job_module.parse_job(self.job(), "/")
        names = [p.relative_to(self.tmp).parts[:2] for p in job_module.preread_files(job)]
        self.assertEqual(names, [("weights", "lora"), ("weights", "lora"), ("MiniMax-H3", "audio_vae"),
                                 ("MiniMax-H3", "transformer")])
        loaded = job_module.loaded_files(job)
        self.assertEqual({p.parent.name for p in loaded},
                         {"transformer", "vae", "audio_vae", "lora", "latent_upsampler"})
        self.assertTrue(all(p.suffix == ".safetensors" for p in loaded))
        self.assertEqual(job_module.preread(loaded, workers=3, chunk=512), sum(p.stat().st_size for p in loaded))
        self.assertEqual(job_module.drop_from_page_cache(loaded + [self.tmp / "missing"]), len(loaded))

    def test_plan_uses_the_package_sigmas(self):
        namespace = {}
        exec(STAND_IN_SIGMAS, namespace)
        schedules, grid = namespace["SCHEDULES"], namespace["shifted_grid"]
        five = job_module.frame_case(121)
        job = dict(job_module.parse_job(self.job(steps=3, start_sigma=None), "/"), orientation="landscape",
                   frames=five)
        self.assertEqual(job_module.plan(job, schedules, grid),
                         {"orientation": "landscape", "size": [2560, 1440],
                          "frames": {"kept": 121, "made": 124, "pad": 3, "T": 37}, "steps": 3,
                          "sigmas": [0.9, 0.8, 0.6, 0.0], "lean_chunks": None})
        job = dict(job_module.parse_job(self.job(resolution="4k", steps=2, start_sigma=0.5), "/"),
                   orientation="landscape", frames=job_module.frame_case(124))
        planned = job_module.plan(job, schedules, grid)
        self.assertEqual((planned["sigmas"], planned["size"], planned["lean_chunks"], planned["frames"]),
                         ([0.5, 0.25, 0.0], [3840, 2160], 16, {"kept": 124, "made": 124, "pad": 0, "T": 37}))
        for resolution, size, lean in (("1080p", [1080, 1920], None), ("2k", [1440, 2560], None),
                                       ("4k", [2160, 3840], 16)):         # 9:16: the same tokens, the same forward
            job = dict(job_module.parse_job(self.job(resolution=resolution), "/"), orientation="portrait",
                       frames=five)
            planned = job_module.plan(job, schedules, grid)
            self.assertEqual((planned["orientation"], planned["size"], planned["lean_chunks"]),
                             ("portrait", size, lean))
        with self.assertRaisesRegex(JobError, "no 3-step schedule"):
            job_module.plan(dict(job_module.parse_job(self.job(steps=3, start_sigma=None), "/"),
                                 orientation="landscape", frames=five), {2: [0.7, 0.4, 0]}, grid)


PIL_NUMPY = importlib.util.find_spec("PIL") is not None and importlib.util.find_spec("numpy") is not None


def picture(width, height, seed):
    """A smooth picture of its own per seed (sums of sines in each channel): (height, width, 3) uint8."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    channels = [sum(np.sin(2 * np.pi * (fx * x + fy * y) + phase) for fx, fy, phase in
                    rng.uniform((1, 1, 0), (5, 5, 6.28), (3, 3))) for _ in range(3)]
    return np.clip(np.stack(channels, -1) * 35 + 128, 0, 255).astype(np.uint8)


def on_canvas(array, canvas, fit):
    """A picture as the clip's model put it on its canvas (diffusers 0.40: as is, stretched, or cover-cropped with
    H3's rounding), then on the 768p case view the job reads (the server's LANCZOS stretch): a PIL image."""
    from PIL import Image
    image = Image.fromarray(array)
    if fit == "stretch":
        image = image.resize(canvas, Image.Resampling.LANCZOS)
    elif fit == "cover-crop":
        resized, crop = job_module.cover_crop(image.size, canvas)
        image = image.resize(resized, Image.Resampling.LANCZOS).crop(crop)
    size = (1344, 768) if canvas[0] > canvas[1] else (768, 1344)
    return image.resize(size, Image.Resampling.LANCZOS) if image.size != size else image


def frames_mp4(path, frames):
    """An MP4 (H.264, CRF 12, 24 fps) of `frames`, PIL images of one size, in order."""
    width, height = frames[0].size
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                    f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast", "-crf",
                    "12", "-pix_fmt", "yuv420p", str(path)], input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@unittest.skipUnless(PIL_NUMPY, "needs Pillow and numpy (the Regen-2K environment has both)")
class PinCheckTests(Fixture):
    """check_pin on ffmpeg-made clips whose first and last frames show pictures as a LightX2V Turbo clip's do (fitted
    to its 1024x576 canvas, stretched to the 768p case view): the ends it keeps, the ones it does not and why, the
    EXIF turn, the fit at the grid, and --check."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from PIL import Image
        cls.first, cls.last, cls.other = picture(1344, 768, 1), picture(1344, 800, 2), picture(1344, 768, 3)
        canvas = (1024, 576)
        start, end = on_canvas(cls.first, canvas, "stretch"), on_canvas(cls.last, canvas, "cover-crop")
        middle = Image.fromarray(picture(1344, 768, 4))
        cls.whole = frames_mp4(cls.media / "keyframes124.mp4", [start] + [middle] * 122 + [end])
        cls.cut = frames_mp4(cls.media / "keyframes121.mp4", [start] + [middle] * 119 + [end])

    def setUp(self):
        super().setUp()
        from PIL import Image
        self.pictures = self.tmp / "input" / "keyframe"
        self.pictures.mkdir(parents=True)
        for name, array in (("first", self.first), ("last", self.last), ("other", self.other)):
            Image.fromarray(array).save(self.pictures / f"{name}.png")

    def keyframe(self, anchor, name=None, fit=None, canvas=(1024, 576)):
        path = self.pictures / f"{name or anchor}.png"
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else SHA
        return {"anchor": anchor, "image": str(path), "sha256": digest,
                "fit": fit or ("stretch" if anchor == "first" else "cover-crop"), "canvas": list(canvas)}

    def check(self, *keyframes, clip=None, kept=124, orientation="landscape", **changes):
        job = {"pin": job_module.parse_pin({"keyframes": list(keyframes), **changes}),
               "frames": job_module.frame_case(kept), "orientation": orientation}
        return job_module.check_pin(job, clip or (self.whole if kept == 124 else self.cut))

    def test_both_ends_kept(self):
        """A clip that starts and ends on its pictures: both ends used, well over 28 dB at 672x384, frames 0 and 123;
        the last end's box is the cover-crop's in the picture, the first's None (stretched)."""
        results = self.check(self.keyframe("first"), self.keyframe("last"))
        self.assertEqual(list(results), ["first", "last"])
        first, last = results["first"], results["last"]
        self.assertEqual((first["used"], first["frame"], first["fit"], first["box"], first["size"], first["why"]),
                         (True, 0, "stretch", None, [1344, 768], None))
        self.assertEqual((last["used"], last["frame"], last["fit"], last["size"], last["why"]),
                         (True, 123, "cover-crop", [1344, 800], None))
        self.assertEqual(last["box"], [round(v, 2) for v in job_module.fit_box((1344, 800), (1024, 576),
                                                                               "cover-crop")])
        for end in (first, last):
            self.assertGreater(end["agreement_db"], 35)

    def test_ends_it_does_not_keep(self):
        """Each reason an end is not used (the check never refuses): the clip's frame far from the picture (another
        picture, or the min_db above it), the picture gone, changed or not a picture, a last end the clip stops 3
        frames short of (121 kept of 124 made: a Sol-H3 clip, or a Turbo clip cut to 121)."""
        from PIL import Image
        far = self.check(self.keyframe("first", "other"))["first"]
        self.assertFalse(far["used"])
        self.assertLess(far["agreement_db"], 28)
        self.assertEqual(far["why"],
                         f"the clip's frame 0 is {far['agreement_db']:.1f} dB from your picture (28 needed)")
        high = self.check(self.keyframe("first"), self.keyframe("last"), check={"min_db": 60})
        self.assertEqual([end["used"] for end in high.values()], [False, False])
        self.assertRegex(high["last"]["why"], r"^the clip's frame 123 is \d+\.\d dB from your picture \(60 needed\)$")
        gone = self.keyframe("first")
        (self.pictures / "first.png").unlink()
        self.assertEqual(self.check(gone)["first"], {"used": False, "agreement_db": None, "frame": 0,
                                                     "fit": "stretch", "box": None, "size": None,
                                                     "why": "your picture is no longer on disk"})
        changed = self.keyframe("last")
        (self.pictures / "last.png").write_bytes((self.pictures / "other.png").read_bytes())
        self.assertEqual(self.check(changed)["last"]["why"], "your picture changed since the clip was made")
        (self.pictures / "first.png").write_bytes(b"not a picture")
        self.assertEqual(self.check(self.keyframe("first"))["first"]["why"], "your picture does not open")
        Image.fromarray(self.last).save(self.pictures / "last.png")
        cut = self.check(self.keyframe("first", "other"), self.keyframe("last"), kept=121)
        self.assertEqual((cut["last"]["used"], cut["last"]["frame"], cut["last"]["why"]),
                         (False, 120, "the clip stops 3 frames before your last picture"))
        self.assertIsNotNone(cut["last"]["agreement_db"])               # frame 120 against it, for the record
        self.assertFalse(cut["first"]["used"])

    def test_the_exif_turn(self):
        """A picture stored turned with an EXIF orientation (6 and 8, as phones write them) is checked upright: the
        same answer as the upright file, its size the turned one, the box in the turned picture."""
        from PIL import Image
        upright = self.check(self.keyframe("first"), self.keyframe("last"))
        for orientation, stored in ((6, Image.Transpose.ROTATE_90), (8, Image.Transpose.ROTATE_270)):
            with self.subTest(orientation=orientation):
                for name, array in (("first", self.first), ("last", self.last)):
                    image = Image.fromarray(array).transpose(stored)
                    exif = Image.Exif()
                    exif[0x0112] = orientation
                    image.save(self.pictures / f"{name}.png", exif=exif)
                    self.assertEqual(job_module.open_keyframe((self.pictures / f"{name}.png").read_bytes()).size,
                                     (array.shape[1], array.shape[0]))
                turned = self.check(self.keyframe("first"), self.keyframe("last"))
                self.assertEqual(turned, upright)

    def test_the_picture_on_the_grid(self):
        """keyframe_image: with no box the whole picture stretched (LANCZOS) as the bench wrapper's encode_keyframe;
        one of the grid's size as it is; with a cover-crop one LANCZOS step from the box in the picture (the Turbo
        receipt's crop mapped back), not the canvas's crop upscaled."""
        from PIL import Image
        image = Image.fromarray(self.first)
        for grid in ((1920, 1088), (2560, 1440), (672, 384)):
            for fit in ("stretch", "as is"):
                self.assertEqual(job_module.keyframe_image(image, grid, fit, (1024, 576)).tobytes(),
                                 image.resize(grid, Image.Resampling.LANCZOS).tobytes())
        self.assertIs(job_module.keyframe_image(image, (1344, 768), "as is", (1344, 768)), image)
        # the 576p receipt's last keyframe: resized to 1024x585, cut at (0, 4, 1024, 580)
        box = (0, 4 * 768 / 585, 1344, 580 * 768 / 585)
        got = job_module.keyframe_image(image, (1920, 1088), "cover-crop", (1024, 576))
        self.assertEqual(got.tobytes(), image.resize((1920, 1088), Image.Resampling.LANCZOS, box=box).tobytes())
        two_steps = image.resize((1024, 585), Image.Resampling.LANCZOS).crop((0, 4, 1024, 580)).resize(
            (1920, 1088), Image.Resampling.LANCZOS)
        self.assertNotEqual(got.tobytes(), two_steps.tobytes())
        import numpy as np
        self.assertGreater(job_module.luma_psnr(np.asarray(got), np.asarray(two_steps)), 35)   # the same part of it
        self.assertEqual(job_module.luma_psnr(self.first, self.first), 99.0)

    def test_the_clips_frame(self):
        """clip_frame gives frame N of the clip, the same pixels as NVIDIA's read (here without tpad) at that size."""
        import numpy as np
        mp4 = moving_mp4(self.tmp / "moving.mp4", 124, "1344x768")
        raw = subprocess.run(job_module.clip_read_command(mp4, (672, 384), 124, 0, "ffmpeg"), capture_output=True,
                             check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(124, 384, 672, 3)
        for index in (0, 1, 120, 123):
            self.assertTrue(np.array_equal(job_module.clip_frame(mp4, (672, 384), index), frames[index]), index)
        self.assertIn("select=eq(n\\,123),scale=672:384:", job_module.frame_command(mp4, (672, 384), 123)[7])
        with self.assertRaisesRegex(JobError, r"^cannot read frame 124 of moving\.mp4 at 672x384"):
            job_module.clip_frame(mp4, (672, 384), 124)

    def test_check_prints_each_end_and_writes_nothing(self):
        """--check with a pin (in the Regen-2K environment): each end's answer in the plan, nothing written."""
        for mp4 in (self.case / "stage2").glob("refined_*.mp4"):
            mp4.unlink()
        shutil.copy(self.whole, self.case / "stage2" / "refined_1344x768_124f.mp4")
        keyframes = [self.keyframe("first"), self.keyframe("last", "other")]
        result = self.script("--job", str(self.write_job(pin={"keyframes": keyframes})), "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        planned = json.loads(result.stdout)
        self.assertEqual((planned["pin"]["first"]["used"], planned["pin"]["first"]["frame"],
                          planned["pin"]["last"]["used"], planned["pin"]["last"]["frame"]), (True, 0, False, 123))
        self.assertRegex(planned["pin"]["last"]["why"], r"^the clip's frame 123 is \d+\.\d dB from your picture")
        self.assertEqual(planned["frames"], {"kept": 124, "made": 124, "pad": 0, "T": 37})
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])


class LineAndCommandTests(unittest.TestCase):
    def test_progress_lines(self):
        for name in ("encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"):
            self.assertEqual(job_module.parse_line(job_module.phase_line(name) + "\n"), ("phase", name))
        line = job_module.error_line("RuntimeError: out of\nmemory\n  at step 2")
        self.assertEqual(line, "[lf-studio] error RuntimeError: out of memory at step 2")
        self.assertEqual(job_module.parse_line(line), ("error", "RuntimeError: out of memory at step 2"))
        self.assertLessEqual(len(job_module.error_line("x" * 5000)), 520)
        for other in ("[lf-studio] 01:02:03 phase loading", "[lf-studio] phase ", "phase loading",
                      "[MinimaxH3-3D] Loaded upscale model: upscaler", "", "[lf-studio] phaseloading"):
            self.assertIsNone(job_module.parse_line(other), other)

    def test_worker_command(self):
        self.assertEqual(job_module.worker_command(Path("/env/bin/python"), SCRIPT, Path("/out/job.json")),
                         ["/env/bin/python", "-I", "-B", str(SCRIPT), "--job", "/out/job.json"])
        self.assertEqual(job_module.worker_command(Path("/env/bin/python"), SCRIPT, Path("/out/job.json"), 4321)[-2:],
                         ["--parent", "4321"])

    def test_worker_environment(self):
        env = job_module.worker_environment({"HF_TOKEN": "hf_x", "HUGGING_FACE_HUB_TOKEN": "hf_y", "PATH": "/bin",
                                             "HF_HUB_OFFLINE": "0", "H3_ATTN_BACKEND": "_flash_3_hub",
                                             "PYTHONPATH": "/x", "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"})
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "H3_ATTN_BACKEND", "PYTHONPATH"):
            self.assertNotIn(key, env)
        self.assertEqual((env["PATH"], env["HF_HUB_OFFLINE"], env["TRANSFORMERS_OFFLINE"], env["HF_TOKEN_PATH"]),
                         ("/bin", "1", "1", "/nonexistent"))
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "backend:cudaMallocAsync")
        self.assertEqual(job_module.worker_environment({})["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(job_module.worker_environment({})["MIMALLOC_PURGE_DELAY"], "0")
        self.assertEqual(job_module.worker_environment({"IMAGEIO_FFMPEG_EXE": "/opt/ff"})["IMAGEIO_FFMPEG_EXE"],
                         "/opt/ff")
        environ = {"HF_TOKEN": "hf_x", "KEEP": "1"}
        job_module.apply_environment(environ)
        self.assertEqual((environ.get("HF_TOKEN"), environ["KEEP"], environ["HF_HUB_OFFLINE"]), (None, "1", "1"))

    def test_mux_command_copies_the_parent_audio(self):
        tmp = Path(tempfile.mkdtemp(prefix="upscale-mux-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        silent = make_mp4(tmp / "video_noaudio.mp4", width=256, height=144, audio=False)
        parent = make_mp4(tmp / "parent.mp4", width=128, height=72)
        out = tmp / "regen_256x144_121f.mp4"
        cmd = job_module.mux_command(silent, parent, out)
        self.assertEqual(cmd[:2], ["ffmpeg", "-nostdin"])
        subprocess.run(cmd, check=True)
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,nb_frames",
             "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)["streams"]
        self.assertEqual([(s["codec_type"], s["codec_name"]) for s in probe], [("video", "h264"), ("audio", "aac")])
        self.assertEqual((probe[0]["width"], probe[0]["height"], probe[0]["nb_frames"]), (256, 144, "121"))

        def audio_packets(path):
            return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                   "packet=size", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                                  check=True).stdout.split()
        self.assertEqual(audio_packets(out), audio_packets(parent))
        self.assertGreater(len(audio_packets(out)), 150)
        self.assertNotIn("-shortest", cmd)

    def test_write_json_is_atomic(self):
        tmp = Path(tempfile.mkdtemp(prefix="upscale-json-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        job_module.write_json(tmp / "receipt.json", {"state": "running"})
        job_module.write_json(tmp / "receipt.json", {"state": "done"})
        self.assertEqual(json.loads((tmp / "receipt.json").read_text()), {"state": "done"})
        self.assertEqual([p.name for p in tmp.iterdir()], ["receipt.json"])


class MemTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="upscale-mem-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = self.tmp / "meminfo"
        self.set(40)

    def set(self, gib):
        """A fake /proc/meminfo, replaced whole (the sampler thread must never read half of it)."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(f"MemTotal:       127535648 kB\nMemFree:        {gib * 1048576 // 2} kB\n"
                       f"MemAvailable:   {gib * 1048576} kB\nCached:          1048576 kB\n")
        os.replace(tmp, self.path)

    def test_meminfo(self):
        self.assertEqual(job_module.meminfo(self.path), {"MemFree": 20.0, "MemAvailable": 40.0, "Cached": 1.0})
        self.assertIn("MemAvailable", job_module.meminfo())

    def test_lowest_per_label(self):
        trace = job_module.MemTrace(self.path, interval=0.01)
        trace.start()
        trace.mark("load")
        self.set(15)
        deadline = time.time() + 5
        while trace.low.get("load", 99) > 15 and time.time() < deadline:
            time.sleep(0.02)
        self.set(30)
        trace.mark("refine 1/2")
        trace.stop()
        self.assertFalse(trace.is_alive())
        self.assertEqual((trace.low["load"], trace.low["refine 1/2"]), (15.0, 30.0))


class ScriptTests(Fixture):
    def test_importing_it_needs_no_torch(self):
        probe = (f"import importlib.util, sys; spec = importlib.util.spec_from_file_location('j', {str(SCRIPT)!r}); "
                 "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); f = m.load_fastload(); "
                 "print(f.__file__, sorted({k.split('.')[0] for k in sys.modules} & {'torch', 'numpy', 'diffusers', "
                 "'open_h3', 'safetensors', 'fastload'}))")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), f"{FASTLOAD} []")        # upscale/fastload.py, by its path
        self.assertIn("--check", self.script("--help", check=True).stdout)

    def test_check_prints_the_plan_and_writes_nothing(self):
        path = self.write_job()
        result = self.script("--job", str(path), "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        planned = json.loads(result.stdout)
        self.assertEqual((planned["ok"], planned["sigmas"], planned["size"], planned["steps"], planned["lean_chunks"]),
                         (True, [0.7, 0.35, 0.0], [2560, 1440], 2, None))
        self.assertEqual(set(planned["inputs"]), {"conditioning", "final", "pcm"})
        self.assertEqual((planned["job"]["meta"]["strength"], planned["job"]["load"]), ("balanced", "fast"))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])
        path = self.write_job(resolution="4k", steps=3, start_sigma=None, audio="silence")
        planned = json.loads(self.script("--job", str(path), "--check", check=True).stdout)
        self.assertEqual((planned["sigmas"], planned["size"], planned["lean_chunks"]),
                         ([0.9, 0.8, 0.6, 0.0], [3840, 2160], 16))
        self.assertEqual(set(planned["inputs"]), {"conditioning", "final"})

    def test_check_of_a_9x16_clip(self):
        (self.case / "stage2" / "refined_1344x768_121f.mp4").unlink()
        shutil.copy(self.portrait, self.case / "stage2" / "refined_768x1344_121f.mp4")
        planned = json.loads(self.script("--job", str(self.write_job()), "--check", check=True).stdout)
        self.assertEqual((planned["orientation"], planned["job"]["orientation"], planned["size"], planned["frames"]),
                         ("portrait", "portrait", [1440, 2560], {"kept": 121, "made": 124, "pad": 3, "T": 37}))
        planned = json.loads(self.script("--job", str(self.write_job(resolution="1080p")), "--check",
                                          check=True).stdout)
        self.assertEqual(planned["size"], [1080, 1920])

    def test_check_of_a_124_frame_clip(self):
        """Every frame H3 makes (a LightX2V Turbo clip with a last keyframe): taken whole, no copies added."""
        (self.case / "stage2" / "refined_1344x768_121f.mp4").unlink()
        shutil.copy(self.final124, self.case / "stage2" / "refined_1344x768_124f.mp4")
        planned = json.loads(self.script("--job", str(self.write_job(resolution="1080p")), "--check",
                                          check=True).stdout)
        self.assertEqual((planned["orientation"], planned["size"], planned["frames"]),
                         ("landscape", [1920, 1080], {"kept": 124, "made": 124, "pad": 0, "T": 37}))

    def test_a_pin_without_pillow_is_refused(self):
        """The pin's check needs Pillow and numpy, which the Regen-2K environment has: elsewhere a job with a pin is
        refused (exit 2), before anything loads, with a receipt."""
        if PIL_NUMPY:
            self.skipTest("Pillow and numpy are here (PinCheckTests checks the pin)")
        result = self.script("--job", str(self.write_job(pin=pin("first"))), "--check")
        self.assertEqual((result.returncode, result.stdout.splitlines()[-1]),
                         (2, "[lf-studio] error the pin's check needs Pillow and numpy (the Upscale's environment has "
                             "both)"))

    def test_check_refuses(self):
        make_mp4(self.case / "stage2" / "refined_768x1344_241f.mp4", width=768, height=1344, frames=241)
        (self.case / "stage2" / "refined_1344x768_121f.mp4").unlink()
        result = self.script("--job", str(self.write_job()), "--check")
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout.splitlines()[-1], r"^\[lf-studio\] error refined_768x1344_241f\.mp4 is "
                                                         r"768x1344, 241 frames .* not a 5 s 16:9 or 9:16 clip")
        self.assertFalse((self.out / "receipt.json").exists())
        (self.package / "open_h3" / "sigmas.py").unlink()
        make_mp4(self.case / "stage2" / "refined_1344x768_121f.mp4")
        (self.case / "stage2" / "refined_768x1344_241f.mp4").unlink()
        result = self.script("--job", str(self.write_job()), "--check")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot import open_h3.sigmas", result.stdout)
        result = self.script("--job", str(self.write_job(source="draft")), "--check")     # no draft source any more
        self.assertEqual((result.returncode, result.stdout.splitlines()[-1]),
                         (2, "[lf-studio] error unknown field(s): source"))

    def test_a_refused_job_leaves_a_failed_receipt(self):
        path = self.write_job(resolution="768p")
        result = self.script("--job", str(path))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1])[0], "error")
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("failed", 2))
        self.assertIn("resolution", receipt["error"])
        path.write_text("{")
        self.assertEqual(self.script("--job", str(path)).returncode, 2)
        self.assertIn("not JSON", json.loads((self.out / "receipt.json").read_text())["error"])
        (self.case / "qwen" / "conditioning.pt").unlink()
        self.assertEqual(self.script("--job", str(self.write_job())).returncode, 2)
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertIn("prompt conditioning", receipt["error"])
        self.assertEqual(receipt["job"]["case"], str(self.case))

    def test_a_run_that_cannot_import_its_parts_fails_with_a_receipt(self):
        """Without torch (or, with torch, without NVIDIA's real package) the run fails before the GPU is touched."""
        result = self.script("--job", str(self.write_job()))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        receipt = json.loads((self.out / "receipt.json").read_text())
        self.assertEqual((receipt["state"], receipt["exit_status"]), ("failed", 1))
        self.assertRegex(receipt["error"], "^(ModuleNotFoundError|ImportError): ")
        self.assertEqual(receipt["inputs_sha256"]["conditioning"], hashlib.sha256(b"conditioning").hexdigest())
        self.assertEqual(receipt["inputs_sha256"]["final"],
                         hashlib.sha256((self.case / "stage2" / "refined_1344x768_121f.mp4").read_bytes()).hexdigest())
        self.assertEqual(set(receipt["inputs_sha256"]), {"conditioning", "final", "pcm"})
        self.assertEqual(receipt["job"]["meta"]["parent"], "a" * 32)
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1]), ("error", receipt["error"][:500]))
        self.assertFalse((self.out / "video_noaudio.mp4").exists())


class MainTests(Fixture):
    """main() around a stand-in run(): the receipt for done, failed and SIGTERM."""

    def main(self, run):
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        stdout = io.StringIO()
        with mock.patch.object(job_module, "run", run), mock.patch.object(job_module, "apply_environment") as env, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = job_module.main(["--job", str(self.write_job())])
        env.assert_called_once_with(os.environ)
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        receipt = json.loads((self.out / "receipt.json").read_text())
        last = job_module.parse_line(stdout.getvalue().splitlines()[-1])
        self.assertEqual(last, ("error", receipt["error"][:500]) if status else None)
        return status, receipt

    def test_done(self):
        def run(job, files, receipt, trace):
            self.assertEqual(json.loads((self.out / "receipt.json").read_text())["state"], "running")
            receipt["peak_gib"].update(load={"allocated": 75.0, "reserved": 76.0},
                                       refine={"allocated": 91.2, "reserved": 93.0})
            receipt.update(stage="mp4", phase="saving")
            trace.mark("refine 1/2")
            (job["output"] / "video_noaudio.mp4").write_bytes(b"video")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"], receipt["exit_status"]), (0, "done", 0))
        self.assertEqual(receipt["frames"], {"kept": 121, "made": 124, "pad": 3, "T": 37})     # always, from the MP4
        self.assertNotIn("pin", receipt)                                                        # only with one
        self.assertEqual(receipt["peak_allocated_gib"], 91.2)
        self.assertIn("refine 1/2", receipt["lowest_mem_available_gib"])
        self.assertIn("all", receipt["lowest_mem_available_gib"])
        self.assertNotIn("stage", receipt)
        self.assertNotIn("phase", receipt)

    def test_failed(self):
        def run(job, files, receipt, trace):
            receipt.update(stage="vae_decode", phase="decoding")
            raise RuntimeError("CUDA out of memory.\nTried to allocate 35.00 GiB")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"], receipt["exit_status"]), (1, "failed", 1))
        self.assertEqual(receipt["error"], "RuntimeError: CUDA out of memory. Tried to allocate 35.00 GiB")
        self.assertEqual((receipt["stage"], receipt["phase"]), ("vae_decode", "decoding"))
        self.assertTrue(any(line.startswith("RuntimeError: CUDA out of memory.") for line in receipt["traceback"]))

    def test_refused_inside_the_run(self):
        def run(job, files, receipt, trace):
            raise JobError("the conditioning has 254 non-text rows (fl2va/ref2va): it needs a text-only encode")
        status, receipt = self.main(run)
        self.assertEqual((status, receipt["state"]), (2, "failed"))

    def test_sigterm_is_a_cancel(self):
        def run(job, files, receipt, trace):
            receipt.update(stage="refine", phase="refine 1/2")
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(10)
            self.fail("SIGTERM did not stop the run")
        started = time.time()
        status, receipt = self.main(run)
        self.assertLess(time.time() - started, 5)
        self.assertEqual((status, receipt["state"], receipt["exit_status"]), (143, "cancelled", 143))
        self.assertEqual(receipt["error"], "stopped by SIGTERM")
        self.assertEqual(receipt["phase"], "refine 1/2")


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
        probe = (f"import importlib.util, os; spec = importlib.util.spec_from_file_location('j', {str(SCRIPT)!r}); "
                 "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                 "print(m.tie_to_parent(os.getppid()), m.tie_to_parent(os.getppid() + 1))")
        result = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.split(), ["True", "False"])
        result = self.script("--job", str(self.write_job()), "--parent", str(os.getpid() + 1))
        self.assertEqual(result.returncode, 143, result.stdout + result.stderr)
        self.assertEqual(job_module.parse_line(result.stdout.splitlines()[-1]),
                         ("error", f"the studio's server (pid {os.getpid() + 1}) is gone; not starting"))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])        # nothing written


NARROWED = []                           # each StandInTensor.narrow: (shape, dim, start, length)


class StandInTensor:
    """Just enough of a tensor for run(): a shape, values for flatten().tolist(), and a tag that follows copies."""

    def __init__(self, *shape, values=None, tag=None, device="cpu"):
        self.shape, self.values, self.tag, self.device = tuple(shape), values, tag, device

    def _copy(self, shape=None, device=None):
        return StandInTensor(*(self.shape if shape is None else shape), values=self.values, tag=self.tag,
                             device=device or self.device)

    def __getitem__(self, index):
        if isinstance(index, tuple):                 # frames[:, top:top + rows], NVIDIA's row crop: as narrow(1, ...)
            assert index[0] == slice(None) and index[1].step is None and len(index) == 2
            return self.narrow(1, index[1].start, index[1].stop - index[1].start)
        assert index == 0
        return self._copy(self.shape[1:])

    def narrow(self, dim, start, length):
        assert 0 <= start and start + length <= self.shape[dim]
        NARROWED.append((self.shape, dim, start, length))
        return self._copy(self.shape[:dim] + (length,) + self.shape[dim + 1:])

    def to(self, *args, **kwargs):
        device = next((a for a in args if a in ("cuda", "cpu")), None)
        return self._copy(device=device)

    def float(self):
        return self._copy()

    long = cpu = flatten = float

    def tolist(self):
        return list(self.values)

    def numel(self):
        return math.prod(self.shape)


def stand_in_torch(loaded):
    """A torch module for run() on the CPU: torch.load answers from `loaded` by file name, the CUDA calls do nothing
    and report fixed memory figures."""
    fake = types.ModuleType("torch")
    fake.__version__, fake.bfloat16, fake.float32 = "0.0+stand-in", "bfloat16", "float32"
    def load(path, map_location=None, weights_only=None):
        # every load in run() reads plain tensors onto the CPU: weights_only=True (no pickled code), map_location cpu
        if weights_only is not True or map_location != "cpu":
            raise AssertionError(f"torch.load({Path(path).name}, map_location={map_location!r}, "
                                 f"weights_only={weights_only!r})")
        return loaded[Path(path).name]
    fake.load = load
    fake.isfinite = lambda x: types.SimpleNamespace(all=lambda: True)
    fake.zeros = lambda *shape, **kwargs: StandInTensor(*shape)
    fake.device = lambda kind, *rest: types.SimpleNamespace(type=kind)
    fake.no_grad = contextlib.nullcontext
    fake.version = types.SimpleNamespace(cuda="13.0")
    fake.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 91900))
    nothing = lambda *args, **kwargs: None  # noqa: E731
    fake.emptied = []                   # each empty_cache: how many calls the stand-in open_h3.pipeline had then

    def empty_cache():
        pipeline = sys.modules.get("open_h3.pipeline")
        fake.emptied.append(len(pipeline.CALLS) if pipeline else 0)
    fake.cuda = types.SimpleNamespace(
        set_device=nothing, synchronize=nothing, reset_peak_memory_stats=nothing, empty_cache=empty_cache,
        max_memory_allocated=lambda: 81 * 2 ** 30, max_memory_reserved=lambda: 83 * 2 ** 30,
        memory_allocated=lambda: 5 * 2 ** 30, is_available=lambda: True, get_device_name=lambda: "stand-in GPU",
        get_device_capability=lambda: (12, 1))
    fake.Tensor = StandInTensor
    return fake


def weight_name(path):
    """A weight file by its directory and name (the base model's parts share one name)."""
    return f"{Path(path).parent.name}/{Path(path).name}"


def stand_in_loaders(loads):
    """safetensors and diffusers as NVIDIA's code uses them: {module name: module}. safetensors.torch.load_file records
    ("load_file", file, device); from_pretrained of diffusers' three classes records ("from_pretrained", class,
    subfolder, device_map) and reads its subfolder's .safetensors files with safetensors.torch.load_file, by attribute
    when it is called, as diffusers' load_state_dict does."""
    safetensors, st_torch, diffusers = (types.ModuleType(name) for name in ("safetensors", "safetensors.torch",
                                                                             "diffusers"))

    def load_file(filename, device="cpu"):
        loads.append(("load_file", weight_name(filename), str(device)))
        return {}
    st_torch.load_file = st_torch.stock_load_file = load_file
    safetensors.torch = st_torch

    class ModelMixin:
        @classmethod
        def from_pretrained(cls, path, subfolder=None, **kwargs):
            loads.append(("from_pretrained", cls.__name__, subfolder, kwargs.get("device_map")))
            for file in sorted((Path(path) / subfolder).glob("*.safetensors")):
                st_torch.load_file(str(file), device="cpu")
            return cls()

        def cuda(self):
            return self
        eval = cuda
    for name in job_module.DIRECT_LOAD:
        setattr(diffusers, name, type(name, (ModelMixin,), {}))
    diffusers.ModelMixin = ModelMixin
    return {"safetensors": safetensors, "safetensors.torch": st_torch, "diffusers": diffusers}


def stand_in_fastload(loads, fallback=()):
    """upscale/fastload.py as the job uses it: load_file records ("fastload", file, device). For the files named in
    `fallback` it does what fastload does where O_DIRECT can't be used: safetensors.torch.load_file as it is then."""
    fastload = types.ModuleType("lf_studio_fastload")
    fastload.ENABLED = fastload.DIRECT = True

    def load_file(filename, device="cpu"):
        if weight_name(filename) in fallback:
            from safetensors.torch import load_file as original
            return original(filename, device=device)
        loads.append(("fastload", weight_name(filename), str(device)))
        return {}
    fastload.load_file = load_file
    return fastload


# NVIDIA's open_h3 as far as run() uses it; every call goes into pipeline.CALLS, every load (diffusers'
# from_pretrained, safetensors' load_file, imported by name as adapters does) into the stand-in loaders' list
STAND_IN_ADAPTERS = f'''import types
from pathlib import Path

from safetensors.torch import load_file

LORA_FILE = {job_module.LORA_FILE!r}
CURVE_FILE = {job_module.CURVE_FILE!r}
CURVE_TABLE = {job_module.CURVE_TABLE!r}


def attach(model, weights_dir):
    load_file(str(Path(weights_dir) / LORA_FILE), device="cpu")
    load_file(str(Path(weights_dir) / CURVE_FILE), device="cpu")
    return {{}}, types.SimpleNamespace(strength=0.0)


class StageMerger:
    def __init__(self, *args, **kwargs):
        raise AssertionError("NVIDIA's StageMerger was used")
'''
STAND_IN_PIPELINE = f'''import types
from pathlib import Path

import torch

from . import adapters, upscaler
from .sigmas import SCHEDULES, shifted_grid

NATIVE_SHAPE = (24, 37, 48, 84)
INPUT_SIZES = {{"768p": (1344, 768), "576p": (1024, 576)}}
NATIVE_SHAPES = {{name: (24, 37, h // 16, w // 16) for name, (w, h) in INPUT_SIZES.items()}}
UPSCALED_SHAPE = (24, 37, 90, 160)
RESOLUTIONS = {{"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080), "2k": (1440, 2560, 1440),
               "4k": (2176, 3840, 2160)}}
UPSCALER_FILE = {job_module.UPSCALER_FILE!r}
CALLS = []


def encode_video(base_model, path, size="768p", seed=42):
    from diffusers import AutoencoderKLMiniMaxH3
    width, height = INPUT_SIZES[size]                 # as NVIDIA's: the size by its name, looked up at call time
    AutoencoderKLMiniMaxH3.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32,
                                           local_files_only=True).cuda().eval()
    CALLS.append(("encode_video", Path(path).name, size, seed))
    return torch.Tensor(*NATIVE_SHAPES[size], tag="final")


def encode_mp4(frames, path, fps=24, crf=10):
    CALLS.append(("encode_mp4", Path(path).name, frames.shape))
    Path(path).write_bytes(b"mp4")


class Node:
    MODEL_CACHE = {{}}

    def load_model(self, name, device, precision):
        CALLS.append(("load_model", name, precision))
        self.MODEL_CACHE[name] = object()


class Pipeline:
    def __init__(self, weights_dir, base_model, fwd="sol_engine", hyperflow=None, vae_batch=26, steps=3,
                 start_sigma=None, resolution="2k", vae="standard"):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
        CALLS.append(("Pipeline", dict(fwd=fwd, hyperflow=hyperflow, vae_batch=vae_batch, steps=steps,
                                       start_sigma=start_sigma, resolution=resolution, vae=vae)))
        AutoencoderKLMiniMaxH3Audio.from_pretrained(base_model, subfolder="audio_vae", torch_dtype=torch.float32,
                                                    local_files_only=True).cuda().eval()     # encode_silence
        MiniMaxH3Transformer3DModel.from_pretrained(base_model, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True, local_files_only=True)
        self.sigmas = SCHEDULES[steps] if start_sigma is None else shifted_grid(start_sigma, steps)
        height, width, self.keep_rows = RESOLUTIONS[resolution]
        self.target_shape = (24, 37, height // 16, width // 16)
        self.device = "cuda"
        self.audio = torch.Tensor(2, 32, 207, tag="silence", device="cuda")
        self.model = types.SimpleNamespace(transformer_blocks=[
            types.SimpleNamespace(ff=types.SimpleNamespace(forward=lambda x: x)) for _ in range(3)])
        lora, self.curve = adapters.attach(self.model, weights_dir)
        self.merger = adapters.StageMerger(self.model, {{"refine": [(lora, 1.0)]}}, self.device, resident=False)
        self.node = Node()
        self.fast_vae = None
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32,
                                                          local_files_only=True).cuda().eval()    # load_video_vae

    def set_stage(self, stage):
        self.curve.strength = 1.0
        self.merger.merge(stage)
        CALLS.append(("set_stage", stage))

    def on_step(self, stage, index):
        CALLS.append(("on_step", stage, index))

    def upscale(self, native):
        CALLS.append(("upscale", native.shape, native.tag))
        target = self.target_shape                    # NVIDIA's Pipeline.upscale from here on (pipeline.py:223-228)
        if tuple(native.shape) == target:
            return native.float().to(self.device)
        if target[2] > UPSCALED_SHAPE[2]:
            native = upscaler.upscale(self.node, native, UPSCALED_SHAPE)
        return upscaler.upscale(self.node, native, target).to(self.device)

    def refine(self, low, text, seed):
        CALLS.append(("refine", low.shape, text["prompt_embeds"].shape, seed, self.audio.tag, self.curve.strength))
        for index in range(len(self.sigmas) - 1):
            self.on_step("refine", index)
        return torch.Tensor(*low.shape)

    def decode(self, latents):
        """NVIDIA's: computed at the target grid, the centre keep_rows rows kept."""
        CALLS.append(("decode", self.model is None, self.vae is not None, dict(self.node.MODEL_CACHE)))
        return torch.Tensor(121, self.keep_rows, self.target_shape[3] * 16, 3)
'''
# NVIDIA's open_h3.upscaler: the latent upscaler to a target grid (the node is NVIDIA's too), into pipeline.CALLS
STAND_IN_UPSCALER = '''import torch


def upscale(node, native, target_shape):
    from .pipeline import CALLS
    CALLS.append(("upscaler.upscale", tuple(native.shape), tuple(target_shape)))
    return torch.Tensor(*native.shape[:2], *target_shape[2:])
'''


class RunTests(Fixture):
    """run() through main(), on the CPU: a stand-in torch and a stand-in of NVIDIA's open_h3 that records its calls."""

    def setUp(self):
        super().setUp()
        for name, text in (("adapters", STAND_IN_ADAPTERS), ("pipeline", STAND_IN_PIPELINE),
                           ("upscaler", STAND_IN_UPSCALER), ("upscaler_node", "# a stand-in, never imported\n")):
            (self.package / "open_h3" / f"{name}.py").write_text(text)
        self.loaded = {
            "conditioning.pt": {"prompt_embeds": StandInTensor(1, 41, 5120), "task": "t2va", "prompt": "a test",
                                "text_token_tags": StandInTensor(41, values=[1] * 41)},
            "stage1_direct_tensors.pt": {"audio": StandInTensor(1, 2, 161333)}}

    def run_job(self, fallback=(), seams=None, **changes):
        """main() on job.json with `changes`; the loads go into self.loads, the stand-in modules into self.modules.
        `seams`: {name: stand-in} for the job's own functions."""
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        fake = stand_in_torch(self.loaded)
        stdout = io.StringIO()
        clip_audio = StandInTensor(2, 32, 207, tag="clip")
        self.loads = []
        self.modules = stand_in_loaders(self.loads)
        NARROWED.clear()

        def encode_audio(base, pcm, torch):            # the audio VAE as the job's encode_audio loads it
            self.modules["diffusers"].AutoencoderKLMiniMaxH3Audio.from_pretrained(
                base, subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
            return clip_audio
        with contextlib.ExitStack() as stack:
            for name, value in (seams or {}).items():
                stack.enter_context(mock.patch.object(job_module, name, value))
            stack.enter_context(mock.patch.dict(sys.modules, {"torch": fake, **self.modules}))
            stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
            stack.enter_context(mock.patch.object(job_module, "apply_environment"))
            stack.enter_context(mock.patch.object(job_module, "load_fastload",
                                                  return_value=stand_in_fastload(self.loads, fallback)))
            encode_audio = stack.enter_context(mock.patch.object(job_module, "encode_audio",
                                                                 side_effect=encode_audio))
            lean = stack.enter_context(mock.patch.object(job_module, "install_lean_blocks", return_value=3))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            status = job_module.main(["--job", str(self.write_job(**changes))])
            calls = list(sys.modules["open_h3.pipeline"].CALLS) if "open_h3.pipeline" in sys.modules else []
            pipeline = sys.modules.get("open_h3.pipeline")       # NVIDIA's module dicts as the run left them
            self.dicts = {name: dict(getattr(pipeline, name)) for name in ("INPUT_SIZES", "NATIVE_SHAPES",
                                                                           "RESOLUTIONS")} if pipeline else None
        phases = [p[1] for p in map(job_module.parse_line, stdout.getvalue().splitlines()) if p and p[0] == "phase"]
        self.lean, self.stdout, self.emptied = lean, stdout.getvalue(), fake.emptied
        return status, json.loads((self.out / "receipt.json").read_text()), calls, phases, encode_audio

    def expected_loads(self, fast, audio=True, fallback=()):
        """Every load of a run in order: the encodes (the clip's video VAE, and with `audio` its audio VAE), then
        NVIDIA's Pipeline (the audio VAE for its silence, the DiT, the LoRA and the Sharp curves, the video VAE).
        Fast: from_pretrained with device_map cuda and every file through fastload to the GPU (a fallback through the
        stock loader, to the GPU); standard: NVIDIA's own calls, unchanged."""
        def model(cls, part):
            name = f"{part}/diffusion_pytorch_model.safetensors"
            return [("from_pretrained", cls, part, {"": "cuda"} if fast else None), read(name)]

        def read(name):
            if not fast:
                return ("load_file", name, "cpu")
            return ("load_file", name, "cuda") if name in fallback else ("fastload", name, "cuda")
        loads = model("AutoencoderKLMiniMaxH3", "vae") + (model("AutoencoderKLMiniMaxH3Audio", "audio_vae")
                                                          if audio else [])
        return loads + model("AutoencoderKLMiniMaxH3Audio", "audio_vae") + \
            model("MiniMaxH3Transformer3DModel", "transformer") + \
            [read(job_module.LORA_FILE), read(job_module.CURVE_FILE)] + model("AutoencoderKLMiniMaxH3", "vae")

    def assert_loaders(self, fast):
        """After a run: the fast load replaced the loaders, the standard one left them as they were."""
        st_torch, diffusers = self.modules["safetensors.torch"], self.modules["diffusers"]
        self.assertEqual(st_torch.load_file is not st_torch.stock_load_file, fast)
        for name in job_module.DIRECT_LOAD:
            self.assertEqual(getattr(diffusers, name).from_pretrained.__func__ is
                             diffusers.ModelMixin.from_pretrained.__func__, not fast, name)

    def test_final_with_the_clip_audio(self):
        """The default job: the final clip, its audio, 2K, the fast load."""
        status, receipt, calls, phases, encode_audio = self.run_job()
        lean = self.lean
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.loads, self.expected_loads(fast=True))     # patched before the first load
        self.assertEqual((receipt["job"]["load"], receipt["load"]),
                         ("fast", {"mode": "fast", "o_direct": True, "files": 7, "fallbacks": 0}))
        self.assert_loaders(fast=True)
        for line in ("fast load: upscale/fastload.py to the GPU (O_DIRECT on), from_pretrained with device_map cuda",
                     "fast load: 7 files, 0 of them through mmap"):              # worker.log, TROUBLESHOOTING
            self.assertRegex(self.stdout, rf"(?m)^\[lf-studio\] \d\d:\d\d:\d\d {re.escape(line)}$")
        # the allocator's cache given back once everything is loaded (the fast load's LoRA came to the GPU), before
        # the upscale: right after the upscaler's load_model
        self.assertIn([c[0] for c in calls].index("load_model") + 1, self.emptied)
        self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])
        self.assertEqual([c[0] for c in calls], ["encode_video", "Pipeline", "load_model", "upscale",
                                                 "upscaler.upscale", "set_stage", "refine", "on_step", "on_step",
                                                 "decode", "encode_mp4"])
        self.assertEqual(calls[0], ("encode_video", "refined_1344x768_121f.mp4", "768p", 42))
        self.assertEqual(calls[1][1], {"fwd": "standard", "hyperflow": None, "vae_batch": 26, "steps": 2,
                                       "start_sigma": 0.7, "resolution": "2k", "vae": "standard"})
        self.assertEqual(calls[3], ("upscale", (24, 37, 48, 84), "final"))
        self.assertEqual(calls[6], ("refine", (24, 37, 90, 160), (41, 5120), 1234, "clip", 1.0))
        self.assertEqual(calls[9], ("decode", True, True, {}))      # the DiT and the upscaler freed, the VAE kept
        self.assertEqual(calls[10], ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, 1440, 2560, 3)))
        self.assertEqual(encode_audio.call_args.args[0], self.base)
        self.assertEqual(encode_audio.call_args.args[1].shape, (2, 161333))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json", "receipt.json", "video_noaudio.mp4"])
        self.assertEqual(receipt["output"], {"file": "video_noaudio.mp4", "size": [2560, 1440], "frames": 121,
                                             "bytes": 3})
        self.assertEqual((receipt["orientation"], receipt["upscale_route"]), ("landscape", [[90, 160]]))
        self.assertEqual((receipt["sigmas"], receipt["steps"], len(receipt["steps_s"])), ([0.7, 0.35, 0.0], 2, 2))
        self.assertEqual(set(receipt["times_s"]), {"encode_video", "encode_audio", "load", "upscale", "merge_refine",
                                                   "refine", "vae_decode", "mp4"})
        self.assertEqual(set(receipt["peak_gib"]), set(receipt["times_s"]))
        self.assertEqual(receipt["peak_allocated_gib"], 81.0)
        self.assertEqual((receipt["merge"]["weights"], receipt["lean_chunks"], receipt["text_rows"]), (0, None, 41))
        lean.assert_not_called()                                    # the stock block forward below 4K
        self.assertEqual((receipt["prompt"], receipt["versions"]["gpu"], receipt["versions"]["torch"]),
                         ("a test", "stand-in GPU", "0.0+stand-in"))
        self.assertEqual(receipt["allocated_before_decode_gib"], 5.0)
        self.assertIn("refine 2/2", receipt["lowest_mem_available_gib"])

    def test_landscape_makes_the_calls_it_made_before_portrait(self):
        """A 16:9 clip's run makes the same calls into NVIDIA's code, in the same order with the same arguments, as
        the job did before it took portrait clips (recorded with the job of commit 2e0dadf against these stand-ins):
        encode_video at "768p", Pipeline at the resolution's own key, NVIDIA's own Pipeline.upscale (the upscaler
        once below 4K, through the landscape 2K grid at 4K), the refine, NVIDIA's decode with its row crop and
        encode_mp4. It loads the same files, adds nothing to NVIDIA's module dicts and crops nothing itself."""
        nvidia_dicts = {"INPUT_SIZES": {"768p": (1344, 768), "576p": (1024, 576)},
                        "NATIVE_SHAPES": {"768p": (24, 37, 48, 84), "576p": (24, 37, 36, 64)},
                        "RESOLUTIONS": {"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080),
                                        "2k": (1440, 2560, 1440), "4k": (2176, 3840, 2160)}}
        pipeline = {"fwd": "standard", "hyperflow": None, "vae_batch": 26, "steps": 2, "start_sigma": 0.7,
                    "vae": "standard"}
        before = {
            "1080p": [("encode_video", "refined_1344x768_121f.mp4", "768p", 42),
                      ("Pipeline", dict(pipeline, resolution="1080p")),
                      ("load_model", "upscaler", "bf16"),
                      ("upscale", (24, 37, 48, 84), "final"),
                      ("upscaler.upscale", (24, 37, 48, 84), (24, 37, 68, 120)),
                      ("set_stage", "refine"),
                      ("refine", (24, 37, 68, 120), (41, 5120), 1234, "clip", 1.0),
                      ("on_step", "refine", 0), ("on_step", "refine", 1),
                      ("decode", True, True, {}),
                      ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, 1080, 1920, 3))],
            "2k": [("encode_video", "refined_1344x768_121f.mp4", "768p", 42),
                   ("Pipeline", dict(pipeline, resolution="2k")),
                   ("load_model", "upscaler", "bf16"),
                   ("upscale", (24, 37, 48, 84), "final"),
                   ("upscaler.upscale", (24, 37, 48, 84), (24, 37, 90, 160)),
                   ("set_stage", "refine"),
                   ("refine", (24, 37, 90, 160), (41, 5120), 1234, "clip", 1.0),
                   ("on_step", "refine", 0), ("on_step", "refine", 1),
                   ("decode", True, True, {}),
                   ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, 1440, 2560, 3))],
            "4k": [("encode_video", "refined_1344x768_121f.mp4", "768p", 42),
                   ("Pipeline", dict(pipeline, resolution="4k")),
                   ("load_model", "upscaler", "bf16"),
                   ("upscale", (24, 37, 48, 84), "final"),
                   ("upscaler.upscale", (24, 37, 48, 84), (24, 37, 90, 160)),
                   ("upscaler.upscale", (24, 37, 90, 160), (24, 37, 136, 240)),
                   ("set_stage", "refine"),
                   ("refine", (24, 37, 136, 240), (41, 5120), 1234, "silence", 1.0),
                   ("on_step", "refine", 0), ("on_step", "refine", 1),
                   ("decode", True, True, {}),
                   ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, 2160, 3840, 3))]}
        for resolution, changes in (("1080p", {}), ("2k", {}), ("4k", {"audio": "silence", "load": "standard"})):
            with self.subTest(resolution=resolution):
                status, receipt, calls, _, _ = self.run_job(resolution=resolution, **changes)
                self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or
                                 receipt.get("error"))
                self.assertEqual(calls, before[resolution])
                self.assertEqual(self.loads, self.expected_loads(fast=not changes, audio=not changes))
                self.assertEqual(self.dicts, nvidia_dicts)
                self.assertEqual(NARROWED, [])
                self.assertEqual(receipt["lean_chunks"], 16 if resolution == "4k" else None)

    def portrait_clip(self, pin=True):
        """The clip a 9:16 one; with `pin`, NVIDIA_FILES pinned to the stand-in package's files."""
        (self.case / "stage2" / "refined_1344x768_121f.mp4").unlink()
        shutil.copy(self.portrait, self.case / "stage2" / "refined_768x1344_121f.mp4")
        if pin:
            self.pin_stand_ins()

    def pin_stand_ins(self):
        """NVIDIA_FILES pinned to the stand-in package's files."""
        pins = {rel: hashlib.sha256((self.package / rel).read_bytes()).hexdigest() for rel in job_module.NVIDIA_FILES}
        patcher = mock.patch.dict(job_module.NVIDIA_FILES, pins)
        patcher.start()
        self.addCleanup(patcher.stop)

    def whole_clip(self, orientation="landscape", pin=True):
        """The clip one of 124 frames, every frame H3 made; with `pin`, NVIDIA_FILES pinned to the stand-ins."""
        for mp4 in (self.case / "stage2").glob("refined_*.mp4"):
            mp4.unlink()
        width, height = job_module.SHAPES[orientation]["input"]
        shutil.copy(self.final124 if orientation == "landscape" else
                    make_mp4(self.tmp / "portrait124.mp4", width=768, height=1344, frames=124),
                    self.case / "stage2" / f"refined_{width}x{height}_124f.mp4")
        if pin:
            self.pin_stand_ins()

    def own_seams(self):
        """encode_clip and decode_released as stand-ins that record their calls (self.encoded, self.decoded): the
        video VAE loaded as NVIDIA's encode_video loads it, the latent of the clip's shape; frames at the computed
        grid."""
        self.encoded, self.decoded = [], []

        def encode_clip(base, mp4, size, kept, pad, seed, torch, keyframes=(), encode_seed=None, **kwargs):
            self.modules["diffusers"].AutoencoderKLMiniMaxH3.from_pretrained(
                base, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).cuda().eval()
            self.encoded.append((Path(mp4).name, size, kept, pad, seed, [a for a, _ in keyframes], encode_seed))
            native = job_module.SHAPES["landscape" if size[0] > size[1] else "portrait"]["native"]
            return torch.Tensor(*native, tag="final"), {a: torch.Tensor(1, 24, 1, 1, 1, tag=a) for a, _ in keyframes}, \
                1.5

        def decode_released(vae, latents, kept, torch, device="cuda"):
            self.decoded.append((latents.shape, kept, device))
            return torch.Tensor(kept, latents.shape[2] * 16, latents.shape[3] * 16, 3)
        return {"encode_clip": encode_clip, "decode_released": decode_released}

    def test_portrait(self):
        """A 9:16 clip: NVIDIA's encode_video at 768x1344 and its Pipeline at the portrait grid (through the entries
        the job adds to its dicts), NVIDIA's upscaler to that grid (through the 2K grid turned, 160x90 latents, at
        4K only), never NVIDIA's Pipeline.upscale (which would go through the landscape 2K grid), NVIDIA's decode
        with every row kept, then the centre columns kept by the job."""
        self.portrait_clip()
        pipeline = {"fwd": "standard", "hyperflow": None, "vae_batch": 26, "steps": 2, "start_sigma": 0.7,
                    "vae": "standard"}
        nvidia_dicts = {"INPUT_SIZES": {"768p": (1344, 768), "576p": (1024, 576), "768p-portrait": (768, 1344)},
                        "NATIVE_SHAPES": {"768p": (24, 37, 48, 84), "576p": (24, 37, 36, 64),
                                          "768p-portrait": (24, 37, 84, 48)},
                        "RESOLUTIONS": {"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080),
                                        "2k": (1440, 2560, 1440), "4k": (2176, 3840, 2160),
                                        "1080p-portrait": (1920, 1088, 1920), "2k-portrait": (2560, 1440, 2560),
                                        "4k-portrait": (3840, 2176, 3840)}}
        cases = {"1080p": ([(24, 37, 120, 68)], (121, 1920, 1088, 3), 4, (1080, 1920), None),
                 "2k": ([(24, 37, 160, 90)], (121, 2560, 1440, 3), 0, (1440, 2560), None),
                 "4k": ([(24, 37, 160, 90), (24, 37, 240, 136)], (121, 3840, 2176, 3), 8, (2160, 3840), 16)}
        for resolution, (grids, decoded, left, (width, height), lean) in cases.items():
            with self.subTest(resolution=resolution):
                status, receipt, calls, phases, encode_audio = self.run_job(resolution=resolution)
                self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or
                                 receipt.get("error"))
                natives = [(24, 37, 84, 48)] + grids[:-1]
                self.assertEqual(calls, [
                    ("encode_video", "refined_768x1344_121f.mp4", "768p-portrait", 42),
                    ("Pipeline", dict(pipeline, resolution=f"{resolution}-portrait")),
                    ("load_model", "upscaler", "bf16")] +
                    [("upscaler.upscale", native, grid) for native, grid in zip(natives, grids)] + [
                    ("set_stage", "refine"),
                    ("refine", grids[-1], (41, 5120), 1234, "clip", 1.0),
                    ("on_step", "refine", 0), ("on_step", "refine", 1),
                    ("decode", True, True, {}),
                    ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, height, width, 3))])
                self.assertEqual(NARROWED, [(decoded, 2, left, width)])
                self.assertEqual(self.dicts, nvidia_dicts)
                self.assertEqual(self.loads, self.expected_loads(fast=True))
                self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])
                self.assertEqual((receipt["orientation"], receipt["upscale_route"], receipt["lean_chunks"]),
                                 ("portrait", [list(grid[2:]) for grid in grids], lean))
                self.assertEqual(receipt["output"], {"file": "video_noaudio.mp4", "size": [width, height],
                                                     "frames": 121, "bytes": 3})
                self.assertEqual(encode_audio.call_args.args[1].shape, (2, 161333))    # the clip's audio, as 16:9

    def test_a_clip_of_every_frame_made(self):
        """124 frames (a LightX2V Turbo clip with a last keyframe): encode_clip reads them all at NVIDIA's input size
        with no copies, decode_released keeps them all, then NVIDIA's centre rows (and for 9:16 the job's centre
        columns); NVIDIA's encode_video and decode are not called, the rest is NVIDIA's as for 121 frames, and the
        output has 124 frames."""
        pipeline = {"fwd": "standard", "hyperflow": None, "vae_batch": 26, "steps": 2, "start_sigma": 0.7,
                    "vae": "standard"}
        cases = {("landscape", "1080p"): ([(24, 37, 68, 120)], [((124, 1088, 1920, 3), 1, 4, 1080)]),
                 ("landscape", "2k"): ([(24, 37, 90, 160)], [((124, 1440, 2560, 3), 1, 0, 1440)]),
                 ("landscape", "4k"): ([(24, 37, 90, 160), (24, 37, 136, 240)], [((124, 2176, 3840, 3), 1, 8, 2160)]),
                 ("portrait", "1080p"): ([(24, 37, 120, 68)], [((124, 1920, 1088, 3), 1, 0, 1920),
                                                               ((124, 1920, 1088, 3), 2, 4, 1080)]),
                 ("portrait", "2k"): ([(24, 37, 160, 90)], [((124, 2560, 1440, 3), 1, 0, 2560),
                                                            ((124, 2560, 1440, 3), 2, 0, 1440)])}
        for orientation in job_module.SHAPES:
            with self.subTest(orientation=orientation):
                self.whole_clip(orientation)
                shape = job_module.SHAPES[orientation]
                for (which, resolution), (grids, narrowed) in cases.items():
                    if which != orientation:
                        continue
                    status, receipt, calls, phases, encode_audio = self.run_job(seams=self.own_seams(),
                                                                                resolution=resolution)
                    self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or
                                     receipt.get("error"))
                    width, height = shape["sizes"][resolution]
                    natives = [shape["native"]] + grids[:-1]
                    self.assertEqual(calls, [
                        ("Pipeline", dict(pipeline, resolution=shape["resolutions"][resolution])),
                        ("load_model", "upscaler", "bf16")] +
                        ([("upscale", shape["native"], "final")] if orientation == "landscape" else []) +
                        [("upscaler.upscale", native, grid) for native, grid in zip(natives, grids)] + [
                        ("set_stage", "refine"),
                        ("refine", grids[-1], (41, 5120), 1234, "clip", 1.0),
                        ("on_step", "refine", 0), ("on_step", "refine", 1),
                        ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (124, height, width, 3))])
                    self.assertEqual(self.encoded, [(f"refined_{shape['input'][0]}x{shape['input'][1]}_124f.mp4",
                                                     shape["input"], 124, 0, 42, [], None)])
                    self.assertEqual(self.decoded, [(grids[-1], 124, "cuda")])
                    self.assertEqual(NARROWED, narrowed)
                    self.assertEqual(self.loads, self.expected_loads(fast=True))
                    self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding",
                                              "saving"])
                    self.assertEqual(receipt["frames"], {"kept": 124, "made": 124, "pad": 0, "T": 37})
                    self.assertEqual(receipt["output"], {"file": "video_noaudio.mp4", "size": [width, height],
                                                         "frames": 124, "bytes": 3})
                    self.assertIn("encode_video", receipt["times_s"])
                    self.assertEqual(encode_audio.call_args.args[1].shape, (2, 161333))

    def pin_seams(self, ends, changed=()):
        """The pin's parts as stand-ins on top of own_seams: check_pin answering `ends` ({anchor: (used, dB)}),
        pin_pixels the pictures of the ends used (those in `changed` turned off, as a picture changed since the
        check), refine_pinned recording its calls (self.pinned) and stepping as NVIDIA's refine does, tensor_sha256
        the tensor's tag."""
        seams, self.pinned, self.grids = self.own_seams(), [], []

        def check_pin(job, final, ffmpeg=None):
            return {anchor: {"used": used, "agreement_db": db, "frame": 0 if anchor == "first" else
                             job["frames"]["kept"] - 1, "fit": "stretch", "box": None, "size": [1344, 768],
                             "why": None if used else "the clip's frame is far from your picture"}
                    for anchor, (used, db) in ends.items()}

        def pin_pixels(job, grid):
            self.grids.append(grid)
            pixels = []
            for keyframe in job["pin"]["keyframes"]:
                record = job["pin_check"][keyframe["anchor"]]
                if record["used"] and keyframe["anchor"] in changed:
                    record.update(used=False, why=job_module.PIN_WHY["changed"])
                elif record["used"]:
                    pixels.append((keyframe["anchor"], f"pixels of {keyframe['anchor']}"))
            return pixels

        def refine_pinned(pipe, low, text, seed, keyframes, made, noise_aug, noise_seed):
            self.pinned.append((low.shape, seed, [(anchor, latent.tag) for anchor, latent in keyframes], made,
                                noise_aug, noise_seed))
            for index in range(len(pipe.sigmas) - 1):
                pipe.on_step("refine", index)
            return StandInTensor(*low.shape, tag="refined"), {
                "anchors": [anchor for anchor, _ in keyframes], "sequence_rows": 80170, "condition_rows": 4080,
                "timesteps": [[0.3, 0.999, 1.0], [0.505882, 0.999, 1.0]], "condition_sha256": "c" * 64,
                "noise_aug": noise_aug, "noise_seed": noise_seed}
        seams.update(check_pin=check_pin, pin_pixels=pin_pixels, refine_pinned=refine_pinned,
                     tensor_sha256=lambda tensor: f"sha of {tensor.tag}")
        return seams

    def test_a_pinned_run(self):
        """The ends check_pin keeps go through encode_clip (on the VAE that encodes the clip, at the refine's computed
        grid) and refine_pinned, in packed order, with the frames H3 made; NVIDIA's refine is not called, the rest is
        as without a pin. 121 frames: NVIDIA's read of the clip (121, 3 copies) and NVIDIA's decode; 124 frames: every
        frame, decode_released. The receipt's pin has each end, the encode's time and the sha256 of the latents."""
        self.pin_stand_ins()
        for kept, ends, resolution in ((121, ("first",), "1080p"), (124, ("first", "last"), "1080p"),
                                       (124, ("last",), "2k")):
            with self.subTest(kept=kept, ends=ends):
                if kept == 124:
                    self.whole_clip()
                status, receipt, calls, phases, _ = self.run_job(
                    seams=self.pin_seams({end: (True, 33.5) for end in ends}), resolution=resolution,
                    pin=pin(*ends))
                self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or
                                 receipt.get("error"))
                grid = job_module.computed_grid("landscape", resolution)
                self.assertEqual(self.grids, [grid])
                self.assertEqual(self.encoded, [(f"refined_1344x768_{kept}f.mp4", (1344, 768), kept,
                                                 job_module.FRAME_CASES[kept]["pad"], 42,
                                                 list(ends), 42)])
                low = LANDSCAPE_TARGETS[resolution]
                self.assertEqual(self.pinned, [(low, 1234, [(end, end) for end in ends], 124, 0.999, 42)])
                names = [c[0] for c in calls]
                self.assertNotIn("encode_video", names)          # NVIDIA's own encode: not with a pin held
                self.assertNotIn("refine", names)
                self.assertEqual(names.count("decode"), 1 if kept == 121 else 0)
                self.assertEqual(self.decoded, [] if kept == 121 else [(low, 124, "cuda")])
                self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])
                self.assertEqual(self.loads, self.expected_loads(fast=True))     # one video VAE load for both
                record = receipt["pin"]
                self.assertEqual((record["asked"], record["grid"], record["check_grid"], record["min_db"],
                                  record["noise_aug"], record["encode_seed"], record["noise_seed"]),
                                 (list(ends), list(grid), [672, 384], 28, 0.999, 42, 42))
                for end in ends:
                    self.assertEqual((record[end]["used"], record[end]["agreement_db"], record[end]["why"],
                                      record[end]["latent_sha256"], record[end]["latent_shape"]),
                                     (True, 33.5, None, f"sha of {end}", [1, 24, 1, 1, 1]))
                self.assertEqual((record["encode_s"], record["native_sha256"], record["refined_sha256"],
                                  record["condition_sha256"]), (1.5, "sha of final", "sha of refined", "c" * 64))
                self.assertEqual(record["timesteps"], [[0.3, 0.999, 1.0], [0.505882, 0.999, 1.0]])
                self.assertEqual(record["layout"], {"anchors": list(ends), "sequence_rows": 80170,
                                                    "condition_rows": 4080})
                self.assertEqual((receipt["frames"]["kept"], receipt["output"]["frames"]), (kept, kept))
                self.assertRegex(self.stdout, rf"(?m)^\[lf-studio\] \d\d:\d\d:\d\d pin {ends[0]}: frame \d+ 33\.5 dB "
                                              r"from the picture, held in the refine$")

    def test_ends_the_check_does_not_keep(self):
        """An end check_pin did not keep is not held: none kept, a 121-frame clip runs NVIDIA's encode, refine and
        decode, as without a pin (its receipt says why); one of two kept, only that one; a picture changed since the
        check is not held either (pin_pixels), and the refine is NVIDIA's when none is left."""
        status, receipt, calls, _, _ = self.run_job(seams=self.pin_seams({"first": (False, 21.8)}), pin=pin("first"))
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual([c[0] for c in calls], ["encode_video", "Pipeline", "load_model", "upscale",
                                                 "upscaler.upscale", "set_stage", "refine", "on_step", "on_step",
                                                 "decode", "encode_mp4"])     # NVIDIA's, as before the pin
        self.assertEqual((self.encoded, self.pinned, self.grids), ([], [], []))
        self.assertEqual((receipt["pin"]["first"]["used"], receipt["pin"]["first"]["why"],
                          receipt["pin"]["native_sha256"]),
                         (False, "the clip's frame is far from your picture", "sha of final"))
        for key in ("encode_s", "layout", "timesteps", "condition_sha256", "refined_sha256"):
            self.assertNotIn(key, receipt["pin"])
        self.assertRegex(self.stdout, r"(?m)^\[lf-studio\] \d\d:\d\d:\d\d pin first: frame 0 21\.8 dB from the "
                                      r"picture, not held: the clip's frame is far from your picture$")
        self.pin_stand_ins()
        self.whole_clip()
        status, receipt, calls, _, _ = self.run_job(seams=self.pin_seams({"first": (True, 34.0),
                                                                          "last": (False, 25.4)}),
                                                    pin=pin("first", "last"))
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.pinned[0][2], [("first", "first")])
        self.assertEqual((self.encoded[0][5], receipt["pin"]["last"]["used"]), (["first"], False))
        status, receipt, calls, _, _ = self.run_job(seams=self.pin_seams({"first": (True, 34.0), "last": (True, 32.0)},
                                                                         changed=("last",)), pin=pin("first", "last"))
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual((self.pinned[0][2], self.encoded[0][5]), ([("first", "first")], ["first"]))
        self.assertEqual((receipt["pin"]["last"]["used"], receipt["pin"]["last"]["why"]),
                         (False, "your picture changed since the clip was made"))
        status, receipt, calls, _, _ = self.run_job(seams=self.pin_seams({"first": (True, 34.0)}, changed=("first",)),
                                                    pin=pin("first"))
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual((self.pinned, self.encoded[0][5:], [c[0] for c in calls].count("refine")),
                         ([], ([], None), 1))                 # the clip read whole, NVIDIA's refine

    def test_a_pinned_9x16_clip(self):
        """9:16 with 124 frames and both ends: the pictures on the portrait grid, the portrait route, decode_released
        and the centre columns."""
        self.whole_clip("portrait")
        status, receipt, calls, _, _ = self.run_job(seams=self.pin_seams({"first": (True, 35.4), "last": (True, 31.9)}),
                                                    resolution="1080p", pin=pin("first", "last"))
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual((self.grids, receipt["pin"]["grid"], receipt["pin"]["check_grid"]),
                         ([(1088, 1920)], [1088, 1920], [384, 672]))
        self.assertEqual(self.pinned, [((24, 37, 120, 68), 1234, [("first", "first"), ("last", "last")], 124, 0.999,
                                        42)])
        self.assertEqual([c for c in calls if c[0] == "upscaler.upscale"], [("upscaler.upscale", (24, 37, 84, 48),
                                                                               (24, 37, 120, 68))])
        self.assertEqual(NARROWED, [((124, 1920, 1088, 3), 1, 0, 1920), ((124, 1920, 1088, 3), 2, 4, 1080)])
        self.assertEqual(receipt["output"]["size"], [1080, 1920])

    def test_a_held_end_needs_nvidias_pipeline_pinned(self):
        """A pin held on a 121-frame clip reads it through encode_clip, NVIDIA's code copied: NVIDIA's pipeline.py
        must be the pinned one, else it is refused before anything loads."""
        status, receipt, calls, phases, _ = self.run_job(seams=self.pin_seams({"first": (True, 33.0)}),
                                                         pin=pin("first"))
        self.assertEqual((status, receipt["state"]), (2, "failed"), receipt.get("traceback"))
        self.assertRegex(receipt["error"], r"'open_h3/pipeline\.py': '[0-9a-f]{64}'\}$")
        self.assertEqual((calls, phases, self.loads, self.encoded), ([], [], [], []))
        self.assertEqual(receipt["pin"]["first"]["used"], True)              # the check's answer, kept

    def test_121_frames_never_reach_the_copies(self):
        """A 121-frame clip runs NVIDIA's encode_video and decode, never encode_clip or decode_released, and does not
        need NVIDIA's pipeline.py pinned (16:9)."""
        status, receipt, calls, _, _ = self.run_job(seams=self.own_seams(), resolution="1080p")
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual((self.encoded, self.decoded), ([], []))
        self.assertEqual(([c[0] for c in calls][0], [c[0] for c in calls][-2]), ("encode_video", "decode"))
        self.assertEqual(receipt["frames"], {"kept": 121, "made": 124, "pad": 3, "T": 37})

    def test_the_copies_need_nvidias_pipeline_pinned(self):
        """encode_clip and decode_released copy NVIDIA's pipeline.py: with another one (here the stand-in, not
        pinned) a 124-frame clip is refused before anything loads."""
        self.whole_clip(pin=False)
        status, receipt, calls, phases, _ = self.run_job(seams=self.own_seams())
        self.assertEqual((status, receipt["state"]), (2, "failed"), receipt.get("traceback"))
        self.assertRegex(receipt["error"], r"^NVIDIA's package is not the one this wrapper was written for "
                                           r"\(6e4a440\): \{'open_h3/pipeline\.py': '[0-9a-f]{64}'\}$")
        self.assertEqual((calls, phases, self.loads, self.encoded), ([], [], [], []))

    def test_portrait_needs_the_pinned_nvidia_files(self):
        """A 9:16 clip with NVIDIA's package not at the pinned files (here the stand-ins) is refused before anything
        loads; a 16:9 clip does not read them."""
        self.portrait_clip(pin=False)
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"], receipt["orientation"]), (2, "failed", "portrait"))
        self.assertRegex(receipt["error"], r"^NVIDIA's package is not the one this wrapper was written for "
                                           r"\(6e4a440\): \{'open_h3/pipeline\.py': '[0-9a-f]{64}', "
                                           r"'open_h3/upscaler\.py': '[0-9a-f]{64}', "
                                           r"'open_h3/upscaler_node\.py': '[0-9a-f]{64}'\}$")
        self.assertEqual((calls, phases, self.loads), ([], [], []))

    def test_portrait_without_a_pinned_file_is_refused(self):
        """A pinned file that is not in NVIDIA's package: refused (exit 2, its digest None), not a traceback."""
        self.portrait_clip()
        (self.package / "open_h3" / "upscaler_node.py").unlink()
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"]), (2, "failed"), receipt.get("traceback"))
        self.assertEqual(receipt["error"], "NVIDIA's package is not the one this wrapper was written for (6e4a440): "
                                           "{'open_h3/upscaler_node.py': None}")
        self.assertNotIn("traceback", receipt)
        self.assertEqual((calls, phases, self.loads), ([], [], []))

    def test_portrait_with_the_standard_load(self):
        """A 9:16 clip with LFSTUDIO_UPSCALE_LOAD=standard (as the measured 16:9 job ran): NVIDIA's own loaders, and
        the same calls into NVIDIA's code as with the fast load."""
        self.portrait_clip()
        for resolution in job_module.SIZES:
            with self.subTest(resolution=resolution):
                runs = {}
                for load in job_module.LOADS:
                    status, receipt, calls, phases, _ = self.run_job(resolution=resolution, load=load)
                    self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or
                                     receipt.get("error"))
                    self.assertEqual(self.loads, self.expected_loads(fast=load == "fast"))
                    self.assert_loaders(fast=load == "fast")
                    self.assertEqual((receipt["orientation"], receipt["job"]["load"], receipt["load"]["mode"]),
                                     ("portrait", load, load))
                    runs[load] = (calls, phases, list(NARROWED), receipt["upscale_route"], receipt["output"])
                self.assertEqual(runs["standard"], runs["fast"])

    def test_the_standard_load_leaves_nvidia_loaders_alone(self):
        status, receipt, calls, phases, _ = self.run_job(load="standard")
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.loads, self.expected_loads(fast=False))
        self.assertEqual((receipt["job"]["load"], receipt["load"]), ("standard", {"mode": "standard"}))
        self.assert_loaders(fast=False)
        self.assertNotIn("preread", receipt)
        self.assertNotIn("fast load", self.stdout)
        self.assertIn([c[0] for c in calls].index("load_model") + 1, self.emptied)
        self.assertEqual([c[0] for c in calls], ["encode_video", "Pipeline", "load_model", "upscale",
                                                 "upscaler.upscale", "set_stage", "refine", "on_step", "on_step",
                                                 "decode", "encode_mp4"])
        self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])

    def test_a_file_without_o_direct_loads_through_the_stock_loader_to_the_gpu(self):
        """fastload falls back to safetensors.torch.load_file, which is the job's own then: the stock loader (once,
        no recursion), to the GPU as the fast load wants it; the receipt counts it."""
        fallback = (job_module.CURVE_FILE, "transformer/diffusion_pytorch_model.safetensors")
        status, receipt, _, _, _ = self.run_job(fallback=fallback)
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.loads, self.expected_loads(fast=True, fallback=fallback))
        self.assertEqual(receipt["load"], {"mode": "fast", "o_direct": True, "files": 7, "fallbacks": 2})
        self.assertRegex(self.stdout, r"(?m)^\[lf-studio\] \d\d:\d\d:\d\d fast load: 7 files, 2 of them through "
                                      r"mmap$")                          # worker.log, as TROUBLESHOOTING quotes it

    def test_silence_at_4k(self):
        """The clip's video encoded, its audio not; the standard load with preread (only allowed with it)."""
        status, receipt, calls, phases, encode_audio = self.run_job(audio="silence", load="standard", preread=True,
                                                                     resolution="4k", steps=3, start_sigma=None)
        lean = self.lean
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.loads, self.expected_loads(fast=False, audio=False))
        self.assertEqual(receipt["preread"], {"files": 4, "gb": 0.0})
        self.assertIn("preread", receipt["times_s"])
        self.assertEqual(phases, ["encoding", "loading", "refine 1/3", "refine 2/3", "refine 3/3", "decoding",
                                  "saving"])
        self.assertEqual([c[0] for c in calls][:7], ["encode_video", "Pipeline", "load_model", "upscale",
                                                     "upscaler.upscale", "upscaler.upscale", "set_stage"])
        self.assertEqual(calls[3], ("upscale", (24, 37, 48, 84), "final"))
        self.assertEqual(calls[7][4], "silence")
        encode_audio.assert_not_called()
        self.assertEqual(set(receipt["times_s"]), {"encode_video", "preread", "load", "upscale", "merge_refine",
                                                   "refine", "vae_decode", "mp4"})
        self.assertEqual((receipt["sigmas"], receipt["lean_chunks"], receipt["output"]["size"]),
                         ([0.9, 0.8, 0.6, 0.0], 16, [3840, 2160]))
        self.assertEqual(lean.call_count, 1)                        # the DiT's blocks, 16 chunks, after the load
        self.assertEqual(lean.call_args.args[1], 16)
        self.assertEqual(len(lean.call_args.args[0].transformer_blocks), 3)
        self.assertEqual(set(receipt["inputs_sha256"]), {"conditioning", "final"})

    def test_the_stand_in_load_wants_weights_only(self):
        load = stand_in_torch(self.loaded).load
        self.assertIs(load(self.case / "qwen" / "conditioning.pt", map_location="cpu", weights_only=True),
                      self.loaded["conditioning.pt"])
        for kwargs in ({"map_location": "cpu"}, {"map_location": "cpu", "weights_only": False},
                       {"weights_only": True}, {"map_location": "cuda", "weights_only": True}):
            with self.assertRaises(AssertionError):
                load(self.case / "qwen" / "conditioning.pt", **kwargs)

    def test_silence_with_the_fast_load(self):
        status, receipt, _, phases, _ = self.run_job(audio="silence")
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(self.loads, self.expected_loads(fast=True, audio=False))
        self.assertEqual(receipt["load"]["files"], 6)
        self.assertEqual(phases[:2], ["encoding", "loading"])

    def test_a_draft_source_is_refused_before_anything_loads(self):
        status, receipt, calls, phases, _ = self.run_job(source="draft")
        self.assertEqual((status, receipt["state"], receipt["error"]), (2, "failed", "unknown field(s): source"))
        self.assertEqual((calls, phases, self.loads), ([], [], []))
        self.assert_loaders(fast=False)

    def test_preread_with_the_fast_load_is_refused_before_anything_loads(self):
        status, receipt, calls, phases, _ = self.run_job(preread=True)
        self.assertEqual((status, receipt["state"]), (2, "failed"))
        self.assertIn('preread: only with load "standard"', receipt["error"])
        self.assertEqual((calls, phases, self.loads), ([], [], []))

    def test_another_package_is_refused(self):
        (self.package / "open_h3" / "adapters.py").write_text(
            STAND_IN_ADAPTERS.replace("open_h3_regenerate_2k_lora", "open_h3_regenerate_2k_lora_v2"))
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"]), (2, "failed"))
        self.assertIn("not the one this wrapper was written for", receipt["error"])
        self.assertEqual((calls, phases, self.loads), ([], [], []))
        self.assert_loaders(fast=False)                             # refused before the fast load is installed

    def test_fl2va_conditioning_is_refused_before_anything_loads(self):
        self.loaded["conditioning.pt"]["text_token_tags"] = StandInTensor(41, values=[1] * 6 + [0] * 35)
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"]), (2, "failed"))
        self.assertIn("35 non-text rows", receipt["error"])
        self.assertEqual((calls, phases, self.loads), ([], [], []))
        self.assert_loaders(fast=False)


class FastLoadTests(unittest.TestCase):
    """install_fast_load with the real upscale/fastload.py, no torch: where fastload can't read with O_DIRECT it calls
    safetensors.torch.load_file, which is then the job's own; that must end in the stock loader, once."""

    def test_the_real_fastload_falls_back_to_the_stock_loader_once(self):
        loads = []
        modules = stand_in_loaders(loads)
        fastload = job_module.load_fastload()
        self.assertEqual(Path(fastload.__file__), FASTLOAD)
        self.assertIsNot(fastload, job_module.load_fastload())       # a module of its own each time, not imported
        fastload.DIRECT = False                                      # as SPARK_DIRECT_IO=0
        adapters = types.SimpleNamespace(load_file=modules["safetensors.torch"].load_file)    # imported by name
        with mock.patch.dict(sys.modules, modules):
            record = job_module.install_fast_load(adapters, fastload, modules["safetensors.torch"],
                                                  modules["diffusers"])
            self.assertEqual(adapters.load_file("/w/lora/a.safetensors", device="cpu"), {})
            self.assertEqual(modules["safetensors.torch"].load_file("/w/vae/b.safetensors"), {})
        self.assertEqual(loads, [("load_file", "lora/a.safetensors", "cuda"),
                                 ("load_file", "vae/b.safetensors", "cuda")])
        self.assertEqual(record, {"mode": "fast", "o_direct": False, "files": 2, "fallbacks": 2})

    def test_loads_from_threads(self):
        """diffusers' parallel loading (HF_ENABLE_PARALLEL_LOADING) calls load_file from threads: every file and every
        fallback counted, and each thread's fallback goes to the stock loader (the re-entry flag is per thread)."""
        loads = []
        modules = stand_in_loaders(loads)
        fallback = {f"t{n}/f{i}.safetensors" for n in range(8) for i in range(0, 300, 3)}
        record = job_module.install_fast_load(types.SimpleNamespace(), stand_in_fastload(loads, fallback),
                                              modules["safetensors.torch"], modules["diffusers"])
        start = threading.Barrier(8)

        def load(n):
            start.wait()
            for i in range(300):
                modules["safetensors.torch"].load_file(f"/w/t{n}/f{i}.safetensors")
        with mock.patch.dict(sys.modules, modules):
            threads = [threading.Thread(target=load, args=(n,)) for n in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
        self.assertEqual(record, {"mode": "fast", "o_direct": True, "files": 2400, "fallbacks": 800})
        self.assertEqual(sorted(loads), sorted([("load_file", name, "cuda") for name in fallback] +
                                               [("fastload", f"t{n}/f{i}.safetensors", "cuda") for n in range(8)
                                                for i in range(300) if f"t{n}/f{i}.safetensors" not in fallback]))

    def test_device_map_is_a_default(self):
        loads = []
        modules = stand_in_loaders(loads)
        job_module.install_fast_load(types.SimpleNamespace(), types.SimpleNamespace(ENABLED=True, DIRECT=True),
                                     modules["safetensors.torch"], modules["diffusers"])
        diffusers = modules["diffusers"]
        with tempfile.TemporaryDirectory() as empty:
            self.assertIsInstance(diffusers.MiniMaxH3Transformer3DModel.from_pretrained(empty, subfolder="t"),
                                  diffusers.MiniMaxH3Transformer3DModel)
            diffusers.AutoencoderKLMiniMaxH3.from_pretrained(empty, subfolder="v", device_map="auto")
        self.assertEqual(loads, [("from_pretrained", "MiniMaxH3Transformer3DModel", "t", {"": "cuda"}),
                                 ("from_pretrained", "AutoencoderKLMiniMaxH3", "v", "auto")])


@unittest.skipUnless(torch, "needs torch (CPU is enough)")
class TorchTests(unittest.TestCase):
    def test_one_shot_merge(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(64, 1100, bias=False), torch.nn.Linear(1100, 48, bias=False))
        model = model.to(torch.bfloat16)
        original = [layer.weight.detach().clone() for layer in model]
        pairs = {"0": (torch.randn(8, 64) * 0.02, torch.randn(1100, 8) * 0.02),
                 "1": (torch.randn(8, 1100) * 0.02, torch.randn(48, 8) * 0.02)}
        merger = job_module.OneShotMerger(model, {"refine": [(pairs, 1.0)]}, torch.device("cpu"))
        self.assertEqual(merger.merged, 2)
        merger.merge("refine")
        for index, (a, b) in enumerate((pairs["0"], pairs["1"])):
            expected = (original[index].float() + b @ a).to(torch.bfloat16)
            got = model[index].weight
            self.assertEqual(got.dtype, torch.bfloat16)
            self.assertFalse(torch.equal(got, original[index]))
            self.assertLessEqual((got.float() - expected.float()).abs().max().item(), 2 ** -7)
        with self.assertRaises(RuntimeError):
            merger.merge("hyperflow")
        with self.assertRaises(RuntimeError):
            merger.attach_quantized([])
        for stages, resident in (({"refine": [(pairs, 1.0)], "hyperflow": [(pairs, 1.0)]}, False),
                                 ({"refine": [(pairs, 1.0)]}, True)):
            with self.assertRaises(RuntimeError):
                job_module.OneShotMerger(model, stages, torch.device("cpu"), resident=resident)

    def test_one_shot_merge_skips_zero_scales(self):
        model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False)).to(torch.bfloat16)
        before = model[0].weight.detach().clone()
        pairs = {"0": (torch.randn(2, 16), torch.randn(16, 2))}
        self.assertEqual(job_module.OneShotMerger(model, {"refine": [(pairs, 0.0)]}, torch.device("cpu")).merged, 0)
        self.assertTrue(torch.equal(model[0].weight, before))

    def test_lean_forward_matches_the_stock_block(self):
        """lean_forward against a small block written as diffusers' MiniMaxH3TransformerBlock.forward and its
        attention processor (SDPA), with diffusers' rotary embedding: the same result for any number of chunks, also
        ones that do not divide the rows."""
        torch.manual_seed(0)
        nn, F = torch.nn, torch.nn.functional
        hidden, heads, dim, table = 32, 4, 8, 6

        def rotary(x, cos, sin):                         # diffusers' _apply_rotary_emb
            width = cos.shape[-1]
            turned, passed = x[..., :width], x[..., width:]
            cos, sin = cos.to(x.dtype)[None, :, None, :], sin.to(x.dtype)[None, :, None, :]
            x1, x2 = turned.chunk(2, dim=-1)
            return torch.cat((turned * cos + torch.cat((-x2, x1), dim=-1) * sin, passed), dim=-1).contiguous()

        class Modulation(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(6 * hidden, 6 * hidden)

            def forward(self, temb):                     # six (table rows, hidden) tensors
                return self.linear(temb).chunk(6, dim=-1)

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads, self.head_dim = heads, dim
                self.to_q = nn.Linear(hidden, heads * dim, bias=False)
                self.to_k = nn.Linear(hidden, heads * dim, bias=False)
                self.to_v = nn.Linear(hidden, heads * dim, bias=False)
                self.norm_q, self.norm_k = nn.RMSNorm(dim), nn.RMSNorm(dim)
                self.to_out = nn.ModuleList([nn.Linear(heads * dim, hidden, bias=False), nn.Dropout(0.0)])

            def forward(self, x, rotary_emb, attention_mask=None):      # MiniMaxH3AttnProcessor
                q = rotary(self.norm_q(self.to_q(x).unflatten(-1, (heads, -1))), *rotary_emb)
                k = rotary(self.norm_k(self.to_k(x).unflatten(-1, (heads, -1))), *rotary_emb)
                v = self.to_v(x).unflatten(-1, (heads, -1))
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                return self.to_out[1](self.to_out[0](out.transpose(1, 2).flatten(2, 3).type_as(q)))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1, self.norm2 = nn.RMSNorm(hidden), nn.RMSNorm(hidden)
                self.attn = Attention()
                self.ff = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, hidden))
                self.adaln_proj = Modulation()

            def forward(self, x, temb, index, rotary_emb, attention_mask=None):     # the stock block forward
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
                y = self.norm1(x) * (1.0 + scale_msa.index_select(0, index)) + shift_msa.index_select(0, index)
                x = x + gate_msa.index_select(0, index) * self.attn(y, rotary_emb, attention_mask)
                y = self.norm2(x) * (1.0 + scale_mlp.index_select(0, index)) + shift_mlp.index_select(0, index)
                return x + gate_mlp.index_select(0, index) * self.ff(y)

        block = Block().eval()
        temb = torch.randn(table, 6 * hidden)
        for rows in (37, 64):
            x = torch.randn(1, rows, hidden)
            index = torch.randint(0, table, (rows,))
            angles = torch.rand(rows, 3) * 6.28          # the rotary embedding turns 6 of each head's 8 channels
            cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)
            with torch.no_grad():
                want = block(x, temb, index, (cos, sin))
                for chunks in (1, 4, 16):
                    got = job_module.lean_forward(block, x, temb, index, (cos, sin), chunks=chunks, rotary=rotary)
                    self.assertEqual(got.shape, want.shape)
                    self.assertTrue(torch.allclose(got, want, atol=1e-5), (rows, chunks))
        with self.assertRaisesRegex(RuntimeError, "unmasked"):
            job_module.lean_forward(block, x, temb, index, (cos, sin), torch.ones(rows, rows, dtype=torch.bool),
                                    chunks=4, rotary=rotary)

    def test_tensor_inputs(self):
        cond = {"prompt_embeds": torch.randn(1, 5, 5120), "text_token_tags": torch.ones(5, dtype=torch.int64),
                "task": "t2va", "prompt": "a test"}
        text = job_module.text_input(cond, torch)
        self.assertEqual((tuple(text["prompt_embeds"].shape), text["prompt_embeds"].dtype), ((5, 5120), torch.bfloat16))
        self.assertEqual(text["text_token_tags"].dtype, torch.int64)
        cond["prompt_embeds"][0, 1, 2] = float("inf")
        with self.assertRaisesRegex(JobError, "not finite"):
            job_module.text_input(cond, torch)
        cond["text_token_tags"][3] = 0
        with self.assertRaisesRegex(JobError, "non-text"):
            job_module.text_input(cond, torch)
        pcm = job_module.pcm_input({"audio": torch.zeros(1, 2, 161333)}, torch)
        self.assertEqual(tuple(pcm.shape), (2, 161333))
        with self.assertRaisesRegex(JobError, "no audio"):
            job_module.pcm_input({"latent": torch.zeros(1)}, torch)

    def test_the_column_crop_is_nvidias_row_crop_turned(self):
        """keep_columns on decoded frames is NVIDIA's decode crop (pipeline.py:265-266) of the frames turned, and
        the bytes encode_mp4 writes (.cpu().numpy().tobytes()) are those of a plain copy."""
        frames = torch.randint(0, 256, (3, 24, 17, 3), dtype=torch.uint8)       # 17 columns, the centre 15 kept
        kept = job_module.keep_columns(frames, 15)
        turned = frames.transpose(1, 2)
        top = (turned.shape[1] - 15) // 2
        self.assertTrue(torch.equal(kept.transpose(1, 2), turned[:, top:top + 15]))
        self.assertTrue(torch.equal(kept, frames[:, :, 1:16]))
        if importlib.util.find_spec("numpy"):                      # as encode_mp4 hands them to ffmpeg
            self.assertEqual(kept.cpu().numpy().tobytes(), frames[:, :, 1:16].contiguous().numpy().tobytes())


def has_modules(*names):
    return all(importlib.util.find_spec(name) for name in names)


_NVIDIA = {}


def nvidia_modules():
    """NVIDIA's real open_h3 (where LFSTUDIO_UPSCALE_PKG names it), imported once on the CPU and then kept out of
    sys.modules (RunTests import stand-ins of it under the same name): {name: module}, empty without the package."""
    if not _NVIDIA:
        package = nvidia_package()
        if package is None:
            return {}
        sys.path.insert(0, str(package))
        try:
            import diffusers.models.attention_dispatch  # noqa: F401 - imported here, not inside a test
            import diffusers.modular_pipelines.minimax_h3.encoders  # noqa: F401
            import open_h3.geometry  # noqa: F401
            import open_h3.parallel  # noqa: F401
            import open_h3.pipeline  # noqa: F401
            import open_h3.sigmas  # noqa: F401
        finally:
            sys.path.remove(str(package))
        _NVIDIA.update({name: sys.modules.pop(name) for name in list(sys.modules)
                        if name == "open_h3" or name.startswith("open_h3.")})
    return _NVIDIA


@contextlib.contextmanager
def nvidia_open_h3():
    """NVIDIA's open_h3 in sys.modules for a test (its functions import from it at call time): its pipeline."""
    modules = nvidia_modules()
    sys.modules.update(modules)
    try:
        yield modules["open_h3.pipeline"]
    finally:
        for name in [name for name in sys.modules if name == "open_h3" or name.startswith("open_h3.")]:
            del sys.modules[name]


@contextlib.contextmanager
def nvidia_on_the_cpu(vae_class):
    """NVIDIA's encode_video and decode, written for CUDA, on the CPU: Tensor.cuda() is the tensor itself,
    torch.tensor(device="cuda") is made on the CPU, and diffusers' AutoencoderKLMiniMaxH3 is `vae_class`."""
    import diffusers
    real = torch.tensor

    def tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real(*args, **kwargs)
    with mock.patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self), \
            mock.patch.object(torch, "tensor", tensor), \
            mock.patch.object(diffusers, "AutoencoderKLMiniMaxH3", vae_class):
        yield


SPANS = (1, 4, 4, 4, 4)              # the pixel frames of each latent frame, as H3's video VAE groups them


def frame_groups(frames):
    """The pixel frames of each latent frame of `frames` (H3's 17 per 5 latent frames: 124 -> 37 groups)."""
    groups, start = [], 0
    while start < frames:
        span = SPANS[len(groups) % len(SPANS)]
        groups.append(list(range(start, min(start + span, frames))))
        start += span
    return groups


def stand_in_vae(record):
    """A class standing in for diffusers' AutoencoderKLMiniMaxH3, small and deterministic (its weights from a fixed
    seed), so NVIDIA's code and the job's copies of it run on the CPU and can be compared. encode: each latent frame
    the mean of its pixel frames (H3's grouping) over 16x16 blocks, its 3 channels mixed into 24; the posterior's
    sample adds noise from the generator it is given. decode: each latent frame back over its pixel frames (each a
    little different), 16x16 blocks, 24 channels mixed into 3. `record` gets ("load", kwargs), ("tiling", kwargs),
    ("encode", its input) and ("decode", its input)."""
    class Posterior:
        def __init__(self, mean):
            self.mean = mean

        def sample(self, generator=None):
            noise = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
            return self.mean + 0.1 * noise.to(self.mean.device)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weights = torch.Generator().manual_seed(7)
            self.mix = torch.nn.Parameter(torch.randn(24, 3, generator=weights), requires_grad=False)
            self.unmix = torch.nn.Parameter(torch.randn(3, 24, generator=weights) * 0.2, requires_grad=False)
            self.config = types.SimpleNamespace(latents_mean=[0.01 * i for i in range(24)],
                                                latents_std=[1 + 0.02 * i for i in range(24)])

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            record.append(("load", kwargs))
            return cls().to(kwargs.get("torch_dtype", torch.float32))

        def cuda(self):                      # NVIDIA's code moves it to the GPU: here it stays on the CPU
            return self

        def enable_tiling(self, **kwargs):
            record.append(("tiling", kwargs))

        def encode(self, x, return_dict=True):
            record.append(("encode", x.detach().clone()))
            frames = torch.stack([x[:, :, group].mean(2) for group in frame_groups(x.shape[2])], 2)
            pooled = torch.nn.functional.avg_pool3d(frames.float(), (1, 16, 16))
            return (Posterior(torch.einsum("oc,bcthw->bothw", self.mix.float(), pooled)),)

        def decode(self, z, return_dict=True):
            record.append(("decode", z.detach().clone()))
            groups = frame_groups(17 * ((z.shape[2] - 2) // 5) + 5)
            rgb = torch.einsum("co,bothw->bcthw", self.unmix.float(), z.float())
            rgb = rgb.repeat_interleave(16, -1).repeat_interleave(16, -2)
            frames = [rgb[:, :, k:k + 1] + 0.05 * torch.arange(len(group)).view(1, 1, -1, 1, 1)
                      for k, group in enumerate(groups)]
            return (torch.cat(frames, 2).to(z.dtype),)
    return VAE


def moving_mp4(path, frames, size="128x72"):
    """A test pattern that changes every frame (ffmpeg's testsrc2), no audio."""
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24",
                    "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


@unittest.skipUnless(torch is not None and has_modules("numpy", "diffusers", "imageio_ffmpeg"),
                     "needs torch, numpy, diffusers and imageio-ffmpeg (the Regen-2K environment; CPU is enough)")
class OwnCopyTests(unittest.TestCase):
    """encode_clip and decode_released, the job's copies of NVIDIA's encode_video and _decode_released with the frames
    as parameters, against NVIDIA's own run on the CPU with a stand-in VAE (where LFSTUDIO_UPSCALE_PKG names NVIDIA's
    package): 121 frames give NVIDIA's latent and frames bit for bit, 124 read and keep the clip's own frames."""

    @classmethod
    def setUpClass(cls):
        cls.media = Path(tempfile.mkdtemp(prefix="upscale-own-"))
        cls.five = moving_mp4(cls.media / "five.mp4", 121)
        cls.whole = moving_mp4(cls.media / "whole.mp4", 124)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.media, ignore_errors=True)

    def setUp(self):
        self.record = []
        self.vae = stand_in_vae(self.record)

    def encode(self, mp4, kept, pad, keyframes=()):
        with nvidia_on_the_cpu(self.vae):
            return job_module.encode_clip(Path("/base"), mp4, (64, 32), kept, pad, 42, torch, keyframes=keyframes,
                                          encode_seed=42, device="cpu")

    def read(self, mp4, frames):
        """Every frame of `mp4` as ffmpeg gives it at 64x32 (the same scale and crop, no tpad): (frames, 32, 64, 3)."""
        import imageio_ffmpeg
        import numpy as np
        raw = subprocess.run(job_module.clip_read_command(mp4, (64, 32), frames, 0, imageio_ffmpeg.get_ffmpeg_exe()),
                             capture_output=True, check=True).stdout
        return torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(frames, 32, 64, 3).copy())

    def test_121_frames_are_nvidias_encode(self):
        """encode_clip(121, 3) gives NVIDIA's encode_video's latent bit for bit, from the same VAE input (frames 121-123
        copies of frame 120), the same load and tiling."""
        if not nvidia_modules():
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")
        with nvidia_open_h3() as pipeline, nvidia_on_the_cpu(self.vae), \
                mock.patch.dict(pipeline.INPUT_SIZES, {"test": (64, 32)}), \
                mock.patch.dict(pipeline.NATIVE_SHAPES, {"test": (24, 37, 2, 4)}):
            want = pipeline.encode_video(Path("/base"), self.five, "test", seed=42)
            nvidia, self.record[:] = list(self.record), []
            got, keyframes, _ = self.encode(self.five, 121, 3)
        self.assertEqual((tuple(got.shape), got.dtype, keyframes), ((24, 37, 2, 4), torch.float32, {}))
        self.assertTrue(torch.equal(got, want))
        self.assertEqual([kind for kind, _ in self.record], ["load", "tiling", "encode"])
        for (kind, mine), (_, theirs) in zip(self.record, nvidia):
            if kind == "encode":
                self.assertTrue(torch.equal(mine, theirs))
            else:
                self.assertEqual(mine, theirs, kind)
        pixels = self.record[2][1]
        self.assertEqual(tuple(pixels.shape), (1, 3, 124, 32, 64))
        for copy in (121, 122, 123):
            self.assertTrue(torch.equal(pixels[:, :, copy], pixels[:, :, 120]), copy)

    def test_124_frames_are_the_clips_own(self):
        """encode_clip(124, 0) reads the clip's 124 frames as they are (its last 3 its own, not copies of frame 120)
        into the VAE's 124: the same 37 latent frames."""
        got, _, _ = self.encode(self.whole, 124, 0)
        self.assertEqual(tuple(got.shape), (24, 37, 2, 4))
        pixels = self.record[-1][1]
        self.assertEqual(tuple(pixels.shape), (1, 3, 124, 32, 64))
        mean = torch.tensor(job_module.PIXEL_MEAN).view(1, 3, 1, 1, 1)
        std = torch.tensor(job_module.PIXEL_STD).view(1, 3, 1, 1, 1)
        read = self.read(self.whole, 124).permute(3, 0, 1, 2)[None].float() / 255.0
        self.assertTrue(torch.equal(pixels, (read - mean) / std))
        for frame in (121, 122, 123):
            self.assertFalse(torch.equal(pixels[:, :, frame], pixels[:, :, 120]), frame)
        # the same clip cut to 121 frames: NVIDIA's three copies of frame 120 instead, the same first 121 frames
        self.encode(self.whole, 121, 3)
        self.assertTrue(torch.equal(self.record[-1][1][:, :, :121], pixels[:, :, :121]))
        self.assertTrue(torch.equal(self.record[-1][1][:, :, 123], pixels[:, :, 120]))

    def test_a_short_clip_fails_the_length_check(self):
        """Without tpad a 121-frame MP4 read for 124 frames reads short: refused, not padded."""
        with self.assertRaisesRegex(ValueError, r"could not decode .*five\.mp4 into 124 frames at 64x32"):
            self.encode(self.five, 124, 0)

    def test_keyframes_on_the_same_vae(self):
        """The pin's keyframes go through the VAE the clip's encode loaded (one load), after the clip, each as H3
        encodes a keyframe (diffusers' encode_vae_condition)."""
        import numpy as np
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        rng = np.random.default_rng(3)
        first, last = (rng.integers(0, 256, (32, 64, 3), dtype=np.uint8) for _ in range(2))
        native, keyframes, seconds = self.encode(self.whole, 124, 0, [("first", first), ("last", last)])
        self.assertEqual([kind for kind, _ in self.record], ["load", "tiling", "encode", "encode", "encode"])
        self.assertEqual(tuple(self.record[2][1].shape), (1, 3, 124, 32, 64))
        self.assertEqual(list(keyframes), ["first", "last"])
        self.assertGreaterEqual(seconds, 0)
        for anchor, pixels in (("first", first), ("last", last)):
            want = encode_vae_condition(self.vae(), torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None],
                                        job_module.PIXEL_MEAN, job_module.PIXEL_STD, encode_seed=42)
            self.assertEqual(tuple(keyframes[anchor].shape), (1, 24, 1, 2, 4))
            self.assertTrue(torch.equal(keyframes[anchor], want), anchor)
        self.assertFalse(torch.equal(keyframes["first"], keyframes["last"]))

    def test_the_decode(self):
        """decode_released is NVIDIA's _decode_released bit for bit at 121 frames; with 124 it keeps all of them (the
        first 121 NVIDIA's), and decode crops NVIDIA's centre rows. A 121-frame clip goes to NVIDIA's own decode."""
        if not nvidia_modules():
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")
        latents = torch.randn(24, 37, 2, 4, generator=torch.Generator().manual_seed(5))
        with nvidia_open_h3() as pipeline, nvidia_on_the_cpu(self.vae):
            pipe = object.__new__(pipeline.Pipeline)
            pipe.vae, pipe.fast_vae, pipe.keep_rows = self.vae().to(torch.bfloat16), None, 24
            want = pipeline.Pipeline.decode(pipe, latents)
            released = pipe._decode_released(latents)
            self.assertTrue(torch.equal(job_module.decode_released(pipe.vae, latents, 121, torch, "cpu"), released))
            got = job_module.decode(pipe, latents, 124, torch, device="cpu")
            with mock.patch.object(pipe, "decode", wraps=pipe.decode) as nvidia:
                self.assertTrue(torch.equal(job_module.decode(pipe, latents, 121, torch, device="cpu"), want))
            nvidia.assert_called_once_with(latents)
            pipe.fast_vae = object()
            with self.assertRaisesRegex(RuntimeError, "fast VAE"):
                job_module.decode(pipe, latents, 124, torch, device="cpu")
        self.assertEqual((tuple(want.shape), tuple(got.shape), got.dtype), ((121, 24, 64, 3), (124, 24, 64, 3),
                                                                             torch.uint8))
        self.assertTrue(torch.equal(got[:121], want))
        full = job_module.decode_released(pipe.vae, latents, 124, torch, "cpu")
        self.assertTrue(torch.equal(got, full[:, 4:28]))                   # the centre 24 of 32 rows
        self.assertFalse(torch.equal(got[123], got[120]))


class StandInTransformer:
    """H3's DiT as a fixed function of everything it is given: each video row's velocity from its own latent, its
    timestep and its rotary position, the mean of every video row (as attention mixes them, so the keyframe rows
    reach the generated ones), the audio, the prompt and the text's tags. `poison` leading video rows get NaN (rows
    that must never be stepped). Each call's inputs and output in `calls`."""

    def __init__(self, poison=0):
        self.transformer_blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(
            processor=types.SimpleNamespace(valid_rows=None))) for _ in range(2)]
        self.poison, self.calls = poison, []

    def __call__(self, *, hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                 token_tags, position_ids, video_indices, audio_indices, text_indices, attention_kwargs, return_dict):
        x = hidden_states.float()
        t = timestep[timestep_indices][video_indices].float().view(1, -1, 1)
        where = position_ids[video_indices].float()[None]
        context = (audio_hidden_states.float().mean() + encoder_hidden_states.float().mean()
                   + 0.1 * token_tags[text_indices].float().mean() + 0.001 * position_ids[audio_indices].float().mean())
        velocity = torch.tanh(0.5 * x + 0.3 * t + 0.01 * where[..., :1] + 0.05 * (where[..., 1:2] - where[..., 2:3])
                              + 0.2 * x.mean(1, keepdim=True) + 0.1 * context)
        if self.poison:
            velocity[:, :self.poison] = float("nan")
        velocity = velocity.to(torch.bfloat16)
        self.calls.append({"hidden": hidden_states.clone(), "times": timestep[timestep_indices][video_indices].clone(),
                           "velocity": velocity.clone(), "valid_rows": self.transformer_blocks[0].attn.processor
                           .valid_rows})
        return velocity, None


def bench_k1(pipe, low, text, seed, keyframe, noise_aug=0.999, noise_seed=42):
    """The bench wrapper's K1 refine, as the research runs held a first frame with it: its refine_keyframe with mode
    "prefix" and the layout its keyframe_layout builds for it, copied here (the bench is not part of this repo)."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       patchify_video_latents)
    from open_h3.geometry import (H3_AUDIO_TAG, H3_VIDEO_TAG, MiniMaxH3Geometry, MiniMaxH3PackedLayout,
                                  build_row_timestep_plan, pack_audio_latents, unpatchify_video_rows)
    device = pipe.device
    geometry = MiniMaxH3Geometry(num_latent_frames=low.shape[1], latent_height=low.shape[2],
                                 latent_width=low.shape[3], num_audio_latents=pipe.audio.shape[-1])
    rows = geometry.rows_per_video_frame
    tags = torch.as_tensor(text["text_token_tags"], dtype=torch.long).flatten().cpu()
    layout = MiniMaxH3PackedLayout(*MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=tags, num_latent_frames=geometry.num_latent_frames, latent_height=geometry.latent_height,
        latent_width=geometry.latent_width, num_audio_latents=geometry.num_audio_latents,
        patch_size=geometry.patch_size, audio_channels=geometry.audio_channels, audio_tag=H3_AUDIO_TAG,
        video_tag=H3_VIDEO_TAG, keyframe_anchors=("first",)))
    clean = keyframe.detach().float().cpu()
    eps = torch.randn(clean.shape, generator=torch.Generator().manual_seed(noise_seed), dtype=torch.float32)
    t = torch.tensor(noise_aug, dtype=torch.float32)
    noised = t * clean + (1.0 - t) * eps
    condition = patchify_video_latents(noised, geometry.patch_size).reshape(1, rows, -1).to(device)
    layout = layout.to(device)
    audio = pack_audio_latents(pipe.audio[None]).to(device)
    prompt = text["prompt_embeds"][None].to(device)
    low = patchify_video_latents(low[None], geometry.patch_size).reshape(
        1, geometry.num_video_rows, geometry.video_row_dim).to(device, torch.float32)
    for block in pipe.model.transformer_blocks:
        block.attn.processor.valid_rows = layout.token_tags.numel()
    with torch.inference_mode():
        noise = torch.randn(low.shape, device=device, dtype=torch.float32,
                            generator=torch.Generator(device=device).manual_seed(seed))
        sigmas = pipe.sigmas
        state = (1 - sigmas[0]) * low + sigmas[0] * noise
        for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            pipe.on_step("refine", index)
            times, indices = build_row_timestep_plan(layout, video_sigma=sigma, audio_sigma=0.,
                                                     keyframe_noise_aug=noise_aug)
            video = torch.cat([condition, state], 1)
            velocity, _ = pipe.model(
                hidden_states=video.to(torch.bfloat16), audio_hidden_states=audio, encoder_hidden_states=prompt,
                timestep=times, timestep_indices=indices, token_tags=layout.token_tags,
                position_ids=layout.position_ids, video_indices=layout.video_indices,
                audio_indices=layout.audio_indices, text_indices=layout.text_indices, attention_kwargs={},
                return_dict=False)
            state = state + (sigma - following) * velocity[:, rows:].float()
    return unpatchify_video_rows(state, geometry)[0]


def h3_geometry(frames, height, width, audio=207):
    from open_h3.geometry import MiniMaxH3Geometry
    return MiniMaxH3Geometry(num_latent_frames=frames, latent_height=height, latent_width=width,
                             num_audio_latents=audio)


@unittest.skipUnless(torch is not None and has_modules("numpy", "diffusers"),
                     "needs torch, numpy and diffusers (the Regen-2K environment; CPU is enough)")
class PinLayoutTests(unittest.TestCase):
    """pin_layout and check_pin_layout with diffusers' build_packed_sequence and NVIDIA's t2va layout (where
    LFSTUDIO_UPSCALE_PKG names NVIDIA's package): the rows and the keyframe blocks' positions at the real sizes."""

    def setUp(self):
        if not nvidia_modules():
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")
        stack = contextlib.ExitStack()
        stack.enter_context(nvidia_open_h3())
        self.addCleanup(stack.close)
        self.tags = torch.ones(196, dtype=torch.long)                 # a prompt of 196 text rows (L)

    def layout(self, anchors, geometry, made=124):
        layout, t2va = job_module.pin_layout(self.tags, geometry, anchors)
        return layout, t2va, job_module.check_pin_layout(layout, t2va, geometry, anchors, made)

    def test_the_rows_at_1080p_and_2k(self):
        """1080p (R 2,040 rows a frame) and 2K (R 3,600): t2va's rows, + R per end, and the condition rows R per
        end; the first block at frame 0's time (L), the last at L + 205 (frame 123 of the 124 made), the same grid as
        frame 0 and the video tag; every generated, audio and text row where t2va has it."""
        for (height, width), rows, sequences in (((68, 120), 2040, (76090, 78130, 80170)),
                                                 ((90, 160), 3600, (133810, 137410, 141010))):
            geometry = h3_geometry(37, height, width)
            for anchors in ((), ("first",), ("last",), ("first", "last")):
                with self.subTest(grid=(height, width), anchors=anchors):
                    layout, t2va, facts = self.layout(anchors, geometry)
                    self.assertEqual((t2va.sequence_length, layout.sequence_length, layout.num_condition_video_rows),
                                     (sequences[0], sequences[len(anchors)], rows * len(anchors)))
                    self.assertEqual((facts["rows_per_frame"], facts["sequence_rows"], facts["condition_rows"]),
                                     (rows, sequences[len(anchors)], rows * len(anchors)))
                    times = {"first": 0.0, "last": 205.0}
                    self.assertEqual(facts["blocks"], [{"anchor": a, "time": times[a]} for a in anchors])
                    n = rows * len(anchors)
                    positions, video = layout.position_ids, layout.video_indices
                    self.assertTrue(torch.equal(positions[video[n:]], t2va.position_ids[t2va.video_indices]))
                    self.assertTrue(torch.equal(positions[layout.audio_indices], t2va.position_ids[t2va.audio_indices]))
                    self.assertTrue(torch.equal(positions[layout.text_indices], t2va.position_ids[t2va.text_indices]))
                    for index, anchor in enumerate(anchors):
                        block = positions[video[index * rows:(index + 1) * rows]]
                        self.assertEqual(float(block[0, 0]), {"first": 196.0, "last": 400.99999999999994}[anchor])
                        self.assertTrue(torch.equal(block[:, 1:], positions[video[n:n + rows], 1:]))
                        self.assertTrue(bool((layout.token_tags[video[index * rows:(index + 1) * rows]] == 0).all()))
                    checks = {k: v for k, v in facts.items() if isinstance(v, bool)}
                    self.assertEqual(set(checks), {"r_per_anchor", "video_as_t2va", "audio_as_t2va", "text_as_t2va"} |
                                     {f"{a}_{c}" for a in anchors for c in ("at_frame0_grid", "tag_video", "time")})
                    self.assertTrue(all(checks.values()), checks)
        layout, t2va = job_module.pin_layout(self.tags, h3_geometry(37, 68, 120), ())
        self.assertIs(layout, t2va)

    def test_the_last_block_at_other_lengths(self):
        """The last block at (5/3)(made - 1) after the text for the other lengths H3 makes (158, 243, 362 frames;
        diffusers sums the spans with numpy, as the reference does: these are its values)."""
        for made, frames, time in ((124, 37, 400.99999999999994), (158, 47, 457.66666666666663),
                                   (243, 72, 599.3333333333334), (362, 107, 797.6666666666665)):
            with self.subTest(made=made):
                layout, t2va, facts = self.layout(("first", "last"), h3_geometry(frames, 4, 6), made)
                self.assertEqual(float(layout.position_ids[layout.video_indices[6], 0]), time)
                self.assertLess(abs(time - 196 - 5 / 3 * (made - 1)), 1e-9)
                self.assertEqual(facts["blocks"][1], {"anchor": "last", "time": round(5 / 3 * (made - 1), 4)})
                self.assertEqual(facts["made"], made)

    def test_a_wrong_layout_is_refused(self):
        geometry = h3_geometry(37, 4, 6)
        layout, t2va = job_module.pin_layout(self.tags, geometry, ("first", "last"))
        with self.assertRaisesRegex(RuntimeError, "158 frames made are not 37 latent frames"):
            job_module.check_pin_layout(layout, t2va, geometry, ("first", "last"), 158)
        with self.assertRaisesRegex(RuntimeError, r"pin layout check failed: .*'last_time': False"):
            job_module.check_pin_layout(layout, t2va, geometry, ("last", "first"), 124)
        with self.assertRaisesRegex(RuntimeError, r"'r_per_anchor': False"):
            job_module.check_pin_layout(layout, t2va, geometry, ("first",), 124)


@unittest.skipUnless(torch is not None and has_modules("numpy", "diffusers"),
                     "needs torch, numpy and diffusers (the Regen-2K environment; CPU is enough)")
class PinRefineTests(unittest.TestCase):
    """refine_pinned on the CPU with a stand-in transformer (a fixed function of its inputs) and NVIDIA's own code
    where LFSTUDIO_UPSCALE_PKG names its package: no keyframe is NVIDIA's Pipeline.refine bit for bit, a first frame
    the bench wrapper's K1 bit for bit, and with both ends the first block unchanged, the condition rows never
    written, only the rows after them stepped, the timesteps H3's."""

    def setUp(self):
        if not nvidia_modules():
            self.skipTest("NVIDIA's Open-H3-Regenerate-2K is not installed (LFSTUDIO_UPSCALE_PKG)")
        stack = contextlib.ExitStack()
        self.pipeline = stack.enter_context(nvidia_open_h3())
        self.addCleanup(stack.close)
        draw = torch.Generator().manual_seed(11)
        self.low = torch.randn(24, 37, 4, 6, generator=draw)               # R = 2 x 3 = 6 rows a frame
        self.text = {"prompt_embeds": torch.randn(5, 16, generator=draw).to(torch.bfloat16),
                     "text_token_tags": torch.ones(5, dtype=torch.long)}
        self.audio = torch.randn(2, 32, 207, generator=draw).to(torch.bfloat16)
        self.first, self.last = (torch.randn(1, 24, 1, 4, 6, generator=draw) for _ in range(2))
        self.sigmas = sys.modules["open_h3.sigmas"].shifted_grid(0.7, 2)  # balanced: 2 steps from 0.7

    def pipe(self, poison=0):
        """NVIDIA's Pipeline without its __init__ (no weights): on the CPU, the stand-in transformer."""
        pipe = object.__new__(self.pipeline.Pipeline)
        pipe.device, pipe.sigmas, pipe.audio = torch.device("cpu"), list(self.sigmas), self.audio
        pipe.model, pipe.fwd = StandInTransformer(poison), types.SimpleNamespace(plan=None, graphs=None)
        return pipe

    def same_calls(self, a, b):
        self.assertEqual(len(a.model.calls), len(b.model.calls))
        for one, other in zip(a.model.calls, b.model.calls):
            for key in ("hidden", "times", "velocity"):
                self.assertTrue(torch.equal(one[key], other[key]), key)
            self.assertEqual(one["valid_rows"], other["valid_rows"])

    def test_no_keyframe_is_nvidias_refine(self):
        nvidia, mine = self.pipe(), self.pipe()
        want = self.pipeline.Pipeline.refine(nvidia, self.low, self.text, 1234)
        got, facts = job_module.refine_pinned(mine, self.low, self.text, 1234, [], 124)
        self.assertTrue(torch.equal(got, want))
        self.same_calls(nvidia, mine)
        self.assertEqual((facts["anchors"], facts["condition_rows"], facts["blocks"]), ([], 0, []))
        self.assertEqual([[round(t, 3) for t in step] for step in facts["timesteps"]], [[0.3, 1.0], [0.506, 1.0]])
        self.assertNotIn("condition_sha256", facts)

    def test_a_first_frame_is_the_bench_k1(self):
        bench, mine = self.pipe(), self.pipe()
        want = bench_k1(bench, self.low, self.text, 1234, self.first)
        got, facts = job_module.refine_pinned(mine, self.low, self.text, 1234, [("first", self.first)], 124)
        self.assertTrue(torch.equal(got, want))
        self.same_calls(bench, mine)
        self.assertEqual((facts["anchors"], facts["sequence_rows"], facts["t2va_rows"], facts["condition_rows"],
                          facts["blocks"]), (["first"], 5 + 414 + 37 * 6 + 6, 5 + 414 + 37 * 6, 6,
                                             [{"anchor": "first", "time": 0.0}]))
        self.assertEqual((facts["noise_aug"], facts["noise_seed"]), (0.999, 42))
        self.assertRegex(facts["condition_sha256"], "^[0-9a-f]{64}$")
        # NVIDIA's refine differs: the keyframe's rows change what the model gives the generated ones
        self.assertFalse(torch.equal(got, self.pipeline.Pipeline.refine(self.pipe(), self.low, self.text, 1234)))

    def test_both_ends(self):
        """First and last: the first block is the same draw as with the first alone (eps from one CPU generator,
        first then last), the last block the generator's second draw on the last latent; the condition rows are
        the same in every step and never written, and only the rows after them are stepped (poisoned condition
        velocities change nothing); the timesteps hold them at 0.999."""
        rows = 6
        alone, both, poisoned = self.pipe(), self.pipe(), self.pipe(poison=2 * rows)
        job_module.refine_pinned(alone, self.low, self.text, 1234, [("first", self.first)], 124)
        got, facts = job_module.refine_pinned(both, self.low, self.text, 1234, [("first", self.first),
                                                                                ("last", self.last)], 124)
        again, _ = job_module.refine_pinned(poisoned, self.low, self.text, 1234, [("first", self.first),
                                                                                  ("last", self.last)], 124)
        self.assertTrue(torch.equal(got, again))
        self.assertTrue(bool(torch.isfinite(again).all()))
        first = alone.model.calls[0]["hidden"][:, :rows]
        self.assertTrue(torch.equal(both.model.calls[0]["hidden"][:, :rows], first))
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        draw = torch.Generator().manual_seed(42)
        t = torch.tensor(0.999, dtype=torch.float32)
        blocks = []
        for latent in (self.first, self.last):
            eps = torch.randn(latent.shape, generator=draw, dtype=torch.float32)
            blocks.append(patchify_video_latents(t * latent + (1.0 - t) * eps, (1, 2, 2)).reshape(1, rows, -1))
        condition = torch.cat(blocks, 1).to(torch.bfloat16)
        for call in both.model.calls:                                # never written: the same rows every step
            self.assertTrue(torch.equal(call["hidden"][:, :2 * rows], condition))
            self.assertTrue(bool((call["times"][:2 * rows] == torch.tensor(0.999)).all()))
        # the step: NVIDIA's start, then only the rows after the 2 R condition rows move, by their own velocity
        low = patchify_video_latents(self.low[None], (1, 2, 2)).reshape(1, 37 * rows, -1)
        noise = torch.randn(low.shape, dtype=torch.float32, generator=torch.Generator().manual_seed(1234))
        state = (1 - self.sigmas[0]) * low + self.sigmas[0] * noise
        self.assertTrue(torch.equal(both.model.calls[0]["hidden"][:, 2 * rows:], state.to(torch.bfloat16)))
        state = state + (self.sigmas[0] - self.sigmas[1]) * both.model.calls[0]["velocity"][:, 2 * rows:].float()
        self.assertTrue(torch.equal(both.model.calls[1]["hidden"][:, 2 * rows:], state.to(torch.bfloat16)))
        self.assertTrue(bool((both.model.calls[0]["times"][2 * rows:] == torch.tensor(1 - self.sigmas[0])).all()))
        self.assertEqual([[round(t, 3) for t in step] for step in facts["timesteps"]],
                         [[0.3, 0.999, 1.0], [0.506, 0.999, 1.0]])
        self.assertEqual((facts["blocks"], facts["condition_rows"]),
                         ([{"anchor": "first", "time": 0.0}, {"anchor": "last", "time": 205.0}], 2 * rows))
        self.assertEqual(both.model.calls[0]["valid_rows"], 5 + 414 + 37 * rows + 2 * rows)
        last, _ = job_module.refine_pinned(self.pipe(), self.low, self.text, 1234, [("last", self.last)], 124)
        self.assertFalse(torch.equal(last, got))

    def test_a_latent_of_another_grid_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"the last keyframe's latent is \(1, 24, 1, 8, 6\), the refine's "
                                                r"\(1, 24, 1, 4, 6\)"):
            job_module.refine_pinned(self.pipe(), self.low, self.text, 1234, [("first", self.first),
                                                                              ("last", torch.zeros(1, 24, 1, 8, 6))],
                                     124)


if __name__ == "__main__":
    unittest.main()
