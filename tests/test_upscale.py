"""upscale/regen2k_job.py (the experimental Upscale's GPU job) on the CPU: job.json, the clip's and the weights'
files (5 s MP4s made by ffmpeg, 16:9 and 9:16, and the lengths and sizes it refuses), the shape table, the portrait
entries for NVIDIA's dicts, the upscale route and the column crop, check_package, the text check on a stand-in, the
command lines and the environment, the progress lines, the memory trace, the script's --check and its refusals,
main's receipt when the GPU part finishes, fails or gets SIGTERM (a stand-in run()), and run() itself with a stand-in
torch and a stand-in of NVIDIA's open_h3 that records its calls: for 16:9 the same calls as before the job took 9:16
clips, for 9:16 the portrait route, with both loads (fast and standard; stand-ins for safetensors, diffusers and
upscale/fastload.py record every load). No GPU, no weights: NVIDIA's package is a stand-in (open_h3/sigmas.py, and for
run() adapters, pipeline and upscaler), the weights are small files. upscale/fastload.py is checked against the
patch's runtime/fastload.py. With NVIDIA's real package where LFSTUDIO_UPSCALE_PKG (or its default) names it, its
pinned files and its entries are checked too (read, not imported); skipped without it.

The parts that need torch (the merger, the lean block forward, the tensor checks, the column crop) run on the CPU when
torch is importable and are skipped otherwise. python3 -m unittest tests.test_upscale -v   (needs ffmpeg)
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


def patch_new_file(rel):
    """A file that patches/lf-studio.patch adds: (its content, the abbreviated blob hash of its index line)."""
    text = (REPO / "patches" / "lf-studio.patch").read_text()
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
        """1344x768 (16:9) and 768x1344 (9:16), 121 frames at 24 fps; check_final says which. Any other size, length
        or rate is refused with what it is."""
        self.assertEqual(job_module.check_final(self.case / "stage2" / "refined_1344x768_121f.mp4"), "landscape")
        self.assertEqual(job_module.check_final(self.portrait), "portrait")
        for name, kwargs, message in (("ten.mp4", {"frames": 241}, "1344x768, 241 frames"),
                                      ("ten9x16.mp4", {"frames": 241, "width": 768, "height": 1344},
                                       "768x1344, 241 frames"),
                                      ("three.mp4", {"frames": 73, "width": 768, "height": 1344}, "73 frames"),
                                      ("fps.mp4", {"fps": 25}, "25/1"),
                                      ("576p.mp4", {"width": 1024, "height": 576}, "1024x576"),
                                      ("1080p9x16.mp4", {"width": 1080, "height": 1920}, "1080x1920")):
            with self.subTest(name=name), self.assertRaisesRegex(
                    JobError, message + r".*: not a 5 s 16:9 or 9:16 clip \(1344x768 or 768x1344, 121 frames at 24 "
                                        r"fps\)$"):
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
        job = dict(job_module.parse_job(self.job(steps=3, start_sigma=None), "/"), orientation="landscape")
        self.assertEqual(job_module.plan(job, schedules, grid),
                         {"orientation": "landscape", "size": [2560, 1440], "frames": 121, "steps": 3,
                          "sigmas": [0.9, 0.8, 0.6, 0.0], "lean_chunks": None})
        job = dict(job_module.parse_job(self.job(resolution="4k", steps=2, start_sigma=0.5), "/"),
                   orientation="landscape")
        planned = job_module.plan(job, schedules, grid)
        self.assertEqual((planned["sigmas"], planned["size"], planned["lean_chunks"]),
                         ([0.5, 0.25, 0.0], [3840, 2160], 16))
        for resolution, size, lean in (("1080p", [1080, 1920], None), ("2k", [1440, 2560], None),
                                       ("4k", [2160, 3840], 16)):         # 9:16: the same tokens, the same forward
            job = dict(job_module.parse_job(self.job(resolution=resolution), "/"), orientation="portrait")
            planned = job_module.plan(job, schedules, grid)
            self.assertEqual((planned["orientation"], planned["size"], planned["lean_chunks"]),
                             ("portrait", size, lean))
        with self.assertRaisesRegex(JobError, "no 3-step schedule"):
            job_module.plan(dict(job_module.parse_job(self.job(steps=3, start_sigma=None), "/"),
                                 orientation="landscape"), {2: [0.7, 0.4, 0]}, grid)


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
                         ("portrait", "portrait", [1440, 2560], 121))
        planned = json.loads(self.script("--job", str(self.write_job(resolution="1080p")), "--check",
                                          check=True).stdout)
        self.assertEqual(planned["size"], [1080, 1920])

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
                           ("upscaler", STAND_IN_UPSCALER)):
            (self.package / "open_h3" / f"{name}.py").write_text(text)
        self.loaded = {
            "conditioning.pt": {"prompt_embeds": StandInTensor(1, 41, 5120), "task": "t2va", "prompt": "a test",
                                "text_token_tags": StandInTensor(41, values=[1] * 41)},
            "stage1_direct_tensors.pt": {"audio": StandInTensor(1, 2, 161333)}}

    def run_job(self, fallback=(), **changes):
        """main() on job.json with `changes`; the loads go into self.loads, the stand-in modules into self.modules."""
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
        with mock.patch.dict(sys.modules, {"torch": fake, **self.modules}), \
                mock.patch.object(sys, "path", list(sys.path)), \
                mock.patch.object(job_module, "apply_environment"), \
                mock.patch.object(job_module, "load_fastload", return_value=stand_in_fastload(self.loads, fallback)), \
                mock.patch.object(job_module, "encode_audio", side_effect=encode_audio) as encode_audio, \
                mock.patch.object(job_module, "install_lean_blocks", return_value=3) as lean, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
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
            pins = {rel: hashlib.sha256((self.package / rel).read_bytes()).hexdigest()
                    for rel in job_module.NVIDIA_FILES}
            patcher = mock.patch.dict(job_module.NVIDIA_FILES, pins)
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def test_portrait_needs_the_pinned_nvidia_files(self):
        """A 9:16 clip with NVIDIA's package not at the pinned files (here the stand-ins) is refused before anything
        loads; a 16:9 clip does not read them."""
        self.portrait_clip(pin=False)
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"], receipt["orientation"]), (2, "failed", "portrait"))
        self.assertRegex(receipt["error"], r"^NVIDIA's package is not the one this wrapper was written for "
                                           r"\(6e4a440\): \{'open_h3/pipeline\.py': '[0-9a-f]{64}', "
                                           r"'open_h3/upscaler\.py': '[0-9a-f]{64}'\}$")
        self.assertEqual((calls, phases, self.loads), ([], [], []))

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


if __name__ == "__main__":
    unittest.main()
