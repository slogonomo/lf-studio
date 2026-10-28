"""upscale/regen2k_job.py (the experimental Upscale's GPU job) on the CPU: job.json, the clip's and the weights'
files, the text check on a stand-in, the command lines and the environment, the progress lines, the memory trace,
the script's --check and its refusals, main's receipt when the GPU part finishes, fails or gets SIGTERM (a stand-in
run()), and run() itself with a stand-in torch and a stand-in of NVIDIA's open_h3 that records its calls. No GPU, no
weights: NVIDIA's package is a stand-in (open_h3/sigmas.py, and for run() adapters, pipeline and upscaler), the
weights are small files.

The parts that need torch (the merger, the FFN chunks, the tensor checks) run on the CPU when torch is importable
and are skipped otherwise. python3 -m unittest tests.test_upscale -v   (needs ffmpeg)
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "upscale" / "regen2k_job.py"

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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.media, ignore_errors=True)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="upscale-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.case = self.tmp / "outputs" / "webui" / "20261028-010203-t2va-landscape-5s" / "web_010203_abcdef"
        for rel, data in (("qwen/conditioning.pt", b"conditioning"), ("stage1/h3_normalized_native.pt", b"draft"),
                          ("stage1/stage1_direct_tensors.pt", b"pcm")):
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
        data = {"version": 1, "case": str(self.case), "source": "final", "resolution": "2k", "steps": 2,
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


class JobTests(Fixture):
    def test_a_valid_job_and_its_defaults(self):
        job = job_module.parse_job(self.job(preread=DROP, vae_batch=DROP, version=DROP, meta=DROP), "/jobs/x")
        self.assertEqual((job["source"], job["resolution"], job["audio"], job["steps"], job["seed"]),
                         ("final", "2k", "clip", 2, 1234))
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
        for field, value in (("source", "upscaled"), ("source", None), ("resolution", "768p"), ("resolution", "2K"),
                             ("resolution", ["2k"]), ("audio", "both"), ("steps", 0), ("steps", 5), ("steps", "2"),
                             ("steps", True), ("steps", 2.0), ("start_sigma", 0), ("start_sigma", 1),
                             ("start_sigma", -0.1), ("start_sigma", "0.7"), ("start_sigma", True),
                             ("start_sigma", nan), ("seed", -1), ("seed", 2 ** 63), ("seed", 1.5), ("seed", True),
                             ("seed", None), ("preread", 1), ("preread", "yes"), ("vae_batch", 0),
                             ("vae_batch", 257), ("vae_batch", "26"), ("version", 2), ("version", True),
                             ("meta", []), ("case", "relative/dir"), ("case", ""), ("case", 7), ("paths", []),
                             ("paths", None)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(JobError, field):
                    job_module.parse_job(self.job(**{field: value}), "/")

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
                                    "audio": "clip", "preread": False, "min_free_gib": 110.0, "guard_gib": 5.0})
        settings, _ = job_module.studio_settings({"LFSTUDIO_HOME": "/srv/lf", "LFSTUDIO_CHECKPOINTS": "/data/ckpt",
                                                  "LFSTUDIO_UPSCALE_STRENGTH": ""})      # empty: the default
        self.assertEqual((settings["weights"], settings["base_model"], settings["strength"]),
                         (Path("/data/ckpt/Open-H3-Regenerate-2K"), Path("/data/ckpt/MiniMax-H3"), "balanced"))
        with mock.patch.dict(os.environ, {"HOME": "/srv/someone"}):
            self.assertEqual(job_module.studio_settings({"LFSTUDIO_UPSCALE_PKG": "~/regen"})[0]["package"],
                             Path("/srv/someone/regen"))

    def test_values(self):
        settings, problems = job_module.studio_settings({
            "LFSTUDIO_UPSCALE_STRENGTH": "full", "LFSTUDIO_UPSCALE_AUDIO": "silence", "LFSTUDIO_UPSCALE_PREREAD": "1",
            "LFSTUDIO_UPSCALE_MIN_FREE_GIB": "0", "LFSTUDIO_UPSCALE_GUARD_GIB": "2.5"})
        self.assertEqual(problems, [])
        self.assertEqual([settings[k] for k in ("strength", "audio", "preread", "min_free_gib", "guard_gib")],
                         ["full", "silence", True, 0.0, 2.5])
        _, problems = job_module.studio_settings({
            "LFSTUDIO_UPSCALE_PYTHON": "python3", "LFSTUDIO_UPSCALE_STRENGTH": "light",
            "LFSTUDIO_UPSCALE_AUDIO": "Clip", "LFSTUDIO_UPSCALE_PREREAD": "yes", "LFSTUDIO_UPSCALE_MIN_FREE_GIB": "-1",
            "LFSTUDIO_UPSCALE_GUARD_GIB": "inf"})
        self.assertEqual(problems, [
            "LFSTUDIO_UPSCALE_PYTHON=python3: must be an absolute path",
            "LFSTUDIO_UPSCALE_STRENGTH=light: must be faithful, balanced or full",
            "LFSTUDIO_UPSCALE_AUDIO=Clip: must be clip or silence", "LFSTUDIO_UPSCALE_PREREAD=yes: must be 0 or 1",
            "LFSTUDIO_UPSCALE_MIN_FREE_GIB=-1: must be a number of GiB, 0 or more",
            "LFSTUDIO_UPSCALE_GUARD_GIB=inf: must be a number of GiB, 0 or more (see bin/env.example)"])


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

    def test_draft_and_audio_shapes(self):
        job_module.check_draft_shape((1, 24, 37, 24, 42))
        for shape in ((1, 24, 72, 24, 42), (1, 24, 37, 42, 24), (24, 37, 24, 42)):
            with self.subTest(shape=shape), self.assertRaisesRegex(JobError, "not a 5 s 16:9"):
                job_module.check_draft_shape(shape)
        self.assertEqual(job_module.AUDIO_SAMPLES, 165333)
        for samples in (161333, 165333, 1):
            job_module.check_pcm_shape((1, 2, samples))
        for shape in ((1, 2, 165334), (1, 2, 0), (2, 161333), (1, 1, 161333), (1, 2, 161333, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(JobError, "not stereo PCM"):
                job_module.check_pcm_shape(shape)


class InputTests(Fixture):
    def test_the_files_each_job_reads(self):
        files = job_module.case_inputs(job_module.parse_job(self.job(audio="silence"), "/"))
        self.assertEqual(set(files), {"conditioning", "final"})
        self.assertEqual(files["final"], self.case / "stage2" / "refined_1344x768_121f.mp4")
        files = job_module.case_inputs(job_module.parse_job(self.job(source="draft", audio="clip"), "/"))
        self.assertEqual(files["draft"], self.case / "stage1" / "h3_normalized_native.pt")
        self.assertEqual(files["pcm"], self.case / "stage1" / "stage1_direct_tensors.pt")

    def test_missing_files_are_refused(self):
        for rel, changes, message in (("stage1/h3_normalized_native.pt", {"source": "draft"}, "Stage-1 latent"),
                                      ("stage1/stage1_direct_tensors.pt", {"audio": "clip"}, "Stage-1 audio"),
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

    def test_only_a_5s_16x9_clip_passes(self):
        job_module.check_final(self.case / "stage2" / "refined_1344x768_121f.mp4")
        for name, kwargs, message in (("portrait.mp4", {"width": 768, "height": 1344}, "768x1344"),
                                      ("ten.mp4", {"frames": 241}, "241 frames"),
                                      ("fps.mp4", {"fps": 25}, "25/1")):
            with self.subTest(name=name), self.assertRaisesRegex(JobError, message + ".*not a 5 s 16:9 clip"):
                job_module.check_final(make_mp4(self.tmp / name, **kwargs))
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
        job = job_module.parse_job(self.job(steps=3, start_sigma=None), "/")
        self.assertEqual(job_module.plan(job, schedules, grid),
                         {"size": [2560, 1440], "frames": 121, "steps": 3, "sigmas": [0.9, 0.8, 0.6, 0.0],
                          "ffn_chunks": 1})
        job = job_module.parse_job(self.job(resolution="4k", steps=2, start_sigma=0.5), "/")
        planned = job_module.plan(job, schedules, grid)
        self.assertEqual((planned["sigmas"], planned["size"], planned["ffn_chunks"]),
                         ([0.5, 0.25, 0.0], [3840, 2160], 8))
        with self.assertRaisesRegex(JobError, "no 3-step schedule"):
            job_module.plan(job_module.parse_job(self.job(steps=3, start_sigma=None), "/"), {2: [0.7, 0.4, 0]}, grid)


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
                 "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                 "print(sorted({k.split('.')[0] for k in sys.modules} & {'torch', 'numpy', 'diffusers', 'open_h3'}))")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", probe], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "[]")
        self.assertIn("--check", self.script("--help", check=True).stdout)

    def test_check_prints_the_plan_and_writes_nothing(self):
        path = self.write_job()
        result = self.script("--job", str(path), "--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        planned = json.loads(result.stdout)
        self.assertEqual((planned["ok"], planned["sigmas"], planned["size"], planned["steps"], planned["ffn_chunks"]),
                         (True, [0.7, 0.35, 0.0], [2560, 1440], 2, 1))
        self.assertEqual(set(planned["inputs"]), {"conditioning", "final", "pcm"})
        self.assertEqual(planned["job"]["meta"]["strength"], "balanced")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json"])
        path = self.write_job(resolution="4k", steps=3, start_sigma=None, source="draft", audio="silence")
        planned = json.loads(self.script("--job", str(path), "--check", check=True).stdout)
        self.assertEqual((planned["sigmas"], planned["size"], planned["ffn_chunks"]),
                         ([0.9, 0.8, 0.6, 0.0], [3840, 2160], 8))
        self.assertEqual(set(planned["inputs"]), {"conditioning", "final", "draft"})

    def test_check_refuses(self):
        make_mp4(self.case / "stage2" / "refined_768x1344_121f.mp4", width=768, height=1344)
        (self.case / "stage2" / "refined_1344x768_121f.mp4").unlink()
        result = self.script("--job", str(self.write_job()), "--check")
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout.splitlines()[-1], r"^\[lf-studio\] error refined_768x1344_121f\.mp4 is "
                                                         r"768x1344, 121 frames .* not a 5 s 16:9 clip")
        self.assertFalse((self.out / "receipt.json").exists())
        (self.package / "open_h3" / "sigmas.py").unlink()
        make_mp4(self.case / "stage2" / "refined_1344x768_121f.mp4")
        (self.case / "stage2" / "refined_768x1344_121f.mp4").unlink()
        result = self.script("--job", str(self.write_job()), "--check")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot import open_h3.sigmas", result.stdout)

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
    fake.cuda = types.SimpleNamespace(
        set_device=nothing, synchronize=nothing, reset_peak_memory_stats=nothing, empty_cache=nothing,
        max_memory_allocated=lambda: 81 * 2 ** 30, max_memory_reserved=lambda: 83 * 2 ** 30,
        memory_allocated=lambda: 5 * 2 ** 30, is_available=lambda: True, get_device_name=lambda: "stand-in GPU",
        get_device_capability=lambda: (12, 1))
    fake.Tensor = StandInTensor
    return fake


# NVIDIA's open_h3 as far as run() uses it; every call goes into pipeline.CALLS
STAND_IN_ADAPTERS = f'''LORA_FILE = {job_module.LORA_FILE!r}
CURVE_FILE = {job_module.CURVE_FILE!r}
CURVE_TABLE = {job_module.CURVE_TABLE!r}


class StageMerger:
    def __init__(self, *args, **kwargs):
        raise AssertionError("NVIDIA's StageMerger was used")
'''
STAND_IN_PIPELINE = f'''import types
from pathlib import Path

import torch

from . import adapters
from .sigmas import SCHEDULES, shifted_grid

NATIVE_SHAPE = (24, 37, 48, 84)
RESOLUTIONS = {{"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080), "2k": (1440, 2560, 1440),
               "4k": (2176, 3840, 2160)}}
UPSCALER_FILE = {job_module.UPSCALER_FILE!r}
CALLS = []


def encode_video(base_model, path, size="768p", seed=42):
    CALLS.append(("encode_video", Path(path).name, size, seed))
    return torch.Tensor(*NATIVE_SHAPE, tag="final")


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
        CALLS.append(("Pipeline", dict(fwd=fwd, hyperflow=hyperflow, vae_batch=vae_batch, steps=steps,
                                       start_sigma=start_sigma, resolution=resolution, vae=vae)))
        self.sigmas = SCHEDULES[steps] if start_sigma is None else shifted_grid(start_sigma, steps)
        height, width, self.keep_rows = RESOLUTIONS[resolution]
        self.target_shape = (24, 37, height // 16, width // 16)
        self.device = "cuda"
        self.audio = torch.Tensor(2, 32, 207, tag="silence", device="cuda")
        self.model = types.SimpleNamespace(transformer_blocks=[
            types.SimpleNamespace(ff=types.SimpleNamespace(forward=lambda x: x)) for _ in range(3)])
        self.curve = types.SimpleNamespace(strength=0.0)
        self.merger = adapters.StageMerger(self.model, {{"refine": [({{}}, 1.0)]}}, self.device, resident=False)
        self.node = Node()
        self.vae = object()

    def set_stage(self, stage):
        self.curve.strength = 1.0
        self.merger.merge(stage)
        CALLS.append(("set_stage", stage))

    def on_step(self, stage, index):
        CALLS.append(("on_step", stage, index))

    def upscale(self, native):
        CALLS.append(("upscale", native.shape, native.tag))
        return torch.Tensor(*self.target_shape)

    def refine(self, low, text, seed):
        CALLS.append(("refine", low.shape, text["prompt_embeds"].shape, seed, self.audio.tag, self.curve.strength))
        for index in range(len(self.sigmas) - 1):
            self.on_step("refine", index)
        return torch.Tensor(*low.shape)

    def decode(self, latents):
        CALLS.append(("decode", self.model is None, self.vae is not None, dict(self.node.MODEL_CACHE)))
        return torch.Tensor(121, self.keep_rows, self.target_shape[3] * 16, 3)
'''
STAND_IN_UPSCALER = '''from . import pipeline


def upscale(node, native, target_shape):
    pipeline.CALLS.append(("draft_x2", native.shape, target_shape))
    return pipeline.torch.Tensor(*target_shape, tag="draft")
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
            "h3_normalized_native.pt": {"h3_normalized": StandInTensor(1, 24, 37, 24, 42)},
            "stage1_direct_tensors.pt": {"audio": StandInTensor(1, 2, 161333)}}

    def run_job(self, **changes):
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        fake = stand_in_torch(self.loaded)
        stdout = io.StringIO()
        clip_audio = StandInTensor(2, 32, 207, tag="clip")
        with mock.patch.dict(sys.modules, {"torch": fake}), mock.patch.object(sys, "path", list(sys.path)), \
                mock.patch.object(job_module, "apply_environment"), \
                mock.patch.object(job_module, "encode_audio", return_value=clip_audio) as encode_audio, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = job_module.main(["--job", str(self.write_job(**changes))])
            calls = list(sys.modules["open_h3.pipeline"].CALLS) if "open_h3.pipeline" in sys.modules else []
        phases = [p[1] for p in map(job_module.parse_line, stdout.getvalue().splitlines()) if p and p[0] == "phase"]
        return status, json.loads((self.out / "receipt.json").read_text()), calls, phases, encode_audio

    def test_final_with_the_clip_audio(self):
        status, receipt, calls, phases, encode_audio = self.run_job()
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(phases, ["encoding", "loading", "refine 1/2", "refine 2/2", "decoding", "saving"])
        self.assertEqual([c[0] for c in calls], ["encode_video", "Pipeline", "load_model", "upscale", "set_stage",
                                                 "refine", "on_step", "on_step", "decode", "encode_mp4"])
        self.assertEqual(calls[0], ("encode_video", "refined_1344x768_121f.mp4", "768p", 42))
        self.assertEqual(calls[1][1], {"fwd": "standard", "hyperflow": None, "vae_batch": 26, "steps": 2,
                                       "start_sigma": 0.7, "resolution": "2k", "vae": "standard"})
        self.assertEqual(calls[3], ("upscale", (24, 37, 48, 84), "final"))
        self.assertEqual(calls[5], ("refine", (24, 37, 90, 160), (41, 5120), 1234, "clip", 1.0))
        self.assertEqual(calls[8], ("decode", True, True, {}))      # the DiT and the upscaler freed, the VAE kept
        self.assertEqual(calls[9], ("encode_mp4", ".video_noaudio.mp4.partial.mp4", (121, 1440, 2560, 3)))
        self.assertEqual(encode_audio.call_args.args[0], self.base)
        self.assertEqual(encode_audio.call_args.args[1].shape, (2, 161333))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["job.json", "receipt.json", "video_noaudio.mp4"])
        self.assertEqual(receipt["output"], {"file": "video_noaudio.mp4", "size": [2560, 1440], "frames": 121,
                                             "bytes": 3})
        self.assertEqual((receipt["sigmas"], receipt["steps"], len(receipt["steps_s"])), ([0.7, 0.35, 0.0], 2, 2))
        self.assertEqual(set(receipt["times_s"]), {"encode_video", "encode_audio", "load", "upscale", "merge_refine",
                                                   "refine", "vae_decode", "mp4"})
        self.assertEqual(set(receipt["peak_gib"]), set(receipt["times_s"]))
        self.assertEqual(receipt["peak_allocated_gib"], 81.0)
        self.assertEqual((receipt["merge"]["weights"], receipt["ffn_chunks"], receipt["text_rows"]), (0, 1, 41))
        self.assertEqual((receipt["prompt"], receipt["versions"]["gpu"], receipt["versions"]["torch"]),
                         ("a test", "stand-in GPU", "0.0+stand-in"))
        self.assertEqual(receipt["allocated_before_decode_gib"], 5.0)
        self.assertIn("refine 2/2", receipt["lowest_mem_available_gib"])

    def test_draft_with_silence_at_4k(self):
        status, receipt, calls, phases, encode_audio = self.run_job(source="draft", audio="silence",
                                                                     resolution="4k", steps=3, start_sigma=None)
        self.assertEqual((status, receipt["state"]), (0, "done"), receipt.get("traceback") or receipt.get("error"))
        self.assertEqual(phases, ["loading", "refine 1/3", "refine 2/3", "refine 3/3", "decoding", "saving"])
        names = [c[0] for c in calls]
        self.assertEqual(names[:5], ["Pipeline", "load_model", "draft_x2", "upscale", "set_stage"])
        self.assertEqual(calls[2], ("draft_x2", (24, 37, 24, 42), (24, 37, 48, 84)))
        self.assertEqual(calls[3], ("upscale", (24, 37, 48, 84), "draft"))
        self.assertEqual(calls[5][4], "silence")
        encode_audio.assert_not_called()
        self.assertEqual((receipt["sigmas"], receipt["ffn_chunks"], receipt["output"]["size"]),
                         ([0.9, 0.8, 0.6, 0.0], 8, [3840, 2160]))
        self.assertEqual(set(receipt["inputs_sha256"]), {"conditioning", "final", "draft"})

    def test_another_package_is_refused(self):
        (self.package / "open_h3" / "adapters.py").write_text(
            STAND_IN_ADAPTERS.replace("open_h3_regenerate_2k_lora", "open_h3_regenerate_2k_lora_v2"))
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"]), (2, "failed"))
        self.assertIn("not the one this wrapper was written for", receipt["error"])
        self.assertEqual((calls, phases), ([], []))

    def test_fl2va_conditioning_is_refused_before_anything_loads(self):
        self.loaded["conditioning.pt"]["text_token_tags"] = StandInTensor(41, values=[1] * 6 + [0] * 35)
        status, receipt, calls, phases, _ = self.run_job()
        self.assertEqual((status, receipt["state"]), (2, "failed"))
        self.assertIn("35 non-text rows", receipt["error"])
        self.assertEqual((calls, phases), ([], []))


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

    def test_ffn_chunks(self):
        torch.manual_seed(0)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Linear(64, 16))

        model = torch.nn.Module()
        model.transformer_blocks = torch.nn.ModuleList([Block(), Block()])
        x = torch.randn(1, 37, 16)
        expected = [block.ff(x) for block in model.transformer_blocks]
        self.assertEqual(job_module.chunk_ffn(model, 1), 0)
        self.assertEqual(job_module.chunk_ffn(model, 8), 2)
        for block, want in zip(model.transformer_blocks, expected):
            self.assertTrue(torch.allclose(block.ff(x), want, atol=1e-5))
            self.assertTrue(torch.allclose(block.ff(x[0]), want[0], atol=1e-5))

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
        draft = job_module.draft_input({"h3_normalized": torch.zeros(1, 24, 37, 24, 42)}, torch)
        self.assertEqual((tuple(draft.shape), draft.dtype), ((24, 37, 24, 42), torch.float32))
        with self.assertRaisesRegex(JobError, "not a 5 s 16:9"):
            job_module.draft_input({"h3_normalized": torch.zeros(1, 24, 72, 24, 42)}, torch)
        pcm = job_module.pcm_input({"audio": torch.zeros(1, 2, 161333)}, torch)
        self.assertEqual(tuple(pcm.shape), (2, 161333))
        with self.assertRaisesRegex(JobError, "no audio"):
            job_module.pcm_input({"h3_normalized": torch.zeros(1)}, torch)


if __name__ == "__main__":
    unittest.main()
