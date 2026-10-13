"""bench/replay (the Stage-2 replay bench) on the CPU: draft store, arms, plan and --dry, the GPU run object with
stand-ins, the report and the SyncNet table. No torch, no GPU, no model weights.

The package's geometry comes from patches/sol-h3-studio.patch itself (runtime/geometry.py is a new file there), so
the plan's shapes are the patch's. python3 -m unittest tests.test_replay -v   (needs ffmpeg)
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from replay import arms, drafts, media, plan, quality, report, stage2, syncnet  # noqa: E402

if not shutil.which("ffmpeg"):
    raise unittest.SkipTest("ffmpeg not installed")

PKG_REL = "models/minimax_h3/Sol-H3-Spark"
# the parts of Sol-H3-Spark's configs/default.json (upstream, not in the patch) that the geometry reads
RECIPE = {"stage1": {"width": 672, "height": 384, "frames": 124},
          "output": {"width": 1344, "height": 768, "frames": 121},
          "handoff": {"refiner_input_shape": [1, 128, 16, 24, 42]},
          "stage2": {"sigmas": [0.909375, 0.725, 0.421875, 0.0],
                     "tiles": {"temporal": 128, "temporal_overlap": 24, "height": 448, "height_overlap": 64,
                               "width": 768, "width_overlap": 64}}}


def patch_new_file(rel):
    """The content of a file that patches/sol-h3-studio.patch adds."""
    text = (REPO / "patches" / "sol-h3-studio.patch").read_text()
    for section in re.split(r"(?m)^(?=diff --git )", text):
        if section.startswith(f"diff --git a/{rel} b/{rel}\n") and "\nnew file mode" in section:
            body = section.split("\n+++ ", 1)[1].split("\n", 1)[1]
            return "".join(line[1:] + "\n" for line in body.splitlines() if line.startswith("+"))
    raise AssertionError(f"{rel} is not a new file in the patch")


def make_package(root):
    sana = Path(root) / "Sana"
    pkg = sana / PKG_REL
    (pkg / "runtime").mkdir(parents=True)
    (pkg / "configs").mkdir()
    (pkg / "runtime" / "__init__.py").write_text("")
    (pkg / "runtime" / "geometry.py").write_text(patch_new_file(f"{PKG_REL}/runtime/geometry.py"))
    (pkg / "configs" / "default.json").write_text(json.dumps(RECIPE))
    return sana, pkg


def tiny_mp4(path, color="red", seconds=0.5, tone=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=64x36:r=24:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=32000:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def make_case(root, geometry, name, task="t2va", seconds=5, orientation="landscape", studio_index=1, studio=True):
    frames = geometry.frame_geometry(geometry.draft_frames_for_seconds(seconds))
    h, w = (42, 24) if orientation == "portrait" else (24, 42)
    case = Path(root) / "outputs" / "webui" / f"session-{task}-{orientation}-{seconds}s" / name
    (case / "stage1").mkdir(parents=True)
    payload = os.urandom(4096)
    (case / "stage1" / "stage1_direct_tensors.pt").write_bytes(payload)
    capture = {"status": "PASS", "latent_only_transfer": True, "same_request": True, "task": task, "seed": 7,
               "case_id": name, "request_id": name, "prompt": "a test prompt", "source_index": 0,
               "payload_sha256": hashlib.sha256(payload).hexdigest(),
               "h3_normalized": {"shape": [1, 24, frames["h3_latent_frames"], h, w]},
               "audio": {"shape": [1, 2, frames["audio_samples"]]}}
    (case / "stage1" / "capture.json").write_text(json.dumps(capture))
    if studio:
        out_w, out_h = (768, 1344) if orientation == "portrait" else (1344, 768)
        mp4 = tiny_mp4(case / "stage2" / f"refined_{out_w}x{out_h}_{frames['output_frames']}f.mp4")
        (case / "stage2" / "stage2_result.json").write_text(json.dumps(
            {"status": "PASS", "output": str(mp4), "output_sha256": media.sha256(mp4), "request_index": studio_index,
             "stage2_audio": "frozen", "stage2_request_s": 36.0}))
    return case


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="replay-test-"))
        self.sana, self.pkg = make_package(self.tmp)
        self.geometry = drafts.load_geometry(self.pkg)
        self.rt = self.tmp / "rt"
        self.store = drafts.store_dir(self.rt)
        self.store.mkdir(parents=True)
        for task in ("t2va", "fl2va", "ref2va"):
            (self.tmp / f"paths-{task}.json").write_text(json.dumps(
                {key: (sys.executable if key == "stage2_python" else f"/models/{key}") for key in plan.STAGE2_PATH_KEYS}))
        self.paths_template = str(self.tmp / "paths-{task}.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add(self, name, **kw):
        extra = {k: kw.pop(k) for k in ("dialogue", "turns", "mouth", "force") if k in kw}
        case = make_case(self.rt, self.geometry, name, **kw)
        return drafts.add(self.store, name, case, self.geometry, **extra)


class PatchGeometryTests(Fixture):
    def test_geometry_from_the_patch(self):
        five = self.geometry.frame_geometry(self.geometry.DRAFT_FRAMES)
        self.assertEqual((five["draft_frames"], five["output_frames"], five["h3_latent_frames"]), (124, 121, 37))
        ten = self.geometry.frame_geometry(self.geometry.draft_frames_for_seconds(10))
        self.assertEqual((ten["draft_frames"], ten["output_frames"], ten["h3_latent_frames"]), (243, 241, 72))


class DraftTests(Fixture):
    def test_add_infers_format_and_copies(self):
        info = self.add("t2va5", task="t2va", seconds=5)
        self.assertEqual((info["task"], info["orientation"], info["seconds"], info["output_frames"]),
                         ("t2va", "landscape", 5, 121))
        self.assertTrue(info["studio"]["warm"])
        d = self.store / "t2va5"
        for rel in ("draft.json", "stage1/capture.json", "stage1/stage1_direct_tensors.pt",
                    "studio/stage2_result.json", "studio/refined_1344x768_121f.mp4"):
            self.assertTrue((d / rel).is_file(), rel)
        self.assertTrue(info["studio"]["sha256_matches_receipt"])
        ten = self.add("fl2va10", task="fl2va", seconds=10, turns="R:0-0.99,L:4-6.33")
        self.assertEqual((ten["seconds"], ten["draft_frames"], ten["output_frames"]), (10, 243, 241))
        self.assertTrue(ten["dialogue"])
        portrait = self.add("tall", orientation="portrait", studio_index=0)
        self.assertEqual(portrait["orientation"], "portrait")
        self.assertFalse(portrait["studio"]["warm"])
        self.assertEqual([d["name"] for d in drafts.all_drafts(self.store)], ["fl2va10", "t2va5", "tall"])

    def test_copy_survives_the_case_folder(self):
        case = make_case(self.rt, self.geometry, "gone")
        drafts.add(self.store, "kept", case, self.geometry)
        shutil.rmtree(case)                       # what emptying the studio's trash does
        self.assertTrue((self.store / "kept" / "stage1" / "stage1_direct_tensors.pt").is_file())

    def test_refusals(self):
        case = make_case(self.rt, self.geometry, "c1")
        with self.assertRaises(ValueError):
            drafts.add(self.store, "bad name", case, self.geometry)
        with self.assertRaises(ValueError):
            drafts.add(self.store, "x", case, self.geometry, turns="R0-1")
        with self.assertRaises(ValueError):
            drafts.add(self.store, "x", case, self.geometry, turns="R:2-1")
        with self.assertRaises(ValueError):
            drafts.add(self.store, "x", case, self.geometry, mouth="10:10:20")
        drafts.add(self.store, "x", case, self.geometry)
        with self.assertRaises(FileExistsError):
            drafts.add(self.store, "x", case, self.geometry)
        drafts.add(self.store, "x", case, self.geometry, force=True)
        (case / "stage1" / "stage1_direct_tensors.pt").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            drafts.add(self.store, "y", case, self.geometry)
        with self.assertRaises(FileNotFoundError):
            drafts.add(self.store, "z", case / "stage1", self.geometry)

    def test_without_studio_clip(self):
        info = self.add("nostudio", studio=False)
        self.assertIsNone(info["studio"])

    def test_turns(self):
        self.assertEqual(drafts.parse_turns("R:0-1.2,L:2.4-4.7"),
                         [{"speaker": "R", "t0": 0.0, "t1": 1.2}, {"speaker": "L", "t0": 2.4, "t1": 4.7}])
        self.assertEqual(syncnet.turns_spec(drafts.parse_turns("R:0-1.2,L:2.4-4.7")), "R:0-1.2,L:2.4-4.7")


class ArmTests(Fixture):
    def native(self, orientation="landscape", seconds=5):
        return plan.native_geometry(self.pkg, self.geometry, orientation, seconds)

    def test_every_arm_is_a_named_function(self):
        native, sigmas = self.native()
        for name, a in arms.ARMS.items():
            self.assertEqual(a.fn.__name__, name)
            self.assertIn(a.canvas, arms.CANVASES)
            s = arms.settings(name, sigmas, native)
            self.assertEqual(s["arm"], name)
            deviates = s["steps"] != 3 or not s["denoise"] or a.canvas != 768
            self.assertEqual(bool(s["relaxes"]), deviates, name)

    def test_baseline_changes_nothing(self):
        native, sigmas = self.native()
        s = arms.settings("baseline", sigmas, native)
        self.assertEqual((s["sigmas"], s["steps"], s["output"], s["relaxes"]), (list(sigmas), 3, [1344, 768], []))
        self.assertTrue(s["tiles_default"])
        self.assertEqual(s["tiles"], {"frames": [128, 24], "height": [448, 64], "width": [768, 64]})
        self.assertEqual((s["upscale_scale"], s["video_tokens"]), (2.0, 16128))

    def test_steps_preview_tiles_profile(self):
        native, sigmas = self.native()
        s = arms.settings("steps2_from_0725", sigmas, native)
        self.assertEqual((s["sigmas"], s["steps"]), ([0.725, 0.421875, 0.0], 2))
        self.assertIn("2 steps (2 dense, 94 Sol calls)", s["relaxes"][0])
        p = arms.settings("preview", sigmas, native)
        self.assertEqual((p["sigmas"], p["steps"], p["denoise"], len(p["relaxes"])), ([0.0], 0, False, 2))
        u = arms.settings("decode_untiled", sigmas, native)
        self.assertEqual((u["tiles"]["height"], u["tiles"]["width"], u["relaxes"]), ([0, 0], [0, 0], []))
        self.assertEqual(arms.settings("profile_step", sigmas, native)["profile_step"], 1)
        tall, _ = self.native("portrait")
        t = arms.settings("decode_long768", sigmas, tall)
        self.assertEqual((t["tiles"]["height"], t["tiles"]["width"]), ([768, 64], [0, 0]))

    def test_canvas_shapes(self):
        native, sigmas = self.native()
        self.assertEqual(arms.canvas_geometry(native, 768), native)
        g480 = arms.canvas_geometry(native, 480)
        self.assertEqual((g480["output_width"], g480["output_height"]), (832, 480))
        self.assertEqual(tuple(g480["h3_upscaled"]), (1, 24, 37, 30, 52))
        self.assertEqual(tuple(g480["refiner_input"]), (1, 128, 16, 15, 26))
        self.assertEqual((arms.upscale_scale(g480), arms.video_tokens(g480)), (1.25, 6240))
        g384 = arms.canvas_geometry(native, 384)
        self.assertEqual(tuple(g384["h3_upscaled"]), tuple(g384["h3"]))
        self.assertEqual(arms.video_tokens(g384), 4032)
        self.assertIn("skipped", " ".join(arms.settings("canvas384", sigmas, native)["relaxes"]))
        tall, _ = self.native("portrait")
        t480 = arms.canvas_geometry(tall, 480)
        self.assertEqual((t480["output_width"], t480["output_height"]), (480, 832))
        self.assertEqual(tuple(t480["h3_upscaled"]), (1, 24, 37, 52, 30))
        ten, _ = self.native(seconds=10)
        self.assertEqual(arms.video_tokens(arms.canvas_geometry(ten, 480)), 31 * 15 * 26)

    def test_recorder_refuses_bad_settings(self):
        native, sigmas = self.native()
        rec = arms.Recorder(sigmas, native)
        with self.assertRaises(ValueError):
            rec.sigmas((0.5, 0.7, 0.0))
        with self.assertRaises(ValueError):
            rec.sigmas((0.5, 0.1))
        with self.assertRaises(ValueError):
            rec.tiles(short=(64, 64))


class PlanTests(Fixture):
    def build(self, names, arm_text, **kw):
        selected = [drafts.load(self.store, n) for n in names]
        return plan.build(selected, plan.parse_arms(arm_text, kw.pop("repeats", 2)), package=self.pkg,
                          geometry=self.geometry, paths_template=self.paths_template, out=self.tmp / "out", **kw)

    def test_parse_arms(self):
        self.assertEqual(plan.parse_arms("baseline:3, preview", 2), [("baseline", 3), ("preview", 2)])
        for bad in ("nope", "baseline:0", "baseline:x", "baseline,baseline", ""):
            with self.assertRaises(ValueError, msg=bad):
                plan.parse_arms(bad, 2)

    def test_groups_and_runs(self):
        self.add("a5", task="t2va")
        self.add("b5", task="ref2va")
        self.add("c10", task="fl2va", seconds=10)
        self.add("tall", orientation="portrait")
        p = self.build(["a5", "b5", "c10", "tall"], "baseline:2,steps2_from_0725:1,canvas480_steps2:1")
        ids = [g["id"] for g in p["groups"]]
        self.assertEqual(ids, ["1-landscape-5s-1344x768", "2-landscape-10s-1344x768", "3-portrait-5s-768x1344",
                               "4-landscape-5s-832x480", "5-landscape-10s-832x480", "6-portrait-5s-480x832"])
        g1 = p["groups"][0]
        self.assertEqual(g1["drafts"], ["a5", "b5"])
        self.assertEqual(g1["tasks"], ["t2va", "ref2va"])
        firsts = [r for r in g1["runs"] if r["first_in_process"]]
        self.assertEqual(firsts, [g1["runs"][0]])
        self.assertEqual((firsts[0]["draft"], firsts[0]["arm"], firsts[0]["repeat"]), ("a5", "baseline", 0))
        # 1 first run + 2 drafts x (2 baseline + 1 steps2)
        self.assertEqual(len(g1["runs"]), 7)
        self.assertEqual(len({r["dir"] for g in p["groups"] for r in g["runs"]}),
                         sum(len(g["runs"]) for g in p["groups"]))
        self.assertEqual(list(p["groups"][3]["arms"]), ["canvas480_steps2"])
        self.assertEqual(p["groups"][1]["geometry"]["output_frames"], 241)
        self.assertGreater(p["est_s"], 0)
        text = plan.describe(p)
        self.assertIn("first in process", text)
        self.assertIn("relaxes: the H3 upscaler runs at scale 1.25", text)

    def test_manifests_must_agree_on_stage2(self):
        self.add("a5", task="t2va")
        self.add("b5", task="fl2va")
        values = json.loads((self.tmp / "paths-fl2va.json").read_text())
        values["transformer"] = "/models/other"
        (self.tmp / "paths-fl2va.json").write_text(json.dumps(values))
        with self.assertRaises(ValueError):
            self.build(["a5", "b5"], "baseline")

    def test_package_info_without_git(self):
        info = plan.package_info(self.pkg, self.sana, REPO)
        self.assertIsNone(info["patch_sha256"])
        self.assertEqual(info["repo_patch_sha256"], media.sha256(REPO / "patches" / "sol-h3-studio.patch"))
        self.assertEqual(info["stage2_code_files"], 1)


class ScriptTests(Fixture):
    """bench/replay.sh itself: settings from the env file, the memory check, --dry writes nothing."""

    def script(self, *args, extra=None):
        env_file = self.tmp / "env"
        env_file.write_text(f'SOLH3_HOME="{self.tmp}"\nSOLH3_SANA="{self.sana}"\nSOLH3_PATHS="{self.paths_template}"\n'
                            f'SOL_H3_SPARK_RUNTIME_ROOT="{self.rt}"\n')
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOLH3_", "SOL_H3_"))}
        env.update(SOLH3_ENV=str(env_file), **(extra or {}))
        return subprocess.run(["bash", str(REPO / "bench" / "replay.sh"), *args], capture_output=True, text=True,
                              env=env, timeout=120)

    def test_add_list_dry(self):
        case = make_case(self.rt, self.geometry, "web_1")
        r = self.script("add", "d5", str(case), "--dialogue")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("added d5: t2va landscape 5 s", r.stdout)
        self.assertIn("d5:", self.script("list").stdout)
        self.assertIn("canvas480_steps2", self.script("arms").stdout)
        r = self.script("run", "--dry", "--arms", "baseline:2,preview:1,canvas384:1", extra={"SOLH3_MIN_FREE_GIB": "100000"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("process 1/2 [1-landscape-5s-1344x768]", r.stdout)
        self.assertIn("refiner (1,128,16,24,42), 16128 video tokens", r.stdout)
        self.assertIn("process 2/2 [2-landscape-5s-672x384]", r.stdout)
        self.assertIn("dry run: nothing was started or written", r.stdout)
        self.assertEqual(sorted(p.name for p in (self.rt / "outputs" / "bench").iterdir()), ["replay-drafts"])
        r = self.script("run", "--dry", "--arms", "nope")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown arm", r.stderr)

    def test_run_checks_free_memory_first(self):
        make_case(self.rt, self.geometry, "web_1")
        r = self.script("run", extra={"SOLH3_MIN_FREE_GIB": "100000"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("MemAvailable, need 100000", r.stderr)
        self.assertEqual([p.name for p in (self.rt / "outputs" / "bench").iterdir()], ["replay-drafts"])

    def test_run_end_to_end_with_a_stand_in_stage2(self):
        """run without --dry, with a stand-in for the Stage-2 python: cache copy, memory guard, the studio's worker
        settings, one process per group, the report, and the cache copy removed afterwards."""
        fake = self.tmp / "stage2-python"
        fake.write_text(STAND_IN)
        fake.chmod(0o755)
        for task in ("t2va", "fl2va", "ref2va"):
            values = json.loads((self.tmp / f"paths-{task}.json").read_text())
            values["stage2_python"] = str(fake)
            (self.tmp / f"paths-{task}.json").write_text(json.dumps(values))
        cache = self.tmp / "cache"
        (cache / "stage2" / "inductor").mkdir(parents=True)
        (cache / "stage2" / "inductor" / "entry").write_text("x")
        (cache / "stage1").mkdir()
        studio_case = make_case(self.rt, self.geometry, "web_1")
        self.assertEqual(self.script("add", "d5", str(studio_case)).returncode, 0)
        r = self.script("run", "--arms", "baseline:1,canvas384:1", "--label", "e2e",
                        extra={"SOLH3_MIN_FREE_GIB": "0", "SOL_H3_COMPILE_CACHE": str(cache),
                               "CUDA_VISIBLE_DEVICES": "", "SOL_H3_SECONDS": "10"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        run = next((self.rt / "outputs" / "bench").glob("replay-*-e2e"))
        envs = sorted(run.glob("env-*.json"))
        self.assertEqual([e.name for e in envs],
                         ["env-1-landscape-5s-1344x768.json", "env-2-landscape-5s-672x384.json"])
        env = json.loads(envs[0].read_text())
        self.assertEqual(env["cwd"], str(self.pkg))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(self.pkg))
        self.assertEqual(env["SOL_H3_COMPILE_CACHE"], str(run / "compile"))
        self.assertTrue(env["cache_copied"])
        self.assertEqual((env["OMP_NUM_THREADS"], env["PYTORCH_CUDA_ALLOC_CONF"], env["MIMALLOC_PURGE_DELAY"],
                          env["TRITON_CACHE_AUTOTUNING"], env["SOL_H3_SECONDS"]),
                         ("1", "expandable_segments:True", "0", "1", None))
        self.assertFalse((run / "compile").exists())
        self.assertTrue((cache / "stage2" / "inductor" / "entry").is_file())
        self.assertIn("MEMGUARD: exit=0", (run / "logs" / "1-landscape-5s-1344x768.log").read_text())
        summary = (run / "summary.md").read_text()
        self.assertIn("4 of 4 planned runs passed", summary)
        self.assertIn("| d5 | baseline | 1344x768 | 3 | 1 |", summary)
        with open(run / "runs.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["arm"], r["repeat"], r["same_as_studio"]) for r in rows if r["arm"] == "baseline"],
                         [("baseline", "0", "n/a: first run in this process"), ("baseline", "1", "yes")])

    def test_no_switchboard_sudo_or_studio_ports(self):
        files = [REPO / "bench" / "replay.sh"] + sorted((REPO / "bench" / "replay").glob("*.py"))
        for f in files:
            text = f.read_text()
            for word in ("switchboard", "sudo", "8188", "8189", "8190", "8191", "8800", "8889", "12000"):
                self.assertNotIn(word, text, f"{f.name} mentions {word}")


# Stands in for "<stage2 python> -m torch.distributed.run ... --module replay.stage2 --plan P --group G": records its
# environment and writes one PASS row per planned run, with the studio's clip as every run's output.
STAND_IN = r'''#!/usr/bin/env bash
exec python3 - "$@" <<'PY'
import json, os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
plan = json.loads(Path(a[a.index("--plan") + 1]).read_text())
gid = a[a.index("--group") + 1]
group = next(g for g in plan["groups"] if g["id"] == gid)
out = Path(plan["out"])
keys = ("PYTHONPATH", "SOL_H3_COMPILE_CACHE", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "MIMALLOC_PURGE_DELAY",
        "TRITON_CACHE_AUTOTUNING", "SOL_H3_SECONDS")
env = {k: os.environ.get(k) for k in keys}
env.update(cwd=os.getcwd(), cache_copied=(Path(os.environ["SOL_H3_COMPILE_CACHE"]) / "stage2/inductor/entry").is_file())
(out / f"env-{gid}.json").write_text(json.dumps(env))
for i, run in enumerate(group["runs"]):
    d = plan["drafts"][run["draft"]]
    target = out / run["dir"] / "refined.mp4"
    target.parent.mkdir(parents=True)
    shutil.copy(Path(d["dir"]) / d["studio"]["mp4"], target)
    s = group["arms"][run["arm"]]
    row = {"group": gid, "draft": run["draft"], "arm": run["arm"], "repeat": run["repeat"], "status": "PASS",
           "request_index": i, "stage2_s": 36.0, "output": str(target), "stage2_audio": "frozen", "settings": s,
           "relaxes": s["relaxes"], "phases_s": {}, "step_s": []}
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
PY
'''


# ---- the GPU run object, with stand-ins for torch, the session and the LTX tiling classes -------------------------
class FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(values, device=None, dtype=None):
        return ("tensor", tuple(values), device, dtype)


@dataclass
class Dim:
    tile_size: int
    overlap: int


@dataclass
class Tiles:
    frames: Dim
    height: Dim
    width: Dim

    def validate(self, factors, shape):
        Tiles.validated = (factors, shape)


class FakeCompat:
    DimensionSizeConfig, TileSizeConfig = Dim, Tiles

    @staticmethod
    def tiling_scale_factors_for_vae(path):
        return ("factors", path)

    @staticmethod
    def VideoPixelShape(**kw):
        return kw


class Stats:
    def __init__(self, steps):
        self.steps, self.began = steps, 0

    def begin_denoise(self):
        self.began += 1

    def stats(self):
        n = self.steps
        return {"completed_steps": n, "dense_calls": n, "sol_calls": 47 * n,
                "kernel": {"kernel_calls": 47 * n, "dense_guard_calls": 0, "hunyuan_calls": 0}}


class Upstream:
    def denoise_multimodal(self, *states):
        return "upstream denoise"


class Refiner(Upstream):
    def _checked_attention_stats(self):
        return "runtime's own check"


class GpuRunTests(Fixture):
    def setUp(self):
        super().setUp()
        native, self.sigmas = plan.native_geometry(self.pkg, self.geometry, "landscape", 5)
        models = Refiner()
        models.stage2_sigmas, models.sigmas, models.device = tuple(self.sigmas), "orig tensor", "cuda:0"
        models.tiling_config = Tiles(Dim(128, 24), Dim(448, 64), Dim(768, 64))
        models.sol_attention, models.sol_backend = Stats(3), "triton"
        self.models = models
        self.session = type("S", (), {})()
        self.session.models, self.session.geometry = models, native
        self.session.paths = {"output_video_vae": "/vae"}
        self.original_check = vars(Refiner)["_checked_attention_stats"]

    def run_arm(self, name):
        run = stage2.GpuRun(self.session, FakeTorch, FakeCompat, self.sigmas)
        run.apply(name)
        return run

    def test_baseline_keeps_the_runtime(self):
        run = self.run_arm("baseline")
        self.assertEqual(run.actual()["checks"], "runtime's own")
        self.assertIs(vars(Refiner)["_checked_attention_stats"], self.original_check)
        self.assertEqual(self.models.sigmas, "orig tensor")
        run.undo()

    def test_steps2_relaxes_and_restores(self):
        run = self.run_arm("steps2_from_0725")
        self.assertEqual(self.models.stage2_sigmas, (0.725, 0.421875, 0.0))
        self.assertEqual(self.models.sigmas[1], (0.725, 0.421875, 0.0))
        self.assertEqual(run.actual()["checks"], "relaxed")
        self.models.sol_attention.steps = 2
        self.assertEqual(self.models._checked_attention_stats()["sol_calls"], 94)
        self.models.sol_attention.steps = 3
        with self.assertRaises(RuntimeError):
            self.models._checked_attention_stats()
        run.undo()
        self.assertIs(vars(Refiner)["_checked_attention_stats"], self.original_check)
        self.assertEqual((self.models.stage2_sigmas, self.models.sigmas), (tuple(self.sigmas), "orig tensor"))

    def test_preview_skips_denoise_and_restores(self):
        run = self.run_arm("preview")
        self.models.sol_attention.steps = 0
        self.assertEqual(self.models.denoise_multimodal("v", "a", None, None), ("v", "a"))
        self.assertEqual(self.models.sol_attention.began, 1)
        run.undo()
        self.assertNotIn("denoise_multimodal", vars(Refiner))
        self.assertEqual(self.models.denoise_multimodal(), "upstream denoise")

    def test_tiles_validated_and_restored(self):
        before = self.models.tiling_config
        run = self.run_arm("decode_long768")
        self.assertEqual(self.models.tiling_config, Tiles(Dim(128, 24), Dim(0, 0), Dim(768, 64)))
        self.assertEqual(Tiles.validated[1]["width"], 1344)
        self.assertEqual(run.actual()["tiles"]["height"], {"tile_size": 0, "overlap": 0})
        run.undo()
        self.assertIs(self.models.tiling_config, before)

    def test_contract_and_verify(self):
        model = type("M", (), {})()
        model.sol_backend, model.sol_attention = "triton", Stats(0)
        self.assertEqual(stage2.attention_contract(0)(model)["completed_steps"], 0)
        model.sol_backend = "cudnn"
        with self.assertRaises(RuntimeError):
            stage2.attention_contract(0)(model)
        expected = {"steps": 2, "output": [1344, 768], "frames": 121,
                    "tiles": {"frames": [128, 24], "height": [448, 64], "width": [768, 64]}}
        receipt = {"result": {"attention": {"completed_steps": 2, "sol_calls": 94},
                              "output_info": {"width": 1344, "height": 768, "frames": 121},
                              "official_vae_tiling": {"config": {"frames": {"tile_size": 128, "overlap": 24},
                                                                 "height": {"tile_size": 448, "overlap": 64},
                                                                 "width": {"tile_size": 768, "overlap": 64}}}}}
        self.assertEqual(stage2.verify(receipt, expected), [])
        receipt["result"]["attention"]["completed_steps"] = 3
        receipt["result"]["official_vae_tiling"]["config"]["height"]["tile_size"] = 0
        self.assertEqual(len(stage2.verify(receipt, expected)), 2)

    def test_env_differences(self):
        expected = {"OMP_NUM_THREADS": "1", "PYTHONPATH": "/pkg", "HOME": "/h"}
        self.assertEqual(stage2.env_differences(expected, {"OMP_NUM_THREADS": "1", "PYTHONPATH": "/pkg:/b"}), {})
        self.assertEqual(stage2.env_differences(expected, {}), {"OMP_NUM_THREADS": ("1", None)})


# ---- report, quality and SyncNet ----------------------------------------------------------------------------------
class ReportTests(Fixture):
    def test_same_as_columns_and_summary(self):
        self.add("d5")
        d = drafts.load(self.store, "d5")
        studio = Path(d["dir"]) / d["studio"]["mp4"]
        out = self.tmp / "run"
        p = plan.build([d], plan.parse_arms("baseline:2,preview:1", 2), package=self.pkg, geometry=self.geometry,
                       paths_template=self.paths_template, out=out)
        out.mkdir()
        (out / "plan.json").write_text(json.dumps(p))
        first = tiny_mp4(out / "r0.mp4", color="blue")
        warm1 = out / "r1.mp4"
        shutil.copy(studio, warm1)                      # the warm baseline equals the studio's warm clip
        warm2 = out / "r2.mp4"
        shutil.copy(studio, warm2)
        prev = tiny_mp4(out / "p1.mp4", color="green")
        settings = p["groups"][0]["arms"]
        rows = [(0, "baseline", first, 0), (1, "baseline", warm1, 1), (1, "preview", prev, 2), (2, "baseline", warm2, 3)]
        with open(out / "runs.jsonl", "w") as f:
            for repeat, arm, mp4, index in rows:
                f.write(json.dumps({"group": p["groups"][0]["id"], "draft": "d5", "arm": arm, "repeat": repeat,
                                    "status": "PASS", "request_index": index, "stage2_s": 35.0 + index,
                                    "output": str(mp4), "stage2_audio": "frozen", "settings": settings[arm],
                                    "relaxes": settings[arm]["relaxes"],
                                    "phases_s": {"joint_video_audio_stage2_s": 24.0}, "step_s": [8.1, 8.2, 8.0]})
                        + "\n")
        text, all_ok = report.report(out)
        self.assertTrue(all_ok)
        with open(out / "runs.csv") as f:
            rows = list(csv.DictReader(f))
        by = {(r["arm"], r["repeat"]): r for r in rows}
        self.assertEqual(by[("baseline", "1")]["same_as_baseline"], "ref")
        self.assertEqual(by[("baseline", "2")]["same_as_baseline"], "yes")
        self.assertEqual(by[("preview", "1")]["same_as_baseline"], "no")
        self.assertEqual(by[("baseline", "1")]["same_as_studio"], "yes")
        self.assertEqual(by[("baseline", "0")]["same_as_studio"], "n/a: first run in this process")
        self.assertEqual(by[("baseline", "0")]["same_as_baseline"], "no")
        self.assertEqual(len(report.load_rows(out)), 4)
        self.assertIn("d5 baseline: 2 warm runs, all identical", text)
        self.assertIn("differs from the warm run (the noise floor", text)
        self.assertIn("| landscape | 5 s | d5 | 2 | 37.0 |", text)
        self.assertIn("vs the studio's clip (warm, Stage 2 36.0 s in the studio): yes", text)
        self.assertIn("audio identical in every run", text)
        self.assertIn("4 of 4 planned runs passed", text)

        qdir, complete = quality.quality(out, log=lambda *a: None)
        self.assertIsNone(complete)
        with open(qdir / "quality.csv") as f:
            q = list(csv.DictReader(f))
        tests = {(r["test"], r["reference"]) for r in q}
        self.assertIn(("preview", "baseline"), tests)
        self.assertIn(("studio clip", "baseline"), tests)
        self.assertIn(("baseline first in process", "baseline warm"), tests)
        studio_row = next(r for r in q if r["test"] == "studio clip")
        self.assertEqual(studio_row["psnr_db"], "inf")
        self.assertTrue((qdir / "sheets" / "d5_overview.jpg").is_file())
        self.assertTrue((qdir / "sheets" / "d5_preview_vs_baseline.mp4").is_file())


class SyncnetTests(Fixture):
    def test_table_uses_the_manifest(self):
        # arm names with underscores and digits broke the old file-name parse (IndexError)
        manifest = {"/w/c14__A3_480.mp4": ("c14", "A3_480"), "/w/c14__canvas480_steps2.mp4": ("c14", "canvas480_steps2"),
                    "/w/c14__baseline.mp4": ("c14", "baseline")}
        face = lambda c: {"det_rate": 0.8, "speaking": {"offset": -2, "lse_d": 9.0, "lse_c": c}}
        results = [{"file": "/w/c14__A3_480.mp4", "faces": {"L": face(5.0), "R": face(7.0)}},
                   {"file": "/w/c14__baseline.mp4", "faces": {"L": face(5.5), "R": None}}]
        rows = syncnet.table(results, manifest)
        status = {(r["arm"], r["face"]): r["status"] for r in rows}
        self.assertEqual(status[("A3_480", "L")], "ok")
        self.assertEqual(status[("baseline", "R")], "face not found")
        self.assertEqual(status[("canvas480_steps2", "")], "missing: no result from the scorer")
        self.assertFalse(syncnet.complete(rows, manifest))
        results.append({"file": "/w/c14__canvas480_steps2.mp4", "faces": {"L": face(4.0)}})
        rows = syncnet.table(results, manifest)
        self.assertTrue(syncnet.complete(rows, manifest))
        self.assertIn("Complete: every dialogue draft", syncnet.markdown(rows, manifest))

    def test_score_with_a_stand_in_scorer(self):
        scorer = self.tmp / "scorer.py"
        scorer.write_text(
            "import json, sys\n"
            "a = sys.argv[1:]; out = a[a.index('--json') + 1]\n"
            "turns = a[a.index('--turns') + 1] if '--turns' in a else None\n"
            "files = [x for x in a if x.endswith('.mp4')]\n"
            "t = [{'speaker': 'R', 't0': 0.0, 't1': 1.5}]\n"
            "res = [{'file': f, 'turns': t, 'given': turns, 'faces': {'R': {'det_rate': 1.0, 'speaking':"
            " {'offset': -1, 'lse_d': 8.0, 'lse_c': 7.0}}}} for f in files]\n"
            "json.dump(res, open(out, 'w'))\n")
        clip = tiny_mp4(self.tmp / "clip.mp4")
        self.add("talk", dialogue=True)
        d = {"talk": drafts.load(self.store, "talk")}
        rows, done = syncnet.score(f"{sys.executable} {scorer}", self.tmp / "sync",
                                   {"talk": {"baseline": str(clip), "steps2_from_0725": str(clip)}}, d)
        self.assertTrue(done)
        self.assertEqual(len([r for r in rows if r["status"] == "ok"]), 2)
        self.assertEqual((self.tmp / "sync" / "talk" / "turns.txt").read_text().splitlines(),
                         ["R:0-1.5", "detected on the baseline output"])
        given = {r["given"] for r in json.loads((self.tmp / "sync" / "talk.json").read_text())}
        self.assertEqual(given, {"R:0-1.5"})


if __name__ == "__main__":
    unittest.main()
