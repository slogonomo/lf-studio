"""LFSTUDIO_BACKEND (docs/RTX5090.md): the launcher, the server, the page and the install scripts with no backend set
(the DGX Spark, as before), with rtx5090, and with a wrong value.

CPU only, against the stub runtime. nvidia-smi, uname, gcc, docker and sg are small fakes on PATH, so the GPU, the
CPU type and the compiler of the machine running the tests do not matter. The install scripts run with --dry-run
only, with their home, settings file and temp files in a temp directory.
"""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio, free_port  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUB = Path(__file__).resolve().parent / "stub"
LAUNCHER = REPO / "bin" / "lf-studio"
RTX_BASE = "801f04055badca8cc1f3644e973188068f14ad98"

FAKE_NVIDIA_SMI = """#!/bin/bash
# a stand-in for nvidia-smi: FAKE_GPU, FAKE_DRIVER, FAKE_VRAM_FREE_MIB (read-only queries only)
case "$*" in
    *--query-gpu=name,driver_version*) echo "${FAKE_GPU:-NVIDIA GB10}, ${FAKE_DRIVER:-580.178.04}" ;;
    *--query-gpu=name*) echo "${FAKE_GPU:-NVIDIA GB10}" ;;
    *--query-gpu=memory.free*) echo "${FAKE_VRAM_FREE_MIB:-30000}" ;;
    *--query-compute-apps*) ;;
    *) exit 1 ;;
esac
"""
FAKE_UNAME = """#!/bin/bash
if [[ "$1" == -m ]]; then echo "${FAKE_ARCH:-aarch64}"; else exec /usr/bin/env -u FAKE_ARCH uname "$@"; fi
"""
FAKE_GCC = """#!/bin/bash
if [[ "$1" == -dumpversion ]]; then echo "${FAKE_GCC:-16}"; else exit 1; fi
"""
FAILING = "#!/bin/bash\nexit 1\n"          # docker and sg: no Docker here


def fake_bin(directory):
    """A directory of fakes to put first on PATH."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in (("nvidia-smi", FAKE_NVIDIA_SMI), ("gcc", FAKE_GCC), ("docker", FAILING), ("sg", FAILING)):
        (d / name).write_text(text)
        (d / name).chmod(0o755)
    # uname: the fake must not find itself again on PATH
    real_uname = shutil.which("uname")
    (d / "uname").write_text(FAKE_UNAME.replace("/usr/bin/env -u FAKE_ARCH uname", real_uname))
    (d / "uname").chmod(0o755)
    (d / "python3.12").symlink_to(sys.executable)    # the Spark's install scripts call python3.12 by name
    return d


def clean_env(**settings):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_", "FAKE_", "SPARK_", "PIP_", "UV_", "FORCE"))}
    env.update(PYTHONDONTWRITEBYTECODE="1")
    env.update(settings)
    return env


def rtx_package(sana, offload=True):
    """The stub runtime as a patched Sol-H3-RTX5090 package (runtime/offload.py exists only in that package)."""
    package = Path(sana) / "models" / "minimax_h3" / "Sol-H3-RTX5090"
    shutil.copytree(STUB / "runtime", package / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
    if offload:
        (package / "runtime" / "offload.py").write_text("# stand-in: only Sol-H3-RTX5090 has this module\n")
    return package


def spark_package(sana):
    """The stub as the patched Sol-H3-Spark package (as in test_server.LauncherSettingsCheck)."""
    (Path(sana) / "models" / "minimax_h3").mkdir(parents=True, exist_ok=True)
    package = Path(sana) / "models" / "minimax_h3" / "Sol-H3-Spark"
    package.symlink_to(STUB)
    return package


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1048576
    return 0


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-backend-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        self.sana = self.d / "sana"
        for task in ("t2va", "fl2va", "ref2va"):
            (self.d / f"paths-{task}.json").write_text("{}")

    def launch_env(self, *args, **settings):
        """The launcher's environment for these arguments: everything in the temp directory."""
        env = clean_env(LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_SANA=str(self.sana),
                        LFSTUDIO_PYTHON=sys.executable, LFSTUDIO_PATHS=str(self.d / "paths-{task}.json"),
                        SOL_H3_SPARK_RUNTIME_ROOT=str(self.d / "runtime"), LFSTUDIO_STATE_DIR=str(self.d / "state"),
                        LFSTUDIO_HOST="127.0.0.1", LFSTUDIO_PORT=str(self.port),
                        LFSTUDIO_SERVER=str(self.d / "no-server.py") if "--kill" in args else str(REPO / "webui/server.py"),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}")
        env.update(settings)
        return env

    def launch(self, *args, **settings):
        run = subprocess.run(["bash", str(LAUNCHER), *(args or ("--check",))], env=self.launch_env(*args, **settings),
                             capture_output=True, text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    port = None

    def check(self, **settings):
        self.port = self.port or free_port()
        return self.launch("--check", **settings)


class LauncherSpark(LauncherTest):
    """Without LFSTUDIO_BACKEND the launcher does what it did before: the Sol-H3-Spark package, the Docker checks and
    the 110 GiB floor; LFSTUDIO_BACKEND=spark is the same."""

    def setUp(self):
        super().setUp()
        self.package = spark_package(self.sana)

    def test_unset_is_the_spark(self):
        code, out = self.check(LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 0, out)
        self.assertIn(f"checking (package {self.package}, port", out)
        self.assertIn("checks passed", out)
        self.assertNotIn("GPU memory check", out)                      # no VRAM check on the Spark
        status = lambda text: [line for line in text.splitlines() if line.startswith(">>> STATUS:")]  # noqa: E731
        again = self.check(LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0", LFSTUDIO_BACKEND="spark")
        self.assertEqual((again[0], status(again[1])), (code, status(out)))     # log lines carry times and MemFree

    def test_unset_needs_docker(self):
        code, out = self.check(LFSTUDIO_MIN_FREE_GIB="0")                  # docker and sg fail (fakes)
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - cannot talk to docker", out)

    def test_unset_floor_is_110(self):
        avail = mem_available_gib()
        code, out = self.check(LFSTUDIO_TEST_SKIP_DOCKER="1")
        if avail < 108:
            self.assertEqual(code, 1, out)
            self.assertIn("REFUSED - LF Studio needs ~110 GiB free memory", out)
        elif avail > 112:
            self.assertEqual(code, 0, out)
            self.assertIn("(need 110)", out)

    def test_vram_setting_ignored(self):
        """LFSTUDIO_MIN_FREE_VRAM_GIB is an RTX 5090 setting: the Spark's GPU memory is its RAM."""
        code, out = self.check(LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0",
                               LFSTUDIO_MIN_FREE_VRAM_GIB="999", FAKE_VRAM_FREE_MIB="10")
        self.assertEqual(code, 0, out)
        self.assertIn("note: LFSTUDIO_MIN_FREE_VRAM_GIB is set, but only the RTX 5090 reads it", out)

    def test_rtx_package_refused(self):
        """A Sol-H3-Spark directory with runtime/offload.py is not the Spark's package."""
        sana = self.d / "sana-odd"
        package = sana / "models" / "minimax_h3" / "Sol-H3-Spark"
        shutil.copytree(STUB / "runtime", package / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        (package / "runtime" / "offload.py").write_text("")
        code, out = self.check(LFSTUDIO_SANA=str(sana), LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("which only Sol-H3-RTX5090 has; this is not the Sol-H3-Spark package", out)


class LauncherRtx5090(LauncherTest):
    """LFSTUDIO_BACKEND=rtx5090: the Sol-H3-RTX5090 package, no Docker, its RAM floor and a free GPU memory floor."""

    def setUp(self):
        super().setUp()
        self.package = rtx_package(self.sana)

    def test_passes_without_docker(self):
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="30000")
        self.assertEqual(code, 0, out)
        self.assertIn(f"checking (package {self.package}, port", out)
        self.assertIn("GPU memory check OK: 29 GiB free (need 25)", out)
        self.assertNotIn("docker", out.lower())

    def test_memory_folders(self):
        """The folders the launcher makes: "~/..." under HOME, a value of only spaces as unset (the default next to the
        checkpoints), an off word not at all."""
        home, ckpt = self.d / "home", self.d / "models" / "checkpoints"
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="30000",
                               HOME=str(home), LFSTUDIO_CHECKPOINTS=str(ckpt), SOL_H3_FILEBACKED_DIR="~/scratch-x",
                               SOL_H3_STAGE1_SNAPSHOTS="   ", SOL_H3_LTX_BLOCKS=" OFF ")
        self.assertEqual(code, 0, out)
        self.assertTrue((home / "scratch-x").is_dir())
        self.assertTrue((self.d / "models" / "derived" / "stage1-snapshots").is_dir())
        self.assertFalse((self.d / "models" / "derived" / "ltx-fused-blocks").exists())
        self.assertFalse(Path("~").exists())

    def test_ram_floor(self):
        avail = mem_available_gib()
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", FAKE_VRAM_FREE_MIB="30000")
        if avail < 42:
            self.assertEqual(code, 1, out)
            self.assertIn("REFUSED - LF Studio needs ~42 GiB free memory", out)
        else:
            self.assertEqual(code, 0, out)
            self.assertIn("(need 42)", out)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="100000")
        self.assertEqual(code, 1, out)
        self.assertIn("needs ~100000 GiB free memory", out)
        self.assertIn("note: LFSTUDIO_MIN_FREE_GIB=100000 is more than this machine's", out)
        self.assertIn("the default here is 42", out)

    def test_vram_floor(self):
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="10000")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - LF Studio needs ~25 GiB free GPU memory, only 9 GiB free", out)
        self.assertIn("LFSTUDIO_MIN_FREE_VRAM_GIB=0", out)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", LFSTUDIO_MIN_FREE_VRAM_GIB="8",
                               FAKE_VRAM_FREE_MIB="10000")
        self.assertEqual(code, 0, out)
        self.assertIn("GPU memory check OK: 9 GiB free (need 8)", out)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="[N/A]")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - cannot read the free GPU memory from nvidia-smi", out)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", LFSTUDIO_MIN_FREE_VRAM_GIB="0",
                               FAKE_VRAM_FREE_MIB="[N/A]")
        self.assertEqual(code, 0, out)
        self.assertNotIn("GPU memory check", out)

    def test_patch_settings_checked(self):
        """The patch's settings checks run in this package too."""
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", SOL_H3_PROMPT_TOKENS="special")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - SOL_H3_PROMPT_TOKENS must be one of", out)

    def test_stage2_keyframe_is_spark_only(self):
        """SOL_H3_STAGE2_KEYFRAME (#57) is in the Spark patch only so far: with an RTX 5090 patch without it, the
        default (off) passes, and pin or draft are refused as DGX Spark only, not sent to install/10-sana.sh."""
        with (self.package / "runtime" / "config.py").open("a") as f:
            f.write("\ndel stage2_keyframe_mode   # the RTX 5090 patch has no such setting yet\n")
        settings = dict(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="30000")
        self.assertEqual(self.check(**settings)[0], 0)
        self.assertEqual(self.check(SOL_H3_STAGE2_KEYFRAME="off", **settings)[0], 0)
        for value in ("pin", "Draft"):
            code, out = self.check(SOL_H3_STAGE2_KEYFRAME=value, **settings)
            self.assertEqual(code, 1, out)
            self.assertIn(f"REFUSED - SOL_H3_STAGE2_KEYFRAME={value}: the RTX 5090 patch does not have this setting "
                          "yet (DGX Spark only) (see bin/env.example)", out)
            self.assertNotIn("install/10-sana.sh", out)

    def test_no_upscale(self):
        """The experimental Upscale (LFSTUDIO_UPSCALE=1) needs the DGX Spark: refused first; 0 passes."""
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="1", FAKE_VRAM_FREE_MIB="30000")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - LFSTUDIO_UPSCALE=1: the experimental Upscale runs on the DGX Spark only, not with "
                      "LFSTUDIO_BACKEND=rtx5090", out)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="0", LFSTUDIO_MIN_FREE_GIB="0",
                               FAKE_VRAM_FREE_MIB="30000")
        self.assertEqual(code, 0, out)


class LauncherBackendRefusals(LauncherTest):
    def test_invalid_backend(self):
        spark_package(self.sana)
        for args in (("--check",), ("--kill",)):
            self.port = free_port()
            code, out = self.launch(*args, LFSTUDIO_BACKEND="gb10", LFSTUDIO_TEST_SKIP_DOCKER="1",
                                    LFSTUDIO_MIN_FREE_GIB="0")
            self.assertEqual(code, 1, (args, out))
            self.assertIn("REFUSED - LFSTUDIO_BACKEND=gb10: must be spark or rtx5090", out)

    def test_mismatched_package(self):
        """A checkout patched for the other backend is named, with the setting that fits it."""
        spark_package(self.sana)
        code, out = self.check(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-Spark, not Sol-H3-RTX5090: set LFSTUDIO_BACKEND=spark", out)

        rtx = self.d / "sana-rtx"
        rtx_package(rtx)
        on_a_5090 = dict(FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        code, out = self.check(LFSTUDIO_SANA=str(rtx), LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0",
                               **on_a_5090)
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-RTX5090, not Sol-H3-Spark: set LFSTUDIO_BACKEND=rtx5090 "
                      "(it is spark, the default)", out)
        code, out = self.check(LFSTUDIO_SANA=str(rtx), LFSTUDIO_BACKEND="spark", LFSTUDIO_TEST_SKIP_DOCKER="1",
                               LFSTUDIO_MIN_FREE_GIB="0", **on_a_5090)
        self.assertEqual(code, 1, out)
        self.assertIn("set LFSTUDIO_BACKEND=rtx5090 (it is spark), or point", out)       # set by the user: no "default"
        # On a GB10 the other backend can't run: the checkout is the wrong one, and the answer says so.
        code, out = self.check(LFSTUDIO_SANA=str(rtx), LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-RTX5090, but this machine needs Sol-H3-Spark", out)
        self.assertNotIn("set LFSTUDIO_BACKEND=rtx5090", out)

        unpatched = self.d / "sana-plain"
        (unpatched / "models/minimax_h3/Sol-H3-RTX5090/runtime").mkdir(parents=True)
        (unpatched / "models/minimax_h3/Sol-H3-RTX5090/runtime/offload.py").write_text("")
        code, out = self.check(LFSTUDIO_SANA=str(unpatched), LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("is not patched (runtime/geometry.py missing); apply patches/lf-studio-rtx5090.patch", out)

        code, out = self.check(LFSTUDIO_SANA=str(self.d / "nothing"), LFSTUDIO_BACKEND="rtx5090",
                               LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - no Sol-H3-RTX5090 package at", out)

        no_offload = self.d / "sana-no-offload"
        rtx_package(no_offload, offload=False)
        code, out = self.check(LFSTUDIO_SANA=str(no_offload), LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("is not NVIDIA's Sol-H3-RTX5090 package (runtime/offload.py missing)", out)


class LauncherWorkers(LauncherTest):
    """--kill stops the pipeline workers of its backend's package (WORKER_RE) and leaves the other package's alone."""

    def sleeper(self, *argv):
        """A process whose command line names argv, as a worker's names its package or options file."""
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", *argv])
        self.addCleanup(lambda: (p.poll() is None and p.kill(), p.wait()))
        return p

    def stopped(self, p, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if p.poll() is not None:
                return True
            time.sleep(0.05)
        return False

    def test_kill_matches_rtx5090_workers(self):
        runtime = self.d / "runtime"
        rtx_worker = self.sleeper(str(self.sana / "models/minimax_h3/Sol-H3-RTX5090/runtime/worker.py"))
        by_options = self.sleeper("runtime.worker", "--module", "stage2", "--options",
                                  str(runtime / "outputs/webui/s1/options.json"))
        spark_worker = self.sleeper(str(self.sana / "models/minimax_h3/Sol-H3-Spark/runtime/worker.py"))
        time.sleep(0.3)
        self.port = free_port()
        code, out = self.launch("--kill", LFSTUDIO_BACKEND="rtx5090")
        self.assertEqual(code, 0, out)
        self.assertTrue(self.stopped(rtx_worker), "the Sol-H3-RTX5090 worker still runs")
        self.assertTrue(self.stopped(by_options), "the worker named by its options file still runs")
        self.assertFalse(self.stopped(spark_worker, timeout=1), "the Sol-H3-Spark worker was stopped too")
        code, out = self.launch("--kill", LFSTUDIO_TEST_SKIP_DOCKER="1")        # no backend: the Spark's workers
        self.assertEqual(code, 0, out)
        self.assertTrue(self.stopped(spark_worker), "the Sol-H3-Spark worker still runs")


class ServerBackend(unittest.TestCase):
    """The server reports LFSTUDIO_BACKEND in /api/state ("backend", spark when unset), names it in the page's
    lf-studio-backend meta tag, and refuses another value. Nothing else depends on it."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-backend-"))
        self.addCleanup(shutil.rmtree, self.d, True)

    def studio(self, name, **env):
        s = StubStudio(self.d / name, load_s=1, clip_s=1, preload=False)
        s.env.pop("LFSTUDIO_BACKEND", None)
        s.env.update(env)
        return s

    def test_unset_is_spark(self):
        s = self.studio("unset").start()
        try:
            self.assertEqual(s.state()["backend"], "spark")
            page = s.get("/")[2].decode()
            self.assertEqual(page, (REPO / "webui" / "index.html").read_text())     # the Spark's page as it is
            self.assertIn('<meta name="lf-studio-backend" content="spark">', page)
        finally:
            s.stop()

    def test_rtx5090(self):
        s = self.studio("rtx", LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="0").start()   # (the stub turns it on)
        try:
            self.assertEqual(s.state()["backend"], "rtx5090")
            self.assertIs(s.state()["upscale"], False)
            page = s.get("/")[2].decode()
            self.assertIn('<meta name="lf-studio-backend" content="rtx5090">', page)
            self.assertNotIn('content="spark"', page)
        finally:
            s.stop()

    def test_invalid_refused(self):
        with self.assertRaises(RuntimeError) as e:
            self.studio("bad", LFSTUDIO_BACKEND="gb10").start()
        self.assertIn("LFSTUDIO_BACKEND=gb10: must be spark or rtx5090", str(e.exception))

    def test_stage2_keyframe_is_spark_only(self):
        """SOL_H3_STAGE2_KEYFRAME (#57): a package without the setting, as the RTX 5090 patch is so far, starts with the
        default; pin stops the server, on the RTX 5090 as DGX Spark only, on the Spark as an older patch."""
        package = self.d / "package"
        shutil.copytree(STUB / "runtime", package / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        with (package / "runtime" / "config.py").open("a") as f:
            f.write("\ndel stage2_keyframe_mode   # a patch without the setting\n")
        s = self.studio("rtx-off", LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="0")
        s.package = package
        s.start()
        s.stop()
        for name, env, refusal in (
                ("rtx-pin", dict(LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="0"),
                 "SOL_H3_STAGE2_KEYFRAME=pin: the RTX 5090 patch does not have this setting yet (DGX Spark only"),
                ("spark-pin", {}, f"SOL_H3_STAGE2_KEYFRAME=pin needs a newer patch than the one applied in "
                                  f"{package.resolve()}; run install/10-sana.sh")):
            s = self.studio(name, SOL_H3_STAGE2_KEYFRAME="pin", **env)
            s.package = package
            with self.assertRaises(RuntimeError) as e:
                s.start()
            self.assertIn(refusal, str(e.exception))

    def test_upscale_refused_on_rtx5090(self):
        """The experimental Upscale needs the DGX Spark: the server stops with LFSTUDIO_UPSCALE=1 on the RTX 5090."""
        with self.assertRaises(RuntimeError) as e:
            self.studio("rtx-upscale", LFSTUDIO_BACKEND="rtx5090", LFSTUDIO_UPSCALE="1").start()
        self.assertIn("LFSTUDIO_UPSCALE=1: the experimental Upscale runs on the DGX Spark only, not with "
                      "LFSTUDIO_BACKEND=rtx5090", str(e.exception))
        s = self.studio("spark-upscale", LFSTUDIO_UPSCALE="1").start()      # unset backend: the Spark
        try:
            self.assertIs(s.state()["upscale"], True)
        finally:
            s.stop()


PAGE = (REPO / "webui" / "index.html").read_text()


def machines():
    """The page's per-backend texts and clip times (JSON between its /*machines*/ markers)."""
    m = re.search(r"const MACHINES=/\*machines\*/(.*?)/\*machines\*/;", PAGE, re.S)
    return json.loads(m.group(1))


class PageTexts(unittest.TestCase):
    """The page's texts per backend, without a browser (tests/test_page.py has the browser test)."""

    def test_table(self):
        m = machines()
        self.assertEqual(set(m), {"spark", "rtx5090"})
        offered = set(re.findall(r'<input type="radio" name="seconds" id="s\d+" value="(\d+)">', PAGE))
        for backend, texts in m.items():
            self.assertEqual(set(texts["eta"]), offered, backend)           # a time for every length offered
            self.assertTrue(all(isinstance(v, int) and v > 0 for v in texts["eta"].values()), backend)
        self.assertEqual(m["spark"]["name"], "one DGX Spark")
        self.assertEqual(m["rtx5090"]["name"], "one RTX 5090")
        self.assertIn("measured on an RTX 5090", "".join(m["rtx5090"]["times"]))   # measured (docs/RTX5090.md)
        self.assertIn("measured on a DGX Spark", "".join(m["spark"]["times"]))

    def test_static_text_is_the_spark(self):
        """Before any script runs, the page reads as it did: the Spark's texts and times."""
        spark = machines()["spark"]
        self.assertIn(f'video with sound, on <span id="machine">{spark["name"]}</span></p>', PAGE)
        t = spark["times"]
        self.assertIn(f'<span id="timesnote">{t[0]}<b>{t[1]}</b>{t[2]}</span>', PAGE)
        self.assertEqual(spark["eta"], {"5": 60, "6": 70, "7": 80, "8": 90, "10": 125, "12": 150, "15": 195})

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_apply_machine(self):
        """The page's own applyMachine (from the MACHINES table to the call that reads the meta tag) with a small
        stand-in DOM in node: the texts and ETA table it picks per backend."""
        start = PAGE.index("const MACHINES=")
        end = PAGE.index("\n", PAGE.index("applyMachine((document.querySelector"))
        block = PAGE[start:end]
        harness = """
const els={machine:{textContent:'one DGX Spark'},b:{textContent:'measured on a DGX Spark'},
  timesnote:{firstChild:{textContent:'Times shown are '},lastChild:{textContent:' with nothing else running.'}}};
els.timesnote.querySelector=()=>els.b;
const $=id=>els[id];
function setText(el,t){if(el.textContent!==t)el.textContent=t;}
const document={querySelector:()=>(BACKEND===null?null:{content:BACKEND})};
%s
console.log(JSON.stringify({machine,name:els.machine.textContent,times:[els.timesnote.firstChild.textContent,
  els.b.textContent,els.timesnote.lastChild.textContent],eta5:ETA[5]}));
"""
        m = machines()
        for meta, expect in (("spark", "spark"), ("rtx5090", "rtx5090"), ("gb10", "spark"), (None, "spark")):
            src = f"const BACKEND={json.dumps(meta)};" + harness % block
            run = subprocess.run(["node", "-e", src], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            got = json.loads(run.stdout)
            self.assertEqual(got["machine"], expect, meta)
            self.assertEqual(got["name"], m[expect]["name"], meta)
            self.assertEqual(got["times"], m[expect]["times"], meta)
            self.assertEqual(got["eta5"], m[expect]["eta"]["5"], meta)

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_script_parses(self):
        script = PAGE[PAGE.index("<script>") + len("<script>"):PAGE.rindex("</script>")]
        with tempfile.NamedTemporaryFile("w", suffix=".js") as f:
            f.write(script)
            f.flush()
            run = subprocess.run(["node", "--check", f.name], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)


INSTALL = sorted(p.name for p in (REPO / "install").glob("[0-9][0-9]-*.sh"))


class InstallDryRun(unittest.TestCase):
    """Every install script with --dry-run under both backends (and none), with a fake GPU, CPU type and gcc."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-install-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        (self.d / "home").mkdir()
        (self.d / "tmp").mkdir()

    def run_script(self, name, *args, repo=REPO, **settings):
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                        LFSTUDIO_HOME=str(self.d / "home" / "sol-h3"),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}")
        env.update(settings)
        run = subprocess.run(["bash", str(Path(repo) / "install" / name), "--dry-run", *args], env=env, capture_output=True,
                             text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    def all_scripts(self, **settings):
        return {name: self.run_script(name, **settings) for name in INSTALL}

    def spark(self, **settings):
        return self.all_scripts(FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10", **settings)

    def rtx(self, **settings):
        return self.all_scripts(FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090", LFSTUDIO_BACKEND="rtx5090",
                                **settings)

    def assert_pillow_13(self, text, stage, removed=""):
        """#69: Pillow 13.0.0 in the venv's own install list and by pillow_update, which also removes the packages
        that cap it below 13 (Stage 1: Gradio and MoviePy, no longer installed)."""
        py = f"{self.d}/home/sol-h3/envs/{stage}/bin/python"
        self.assertIn('pillow==13.0.0', text.split("Pillow >= 13 in")[0])          # the build's list
        self.assertIn(f"Pillow >= 13 in {self.d}/home/sol-h3/envs/{stage} (", text)
        self.assertIn(f"+ {py} -m pip install --only-binary pillow pillow==13.0.0", text)
        if removed:
            self.assertIn(f"+ {py} -m pip uninstall -y {removed}", text)
        else:
            self.assertNotIn("pip uninstall", text)
        self.assertNotIn("gradio==", text)
        self.assertNotIn("moviepy>=", text)

    def assert_pip_update(self, text, venv, build=False):
        """pip 26.2.2 and certifi >= 2026.10.19 through the venv's python (never bin/pip: a copied venv's bin/pip
        installs into the original), before Pillow; on a new venv also before anything else is installed."""
        py = f"{self.d}/home/sol-h3/envs/{venv}/bin/python"
        pip, certifi = (f"+ {py} -m pip install --no-deps pip==26.2.2",
                        f"+ {py} -m pip install --no-deps certifi\\>=2026.10.19")
        self.assertIn(pip, text)
        self.assertIn(certifi, text)
        self.assertLess(text.index(pip), text.index(certifi))
        if "Pillow >= 13 in" in text:
            self.assertLess(text.rindex(certifi), text.index("Pillow >= 13 in"))
        if build:
            installs = [i for i, line in enumerate(text.splitlines()) if " -m pip install" in line]
            self.assertIn(pip, text.splitlines()[installs[0]])
        self.assertNotIn("bin/pip ", text)
        self.assertNotIn("pip==25.2", text)

    def test_scripts(self):
        self.assertEqual(INSTALL, ["00-check.sh", "05-python.sh", "10-sana.sh", "20-weights.sh", "30-sources.sh",
                                   "40-stage1.sh", "50-stage2.sh", "60-qwen-image.sh", "65-regen2k.sh",
                                   "70-manifests.sh", "80-prompt-cache.sh", "90-env.sh"])

    def test_spark(self):
        out = self.spark()
        pid = re.compile(r"probe-\d+")                                # 30-sources.sh names a file after its pid
        self.assertEqual({k: (c, pid.sub("probe", t)) for k, (c, t) in out.items()},
                         {k: (c, pid.sub("probe", t)) for k, (c, t) in self.spark(LFSTUDIO_BACKEND="spark").items()})
        base = (REPO / "patches" / "BASE_COMMIT").read_text().strip()
        self.assertIn("ok    aarch64", out["00-check.sh"][1])
        self.assertIn("ok    GPU: NVIDIA GB10", out["00-check.sh"][1])
        self.assertIn("+ docker info   (dry run: not run)", out["00-check.sh"][1])
        self.assertEqual(out["05-python.sh"][0], 0)
        self.assertIn("nothing to do", out["05-python.sh"][1])
        self.assertIn(f"sol-engine @ {base}", out["10-sana.sh"][1])
        self.assertIn("patches/lf-studio.patch", out["10-sana.sh"][1])
        self.assertIn("154 CPU tests", out["10-sana.sh"][1])
        self.assertIn("python3.12 -m venv", out["20-weights.sh"][1])
        self.assertIn("models/minimax_h3/Sol-H3-Spark &&", out["20-weights.sh"][1])
        self.assertIn("--include-offline", out["20-weights.sh"][1])
        self.assertIn("python3.12 prepare.py --fetch-sources", out["30-sources.sh"][1])
        self.assertIn("uv-aarch64-unknown-linux-gnu.tar.gz", out["65-regen2k.sh"][1])        # InstallRegen2k: more
        code, text = out["40-stage1.sh"]
        if code == 0:                                       # needs /usr/local/cuda, as before
            self.assertIn("fastvideo-kernel", text)
            self.assertNotIn("comfy-kitchen", text)
            self.assert_pillow_13(text, "stage1", "gradio gradio-client moviepy")
            self.assert_pip_update(text, "stage1", build=True)
        else:
            self.assertIn("CUDA 13 toolkit not found at /usr/local/cuda", text)
        self.assertIn('TORCH_CUDA_ARCH_LIST=12.0\\;12.1', out["50-stage2.sh"][1])
        self.assertIn("offline-context", out["50-stage2.sh"][1])
        self.assertNotIn("micromamba", out["50-stage2.sh"][1])
        self.assert_pillow_13(out["50-stage2.sh"][1], "stage2")
        self.assert_pip_update(out["50-stage2.sh"][1], "stage2", build=True)
        self.assert_pip_update(out["20-weights.sh"][1], "download", build=True)
        self.assertIn("docker build -f Dockerfile.qwen", out["60-qwen-image.sh"][1])
        self.assertRegex(out["60-qwen-image.sh"][1], r"docker build --build-arg BASE=sol-h3-spark-qwen:base "
                                                     r"--label lf-studio\.layer=[0-9a-f]{16} -t sol-h3-spark-qwen ")
        self.assertIn("--python-qwen", out["70-manifests.sh"][1])
        self.assertIn("Sol-H3-Spark/qwen_python.sh", out["70-manifests.sh"][1])
        self.assertIn("--cache-pending", out["70-manifests.sh"][1])
        self.assertIn("runtime.cache_builder", out["80-prompt-cache.sh"][1])
        env = out["90-env.sh"][1]
        for line in ("LFSTUDIO_BACKEND=spark", "SOL_H3_SPARK_QWEN_IMAGE=sol-h3-spark-qwen",
                     "LFSTUDIO_MIN_FREE_GIB=110"):
            self.assertIn(line, env)
        for name, (code, text) in out.items():
            if name not in ("00-check.sh", "40-stage1.sh"):        # 00 and 40 check this machine for real
                self.assertEqual(code, 0, f"{name}: {text}")

    def test_rtx5090(self):
        out = self.rtx()
        check = out["00-check.sh"][1]
        for line in ("ok    LFSTUDIO_BACKEND=rtx5090", "ok    x86_64", "ok    GPU: NVIDIA GeForce RTX 5090",
                     "gcc 16 is newer than nvcc 13.2 accepts"):
            self.assertIn(line, check)
        self.assertNotIn("docker", check)
        self.assertNotIn("/usr/local/cuda", check)
        py312 = str(self.d / "home/sol-h3/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12")
        py313 = str(self.d / "home/sol-h3/python/cpython-3.13.16-linux-x86_64-gnu/bin/python3.13")
        text = out["05-python.sh"][1]
        self.assertIn("releases/download/0.13.4/uv-x86_64-unknown-linux-gnu.tar.gz", text)
        self.assertIn("2b428396f12572d4429cf210ca234dd599a33b3af0958a4d5ed569d63e3f344f", text)
        self.assertIn("check the SHA-256", text)
        self.assertIn("python install --no-bin 3.12.15 3.13.16", text)
        text = out["10-sana.sh"][1]
        self.assertIn(f"sol-engine @ {RTX_BASE}", text)
        self.assertIn("patches/lf-studio-rtx5090.patch", text)
        self.assertIn("201 CPU tests", text)
        self.assertIn(f"{py312} -m venv", out["20-weights.sh"][1])
        self.assertIn("models/minimax_h3/Sol-H3-RTX5090 &&", out["20-weights.sh"][1])
        self.assertNotIn("--include-offline", out["20-weights.sh"][1])
        self.assert_pip_update(out["20-weights.sh"][1], "download", build=True)
        self.assertIn(f"{py312} prepare.py --fetch-sources", out["30-sources.sh"][1])
        text = out["40-stage1.sh"][1]
        self.assertIn(f"venv --seed --python {py312}", text)
        self.assertIn("comfy-kitchen", text)
        self.assertIn("ComfyUI/requirements.txt", text)
        self.assertIn("stage1-rtx5090-constraints.txt", text)
        self.assertIn("stage1_dependencies", text)
        self.assertNotIn("fastvideo-kernel", text.replace("no fastvideo-kernel", ""))
        self.assert_pillow_13(text, "stage1", "gradio gradio-client moviepy")
        self.assert_pip_update(text, "stage1", build=True)
        text = out["50-stage2.sh"][1]
        self.assertIn(f"venv --seed --python {py313}", text)
        self.assertIn("TORCH_CUDA_ARCH_LIST=12.0 ", text)
        self.assertIn("micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64", text)
        self.assertIn("gxx_linux-64=14", text)
        self.assertIn("NVCC_CCBIN=", text)
        self.assertIn("CFLAGS=-w", text)
        self.assertIn("not needed on the RTX 5090", text)
        self.assert_pillow_13(text, "stage2")
        self.assert_pip_update(text, "stage2", build=True)
        self.assertEqual(out["60-qwen-image.sh"][0], 0)
        self.assertNotIn("docker build", out["60-qwen-image.sh"][1])
        self.assertIn("nothing to do (the experimental Upscale runs on the DGX Spark only)", out["65-regen2k.sh"][1])
        self.assertNotIn("+ ", out["65-regen2k.sh"][1])
        text = out["70-manifests.sh"][1]
        self.assertIn(f"--python-qwen {self.d}/home/sol-h3/envs/stage1/bin/python", text)
        self.assertNotIn("qwen_python.sh", text)
        self.assertNotIn("--cache-pending", text)
        code, text = out["80-prompt-cache.sh"]
        self.assertEqual(code, 1, text)
        self.assertIn("LFSTUDIO_PROMPT_CACHE_FROM is not set", text)
        self.assertIn("needs an SM121 GPU", text)
        self.assertNotIn("cache_builder", text)
        env = out["90-env.sh"][1]
        self.assertIn("LFSTUDIO_BACKEND=rtx5090", env)
        self.assertNotIn("SOL_H3_SPARK_QWEN_IMAGE", env)
        self.assertNotIn("\n    LFSTUDIO_MIN_FREE_GIB=", env)                  # the launcher's defaults hold
        self.assertIn("# LFSTUDIO_MIN_FREE_GIB, LFSTUDIO_MIN_FREE_VRAM_GIB: the launcher's defaults", env)
        for name, (code, text) in out.items():
            if name not in ("00-check.sh", "80-prompt-cache.sh"):
                self.assertEqual(code, 0, f"{name}: {text}")

    def test_rtx5090_variants(self):
        """With gcc 15 or older no toolchain is installed; a cache from elsewhere is checked, then copied."""
        text = self.run_script("50-stage2.sh", LFSTUDIO_BACKEND="rtx5090", FAKE_GCC="14")[1]
        self.assertNotIn("micromamba", text)
        self.assertNotIn("CC=", text)
        self.assertIn("CFLAGS=-w", text)
        cache = self.d / "spark-fixed-prompt.pt"
        cache.write_bytes(b"stand-in")
        code, text = self.run_script("80-prompt-cache.sh", LFSTUDIO_BACKEND="rtx5090",
                                     LFSTUDIO_PROMPT_CACHE_FROM=str(cache))
        self.assertEqual(code, 0, text)
        self.assertIn("runtime.prompt_cache.load_cache", text)
        self.assertIn(f"+ cp {cache} {self.d}/home/sol-h3/fixed-prompt.pt", text)
        code, text = self.run_script("80-prompt-cache.sh", LFSTUDIO_BACKEND="rtx5090",
                                     LFSTUDIO_PROMPT_CACHE_FROM=str(self.d / "missing.pt"))
        self.assertEqual(code, 1, text)
        self.assertIn("no such file", text)

    def settings(self, *names, **settings):
        """install/env.sh's values of names, as the install scripts see them."""
        env = clean_env(HOME=str(self.d / "home"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                        LFSTUDIO_HOME=str(self.d / "home" / "sol-h3"), **settings)
        script = 'source "$1"; shift; for n in "$@"; do printf "%s=%s\n" "$n" "${!n-<unset>}"; done'
        run = subprocess.run(["bash", "-c", script, "-", str(REPO / "install" / "env.sh"), *names], env=env,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        return dict(line.split("=", 1) for line in run.stdout.splitlines())

    def test_caches(self):
        """On the RTX 5090 pip's and uv's caches are under $LFSTUDIO_HOME; the Spark keeps pip's default, as before."""
        home = self.d / "home" / "sol-h3"
        self.assertEqual(self.settings("PIP_CACHE_DIR", "UV_CACHE_DIR", LFSTUDIO_BACKEND="rtx5090"),
                         {"PIP_CACHE_DIR": f"{home}/cache/pip", "UV_CACHE_DIR": f"{home}/cache/uv"})
        self.assertEqual(self.settings("PIP_CACHE_DIR", LFSTUDIO_BACKEND="rtx5090", PIP_CACHE_DIR="/elsewhere"),
                         {"PIP_CACHE_DIR": "/elsewhere"})
        self.assertEqual(self.settings("PIP_CACHE_DIR", "UV_CACHE_DIR"),
                         {"PIP_CACHE_DIR": "<unset>", "UV_CACHE_DIR": "<unset>"})

    def test_toolchain_kept(self):
        """FORCE=1 rebuilds Stage 2 but keeps the GCC 14 toolchain (the extensions load its libstdc++);
        FORCE_TOOLCHAIN=1 moves the old one aside before micromamba creates the prefix again."""
        tc = self.d / "home/sol-h3/toolchains/gcc14"
        (tc / "bin").mkdir(parents=True)
        (tc / "bin/x86_64-conda-linux-gnu-g++").write_text("#!/bin/sh\n")
        (tc / "bin/x86_64-conda-linux-gnu-g++").chmod(0o755)
        code, text = self.run_script("50-stage2.sh", LFSTUDIO_BACKEND="rtx5090", FORCE="1")
        self.assertEqual(code, 0, text)
        self.assertIn("already there (FORCE_TOOLCHAIN=1 installs it again)", text)
        self.assertNotIn(" create ", text)
        self.assertIn(f"NVCC_CCBIN={tc}/bin/x86_64-conda-linux-gnu-g++", text)
        code, text = self.run_script("50-stage2.sh", LFSTUDIO_BACKEND="rtx5090", FORCE="1", FORCE_TOOLCHAIN="1")
        self.assertEqual(code, 0, text)
        moved = re.search(rf"^\+ mv {re.escape(str(tc))} "
                          rf"{re.escape(str(self.d))}/home/sol-h3/cache/gcc14\.\d{{8}}-\d{{6}}$", text, re.M)
        self.assertTrue(moved, text)
        self.assertLess(moved.start(), text.index(" create -y -p "))

    def fake_venv(self, stage, freeze):
        """A venv whose python prints freeze for `-m pip freeze` (and does nothing else)."""
        py = self.d / f"home/sol-h3/envs/{stage}/bin/python"
        py.parent.mkdir(parents=True)
        (py.parent / "freeze.txt").write_text(freeze)
        py.write_text('#!/bin/bash\nif [[ "$*" == "-m pip freeze" ]]; then cat "$(dirname "$0")/freeze.txt"; fi\n')
        py.chmod(0o755)

    def test_adopt_by_freeze(self):
        """A venv built by hand without the marker is adopted when its pip freeze is the reference freeze, and
        built as usual when any line differs."""
        for stage, script in (("stage1", "40-stage1.sh"), ("stage2", "50-stage2.sh")):
            ref = (REPO / f"install/freeze/{stage}-rtx5090-freeze.txt").read_text()
            freeze = "".join(line + "\n" for line in ref.splitlines() if not line.startswith("#"))
            self.fake_venv(stage, freeze)
            done = self.d / f"home/sol-h3/envs/{stage}/.lf-studio-done"
            code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090")
            self.assertEqual(code, 0, text)
            self.assertIn(f"has no marker but matches install/freeze/{stage}-rtx5090-freeze.txt: adopted", text)
            self.assertIn(f"+ touch {done}", text)
            self.assertNotIn("install -U setuptools wheel", text)
            self.assert_pip_update(text, stage)                                 # adopted: the update steps still run
            self.assertNotIn("ltx-kernels-build", text)
            self.assertNotIn("micromamba", text)
            code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090", FORCE="1")         # FORCE=1 rebuilds
            self.assertNotIn("adopted", text)
            self.assertIn("install -U setuptools wheel", text)
            # after the Pillow step (#69) the venv is still adopted: Pillow, Gradio and MoviePy don't count
            stepped = "".join(line + "\n" for line in freeze.splitlines()
                              if not re.match(r"(?i)(gradio|gradio_client|moviepy)==", line))
            stepped = re.sub(r"(?im)^pillow==.*$", "pillow==13.0.0", stepped)
            (self.d / f"home/sol-h3/envs/{stage}/bin/freeze.txt").write_text(stepped)
            code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090")
            self.assertIn(f"has no marker but matches install/freeze/{stage}-rtx5090-freeze.txt: adopted", text)
            # and after pip_update, also when certifi or Pillow came from a file (`name @ URL` in pip freeze)
            for certifi in ("certifi==2026.10.19", "certifi @ file:///tmp/certifi-2026.10.19-py3-none-any.whl"):
                updated = re.sub(r"(?im)^certifi==.*$", certifi, stepped)
                updated = re.sub(r"(?im)^pillow==.*$", "pillow @ file:///tmp/pillow-13.0.0.whl", updated)
                (self.d / f"home/sol-h3/envs/{stage}/bin/freeze.txt").write_text(updated)
                code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090")
                self.assertIn(f"has no marker but matches install/freeze/{stage}-rtx5090-freeze.txt: adopted", text)
            (self.d / f"home/sol-h3/envs/{stage}/bin/freeze.txt").write_text(freeze + "extra==1.0\n")
            code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090")
            self.assertEqual(code, 0, text)
            self.assertNotIn("adopted", text)
            self.assertIn("install -U setuptools wheel", text)
            self.assertIn(f"+ touch {done}", text)

    def test_marked_venv_gets_the_update_steps(self):
        """A venv with the done marker (also 0.4's) gets pip, certifi and Pillow, and is not built into."""
        for stage, script in (("stage1", "40-stage1.sh"), ("stage2", "50-stage2.sh")):
            self.fake_venv(stage, "")
            for marker in (".lf-studio-done", ".sol-h3-studio-done"):
                with self.subTest(stage=stage, marker=marker):
                    done = self.d / f"home/sol-h3/envs/{stage}/{marker}"
                    done.touch()
                    code, text = self.run_script(script, LFSTUDIO_BACKEND="rtx5090")
                    self.assertEqual(code, 0, text)
                    self.assertIn(f"{stage.capitalize()} env already built", text)
                    self.assert_pip_update(text, stage)
                    for build in ("venv --seed", "install -U setuptools wheel", "torch==", "ltx-kernels-build"):
                        self.assertNotIn(build, text)
                    done.unlink()

    def test_torchaudio_fix_only_installs_nothing(self):
        self.fake_venv("stage2", "")
        for backend in ("spark", "rtx5090"):
            with self.subTest(backend=backend):
                code, text = self.run_script("50-stage2.sh", "--torchaudio-fix-only", LFSTUDIO_BACKEND=backend)
                self.assertEqual(code, 0, text)
                self.assertIn("--torchaudio-fix-only: leaving the rest", text)
                self.assertNotIn(" -m pip ", text)

    def test_prompt_cache_backup_dated(self):
        """FORCE=1 on the RTX 5090 keeps the old cache under a dated name (it cannot be rebuilt there)."""
        home = self.d / "home/sol-h3"
        home.mkdir(parents=True)
        (home / "fixed-prompt.pt").write_bytes(b"old")
        cache = self.d / "spark-fixed-prompt.pt"
        cache.write_bytes(b"stand-in")
        code, text = self.run_script("80-prompt-cache.sh", LFSTUDIO_BACKEND="rtx5090", FORCE="1",
                                     LFSTUDIO_PROMPT_CACHE_FROM=str(cache))
        self.assertEqual(code, 0, text)
        self.assertRegex(text, rf"(?m)^\+ mv {re.escape(str(home))}/fixed-prompt\.pt "
                               rf"{re.escape(str(home))}/fixed-prompt\.pt\.\d{{8}}-\d{{6}}\.bak$")
        self.assertNotIn("mv -f", text)

    def test_detection(self):
        """On an RTX 5090 without LFSTUDIO_BACKEND, 00-check.sh says what to set and 90-env.sh does not write spark."""
        code, text = self.run_script("00-check.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 1, text)
        self.assertIn("this machine looks like LFSTUDIO_BACKEND=rtx5090, but the install uses spark (the default)",
                      text)
        self.assertIn("export LFSTUDIO_BACKEND=rtx5090", text)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 1, text)
        self.assertIn("looks like LFSTUDIO_BACKEND=rtx5090, but LFSTUDIO_BACKEND is not set", text)
        # a copy of bin/env.example as the settings file does not set it either
        example = self.d / "env-from-example"
        shutil.copy(REPO / "bin" / "env.example", example)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090",
                                     LFSTUDIO_ENV=str(example))
        self.assertEqual(code, 1, text)
        self.assertIn("looks like LFSTUDIO_BACKEND=rtx5090, but LFSTUDIO_BACKEND is not set", text)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 4090")
        self.assertEqual(code, 0, text)                                     # nothing detected: spark, as before
        self.assertIn("LFSTUDIO_BACKEND=spark", text)

    def test_sana_old_applied_copy(self):
        """Until 0.6: a checkout last patched by 0.4 keeps its copy of the patch as sol-h3-studio-applied.patch.
        10-sana.sh finds the earlier patch there, replaces it and keeps the new copy under the new name."""
        sana = self.d / "home/sol-h3/Sana"
        git = ["git", "-C", str(sana), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        subprocess.run(["git", "init", "-q", str(sana)], check=True)
        (sana / "infer.py").write_text("print('upstream')\n")
        subprocess.run(git + ["add", "infer.py"], check=True)
        subprocess.run(git + ["commit", "-qm", "base"], check=True)
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        (sana / "infer.py").write_text("# Modified by sol-h3-studio\nprint('patched')\n")
        old = sana / ".git/sol-h3-studio-applied.patch"
        old.write_text(subprocess.run(git + ["diff"], capture_output=True, text=True, check=True).stdout)
        (sana / ".git/sol-h3-studio-applied.patch.base").write_text(head + "\n")
        code, text = self.run_script("10-sana.sh", LFSTUDIO_SANA_COMMIT=head)
        self.assertEqual(code, 0, text)
        self.assertIn(f"the checkout carries an earlier version of the patch ({old}); replacing it", text)
        self.assertIn(f"+ git -C {sana} apply -R {old}\n+ git -C {sana} apply {REPO}/patches/lf-studio.patch", text)
        self.assertIn(f"+ cp {REPO}/patches/lf-studio.patch {sana}/.git/lf-studio-applied.patch", text)
        # until 0.6 the copy is also kept under the 0.4 name, so 0.4's install/10-sana.sh can go back
        self.assertIn(f"+ cp {REPO}/patches/lf-studio.patch {old}\n", text)
        self.assertIn(f"+ git -C {sana} rev-parse HEAD > {old}.base", text)
        self.assertNotIn("warn", text.lower())

    def test_sana_patch_in_history(self):
        """A checkout that carries an earlier version of the patch and no copy of it (patched by hand, docs/PATCH.md):
        10-sana.sh finds that version in this repo's history, under the patch's name and, until 0.6, under its 0.4
        name patches/sol-h3-studio.patch. The repo here is a copy of install/ and patches/BASE_COMMIT* with a history
        of three made-up patch versions: 0.4's, renamed to lf-studio.patch in the next commit, then changed again."""
        repo = self.d / "repo"
        shutil.copytree(REPO / "install", repo / "install", ignore=shutil.ignore_patterns("__pycache__"))
        (repo / "patches").mkdir()
        for name in ("BASE_COMMIT", "BASE_COMMIT_RTX5090"):
            shutil.copy(REPO / "patches" / name, repo / "patches" / name)
        sana = self.d / "home/sol-h3/Sana"
        config = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]

        def git(where, *args):
            return subprocess.run(["git", "-C", str(where), *config, *args], capture_output=True, text=True,
                                  check=True).stdout
        for where in (repo, sana):
            subprocess.run(["git", "init", "-q", str(where)], check=True)
        upstream = "print('upstream')\n"
        (sana / "infer.py").write_text(upstream)
        git(sana, "add", "infer.py")
        git(sana, "commit", "-qm", "base")
        head = git(sana, "rev-parse", "HEAD").strip()
        versions = []                                  # (file name, patched infer.py, commit in repo)
        for name, patched in (("sol-h3-studio.patch", "# Modified by sol-h3-studio\nprint('0.4')\n"),
                              ("lf-studio.patch", "# Modified by lf-studio\nprint('0.5 draft')\n"),
                              ("lf-studio.patch", "# Modified by lf-studio\nprint('0.5')\n")):
            (sana / "infer.py").write_text(patched)
            for old in (repo / "patches").glob("*.patch"):
                old.unlink()
            (repo / "patches" / name).write_text(git(sana, "diff"))
            git(repo, "add", "-A")
            git(repo, "commit", "-qm", name)
            versions.append((name, patched, git(repo, "rev-parse", "HEAD").strip()))
        for name, patched, commit in versions[:2]:
            with self.subTest(name=name):
                (sana / "infer.py").write_text(patched)
                code, text = self.run_script("10-sana.sh", repo=repo, LFSTUDIO_SANA_COMMIT=head)
                self.assertEqual(code, 0, text)
                found = re.search(r"the checkout carries an earlier version of the patch "
                                  r"\((/\S+/patch-\d+-(\w+)\.patch)\); replacing it", text)
                self.assertIsNotNone(found, text)
                self.assertEqual(found.group(2), commit[:7])
                self.assertIn(f"+ git -C {sana} apply -R {found.group(1)}\n"
                              f"+ git -C {sana} apply {repo}/patches/lf-studio.patch", text)
                self.assertIn(f"+ cp {repo}/patches/lf-studio.patch {sana}/.git/lf-studio-applied.patch", text)
                self.assertNotIn("warn", text.lower())

    def test_machine_mismatch_stops_the_steps(self):
        """A backend that doesn't match the machine stops every step but 00-check.sh (a dry run only warns)."""
        code, text = self.run_script("10-sana.sh", FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10",
                                     LFSTUDIO_BACKEND="rtx5090")
        self.assertIn("WARNING: this machine looks like LFSTUDIO_BACKEND=spark, but the install uses rtx5090", text)
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"),
                        LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_HOME=str(self.d / "home" / "sol-h3"),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10", LFSTUDIO_BACKEND="rtx5090")
        run = subprocess.run(["bash", str(REPO / "install" / "20-weights.sh")], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("ERROR: this machine looks like LFSTUDIO_BACKEND=spark, but the install uses rtx5090", run.stderr)
        self.assertNotIn("===", run.stdout)                                    # stopped before its first step

    def test_sana_refuses_a_checkout_of_the_other_backend(self):
        """10-sana.sh with LFSTUDIO_BACKEND=rtx5090 on a checkout carrying a Spark patch changes nothing."""
        sana = self.d / "sana"
        sana.mkdir()
        git = ["git", "-C", str(sana), "-c", "user.name=t", "-c", "user.email=t@t"]
        pkg = sana / "models/minimax_h3/Sol-H3-Spark/runtime"
        pkg.mkdir(parents=True)
        (pkg / "a.py").write_text("a = 1\n")
        subprocess.run(["git", "init", "-q", str(sana)], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "base"], check=True)
        (pkg / "a.py").write_text("a = 2\n")                                  # the Spark's patch, applied
        spark_patch = subprocess.run([*git, "diff"], capture_output=True, text=True, check=True).stdout
        gitdir = sana / ".git"
        (gitdir / "lf-studio-applied.patch").write_text(spark_patch)
        before = (pkg / "a.py").read_text()
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"),
                        LFSTUDIO_ENV=str(self.d / "no-env-file"), LFSTUDIO_HOME=str(self.d / "home" / "sol-h3"),
                        LFSTUDIO_SANA=str(sana),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090", LFSTUDIO_BACKEND="rtx5090")
        run = subprocess.run(["bash", str(REPO / "install" / "10-sana.sh")], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("carries the patch for Sol-H3-Spark (LFSTUDIO_BACKEND=spark), not for Sol-H3-RTX5090; "
                      "nothing was changed", run.stderr)
        self.assertEqual((pkg / "a.py").read_text(), before)

    def test_invalid_backend(self):
        for name in INSTALL:
            code, text = self.run_script(name, LFSTUDIO_BACKEND="gb10")
            self.assertEqual(code, 2, f"{name}: {text}")
            self.assertIn("LFSTUDIO_BACKEND=gb10: must be spark or rtx5090", text)



# uv for install/65-regen2k.sh run for real: it reports 0.13.4, installs nothing (pip-fails next to it: pip install
# fails), and its venv's python only logs its calls and exits FAKE_PY_RC (default 1: the import check fails)
FAKE_UV = r"""#!/bin/bash
d=$(dirname "$0")
echo "$*" >> "$d/uv.log"
case "$1" in
    --version) echo "uv 0.13.4 (aarch64-unknown-linux-gnu)" ;;
    python) [[ "$2" == install ]] ;;
    venv) v="${@: -1}"; mkdir -p "$v/bin"
          printf '#!/bin/bash\necho "$*" >> "$(dirname "$0")/python.log"\nexit ${FAKE_PY_RC:-1}\n' > "$v/bin/python"
          chmod 755 "$v/bin/python" ;;
    pip) [[ ! -f "$d/pip-fails" ]] ;;
    *) exit 3 ;;
esac
"""
SANA_REV = "6e4a44002149418a547434b91d7280a8bbdcace3"


class InstallRegen2k(unittest.TestCase):
    """install/65-regen2k.sh, the experimental Upscale's optional step (DGX Spark only): its --dry-run, and runs for
    real with a fake uv, where every failed step stops it and the venv is marked done only after its import check
    (#74). Everything in a temp directory; the Sana clone is replaced by an extract that is already there."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-regen2k-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        self.home = self.d / "home" / "sol-h3"
        (self.d / "tmp").mkdir()
        self.home.mkdir(parents=True)

    def run_65(self, *args, **settings):
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                        LFSTUDIO_HOME=str(self.home), PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10")
        env.update(settings)
        run = subprocess.run(["bash", str(REPO / "install" / "65-regen2k.sh"), *args], env=env, capture_output=True,
                             text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    def test_dry_run(self):
        code, text = self.run_65("--dry-run")
        self.assertEqual(code, 0, text)
        h, lock = str(self.home), str(REPO / "install" / "freeze" / "regen2k-lock.txt")
        uv = f"{h}/bin/uv"
        for line in (
                "+ curl -fL --retry 3 -o "
                f"{h}/cache/downloads/uv-0.13.4/uv-aarch64-unknown-linux-gnu.tar.gz.part "
                "https://github.com/astral-sh/uv/releases/download/0.13.4/uv-aarch64-unknown-linux-gnu.tar.gz",
                "1ae72b4718729e2ccc4f1ae2c6d1e531ca8467800e0546e9130b3886023a60ab",
                f"+ {uv} python install --no-bin 3.11.17",
                f"+ git -C {h}/Sana fetch --filter=blob:none origin {SANA_REV}",
                f"+ git -C {h}/Sana archive {SANA_REV} models/Open-H3-Regenerate-2K techniques/sparse_backends | tar",
                f"+ ln -sfn {h}/sources/sana-6e4a440/models/Open-H3-Regenerate-2K {h}/Open-H3-Regenerate-2K",
                f"+ {uv} venv --python 3.11.17 {h}/envs/regen2k\n",
                f"+ {uv} pip install --python {h}/envs/regen2k/bin/python --no-deps --require-hashes --only-binary "
                ":all: --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 "
                f"--index-strategy unsafe-best-match -r {lock}\n",
                f"+ {uv} pip install --python {h}/envs/regen2k/bin/python --no-deps --no-build-isolation "
                f"{h}/sources/sana-6e4a440/techniques/sparse_backends\n",
                f"+ {h}/envs/regen2k/bin/python -I -B {REPO}/install/regen2k_weights.py --weights "
                f"{h}/checkpoints/Open-H3-Regenerate-2K --upscaler-copy "
                f"{h}/checkpoints/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors\n"):
            self.assertIn(line, text)
        order = [text.index(x) for x in ("venv --python", "--require-hashes", "sparse_backends\n", "echo done",
                                         "regen2k_weights.py")]
        self.assertEqual(order, sorted(order), "the venv is marked done after its import check, before the weights")
        self.assertNotIn("pip install --python", text.split("venv --python")[0])
        self.assertNotIn(" -m pip", text)                   # uv's venv has no pip, and none is used
        self.assertFalse(any(self.home.iterdir()), "a dry run writes nothing")

    def test_dry_run_settings(self):
        """LFSTUDIO_UPSCALE_PKG and _WEIGHTS go where they say; another LFSTUDIO_UPSCALE_PYTHON is refused, since this
        step builds its own venv only; x86_64 is refused unless dry."""
        code, text = self.run_65("--dry-run", LFSTUDIO_UPSCALE_PKG=str(self.d / "pkg"),
                                 LFSTUDIO_UPSCALE_WEIGHTS=str(self.d / "w"))
        self.assertEqual(code, 0, text)
        self.assertIn(f"ln -sfn {self.home}/sources/sana-6e4a440/models/Open-H3-Regenerate-2K {self.d}/pkg\n", text)
        self.assertIn(f"--weights {self.d}/w ", text)
        code, text = self.run_65("--dry-run", LFSTUDIO_UPSCALE_PYTHON="/usr/bin/python3")
        self.assertEqual(code, 1, text)
        self.assertIn("LFSTUDIO_UPSCALE_PYTHON=/usr/bin/python3: this script builds", text)
        code, text = self.run_65(FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GH200")
        self.assertEqual(code, 1, text)
        self.assertIn("uname -m is x86_64", text)

    def fake_uv(self, pip_fails=False):
        uv = self.home / "bin" / "uv"
        uv.parent.mkdir(parents=True, exist_ok=True)
        uv.write_text(FAKE_UV)
        uv.chmod(0o755)
        if pip_fails:
            (uv.parent / "pip-fails").touch()
        src = self.home / "sources" / "sana-6e4a440"                # the extract, as if made before
        (src / "models" / "Open-H3-Regenerate-2K").mkdir(parents=True)
        (src / ".lf-studio-commit").write_text(SANA_REV + "\n")
        return uv.parent

    def test_a_failed_step_stops_it(self):
        """#74: a failed install stops the script and leaves the venv unmarked; the next run goes on in it; the
        import check gates the done mark; FORCE=1 moves the venv aside and builds a new one."""
        bin_dir = self.fake_uv(pip_fails=True)
        venv, lock = self.home / "envs" / "regen2k", REPO / "install" / "freeze" / "regen2k-lock.txt"
        mark = venv / "lf-studio-install.txt"
        sha = subprocess.run(["sha256sum", str(lock)], capture_output=True, text=True, check=True).stdout.split()[0]
        code, text = self.run_65()
        self.assertEqual(code, 1, text)
        self.assertIn("ERROR: the packages of install/freeze/regen2k-lock.txt did not install", text)
        self.assertNotIn("=== 65-regen2k.sh: weights", text)
        self.assertEqual(mark.read_text(), f"building {sha}\n")
        self.assertEqual((self.home / "Open-H3-Regenerate-2K").resolve(),
                         self.home / "sources" / "sana-6e4a440" / "models" / "Open-H3-Regenerate-2K")
        (bin_dir / "pip-fails").unlink()
        code, text = self.run_65()                                          # the import check fails (FAKE_PY_RC)
        self.assertEqual(code, 1, text)
        self.assertIn("an earlier run stopped while installing into it: going on", text)
        self.assertIn("ERROR: the import check of", text)
        self.assertEqual(mark.read_text(), f"building {sha}\n")
        self.assertEqual(sum(line.startswith("venv ") for line in (bin_dir / "uv.log").read_text().splitlines()), 1)
        code, text = self.run_65(FAKE_PY_RC="0")
        self.assertEqual(code, 0, text)
        self.assertEqual(mark.read_text(), f"done {sha}\n")
        calls = (venv / "bin" / "python.log").read_text()
        self.assertIn(f"-I -B {REPO}/install/regen2k_weights.py --weights {self.home}/checkpoints/Open-H3-Regenerate-2K",
                      calls)
        self.assertIn(f"{REPO}/upscale/regen2k_job.py {self.home}/Open-H3-Regenerate-2K", calls)   # the job's check
        code, text = self.run_65(FAKE_PY_RC="0")
        self.assertEqual(code, 0, text)
        self.assertIn("already there (FORCE=1 builds it again)", text)
        code, text = self.run_65(FAKE_PY_RC="0", FORCE="1")
        self.assertEqual(code, 0, text)
        moved = [p.name for p in venv.parent.iterdir() if p.name.startswith("regen2k.")]
        self.assertEqual(len(moved), 1, moved)
        self.assertRegex(moved[0], r"^regen2k\.\d{8}-\d{6}$")
        self.assertEqual(mark.read_text(), f"done {sha}\n")

    def test_a_package_link_of_someone_else_is_not_touched(self):
        """LFSTUDIO_UPSCALE_PKG: a link this step made to an earlier extract is replaced; a directory or a link of
        someone else's stops it."""
        self.fake_uv()
        pkg, ours = self.home / "Open-H3-Regenerate-2K", self.home / "sources" / "sana-6e4a440"
        pkg.symlink_to(self.home / "sources" / "sana-0000000" / "models" / "Open-H3-Regenerate-2K")
        code, text = self.run_65(FAKE_PY_RC="0")
        self.assertEqual(code, 0, text)
        self.assertEqual(os.readlink(pkg), str(ours / "models" / "Open-H3-Regenerate-2K"))
        pkg.unlink()
        pkg.symlink_to(self.d / "my-own-copy")
        code, text = self.run_65(FAKE_PY_RC="0")
        self.assertEqual(code, 1, text)
        self.assertIn("exists and is not a link this script made", text)
        self.assertEqual(os.readlink(pkg), str(self.d / "my-own-copy"))

    def test_a_venv_of_someone_else_is_not_touched(self):
        self.fake_uv()
        other = self.home / "envs" / "regen2k" / "bin"
        other.mkdir(parents=True)
        (other / "keep").write_text("mine")
        code, text = self.run_65(FAKE_PY_RC="0")
        self.assertEqual(code, 1, text)
        self.assertIn("exists but this script did not make it", text)
        self.assertEqual(sorted(p.name for p in other.iterdir()), ["keep"])
        self.assertNotIn("pip", (self.home / "bin" / "uv.log").read_text())


class Regen2kWeights(unittest.TestCase):
    """install/regen2k_weights.py: the release's files are kept, linked or downloaded and checked (with a stand-in
    huggingface_hub), and Sharp's AdaLN table is read by HTTP range from a fake Hugging Face (HF_ENDPOINT)."""

    def setUp(self):
        import hashlib
        import importlib.util
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-weights-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        spec = importlib.util.spec_from_file_location("regen2k_weights", REPO / "install" / "regen2k_weights.py")
        self.w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.w)
        self.w.say = lambda text: None                                     # its progress lines
        self.data = {name: f"bytes of {name}".encode() for name in self.w.FILES}
        self.w.FILES = {name: hashlib.sha256(b).hexdigest() for name, b in self.data.items()}
        self.cache, self.downloads = {}, []
        fake = type(sys)("huggingface_hub")

        def hf_hub_download(repo, name, revision=None, local_files_only=False, local_dir=None):
            self.assertEqual((repo, revision), (self.w.RELEASE["repo"], self.w.RELEASE["revision"]))
            if local_files_only:
                if name not in self.cache:
                    raise FileNotFoundError(name)
                return str(self.cache[name])
            self.downloads.append(name)
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.data[name] if name != "lora/sharp_curve_branches.safetensors" or not self.corrupt
                             else b"corrupt")
            return str(path)
        fake.hf_hub_download = hf_hub_download
        self.corrupt = False
        sys.modules["huggingface_hub"] = fake
        self.addCleanup(sys.modules.pop, "huggingface_hub", None)

    def test_kept_linked_downloaded(self):
        weights = self.d / "weights"
        lora = "lora/open_h3_regenerate_2k_lora.safetensors"
        cached = self.d / "hub" / "snapshot" / lora                        # in the Hugging Face cache already
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.data[lora])
        self.cache[lora] = cached
        copy = self.d / "H3-upscaler.safetensors"                          # the studio's own upscaler, same bytes
        copy.write_bytes(self.data[self.w.UPSCALER])
        for name, digest in self.w.FILES.items():
            self.w.release_file(weights, name, digest, str(copy))
        self.assertEqual(os.readlink(weights / lora), str(cached))
        self.assertEqual(os.readlink(weights / self.w.UPSCALER), str(copy))
        self.assertEqual(self.downloads, ["lora/sharp_curve_branches.safetensors"])
        self.assertFalse((weights / "lora/sharp_curve_branches.safetensors").is_symlink())
        before = sorted(p.stat().st_mtime_ns for p in self.d.rglob("*") if p.is_file())
        for name, digest in self.w.FILES.items():                         # again: everything is kept
            self.w.release_file(weights, name, digest, str(copy))
        self.assertEqual(len(self.downloads), 1)
        self.assertEqual(sorted(p.stat().st_mtime_ns for p in self.d.rglob("*") if p.is_file()), before)
        self.assertEqual(cached.read_bytes(), self.data[lora])            # the cache was only read

    def test_a_bad_download_is_set_aside(self):
        self.corrupt = True
        weights, name = self.d / "weights", "lora/sharp_curve_branches.safetensors"
        with self.assertRaises(self.w.Failed) as e:
            self.w.release_file(weights, name, self.w.FILES[name])
        self.assertIn("does not have NVIDIA's SHA-256", str(e.exception))
        self.assertEqual(sorted(p.name for p in (weights / "lora").iterdir()), ["sharp_curve_branches.safetensors.bad"])

    def serve(self, body, ranges=True, redirect=None, auth=None):
        """A fake Hugging Face that serves `body` for the pinned Comfy-Org file, by Range unless `ranges` is false, or
        redirects it to `redirect` (another server's address). `auth`: a list for the Authorization headers seen."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        seen = []
        path = f"/{self.w.CURVE['repo']}/resolve/{self.w.CURVE['revision']}/{self.w.CURVE['file']}"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Range"))
                if auth is not None:
                    auth.append(self.headers.get("Authorization"))
                if redirect:                                # as Hugging Face sends a file to its CDN
                    self.send_response(302)
                    self.send_header("Location", redirect + self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path != path:
                    self.send_response(404)
                    self.end_headers()
                    return
                m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
                data = body[int(m.group(1)):int(m.group(2)) + 1] if m and ranges else body
                self.send_response(206 if m and ranges else 200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}", seen

    def checkpoint(self, table, dtype="F32", shape=(1025, 8)):
        """A safetensors file whose second tensor is `table` (bytes)."""
        import struct
        header = json.dumps({"other": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                             "adaln_t_table": {"dtype": dtype, "shape": list(shape),
                                               "data_offsets": [16, 16 + len(table)]}}).encode()
        return struct.pack("<Q", len(header)) + header + b"\x00" * 16 + table

    def test_table_by_range(self):
        import hashlib
        table = bytes(range(256)) * 128 + b"\x01" * 32                     # 32,800 bytes, like the real one
        url, seen = self.serve(self.checkpoint(table) + b"\xff" * 4096)     # (the rest of the checkpoint)
        dest = self.d / "weights" / "lora" / "adaln_t_table.bin"
        with self.assertRaises(self.w.Failed) as e:                         # not NVIDIA's bytes: nothing written
            self.w.fetch_table(dest, url)
        self.assertIn("does not have the SHA-256", str(e.exception))
        self.assertFalse(dest.parent.exists() and any(dest.parent.iterdir()))
        self.w.CURVE = dict(self.w.CURVE, sha256=hashlib.sha256(table).hexdigest())
        self.w.fetch_table(dest, url)
        self.assertEqual(dest.read_bytes(), table)
        self.assertTrue(all(r and r.startswith("bytes=") for r in seen), seen)         # never the whole file
        url, _ = self.serve(self.checkpoint(table), ranges=False)
        with self.assertRaises(self.w.Failed) as e:
            self.w.fetch_table(self.d / "other.bin", url)
        self.assertIn("ignored the Range request", str(e.exception))
        url, _ = self.serve(self.checkpoint(table[:32768], shape=(1024, 8)))
        with self.assertRaises(self.w.Failed) as e:
            self.w.fetch_table(self.d / "other.bin", url)
        self.assertIn("is F32 [1024, 8], not F32 [1025, 8]", str(e.exception))
        self.assertFalse((self.d / "other.bin").exists())

    def test_no_token_goes_along(self):
        """HF_TOKEN in the environment never reaches the table's host, nor the host it redirects to (urllib passes
        an Authorization header on through a redirect, to any host)."""
        import hashlib
        table = bytes(range(256)) * 128 + b"\x02" * 32
        self.w.CURVE = dict(self.w.CURVE, sha256=hashlib.sha256(table).hexdigest())
        cdn_auth, hub_auth = [], []
        cdn, cdn_seen = self.serve(self.checkpoint(table), auth=cdn_auth)
        hub, _ = self.serve(b"", redirect=cdn.replace("127.0.0.1", "localhost"), auth=hub_auth)   # another host name
        weights = self.d / "weights"
        for name, data in self.data.items():                                 # the release files are in place
            (weights / name).parent.mkdir(parents=True, exist_ok=True)
            (weights / name).write_bytes(data)
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret", "HF_ENDPOINT": hub}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.w.main(["--weights", str(weights)]), 0)
        self.assertEqual((weights / "lora" / "adaln_t_table.bin").read_bytes(), table)
        self.assertEqual(len(cdn_seen), 3)                                  # header length, header, the tensor
        self.assertEqual((hub_auth, cdn_auth), ([None] * 3, [None] * 3))


# A venv's python for pillow_update: `pip show -q`, `pip uninstall -y` and `pip install` act on the package names in
# installed.txt and the Pillow version in pillow.txt, next to it; every call is logged in calls.log.
FAKE_VENV_PYTHON = r"""#!/bin/bash
d=$(dirname "$0")
echo "$*" >> "$d/calls.log"
case "$*" in
    "-m pip show -q "*) grep -qx "$5" "$d/installed.txt" ;;
    "-m pip uninstall -y "*) shift 4; for p in "$@"; do grep -vx "$p" "$d/installed.txt" > "$d/x"; mv "$d/x" "$d/installed.txt"; done ;;
    "-m pip install --only-binary pillow pillow==13.0.0") [[ -f "$d/install-fails" ]] && exit 1; echo 13.0.0 > "$d/pillow.txt" ;;
    "-m pip install --no-deps pip==26.2.2"|"-m pip install --no-deps certifi>=2026.10.19") [[ ! -f "$d/pip-fails" ]] ;;
    *"-c "*print*) cat "$d/pillow.txt" ;;
    *"-c "*) v=$(cat "$d/pillow.txt"); (( ${v%%.*} >= 13 )) ;;
    *) exit 3 ;;
esac
"""

# docker for 60-qwen-image.sh: the images named in images.txt exist, and the layer image's label is label.txt.
FAKE_DOCKER = r"""#!/bin/bash
d=$(dirname "$0")
echo "$*" >> "$d/docker.log"
case "$1 $2" in
    "info "*|"build "*|"run "*) exit 0 ;;
    "image inspect") if [[ "$3" == -f ]]; then cat "$d/label.txt"; else grep -qx "$3" "$d/images.txt"; fi ;;
    *) exit 3 ;;
esac
"""


class PillowUpdate(unittest.TestCase):
    """#69, for real (not --dry-run) against fakes: pillow_update removes the packages that cap Pillow and installs
    Pillow 13.0.0 only when needed, and 60-qwen-image.sh rebuilds the LF Studio layer when qwen-image/ changed."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-pillow-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        (self.d / "home").mkdir()
        self.env = clean_env(HOME=str(self.d / "home"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                             LFSTUDIO_HOME=str(self.d / "home" / "sol-h3"), FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10",
                             PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}")
        self.bin = self.d / "venv" / "bin"
        self.bin.mkdir(parents=True)
        (self.bin / "python").write_text(FAKE_VENV_PYTHON)
        (self.bin / "python").chmod(0o755)

    def update(self, pillow, installed, *remove):
        (self.bin / "pillow.txt").write_text(pillow + "\n")
        (self.bin / "installed.txt").write_text("".join(p + "\n" for p in installed))
        return self.pillow_update(*remove)

    def pillow_update(self, *remove):
        (self.bin / "calls.log").write_text("")
        run = subprocess.run(["bash", "-c", 'r=$1 v=$2; shift 2; set -- "$@"; args=("$@"); set --; '
                              'source "$r/install/common.sh"; pillow_update "$v" "${args[@]}"',
                              "_", str(REPO), str(self.d / "venv"), *remove],
                             env=self.env, capture_output=True, text=True, timeout=60)
        calls = [c for c in (self.bin / "calls.log").read_text().splitlines()            # what changes the venv
                 if c.startswith("-m pip") and not c.startswith("-m pip show")]
        return run.returncode, run.stdout + run.stderr, calls

    def test_old_venv_is_updated(self):
        code, out, calls = self.update("11.3.0", ["gradio", "gradio-client", "numpy"], "gradio", "gradio-client", "moviepy")
        self.assertEqual(code, 0, out)
        self.assertEqual(calls[:2], ["-m pip uninstall -y gradio gradio-client",
                                     "-m pip install --only-binary pillow pillow==13.0.0"], out)
        self.assertEqual((self.bin / "installed.txt").read_text(), "numpy\n")
        self.assertEqual((self.bin / "pillow.txt").read_text(), "13.0.0\n")
        code, out, calls = self.pillow_update("gradio", "gradio-client", "moviepy")       # a second run: nothing to do
        self.assertEqual(code, 0, out)
        self.assertNotIn("-m pip uninstall", " ".join(calls))
        self.assertNotIn("-m pip install", " ".join(calls))
        self.assertIn(">>> Pillow 13.0.0 already", out)

    def test_newer_pillow_is_kept(self):
        code, out, calls = self.update("13.1.0", [])
        self.assertEqual(code, 0, out)
        self.assertNotIn("-m pip install", " ".join(calls))
        self.assertIn(">>> Pillow 13.1.0 already", out)

    def test_failed_install_stops(self):
        (self.bin / "install-fails").write_text("")
        code, out, calls = self.update("12.0.0", [])
        self.assertNotEqual(code, 0)
        self.assertIn("ERROR: Pillow 13.0.0 did not install into", out)

    def test_pillow_only_never_builds(self):
        """--pillow-only runs only the Pillow step on the venv that is there, also one without the done marker (one
        built before the markers would otherwise be built into again: pip, the install lists, the kernels)."""
        for stage, script, removed in (("stage1", "40-stage1.sh", "gradio gradio-client"), ("stage2", "50-stage2.sh", "")):
            venv = self.d / "home/sol-h3/envs" / stage
            (venv / "bin").mkdir(parents=True)
            (venv / "bin/python").write_text(FAKE_VENV_PYTHON)
            (venv / "bin/python").chmod(0o755)
            (venv / "bin/pillow.txt").write_text("11.3.0\n")
            (venv / "bin/installed.txt").write_text("gradio\ngradio-client\nnumpy\n")
            run = subprocess.run(["bash", str(REPO / "install" / script), "--pillow-only", "--dry-run"], env=self.env,
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn(f"+ {venv}/bin/python -m pip install --only-binary pillow pillow==13.0.0", run.stdout)
            self.assertNotIn("install -U pip", run.stdout)
            run = subprocess.run(["bash", str(REPO / "install" / script), "--pillow-only"], env=self.env,
                                 capture_output=True, text=True, timeout=60)
            out = run.stdout + run.stderr
            self.assertEqual(run.returncode, 0, out)
            self.assertIn(">>> done (--pillow-only)", out)
            calls = [c for c in (venv / "bin/calls.log").read_text().splitlines()
                     if c.startswith("-m pip") and not c.startswith("-m pip show")]
            self.assertEqual(calls, ["-m pip install --no-deps pip==26.2.2", "-m pip install --no-deps certifi>=2026.10.19"]
                             + ([f"-m pip uninstall -y {removed}"] if removed else [])
                             + ["-m pip install --only-binary pillow pillow==13.0.0"], out)
            self.assertEqual((venv / "bin/pillow.txt").read_text(), "13.0.0\n")
            self.assertFalse((venv / ".lf-studio-done").exists())
            self.assertNotIn("+ touch", out)

    def test_failed_pip_update_stops(self):
        """--pillow-only stops at a failed pip or certifi install, before Pillow."""
        venv = self.d / "home/sol-h3/envs/stage2"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").write_text(FAKE_VENV_PYTHON)
        (venv / "bin/python").chmod(0o755)
        (venv / "bin/pillow.txt").write_text("12.3.0\n")
        (venv / "bin/installed.txt").write_text("")
        (venv / "bin/pip-fails").write_text("")
        run = subprocess.run(["bash", str(REPO / "install/50-stage2.sh"), "--pillow-only"], env=self.env,
                             capture_output=True, text=True, timeout=60)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("ERROR: pip 26.2.2 did not install into", run.stderr)
        self.assertNotIn("pillow==13.0.0", (venv / "bin/calls.log").read_text())

    def test_pillow_only_needs_a_venv(self):
        for script, stage in (("40-stage1.sh", "Stage1"), ("50-stage2.sh", "Stage2")):
            run = subprocess.run(["bash", str(REPO / "install" / script), "--pillow-only"], env=self.env,
                                 capture_output=True, text=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(f"(--pillow-only needs an existing {stage} venv)", run.stderr)

    def test_qwen_layer_rebuilt_when_changed(self):
        (self.fake / "docker").write_text(FAKE_DOCKER)
        (self.fake / "sg").write_text('#!/bin/bash\nexec bash -c "$3"\n')          # sg docker -c CMD
        pkg = self.d / "home/sol-h3/Sana/models/minimax_h3/Sol-H3-Spark"
        pkg.mkdir(parents=True)
        (pkg / "Dockerfile.qwen").write_text("FROM scratch\n")
        (self.fake / "images.txt").write_text("sol-h3-spark-qwen:base\nsol-h3-spark-qwen\n")

        def run60(label):
            (self.fake / "label.txt").write_text(label + "\n")
            (self.fake / "docker.log").write_text("")
            run = subprocess.run(["bash", str(REPO / "install/60-qwen-image.sh")], env=self.env, capture_output=True,
                                 text=True, timeout=60)
            builds = [c for c in (self.fake / "docker.log").read_text().splitlines() if c.startswith("build ")]
            return run.returncode, run.stdout + run.stderr, builds

        code, out, builds = run60("")                        # an image from before 0.5: no label
        self.assertEqual(code, 0, out)
        self.assertIn("qwen-image/ changed since this image was built: rebuilding the layer", out)
        self.assertEqual(len(builds), 1, builds)               # the layer only; the base is kept
        layer = re.search(r"--label lf-studio\.layer=([0-9a-f]{16}) -t sol-h3-spark-qwen ", builds[0])
        self.assertTrue(layer, builds)
        code, out, builds = run60(layer.group(1))             # built with this qwen-image/: kept
        self.assertEqual(code, 0, out)
        self.assertEqual(builds, [])
        self.assertIn("already built", out)
        code, out, builds = run60("0123456789abcdef")         # built with another qwen-image/: rebuilt
        self.assertEqual(len(builds), 1, out)


if __name__ == "__main__":
    unittest.main()
