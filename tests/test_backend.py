"""SOLH3_BACKEND (docs/RTX5090.md): the launcher, the server, the page and the install scripts with no backend set
(the DGX Spark, as before), with rtx5090, and with a wrong value.

CPU only, against the stub runtime. nvidia-smi, uname, gcc, docker and sg are small fakes on PATH, so the GPU, the
CPU type and the compiler of the machine running the tests do not matter. The install scripts run with --dry-run
only, with their home, settings file and temp files in a temp directory.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "stub"))
from studio import StubStudio, free_port  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUB = Path(__file__).resolve().parent / "stub"
LAUNCHER = REPO / "bin" / "sol-h3-studio"
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
           if not k.startswith(("SOL_H3_", "SOLH3_", "FAKE_", "SPARK_", "PIP_", "UV_", "FORCE"))}
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
        self.d = Path(tempfile.mkdtemp(prefix="sol-h3-backend-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        self.sana = self.d / "sana"
        for task in ("t2va", "fl2va", "ref2va"):
            (self.d / f"paths-{task}.json").write_text("{}")

    def launch(self, *args, **settings):
        env = clean_env(SOLH3_ENV=str(self.d / "no-env-file"), SOLH3_SANA=str(self.sana), SOLH3_PYTHON=sys.executable,
                        SOLH3_PATHS=str(self.d / "paths-{task}.json"), SOL_H3_SPARK_RUNTIME_ROOT=str(self.d / "runtime"),
                        SOLH3_STATE_DIR=str(self.d / "state"), SOLH3_HOST="127.0.0.1", SOLH3_PORT=str(self.port),
                        SOLH3_SERVER=str(self.d / "no-server.py") if "--kill" in args else str(REPO / "webui/server.py"),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}")
        env.update(settings)
        run = subprocess.run(["bash", str(LAUNCHER), *(args or ("--check",))], env=env, capture_output=True, text=True,
                             timeout=120)
        return run.returncode, run.stdout + run.stderr

    port = None

    def check(self, **settings):
        self.port = self.port or free_port()
        return self.launch("--check", **settings)


class LauncherSpark(LauncherTest):
    """Without SOLH3_BACKEND the launcher does what it did before: the Sol-H3-Spark package, the Docker checks and the
    110 GiB floor; SOLH3_BACKEND=spark is the same."""

    def setUp(self):
        super().setUp()
        self.package = spark_package(self.sana)

    def test_unset_is_the_spark(self):
        code, out = self.check(SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 0, out)
        self.assertIn(f"checking (package {self.package}, port", out)
        self.assertIn("checks passed", out)
        self.assertNotIn("GPU memory check", out)                      # no VRAM check on the Spark
        status = lambda text: [line for line in text.splitlines() if line.startswith(">>> STATUS:")]  # noqa: E731
        again = self.check(SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0", SOLH3_BACKEND="spark")
        self.assertEqual((again[0], status(again[1])), (code, status(out)))     # log lines carry times and MemFree

    def test_unset_needs_docker(self):
        code, out = self.check(SOLH3_MIN_FREE_GIB="0")                  # docker and sg fail (fakes)
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - cannot talk to docker", out)

    def test_unset_floor_is_110(self):
        avail = mem_available_gib()
        code, out = self.check(SOLH3_TEST_SKIP_DOCKER="1")
        if avail < 108:
            self.assertEqual(code, 1, out)
            self.assertIn("REFUSED - Sol-H3 Studio needs ~110 GiB free memory", out)
        elif avail > 112:
            self.assertEqual(code, 0, out)
            self.assertIn("(need 110)", out)

    def test_vram_setting_ignored(self):
        """SOLH3_MIN_FREE_VRAM_GIB is an RTX 5090 setting: the Spark's GPU memory is its RAM."""
        code, out = self.check(SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0", SOLH3_MIN_FREE_VRAM_GIB="999",
                               FAKE_VRAM_FREE_MIB="10")
        self.assertEqual(code, 0, out)
        self.assertIn("note: SOLH3_MIN_FREE_VRAM_GIB is set, but only the RTX 5090 reads it", out)

    def test_rtx_package_refused(self):
        """A Sol-H3-Spark directory with runtime/offload.py is not the Spark's package."""
        sana = self.d / "sana-odd"
        package = sana / "models" / "minimax_h3" / "Sol-H3-Spark"
        shutil.copytree(STUB / "runtime", package / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        (package / "runtime" / "offload.py").write_text("")
        code, out = self.check(SOLH3_SANA=str(sana), SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("which only Sol-H3-RTX5090 has; this is not the Sol-H3-Spark package", out)


class LauncherRtx5090(LauncherTest):
    """SOLH3_BACKEND=rtx5090: the Sol-H3-RTX5090 package, no Docker, its RAM floor and a free GPU memory floor."""

    def setUp(self):
        super().setUp()
        self.package = rtx_package(self.sana)

    def test_passes_without_docker(self):
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="30000")
        self.assertEqual(code, 0, out)
        self.assertIn(f"checking (package {self.package}, port", out)
        self.assertIn("GPU memory check OK: 29 GiB free (need 25)", out)
        self.assertNotIn("docker", out.lower())

    def test_ram_floor(self):
        avail = mem_available_gib()
        code, out = self.check(SOLH3_BACKEND="rtx5090", FAKE_VRAM_FREE_MIB="30000")
        if avail < 42:
            self.assertEqual(code, 1, out)
            self.assertIn("REFUSED - Sol-H3 Studio needs ~42 GiB free memory", out)
        else:
            self.assertEqual(code, 0, out)
            self.assertIn("(need 42)", out)
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="100000")
        self.assertEqual(code, 1, out)
        self.assertIn("needs ~100000 GiB free memory", out)
        self.assertIn("note: SOLH3_MIN_FREE_GIB=100000 is more than this machine's", out)
        self.assertIn("the default here is 42", out)

    def test_vram_floor(self):
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="10000")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - Sol-H3 Studio needs ~25 GiB free GPU memory, only 9 GiB free", out)
        self.assertIn("SOLH3_MIN_FREE_VRAM_GIB=0", out)
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", SOLH3_MIN_FREE_VRAM_GIB="8",
                               FAKE_VRAM_FREE_MIB="10000")
        self.assertEqual(code, 0, out)
        self.assertIn("GPU memory check OK: 9 GiB free (need 8)", out)
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", FAKE_VRAM_FREE_MIB="[N/A]")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - cannot read the free GPU memory from nvidia-smi", out)
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", SOLH3_MIN_FREE_VRAM_GIB="0",
                               FAKE_VRAM_FREE_MIB="[N/A]")
        self.assertEqual(code, 0, out)
        self.assertNotIn("GPU memory check", out)

    def test_patch_settings_checked(self):
        """The patch's settings checks run in this package too."""
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0", SOL_H3_PROMPT_TOKENS="special")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - SOL_H3_PROMPT_TOKENS must be one of", out)


class LauncherBackendRefusals(LauncherTest):
    def test_invalid_backend(self):
        spark_package(self.sana)
        for args in (("--check",), ("--kill",)):
            self.port = free_port()
            code, out = self.launch(*args, SOLH3_BACKEND="gb10", SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0")
            self.assertEqual(code, 1, (args, out))
            self.assertIn("REFUSED - SOLH3_BACKEND=gb10: must be spark or rtx5090", out)

    def test_mismatched_package(self):
        """A checkout patched for the other backend is named, with the setting that fits it."""
        spark_package(self.sana)
        code, out = self.check(SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-Spark, not Sol-H3-RTX5090: set SOLH3_BACKEND=spark", out)

        rtx = self.d / "sana-rtx"
        rtx_package(rtx)
        on_a_5090 = dict(FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        code, out = self.check(SOLH3_SANA=str(rtx), SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0", **on_a_5090)
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-RTX5090, not Sol-H3-Spark: set SOLH3_BACKEND=rtx5090 "
                      "(it is spark, the default)", out)
        code, out = self.check(SOLH3_SANA=str(rtx), SOLH3_BACKEND="spark", SOLH3_TEST_SKIP_DOCKER="1",
                               SOLH3_MIN_FREE_GIB="0", **on_a_5090)
        self.assertEqual(code, 1, out)
        self.assertIn("set SOLH3_BACKEND=rtx5090 (it is spark), or point", out)       # set by the user: no "default"
        # On a GB10 the other backend can't run: the checkout is the wrong one, and the answer says so.
        code, out = self.check(SOLH3_SANA=str(rtx), SOLH3_TEST_SKIP_DOCKER="1", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("carries the patch for Sol-H3-RTX5090, but this machine needs Sol-H3-Spark", out)
        self.assertNotIn("set SOLH3_BACKEND=rtx5090", out)

        unpatched = self.d / "sana-plain"
        (unpatched / "models/minimax_h3/Sol-H3-RTX5090/runtime").mkdir(parents=True)
        (unpatched / "models/minimax_h3/Sol-H3-RTX5090/runtime/offload.py").write_text("")
        code, out = self.check(SOLH3_SANA=str(unpatched), SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("is not patched (runtime/geometry.py missing); apply patches/sol-h3-studio-rtx5090.patch", out)

        code, out = self.check(SOLH3_SANA=str(self.d / "nothing"), SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED - no Sol-H3-RTX5090 package at", out)

        no_offload = self.d / "sana-no-offload"
        rtx_package(no_offload, offload=False)
        code, out = self.check(SOLH3_SANA=str(no_offload), SOLH3_BACKEND="rtx5090", SOLH3_MIN_FREE_GIB="0")
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
        code, out = self.launch("--kill", SOLH3_BACKEND="rtx5090")
        self.assertEqual(code, 0, out)
        self.assertTrue(self.stopped(rtx_worker), "the Sol-H3-RTX5090 worker still runs")
        self.assertTrue(self.stopped(by_options), "the worker named by its options file still runs")
        self.assertFalse(self.stopped(spark_worker, timeout=1), "the Sol-H3-Spark worker was stopped too")
        code, out = self.launch("--kill", SOLH3_TEST_SKIP_DOCKER="1")        # no backend: the Spark's workers
        self.assertEqual(code, 0, out)
        self.assertTrue(self.stopped(spark_worker), "the Sol-H3-Spark worker still runs")


class ServerBackend(unittest.TestCase):
    """The server reports SOLH3_BACKEND in /api/state ("backend", spark when unset), names it in the page's
    sol-h3-backend meta tag, and refuses another value. Nothing else depends on it."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="sol-h3-backend-"))
        self.addCleanup(shutil.rmtree, self.d, True)

    def studio(self, name, **env):
        s = StubStudio(self.d / name, load_s=1, clip_s=1, preload=False)
        s.env.pop("SOLH3_BACKEND", None)
        s.env.update(env)
        return s

    def test_unset_is_spark(self):
        s = self.studio("unset").start()
        try:
            self.assertEqual(s.state()["backend"], "spark")
            page = s.get("/")[2].decode()
            self.assertEqual(page, (REPO / "webui" / "index.html").read_text())     # the Spark's page as it is
            self.assertIn('<meta name="sol-h3-backend" content="spark">', page)
        finally:
            s.stop()

    def test_rtx5090(self):
        s = self.studio("rtx", SOLH3_BACKEND="rtx5090").start()
        try:
            self.assertEqual(s.state()["backend"], "rtx5090")
            page = s.get("/")[2].decode()
            self.assertIn('<meta name="sol-h3-backend" content="rtx5090">', page)
            self.assertNotIn('content="spark"', page)
        finally:
            s.stop()

    def test_invalid_refused(self):
        with self.assertRaises(RuntimeError) as e:
            self.studio("bad", SOLH3_BACKEND="gb10").start()
        self.assertIn("SOLH3_BACKEND=gb10: must be spark or rtx5090", str(e.exception))


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
        self.assertEqual(spark["eta"], {"3": 40, "4": 45, "5": 60, "6": 70, "7": 80, "8": 90, "10": 125, "12": 150,
                                        "15": 195})

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
        self.d = Path(tempfile.mkdtemp(prefix="sol-h3-install-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        (self.d / "home").mkdir()
        (self.d / "tmp").mkdir()

    def run_script(self, name, **settings):
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"), SOLH3_ENV=str(self.d / "no-env-file"),
                        SOLH3_HOME=str(self.d / "home" / "sol-h3"), PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}")
        env.update(settings)
        run = subprocess.run(["bash", str(REPO / "install" / name), "--dry-run"], env=env, capture_output=True,
                             text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    def all_scripts(self, **settings):
        return {name: self.run_script(name, **settings) for name in INSTALL}

    def spark(self, **settings):
        return self.all_scripts(FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10", **settings)

    def rtx(self, **settings):
        return self.all_scripts(FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090", SOLH3_BACKEND="rtx5090",
                                **settings)

    def test_scripts(self):
        self.assertEqual(INSTALL, ["00-check.sh", "05-python.sh", "10-sana.sh", "20-weights.sh", "30-sources.sh",
                                   "40-stage1.sh", "50-stage2.sh", "60-qwen-image.sh", "70-manifests.sh",
                                   "80-prompt-cache.sh", "90-env.sh"])

    def test_spark(self):
        out = self.spark()
        pid = re.compile(r"probe-\d+")                                # 30-sources.sh names a file after its pid
        self.assertEqual({k: (c, pid.sub("probe", t)) for k, (c, t) in out.items()},
                         {k: (c, pid.sub("probe", t)) for k, (c, t) in self.spark(SOLH3_BACKEND="spark").items()})
        base = (REPO / "patches" / "BASE_COMMIT").read_text().strip()
        self.assertIn("ok    aarch64", out["00-check.sh"][1])
        self.assertIn("ok    GPU: NVIDIA GB10", out["00-check.sh"][1])
        self.assertIn("+ docker info   (dry run: not run)", out["00-check.sh"][1])
        self.assertEqual(out["05-python.sh"][0], 0)
        self.assertIn("nothing to do", out["05-python.sh"][1])
        self.assertIn(f"sol-engine @ {base}", out["10-sana.sh"][1])
        self.assertIn("patches/sol-h3-studio.patch", out["10-sana.sh"][1])
        self.assertIn("135 CPU tests", out["10-sana.sh"][1])
        self.assertIn("python3.12 -m venv", out["20-weights.sh"][1])
        self.assertIn("models/minimax_h3/Sol-H3-Spark &&", out["20-weights.sh"][1])
        self.assertIn("--include-offline", out["20-weights.sh"][1])
        self.assertIn("python3.12 prepare.py --fetch-sources", out["30-sources.sh"][1])
        code, text = out["40-stage1.sh"]
        if code == 0:                                       # needs /usr/local/cuda, as before
            self.assertIn("fastvideo-kernel", text)
            self.assertNotIn("comfy-kitchen", text)
        else:
            self.assertIn("CUDA 13 toolkit not found at /usr/local/cuda", text)
        self.assertIn('TORCH_CUDA_ARCH_LIST=12.0\\;12.1', out["50-stage2.sh"][1])
        self.assertIn("offline-context", out["50-stage2.sh"][1])
        self.assertNotIn("micromamba", out["50-stage2.sh"][1])
        self.assertIn("docker build -f Dockerfile.qwen", out["60-qwen-image.sh"][1])
        self.assertIn("--python-qwen", out["70-manifests.sh"][1])
        self.assertIn("Sol-H3-Spark/qwen_python.sh", out["70-manifests.sh"][1])
        self.assertIn("--cache-pending", out["70-manifests.sh"][1])
        self.assertIn("runtime.cache_builder", out["80-prompt-cache.sh"][1])
        env = out["90-env.sh"][1]
        for line in ("SOLH3_BACKEND=spark", "SOL_H3_SPARK_QWEN_IMAGE=sol-h3-spark-qwen", "SOLH3_MIN_FREE_GIB=110"):
            self.assertIn(line, env)
        for name, (code, text) in out.items():
            if name not in ("00-check.sh", "40-stage1.sh"):        # 00 and 40 check this machine for real
                self.assertEqual(code, 0, f"{name}: {text}")

    def test_rtx5090(self):
        out = self.rtx()
        check = out["00-check.sh"][1]
        for line in ("ok    SOLH3_BACKEND=rtx5090", "ok    x86_64", "ok    GPU: NVIDIA GeForce RTX 5090",
                     "gcc 16 is newer than nvcc 13.2 accepts"):
            self.assertIn(line, check)
        self.assertNotIn("docker", check)
        self.assertNotIn("/usr/local/cuda", check)
        py312 = str(self.d / "home/sol-h3/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12")
        py313 = str(self.d / "home/sol-h3/python/cpython-3.13.16-linux-x86_64-gnu/bin/python3.13")
        text = out["05-python.sh"][1]
        self.assertIn("releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz", text)
        self.assertIn("check the SHA-256", text)
        self.assertIn("python install --no-bin 3.12.15 3.13.16", text)
        text = out["10-sana.sh"][1]
        self.assertIn(f"sol-engine @ {RTX_BASE}", text)
        self.assertIn("patches/sol-h3-studio-rtx5090.patch", text)
        self.assertIn("198 CPU tests", text)
        self.assertIn(f"{py312} -m venv", out["20-weights.sh"][1])
        self.assertIn("models/minimax_h3/Sol-H3-RTX5090 &&", out["20-weights.sh"][1])
        self.assertNotIn("--include-offline", out["20-weights.sh"][1])
        self.assertIn(f"{py312} prepare.py --fetch-sources", out["30-sources.sh"][1])
        text = out["40-stage1.sh"][1]
        self.assertIn(f"venv --seed --python {py312}", text)
        self.assertIn("comfy-kitchen", text)
        self.assertIn("ComfyUI/requirements.txt", text)
        self.assertIn("stage1-rtx5090-constraints.txt", text)
        self.assertIn("stage1_dependencies", text)
        self.assertNotIn("fastvideo-kernel", text.replace("no fastvideo-kernel", ""))
        text = out["50-stage2.sh"][1]
        self.assertIn(f"venv --seed --python {py313}", text)
        self.assertIn("TORCH_CUDA_ARCH_LIST=12.0 ", text)
        self.assertIn("micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64", text)
        self.assertIn("gxx_linux-64=14", text)
        self.assertIn("NVCC_CCBIN=", text)
        self.assertIn("CFLAGS=-w", text)
        self.assertIn("not needed on the RTX 5090", text)
        self.assertEqual(out["60-qwen-image.sh"][0], 0)
        self.assertNotIn("docker build", out["60-qwen-image.sh"][1])
        text = out["70-manifests.sh"][1]
        self.assertIn(f"--python-qwen {self.d}/home/sol-h3/envs/stage1/bin/python", text)
        self.assertNotIn("qwen_python.sh", text)
        self.assertNotIn("--cache-pending", text)
        code, text = out["80-prompt-cache.sh"]
        self.assertEqual(code, 1, text)
        self.assertIn("SOLH3_PROMPT_CACHE_FROM is not set", text)
        self.assertIn("needs an SM121 GPU", text)
        self.assertNotIn("cache_builder", text)
        env = out["90-env.sh"][1]
        self.assertIn("SOLH3_BACKEND=rtx5090", env)
        self.assertNotIn("SOL_H3_SPARK_QWEN_IMAGE", env)
        self.assertNotIn("\n    SOLH3_MIN_FREE_GIB=", env)                  # the launcher's defaults hold
        self.assertIn("# SOLH3_MIN_FREE_GIB, SOLH3_MIN_FREE_VRAM_GIB: the launcher's defaults", env)
        for name, (code, text) in out.items():
            if name not in ("00-check.sh", "80-prompt-cache.sh"):
                self.assertEqual(code, 0, f"{name}: {text}")

    def test_rtx5090_variants(self):
        """With gcc 15 or older no toolchain is installed; a cache from elsewhere is checked, then copied."""
        text = self.run_script("50-stage2.sh", SOLH3_BACKEND="rtx5090", FAKE_GCC="14")[1]
        self.assertNotIn("micromamba", text)
        self.assertNotIn("CC=", text)
        self.assertIn("CFLAGS=-w", text)
        cache = self.d / "spark-fixed-prompt.pt"
        cache.write_bytes(b"stand-in")
        code, text = self.run_script("80-prompt-cache.sh", SOLH3_BACKEND="rtx5090",
                                     SOLH3_PROMPT_CACHE_FROM=str(cache))
        self.assertEqual(code, 0, text)
        self.assertIn("runtime.prompt_cache.load_cache", text)
        self.assertIn(f"+ cp {cache} {self.d}/home/sol-h3/fixed-prompt.pt", text)
        code, text = self.run_script("80-prompt-cache.sh", SOLH3_BACKEND="rtx5090",
                                     SOLH3_PROMPT_CACHE_FROM=str(self.d / "missing.pt"))
        self.assertEqual(code, 1, text)
        self.assertIn("no such file", text)

    def settings(self, *names, **settings):
        """install/env.sh's values of names, as the install scripts see them."""
        env = clean_env(HOME=str(self.d / "home"), SOLH3_ENV=str(self.d / "no-env-file"),
                        SOLH3_HOME=str(self.d / "home" / "sol-h3"), **settings)
        script = 'source "$1"; shift; for n in "$@"; do printf "%s=%s\n" "$n" "${!n-<unset>}"; done'
        run = subprocess.run(["bash", "-c", script, "-", str(REPO / "install" / "env.sh"), *names], env=env,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        return dict(line.split("=", 1) for line in run.stdout.splitlines())

    def test_caches(self):
        """On the RTX 5090 pip's and uv's caches are under $SOLH3_HOME; the Spark keeps pip's default, as before."""
        home = self.d / "home" / "sol-h3"
        self.assertEqual(self.settings("PIP_CACHE_DIR", "UV_CACHE_DIR", SOLH3_BACKEND="rtx5090"),
                         {"PIP_CACHE_DIR": f"{home}/cache/pip", "UV_CACHE_DIR": f"{home}/cache/uv"})
        self.assertEqual(self.settings("PIP_CACHE_DIR", SOLH3_BACKEND="rtx5090", PIP_CACHE_DIR="/elsewhere"),
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
        code, text = self.run_script("50-stage2.sh", SOLH3_BACKEND="rtx5090", FORCE="1")
        self.assertEqual(code, 0, text)
        self.assertIn("already there (FORCE_TOOLCHAIN=1 installs it again)", text)
        self.assertNotIn(" create ", text)
        self.assertIn(f"NVCC_CCBIN={tc}/bin/x86_64-conda-linux-gnu-g++", text)
        code, text = self.run_script("50-stage2.sh", SOLH3_BACKEND="rtx5090", FORCE="1", FORCE_TOOLCHAIN="1")
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
            done = self.d / f"home/sol-h3/envs/{stage}/.sol-h3-studio-done"
            code, text = self.run_script(script, SOLH3_BACKEND="rtx5090")
            self.assertEqual(code, 0, text)
            self.assertIn(f"has no marker but matches install/freeze/{stage}-rtx5090-freeze.txt: adopted", text)
            self.assertIn(f"+ touch {done}", text)
            self.assertNotIn("pip install -U pip", text)
            self.assertNotIn("ltx-kernels-build", text)
            self.assertNotIn("micromamba", text)
            code, text = self.run_script(script, SOLH3_BACKEND="rtx5090", FORCE="1")         # FORCE=1 rebuilds
            self.assertNotIn("adopted", text)
            self.assertIn("pip install -U pip", text)
            (self.d / f"home/sol-h3/envs/{stage}/bin/freeze.txt").write_text(freeze + "extra==1.0\n")
            code, text = self.run_script(script, SOLH3_BACKEND="rtx5090")
            self.assertEqual(code, 0, text)
            self.assertNotIn("adopted", text)
            self.assertIn("pip install -U pip", text)
            self.assertIn(f"+ touch {done}", text)

    def test_prompt_cache_backup_dated(self):
        """FORCE=1 on the RTX 5090 keeps the old cache under a dated name (it cannot be rebuilt there)."""
        home = self.d / "home/sol-h3"
        home.mkdir(parents=True)
        (home / "fixed-prompt.pt").write_bytes(b"old")
        cache = self.d / "spark-fixed-prompt.pt"
        cache.write_bytes(b"stand-in")
        code, text = self.run_script("80-prompt-cache.sh", SOLH3_BACKEND="rtx5090", FORCE="1",
                                     SOLH3_PROMPT_CACHE_FROM=str(cache))
        self.assertEqual(code, 0, text)
        self.assertRegex(text, rf"(?m)^\+ mv {re.escape(str(home))}/fixed-prompt\.pt "
                               rf"{re.escape(str(home))}/fixed-prompt\.pt\.\d{{8}}-\d{{6}}\.bak$")
        self.assertNotIn("mv -f", text)

    def test_detection(self):
        """On an RTX 5090 without SOLH3_BACKEND, 00-check.sh says what to set and 90-env.sh does not write spark."""
        code, text = self.run_script("00-check.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 1, text)
        self.assertIn("this machine looks like SOLH3_BACKEND=rtx5090, but the install uses spark (the default)", text)
        self.assertIn("export SOLH3_BACKEND=rtx5090", text)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 1, text)
        self.assertIn("looks like SOLH3_BACKEND=rtx5090, but SOLH3_BACKEND is not set", text)
        # a copy of bin/env.example as the settings file does not set it either
        example = self.d / "env-from-example"
        shutil.copy(REPO / "bin" / "env.example", example)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090",
                                     SOLH3_ENV=str(example))
        self.assertEqual(code, 1, text)
        self.assertIn("looks like SOLH3_BACKEND=rtx5090, but SOLH3_BACKEND is not set", text)
        code, text = self.run_script("90-env.sh", FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 4090")
        self.assertEqual(code, 0, text)                                     # nothing detected: spark, as before
        self.assertIn("SOLH3_BACKEND=spark", text)

    def test_machine_mismatch_stops_the_steps(self):
        """A backend that doesn't match the machine stops every step but 00-check.sh (a dry run only warns)."""
        code, text = self.run_script("10-sana.sh", FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10", SOLH3_BACKEND="rtx5090")
        self.assertIn("WARNING: this machine looks like SOLH3_BACKEND=spark, but the install uses rtx5090", text)
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"), SOLH3_ENV=str(self.d / "no-env-file"),
                        SOLH3_HOME=str(self.d / "home" / "sol-h3"), PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10", SOLH3_BACKEND="rtx5090")
        run = subprocess.run(["bash", str(REPO / "install" / "20-weights.sh")], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("ERROR: this machine looks like SOLH3_BACKEND=spark, but the install uses rtx5090", run.stderr)
        self.assertNotIn("===", run.stdout)                                    # stopped before its first step

    def test_sana_refuses_a_checkout_of_the_other_backend(self):
        """10-sana.sh with SOLH3_BACKEND=rtx5090 on a checkout carrying a Spark patch changes nothing."""
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
        (gitdir / "sol-h3-studio-applied.patch").write_text(spark_patch)
        before = (pkg / "a.py").read_text()
        env = clean_env(HOME=str(self.d / "home"), TMPDIR=str(self.d / "tmp"), SOLH3_ENV=str(self.d / "no-env-file"),
                        SOLH3_HOME=str(self.d / "home" / "sol-h3"), SOLH3_SANA=str(sana),
                        PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090", SOLH3_BACKEND="rtx5090")
        run = subprocess.run(["bash", str(REPO / "install" / "10-sana.sh")], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("carries the patch for Sol-H3-Spark (SOLH3_BACKEND=spark), not for Sol-H3-RTX5090; nothing was "
                      "changed", run.stderr)
        self.assertEqual((pkg / "a.py").read_text(), before)

    def test_invalid_backend(self):
        for name in INSTALL:
            code, text = self.run_script(name, SOLH3_BACKEND="gb10")
            self.assertEqual(code, 2, f"{name}: {text}")
            self.assertIn("SOLH3_BACKEND=gb10: must be spark or rtx5090", text)


if __name__ == "__main__":
    unittest.main()
