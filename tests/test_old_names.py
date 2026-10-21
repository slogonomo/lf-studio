"""Until 0.6: an install made with 0.4, when the studio was called Sol-H3 Studio, keeps working with no action
(docs/INSTALL.md, "From 0.4 to 0.5"): the settings' old names SOLH3_* in the environment and in the settings file, the
old places of the settings file, the state directory and the MCP server's clips, the old executables bin/sol-h3-studio
and bin/sol-h3 (shims), the launcher's check for a running copy, and the page's old localStorage keys. The old name of
install/10-sana.sh's applied copy is tested in test_backend.py and test_replay.py. This file goes with the old names
in 0.6.

CPU only, everything in temp directories (HOME too, wherever a default place is read).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_backend import LAUNCHER, REPO, LauncherTest, clean_env, fake_bin, spark_package  # noqa: E402
from test_cli import BIN, client_env  # noqa: E402
from test_mcp import McpTest, text  # noqa: E402
from studio import StubStudio, free_port  # noqa: E402  (test_backend put tests/stub on the path)

OLD_LAUNCHER = REPO / "bin" / "sol-h3-studio"
OLD_BIN = REPO / "bin" / "sol-h3"
WARNING = "settings under their 0.4 names, which work until 0.6"

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class LauncherSettings(LauncherTest):
    """The launcher reads every setting by its new name, else by its old name; the new name wins, the environment
    wins over the file, and one warning on stderr lists the old names used. The memory floor is the setting to watch:
    a refusal names it."""

    def setUp(self):
        super().setUp()
        self.package = spark_package(self.sana)

    def refusal(self, **settings):
        code, out = self.check(LFSTUDIO_TEST_SKIP_DOCKER="1", **settings)
        self.assertEqual(code, 1, out)
        return out

    def test_environment(self):
        out = self.refusal(SOLH3_MIN_FREE_GIB="999")                               # the old name only
        self.assertIn("REFUSED - LF Studio needs ~999 GiB free memory", out)
        self.assertIn(f"lf-studio: {WARNING}: SOLH3_MIN_FREE_GIB. Rename them LFSTUDIO_*", out)
        out = self.refusal(LFSTUDIO_MIN_FREE_GIB="999")                            # the new name only
        self.assertIn("REFUSED - LF Studio needs ~999 GiB free memory", out)
        self.assertNotIn(WARNING, out)
        out = self.refusal(SOLH3_MIN_FREE_GIB="0", LFSTUDIO_MIN_FREE_GIB="998", SOLH3_GUARD_GIB="3")   # both differ
        self.assertIn("REFUSED - LF Studio needs ~998 GiB free memory", out)
        self.assertIn(f"{WARNING}: SOLH3_GUARD_GIB, SOLH3_MIN_FREE_GIB (LFSTUDIO_MIN_FREE_GIB wins).", out)
        self.assertEqual(out.count(WARNING), 1)

    def test_test_hooks_keep_their_names(self):
        """A test hook has no old name: SOLH3_TEST_SKIP_DOCKER is not LFSTUDIO_TEST_SKIP_DOCKER (Docker is checked)."""
        code, out = self.check(SOLH3_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 1, out)
        self.assertIn("cannot talk to docker", out)
        self.assertNotIn(WARNING, out)

    def test_file(self):
        env_file = self.d / "env"
        wins = "SOLH3_MIN_FREE_GIB (LFSTUDIO_MIN_FREE_GIB wins)."
        cases = [("SOLH3_GUARD_GIB=997\nSOLH3_MIN_FREE_GIB=$SOLH3_GUARD_GIB\n", {}, 997,      # an old name in a value
                  "SOLH3_GUARD_GIB, SOLH3_MIN_FREE_GIB."),
                 ("LFSTUDIO_MIN_FREE_GIB=996\nSOLH3_MIN_FREE_GIB=0\n", {}, 996, wins),       # in either order
                 ("SOLH3_MIN_FREE_GIB=0\nLFSTUDIO_MIN_FREE_GIB=995\n", {}, 995, wins),
                 # spark-switchboard's 0.4 service file passes old names, which win over the file's new ones
                 ("LFSTUDIO_MIN_FREE_GIB=999\n", {"SOLH3_MIN_FREE_GIB": "994"}, 994, "SOLH3_MIN_FREE_GIB."),
                 ("SOLH3_MIN_FREE_GIB=999\n", {"LFSTUDIO_MIN_FREE_GIB": "993"}, 993, None)]
        for content, environment, floor, warning in cases:
            with self.subTest(content=content, environment=environment):
                env_file.write_text(content)
                out = self.refusal(LFSTUDIO_ENV=str(env_file), **environment)
                self.assertIn(f"REFUSED - LF Studio needs ~{floor} GiB free memory", out)
                if warning:
                    self.assertIn(f"{WARNING}: {warning}", out)
                else:
                    self.assertNotIn(WARNING, out)

    def default_places(self, home, *args):
        """The launcher with HOME=home and no settings file or state directory set: the default places."""
        env = self.launch_env(*args, HOME=str(home), LFSTUDIO_TEST_SKIP_DOCKER="1")
        del env["LFSTUDIO_ENV"], env["LFSTUDIO_STATE_DIR"]
        run = subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=120)
        return run.returncode, run.stdout + run.stderr

    def test_old_places(self):
        """The 0.4 settings file while the new one does not exist, and the 0.4 state directory while it has the clip
        history (jobs.db) and the new one does not, with a note naming both. A new directory that holds only a
        pidfile (the new spark-switchboard file names one there) does not count."""
        self.port = free_port()
        home = self.d / "home"
        old_config, new_config = home / ".config/sol-h3-studio", home / ".config/lf-studio"
        old_state, new_state = home / ".local/state/sol-h3-studio", home / ".local/state/lf-studio"
        old_config.mkdir(parents=True)
        (old_config / "env").write_text("SOLH3_MIN_FREE_GIB=992\n")
        old_state.mkdir(parents=True)
        (old_state / "jobs.db").write_bytes(b"")
        new_config.mkdir(parents=True)                 # a directory without a settings file in it
        new_state.mkdir(parents=True)
        (new_state / "launcher.pid").write_text("1\n")
        code, out = self.default_places(home, "--check")
        self.assertEqual(code, 1, out)
        self.assertIn(f"lf-studio: using {old_config}/env: {new_config}/env does not exist", out)
        self.assertIn(f"lf-studio: using {old_state}: {new_state} has no jobs.db", out)
        self.assertIn("needs ~992 GiB", out)
        self.assertTrue((old_state / "launcher.log").is_file())
        self.assertFalse((new_state / "launcher.log").exists())
        # new places with content win, without a note
        (new_config / "env").write_text("LFSTUDIO_MIN_FREE_GIB=991\n")
        (new_state / "jobs.db").write_bytes(b"")
        code, out = self.default_places(home, "--check")
        self.assertIn("needs ~991 GiB", out)
        self.assertNotIn("lf-studio: using", out)
        self.assertTrue((new_state / "launcher.log").is_file())

    def test_new_install(self):
        """Without the old places the new ones are used (and made), with no note."""
        self.port = free_port()
        home = self.d / "fresh"
        code, out = self.default_places(home, "--check")
        self.assertNotIn("lf-studio: using", out)
        self.assertTrue((home / ".local/state/lf-studio/launcher.log").is_file())
        self.assertFalse((home / ".local/state/sol-h3-studio").exists())

    def test_running_copy_by_either_name(self):
        """The launcher refuses to start next to a running launcher with its pidfile, by either name (one started
        as bin/sol-h3-studio before the update)."""
        pidfile = self.d / "launcher.pid"
        server = self.d / "exits.py"                  # were the refusal missing, nothing would keep running
        server.write_text("raise SystemExit(3)\n")
        for name in ("bin/sol-h3-studio", "bin/lf-studio"):
            with self.subTest(name=name):
                other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", name])
                try:
                    pidfile.write_text(f"{other.pid}\n")
                    self.port = free_port()
                    code, out = self.launch("", LFSTUDIO_PIDFILE=str(pidfile), LFSTUDIO_SERVER=str(server),
                                            LFSTUDIO_TEST_SKIP_DOCKER="1", LFSTUDIO_MIN_FREE_GIB="0")
                finally:
                    other.kill()
                    other.wait()
                self.assertEqual(code, 1, out)
                self.assertIn(f"REFUSED - another lf-studio launcher is running (pid {other.pid}", out)


class Shims(LauncherTest):
    """bin/sol-h3-studio and bin/sol-h3 run the new files with the same arguments: the same stdout and exit code,
    and one more line on stderr."""

    def setUp(self):
        super().setUp()
        self.package = spark_package(self.sana)
        self.port = free_port()

    def compare(self, old, new, env, *args, line, cwd=None, stdin=None):
        runs = [subprocess.run(cmd + list(args), env=env, capture_output=True, text=True, timeout=60, cwd=cwd,
                               input=stdin) for cmd in (old, new)]
        self.assertEqual(runs[0].returncode, runs[1].returncode)
        self.assertEqual(runs[0].stdout, runs[1].stdout)
        self.assertEqual(runs[0].stderr, line + runs[1].stderr)
        return runs[1]

    def test_launcher(self):
        line = ('sol-h3-studio: renamed to bin/lf-studio; the old name works until 0.6 (docs/INSTALL.md, '
                '"From 0.4 to 0.5")\n')
        env = self.launch_env("--check", LFSTUDIO_BACKEND="gb10")
        run = self.compare(["bash", str(OLD_LAUNCHER)], ["bash", str(LAUNCHER)], env, "--check", line=line)
        self.assertEqual(run.returncode, 1)
        self.assertIn("LFSTUDIO_BACKEND=gb10: must be spark or rtx5090", run.stdout)
        run = self.compare(["bash", str(OLD_LAUNCHER)], ["bash", str(LAUNCHER)], env, "--help", line=line)
        self.assertEqual(run.returncode, 0)
        self.assertTrue(run.stdout.startswith("lf-studio: start the LF Studio web UI"))

    def test_client(self):
        line = 'sol-h3: renamed to bin/lf-cli; the old name works until 0.6 (docs/INSTALL.md, "From 0.4 to 0.5")\n'
        dead = f"http://127.0.0.1:{free_port()}"
        for args, code in ((["--version"], 0), (["nonsense"], 2), (["--url", dead, "status"], 3)):
            with self.subTest(args=args):
                run = self.compare([sys.executable, str(OLD_BIN)], [sys.executable, str(BIN)], client_env(), *args,
                                   line=line)
                self.assertEqual(run.returncode, code, run.stderr)
        # an MCP host registered with `sol-h3 mcp`: stdout carries the protocol only, the note goes to stderr
        request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}) + "\n"
        run = self.compare([sys.executable, str(OLD_BIN)], [sys.executable, str(BIN)], client_env(dead), "mcp",
                           line=line, stdin=request, cwd=self.d)
        answers = [json.loads(x) for x in run.stdout.splitlines()]
        self.assertEqual(answers[0]["result"]["serverInfo"]["name"], "lf-studio")

    def test_lone_copy(self):
        """bin/sol-h3 alone (the old download URL) says where the client went instead of failing obscurely."""
        shutil.copy(OLD_BIN, self.d / "sol-h3")
        run = subprocess.run([sys.executable, str(self.d / "sol-h3"), "--version"], capture_output=True, text=True,
                             timeout=30, env=client_env())
        self.assertEqual((run.returncode, run.stdout), (1, ""))
        self.assertIn("renamed to lf-cli, which is not next to this file", run.stderr)
        self.assertIn("https://raw.githubusercontent.com/slogonomo/lf-studio/main/bin/lf-cli", run.stderr)


class Install(unittest.TestCase):
    """The install scripts (install/env.sh) with the old setting names, the 0.4 settings file and state directory,
    and venvs that 0.4 marked as built. Dry runs, with HOME and everything else in a temp directory."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-install-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        self.home = self.d / "home"
        (self.d / "tmp").mkdir()
        self.data = self.home / "sol-h3"
        self.data.mkdir(parents=True)

    def run_script(self, name, *args, **settings):
        env = clean_env(HOME=str(self.home), TMPDIR=str(self.d / "tmp"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                        LFSTUDIO_HOME=str(self.data), PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10")
        for key, value in settings.items():            # None: unset
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        run = subprocess.run(["bash", str(REPO / "install" / name), *args], env=env, capture_output=True, text=True,
                             timeout=120)
        return run.returncode, run.stdout + run.stderr

    def test_settings(self):
        code, out = self.run_script("20-weights.sh", "--dry-run", SOLH3_TASKS="t2va")
        self.assertEqual(code, 0, out)
        self.assertIn("weights for t2va", out)
        self.assertNotIn("weights for fl2va", out)
        self.assertIn(f"lf-studio: {WARNING}: SOLH3_TASKS. Rename them LFSTUDIO_*", out)
        code, out = self.run_script("20-weights.sh", "--dry-run", SOLH3_TASKS="t2va", LFSTUDIO_TASKS="fl2va")
        self.assertIn("weights for fl2va", out)
        self.assertNotIn("weights for t2va", out)
        self.assertIn(f"{WARNING}: SOLH3_TASKS (LFSTUDIO_TASKS wins).", out)
        code, out = self.run_script("20-weights.sh", "--dry-run", LFSTUDIO_TASKS="ref2va")
        self.assertNotIn(WARNING, out)
        # the dry-run switch by its old name (05-python.sh does nothing on a Spark, dry run or not)
        code, out = self.run_script("05-python.sh", SOLH3_DRY_RUN="1")
        self.assertEqual(code, 0, out)
        self.assertIn("(dry run: commands that change something are printed, not run)", out)
        self.assertIn(f"{WARNING}: SOLH3_DRY_RUN.", out)

    def test_old_places(self):
        """90-env.sh keeps the 0.4 settings file and state directory while the new ones do not exist."""
        old_config, old_state = self.home / ".config/sol-h3-studio", self.home / ".local/state/sol-h3-studio"
        old_config.mkdir(parents=True)
        (old_config / "env").write_text(f'SOLH3_HOME="{self.data}"\nSOLH3_SANA="$SOLH3_HOME/Sana"\nSOLH3_PORT=8191\n')
        old_state.mkdir(parents=True)
        (old_state / "jobs.db").write_bytes(b"")
        code, out = self.run_script("90-env.sh", "--dry-run", LFSTUDIO_ENV=None, LFSTUDIO_HOME=None)
        self.assertEqual(code, 0, out)
        self.assertIn(f"=== 90-env.sh: settings file {old_config}/env\n>>> exists; kept", out)
        self.assertIn(f"lf-studio: using {old_config}/env: {self.home}/.config/lf-studio/env does not exist", out)
        self.assertIn(f"lf-studio: using {old_state}: {self.home}/.local/state/lf-studio has no jobs.db", out)
        for line in (f"LFSTUDIO_SANA={self.data}/Sana", "LFSTUDIO_PORT=8191", f"LFSTUDIO_STATE_DIR={old_state}"):
            self.assertIn(f"\n    {line}\n", out)
        self.assertNotIn("SOLH3_", out.split("It would now contain:")[1].split("+ mkdir")[0])

    def test_done_markers(self):
        """A venv that 0.4 marked as built (.sol-h3-studio-done) counts as built."""
        for stage, script in (("stage1", "40-stage1.sh"), ("stage2", "50-stage2.sh")):
            venv = self.data / "envs" / stage
            venv.mkdir(parents=True)
            (venv / ".sol-h3-studio-done").touch()
            code, out = self.run_script(script, "--dry-run", LFSTUDIO_BACKEND="rtx5090", FAKE_ARCH="x86_64",
                                        FAKE_GPU="NVIDIA GeForce RTX 5090")
            self.assertEqual(code, 0, out)
            self.assertIn(f"{stage.capitalize()} env already built ({venv}); FORCE=1 to rebuild", out)
            self.assertNotIn("pip install -U pip", out)

    def test_torchaudio_relaxed_by_0_4(self):
        """50-stage2.sh's torchaudio fix counts a check relaxed under 0.4's comment as relaxed."""
        venv = self.data / "envs" / "stage2"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
        purelib = subprocess.run([str(venv / "bin/python"), "-c",
                                  "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        utils = Path(purelib) / "torchaudio/_extension/utils.py"
        utils.parent.mkdir(parents=True)
        utils.write_text("def check(ta_version, t_version):\n"
                         "        # sol-h3-studio: only a CUDA major mismatch is fatal (cu130 torchaudio next to cu132 "
                         "torch;\n        # no aarch64 cu132 torchaudio wheel exists)\n"
                         "        if ta_version.split('.')[0] != t_version.split('.')[0]:\n"
                         "            raise RuntimeError\n")
        code, out = self.run_script("50-stage2.sh", "--torchaudio-fix-only", "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn(f"already relaxed: {utils}", out)


class ClientSettings(unittest.TestCase):
    """bin/lf-cli: LFSTUDIO_URL, else SOLH3_URL (one warning); the new name wins. The studio address shows in the
    message for a studio that can't be reached."""

    def status(self, **env):
        run = subprocess.run([sys.executable, str(BIN), "status"], capture_output=True, text=True, timeout=30,
                             env=client_env(**env))
        self.assertEqual(run.returncode, 3, run.stderr)
        return run.stderr

    def test_url(self):
        port = free_port()
        old, new = f"http://127.0.0.1:{port}", f"http://localhost:{port}"
        err = self.status(SOLH3_URL=old)
        self.assertIn(f"can't reach the studio at {old}", err)
        self.assertIn(f"lf-cli: {WARNING}: SOLH3_URL. Rename them LFSTUDIO_*", err)
        err = self.status(LFSTUDIO_URL=new)
        self.assertIn(f"can't reach the studio at {new}", err)
        self.assertNotIn(WARNING, err)
        err = self.status(SOLH3_URL=old, LFSTUDIO_URL=new, SOLH3_MCP_WAIT_S="5")
        self.assertIn(f"can't reach the studio at {new}", err)
        self.assertIn(f"{WARNING}: SOLH3_URL (LFSTUDIO_URL wins), SOLH3_MCP_WAIT_S.", err)
        self.assertEqual(err.count(WARNING), 1)


class McpClipsDirectory(McpTest):
    """The MCP server's default clips directory: ./lf-clips, or ./sol-h3-clips (0.4's) while that exists and
    ./lf-clips is missing or empty."""

    def clip_dir(self, **env):
        m = self.mcp(**env)
        r = m.call("generate", {"prompt": "a lighthouse", "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        return Path(r["structuredContent"]["path"]).parent, Path(m.log.name)

    def test_old_directory(self):
        (self.work / "sol-h3-clips").mkdir()
        where, log = self.clip_dir()
        self.assertEqual(where, self.work / "sol-h3-clips")
        self.assertFalse((self.work / "lf-clips").exists())
        self.assertIn(f"using {self.work}/sol-h3-clips: {self.work}/lf-clips is missing or empty", log.read_text())
        (self.work / "lf-clips").mkdir()
        self.assertEqual(self.clip_dir()[0], self.work / "sol-h3-clips")
        (self.work / "lf-clips" / "notes.txt").write_text("x\n")
        self.assertEqual(self.clip_dir()[0], self.work / "lf-clips")

    def test_old_setting(self):
        where, log = self.clip_dir(SOLH3_OUT_DIR=str(self.dir / "clips"))
        self.assertEqual(where, self.dir / "clips")
        self.assertIn(f"{WARNING}: SOLH3_OUT_DIR.", log.read_text())


class ServerSettings(unittest.TestCase):
    """webui/server.py run without the launcher: the old names, and the 0.4 state directory while it has the clip
    history and the new one does not."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-old-"))
        self.addCleanup(shutil.rmtree, self.d, True)

    def start(self, **env):
        """A stub studio with these settings (None: unset)."""
        studio = StubStudio(self.d / "studio", port=free_port(), load_s=1, clip_s=0.5, preload=False)
        for name in ("LFSTUDIO_TRASH_DAYS", "SOLH3_TRASH_DAYS"):
            studio.env.pop(name, None)
        for name, value in env.items():
            if value is None:
                studio.env.pop(name, None)
            else:
                studio.env[name] = value
        studio.start()
        self.addCleanup(studio.stop)
        return studio

    def test_old_names(self):
        for env, days, warning in (({"SOLH3_TRASH_DAYS": "3"}, 3, "SOLH3_TRASH_DAYS."),
                                   ({"LFSTUDIO_TRASH_DAYS": "4"}, 4, None),
                                   ({"SOLH3_TRASH_DAYS": "3", "LFSTUDIO_TRASH_DAYS": "5"}, 5,
                                    "SOLH3_TRASH_DAYS (LFSTUDIO_TRASH_DAYS wins).")):
            with self.subTest(env=env):
                studio = self.start(**env)
                self.assertEqual(studio.state()["trash_days"], days)
                log = studio.log.read_text()
                studio.stop()
                studio.log.unlink()
                if warning:
                    self.assertIn(f"lf-studio: {WARNING}: {warning}", log)
                else:
                    self.assertNotIn(WARNING, log)

    def test_old_state_directory(self):
        home = self.d / "home"
        old, new = home / ".local/state/sol-h3-studio", home / ".local/state/lf-studio"
        old.mkdir(parents=True)
        (old / "jobs.db").write_bytes(b"")
        new.mkdir(parents=True)
        (new / "launcher.pid").write_text("1\n")       # what the new spark-switchboard file puts there
        studio = self.start(HOME=str(home), LFSTUDIO_STATE_DIR=None)
        log = studio.log.read_text()
        self.assertIn(f"lf-studio: using {old}: {new} has no jobs.db", log)
        self.assertIn(f"state {old})", log)
        self.assertGreater((old / "jobs.db").stat().st_size, 0)
        self.assertFalse((new / "jobs.db").exists())


@unittest.skipUnless(sync_playwright and shutil.which("ffmpeg"), "needs playwright (webkit) and ffmpeg")
class PageKeys(unittest.TestCase):
    """The page copies the settings this browser kept under solh3.* (up to 0.4) to lfstudio.*, where those are
    missing, and then reads the new keys."""

    def test_copy(self):
        d = Path(tempfile.mkdtemp(prefix="lf-studio-page-"))
        self.addCleanup(shutil.rmtree, d, True)
        studio = StubStudio(d, load_s=1, clip_s=0.5, preload=False).start()
        self.addCleanup(studio.stop)
        with sync_playwright() as pw:
            browser = pw.webkit.launch()
            try:
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(studio.url + "/")
                page.evaluate("""() => { localStorage.clear(); localStorage.setItem('solh3.seed', '777');
                                         localStorage.setItem('solh3.prompt', 'kept by 0.4');
                                         localStorage.setItem('lfstudio.prompt', 'kept by 0.5'); }""")
                page.reload()
                page.wait_for_timeout(500)
                self.assertEqual(page.input_value("#seed"), "777")
                self.assertEqual(page.input_value("#prompt"), "kept by 0.5")
                self.assertEqual(page.evaluate("localStorage.getItem('lfstudio.seed')"), "777")
                self.assertEqual(page.evaluate("localStorage.getItem('solh3.seed')"), "777")   # left as it was
                self.assertEqual(errors, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
