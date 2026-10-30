"""The names up to 0.4, when LF Studio was called Sol-H3 Studio, which 0.6 no longer reads (docs/INSTALL.md, "From 0.5
to 0.6"): bin/lf-studio in every mode, the install steps and bench/*.sh (install/old-names.sh, through install/env.sh)
and bin/lf-cli (old_names) stop before anything starts, with one message that names each old name still in use and its
new name. Other names that merely look alike don't stop them, the files of 0.4's names are gone, and nothing else reads
an old name. The stop goes in a later release, and this file with it.

CPU only, everything in temp directories (HOME too, wherever a default place is read).
"""
import ast
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_backend import LAUNCHER, REPO, LauncherTest, clean_env, fake_bin, spark_package  # noqa: E402
from test_cli import BIN, client_env  # noqa: E402
from studio import free_port  # noqa: E402  (test_backend put tests/stub on the path)

STOP = REPO / "install" / "old-names.sh"
HEADER = ('LF Studio 0.6 no longer reads the names it had up to 0.4. Rename each, then try again (docs/INSTALL.md, '
          '"From 0.5 to 0.6"):')
REFUSED = f">>> STATUS: REFUSED - {HEADER}"
# what an old name looks like anywhere in the code (not in the docs, the tests or the patches)
OLD_NAME = re.compile(r"SOLH3_|sol-h3-studio|sol-h3-clips|\bsolh3\.|bin/sol-h3\b")


def bash_settings():
    return re.search(r'^OLD_SETTINGS="([^"]*)"', STOP.read_text(), re.M).group(1).split()


def stop_lines(out):
    """The lines of the stop's message after its header (none: no stop)."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if HEADER in line:
            return [x.strip() for x in lines[i + 1:] if x.startswith("    ")]
    return None


class Listener:
    """A TCP port that listens and never answers: it tells whether anything connected."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"

    def connected(self):
        self.sock.settimeout(0.2)
        try:
            self.sock.accept()[0].close()
            return True
        except OSError:
            return False

    def close(self):
        self.sock.close()


class Launcher(LauncherTest):
    """bin/lf-studio stops in every mode (start, --check, --kill, the client commands) before anything starts, and the
    memory floor is the setting to watch: had a setting been read, the refusal would name its floor."""

    def setUp(self):
        super().setUp()
        self.package = spark_package(self.sana)
        self.port = free_port()
        self.server = self.d / "exits.py"           # were the stop missing, nothing would keep running
        self.server.write_text("raise SystemExit(3)\n")

    def run_launcher(self, *args, unset=(), launcher=LAUNCHER, **settings):
        env = self.launch_env(*args, LFSTUDIO_TEST_SKIP_DOCKER="1", **settings)
        if "--kill" not in args:
            env["LFSTUDIO_SERVER"] = str(self.server)
        for name in unset:
            env.pop(name, None)
        run = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True, timeout=120)
        return run.returncode, run.stdout, run.stderr

    def assert_stop(self, result, lines):
        code, out, err = result
        self.assertEqual(code, 2, out + err)
        self.assertEqual(out, "")                    # nothing else ran, and stdout stays clean (the client's MCP)
        self.assertEqual(err.splitlines()[0], REFUSED, err)
        self.assertEqual(stop_lines(err), lines)

    def test_every_mode(self):
        for args in (("",), ("--check",), ("--kill",), ("status",), ("mcp",)):
            with self.subTest(args=args):
                self.assert_stop(self.run_launcher(*args, SOLH3_MIN_FREE_GIB="0"),
                                 ["SOLH3_MIN_FREE_GIB (environment) -> LFSTUDIO_MIN_FREE_GIB"])
        self.assertFalse((self.d / "state").exists())  # not even the state directory was made

    def test_environment(self):
        result = self.run_launcher("--check", SOLH3_URL="x", SOLH3_PORT="1", LFSTUDIO_MIN_FREE_GIB="0",
                                   SOLH3_STOP_WAIT_S="5")
        self.assert_stop(result, ["SOLH3_PORT (environment) -> LFSTUDIO_PORT",
                                  "SOLH3_STOP_WAIT_S (environment) -> LFSTUDIO_STOP_WAIT_S",
                                  "SOLH3_URL (environment) -> LFSTUDIO_URL"])

    def test_file(self):
        """Each old name in the settings file, with the file and line; a value may use one (no parse error)."""
        env_file = self.d / "env"
        env_file.write_text('# a comment naming SOLH3_HOME=x\nSOLH3_HOME="/data"\nLFSTUDIO_MIN_FREE_GIB=0\n'
                            '  export SOLH3_SANA="$SOLH3_HOME/Sana"\nSOLH3_HOME=/elsewhere\n')
        self.assert_stop(self.run_launcher("--check", LFSTUDIO_ENV=str(env_file)),
                         [f"SOLH3_HOME ({env_file} line 2) -> LFSTUDIO_HOME",
                          f"SOLH3_SANA ({env_file} line 4) -> LFSTUDIO_SANA",
                          f"$SOLH3_HOME ({env_file} line 4) -> $LFSTUDIO_HOME",
                          f"SOLH3_HOME ({env_file} line 5) -> LFSTUDIO_HOME"])
        # the environment and the file together, in one message
        self.assert_stop(self.run_launcher("--check", LFSTUDIO_ENV=str(env_file), SOLH3_HOME="/data"),
                         ["SOLH3_HOME (environment) -> LFSTUDIO_HOME",
                          f"SOLH3_HOME ({env_file} line 2) -> LFSTUDIO_HOME",
                          f"SOLH3_SANA ({env_file} line 4) -> LFSTUDIO_SANA",
                          f"$SOLH3_HOME ({env_file} line 4) -> $LFSTUDIO_HOME",
                          f"SOLH3_HOME ({env_file} line 5) -> LFSTUDIO_HOME"])

    def test_old_name_in_a_value(self):
        """0.5 read $SOLH3_X in a value as $LFSTUDIO_X, and 0.4's env.example wrote values that way. Each one is named,
        also where the line's key has its new name and nothing sets the old name (0.6 would read it as unset: a parse
        error under set -u, or the default of ${SOLH3_X:-...}), and in the same message as the old places, which are
        checked once the file is read. Not in a comment, and not NVIDIA's own SOLH3_ names."""
        home, data = self.d / "home", self.d / "data"
        env_file = home / ".config/lf-studio/env"
        env_file.parent.mkdir(parents=True)
        env_file.write_text(f'LFSTUDIO_HOME="{data}"\n# LFSTUDIO_SANA="$SOLH3_HOME/Sana"\n'
                            'LFSTUDIO_CHECKPOINTS="$SOLH3_HOME/checkpoints"\nLFSTUDIO_PYTHON="$SOLH3_HOME/python"\n'
                            'SOL_H3_SPARK_RUNTIME_ROOT="${SOLH3_HOME:-/elsewhere}/runtime"\n'
                            'SOL_H3_SPARK_COMFY_ROOT="$SOLH3_FP8_COMPILE/x:${SOLH3_CHECKPOINTS}/y:$SOLH3_HOME/z"\n'
                            'LFSTUDIO_MIN_FREE_GIB=0\n')
        old_state = home / ".local/state/sol-h3-studio"
        old_state.mkdir(parents=True)
        (old_state / "jobs.db").write_bytes(b"")

        def run():
            return self.run_launcher("--check", HOME=str(home), SOLH3_FP8_COMPILE="1", unset=(
                "LFSTUDIO_ENV", "LFSTUDIO_STATE_DIR", "LFSTUDIO_PYTHON", "SOL_H3_SPARK_RUNTIME_ROOT"))

        code, out, err = run()
        self.assertNotIn("cannot parse", err)
        self.assert_stop((code, out, err),
                         [f"$SOLH3_HOME ({env_file} line 3) -> $LFSTUDIO_HOME",
                          f"$SOLH3_HOME ({env_file} line 4) -> $LFSTUDIO_HOME",
                          f"$SOLH3_HOME ({env_file} line 5) -> $LFSTUDIO_HOME",
                          f"$SOLH3_CHECKPOINTS ({env_file} line 6) -> $LFSTUDIO_CHECKPOINTS",
                          f"$SOLH3_HOME ({env_file} line 6) -> $LFSTUDIO_HOME",
                          f"{old_state} -> {home}/.local/state/lf-studio (the state folder: the clip history)"])
        # renamed and moved as the message says: the file is read, with the values 0.5 gave
        env_file.write_text(env_file.read_text().replace("SOLH3_HOME", "LFSTUDIO_HOME")
                            .replace("SOLH3_CHECKPOINTS", "LFSTUDIO_CHECKPOINTS"))
        old_state.rename(home / ".local/state/lf-studio")
        data.mkdir()
        os.symlink(sys.executable, data / "python")
        code, out, err = run()
        self.assertEqual(code, 0, out + err)
        self.assertNotIn(HEADER, out + err)
        self.assertTrue((data / "runtime").is_dir())
        self.assertFalse(Path("/elsewhere/runtime").exists())

    def test_settings_file_named_by_its_old_name(self):
        """SOLH3_ENV picked the settings file in 0.5: that file's old names are in the same message."""
        env_file = self.d / "env.test"
        env_file.write_text("SOLH3_PORT=1\nSOLH3_HOME=/x\n")
        self.assert_stop(self.run_launcher("--check", unset=("LFSTUDIO_ENV",), HOME=str(self.d / "home"),
                                           SOLH3_ENV=str(env_file), LFSTUDIO_MIN_FREE_GIB="0"),
                         ["SOLH3_ENV (environment) -> LFSTUDIO_ENV",
                          f"SOLH3_PORT ({env_file} line 1) -> LFSTUDIO_PORT",
                          f"SOLH3_HOME ({env_file} line 2) -> LFSTUDIO_HOME"])

    def test_settings_file_in_the_moved_folder(self):
        """A second studio's settings file named in the old settings folder, after that folder was moved: it would
        start on the defaults (the other studio's install and clip history), so it stops, naming the new place and
        that file's old names. With a link left at the old place it reads the file there."""
        home = self.d / "home"
        old_config, new_config = home / ".config/sol-h3-studio", home / ".config/lf-studio"
        new_config.mkdir(parents=True)
        (new_config / "env").write_text("LFSTUDIO_MIN_FREE_GIB=0\n")
        (new_config / "env.test").write_text("LFSTUDIO_MIN_FREE_GIB=992\nSOLH3_PORT=8191\n")
        named = f"{old_config}/env.test"
        moved = [f"{named} -> {new_config}/env.test (the settings file: its folder was moved)",
                 f"SOLH3_PORT ({new_config}/env.test line 2) -> LFSTUDIO_PORT"]
        self.assert_stop(self.run_launcher("--check", HOME=str(home), LFSTUDIO_ENV=named), moved)
        self.assert_stop(self.run_launcher("--check", unset=("LFSTUDIO_ENV",), HOME=str(home), SOLH3_ENV=named),
                         ["SOLH3_ENV (environment) -> LFSTUDIO_ENV"] + moved)
        (new_config / "env.test").write_text("LFSTUDIO_MIN_FREE_GIB=992\n")
        old_config.symlink_to(new_config)
        code, out, err = self.run_launcher("--check", HOME=str(home), LFSTUDIO_ENV=named)
        self.assertEqual(code, 1, out + err)
        self.assertIn("needs ~992 GiB", out)
        self.assertNotIn(HEADER, out + err)

    def test_names_that_only_look_alike(self):
        """Not the studio's 0.4 settings: NVIDIA's own SOLH3_ names, a test hook, a name that merely contains
        SOLH3_, the pipeline's SOL_H3_ switches."""
        env_file = self.d / "env"
        env_file.write_text("SOLH3_FP8_COMPILE=1\nCOMFY_SOLH3_PORT=1\nSOL_H3_UNUSED=1\n")
        code, out, err = self.run_launcher("--check", LFSTUDIO_ENV=str(env_file), LFSTUDIO_MIN_FREE_GIB="0",
                                           SOLH3_TEST_SKIP_DOCKER="0", SOLH3_DUMP_X="1", COMFY_SOLH3_HOME="x")
        self.assertEqual(code, 0, out + err)
        self.assertIn("checks passed", out)
        self.assertNotIn(HEADER, out + err)

    def default_places(self, home, *args, **settings):
        """The launcher with HOME=home and no settings file or state directory set: the default places."""
        return self.run_launcher(*args, unset=("LFSTUDIO_ENV", "LFSTUDIO_STATE_DIR"), HOME=str(home), **settings)

    def test_old_places(self):
        """The 0.4 settings folder while the new one has no settings file (its file's old names are listed too), and
        the 0.4 state directory while it has the clip history (jobs.db) and the new one does not."""
        home = self.d / "home"
        old_config, new_config = home / ".config/sol-h3-studio", home / ".config/lf-studio"
        old_state, new_state = home / ".local/state/sol-h3-studio", home / ".local/state/lf-studio"
        old_config.mkdir(parents=True)
        (old_config / "env").write_text("LFSTUDIO_MIN_FREE_GIB=992\nSOLH3_PORT=8191\n")
        old_state.mkdir(parents=True)
        (old_state / "jobs.db").write_bytes(b"")
        expected = [f"{old_config} -> {new_config} (the settings folder)",
                    f"SOLH3_PORT ({old_config}/env line 2) -> LFSTUDIO_PORT",
                    f"{old_state} -> {new_state} (the state folder: the clip history)"]
        self.assert_stop(self.default_places(home, "--check"), expected)
        new_config.mkdir(parents=True)                  # a folder without a settings file in it
        new_state.mkdir(parents=True)
        (new_state / "launcher.pid").write_text("1\n")   # a pidfile is not the history
        self.assert_stop(self.default_places(home, "--check"), expected)
        self.assert_stop(self.default_places(home, "--check", SOLH3_STATE_DIR=str(self.d / "state")),   # a setting
                         ["SOLH3_STATE_DIR (environment) -> LFSTUDIO_STATE_DIR"] + expected[:2])
        # once moved (a link left at the old place does not count), the settings and the history are the new ones
        shutil.rmtree(new_config)
        shutil.rmtree(new_state)
        old_config.rename(new_config)
        old_config.symlink_to(new_config)
        old_state.rename(new_state)
        self.assert_stop(self.default_places(home, "--check"),
                         [f"SOLH3_PORT ({new_config}/env line 2) -> LFSTUDIO_PORT"])
        (new_config / "env").write_text("LFSTUDIO_MIN_FREE_GIB=992\nLFSTUDIO_PORT=8191\n")
        code, out, err = self.default_places(home, "--check")
        self.assertEqual(code, 1, out + err)
        self.assertIn("needs ~992 GiB", out)
        self.assertIn(f"port 127.0.0.1:{self.port})", out)    # the environment wins over the file, as before
        self.assertNotIn(HEADER, out + err)
        self.assertTrue((new_state / "launcher.log").is_file())

    def test_settings_name_another_place(self):
        """A settings file or state directory named by a setting: the old default places don't count."""
        home = self.d / "home"
        (home / ".config/sol-h3-studio").mkdir(parents=True)
        (home / ".local/state/sol-h3-studio").mkdir(parents=True)
        (home / ".local/state/sol-h3-studio/jobs.db").write_bytes(b"")
        code, out, err = self.run_launcher("--check", HOME=str(home), LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 0, out + err)
        self.assertNotIn(HEADER, out + err)

    def test_new_install(self):
        """Without the old places the new ones are used (and made)."""
        home = self.d / "fresh"
        code, out, err = self.default_places(home, "--check", LFSTUDIO_MIN_FREE_GIB="0")
        self.assertEqual(code, 0, out + err)
        self.assertTrue((home / ".local/state/lf-studio/launcher.log").is_file())
        self.assertFalse((home / ".local/state/sol-h3-studio").exists())

    def test_run_by_its_old_name(self):
        """A link named sol-h3-studio (the launcher's name up to 0.4) next to install/."""
        tree = self.d / "tree"
        (tree / "bin").mkdir(parents=True)
        (tree / "install").symlink_to(REPO / "install")
        (tree / "bin" / "sol-h3-studio").symlink_to(LAUNCHER)
        self.assert_stop(self.run_launcher("--check", launcher=tree / "bin" / "sol-h3-studio",
                                           LFSTUDIO_MIN_FREE_GIB="0"),
                         ["bin/sol-h3-studio -> bin/lf-studio (the launcher)"])

    def test_running_copy(self):
        """The launcher refuses to start next to a running launcher with its pidfile."""
        pidfile = self.d / "launcher.pid"
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "bin/lf-studio"])
        try:
            pidfile.write_text(f"{other.pid}\n")
            code, out, err = self.run_launcher("", LFSTUDIO_PIDFILE=str(pidfile), LFSTUDIO_MIN_FREE_GIB="0")
        finally:
            other.kill()
            other.wait()
        self.assertEqual(code, 1, out + err)
        self.assertIn(f"REFUSED - another lf-studio launcher is running (pid {other.pid}", out)


class Install(unittest.TestCase):
    """The install steps and bench/*.sh stop in install/env.sh, before any step: also on the venvs' 0.4 done markers
    and a checkout whose only copy of the applied patch has its 0.4 name. Dry runs, with HOME and everything else in a
    temp directory."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-install-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.fake = fake_bin(self.d / "fake")
        self.home = self.d / "home"
        (self.d / "tmp").mkdir()
        self.data = self.home / "sol-h3"
        self.data.mkdir(parents=True)

    def run_script(self, path, *args, **settings):
        env = clean_env(HOME=str(self.home), TMPDIR=str(self.d / "tmp"), LFSTUDIO_ENV=str(self.d / "no-env-file"),
                        LFSTUDIO_HOME=str(self.data), PATH=f"{self.fake}{os.pathsep}{os.environ['PATH']}",
                        FAKE_ARCH="aarch64", FAKE_GPU="NVIDIA GB10")
        for key, value in settings.items():            # None: unset
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        run = subprocess.run(["bash", str(REPO / path), *args], env=env, capture_output=True, text=True, timeout=120)
        return run.returncode, run.stdout, run.stderr

    def assert_stop(self, result, lines):
        code, out, err = result
        self.assertEqual(code, 2, out + err)
        self.assertEqual(out, "")                     # not a single step
        self.assertEqual(err.splitlines()[0], REFUSED, err)
        self.assertEqual(stop_lines(err), lines)

    def test_settings(self):
        self.assert_stop(self.run_script("install/20-weights.sh", "--dry-run", SOLH3_TASKS="t2va"),
                         ["SOLH3_TASKS (environment) -> LFSTUDIO_TASKS"])
        env_file = self.d / "env"
        env_file.write_text("SOLH3_SANA=/somewhere\n")
        self.assert_stop(self.run_script("install/10-sana.sh", "--dry-run", LFSTUDIO_ENV=str(env_file)),
                         [f"SOLH3_SANA ({env_file} line 1) -> LFSTUDIO_SANA"])
        # an old name in a value that nothing sets: no parse error, and the checks after the file still run
        venv = self.data / "envs" / "stage1"
        venv.mkdir(parents=True)
        (venv / ".sol-h3-studio-done").touch()
        env_file.write_text('LFSTUDIO_PYTHON="$SOLH3_HOME/envs/stage1/bin/python"\n')
        code, out, err = self.run_script("install/40-stage1.sh", "--dry-run", LFSTUDIO_ENV=str(env_file))
        self.assertNotIn("cannot parse", err)
        self.assert_stop((code, out, err),
                         [f"$SOLH3_HOME ({env_file} line 1) -> $LFSTUDIO_HOME",
                          f"{venv}/.sol-h3-studio-done -> {venv}/.lf-studio-done (the venv's done marker)"])

    def test_dry_run_by_its_old_name(self):
        """SOLH3_DRY_RUN=1 alone: the step stops, it does not run for real."""
        self.assert_stop(self.run_script("install/90-env.sh", SOLH3_DRY_RUN="1"),
                         ["SOLH3_DRY_RUN (environment) -> LFSTUDIO_DRY_RUN"])
        self.assertFalse((self.home / ".config").exists())
        self.assertFalse((self.home / ".local").exists())

    def test_old_places(self):
        old_config, old_state = self.home / ".config/sol-h3-studio", self.home / ".local/state/sol-h3-studio"
        old_config.mkdir(parents=True)
        (old_config / "env").write_text(f'SOLH3_HOME="{self.data}"\nLFSTUDIO_PORT=8191\n')
        old_state.mkdir(parents=True)
        (old_state / "jobs.db").write_bytes(b"")
        self.assert_stop(self.run_script("install/90-env.sh", "--dry-run", LFSTUDIO_ENV=None, LFSTUDIO_HOME=None),
                         [f"{old_config} -> {self.home}/.config/lf-studio (the settings folder)",
                          f"SOLH3_HOME ({old_config}/env line 1) -> LFSTUDIO_HOME",
                          f"{old_state} -> {self.home}/.local/state/lf-studio (the state folder: the clip history)"])

    def test_done_markers(self):
        """A venv that only 0.4 marked as built would be built into: stop, and name the marker to rename."""
        venvs = [self.data / "envs" / stage for stage in ("stage1", "stage2")]
        for venv in venvs:
            venv.mkdir(parents=True)
            (venv / ".sol-h3-studio-done").touch()
        expected = [f"{v}/.sol-h3-studio-done -> {v}/.lf-studio-done (the venv's done marker)" for v in venvs]
        for script in ("install/40-stage1.sh", "install/50-stage2.sh", "install/00-check.sh"):
            with self.subTest(script=script):
                self.assert_stop(self.run_script(script, "--dry-run"), expected)
        # the marker's home is the setting's, by either name
        self.assert_stop(self.run_script("install/40-stage1.sh", "--dry-run", LFSTUDIO_HOME=None,
                                         SOLH3_HOME=str(self.data)),
                         ["SOLH3_HOME (environment) -> LFSTUDIO_HOME"] + expected)
        for venv in venvs:                             # renamed as the message says
            (venv / ".sol-h3-studio-done").rename(venv / ".lf-studio-done")
        code, out, err = self.run_script("install/40-stage1.sh", "--dry-run", LFSTUDIO_BACKEND="rtx5090",
                                         FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"Stage1 env already built ({venvs[0]}); FORCE=1 to rebuild", out)
        # both markers (0.5 kept the old one): the new one counts, no stop
        (venvs[0] / ".sol-h3-studio-done").touch()
        code, out, err = self.run_script("install/40-stage1.sh", "--dry-run", LFSTUDIO_BACKEND="rtx5090",
                                         FAKE_ARCH="x86_64", FAKE_GPU="NVIDIA GeForce RTX 5090")
        self.assertEqual(code, 0, out + err)
        self.assertNotIn(HEADER, err)

    def test_applied_copy(self):
        """A checkout last patched by 0.4 has its copy of the patch only as sol-h3-studio-applied.patch: 10-sana.sh
        stops and names it. Renamed as the message says, 10-sana.sh finds the earlier patch there, replaces it and
        keeps the new copy, under the new name only. A stale 0.4 copy next to the new one (0.5 wrote both) is left
        alone."""
        sana = self.data / "Sana"
        git = ["git", "-C", str(sana), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        subprocess.run(["git", "init", "-q", str(sana)], check=True)
        (sana / "infer.py").write_text("print('upstream')\n")
        subprocess.run(git + ["add", "infer.py"], check=True)
        subprocess.run(git + ["commit", "-qm", "base"], check=True)
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        (sana / "infer.py").write_text("# Modified by an earlier version\nprint('patched')\n")
        gitdir = sana / ".git"
        old = gitdir / "sol-h3-studio-applied.patch"
        old.write_text(subprocess.run(git + ["diff"], capture_output=True, text=True, check=True).stdout)
        (gitdir / "sol-h3-studio-applied.patch.base").write_text(head + "\n")
        self.assert_stop(self.run_script("install/10-sana.sh", "--dry-run", LFSTUDIO_SANA_COMMIT=head),
                         [f"{old} -> {gitdir}/lf-studio-applied.patch (and its .base: the patch applied)"])
        new = gitdir / "lf-studio-applied.patch"
        old.rename(new)
        old.with_name(old.name + ".base").rename(new.with_name(new.name + ".base"))
        code, out, err = self.run_script("install/10-sana.sh", "--dry-run", LFSTUDIO_SANA_COMMIT=head)
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"the checkout carries an earlier version of the patch ({new}); replacing it", out)
        self.assertIn(f"+ git -C {sana} apply -R {new}\n+ git -C {sana} apply {REPO}/patches/lf-studio.patch", out)
        self.assertIn(f"+ cp {REPO}/patches/lf-studio.patch {new}\n", out)
        self.assertNotIn("sol-h3-studio-applied", out + err)        # neither read nor written
        old.write_text("a stale copy\n")
        code, out, err = self.run_script("install/10-sana.sh", "--dry-run", LFSTUDIO_SANA_COMMIT=head)
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"the checkout carries an earlier version of the patch ({new}); replacing it", out)
        self.assertNotIn("sol-h3-studio-applied", out + err)

    def test_bench(self):
        env_file = self.d / "env"
        env_file.write_text(f'SOLH3_HOME="{self.data}"\n')
        for path, args in (("bench/replay.sh", ("list",)), ("bench/first-clip.sh", ("--help",))):
            with self.subTest(path=path):
                self.assert_stop(self.run_script(path, *args, LFSTUDIO_ENV=str(env_file)),
                                 [f"SOLH3_HOME ({env_file} line 1) -> LFSTUDIO_HOME"])


class Client(unittest.TestCase):
    """bin/lf-cli: the old names it read until 0.6 stop it with exit 2 before anything is sent or read from stdin, with
    nothing on stdout (where MCP speaks)."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="lf-studio-cli-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.studio = Listener()
        self.addCleanup(self.studio.close)

    def run_cli(self, *args, program=BIN, cwd=None, stdin="", url=None, **env):
        run = subprocess.run([sys.executable, str(program), *args], env=client_env(url or self.studio.url, **env),
                             capture_output=True, text=True, timeout=60, cwd=cwd or self.d, input=stdin)
        return run.returncode, run.stdout, run.stderr

    def assert_no_stop(self, *args, **env):
        """No stop: the client goes on to the studio (one that can't be reached, exit 3)."""
        dead = f"http://127.0.0.1:{free_port()}"
        code, out, err = self.run_cli(*args, url=dead, **env)
        self.assertEqual(code, 3, err)
        self.assertIn(f"can't reach the studio at {dead}", err)
        self.assertNotIn(HEADER, err)

    def assert_stop(self, result, lines):
        code, out, err = result
        self.assertEqual((code, out), (2, ""), err)
        self.assertEqual(err.splitlines()[0], f"lf-cli: {HEADER}", err)
        self.assertEqual(stop_lines(err), lines)
        self.assertFalse(self.studio.connected())

    def test_settings(self):
        names = ast.literal_eval(re.search(r"^OLD_SETTINGS = (\(.*?\))", BIN.read_text(), re.M | re.S).group(1))
        self.assertEqual(sorted(set(names) - set(bash_settings())), [])       # the launcher's names, and fewer
        for name in names:
            with self.subTest(name=name):
                self.assert_stop(self.run_cli("status", **{f"SOLH3_{name}": "1"}),
                                 [f"SOLH3_{name} (environment) -> LFSTUDIO_{name}"])
        self.assert_stop(self.run_cli("generate", "--prompt", "x", SOLH3_URL="x", SOLH3_MCP_WAIT_S="5"),
                         ["SOLH3_URL (environment) -> LFSTUDIO_URL",
                          "SOLH3_MCP_WAIT_S (environment) -> LFSTUDIO_MCP_WAIT_S"])

    def test_mcp(self):
        request = ('{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", '
                   '"capabilities": {}}}\n')
        self.assert_stop(self.run_cli("mcp", stdin=request, SOLH3_URL=self.studio.url),
                         ["SOLH3_URL (environment) -> LFSTUDIO_URL"])

    def test_names_that_only_look_alike(self):
        """Not the client's 0.4 settings: they don't stop it."""
        self.assert_no_stop("status", SOLH3_HOME="x", SOLH3_PIDFILE="x", COMFY_SOLH3_URL="x", SOL_H3_URL="x")

    def test_run_by_its_old_name(self):
        link = self.d / "sol-h3"
        link.symlink_to(BIN)
        self.assert_stop(self.run_cli("--url", self.studio.url, "status", program=link),
                         ["sol-h3 -> lf-cli (the client)"])

    def test_mcp_clips_folder(self):
        """The MCP server's 0.4 clips folder while ./lf-clips is missing or empty and no setting names a folder."""
        work = self.d / "work"
        (work / "sol-h3-clips").mkdir(parents=True)
        line = [f"{work}/sol-h3-clips -> {work}/lf-clips (the MCP server's clips)"]
        self.assert_stop(self.run_cli("mcp", cwd=work), line)
        (work / "lf-clips").mkdir()
        self.assert_stop(self.run_cli("mcp", cwd=work), line)
        self.assert_stop(self.run_cli("mcp", cwd=work, SOLH3_OUT_DIR="lf-clips"),
                         ["SOLH3_OUT_DIR (environment) -> LFSTUDIO_OUT_DIR"])
        for settings in ({"LFSTUDIO_OUT_DIR": "lf-clips"}, {}):
            if not settings:
                (work / "lf-clips" / "clip.mp4").write_bytes(b"")
            with self.subTest(settings=settings):
                code, out, err = self.run_cli("mcp", cwd=work, **settings)       # stdin closed: the server ends
                self.assertEqual((code, out), (0, ""), err)
                self.assertNotIn(HEADER, err)
        shutil.rmtree(work / "lf-clips")
        self.assert_no_stop("status", cwd=work)          # only the MCP server has the folder


class Message(unittest.TestCase):
    """The message points to a section that exists, and that section's table has every old setting name."""

    def test_upgrade_section(self):
        install = (REPO / "docs" / "INSTALL.md").read_text()
        self.assertIn("\n### From 0.5 to 0.6\n", install)
        section = install.split("\n### From 0.5 to 0.6\n", 1)[1].split("\n### ", 1)[0]
        table = re.findall(r"\| `SOLH3_([A-Z0-9_]+)` \| `LFSTUDIO_([A-Z0-9_]+)` ", section)
        self.assertEqual([old for old, new in table], [new for old, new in table])
        self.assertEqual(sorted(old for old, new in table), sorted(bash_settings()))
        self.assertEqual(len(bash_settings()), len(set(bash_settings())))


class Gone(unittest.TestCase):
    """The files of the 0.4 names are gone, and nothing but the stop reads an old name."""

    def test_files(self):
        for path in ("bin/sol-h3-studio", "bin/sol-h3", "integrations/spark-switchboard/sol-h3-studio.toml",
                     "integrations/systemd/sol-h3-studio.service"):
            self.assertFalse((REPO / path).exists(), path)

    def test_no_old_name_outside_the_stop(self):
        """The code (not the docs, tests or patches) names a 0.4 name only in the stop and in comments that say so:
        install/old-names.sh, bin/lf-cli's docstring and old_names, comments in bin/lf-studio and install/10-sana.sh."""
        found = []
        for top in ("bin", "webui", "install", "bench", "jobs", "upscale", "integrations", "qwen-image"):
            for path in sorted((REPO / top).rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path == STOP:
                    continue
                try:
                    text = path.read_text()
                except UnicodeDecodeError:
                    continue
                rel = str(path.relative_to(REPO))
                if rel == "bin/lf-cli":                  # the docstring and the stop
                    text = text.split('"""\nimport argparse', 1)[1]
                    head, stop = text.split("# The stop for the names up to 0.4", 1)
                    text = head + stop.split("\ndef main(", 1)[1]
                for n, line in enumerate(text.splitlines(), 1):
                    if OLD_NAME.search(line) and not (rel in ("bin/lf-studio", "install/10-sana.sh")
                                                      and line.lstrip().startswith("#")):
                        found.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
