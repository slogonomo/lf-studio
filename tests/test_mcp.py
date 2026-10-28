"""bin/lf-cli mcp, the MCP server for agents (#29), driven over stdio with raw JSON-RPC lines (no SDK), against the
stub studio (tests/stub) and small fake servers. No models, no GPU.

python3 -m unittest tests.test_mcp -v      (or with the rest: python3 -m unittest discover -s tests)
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cli import BIN, JSON, Fake, client_env, duration, free_port, load_cli, png, switchboard_error  # noqa: E402
from studio import StubStudio  # noqa: E402  (test_cli put tests/stub on the path)

MODERN_META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
               "io.modelcontextprotocol/clientCapabilities": {},
               "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"}}


class Mcp:
    """bin/lf-cli mcp as a subprocess: send requests, read responses by id, keep notifications and every stdout line."""

    def __init__(self, cwd, url, log, **env):
        self.log = open(log, "w")
        self.proc = subprocess.Popen([sys.executable, str(BIN), "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, cwd=cwd, env=client_env(url, **env))
        self.lines, self.inbox = [], queue.Queue()
        self.responses, self.notes = {}, []
        self.next_id = 100
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            try:
                self.inbox.put(json.loads(line))
            except ValueError:
                self.inbox.put({"unparsable": line})
        self.inbox.put(None)

    def send(self, msg):
        self.proc.stdin.write(json.dumps(msg).encode() + b"\n")
        self.proc.stdin.flush()

    def pump(self, timeout):
        """Read what arrives within `timeout` seconds into responses and notes."""
        try:
            msg = self.inbox.get(timeout=timeout)
        except queue.Empty:
            return
        if msg is None:
            raise AssertionError("the MCP server closed its stdout")
        if msg.get("id") is not None and "method" not in msg:
            self.responses[json.dumps(msg["id"])] = msg
        else:
            self.notes.append(msg)

    def response(self, rid, timeout=90):
        deadline = time.time() + timeout
        while json.dumps(rid) not in self.responses:
            if time.time() > deadline:
                raise AssertionError(f"no response to {rid} in {timeout} s")
            self.pump(0.2)
        return self.responses.pop(json.dumps(rid))

    def request(self, method, params=None, rid=None, timeout=90):
        if rid is None:
            self.next_id += 1
            rid = self.next_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.response(rid, timeout)

    def call(self, name, arguments=None, meta=None, timeout=90):
        """tools/call: the result (a protocol error fails the test)."""
        params = {"name": name, "arguments": arguments or {}}
        if meta:
            params["_meta"] = meta
        msg = self.request("tools/call", params, timeout=timeout)
        assert "result" in msg, msg
        return msg["result"]

    def close(self):
        if not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=5)
        self.proc.stdout.close()
        self.log.close()


def text(result):
    return result["content"][0]["text"]


class McpTest(unittest.TestCase):
    """One fresh stub studio (with studio_env's settings) and one MCP server per test; the MCP server runs in
    self.work."""
    load_s, clip_s = 1, 0.5
    studio_needed = True
    studio_env = {}

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-mcp-"))
        self.work = self.dir / "work"
        self.work.mkdir()
        self.studio = None
        if self.studio_needed:
            self.studio = StubStudio(self.dir / "studio", port=free_port(), load_s=self.load_s, clip_s=self.clip_s,
                                     preload=False, env=self.studio_env).start()
        self.servers = []

    def mcp(self, url=None, cwd=None, **env):
        server = Mcp(cwd or self.work, url or self.studio.url, self.dir / f"mcp-{len(self.servers)}.log", **env)
        self.servers.append(server)
        return server

    def tearDown(self):
        for server in self.servers:
            server.close()
            for line in server.lines:                    # 9.3 #11: stdout carries protocol messages only
                json.loads(line)
        failed = hasattr(self, "_outcome") and not getattr(self._outcome, "success", True)
        if failed:
            for server in self.servers:
                print(f"\n--- MCP stderr\n{Path(server.log.name).read_text()[-3000:]}")
        if self.studio:
            self.studio.stop()
            if failed:
                print(f"\n--- server log\n{self.studio.log.read_text()[-3000:]}")
        shutil.rmtree(self.dir, ignore_errors=True)

    def frame(self, name="f.png", width=42, height=24):
        (self.work / name).write_bytes(png(width, height))
        return name


class Protocol(McpTest):
    studio_needed = False

    def setUp(self):
        super().setUp()
        self.m = self.mcp(url=f"http://127.0.0.1:{free_port()}")    # these tests never reach a studio

    def test_handshake(self):
        """9.3 #1: initialize echoes a version it knows, else answers the newest; initialized gets no reply; ping."""
        r = self.m.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "t", "version": "1"}})["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")
        self.assertEqual(r["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(r["serverInfo"]["name"], "lf-studio")
        self.assertLessEqual(len(r["instructions"]), 2048)
        r = self.m.request("initialize", {"protocolVersion": "1999-01-01", "capabilities": {}})["result"]
        self.assertEqual(r["protocolVersion"], "2025-11-25")
        self.m.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(self.m.request("ping")["result"], {"resultType": "complete"})
        self.assertEqual(self.m.notes, [])

    def test_modern_era(self):
        """9.3 #2: server/discover; an unknown modern version is -32022; modern requests need clientCapabilities."""
        r = self.m.request("server/discover", {"_meta": MODERN_META})["result"]
        self.assertIn("2026-07-28", r["supportedVersions"])
        self.assertEqual((r["resultType"], r["cacheScope"], r["ttlMs"]), ("complete", "public", 3600000))
        self.assertEqual(r["_meta"]["io.modelcontextprotocol/serverInfo"]["name"], "lf-studio")
        meta = dict(MODERN_META, **{"io.modelcontextprotocol/protocolVersion": "2099-01-01"})
        e = self.m.request("tools/list", {"_meta": meta})["error"]
        self.assertEqual(e["code"], -32022)
        self.assertEqual(e["data"]["requested"], "2099-01-01")
        self.assertIn("2026-07-28", e["data"]["supported"])
        self.assertIn("2025-11-25", e["data"]["supported"])
        r = self.m.request("tools/list", {"_meta": MODERN_META})["result"]
        self.assertEqual(len(r["tools"]), 6)
        self.assertIn("_meta", r)
        e = self.m.request("tools/list", {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}})["error"]
        self.assertEqual(e["code"], -32602)
        r = self.m.call("prompt_guide", meta=MODERN_META)
        self.assertEqual(r["resultType"], "complete")

    def test_errors(self):
        """9.3 #3: unknown method -32601, unknown tool -32602, a bad line -32700, a batch -32600, bad arguments are a
        tool error the agent can read."""
        self.assertEqual(self.m.request("resources/list")["error"]["code"], -32601)
        self.assertEqual(self.m.request("tools/call", {"name": "nope", "arguments": {}})["error"]["code"], -32602)
        self.m.send({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": "not an object"})
        self.assertEqual(self.m.response(7)["error"]["code"], -32602)
        self.m.proc.stdin.write(b"{not json\n[1, 2]\n")
        self.m.proc.stdin.flush()
        deadline = time.time() + 10
        while len([n for n in self.m.notes if n.get("error")]) < 2 and time.time() < deadline:
            self.m.pump(0.2)
        codes = [n["error"]["code"] for n in self.m.notes if n.get("error")]
        self.assertEqual(sorted(codes), [-32700, -32600])
        for args, words in [({"prompt": "x", "frist_frame": "a.png"}, 'unknown argument "frist_frame"'),
                            ({"prompt": "x", "seconds": 9}, "seconds must be one of 5, 6, 7, 8, 10, 12, 15"),
                            ({"prompt": "x", "seconds": 3}, "seconds must be one of 5, 6, 7, 8, 10, 12, 15"),
                            ({"prompt": "x", "seconds": 4}, "seconds must be one of 5, 6, 7, 8, 10, 12, 15"),
                            ({"prompt": "x", "seconds": "5"}, "seconds must be an integer"),
                            ({"prompt": "x", "seed": -1}, "seed must be 0 or more"),
                            ({"prompt": "x", "references": "a.png"}, "references must be a list of strings"),
                            ({"prompt": "dawn \ud83c"}, "the prompt is not valid text at character 6"),
                            ({}, "prompt is required")]:
            r = self.m.call("generate", args)
            self.assertTrue(r["isError"], args)
            self.assertIn(words, text(r))

    def test_tools_list(self):
        """9.3 #4: six flat tools Claude Code accepts, with short enough descriptions."""
        r = self.m.request("tools/list")["result"]
        self.assertEqual([t["name"] for t in r["tools"]], ["generate", "wait", "status", "cancel", "fetch",
                                                            "prompt_guide"])
        self.assertEqual((r["ttlMs"], r["cacheScope"]), (3600000, "public"))
        for tool in r["tools"]:
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object")
            self.assertFalse({"anyOf", "oneOf", "allOf"} & set(schema), tool["name"])
            self.assertFalse(schema["additionalProperties"])
            for name in schema["properties"]:
                self.assertRegex(name, r"^[A-Za-z0-9_.-]{1,64}$")
            self.assertLessEqual(len(tool["description"]), 2048)
        self.assertEqual(r["tools"][0]["inputSchema"]["properties"]["seconds"],
                         {"type": "integer", "enum": [5, 6, 7, 8, 10, 12, 15], "default": 5})   # 3 and 4 s went in 0.5
        generate = r["tools"][0]["description"]
        self.assertIn("stretched or cropped", generate)                          # DECISIONS 12
        self.assertIn("change the seed", generate)                               # DECISIONS 14
        self.assertIn("never removes a clip", generate)

    def test_end_of_input(self):
        """9.3 #11: closing stdin ends the process within 2 s."""
        self.m.request("ping")
        t = time.time()
        self.m.proc.stdin.close()
        self.m.proc.wait(timeout=5)
        self.assertLess(time.time() - t, 2)


class Tools(McpTest):
    def test_generate_and_save(self):
        """9.3 #5: generate with a local first frame waits, saves the MP4 and returns text, link and poster."""
        m = self.mcp()
        r = m.call("generate", {"prompt": "a lighthouse", "first_frame": self.frame("tall.png", 24, 42),
                                "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual([c["type"] for c in r["content"]], ["text", "resource_link", "image"])
        rec = r["structuredContent"]
        self.assertEqual((rec["state"], rec["task"], rec["orientation"]), ("done", "fl2va", "portrait"))
        path = Path(rec["path"])
        self.assertTrue(path.is_absolute() and path.is_file())
        self.assertEqual(path.parent, self.work / "lf-clips")
        self.assertAlmostEqual(duration(path), 5.04, delta=0.1)
        self.assertEqual(rec["bytes"], path.stat().st_size)
        self.assertEqual(Path(rec["poster_path"]).read_bytes()[:2], b"\xff\xd8")
        self.assertEqual(rec["labels"], [{"label": "<Picture 1>", "zone": "first_frame", "index": 0,
                                          "name": "tall.png"}])
        self.assertEqual(rec["url"], f"{self.studio.url}/video/{rec['id']}")
        self.assertIn(f"Clip {rec['id']} is done: saved {path} (", text(r))
        self.assertIn("Labels: <Picture 1> = first_frame tall.png.", text(r))
        self.assertIn("Orientation: portrait, from the shape of first_frame (24x42).", text(r))
        link = r["content"][1]
        self.assertEqual((link["uri"], link["mimeType"], link["name"]), (rec["url"], "video/mp4", path.name))
        self.assertEqual(r["content"][2]["mimeType"], "image/jpeg")
        self.assertEqual(json.loads(text(r).split("\n\n", 1)[1]), rec)          # the JSON is in the text as well

    def test_older_protocol_versions_get_no_resource_link(self):
        """resource_link content exists from 2025-06-18 on: a client that agreed on 2025-03-26 or 2024-11-05 (the
        official SDK before 1.10) rejects a result holding one, so it gets the text (with the studio URL in the JSON)
        and the poster only."""
        m = self.mcp()
        hello = {"capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}
        m.request("initialize", dict(hello, protocolVersion="2025-03-26"))
        r = m.call("generate", {"prompt": "x", "seconds": 6, "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual([c["type"] for c in r["content"]], ["text", "image"])
        jid, url = r["structuredContent"]["id"], r["structuredContent"]["url"]
        self.assertEqual(json.loads(text(r).split("\n\n", 1)[1])["url"], url)
        m.request("initialize", dict(hello, protocolVersion="2024-11-05"))
        self.assertEqual([c["type"] for c in m.call("fetch", {"id": jid})["content"]], ["text", "image"])
        r = m.call("fetch", {"id": jid}, meta=MODERN_META)                       # a 2026-07-28 request
        self.assertEqual([c["type"] for c in r["content"]], ["text", "resource_link", "image"])
        m.request("initialize", dict(hello, protocolVersion="2025-06-18"))
        r = m.call("fetch", {"id": jid})
        self.assertEqual([c["type"] for c in r["content"]], ["text", "resource_link", "image"])

    def test_short_waits(self):
        """9.3 #6: wait_s 0 returns the id at once; a short wait says it is still running (not an error); wait."""
        m = self.mcp()
        self.studio.generate(prompt="ahead of it")
        r = m.call("generate", {"prompt": "x", "wait_s": 0})
        rec = r["structuredContent"]
        self.assertEqual((r["isError"], rec["state"]), (False, "queued"))
        self.assertIn(rec["ahead"], (0, 1))
        self.assertIn(f'Call wait with id "{rec["id"]}"', text(r))
        r = m.call("wait", {"id": rec["id"], "wait_s": 0})
        self.assertFalse(r["isError"])
        self.assertIn(f"Clip {rec['id']} is ", text(r))
        r = m.call("wait", {"id": rec["id"], "wait_s": 30, "poster": False})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual([c["type"] for c in r["content"]], ["text", "resource_link"])
        self.assertNotIn("poster_path", r["structuredContent"])
        r = m.call("status", {"id": rec["id"]})
        self.assertEqual(r["structuredContent"]["state"], "done")
        r = m.call("status")                                                     # 9.3 #10
        self.assertTrue(text(r).startswith("Submitted from this session (oldest first):\n" + rec["id"]))
        self.assertEqual(len(r["structuredContent"]["jobs"]), 2)
        self.assertEqual(r["structuredContent"]["session"][0]["state"], "done")

    def test_failures(self):
        """9.3 #9: a failed clip, an unknown id, cancel on a finished clip are tool errors; cancel on a queued one
        works."""
        m = self.mcp()
        r = m.call("generate", {"prompt": "this one will STUB_FAIL", "wait_s": 30})
        self.assertTrue(r["isError"])
        self.assertIn("failed: RuntimeError: Worker exited with code 1 (STUB_FAIL)", text(r))
        done = r["structuredContent"]["id"]
        for tool in ("wait", "status", "fetch", "cancel"):
            r = m.call(tool, {"id": "f" * 32})
            self.assertTrue(r["isError"], tool)
            self.assertIn("o clip with id ffff", text(r))
        r = m.call("wait", {"id": "nope"})
        self.assertTrue(r["isError"])
        self.assertIn("not a clip id", text(r))
        r = m.call("cancel", {"id": done})
        self.assertTrue(r["isError"])
        self.assertIn("the clip has already failed", text(r))
        self.studio.generate(prompt="first")
        jid = m.call("generate", {"prompt": "second", "wait_s": 0})["structuredContent"]["id"]
        r = m.call("cancel", {"id": jid})
        self.assertFalse(r["isError"])
        self.assertEqual(text(r).split("\n")[0], f"Clip {jid} was cancelled (it had not started).")
        r = m.call("wait", {"id": jid})
        self.assertFalse(r["isError"])
        self.assertTrue(text(r).startswith(f"Clip {jid} was cancelled."))

    def test_prompt_guide(self):
        """9.3 #10: prompt_guide returns the guide, the same text as lf-cli guide."""
        r = self.mcp().call("prompt_guide")
        self.assertEqual(r["content"], [{"type": "text", "text": load_cli().GUIDE}])

    def test_identical_generate(self):
        """DECISIONS 14: an identical generate returns the earlier clip; again: true or a new seed makes another; a
        failed one is made again."""
        m = self.mcp()
        args = {"prompt": "x", "first_frame": self.frame(), "wait_s": 30}
        first = m.call("generate", args)["structuredContent"]
        r = m.call("generate", args)
        self.assertIn(f"Already submitted as {first['id']}", text(r))
        self.assertEqual(r["structuredContent"]["id"], first["id"])
        self.assertEqual(r["structuredContent"]["path"], first["path"])          # not downloaded a second time
        self.assertEqual(len(list((self.work / "lf-clips").glob("*.mp4"))), 1)
        r = m.call("generate", dict(args, wait_s=0))                             # wait_s doesn't make it different
        self.assertEqual(r["structuredContent"]["id"], first["id"])
        again = m.call("generate", dict(args, again=True))["structuredContent"]
        other = m.call("generate", dict(args, seed=5))["structuredContent"]
        self.assertEqual(len({first["id"], again["id"], other["id"]}), 3)
        (self.work / "f.png").write_bytes(png(42, 24) + b"\x00")                 # another file under the same name
        changed = m.call("generate", args)["structuredContent"]
        self.assertNotIn(changed["id"], {first["id"], again["id"], other["id"]})
        failing = {"prompt": "STUB_FAIL", "wait_s": 30}
        a = m.call("generate", failing)["structuredContent"]["id"]
        b = m.call("generate", failing)["structuredContent"]["id"]
        self.assertNotEqual(a, b)

    def test_out(self):
        """DECISIONS 15: out is an existing directory or a .mp4 name; an existing file is never replaced; out stays
        in LFSTUDIO_OUT_DIR or the working directory unless LFSTUDIO_MCP_OUT_ANYWHERE=1."""
        outside = self.dir / "outside"
        outside.mkdir()
        m = self.mcp()
        (self.work / "c.mp4").write_bytes(b"keep me")
        r = m.call("generate", {"prompt": "x", "wait_s": 30, "out": "c.mp4"})
        self.assertEqual(Path(r["structuredContent"]["path"]), self.work / "c-1.mp4")
        self.assertEqual((self.work / "c.mp4").read_bytes(), b"keep me")
        self.assertEqual((self.work / "c-1.jpg").read_bytes()[:2], b"\xff\xd8")
        (self.work / "sub").mkdir()
        r = m.call("fetch", {"id": r["structuredContent"]["id"], "out": "sub", "poster": False})
        self.assertEqual(Path(r["structuredContent"]["path"]).parent, self.work / "sub")
        jobs = len(self.studio.state()["jobs"])
        for out, words in [(str(outside / "x.mp4"), "out must be inside LFSTUDIO_OUT_DIR"),
                           (str(outside), "out must be inside LFSTUDIO_OUT_DIR"),
                           ("../outside/x.mp4", "out must be inside LFSTUDIO_OUT_DIR"),
                           ("notes.txt", "out must be an existing directory or a file name ending in .mp4"),
                           ("newdir/", "out must be an existing directory")]:
            r = m.call("generate", {"prompt": "x", "wait_s": 30, "out": out})
            self.assertTrue(r["isError"], out)
            self.assertIn(words, text(r))
        self.assertEqual(len(self.studio.state()["jobs"]), jobs)                 # nothing was submitted
        (self.work / "link").symlink_to(outside)
        r = m.call("generate", {"prompt": "x", "wait_s": 30, "out": "link/y.mp4"})
        self.assertTrue(r["isError"])
        m2 = self.mcp(LFSTUDIO_MCP_OUT_ANYWHERE="1", LFSTUDIO_OUT_DIR=str(self.dir / "clips"))
        r = m2.call("generate", {"prompt": "x", "wait_s": 30, "out": str(outside / "x.mp4")})
        self.assertEqual(Path(r["structuredContent"]["path"]), outside / "x.mp4")
        r = m2.call("generate", {"prompt": "y", "wait_s": 30})
        self.assertEqual(Path(r["structuredContent"]["path"]).parent, self.dir / "clips")
        self.assertEqual([p.name for p in self.work.glob("*.part")] + [p.name for p in outside.glob(".*")], [])

    def test_unwritable_places(self):
        """A host that starts the server where it can't write, without LFSTUDIO_OUT_DIR: generate refuses before it
        submits and names LFSTUDIO_OUT_DIR; so does an out it can't write to. A save that fails all the same says the
        clip is done and how to fetch it."""
        if os.geteuid() == 0:
            self.skipTest("root can write anywhere")
        ro = self.work / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        self.addCleanup(lambda: ro.exists() and ro.chmod(0o755))
        m = self.mcp(cwd=ro)
        r = m.call("generate", {"prompt": "x", "wait_s": 30})
        self.assertTrue(r["isError"])
        self.assertEqual(text(r), f"can't save clips in {ro}/lf-clips (the default; LFSTUDIO_OUT_DIR is unset): no "
                                  f"permission to write to {ro}. Nothing was submitted or downloaded. Set "
                                  "LFSTUDIO_OUT_DIR in this server's settings to a directory it can write to, or pass "
                                  f"out with a writable directory inside its working directory ({ro}).")
        for out in (str(ro), "new/c.mp4"):
            r = m.call("generate", {"prompt": "x", "wait_s": 30, "out": out})
            self.assertTrue(r["isError"], out)
            self.assertIn(f"no permission to write to {ro}. Nothing was submitted or downloaded.", text(r))
        self.assertEqual(self.studio.state()["jobs"], [])
        m = self.mcp(LFSTUDIO_OUT_DIR=str(self.dir / "clips"))
        (self.work / "late").mkdir()
        m.send({"jsonrpc": "2.0", "id": "g", "method": "tools/call",
                "params": {"name": "generate", "arguments": {"prompt": "x", "wait_s": 30, "out": "late"}}})
        deadline = time.time() + 10
        while not self.studio.state()["jobs"] and time.time() < deadline:
            time.sleep(0.05)
        jid = self.studio.state()["jobs"][0]["id"]                          # checked and submitted: now take it away
        (self.work / "late").chmod(0o555)
        self.addCleanup(lambda: (self.work / "late").exists() and (self.work / "late").chmod(0o755))
        r = m.response("g")["result"]
        self.assertTrue(r["isError"])
        self.assertIn(f'The clip is done on the studio: call fetch with id "{jid}" and an out this server can write '
                      "to", text(r))
        (self.work / "late").chmod(0o755)
        r = m.call("fetch", {"id": jid, "out": "late"})
        self.assertEqual(Path(r["structuredContent"]["path"]).parent, self.work / "late")

    def test_relative_paths(self):
        """Relative paths start from the server's working directory, also out when LFSTUDIO_OUT_DIR is set (as the docs
        say); a missing file's error names the absolute path it looked for."""
        m = self.mcp(LFSTUDIO_OUT_DIR=str(self.dir / "clips"))
        r = m.call("generate", {"prompt": "x", "first_frame": "nothere.png"})
        self.assertTrue(r["isError"])
        self.assertEqual(text(r), f"first_frame: no such file: nothere.png (looked for {self.work / 'nothere.png'})")
        r = m.call("generate", {"prompt": "x", "first_frame": self.frame(), "out": "mine.mp4", "wait_s": 30})
        self.assertEqual(Path(r["structuredContent"]["path"]), self.work / "mine.mp4")


class ApiOff(McpTest):
    """A studio with LFSTUDIO_API=0 (#58): generate, wait, status with an id and fetch are tool errors with the CLI's
    text, which names the setting; generate queues nothing. status without an id reads the page's /api/state."""
    studio_env = {"LFSTUDIO_API": "0"}

    def test_tool_errors(self):
        off = load_cli().API_OFF
        m = self.mcp()
        for args in ({"prompt": "x"}, {"prompt": "x", "wait_s": 0}, {"prompt": "x", "first_frame": self.frame()}):
            r = m.call("generate", args)
            self.assertTrue(r["isError"], text(r))
            self.assertTrue(text(r).endswith(off), text(r))
            self.assertNotIn("structuredContent", r)
        self.assertEqual(self.studio.state()["jobs"], [])                         # nothing was queued
        jid = self.studio.generate()[1]["id"]                                     # a clip made in the page
        self.studio.wait([jid], timeout=30)
        for tool, args in (("wait", {"id": jid}), ("wait", {"id": jid, "wait_s": 0}), ("status", {"id": jid}),
                           ("fetch", {"id": jid})):
            r = m.call(tool, args)
            self.assertEqual((r["isError"], r["content"]), (True, [{"type": "text", "text": off}]), tool)
        self.assertFalse((self.work / "lf-clips").exists() and any((self.work / "lf-clips").iterdir()))
        r = m.call("status")
        self.assertFalse(r["isError"], text(r))
        self.assertIn(f"{jid}  done", text(r))


class LongClips(McpTest):
    """The models load for 12 s, so a call waits longer than one progress interval (10 s)."""
    load_s, clip_s = 12, 0.5

    def test_progress(self):
        """9.3 #7: a request with a progressToken gets notifications/progress while it waits, increasing, and none
        after its response."""
        m = self.mcp()
        r = m.request("tools/call", {"name": "generate", "arguments": {"prompt": "x", "wait_s": 30},
                                     "_meta": {"progressToken": "tok"}}, rid="g")["result"]
        self.assertFalse(r["isError"], text(r))
        m.pump(1.5)
        progress = [n["params"] for n in m.notes if n.get("method") == "notifications/progress"]
        self.assertGreaterEqual(len(progress), 1)
        self.assertTrue(all(p["progressToken"] == "tok" for p in progress))
        values = [p["progress"] for p in progress]
        self.assertEqual(values, sorted(set(values)))
        self.assertGreaterEqual(values[0], 9.5)
        self.assertRegex(progress[0]["message"], "^(loading models, then generating|queued)")
        messages = [json.loads(line) for line in m.lines]
        answered = messages.index(next(x for x in messages if x.get("id") == "g"))
        self.assertFalse([x for x in messages[answered:] if x.get("method") == "notifications/progress"])

    def test_cancelled_call_keeps_the_clip(self):
        """9.3 #8: notifications/cancelled during a wait: no response for it, the clip still finishes, and the
        server keeps serving."""
        m = self.mcp()
        jid = m.call("generate", {"prompt": "x", "wait_s": 0})["structuredContent"]["id"]
        m.send({"jsonrpc": "2.0", "id": "w1", "method": "tools/call",
                "params": {"name": "wait", "arguments": {"id": jid, "wait_s": 60}}})
        time.sleep(1)
        m.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": "w1",
                                                                                  "reason": "user"}})
        self.assertEqual(m.request("ping")["result"], {"resultType": "complete"})
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        m.pump(3)
        while not m.inbox.empty():
            m.pump(0.1)
        self.assertNotIn(json.dumps("w1"), m.responses)
        self.assertFalse((self.work / "lf-clips").exists() and list((self.work / "lf-clips").iterdir()))
        r = m.call("wait", {"id": jid, "wait_s": 10})
        self.assertFalse(r["isError"])

    def test_wait_budget(self):
        """A generate whose wait_s runs out says the clip is still running, with its id, and is not an error."""
        m = self.mcp(LFSTUDIO_MCP_WAIT_S="2")
        t = time.time()
        r = m.call("generate", {"prompt": "x"})
        self.assertLess(time.time() - t, 6)
        self.assertFalse(r["isError"])
        jid = r["structuredContent"]["id"]
        self.assertRegex(text(r), rf"Clip {jid} is loading models, then generating .*, 0:0\d since it started\. "
                                  rf'Call wait with id "{jid}" to keep waiting\.')


class FakeStudio(McpTest):
    """Against fake servers: switchboard's answers (DECISIONS 13) and a submit that outlives its call (DECISIONS 14)."""
    studio_needed = False
    JID = "0123456789abcdef0123456789abcdef"

    def fake(self, answer):
        fake = Fake(answer)
        self.addCleanup(fake.stop)
        return fake

    def test_stopped_studio(self):
        """The studio is stopped: wait and fetch say to fetch with wake: true, status reports it, cancel refuses,
        and nothing but switchboard's status is asked for."""
        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "stopped", "paused": None, "message": "LF Studio is not running"}
            return switchboard_error("stopped", "LF Studio is not running")
        fake = self.fake(answer)
        m = self.mcp(url=fake.url)
        r = m.call("wait", {"id": self.JID})
        self.assertTrue(r["isError"])
        self.assertIn(f'Call fetch with id "{self.JID}" and wake: true', text(r))
        r = m.call("fetch", {"id": self.JID})
        self.assertTrue(r["isError"])
        self.assertIn("fetching would start it (it needs most of the machine's memory). Call fetch again with wake: "
                      "true", text(r))
        r = m.call("status", {"id": self.JID})
        self.assertFalse(r["isError"])
        self.assertIn("the studio is stopped", text(r))
        r = m.call("status")
        self.assertIn("The studio is stopped", text(r))
        r = m.call("cancel", {"id": self.JID})
        self.assertTrue(r["isError"])
        self.assertEqual(set(fake.paths()), {"/__switchboard/status"})

    def test_stopped_while_waiting(self):
        """DECISIONS 13: switchboard stops the studio during a wait: fetch with wake: true (most likely done)."""
        running = {"id": self.JID, "state": "running", "phase": "generating", "elapsed_s": 3}
        polls = []

        def answer(method, path, headers, body):
            if path == "/__switchboard/status":
                return 200, JSON, {"state": "ready", "paused": None}
            polls.append(path)
            return (202, JSON, running) if len(polls) == 1 else switchboard_error("stopped", "not running")
        m = self.mcp(url=self.fake(answer).url)
        r = m.call("wait", {"id": self.JID, "wait_s": 20})
        self.assertTrue(r["isError"])
        self.assertIn(f'the clip most likely finished first. Call fetch with id "{self.JID}" and wake: true', text(r))

    def test_switchboard_refuses_the_submit(self):
        """A switchboard 503 on generate: NOT submitted, call again in Retry-After s; the hold is the budget less the
        upload margin (50 - 15 s)."""
        def answer(method, path, headers, body):
            return switchboard_error("waiting", "LF Studio will start once memory is free.", retry=90)
        fake = self.fake(answer)
        r = self.mcp(url=fake.url).call("generate", {"prompt": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("will start once memory is free. The clip was NOT submitted; call generate again in about 90 s.",
                      text(r))
        method, path, headers, body = fake.requests[-1]
        self.assertEqual((method, path, headers["X-Switchboard-Wait"]), ("POST", "/api/generate", "35"))
        self.assertNotIn("Authorization", headers)

    def test_submit_outlives_a_cancelled_call(self):
        """DECISIONS 14: the POST finishes after its call was cancelled; status lists the clip, and an identical
        generate returns it rather than making another."""
        record = {"id": self.JID, "state": "queued", "phase": None, "ahead": 0, "task": "t2va", "status": "queued",
                  "created": time.time(), "orientation": "landscape", "seconds": 5, "seed": 1}

        def answer(method, path, headers, body):
            if path == "/api/generate":
                time.sleep(3)
                return 200, JSON, {"id": self.JID, "job": record}
            if path == "/api/state":
                return 200, JSON, {"now": time.time(), "jobs": [record]}
            if path.startswith(f"/api/jobs/{self.JID}/video"):
                return 202, JSON, record
            if path == f"/api/jobs/{self.JID}":
                return 200, JSON, record
            return 404, JSON, {"error": "not found"}
        fake = self.fake(answer)
        m = self.mcp(url=fake.url)
        m.send({"jsonrpc": "2.0", "id": "g1", "method": "tools/call",
                "params": {"name": "generate", "arguments": {"prompt": "x"}}})
        time.sleep(0.5)
        r = m.call("status")
        self.assertIn("(being submitted, 0:0", text(r))
        m.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": "g1"}})
        r = m.call("generate", {"prompt": "x", "wait_s": 0})
        self.assertIn("An identical generate call is still submitting its clip", text(r))
        time.sleep(3.5)
        r = m.call("status")
        self.assertTrue(text(r).startswith(f"Submitted from this session (oldest first):\n{self.JID}  queued"), text(r))
        self.assertEqual(r["structuredContent"]["session"][0]["id"], self.JID)
        r = m.call("generate", {"prompt": "x", "wait_s": 0})
        self.assertIn(f"Already submitted as {self.JID}", text(r))
        self.assertEqual([p for p in fake.paths() if p == "/api/generate"], ["/api/generate"])
        self.assertNotIn(json.dumps("g1"), m.responses)

    def test_identical_generate_while_the_studio_is_unavailable(self):
        """DECISIONS 14: only the studio's 404 for the earlier clip (deleted) makes an identical generate submit
        again. While the studio can't say (it is starting, stopped, unreachable or in trouble), the earlier clip is
        kept, and the record is read with X-Switchboard-Wait: 0, so switchboard doesn't hold it."""
        record = {"id": self.JID, "state": "queued", "phase": None, "ahead": 0, "task": "t2va", "status": "queued",
                  "orientation": "landscape", "seconds": 5, "seed": 1}
        read = {"answer": None}

        def answer(method, path, headers, body):
            if path == "/api/generate":
                return 200, JSON, {"id": self.JID, "job": record}
            if path == f"/api/jobs/{self.JID}":
                return read["answer"]
            if path.startswith(f"/api/jobs/{self.JID}/video"):
                return 202, JSON, record
            return 404, JSON, {"error": "not found"}
        fake = self.fake(answer)
        m = self.mcp(url=fake.url)
        args = {"prompt": "x", "seed": 1, "wait_s": 0}
        self.assertFalse(m.call("generate", args)["isError"])
        for trouble in (None,                                                   # no answer (a held read timing out)
                        (500, JSON, {"error": "Internal error"}),
                        switchboard_error("loading", "LF Studio is starting")):
            read["answer"] = trouble
            r = m.call("generate", args)
            self.assertIn(f"Already submitted as {self.JID}", text(r), trouble)
        self.assertEqual(fake.paths().count("/api/generate"), 1)
        reads = [h for _, p, h, _ in fake.requests if p == f"/api/jobs/{self.JID}"]
        self.assertEqual([h.get("X-Switchboard-Wait") for h in reads], ["0"] * 3)
        read["answer"] = (404, JSON, {"error": "no such clip"})                  # deleted: made again
        r = m.call("generate", args)
        self.assertNotIn("Already submitted", text(r))
        self.assertEqual(fake.paths().count("/api/generate"), 2)


if __name__ == "__main__":
    unittest.main()
