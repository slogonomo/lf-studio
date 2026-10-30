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
    models = None                  # LFSTUDIO_MODELS (the stub studio's default: sol-h3 alone, the model choice off)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lf-studio-mcp-"))
        self.work = self.dir / "work"
        self.work.mkdir()
        self.studio = None
        if self.studio_needed:
            self.studio = StubStudio(self.dir / "studio", port=free_port(), load_s=self.load_s, clip_s=self.clip_s,
                                     preload=False, env=self.studio_env, models=self.models).start()
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
        self.assertEqual(len(r["tools"]), 8)
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
        """9.3 #4: eight flat tools Claude Code accepts (upscale is the experimental Upscale's, models lists the
        studio's models), with short enough descriptions."""
        r = self.m.request("tools/list")["result"]
        self.assertEqual([t["name"] for t in r["tools"]], ["generate", "wait", "status", "cancel", "fetch",
                                                            "upscale", "models", "prompt_guide"])
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
                         {"type": "integer", "enum": [5, 6, 7, 8, 10, 12, 15], "default": 5})   # 3 and 4 s went after 0.5.0
        upscale = r["tools"][5]["description"]
        for words in ("Experimental", "Clips of every length, 16:9 or 9:16", "of every mode and model",
                      "1080p takes every length, 2K up to 10 s, 4K 5 s clips; another size is refused with the reason",
                      "Keyframes or Reference clip", "stretched to 768p", "keeps all its frames (a 5 s one at 4K its "
                      "first 121)", "pin (Keep my frames, experimental)", "at 1080p and 2K with balanced",
                      "2K is 1440x2560", "redraws faces", "longer clips take longer (1080p 24 min at 10 s",
                      "closes the studio's models", "4K has run once, with a 16:9 clip; not yet with a 9:16 one"):
            self.assertIn(words, upscale)
        self.assertNotIn("5 s clips, 16:9", upscale)
        self.assertEqual(r["tools"][5]["inputSchema"]["properties"]["resolution"],
                         {"type": "string", "enum": ["1080p", "2k", "4k"], "default": "2k",
                          "description": "Default 2k; a 12 or 15 s clip needs 1080p"})
        self.assertNotIn("untested", upscale)
        self.assertNotIn("draft", upscale)                                       # from the clip's MP4 only
        self.assertEqual(set(r["tools"][5]["inputSchema"]["properties"]),
                         {"id", "resolution", "strength", "audio", "seed", "pin", "wait_s", "out", "poster"})
        self.assertEqual(r["tools"][5]["inputSchema"]["properties"]["pin"]["enum"], ["auto", "off"])
        generate = r["tools"][0]["description"]
        self.assertIn("stretched or cropped", generate)                          # DECISIONS 12
        self.assertIn("change the seed", generate)                               # DECISIONS 14
        self.assertIn("never removes a clip", generate)
        self.assertIn("model: sol-h3 (Sol-H3, the default) or another model the studio lists (the models tool",
                      generate)
        props = r["tools"][0]["inputSchema"]["properties"]
        # no enum for the model or its options: a model added to the studio later needs no new client
        self.assertEqual(props["model"]["type"], "string")
        self.assertEqual((props["resolution"]["type"], props["steps"]["type"], props["options"]["type"]),
                         ("string", "integer", "object"))
        for name in ("model", "resolution", "steps", "options"):
            self.assertNotIn("enum", props[name], name)
        models = r["tools"][6]
        self.assertEqual((models["annotations"], models["inputSchema"]["properties"]),
                         ({"readOnlyHint": True}, {}))
        self.assertIn("Never starts a stopped studio", models["description"])

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


class Upscale(McpTest):
    """The upscale tool (the experimental Upscale: the stub studio has it on, with a stand-in job of about 1 s)."""

    def clip(self, **kw):
        jid = self.studio.generate(**kw)[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        return jid

    def test_upscale_and_save(self):
        jid = self.clip(seed=9)
        m = self.mcp()
        r = m.call("upscale", {"id": jid, "resolution": "4k", "strength": "full", "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual([c["type"] for c in r["content"]], ["text", "resource_link", "image"])
        rec = r["structuredContent"]
        uid, path = rec["id"], Path(rec["path"])
        self.assertEqual((rec["kind"], rec["parent"], rec["state"], rec["resolution"], rec["strength"]),
                         ("upscale", jid, "done", "4k", "full"))
        self.assertEqual(path, self.work / "lf-clips" / f"lf-studio_t2va_5s_seed9_{jid[:6]}_4k_{uid[:6]}.mp4")
        self.assertTrue(path.is_file())
        self.assertTrue(text(r).startswith(f"Queued upscale {uid} of clip {jid} (4k, full). Upscale "
                                           f"{uid} is done: saved {path} ("), text(r))
        self.assertIn(f"4k of clip {jid}, full).", text(r))
        r = m.call("upscale", {"id": jid, "wait_s": 0})                         # the studio's defaults, no wait
        self.assertFalse(r["isError"], text(r))
        rec = r["structuredContent"]
        self.assertEqual((rec["state"], rec["resolution"], rec["strength"], rec["audio"]),
                         ("queued", "2k", "balanced", "clip"))
        self.assertIn(f'Upscale {rec["id"]} is queued, next in line. Call wait with id "{rec["id"]}"', text(r))
        r = m.call("wait", {"id": rec["id"], "wait_s": 30, "poster": False})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(Path(r["structuredContent"]["path"]).name,
                         f"lf-studio_t2va_5s_seed9_{jid[:6]}_2k_{rec['id'][:6]}.mp4")

    def test_keep_my_frames(self):
        """pin: off sends no end; auto (the default) the clip's first frame (Stage 2 held it); the text says what
        became of it; another value is refused before anything is sent."""
        line = "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced."
        jid = self.studio.generate(task="fl2va", prompt=line + "\nA red kite rises.", seed=4,
                                   files={"first_frame": [("f.png", png())]})[1]["id"]
        self.assertEqual(self.studio.wait([jid], timeout=60)[0]["status"], "done")
        m = self.mcp()
        r = m.call("upscale", {"id": jid, "resolution": "1080p", "pin": "off", "wait_s": 0})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(r["structuredContent"]["pin"], {"asked": "off", "ends": [], "why": {}})
        r = m.call("upscale", {"id": jid, "resolution": "1080p", "wait_s": 60, "poster": False})
        self.assertFalse(r["isError"], text(r))
        rec = r["structuredContent"]
        self.assertEqual((rec["state"], rec["pin"]["ends"], rec["pin"]["result"]["first"]["used"]),
                         ("done", ["first"], True))
        self.assertIn(f"1080p of clip {jid}, balanced, first frame kept).", text(r))
        r = m.call("status", {"id": rec["id"]})
        self.assertIn("1080p upscale of", text(r))
        self.assertIn("first frame kept", text(r))
        r = m.call("upscale", {"id": jid, "pin": "on"})
        self.assertTrue(r["isError"])
        self.assertIn("pin must be one of auto, off", text(r))

    def test_errors(self):
        ten = self.clip(seconds=10)
        m = self.mcp()
        for args, words in (({"id": ten, "resolution": "4k"}, "4K for 10 s clips comes later: 1080p and 2K take them"),
                            ({"id": "f" * 32}, "no such clip"), ({"id": "nope"}, "not a clip id"),
                            ({"id": ten, "resolution": "8k"}, "resolution must be one of 1080p, 2k, 4k"),
                            ({"id": ten, "seed": -1}, "seed must be 0 or more"), ({}, "id is required"),
                            ({"id": ten, "steps": 3}, 'unknown argument "steps"'),
                            ({"id": ten, "source": "draft"}, 'unknown argument "source"')):
            r = m.call("upscale", args)
            self.assertTrue(r["isError"], args)
            self.assertIn(words, text(r))
        self.assertEqual([j.get("kind") for j in self.studio.state()["jobs"]], [None])       # nothing queued


class UpscaleCancel(McpTest):
    """cancel on an upscale says so, queued or running (its job stops within about 20 s, not the clip's ~40 s)."""
    load_s, clip_s = 1, 4
    studio_env = {"STUB_UPSCALE_S": "30"}

    def test_queued_and_running(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid], timeout=60)
        m = self.mcp()
        busy = self.studio.generate(seed=2)[1]["id"]                         # the upscale waits for it
        uid = m.call("upscale", {"id": jid, "wait_s": 0})["structuredContent"]["id"]
        r = m.call("cancel", {"id": uid})
        self.assertEqual((r["isError"], text(r).split("\n\n")[0]),
                         (False, f"Upscale {uid} was cancelled (it had not started)."))
        self.studio.wait([busy], timeout=60)
        uid = m.call("upscale", {"id": jid, "wait_s": 0})["structuredContent"]["id"]
        self.studio.wait_for(lambda st: any(j["id"] == uid and j["status"] == "upscaling" for j in st["jobs"]), 30)
        r = m.call("cancel", {"id": uid})
        self.assertEqual((r["isError"], text(r).split("\n\n")[0]),
                         (False, f"Upscale {uid} is being cancelled: its job stops within about 20 s."))
        self.assertEqual(self.studio.wait([uid], timeout=30)[0]["status"], "cancelled")


class UpscaleOff(McpTest):
    studio_env = {"LFSTUDIO_UPSCALE": "0"}

    def test_off(self):
        jid = self.studio.generate()[1]["id"]
        self.studio.wait([jid], timeout=60)
        r = self.mcp().call("upscale", {"id": jid})
        self.assertEqual((r["isError"], text(r)), (True, "the experimental Upscale is off on this studio "
                                                         "(LFSTUDIO_UPSCALE=1 turns it on)"))


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


TURBO = "lightx2v-turbo"


class Models(McpTest):
    """The model choice on (LFSTUDIO_MODELS=sol-h3,lightx2v-turbo): the models tool, generate with a model and its
    options (checked against the studio's table before anything is sent), and the repeat key with the model."""
    models = "sol-h3,lightx2v-turbo"

    def test_models_tool(self):
        r = self.mcp().call("models")
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(r["structuredContent"], {"model_choice": True, "models": self.studio.state()["models"]})
        lines = text(r).split("\n\n")[0].splitlines()
        self.assertTrue(lines[0].startswith("sol-h3: Sol-H3 (the default), "), lines)
        self.assertIn("  not yet: ref2va: LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not "
                      "Reference (ref2va): use Sol-H3", lines)
        self.assertNotIn("  not yet: 15 s", "\n".join(lines))
        self.assertIn("  makes t2va or fl2va (fl2va keyframes: first, last or first+last), 5, 6, 7, 8, 10, 12 or 15 s, "
                      "landscape or portrait, resolution 576p or 768p, steps 4", lines)
        self.assertIn("  time per clip, resolution 576p, steps 4: 5 s about 3:20, 6 s about 4:10, 7 s about 4:40, 8 s "
                      "about 5:10, 10 s about 6:40, 12 s about 8:20, 15 s about 11:00 (estimates until one is measured "
                      "here)", lines)
        self.assertIn("  option resolution: 576p (the default; 1024x576, 9:16 576x1024), 768p (1344x768, 9:16 "
                      "768x1344)", lines)

    def test_generate_with_a_model(self):
        """model and resolution (or options) make a LightX2V Turbo clip; the text says so; the MP4 is saved under
        the studio's name for it."""
        m = self.mcp()
        r = m.call("generate", {"prompt": "x", "seed": 2, "model": TURBO, "options": {"resolution": "768p"},
                                "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        rec = r["structuredContent"]
        self.assertEqual((rec["model"], rec["model_opts"]["resolution"]), (TURBO, "768p"))
        self.assertIn(f"seed 2, {TURBO} 768p 4 steps).", text(r))
        self.assertEqual(Path(rec["path"]).name, f"lf-studio_{TURBO}-768p_t2va_5s_seed2_{rec['id'][:6]}.mp4")
        r = m.call("generate", {"prompt": "x", "seed": 2, "model": TURBO, "resolution": "576p", "steps": 4,
                                "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(r["structuredContent"]["model_opts"]["resolution"], "576p")

    def test_the_repeat_key_has_the_model(self):
        """An identical call returns the earlier clip; the same prompt and seed with another model, or other
        options, makes a clip of its own."""
        m = self.mcp()
        args = {"prompt": "x", "seed": 4, "wait_s": 0}
        sol = m.call("generate", args)["structuredContent"]["id"]
        turbo = m.call("generate", dict(args, model=TURBO))["structuredContent"]["id"]
        big = m.call("generate", dict(args, model=TURBO, resolution="768p"))["structuredContent"]["id"]
        self.assertEqual(len({sol, turbo, big}), 3)
        r = m.call("generate", dict(args, model=TURBO))
        self.assertIn(f"Already submitted as {turbo}", text(r))
        r = m.call("generate", dict(args, model=TURBO, options={"resolution": "768p"}))   # the same as resolution
        self.assertIn(f"Already submitted as {big}", text(r))
        self.assertEqual(len(self.studio.state()["jobs"]), 3)

    def test_a_last_frame_alone_at_8_s(self):
        """What the GPU validation cleared: a last frame alone, and lengths past 5 s, as LightX2V Turbo clips."""
        r = self.mcp().call("generate", {"prompt": "x", "model": TURBO, "seconds": 8, "last_frame": self.frame(),
                                         "wait_s": 30})
        self.assertFalse(r["isError"], text(r))
        rec = r["structuredContent"]
        self.assertEqual((rec["model"], rec["task"], rec["seconds"], rec["state"]), (TURBO, "fl2va", 8, "done"))

    def test_refusals(self):
        """What the model doesn't make yet, an unknown model or option, a value it lacks, a core field among the
        options: a tool error with the reason, and nothing reaches the studio."""
        m = self.mcp()
        for args, words in [
                ({"model": TURBO, "references": [self.frame()]},
                 "LightX2V Turbo makes Text (t2va) and Keyframes (fl2va) clips for now, not Reference (ref2va): use "
                 "Sol-H3. Nothing was submitted."),
                ({"model": TURBO, "resolution": "4k"}, "resolution must be 576p or 768p for lightx2v-turbo"),
                ({"model": TURBO, "options": {"steps": "8"}}, "steps must be 4 for lightx2v-turbo"),
                ({"model": TURBO, "options": {"seconds": 5}}, "options seconds: seconds is a field of every clip"),
                ({"model": TURBO, "options": {"resolution": "768p"}, "resolution": "576p"},
                 "resolution: resolution is given twice"),
                ({"model": TURBO, "options": {"resolution": ["768p"]}}, "options must be an object of names and "
                                                                       "values (strings or integers)"),
                ({"model": "nope"}, "this studio has no model nope; it has sol-h3 (Sol-H3), lightx2v-turbo "
                                    "(LightX2V Turbo)"),
                ({"model": "Turbo"}, "model must be a model id such as sol-h3 or lightx2v-turbo (the models tool"),
                ({"steps": 4}, "steps is not an option of sol-h3; it takes none")]:
            r = m.call("generate", dict({"prompt": "x"}, **args))
            self.assertTrue(r["isError"], args)
            self.assertIn(words, text(r), args)
        self.assertEqual(self.studio.state()["jobs"], [])


class ModelsOff(McpTest):
    """The model choice off (the default): the models tool says every clip is Sol-H3; another model or an option is
    refused before anything is sent."""

    def test_off(self):
        m = self.mcp()
        r = m.call("models")
        self.assertFalse(r["isError"])
        self.assertTrue(text(r).startswith("This studio has no model choice: every clip is Sol-H3 (sol-h3)"))
        self.assertEqual(r["structuredContent"], {"model_choice": False, "models": []})
        r = m.call("generate", {"prompt": "x", "model": TURBO})
        self.assertTrue(r["isError"])
        self.assertIn(f"so it can't make a {TURBO} clip; leave model and the options out", text(r))
        self.assertEqual(self.studio.state()["jobs"], [])


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

    def test_pin_for_an_older_studio(self):
        """An older studio (no "upscale_pin" in /api/state) never keeps the frames: the upscale goes without pin."""
        posts = []

        def answer(method, path, headers, body):
            if path == "/api/state":
                return 200, JSON, {"now": time.time(), "jobs": [], "upscale": True}
            if path == "/api/upscale":
                posts.append(json.loads(body))
                return 200, JSON, {"id": "f" * 32, "job": {"id": "f" * 32, "kind": "upscale", "state": "queued"}}
            return 404, JSON, {"error": "not found"}
        r = self.mcp(url=self.fake(answer).url).call("upscale", {"id": self.JID, "pin": "auto", "wait_s": 0})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(posts, [{"id": self.JID}])

    def test_pin_for_a_stopped_studio(self):
        """switchboard has the studio stopped, so /api/state can't say whether it takes pin without starting it: the
        upscale goes with pin, and again without it when the studio answers that it does not know the key (an older
        studio); another refusal is not sent again."""
        unknown = 'unknown key "pin" (known: id, resolution, strength, audio, seed)'
        for name, answers, bodies in (("takes it", [None], [{"pin": "auto"}]),
                                      ("older", [unknown, None], [{"pin": "auto"}, {}]),
                                      ("refused", ["no such clip"], [{"pin": "auto"}])):
            with self.subTest(name):
                posts = []

                def answer(method, path, headers, body):
                    if path == "/api/state":
                        return switchboard_error("stopped", "LF Studio is not running")
                    if path == "/api/upscale":
                        posts.append(json.loads(body))
                        if answers[len(posts) - 1]:
                            return 400, JSON, {"error": answers[len(posts) - 1]}
                        return 200, JSON, {"id": "f" * 32, "job": {"id": "f" * 32, "kind": "upscale",
                                                                   "state": "queued"}}
                    return 404, JSON, {"error": "not found"}
                r = self.mcp(url=self.fake(answer).url).call("upscale", {"id": self.JID, "pin": "auto", "wait_s": 0})
                self.assertEqual(r["isError"], name == "refused", text(r))
                self.assertEqual(posts, [dict({"id": self.JID}, **b) for b in bodies])
                if name == "refused":
                    self.assertIn("no such clip", text(r))

    def test_a_model_on_a_stopped_studio(self):
        """switchboard has the studio stopped, so its models can't be read without starting it: the submit goes
        (and starts it), and an answer without the model (the studio has no model choice) is cancelled at once."""
        def answer(method, path, headers, body):
            if path == "/api/state":
                return switchboard_error("stopped", "LF Studio is not running")
            if path == "/api/generate":
                return 200, JSON, {"id": self.JID, "job": {"id": self.JID, "state": "queued", "task": "t2va"}}
            if path == "/api/cancel":
                return 200, JSON, {"ok": True, "job": {"id": self.JID, "status": "cancelled"}}
            return 404, JSON, {"error": "not found"}
        fake = self.fake(answer)
        r = self.mcp(url=fake.url).call("generate", {"prompt": "x", "model": TURBO, "wait_s": 20})
        self.assertTrue(r["isError"])
        self.assertIn(f"so it can't make a {TURBO} clip; leave model and the options out", text(r))
        self.assertIn(f"the clip it queued instead ({self.JID}, a Sol-H3 clip) is cancelled", text(r))
        self.assertEqual([(m, p) for m, p, _, _ in fake.requests],
                         [("GET", "/api/state"), ("POST", "/api/generate"), ("POST", "/api/cancel")])

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
