"""bench/h3turbo_encode.py (LightX2V Turbo's prompt encode: a clip's prompt conditioning from a transient Qwen worker,
which the server's encode_prompt runs) and its wrapper bench/h3turbo-encode.sh, on the CPU: text_only_prompt on every
form of the keyframe alignment line the studio writes (the prompt guide's examples, read from bin/lf-cli, and the page's
alignLine() for each set of keyframes, read from webui/index.html), on the inline references, on lines that only start
like an alignment line and on dialogue (never changed); ref_text_only_prompt, a Reference prompt as t2va's three fields
for the Upscale (the guide's and the page's examples, the summary form, free text, t2va's own fields, a prompt with
nothing left); --upscale (the Upscale's encode: text alone, plain tokens allowed); the prompt as the server takes it (stripped, its length limit);
the arguments and the environment it refuses; --check; and the encodes against a stand-in of the patch's runtime package
(its Worker records the recipe it got and the calls, and writes conditioning files as the Qwen worker does): the canvas
per job, the clip's encode and the text-only one, the receipt checks, the worker closed in every case. Standard library
only. (The server's tests run it through its own encode() too: TurboEncodeSeam in tests/test_server.py;
tests/stub/h3turbo_encode.py stands in for it in the stub studio.)
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bench" / "h3turbo_encode.py"
WRAPPER = REPO / "bench" / "h3turbo-encode.sh"


def load_helper():
    spec = importlib.util.spec_from_file_location("h3turbo_encode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper()


def guide_examples():
    """The prompt guide's fl2va examples (bin/lf-cli GUIDE, the same as docs/API.md's): {title: prompt}."""
    text = (REPO / "bin" / "lf-cli").read_text()
    found = dict(re.findall(r"\n(fl2va, [^\n]*):\n\n```text\n(.*?)\n```", text, re.S))
    assert len(found) == 3, sorted(found)
    return found


def page_lines():
    """The page's keyframe lines per set of keyframes (webui/index.html's ALIGN, which alignLine() fills in)."""
    page = (REPO / "webui" / "index.html").read_text()
    lines = json.loads(re.search(r"const ALIGN=(\{.*?\});\n", page, re.S).group(1))
    assert set(lines) == {"first", "last", "first+last"}, lines
    assert "const alignLine=frames=>(ALIGN[frames]||ALIGN['first+last']).replace('S.SS'," in page
    return lines


def page_example(seconds=5, frames="first+last"):
    """The page's fl2va example with its alignLine() for `seconds` and the keyframes attached (webui/index.html's
    EX.fl2va and alignLine; with none attached yet, the first + last line)."""
    page = (REPO / "webui" / "index.html").read_text()
    line = page_lines()[frames].replace("S.SS", f"{seconds:.2f}")
    example = re.search(r'fl2va:"(ALIGN\\n.*?)",\n', page).group(1)
    return json.loads(f'"{example}"').replace("ALIGN", line)


class TextOnlyTests(unittest.TestCase):
    def test_the_guide_first_frame_example(self):
        prompt = guide_examples()["fl2va, first frame only, 5 s"]
        self.assertTrue(prompt.startswith("For the target video, at 0.00 seconds into the target video, <Picture 1> "
                                          "(from [Shot 1]) is fully referenced.\n\n"))
        result = helper.text_only_prompt(prompt)
        self.assertEqual(result["dropped"], ["alignment line", "1 x ' shown in <Picture N>'"])
        text = result["prompt"]
        self.assertTrue(text.startswith("integrated_multimodal_description: [Shot 1] Live-action, cinematic, the scene "
                                        "opens on the kitchen, keeping the woman's appearance,"), text[:200])
        self.assertNotIn("Picture", text)
        self.assertIn("\n\noverall_soundscape: Water pours", text)
        self.assertTrue(text.endswith("non_diegetic_music: N/A"))

    def test_the_guide_first_and_last_example(self):
        """The em-dash line, and the body's bare "Picture N" (the guide writes them without angle brackets)."""
        prompt = guide_examples()["fl2va, first and last frame, 8 s"]
        self.assertIn("align with the target video — Picture 1 (from Shot 1)", prompt)
        result = helper.text_only_prompt(prompt)
        self.assertEqual(result["dropped"], ["alignment line", "2 x ' in/of/from <Picture N>'"])
        self.assertNotIn("Picture", result["prompt"])
        self.assertIn("a wide shot begins in the composition, with the small sailboat", result["prompt"])
        self.assertIn("settles into the position, sail shape and framing at the end of the shot.", result["prompt"])
        self.assertTrue(result["prompt"].startswith("integrated_multimodal_description:"))

    def test_the_guide_last_frame_example(self):
        result = helper.text_only_prompt(guide_examples()["fl2va, last frame only, 6 s"])
        self.assertEqual(result["dropped"], ["alignment line", "1 x ' in/of/from <Picture N>'"])
        self.assertIn("the table settles into the exact arrangement, lighting and framing.\n", result["prompt"])
        self.assertNotIn("Picture", result["prompt"])

    def test_the_pages_alignment_line(self):
        """alignLine() with both keyframes: a hyphen, no angle brackets, "-second mark" without "of the target video";
        one newline. With one keyframe it writes the guide's own line for it (the forms the GPU validation ran: a
        first frame alone fully referenced at 0 s, a last frame alone as <Picture 1> at the length)."""
        guide = guide_examples()
        for seconds in (5, 8, 15):
            prompt = page_example(seconds)
            self.assertTrue(prompt.startswith("How the reference pictures align with the target video - Picture 1 "
                                              f"aligns with the 0.00-second mark; Picture 2 aligns with the "
                                              f"{seconds:.2f}-second mark.\nintegrated_multimodal_description:"))
            for frames in ("first+last", "first", "last"):
                prompt = page_example(seconds, frames)
                result = helper.text_only_prompt(prompt)
                self.assertEqual(result, {"prompt": prompt.split("\n", 1)[1], "dropped": ["alignment line"],
                                          "notes": []}, frames)
        self.assertEqual(page_example(5, "first").split("\n")[0],
                         guide["fl2va, first frame only, 5 s"].split("\n")[0])
        self.assertEqual(page_example(6, "last").split("\n")[0], guide["fl2va, last frame only, 6 s"].split("\n")[0])

    def test_alignment_lines_in_other_places(self):
        """In front of the text on the same line (dialogue after it unchanged), after a blank line, with Windows line
        ends, or in a form of its own (a line that starts as one and names a picture or a time): the line goes, the
        rest stays as it was."""
        first = guide_examples()["fl2va, first frame only, 5 s"].split("\n", 1)[0]
        same_line = helper.text_only_prompt(first + " integrated_multimodal_description: [Shot 1] A lake.")
        self.assertEqual(same_line, {"prompt": "integrated_multimodal_description: [Shot 1] A lake.",
                                     "dropped": ["alignment line"], "notes": []})
        spoken = helper.text_only_prompt(first + " She says <d>[English] Hi,  you.</d>")
        self.assertEqual(spoken["prompt"], "She says <d>[English] Hi,  you.</d>")
        crlf = helper.text_only_prompt(first + "\r\n\r\nintegrated_multimodal_description: A lake.\r\n"
                                       "overall_soundscape: Wind.\r\n")
        self.assertEqual(crlf["prompt"], "integrated_multimodal_description: A lake.\r\noverall_soundscape: Wind.")
        later = helper.text_only_prompt("integrated_multimodal_description: A lake.\n\n" + first + "\n\nmore")
        self.assertEqual(later["prompt"], "integrated_multimodal_description: A lake.\n\nmore")
        for line in ("How the reference pictures align with the target video: Picture 1 at the start, Picture 2 at "
                     "the end.", "For the target video, at 0.00 seconds into the target video, the first image is "
                     "fully referenced.", "How the reference pictures align - the image aligns with the 5-second "
                     "mark."):
            own = helper.text_only_prompt(line + "\nintegrated: A lake.")
            self.assertEqual(own, {"prompt": "integrated: A lake.", "dropped": ["alignment line"], "notes": []})

    def test_lines_that_only_start_like_an_alignment_line_stay(self):
        """A line that starts with an alignment line's opening but names no picture or time is text: kept, with a
        note."""
        for prompt in ("integrated_multimodal_description: A lake at dawn.\nFor the target video, the mood is calm.",
                       "For the target video, keep the camera locked off and the light soft.\nintegrated: A lake.",
                       "How the reference pictures align: first at the start\nintegrated: A lake."):
            result = helper.text_only_prompt(prompt)
            self.assertEqual((result["prompt"], result["dropped"]), (prompt, []))
            self.assertEqual(len(result["notes"]), 1)
            self.assertTrue(result["notes"][0].startswith("kept a line that starts like an alignment line: '"),
                            result["notes"])

    def test_inline_references(self):
        """The research probe's rewrites, in its order, then any <Picture N> left is dropped; lines without one are kept
        byte for byte, MiniMax's tokens too."""
        result = helper.text_only_prompt(
            "integrated_multimodal_description: [Shot 1] The kitchen shown in <Picture 1> stays, with the "
            "composition of <Picture 1>; the woman from <Picture 1> walks to <Picture 2> , then on.\n"
            "overall_soundscape:  She says: <d>[English] Hello.</d>  ")
        self.assertEqual(result["prompt"], "integrated_multimodal_description: [Shot 1] The kitchen stays, with this "
                                           "composition; the woman walks to, then on.\n"
                                           "overall_soundscape:  She says: <d>[English] Hello.</d>")
        self.assertEqual(result["dropped"], ["1 x ' shown in <Picture N>'",
                                             "1 x 'with the composition of <Picture N> -> with this composition'",
                                             "1 x ' in/of/from <Picture N>'", "1 x '<Picture N>'"])
        plain = "integrated_multimodal_description: a picture 1 meter wide.\n\nnon_diegetic_music: N/A"
        self.assertEqual(helper.text_only_prompt(plain), {"prompt": plain, "dropped": [], "notes": []})

    def test_the_guides_suffix_and_empty_parentheses(self):
        result = helper.text_only_prompt("the woman in <Picture 1> (from [Shot 1]) walks; the dog of Picture 2 (from "
                                         "Shot 2) runs; a man (<Picture 3>) waits, then <Picture 1> (from [Shot 1]).")
        self.assertEqual(result, {"prompt": "the woman walks; the dog runs; a man waits, then.",
                                  "dropped": ["2 x ' in/of/from <Picture N>'", "2 x '<Picture N>'"], "notes": []})

    def test_dialogue_is_never_changed(self):
        """<d>...</d> is what the model speaks: no rule touches it, also over several lines; a bare "Picture N" outside
        the inline rules stays too, and the notes count what stayed."""
        result = helper.text_only_prompt("<d>Look at Picture 1, it is us.</d> The man points at <Picture 2>.")
        self.assertEqual((result["prompt"], result["dropped"]),
                         ("<d>Look at Picture 1, it is us.</d> The man points at.", ["1 x '<Picture N>'"]))
        self.assertEqual(result["notes"], ["1 x 'Picture N' left as written (in dialogue, or outside the references "
                                           "the rules know)"])
        result = helper.text_only_prompt("She says <d>See <Picture 1>  now,\nin <Picture 2> .</d>  The dog of "
                                         "<Picture 3> runs. Picture 4 is a lake.")
        self.assertEqual(result["prompt"], "She says <d>See <Picture 1>  now,\nin <Picture 2> .</d> The dog runs. "
                                           "Picture 4 is a lake.")
        self.assertEqual(result["dropped"], ["1 x ' in/of/from <Picture N>'"])
        self.assertIn("3 x 'Picture N' left as written", result["notes"][0])
        result = helper.text_only_prompt("The dog of <Picture 1> barks: <d>Hey,  \n  you there </d> it ends.")
        self.assertEqual(result["prompt"], "The dog barks: <d>Hey,  \n  you there </d> it ends.")

    def test_nothing_left(self):
        first = guide_examples()["fl2va, first frame only, 5 s"].split("\n", 1)[0]
        for prompt in (first, first + "\n\n", "<Picture 1>", "  \n"):
            with self.assertRaisesRegex(ValueError, "nothing is left"):
                helper.text_only_prompt(prompt)


def guide_reference_example():
    """The prompt guide's ref2va example (bin/lf-cli GUIDE, the same as docs/API.md's)."""
    text = (REPO / "bin" / "lf-cli").read_text()
    found = re.findall(r"\n(ref2va, [^\n]*):\n\n```text\n(.*?)\n```", text, re.S)
    assert len(found) == 1, found
    return found[0][1]


def page_reference_example():
    """The page's ref2va example (webui/index.html's EX.ref2va)."""
    page = (REPO / "webui" / "index.html").read_text()
    return json.loads('"' + re.search(r'ref2va:"(subject_definitions:.*?)"\}', page).group(1) + '"')


LABEL = re.compile(r"<(?:Picture|Video|Audio|Subject) \d+>")


def dialogue(text):
    return re.findall(r"<d>.*?</d>", text, re.S)


class RefTextOnlyTests(unittest.TestCase):
    """ref_text_only_prompt: a Reference prompt as t2va's three fields, for the Upscale's text-only encode."""

    def test_the_guides_example(self):
        prompt = guide_reference_example()
        result = helper.ref_text_only_prompt(prompt)
        text = result["prompt"]
        self.assertEqual(result["form"], "detailed")
        self.assertEqual([line.split(":")[0] for line in text.split("\n\n")],
                         ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"])
        self.assertIsNone(LABEL.search(text), text)
        self.assertEqual(dialogue(text), dialogue(prompt))                  # what the model speaks, unchanged
        self.assertIn("frames the woman (S1), the woman with short red hair", text)
        self.assertIn("says, <d>[English] Welcome back.", text)            # "in the ... voice referenced from" goes
        for gone in ("subject_definitions", "retention_analysis", "summary", "[reference generation", "fully_preserved",
                     "voice-timbre reference"):
            self.assertNotIn(gone, text)
        self.assertTrue(text.startswith("integrated_multimodal_description: The target video is a live-action"))
        self.assertTrue(text.endswith("overall_soundscape: Soft wind and the hush of falling snow, with distant "
                                      "traffic on wet pavement.\n\nnon_diegetic_music: N/A"))
        self.assertEqual(result["dropped"], ["subject_definitions (1 subject)", "retention_analysis", "summary",
                                             "1 x '<Subject N>' (as its definition)",
                                             "1 x ' in the voice (timbre) of <Audio N>'"])
        self.assertEqual(result["notes"], [])
        self.assertLess(len(text), len(prompt))

    def test_the_pages_example(self):
        """The sections on one line each, two definitions in one line, the subject opening a sentence."""
        result = helper.ref_text_only_prompt(page_reference_example())
        self.assertEqual(result["prompt"], "integrated_multimodal_description: [Shot 1] A static medium shot. The "
                                           "person faces the camera and says, <d>[English] Welcome to the garden.</d>"
                                           "\n\noverall_soundscape: Quiet outdoor ambience.\n\nnon_diegetic_music: N/A")

    def test_one_line_definitions_and_the_summary(self):
        """Without detailed_description the summary is the description: its task tags go, and each subject is its
        whole definition at its first mention, its short form after it."""
        prompt = ("subject_definitions: <Subject 1> is the man in <Picture 1>, wearing a grey suit. <Subject 2> is the "
                  "dog in <Video 1>.\nsummary: [reference generation + audio reference] <Subject 1> walks <Subject 2> "
                  "through a park and talks in the voice timbre of <Audio 1>. <Subject 1> laughs.\n"
                  "retention_analysis: <Subject 1>: fully_preserved.")
        result = helper.ref_text_only_prompt(prompt)
        self.assertEqual(result["form"], "summary")
        self.assertEqual(result["prompt"], "integrated_multimodal_description: The man, wearing a grey suit, walks the "
                                           "dog through a park and talks. The man laughs.")
        self.assertIn("the summary's task tags", result["dropped"])
        self.assertIn("subject_definitions (2 subjects)", result["dropped"])

    def test_free_text_and_t2va_fields(self):
        """No Reference sections: only the labels go (dialogue keeps its own); t2va's own fields stay fields."""
        free = "A singer like <Picture 1> sings on a stage in the timbre of <Audio 1>. <d>[English] Hi <Picture 1>.</d>"
        result = helper.ref_text_only_prompt(free)
        self.assertEqual((result["form"], result["prompt"]),
                         ("free", "A singer sings on a stage. <d>[English] Hi <Picture 1>.</d>"))
        self.assertEqual(result["notes"], ["1 label(s) left as written, in dialogue"])
        fields = ("integrated_multimodal_description: [Shot 1] <Subject 1> shown in <Picture 1> waves.\n\n"
                  "overall_soundscape: Birds.\n\nnon_diegetic_music: N/A")
        result = helper.ref_text_only_prompt(fields)
        self.assertEqual((result["form"], result["prompt"]),
                         ("fields", "integrated_multimodal_description: [Shot 1] The subject waves.\n\n"
                                    "overall_soundscape: Birds.\n\nnon_diegetic_music: N/A"))
        self.assertEqual(result["notes"], [])

    def test_nothing_left(self):
        for prompt in ("<Picture 1> <Audio 1>", "  ", "subject_definitions: <Subject 1> is <Picture 1>.\nsummary: "
                       "[reference generation]\noverall_soundscape: Wind.", "detailed_description: [Shot 1] <Video 1>"):
            with self.subTest(prompt):
                with self.assertRaisesRegex(ValueError, "nothing is left of the prompt without its references"):
                    helper.ref_text_only_prompt(prompt)

    def test_an_undefined_subject(self):
        result = helper.ref_text_only_prompt("subject_definitions: <Subject 1> is a red kite.\ndetailed_description: "
                                             "<Subject 1> rises over <Subject 2>.")
        self.assertEqual(result["prompt"], "integrated_multimodal_description: A red kite rises over the subject.")
        self.assertEqual(result["notes"], ["<Subject 2> has no definition: it reads 'the subject'"])


# a stand-in of the patch's runtime package: runtime.config (load_paths, load_recipe, normalize_case,
# prompt_tokens_mode) and runtime.pipeline.Worker, whose "qwen" run writes conditioning.pt and conditioning.json as the
# real worker does, for the canvas of the recipe it got (stage1 height and width); every start, call and close goes
# into $STUB_QWEN_LOG. STUB_QWEN_BAD=canvas: the receipt names another canvas.
STUB_CONFIG = '''import json, os, re
from pathlib import Path
TASKS = ("t2va", "fl2va", "ref2va")

def prompt_tokens_mode(value=None):
    raw = os.environ.get("SOL_H3_PROMPT_TOKENS", "") if value is None else value
    mode = str(raw).strip().lower() or "minimax"
    if mode not in ("plain", "minimax"):
        raise ValueError(f"SOL_H3_PROMPT_TOKENS must be one of ('plain', 'minimax'), not {raw!r}")
    return mode

def load_recipe(task="t2va", *, orientation=None, **_):
    height, width = (384, 672) if (orientation or "landscape") == "landscape" else (672, 384)
    return {"stage1": {"height": height, "width": width, "task": task, "prompt_tokens": prompt_tokens_mode()},
            "orientation": orientation}

def load_paths(filename, *, task="t2va"):
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    return json.loads(Path(filename).read_text())

def normalize_case(case, *, base_dir=None, default_task="t2va"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", case.get("case_id") or ""):
        raise ValueError("case_id must be a simple file-safe name")
    result = {k: case[k] for k in ("case_id", "prompt", "seed", "task")}
    for key in ("first_frame", "last_frame"):
        if case.get(key) is not None:
            result[key] = str(Path(case[key]).resolve(strict=True))
    return result
'''
STUB_PIPELINE = '''import hashlib, json, os
from pathlib import Path

def log(entry):
    with open(os.environ["STUB_QWEN_LOG"], "a") as handle:
        handle.write(json.dumps(entry) + "\\n")

class Worker:
    def __init__(self, name, root, paths, config):
        self.root = Path(root)
        self.root.mkdir()
        self.config = config
        (self.root / "options.json").write_text(json.dumps({"paths": paths, "config": config}))
        log({"event": "start", "name": name, "root": str(self.root), "stage1": config["stage1"],
             "cwd": os.getcwd()})

    def call(self, op="run", case=None, output_root=None):
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=False)
        height, width = self.config["stage1"]["height"], self.config["stage1"]["width"]
        if os.environ.get("STUB_QWEN_BAD") == "canvas":
            height, width = height * 2, width * 2
        frames = [(a, case[f"{a}_frame"]) for a in ("first", "last") if case.get(f"{a}_frame")]
        tokens = (width // 32) * (height // 32)
        text = 40 + len(case["prompt"]) // 10
        rows = text + len(frames) * (tokens + 2)
        (root / "conditioning.pt").write_bytes(f"stand-in conditioning {case['prompt']}".encode())
        receipt = {"status": "PASS", "task": case["task"], "prompt": case["prompt"], "case_id": case["case_id"],
                   "seed": case["seed"], "shape": [1, rows, 5120], "text_tokens": text,
                   "prompt_tokens": self.config["stage1"]["prompt_tokens"],
                   "prepared_media": [{"type": "image", "path": p, "anchor": a, "shape": [height, width, 3]}
                                      for a, p in frames],
                   "payload_sha256": hashlib.sha256((root / "conditioning.pt").read_bytes()).hexdigest(),
                   "request_wall_s": 1.5}
        (root / "conditioning.json").write_text(json.dumps(receipt))
        log({"event": "run", "case": case, "output_root": str(root)})
        return receipt

    def close(self):
        log({"event": "close"})
'''


class Fixture(unittest.TestCase):
    """A runtime root with a clip's inputs, the studio's path manifests and a stand-in patched package."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3turbo-encode-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runtime = self.tmp / "runtime"
        self.inputs = self.runtime / "inputs" / "webui" / ("a" * 32)
        self.inputs.mkdir(parents=True)
        self.first = self.inputs / "first.png"
        self.first.write_bytes(b"first")
        self.last = self.inputs / "last.png"
        self.last.write_bytes(b"last")
        self.case = self.runtime / "outputs" / "bench" / "h3turbo-test" / "lake-first"
        self.package = self.tmp / "Sana" / "models" / "minimax_h3" / "Sol-H3-Spark"
        (self.package / "runtime").mkdir(parents=True)
        (self.package / "runtime" / "__init__.py").write_text("")
        (self.package / "runtime" / "config.py").write_text(STUB_CONFIG)
        (self.package / "runtime" / "pipeline.py").write_text(STUB_PIPELINE)
        for task in ("t2va", "fl2va"):
            (self.tmp / f"paths-{task}.json").write_text(json.dumps({"qwen_python": "/x/qwen_python.sh"}))
        self.prompt = self.tmp / "prompt.txt"
        self.prompt.write_text(guide_examples()["fl2va, first frame only, 5 s"], encoding="utf-8")
        self.log = self.tmp / "qwen.log"
        self.env = {"SOL_H3_SPARK_RUNTIME_ROOT": str(self.runtime),
                    "LFSTUDIO_PATHS": str(self.tmp / "paths-{task}.json"), "STUB_QWEN_LOG": str(self.log)}

    def args(self, *extra, first=True):
        out = ["--case", str(self.case), "--prompt-file", str(self.prompt), "--orientation", "landscape",
               "--package", str(self.package)]
        return out + (["--first", str(self.first)] if first else []) + list(extra)

    def run_helper(self, *args, env=None):
        environ = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "STUB_"))}
        environ.update(self.env, **(env or {}))
        return subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], capture_output=True, text=True,
                              timeout=60, env=environ, cwd=str(self.tmp))

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


class PlanTests(Fixture):
    def plan(self, *extra, first=True, env=None):
        args = helper.parse_args(self.args(*extra, first=first))
        return helper.make_plan(args, dict(self.env, **(env or {})), helper.load_job_module())

    def test_fl2va_at_the_draft_canvas(self):
        plan = self.plan()
        self.assertEqual((plan["task"], plan["keyframes"], plan["qwen_canvas"], plan["image_tokens"],
                          plan["canvas_mode"], plan["case_id"], plan["seed"]),
                         ("fl2va", ["first"], [672, 384], 252, "draft", "lake-first", 42))
        self.assertEqual(plan["manifest"], str(self.tmp / "paths-fl2va.json"))
        self.assertEqual(plan["text_only"]["dropped"], ["alignment line", "1 x ' shown in <Picture N>'"])
        self.assertEqual(plan["notes"], [])

    def test_the_canvas_per_job(self):
        for orientation, resolution, canvas, tokens in (("portrait", "576p", [576, 1024], 576),
                                                        ("landscape", "768p", [1344, 768], 1008),
                                                        ("portrait", "768p", [384, 672], 252)):
            mode = "draft" if tokens == 252 else "canvas"
            plan = self.plan("--orientation", orientation, "--resolution", resolution, "--canvas", mode)
            self.assertEqual((plan["qwen_canvas"], plan["image_tokens"]), (canvas, tokens))

    def test_t2va_and_a_last_frame_alone(self):
        plan = self.plan(first=False)
        self.assertEqual((plan["task"], plan["keyframes"], plan["image_tokens"], plan["manifest"]),
                         ("t2va", [], 0, str(self.tmp / "paths-t2va.json")))
        self.assertNotIn("text_only", plan)
        plan = self.plan("--last", str(self.last), first=False)
        self.assertEqual((plan["task"], plan["keyframes"]), ("fl2va", ["last"]))
        self.assertEqual(plan["notes"], [])          # a last frame alone: the studio offers it
        plan = self.plan("--last", str(self.last), "--no-text-only")
        self.assertEqual(plan["keyframes"], ["first", "last"])
        self.assertNotIn("text_only", plan)

    def test_refusals(self):
        outside = self.tmp / "outside.png"
        outside.write_bytes(b"x")
        (self.tmp / "empty.txt").write_text(" \n")
        (self.tmp / "only-align.txt").write_text(guide_examples()["fl2va, first frame only, 5 s"].split("\n")[0])
        for extra, env, message in (
                ((), {"SOL_H3_SPARK_RUNTIME_ROOT": ""}, "SOL_H3_SPARK_RUNTIME_ROOT must be set"),
                ((), {"SOL_H3_SPARK_RUNTIME_ROOT": "runtime"}, "must be set to an absolute path"),
                (("--case", str(self.tmp / "case")), {}, "--case .* is not under SOL_H3_SPARK_RUNTIME_ROOT"),
                (("--first", str(outside)), {}, "--first .* is not under SOL_H3_SPARK_RUNTIME_ROOT"),
                (("--last", str(self.inputs / "missing.png")), {}, "--last .*: no such file"),
                (("--prompt-file", str(self.tmp / "empty.txt")), {}, "the prompt is empty"),
                (("--prompt-file", str(self.tmp / "missing.txt")), {}, "cannot read --prompt-file"),
                (("--prompt-file", str(self.tmp / "only-align.txt")), {}, "nothing is left .*--no-text-only"),
                (("--case-id", "qwen-worker"), {}, "case id 'qwen-worker'"),
                (("--case-id", "a b"), {}, "case id 'a b'"),
                (("--seed", "-1"), {}, "--seed -1"),
                ((), {"LFSTUDIO_PATHS": str(self.tmp / "paths.json")}, "must contain {task}"),
                ((), {"LFSTUDIO_PATHS": str(self.tmp / "none-{task}.json")}, "no path manifest for fl2va"),
                (("--package", str(self.tmp)), {}, "no runtime/pipeline.py")):
            with self.subTest(message):
                with self.assertRaisesRegex(helper.Refused, message):
                    self.plan(*extra, env=env)
        plan = self.plan("--no-text-only", "--prompt-file", str(self.tmp / "only-align.txt"))
        self.assertNotIn("text_only", plan)
        (self.case / "qwen").mkdir(parents=True)
        with self.assertRaisesRegex(helper.Refused, "qwen exists: use a new --case"):
            self.plan()

    def test_the_prompt_as_the_server_takes_it(self):
        """--prompt-file and --prompt lose their leading and trailing whitespace (a file's last newline too), as
        webui/server.py's check_fields strips a prompt; its length limit and the text check are the server's."""
        text = self.prompt.read_text()
        (self.tmp / "newline.txt").write_text("  " + text + "\n", encoding="utf-8")
        plan = self.plan("--prompt-file", str(self.tmp / "newline.txt"))
        self.assertEqual(plan["prompt"], text)
        self.assertEqual(plan["prompt_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(plan["notes"], ["the prompt's leading and trailing whitespace (3 characters) is dropped, as "
                                         "the server drops it"])
        args = [a for a in self.args() if a not in ("--prompt-file", str(self.prompt))]
        plan = helper.make_plan(helper.parse_args(args + ["--prompt", "\tA lake.\r\n"]), self.env,
                                helper.load_job_module())
        self.assertEqual(plan["prompt"], "A lake.")
        server = (REPO / "webui" / "server.py").read_text()
        self.assertEqual(int(re.search(r"(?m)^PROMPT_MAX = (\d+)", server).group(1)), helper.PROMPT_MAX)
        self.assertIn('prompt = fields.get("prompt", "").strip()', server)
        for prompt, message in (("x" * (helper.PROMPT_MAX + 1), f"too long \\({helper.PROMPT_MAX + 1} characters"),
                                ("A lake \ud83c.", "not valid text"), (" \n ", "the prompt is empty")):
            with self.subTest(message):
                with self.assertRaisesRegex(helper.Refused, message):
                    helper.make_plan(helper.parse_args(args + ["--prompt", prompt]), self.env,
                                     helper.load_job_module())
        plan = helper.make_plan(helper.parse_args(args + ["--prompt", "x" * helper.PROMPT_MAX]), self.env,
                                helper.load_job_module())
        self.assertEqual(len(plan["prompt"]), helper.PROMPT_MAX)

    def test_the_code_is_recorded(self):
        plan = self.plan()
        self.assertEqual(plan["code_sha256"], {
            "helper": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "wrapper": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
            "job": hashlib.sha256((REPO / "jobs" / "h3turbo_job.py").read_bytes()).hexdigest()})

    def test_the_prompt_of_an_earlier_clip(self):
        earlier = self.runtime / "outputs" / "webui" / "earlier" / "web_000000_abcdef"
        (earlier / "qwen").mkdir(parents=True)
        prompt = guide_examples()["fl2va, first and last frame, 8 s"] + "\r\n"
        (earlier / "qwen" / "conditioning.json").write_text(json.dumps({"prompt": prompt, "task": "fl2va"}))
        args = [a for a in self.args() if a not in ("--prompt-file", str(self.prompt))]
        plan = helper.make_plan(helper.parse_args(args + ["--prompt-from-case", str(earlier)]), self.env,
                                helper.load_job_module())
        self.assertEqual(plan["prompt"], prompt)                 # exactly as it was encoded: nothing stripped
        self.assertEqual(plan["notes"], [])
        (earlier / "qwen" / "conditioning.json").write_text(json.dumps({"task": "fl2va"}))
        with self.assertRaisesRegex(helper.Refused, "has no prompt"):
            helper.make_plan(helper.parse_args(args + ["--prompt-from-case", str(earlier)]), self.env,
                             helper.load_job_module())
        with self.assertRaisesRegex(helper.Refused, "cannot read the prompt of --prompt-from-case"):
            helper.make_plan(helper.parse_args(args + ["--prompt-from-case", str(self.tmp)]), self.env,
                             helper.load_job_module())

    def test_the_default_manifest_follows_lfstudio_home(self):
        home = self.tmp / "home"
        home.mkdir()
        (home / "paths-fl2va.json").write_text("{}")
        plan = self.plan(env={"LFSTUDIO_PATHS": "", "LFSTUDIO_HOME": str(home)})
        self.assertEqual(plan["manifest"], str(home / "paths-fl2va.json"))


class ScriptTests(Fixture):
    def test_check_starts_nothing_and_writes_nothing(self):
        result = self.run_helper(*self.args("--canvas", "canvas", "--orientation", "portrait", "--check"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan["ok"], plan["task"], plan["qwen_canvas"], plan["image_tokens"]),
                         (True, "fl2va", [576, 1024], 576))
        self.assertEqual(plan["recipe_stage1"], {"height": 1024, "width": 576, "prompt_tokens": "minimax"})
        self.assertNotIn("prompt", plan)
        self.assertFalse(self.case.exists())
        self.assertEqual(self.events(), [])

    def test_refusals_exit_2(self):
        result = self.run_helper(*self.args("--case", str(self.tmp / "elsewhere")))
        self.assertEqual(result.returncode, 2)
        self.assertRegex(result.stdout, r"^\[h3turbo-encode\] refused: --case .* is not under")
        (self.tmp / "speech.txt").write_text(self.prompt.read_text() + " He says: <d>[English] Hi.</d>")
        result = self.run_helper(*self.args("--prompt-file", str(self.tmp / "speech.txt")),
                                 env={"SOL_H3_PROMPT_TOKENS": "plain"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("SOL_H3_PROMPT_TOKENS=plain would split MiniMax's tokens ['<d>', '</d>']", result.stdout)
        self.assertEqual(self.run_helper(*self.args("--prompt-file", str(self.tmp / "speech.txt")), "--check",
                                         env={"SOL_H3_PROMPT_TOKENS": "plain"}).returncode, 2)
        result = self.run_helper("--case", str(self.case))
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        self.assertEqual(self.events(), [])

    def test_fl2va_with_the_text_only_encode(self):
        """One worker for both encodes, the recipe's canvas set to the clip's, the text-only prompt as t2va, the
        receipts checked, the worker closed; the files and encode.json."""
        result = self.run_helper(*self.args("--canvas", "canvas", "--resolution", "576p", "--seed", "7"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        events = self.events()
        self.assertEqual([e["event"] for e in events], ["start", "run", "run", "close"])
        self.assertEqual((events[0]["root"], events[0]["stage1"]["height"], events[0]["stage1"]["width"],
                          events[0]["cwd"]), (str(self.case / "qwen-worker"), 576, 1024, str(self.package)))
        clip, text = events[1]["case"], events[2]["case"]
        self.assertEqual((clip["task"], clip["first_frame"], clip["seed"], clip["case_id"], events[1]["output_root"]),
                         ("fl2va", str(self.first), 7, "lake-first", str(self.case / "qwen")))
        self.assertEqual(clip["prompt"], self.prompt.read_text())
        self.assertEqual((text["task"], events[2]["output_root"]), ("t2va", str(self.case / "qwen-text")))
        self.assertNotIn("first_frame", text)
        self.assertEqual(text["prompt"], helper.text_only_prompt(clip["prompt"])["prompt"])
        self.assertEqual((self.case / "qwen-text" / "prompt.txt").read_text(), text["prompt"])
        text_only = json.loads((self.case / "qwen-text" / "text-only.json").read_text())
        self.assertEqual((text_only["dropped"], text_only["notes"]), (["alignment line", "1 x ' shown in <Picture N>'"],
                                                                      []))
        saved = json.loads((self.case / "encode.json").read_text())
        self.assertEqual(saved, summary)
        self.assertEqual(saved["code_sha256"]["helper"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(saved["text_only"]["notes"], [])
        self.assertEqual(saved["encodes"]["qwen"]["vision_rows"], 578)
        self.assertEqual(saved["encodes"]["qwen-text"]["vision_rows"], 0)
        self.assertEqual(saved["model"], "lightx2v-turbo")
        self.assertEqual(saved["for_job"], {"task": "fl2va", "orientation": "landscape", "resolution": "576p",
                                            "conditioning": str(self.case / "qwen" / "conditioning.pt"),
                                            "qwen_canvas": "canvas", "first_frame": str(self.first),
                                            "last_frame": None})
        self.assertNotIn("prompt", saved)
        # for_job and the job's other fields make a job.json the job takes
        job = helper.load_job_module()
        data = dict(json.loads(json.dumps(saved["for_job"])), seconds=5, seed=7, steps=4, lora="/l/lora.safetensors",
                    shift=[6, 3], paths={"base_model": "/b"})
        self.assertEqual(job.parse_job(data, str(self.case / "turbo"))["keyframes"], ["first"])

    def test_the_upscales_encode(self):
        """--upscale: the prompt as given, as t2va; MiniMax's tokens in plain mode are the studio's choice, not
        refused; keyframes are; encode.json says what it is for and has no for_job."""
        (self.tmp / "speech.txt").write_text("integrated_multimodal_description: A man says: <d>[English] Hi.</d>")
        args = self.args("--prompt-file", str(self.tmp / "speech.txt"), "--upscale", first=False)
        result = self.run_helper(*args, env={"SOL_H3_PROMPT_TOKENS": "plain"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e["event"] for e in self.events()], ["start", "run", "close"])
        case = self.events()[1]["case"]
        self.assertEqual((case["task"], case["prompt"]), ("t2va", (self.tmp / "speech.txt").read_text()))
        saved = json.loads((self.case / "encode.json").read_text())
        self.assertEqual((saved["use"], saved["task"], saved["prompt_tokens"]), ("upscale", "t2va", "plain"))
        self.assertNotIn("for_job", saved)
        self.assertEqual(saved["encodes"]["qwen"]["vision_rows"], 0)
        self.assertFalse((self.case / "qwen-text").exists())
        result = self.run_helper(*self.args("--upscale", "--case", str(self.case.parent / "other")))
        self.assertEqual(result.returncode, 2)
        self.assertIn("refused: --upscale encodes text alone: no --first or --last", result.stdout)

    def test_t2va(self):
        result = self.run_helper(*self.args(first=False))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e["event"] for e in self.events()], ["start", "run", "close"])
        self.assertEqual(self.events()[0]["stage1"]["height"], 384)
        self.assertFalse((self.case / "qwen-text").exists())
        self.assertEqual(json.loads((self.case / "encode.json").read_text())["for_job"]["task"], "t2va")

    def test_a_receipt_that_does_not_match_fails_and_closes_the_worker(self):
        result = self.run_helper(*self.args(), env={"STUB_QWEN_BAD": "canvas"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"failed: the qwen/ encode does not match: \['canvas', 'vision rows 1010, "
                                        r"want 254'\]")
        self.assertEqual([e["event"] for e in self.events()], ["start", "run", "close"])
        self.assertFalse((self.case / "encode.json").exists())

    def test_the_wrapper(self):
        """bench/h3turbo-encode.sh: the studio's settings (install/env.sh, an empty env file here), the Stage-1 Python
        ($LFSTUDIO_PYTHON) and the package under $LFSTUDIO_SANA; relative paths are the caller's."""
        sana = self.tmp / "Sana"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SOL_H3_", "LFSTUDIO_", "SOLH3_", "STUB_"))}
        env.update(self.env, LFSTUDIO_ENV=os.devnull, LFSTUDIO_HOME=str(self.tmp / "home"), LFSTUDIO_SANA=str(sana),
                   LFSTUDIO_PYTHON=sys.executable, SOL_H3_SECONDS="8")
        rel_prompt = os.path.relpath(self.prompt, self.tmp)
        result = subprocess.run(["bash", str(WRAPPER), "--case", str(self.case), "--prompt-file", rel_prompt,
                                 "--orientation", "portrait", "--first", str(self.first), "--check"],
                                capture_output=True, text=True, timeout=60, env=env, cwd=str(self.tmp))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan["package"], plan["qwen_canvas"]), (str(self.package), [384, 672]))
        result = subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("h3turbo-encode.sh --case DIR", result.stdout)
        env["LFSTUDIO_SANA"] = str(self.tmp / "nowhere")
        result = subprocess.run(["bash", str(WRAPPER), "--check"], capture_output=True, text=True, timeout=60,
                                env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no patched package", result.stderr)


if __name__ == "__main__":
    unittest.main()
