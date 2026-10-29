"""The replay bench's arms: one named function per arm. Standard library only (no torch at import).

An arm is a change to Stage 2 (or no change: `baseline`). Its function receives a run object and calls the setters
below. On the GPU that object is `replay.stage2.GpuRun`, which applies each change to the live Stage-2 session for one
run and undoes it afterwards. In `--dry` and in the tests it is `Recorder`, which only writes the settings down.

  run.sigmas(values)             the Stage-2 sigma schedule; the noise added is values[0], the steps len(values) - 1
  run.no_denoise()               no refine: the adapter's latent is decoded as it is (a preview of the draft)
  run.tiles(short=(size, overlap), long=(size, overlap))
                                 decoder tiles along the frame's short and long side; (0, 0) = one tile on that side.
                                 The temporal tiles stay 128/24.
  run.profile_step(index)        torch.profiler around one denoise step (its timings are perturbed)
  run.keyframe(mode)             SOL_H3_STAGE2_KEYFRAME for this run: off, pin (the first frame held at latent frame 0)
                                 or draft (the draft's own frame 0 held). It applies to fl2va drafts with a first frame
                                 only; other drafts run as off. pin builds the image encoder once if the process lacks
                                 it (started without SOL_H3_STAGE2_KEYFRAME=pin), and it then stays loaded.

The output canvas is fixed per process (Inductor compiles one shape), so an arm declares it (`canvas=480`) instead
of setting it per run. 768 means the draft's own 2x canvas, as the studio renders.

Where an arm leaves NVIDIA's validated route, the runtime checks that it would fail are named in `relaxes`. Every
other runtime check stays in force; the bench also checks after each run that the receipt shows the arm's settings.
"""

SOL_LAYERS = 47                  # Sol attention layers per step (48-layer refiner, layer 0 dense): 3 steps = 141 calls
KEYFRAME_MODES = ("off", "pin", "draft")       # runtime/config.py STAGE2_KEYFRAME_MODES
TEMPORAL_TILES = (128, 24)
# canvas -> (short side, long side) of the output; 768 = the draft's own 2x canvas (1344x768 or 768x1344)
CANVASES = {768: None, 480: (480, 832), 384: (384, 672)}

ARMS = {}


class Arm:
    def __init__(self, name, fn, canvas, est_s, desc):
        self.name, self.fn, self.canvas, self.est_s, self.desc = name, fn, canvas, est_s, desc

    def estimate_s(self, output_frames):
        """Estimated Stage-2 seconds per run (from the 2026-10-12 bench at 5 s; inferred for other lengths)."""
        return self.est_s * (output_frames / 121) ** 1.07


def arm(name, *, canvas=768, est_s, desc):
    """Register an arm. est_s: Stage-2 seconds measured for a 5 s clip on 2026-10-12 (an estimate for the plan)."""
    if canvas not in CANVASES:
        raise ValueError(f"canvas must be one of {sorted(CANVASES)}")

    def register(fn):
        if name in ARMS:
            raise ValueError(f"arm {name} registered twice")
        ARMS[name] = Arm(name, fn, canvas, est_s, desc)
        return fn
    return register


# ---- the draft's own canvas (1344x768 or 768x1344) ---------------------------------------------------------------
@arm("baseline", est_s=35.6, desc="today's recipe, unchanged: 3 steps from sigma 0.909375, default decode tiles")
def baseline(run):
    pass


@arm("steps2_from_0725", est_s=27.5, desc="2-step refine 0.725/0.421875/0 (NVIDIA's skip_0p9 schedule)")
def steps2_from_0725(run):
    run.sigmas((0.725, 0.421875, 0.0))


@arm("steps2_from_0625", est_s=27.5, desc="2-step refine 0.625/0.4/0 (LTX-2's distilled TDP sigmas)")
def steps2_from_0625(run):
    run.sigmas((0.625, 0.4, 0.0))


@arm("preview", est_s=11.0, desc="no refine: upscale + adapter + LTX decode of the draft (what the draft looks like)")
def preview(run):
    run.no_denoise()


@arm("decode_untiled", est_s=34.5, desc="decoder spatially untiled (one tile per frame)")
def decode_untiled(run):
    run.tiles(short=(0, 0), long=(0, 0))


@arm("decode_long768", est_s=34.8, desc="decoder: short side untiled, long side 768/64 (2 tiles instead of 4)")
def decode_long768(run):
    run.tiles(short=(0, 0), long=(768, 64))


@arm("profile_step", est_s=36.7, desc="baseline with torch.profiler around denoise step 1 (timings perturbed)")
def profile_step(run):
    run.profile_step(1)


# ---- the user's first frame in Stage 2 (SOL_H3_STAGE2_KEYFRAME, issue #57; fl2va drafts with a first frame) -------
@arm("keyframe_off", est_s=35.6, desc="no keyframe pin, whatever SOL_H3_STAGE2_KEYFRAME says (today's default)")
def keyframe_off(run):
    run.keyframe("off")


@arm("keyframe_pin", est_s=36.6, desc="the first frame, stretched to the canvas and encoded by the LTX-2.5 VAE, held "
                                      "clean at latent frame 0 (estimate: +1 s)")
def keyframe_pin(run):
    run.keyframe("pin")


@arm("keyframe_draft", est_s=35.6, desc="the draft's own frame-0 latent held clean at latent frame 0 (a control for "
                                        "keyframe_pin)")
def keyframe_draft(run):
    run.keyframe("draft")


@arm("keyframe_pin_steps2", est_s=28.5, desc="keyframe_pin with the 2-step refine 0.725/0.421875/0")
def keyframe_pin_steps2(run):
    run.keyframe("pin")
    run.sigmas((0.725, 0.421875, 0.0))


# ---- 832x480 (480x832 for portrait): own process, compiles new shapes on its first run (+70-80 s) ---------------
@arm("canvas480", canvas=480, est_s=14.4, desc="refine at 832x480: H3 upscale x1.25 + adapter + 3 steps")
def canvas480(run):
    pass


@arm("canvas480_untiled", canvas=480, est_s=13.7, desc="canvas480 with the decoder spatially untiled")
def canvas480_untiled(run):
    run.tiles(short=(0, 0), long=(0, 0))


@arm("canvas480_steps2", canvas=480, est_s=11.2, desc="canvas480 with the 2-step refine 0.725/0.421875/0")
def canvas480_steps2(run):
    run.sigmas((0.725, 0.421875, 0.0))


@arm("preview480", canvas=480, est_s=5.5, desc="no refine at 832x480: H3 upscale x1.25 + adapter + LTX decode")
def preview480(run):
    run.no_denoise()


# ---- 672x384 (384x672): the draft's own size, no upscaler ----------------------------------------------------------
@arm("canvas384", canvas=384, est_s=8.4, desc="refine at 672x384 (the draft's size, no upscaler) + 3 steps")
def canvas384(run):
    pass


@arm("preview384", canvas=384, est_s=2.2, desc="no refine at 672x384: adapter + LTX decode, no upscaler")
def preview384(run):
    run.no_denoise()


# ---- shapes ---------------------------------------------------------------------------------------------------------
def canvas_geometry(native, canvas):
    """The Stage-2 shapes for an arm's canvas, from the runtime's own geometry for the draft (runtime.geometry
    from_recipe). The temporal shapes stay; the spatial ones follow the new output size."""
    if CANVASES[canvas] is None:
        return dict(native)
    short, long = CANVASES[canvas]
    portrait = native["output_height"] > native["output_width"]
    out_h, out_w = (long, short) if portrait else (short, long)
    t_h3, t_ad, t_ref = native["h3"][2], native["adapter_output"][2], native["refiner_input"][2]
    geo = dict(native)
    geo.update(output_height=out_h, output_width=out_w,
               h3_upscaled=(1, native["h3_upscaled"][1], t_h3, out_h // 16, out_w // 16),
               adapter_output=(1, native["adapter_output"][1], t_ad, out_h // 32, out_w // 32),
               refiner_input=(1, native["refiner_input"][1], t_ref, out_h // 32, out_w // 32))
    return geo


def upscale_scale(geo):
    """The H3 upscaler's scale input: short side of the upscaled latent over the draft's (2.0 in the runtime)."""
    return min(geo["h3_upscaled"][3:]) / min(geo["h3"][3:])


def video_tokens(geo):
    t, h, w = geo["refiner_input"][2:]
    return t * h * w


def default_tiles(geo):
    """The geometry's decoder tiles as {short, long, temporal: (size, overlap)}."""
    tiles = geo["tiles"]
    pair = lambda key: (tiles[key]["tile_size"], tiles[key]["overlap"])
    portrait = geo["output_height"] > geo["output_width"]
    short, long = ("width", "height") if portrait else ("height", "width")
    return {"short": pair(short), "long": pair(long), "temporal": pair("frames")}


def tiles_hw(tiles, geo):
    """{short, long, temporal} -> {height, width, frames} for this geometry's orientation."""
    portrait = geo["output_height"] > geo["output_width"]
    height, width = (tiles["long"], tiles["short"]) if portrait else (tiles["short"], tiles["long"])
    return {"frames": list(tiles["temporal"]), "height": list(height), "width": list(width)}


# ---- recording an arm's settings (--dry, tests, and the row of every run) -----------------------------------------
class Recorder:
    """Records what an arm function sets, against the recipe's own values."""

    def __init__(self, recipe_sigmas, geo):
        self.recipe_sigmas = tuple(recipe_sigmas)
        self.geo = geo
        self.values = {"sigmas": list(recipe_sigmas), "denoise": True, "tiles": default_tiles(geo),
                       "profile_step": None, "keyframe": None}
        self.changed = []

    def sigmas(self, values):
        values = tuple(float(v) for v in values)
        if len(values) < 2 or values[-1] != 0.0 or any(a <= b for a, b in zip(values, values[1:])):
            raise ValueError(f"sigmas must fall to 0 and have at least one step: {values}")
        self.values["sigmas"] = list(values)
        self.changed.append("sigmas")

    def no_denoise(self):
        self.values["sigmas"] = [0.0]
        self.values["denoise"] = False
        self.changed.append("denoise")

    def tiles(self, short=None, long=None):
        tiles = dict(self.values["tiles"])
        for side, value in (("short", short), ("long", long)):
            if value is not None:
                size, overlap = (int(v) for v in value)
                if size < 0 or overlap < 0 or (size and overlap >= size):
                    raise ValueError(f"bad {side} tile {value}")
                tiles[side] = (size, overlap)
        self.values["tiles"] = tiles
        self.changed.append("tiles")

    def profile_step(self, index):
        self.values["profile_step"] = int(index)
        self.changed.append("profile_step")

    def keyframe(self, mode):
        if mode not in KEYFRAME_MODES:
            raise ValueError(f"keyframe mode must be one of {KEYFRAME_MODES}, not {mode!r}")
        self.values["keyframe"] = mode
        self.changed.append("keyframe")

    @property
    def steps(self):
        return 0 if not self.values["denoise"] else len(self.values["sigmas"]) - 1


def settings(name, recipe_sigmas, native):
    """What arm `name` runs with: canvas, shapes, sigmas, steps, tiles, and the runtime checks it relaxes."""
    a = ARMS[name]
    geo = canvas_geometry(native, a.canvas)
    rec = Recorder(recipe_sigmas, geo)
    a.fn(rec)
    steps = rec.steps
    relaxes = []
    recipe_steps = len(recipe_sigmas) - 1
    if not rec.values["denoise"]:
        relaxes.append("denoise skipped: the refiner's denoise_multimodal returns the noise-free states unchanged "
                       "(sigmas [0.0], noise scale 0)")
    if steps != recipe_steps:
        relaxes.append(f"attention-call check (_checked_attention_stats): {steps} steps ({steps} dense, "
                       f"{SOL_LAYERS * steps} Sol calls) instead of the recipe's {recipe_steps} "
                       f"({recipe_steps} dense, {SOL_LAYERS * recipe_steps} Sol calls)")
    if CANVASES[a.canvas] is not None:
        scale = upscale_scale(geo)
        relaxes.append(f"runtime.geometry.from_recipe requires the output to be exactly 2x the draft: the bench "
                       f"derives the {geo['output_width']}x{geo['output_height']} shapes itself")
        relaxes.append("the H3 upscaler is skipped (the draft is already this size)" if geo["h3_upscaled"] == geo["h3"]
                       else f"the H3 upscaler runs at scale {scale:g} (the runtime uses 2)")
        relaxes.append("the LTX refiner modules' 1344x768 contract is checked on the draft's own geometry, then their "
                       f"size globals are set to {geo['output_width']}x{geo['output_height']}")
    return {"arm": name, "canvas": a.canvas, "output": [geo["output_width"], geo["output_height"]],
            "sigmas": rec.values["sigmas"], "steps": steps, "denoise": rec.values["denoise"],
            "tiles": tiles_hw(rec.values["tiles"], geo), "tiles_default": "tiles" not in rec.changed,
            "upscale_scale": round(upscale_scale(geo), 4), "video_tokens": video_tokens(geo),
            "profile_step": rec.values["profile_step"], "keyframe": rec.values["keyframe"], "relaxes": relaxes,
            "desc": a.desc}


def keyframe_expected(mode, capture):
    """The keyframe mode a run's Stage-2 receipt must show (stage2_keyframe): `mode` for an fl2va draft with a first
    frame (its capture.json input_spec), "off" for any other draft (runtime/stage2.py keyframe_mode_for)."""
    spec = capture.get("input_spec") or {}
    return mode if capture.get("task") == "fl2va" and spec.get("first_frame") else "off"
