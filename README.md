# Sol-H3 Studio

Unofficial web studio and faster cold start for NVIDIA's Sol-H3-Spark video+audio pipeline on one DGX Spark.

Not affiliated with or endorsed by NVIDIA, MiniMax or Lightricks. Model weights are not included; you download
them yourself under each model's own terms ([Model terms](#model-terms)).

Needs one DGX Spark to itself while it runs (~110 GiB of its memory free), ~300 GB of disk and a Hugging Face account
with access to the gated LTX-2.5 weights. The timed install steps (the ~224 GB download ~48 min, the Stage1 environment
~22 min, the Docker image ~20 min, Stage2 ~6 min, steps 30 and 80 seconds) add up to about 1.6 h; steps 00, 10, 70
and 90 were not timed. Details: [Requirements](#requirements).

## What Sol-H3-Spark is

[Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/) is NVIDIA Research's two-stage video and
audio pipeline for a single DGX Spark (GB10). It is part of the Sol Video Inference Engine and lives in
[NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark), branch `sol-engine`:

1. A resident NVFP4 Qwen encoder (in a Docker container) encodes the prompt.
2. MiniMax-H3 drafts 672×384×124 frames (FP8, FastH3 4-step LoRA, sparse attention).
3. An H3 ×2 latent upscaler and an H3-to-LTX latent adapter move the draft to LTX-2.5.
4. LTX-2.5 refines it in 3 steps and decodes 1344×768, 121 frames at 24 fps, with H3's audio.

NVIDIA ships it as a command-line tool (`infer.py`) and a Python API, and reports 56 s for a warm 5 s clip.
Modes: text (t2va), first/last frame (fl2va), references (ref2va).

## What this repo adds

**A web studio** (`webui/`, Python standard library only) that keeps the pipeline loaded between clips:

- Modes Text, Keyframes and Reference (images, videos, audio, with H3's numbering), 16:9 or 9:16, 3-15 s.
- Prompt examples and section chips in H3's prompt formats (from MiniMax's
  [prompt-writing guides](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs)); seed roll; settings
  remembered per browser.
- A queue. Clips appear as cards with a timer and progress (over the clip's first keyframe, reference image or
  reference video frame, faded, to tell queued clips apart), then a player with Remix, Download and Delete; the
  seed is in the line with the time (a click or Tab selects it, to paste into the Seed box). Remix puts the
  clip's settings, prompt, seed and input files back into the form (with "New random seed every run" ticked, the
  next clip still gets a new seed); the server reuses the files it kept (uploads are stored per clip under
  `$SOL_H3_SPARK_RUNTIME_ROOT/inputs/webui/`). The gallery keeps every clip and loads older ones as you scroll.
  Delete moves a clip and its files to a trash folder for 7 days, with Undo. Cancel takes a queued clip out of the
  queue, or stops a running one (the models then load again for the next clip, about 1-2 min); its card stays as
  Cancelled, with Remix and Delete.
- The Generate button says when your settings need a model reload; the clip card shows how long the last load took.
  "Load selected now" under it starts loading other settings while you finish the prompt (nothing loads on a
  selection change alone).
- Warm-up at launch with the settings of your last clip; a clip with those settings submitted early in the warm-up
  takes it over, and the Generate button says when it would be ready. Unload button. Dark and light themes, phone layout.
- Uploaded reference videos are normalized (at most 672×384, 24 fps, 2-15 s).

**A patch** to Sol-H3-Spark (`patches/sol-h3-studio.patch`, details in [docs/PATCH.md](docs/PATCH.md)). Measured on
one DGX Spark; each "Before" is described in its row:

| | Before | With this patch | Before = |
|---|---|---|---|
| t2va 16:9, 5 s: launch to first finished clip | 282 s | 99 s | NVIDIA's flow plus this patch's persistent compile cache (without it: ~359 s) |
| t2va 16:9, 15 s, first clip | 681 s | 238 s | the pipeline before these changes: NVIDIA's flow with only this patch's length change (NVIDIA supports 5 s only) |
| ref2va with a 5 s reference video, each clip | 400 s | 133 s | the same pipeline with NVIDIA's reference-video canvas (H3's native 768p) instead of this repo's draft canvas |
| Formats | 16:9 | 16:9 and 9:16 | NVIDIA's Sol-H3-Spark |
| Clip length | 5 s | 3-15 s | NVIDIA's Sol-H3-Spark |
| Lip sync in dialogue: SyncNet confidence, man / woman, 8 clips (higher is better) | 4.4 / 3.5 | 7.8 / 5.2 | NVIDIA's Stage 2, which redraws the lips to a copy of the audio that it regenerates and then discards ([details](docs/PATCH.md#15-stage-2-audio-held-fixed-lip-sync)); H3's own draft scores 8.2 / 6.5 (4 of the clips) |

The VAE fix alone, measured with this repo's loading changes already in on both sides: first clip fl2va (first +
last frame, 5 s) 194 → 115 s, ref2va (image + video reference, 5 s) 254 → 178 s. No stock NVIDIA figure was
measured for these two modes.

Where the time went: [docs/PERFORMANCE.md](docs/PERFORMANCE.md). In short: the loading pass renders your clip
instead of a throwaway warm-up, the stages load in parallel, weights are read with O_DIRECT, the first-request
attention self-check covers 8 calls, autotune results persist (and an Inductor autotune-cache bug is worked around),
the VAEs are built on the GPU, and freed CPU memory goes back to the system (on GB10 it is GPU memory too). The
persistent compile cache is also part of the patch, but it is already in the 282 s.

**Install scripts** (`install/00-90`) that build the three environments, the Docker image, the path manifests and
the prompt cache with pinned versions, including fixes found on a DGX Spark (torchaudio on cu132, nvvm/crt pins
for ltx-kernels, a torchaudio stub for the Qwen image). **A launcher** (`bin/sol-h3-studio`) with memory checks
and clean shutdown for a terminal, NVIDIA Sync, systemd or
[spark-switchboard](https://github.com/slogonomo/spark-switchboard). **Bench scripts** (`bench/`).
**Tests** (`tests/`): the server end to end against a GPU-free stub of the pipeline, and the page in a
browser; see [AGENTS.md](AGENTS.md).

## Deviations from NVIDIA's validated recipe

NVIDIA validated the 5 s, 16:9 route. Portrait output is not quality-validated (the LoRAs, upscaler and adapter
were trained on landscape). Lengths other than 5 s are not validated by NVIDIA, and 3 s is below the 4-15 s that
MiniMax lists for H3. The first-request attention self-check covers 8 calls instead of 200. Portrait and length
are derived after the frozen-recipe hash check. Stage 2 refines the video against the H3 audio held fixed instead of
a re-noised copy that it regenerates and then discards (NVIDIA's contract), for better lip sync. All of it can be
switched back; see
[docs/PATCH.md](docs/PATCH.md#deviations-from-nvidias-validated-recipe).

## Requirements

- One NVIDIA DGX Spark (GB10, 128 GB). The pipeline needs nearly all memory: ~110 GiB free to start.
- DGX OS (Ubuntu 24.04 aarch64), CUDA 13 at `/usr/local/cuda`, Python 3.12 with venv, git, curl, ffmpeg, Docker
  with the NVIDIA runtime. Tested with driver 580.178.04.
- ~300 GB free disk (weights ~224 GB); ~60 GB if you reuse weights you already have (`SOLH3_CHECKPOINTS`).
- A Hugging Face account that accepted the [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) terms,
  and a read token for it (the quick start logs in with it).

## Quick start

```bash
git clone https://github.com/slogonomo/sol-h3-studio ~/src/sol-h3-studio && cd ~/src/sol-h3-studio
# Hugging Face login, once (accept the LTX-2.5 terms in the browser first). This is the download tool that
# install/20-weights.sh uses; the paths assume the default SOLH3_HOME=~/sol-h3.
python3.12 -m venv ~/sol-h3/envs/download && ~/sol-h3/envs/download/bin/pip install huggingface-hub==1.29.0
~/sol-h3/envs/download/bin/hf auth login     # or: export HF_TOKEN=<your read token>
install/00-check.sh            # prerequisites; changes nothing
install/10-sana.sh             # NVlabs/Sana sol-engine + this patch
install/20-weights.sh          # ~224 GB, ~48 min at ~80 MB/s
install/30-sources.sh
install/40-stage1.sh           # ~22 min
install/50-stage2.sh           # ~6 min
install/60-qwen-image.sh       # ~20 min
install/70-manifests.sh
install/80-prompt-cache.sh     # uses the GPU for ~20 s
install/90-env.sh              # writes ~/.config/sol-h3-studio/env
bin/sol-h3-studio              # then open http://localhost:8190
```

Every install script takes `--dry-run`. Paths, ports and switches: [bin/env.example](bin/env.example). Details,
times, disk use, which scripts have been run end to end, and updating: [docs/INSTALL.md](docs/INSTALL.md).
Problems: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

The server listens on 127.0.0.1:8190. From a laptop use [NVIDIA Sync](docs/nvidia-sync.md) or
`ssh -L 8190:127.0.0.1:8190 you@spark-xxxx.local`. `SOLH3_HOST=0.0.0.0` serves the LAN; there is no login, so only
on a network you trust (cross-site form posts are refused).

## Model terms

This repo contains no model weights and no H3-generated media. `install/20-weights.sh` runs NVIDIA's
`download_checkpoints.py`, which fetches pinned revisions from Hugging Face with your account. Each model has its
own license; read it before you download or use the model or its outputs:

| Model | Used for | Terms |
|---|---|---|
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | draft model | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). It does not cover the EU, the UK, South Korea or the USA (its §I.5 and §V.4); MiniMax links an [application form](https://platform.minimax.io/h3-license) for a license there. |
| [FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA) | 4-step draft LoRA (t2va, fl2va) | see the model card; NVIDIA's notices list it under the MiniMax-H3 license |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | 4-step draft LoRA (ref2va) | the card declares Apache-2.0; the MiniMax-H3 base model keeps its own terms |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | NVFP4 prompt encoder file | see the model card; NVIDIA's notices list it under the MiniMax-H3 license |
| [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | refiner, LoRA, VAEs, offline prompt-cache weights | gated; LTX-2.x Community License Agreement |
| [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) ([source](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)) | H3 ×2 latent upscaler | weights: the model card states Apache-2.0 (in its text at NVIDIA's pinned revision 13ccf95, and in its metadata since 2026-09-17). Source: MIT since 2026-09-17 (commit 40316cf); NVIDIA's pinned source revision d7c01b9 has no license file. NVIDIA's notices say no license is declared. Not redistributed here or by NVIDIA. |
| [Efficient-Large-Model/H3-to-LTX-Latent-Adapter](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter) | H3-to-LTX latent adapter | the model card assigns no separate weight license |

Sol-H3-Spark's own [THIRD_PARTY_NOTICES.md](https://github.com/NVlabs/Sana/blob/sol-engine/models/minimax_h3/Sol-H3-Spark/THIRD_PARTY_NOTICES.md)
lists the pinned revisions and terms.

## Credits

- **Sol-H3-Spark** and the Sol Video Inference Engine: NVIDIA Research
  ([project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/),
  [code](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3-Spark)). The two-stage method,
  the runtime and the install flow this repo builds on are theirs.
- Built on MiniMax-H3, FastVideo/FastH3 (Hao AI Lab), LTX-2.5 (Lightricks), Sol-Attn, ComfyUI, LightX2V, the
  LBH-123-AI H3 latent upscaler and the Efficient-Large-Model H3-to-LTX adapter.

```bibtex
@misc{li2026solvideoinferenceengine,
  title         = {Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation},
  author        = {Yitong Li and Junsong Chen and Haopeng Li and Haozhe Liu and Jincheng Yu and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
  year          = {2026},
  eprint        = {2606.23743},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.23743},
  url           = {https://arxiv.org/abs/2606.23743},
}

@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

## Related projects

Other public MiniMax-H3 / Sol-H3 work, on DGX Spark and in ComfyUI, as of 2026-10 (not affiliated with this repo):

- [trahane/dgx-spark-sol-h3-ref2v](https://github.com/trahane/dgx-spark-sol-h3-ref2v): a reproducible wrapper
  for Sol-H3-Spark's ref2va pipeline (CLI).
- [scottzx/dgx-spark-deploy](https://github.com/scottzx/dgx-spark-deploy): a Spark toolkit with scripts that run
  Sol-H3-Spark next to an LLM (CLI).
- [wassermanproductions/wasserman-spark-director-h3-speed-recipe](https://github.com/wassermanproductions/wasserman-spark-director-h3-speed-recipe):
  a two-Spark recipe that starts from Sol-H3-Spark.
- [Xplore-LAB/minimax-h3-dgx-spark](https://github.com/Xplore-LAB/minimax-h3-dgx-spark): a MiniMax-H3 on DGX Spark
  recipe; NVIDIA's `configs/dependencies.json` cites it as the source of its ComfyUI pin.
- ComfyUI nodes: [T8mars/comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
  (H3 draft → LTX-2.5 refine as nodes) and [xmarre/ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3)
  (H3 inference acceleration for ComfyUI in general, not specific to DGX Spark).
- [spark-switchboard](https://github.com/slogonomo/spark-switchboard), the companion repo: starts and stops apps
  like this studio on demand on one Spark (`integrations/spark-switchboard/`).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

- The patch modifies Apache-2.0 code from NVlabs/Sana; `runtime/qwen_ops/media.py` also contains FastVideo-derived
  Apache-2.0 code. Each modified file is marked "Modified by sol-h3-studio".
- ComfyUI (GPL-3.0) runs inside the Qwen container that you build yourself with `install/60-qwen-image.sh`, on top
  of NVIDIA's NGC PyTorch image (NVIDIA terms). This repo ships only the Dockerfile layer and three stub files.
  Never redistribute the built image.
- Model weights are not covered by this license; see [Model terms](#model-terms).

NVIDIA, DGX Spark and NVIDIA Sync are trademarks of NVIDIA Corporation; MiniMax and MiniMax H3 are trademarks of
MiniMax; LTX is a trademark of Lightricks. The names are used only to describe what this project works with.
