# LF Studio layer on top of NVIDIA's Sol-H3-Spark Qwen image (install/60-qwen-image.sh builds both).
#
# Why: NVIDIA's Dockerfile.qwen starts from NGC PyTorch 25.11, which has no torchaudio. ComfyUI v0.30.0, which
# that image installs for the Qwen tokenizer and visual encoder, imports torchaudio (and torchaudio.functional,
# torchaudio.transforms.MelScale) at module load for its LTX audio VAE. The Qwen prompt encoder never uses audio,
# but the import fails without the package. These stubs make the imports succeed; any real use raises ImportError.
#
# The built image contains GPL-3.0 ComfyUI and NGC components. Build it yourself; do not redistribute it.
ARG BASE=sol-h3-spark-qwen:base
FROM ${BASE}
COPY torchaudio_stub.py /usr/local/lib/python3.12/dist-packages/torchaudio/__init__.py
COPY functional_stub.py /usr/local/lib/python3.12/dist-packages/torchaudio/functional.py
COPY transforms_stub.py /usr/local/lib/python3.12/dist-packages/torchaudio/transforms.py
