# LF Studio layer on top of NVIDIA's Sol-H3-Spark Qwen image (install/60-qwen-image.sh builds both): torchaudio
# stubs and Pillow 13.0.0 (below).
#
# The stubs: NVIDIA's Dockerfile.qwen starts from NGC PyTorch 25.11, which has no torchaudio. ComfyUI v0.30.0, which
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
# Pillow 13.0.0 (#69): NGC PyTorch 25.11 ships Pillow 12.0.0, which the image-decoder advisories of 2026-10-22 affect
# (GHSA-v9pv-5gc8-w2m6 = CVE-2026-96639 and others), and the Qwen encoder opens the keyframes and reference images
# clients upload. Nothing in the image caps Pillow below 13.
RUN python3 -m pip install --no-cache-dir --no-deps --only-binary :all: pillow==13.0.0 && \
    python3 -c 'import PIL; assert PIL.__version__ == "13.0.0", PIL.__version__'
