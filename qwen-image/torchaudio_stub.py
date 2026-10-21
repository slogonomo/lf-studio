"""Stub torchaudio for the Sol-H3 Qwen worker image (lf-studio).
The NGC PyTorch 25.11 base has no torchaudio; ComfyUI v0.30 imports it at module load for the LTX audio VAE,
which the Qwen prompt/vision encoder never uses. Any real use raises a clear error."""
__version__ = "0.0.0+lf-studio-stub"
def __getattr__(name):
    raise ImportError(f"torchaudio is a stub in the sol-h3-spark-qwen image (attribute {name!r} requested); "
                      "audio is not available in the Qwen worker")
