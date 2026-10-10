"""torchaudio.functional stub (sol-h3-spark-qwen)."""
def __getattr__(name):
    def _f(*a, **k):
        raise ImportError(f"torchaudio.functional.{name} is unavailable (stub in the sol-h3-spark-qwen image)")
    return _f
