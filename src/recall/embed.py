"""Embedding wrapper around fastembed (ONNX, no torch).

The model (``BAAI/bge-small-en-v1.5``, 384-dim) is lazy-loaded on first use and
downloaded (~130MB) on first ever run. Use ``warmup()`` to pre-fetch.
"""

from __future__ import annotations

from functools import lru_cache

from .models import EMBED_DIM

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so CLI commands that don't embed (e.g. `list`) stay fast
    # and don't pay the onnxruntime import cost.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=MODEL_NAME)


def embed(text: str) -> list[float]:
    """Embed a single string into a 384-dim float vector."""
    vec = next(iter(_model().embed([text])))
    out = vec.tolist()
    if len(out) != EMBED_DIM:  # guard against a model/dim mismatch
        raise ValueError(f"expected {EMBED_DIM}-dim embedding, got {len(out)}")
    return out


def embed_many(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings."""
    return [v.tolist() for v in _model().embed(texts)]


def warmup() -> None:
    """Force model download + load."""
    embed("warmup")
