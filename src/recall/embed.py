"""Embedding wrapper around fastembed (ONNX, no torch).

The model (``BAAI/bge-small-en-v1.5``, 384-dim) is lazy-loaded on first use and
downloaded (~130MB) on first ever run. Use ``warmup()`` to pre-fetch.

The cache dir is pinned to ``~/.recall/models`` (override with ``RECALL_MODEL_DIR``)
rather than fastembed's default ``$TMPDIR/fastembed_cache`` — the temp default gets
reaped by the OS, forcing silent re-downloads, which is especially bad on the MCP
path where nobody runs ``warmup`` by hand.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .models import EMBED_DIM

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CACHE_DIR = Path(
    os.environ.get("RECALL_MODEL_DIR", Path.home() / ".recall" / "models")
)


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so CLI commands that don't embed (e.g. `list`) stay fast
    # and don't pay the onnxruntime import cost.
    from fastembed import TextEmbedding

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))


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
