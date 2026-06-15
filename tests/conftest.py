"""Shared fixtures. Stubs the embedding model so tests don't download ~130MB.

The stub maps text -> a deterministic 384-dim vector built from token hashing,
so semantically-overlapping strings get measurably closer vectors. This keeps
the search/RRF/dedup logic under test without pulling in onnxruntime.
"""

from __future__ import annotations

import math
import re

import pytest

from recall import embed, store
from recall.models import EMBED_DIM

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _fake_vector(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for tok in _TOKEN.findall(text.lower()):
        vec[hash(tok) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(embed, "embed", _fake_vector)
    monkeypatch.setattr(embed, "embed_many", lambda texts: [_fake_vector(t) for t in texts])


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()
