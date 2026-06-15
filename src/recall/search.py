"""Hybrid retrieval: FTS5 (BM25) + sqlite-vec KNN, fused with RRF.

Why hybrid: issue text is dominated by rare literal tokens — error codes, class
names, file paths — which BM25 nails and dense embeddings smear. Vectors in turn
catch paraphrases of the same problem. Reciprocal Rank Fusion combines the two
ranked lists without needing to normalize incomparable scores (BM25 vs cosine).
"""

from __future__ import annotations

import re
import sqlite3

from . import embed, store
from .models import SearchHit
from .store import _pack

RRF_K = 60  # standard RRF damping constant
OVERFETCH = 30  # candidates pulled per branch before fusion
DEDUP_THRESHOLD = 0.86  # cosine similarity above which a write is a likely dup

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(text: str) -> str:
    """Build an FTS5-safe MATCH expression from arbitrary user text.

    Stack traces and error strings contain ``:`` ``.`` ``()`` and quotes that are
    FTS5 operators and would raise syntax errors. We extract bare tokens, quote
    each, and OR them so any term can match.
    """
    tokens = _TOKEN_RE.findall(text)
    tokens = [t for t in tokens if len(t) > 1][:64]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _scopes(scope: str | None) -> list[str] | None:
    """Default search set: the given scope plus global. None means all scopes."""
    if scope is None:
        return None
    return ["global"] if scope == "global" else [scope, "global"]


def recall(
    conn: sqlite3.Connection,
    query: str,
    scope: str | None = "global",
    k: int = 5,
) -> list[SearchHit]:
    """Return the top-``k`` issues for ``query`` via hybrid search + RRF."""
    scopes = _scopes(scope)

    # --- FTS5 branch -------------------------------------------------------
    fts_ranks: dict[int, int] = {}
    match = _fts_query(query)
    if match:
        rows = conn.execute(
            "SELECT rowid FROM issues_fts WHERE issues_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, OVERFETCH),
        ).fetchall()
        for rank, row in enumerate(rows, start=1):
            fts_ranks[int(row["rowid"])] = rank

    # --- vector branch -----------------------------------------------------
    vec_ranks: dict[int, int] = {}
    qvec = _pack(embed.embed(query))
    rows = conn.execute(
        "SELECT issue_id, distance FROM issues_vec "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (qvec, OVERFETCH),
    ).fetchall()
    for rank, row in enumerate(rows, start=1):
        vec_ranks[int(row["issue_id"])] = rank

    # --- fuse with RRF -----------------------------------------------------
    fused: dict[int, float] = {}
    for ids in (fts_ranks, vec_ranks):
        for issue_id, rank in ids.items():
            fused[issue_id] = fused.get(issue_id, 0.0) + 1.0 / (RRF_K + rank)

    hits: list[SearchHit] = []
    for issue_id, score in fused.items():
        issue = store.get_issue(conn, issue_id)
        if issue is None:
            continue
        if scopes is not None and issue.scope not in scopes:
            continue
        # mild boost for proven-useful entries
        score += 0.001 * issue.helpful_count
        hits.append(
            SearchHit(
                issue=issue,
                score=score,
                fts_rank=fts_ranks.get(issue_id),
                vec_rank=vec_ranks.get(issue_id),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def find_duplicate(
    conn: sqlite3.Connection,
    text: str,
    scope: str,
    threshold: float = DEDUP_THRESHOLD,
) -> tuple[int, float] | None:
    """Return ``(issue_id, similarity)`` if a near-duplicate exists in scope.

    Uses cosine similarity (sqlite-vec ``vec_distance_cosine``) on the embedding,
    independent of the RRF ranking, so the dup gate is a clean similarity check.
    """
    qvec = _pack(embed.embed(text))
    row = conn.execute(
        "SELECT v.issue_id, vec_distance_cosine(v.embedding, ?) AS dist "
        "FROM issues_vec v JOIN issues i ON i.id = v.issue_id "
        "WHERE i.scope = ? ORDER BY dist LIMIT 1",
        (qvec, scope),
    ).fetchone()
    if row is None:
        return None
    similarity = 1.0 - float(row["dist"])
    if similarity >= threshold:
        return int(row["issue_id"]), similarity
    return None
