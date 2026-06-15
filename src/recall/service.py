"""Orchestration shared by the CLI and MCP server.

Keeps the embed -> dedup -> insert flow in one place so both entry points
behave identically.
"""

from __future__ import annotations

import sqlite3

from . import embed, search, store
from .models import Issue, SaveResult


def save_issue(conn: sqlite3.Connection, issue: Issue) -> SaveResult:
    """Embed, dedup-check, and insert an issue.

    Returns a :class:`SaveResult` carrying the new id and, when the new entry is
    a near-duplicate of an existing one in the same scope, a ``duplicate_of``
    hint so the caller can link/update instead of piling on noise.
    """
    text = issue.embed_text()
    dup = search.find_duplicate(conn, text, issue.scope)

    vector = embed.embed(text)
    issue_id = store.insert_issue(conn, issue, vector)

    return SaveResult(
        id=issue_id,
        status=issue.status,
        duplicate_of=dup[0] if dup else None,
        duplicate_score=round(dup[1], 4) if dup else None,
    )


def reembed(conn: sqlite3.Connection, issue_id: int) -> None:
    """Recompute and store the embedding for an existing issue."""
    issue = store.get_issue(conn, issue_id)
    if issue is None:
        return
    store.update_issue(conn, issue_id, {}, embedding=embed.embed(issue.embed_text()))
