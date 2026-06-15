"""FastMCP server exposing recall to any coding agent.

Tools:
    recall_issues(problem, scope?, k?)              -> list of past matches
    save_resolution(title, symptom, ..., scope?)    -> queued for review (+dup hint)
    mark_helpful(issue_id)                          -> feedback signal

Agents should be instructed (one line in their system prompt) to call
``recall_issues`` BEFORE investigating a new problem, and ``save_resolution``
after resolving one. Agent-saved entries land as ``pending_review`` — searchable
immediately but flagged as unreviewed, so a human gate (``recall review``)
controls what becomes trusted memory.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import config, search, service, store
from .models import Issue

mcp = FastMCP("recall")


def _conn():
    return store.connect(config.store_path())


@mcp.tool()
def recall_issues(
    problem: str,
    scope: Optional[str] = None,
    k: int = 5,
) -> list[dict]:
    """Find previously-resolved issues similar to ``problem``.

    Call this FIRST when investigating an error or unexpected behavior. Returns
    past symptom/cause/fix plus trust signals (whether the entry was reviewed,
    how often it helped, when it was added) so you can weigh staleness.

    Args:
        problem: The error message, stack trace, or behavior description.
        scope: Repo scope to search; defaults to the current repo + global.
        k: Max results.
    """
    conn = _conn()
    try:
        s = scope or config.current_scope()
        hits = search.recall(conn, problem, s, k)
        return [
            {
                "id": h.issue.id,
                "title": h.issue.title,
                "symptom": h.issue.symptom,
                "context": h.issue.context,
                "root_cause": h.issue.root_cause,
                "fix": h.issue.fix,
                "tags": h.issue.tags,
                "score": round(h.score, 4),
                "status": h.issue.status,
                "helpful_count": h.issue.helpful_count,
                "added": h.issue.created_at,
                "trust": h.trust_note(),
            }
            for h in hits
        ]
    finally:
        conn.close()


@mcp.tool()
def save_resolution(
    title: str,
    symptom: str,
    root_cause: str,
    fix: str,
    context: Optional[str] = None,
    tags: Optional[list[str]] = None,
    scope: Optional[str] = None,
) -> dict:
    """Record a resolved issue for future recall (queued for human review).

    Call after you've diagnosed and fixed a problem. The entry is searchable
    right away but marked ``pending_review`` until a human approves it.

    Args:
        title: Short, distinctive summary of the issue.
        symptom: The error / observed behavior, verbatim where possible.
        root_cause: Why it happened.
        fix: What resolved it.
        context: Module, language, stack, versions.
        tags: Optional labels.
        scope: Repo scope; defaults to the current repo.

    Returns id, status, and a ``duplicate_of`` hint if a near-identical entry
    already exists — in which case prefer updating that one over duplicating.
    """
    conn = _conn()
    try:
        issue = Issue(
            title=title,
            symptom=symptom,
            context=context,
            root_cause=root_cause,
            fix=fix,
            tags=tags or [],
            scope=scope or config.current_scope(),
            status="pending_review",
            source="agent",
        )
        result = service.save_issue(conn, issue)
        return result.model_dump()
    finally:
        conn.close()


@mcp.tool()
def mark_helpful(issue_id: int) -> dict:
    """Signal that issue ``issue_id`` helped resolve the current problem."""
    conn = _conn()
    try:
        count = store.bump_helpful(conn, issue_id)
        return {"id": issue_id, "helpful_count": count}
    finally:
        conn.close()


def main() -> None:
    # Pull the ~130MB model off the hot path: download/load in the background at
    # startup so the agent's first recall/save call doesn't block (and risk the
    # MCP client's tool timeout) on a cold cache.
    import threading

    from . import embed

    threading.Thread(target=embed.warmup, daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
