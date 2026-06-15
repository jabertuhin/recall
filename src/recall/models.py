"""Pydantic models for recall records and search results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["pending_review", "approved"]
Source = Literal["agent", "manual"]

EMBED_DIM = 384


class Issue(BaseModel):
    """A resolved (or proposed) issue: symptom -> context -> cause -> fix."""

    id: int | None = None
    title: str
    symptom: str
    context: str | None = None
    root_cause: str | None = None
    fix: str | None = None
    tags: list[str] = Field(default_factory=list)
    scope: str = "global"
    status: Status = "approved"
    source: Source = "manual"
    helpful_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None

    def embed_text(self) -> str:
        """Concatenated text used to build the semantic embedding."""
        parts = [self.title, self.symptom, self.context, self.root_cause, self.fix]
        return "\n".join(p for p in parts if p)


class SearchHit(BaseModel):
    """A retrieval result with fused score and trust signals."""

    issue: Issue
    score: float
    fts_rank: int | None = None
    vec_rank: int | None = None

    def trust_note(self) -> str:
        flags = []
        if self.issue.status == "pending_review":
            flags.append("UNREVIEWED")
        if self.issue.helpful_count:
            flags.append(f"helpful×{self.issue.helpful_count}")
        if self.issue.created_at:
            flags.append(f"added {self.issue.created_at[:10]}")
        return " | ".join(flags)


class SaveResult(BaseModel):
    """Outcome of a write, including a possible duplicate hint."""

    id: int
    status: Status
    duplicate_of: int | None = None
    duplicate_score: float | None = None
