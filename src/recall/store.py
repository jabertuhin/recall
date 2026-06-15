"""SQLite store: connection, sqlite-vec loading, migrations, CRUD.

The store owns three coordinated tables keyed by ``issues.id``:

* ``issues``      — canonical record (source of truth)
* ``issues_fts``  — FTS5 keyword index (kept in sync by triggers)
* ``issues_vec``  — sqlite-vec KNN index (written explicitly from Python)
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from .models import EMBED_DIM, Issue

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pack(vec: Sequence[float]) -> bytes:
    """Pack a float vector into the compact little-endian blob sqlite-vec wants."""
    return struct.pack(f"{len(vec)}f", *vec)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and the schema migrated."""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Create tables and triggers if absent. Idempotent."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return

    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS issues (
            id            INTEGER PRIMARY KEY,
            title         TEXT NOT NULL,
            symptom       TEXT NOT NULL,
            context       TEXT,
            root_cause    TEXT,
            fix           TEXT,
            tags          TEXT NOT NULL DEFAULT '[]',
            scope         TEXT NOT NULL DEFAULT 'global',
            status        TEXT NOT NULL DEFAULT 'approved',
            source        TEXT NOT NULL DEFAULT 'manual',
            helpful_count INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS issues_fts USING fts5(
            title, symptom, context, root_cause, fix, tags,
            content='issues', content_rowid='id'
        );

        -- keep FTS in sync with the canonical table
        CREATE TRIGGER IF NOT EXISTS issues_ai AFTER INSERT ON issues BEGIN
            INSERT INTO issues_fts(rowid, title, symptom, context, root_cause, fix, tags)
            VALUES (new.id, new.title, new.symptom, new.context, new.root_cause, new.fix, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS issues_ad AFTER DELETE ON issues BEGIN
            INSERT INTO issues_fts(issues_fts, rowid, title, symptom, context, root_cause, fix, tags)
            VALUES ('delete', old.id, old.title, old.symptom, old.context, old.root_cause, old.fix, old.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS issues_au AFTER UPDATE ON issues BEGIN
            INSERT INTO issues_fts(issues_fts, rowid, title, symptom, context, root_cause, fix, tags)
            VALUES ('delete', old.id, old.title, old.symptom, old.context, old.root_cause, old.fix, old.tags);
            INSERT INTO issues_fts(rowid, title, symptom, context, root_cause, fix, tags)
            VALUES (new.id, new.title, new.symptom, new.context, new.root_cause, new.fix, new.tags);
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS issues_vec USING vec0(
            issue_id INTEGER PRIMARY KEY,
            embedding float[{EMBED_DIM}],
            +scope TEXT
        );

        PRAGMA user_version = {SCHEMA_VERSION};
        """
    )
    conn.commit()


def _row_to_issue(row: sqlite3.Row) -> Issue:
    data = dict(row)
    data["tags"] = json.loads(data.get("tags") or "[]")
    return Issue(**data)


def insert_issue(
    conn: sqlite3.Connection, issue: Issue, embedding: Sequence[float]
) -> int:
    """Insert a new issue and its embedding. Returns the new id."""
    ts = _now()
    cur = conn.execute(
        """
        INSERT INTO issues
            (title, symptom, context, root_cause, fix, tags, scope, status,
             source, helpful_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            issue.title,
            issue.symptom,
            issue.context,
            issue.root_cause,
            issue.fix,
            json.dumps(issue.tags),
            issue.scope,
            issue.status,
            issue.source,
            issue.helpful_count,
            ts,
            ts,
        ),
    )
    issue_id = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO issues_vec(issue_id, embedding, scope) VALUES (?, ?, ?)",
        (issue_id, _pack(embedding), issue.scope),
    )
    conn.commit()
    return issue_id


def get_issue(conn: sqlite3.Connection, issue_id: int) -> Issue | None:
    row = conn.execute("SELECT * FROM issues WHERE id = ?", (issue_id,)).fetchone()
    return _row_to_issue(row) if row else None


def list_issues(
    conn: sqlite3.Connection, status: str | None = None, limit: int = 50
) -> list[Issue]:
    if status:
        rows = conn.execute(
            "SELECT * FROM issues WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM issues ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_issue(r) for r in rows]


def update_issue(
    conn: sqlite3.Connection,
    issue_id: int,
    fields: dict,
    embedding: Sequence[float] | None = None,
) -> None:
    """Patch selected fields. Re-embed by passing ``embedding``."""
    if not fields and embedding is None:
        return
    if fields:
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"])
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE issues SET {cols} WHERE id = ?",
            (*fields.values(), issue_id),
        )
    if embedding is not None:
        scope_row = conn.execute(
            "SELECT scope FROM issues WHERE id = ?", (issue_id,)
        ).fetchone()
        scope = scope_row["scope"] if scope_row else "global"
        conn.execute("DELETE FROM issues_vec WHERE issue_id = ?", (issue_id,))
        conn.execute(
            "INSERT INTO issues_vec(issue_id, embedding, scope) VALUES (?, ?, ?)",
            (issue_id, _pack(embedding), scope),
        )
    conn.commit()


def set_status(conn: sqlite3.Connection, issue_id: int, status: str) -> None:
    conn.execute(
        "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
        (status, _now(), issue_id),
    )
    conn.commit()


def bump_helpful(conn: sqlite3.Connection, issue_id: int) -> int:
    conn.execute(
        "UPDATE issues SET helpful_count = helpful_count + 1 WHERE id = ?",
        (issue_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT helpful_count FROM issues WHERE id = ?", (issue_id,)
    ).fetchone()
    return int(row["helpful_count"]) if row else 0


def delete_issue(conn: sqlite3.Connection, issue_id: int) -> None:
    conn.execute("DELETE FROM issues_vec WHERE issue_id = ?", (issue_id,))
    conn.execute("DELETE FROM issues WHERE id = ?", (issue_id,))
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM issues WHERE status = 'pending_review'"
    ).fetchone()[0]
    scopes = conn.execute(
        "SELECT scope, COUNT(*) c FROM issues GROUP BY scope ORDER BY c DESC"
    ).fetchall()
    return {
        "total": total,
        "pending_review": pending,
        "scopes": {r["scope"]: r["c"] for r in scopes},
    }
