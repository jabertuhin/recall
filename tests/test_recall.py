"""Tests for schema, write/dedup, hybrid search + RRF, scope, and feedback."""

from __future__ import annotations

from recall import search, service, store
from recall.models import EMBED_DIM, Issue


def _issue(**kw) -> Issue:
    base = dict(title="t", symptom="s", scope="repoA", status="approved")
    base.update(kw)
    return Issue(**base)


def test_migration_creates_tables(conn):
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table')"
        ).fetchall()
    }
    assert {"issues", "issues_fts", "issues_vec"} <= names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_embed_dim_constant():
    assert EMBED_DIM == 384


def test_save_and_get(conn):
    res = service.save_issue(
        conn,
        _issue(
            title="Spark NPE on null partition col",
            symptom="NullPointerException at PartitionWriter.scala:42",
            root_cause="partition column null in source",
            fix="coalesce the column before write",
        ),
    )
    assert res.id == 1 and res.status == "approved"
    got = store.get_issue(conn, 1)
    assert got.title.startswith("Spark NPE")
    assert got.tags == []


def test_dedup_hint_on_near_duplicate(conn):
    sym = "Kafka consumer rebalance loop CommitFailedException group data-ingest"
    first = service.save_issue(conn, _issue(title="kafka rebalance", symptom=sym))
    assert first.duplicate_of is None

    dup = service.save_issue(
        conn, _issue(title="kafka rebalance again", symptom=sym)
    )
    assert dup.duplicate_of == first.id
    assert dup.duplicate_score >= search.DEDUP_THRESHOLD


def test_unrelated_is_not_flagged_duplicate(conn):
    service.save_issue(conn, _issue(symptom="terraform plan drift on lb static ip"))
    other = service.save_issue(
        conn, _issue(symptom="bigquery full table scan cost spike on events")
    )
    assert other.duplicate_of is None


def test_hybrid_beats_single_branch(conn):
    # exact rare token only in A; paraphrase semantics only in B
    service.save_issue(
        conn,
        _issue(
            title="A",
            symptom="ClassCastException SchemaCoercion in OrderEnrich job",
            root_cause="schema mismatch",
        ),
    )
    service.save_issue(
        conn,
        _issue(
            title="B",
            symptom="type conversion error while enriching order records",
        ),
    )
    service.save_issue(conn, _issue(title="C", symptom="unrelated dns timeout"))

    hits = search.recall(conn, "SchemaCoercion type conversion enriching order", "repoA", k=3)
    titles = [h.issue.title for h in hits]
    assert "A" in titles and "B" in titles
    assert "C" not in titles[:2]


def test_scope_filter(conn):
    service.save_issue(conn, _issue(title="inA", symptom="oauth token refresh 403", scope="repoA"))
    service.save_issue(conn, _issue(title="inB", symptom="oauth token refresh 403", scope="repoB"))
    service.save_issue(conn, _issue(title="global", symptom="oauth token refresh 403", scope="global"))

    hits = search.recall(conn, "oauth token refresh 403", "repoA", k=10)
    scopes = {h.issue.scope for h in hits}
    assert scopes <= {"repoA", "global"}
    assert "repoB" not in scopes


def test_helpful_bumps_and_boosts(conn):
    res = service.save_issue(conn, _issue(symptom="redis connection pool exhausted"))
    assert store.bump_helpful(conn, res.id) == 1
    assert store.get_issue(conn, res.id).helpful_count == 1


def test_review_flow_status(conn):
    issue = _issue(status="pending_review", source="agent")
    res = service.save_issue(conn, issue)
    assert store.get_issue(conn, res.id).status == "pending_review"
    store.set_status(conn, res.id, "approved")
    assert store.get_issue(conn, res.id).status == "approved"


def test_fts_query_survives_punctuation(conn):
    service.save_issue(
        conn, _issue(symptom="java.lang.IllegalStateException: closed() at Foo$1.run()")
    )
    # punctuation-heavy query must not raise an FTS5 syntax error
    hits = search.recall(conn, "java.lang.IllegalStateException: closed()", "repoA", k=3)
    assert len(hits) >= 1
