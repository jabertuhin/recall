# recall

Local issue-memory for coding agents. Stores resolved issues
(**symptom → context → root cause → fix**) in a single SQLite file and retrieves
the closest past cases with **hybrid search** — FTS5 keyword (BM25) + `sqlite-vec`
semantic KNN, fused with Reciprocal Rank Fusion.

Two faces, one store:

- **CLI** — you add/search/review issues directly.
- **MCP server** — any coding agent (Claude Code, Codex, Cursor, …) recalls past
  issues *before* investigating and saves resolutions *after* fixing.

Why hybrid: error text lives in rare literal tokens (codes, class names, paths)
that BM25 nails and embeddings smear; vectors catch paraphrases. RRF combines the
two without normalizing incomparable scores.

## Install

```bash
cd recall
uv sync                 # creates .venv, installs deps
uv run recall warmup    # pre-fetch the embedding model (~130MB, first run only)
uv run recall init
```

Embeddings: `fastembed` → `BAAI/bge-small-en-v1.5` (384-dim, ONNX, **no torch**),
runs fully offline after the first download.

## CLI

```bash
recall add -t "Spark NPE on null partition col" \
  -s "NullPointerException at PartitionWriter.scala:42" \
  --cause "partition column was null in source" \
  --fix "coalesce the column before write" \
  --tags spark,partitioning

recall search "null pointer when writing partitioned output"
recall review            # approve/reject agent-proposed entries
recall list --status pending_review
recall stats
```

Scope: entries are tagged with the current repo (git remote → `org/repo`, else
top-level dir name, else `global`). Searches default to **current repo + global**;
`--all` widens to every scope. Override the store path with `RECALL_DB`.

## MCP (agent integration)

Register the server:

```bash
# Claude Code
claude mcp add recall -- uv run --directory /path/to/recall recall mcp
```

```jsonc
// Codex / other MCP clients: ~/.codex/config.toml or client config
{
  "mcpServers": {
    "recall": { "command": "uv", "args": ["run", "--directory", "/path/to/recall", "recall", "mcp"] }
  }
}
```

Tools exposed: `recall_issues(problem, scope?, k?)`,
`save_resolution(title, symptom, root_cause, fix, context?, tags?, scope?)`,
`mark_helpful(issue_id)`.

**Add this line to your agent's system prompt / project instructions** so it uses
the memory:

> Before investigating any error or unexpected behavior, call `recall_issues`
> with the symptom. After you resolve a non-trivial issue, call `save_resolution`.

## Review gate

Agent-saved entries land as `pending_review` — searchable immediately but flagged
`UNREVIEWED`, and never silently trusted. `recall review` is the human gate that
promotes them to `approved`. Near-duplicate writes return a `duplicate_of` hint so
the store doesn't rot.

## Test

```bash
uv run pytest          # embeddings are stubbed; no model download needed
```
