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
runs fully offline after the first download. The model is cached in
`~/.recall/models` (override with `RECALL_MODEL_DIR`) — a persistent dir, not
fastembed's default `$TMPDIR` which the OS reaps. The MCP server warms the model
in a background thread at startup, so the agent's first tool call doesn't block on
the download; `recall warmup` is only needed if you want to pre-fetch for CLI use.

> **Note:** the background warmup only *starts* the download at boot — it doesn't
> guarantee the model is ready. If the agent fires a tool call within the first
> ~20s of the server starting (before the download finishes), that call still
> blocks until the model is ready. The thread just means it's already downloading
> rather than starting cold, turning a guaranteed first-call stall into one that
> only happens if you're very fast off the line — and only once per machine. For a
> hard guarantee, run `recall warmup` once as a post-install step.

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

## Project separation (scope)

Projects are separated **logically inside one shared SQLite file** — not a file
per project. Every row carries a `scope` string, and isolation is enforced at
query time by filtering on it.

**How scope is derived** (in order):

1. git `remote.origin.url` → normalized to `org/repo`
2. else the git top-level directory name
3. else `global` (outside any repo)

Two clones of the same repo therefore share a scope (same remote).

| Operation | Scope behavior |
| --- | --- |
| **Search** | current repo **+ `global`** (shared knowledge surfaces everywhere); `--all` widens to every scope |
| **Dedup**  | exact scope only |
| **Write**  | tagged with the derived scope unless `--scope` / the `scope` arg overrides it |

- `global`-scoped entries **intentionally surface in every project's search** —
  that's the cross-project knowledge channel. Add with `--scope global`.
- For **hard physical isolation** (e.g. client repos that must never share
  memory), point each at its own file via `RECALL_DB=/path/to/that-project.db` —
  same binary, separate stores.
- Edge case: a repo with **no remote** falls back to its directory name, so two
  unrelated dirs sharing a name would collide into one scope. Set `--scope`
  explicitly if that matters.

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
