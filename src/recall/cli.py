"""Typer CLI for recall.

Commands::

    recall init                 # create the store
    recall add                  # add an issue (manual, approved by default)
    recall search "<problem>"   # hybrid recall
    recall review               # approve/edit/reject pending agent entries
    recall list                 # list recent issues
    recall show <id>            # full record
    recall helpful <id>         # feedback signal
    recall rm <id>              # delete
    recall stats                # store summary
    recall warmup               # pre-fetch the embedding model
    recall mcp                  # run the MCP server (stdio)
"""

from __future__ import annotations

from typing import Optional

import typer

from . import config, service, store
from . import search as search_mod
from .models import Issue

app = typer.Typer(
    add_completion=False,
    help="Local issue-memory for coding agents (hybrid keyword + semantic recall).",
    no_args_is_help=True,
)


def _conn():
    return store.connect(config.store_path())


def _print_hit(hit, idx: int) -> None:
    i = hit.issue
    typer.secho(f"\n[{idx}] {i.title}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"    id={i.id}  scope={i.scope}  score={hit.score:.4f}")
    note = hit.trust_note()
    if note:
        typer.secho(f"    {note}", fg=typer.colors.YELLOW)
    typer.echo(f"    symptom : {i.symptom.strip()[:200]}")
    if i.root_cause:
        typer.echo(f"    cause   : {i.root_cause.strip()[:200]}")
    if i.fix:
        typer.secho(f"    fix     : {i.fix.strip()[:300]}", fg=typer.colors.GREEN)


@app.command()
def init() -> None:
    """Create the store (idempotent) and report its path."""
    conn = _conn()
    conn.close()
    typer.secho(f"store ready at {config.store_path()}", fg=typer.colors.GREEN)


@app.command()
def add(
    title: str = typer.Option(..., "--title", "-t"),
    symptom: str = typer.Option(..., "--symptom", "-s", help="error / observed behavior"),
    context: Optional[str] = typer.Option(None, "--context", "-c"),
    cause: Optional[str] = typer.Option(None, "--cause"),
    fix: Optional[str] = typer.Option(None, "--fix", "-f"),
    tags: Optional[str] = typer.Option(None, "--tags", help="comma-separated"),
    scope: Optional[str] = typer.Option(None, "--scope", help="default: current repo"),
) -> None:
    """Add an issue directly (manual, approved)."""
    conn = _conn()
    issue = Issue(
        title=title,
        symptom=symptom,
        context=context,
        root_cause=cause,
        fix=fix,
        tags=[t.strip() for t in tags.split(",")] if tags else [],
        scope=scope or config.current_scope(),
        status="approved",
        source="manual",
    )
    result = service.save_issue(conn, issue)
    typer.secho(f"saved issue #{result.id} (scope={issue.scope})", fg=typer.colors.GREEN)
    if result.duplicate_of:
        typer.secho(
            f"  note: similar to #{result.duplicate_of} "
            f"(sim={result.duplicate_score}); consider `recall rm` if redundant",
            fg=typer.colors.YELLOW,
        )
    conn.close()


@app.command()
def search(
    problem: str = typer.Argument(..., help="describe the problem / paste the error"),
    scope: Optional[str] = typer.Option(None, "--scope"),
    all_scopes: bool = typer.Option(False, "--all", help="search every scope"),
    k: int = typer.Option(5, "--k"),
) -> None:
    """Hybrid recall of past issues."""
    conn = _conn()
    s = None if all_scopes else (scope or config.current_scope())
    hits = search_mod.recall(conn, problem, s, k)
    if not hits:
        typer.echo("no matches.")
    for idx, hit in enumerate(hits, start=1):
        _print_hit(hit, idx)
    conn.close()


@app.command(name="list")
def list_cmd(
    status: Optional[str] = typer.Option(None, "--status", help="pending_review|approved"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recent issues."""
    conn = _conn()
    for i in store.list_issues(conn, status=status, limit=limit):
        flag = "⏳" if i.status == "pending_review" else "  "
        typer.echo(f"{flag} #{i.id:>4}  [{i.scope}]  {i.title}")
    conn.close()


@app.command()
def show(issue_id: int = typer.Argument(...)) -> None:
    """Print a full issue record."""
    conn = _conn()
    i = store.get_issue(conn, issue_id)
    if not i:
        typer.secho(f"no issue #{issue_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"#{i.id}  {i.title}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"scope={i.scope} status={i.status} source={i.source} "
               f"helpful={i.helpful_count} created={i.created_at}")
    typer.echo(f"\nSYMPTOM\n{i.symptom}")
    if i.context:
        typer.echo(f"\nCONTEXT\n{i.context}")
    if i.root_cause:
        typer.echo(f"\nROOT CAUSE\n{i.root_cause}")
    if i.fix:
        typer.echo(f"\nFIX\n{i.fix}")
    if i.tags:
        typer.echo(f"\ntags: {', '.join(i.tags)}")
    conn.close()


@app.command()
def review() -> None:
    """Walk pending agent-proposed entries: approve / reject / skip."""
    conn = _conn()
    pending = store.list_issues(conn, status="pending_review", limit=100)
    if not pending:
        typer.echo("nothing pending review.")
        return
    for i in pending:
        typer.secho(f"\n#{i.id}  {i.title}", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"  symptom: {i.symptom[:200]}")
        if i.root_cause:
            typer.echo(f"  cause  : {i.root_cause[:200]}")
        if i.fix:
            typer.echo(f"  fix    : {i.fix[:200]}")
        choice = typer.prompt("  [a]pprove / [r]eject / [s]kip", default="s")
        if choice.lower().startswith("a"):
            store.set_status(conn, i.id, "approved")
            typer.secho("  approved", fg=typer.colors.GREEN)
        elif choice.lower().startswith("r"):
            store.delete_issue(conn, i.id)
            typer.secho("  rejected", fg=typer.colors.RED)
    conn.close()


@app.command()
def helpful(issue_id: int = typer.Argument(...)) -> None:
    """Mark an issue as having helped (feedback signal)."""
    conn = _conn()
    count = store.bump_helpful(conn, issue_id)
    typer.secho(f"#{issue_id} helpful_count={count}", fg=typer.colors.GREEN)
    conn.close()


@app.command()
def rm(issue_id: int = typer.Argument(...)) -> None:
    """Delete an issue."""
    conn = _conn()
    store.delete_issue(conn, issue_id)
    typer.secho(f"deleted #{issue_id}", fg=typer.colors.GREEN)
    conn.close()


@app.command()
def stats() -> None:
    """Show store summary."""
    conn = _conn()
    s = store.stats(conn)
    typer.echo(f"total: {s['total']}  pending_review: {s['pending_review']}")
    for scope, count in s["scopes"].items():
        typer.echo(f"  {scope}: {count}")
    conn.close()


@app.command()
def warmup() -> None:
    """Pre-fetch and load the embedding model (~130MB on first run)."""
    from . import embed

    typer.echo("downloading / loading embedding model ...")
    embed.warmup()
    typer.secho("ready.", fg=typer.colors.GREEN)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio."""
    from .mcp_server import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    app()
