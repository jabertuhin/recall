"""Store path resolution and scope derivation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def store_path() -> Path:
    """Resolve the SQLite store path.

    Precedence: ``RECALL_DB`` env var, else ``~/.recall/recall.db``.
    """
    env = os.environ.get("RECALL_DB")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".recall" / "recall.db"


def current_scope(cwd: Path | None = None) -> str:
    """Derive a scope id for the current directory.

    Uses the git remote URL if available, else the repo's top-level directory
    name. Falls back to ``"global"`` outside a repo.
    """
    cwd = cwd or Path.cwd()
    try:
        remote = subprocess.run(
            ["git", "-C", str(cwd), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            return _normalize_remote(remote.stdout.strip())

        toplevel = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if toplevel.returncode == 0 and toplevel.stdout.strip():
            return Path(toplevel.stdout.strip()).name
    except (subprocess.SubprocessError, OSError):
        pass
    return "global"


def _normalize_remote(url: str) -> str:
    """Reduce a git remote URL to a stable ``org/repo`` scope id."""
    url = url.removesuffix(".git")
    # git@github.com:org/repo  ->  org/repo
    if ":" in url and "//" not in url:
        url = url.split(":", 1)[1]
    else:  # https://host/org/repo -> org/repo
        parts = url.split("/")
        url = "/".join(parts[-2:]) if len(parts) >= 2 else url
    return url
