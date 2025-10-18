"""Runtime helpers shared across the pipeline.

The goal is to keep small, explicit utilities that all entrypoints can
reuse instead of duplicating logic for run discovery, paths and timestamps.
Nothing here is fancy on purpose: plain functions, no classes.
"""

import os
from datetime import datetime, timezone
from pathlib import Path


def repo_root():
    """Return the repository root (resolved Path)."""

    return Path(__file__).resolve().parents[1]


def exports_root(*parts):
    """Return the exports/ directory joined with optional sub-parts."""

    base = repo_root() / "exports"
    for part in parts:
        base = base / part
    return base


def run_id_from_env():
    """Pick the first run identifier found in the usual environment vars."""

    for key in ("ORCH_RUN_ID", "RUN_ID", "EZSMAD_RUN_ID"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    return None


def timestamp_run_id(dt=None):
    """Return a UTC timestamp in the repository RUN_ID format."""

    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def list_run_dirs(root):
    """Return the list of run sub-directories inside ``root``.

    ``root`` can be absolute or relative to the repository root.
    """

    base = Path(root)
    if not base.is_absolute():
        base = repo_root() / base
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def latest_run_id(*roots):
    """Return the newest run identifier across the provided roots."""

    best_id = None
    best_mtime = -1.0
    for root in roots:
        for d in list_run_dirs(root):
            try:
                mtime = d.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_id = d.name
    return best_id


def resolve_run_id(preferred=None, search_roots=None, allow_new=False):
    """Resolve a run identifier.

    Priority:
    1. ``preferred`` argument (if truthy)
    2. Environment variables (``ORCH_RUN_ID`` → ``RUN_ID`` → ``EZSMAD_RUN_ID``)
    3. Latest run folder found under any of ``search_roots``
    4. Fresh timestamp if ``allow_new`` is True
    """

    run_id = preferred or run_id_from_env()
    if not run_id and search_roots:
        run_id = latest_run_id(*search_roots)
    if not run_id and allow_new:
        run_id = timestamp_run_id()
    if not run_id:
        raise RuntimeError("Unable to resolve RUN_ID")
    return run_id


def ensure_dir(path, *parts):
    """Create a directory (``path`` may be relative to the repo root)."""

    base = Path(path)
    if not base.is_absolute():
        base = repo_root() / base
    for part in parts:
        base = base / part
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_run_dir(root, run_id):
    """Helper to create ``<root>/<run_id>`` and return the Path."""

    return ensure_dir(root, run_id)
