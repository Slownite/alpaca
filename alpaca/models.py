"""Hugging Face model pull/cache and the local alias registry."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from alpaca.path import path_helper


def _models_dir() -> Path:
    return path_helper("models/")


def _registry_path() -> Path:
    return _models_dir().parent / "registry.json"


def _load_registry() -> list[dict]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_registry(entries: list[dict]) -> None:
    _registry_path().write_text(json.dumps(entries, indent=2))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _upsert(entries: list[dict], entry: dict) -> list[dict]:
    same = lambda e: e.get("repo_id") == entry["repo_id"] and e.get("revision") == entry["revision"]
    return [e for e in entries if not same(e)] + [entry]


def pull(repo_id: str, revision: str | None = None) -> str:
    """Download ``repo_id`` into ``$ALPACA_HOME/models`` and record it in the registry.

    Idempotent: ``huggingface_hub.snapshot_download`` reuses the cache when the
    snapshot is already present, so a re-pull at the same revision is a fast
    no-op that just refreshes the registry timestamp.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    rev = revision or "main"
    models_dir = _models_dir()
    os.environ["HF_HOME"] = str(models_dir)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=rev,
            cache_dir=str(models_dir),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"access to '{repo_id}' is gated; set HF_TOKEN to a token granted access"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"repository '{repo_id}' was not found on Hugging Face") from exc
    except RevisionNotFoundError as exc:
        raise RuntimeError(f"revision '{rev}' was not found for '{repo_id}'") from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise RuntimeError(
                f"authentication required to pull '{repo_id}'; set HF_TOKEN"
            ) from exc
        raise RuntimeError(f"could not pull '{repo_id}': {exc}") from exc

    entry = {
        "alias": repo_id,
        "repo_id": repo_id,
        "revision": rev,
        "path": path,
        "pulled_at": _now_iso(),
    }
    _save_registry(_upsert(_load_registry(), entry))
    return path


def registry() -> list[dict]:
    """Return the list of cached model entries."""
    return _load_registry()


def _find(ref: str) -> dict | None:
    for e in _load_registry():
        if e.get("alias") == ref or e.get("repo_id") == ref:
            return e
    return None


def resolve(ref: str) -> str:
    """Resolve a registry alias, a HF repo id, or a local path to a local path.

    A bare repo id that isn't cached yet is pulled on demand.
    """
    p = Path(ref).expanduser()
    if p.exists():
        return str(p)
    entry = _find(ref)
    if entry and Path(entry["path"]).exists():
        return entry["path"]
    if "/" in ref:
        return pull(ref, revision=entry["revision"] if entry else None)
    raise RuntimeError(f"unknown model reference '{ref}'")


def remove(ref: str) -> bool:
    """Delete cached weights and the registry entry for ``ref``."""
    entries = _load_registry()
    target = next(
        (e for e in entries if e.get("alias") == ref or e.get("repo_id") == ref),
        None,
    )
    if target is None:
        return False
    p = Path(target["path"])
    repo_dir = next(
        (parent for parent in (p, *p.parents) if parent.name.startswith("models--")),
        p,
    )
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    _save_registry([e for e in entries if e is not target])
    return True
