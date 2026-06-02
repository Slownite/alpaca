import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpaca import models


@pytest.fixture
def alpaca_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_snapshot(monkeypatch):
    """Mock huggingface_hub.snapshot_download to materialize a fake HF cache layout."""
    calls: list[dict] = []

    def _fake(repo_id, revision=None, cache_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir})
        cache = Path(cache_dir)
        snap = (
            cache
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / (revision or "main")
        )
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text("{}")
        return str(snap)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake)
    return calls


# --- pull ---


def test_pull_downloads_into_models_dir(alpaca_home, fake_snapshot):
    path = models.pull("meta-llama/Meta-Llama-3-8B-Instruct")
    assert Path(path).exists()
    assert Path(path).is_relative_to(alpaca_home / "models")


def test_pull_writes_registry_entry(alpaca_home, fake_snapshot):
    path = models.pull("meta-llama/Meta-Llama-3-8B-Instruct", revision="main")
    reg = json.loads((alpaca_home / "registry.json").read_text())
    assert len(reg) == 1
    e = reg[0]
    assert e["repo_id"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert e["revision"] == "main"
    assert e["path"] == path
    assert e["pulled_at"]


def test_pull_default_revision_is_main(alpaca_home, fake_snapshot):
    models.pull("foo/bar")
    assert fake_snapshot[0]["revision"] == "main"
    assert json.loads((alpaca_home / "registry.json").read_text())[0]["revision"] == "main"


def test_pull_is_idempotent_same_revision(alpaca_home, fake_snapshot):
    p1 = models.pull("foo/bar")
    p2 = models.pull("foo/bar")
    assert p1 == p2
    reg = json.loads((alpaca_home / "registry.json").read_text())
    assert len(reg) == 1


def test_pull_uses_hf_cache_no_redownload(alpaca_home, fake_snapshot):
    """The second pull at the same revision returns the existing snapshot path.

    The HF library guarantees no re-download when blobs match; we just verify
    that alpaca passes the same cache_dir on both calls.
    """
    models.pull("foo/bar")
    models.pull("foo/bar")
    assert fake_snapshot[0]["cache_dir"] == fake_snapshot[1]["cache_dir"]


def test_pull_different_revisions_keeps_both(alpaca_home, fake_snapshot):
    models.pull("foo/bar", revision="v1")
    models.pull("foo/bar", revision="v2")
    reg = json.loads((alpaca_home / "registry.json").read_text())
    assert {e["revision"] for e in reg} == {"v1", "v2"}


def test_pull_sets_hf_home(alpaca_home, fake_snapshot, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    models.pull("foo/bar")
    import os

    assert Path(os.environ["HF_HOME"]) == alpaca_home / "models"


# --- resolve ---


def test_resolve_local_path(alpaca_home, fake_snapshot, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    assert models.resolve(str(local)) == str(local)


def test_resolve_alias_uses_cached_path(alpaca_home, fake_snapshot):
    p = models.pull("foo/bar")
    fake_snapshot.clear()
    assert models.resolve("foo/bar") == p
    assert fake_snapshot == []  # cached → no second snapshot call


def test_resolve_repo_id_pulls_on_demand(alpaca_home, fake_snapshot):
    assert not (alpaca_home / "registry.json").exists()
    p = models.resolve("foo/bar")
    assert Path(p).exists()
    assert (alpaca_home / "registry.json").exists()


def test_resolve_unknown_bare_ref_raises(alpaca_home):
    with pytest.raises(RuntimeError, match="unknown model reference"):
        models.resolve("no-slash-here")


# --- registry ---


def test_registry_empty_when_no_pulls(alpaca_home):
    assert models.registry() == []


def test_registry_lists_pulled_entries(alpaca_home, fake_snapshot):
    models.pull("foo/bar")
    models.pull("baz/qux", revision="v1")
    repos = {e["repo_id"] for e in models.registry()}
    assert repos == {"foo/bar", "baz/qux"}


# --- remove ---


def test_remove_deletes_weights_and_entry(alpaca_home, fake_snapshot):
    p = models.pull("foo/bar")
    repo_dir = Path(p).parent.parent  # snapshot/<rev> → snapshots → models--foo--bar
    assert repo_dir.exists()
    assert models.remove("foo/bar") is True
    assert not repo_dir.exists()
    assert models.registry() == []


def test_remove_returns_false_when_unknown(alpaca_home):
    assert models.remove("unknown/repo") is False


# --- error surfacing ---


def test_gated_repo_error_message(alpaca_home, monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    def _raises(*a, **kw):
        raise GatedRepoError("denied", response=MagicMock(status_code=403))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _raises)
    with pytest.raises(RuntimeError, match="gated"):
        models.pull("meta-llama/foo")


def test_repository_not_found_error_message(alpaca_home, monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    def _raises(*a, **kw):
        raise RepositoryNotFoundError("missing", response=MagicMock(status_code=404))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _raises)
    with pytest.raises(RuntimeError, match="not found"):
        models.pull("nope/nope")


def test_revision_not_found_error_message(alpaca_home, monkeypatch):
    from huggingface_hub.errors import RevisionNotFoundError

    def _raises(*a, **kw):
        raise RevisionNotFoundError("bad rev", response=MagicMock(status_code=404))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _raises)
    with pytest.raises(RuntimeError, match="revision"):
        models.pull("foo/bar", revision="nope")


def test_auth_error_message(alpaca_home, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    def _raises(*a, **kw):
        raise HfHubHTTPError("unauth", response=MagicMock(status_code=401))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _raises)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        models.pull("foo/bar")
