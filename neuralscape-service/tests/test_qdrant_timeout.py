"""The Qdrant request timeout is a setting that reaches the client.

qdrant-client's own default is about 5 s, which a loaded host exceeds on pool scrolls and large
retrieves. ``QDRANT_TIMEOUT`` (seconds, default 30) is carried by ``get_mem0_config`` into the
mem0 fork's ``QdrantConfig`` and on to ``QdrantClient``.
"""

import third_party_telemetry  # noqa: F401  (must precede any mem0 import)
import pytest

from config import Settings


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, google_api_key="k", neo4j_password="x", **kw)


def test_default_is_thirty_seconds(monkeypatch):
    monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)
    assert _settings().qdrant_timeout == 30


def test_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_TIMEOUT", "90")
    assert _settings().qdrant_timeout == 90


@pytest.mark.parametrize("url", ["http://localhost:6333", None])
def test_mem0_config_carries_timeout(url):
    cfg = _settings(qdrant_url=url, qdrant_timeout=47).get_mem0_config()
    assert cfg["vector_store"]["config"]["timeout"] == 47


class _FakeClient:
    seen: list = []

    def __init__(self, **kwargs):
        type(self).seen.append(kwargs)


def _build(monkeypatch, **kwargs):
    import mem0.vector_stores.qdrant as q

    _FakeClient.seen = []
    monkeypatch.setattr(q, "QdrantClient", _FakeClient)
    monkeypatch.setattr(q.Qdrant, "create_col", lambda self, *a, **k: None)
    q.Qdrant(collection_name="c", embedding_model_dims=4, **kwargs)
    return _FakeClient.seen[0]


def test_fork_passes_timeout_to_client(monkeypatch):
    assert _build(monkeypatch, url="http://q:6333", timeout=42)["timeout"] == 42


def test_fork_passes_timeout_in_local_path_mode(monkeypatch):
    params = _build(monkeypatch, path="/tmp/does-not-matter", timeout=42)
    assert params["timeout"] == 42 and params["path"] == "/tmp/does-not-matter"


def test_fork_omits_timeout_when_unset(monkeypatch):
    assert "timeout" not in _build(monkeypatch, url="http://q:6333")


def test_fork_config_accepts_timeout():
    from mem0.configs.vector_stores.qdrant import QdrantConfig

    assert QdrantConfig(url="http://q:6333", api_key="k", timeout=12).timeout == 12
    assert QdrantConfig(path="/tmp/x").timeout is None


def test_zero_and_negative_timeout_rejected(monkeypatch):
    monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            _settings(qdrant_timeout=bad)
    monkeypatch.setenv("QDRANT_TIMEOUT", "0")
    with pytest.raises(ValueError):
        _settings()


def test_default_is_defined_once():
    import config

    assert config.DEFAULT_QDRANT_TIMEOUT_S == 30
    assert _settings().qdrant_timeout == config.DEFAULT_QDRANT_TIMEOUT_S


def test_policy_migration_pool_source_sets_timeout(monkeypatch):
    import qdrant_client

    import config
    import policy_migration as pm

    _FakeClient.seen = []
    monkeypatch.setattr(qdrant_client, "QdrantClient", _FakeClient)
    pm.QdrantPoolSource("http://q:6333", "c")
    pm.QdrantPoolSource("http://q:6333", "c", timeout=7)
    assert [k["timeout"] for k in _FakeClient.seen] == [config.DEFAULT_QDRANT_TIMEOUT_S, 7]


def test_no_async_qdrant_client_is_built_anywhere():
    """The service and fork only build the sync client; if an async one appears it must take the timeout."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    hits = [
        str(p) for base in ("neuralscape-service", "mem0/mem0", "graphiti/graphiti_core")
        if (root / base).is_dir()
        for p in (root / base).rglob("*.py")
        if not any(x.startswith(".") for x in p.parts) and "tests" not in p.parts and "integration_tests" not in p.parts and "scripts" not in p.parts
        and "AsyncQdrantClient" in p.read_text()
    ]
    assert hits == []


def test_health_probe_keeps_its_own_short_bound():
    import main

    assert main._VECTOR_STORE_PROBE_TIMEOUT_S == 2.0
