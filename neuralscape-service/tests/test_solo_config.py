"""Tests for the NS_MODE deployment profile (solo engine, unit 1).

Two invariants dominate (docs/neuralscape/28-solo-engine.md):

- ``team`` mode (the default) is bit-identical to the pre-NS_MODE behavior —
  same resolved backends, same required-config gates;
- ``solo`` mode boots with only GOOGLE_API_KEY, resolves to the embedded
  backends, and rejects hybrid topologies (a solo daemon pointed at team
  services) as config errors rather than silent half-hybrids.
"""

import pytest

from config import Settings


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip mode/backend env vars so kwargs and class defaults fully control
    the Settings under test (same pattern as test_config.py: graphiti_core's
    import-time load_dotenv() leaks the repo .env into os.environ)."""
    for var in (
        "NS_MODE", "GRAPH_PROVIDER", "TASK_BACKEND", "SCHEDULER_MODE",
        "KUZU_PATH", "QDRANT_URL", "REDIS_URL", "DOCLING_ENABLED",
        "GOOGLE_API_KEY", "NEO4J_PASSWORD", "NEO4J_URI",
        "LLM_GATEWAY_ENABLED", "LLM_GATEWAY_BASE_URL", "LLM_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**kw) -> Settings:
    # Hermetic: don't read a real .env so the test only sees its kwargs.
    return Settings(_env_file=None, **kw)


class TestTeamModeUnchanged:
    def test_defaults_resolve_to_team_backends(self):
        s = _settings()
        assert s.ns_mode == "team"
        assert s.graph_provider == "neo4j"
        assert s.task_backend == "redis"
        assert s.scheduler_mode == "off"
        assert s.docling_enabled is True

    def test_team_still_requires_neo4j_and_redis(self):
        s = _settings(google_api_key="k", neo4j_password="", redis_url="")
        with pytest.raises(ValueError) as exc:
            s.validate_required()
        msg = str(exc.value)
        assert "NEO4J_PASSWORD" in msg
        assert "REDIS_URL" in msg

    def test_team_valid_config_passes(self):
        s = _settings(google_api_key="k", neo4j_password="pw")
        s.validate_required()

    def test_team_rejects_solo_only_backends(self):
        with pytest.raises(ValueError, match="solo-only"):
            _settings(graph_provider="kuzu")
        with pytest.raises(ValueError, match="solo-only"):
            _settings(task_backend="inline")
        with pytest.raises(ValueError, match="solo-only"):
            _settings(scheduler_mode="inproc")

    def test_team_mem0_config_carries_provider_fields(self):
        cfg = _settings(google_api_key="k", neo4j_password="pw").get_mem0_config()
        gs = cfg["graph_store"]["config"]
        assert gs["graph_provider"] == "neo4j"
        assert gs["url"]  # neo4j connection block intact


class TestSoloProfile:
    def test_solo_defaults(self):
        s = _settings(ns_mode="solo")
        assert s.graph_provider == "kuzu"
        assert s.task_backend == "inline"
        assert s.scheduler_mode == "inproc"
        assert s.docling_enabled is False

    def test_solo_validates_with_only_api_key(self):
        _settings(ns_mode="solo", google_api_key="k").validate_required()

    def test_solo_still_requires_api_key(self):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            _settings(ns_mode="solo").validate_required()

    def test_solo_neo4j_fallback_reinstates_neo4j_gates(self):
        s = _settings(ns_mode="solo", graph_provider="neo4j", google_api_key="k")
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            s.validate_required()
        _settings(
            ns_mode="solo", graph_provider="neo4j",
            google_api_key="k", neo4j_password="pw",
        ).validate_required()

    def test_solo_explicit_docling_override_respected(self):
        assert _settings(ns_mode="solo", docling_enabled=True).docling_enabled is True

    def test_invalid_values_rejected(self):
        with pytest.raises(ValueError, match="NS_MODE"):
            _settings(ns_mode="duo")
        with pytest.raises(ValueError, match="GRAPH_PROVIDER"):
            _settings(graph_provider="falkordb")
        with pytest.raises(ValueError, match="TASK_BACKEND"):
            _settings(task_backend="celery")
        with pytest.raises(ValueError, match="SCHEDULER_MODE"):
            _settings(scheduler_mode="cron")


class TestHybridGuardrails:
    def test_solo_rejects_qdrant_server(self):
        with pytest.raises(ValueError, match="QDRANT_URL"):
            _settings(ns_mode="solo", qdrant_url="http://localhost:6333")

    def test_solo_rejects_explicit_redis(self):
        with pytest.raises(ValueError, match="REDIS_URL"):
            _settings(ns_mode="solo", redis_url="redis://localhost:6379")

    def test_solo_cleared_redis_url_allowed(self):
        # REDIS_URL="" means "cleared", not "pointed at a server" — must not
        # trip the guardrail (installers may blank the var rather than drop it).
        s = _settings(ns_mode="solo", redis_url="")
        assert s.task_backend == "inline"

    def test_solo_rejects_redis_task_backend(self):
        with pytest.raises(ValueError, match="TASK_BACKEND=inline"):
            _settings(ns_mode="solo", task_backend="redis")


class TestMem0ConfigProviderFields:
    def test_solo_kuzu_config_builds_and_carries_provider_fields(self):
        cfg = _settings(ns_mode="solo", google_api_key="k").get_mem0_config()
        gs = cfg["graph_store"]["config"]
        assert gs["graph_provider"] == "kuzu"
        assert gs["kuzu_path"].endswith("graph.kuzu")
        assert "~" not in gs["kuzu_path"]  # expanded

    def test_solo_neo4j_fallback_builds_config_with_embedded_qdrant(self):
        cfg = _settings(
            ns_mode="solo", graph_provider="neo4j",
            google_api_key="k", neo4j_password="pw",
        ).get_mem0_config()
        vs = cfg["vector_store"]["config"]
        assert "path" in vs and "url" not in vs  # embedded Qdrant enforced
        assert cfg["graph_store"]["config"]["graph_provider"] == "neo4j"
