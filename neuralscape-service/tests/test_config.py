"""Tests for config.py — the AI Studio ↔ LLM-gateway provider toggle.

The toggle is a single flag (``llm_gateway_enabled``) that flips the mem0 LLM +
embedder (and the graphiti llm/embedder/reranker) between Google AI Studio
(``gemini`` provider) and an OpenAI-compatible gateway (``openai`` provider with
the gateway base_url + key). graphiti's OpenAI clients read the base_url from
``OPENAI_BASE_URL``, which get_mem0_config sets when the flag is on.
"""

import os

import pytest

from config import Settings


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip provider/cred env vars before each test in this module.

    ``graphiti_core`` calls ``load_dotenv()`` at import time (graphiti.py /
    helpers.py / driver.py), leaking the repo ``.env`` into ``os.environ``.
    ``_settings(_env_file=None, ...)`` disables the .env file but pydantic still
    reads ``os.environ``, so a leaked ``LLM_GATEWAY_BASE_URL`` would defeat the
    "missing creds → error" assertions. Clearing them makes ``_settings`` truly
    hermetic — kwargs and class defaults fully control the Settings under test.
    """
    for var in (
        "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "LLM_GATEWAY_ENABLED", "LLM_GATEWAY_BASE_URL", "LLM_GATEWAY_API_KEY",
        "LLM_GATEWAY_GRAPHITI_ENABLED",
        "LLM_GATEWAY_LLM_MODEL", "LLM_GATEWAY_LLM_FALLBACK_MODEL",
        "LLM_GATEWAY_EMBEDDER_MODEL", "LLM_GATEWAY_GRAPHITI_MODEL",
        "GOOGLE_API_KEY", "NEO4J_PASSWORD", "GRAPH_QUEUE_NAME",
        "GEMINI_LLM_MODEL", "GEMINI_LLM_FALLBACK_MODEL", "GEMINI_EMBEDDER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**kw) -> Settings:
    # Hermetic: don't read a real .env so the test only sees its kwargs.
    return Settings(_env_file=None, **kw)


class TestProviderToggle:
    def test_default_uses_gemini_direct(self):
        cfg = _settings(llm_gateway_enabled=False).get_mem0_config()
        assert cfg["llm"]["provider"] == "gemini"
        assert cfg["embedder"]["provider"] == "gemini"
        gs = cfg["graph_store"]["config"]
        assert gs["graphiti_llm_provider"] == "gemini"
        assert gs["graphiti_embedder_provider"] == "gemini"
        assert gs["graphiti_reranker_provider"] == "gemini"

    def test_gateway_routes_through_openai_compatible(self):
        prev = os.environ.get("OPENAI_BASE_URL")
        try:
            cfg = _settings(
                llm_gateway_enabled=True,
                llm_gateway_base_url="https://gw.example.com",
                llm_gateway_api_key="k",
                llm_gateway_llm_model="gemini-3.1-flash-lite",
                llm_gateway_embedder_model="google-vertex/gemini-embedding-001",
            ).get_mem0_config()

            assert cfg["llm"]["provider"] == "openai"
            assert cfg["llm"]["config"]["openai_base_url"] == "https://gw.example.com/v1"
            assert cfg["llm"]["config"]["model"] == "gemini-3.1-flash-lite"
            assert cfg["llm"]["config"]["api_key"] == "k"

            emb = cfg["embedder"]["config"]
            assert cfg["embedder"]["provider"] == "openai"
            assert emb["model"] == "google-vertex/gemini-embedding-001"
            assert emb["embedding_dims"] == 768
            assert emb["openai_base_url"] == "https://gw.example.com/v1"

            # graphiti stays on AI Studio (gemini) even in gateway mode: the mem0
            # graphiti adapter can't yet configure it for the Vertex gateway
            # (batched embeddings rejected, hardcoded gpt-4.1-nano small_model,
            # OpenAIRerankerClient(api_key=) init bug). mem0's vector path above
            # is what routes through the gateway and stabilizes the API.
            gs = cfg["graph_store"]["config"]
            assert gs["graphiti_llm_provider"] == "gemini"
            assert gs["graphiti_embedder_provider"] == "gemini"
            assert gs["graphiti_reranker_provider"] == "gemini"

            # OPENAI_BASE_URL is still set (mem0's openai clients + a future
            # graphiti-on-gateway both rely on it).
            assert os.environ["OPENAI_BASE_URL"] == "https://gw.example.com/v1"
        finally:
            if prev is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = prev

    def test_graphiti_gateway_flag_routes_graphiti_through_gateway(self):
        # LLM_GATEWAY_GRAPHITI_ENABLED=True opts graphiti into the gateway too.
        prev = os.environ.get("OPENAI_BASE_URL")
        try:
            gs = _settings(
                llm_gateway_enabled=True,
                llm_gateway_graphiti_enabled=True,
                llm_gateway_base_url="https://gw.example.com",
                llm_gateway_api_key="k",
            ).get_mem0_config()["graph_store"]["config"]
            assert gs["graphiti_llm_provider"] == "openai"
            assert gs["graphiti_reranker_provider"] == "openai"
            # main vs small model are decoupled and both google-vertex/ prefixed
            assert gs["graphiti_llm_model"] == "google-vertex/gemini-3.1-flash-lite"
            assert gs["graphiti_llm_small_model"] == "google-vertex/gemini-3.1-flash-lite"
            # embedder stays on AI Studio (Vertex rejects batched embeds)
            assert gs["graphiti_embedder_provider"] == "gemini"
        finally:
            if prev is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = prev

    def test_gateway_base_url_v1_not_double_appended(self):
        prev = os.environ.get("OPENAI_BASE_URL")
        try:
            cfg = _settings(
                llm_gateway_enabled=True,
                llm_gateway_base_url="https://gw.example.com/v1/",
                llm_gateway_api_key="k",
            ).get_mem0_config()
            assert cfg["llm"]["config"]["openai_base_url"] == "https://gw.example.com/v1"
        finally:
            if prev is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = prev

    def test_validate_required_gateway_creds(self):
        # Gateway enabled but missing gateway creds → error.
        with pytest.raises(ValueError, match="LLM_GATEWAY_BASE_URL"):
            _settings(
                llm_gateway_enabled=True, google_api_key="g", neo4j_password="p"
            ).validate_required()
        # Gateway enabled with creds → ok (and GOOGLE_API_KEY still required).
        _settings(
            llm_gateway_enabled=True,
            llm_gateway_base_url="https://gw",
            llm_gateway_api_key="k",
            google_api_key="g",
            neo4j_password="p",
        ).validate_required()
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            _settings(
                llm_gateway_enabled=True,
                llm_gateway_base_url="https://gw",
                llm_gateway_api_key="k",
                neo4j_password="p",
            ).validate_required()

    def test_embedding_dims_stay_768_both_modes(self):
        # Existing Qdrant collection is 768-dim; both routes must keep that.
        off = _settings(llm_gateway_enabled=False).get_mem0_config()
        assert off["vector_store"]["config"]["embedding_model_dims"] == 768
        assert off["embedder"]["config"]["embedding_dims"] == 768


class TestDictatorRole:
    def test_flags_default_off(self):
        cfg = _settings()
        assert cfg.standards_enabled is False
        assert cfg.processes_enabled is False
        assert cfg.dictator_user_ids == ""
        assert cfg.dictator_user_ids_set() == set()
        assert cfg.is_dictator("mark") is False

    def test_dictator_csv_parsed(self):
        cfg = _settings(dictator_user_ids=" mark , alice ,")
        assert cfg.dictator_user_ids_set() == {"mark", "alice"}
        assert cfg.is_dictator("mark") is True
        assert cfg.is_dictator("alice") is True
        assert cfg.is_dictator("bob") is False

    def test_is_dictator_none_or_empty(self):
        cfg = _settings(dictator_user_ids="mark")
        assert cfg.is_dictator(None) is False
        assert cfg.is_dictator("") is False


class TestCodeRepos:
    """E2 config seam: the native code-intel engine resolves repo:<name> refs
    through settings.code_repos. Without this field it was always {} and the
    native engine was unreachable in production (Copilot I1 #5)."""

    def test_default_empty(self):
        cfg = _settings()
        assert cfg.code_repos == {}

    def test_dict_passed_through(self):
        cfg = _settings(code_repos={"myrepo": "/abs/path"})
        assert cfg.code_repos.get("myrepo") == "/abs/path"

    def test_env_json_parsed(self, monkeypatch):
        monkeypatch.setenv("CODE_REPOS", '{"r1":"/p1","r2":"/p2"}')
        cfg = _settings()
        assert cfg.code_repos == {"r1": "/p1", "r2": "/p2"}
