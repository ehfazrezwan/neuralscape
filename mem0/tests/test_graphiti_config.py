"""Unit tests for Graphiti config and factory registration."""

import pytest
from pydantic import ValidationError

from mem0.graphs.configs import GraphitiConfig, GraphStoreConfig
from mem0.utils.factory import GraphStoreFactory


VALID_CONFIG = {
    "url": "neo4j://127.0.0.1:7687",
    "username": "neo4j",
    "password": "secret",
}


class TestGraphitiConfigValid:
    def test_valid_config(self):
        cfg = GraphitiConfig(**VALID_CONFIG)
        assert cfg.url == "neo4j://127.0.0.1:7687"
        assert cfg.username == "neo4j"
        assert cfg.password == "secret"

    def test_defaults(self):
        cfg = GraphitiConfig(**VALID_CONFIG)
        assert cfg.database == "neo4j"
        assert cfg.graphiti_llm_provider == "gemini"
        assert cfg.graphiti_embedder_provider == "gemini"
        assert cfg.graphiti_reranker_provider is None
        assert cfg.store_raw_episode_content is True
        assert cfg.update_communities is False
        assert cfg.graphiti_llm_model is None
        assert cfg.graphiti_llm_api_key is None
        assert cfg.graphiti_embedder_model is None
        assert cfg.graphiti_embedder_api_key is None

    def test_custom_provider_fields(self):
        cfg = GraphitiConfig(
            **VALID_CONFIG,
            graphiti_llm_provider="openai",
            graphiti_llm_model="gpt-4o",
            graphiti_llm_api_key="sk-test",
            graphiti_embedder_provider="openai",
            graphiti_embedder_model="text-embedding-3-small",
            graphiti_embedder_api_key="sk-test",
            graphiti_reranker_provider="bge",
            database="mydb",
        )
        assert cfg.graphiti_llm_provider == "openai"
        assert cfg.graphiti_llm_model == "gpt-4o"
        assert cfg.graphiti_embedder_provider == "openai"
        assert cfg.graphiti_reranker_provider == "bge"
        assert cfg.database == "mydb"


class TestGraphitiConfigValidation:
    def test_missing_url_raises(self):
        with pytest.raises(ValidationError):
            GraphitiConfig(url="", username="neo4j", password="secret")

    def test_missing_username_raises(self):
        with pytest.raises(ValidationError):
            GraphitiConfig(url="neo4j://localhost:7687", username="", password="secret")

    def test_missing_password_raises(self):
        with pytest.raises(ValidationError):
            GraphitiConfig(url="neo4j://localhost:7687", username="neo4j", password="")


class TestGraphStoreConfigProvider:
    def test_graphiti_provider_resolves(self):
        gsc = GraphStoreConfig(provider="graphiti", config=VALID_CONFIG)
        assert isinstance(gsc.config, GraphitiConfig)
        assert gsc.config.url == VALID_CONFIG["url"]

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValidationError):
            GraphStoreConfig(provider="nonexistent", config=VALID_CONFIG)


class TestGraphStoreFactory:
    def test_graphiti_registered_in_factory(self):
        assert "graphiti" in GraphStoreFactory.provider_to_class

    def test_correct_class_path(self):
        assert GraphStoreFactory.provider_to_class["graphiti"] == "mem0.memory.graphiti_memory.MemoryGraph"
