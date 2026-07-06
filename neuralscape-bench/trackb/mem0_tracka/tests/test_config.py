"""Tests for config module."""

import os
from pathlib import Path

import pytest

from trackb.mem0_tracka.config import (
    ANSWER_MODEL,
    BACKBONE_MODEL,
    EMBEDDER_MODEL,
    JUDGE_MODEL,
    Mem0Config,
    get_config,
)


def test_locked_models():
    """Verify locked model names for controlled comparison."""
    assert BACKBONE_MODEL == "gemini-flash-1.5"
    assert EMBEDDER_MODEL == "gemini-embedding-001"
    assert JUDGE_MODEL == "gemini-3.1-flash-lite"
    assert ANSWER_MODEL == "gemini-3.1-flash-lite"


def test_mem0_config_dict():
    """Test config dict generation for mem0.Memory."""
    cfg = Mem0Config(api_key="test-key", vector_store_path=Path("/tmp/test"))
    config_dict = cfg.to_mem0_dict()

    assert config_dict["llm"]["provider"] == "google-genai"
    assert config_dict["llm"]["config"]["model"] == BACKBONE_MODEL
    assert config_dict["llm"]["config"]["api_key"] == "test-key"
    assert config_dict["llm"]["config"]["temperature"] == 0.0

    assert config_dict["embedder"]["provider"] == "google-genai"
    assert config_dict["embedder"]["config"]["model"] == EMBEDDER_MODEL

    assert config_dict["vector_store"]["provider"] == "qdrant"
    assert config_dict["vector_store"]["config"]["collection_name"] == "mem0_tracka"
    assert config_dict["vector_store"]["config"]["path"] == "/tmp/test"


def test_get_config_missing_api_key(monkeypatch):
    """Should raise if GOOGLE_API_KEY not set."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_config()


def test_get_config_with_api_key(monkeypatch):
    """Should build config with API key from env."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    cfg = get_config()
    assert cfg.api_key == "test-key-123"
    assert cfg.vector_store_path.name == ".mem0_tracka_qdrant"


def test_get_config_custom_path(monkeypatch):
    """Should accept custom vector store path."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    custom_path = Path("/custom/path")
    cfg = get_config(vector_store_path=custom_path)
    assert cfg.vector_store_path == custom_path
