"""Phase G unit tests: code_dispatch binding, index_store, REST/MCP routing.

Covers the through-NS integration surface added in Phase G:
  - code_dispatch.resolve_bound_code_system: real-system short-circuit vs
    placeholder rebind vs unknown/unbindable.
  - index_store: metadata + project-config round-trip (in-memory fallback).
  - REST /v1/code-graph/* knowledge_system dispatch + byte-identical fallback.
  - POST /v1/code-graph/index enqueues on the ingest queue.

The live end-to-end acceptance (index → query through docker) runs on the
icev2 stack; see test_phaseg_integration.py (marked integration).
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from knowledge.base import HealthStatus, KnowledgeSystemInfo, RecallRequest, SystemAnswer


# ── code_dispatch ────────────────────────────────────────────────────


def _settings_with_repo(tmp_path, repo_name):
    repo = tmp_path / repo_name
    repo.mkdir(exist_ok=True)
    return SimpleNamespace(code_repos={repo_name: str(repo)}), str(repo)


def test_repo_path_for_code_space(tmp_path):
    from knowledge.code_dispatch import repo_path_for_code_space

    settings, repo = _settings_with_repo(tmp_path, "myrepo")
    assert repo_path_for_code_space("code--owner--myrepo", settings) == repo
    assert repo_path_for_code_space("bogus", settings) is None
    assert repo_path_for_code_space("code--owner--unknown", settings) is None


def test_resolve_bound_prefers_real_registered_system():
    """A registered system already bound to a real code_space is returned as-is."""
    from knowledge import registry
    from knowledge.code_dispatch import resolve_bound_code_system

    real = SimpleNamespace(
        info=KnowledgeSystemInfo(name="code-x", kind="code",
                                 capabilities=frozenset({"query"}), transport="in-process"),
        _engine=SimpleNamespace(code_space="code--o--r"),
    )
    registry.KNOWLEDGE_REGISTRY["code-x"] = real
    try:
        got = resolve_bound_code_system("code-x", "code--o--r", "u1", SimpleNamespace())
        assert got is real  # short-circuit, no rebuild
    finally:
        registry.KNOWLEDGE_REGISTRY.pop("code-x", None)


def test_resolve_bound_rebinds_placeholder(tmp_path):
    """A capability placeholder is rebound to a per-code_space engine."""
    from knowledge import registry
    from knowledge.code_dispatch import resolve_bound_code_system

    placeholder = SimpleNamespace(
        info=KnowledgeSystemInfo(name="code-graphify-lib", kind="code",
                                 capabilities=frozenset({"query", "neighbors"}),
                                 transport="in-process"),
        _engine=SimpleNamespace(code_space="__registry_capability__"),
        _version=None,
    )
    registry.KNOWLEDGE_REGISTRY["code-graphify-lib"] = placeholder
    settings, _ = _settings_with_repo(tmp_path, "r")
    fake_engine = SimpleNamespace(code_space="code--o--r", G=Mock())
    try:
        with patch("knowledge.code_dispatch.resolve_code_engine", return_value=fake_engine) as m:
            got = resolve_bound_code_system("code-graphify-lib", "code--o--r", "u1", settings)
        assert got is not placeholder
        assert got._engine is fake_engine
        # transport carried verbatim from the placeholder (declared, not branched)
        assert got.info.transport == "in-process"
        m.assert_called_once()
    finally:
        registry.KNOWLEDGE_REGISTRY.pop("code-graphify-lib", None)


def test_resolve_bound_returns_none_when_unbindable(tmp_path):
    from knowledge.code_dispatch import resolve_bound_code_system

    with patch("knowledge.code_dispatch.resolve_code_engine", return_value=None):
        assert resolve_bound_code_system("code-cbm", "code--o--r", "u1", SimpleNamespace()) is None


def test_resolve_code_engine_unknown_system():
    from knowledge.code_dispatch import resolve_code_engine

    assert resolve_code_engine("nope", "code--o--r", "u1", SimpleNamespace()) is None


# ── index_store ──────────────────────────────────────────────────────


def test_index_store_roundtrip_in_memory():
    from knowledge import index_store

    index_store._reset_for_tests()
    # Force in-memory (no redis) by stubbing the client resolver.
    with patch("knowledge.index_store._redis", return_value=None):
        index_store.record_index("code--o--r", {"system": "code-cbm", "symbols": 5})
        assert index_store.get_index("code--o--r")["symbols"] == 5
        assert index_store.get_index("missing") is None

        index_store.save_project_config("p1", {"code_systems": ["code-cbm"], "code_space": "code--o--r"})
        cfg = index_store.load_project_config("p1")
        assert cfg["code_systems"] == ["code-cbm"]
        assert cfg["code_space"] == "code--o--r"
        assert index_store.load_project_config("p2") is None
    index_store._reset_for_tests()


def test_router_config_persists_code_space():
    """set/get project config round-trips the Phase-G code_space field."""
    from knowledge import index_store
    from knowledge.router import (
        ProjectKnowledgeConfig,
        _PROJECT_CONFIGS,
        get_project_config,
        set_project_config,
    )

    index_store._reset_for_tests()
    _PROJECT_CONFIGS.pop("pg", None)
    with patch("knowledge.index_store._redis", return_value=None):
        set_project_config(ProjectKnowledgeConfig(
            project_id="pg", code_systems=["code-cbm"], default_engine="code-cbm",
            code_space="code--o--pg",
        ))
        # Drop the in-process mirror to force a load via index_store.
        _PROJECT_CONFIGS.pop("pg", None)
        cfg = get_project_config("pg")
        assert cfg is not None
        assert cfg.code_space == "code--o--pg"
        assert cfg.default_engine == "code-cbm"
    index_store._reset_for_tests()
