"""Real-store proof that indexing merges into the project's knowledge config, and that the
caller's own private anchored memory reaches ``neighbors`` (and nobody else's does).

Real Redis holds the config; the fixture repo is indexed by ``worker.process_code_index``
(the ingest worker's own function, native engine, deterministic, no LLM spend on the code
side). Fields an admin set before the index (``fuse_code_into_recall=False``,
``op_preference``, a pinned ``default_engine``) must survive it, the new system is added,
and ``code_space`` is recorded only for the pinned engine. A private memory anchored to a
symbol is attached to the symbol's neighbors for its author and absent for another user.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import importlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "code_demo_repo"
TARGET_FQN = "demo.core.parse_arguments"
ANCHOR_KEY = f"demo::{TARGET_FQN}"  # <repo>::<canonical FQN>; repo is the last code_space segment
QUESTION = "where is parse_arguments defined, the function that parses raw strings into Argument objects"


@pytest.fixture
def repo_dir(tmp_path) -> Path:
    """A throwaway copy of the fixture repo (engines write caches next to the source)."""
    dest = tmp_path / "demo"
    shutil.copytree(FIXTURE_REPO, dest)
    return dest


@pytest.fixture
def code_repos(flags, monkeypatch, repo_dir):
    """CODE_REPOS -> the fixture repo copy; dense/LLM legs off (deterministic)."""
    import config

    monkeypatch.setattr(config.settings, "code_repos", {"demo": str(repo_dir)})
    monkeypatch.setenv("CODE_REPOS", json.dumps({"demo": str(repo_dir)}))
    flags(code_embedder="off", code_index_embeddings=False)
    from adapters.code_graph import query as cg_query

    yield config.settings
    for key in [k for k in cg_query._ctx_cache if "--demo" in str(k)]:
        cg_query._ctx_cache.pop(key, None)


@pytest.fixture
def registry_snapshot():
    """Restore the knowledge registry exactly as it was, even if the test fails."""
    from knowledge.registry import KNOWLEDGE_REGISTRY

    saved = dict(KNOWLEDGE_REGISTRY)
    try:
        yield saved
    finally:
        KNOWLEDGE_REGISTRY.clear()
        KNOWLEDGE_REGISTRY.update(saved)


@pytest.fixture
def native_on(monkeypatch, code_repos, registry_snapshot):
    """Re-run knowledge registration with CODE_NATIVE_ENABLED=true."""
    import knowledge
    from knowledge.registry import KNOWLEDGE_REGISTRY

    monkeypatch.setenv("CODE_NATIVE_ENABLED", "true")
    importlib.reload(knowledge)
    assert "code-native" in KNOWLEDGE_REGISTRY, "CODE_NATIVE_ENABLED=true did not register code-native"


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns, monkeypatch):
    """A MemoryService over a throwaway Qdrant collection, installed as the shared service.

    The MCP handler reads through ``mcp_server._service`` and the anchor join
    through ``memory_service.get_shared_service()``: both must see the same
    throwaway collection.
    """
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url, redis_url=stores.redis_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    import memory_service
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    monkeypatch.setattr(memory_service, "_shared_service", service)
    import mcp_server

    monkeypatch.setattr(mcp_server, "_service", service)
    return service




def _raw_config(project):
    from knowledge import index_store

    r = index_store._redis()
    assert r is not None, "real Redis expected"
    blob = r.get(index_store._CFG_PREFIX + project)
    return json.loads(blob) if blob else None


def test_index_keeps_admin_config_and_private_anchor_reaches_neighbors_for_its_author_only(
    flags, svc, stores, llm_key, unique_ns, native_on, code_repos, repo_dir
):
    from knowledge import code_dispatch, index_store
    from knowledge.base import RecallRequest
    from knowledge.router import ProjectKnowledgeConfig, _PROJECT_CONFIGS, set_project_config
    import worker

    author = unique_ns.user_id
    other = f"{author}-b"
    project = unique_ns.project_id
    code_space = f"code--{author}--demo"
    tag = uuid.uuid4().hex[:6]
    index_store._reset_for_tests()

    def cleanup():
        code_dispatch.teardown_code_space("code-native", code_space, author, code_repos)
        _PROJECT_CONFIGS.pop(project, None)
        r = index_store._redis()
        if r is not None:
            r.delete(index_store._KEY_PREFIX + code_space, index_store._CFG_PREFIX + project)
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(stores.neo4j_uri, auth=(stores.neo4j_user, stores.neo4j_password))
        try:
            with driver.session(database=stores.neo4j_database) as session:
                session.run("MATCH (n {code_space: $cs}) DETACH DELETE n", cs=code_space).consume()
        finally:
            driver.close()

    unique_ns.on_cleanup(cleanup)

    # An admin configured the project before indexing, through the real store.
    set_project_config(
        ProjectKnowledgeConfig(
            project_id=project, code_systems=["code-native"], fuse_code_into_recall=False,
            default_engine="code-native", op_preference={"locate": ["code-native"]},
        )
    )
    assert _raw_config(project)["fuse_code_into_recall"] is False

    result = asyncio.run(
        worker.process_code_index(
            {"service": svc},
            {"system": "code-native", "repo_source": str(repo_dir), "code_space": code_space,
             "project_id": project, "user_id": author},
        )
    )
    assert result["ok"] is True, result
    cfg = _raw_config(project)  # straight from Redis, not this process's cache
    assert cfg["fuse_code_into_recall"] is False, cfg
    assert cfg["op_preference"] == {"locate": ["code-native"]}, cfg
    assert cfg["default_engine"] == "code-native" and cfg["code_systems"] == ["code-native"], cfg
    assert cfg["code_space"] == code_space, cfg

    # A project pinned to another engine: native is added, the pin and its space are kept.
    pinned = f"{project}-pinned"
    unique_ns.on_cleanup(lambda: index_store._redis().delete(index_store._CFG_PREFIX + pinned))
    set_project_config(
        ProjectKnowledgeConfig(project_id=pinned, code_systems=["code-cbm"], default_engine="code-cbm",
                               code_space="code--someone--cbm-space")
    )
    asyncio.run(
        worker.process_code_index(
            {"service": svc},
            {"system": "code-native", "repo_source": str(repo_dir), "code_space": code_space,
             "project_id": pinned, "user_id": author},
        )
    )
    kept = _raw_config(pinned)
    assert kept["default_engine"] == "code-cbm" and kept["code_space"] == "code--someone--cbm-space", kept
    assert kept["code_systems"] == ["code-cbm", "code-native"], kept

    # A private memory anchored to a symbol, written by the author.
    marker = f"zq{tag}-anchor"
    out = svc.store_raw(
        content=f"Decision {marker}: parse_arguments must keep the caller's argument order.",
        user_id=author, category="decision", scope="project", project_id=project,
        visibility="private", source_ref={"external_id": ANCHOR_KEY}, add_to_graph=False,
    )
    assert out

    def neighbors(user_id):
        bound = code_dispatch.resolve_bound_code_system("code-native", code_space, user_id, code_repos)
        return None if bound is None else bound.recall(
            RecallRequest(query="parse_arguments", user_id=user_id, operation="neighbors", label="parse_arguments")
        ).content

    mine = neighbors(author)
    assert marker in mine, f"the author's own private anchor must be attached: {mine[:300]}"
    flags(dictator_user_ids=other)  # an admin reader may bind the author's space
    theirs = neighbors(other)
    assert theirs is not None and "parse_arguments" in theirs
    assert marker not in theirs, "another user must not see the author's private anchored memory"
    flags(dictator_user_ids="")
    assert neighbors(f"{author}-c") is None, "a non-owner, non-admin cannot bind the space at all"
