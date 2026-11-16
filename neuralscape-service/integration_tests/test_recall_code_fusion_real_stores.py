"""Flag-on proof for code-graph fusion in recall (MCP ``recall_memories``).

Atlas node ``read.recall-code-fusion``. The feature is "on" for a project when
its knowledge config names a code system and a ``code_space``; without that the
recall output is byte-for-byte plain recall. This test turns it on the way
production does and drives the real MCP handler against real stores, with no
LLM or embedding spend on the code side:

1. ``CODE_NATIVE_ENABLED`` registers the native engine; the fixture repo is
   indexed by ``worker.process_code_index`` (the ingest worker's own function,
   deterministic, BM25 locate, ``code_embedder=off``). That function is also
   what writes the project's knowledge config (code system + ``code_space``),
   the same record ``set_project_config`` stores for the router.
2. Memories are written through ``MemoryService.store_raw`` with
   ``source_ref.external_id = <repo>::<canonical FQN>`` (the anchor key the
   join reads): one PRIVATE anchored memory by the owner, one SHARED anchored
   memory by the owner.
3. ``mcp_server.call_tool("recall_memories", ...)`` for a code question
   returns the fused envelope. The owner sees both anchored memories in
   ``sections.semantics``. A second user who is allowed to read the code space
   (an admin) also gets fused output, but only the shared memory: the private
   anchor stays private. A second user without access to the code space gets
   plain recall and sees neither anchored memory's text.
4. After the owner's private anchored memory is dream-tombstoned it is gone
   from the owner's ``semantics`` too, while the shared one stays.
5. A project with no knowledge config returns plain recall (a JSON list, no
   ``fused`` envelope) for the very same question.

Needs Neo4j, Qdrant, Redis and ``GOOGLE_API_KEY`` (embeddings for the memory
rows; the code leg itself is deterministic). On the gate
(``NS_INTEGRATION_REQUIRED=1``) missing infrastructure fails the test.
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


def _recall(user_id, project_id, query=QUESTION):
    import mcp_server

    out = asyncio.run(
        mcp_server.call_tool(
            "recall_memories", {"query": query, "user_id": user_id, "project_id": project_id, "limit": 10}
        )
    )
    return out[0].text


def _semantic_ids(text: str) -> set[str]:
    sections = json.loads(text)["sections"]
    return {m["id"] for mems in (sections.get("semantics") or {}).values() for m in mems}


def test_recall_fuses_code_structure_and_anchored_memories_with_the_project_config_on(
    flags, svc, stores, llm_key, unique_ns, native_on, code_repos, repo_dir
):
    from extensions.dreaming.consolidate import _tombstone
    from knowledge import code_dispatch, index_store
    from knowledge.router import _PROJECT_CONFIGS, get_project_config
    import worker

    owner = unique_ns.user_id
    teammate = f"{owner}-mate"  # may read the code space only because the test makes it an admin
    outsider = f"{owner}-out"  # neither owner nor admin
    project = unique_ns.project_id
    no_config_project = f"{project}-plain"
    code_space = f"code--{owner}--demo"
    tag = uuid.uuid4().hex[:6]

    def cleanup():
        torn = code_dispatch.teardown_code_space("code-native", code_space, owner, code_repos)
        _PROJECT_CONFIGS.pop(project, None)
        r = index_store._redis()
        if r is not None:
            r.delete(index_store._KEY_PREFIX + code_space, index_store._CFG_PREFIX + project)
        # The native engine deliberately preserves :CodeAnchor nodes (and index()
        # creates one per symbol), so remove everything carrying this code_space.
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(stores.neo4j_uri, auth=(stores.neo4j_user, stores.neo4j_password))
        try:
            with driver.session(database=stores.neo4j_database) as session:
                session.run("MATCH (n {code_space: $cs}) DETACH DELETE n", cs=code_space).consume()
                left = session.run("MATCH (n {code_space: $cs}) RETURN count(n) AS c", cs=code_space).single()["c"]
        finally:
            driver.close()
        assert torn.get("deleted") is True, f"teardown_code_space refused: {torn}"
        assert left == 0, f"{left} nodes with code_space {code_space} remain"

    unique_ns.on_cleanup(cleanup)

    # ── the knowledge config: written by the real indexing path ────────────
    result = asyncio.run(
        worker.process_code_index(
            {"service": svc},
            {
                "system": "code-native",
                "repo_source": str(repo_dir),
                "code_space": code_space,
                "project_id": project,
                "user_id": owner,
            },
        )
    )
    assert result["ok"] is True, result
    cfg = get_project_config(project)
    assert cfg.code_systems == ["code-native"] and cfg.code_space == code_space and cfg.fuse_code_into_recall
    assert get_project_config(no_config_project) is None

    # ── anchored memories through the real write path ──────────────────────
    private_text = f"Decision zq{tag}-private: parse_arguments must keep the caller's argument order."
    shared_text = f"Convention zq{tag}-shared: parse_arguments never mutates its input list."
    kept_private_text = f"Note zq{tag}-kept: parse_arguments is the only entry point that builds Argument objects."
    ids = {}
    for key, text, visibility in (
        ("private", private_text, "private"),
        ("shared", shared_text, "shared"),
        ("kept_private", kept_private_text, "private"),
    ):
        out = svc.store_raw(
            content=text, user_id=owner, category="decision", scope="project", project_id=project,
            visibility=visibility, source_ref={"external_id": ANCHOR_KEY},
            add_to_graph=False,
        )
        ids[key] = out[0].id
    assert len(set(ids.values())) == 3

    # ── the owner: fused envelope, structure + both anchored memories ──────
    text = _recall(owner, project)
    fused = json.loads(text)
    assert fused["fused"] is True, f"expected a fused envelope, got: {text[:300]}"
    structure = fused["sections"]["structure"]
    assert structure["system"] == "code-native"
    assert TARGET_FQN in {h["fqn"] for h in structure["hits"]}, structure["hits"]
    assert fused["routing"]["op"] == "locate" and fused["routing"]["served_by"] == "code-native"
    assert TARGET_FQN in fused["sections"]["semantics"], list(fused["sections"].get("semantics", {}))
    assert {ids["private"], ids["shared"], ids["kept_private"]} <= _semantic_ids(text), (
        "the owner must see their private anchored memories and the shared one in `semantics`"
    )

    # ── a second user who can read the code space: fused, but only shared ──
    flags(dictator_user_ids=teammate)  # is_admin_user(teammate) -> may bind the owner's code space
    mate_text = _recall(teammate, project)
    mate = json.loads(mate_text)
    assert mate["fused"] is True, "an admin reader still gets the fused envelope"
    mate_ids = _semantic_ids(mate_text)
    assert ids["shared"] in mate_ids, "the shared anchored memory is visible to teammates"
    assert not mate_ids & {ids["private"], ids["kept_private"]}, "a private anchored memory leaked to another user"
    assert f"zq{tag}-private" not in mate_text and f"zq{tag}-kept" not in mate_text

    # ── a user with no access to the code space: plain recall, nothing leaks ─
    out_text = _recall(outsider, project)
    assert isinstance(json.loads(out_text), list), "no code-space access -> plain recall"
    assert f"zq{tag}-private" not in out_text and f"zq{tag}-kept" not in out_text

    # ── a dream-tombstoned anchored memory drops out of the join ───────────
    _tombstone(svc, ids["private"], superseded_by=ids["kept_private"])
    after = _recall(owner, project)
    assert json.loads(after)["fused"] is True
    after_ids = _semantic_ids(after)
    assert ids["private"] not in after_ids, "a tombstoned anchored memory must not be served"
    assert {ids["shared"], ids["kept_private"]} <= after_ids
    assert f"zq{tag}-private" not in after

    # ── no knowledge config: plain recall for the same question ────────────
    plain = json.loads(_recall(owner, no_config_project))
    assert isinstance(plain, list), "without a project knowledge config recall must stay a plain list"
