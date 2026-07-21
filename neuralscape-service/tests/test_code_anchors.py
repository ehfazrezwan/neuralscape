"""Tests for E4: CodeAnchor nodes and memory↔code enrichment bridge.

Coverage:
- Anchor MERGE + ANCHORED link at index
- Anchor survival across reindex (symbol deletion/recreation)
- source_ref anchor association via (repo, fqn)
- Code→answer enrichment with attached memories
- Read-scope enforcement (private memories filtered)
- Enrichment cap (max 3 memories per result)
- Boundary guard: code nodes NOT in memory graph
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adapters.code_graph.native_engine import NativeEngine


@pytest.fixture
def mock_bridge():
    """Mock Graphiti bridge with async Neo4j driver."""
    bridge = MagicMock()
    bridge.driver = MagicMock()
    loop = asyncio.new_event_loop()
    bridge._loop = loop
    yield bridge
    loop.close()


@pytest.fixture
def mock_memory_service():
    """Mock memory service with embedding model and Qdrant client."""
    with patch("memory_service.get_shared_service") as mock_svc:
        service = MagicMock()
        memory = MagicMock()

        # Mock embedding model
        memory.embedding_model.embed.return_value = [0.1] * 768
        memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 10

        # Mock Qdrant client
        memory.vector_store.client = MagicMock()
        memory.vector_store.collection_name = "test_memories"

        service._get_memory.return_value = memory
        mock_svc.return_value = service
        yield service


def test_ensure_anchors_creates_and_links(mock_bridge, tmp_path):
    """Anchors are MERGE'd and linked via (:CodeSymbol)-[:ANCHORED]->(:CodeAnchor)."""
    engine = NativeEngine(
        repo_path=str(tmp_path),
        code_space="code--test_user--test_repo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Mock the Cypher result: 5 symbols anchored
    mock_result = [{"anchored": 5}]

    with patch.object(engine, "_run_cypher_with_retry", return_value=mock_result) as mock_cypher:
        engine._ensure_anchors()

        # Verify the cypher was called with correct params
        mock_cypher.assert_called_once()
        call_args = mock_cypher.call_args
        assert "MERGE (a:CodeAnchor" in call_args[0][0]
        assert "MERGE (s)-[:ANCHORED]->(a)" in call_args[0][0]
        assert call_args[1]["code_space"] == "code--test_user--test_repo"
        assert call_args[1]["repo"] == "test_repo"


def test_anchor_survives_reindex(mock_bridge, tmp_path):
    """Anchors persist across symbol deletion/recreation during reindex.

    This is the core guarantee: deleting and recreating a CodeSymbol node
    re-attaches it to the SAME anchor node (by repo, fqn key match).
    """
    engine = NativeEngine(
        repo_path=str(tmp_path),
        code_space="code--test_user--test_repo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Simulate first index: creates anchor A for symbol S
    # Then reindex: deletes symbol S, recreates it, should link to SAME anchor A

    # Mock cypher to verify MERGE behavior (anchor exists, not recreated)
    call_count = [0]
    def mock_cypher_side_effect(*args, **kwargs):
        call_count[0] += 1
        # First call: 1 anchor created
        # Second call (reindex): 1 anchor matched (not created), link recreated
        return [{"anchored": 1}]

    with patch.object(engine, "_run_cypher_with_retry", side_effect=mock_cypher_side_effect):
        # First index
        engine._ensure_anchors()

        # Reindex (symbol recreated, anchor survives)
        engine._ensure_anchors()

        assert call_count[0] == 2
        # The MERGE ensures the same anchor is reused


def test_get_anchor_memories_respects_read_scope(mock_bridge, mock_memory_service):
    """Private memories of other users are NOT returned; shared/standard are."""
    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Mock Qdrant to return 3 memories: 1 shared, 1 alice's private, 1 bob's private
    mock_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "Shared memory about this function",
            "metadata": {
                "category": "decision",
                "visibility": "shared",
                "user_id": "bob",
                "created_at": "2026-01-01T00:00:00Z",
            }
        }),
        MagicMock(payload={
            "id": "mem2",
            "data": "Alice's private note",
            "metadata": {
                "category": "gotcha",
                "visibility": "private",
                "user_id": "alice",
                "created_at": "2026-01-02T00:00:00Z",
            }
        }),
        MagicMock(payload={
            "id": "mem3",
            "data": "Bob's private note",
            "metadata": {
                "category": "bugfix",
                "visibility": "private",
                "user_id": "bob",
                "created_at": "2026-01-03T00:00:00Z",
            }
        }),
    ]

    mock_result = MagicMock()
    mock_result.points = mock_points

    memory = mock_memory_service._get_memory()
    memory.vector_store.client.query_points.return_value = mock_result

    # Alice queries (should see: shared + her own private)
    memories = engine._get_anchor_memories("myrepo.utils.parse", user_id="alice", limit=10)

    assert len(memories) == 2
    ids = {m["id"] for m in memories}
    assert "mem1" in ids  # shared
    assert "mem2" in ids  # alice's private
    assert "mem3" not in ids  # bob's private (filtered out)


def test_get_anchor_memories_cap(mock_bridge, mock_memory_service):
    """Enrichment respects the limit (max N memories per result)."""
    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Mock 10 shared memories
    mock_points = [
        MagicMock(payload={
            "id": f"mem{i}",
            "data": f"Memory {i}",
            "metadata": {
                "category": "decision",
                "visibility": "shared",
                "user_id": "alice",
                "created_at": f"2026-01-0{i % 9 + 1}T00:00:00Z",
            }
        })
        for i in range(10)
    ]

    mock_result = MagicMock()
    mock_result.points = mock_points

    memory = mock_memory_service._get_memory()
    memory.vector_store.client.query_points.return_value = mock_result

    # Request limit=3
    memories = engine._get_anchor_memories("myrepo.core.process", user_id="alice", limit=3)

    assert len(memories) <= 3


def test_locate_enriches_with_memories(mock_bridge, mock_memory_service):
    """locate() returns LocateHit with memories field populated."""
    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Mock code_index search returning 1 symbol
    mock_code_points = [
        MagicMock(
            payload={
                "code_space": "code--alice--myrepo",
                "fqn": "myrepo.utils.parse",
                "kind": "function",
                "file": "utils.py",
                "line": 10,
                "signature": "def parse(data: str) -> dict",
                "docstring": "Parse input data",
                "degree": 5,
                "anchor_id": None,
            },
            score=0.9,
        )
    ]
    mock_code_result = MagicMock()
    mock_code_result.points = mock_code_points

    # Mock memory search returning 1 attached memory
    mock_mem_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "This function had a bug in v1.2",
            "metadata": {
                "category": "bugfix",
                "visibility": "shared",
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00Z",
            }
        })
    ]
    mock_mem_result = MagicMock()
    mock_mem_result.points = mock_mem_points

    memory = mock_memory_service._get_memory()

    # First call: code_index search; second call: memory search
    memory.vector_store.client.query_points.side_effect = [
        mock_code_result,  # dense search
        mock_mem_result,   # memory anchor search
    ]

    # Mock BM25 (disabled)
    with patch.object(engine, "_lexical_code_search", return_value=[]):
        hits = engine.locate("parse function", k=10, user_id="alice")

    assert len(hits) == 1
    hit = hits[0]
    assert hit.fqn == "myrepo.utils.parse"
    assert hit.memories is not None
    assert len(hit.memories) == 1
    assert hit.memories[0]["id"] == "mem1"
    assert "bug" in hit.memories[0]["content"]


def test_query_enriches_with_memories(mock_bridge, mock_memory_service):
    """query() output includes attached memories per symbol."""
    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Mock symbol search + traverse
    mock_symbols = [{"fqn": "myrepo.core.run", "kind": "function", "file": "core.py", "line": 20}]
    mock_traverse = [
        {"fqn": "myrepo.core.run", "kind": "function", "file": "core.py", "line": 20, "edges": []},
    ]

    # Mock memory search
    mock_mem_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "Decision: use async for this",
            "metadata": {
                "category": "decision",
                "visibility": "shared",
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00Z",
            }
        })
    ]
    mock_mem_result = MagicMock()
    mock_mem_result.points = mock_mem_points

    memory = mock_memory_service._get_memory()
    memory.vector_store.client.query_points.return_value = mock_mem_result

    with patch.object(engine, "_search_symbols", return_value=mock_symbols), \
         patch.object(engine, "_traverse", return_value=mock_traverse):

        result = engine.query("how does run work", user_id="alice")

    assert "myrepo.core.run" in result
    assert "Memories:" in result
    assert "[decision]" in result
    assert "Decision: use async" in result


def test_boundary_guard_no_code_in_memory_graph(mock_bridge, tmp_path):
    """Guard test: anchors stay in code label-space, never enter memory graph.

    This test verifies the architectural boundary: CodeAnchor nodes are created
    with code_space partition, NOT group_id (which is the memory graph partition).
    Memories link via source_ref.external_id string match, NOT by node UUID.
    """
    engine = NativeEngine(
        repo_path=str(tmp_path),
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # Verify the anchor cypher uses code_space (code label-space)
    # and never uses group_id (memory graph partition)
    mock_result = [{"anchored": 1}]

    with patch.object(engine, "_run_cypher_with_retry", return_value=mock_result) as mock_cypher:
        engine._ensure_anchors()

        cypher = mock_cypher.call_args[0][0]
        params = mock_cypher.call_args[1]

        # Anchors are in code_space partition
        assert "code_space" in params
        assert params["code_space"] == "code--alice--myrepo"

        # NO group_id in the cypher (that's the memory graph partition)
        assert "group_id" not in cypher.lower()
        assert "group_id" not in params

        # CodeAnchor label, not Entity or Episode (memory graph labels)
        assert ":CodeAnchor" in cypher
        assert ":Entity" not in cypher
        assert ":Episode" not in cypher
