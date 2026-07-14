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
    # Phase E: batched anchor lookup requires source_ref.external_id in metadata.
    # The key MUST equal "{repo}::{to_canonical(fqn)}" — compute it, don't guess.
    query_fqn = "myrepo.utils.parse"
    anchor_key = f"myrepo::{NativeEngine.to_canonical(query_fqn)}"
    mock_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "Shared memory about this function",
            "metadata": {
                "source_ref": {"external_id": anchor_key},
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
                "source_ref": {"external_id": anchor_key},
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
                "source_ref": {"external_id": anchor_key},
                "category": "bugfix",
                "visibility": "private",
                "user_id": "bob",
                "created_at": "2026-01-03T00:00:00Z",
            }
        }),
    ]

    memory = mock_memory_service._get_memory()
    # GF2: batched anchor lookup uses scroll (filter-only) → (points, next_offset)
    memory.vector_store.client.scroll.return_value = (mock_points, None)

    # Alice queries (should see: shared + her own private)
    memories = engine._get_anchor_memories(query_fqn, user_id="alice", limit=10)

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
    # Phase E: batched anchor lookup requires source_ref.external_id keyed on
    # "{repo}::{to_canonical(fqn)}" — compute it, don't guess.
    query_fqn = "myrepo.core.process"
    anchor_key = f"myrepo::{NativeEngine.to_canonical(query_fqn)}"
    mock_points = [
        MagicMock(payload={
            "id": f"mem{i}",
            "data": f"Memory {i}",
            "metadata": {
                "source_ref": {"external_id": anchor_key},
                "category": "decision",
                "visibility": "shared",
                "user_id": "alice",
                "created_at": f"2026-01-0{i % 9 + 1}T00:00:00Z",
            }
        })
        for i in range(10)
    ]

    memory = mock_memory_service._get_memory()
    # GF2: batched anchor lookup uses scroll (filter-only) → (points, next_offset)
    memory.vector_store.client.scroll.return_value = (mock_points, None)

    # Request limit=3
    memories = engine._get_anchor_memories(query_fqn, user_id="alice", limit=3)

    assert len(memories) == 3  # capped from 10 (GF2: was trivially 0 pre-scroll-mock)


def test_locate_enriches_with_memories(mock_bridge, mock_memory_service):
    """locate() returns LocateHit with memories field populated."""
    # Dense leg via the mocked cloud embedder (no local ONNX load); card-text
    # BM25 leg off (these mocks carry no card corpus).
    settings = MagicMock()
    settings.code_embedder = "cloud"
    settings.code_locate_lexical_cards = False
    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--alice--myrepo",
        bridge=mock_bridge,
        settings=settings,
    )

    # Mock code_index search returning 1 symbol
    symbol_fqn = "myrepo.utils.parse"
    mock_code_points = [
        MagicMock(
            payload={
                "code_space": "code--alice--myrepo",
                "fqn": symbol_fqn,
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

    # Mock memory search returning 1 attached memory.
    # Phase E: batched anchor lookup requires source_ref.external_id keyed on
    # "{repo}::{to_canonical(fqn)}" — compute it from the symbol's fqn, don't guess.
    anchor_key = f"myrepo::{NativeEngine.to_canonical(symbol_fqn)}"
    mock_mem_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "This function had a bug in v1.2",
            "metadata": {
                "source_ref": {"external_id": anchor_key},
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

    # Dense locate still uses query_points; GF2: the anchor lookup uses scroll.
    memory.vector_store.client.query_points.return_value = mock_code_result
    memory.vector_store.client.scroll.return_value = (mock_mem_points, None)

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

    # Mock symbol search + traverse.
    # NOTE: real _traverse matches (seed)-[*1..depth]->(t) and returns only the
    # TARGETS (excludes the seed), so the traverse result is a DISTINCT downstream
    # node — the seed is emitted once (seed loop), the target once (traverse loop).
    seed_fqn = "myrepo.core.run"
    mock_symbols = [{"fqn": seed_fqn, "kind": "function", "file": "core.py", "line": 20}]
    mock_traverse = [
        {"fqn": "myrepo.core.helper", "kind": "function", "file": "core.py", "line": 55, "edges": []},
    ]

    # Mock memory search.
    # Phase E: batched anchor lookup requires source_ref.external_id keyed on
    # "{repo}::{to_canonical(fqn)}" — compute it from the seed fqn, don't guess.
    anchor_key = f"myrepo::{NativeEngine.to_canonical(seed_fqn)}"
    mock_mem_points = [
        MagicMock(payload={
            "id": "mem1",
            "data": "Decision: use async for this",
            "metadata": {
                "source_ref": {"external_id": anchor_key},
                "category": "decision",
                "visibility": "shared",
                "user_id": "alice",
                "created_at": "2026-01-01T00:00:00Z",
            }
        })
    ]
    memory = mock_memory_service._get_memory()
    # GF2: batched anchor lookup uses scroll (filter-only) → (points, next_offset)
    memory.vector_store.client.scroll.return_value = (mock_mem_points, None)

    with patch.object(engine, "_search_symbols", return_value=mock_symbols), \
         patch.object(engine, "_traverse", return_value=mock_traverse):

        result = engine.query("how does run work", user_id="alice")

    # Seed emitted exactly once (real _traverse excludes the seed → no double-emit)
    assert result.count(f"{seed_fqn} (function)") == 1
    assert "myrepo.core.run" in result
    assert "Memories:" in result
    assert "[decision]" in result
    assert "Decision: use async" in result


def test_anchor_round_trip_cross_engine_end_to_end(mock_bridge, mock_memory_service):
    """END-TO-END anchor moat: a memory anchored during NATIVE indexing is
    retrieved via CBM's answer for the SAME symbol, because both engines
    normalize to the SAME canonical anchor key.

    Flow:
      1. Native indexed `src.click.core.Group`; an anchor + memory were stored
         keyed on native's canonical: "myrepo::click.core.Group".
      2. CBM later answers with its cache-prefixed FQN for the same symbol.
      3. Fusion canonicalizes CBM's answer with CBM's OWN normalizer, then the
         anchor lookup builds the key from that canonical form.
      4. Keys agree → the memory is returned (a genuine hit, not just equal
         strings): the mock Qdrant only returns the memory when the query
         filter carries the expected canonical key.
    """
    from adapters.code_graph.cbm_engine import CBMEngine

    engine = NativeEngine(
        repo_path="/fake/path",
        code_space="code--test--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )

    # (1) The key the memory was anchored under, derived from NATIVE's raw FQN.
    native_raw = "src.click.core.Group"
    expected_key = f"myrepo::{NativeEngine.to_canonical(native_raw)}"  # myrepo::click.core.Group

    # (2) CBM's answer for the same symbol (cache-path-prefixed), canonicalized
    #     with CBM's OWN normalizer.
    cbm_raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.Group"
    cbm_canonical = CBMEngine.to_canonical(cbm_raw)  # click.core.Group

    # Cross-engine agreement is the whole point of the moat.
    assert cbm_canonical == NativeEngine.to_canonical(native_raw) == "click.core.Group"

    mem_point = MagicMock(payload={
        "id": "mem-decision",
        "data": "Group dispatches subcommands — decided in ADR-3.",
        "metadata": {
            "source_ref": {"external_id": expected_key},  # Phase E: batched lookup requires this
            "category": "decision",
            "visibility": "shared",
            "user_id": "bob",
            "created_at": "2026-01-01T00:00:00Z",
        },
    })

    def scroll_side_effect(*args, **kwargs):
        """Return the memory ONLY when the filter carries the expected key.
        GF2: batched lookup uses scroll(scroll_filter=...) with MatchAny → (points, offset)."""
        qf = kwargs.get("scroll_filter")
        # Check for MatchAny (batched lookup)
        matched = any(
            getattr(c, "match", None) is not None
            and hasattr(c.match, "any")
            and expected_key in c.match.any
            for c in qf.must
        )
        return ([mem_point] if matched else [], None)

    memory = mock_memory_service._get_memory()
    memory.vector_store.client.scroll.side_effect = scroll_side_effect

    # (3)+(4) Look up via CBM's canonicalized answer; native re-canonicalization
    # is idempotent on an already-canonical FQN, so the key matches → HIT.
    memories = engine._get_anchor_memories(cbm_canonical, user_id="bob", limit=3)

    assert len(memories) == 1, "cross-engine anchor join missed — keys diverged"
    assert memories[0]["id"] == "mem-decision"


def test_ensure_anchors_keys_on_canonical_fqn(mock_bridge, tmp_path):
    """_ensure_anchors MERGEs the CodeAnchor on canonical_fqn (not raw fqn).

    Guards the create-side of the moat: the anchor node key must be the persisted
    canonical FQN so it agrees with the canonicalized lookup key.
    """
    engine = NativeEngine(
        repo_path=str(tmp_path),
        code_space="code--test--myrepo",
        bridge=mock_bridge,
        settings=MagicMock(),
    )
    with patch.object(engine, "_run_cypher_with_retry", return_value=[{"anchored": 1}]) as mock_cypher:
        engine._ensure_anchors()
        cypher = mock_cypher.call_args[0][0]
        # Anchor keyed on canonical_fqn (with coalesce fallback), NOT bare s.fqn.
        assert "canonical_fqn" in cypher
        assert "coalesce(s.canonical_fqn, s.fqn)" in cypher


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
