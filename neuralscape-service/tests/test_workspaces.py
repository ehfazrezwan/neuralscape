"""Test workspace partition (WT6 v1 MVP).

Validates:
- Absent/None/"memory" workspaces are byte-identical to legacy (backward compat)
- Reference workspaces excluded from default search
- Reference workspaces excluded from identity card
- Reference workspaces run restricted dreaming actions (MERGE + PRUNE only)
- Workspace affects dedup key (same content in memory vs reference = distinct)
- retag_memories set_workspace operation
"""

import pytest
from unittest.mock import MagicMock
from memory_service import MemoryService
from memory.groups import _build_group_id
from extensions.dreaming.consolidate import pool_key, PoolBatch, decide
from extensions.dreaming.card import card_target
from pathlib import Path


def test_workspace_backward_compat_group_id():
    """Absent/None/"memory" workspace produces identical group_id to legacy."""
    # All three should be byte-identical to the no-workspace call
    legacy = _build_group_id("private", "user123", None)
    assert legacy == "user--user123"

    # Absent workspace
    assert _build_group_id("private", "user123", None, None) == legacy
    # "memory" workspace
    assert _build_group_id("private", "user123", None, "memory") == legacy

    # Reference workspace appends suffix
    ref = _build_group_id("private", "user123", None, "ref-trading")
    assert ref == "user--user123--ws--ref-trading"
    assert ref != legacy


def test_workspace_pool_key():
    """pool_key includes workspace suffix for reference workspaces."""
    # Memory workspace (absent or explicit "memory")
    assert pool_key(visibility="private", owner_user_id="u1", project_id=None) == "user--u1"
    assert pool_key(visibility="private", owner_user_id="u1", project_id=None, workspace=None) == "user--u1"
    assert pool_key(visibility="private", owner_user_id="u1", project_id=None, workspace="memory") == "user--u1"

    # Reference workspace
    ref_key = pool_key(visibility="private", owner_user_id="u1", project_id=None, workspace="ref-book")
    assert ref_key == "user--u1--ws--ref-book"


def test_workspace_dedup_isolation(service):
    """Memory vs reference workspace target different dedup filters, so the same
    content can't collapse across workspaces; two memory-type lookups build the
    same filter, so same-workspace content still dedups."""

    def _dedup_must(workspace):
        client = MagicMock()
        client.scroll.return_value = ([], None)
        service._memory.vector_store.client = client
        service._find_by_content_hash(
            user_id="user123",
            content_hash="samehash",
            scope="global",
            workspace=workspace,
        )
        return client.scroll.call_args[1]["scroll_filter"].must

    mem_must = _dedup_must(None)
    ref_must = _dedup_must("ref-manual")
    mem_must_again = _dedup_must(None)

    # Memory and reference resolve to different row sets — no cross-workspace
    # dedup collapse (a user note and a book sentence stay distinct).
    assert mem_must != ref_must
    # Two memory-type lookups are identical — same-workspace content dedups.
    assert mem_must == mem_must_again


def test_workspace_card_exclusion():
    """Reference workspaces are excluded from card_target."""
    vault = Path("/tmp/vault")

    # Memory pool qualifies for card
    mem_batch = PoolBatch(
        pool="user--u1",
        group_id="user--u1",
        visibility="private",
        owner_user_id="u1",
        project_id=None,
        workspace=None,  # memory type
    )
    qualifies, _ = card_target(vault, mem_batch, "u1")
    assert qualifies is True

    # Reference workspace does NOT qualify
    ref_batch = PoolBatch(
        pool="user--u1--ws--ref-book",
        group_id="user--u1--ws--ref-book",
        visibility="private",
        owner_user_id="u1",
        project_id=None,
        workspace="ref-book",
    )
    qualifies, path = card_target(vault, ref_batch, "u1")
    assert qualifies is False
    assert path is None


@pytest.mark.asyncio
async def test_workspace_dreaming_action_filter():
    """Reference workspaces only allow MERGE and PRUNE actions."""
    # Mock LLM call that returns all action types
    async def mock_llm(prompt):
        return '''{"actions": [
            {"type": "merge", "memory_ids": ["m1", "m2"], "survivor_id": "m1", "content": "merged", "confidence": 0.9},
            {"type": "prune", "memory_ids": ["m3"], "confidence": 0.9},
            {"type": "invalidate", "memory_ids": ["m4"], "confidence": 0.9},
            {"type": "temporal_reframe", "memory_ids": ["m5"], "content": "reframed", "confidence": 0.9},
            {"type": "rewrite", "memory_ids": ["m6"], "content": "rewritten", "confidence": 0.9}
        ]}'''

    # Memory pool: all actions allowed
    mem_batch = PoolBatch(
        pool="user--u1",
        group_id="user--u1",
        visibility="private",
        owner_user_id="u1",
        project_id=None,
        workspace=None,
        memories=[
            {"memory_id": f"m{i}", "content": f"mem{i}"} for i in range(1, 7)
        ],
    )
    mem_actions = await decide(mem_batch, mock_llm)
    # All 5 action types survive validation
    mem_types = {a["type"] for a in mem_actions}
    assert "merge" in mem_types
    assert "prune" in mem_types
    assert "invalidate" in mem_types
    assert "temporal_reframe" in mem_types
    assert "rewrite" in mem_types

    # Reference workspace pool: only MERGE and PRUNE
    ref_batch = PoolBatch(
        pool="user--u1--ws--ref-book",
        group_id="user--u1--ws--ref-book",
        visibility="private",
        owner_user_id="u1",
        project_id=None,
        workspace="ref-book",
        memories=[
            {"memory_id": f"m{i}", "content": f"mem{i}"} for i in range(1, 7)
        ],
    )
    ref_actions = await decide(ref_batch, mock_llm)
    # Only MERGE and PRUNE survive
    ref_types = {a["type"] for a in ref_actions}
    assert ref_types == {"merge", "prune"}
    assert "invalidate" not in ref_types
    assert "temporal_reframe" not in ref_types
    assert "rewrite" not in ref_types


def test_workspace_search_default_excludes_reference(service):
    """Default search (workspaces=None → ["memory"]) fences reference content out
    at the pool post-filter; an explicit workspace list opens the door to it."""
    # Isolate the vector-pool post-filter: stub the graph + lexical legs.
    service._search_graph_for_visibility = MagicMock(return_value=[])
    service._lexical_pool_hits = MagicMock(return_value=[])

    mem_hit = _qhit(
        "m1", "User likes Python",
        {"workspace": None, "category": "preference", "scope": "global"},
    )
    ref_hit = _qhit(
        "r1", "Reference book chapter on Python",
        {"workspace": "ref-book", "category": "domain_knowledge", "scope": "global"},
    )
    service._memory.vector_store.client.query_points.return_value = _qresult([mem_hit, ref_hit])

    # Default: memory only.
    memories = [r.memory for r in service.search(query="Python", user_id="user123", workspaces=None)]
    assert "User likes Python" in memories
    assert "Reference book chapter on Python" not in memories

    # Explicit reference workspace: reference only.
    service._memory.vector_store.client.query_points.return_value = _qresult([mem_hit, ref_hit])
    ref_memories = [
        r.memory
        for r in service.search(query="Python", user_id="user123", workspaces=["ref-book"])
    ]
    assert "Reference book chapter on Python" in ref_memories
    assert "User likes Python" not in ref_memories


def test_workspace_retag_operation(service):
    """retag_memories set_workspace moves a matched row into a reference
    workspace via a payload update carrying metadata.workspace."""
    pt = MagicMock()
    pt.id = "mid-1"
    pt.payload = {
        "user_id": "user123",
        "data": "Test fact",
        "metadata": {
            "tags": ["test"],
            "category": "preference",  # global scope — no project_id needed
            "visibility": "private",
            "owner_user_id": "user123",
        },
    }
    client = service._memory.vector_store.client
    client.scroll.return_value = ([pt], None)  # offset None → single sweep

    result = service.retag_memories(
        caller_user_id="user123",  # owner → edit permitted
        filters={"tags_contains": ["test"]},
        ops={"set_workspace": "ref-migrated"},
        dry_run=False,
    )
    assert result["matched"] == 1
    assert result["updated"] == 1

    # The row was rewritten with the new workspace stamp.
    service._memory.vector_store.update.assert_called_once()
    _, kwargs = service._memory.vector_store.update.call_args
    assert kwargs["payload"]["metadata"]["workspace"] == "ref-migrated"


def _qhit(mid, data, metadata, score=0.9):
    """Build a mock Qdrant point (query_points hit)."""
    h = MagicMock()
    h.id = mid
    h.score = score
    h.payload = {"data": data, "metadata": metadata}
    return h


def _qresult(hits):
    """Wrap hits in a mock query_points result (.points)."""
    r = MagicMock()
    r.points = hits
    return r


@pytest.fixture
def service():
    """MemoryService with mocked internals — no live embedding/DB (unit test)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    return svc
