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
from memory_service import get_shared_service
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
    """Same content in memory vs reference workspace = distinct memories."""
    # Store the same content in memory workspace (default)
    mem1 = service.store_raw(
        content="The user prefers dark mode.",
        user_id="user123",
        category="preference",
        workspace=None,  # memory type
    )
    # Store same content in reference workspace
    ref1 = service.store_raw(
        content="The user prefers dark mode.",
        user_id="user123",
        category="preference",
        workspace="ref-manual",
    )
    # Different IDs — dedup didn't collapse them
    assert mem1[0].id != ref1[0].id

    # Storing same content again in memory should dedup
    mem2 = service.store_raw(
        content="The user prefers dark mode.",
        user_id="user123",
        category="preference",
        workspace=None,
    )
    assert mem2[0].id == mem1[0].id  # dedup hit


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
    """Default search (workspaces=None) excludes reference content."""
    # Store in memory workspace
    service.store_raw(
        content="User likes Python",
        user_id="user123",
        category="preference",
        workspace=None,
    )
    # Store in reference workspace
    service.store_raw(
        content="Reference book chapter on Python",
        user_id="user123",
        category="domain_knowledge",
        workspace="ref-book",
    )

    # Default search (workspaces=None → ["memory"])
    results = service.search(
        query="Python",
        user_id="user123",
        workspaces=None,  # defaults to ["memory"]
    )
    # Only the memory-type result
    assert len(results) == 1
    assert "User likes Python" in results[0].memory

    # Explicit reference workspace search
    ref_results = service.search(
        query="Python",
        user_id="user123",
        workspaces=["ref-book"],
    )
    assert len(ref_results) == 1
    assert "Reference book" in ref_results[0].memory


def test_workspace_retag_operation(service):
    """retag_memories supports set_workspace operation."""
    # Store a memory
    mem = service.store_raw(
        content="Test fact",
        user_id="user123",
        category="domain_knowledge",
        tags=["test"],
    )
    mid = mem[0].id

    # Retag to move to reference workspace
    result = service.retag_memories(
        caller_user_id="user123",
        filters={"tags_contains": ["test"]},
        ops={"set_workspace": "ref-migrated"},
        dry_run=False,
    )
    assert result["matched"] >= 1
    assert result["updated"] >= 1

    # Verify the workspace changed (search in the new workspace)
    results = service.search(
        query="Test fact",
        user_id="user123",
        workspaces=["ref-migrated"],
    )
    assert any(r.id == mid for r in results)

    # Default search no longer finds it
    default_results = service.search(
        query="Test fact",
        user_id="user123",
        workspaces=None,  # defaults to ["memory"]
    )
    assert not any(r.id == mid for r in default_results)


@pytest.fixture
def service():
    """Shared MemoryService for tests (mocked dependencies)."""
    # This would use a real service in integration tests;
    # for unit tests, mock Qdrant/Neo4j/Redis.
    return get_shared_service()
