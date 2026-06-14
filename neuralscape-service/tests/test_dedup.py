"""Tests for Qdrant memory deduplication."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_service import MemoryService


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


def _make_point(id, payload):
    """Create a mock Qdrant point with .id and .payload attributes."""
    pt = MagicMock()
    pt.id = id
    pt.payload = payload
    return pt


def _make_hit(id, score, payload):
    """Create a mock Qdrant search hit with .id, .score, .payload."""
    hit = MagicMock()
    hit.id = id
    hit.score = score
    hit.payload = payload
    return hit


@pytest.fixture
def service():
    """Create a MemoryService with mocked internals for dedup testing."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    # Default: embedding_model.embed returns a dummy vector
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    # Default: vector_store.search returns empty
    svc._memory.vector_store.search.return_value = []
    # Default: vector_store.delete succeeds
    svc._memory.vector_store.delete.return_value = None
    return svc


# ──────────────────────────────────────────────
# _scroll_all_user_memories
# ──────────────────────────────────────────────


class TestScrollAllUserMemories:
    def test_single_page(self, service):
        pt = _make_point("p1", {"user_id": "u1", "data": "hello"})
        service._memory.vector_store.client.scroll.return_value = ([pt], None)

        result = service._scroll_all_user_memories("u1")

        assert len(result) == 1
        assert result[0]["id"] == "p1"
        assert result[0]["payload"]["data"] == "hello"

    def test_paginates_correctly(self, service):
        pt1 = _make_point("p1", {"user_id": "u1", "data": "a"})
        pt2 = _make_point("p2", {"user_id": "u1", "data": "b"})
        service._memory.vector_store.client.scroll.side_effect = [
            ([pt1], "offset-2"),
            ([pt2], None),
        ]

        result = service._scroll_all_user_memories("u1", batch_size=1)

        assert len(result) == 2
        assert service._memory.vector_store.client.scroll.call_count == 2

    def test_empty_collection(self, service):
        service._memory.vector_store.client.scroll.return_value = ([], None)

        result = service._scroll_all_user_memories("u1")

        assert result == []


# ──────────────────────────────────────────────
# get_all_user_ids
# ──────────────────────────────────────────────


class TestGetAllUserIds:
    def test_collects_unique_ids(self, service):
        pt1 = _make_point("p1", {"user_id": "alice"})
        pt2 = _make_point("p2", {"user_id": "bob"})
        pt3 = _make_point("p3", {"user_id": "alice"})
        service._memory.vector_store.client.scroll.return_value = (
            [pt1, pt2, pt3],
            None,
        )

        result = service.get_all_user_ids()

        assert set(result) == {"alice", "bob"}

    def test_paginates(self, service):
        pt1 = _make_point("p1", {"user_id": "alice"})
        pt2 = _make_point("p2", {"user_id": "bob"})
        service._memory.vector_store.client.scroll.side_effect = [
            ([pt1], "next"),
            ([pt2], None),
        ]

        result = service.get_all_user_ids(batch_size=1)

        assert set(result) == {"alice", "bob"}
        assert service._memory.vector_store.client.scroll.call_count == 2

    def test_projects_only_user_id_field(self, service):
        """Scroll fetches ONLY the user_id field, not the whole memory payload —
        the point of the cleanup is to stop hauling full payloads just to
        collect distinct authors."""
        service._memory.vector_store.client.scroll.return_value = ([], None)

        service.get_all_user_ids()

        call_kwargs = service._memory.vector_store.client.scroll.call_args[1]
        assert call_kwargs["with_payload"] == ["user_id"]
        assert call_kwargs["with_vectors"] is False


# ──────────────────────────────────────────────
# dedup_memories — Exact phase
# ──────────────────────────────────────────────


class TestDedupExact:
    def test_removes_older_duplicate_keeps_newest(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-06-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 1
        # p1 is older, should be deleted
        service._memory.vector_store.delete.assert_called_once_with("p1")

    def test_handles_three_duplicates(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-03-01T00:00:00"}),
            _make_point("p3", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-06-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 2
        deleted_ids = [call.args[0] for call in service._memory.vector_store.delete.call_args_list]
        assert "p1" in deleted_ids
        assert "p2" in deleted_ids
        assert "p3" not in deleted_ids  # newest kept

    def test_no_duplicates_nothing_removed(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa", "data": "fact1", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "bbb", "data": "fact2", "created_at": "2025-01-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 0
        assert result["semantic_duplicates_removed"] == 0


# ──────────────────────────────────────────────
# dedup_memories — Semantic phase
# ──────────────────────────────────────────────


class TestDedupSemantic:
    def test_search_uses_top_k_not_limit(self, service):
        """Regression for the mem0 v2.0.2 kwarg rename — ``Qdrant.search()``
        accepts ``top_k`` not ``limit``. Calling with ``limit`` raised
        ``unexpected keyword argument`` for every dedup pass, silently
        breaking semantic dedup for all users."""
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa", "data": "X", "created_at": "2025-01-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)
        service._memory.vector_store.search.return_value = []

        service.dedup_memories("u1")

        # vector_store.search may be called zero or N times depending on
        # the user's pool; what matters is that EVERY call uses ``top_k``
        # not ``limit``.
        for call in service._memory.vector_store.search.call_args_list:
            kwargs = call.kwargs
            assert "limit" not in kwargs, (
                "vector_store.search must not pass `limit` — mem0 v2.0.2 "
                "renamed it to `top_k`. See PR #47 follow-up."
            )
            # If any results were requested, top_k must be set.
            if kwargs:
                assert "top_k" in kwargs or "vectors" in kwargs

    def test_deletes_older_above_threshold(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa", "data": "User prefers dark mode", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "bbb", "data": "The user likes dark mode", "created_at": "2025-06-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        # When searching for p1's embedding, return p2 as a high-score hit
        hit = _make_hit("p2", 0.97, {"user_id": "u1", "data": "The user likes dark mode", "created_at": "2025-06-01T00:00:00"})
        # For p2's search, return p1 (but p1 already deleted)
        service._memory.vector_store.search.side_effect = [
            [hit],
            [],  # p2 search returns nothing relevant
        ]

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        # p1 is older, should be deleted
        service._memory.vector_store.delete.assert_called_once_with("p1")

    def test_keeps_both_below_threshold(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa", "data": "Likes Python", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "bbb", "data": "Likes JavaScript", "created_at": "2025-06-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        hit = _make_hit("p2", 0.80, {"user_id": "u1", "data": "Likes JavaScript", "created_at": "2025-06-01T00:00:00"})
        service._memory.vector_store.search.side_effect = [
            [hit],
            [],
        ]

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        service._memory.vector_store.delete.assert_not_called()

    def test_skips_already_deleted_ids(self, service):
        """IDs deleted in exact phase should be skipped in semantic phase."""
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00"}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-06-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        result = service.dedup_memories("u1")

        # p1 deleted in exact phase, only p2 remains for semantic
        # Semantic search only called for p2
        assert result["exact_duplicates_removed"] == 1
        assert service._memory.embedding_model.embed.call_count == 1


# ──────────────────────────────────────────────
# Graph edge cleanup
# ──────────────────────────────────────────────


class TestDedupGraphCleanup:
    def test_expires_graph_edges_on_delete(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00", "metadata": {"scope": "global"}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-06-01T00:00:00", "metadata": {"scope": "global"}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        # Mock _expire_graph_edges_for_memory to track calls
        with patch.object(service, "_expire_graph_edges_for_memory") as mock_expire:
            service.dedup_memories("u1")
            mock_expire.assert_called_once()

    def test_graph_failure_does_not_block_dedup(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00", "metadata": {}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-06-01T00:00:00", "metadata": {}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        with patch.object(service, "_expire_graph_edges_for_memory", side_effect=Exception("Neo4j down")):
            result = service.dedup_memories("u1")

        # Should still succeed — graph failure is non-critical
        assert result["exact_duplicates_removed"] == 1


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestDedupEdgeCases:
    def test_empty_collection(self, service):
        service._memory.vector_store.client.scroll.return_value = ([], None)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 0
        assert result["semantic_duplicates_removed"] == 0
        assert result["total_checked"] == 0

    def test_single_memory(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact", "created_at": "2025-01-01T00:00:00"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        result = service.dedup_memories("u1")

        assert result["exact_duplicates_removed"] == 0
        assert result["total_checked"] == 1


# ──────────────────────────────────────────────
# dedup_all_memories cron wrapper
# ──────────────────────────────────────────────


class TestDedupAllMemories:
    @pytest.mark.asyncio
    async def test_processes_all_users(self):
        from worker import dedup_all_memories

        svc = MagicMock(spec=MemoryService)
        svc.get_all_user_ids.return_value = ["alice", "bob"]
        svc.dedup_memories.return_value = {
            "user_id": "alice",
            "exact_duplicates_removed": 1,
            "semantic_duplicates_removed": 0,
            "total_checked": 5,
        }

        ctx = {"service": svc}
        result = await dedup_all_memories(ctx)

        assert result["users_processed"] == 2
        assert svc.dedup_memories.call_count == 2

    @pytest.mark.asyncio
    async def test_continues_on_per_user_failure(self):
        from worker import dedup_all_memories

        svc = MagicMock(spec=MemoryService)
        svc.get_all_user_ids.return_value = ["alice", "bob"]
        svc.dedup_memories.side_effect = [
            Exception("Qdrant timeout"),
            {
                "user_id": "bob",
                "exact_duplicates_removed": 2,
                "semantic_duplicates_removed": 0,
                "total_checked": 10,
            },
        ]

        ctx = {"service": svc}
        result = await dedup_all_memories(ctx)

        assert result["users_processed"] == 2
        # Bob's results should be counted even though Alice failed
        assert result["total_exact_removed"] == 2
