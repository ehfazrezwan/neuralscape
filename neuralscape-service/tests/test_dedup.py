"""Tests for Qdrant memory deduplication."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_service import (
    REINFORCEMENT_BOOST_K,
    MemoryService,
    _reinforcement_boost,
    _times_derived_from_metadata,
)
from schemas import MemoryResponse


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


# ──────────────────────────────────────────────
# Reinforcement-aware dedup (times_derived) — A2
# ──────────────────────────────────────────────


def _set_retrieve(service, payload_by_id: dict):
    """Make client.retrieve return real point mocks for known ids."""

    def _retrieve(collection_name=None, ids=None, **kwargs):
        out = []
        for mid in ids or []:
            if mid in payload_by_id:
                out.append(_make_point(mid, payload_by_id[mid]))
        return out

    service._memory.vector_store.client.retrieve.side_effect = _retrieve


class TestTimesDerivedHelpers:
    def test_defaults_to_one(self):
        assert _times_derived_from_metadata(None) == 1
        assert _times_derived_from_metadata({}) == 1
        assert _times_derived_from_metadata({"times_derived": None}) == 1
        assert _times_derived_from_metadata({"times_derived": "garbage"}) == 1
        assert _times_derived_from_metadata({"times_derived": 0}) == 1

    def test_reads_value_and_unwraps_double_metadata(self):
        assert _times_derived_from_metadata({"times_derived": 4}) == 4
        # mem0 double-wrap: {"metadata": {"metadata": {...}}}
        assert _times_derived_from_metadata({"metadata": {"times_derived": 3}}) == 3

    def test_boost_is_noop_for_unreinforced_and_none(self):
        assert _reinforcement_boost(None, {"times_derived": 9}) is None
        assert _reinforcement_boost(0.8, None) == 0.8
        assert _reinforcement_boost(0.8, {"times_derived": 1}) == 0.8

    def test_boost_formula_and_monotonicity(self):
        base = 0.8
        b3 = _reinforcement_boost(base, {"times_derived": 3})
        b10 = _reinforcement_boost(base, {"times_derived": 10})
        assert b3 == pytest.approx(base * (1 + REINFORCEMENT_BOOST_K * math.log1p(2)))
        assert base < b3 < b10  # monotonic in times_derived
        assert b10 < base * 1.2  # k=0.05 stays a small nudge


class TestExactDedupReinforcement:
    def test_survivor_absorbs_dropped_counters(self, service):
        """3 exact copies → survivor's times_derived becomes the group sum."""
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-01-01T00:00:00", "metadata": {}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-03-01T00:00:00", "metadata": {}}),
            _make_point("p3", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-06-01T00:00:00", "metadata": {"scope": "global"}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)
        _set_retrieve(service, {"p3": {"metadata": {"scope": "global"}}})

        result = service.dedup_memories("u1", semantic=False)

        assert result["exact_duplicates_removed"] == 2
        service._memory.vector_store.client.set_payload.assert_called_once()
        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["p3"]
        # survivor(1) + p1(1) + p2(1) = 3; nested-key merge (audit 27 #30)
        # touches ONLY the counter — existing metadata survives server-side
        assert kwargs["key"] == "metadata"
        assert kwargs["payload"] == {"times_derived": 3}

    def test_dropped_duplicate_counters_sum_not_count(self, service):
        """A dropped dup that already accumulated reinforcements transfers
        its full counter — sum semantics, same rule as the dreaming MERGE."""
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-01-01T00:00:00",
                               "metadata": {"times_derived": 4}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-06-01T00:00:00",
                               "metadata": {"times_derived": 2}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)
        _set_retrieve(service, {"p2": {"metadata": {"times_derived": 2}}})

        service.dedup_memories("u1", semantic=False)

        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["p2"]
        assert kwargs["key"] == "metadata"  # nested-key merge (audit 27 #30)
        assert kwargs["payload"] == {"times_derived": 6}  # 2 + 4

    def test_no_duplicates_no_bump(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa", "data": "f1",
                               "created_at": "2025-01-01T00:00:00", "metadata": {}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)

        service.dedup_memories("u1", semantic=False)

        service._memory.vector_store.client.set_payload.assert_not_called()

    def test_bump_failure_never_blocks_dedup(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-01-01T00:00:00", "metadata": {}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-06-01T00:00:00", "metadata": {}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)
        service._memory.vector_store.client.retrieve.side_effect = Exception("Qdrant down")

        result = service.dedup_memories("u1", semantic=False)

        assert result["exact_duplicates_removed"] == 1  # delete still happened


class TestSemanticDedupReinforcement:
    def test_newer_near_duplicate_absorbs_older_counter(self, service):
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "aaa",
                               "data": "User prefers dark mode",
                               "created_at": "2025-01-01T00:00:00",
                               "metadata": {"times_derived": 3}}),
            _make_point("p2", {"user_id": "u1", "hash": "bbb",
                               "data": "The user likes dark mode",
                               "created_at": "2025-06-01T00:00:00", "metadata": {}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (memories, None)
        hit = _make_hit("p2", 0.97, {"user_id": "u1", "data": "The user likes dark mode",
                                     "created_at": "2025-06-01T00:00:00", "metadata": {}})
        service._memory.vector_store.search.side_effect = [[hit], []]
        _set_retrieve(service, {"p2": {"metadata": {}}})

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["p2"]
        assert kwargs["key"] == "metadata"  # nested-key merge (audit 27 #30)
        assert kwargs["payload"] == {"times_derived": 4}  # 1 + 3


class TestWritePathReinforcement:
    def test_store_raw_dedup_hit_bumps_existing(self, service):
        existing = MemoryResponse(
            id="exist1", memory="User prefers dark mode", category="preference",
            scope="global", source="vector", created_at="2025-01-01T00:00:00",
        )
        _set_retrieve(service, {"exist1": {"metadata": {"category": "preference"}}})

        with patch.object(service, "_find_by_content_hash", return_value=existing):
            responses, created = service.store_raw(
                content="User prefers dark mode", user_id="u1",
                category="preference", return_created=True,
            )

        assert created is False
        assert responses[0].id == "exist1"
        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["exist1"]
        assert kwargs["key"] == "metadata"  # nested-key merge (audit 27 #30)
        assert kwargs["payload"] == {"times_derived": 2}
        # dedup hit must not insert a new row
        service._memory.vector_store.insert.assert_not_called()

    def test_store_raw_new_memory_does_not_bump(self, service):
        service._memory.vector_store.client.scroll.return_value = ([], None)
        service._memory.embedding_model.embed.return_value = [0.1] * 8

        service.store_raw(
            content="Brand new fact", user_id="u1",
            category="preference", add_to_graph=False,
        )

        service._memory.vector_store.client.set_payload.assert_not_called()
        service._memory.vector_store.insert.assert_called_once()


class TestSearchReinforcementBoost:
    def _hits(self):
        # A: higher raw similarity, never reinforced.
        # B: slightly lower similarity, reinforced 10×.
        hit_a = _make_hit("a", 0.80, {"data": "one-off fact", "created_at": "2025-01-01",
                                      "metadata": {"category": "preference"}})
        hit_b = _make_hit("b", 0.78, {"data": "reinforced fact", "created_at": "2025-01-01",
                                      "metadata": {"category": "preference",
                                                   "times_derived": 10}})
        return [hit_a, hit_b]

    def test_personal_pool_boost_reorders(self, service):
        result = MagicMock()
        result.points = self._hits()
        service._memory.vector_store.client.query_points.return_value = result

        out = service._search_personal_pool(
            m=service._memory, user_id="u1", query_embedding=[0.1] * 8,
            project_id=None, categories=None, scope=None, domain=None,
            observation_type=None, concepts=None, limit=10,
        )

        scores = {r.id: r.score for r in out}
        assert scores["a"] == 0.80  # absent field → raw score untouched
        expected_b = 0.78 * (1 + REINFORCEMENT_BOOST_K * math.log1p(9))
        assert scores["b"] == pytest.approx(expected_b)
        assert scores["b"] > scores["a"]  # reinforced row outranks the one-off

    def test_shared_pool_applies_identical_boost(self, service):
        result = MagicMock()
        result.points = self._hits()
        service._memory.vector_store.client.query_points.return_value = result

        out = service._search_shared_pool(
            m=service._memory, query="q", project_id=None, categories=None,
            scope=None, domain=None, observation_type=None, concepts=None,
            limit=10, query_embedding=[0.1] * 8,
        )

        scores = {r.id: r.score for r in out}
        assert scores["a"] == 0.80
        assert scores["b"] == pytest.approx(
            0.78 * (1 + REINFORCEMENT_BOOST_K * math.log1p(9))
        )
