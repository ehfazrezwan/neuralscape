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


class _UpRedis:
    """A Redis that always answers and holds no foreign lock: dedup takes its pool locks
    on it (it does nothing destructive without Redis, so these tests need one)."""

    def __init__(self):
        self.keys: dict[str, str] = {}

    def ping(self):
        return True

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    def get(self, key):
        return self.keys.get(key)

    def delete(self, key):
        self.keys.pop(key, None)

    def pexpire(self, key, ms):
        return 1 if key in self.keys else 0


@pytest.fixture
def service(monkeypatch):
    """Create a MemoryService with mocked internals for dedup testing."""
    redis = _UpRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: redis)
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
    # Default: a re-read (taken once a pool lock is held) sees the rows the test's
    # scroll serves, unchanged. _FakeQdrant installs its own, richer retrieve.
    client = svc._memory.vector_store.client

    def retrieve_scrolled_rows(collection_name=None, ids=None, **kwargs):
        served = client.scroll.return_value
        points = served[0] if isinstance(served, tuple) else []
        return [p for p in points if p.id in set(ids or [])]

    client.retrieve.side_effect = retrieve_scrolled_rows
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
        assert result["semantic_mode"] == "delete"
        # p1 is older, should be deleted (the default; tombstoning is opt-in)
        service._memory.vector_store.delete.assert_called_once_with("p1")
        assert _tombstone_calls(service) == []

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

    client = service._memory.vector_store.client
    served = client.scroll.return_value
    scrolled = {p.id: p for p in (served[0] if isinstance(served, tuple) else [])}

    def _retrieve(collection_name=None, ids=None, **kwargs):
        out = []
        for mid in ids or []:
            if mid in payload_by_id:
                # the override on top of what the scroll served (a re-read of the same row)
                base = scrolled[mid].payload if mid in scrolled else {}
                out.append(_make_point(mid, {**base, **payload_by_id[mid]}))
            elif mid in scrolled:  # a re-read of a row the scroll served: unchanged
                out.append(scrolled[mid])
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
        # All three copies carry the same scope stamp, as real same-partition
        # copies do (phase 1 groups by (hash, partition); an unstamped row is
        # its own partition, like the write path's content-hash lookup).
        memories = [
            _make_point("p1", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-01-01T00:00:00", "metadata": {"scope": "global"}}),
            _make_point("p2", {"user_id": "u1", "hash": "abc", "data": "fact",
                               "created_at": "2025-03-01T00:00:00", "metadata": {"scope": "global"}}),
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
        # the group's re-read works; the counter write on the survivor fails
        service._memory.vector_store.client.set_payload.side_effect = Exception("Qdrant down")

        result = service.dedup_memories("u1", semantic=False)

        assert result["exact_duplicates_removed"] == 1  # delete still happened
        service._memory.vector_store.client.set_payload.assert_called_once()


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


# ──────────────────────────────────────────────
# Semantic phase partitioning + stored vectors (P0)
# ──────────────────────────────────────────────


def _text_vec(text: str) -> list[float]:
    """Deterministic 16-d vector per text: same text ⇒ identical vector,
    different text ⇒ near-orthogonal (signed bytes of an md5)."""
    import hashlib

    digest = hashlib.md5(text.encode()).digest()
    return [float(b) - 127.5 for b in digest]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _dig(payload: dict, dotted: str):
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class _FakeQdrant:
    """Stand-in for the fork's Qdrant wrapper + raw client over a tiny store.

    ``search`` honours the fork's dict filter (equality / list-membership on
    dotted payload paths) and ranks by real cosine over the STORED vectors,
    so a test exercises the same filter the production call sends. Points
    carry the unnamed dense vector the way a bm25-enabled collection
    returns it: ``{"": dense, "bm25": ...}``.
    """

    def __init__(self, service, rows: list[tuple[str, dict, list[float] | None]]):
        self.payloads = {pid: payload for pid, payload, _ in rows}
        self.vectors = {pid: vec for pid, _, vec in rows if vec is not None}
        self.search_calls: list[dict] = []
        vs = service._memory.vector_store
        self.scroll_calls = 0
        vs.client.scroll.side_effect = self.scroll
        vs.client.set_payload.side_effect = self.set_payload
        vs.client.retrieve.side_effect = self.retrieve
        vs.search.side_effect = self.search
        vs.search.return_value = None
        vs.delete.side_effect = self.delete
        service._memory.embedding_model.embed.side_effect = (
            lambda text, **kw: _text_vec(text)
        )

    def scroll(self, collection_name=None, scroll_filter=None, limit=100, offset=None, **kwargs):
        """Live view of the store, paged: ``offset`` is the next row index."""
        self.scroll_calls += 1
        start = int(offset or 0)
        items = list(self.payloads.items())
        page = items[start:start + limit]
        nxt = start + limit if start + limit < len(items) else None
        return [_make_point(pid, payload) for pid, payload in page], nxt

    def set_payload(self, collection_name=None, payload=None, points=None, key=None, **kwargs):
        """Nested-key merge, as Qdrant does it: only the patch keys change."""
        for pid in points or []:
            target = self.payloads[pid]
            for part in (key.split(".") if key else []):
                target = target.setdefault(part, {})
            target.update(payload)

    def retrieve(self, collection_name=None, ids=None, with_vectors=False, **kwargs):
        out = []
        for pid in ids or []:
            if pid not in self.payloads:
                continue
            pt = _make_point(pid, self.payloads[pid])
            pt.vector = (
                {"": self.vectors[pid], "bm25": MagicMock(name="sparse")}
                if with_vectors and pid in self.vectors
                else None
            )
            out.append(pt)
        return out

    def _matches(self, payload: dict, filters: dict) -> bool:
        for key, value in (filters or {}).items():
            assert not isinstance(value, dict), f"fake filter does not model {key}={value!r}"
            actual = _dig(payload, key)
            if isinstance(value, list):
                if actual not in value:
                    return False
            elif actual != value:
                return False
        return True

    def search(self, query=None, vectors=None, top_k=5, filters=None):
        self.search_calls.append({"query": query, "vectors": vectors, "filters": filters, "top_k": top_k})
        hits = []
        for pid, payload in self.payloads.items():
            if pid not in self.vectors or not self._matches(payload, filters):
                continue
            hits.append(_make_hit(pid, _cosine(vectors, self.vectors[pid]), payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete(self, vector_id):
        self.payloads.pop(vector_id, None)
        self.vectors.pop(vector_id, None)


def _row(pid, text, created_at, *, user="u1", **meta):
    payload = {
        "user_id": user,
        "hash": f"h-{pid}",  # distinct hashes: exercise phase 2, not phase 1
        "data": text,
        "created_at": created_at,
        "metadata": meta,
    }
    return (pid, payload, _text_vec(text))


def _deleted(service) -> list[str]:
    return [c.args[0] for c in service._memory.vector_store.delete.call_args_list]


def _tombstone_calls(service) -> list[dict]:
    """set_payload calls that tombstone a row (the reinforcement bump is not one)."""
    return [
        c.kwargs
        for c in service._memory.vector_store.client.set_payload.call_args_list
        if c.kwargs["payload"].get("dream_tombstoned")
    ]


def _tombstoned(service) -> list[str]:
    return [pt for call in _tombstone_calls(service) for pt in call["points"]]


class TestSemanticDedupPartitions:
    TEXT = "Deploys go through the staging gate before prod"

    def test_same_text_in_two_projects_both_kept(self, service):
        fake = _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00",
                 scope="project", project_id="alpha", visibility="private"),
            _row("p2", self.TEXT, "2025-06-01T00:00:00",
                 scope="project", project_id="beta", visibility="private"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []
        # The project partition is pushed down to Qdrant, not just post-filtered.
        assert {c["filters"].get("metadata.project_id") for c in fake.search_calls} == {
            "alpha", "beta",
        }

    def test_global_row_not_merged_into_project_row(self, service):
        """A no-project (global) memory and a project memory of the same text
        are distinct; the absent project_id can't be pushed down (no is-empty
        operator in the fork's filter syntax) so the hit-side check holds it."""
        _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("p2", self.TEXT, "2025-06-01T00:00:00",
                 scope="global", project_id="alpha", visibility="private"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []

    def test_private_and_shared_copies_both_kept(self, service):
        fake = _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("p2", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="shared"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []
        assert {c["filters"].get("metadata.visibility") for c in fake.search_calls} == {
            "private", "shared",
        }

    def test_different_workspaces_both_kept(self, service):
        """A reference-corpus passage and a memory with the same text."""
        _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("p2", self.TEXT, "2025-06-01T00:00:00",
                 scope="global", visibility="private", workspace="handbook"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []

    def test_missing_workspace_is_memory_workspace(self, service):
        """Absent workspace and explicit "memory" are the same partition."""
        _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("p2", self.TEXT, "2025-06-01T00:00:00",
                 scope="global", visibility="private", workspace="memory"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        assert _deleted(service) == ["p1"]

    def test_same_partition_older_deleted_newest_kept_counter_moves(self, service):
        """In-partition near-duplicates still collapse (keep newest, transfer
        times_derived); the same text in a sibling project is untouched."""
        _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00",
                 scope="project", project_id="alpha", visibility="private", times_derived=3),
            _row("p2", self.TEXT, "2025-06-01T00:00:00",
                 scope="project", project_id="alpha", visibility="private"),
            _row("p3", self.TEXT, "2025-03-01T00:00:00",
                 scope="project", project_id="beta", visibility="private"),
        ])

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        assert _deleted(service) == ["p1"]
        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["p2"]
        assert kwargs["key"] == "metadata"
        assert kwargs["payload"] == {"times_derived": 4}  # 1 + 3

    def test_uses_stored_vectors_not_embedder(self, service):
        """Phase 2 searches with each row's stored dense vector; re-embedding
        every memory text on every pass is the bug."""
        fake = _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("p2", "Unrelated: the cafe opens at nine", "2025-06-01T00:00:00",
                 scope="global", visibility="private"),
        ])

        service.dedup_memories("u1")

        service._memory.embedding_model.embed.assert_not_called()
        assert [c["vectors"] for c in fake.search_calls] == [
            fake.vectors["p1"], fake.vectors["p2"],
        ]

    def test_embedder_fallback_only_for_rows_without_vector(self, service):
        fake = _FakeQdrant(service, [
            _row("p1", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            ("p2", {"user_id": "u1", "hash": "h-p2", "data": "No stored vector",
                    "created_at": "2025-06-01T00:00:00",
                    "metadata": {"scope": "global", "visibility": "private"}}, None),
        ])

        service.dedup_memories("u1")

        service._memory.embedding_model.embed.assert_called_once_with(
            "No stored vector", memory_action="add"
        )
        assert len(fake.search_calls) == 2


class TestExactDedupPartitions:
    """Phase 1 keys on (hash, partition): the hash is an md5 of the text only,
    so the same text in another project or workspace shares the real hash."""

    TEXT = "Deploys go through the staging gate before prod"

    def _same_hash_row(self, pid, created_at, **meta):
        from memory.hashing import content_hash

        pid, payload, vec = _row(pid, self.TEXT, created_at, **meta)
        payload["hash"] = content_hash(self.TEXT)
        return (pid, payload, vec)

    @pytest.mark.parametrize("semantic", [False, True])
    def test_same_hash_across_projects_and_workspaces_all_kept(self, service, semantic):
        _FakeQdrant(service, [
            self._same_hash_row("p1", "2025-01-01T00:00:00",
                                scope="project", project_id="alpha", visibility="private"),
            self._same_hash_row("p2", "2025-03-01T00:00:00",
                                scope="project", project_id="beta", visibility="private"),
            self._same_hash_row("p3", "2025-06-01T00:00:00",
                                scope="project", project_id="alpha", visibility="private",
                                workspace="handbook"),
            self._same_hash_row("p4", "2025-07-01T00:00:00",
                                scope="global", visibility="private"),
        ])

        result = service.dedup_memories("u1", semantic=semantic)

        assert result["exact_duplicates_removed"] == 0
        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []
        service._memory.vector_store.client.set_payload.assert_not_called()

    @pytest.mark.parametrize("semantic", [False, True])
    def test_same_hash_same_partition_older_deleted_counter_moves(self, service, semantic):
        _FakeQdrant(service, [
            self._same_hash_row("p1", "2025-01-01T00:00:00",
                                scope="project", project_id="alpha", visibility="private",
                                times_derived=3),
            # absent workspace and "memory" are one partition
            self._same_hash_row("p2", "2025-06-01T00:00:00",
                                scope="project", project_id="alpha", visibility="private",
                                workspace="memory"),
            self._same_hash_row("p3", "2025-03-01T00:00:00",
                                scope="project", project_id="beta", visibility="private"),
        ])

        result = service.dedup_memories("u1", semantic=semantic)

        assert result["exact_duplicates_removed"] == 1
        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == ["p1"]
        kwargs = service._memory.vector_store.client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["p2"]
        assert kwargs["payload"] == {"times_derived": 4}  # 1 + 3


@pytest.fixture
def tombstone_on(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", True)


def _merged(service) -> set[str]:
    """Rows phase 2 retired, whichever way (hard delete or tombstone)."""
    return set(_deleted(service)) | set(_tombstoned(service))


class TestDreamingOnSemanticPhase:
    """Dreaming on (the flags-on stack) must not switch off the REVERSIBLE semantic phase.

    Acceptance ran ``dedup-tombstone`` on a stack that also has DREAMING_ENABLED, and the
    compile's dedup passed ``semantic=False`` there, so nothing was ever tombstoned.
    """

    TEXT = "The nightly report job runs at 02:00 UTC on weekdays"

    def _pair(self, service):
        return _FakeQdrant(service, [
            _row("old", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("new", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])

    def _compile(self, service, tmp_path):
        import asyncio
        from unittest.mock import AsyncMock, patch

        pytest.importorskip("fcntl")
        from extensions.conversation_compiler.compile import compile_date
        from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new=AsyncMock(return_value="summary"),
        ):
            return asyncio.run(
                compile_date("2026-11-16", service, ObsidianWriter(vault_path=tmp_path / "v"), user_id="u1")
            )

    def test_compile_dedup_tombstones_with_dreaming_on_when_the_flag_is_on(
        self, service, tmp_path, monkeypatch, tombstone_on
    ):
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        # dedup takes the pool's dream lock here: a stand-in for the Redis the stack runs
        from tests.test_dedup_dream_exclusion import _FakeRedis

        monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda r=_FakeRedis(): r)
        fake = self._pair(service)

        result = self._compile(service, tmp_path)

        assert result.dedup_triggered is True
        assert fake.payloads["old"]["metadata"]["dream_tombstoned"] is True
        assert "old" in fake.payloads and _deleted(service) == []

    def test_dreaming_on_without_the_flag_still_skips_the_lossy_phase(
        self, service, tmp_path, monkeypatch
    ):
        from config import settings
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", False)
        fake = self._pair(service)

        self._compile(service, tmp_path)

        assert _merged(service) == set()
        assert "dream_tombstoned" not in fake.payloads["old"]["metadata"]

    @pytest.mark.parametrize(
        "dreaming, flag, expected",
        [(False, False, True), (False, True, True), (True, False, False), (True, True, True)],
    )
    def test_cron_and_compile_share_one_rule(self, monkeypatch, dreaming, flag, expected):
        from config import settings
        from extensions.conversation_compiler.compile import _semantic_dedup_enabled
        from extensions.dreaming.config import dreaming_settings
        from memory.delete import semantic_dedup_phase_enabled

        monkeypatch.setattr(dreaming_settings, "enabled", dreaming)
        monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", flag)
        assert semantic_dedup_phase_enabled() is expected
        assert _semantic_dedup_enabled() is expected

    @pytest.mark.parametrize(
        "dreaming, flag, expected",
        [(False, False, True), (False, True, True), (True, False, False), (True, True, True)],
    )
    def test_dedup_cron_passes_the_shared_rule(self, monkeypatch, dreaming, flag, expected):
        """The cron's whole truth table: forcing ``semantic`` either way fails a cell."""
        import asyncio
        from unittest.mock import MagicMock

        import worker
        from config import settings
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", dreaming)
        monkeypatch.setattr(settings, "dedup_semantic_tombstone_enabled", flag)
        svc = MagicMock()
        svc.get_all_user_ids.return_value = ["u1"]
        svc.dedup_memories.return_value = {
            "exact_duplicates_removed": 0, "semantic_duplicates_removed": 0, "graph_jobs": [],
        }
        asyncio.run(worker.dedup_all_memories({"service": svc}))
        svc.dedup_memories.assert_called_once_with("u1", semantic=expected)


@pytest.mark.usefixtures("tombstone_on")
class TestSemanticDedupIsReversible:
    """With DEDUP_SEMANTIC_TOMBSTONE_ENABLED, phase 2 tombstones the older near-duplicate."""

    TEXT = "Deploys go through the staging gate before prod"

    def _pair(self, service, **extra):
        return _FakeQdrant(service, [
            _row("old", self.TEXT, "2025-01-01T00:00:00",
                 scope="global", visibility="private", times_derived=2, **extra),
            _row("new", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])

    def test_older_row_is_tombstoned_not_deleted(self, service):
        fake = self._pair(service)

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 1
        assert result["semantic_mode"] == "tombstone"
        assert _deleted(service) == []
        assert "old" in fake.payloads  # the row and its text are still on disk
        assert fake.payloads["old"]["data"] == self.TEXT
        meta = fake.payloads["old"]["metadata"]
        assert meta["dream_tombstoned"] is True
        assert meta["superseded_by"] == "new"
        assert meta["superseded_reason"] == "semantic_dedup"
        assert meta["superseded_at"]
        assert meta["scope"] == "global"  # nested-key merge kept the rest of the metadata
        assert "dream_tombstoned" not in fake.payloads["new"]["metadata"]
        # reinforcement moved to the survivor
        assert fake.payloads["new"]["metadata"]["times_derived"] == 3

    def test_tombstone_uses_the_atomic_nested_key_patch(self, service):
        self._pair(service)
        service.dedup_memories("u1")
        (call,) = _tombstone_calls(service)
        assert call["key"] == "metadata" and call["points"] == ["old"]

    def test_tombstone_produces_no_graph_jobs(self, service):
        self._pair(service)
        result = service.dedup_memories("u1")
        assert result["graph_jobs"] == [] and result["graph_jobs_count"] == 0

    def test_tombstoned_rows_are_not_candidates_on_the_next_run(self, service):
        fake = self._pair(service)
        service.dedup_memories("u1")
        service._memory.vector_store.client.set_payload.reset_mock(side_effect=False)
        fake.search_calls.clear()

        second = service.dedup_memories("u1")

        assert second["semantic_duplicates_removed"] == 0
        assert _tombstoned(service) == []  # not tombstoned twice
        assert [c["vectors"] for c in fake.search_calls] == [fake.vectors["new"]]

    def test_tombstoned_row_never_shadows_a_live_exact_duplicate(self, service):
        """A tombstoned row with the same hash must not delete the live one."""
        fake = _FakeQdrant(service, [
            ("live", {"user_id": "u1", "hash": "same", "data": "x", "created_at": "2025-01-01T00:00:00",
                      "metadata": {"scope": "global"}}, _text_vec("x")),
            ("dead", {"user_id": "u1", "hash": "same", "data": "x", "created_at": "2025-06-01T00:00:00",
                      "metadata": {"scope": "global", "dream_tombstoned": True}}, _text_vec("x")),
        ])
        result = service.dedup_memories("u1")
        assert result["exact_duplicates_removed"] == 0
        assert set(fake.payloads) == {"live", "dead"}

    def test_rederiving_the_fact_revives_the_tombstone(self, service):
        """Write-path dedup hit on the tombstoned row clears the flag (existing contract)."""
        fake = self._pair(service)
        service.dedup_memories("u1")
        assert fake.payloads["old"]["metadata"]["dream_tombstoned"] is True

        assert service._revive_if_tombstoned("old") is True

        assert fake.payloads["old"]["metadata"]["dream_tombstoned"] is False
        assert fake.payloads["old"]["metadata"]["dream_revived_at"]
        # a revived row is a normal candidate again
        service._memory.vector_store.client.set_payload.reset_mock(side_effect=False)
        again = service.dedup_memories("u1")
        assert again["semantic_duplicates_removed"] == 1

    def test_tombstone_failure_is_not_counted_and_never_deletes(self, service):
        fake = self._pair(service)
        service._memory.vector_store.client.set_payload.side_effect = RuntimeError("qdrant down")

        result = service.dedup_memories("u1")

        assert result["semantic_duplicates_removed"] == 0
        assert _deleted(service) == []
        assert "dream_tombstoned" not in fake.payloads["old"].get("metadata", {})


@pytest.mark.usefixtures("tombstone_on")
class TestSemanticDedupWrappedMetadata:
    """Rows whose metadata is double-wrapped (``metadata.metadata``)."""

    TEXT = "Deploys go through the staging gate before prod"

    def _wrapped_pair(self, service):
        def wrapped(pid, created, **inner):
            payload = {"user_id": "u1", "hash": f"h-{pid}", "data": self.TEXT, "created_at": created,
                       "metadata": {"metadata": {"scope": "global", "visibility": "private", **inner}}}
            return (pid, payload, _text_vec(self.TEXT))

        return _FakeQdrant(service, [wrapped("old", "2025-01-01T00:00:00"), wrapped("new", "2025-06-01T00:00:00")])

    def test_wrapped_older_row_is_patched_at_both_levels(self, service):
        fake = self._wrapped_pair(service)
        result = service.dedup_memories("u1")
        assert result["semantic_duplicates_removed"] == 1
        keys = sorted(c["key"] for c in _tombstone_calls(service))
        assert keys == ["metadata", "metadata.metadata"]
        outer = fake.payloads["old"]["metadata"]
        assert outer["dream_tombstoned"] is True  # what the recall filter reads
        assert outer["metadata"]["dream_tombstoned"] is True
        assert outer["metadata"]["superseded_by"] == "new"

    def test_tombstone_flag_on_either_level_hides_the_row(self):
        from memory.delete import _is_tombstoned

        assert _is_tombstoned({"metadata": {"dream_tombstoned": True}})
        assert _is_tombstoned({"metadata": {"metadata": {"dream_tombstoned": True}}})
        assert _is_tombstoned({"metadata": {"dream_tombstoned": True, "metadata": {"scope": "global"}}})
        assert not _is_tombstoned({"metadata": {"metadata": {"dream_tombstoned": False}}})
        assert not _is_tombstoned({})

    def test_outer_flagged_wrapped_row_is_not_a_candidate(self, service):
        fake = _FakeQdrant(service, [
            ("dead", {"user_id": "u1", "hash": "h1", "data": self.TEXT, "created_at": "2025-06-01T00:00:00",
                      "metadata": {"dream_tombstoned": True, "metadata": {"scope": "global"}}}, _text_vec(self.TEXT)),
            ("live", {"user_id": "u1", "hash": "h2", "data": self.TEXT, "created_at": "2025-01-01T00:00:00",
                      "metadata": {"metadata": {"scope": "global"}}}, _text_vec(self.TEXT)),
        ])
        result = service.dedup_memories("u1")
        assert result["semantic_duplicates_removed"] == 0
        assert _tombstoned(service) == []


class TestSemanticDedupMissedDuplicates:
    TEXT = "Deploys go through the staging gate before prod"

    def test_other_partition_copies_cannot_crowd_out_an_in_partition_duplicate(self, service):
        """Global rows have no project_id to push down; 12 identical copies in
        projects score ahead of the real duplicate and used to fill top_k=5."""
        rows = [_row("dup-old", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
                _row("dup-new", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="private")]
        rows += [
            _row(f"x{i}", self.TEXT, f"2025-03-{i + 1:02d}T00:00:00",
                 scope="global", visibility="private", project_id=f"proj-{i}")
            for i in range(12)
        ]
        fake = _FakeQdrant(service, rows)

        result = service.dedup_memories("u1")

        assert _merged(service) == {"dup-old"}
        assert result["semantic_duplicates_removed"] == 1
        assert max(c["top_k"] for c in fake.search_calls) > 5

    def test_the_search_page_widens_only_while_it_is_full_and_above_threshold(self, service):
        fake = _FakeQdrant(service, [
            _row("a", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("b", "Unrelated: the cafe opens at nine", "2025-06-01T00:00:00",
                 scope="global", visibility="private"),
        ])
        service.dedup_memories("u1")
        assert {c["top_k"] for c in fake.search_calls} == {10}  # first page was not full: no widening


class TestDedupStreamsThePool:
    def test_pool_is_read_in_pages_not_one_big_scroll(self, service):
        rows = [
            _row(f"r{i}", f"Distinct fact number {i} about topic {i * 7}", "2025-01-01T00:00:00",
                 scope="global", visibility="private")
            for i in range(7)
        ]
        fake = _FakeQdrant(service, rows)
        from config import settings

        with patch.object(settings, "dedup_batch_size", 3):
            result = service.dedup_memories("u1")

        assert result["total_checked"] == 7
        # pass 1 (count keys) + phase 2 over 3 pages; no duplicates so no row pass
        assert fake.scroll_calls >= 6

    def test_pages_are_yielded_lazily(self, service):
        fake = _FakeQdrant(service, [
            _row(f"r{i}", f"fact {i}", "2025-01-01T00:00:00", scope="global") for i in range(5)
        ])
        pages = service._iter_user_memory_pages("u1", batch_size=2)
        first = next(pages)
        assert len(first) == 2 and fake.scroll_calls == 1


class TestSemanticDedupFlagOffIsTheHardDeleteWithCascade:
    """The default: DEDUP_SEMANTIC_TOMBSTONE_ENABLED off keeps today's delete + #470 cascade."""

    TEXT = "Deploys go through the staging gate before prod"

    def test_default_is_off(self):
        from config import Settings

        assert Settings(_env_file=None).dedup_semantic_tombstone_enabled is False

    def test_older_row_is_hard_deleted_with_graph_cleanup(self, service):
        fake = _FakeQdrant(service, [
            _row("old", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private", times_derived=2),
            _row("new", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        with patch.object(service, "_delete_qdrant_memory_with_graph_cleanup",
                          wraps=service._delete_qdrant_memory_with_graph_cleanup) as cleanup:
            result = service.dedup_memories("u1")

        assert result["semantic_mode"] == "delete"
        assert _deleted(service) == ["old"] and "old" not in fake.payloads
        assert _tombstone_calls(service) == []
        assert cleanup.call_args.args[0] == "old" and cleanup.call_args.kwargs["collector"] is not None
        assert fake.payloads["new"]["metadata"]["times_derived"] == 3  # reinforcement still moves

    def test_flag_on_does_not_cascade_or_delete(self, service, tombstone_on):
        _FakeQdrant(service, [
            _row("old", self.TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
            _row("new", self.TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
        ])
        with patch.object(service, "_delete_qdrant_memory_with_graph_cleanup") as cleanup:
            result = service.dedup_memories("u1")
        cleanup.assert_not_called()
        assert result["semantic_mode"] == "tombstone" and _deleted(service) == []
