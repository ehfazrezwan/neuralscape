"""Conversation-extracted rows carry owner_user_id and visibility=private.

Gate failure this closes: integration_tests/test_hybrid_integration.py::
test_extraction_on_stores_verbatim_source_facts_with_evidence asserted
``reread.owner_user_id == uid`` and got None, because ``_batch_store_facts``
never stamped the owner (only the payload's top-level user_id carried it).
Atlas: model.visibility-tiers (conversation-extracted rows carry no
metadata.visibility).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memory.delete import _dedup_partition, _dedup_partition_filters
from memory_service import MemoryService


@pytest.fixture
def service(monkeypatch):
    from tests.test_dedup import _UpRedis

    up = _UpRedis()  # dedup takes pool locks and does nothing destructive without Redis
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: up)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = [0.1] * 8
    svc._memory.embedding_model.embed_batch.return_value = [[0.1] * 8]
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    svc._memory.vector_store.insert.return_value = None
    svc._memory.db.add_history.return_value = None
    return svc


def _inserted_payload(service):
    call = service._memory.vector_store.insert.call_args
    return call.kwargs["payloads"][0]


class TestWriteStampsOwnerAndTier:
    def test_new_row_carries_owner_and_private(self, service):
        [resp] = service._batch_store_facts(
            facts=[("decision", "Decided to use PostgreSQL for JSONB")], user_id="alice",
        )
        meta = _inserted_payload(service)["metadata"]
        assert meta["owner_user_id"] == "alice"
        assert meta["visibility"] == "private"
        assert resp.owner_user_id == "alice" and resp.visibility == "private"

    def test_sensitive_row_still_private_and_owned(self, service):
        service._batch_store_facts(
            facts=[("decision", "Approved a $50,000 client contract renewal")], user_id="alice",
        )
        meta = _inserted_payload(service)["metadata"]
        assert meta["visibility"] == "private" and meta["owner_user_id"] == "alice"

    def test_no_caller_can_widen_the_tier(self, service):
        # the path exposes no visibility parameter at all
        import inspect

        assert "visibility" not in inspect.signature(service._batch_store_facts).parameters


class TestLegacyRowsStillReadTheirOwner:
    def test_point_without_owner_metadata_falls_back_to_payload_user_id(self, service):
        point = SimpleNamespace(id="p1", payload={
            "data": "legacy fact", "user_id": "alice",
            "metadata": {"category": "decision", "scope": "global"},
        })
        resp = service._point_to_response(point)
        assert resp.owner_user_id == "alice"
        assert resp.visibility is None  # the read model does not invent a tier

    def test_metadata_owner_wins_over_payload_user_id(self, service):
        point = SimpleNamespace(id="p1", payload={
            "data": "x", "user_id": "writer",
            "metadata": {"owner_user_id": "owner", "visibility": "shared"},
        })
        assert service._point_to_response(point).owner_user_id == "owner"

    def test_mem0_shaped_result_uses_top_level_user_id(self, service):
        resp = service._mem_to_response({"id": "m", "memory": "x", "user_id": "alice", "metadata": {}})
        assert resp.owner_user_id == "alice"

    def test_nothing_known_stays_none(self, service):
        assert service._mem_to_response({"id": "m", "memory": "x", "metadata": {}}).owner_user_id is None


class TestDedupPartitionParity:
    LEGACY = {"user_id": "u", "metadata": {"scope": "global", "category": "decision"}}
    NEW = {"user_id": "u", "metadata": {"scope": "global", "category": "decision",
                                         "owner_user_id": "u", "visibility": "private"}}

    def test_legacy_and_new_private_rows_share_a_partition(self):
        assert _dedup_partition(self.LEGACY) == _dedup_partition(self.NEW)

    def test_other_tiers_stay_distinct(self):
        shared = {"user_id": "u", "metadata": {"scope": "global", "visibility": "shared"}}
        standard = {"user_id": "u", "metadata": {"scope": "global", "visibility": "standard"}}
        assert len({_dedup_partition(r) for r in (self.NEW, shared, standard)}) == 3

    def test_legacy_stringified_enum_normalises(self):
        odd = {"user_id": "u", "metadata": {"scope": "global", "visibility": "MemoryVisibility.PRIVATE"}}
        assert _dedup_partition(odd) == _dedup_partition(self.NEW)

    def test_unrecognised_value_is_its_own_partition(self):
        odd = {"user_id": "u", "metadata": {"scope": "global", "visibility": "weird"}}
        assert _dedup_partition(odd) != _dedup_partition(self.NEW)

    def test_semantic_pair_is_found_from_the_unstamped_side(self):
        # A stamped-private source pushes metadata.visibility=private into the
        # Qdrant filter (so it cannot see unstamped twins), but the unstamped
        # twin's own search pushes no visibility filter and finds the stamped
        # row; the strict _dedup_partition check on the hit then matches.
        assert "metadata.visibility" not in _dedup_partition_filters("u", self.LEGACY)
        assert _dedup_partition_filters("u", self.NEW)["metadata.visibility"] == "private"
        assert _dedup_partition(self.LEGACY) == _dedup_partition(self.NEW)

    def test_dedup_merges_a_new_row_with_its_legacy_twin(self, service):
        # exact phase: same hash, one legacy, one stamped -> one survivor
        rows = [
            {"id": "old", "payload": {**self.LEGACY, "hash": "H", "created_at": "2026-01-01"}},
            {"id": "new", "payload": {**self.NEW, "hash": "H", "created_at": "2026-02-01"}},
        ]
        service._iter_user_memory_pages = MagicMock(side_effect=lambda uid, bs: iter([rows]))
        service._live_rows = MagicMock(side_effect=lambda r: r)  # the re-read after the pool lock
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()
        out = service.dedup_memories("u", semantic=False)
        assert out["exact_duplicates_removed"] == 1
        assert service._delete_qdrant_memory_with_graph_cleanup.call_args[0][0] == "old"
