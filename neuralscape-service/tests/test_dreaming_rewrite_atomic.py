"""Dream rewrites keep the BM25 sparse vector and patch atomically.

Pre-fix ``_rewrite_content`` upserted ``PointStruct(id, vector=<dense list>,
payload=<whole payload from a stale read>)``. An upsert replaces the point,
so (1) the ``bm25`` sparse vector the write path stores next to the dense one
vanished (every merged / rewritten / reframed row lost its lexical leg) and
(2) a concurrent ``times_derived`` bump or tombstone landing between the
retrieve and the upsert was overwritten.

Unit tests: the Qdrant client is a recording fake that applies
``set_payload`` / ``update_vectors`` / ``delete_payload`` like the server.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import SparseVector

from extensions.dreaming import consolidate
from extensions.dreaming.consolidate import PoolBatch
from memory_service import content_hash


def rewrite_view(client, memory_id: str) -> dict:
    """Reconstruct what ``_rewrite_content`` wrote for one point from the
    mock client's ``set_payload`` calls: top-level keys at the top, nested
    ``key="metadata"`` patches under ``"metadata"`` (patch keys only)."""
    out: dict = {"metadata": {}}
    for call in client.set_payload.call_args_list:
        if call.kwargs.get("points") != [memory_id]:
            continue
        if call.kwargs.get("key") == "metadata":
            out["metadata"].update(call.kwargs["payload"])
        else:
            out.update(call.kwargs["payload"])
    return out


class FakeQdrant:
    """Server-like point store: applies patches to a live stored payload."""

    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads
        self.vectors: dict[str, object] = {}
        self.upsert = MagicMock(name="upsert")
        self.on_retrieve = None  # hook: simulate a concurrent writer
        self.calls: list[str] = []

    def retrieve(self, collection_name=None, ids=None, **kw):
        # Snapshot (deep enough) BEFORE the concurrent hook runs: the
        # caller's view is stale by the time it writes.
        import copy

        out = [
            SimpleNamespace(id=i, payload=copy.deepcopy(self.payloads[i]))
            for i in ids or [] if i in self.payloads
        ]
        if self.on_retrieve:
            self.on_retrieve(self)
        return out

    def set_payload(self, collection_name=None, payload=None, points=None, key=None):
        self.calls.append("set_payload" + (":" + key if key else ""))
        for pid in points:
            target = self.payloads[pid]
            if key:
                target.setdefault(key, {}).update(payload)
            else:
                target.update(payload)

    def delete_payload(self, collection_name=None, keys=None, points=None):
        self.calls.append("delete_payload")
        for pid in points:
            for k in keys:
                self.payloads[pid].pop(k, None)

    def update_vectors(self, collection_name=None, points=None):
        self.calls.append("update_vectors")
        for pv in points:
            self.vectors[pv.id] = pv.vector


def _service(payloads, *, bm25_slot, sparse=None):
    client = FakeQdrant(payloads)
    vs = MagicMock()
    vs.client = client
    vs._has_bm25_slot = bm25_slot
    vs._encode_bm25 = MagicMock(return_value=sparse)
    mem = MagicMock()
    mem.vector_store = vs
    mem.embedding_model.embed.return_value = [0.25] * 8
    service = MagicMock()
    service._get_memory.return_value = mem
    service._memory = mem
    service._bridge = None
    return service, client, vs


SPARSE = SparseVector(indices=[3, 9], values=[1.0, 0.5])


def _batch(ids):
    return PoolBatch(
        pool="p", group_id="p", visibility="shared", owner_user_id=None,
        project_id=None, memories=[{"memory_id": i, "content": "x"} for i in ids],
    )


class TestBm25Preserved:
    def test_rewrite_writes_dense_and_bm25_named_vectors(self):
        service, client, vs = _service(
            {"m1": {"data": "old text", "metadata": {}}}, bm25_slot=True, sparse=SPARSE
        )
        consolidate._rewrite_content(service, "m1", "kubernetes migration notes")
        vs._encode_bm25.assert_called_once_with("kubernetes migration notes")
        assert client.vectors["m1"] == {"": [0.25] * 8, "bm25": SPARSE}
        client.upsert.assert_not_called()

    def test_reframe_keeps_bm25(self):
        service, client, _ = _service(
            {"m1": {"data": "plans to ship in July", "metadata": {}}},
            bm25_slot=True, sparse=SPARSE,
        )
        consolidate._rewrite_content(
            service, "m1", "[stale: July passed] plans to ship in July", reframed=True
        )
        assert "bm25" in client.vectors["m1"]
        assert client.payloads["m1"]["metadata"]["dream_temporal_reframed"] is True

    @pytest.mark.asyncio
    async def test_merge_survivor_keeps_bm25(self):
        service, client, _ = _service(
            {"a": {"data": "x", "metadata": {}}, "b": {"data": "y", "metadata": {}}},
            bm25_slot=True, sparse=SPARSE,
        )
        result = await consolidate.apply_actions(
            service, _batch(["a", "b"]),
            [{"type": "merge", "memory_ids": ["a", "b"], "survivor_id": "a",
              "content": "merged fact", "confidence": 0.9}],
            dry_run=False,
        )
        assert not result.errors
        assert client.vectors["a"]["bm25"] == SPARSE
        client.upsert.assert_not_called()

    def test_collection_without_bm25_slot_updates_dense_only(self):
        service, client, vs = _service(
            {"m1": {"data": "old", "metadata": {}}}, bm25_slot=False
        )
        consolidate._rewrite_content(service, "m1", "new")
        vs._encode_bm25.assert_not_called()
        assert client.vectors["m1"] == [0.25] * 8

    def test_encoder_unavailable_still_updates_dense(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "metadata": {}}}, bm25_slot=True, sparse=None
        )
        consolidate._rewrite_content(service, "m1", "new")
        assert client.vectors["m1"] == {"": [0.25] * 8}

    def test_stale_text_lemmatized_shadow_is_dropped(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "text_lemmatized": "old", "metadata": {}}},
            bm25_slot=True, sparse=SPARSE,
        )
        consolidate._rewrite_content(service, "m1", "new")
        assert "text_lemmatized" not in client.payloads["m1"]


class TestAtomicPatch:
    def test_concurrent_times_derived_bump_survives_rewrite(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "hash": "h", "metadata": {"times_derived": 1}}},
            bm25_slot=True, sparse=SPARSE,
        )

        def bump(c):
            c.payloads["m1"]["metadata"]["times_derived"] = 7  # write path races in
            c.on_retrieve = None

        client.on_retrieve = bump
        consolidate._rewrite_content(service, "m1", "new text")
        meta = client.payloads["m1"]["metadata"]
        assert meta["times_derived"] == 7  # not clobbered by the stale snapshot
        assert meta["dream_prev_content"] == "old"
        assert client.payloads["m1"]["data"] == "new text"

    def test_concurrent_tombstone_survives_rewrite(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "metadata": {}}}, bm25_slot=False
        )

        def tomb(c):
            c.payloads["m1"]["metadata"]["dream_tombstoned"] = True
            c.on_retrieve = None

        client.on_retrieve = tomb
        consolidate._rewrite_content(service, "m1", "new")
        assert client.payloads["m1"]["metadata"]["dream_tombstoned"] is True

    def test_merge_sums_times_derived_and_unions_derived_from(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "metadata": {"derived_from": ["s"], "times_derived": 2}}},
            bm25_slot=False,
        )
        consolidate._rewrite_content(
            service, "m1", "merged", derived_from_add=["l1", "s", "l2"],
            extra_meta={"times_derived": 5},
        )
        meta = client.payloads["m1"]["metadata"]
        assert meta["derived_from"] == ["s", "l1", "l2"]
        assert meta["times_derived"] == 5

    def test_undo_stash_and_hash_follow_content(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "hash": "stale", "metadata": {}}}, bm25_slot=False
        )
        consolidate._rewrite_content(service, "m1", "fresh")
        p = client.payloads["m1"]
        assert p["metadata"]["dream_prev_content"] == "old"
        assert p["hash"] == content_hash("fresh")
        assert "dream_rewritten_at" in p["metadata"]
        assert p["updated_at"]

    def test_noop_rewrite_does_not_touch_stash(self):
        service, client, _ = _service(
            {"m1": {"data": "same", "metadata": {"dream_prev_content": "older"}}},
            bm25_slot=False,
        )
        consolidate._rewrite_content(service, "m1", "same")
        assert client.payloads["m1"]["metadata"]["dream_prev_content"] == "older"

    def test_metadata_lands_before_visible_text_flips(self):
        service, client, _ = _service(
            {"m1": {"data": "old", "metadata": {}}}, bm25_slot=False
        )
        consolidate._rewrite_content(service, "m1", "new")
        assert client.calls == ["set_payload:metadata", "update_vectors", "set_payload"]

    def test_missing_point_raises(self):
        service, _, _ = _service({}, bm25_slot=False)
        with pytest.raises(ValueError):
            consolidate._rewrite_content(service, "nope", "x")
