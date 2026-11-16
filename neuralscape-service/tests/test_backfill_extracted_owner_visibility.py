"""The one-off backfill that stamps owner/tier on already-stored extracted rows.

Runs against Qdrant local mode (in-memory), no service.
"""

import importlib.util
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "backfill_extracted_owner_visibility.py"
_spec = importlib.util.spec_from_file_location("backfill_extracted_owner_visibility", _PATH)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

COLL = "t"


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    c.create_collection(COLL, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    return c


def _put(client, pid, user_id, **meta):
    payload = {"data": f"fact {pid}", "metadata": {"category": "decision", **meta}}
    if user_id is not None:
        payload["user_id"] = user_id
    client.upsert(COLL, [PointStruct(id=pid, vector=[0.1, 0.2], payload=payload)])


def _meta(client, pid):
    return client.retrieve(COLL, [pid])[0].payload["metadata"]


def test_dry_run_writes_nothing_and_counts(client):
    _put(client, 1, "alice", source="conversation")
    out = bf.backfill(client, COLL)
    assert out["dry_run"] is True and out["rows_to_stamp"] == 1 and out["stamped"] == 0
    assert "owner_user_id" not in _meta(client, 1) and "visibility" not in _meta(client, 1)


def test_apply_stamps_owner_from_payload_user_id_and_private(client):
    _put(client, 1, "alice", source="conversation")
    _put(client, 2, "bob", source="conversation")
    out = bf.backfill(client, COLL, apply=True)
    assert out["stamped"] == 2 and out["owner_to_set"] == 2 and out["visibility_to_set"] == 2
    assert _meta(client, 1)["owner_user_id"] == "alice" and _meta(client, 1)["visibility"] == "private"
    assert _meta(client, 2)["owner_user_id"] == "bob"
    assert _meta(client, 1)["category"] == "decision"  # rest of metadata preserved


def test_idempotent_second_run_is_a_noop(client):
    _put(client, 1, "alice", source="conversation")
    bf.backfill(client, COLL, apply=True)
    again = bf.backfill(client, COLL, apply=True)
    assert again["rows_to_stamp"] == 0 and again["stamped"] == 0


def test_only_missing_fields_and_never_changes_an_existing_tier(client):
    _put(client, 1, "alice", source="conversation", visibility="shared")          # tier kept
    _put(client, 2, "alice", source="conversation", owner_user_id="alice")        # owner kept
    _put(client, 3, "alice", source="conversation", visibility="MemoryVisibility.PRIVATE")  # legacy value untouched
    out = bf.backfill(client, COLL, apply=True)
    assert _meta(client, 1)["visibility"] == "shared" and _meta(client, 1)["owner_user_id"] == "alice"
    assert _meta(client, 2)["visibility"] == "private" and _meta(client, 2)["owner_user_id"] == "alice"
    assert _meta(client, 3)["visibility"] == "MemoryVisibility.PRIVATE"
    assert out["visibility_to_set"] == 1 and out["owner_to_set"] == 2


def test_only_conversation_provenance_is_touched(client):
    _put(client, 1, "alice", source="explicit")
    _put(client, 2, "alice")                              # no source at all
    _put(client, 3, "alice", source="conversation")
    out = bf.backfill(client, COLL, apply=True)
    assert out["rows_to_stamp"] == 1
    assert "visibility" not in _meta(client, 1) and "visibility" not in _meta(client, 2)
    assert _meta(client, 3)["visibility"] == "private"


def test_extra_source_is_opt_in(client):
    _put(client, 1, "alice", source="compiler")
    assert bf.backfill(client, COLL, apply=True)["rows_to_stamp"] == 0
    assert bf.backfill(client, COLL, apply=True, sources=("compiler",))["rows_to_stamp"] == 1


def test_row_without_top_level_user_id_gets_tier_only_and_is_reported(client):
    _put(client, 1, None, source="conversation")
    out = bf.backfill(client, COLL, apply=True)
    assert out["skipped_no_user_id"] == 1
    assert _meta(client, 1)["visibility"] == "private" and "owner_user_id" not in _meta(client, 1)


def test_batches_across_pages(client):
    for i in range(1, 8):
        _put(client, i, "alice", source="conversation")
    out = bf.backfill(client, COLL, apply=True, batch_size=3)
    assert out["stamped"] == 7 and out["scanned"] == 7
    assert all(_meta(client, i)["visibility"] == "private" for i in range(1, 8))
