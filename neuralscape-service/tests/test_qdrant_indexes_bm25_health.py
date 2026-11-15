"""Payload indexes for every filtered field, and the BM25-slot health signal.

Audit findings:
- vec.qdrant-point-store: no payload index on hash / metadata.project_id /
  metadata.category (dedup probe and category filters scanned the payload).
- vec.bm25-sparse: a pre-v3 collection silently lost the keyword leg.
"""

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import memory.core as core
from config import settings
from memory_service import MemoryService

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"


class FakeQdrantClient:
    """Idempotent like Qdrant: re-creating an existing index is a no-op."""

    def __init__(self):
        self.indexes: dict[str, object] = {}
        self.calls = 0

    def create_payload_index(self, collection_name, field_name, field_schema, wait=True):
        self.calls += 1
        self.indexes[field_name] = field_schema


def _service(has_bm25=True, client=None):
    svc = MemoryService()
    svc._memory = SimpleNamespace(
        vector_store=SimpleNamespace(client=client or FakeQdrantClient(), _has_bm25_slot=has_bm25)
    )
    return svc


@pytest.fixture(autouse=True)
def _reset_warned():
    core._bm25_warned = False
    yield
    core._bm25_warned = False


# ---------------------------------------------------------------- indexes


def test_dedup_and_category_indexes_requested():
    client = FakeQdrantClient()
    svc = _service(client=client)
    svc._ensure_filter_indexes()
    for field in ("hash", "metadata.project_id", "metadata.category"):
        assert field in client.indexes, field
    assert str(client.indexes["hash"]).lower().endswith("keyword")


def test_second_ensure_is_a_noop_and_never_raises():
    client = FakeQdrantClient()
    svc = _service(client=client)
    svc._ensure_filter_indexes()
    first = client.calls
    svc._ensure_filter_indexes()
    assert client.calls == first
    # A fresh service on an already-indexed collection (restart) is fine too.
    _service(client=client)._ensure_filter_indexes()
    assert set(client.indexes) >= {f for f, _ in core.CoreMixin._FILTER_INDEX_FIELDS}


def test_every_filtered_metadata_key_is_indexed():
    """Grep the service's own filters: nothing it filters on may be unindexed."""
    keys: set[str] = set()
    for path in MEMORY_DIR.glob("*.py"):
        keys |= set(re.findall(r'(?:FieldCondition|PayloadField)\(\s*key="([\w.]+)"', path.read_text()))
        keys |= set(re.findall(r'PayloadField\(key="([\w.]+)"', path.read_text()))
    indexed = {f for f, _ in core.CoreMixin._FILTER_INDEX_FIELDS}
    # user_id: mem0's own store; created_at: _ensure_created_at_index;
    # metadata: provenance JSON blob, not a filter key.
    exempt = {"user_id", "created_at", "metadata"}
    assert not (keys - indexed - exempt), sorted(keys - indexed - exempt)


# ----------------------------------------------------------------- health


def test_capabilities_true_when_store_has_slot():
    assert _service(True).vector_store_capabilities() == {"bm25_sparse": True}


def test_capabilities_false_with_detail_when_slot_missing():
    caps = _service(False).vector_store_capabilities()
    assert caps["bm25_sparse"] is False
    assert "BM25" in caps["detail"] and "dense-only" in caps["detail"]


def test_capabilities_unknown_before_init():
    assert MemoryService().vector_store_capabilities()["bm25_sparse"] is None


def test_missing_slot_warns_once_per_process(caplog):
    svc = _service(False)
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        for _ in range(3):
            svc.vector_store_capabilities()
            svc._filter_indexes_ok = False
            svc._ensure_filter_indexes()
    warnings = [r for r in caplog.records if "vector_store_bm25_slot_missing" in r.getMessage()]
    assert len(warnings) == 1


def test_present_slot_never_warns(caplog):
    svc = _service(True)
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        svc.vector_store_capabilities()
        svc._ensure_filter_indexes()
    assert not [r for r in caplog.records if "bm25" in r.getMessage().lower()]


@pytest.mark.parametrize("has_slot", [True, False])
def test_health_endpoint_reports_bm25_without_failing(has_slot):
    from unittest.mock import AsyncMock

    tm = MagicMock()
    tm.pool = MagicMock()
    tm.pool.ping = AsyncMock(return_value=True)
    svc = _service(has_slot)
    svc._graphiti = MagicMock()
    orig_tm, orig_svc = main._task_manager, main._service
    main._task_manager, main._service = tm, svc
    try:
        resp = TestClient(main.app, raise_server_exceptions=False).get("/health")
    finally:
        main._task_manager, main._service = orig_tm, orig_svc
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_store"]["bm25_sparse"] is has_slot
    assert ("detail" in body["vector_store"]) is (not has_slot)
    assert body["checks"]["vector_store"] == "ok"
