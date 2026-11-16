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

    def get_collections(self):
        """The /health readiness probe makes this real round trip."""
        return []

    def create_payload_index(self, collection_name, field_name, field_schema, wait=True):
        self.calls += 1
        self.indexes[field_name] = field_schema


def _service(has_bm25=True, client=None, encoder=None):
    svc = MemoryService()
    svc._memory = SimpleNamespace(
        vector_store=SimpleNamespace(
            client=client or FakeQdrantClient(), _has_bm25_slot=has_bm25, _bm25_encoder=encoder
        )
    )
    return svc


@pytest.fixture(autouse=True)
def _reset_warned():
    core._bm25_warned = False
    core._bm25_encoder_warned = False
    yield
    core._bm25_warned = False
    core._bm25_encoder_warned = False


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
    assert svc._filter_indexes_ok is True
    first = client.calls
    svc._ensure_filter_indexes()
    assert client.calls == first
    # A fresh service on an already-indexed collection (restart) is fine too.
    _service(client=client)._ensure_filter_indexes()
    assert set(client.indexes) >= {f for f, _ in core.CoreMixin._FILTER_INDEX_FIELDS}


# Production code that filters the memories collection. scripts/ are one-off
# operator tools; adapters/code_graph filters its OWN collection.
GUARDED_DIRS = ("memory", "okf", "knowledge", "extensions", "adapters/trading")
FILTER_CALLS = {"FieldCondition", "PayloadField", "IsEmptyCondition", "IsNullCondition"}
# Non-literal keys that are allowed, with the reason; each resolves to keys
# that ARE in _FILTER_INDEX_FIELDS (asserted below).
NON_LITERAL_ALLOWED = {
    "memory/provenance.py": (
        "prefix + 'graph_episode_uuid'/'graph_episode_name' (prefix is '' or 'metadata.')",
        {"graph_episode_uuid", "graph_episode_name"},
    ),
    "memory/edit.py": (
        "parent_key + '.revision' in the edit compare-and-set filter (parent_key is 'metadata' or 'metadata.metadata')",
        {"revision"},
    ),
}


def _filter_keys():
    """(literal keys, non-literal sites) from every filter call, via the AST."""
    import ast

    root = MEMORY_DIR.parent
    literal: set[str] = set()
    dynamic: list[tuple[str, int]] = []
    for d in GUARDED_DIRS:
        for path in sorted((root / d).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", getattr(func, "attr", None))
                if name not in FILTER_CALLS:
                    continue
                for kw in node.keywords:
                    if kw.arg != "key":
                        continue
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        literal.add(kw.value.value)
                    else:
                        dynamic.append((rel, node.lineno))
    return literal, dynamic


def test_every_filtered_metadata_key_is_indexed():
    """Scan memory/, okf/, knowledge/, extensions/, adapters/trading: no unindexed filter key."""
    literal, dynamic = _filter_keys()
    assert "metadata.category" in literal and "metadata.source_ref.external_id" in literal  # scan sees extensions
    indexed = {f for f, _ in core.CoreMixin._FILTER_INDEX_FIELDS}
    # user_id: mem0's own store; created_at: _ensure_created_at_index;
    # metadata: provenance JSON blob patch, not a filter key.
    exempt = {"user_id", "created_at", "metadata"}
    assert not (literal - indexed - exempt), sorted(literal - indexed - exempt)

    unexplained = [(f, ln) for f, ln in dynamic if f not in NON_LITERAL_ALLOWED]
    assert not unexplained, f"non-literal FieldCondition key (allowlist with a reason): {unexplained}"
    for _reason, suffixes in NON_LITERAL_ALLOWED.values():
        for suffix in suffixes:
            assert f"metadata.{suffix}" in indexed, suffix


# ---------------------------------------------- startup + failure handling


def test_service_init_alone_creates_indexes_without_a_search(monkeypatch):
    """_get_memory() must ensure indexes: workers dedup-probe before any search."""
    client = FakeQdrantClient()
    fake_memory = SimpleNamespace(vector_store=SimpleNamespace(client=client, _has_bm25_slot=True, _bm25_encoder=None))
    monkeypatch.setattr(core, "_memory_from_config", lambda cls, cfg: fake_memory)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    svc = MemoryService()
    svc._get_memory()  # no search, no timeline
    svc._index_ensure_thread.join(timeout=5)
    for field in ("hash", "metadata.project_id", "metadata.category", "created_at"):
        assert field in client.indexes, field


def test_startup_ensure_never_raises_on_broken_qdrant(monkeypatch):
    class Broken:
        def create_payload_index(self, **_):
            raise RuntimeError("qdrant down")

    fake_memory = SimpleNamespace(vector_store=SimpleNamespace(client=Broken(), _has_bm25_slot=True, _bm25_encoder=None))
    monkeypatch.setattr(core, "_memory_from_config", lambda cls, cfg: fake_memory)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    svc = MemoryService()
    assert svc._get_memory() is fake_memory
    svc._index_ensure_thread.join(timeout=5)


class FlakyClient(FakeQdrantClient):
    def __init__(self, fail_fields):
        super().__init__()
        self.fail_fields = set(fail_fields)

    def create_payload_index(self, collection_name, field_name, field_schema, wait=True):
        if field_name in self.fail_fields:
            raise RuntimeError("transient")
        super().create_payload_index(collection_name, field_name, field_schema, wait)


def test_failed_field_warns_once_and_is_rechecked_later(caplog, monkeypatch):
    client = FlakyClient({"hash"})
    svc = _service(client=client)
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        svc._ensure_filter_indexes()
        assert svc._filter_indexes_ok is False
        svc._ensure_filter_indexes()  # inside the cool-down: no retry, no second warning
        msgs = [r.getMessage() for r in caplog.records if "qdrant_payload_index_ensure_failed" in r.getMessage()]
        assert len(msgs) == 1 and "hash" in msgs[0] and "metadata.scope" not in msgs[0]
        assert "hash" not in client.indexes

        # Qdrant recovers; once the cool-down has passed the next call re-checks.
        client.fail_fields.clear()
        svc._filter_index_retry_at = 0.0
        svc._ensure_filter_indexes()
    assert "hash" in client.indexes
    assert svc._filter_indexes_ok is True
    assert len([r for r in caplog.records if "qdrant_payload_index_ensure_failed" in r.getMessage()]) == 1


def test_created_at_failure_warns_once_and_is_rechecked(caplog):
    client = FlakyClient({"created_at"})
    svc = _service(client=client)
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        svc._ensure_created_at_index()
        assert not svc._created_at_index_ok
        svc._ensure_created_at_index()  # cool-down: no retry, no second warning
        client.fail_fields.clear()
        svc._created_at_retry_at = 0.0
        svc._ensure_created_at_index()
    assert "created_at" in client.indexes and svc._created_at_index_ok is True
    msgs = [r for r in caplog.records if "ensure_failed fields=created_at" in r.getMessage()]
    assert len(msgs) == 1


def test_write_only_worker_recovers_from_a_failed_startup_ensure(monkeypatch):
    """No search ever runs: later _get_memory() calls (every write makes one) retry."""
    client = FlakyClient({"hash", "created_at"})
    fake_memory = SimpleNamespace(vector_store=SimpleNamespace(client=client, _has_bm25_slot=True, _bm25_encoder=None))
    monkeypatch.setattr(core, "_memory_from_config", lambda cls, cfg: fake_memory)
    monkeypatch.setattr(type(settings), "get_mem0_config", lambda self: {})
    svc = MemoryService()
    svc._get_memory()
    svc._index_ensure_thread.join(timeout=5)
    assert "hash" not in client.indexes and not svc._filter_indexes_ok

    svc._get_memory()  # inside the cool-down: nothing happens
    assert not svc._index_ensure_thread.is_alive() and "hash" not in client.indexes

    client.fail_fields.clear()  # Qdrant is up now
    svc._filter_index_retry_at = svc._created_at_retry_at = 1e-9  # cool-down elapsed
    svc._get_memory()
    svc._index_ensure_thread.join(timeout=5)
    assert "hash" in client.indexes and "created_at" in client.indexes
    assert svc._filter_indexes_ok and svc._created_at_index_ok


# ----------------------------------------------------------------- health


def test_capabilities_true_when_store_has_slot():
    assert _service(True).vector_store_capabilities() == {"bm25_sparse": True, "bm25_encoder": "not_loaded"}


def test_capabilities_false_when_slot_missing_detail_only_in_log(caplog):
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        caps = _service(False).vector_store_capabilities()
    assert caps == {"bm25_sparse": False, "bm25_encoder": "not_loaded"}  # no free text on the public payload
    assert any("dense-only" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "encoder,state", [(None, "not_loaded"), (False, "unavailable"), (object(), "loaded")]
)
def test_capabilities_report_encoder_state(encoder, state):
    assert _service(True, encoder=encoder).vector_store_capabilities() == {
        "bm25_sparse": True,
        "bm25_encoder": state,
    }


def test_unavailable_encoder_warns_once(caplog):
    svc = _service(True, encoder=False)
    with caplog.at_level(logging.WARNING, logger="memory.core"):
        for _ in range(3):
            svc.vector_store_capabilities()
    assert len([r for r in caplog.records if "bm25_encoder_unavailable" in r.getMessage()]) == 1


def test_capabilities_unknown_before_init():
    assert MemoryService().vector_store_capabilities() == {"bm25_sparse": None, "bm25_encoder": None}


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
    assert body["vector_store"] == {"bm25_sparse": has_slot, "bm25_encoder": "not_loaded"}
    assert "detail" not in body["vector_store"]
    assert body["checks"]["vector_store"] == "ok"
