"""V3 engine switch: authorized recall when the engine misbehaves or fails, and preserved taxonomy/code routing.

Same real path as ``test_v3_engine_switch_integration`` (HTTP API, Redis-woken ledger worker, real Qdrant and
pgvector, the router), reusing its harness.  Two hazards the end-to-end test does not exercise:

* the serving engine returns a record the caller may not read (here a stale label on another user's private
  memory): hydration through the caller's batch read must drop it.  The batch read is the product's
  ``MemoryService.get_memories_by_ids`` with its own readability rule; only the row store under it is a
  stand-in serving the stored payload shape;
* the serving engine fails: the request must still be answered, by the legacy vector path, and only with what
  the caller may read.  The legacy path's own retrieval needs the real memory stack (mem0, embeddings), so it
  is a labelled stand-in here that applies the same own-plus-shared rule; what is asserted is that the switch
  hands recall to it exactly when the engine fails, and never serves an engine result it could not authorize.

Plus: with routing enabled, a knowledge-adapter category is accepted, carried to the memory service and
projected; an unregistered category is still refused; code-graph engine selection and the knowledge-system
registry are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_v3_engine_switch_integration import (  # noqa: F401  (fixture and helpers)
    PG_DSN,
    QDRANT_URL,
    REDIS_URL,
    U1,
    ServiceUnderTest,
    qdrant_adapter,
    world,
)

from ledger_env import require_env  # noqa: E402


@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    require_env(Redis=REDIS_URL, Qdrant=QDRANT_URL, PostgreSQL=PG_DSN)


import engine_adapters_switch as sw  # noqa: E402
from engine_ports import EngineRecord, OperationContext  # noqa: E402
from memory_service import MemoryService  # noqa: E402
from write_producer import DEFAULT_TENANT  # noqa: E402


class _LegacyPathReached(Exception):
    pass


class _RowPoints:
    """The vector-store ``retrieve`` the product's batch read calls, serving the rows as stored points.

    Payloads have the stored shape (``user_id``, ``data`` and the nested ``metadata`` with owner and
    visibility), so the product's own ``get_memories_by_ids`` and its readability rule decide what is served.
    """

    def __init__(self, rows):
        self.vector_store = SimpleNamespace(client=self)
        self._rows = rows

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [
            SimpleNamespace(id=memory_id, payload={
                "user_id": row["owner"], "data": row["content"],
                "metadata": {"owner_user_id": row["owner"], "visibility": row["visibility"],
                             "project_id": row["project"], "tags": row["tags"]},
            })
            for memory_id in ids if (row := self._rows.get(memory_id)) is not None
        ]


class LegacyRecordingService(ServiceUnderTest):
    """The real ``search`` (routed hook first) and the real batch read that hydrates engine hits; the legacy
    vector path is a labelled, authorizing stand-in."""

    _hydrating = False

    def get_memories_by_ids(self, ids, caller_user_id):
        # The product's batch read and readability rule, not the harness's: only the row store is stood in.
        self._hydrating = True
        try:
            return MemoryService.get_memories_by_ids(self, ids, caller_user_id)
        finally:
            self._hydrating = False

    def _get_memory(self):
        if self._hydrating:
            return _RowPoints(self.rows)
        raise _LegacyPathReached  # the first thing the legacy path does

    def store_raw(self, **kw):
        self.categories = getattr(self, "categories", {})
        result = super().store_raw(**kw)
        mems = result[0] if isinstance(result, tuple) else result
        self.categories[mems[0].id] = kw.get("category")
        return result

    def search(self, query, user_id, *args, **kwargs):
        try:
            return super().search(query, user_id, *args, **kwargs)
        except _LegacyPathReached:
            self.legacy_calls = getattr(self, "legacy_calls", 0) + 1
            words = set(query.lower().split())
            return [
                self._response(memory_id, row) for memory_id, row in self.rows.items()
                if (row["owner"] == user_id or row["visibility"] == "shared")
                and words & set(row["content"].lower().split())
            ]


def _install(w):
    w.service.__class__ = LegacyRecordingService
    w.service.legacy_calls = 0
    w.service.categories = {}


def _ready(w):
    w.write(U1, "tea notes number one about green tea")
    w.write(U1, "tea tasting in the morning")
    w.write("u2", "private note of the other user about tea")
    w.write("u2", "shared tea calendar for the team", visibility="shared")
    w.drain()
    assert w.admin("catch-up").json()["recall_ready"] is True


def test_an_engine_record_the_caller_may_not_read_is_never_served(world):
    w = world
    _install(w)
    _ready(w)
    private_of_u2 = w.memory_id("u2", "private note of the other user about tea")
    assert private_of_u2 not in w.search(U1, "tea")
    # The serving engine now holds a stale label claiming that private memory is shared.
    ctx = OperationContext(tenant_id=DEFAULT_TENANT, correlation_id="t", deadline_ms=50)
    qdrant_adapter(w).apply(ctx, EngineRecord(
        record_id=private_of_u2, revision=2, tenant_id=DEFAULT_TENANT,
        text="private note of the other user about tea", labels=sw.labels_for("u2", "shared", None),
    ))
    hits = [h.record_id for h in w.switch.recall(U1, "tea", 50)]
    assert private_of_u2 in hits  # the engine itself would hand it out
    served = w.search(U1, "tea")
    assert private_of_u2 not in served  # hydration through the caller's batch read drops it
    assert w.memory_id(U1, "tea tasting in the morning") in served
    assert w.service.legacy_calls == 0  # served by the engine, not the fallback


def test_when_the_serving_engine_fails_recall_falls_back_and_stays_authorized(world):
    w = world
    _install(w)
    _ready(w)
    engine_served = set(w.search(U1, "tea"))
    assert w.service.legacy_calls == 0 and engine_served
    qdrant_adapter(w)._store.client.delete_collection(w.collection)  # the serving engine breaks
    served = set(w.search(U1, "tea"))
    assert w.service.legacy_calls == 1  # handed to the legacy path, the request still answered
    assert served == engine_served  # own + shared, as the engine served them
    assert w.memory_id("u2", "private note of the other user about tea") not in served
    assert w.memory_id("u2", "shared tea calendar for the team") in served


def test_taxonomy_and_code_routing_are_preserved_with_routing_enabled(world, tmp_path):
    from unittest.mock import Mock, patch

    import networkx as nx

    from adapters.code_graph.graphify_engine import GraphifyJsonEngine
    from adapters.code_graph.query import get_engine
    from knowledge import get_system, list_systems
    from schemas import MEMORY_CATEGORIES

    w = world
    _install(w)
    assert "setup" in MEMORY_CATEGORIES  # registered by the trading knowledge adapter
    response = w.client.post("/v1/memories/raw", json={"content": "kangaroo tail setup at support", "user_id": U1,
                                                       "category": "setup"})
    assert response.status_code == 202, response.text
    refused = w.client.post("/v1/memories/raw", json={"content": "x", "user_id": U1, "category": "not_a_category"})
    assert refused.status_code in (400, 422), refused.text
    w.drain()
    memory_id = w.memory_id(U1, "kangaroo tail setup at support")
    assert w.service.categories.get(memory_id) == "setup"  # the adapter category reached the service unchanged
    ctx = OperationContext(tenant_id=DEFAULT_TENANT, correlation_id="t", deadline_ms=50)
    assert qdrant_adapter(w).get(ctx, memory_id) is not None  # and was projected by the routed engine

    graph_json = tmp_path / "graph.json"
    graph_json.write_text('{"nodes": [], "edges": []}')
    code_settings = Mock()
    code_settings.code_graph_json_path = str(graph_json)
    with patch("adapters.code_graph.query.load_code_graph", return_value=nx.Graph()):
        assert isinstance(get_engine(None, U1, code_settings), GraphifyJsonEngine)
    assert "ns-memory" in list_systems() and get_system("ns-memory") is not None
