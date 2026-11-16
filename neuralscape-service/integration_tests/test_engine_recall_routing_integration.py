"""Routed-engine recall in ``MemoryService.search`` on real stores (``ENGINE_ROUTING_RECALL_ENABLED=true``).

Real: a Qdrant server holds both the memory rows (the collection the service searches) and the routed engine's
projection collection; a SQLite ledger is the canonical source the engine is brought to canonical state from; the
switch is the one the service builds at start-up (``engine_adapters_switch.start``).  Stood in for: the embedder
(a deterministic token hash, no model is called).

Shown, with the flag on: a plain request is served by the engine and returns engine hits; a request the engine
cannot apply (a category filter) is served by the normal path and never consults the engine; and nothing returned
is another user's private row, another project's row, a consolidation tombstone or a reference-workspace row,
including rows whose state changed after they were projected.  With the flag off the engine is never consulted.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

import engine_env  # noqa: F401  (loads the third-party telemetry guard before mem0 or Graphiti can be imported)

pytestmark = [pytest.mark.integration]

DIMS = 64
TENANT = "tenant-recall"
ALICE, BOB = "nsit-alice", "nsit-bob"
QUERY = "kestrel nest"


def _id() -> str:
    return str(uuid.uuid4())


A_P1, A_P2, A_NONE, B_SHARED_P1, B_PRIVATE_P1, B_STALE, A_TOMB, A_REF = (_id() for _ in range(8))


class Rows:
    """The service's row view for the engine's canonical source (what ``get_memory`` returns)."""

    def __init__(self):
        self.rows: dict[str, SimpleNamespace] = {}

    def get_memory(self, memory_id):
        return self.rows.get(memory_id)


@pytest.fixture()
def stack(flags, stores, unique_ns, tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    import engine_adapters_switch as switch_module
    from engine_adapters_current import hashed_bag_embedder
    from ledger.port import ObjectRef
    from ledger.sqlite import SqliteLedger
    from memory_service import MemoryService

    rows_collection = unique_ns.qdrant_collection
    engine_collection = f"{rows_collection}_engine"
    flags(
        qdrant_collection=rows_collection,
        engine_routing_enabled=True,
        engine_routing_recall_enabled=True,
        engine_routing_qdrant_collection=engine_collection,
        engine_routing_dimensions=DIMS,
        standards_enabled=False,
        graph_leg_budget_enabled=False,
    )
    client = QdrantClient(url=stores.qdrant_url)
    client.create_collection(rows_collection, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    embed = hashed_bag_embedder(DIMS)
    view = Rows()
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))

    def put(memory_id, label, owner, *, visibility="private", project=None, workspace=None, tombstoned=False,
            register=True):
        text = f"{QUERY} {label}"
        metadata = {"owner_user_id": owner, "visibility": visibility, "category": "fact"}
        if project:
            metadata["project_id"] = project
        if workspace:
            metadata["workspace"] = workspace
        if tombstoned:
            metadata["dream_tombstoned"] = True
        client.upsert(rows_collection, points=[PointStruct(
            id=memory_id, vector=embed(text),
            payload={"data": text, "user_id": owner, "metadata": metadata, "created_at": "2026-01-01T00:00:00+00:00"},
        )])
        view.rows[memory_id] = SimpleNamespace(
            memory=text, owner_user_id=owner, visibility=visibility, project_id=project,
        )
        if register:
            ledger.put_object_ref(ObjectRef(TENANT, memory_id, 1, 1))

    class Service(MemoryService):
        def __init__(self):
            self._memory = SimpleNamespace(
                vector_store=SimpleNamespace(client=client),
                embedding_model=SimpleNamespace(embed=lambda text, memory_action=None: embed(text)),
                graph=None,
            )
            self._graphiti = None
            self._bridge = None
            self._filter_indexes_ok = False

        def _get_memory(self):
            return self._memory

    service = Service()
    put(A_P1, "alice project one", ALICE, project="p1")
    put(A_P2, "alice project two", ALICE, project="p2")
    put(A_NONE, "alice no project", ALICE)
    put(B_SHARED_P1, "bob shared", BOB, visibility="shared", project="p1")
    put(B_PRIVATE_P1, "bob private", BOB, project="p1")
    put(B_STALE, "bob projected shared", BOB, visibility="shared", project="p1")
    put(A_TOMB, "alice merged away", ALICE, project="p1")
    put(A_REF, "alice reference doc", ALICE, project="p1")

    from config import settings

    switch = switch_module.start(
        settings, ledger=ledger, service=view, tenant_for=lambda user: TENANT, embed=embed, qdrant_client=client,
    )
    assert switch is not None, switch_module.unavailable_reason()
    switch.catch_up()  # no window open: backfill the serving engine to canonical state
    assert switch.recall_ready() == (True, None)

    # State changes after projection: the engine's index is now stale for these three rows.
    put(B_STALE, "bob projected shared", BOB, visibility="private", project="p1", register=False)  # withdrawn from the shared pool
    put(A_TOMB, "alice merged away", ALICE, project="p1", tombstoned=True, register=False)
    put(A_REF, "alice reference doc", ALICE, project="p1", workspace="reference", register=False)

    calls = []
    real_recall = switch.recall
    switch.recall = lambda *a, **kw: calls.append((a, kw)) or real_recall(*a, **kw)
    try:
        yield SimpleNamespace(service=service, calls=calls, switch=switch)
    finally:
        switch_module.stop()
        ledger.close()
        for name in (rows_collection, engine_collection):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - best effort; the nsit prefix finds leftovers
                pass
        client.close()


def _ids(results):
    return {r.id for r in results}


def test_a_plain_project_request_is_served_by_the_engine_and_returns_only_readable_project_rows(stack):
    results = stack.service.search(query=QUERY, user_id=ALICE, project_id="p1", limit=10, include_shared=True)
    assert len(stack.calls) == 1, "the engine was not consulted"
    # alice's own project row and bob's shared project row: the engine's hits, re-checked against the rows
    assert _ids(results) == {A_P1, B_SHARED_P1}
    # never: another project's row, a row without a project, another user's private row, a row that was
    # withdrawn from the shared pool after projection, a tombstone, a reference-workspace row
    assert not _ids(results) & {A_P2, A_NONE, B_PRIVATE_P1, B_STALE, A_TOMB, A_REF}


def test_a_plain_request_without_a_project_covers_the_callers_rows_and_the_shared_pool(stack):
    results = stack.service.search(query=QUERY, user_id=ALICE, limit=10)
    assert len(stack.calls) == 1
    assert _ids(results) == {A_P1, A_P2, A_NONE, B_SHARED_P1}
    assert not _ids(results) & {B_PRIVATE_P1, B_STALE, A_TOMB, A_REF}


def test_another_user_never_sees_the_first_users_private_rows(stack):
    results = stack.service.search(query=QUERY, user_id=BOB, limit=10)
    assert len(stack.calls) == 1
    assert not _ids(results) & {A_P1, A_P2, A_NONE, A_TOMB, A_REF}
    assert {B_SHARED_P1, B_PRIVATE_P1, B_STALE} <= _ids(results)  # bob's own rows, whatever their visibility


def test_a_request_the_engine_cannot_apply_is_served_by_the_normal_path(stack):
    results = stack.service.search(
        query=QUERY, user_id=ALICE, project_id="p1", limit=10, categories=["fact"], vector_only=False,
    )
    assert stack.calls == [], "a filtered request must not be routed"
    assert {A_P1, B_SHARED_P1} <= _ids(results)  # the normal path's own pools answered it
    assert not _ids(results) & {B_PRIVATE_P1, B_STALE, A_TOMB, A_REF, A_P2}


def test_with_the_recall_flag_off_the_engine_is_never_consulted(stack, flags):
    flags(engine_routing_recall_enabled=False)
    results = stack.service.search(query=QUERY, user_id=ALICE, project_id="p1", limit=10)
    assert stack.calls == []
    assert {A_P1, B_SHARED_P1} <= _ids(results)
