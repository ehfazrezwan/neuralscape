"""Routed-engine recall in ``MemoryService.search`` on real stores (``ENGINE_ROUTING_RECALL_ENABLED=true``).

Real: a Qdrant server holds both the memory rows (the collection the service searches) and the routed engine's
projection collection; a SQLite ledger is the canonical source the engine is brought to canonical state from; the
switch is the one the service builds at start-up (``engine_adapters_switch.start``).  Stood in for: the embedder
(a deterministic token hash, no model is called).

Shown, with the flag on: a plain request is served by the engine and returns engine hits; a request the engine
cannot apply (a category filter) is served by the normal path and never consults the engine; and nothing returned
is another user's private row, another project's row, a consolidation tombstone or a reference-workspace row,
including rows whose state changed after they were projected.  With the flag off the engine is never consulted.

Also shown: a global-category row written inside a project carries both the project label and ``scope:global``
and is found by another project's recall; a global row projected before the scope label existed is relabelled by
the backfill and found afterwards; and, while no projection producer is wired (the state of this revision),
routed recall is refused with its own reason and the normal path serves.  The readiness logic past that gate is
shown with the producer flag set for the test.
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


A_P1, A_P2, A_NONE, B_SHARED_P1, B_PRIVATE_P1, B_STALE, A_TOMB, A_REF, A_GLOBAL_IN_P3, A_GLOBAL_LEGACY = (
    _id() for _ in range(10)
)


class Rows:
    """The service's row view for the engine's canonical source (what ``get_memory`` returns)."""

    def __init__(self):
        self.rows: dict[str, SimpleNamespace] = {}

    def get_memory(self, memory_id):
        return self.rows.get(memory_id)


@pytest.fixture()
def stack(flags, stores, unique_ns, tmp_path, monkeypatch):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    import engine_adapters_switch as switch_module
    from engine_adapters_current import hashed_bag_embedder
    from engine_ports import EngineRecord, OperationContext
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
            register=True, scope=None):
        text = f"{QUERY} {label}"
        metadata = {"owner_user_id": owner, "visibility": visibility, "category": "fact"}
        if project:
            metadata["project_id"] = project
        if workspace:
            metadata["workspace"] = workspace
        if tombstoned:
            metadata["dream_tombstoned"] = True
        if scope:
            metadata["scope"] = scope
        client.upsert(rows_collection, points=[PointStruct(
            id=memory_id, vector=embed(text),
            payload={"data": text, "user_id": owner, "metadata": metadata, "created_at": "2026-01-01T00:00:00+00:00"},
        )])
        view.rows[memory_id] = SimpleNamespace(
            memory=text, owner_user_id=owner, visibility=visibility, project_id=project, scope=scope,
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

    def engine_ctx():
        return OperationContext(tenant_id=TENANT, correlation_id="it", deadline_ms=10_000)

    service = Service()
    put(A_P1, "alice project one", ALICE, project="p1")
    put(A_P2, "alice project two", ALICE, project="p2")
    put(A_NONE, "alice no project", ALICE)
    put(B_SHARED_P1, "bob shared", BOB, visibility="shared", project="p1")
    put(B_PRIVATE_P1, "bob private", BOB, project="p1")
    put(B_STALE, "bob projected shared", BOB, visibility="shared", project="p1")
    put(A_TOMB, "alice merged away", ALICE, project="p1")
    put(A_REF, "alice reference doc", ALICE, project="p1")
    # A global-category fact written inside project p3: it keeps p3's project_id yet belongs to alice's global rows.
    put(A_GLOBAL_IN_P3, "alice global fact from project three", ALICE, project="p3", scope="global")
    # A global row projected before the scope label existed: its projection carries the owner and visibility labels only.
    put(A_GLOBAL_LEGACY, "alice older global fact", ALICE, scope="global")

    from config import settings

    switch = switch_module.start(
        settings, ledger=ledger, service=view, tenant_for=lambda user: TENANT, embed=embed, qdrant_client=client,
    )
    assert switch is not None, switch_module.unavailable_reason()
    switch.router._active.adapter.apply(  # the pre-label projection of the older global row
        engine_ctx(), EngineRecord(
            record_id=A_GLOBAL_LEGACY, revision=1, tenant_id=TENANT,
            text=f"{QUERY} alice older global fact", labels=(f"owner:{ALICE}", "vis:private"),
        ),
    )
    # No projection producer is wired at this revision: routed recall is refused with its own reason.
    assert switch.recall_ready() == (False, "projection_producer_inactive")
    monkeypatch.setattr(switch_module, "PROJECTION_PRODUCER_ACTIVE", True)  # the readiness logic past that gate
    assert switch.recall_ready() == (False, "projection_not_synced")
    backfill = switch.catch_up()  # no window open: backfill the serving engine to canonical state
    assert backfill["report"]["relabeled"] == 1, backfill  # the pre-label row was relabelled, not re-copied
    assert switch.recall_ready() == (True, None)

    # State changes after projection: the engine's index is now stale for these three rows.
    put(B_STALE, "bob projected shared", BOB, visibility="private", project="p1", register=False)  # withdrawn from the shared pool
    put(A_TOMB, "alice merged away", ALICE, project="p1", tombstoned=True, register=False)
    put(A_REF, "alice reference doc", ALICE, project="p1", workspace="reference", register=False)

    calls = []
    real_recall = switch.recall
    switch.recall = lambda *a, **kw: calls.append((a, kw)) or real_recall(*a, **kw)
    try:
        yield SimpleNamespace(service=service, calls=calls, switch=switch, backfill=backfill, monkeypatch=monkeypatch, engine_ctx=engine_ctx)
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
    # plus alice's global rows, which every project's recall includes (one was written inside project p3)
    assert _ids(results) == {A_P1, B_SHARED_P1, A_GLOBAL_IN_P3, A_GLOBAL_LEGACY}
    # never: another project's row, a row without a project, another user's private row, a row that was
    # withdrawn from the shared pool after projection, a tombstone, a reference-workspace row
    assert not _ids(results) & {A_P2, A_NONE, B_PRIVATE_P1, B_STALE, A_TOMB, A_REF}


def test_a_plain_request_without_a_project_covers_the_callers_rows_and_the_shared_pool(stack):
    results = stack.service.search(query=QUERY, user_id=ALICE, limit=10)
    assert len(stack.calls) == 1
    assert _ids(results) == {A_P1, A_P2, A_NONE, B_SHARED_P1, A_GLOBAL_IN_P3, A_GLOBAL_LEGACY}
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


def test_a_global_row_written_inside_a_project_is_found_by_another_projects_recall_and_by_its_own(stack):
    for project in ("p2", "p3"):
        stack.calls.clear()
        results = stack.service.search(query=QUERY, user_id=ALICE, project_id=project, limit=10)
        assert len(stack.calls) == 1, "the engine was not consulted"
        assert A_GLOBAL_IN_P3 in _ids(results), project
    held = stack.switch.router._active.adapter.get(stack.engine_ctx(), A_GLOBAL_IN_P3)
    assert "project:p3" in held.labels and "scope:global" in held.labels


def test_a_global_row_projected_before_the_scope_label_is_found_after_the_backfill_relabelled_it(stack):
    assert stack.backfill["report"]["relabeled"] == 1  # asserted where the backfill ran
    adapter = stack.switch.router._active.adapter
    # The relabel really reached the stored point on the real server (read back, not just counted).
    assert "scope:global" in adapter.get(stack.engine_ctx(), A_GLOBAL_LEGACY).labels
    # The fixture then withdrew one projected row from the shared pool (B_STALE: vis:shared -> vis:private)
    # on purpose, so a second backfill has exactly that one canonical label change to apply and nothing
    # else: the relabelled row is already canonical and nothing is copied again.
    again = stack.switch.catch_up()
    assert again["report"]["copied"] == 0 and again["report"]["relabeled"] == 1, again
    assert "vis:private" in adapter.get(stack.engine_ctx(), B_STALE).labels
    assert "scope:global" in adapter.get(stack.engine_ctx(), A_GLOBAL_LEGACY).labels
    third = stack.switch.catch_up()
    assert third["report"]["copied"] == 0 and third["report"]["relabeled"] == 0, third  # now it writes nothing
    results = stack.service.search(query=QUERY, user_id=ALICE, project_id="p1", limit=10)
    assert len(stack.calls) == 1 and A_GLOBAL_LEGACY in _ids(results)


def test_routed_recall_is_refused_while_no_projection_producer_is_wired_and_the_normal_path_serves(stack):
    import engine_adapters_switch as switch_module

    stack.monkeypatch.setattr(switch_module, "PROJECTION_PRODUCER_ACTIVE", False)  # the default
    assert stack.switch.recall_ready() == (False, "projection_producer_inactive")
    results = stack.service.search(query=QUERY, user_id=ALICE, project_id="p1", limit=10)
    assert stack.calls == [], "routed recall must be refused without a producer"
    assert {A_P1, B_SHARED_P1} <= _ids(results)  # the normal path answered
