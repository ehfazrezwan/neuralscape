"""Relabel: an engine filled before the scope label existed is brought to canonical labels by one backfill.

The same revision and text with other labels is an update in every vector adapter (not an idempotency
conflict); other text at the revision stays a conflict.  Backfill compares stored with computed labels,
writes only the mismatches, and a second run writes nothing.
"""

from types import SimpleNamespace

import pytest
from mem0.vector_stores.qdrant import Qdrant
from qdrant_client import QdrantClient

from contracts_errors import InternalErrorCode
from engine_adapters_current import QdrantVectorAdapter, hashed_bag_embedder
from engine_adapters_pgvector import PgVectorAdapter
from engine_adapters_switch import EngineSwitch, labels_for
from engine_ports import EngineRecord, OperationContext, PortFailure, ReferenceVectorStore, TenantFilter
from engine_routing import Backend, CanonicalState, EngineRouter
from tests.test_engine_adapters_pgvector import FakeConn

LEGACY = ("owner:alice", "vis:private")  # a global row projected before the scope label existed
GLOBAL = labels_for("alice", "private", None, "global")
PROJECT = labels_for("alice", "private", "p1", "project")


def ctx(tenant="t"):
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10)


def rec(rid, text, labels, revision=1):
    return EngineRecord(record_id=rid, revision=revision, tenant_id="t", text=text, labels=tuple(labels))


@pytest.fixture(autouse=True)
def offline_bm25(monkeypatch):
    monkeypatch.setattr(Qdrant, "_get_bm25_encoder", lambda self: None)


def _reference():
    return ReferenceVectorStore()


def _qdrant():
    store = Qdrant(collection_name="relabel", embedding_model_dims=64, client=QdrantClient(":memory:"))
    return QdrantVectorAdapter(store, hashed_bag_embedder(64))


@pytest.fixture(params=[_reference, _qdrant], ids=["reference", "qdrant"])
def store(request):
    return request.param()


def test_same_revision_and_text_with_other_labels_is_a_relabel_not_a_conflict(store):
    store.apply(ctx(), rec("m1", "a global fact", LEGACY))
    store.apply(ctx(), rec("m1", "a global fact", GLOBAL))
    held = store.get(ctx(), "m1")
    assert held.labels == GLOBAL and held.revision == 1 and held.text == "a global fact"
    hits = store.search(ctx(), "global fact", TenantFilter(tenant_id="t", labels_all=("scope:global",)), 5)
    assert [h.record_id for h in hits] == ["m1"]  # the label filter sees the new labels
    assert store.search(ctx(), "global fact", TenantFilter(tenant_id="t", labels_all=("vis:private",)), 5)


def test_other_text_at_the_same_revision_stays_a_conflict_and_changes_nothing(store):
    store.apply(ctx(), rec("m1", "a global fact", LEGACY))
    with pytest.raises(PortFailure) as raised:
        store.apply(ctx(), rec("m1", "another text", GLOBAL))
    assert raised.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT
    held = store.get(ctx(), "m1")
    assert held.text == "a global fact" and held.labels == LEGACY


def test_a_removed_record_at_the_same_revision_is_still_a_conflict(store):
    store.apply(ctx(), rec("m1", "a global fact", LEGACY))
    store.remove(ctx(), "m1", 1)
    with pytest.raises(PortFailure) as raised:
        store.apply(ctx(), rec("m1", "a global fact", GLOBAL))
    assert raised.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT


def test_a_relabel_in_the_qdrant_adapter_does_not_embed_again():
    adapter = _qdrant()
    calls = []
    real = adapter._embed
    adapter._embed = lambda text: calls.append(text) or real(text)
    adapter.apply(ctx(), rec("m1", "a global fact", LEGACY))
    adapter.apply(ctx(), rec("m1", "a global fact", GLOBAL))
    assert calls == ["a global fact"]  # the relabel changed the payload only


def test_a_relabel_in_the_pgvector_adapter_updates_the_labels_only():
    row = [(3, False, "a global fact", list(LEGACY))]
    conn = FakeConn({"FROM ns_engine.records": row})
    adapter = PgVectorAdapter(embed=lambda text: [0.1, 0.2, 0.3], dimensions=3, connect=lambda: conn)
    adapter.apply(ctx(), rec("m1", "a global fact", GLOBAL, revision=3))
    writes = [(sql, params) for sql, params in conn.statements if sql.startswith(("INSERT", "UPDATE"))]
    assert len(writes) == 1 and writes[0][0].startswith("UPDATE") and "labels = %s" in writes[0][0]
    assert "embedding" not in writes[0][0] and "AND NOT removed" in writes[0][0]
    assert writes[0][1] == (list(GLOBAL), "t", "m1", 3)
    with pytest.raises(PortFailure) as raised:
        adapter.apply(ctx(), rec("m1", "other text", GLOBAL, revision=3))
    assert raised.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT
    removed = FakeConn({"FROM ns_engine.records": [(3, True, "", [])]})
    gone = PgVectorAdapter(embed=lambda text: [0.1, 0.2, 0.3], dimensions=3, connect=lambda: removed)
    with pytest.raises(PortFailure) as raised:
        gone.apply(ctx(), rec("m1", "a global fact", GLOBAL, revision=3))
    assert raised.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT


class Source:
    def __init__(self, *states):
        self._states = states

    def states(self):
        return iter(self._states)


def _canonical():
    return Source(
        CanonicalState("t", "m1", 1, False, "a global fact", GLOBAL),
        CanonicalState("t", "m2", 1, False, "a project fact", PROJECT),
    )


def test_backfill_relabels_only_mismatches_and_a_second_run_changes_nothing(store):
    store.apply(ctx(), rec("m1", "a global fact", LEGACY))  # filled before the label existed
    store.apply(ctx(), rec("m2", "a project fact", PROJECT))  # already canonical
    writes = []
    real_apply = store.apply
    store.apply = lambda c, r: writes.append(r.record_id) or real_apply(c, r)
    router = EngineRouter(Backend("e", store))

    first = router.backfill(_canonical())
    assert (first.relabeled, first.unchanged, first.copied, first.conflicts) == (1, 1, 0, [])
    assert writes == ["m1"]  # only the mismatch was written
    assert router.active_synced is True
    assert store.get(ctx(), "m1").labels == GLOBAL

    writes.clear()
    second = router.backfill(_canonical())
    assert (second.relabeled, second.unchanged, second.copied, second.conflicts) == (0, 2, 0, [])
    assert writes == []  # idempotent: nothing was written


def test_backfill_of_a_pre_label_engine_reports_no_conflict_and_marks_it_synced():
    store = _reference()
    store.apply(ctx(), rec("m1", "a global fact", LEGACY))
    router = EngineRouter(Backend("e", store))
    report = router.backfill(_canonical())
    assert report.conflicts == [] and router.active_synced is True


def test_backfill_with_pgvector_writes_the_mismatch_once():
    state = {"labels": list(LEGACY)}

    class Conn(FakeConn):
        def execute(self, sql, params=None):
            if sql.startswith("UPDATE") and "labels = %s" in sql:
                state["labels"] = list(params[0])
            self.script = {"FROM ns_engine.records": [(1, False, "a global fact", state["labels"])]}
            return super().execute(sql, params)

    conn = Conn()
    adapter = PgVectorAdapter(embed=lambda text: [0.1, 0.2, 0.3], dimensions=3, connect=lambda: conn)
    router = EngineRouter(Backend("pg", adapter))
    source = Source(CanonicalState("t", "m1", 1, False, "a global fact", GLOBAL))
    assert router.backfill(source).relabeled == 1
    before = len(conn.statements)
    again = router.backfill(source)
    assert (again.relabeled, again.unchanged) == (0, 1)
    assert not [s for s, _ in conn.statements[before:] if s.startswith(("INSERT", "UPDATE"))]


def test_recall_finds_a_pre_label_global_row_only_after_the_backfill():
    store = _reference()
    store.apply(ctx(), rec("m1", "harbour cranes", LEGACY))
    store.apply(ctx(), rec("m2", "harbour pilots", PROJECT))
    router = EngineRouter(Backend("e", store))
    switch = EngineSwitch(router, ledger=None, service=None, tenant_for=lambda user: "t")

    def recall():
        return [h.record_id for h in switch.recall("alice", "harbour", 5, project_id="p1", visibility="private",
                                                    include_global=True)]

    assert recall() == ["m2"]  # before the backfill: the unlabelled global row is not returned
    router.backfill(Source(
        CanonicalState("t", "m1", 1, False, "harbour cranes", GLOBAL),
        CanonicalState("t", "m2", 1, False, "harbour pilots", PROJECT),
    ))
    assert sorted(recall()) == ["m1", "m2"]


def test_the_canonical_source_labels_by_the_rows_scope():
    switch = EngineSwitch(EngineRouter(Backend("e", _reference())), ledger=None, service=None,
                          tenant_for=lambda user: "t")
    row = SimpleNamespace(owner_user_id="alice", visibility="private", project_id="p9", scope="global")
    assert switch._row_labels(row) == ("owner:alice", "vis:private", "project:p9", "scope:global")
    row = SimpleNamespace(owner_user_id="alice", visibility="private", project_id="p9", scope="project")
    assert switch._row_labels(row) == ("owner:alice", "vis:private", "project:p9")


def test_a_projection_written_with_the_rows_scope_carries_the_global_label():
    store = _reference()
    ledger = SimpleNamespace(get_tombstone=lambda t, m: None, get_object_ref=lambda t, m, r: None)
    switch = EngineSwitch(EngineRouter(Backend("e", store)), ledger=ledger, service=None, tenant_for=lambda user: "t")
    switch.project_apply("t", "m1", text="likes tea", user_id="alice", visibility="private", project_id="p9",
                         scope="global", revision=1)
    assert store.get(ctx(), "m1").labels == ("owner:alice", "vis:private", "project:p9", "scope:global")


# -- the relabel is fenced against a newer revision landing between the check and the write ---------------------------


def _land_a_newer_revision_before_each_label_write(adapter, record):
    """Make ``record`` (a newer revision) land in the store right before the adapter's relabel write."""

    client = adapter._store.client
    real = client.set_payload
    landed = []

    def set_payload(*args, **kwargs):
        if not landed:
            landed.append(True)
            adapter.apply(ctx(), record)  # the newer revision arrives between the revision check and this write
        return real(*args, **kwargs)

    client.set_payload = set_payload
    return landed


def test_a_newer_revision_landing_between_the_check_and_the_relabel_keeps_its_own_labels():
    adapter = _qdrant()
    adapter.apply(ctx(), rec("m1", "likes coffee", LEGACY))
    newer = rec("m1", "likes coffee now", PROJECT, revision=2)
    landed = _land_a_newer_revision_before_each_label_write(adapter, newer)
    with pytest.raises(PortFailure) as raised:
        adapter.apply(ctx(), rec("m1", "likes coffee", GLOBAL))
    assert landed and raised.value.code is InternalErrorCode.STALE_REVISION
    held = adapter.get(ctx(), "m1")
    assert (held.revision, held.text, held.labels) == (2, "likes coffee now", tuple(PROJECT))


def test_a_removal_landing_between_the_check_and_the_relabel_is_not_overwritten_with_labels():
    adapter = _qdrant()
    adapter.apply(ctx(), rec("m1", "likes coffee", LEGACY))
    client = adapter._store.client
    real = client.set_payload

    def set_payload(*args, **kwargs):
        adapter.remove(ctx(), "m1", 1)
        return real(*args, **kwargs)

    client.set_payload = set_payload
    with pytest.raises(PortFailure) as raised:
        adapter.apply(ctx(), rec("m1", "likes coffee", GLOBAL))
    assert raised.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT
    assert adapter.get(ctx(), "m1") is None  # still removed


def test_backfill_reports_the_raced_record_as_superseded_and_does_not_lose_the_newer_labels():
    adapter = _qdrant()
    adapter.apply(ctx(), rec("m1", "likes coffee", LEGACY))
    _land_a_newer_revision_before_each_label_write(adapter, rec("m1", "likes coffee now", PROJECT, revision=2))
    router = EngineRouter(Backend("e", adapter))
    report = router.backfill(Source(CanonicalState("t", "m1", 1, False, "likes coffee", GLOBAL)))
    assert (report.superseded, report.relabeled, report.conflicts) == (1, 0, [])
    held = adapter.get(ctx(), "m1")
    assert (held.revision, held.labels) == (2, tuple(PROJECT))


def test_a_live_write_cannot_land_inside_a_backfills_copy_of_one_record():
    """Across the engine's check and write the router's write lock is held, so a live write waits for it."""

    import threading

    adapter = _qdrant()
    adapter.apply(ctx(), rec("m1", "likes coffee", LEGACY))
    router = EngineRouter(Backend("e", adapter))
    client = adapter._store.client
    real = client.set_payload
    finished_inside_the_gap = []

    def set_payload(*args, **kwargs):
        worker = threading.Thread(
            target=lambda: router.apply(router.route(), ctx(), rec("m1", "likes coffee now", PROJECT, revision=2))
        )
        worker.start()
        worker.join(timeout=0.4)
        finished_inside_the_gap.append(not worker.is_alive())
        result = real(*args, **kwargs)
        set_payload.worker = worker
        return result

    client.set_payload = set_payload
    report = router.backfill(Source(CanonicalState("t", "m1", 1, False, "likes coffee", GLOBAL)))
    set_payload.worker.join(timeout=5)
    assert finished_inside_the_gap == [False]  # the live write waited for the copy to finish
    assert report.relabeled == 1
    held = adapter.get(ctx(), "m1")
    assert (held.revision, held.labels) == (2, tuple(PROJECT))


def test_project_row_projects_the_current_row_with_its_scope_label():
    store = _reference()
    row = SimpleNamespace(memory="likes tea", owner_user_id="alice", visibility="private", project_id="p9", scope="global")
    service = SimpleNamespace(get_memory=lambda memory_id: row if memory_id == "m1" else None)
    ledger = SimpleNamespace(get_tombstone=lambda t, m: None, get_object_ref=lambda t, m, r: object() if r == 1 else None)
    switch = EngineSwitch(EngineRouter(Backend("e", store)), ledger=ledger, service=service, tenant_for=lambda user: "t")
    switch.project_row("t", "m1")
    held = store.get(ctx(), "m1")
    assert (held.text, held.revision) == ("likes tea", 1)
    assert held.labels == ("owner:alice", "vis:private", "project:p9", "scope:global")
    switch.project_row("t", "missing")  # no row: nothing is projected, nothing raises
    assert store.get(ctx(), "missing") is None


def test_project_apply_needs_the_scope_so_a_global_row_cannot_lose_its_label():
    switch = EngineSwitch(EngineRouter(Backend("e", _reference())), ledger=None, service=None, tenant_for=lambda u: "t")
    with pytest.raises(TypeError):
        switch.project_apply("t", "m1", text="x", user_id="alice", project_id="p9")
