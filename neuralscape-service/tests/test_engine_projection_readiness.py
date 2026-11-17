"""Routed recall is refused while nothing keeps the routed engine current, and ``synced`` stays honest.

No production write calls ``project_apply`` / ``project_remove`` yet, so an engine backfilled once goes stale on the
next write while still reporting itself synced.  Until a producer is wired (``PROJECTION_PRODUCER_ACTIVE``) routed
recall is refused with its own reason and the ordinary path serves.  A backfill that ends with conflicts or fails
withdraws an earlier ``synced``; keys the ledger lists but the row store cannot read are counted in the report.
"""

from types import SimpleNamespace

import pytest

import engine_adapters_switch as switch_module
from config import settings
from engine_adapters_switch import EngineSwitch, LedgerCanonicalSource, recall_switch
from engine_ports import EngineRecord, OperationContext, ReferenceVectorStore
from engine_routing import Backend, CanonicalState, EngineRouter
from tests.test_engine_recall_routing import OWN, SHARED_OTHER, _normal_path_ran, service  # noqa: F401

TENANT = "t"


def _ctx():
    return OperationContext(tenant_id=TENANT, correlation_id="c", deadline_ms=10)


class Source:
    def __init__(self, *states, skipped=0):
        self._states, self.skipped = states, skipped

    def states(self):
        return iter(self._states)


class Ledger:
    """The ledger calls a canonical source and the projection use, over plain dicts."""

    def __init__(self, refs, tombstones=()):
        self.refs = dict(refs)  # (tenant, id) -> revision
        self.tombstones = dict(tombstones)

    def object_heads(self, after, size):
        keys = sorted(k for k in self.refs if after is None or k > tuple(after))
        return [(t, i, self.refs[(t, i)]) for t, i in keys[:size]]

    def tombstone_heads(self, after, size):
        keys = sorted(k for k in self.tombstones if after is None or k > tuple(after))
        return [(t, i, self.tombstones[(t, i)]) for t, i in keys[:size]]

    def get_tombstone(self, tenant, record_id):
        revision = self.tombstones.get((tenant, record_id))
        return None if revision is None else SimpleNamespace(content_revision=revision)

    def get_object_ref(self, tenant, record_id, revision):
        return object() if self.refs.get((tenant, record_id), 0) >= revision else None


class Rows:
    def __init__(self, **rows):
        self.rows = rows

    def get_memory(self, memory_id):
        text = self.rows.get(memory_id)
        return None if text is None else SimpleNamespace(
            memory=text, owner_user_id="alice", visibility="private", project_id=None, scope="global"
        )


def _switch(ledger, rows, *, recall=True):
    router = EngineRouter(Backend("e", ReferenceVectorStore()))
    return EngineSwitch(router, ledger=ledger, service=rows, tenant_for=lambda user: TENANT, recall_enabled=recall)


@pytest.fixture
def flags_on(monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", True)
    monkeypatch.setattr(settings, "engine_routing_recall_enabled", True)
    monkeypatch.setattr(settings, "standards_enabled", False)
    monkeypatch.setattr(switch_module, "_switch", None)
    monkeypatch.setattr(switch_module, "_refusal_logged", set())


# -- routed recall is refused while no producer keeps the engine current ------------------------------------------


def test_a_backfilled_engine_is_not_ready_for_recall_while_no_producer_is_wired():
    switch = _switch(Ledger({(TENANT, "m1"): 1}), Rows(m1="likes tea"))
    switch.catch_up()
    assert switch.router.active_synced is True  # backfilled and clean ...
    assert switch.recall_ready() == (False, "projection_producer_inactive")  # ... and still refused
    status = switch.status()
    assert status["recall_ready"] is False and status["recall_blocked_reason"] == "projection_producer_inactive"


def test_with_a_producer_the_old_readiness_rules_apply(monkeypatch):
    monkeypatch.setattr(switch_module, "PROJECTION_PRODUCER_ACTIVE", True)
    switch = _switch(Ledger({(TENANT, "m1"): 1}), Rows(m1="likes tea"))
    assert switch.recall_ready() == (False, "projection_not_synced")
    switch.catch_up()
    assert switch.recall_ready() == (True, None)
    assert _switch(Ledger({}), Rows(), recall=False).recall_ready() == (False, "recall_disabled")


def test_the_refusal_is_logged_once_and_names_the_missing_producer(flags_on, caplog):
    switch = _switch(Ledger({(TENANT, "m1"): 1}), Rows(m1="likes tea"))
    switch.catch_up()
    switch_module.configure(switch)
    with caplog.at_level("WARNING", logger="neuralscape.engine_switch"):
        assert recall_switch(settings) is None and recall_switch(settings) is None
    refusals = [r for r in caplog.records if "routed recall is refused" in r.getMessage()]
    assert len(refusals) == 1 and "projection_producer_inactive" in refusals[0].getMessage()
    assert "projection producer" in refusals[0].getMessage()


def test_search_serves_from_the_normal_path_while_no_producer_is_wired(flags_on, service):  # noqa: F811
    switch = _switch(Ledger({(TENANT, OWN): 1}), Rows(**{OWN: "alice own"}))
    switch.catch_up()
    calls = []
    real = switch.recall
    switch.recall = lambda *a, **kw: calls.append(a) or real(*a, **kw)
    switch_module.configure(switch)
    service.search(query="q", user_id="alice", project_id="p1", limit=5)
    assert calls == [] and _normal_path_ran(service)


def test_search_is_served_by_the_engine_once_a_producer_is_wired(flags_on, service, monkeypatch):  # noqa: F811
    monkeypatch.setattr(switch_module, "PROJECTION_PRODUCER_ACTIVE", True)
    switch = _switch(Ledger({(TENANT, OWN): 1}), Rows(**{OWN: "alice own"}))
    switch.catch_up()
    calls = []
    real = switch.recall
    switch.recall = lambda *a, **kw: calls.append(a) or real(*a, **kw)
    switch_module.configure(switch)
    service.search(query="q", user_id="alice", project_id="p1", limit=5)
    assert len(calls) == 1 and not _normal_path_ran(service)


# -- synced stays honest -----------------------------------------------------------------------------------------


def test_a_later_backfill_that_ends_with_a_conflict_withdraws_an_earlier_synced():
    store = ReferenceVectorStore()
    router = EngineRouter(Backend("e", store))
    clean = Source(CanonicalState(TENANT, "r", 1, False, "canonical text", ("owner:alice",)))
    router.backfill(clean)
    assert router.active_synced is True
    # The engine is later changed under the same revision with other text (a conflict for the next run).
    store.remove(_ctx(), "r", 1)
    store.apply(_ctx(), EngineRecord(record_id="r", revision=2, tenant_id=TENANT, text="other", labels=()))
    conflicting = Source(CanonicalState(TENANT, "r", 2, False, "canonical text", ("owner:alice",)))
    report = router.backfill(conflicting)
    assert report.conflicts and router.active_synced is False


def test_a_backfill_that_fails_part_way_withdraws_an_earlier_synced():
    router = EngineRouter(Backend("e", ReferenceVectorStore()))
    router.backfill(Source(CanonicalState(TENANT, "r", 1, False, "text", ())))
    assert router.active_synced is True

    class Broken(Source):
        def states(self):
            yield CanonicalState(TENANT, "r2", 1, False, "more", ())
            raise RuntimeError("ledger went away")

    with pytest.raises(RuntimeError):
        router.backfill(Broken())
    assert router.active_synced is False


def test_a_clean_backfill_after_a_conflicting_one_marks_the_engine_synced_again():
    store = ReferenceVectorStore()
    router = EngineRouter(Backend("e", store))
    store.apply(_ctx(), EngineRecord(record_id="r", revision=1, tenant_id=TENANT, text="engine text", labels=()))
    assert router.backfill(Source(CanonicalState(TENANT, "r", 1, False, "canonical", ()))).conflicts
    assert router.active_synced is False
    assert not router.backfill(Source(CanonicalState(TENANT, "r", 2, False, "canonical", ()))).conflicts
    assert router.active_synced is True


def test_keys_the_ledger_lists_but_the_row_store_cannot_read_are_counted_in_the_report():
    ledger = Ledger({(TENANT, "m1"): 1, (TENANT, "gone"): 1})
    switch = _switch(ledger, Rows(m1="likes tea"))
    report = switch.catch_up()["report"]
    assert report["copied"] == 1 and report["skipped"] == 1


def test_the_snapshot_path_counts_unreadable_keys_too():
    tables = {"object_refs": [{"tenant_id": TENANT, "record_id": "gone", "content_revision": 1}], "tombstones": []}
    ledger = SimpleNamespace(snapshot=lambda: {"tables": tables})
    source = LedgerCanonicalSource(ledger, Rows(), lambda row: ())
    assert list(source.states()) == [] and source.skipped == 1
    list(source.states())
    assert source.skipped == 1  # reset on every walk, not accumulated


def test_a_memory_the_ledger_never_registered_is_not_projected_and_cannot_be_counted():
    """Pinned limit: the canonical source is the ledger, so a row it does not list is invisible to a backfill."""

    switch = _switch(Ledger({(TENANT, "m1"): 1}), Rows(m1="registered", unregistered="never in the ledger"))
    report = switch.catch_up()["report"]
    assert (report["copied"], report["skipped"]) == (1, 0)
    assert switch.router.active_synced is True
    assert switch.router._active.adapter.get(_ctx(), "unregistered") is None
