"""Generation-fenced routing, dual write, catch-up, flip and rollback (reference stores)."""

import threading

import pytest

from contracts_errors import InternalErrorCode
from engine_adapters_current import CurrentMemoryEngineAdapter
from engine_ports import (
    AdapterRegistry,
    EngineRecord,
    OperationContext,
    PortFailure,
    ReferenceVectorStore,
    RoutingGenerationFence,
    TenantFilter,
    run_memory_engine_conformance,
    run_vector_store_conformance,
)
from engine_routing import (
    Backend,
    CanonicalState,
    EngineRouter,
    InMemoryCursorStore,
    InMemoryJournal,
    RoutePhase,
    RoutingError,
)


def ctx(tenant="tenant-a"):
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10)


def rec(rid, revision=1, text="alpha", tenant="tenant-a", labels=()):
    return EngineRecord(record_id=rid, revision=revision, tenant_id=tenant, text=text, labels=tuple(labels))


def make():
    a, b = ReferenceVectorStore("store-a"), ReferenceVectorStore("store-b")
    return EngineRouter(Backend("a", a)), a, b


def state(adapter, ids, tenant="tenant-a"):
    out = {}
    for rid in ids:
        row = adapter.get(ctx(tenant), rid)
        out[rid] = None if row is None else (row.revision, row.text, row.labels)
    return out


class Canonical:
    def __init__(self, *states):
        self._states = list(states)

    def states(self):
        return iter(self._states)


def stale(call):
    with pytest.raises(PortFailure) as raised:
        call()
    assert raised.value.code is InternalErrorCode.STALE_REVISION


# ---- generation fence ----------------------------------------------------


def test_fence_is_monotonic_and_rejects_every_other_generation():
    fence = RoutingGenerationFence(1)
    assert fence.advance() == 2
    fence.check(2)
    stale(lambda: fence.check(1))
    stale(lambda: fence.check(3))
    for bad in (-1, True, "1"):
        with pytest.raises(ValueError):
            RoutingGenerationFence(bad)


def test_registry_routes_with_the_generation_and_rejects_stale_ones():
    registry = AdapterRegistry()
    first, second = ReferenceVectorStore("v1"), ReferenceVectorStore("v2")
    from contracts_engines import EngineAdapterKind

    ids = [registry.register(first, EngineAdapterKind.VECTOR_STORE), registry.register(second, EngineAdapterKind.VECTOR_STORE)]
    registry.activate(ids[0], ())
    routed = registry.route(EngineAdapterKind.VECTOR_STORE)
    assert routed.port is first and routed.generation == registry.routing_generation
    registry.check_generation(routed.generation)
    registry.activate(ids[1], ())
    stale(lambda: registry.check_generation(routed.generation))  # an old route is refused after the switch
    registry.deactivate(EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(PortFailure):
        registry.route(EngineAdapterKind.VECTOR_STORE)


def test_stale_token_write_is_rejected_after_a_flip_and_never_reaches_the_old_backend():
    router, a, b = make()
    old = router.route()
    router.apply(old, ctx(), rec("r1"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r1", 1, False, "alpha")))
    generation = router.flip()
    assert generation == 2 and router.active_label == "b"
    stale(lambda: router.apply(old, ctx(), rec("r2")))
    stale(lambda: router.remove(old, ctx(), "r1", 2))
    stale(lambda: router.get(old, ctx(), "r1"))
    stale(lambda: router.search(old, ctx(), "alpha", TenantFilter(tenant_id="tenant-a"), 3))
    assert a.get(ctx(), "r2") is None and b.get(ctx(), "r2") is None and a.get(ctx(), "r1") is not None
    router.apply(router.route(), ctx(), rec("r2"))  # a freshly routed write goes to the new backend only
    assert b.get(ctx(), "r2") is not None and a.get(ctx(), "r2") is None


# ---- dual write and catch-up ---------------------------------------------


def test_dual_write_reaches_both_and_catch_up_copies_prewindow_state_from_canonical():
    router, a, b = make()
    t = router.route()
    router.apply(t, ctx(), rec("old", 1, "before window"))
    router.apply(t, ctx(), rec("gone", 1, "will be removed"))
    router.remove(t, ctx(), "gone", 2)
    router.begin_dual_write(Backend("b", b))
    router.apply(t, ctx(), rec("live", 1, "during window"))
    assert b.get(ctx(), "live") is not None and b.get(ctx(), "old") is None
    assert router.lag() > 0
    with pytest.raises(RoutingError, match="lagging"):
        router.flip()
    report = router.catch_up(Canonical(
        CanonicalState("tenant-a", "old", 1, False, "before window"),
        CanonicalState("tenant-a", "gone", 2, True),
        CanonicalState("tenant-a", "live", 1, False, "during window"),
    ))
    assert report.copied == 1 and report.unchanged == 1 and report.tombstoned == 1 and not report.conflicts
    # 'live' reached the shadow through the dual write: it is already canonical, so nothing was written for it
    assert router.lag() == 0
    ids = ["old", "gone", "live"]
    assert state(a, ids) == state(b, ids)
    stale_on_b = lambda: b.apply(ctx(), rec("gone", 1, "resurrect"))
    stale(stale_on_b)  # the tombstone fence exists on the shadow too


def test_dirty_keys_are_repaired_from_canonical_state():
    router, a, b = make()
    t = router.route()
    router.apply(t, ctx(), rec("x", 1, "one"))
    router.apply(t, ctx(), rec("y", 1, "two"))
    router.remove(t, ctx(), "y", 2)
    router.begin_dual_write(Backend("b", b))

    real_apply = b.apply
    flaky = {"fail": True}

    def apply(c, r):
        if flaky["fail"] and r.record_id == "z":
            raise RuntimeError("shadow down")
        return real_apply(c, r)

    b.apply = apply
    router.apply(t, ctx(), rec("z", 1, "during window"))  # the caller still succeeds
    assert a.get(ctx(), "z") is not None and b.get(ctx(), "z") is None
    assert router.lag() >= 1
    flaky["fail"] = False
    report = router.catch_up(Canonical(
        CanonicalState("tenant-a", "x", 1, False, "one"),
        CanonicalState("tenant-a", "y", 2, True),
        CanonicalState("tenant-a", "z", 1, False, "during window"),
    ))
    assert report.repaired_dirty == 1 and router.lag() == 0
    assert state(a, ["x", "y", "z"]) == state(b, ["x", "y", "z"])
    stale(lambda: b.apply(ctx(), rec("y", 1, "back")))


def test_conflicting_same_revision_payload_is_reported_not_overwritten():
    router, a, b = make()
    router.apply(router.route(), ctx(), rec("r", 1, "canonical text"))
    router.begin_dual_write(Backend("b", b))
    b.apply(ctx(), rec("r", 1, "different text"))  # divergent shadow
    report = router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "canonical text")))
    assert ("tenant-a", "r", 1) in report.conflicts
    assert b.get(ctx(), "r").text == "different text"


def test_phase_errors_and_abandon():
    router, a, b = make()
    with pytest.raises(RoutingError):
        router.catch_up()
    with pytest.raises(RoutingError):
        router.flip()
    with pytest.raises(RoutingError):
        router.rollback()
    router.begin_dual_write(Backend("b", b))
    with pytest.raises(RoutingError):
        router.begin_dual_write(Backend("c", ReferenceVectorStore("c")))
    router.abandon_dual_write()
    assert router.phase is RoutePhase.STEADY and router.generation == 1
    with pytest.raises(RoutingError):
        router.begin_dual_write(Backend("same", a))


def test_flip_parity_check_can_refuse():
    router, a, b = make()
    router.begin_dual_write(Backend("b", b))
    router.catch_up()
    with pytest.raises(RoutingError, match="parity"):
        router.flip(parity_check=lambda old, new: False)
    assert router.active_label == "a" and router.generation == 1
    assert router.flip(parity_check=lambda old, new: True) == 2


# ---- rollback --------------------------------------------------------------


def test_rollback_replays_post_switch_writes_under_a_new_generation_and_keeps_fences():
    router, a, b = make()
    t = router.route()
    router.apply(t, ctx(), rec("keep", 1, "kept"))
    router.apply(t, ctx(), rec("doomed", 1, "text to erase"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "keep", 1, False, "kept"),
                              CanonicalState("tenant-a", "doomed", 1, False, "text to erase")))
    assert router.flip() == 2
    t2 = router.route()
    router.apply(t2, ctx(), rec("keep", 2, "kept v2"))        # update after the switch
    router.apply(t2, ctx(), rec("new", 1, "created after"))    # create after the switch
    router.remove(t2, ctx(), "doomed", 2)                      # erase after the switch
    assert a.get(ctx(), "keep").revision == 1 and a.get(ctx(), "new") is None  # old backend is behind
    canonical = Canonical(
        CanonicalState("tenant-a", "keep", 2, False, "kept v2"),
        CanonicalState("tenant-a", "new", 1, False, "created after"),
        CanonicalState("tenant-a", "doomed", 2, True),
    )
    assert router.rollback(canonical) == 3                     # a new number, never a return to 1
    assert router.active_label == "a" and router.previous_label == "b"
    ids = ["keep", "new", "doomed"]
    assert state(a, ids) == state(b, ids)
    assert a.get(ctx(), "keep").text == "kept v2" and a.get(ctx(), "doomed") is None
    stale(lambda: a.apply(ctx(), rec("doomed", 1, "older after removal")))  # the fence replayed too
    stale(lambda: router.apply(t2, ctx(), rec("late", 1)))                  # generation-2 writes are refused now
    router.apply(router.route(), ctx(), rec("late", 1))                     # and generation 3 works


def test_rollback_then_roll_forward_again():
    router, a, b = make()
    t = router.route()
    router.apply(t, ctx(), rec("r", 1))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "alpha")))
    router.flip()
    router.apply(router.route(), ctx(), rec("r", 2, "two"))
    router.rollback(Canonical(CanonicalState("tenant-a", "r", 2, False, "two")))
    router.apply(router.route(), ctx(), rec("r", 3, "three"))
    assert router.rollback(Canonical(CanonicalState("tenant-a", "r", 3, False, "three"))) == 4  # back to b
    assert b.get(ctx(), "r").revision == 3 and router.active_label == "b"


def test_hooks_and_cursors_are_used_and_hook_errors_never_break_routing():
    seen = []

    class Hooks:
        def on_phase_change(self, phase, generation, detail):
            seen.append((phase, generation))
            raise RuntimeError("observer bug")

    cursors = InMemoryCursorStore()
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a), hooks=Hooks(), cursors=cursors)
    router.apply(router.route(), ctx(), rec("r"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "alpha")))
    router.flip(); router.rollback()
    assert [p for p, _ in seen] == ["dual_write_started", "caught_up", "flipped", "rolled_back"]
    assert cursors.get("catchup@1") == 0  # the journal position the shadow covers: none before a flip


# ---- concurrency -----------------------------------------------------------


def test_writes_racing_the_flip_are_applied_once_or_refused_never_lost():
    for _ in range(10):
        router, a, b = make()
        t = router.route()
        router.apply(t, ctx(), rec("seed", 1))
        router.begin_dual_write(Backend("b", b))
        router.catch_up(Canonical(CanonicalState("tenant-a", "seed", 1, False, "alpha")))
        accepted, refused = [], []
        lock = threading.Lock()
        barrier = threading.Barrier(9)

        def writer(i):
            token = router.route()           # routed before the flip may happen
            barrier.wait()
            try:
                router.apply(token, ctx(), rec(f"w{i}", 1, f"text {i}"))
                with lock: accepted.append(i)
            except PortFailure as failure:
                assert failure.code is InternalErrorCode.STALE_REVISION
                with lock: refused.append(i)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        [th.start() for th in threads]
        barrier.wait()
        router.flip()
        [th.join() for th in threads]
        assert sorted(accepted + refused) == list(range(8))
        ids = [f"w{i}" for i in range(8)]
        for i in accepted:
            assert b.get(ctx(), f"w{i}") is not None      # every accepted write reached the new backend
        for i in refused:
            assert a.get(ctx(), f"w{i}") is None and b.get(ctx(), f"w{i}") is None
        router.rollback()
        assert state(a, ids + ["seed"]) == state(b, ids + ["seed"])


# ---- the routed port honors the vector contract in every phase -------------------


def _in_steady():
    router, a, b = make()
    return router.port()


def _in_dual_write():
    router, a, b = make()
    router.begin_dual_write(Backend("b", b))
    return router.port()


def _after_flip():
    router, a, b = make()
    router.begin_dual_write(Backend("b", b)); router.catch_up(); router.flip()
    return router.port()


def _after_rollback():
    router, a, b = make()
    router.begin_dual_write(Backend("b", b)); router.catch_up(); router.flip(); router.rollback()
    return router.port()


@pytest.mark.parametrize("factory", [_in_steady, _in_dual_write, _after_flip, _after_rollback])
def test_routed_port_passes_the_vector_conformance_suite_in_every_phase(factory):
    assert run_vector_store_conformance(factory) == ()


@pytest.mark.parametrize("factory", [_in_steady, _after_flip])
def test_composite_engine_with_a_routed_vector_half_passes_memory_engine_conformance(factory):
    assert run_memory_engine_conformance(lambda: CurrentMemoryEngineAdapter(factory())) == ()


def test_no_runtime_imports():
    import ast
    from pathlib import Path

    import engine_routing

    tree = ast.parse(Path(engine_routing.__file__).read_text())
    names = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    names |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not names & {"memory", "memory_service", "worker", "main", "config", "qdrant_client", "psycopg"}


# ---- review fixes: shadow writes off the write lock; content-free bounded journal ----


import time  # noqa: E402


class SlowShadow(ReferenceVectorStore):
    """A shadow backend whose writes take a fixed time (or hang until released)."""

    def __init__(self, label, delay=0.0, gate=None):
        super().__init__(label)
        self.delay, self.gate = delay, gate

    def apply(self, c, r):
        if self.gate is not None:
            self.gate.wait(10)
        time.sleep(self.delay)
        return super().apply(c, r)


def _timed_writers(router, count=3):
    token = router.route()
    finished = []
    started = time.monotonic()

    def run(i):
        router.apply(token, ctx(), rec(f"k{i}", 1, f"text {i}"))
        finished.append(time.monotonic() - started)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    [t.start() for t in threads]
    [t.join(15) for t in threads]
    return sorted(finished)


def test_a_slow_shadow_does_not_serialize_writes_to_the_live_backend():
    a, b = ReferenceVectorStore("a"), SlowShadow("b", delay=1.5)
    router = EngineRouter(Backend("a", a), shadow_timeout_s=0.2)
    router.begin_dual_write(Backend("b", b))
    finished = _timed_writers(router)
    assert len(finished) == 3 and finished[-1] < 1.0  # was 1.5, 3.0, 4.5 s serialized
    assert all(a.get(ctx(), f"k{i}") is not None for i in range(3))


def test_a_hung_shadow_marks_keys_dirty_and_blocks_the_flip_until_repaired():
    gate = threading.Event()
    a, b = ReferenceVectorStore("a"), SlowShadow("b", gate=gate)
    router = EngineRouter(Backend("a", a), shadow_timeout_s=0.1)
    router.begin_dual_write(Backend("b", b))
    started = time.monotonic()
    router.apply(router.route(), ctx(), rec("k", 1, "text"))  # the caller is not failed or stalled
    assert time.monotonic() - started < 1.0
    assert router.lag() >= 1
    with pytest.raises(RoutingError, match="lagging"):
        router.flip()
    gate.set()
    time.sleep(0.3)  # the late write lands; the key stays dirty until a canonical repair
    report = router.catch_up(Canonical(CanonicalState("tenant-a", "k", 1, False, "text")))
    assert report.repaired_dirty == 1 and router.lag() == 0
    assert router.flip() == 2


def test_shadow_writes_beyond_the_outstanding_bound_are_dirty_not_queued():
    gate = threading.Event()
    a, b = ReferenceVectorStore("a"), SlowShadow("b", gate=gate)
    router = EngineRouter(Backend("a", a), shadow_timeout_s=0.05, shadow_max_outstanding=2)
    router.begin_dual_write(Backend("b", b))
    started = time.monotonic()
    for i in range(6):
        router.apply(router.route(), ctx(), rec(f"k{i}", 1))
    assert time.monotonic() - started < 2.0
    assert router.lag() == 6
    gate.set()


def test_a_flip_cannot_overtake_a_shadow_write_that_is_still_in_flight():
    gate = threading.Event()
    a, b = ReferenceVectorStore("a"), SlowShadow("b", gate=gate)
    router = EngineRouter(Backend("a", a), shadow_timeout_s=5.0)
    router.begin_dual_write(Backend("b", b))
    worker = threading.Thread(target=lambda: router.apply(router.route(), ctx(), rec("k", 1)))
    worker.start()
    time.sleep(0.2)
    assert router.lag() >= 1
    with pytest.raises(RoutingError, match="lagging"):
        router.flip()
    gate.set()
    worker.join(5)
    assert router.lag() == 0 and router.flip() == 2


def test_steady_state_writes_do_not_grow_the_journal():
    journal = InMemoryJournal()
    router = EngineRouter(Backend("a", ReferenceVectorStore("a")), journal=journal)
    for i in range(300):
        router.apply(router.route(), ctx(), rec(f"r{i}", 1, f"body {i}"))
    assert journal.head() == 0 and journal.entries_after(0) == []


def test_journal_entries_never_carry_record_text():
    journal = InMemoryJournal()
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a), journal=journal)
    router.apply(router.route(), ctx(), rec("r", 1, "SECRET WORDS"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "SECRET WORDS")))
    router.flip()
    router.apply(router.route(), ctx(), rec("r", 2, "MORE SECRET WORDS"))
    router.remove(router.route(), ctx(), "r", 3)
    entries = journal.entries_after(0)
    assert [e.kind for e in entries] == ["remove"]  # coalesced to the newest entry for the record
    assert "SECRET" not in repr(entries) and not any(hasattr(e, "record") for e in entries)


def test_rollback_repairs_post_flip_writes_from_the_canonical_source():
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a))
    router.begin_dual_write(Backend("b", b))
    router.catch_up()
    router.flip()
    router.apply(router.route(), ctx(), rec("n", 1, "created after"))
    with pytest.raises(RoutingError, match="canonical"):
        router.rollback()  # content is not in the journal, so the canonical source is required
    router.rollback(Canonical(CanonicalState("tenant-a", "n", 1, False, "created after")))
    assert a.get(ctx(), "n").text == "created after"


def test_the_rollback_horizon_can_be_released_and_trims_the_journal():
    journal = InMemoryJournal()
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a), journal=journal)
    router.begin_dual_write(Backend("b", b)); router.catch_up(); router.flip()
    for i in range(5):
        router.apply(router.route(), ctx(), rec(f"p{i}", 1))
    assert journal.head() == 5
    router.release_rollback_target()
    assert journal.entries_after(0) == [] and router.previous_label is None
    router.apply(router.route(), ctx(), rec("after", 1))
    assert journal.entries_after(0) == []  # no rollback target: nothing is journaled again
    with pytest.raises(RoutingError):
        router.rollback()


def test_a_flip_after_unsynced_steady_writes_needs_a_canonical_catch_up():
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a))
    router.apply(router.route(), ctx(), rec("old", 1, "written before the window"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up()  # no canonical source: cannot populate the shadow
    with pytest.raises(RoutingError, match="lagging"):
        router.flip()
    router.catch_up(Canonical(CanonicalState("tenant-a", "old", 1, False, "written before the window")))
    assert router.lag() == 0 and router.flip() == 2 and b.get(ctx(), "old") is not None


# ---- erasure reaches the previous generation; release is explicit and recorded; the journal is bounded ----


def _flipped(**router_kwargs):
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a), **router_kwargs)
    router.apply(router.route(), ctx(), rec("r", 1, "ERASEME-TEXT"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "ERASEME-TEXT")))
    router.flip()
    return router, a, b


def _all_text(store):
    return " ".join(str(row) for row in store._rows.values())


def test_a_removal_after_a_flip_also_purges_the_previous_generation():
    router, a, b = _flipped()
    router.remove(router.route(), ctx(), "r", 2)
    assert "ERASEME-TEXT" not in _all_text(b)
    assert "ERASEME-TEXT" not in _all_text(a)  # the previous generation no longer holds the erased text
    assert router.previous_purge_pending == 0


def test_an_unreachable_previous_generation_is_visible_as_pending_purge_and_blocks_release():
    router, a, b = _flipped(shadow_timeout_s=0.2)
    real_remove = a.remove
    a.remove = lambda c, rid, rev: (_ for _ in ()).throw(RuntimeError("previous store down"))
    router.remove(router.route(), ctx(), "r", 2)  # the caller is not failed
    assert router.previous_purge_pending == 1 and "ERASEME-TEXT" in _all_text(a)
    with pytest.raises(RoutingError, match="unpurged"):
        router.release_rollback_target()
    a.remove = real_remove
    router.rollback(Canonical(CanonicalState("tenant-a", "r", 2, True)))  # rollback repairs it from canonical state
    assert "ERASEME-TEXT" not in _all_text(a)


def test_release_destroys_the_previous_generation_and_leaves_an_audit_record():
    destroyed = []
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a, destroy=lambda: destroyed.append("a")))
    router.apply(router.route(), ctx(), rec("r", 1, "ERASEME-TEXT"))
    router.begin_dual_write(Backend("b", b))
    router.catch_up(Canonical(CanonicalState("tenant-a", "r", 1, False, "ERASEME-TEXT")))
    router.flip()
    audit = router.release_rollback_target()
    assert destroyed == ["a"] and router.previous_label is None
    assert audit["destroyed"] is True and audit["label"] == "a" and router.release_log == [audit]
    with pytest.raises(RoutingError):
        router.rollback()


def test_a_failed_destroy_keeps_the_previous_generation_and_says_so():
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a, destroy=lambda: (_ for _ in ()).throw(RuntimeError("drop failed"))))
    router.begin_dual_write(Backend("b", b)); router.catch_up(); router.flip()
    with pytest.raises(RoutingError, match="could not be destroyed"):
        router.release_rollback_target()
    assert router.previous_label == "a" and router.release_log == []


def test_the_rollback_window_releases_and_destroys_automatically_and_records_it():
    now = {"t": 0.0}
    destroyed = []
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a, destroy=lambda: destroyed.append("a")), rollback_window_s=60.0, clock=lambda: now["t"])
    router.begin_dual_write(Backend("b", b)); router.catch_up(); router.flip()
    router.apply(router.route(), ctx(), rec("x", 1))  # inside the window: kept
    assert router.previous_label == "a" and destroyed == []
    now["t"] = 61.0
    router.apply(router.route(), ctx(), rec("y", 1))  # the first write after the window closes it
    assert router.previous_label is None and destroyed == ["a"] and router.release_log[0]["reason"] == "window_elapsed"


def test_the_post_flip_journal_is_bounded_by_distinct_records_not_by_writes():
    journal = InMemoryJournal()
    router, a, b = _flipped(journal=journal)
    for revision in range(2, 5002):
        router.apply(router.route(), ctx(), rec("r", revision, f"text {revision}"))
    for i in range(50):
        router.apply(router.route(), ctx(), rec(f"k{i}", 1))
        router.apply(router.route(), ctx(), rec(f"k{i}", 2))
    entries = journal.entries_after(0)
    assert len(entries) == 51  # one newest entry per record
    assert {e.record_id: e.revision for e in entries}["r"] == 5001 and all(e.seq > 0 for e in entries)
    seqs = [e.seq for e in entries]; assert seqs == sorted(seqs)  # still totally ordered


def test_coalescing_keeps_the_rollback_repair_correct():
    router, a, b = _flipped()
    router.apply(router.route(), ctx(), rec("r", 2, "two"))
    router.apply(router.route(), ctx(), rec("r", 3, "three"))
    router.rollback(Canonical(CanonicalState("tenant-a", "r", 3, False, "three")))
    assert a.get(ctx(), "r").revision == 3 and a.get(ctx(), "r").text == "three"


# ---- recall readiness: the serving engine must have been brought to canonical state ----


def test_a_new_router_is_not_synced_until_a_canonical_backfill():
    a = ReferenceVectorStore("a")
    router = EngineRouter(Backend("a", a))
    assert router.active_synced is False
    report = router.backfill(Canonical(CanonicalState("tenant-a", "old", 1, False, "written before the ledger")))
    assert report.copied == 1 and router.active_synced is True and a.get(ctx(), "old").text == "written before the ledger"


def test_a_backfill_with_a_conflict_does_not_mark_the_engine_synced():
    a = ReferenceVectorStore("a")
    a.apply(ctx(), rec("r", 1, "different text"))
    router = EngineRouter(Backend("a", a))
    report = router.backfill(Canonical(CanonicalState("tenant-a", "r", 1, False, "canonical text")))
    assert report.conflicts and router.active_synced is False


def test_the_flipped_engine_is_synced_only_after_a_canonical_catch_up_and_a_rollback_restores_the_old_flag():
    a, b = ReferenceVectorStore("a"), ReferenceVectorStore("b")
    router = EngineRouter(Backend("a", a))
    router.begin_dual_write(Backend("b", b)); router.catch_up(Canonical()); router.flip()
    assert router.active_synced is True
    router.rollback()
    assert router.active_synced is False  # the old generation was never backfilled
    unsynced = EngineRouter(Backend("a", ReferenceVectorStore("a2")))
    unsynced.begin_dual_write(Backend("b", ReferenceVectorStore("b2"))); unsynced.catch_up(); unsynced.flip()
    assert unsynced.active_synced is False  # no canonical source was used to populate it


def test_backfill_is_refused_while_a_window_is_open():
    router = EngineRouter(Backend("a", ReferenceVectorStore("a")))
    router.begin_dual_write(Backend("b", ReferenceVectorStore("b")))
    with pytest.raises(RoutingError):
        router.backfill(Canonical())


def test_catch_up_reports_the_keys_its_canonical_source_could_not_read():
    """Before: backfill copied ``source.skipped`` into the report and catch-up did not, so a dual-write
    catch-up reported ``skipped == 0`` for keys no projection was written for."""
    router, a, b = make()
    router.apply(router.route(), ctx(), rec("old", 1, "before window"))
    router.begin_dual_write(Backend("b", b))

    class Source(Canonical):
        skipped = 2

    report = router.catch_up(Source(CanonicalState("tenant-a", "old", 1, False, "before window")))
    assert report.skipped == 2
