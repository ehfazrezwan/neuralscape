"""The PostgreSQL usage store survives a dropped connection and never stalls the provider-call thread.

No server needed: a stand-in driver models a connection that a restart or a killed backend has made
unusable.  The real-server proof (``pg_terminate_backend``) is in
``integration_tests/test_usage_store_pg_integration.py``.
"""

from __future__ import annotations

import contextlib
import threading
import time
import types

import pytest

import usage_capture as uc
import usage_store
from tests.test_usage_store import TENANT, event


class OperationalError(Exception):
    pass


class InterfaceError(Exception):
    pass


class World:
    """Stand-in for the database and the driver."""

    def __init__(self):
        self.up = True
        self.opened = 0
        self.stored: list[str] = []
        self.set_configs: list[tuple] = []
        self.connect_attempts = 0

    def connect(self, dsn, **kwargs):
        self.connect_attempts += 1
        if not self.up:
            raise OperationalError("could not connect")
        self.opened += 1
        return Conn(self)

    def module(self):
        return types.SimpleNamespace(
            connect=self.connect, OperationalError=OperationalError, InterfaceError=InterfaceError
        )


class Conn:
    def __init__(self, world):
        self.world = world
        self.closed = False
        self.broken = False
        self.killed = False

    def execute(self, sql, params=None):
        if self.killed:
            self.broken = True
            raise OperationalError("terminating connection due to administrator command")
        if "set_config" in sql:
            self.world.set_configs.append(params)
            return types.SimpleNamespace(fetchone=lambda: ("",))
        if sql.startswith("INSERT INTO usage_events"):
            self.world.stored.append(params[1])
            return types.SimpleNamespace(rowcount=1)
        return types.SimpleNamespace(fetchone=lambda: ("on",), rowcount=1)

    @contextlib.contextmanager
    def transaction(self):
        yield self

    def close(self):
        self.closed = True


def make_store(world, monkeypatch, **kwargs):
    monkeypatch.setitem(__import__("sys").modules, "psycopg", world.module())
    store = usage_store.PostgresUsageStore.__new__(usage_store.PostgresUsageStore)
    usage_store._SqlUsageStore.__init__(store, time.time)
    store._psycopg = world.module()
    store._dsn = "dbname=x"
    store._connect_timeout = 10
    store._statement_timeout_ms = 30_000
    store._lock_timeout_ms = 10_000
    store._append_budget_s = kwargs.get("append_budget_s", 2.0)
    store._reconnect_cooldown_s = kwargs.get("reconnect_cooldown_s", 5.0)
    store._monotonic = kwargs.get("monotonic", time.monotonic)
    store._retry_after = 0.0
    store.reconnects = 0
    store._conn = store._open_connection(10)
    return store


def test_an_append_after_the_connection_was_killed_reconnects_and_succeeds(monkeypatch):
    world = World()
    store = make_store(world, monkeypatch)
    assert store.append(event("e1")) is True
    store._conn.killed = True  # pg_terminate_backend / a restart: the next statement fails
    assert store.append(event("e2")) is True
    assert world.opened == 2 and store.reconnects == 1
    assert world.stored == ["e1", "e2"]


def test_a_connection_the_driver_reports_closed_is_replaced_before_use(monkeypatch):
    world = World()
    store = make_store(world, monkeypatch)
    store._conn.closed = True
    assert store.append(event("e1")) is True
    assert world.opened == 2


def test_an_unreachable_database_fails_fast_during_the_cooldown(monkeypatch):
    world = World()
    now = {"t": 100.0}
    store = make_store(world, monkeypatch, monotonic=lambda: now["t"], reconnect_cooldown_s=5.0)
    world.up = False
    store._conn.killed = True
    with pytest.raises(OperationalError):
        store.append(event("e1"))  # the reconnect fails
    attempts = world.connect_attempts
    with pytest.raises(usage_store.UsageStoreError, match="backing off"):
        store.append(event("e2"))  # no new connect while backing off
    assert world.connect_attempts == attempts
    now["t"] += 6.0
    world.up = True
    assert store.append(event("e3")) is True  # the cooldown is over: reconnected
    assert world.stored == ["e3"]


def test_an_append_never_waits_longer_than_its_budget_for_the_connection(monkeypatch):
    world = World()
    store = make_store(world, monkeypatch, append_budget_s=0.2)
    held, release = threading.Event(), threading.Event()

    def hold():
        with store._lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    started = time.monotonic()
    try:
        with pytest.raises(usage_store.UsageStoreError, match="busy"):
            store.append(event("e1"))
    finally:
        release.set()
        holder.join(5)
    assert time.monotonic() - started < 2.0


def test_each_append_runs_under_its_own_short_statement_and_lock_timeout(monkeypatch):
    world = World()
    store = make_store(world, monkeypatch, append_budget_s=1.5)
    world.set_configs.clear()
    store.append(event("e1"))
    (statement, lock), = world.set_configs
    assert statement == lock
    assert 100 <= int(statement.removesuffix("ms")) <= 1500  # far below the session's 30 s


def test_the_recorder_does_not_chain_retries_past_its_budget():
    clock = {"t": 0.0}

    class Slow:
        calls = 0

        def append(self, event, **kw):
            Slow.calls += 1
            clock["t"] += 3.0  # one attempt spends the whole budget
            raise OperationalError("slow")

    recorder = uc.DurableUsageRecorder(
        Slow(), retries=2, sleep=lambda s: None, monotonic=lambda: clock["t"], record_budget_s=2.5,
    )
    assert recorder.record(event("e1")) is False
    assert Slow.calls == 1


def test_the_recorders_gap_memory_is_bounded_and_counts_what_it_drops():
    class Down:
        def append(self, event, **kw):
            raise OperationalError("down")

        def record_gap(self, *a, **kw):
            raise OperationalError("down")

    recorder = uc.DurableUsageRecorder(
        Down(), retries=0, sleep=lambda s: None, max_unrecorded_keys=3
    )
    for index in range(5):
        assert recorder.record(event(f"e{index}", task=f"task-{index}")) is False
    assert len(recorder.unrecorded) == 3
    assert recorder.gaps_overflowed == 2
    assert recorder.record(event("again", task="task-0")) is False  # a known task keeps counting
    assert recorder.unrecorded[(TENANT, "task-0")] == 2
    recorder.flush_gaps()  # the store is still down: nothing is lost beyond the cap, nothing grows
    assert len(recorder.unrecorded) == 3


def test_a_reconnect_is_bounded_by_the_remaining_budget_not_a_fixed_two_seconds(monkeypatch):
    """Before: the reconnect always waited up to two seconds, so a dropped connection plus a database that
    accepts connections slowly held the append (the provider-call thread) past its own budget."""
    world = World()
    store = make_store(world, monkeypatch, append_budget_s=0.3)
    release = threading.Event()
    real_connect = world.connect

    def slow_connect(dsn, **kwargs):
        release.wait(5)  # a database that does not answer
        return real_connect(dsn, **kwargs)

    world.connect = slow_connect
    monkeypatch.setitem(__import__("sys").modules, "psycopg", world.module())
    store._psycopg = world.module()
    store._conn.killed = True  # the next statement finds the connection dead; the retry must reconnect
    started = time.monotonic()
    try:
        with pytest.raises(usage_store.UsageStoreError, match="reconnect"):
            store.append(event("e1"))
        assert time.monotonic() - started < 1.0
    finally:
        release.set()


def test_a_reconnect_that_finishes_after_its_deadline_is_closed_not_adopted(monkeypatch):
    world = World()
    store = make_store(world, monkeypatch, append_budget_s=0.2)
    release = threading.Event()
    real_connect = world.connect
    opened = []

    def slow_connect(dsn, **kwargs):
        release.wait(5)
        conn = real_connect(dsn, **kwargs)
        opened.append(conn)
        return conn

    world.connect = slow_connect
    store._psycopg = world.module()
    store._conn.killed = True
    with pytest.raises(usage_store.UsageStoreError):
        store.append(event("e1"))
    release.set()
    deadline = time.monotonic() + 3
    while not (opened and opened[0].closed) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert opened and opened[0].closed and store._conn is None  # abandoned: closed, never installed


def test_failed_flush_cycles_do_not_grow_the_gap_map_past_its_key_bound():
    """Before: a flush claimed the whole map, so the emptied map admitted a fresh set of keys while the
    claimed ones were in flight; both sets came back when the flush failed, and every cycle grew the map."""

    class Down:
        recorder = None

        def append(self, event, **kw):
            raise OperationalError("down")

        def record_gap(self, *a, **kw):
            # a new key is recorded while the flush holds the claimed ones (the flush then fails)
            Down.recorder.record(event(f"during-{len(Down.recorder.unrecorded)}-{time.monotonic()}",
                                       task=f"fresh-{Down.seq.__next__()}"))
            raise OperationalError("down")

    import itertools

    Down.seq = itertools.count()
    recorder = uc.DurableUsageRecorder(Down(), retries=0, sleep=lambda s: None, max_unrecorded_keys=3)
    Down.recorder = recorder
    for index in range(3):
        recorder.record(event(f"seed-{index}", task=f"seed-{index}"))
    for _ in range(4):
        recorder.flush_gaps()
        assert len(recorder.unrecorded) <= 3
    assert recorder.gaps_overflowed > 0  # the new keys were counted as dropped, not silently admitted


def _blocking_gap_write(monkeypatch):
    """A write the server cancels only when its statement_timeout expires (a blocked row lock)."""

    class Cancelled(Exception):
        pass

    real = Conn.execute
    seen = {"timeouts": []}

    def execute(self, sql, params=None):
        if "set_config('statement_timeout'" in sql:
            self.stmt_ms = int(params[0].rstrip("ms"))
            seen["timeouts"].append(self.stmt_ms)
        if sql.startswith("INSERT INTO usage_gaps") and getattr(self.world, "block", False):
            time.sleep(self.stmt_ms / 1000.0)
            raise Cancelled("canceling statement due to statement timeout")
        return real(self, sql, params)

    monkeypatch.setattr(Conn, "execute", execute)
    return Cancelled, seen


def test_a_gap_write_after_a_slow_successful_reconnect_stays_inside_the_deadline(monkeypatch):
    """Before: the statement budget was computed before the reconnect (and floored at 100 ms), so a reconnect that
    took most of the budget was followed by a statement allowed to run for the whole original budget."""
    cancelled, seen = _blocking_gap_write(monkeypatch)
    world = World()
    store = make_store(world, monkeypatch, append_budget_s=0.5)
    real_connect = world.connect

    def slow(dsn, **kwargs):
        time.sleep(0.35)  # succeeds, but spends most of the budget
        return real_connect(dsn, **kwargs)

    world.connect = slow
    store._psycopg = world.module()
    world.block = True
    store._conn.killed = True
    started = time.monotonic()
    with pytest.raises(cancelled):
        store.record_gap(TENANT, "task-1", 1)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5 + 0.15, elapsed
    assert seen["timeouts"] and seen["timeouts"][-1] <= 160  # what was left after the reconnect, not 499 ms


def test_a_reconnect_that_spends_the_whole_budget_fails_before_any_statement_runs(monkeypatch):
    cancelled, seen = _blocking_gap_write(monkeypatch)
    world = World()
    clock = {"now": 100.0}
    store = make_store(world, monkeypatch, append_budget_s=0.5, monotonic=lambda: clock["now"])
    real_connect = world.connect

    def spend(dsn, **kwargs):
        clock["now"] += 0.6  # the connect itself used up the budget (on the store's clock)
        return real_connect(dsn, **kwargs)

    world.connect = spend
    store._psycopg = world.module()
    world.block = True
    store._conn.killed = True
    with pytest.raises(usage_store.UsageStoreError, match="budget"):
        store.record_gap(TENANT, "task-1", 1)
    assert seen["timeouts"] == [500]  # only the first try (on the killed connection); none after the spent reconnect
