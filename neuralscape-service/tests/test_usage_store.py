"""Durable usage store: shared cases run on SQLite here and PostgreSQL in the
integration suite (``tests/test_usage_store_pg_integration.py``)."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

import usage_capture as uc
from contracts_usage import UsageEvent
from contracts_usage_reconcile import reconcile_usage_events
from usage_store import (
    SqliteUsageStore,
    UsageEventConflict,
    UsageStoreError,
    open_usage_store,
)

TENANT = "tenant-1"


def attribution(**over):
    base = dict(
        producer="llm_extraction",
        bill_owner=None,
        bill_owner_missing_reason="dependency_unavailable",
        execution_location="provider_hosted",
        provider="google",
        provider_missing_reason=None,
        model="m-1",
        model_missing_reason=None,
        recipients=None,
        recipients_missing_reason="dependency_unavailable",
    )
    base.update(over)
    return base


def qty(value=None, reason=None):
    return {"value": value, "missing_reason": reason}


def usage(inp=10, out=3):
    return dict(
        measurement="actual",
        input_tokens=qty(inp),
        output_tokens=qty(out),
        cache_read_tokens=qty(None, "provider_did_not_report"),
        cache_write_tokens=qty(None, "not_applicable"),
        reasoning_tokens=qty(None, "provider_did_not_report"),
        cache_read_inclusion="included_in_input",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="additional_to_output",
    )


def event(event_id, *, attempt="a1", status="final", outcome="completed", tokens=(10, 3),
          predecessor=None, late=False, tenant=TENANT, task="task-1", intent=None, ledger="service"):
    payload = dict(
        schema_version="candidate-v1",
        event_id=event_id,
        tenant_id=tenant,
        task_id=task,
        session_id=None,
        intent_id=intent,
        attempt_id=attempt,
        ledger=ledger,
        operation="llm_extraction",
        predecessor_event_id=predecessor,
        status=status,
        attempt_outcome=outcome,
        late_after_cancellation=late,
        attribution=attribution(),
        usage=usage(*tokens) if status != "unavailable" else None,
        usage_missing_reason=(
            "not_yet_reported" if status == "unavailable" and outcome == "cancelled"
            else "provider_did_not_report" if status == "unavailable" else None
        ),
    )
    return UsageEvent.model_validate(payload)


def service_total(result):
    return next(l for l in result.ledgers if l.ledger == "service")


# ---- shared cases: each takes a store factory ``make()`` returning a fresh store -------

def case_idempotent_duplicate(make):
    store = make()
    e = event("e1")
    assert store.append(e) is True
    assert store.append(e) is False
    assert len(store.events(TENANT)) == 1


def case_conflicting_event_id(make):
    store = make()
    store.append(event("e1", tokens=(10, 3)))
    with pytest.raises(UsageEventConflict):
        store.append(event("e1", tokens=(99, 3)))
    (stored,) = store.events(TENANT)
    assert stored.usage.input_tokens.value == 10


def case_tenant_isolation(make):
    store = make()
    store.append(event("e1", tenant="t-a"))
    store.append(event("e1", tenant="t-b", tokens=(5, 1)))
    assert [e.tenant_id for e in store.events("t-a")] == ["t-a"]
    assert store.events("t-b")[0].usage.input_tokens.value == 5
    assert store.events("t-none") == ()


def case_correction_counts_once(make):
    store = make()
    store.append(event("e1", status="pending", outcome="in_progress", tokens=(0, 0)))
    store.append(event("e2", predecessor="e1", tokens=(10, 3)))
    store.append(event("e2", predecessor="e1", tokens=(10, 3)))  # duplicate delivery
    result = store.reconcile(TENANT)
    # The pending observation contributes nothing and the duplicate final is
    # counted once; unreported cache/reasoning categories keep the total unknown.
    assert service_total(result).known_token_subtotal == 13
    assert service_total(result).total_tokens is None
    assert len(result.streams) == 1
    assert result.streams[0].head_event_id == "e2"
    in_memory = reconcile_usage_events(
        [event("e1", status="pending", outcome="in_progress", tokens=(0, 0)),
         event("e2", predecessor="e1", tokens=(10, 3))]
    )
    assert result.model_dump() == in_memory.model_dump()


def case_cancelled_late_counted_once(make):
    store = make()
    store.append(event("c1", status="unavailable", outcome="cancelled"))
    late = event("c2", predecessor="c1", outcome="cancelled", late=True, tokens=(7, 2))
    store.append(late)
    store.append(late)
    result = store.reconcile(TENANT)
    (stream,) = result.streams
    assert stream.late_after_cancellation is True and stream.head_event_id == "c2"
    assert stream.known_token_subtotal == 9  # late usage counted once, not twice
    in_memory = reconcile_usage_events(
        [event("c1", status="unavailable", outcome="cancelled"), late]
    )
    assert result.model_dump() == in_memory.model_dump()


def case_unknown_is_never_zero(make):
    store = make()
    store.append(event("u1", status="unavailable"))
    result = store.reconcile(TENANT)
    (stream,) = result.streams
    assert stream.total_tokens is None and stream.usage is None
    ledger = service_total(result)
    assert ledger.total_tokens is None and ledger.incomplete_attempt_ids == ("a1",)
    (stored,) = store.events(TENANT)
    assert stored.usage is None and stored.usage_missing_reason == "provider_did_not_report"


def case_insertion_order_independent(make):
    events = [
        event("e1", status="pending", outcome="in_progress", tokens=(0, 0)),
        event("e2", predecessor="e1", tokens=(10, 3)),
        event("o1", attempt="a2", tokens=(4, 1)),
    ]
    forward, backward = make(), make()
    for e in events:
        forward.append(e)
    for e in reversed(events):
        backward.append(e)
    assert forward.reconcile(TENANT).model_dump() == backward.reconcile(TENANT).model_dump()


def case_reconcile_empty_is_none(make):
    assert make().reconcile(TENANT) is None


def case_task_filter(make):
    store = make()
    store.append(event("e1", task="t1"))
    store.append(event("e2", task="t2", attempt="a2"))
    assert [e.task_id for e in store.events(TENANT, "t2")] == ["t2"]
    assert len(store.reconcile(TENANT, "t1").streams) == 1


def case_ledger_attempt_correlation(make):
    store = make()
    store.append(event("e1", intent="intent-9"), ledger_attempt=2)
    store.append(event("e2", intent="intent-9", attempt="a2"))
    assert store.ledger_attempts(TENANT, "intent-9") == {"e1": 2, "e2": None}
    with pytest.raises(ValueError):
        store.append(event("e3"), ledger_attempt=-1)
    with pytest.raises(ValueError):
        store.append(event("e4"), ledger_attempt=True)


def case_invalid_event_not_persisted(make):
    store = make()
    bad = UsageEvent.model_construct(**dict(event("b1").__dict__, status="final", usage=None))
    with pytest.raises(Exception):
        store.append(bad)
    with pytest.raises(TypeError):
        store.append({"event_id": "x"})
    assert store.events(TENANT) == ()


def case_concurrent_appends_count_once(make):
    store = make()
    events = [event(f"e{i}", attempt=f"a{i}") for i in range(20)]
    results = []

    def work(chunk):
        for e in chunk:
            results.append(store.append(e))

    threads = [threading.Thread(target=work, args=(events,)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 20 and results.count(False) == 60
    assert len(store.events(TENANT)) == 20


SHARED_CASES = [
    case_idempotent_duplicate,
    case_conflicting_event_id,
    case_tenant_isolation,
    case_correction_counts_once,
    case_cancelled_late_counted_once,
    case_unknown_is_never_zero,
    case_insertion_order_independent,
    case_reconcile_empty_is_none,
    case_task_filter,
    case_ledger_attempt_correlation,
    case_invalid_event_not_persisted,
    case_concurrent_appends_count_once,
]


@pytest.fixture()
def make_sqlite(tmp_path):
    opened = []
    counter = iter(range(1000))

    def make():
        store = SqliteUsageStore(str(tmp_path / f"usage-{next(counter)}.sqlite3"))
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


@pytest.mark.parametrize("case", SHARED_CASES, ids=lambda c: c.__name__)
def test_shared_case_on_sqlite(case, make_sqlite):
    case(make_sqlite)


# ---- SQLite-specific ------------------------------------------------------------------

def test_rows_are_content_free_and_match_the_contract(tmp_path):
    path = str(tmp_path / "u.sqlite3")
    store = SqliteUsageStore(path)
    store.append(event("e1", intent="i1"), ledger_attempt=1)
    raw = sqlite3.connect(path)
    columns = [r[1] for r in raw.execute("PRAGMA table_info(usage_events)")]
    assert columns == [
        "seq", "tenant_id", "event_id", "task_id", "session_id", "intent_id", "attempt_id",
        "ledger", "operation", "predecessor_event_id", "status", "attempt_outcome",
        "late_after_cancellation", "ledger_attempt", "digest", "body", "recorded_at",
    ]
    (body,) = raw.execute("SELECT body FROM usage_events").fetchone()
    assert set(json.loads(body)) == set(UsageEvent.model_fields)
    assert not any(k in json.loads(body) for k in ("prompt", "text", "content", "response"))
    raw.close()
    store.close()


def test_persists_across_reopen_and_schema_version(tmp_path):
    path = str(tmp_path / "u.sqlite3")
    first = SqliteUsageStore(path)
    first.append(event("e1"))
    assert first.schema_version() == 2
    first.close()
    second = SqliteUsageStore(path)
    assert [e.event_id for e in second.events(TENANT)] == ["e1"]
    assert second.schema_version() == 2
    second.close()


def test_digest_tamper_is_detected(tmp_path):
    path = str(tmp_path / "u.sqlite3")
    store = SqliteUsageStore(path)
    store.append(event("e1"))
    raw = sqlite3.connect(path, isolation_level=None)
    (body,) = raw.execute("SELECT body FROM usage_events").fetchone()
    raw.execute("UPDATE usage_events SET body=?", (body.replace('"final"', '"pending"'),))
    raw.close()
    with pytest.raises(UsageStoreError):
        store.events(TENANT)
    store.close()


def test_newer_or_altered_schema_is_refused(tmp_path):
    path = str(tmp_path / "u.sqlite3")
    SqliteUsageStore(path).close()
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("INSERT INTO usage_schema_version VALUES (99, 'future', 'x', 0)")
    raw.close()
    with pytest.raises(UsageStoreError, match="newer"):
        SqliteUsageStore(path)
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("DELETE FROM usage_schema_version WHERE version=99")
    raw.execute("UPDATE usage_schema_version SET checksum='tampered' WHERE version=1")
    raw.close()
    with pytest.raises(UsageStoreError, match="checksum"):
        SqliteUsageStore(path)


def test_synchronous_normal_is_refused(tmp_path):
    with pytest.raises(ValueError):
        SqliteUsageStore(str(tmp_path / "u.sqlite3"), synchronous="NORMAL")


def test_wal_and_full_synchronous_asserted(tmp_path):
    store = SqliteUsageStore(str(tmp_path / "u.sqlite3"))
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    store.close()


def test_kill_minus_nine_keeps_every_acknowledged_event(tmp_path):
    path = str(tmp_path / "crash.sqlite3")
    script = textwrap.dedent(
        """
        import os, signal, sys
        sys.path.insert(0, sys.argv[2])
        from tests.test_usage_store import event
        from usage_store import SqliteUsageStore
        store = SqliteUsageStore(sys.argv[1])
        for i in range(25):
            store.append(event(f"k{i}", attempt=f"a{i}"))
            print(i, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", script, path, root],
        capture_output=True, text=True, cwd=root,
    )
    assert proc.returncode == -signal.SIGKILL
    acknowledged = len(proc.stdout.split())
    assert acknowledged == 25
    store = SqliteUsageStore(path)
    assert len(store.events(TENANT)) == acknowledged
    assert store.reconcile(TENANT) is not None
    store.close()


# ---- recorder + configuration ---------------------------------------------------------

def _response(prompt=5, out=2):
    return SimpleNamespace(
        text="ok",
        model_version="m-1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=out,
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
    )


def test_durable_recorder_end_to_end_reconciles(make_sqlite):
    store = make_sqlite()
    recorder = uc.DurableUsageRecorder(store)
    previous = uc.set_usage_recorder(recorder)
    try:
        ctx = uc.AttemptContext(tenant_id=TENANT, task_id="task-1", attempt_id="att", intent_id="i1", ledger_attempt=3)
        with uc.attempt_context(ctx):
            uc.record_generation_usage(_response(5, 2), operation="llm_extraction", model="m")
            uc.record_generation_usage(_response(7, 1), operation="llm_extraction", model="m")
    finally:
        uc.set_usage_recorder(previous)
    assert recorder.failures == 0
    result = store.reconcile(TENANT, "task-1")
    assert len(result.streams) == 2
    assert set(store.ledger_attempts(TENANT, "i1").values()) == {3}


def test_store_failure_is_counted_and_never_raised(make_sqlite):
    store = make_sqlite()
    store.close()
    recorder = uc.DurableUsageRecorder(store)
    previous = uc.set_usage_recorder(recorder)
    try:
        with uc.attempt_context(uc.AttemptContext(tenant_id=TENANT, task_id="t", attempt_id="a")):
            uc.record_generation_usage(_response(), operation="llm_extraction", model="m")
    finally:
        uc.set_usage_recorder(previous)
    assert recorder.failures == 1


def test_disabled_by_default_and_opens_sqlite_when_enabled(tmp_path):
    from config import Settings

    assert Settings().usage_store_enabled is False
    assert open_usage_store(Settings()) is None
    enabled = Settings(usage_store_enabled=True, usage_store_sqlite_path=str(tmp_path / "x.sqlite3"))
    store = open_usage_store(enabled)
    try:
        assert store.schema_version() == 2
    finally:
        store.close()
    with pytest.raises(ValueError):
        open_usage_store(Settings(usage_store_enabled=True, usage_store_backend="postgres"))
    with pytest.raises(ValueError):
        open_usage_store(Settings(usage_store_enabled=True, usage_store_backend="redis"))


def test_a_rollup_over_the_event_cap_is_refused_not_truncated(tmp_path, monkeypatch):
    import usage_store

    store = SqliteUsageStore(str(tmp_path / "cap.db"))
    for i in range(4):
        store.append(event(f"e{i}", attempt=f"a{i}"))
    assert len(store.events("tenant-1", "task-1", limit=4)) == 4
    with pytest.raises(usage_store.UsageRollupTooLarge):
        store.events("tenant-1", "task-1", limit=3)
    monkeypatch.setattr(usage_store, "MAX_ROLLUP_EVENTS", 3)
    with pytest.raises(usage_store.UsageRollupTooLarge):
        usage_store.task_rollup(store, "tenant-1", "task-1")
    monkeypatch.setattr(usage_store, "MAX_ROLLUP_EVENTS", 4)
    assert len(usage_store.task_rollup(store, "tenant-1", "task-1")["attempts"]) == 4
    store.close()
