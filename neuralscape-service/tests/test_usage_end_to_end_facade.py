"""Facade attempts reach the durable usage store with exact ledger correlation.

Covers what the in-memory-recorder facade tests cannot: the late-after-
cancellation event must chain onto the cancelled event so the stored stream
reconciles (it used to be a second root and made reconciliation reject the
whole task), pending-first events, and exact attempt ids derived from the bound
ledger claim.
"""

from __future__ import annotations

import threading
import time

import pytest

import usage_capture as uc
from contracts_usage_reconcile import reconcile_usage_events
from tests.test_inference_facade import (
    CTX,
    FakeProvider,
    TransientError,
    gen_request,
    make,
    proposed,
)
from usage_store import SqliteUsageStore, task_rollup

@pytest.fixture
def store(tmp_path):
    s = SqliteUsageStore(str(tmp_path / "usage.db"))
    yield s
    s.close()


@pytest.fixture
def recorder(store):
    return uc.DurableUsageRecorder(store)


def test_late_usage_after_the_deadline_reconciles_and_counts_once(store, recorder):
    gate = threading.Event()
    primary = FakeProvider((gate, proposed("LATE")))
    facade = make(primary, FakeProvider(proposed("fallback")), recorder=recorder)
    result = facade.generate(CTX, gen_request(deadline_ms=100))
    assert result.reason_code == "deadline_exceeded"

    gate.set()
    deadline = time.time() + 5
    while len(store.events("tenant-a", "task-1")) < 3 and time.time() < deadline:
        time.sleep(0.01)
    events = store.events("tenant-a", "task-1")
    pending, cancelled, late = events
    assert (pending.status, cancelled.attempt_outcome, late.late_after_cancellation) == (
        "pending", "cancelled", True)
    assert cancelled.predecessor_event_id == pending.event_id
    assert late.predecessor_event_id == cancelled.event_id  # not a second stream root

    rolled = reconcile_usage_events(events)  # used to raise multiple_stream_roots
    (stream,) = [s for s in rolled.streams if s.ledger == "service"]
    # Reasoning tokens were not reported, so 15 is a known subtotal, not a total.
    assert stream.late_after_cancellation is True and stream.known_token_subtotal == 15
    assert stream.total_tokens is None and stream.incomplete_categories == ("reasoning_tokens",)

    for event in events:  # replaying the stored events changes nothing
        assert store.append(event) is False
    assert task_rollup(store, "tenant-a", "task-1")["service"]["known_token_subtotal"] == 15


def test_retry_then_fallback_attempts_have_exact_distinct_ids(store, recorder):
    primary = FakeProvider(TransientError("503"), TransientError("503"))
    fallback = FakeProvider(proposed("ok"))
    facade = make(primary, fallback, recorder=recorder, max_retries=1)
    result = facade.generate(CTX, gen_request())
    assert result.proposals and len(primary.calls) == 2 and len(fallback.calls) == 1

    rollup = task_rollup(store, "tenant-a", "task-1")
    attempts = rollup["attempts"]
    assert [a["attempt_outcome"] for a in attempts] == ["failed", "failed", "completed"]
    assert len({a["attempt_id"] for a in attempts}) == 3
    assert [a["recipients"] for a in attempts] == [["r1"], ["r1"], ["r2"]]
    assert all(a["total_tokens"] is None for a in attempts[:2])  # failed: unknown, not zero
    assert attempts[2]["known_token_subtotal"] == 15 and attempts[2]["total_tokens"] is None
    assert rollup["service"]["total_tokens"] is None  # honest about the unknown attempts
