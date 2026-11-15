"""The queue-empty webhook with the flag ON (profile ``flags-on``).

WEBHOOK_QUEUE_EMPTY_URL points at the ``nsc-acc-sink`` sidecar. After a write drains the fast
queue the worker POSTs a small JSON event; the sink logs it and the scenario reads the log.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, profiles, proves
from nsaccept.helpers import write_raw
from nsaccept.sidecar import QUEUE_EMPTY_PATH

pytestmark = [profiles("flags-on"), needs("compose")]

WEBHOOK_WAIT = 120


def events_for(sink, user_id: str) -> list[dict]:
    return [e for e in sink.json_bodies(QUEUE_EMPTY_PATH) if e.get("user_id") == user_id]


def check_queue_empty_event(sink, user_id: str, before: int = 0) -> dict:
    """A NEW event for ``user_id``: more of them than the ``before`` count snapshotted ahead of the write."""
    events = events_for(sink, user_id)
    assert len(events) > before, (
        f"the sink saw no NEW queue.empty event for {user_id} ({len(events)} now, {before} before the write; "
        f"{len(sink.requests('POST', QUEUE_EMPTY_PATH))} POSTs in all)"
    )
    ev = events[-1]
    assert ev.get("event") == "queue.empty", f"event {ev.get('event')!r}"
    assert isinstance(ev.get("queue"), str) and ev["queue"], f"queue {ev.get('queue')!r}"
    assert ev.get("job_id"), "event carries no job_id"
    datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))  # raises (a non-assertion error) on a malformed stamp
    return ev


@pytest.fixture(scope="module")
def drained(ctx, a, sink):
    marker = f"zq-{ctx.run_id}-webhook"
    before = len(events_for(sink, ctx.user_a))  # earlier events (e.g. from other scenarios' writes) must not count
    w = write_raw(a, f"{marker} The signal flag on the harbour mast is blue on Tuesdays.", "preference", marker=marker, visibility="private")
    w.before = before
    return w


@proves("queue.queue-empty-webhook")
def test_draining_the_queue_posts_a_queue_empty_event_for_the_writer(ctx, sink, drained):
    drained.require()
    ev = eventually(lambda: check_queue_empty_event(sink, ctx.user_a, drained.before), timeout=WEBHOOK_WAIT, interval=3, what="queue.empty webhook")
    posts = sink.requests("POST", QUEUE_EMPTY_PATH)
    assert all(p["method"] == "POST" for p in posts) and ev["queue"], f"posts: {posts[:2]}"


@proves("queue.queue-empty-webhook")
@negative_control
def test_event_check_fails_for_a_user_who_never_wrote_and_for_no_new_event(ctx, sink, drained):
    drained.require()
    expect_check_fails(check_queue_empty_event, sink, "acc-nobody-wrote")
    # earlier events must not satisfy the check: with the snapshot taken AFTER the event arrived, nothing is new
    eventually(lambda: check_queue_empty_event(sink, ctx.user_a, drained.before), timeout=WEBHOOK_WAIT, interval=3, what="queue.empty webhook")
    expect_check_fails(check_queue_empty_event, sink, ctx.user_a, len(events_for(sink, ctx.user_a)))
