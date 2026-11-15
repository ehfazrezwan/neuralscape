"""Fault injection (gap O-12): the write path survives a vector-store outage.

Needs ``--compose-project``: the harness pauses / stops the Qdrant container of
the throwaway campaign stack (never the production project: ``ctx.compose``
refuses it) and always restores it. Runs last (see conftest ``RUN_LAST``).

Documented behaviour: an accepted write (202) is retried by the worker while
the vector store is unreachable and completes once it is back; the memory is
then recallable. A write that ends ``failed`` after the outage is a finding.
"""

from __future__ import annotations

import os
import time

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, memory_ids_in, rows_with, list_rows

QDRANT = os.environ.get("NS_QDRANT_SERVICE", "qdrant")
OUTAGE_SECONDS = 20
RECOVERY_WAIT = 240


def enqueue(client: NSClient, marker: str, project_id: str) -> Written:
    w = Written(marker=marker, content=f"{marker} The lamplighter's ladder is kept behind the net loft.")
    w.task_id = client.store_raw(
        w.content, "tech_stack", scope="project", project_id=project_id, visibility="private"
    )
    return w


def check_write_survived(client: NSClient, w: Written, project_id: str, timeout: float = RECOVERY_WAIT) -> None:
    """The task completed and the memory is listed and recalled."""
    w.status = client.poll_task(w.task_id, timeout=timeout, not_found_grace=min(10.0, timeout))
    assert w.status.get("status") == "completed", f"write enqueued during the outage ended as {w.status}"
    ids = memory_ids_in(w.status)
    assert ids, f"completed without a memory: {w.status}"
    assert rows_with(list_rows(client, project_id=project_id), w.marker), "memory missing from the listing after recovery"
    results = client.search(w.content, project_id=project_id, limit=10)
    assert any(w.marker in r.get("memory", "") for r in results), "memory not recallable after recovery"


@pytest.fixture
def restore_qdrant(ctx):
    """Whatever a scenario did to Qdrant, leave it running."""
    yield
    ctx.compose("unpause", QDRANT, check=False)
    ctx.compose("start", QDRANT, check=False)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # wait for the container to be up again
        out = ctx.compose("ps", "--status", "running", "--services", check=False)
        if QDRANT in (out.stdout or "").split():
            return
        time.sleep(2)


@proves("fork.mem0.qdrant-cold-start", "write.remember-raw", "vec.qdrant-point-store")
@needs("compose")
def test_write_enqueued_while_qdrant_is_paused_completes_after_it_resumes(ctx, a, restore_qdrant):
    marker = f"zq-{ctx.run_id}-pause"
    ctx.compose("pause", QDRANT)
    try:
        w = enqueue(a, marker, ctx.project_id)  # the API only needs Redis to accept the write
        time.sleep(OUTAGE_SECONDS)
    finally:
        ctx.compose("unpause", QDRANT)
    check_write_survived(a, w, ctx.project_id)


@proves("fork.mem0.qdrant-cold-start", "write.remember-raw", "vec.qdrant-point-store")
@needs("compose")
def test_write_enqueued_while_qdrant_is_stopped_completes_after_it_restarts(ctx, a, restore_qdrant):
    marker = f"zq-{ctx.run_id}-stop"
    ctx.compose("stop", QDRANT)
    try:
        w = enqueue(a, marker, ctx.project_id)
        time.sleep(OUTAGE_SECONDS)
    finally:
        ctx.compose("start", QDRANT)
    check_write_survived(a, w, ctx.project_id)


@proves("fork.mem0.qdrant-cold-start", "write.remember-raw")
@negative_control
def test_survival_check_fails_for_a_write_that_never_happened(ctx, a):
    ghost = Written(marker=f"zq-{ctx.run_id}-ghost", content="never written", task_id="00000000-0000-0000-0000-000000000000")
    expect_check_fails(check_write_survived, a, ghost, ctx.project_id, 3)
