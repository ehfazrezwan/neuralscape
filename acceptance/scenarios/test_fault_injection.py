"""Fault injection (gap O-12): what the stack guarantees when Qdrant is down, and
that the cold-start collection-creation race does not crash the services.

Needs ``--compose-project``: the harness pauses / stops containers of the
throwaway campaign stack (never the production project: ``ctx.compose`` refuses
it) and always restores them, failing loudly if it cannot. Runs last (see
``RUN_LAST`` in conftest).

Outage scenarios assert only what dev guarantees: while the vector store is
down the API still accepts a write (202, never a 5xx), the task reaches a
terminal state (completed, or failed with an error message) within a bound,
health does not claim everything is fine, and after recovery a fresh write
completes and is recallable. They do NOT assert that a write enqueued during
the outage is eventually stored: the worker has no retry for store errors.

The cold-start scenario deletes the Qdrant collections, then starts the API and
the three workers at the same moment so they race to create them (PR #453).
That exercises the race itself; it cannot force Qdrant's specific 500 response,
so recovery from that live 500 remains unproven.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from nsaccept import eventually, expect_check_fails, guard, needs, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, memory_ids_in, rows_with, list_rows
from nsaccept.runctx import container_restart_count, parse_ps_json

QDRANT = os.environ.get("NS_QDRANT_SERVICE", "qdrant")
APP_SERVICES = os.environ.get(
    "NS_APP_SERVICES", "neuralscape,neuralscape-worker,neuralscape-graph-worker,neuralscape-ingest-worker"
).split(",")
COLLECTION = os.environ.get("NS_QDRANT_COLLECTION", "neuralscape_memories")
COLLECTIONS = [COLLECTION, f"{COLLECTION}_entities"]  # memories + mem0's entity store
OUTAGE_SECONDS = 20
TERMINAL_WAIT = 240
READY_WAIT = 120
COLD_CYCLES = 3
COLD_SETTLE_SECONDS = 90


# ── helpers ──────────────────────────────────────────────────────


def enqueue(client: NSClient, marker: str, project_id: str) -> Written:
    w = Written(marker=marker, content=f"{marker} The lamplighter's ladder is kept behind the net loft.")
    resp = client.post(
        "/v1/memories/raw",
        json={"content": w.content, "category": "tech_stack", "scope": "project", "project_id": project_id, "visibility": "private"},
    )
    assert resp.status_code == 202, f"write during the outage -> {resp.status_code} (want 202): {resp.text[:200]}"
    w.task_id = resp.json()["task_id"]
    return w


def check_task_terminal(client: NSClient, w: Written, timeout: float = TERMINAL_WAIT) -> dict:
    """The task ended cleanly: completed, or failed WITH an error message."""
    w.status = client.poll_task(w.task_id, timeout=timeout, not_found_grace=min(10.0, timeout))
    state = w.status.get("status")
    assert state in ("completed", "failed"), f"task did not reach a terminal state in {timeout:.0f}s: {w.status}"
    if state == "failed":
        assert w.status.get("error"), f"task failed without an error detail: {w.status}"
    return w.status


def check_recallable(client: NSClient, w: Written, project_id: str) -> None:
    ids = memory_ids_in(w.status)
    assert ids, f"completed without a memory: {w.status}"
    assert rows_with(list_rows(client, project_id=project_id), w.marker), "memory missing from the listing"
    results = client.search(w.content, project_id=project_id, limit=10)
    assert any(w.marker in r.get("memory", "") for r in results), "memory not recallable"


def fresh_write_works(client: NSClient, marker: str, project_id: str) -> None:
    w = enqueue(client, marker, project_id)
    status = client.poll_task(w.task_id, timeout=TERMINAL_WAIT)
    assert status.get("status") == "completed", f"fresh write after recovery ended as {status}"
    w.status = status
    eventually(lambda: check_recallable(client, w, project_id), timeout=90, what="fresh write recallable")


def health_snapshot(base_url: str) -> tuple[int, dict]:
    try:
        r = httpx.get(base_url.rstrip("/") + "/health", timeout=20)
    except httpx.TimeoutException:
        raise AssertionError("GET /health did not answer within 20s while the vector store was down") from None
    return r.status_code, r.json()


def check_health_notices_outage(base_url: str) -> None:
    """Specifically the vector_store check must stop saying "ok"."""
    _, body = health_snapshot(base_url)
    vector = (body.get("checks") or {}).get("vector_store")
    assert vector not in (None, "ok"), f"/health checks.vector_store is {vector!r} while Qdrant is down"


def ps_rows(ctx, services: list[str]) -> list[dict]:
    out = ctx.compose("ps", "--all", "--format", "json", *services, check=False)
    return parse_ps_json(out.stdout or "")


def wait_ready(ctx, client: NSClient, what: str) -> None:
    """Qdrant + app containers running, /health answering 200, a search succeeding;
    otherwise fail loudly with the ``ps`` output."""
    deadline = time.monotonic() + READY_WAIT
    last = ""
    while time.monotonic() < deadline:
        try:
            running = {r.get("Service") for r in ps_rows(ctx, [QDRANT, *APP_SERVICES]) if r.get("State") == "running"}
            if {QDRANT, *APP_SERVICES} <= running:
                if httpx.get(ctx.base_url + "/health", timeout=10).status_code == 200:
                    client.search("readiness probe", limit=1)
                    return
            last = f"running: {sorted(x for x in running if x)}"
        except (httpx.HTTPError, AssertionError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    ps = ctx.compose("ps", "--all", check=False).stdout
    pytest.fail(f"stack not ready {READY_WAIT}s after {what}; last: {last}\n{ps}")


@pytest.fixture
def restore_stack(ctx, a):
    """Whatever a scenario did, leave Qdrant and the app services running and ready."""
    yield
    ctx.compose("unpause", QDRANT, check=False)
    ctx.compose("start", QDRANT, *APP_SERVICES, check=False)
    wait_ready(ctx, a, "restoring the stack")


# ── Qdrant outage ────────────────────────────────────────────────


def _outage_cycle(ctx, a, down, up, tag: str) -> None:
    marker = f"zq-{ctx.run_id}-{tag}"
    down()
    try:
        w = enqueue(a, marker, ctx.project_id)  # the API only needs Redis to accept the write
        time.sleep(OUTAGE_SECONDS)
    finally:
        up()
    check_task_terminal(a, w)
    wait_ready(ctx, a, f"{tag} recovery")
    fresh_write_works(a, f"{marker}-after", ctx.project_id)


@proves("write.remember-raw", "queue.task-manager")
@needs("compose")
def test_paused_qdrant_write_is_accepted_ends_cleanly_and_a_fresh_write_works(ctx, a, restore_stack):
    _outage_cycle(ctx, a, lambda: ctx.compose("pause", QDRANT), lambda: ctx.compose("unpause", QDRANT), "pause")


@proves("write.remember-raw", "queue.task-manager")
@needs("compose")
def test_stopped_qdrant_write_is_accepted_ends_cleanly_and_a_fresh_write_works(ctx, a, restore_stack):
    _outage_cycle(ctx, a, lambda: ctx.compose("stop", QDRANT), lambda: ctx.compose("start", QDRANT), "stop")


@proves("ops.health-endpoints")
@needs("compose")
def test_health_does_not_claim_everything_is_fine_while_qdrant_is_down(ctx, restore_stack):
    """Correct behaviour: the vector_store check reports the outage. EXPECTED RED on dev
    until lane sec PR sec-08 (bug/health-probe-and-request-ids, "/health probes Qdrant")
    lands: today /health only checks that mem0 is initialised and never contacts Qdrant."""
    ctx.compose("pause", QDRANT)
    try:
        time.sleep(5)
        check_health_notices_outage(ctx.base_url)
    finally:
        ctx.compose("unpause", QDRANT)


@proves("write.remember-raw", "queue.task-manager")
@negative_control
def test_terminal_check_fails_for_a_task_that_never_existed(ctx, a):
    ghost = Written(marker=f"zq-{ctx.run_id}-ghost", content="never written", task_id="00000000-0000-0000-0000-000000000000")
    expect_check_fails(check_task_terminal, a, ghost, 3)


@proves("ops.health-endpoints")
@negative_control
def test_health_outage_check_fails_while_everything_is_up(ctx):
    expect_check_fails(check_health_notices_outage, ctx.base_url)


# ── cold start: concurrent first creation of the collections ─────


def qdrant_http(ctx, method: str, path: str) -> str:
    """Call Qdrant's REST API from inside its own container (curl, else bash /dev/tcp)."""
    url = f"http://localhost:6333{path}"
    raw = (
        f"exec 3<>/dev/tcp/localhost/6333; printf '{method} {path} HTTP/1.1\\r\\nHost: localhost\\r\\n"
        f"Connection: close\\r\\n\\r\\n' >&3; cat <&3"
    )
    script = f"curl -sS -X {method} {url} 2>/dev/null || bash -c \"{raw}\""
    return ctx.compose("exec", "-T", QDRANT, "sh", "-c", script).stdout or ""


def check_stack_stable(rows: list[dict], restarts: dict[str, int], services: list[str]) -> None:
    """Every service running, never exited, never restarted."""
    by_service = {r.get("Service"): r for r in rows}
    for svc in services:
        row = by_service.get(svc)
        assert row is not None, f"{svc} missing from `compose ps`"
        assert row.get("State") == "running", f"{svc} is {row.get('State')!r} (exit code {row.get('ExitCode')})"
        assert not row.get("ExitCode"), f"{svc} exited with {row.get('ExitCode')}"
        assert restarts.get(svc, 0) == 0, f"{svc} restarted {restarts[svc]} time(s)"


@proves("fork.mem0.qdrant-cold-start", "write.remember-raw")
@needs("compose")
def test_api_and_workers_started_together_on_empty_collections_stay_up(ctx, a, restore_stack):
    """DESTRUCTIVE: wipes the vector store. Runs only with NS_ALLOW_DESTRUCTIVE=1 and a
    compose project that passes the disposable-prefix allowlist (guard.py).

    Stop the API and the three workers, delete both Qdrant collections, start all
    four at once (they race to create the collections), then require: all running,
    zero exits and restarts after 90 s, and a write that works and is recalled.
    Three cycles."""
    if os.environ.get("NS_ALLOW_DESTRUCTIVE") != "1" or not ctx.compose_project:
        pytest.skip("destructive: needs NS_ALLOW_DESTRUCTIVE=1 and a compose project")
    guard.check_compose_project(ctx.compose_project)  # refuses anything outside the allowlist
    for cycle in range(1, COLD_CYCLES + 1):
        ctx.compose("stop", *APP_SERVICES)
        for name in COLLECTIONS:
            qdrant_http(ctx, "DELETE", f"/collections/{name}")
        listing = qdrant_http(ctx, "GET", "/collections")
        assert not any(f'"{n}"' in listing for n in COLLECTIONS), f"cycle {cycle}: collections still exist: {listing[:300]}"
        ctx.compose("up", "-d", "--no-build", "--no-recreate", "--no-deps", *APP_SERVICES)
        marker = f"zq-{ctx.run_id}-cold{cycle}"
        # the first requests land while the processes are still initialising
        deadline = time.monotonic() + READY_WAIT
        w = None
        while w is None and time.monotonic() < deadline:
            try:
                w = enqueue(a, marker, ctx.project_id)
            except (httpx.HTTPError, AssertionError):
                time.sleep(2)
        assert w is not None, f"cycle {cycle}: the API never accepted a write after start"
        time.sleep(COLD_SETTLE_SECONDS)
        rows = ps_rows(ctx, APP_SERVICES)
        restarts = {r["Service"]: container_restart_count(r["ID"]) for r in rows if r.get("ID")}
        check_stack_stable(rows, restarts, APP_SERVICES)
        wait_ready(ctx, a, f"cold-start cycle {cycle}")
        w.status = a.poll_task(w.task_id, timeout=TERMINAL_WAIT)
        assert w.status.get("status") == "completed", f"cycle {cycle}: write during start-up ended as {w.status}"
        eventually(lambda: check_recallable(a, w, ctx.project_id), timeout=90, what=f"cycle {cycle} write recallable")


@proves("fork.mem0.qdrant-cold-start")
@negative_control
def test_stability_check_fails_for_a_restarting_or_exited_service():
    ok = [{"Service": "api", "State": "running", "ExitCode": 0}]
    check_stack_stable(ok, {"api": 0}, ["api"])
    expect_check_fails(check_stack_stable, ok, {"api": 1}, ["api"])
    expect_check_fails(check_stack_stable, [{"Service": "api", "State": "exited", "ExitCode": 1}], {}, ["api"])
    expect_check_fails(check_stack_stable, [], {}, ["api"])
