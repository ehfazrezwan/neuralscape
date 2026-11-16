"""Fault injection: the autoheal sidecar restarts a wedged worker (atlas ``ops.autoheal``).

Docker restarts a container only when its process exits; a worker whose event loop is alive but
which has stopped beating is "wedged", and only the ``autoheal`` sidecar acts on that. The product
has a reversible test hook for it: while the file named by ``NS_TEST_STALL_HEARTBEAT_FILE`` exists
the worker stops touching its liveness heartbeat, so its compose healthcheck (heartbeat older than
240 s, checked every 30 s, 3 failures) turns ``unhealthy`` and the sidecar (15 s poll, 10 s stop
timeout) restarts it.

The scenario creates that file in the FAST worker, waits for ``unhealthy`` and then for the
container's ``StartedAt`` to change (autoheal restarts through Docker's restart API, so
``RestartCount`` does NOT move; a crash-restart would move it), removes the file, waits for
``healthy`` again and proves a fresh REST write is processed. It is destructive in the sense that
it wedges a worker for about ten minutes: it needs ``NS_ALLOW_DESTRUCTIVE=1``, a disposable compose
project, the ``autoheal`` profile (``profiles/autoheal.env``) and a stack that runs the ``autoheal``
service. A missing precondition FAILS, with the reason (this profile exists only for this scenario).

The stall file is under ``/dev/shm``, a per-container tmpfs that a container restart re-creates
empty; the scenario does not rely on that: it deletes the file right after it sees the restart, and
again in a ``finally``. If autoheal never acts, the worker is restored with ``compose restart`` and
the scenario fails loudly.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from nsaccept import eventually, expect_check_fails, guard, needs, negative_control, precondition, profiles, proves
from nsaccept import restore, stack
from nsaccept.helpers import list_rows, rows_with, write_raw
from nsaccept.runctx import container_inspect

AUTOHEAL = os.environ.get("NS_AUTOHEAL_SERVICE", "autoheal")
WORKER = os.environ.get("NS_FAST_WORKER_SERVICE", "neuralscape-worker")
STALL_PATH = restore.STALL_PATH
LABEL_PREFIX = "com.neuralscape.autoheal."

POLL_SECONDS = 2.0
RESTART_WAIT = 540.0  # stall -> 240 s stale heartbeat -> up to 3 failed checks (30 s) -> 15 s autoheal poll -> 10 s stop
HEALTHY_WAIT = 240.0  # restart -> first healthcheck (30 s) passes; the check only reads one file's mtime
STARTING_WAIT = 180.0
TERMINAL_WAIT = 240.0


# ── pure checks (no stack; the negative control runs the same code on canned data) ──


def label_key(compose_project: str) -> str:
    """The project-scoped label key the stack's workers carry and the sidecar filters on."""
    return LABEL_PREFIX + compose_project.strip().lower()


def check_autoheal_wiring(autoheal_row: dict | None, worker: dict, compose_project: str) -> None:
    """Precondition of the scenario: the sidecar runs and the worker carries ITS project's label with value ``true``."""
    assert autoheal_row is not None, f"PRECONDITION: no `{AUTOHEAL}` service in this compose project (the autoheal profile needs the sidecar running)"
    assert autoheal_row.get("State") == "running", f"PRECONDITION: `{AUTOHEAL}` is {autoheal_row.get('State')!r}, not running"
    key = label_key(compose_project)
    labels = worker.get("labels") or {}
    have = {k: v for k, v in labels.items() if k.startswith(LABEL_PREFIX.rstrip("."))}
    assert labels.get(key) == "true", (
        f"PRECONDITION: the {WORKER} container does not carry {key}=true (autoheal labels it has: {have or 'none'}); "
        "the sidecar would never act on it"
    )


def check_autoheal_recovery(before: dict, samples: list[dict]) -> None:
    """``before`` = the worker inspected before the stall; ``samples`` = every inspection from the stall until it was
    seen healthy again, in time order. Wedged -> unhealthy -> restarted exactly once by autoheal -> healthy."""
    assert before.get("health") == "healthy", f"worker was {before.get('health')!r} before the stall, not healthy"
    assert samples, "no inspections recorded"
    unhealthy = [i for i, s in enumerate(samples) if s.get("health") == "unhealthy"]
    assert unhealthy, "the stalled worker never became unhealthy (the heartbeat stall had no effect)"
    restarted = [i for i, s in enumerate(samples) if s.get("started_at") != before.get("started_at")]
    assert restarted, "autoheal did not restart the unhealthy worker: StartedAt never changed"
    assert restarted[0] > unhealthy[0], "the worker restarted without first being seen unhealthy (not an autoheal restart)"
    assert len({s.get("started_at") for s in samples[restarted[0]:]}) == 1, "the worker was restarted more than once (restart churn)"
    last = samples[-1]
    assert last.get("restart_count") == before.get("restart_count"), (
        f"RestartCount moved {before.get('restart_count')} -> {last.get('restart_count')}: the container crashed; "
        "autoheal restarts through Docker's restart API and leaves it unchanged"
    )
    assert last.get("status") == "running", f"worker is {last.get('status')!r} after the restart"
    assert last.get("health") == "healthy", f"worker is {last.get('health')!r} after the restart, not healthy again"


# ── driving the stall ────────────────────────────────────────────


def remove_stall(ctx, worker: str, *, wait: float = 60.0, sleep=time.sleep, clock=time.monotonic) -> bool:
    """``rm -f`` the stall file in the worker; retried while the container is not up (a restart is under way)."""
    deadline = clock() + wait
    while True:
        proc = ctx.compose("exec", "-T", worker, "rm", "-f", STALL_PATH, check=False)
        if proc.returncode == 0:
            return True
        if clock() >= deadline:
            return False
        sleep(POLL_SECONDS)


def drive_autoheal_cycle(
    ctx, worker: str, cid: str, before: dict, *, inspect=container_inspect, sleep=time.sleep, clock=time.monotonic,
    restart_wait: float = RESTART_WAIT, healthy_wait: float = HEALTHY_WAIT,
) -> list[dict]:
    """Stall the worker, watch autoheal restart it, release the stall, wait until it is healthy, and check the
    whole sequence. Always removes the stall file; if the worker was not healed by autoheal it is restarted here
    (and the failure still propagates)."""
    samples: list[dict] = []
    healed = False
    released = False
    try:
        ctx.compose("exec", "-T", worker, "touch", STALL_PATH)
        deadline = clock() + restart_wait
        while True:
            s = inspect(cid)
            samples.append(s)
            if s["started_at"] != before["started_at"]:
                break
            if clock() >= deadline:
                logs = ctx.compose("logs", "--tail", "30", AUTOHEAL, check=False).stdout or ""
                raise AssertionError(
                    f"autoheal did not restart {worker} within {restart_wait:.0f}s of the stall "
                    f"(last health {s['health']!r}, StartedAt unchanged). autoheal log tail:\n{logs[-1500:]}"
                )
            sleep(POLL_SECONDS)
        released = remove_stall(ctx, worker, sleep=sleep, clock=clock)  # the restarted tmpfs is already clean; do not rely on it
        deadline = clock() + healthy_wait
        while samples[-1]["health"] != "healthy" and clock() < deadline:
            sleep(POLL_SECONDS)
            samples.append(inspect(cid))
        check_autoheal_recovery(before, samples)
        healed = True
        return samples
    finally:
        problems: list[str] = []
        propagating = sys.exc_info()[0] is not None
        if not released:
            try:
                if not remove_stall(ctx, worker, sleep=sleep, clock=clock):
                    problems.append(f"could not remove {STALL_PATH} from {worker}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"removing {STALL_PATH} failed: {exc!r}")
        if not healed:
            try:
                ctx.compose("restart", worker)
                stack.wait_services(ctx, [worker], "restoring the worker after a failed autoheal scenario")
            except BaseException as exc:  # noqa: BLE001 - pytest.fail raises a BaseException
                problems.append(f"restoring {worker} failed: {exc!r}")
        if problems:
            msg = "; ".join(problems)
            if propagating:
                print(f"WARNING: autoheal scenario cleanup: {msg}")
            else:
                pytest.fail(f"autoheal scenario cleanup: {msg}")


def wait_until_healthy(cid: str, *, inspect=container_inspect) -> dict:
    def healthy() -> dict:
        s = inspect(cid)
        assert s["status"] == "running" and s["health"] == "healthy", f"worker is {s['status']!r}/{s['health']!r}"
        return s

    return eventually(healthy, timeout=STARTING_WAIT, interval=3.0, what="worker healthy before the stall")


# ── the scenario ─────────────────────────────────────────────────


@proves("ops.autoheal")
@profiles("autoheal")
@needs("compose")
def test_autoheal_restarts_a_wedged_worker_and_it_takes_work_again(ctx, a):
    """DESTRUCTIVE (wedges the fast worker for ~10 min): NS_ALLOW_DESTRUCTIVE=1, a disposable compose project,
    ``--profile autoheal`` and the ``autoheal`` service running. Missing preconditions fail, never skip."""
    assert os.environ.get("NS_ALLOW_DESTRUCTIVE") == "1", "PRECONDITION: the autoheal profile needs NS_ALLOW_DESTRUCTIVE=1 (it wedges a worker on purpose)"
    assert ctx.compose_project, "PRECONDITION: needs --compose-project"
    guard.check_compose_project(ctx.compose_project)
    assert ctx.profile_flags.get("NS_TEST_STALL_HEARTBEAT_FILE") == STALL_PATH, (
        f"PRECONDITION: the profile must set NS_TEST_STALL_HEARTBEAT_FILE={STALL_PATH} on the workers"
    )

    rows = {r.get("Service"): r for r in stack.ps_rows(ctx, [AUTOHEAL, WORKER])}
    worker_row = rows.get(WORKER)
    assert worker_row and worker_row.get("ID") and worker_row.get("State") == "running", f"PRECONDITION: {WORKER} is not running: {worker_row}"
    cid = worker_row["ID"]
    check_autoheal_wiring(rows.get(AUTOHEAL), container_inspect(cid), ctx.compose_project)
    before = wait_until_healthy(cid)

    drive_autoheal_cycle(ctx, WORKER, cid, before)

    marker = f"zq-{ctx.run_id}-autoheal"
    w = write_raw(
        a, f"{marker} The lamplighter's ladder is kept behind the net loft.", "tech_stack", marker=marker,
        timeout=TERMINAL_WAIT, scope="project", project_id=ctx.project_id,
    ).require()
    eventually(
        lambda: _listed(a, ctx.project_id, w.marker), timeout=90, what="the write made after the restart is listed",
    )


def _listed(client, project_id: str, marker: str) -> None:
    assert rows_with(list_rows(client, project_id=project_id), marker), "memory missing from the listing"


# ── negative control ─────────────────────────────────────────────

_BEFORE = {"started_at": "2026-01-01T00:00:00Z", "health": "healthy", "restart_count": 0, "status": "running", "labels": {}}
_RESTARTED = {**_BEFORE, "started_at": "2026-01-01T00:10:00Z"}
_GOOD = [
    {**_BEFORE, "health": "healthy"}, {**_BEFORE, "health": "unhealthy"},
    {**_RESTARTED, "health": "starting"}, {**_RESTARTED, "health": "healthy"},
]


@proves("ops.autoheal")
@profiles("default", "autoheal")
@negative_control
def test_recovery_check_fails_without_a_restart_or_a_return_to_healthy():
    precondition(check_autoheal_recovery, _BEFORE, _GOOD, what="canned wedge -> restart -> healthy")
    wedged_only = [{**_BEFORE, "health": "healthy"}, {**_BEFORE, "health": "unhealthy"}, {**_BEFORE, "health": "unhealthy"}]
    expect_check_fails(check_autoheal_recovery, _BEFORE, wedged_only)  # StartedAt did not change: nobody restarted it
    stays_unhealthy = [*_GOOD[:3], {**_RESTARTED, "health": "unhealthy"}]
    expect_check_fails(check_autoheal_recovery, _BEFORE, stays_unhealthy)  # restarted but never healthy again
    expect_check_fails(check_autoheal_recovery, _BEFORE, [_GOOD[0], _GOOD[2], _GOOD[3]])  # never seen unhealthy
    expect_check_fails(check_autoheal_recovery, _BEFORE, [*_GOOD[:3], {**_RESTARTED, "health": "healthy", "restart_count": 1}])  # a crash
    expect_check_fails(check_autoheal_recovery, {**_BEFORE, "health": "starting"}, _GOOD)  # not healthy to begin with
    expect_check_fails(check_autoheal_recovery, _BEFORE, [])


@proves("ops.autoheal")
@profiles("default", "autoheal")
@negative_control
def test_wiring_check_fails_without_the_sidecar_or_the_project_scoped_label():
    key = label_key("nstest-acceptance")
    row = {"Service": AUTOHEAL, "State": "running"}
    worker = {"labels": {key: "true"}}
    precondition(check_autoheal_wiring, row, worker, "nstest-acceptance", what="canned sidecar + labelled worker")
    expect_check_fails(check_autoheal_wiring, None, worker, "nstest-acceptance")
    expect_check_fails(check_autoheal_wiring, {**row, "State": "exited"}, worker, "nstest-acceptance")
    expect_check_fails(check_autoheal_wiring, row, {"labels": {"com.neuralscape.autoheal": "true"}}, "nstest-acceptance")  # the old, unscoped key
    expect_check_fails(check_autoheal_wiring, row, {"labels": {label_key("nstest-other"): "true"}}, "nstest-acceptance")  # another project's key
    expect_check_fails(check_autoheal_wiring, row, {"labels": {key: "false"}}, "nstest-acceptance")  # opted out
