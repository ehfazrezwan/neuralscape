"""The running stack itself: the image and compose files boot every service, each worker
queue completes a job, and the API logs structured JSON events.

- Stack shape (``--compose-project``): the services the compose files define for the
  stack (API, the three ARQ workers, Neo4j, Redis, Qdrant) are all running, none has exited
  or restarted, and every one that declares a health check reports healthy.
- Worker queues: a job completes on the fast (main), the graph and the ingest queue. The cron
  schedules of those workers are not observable in a run this short and are not claimed.
- Logging: the API's stdout lines that are JSON are structlog events. The request-id header
  and the level/timestamp keys on every line are the sec-08 behaviour: **expected red until
  sec-08 (PR 515)**.
"""

from __future__ import annotations

import json
import os

from conftest import GraphSeed
from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import graph_listing
from nsaccept.runctx import app_services, container_restart_count, parse_ps_json, qdrant_service

API = os.environ.get("NS_API_SERVICE", "neuralscape")
REDIS = os.environ.get("NS_REDIS_SERVICE", "redis")
NEO4J = os.environ.get("NS_NEO4J_SERVICE", "neo4j")
GRAPH_WAIT = 240


# ── the real checks ──────────────────────────────────────────────


def check_stack_shape(rows: list[dict], restarts: dict[str, int], required: list[str]) -> None:
    by_service = {r.get("Service"): r for r in rows}
    for svc in required:
        row = by_service.get(svc)
        assert row is not None, f"{svc} missing from `compose ps` (have {sorted(k for k in by_service if k)})"
        assert row.get("State") == "running", f"{svc} is {row.get('State')!r} (exit code {row.get('ExitCode')})"
        assert not row.get("ExitCode"), f"{svc} exited with {row.get('ExitCode')}"
        assert (row.get("Health") or "") in ("", "healthy"), f"{svc} health is {row.get('Health')!r}"
        assert restarts.get(svc, 0) == 0, f"{svc} restarted {restarts[svc]} time(s)"


def check_episode_exists(client: NSClient, project: str, marker: str) -> None:
    episodes = graph_listing(client, "episodes", project)
    assert any(marker in (e.get("content") or "") for e in episodes), f"no episode carries {marker} among {len(episodes)}"


def json_log_events(text: str) -> list[dict]:
    events = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise AssertionError(f"a log line that starts like JSON does not parse: {line[:120]!r}") from None
        assert isinstance(obj, dict), f"a JSON log line is not an object: {line[:120]!r}"
        events.append(obj)
    return events


def check_json_events(text: str) -> None:
    events = json_log_events(text)
    assert len(events) >= 3, f"only {len(events)} JSON event line(s) in the API log tail"
    missing = [e for e in events if not e.get("event")]
    assert not missing, f"{len(missing)} JSON log line(s) have no 'event' key: {missing[0]}"


def check_levelled_timestamped(text: str) -> None:
    events = json_log_events(text)
    assert events, "no JSON event lines in the API log tail"
    bare = [e for e in events if not (e.get("level") and e.get("timestamp"))]
    assert not bare, f"{len(bare)} of {len(events)} JSON log lines lack level/timestamp, e.g. {bare[0]}"


def check_request_id_echo(resp_headers, sent: str) -> None:
    got = resp_headers.get("x-request-id")
    assert got == sent, f"X-Request-ID answered {got!r}, sent {sent!r}"


# ── stack shape ──────────────────────────────────────────────────


@proves("ops.compose-dev", "ops.dockerfile")
@needs("compose")
def test_every_service_of_the_stack_is_running_healthy_and_has_never_restarted(ctx):
    """The runtime image (built from the Dockerfile) started the API and all three worker roles under the
    compose files' commands, and the backing services are up. A crashing entrypoint, a missing COPY path or a
    broken healthcheck shows here as an exited, restarting or unhealthy service."""
    required = [*app_services(), NEO4J, REDIS, qdrant_service()]
    out = ctx.compose("ps", "--all", "--format", "json", check=False)
    rows = parse_ps_json(out.stdout or "")
    restarts = {r["Service"]: container_restart_count(r["ID"]) for r in rows if r.get("ID") and r.get("Service") in required}
    check_stack_shape(rows, restarts, required)


@proves("ops.compose-dev", "ops.dockerfile")
@negative_control
def test_stack_shape_check_fails_for_an_exited_unhealthy_restarted_or_absent_service():
    ok = [{"Service": "api", "State": "running", "ExitCode": 0, "Health": "healthy"}]
    check_stack_shape(ok, {"api": 0}, ["api"])
    expect_check_fails(check_stack_shape, [{**ok[0], "State": "exited", "ExitCode": 1}], {}, ["api"])
    expect_check_fails(check_stack_shape, [{**ok[0], "Health": "unhealthy"}], {}, ["api"])
    expect_check_fails(check_stack_shape, ok, {"api": 2}, ["api"])
    expect_check_fails(check_stack_shape, ok, {}, ["api", "worker"])


# ── worker queues ────────────────────────────────────────────────


@proves("queue.arq-workers")
def test_a_job_completes_on_each_of_the_main_graph_and_ingest_queues(ctx, a, a_fact, graph_seed: GraphSeed):
    """main: the raw write task completed (fast worker). ingest: a text ingest runs to completion on the
    ingest worker. graph: the first seed write's episode was written by the graph worker."""
    assert a_fact.status.get("status") == "completed", f"fast-queue write ended as {a_fact.status}"
    marker = f"zq-{ctx.run_id}-qingest"
    r = a.post("/v1/ingest/text", json={
        "content": f"# Queue note\n\n{marker} Bulk text travels on the ingest queue to its own worker.\n",
        "title": f"{marker}-note", "category": "tech_stack", "scope": "project", "project_id": ctx.project_id,
        "visibility": "private", "extract_facts": False,
    })
    assert r.status_code == 202, f"POST /v1/ingest/text -> {r.status_code}: {r.text[:200]}"
    status = a.poll_task(r.json()["task_id"], timeout=180)
    assert status.get("status") == "completed", f"ingest-queue job ended as {status}"
    result = status.get("result") or {}
    assert not result.get("skipped") and result.get("passages", 0) >= 1, f"ingest result: {result}"
    graph_seed.first.require()
    eventually(
        lambda: check_episode_exists(a, ctx.project_id, graph_seed.first.marker),
        timeout=GRAPH_WAIT, interval=10, what="graph-queue job's episode",
    )


@proves("queue.arq-workers")
@negative_control
def test_queue_checks_fail_for_a_job_that_never_ran(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    expect_check_fails(check_episode_exists, a, ctx.project_id, f"zq-{ctx.run_id}-no-such-job")
    unknown = a.get("/v1/memories/status/00000000-0000-0000-0000-000000000000")
    assert unknown.status_code == 404, f"an unknown task id answered {unknown.status_code}"


# ── structured logging ───────────────────────────────────────────


def api_log_tail(ctx, a: NSClient) -> str:
    a.get("/v1/queue/status")  # make sure the tail is not empty
    out = ctx.compose("logs", "--no-color", "--no-log-prefix", "--tail", "500", API, check=False)
    return out.stdout or ""


@proves("ops.structured-logging")
@needs("compose")
def test_the_api_log_is_json_events(ctx, a):
    check_json_events(api_log_tail(ctx, a))


@proves("ops.structured-logging")
@needs("compose")
def test_every_json_log_line_carries_level_and_timestamp(ctx, a):
    """Expected red until sec-08 (PR 515): plain `logging` records are rendered without level or timestamp today."""
    check_levelled_timestamped(api_log_tail(ctx, a))


@proves("ops.structured-logging")
def test_each_response_echoes_or_generates_a_request_id(ctx, a, anon):
    """Expected red until sec-08 (PR 515): a sane incoming X-Request-ID is echoed, an unsafe one is replaced by a
    generated id, and a request without one gets one, on 200s and on auth rejections alike."""
    rid = f"zq-{ctx.run_id}-rid"
    check_request_id_echo(a.get("/v1/queue/status", headers={"X-Request-ID": rid}).headers, rid)
    generated = a.get("/v1/queue/status").headers.get("x-request-id")
    assert generated and generated != rid, f"no generated request id: {generated!r}"
    unsafe = a.get("/v1/queue/status", headers={"X-Request-ID": "not safe, has spaces"}).headers.get("x-request-id")
    assert unsafe and unsafe != "not safe, has spaces" and " " not in unsafe, f"unsafe id echoed or none generated: {unsafe!r}"
    check_request_id_echo(anon.get("/v1/queue/status", headers={"X-Request-ID": rid}).headers, rid)  # a 401 carries it too


@proves("ops.structured-logging")
@negative_control
def test_log_and_request_id_checks_fail_for_console_lines_and_missing_headers():
    console = "INFO  something happened\nINFO  something else\nplain text\n"
    expect_check_fails(check_json_events, console)
    expect_check_fails(check_json_events, '{"event": "a"}\n{"event": "b"}\n{"no_event": 1}\n{"event": "c"}')
    expect_check_fails(check_levelled_timestamped, '{"event": "plain logging record"}')
    check_levelled_timestamped('{"event": "x", "level": "info", "timestamp": "2026-01-01T00:00:00Z"}')
    expect_check_fails(check_request_id_echo, {}, "zq-1")
