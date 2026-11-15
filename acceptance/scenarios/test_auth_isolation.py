"""Identity and isolation: tokens bind identity, tasks are private to their enqueuer.

Documented behaviour under test (auth.py, main.py):
- the token's user_id is the only identity; a body ``user_id`` that disagrees is a 400
- a task id is readable only by the user who enqueued it; anyone else gets the
  same 404 as for an id that never existed (never a 200 carrying the task)
- a tampered or expired token is a 401, not a silent fallback to another user
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from conftest import Fact
from nsaccept import eventually, expect_check_fails, needs, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, mcp_json, write_raw
from nsaccept.mcp import MCPClient
from nsaccept.tokens import mint_local, mint_via_compose


@pytest.fixture(scope="module")
def a_task(ctx, a, graph_seed) -> Written:
    """A's own raw write; its task id is what B tries to read.

    ``graph_seed`` is requested first on purpose: it enqueues the graph facts
    so the graph worker is busy while the rest of the suite runs.
    """
    marker = f"zq-{ctx.run_id}-authtask"
    return write_raw(a, f"{marker} The probe's lunch order is a vermilion salad.", "preference", marker=marker)


# ── the real checks (reused by the negative controls) ────────────


def check_task_readable(client: NSClient, path_prefix: str, w: Written) -> None:
    """The client can read the task: 200 with the task's own terminal state."""
    r = client.get(f"{path_prefix}/{w.task_id}")
    assert r.status_code == 200, f"GET {path_prefix}/<task> -> {r.status_code}: {r.text[:120]}"
    body = r.json()
    assert body.get("task_id") == w.task_id, f"status body names another task: {body}"
    assert body.get("status") == "completed", f"status was {body.get('status')!r}"
    assert w.memory_id in json.dumps(body), "task result does not carry the written memory id"


def check_task_hidden(client: NSClient, path_prefix: str, w: Written) -> None:
    """The client may NOT read the task: 403/404, and nothing of the task leaks."""
    r = client.get(f"{path_prefix}/{w.task_id}")
    assert r.status_code in (403, 404), f"GET {path_prefix}/<task> -> {r.status_code} (want 403/404): {r.text[:120]}"
    assert w.memory_id not in r.text and w.marker not in r.text, "refusal body leaks the task's data"


def check_identity(client: NSClient, expected_user: str) -> None:
    r = client.get("/v1/queue/status")
    assert r.status_code == 200, f"GET /v1/queue/status -> {r.status_code}: {r.text[:120]}"
    got = r.json().get("user_id")
    assert got == expected_user, f"server resolved identity {got!r}, token says {expected_user!r}"


def check_mcp_recall_finds(mcp: MCPClient, fact: Fact) -> None:
    rows = mcp_json(mcp, "recall_memories", {"query": fact.content, "project_id": fact.project_id, "limit": 10})
    assert isinstance(rows, list), f"recall_memories returned {type(rows).__name__}: {str(rows)[:120]}"
    assert any(fact.marker in (r.get("memory") or "") for r in rows), (
        f"marker {fact.marker} not in {len(rows)} MCP recall results"
    )


def _retarget(token: str, user_id: str) -> str:
    """Same signature, payload re-pointed at ``user_id`` (a forged identity swap)."""
    payload_b64, sig = token.split(".", 1)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["user_id"] = user_id
    forged = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(forged).rstrip(b"=").decode() + "." + sig


# ── task ownership ───────────────────────────────────────────────


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
def test_owner_reads_own_task_on_v1_and_legacy_routes(a, a_task):
    a_task.require()
    check_task_readable(a, "/v1/memories/status", a_task)
    check_task_readable(a, "/memories/status", a_task)


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
def test_other_user_cannot_read_task_on_v1_route(b, a_task):
    a_task.require()
    check_task_hidden(b, "/v1/memories/status", a_task)


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
def test_other_user_cannot_read_task_on_legacy_route(b, a_task):
    a_task.require()
    check_task_hidden(b, "/memories/status", a_task)


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
@negative_control
def test_task_hidden_check_fails_for_the_owner(a, a_task):
    a_task.require()
    expect_check_fails(check_task_hidden, a, "/v1/memories/status", a_task)


@proves("auth.task-status-isolation", "queue.task-owner-isolation")
@negative_control
def test_task_readable_check_fails_for_the_other_user(b, a_task):
    a_task.require()
    expect_check_fails(check_task_readable, b, "/v1/memories/status", a_task)


# ── identity resolution ──────────────────────────────────────────


@proves("auth.request-identity-resolution")
def test_server_resolves_each_token_to_its_own_user(ctx, a, b):
    check_identity(a, ctx.user_a)
    check_identity(b, ctx.user_b)


@proves("auth.request-identity-resolution")
@negative_control
def test_identity_check_fails_when_token_and_expected_user_differ(ctx, a):
    expect_check_fails(check_identity, a, ctx.user_b)


@proves("auth.request-identity-resolution")
def test_body_user_id_cannot_impersonate_another_user(ctx, a, b):
    marker = f"zq-{ctx.run_id}-spoof"
    r = a.post(
        "/v1/memories/raw",
        json={"content": f"{marker} spoofed", "category": "preference", "user_id": ctx.user_b},
    )
    assert r.status_code == 400, f"raw write with body user_id=B as A -> {r.status_code}: {r.text[:160]}"
    s = a.post("/v1/search", json={"query": marker, "user_id": ctx.user_b})
    assert s.status_code == 400, f"search with body user_id=B as A -> {s.status_code}: {s.text[:160]}"
    batch = a.post(
        "/v1/memories/raw/batch",
        json={"memories": [{"content": f"{marker} spoofed batch", "category": "preference", "user_id": ctx.user_b}]},
    )
    assert batch.status_code == 400, f"batch item user_id=B as A -> {batch.status_code}: {batch.text[:160]}"
    rows = b.get("/v1/memories", params={"limit": 500}).json()
    assert marker not in json.dumps(rows), "a spoofed write landed in user B's memories"


@proves("auth.request-identity-resolution")
def test_body_user_id_equal_to_token_user_is_accepted(ctx, a):
    marker = f"zq-{ctx.run_id}-selfid"
    r = a.post(
        "/v1/search", json={"query": marker, "user_id": ctx.user_a},
    )
    assert r.status_code == 200, f"search with matching body user_id -> {r.status_code}: {r.text[:160]}"


@proves("auth.request-identity-resolution", "auth.bearer-middleware")
def test_token_with_swapped_user_id_is_rejected(ctx, a):
    forged = NSClient(ctx.base_url, _retarget(ctx.token_a, ctx.user_b), ctx.user_b)
    try:
        r = forged.get("/v1/memories", params={"limit": 1})
    finally:
        forged.close()
    assert r.status_code == 401, f"token with swapped payload -> {r.status_code} (want 401): {r.text[:120]}"


@proves("auth.bearer-middleware")
def test_expired_token_is_rejected(ctx):
    secret = os.environ.get("NS_TOKEN_SECRET", "").strip()
    if not secret:
        pytest.skip("NS_TOKEN_SECRET not set: cannot sign an expired token for this stack")
    stale = mint_local(ctx.user_a, secret, ttl_seconds=-3600)
    c = NSClient(ctx.base_url, stale, ctx.user_a)
    fresh = NSClient(ctx.base_url, mint_local(ctx.user_a, secret, ttl_seconds=3600), ctx.user_a)
    try:
        r_stale = c.get("/v1/memories", params={"limit": 1})
        r_fresh = fresh.get("/v1/memories", params={"limit": 1})
    finally:
        c.close()
        fresh.close()
    assert r_fresh.status_code == 200, f"control token signed with the same secret -> {r_fresh.status_code}"
    assert r_stale.status_code == 401, f"expired token -> {r_stale.status_code} (want 401)"


# ── token issuance (docker compose exec path) ────────────────────


@proves("auth.token-issuance")
@needs("compose")
def test_compose_issued_token_works_and_carries_its_user(ctx):
    user_c = f"acc-{ctx.run_id}-c"
    token = mint_via_compose(ctx.compose_project, user_c, os.environ.get("NS_API_SERVICE", "neuralscape"))
    c = NSClient(ctx.base_url, token, user_c)
    try:
        check_identity(c, user_c)
        assert c.get("/v1/memories", params={"limit": 1}).status_code == 200
    finally:
        c.close()


@proves("auth.token-issuance")
@negative_control
@needs("compose")
def test_issued_token_identity_check_fails_for_wrong_user(ctx, a):
    expect_check_fails(check_identity, a, f"acc-{ctx.run_id}-c")


# ── MCP identity (contextvar) ────────────────────────────────────


@proves("auth.mcp-identity-contextvar")
def test_mcp_recall_returns_own_fact_to_a(mcp_a, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    mcp_a.initialize()
    eventually(lambda: check_mcp_recall_finds(mcp_a, a_fact), timeout=60, what="A's MCP recall finds A's fact")


@proves("auth.mcp-identity-contextvar")
def test_mcp_recall_never_returns_a_fact_to_b(mcp_b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    mcp_b.initialize()
    rows = mcp_json(mcp_b, "recall_memories", {"query": a_fact.content, "project_id": a_fact.project_id, "limit": 20})
    assert a_fact.marker not in json.dumps(rows), "B's MCP recall_memories leaked A's private fact"


@proves("auth.mcp-identity-contextvar")
@negative_control
def test_mcp_recall_check_fails_for_b(mcp_b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    mcp_b.initialize()
    expect_check_fails(check_mcp_recall_finds, mcp_b, a_fact)
