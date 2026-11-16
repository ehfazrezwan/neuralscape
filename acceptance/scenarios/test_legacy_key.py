"""The deprecated shared API key (NEURALSCAPE_API_KEY), on a stack that also has per-user tokens.

Documented behaviour under test (AUTH.md "Legacy shared key: one identity", auth.py):
- the key acts as exactly ONE identity, ``NEURALSCAPE_API_KEY_USER_ID`` (here the profile's fixed ``acc-legacy``),
  on REST and on MCP, whatever a body/query ``user_id`` or an MCP ``arguments["user_id"]`` names
- a REST body/query ``user_id`` naming anyone else is a 400 (``NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID`` is
  left at its default, off); naming the bound user itself is fine
- every response to the key carries ``X-Neuralscape-Deprecation``; a per-user token's does not
- a wrong key, or a near miss of the right one, is a 401
- per-user tokens keep working on the same stack; private memory and task status never cross between the
  legacy identity and A or B

Runs only under the ``legacy-key`` profile, whose stack the gate starts with a generated key and hands to
the harness as ``NS_LEGACY_KEY``; without it the fixtures fail (they never skip).
"""

from __future__ import annotations

import json
import secrets

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, precondition, profiles, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, list_rows, mcp_json, rows_with, write_raw
from nsaccept.mcp import MCPClient

pytestmark = profiles("legacy-key")

DEPRECATION_HEADER = "X-Neuralscape-Deprecation"
READ = "/v1/memories?limit=1"
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "ns-acceptance", "version": "1"}},
}


# ── the real checks (reused by the negative controls) ────────────


def check_identity(client: NSClient, expected_user: str) -> None:
    r = client.get("/v1/queue/status")
    assert r.status_code == 200, f"GET /v1/queue/status -> {r.status_code}: {r.text[:120]}"
    got = r.json().get("user_id")
    assert got == expected_user, f"server resolved identity {got!r}, expected {expected_user!r}"


def check_mcp_identity(mcp: MCPClient, expected_user: str, claimed_user: str) -> None:
    """MCP ``queue_status`` while ``arguments["user_id"]`` names ``claimed_user``: the answer is for the credential's user."""
    body = mcp_json(mcp, "queue_status", {"user_id": claimed_user})
    assert body.get("user_id") == expected_user, f"MCP queue_status acted as {body.get('user_id')!r}, expected {expected_user!r} (argument named {claimed_user!r})"


def check_body_user_refused(client: NSClient, other_user: str, marker: str) -> None:
    """A search, a list and a raw write that name ``other_user`` are all 400 (the write last: nothing is enqueued by a refusal)."""
    s = client.post("/v1/search", json={"query": marker, "user_id": other_user})
    assert s.status_code == 400, f"POST /v1/search naming {other_user!r} -> {s.status_code} (want 400): {s.text[:160]}"
    g = client.get("/v1/memories", params={"user_id": other_user, "limit": 1})
    assert g.status_code == 400, f"GET /v1/memories?user_id={other_user!r} -> {g.status_code} (want 400): {g.text[:160]}"
    w = client.post("/v1/memories/raw", json={"content": f"{marker} refused", "category": "preference", "user_id": other_user})
    assert w.status_code == 400, f"POST /v1/memories/raw naming {other_user!r} -> {w.status_code} (want 400): {w.text[:160]}"


def check_deprecation_header(client: NSClient) -> None:
    r = client.get(READ)
    assert r.status_code == 200, f"GET {READ} -> {r.status_code}: {r.text[:120]}"
    value = r.headers.get(DEPRECATION_HEADER, "")
    assert "deprecated" in value.lower(), f"no {DEPRECATION_HEADER} header with a deprecation notice on a REST response (got {value!r})"


def check_mcp_deprecation_header(mcp: MCPClient) -> None:
    r = mcp.post_raw(INIT)
    assert r.status_code == 200, f"MCP initialize -> {r.status_code}: {r.text[:120]}"
    value = r.headers.get(DEPRECATION_HEADER, "")
    assert "deprecated" in value.lower(), f"no {DEPRECATION_HEADER} header with a deprecation notice on an MCP response (got {value!r})"


def check_key_refused(base_url: str, token: str) -> None:
    """REST and MCP both answer 401 for this bearer string."""
    c = NSClient(base_url, token)
    m = MCPClient(base_url, token)
    try:
        r = c.get(READ)
        assert r.status_code == 401, f"GET {READ} with that credential -> {r.status_code} (want 401)"
        assert DEPRECATION_HEADER not in r.headers, "a refused credential was answered as the legacy key"
        rm = m.post_raw(INIT)
        assert rm.status_code == 401, f"MCP initialize with that credential -> {rm.status_code} (want 401)"
    finally:
        c.close()
        m.close()


def check_per_user_token(client: NSClient, expected_user: str) -> None:
    """A per-user token on the legacy stack: 200, its own identity, and not answered as the legacy key."""
    r = client.get(READ)
    assert r.status_code == 200, f"GET {READ} -> {r.status_code}: {r.text[:120]}"
    assert DEPRECATION_HEADER not in r.headers, "a per-user token was answered with the legacy deprecation header"
    check_identity(client, expected_user)


def recalled(client: NSClient, marker: str) -> bool:
    return any(marker in r.get("memory", "") for r in client.search(marker, limit=10))


def check_recalls(client: NSClient, marker: str) -> None:
    assert recalled(client, marker), f"{marker} not recalled"


def check_listed(client: NSClient, marker: str) -> None:
    assert rows_with(list_rows(client), marker), f"{marker} not in GET /v1/memories"


def check_mcp_recalls(mcp: MCPClient, marker: str) -> None:
    rows = mcp_json(mcp, "recall_memories", {"query": marker, "limit": 10})
    assert isinstance(rows, list), f"recall_memories returned {type(rows).__name__}: {str(rows)[:120]}"
    assert any(marker in (r.get("memory") or "") for r in rows), f"{marker} not in {len(rows)} MCP recall results"


def check_task_readable(client: NSClient, prefix: str, w: Written) -> None:
    r = client.get(f"{prefix}/{w.task_id}")
    assert r.status_code == 200, f"GET {prefix}/<task> -> {r.status_code}: {r.text[:120]}"
    body = r.json()
    assert body.get("task_id") == w.task_id and body.get("status") == "completed", f"status body: {str(body)[:160]}"
    assert w.memory_id in json.dumps(body), "task result does not carry the written memory id"


def check_task_hidden(client: NSClient, prefix: str, w: Written) -> None:
    r = client.get(f"{prefix}/{w.task_id}")
    assert r.status_code in (403, 404), f"GET {prefix}/<task> -> {r.status_code} (want 403/404): {r.text[:120]}"
    assert w.memory_id not in r.text and w.marker not in r.text, "refusal body leaks the task's data"


def near_miss(key: str) -> str:
    return key[:-1] + ("0" if key[-1] != "0" else "1")


# ── module fixtures: one write per identity ──────────────────────


@pytest.fixture(scope="module")
def legacy_write(ctx, legacy) -> Written:
    marker = f"zq-{ctx.run_id}-legacykey"
    return write_raw(legacy, f"{marker} The legacy key's probe keeps its notes beside a copper kettle.", "preference",
                     marker=marker, scope="global", visibility="private")


@pytest.fixture(scope="module")
def a_write(ctx, a) -> Written:
    marker = f"zq-{ctx.run_id}-legacyakey"
    return write_raw(a, f"{marker} User A's probe keeps its notes beside a brass lantern.", "preference",
                     marker=marker, scope="global", visibility="private")


# ── identity: exactly one ────────────────────────────────────────


@proves("auth.legacy-shared-key")
def test_the_legacy_key_acts_as_exactly_the_configured_identity_on_rest(ctx, legacy):
    assert ctx.legacy_user not in (ctx.user_a, ctx.user_b), "the profile's legacy identity must be its own user"
    check_identity(legacy, ctx.legacy_user)
    # No user_id at all, and the bound user's own id, are both the same single identity.
    own = legacy.post("/v1/search", json={"query": f"zq-{ctx.run_id}-nothing", "user_id": ctx.legacy_user})
    assert own.status_code == 200, f"naming the bound user itself -> {own.status_code}: {own.text[:160]}"


@proves("auth.legacy-shared-key")
@negative_control
def test_identity_check_fails_for_another_user_and_for_a_per_user_token(ctx, a, legacy):
    precondition(check_identity, legacy, ctx.legacy_user, what="the legacy key resolves to its configured user")
    expect_check_fails(check_identity, legacy, ctx.user_a)
    expect_check_fails(check_identity, a, ctx.legacy_user)


@proves("auth.legacy-shared-key")
def test_the_legacy_key_acts_as_exactly_the_configured_identity_on_mcp(ctx, mcp_legacy):
    mcp_legacy.initialize()
    check_mcp_identity(mcp_legacy, ctx.legacy_user, claimed_user=ctx.user_a)
    check_mcp_identity(mcp_legacy, ctx.legacy_user, claimed_user=ctx.user_b)
    check_mcp_identity(mcp_legacy, ctx.legacy_user, claimed_user=ctx.legacy_user)


@proves("auth.legacy-shared-key")
@negative_control
def test_mcp_identity_check_fails_for_another_user_and_for_a_per_user_token(ctx, mcp_a, mcp_legacy):
    mcp_a.initialize()
    mcp_legacy.initialize()
    precondition(check_mcp_identity, mcp_legacy, ctx.legacy_user, ctx.user_a, what="MCP resolves the legacy key to its configured user")
    expect_check_fails(check_mcp_identity, mcp_legacy, ctx.user_a, ctx.user_a)
    expect_check_fails(check_mcp_identity, mcp_a, ctx.legacy_user, ctx.legacy_user)  # a per-user token wins over any argument


# ── writes land under the one identity ───────────────────────────


@proves("auth.legacy-shared-key")
def test_a_rest_write_with_the_legacy_key_lands_under_the_configured_identity(ctx, a, b, legacy, legacy_write):
    legacy_write.require()
    eventually(lambda: check_recalls(legacy, legacy_write.marker), timeout=90, what="legacy write recalled by the legacy identity")
    check_listed(legacy, legacy_write.marker)
    assert not recalled(a, legacy_write.marker), "the legacy identity's private memory is recalled by user A"
    assert not recalled(b, legacy_write.marker), "the legacy identity's private memory is recalled by user B"
    assert not rows_with(list_rows(a), legacy_write.marker), "the legacy identity's memory is listed for user A"


@proves("auth.legacy-shared-key")
@negative_control
def test_recall_check_fails_for_a_user_that_did_not_write(a, b, legacy, legacy_write):
    legacy_write.require()
    precondition(lambda: eventually(lambda: check_recalls(legacy, legacy_write.marker), timeout=90), what="the legacy identity recalls its own write")
    expect_check_fails(check_recalls, a, legacy_write.marker)
    expect_check_fails(check_recalls, b, legacy_write.marker)


@proves("auth.legacy-shared-key")
def test_an_mcp_write_naming_another_user_lands_under_the_legacy_identity(ctx, a, legacy, mcp_legacy):
    marker = f"zq-{ctx.run_id}-legacymcp"
    mcp_legacy.initialize()
    body = mcp_json(mcp_legacy, "remember", {
        "content": f"{marker} Written over MCP with the legacy key while naming user A.",
        "category": "preference", "visibility": "private", "user_id": ctx.user_a, "wait": True,
    })
    assert body.get("status") == "completed", f"MCP remember ended as {body.get('status')!r}: {str(body)[:160]}"
    eventually(lambda: check_listed(legacy, marker), timeout=60, what="MCP write listed for the legacy identity")
    assert not rows_with(list_rows(a), marker), "an MCP write naming user A landed in A's memories"
    assert not recalled(a, marker), "an MCP write naming user A is recalled by A"


@proves("auth.legacy-shared-key")
@negative_control
def test_listed_check_fails_for_a_user_that_did_not_write(ctx, a, legacy, mcp_legacy):
    marker = f"zq-{ctx.run_id}-legacymcpctl"
    mcp_legacy.initialize()
    body = mcp_json(mcp_legacy, "remember", {
        "content": f"{marker} Control write over MCP with the legacy key.", "category": "preference", "visibility": "private", "wait": True,
    })
    assert body.get("status") == "completed", f"MCP remember ended as {body.get('status')!r}"
    precondition(lambda: eventually(lambda: check_listed(legacy, marker), timeout=60), what="the legacy identity lists its own MCP write")
    expect_check_fails(check_listed, a, marker)


# ── a body user_id naming someone else ───────────────────────────


@proves("auth.legacy-shared-key")
def test_a_body_or_query_user_id_naming_another_user_is_refused_with_400(ctx, a, b, legacy):
    marker = f"zq-{ctx.run_id}-legacyspoof"
    check_body_user_refused(legacy, ctx.user_a, marker)
    check_body_user_refused(legacy, ctx.user_b, marker)
    assert not rows_with(list_rows(a), marker) and not rows_with(list_rows(b), marker), "a refused write landed in another user's memories"
    assert not rows_with(list_rows(legacy), marker), "a refused write landed under the legacy identity"


@proves("auth.legacy-shared-key")
@negative_control
def test_refusal_check_fails_when_the_request_names_the_bound_user(ctx, legacy):
    marker = f"zq-{ctx.run_id}-legacyspoofctl"
    precondition(check_body_user_refused, legacy, ctx.user_a, marker, what="naming another user is refused")
    expect_check_fails(check_body_user_refused, legacy, ctx.legacy_user, marker)  # naming itself is allowed, so it is not a 400


# ── the deprecation header ───────────────────────────────────────


@proves("auth.legacy-shared-key")
def test_every_legacy_key_response_carries_the_deprecation_header(legacy, mcp_legacy):
    check_deprecation_header(legacy)
    check_mcp_deprecation_header(mcp_legacy)
    refused = legacy.get("/v1/memories", params={"user_id": "someone-else", "limit": 1})
    assert refused.status_code == 400 and DEPRECATION_HEADER in refused.headers, "a 400 answered to the legacy key lacks the deprecation header"


@proves("auth.legacy-shared-key")
@negative_control
def test_deprecation_header_check_fails_for_a_per_user_token(a, mcp_a, legacy, mcp_legacy):
    precondition(check_deprecation_header, legacy, what="the legacy key's REST responses carry the header")
    precondition(check_mcp_deprecation_header, mcp_legacy, what="the legacy key's MCP responses carry the header")
    expect_check_fails(check_deprecation_header, a)
    expect_check_fails(check_mcp_deprecation_header, mcp_a)


# ── wrong keys ───────────────────────────────────────────────────


@proves("auth.legacy-shared-key", "auth.bearer-middleware")
def test_a_wrong_or_near_miss_key_is_401_on_rest_and_mcp(ctx, bad):
    key = ctx.legacy_key
    for label, token in (
        ("near miss (last character)", near_miss(key)),
        ("truncated", key[:-1]),
        ("key plus a suffix", key + "0"),
        ("random key", secrets.token_hex(32)),
        ("wrong-secret per-user token", bad.token),
    ):
        try:
            check_key_refused(ctx.base_url, token)
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from None


@proves("auth.legacy-shared-key", "auth.bearer-middleware")
@negative_control
def test_key_refused_check_fails_for_the_real_key(ctx):
    precondition(check_key_refused, ctx.base_url, near_miss(ctx.legacy_key), what="a near-miss key is refused")
    expect_check_fails(check_key_refused, ctx.base_url, ctx.legacy_key)


# ── per-user tokens alongside ────────────────────────────────────


@proves("auth.legacy-shared-key")
def test_per_user_tokens_still_work_on_the_same_stack(ctx, a, b):
    check_per_user_token(a, ctx.user_a)
    check_per_user_token(b, ctx.user_b)


@proves("auth.legacy-shared-key")
@negative_control
def test_per_user_token_check_fails_for_the_legacy_key_and_a_wrong_secret_token(ctx, a, bad, legacy):
    precondition(check_per_user_token, a, ctx.user_a, what="a per-user token works on the legacy stack")
    expect_check_fails(check_per_user_token, legacy, ctx.legacy_user)  # right identity, but answered as the legacy key
    expect_check_fails(check_per_user_token, bad, ctx.user_a)


# ── isolation both ways ──────────────────────────────────────────


@proves("auth.legacy-shared-key")
def test_the_legacy_identity_cannot_see_user_as_private_memory(a, legacy, mcp_legacy, a_write):
    a_write.require()
    eventually(lambda: check_recalls(a, a_write.marker), timeout=90, what="A's write recalled by A")
    assert not recalled(legacy, a_write.marker), "the legacy identity recalls user A's private memory over REST"
    assert not rows_with(list_rows(legacy), a_write.marker), "the legacy identity lists user A's private memory"
    mcp_legacy.initialize()
    rows = mcp_json(mcp_legacy, "recall_memories", {"query": a_write.marker, "limit": 10})
    assert a_write.marker not in json.dumps(rows), "the legacy identity recalls user A's private memory over MCP"


@proves("auth.legacy-shared-key")
@negative_control
def test_recall_checks_fail_for_the_legacy_identity_on_a_private_memory_of_a(a, legacy, mcp_a, mcp_legacy, a_write):
    a_write.require()
    mcp_a.initialize()
    mcp_legacy.initialize()
    precondition(lambda: eventually(lambda: check_recalls(a, a_write.marker), timeout=90), what="A recalls its own write")
    precondition(lambda: eventually(lambda: check_mcp_recalls(mcp_a, a_write.marker), timeout=90), what="A's MCP recall finds its own write")
    expect_check_fails(check_recalls, legacy, a_write.marker)
    expect_check_fails(check_mcp_recalls, mcp_legacy, a_write.marker)


# ── task status ──────────────────────────────────────────────────


@proves("auth.legacy-shared-key")
def test_task_status_of_a_legacy_key_write_is_readable_only_by_the_legacy_identity(a, b, legacy, legacy_write, a_write):
    legacy_write.require()
    a_write.require()
    for prefix in ("/v1/memories/status", "/memories/status"):
        check_task_readable(legacy, prefix, legacy_write)
        check_task_hidden(a, prefix, legacy_write)
        check_task_hidden(b, prefix, legacy_write)
        check_task_hidden(legacy, prefix, a_write)  # and the legacy identity cannot read A's task either


@proves("auth.legacy-shared-key")
@negative_control
def test_task_checks_fail_for_the_wrong_reader(a, legacy, legacy_write):
    legacy_write.require()
    precondition(check_task_readable, legacy, "/v1/memories/status", legacy_write, what="the legacy identity reads its own task")
    expect_check_fails(check_task_hidden, legacy, "/v1/memories/status", legacy_write)
    expect_check_fails(check_task_readable, a, "/v1/memories/status", legacy_write)
