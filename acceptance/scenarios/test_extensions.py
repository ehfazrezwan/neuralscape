"""Extensions and platform surfaces: registry + events, conversation-compiler
query/lint, live event stream (SSE), knowledge systems + router config, admin
remediation gates, session summaries and context assembly.

Documented behaviour under test: events are bound to the caller's token
identity; the SSE feed carries only the caller's own and shared events;
project knowledge config and the private-leakage admin tools are admin-only
(on a default stack the acceptance users are not admins, so they are refused).
"""

from __future__ import annotations

import time

import pytest

from conftest import ConvSeed
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import mcp_json, write_raw
from nsaccept.sse import SSEReader

SSE_WAIT = 60
GRAPH_WAIT = 180
EXPECTED_EXTENSIONS = {"conversation-compiler", "dreaming", "strategy-synthesizer"}


# ── the real checks ──────────────────────────────────────────────


def check_extension_list(body: dict) -> None:
    assert body.get("status") == "ok", f"status {body.get('status')!r}"
    exts = {e["name"]: e for e in body["extensions"]}
    missing = sorted(EXPECTED_EXTENSIONS - set(exts))
    assert not missing, f"extensions not registered: {missing} (have {sorted(exts)})"
    for name in EXPECTED_EXTENSIONS:
        assert {"name", "version", "description", "status", "hooks"} <= set(exts[name]), f"{name} keys: {sorted(exts[name])}"
    assert exts["conversation-compiler"]["status"] == "started", f"compiler status {exts['conversation-compiler']['status']!r}"


def check_lint_shape(body: dict) -> None:
    assert isinstance(body.get("findings"), list), f"findings: {body.get('findings')!r}"
    assert isinstance(body.get("checks_run"), int) and body["checks_run"] >= 1, f"checks_run: {body.get('checks_run')!r}"
    assert isinstance(body.get("files_scanned"), int), f"files_scanned: {body.get('files_scanned')!r}"
    for f in body["findings"]:
        assert {"check", "severity", "message"} <= set(f), f"finding keys: {sorted(f)}"


def check_query_shape(body: dict) -> None:
    assert isinstance(body.get("answer"), str) and body["answer"], f"answer: {body.get('answer')!r}"
    assert isinstance(body.get("sources"), list), f"sources: {body.get('sources')!r}"


def check_stream_got(frames: list[dict], marker: str, memory_id: str) -> dict:
    for ev in frames:
        if ev.get("memory_id") == memory_id or marker in str(ev.get("content", "")):
            return ev
    raise AssertionError(f"no stream event for {marker} / {memory_id} among {len(frames)} events")


def check_refused_as_non_admin(client: NSClient, method: str, path: str, body: dict) -> None:
    r = client.request(method, path, json=body)
    assert r.status_code == 403, f"{method} {path} -> {r.status_code} (want 403): {r.text[:120]}"


# ── extension registry + events ──────────────────────────────────


@proves("ext.extension-registry")
def test_registry_lists_the_three_local_extensions(a):
    r = a.get("/v1/extensions")
    assert r.status_code == 200, f"GET /v1/extensions -> {r.status_code}"
    check_extension_list(r.json())


@proves("ext.extension-registry")
@negative_control
def test_registry_check_fails_without_the_compiler(a):
    body = a.get("/v1/extensions").json()
    body["extensions"] = [e for e in body["extensions"] if e["name"] != "conversation-compiler"]
    expect_check_fails(check_extension_list, body)


@proves("ext.extension-registry", "auth.request-identity-resolution")
def test_events_are_accepted_for_the_caller_and_refused_for_another_user(ctx, a):
    kind = f"zq_{ctx.run_id}_probe"
    own = a.post("/v1/extensions/events", json={"event_type": kind, "payload": {"note": "acceptance"}})
    assert own.status_code == 200, f"custom event -> {own.status_code}: {own.text[:160]}"
    assert own.json()["event_type"] == kind and isinstance(own.json()["extensions_notified"], int), f"event body: {own.text[:200]}"
    spoof = a.post("/v1/extensions/events", json={"event_type": kind, "payload": {"user_id": ctx.user_b}})
    assert spoof.status_code == 400, f"event naming another user -> {spoof.status_code} (want 400): {spoof.text[:160]}"
    bad = a.post("/v1/extensions/events", json={"event_type": "session_start", "payload": {}})
    assert bad.status_code == 422, f"known event with an invalid payload -> {bad.status_code} (want 422)"


@proves("ext.extension-registry", "auth.request-identity-resolution")
def test_compiler_flush_is_bound_to_the_token_identity(ctx, a):
    r = a.post(
        "/v1/extensions/conversation-compiler/flush",
        json={"user_message": "x", "assistant_response": "y", "session_id": f"acc-{ctx.run_id}-flush", "user_id": ctx.user_b},
    )
    assert r.status_code == 400, f"flush naming another user -> {r.status_code} (want 400): {r.text[:160]}"


# ── conversation compiler: query + lint ──────────────────────────


@proves("ext.compiler-query-lint")
def test_structural_lint_answers_with_the_documented_shape(a):
    r = a.post("/v1/extensions/conversation-compiler/lint", json={"structural_only": True})
    assert r.status_code == 200, f"lint -> {r.status_code}: {r.text[:200]}"
    check_lint_shape(r.json())


@proves("ext.compiler-query-lint")
def test_query_answers_with_the_documented_shape(ctx, a):
    r = a.post(
        "/v1/extensions/conversation-compiler/query",
        json={"question": f"What does the vault say about zq-{ctx.run_id}?", "user_id": ctx.user_a}, timeout=120,
    )
    assert r.status_code == 200, f"query -> {r.status_code}: {r.text[:200]}"
    check_query_shape(r.json())


@proves("ext.compiler-query-lint")
@negative_control
def test_shape_checks_fail_on_the_wrong_payloads(a):
    expect_check_fails(check_lint_shape, a.get("/v1/extensions").json())
    expect_check_fails(check_query_shape, {"answer": "", "sources": []})


# ── live event stream (SSE) ──────────────────────────────────────


@pytest.fixture(scope="module")
def stream(ctx, a, b) -> dict:
    """Both users open /v1/stream, then A writes one private fact."""
    out: dict = {"error": "", "marker": f"zq-{ctx.run_id}-sse"}
    ra = SSEReader(ctx.base_url, ctx.token_a).start()
    rb = SSEReader(ctx.base_url, ctx.token_b).start()
    try:
        assert ra.wait_connected() and rb.wait_connected(), f"streams did not connect: A {ra.status_code} {ra.error}, B {rb.status_code} {rb.error}"
        w = write_raw(
            a, f"{out['marker']} The lighthouse lamp is trimmed at dusk.", "preference",
            marker=out["marker"], visibility="private",
        )
        out["write"] = w
        assert not w.error and w.completed, f"write for the stream scenario failed: {w.error or w.status}"
        out["a_event"] = ra.wait_for(lambda f: out["marker"] in f.data or w.memory_id in f.data, timeout=SSE_WAIT)
        time.sleep(5)  # give a wrongly-routed copy time to reach B
        out["a_events"], out["b_events"] = ra.events(), rb.events()
        out["a_status"], out["b_status"] = ra.status_code, rb.status_code
    except AssertionError as exc:
        out["error"] = str(exc)
    finally:
        ra.close()
        rb.close()
    return out


@proves("queue.event-stream")
def test_stream_delivers_the_callers_own_write_event(ctx, stream):
    assert not stream["error"], stream["error"]
    assert stream["a_status"] == 200, f"GET /v1/stream -> {stream['a_status']}"
    assert stream["a_event"] is not None, f"no stream event for the write within {SSE_WAIT}s ({len(stream['a_events'])} events seen)"
    assert stream["a_event"].event == "memory_stored", f"event type {stream['a_event'].event!r}"
    ev = check_stream_got(stream["a_events"], stream["marker"], stream["write"].memory_id)
    assert ev.get("memory_id") == stream["write"].memory_id or stream["marker"] in ev.get("content", ""), f"event: {ev}"
    assert ev.get("user_id") == ctx.user_a or ev.get("owner_user_id") == ctx.user_a, f"event owner: {ev}"


@proves("queue.event-stream")
def test_stream_never_delivers_a_private_event_to_another_user(stream):
    assert not stream["error"], stream["error"]
    assert stream["a_event"] is not None, "precondition: A never received its own event"
    assert stream["b_status"] == 200, f"B's GET /v1/stream -> {stream['b_status']}"
    leaked = [e for e in stream["b_events"] if stream["marker"] in str(e) or stream["write"].memory_id in str(e)]
    assert not leaked, f"B's stream received A's private event: {leaked}"


@proves("queue.event-stream")
@negative_control
def test_stream_check_fails_for_the_other_users_stream(stream):
    assert not stream["error"], stream["error"]
    assert stream["a_event"] is not None, "precondition: A never received its own event"
    expect_check_fails(check_stream_got, stream["b_events"], stream["marker"], stream["write"].memory_id)


# ── knowledge systems + router ───────────────────────────────────


@proves("knowledge.registry", "ops.health-endpoints")
def test_health_reports_every_registered_knowledge_system(a, anon):
    a.get("/v1/graph/episodes", params={"limit": 1})  # health reports lazily-initialised backends
    r = anon.get("/health")
    assert r.status_code == 200, f"GET /health -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["status"] in ("ok", "degraded") and set(body["checks"]) >= {"redis", "vector_store", "graph_store"}, f"health: {body}"
    systems = body.get("knowledge_systems") or []
    assert systems, "no knowledge systems reported by /health"
    for s in systems:
        assert {"name", "kind", "capabilities", "transport", "health"} <= set(s), f"system keys: {sorted(s)}"
    memory_system = [s for s in systems if "memory" in s["name"]]
    assert memory_system and memory_system[0]["health"] == "ok", f"memory knowledge system: {memory_system or systems}"


@proves("knowledge.registry")
@negative_control
def test_knowledge_registry_check_fails_without_systems(anon):
    body = anon.get("/health").json()

    def check(health: dict):
        assert health.get("knowledge_systems"), "no knowledge systems"

    expect_check_fails(check, {k: v for k, v in body.items() if k != "knowledge_systems"})


def check_default_config(body: dict, project_id: str) -> None:
    assert body == {"project_id": project_id, "code_systems": [], "fuse_code_into_recall": True, "default_engine": None}, (
        f"config: {body}"
    )


@proves("knowledge.router")
def test_project_knowledge_config_defaults_and_is_admin_write_only(ctx, a, b, mcp_a):
    path = f"/v1/projects/{ctx.project_id}/knowledge-config"
    check_default_config(a.get(path).json(), ctx.project_id)
    for client in (a, b):
        r = client.request("PUT", path, json={"project_id": ctx.project_id, "code_systems": ["code-native"], "fuse_code_into_recall": False})
        assert r.status_code == 403, f"non-admin PUT knowledge-config -> {r.status_code} (want 403): {r.text[:160]}"
    check_default_config(a.get(path).json(), ctx.project_id)  # the refused writes changed nothing
    mcp_a.initialize()
    check_default_config(mcp_json(mcp_a, "get_project_knowledge_config", {"project_id": ctx.project_id}), ctx.project_id)
    refused = mcp_json(mcp_a, "set_project_knowledge_config", {"project_id": ctx.project_id, "code_systems": ["code-native"]})
    assert "not authorized" in str(refused.get("error", "")), f"MCP set_project_knowledge_config answered {refused}"
    check_default_config(a.get(path).json(), ctx.project_id)


@proves("knowledge.router")
@negative_control
def test_default_config_check_fails_for_a_changed_config(ctx):
    expect_check_fails(
        check_default_config,
        {"project_id": ctx.project_id, "code_systems": ["code-native"], "fuse_code_into_recall": False, "default_engine": "x"},
        ctx.project_id,
    )


# ── admin remediation gates ──────────────────────────────────────


@proves("hygiene.private-leak-remediation")
def test_leak_audit_and_rescope_are_refused_to_non_dictators(ctx, a, b, mcp_a):
    for client in (a, b):
        check_refused_as_non_admin(client, "POST", "/v1/admin/audit-private-leakage", {})
        check_refused_as_non_admin(client, "POST", "/v1/admin/rescope-private-derivatives", {"dry_run": True})
    # naming another user as the target changes nothing about who is asking
    check_refused_as_non_admin(b, "POST", "/v1/admin/audit-private-leakage", {"user_id": ctx.user_a})
    mcp_a.initialize()
    for tool in ("audit_private_leakage", "rescope_private_derivatives"):
        out = mcp_json(mcp_a, tool, {})
        assert "not authorized" in str(out.get("error", "")), f"MCP {tool} answered {out}"


@proves("hygiene.private-leak-remediation")
@negative_control
def test_admin_refusal_check_fails_for_an_open_route(a):
    expect_check_fails(check_refused_as_non_admin, a, "POST", "/v1/search", {"query": "x"})


# ── session summaries + context assembly ─────────────────────────


def check_session_buffered(client: NSClient, seed: ConvSeed, fmt: str = "plain") -> dict:
    r = client.post("/v1/context/assemble", json={"budget_tokens": 4000, "session_id": seed.session_id, "format": fmt})
    assert r.status_code == 200, f"POST /v1/context/assemble -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["sections"]["messages_buffered"] >= 20, f"session buffer holds {body['sections']['messages_buffered']} messages"
    return body


@proves("read.context-assemble")
def test_assemble_serves_the_session_within_budget_in_each_format(ctx, a, conv_seed: ConvSeed):
    assert conv_seed.write.completed, f"conversation write ended as {conv_seed.write.status or conv_seed.write.error}"
    plain = check_session_buffered(a, conv_seed, "plain")
    assert plain["used_tokens"] <= plain["budget_tokens"] and "text" in plain["bundle"], f"plain bundle: {str(plain)[:200]}"
    assert plain["sections"]["messages_included"] >= 1
    anth = check_session_buffered(a, conv_seed, "anthropic")
    assert {"system", "messages"} <= set(anth["bundle"]), f"anthropic bundle keys {sorted(anth['bundle'])}"
    oai = check_session_buffered(a, conv_seed, "openai")
    assert "messages" in oai["bundle"], f"openai bundle keys {sorted(oai['bundle'])}"
    tiny = a.post("/v1/context/assemble", json={"budget_tokens": 100, "session_id": conv_seed.session_id})
    assert tiny.status_code == 200 and tiny.json()["used_tokens"] <= 100, f"100-token budget: {tiny.text[:200]}"
    assert a.post("/v1/context/assemble", json={"budget_tokens": 1000, "format": "xml"}).status_code == 422, "bad format accepted"


@proves("read.context-assemble")
@negative_control
def test_session_buffer_check_fails_for_user_b(b, conv_seed: ConvSeed):
    assert conv_seed.write.completed, f"conversation write ended as {conv_seed.write.status or conv_seed.write.error}"
    expect_check_fails(check_session_buffered, b, conv_seed)


@proves("queue.session-summarizer")
def test_a_long_session_gets_a_short_summary_slot(a, conv_seed: ConvSeed):
    assert conv_seed.write.completed, f"conversation write ended as {conv_seed.write.status or conv_seed.write.error}"

    def slot_ready():
        body = check_session_buffered(a, conv_seed)
        assert body["sections"]["summary_slot"] in ("short", "long"), f"summary_slot is {body['sections']['summary_slot']!r}"
        assert body["sections"]["summary_tokens"] > 0, "summary slot is empty"
        return body

    eventually(slot_ready, timeout=GRAPH_WAIT, interval=5, what="session summary slot refreshed by the graph worker")


@proves("queue.session-summarizer")
@negative_control
def test_summary_slot_check_fails_for_a_session_nobody_wrote(ctx, a):
    def check():
        body = a.post("/v1/context/assemble", json={"budget_tokens": 1000, "session_id": f"acc-{ctx.run_id}-nosuch"}).json()
        assert body["sections"]["summary_slot"] in ("short", "long"), f"summary_slot is {body['sections']['summary_slot']!r}"

    expect_check_fails(check)
