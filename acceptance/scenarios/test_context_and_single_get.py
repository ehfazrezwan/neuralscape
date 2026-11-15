"""Single-memory read (#437), session-start context, and domain_knowledge scope (#224).

LLM-free: facts go through /v1/memories/raw or the MCP ``remember`` tool (both
store a pre-categorised fact). Assertions are on unique markers only.
"""

from __future__ import annotations

import json

from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.mcp import MCPClient

UNKNOWN_ID = "99999999-9999-4999-8999-999999999999"


def _rows(ctx_json: dict) -> list[dict]:
    return [m for rows in (ctx_json.get("categories") or {}).values() for m in rows]


def check_get_is_404(client, memory_id: str) -> None:
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code == 404, f"GET /v1/memories/{memory_id[:40]} -> {r.status_code}: {r.text[:120]}"


def check_context_has_marker(client, project_id: str, marker: str) -> None:
    r = client.get(f"/v1/context/{project_id}", params={"limit": 200})
    assert r.status_code == 200, f"GET /v1/context -> {r.status_code}"
    assert marker in r.text, f"marker {marker} not in project context"


def _store_and_wait(client, content, category, **kw) -> str:
    task = client.store_raw(content, category, **kw)
    assert client.poll_task(task, timeout=120).get("status") == "completed", "write did not complete"
    return task


@proves("read.single-memory-get")
def test_malformed_memory_id_is_404_not_500(a):
    for bad in ("not-a-uuid", "123-45", "x" * 64):
        check_get_is_404(a, bad)


@proves("read.single-memory-get")
def test_wellformed_unknown_memory_id_is_404(a):
    check_get_is_404(a, UNKNOWN_ID)


@proves("read.single-memory-get")
@negative_control
def test_get_404_check_fails_for_a_readable_memory(ctx, a, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    hits = a.search(a_fact.content, project_id=a_fact.project_id)
    known = next((h["id"] for h in hits if a_fact.marker in h.get("memory", "")), None)
    assert known, "precondition: could not find A's memory id"
    expect_check_fails(check_get_is_404, a, known)


@proves("read.project-context", "read.global-context")
def test_project_context_shows_teammates_shared_memory_not_private(ctx, a, b):
    marker_shared = f"zq-{ctx.run_id}-ctx-shared"
    marker_private = f"zq-{ctx.run_id}-ctx-private"
    _store_and_wait(b, f"{marker_shared} shared project fact", "tech_stack",
                    scope="project", project_id=ctx.project_id, visibility="shared")
    _store_and_wait(b, f"{marker_private} private project fact", "tech_stack",
                    scope="project", project_id=ctx.project_id, visibility="private")
    eventually(lambda: check_context_has_marker(a, ctx.project_id, marker_shared),
               timeout=90, what="A's session context shows B's shared memory")
    r = a.get(f"/v1/context/{ctx.project_id}", params={"limit": 200})
    assert marker_private not in r.text, "A's session context leaks B's private memory"


@proves("read.project-context")
@negative_control
def test_context_check_fails_for_private_marker(ctx, a, b):
    marker = f"zq-{ctx.run_id}-ctx-private"
    expect_check_fails(check_context_has_marker, a, ctx.project_id, marker)


@proves("read.project-context")
def test_project_context_paging_is_true_newest_first(ctx, a):
    # Three project rows written one after another; the newest must lead the page.
    markers = [f"zq-{ctx.run_id}-page-{i}" for i in range(3)]
    for m in markers:
        _store_and_wait(a, f"{m} paging fact", "tech_stack", scope="project",
                        project_id=ctx.project_id, visibility="private")
    eventually(lambda: check_context_has_marker(a, ctx.project_id, markers[-1]),
               timeout=90, what="newest paging fact in context")
    full = a.get(f"/v1/context/{ctx.project_id}", params={"limit": 500}).json()
    rows = _rows(full)
    assert full["total"] >= len(rows) >= 3, f"total={full['total']} rows={len(rows)}"
    one = a.get(f"/v1/context/{ctx.project_id}", params={"limit": 1}).json()
    assert one["returned"] == 1 and one["total"] == full["total"]
    assert one["has_more"] is (full["total"] > 1)
    newest = max(rows, key=lambda m: m.get("created_at") or "")
    assert _rows(one)[0]["id"] == newest["id"], "limit=1 page is not the newest row"


@proves("model.category-taxonomy", "write.conversation-extraction")
def test_mcp_remember_domain_knowledge_with_project_id_is_project_scoped(ctx, a, mcp_a):
    marker = f"zq-{ctx.run_id}-domk"
    mcp_a.initialize()
    res = mcp_a.call_tool("remember", {
        "content": f"{marker} domain knowledge scoped to the project",
        "category": "domain_knowledge",
        "project_id": ctx.project_id,
        "wait": True,
    })
    assert not res.get("isError"), f"remember isError: {MCPClient.result_text(res)[:200]}"
    body = json.loads(MCPClient.result_text(res))
    assert body.get("status") == "completed", f"remember ended as {body.get('status')}"

    def _stored_as_project():
        r = a.get("/v1/memories", params={"project_id": ctx.project_id, "limit": 200})
        assert r.status_code == 200
        row = next((m for m in r.json() if marker in m.get("memory", "")), None)
        assert row is not None, "stored memory not listed under the project"
        assert row.get("scope") == "project", f"domain_knowledge stored with scope={row.get('scope')}"

    eventually(_stored_as_project, timeout=60, what="domain_knowledge stored with scope=project")
