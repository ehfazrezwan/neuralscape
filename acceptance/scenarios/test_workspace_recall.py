"""Reference-workspace recall: content ingested with a non-default adapter is
stored in ``ref-<adapter>--<title-slug>`` and is returned only when the caller
names that workspace (REST ``workspaces`` field and MCP ``recall_memories``).

No LLM wording is asserted: the ingest runs with extract_facts=false, so only
verbatim passages are stored, and the check is on a unique marker.
"""

from __future__ import annotations

import re

from nsaccept import eventually, expect_check_fails, negative_control, proves

ADAPTER = "trading_strategy"


def _workspace_for(title: str) -> str:
    # The pipeline slugifies the stored artifact file name ("<title>.md").
    slug = re.sub(r"[^a-z0-9\-]+", "-", f"{title}.md".lower()).strip("-")[:30]
    return f"ref-{ADAPTER}--{slug}"


def _ingest(client, ctx, title: str, marker: str) -> None:
    resp = client.post("/v1/ingest/text", json={
        "content": f"{marker} reference chapter about position sizing and stops.",
        "title": title,
        "adapter": ADAPTER,
        "extract_facts": False,
        "project_id": ctx.project_id,
        "scope": "project",
    })
    assert resp.status_code == 202, f"POST /v1/ingest/text -> {resp.status_code}: {resp.text[:200]}"
    task_id = resp.json().get("task_id")
    assert task_id, f"no task id in {resp.text[:200]}"
    assert client.poll_task(task_id, timeout=180)["status"] == "completed"


def check_rest_recall(client, ctx, marker, workspaces):
    extra = {"workspaces": workspaces} if workspaces else {}
    hits = client.search(marker, project_id=ctx.project_id, **extra)
    assert any(marker in h.get("memory", "") for h in hits), "marker not recalled"


def check_mcp_recall(mcp, ctx, marker, workspaces):
    args = {"query": marker, "project_id": ctx.project_id}
    if workspaces:
        args["workspaces"] = workspaces
    text = mcp.result_text(mcp.call_tool("recall_memories", args))
    assert marker in text, "marker not recalled over MCP"


@proves("ext.workspaces", "ingest.pipeline", "ext.trading-adapter", "read.hybrid-search-core")
def test_reference_workspace_recall_over_rest_and_mcp(ctx, a, mcp_a):
    title = f"wsacc{ctx.run_id}"
    marker = f"zq-{ctx.run_id}-wsref"
    ws = _workspace_for(title)
    _ingest(a, ctx, title, marker)

    # Present: returned.
    eventually(lambda: check_rest_recall(a, ctx, marker, [ws]), timeout=90, what="REST recall with workspace")
    check_mcp_recall(mcp_a, ctx, marker, [ws])
    # Absent: the unchanged default still fences it out.
    expect_check_fails(check_rest_recall, a, ctx, marker, None)
    expect_check_fails(check_mcp_recall, mcp_a, ctx, marker, None)


@proves("ext.workspaces", "read.hybrid-search-core")
def test_invalid_workspace_names_are_rejected(a):
    r = a.post("/v1/search", json={"query": "x", "workspaces": ["bad name"]})
    assert r.status_code == 422, f"expected 422, got {r.status_code}"


@proves("ext.workspaces", "read.hybrid-search-core")
@negative_control
def test_workspace_recall_check_fails_for_wrong_workspace(ctx, a):
    expect_check_fails(check_rest_recall, a, ctx, f"zq-{ctx.run_id}-wsref", ["ref-no-such-workspace"])
