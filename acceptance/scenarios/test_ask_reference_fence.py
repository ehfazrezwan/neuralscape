"""``ask`` at the medium tier does not pull the caller's reference-workspace rows into its evidence.

The keyword pass of ``/v1/ask`` (medium and high) is a store-wide scan; like recall and the session-start
context it must leave reference workspaces (content ingested through a non-default adapter) out. A memory
row carrying the same rare words is still found and cited. The ingest runs with extract_facts=false, so only
a verbatim passage is stored in the reference workspace.
"""

from __future__ import annotations

import re

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.helpers import write_raw

ADAPTER = "trading_strategy"


def _workspace_for(title: str) -> str:
    slug = re.sub(r"[^a-z0-9\-]+", "-", f"{title}.md".lower()).strip("-")[:30]
    return f"ref-{ADAPTER}--{slug}"


@pytest.fixture(scope="module")
def fence(ctx, a) -> dict:
    run = ctx.run_id
    title = f"askfence{run}"
    rare = f"Zx{run.capitalize()}Fence"
    ref_text = f"{rare} gridlocked sprocket handbook chapter on tolerance bands."
    mem_text = f"{rare} gridlocked sprocket is calibrated every Thursday by the pilot team."
    resp = a.post("/v1/ingest/text", json={
        "content": ref_text, "title": title, "adapter": ADAPTER, "extract_facts": False,
        "project_id": ctx.project_id, "scope": "project",
    })
    assert resp.status_code == 202, f"POST /v1/ingest/text -> {resp.status_code}: {resp.text[:200]}"
    done = a.poll_task(resp.json()["task_id"], timeout=180)
    assert done["status"] == "completed", done
    mem = write_raw(
        a, mem_text, "tech_stack", marker=rare, scope="project", project_id=ctx.project_id, visibility="private",
    )
    return {"rare": rare, "workspace": _workspace_for(title), "mem": mem}


def _ids(rows: list[dict], text: str) -> set[str]:
    return {r["id"] for r in rows if text in r.get("memory", "")}


def check_ask_cites_only_memory_rows(client, ctx, fence, level: str = "medium") -> None:
    rare = fence["rare"]
    ref_rows = client.search(rare, project_id=ctx.project_id, workspaces=[fence["workspace"]])
    ref_ids = _ids(ref_rows, "handbook chapter")
    assert ref_ids, "the reference passage is not searchable in its own workspace"
    question = f"What is said about the {rare} gridlocked sprocket? Name the handbook chapter or the calibration day."
    r = client.post(
        "/v1/ask", json={"question": question, "reasoning_level": level, "project_id": ctx.project_id}, timeout=240,
    )
    assert r.status_code == 200, f"POST /v1/ask -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert any(s.startswith("keyword:") for s in body["searches"]), f"no keyword pass among {body['searches']}"
    leaked = ref_ids & set(body["citations"])
    assert not leaked, f"ask cited reference-workspace rows {sorted(leaked)}: {body['citations']}"
    assert fence["mem"].memory_id in body["citations"] or "Thursday" in body["answer"], (
        f"ask did not use the memory row: {body['answer'][:160]!r} {body['citations']}"
    )


@proves("read.keyword-scan", "read.ask-tiers")
def test_ask_medium_keyword_pass_leaves_reference_rows_out(ctx, a, fence):
    fence["mem"].require()
    eventually(
        lambda: check_ask_cites_only_memory_rows(a, ctx, fence), timeout=120, what="ask cites only the memory row",
    )


@proves("read.keyword-scan")
@negative_control
def test_ask_fence_check_fails_for_user_b(ctx, b, fence):
    fence["mem"].require()
    expect_check_fails(check_ask_cites_only_memory_rows, b, ctx, fence)
