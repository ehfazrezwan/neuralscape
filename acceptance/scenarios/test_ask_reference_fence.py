"""``ask`` at the medium tier does not pull the caller's reference-workspace rows into its evidence.

The keyword pass of ``/v1/ask`` (medium and high) is a store-wide scan; like recall and the session-start
context it must leave reference workspaces (content ingested through a non-default adapter) out. The only hard assertion is that the reference passage is neither cited nor quoted; the real-store
integration test proves the scan itself. The ingest runs with extract_facts=false, so only
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


def check_no_reference_row_used(body: dict, ref_ids: set[str]) -> None:
    """The deterministic half: a fenced reference row can neither be cited nor quoted."""
    leaked = ref_ids & set(body["citations"])
    assert not leaked, f"ask cited reference-workspace rows {sorted(leaked)}: {body['citations']}"
    assert "tolerance bands" not in body["answer"], f"ask quoted the reference passage: {body['answer'][:160]!r}"


def check_ask_leaves_reference_rows_out(client, ctx, fence, level: str = "medium") -> None:
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
    check_no_reference_row_used(body, ref_ids)


@proves("read.ask-tiers")
def test_ask_medium_keyword_pass_leaves_reference_rows_out(ctx, a, fence):
    fence["mem"].require()
    eventually(
        lambda: check_ask_leaves_reference_rows_out(a, ctx, fence), timeout=120, what="ask leaves the reference row out",
    )


@proves("read.ask-tiers")
@negative_control
def test_reference_row_check_fails_when_a_reference_row_is_cited():
    expect_check_fails(check_no_reference_row_used, {"citations": ["ref-1"], "answer": ""}, {"ref-1"})
    expect_check_fails(check_no_reference_row_used, {"citations": [], "answer": "see tolerance bands"}, {"ref-1"})
