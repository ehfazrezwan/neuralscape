"""Graphify bundle ingest and the graph.json engine (default profile; no flag, the code-graph extra ships in the image).

A tiny ``graph.json`` (four nodes, three edges, one labelled community, symbols named with a unique
run prefix) is uploaded through ``POST /v1/ingest/files``. The ingest worker recognises it by file name
and shape, stores it as an owner-scoped artifact whose id is the bundle's ``graph_id``, and distils its
stable semantic layer (here one module fact). ``query_code_graph``, ``get_code_neighbors`` and ``code_path``
(REST and MCP) then answer from that exact file; another user cannot resolve the id.
"""

from __future__ import annotations

import json

import pytest

from cg_helpers import cg_get, check_mentions, mcp_text
from nsaccept import expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.upload import upload_files

INGEST_WAIT = 240


def build_graph(run_id: str) -> tuple[bytes, dict[str, str]]:
    """A Graphify node-link graph.json whose symbols carry the run prefix; returns (bytes, names)."""
    pre = f"Zq{run_id.capitalize()}"
    names = {"cls": f"{pre}Dock", "moor": f"{pre}_moor()", "cast": f"{pre}_cast_off()"}

    def node(id_: str, label: str, loc: str) -> dict:
        return {
            "id": id_, "label": label, "file_type": "code", "source_file": "src/svc/pier.py", "source_location": loc,
            "community": 0, "community_name": "Pier logistics", "norm_label": label.lower(),
        }

    def edge(source: str, target: str, relation: str) -> dict:
        return {"source": source, "target": target, "relation": relation, "confidence": "EXTRACTED", "weight": 1.0}

    graph = {
        "directed": True, "multigraph": False, "graph": {},
        "nodes": [
            node("src/svc/pier.py", "pier.py", "L1"),
            node(f"pier.{pre}Dock", names["cls"], "L10"),
            node(f"pier.{pre}Dock.moor", names["moor"], "L22"),
            node(f"pier.{pre}Dock.cast_off", names["cast"], "L31"),
        ],
        "links": [
            edge("src/svc/pier.py", f"pier.{pre}Dock", "contains"),
            edge(f"pier.{pre}Dock", f"pier.{pre}Dock.moor", "method"),
            edge(f"pier.{pre}Dock.moor", f"pier.{pre}Dock.cast_off", "calls"),
        ],
    }
    return json.dumps(graph).encode(), names


# ── the real checks ──────────────────────────────────────────────


def check_graph_answers(client: NSClient, graph_id: str, names: dict[str, str]) -> None:
    """query / neighbors / path over the uploaded graph name the symbols that are in it."""
    q = cg_get(client, "query", question=names["cls"], graph_id=graph_id)
    check_mentions(q["result"], names["cls"], names["moor"])
    nb = cg_get(client, "neighbors", label=names["cls"], graph_id=graph_id)
    check_mentions(nb["result"], names["moor"])
    path = cg_get(client, "path", source=names["moor"], target=names["cast"], graph_id=graph_id)
    check_mentions(path["result"], names["moor"], names["cast"])
    assert not path["result"].lower().startswith("no path"), f"path: {path['result'][:160]}"


# ── fixture: upload once ─────────────────────────────────────────


@pytest.fixture(scope="module")
def bundle(ctx, a) -> dict:
    data, names = build_graph(ctx.run_id)
    out: dict = {"names": names, "error": ""}
    try:
        r = upload_files(
            a, {"graph.json": data}, mime="application/json", category="tech_stack", scope="project",
            project_id=f"{ctx.project_id}-bundle", visibility="private", extract_facts=False,
        )
        info = r.json()
        assert info["count"] == 1, f"upload response: {info}"
        out["file_id"] = info["files"][0]["file_id"]
        status = a.poll_task(info["files"][0]["task_id"], timeout=INGEST_WAIT)
        assert status.get("status") == "completed", f"graph.json ingest ended as {status}"
        out["result"] = status.get("result") or {}
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@proves("knowledge.graphify-json-engine")
def test_an_uploaded_graph_json_is_ingested_as_a_code_graph_bundle(bundle):
    assert not bundle["error"], bundle["error"]
    res = bundle["result"]
    assert not res.get("skipped"), f"the bundle was skipped: {res.get('reason')}"
    assert res.get("doc_type") == "code_graph", f"graph.json was handled as {res.get('doc_type')!r}"
    assert res.get("graph_id") == bundle["file_id"], f"graph_id {res.get('graph_id')!r} is not the artifact id {bundle['file_id']!r}"
    assert (res.get("graph_nodes"), res.get("graph_edges")) == (4, 3), f"graph size: {res.get('graph_nodes')} nodes / {res.get('graph_edges')} edges"
    assert res.get("facts", 0) >= 1, f"no semantic-layer fact distilled: {res}"


@proves("knowledge.graphify-json-engine")
def test_query_neighbors_and_path_answer_from_the_uploaded_graph_over_rest_and_mcp(a, mcp_a, bundle):
    assert not bundle["error"], bundle["error"]
    names, gid = bundle["names"], bundle["file_id"]
    check_graph_answers(a, gid, names)
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "query_code_graph", {"question": names["cls"], "graph_id": gid}), names["cls"], names["moor"])
    check_mentions(mcp_text(mcp_a, "get_code_neighbors", {"label": names["cls"], "graph_id": gid}), names["moor"])
    check_mentions(mcp_text(mcp_a, "code_path", {"source": names["moor"], "target": names["cast"], "graph_id": gid}), names["cast"])


@proves("knowledge.graphify-json-engine")
@negative_control
def test_graph_answers_check_fails_for_another_users_id_and_for_a_symbol_not_in_the_graph(ctx, a, b, bundle):
    assert not bundle["error"], bundle["error"]
    gid = bundle["file_id"]
    foreign = b.get("/v1/code-graph/query", params={"question": bundle["names"]["cls"], "graph_id": gid})
    assert foreign.status_code in (403, 404), f"another user's query over A's graph_id -> {foreign.status_code}"
    assert bundle["names"]["cls"] not in foreign.text, "the refusal leaks the graph's content"
    expect_check_fails(check_graph_answers, b, gid, bundle["names"])
    missing = cg_get(a, "query", question=f"Zq{ctx.run_id.capitalize()}NeverInTheGraph", graph_id=gid)
    expect_check_fails(check_mentions, missing["result"], bundle["names"]["cls"])
