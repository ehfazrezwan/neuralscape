"""The native code-intelligence engine with CODE_NATIVE_ENABLED (profile ``flags-on``).

Folds scripts/code_graph_e2e.py into the through-NS flow: index a repo that already exists inside
the image (the service's own ``knowledge/`` package, named by the profile's CODE_REPOS) over
POST /v1/code-graph/index, poll the ingest-queue task, then ask every query tool about symbols
that certainly exist in that source (``resolve_systems`` calls ``_has_coding_signal``, both in
router.py), over REST and MCP, route with knowledge_system=auto, fuse the code leg into recall,
and finally reset the code space.

The embedder is off in this profile (lexical locate only: no model download), so a locate query
must carry words that are in the symbol's name, signature or docstring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, profiles, proves
from nsaccept.mcp import MCPClient

from cg_helpers import CALLEE, SYMBOL, cg_get, check_mentions, mcp_text

pytestmark = profiles("flags-on")

INDEX_WAIT = 900
SYSTEM = "code-native"
KNOWN_SYSTEMS = {"code-native", "code-graphify-lib", "code-cbm"}


# ── the real checks ──────────────────────────────────────────────



# ── fixture: index once ──────────────────────────────────────────


@dataclass
class Native:
    code_space: str
    project: str
    repo_source: str
    index: dict = field(default_factory=dict)
    error: str = ""


def repo_source_of(ctx) -> str:
    repos = json.loads(ctx.profile_flags.get("CODE_REPOS") or "{}")
    assert "svc" in repos, f"the profile's CODE_REPOS has no 'svc' repo: {ctx.profile_flags.get('CODE_REPOS')!r}"
    return repos["svc"]


@pytest.fixture(scope="module")
def native(ctx, a) -> Native:
    n = Native(code_space=f"code--{ctx.user_a}--svc", project=f"{ctx.project_id}-code", repo_source=repo_source_of(ctx))
    try:
        r = a.post("/v1/code-graph/index", json={"repo_source": n.repo_source, "system": SYSTEM, "project_id": n.project, "code_space": n.code_space})
        assert r.status_code == 202, f"POST /v1/code-graph/index -> {r.status_code}: {r.text[:200]}"
        body = r.json()
        assert body["system"] == SYSTEM and body["code_space"] == n.code_space and body["status"] == "accepted", f"index accepted body: {body}"
        st = a.poll_task(body["task_id"], timeout=INDEX_WAIT)
        assert st.get("status") == "completed", f"index task ended as {st}"
        n.index = st["result"]
    except AssertionError as exc:
        n.error = str(exc)
    return n


def _ready(n: Native) -> Native:
    assert not n.error, n.error
    return n


# ── engine + index pipeline ──────────────────────────────────────


@proves("knowledge.native-engine", "knowledge.code-index-pipeline", "knowledge.code-dispatch")
def test_the_native_engine_is_registered_and_the_through_ns_index_builds_a_graph(a, anon, native: Native):
    n = _ready(native)
    idx = n.index
    assert idx.get("ok") is True, f"index result: {idx}"
    assert idx["system"] == SYSTEM and idx["code_space"] == n.code_space, f"index result: {idx}"
    assert idx["files_indexed"] > 0 and idx["symbols_indexed"] > 0 and idx["edges_indexed"] > 0, f"nothing indexed: {idx}"
    health = anon.get("/health").json()
    systems = {s["name"]: s for s in health.get("knowledge_systems", [])}
    assert SYSTEM in systems and systems[SYSTEM]["health"] == "ok", f"/health knowledge_systems: {sorted(systems)}"
    cfg = a.get(f"/v1/projects/{n.project}/knowledge-config").json()  # the index set the project's routing
    assert cfg["code_systems"] == [SYSTEM] and cfg["default_engine"] == SYSTEM, f"project config after the index: {cfg}"


@proves("knowledge.native-engine", "knowledge.code-index-pipeline")
@negative_control
def test_index_pipeline_reports_an_unconfigured_repo_as_not_indexed(ctx, a):
    r = a.post("/v1/code-graph/index", json={"repo_source": "/nowhere", "system": SYSTEM, "code_space": f"code--{ctx.user_a}--not-configured"})
    assert r.status_code == 202, f"index request -> {r.status_code}"  # accepted, then answered honestly by the worker
    st = a.poll_task(r.json()["task_id"], timeout=300)
    indexed = st.get("status") == "completed" and (st.get("result") or {}).get("ok") is True

    def check():
        assert indexed, f"index of an unconfigured repo ended as {st.get('status')}: {st.get('error') or st.get('result')}"

    expect_check_fails(check)  # the "it was indexed" check cannot pass for a repo that is not in CODE_REPOS


# ── query tools ──────────────────────────────────────────────────


@proves("knowledge.code-query-tools", "knowledge.code-dispatch", "knowledge.native-engine")
def test_query_finds_a_symbol_over_rest_and_mcp(ctx, a, mcp_a, native: Native):
    n = _ready(native)
    body = cg_get(a, "query", question=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM)
    assert body["system"] == SYSTEM and body["routed_by"] == SYSTEM, f"served by {body.get('system')}, routed_by {body.get('routed_by')}"
    check_mentions(body["result"], SYMBOL, "router.py")
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "query_code_graph", {"question": SYMBOL, "graph_id": n.code_space, "knowledge_system": SYSTEM}), SYMBOL)


@proves("knowledge.code-query-tools")
@negative_control
def test_query_check_fails_for_a_symbol_that_does_not_exist(ctx, a, native: Native):
    n = _ready(native)
    body = cg_get(a, "query", question=f"zq_{ctx.run_id}_no_such_symbol", graph_id=n.code_space, knowledge_system=SYSTEM)
    expect_check_fails(check_mentions, body["result"], SYMBOL)


@proves("knowledge.native-call-resolver", "knowledge.code-query-tools")
def test_neighbors_and_path_follow_a_real_call_edge(a, mcp_a, native: Native):
    n = _ready(native)
    nb = cg_get(a, "neighbors", label=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM)
    check_mentions(nb["result"], SYMBOL, CALLEE)  # resolve_systems --CALLS--> _has_coding_signal
    path = cg_get(a, "path", source=SYMBOL, target=CALLEE, graph_id=n.code_space, knowledge_system=SYSTEM)
    check_mentions(path["result"], SYMBOL, CALLEE)
    assert not path["result"].lower().startswith("no path"), f"path: {path['result'][:200]}"
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "get_code_neighbors", {"label": SYMBOL, "graph_id": n.code_space, "knowledge_system": SYSTEM}), CALLEE)
    check_mentions(mcp_text(mcp_a, "code_path", {"source": SYMBOL, "target": CALLEE, "graph_id": n.code_space, "knowledge_system": SYSTEM}), CALLEE)


@proves("knowledge.native-call-resolver")
@negative_control
def test_neighbors_check_fails_for_a_symbol_it_does_not_call(a, native: Native):
    n = _ready(native)
    nb = cg_get(a, "neighbors", label=CALLEE, graph_id=n.code_space, knowledge_system=SYSTEM)
    expect_check_fails(check_mentions, nb["result"], "get_project_config")  # a real symbol of the package, but not a neighbour


# ── locate + impact ──────────────────────────────────────────────


def check_located(body: dict, symbol: str) -> dict:
    hits = body.get("results") or []
    hit = next((h for h in hits if str(h.get("fqn", "")).endswith(symbol)), None)
    assert hit is not None, f"{symbol} not among {[h.get('fqn') for h in hits]}"
    assert hit.get("file", "").endswith("router.py") and int(hit.get("line", 0)) > 0, f"hit {hit}"
    return hit


@proves("knowledge.native-locate", "knowledge.locate-impact-tools")
def test_locate_finds_a_symbol_by_name_and_by_its_docstring(a, mcp_a, native: Native):
    n = _ready(native)
    by_name = cg_get(a, "locate", query=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM, k=5)
    hit = check_located(by_name, SYMBOL)
    assert "def " in hit.get("signature", "") and SYMBOL in hit["signature"], f"signature {hit.get('signature')!r}"
    by_doc = cg_get(a, "locate", query="three layer deterministic resolver which knowledge systems to query", graph_id=n.code_space, knowledge_system=SYSTEM, k=5)
    check_located(by_doc, SYMBOL)
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "locate", {"query": SYMBOL, "graph_id": n.code_space, "knowledge_system": SYSTEM, "k": 5}), SYMBOL)


@proves("knowledge.native-locate")
@negative_control
def test_locate_check_fails_for_an_unrelated_query(ctx, a, native: Native):
    n = _ready(native)
    body = cg_get(a, "locate", query=f"zq{ctx.run_id}unrelated", graph_id=n.code_space, knowledge_system=SYSTEM, k=5)
    expect_check_fails(check_located, body, SYMBOL)


@proves("knowledge.locate-impact-tools", "knowledge.code-query-tools")
def test_impact_lists_the_caller_of_a_symbol(a, mcp_a, native: Native):
    n = _ready(native)
    body = cg_get(a, "impact", symbol=CALLEE, graph_id=n.code_space, knowledge_system=SYSTEM, max_hops=3)
    check_mentions(body["text"], SYMBOL)  # changing the callee affects its caller
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "code_impact", {"symbol": CALLEE, "graph_id": n.code_space, "knowledge_system": SYSTEM}), SYMBOL)


@proves("knowledge.locate-impact-tools")
@negative_control
def test_impact_check_fails_for_a_symbol_nothing_calls_it_through(a, native: Native):
    n = _ready(native)
    body = cg_get(a, "impact", symbol=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM, max_hops=1)
    expect_check_fails(check_mentions, body["text"], "get_system_that_does_not_exist")


# ── auto-routing ─────────────────────────────────────────────────


@proves("knowledge.autoroute", "knowledge.code-dispatch")
def test_auto_routing_picks_a_healthy_engine_per_operation_and_says_which(a, mcp_a, native: Native):
    n = _ready(native)
    for path, params, key in (
        ("query", {"question": SYMBOL}, "result"),
        ("locate", {"query": SYMBOL, "k": 5}, "result"),
        ("neighbors", {"label": SYMBOL}, "result"),
    ):
        body = cg_get(a, path, graph_id=n.code_space, knowledge_system="auto", **params)
        assert body["routed_by"] == "auto", f"{path}: routed_by {body.get('routed_by')!r}"
        assert body["system"] in KNOWN_SYSTEMS, f"{path}: served by {body.get('system')!r}"
        check_mentions(body[key], SYMBOL)
    locate = cg_get(a, "locate", query=SYMBOL, graph_id=n.code_space, knowledge_system="auto", k=5)
    assert locate["system"] == SYSTEM, f"locate prefers the native engine; served by {locate['system']!r}"
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "query_code_graph", {"question": SYMBOL, "graph_id": n.code_space, "knowledge_system": "auto"}), SYMBOL)


@proves("knowledge.autoroute")
@negative_control
def test_auto_routing_refuses_a_code_space_nobody_indexed(ctx, a):
    r = a.get("/v1/code-graph/query", params={"question": SYMBOL, "graph_id": f"code--{ctx.user_a}--never-indexed", "knowledge_system": "auto"})

    def check():
        assert r.status_code == 200 and SYMBOL in r.text, f"auto-routed query over an unindexed space: {r.status_code} {r.text[:160]}"

    expect_check_fails(check)


# ── memory fusion ────────────────────────────────────────────────


def fused_recall(mcp: MCPClient, project: str, query: str) -> object:
    res = mcp.call_tool("recall_memories", {"query": query, "project_id": project, "limit": 10})
    text = MCPClient.result_text(res)
    assert not res.get("isError"), f"recall_memories isError: {text[:200]}"
    return json.loads(text)


def check_fused(body: object, *, symbol: str, marker: str) -> dict:
    assert isinstance(body, dict) and body.get("fused") is True, f"recall was not fused: {str(body)[:200]}"
    sections = body["sections"]
    check_mentions(sections["structure"]["content"], symbol)
    assert body["routing"]["served_by"] in KNOWN_SYSTEMS, f"routing: {body.get('routing')}"
    assert any(marker in m.get("memory", "") for m in sections.get("memories", [])), f"{marker} missing from the memory section"
    return body


@proves("read.recall-code-fusion", "knowledge.fusion", "knowledge.router")
def test_recall_fuses_a_code_hit_with_memories_when_the_project_is_configured_for_it(ctx, a, mcp_a, mcp_dictator, native: Native):
    n = _ready(native)
    marker = f"zq-{ctx.run_id}-fusion"
    w = a.store_raw(f"{marker} The routing layer of the knowledge package is explained in the docstring of resolve_systems.", "architecture",
                    scope="project", project_id=n.project, visibility="private")
    assert a.poll_task(w, timeout=120)["status"] == "completed", "the fusion note was not stored"
    mcp_a.initialize()
    mcp_dictator.initialize()
    query = f"callers of `{SYMBOL}` and how the {marker} routing layer works"

    def fused():
        return check_fused(fused_recall(mcp_a, n.project, query), symbol=SYMBOL, marker=marker)

    try:
        eventually(fused, timeout=90, interval=3, what="code leg fused into recall")
        # the project's own switch: off -> plain memory recall, on -> fused again
        off = mcp_dictator.call_tool("set_project_knowledge_config", {"project_id": n.project, "code_systems": [SYSTEM], "fuse_code_into_recall": False, "default_engine": SYSTEM})
        assert "error" not in json.loads(MCPClient.result_text(off)), f"dictator could not switch fusion off: {MCPClient.result_text(off)[:160]}"
        edited = a.get(f"/v1/projects/{n.project}/knowledge-config").json()  # the edit must not erase what the index recorded
        assert edited["code_space"] == n.code_space and edited["fuse_code_into_recall"] is False, f"config after the edit: {edited}"
        plain = fused_recall(mcp_a, n.project, query)
        assert not (isinstance(plain, dict) and plain.get("fused")), "recall still fused with fuse_code_into_recall=false"
    finally:
        mcp_dictator.call_tool("set_project_knowledge_config", {"project_id": n.project, "code_systems": [SYSTEM], "fuse_code_into_recall": True, "default_engine": SYSTEM})
    eventually(fused, timeout=60, interval=3, what="fusion back on")
    assert a.get(f"/v1/projects/{n.project}/knowledge-config").json()["code_space"] == n.code_space, "code_space lost after the config edits"


@proves("read.recall-code-fusion", "knowledge.fusion")
@negative_control
def test_fusion_check_fails_for_prose_recall_and_for_a_plain_list(ctx, a, mcp_a, native: Native):
    n = _ready(native)
    mcp_a.initialize()
    prose = fused_recall(mcp_a, n.project, "what colour is the lighthouse lamp")  # no coding signal: the router keeps it base-only
    expect_check_fails(check_fused, prose, symbol=SYMBOL, marker="x")
    expect_check_fails(check_fused, [], symbol=SYMBOL, marker="x")


@proves("knowledge.code-anchors", "knowledge.fusion")
def test_a_memory_anchored_to_a_symbol_comes_back_with_the_symbol(ctx, a, mcp_a, native: Native):
    n = _ready(native)
    located = cg_get(a, "locate", query=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM, k=5)
    fqn = check_located(located, SYMBOL)["fqn"]
    canonical = ".".join(p for i, p in enumerate(fqn.split(".")) if not (i == 0 and p in ("src", "lib")))
    marker = f"zq-{ctx.run_id}-anchor"
    task = a.store_raw(
        f"{marker} We keep resolve_systems side-effect free so the router can be called on every recall.", "decision",
        scope="project", project_id=n.project, visibility="private",
        source_ref={"connector_id": f"acc-{ctx.run_id}-anchor", "connector_type": "file_upload", "external_id": f"svc::{canonical}"},
    )
    assert a.poll_task(task, timeout=120)["status"] == "completed", "the anchored memory was not stored"

    def attached():
        nb = cg_get(a, "neighbors", label=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM)
        check_mentions(nb["result"], marker)

    eventually(attached, timeout=60, interval=3, what="anchored memory attached to the symbol")
    mcp_a.initialize()
    body = fused_recall(mcp_a, n.project, f"callers of `{SYMBOL}`")
    assert isinstance(body, dict) and marker in json.dumps(body.get("sections", {}).get("semantics", {})), "the anchored memory is missing from the fused semantics section"


@proves("knowledge.code-anchors")
@negative_control
def test_anchor_check_fails_for_another_user(ctx, b, native: Native):
    _ready(native)
    nb = cg_get(b, "neighbors", label=SYMBOL, graph_id=f"code--{ctx.user_b}--svc", knowledge_system=SYSTEM)
    expect_check_fails(check_mentions, nb["result"], f"zq-{ctx.run_id}-anchor")


# ── cold delete ──────────────────────────────────────────────────


@proves("knowledge.code-graph-delete")
def test_deleting_the_code_space_empties_it_and_keeps_the_memories(ctx, a, mcp_a, native: Native):
    n = _ready(native)
    before = cg_get(a, "query", question=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM)["result"]
    check_mentions(before, SYMBOL)
    r = a.delete("/v1/code-graph/graph", params={"graph_id": n.code_space, "system": SYSTEM})
    assert r.status_code == 200 and r.json().get("deleted") is True, f"DELETE /v1/code-graph/graph -> {r.status_code}: {r.text[:200]}"
    assert r.json()["engine_result"]["nodes_deleted"] > 0, f"nothing was deleted from the indexed space: {r.json()}"
    after = cg_get(a, "query", question=SYMBOL, graph_id=n.code_space, knowledge_system=SYSTEM)["result"]
    expect_check_fails(check_mentions, after, SYMBOL)  # the graph is gone
    anchored = [h for h in a.search(f"zq-{ctx.run_id}-anchor resolve_systems side-effect free", project_id=n.project) if f"zq-{ctx.run_id}-anchor" in h.get("memory", "")]
    assert anchored, "deleting the code graph deleted the memory anchored to it"
    again = a.delete("/v1/code-graph/graph", params={"graph_id": n.code_space, "system": SYSTEM})
    assert again.status_code == 200, f"second DELETE (idempotent) -> {again.status_code}"
    mcp_a.initialize()
    out = json.loads(mcp_text(mcp_a, "code_graph_delete", {"graph_id": n.code_space, "system": SYSTEM}))
    assert "error" not in out, f"MCP code_graph_delete: {out}"


@proves("knowledge.code-graph-delete")
@negative_control
def test_delete_check_fails_for_a_space_that_was_never_there(ctx, b):
    r = b.delete("/v1/code-graph/graph", params={"graph_id": f"code--{ctx.user_b}--svc", "system": SYSTEM})
    assert r.status_code == 200, f"DELETE of an unindexed space -> {r.status_code}"

    def check():
        assert r.json()["engine_result"]["nodes_deleted"] > 0, f"deleted something: {r.json()}"

    expect_check_fails(check)
