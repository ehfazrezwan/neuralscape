"""Read paths: hybrid recall (dense + keyword), index-first recall and batch get,
timeline, provenance chains, listings, contexts, ask, metrics, OKF export,
REST/MCP surface inventories.

Writes are LLM-free (/v1/memories/raw) and happen once per module; the only LLM
calls are the /ask questions (three for user A, one for user B's negative control).
"""

from __future__ import annotations

import io
import re
import uuid
import zipfile

import pytest

from conftest import shared_project
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, list_rows, mcp_json, rows_with, write_raw
from nsaccept.probe import probe_route

from route_table import (
    COVERED_ELSEWHERE, EXCLUDED_PREFIXES, MCP_CODE_TOOLS, MCP_CORE_TOOLS, PROBES,
)

PARAPHRASE = "which tool builds our artifacts each night on the staging cluster"
ASK_Q = "What release tool does the vermiculite team use every Thursday? Give its exact name."


# ── module fixtures ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def rd(ctx, a, graph_seed) -> dict:
    """Facts for the read scenarios (``graph_seed`` first so graph work overlaps)."""
    run = ctx.run_id
    kw = dict(scope="project", project_id=ctx.project_id, visibility="private")
    rare = f"Zx{run.capitalize()}Rare"
    dense = write_raw(
        a,
        f"zq-{run}-rd The Quokkaforge toolchain compiles our artefacts every night on the staging cluster. {rare}",
        "tech_stack", marker=f"zq-{run}-rd", **kw,
    )
    premise = write_raw(
        a, f"zq-{run}-premise The ferry timetable is printed on blue paper.", "tech_stack",
        marker=f"zq-{run}-premise", **kw,
    )
    derived = Written(f"zq-{run}-derived", "", error="premise write failed")
    if premise.memory_id:
        derived = write_raw(
            a, f"zq-{run}-derived Therefore the ferry office stocks blue paper.", "tech_stack",
            marker=f"zq-{run}-derived", derived_from=[premise.memory_id], epistemic_level="deductive", **kw,
        )
    glob_a = write_raw(
        a, f"zq-{run}-globa My preferred note-taking tint is a pale vermilion.", "preference",
        marker=f"zq-{run}-globa", scope="global", visibility="private",
    )
    ask_tool = f"zq-{run}-quasarforge"
    ask = write_raw(
        a, f"{ask_tool} is the release tool that the vermiculite team uses every Thursday.", "tech_stack",
        marker=ask_tool, **kw,
    )
    return {
        "rare": rare, "dense": dense, "premise": premise, "derived": derived,
        "glob_a": glob_a, "ask": ask, "ask_tool": ask_tool,
    }


@pytest.fixture(scope="module")
def glob_b(ctx, b) -> Written:
    return write_raw(
        b, f"zq-{ctx.run_id}-globb My preferred note-taking tint is a pale cerulean.", "preference",
        marker=f"zq-{ctx.run_id}-globb", scope="global", visibility="private",
    )


# ── the real checks ──────────────────────────────────────────────


def check_search_finds(client: NSClient, marker: str, query: str, project_id: str | None, **extra) -> None:
    results = client.search(query, project_id=project_id, limit=20, **extra)
    assert any(marker in r.get("memory", "") for r in results), f"{marker} not in {len(results)} results for {query!r}"


def check_batch_get(client: NSClient, memory_id: str, marker: str) -> None:
    r = client.post("/v1/memories/batch-get", json={"ids": [memory_id]})
    assert r.status_code == 200, f"batch-get -> {r.status_code}: {r.text[:160]}"
    body = r.json()
    got = [m["id"] for m in body["results"]]
    assert got == [memory_id], f"results {got}, missing {body['missing']}"
    assert marker in body["results"][0]["memory"], "batch-get row has no body text"


def check_chain(client: NSClient, derived_id: str, premise_id: str) -> None:
    r = client.get(f"/v1/memories/{derived_id}/reasoning_chain")
    assert r.status_code == 200, f"reasoning_chain -> {r.status_code}: {r.text[:160]}"
    chain = r.json()["chain"]
    assert chain["memory_id"] == derived_id and chain["epistemic_level"] == "deductive", f"root: {chain}"
    kids = {c["memory_id"]: c for c in chain["children"]}
    assert premise_id in kids and "missing" not in kids[premise_id], f"premise not expanded: {chain['children']}"


def check_category_has(client: NSClient, path: str, marker: str, category: str | None = None) -> None:
    r = client.get(path)
    assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:160]}"
    cats = r.json()["categories"]
    rows = cats.get(category, []) if category else [m for ms in cats.values() for m in ms]
    assert any(marker in m.get("memory", "") for m in rows), f"{marker} not in context categories {sorted(cats)}"


def check_queue_status(body: dict, user_id: str) -> None:
    assert body.get("user_id") == user_id, f"queue status for {body.get('user_id')!r}, want {user_id!r}"
    assert body.get("tracked", 0) >= 1, f"queue status tracks no recent task: {body}"
    assert set(body.get("queues", {})) >= {"main", "graph", "ingest"}, f"queues: {body.get('queues')}"
    assert set(body.get("counts", {})) >= {"queued", "processing", "completed", "failed", "expired"}
    assert isinstance(body.get("caught_up"), bool) and body.get("window_seconds", 0) > 0


def okf_texts(client: NSClient, **params) -> tuple[list[str], dict]:
    r = client.get("/v1/export/okf", params=params)
    assert r.status_code == 200, f"GET /v1/export/okf -> {r.status_code}: {r.text[:160]}"
    assert r.headers.get("content-type", "").startswith("application/zip"), r.headers.get("content-type")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        texts = [zf.read(n).decode("utf-8", "replace") for n in zf.namelist() if n.endswith(".md")]
    return texts, dict(r.headers)


def check_okf_has(client: NSClient, marker: str, project_id: str) -> None:
    texts, _ = okf_texts(client, project_id=project_id)
    assert any(marker in t for t in texts), f"{marker} not in the OKF bundle ({len(texts)} documents)"
    assert any("okf_version" in t for t in texts), "bundle has no okf_version marker"


def check_ask_answers(client: NSClient, question: str, memory_id: str, marker: str, level: str = "minimal") -> dict:
    r = client.post("/v1/ask", json={"question": question, "reasoning_level": level}, timeout=180)
    assert r.status_code == 200, f"POST /v1/ask -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["reasoning_level"] == level and isinstance(body["answer"], str) and body["answer"]
    assert body["memories_considered"] >= 1, f"ask retrieved no evidence: {body}"
    assert marker in body["answer"] or memory_id in body["citations"], (
        f"neither the answer nor the citations point at the fact: {body['answer'][:160]!r} {body['citations']}"
    )
    return body


def meter_snapshot(client: NSClient) -> dict:
    r = client.get("/v1/metrics")
    assert r.status_code == 200, f"GET /v1/metrics -> {r.status_code}"
    return r.json()["savings_meter"]


def check_meter_belongs_to(client: NSClient, user_id: str) -> None:
    snap = meter_snapshot(client)
    assert snap["enabled"] is True, "savings meter disabled"
    assert snap["user"]["user_id"] == user_id, f"meter totals are for {snap['user']['user_id']!r}, want {user_id!r}"


# ── hybrid recall ────────────────────────────────────────────────


LEGS_TEXT = "Quokkaforge compiles artefacts nightly on the staging cluster."
LEGS_QUERY = "which utility assembles our deliverables each evening inside a rehearsal environment"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@pytest.fixture(scope="module")
def legs(ctx, a) -> dict:
    """One fact alone in its own project, and a query sharing NO word with it."""
    marker = f"zq-{ctx.run_id}-legs"
    project = f"{ctx.project_id}-legs"
    w = write_raw(
        a, f"{marker} {LEGS_TEXT}", "tech_stack", marker=marker, scope="project", project_id=project, visibility="private",
    )
    decoys = [
        write_raw(
            a, f"zq-{ctx.run_id}-decoy{i} {text}", "tech_stack", marker=f"zq-{ctx.run_id}-decoy{i}",
            scope="project", project_id=project, visibility="private",
        )
        for i, text in enumerate((
            "The orchard gate is painted green and squeaks in winter.",
            "Grandmother's recipe needs two cups of oat flour and a pinch of salt.",
            "The choir rehearses on Thursdays in the old chapel.",
            "A heron nests near the third bridge every spring.",
        ))
    ]
    return {"w": w, "project": project, "marker": marker, "decoys": decoys}


@proves("read.hybrid-search-core", "read.dense-pool-legs", "fork.mem0.gemini-embedder")
def test_query_with_no_shared_word_recalls_the_fact_through_the_dense_leg(a, legs):
    """The query and the stored text share no token at all (checked here), so the
    lexical (BM25) leg cannot match: only the embedding leg can rank the fact first
    among unrelated decoys in the same project."""
    legs["w"].require()
    for d in legs["decoys"]:
        d.require()
    shared = _words(LEGS_QUERY) & _words(f"{legs['marker']} {LEGS_TEXT}")
    assert not shared, f"test data bug: query and fact share words {sorted(shared)}"
    eventually(
        lambda: check_search_finds(a, legs["marker"], LEGS_QUERY, legs["project"]),
        timeout=90, what="the fact is searchable at all",
    )
    # Ranking is asserted once, outside the retry loop, with a margin.
    results = a.search(LEGS_QUERY, project_id=legs["project"], limit=5)
    assert results and legs["marker"] in results[0].get("memory", ""), (
        f"target is not the top hit: {[r.get('memory', '')[:40] for r in results[:3]]}"
    )
    scores = [r.get("score") for r in results]
    if all(isinstance(x, (int, float)) for x in scores) and len(scores) > 1:
        assert scores[0] > max(scores[1:]), f"target's score has no margin over the decoys: {scores}"
    else:  # no scores in the response: require the same rank-1 result twice in a row
        again = a.search(LEGS_QUERY, project_id=legs["project"], limit=5)
        assert again and again[0].get("id") == results[0].get("id"), "rank 1 is not stable across two searches"


@proves("read.hybrid-search-core")
def test_bare_rare_token_recalls_the_fact(ctx, a, rd):
    rd["dense"].require()
    eventually(
        lambda: check_search_finds(a, rd["dense"].marker, rd["rare"], ctx.project_id),
        timeout=90, what="rare-token recall",
    )


@proves("read.hybrid-search-core", "read.dense-pool-legs")
@negative_control
def test_recall_check_fails_for_user_b(b, legs):
    legs["w"].require()
    expect_check_fails(check_search_finds, b, legs["marker"], LEGS_QUERY, legs["project"])


# ── index-first recall, batch get ────────────────────────────────


@proves("read.index-format", "read.hybrid-search-core")
def test_index_only_search_returns_compact_rows_without_bodies(ctx, a, rd):
    rd["dense"].require()

    def check():
        r = a.post("/v1/search", json={"query": rd["rare"], "project_id": ctx.project_id, "index_only": True, "limit": 20})
        assert r.status_code == 200, f"index_only search -> {r.status_code}: {r.text[:160]}"
        body = r.json()
        assert body.get("index_only") is True
        row = next((x for x in body["results"] if x["id"] == rd["dense"].memory_id), None)
        assert row is not None, f"dense fact not in {len(body['results'])} index rows"
        assert {"id", "title", "category", "glyph", "age", "tokens"} <= set(row), f"index row keys: {sorted(row)}"
        assert "memory" not in row, "index row carries the full body"

    eventually(check, timeout=90, what="index-only recall")


@proves("read.get-memories-batch", "read.index-format")
def test_batch_get_expands_ids_and_reports_unreadable_or_malformed_as_missing(a, b, rd):
    rd["dense"].require()
    check_batch_get(a, rd["dense"].memory_id, rd["dense"].marker)
    ghost, malformed = str(uuid.uuid4()), "not-a-uuid"
    r = a.post("/v1/memories/batch-get", json={"ids": [rd["dense"].memory_id, ghost, malformed]})
    assert r.status_code == 200, f"batch-get with unknown ids -> {r.status_code}"
    assert set(r.json()["missing"]) == {ghost, malformed}, f"missing: {r.json()['missing']}"
    rb = b.post("/v1/memories/batch-get", json={"ids": [rd["dense"].memory_id]})
    assert rb.status_code == 200 and rb.json()["results"] == [], "B's batch-get returned A's private memory"
    assert rb.json()["missing"] == [rd["dense"].memory_id], "unreadable id is not reported as missing"


@proves("read.get-memories-batch")
@negative_control
def test_batch_get_check_fails_for_user_b(b, rd):
    rd["dense"].require()
    expect_check_fails(check_batch_get, b, rd["dense"].memory_id, rd["dense"].marker)


@proves("read.get-memories-batch", "read.index-format", "read.hybrid-search-core")
def test_mcp_index_recall_then_get_memories(ctx, mcp_a, rd):
    rd["dense"].require()
    mcp_a.initialize()

    def check():
        body = mcp_json(mcp_a, "recall_memories", {"query": rd["rare"], "project_id": ctx.project_id, "index_only": True, "limit": 20})
        assert body.get("index_only") is True, f"MCP index_only body: {str(body)[:160]}"
        assert rd["dense"].memory_id in [r["id"] for r in body["results"]], "dense fact not in MCP index rows"

    eventually(check, timeout=90, what="MCP index recall")
    got = mcp_json(mcp_a, "get_memories", {"ids": [rd["dense"].memory_id]})
    assert [m["id"] for m in got["results"]] == [rd["dense"].memory_id] and got["missing"] == [], f"get_memories: {str(got)[:200]}"


# ── timeline ─────────────────────────────────────────────────────


def check_timeline_anchor(client: NSClient, anchor: str, project_id: str) -> dict:
    r = client.post("/v1/timeline", json={"anchor": anchor, "depth": 5, "project_id": project_id})
    assert r.status_code == 200, f"POST /v1/timeline -> {r.status_code}: {r.text[:160]}"
    body = r.json()
    anchors = [row["id"] for row in body["results"] if row.get("anchor")]
    assert anchors == [anchor] and body["anchor_id"] == anchor, f"anchor rows {anchors}, anchor_id {body['anchor_id']}"
    return body


@proves("read.timeline")
def test_rest_timeline_marks_the_anchor_and_refuses_unknown_or_foreign_anchors(ctx, a, b, rd):
    rd["dense"].require()
    check_timeline_anchor(a, rd["dense"].memory_id, ctx.project_id)
    assert a.post("/v1/timeline", json={"anchor": str(uuid.uuid4())}).status_code == 404, "unknown anchor is not a 404"
    assert b.post("/v1/timeline", json={"anchor": rd["dense"].memory_id}).status_code == 404, "B anchored on A's private memory"


@proves("read.timeline")
def test_mcp_timeline_by_id(ctx, mcp_a, rd):
    rd["dense"].require()
    mcp_a.initialize()
    body = mcp_json(mcp_a, "timeline", {"anchor": rd["dense"].memory_id, "depth": 3, "project_id": ctx.project_id})
    assert body.get("anchor_id") == rd["dense"].memory_id, f"MCP timeline: {str(body)[:200]}"
    assert any(row.get("anchor") for row in body["results"]), "no anchor row in MCP timeline"


@proves("read.timeline")
@negative_control
def test_timeline_check_fails_for_user_b(ctx, b, rd):
    rd["dense"].require()
    expect_check_fails(check_timeline_anchor, b, rd["dense"].memory_id, ctx.project_id)


# ── provenance ───────────────────────────────────────────────────


@proves("read.reasoning-chain", "model.provenance-epistemics")
def test_reasoning_chain_walks_derived_from_to_the_premise(a, b, rd):
    rd["derived"].require()
    rd["premise"].require()
    check_chain(a, rd["derived"].memory_id, rd["premise"].memory_id)
    mem = a.get(f"/v1/memories/{rd['derived'].memory_id}").json()
    assert mem["derived_from"] == [rd["premise"].memory_id] and mem["epistemic_level"] == "deductive", f"stored: {mem}"
    assert b.get(f"/v1/memories/{rd['derived'].memory_id}/reasoning_chain").status_code == 404, "B can walk A's private chain"


@proves("read.reasoning-chain")
def test_mcp_reasoning_chain(mcp_a, rd):
    rd["derived"].require()
    mcp_a.initialize()
    out = mcp_json(mcp_a, "get_reasoning_chain", {"memory_id": rd["derived"].memory_id})
    assert out["chain"]["memory_id"] == rd["derived"].memory_id and out["chain"]["children"], f"MCP chain: {str(out)[:200]}"


@proves("read.reasoning-chain", "model.provenance-epistemics")
@negative_control
def test_chain_check_fails_for_user_b(b, rd):
    rd["derived"].require()
    rd["premise"].require()
    expect_check_fails(check_chain, b, rd["derived"].memory_id, rd["premise"].memory_id)


# ── listings and projects ────────────────────────────────────────


@proves("read.list-memories")
def test_list_filters_by_category_scope_project_and_index_fields(ctx, a, b, rd):
    rd["dense"].require()
    rd["glob_a"].require()
    rows = list_rows(a, project_id=ctx.project_id, category="tech_stack")
    assert rd["dense"].memory_id in [r["id"] for r in rows] and all(r["category"] == "tech_stack" for r in rows)
    globals_ = list_rows(a, scope="global", category="preference")
    assert rd["glob_a"].memory_id in [r["id"] for r in globals_] and all(r["scope"] == "global" for r in globals_)
    assert not rows_with(list_rows(a, scope="global"), rd["dense"].marker), "a project fact listed under scope=global"
    idx = [r for r in list_rows(a, project_id=ctx.project_id, fields="index") if r["id"] == rd["dense"].memory_id]
    assert idx and idx[0]["memory"] == "" and idx[0].get("title"), f"index-field row: {idx}"
    assert not rows_with(list_rows(b, project_id=ctx.project_id), rd["dense"].marker), "B lists A's private memory"
    assert len(list_rows(a, limit=1)) == 1, "limit=1 not honoured"


@proves("read.list-memories")
@negative_control
def test_list_check_fails_for_user_b(ctx, b, rd):
    rd["dense"].require()

    def check(client):
        assert rows_with(list_rows(client, project_id=ctx.project_id), rd["dense"].marker), "marker not listed"

    expect_check_fails(check, b)


@proves("read.list-projects")
def test_project_is_listed_for_its_owner_only(ctx, a, b, mcp_a, graph_seed):
    graph_seed.first.require()

    def check(client):
        r = client.get("/v1/projects")
        assert r.status_code == 200, f"GET /v1/projects -> {r.status_code}: {r.text[:160]}"
        assert ctx.project_id in r.json()["projects"], f"{ctx.project_id} not in {r.json()['projects']}"

    eventually(lambda: check(a), timeout=180, interval=5, what="project appears after graph enrichment")
    expect_check_fails(check, b)
    mcp_a.initialize()
    assert ctx.project_id in mcp_json(mcp_a, "list_projects")["projects"], "MCP list_projects misses the project"


# ── queue status ─────────────────────────────────────────────────


@proves("read.queue-status")
def test_queue_status_over_rest_and_mcp(ctx, a, mcp_a, rd):
    rd["dense"].require()
    r = a.get("/v1/queue/status")
    assert r.status_code == 200, f"GET /v1/queue/status -> {r.status_code}"
    check_queue_status(r.json(), ctx.user_a)
    mcp_a.initialize()
    body = mcp_json(mcp_a, "queue_status")
    assert body.get("status") == "ok"
    check_queue_status(body, ctx.user_a)


@proves("read.queue-status")
@negative_control
def test_queue_status_check_fails_for_another_users_expectation(ctx, a):
    expect_check_fails(check_queue_status, a.get("/v1/queue/status").json(), ctx.user_b)


# ── contexts ─────────────────────────────────────────────────────


@proves("read.global-context")
def test_global_context_holds_own_global_facts_only(a, b, rd, glob_b):
    rd["glob_a"].require()
    glob_b.require()
    check_category_has(a, "/v1/context/global", rd["glob_a"].marker, "preference")
    check_category_has(b, "/v1/context/global", glob_b.marker, "preference")
    assert rd["glob_a"].marker not in b.get("/v1/context/global").text, "B's global context leaks A's fact"
    assert glob_b.marker not in a.get("/v1/context/global").text, "A's global context leaks B's fact"


@proves("read.global-context")
@negative_control
def test_global_context_check_fails_for_user_b(b, rd):
    rd["glob_a"].require()
    expect_check_fails(check_category_has, b, "/v1/context/global", rd["glob_a"].marker, "preference")


@proves("read.project-context")
def test_project_context_serves_the_callers_project_facts(ctx, a, rd):
    rd["dense"].require()
    check_category_has(a, f"/v1/context/{ctx.project_id}", rd["dense"].marker, "tech_stack")


@proves("read.project-context")
def test_project_context_pages_newest_first(ctx, a, rd):
    rd["dense"].require()
    rd["premise"].require()
    ids_of = lambda body: [m["id"] for ms in body["categories"].values() for m in ms]  # noqa: E731
    first = a.get(f"/v1/context/{ctx.project_id}", params={"limit": 1, "offset": 0}).json()
    second = a.get(f"/v1/context/{ctx.project_id}", params={"limit": 1, "offset": 1}).json()
    assert first["returned"] == 1 and second["returned"] == 1, f"page sizes {first['returned']}/{second['returned']}"
    assert first["total"] >= 2 and first["has_more"] is True, f"total {first['total']}, has_more {first['has_more']}"
    assert ids_of(first) != ids_of(second), "offset=1 returned the same memory as offset=0"
    everything = a.get(f"/v1/context/{ctx.project_id}").json()
    newest = max((m for ms in everything["categories"].values() for m in ms), key=lambda m: m.get("created_at") or "")
    assert ids_of(first) == [newest["id"]], f"page 1 is {ids_of(first)}, the newest memory is {newest['id']}"


@proves("read.project-context")
def test_project_context_includes_the_teams_shared_project_memories(ctx, b, shared_fact):
    """Atlas: degraded. Session-start context omits teammates' shared memories
    (reads.py filters on the caller's user_id only). Correct: B sees A's shared fact."""
    shared_fact.require()
    check_category_has(b, f"/v1/context/{shared_project(ctx)}", shared_fact.marker)


@proves("read.project-context")
@negative_control
def test_project_context_check_fails_for_a_private_fact(ctx, b, rd):
    rd["dense"].require()
    expect_check_fails(check_category_has, b, f"/v1/context/{ctx.project_id}", rd["dense"].marker)


def inject_text(client: NSClient, **params) -> str:
    r = client.get("/v1/context/inject", params={"max_chars": 32000, **params})
    assert r.status_code == 200, f"GET /v1/context/inject -> {r.status_code}: {r.text[:160]}"
    return r.json()["additionalContext"]


@proves("read.context-inject")
def test_inject_endpoint_returns_markdown_with_the_facts(ctx, a, b, rd):
    rd["dense"].require()
    rd["glob_a"].require()
    text = inject_text(a, project_id=ctx.project_id)
    assert text.startswith("# Neuralscape Memory Context") and rd["dense"].marker in text, f"inject text: {text[:200]!r}"
    g = inject_text(a)
    assert rd["glob_a"].marker in g and rd["dense"].marker not in g, "global inject has the wrong facts"
    assert rd["dense"].marker not in inject_text(b, project_id=ctx.project_id), "B's injected context leaks A's fact"


@proves("read.context-inject")
@negative_control
def test_inject_check_fails_for_user_b(ctx, b, rd):
    rd["dense"].require()

    def check(client):
        assert rd["dense"].marker in inject_text(client, project_id=ctx.project_id), "marker not injected"

    expect_check_fails(check, b)


# ── ask ──────────────────────────────────────────────────────────


@proves("read.ask-tiers")
def test_ask_minimal_answers_from_memory_and_rejects_bad_levels(ctx, a, rd):
    rd["ask"].require()
    eventually(
        lambda: check_search_finds(a, rd["ask_tool"], ASK_Q, ctx.project_id), timeout=90, what="ask fact searchable",
    )
    check_ask_answers(a, ASK_Q, rd["ask"].memory_id, rd["ask_tool"], "minimal")
    bad = a.post("/v1/ask", json={"question": ASK_Q, "reasoning_level": "ultra"})
    assert bad.status_code == 422, f"unknown reasoning_level -> {bad.status_code} (want 422)"


@proves("read.ask-tiers")
def test_ask_medium_runs_the_keyword_scan_pass(a, rd):
    rd["ask"].require()
    body = check_ask_answers(a, ASK_Q, rd["ask"].memory_id, rd["ask_tool"], "medium")
    assert any(s.startswith("keyword:") for s in body["searches"]), f"no keyword pass among {body['searches']}"


@proves("read.ask-tiers")
def test_mcp_ask_memory_minimal(mcp_a, rd):
    rd["ask"].require()
    mcp_a.initialize()
    body = mcp_json(mcp_a, "ask_memory", {"question": ASK_Q, "reasoning_level": "minimal"})
    assert body.get("memories_considered", 0) >= 1 and body.get("reasoning_level") == "minimal", f"MCP ask: {str(body)[:200]}"
    assert rd["ask_tool"] in body["answer"] or rd["ask"].memory_id in body["citations"], f"MCP ask answer: {body['answer'][:160]!r}"


@proves("read.ask-tiers")
@negative_control
def test_ask_check_fails_for_user_b(b, rd):
    rd["ask"].require()
    expect_check_fails(check_ask_answers, b, ASK_Q, rd["ask"].memory_id, rd["ask_tool"], "minimal")


# ── savings meter ────────────────────────────────────────────────


@proves("read.savings-meter", "ops.telemetry-executor", "queue.telemetry-executor")
def test_metrics_count_this_users_recall_events(ctx, a, b, rd):
    rd["dense"].require()
    check_meter_belongs_to(a, ctx.user_a)
    before = meter_snapshot(a)
    assert before["tool_schema_overhead_tokens"] > 0 and before["lifecycle_stages"], f"meter labels: {str(before)[:200]}"
    a.post("/v1/search", json={"query": rd["rare"], "project_id": ctx.project_id, "index_only": True})

    def grew():
        assert meter_snapshot(a)["user"]["events"] > before["user"]["events"], "no new savings event recorded"

    eventually(grew, timeout=60, what="savings event recorded")
    check_meter_belongs_to(b, ctx.user_b)


@proves("read.savings-meter")
@negative_control
def test_metrics_check_fails_for_the_wrong_user(ctx, a):
    expect_check_fails(check_meter_belongs_to, a, ctx.user_b)


# ── OKF export ───────────────────────────────────────────────────


@proves("read.okf-export", "ingest.okf-export")
def test_okf_export_contains_own_facts_and_never_anothers(ctx, a, b, rd):
    rd["dense"].require()
    check_okf_has(a, rd["dense"].marker, ctx.project_id)
    texts, headers = okf_texts(b, project_id=ctx.project_id)
    assert not any(rd["dense"].marker in t for t in texts), "B's OKF bundle contains A's private memory"
    assert "x-okf-concepts" in headers and "x-okf-files" in headers, f"export headers: {sorted(headers)}"


@proves("read.okf-export", "ingest.okf-export")
@negative_control
def test_okf_check_fails_for_user_b(ctx, b, rd):
    rd["dense"].require()
    expect_check_fails(check_okf_has, b, rd["dense"].marker, ctx.project_id)


# ── single read + legacy routes ──────────────────────────────────


@proves("read.single-memory-get")
def test_get_memory_by_id_is_4xx_for_malformed_unknown_and_foreign_ids(a, b, rd):
    rd["dense"].require()
    assert a.get(f"/v1/memories/{rd['dense'].memory_id}").status_code == 200
    assert a.get(f"/v1/memories/{uuid.uuid4()}").status_code == 404, "unknown id is not a 404"
    malformed = a.get("/v1/memories/not-a-uuid")
    assert malformed.status_code in (400, 404), f"malformed id -> {malformed.status_code} (want 400/404, not a 500)"
    assert b.get(f"/v1/memories/{rd['dense'].memory_id}").status_code == 404, "B reads A's private memory by id"


@proves("read.single-memory-get")
@negative_control
def test_single_get_check_fails_for_user_b(b, rd):
    rd["dense"].require()

    def check(client):
        assert client.get(f"/v1/memories/{rd['dense'].memory_id}").status_code == 200, "not readable"

    expect_check_fails(check, b)


@proves("read.legacy-rest-reads", "fork.mem0.v2-kwarg-api")
def test_legacy_list_and_search_routes_answer_with_the_callers_memories(ctx, a, rd):
    """Atlas: broken (issue #430): the legacy routes call mem0 with keyword
    arguments the fork no longer accepts and answer 500."""
    rd["dense"].require()
    listed = a.get("/memories", params={"limit": 100})
    assert listed.status_code == 200, f"GET /memories -> {listed.status_code}"
    searched = a.post("/search", json={"query": rd["dense"].marker, "limit": 10})
    assert searched.status_code == 200, f"POST /search -> {searched.status_code}"
    assert rd["dense"].marker in searched.text, "legacy search does not return the caller's memory"


@proves("read.legacy-rest-reads")
def test_legacy_graph_routes_answer_while_the_service_graph_is_healthy(ctx, a):
    """Atlas: degraded (issue #432): the legacy graph routes use a second handle and 503."""
    for path in ("/graph/nodes", "/graph/edges", "/graph/episodes", "/graph/communities"):
        r = a.get(path, params={"limit": 5})
        assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:120]}"
    r = a.post("/graph/search", json={"query": "probe", "limit": 3})
    assert r.status_code == 200, f"POST /graph/search -> {r.status_code}: {r.text[:120]}"


@proves("read.legacy-rest-reads")
@negative_control
def test_legacy_check_fails_for_a_route_that_does_not_exist(a):
    def check():
        r = a.get("/memories-that-do-not-exist")
        assert r.status_code == 200, f"-> {r.status_code}"

    expect_check_fails(check)


# ── surface inventories ──────────────────────────────────────────


def _resolve(path: str, ctx, rand: str, foreign: str) -> str:
    return path.replace("{rand}", rand).replace("{project}", ctx.project_id).replace("{foreign_memory}", foreign)


@proves("read.rest-v1-surface", "ops.health-endpoints")
def test_every_v1_route_is_mounted_and_answers_without_a_server_error(ctx, anon, b, shared_fact):
    shared_fact.require()  # a memory id owned by A, so B's DELETE on it is refused, never executed
    failures = []
    for method, path, kw in PROBES:
        client = anon if path.startswith("/health") else b
        probe = probe_route(client, method, _resolve(path, ctx, str(uuid.uuid4()), shared_fact.memory_id), **kw)
        if not probe.present:
            failures.append(probe.line())
    assert not failures, "routes missing or erroring:\n  " + "\n  ".join(failures)


def _shape(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


@proves("read.rest-v1-surface")
def test_every_route_the_stack_declares_is_in_the_probe_table(a):
    r = a.get("/openapi.json")
    if r.status_code == 404:
        pytest.skip("stack does not serve /openapi.json")
    assert r.status_code == 200, f"GET /openapi.json -> {r.status_code}"
    declared = {
        (m.upper(), _shape(p))
        for p, ops in r.json()["paths"].items()
        if (p.startswith("/v1") or p.startswith("/health")) and not p.startswith(EXCLUDED_PREFIXES)
        for m in ops
        if m.lower() in {"get", "post", "put", "patch", "delete"}
    }
    covered = {(m, _shape(p)) for m, p, _ in PROBES} | {(m, _shape(p)) for m, p in COVERED_ELSEWHERE}
    missing = sorted(f"{m} {p}" for m, p in declared - covered)
    assert not missing, f"routes the stack declares but the sweep never probes: {missing}"


@proves("read.rest-v1-surface")
@negative_control
def test_route_probe_flags_a_route_that_does_not_exist(b):
    ghost = probe_route(b, "GET", f"/v1/zq-no-such-route-{uuid.uuid4().hex[:8]}")
    assert ghost.present is False, "the probe accepted a non-existent route"
    wrong_verb = probe_route(b, "DELETE", "/v1/categories")
    assert wrong_verb.present is False, "the probe accepted a verb the route does not serve"


@proves("read.mcp-tool-surface")
def test_mcp_tool_list_is_the_documented_surface(mcp_a):
    mcp_a.initialize()
    tools = mcp_a.list_tools()
    names = {t["name"] for t in tools}
    assert MCP_CORE_TOOLS <= names, f"missing core tools: {sorted(MCP_CORE_TOOLS - names)}"
    extras = names - MCP_CORE_TOOLS - MCP_CODE_TOOLS
    assert not extras, f"undocumented MCP tools: {sorted(extras)}"
    code = names & MCP_CODE_TOOLS
    assert code in (set(), MCP_CODE_TOOLS), f"partial code-graph tool set: {sorted(code)}"
    for t in tools:
        assert t.get("description") and t.get("inputSchema", {}).get("type") == "object", (
            f"tool {t['name']} lacks a description or object inputSchema"
        )


@proves("read.mcp-tool-surface")
@negative_control
def test_mcp_surface_check_fails_when_a_core_tool_is_dropped(mcp_a):
    mcp_a.initialize()
    names = {t["name"] for t in mcp_a.list_tools()} - {"recall_memories"}

    def check():
        assert MCP_CORE_TOOLS <= names, f"missing core tools: {sorted(MCP_CORE_TOOLS - names)}"

    expect_check_fails(check)
