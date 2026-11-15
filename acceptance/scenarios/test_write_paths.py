"""Write paths: raw store, v2 fields, visibility tiers, dedup, batch, checkpoint,
taxonomy, sensitivity gate, conversation extraction, operator instructions.

Everything except the conversation write is LLM-free (/v1/memories/raw). Facts
are written once per module fixture and asserted many times.
"""

from __future__ import annotations

import json

import pytest

from conftest import ConvSeed, shared_project
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import Written, list_rows, mcp_json, memory_ids_in, rows_with, write_raw

CORE_CATEGORIES = {
    "preference", "personal_fact", "technical_skill", "domain_knowledge", "tech_stack",
    "convention", "architecture", "dependency", "decision", "interaction", "workflow",
    "procedure", "task_context",
}


def get_memory(client: NSClient, memory_id: str) -> dict:
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code == 200, f"GET /v1/memories/<id> -> {r.status_code}: {r.text[:160]}"
    return r.json()


# ── module fixtures: one write each, many assertions ──────────────


@pytest.fixture(scope="module")
def v2(ctx, a) -> Written:
    marker = f"zq-{ctx.run_id}-v2"
    return write_raw(
        a,
        f"{marker} We settled on the Fennel runner for nightly jobs because it restarts cleanly.",
        "decision",
        marker=marker,
        scope="project",
        project_id=ctx.project_id,
        visibility="private",
        tags=["acc-t1", "acc-t2"],
        domain="coding",
        observation_type="decision",
        concepts=["why-it-exists"],
        source_type="explicit",
        confidence=0.8,
    )


@pytest.fixture(scope="module")
def dedup(ctx, a) -> tuple[Written, Written]:
    marker = f"zq-{ctx.run_id}-dedup"
    text = f"{marker} The lighthouse keeper logs the tide table every dawn."
    kw = dict(marker=marker, scope="project", project_id=ctx.project_id, visibility="private")
    first = write_raw(a, text, "tech_stack", **kw)
    return first, write_raw(a, text, "tech_stack", **kw)


@pytest.fixture(scope="module")
def other_category(ctx, a, dedup) -> Written:
    """The SAME text as ``dedup`` under another category (a distinct memory)."""
    first, _ = dedup
    return write_raw(
        a, first.content, "convention", marker=first.marker,
        scope="project", project_id=ctx.project_id, visibility="private",
    )


@pytest.fixture(scope="module")
def batch(ctx, a) -> dict:
    markers = [f"zq-{ctx.run_id}-batch{i}" for i in range(3)]
    out: dict = {"markers": markers, "error": "", "status": {}}
    try:
        resp = a.post(
            "/v1/memories/raw/batch",
            json={
                "memories": [
                    {
                        "content": f"{m} Batch item {i}: the lantern festival route passes the old mill.",
                        "category": "tech_stack",
                        "scope": "project",
                        "project_id": ctx.project_id,
                        "visibility": "private",
                    }
                    for i, m in enumerate(markers)
                ]
            },
        )
        assert resp.status_code == 202, f"POST /v1/memories/raw/batch -> {resp.status_code}: {resp.text[:200]}"
        out["task_id"] = resp.json()["task_id"]
        out["status"] = a.poll_task(out["task_id"], timeout=120)
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@pytest.fixture(scope="module")
def checkpoint(ctx, a) -> dict:
    marker = f"zq-{ctx.run_id}-ckpt"
    body = {
        "project_id": ctx.project_id,
        "memories": [
            {
                "content": f"{marker}-one The harbour bell rings twice at low tide.",
                "category": "tech_stack", "scope": "project", "project_id": ctx.project_id,
                "visibility": "private",
            },
            {
                "content": f"{marker}-two The harbour master prefers brass lanterns.",
                "category": "tech_stack", "scope": "project", "project_id": ctx.project_id,
                "visibility": "private",
            },
        ],
        "session_note": {"request": f"{marker}-note investigate the harbour", "completed": "mapped the quay"},
    }
    out: dict = {"marker": marker, "body": body, "error": "", "status": {}}
    try:
        resp = a.post("/v1/checkpoint", json=body)
        assert resp.status_code == 202, f"POST /v1/checkpoint -> {resp.status_code}: {resp.text[:200]}"
        out["first"] = resp.json()
        out["status"] = a.poll_task(out["first"]["task_id"], timeout=120)
        out["ids"] = memory_ids_in(out["status"])
    except (AssertionError, KeyError) as exc:
        out["error"] = str(exc)
    return out


@pytest.fixture(scope="module")
def gated(ctx, a) -> dict[str, Written]:
    """Sensitive text written as shared, with and without the documented bypass."""
    base = f"zq-{ctx.run_id}-gate"
    kw = dict(scope="project", project_id=shared_project(ctx))  # shared writes stay out of the private project
    return {
        "forced": write_raw(
            a, f"{base}-forced The payroll run happens on the first of the month.", "tech_stack",
            marker=f"{base}-forced", visibility="shared", **kw,
        ),
        "bypassed": write_raw(
            a, f"{base}-bypassed The payroll export is a CSV file named after the month.", "tech_stack",
            marker=f"{base}-bypassed", visibility="shared", sensitivity_override=True, **kw,
        ),
        "clean": write_raw(
            a, f"{base}-clean The tide chart is pinned next to the kettle.", "tech_stack",
            marker=f"{base}-clean", visibility="shared", **kw,
        ),
    }


# ── the real checks ──────────────────────────────────────────────


def check_fields_round_trip(client: NSClient, w: Written, expected: dict) -> None:
    mem = get_memory(client, w.memory_id)
    for key, want in expected.items():
        assert mem.get(key) == want, f"{key}: stored {mem.get(key)!r}, wrote {want!r}"
    assert w.marker in mem.get("memory", ""), "content did not round-trip"


V2_EXPECTED = lambda ctx: {  # noqa: E731
    "category": "decision", "scope": "project", "project_id": ctx.project_id,
    "visibility": "private", "tags": ["acc-t1", "acc-t2"], "domain": "coding",
    "observation_type": "decision", "concepts": ["why-it-exists"], "source_type": "explicit",
    "confidence": 0.8,
}


def check_forced_private(client: NSClient, w: Written) -> None:
    """A gated write asked to be shared lands private and says why."""
    mem = get_memory(client, w.memory_id)
    assert mem.get("visibility") == "private", f"visibility is {mem.get('visibility')!r}, not forced private"
    assert mem.get("sensitivity") == "financial", f"sensitivity class is {mem.get('sensitivity')!r}"
    assert mem.get("sensitivity_source") == "regex", f"sensitivity_source is {mem.get('sensitivity_source')!r}"


def check_core_categories(categories: dict) -> None:
    missing = sorted(CORE_CATEGORIES - set(categories))
    assert not missing, f"core categories missing from taxonomy: {missing}"
    assert all(isinstance(v, str) and v for v in categories.values()), "a category has no description"


def check_category_refused(client: NSClient, category: str) -> None:
    r = client.post("/v1/memories/raw", json={"content": "zq-category-probe", "category": category})
    assert r.status_code == 400, f"category {category!r} -> {r.status_code} (want 400): {r.text[:120]}"
    assert "Invalid category" in r.text, f"refusal does not name the problem: {r.text[:120]}"


def check_deduplicated(client: NSClient, project_id: str, first: Written, second: Written) -> None:
    """Two writes of one text are one memory: same id, one listed row."""
    assert second.memory_id == first.memory_id, f"second write made {second.memory_id}, first {first.memory_id}"
    rows = [
        r for r in rows_with(list_rows(client, project_id=project_id), first.marker)
        if r.get("category") == "tech_stack"
    ]
    assert len(rows) == 1, f"{len(rows)} rows carry the text: {[r['id'] for r in rows]}"


def check_verdicts(response: dict, expected: list[str]) -> None:
    got = [v["verdict"] for v in response["verdicts"]]
    assert got == expected, f"checkpoint verdicts {got}, want {expected}"


def check_search_marker(client: NSClient, marker: str, project_id: str, query: str | None = None) -> None:
    results = client.search(query or f"{marker} harbour lantern tide", project_id=project_id, limit=20)
    assert any(marker in r.get("memory", "") for r in results), f"{marker} not in {len(results)} search results"


# ── raw write: fields, tiers, store ──────────────────────────────


@proves(
    "write.remember-raw", "write.memory-service-facade", "model.v2-fields",
    "vec.qdrant-point-store", "fork.mem0.qdrant-store", "read.single-memory-get",
)
def test_raw_write_stores_every_field_it_was_given(ctx, a, v2):
    v2.require()
    check_fields_round_trip(a, v2, V2_EXPECTED(ctx))
    mem = get_memory(a, v2.memory_id)
    assert mem.get("owner_user_id") == ctx.user_a, f"owner_user_id is {mem.get('owner_user_id')!r}"


@proves("write.remember-raw", "model.v2-fields", "vec.qdrant-point-store")
@negative_control
def test_field_round_trip_check_fails_for_a_wrong_expectation(ctx, a, v2):
    v2.require()
    wrong = {**V2_EXPECTED(ctx), "domain": "ops"}
    expect_check_fails(check_fields_round_trip, a, v2, wrong)


@proves("write.remember-raw", "vec.qdrant-point-store", "fork.mem0.qdrant-store")
def test_raw_write_is_listed_and_searchable_by_its_owner(ctx, a, v2):
    v2.require()
    rows = rows_with(list_rows(a, project_id=ctx.project_id), v2.marker)
    assert [r["id"] for r in rows] == [v2.memory_id], f"list shows {[r['id'] for r in rows]}"
    eventually(
        lambda: check_search_marker(a, v2.marker, ctx.project_id, f"{v2.marker} nightly jobs runner"),
        timeout=90, what="A recalls the v2 fact",
    )


@proves("model.visibility-tiers")
def test_private_is_owner_only_and_shared_is_team_readable(ctx, a, b, v2, shared_fact):
    v2.require()
    shared_fact.require()
    assert get_memory(a, shared_fact.memory_id).get("visibility") == "shared"
    # B reads A's shared fact by id, through search, and not A's private one.
    assert get_memory(b, shared_fact.memory_id)["id"] == shared_fact.memory_id
    eventually(
        lambda: check_search_marker(b, shared_fact.marker, shared_project(ctx), f"{shared_fact.marker} retrospective room"),
        timeout=90, what="B recalls A's shared fact",
    )
    assert b.get(f"/v1/memories/{v2.memory_id}").status_code == 404, "B can read A's private memory by id"
    hits = b.search(f"{v2.marker} nightly jobs runner", project_id=ctx.project_id, limit=20)
    assert v2.marker not in json.dumps(hits), "B's search returned A's private memory"


@proves("model.visibility-tiers")
@negative_control
def test_shared_recall_check_fails_for_a_private_fact(ctx, b, v2):
    v2.require()
    expect_check_fails(check_search_marker, b, v2.marker, ctx.project_id, f"{v2.marker} nightly jobs runner")


@proves("model.visibility-tiers")
def test_standard_tier_is_closed_to_ordinary_users(a):
    r = a.post(
        "/v1/memories/raw",
        json={"content": "standard probe", "category": "convention", "visibility": "standard"},
    )
    assert r.status_code == 403, f"standard-tier write by a non-dictator -> {r.status_code}: {r.text[:160]}"


# ── dedup ────────────────────────────────────────────────────────


@proves("write.content-hash-dedup", "write.remember-raw")
def test_identical_raw_content_yields_one_memory(ctx, a, dedup):
    first, second = dedup
    first.require()
    second.require()
    check_deduplicated(a, ctx.project_id, first, second)


@proves("write.content-hash-dedup")
def test_same_text_under_another_category_is_a_distinct_memory(dedup, other_category):
    first, _ = dedup
    first.require()
    other_category.require()
    assert other_category.memory_id != first.memory_id, (
        "re-categorised duplicate was silently absorbed into the old row (dedup key ignores category)"
    )


@proves("write.content-hash-dedup")
@negative_control
def test_dedup_check_fails_for_distinct_content(ctx, a, dedup, v2):
    first, _ = dedup
    first.require()
    v2.require()
    expect_check_fails(check_deduplicated, a, ctx.project_id, first, v2)


# ── batch + checkpoint ───────────────────────────────────────────


@proves("write.batch-store", "write.remember-raw")
def test_batch_write_stores_every_item_and_they_are_recalled(ctx, a, batch):
    assert not batch["error"], batch["error"]
    assert batch["status"].get("status") == "completed", f"batch task ended as {batch['status']}"
    ids = memory_ids_in(batch["status"])
    assert len(ids) == 3, f"batch stored {len(ids)} of 3 items: {ids}"
    for m in batch["markers"]:
        eventually(
            lambda m=m: check_search_marker(a, m, ctx.project_id, f"{m} lantern festival route old mill"),
            timeout=90, what=f"batch item {m} recalled",
        )


@proves("write.batch-store")
@negative_control
def test_batch_recall_check_fails_for_user_b(ctx, b, batch):
    assert not batch["error"], batch["error"]
    m = batch["markers"][0]
    expect_check_fails(check_search_marker, b, m, ctx.project_id, f"{m} lantern festival route old mill")


@proves("write.checkpoint", "write.batch-store")
def test_rest_checkpoint_stores_memories_and_session_note(ctx, a, checkpoint):
    assert not checkpoint["error"], checkpoint["error"]
    assert checkpoint["status"].get("status") == "completed", f"checkpoint task ended as {checkpoint['status']}"
    first = checkpoint["first"]
    check_verdicts(first, ["new", "new"])
    assert first["session_note_included"] is True
    assert first["enqueued"] >= 2, f"enqueued {first['enqueued']} (2 memories + the note)"
    rows = list_rows(a, project_id=ctx.project_id)
    for suffix in ("-one", "-two", "-note"):
        assert rows_with(rows, checkpoint["marker"] + suffix), f"{suffix} not in the memory list"
    note = rows_with(rows, checkpoint["marker"] + "-note")[0]
    assert note.get("category") == "task_context" and note.get("observation_type") == "meeting_outcome", (
        f"session note stored as {note.get('category')}/{note.get('observation_type')}"
    )


@proves("write.checkpoint", "read.timeline")
def test_checkpoint_memories_appear_in_the_timeline(ctx, a, checkpoint):
    assert not checkpoint["error"], checkpoint["error"]
    ids = checkpoint["ids"]
    assert len(ids) >= 2, f"checkpoint result had {len(ids)} memories"
    r = a.post("/v1/timeline", json={"anchor": ids[0], "depth": 10, "project_id": ctx.project_id})
    assert r.status_code == 200, f"POST /v1/timeline -> {r.status_code}: {r.text[:160]}"
    rows = r.json()["results"]
    assert any(row["id"] == ids[0] and row.get("anchor") for row in rows), "anchor row missing from timeline"
    assert ids[1] in [row["id"] for row in rows], "second checkpoint memory not in the timeline window"


@proves("write.checkpoint")
def test_checkpoint_repeat_reports_duplicates_without_enqueuing(a, checkpoint):
    assert not checkpoint["error"], checkpoint["error"]
    body = {"memories": checkpoint["body"]["memories"]}
    r = a.post("/v1/checkpoint", json=body)
    assert r.status_code == 200, f"repeat checkpoint -> {r.status_code}: {r.text[:200]}"
    out = r.json()
    assert out["duplicates"] == 2 and out["enqueued"] == 0, f"repeat checkpoint: {out}"
    check_verdicts(out, ["duplicate", "duplicate"])
    assert {v.get("existing_id") for v in out["verdicts"]} <= set(checkpoint["ids"]), "existing_id names an unknown memory"


@proves("write.checkpoint")
def test_mcp_checkpoint_stores_a_memory(ctx, a, mcp_a):
    mcp_a.initialize()
    marker = f"zq-{ctx.run_id}-mcpckpt"
    out = mcp_json(
        mcp_a, "checkpoint",
        {"memories": [{
            "content": f"{marker} The signal flag for fog is a yellow pennant.",
            "category": "tech_stack", "scope": "project", "project_id": ctx.project_id, "visibility": "private",
        }]},
    )
    assert out.get("status") == "accepted" and out.get("task_id"), f"MCP checkpoint answered {out}"
    status = a.poll_task(out["task_id"], timeout=120)
    assert status.get("status") == "completed", f"MCP checkpoint task ended as {status}"
    assert rows_with(list_rows(a, project_id=ctx.project_id), marker), "MCP checkpoint memory not in the list"


@proves("write.checkpoint")
@negative_control
def test_duplicate_verdict_check_fails_for_the_first_save(checkpoint):
    assert not checkpoint["error"], checkpoint["error"]
    expect_check_fails(check_verdicts, checkpoint["first"], ["duplicate", "duplicate"])


# ── taxonomy ─────────────────────────────────────────────────────


@proves("model.category-taxonomy")
def test_taxonomy_lists_the_thirteen_core_categories(a):
    r = a.get("/v1/categories")
    assert r.status_code == 200, f"GET /v1/categories -> {r.status_code}"
    check_core_categories(r.json()["categories"])


@proves("model.category-taxonomy")
def test_unknown_category_is_refused_at_every_write_boundary(ctx, a):
    check_category_refused(a, f"zq-{ctx.run_id}-nonsense")
    r = a.post(
        "/v1/memories/raw/batch",
        json={"memories": [{"content": "x", "category": f"zq-{ctx.run_id}-nonsense"}]},
    )
    assert r.status_code == 400, f"batch with unknown category -> {r.status_code}"
    r = a.post("/v1/ingest/text", json={"content": "x", "category": f"zq-{ctx.run_id}-nonsense"})
    assert r.status_code == 422, f"ingest/text with unknown category -> {r.status_code} (want 422)"


@proves("model.category-taxonomy")
@negative_control
def test_taxonomy_checks_fail_on_wrong_input(a):
    cats = a.get("/v1/categories").json()["categories"]
    expect_check_fails(check_core_categories, {k: v for k, v in cats.items() if k != "decision"})
    expect_check_fails(check_category_refused, a, "decision")  # a valid category is NOT refused


# ── MCP remember ─────────────────────────────────────────────────


def check_project_scoped(client: NSClient, memory_id: str, project_id: str) -> None:
    mem = get_memory(client, memory_id)
    assert mem.get("scope") == "project" and mem.get("project_id") == project_id, (
        f"stored with scope {mem.get('scope')!r}, project {mem.get('project_id')!r} (asked for project {project_id!r})"
    )


@pytest.fixture(scope="module")
def mcp_domain_fact(ctx, mcp_a) -> dict:
    mcp_a.initialize()
    marker = f"zq-{ctx.run_id}-mcpdk"
    out = mcp_json(mcp_a, "remember", {
        "content": f"{marker} The tide gauge calibration notes live in the north shed.",
        "category": "domain_knowledge", "project_id": ctx.project_id, "visibility": "private", "wait": True,
    })
    ids = memory_ids_in({"result": out.get("result")})
    return {"marker": marker, "reply": out, "id": ids[0] if ids else ""}


@proves("write.remember-raw", "write.memory-service-facade")
def test_mcp_remember_waits_and_stores_the_memory(ctx, a, mcp_domain_fact):
    d = mcp_domain_fact
    assert d["reply"].get("status") == "completed" and d["id"], f"MCP remember(wait): {str(d['reply'])[:240]}"
    assert d["marker"] in get_memory(a, d["id"])["memory"], "MCP-written memory has the wrong content"


@proves("model.category-taxonomy")
def test_domain_knowledge_with_a_project_id_is_stored_project_scoped(ctx, a, mcp_domain_fact):
    """Atlas bug (issue #224): domain_knowledge is in the global-category set, so the
    write path forces scope=global even when a project_id was given and project-only
    reads then miss the row. Correct: an explicit project keeps the memory in it."""
    assert mcp_domain_fact["id"], f"MCP remember failed: {str(mcp_domain_fact['reply'])[:200]}"
    check_project_scoped(a, mcp_domain_fact["id"], ctx.project_id)


@proves("model.category-taxonomy")
@negative_control
def test_project_scope_check_fails_for_a_global_memory(ctx, a):
    glob = write_raw(a, f"zq-{ctx.run_id}-globscope A global preference.", "preference", marker=f"zq-{ctx.run_id}-globscope", scope="global")
    glob.require()
    expect_check_fails(check_project_scoped, a, glob.memory_id, ctx.project_id)


# ── sensitivity gate ─────────────────────────────────────────────


@proves("model.sensitivity-gate", "model.visibility-tiers")
def test_sensitive_content_asked_shared_is_forced_private(ctx, a, b, gated):
    forced = gated["forced"].require()
    check_forced_private(a, forced)
    assert b.get(f"/v1/memories/{forced.memory_id}").status_code == 404, "B can read the gated memory"


@proves("model.sensitivity-gate")
def test_sensitivity_override_with_explicit_visibility_bypasses_the_gate(a, gated):
    bypassed = gated["bypassed"].require()
    mem = get_memory(a, bypassed.memory_id)
    assert mem.get("visibility") == "shared", f"bypass ignored: visibility {mem.get('visibility')!r}"
    assert mem.get("sensitivity_source") == "bypassed", f"sensitivity_source {mem.get('sensitivity_source')!r}"


@proves("model.sensitivity-gate")
@negative_control
def test_forced_private_check_fails_for_non_sensitive_shared_content(a, gated):
    clean = gated["clean"].require()
    expect_check_fails(check_forced_private, a, clean)


# ── conversation extraction (the suite's one LLM write) ──────────


def check_conversation_recalled(client: NSClient, seed: ConvSeed) -> None:
    for query in (f"what is my build tool {seed.forge}", f"staging database {seed.lagoon}"):
        results = client.search(query, project_id=seed.project_id, limit=20)
        text = json.dumps(results)
        if seed.forge in text or seed.lagoon in text:
            return
    raise AssertionError("neither invented fact from the conversation was recalled")


@proves("write.conversation-extraction", "fork.mem0.gemini-embedder")
def test_conversation_write_extracts_facts_that_are_recalled(a, conv_seed):
    assert not conv_seed.write.error, conv_seed.write.error
    assert conv_seed.write.completed, f"conversation task ended as {conv_seed.write.status}"
    assert conv_seed.memory_ids, f"extraction stored no memories: {conv_seed.write.status}"
    eventually(lambda: check_conversation_recalled(a, conv_seed), timeout=90, what="extracted facts recalled")


@proves("write.conversation-extraction")
@negative_control
def test_conversation_recall_check_fails_for_user_b(b, conv_seed):
    assert not conv_seed.write.error, conv_seed.write.error
    expect_check_fails(check_conversation_recalled, b, conv_seed)


# ── operator extraction instructions ─────────────────────────────


def check_instructions(client: NSClient, user_id: str, expected: str | None) -> None:
    r = client.get("/v1/settings/extraction-instructions")
    assert r.status_code == 200, f"GET extraction-instructions -> {r.status_code}: {r.text[:120]}"
    body = r.json()
    assert body["scope"] == "user" and body["target_id"] == user_id, f"scope/target: {body['scope']}/{body['target_id']}"
    assert body["instructions"] == expected, f"stored instructions {body['instructions']!r}, want {expected!r}"


@proves("write.operator-extraction-instructions")
def test_user_extraction_instructions_round_trip_and_clear(ctx, b):
    text = f"zq-{ctx.run_id}-guidance Prefer short factual sentences."
    r = b.request("PUT", "/v1/settings/extraction-instructions", json={"instructions": text})
    assert r.status_code == 200, f"PUT -> {r.status_code}: {r.text[:160]}"
    put = r.json()
    assert put["instructions"] == text and put["tokens"] > 0 and put["updated_by"] == ctx.user_b, f"PUT body: {put}"
    check_instructions(b, ctx.user_b, text)
    cleared = b.request("PUT", "/v1/settings/extraction-instructions", json={"instructions": "  "})
    assert cleared.status_code == 200, f"clearing PUT -> {cleared.status_code}"
    check_instructions(b, ctx.user_b, None)


@proves("write.operator-extraction-instructions")
def test_project_wide_instructions_are_dictator_only(ctx, b):
    r = b.request(
        "PUT", "/v1/settings/extraction-instructions",
        json={"instructions": "project guidance", "project_id": ctx.project_id},
    )
    assert r.status_code == 403, f"project-wide PUT by a non-dictator -> {r.status_code}: {r.text[:160]}"


@proves("write.operator-extraction-instructions")
@negative_control
def test_instructions_check_fails_for_the_wrong_user(ctx, a):
    expect_check_fails(check_instructions, a, ctx.user_b, f"zq-{ctx.run_id}-guidance Prefer short factual sentences.")
