"""Edit and delete: PATCH, retag, single and bulk delete, graph cleanup, ownership.

Runs near the end of the session (see ``RUN_LAST`` in conftest.py) because the
conversation-cascade scenario deletes one of the session's shared conversation
facts. Writes are LLM-free raw writes, once per module.

Documented behaviour: a private memory can be edited or deleted only by its
owner; deleting a memory removes it from vector search, listings and by-id
reads, and expires its graph episode and edges.
"""

from __future__ import annotations

import uuid

import pytest

from conftest import ConvSeed
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import (
    blob_has, graph_listing, list_rows, mcp_json, rows_with, write_raw,
)

GRAPH_WAIT = 180


# ── module fixtures ──────────────────────────────────────────────


def _kw(ctx, project=None):
    return dict(scope="project", project_id=project or ctx.project_id, visibility="private")


@pytest.fixture(scope="module")
def ed(ctx, a) -> dict:
    run = ctx.run_id
    ent = lambda tag: f"Zq{run.capitalize()}{tag}"  # noqa: E731
    out: dict = {"retag": f"zq-{run}-retag", "bulk": f"{ctx.project_id}-bulk", "bulk2": f"{ctx.project_id}-bulk2"}
    out["patch"] = write_raw(
        a, f"zq-{run}-old The deploy window is Tuesday at noon.", "tech_stack", marker=f"zq-{run}-old",
        tags=["acc-edit"], **_kw(ctx),
    )
    out["new_marker"] = f"zq-{run}-new"
    out["meta"] = write_raw(
        a, f"zq-{run}-meta The ferry horn is pitched in B flat.", "tech_stack", marker=f"zq-{run}-meta",
        tags=["acc-meta"], **_kw(ctx),
    )
    for n in ("a", "b"):
        out[f"retag_{n}"] = write_raw(
            a, f"zq-{run}-retag-{n} Fixture {n}: the weather vane faces east.", "tech_stack",
            marker=f"zq-{run}-retag-{n}", tags=[out["retag"]], **_kw(ctx),
        )
    out["ent_del"] = ent("Del")
    out["del_rest"] = write_raw(
        a, f"zq-{run}-delrest {out['ent_del']} is a service that runs on the {ent('Host')} cluster.",
        "tech_stack", marker=f"zq-{run}-delrest", **_kw(ctx),
    )
    out["del_mcp"] = write_raw(
        a, f"zq-{run}-delmcp The kite festival is held on the north pier.", "tech_stack",
        marker=f"zq-{run}-delmcp", **_kw(ctx),
    )
    for n in (1, 2):
        out[f"bulk_{n}"] = write_raw(
            a, f"zq-{run}-bulk{n} Bulk fixture {n}: lantern oil is stored in the cellar.", "tech_stack",
            marker=f"zq-{run}-bulk{n}", **_kw(ctx, out["bulk"]),
        )
    out["bulk_3"] = write_raw(
        a, f"zq-{run}-bulk3 Bulk fixture 3: the harbour chart hangs by the door.", "tech_stack",
        marker=f"zq-{run}-bulk3", **_kw(ctx, out["bulk2"]),
    )
    out["foreign"] = write_raw(
        a, f"zq-{run}-foreign Only user A may change this: the tide table lives in the loft.", "tech_stack",
        marker=f"zq-{run}-foreign", **_kw(ctx),
    )
    return out


# ── the real checks ──────────────────────────────────────────────


def get_row(client: NSClient, memory_id: str) -> dict:
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code == 200, f"GET /v1/memories/<id> -> {r.status_code}: {r.text[:120]}"
    return r.json()


def check_gone(client: NSClient, memory_id: str, marker: str, project_id: str) -> None:
    """The memory is unreadable by id, absent from the listing and from recall."""
    r = client.get(f"/v1/memories/{memory_id}")
    assert r.status_code == 404, f"GET by id -> {r.status_code} (want 404)"
    assert not rows_with(list_rows(client, project_id=project_id), marker), "still listed"
    hits = client.search(marker, project_id=project_id, limit=20)
    assert not any(marker in h.get("memory", "") for h in hits), "still recalled by search"


def check_content_replaced(client: NSClient, memory_id: str, old: str, new: str, project_id: str) -> None:
    assert new in get_row(client, memory_id)["memory"], "by-id read does not show the new content"
    hits = client.search(new, project_id=project_id, limit=20)
    assert any(new in h.get("memory", "") for h in hits), "search does not find the new content"
    assert not any(old in h.get("memory", "") for h in client.search(old, project_id=project_id, limit=20)), (
        "search still returns the old content"
    )


def check_episode_present(client: NSClient, project_id: str, marker: str) -> None:
    eps = graph_listing(client, "episodes", project_id)
    assert any(marker in (e.get("content") or "") for e in eps), f"no episode with {marker}"


def check_episode_absent(client: NSClient, project_id: str, marker: str) -> None:
    eps = graph_listing(client, "episodes", project_id)
    assert not any(marker in (e.get("content") or "") for e in eps), f"episode with {marker} still live"


# ── PATCH ────────────────────────────────────────────────────────


@proves("edit.patch-memory", "model.v2-fields")
def test_patch_content_replaces_recall_and_requeues_the_graph(ctx, a, ed):
    p = ed["patch"].require()
    old, new = p.marker, ed["new_marker"]
    before = get_row(a, p.memory_id)
    r = a.request(
        "PATCH", f"/v1/memories/{p.memory_id}",
        json={"content": f"{new} The deploy window is Wednesday at dawn.", "tags": ["acc-edited", "acc-two"]},
    )
    assert r.status_code == 200, f"PATCH -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body["status"] == "ok" and new in body["memory"]["memory"], f"PATCH body: {str(body)[:200]}"
    assert body["graph"] == "reingest_queued" and body["graph_task_id"], f"graph bookkeeping: {body['graph']!r} {body['graph_task_id']!r}"
    after = get_row(a, p.memory_id)
    assert after["id"] == p.memory_id and after["created_at"] == before["created_at"], "edit changed id/created_at"
    assert after["owner_user_id"] == before["owner_user_id"] and after["tags"] == ["acc-edited", "acc-two"], f"after: {after}"
    eventually(lambda: check_content_replaced(a, p.memory_id, old, new, ctx.project_id), timeout=90, what="vector recall swaps content")
    graph_task = a.poll_task(body["graph_task_id"], timeout=GRAPH_WAIT)
    assert graph_task.get("status") == "completed", f"graph re-ingest task ended as {graph_task}"
    eventually(
        lambda: check_episode_present(a, ctx.project_id, new), timeout=GRAPH_WAIT, interval=5, what="new content reaches the graph",
    )


@proves("edit.patch-memory")
@negative_control
def test_content_replaced_check_fails_for_an_unedited_memory(ctx, a, ed):
    f = ed["foreign"].require()
    expect_check_fails(check_content_replaced, a, f.memory_id, f.marker, ed["new_marker"], ctx.project_id)


@proves("edit.patch-memory")
def test_patch_rejects_empty_and_invalid_edits(ctx, a, ed):
    f = ed["foreign"].require()
    assert a.request("PATCH", f"/v1/memories/{f.memory_id}", json={}).status_code == 400, "empty PATCH accepted"
    bad = a.request("PATCH", f"/v1/memories/{f.memory_id}", json={"category": f"zq-{ctx.run_id}-nonsense"})
    assert bad.status_code == 400, f"PATCH with an invalid category -> {bad.status_code}"
    assert a.request("PATCH", f"/v1/memories/{uuid.uuid4()}", json={"tags": ["x"]}).status_code == 404, "PATCH of an unknown id is not a 404"
    assert f.marker in get_row(a, f.memory_id)["memory"], "a rejected edit changed the memory"


@proves("edit.patch-memory")
def test_mcp_edit_memory_updates_metadata_in_place(ed, mcp_a, a):
    m = ed["meta"].require()
    mcp_a.initialize()
    out = mcp_json(mcp_a, "edit_memory", {"memory_id": m.memory_id, "tags": ["acc-mcp-edited"], "domain": "ops"})
    assert out.get("status") == "ok" and out["memory"]["tags"] == ["acc-mcp-edited"], f"edit_memory: {str(out)[:200]}"
    row = get_row(a, m.memory_id)
    assert row["tags"] == ["acc-mcp-edited"] and row["domain"] == "ops" and m.marker in row["memory"], f"row after edit: {row}"
    assert out["graph"] == "unchanged", "a tags/domain edit must not touch the graph"


@proves("edit.patch-memory", "model.visibility-tiers")
def test_other_user_cannot_edit_or_delete_a_private_memory(ctx, a, b, ed):
    f = ed["foreign"].require()
    hijack = b.request("PATCH", f"/v1/memories/{f.memory_id}", json={"content": "hijacked"})
    assert hijack.status_code in (403, 404), f"B's PATCH -> {hijack.status_code}: {hijack.text[:120]}"
    wipe = b.delete(f"/v1/memories/{f.memory_id}")
    assert wipe.status_code in (403, 404), f"B's DELETE -> {wipe.status_code}: {wipe.text[:120]}"
    bulk = b.request("DELETE", "/v1/memories", json={"project_id": ctx.project_id})
    assert bulk.status_code == 200, f"B's bulk delete in A's project -> {bulk.status_code}"
    row = get_row(a, f.memory_id)
    assert f.marker in row["memory"] and "hijacked" not in row["memory"], "A's memory was changed or removed by B"


@proves("edit.patch-memory", "delete.single")
@negative_control
def test_ownership_check_fails_for_the_owner(ctx, a, ed):
    f = ed["foreign"].require()

    def check_refused(client):
        r = client.request("PATCH", f"/v1/memories/{f.memory_id}", json={"tags": ["acc-owner-edit"]})
        assert r.status_code in (403, 404), f"PATCH -> {r.status_code}"

    expect_check_fails(check_refused, a)


# ── retag ────────────────────────────────────────────────────────


def retag_body(ctx, tag: str, **ops) -> dict:
    return {"filters": {"tags_contains": [tag], "project_id": ctx.project_id}, "ops": ops}


@proves("edit.retag")
def test_rest_retag_dry_run_then_apply_changes_category_and_tags(ctx, a, b, ed):
    ra, rb = ed["retag_a"].require(), ed["retag_b"].require()
    ops = dict(add_tags=[ed["retag"] + "-new"], set_category="convention")
    dry = a.post("/v1/memories/retag", json={**retag_body(ctx, ed["retag"], **ops), "dry_run": True})
    assert dry.status_code == 200 and dry.json()["matched"] == 2, f"dry run: {dry.status_code} {dry.text[:160]}"
    assert get_row(a, ra.memory_id)["category"] == "tech_stack", "a dry run changed a memory"
    other = b.post("/v1/memories/retag", json={**retag_body(ctx, ed["retag"], **ops), "dry_run": True})
    assert other.status_code == 200 and other.json()["matched"] == 0, f"B's dry run matched A's memories: {other.text[:160]}"
    r = a.post("/v1/memories/retag", json=retag_body(ctx, ed["retag"], **ops))
    assert r.status_code == 202, f"retag -> {r.status_code}: {r.text[:160]}"
    done = a.poll_task(r.json()["task_id"], timeout=120)
    assert done.get("status") == "completed" and done["result"]["updated"] == 2, f"retag task: {done}"
    for w in (ra, rb):
        row = get_row(a, w.memory_id)
        assert row["category"] == "convention", f"category {row['category']!r}"
        assert set(row["tags"]) == {ed["retag"], ed["retag"] + "-new"}, f"tags {row['tags']}"


@proves("edit.retag")
def test_mcp_retag_memories_adds_a_tag(ctx, a, mcp_a, ed):
    rb = ed["retag_b"].require()
    mcp_a.initialize()
    mcp_tag = ed["retag"] + "-mcp"
    out = mcp_json(mcp_a, "retag_memories", {
        "filters": {"tags_contains": [ed["retag"]], "project_id": ctx.project_id},
        "ops": {"add_tags": [mcp_tag]},
    })
    assert out.get("status") == "accepted" and out.get("task_id"), f"MCP retag: {out}"
    done = a.poll_task(out["task_id"], timeout=120)
    assert done.get("status") == "completed", f"MCP retag task: {done}"
    assert mcp_tag in get_row(a, rb.memory_id)["tags"], f"tags after MCP retag: {get_row(a, rb.memory_id)['tags']}"


def check_retag_refused(client: NSClient, body: dict) -> None:
    r = client.post("/v1/memories/retag", json=body)
    assert r.status_code == 422, f"retag -> {r.status_code} (want 422): {r.text[:120]}"


@proves("edit.retag")
def test_retag_refuses_an_unfiltered_sweep_and_an_empty_operation(ctx, a):
    check_retag_refused(a, {"filters": {}, "ops": {"add_tags": ["x"]}})
    check_retag_refused(a, {"filters": {"project_id": ctx.project_id}, "ops": {}})


@proves("edit.retag")
@negative_control
def test_retag_refusal_check_fails_for_a_valid_request(ctx, a):
    valid = {"filters": {"project_id": ctx.project_id}, "ops": {"add_tags": ["x"]}, "dry_run": True}
    expect_check_fails(check_retag_refused, a, valid)


# ── delete: single ───────────────────────────────────────────────


@proves("delete.single", "model.bitemporal-liveness", "graph.graph-admin")
def test_deleting_one_memory_removes_it_everywhere_including_the_graph(ctx, a, ed):
    w = ed["del_rest"].require()
    eventually(
        lambda: check_episode_present(a, ctx.project_id, w.marker),
        timeout=GRAPH_WAIT, interval=5, what="the memory's graph episode exists before the delete",
    )
    r = a.delete(f"/v1/memories/{w.memory_id}")
    assert r.status_code == 200, f"DELETE -> {r.status_code}: {r.text[:160]}"
    eventually(lambda: check_gone(a, w.memory_id, w.marker, ctx.project_id), timeout=60, what="memory gone from vector store")
    eventually(
        lambda: check_episode_absent(a, ctx.project_id, w.marker), timeout=90, interval=3, what="graph episode removed",
    )
    assert not blob_has(graph_listing(a, "edges", ctx.project_id), ed["ent_del"]), (
        "a live graph edge still mentions the deleted memory's entity"
    )


@proves("delete.single")
def test_mcp_delete_memories_by_id(ctx, a, mcp_a, ed):
    w = ed["del_mcp"].require()
    mcp_a.initialize()
    out = mcp_json(mcp_a, "delete_memories", {"memory_id": w.memory_id})
    assert "error" not in out, f"MCP delete: {out}"
    eventually(lambda: check_gone(a, w.memory_id, w.marker, ctx.project_id), timeout=60, what="MCP-deleted memory gone")


@proves("delete.single")
def test_deleting_an_unknown_or_malformed_id_is_never_a_server_error(a):
    for target in (str(uuid.uuid4()), "not-a-uuid"):
        r = a.delete(f"/v1/memories/{target}")
        assert r.status_code < 500, f"DELETE /v1/memories/{target} -> {r.status_code}: {r.text[:160]}"


@proves("delete.single")
@negative_control
def test_gone_check_fails_for_a_live_memory(ctx, a, ed):
    w = ed["foreign"].require()
    expect_check_fails(check_gone, a, w.memory_id, w.marker, ctx.project_id)


# ── delete: bulk ─────────────────────────────────────────────────


@proves("delete.bulk")
def test_rest_bulk_delete_by_project_removes_exactly_that_project(ctx, a, ed):
    b1, b2 = ed["bulk_1"].require(), ed["bulk_2"].require()
    r = a.request("DELETE", "/v1/memories", json={"project_id": ed["bulk"]})
    assert r.status_code == 200, f"bulk DELETE -> {r.status_code}: {r.text[:160]}"
    for w in (b1, b2):
        eventually(lambda w=w: check_gone(a, w.memory_id, w.marker, ed["bulk"]), timeout=60, what=f"{w.marker} gone")
    assert get_row(a, ed["foreign"].require().memory_id), "bulk delete of one project removed another project's memory"


@proves("delete.bulk")
def test_mcp_bulk_delete_by_project(a, mcp_a, ed):
    w = ed["bulk_3"].require()
    mcp_a.initialize()
    out = mcp_json(mcp_a, "delete_memories", {"project_id": ed["bulk2"]})
    assert "error" not in out, f"MCP bulk delete: {out}"
    eventually(lambda: check_gone(a, w.memory_id, w.marker, ed["bulk2"]), timeout=60, what="MCP bulk-deleted memory gone")


@proves("delete.bulk")
@negative_control
def test_bulk_gone_check_fails_for_a_project_that_was_not_deleted(ctx, a, ed):
    w = ed["foreign"].require()
    expect_check_fails(check_gone, a, w.memory_id, w.marker, ctx.project_id)


# ── conversation-episode cascade (atlas: graph.episode-cascade) ──


@proves("graph.episode-cascade", "delete.single")
def test_deleting_one_extracted_fact_keeps_its_siblings_graph_evidence(ctx, a, conv_seed: ConvSeed):
    """One conversation is ONE graph episode shared by all facts extracted from it.
    Correct: deleting one fact leaves graph evidence for the others. Atlas bug:
    the cascade expires the whole episode, so the siblings lose their graph
    evidence while their vector rows remain. Red until write-02 lands."""
    assert conv_seed.write.completed, f"conversation write ended as {conv_seed.write.status or conv_seed.write.error}"
    rows = list_rows(a, project_id=conv_seed.project_id)
    forge_rows, lagoon_rows = rows_with(rows, conv_seed.forge), rows_with(rows, conv_seed.lagoon)
    if not forge_rows or not lagoon_rows or forge_rows[0]["id"] == lagoon_rows[0]["id"]:
        pytest.skip("extraction did not yield two separate facts for the two invented names")
    eventually(
        lambda: check_episode_present(a, conv_seed.project_id, conv_seed.lagoon),
        timeout=GRAPH_WAIT, interval=5, what="the conversation's shared episode exists before the delete",
    )
    assert a.delete(f"/v1/memories/{forge_rows[0]['id']}").status_code == 200, "deleting the first fact failed"
    assert get_row(a, lagoon_rows[0]["id"])["id"] == lagoon_rows[0]["id"], "the sibling fact's vector row was lost"

    def sibling_evidence_alive():
        eps = graph_listing(a, "episodes", conv_seed.project_id)
        edges = graph_listing(a, "edges", conv_seed.project_id)
        alive = any(conv_seed.lagoon in (e.get("content") or "") for e in eps) or blob_has(edges, conv_seed.lagoon)
        assert alive, "the sibling fact has no live graph episode or edge after its sibling was deleted"

    eventually(sibling_evidence_alive, timeout=120, interval=5, what="sibling graph evidence")


@proves("graph.episode-cascade")
@negative_control
def test_sibling_evidence_check_fails_for_an_unwritten_name(ctx, a):
    def check():
        eps = graph_listing(a, "episodes", ctx.project_id)
        assert any(f"zq-{ctx.run_id}-never-written" in (e.get("content") or "") for e in eps), "no such episode"

    expect_check_fails(check)
