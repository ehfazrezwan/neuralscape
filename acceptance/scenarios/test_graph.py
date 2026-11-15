"""Knowledge graph: raw and conversation writes become episodes, entities and
edges; listings, search and admin; group partitioning and bi-temporal liveness.

Graph enrichment is asynchronous on the graph worker (Graphiti + Gemini, tens
of seconds to minutes each), so effects are awaited with generous bounded
``eventually`` windows (at most 180 s each). The facts were enqueued early by
the ``graph_seed`` session fixture so this module mostly finds them done.
Assertions use invented single-token entity names and episode text, never LLM
wording.
"""

from __future__ import annotations

import uuid

from conftest import ConvSeed, GraphSeed
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import blob_has, graph_listing, mcp_json

GRAPH_WAIT = 180


# ── the real checks ──────────────────────────────────────────────


def check_episode(client: NSClient, project_id: str, marker: str) -> dict:
    episodes = graph_listing(client, "episodes", project_id)
    hits = [e for e in episodes if marker in (e.get("content") or "")]
    assert hits, f"no episode carries {marker} among {len(episodes)} episodes"
    return hits[0]


def check_exactly_one_episode(client: NSClient, project_id: str, marker: str) -> None:
    episodes = [e for e in graph_listing(client, "episodes", project_id) if marker in (e.get("content") or "")]
    assert len(episodes) == 1, f"{len(episodes)} episodes carry {marker}: {[e['uuid'] for e in episodes]}"


def check_entity_listed(client: NSClient, project_id: str, entity: str) -> None:
    nodes = graph_listing(client, "nodes", project_id)
    edges = graph_listing(client, "edges", project_id)
    assert blob_has(nodes, entity) or blob_has(edges, entity), (
        f"entity {entity} in none of {len(nodes)} nodes / {len(edges)} edges"
    )


def check_graph_search_finds(client: NSClient, project_id: str, entity: str, other: str) -> dict:
    r = client.post("/v1/graph/search", json={"query": f"{entity} depends on {other}", "project_id": project_id, "limit": 10})
    assert r.status_code == 200, f"POST /v1/graph/search -> {r.status_code}: {r.text[:160]}"
    body = r.json()
    assert body.get("status") == "ok" and {"edges", "nodes", "episodes", "communities"} <= set(body), f"keys: {sorted(body)}"
    assert blob_has(body["edges"], entity) or blob_has(body["nodes"], entity), f"{entity} not in graph search edges/nodes"
    return body


# ── episodes, entities, edges ────────────────────────────────────


@proves(
    "graph.graphiti-bridge", "graph.episodes-entities-edges", "fork.graphiti.add-episode-pipeline",
    "read.graph-listing", "model.graph-group-ids", "fork.graphiti.bitemporal",
)
def test_raw_write_becomes_a_dated_episode_in_the_owners_group(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    ep = eventually(
        lambda: check_episode(a, ctx.project_id, graph_seed.first.marker),
        timeout=GRAPH_WAIT, interval=5, what="graph episode for the raw write",
    )
    assert ep["group_id"] == f"user--{ctx.user_a}--project--{ctx.project_id}", f"episode group_id {ep['group_id']!r}"
    assert (ep.get("valid_at") or "").startswith("2021-03-04"), (
        f"episode valid_at {ep.get('valid_at')!r} does not carry the write's occurred_at ({graph_seed.occurred_at})"
    )
    assert {"uuid", "name", "content", "source_description", "created_at"} <= set(ep), f"episode keys {sorted(ep)}"


@proves("graph.episodes-entities-edges", "fork.graphiti.gemini-client-patches", "fork.graphiti.add-episode-pipeline", "read.graph-listing")
def test_entities_and_live_edges_are_extracted_and_listed(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    eventually(
        lambda: check_entity_listed(a, ctx.project_id, graph_seed.ent_a),
        timeout=GRAPH_WAIT, interval=5, what="entity extracted into the graph",
    )
    nodes = graph_listing(a, "nodes", ctx.project_id)
    assert nodes, "no entity nodes listed after extraction"
    assert {"uuid", "name", "summary", "labels", "group_id", "created_at"} <= set(nodes[0]), f"node keys {sorted(nodes[0])}"
    edges = graph_listing(a, "edges", ctx.project_id)
    assert edges, "no live edges listed after extraction"
    assert {"uuid", "fact", "valid_at", "invalid_at", "expired_at", "source_node_uuid", "target_node_uuid"} <= set(edges[0])
    assert all(e["invalid_at"] is None and e["expired_at"] is None for e in edges), "the default edge listing returned a non-live edge"


@proves("graph.episodes-entities-edges", "read.graph-listing")
def test_communities_listing_answers(ctx, a):
    assert isinstance(graph_listing(a, "communities", ctx.project_id), list)


@proves("graph.episodes-entities-edges")
@negative_control
def test_entity_check_fails_for_a_name_that_was_never_written(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    expect_check_fails(check_entity_listed, a, ctx.project_id, f"Zq{ctx.run_id.capitalize()}NeverWritten")


# ── graph search ─────────────────────────────────────────────────


@proves("read.search-knowledge-graph", "fork.graphiti.hybrid-edge-search")
def test_rest_graph_search_finds_the_entity(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    eventually(
        lambda: check_graph_search_finds(a, ctx.project_id, graph_seed.ent_a, graph_seed.ent_b),
        timeout=GRAPH_WAIT, interval=5, what="graph search finds the entity",
    )


@proves("read.search-knowledge-graph", "fork.graphiti.hybrid-edge-search")
def test_mcp_search_knowledge_graph_finds_the_entity(ctx, mcp_a, graph_seed: GraphSeed):
    graph_seed.first.require()
    mcp_a.initialize()

    def check():
        body = mcp_json(mcp_a, "search_knowledge_graph", {"query": f"{graph_seed.ent_a} depends on {graph_seed.ent_b}", "project_id": ctx.project_id})
        assert {"edges", "nodes", "episodes", "communities"} <= set(body), f"MCP graph search keys: {sorted(body)}"
        assert blob_has(body["edges"], graph_seed.ent_a) or blob_has(body["nodes"], graph_seed.ent_a), "entity not in MCP results"

    eventually(check, timeout=GRAPH_WAIT, interval=5, what="MCP graph search finds the entity")


@proves("read.search-knowledge-graph", "fork.graphiti.hybrid-edge-search")
@negative_control
def test_graph_search_check_fails_for_user_b(ctx, b, graph_seed: GraphSeed):
    graph_seed.first.require()
    expect_check_fails(check_graph_search_finds, b, ctx.project_id, graph_seed.ent_a, graph_seed.ent_b)


# ── partitioning ─────────────────────────────────────────────────


@proves("model.graph-group-ids", "read.graph-listing")
def test_other_users_graph_never_holds_this_users_entities(ctx, a, b, graph_seed: GraphSeed):
    graph_seed.first.require()
    eventually(
        lambda: check_episode(a, ctx.project_id, graph_seed.first.marker),
        timeout=GRAPH_WAIT, interval=5, what="A's episode exists (so B's emptiness means something)",
    )
    for kind in ("episodes", "nodes", "edges"):
        rows = graph_listing(b, kind, ctx.project_id)
        assert not blob_has(rows, graph_seed.ent_a) and not blob_has(rows, graph_seed.first.marker), (
            f"B's graph {kind} listing contains A's private entity or text"
        )
    body = b.post("/v1/graph/search", json={"query": graph_seed.ent_a, "project_id": ctx.project_id}).json()
    assert not blob_has(body, graph_seed.ent_a), "B's graph search returned A's private entity"


@proves("model.graph-group-ids")
@negative_control
def test_episode_check_fails_for_user_b(ctx, b, graph_seed: GraphSeed):
    graph_seed.first.require()
    expect_check_fails(check_episode, b, ctx.project_id, graph_seed.first.marker)


# ── idempotency ──────────────────────────────────────────────────


@proves("write.idempotent-graph-episode", "fork.graphiti.episode-idempotency", "write.content-hash-dedup")
def test_writing_the_same_fact_twice_yields_one_episode(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    graph_seed.repeat.require()
    assert graph_seed.repeat.memory_id == graph_seed.first.memory_id, "the repeat write created a second memory"
    eventually(
        lambda: check_episode(a, ctx.project_id, graph_seed.first.marker),
        timeout=GRAPH_WAIT, interval=5, what="episode for the repeated write",
    )
    check_exactly_one_episode(a, ctx.project_id, graph_seed.first.marker)


@proves("write.idempotent-graph-episode", "fork.graphiti.episode-idempotency")
@negative_control
def test_one_episode_check_fails_where_no_episode_exists(ctx, b, graph_seed: GraphSeed):
    graph_seed.first.require()
    expect_check_fails(check_exactly_one_episode, b, ctx.project_id, graph_seed.first.marker)


# ── conversation episodes ────────────────────────────────────────


@proves("fork.graphiti.message-episodes", "fork.graphiti.episode-idempotency", "fork.graphiti.add-episode-pipeline")
def test_conversation_becomes_one_content_named_episode(ctx, a, conv_seed: ConvSeed):
    assert conv_seed.write.completed, f"conversation write ended as {conv_seed.write.status or conv_seed.write.error}"
    ep = eventually(
        lambda: check_episode(a, ctx.project_id, conv_seed.forge),
        timeout=GRAPH_WAIT, interval=5, what="conversation episode",
    )
    assert ep["name"].startswith("mem0_episode_"), f"conversation episode is not content-named: {ep['name']!r}"
    assert conv_seed.lagoon in ep["content"], "one conversation should be one episode carrying both facts"


# ── bi-temporal liveness ─────────────────────────────────────────


@proves("model.bitemporal-liveness", "fork.graphiti.bitemporal")
def test_expired_edges_are_a_dictator_only_view(ctx, a):
    for kind in ("edges", "nodes"):
        r = a.get(f"/v1/graph/{kind}", params={"project_id": ctx.project_id, "include_expired": "true"})
        assert r.status_code == 403, f"include_expired=true on {kind} by a non-dictator -> {r.status_code}: {r.text[:120]}"
    ok = a.get("/v1/graph/edges", params={"project_id": ctx.project_id, "include_expired": "false"})
    assert ok.status_code == 200, f"include_expired=false -> {ok.status_code}"


@proves("model.bitemporal-liveness")
@negative_control
def test_dictator_view_check_fails_for_the_plain_listing(ctx, a):
    def check():
        r = a.get("/v1/graph/edges", params={"project_id": ctx.project_id})
        assert r.status_code == 403, f"plain listing -> {r.status_code}"

    expect_check_fails(check)


# ── admin ────────────────────────────────────────────────────────


@proves("graph.graph-admin")
def test_episode_delete_is_owner_only_and_removes_the_episode(ctx, a, b, graph_seed: GraphSeed):
    graph_seed.first.require()
    graph_seed.sibling.require()
    keep = eventually(lambda: check_episode(a, ctx.project_id, graph_seed.first.marker), timeout=GRAPH_WAIT, interval=5, what="first episode")
    doomed = eventually(lambda: check_episode(a, ctx.project_id, graph_seed.sibling.marker), timeout=GRAPH_WAIT, interval=5, what="sibling episode")
    refused = b.delete(f"/v1/graph/episodes/{doomed['uuid']}", params={"project_id": ctx.project_id})
    assert refused.status_code == 403, f"B deleting A's episode -> {refused.status_code}: {refused.text[:120]}"
    assert check_episode(a, ctx.project_id, graph_seed.sibling.marker)["uuid"] == doomed["uuid"], "B's refused delete still removed the episode"
    ok = a.delete(f"/v1/graph/episodes/{doomed['uuid']}", params={"project_id": ctx.project_id})
    assert ok.status_code == 200, f"A deleting own episode -> {ok.status_code}: {ok.text[:120]}"
    uuids = [e["uuid"] for e in graph_listing(a, "episodes", ctx.project_id)]
    assert doomed["uuid"] not in uuids and keep["uuid"] in uuids, "delete removed the wrong episode set"


@proves("graph.graph-admin")
def test_deleting_an_unknown_episode_is_a_404_not_a_server_error(ctx, a):
    r = a.delete(f"/v1/graph/episodes/{uuid.uuid4()}", params={"project_id": ctx.project_id})
    assert r.status_code == 404, f"unknown episode -> {r.status_code} (want 404): {r.text[:160]}"


@proves("graph.graph-admin")
def test_junk_episode_cleanup_dry_run_reports_without_deleting(ctx, a, graph_seed: GraphSeed):
    graph_seed.first.require()
    r = a.delete("/v1/graph/episodes/junk", params={"project_id": ctx.project_id, "dry_run": "true"})
    assert r.status_code == 200 and r.json().get("status") == "ok", f"junk dry-run -> {r.status_code}: {r.text[:160]}"
    assert check_episode(a, ctx.project_id, graph_seed.first.marker), "a dry run removed a real episode"


@proves("graph.graph-admin")
@negative_control
def test_admin_delete_check_fails_for_a_foreign_episode_id(ctx, b):
    def check():
        r = b.delete(f"/v1/graph/episodes/{uuid.uuid4()}", params={"project_id": ctx.project_id})
        assert r.status_code == 200, f"delete -> {r.status_code}"

    expect_check_fails(check)
