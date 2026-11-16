"""Real Neo4j + Graphiti + Gemini: graph enrichment stamps provenance from its own episode.

Two ``enrich_graph`` calls run concurrently into ONE group; every stamped node's
``memory_id`` / ``memory_ids`` must come only from memories whose episode
mentions it (the old creation-time window let each job stamp the other's
nodes). A repeated enrichment of the same memory leaves one episode, and after
service initialisation Graphiti's indexes exist.

Env: the ``stores`` fixture's variables plus GOOGLE_API_KEY (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.integration

ENRICH_ATTEMPTS = 3  # enrich_graph swallows Graphiti/LLM errors and returns False


@pytest.fixture(scope="module")
def svc(stores, llm_key):
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    yield service


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _enrich(svc, uid, mid, content, out):
    ok = False
    for _ in range(ENRICH_ATTEMPTS):
        ok = svc.enrich_graph(content=content, user_id=uid, project_id=None, visibility="private",
                              memory_id=mid, result_out=out)
        if ok:
            break
    return ok


def _store(svc, uid, content):
    stored, created = svc.store_raw(content=content, user_id=uid, category="personal_fact",
                                    visibility="private", add_to_graph=False, return_created=True)
    assert created and stored
    return stored[0].id


def test_graphiti_indexes_exist_after_service_initialisation(svc):
    # First in the module on purpose: no graph add/search has run yet, so the
    # indexes can only come from the start-up hook.
    # _ensure_indexes_at_startup builds them on a daemon thread at init.
    svc._index_ensure_thread.join(60)
    deadline = time.monotonic() + 60
    names: set[str] = set()
    while time.monotonic() < deadline:
        names = {r["name"] for r in _cypher(svc, "SHOW INDEXES YIELD name RETURN name")}
        if {"episode_uuid", "entity_uuid"} & names or any("entity" in n.lower() for n in names):
            break
        time.sleep(1)
    assert any("entity" in n.lower() for n in names) and any("episode" in n.lower() for n in names), sorted(names)


def test_concurrent_enrichments_stamp_only_their_own_episode_nodes(svc, unique_ns):
    from memory.groups import _build_group_id

    uid = unique_ns.user_id
    group = _build_group_id("private", uid)
    tok_a, tok_b = (f"Zorblax{uuid.uuid4().hex[:8]}", f"Quimbly{uuid.uuid4().hex[:8]}")
    ca = f"My colleague {tok_a} leads the Lisbon office and drives a green tractor."
    cb = f"My neighbour {tok_b} breeds alpacas in Patagonia and plays the cello."

    def cleanup():
        svc.delete_memories(user_id=uid, include_shared=True)
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group)

    unique_ns.on_cleanup(cleanup)
    ma, mb = _store(svc, uid, ca), _store(svc, uid, cb)
    out_a: dict = {}
    out_b: dict = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_enrich, svc, uid, ma, ca, out_a)
        fb = pool.submit(_enrich, svc, uid, mb, cb, out_b)
        assert fa.result() is True and fb.result() is True
    assert out_a["episode_uuid"] and out_b["episode_uuid"] and out_a["episode_uuid"] != out_b["episode_uuid"]

    episodes = {out_a["episode_uuid"]: ma, out_b["episode_uuid"]: mb}
    rows = _cypher(
        svc,
        "MATCH (n:Entity {group_id: $g}) WHERE n.memory_id IS NOT NULL OR n.memory_ids IS NOT NULL "
        "OPTIONAL MATCH (ep:Episodic)-[:MENTIONS]->(n) "
        "RETURN n.name AS name, n.memory_id AS mid, n.memory_ids AS mids, collect(DISTINCT ep.uuid) AS eps",
        g=group,
    )
    assert rows, "no entity was stamped with provenance"
    for r in rows:
        mentioning = {episodes[e] for e in r["eps"] if e in episodes}
        assert r["mid"] in mentioning, f"{r['name']}: memory_id {r['mid']} is not a mentioning episode's memory"
        assert set(r["mids"] or []) <= mentioning, f"{r['name']}: memory_ids {r['mids']} cite a non-mentioning memory"
        assert r["mid"] in (r["mids"] or [])
    eps = _cypher(svc, "MATCH (e:Episodic {group_id: $g}) RETURN e.uuid AS u, e.memory_id AS mid", g=group)
    assert {e["u"]: e["mid"] for e in eps} == episodes


def test_repeated_enrichment_of_one_memory_leaves_one_episode(svc, unique_ns):
    from memory.groups import _build_group_id

    uid = unique_ns.user_id
    group = _build_group_id("private", uid)
    tok = f"Wexford{uuid.uuid4().hex[:8]}"
    content = f"My sister {tok} restores antique clocks in Prague."

    unique_ns.on_cleanup(lambda: (svc.delete_memories(user_id=uid, include_shared=True),
                                  _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group)))
    mid = _store(svc, uid, content)
    first: dict = {}
    assert _enrich(svc, uid, mid, content, first) is True
    again: dict = {}
    assert svc.enrich_graph(content=content, user_id=uid, project_id=None, visibility="private",
                            memory_id=mid, result_out=again) is True
    count = _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN count(e) AS c",
                    g=group, t=tok)[0]["c"]
    assert count == 1, f"repeat enrichment minted {count} episodes"
    assert again["episode_uuid"] == first["episode_uuid"]


def test_communities_of_the_episodes_entities_are_stamped_once_each(svc, unique_ns):
    """Real Neo4j semantics of the selector: a Community that has the episode's entity
    as HAS_MEMBER is stamped (memory_id + one memory_ids entry); a community of
    another group or of an unrelated entity is not."""
    from datetime import datetime, timezone

    from extensions.dreaming.graph_patcher import attach_memory_id

    g, other_g = f"{unique_ns.group_prefix}-c", f"{unique_ns.group_prefix}-x"
    unique_ns.on_cleanup(lambda: _cypher(svc, "MATCH (n) WHERE n.group_id IN $g DETACH DELETE n", g=[g, other_g]))
    _cypher(
        svc,
        "CREATE (ep:Episodic {uuid: 'ep-u', group_id: $g, name: 'n'}) "
        "CREATE (e1:Entity {uuid: 'e1', group_id: $g}), (e2:Entity {uuid: 'e2', group_id: $g}), "
        "       (e3:Entity {uuid: 'e3', group_id: $g}) "
        "CREATE (ep)-[:MENTIONS]->(e1), (ep)-[:MENTIONS]->(e2) "
        "CREATE (c1:Community {uuid: 'c1', group_id: $g})-[:HAS_MEMBER]->(e1), (c1)-[:HAS_MEMBER]->(e2) "
        "CREATE (c2:Community {uuid: 'c2', group_id: $g})-[:HAS_MEMBER]->(e3) "
        "CREATE (cx:Community {uuid: 'cx', group_id: $o})-[:HAS_MEMBER]->(e1)",
        g=g, o=other_g,
    )

    async def go():
        return await attach_memory_id(
            svc._graphiti.driver, group_id=g, memory_id="mem-1", visibility="private",
            owner_user_id="u", write_started_at=datetime.now(timezone.utc), episode_uuid="ep-u",
        )

    patched = svc._run_on_bridge(go(), timeout=60.0)
    rows = {r["u"]: r for r in _cypher(
        svc, "MATCH (n) WHERE n.group_id IN $g RETURN n.uuid AS u, n.memory_id AS m, n.memory_ids AS ms",
        g=[g, other_g])}
    assert patched == 4  # ep, e1, e2, c1 -- c1 is reached twice but stamped once
    for u in ("ep-u", "e1", "e2", "c1"):
        assert rows[u]["m"] == "mem-1" and rows[u]["ms"] == ["mem-1"], u
    for u in ("e3", "c2", "cx"):
        assert rows[u]["m"] is None and rows[u]["ms"] is None, u
