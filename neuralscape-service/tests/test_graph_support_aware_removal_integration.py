"""Support-aware graph removal on the service's delete and correct paths, against a REAL Neo4j.

The service runs with a real Graphiti (Neo4j driver; LLM, embedder and reranker are inert stubs, no model is
called) on a real async bridge.  The memory row store is a labelled in-memory stand-in (no Qdrant/mem0); the
graph is seeded structurally in Graphiti's own schema: two of alice's memories, A and B, whose episodes both
support the edge X->Y, while only A supports X->Z.

* ``DELETE /v1/memories/{A}`` (the API route) and the ledger worker's delete stage (governed path): with
  ``GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED`` the shared fact X->Y stays live, now supported by B only; X->Z and
  the entity Z (referenced by nothing else) are removed; the mentioned entities' summaries are cleared; A's
  episode is gone; no lock node is left.  With the flag off the legacy most-restrictive cascade expires X->Y
  (pinned as the documented difference).
* Correction: re-enriching A with corrected text that still asserts X->Y removes A's superseded episode after
  the new one exists, so X->Y keeps the new support and B's, and X->Z goes.
* One conversation, several memories: conversation extraction stamps every fact it creates with the same
  episode.  A correction or delete of one of them removes that episode anyway (the removed memory's facts must
  not stay retrievable; facts another episode supports stay), and every surviving sibling is re-enriched into a
  fresh episode of its own (inline here, or as a durable ledger job), so its facts come back.  A correction whose removal fails records the superseded episode on the row and the
  next correction or the delete retries it, in the group the episode lives in.

Skipped unless ``ENGINE_TEST_NEO4J_URI`` points at a disposable Neo4j (it is wiped between checks).
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

NEO4J_URI = os.environ.get("ENGINE_TEST_NEO4J_URI")
NEO4J_PASSWORD = os.environ.get("ENGINE_TEST_NEO4J_PASSWORD", "neo4j")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not NEO4J_URI, reason="ENGINE_TEST_NEO4J_URI is not set (graph support-aware removal)"),
]

from config import settings  # noqa: E402
from memory.groups import _build_group_id  # noqa: E402
from memory_service import MemoryService  # noqa: E402

OWNER = "alice"
GROUP = _build_group_id("private", OWNER, None, None)


@pytest.fixture(scope="module")
def graph():
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client import LLMClient
    from mem0.memory.graphiti_memory import _AsyncBridge

    class InertLLM(LLMClient):
        def __init__(self):
            super().__init__(None)

        async def _generate_response(self, *args, **kwargs):
            raise RuntimeError("no model is available in this test")

    class InertEmbedder(EmbedderClient):
        async def create(self, input_data):
            return [0.0] * 8

    class InertReranker(CrossEncoderClient):
        async def rank(self, query, passages):
            return []

    bridge = _AsyncBridge()

    async def build():
        driver = Neo4jDriver(uri=NEO4J_URI, user="neo4j", password=NEO4J_PASSWORD, database="neo4j")
        return Graphiti(llm_client=InertLLM(), embedder=InertEmbedder(), cross_encoder=InertReranker(),
                        graph_driver=driver)

    graphiti = bridge.run(build())
    try:
        yield graphiti, bridge
    finally:
        bridge.run(graphiti.driver.execute_query("MATCH (n) DETACH DELETE n"))
        bridge.run(graphiti.close())


class RowStore:
    """Labelled stand-in for the vector row store: ``get``/``delete`` and the metadata payload patch."""

    def __init__(self):
        self.rows = {}
        store = self

        class Client:
            def set_payload(self, collection_name, payload, points, key):
                for point in points:
                    store.rows[point]["metadata"].update(payload)

            def scroll(self, collection_name, scroll_filter, limit, offset=None, **kwargs):
                # Qdrant's semantics for the episode-reference filter: any ``should`` condition matches.
                wanted = {(c.key.split(".", 1)[1], c.match.value) for c in scroll_filter.should}
                hits = [SimpleNamespace(id=memory_id, payload={"data": row["memory"], "user_id": OWNER,
                                                               "metadata": dict(row["metadata"])})
                        for memory_id, row in store.rows.items()
                        if any(row["metadata"].get(key) == value for key, value in wanted)]
                start = offset or 0
                return hits[start:start + limit], (start + limit if start + limit < len(hits) else None)

        self.vector_store = SimpleNamespace(client=Client())
        self.graph = SimpleNamespace(add=None)

    def get(self, memory_id):
        row = self.rows.get(memory_id)
        return None if row is None else {"id": memory_id, "memory": row["memory"], "user_id": OWNER,
                                         "metadata": dict(row["metadata"])}

    def delete(self, memory_id):
        self.rows.pop(memory_id, None)
        return {"message": "Memory deleted successfully!"}


class GraphService(MemoryService):
    def __init__(self, graphiti, bridge):
        self._memory = RowStore()
        self._graphiti = graphiti
        self._bridge = bridge

    def _get_memory(self):
        return self._memory


def q(graphiti, bridge, cypher, **params):
    records, _, _ = bridge.run(graphiti.driver.execute_query(cypher, **params))
    return [dict(r) for r in records]


def seed(graphiti, bridge):
    q(graphiti, bridge, "MATCH (n) DETACH DELETE n")
    ids = {name: str(uuid.uuid4()) for name in ("EA", "EB", "X", "Y", "Z", "XY", "XZ")}
    q(graphiti, bridge, """
        CREATE (x:Entity {uuid: $X, group_id: $g, name: 'tea', summary: 'about tea', created_at: datetime()})
        CREATE (y:Entity {uuid: $Y, group_id: $g, name: 'green tea', summary: 'about green tea', created_at: datetime()})
        CREATE (z:Entity {uuid: $Z, group_id: $g, name: 'morning', summary: 'only from A', created_at: datetime()})
        CREATE (a:Episodic {uuid: $EA, group_id: $g, name: 'ep-a', content: 'memory A text', source: 'text',
                            source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                            entity_edges: [$XY, $XZ]})
        CREATE (b:Episodic {uuid: $EB, group_id: $g, name: 'ep-b', content: 'memory B text', source: 'text',
                            source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                            entity_edges: [$XY]})
        CREATE (a)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
        CREATE (a)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(y)
        CREATE (a)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(z)
        CREATE (b)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
        CREATE (b)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(y)
        CREATE (x)-[:RELATES_TO {uuid: $XY, group_id: $g, name: 'IS_A', fact: 'green tea is a tea',
                                 episodes: [$EA, $EB], created_at: datetime()}]->(y)
        CREATE (x)-[:RELATES_TO {uuid: $XZ, group_id: $g, name: 'IN', fact: 'tea in the morning',
                                 episodes: [$EA], created_at: datetime()}]->(z)
        """, g=GROUP, **ids)
    return ids


def make_service(graph, ids):
    graphiti, bridge = graph
    service = GraphService(graphiti, bridge)
    for memory, episode in (("mem-a", ids["EA"]), ("mem-b", ids["EB"])):
        service._memory.rows[memory] = {
            "memory": f"memory {memory[-1].upper()} text",
            "metadata": {"owner_user_id": OWNER, "visibility": "private", "graph_episode_uuid": episode},
        }
    return service


def edge(graph, edge_uuid):
    graphiti, bridge = graph
    rows = q(graphiti, bridge, "MATCH ()-[r:RELATES_TO {uuid: $u}]->() RETURN r.episodes AS episodes, "
             "r.expired_at IS NOT NULL AS expired", u=edge_uuid)
    return rows[0] if rows else None


def node(graph, node_uuid):
    graphiti, bridge = graph
    rows = q(graphiti, bridge, "MATCH (n:Entity {uuid: $u}) RETURN n.summary AS summary", u=node_uuid)
    return rows[0] if rows else None


def exists(graph, label, node_uuid):
    graphiti, bridge = graph
    return bool(q(graphiti, bridge, f"MATCH (n:{label} {{uuid: $u}}) RETURN n", u=node_uuid))


def locks_left(graph):
    graphiti, bridge = graph
    return q(graphiti, bridge, "MATCH (l:NsLock) RETURN count(l) AS n")[0]["n"]


def delete_through_the_api(service, monkeypatch, memory_id):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(settings, "default_user_id", OWNER)
    monkeypatch.setattr(settings, "ledger_enabled", False)
    response = TestClient(main.app).delete(f"/v1/memories/{memory_id}")
    assert response.status_code == 200, response.text
    return response.json()


def assert_support_aware_outcome(graph, ids):
    shared = edge(graph, ids["XY"])
    assert shared is not None and shared["expired"] is False
    assert shared["episodes"] == [ids["EB"]]  # still live, supported by B only
    assert edge(graph, ids["XZ"]) is None  # its only support was A
    assert not exists(graph, "Entity", ids["Z"])  # nothing references it any more
    assert node(graph, ids["X"])["summary"] == "" and node(graph, ids["Y"])["summary"] == ""  # cleared
    assert not exists(graph, "Episodic", ids["EA"]) and exists(graph, "Episodic", ids["EB"])
    assert locks_left(graph) == 0


def test_api_delete_keeps_a_fact_another_memory_still_supports(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    result = delete_through_the_api(make_service(graph, ids), monkeypatch, "mem-a")
    assert result.get("graph_cascade") == "resolved"
    assert_support_aware_outcome(graph, ids)


def test_the_governed_ledger_delete_stage_keeps_a_fact_another_memory_still_supports(graph, monkeypatch, tmp_path):
    import asyncio

    from ledger.sqlite import SqliteLedger
    from write_producer import DurableWriteProducer, LedgerWriteWorker, SqlitePayloadStore

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    try:
        producer = DurableWriteProducer(ledger, payloads)
        accepted = producer.accept_delete(user_id=OWNER, memory_id="mem-a",
                                          memory_row={"owner_user_id": OWNER, "visibility": "private"})
        worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0,
                                   retry_delay=lambda attempt: 0.0)
        asyncio.run(worker.drain())
        stored = ledger.get_intent(producer.tenant_id, accepted.intent_id)
        assert stored.status.value == "applied", stored.status
    finally:
        payloads.close()
        ledger.close()
    assert_support_aware_outcome(graph, ids)


def test_with_the_flag_off_the_legacy_cascade_still_expires_the_shared_fact(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    ids = seed(*graph)
    delete_through_the_api(make_service(graph, ids), monkeypatch, "mem-a")
    assert edge(graph, ids["XY"])["expired"] is True  # most-restrictive: expired despite B's support
    only_a = edge(graph, ids["XZ"])
    assert only_a is None or only_a["expired"] is True  # gone with the sole-mentioned entity, or expired


def test_a_correction_replaces_the_superseded_episode_and_keeps_still_asserted_facts(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    graphiti, bridge = graph
    new_episode = str(uuid.uuid4())

    def add_corrected(**kwargs):
        # Structural stand-in for Graphiti extraction of the corrected text: a new episode that
        # reaffirms X->Y (the corrected text still says it) and no longer mentions the morning.
        q(graphiti, bridge, """
            MATCH (x:Entity {uuid: $X}), (y:Entity {uuid: $Y})
            MATCH ()-[r:RELATES_TO {uuid: $XY}]->()
            CREATE (e:Episodic {uuid: $E, group_id: $g, name: 'ep-a2', content: 'corrected A', source: 'text',
                                source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                                entity_edges: [$XY]})
            CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
            CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(y)
            SET r.episodes = r.episodes + $E
            """, X=ids["X"], Y=ids["Y"], XY=ids["XY"], E=new_episode, g=GROUP)
        return {"episode_uuid": new_episode, "episode_name": "ep-a2"}

    service._memory.graph.add = add_corrected
    assert service.enrich_graph("corrected A", OWNER, None, "private", "mem-a") is True
    shared = edge(graph, ids["XY"])
    assert shared is not None and shared["expired"] is False
    assert sorted(shared["episodes"]) == sorted([ids["EB"], new_episode])
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert not exists(graph, "Episodic", ids["EA"])
    assert service._memory.rows["mem-a"]["metadata"]["graph_episode_uuid"] == new_episode
    assert locks_left(graph) == 0


def test_with_the_flag_off_a_correction_leaves_the_superseded_episode(graph, monkeypatch):
    """Pinned legacy behavior: the old episode keeps its facts live (the defect the flag fixes)."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    ids = seed(*graph)
    service = make_service(graph, ids)
    service._memory.graph.add = lambda **kwargs: {"episode_uuid": str(uuid.uuid4())}
    service.enrich_graph("corrected A", OWNER, None, "private", "mem-a")
    assert exists(graph, "Episodic", ids["EA"]) and edge(graph, ids["XZ"])["expired"] is False


SIBLING_TEXT = "the C fact of the A conversation"


def add_sibling(service, graph, ids, memory_id="mem-c", *, by_name_only=False):
    """Another fact of A's conversation.  Conversation extraction stamped it with the same episode EA, which
    therefore also supports C's own fact X->W (seeded here) next to A's X->Z and the shared X->Y."""
    graphiti, bridge = graph
    ids["W"], ids["XW"] = str(uuid.uuid4()), str(uuid.uuid4())
    q(graphiti, bridge, """
        MATCH (x:Entity {uuid: $X}), (a:Episodic {uuid: $EA})
        CREATE (w:Entity {uuid: $W, group_id: $g, name: 'evening', summary: 'from the conversation', created_at: datetime()})
        CREATE (a)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(w)
        CREATE (x)-[:RELATES_TO {uuid: $XW, group_id: $g, name: 'AT', fact: 'tea in the evening',
                                 episodes: [$EA], created_at: datetime()}]->(w)
        SET a.entity_edges = a.entity_edges + $XW
        """, X=ids["X"], EA=ids["EA"], W=ids["W"], XW=ids["XW"], g=GROUP)
    metadata = {"owner_user_id": OWNER, "visibility": "private", "graph_episode_name": "ep-a"}
    if not by_name_only:
        metadata["graph_episode_uuid"] = ids["EA"]
    service._memory.rows[memory_id] = {"memory": SIBLING_TEXT, "metadata": metadata}


def add_sibling_episode(graph, ids, group=GROUP):
    """Structural stand-in for Graphiti re-extracting the sibling alone: a fresh episode asserting C's fact."""
    graphiti, bridge = graph
    episode, fact = str(uuid.uuid4()), str(uuid.uuid4())
    q(graphiti, bridge, """
        MATCH (x:Entity {uuid: $X})
        MERGE (w:Entity {uuid: $W}) ON CREATE SET w.group_id = $g, w.name = 'evening', w.created_at = datetime()
        CREATE (e:Episodic {uuid: $E, group_id: $g, name: 'ep-' + $E, content: $text, source: 'text',
                            source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                            entity_edges: [$F]})
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(w)
        CREATE (x)-[:RELATES_TO {uuid: $F, group_id: $g, name: 'AT', fact: 'tea in the evening',
                                 episodes: [$E], created_at: datetime()}]->(w)
        """, X=ids["X"], W=ids["W"], E=episode, F=fact, text=SIBLING_TEXT, g=group)
    ids.setdefault("sibling_episodes", []).append(episode)
    ids["XW2"] = fact
    return episode


def facts(graph, text):
    graphiti, bridge = graph
    return q(graphiti, bridge, "MATCH ()-[r:RELATES_TO {fact: $f}]->() RETURN r.uuid AS uuid, r.episodes AS episodes, "
             "r.expired_at IS NOT NULL AS expired", f=text)


def add_episode_supporting_xy(graph, ids):
    """Structural stand-in for Graphiti extracting corrected text that still asserts X->Y."""
    graphiti, bridge = graph
    new_episode = str(uuid.uuid4())
    q(graphiti, bridge, """
        MATCH (x:Entity {uuid: $X}), (y:Entity {uuid: $Y})
        MATCH ()-[r:RELATES_TO {uuid: $XY}]->()
        CREATE (e:Episodic {uuid: $E, group_id: $g, name: 'ep-' + $E, content: 'corrected', source: 'text',
                            source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                            entity_edges: [$XY]})
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(y)
        SET r.episodes = r.episodes + $E
        """, X=ids["X"], Y=ids["Y"], XY=ids["XY"], E=new_episode, g=GROUP)
    return new_episode


def install_graph_add(service, graph, ids):
    """``graph.add`` stand-in: the sibling's text yields its own fresh episode, any other text the correction's."""
    created = {}

    def add(**kwargs):
        group = kwargs["filters"]["group_id"]
        if kwargs["data"] == SIBLING_TEXT:
            episode = add_sibling_episode(graph, ids, group)
        elif group == GROUP:
            episode = created["correction"] = add_episode_supporting_xy(graph, ids)
        else:  # extraction into another group writes that group's own nodes, none of the seeded ones
            graphiti, bridge = graph
            episode = created["correction"] = str(uuid.uuid4())
            q(graphiti, bridge, "CREATE (:Episodic {uuid: $E, group_id: $g, name: 'ep-' + $E, content: 'corrected', "
              "source: 'text', source_description: 'memory', created_at: datetime(), valid_at: datetime(), "
              "entity_edges: []})", E=episode, g=group)
        return {"episode_uuid": episode, "episode_name": "ep-" + episode}

    service._memory.graph.add = add
    return created


def correct(service, graph, ids, memory_id="mem-a", visibility="private"):
    created = install_graph_add(service, graph, ids)
    assert service.enrich_graph("corrected", OWNER, None, visibility, memory_id) is True
    return created["correction"]


def assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids):
    """A's own fact is gone, the fact another episode supports is untouched, and C's fact is back on C's
    fresh episode (C's row now names that episode and no longer waits for re-enrichment)."""
    assert not exists(graph, "Episodic", ids["EA"])
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])  # A's fact, gone
    assert edge(graph, ids["XW"]) is None  # C's fact as the shared episode held it, gone with it ...
    (fresh,) = ids["sibling_episodes"]
    assert facts(graph, "tea in the evening") == [{"uuid": ids["XW2"], "episodes": [fresh], "expired": False}]
    row = service._memory.rows["mem-c"]["metadata"]
    assert row["graph_episode_uuid"] == fresh and row.get("graph_reenrich_pending") is None  # ... and back
    assert ids["EB"] in edge(graph, ids["XY"])["episodes"] and edge(graph, ids["XY"])["expired"] is False
    assert locks_left(graph) == 0


def test_a_delete_removes_the_shared_episode_and_re_enriches_the_surviving_sibling(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    result = delete_through_the_api(service, monkeypatch, "mem-a")
    assert result.get("graph_cascade") == "resolved"
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)
    assert edge(graph, ids["XY"])["episodes"] == [ids["EB"]]


def test_a_correction_removes_the_shared_episode_and_re_enriches_the_surviving_sibling(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    new_episode = correct(service, graph, ids)
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)
    assert sorted(edge(graph, ids["XY"])["episodes"]) == sorted([ids["EB"], new_episode])
    assert service._memory.rows["mem-a"]["metadata"]["graph_episode_uuid"] == new_episode
    assert not service._memory.rows["mem-a"]["metadata"].get("graph_episode_superseded")


def test_a_sibling_referencing_the_episode_by_name_only_is_re_enriched_too(graph, monkeypatch):
    """A re-run conversation stamps only the deterministic episode name on its new rows."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    service._memory.rows["mem-a"]["metadata"]["graph_episode_name"] = "ep-a"
    add_sibling(service, graph, ids, by_name_only=True)
    install_graph_add(service, graph, ids)
    delete_through_the_api(service, monkeypatch, "mem-a")
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)


def test_until_re_enrichment_runs_the_sibling_waits_marked_and_its_facts_are_absent(graph, monkeypatch):
    """With a durable dispatch installed the delete only schedules the sibling: availability loss, no exposure."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    jobs = []
    service.set_graph_reenrich_dispatch(jobs.append)
    delete_through_the_api(service, monkeypatch, "mem-a")
    assert [job["memory_id"] for job in jobs] == ["mem-c"] and jobs[0]["removed_episode"] == ids["EA"]
    row = service._memory.rows["mem-c"]["metadata"]
    assert row["graph_reenrich_pending"] == ids["EA"] and row["graph_episode_uuid"] is None
    assert not exists(graph, "Episodic", ids["EA"]) and facts(graph, "tea in the evening") == []
    assert service.reenrich_memory_graph("mem-c") == {"memory_ids": ["mem-c"]}  # the job, when it runs
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)


def test_the_governed_ledger_runs_the_sibling_re_enrichment_as_its_own_durable_job(graph, monkeypatch, tmp_path):
    import asyncio

    from ledger.sqlite import SqliteLedger
    from memory.provenance import ledger_reenrich_dispatch
    from write_producer import DurableWriteProducer, LedgerWriteWorker, SqlitePayloadStore

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    try:
        producer = DurableWriteProducer(ledger, payloads)
        dispatch, jobs = ledger_reenrich_dispatch(producer), []
        service.set_graph_reenrich_dispatch(lambda job: jobs.append(dispatch(job)))
        accepted = producer.accept_delete(user_id=OWNER, memory_id="mem-a",
                                          memory_row={"owner_user_id": OWNER, "visibility": "private"})
        worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0,
                                   retry_delay=lambda attempt: 0.0)
        asyncio.run(worker.drain())  # the delete, which accepts the sibling's job
        asyncio.run(worker.drain())  # and that job
        assert ledger.get_intent(producer.tenant_id, accepted.intent_id).status.value == "applied"
        (job,) = jobs
        assert ledger.get_intent(producer.tenant_id, job.intent_id).status.value == "applied"
    finally:
        payloads.close()
        ledger.close()
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)


def test_with_the_flag_off_a_delete_still_removes_a_shared_episode(graph, monkeypatch):
    """Pinned legacy behavior: the most-restrictive cascade does not look at sibling rows."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    delete_through_the_api(service, monkeypatch, "mem-a")
    assert not exists(graph, "Episodic", ids["EA"]) and edge(graph, ids["XY"])["expired"] is True


def test_a_ghost_episode_id_in_an_edge_does_not_keep_the_edge_alive(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    graphiti, bridge = graph
    q(graphiti, bridge, "MATCH ()-[r:RELATES_TO {uuid: $u}]->() SET r.episodes = r.episodes + $ghost",
      u=ids["XZ"], ghost=str(uuid.uuid4()))  # an id no episode has (e.g. one removed out of band)
    delete_through_the_api(make_service(graph, ids), monkeypatch, "mem-a")
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert_support_aware_outcome(graph, ids)


def test_a_failed_correction_removal_is_recorded_and_retried_by_the_next_correction(graph, monkeypatch):
    import engine_adapters_current

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    real = engine_adapters_current.GraphitiGraphAdapter.remove_episodes
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("neo4j unavailable")))
    first = correct(service, graph, ids)
    assert exists(graph, "Episodic", ids["EA"])  # not removed now ...
    assert service._memory.rows["mem-a"]["metadata"]["graph_episode_superseded"] == [ids["EA"]]  # ... but kept
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes", real)
    second = correct(service, graph, ids)
    assert not exists(graph, "Episodic", ids["EA"]) and not exists(graph, "Episodic", first)
    assert edge(graph, ids["XZ"]) is None
    assert sorted(edge(graph, ids["XY"])["episodes"]) == sorted([ids["EB"], second])
    assert service._memory.rows["mem-a"]["metadata"]["graph_episode_superseded"] == []
    assert locks_left(graph) == 0


def test_a_failed_correction_removal_is_retried_by_the_delete(graph, monkeypatch):
    import engine_adapters_current

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    real = engine_adapters_current.GraphitiGraphAdapter.remove_episodes
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("neo4j unavailable")))
    first = correct(service, graph, ids)
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes", real)
    delete_through_the_api(service, monkeypatch, "mem-a")
    assert not exists(graph, "Episodic", ids["EA"]) and not exists(graph, "Episodic", first)
    assert_support_aware_outcome_without_a(graph, ids)


def assert_support_aware_outcome_without_a(graph, ids):
    shared = edge(graph, ids["XY"])
    assert shared == {"episodes": [ids["EB"]], "expired": False}
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert locks_left(graph) == 0


def test_a_correction_into_another_group_removes_the_superseded_episode_where_it_lives(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    new_episode = correct(service, graph, ids, visibility="shared")  # the corrected memory moved group
    assert _build_group_id("shared", OWNER, None, None) != GROUP
    assert not exists(graph, "Episodic", ids["EA"])  # removed from the private group it was written to
    assert exists(graph, "Episodic", new_episode)
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert edge(graph, ids["XY"]) == {"episodes": [ids["EB"]], "expired": False}  # B still supports it
    assert locks_left(graph) == 0


# ---- a memory deleted while it is being enriched leaves nothing in the graph


def episodes_with(graph, text):
    graphiti, bridge = graph
    return [r["uuid"] for r in q(graphiti, bridge, "MATCH (e:Episodic) WHERE e.content = $t RETURN e.uuid AS uuid", t=text)]


@pytest.mark.parametrize("support_aware", [True, False], ids=["support-aware", "legacy"])
def test_a_memory_deleted_during_its_enrichment_leaves_no_episode(graph, monkeypatch, support_aware):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", support_aware)
    ids = seed(*graph)
    service = make_service(graph, ids)
    service._memory.rows["mem-new"] = {"memory": "oolong in the afternoon",
                                       "metadata": {"owner_user_id": OWNER, "visibility": "private"}}
    graphiti, bridge = graph

    def add_while_the_user_deletes(**kwargs):
        # The delete lands during extraction: it finds no episode yet, so it removes nothing.
        result = service.delete_memory("mem-new")
        assert result.get("graph_cascade") == "unresolved"
        episode = str(uuid.uuid4())
        q(graphiti, bridge, """
            MATCH (x:Entity {uuid: $X})
            CREATE (e:Episodic {uuid: $E, group_id: $g, name: 'ep-' + $E, content: $text, source: 'text',
                                source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                                entity_edges: []})
            CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: $g, created_at: datetime()}]->(x)
            """, X=ids["X"], E=episode, text=kwargs["data"], g=GROUP)
        return {"episode_uuid": episode, "episode_name": "ep-" + episode}

    service._memory.graph.add = add_while_the_user_deletes
    service.enrich_graph("oolong in the afternoon", OWNER, None, "private", "mem-new")
    assert episodes_with(graph, "oolong in the afternoon") == []  # the deleted memory's text is not in the graph
    assert exists(graph, "Episodic", ids["EA"]) and exists(graph, "Episodic", ids["EB"])  # others untouched
    assert locks_left(graph) == 0


def test_a_survivor_deleted_during_its_re_enrichment_leaves_nothing_in_the_graph(graph, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids)
    add_sibling(service, graph, ids)
    jobs = []
    service.set_graph_reenrich_dispatch(jobs.append)
    install_graph_add(service, graph, ids)
    delete_through_the_api(service, monkeypatch, "mem-a")  # schedules the sibling's re-enrichment
    assert [job["memory_id"] for job in jobs] == ["mem-c"]
    real_add = service._memory.graph.add

    def add_while_the_user_deletes(**kwargs):
        service.delete_memory("mem-c")  # the user deletes the survivor while its job extracts
        return real_add(**kwargs)

    service._memory.graph.add = add_while_the_user_deletes
    service.reenrich_memory_graph("mem-c")
    assert episodes_with(graph, SIBLING_TEXT) == []
    assert facts(graph, "tea in the evening") == []
    assert locks_left(graph) == 0
