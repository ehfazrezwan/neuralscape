"""Support-aware graph removal (``GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED``) on the service's delete and correct
paths, against a REAL Neo4j and a REAL Qdrant.

The service runs with a real Graphiti (Neo4j driver; the LLM, embedder and reranker are inert stand-ins, no
model is called) on a real async bridge.  The memory rows are real Qdrant points in a collection of their own
(the service's own filters, payload patches and deletes run against the server); the graph is seeded
structurally in Graphiti's own schema: two of alice's memories, A and B, whose episodes both support the edge
X->Y, while only A supports X->Z.

* ``DELETE /v1/memories/{A}`` (the API route): with the flag on the shared fact X->Y stays live, now supported
  by B only; X->Z and the entity Z (referenced by nothing else) are removed; the mentioned entities'
  summaries are cleared; A's episode is gone; no lock node is left.  With the flag off the legacy
  most-restrictive cascade expires X->Y (pinned as the documented difference).
* Correction: re-enriching A with corrected text that still asserts X->Y removes A's superseded episode after
  the new one exists, so X->Y keeps the new support and B's, and X->Z goes.
* One conversation, several memories: conversation extraction stamps every fact it creates with the same
  episode.  A correction or delete of one of them removes that episode anyway (the removed memory's facts must
  not stay retrievable; facts another episode supports stay), and every surviving sibling is re-enriched into a
  fresh episode of its own, so its facts come back.  A correction whose removal fails records the superseded
  episode on the row and the next correction or the delete retries it, in the group the episode lives in.
* A memory deleted while it is being enriched leaves nothing in the graph: through the graph worker job, with
  ``GRAPH_DELETE_RACE_CLEANUP_ENABLED`` on and off and the support-aware removal on and off; the job returns the
  same skip every time and never takes the lossy fallback.

The durable (ledger job) form of the sibling re-enrichment belongs with the write producer (a later slice).
A missing store fails the test on the gate and skips it elsewhere (``integration_tests/README.md``).
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from memory.groups import _build_group_id
from memory_service import MemoryService

OWNER = f"nsit-user-{uuid.uuid4().hex[:10]}"
GROUP = _build_group_id("private", OWNER, None, None)
MEM_A, MEM_B, MEM_C, MEM_NEW = (str(uuid.uuid4()) for _ in range(4))


@pytest.fixture(scope="module")
def graph(stores):
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
        driver = Neo4jDriver(uri=stores.neo4j_uri, user=stores.neo4j_user, password=stores.neo4j_password,
                             database=stores.neo4j_database)
        return Graphiti(llm_client=InertLLM(), embedder=InertEmbedder(), cross_encoder=InertReranker(),
                        graph_driver=driver)

    graphiti = bridge.run(build())
    try:
        yield graphiti, bridge
    finally:
        # Only this module's own group: the Neo4j is shared with the other integration tests.
        bridge.run(graphiti.driver.execute_query("MATCH (n {group_id: $g}) DETACH DELETE n", g=GROUP))
        bridge.run(graphiti.close())


@pytest.fixture(scope="module")
def rows_collection(stores):
    """A Qdrant collection of this module's own: the memory rows of the scenarios."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    name = f"nsit_support_aware_{uuid.uuid4().hex[:10]}"
    client = QdrantClient(url=stores.qdrant_url)
    client.create_collection(name, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    try:
        yield client, name
    finally:
        client.delete_collection(name)


class RowStore:
    """The vector row store behind ``MemoryService``: real Qdrant points with mem0's payload shape
    (``data``, ``user_id``, ``metadata``); ``get``/``delete`` as mem0's ``Memory`` offers them."""

    def __init__(self, client, collection):
        self.client, self.collection = client, collection
        self.vector_store = SimpleNamespace(client=client)
        self.graph = SimpleNamespace(add=None)

    def put(self, memory_id, text, metadata):
        from qdrant_client.models import PointStruct

        self.client.upsert(self.collection, points=[PointStruct(
            id=memory_id, vector=[1.0, 0.0], payload={"data": text, "user_id": OWNER, "metadata": dict(metadata)})])

    def meta(self, memory_id):
        (point,) = self.client.retrieve(self.collection, ids=[memory_id], with_payload=True)
        return (point.payload or {}).get("metadata") or {}

    def get(self, memory_id):
        points = self.client.retrieve(self.collection, ids=[memory_id], with_payload=True)
        if not points:
            return None
        payload = points[0].payload or {}
        return {"id": memory_id, "memory": payload.get("data", ""), "user_id": payload.get("user_id", OWNER),
                "metadata": dict(payload.get("metadata") or {})}

    def delete(self, memory_id):
        from qdrant_client.models import PointIdsList

        self.client.delete(self.collection, points_selector=PointIdsList(points=[memory_id]))
        return {"message": "Memory deleted successfully!"}


class GraphService(MemoryService):
    def __init__(self, graphiti, bridge, rows):
        self._memory = rows
        self._graphiti = graphiti
        self._bridge = bridge
        self._filter_indexes_ok = False

    def _get_memory(self):
        return self._memory


def q(graphiti, bridge, cypher, **params):
    records, _, _ = bridge.run(graphiti.driver.execute_query(cypher, **params))
    return [dict(r) for r in records]


def seed(graphiti, bridge):
    q(graphiti, bridge, "MATCH (n {group_id: $g}) DETACH DELETE n", g=GROUP)
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


@pytest.fixture(autouse=True)
def _clean_rows(rows_collection, flags):
    from qdrant_client.models import PointIdsList

    client, name = rows_collection
    flags(qdrant_collection=name)
    points, _ = client.scroll(name, limit=1000, with_payload=False)
    if points:
        client.delete(name, points_selector=PointIdsList(points=[p.id for p in points]))
    yield


def make_service(graph, ids, rows_collection):
    graphiti, bridge = graph
    service = GraphService(graphiti, bridge, RowStore(*rows_collection))
    for memory, episode, text in ((MEM_A, ids["EA"], "memory A text"), (MEM_B, ids["EB"], "memory B text")):
        service._memory.put(memory, text, {"owner_user_id": OWNER, "visibility": "private",
                                           "graph_episode_uuid": episode})
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


def delete_through_the_api(service, monkeypatch, flags, memory_id):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "_service", service)
    flags(default_user_id=OWNER)
    flags(ledger_enabled=False)
    flags(neuralscape_auth_disabled=True)
    flags(neuralscape_user_token_secret="")
    flags(neuralscape_api_key="")
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


def test_api_delete_keeps_a_fact_another_memory_still_supports(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    result = delete_through_the_api(make_service(graph, ids, rows_collection), monkeypatch, flags, MEM_A)
    assert result.get("graph_cascade") == "resolved"
    assert_support_aware_outcome(graph, ids)


def test_with_the_flag_off_the_legacy_cascade_still_expires_the_shared_fact(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=False)
    ids = seed(*graph)
    delete_through_the_api(make_service(graph, ids, rows_collection), monkeypatch, flags, MEM_A)
    assert edge(graph, ids["XY"])["expired"] is True  # most-restrictive: expired despite B's support
    only_a = edge(graph, ids["XZ"])
    assert only_a is None or only_a["expired"] is True  # gone with the sole-mentioned entity, or expired


def test_a_correction_replaces_the_superseded_episode_and_keeps_still_asserted_facts(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
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
    assert service.enrich_graph("corrected A", OWNER, None, "private", MEM_A) is True
    shared = edge(graph, ids["XY"])
    assert shared is not None and shared["expired"] is False
    assert sorted(shared["episodes"]) == sorted([ids["EB"], new_episode])
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert not exists(graph, "Episodic", ids["EA"])
    assert service._memory.meta(MEM_A)["graph_episode_uuid"] == new_episode
    assert locks_left(graph) == 0


def test_with_the_flag_off_a_correction_leaves_the_superseded_episode(graph, rows_collection, monkeypatch, flags):
    """Pinned legacy behavior: the old episode keeps its facts live (the defect the flag fixes)."""
    flags(graph_support_aware_removal_enabled=False)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    service._memory.graph.add = lambda **kwargs: {"episode_uuid": str(uuid.uuid4())}
    service.enrich_graph("corrected A", OWNER, None, "private", MEM_A)
    assert exists(graph, "Episodic", ids["EA"]) and edge(graph, ids["XZ"])["expired"] is False


SIBLING_TEXT = "the C fact of the A conversation"


def add_sibling(service, graph, ids, memory_id=MEM_C, *, by_name_only=False):
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
    service._memory.put(memory_id, SIBLING_TEXT, metadata)


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


def correct(service, graph, ids, memory_id=MEM_A, visibility="private"):
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
    row = service._memory.meta(MEM_C)
    assert row["graph_episode_uuid"] == fresh and row.get("graph_reenrich_pending") is None  # ... and back
    assert ids["EB"] in edge(graph, ids["XY"])["episodes"] and edge(graph, ids["XY"])["expired"] is False
    assert locks_left(graph) == 0


def test_a_delete_removes_the_shared_episode_and_re_enriches_the_surviving_sibling(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    result = delete_through_the_api(service, monkeypatch, flags, MEM_A)
    assert result.get("graph_cascade") == "resolved"
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)
    assert edge(graph, ids["XY"])["episodes"] == [ids["EB"]]


def test_a_correction_removes_the_shared_episode_and_re_enriches_the_surviving_sibling(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    new_episode = correct(service, graph, ids)
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)
    assert sorted(edge(graph, ids["XY"])["episodes"]) == sorted([ids["EB"], new_episode])
    assert service._memory.meta(MEM_A)["graph_episode_uuid"] == new_episode
    assert not service._memory.meta(MEM_A).get("graph_episode_superseded")


def test_a_sibling_referencing_the_episode_by_name_only_is_re_enriched_too(graph, rows_collection, monkeypatch, flags):
    """A re-run conversation stamps only the deterministic episode name on its new rows."""
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    service._memory.client.set_payload(service._memory.collection, payload={"graph_episode_name": "ep-a"}, points=[MEM_A], key="metadata")
    add_sibling(service, graph, ids, by_name_only=True)
    install_graph_add(service, graph, ids)
    delete_through_the_api(service, monkeypatch, flags, MEM_A)
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)


def test_until_re_enrichment_runs_the_sibling_waits_marked_and_its_facts_are_absent(graph, rows_collection, monkeypatch, flags):
    """With a durable dispatch installed the delete only schedules the sibling: availability loss, no exposure."""
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    jobs = []
    service.set_graph_reenrich_dispatch(jobs.append)
    assert service._memory.meta(MEM_C)["graph_episode_uuid"] == ids["EA"]  # the sibling starts out on the shared episode
    delete_through_the_api(service, monkeypatch, flags, MEM_A)
    assert [job["memory_id"] for job in jobs] == [MEM_C] and jobs[0]["removed_episode"] == ids["EA"]
    row = service._memory.meta(MEM_C)
    # The row's episode reference is cleared (Qdrant may drop the key or keep it null): read it with ``get``.
    assert row["graph_reenrich_pending"] == ids["EA"] and row.get("graph_episode_uuid") is None
    assert not exists(graph, "Episodic", ids["EA"]) and facts(graph, "tea in the evening") == []
    assert service.reenrich_memory_graph(MEM_C) == {"memory_ids": [MEM_C]}  # the job, when it runs
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)


def test_with_the_flag_off_a_delete_still_removes_a_shared_episode(graph, rows_collection, monkeypatch, flags):
    """Pinned legacy behavior: the most-restrictive cascade does not look at sibling rows."""
    flags(graph_support_aware_removal_enabled=False)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    delete_through_the_api(service, monkeypatch, flags, MEM_A)
    assert not exists(graph, "Episodic", ids["EA"]) and edge(graph, ids["XY"])["expired"] is True


def test_a_ghost_episode_id_in_an_edge_does_not_keep_the_edge_alive(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    graphiti, bridge = graph
    q(graphiti, bridge, "MATCH ()-[r:RELATES_TO {uuid: $u}]->() SET r.episodes = r.episodes + $ghost",
      u=ids["XZ"], ghost=str(uuid.uuid4()))  # an id no episode has (e.g. one removed out of band)
    delete_through_the_api(make_service(graph, ids, rows_collection), monkeypatch, flags, MEM_A)
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert_support_aware_outcome(graph, ids)


def test_a_failed_correction_removal_is_recorded_and_retried_by_the_next_correction(graph, rows_collection, monkeypatch, flags):
    import engine_adapters_current

    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    real = engine_adapters_current.GraphitiGraphAdapter.remove_episodes
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("neo4j unavailable")))
    first = correct(service, graph, ids)
    assert exists(graph, "Episodic", ids["EA"])  # not removed now ...
    assert service._memory.meta(MEM_A)["graph_episode_superseded"] == [ids["EA"]]  # ... but kept
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes", real)
    second = correct(service, graph, ids)
    assert not exists(graph, "Episodic", ids["EA"]) and not exists(graph, "Episodic", first)
    assert edge(graph, ids["XZ"]) is None
    assert sorted(edge(graph, ids["XY"])["episodes"]) == sorted([ids["EB"], second])
    assert service._memory.meta(MEM_A)["graph_episode_superseded"] == []
    assert locks_left(graph) == 0


def test_a_failed_correction_removal_is_retried_by_the_delete(graph, rows_collection, monkeypatch, flags):
    import engine_adapters_current

    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    real = engine_adapters_current.GraphitiGraphAdapter.remove_episodes
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes",
                        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("neo4j unavailable")))
    first = correct(service, graph, ids)
    monkeypatch.setattr(engine_adapters_current.GraphitiGraphAdapter, "remove_episodes", real)
    delete_through_the_api(service, monkeypatch, flags, MEM_A)
    assert not exists(graph, "Episodic", ids["EA"]) and not exists(graph, "Episodic", first)
    assert_support_aware_outcome_without_a(graph, ids)


def assert_support_aware_outcome_without_a(graph, ids):
    shared = edge(graph, ids["XY"])
    assert shared == {"episodes": [ids["EB"]], "expired": False}
    assert edge(graph, ids["XZ"]) is None and not exists(graph, "Entity", ids["Z"])
    assert locks_left(graph) == 0


def test_a_correction_into_another_group_removes_the_superseded_episode_where_it_lives(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
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


DELETED_TEXT = "oolong in the afternoon"


def arm_the_delete_during_extraction(service, graph, ids, after_the_delete=None):
    """The user deletes the memory while the graph write runs: the delete finds no episode yet, so it removes nothing.

    That delete itself takes the lossy substring fallback (nothing to resolve); ``after_the_delete`` is called
    once it has returned, so a test can start counting fallbacks from what the job does afterwards.
    """
    service._memory.put(MEM_NEW, DELETED_TEXT, {"owner_user_id": OWNER, "visibility": "private"})
    graphiti, bridge = graph

    def add_while_the_user_deletes(**kwargs):
        result = service.delete_memory(MEM_NEW)
        assert result.get("graph_cascade") == "unresolved"
        if after_the_delete is not None:
            after_the_delete()
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


@pytest.mark.parametrize("support_aware", [True, False], ids=["support-aware", "legacy"])
@pytest.mark.parametrize("race_cleanup", [True, False], ids=["race-cleanup-on", "race-cleanup-off"])
def test_a_memory_deleted_during_its_graph_job_leaves_no_episode_and_the_job_resolves(
    graph, rows_collection, monkeypatch, support_aware, race_cleanup, flags
):
    """The graph worker job (``process_graph_enrichment``) around the real ``enrich_graph``.

    With ``GRAPH_DELETE_RACE_CLEANUP_ENABLED`` off the worker's own post-enrichment check removes the episode; on, the
    enrichment removes it and the worker takes that as done.  Either way the job returns the same skip, raises
    nothing, never falls back to the lossy substring expiry, and the deleted text is not in the graph.
    """
    import asyncio

    import worker

    flags(graph_support_aware_removal_enabled=support_aware, graph_delete_race_cleanup_enabled=race_cleanup)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    substring_fallbacks = []
    # The user's own delete (nothing to resolve yet) uses the fallback; only what the job does afterwards counts.
    arm_the_delete_during_extraction(service, graph, ids, after_the_delete=substring_fallbacks.clear)
    monkeypatch.setattr(service, "_expire_graph_edges_for_memory", lambda mem: substring_fallbacks.append(mem))
    outcome = asyncio.run(worker.process_graph_enrichment({"service": service}, MEM_NEW, DELETED_TEXT, OWNER, None, "private"))
    assert outcome == {"memory_id": MEM_NEW, "enriched": False, "skipped": "changed_during_enrichment"}
    assert substring_fallbacks == []  # no lossy fallback, whichever side removed the episode
    assert episodes_with(graph, DELETED_TEXT) == []  # the deleted memory's text is not in the graph
    assert exists(graph, "Episodic", ids["EA"]) and exists(graph, "Episodic", ids["EB"])  # others untouched
    assert locks_left(graph) == 0


@pytest.mark.parametrize("support_aware", [True, False], ids=["support-aware", "legacy"])
def test_an_inline_enrichment_of_a_memory_deleted_meanwhile_is_cleaned_up_with_the_flag_on(
    graph, rows_collection, support_aware, flags
):
    flags(graph_support_aware_removal_enabled=support_aware, graph_delete_race_cleanup_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    arm_the_delete_during_extraction(service, graph, ids)
    done = {}
    assert service.enrich_graph(DELETED_TEXT, OWNER, None, "private", MEM_NEW, outcome=done) is True
    assert done == {"cleaned_up": True}
    assert episodes_with(graph, DELETED_TEXT) == []
    assert exists(graph, "Episodic", ids["EA"]) and exists(graph, "Episodic", ids["EB"])
    assert locks_left(graph) == 0


def test_with_the_flag_off_an_inline_enrichment_leaves_the_cleanup_to_its_caller(graph, rows_collection, flags):
    """The default path is unchanged: ``enrich_graph`` alone does not remove the episode (the worker job does)."""
    flags(graph_delete_race_cleanup_enabled=False)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    arm_the_delete_during_extraction(service, graph, ids)
    done = {}
    assert service.enrich_graph(DELETED_TEXT, OWNER, None, "private", MEM_NEW, outcome=done) is True
    assert done == {}
    assert len(episodes_with(graph, DELETED_TEXT)) == 1


def test_a_survivor_deleted_during_its_re_enrichment_leaves_nothing_in_the_graph(graph, rows_collection, monkeypatch, flags):
    flags(graph_support_aware_removal_enabled=True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    jobs = []
    service.set_graph_reenrich_dispatch(jobs.append)
    install_graph_add(service, graph, ids)
    delete_through_the_api(service, monkeypatch, flags, MEM_A)  # schedules the sibling's re-enrichment
    assert [job["memory_id"] for job in jobs] == [MEM_C]
    real_add = service._memory.graph.add

    def add_while_the_user_deletes(**kwargs):
        service.delete_memory(MEM_C)  # the user deletes the survivor while its job extracts
        return real_add(**kwargs)

    service._memory.graph.add = add_while_the_user_deletes
    service.reenrich_memory_graph(MEM_C)
    assert episodes_with(graph, SIBLING_TEXT) == []
    assert facts(graph, "tea in the evening") == []
    assert locks_left(graph) == 0


# ---- the episode-reference check the shared-episode handling rests on, against a real Qdrant


def test_the_reference_check_against_a_real_qdrant(graph, rows_collection):
    import time

    from qdrant_client.models import PointStruct

    client, collection = rows_collection
    service = GraphService(*graph, RowStore(client, collection))
    ids = [str(uuid.uuid4()) for _ in range(70)]  # more than one page of references
    named = str(uuid.uuid4())
    client.upsert(collection, points=[
        PointStruct(id=i, vector=[1.0, 0.0], payload={"data": i, "metadata": {"graph_episode_uuid": "ep-1"}})
        for i in ids
    ] + [PointStruct(id=named, vector=[1.0, 0.0], payload={"metadata": {"graph_episode_name": "ep-name"}})])
    check = service._episode_survivors
    assert [m for m, _ in check("ep-1", None, tuple(ids[:-1]))] == [ids[-1]]  # outside the batch
    assert check("ep-1", None, tuple(ids)) == []
    assert [m for m, _ in check("ep-1", "ep-name", tuple(ids))] == [named]  # a name-only row references it
    assert check("ep-unknown", None, ()) == []
    assert check("ep-1", None, tuple(ids[:-1]))[0][1]["data"] == ids[-1]  # payload carried for re-enrichment
    wanted = {"metadata.graph_episode_uuid", "metadata.graph_episode_name"}
    deadline = time.monotonic() + 10  # the index is requested without waiting for the build
    while not wanted <= set(client.get_collection(collection).payload_schema) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert wanted <= set(client.get_collection(collection).payload_schema)  # indexed
