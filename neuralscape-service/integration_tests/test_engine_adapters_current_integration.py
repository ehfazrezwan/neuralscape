"""Conformance of the current stack against real Qdrant and Neo4j.

Needs both services configured for disposable use (see ``engine_env``; fails instead of
skipping when ``NS_INTEGRATION_REQUIRED`` is set):

  ENGINE_TEST_QDRANT_URL      (or QDRANT_URL)      e.g. http://127.0.0.1:6333
  ENGINE_TEST_NEO4J_URI       (or NEO4J_URI)       e.g. neo4j://127.0.0.1:7687
  ENGINE_TEST_NEO4J_PASSWORD  (or NEO4J_PASSWORD)  password for user ``neo4j`` (default ``neo4j``)

The Neo4j instance is WIPED between checks; point it only at a throwaway
database.  Data is synthetic and no model is called: embeddings are a
deterministic hash and Graphiti's LLM/embedder/reranker are inert stubs.
"""

import threading
import uuid

import pytest

from engine_env import NEO4J_PASSWORD, NEO4J_URI, QDRANT_URL, require_env

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True, scope="module")
def _stores_configured():
    require_env(Qdrant=QDRANT_URL, Neo4j=NEO4J_URI)

from engine_adapters_current import (  # noqa: E402
    CurrentMemoryEngineAdapter,
    GraphitiGraphAdapter,
    QdrantVectorAdapter,
    hashed_bag_embedder,
)
from engine_ports import (  # noqa: E402
    EngineRecord,
    GraphEdge,
    OperationContext,
    run_graph_store_conformance,
    run_memory_engine_conformance,
    run_vector_store_conformance,
)

DIMS = 64


_COLLECTIONS: list = []


def _drop_collections() -> None:
    # Every collection holds open segment files; leaking one per check exhausts
    # the server's file descriptors, so each check drops its predecessors.
    while _COLLECTIONS:
        store = _COLLECTIONS.pop()
        try:
            store.client.delete_collection(store.collection_name)
        except Exception:
            pass


@pytest.fixture(autouse=True, scope="module")
def _cleanup_collections():
    yield
    _drop_collections()


def vector_adapter() -> QdrantVectorAdapter:
    from mem0.vector_stores.qdrant import Qdrant

    _drop_collections()
    store = Qdrant(
        collection_name="ns_engine_test_" + uuid.uuid4().hex[:10],
        embedding_model_dims=DIMS,
        url=QDRANT_URL,
    )
    _COLLECTIONS.append(store)
    return QdrantVectorAdapter(store, hashed_bag_embedder(DIMS))


@pytest.fixture(scope="module")
def graph_runtime():
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
        return Graphiti(
            llm_client=InertLLM(),
            embedder=InertEmbedder(),
            cross_encoder=InertReranker(),
            graph_driver=driver,
        )

    graphiti = bridge.run(build())
    try:
        yield graphiti, bridge.run
    finally:
        # The adapters leave graph content and exclusive-write fence nodes by
        # design.  Leave the shared disposable database empty so another suite
        # that expects a clean database can run against it afterwards.
        async def wipe():
            await graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
            records, _, _ = await graphiti.driver.execute_query("MATCH (n) RETURN count(n) AS n")
            return records[0]["n"]

        try:
            assert bridge.run(wipe()) == 0
        finally:
            bridge.run(graphiti.close())


def graph_factory(graph_runtime):
    graphiti, run = graph_runtime

    def make() -> GraphitiGraphAdapter:
        run(graphiti.driver.execute_query("MATCH (n) DETACH DELETE n"))
        return GraphitiGraphAdapter(graphiti, run)

    return make


def failed(failures):
    return {failure.check for failure in failures}


def test_qdrant_vector_adapter_passes_conformance_on_a_real_server():
    assert run_vector_store_conformance(vector_adapter) == ()


def test_current_engine_vector_half_passes_conformance_on_a_real_server():
    assert run_memory_engine_conformance(lambda: CurrentMemoryEngineAdapter(vector_adapter())) == ()


def test_graphiti_adapter_health_probe_is_live(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    assert all(s.health.value == "healthy" for s in adapter.manifest().operations)


def test_graphiti_adapter_passes_full_graph_conformance(graph_runtime):
    # Support-aware removal replaces Graphiti's native remove_episode, which
    # deletes edges still supported by another source.
    assert run_graph_store_conformance(graph_factory(graph_runtime)) == ()


def _episode(eid, source, edges, revision=1, tenant="tenant-a"):
    from engine_ports import GraphEpisode

    return GraphEpisode(
        episode_id=eid,
        revision=revision,
        tenant_id=tenant,
        source_record_id=source,
        edges=tuple(GraphEdge(edge_id=e, subject_id=s, predicate="likes", object_id=o) for e, s, o in edges),
    )


def _count(graph_runtime, cypher, **params):
    graphiti, run = graph_runtime

    async def query():
        records, _, _ = await graphiti.driver.execute_query(cypher, routing_="r", **params)
        return records[0]["n"]

    return run(query())


@pytest.mark.parametrize("first_removed", ["src-1", "src-2"])
def test_shared_edge_survives_until_its_last_source_is_removed_in_either_order(graph_runtime, first_removed):
    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    from engine_ports import TenantFilter

    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")]))
    adapter.apply_episode(ctx, _episode("ep-2", "src-2", [("e1", "n1", "n2")]))
    other = "src-2" if first_removed == "src-1" else "src-1"
    adapter.remove_by_source(ctx, first_removed, 2)
    assert adapter.resolve_edge(ctx, "e1") is not None
    assert [e.edge_id for e in adapter.neighbors(ctx, "n1", TenantFilter(tenant_id="tenant-a"))] == ["e1"]
    adapter.remove_by_source(ctx, other, 2)
    assert adapter.resolve_edge(ctx, "e1") is None
    assert _count(graph_runtime, "MATCH (n:Entity) RETURN count(n) AS n") == 0


def test_removal_deletes_only_orphaned_entities_and_leaves_no_content(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2"), ("e2", "n1", "n3")]))
    adapter.apply_episode(ctx, _episode("ep-2", "src-2", [("e1", "n1", "n2")]))
    adapter.remove_by_source(ctx, "src-1", 2)
    names = _count(graph_runtime, "MATCH (n:Entity) WHERE n.name = 'n3' RETURN count(n) AS n")
    assert names == 0  # only src-1 supported n3 and e2
    assert _count(graph_runtime, "MATCH (n:Entity) WHERE n.name IN ['n1','n2'] RETURN count(n) AS n") == 2
    assert adapter.resolve_edge(ctx, "e2") is None and adapter.resolve_edge(ctx, "e1") is not None
    assert _count(graph_runtime, "MATCH (e:Episodic) RETURN count(e) AS n") == 1
    # the fence keeps only a hashed key and the removal revision; no mutex node remains
    assert _count(
        graph_runtime,
        "MATCH (f:NsFence) WHERE f.removed_revision = 2 AND size(keys(f)) = 2 RETURN count(f) AS n",
    ) == 1
    assert _count(graph_runtime, "MATCH (f:NsFence) RETURN count(f) AS n") == 1
    assert _count(graph_runtime, "MATCH (l:NsLock) RETURN count(l) AS n") == 0


def test_reapplying_a_higher_revision_replaces_only_that_episode(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")]))
    adapter.apply_episode(ctx, _episode("ep-2", "src-2", [("e1", "n1", "n2")]))
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e3", "n1", "n4")], revision=2))
    assert adapter.resolve_edge(ctx, "e1") is not None  # still supported by src-2
    assert adapter.resolve_edge(ctx, "e3") is not None
    adapter.remove_by_source(ctx, "src-2", 2)
    assert adapter.resolve_edge(ctx, "e1") is None
    assert adapter.resolve_edge(ctx, "e3") is not None


def test_removed_source_can_return_with_a_newer_revision_only(graph_runtime):
    from engine_ports import PortFailure

    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")]))
    adapter.remove_by_source(ctx, "src-1", 3)
    with pytest.raises(PortFailure):
        adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")], revision=3))
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")], revision=4))
    assert adapter.resolve_edge(ctx, "e1") is not None


def _parallel(*calls):
    barrier = threading.Barrier(len(calls))
    results = [None] * len(calls)

    def runner(index, call):
        barrier.wait()
        try:
            results[index] = call()
        except Exception as error:  # noqa: BLE001
            results[index] = error

    threads = [threading.Thread(target=runner, args=(i, c)) for i, c in enumerate(calls)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return results


def _ctx():
    return OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)


def test_concurrent_applies_sharing_an_edge_keep_one_edge_with_all_support(graph_runtime):
    # Reproduced on the previous adapter: concurrent apply lost supporting
    # episodes and created duplicate relationships.
    make = graph_factory(graph_runtime)
    for _ in range(15):
        adapter = make()
        _parallel(
            lambda: adapter.apply_episode(_ctx(), _episode("ep-a", "src-a", [("e1", "n1", "n2")])),
            lambda: adapter.apply_episode(_ctx(), _episode("ep-b", "src-b", [("e1", "n1", "n2")])),
        )
        assert _count(graph_runtime, "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n") == 1
        assert _count(graph_runtime, "MATCH ()-[r:RELATES_TO]->() RETURN size(r.episodes) AS n") == 2
        assert _count(graph_runtime, "MATCH (n:Entity) RETURN count(n) AS n") == 2
        adapter.remove_by_source(_ctx(), "src-a", 2)
        assert adapter.resolve_edge(_ctx(), "e1") is not None
        adapter.remove_by_source(_ctx(), "src-b", 2)
        assert adapter.resolve_edge(_ctx(), "e1") is None
        assert _count(graph_runtime, "MATCH (n:Entity) RETURN count(n) AS n") == 0


def test_many_concurrent_applies_and_removals_converge(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    sources = [f"src-{i}" for i in range(6)]
    _parallel(
        *[
            (lambda s=s: adapter.apply_episode(_ctx(), _episode(f"ep-{s}", s, [("e-shared", "n1", "n2"), (f"e-{s}", "n1", f"x-{s}")])))
            for s in sources
        ]
    )
    assert _count(graph_runtime, "MATCH ()-[r:RELATES_TO {uuid: $u}]->() RETURN size(r.episodes) AS n", u=adapter_edge_uuid("e-shared")) == 6
    _parallel(*[(lambda s=s: adapter.remove_by_source(_ctx(), s, 2)) for s in sources[:5]])
    assert adapter.resolve_edge(_ctx(), "e-shared") is not None  # src-5 still supports it
    assert adapter.resolve_edge(_ctx(), "e-src-5") is not None
    adapter.remove_by_source(_ctx(), "src-5", 2)
    assert _count(graph_runtime, "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n") == 0
    assert _count(graph_runtime, "MATCH (n:Entity) RETURN count(n) AS n") == 0
    assert _count(graph_runtime, "MATCH (e:Episodic) RETURN count(e) AS n") == 0


def adapter_edge_uuid(edge_id):
    from engine_adapters_current import _stable_uuid

    return _stable_uuid("edge", "tenant-a", edge_id)


def test_apply_racing_a_removal_never_survives_the_fence(graph_runtime):
    make = graph_factory(graph_runtime)
    for _ in range(20):
        adapter = make()
        adapter.apply_episode(_ctx(), _episode("ep-0", "S", [("e-old", "n1", "n2")]))
        _parallel(
            lambda: adapter.remove_by_source(_ctx(), "S", 5),
            lambda: adapter.apply_episode(_ctx(), _episode("ep-1", "S", [("e-new", "n1", "n3")], revision=3)),
        )
        assert adapter.resolve_edge(_ctx(), "e-new") is None
        assert adapter.resolve_edge(_ctx(), "e-old") is None
        assert _count(graph_runtime, "MATCH (e:Episodic) RETURN count(e) AS n") == 0


def test_removal_of_a_never_applied_source_still_fences_a_late_apply(graph_runtime):
    from engine_ports import PortFailure

    adapter = graph_factory(graph_runtime)()
    with pytest.raises(PortFailure) as missing:
        adapter.remove_by_source(_ctx(), "NEVER", 9)
    assert missing.value.code.value == "missing_resource"
    with pytest.raises(PortFailure) as stale:
        adapter.apply_episode(_ctx(), _episode("late", "NEVER", [("e9", "n1", "n2")], revision=4))
    assert stale.value.code.value == "stale_revision"
    adapter.apply_episode(_ctx(), _episode("new", "NEVER", [("e9", "n1", "n2")], revision=10))
    assert adapter.resolve_edge(_ctx(), "e9") is not None


def test_repeated_removal_is_safe_and_the_fence_never_lowers(graph_runtime):
    from engine_ports import PortFailure

    adapter = graph_factory(graph_runtime)()
    adapter.apply_episode(_ctx(), _episode("ep-1", "src-1", [("e1", "n1", "n2")]))
    adapter.remove_by_source(_ctx(), "src-1", 5)
    adapter.remove_by_source(_ctx(), "src-1", 5)  # an identical repeat is idempotent
    with pytest.raises(PortFailure):  # an older removal is stale and never lowers the fence
        adapter.remove_by_source(_ctx(), "src-1", 3)
    assert _count(graph_runtime, "MATCH (f:NsFence) WHERE f.removed_revision = 5 RETURN count(f) AS n") == 1
    with pytest.raises(PortFailure):
        adapter.apply_episode(_ctx(), _episode("ep-1", "src-1", [("e1", "n1", "n2")], revision=4))


def test_composed_engine_round_trip_on_real_stores(graph_runtime):
    graph = graph_factory(graph_runtime)()
    engine = CurrentMemoryEngineAdapter(
        vector_adapter(),
        graph,
        lambda r: (GraphEdge(edge_id="edge-" + r.record_id, subject_id="n1", predicate="likes", object_id="n2"),),
    )
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    engine.ingest(ctx, EngineRecord(record_id="rec-1", revision=1, tenant_id="tenant-a", text="red apple"))
    assert graph.resolve_edge(ctx, "edge-rec-1") is not None
    engine.delete(ctx, "rec-1", 2)
    assert engine.resolve(ctx, "rec-1") is None
    assert graph.resolve_edge(ctx, "edge-rec-1") is None


# ---- lock lifecycle, fence privacy, idempotent repeated removal ----------------


def test_locks_are_released_after_each_write_and_only_content_free_fences_remain(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    sources = ["src-alpha", "src-beta", "src-gamma"]
    for s in sources:
        adapter.apply_episode(ctx, _episode(f"ep-{s}", s, [("e-shared", "n1", "n2"), (f"e-{s}", "n1", f"x-{s}")]))
    # Mutual-exclusion nodes are not content: none outlives the write that took them.
    assert _count(graph_runtime, "MATCH (l:NsLock) RETURN count(l) AS n") == 0
    adapter.remove_by_source(ctx, "src-alpha", 2)
    adapter.remove_by_source(ctx, "src-beta", 2)
    assert _count(graph_runtime, "MATCH (l:NsLock) RETURN count(l) AS n") == 0  # mutex nodes never persist
    assert _count(graph_runtime, "MATCH (f:NsFence) RETURN count(f) AS n") == 2  # one content-free fence per erased source
    assert _count(graph_runtime, "MATCH (f:NsFence) WHERE f.removed_revision IS NULL RETURN count(f) AS n") == 0
    # The erased source id is not kept in the clear in any fence key.
    assert _count(graph_runtime, "MATCH (f:NsFence) WHERE f.key CONTAINS 'src-' RETURN count(f) AS n") == 0
    assert _count(graph_runtime, "MATCH (f:NsFence) WHERE f.key CONTAINS 'alpha' OR f.key CONTAINS 'beta' RETURN count(f) AS n") == 0
    # The hashed fence still fences.
    from engine_ports import PortFailure

    with pytest.raises(PortFailure):
        adapter.apply_episode(ctx, _episode("ep-src-alpha", "src-alpha", [("e1", "n1", "n2")], revision=2))


def test_repeated_removal_is_idempotent_and_keeps_the_fence(graph_runtime):
    from engine_ports import PortFailure

    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")]))
    assert adapter.remove_by_source(ctx, "src-1", 5).operation == "removed"
    assert adapter.remove_by_source(ctx, "src-1", 5).operation == "removed"  # a repeat is not an error
    assert adapter.remove_by_source(ctx, "src-1", 7).operation == "removed"  # a later removal raises the fence
    with pytest.raises(PortFailure) as older:
        adapter.remove_by_source(ctx, "src-1", 3)  # an older removal is stale information
    assert older.value.code.value == "stale_revision"
    with pytest.raises(PortFailure):
        adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")], revision=7))
    adapter.apply_episode(ctx, _episode("ep-1", "src-1", [("e1", "n1", "n2")], revision=8))
    assert adapter.resolve_edge(ctx, "e1") is not None


def test_lock_release_survives_heavy_contention_without_losing_support_or_fences(graph_runtime):
    from engine_ports import PortFailure

    make = graph_factory(graph_runtime)
    for _ in range(3):
        adapter = make()
        sources = [f"src-{i}" for i in range(6)]

        def worker(s):
            def run():
                for revision in (1, 2, 3):
                    try:
                        adapter.apply_episode(_ctx(), _episode(f"ep-{s}", s, [("e-shared", "n1", "n2"), (f"e-{s}", "n1", f"x-{s}")], revision=revision))
                    except PortFailure as failure:
                        assert failure.code.value == "stale_revision"
                try:
                    adapter.remove_by_source(_ctx(), s, 9)
                except PortFailure as failure:
                    assert failure.code.value in ("missing_resource", "stale_revision")
            return run

        results = _parallel(*[worker(s) for s in sources])
        assert all(r is None for r in results), results
        assert _count(graph_runtime, "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n") == 0
        assert _count(graph_runtime, "MATCH (n:Entity) RETURN count(n) AS n") == 0
        assert _count(graph_runtime, "MATCH (e:Episodic) RETURN count(e) AS n") == 0
        assert _count(graph_runtime, "MATCH (l:NsLock) RETURN count(l) AS n") == 0
        assert _count(graph_runtime, "MATCH (f:NsFence) RETURN count(f) AS n") == len(sources)


def test_mutex_nodes_do_not_accumulate_across_many_apply_and_remove_cycles(graph_runtime):
    adapter = graph_factory(graph_runtime)()
    ctx = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=10)
    for i in range(40):
        source = f"src-{i}"
        adapter.apply_episode(ctx, _episode(f"ep-{i}", source, [("e-shared", "n1", "n2"), (f"e-{i}", "n1", f"x-{i}")]))
        adapter.remove_by_source(ctx, source, 2)
    assert _count(graph_runtime, "MATCH (l:NsLock) RETURN count(l) AS n") == 0  # was one per removed source
    assert _count(graph_runtime, "MATCH (f:NsFence) RETURN count(f) AS n") == 40  # the tombstone: one hashed, content-free node per erasure
    assert _count(graph_runtime, "MATCH (n) WHERE NOT n:NsFence RETURN count(n) AS n") == 0  # nothing else remains
