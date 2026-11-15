"""Real-store smoke tests for the two vendored forks (``mem0/``, ``graphiti/``).

The unit suite mocks Qdrant, Neo4j and Gemini, so a sync that changes mem0's
``Memory`` internals, a Qdrant store signature (the ``top_k`` rename broke
``memory/delete.py`` once) or Graphiti's episode pipeline passes every test and
fails in production.  These tests run the fork code paths the service really
uses, against real Qdrant and Neo4j:

* ``_memory_from_config`` -> ``Memory.from_config`` creates the collection with
  the BM25 sparse slot, idempotently;
* ``vector_store.insert`` / ``query_points`` / ``search(top_k=)`` / ``delete``;
* the ``memory.core`` re-attach shim that wires ``MemoryGraph`` back onto
  ``Memory`` (mem0 2.x dropped ``Memory.graph``) and the ``_AsyncBridge``;
* ``MemoryGraph.add`` (one real Graphiti episode, Gemini) -> ``search`` -> an
  edge query by ``group_id``;
* mem0's lazy ``<collection>_entities`` store under a concurrent first touch.

Run it after every ``scripts/sync-upstream.sh`` (the script says so)::

    NS_INTEGRATION_REQUIRED=1 pytest integration_tests/test_fork_smoke.py -v -p no:cacheprovider

Cost: the only LLM spend is ONE ``add_episode`` (``test_graph_episode_round_trip``,
skipped without ``GOOGLE_API_KEY``) plus the embeddings its search needs.  Every
name is unique (``unique_ns``) and removed afterwards.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

import pytest

DIMS = 768  # embedding_dims in Settings.get_mem0_config() for the Gemini embedder


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def service_settings(stores, unique_ns, monkeypatch):
    """Point the service's Settings at the real stores and this test's names.

    ``Settings.get_mem0_config()`` then builds the exact config the service
    hands to ``Memory.from_config``.
    """

    import config
    from qdrant_client import QdrantClient

    s = config.settings
    for name, value in {
        "qdrant_url": stores.qdrant_url,
        "qdrant_collection": unique_ns.qdrant_collection,
        "neo4j_uri": stores.neo4j_uri,
        "neo4j_user": stores.neo4j_user,
        "neo4j_password": stores.neo4j_password,
        "neo4j_database": stores.neo4j_database,
    }.items():
        monkeypatch.setattr(s, name, value)
    if getattr(s, "llm_gateway_enabled", False):
        monkeypatch.setattr(s, "llm_gateway_enabled", False)  # AI Studio path: no gateway side effects
    if not s.google_api_key:
        # Building the clients needs a non-empty key; vector tests never call the LLM.
        monkeypatch.setattr(s, "google_api_key", "unused-no-llm-call")

    def drop_collections():
        client = QdrantClient(url=stores.qdrant_url)
        for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - already gone
                pass

    unique_ns.on_cleanup(drop_collections)
    return s


def _unit_vector(index: int) -> list[float]:
    vec = [0.0] * DIMS
    vec[index] = 1.0
    return vec


@pytest.fixture
def memory(service_settings):
    """A real mem0 ``Memory`` on a fresh collection, built the way the service builds it."""

    from mem0 import Memory
    from memory.core import _memory_from_config

    return _memory_from_config(Memory, service_settings.get_mem0_config())


# --------------------------------------------------------------------------
# Memory.from_config against real Qdrant
# --------------------------------------------------------------------------


def test_memory_from_config_creates_collection_with_bm25_sparse_slot(memory, unique_ns):
    info = memory.vector_store.client.get_collection(unique_ns.qdrant_collection)
    sparse = info.config.params.sparse_vectors or {}
    assert "bm25" in sparse, "fork's Qdrant.create_col lost the BM25 sparse slot"
    assert info.config.params.vectors.size == DIMS
    assert memory.vector_store._has_bm25_slot is True


def test_memory_from_config_on_an_existing_collection_is_idempotent(memory, service_settings, unique_ns):
    """The cold-start path every second process takes: the collection is already there."""

    from mem0 import Memory
    from memory.core import _memory_from_config

    again = _memory_from_config(Memory, service_settings.get_mem0_config())
    assert again.vector_store.collection_name == unique_ns.qdrant_collection
    assert again.vector_store._has_bm25_slot is True


# --------------------------------------------------------------------------
# vector_store insert / query_points / search(top_k) / delete
# --------------------------------------------------------------------------


def test_vector_store_insert_query_search_delete_round_trip(memory, unique_ns):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    vs = memory.vector_store
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    vs.insert(
        vectors=[_unit_vector(0), _unit_vector(1)],
        ids=ids,
        payloads=[
            {"user_id": unique_ns.user_id, "data": "fork smoke canary about zebras"},
            {"user_id": unique_ns.user_id, "data": "fork smoke canary about giraffes"},
        ],
    )
    owner = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=unique_ns.user_id))])

    # The service's dense read path (memory/search.py): raw query_points with `limit=`.
    result = vs.client.query_points(
        collection_name=unique_ns.qdrant_collection,
        query=_unit_vector(0),
        query_filter=owner,
        limit=2,
        with_payload=True,
    )
    points = list(getattr(result, "points", result))
    assert [str(p.id) for p in points][0] == ids[0]
    assert points[0].payload["data"].endswith("zebras")

    # mem0's wrapper, called the way memory/delete.py calls it: `top_k=`, not `limit=`.
    hits = vs.search(query="zebras", vectors=_unit_vector(0), top_k=2, filters={"user_id": unique_ns.user_id})
    assert [str(h.id) for h in hits][0] == ids[0]
    assert len(hits) == 2
    with pytest.raises(TypeError):
        vs.search(query="zebras", vectors=_unit_vector(0), limit=2)  # the pre-2.0 kwarg must stay gone

    vs.delete(ids[0])
    remaining = vs.search(query="zebras", vectors=_unit_vector(0), top_k=5, filters={"user_id": unique_ns.user_id})
    assert [str(h.id) for h in remaining] == [ids[1]]


def test_bm25_sparse_leg_round_trip(memory, unique_ns):
    """insert writes a `bm25` sparse vector and keyword_search finds it by text.

    Needs fastembed's ``Qdrant/bm25`` model (downloaded on first use); a failure
    here is isolated from the dense round trip above.
    """

    vs = memory.vector_store
    if vs._get_bm25_encoder() is None:
        pytest.fail("BM25 encoder (fastembed Qdrant/bm25) is unavailable: hybrid search is dead")
    mid = str(uuid.uuid4())
    vs.insert(
        vectors=[_unit_vector(2)],
        ids=[mid],
        payloads=[{"user_id": unique_ns.user_id, "data": "quillfeather zeppelin maintenance schedule"}],
    )
    record = vs.client.retrieve(unique_ns.qdrant_collection, ids=[mid], with_vectors=True)[0]
    assert "bm25" in record.vector, "insert did not write the BM25 sparse vector"
    hits = vs.keyword_search("quillfeather zeppelin", top_k=3, filters={"user_id": unique_ns.user_id})
    assert hits and str(hits[0].id) == mid


# --------------------------------------------------------------------------
# the lazy <collection>_entities store, concurrent first touch (real Qdrant)
# --------------------------------------------------------------------------


def test_entity_store_concurrent_first_touch(service_settings, unique_ns):
    """fork.mem0.entity-store: concurrent first callers must not 409.

    Two layers, both through real Qdrant: many threads on ONE ``Memory`` get a
    single store (the lock); several ``Memory`` objects (standing in for several
    processes, so no shared lock) all touching it at once lose the create race
    to each other, and that 409 is treated as success.
    """

    from mem0 import Memory
    from memory.core import _memory_from_config

    cfg = service_settings.get_mem0_config()
    mems = [_memory_from_config(Memory, cfg) for _ in range(6)]
    entities = f"{unique_ns.qdrant_collection}_entities"

    # Layer 1: one Memory, many threads.
    one = mems[0]
    gate = threading.Barrier(8)
    seen: list = []
    errors: list = []

    def touch(mem, sink):
        try:
            gate.wait(timeout=20)
            sink.append(mem.entity_store)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=touch, args=(one, seen)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]
    assert errors == []
    assert len(seen) == 8 and all(s is seen[0] for s in seen)

    # Layer 2: the collection now exists; drop it and race several Memory objects for the re-create.
    one.vector_store.client.delete_collection(entities)
    for m in mems:
        m._entity_store = None
    gate = threading.Barrier(len(mems))
    stores_seen: list = []
    threads = [threading.Thread(target=touch, args=(m, stores_seen)) for m in mems]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]
    assert errors == []
    assert len(stores_seen) == len(mems)
    info = one.vector_store.client.get_collection(entities)
    assert "bm25" in (info.config.params.sparse_vectors or {})


# --------------------------------------------------------------------------
# the re-attach shim and the bridge, against real Neo4j
# --------------------------------------------------------------------------


@pytest.fixture
def service(service_settings, unique_ns):
    """``CoreMixin`` is what MemoryService inherits ``_get_memory`` and ``_run_on_bridge`` from."""

    from memory.core import CoreMixin

    svc = CoreMixin()
    svc._get_memory()

    def stop_bridge():
        bridge = svc._bridge
        if bridge is not None:
            bridge._loop.call_soon_threadsafe(bridge._loop.stop)

    unique_ns.on_cleanup(stop_bridge)
    return svc


def test_reattach_shim_wires_the_graphiti_adapter_onto_memory(service):
    from mem0.memory.graphiti_memory import MemoryGraph

    memory = service._get_memory()
    assert isinstance(getattr(memory, "graph", None), MemoryGraph), (
        "mem0 no longer carries Memory.graph and the shim did not attach it: every graph write is a silent no-op"
    )
    assert service._graphiti is memory.graph.graphiti
    assert service._bridge is memory.graph._bridge


def test_bridge_runs_real_neo4j_queries_and_times_out_a_hang(service):
    graphiti = service._get_graphiti()

    async def one():
        records, _, _ = await graphiti.driver.execute_query("RETURN 1 AS n")
        return records[0]["n"]

    assert service._run_on_bridge(one(), timeout=30) == 1

    async def never():
        await asyncio.sleep(3600)

    # The adapter's own bridge call is bounded now (fork patch), not only the service wrapper.
    with pytest.raises(TimeoutError):
        service._bridge.run(never(), timeout=1.0)
    assert service._run_on_bridge(one(), timeout=30) == 1  # the loop survived the cancel


def test_graph_episode_round_trip(llm_key, service, unique_ns):
    """add_episode (real Gemini + Neo4j) -> search -> edge query by group_id.  ONE episode."""

    from graphiti_core.edges import EntityEdge

    # `llm_key` is first so the test skips before the `service` fixture builds Graphiti;
    # the real key is already in Settings (service_settings only fills an empty one).
    group_id = f"{unique_ns.group_prefix}_smoke"
    graph = service._get_memory().graph

    def purge():
        try:
            graph.delete_all({"group_id": group_id})
        except Exception:  # noqa: BLE001
            pass

    unique_ns.on_cleanup(purge)

    result = graph.add(
        data="Alice Marlowe is a staff engineer at Quillfeather Labs. She works at Quillfeather Labs in Lisbon.",
        filters={"user_id": unique_ns.user_id, "group_id": group_id},
        episode_name=f"fork-smoke-{unique_ns.group_prefix}",
    )
    assert result["episode_uuid"] and result["episode_name"].startswith("fork-smoke-")

    found = graph.search("Where does Alice Marlowe work?", {"group_id": group_id}, limit=10)
    assert found, "Graphiti search returned nothing for a fact just ingested"
    assert "quillfeather" in " ".join(str(t.get("fact", "")) + str(t.get("destination", "")) for t in found).lower()

    async def edges_in_group():
        return await EntityEdge.get_by_group_ids(graph.graphiti.driver, group_ids=[group_id], limit=50)

    edges = service._run_on_bridge(edges_in_group(), timeout=60)
    assert edges and all(e.group_id == group_id for e in edges)
