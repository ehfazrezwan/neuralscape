"""GRAPH_LEG_BUDGET_ENABLED against REAL Qdrant / Neo4j (and Gemini).

Real: the MemoryService, its Qdrant rows (in this test's own collection), a
fact written INTO the Graphiti graph (real LLM extraction), the Graphiti bridge
and the graph leg (``_search_graph_for_visibility`` -> ``search_graph`` ->
bridge).  Injected: a delay in front of the REAL ``_run_on_bridge``, standing in
for a wedged or slow Neo4j without touching the store.  The flag-off run with
the same delay is the contrast: it waits the delay out.

The bounds below time the whole ``search()``: a real Gemini query embed plus a
Qdrant query ride inside them, hence the generous margin over the budget.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (must precede any mem0/graphiti import)

import threading
import time

import pytest

import memory.search as search_mod

BUDGET_S = 1.5
MARGIN_S = 3.0  # query embed + Qdrant round trip on top of the budget
DELAY_S = 12.0  # well past budget + margin, well under the 30 s bridge timeout
FACT = "The Quillfeather deployment gate requires a signed manifest before release."
QUERY = "What does the Quillfeather deployment gate require?"


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


@pytest.fixture
def service(stores, llm_key, unique_ns, monkeypatch):
    import config

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

    from memory_service import MemoryService
    from qdrant_client import QdrantClient

    svc = MemoryService()
    svc._get_memory()
    assert svc._graphiti is not None and svc._bridge is not None, "graph store not initialised"
    group = f"user--{unique_ns.user_id}"

    def cleanup():
        try:
            _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group)
        finally:
            client = QdrantClient(url=stores.qdrant_url)
            for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
                if client.collection_exists(name):
                    client.delete_collection(name)
            svc.close()

    unique_ns.on_cleanup(cleanup)
    return svc


@pytest.fixture
def seeded(service, unique_ns):
    """One fact in Qdrant AND in the graph; waits until the graph holds a live edge."""
    stored = service.store_raw(FACT, unique_ns.user_id, "domain_knowledge", add_to_graph=True)
    group = f"user--{unique_ns.user_id}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        edges = _cypher(
            service,
            "MATCH ()-[r:RELATES_TO {group_id: $g}]->() WHERE r.expired_at IS NULL RETURN r.uuid AS uuid",
            g=group,
        )
        if edges:
            break
        time.sleep(3)
    else:
        pytest.fail("the graph never received an edge for the seeded fact within 180 s")
    return {m.id for m in stored}


@pytest.fixture
def delay_bridge(service, monkeypatch):
    """Delay the REAL bridge call while ``.active``; ``.finished`` is set when a delayed call ended."""
    real = service._run_on_bridge

    class Delay:
        active = True
        finished = threading.Event()

    def delayed(coro, timeout: float = 30.0):
        if not Delay.active:
            return real(coro, timeout)
        time.sleep(DELAY_S)
        try:
            return real(coro, timeout)
        finally:
            Delay.finished.set()

    monkeypatch.setattr(service, "_run_on_bridge", delayed)
    return Delay


def _vector_ids(results):
    return {r.id for r in results if r.source != "graph"}


def _graph_rows(results):
    return [r for r in results if r.source == "graph"]


def test_flag_on_returns_vector_results_within_budget_and_next_search_has_graph_rows(
    flags, service, seeded, delay_bridge, unique_ns
):
    flags(graph_leg_budget_enabled=True, graph_leg_budget_s=BUDGET_S)
    before = search_mod._graph_leg_budget_timeouts

    t0 = time.monotonic()
    results = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    elapsed = time.monotonic() - t0

    assert seeded <= _vector_ids(results), "vector results must still be returned"
    assert not _graph_rows(results), "the delayed graph leg must have been dropped"
    assert elapsed < BUDGET_S + MARGIN_S, f"recall waited on the slow graph leg: {elapsed:.2f}s"
    assert elapsed < DELAY_S, "recall must not wait out the injected delay"
    assert search_mod._graph_leg_budget_timeouts == before + 1

    # The late leg finishes on its pool thread without breaking the bridge or the next search.
    assert delay_bridge.finished.wait(timeout=90), "late graph leg never finished"
    delay_bridge.active = False
    flags(graph_leg_budget_enabled=True, graph_leg_budget_s=30.0)  # a healthy leg must fit; the budget is not under test now
    again = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    assert seeded <= _vector_ids(again)
    assert _graph_rows(again), "after the late leg ended, the next search must return graph-sourced results"


def test_flag_off_waits_out_the_slow_graph_leg(flags, service, seeded, delay_bridge, unique_ns):
    """Contrast: same delay, flag off -> the recall takes at least the delay and gets graph rows."""
    flags(graph_leg_budget_enabled=False, graph_leg_budget_s=BUDGET_S)
    before = search_mod._graph_leg_budget_timeouts

    t0 = time.monotonic()
    results = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    elapsed = time.monotonic() - t0

    assert seeded <= _vector_ids(results)
    assert elapsed >= DELAY_S, f"flag off must keep the old join; returned in {elapsed:.2f}s"
    assert _graph_rows(results), "flag off waits for the graph leg and merges it"
    assert search_mod._graph_leg_budget_timeouts == before
