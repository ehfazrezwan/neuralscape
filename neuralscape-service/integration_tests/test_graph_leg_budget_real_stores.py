"""GRAPH_LEG_BUDGET_ENABLED against REAL Qdrant / Neo4j (and Gemini embeddings).

Real: the MemoryService, its Qdrant rows, the Graphiti bridge and the Neo4j
graph leg (``_search_graph_for_visibility`` -> ``search_graph`` -> bridge).
Injected: a delay in front of the REAL bridge call (``_run_on_bridge``), which
stands in for a wedged or slow Neo4j without touching the store.  The flag-off
run with the same delay is the contrast: it waits the delay out.
"""

from __future__ import annotations

import threading
import time

import pytest

import memory.search as search_mod

BUDGET_S = 1.5
DELAY_S = 6.0  # well past the budget, well under the 30 s bridge timeout
FACT = "The Quillfeather deployment gate requires a signed manifest before release."
QUERY = "What does the Quillfeather deployment gate require?"


@pytest.fixture
def service(stores, llm_key, unique_ns):
    from memory_service import MemoryService

    svc = MemoryService()
    svc._get_memory()
    ids: list[str] = []
    svc._nsit_ids = ids
    yield svc
    for mid in ids:
        try:
            svc.delete_memory(mid, unique_ns.user_id)
        except Exception:
            pass
    try:
        svc.close()
    except Exception:
        pass


@pytest.fixture
def seeded(service, unique_ns):
    stored = service.store_raw(FACT, unique_ns.user_id, "domain_knowledge", add_to_graph=False)
    service._nsit_ids.extend(m.id for m in stored)
    return {m.id for m in stored}


@pytest.fixture
def delay_bridge(service, monkeypatch):
    """Delay the REAL bridge call; returns an Event set when a delayed call has finished."""
    real = service._run_on_bridge
    finished = threading.Event()

    def delayed(coro, timeout: float = 30.0):
        time.sleep(DELAY_S)
        try:
            return real(coro, timeout)
        finally:
            finished.set()

    monkeypatch.setattr(service, "_run_on_bridge", delayed)
    return finished


def _ids(results):
    return {r.id for r in results if r.source != "graph"}


def test_flag_on_returns_vector_results_within_budget_and_next_search_is_healthy(
    flags, service, seeded, delay_bridge, unique_ns, monkeypatch
):
    flags(graph_leg_budget_enabled=True, graph_leg_budget_s=BUDGET_S)
    before = search_mod._graph_leg_budget_timeouts

    t0 = time.monotonic()
    results = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    elapsed = time.monotonic() - t0

    assert seeded <= _ids(results), "vector results must still be returned"
    assert elapsed < BUDGET_S + 1.0, f"recall waited on the slow graph leg: {elapsed:.2f}s"
    assert elapsed < DELAY_S, "recall must not wait out the injected delay"
    assert search_mod._graph_leg_budget_timeouts == before + 1

    # The late leg finishes on its pool thread without breaking the bridge or the next search.
    assert delay_bridge.wait(timeout=60), "late graph leg never finished"
    monkeypatch.undo()  # drop the injected delay (also restores flags: re-apply them)
    flags(graph_leg_budget_enabled=True, graph_leg_budget_s=BUDGET_S)
    t1 = time.monotonic()
    again = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    assert seeded <= _ids(again)
    assert time.monotonic() - t1 < BUDGET_S + 1.0


def test_flag_off_waits_out_the_slow_graph_leg(flags, service, seeded, delay_bridge, unique_ns):
    """Contrast: same delay, flag off -> the recall takes at least the delay."""
    flags(graph_leg_budget_enabled=False, graph_leg_budget_s=BUDGET_S)
    before = search_mod._graph_leg_budget_timeouts

    t0 = time.monotonic()
    results = service.search(QUERY, unique_ns.user_id, limit=10, include_shared=False)
    elapsed = time.monotonic() - t0

    assert seeded <= _ids(results)
    assert elapsed >= DELAY_S, f"flag off must keep the old join; returned in {elapsed:.2f}s"
    assert search_mod._graph_leg_budget_timeouts == before
