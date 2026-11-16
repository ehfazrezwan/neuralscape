"""Graph-leg latency budget (flag ON): a recall answers inside the budget window.

With GRAPH_LEG_BUDGET_ENABLED=true and GRAPH_LEG_BUDGET_S=4, POST /v1/search must
come back with the stored fact in well under the plugin's ~8 s client timeout,
whether or not the graph leg finished in time (a late graph leg is dropped, the
vector results are returned). The wedged case pauses Neo4j on the throwaway
stack (gated like the destructive fault-injection scenarios: needs
NS_ALLOW_DESTRUCTIVE=1 and a compose project named nscampaign*) and always
unpauses it.
"""

from __future__ import annotations

import time

import os

import pytest

from nsaccept import eventually, expect_check_fails, needs, negative_control, proves, profiles

BOUND_S = 8.0  # the plugin's client timeout; budget (4 s) + embed + Qdrant leaves margin


def check_recalled_fast(client, marker: str, query: str) -> None:
    t0 = time.monotonic()
    texts = [r.get("memory", "") for r in client.search(query, limit=20)]
    elapsed = time.monotonic() - t0
    assert any(marker in t for t in texts), f"{marker} not recalled ({len(texts)} results)"
    assert elapsed < BOUND_S, f"recall took {elapsed:.1f}s (bound {BOUND_S}s)"


@proves("read.graph-hybrid-leg", "read.hybrid-search-core", "write.remember-raw")
@profiles("graph-leg-budget")
def test_recall_returns_within_the_graph_budget(ctx, a):
    marker = f"zq-{ctx.run_id}-glb"
    content = f"{marker} the budget fact lives in the vector store."
    task = a.store_raw(content, "personal_fact", scope="global", visibility="private")
    assert a.poll_task(task, timeout=120).get("status") == "completed"
    eventually(lambda: check_recalled_fast(a, marker, content), timeout=90, what="fact recalled within the bound")


@proves("read.graph-hybrid-leg", "read.hybrid-search-core")
@profiles("graph-leg-budget")
@negative_control
def test_budget_check_fails_for_another_users_fact(a_fact, b):
    """The check must fail where the fact is not visible (user B)."""
    assert a_fact.visible_to_a, "precondition: fact never became searchable"
    expect_check_fails(check_recalled_fast, b, a_fact.marker, a_fact.content)


NEO4J = os.environ.get("NS_NEO4J_SERVICE", "neo4j")


@proves("read.graph-hybrid-leg", "read.hybrid-search-core", "write.remember-raw")
@profiles("graph-leg-budget")
@needs("compose")
def test_wedged_graph_does_not_stall_the_recall(ctx, a):
    if os.environ.get("NS_ALLOW_DESTRUCTIVE") != "1" or not (ctx.compose_project or "").startswith("nscampaign"):
        pytest.skip("destructive: needs NS_ALLOW_DESTRUCTIVE=1 and a compose project named nscampaign*")
    marker = f"zq-{ctx.run_id}-glbw"
    content = f"{marker} the wedged-graph fact lives in the vector store."
    task = a.store_raw(content, "personal_fact", scope="global", visibility="private")
    assert a.poll_task(task, timeout=120).get("status") == "completed"
    eventually(lambda: check_recalled_fast(a, marker, content), timeout=90, what="fact recalled before the wedge")
    ctx.compose("pause", NEO4J)
    try:
        # Without the budget this search takes ~30-45 s (bridge timeout); with it, ~budget.
        check_recalled_fast(a, marker, content)
    finally:
        ctx.compose("unpause", NEO4J, check=False)
    eventually(lambda: check_recalled_fast(a, marker, content), timeout=90, what="healthy again after unpause")
