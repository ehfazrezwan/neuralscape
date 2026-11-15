"""Graph-leg latency budget (flag ON): a recall answers inside the budget window.

With GRAPH_LEG_BUDGET_ENABLED=true and GRAPH_LEG_BUDGET_S=4, POST /v1/search must
come back with the stored fact in well under the plugin's ~8 s client timeout,
whether or not the graph leg finished in time (a late graph leg is dropped, the
vector results are returned). The wedged-graph case (Neo4j paused) needs
container control a black-box scenario does not have: it is documented in the PR
as a manual "See it work" run and proven by the integration test
`integration_tests/test_graph_leg_budget_real_stores.py`.
"""

from __future__ import annotations

import time

from nsaccept import eventually, expect_check_fails, negative_control, proves, profiles

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
