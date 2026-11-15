"""Graph-contradiction supersession (flag ON): a contradicting fact retires the old one.

Fact A is stored and enriched into the graph; a contradicting fact B follows.
Graphiti invalidates A's edge and the service must stop recalling A while B
stays recallable. Black-box: REST only, unique marker, no assertion on LLM
wording. The contradiction itself is the LLM's judgement, so the facts are
deliberately unambiguous; if the edge is not invalidated the scenario fails
with a message saying so rather than passing.
"""

from __future__ import annotations

from nsaccept import eventually, expect_check_fails, negative_control, proves, profiles


def check_recalled(client, marker: str, query: str) -> None:
    texts = [r.get("memory", "") for r in client.search(query, limit=20)]
    assert any(marker in t for t in texts), f"{marker} not recalled ({len(texts)} results)"


def check_gone(client, marker: str, query: str) -> None:
    texts = [r.get("memory", "") for r in client.search(query, limit=20)]
    assert not any(marker in t for t in texts), f"{marker} is still recalled"


def check_graph_has(client, marker: str) -> None:
    r = client.post("/v1/graph/search", json={"query": marker, "limit": 20})
    assert r.status_code == 200, f"POST /v1/graph/search -> {r.status_code}"
    assert marker in r.text, f"graph has no edge mentioning {marker} yet"


def _write(client, content: str) -> None:
    task = client.store_raw(content, "personal_fact", scope="global", visibility="private")
    status = client.poll_task(task, timeout=120)
    assert status.get("status") == "completed", f"write ended as {status}"


@proves("model.contradiction-supersession", "write.remember-raw", "read.hybrid-search-core")
@profiles("graph-supersession")
def test_contradicting_fact_retires_the_old_one(ctx, a):
    marker = f"zq-{ctx.run_id}-sup"
    old = f"{marker} lives in Paris, France."
    new = f"{marker} moved to London and no longer lives in Paris."
    _write(a, old)
    eventually(lambda: check_recalled(a, marker, old), timeout=90, what="old fact recallable")
    # Wait until the graph worker enriched A, otherwise B has nothing to contradict.
    eventually(lambda: check_graph_has(a, marker), timeout=300, interval=5, what="A enriched into the graph")

    _write(a, new)
    eventually(lambda: check_recalled(a, marker, new), timeout=90, what="new fact recallable")

    def old_retired():
        texts = [r.get("memory", "") for r in a.search(old, limit=20)]
        assert any("London" in t for t in texts), "new fact missing"
        assert not any("lives in Paris, France" in t and marker in t for t in texts), (
            "old fact still recalled: Graphiti did not invalidate its edge (LLM judged the "
            "facts compatible) or the supersession hook did not run"
        )

    eventually(old_retired, timeout=300, interval=5, what="old fact superseded")
    check_not_listed_old(a, marker)


def check_not_listed_old(client, marker: str) -> None:
    r = client.get("/v1/memories", params={"limit": 200})
    assert r.status_code == 200, f"GET /v1/memories -> {r.status_code}"
    rows = [m for m in r.json() if marker in m.get("memory", "")]
    assert not any("lives in Paris, France" in m["memory"] for m in rows), "old fact still listed"
    assert any("London" in m["memory"] for m in rows), "new fact not listed"


@proves("model.contradiction-supersession", "read.hybrid-search-core")
@profiles("graph-supersession")
@negative_control
def test_supersession_check_fails_for_a_live_fact(a_fact, a):
    """The 'old fact is gone' check must fail for a fact nothing contradicted."""
    assert a_fact.visible_to_a, "precondition: fact never became searchable"
    expect_check_fails(check_gone, a, a_fact.marker, a_fact.content)
