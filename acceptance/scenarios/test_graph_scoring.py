"""Graph edges receive a real cosine score in unified recall.

The graph store (Graphiti, 1024-d by default) and the vector store (768-d)
use different embedding lengths. Recall must still score each graph edge
against the query, so a `source: graph` row carries a numeric `score`
instead of `null` (which ranked every graph row last).

The fact is written through the LLM-free raw route; the graph worker then
extracts edges from it asynchronously, so the check is retried within a
bounded window. Assertions are on source/score only, never on wording.
"""

from __future__ import annotations

from conftest import Fact
from nsaccept import eventually, expect_check_fails, negative_control, proves


def check_graph_row_has_numeric_score(client, fact: Fact) -> None:
    results = client.search(fact.content, project_id=fact.project_id, limit=20)
    graph_rows = [r for r in results if r.get("source") == "graph"]
    assert graph_rows, f"no graph rows yet among {len(results)} results"
    scored = [r for r in graph_rows if isinstance(r.get("score"), (int, float))]
    assert scored, f"{len(graph_rows)} graph rows, none with a numeric score: {[r.get('score') for r in graph_rows]}"
    assert all(0.0 <= r["score"] <= 1.0 for r in scored), "graph score outside [0, 1]"
    # The query IS the stored fact's text, and the edge is extracted from it, so
    # the best graph row must be a close semantic match. A prefix comparison of
    # unrelated halves (the failure mode if the MRL premise were false) scores
    # near zero or at an arbitrary mid value; a real match is well above 0.5.
    # The exact value is checked against directly computed real vectors in
    # integration_tests/test_graph_embedding_mrl_gemini.py (a black-box scenario
    # cannot embed text itself).
    best = max(r["score"] for r in scored)
    assert best >= 0.5, f"best graph score {best:.3f} is too low for a row derived from the query text"


@proves(
    "fork.graphiti.embedder-dim-mismatch",
    "graph.embedder-client",
    "fork.graphiti.graph-embedder",
    "fork.mem0.gemini-embedder",
)
def test_graph_rows_get_numeric_scores(a, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    eventually(
        lambda: check_graph_row_has_numeric_score(a, a_fact),
        timeout=240,
        what="a scored graph row for A's fact",
    )


@proves("fork.graphiti.embedder-dim-mismatch", "graph.embedder-client")
@negative_control
def test_graph_score_check_fails_for_user_b(b, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    expect_check_fails(check_graph_row_has_numeric_score, b, a_fact)
