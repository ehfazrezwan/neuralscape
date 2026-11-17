"""The supersession probe only counts edges the probing write's own episode invalidated.

Several graph jobs of one user run at once. The probe's time window also covers an
edge that a concurrent write invalidated, and a query that only looked at the window
handed that edge (and so the concurrent write's freshly stored row) to whichever job
finished next, which then tombstoned a row it had nothing to do with.

Graphiti stores the exact attribution: the ``Episodic`` node of a write lists, in
``entity_edges``, the edges that episode resolved and the edges it invalidated. The
probe keeps an invalidated edge only when it is in the probing episode's own list:
no time comparison, no fallback. The fake graph below evaluates that clause from the
query text, so a query without it lets every edge in the window through.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from extensions.dreaming.graph_patcher import find_contradicted_edges
from tests.test_contradiction_supersession import (  # noqa: F401  (env is a fixture)
    FakeGraph,
    FakeQdrant,
    _enrich,
    _graph_add,
    env,
)

SINCE = datetime.now(timezone.utc) - timedelta(seconds=5)


def _probe(graph, episode):
    return asyncio.run(
        find_contradicted_edges(
            graph, group_id="user--u1", write_started_at=SINCE, new_episode_uuid=episode
        )
    )


def test_concurrent_writes_sharing_a_timestamp_each_get_only_their_own_invalidation():
    """X and Y share an ``occurred_at``, so every ``valid_at`` they produce is equal and no
    time comparison can tell them apart; only the episode's own edge list can."""
    graph = FakeGraph()
    graph.add_edge("old-a", "lives in Paris", ["ep-old-a"])
    graph.add_edge("old-b", "works at Acme", ["ep-old-b"])
    graph.add_edge("x-new", "lives in London", ["ep-x"])
    graph.add_edge("y-new", "works at Globex", ["ep-y"])
    graph.invalidate("old-a", by="ep-x")
    graph.invalidate("old-b", by="ep-y")

    assert [e["uuid"] for e in _probe(graph, "ep-x")] == ["old-a"]
    assert [e["uuid"] for e in _probe(graph, "ep-y")] == ["old-b"]


def test_a_write_with_no_edges_in_its_list_supersedes_nothing_even_in_a_busy_window():
    graph = FakeGraph()
    graph.add_edge("old-a", "lives in Paris", ["ep-old-a"])
    graph.invalidate("old-a", by="ep-concurrent")
    # the probing episode exists but resolved and invalidated nothing
    graph.episodes["ep-quiet"] = []

    assert _probe(graph, "ep-quiet") == []
    # an episode Graphiti never stored an edge list for behaves the same
    assert _probe(graph, "ep-unknown") == []


def test_an_edge_the_probing_write_invalidated_is_kept_and_cites_only_other_episodes():
    graph = FakeGraph()
    graph.add_edge("old-fact", "lives in Paris", ["ep-old"])
    graph.edges["old-fact"]["episodes"].append("ep-new")  # dedup appends the new episode
    graph.invalidate("old-fact", by="ep-new")

    out = _probe(graph, "ep-new")

    assert [(e["uuid"], e["episodes"]) for e in out] == [("old-fact", ["ep-old"])]


def test_an_edge_listed_for_the_episode_but_expired_before_the_write_started_is_old_news():
    graph = FakeGraph()
    graph.add_edge("old-fact", "lives in Paris", ["ep-old"])
    graph.invalidate("old-fact", by="ep-new")
    graph.edges["old-fact"]["expired_at"] = datetime(2020, 1, 1, tzinfo=timezone.utc)

    assert _probe(graph, "ep-new") == []


def _old_row(qdrant, graph):
    qdrant.add(
        "old", user="u1", data="User lives in Paris",
        metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"},
    )
    graph.add_edge("E1", "User lives in Paris", ["ep-old"])


def test_a_genuine_contradiction_supersedes_the_older_fact_and_names_the_newer_memory(env):
    svc, qdrant, graph = env
    _old_row(qdrant, graph)
    qdrant.add("new", user="u1", data="User moved to London")
    qdrant.add("third", user="u1", data="User likes tea",
               metadata={"graph_episode_uuid": "ep-third", "graph_episode_kind": "fact"})
    graph.add_edge("E3", "User likes tea", ["ep-third"])
    svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

    _enrich(svc)

    meta = qdrant.meta("old")
    assert meta["dream_tombstoned"] is True
    assert meta["superseded_by"] == "new"
    assert meta["superseded_by_episode"] == "ep-new"
    assert not qdrant.meta("new").get("dream_tombstoned")
    # an unrelated third write is untouched
    assert not qdrant.meta("third").get("dream_tombstoned")


def test_a_concurrent_writes_invalidation_does_not_tombstone_the_probing_writes_row(env):
    """The old fact is invalidated by write X while write Y (an unrelated fact) is still
    enriching: Y's probe sees the edge in its window, but the edge is not in Y's list."""
    svc, qdrant, graph = env
    _old_row(qdrant, graph)
    qdrant.add("y-row", user="u1", data="User plays chess")

    def _add_for_y(**kwargs):
        graph.invalidate("E1", by="ep-x")  # write X finishes while Y is enriching
        return {"episode_uuid": "ep-y", "episode_name": "name-ep-y"}

    svc._memory.graph.add.side_effect = _add_for_y

    _enrich(svc, memory_id="y-row", content="User plays chess")

    assert not qdrant.meta("old").get("dream_tombstoned")
    assert not qdrant.meta("y-row").get("dream_tombstoned")
    assert "superseded_by" not in qdrant.meta("old")
