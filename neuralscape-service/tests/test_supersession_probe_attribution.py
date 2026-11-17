"""The supersession probe only counts edges the probing write's own episode invalidated.

Several graph jobs of one user run at once. The probe's time window also covers an
edge that a concurrent write invalidated, and the old query handed that edge (and so
the concurrent write's freshly stored row) to whichever job finished next, which then
tombstoned a row it had nothing to do with.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from extensions.dreaming.graph_patcher import find_contradicted_edges

T0 = datetime(2026, 11, 16, 23, 19, 21, tzinfo=timezone.utc)
SINCE = T0 - timedelta(seconds=5)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    async def data(self):
        return [dict(r) for r in self._rows]


class _Session:
    def __init__(self, driver):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        self._driver.queries.append(cypher)
        if "RETURN n.valid_at" in cypher:
            cited = [e for e in self._driver.edges if params["new_episode_uuid"] in e["episodes"] and e["valid_at"]]
            return _Result([{"valid_at": e["valid_at"]} for e in cited])
        assert "invalid_at IS NOT NULL" in cypher
        invalidated = [e for e in self._driver.edges if e["invalid_at"] is not None]
        return _Result(
            [{"uuid": e["uuid"], "fact": e["fact"], "episodes": list(e["episodes"]), "invalid_at": e["invalid_at"]} for e in invalidated]
        )


class _Driver:
    def __init__(self, edges):
        self.edges = edges
        self.queries: list[str] = []

    def session(self):
        return _Session(self)


def _edge(uuid, episodes, valid_at, invalid_at=None):
    return {"uuid": uuid, "fact": f"fact {uuid}", "episodes": episodes, "valid_at": valid_at, "invalid_at": invalid_at}


def _probe(driver, episode):
    return asyncio.run(
        find_contradicted_edges(driver, group_id="g", write_started_at=SINCE, new_episode_uuid=episode)
    )


def test_an_edge_invalidated_by_a_concurrent_write_is_not_attributed_to_the_probing_write():
    nine = T0 + timedelta(seconds=6)
    ten = T0 + timedelta(seconds=7)
    edges = [
        # the concurrent write "ep-new": its own edge was born expired by its own second edge
        _edge("born-dead", ["ep-new"], T0, invalid_at=nine),
        _edge("new-live", ["ep-new"], nine),
        # the probing write "ep-pat": an unrelated dated edge
        _edge("pat-live", ["ep-pat"], ten),
    ]
    assert _probe(_Driver(edges), "ep-pat") == []


def test_an_edge_the_probing_write_invalidated_is_kept_and_cites_only_other_episodes():
    when = T0 + timedelta(seconds=9)
    edges = [
        _edge("old-fact", ["ep-old", "ep-new"], T0 - timedelta(days=1), invalid_at=when),
        _edge("new-fact", ["ep-new"], when),
    ]
    out = _probe(_Driver(edges), "ep-new")
    assert [(e["uuid"], e["episodes"]) for e in out] == [("old-fact", ["ep-old"])]
    assert "invalid_at" not in out[0]


def test_only_the_probing_writes_invalidation_survives_when_both_happen_in_the_window():
    mine, theirs = T0 + timedelta(seconds=9), T0 + timedelta(seconds=7)
    edges = [
        _edge("mine-old", ["ep-old"], T0 - timedelta(days=1), invalid_at=mine),
        _edge("theirs-old", ["ep-theirs-old"], T0 - timedelta(days=1), invalid_at=theirs),
        _edge("mine-new", ["ep-me"], mine),
        _edge("theirs-new", ["ep-theirs"], theirs),
    ]
    assert [e["uuid"] for e in _probe(_Driver(edges), "ep-me")] == ["mine-old"]


def test_without_any_dated_edge_for_the_probing_episode_the_window_is_kept_as_before():
    when = T0 + timedelta(seconds=9)
    edges = [_edge("seeded-dead", ["ep-one"], T0 - timedelta(days=1), invalid_at=when)]
    assert [e["uuid"] for e in _probe(_Driver(edges), "ep-new")] == ["seeded-dead"]
