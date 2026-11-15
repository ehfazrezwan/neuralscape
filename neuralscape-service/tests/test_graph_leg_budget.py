"""GRAPH_LEG_BUDGET_ENABLED: a slow graph leg must not stall a recall.

Defect: search() joined the Graphiti leg with a 45 s cap (inner bridge call
30 s) even though the vector results were ready, so REST/MCP callers with
short client timeouts got nothing.  With the flag on, the join waits at most
GRAPH_LEG_BUDGET_S measured from when the leg was SUBMITTED, then returns the
vector-only results.  With the flag off nothing changes.
"""

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import qdrant_client  # noqa: F401  pre-import: search() lazily imports it (~2 s cold) and the timings below must not include that
import memory.search as search_mod
from config import Settings, settings
from memory_service import MemoryService


def _qresult(hits):
    r = MagicMock()
    r.points = hits
    return r


def _hit(id, score, text):
    h = MagicMock()
    h.id = id
    h.score = score
    h.payload = {"data": text, "created_at": "2026-01-01T00:00:00+00:00", "metadata": {}}
    return h


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult(
        [_hit("v0", 0.9, "vector fact")]
    )
    return svc


def _slow_graph(delay_s, edges=2):
    """A graph leg that blocks for delay_s, then returns edges. Records completion."""
    done = threading.Event()

    def fake(*_a, **_kw):
        time.sleep(delay_s)
        done.set()
        return {
            "edges": [{"uuid": f"g{i}", "fact": f"graph fact {i}"} for i in range(edges)],
            "nodes": [], "episodes": [], "communities": [],
        }

    return fake, done


class TestBudgetOn:
    def test_slow_graph_returns_vector_results_within_budget(self, service, monkeypatch, caplog):
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", True)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 0.5)
        fake, done = _slow_graph(3.0)
        before = search_mod._graph_leg_budget_timeouts
        with patch.object(service, "_search_graph_for_visibility", side_effect=fake):
            with caplog.at_level(logging.WARNING, logger="memory.search"):
                t0 = time.monotonic()
                results = service.search(query="q", user_id="u1", limit=5)
                elapsed = time.monotonic() - t0
        assert elapsed < 0.5 + 1.0, f"recall waited for the graph leg: {elapsed:.2f}s"
        assert [r.id for r in results] == ["v0"]
        assert search_mod._graph_leg_budget_timeouts == before + 1
        warned = [r for r in caplog.records if getattr(r, "event", "") == "graph_leg_budget_exceeded"]
        assert len(warned) == 1 and warned[0].budget_s == 0.5
        # The late leg is left to finish on its pool thread; it does not break anything.
        assert done.wait(timeout=10)

    def test_budget_counts_from_submit_not_from_join(self, service, monkeypatch):
        """Vector work that eats the budget leaves the join no extra wait."""
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", True)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 0.5)
        calls = []

        def slow_query(*a, **kw):
            if not calls:
                calls.append(1)
                time.sleep(0.6)  # vector work alone exceeds the whole budget
            return _qresult([_hit("v0", 0.9, "vector fact")])

        service._memory.vector_store.client.query_points.side_effect = slow_query
        fake, done = _slow_graph(3.0)
        with patch.object(service, "_search_graph_for_visibility", side_effect=fake):
            t0 = time.monotonic()
            results = service.search(query="q", user_id="u1", limit=5)
            elapsed = time.monotonic() - t0
        # Join starts after ~0.6 s of vector work; budget from submit is already spent,
        # so total stays near the vector time, not vector + budget.
        assert elapsed < 0.6 + 0.4, f"budget measured from join, not submit: {elapsed:.2f}s"
        assert [r.id for r in results] == ["v0"]
        assert done.wait(timeout=10)

    def test_graph_leg_inside_budget_is_still_merged(self, service, monkeypatch):
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", True)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 2.0)
        fake, _ = _slow_graph(0.05)
        before = search_mod._graph_leg_budget_timeouts
        with patch.object(service, "_search_graph_for_visibility", side_effect=fake):
            results = service.search(query="q", user_id="u1", limit=10)
        assert {r.id for r in results if r.source == "graph"} == {"g0", "g1"}
        assert search_mod._graph_leg_budget_timeouts == before

    def test_next_search_after_a_timeout_is_unaffected(self, service, monkeypatch):
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", True)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 0.3)
        slow, done = _slow_graph(1.0)
        with patch.object(service, "_search_graph_for_visibility", side_effect=slow):
            service.search(query="q", user_id="u1", limit=5)
        assert done.wait(timeout=10)
        fast, _ = _slow_graph(0.0)
        with patch.object(service, "_search_graph_for_visibility", side_effect=fast):
            results = service.search(query="q", user_id="u1", limit=10)
        assert {r.id for r in results if r.source == "graph"} == {"g0", "g1"}


class TestBudgetOff:
    def test_flag_defaults_off_and_budget_is_four_seconds(self):
        s = Settings(_env_file=None)
        assert s.graph_leg_budget_enabled is False
        assert s.graph_leg_budget_s == 4.0

    def test_flag_off_uses_the_existing_join_constant(self, service, monkeypatch):
        """Off: the join timeout is _GRAPH_SEARCH_JOIN_TIMEOUT_S, untouched by the budget."""
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", False)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 0.1)  # must be ignored
        monkeypatch.setattr(search_mod, "_GRAPH_SEARCH_JOIN_TIMEOUT_S", 0.4)
        fake, done = _slow_graph(1.0)
        before = search_mod._graph_leg_budget_timeouts
        with patch.object(service, "_search_graph_for_visibility", side_effect=fake):
            t0 = time.monotonic()
            results = service.search(query="q", user_id="u1", limit=5)
            elapsed = time.monotonic() - t0
        assert 0.4 <= elapsed < 0.9  # waited the constant, not the 0.1 s budget
        assert [r.id for r in results] == ["v0"]
        assert search_mod._graph_leg_budget_timeouts == before  # budget path not taken
        assert done.wait(timeout=10)

    def test_flag_off_waits_for_a_leg_slower_than_the_budget(self, service, monkeypatch):
        monkeypatch.setattr(settings, "graph_leg_budget_enabled", False)
        monkeypatch.setattr(settings, "graph_leg_budget_s", 0.1)
        fake, _ = _slow_graph(0.5)
        with patch.object(service, "_search_graph_for_visibility", side_effect=fake):
            results = service.search(query="q", user_id="u1", limit=10)
        assert {r.id for r in results if r.source == "graph"} == {"g0", "g1"}


class TestSettingsValidation:
    @pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
    def test_budget_must_be_finite_and_positive(self, bad):
        with pytest.raises(ValueError):
            Settings(_env_file=None, graph_leg_budget_s=bad)

    def test_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("GRAPH_LEG_BUDGET_ENABLED", "true")
        monkeypatch.setenv("GRAPH_LEG_BUDGET_S", "2.5")
        s = Settings(_env_file=None)
        assert s.graph_leg_budget_enabled is True and s.graph_leg_budget_s == 2.5
