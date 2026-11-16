"""Direct unit tests for knowledge/ns_memory.py (the base KnowledgeSystem)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from knowledge.base import IndexReport, IndexRequest, RecallRequest
from knowledge.ns_memory import NSMemorySystem


def _memory(text, category="preference", **extra):
    m = MagicMock()
    m.memory = text
    m.category = category
    m.model_dump.return_value = {"memory": text, "category": category, **extra}
    return m


def _system_with(service):
    sysm = NSMemorySystem()
    sysm._service = service
    return sysm


# ── identity ─────────────────────────────────────────────────────────


def test_info_declares_base_system_and_capabilities():
    info = NSMemorySystem.info
    assert info.name == "ns-memory"
    assert info.kind == "base"
    assert info.transport == "in-process"
    assert info.capabilities == frozenset({"recall", "timeline", "cards", "ask", "graph_search"})


def test_service_is_lazy_and_loaded_once():
    sentinel = object()
    with patch("knowledge.ns_memory.get_shared_service", return_value=sentinel) as get:
        sysm = NSMemorySystem()
        get.assert_not_called()  # constructing does not touch the service
        assert sysm._get_service() is sentinel
        assert sysm._get_service() is sentinel
        get.assert_called_once()


# ── health ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vector,graph,status,details",
    [
        (object(), object(), "ok", {"vector_store": "ok", "graph_store": "ok"}),
        (object(), None, "degraded", {"vector_store": "ok", "graph_store": "not_initialized"}),
        (None, object(), "degraded", {"vector_store": "not_initialized", "graph_store": "ok"}),
        (None, None, "degraded", {"vector_store": "not_initialized", "graph_store": "not_initialized"}),
    ],
)
def test_health_reports_store_state(vector, graph, status, details):
    h = _system_with(SimpleNamespace(_memory=vector, _graphiti=graph)).health()
    assert h.status == status
    assert h.details == details


def test_health_unreachable_when_probe_raises():
    class Boom:
        @property
        def _memory(self):
            raise RuntimeError("qdrant down")

    h = _system_with(Boom()).health()
    assert h.status == "unreachable"
    assert "qdrant down" in h.details["error"]


# ── recall ───────────────────────────────────────────────────────────


def test_recall_requires_user_id_and_never_searches():
    svc = MagicMock()
    with pytest.raises(ValueError, match="user_id"):
        _system_with(svc).recall(RecallRequest(query="q"))
    svc.search.assert_not_called()


def test_recall_forwards_request_to_search_with_unfiltered_defaults():
    svc = MagicMock()
    svc.search.return_value = []
    _system_with(svc).recall(RecallRequest(query="tabs or spaces", user_id="u1", project_id="p1", limit=7))
    svc.search.assert_called_once_with(
        query="tabs or spaces",
        user_id="u1",
        project_id="p1",
        categories=None,
        scope=None,
        limit=7,
    )


def test_recall_empty_result():
    svc = MagicMock()
    svc.search.return_value = []
    ans = _system_with(svc).recall(RecallRequest(query="x", user_id="u"))
    assert ans.system_name == "ns-memory"
    assert ans.system_version is None
    assert ans.content == "No memories found."
    assert ans.hits == []
    assert ans.metadata == {"result_count": 0}


def test_recall_formats_numbered_content_and_structured_hits():
    svc = MagicMock()
    svc.search.return_value = [_memory("likes pytest", "preference"), _memory("uses uv", "tech_stack")]
    ans = _system_with(svc).recall(RecallRequest(query="tooling", user_id="u"))
    assert ans.content.splitlines() == [
        "1. likes pytest (category=preference)",
        "2. uses uv (category=tech_stack)",
    ]
    assert [h["memory"] for h in ans.hits] == ["likes pytest", "uses uv"]
    assert ans.metadata == {"result_count": 2}


def test_recall_propagates_search_errors():
    svc = MagicMock()
    svc.search.side_effect = RuntimeError("backend failed")
    with pytest.raises(RuntimeError, match="backend failed"):
        _system_with(svc).recall(RecallRequest(query="x", user_id="u"))


# ── index ────────────────────────────────────────────────────────────


def test_index_is_a_noop_report():
    rep = NSMemorySystem().index(IndexRequest(source="/anything"))
    assert isinstance(rep, IndexReport)
    assert (rep.files_indexed, rep.symbols_indexed, rep.edges_indexed) == (0, 0, 0)
    assert rep.incremental is False
    assert rep.duration_s == 0.0
    assert rep.system_version is None
