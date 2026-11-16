"""The graph leg of a search runs on a pool thread and must stay attributed to the caller.

``ThreadPoolExecutor.submit`` does not carry ContextVars into the worker thread, so Graphiti's own embedding and
rerank calls (made inside the leg) found no bound usage attempt and were dropped, with no gap recorded.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import qdrant_client  # noqa: F401  pre-import: search() imports it lazily
import usage_capture as uc
from memory_service import MemoryService


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
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    result = MagicMock()
    result.points = [_hit("v0", 0.9, "vector fact")]
    svc._memory.vector_store.client.query_points.return_value = result
    return svc


def _provider_response():
    return SimpleNamespace(
        text="",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=0,
            cached_content_token_count=None, thoughts_token_count=None,
        ),
    )


def _graphiti_leg(seen):
    def leg(*_args, **_kwargs):
        # What Graphiti's wrapped embedder does inside the leg, on the pool thread.
        seen.append(uc.current_attempt_context())
        uc.record_provider_usage(_provider_response(), operation="graphiti_embed", model="m", kind="generation")
        return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    return leg


def test_a_search_graph_leg_records_its_provider_calls_for_the_caller(service):
    recorder = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(recorder)
    seen = []
    try:
        with patch.object(service, "_search_graph_for_visibility", side_effect=_graphiti_leg(seen)):
            results = service.search(query="q", user_id="alice", limit=5)  # not vector_only: the graph leg runs
    finally:
        uc.set_usage_recorder(previous)
    assert [r.id for r in results] == ["v0"]
    assert seen and seen[0] is not None and seen[0].tenant_id == "alice"
    assert [e.tenant_id for e in recorder.events] == ["alice"]
    assert recorder.events[0].operation == "graphiti_embed"


def test_the_graph_leg_binds_nothing_when_no_recorder_is_installed(service):
    seen = []
    with patch.object(service, "_search_graph_for_visibility", side_effect=_graphiti_leg(seen)):
        results = service.search(query="q", user_id="alice", limit=5)
    assert [r.id for r in results] == ["v0"]
    assert seen == [None]


def test_two_concurrent_searches_keep_their_own_callers(service):
    import threading

    recorder = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(recorder)
    barrier = threading.Barrier(2, timeout=5)
    seen = []

    def leg(*_a, **_kw):
        barrier.wait()  # both legs are in the pool at once
        seen.append(uc.current_attempt_context().tenant_id)
        uc.record_provider_usage(_provider_response(), operation="graphiti_embed", model="m")
        return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    try:
        with patch.object(service, "_search_graph_for_visibility", side_effect=leg):
            threads = [
                threading.Thread(target=service.search, kwargs={"query": "q", "user_id": user, "limit": 5})
                for user in ("alice", "bob")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
    finally:
        uc.set_usage_recorder(previous)
    assert sorted(seen) == ["alice", "bob"]
    assert sorted(e.tenant_id for e in recorder.events) == ["alice", "bob"]
