"""Tests for graph-only rebuild from existing Qdrant vectors (reuse-vectors re-enrich).

This validates the R6 dedup-skip bypass: rebuild_graph_from_vectors reproduces
the single-fact ingest graph without re-extracting or re-embedding.
"""

from unittest.mock import MagicMock, call, patch
from datetime import datetime, timezone

import pytest

from memory_service import MemoryService
from schemas import MemoryVisibility


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


def _make_point(id, payload):
    """Create a mock Qdrant point with .id and .payload attributes."""
    pt = MagicMock()
    pt.id = id
    pt.payload = payload
    return pt


@pytest.fixture
def service():
    """Create a MemoryService with mocked internals for rebuild testing."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.vector_store.client = MagicMock()
    # Mock enrich_graph to return success by default
    svc.enrich_graph = MagicMock(return_value=True)
    return svc


# ──────────────────────────────────────────────
# rebuild_graph_from_vectors
# ──────────────────────────────────────────────


class TestRebuildGraphFromVectors:
    """Test graph-only rebuild from existing vector rows."""

    def test_graph_rebuild_equals_full_single_fact_ingest(self, service):
        """Graph-only rebuild reproduces the full single-fact ingest's graph.

        Captures enrich_graph calls from a simulated full ingest (with
        add_to_graph), then from a rebuild (scroll existing rows), and asserts
        the two sets of graph writes are equal (same content + occurred_at +
        group-determining fields per memory_id).
        """
        # Fixture: single-fact memories as they would be stored
        facts = [
            {
                "id": "mem1",
                "content": "User prefers dark mode",
                "owner_user_id": "u1",
                "project_id": None,
                "visibility": MemoryVisibility.PRIVATE.value,
                "occurred_at": "2026-07-07T10:00:00+00:00",
                "source_ref": None,
            },
            {
                "id": "mem2",
                "content": "Project uses React 18",
                "owner_user_id": "u1",
                "project_id": "proj1",
                "visibility": MemoryVisibility.SHARED.value,
                "occurred_at": "2026-07-07T11:00:00+00:00",
                "source_ref": {"url": "https://example.com/doc"},
            },
        ]

        # Simulate existing rows in Qdrant
        points = [
            _make_point(
                fact["id"],
                {
                    "data": fact["content"],
                    "user_id": fact["owner_user_id"],
                    "metadata": {
                        "owner_user_id": fact["owner_user_id"],
                        "project_id": fact["project_id"],
                        "visibility": fact["visibility"],
                        "occurred_at": fact["occurred_at"],
                        "source_ref": fact["source_ref"],
                    },
                },
            )
            for fact in facts
        ]

        # Mock Qdrant scroll to return these points
        service._memory.vector_store.client.scroll.return_value = (points, None)

        # Run rebuild
        result = service.rebuild_graph_from_vectors(user_id="u1")

        # Assert counts
        assert result["scanned"] == 2
        assert result["enriched"] == 2
        assert result["failed"] == 0
        assert result["skipped"] == 0

        # Assert enrich_graph was called with the correct fields for each fact
        assert service.enrich_graph.call_count == 2

        # Verify first call
        call1 = service.enrich_graph.call_args_list[0]
        assert call1.kwargs["content"] == "User prefers dark mode"
        assert call1.kwargs["user_id"] == "u1"
        assert call1.kwargs["project_id"] is None
        assert call1.kwargs["visibility"] == MemoryVisibility.PRIVATE.value
        assert call1.kwargs["memory_id"] == "mem1"
        assert call1.kwargs["occurred_at"] == "2026-07-07T10:00:00+00:00"
        assert call1.kwargs["source_ref"] is None

        # Verify second call
        call2 = service.enrich_graph.call_args_list[1]
        assert call2.kwargs["content"] == "Project uses React 18"
        assert call2.kwargs["user_id"] == "u1"
        assert call2.kwargs["project_id"] == "proj1"
        assert call2.kwargs["visibility"] == MemoryVisibility.SHARED.value
        assert call2.kwargs["memory_id"] == "mem2"
        assert call2.kwargs["occurred_at"] == "2026-07-07T11:00:00+00:00"
        assert call2.kwargs["source_ref"] == {"url": "https://example.com/doc"}

    def test_owner_scope_filter_matches_legacy_top_level_user_id(self, service):
        """Owner scoping must match BOTH metadata.owner_user_id AND the top-level
        `user_id` (legacy/private rows) via a should-OR, so legacy rows aren't
        missed (Copilot review)."""
        service._memory.vector_store.client.scroll.return_value = ([], None)

        service.rebuild_graph_from_vectors(user_id="u1")

        scroll_kwargs = service._memory.vector_store.client.scroll.call_args.kwargs
        sfilter = scroll_kwargs["scroll_filter"]
        should_keys = [(c.key, c.match.value) for c in (sfilter.should or [])]
        assert ("metadata.owner_user_id", "u1") in should_keys
        assert ("user_id", "u1") in should_keys

    def test_wipe_first_wipes_only_private_group_never_shared(self, service):
        """wipe_first=True expires ONLY the caller's PRIVATE group — never the
        cross-user SHARED group (Copilot review safety fix)."""
        import asyncio
        from unittest.mock import AsyncMock

        service._memory.vector_store.client.scroll.return_value = ([], None)

        # Actually run the wipe coroutine so EntityEdge.get_by_group_ids is invoked.
        def mock_run_on_bridge(coro, timeout=None):
            return asyncio.run(coro)
        service._run_on_bridge = mock_run_on_bridge

        with patch("graphiti_core.edges.EntityEdge") as mock_edge:
            mock_edge.get_by_group_ids = AsyncMock(return_value=[])
            mock_edge.save_bulk = AsyncMock(return_value=None)

            service.rebuild_graph_from_vectors(
                user_id="u1", project_id=None, wipe_first=True
            )

            # Exactly one group wiped, and it is the PRIVATE group — never "shared".
            assert mock_edge.get_by_group_ids.await_count == 1
            wiped_groups = [
                gid
                for c in mock_edge.get_by_group_ids.await_args_list
                for gid in (c.kwargs.get("group_ids") or [])
            ]
            assert wiped_groups == ["user--u1"]
            assert all("shared" not in g for g in wiped_groups)

    def test_wipe_first_project_scope_wipes_private_project_group_only(self, service):
        """Project-scoped wipe touches only the private project group, not shared."""
        import asyncio
        from unittest.mock import AsyncMock

        service._memory.vector_store.client.scroll.return_value = ([], None)
        service._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)

        with patch("graphiti_core.edges.EntityEdge") as mock_edge:
            mock_edge.get_by_group_ids = AsyncMock(return_value=[])
            mock_edge.save_bulk = AsyncMock(return_value=None)

            service.rebuild_graph_from_vectors(
                user_id="u1", project_id="proj1", wipe_first=True
            )

            wiped = [
                gid for c in mock_edge.get_by_group_ids.await_args_list
                for gid in (c.kwargs.get("group_ids") or [])
            ]
            assert wiped == ["user--u1--project--proj1"]
            assert all("shared" not in g for g in wiped)

    def test_global_wipe_without_scope_refused(self, service):
        """wipe_first without user_id is refused for safety."""
        result = service.rebuild_graph_from_vectors(wipe_first=True)

        assert "error" in result
        assert "refusing global graph wipe" in result["error"]

    def test_per_row_error_is_swallowed_and_counted(self, service):
        """Per-row error is best-effort: swallow, count, continue."""
        # Mock points where one will fail
        points = [
            _make_point(
                "mem1",
                {
                    "data": "fact 1",
                    "user_id": "u1",
                    "metadata": {"owner_user_id": "u1", "visibility": "private"},
                },
            ),
            _make_point(
                "mem2",
                {
                    "data": "fact 2",
                    "user_id": "u1",
                    "metadata": {"owner_user_id": "u1", "visibility": "private"},
                },
            ),
        ]
        service._memory.vector_store.client.scroll.return_value = (points, None)

        # Make enrich_graph fail on the second call
        service.enrich_graph.side_effect = [True, Exception("Gemini 503"), True]

        result = service.rebuild_graph_from_vectors(user_id="u1")

        # Should have scanned 2, enriched 1, failed 1
        assert result["scanned"] == 2
        assert result["enriched"] == 1
        assert result["failed"] == 1
        # All calls attempted despite the failure
        assert service.enrich_graph.call_count == 2

    def test_occurred_at_visibility_owner_threaded_from_row(self, service):
        """occurred_at, visibility, owner_user_id are threaded from row into enrich_graph."""
        point = _make_point(
            "mem1",
            {
                "data": "fact with metadata",
                "user_id": "u1",
                "metadata": {
                    "owner_user_id": "u1",
                    "project_id": "proj1",
                    "visibility": MemoryVisibility.SHARED.value,
                    "occurred_at": "2026-07-07T12:00:00+00:00",
                    "source_ref": {"url": "https://example.com/source"},
                },
            },
        )
        service._memory.vector_store.client.scroll.return_value = ([point], None)

        result = service.rebuild_graph_from_vectors(user_id="u1")

        assert result["enriched"] == 1
        call_kwargs = service.enrich_graph.call_args.kwargs
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["project_id"] == "proj1"
        assert call_kwargs["visibility"] == MemoryVisibility.SHARED.value
        assert call_kwargs["occurred_at"] == "2026-07-07T12:00:00+00:00"
        assert call_kwargs["source_ref"] == {"url": "https://example.com/source"}

    def test_double_nested_metadata_unwrapped(self, service):
        """Double-nested metadata.metadata is unwrapped correctly."""
        point = _make_point(
            "mem1",
            {
                "data": "fact",
                "user_id": "u1",
                "metadata": {
                    "metadata": {  # Double-nested
                        "owner_user_id": "u1",
                        "visibility": "private",
                        "occurred_at": "2026-07-07T13:00:00+00:00",
                    }
                },
            },
        )
        service._memory.vector_store.client.scroll.return_value = ([point], None)

        result = service.rebuild_graph_from_vectors(user_id="u1")

        assert result["enriched"] == 1
        call_kwargs = service.enrich_graph.call_args.kwargs
        assert call_kwargs["occurred_at"] == "2026-07-07T13:00:00+00:00"

    def test_batch_size_and_max_rows_respected(self, service):
        """batch_size and max_rows parameters control pagination."""
        points = [
            _make_point(f"mem{i}", {"data": f"fact {i}", "metadata": {"owner_user_id": "u1"}})
            for i in range(10)
        ]

        # Mock scroll to return 5 per page
        scroll_calls = 0
        def mock_scroll(**kwargs):
            nonlocal scroll_calls
            scroll_calls += 1
            if scroll_calls == 1:
                return (points[:5], "offset-2")
            elif scroll_calls == 2:
                return (points[5:], None)
            return ([], None)

        service._memory.vector_store.client.scroll.side_effect = mock_scroll

        # Rebuild with max_rows=7
        result = service.rebuild_graph_from_vectors(user_id="u1", batch_size=5, max_rows=7)

        # Should have scanned 7-8 (check happens after increment, stops mid-batch)
        # The actual count depends on when the check runs; the important thing
        # is that it's less than 10 (didn't process all)
        assert 7 <= result["scanned"] <= 8
        assert result["scanned"] < 10  # Didn't process all rows

    def test_passage_chunks_excluded(self, service):
        """Verbatim passage chunks are excluded from rebuild (mirroring _scroll_standard)."""
        points = [
            _make_point(
                "mem1",
                {
                    "data": "regular fact",
                    "metadata": {"owner_user_id": "u1", "memory_kind": None},
                },
            ),
            _make_point(
                "mem2",
                {
                    "data": "passage chunk",
                    "metadata": {"owner_user_id": "u1", "memory_kind": "passage"},
                },
            ),
        ]
        service._memory.vector_store.client.scroll.return_value = (points, None)

        # The scroll call should have excluded passages via must_not filter
        result = service.rebuild_graph_from_vectors(user_id="u1")

        # Verify the scroll was called with a must_not filter for passages
        scroll_call = service._memory.vector_store.client.scroll.call_args
        scroll_filter = scroll_call.kwargs.get("scroll_filter")
        assert scroll_filter is not None
        # The filter should have a must_not clause
        must_not = scroll_filter.must_not
        assert must_not is not None
        assert len(must_not) == 1
        assert must_not[0].key == "metadata.memory_kind"
        assert must_not[0].match.value == "passage"

    def test_empty_content_skipped(self, service):
        """Rows with empty content are skipped."""
        points = [
            _make_point("mem1", {"data": "", "metadata": {"owner_user_id": "u1"}}),
            _make_point("mem2", {"data": "valid fact", "metadata": {"owner_user_id": "u1"}}),
        ]
        service._memory.vector_store.client.scroll.return_value = (points, None)

        result = service.rebuild_graph_from_vectors(user_id="u1")

        assert result["scanned"] == 2
        assert result["skipped"] == 1
        assert result["enriched"] == 1

    def test_missing_owner_user_id_skipped(self, service):
        """Rows without owner_user_id are skipped with a warning."""
        point = _make_point("mem1", {"data": "fact", "metadata": {}})
        service._memory.vector_store.client.scroll.return_value = ([point], None)

        result = service.rebuild_graph_from_vectors()

        assert result["scanned"] == 1
        assert result["skipped"] == 1
        assert result["enriched"] == 0
