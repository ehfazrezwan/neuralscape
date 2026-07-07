"""R4: Tests for bi-temporal edge metadata surfacing to the answer layer.

Verifies that Graphiti's valid_at/invalid_at are carried through from the
graph edges to MemoryResponse objects and rendered in ask() evidence.
"""

import asyncio
import uuid as _uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService
from schemas import MemoryResponse
from ask import _render_evidence


@pytest.fixture
def service():
    """Minimal MemoryService for exercising the real recall path."""
    return MemoryService()


class TestR4BitemporalMetadata:
    """R4: bi-temporal validity fields surface to the answer layer."""

    def test_do_graph_search_edges_include_iso_temporal_fields(self, service):
        """_do_graph_search (real code under test) converts a LIVE edge's datetime
        valid_at/created_at to ISO strings on the returned edge dict. Recall only
        surfaces live edges (invalid_at/expired_at unset — see _edge_is_invalidated),
        so invalid_at is None here; the invalid_at→ISO path is covered by the
        adapter test below (that path does not apply the live-edges filter)."""
        edge = SimpleNamespace(
            uuid="test-uuid-123",
            name="works_at",
            fact="Alice works at Acme Corp",
            valid_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            invalid_at=None,  # live edge — otherwise filtered out of recall
            created_at=datetime(2024, 1, 16, 14, 0, 0, tzinfo=timezone.utc),
            expired_at=None,
        )
        mock_graphiti = MagicMock()
        mock_results = SimpleNamespace(edges=[edge], nodes=[], episodes=[], communities=[])
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, "_get_graphiti", return_value=mock_graphiti):
            with patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
                # _enrich_graph_results does a Cypher round-trip; stub it out.
                with patch.object(service, "_enrich_graph_results", lambda *a, **k: None):
                    result = service._do_graph_search(
                        query="q", group_ids=["user--test"], limit=10,
                    )
        assert len(result["edges"]) == 1
        row = result["edges"][0]
        assert row["valid_at"] == "2024-01-15T10:30:00+00:00"
        assert row["invalid_at"] is None
        assert row["created_at"] == "2024-01-16T14:00:00+00:00"

    def test_do_graph_search_edges_handle_none_temporal_fields(self, service):
        """None temporal fields on the edge come back as None (not a crash)."""
        edge = SimpleNamespace(
            uuid="test-uuid-456", name="knows", fact="Bob knows Charlie",
            valid_at=None, invalid_at=None, created_at=None, expired_at=None,
        )
        mock_graphiti = MagicMock()
        mock_results = SimpleNamespace(edges=[edge], nodes=[], episodes=[], communities=[])
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, "_get_graphiti", return_value=mock_graphiti):
            with patch.object(service, "_run_on_bridge", side_effect=lambda x: x):
                with patch.object(service, "_enrich_graph_results", lambda *a, **k: None):
                    result = service._do_graph_search(
                        query="q", group_ids=["user--test"], limit=10,
                    )
        row = result["edges"][0]
        assert row["valid_at"] is None
        assert row["invalid_at"] is None
        assert row["created_at"] is None

    def test_recall_fusion_sets_temporal_fields_on_memory_response(self):
        """Recall fusion populates valid_at/invalid_at on graph MemoryResponse."""
        # Simulate an edge dict from _do_graph_search (already ISO strings)
        edge = {
            "uuid": "edge-abc-123",
            "fact": "Product X launched in Q1 2024",
            "valid_at": "2024-01-10T00:00:00+00:00",
            "invalid_at": "2024-04-01T00:00:00+00:00",
            "created_at": "2024-01-15T12:00:00+00:00",
        }

        # Simulate the recall fusion MemoryResponse construction (search.py:592-602)
        response = MemoryResponse(
            id=edge.get("uuid", ""),
            memory=edge.get("fact", ""),
            source="graph",
            score=0.85,
            valid_at=edge.get("valid_at"),
            invalid_at=edge.get("invalid_at"),
            created_at=edge.get("created_at"),
        )

        assert response.valid_at == "2024-01-10T00:00:00+00:00"
        assert response.invalid_at == "2024-04-01T00:00:00+00:00"
        assert response.created_at == "2024-01-15T12:00:00+00:00"

    def test_render_evidence_appends_validity_interval_when_present(self):
        """_render_evidence appends 'valid <date>–<date>' when valid_at is set."""
        mem = MemoryResponse(
            id="fact-1",
            memory="Alice works at Acme Corp",
            category="tech_stack",
            created_at="2024-01-16T14:00:00+00:00",
            valid_at="2024-01-15T10:30:00+00:00",
            invalid_at="2025-06-01T09:00:00+00:00",
        )

        evidence = _render_evidence([mem])

        # Should render: [fact-1] (2024-01-16T14:00:00+00:00; tech_stack; valid 2024-01-15–2025-06-01) Alice works at Acme Corp
        assert "valid 2024-01-15–2025-06-01" in evidence
        assert "[fact-1]" in evidence
        assert "Alice works at Acme Corp" in evidence

    def test_render_evidence_shows_present_when_invalid_at_is_none(self):
        """_render_evidence renders 'valid <date>–Present' when invalid_at is None."""
        mem = MemoryResponse(
            id="fact-2",
            memory="Bob is the CTO",
            category="personal_fact",
            created_at="2024-03-20T10:00:00+00:00",
            valid_at="2024-03-20T00:00:00+00:00",
            invalid_at=None,  # still valid
        )

        evidence = _render_evidence([mem])

        assert "valid 2024-03-20–Present" in evidence
        assert "[fact-2]" in evidence
        assert "Bob is the CTO" in evidence

    def test_render_evidence_omits_validity_when_valid_at_is_none(self):
        """_render_evidence does NOT add 'valid' annotation when valid_at is None."""
        mem = MemoryResponse(
            id="fact-3",
            memory="Charlie loves Python",
            category="preference",
            created_at="2024-05-10T08:00:00+00:00",
            valid_at=None,
            invalid_at=None,
        )

        evidence = _render_evidence([mem])

        # Should be byte-identical to pre-R4: no "valid" token
        assert "valid" not in evidence
        assert "[fact-3]" in evidence
        assert "(2024-05-10T08:00:00+00:00; preference)" in evidence
        assert "Charlie loves Python" in evidence

    def test_memory_response_exclude_none_omits_unset_temporal_fields(self):
        """MemoryResponse.model_dump(exclude_none=True) omits valid_at/invalid_at when None (backward-compat)."""
        response = MemoryResponse(
            id="mem-456",
            memory="Some fact",
            category="domain_knowledge",
            valid_at=None,
            invalid_at=None,
        )

        dumped = response.model_dump(exclude_none=True)

        # valid_at and invalid_at should NOT be in the dict when None
        assert "valid_at" not in dumped
        assert "invalid_at" not in dumped
        assert dumped["id"] == "mem-456"
        assert dumped["memory"] == "Some fact"

    def test_memory_response_exclude_none_includes_set_temporal_fields(self):
        """MemoryResponse.model_dump(exclude_none=True) includes valid_at/invalid_at when set."""
        response = MemoryResponse(
            id="mem-789",
            memory="Another fact",
            category="decision",
            valid_at="2024-02-01T00:00:00+00:00",
            invalid_at="2024-08-01T00:00:00+00:00",
        )

        dumped = response.model_dump(exclude_none=True)

        # valid_at and invalid_at SHOULD be in the dict when set
        assert dumped["valid_at"] == "2024-02-01T00:00:00+00:00"
        assert dumped["invalid_at"] == "2024-08-01T00:00:00+00:00"
        assert dumped["id"] == "mem-789"

    def _resolve_one_edge(self, edge):
        """Invoke the REAL adapter MemoryGraph._resolve_edge_names on a single
        edge, with EntityNode.get_by_uuids stubbed to name the endpoints."""
        from mem0.memory.graphiti_memory import MemoryGraph
        import mem0.memory.graphiti_memory as gm

        graph = object.__new__(MemoryGraph)
        graph.graphiti = SimpleNamespace(driver=MagicMock())

        async def _fake_get_by_uuids(driver, uuids):
            names = {edge.source_node_uuid: "Alice", edge.target_node_uuid: "Bob"}
            return [SimpleNamespace(uuid=u, name=names.get(u, u)) for u in uuids]

        with patch.object(gm.EntityNode, "get_by_uuids", side_effect=_fake_get_by_uuids):
            return asyncio.run(graph._resolve_edge_names([edge]))[0]

    def test_adapter_resolve_edge_names_includes_temporal_fields(self):
        """The REAL _resolve_edge_names emits ISO valid_at/invalid_at (additive).
        The adapter/MCP path does not apply the live-edges filter, so an
        invalid_at-bounded edge is preserved here."""
        edge = SimpleNamespace(
            source_node_uuid="node-a", target_node_uuid="node-b",
            name="works_with", fact="Alice works with Bob",
            valid_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            invalid_at=datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )
        result = self._resolve_one_edge(edge)

        assert result["valid_at"] == "2024-01-01T00:00:00+00:00"
        assert result["invalid_at"] == "2024-12-31T23:59:59+00:00"
        # Existing keys unchanged
        assert result["source"] == "Alice"
        assert result["relationship"] == "works_with"
        assert result["destination"] == "Bob"
        assert result["fact"] == "Alice works with Bob"

    def test_adapter_resolve_edge_names_handles_none_temporal_fields(self):
        """The REAL _resolve_edge_names handles None valid_at/invalid_at."""
        edge = SimpleNamespace(
            source_node_uuid="node-a", target_node_uuid="node-b",
            name="knows", fact="Charlie knows Diana",
            valid_at=None, invalid_at=None,
        )
        result = self._resolve_one_edge(edge)

        assert result["valid_at"] is None
        assert result["invalid_at"] is None
        assert result["fact"] == "Charlie knows Diana"
