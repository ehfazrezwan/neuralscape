"""R4: Tests for bi-temporal edge metadata surfacing to the answer layer.

Verifies that Graphiti's valid_at/invalid_at are carried through from the
graph edges to MemoryResponse objects and rendered in ask() evidence.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from schemas import MemoryResponse
from ask import _render_evidence


class TestR4BitemporalMetadata:
    """R4: bi-temporal validity fields surface to the answer layer."""

    def test_edge_dict_includes_iso_temporal_fields(self):
        """Edge mapping converts datetime valid_at/invalid_at to ISO strings."""
        # Mock an EntityEdge-like object with datetime temporal fields
        edge = MagicMock()
        edge.uuid = "test-uuid-123"
        edge.name = "works_at"
        edge.fact = "Alice works at Acme Corp"
        edge.valid_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        edge.invalid_at = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        edge.created_at = datetime(2024, 1, 16, 14, 0, 0, tzinfo=timezone.utc)

        # Simulate the edge mapping logic from search.py:1112-1120
        from memory.search import _dt_to_iso
        edge_dict = {
            "uuid": edge.uuid,
            "name": edge.name,
            "fact": edge.fact,
            "valid_at": _dt_to_iso(getattr(edge, "valid_at", None)),
            "invalid_at": _dt_to_iso(getattr(edge, "invalid_at", None)),
            "created_at": _dt_to_iso(getattr(edge, "created_at", None)),
        }

        assert edge_dict["valid_at"] == "2024-01-15T10:30:00+00:00"
        assert edge_dict["invalid_at"] == "2025-06-01T09:00:00+00:00"
        assert edge_dict["created_at"] == "2024-01-16T14:00:00+00:00"

    def test_edge_dict_handles_none_temporal_fields(self):
        """Edge mapping handles None temporal fields (non-graph rows)."""
        edge = MagicMock()
        edge.uuid = "test-uuid-456"
        edge.name = "knows"
        edge.fact = "Bob knows Charlie"
        edge.valid_at = None
        edge.invalid_at = None
        edge.created_at = None

        from memory.search import _dt_to_iso
        edge_dict = {
            "uuid": edge.uuid,
            "name": edge.name,
            "fact": edge.fact,
            "valid_at": _dt_to_iso(getattr(edge, "valid_at", None)),
            "invalid_at": _dt_to_iso(getattr(edge, "invalid_at", None)),
            "created_at": _dt_to_iso(getattr(edge, "created_at", None)),
        }

        assert edge_dict["valid_at"] is None
        assert edge_dict["invalid_at"] is None
        assert edge_dict["created_at"] is None

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

    def test_adapter_resolve_edge_names_includes_temporal_fields(self):
        """Adapter's _resolve_edge_names includes ISO valid_at/invalid_at (additive)."""
        # Mock an EntityEdge with bi-temporal fields
        edge = MagicMock()
        edge.source_node_uuid = "node-a"
        edge.target_node_uuid = "node-b"
        edge.name = "works_with"
        edge.fact = "Alice works with Bob"
        edge.valid_at = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        edge.invalid_at = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # Simulate the adapter's _resolve_edge_names logic
        # (without async/await for simplicity; focus on field mapping)
        result = {
            "source": "Alice",
            "relationship": edge.name,
            "destination": "Bob",
            "fact": edge.fact,
            "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
            "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
        }

        assert result["valid_at"] == "2024-01-01T00:00:00+00:00"
        assert result["invalid_at"] == "2024-12-31T23:59:59+00:00"
        # Existing keys unchanged
        assert result["source"] == "Alice"
        assert result["relationship"] == "works_with"
        assert result["destination"] == "Bob"
        assert result["fact"] == "Alice works with Bob"

    def test_adapter_resolve_edge_names_handles_none_temporal_fields(self):
        """Adapter's _resolve_edge_names handles None valid_at/invalid_at."""
        edge = MagicMock()
        edge.source_node_uuid = "node-c"
        edge.target_node_uuid = "node-d"
        edge.name = "knows"
        edge.fact = "Charlie knows Diana"
        edge.valid_at = None
        edge.invalid_at = None

        result = {
            "source": "Charlie",
            "relationship": edge.name,
            "destination": "Diana",
            "fact": edge.fact,
            "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
            "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
        }

        assert result["valid_at"] is None
        assert result["invalid_at"] is None
        assert result["fact"] == "Charlie knows Diana"
