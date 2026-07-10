"""Phase E tests: fusion + batched anchor join + liveness.

Acceptance (per plan E row):
- Flagship E2E: decision memory anchored to function → "who calls fn" returns
  BOTH structure AND the memory; rename fn → reindex → temporal_reframe proposed;
  visibility rules preserved (private memory of another user NOT returned).
- Latency: fused neighbors p50 ≤100ms (micro-benchmark).
- Batched anchor join is ONE query (assert query count via mock/spy).
- Plain-prose generic recall byte-identical (regression test).
- Transport-uniformity: fusion never branches on transport (test it).
"""

import pytest
from unittest.mock import MagicMock, patch, call
from knowledge.fusion import (
    compose_fusion_answer,
    extract_fqns_from_code_answer,
    batched_anchor_lookup,
)
from knowledge.base import SystemAnswer


def test_compose_fusion_answer_all_sections():
    """Fusion composer includes all three sections when present."""
    code_answer = SystemAnswer(
        system_name="code-cbm",
        content="src/click/core.py:42 CommandCollection.get_command(name)\n  --> CALLS BaseCommand.__init__",
    )

    anchor_memories = {
        "click.core.CommandCollection": [
            {"id": "mem1", "content": "Switched to lazy loading", "category": "decision"},
            {"id": "mem2", "content": "Returns None for unknown", "category": "gotcha"},
        ]
    }

    base_answer = SystemAnswer(
        system_name="ns-memory",
        content="1. [tech_stack] Project uses Click 8.1\n2. [convention] CLI commands inherit from BaseCommand",
    )

    result = compose_fusion_answer(
        code_answer=code_answer,
        anchor_memories=anchor_memories,
        base_answer=base_answer,
    )

    assert "[structure]" in result
    assert "(code-cbm)" in result
    assert "CommandCollection.get_command" in result

    assert "[semantics]" in result
    assert "click.core.CommandCollection:" in result
    assert "[decision] Switched to lazy loading" in result
    assert "[gotcha] Returns None for unknown" in result

    assert "[memory]" in result
    assert "(NS base recall)" in result
    assert "Project uses Click 8.1" in result


def test_compose_fusion_answer_code_only():
    """Fusion with code answer and anchor memories, no base recall."""
    code_answer = SystemAnswer(
        system_name="code-native",
        content="Symbol found: src/lib.py:10 MyClass.method()",
    )

    anchor_memories = {
        "lib.MyClass": [
            {"id": "mem1", "content": "Critical class for feature X", "category": "architecture"},
        ]
    }

    result = compose_fusion_answer(
        code_answer=code_answer,
        anchor_memories=anchor_memories,
        base_answer=None,
    )

    assert "[structure]" in result
    assert "[semantics]" in result
    assert "[memory]" not in result  # base_answer was None


def test_compose_fusion_answer_empty():
    """Fusion with no sections returns fallback message."""
    result = compose_fusion_answer(
        code_answer=None,
        anchor_memories=None,
        base_answer=None,
    )

    assert "No results found" in result


def test_extract_fqns_from_code_answer():
    """Extract FQNs from structured code answer hits."""
    code_answer = SystemAnswer(
        system_name="code-cbm",
        content="some text",
        hits=[
            {"fqn": "src.click.core.CommandCollection", "kind": "class"},
            {"fqn": "src.click.core.BaseCommand", "kind": "class"},
            {"no_fqn": "this should be ignored"},
        ],
    )

    fqns = extract_fqns_from_code_answer(code_answer)

    assert len(fqns) == 2
    assert "src.click.core.CommandCollection" in fqns
    assert "src.click.core.BaseCommand" in fqns


def test_extract_fqns_from_code_answer_no_hits():
    """Extract FQNs when answer has no hits and no FQN-shaped tokens."""
    code_answer = SystemAnswer(
        system_name="code-native",
        content="text only answer",
        hits=None,
    )

    fqns = extract_fqns_from_code_answer(code_answer)

    assert fqns == []


def test_extract_fqns_content_fallback_query_op():
    """Phase G-final GF2: query-op answers carry no hits — parse FQNs from content.

    The fusion code leg runs the query op (hits=None); without a content fallback
    the batched anchor join never fires. Dotted FQN-shaped tokens are extracted;
    file paths are skipped.
    """
    code_answer = SystemAnswer(
        system_name="code-native",
        content=(
            "Code graph search results for: Command\n\n"
            "click.core.Command (class) in src/click/core.py:956\n"
            "click.core.BaseCommand (class) in src/click/core.py:900\n"
        ),
        hits=None,
    )
    fqns = extract_fqns_from_code_answer(code_answer)
    assert "click.core.Command" in fqns
    assert "click.core.BaseCommand" in fqns
    # File paths (core.py) must NOT be treated as FQNs.
    assert not any(f.endswith(".py") for f in fqns)


def test_extract_fqns_hits_take_precedence_over_content():
    """Structured hits win; the content fallback only fires when hits are empty."""
    code_answer = SystemAnswer(
        system_name="code-cbm",
        content="unrelated.module.symbol mentioned in prose",
        hits=[{"fqn": "click.core.Command", "kind": "class"}],
    )
    fqns = extract_fqns_from_code_answer(code_answer)
    assert fqns == ["click.core.Command"]


def test_batched_anchor_lookup_single_query():
    """CRITICAL: batched anchor join must be ONE Qdrant query, not N."""
    from memory_service import get_shared_service
    import numpy as np

    # Mock the MemoryService and Qdrant client
    with patch("memory_service.get_shared_service") as mock_service:
        mock_m = MagicMock()
        mock_client = MagicMock()
        mock_service.return_value._get_memory.return_value = mock_m
        mock_m.vector_store.client = mock_client
        mock_m.embedding_model.embed.return_value = [0.0] * 768

        # Mock query_points to return some results
        mock_result = MagicMock()
        mock_result.points = [
            MagicMock(
                payload={
                    "id": "mem1",
                    "data": "Test memory 1",
                    "metadata": {
                        "source_ref": {"external_id": "click::click.core.CommandCollection"},
                        "category": "decision",
                        "visibility": "shared",
                    },
                }
            ),
            MagicMock(
                payload={
                    "id": "mem2",
                    "data": "Test memory 2",
                    "metadata": {
                        "source_ref": {"external_id": "click::click.core.BaseCommand"},
                        "category": "gotcha",
                        "visibility": "shared",
                    },
                }
            ),
        ]
        mock_client.query_points.return_value = mock_result

        # Call batched_anchor_lookup with 3 FQNs
        def mock_to_canonical(fqn):
            return fqn.replace("src.", "")

        result = batched_anchor_lookup(
            fqns=["src.click.core.CommandCollection", "src.click.core.BaseCommand", "src.click.utils.Utils"],
            repo="click",
            to_canonical_fn=mock_to_canonical,
            user_id="test_user",
            limit_per_anchor=3,
        )

        # CRITICAL ASSERTION: query_points called EXACTLY ONCE (batched query)
        assert mock_client.query_points.call_count == 1

        # Verify the filter had all 3 anchor keys in MatchAny
        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        match_any_condition = query_filter.must[0].match
        assert len(match_any_condition.any) == 3
        assert "click::click.core.CommandCollection" in match_any_condition.any
        assert "click::click.core.BaseCommand" in match_any_condition.any
        assert "click::click.utils.Utils" in match_any_condition.any

        # Verify results grouped by canonical FQN
        assert "click.core.CommandCollection" in result
        assert "click.core.BaseCommand" in result
        assert len(result["click.core.CommandCollection"]) == 1
        assert result["click.core.CommandCollection"][0]["id"] == "mem1"


def test_batched_anchor_lookup_visibility_preserved():
    """CRITICAL: visibility rules preserved (private memories of other users NOT returned)."""
    from memory_service import get_shared_service

    with patch("memory_service.get_shared_service") as mock_service:
        mock_m = MagicMock()
        mock_client = MagicMock()
        mock_service.return_value._get_memory.return_value = mock_m
        mock_m.vector_store.client = mock_client
        mock_m.embedding_model.embed.return_value = [0.0] * 768

        # Mock results: shared, private (same user), private (different user), standard
        mock_result = MagicMock()
        mock_result.points = [
            MagicMock(
                payload={
                    "id": "mem_shared",
                    "data": "Shared memory",
                    "metadata": {
                        "source_ref": {"external_id": "repo::Foo"},
                        "category": "decision",
                        "visibility": "shared",
                        "user_id": "other_user",
                    },
                }
            ),
            MagicMock(
                payload={
                    "id": "mem_private_same",
                    "data": "Private same user",
                    "metadata": {
                        "source_ref": {"external_id": "repo::Foo"},
                        "category": "preference",
                        "visibility": "private",
                        "user_id": "test_user",
                    },
                }
            ),
            MagicMock(
                payload={
                    "id": "mem_private_other",
                    "data": "Private other user",
                    "metadata": {
                        "source_ref": {"external_id": "repo::Foo"},
                        "category": "preference",
                        "visibility": "private",
                        "user_id": "other_user",
                    },
                }
            ),
            MagicMock(
                payload={
                    "id": "mem_standard",
                    "data": "Standard memory",
                    "metadata": {
                        "source_ref": {"external_id": "repo::Foo"},
                        "category": "convention",
                        "visibility": "standard",
                        "user_id": "dictator",
                    },
                }
            ),
        ]
        mock_client.query_points.return_value = mock_result

        result = batched_anchor_lookup(
            fqns=["Foo"],
            repo="repo",
            to_canonical_fn=lambda x: x,
            user_id="test_user",
            limit_per_anchor=10,
        )

        # Verify visibility rules: shared + standard + caller's private = 3 results
        assert "Foo" in result
        memories = result["Foo"]
        assert len(memories) == 3

        memory_ids = {m["id"] for m in memories}
        assert "mem_shared" in memory_ids  # shared visible to all
        assert "mem_private_same" in memory_ids  # caller's own private
        assert "mem_standard" in memory_ids  # standard visible to all
        assert "mem_private_other" not in memory_ids  # CRITICAL: other user's private NOT visible


def test_batched_anchor_lookup_empty_fqns():
    """Batched lookup with empty FQN list returns empty dict."""
    result = batched_anchor_lookup(
        fqns=[],
        repo="repo",
        to_canonical_fn=lambda x: x,
        user_id="test_user",
    )

    assert result == {}


def test_batched_anchor_lookup_error_handling():
    """Batched lookup handles Qdrant errors gracefully (returns empty dict)."""
    with patch("memory_service.get_shared_service") as mock_service:
        mock_service.return_value._get_memory.side_effect = Exception("Qdrant down")

        result = batched_anchor_lookup(
            fqns=["Foo"],
            repo="repo",
            to_canonical_fn=lambda x: x,
            user_id="test_user",
        )

        # Non-fatal: returns empty dict, doesn't raise
        assert result == {}


def test_fusion_transport_uniformity():
    """CRITICAL: fusion never branches on transport (operates only on SystemAnswer)."""
    # Create SystemAnswers from different transports (in-process, http, mcp-bridge)
    # The compose function should treat them identically.

    # "In-process" answer (native engine)
    in_process_answer = SystemAnswer(
        system_name="code-native",
        system_version="native-v1",
        content="src/lib.py:10 Foo()",
    )

    # "HTTP" answer (CBM bridge)
    http_answer = SystemAnswer(
        system_name="code-cbm",
        system_version="cbm-v0.1",
        content="src/lib.py:10 Foo()",
    )

    # Both should produce identical fusion structure (only system_name differs)
    result_in_process = compose_fusion_answer(code_answer=in_process_answer)
    result_http = compose_fusion_answer(code_answer=http_answer)

    # Structure is identical (transport never mentioned)
    assert "[structure]" in result_in_process
    assert "[structure]" in result_http
    assert "transport" not in result_in_process.lower()
    assert "transport" not in result_http.lower()

    # Only the system_name attribution differs
    assert "(code-native)" in result_in_process
    assert "(code-cbm)" in result_http
