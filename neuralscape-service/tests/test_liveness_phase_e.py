"""Phase E liveness tests: generalized inventory-diff liveness across engines.

Tests that liveness works regardless of which engine (native/CBM/graphify) indexed.
The consumer (temporal_reframe) is unchanged; only the per-driver diff producer is new.
"""

import pytest
from unittest.mock import MagicMock, patch
from extensions.dreaming.liveness import (
    detect_inventory_diff_liveness,
    _fetch_anchor_inventory,
)


def test_detect_inventory_diff_liveness_deleted_symbols():
    """Inventory diff detects deleted symbols and flags affected memories."""
    # Mock engine with symbol inventory
    mock_engine = MagicMock()
    mock_engine.get_symbol_inventory.return_value = {
        "lib.Foo",
        "lib.Bar",
        # "lib.Baz" was deleted (present in previous_inventory but not current)
    }

    # Previous inventory had lib.Baz
    previous_inventory = {"lib.Foo", "lib.Bar", "lib.Baz"}

    # Mock MemoryService
    mock_service = MagicMock()
    mock_m = MagicMock()
    mock_client = MagicMock()
    mock_service._get_memory.return_value = mock_m
    mock_m.vector_store.client = mock_client
    mock_m.vector_store.collection_name = "test_collection"

    # Mock scroll to return a memory anchored to the deleted symbol
    mock_client.scroll.return_value = (
        [
            MagicMock(
                payload={
                    "id": "mem_baz",
                    "metadata": {
                        "source_ref": {"external_id": "repo::lib.Baz"},
                    },
                }
            )
        ],
        None,  # no next offset
    )

    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = True

        result = detect_inventory_diff_liveness(
            service=mock_service,
            code_space="code--user--repo",
            engine=mock_engine,
            previous_inventory=previous_inventory,
            dry_run=False,
        )

        assert result["flagged"] == 1
        assert "lib.Baz" in result["summary"] or "1 deleted symbols" in result["summary"]

        # Verify set_payload was called to flag the memory
        assert mock_client.set_payload.call_count == 1
        call_args = mock_client.set_payload.call_args
        assert call_args.kwargs["points"] == ["mem_baz"]
        assert call_args.kwargs["payload"]["code_liveness_stale"] is True


def test_detect_inventory_diff_liveness_no_deleted_symbols():
    """Inventory diff with no deleted symbols returns early."""
    mock_engine = MagicMock()
    mock_engine.get_symbol_inventory.return_value = {"lib.Foo", "lib.Bar"}

    previous_inventory = {"lib.Foo", "lib.Bar"}  # same as current

    mock_service = MagicMock()

    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = True

        result = detect_inventory_diff_liveness(
            service=mock_service,
            code_space="code--user--repo",
            engine=mock_engine,
            previous_inventory=previous_inventory,
            dry_run=False,
        )

        assert result["flagged"] == 0
        assert "no deleted symbols" in result["summary"]


def test_detect_inventory_diff_liveness_engine_without_method():
    """Engines without get_symbol_inventory return gracefully."""
    mock_engine = MagicMock()
    del mock_engine.get_symbol_inventory  # engine doesn't support it

    mock_service = MagicMock()

    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = True

        result = detect_inventory_diff_liveness(
            service=mock_service,
            code_space="code--user--repo",
            engine=mock_engine,
            dry_run=False,
        )

        assert result["flagged"] == 0
        assert "inventory method unavailable" in result["summary"]


def test_detect_inventory_diff_liveness_dreaming_disabled():
    """Liveness pass skipped when dreaming is disabled."""
    mock_engine = MagicMock()
    mock_service = MagicMock()

    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = False

        result = detect_inventory_diff_liveness(
            service=mock_service,
            code_space="code--user--repo",
            engine=mock_engine,
            dry_run=False,
        )

        assert result["flagged"] == 0
        assert "dreaming disabled" in result["summary"]


def test_detect_inventory_diff_liveness_dry_run():
    """Dry run reports what would be flagged without writing."""
    mock_engine = MagicMock()
    mock_engine.get_symbol_inventory.return_value = {"lib.Foo"}

    previous_inventory = {"lib.Foo", "lib.Bar"}  # lib.Bar deleted

    mock_service = MagicMock()
    mock_m = MagicMock()
    mock_client = MagicMock()
    mock_service._get_memory.return_value = mock_m
    mock_m.vector_store.client = mock_client
    mock_m.vector_store.collection_name = "test_collection"

    mock_client.scroll.return_value = (
        [
            MagicMock(
                payload={
                    "id": "mem_bar",
                    "metadata": {
                        "source_ref": {"external_id": "repo::lib.Bar"},
                    },
                }
            )
        ],
        None,
    )

    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = True

        result = detect_inventory_diff_liveness(
            service=mock_service,
            code_space="code--user--repo",
            engine=mock_engine,
            previous_inventory=previous_inventory,
            dry_run=True,  # DRY RUN
        )

        # Dry run counts the event but doesn't write
        assert result["flagged"] == 1
        assert "1 deleted symbols" in result["summary"]

        # Verify set_payload was NOT called (dry run)
        assert mock_client.set_payload.call_count == 0


def test_fetch_anchor_inventory():
    """_fetch_anchor_inventory retrieves canonical FQNs from CodeAnchor nodes."""
    code_space = "code--user--repo"

    with patch("adapters.code_graph.query.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        # Mock Cypher query result
        mock_engine._run_cypher.return_value = [
            {"fqn": "lib.Foo"},
            {"fqn": "lib.Bar"},
            {"fqn": None},  # should be filtered out
        ]

        result = _fetch_anchor_inventory(code_space)

        assert result == {"lib.Foo", "lib.Bar"}
        assert None not in result


def test_fetch_anchor_inventory_error_handling():
    """_fetch_anchor_inventory handles errors gracefully (returns empty set)."""
    code_space = "code--user--repo"

    with patch("adapters.code_graph.query.get_engine") as mock_get_engine:
        mock_get_engine.side_effect = Exception("Neo4j down")

        result = _fetch_anchor_inventory(code_space)

        # Non-fatal: returns empty set
        assert result == set()


def test_inventory_diff_liveness_multiple_engines():
    """Liveness works uniformly across native, CBM, graphify engines."""
    # This is a meta-test verifying the abstraction: the same
    # detect_inventory_diff_liveness function works with any engine
    # as long as it implements get_symbol_inventory().

    for engine_name in ["native", "cbm", "graphify"]:
        mock_engine = MagicMock()
        mock_engine.get_symbol_inventory.return_value = {"lib.Foo"}

        previous_inventory = {"lib.Foo", "lib.Bar"}  # lib.Bar deleted

        mock_service = MagicMock()
        mock_m = MagicMock()
        mock_client = MagicMock()
        mock_service._get_memory.return_value = mock_m
        mock_m.vector_store.client = mock_client
        mock_m.vector_store.collection_name = "test_collection"

        mock_client.scroll.return_value = ([], None)  # no memories found

        with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
            mock_settings.enabled = True

            result = detect_inventory_diff_liveness(
                service=mock_service,
                code_space=f"code--user--{engine_name}-repo",
                engine=mock_engine,
                previous_inventory=previous_inventory,
                dry_run=False,
            )

            # Same behavior regardless of engine (transport-agnostic)
            assert result["flagged"] == 0  # no memories found
            assert "1 deleted symbols" in result["summary"]
