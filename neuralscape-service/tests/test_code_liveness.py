"""Unit tests for E5 code-intel liveness (detect_changes + dreaming integration).

Tests:
1. detect_changes correctly classifies deleted/modified/added symbols
2. Blast-radius BFS reaches transitively-affected symbols
3. Liveness consumer flags ONLY anchored memories in blast radius (scope test)
4. Liveness flags are reversible (not hard deletes)
5. Liveness pass respects dreaming enable gate
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.code_graph.engine import ChangeReport
from adapters.code_graph.native_engine import NativeEngine


@pytest.fixture
def mock_bridge():
    """Mock Graphiti bridge with a fake Neo4j driver and event loop."""
    import asyncio
    import threading

    bridge = Mock()

    # Create a background event loop running in a thread
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    bridge._loop = loop

    # Track cypher queries for verification
    cypher_calls = []

    async def mock_run(cypher, **params):
        cypher_calls.append((cypher, params))
        result = Mock()

        # Return appropriate mocks based on query
        if "CodeSymbol" in cypher and "RETURN s.fqn" in cypher:
            # _fetch_persisted_symbols
            async def data_coro():
                return [
                    {
                        "fqn": "test_module.old_function",
                        "kind": "function",
                        "file": "test.py",
                        "span": "1:3",
                        "body_hash": "old_hash_123",
                    },
                    {
                        "fqn": "test_module.unchanged_function",
                        "kind": "function",
                        "file": "test.py",
                        "span": "5:7",
                        "body_hash": "unchanged_hash",
                    },
                    {
                        "fqn": "test_module.modified_function",
                        "kind": "function",
                        "file": "test.py",
                        "span": "9:11",
                        "body_hash": "old_modified_hash",
                    },
                ]
            result.data = data_coro
        elif "type(r) IN ['CALLS', 'IMPORTS']" in cypher:
            # _get_blast_neighbors
            fqn = params.get("fqn", "")
            if "modified_function" in fqn or "old_function" in fqn:
                # Modified function is called by other functions
                async def data_coro():
                    return [{"neighbor": "test_module.caller_function"}]
                result.data = data_coro
            else:
                async def data_coro():
                    return []
                result.data = data_coro
        else:
            async def data_coro():
                return []
            result.data = data_coro

        result.single = Mock(return_value=None)
        return result

    class AsyncContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    mock_session = Mock()
    mock_session.run = mock_run

    mock_driver = Mock()
    mock_driver.session = Mock(return_value=AsyncContextManager())
    bridge.driver = mock_driver
    bridge._cypher_calls = cypher_calls

    yield bridge

    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock()
    settings.code_graph_extracted_confidence = 0.9
    return settings


@pytest.fixture
def temp_repo():
    """Create a temporary repo with Python files (fresh parse)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        test_file = repo_path / "test.py"
        # Fresh version: old_function deleted, modified_function changed, new_function added
        test_file.write_text(
            """
def unchanged_function():
    '''Unchanged function.'''
    return "unchanged"

def modified_function():
    '''Modified function - body changed.'''
    return "MODIFIED"  # <-- changed line

def new_function():
    '''Newly added function.'''
    return "new"
"""
        )
        yield repo_path


def test_detect_changes_classifies_symbols(mock_bridge, mock_settings, temp_repo):
    """Test that detect_changes correctly classifies deleted/modified/added symbols."""
    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--test--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock tree-sitter availability
    with patch("adapters.code_graph.native_engine.NativeEngine._parse_file") as mock_parse:
        # Fresh parse returns: unchanged, modified (new hash), new_function
        mock_parse.return_value = (
            [
                Mock(
                    fqn="test_module.unchanged_function",
                    kind="function",
                    file="test.py",
                    line=2,
                    end_line=4,
                ),
                Mock(
                    fqn="test_module.modified_function",
                    kind="function",
                    file="test.py",
                    line=6,
                    end_line=8,
                ),
                Mock(
                    fqn="test_module.new_function",
                    kind="function",
                    file="test.py",
                    line=10,
                    end_line=12,
                ),
            ],
            [],
        )

        # Mock body_hash computation
        def mock_body_hash(rel_path, sym):
            if "unchanged" in sym.fqn:
                return "unchanged_hash"
            elif "modified" in sym.fqn:
                return "new_modified_hash"  # different from old
            else:
                return "new_hash"

        with patch.object(engine, "_compute_symbol_body_hash", side_effect=mock_body_hash):
            report = engine.detect_changes()

    # Verify classification
    assert "test_module.old_function" in report.deleted_symbols
    assert "test_module.modified_function" in report.modified_symbols
    assert "test_module.new_function" in report.added_symbols
    assert "test_module.unchanged_function" not in report.deleted_symbols
    assert "test_module.unchanged_function" not in report.modified_symbols


def test_blast_radius_bfs(mock_bridge, mock_settings, temp_repo):
    """Test that blast-radius BFS reaches transitively-affected symbols."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--test--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock the new single-query _blast_radius_bfs implementation (Fix 4)
    mock_result = [
        {
            "root_fqn": "test_module.modified_function",
            "callers": ["test_module.caller_function"],
            "callees": [],
        }
    ]

    # Test BFS directly
    roots = ["test_module.modified_function"]
    with patch.object(engine, "_run_cypher", return_value=mock_result):
        affected = engine._blast_radius_bfs(roots, max_depth=2)

    # Should include the root and its callers
    assert "test_module.modified_function" in affected
    assert "test_module.caller_function" in affected


def test_collect_affected_anchors(mock_bridge, mock_settings, temp_repo):
    """Test that affected anchors are correctly built from FQNs."""
    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--test--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    affected_fqns = {
        "test_module.modified_function",
        "test_module.caller_function",
    }
    anchors = engine._collect_affected_anchors(affected_fqns)

    # Anchors should be in format "<repo>::<fqn>"
    assert "repo::test_module.modified_function" in anchors
    assert "repo::test_module.caller_function" in anchors


def test_liveness_detect_affected_memories():
    """Test that liveness consumer detects memories anchored to affected symbols (using scroll)."""
    from extensions.dreaming.liveness import detect_affected_memories

    # Mock service
    mock_service = Mock()
    mock_memory = Mock()
    mock_vector_store = Mock()
    mock_client = Mock()

    # Mock Qdrant scroll response (no embedder call needed)
    mock_hit1 = Mock()
    mock_hit1.payload = {
        "id": "mem_123",
        "metadata": {
            "source_ref": {
                "external_id": "repo::test_module.modified_function",
            }
        },
    }
    mock_hit2 = Mock()
    mock_hit2.payload = {
        "id": "mem_456",
        "metadata": {
            "source_ref": {
                "external_id": "repo::test_module.caller_function",
            }
        },
    }

    # scroll returns (records, next_offset)
    mock_client.scroll = Mock(return_value=([mock_hit1, mock_hit2], None))

    mock_vector_store.client = mock_client
    mock_vector_store.collection_name = "test_collection"
    mock_memory.vector_store = mock_vector_store
    mock_service._get_memory = Mock(return_value=mock_memory)

    # Create change report
    change_report = ChangeReport(
        deleted_symbols=["test_module.old_function"],
        modified_symbols=["test_module.modified_function"],
        added_symbols=["test_module.new_function"],
        affected_anchors=[
            "repo::test_module.modified_function",
            "repo::test_module.caller_function",
        ],
        summary="test",
    )

    # Detect affected memories
    events = detect_affected_memories(
        mock_service, change_report, code_space="code--test--repo"
    )

    # Should detect 2 memories
    assert len(events) == 2
    assert events[0].memory_id == "mem_123"
    assert events[1].memory_id == "mem_456"
    assert events[0].anchor_key == "repo::test_module.modified_function"

    # Verify scroll was called (not query_points)
    mock_client.scroll.assert_called()
    assert not hasattr(mock_client, 'query_points') or not mock_client.query_points.called


def test_liveness_flags_are_reversible():
    """Test that liveness flags are reversible metadata patches, not hard deletes."""
    from extensions.dreaming.liveness import LivenessEvent, apply_liveness_events

    # Mock service
    mock_service = Mock()
    mock_memory = Mock()
    mock_vector_store = Mock()
    mock_client = Mock()
    mock_vector_store.client = mock_client
    mock_memory.vector_store = mock_vector_store
    mock_service._get_memory = Mock(return_value=mock_memory)

    # Create liveness event
    events = [
        LivenessEvent(
            memory_id="mem_123",
            anchor_key="repo::test_module.modified_function",
            reason="symbol modified",
            confidence=0.95,
        )
    ]

    # Apply events
    with patch("config.settings") as mock_settings:
        mock_settings.qdrant_collection = "test_collection"
        flagged = apply_liveness_events(mock_service, events, dry_run=False)

    # Verify set_payload was called (reversible metadata patch)
    assert flagged == 1
    mock_client.set_payload.assert_called_once()
    call_args = mock_client.set_payload.call_args

    # Verify it's a metadata patch, not a delete
    assert call_args[1]["key"] == "metadata"
    payload = call_args[1]["payload"]
    assert payload["code_liveness_stale"] is True
    assert "code_liveness_anchor" in payload
    assert "code_liveness_reason" in payload


def test_liveness_respects_dreaming_gate():
    """Test that liveness pass respects dreaming enable gate."""
    from extensions.dreaming.liveness import process_code_changes_for_liveness

    # Mock service
    mock_service = Mock()

    # Mock dreaming settings (disabled)
    with patch("extensions.dreaming.config.dreaming_settings") as mock_settings:
        mock_settings.enabled = False

        result = process_code_changes_for_liveness(
            mock_service, code_space="code--test--repo", dry_run=False
        )

    # Should skip when disabled
    assert result["flagged"] == 0
    assert "disabled" in result["summary"]


def test_liveness_scoped_invalidation():
    """Test that only memories in blast radius are flagged (scoped invalidation)."""
    from extensions.dreaming.liveness import detect_affected_memories

    # Mock service
    mock_service = Mock()
    mock_memory = Mock()
    mock_vector_store = Mock()
    mock_client = Mock()

    # Only memories with matching anchor_keys should be returned
    mock_hit1 = Mock()
    mock_hit1.payload = {
        "id": "mem_in_radius",
        "metadata": {
            "source_ref": {
                "external_id": "repo::test_module.modified_function",
            }
        },
    }

    # scroll returns (records, next_offset)
    mock_client.scroll = Mock(return_value=([mock_hit1], None))

    mock_vector_store.client = mock_client
    mock_vector_store.collection_name = "test_collection"
    mock_memory.vector_store = mock_vector_store
    mock_service._get_memory = Mock(return_value=mock_memory)

    # Change report with specific affected anchors
    change_report = ChangeReport(
        deleted_symbols=[],
        modified_symbols=["test_module.modified_function"],
        added_symbols=[],
        affected_anchors=["repo::test_module.modified_function"],  # Only this one
        summary="test",
    )

    events = detect_affected_memories(
        mock_service, change_report, code_space="code--test--repo"
    )

    # Should only detect the 1 memory in the blast radius
    assert len(events) == 1
    assert events[0].memory_id == "mem_in_radius"


def test_liveness_no_embedder_call():
    """Test that detect_affected_memories does NOT call the embedder (scroll-based)."""
    from extensions.dreaming.liveness import detect_affected_memories

    # Mock service
    mock_service = Mock()
    mock_memory = Mock()
    mock_vector_store = Mock()
    mock_client = Mock()

    # Mock embedding model (should NOT be called)
    embedding_model = Mock()
    embedding_model.embed = Mock()
    mock_memory.embedding_model = embedding_model

    # Mock scroll response
    mock_client.scroll = Mock(return_value=([], None))

    mock_vector_store.client = mock_client
    mock_vector_store.collection_name = "test_collection"
    mock_memory.vector_store = mock_vector_store
    mock_service._get_memory = Mock(return_value=mock_memory)

    change_report = ChangeReport(
        deleted_symbols=[],
        modified_symbols=[],
        added_symbols=[],
        affected_anchors=["repo::test.func"],
        summary="test",
    )

    detect_affected_memories(mock_service, change_report, code_space="code--test--repo")

    # ASSERT: embedder was NOT called
    embedding_model.embed.assert_not_called()


def _make_scroll_service(scroll_return):
    """Build a mock service whose vector_store.client.scroll returns scroll_return.

    The staged batch rows deliberately DO NOT carry the flag (matching
    consolidate.hydrate_pool's real shape: flat rows keyed on ``content``, no
    nested ``metadata``) — so a consumer that reads staged rows sees nothing and
    the test fails. The flag lives ONLY in the scrolled Qdrant payload.
    """
    mock_service = Mock()
    mock_memory = Mock()
    mock_vector_store = Mock()
    mock_client = Mock()
    mock_client.scroll = Mock(return_value=scroll_return)
    mock_vector_store.client = mock_client
    mock_vector_store.collection_name = "test_collection"
    mock_memory.vector_store = mock_vector_store
    mock_service._get_memory = Mock(return_value=mock_memory)
    return mock_service, mock_client


def test_gather_liveness_actions_reads_from_qdrant():
    """Consumer builds temporal_reframe from the SCROLLED payload, not staged rows.

    The staged rows use consolidate.hydrate_pool's real shape (key ``content``,
    no nested ``metadata``, no flag). If someone reverts to reading
    ``mem["data"]`` / ``mem["metadata"]`` this test fails: the flag is only in
    the Qdrant scroll payload.
    """
    from extensions.dreaming.sweep import _gather_code_liveness_actions

    # Raw Qdrant record: flag + text live in .payload; .id is the point id.
    flagged_record = Mock()
    flagged_record.id = "mem_stale"
    flagged_record.payload = {
        "data": "test_function does something",
        "metadata": {
            "code_liveness_stale": True,
            "code_liveness_anchor": "repo::test.func",
            "code_liveness_reason": "symbol deleted",
        },
    }
    mock_service, mock_client = _make_scroll_service(([flagged_record], None))

    # Staged batch rows: REAL hydrate_pool shape — flat, keyed on `content`,
    # NO nested metadata dict, NO flag. mem_stale is in this pool.
    batch = Mock()
    batch.pool = "test_pool"
    batch.memories = [
        {"memory_id": "mem_stale", "content": "test_function does something"},
        {"memory_id": "mem_fresh", "content": "other memory"},
    ]

    actions, flagged_ids = _gather_code_liveness_actions(mock_service, batch)

    # Reads via scroll, not staged-row metadata.
    mock_client.scroll.assert_called()
    assert len(actions) == 1
    assert actions[0]["type"] == "temporal_reframe"
    assert actions[0]["memory_ids"] == ["mem_stale"]
    assert actions[0]["content"]  # non-empty
    assert "[stale: symbol deleted]" in actions[0]["content"]
    assert "test_function does something" in actions[0]["content"]
    assert flagged_ids == ["mem_stale"]

    # Consumer must NOT clear flags itself (deferred to post-apply).
    mock_client.set_payload.assert_not_called()


def test_gather_liveness_actions_scopes_to_pool():
    """A flagged memory NOT staged in this pool is skipped (no cross-pool reframe)."""
    from extensions.dreaming.sweep import _gather_code_liveness_actions

    other_pool_record = Mock()
    other_pool_record.id = "mem_other_pool"
    other_pool_record.payload = {
        "data": "belongs to another pool",
        "metadata": {"code_liveness_stale": True, "code_liveness_reason": "symbol deleted"},
    }
    mock_service, _ = _make_scroll_service(([other_pool_record], None))

    batch = Mock()
    batch.pool = "test_pool"
    batch.memories = [{"memory_id": "mem_in_pool", "content": "x"}]

    actions, flagged_ids = _gather_code_liveness_actions(mock_service, batch)
    assert actions == []
    assert flagged_ids == []


def test_clear_liveness_flags_only_after_apply():
    """_clear_code_liveness_flags sets code_liveness_stale=False via set_payload."""
    from extensions.dreaming.sweep import _clear_code_liveness_flags

    mock_service, mock_client = _make_scroll_service(([], None))

    _clear_code_liveness_flags(mock_service, ["mem_stale"])

    mock_client.set_payload.assert_called_once()
    call_args = mock_client.set_payload.call_args
    assert call_args[1]["payload"]["code_liveness_stale"] is False
    assert call_args[1]["points"] == ["mem_stale"]
    assert call_args[1]["key"] == "metadata"


def test_clear_liveness_flags_noop_on_empty():
    """No memory_ids ⇒ no set_payload call (dry-run / failed-apply leaves flags set)."""
    from extensions.dreaming.sweep import _clear_code_liveness_flags

    mock_service, mock_client = _make_scroll_service(([], None))

    _clear_code_liveness_flags(mock_service, [])

    mock_client.set_payload.assert_not_called()


def _dream_pool_batch(pool="test_pool"):
    """A minimal staged PoolBatch stand-in for _dream_pool integration tests."""
    batch = Mock()
    batch.pool = pool
    batch.memories = [{"memory_id": "mem_stale", "content": "test_function does X"}]
    batch.new_count = 1
    batch.visibility = "private"
    batch.owner_user_id = "u1"
    batch.project_id = None
    batch.group_id = "g1"
    batch.workspace = "memory"
    return batch


def _dream_pool_settings():
    settings = Mock()
    settings.min_hours = 0
    settings.settling_minutes = 0
    settings.min_new_memories = 0
    settings.max_memories_per_pool = 100
    settings.strength_half_life_days = 30
    settings.prune_strength_threshold = 0.1
    settings.dynamics_enabled = False
    settings.auto_apply_confidence = 0.8
    settings.reflection_enabled = False
    settings.vault_pages_enabled = False
    settings.identity_card_enabled = False
    settings.surprisal_top_k = 0
    settings.dreams_dir = "/tmp/dreams"
    settings.vault_path = "/tmp/vault"
    return settings


@pytest.mark.asyncio
async def test_dream_pool_clears_flags_after_apply():
    """Live path (not dry_run): reframe applies, THEN the flag is cleared."""
    from extensions.dreaming import sweep

    flagged_record = Mock()
    flagged_record.id = "mem_stale"
    flagged_record.payload = {
        "data": "test_function does X",
        "metadata": {"code_liveness_stale": True, "code_liveness_reason": "symbol deleted"},
    }
    service, client = _make_scroll_service(([flagged_record], None))

    batch = _dream_pool_batch()

    async def fake_decide(b, llm):
        return []

    async def fake_apply(svc, b, actions, *, dry_run):
        # Echo the reframe back as applied (simulating a successful apply).
        return consolidate.ActionResult(applied=list(actions))

    from extensions.dreaming import consolidate

    with patch.object(sweep.gate, "get_gate_state", return_value={}), \
         patch.object(sweep.gate, "acquire_lock", return_value="tok"), \
         patch.object(sweep.gate, "release_lock"), \
         patch.object(sweep.gate, "record_completion"), \
         patch.object(sweep.consolidate, "hydrate_pool", side_effect=lambda s, b: b), \
         patch.object(sweep.consolidate, "stage_pool", side_effect=lambda b, r, **k: b), \
         patch.object(sweep.consolidate, "decide", side_effect=fake_decide), \
         patch.object(sweep.consolidate, "apply_actions", side_effect=fake_apply), \
         patch.object(sweep.consolidate, "reconcile_batch"), \
         patch.object(sweep.reflect, "render_diary_entry", return_value={}), \
         patch.object(sweep.reflect, "write_diary", return_value=None):
        report = await sweep._dream_pool(
            service=service, settings=_dream_pool_settings(), redis=MagicMock(),
            llm_call=Mock(), batch=batch, dry_run=False, force=True,
        )

    assert report.status == "dreamt"
    # Flag cleared AFTER apply, for the applied memory id.
    client.set_payload.assert_called_once()
    assert client.set_payload.call_args[1]["payload"]["code_liveness_stale"] is False
    assert client.set_payload.call_args[1]["points"] == ["mem_stale"]


@pytest.mark.asyncio
async def test_dream_pool_does_not_clear_flags_in_dry_run():
    """Dry run: reframe is proposed but the flag is NOT cleared (re-swept next time)."""
    from extensions.dreaming import consolidate, sweep

    flagged_record = Mock()
    flagged_record.id = "mem_stale"
    flagged_record.payload = {
        "data": "test_function does X",
        "metadata": {"code_liveness_stale": True, "code_liveness_reason": "symbol deleted"},
    }
    service, client = _make_scroll_service(([flagged_record], None))

    batch = _dream_pool_batch()

    async def fake_decide(b, llm):
        return []

    async def fake_apply(svc, b, actions, *, dry_run):
        return consolidate.ActionResult(applied=[{**a, "dry_run": True} for a in actions])

    with patch.object(sweep.gate, "get_gate_state", return_value={}), \
         patch.object(sweep.gate, "acquire_lock", return_value="tok"), \
         patch.object(sweep.gate, "release_lock"), \
         patch.object(sweep.gate, "record_completion"), \
         patch.object(sweep.consolidate, "hydrate_pool", side_effect=lambda s, b: b), \
         patch.object(sweep.consolidate, "stage_pool", side_effect=lambda b, r, **k: b), \
         patch.object(sweep.consolidate, "decide", side_effect=fake_decide), \
         patch.object(sweep.consolidate, "apply_actions", side_effect=fake_apply), \
         patch.object(sweep.consolidate, "reconcile_batch"), \
         patch.object(sweep.reflect, "render_diary_entry", return_value={}):
        report = await sweep._dream_pool(
            service=service, settings=_dream_pool_settings(), redis=MagicMock(),
            llm_call=Mock(), batch=batch, dry_run=True, force=True,
        )

    assert report.status == "dreamt"
    # Scroll happened (consumer ran) but NO flag was cleared in dry run.
    client.scroll.assert_called()
    client.set_payload.assert_not_called()


def test_process_code_changes_imports_correctly():
    """Test that process_code_changes_for_liveness uses the correct import path."""
    from extensions.dreaming.liveness import process_code_changes_for_liveness

    # Mock service
    mock_service = Mock()

    # Mock settings
    with patch("config.settings") as mock_settings:
        mock_settings.code_repos = {"test": "/tmp/test"}

        # Mock dreaming settings (enabled)
        with patch("extensions.dreaming.config.dreaming_settings") as mock_dream_settings:
            mock_dream_settings.enabled = True

            # Mock get_engine (should be from adapters.code_graph.query)
            with patch("adapters.code_graph.query.get_engine") as mock_get_engine:
                mock_engine = Mock()
                mock_engine.detect_changes = Mock(return_value=Mock(
                    deleted_symbols=[],
                    modified_symbols=[],
                    added_symbols=[],
                    affected_anchors=[],
                    summary="no changes",
                ))
                mock_get_engine.return_value = mock_engine

                result = process_code_changes_for_liveness(
                    mock_service, code_space="code--user123--test", dry_run=True
                )

                # Verify get_engine was called with correct signature
                mock_get_engine.assert_called_once()
                call_args = mock_get_engine.call_args
                assert call_args[0][0] == "repo:test"  # graph_id
                assert call_args[0][1] == "user123"  # user_id


def test_detect_changes_no_changes(mock_bridge):
    """Test detect_changes with no changes (all symbols unchanged)."""
    import asyncio
    import threading

    # Replace the mock_run in mock_bridge to return same symbols
    async def mock_run_no_changes(cypher, **params):
        result = Mock()
        # Same symbols in persisted and fresh
        if "CodeSymbol" in cypher and "RETURN s.fqn" in cypher:
            async def data_coro():
                return [
                    {
                        "fqn": "test.unchanged",
                        "kind": "function",
                        "file": "test.py",
                        "span": "1:3",
                        "body_hash": "same_hash",
                    }
                ]
            result.data = data_coro
        else:
            async def data_coro():
                return []
            result.data = data_coro
        result.single = Mock(return_value=None)
        return result

    class AsyncContextManager:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    mock_session = Mock()
    mock_session.run = mock_run_no_changes

    mock_bridge.driver.session = Mock(return_value=AsyncContextManager())

    settings = Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        test_file = repo_path / "test.py"
        test_file.write_text("def unchanged(): pass\n")

        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space="code--test--repo",
            bridge=mock_bridge,
            settings=settings,
        )

        with patch("adapters.code_graph.native_engine.NativeEngine._parse_file") as mock_parse:
            mock_parse.return_value = (
                [Mock(fqn="test.unchanged", kind="function", file="test.py", line=1, end_line=1)],
                [],
            )
            with patch.object(engine, "_compute_symbol_body_hash", return_value="same_hash"):
                report = engine.detect_changes()

        # No changes
        assert len(report.deleted_symbols) == 0
        assert len(report.modified_symbols) == 0
        assert len(report.added_symbols) == 0
        assert len(report.affected_anchors) == 0
