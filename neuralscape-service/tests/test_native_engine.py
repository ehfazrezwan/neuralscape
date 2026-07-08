"""Unit tests for NativeEngine (E2 native code-intel indexer)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from adapters.code_graph.engine import EngineCapabilityError, IndexReport
from adapters.code_graph.native_engine import NativeEngine


@pytest.fixture
def mock_bridge():
    """Mock Graphiti bridge with a fake Neo4j driver."""
    import asyncio

    bridge = Mock()
    bridge._loop = asyncio.new_event_loop()

    # Mock driver that returns empty results
    async def mock_run(cypher, **params):
        result = Mock()
        result.data = Mock(return_value=[])
        result.single = Mock(return_value=None)
        return result

    mock_session = Mock()
    mock_session.run = mock_run
    mock_session.__aenter__ = Mock(return_value=mock_session)
    mock_session.__aexit__ = Mock(return_value=None)

    mock_driver = Mock()
    mock_driver.session = Mock(return_value=mock_session)
    bridge.driver = mock_driver

    return bridge


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    settings = Mock()
    settings.code_graph_extracted_confidence = 0.9
    settings.code_graph_inferred_confidence = 0.7
    settings.code_graph_ambiguous_confidence = 0.5
    settings.code_graph_ambiguous_floor = 0.6
    settings.code_graph_max_communities = 10
    settings.code_graph_max_god_nodes = 15
    return settings


@pytest.fixture
def temp_repo():
    """Create a temporary repo with a simple Python file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        test_file = repo_path / "test_module.py"
        test_file.write_text(
            """
def greet(name):
    '''Say hello.'''
    return f"Hello, {name}"

class Greeter:
    def greet(self, name):
        return greet(name)
"""
        )
        yield repo_path


def test_native_engine_init(mock_bridge, mock_settings):
    """Test NativeEngine initialization."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )
    assert engine.code_space == "code--user123--testrepo"
    assert str(engine.repo_path) == "/tmp/test"


def test_native_engine_locate_implemented(mock_bridge, mock_settings):
    """Test that locate() now works in E3 (requires mocked embedding)."""
    from unittest.mock import patch, Mock
    import uuid

    # Mock the memory service
    mock_service = Mock()
    mock_memory = Mock()

    # Mock embedding model
    embedding_model = Mock()
    embedding_model.embed = Mock(return_value=[0.1] * 768)
    mock_memory.embedding_model = embedding_model

    # Mock vector store with Qdrant client
    vector_store = Mock()
    qdrant_client = Mock()

    # Mock empty hits
    qdrant_client.query_points = Mock(return_value=Mock(points=[]))
    qdrant_client.get_collection = Mock(side_effect=Exception("Not found"))
    qdrant_client.create_collection = Mock()

    vector_store.client = qdrant_client
    vector_store._has_bm25_slot = False
    mock_memory.vector_store = vector_store

    mock_service._get_memory = Mock(return_value=mock_memory)

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Should not raise EngineCapabilityError anymore (E3 implements it)
    with patch("memory_service.get_shared_service", return_value=mock_service):
        hits = engine.locate("test query")

    # Should return empty list (no hits from mocked Qdrant)
    assert isinstance(hits, list)
    assert len(hits) == 0


def test_native_engine_detect_changes_implemented(temp_repo, mock_bridge, mock_settings):
    """Test that detect_changes() works in E5 (persisted vs fresh)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock the _fetch_persisted_symbols and _parse_fresh_symbols
    with patch.object(engine, "_fetch_persisted_symbols") as mock_persisted:
        with patch.object(engine, "_parse_fresh_symbols") as mock_fresh:
            mock_persisted.return_value = []
            mock_fresh.return_value = []

            # Should not raise EngineCapabilityError anymore (E5 implements it)
            report = engine.detect_changes()

            # Should return a ChangeReport
            assert hasattr(report, "deleted_symbols")
            assert hasattr(report, "modified_symbols")
            assert hasattr(report, "added_symbols")
            assert hasattr(report, "affected_anchors")
            assert hasattr(report, "summary")


def test_native_engine_semantic_layer_implemented(mock_bridge, mock_settings):
    """Test that semantic_layer() reads stored properties and emits facts (I2)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock symbols with community_id and degree
    mock_symbols = [
        {"fqn": "mod.foo", "kind": "function", "file": "mod.py", "degree": 15, "community_id": 0},
        {"fqn": "mod.bar", "kind": "class", "file": "mod.py", "degree": 8, "community_id": 0},
        {"fqn": "util.baz", "kind": "function", "file": "util.py", "degree": 3, "community_id": 1},
        {"fqn": "main.run", "kind": "function", "file": "main.py", "degree": 1, "community_id": -1},  # singleton
    ]

    with patch.object(engine, "_run_cypher", return_value=mock_symbols):
        facts = engine.semantic_layer()

    # Should return SemanticFact objects
    assert isinstance(facts, list)
    assert len(facts) > 0

    # Should have community facts
    community_facts = [f for f in facts if f.category == "module"]
    assert len(community_facts) == 2  # communities 0 and 1

    # Should have hotspot facts (degree >= 10)
    hotspot_facts = [f for f in facts if f.category == "hotspot"]
    assert len(hotspot_facts) == 1  # only mod.foo has degree >= 10

    # Verify hotspot fact content
    hotspot = hotspot_facts[0]
    assert "mod.foo" in hotspot.content
    assert "degree 15" in hotspot.content
    # Hotspot facts are structure-derived → deductive (matches semantic.py)
    assert hotspot.epistemic_level == "deductive"

    # Community/module facts are inductive
    assert all(f.epistemic_level == "inductive" for f in community_facts)


def test_native_engine_export_snapshot_implemented(mock_bridge, mock_settings):
    """Test that export_snapshot() produces deterministic bytes (E6)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock fixture graph data
    fixture_nodes = [
        {"labels": ["CodeRepo"], "props": {"code_space": "code--user--repo", "name": "repo", "path": "/repo"}},
        {"labels": ["CodeSymbol"], "props": {"code_space": "code--user--repo", "fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "abc123"}},
    ]
    fixture_edges = [
        {
            "rel_type": "ANCHORED",
            "props": {},
            "source_labels": ["CodeSymbol"],
            "source_props": {"code_space": "code--user--repo", "fqn": "mod.foo"},
            "target_labels": ["CodeAnchor"],
            "target_props": {"code_space": "code--user--repo", "repo": "repo", "fqn": "mod.foo"},
        }
    ]

    with patch.object(engine, "_run_cypher") as mock_cypher:
        # First call: nodes query
        # Second call: edges query
        mock_cypher.side_effect = [fixture_nodes, fixture_edges]

        snapshot_bytes = engine.export_snapshot()

    # Verify it's compressed bytes
    assert isinstance(snapshot_bytes, bytes)
    assert len(snapshot_bytes) > 0

    # Verify deterministic: same input → same output
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [fixture_nodes, fixture_edges]
        snapshot_bytes2 = engine.export_snapshot()
    assert snapshot_bytes == snapshot_bytes2


def test_native_engine_parse_file(temp_repo, mock_bridge, mock_settings):
    """Test parsing a Python file with tree-sitter."""
    pytest.importorskip("tree_sitter_language_pack")

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    test_file = temp_repo / "test_module.py"
    symbols, edges = engine._parse_file(test_file, temp_repo, "python")

    # Should extract at least the function and class
    assert len(symbols) >= 2
    fqns = [s.fqn for s in symbols]
    assert "test_module.greet" in fqns
    assert "test_module.Greeter" in fqns

    # Should have some edges (at least the DEFINES for the method)
    assert len(edges) > 0


def test_export_import_snapshot_roundtrip(mock_bridge, mock_settings):
    """Test snapshot export → import round-trip preserves nodes/edges (E6)."""
    from unittest.mock import patch, call

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Fixture graph: 2 symbols, 1 anchor, 2 edges
    fixture_nodes = [
        {"labels": ["CodeRepo"], "props": {"code_space": "code--user--repo", "name": "repo", "path": "/repo"}},
        {"labels": ["CodeSymbol"], "props": {"code_space": "code--user--repo", "fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "abc", "degree": 1}},
        {"labels": ["CodeSymbol"], "props": {"code_space": "code--user--repo", "fqn": "mod.bar", "kind": "function", "file": "mod.py", "span": "7:10", "body_hash": "def", "degree": 0}},
        {"labels": ["CodeAnchor"], "props": {"code_space": "code--user--repo", "repo": "repo", "fqn": "mod.foo"}},
    ]
    fixture_edges = [
        {
            "rel_type": "CALLS",
            "props": {"extraction": "extracted", "epistemic_level": "explicit"},
            "source_labels": ["CodeSymbol"],
            "source_props": {"code_space": "code--user--repo", "fqn": "mod.foo"},
            "target_labels": ["CodeSymbol"],
            "target_props": {"code_space": "code--user--repo", "fqn": "mod.bar"},
        },
        {
            "rel_type": "ANCHORED",
            "props": {},
            "source_labels": ["CodeSymbol"],
            "source_props": {"code_space": "code--user--repo", "fqn": "mod.foo"},
            "target_labels": ["CodeAnchor"],
            "target_props": {"code_space": "code--user--repo", "repo": "repo", "fqn": "mod.foo"},
        },
    ]

    # Export
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [fixture_nodes, fixture_edges]
        snapshot_bytes = engine.export_snapshot()

    # Import into a fresh engine (simulate deployment)
    with patch.object(engine, "_run_cypher_with_retry") as mock_retry:
        engine.import_snapshot(snapshot_bytes)

        # Verify all nodes were MERGE'd (4 nodes)
        merge_calls = [c for c in mock_retry.call_args_list]
        assert len(merge_calls) >= 4  # 4 nodes + 2 edges

        # Check that repo, symbols, anchor were all merged
        # Extract both positional and keyword args
        cypher_texts = []
        for c in merge_calls:
            if c.args:
                cypher_texts.append(str(c.args[0]))
            elif c.kwargs and "cypher" in c.kwargs:
                cypher_texts.append(str(c.kwargs["cypher"]))

        # Debug: print actual calls if assertions fail
        if not any("CodeRepo" in c for c in cypher_texts):
            print(f"DEBUG: No CodeRepo found in {len(cypher_texts)} calls")
            for i, c in enumerate(cypher_texts):
                print(f"  Call {i}: {c[:100]}")

        assert any("CodeRepo" in c for c in cypher_texts), f"CodeRepo not found in calls: {cypher_texts}"
        assert any("CodeSymbol" in c for c in cypher_texts), f"CodeSymbol not found in calls: {cypher_texts}"
        assert any("CodeAnchor" in c for c in cypher_texts), f"CodeAnchor not found in calls: {cypher_texts}"


def test_import_snapshot_idempotent(mock_bridge, mock_settings):
    """Test that re-importing the same snapshot produces no duplicates (E6)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Minimal fixture
    fixture_nodes = [
        {"labels": ["CodeSymbol"], "props": {"code_space": "code--user--repo", "fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "abc"}},
    ]
    fixture_edges = []

    # Export
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [fixture_nodes, fixture_edges]
        snapshot_bytes = engine.export_snapshot()

    # Import twice
    with patch.object(engine, "_run_cypher_with_retry") as mock_retry:
        engine.import_snapshot(snapshot_bytes)
        first_call_count = mock_retry.call_count

        mock_retry.reset_mock()
        engine.import_snapshot(snapshot_bytes)
        second_call_count = mock_retry.call_count

        # Same number of MERGE calls (idempotent)
        assert first_call_count == second_call_count


def test_detect_changes_snapshot_based(temp_repo, mock_bridge, mock_settings):
    """Test detect_changes with snapshot baseline vs current (E6)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Create a snapshot with 2 symbols
    snapshot_symbols = [
        {"fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "old_hash"},
        {"fqn": "mod.bar", "kind": "function", "file": "mod.py", "span": "7:10", "body_hash": "bar_hash"},
    ]

    # Fresh parse has 3 symbols: foo (modified), bar (unchanged), baz (added)
    fresh_symbols = [
        {"fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "new_hash"},  # modified
        {"fqn": "mod.bar", "kind": "function", "file": "mod.py", "span": "7:10", "body_hash": "bar_hash"},  # unchanged
        {"fqn": "mod.baz", "kind": "function", "file": "mod.py", "span": "12:15", "body_hash": "baz_hash"},  # added
    ]

    # Create snapshot bytes
    import gzip
    import json
    snapshot_data = {
        "nodes": [
            {"labels": ["CodeSymbol"], "properties": s}
            for s in snapshot_symbols
        ],
        "edges": [],
    }
    header = {
        "format_version": "1.0",
        "code_space": "code--user--repo",
        "repo": "repo",
        "symbol_count": 2,
        "edge_count": 0,
        "content_hash": "dummy",
    }
    import hashlib
    snapshot_json = json.dumps(snapshot_data, sort_keys=True)
    header["content_hash"] = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
    envelope = {"header": header, "snapshot": snapshot_data}
    snapshot_bytes = gzip.compress(json.dumps(envelope, sort_keys=True).encode("utf-8"))

    with patch.object(engine, "_parse_fresh_symbols", return_value=fresh_symbols):
        with patch.object(engine, "_blast_radius_bfs", return_value=set()):
            report = engine.detect_changes(since=snapshot_bytes)

    # Verify changes detected
    assert "mod.foo" in report.modified_symbols  # body_hash changed
    assert "mod.baz" in report.added_symbols  # new symbol
    assert len(report.deleted_symbols) == 0  # nothing deleted
    assert "snapshot" in report.summary  # confirms snapshot baseline


def test_detect_changes_persisted_still_works(temp_repo, mock_bridge, mock_settings):
    """Test that E5's persisted-vs-fresh path still works after E6 (E5 regression test)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    persisted = [
        {"fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "abc"},
    ]
    fresh = [
        {"fqn": "mod.bar", "kind": "function", "file": "mod.py", "span": "7:10", "body_hash": "def"},
    ]

    with patch.object(engine, "_fetch_persisted_symbols", return_value=persisted):
        with patch.object(engine, "_parse_fresh_symbols", return_value=fresh):
            with patch.object(engine, "_blast_radius_bfs", return_value=set()):
                report = engine.detect_changes(since=None)

    # Verify E5 behavior: persisted baseline
    assert "mod.foo" in report.deleted_symbols
    assert "mod.bar" in report.added_symbols
    assert "persisted" in report.summary  # confirms persisted baseline


def test_snapshot_anchors_survive_roundtrip(mock_bridge, mock_settings):
    """Test that CodeAnchor nodes and ANCHORED edges survive snapshot round-trip (E6)."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user--repo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Fixture with anchor
    fixture_nodes = [
        {"labels": ["CodeSymbol"], "props": {"code_space": "code--user--repo", "fqn": "mod.foo", "kind": "function", "file": "mod.py", "span": "1:5", "body_hash": "abc"}},
        {"labels": ["CodeAnchor"], "props": {"code_space": "code--user--repo", "repo": "repo", "fqn": "mod.foo"}},
    ]
    fixture_edges = [
        {
            "rel_type": "ANCHORED",
            "props": {},
            "source_labels": ["CodeSymbol"],
            "source_props": {"code_space": "code--user--repo", "fqn": "mod.foo"},
            "target_labels": ["CodeAnchor"],
            "target_props": {"code_space": "code--user--repo", "repo": "repo", "fqn": "mod.foo"},
        }
    ]

    # Export
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [fixture_nodes, fixture_edges]
        snapshot_bytes = engine.export_snapshot()

    # Import and verify anchor was restored
    with patch.object(engine, "_run_cypher_with_retry") as mock_retry:
        engine.import_snapshot(snapshot_bytes)

        cypher_texts = [str(c.args[0]) if c.args else "" for c in mock_retry.call_args_list]
        # Verify CodeAnchor MERGE happened
        assert any("CodeAnchor" in c for c in cypher_texts)
        # Verify ANCHORED edge was created
        assert any("ANCHORED" in c for c in cypher_texts)


def test_native_engine_file_hash(temp_repo, mock_bridge, mock_settings):
    """Test file content hashing."""
    engine = NativeEngine(
        repo_path=str(temp_repo),
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    test_file = temp_repo / "test_module.py"
    hash1 = engine._file_hash(test_file)
    assert len(hash1) == 64  # SHA256 hex digest

    # Same file should have same hash
    hash2 = engine._file_hash(test_file)
    assert hash1 == hash2


def test_extraction_to_epistemic_mapping(mock_bridge, mock_settings):
    """Test confidence → epistemic level mapping."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    assert engine._extraction_to_epistemic("extracted") == "explicit"
    assert engine._extraction_to_epistemic("inferred") == "deductive"
    assert engine._extraction_to_epistemic("ambiguous") == "deductive"
    assert engine._extraction_to_epistemic("unknown") == "deductive"


def test_native_engine_query_no_symbols(mock_bridge, mock_settings):
    """Test query() when no symbols match."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _search_symbols to return empty
    engine._search_symbols = Mock(return_value=[])

    result = engine.query("nonexistent")
    assert "No symbols matching" in result
    assert "code--user123--testrepo" in result


def test_native_engine_neighbors_no_symbol(mock_bridge, mock_settings):
    """Test neighbors() when symbol not found."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _find_symbol to return empty
    engine._find_symbol = Mock(return_value=[])

    result = engine.neighbors("nonexistent")
    assert "No symbol matching" in result


def test_native_engine_path_no_source(mock_bridge, mock_settings):
    """Test path() when source symbol not found."""
    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock _find_symbol to return empty for source
    engine._find_symbol = Mock(return_value=[])

    result = engine.path("nonexistent", "target")
    assert "No symbol matching source" in result


def test_snapshot_cli_uses_correct_bridge_api(mock_bridge, mock_settings):
    """Test that snapshot_cli constructs engine with correct API (I3 bug fix).

    The bug was: snapshot_cli used service.graphiti_client and service.config,
    which don't exist. Should use service._bridge and module-level settings.
    """
    from unittest.mock import patch, MagicMock
    import tempfile
    from pathlib import Path

    # Mock get_shared_service
    mock_service = MagicMock()
    mock_service._bridge = mock_bridge

    with patch("memory_service.get_shared_service", return_value=mock_service):
        with patch("config.settings", mock_settings):
            with patch("adapters.code_graph.native_engine.NativeEngine") as MockEngine:
                from adapters.code_graph.snapshot_cli import export_snapshot, import_snapshot

                mock_engine = MagicMock()
                mock_engine.export_snapshot.return_value = b"snapshot_data"
                MockEngine.return_value = mock_engine

                # Test export_snapshot
                with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    export_snapshot("/tmp/repo", str(tmp_path), "code--user--test")

                    # Verify service._get_memory was called (initializes bridge)
                    mock_service._get_memory.assert_called()

                    # Verify NativeEngine was constructed with correct params
                    MockEngine.assert_called_with(
                        repo_path="/tmp/repo",
                        code_space="code--user--test",
                        bridge=mock_bridge,  # from service._bridge, not service.graphiti_client
                        settings=mock_settings,  # from module-level, not service.config
                    )

                    # Verify export was called
                    mock_engine.export_snapshot.assert_called_once()
                finally:
                    tmp_path.unlink(missing_ok=True)

                # Reset mocks
                mock_service.reset_mock()
                MockEngine.reset_mock()
                mock_engine.reset_mock()

                # Test import_snapshot
                with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp_path.write_bytes(b"snapshot_data")

                try:
                    import_snapshot(str(tmp_path), "code--user--test")

                    # Verify service._get_memory was called
                    mock_service._get_memory.assert_called()

                    # Verify NativeEngine was constructed with correct params
                    MockEngine.assert_called_with(
                        repo_path="/tmp",
                        code_space="code--user--test",
                        bridge=mock_bridge,  # from service._bridge
                        settings=mock_settings,  # from module-level
                    )

                    # Verify import was called
                    mock_engine.import_snapshot.assert_called_once_with(b"snapshot_data")
                finally:
                    tmp_path.unlink(missing_ok=True)


# ── I2: Louvain community tests ──────────────────────────────────────


def test_compute_communities_stable_ids(mock_bridge, mock_settings):
    """Test that community_id assignment is deterministic (same graph => same ids)."""
    from unittest.mock import patch, MagicMock

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Mock a connected graph: A -> B -> C, D -> E (2 components)
    edges = [
        {"source": "mod.A", "target": "mod.B"},
        {"source": "mod.B", "target": "mod.C"},
        {"source": "mod.D", "target": "mod.E"},
    ]
    all_symbols = [
        {"fqn": "mod.A"},
        {"fqn": "mod.B"},
        {"fqn": "mod.C"},
        {"fqn": "mod.D"},
        {"fqn": "mod.E"},
        {"fqn": "mod.F"},  # isolated symbol
    ]

    # Run community computation twice, capture the batch data from the UNWIND call
    batch_data_list = []

    # First run
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [edges, all_symbols, []]
        engine._compute_communities()

        # Find the call with batch parameter
        for call in mock_cypher.call_args_list:
            if "batch" in call.kwargs:
                batch = sorted(call.kwargs["batch"], key=lambda x: x["fqn"])
                batch_data_list.append(batch)

    # Second run
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [edges, all_symbols, []]
        engine._compute_communities()

        # Find the call with batch parameter
        for call in mock_cypher.call_args_list:
            if "batch" in call.kwargs:
                batch = sorted(call.kwargs["batch"], key=lambda x: x["fqn"])
                batch_data_list.append(batch)

    # Verify both runs produced identical assignments
    assert len(batch_data_list) == 2
    assert batch_data_list[0] == batch_data_list[1]

    # Verify isolated symbol got community_id = -1
    batch_data = batch_data_list[0]
    isolated = [b for b in batch_data if b["fqn"] == "mod.F"]
    assert len(isolated) == 1
    assert isolated[0]["community_id"] == -1


def test_compute_communities_guards_large_graphs(mock_bridge, mock_settings):
    """Test that >200k edge graphs are skipped with a warning."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Lightweight stand-in for a >200k-edge result: the guard only calls len(),
    # then returns immediately — no need to allocate 200k real elements.
    class _FakeEdges:
        def __len__(self):
            return 200_001

    with patch.object(engine, "_run_cypher", return_value=_FakeEdges()):
        with patch("adapters.code_graph.native_engine.logger") as mock_logger:
            engine._compute_communities()

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "200k limit" in warning_msg or "200000" in warning_msg


def test_compute_communities_empty_graph(mock_bridge, mock_settings):
    """Test that empty graphs (no CALLS/IMPORTS) still populate community_id = -1.

    The no-edges early return MUST NOT leave community_id unset — otherwise
    semantic_layer() (which filters community_id IS NOT NULL) drops all symbols.
    """
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    with patch.object(engine, "_run_cypher", return_value=[]) as mock_cypher:
        with patch.object(engine, "_run_cypher_with_retry") as mock_retry:
            with patch("adapters.code_graph.native_engine.logger") as mock_logger:
                engine._compute_communities()

                # Verify info log about no graph
                assert mock_logger.info.called

                # Verify singleton community_id = -1 is persisted on all symbols
                assert mock_retry.called
                persist_cypher = mock_retry.call_args[0][0]
                assert "SET s.community_id = -1" in persist_cypher


def test_reindex_stable_community_ids(mock_bridge, mock_settings):
    """Test that reindexing with unchanged graph keeps community_id stable."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Fixed graph: A <-> B, C <-> D
    edges = [
        {"source": "mod.A", "target": "mod.B"},
        {"source": "mod.B", "target": "mod.A"},
        {"source": "mod.C", "target": "mod.D"},
        {"source": "mod.D", "target": "mod.C"},
    ]
    all_symbols = [
        {"fqn": "mod.A"},
        {"fqn": "mod.B"},
        {"fqn": "mod.C"},
        {"fqn": "mod.D"},
    ]

    # Run twice, capture assignments
    assignments = []

    # First run
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [edges, all_symbols, []]
        engine._compute_communities()

        # Extract batch from the UNWIND call
        for call in mock_cypher.call_args_list:
            if "batch" in call.kwargs:
                batch = sorted(call.kwargs["batch"], key=lambda x: x["fqn"])
                assignments.append({b["fqn"]: b["community_id"] for b in batch})

    # Second run
    with patch.object(engine, "_run_cypher") as mock_cypher:
        mock_cypher.side_effect = [edges, all_symbols, []]
        engine._compute_communities()

        # Extract batch from the UNWIND call
        for call in mock_cypher.call_args_list:
            if "batch" in call.kwargs:
                batch = sorted(call.kwargs["batch"], key=lambda x: x["fqn"])
                assignments.append({b["fqn"]: b["community_id"] for b in batch})

    # Verify both runs produced identical assignments
    assert len(assignments) == 2
    assert assignments[0] == assignments[1]


def test_semantic_layer_no_communities(mock_bridge, mock_settings):
    """Test semantic_layer() gracefully handles no community data."""
    from unittest.mock import patch

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    with patch.object(engine, "_run_cypher", return_value=[]):
        facts = engine.semantic_layer()

    # Should return empty list, not crash
    assert isinstance(facts, list)
    assert len(facts) == 0


def test_semantic_layer_respects_config_limits(mock_bridge, mock_settings):
    """Test that semantic_layer() respects max_communities and max_god_nodes settings."""
    from unittest.mock import patch

    # Override settings
    mock_settings.code_graph_max_communities = 2
    mock_settings.code_graph_max_god_nodes = 1

    engine = NativeEngine(
        repo_path="/tmp/test",
        code_space="code--user123--testrepo",
        bridge=mock_bridge,
        settings=mock_settings,
    )

    # Create 3 communities with varying sizes
    mock_symbols = [
        # Community 0: 5 symbols (largest)
        {"fqn": "c0.a", "kind": "function", "file": "c0.py", "degree": 20, "community_id": 0},
        {"fqn": "c0.b", "kind": "function", "file": "c0.py", "degree": 15, "community_id": 0},
        {"fqn": "c0.c", "kind": "function", "file": "c0.py", "degree": 12, "community_id": 0},
        {"fqn": "c0.d", "kind": "function", "file": "c0.py", "degree": 8, "community_id": 0},
        {"fqn": "c0.e", "kind": "function", "file": "c0.py", "degree": 5, "community_id": 0},
        # Community 1: 3 symbols
        {"fqn": "c1.a", "kind": "function", "file": "c1.py", "degree": 11, "community_id": 1},
        {"fqn": "c1.b", "kind": "function", "file": "c1.py", "degree": 3, "community_id": 1},
        {"fqn": "c1.c", "kind": "function", "file": "c1.py", "degree": 2, "community_id": 1},
        # Community 2: 2 symbols (smallest, should be excluded)
        {"fqn": "c2.a", "kind": "function", "file": "c2.py", "degree": 4, "community_id": 2},
        {"fqn": "c2.b", "kind": "function", "file": "c2.py", "degree": 1, "community_id": 2},
    ]

    with patch.object(engine, "_run_cypher", return_value=mock_symbols):
        facts = engine.semantic_layer()

    # Should have exactly 2 community facts (max_communities=2)
    community_facts = [f for f in facts if f.category == "module"]
    assert len(community_facts) == 2

    # Should have exactly 1 hotspot fact (max_god_nodes=1), the highest degree
    hotspot_facts = [f for f in facts if f.category == "hotspot"]
    assert len(hotspot_facts) == 1
    assert "c0.a" in hotspot_facts[0].content  # degree 20 is highest
