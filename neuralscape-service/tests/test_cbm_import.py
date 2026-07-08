"""Tests for CBM graph.db.zst migration importer (I3).

Tests the one-shot CBM SQLite → native label-space importer:
- Decompress .zst archives with size guards
- Validate SQLite schema
- Read CBM nodes/edges
- Write to Neo4j code label-space
- Reject malformed/oversized archives cleanly
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import guarded (feature degrades if code-graph extra not installed)
try:
    from adapters.code_graph.cbm_import import (
        CBMArchiveTooLarge,
        CBMImportError,
        MAX_DECOMPRESSED_SIZE,
        MAX_SQLITE_SIZE,
        _decompress_zst,
        _read_cbm_database,
        _validate_sqlite,
        _write_to_neo4j,
        import_cbm_archive,
    )

    CBM_AVAILABLE = True
except ImportError:
    CBM_AVAILABLE = False


@pytest.mark.skipif(not CBM_AVAILABLE, reason="code-graph extra not installed")
class TestCBMImport:
    """CBM import tests (requires code-graph extra)."""

    def test_decompress_zst_success(self):
        """Decompress a valid .zst file."""
        try:
            import zstandard as zstd
        except ImportError:
            pytest.skip("zstandard not installed")

        content = b"hello CBM world" * 100
        compressed = zstd.ZstdCompressor().compress(content)

        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(compressed)

        try:
            decompressed = _decompress_zst(tmp_path)
            assert decompressed == content
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_decompress_zst_missing_file(self):
        """Reject missing file."""
        with pytest.raises(CBMImportError, match="File not found"):
            _decompress_zst(Path("/nonexistent/file.zst"))

    def test_decompress_zst_oversized(self):
        """Reject archive that exceeds decompression cap."""
        try:
            import zstandard as zstd
        except ImportError:
            pytest.skip("zstandard not installed")

        # Create content that will exceed MAX_DECOMPRESSED_SIZE
        # (this would actually compress well, but we're testing the guard)
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Write a small compressed file that claims to decompress to huge size
            # zstd will enforce max_output_size during decompress
            compressor = zstd.ZstdCompressor()
            # Create data that's just under the limit, then we'll patch MAX in the call
            tmp.write(compressor.compress(b"x" * 1000))

        try:
            # Patch the max size to be smaller to trigger the guard
            with patch("adapters.code_graph.cbm_import.MAX_DECOMPRESSED_SIZE", 100):
                with pytest.raises(CBMArchiveTooLarge, match="Decompressed size exceeds"):
                    _decompress_zst(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_decompress_zst_invalid_archive(self):
        """Reject non-zst file."""
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"not a zst file")

        try:
            with pytest.raises(CBMImportError, match="Not a valid zstandard archive"):
                _decompress_zst(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_validate_sqlite_success(self):
        """Validate a well-formed CBM database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create minimal CBM schema
            conn = sqlite3.connect(tmp_path)
            conn.execute("""
                CREATE TABLE nodes (
                    id INTEGER PRIMARY KEY,
                    fqn TEXT,
                    kind TEXT,
                    file TEXT,
                    line INTEGER,
                    end_line INTEGER,
                    docstring TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE edges (
                    source_id INTEGER,
                    target_id INTEGER,
                    relation TEXT,
                    extraction TEXT
                )
            """)
            conn.commit()
            conn.close()

            # Should not raise
            _validate_sqlite(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_validate_sqlite_missing_tables(self):
        """Reject SQLite missing expected tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create DB with wrong schema
            conn = sqlite3.connect(tmp_path)
            conn.execute("CREATE TABLE wrong_table (id INTEGER)")
            conn.commit()
            conn.close()

            with pytest.raises(CBMImportError, match="missing expected tables"):
                _validate_sqlite(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_validate_sqlite_not_a_database(self):
        """Reject non-SQLite file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"not sqlite")

        try:
            with pytest.raises(CBMImportError, match="Not a valid SQLite database"):
                _validate_sqlite(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_read_cbm_database(self):
        """Read symbols and edges from a CBM database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create minimal CBM database with test data
            conn = sqlite3.connect(tmp_path)
            conn.execute("""
                CREATE TABLE nodes (
                    id INTEGER PRIMARY KEY,
                    fqn TEXT,
                    kind TEXT,
                    file TEXT,
                    line INTEGER,
                    end_line INTEGER,
                    docstring TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE edges (
                    source_id INTEGER,
                    target_id INTEGER,
                    relation TEXT,
                    extraction TEXT
                )
            """)

            # Insert test data
            conn.execute(
                "INSERT INTO nodes VALUES (1, 'main.py::foo', 'function', 'main.py', 10, 20, 'A function')"
            )
            conn.execute(
                "INSERT INTO nodes VALUES (2, 'main.py::bar', 'function', 'main.py', 30, 40, 'Another function')"
            )
            conn.execute(
                "INSERT INTO edges VALUES (1, 2, 'CALLS', 'extracted')"
            )
            conn.commit()
            conn.close()

            symbols, edges = _read_cbm_database(tmp_path)

            assert len(symbols) == 2
            assert symbols[0]["fqn"] == "main.py::foo"
            assert symbols[0]["kind"] == "function"
            assert symbols[0]["file"] == "main.py"
            assert symbols[0]["line"] == 10
            assert symbols[0]["end_line"] == 20

            assert len(edges) == 1
            assert edges[0]["source_fqn"] == "main.py::foo"
            assert edges[0]["target_fqn"] == "main.py::bar"
            assert edges[0]["relation"] == "CALLS"
            assert edges[0]["extraction"] == "extracted"

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_write_to_neo4j_mocked(self):
        """Write symbols and edges to Neo4j (mocked)."""
        symbols = [
            {
                "fqn": "test.py::func1",
                "kind": "function",
                "file": "test.py",
                "line": 1,
                "end_line": 10,
                "docstring": "",
            },
            {
                "fqn": "test.py::func2",
                "kind": "function",
                "file": "test.py",
                "line": 20,
                "end_line": 30,
                "docstring": "",
            },
        ]
        edges = [
            {
                "source_fqn": "test.py::func1",
                "target_fqn": "test.py::func2",
                "relation": "CALLS",
                "extraction": "extracted",
            }
        ]

        # Mock the bridge and settings
        mock_bridge = MagicMock()
        mock_settings = MagicMock()

        # Mock the NativeEngine's _run_cypher_with_retry method
        # Patch at import site inside the function
        with patch("adapters.code_graph.native_engine.NativeEngine._run_cypher_with_retry") as mock_run:
            _write_to_neo4j(symbols, edges, "code--user--test", mock_bridge, mock_settings)

            # Verify Cypher was executed (at least for symbols and edges)
            assert mock_run.call_count >= 2

    def test_write_to_neo4j_span_uses_colon(self):
        """Span is serialized as 'line:end_line' (colon), matching NativeEngine."""
        symbols = [
            {
                "fqn": "test.py::func1",
                "kind": "function",
                "file": "test.py",
                "line": 5,
                "end_line": 12,
                "docstring": "",
            },
        ]
        mock_bridge = MagicMock()
        mock_settings = MagicMock()

        with patch("adapters.code_graph.native_engine.NativeEngine._run_cypher_with_retry") as mock_run:
            _write_to_neo4j(symbols, [], "code--user--test", mock_bridge, mock_settings)

            # First call is the symbols batch; inspect the span in the params.
            _, kwargs = mock_run.call_args_list[0]
            span = kwargs["symbols"][0]["span"]
            assert span == "5:12", f"span should use colon, got {span!r}"

    def test_write_to_neo4j_drops_disallowed_relation(self):
        """A relation outside the whitelist is dropped, never interpolated (Cypher injection guard)."""
        symbols = [
            {"fqn": "a", "kind": "function", "file": "a.py", "line": 1, "end_line": 2, "docstring": ""},
            {"fqn": "b", "kind": "function", "file": "a.py", "line": 3, "end_line": 4, "docstring": ""},
        ]
        edges = [
            # Legit relation — should be written.
            {"source_fqn": "a", "target_fqn": "b", "relation": "CALLS", "extraction": "extracted"},
            # Malicious/unknown relation — must be dropped, never interpolated.
            {
                "source_fqn": "a",
                "target_fqn": "b",
                "relation": "CALLS]->() DETACH DELETE n //",
                "extraction": "extracted",
            },
        ]
        mock_bridge = MagicMock()
        mock_settings = MagicMock()

        with patch("adapters.code_graph.native_engine.NativeEngine._run_cypher_with_retry") as mock_run:
            _write_to_neo4j(symbols, edges, "code--user--test", mock_bridge, mock_settings)

            # Collect every cypher string passed to the driver.
            cyphers = [call.args[0] for call in mock_run.call_args_list]
            joined = "\n".join(cyphers)
            # The injection payload must never appear in any executed Cypher.
            assert "DETACH DELETE" not in joined
            # The legit CALLS edge must still be written.
            assert any("[r:CALLS]" in c for c in cyphers)

    def test_import_cbm_archive_end_to_end_mocked(self):
        """End-to-end import with mocked Neo4j."""
        try:
            import zstandard as zstd
        except ImportError:
            pytest.skip("zstandard not installed")

        # Create a minimal CBM database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_tmp:
            db_path = Path(db_tmp.name)

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY,
                fqn TEXT,
                kind TEXT,
                file TEXT,
                line INTEGER,
                end_line INTEGER,
                docstring TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE edges (
                source_id INTEGER,
                target_id INTEGER,
                relation TEXT,
                extraction TEXT
            )
        """)
        conn.execute(
            "INSERT INTO nodes VALUES (1, 'test.py::main', 'function', 'test.py', 1, 5, '')"
        )
        conn.commit()
        conn.close()

        # Compress it
        compressed = zstd.ZstdCompressor().compress(db_path.read_bytes())
        with tempfile.NamedTemporaryFile(suffix=".db.zst", delete=False) as zst_tmp:
            zst_path = Path(zst_tmp.name)
            zst_tmp.write(compressed)

        try:
            # Mock the service and bridge
            mock_service = MagicMock()
            mock_bridge = MagicMock()
            mock_service._bridge = mock_bridge
            mock_settings = MagicMock()

            with patch("memory_service.get_shared_service", return_value=mock_service):
                with patch("config.settings", mock_settings):
                    with patch("adapters.code_graph.native_engine.NativeEngine._run_cypher_with_retry"):
                        # Should not raise
                        import_cbm_archive(str(zst_path), "code--user--test", "user")

                        # Verify service was initialized
                        mock_service._get_memory.assert_called_once()

        finally:
            db_path.unlink(missing_ok=True)
            zst_path.unlink(missing_ok=True)

    def test_import_cbm_archive_file_not_found(self):
        """Reject import of missing file."""
        with pytest.raises(CBMImportError, match="File not found"):
            import_cbm_archive("/nonexistent.db.zst", "code--user--test", "user")

    def test_import_cbm_archive_no_bridge(self):
        """Reject import when Neo4j bridge unavailable."""
        try:
            import zstandard as zstd
        except ImportError:
            pytest.skip("zstandard not installed")

        # Mock service with no bridge
        mock_service = MagicMock()
        mock_service._bridge = None

        # Create a minimal valid zst file
        content = b"dummy content"
        compressed = zstd.ZstdCompressor().compress(content)

        with tempfile.NamedTemporaryFile(suffix=".db.zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(compressed)

        try:
            with patch("memory_service.get_shared_service", return_value=mock_service):
                with pytest.raises(CBMImportError, match="Neo4j unavailable"):
                    import_cbm_archive(str(tmp_path), "code--user--test", "user")
        finally:
            tmp_path.unlink(missing_ok=True)


@pytest.mark.skipif(not CBM_AVAILABLE, reason="code-graph extra not installed")
def test_decompress_zst_missing_zstandard_dep():
    """_decompress_zst raises CBMImportError when zstandard is not installed.

    The module imports fine without the extra (zstandard is imported lazily
    inside _decompress_zst), so we simulate the missing dependency by poisoning
    sys.modules so the `import zstandard` inside the function raises ImportError.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("No module named 'zstandard'")
        return real_import(name, *args, **kwargs)

    # Point at an existing file so we get past the exists() check and reach
    # the guarded `import zstandard`.
    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(b"anything")

    try:
        # Drop any cached zstandard module so the import statement re-runs.
        with patch.dict("sys.modules", {"zstandard": None}):
            with patch("builtins.__import__", side_effect=fake_import):
                with pytest.raises(CBMImportError, match="zstandard not installed"):
                    _decompress_zst(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.skipif(not CBM_AVAILABLE, reason="code-graph extra not installed")
def test_import_cbm_archive_owner_mismatch():
    """import_cbm_archive rejects a code_space that doesn't belong to owner."""
    with pytest.raises(CBMImportError, match="does not belong to owner"):
        import_cbm_archive("/tmp/whatever.db.zst", "code--alice--repo", "bob")
