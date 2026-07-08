"""CBM graph.db.zst migration importer (I3).

One-shot importer: CBM SQLite archive (graph.db.zst) → native label-space
nodes/edges under a target code_space. Reads the compressed SQLite database,
extracts symbols and relationships, and writes them into Neo4j using the same
label-space schema as NativeEngine.

CBM schema (synthesized from typical code-graph databases):
  - nodes table: id, fqn, kind, file, line, end_line, docstring
  - edges table: source_id, target_id, relation, extraction

Maps to native label-space:
  (:CodeSymbol {code_space, fqn, kind, file, span})
  CALLS | IMPORTS | DEFINES | INHERITS | REFERENCES edges

Usage:
    python -m adapters.code_graph.cbm_import graph.db.zst code--owner--repo owner
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Archive-bomb guards (mirroring ingest/archive.py patterns)
MAX_DECOMPRESSED_SIZE = 500 * 1024 * 1024  # 500 MB uncompressed
MAX_SQLITE_SIZE = 100 * 1024 * 1024  # 100 MB SQLite after decompression

# Allowed relationship types (native label-space). CBM edge relations are
# read from untrusted SQLite and interpolated into the Cypher relationship
# type, so they MUST be whitelisted — never interpolate an unvalidated string
# into a query. Any other value is logged and the edge is dropped.
ALLOWED_RELATIONS = frozenset(
    {"CALLS", "IMPORTS", "DEFINES", "INHERITS", "REFERENCES"}
)


class CBMImportError(Exception):
    """Base error for CBM import failures."""


class CBMArchiveTooLarge(CBMImportError):
    """Raised when the archive exceeds size caps."""


def _decompress_zst(input_path: Path) -> bytes:
    """Decompress a .zst file with archive-bomb guards.

    Args:
        input_path: Path to the .zst file.

    Returns:
        Decompressed bytes.

    Raises:
        CBMImportError: If zstandard not installed or file invalid.
        CBMArchiveTooLarge: If decompressed size exceeds cap.
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise CBMImportError(
            "zstandard not installed. Install with: pip install 'neuralscape-service[code-graph]'"
        ) from None

    if not input_path.exists():
        raise CBMImportError(f"File not found: {input_path}")

    try:
        dctx = zstd.ZstdDecompressor()
        with input_path.open("rb") as fh:
            # Decompress with size cap
            decompressed = dctx.decompress(fh.read(), max_output_size=MAX_DECOMPRESSED_SIZE)
    except zstd.ZstdError as e:
        raise CBMImportError(f"Not a valid zstandard archive: {e}") from e

    if len(decompressed) >= MAX_DECOMPRESSED_SIZE:
        raise CBMArchiveTooLarge(
            f"Decompressed size exceeds {MAX_DECOMPRESSED_SIZE} bytes"
        )

    return decompressed


def _validate_sqlite(db_path: Path):
    """Validate that the database is a valid SQLite file with expected schema.

    Args:
        db_path: Path to the SQLite database.

    Raises:
        CBMImportError: If not a valid SQLite DB or missing expected tables.
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError as e:
        raise CBMImportError(f"Not a valid SQLite database: {e}") from e

    try:
        cursor = conn.cursor()
        # Check for expected tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if "nodes" not in tables or "edges" not in tables:
            raise CBMImportError(
                f"SQLite database missing expected tables. Found: {tables}"
            )
    except sqlite3.DatabaseError as e:
        raise CBMImportError(f"Not a valid SQLite database: {e}") from e
    finally:
        conn.close()


def _read_cbm_database(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Read symbols and edges from a CBM SQLite database.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Tuple of (symbols, edges) as list of dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()

        # Read nodes/symbols
        cursor.execute("""
            SELECT id, fqn, kind, file, line, end_line, docstring
            FROM nodes
        """)
        symbols = []
        id_to_fqn = {}
        for row in cursor.fetchall():
            symbol = {
                "fqn": row["fqn"],
                "kind": row["kind"],
                "file": row["file"],
                "line": row["line"],
                "end_line": row["end_line"],
                "docstring": row["docstring"] or "",
            }
            symbols.append(symbol)
            id_to_fqn[row["id"]] = row["fqn"]

        # Read edges
        cursor.execute("""
            SELECT source_id, target_id, relation, extraction
            FROM edges
        """)
        edges = []
        for row in cursor.fetchall():
            source_fqn = id_to_fqn.get(row["source_id"])
            target_fqn = id_to_fqn.get(row["target_id"])
            if source_fqn and target_fqn:
                edge = {
                    "source_fqn": source_fqn,
                    "target_fqn": target_fqn,
                    "relation": row["relation"],
                    "extraction": row["extraction"] or "extracted",
                }
                edges.append(edge)
    finally:
        conn.close()

    logger.info(f"Read {len(symbols)} symbols and {len(edges)} edges from CBM database")
    return symbols, edges


def _write_to_neo4j(
    symbols: list[dict],
    edges: list[dict],
    code_space: str,
    bridge: Any,
    settings: Any,
    driver: Any = None,
):
    """Write CBM symbols and edges to Neo4j using batched UNWIND MERGE.

    Args:
        symbols: List of symbol dicts.
        edges: List of edge dicts.
        code_space: Partition key (code--{owner}--{repo}).
        bridge: Graphiti async bridge (provides the event loop). The real
            _AsyncBridge does NOT expose .driver.
        settings: Config object.
        driver: Graphiti Neo4j async driver (service._graphiti.driver). Required
            for live Neo4j execution; the bridge alone cannot run Cypher.
    """
    from adapters.code_graph.native_engine import NativeEngine

    # Create a temporary engine instance to reuse its Cypher execution methods
    engine = NativeEngine(
        repo_path="/tmp",  # not used for import
        code_space=code_space,
        bridge=bridge,
        settings=settings,
        driver=driver,
    )

    logger.info(f"Writing {len(symbols)} symbols to Neo4j (code_space={code_space})...")

    # Write symbols in batches using UNWIND MERGE
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        symbol_records = [
            {
                "code_space": code_space,
                "fqn": s["fqn"],
                "kind": s["kind"],
                "file": s["file"],
                # Canonical span format is "line:end_line" (colon), matching
                # NativeEngine (native_engine.py) — a dash would break span parsers.
                "span": f"{s['line']}:{s['end_line']}",
                "degree": 0,  # will be computed later if needed
            }
            for s in batch
        ]

        cypher = """
        UNWIND $symbols AS sym
        MERGE (s:CodeSymbol {code_space: sym.code_space, fqn: sym.fqn})
        SET s.kind = sym.kind,
            s.file = sym.file,
            s.span = sym.span,
            s.degree = sym.degree
        """
        engine._run_cypher_with_retry(cypher, symbols=symbol_records)

    logger.info(f"Writing {len(edges)} edges to Neo4j...")

    # Write edges in batches
    for i in range(0, len(edges), batch_size):
        batch = edges[i : i + batch_size]
        edge_records = [
            {
                "code_space": code_space,
                "src_fqn": e["source_fqn"],
                "tgt_fqn": e["target_fqn"],
                "relation": e["relation"],
                "extraction": e["extraction"],
            }
            for e in batch
        ]

        # Group by relation type for UNWIND (each relation needs separate query).
        # SECURITY: `relation` is untrusted SQLite data interpolated into the
        # Cypher relationship type, so it MUST be whitelisted — anything outside
        # ALLOWED_RELATIONS is logged and dropped (never interpolated).
        by_relation = {}
        for rec in edge_records:
            rel = rec["relation"]
            if rel not in ALLOWED_RELATIONS:
                logger.warning(
                    "Dropping CBM edge with disallowed relation %r "
                    "(%s -> %s); allowed: %s",
                    rel,
                    rec["src_fqn"],
                    rec["tgt_fqn"],
                    sorted(ALLOWED_RELATIONS),
                )
                continue
            if rel not in by_relation:
                by_relation[rel] = []
            by_relation[rel].append(rec)

        for relation, rels in by_relation.items():
            # relation is guaranteed to be in ALLOWED_RELATIONS here.
            cypher = f"""
            UNWIND $edges AS edge
            MERGE (src:CodeSymbol {{code_space: edge.code_space, fqn: edge.src_fqn}})
            MERGE (tgt:CodeSymbol {{code_space: edge.code_space, fqn: edge.tgt_fqn}})
            MERGE (src)-[r:{relation}]->(tgt)
            SET r.extraction = edge.extraction
            """
            engine._run_cypher_with_retry(cypher, edges=rels)

    logger.info("CBM import complete")


def import_cbm_archive(
    input_file: str,
    code_space: str,
    owner: str,
):
    """Import a CBM graph.db.zst archive into Neo4j.

    Args:
        input_file: Path to the graph.db.zst file.
        code_space: Partition key (code--{owner}--{repo}).
        owner: Owner ID — must match the code_space's owner segment so a caller
            cannot accidentally import into another owner's partition.

    Raises:
        CBMImportError: If code_space does not belong to ``owner``, the file is
            missing, or the archive/schema is invalid.
    """
    from config import settings
    from memory_service import get_shared_service

    # Guard: code_space MUST be owned by `owner` (format: code--{owner}--{repo}).
    # This keeps the two arguments coherent and prevents importing into an
    # unintended partition.
    expected_prefix = f"code--{owner}--"
    if not code_space.startswith(expected_prefix):
        raise CBMImportError(
            f"code_space {code_space!r} does not belong to owner {owner!r} "
            f"(expected it to start with {expected_prefix!r})"
        )

    input_path = Path(input_file)
    if not input_path.exists():
        raise CBMImportError(f"File not found: {input_file}")

    # Initialize service and bridge
    service = get_shared_service()
    service._get_memory()
    bridge = service._bridge
    if bridge is None:
        raise CBMImportError("Graphiti bridge not initialized (Neo4j unavailable)")
    driver = getattr(getattr(service, "_graphiti", None), "driver", None)

    # Decompress the archive
    logger.info(f"Decompressing {input_file}...")
    decompressed = _decompress_zst(input_path)

    # Check decompressed size
    if len(decompressed) > MAX_SQLITE_SIZE:
        raise CBMArchiveTooLarge(
            f"Decompressed SQLite exceeds {MAX_SQLITE_SIZE} bytes"
        )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(decompressed)

    try:
        # Validate SQLite schema
        logger.info("Validating CBM database schema...")
        _validate_sqlite(tmp_path)

        # Read CBM data
        logger.info("Reading CBM database...")
        symbols, edges = _read_cbm_database(tmp_path)

        # Write to Neo4j
        _write_to_neo4j(symbols, edges, code_space, bridge, settings, driver=driver)

        logger.info(f"Successfully imported {len(symbols)} symbols and {len(edges)} edges")

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


def main():
    """CLI entry point for CBM import."""
    parser = argparse.ArgumentParser(
        description="Import CBM graph.db.zst into Neuralscape code label-space"
    )
    parser.add_argument("input_file", help="Path to graph.db.zst file")
    parser.add_argument("code_space", help="Code space (code--{owner}--{repo})")
    parser.add_argument("owner", help="Owner ID for scoping")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        import_cbm_archive(args.input_file, args.code_space, args.owner)
    except CBMImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
