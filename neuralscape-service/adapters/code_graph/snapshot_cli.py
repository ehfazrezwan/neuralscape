"""Snapshot CLI utilities for code-intel index export/import.

E6: Thin CLI/entry point for CI/CD workflows:
  - Export a repo's code graph snapshot to a file
  - Import a snapshot from a file into Neo4j

Example usage:
    # Export current index to snapshot
    python -m adapters.code_graph.snapshot_cli export /path/to/repo snapshot.dat

    # Import snapshot into Neo4j
    python -m adapters.code_graph.snapshot_cli import snapshot.dat code--user--repo
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export_snapshot(repo_path: str, output_file: str, code_space: str):
    """Export code graph snapshot to a file.

    Args:
        repo_path: Path to the repository (for engine init).
        output_file: Path to write the snapshot file.
        code_space: Partition key (code--{owner}--{repo}).
    """
    # Import here to avoid circular deps
    from adapters.code_graph.native_engine import NativeEngine
    from memory_service import get_shared_service

    service = get_shared_service()
    bridge = service.graphiti_client
    settings = service.config

    engine = NativeEngine(
        repo_path=repo_path,
        code_space=code_space,
        bridge=bridge,
        settings=settings,
    )

    logger.info(f"Exporting snapshot from {code_space}...")
    snapshot_bytes = engine.export_snapshot()

    output_path = Path(output_file)
    output_path.write_bytes(snapshot_bytes)
    logger.info(f"Snapshot exported to {output_file} ({len(snapshot_bytes)} bytes)")


def import_snapshot(input_file: str, code_space: str):
    """Import code graph snapshot from a file into Neo4j.

    Args:
        input_file: Path to the snapshot file.
        code_space: Partition key (code--{owner}--{repo}).
    """
    from adapters.code_graph.native_engine import NativeEngine
    from memory_service import get_shared_service

    service = get_shared_service()
    bridge = service.graphiti_client
    settings = service.config

    # Create engine (repo_path not needed for import, set dummy)
    engine = NativeEngine(
        repo_path="/tmp",
        code_space=code_space,
        bridge=bridge,
        settings=settings,
    )

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Snapshot file not found: {input_file}")
        sys.exit(1)

    snapshot_bytes = input_path.read_bytes()
    logger.info(f"Importing snapshot from {input_file} ({len(snapshot_bytes)} bytes)...")

    engine.import_snapshot(snapshot_bytes)
    logger.info(f"Snapshot imported successfully for {code_space}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code-intel snapshot export/import CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export snapshot to file")
    export_parser.add_argument("repo_path", help="Path to repository")
    export_parser.add_argument("output_file", help="Output snapshot file")
    export_parser.add_argument(
        "--code-space",
        default=None,
        help="Code space (default: code--user--{repo_name})",
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import snapshot from file")
    import_parser.add_argument("input_file", help="Input snapshot file")
    import_parser.add_argument("code_space", help="Code space (code--{owner}--{repo})")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "export":
        repo_path = args.repo_path
        code_space = args.code_space
        if not code_space:
            # Derive from repo name
            repo_name = Path(repo_path).name
            code_space = f"code--user--{repo_name}"
        export_snapshot(repo_path, args.output_file, code_space)
    elif args.command == "import":
        import_snapshot(args.input_file, args.code_space)


if __name__ == "__main__":
    main()
