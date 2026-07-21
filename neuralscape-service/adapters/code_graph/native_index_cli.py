"""Native index CLI — tree-sitter indexer entrypoint (E7).

Command-line interface for indexing a codebase into Neo4j using NativeEngine.
This is the spec-aligned index trigger (F2 §1: "index is pull/CI-driven").

Usage:
    python -m adapters.code_graph.native_index_cli \
        --repo-path /abs/path/to/repo \
        --code-space code--owner--repo \
        [--incremental]

Or:
    python -m adapters.code_graph.native_index_cli \
        --repo-path /abs/path/to/repo \
        --owner owner \
        --repo-name repo \
        [--incremental]

Prints JSON summary to stdout on success:
    {"code_space": "...", "symbols": N, "edges": N, "files": N,
     "incremental": bool, "wall_s": float}

Exit 0 on success, non-zero on failure.

Post-index: calls process_code_changes_for_liveness (best-effort/non-fatal).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for native indexing.

    Args:
        argv: CLI args (defaults to sys.argv[1:] for normal use).

    Returns:
        Exit code (0 on success, 1+ on failure).
    """
    parser = argparse.ArgumentParser(
        description="Index a codebase with NativeEngine (tree-sitter → Neo4j code label-space)"
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Absolute path to the repository root",
    )
    parser.add_argument(
        "--code-space",
        help="Partition key (code--owner--repo). Mutually exclusive with --owner + --repo-name.",
    )
    parser.add_argument(
        "--owner",
        help="Owner name (used with --repo-name to build code_space)",
    )
    parser.add_argument(
        "--repo-name",
        help="Repo name (used with --owner to build code_space)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental index (skip unchanged files). Default: full/cold.",
    )

    args = parser.parse_args(argv)

    # Validate code_space construction
    if args.code_space:
        if args.owner or args.repo_name:
            print(
                "Error: --code-space is mutually exclusive with --owner/--repo-name",
                file=sys.stderr,
            )
            return 1
        code_space = args.code_space
    elif args.owner and args.repo_name:
        code_space = f"code--{args.owner}--{args.repo_name}"
    else:
        print(
            "Error: must provide either --code-space OR (--owner + --repo-name)",
            file=sys.stderr,
        )
        return 1

    # Validate repo path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.is_dir():
        print(f"Error: repo-path does not exist: {repo_path}", file=sys.stderr)
        return 1

    # Construct the engine (correct API per snapshot_cli.py audit)
    try:
        from memory_service import get_shared_service
        from config import settings  # module-level, NOT service.config

        service = get_shared_service()
        service._get_memory()  # ensure Graphiti bridge init
        bridge = service._bridge
        if bridge is None:
            print("Error: Graphiti bridge not initialized (Neo4j unavailable)", file=sys.stderr)
            return 1

        from adapters.code_graph.native_engine import NativeEngine

        # Neo4j driver lives on the Graphiti client, not the _AsyncBridge.
        driver = getattr(getattr(service, "_graphiti", None), "driver", None)
        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space=code_space,
            bridge=bridge,
            settings=settings,
            driver=driver,
        )
    except Exception as e:
        print(f"Error constructing NativeEngine: {e}", file=sys.stderr)
        logger.exception("Engine construction failed")
        return 1

    # Run the index
    start = time.time()
    try:
        report = engine.index(source=str(repo_path), incremental=args.incremental)
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        logger.exception("Indexing failed")
        return 1

    wall_s = time.time() - start

    # Print JSON summary to stdout
    summary = {
        "code_space": code_space,
        "symbols": report.symbols_indexed,
        "edges": report.edges_indexed,
        "files": report.files_indexed,
        "incremental": args.incremental,
        "wall_s": round(wall_s, 3),
    }
    print(json.dumps(summary))

    # Post-index: trigger liveness pass (best-effort, non-fatal)
    try:
        from extensions.dreaming.liveness import process_code_changes_for_liveness

        liveness_report = process_code_changes_for_liveness(
            service,
            code_space=code_space,
        )
        logger.info("Liveness pass completed: %s", liveness_report.get("summary", ""))
    except Exception as e:
        # Non-fatal: log and continue
        logger.warning("Liveness pass failed (non-fatal): %s", e)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(main())
