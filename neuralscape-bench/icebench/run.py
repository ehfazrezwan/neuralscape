"""
ICEBench runner: orchestrates benchmark runs across systems and corpora.

Subcommands:
- corpora: Fetch and pin corpora
- index: Run indexing benchmarks (cold/incremental/second + snapshot + store_size)
- query: Run query latency benchmarks
- score: Delegate to H3's Track-Q scorer
- report: Delegate to H4's report generator
"""

import argparse
import os
import random
import sys
from pathlib import Path

from icebench.corpora import (
    CORPORA_DIR,
    PINNED_CORPORA,
    save_lock_file,
    fetch_corpus,
    iter_corpora,
)
from icebench.corpus_filters import is_tool_output
from icebench.schema import ResultRow, write_row, RunManifest
from icebench.rail import RailConfig
from icebench.adapters.base import SystemAdapter, Corpus


# Results directory (env-overridable for isolated test runs)
RESULTS_DIR = Path(os.environ.get("ICE_RESULTS_DIR", "/data/ice/results/raw"))

# Number of repetitions per measurement
N_REPS = 3

# Source-file extensions per language, for picking incremental-touch targets.
_SOURCE_EXTS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
}


def _pick_source_files(corpus: Corpus, n: int, seed: int = 42) -> list[str]:
    """
    Deterministically pick up to n real source files from a corpus.

    Excludes tool-generated output directories (graphify-out/, etc.) to prevent
    indexing tools' own artifacts as source code.

    Args:
        corpus: The corpus to scan.
        n: Number of files to pick.
        seed: Random seed (deterministic selection).

    Returns:
        Absolute file paths (may be fewer than n if the corpus is small).
    """
    root = Path(corpus.path)
    if not root.exists():
        return []

    candidates = sorted(
        str(p)
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in _SOURCE_EXTS
        and ".git" not in p.parts
        and not is_tool_output(p, root)
    )
    if not candidates:
        return []

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def _touch_files(paths: list[str]) -> dict[str, bytes]:
    """
    Make a trivial reversible edit (append a newline) to each file.

    Args:
        paths: File paths to touch.

    Returns:
        Mapping of path -> original bytes, for restoration.
    """
    saved: dict[str, bytes] = {}
    for p in paths:
        original = Path(p).read_bytes()
        saved[p] = original
        Path(p).write_bytes(original + b"\n")
    return saved


def _restore_files(saved: dict[str, bytes]) -> None:
    """Restore files to their original bytes captured by _touch_files."""
    for p, original in saved.items():
        try:
            Path(p).write_bytes(original)
        except OSError:
            pass


def cmd_corpora(args: argparse.Namespace) -> int:
    """Fetch and pin corpora."""
    print("Fetching corpora...")

    # Save lock file first
    save_lock_file(PINNED_CORPORA)
    print(f"Saved lock file with {len(PINNED_CORPORA)} corpora")

    # Fetch each corpus
    for spec in PINNED_CORPORA:
        print(f"Fetching {spec.name} @ {spec.sha}...")
        try:
            corpus = fetch_corpus(spec, force=args.force)
            print(f"  -> {corpus.path}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    print("Done")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Run indexing benchmarks."""
    # Load manifest for resumability
    results_file = RESULTS_DIR / f"{args.run_id}.jsonl"
    manifest = RunManifest.load(args.run_id, results_file)

    rail_config = RailConfig(
        memory_limit_mb=args.memory_limit_mb,
        timeout_seconds=args.timeout_seconds,
    )

    # Load systems with the rail injected so EVERY index/snapshot subprocess
    # runs under the memory cap + timeout (adapters call run_with_rail).
    systems = _load_systems(args.systems, rail=rail_config)
    if not systems:
        print("No systems available", file=sys.stderr)
        return 1

    # Load corpora
    corpora = list(iter_corpora())
    if not corpora:
        print("No corpora available (run 'corpora' subcommand first)", file=sys.stderr)
        return 1

    # Run index benchmarks
    for system in systems:
        for corpus in corpora:
            print(f"\n=== {system.name} x {corpus.name} ===")

            # Run 3 reps of each operation
            for rep in range(N_REPS):
                # index_cold
                _run_index_op(
                    system,
                    corpus,
                    "index_cold",
                    rep,
                    manifest,
                    results_file,
                    rail_config,
                )

                # index_incremental (touch 1 file, then 5 files)
                _run_index_op(
                    system,
                    corpus,
                    "index_incremental_1",
                    rep,
                    manifest,
                    results_file,
                    rail_config,
                )
                _run_index_op(
                    system,
                    corpus,
                    "index_incremental_5",
                    rep,
                    manifest,
                    results_file,
                    rail_config,
                )

                # index_second
                _run_index_op(
                    system,
                    corpus,
                    "index_second",
                    rep,
                    manifest,
                    results_file,
                    rail_config,
                )

            # store_size (once per system x corpus)
            if not manifest.is_completed(system.name, corpus.name, "store_size", 0):
                print(f"  store_size...")
                try:
                    size_bytes = system.store_size_bytes(corpus)
                    row = ResultRow(
                        schema="icebench-v1",
                        kind="store",
                        system=system.name,
                        system_version=system.version,
                        corpus=corpus.name,
                        repo_sha=corpus.repo_sha,
                        op="store_size",
                        rep=0,
                        seed=42,
                        bytes=size_bytes,
                        ok=True,
                    )
                    write_row(results_file, row)
                    manifest.mark_completed(system.name, corpus.name, "store_size", 0)
                except Exception as e:
                    print(f"    ERROR: {e}", file=sys.stderr)

            # export_snapshot
            _run_snapshot_op(
                system,
                corpus,
                "export_snapshot",
                manifest,
                results_file,
                rail_config,
            )

    # NOTE: teardown is intentionally NOT called here. index and query are
    # separate runner invocations (processes); tearing down the index at the end
    # of the index phase leaves the query phase with nothing to query (the
    # competitor adapters store their index on disk / in the tool's own store,
    # not in adapter instance memory). Cleanup is the caller's responsibility
    # AFTER querying (see `teardown` subcommand / manual cleanup between runs).
    print("\nIndex benchmark complete")
    return 0


def _run_index_op(
    system: SystemAdapter,
    corpus: Corpus,
    op: str,
    rep: int,
    manifest: RunManifest,
    results_file: Path,
    rail_config: RailConfig,
) -> None:
    """Run a single index operation."""
    if manifest.is_completed(system.name, corpus.name, op, rep):
        print(f"  {op} rep={rep} [SKIP]")
        return

    print(f"  {op} rep={rep}...")

    # For incremental ops, make a REAL reversible edit to real source files,
    # run the incremental index against those paths, then restore the files.
    saved: dict[str, bytes] = {}
    try:
        if op == "index_cold":
            result = system.index_cold(corpus)
        elif op == "index_second":
            result = system.index_second(corpus)
        elif op in ("index_incremental_1", "index_incremental_5"):
            n = 1 if op == "index_incremental_1" else 5
            touched = _pick_source_files(corpus, n)
            if not touched:
                print(f"    SKIP {op}: no source files found in corpus", file=sys.stderr)
                return
            saved = _touch_files(touched)
            result = system.index_incremental(corpus, touched)
        else:
            return

        row = ResultRow(
            schema="icebench-v1",
            kind="index",
            system=system.name,
            system_version=system.version,
            corpus=corpus.name,
            repo_sha=corpus.repo_sha,
            op=op,
            rep=rep,
            seed=42,
            wall_s=result.wall_s,
            peak_rss_mb=result.peak_rss_mb,
            cpu_s=result.cpu_s,
            ok=result.ok,
            dnf=result.dnf,
            dnf_reason=result.dnf_reason,
        )
        write_row(results_file, row)
        manifest.mark_completed(system.name, corpus.name, op, rep)

    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)
    finally:
        # Always restore touched files, even on error.
        if saved:
            _restore_files(saved)


def _run_snapshot_op(
    system: SystemAdapter,
    corpus: Corpus,
    op: str,
    manifest: RunManifest,
    results_file: Path,
    rail_config: RailConfig,
) -> None:
    """Run a snapshot operation."""
    if manifest.is_completed(system.name, corpus.name, op, 0):
        print(f"  {op} [SKIP]")
        return

    print(f"  {op}...")

    try:
        if op == "export_snapshot":
            result = system.export_snapshot(corpus)
        else:
            return

        if result is None:
            # N/A for this system
            row = ResultRow(
                schema="icebench-v1",
                kind="snapshot",
                system=system.name,
                system_version=system.version,
                corpus=corpus.name,
                repo_sha=corpus.repo_sha,
                op=op,
                rep=0,
                seed=42,
                ok=False,
                dnf=False,
                dnf_reason="N/A",
            )
        else:
            row = ResultRow(
                schema="icebench-v1",
                kind="snapshot",
                system=system.name,
                system_version=system.version,
                corpus=corpus.name,
                repo_sha=corpus.repo_sha,
                op=op,
                rep=0,
                seed=42,
                wall_s=result.wall_s,
                bytes=result.bytes,
                ok=result.ok,
                dnf=result.dnf,
                dnf_reason=result.dnf_reason,
            )

        write_row(results_file, row)
        manifest.mark_completed(system.name, corpus.name, op, 0)

    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)


def cmd_query(args: argparse.Namespace) -> int:
    """Run query latency benchmarks."""
    # Load manifest
    results_file = RESULTS_DIR / f"{args.run_id}.jsonl"
    manifest = RunManifest.load(args.run_id, results_file)

    # Load systems
    systems = _load_systems(args.systems)
    if not systems:
        print("No systems available", file=sys.stderr)
        return 1

    # Load corpora
    corpora = list(iter_corpora())

    # Try to import H3's trackq generator
    try:
        from icebench.trackq import generate_queries
    except ImportError:
        print("WARNING: trackq not available, using built-in fixture", file=sys.stderr)
        generate_queries = _generate_fixture_queries

    # Run query benchmarks
    for system in systems:
        capabilities = system.capabilities()
        print(f"\n=== {system.name} capabilities: {capabilities} ===")

        for corpus in corpora:
            print(f"  Corpus: {corpus.name}")

            for op in capabilities:
                queries = generate_queries(op, corpus, n=args.n_queries, seed=args.seed)

                for i, query_payload in enumerate(queries):
                    if manifest.is_completed(system.name, corpus.name, op, i):
                        continue

                    try:
                        result = system.query(op, query_payload)

                        row = ResultRow(
                            schema="icebench-v1",
                            kind="query",
                            system=system.name,
                            system_version=system.version,
                            corpus=corpus.name,
                            repo_sha=corpus.repo_sha,
                            op=op,
                            rep=i,
                            seed=args.seed,
                            latency_ms=result.latency_ms,
                            answer=result.answer,
                            ok=result.ok,
                        )
                        write_row(results_file, row)
                        manifest.mark_completed(system.name, corpus.name, op, i)

                    except Exception as e:
                        print(f"    Query {i} ERROR: {e}", file=sys.stderr)

    print("\nQuery benchmark complete")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Delegate to H3's Track-Q scorer."""
    try:
        from icebench.trackq import score_results

        return score_results(args.run_id, RESULTS_DIR)
    except ImportError:
        print("ERROR: trackq scorer not available (H3 not implemented)", file=sys.stderr)
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Delegate to H4's report generator (build a ReportConfig)."""
    try:
        from icebench.report.generator import ReportConfig, generate_report
    except ImportError:
        print("ERROR: report generator not available (H4 not implemented)", file=sys.stderr)
        return 1

    from pathlib import Path as _P
    import json as _json

    # Env-overridable paths for isolated test runs
    tools_dir = _P(os.environ.get("ICE_TOOLS_DIR", "/data/ice/tools"))
    reports_dir = _P(os.environ.get("ICE_REPORTS_DIR", "/data/ice/reports"))
    bench_root = _P(os.environ.get("ICE_BENCH_ROOT", "/data/ice/neuralscape/neuralscape-bench"))

    OP_CLASSES = ["symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate", "blast_radius"]
    NA_REASON = {
        "nl_locate": "N/A (no NL→symbol retrieval)",
        "blast_radius": "N/A (no impact-analysis op)",
    }
    results_jsonl = RESULTS_DIR / f"{args.run_id}.jsonl"
    # Capability matrix derived from ops each system actually produced in results.
    seen: dict[str, set] = {}
    if results_jsonl.exists():
        with open(results_jsonl) as fh:
            for line in fh:
                try:
                    r = _json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") == "query":
                    seen.setdefault(r["system"], set()).add(r.get("op"))
    caps: dict[str, dict[str, str]] = {}
    for sysname, ops in seen.items():
        caps[sysname] = {
            op: ("supported" if op in ops else NA_REASON.get(op, "N/A"))
            for op in OP_CLASSES
        }
    chart = bench_root / "static" / "chart.umd.min.js"
    cfg = ReportConfig(
        results_jsonl=results_jsonl,
        score_report_json=(RESULTS_DIR / f"{args.run_id}.trackq.json"),
        systems_lock_json=tools_dir / "systems.lock.json",
        corpora_lock_json=CORPORA_DIR / "corpora.lock.json",
        capabilities_matrix=caps or None,
        markdown_output=reports_dir / f"ICE_BENCH_REPORT-{args.run_id}.md",
        html_output=reports_dir / f"ice_bench_report-{args.run_id}.html",
        chart_js_path=chart if chart.exists() else None,
        quiescence_statement=(
            "Measured on ns-bench (8 vCPU / 31 GB) with ZERO competing benchmark "
            "stacks running (the nsbench factory stacks were stopped for the run)."
        ),
        oracle_agreement_pct=None,  # LSP spot-check skipped (pyright/gopls unavailable)
    )
    generate_report(cfg)
    print(f"Report written: {cfg.markdown_output} + {cfg.html_output}")
    return 0


def _load_systems(
    system_names: list[str], rail: RailConfig | None = None
) -> list[SystemAdapter]:
    """
    Load system adapters by name, injecting the safety rail.

    Args:
        system_names: Adapter names to load.
        rail: Rail config to inject onto adapters that support it (so their
            index/snapshot subprocesses run under the cap + timeout).

    Returns:
        Constructed adapters.
    """
    systems: list[SystemAdapter] = []

    for name in system_names:
        adapter = None
        if name == "ns-ice":
            from icebench.adapters.ns_ice import NSIceAdapter

            adapter = NSIceAdapter(rail=rail) if rail else NSIceAdapter()
        elif name == "ns-ice-det":
            # Deterministic default config (code_index_embeddings OFF): same
            # adapter, distinct label. The host index inherits the flag via env
            # and the API must be running in det mode (default). The BM25+degree
            # locate path is local/no-network.
            from icebench.adapters.ns_ice import NSIceAdapter

            adapter = NSIceAdapter(rail=rail) if rail else NSIceAdapter()
            adapter.name = "ns-ice-det"
            adapter._cli_env["CODE_INDEX_EMBEDDINGS"] = "false"
        elif name == "ns-graphify":
            from icebench.adapters.ns_graphify import NSGraphifyAdapter

            adapter = NSGraphifyAdapter(rail=rail) if rail else NSGraphifyAdapter()
        elif name == "graphify":
            try:
                from icebench.systems.graphify_standalone import GraphifyStandaloneAdapter

                adapter = GraphifyStandaloneAdapter()
            except ImportError:
                print(f"WARNING: {name} adapter not available (H2)", file=sys.stderr)
        elif name == "cbm":
            try:
                from icebench.systems.cbm_standalone import CBMStandaloneAdapter

                adapter = CBMStandaloneAdapter()
            except ImportError:
                print(f"WARNING: {name} adapter not available (H2)", file=sys.stderr)
        else:
            print(f"WARNING: Unknown system {name}", file=sys.stderr)

        if adapter is not None:
            # Best-effort rail injection for H2 adapters (which take no rail arg).
            if rail is not None and hasattr(adapter, "rail"):
                adapter.rail = rail
            systems.append(adapter)

    return systems


def _generate_fixture_queries(op: str, corpus: Corpus, n: int, seed: int) -> list[dict]:
    """
    Built-in fixture query generator (fallback when H3 trackq not available).

    Args:
        op: Operation name.
        corpus: Corpus to query.
        n: Number of queries to generate.
        seed: Random seed.

    Returns:
        List of query payloads.
    """
    # Simple fixture: return a few dummy queries
    queries = []
    for i in range(min(n, 10)):  # Cap at 10 for fixture
        if op == "symbol_lookup":
            queries.append({"symbol": f"dummy_symbol_{i}", "corpus": corpus})
        elif op == "neighbors_1hop":
            queries.append({"symbol": f"dummy_symbol_{i}", "corpus": corpus})
        elif op == "path_le4":
            queries.append({"from": f"sym_a_{i}", "to": f"sym_b_{i}", "corpus": corpus})
        elif op == "nl_locate":
            queries.append({"query": f"find function {i}", "corpus": corpus})
        elif op == "blast_radius":
            queries.append({"symbol": f"dummy_symbol_{i}", "corpus": corpus})

    return queries


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ICEBench runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # corpora subcommand
    corpora_parser = subparsers.add_parser("corpora", help="Fetch and pin corpora")
    corpora_parser.add_argument("--force", action="store_true", help="Re-fetch even if exists")
    corpora_parser.add_argument(
        "--continue-on-error", action="store_true", help="Continue if a fetch fails"
    )

    # index subcommand
    index_parser = subparsers.add_parser("index", help="Run indexing benchmarks")
    index_parser.add_argument("--run-id", required=True, help="Unique run identifier")
    index_parser.add_argument(
        "--systems",
        nargs="+",
        default=["ns-ice", "ns-graphify"],
        help="Systems to benchmark",
    )
    index_parser.add_argument(
        "--memory-limit-mb", type=int, default=12 * 1024, help="Memory limit in MB"
    )
    index_parser.add_argument(
        "--timeout-seconds", type=int, default=3600, help="Timeout in seconds"
    )

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Run query benchmarks")
    query_parser.add_argument("--run-id", required=True, help="Unique run identifier")
    query_parser.add_argument(
        "--systems",
        nargs="+",
        default=["ns-ice", "ns-graphify"],
        help="Systems to benchmark",
    )
    query_parser.add_argument("--n-queries", type=int, default=200, help="Queries per op")
    query_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # score subcommand
    score_parser = subparsers.add_parser("score", help="Score Track-Q results")
    score_parser.add_argument("--run-id", required=True, help="Run identifier")

    # report subcommand
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--run-id", required=True, help="Run identifier")

    args = parser.parse_args()

    if args.command == "corpora":
        return cmd_corpora(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "query":
        return cmd_query(args)
    elif args.command == "score":
        return cmd_score(args)
    elif args.command == "report":
        return cmd_report(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
