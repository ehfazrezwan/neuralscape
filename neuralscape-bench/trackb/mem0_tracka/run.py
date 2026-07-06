"""CLI entrypoint for mem0 Track A control runs.

Usage:
    python -m trackb.mem0_tracka.run \\
        --suite locomo \\
        --phase ingest --phase answer --phase judge --phase report \\
        --sample 10

Phases:
    ingest  - Load suite and ingest conversations into mem0
    answer  - Answer QA items using mem0 retrieval + LLM
    judge   - Judge answers with GeminiJudge
    report  - Generate JSON + markdown reports
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Lazy imports for mem0 to allow tests/CLI to run without it installed
MEM0_AVAILABLE = False
try:
    from mem0.memory.main import Memory as Mem0Memory
    MEM0_AVAILABLE = True
except ImportError:
    Mem0Memory = None  # type: ignore

from trackb.mem0_tracka.answer import GeminiAnswerer, answer_suite
from trackb.mem0_tracka.config import ANSWER_MODEL, ANSWER_TEMP, JUDGE_MODEL, get_config
from trackb.mem0_tracka.ingest import ingest_suite
from trackb.mem0_tracka.report import build_result, save_markdown, save_result

# Import NSBench loaders (read-only reuse)
try:
    from neuralscape_bench.accuracy.download import DATASETS_DIR
    from neuralscape_bench.accuracy.judge import GeminiJudge, judge_suite
    from neuralscape_bench.accuracy.manifest import read_jsonl_records
    from neuralscape_bench.accuracy.suites import get_suite
    NSBENCH_AVAILABLE = True
except ImportError:
    DATASETS_DIR = None  # type: ignore
    GeminiJudge = None  # type: ignore
    judge_suite = None  # type: ignore
    read_jsonl_records = None  # type: ignore
    get_suite = None  # type: ignore
    NSBENCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw"
PHASES = ("ingest", "answer", "judge", "report")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _dataset_dir(suite_name: str) -> Path:
    """Get dataset directory for a suite (shared with NSBench)."""
    if suite_name.startswith("longmemeval"):
        return DATASETS_DIR / "longmemeval"
    return DATASETS_DIR / suite_name


def _raw_paths(suite_name: str, run_label: str | None = None) -> tuple[Path, Path]:
    """(answers, judged) JSONL paths for a suite.

    ``run_label`` suffixes both filenames so distinct runs are fully isolated
    on a shared raw dir (mirrors ``neuralscape_bench.accuracy.run``). Default
    (``None``) keeps the unlabeled baseline paths.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-{run_label}" if run_label else ""
    return (
        RAW_DIR / f"answers-{suite_name}{suffix}.jsonl",
        RAW_DIR / f"judged-{suite_name}{suffix}.jsonl",
    )


async def run_suite(args, suite_name: str) -> None:
    """Run all requested phases for one suite."""
    if not NSBENCH_AVAILABLE:
        raise SystemExit(
            "NSBench not available. Install neuralscape-bench package or fix import path."
        )

    if not MEM0_AVAILABLE and any(p in args.phase for p in ("ingest", "answer")):
        raise SystemExit(
            "mem0 not available. Install mem0ai package (or ensure the mem0/ subtree is importable)."
        )

    # Load suite data
    suite = get_suite(suite_name)
    data = suite.load(
        _dataset_dir(suite_name),
        sample=args.sample,
        seed=args.seed,
        options={},
    )
    _log(f"[load] {suite_name}: {data.stats()}")

    answers_path, judged_path = _raw_paths(suite_name, args.run_label)

    # Ingest phase
    if "ingest" in args.phase:
        if args.dry_run:
            _log(f"[dry-run] would ingest {len(data.conversations)} conversations")
        else:
            mem0_cfg = get_config(vector_store_path=args.vector_store_path)
            config_dict = mem0_cfg.to_mem0_dict()
            summary = ingest_suite(
                Mem0Memory,
                config_dict,
                data,
                log=_log,
            )
            _log(f"[ingest] {suite_name}: {summary}")

    # Answer phase
    if "answer" in args.phase:
        if args.dry_run:
            _log(f"[dry-run] would answer {len(data.qa_items)} QA items")
        else:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise SystemExit("GOOGLE_API_KEY required for answer phase")

            mem0_cfg = get_config(vector_store_path=args.vector_store_path)
            config_dict = mem0_cfg.to_mem0_dict()

            answerer = GeminiAnswerer(
                api_key,
                model=ANSWER_MODEL,
                temperature=ANSWER_TEMP,
            )
            try:
                summary = await answer_suite(
                    Mem0Memory,
                    config_dict,
                    data,
                    answerer,
                    out_path=answers_path,
                    k=args.k,
                    concurrency=args.concurrency,
                    log=_log,
                )
                _log(f"[answer] {suite_name}: {summary}")
            finally:
                await answerer.aclose()

    # Judge phase
    if "judge" in args.phase:
        if args.dry_run:
            _log("[dry-run] would judge answered items")
        else:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise SystemExit("GOOGLE_API_KEY required for judge phase")

            # If the answer phase ran in this same invocation, the answers were
            # regenerated from scratch — drop any stale judged records so the
            # judge's (qa_id, qtype) resume-dedup can't skip the fresh answers.
            if "answer" in args.phase and judged_path.exists():
                judged_path.unlink()

            judge = GeminiJudge(api_key, model=JUDGE_MODEL)
            try:
                summary = await judge_suite(
                    judge,
                    data,
                    answers_path,
                    judged_path,
                    concurrency=args.concurrency,
                    log=_log,
                )
                _log(f"[judge] {suite_name}: {summary}")
            finally:
                await judge.aclose()

    # Report phase
    if "report" in args.phase:
        judged = read_jsonl_records(judged_path)
        if not judged:
            _log(f"[report] {suite_name}: no judged records, skipping")
            return

        result = build_result(
            suite_name,
            judged,
            config={
                "k": args.k,
                "sample": args.sample,
                "seed": args.seed,
                "run_label": args.run_label,
            },
            suite_stats=data.stats(),
            mem0_version="2.0.2",  # vendored subtree version
        )

        json_path = save_result(result)
        md_path = save_markdown(result)
        _log(f"[report] {suite_name} → {json_path}, {md_path}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (exposed so tests can verify real flags)."""
    ap = argparse.ArgumentParser(
        description="mem0 Track A control harness (vendored mem0 under NSBench)"
    )
    ap.add_argument(
        "--suite",
        required=True,
        choices=("locomo", "beam", "convomem"),
        help="Suite to run (LoCoMo, BEAM 100k, or ConvoMem)",
    )
    ap.add_argument(
        "--phase",
        action="append",
        default=None,
        choices=PHASES,
        help="Phase(s) to run, in order (repeatable). Default: all phases",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stratified QA sample size (default: full suite)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10, help="Retrieval R@k depth")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--run-label",
        default=None,
        help="Suffix for raw answers/judged files → isolate distinct runs on a "
             "shared raw dir (e.g. --run-label 2026-07-06). Default: baseline.",
    )
    ap.add_argument(
        "--vector-store-path",
        type=Path,
        default=None,
        help="Path for isolated Qdrant local storage (default: .mem0_tracka_qdrant/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without network calls",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.phase is None:
        args.phase = list(PHASES)

    _log(f"mem0 Track A control: {args.suite} (phases: {', '.join(args.phase)})")

    try:
        asyncio.run(run_suite(args, args.suite))
    except KeyboardInterrupt:
        _log("\n[interrupted]")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
