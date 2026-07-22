"""CLI runner for mem0 LoCoMo evaluation (Track B).

Usage:
    python -m trackb.mem0_locomo.run --target http://localhost:8398 --phases ingest answer judge report

Phases:
    ingest: Load conversations into NS (async writes -> poll)
    answer: Answer QA items via /v1/ask
    judge: LLM-judge answers for correctness
    report: Generate JSON + Markdown reports

Arguments:
    --target: NS API base URL
    --token: Bearer token (optional for local dev)
    --dataset: Path to locomo10.json (defaults to ../../datasets/locomo/locomo10.json)
    --sample N: Use stratified sample of N QA items (default: all)
    --seed: Random seed for sampling
    --k: Retrieval top-k (default: 10)
    --reasoning-level: NS /v1/ask reasoning level (default: high)
    --concurrency: Parallel requests (default: 2 for ingest, 5 for answer/judge)
    --phases: Space-separated phases to run (default: all)
    --output-dir: Output directory (default: ./trackb_results)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import IngestManifest, read_jsonl_records
from neuralscape_bench.accuracy.sampling import stratified_sample

from . import __version__
from .loader import load_locomo
from .ingest import ingest_suite
from .answer import answer_suite
from .judge import judge_suite
from .report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="mem0 LoCoMo evaluation harness (Track B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="NS API base URL (e.g., http://localhost:8398)")
    parser.add_argument("--token", help="Bearer token (optional)")
    parser.add_argument("--dataset", type=Path, help="Path to locomo10.json")
    parser.add_argument("--sample", type=int, help="Stratified sample size (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument("--reasoning-level", default="high", help="NS reasoning level (minimal/low/medium/high)")
    parser.add_argument("--concurrency-ingest", type=int, default=2, help="Ingest concurrency")
    parser.add_argument("--concurrency-answer", type=int, default=5, help="Answer concurrency")
    parser.add_argument("--concurrency-judge", type=int, default=5, help="Judge concurrency")
    parser.add_argument("--phases", nargs="+", default=["ingest", "answer", "judge", "report"],
                        choices=["ingest", "answer", "judge", "report"],
                        help="Phases to run")
    parser.add_argument("--output-dir", type=Path, default=Path("./trackb_results"),
                        help="Output directory")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Resolve dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        # Default relative to this file: ../../datasets/locomo/locomo10.json
        here = Path(__file__).parent
        dataset_path = here / "../../datasets/locomo/locomo10.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}", file=sys.stderr)
        print("Specify --dataset path to locomo10.json", file=sys.stderr)
        sys.exit(1)

    # Output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "ingest_manifest.json"
    answers_path = args.output_dir / "answers.jsonl"
    judged_path = args.output_dir / "judged.jsonl"
    report_json = args.output_dir / "report.json"
    report_md = args.output_dir / "report.md"

    print(f"=== mem0 LoCoMo Evaluation (Track B) v{__version__} ===")
    print(f"Target: {args.target}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Phases: {', '.join(args.phases)}")
    print()

    # Load dataset
    print("Loading dataset...")
    data = load_locomo(dataset_path)
    print(f"Loaded: {len(data.conversations)} conversations, {len(data.qa_items)} QA items")

    # Sample if requested
    if args.sample is not None:
        print(f"Sampling {args.sample} QA items (stratified by qtype, seed={args.seed})...")
        data.qa_items = stratified_sample(data.qa_items, args.sample, seed=args.seed)
        print(f"Sampled: {len(data.qa_items)} QA items")

    stats = data.stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print()

    # Initialize client
    client = NeuralscapeClient(args.target, token=args.token)

    try:
        # Phase: ingest
        if "ingest" in args.phases:
            print("=== PHASE: INGEST ===")
            manifest = IngestManifest(manifest_path)
            summary = await ingest_suite(
                client, data,
                manifest=manifest,
                concurrency=args.concurrency_ingest,
                poll_timeout_s=120.0,
                poll_interval_s=1.0,
            )
            print(f"Ingest summary: {json.dumps(summary, indent=2)}")
            print()

        # Phase: answer
        if "answer" in args.phases:
            print("=== PHASE: ANSWER ===")
            summary = await answer_suite(
                client, data,
                out_path=answers_path,
                k=args.k,
                reasoning_level=args.reasoning_level,
                concurrency=args.concurrency_answer,
            )
            print(f"Answer summary: {json.dumps(summary, indent=2)}")
            print()

        # Phase: judge
        if "judge" in args.phases:
            print("=== PHASE: JUDGE ===")
            if not answers_path.exists():
                print(f"Error: Answers file not found at {answers_path}", file=sys.stderr)
                print("Run --phases answer first", file=sys.stderr)
                sys.exit(1)

            records = read_jsonl_records(answers_path)
            judged = await judge_suite(
                records, data.qa_items,
                concurrency=args.concurrency_judge,
            )

            # Write judged records
            with open(judged_path, "w") as f:
                for rec in judged:
                    f.write(json.dumps(rec) + "\n")

            print(f"Judged {len(judged)} records -> {judged_path}")
            print()

        # Phase: report
        if "report" in args.phases:
            print("=== PHASE: REPORT ===")
            if not judged_path.exists():
                print(f"Error: Judged file not found at {judged_path}", file=sys.stderr)
                print("Run --phases judge first", file=sys.stderr)
                sys.exit(1)

            judged = read_jsonl_records(judged_path)
            report = generate_report(
                judged,
                backbone="neuralscape",
                judge="gemini-3.1-flash-lite",
                embedder="text-embedding-004",
                k=args.k,
                reasoning_level=args.reasoning_level,
                output_json=report_json,
                output_md=report_md,
            )

            print(f"Report written:")
            print(f"  JSON: {report_json}")
            print(f"  Markdown: {report_md}")
            print()
            print("=== RESULTS ===")
            print(f"Overall Accuracy: {report['metrics']['overall_accuracy']:.2%}")
            print(f"Retrieval R@{args.k}: {report['metrics']['retrieval_r_at_k']:.2%}")
            print("Category Accuracy:")
            for cat, acc in report['metrics']['category_accuracy'].items():
                print(f"  {cat}: {acc:.2%}")

    finally:
        await client.aclose()

    print()
    print("=== COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())
