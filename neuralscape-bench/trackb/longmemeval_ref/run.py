#!/usr/bin/env python3
"""Track B: LongMemEval Reference Harness CLI.

Usage:
    python run.py --target http://localhost:8398 --token $BENCH_TOKEN \\
                  --judge-key $GOOGLE_API_KEY --sample 50

All phases are resumable. Run specific phases:
    python run.py --target http://localhost:8398 --phases ingest answer
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .harness import LMEHarness


def main():
    parser = argparse.ArgumentParser(
        description="Track B: LongMemEval Reference Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        required=True,
        help="NS API base URL (e.g., http://localhost:8398)",
    )
    parser.add_argument(
        "--token",
        help="Bearer token for NS API (optional for local dev)",
    )
    parser.add_argument(
        "--judge-key",
        help="GOOGLE_API_KEY for LLM judge (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-3.1-flash-lite",
        help="LLM judge model (default: gemini-3.1-flash-lite)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k for retrieval (default: 10)",
    )
    parser.add_argument(
        "--reasoning-level",
        default="high",
        choices=["minimal", "low", "medium", "high"],
        help="NS /v1/ask reasoning level (default: high)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N questions (stratified across types). Default: all 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["ingest", "answer", "judge", "report"],
        help="Phases to run (default: all)",
    )
    parser.add_argument(
        "--concurrency-ingest",
        type=int,
        default=2,
        help="Parallel conversations during ingest (default: 2)",
    )
    parser.add_argument(
        "--concurrency-answer",
        type=int,
        default=2,
        help="Parallel questions during answer (default: 2)",
    )
    parser.add_argument(
        "--concurrency-judge",
        type=int,
        default=4,
        help="Parallel judge calls (default: 4)",
    )

    args = parser.parse_args()

    # Resolve judge key from arg or env
    judge_key = args.judge_key or os.environ.get("GOOGLE_API_KEY")
    if not judge_key and (not args.phases or "judge" in args.phases):
        print("Error: --judge-key or GOOGLE_API_KEY required for judge phase", file=sys.stderr)
        sys.exit(1)

    harness = LMEHarness(
        target=args.target,
        token=args.token,
        judge_key=judge_key,
        judge_model=args.judge_model,
        k=args.k,
        reasoning_level=args.reasoning_level,
        sample=args.sample,
        seed=args.seed,
        concurrency_ingest=args.concurrency_ingest,
        concurrency_answer=args.concurrency_answer,
        concurrency_judge=args.concurrency_judge,
    )

    try:
        asyncio.run(harness.run(phases=args.phases, log=print))
    except KeyboardInterrupt:
        print("\n[trackb-lme] Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[trackb-lme] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
