"""CLI entrypoint for the accuracy battery.

Phases are explicit and individually resumable:

    # offline, free:
    uv run python -m neuralscape_bench.accuracy.run --suite all --phase fetch
    uv run python -m neuralscape_bench.accuracy.run --suite all --phase estimate

    # paid phases, against an ISOLATED stack (never the livestack):
    uv run python -m neuralscape_bench.accuracy.run --stack up
    uv run python -m neuralscape_bench.accuracy.run --suite locomo \\
        --phase ingest --phase answer --phase judge --phase report \\
        --target http://localhost:8398
    uv run python -m neuralscape_bench.accuracy.run --stack down

``--suite all`` expands to every registered suite. ``--sample N --seed S``
runs a deterministic stratified QA sample. ``--dry-run`` prints the plan
for paid phases without network calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from neuralscape_bench.accuracy.costs import CostModel, estimate_suite_cost
from neuralscape_bench.accuracy.download import DATASETS_DIR
from neuralscape_bench.accuracy.manifest import IngestManifest, read_jsonl_records
from neuralscape_bench.accuracy.report import (
    RESULTS_DIR, build_suite_result, save_battery_markdown, save_suite_result,
)
from neuralscape_bench.accuracy.suites import all_suite_names, get_suite

RAW_DIR = RESULTS_DIR / "raw"
PHASES = ("fetch", "estimate", "ingest", "answer", "judge", "report")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _suite_options(args, suite_name: str) -> dict:
    opts = dict(get_suite(suite_name).default_options)
    if suite_name == "beam" and args.beam_tier:
        opts["tier"] = args.beam_tier
    if suite_name == "membench" and args.membench_categories:
        opts["categories"] = (
            "all" if args.membench_categories == "all"
            else args.membench_categories.split(",")
        )
    return opts


def _dataset_dir(suite_name: str) -> Path:
    # The two LongMemEval variants share one dataset dir.
    if suite_name.startswith("longmemeval"):
        return DATASETS_DIR / "longmemeval"
    return DATASETS_DIR / suite_name


def _load(args, suite_name: str):
    suite = get_suite(suite_name)
    return suite.load(_dataset_dir(suite_name), sample=args.sample, seed=args.seed,
                      options=_suite_options(args, suite_name))


def cmd_fetch(args, suite_name: str) -> None:
    suite = get_suite(suite_name)
    dest = _dataset_dir(suite_name)
    opts = _suite_options(args, suite_name)
    _log(f"[fetch] {suite_name} ← {suite.source}")
    # Option-aware fetches for the suites that take them.
    if suite_name == "beam":
        from neuralscape_bench.accuracy.suites import beam as beam_mod
        info = beam_mod.fetch(dest, tier=opts.get("tier", beam_mod.DEFAULT_TIER))
    elif suite_name == "membench":
        from neuralscape_bench.accuracy.suites import membench as mb_mod
        cats = opts.get("categories") or list(mb_mod.DEFAULT_CATEGORIES)
        cats = mb_mod.ALL_CATEGORIES if cats == "all" else tuple(cats)
        info = mb_mod.fetch(dest, categories=cats)
    else:
        info = suite.fetch(dest)
    _log(f"[fetch] {suite_name}: OK {json.dumps(info)[:300]}")


def cmd_estimate(args, suite_name: str, model: CostModel) -> dict:
    data = _load(args, suite_name)
    est = estimate_suite_cost(data, model=model)
    d = est.to_dict(model)
    _log(f"[estimate] {suite_name}: {d['tokens']['total_input']:,} in / "
         f"{d['tokens']['total_output']:,} out tokens ≈ ${d['estimated_usd']}")
    return d


async def _run_paid(args, suite_name: str) -> None:
    from neuralscape_bench.client import NeuralscapeClient
    from neuralscape_bench.accuracy.answer import answer_suite
    from neuralscape_bench.accuracy.ingest import ingest_suite
    from neuralscape_bench.accuracy.judge import GeminiJudge, judge_suite

    data = _load(args, suite_name)
    target_label = args.target.replace("://", "-").replace("/", "").replace(":", "-")
    answers_path = RAW_DIR / f"answers-{suite_name}.jsonl"
    judged_path = RAW_DIR / f"judged-{suite_name}.jsonl"

    if args.dry_run:
        st = data.stats()
        _log(f"[dry-run] {suite_name}: would ingest {st['sessions']} sessions across "
             f"{st['conversations']} conversations, answer {st['qa_items']} questions "
             f"against {args.target}")
        return

    client = NeuralscapeClient(args.target, token=args.token)
    try:
        if "ingest" in args.phase:
            manifest = IngestManifest.for_run(suite_name, target_label)
            summary = await ingest_suite(
                client, data, manifest=manifest, concurrency=args.concurrency,
                poll_timeout_s=args.poll_timeout, log=_log)
            _log(f"[ingest] {suite_name}: {json.dumps(summary)}")
        if "answer" in args.phase:
            summary = await answer_suite(
                client, data, out_path=answers_path, k=args.k,
                reasoning_level=args.reasoning_level,
                concurrency=args.concurrency, log=_log)
            _log(f"[answer] {suite_name}: {json.dumps(summary)}")
        if "judge" in args.phase:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise SystemExit("judge phase needs GOOGLE_API_KEY in the environment")
            judge = GeminiJudge(api_key, model=args.judge_model)
            try:
                summary = await judge_suite(
                    judge, data, answers_path, judged_path,
                    concurrency=args.concurrency, log=_log)
            finally:
                await judge.aclose()
            _log(f"[judge] {suite_name}: {json.dumps(summary)}")
    finally:
        await client.aclose()

    if "report" in args.phase:
        judged = read_jsonl_records(judged_path)
        manifest = IngestManifest.for_run(suite_name, target_label)
        result = build_suite_result(
            suite_name, judged,
            config={
                "target": args.target,
                "k": args.k,
                "reasoning_level": args.reasoning_level,
                "judge_model": args.judge_model,
                "sample": args.sample,
                "seed": args.seed,
                "suite_options": _suite_options(args, suite_name),
            },
            suite_stats=data.stats(),
            run_stats=manifest.totals(),
        )
        path = save_suite_result(result)
        _log(f"[report] {suite_name} → {path}")


def _battery_report(args, suites: list[str]) -> None:
    """Cross-suite markdown from the latest committed per-suite results."""
    results: dict[str, dict | None] = {}
    for s in all_suite_names():
        latest = sorted(RESULTS_DIR.glob(f"accuracy-{s}-*.json"))
        results[s] = json.loads(latest[-1].read_text()) if latest else None
    note = (f"Config: k={args.k}, reasoning_level={args.reasoning_level}, "
            f"judge={args.judge_model}, sample={args.sample or 'full'}, seed={args.seed}.")
    path = save_battery_markdown(results, config_note=note)
    _log(f"[report] battery summary → {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Neuralscape memory-accuracy battery (E5).")
    ap.add_argument("--suite", action="append", default=None,
                    help="Suite id or 'all' (repeatable). "
                         f"Known: {', '.join(all_suite_names())}")
    ap.add_argument("--phase", action="append", default=None, choices=PHASES,
                    help="Phase(s) to run, in order (repeatable). Default: fetch")
    ap.add_argument("--target", default="http://localhost:8398",
                    help="Isolated NS stack base URL (NEVER the livestack :8199)")
    ap.add_argument("--token", default=os.environ.get("BENCH_TOKEN"),
                    help="Bearer token (defaults to $BENCH_TOKEN; use the stack's "
                         "NEURALSCAPE_API_KEY shared key)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Stratified QA sample size per suite (default: full)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10, help="Retrieval R@k depth")
    ap.add_argument("--reasoning-level", default="high",
                    choices=("minimal", "low", "medium", "high"))
    ap.add_argument("--judge-model", default=os.environ.get("BENCH_JUDGE_MODEL",
                                                            "gemini-2.5-flash"))
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--poll-timeout", type=float, default=300.0)
    ap.add_argument("--beam-tier", default=None, choices=("100k", "500k", "1m", "10m"))
    ap.add_argument("--membench-categories", default=None,
                    help="Comma list or 'all' (default: highlevel,lowlevel_rec,RecMultiSession)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stack", default=None, choices=("up", "down"),
                    help="Manage the isolated compose stack and exit")
    ap.add_argument("--stack-port", type=int, default=8398)
    args = ap.parse_args(argv)

    if args.stack:
        from neuralscape_bench.accuracy import stack as stack_mod
        if args.stack == "up":
            stack_mod.stack_up(api_port=args.stack_port)
        else:
            stack_mod.stack_down()
        return 0

    suites = args.suite or ["all"]
    if "all" in suites:
        suites = all_suite_names()
    args.phase = args.phase or ["fetch"]

    estimates = []
    cost_model = CostModel()
    for suite_name in suites:
        if "fetch" in args.phase:
            cmd_fetch(args, suite_name)
        if "estimate" in args.phase:
            estimates.append(cmd_estimate(args, suite_name, cost_model))
        paid = [p for p in args.phase if p in ("ingest", "answer", "judge", "report")]
        if paid:
            asyncio.run(_run_paid(args, suite_name))

    if estimates:
        out = RESULTS_DIR / "cost-estimates.json"
        RESULTS_DIR.mkdir(exist_ok=True)
        out.write_text(json.dumps(
            {"model": cost_model.assumptions(), "suites": estimates}, indent=2))
        total = sum(e["estimated_usd"] for e in estimates)
        _log(f"[estimate] battery total ≈ ${total:.2f} → {out}")

    if "report" in args.phase:
        _battery_report(args, suites)
    return 0


if __name__ == "__main__":
    sys.exit(main())
