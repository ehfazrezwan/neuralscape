"""Core harness orchestration for Track B LongMemEval.

Phases: ingest → answer → judge → report
All phases are resumable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from neuralscape_bench.client import NeuralscapeClient

from .answer import answer_lme_questions
from .grade import grade_lme_answers
from .ingest import ingest_lme_haystacks
from .loader import load_longmemeval_s
from .report import aggregate_results, write_markdown_summary, write_report


class LMEHarness:
    """Track B LongMemEval reference harness."""

    def __init__(
        self,
        target: str,
        token: str | None = None,
        judge_key: str | None = None,
        judge_model: str = "gemini-3.1-flash-lite",
        k: int = 10,
        reasoning_level: str = "high",
        sample: int | None = None,
        seed: int = 42,
        concurrency_ingest: int = 2,
        concurrency_answer: int = 2,
        concurrency_judge: int = 4,
    ):
        self.target = target
        self.token = token
        self.judge_key = judge_key
        self.judge_model = judge_model
        self.k = k
        self.reasoning_level = reasoning_level
        self.sample = sample
        self.seed = seed
        self.concurrency_ingest = concurrency_ingest
        self.concurrency_answer = concurrency_answer
        self.concurrency_judge = concurrency_judge

        # Paths (relative to trackb/longmemeval_ref/)
        base = Path(__file__).parent.parent.parent
        self.manifest_path = base / "state" / "ingest-trackb-lme.json"
        self.answers_path = base / "results" / "raw" / "answers-trackb-lme.jsonl"
        self.judged_path = base / "results" / "raw" / "judged-trackb-lme.jsonl"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.results_json = base / "results" / f"trackb-lme-{timestamp}.json"
        self.results_md = base / "results" / f"trackb-lme-{timestamp}.md"

        # Ensure directories exist
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.answers_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_json.parent.mkdir(parents=True, exist_ok=True)

    async def run(self, phases: list[str] | None = None, log=print) -> dict:
        """Run the harness.

        Args:
            phases: List of phases to run (default: all)
                    Options: ingest, answer, judge, report
            log: Logging function

        Returns:
            Final results dict
        """
        if phases is None:
            phases = ["ingest", "answer", "judge", "report"]

        data = None
        client = None

        try:
            # Load dataset
            log(f"[trackb-lme] Loading LongMemEval_S (sample={self.sample}, seed={self.seed})")
            data = load_longmemeval_s(sample=self.sample, seed=self.seed)
            log(f"[trackb-lme] Loaded {len(data.qa_items)} questions, "
                f"{len(data.conversations)} conversations (haystacks)")

            # Initialize client for ingest/answer phases
            if "ingest" in phases or "answer" in phases:
                client = NeuralscapeClient(self.target, token=self.token)

            # Phase: ingest
            if "ingest" in phases:
                log(f"[trackb-lme] Phase: ingest (target={self.target})")
                ingest_summary = await ingest_lme_haystacks(
                    client,
                    data,
                    manifest_path=self.manifest_path,
                    concurrency=self.concurrency_ingest,
                    log=log,
                )
                log(f"[trackb-lme] Ingest complete: {ingest_summary}")

            # Phase: answer
            if "answer" in phases:
                log(f"[trackb-lme] Phase: answer (k={self.k}, reasoning_level={self.reasoning_level})")
                answer_summary = await answer_lme_questions(
                    client,
                    data,
                    out_path=self.answers_path,
                    k=self.k,
                    reasoning_level=self.reasoning_level,
                    concurrency=self.concurrency_answer,
                    log=log,
                )
                log(f"[trackb-lme] Answer complete: {answer_summary}")

            # Phase: judge
            if "judge" in phases:
                if not self.judge_key:
                    raise ValueError("--judge-key required for judge phase")
                log(f"[trackb-lme] Phase: judge (model={self.judge_model})")
                judge_summary = await grade_lme_answers(
                    data,
                    answers_path=self.answers_path,
                    judged_path=self.judged_path,
                    judge_model=self.judge_model,
                    api_key=self.judge_key,
                    concurrency=self.concurrency_judge,
                    log=log,
                )
                log(f"[trackb-lme] Judge complete: {judge_summary}")

            # Phase: report
            if "report" in phases:
                log(f"[trackb-lme] Phase: report")
                results = aggregate_results(
                    self.judged_path,
                    backbone="neuralscape",
                    judge_model=self.judge_model,
                    embedder="unknown",  # TODO: fetch from NS health endpoint if available
                    k=self.k,
                )
                write_report(results, self.results_json)
                write_markdown_summary(results, self.results_md)
                log(f"[trackb-lme] Results written:")
                log(f"  JSON: {self.results_json}")
                log(f"  MD:   {self.results_md}")
                log(f"[trackb-lme] Overall accuracy: {results['overall_accuracy']:.1%} "
                    f"({results['total_questions']} questions)")
                return results

            return {"status": "phases_complete", "phases": phases}

        finally:
            if client is not None:
                await client.aclose()
