"""Grading protocol for Track B LongMemEval.

Faithful to the LME reference: LLM judge with abstention handling,
per-question_type breakdown. Uses the standard judge (gemini-2.5-flash, temp 0).
"""

from __future__ import annotations

from pathlib import Path

from neuralscape_bench.accuracy.judge import GeminiJudge, judge_suite
from neuralscape_bench.accuracy.schema import SuiteData


async def grade_lme_answers(
    data: SuiteData,
    *,
    answers_path: Path,
    judged_path: Path,
    judge_model: str = "gemini-2.5-flash",
    api_key: str,
    concurrency: int = 4,
    log=print,
) -> dict:
    """Grade LongMemEval answers using LLM judge.

    Args:
        data: SuiteData with qa_items (needed for gold answers)
        answers_path: Input JSONL (from answer phase)
        judged_path: Output JSONL (resumable)
        judge_model: Gemini model for grading (default: gemini-2.5-flash)
        api_key: GOOGLE_API_KEY
        concurrency: Parallel judge calls
        log: Logging function

    Returns:
        Summary dict: {judged, skipped, unparseable, out}
    """
    judge = GeminiJudge(api_key, model=judge_model)
    try:
        summary = await judge_suite(
            judge,
            data,
            answers_path,
            judged_path,
            concurrency=concurrency,
            log=log,
        )
        return summary
    finally:
        await judge.aclose()
