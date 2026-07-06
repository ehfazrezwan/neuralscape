"""Answer protocol for Track B LongMemEval.

Faithful to the LME reference:
1. Retrieve top-k memories from NS for each question
2. Generate answer using NS /v1/ask (which internally does retrieval + answer generation)
3. Compute retrieval recall@k as a diagnostic (NOT the headline metric)
"""

from __future__ import annotations

from pathlib import Path

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.answer import answer_suite
from neuralscape_bench.accuracy.schema import SuiteData


async def answer_lme_questions(
    client: NeuralscapeClient,
    data: SuiteData,
    *,
    out_path: Path,
    k: int = 10,
    reasoning_level: str = "high",
    concurrency: int = 2,
    log=print,
) -> dict:
    """Answer LongMemEval questions using NS /v1/ask.

    For each question:
    1. /v1/search (top-k) → compute retrieval_hit (R@k diagnostic)
    2. /v1/ask (reasoning_level) → answer text + citations + abstained flag

    Args:
        client: NS REST client
        data: SuiteData with qa_items
        out_path: Output JSONL path (resumable)
        k: Top-k for retrieval
        reasoning_level: NS /v1/ask reasoning level (default: high)
        concurrency: Parallel questions
        log: Logging function

    Returns:
        Summary dict: {answered, skipped, ask_errors, wall_s, k, reasoning_level}
    """
    summary = await answer_suite(
        client,
        data,
        out_path=out_path,
        k=k,
        reasoning_level=reasoning_level,
        concurrency=concurrency,
        namespace=None,  # Track B uses default bench-<suite>-<conv_id> naming
        log=log,
    )

    return summary
