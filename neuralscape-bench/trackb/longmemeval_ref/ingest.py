"""Ingest protocol for Track B LongMemEval.

Faithful to the LME reference: ingest each question's haystack sessions
(dated) into NS, one user per question. This matches how LME loads
conversations into the memory system under test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.ingest import ingest_suite
from neuralscape_bench.accuracy.manifest import IngestManifest
from neuralscape_bench.accuracy.schema import SuiteData


async def ingest_lme_haystacks(
    client: NeuralscapeClient,
    data: SuiteData,
    *,
    manifest_path: Path,
    concurrency: int = 2,
    poll_timeout_s: float = 300.0,
    log=print,
) -> dict:
    """Ingest LongMemEval haystacks into NS.

    Each question gets its own NS user (bench-longmemeval_s-<question_id>).
    Session dates are preserved (folded into the first message).

    Args:
        client: NS REST client
        data: SuiteData with conversations (haystacks) and qa_items
        manifest_path: Idempotency manifest (tracks completed sessions)
        concurrency: Parallel conversations
        poll_timeout_s: Poll timeout per write task
        log: Logging function

    Returns:
        Summary dict: {sessions_stored, sessions_skipped, sessions_failed, wall_s, ...}
    """
    manifest = IngestManifest(manifest_path)

    summary = await ingest_suite(
        client,
        data,
        manifest=manifest,
        concurrency=concurrency,
        poll_timeout_s=poll_timeout_s,
        poll_interval_s=1.0,
        namespace=None,  # Track B uses default bench-<suite>-<conv_id> naming
        log=log,
    )

    return summary
