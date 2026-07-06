"""Answer runner following mem0's LoCoMo protocol.

For each QA:
1. Retrieve top-k memories via /v1/search
2. Assemble context from retrieved memories
3. Generate answer using NS /v1/ask (mirrors mem0's answer prompt)
4. Record retrieval metrics + answer

JSONL records written to output file (resumable on qa_id).
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import append_jsonl, read_jsonl_records
from neuralscape_bench.accuracy.metrics import attribute_memory, recall_at_k
from neuralscape_bench.accuracy.schema import QAItem, SuiteData

from .ingest import trackb_user_id


async def _with_backoff(coro_factory, *, max_retries: int = 5, base_delay: float = 2.0):
    """Exponential backoff for transient HTTP errors."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                raise
        except httpx.TransportError:
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


async def answer_one(
    client: NeuralscapeClient,
    data: SuiteData,
    qa: QAItem,
    *,
    k: int,
    reasoning_level: str,
) -> dict:
    """Answer one QA item: retrieve + ask, record metrics."""
    user_id = trackb_user_id(qa.conv_id)
    conv = data.conversation(qa.conv_id)

    record: dict = {
        "qa_id": qa.qa_id,
        "conv_id": qa.conv_id,
        "qtype": qa.qtype,
        "is_abstention": qa.is_abstention,
        "at": datetime.now(timezone.utc).isoformat(),
    }

    # Retrieval probe (R@k)
    t0 = time.perf_counter()
    try:
        res = await _with_backoff(
            lambda: client.search(qa.question, user_id=user_id, limit=k)
        )
        hits = res.get("results") or []
        attributed: list[str | None] = []
        if conv is not None:
            for h in hits:
                sid, _score = attribute_memory(h.get("memory") or "", conv)
                attributed.append(sid)

        record["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        record["retrieved"] = len(hits)
        record["attributed_sessions"] = [s for s in attributed if s]
        record["retrieval_hit"] = recall_at_k(attributed, qa.evidence_session_ids, k)
    except Exception as e:
        record["retrieval_error"] = str(e)[:200]
        record["retrieval_hit"] = None

    # Answer via /v1/ask (mirrors mem0's answer prompt faithfully)
    t0 = time.perf_counter()
    try:
        res = await _with_backoff(
            lambda: client.ask(
                qa.question,
                user_id=user_id,
                reasoning_level=reasoning_level,
            )
        )
        record["answer"] = res.get("answer", "")
        record["abstained"] = bool(res.get("abstained"))
        record["citations"] = len(res.get("citations") or [])
        record["searches"] = len(res.get("searches") or [])
        record["memories_considered"] = res.get("memories_considered", 0)
        record["ask_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        record["answer"] = ""
        record["ask_error"] = str(e)[:200]

    return record


async def answer_suite(
    client: NeuralscapeClient,
    data: SuiteData,
    *,
    out_path: Path,
    k: int = 10,
    reasoning_level: str = "high",
    concurrency: int = 2,
    log: Callable = print,
) -> dict:
    """Answer all QA items, appending to out_path JSONL (resumable on qa_id)."""
    done = {r.get("qa_id") for r in read_jsonl_records(out_path)}
    todo = [qa for qa in data.qa_items if qa.qa_id not in done]

    log(f"Answering {len(todo)} QA items (skipping {len(done)}, k={k}, concurrency={concurrency})")

    sem = asyncio.Semaphore(concurrency)
    answered = 0

    async def _bounded(qa: QAItem) -> dict:
        nonlocal answered
        async with sem:
            rec = await answer_one(client, data, qa, k=k, reasoning_level=reasoning_level)
            append_jsonl(out_path, rec)
            answered += 1
            if answered % 10 == 0:
                log(f"[answer] {answered}/{len(todo)} complete")
            return rec

    await asyncio.gather(*[_bounded(qa) for qa in todo])

    summary = {
        "answered": answered,
        "skipped": len(done),
        "total": len(data.qa_items),
    }
    log(f"Answering complete: {summary}")
    return summary
