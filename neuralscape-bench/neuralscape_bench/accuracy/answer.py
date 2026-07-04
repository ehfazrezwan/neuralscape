"""Answering runner: ``POST /v1/ask`` per QA + retrieval-only R@k probe.

Per question:
1. ``/v1/search`` (top-k) → attribute each hit to a haystack session
   (lexical containment, see metrics.py) → ``retrieval_hit`` where the
   dataset provides evidence annotations;
2. ``/v1/ask`` (configurable reasoning level, default high) → answer text,
   citations, abstained flag.

Raw per-question records (which embed answers referencing conversation
content) go to a gitignored JSONL under ``results/raw/`` — resumable: qa_ids
already present are skipped.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import append_jsonl, load_done_qa_ids
from neuralscape_bench.accuracy.metrics import attribute_memory, recall_at_k
from neuralscape_bench.accuracy.schema import QAItem, SuiteData, bench_user_id


async def _with_backoff(coro_factory, *, max_retries: int = 5, base_delay: float = 2.0):
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


async def answer_one(client: NeuralscapeClient, data: SuiteData, qa: QAItem, *,
                     k: int, reasoning_level: str) -> dict:
    user_id = bench_user_id(data.suite, qa.conv_id)
    conv = data.conversation(qa.conv_id)
    record: dict = {
        "qa_id": qa.qa_id,
        "conv_id": qa.conv_id,
        "qtype": qa.qtype,
        "is_abstention": qa.is_abstention,
        "at": datetime.now(timezone.utc).isoformat(),
    }

    # ── retrieval-only probe (R@k) ──
    t0 = time.perf_counter()
    try:
        res = await _with_backoff(
            lambda: client.search(qa.question, user_id=user_id, limit=k))
        hits = res.get("results") or []
        attributed: list[str | None] = []
        if conv is not None:
            for h in hits:
                sid, _score = attribute_memory(h.get("memory") or "", conv)
                attributed.append(sid)
        record["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        record["retrieved"] = len(hits)
        record["attributed_sessions"] = [s for s in attributed if s]
        # recall_at_k returns None when the item carries no gold evidence —
        # which is how LongMemEval's *_abs instances are skipped (their
        # answer_session_ids are empty). Suites that DO annotate evidence on
        # abstention questions (e.g. BEAM gold_ids) still get scored.
        record["retrieval_hit"] = recall_at_k(attributed, qa.evidence_session_ids, k)
    except Exception as e:  # noqa: BLE001 — a failed probe must not kill the run
        record["retrieval_error"] = str(e)[:200]
        record["retrieval_hit"] = None

    # ── /v1/ask ──
    t0 = time.perf_counter()
    try:
        res = await _with_backoff(
            lambda: client.ask(qa.question, user_id=user_id,
                               reasoning_level=reasoning_level))
        record["answer"] = res.get("answer", "")
        record["abstained"] = bool(res.get("abstained"))
        record["citations"] = len(res.get("citations") or [])
        record["searches"] = len(res.get("searches") or [])
        record["memories_considered"] = res.get("memories_considered", 0)
        record["ask_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        record["answer"] = ""
        record["ask_error"] = str(e)[:200]
    return record


async def answer_suite(client: NeuralscapeClient, data: SuiteData, *,
                       out_path: Path, k: int = 10, reasoning_level: str = "high",
                       concurrency: int = 2, log=print) -> dict:
    """Answer every QA item, appending records to ``out_path`` (resumable)."""
    done = load_done_qa_ids(out_path)
    todo = [qa for qa in data.qa_items if qa.qa_id not in done]
    log(f"[answer] {data.suite}: {len(todo)} to answer ({len(done)} already done)")

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    counters = {"done": 0, "errors": 0}

    async def one(qa: QAItem):
        async with sem:
            rec = await answer_one(client, data, qa, k=k, reasoning_level=reasoning_level)
            async with lock:
                append_jsonl(out_path, rec)
                counters["done"] += 1
                if rec.get("ask_error"):
                    counters["errors"] += 1
                if counters["done"] % 20 == 0 or counters["done"] == len(todo):
                    log(f"[answer] {data.suite}: {counters['done']}/{len(todo)}")

    t0 = time.perf_counter()
    await asyncio.gather(*(one(qa) for qa in todo))
    return {
        "answered": counters["done"],
        "skipped": len(done),
        "ask_errors": counters["errors"],
        "wall_s": round(time.perf_counter() - t0, 1),
        "k": k,
        "reasoning_level": reasoning_level,
        "out": str(out_path),
    }
