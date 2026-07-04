"""Ingestion runner: write suite conversations into an isolated NS stack.

Per conversation → one NS user (``bench-<suite>-<conv_id>``); per session →
one ``POST /v1/memories`` (conversation-extraction path) with
``run_id=<session_id>``, session date folded into the first turn so the
extractor can capture temporal context. Writes are async (202) — we poll
task completion, never re-store. Idempotent + resumable via
:class:`~.manifest.IngestManifest` (session granularity). A concurrency
knob bounds parallel conversations; 429/5xx get exponential backoff.
"""

from __future__ import annotations

import asyncio
import random
import time

import httpx

from neuralscape_bench.client import NeuralscapeClient, TaskTimeout
from neuralscape_bench.accuracy.manifest import IngestManifest
from neuralscape_bench.accuracy.schema import Conversation, Session, SuiteData, bench_user_id

# StoreMemoryRequest caps messages at 500; stay comfortably below.
MAX_MESSAGES_PER_CALL = 400


def session_messages(session: Session) -> list[dict]:
    """Session → NS conversation messages, date folded into the first turn."""
    messages = [{"role": t.role, "content": t.content} for t in session.turns]
    if messages and session.date:
        messages[0] = {
            "role": messages[0]["role"],
            "content": f"[This conversation session took place {session.date}.]\n"
                       + messages[0]["content"],
        }
    return messages


def _batches(messages: list[dict]) -> list[list[dict]]:
    return [messages[i:i + MAX_MESSAGES_PER_CALL]
            for i in range(0, len(messages), MAX_MESSAGES_PER_CALL)]


def est_tokens(messages: list[dict]) -> int:
    """Chars/4 heuristic — recorded per session for cost extrapolation."""
    return sum(len(m["content"]) for m in messages) // 4


async def _with_backoff(coro_factory, *, max_retries: int = 6, base_delay: float = 2.0):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                raise
        except httpx.TransportError:
            if attempt == max_retries - 1:
                raise
        await asyncio.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


async def ingest_conversation(client: NeuralscapeClient, conv: Conversation, *,
                              suite: str, manifest: IngestManifest,
                              poll_timeout_s: float, poll_interval_s: float,
                              log=print) -> dict:
    """Ingest one conversation's sessions in order (temporal fidelity)."""
    user_id = bench_user_id(suite, conv.conv_id)
    done = manifest.sessions_done(conv.conv_id)
    stored = skipped = failed = 0
    for session in conv.sessions:
        if session.session_id in done:
            skipped += 1
            continue
        messages = session_messages(session)
        if not messages:
            manifest.mark_session(conv.conv_id, session.session_id, est_tokens=0)
            continue
        t0 = time.perf_counter()
        task_ids: list[str] = []
        ok = True
        for batch in _batches(messages):
            resp = await _with_backoff(
                lambda b=batch: client.extract_write(
                    b, user_id=user_id, run_id=session.session_id)
            )
            task_id = resp.get("task_id")
            if task_id:
                task_ids.append(task_id)
        for task_id in task_ids:
            try:
                status = await client.wait_for_task(
                    task_id, timeout_s=poll_timeout_s, interval_s=poll_interval_s)
                if status.get("status") not in ("completed", "not_found"):
                    ok = False
            except TaskTimeout:
                ok = False
        elapsed = time.perf_counter() - t0
        if ok:
            stored += 1
            manifest.mark_session(
                conv.conv_id, session.session_id,
                task_id=task_ids[0] if task_ids else None,
                elapsed_s=elapsed, est_tokens=est_tokens(messages),
            )
        else:
            failed += 1
            log(f"[ingest] {suite}/{conv.conv_id} session {session.session_id}: "
                f"task did not complete (will retry on next run)")
    return {"conv_id": conv.conv_id, "stored": stored, "skipped": skipped, "failed": failed}


async def ingest_suite(client: NeuralscapeClient, data: SuiteData, *,
                       manifest: IngestManifest, concurrency: int = 2,
                       poll_timeout_s: float = 300.0, poll_interval_s: float = 1.0,
                       log=print) -> dict:
    """Ingest every conversation (bounded parallelism across conversations)."""
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[dict] = []

    async def one(conv: Conversation):
        async with sem:
            res = await ingest_conversation(
                client, conv, suite=data.suite, manifest=manifest,
                poll_timeout_s=poll_timeout_s, poll_interval_s=poll_interval_s, log=log)
            results.append(res)
            done = len(results)
            if done % 10 == 0 or done == len(data.conversations):
                log(f"[ingest] {data.suite}: {done}/{len(data.conversations)} conversations")

    t0 = time.perf_counter()
    await asyncio.gather(*(one(c) for c in data.conversations))
    wall = time.perf_counter() - t0
    summary = {
        "conversations": len(results),
        "sessions_stored": sum(r["stored"] for r in results),
        "sessions_skipped": sum(r["skipped"] for r in results),
        "sessions_failed": sum(r["failed"] for r in results),
        "wall_s": round(wall, 1),
        **manifest.totals(),
    }
    manifest.record_stats(last_run=summary)
    return summary
