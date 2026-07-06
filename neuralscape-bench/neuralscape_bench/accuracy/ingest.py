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
import re
import time
from datetime import datetime

import httpx

from neuralscape_bench.client import NeuralscapeClient, TaskTimeout
from neuralscape_bench.accuracy.manifest import IngestManifest
from neuralscape_bench.accuracy.schema import Conversation, Session, SuiteData, bench_user_id

# StoreMemoryRequest caps messages at 500; stay comfortably below.
MAX_MESSAGES_PER_CALL = 400


def parse_session_date(date_str: str | None) -> str | None:
    """Parse common dataset date formats into ISO 8601 date strings.

    Handles:
    - LoCoMo format: "1:00 pm on 5 May, 2023"
    - LongMemEval format: "2023/05/05 ..." (extracts YYYY/MM/DD prefix)
    - ISO dates already in the right format

    Returns:
        ISO 8601 date string (YYYY-MM-DD) or None on parse failure.
    """
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip()

    # Try LoCoMo format: "1:00 pm on 5 May, 2023"
    # Pattern: optional time, "on", day, month, year
    locomo_match = re.search(
        r"on\s+(\d{1,2})\s+([A-Za-z]+)[,\s]+(\d{4})",
        date_str,
        re.IGNORECASE
    )
    if locomo_match:
        day = locomo_match.group(1).zfill(2)
        month_name = locomo_match.group(2)
        year = locomo_match.group(3)
        try:
            # Parse month name to number
            dt = datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Try abbreviated month
                dt = datetime.strptime(f"{day} {month_name} {year}", "%d %b %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # Try LongMemEval format: "2023/05/05 ..." (YYYY/MM/DD prefix)
    longmem_match = re.match(r"(\d{4})/(\d{2})/(\d{2})", date_str)
    if longmem_match:
        year, month, day = longmem_match.groups()
        return f"{year}-{month}-{day}"

    # Try ISO date format (YYYY-MM-DD)
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        return date_str[:10]  # Return just the date part

    # Parse failure
    return None


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
                              namespace: str | None = None,
                              log=print) -> dict:
    """Ingest one conversation's sessions in order (temporal fidelity)."""
    user_id = bench_user_id(suite, conv.conv_id, namespace)
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
        # T1.3: parse session date to ISO 8601 for event-time grounding
        occurred_at = parse_session_date(session.date) if session.date else None
        for batch in _batches(messages):
            resp = await _with_backoff(
                lambda b=batch, oa=occurred_at: client.extract_write(
                    b, user_id=user_id, run_id=session.session_id, occurred_at=oa)
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
                       namespace: str | None = None,
                       log=print) -> dict:
    """Ingest every conversation (bounded parallelism across conversations)."""
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[dict] = []

    async def one(conv: Conversation):
        async with sem:
            res = await ingest_conversation(
                client, conv, suite=data.suite, manifest=manifest,
                poll_timeout_s=poll_timeout_s, poll_interval_s=poll_interval_s,
                namespace=namespace, log=log)
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
