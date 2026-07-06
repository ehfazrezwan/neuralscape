"""Ingest runner following mem0's LoCoMo protocol.

Per conversation -> one NS user (trackb-mem0-locomo-<conv_id>)
Per session -> POST /v1/memories with session messages + date context
Async writes (202) -> poll to completion. Idempotent via manifest.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Callable

import httpx

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import IngestManifest
from neuralscape_bench.accuracy.schema import Conversation, Session, SuiteData

MAX_MESSAGES_PER_CALL = 400


def trackb_user_id(conv_id: str) -> str:
    """Generate user ID for Track B mem0 LoCoMo: trackb-mem0-locomo-<conv_id>."""
    return f"trackb-mem0-locomo-{conv_id}"


def session_messages(session: Session) -> list[dict]:
    """Session -> NS conversation messages, date folded into first turn."""
    messages = [{"role": t.role, "content": t.content} for t in session.turns]
    if messages and session.date:
        messages[0] = {
            "role": messages[0]["role"],
            "content": f"[This conversation session took place {session.date}.]\n"
                       + messages[0]["content"],
        }
    return messages


def _batches(messages: list[dict]) -> list[list[dict]]:
    """Split messages into batches under MAX_MESSAGES_PER_CALL."""
    return [messages[i:i + MAX_MESSAGES_PER_CALL]
            for i in range(0, len(messages), MAX_MESSAGES_PER_CALL)]


async def _with_backoff(coro_factory, *, max_retries: int = 6, base_delay: float = 2.0):
    """Exponential backoff for transient HTTP errors."""
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


async def ingest_conversation(
    client: NeuralscapeClient,
    conv: Conversation,
    *,
    manifest: IngestManifest,
    poll_timeout_s: float,
    poll_interval_s: float,
    log: Callable = print,
) -> dict:
    """Ingest one conversation's sessions in order."""
    user_id = trackb_user_id(conv.conv_id)
    done = manifest.sessions_done(conv.conv_id)
    stored = skipped = failed = 0

    for session in conv.sessions:
        if session.session_id in done:
            skipped += 1
            continue

        messages = session_messages(session)
        if not messages:
            # Empty session - mark as done
            manifest.mark_session(conv.conv_id, session.session_id, est_tokens=0)
            continue

        t0 = time.perf_counter()
        task_ids: list[str] = []
        ok = True

        # Submit batches
        for batch in _batches(messages):
            resp = await _with_backoff(
                lambda b=batch: client.extract_write(
                    b, user_id=user_id, run_id=session.session_id
                )
            )
            task_id = resp.get("task_id")
            if task_id:
                task_ids.append(task_id)

        # Poll all tasks
        for task_id in task_ids:
            try:
                status = await client.wait_for_task(
                    task_id, timeout_s=poll_timeout_s, interval_s=poll_interval_s
                )
                if status.get("status") not in ("completed", "not_found"):
                    ok = False
                    log(f"[!] session {conv.conv_id}/{session.session_id} task {task_id} -> {status.get('status')}")
            except Exception as e:
                ok = False
                log(f"[!] session {conv.conv_id}/{session.session_id} poll failed: {e}")

        elapsed = time.perf_counter() - t0
        est_tokens = sum(len(m["content"]) for m in messages) // 4

        if ok:
            manifest.mark_session(conv.conv_id, session.session_id, est_tokens=est_tokens)
            stored += 1
            log(f"[✓] {conv.conv_id}/{session.session_id} ({len(messages)} msgs, ~{est_tokens} tok, {elapsed:.1f}s)")
        else:
            failed += 1

    return {"stored": stored, "skipped": skipped, "failed": failed}


async def ingest_suite(
    client: NeuralscapeClient,
    data: SuiteData,
    *,
    manifest: IngestManifest,
    concurrency: int = 2,
    poll_timeout_s: float = 120.0,
    poll_interval_s: float = 1.0,
    log: Callable = print,
) -> dict:
    """Ingest all conversations with bounded concurrency."""
    log(f"Ingesting {len(data.conversations)} conversations (concurrency={concurrency})")

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(conv: Conversation) -> dict:
        async with sem:
            return await ingest_conversation(
                client, conv,
                manifest=manifest,
                poll_timeout_s=poll_timeout_s,
                poll_interval_s=poll_interval_s,
                log=log,
            )

    results = await asyncio.gather(*[_bounded(c) for c in data.conversations])

    summary = {
        "sessions_stored": sum(r["stored"] for r in results),
        "sessions_skipped": sum(r["skipped"] for r in results),
        "sessions_failed": sum(r["failed"] for r in results),
    }
    log(f"Ingest complete: {summary}")
    return summary
