"""Event-loop liveness heartbeat for the ARQ workers.

A small asyncio task, started from the ARQ ``on_startup`` hook, touches a file
every ``HEARTBEAT_INTERVAL_S`` seconds. The container healthcheck (compose
``healthcheck`` / Helm exec ``livenessProbe``) only reads that file's mtime and
fails once it is older than ``HEARTBEAT_MAX_AGE_S``.

Why the event loop and not a per-job signal: a long graph/ingest job runs in
``asyncio.to_thread`` and leaves the loop free, so the beat keeps going. If the
loop itself is blocked (sync work on the loop, a deadlock) the beat stops, which
is exactly the "wedged" case autoheal should restart. It is liveness only: it
never touches Redis, Qdrant, Neo4j or an LLM.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

HEARTBEAT_FILE = os.environ.get("NS_WORKER_HEARTBEAT_FILE", "/tmp/ns-worker.heartbeat")  # noqa: S108
HEARTBEAT_INTERVAL_S = 10.0
# Twelve missed beats. Compose adds interval*retries on top before a restart.
HEARTBEAT_MAX_AGE_S = 120

_CTX_KEY = "_liveness_task"


def beat(path: str | None = None) -> None:
    """Create/refresh the heartbeat file (mtime = now)."""
    target = path or HEARTBEAT_FILE
    try:
        os.utime(target, None)
    except FileNotFoundError:
        with open(target, "a"):
            pass
        os.utime(target, None)


async def heartbeat_loop(path: str | None = None, interval: float = HEARTBEAT_INTERVAL_S) -> None:
    while True:
        try:
            beat(path)
        except OSError:
            pass  # an unwritable /tmp must not kill the worker; the check will go stale
        await asyncio.sleep(interval)


def start_heartbeat(ctx: dict, path: str | None = None, interval: float = HEARTBEAT_INTERVAL_S) -> None:
    """Beat once now, then keep beating from the running loop. Idempotent."""
    task = ctx.get(_CTX_KEY)
    if task is not None and not task.done():
        return
    with contextlib.suppress(OSError):
        beat(path)
    ctx[_CTX_KEY] = asyncio.get_running_loop().create_task(
        heartbeat_loop(path, interval), name="ns-worker-heartbeat"
    )


async def stop_heartbeat(ctx: dict) -> None:
    task = ctx.pop(_CTX_KEY, None)
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def is_fresh(path: str | None = None, max_age: float = HEARTBEAT_MAX_AGE_S) -> bool:
    """Same predicate as the compose/Helm healthcheck command."""
    try:
        return time.time() - os.stat(path or HEARTBEAT_FILE).st_mtime < max_age
    except OSError:
        return False
