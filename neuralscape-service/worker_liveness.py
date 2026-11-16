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
import logging
import time

HEARTBEAT_FILE = os.environ.get("NS_WORKER_HEARTBEAT_FILE", "/tmp/ns-worker.heartbeat")  # noqa: S108
HEARTBEAT_INTERVAL_S = 10.0
# 24 missed beats. Compose adds (retries-1)*interval, so >= 5 min of missed beats before unhealthy.
HEARTBEAT_MAX_AGE_S = 240

logger = logging.getLogger(__name__)

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


STALL_ENV = "NS_TEST_STALL_HEARTBEAT_FILE"
ALLOW_PUBLIC_ENV = "NS_TEST_HOOKS_ALLOW_WITH_PUBLIC_URL"
_stall_refused_logged = False


def _stall_requested() -> bool:
    """Test-only fault: while the file named by ``NS_TEST_STALL_HEARTBEAT_FILE`` exists, stop beating.

    Reversible (delete the file). Refused, with a warning, whenever the deployment
    has a public URL configured (``NEURALSCAPE_PUBLIC_URL``), unless a second,
    explicit opt-in ``NS_TEST_HOOKS_ALLOW_WITH_PUBLIC_URL=1`` is also set (the
    flags-on acceptance profile sets a public URL). Neither is set by default.
    """
    global _stall_refused_logged
    target = os.environ.get(STALL_ENV)
    if not target:
        return False
    from config import settings

    if settings.neuralscape_public_url and os.environ.get(ALLOW_PUBLIC_ENV) != "1":
        if not _stall_refused_logged:
            _stall_refused_logged = True
            logger.warning(
                "%s ignored: NEURALSCAPE_PUBLIC_URL is set (set %s=1 to allow it in a test profile)",
                STALL_ENV, ALLOW_PUBLIC_ENV,
            )
        return False
    return os.path.exists(target)


async def heartbeat_loop(path: str | None = None, interval: float = HEARTBEAT_INTERVAL_S) -> None:
    while True:
        try:
            if not _stall_requested():
                beat(path)
        except Exception:  # noqa: BLE001 - never let the beat task die silently
            logger.warning("worker heartbeat write failed; the liveness check will go stale", exc_info=True)
        await asyncio.sleep(interval)


def start_heartbeat(ctx: dict, path: str | None = None, interval: float = HEARTBEAT_INTERVAL_S) -> None:
    """Beat once now, then keep beating from the running loop. Idempotent."""
    task = ctx.get(_CTX_KEY)
    if task is not None and not task.done():
        return
    with contextlib.suppress(Exception):
        beat(path)
    task = asyncio.get_running_loop().create_task(heartbeat_loop(path, interval), name="ns-worker-heartbeat")
    task.add_done_callback(_log_if_died)
    ctx[_CTX_KEY] = task


def _log_if_died(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("worker heartbeat task died: %r", task.exception())


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
