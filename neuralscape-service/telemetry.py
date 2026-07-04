"""Fire-and-forget lane for hot-path telemetry side-writes (audit 27 #11).

The savings-meter ledger append, the SSE event publish, and any other
observability write that used to run inline on a read/write response path
goes through :func:`submit` instead: one bounded single-worker thread,
every task wrapped so it can neither raise into nor delay the request
that spawned it. Same pattern as the dreaming recall-trace executor
(extensions/dreaming/traces.py) — telemetry is best-effort by contract,
so when the queue backs up (dead Redis, stalled tokenizer) new tasks are
DROPPED with a debug log rather than growing an unbounded queue.

The worker is a regular (non-daemon) ThreadPoolExecutor thread, so
interpreter shutdown joins it — safe because every task is tiny and the
pending queue is bounded, so shutdown waits on at most a small in-flight
backlog, never an unbounded one.

Ordering across tasks is irrelevant for every current caller (append-only
ledger entries, pub/sub events); the single worker exists to bound
concurrency, not to guarantee sequence.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Ceiling on queued-but-unstarted tasks. Telemetry writes are tiny (a Redis
# pipeline, a pub/sub publish, one body tokenization); a backlog this deep
# means the sink is down — dropping is the correct degradation.
MAX_PENDING = 1000

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ns-telemetry")


def _pending_count(executor) -> int | None:
    """Best-effort queue depth via the executor's private ``_work_queue``.

    Returns ``None`` when the internal isn't inspectable (a future Python
    renaming it, or a test double) — callers must then SUBMIT rather than
    drop: the bound is an overload guard, and an uninspectable queue must
    not silently disable all telemetry.
    """
    try:
        return executor._work_queue.qsize()
    except Exception:
        return None


def submit(fn, *args, **kwargs) -> bool:
    """Schedule ``fn(*args, **kwargs)`` on the telemetry thread.

    Never raises and never blocks: returns ``True`` when queued, ``False``
    when dropped (queue full) or when the executor is already shut down at
    interpreter exit. Exceptions inside ``fn`` are swallowed and logged at
    debug. The drop policy only applies when the queue depth is readable.
    """
    try:
        pending = _pending_count(_executor)
        if pending is not None and pending >= MAX_PENDING:
            logger.debug(
                "telemetry queue full (>= %d pending) — dropping %r",
                MAX_PENDING,
                getattr(fn, "__name__", fn),
            )
            return False
        _executor.submit(_run_safely, fn, args, kwargs)
        return True
    except Exception:
        return False


def _run_safely(fn, args, kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.debug(
            "telemetry task %r failed (non-fatal)",
            getattr(fn, "__name__", fn),
            exc_info=True,
        )


def flush(timeout: float = 5.0) -> None:
    """Wait until every task queued before this call has finished.

    Test/shutdown helper: submits a barrier no-op and waits for it (the
    single worker drains FIFO, so the barrier completing implies all prior
    tasks completed). Swallows timeouts — flush is best-effort too.
    """
    try:
        _executor.submit(lambda: None).result(timeout=timeout)
    except Exception:
        logger.debug("telemetry flush timed out/failed", exc_info=True)
