"""In-process task backend for the solo engine (unit 4 of 28-solo-engine.md).

Replaces Redis/ARQ + the three worker processes with an in-process runner
that executes the SAME coroutines from ``worker.py`` — zero behavioral drift
from the team deployment: the idempotency probes, deferred graph enrichment,
extension events, and adapter resolution all run exactly as they do on the
workers.

Design:

- **Two lanes mirror the queue split.** ``fast`` (vector writes, conversation
  tasks, retag) and ``slow`` (graph enrichment, document/file ingest) with
  small concurrency caps, so a bulk ingest or a minutes-long Graphiti episode
  can't starve interactive writes — the same isolation philosophy as the
  three ARQ queues, collapsed into one process.
- **The pool shim keeps TaskManager untouched.** ``_InlinePool`` satisfies the
  one contract every ``enqueue_*`` method uses (``pool.enqueue_job(...)`` →
  Job-like with ``.job_id``, or ``None`` on duplicate) and raises
  ConnectionError from any other attribute — so best-effort bookkeeping like
  ``_track_task`` degrades exactly as designed. It also serves as
  ``ctx["redis"]`` for the worker coroutines, which routes their own deferred
  enqueues (``process_memory_raw`` → ``process_graph_enrichment``) onto the
  slow lane, exactly like the dedicated graph queue does in team mode.
- **Same 202-and-poll surface.** Task statuses live in a bounded in-memory
  table with completion events; ``InlineTaskManager`` overrides only the
  read side (``get_status`` / ``wait_for_result`` / ``get_queue_status``).

A daemon restart forgets in-flight tasks (they either finished — the data is
in the embedded stores — or they didn't run; pollers see ``not_found`` and
the deterministic job ids make re-submission safe). The journal-on-disk
upgrade is deliberately deferred until real usage shows it matters.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from config import settings
from task_manager import TaskManager

logger = logging.getLogger(__name__)

# Task-name → lane. Names are the ARQ function names; the coroutines are
# resolved from worker.py lazily so importing this module stays cheap.
_LANES: dict[str, str] = {
    "process_memory_store": "fast",
    "process_memory_raw": "fast",
    "process_memory_raw_batch": "fast",
    "process_memory_retag": "fast",
    "process_conversation_flush": "fast",
    "process_conversation_compile": "fast",
    "process_graph_enrichment": "slow",
    "process_session_summary": "slow",
    "process_ingest_document": "slow",
    "process_ingest_file": "slow",
    "process_ingest_okf_bundle": "slow",
    "process_connector_sync": "slow",
}

_FAST_CONCURRENCY = 4
_SLOW_CONCURRENCY = 2
_MAX_FINISHED_TASKS = 2000


class _InlineJob:
    """Job-like return value for ``pool.enqueue_job`` (only .job_id is read)."""

    __slots__ = ("job_id",)

    def __init__(self, job_id: str):
        self.job_id = job_id


class _InlinePool:
    """The single seam TaskManager's enqueue methods need, minus Redis."""

    def __init__(self, runner: "InlineTaskRunner"):
        # Bypass __getattr__ recursion by writing through object.__setattr__
        object.__setattr__(self, "_runner", runner)

    def __bool__(self) -> bool:
        return True  # it IS a working queue — health reports it as "inline"

    async def enqueue_job(
        self,
        function: str,
        *args: Any,
        _job_id: str | None = None,
        _queue_name: str | None = None,
        **kwargs: Any,
    ):
        job_id = _job_id or uuid.uuid4().hex
        accepted = self._runner.submit(function, args, kwargs, job_id)
        return _InlineJob(job_id) if accepted else None

    def __getattr__(self, name: str):
        # zadd/expire/ping/zrevrangebyscore/…: every non-queue use degrades
        # the same way a down Redis would, which callers already handle.
        raise ConnectionError(f"inline task backend has no redis op {name!r}")


class InlineTaskRunner:
    """Two-lane asyncio executor over the worker.py task coroutines."""

    def __init__(self, service: Any, extension_registry: Any = None, vault: Any = None):
        import worker as _worker  # heavy import deferred to bind time

        self._worker = _worker
        self.ctx: dict[str, Any] = {
            "service": service,
            "extension_registry": extension_registry,
            "vault": vault,
            "redis": _InlinePool(self),
        }
        self._sems = {
            "fast": asyncio.Semaphore(_FAST_CONCURRENCY),
            "slow": asyncio.Semaphore(_SLOW_CONCURRENCY),
        }
        # job_id → {status, result, error, function, user_id, enqueued_at, done(Event)}
        self._tasks: "OrderedDict[str, dict]" = OrderedDict()
        self._bg: set[asyncio.Task] = set()

    # -- submission ----------------------------------------------------

    def submit(self, function: str, args: tuple, kwargs: dict, job_id: str) -> bool:
        """Schedule a task; False when an identical live job id exists
        (mirrors ARQ's _job_id dedup, which the enqueue methods rely on)."""
        existing = self._tasks.get(job_id)
        if existing:
            if existing["status"] in ("queued", "processing"):
                return False
            # Finished entry being re-run: drop it so the re-insert lands at
            # the NEW end of the OrderedDict — otherwise _evict() (which
            # trims from the oldest end) could evict the fresh run early.
            del self._tasks[job_id]
        fn = getattr(self._worker, function, None)
        lane = _LANES.get(function)
        if fn is None or lane is None:
            raise ConnectionError(f"inline task backend has no task {function!r}")
        entry = {
            "status": "queued",
            "result": None,
            "error": None,
            "function": function,
            "user_id": None,
            "enqueued_at": time.time(),
            "done": asyncio.Event(),
        }
        self._tasks[job_id] = entry
        self._evict()
        task = asyncio.get_running_loop().create_task(
            self._run(fn, lane, args, kwargs, job_id),
            name=f"inline:{function}:{job_id[:12]}",
        )
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return True

    async def _run(self, fn, lane: str, args: tuple, kwargs: dict, job_id: str) -> None:
        entry = self._tasks.get(job_id)
        if entry is None:  # evicted before it ran — record loss loudly
            logger.error("inline task %s evicted before execution", job_id)
            return
        async with self._sems[lane]:
            entry["status"] = "processing"
            try:
                entry["result"] = await fn(self.ctx, *args, **kwargs)
                entry["status"] = "completed"
            except Exception as e:  # noqa: BLE001 — mirrors ARQ job failure
                entry["status"] = "failed"
                entry["error"] = str(e)
                logger.warning(
                    "inline task %s (%s) failed: %s", job_id, entry["function"], e
                )
            finally:
                entry["done"].set()

    # -- reads -----------------------------------------------------------

    def status(self, task_id: str) -> dict | None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return None
        return {
            "task_id": task_id,
            "status": entry["status"],
            "result": entry["result"] if entry["status"] == "completed" else None,
            "error": entry["error"],
        }

    async def wait(self, task_id: str, timeout: float) -> dict | None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return None
        try:
            await asyncio.wait_for(entry["done"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "task_id": task_id,
                "status": entry["status"],
                "result": None,
                "error": f"timed out after {timeout}s (task still {entry['status']})",
            }
        return self.status(task_id)

    def tag_user(self, task_id: str, user_id: str | None) -> None:
        entry = self._tasks.get(task_id)
        if entry is not None and user_id:
            entry["user_id"] = user_id

    def user_counts(self, user_id: str, window_seconds: int, cap: int) -> tuple[dict, list[str]]:
        counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "expired": 0}
        ids: list[str] = []
        cutoff = time.time() - window_seconds
        for task_id, entry in reversed(self._tasks.items()):
            if len(ids) >= cap:
                break
            if entry["user_id"] != user_id or entry["enqueued_at"] < cutoff:
                continue
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            ids.append(task_id)
        return counts, ids

    def pending_depths(self) -> dict[str, int]:
        depths = {"fast": 0, "slow": 0}
        for entry in self._tasks.values():
            if entry["status"] == "queued":
                depths[_LANES.get(entry["function"], "fast")] += 1
        return depths

    def _evict(self) -> None:
        # Drop oldest FINISHED entries beyond the cap; never evict live ones.
        excess = len(self._tasks) - _MAX_FINISHED_TASKS
        if excess <= 0:
            return
        for task_id in list(self._tasks):
            if excess <= 0:
                break
            if self._tasks[task_id]["status"] in ("completed", "failed"):
                del self._tasks[task_id]
                excess -= 1


class InlineTaskManager(TaskManager):
    """TaskManager backed by InlineTaskRunner instead of Redis/ARQ.

    Every ``enqueue_*`` method is inherited untouched — the pool shim is the
    only write-side difference; the read side (status/wait/queue-status) is
    overridden to consult the in-memory task table.
    """

    def bind(
        self, service: Any, extension_registry: Any = None, vault: Any = None
    ) -> None:
        """Attach the shared MemoryService (+ registry, + connector vault)
        and arm the runner.

        Must be called from the daemon lifespan AFTER the shared service
        (and, when connectors are enabled, the vault) exist — the runner
        reuses them rather than constructing seconds, because the embedded
        stores are single-process and would fight over their locks.
        """
        runner = InlineTaskRunner(service, extension_registry, vault=vault)
        self._runner = runner
        self.pool = runner.ctx["redis"]
        logger.info(
            "Inline task backend armed (fast=%d, slow=%d)",
            _FAST_CONCURRENCY,
            _SLOW_CONCURRENCY,
        )

    async def connect(self) -> None:
        # Binding happens in the lifespan once the shared service exists; an
        # unbound manager keeps the disabled-pool behavior (sync fallbacks).
        if getattr(self, "_runner", None) is None:
            logger.info(
                "task_backend=inline: awaiting bind() — writes fall back to "
                "sync until the daemon lifespan arms the runner"
            )

    async def _track_task(self, user_id: str | None, task_id: str) -> None:
        runner = getattr(self, "_runner", None)
        if runner is not None:
            runner.tag_user(task_id, user_id)

    async def get_status(self, task_id: str) -> dict:
        runner = getattr(self, "_runner", None)
        found = runner.status(task_id) if runner else None
        return found or {
            "task_id": task_id,
            "status": "not_found",
            "result": None,
            "error": None,
        }

    async def wait_for_result(self, task_id: str, timeout: float = 300.0) -> dict:
        runner = getattr(self, "_runner", None)
        found = await runner.wait(task_id, timeout) if runner else None
        return found or {
            "task_id": task_id,
            "status": "not_found",
            "result": None,
            "error": None,
        }

    async def get_queue_status(
        self,
        user_id: str,
        window_seconds: int | None = None,
        cap: int = 200,
    ) -> dict:
        window = int(window_seconds or settings.queue_status_window_s)
        runner = getattr(self, "_runner", None)
        if runner is None:
            counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "expired": 0}
            ids: list[str] = []
            depths: dict[str, int] = {}
        else:
            counts, ids = runner.user_counts(user_id, window, cap)
            depths = runner.pending_depths()
        return {
            "user_id": user_id,
            "window_seconds": window,
            "tracked": len(ids),
            "counts": counts,
            "queues": depths,
            "caught_up": counts["queued"] == 0 and counts["processing"] == 0,
        }

    async def close(self) -> None:
        """No Redis to close; let in-flight inline tasks finish briefly."""
        runner = getattr(self, "_runner", None)
        if runner is None or not runner._bg:
            return
        done, pending = await asyncio.wait(set(runner._bg), timeout=5.0)
        for t in pending:
            t.cancel()
        if pending:
            logger.warning(
                "inline task backend shut down with %d tasks cancelled", len(pending)
            )
