"""In-process maintenance scheduler for the solo engine (unit 5, §5.3).

Fires the same cron coroutines the ARQ workers run — at the same hours and
minutes as the team schedules — by invoking them directly on the inline
runner's SLOW lane semaphore with the runner's ctx. Maintenance jobs have no
poller, so they deliberately bypass the task-status table.

No new dependency: a single asyncio loop that wakes every ~30s, fires
whatever crons match the current UTC hour/minute window, and dedupes per
(cron, date, hour) so a cron never double-fires within its hour.

Only the solo-relevant subset runs here (§5.3): TTL expiry, dedup,
auto-compile (the plugin depends on it), and the dreaming sweep — which
stays inert unless dreaming is enabled, exactly as on the workers.
Playbook synthesis and connector sync remain team/off-by-default features.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from config import settings

logger = logging.getLogger(__name__)

_POLL_SECONDS = 30.0

# Fire windows mirror worker.py's ARQ cron registrations: a cron is due when
# now.hour is in hours() and now.minute is within [minute, minute+2) — the
# two-minute grace covers a poll landing just after the boundary.
_MINUTE_GRACE = 2


def _cron_specs() -> list[tuple[str, Callable[[], Iterable[int]], int]]:
    import worker as _worker

    return [
        # (worker coroutine name, hours provider, minute)
        ("expire_old_memories_cron", lambda: {3}, 15),
        ("dedup_all_memories", lambda: settings.dedup_cron_hours, 0),
        ("auto_compile_check", lambda: {18, 19, 20, 21, 22, 23}, 30),
        ("dream_sweep_cron", lambda: set(_worker._dreaming_cron_hours()), 35),
    ]


def _due(now: datetime, specs, fired: set[tuple[str, str, int]]) -> list[str]:
    """Names of crons due at ``now`` that haven't fired this (date, hour)."""
    due: list[str] = []
    for name, hours, minute in specs:
        key = (name, now.date().isoformat(), now.hour)
        if key in fired:
            continue
        try:
            in_hours = now.hour in set(hours())
        except Exception:  # noqa: BLE001 — a bad hours provider skips, not crashes
            logger.warning("scheduler: hours provider for %s failed", name, exc_info=True)
            continue
        if in_hours and minute <= now.minute < minute + _MINUTE_GRACE:
            fired.add(key)
            due.append(name)
    return due


async def _fire(runner: Any, name: str) -> None:
    """Run one cron coroutine under the slow-lane semaphore."""
    fn = getattr(runner._worker, name, None)
    if fn is None:
        logger.error("scheduler: worker has no cron %s", name)
        return
    async with runner._sems["slow"]:
        try:
            result = await fn(runner.ctx)
            logger.info("scheduler: %s completed: %s", name, result)
        except Exception:  # noqa: BLE001 — maintenance must never kill the loop
            logger.warning("scheduler: %s failed (non-fatal)", name, exc_info=True)


async def _loop(runner: Any) -> None:
    specs = _cron_specs()
    fired: set[tuple[str, str, int]] = set()
    logger.info(
        "In-process scheduler started (%s)",
        ", ".join(f"{n}@{m:02d}" for n, _, m in specs),
    )
    while True:
        now = datetime.now(timezone.utc)
        # Keep the dedup set bounded: entries from previous days are inert.
        if len(fired) > 200:
            today = now.date().isoformat()
            fired = {k for k in fired if k[1] == today}
        for name in _due(now, specs, fired):
            asyncio.get_running_loop().create_task(
                _fire(runner, name), name=f"cron:{name}"
            )
        await asyncio.sleep(_POLL_SECONDS)


def start_scheduler(runner: Any) -> asyncio.Task:
    """Start the maintenance loop on the running event loop.

    Caller (the daemon lifespan) owns the returned task and cancels it on
    shutdown.
    """
    return asyncio.get_running_loop().create_task(_loop(runner), name="ns-scheduler")
