"""The dream sweep orchestrator: gates → LIGHT → DEEP → REM → diary.

One sweep walks every pool (shared, per-project shared, per-user private —
pool isolation per spec §4.2), and for each pool that passes the gate
economy runs the full cycle. The ``DreamRun`` record persists to Redis so
the API process and the graph worker report the same state (unlike the
wiki_synthesizer's process-local snapshot).

Cheap-to-expensive ordering per pool:
  time gate (Redis read) → settling guard (in-memory) → lock →
  LIGHT scroll+stage → volume gate →
  idempotent skip (unchanged staged-id set) → DEEP LLM decide/apply →
  REM reflect/store → diary → gate completion stamp.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import bridges, card, consolidate, gate, librarian, reflect
from .config import DreamingSettings, dreaming_settings

logger = logging.getLogger(__name__)

_RUN_KEY = "dreaming:last_run"
_STAGED_IDS_KEY = "dreaming:staged_ids:{pool}"


@dataclass(slots=True)
class PoolReport:
    pool: str
    status: str                       # dreamt | gated | settling | locked | skipped_unchanged | error
    reason: str = ""
    staged: int = 0
    applied: int = 0
    reported: int = 0
    insights: int = 0
    pages_written: int = 0
    pages_skipped: int = 0
    card_status: str = ""             # updated | stable | unchanged | skipped | ""
    errors: list[str] = field(default_factory=list)
    diary_path: str | None = None


@dataclass(slots=True)
class DreamRun:
    run_id: str
    started_at: str
    finished_at: str | None = None
    dry_run: bool = False
    pools: list[PoolReport] = field(default_factory=list)
    bridges: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "bridges": self.bridges,
            "pools": [asdict(p) for p in self.pools],
            "totals": {
                "pools_dreamt": sum(1 for p in self.pools if p.status == "dreamt"),
                "pools_gated": sum(
                    1 for p in self.pools if p.status in ("gated", "settling", "locked")
                ),
                "actions_applied": sum(p.applied for p in self.pools),
                "actions_reported": sum(p.reported for p in self.pools),
                "insights_stored": sum(p.insights for p in self.pools),
                "errors": sum(len(p.errors) for p in self.pools),
            },
        }


def _get_redis():
    import redis as redis_lib

    from config import settings as core_settings

    return redis_lib.Redis.from_url(core_settings.redis_url, socket_timeout=5)


def get_last_run(redis=None) -> dict | None:
    """Cross-process DreamRun snapshot (API + worker read the same key)."""
    try:
        r = redis or _get_redis()
        raw = r.get(_RUN_KEY)
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("DreamRun read failed", exc_info=True)
        return None


def _save_run(redis, run: DreamRun) -> None:
    try:
        redis.set(_RUN_KEY, json.dumps(run.to_dict()))
    except Exception:
        logger.warning("DreamRun save failed", exc_info=True)


async def _make_llm_call(settings: DreamingSettings):
    """Build the retrying/timeout-wrapped LLM callable for this sweep."""
    from extensions.conversation_compiler.compile import _async_call_gemini

    async def call(prompt: str) -> str:
        last_exc: Exception | None = None
        attempts = max(1, settings.llm_max_retries + 1)
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    _async_call_gemini(prompt), timeout=settings.llm_timeout_seconds
                )
            except Exception as exc:  # timeout or transport — retry with backoff
                last_exc = exc
                logger.warning(
                    "dream LLM call failed (attempt %d/%d): %s",
                    attempt + 1, attempts, exc.__class__.__name__,
                )
                if attempt + 1 < attempts:
                    await asyncio.sleep(2 ** attempt)
        logger.error("dream LLM call exhausted %d attempts: %s", attempts, last_exc)
        return ""

    return call


async def dream_all(
    *,
    service,
    settings: DreamingSettings = dreaming_settings,
    dry_run: bool | None = None,
    only_pool: str | None = None,
    force: bool = False,
) -> DreamRun:
    """Run one full dreaming sweep across all pools.

    Args:
        service: shared MemoryService.
        dry_run: report planned actions without writing (None → config default).
        only_pool: restrict to one pool key (admin/testing).
        force: bypass the time/volume gates (not the lock).
    """
    dry_run = settings.dry_run_default if dry_run is None else dry_run
    run = DreamRun(
        run_id=f"drm_{uuid.uuid4().hex[:10]}",
        started_at=datetime.now(timezone.utc).isoformat(),
        dry_run=dry_run,
    )
    if not settings.enabled and not force:
        logger.info("dreaming disabled — skipping sweep")
        run.finished_at = datetime.now(timezone.utc).isoformat()
        return run

    redis = _get_redis()
    llm_call = await _make_llm_call(settings)

    pools = await asyncio.to_thread(consolidate.enumerate_pools, service)
    if only_pool is not None:
        pools = {k: v for k, v in pools.items() if k == only_pool}

    for pool_key, batch in sorted(pools.items()):
        report = await _dream_pool(
            service=service, settings=settings, redis=redis, llm_call=llm_call,
            batch=batch, dry_run=dry_run, force=force,
        )
        run.pools.append(report)

    # Bridges (B3a) run once per sweep, AFTER every pool's librarian pass —
    # tunnels are cross-pool by nature, so they can't live inside the loop.
    if (
        settings.vault_pages_enabled
        and settings.bridges_enabled
        and any(p.status == "dreamt" for p in run.pools)
    ):
        try:
            graph_rows = await bridges.fetch_graph_rows(
                service, limit=settings.bridge_graph_limit
            )
            run.bridges = await asyncio.to_thread(
                bridges.update_bridges,
                settings.vault_path,
                graph_rows=graph_rows,
                dry_run=dry_run,
            )
        except Exception:
            logger.exception("bridges pass failed (non-fatal)")
            run.bridges = {"error": "bridges"}

    run.finished_at = datetime.now(timezone.utc).isoformat()
    _save_run(redis, run)
    totals = run.to_dict()["totals"]
    logger.info(
        "dream sweep %s complete: dreamt=%d gated=%d applied=%d reported=%d insights=%d errors=%d",
        run.run_id, totals["pools_dreamt"], totals["pools_gated"],
        totals["actions_applied"], totals["actions_reported"],
        totals["insights_stored"], totals["errors"],
    )
    return run


async def _dream_pool(
    *, service, settings: DreamingSettings, redis, llm_call, batch, dry_run: bool, force: bool
) -> PoolReport:
    pool = batch.pool
    report = PoolReport(pool=pool, status="error")

    # 1. cheap time gate
    if not force:
        decision = gate.check_time_gate(redis, pool, min_hours=settings.min_hours)
        if not decision.proceed:
            report.status, report.reason = "gated", decision.reason
            return report

    # 1b. settling guard (A3-lite): a pool written to within the last
    #     DREAMING_SETTLING_MINUTES is mid-conversation — never consolidate
    #     a thought while it's still forming. Defer to the next pass.
    if not force:
        decision = gate.check_settling_gate(
            batch.memories, settling_minutes=settings.settling_minutes
        )
        if not decision.proceed:
            report.status, report.reason = "settling", decision.reason
            return report

    # 2. lock (shared with the dedup cron's semantic-skip check)
    lock_token = gate.acquire_lock(redis, pool)
    if not lock_token:
        report.status, report.reason = "locked", "another sweep holds the pool lock"
        return report

    try:
        state = gate.get_gate_state(redis, pool)
        last_dreamt_at = float(state.get("last_dreamt_at") or 0.0)

        # 3. LIGHT: stage + score
        batch = await asyncio.to_thread(
            consolidate.stage_pool,
            batch, redis,
            last_dreamt_at=last_dreamt_at,
            max_memories=settings.max_memories_per_pool,
            strength_half_life_days=settings.strength_half_life_days,
            prune_strength_threshold=settings.prune_strength_threshold,
            dynamics_enabled=settings.dynamics_enabled,
        )
        report.staged = len(batch.memories)

        # 4. expensive volume gate (count came free with the LIGHT scroll)
        if not force:
            decision = gate.check_volume_gate(
                batch.new_count, min_new_memories=settings.min_new_memories
            )
            if not decision.proceed:
                report.status, report.reason = "gated", decision.reason
                return report

        # 5. idempotent skip: unchanged staged-id set ⇒ the LLM pass would
        #    reproduce itself (carried over from the wiki_synthesizer).
        staged_ids = sorted(m["memory_id"] for m in batch.memories)
        ids_key = _STAGED_IDS_KEY.format(pool=pool)
        try:
            prior = redis.get(ids_key)
            prior_ids = json.loads(prior) if prior else None
        except Exception:
            prior_ids = None
        if not force and prior_ids == staged_ids:
            report.status, report.reason = "skipped_unchanged", "staged id set unchanged"
            return report

        # 6. DEEP: decide + hybrid-posture apply
        actions = await consolidate.decide(batch, llm_call)
        to_apply, to_report = consolidate.split_by_posture(
            actions, auto_apply_confidence=settings.auto_apply_confidence
        )
        applied = await consolidate.apply_actions(service, batch, to_apply, dry_run=dry_run)
        report.applied = len(applied.applied)
        report.reported = len(to_report)
        report.errors.extend(applied.errors)
        # Fold the applied actions into the staged dicts so the REM and
        # librarian passes below see the post-consolidation view — without
        # this, a row tombstoned seconds ago would still land on topic
        # pages and in Home's Essential Story until the next sweep.
        consolidate.reconcile_batch(batch, applied.applied)

        # 7. REM: reflect + store insights
        insights: list[dict] = []
        if settings.reflection_enabled:
            insights = await reflect.reflect(
                batch, llm_call, max_insights=settings.max_reflections_per_pool
            )
            stored = await asyncio.to_thread(
                reflect.store_insights, service, batch, insights, dry_run=dry_run
            )
            report.insights = len(stored) if not dry_run else len(insights)

        # 8. librarian: humane topic pages (the wiki_synthesizer successor)
        if settings.vault_pages_enabled:
            try:
                from config import settings as core_settings

                lib = await librarian.update_vault(
                    batch, llm_call,
                    vault=settings.vault_path,
                    operator_user_id=core_settings.default_user_id,
                    dry_run=dry_run,
                    redis=redis,
                    faded_threshold=settings.prune_strength_threshold,
                )
                report.pages_written = lib["pages_written"]
                report.pages_skipped = lib["pages_skipped"]
            except Exception:
                logger.exception("librarian pass failed for pool %s (non-fatal)", pool)
                report.errors.append("librarian")

        # 8b. identity card (B4): pinned Redis artifact + Card.md render.
        #     Never stored as a searchable memory.
        if settings.identity_card_enabled:
            try:
                from config import settings as core_settings

                card_out = await card.update_card(
                    batch, llm_call,
                    redis=redis,
                    vault=settings.vault_path,
                    operator_user_id=core_settings.default_user_id,
                    dry_run=dry_run,
                    # Vault output off ⇒ cards stay Redis-only; no Card.md
                    # files may appear under a disabled vault path.
                    render_files=settings.vault_pages_enabled,
                )
                report.card_status = card_out.get("status", "")
            except Exception:
                logger.exception("card pass failed for pool %s (non-fatal)", pool)
                report.errors.append("card")

        # 9. diary (review surface; never a promotion source)
        entry = reflect.render_diary_entry(
            pool=pool,
            run_id="dry-run" if dry_run else "sweep",
            applied=applied.applied,
            reported=to_report,
            insights=insights,
        )
        if not dry_run:
            report.diary_path = await asyncio.to_thread(
                reflect.write_diary,
                settings.dreams_dir, pool, entry,
                source_memory_ids=staged_ids,
            )
            try:
                redis.set(ids_key, json.dumps(staged_ids))
            except Exception:
                pass
            gate.record_completion(redis, pool)

        report.status = "dreamt"
        return report
    except Exception as exc:
        logger.exception("dream sweep failed for pool %s", pool)
        report.errors.append(exc.__class__.__name__)
        return report
    finally:
        gate.release_lock(redis, pool, lock_token)
