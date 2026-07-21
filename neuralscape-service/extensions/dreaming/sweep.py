"""The dream sweep orchestrator: gates → LIGHT → DEEP → REM → diary.

One sweep walks every pool (shared, per-project shared, per-user private —
pool isolation per spec §4.2), and for each pool that passes the gate
economy runs the full cycle. The ``DreamRun`` record persists to Redis so
the API process and the graph worker report the same state (unlike the
wiki_synthesizer's process-local snapshot).

Cheap-to-expensive ordering per pool (audit 27 #29: every gate below runs
on the LIGHT enumeration rows — identity + timestamps, no contents; full
rows are hydrated only after all three gates pass):
  time gate (Redis read) → settling guard (in-memory) →
  volume pre-gate (in-memory) → lock → hydrate + stage →
  volume gate (hydrated recount) →
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

from . import bridges, card, consolidate, gate, librarian, liveness, reflect, surprisal
from .config import DreamingSettings, dreaming_settings

logger = logging.getLogger(__name__)

_RUN_KEY = "dreaming:last_run"
_STAGED_IDS_KEY = "dreaming:staged_ids:{pool}"


@dataclass(slots=True)
class PoolReport:
    pool: str
    status: str                       # dreamt | gated | settling | locked | skipped_unchanged | sweep_failed | error
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
                # LLM never examined the pool — gate unstamped, cron retries
                # next cycle (audit 27 #27).
                "pools_failed": sum(
                    1 for p in self.pools if p.status in ("sweep_failed", "error")
                ),
                "actions_applied": sum(p.applied for p in self.pools),
                "actions_reported": sum(p.reported for p in self.pools),
                "insights_stored": sum(p.insights for p in self.pools),
                "errors": sum(len(p.errors) for p in self.pools),
            },
        }


# Cached module-level client (audit 27 #35): callers hit this per API call
# (MCP get_card, status endpoints), and redis-py clients hold a thread-safe
# connection pool — one client per process, not one per call.
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib

        from config import settings as core_settings

        _redis_client = redis_lib.Redis.from_url(core_settings.redis_url, socket_timeout=5)
    return _redis_client


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


def _gather_code_liveness_actions(service, batch) -> tuple[list[dict], list[str]]:
    """Gather code_liveness_stale memories via Qdrant and build temporal_reframe actions.

    This is the I4 liveness consumer. It does NOT read the staged pool rows
    (``consolidate.hydrate_pool`` flattens a fixed field set onto each row keyed
    on ``content`` with NO nested ``metadata`` dict, so the ``code_liveness_stale``
    flag never survives staging). Instead it scrolls Qdrant directly for the flag
    — the raw payload is where the flag and the ``data`` text actually live.

    Flags are NOT cleared here: clearing is deferred until AFTER the action is
    successfully applied (and only when not dry_run) via
    :func:`_clear_code_liveness_flags`, so a failed/aborted apply leaves the flag
    set for the next sweep (idempotent, no lost work).

    Args:
        service: The MemoryService instance.
        batch: The staged pool batch (used only to scope reframes to THIS pool).

    Returns:
        (actions, flagged_ids): temporal_reframe action dicts ready for
        ``consolidate.apply_actions`` and the raw memory_ids that carried the flag.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        m = service._get_memory()
        client = m.vector_store.client
        collection_name = m.vector_store.collection_name
    except Exception:
        logger.warning(
            "code liveness: cannot access vector store for pool %s (non-fatal)",
            batch.pool, exc_info=True,
        )
        return [], []

    # Scope reframes to memories staged in THIS pool. An empty pool id set means
    # nothing to intersect against — bail rather than reframe cross-pool.
    pool_ids = {mem["memory_id"] for mem in batch.memories}
    if not pool_ids:
        return [], []

    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.code_liveness_stale",
                match=MatchValue(value=True),
            )
        ]
    )

    hits = []
    try:
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            hits.extend(records)
            if next_offset is None or not records:
                break
            offset = next_offset
            if len(hits) >= 1000:
                logger.warning(
                    "code liveness: hit 1000-memory scroll cap for pool %s (truncated)",
                    batch.pool,
                )
                break
    except Exception:
        logger.warning(
            "code liveness scroll failed for pool %s (non-fatal)", batch.pool,
            exc_info=True,
        )
        return [], []

    actions: list[dict] = []
    flagged_ids: list[str] = []
    for hit in hits:
        memory_id = str(getattr(hit, "id", "") or "")
        if not memory_id or memory_id not in pool_ids:
            continue  # not in this pool — another pool's sweep handles it

        payload = getattr(hit, "payload", None) or {}
        # Unwrap mem0's possible double-wrapped metadata.
        meta = payload.get("metadata", {}) or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]

        reason = meta.get("code_liveness_reason", "code changed")
        anchor = meta.get("code_liveness_anchor", "unknown")
        content = payload.get("data", "") or ""

        # Deterministic, reversible reframe: prepend a temporal marker. The
        # pre-reframe text is stashed by _rewrite_content (dream_prev_content).
        if content.startswith("[stale:"):
            new_content = content  # already reframed — idempotent
        else:
            new_content = f"[stale: {reason}] {content}"

        if not new_content.strip():
            continue  # never emit an empty-content reframe

        actions.append(
            {
                "type": "temporal_reframe",
                "memory_ids": [memory_id],
                "content": new_content,
                "confidence": 0.95,
                "reason": f"Code anchor {anchor} {reason}",
            }
        )
        flagged_ids.append(memory_id)

    if actions:
        logger.info(
            "code liveness consumer: %d temporal_reframe actions for pool %s",
            len(actions), batch.pool,
        )
    return actions, flagged_ids


def _clear_code_liveness_flags(service, memory_ids: list[str]) -> None:
    """Clear code_liveness_stale on memories whose reframe was applied (idempotency).

    Called ONLY after ``apply_actions`` succeeds and only when not dry_run —
    so a dry run or a failed apply leaves the flag set for the next sweep.
    Best-effort/non-fatal.
    """
    if not memory_ids:
        return
    try:
        m = service._get_memory()
        client = m.vector_store.client
        collection_name = m.vector_store.collection_name
        client.set_payload(
            collection_name=collection_name,
            payload={"code_liveness_stale": False},
            points=list(memory_ids),
            key="metadata",
        )
    except Exception:
        logger.warning(
            "code liveness: failed to clear stale flags for %d memories (non-fatal)",
            len(memory_ids), exc_info=True,
        )


async def _make_llm_call(settings: DreamingSettings):
    """Build the retrying/timeout-wrapped LLM callable for this sweep.

    Exhaustion raises :class:`consolidate.DreamLLMFailure` instead of
    returning ``""`` (audit 27 #27) — a silent empty string used to parse
    to "0 actions", stamping the pool's gate and reporting "dreamt" while
    a broken LLM disabled consolidation forever.
    """
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
        raise consolidate.DreamLLMFailure(
            f"llm exhausted {attempts} attempts: "
            f"{last_exc.__class__.__name__ if last_exc else 'unknown'}"
        )

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
            batch=batch, dry_run=dry_run, force=force, run_id=run.run_id,
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

    # OKF bundle metadata: refresh per-folder index files + the root
    # version marker once per sweep, AFTER the card/diary/bridges passes
    # so their pages are listed too. Byte-idempotent (zero steady-state
    # churn); best-effort.
    if (
        settings.vault_pages_enabled
        and not dry_run
        and any(p.status == "dreamt" for p in run.pools)
    ):
        try:
            from okf.vault import refresh_bundle_indexes

            await asyncio.to_thread(refresh_bundle_indexes, settings.vault_path)
        except Exception:
            logger.warning("okf bundle index refresh failed (non-fatal)", exc_info=True)

    run.finished_at = datetime.now(timezone.utc).isoformat()
    _save_run(redis, run)
    totals = run.to_dict()["totals"]
    logger.info(
        "dream sweep %s complete: dreamt=%d gated=%d failed=%d applied=%d reported=%d insights=%d errors=%d",
        run.run_id, totals["pools_dreamt"], totals["pools_gated"],
        totals["pools_failed"], totals["actions_applied"],
        totals["actions_reported"], totals["insights_stored"], totals["errors"],
    )
    return run


async def _dream_pool(
    *, service, settings: DreamingSettings, redis, llm_call, batch, dry_run: bool,
    force: bool, run_id: str = "",
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
    #     Runs on the LIGHT enumeration rows (timestamps only).
    if not force:
        decision = gate.check_settling_gate(
            batch.memories, settling_minutes=settings.settling_minutes
        )
        if not decision.proceed:
            report.status, report.reason = "settling", decision.reason
            return report

    # 1c. cheap volume pre-gate (audit 27 #29): counted from the light rows
    #     + one Redis read — a quiet pool is skipped before its full rows
    #     are ever pulled into worker RAM (and before any lock churn).
    state = gate.get_gate_state(redis, pool)
    last_dreamt_at = float(state.get("last_dreamt_at") or 0.0)
    if not force:
        decision = gate.check_volume_gate(
            consolidate.count_new_memories(
                batch.memories, last_dreamt_at=last_dreamt_at
            ),
            min_new_memories=settings.min_new_memories,
        )
        if not decision.proceed:
            report.status, report.reason = "gated", decision.reason
            return report

    # 2. lock (shared with the dedup cron's semantic-skip check)
    lock_token = gate.acquire_lock(redis, pool)
    if not lock_token:
        report.status, report.reason = "locked", "another sweep holds the pool lock"
        return report

    try:
        # 3. hydrate (full rows, only now) + LIGHT stage/score
        batch = await asyncio.to_thread(consolidate.hydrate_pool, service, batch)
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

        # 4. authoritative volume gate (recomputed on the hydrated rows —
        #    the pre-gate at 1c used the light enumeration snapshot)
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

        # 5b. Code liveness consumer (I4): deterministic temporal_reframe for stale
        #     anchors. This is a pre-LLM pass: code changes are objective ground
        #     truth, not speculation. Flags are read straight from Qdrant (the
        #     staged rows drop them), and the flag is cleared only AFTER the
        #     reframe applies (step 6b) so a failed apply leaves work for next time.
        liveness_actions: list[dict] = []
        liveness_flagged_ids: list[str] = []
        try:
            liveness_actions, liveness_flagged_ids = _gather_code_liveness_actions(
                service, batch
            )
        except Exception:
            logger.warning(
                "code liveness consumer failed for pool %s (non-fatal)", pool, exc_info=True,
            )

        # 6. DEEP: decide + hybrid-posture apply. An LLM failure here means
        #    the pool was never examined — report sweep_failed WITHOUT
        #    stamping the gate or the staged-id marker, so the next cron
        #    cycle retries (audit 27 #27). A valid {"actions": []} decision
        #    still flows through as a completed sweep.
        try:
            actions = await consolidate.decide(batch, llm_call)
        except consolidate.DreamLLMFailure as exc:
            logger.error("dream consolidation LLM failed for pool %s: %s", pool, exc)
            report.status, report.reason = "sweep_failed", f"consolidation LLM: {exc}"
            return report
        # Merge liveness actions (deterministic) with LLM actions
        actions.extend(liveness_actions)
        to_apply, to_report = consolidate.split_by_posture(
            actions, auto_apply_confidence=settings.auto_apply_confidence
        )
        applied = await consolidate.apply_actions(service, batch, to_apply, dry_run=dry_run)
        report.applied = len(applied.applied)
        report.reported = len(to_report)
        report.errors.extend(applied.errors)

        # 6b. Clear code-liveness flags ONLY for reframes that actually applied
        #     (and never on a dry run) — idempotent: a flag left set is re-swept.
        if liveness_flagged_ids and not dry_run:
            applied_ids = {
                mid
                for act in applied.applied
                if act.get("type") == "temporal_reframe"
                for mid in (act.get("memory_ids") or [])
            }
            to_clear = [mid for mid in liveness_flagged_ids if mid in applied_ids]
            _clear_code_liveness_flags(service, to_clear)

        # Fold the applied actions into the staged dicts so the REM and
        # librarian passes below see the post-consolidation view — without
        # this, a row tombstoned seconds ago would still land on topic
        # pages and in Home's Essential Story until the next sweep.
        consolidate.reconcile_batch(batch, applied.applied)

        # E1: mirror applied consolidation onto the live event stream
        # (fire-and-forget; channel routing enforces pool visibility).
        if applied.applied and not dry_run:
            from event_stream import publish_event

            publish_event("dream_actions_applied", {
                "pool": pool,
                "run_id": run_id,
                "visibility": batch.visibility,
                "user_id": batch.owner_user_id,
                "project_id": batch.project_id,
                "applied": len(applied.applied),
                "reported": len(to_report),
                "action_types": sorted({
                    a.get("type") for a in applied.applied if a.get("type")
                }),
            })

        # 7. REM: reflect + store insights
        # WT6: skip reflection for reference workspaces — reference content is
        # imported doctrine, not user-context to reflect on or synthesize into
        # higher-order insights.
        is_reference_workspace = batch.workspace and batch.workspace != "memory"
        insights: list[dict] = []
        if settings.reflection_enabled and not is_reference_workspace:
            # A5: surprisal-targeted REM. One batched vector retrieve, then
            # each staged dict gains a `surprisal` score (cosine distance
            # from the pool centroid); reflect() biases its substrate toward
            # the top-K anomalies. top_k=0 skips everything — no fetch, no
            # keys, byte-identical uniform substrate. Failures are non-fatal
            # (reflection proceeds uniform).
            if settings.surprisal_top_k > 0:
                try:
                    vectors = await asyncio.to_thread(
                        surprisal.fetch_vectors,
                        service,
                        [m["memory_id"] for m in batch.memories],
                    )
                    surprisal.annotate(batch.memories, vectors)
                except Exception:
                    logger.warning(
                        "surprisal pass failed for pool %s (non-fatal); "
                        "reflection substrate stays uniform", pool, exc_info=True,
                    )
            # Reflection is best-effort ONCE consolidation has applied: an
            # LLM failure here must not fail the pool (the writes already
            # landed) but is recorded honestly instead of masquerading as
            # "no insights tonight".
            stored: list[str] = []
            try:
                insights = await reflect.reflect(
                    batch, llm_call, max_insights=settings.max_reflections_per_pool,
                    surprisal_top_k=settings.surprisal_top_k,
                )
            except consolidate.DreamLLMFailure:
                logger.warning(
                    "reflection LLM failed for pool %s (non-fatal; "
                    "consolidation already applied)", pool,
                )
                report.errors.append("reflection_llm")
                insights = []
            if insights:
                stored = await asyncio.to_thread(
                    reflect.store_insights, service, batch, insights, dry_run=dry_run
                )
            report.insights = len(stored) if not dry_run else len(insights)

            # E1: stream the stored insights (fire-and-forget).
            if stored and not dry_run:
                from event_stream import publish_event

                publish_event("insights_stored", {
                    "pool": pool,
                    "run_id": run_id,
                    "visibility": batch.visibility,
                    "user_id": batch.owner_user_id,
                    "project_id": batch.project_id,
                    "count": len(stored),
                    "memory_ids": stored,
                })

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
            # OKF §7: the diary doubles as bundle history — mirror this
            # sweep into the vault-root log.md (best-effort).
            if report.diary_path:
                await asyncio.to_thread(
                    reflect.update_vault_log,
                    settings.vault_path, pool,
                    diary_rel=report.diary_path,
                    applied=report.applied,
                    insights=report.insights,
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
