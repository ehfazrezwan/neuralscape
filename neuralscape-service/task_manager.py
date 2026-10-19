"""Redis-backed task management via ARQ.

Centralizes task enqueuing and status tracking, replacing the in-memory _tasks dict.
"""

import asyncio
import hashlib
import json
import logging
import time

from arq.connections import ArqRedis, create_pool
from arq.jobs import Job, JobStatus

from config import parse_redis_settings, settings

logger = logging.getLogger(__name__)


def _user_tasks_key(user_id: str) -> str:
    """Per-user sorted set of recently-enqueued task ids (score = enqueue ts)."""
    return f"ns:user-tasks:{user_id}"


def _task_user_key(task_id: str) -> str:
    """Reverse map task id → user id (queue.empty webhook attribution)."""
    return f"ns:task-user:{task_id}"

# Map ARQ JobStatus to our API status strings
_STATUS_MAP = {
    JobStatus.deferred: "queued",
    JobStatus.queued: "queued",
    JobStatus.in_progress: "processing",
    JobStatus.complete: "completed",
    JobStatus.not_found: "not_found",
}


def _fenced_view(result: object) -> tuple[str, str] | None:
    """``("fenced", reason)`` when ``result`` is the cutover fence's own marker, else ``None``.

    The one mapping used by polling (``get_status``) and waiting (``wait_for_result``), so a caller
    is never told a write finished when the fence ran nothing.
    """

    from ledger.cutover import FENCED_RESULT

    if (
        isinstance(result, dict)
        and result.get("status") == FENCED_RESULT["status"]
        and result.get("executed") is False
        and result.get("reason") == FENCED_RESULT["reason"]
    ):
        return "fenced", result["reason"]
    return None


class TaskManager:
    """Redis-backed task status tracking + ARQ job enqueuing."""

    def __init__(self):
        self.pool: ArqRedis | None = None
        # Durable ordinary-write path; both stay None unless LEDGER_ENABLED.
        self.ledger = None
        self.producer = None
        self.authority = None
        # Online cutover phase machine; None unless LEDGER_CUTOVER_ENABLED (then every write follows its phase).
        self.cutover = None
        self._owner_client = None  # synchronous Redis client for job-owner lookups

    async def connect(self) -> None:
        """Initialize the ARQ Redis connection pool."""
        self.pool = await create_pool(
            parse_redis_settings(),
            default_queue_name=settings.arq_queue_name,
        )
        logger.info("TaskManager connected to Redis")
        if settings.ledger_enabled:
            await self._open_ledger()
        else:
            from ledger import warn_if_ledger_bypassed

            await asyncio.to_thread(warn_if_ledger_bypassed, settings, logger)

    async def _open_ledger(self) -> None:
        """Open the durable ledger and wake workers for any committed but
        unqueued obligation (crash between commit and enqueue)."""
        from authority_store_sql import build_authority_store
        from ledger import open_ledger
        from write_producer import DurableWriteProducer, PrincipalResolver, open_payload_store

        self.ledger = await asyncio.to_thread(open_ledger, settings)
        # With the authority store enabled the write path is governed: intents
        # carry the bound principal, and the worker fences each stage.
        self.authority = await asyncio.to_thread(build_authority_store, settings)
        self.producer = DurableWriteProducer(
            self.ledger,
            await asyncio.to_thread(open_payload_store, settings),
            principals=None if self.authority is None else PrincipalResolver(self.authority),
        )
        if settings.ledger_cutover_enabled:
            await self._open_cutover()
        await self.dispatch_ledger()
        logger.info("TaskManager durable ledger enabled (backend=%s)", settings.ledger_backend)

    def _durable_route(self) -> bool:
        """Whether a new write is accepted by the ledger (the cutover phase decides when enabled)."""

        if self.producer is None:
            return False
        return self.cutover is None or self.cutover.writes_use_ledger()

    async def _open_cutover(self) -> None:
        """Build the cutover controller over the ledger's payload store (see ``ledger.cutover``)."""

        from ledger.cutover import CutoverController

        self.cutover = CutoverController(
            self.producer.payloads, legacy_inflight=self._legacy_inflight_sync,
        )

    def _legacy_inflight_sync(self) -> int:
        """Queued plus running legacy jobs across the legacy queues (0 means drained).

        A plain synchronous client: the cutover controller runs in a worker thread, where the ARQ
        pool (bound to the server's event loop) cannot be used.
        """

        import redis

        client = redis.Redis.from_url(settings.redis_url)
        try:
            total = 0
            for name in {settings.arq_queue_name, settings.graph_queue_name, settings.ingest_queue_name}:
                total += int(client.zcard(name))
            for key in client.scan_iter(match="arq:in-progress:*"):
                job_id = (key.decode() if isinstance(key, bytes) else str(key)).removeprefix("arq:in-progress:")
                if not job_id.startswith(("ledger-wake:", "cron:")):  # ledger wake-ups and cron ticks are not legacy writes
                    total += 1
            return total
        finally:
            client.close()

    async def _enqueue_wake(self, intent_id: str, job_id: str) -> None:
        """Queue a wake-up; the worker pulls the obligation from the ledger."""
        from write_producer import WAKE_FUNCTION, ledger_queue_name

        await self.pool.enqueue_job(WAKE_FUNCTION, _queue_name=ledger_queue_name(), _job_id=job_id)

    async def dispatch_ledger(self) -> int:
        """Wake workers for committed obligations not yet queued."""
        if self.producer is None or self.pool is None:
            return 0
        return await self.producer.dispatch_unqueued(self._enqueue_wake)

    async def _track_task(self, user_id: str | None, task_id: str) -> None:
        """Best-effort per-caller task bookkeeping (C4 queue visibility).

        Records the task id in the caller's recent-tasks sorted set (read by
        ``get_queue_status``) and the reverse task→user map (read by the
        queue.empty webhook hook). Trivial by design: two keys with a TTL of
        twice the status window — no per-status bookkeeping, statuses are
        still read from ARQ itself. Never blocks or fails an enqueue.
        """
        if not user_id or self.pool is None:
            return
        if self.producer is not None and task_id.startswith("intent-"):
            return  # durable intents are tracked by the ledger, not ARQ
        try:
            now = time.time()
            ttl = max(int(settings.queue_status_window_s) * 2, 600)
            key = _user_tasks_key(user_id)
            await self.pool.zadd(key, {task_id: now})
            await self.pool.zremrangebyscore(key, "-inf", now - ttl)
            await self.pool.expire(key, ttl)
            await self.pool.set(_task_user_key(task_id), user_id, ex=ttl)
        except Exception as e:  # noqa: BLE001 — bookkeeping must never break writes
            logger.debug(f"Task tracking failed (non-fatal): {e}")

    async def enqueue_store(
        self,
        messages: list[dict],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        occurred_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Enqueue memory extraction task. Returns task_id (job_id).

        Uses a deterministic job ID based on content hash to prevent
        duplicate enqueues of the same conversation. ``occurred_at`` (event
        time, for historical ingestion of old chat exports) is part of the
        key when present — the same transcript imported with a different
        event time is a distinct job, not a duplicate; the plain path's job
        ids are unchanged.
        """
        if self._durable_route():
            return await self.enqueue_job_durable(
                "memory.write.extracted", user_id=user_id, idempotency_key=idempotency_key, project_id=project_id,
                payload={"messages": messages, "user_id": user_id, "project_id": project_id,
                         "agent_id": agent_id, "run_id": run_id, "occurred_at": occurred_at},
            )
        # Generate deterministic job ID from message content
        content_str = "|".join(
            m.get("content", "") for m in messages
        )
        key = f"store:{content_str}"
        if occurred_at:
            key = f"store:{occurred_at}:{content_str}"
        job_id = _generate_job_id(key, user_id)

        job = await self.pool.enqueue_job(
            "process_memory_store",
            messages,
            user_id,
            project_id,
            agent_id,
            run_id,
            occurred_at,
            _job_id=job_id,
        )
        # On a duplicate job ID (job is None) ARQ already has this job —
        # return the ID so the caller can poll the existing job's status.
        task_id = job_id if job is None else job.job_id
        await self._track_task(user_id, task_id)
        return task_id

    async def enqueue_raw(
        self,
        content: str,
        user_id: str,
        category: str,
        scope: str = "global",
        project_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        # Memory-model v2 fields
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        source_type: str | None = None,
        related_memory_ids: list[str] | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
        # Event time (ISO 8601, for historical ingestion)
        occurred_at: str | None = None,
        # Provenance epistemics (A1)
        derived_from: list[str] | None = None,
        epistemic_level: str | None = None,
        # Multi-user model
        visibility: str | None = None,
        # Sensitivity gate override (see memory/sensitivity.py + schemas.
        # RawMemoryRequest.sensitivity_override) — bypasses the forced-private
        # gate only when combined with an explicit `visibility` above.
        sensitivity_override: bool = False,
        # Data-layer connectors
        memory_kind: str | None = None,
        source_ref: dict | None = None,
        # Durable path only (LEDGER_ENABLED): caller-chosen idempotency key.
        idempotency_key: str | None = None,
    ) -> str:
        """Enqueue raw memory storage task. Returns task_id (job_id).

        With ``LEDGER_ENABLED`` the write is durably accepted in the ledger
        first (intent + outbox in one transaction) and the returned task id is
        the intent id; the queue only carries a wake-up.  The same
        ``idempotency_key`` with the same payload returns the original intent;
        with a different payload the ledger's ``IdempotencyConflict`` is
        raised.  Without the flag nothing below changes.

        Uses a deterministic job ID based on content hash to prevent
        duplicate enqueues of the same fact. v2 + multi-user fields are
        forwarded as a kwargs dict to the worker so the signature stays
        backward-compatible.

        The job id includes visibility/scope/project_id so it stays at least as
        granular as store_raw's content-hash dedup key. Otherwise a later write
        of the SAME text at a different tier (e.g. a dictator promoting a private
        note to a `standard`) would coalesce onto the earlier job's id and be
        silently dropped by ARQ before ever reaching store_raw.

        ``sensitivity_override`` is also part of the key: it changes the
        STORED visibility (memory/write.py's sensitivity gate forces
        visibility=private for a sensitive write unless override is set),
        so two otherwise-identical raw stores that differ only by this flag
        must not coalesce onto one job id — that would let an override-on
        request silently absorb (and skip) an override-off request for the
        same content, or vice versa.
        """
        v2_extras = {
            "domain": domain,
            "observation_type": observation_type,
            "concepts": concepts,
            "source_type": source_type,
            "related_memory_ids": related_memory_ids,
            "confidence": confidence,
            "expires_at": expires_at,
            "occurred_at": occurred_at,
            "derived_from": derived_from,
            "epistemic_level": epistemic_level,
            "visibility": visibility,
            "sensitivity_override": sensitivity_override,
            "memory_kind": memory_kind,
            "source_ref": source_ref,
        }
        # Drop None values so the worker signature can default cleanly
        v2_extras = {k: v for k, v in v2_extras.items() if v is not None}

        if self._durable_route():
            return await self._enqueue_raw_durable(
                user_id=user_id,
                payload={
                    "content": content,
                    "user_id": user_id,
                    "category": category,
                    "scope": scope,
                    "project_id": project_id,
                    "tags": tags,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "v2_extras": v2_extras,
                },
                idempotency_key=idempotency_key,
            )

        job_id = _generate_job_id(
            f"raw:{visibility}:{scope}:{project_id}:{sensitivity_override}:{content}", user_id
        )

        job = await self.pool.enqueue_job(
            "process_memory_raw",
            content,
            user_id,
            category,
            scope,
            project_id,
            tags,
            agent_id,
            run_id,
            v2_extras,
            _job_id=job_id,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(user_id, task_id)
        return task_id

    async def _enqueue_raw_durable(
        self, *, user_id: str, payload: dict, idempotency_key: str | None
    ) -> str:
        """Commit the intent and outbox obligation, then wake a worker.

        The client is told "accepted" only after the ledger commit.  A failure
        to enqueue the wake-up never fails the write: the committed obligation
        is recovered by the worker's ledger poll or the next dispatch.
        """
        result = await asyncio.to_thread(
            self.producer.accept_raw,
            user_id=user_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        try:
            await self.dispatch_ledger()
        except Exception:  # noqa: BLE001 - durable accept already happened
            logger.warning("ledger dispatch failed after accept", exc_info=True)
        return result.intent_id

    async def _accepted_durably(self, accept) -> str:
        """Commit through the ledger, then wake a worker (never failing the accepted write)."""

        result = await asyncio.to_thread(accept)
        try:
            await self.dispatch_ledger()
        except Exception:  # noqa: BLE001 - durable accept already happened
            logger.warning("ledger dispatch failed after accept", exc_info=True)
        return result.intent_id

    async def enqueue_delete_durable(
        self, *, user_id: str, memory_id: str, memory_row: dict | None, idempotency_key: str | None = None
    ) -> str:
        """Accept an ordinary delete (raises ``ledger.port.HoldActive`` under a hold)."""

        return await self._accepted_durably(
            lambda: self.producer.accept_delete(
                user_id=user_id, memory_id=memory_id, memory_row=memory_row, idempotency_key=idempotency_key
            )
        )

    async def enqueue_ingest_document(self, doc: dict, *, idempotency_key: str | None = None) -> str:
        """Enqueue a document-ingest task (chunk → passages + facts). Returns job_id.

        Deterministic job id from the content hash + connector instance so the
        same document re-submitted by a re-sync coalesces onto one job. Visibility
        is part of the key so ingesting the same text at a different tier (e.g. a
        dictator ingesting it as a `standard`) isn't coalesced onto an earlier
        non-standard job and dropped. The knowledge adapter is part of the key for
        the same reason — re-ingesting the same content under a different adapter
        (different taxonomy/extractor/ontology) is a distinct job, not a duplicate.
        """
        if self._durable_route():
            return await self._accepted_durably(
                lambda: self.producer.accept_ingest(
                    user_id=doc.get("user_id") or settings.default_user_id, doc=doc,
                    idempotency_key=idempotency_key,
                )
            )
        content = doc.get("content", "")
        connector_id = (doc.get("source") or {}).get("connector_id", "")
        visibility = doc.get("visibility")
        adapter = doc.get("adapter", "default")
        partition = doc.get("user_id") or "ingest"
        job_id = _generate_job_id(
            f"ingest:{visibility}:{adapter}:{connector_id}:{content}", partition
        )

        job = await self.pool.enqueue_job(
            "process_ingest_document",
            doc,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(doc.get("user_id"), task_id)
        return task_id

    async def enqueue_job_durable(
        self, operation: str, *, payload: dict, user_id: str | None = None,
        idempotency_key: str | None = None, visibility: str | None = None,
        project_id: str | None = None, workspace: str | None = None, stable=None,
    ) -> str:
        """Accept one single-stage legacy job through the ledger (raises ``HoldActive`` for deletes)."""

        return await self._accepted_durably(
            lambda: self.producer.accept_job(
                operation=operation, payload=payload, user_id=user_id or settings.default_user_id,
                idempotency_key=idempotency_key, visibility=visibility, project_id=project_id,
                workspace=workspace, stable=stable,
            )
        )

    async def enqueue_system_jobs_durable(self, operation: str, payload: dict | None = None,
                                          idempotency_key: str | None = None) -> list[str]:
        """Accept a cron/admin-started job under explicit service principals (never a default user)."""

        results = await asyncio.to_thread(
            self.producer.accept_system_jobs, operation=operation, payload=payload,
            idempotency_key=idempotency_key,
        )
        try:
            await self.dispatch_ledger()
        except Exception:  # noqa: BLE001 - durable accept already happened
            logger.warning("ledger dispatch failed after accept", exc_info=True)
        return [r.intent_id for r in results]

    async def enqueue_conversation_flush(self, **kwargs) -> str:
        return await self.enqueue_job_durable(
            "conversation.flush", payload=kwargs, user_id=kwargs.get("user_id"),
            project_id=kwargs.get("project_id"),
        )

    async def enqueue_conversation_compile(self, date: str | None, user_id: str) -> str:
        return await self.enqueue_job_durable(
            "conversation.compile", payload={"date": date, "user_id": user_id}, user_id=user_id,
        )

    async def enqueue_markdown_write(self, rel_path: str, content: str, user_id: str | None = None) -> str:
        return await self._accepted_durably(
            lambda: self.producer.accept_markdown_write(rel_path=rel_path, content=content, user_id=user_id)
        )

    async def enqueue_ingest_file(self, payload: dict, *, idempotency_key: str | None = None) -> str:
        """Enqueue a single-file ingest task (parse → passages + facts). Returns job_id.

        ``payload`` carries ``{filename, source_ref, options}`` plus either
        ``stored_path`` or ``data_b64``. The job id is deterministic from the
        artifact's content hash (source_ref.external_id) + owner so re-uploading
        the same file coalesces onto one job (idempotent). Every option that
        changes the write's *semantics* must be part of the key — visibility,
        the knowledge adapter, and page_offset — so the same file uploaded at a
        different tier / adapter / page numbering is a distinct job, not
        coalesced onto (and dropped by) an earlier one. Runs on the ingest queue.
        """
        if self._durable_route():
            options = payload.get("options") or {}
            return await self.enqueue_job_durable(
                "ingest.file", payload={"payload": payload}, user_id=payload.get("user_id"),
                idempotency_key=idempotency_key, visibility=options.get("visibility"),
                project_id=options.get("project_id"),
            )
        partition = payload.get("user_id") or "ingest"
        options = payload.get("options") or {}
        visibility = options.get("visibility")
        adapter = options.get("adapter", "default")
        page_offset = options.get("page_offset") or 0
        content_key = (payload.get("source_ref") or {}).get("external_id") or payload.get(
            "stored_path"
        ) or payload.get("data_b64", "")
        job_id = _generate_job_id(
            f"ingest-file:{visibility}:{adapter}:{page_offset}:"
            f"{payload.get('filename', '')}:{content_key}",
            partition,
        )
        job = await self.pool.enqueue_job(
            "process_ingest_file",
            payload,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(payload.get("user_id"), task_id)
        return task_id

    async def enqueue_ingest_okf_bundle(self, payload: dict, *, idempotency_key: str | None = None) -> str:
        """Enqueue a whole-bundle OKF ingest task. Returns job_id.

        Same payload contract as ``enqueue_ingest_file`` (the artifact is
        the bundle zip), but the worker walks it as ONE knowledge bundle.
        The deterministic id is namespaced apart from the per-file task
        (``ingest-okf:``) and keys on scope/project/visibility so the same
        bundle imported into a different pool is a distinct job rather
        than being coalesced onto (and dropped by) an earlier one.
        """
        if self._durable_route():
            options = payload.get("options") or {}
            return await self.enqueue_job_durable(
                "ingest.okf", payload={"payload": payload}, user_id=payload.get("user_id"),
                idempotency_key=idempotency_key, visibility=options.get("visibility"),
                project_id=options.get("project_id"),
            )
        partition = payload.get("user_id") or "ingest"
        options = payload.get("options") or {}
        content_key = (payload.get("source_ref") or {}).get("external_id") or payload.get(
            "stored_path"
        ) or payload.get("data_b64", "")
        job_id = _generate_job_id(
            f"ingest-okf:{options.get('visibility')}:{options.get('scope')}:"
            f"{options.get('project_id')}:{payload.get('filename', '')}:{content_key}",
            partition,
        )
        job = await self.pool.enqueue_job(
            "process_ingest_okf_bundle",
            payload,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )
        if job is None:
            return job_id
        return job.job_id

    async def enqueue_code_index(self, payload: dict, *, idempotency_key: str | None = None) -> str:
        """Phase G: enqueue a through-NS code-index task on the ingest queue.

        ``payload`` = ``{system, repo_source, code_space, project_id, user_id}``.
        Runs on the ingest queue for uniform observability + 202-task semantics
        (PLAN §5). Indexing is **idempotent at the engine level** — a re-index of
        the same code_space overwrites (CBM full re-index; graphify rebuild;
        native incremental) — so the job id is left UNIQUE per call (ARQ-assigned)
        rather than coalesced on (system, code_space): a completed/failed job's
        result would otherwise be cached under a deterministic id and a later
        re-index would return the stale result instead of re-running.
        """
        if self._durable_route():
            return await self.enqueue_job_durable(
                "code.index", payload={"payload": payload}, user_id=payload.get("user_id"),
                idempotency_key=idempotency_key, project_id=payload.get("project_id"),
            )
        job = await self.pool.enqueue_job(
            "process_code_index",
            payload,
            _queue_name=settings.ingest_queue_name,
        )
        task_id = job.job_id
        await self._track_task(payload.get("user_id"), task_id)
        return task_id

    async def enqueue_connector_sync(
        self, connector_id: str, *, record: dict | None = None, idempotency_key: str | None = None
    ) -> str:
        """Enqueue a connector sync task. Returns job_id.

        Job id is keyed on the connector instance so overlapping syncs of the
        same connector coalesce rather than stacking. Must match the id scheme
        used by ``connector_sync_cron`` in worker.py (``sync-<connector_id>``)
        so a cron-triggered and an API-triggered sync of the same connector
        coalesce instead of racing on the cursor/revision state.
        """
        if self._durable_route():
            owner = (record or {}).get("owner_user_id")
            if not owner:  # never sync under a default identity, from the API any more than from the cron
                raise ValueError("connector has no owner; refusing to sync under a default user")
            return await self._accepted_durably(
                lambda: self.producer.accept_connector_sync(
                    user_id=owner, connector_id=connector_id, record=record, idempotency_key=idempotency_key
                )
            )
        job_id = f"sync-{connector_id}"
        job = await self.pool.enqueue_job(
            "process_connector_sync",
            connector_id,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )
        if job is None:
            return job_id
        return job.job_id

    async def enqueue_raw_batch(self, items: list[dict], *, idempotency_key: str | None = None) -> str:
        """Enqueue a batch of raw memory storage tasks (memory-model v2).

        Items are dispatched as a single ARQ job. Job ID is deterministic
        based on a canonical JSON encoding of the items so distinct batches
        cannot collide even when item content includes delimiter characters
        like ``|`` or quotes.
        """
        if self._durable_route():
            return await self._accepted_durably(
                lambda: self.producer.accept_batch(items=items, idempotency_key=idempotency_key)
            )
        # Canonical JSON gives us a representation that's stable across
        # re-orderings of dict keys (sort_keys=True) and unambiguous w.r.t.
        # special characters — far safer than join(delimiter, ...).
        canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
        # Keep the deterministic-id key partitioned by the first item's user
        # so two users batching the same content still get distinct job ids.
        partition_user = items[0].get("user_id", "batch") if items else "batch"
        job_id = _generate_job_id(f"raw_batch:{canonical}", partition_user)

        job = await self.pool.enqueue_job(
            "process_memory_raw_batch",
            items,
            _job_id=job_id,
        )
        task_id = job_id if job is None else job.job_id
        # Mixed-user batches are legal — track for every distinct writer.
        for uid in {i.get("user_id") for i in items if i.get("user_id")}:
            await self._track_task(uid, task_id)
        return task_id

    async def enqueue_retag(self, caller_user_id: str, filters: dict, ops: dict,
                            *, idempotency_key: str | None = None) -> str:
        """Enqueue a bulk retag task on the fast queue. Returns job_id.

        Canonical-JSON job id (sort_keys) so two *different* retags can never
        collide while an identical replay coalesces within ARQ's dedup window
        — the same guarantee enqueue_raw_batch relies on.
        """
        if self._durable_route():
            return await self.enqueue_job_durable(
                "memory.retag", payload={"caller_user_id": caller_user_id, "filters": filters, "ops": ops},
                user_id=caller_user_id, idempotency_key=idempotency_key, project_id=filters.get("project_id"),
            )
        canonical = json.dumps(
            {"filters": filters, "ops": ops}, sort_keys=True, separators=(",", ":")
        )
        job_id = _generate_job_id(f"retag:{canonical}", caller_user_id or "anon")
        job = await self.pool.enqueue_job(
            "process_memory_retag",
            caller_user_id,
            filters,
            ops,
            _job_id=job_id,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(caller_user_id, task_id)
        return task_id

    async def enqueue_graph_enrichment(
        self,
        memory_id: str,
        content: str,
        user_id: str,
        project_id: str | None = None,
        visibility: str | None = None,
        source_ref: dict | None = None,
    ) -> str:
        """Enqueue a graph re-ingest for an edited memory (graph queue). Returns job_id.

        Used by the memory-edit paths: after a content change or a
        project/visibility partition migration, the memory's content must be
        re-ingested into Graphiti (contradiction detection / new group_id)
        without ever blocking a request thread. The job id keys on the target
        state (memory + visibility + project + content hash) so a second edit
        of the same memory to a *different* state is a distinct job, while a
        replay of the same edit coalesces.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        job_id = _generate_job_id(
            f"graph-edit:{memory_id}:{visibility}:{project_id}:{content_hash}", user_id
        )
        job = await self.pool.enqueue_job(
            "process_graph_enrichment",
            memory_id,
            content,
            user_id,
            project_id,
            visibility,
            source_ref,
            None,  # adapter — regular memories carry no knowledge-adapter ontology
            _job_id=job_id,
            _queue_name=settings.graph_queue_name,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(user_id, task_id)
        return task_id

    def _candidate_queues(self) -> list[str]:
        """Queues a poll-able job could live on (main + ingest), de-duplicated.

        Ingest/connector-sync jobs run on a dedicated queue, so status polling
        must look there too or ``/v1/memories/status/{id}`` would report every
        ingest job as not_found.
        """
        queues = [settings.arq_queue_name, settings.ingest_queue_name]
        seen: set[str] = set()
        return [q for q in queues if not (q in seen or seen.add(q))]

    async def _find_job(self, task_id: str) -> tuple[Job, JobStatus]:
        """Return the (job, status) for ``task_id``, searching candidate queues.

        Returns the first queue where the job is known; falls back to a
        main-queue job with not_found status if it's on none of them.
        """
        fallback: Job | None = None
        for queue_name in self._candidate_queues():
            job = Job(task_id, redis=self.pool, _queue_name=queue_name)
            status = await job.status()
            if status != JobStatus.not_found:
                return job, status
            if fallback is None:
                fallback = job
        # _candidate_queues() always yields at least the main queue, so fallback
        # is set — but construct one defensively to keep the (Job, _) contract.
        if fallback is None:
            fallback = Job(task_id, redis=self.pool, _queue_name=settings.arq_queue_name)
        return fallback, JobStatus.not_found

    async def get_status(self, task_id: str, user_id: str | None = None) -> dict:
        """Get task status from ARQ/Redis.

        Returns:
            Dict with task_id, status, result, error keys.  For a durable
            (ledger) intent the result carries the per-stage receipt.
        """
        if self.producer is not None and task_id.startswith("intent-"):
            return await self._durable_status(task_id, user_id)
        job, status = await self._find_job(task_id)

        api_status = _STATUS_MAP.get(status, "not_found")

        result = None
        error = None

        if status == JobStatus.complete:
            info = await job.result_info()
            if info is not None:
                if info.success:
                    result = info.result
                    fenced = _fenced_view(result)
                    if fenced:
                        # the job ran nothing: a poller must not be told the write finished
                        api_status, error = fenced
                else:
                    api_status = "failed"
                    error = str(info.result) if info.result else "Unknown error"

        return {
            "task_id": task_id,
            "status": api_status,
            "result": result,
            "error": error,
        }

    def _read_owned_intent(self, task_id: str, user_id: str | None):
        """``(tenant, stored, receipts)`` for an intent the caller may read, else ``(tenant, None, ())``.

        With the authority store the read is scoped to the caller's tenant and to the subject that made
        the write.
        """

        producer = self.producer
        tenant = producer.tenant_for(user_id)
        if tenant is None:
            return None, None, ()
        stored = self.ledger.get_intent(tenant, task_id)
        if stored is not None and producer.principals is not None:
            subject = producer.principal_for(user_id).subject_id
            if stored.intent.actor_id != subject:
                stored = None
        return tenant, stored, (self.ledger.receipts(tenant, task_id) if stored else ())

    def _output_resolver(self, task_id: str):
        from write_producer import stage_output_refs

        payloads = self.producer.payloads
        return lambda handle: stage_output_refs(payloads, handle, expected_intent_id=task_id)

    def job_owner(self, tenant_id: str, task_id: str):
        """The principal an accepted job is bound to (synchronous; used by the enforcer).

        A durable intent names its submitting subject in the ledger (only on the
        governed path, where intents carry the bound principal).  A legacy queue job
        is mapped to the submitting user through the short-lived task→user key the
        enqueue path records.  Unknown or unreadable is ``None`` (no access through it).
        """

        from policy_enforcement import JobOwner

        if task_id.startswith("intent-"):
            producer = self.producer
            if self.ledger is None or producer is None or producer.principals is None:
                return None
            stored = self.ledger.get_intent(tenant_id, task_id)
            return None if stored is None else JobOwner(subject_id=stored.intent.actor_id)
        import redis

        client = self._owner_client
        if client is None:
            client = self._owner_client = redis.Redis.from_url(
                settings.redis_url, socket_timeout=2, socket_connect_timeout=2
            )
        user = client.get(_task_user_key(task_id))
        if not user:
            return None
        return JobOwner(user_id=user.decode() if isinstance(user, bytes) else str(user))

    async def _durable_status(self, task_id: str, user_id: str | None = None) -> dict:
        """Receipt-facade status for a ledger intent (never derived from ARQ).

        With the authority store the read is scoped to the caller's tenant and
        to the subject that made the write; anything else is ``not_found``.
        """
        from receipts import legacy_task_status
        from write_producer import MemoryRevisions

        def read():
            tenant, stored, receipts = self._read_owned_intent(task_id, user_id)
            if tenant is None:
                return None, (), None
            revisions = MemoryRevisions(self.ledger)
            return stored, receipts, (lambda memory_id: revisions.latest(tenant, memory_id))

        stored, receipts, revision_of = await asyncio.to_thread(read)
        return legacy_task_status(task_id, stored, receipts, revision_of, self._output_resolver(task_id))

    async def task_outputs(
        self, task_id: str, user_id: str | None = None, *, offset: int = 0, limit: int = 256
    ) -> dict | None:
        """One page of a durable intent's outputs, with aggregated lists expanded; ``None`` if not readable.

        Raises ``LookupError`` when an aggregated list is no longer available (the page would be incomplete).
        """
        from receipts import output_view, outputs_compacted, receipt_outputs
        from write_producer import MemoryRevisions

        if self.producer is None or not task_id.startswith("intent-"):
            return None

        def read():
            tenant, stored, receipts = self._read_owned_intent(task_id, user_id)
            if stored is None:
                return None
            pairs, unavailable = receipt_outputs(receipts, self._output_resolver(task_id))
            if unavailable or outputs_compacted(stored):
                raise LookupError("the outputs of this task are no longer available")
            revisions = MemoryRevisions(self.ledger)
            page = pairs[offset:offset + limit]
            return {
                "task_id": task_id,
                "total": len(pairs),
                "offset": offset,
                "limit": limit,
                "outputs": [output_view(k, i, lambda m: revisions.latest(tenant, m)) for k, i in page],
                "next_offset": offset + len(page) if offset + len(page) < len(pairs) else None,
            }

        return await asyncio.to_thread(read)

    async def enqueue_edit(
        self,
        *,
        memory_id: str,
        user_id: str,
        changes: dict,
        expected_revision: int,
        memory_row: dict,
        idempotency_key: str | None = None,
    ) -> tuple[str, int]:
        """Durably accept a governed correction; returns ``(task_id, new_revision)``.

        Raises ``write_producer.StaleRevision`` (no write) for a stale expected
        revision, and the ledger's ``IdempotencyConflict`` for key reuse.
        """
        result, new_revision = await asyncio.to_thread(
            self.producer.accept_edit,
            user_id=user_id,
            memory_id=memory_id,
            changes=changes,
            expected_revision=expected_revision,
            memory_row=memory_row,
            idempotency_key=idempotency_key,
        )
        try:
            await self.dispatch_ledger()
        except Exception:  # noqa: BLE001 - durable accept already happened
            logger.warning("ledger dispatch failed after accept", exc_info=True)
        return result.intent_id, new_revision

    async def wait_for_result(
        self, task_id: str, timeout: float = 300.0, user_id: str | None = None
    ) -> dict:
        """Wait for a task to complete and return its status.

        Used by MCP tools with wait=true.
        """
        if self.producer is not None and task_id.startswith("intent-"):
            deadline = time.monotonic() + timeout
            while True:
                status = await self._durable_status(task_id, user_id)
                if status["status"] in ("completed", "failed", "not_found"):
                    return status
                if time.monotonic() >= deadline:
                    return {**status, "status": "failed", "error": "timed out waiting for the receipt"}
                await asyncio.sleep(0.25)
        job, _ = await self._find_job(task_id)
        try:
            result = await job.result(timeout=timeout)
            fenced = _fenced_view(result)
            return {
                "task_id": task_id,
                "status": fenced[0] if fenced else "completed",
                "result": result,
                "error": fenced[1] if fenced else None,
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "status": "failed",
                "result": None,
                "error": str(e),
            }

    async def get_queue_status(
        self,
        user_id: str,
        window_seconds: int | None = None,
        cap: int = 200,
    ) -> dict:
        """Aggregate task-status view for one caller (C4 queue visibility).

        Reads the caller's recently-enqueued task ids (recorded by
        ``_track_task``) and aggregates their live ARQ statuses — no new
        per-status bookkeeping, just what task_manager already tracks in
        Redis. ``expired`` counts tasks whose result aged out of ARQ's
        keep_result TTL (indistinguishable from never-existed, reported
        honestly instead of guessed). ``queues`` reports instance-wide
        pending depths per queue (ARQ's queue is a sorted set named after
        the queue). ``caught_up`` is True when the caller has nothing
        queued or in flight.
        """
        window = int(window_seconds or settings.queue_status_window_s)
        counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "expired": 0}
        ids: list[str] = []
        if self.pool is not None:
            now = time.time()
            try:
                raw_ids = await self.pool.zrevrangebyscore(
                    _user_tasks_key(user_id), max=now, min=now - window,
                    start=0, num=cap,
                )
            except Exception as e:  # noqa: BLE001 — a read must degrade, not 500
                logger.warning(f"queue_status: tracked-task read failed: {e}")
                raw_ids = []
            ids = [i.decode() if isinstance(i, bytes) else str(i) for i in raw_ids or []]
            for task_id in ids:
                try:
                    status = (await self.get_status(task_id))["status"]
                except Exception:  # noqa: BLE001
                    status = "not_found"
                if status == "not_found":
                    counts["expired"] += 1
                else:
                    counts[status] = counts.get(status, 0) + 1

        queues: dict[str, int] = {}
        queue_names = (
            ("main", settings.arq_queue_name),
            ("graph", settings.graph_queue_name),
            ("ingest", settings.ingest_queue_name),
        )
        for label, queue_name in queue_names:
            try:
                queues[label] = int(await self.pool.zcard(queue_name))
            except Exception:  # noqa: BLE001
                queues[label] = 0

        return {
            "user_id": user_id,
            "window_seconds": window,
            "tracked": len(ids),
            "counts": counts,
            "queues": queues,
            "caught_up": counts["queued"] == 0 and counts["processing"] == 0,
        }

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self.pool:
            await self.pool.aclose()
            logger.info("TaskManager disconnected from Redis")
        if self.producer is not None:
            await asyncio.to_thread(self.producer.payloads.close)
            await asyncio.to_thread(self.ledger.close)
            if self.authority is not None:
                await asyncio.to_thread(self.authority.close)
            self.producer = self.ledger = self.authority = None


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
