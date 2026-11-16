"""Redis-backed task management via ARQ.

Centralizes task enqueuing and status tracking, replacing the in-memory _tasks dict.
"""

import asyncio
import hashlib
import json
import logging
import time

from arq.connections import ArqRedis, create_pool
from arq.constants import result_key_prefix
from arq.jobs import Job, JobStatus

from config import parse_redis_settings, settings

logger = logging.getLogger(__name__)


def _user_tasks_key(user_id: str) -> str:
    """Per-user sorted set of recently-enqueued task ids (score = enqueue ts)."""
    return f"ns:user-tasks:{user_id}"


def _task_user_key(task_id: str) -> str:
    """Reverse map task id → user id (queue.empty webhook attribution)."""
    return f"ns:task-user:{task_id}"


def _task_owner_key(task_id: str) -> str:
    """Set of user ids allowed to read a task's status and result."""
    return f"ns:task-owner:{task_id}"


# ARQ drops a job that has not started within a day of being enqueued
# (``expires_extra_ms``) and keeps a finished job's result for an hour
# (``keep_result``). The owner record must outlive both, or an owner polling a
# slow job would be told it does not exist.
_OWNER_TTL_FLOOR_S = 2 * 24 * 3600


def _task_owner_ttl() -> int:
    return max(int(settings.queue_status_window_s) * 2, _OWNER_TTL_FLOOR_S)


class TaskStatusUnavailable(Exception):
    """The owner check could not be completed (Redis error).

    The status routes answer this with a 503 for every task id alike. A 404
    would tell the real owner that their task does not exist, so a client
    could give up or resubmit the write; a 503 is retryable and, since it is
    raised before anything about the task is read, says nothing about whether
    the task exists or whose it is.
    """


async def record_task_owner(pool, task_id: str | None, *user_ids: str | None) -> bool:
    """Record ``user_ids`` as allowed to read ``task_id``'s status.

    For every enqueue that hands a task id to a caller who may poll it on
    ``/v1/memories/status`` (TaskManager enqueues, and extension or MCP
    routes that enqueue on the ARQ pool directly). The set member and its
    TTL are written in one MULTI/EXEC so an owner key can never be left
    without an expiry. Best-effort: a failure leaves the task unreadable
    through a per-user token (the read fails closed) and never fails the
    enqueue. Returns whether the record was written.
    """
    owners = sorted({u for u in user_ids if u})
    if pool is None or not task_id or not owners:
        return False
    key = _task_owner_key(task_id)
    try:
        async with pool.pipeline(transaction=True) as pipe:
            pipe.sadd(key, *owners)
            pipe.expire(key, _task_owner_ttl())
            await pipe.execute()
        return True
    except Exception as e:  # noqa: BLE001 — must never break writes
        logger.warning(f"Task owner record failed for {task_id}: {e}")
        return False

# Map ARQ JobStatus to our API status strings
_STATUS_MAP = {
    JobStatus.deferred: "queued",
    JobStatus.queued: "queued",
    JobStatus.in_progress: "processing",
    JobStatus.complete: "completed",
    JobStatus.not_found: "not_found",
}


async def enqueue_graph_jobs_standalone(jobs: list[dict]) -> dict:
    """Enqueue graph re-ingest jobs from code with no queue handle (the dream
    sweep's prune, the post-compile dedup) through a short-lived TaskManager.

    Never raises: returns ``graph_jobs_enqueued`` and
    ``graph_jobs_not_enqueued_ids`` like :meth:`TaskManager.enqueue_graph_jobs`.
    """
    tm = TaskManager()
    try:
        await tm.connect()
        return await tm.enqueue_graph_jobs(jobs)
    except Exception as e:
        logger.warning("Sibling graph re-enrichment could not be queued (non-critical): %s", e)
        return {
            "graph_jobs_enqueued": 0,
            "graph_jobs_not_enqueued_ids": [j.get("memory_id") for j in jobs],
        }
    finally:
        pool = getattr(tm, "pool", None)
        if pool is not None:
            try:
                await pool.aclose()
            except Exception:
                pass


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

    def _durable_route(self) -> bool:
        """Whether a new write is accepted by the ledger (the cutover phase decides when enabled)."""

        if self.producer is None:
            return False
        return self.cutover is None or self.cutover.writes_use_ledger()

    async def _track_task(
        self, user_id: str | None, task_id: str, *, grant_read: bool = True
    ) -> None:
        """Best-effort per-caller task bookkeeping (C4 queue visibility).

        Records the task id in the caller's recent-tasks sorted set (read by
        ``get_queue_status``) and the reverse task→user map (read by the
        queue.empty webhook hook). Trivial by design: two keys with a TTL of
        twice the status window — no per-status bookkeeping, statuses are
        still read from ARQ itself. Never blocks or fails an enqueue.

        With ``grant_read`` (the default) ``user_id`` may also read the task's
        status and result through a per-user token. Queue bookkeeping and the
        read grant are separate: a task whose result is not ``user_id``'s
        alone (a mixed-user batch) is counted for them without the grant.
        """
        if not user_id or self.pool is None:
            return
        if self.producer is not None and task_id.startswith("intent-"):
            return  # durable intents are tracked by the ledger, not ARQ
        if grant_read:
            await self._record_owner(user_id, task_id)
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

    async def _record_owner(self, user_id: str, task_id: str) -> None:
        """Record ``user_id`` as allowed to read ``task_id`` (see ``record_task_owner``).

        A set, because one deterministic job id can be enqueued by several
        users who each submitted that same work (for example two editors
        making the same change to a shared memory).
        """
        await record_task_owner(self.pool, task_id, user_id)

    async def is_task_owner(self, task_id: str, user_id: str | None) -> bool:
        """Whether ``user_id`` enqueued ``task_id`` (False when unknown).

        Raises ``TaskStatusUnavailable`` when Redis cannot answer.
        """
        if not user_id or self.pool is None:
            return False
        try:
            return bool(await self.pool.sismember(_task_owner_key(task_id), user_id))
        except Exception as e:  # noqa: BLE001 — any Redis failure: deny, retryably
            raise TaskStatusUnavailable(str(e)) from e

    async def enqueue_store(
        self,
        messages: list[dict],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        occurred_at: str | None = None,
        # Durable path only (LEDGER_ENABLED): accepted and unused until writes use the ledger.
        idempotency_key: str | None = None,
    ) -> str:
        """Enqueue memory extraction task. Returns task_id (job_id).

        Uses a deterministic job ID based on content hash to prevent
        duplicate enqueues of the same conversation. ``occurred_at`` (event
        time, for historical ingestion of old chat exports), ``project_id``,
        ``agent_id`` and ``run_id`` are part of the key: the same text sent to
        another project, agent or run (or imported with another event time)
        is a distinct job, not a duplicate. A retry of the identical request
        still derives the identical id.
        """
        # Generate deterministic job ID from message content
        content_str = "|".join(
            m.get("content", "") for m in messages
        )
        key = (
            f"store:{occurred_at or ''}:{_scope_key(project_id=project_id, agent_id=agent_id, run_id=run_id)}"
            f":{content_str}"
        )
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
        # Durable path only (LEDGER_ENABLED): accepted and unused until writes use the ledger.
        idempotency_key: str | None = None,
    ) -> str:
        """Enqueue raw memory storage task. Returns task_id (job_id).

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

        Everything else that changes what is stored is in the key too
        (category, tags, agent, run, and the v2 / provenance fields), so two
        writes of the same text that differ in any of them are distinct jobs.
        Tags are compared as a set. A retry of the identical request derives
        the identical id.
        """
        options = _scope_key(
            category=category,
            tags=sorted(set(tags)) if tags else None,
            agent_id=agent_id,
            run_id=run_id,
            domain=domain,
            observation_type=observation_type,
            concepts=sorted(set(concepts)) if concepts else None,
            source_type=source_type,
            related_memory_ids=sorted(set(related_memory_ids)) if related_memory_ids else None,
            confidence=confidence,
            expires_at=expires_at,
            occurred_at=occurred_at,
            derived_from=sorted(set(derived_from)) if derived_from else None,
            epistemic_level=epistemic_level,
            memory_kind=memory_kind,
            source_ref=source_ref,
        )
        job_id = _generate_job_id(
            f"raw:{visibility}:{scope}:{project_id}:{sensitivity_override}:{options}:{content}",
            user_id,
        )

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

    async def enqueue_ingest_document(self, doc: dict) -> str:
        """Enqueue a document-ingest task (chunk → passages + facts). Returns job_id.

        Deterministic job id from the content hash + connector instance so the
        same document re-submitted by a re-sync coalesces onto one job. Visibility
        is part of the key so ingesting the same text at a different tier (e.g. a
        dictator ingesting it as a `standard`) isn't coalesced onto an earlier
        non-standard job and dropped. The knowledge adapter is part of the key for
        the same reason — re-ingesting the same content under a different adapter
        (different taxonomy/extractor/ontology) is a distinct job, not a duplicate.
        """
        content = doc.get("content", "")
        connector_id = (doc.get("source") or {}).get("connector_id", "")
        visibility = doc.get("visibility")
        adapter = doc.get("adapter", "default")
        partition = doc.get("user_id") or "ingest"
        workspace = doc.get("workspace")
        job_id = _generate_job_id(
            f"ingest:{visibility}:{adapter}:{connector_id}:"
            + (f"ws={workspace}:" if workspace else "")
            + content,
            partition,
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

    async def enqueue_ingest_file(self, payload: dict) -> str:
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
        partition = payload.get("user_id") or "ingest"
        options = payload.get("options") or {}
        visibility = options.get("visibility")
        adapter = options.get("adapter", "default")
        page_offset = options.get("page_offset") or 0
        workspace = options.get("workspace")
        content_key = (payload.get("source_ref") or {}).get("external_id") or payload.get(
            "stored_path"
        ) or payload.get("data_b64", "")
        job_id = _generate_job_id(
            f"ingest-file:{visibility}:{adapter}:{page_offset}:"
            + (f"ws={workspace}:" if workspace else "")
            + f"{payload.get('filename', '')}:{content_key}",
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

    async def enqueue_ingest_okf_bundle(self, payload: dict) -> str:
        """Enqueue a whole-bundle OKF ingest task. Returns job_id.

        Same payload contract as ``enqueue_ingest_file`` (the artifact is
        the bundle zip), but the worker walks it as ONE knowledge bundle.
        The deterministic id is namespaced apart from the per-file task
        (``ingest-okf:``) and keys on scope/project/visibility so the same
        bundle imported into a different pool is a distinct job rather
        than being coalesced onto (and dropped by) an earlier one.
        """
        partition = payload.get("user_id") or "ingest"
        options = payload.get("options") or {}
        content_key = (payload.get("source_ref") or {}).get("external_id") or payload.get(
            "stored_path"
        ) or payload.get("data_b64", "")
        job_id = _generate_job_id(
            f"ingest-okf:{options.get('visibility')}:{options.get('scope')}:"
            f"{options.get('project_id')}:"
            + (f"ws={options.get('workspace')}:" if options.get("workspace") else "")
            + f"{payload.get('filename', '')}:{content_key}",
            partition,
        )
        job = await self.pool.enqueue_job(
            "process_ingest_okf_bundle",
            payload,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )
        task_id = job_id if job is None else job.job_id
        await self._track_task(payload.get("user_id"), task_id)
        return task_id

    async def enqueue_code_index(self, payload: dict) -> str:
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
        job = await self.pool.enqueue_job(
            "process_code_index",
            payload,
            _queue_name=settings.ingest_queue_name,
        )
        task_id = job.job_id
        await self._track_task(payload.get("user_id"), task_id)
        return task_id

    async def enqueue_connector_sync(
        self,
        connector_id: str,
        user_id: str | None = None,
        *,
        generation: str | None = None,
        record: dict | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Enqueue a connector sync task. Returns job_id.

        ``user_id`` is the connector's owner (who asked for the sync and may
        poll it) and ``generation`` the vault record's instance identity. The
        job id is derived from the connector id, owner and generation
        (``connector_sync_job_id``), the same scheme ``connector_sync_cron`` in
        worker.py uses, so overlapping cron and API syncs of one connector
        instance coalesce on a single job instead of racing on the cursor,
        while a connector deleted and re-registered (or registered by another
        owner) under the same id gets a new job id: its syncs never share a
        cached job, result or read grant with the previous instance.
        """
        job_id = connector_sync_job_id(connector_id, user_id, generation)
        job = await enqueue_connector_sync_job(self.pool, connector_id, generation, job_id)
        task_id = job_id if job is None else job.job_id
        await self._track_task(user_id, task_id)
        return task_id

    async def enqueue_raw_batch(self, items: list[dict], *, idempotency_key: str | None = None) -> str:
        """Enqueue a batch of raw memory storage tasks (memory-model v2).

        Items are dispatched as a single ARQ job. Job ID is deterministic
        based on a canonical JSON encoding of the items so distinct batches
        cannot collide even when item content includes delimiter characters
        like ``|`` or quotes.
        """
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
        # Mixed-user batches are legal (shared key only) and count in every
        # writer's queue status. The result holds every item, so only a
        # single-user batch grants that user a per-user read of it; a
        # mixed-user batch stays readable only by shared-key callers.
        writers = sorted({i.get("user_id") for i in items if i.get("user_id")})
        for uid in writers:
            await self._track_task(uid, task_id, grant_read=len(writers) == 1)
        return task_id

    async def enqueue_retag(self, caller_user_id: str, filters: dict, ops: dict,
                            *, idempotency_key: str | None = None) -> str:
        """Enqueue a bulk retag task on the fast queue. Returns job_id.

        Canonical-JSON job id (sort_keys) so two *different* retags can never
        collide while an identical replay coalesces within ARQ's dedup window
        — the same guarantee enqueue_raw_batch relies on.
        """
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
        requester_user_id: str | None = None,
    ) -> str:
        """Enqueue a graph re-ingest for an edited memory (graph queue). Returns job_id.

        ``user_id`` is the memory's storage owner, which selects the graph
        partition the job writes to. ``requester_user_id`` is the
        authenticated user who made the edit and receives the task id; the
        task is tracked for, and readable by, that user (falls back to
        ``user_id`` when not given). On a shared memory the two differ when a
        teammate makes an allowed edit.

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
        await self._track_task(requester_user_id or user_id, task_id)
        return task_id

    async def enqueue_graph_jobs(self, jobs: list[dict]) -> dict:
        """Enqueue a list of graph re-ingest jobs (the ``graph_job`` shape).

        Used for the sibling re-enrichment a delete/edit produces. Never
        raises: a delete must succeed even when the queue is unavailable, so
        failures are returned instead — ``graph_jobs_enqueued`` and
        ``graph_jobs_not_enqueued_ids`` (memory ids whose re-ingest did NOT
        get queued, so the caller can report them). A connection-level
        failure short-circuits the remainder (no point timing out N times).
        """
        enqueued = 0
        not_enqueued: list[str] = []
        for i, job in enumerate(jobs or []):
            try:
                await self.enqueue_graph_enrichment(**job)
                enqueued += 1
            except Exception as e:
                logger.warning(
                    "Graph re-enrichment enqueue failed for %s (non-critical): %s",
                    job.get("memory_id"), e,
                )
                not_enqueued.append(job.get("memory_id"))
                if isinstance(e, OSError) or "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
                    not_enqueued.extend(j.get("memory_id") for j in jobs[i + 1:])
                    break
        return {"graph_jobs_enqueued": enqueued, "graph_jobs_not_enqueued_ids": not_enqueued}

    def _candidate_queues(self, *, include_graph: bool = False) -> list[str]:
        """Queues a poll-able job could live on, de-duplicated.

        Ingest/connector-sync jobs run on a dedicated queue, so status polling
        must look there too or ``/v1/memories/status/{id}`` would report every
        ingest job as not_found. With ``include_graph`` the graph queue is
        searched as well, so a graph task handed to a caller (for example the
        ``graph_task_id`` of a memory edit) reports queued while it waits
        instead of not_found. Only per-task status reads ask for it;
        ``get_queue_status`` accounting keeps its existing queue set.
        """
        queues = [settings.arq_queue_name, settings.ingest_queue_name]
        if include_graph:
            queues.append(settings.graph_queue_name)
        seen: set[str] = set()
        return [q for q in queues if not (q in seen or seen.add(q))]

    async def _find_job(
        self, task_id: str, *, include_graph: bool = False
    ) -> tuple[Job, JobStatus]:
        """Return the (job, status) for ``task_id``, searching candidate queues.

        Returns the first queue where the job is known; falls back to a
        main-queue job with not_found status if it's on none of them.
        """
        fallback: Job | None = None
        for queue_name in self._candidate_queues(include_graph=include_graph):
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

    async def get_status(
        self,
        task_id: str,
        user_id: str | None = None,
        *,
        enforce_owner: bool = False,
        include_graph: bool = False,
    ) -> dict:
        """Get task status from ARQ/Redis.

        With ``enforce_owner`` the task is reported ``not_found`` unless
        ``user_id`` enqueued it, before the job is looked up, so a caller
        cannot tell another user's task from one that never existed. If the
        owner check itself fails, ``TaskStatusUnavailable`` propagates (no
        job is looked up). ``include_graph`` also searches the graph queue,
        so a queued graph task reports its real status; the status routes
        ask for it, while ``get_queue_status`` keeps its existing queue set.

        Returns:
            Dict with task_id, status, result, error keys.
        """
        if enforce_owner and not await self.is_task_owner(task_id, user_id):
            return {"task_id": task_id, "status": "not_found", "result": None, "error": None}
        job, status = await self._find_job(task_id, include_graph=include_graph)

        api_status = _STATUS_MAP.get(status, "not_found")

        result = None
        error = None

        if status == JobStatus.complete:
            info = await job.result_info()
            if info is not None:
                if info.success:
                    result = info.result
                else:
                    api_status = "failed"
                    error = str(info.result) if info.result else "Unknown error"

        return {
            "task_id": task_id,
            "status": api_status,
            "result": result,
            "error": error,
        }

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

    async def wait_for_result(self, task_id: str, timeout: float = 300.0) -> dict:
        """Wait for a task to complete and return its status.

        Used by MCP tools with wait=true.
        """
        job, _ = await self._find_job(task_id)
        try:
            result = await job.result(timeout=timeout)
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "error": None,
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


async def enqueue_connector_sync_job(pool, connector_id: str, generation: str | None, job_id: str):
    """Enqueue a connector sync under ``job_id``, coalescing only with a run still in flight.

    Returns the new job, or ``None`` when the sync was coalesced onto a run that
    is queued, deferred or running. ARQ refuses an id whose result is still
    cached (an hour after it finished), so with a deterministic id a sync
    requested after the previous one completed would be silently dropped and a
    poller would read the previous run's result. For a finished run the cached
    result is cleared and a fresh job is enqueued; a client still polling the
    old run with the same id sees the new run's state instead of the old result.
    Shared by ``TaskManager.enqueue_connector_sync`` and ``connector_sync_cron``.
    """

    async def _enqueue():
        return await pool.enqueue_job(
            "process_connector_sync",
            connector_id,
            generation,
            _job_id=job_id,
            _queue_name=settings.ingest_queue_name,
        )

    job = await _enqueue()
    if job is not None:
        return job
    status = await Job(job_id, pool, _queue_name=settings.ingest_queue_name).status()
    if status != JobStatus.complete:
        return None  # queued / deferred / running: coalesce onto it
    await pool.delete(result_key_prefix + job_id)
    return await _enqueue()


def connector_sync_job_id(
    connector_id: str, owner_user_id: str | None, generation: str | None
) -> str:
    """ARQ job id for a sync of one connector instance.

    Shared by ``TaskManager.enqueue_connector_sync`` and the worker's
    ``connector_sync_cron`` so both coalesce on the same job. The owner and
    the vault record's ``generation`` are part of the id, so a connector
    deleted and re-registered, or registered by another owner, under the
    same ``connector_id`` never reuses the previous instance's job.
    """
    instance = hashlib.sha256(
        "\x1f".join([owner_user_id or "", generation or ""]).encode()
    ).hexdigest()[:16]
    return f"sync-{connector_id}-{instance}"


def _scope_key(**fields) -> str:
    """Canonical, order-independent text for the optional fields of a job key.

    Unset (None) fields are omitted so a request that leaves them out derives
    the same text whichever way the caller spelled "not given".
    """
    return json.dumps(
        {k: v for k, v in fields.items() if v is not None},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
