"""Redis-backed task management via ARQ.

Centralizes task enqueuing and status tracking, replacing the in-memory _tasks dict.
"""

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


class TaskManager:
    """Redis-backed task status tracking + ARQ job enqueuing."""

    def __init__(self):
        self.pool: ArqRedis | None = None

    async def connect(self) -> None:
        """Initialize the ARQ Redis connection pool."""
        self.pool = await create_pool(
            parse_redis_settings(),
            default_queue_name=settings.arq_queue_name,
        )
        logger.info("TaskManager connected to Redis")

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
    ) -> str:
        """Enqueue memory extraction task. Returns task_id (job_id).

        Uses a deterministic job ID based on content hash to prevent
        duplicate enqueues of the same conversation.
        """
        # Generate deterministic job ID from message content
        content_str = "|".join(
            m.get("content", "") for m in messages
        )
        job_id = _generate_job_id(f"store:{content_str}", user_id)

        job = await self.pool.enqueue_job(
            "process_memory_store",
            messages,
            user_id,
            project_id,
            agent_id,
            run_id,
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
        # Provenance epistemics (A1)
        derived_from: list[str] | None = None,
        epistemic_level: str | None = None,
        # Multi-user model
        visibility: str | None = None,
        # Data-layer connectors
        memory_kind: str | None = None,
        source_ref: dict | None = None,
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
        """
        job_id = _generate_job_id(
            f"raw:{visibility}:{scope}:{project_id}:{content}", user_id
        )

        v2_extras = {
            "domain": domain,
            "observation_type": observation_type,
            "concepts": concepts,
            "source_type": source_type,
            "related_memory_ids": related_memory_ids,
            "confidence": confidence,
            "expires_at": expires_at,
            "derived_from": derived_from,
            "epistemic_level": epistemic_level,
            "visibility": visibility,
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

    async def enqueue_connector_sync(self, connector_id: str) -> str:
        """Enqueue a connector sync task. Returns job_id.

        Job id is keyed on the connector instance so overlapping syncs of the
        same connector coalesce rather than stacking. Must match the id scheme
        used by ``connector_sync_cron`` in worker.py (``sync-<connector_id>``)
        so a cron-triggered and an API-triggered sync of the same connector
        coalesce instead of racing on the cursor/revision state.
        """
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

    async def enqueue_raw_batch(self, items: list[dict]) -> str:
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
        # Mixed-user batches are legal — track for every distinct writer.
        for uid in {i.get("user_id") for i in items if i.get("user_id")}:
            await self._track_task(uid, task_id)
        return task_id

    async def enqueue_retag(self, caller_user_id: str, filters: dict, ops: dict) -> str:
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

    async def get_status(self, task_id: str) -> dict:
        """Get task status from ARQ/Redis.

        Returns:
            Dict with task_id, status, result, error keys.
        """
        job, status = await self._find_job(task_id)

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


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
