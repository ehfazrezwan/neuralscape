"""Redis-backed task management via ARQ.

Centralizes task enqueuing and status tracking, replacing the in-memory _tasks dict.
"""

import hashlib
import json
import logging

from arq.connections import ArqRedis, create_pool
from arq.jobs import Job, JobStatus

from config import parse_redis_settings, settings

logger = logging.getLogger(__name__)

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
        if job is None:
            # Duplicate job ID — ARQ already has this job. Return the ID
            # so the caller can poll the existing job's status.
            return job_id
        return job.job_id

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
        # Multi-user model
        visibility: str | None = None,
    ) -> str:
        """Enqueue raw memory storage task. Returns task_id (job_id).

        Uses a deterministic job ID based on content hash to prevent
        duplicate enqueues of the same fact. v2 + multi-user fields are
        forwarded as a kwargs dict to the worker so the signature stays
        backward-compatible.
        """
        job_id = _generate_job_id(f"raw:{content}", user_id)

        v2_extras = {
            "domain": domain,
            "observation_type": observation_type,
            "concepts": concepts,
            "source_type": source_type,
            "related_memory_ids": related_memory_ids,
            "confidence": confidence,
            "expires_at": expires_at,
            "visibility": visibility,
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
        if job is None:
            return job_id
        return job.job_id

    async def enqueue_ingest_document(self, doc: dict) -> str:
        """Enqueue a document-ingest task (chunk → passages + facts). Returns job_id.

        Deterministic job id from the content hash + connector instance so the
        same document re-submitted by a re-sync coalesces onto one job.
        """
        content = doc.get("content", "")
        connector_id = (doc.get("source") or {}).get("connector_id", "")
        partition = doc.get("user_id") or "ingest"
        job_id = _generate_job_id(f"ingest:{connector_id}:{content}", partition)

        job = await self.pool.enqueue_job(
            "process_ingest_document",
            doc,
            _job_id=job_id,
        )
        if job is None:
            return job_id
        return job.job_id

    async def enqueue_connector_sync(self, connector_id: str) -> str:
        """Enqueue a connector sync task. Returns job_id.

        Job id is keyed on the connector instance so overlapping syncs of the
        same connector coalesce rather than stacking.
        """
        job_id = _generate_job_id(f"sync:{connector_id}", "connector")
        job = await self.pool.enqueue_job(
            "process_connector_sync",
            connector_id,
            _job_id=job_id,
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
        if job is None:
            return job_id
        return job.job_id

    async def get_status(self, task_id: str) -> dict:
        """Get task status from ARQ/Redis.

        Returns:
            Dict with task_id, status, result, error keys.
        """
        job = Job(task_id, redis=self.pool, _queue_name=settings.arq_queue_name)
        status = await job.status()

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
        job = Job(task_id, redis=self.pool, _queue_name=settings.arq_queue_name)
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

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self.pool:
            await self.pool.aclose()
            logger.info("TaskManager disconnected from Redis")


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
