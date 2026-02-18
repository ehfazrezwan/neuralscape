"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import hashlib
import logging

from arq.cron import cron

from config import parse_redis_settings, settings
from memory_service import MemoryService

logger = logging.getLogger(__name__)


async def process_memory_store(
    ctx: dict,
    messages: list[dict],
    user_id: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Background task: LLM extraction + store each fact."""
    service: MemoryService = ctx["service"]
    memories = service.extract_and_store(
        messages=messages,
        user_id=user_id,
        project_id=project_id,
        agent_id=agent_id,
        run_id=run_id,
    )
    return {"memories": [m.model_dump(exclude_none=True) for m in memories]}


async def process_memory_raw(
    ctx: dict,
    content: str,
    user_id: str,
    category: str,
    scope: str = "global",
    project_id: str | None = None,
    tags: list[str] | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Background task: direct fact storage with idempotency check."""
    service: MemoryService = ctx["service"]

    # Idempotency: check if identical content already exists for this user
    try:
        existing = service.search(
            query=content,
            user_id=user_id,
            project_id=project_id,
            limit=3,
        )
        for mem in existing:
            if mem.memory.strip().lower() == content.strip().lower():
                logger.info(f"Skipping duplicate memory for user {user_id}: {content[:50]}...")
                return {"memories": [mem.model_dump(exclude_none=True)], "deduplicated": True}
    except Exception as e:
        logger.warning(f"Idempotency check failed (proceeding with store): {e}")

    memories = service.store_raw(
        content=content,
        user_id=user_id,
        category=category,
        scope=scope,
        project_id=project_id,
        tags=tags,
        agent_id=agent_id,
        run_id=run_id,
    )
    return {"memories": [m.model_dump(exclude_none=True) for m in memories]}


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"raw-{h}"


async def dedup_all_memories(ctx: dict) -> dict:
    """Cron job: deduplicate Qdrant memories for every user."""
    service: MemoryService = ctx["service"]
    user_ids = service.get_all_user_ids(batch_size=settings.dedup_batch_size)
    logger.info(f"Dedup cron: found {len(user_ids)} users")

    results = []
    for uid in user_ids:
        try:
            result = service.dedup_memories(uid)
            results.append(result)
            removed = result["exact_duplicates_removed"] + result["semantic_duplicates_removed"]
            if removed:
                logger.info(f"Dedup [{uid}]: removed {removed} duplicates")
        except Exception as e:
            logger.error(f"Dedup failed for user {uid}: {e}")
            results.append({"user_id": uid, "error": str(e)})

    total_exact = sum(r.get("exact_duplicates_removed", 0) for r in results)
    total_semantic = sum(r.get("semantic_duplicates_removed", 0) for r in results)
    summary = {
        "users_processed": len(user_ids),
        "total_exact_removed": total_exact,
        "total_semantic_removed": total_semantic,
        "per_user": results,
    }
    logger.info(f"Dedup cron complete: {total_exact} exact + {total_semantic} semantic removed")
    return summary


async def startup(ctx: dict) -> None:
    """Worker startup: initialize MemoryService + connections."""
    logger.info("ARQ worker starting up...")
    service = MemoryService()
    service._get_memory()  # warm up connections
    ctx["service"] = service
    logger.info("ARQ worker ready.")


async def shutdown(ctx: dict) -> None:
    """Worker shutdown: close connections."""
    logger.info("ARQ worker shutting down...")
    service: MemoryService = ctx.get("service")
    if service:
        service.close()
    logger.info("ARQ worker stopped.")


class WorkerSettings:
    functions = [process_memory_store, process_memory_raw]
    cron_jobs = [
        cron(
            dedup_all_memories,
            hour=settings.dedup_cron_hours,
            minute=0,
            timeout=1800,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_settings()
    queue_name = settings.arq_queue_name
    max_jobs = 10
    job_timeout = settings.arq_job_timeout
    max_tries = settings.arq_max_retries
