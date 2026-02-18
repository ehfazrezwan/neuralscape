"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import hashlib
import logging

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
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_settings()
    queue_name = settings.arq_queue_name
    max_jobs = 10
    job_timeout = settings.arq_job_timeout
    max_tries = settings.arq_max_retries
