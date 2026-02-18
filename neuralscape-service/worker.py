"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import logging

from arq.connections import RedisSettings

from config import settings
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
    """Background task: direct fact storage."""
    service: MemoryService = ctx["service"]
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


def _parse_redis_settings() -> RedisSettings:
    """Parse redis_url into ARQ RedisSettings."""
    url = settings.redis_url
    # redis://host:port/db or redis://host:port
    url = url.replace("redis://", "")
    parts = url.split("/")
    host_port = parts[0]
    database = int(parts[1]) if len(parts) > 1 else 0

    if ":" in host_port:
        host, port = host_port.rsplit(":", 1)
        port = int(port)
    else:
        host = host_port
        port = 6379

    return RedisSettings(host=host, port=port, database=database)


class WorkerSettings:
    functions = [process_memory_store, process_memory_raw]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_settings()
    queue_name = settings.arq_queue_name
    max_jobs = 10
    job_timeout = settings.arq_job_timeout
    max_tries = settings.arq_max_retries
