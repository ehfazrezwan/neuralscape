"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import asyncio
import hashlib
import logging
from datetime import datetime

from arq.cron import cron

from config import parse_redis_settings, settings
from memory_service import MemoryService

logger = logging.getLogger(__name__)


def _rebuild_category_index(registry) -> None:
    """Rebuild the Obsidian category index via the conversation-compiler extension."""
    try:
        for ext in registry._extensions:
            writer = getattr(ext, "_writer", None)
            if writer is not None:
                writer.update_category_index()
                break
    except Exception:
        logger.warning("Failed to rebuild category index (non-critical)", exc_info=True)


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

    # Emit memory_stored events so extensions (e.g. conversation-compiler) can write to vault
    registry = ctx.get("extension_registry")
    if registry:
        for mem in memories:
            await registry.emit_event("memory_stored", {
                "user_id": user_id,
                "memory_id": getattr(mem, "id", ""),
                "content": mem.memory,
                "category": getattr(mem, "category", "") or "",
                "scope": getattr(mem, "scope", None),
                "project_id": project_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "source": "worker",
            })
        # Rebuild category index once after all vault writes (not per-fact)
        _rebuild_category_index(registry)

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
    v2_extras: dict | None = None,
) -> dict:
    """Background task: direct fact storage with idempotency check.

    Accepts memory-model v2 fields via the ``v2_extras`` kwargs dict (set by
    the task manager). Content-hash dedup happens inside ``store_raw``;
    the broader semantic-search idempotency check below catches near-dupes.
    """
    service: MemoryService = ctx["service"]
    v2_extras = v2_extras or {}

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

    # Re-hydrate expires_at if it came in as ISO string
    expires_at = v2_extras.get("expires_at")
    if expires_at and isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            expires_at = None

    memories = service.store_raw(
        content=content,
        user_id=user_id,
        category=category,
        scope=scope,
        project_id=project_id,
        tags=tags,
        agent_id=agent_id,
        run_id=run_id,
        domain=v2_extras.get("domain"),
        observation_type=v2_extras.get("observation_type"),
        concepts=v2_extras.get("concepts"),
        source_type=v2_extras.get("source_type"),
        related_memory_ids=v2_extras.get("related_memory_ids"),
        confidence=v2_extras.get("confidence"),
        expires_at=expires_at,
    )

    # Emit memory_stored event so extensions can write to vault
    registry = ctx.get("extension_registry")
    if registry:
        await registry.emit_event("memory_stored", {
            "user_id": user_id,
            "memory_id": memories[0].id if memories else "",
            "content": content,
            "category": category,
            "scope": scope,
            "project_id": project_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "source": "worker",
        })
        # Rebuild category index once after vault write
        _rebuild_category_index(registry)

    return {"memories": [m.model_dump(exclude_none=True) for m in memories]}


async def process_memory_raw_batch(ctx: dict, items: list[dict]) -> dict:
    """Background task: store a batch of pre-categorized facts (memory-model v2).

    Per-item dedup is handled inside ``store_raw`` (content-hash). One bad
    item does not block the others. The returned ``memories`` list is in the
    same order as the input ``items`` for successful stores; failed items
    are dropped, so we re-align by content for event emission rather than
    naively pairing by index.
    """
    service: MemoryService = ctx["service"]

    memories = await asyncio.to_thread(service.store_raw_batch, items)

    # Emit memory_stored events for each successful store so extensions can
    # write to vault. We attribute each event to the originating item's
    # user_id (mixed-user batches are supported), looked up by content match
    # rather than by index — store_raw_batch may skip items that failed
    # validation, so memories[i] does not necessarily correspond to items[i].
    registry = ctx.get("extension_registry")
    if registry:
        # Build a content -> user_id map from the input batch.
        item_user_by_content: dict[str, str] = {}
        for item in items:
            content = item.get("content")
            if content and content not in item_user_by_content:
                item_user_by_content[content] = item.get("user_id", "")
        for mem in memories:
            await registry.emit_event("memory_stored", {
                "user_id": item_user_by_content.get(mem.memory, ""),
                "memory_id": mem.id,
                "content": mem.memory,
                "category": mem.category or "",
                "scope": mem.scope,
                "project_id": mem.project_id,
                "source": "worker",
            })
        _rebuild_category_index(registry)

    return {"memories": [m.model_dump(exclude_none=True) for m in memories]}


async def expire_old_memories_cron(ctx: dict) -> dict:
    """Cron job: purge memories whose expires_at is in the past (memory-model v2)."""
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.expire_old_memories)
    if result.get("deleted_count"):
        logger.info(f"Expiry cron: purged {result['deleted_count']} memories")
    return result


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"raw-{h}"


async def process_conversation_flush(
    ctx: dict,
    user_message: str,
    assistant_response: str,
    session_id: str,
    channel: str,
    timestamp: str | None,
    project_id: str | None,
    user_id: str,
) -> dict:
    """Background task: extract facts from a conversation turn."""
    from extensions.conversation_compiler.flush import flush_conversation_turn
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    service: MemoryService = ctx["service"]
    writer: ObsidianWriter = ctx.get("compiler_writer") or ObsidianWriter()
    ctx.setdefault("compiler_writer", writer)

    result = await flush_conversation_turn(
        user_message=user_message,
        assistant_response=assistant_response,
        session_id=session_id,
        channel=channel,
        timestamp=timestamp,
        project_id=project_id,
        user_id=user_id,
        service=service,
        writer=writer,
    )
    return result.model_dump()


async def process_conversation_compile(
    ctx: dict,
    date: str | None,
    user_id: str,
) -> dict:
    """Background task: compile daily logs into structured articles."""
    from extensions.conversation_compiler.compile import compile_all_pending, compile_date
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    service: MemoryService = ctx["service"]
    writer: ObsidianWriter = ctx.get("compiler_writer") or ObsidianWriter()
    ctx.setdefault("compiler_writer", writer)

    if date:
        result = await compile_date(date, service, writer, user_id=user_id)
        return result.model_dump()
    else:
        results = await compile_all_pending(service, writer, user_id=user_id)
        return {"dates_compiled": len(results), "results": [r.model_dump() for r in results]}


async def auto_compile_check(ctx: dict) -> dict | None:
    """Cron job: check if auto-compilation should run after COMPILE_AFTER_HOUR."""
    from extensions.conversation_compiler.compile import compile_all_pending
    from extensions.conversation_compiler.config import compiler_settings
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    if not compiler_settings.auto_compile:
        return None

    now = datetime.now()
    if now.hour < compiler_settings.compile_after_hour:
        return None

    service: MemoryService = ctx["service"]
    writer: ObsidianWriter = ctx.get("compiler_writer") or ObsidianWriter()
    ctx.setdefault("compiler_writer", writer)

    results = await compile_all_pending(service, writer, user_id=settings.default_user_id)
    if results:
        writer.append_log(f"Auto-compiled {len(results)} date(s) via cron")
        logger.info(f"Auto-compile cron: compiled {len(results)} dates")
        return {"dates_compiled": len(results)}
    return None


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
    """Worker startup: initialize MemoryService + extension registry."""
    logger.info("ARQ worker starting up...")
    service = MemoryService()
    service._get_memory()  # warm up connections
    ctx["service"] = service

    # Initialize extension registry so worker can emit events
    from extensions import ExtensionRegistry

    registry = ExtensionRegistry()
    await registry.discover()
    await registry.startup_all()
    ctx["extension_registry"] = registry
    logger.info("ARQ worker ready.")


async def shutdown(ctx: dict) -> None:
    """Worker shutdown: close connections."""
    logger.info("ARQ worker shutting down...")
    registry = ctx.get("extension_registry")
    if registry:
        await registry.shutdown_all()
    service: MemoryService = ctx.get("service")
    if service:
        service.close()
    logger.info("ARQ worker stopped.")


class WorkerSettings:
    functions = [
        process_memory_store,
        process_memory_raw,
        process_memory_raw_batch,
        process_conversation_flush,
        process_conversation_compile,
    ]
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
        cron(
            auto_compile_check,
            hour={18, 19, 20, 21, 22, 23},
            minute=30,
            timeout=600,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            expire_old_memories_cron,
            hour={3},
            minute=15,
            timeout=600,
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
