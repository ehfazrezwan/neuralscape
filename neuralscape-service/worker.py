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
                "visibility": getattr(mem, "visibility", None),
                "owner_user_id": getattr(mem, "owner_user_id", None),
                "created_at": getattr(mem, "created_at", None),
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
        visibility=v2_extras.get("visibility"),
        memory_kind=v2_extras.get("memory_kind"),
        source_ref=v2_extras.get("source_ref"),
    )

    # Emit memory_stored event so extensions can write to vault
    registry = ctx.get("extension_registry")
    if registry:
        first = memories[0] if memories else None
        await registry.emit_event("memory_stored", {
            "user_id": user_id,
            "memory_id": first.id if first else "",
            "content": content,
            "category": category,
            "scope": scope,
            "visibility": getattr(first, "visibility", None) if first else None,
            "owner_user_id": getattr(first, "owner_user_id", None) if first else None,
            "created_at": getattr(first, "created_at", None) if first else None,
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
                "visibility": getattr(mem, "visibility", None),
                "owner_user_id": getattr(mem, "owner_user_id", None),
                "created_at": getattr(mem, "created_at", None),
                "project_id": mem.project_id,
                "source": "worker",
            })
        _rebuild_category_index(registry)

    return {"memories": [m.model_dump(exclude_none=True) for m in memories]}


async def process_ingest_document(ctx: dict, doc: dict) -> dict:
    """Background task: ingest a document into passages + distilled facts.

    ``doc`` is the serialized :class:`ingest.pipeline.IngestDoc` field set
    (content, source descriptor, user_id, category, scope, options). Re-ingest
    is idempotent via content-hash dedup in ``store_raw``.
    """
    from ingest.pipeline import IngestDoc, ingest_document

    service: MemoryService = ctx["service"]
    ingest_doc = IngestDoc(
        content=doc["content"],
        source=doc["source"],
        user_id=doc["user_id"],
        category=doc.get("category", "domain_knowledge"),
        scope=doc.get("scope", "global"),
        project_id=doc.get("project_id"),
        visibility=doc.get("visibility"),
        tags=doc.get("tags"),
        agent_id=doc.get("agent_id"),
        run_id=doc.get("run_id"),
        extract_facts=doc.get("extract_facts", True),
        index_passages=doc.get("index_passages", True),
        max_chars=doc.get("max_chars"),
        overlap=doc.get("overlap"),
    )
    result = await asyncio.to_thread(ingest_document, service, ingest_doc)

    # Emit a single summary event so extensions know an ingest landed. We don't
    # emit one event per passage — a large doc would flood the vault writer.
    registry = ctx.get("extension_registry")
    if registry and result.get("memory_ids"):
        await registry.emit_event("memory_stored", {
            "user_id": doc["user_id"],
            "memory_id": result["memory_ids"][0],
            "content": f"[ingest] {result['passages']} passages + {result['facts']} facts",
            "category": doc.get("category", "domain_knowledge"),
            "scope": doc.get("scope", "global"),
            "project_id": doc.get("project_id"),
            "source": "ingest",
        })

    return result


async def expire_old_memories_cron(ctx: dict) -> dict:
    """Cron job: purge memories whose expires_at is in the past (memory-model v2)."""
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.expire_old_memories)
    if result.get("deleted_count"):
        logger.info(f"Expiry cron: purged {result['deleted_count']} memories")
    return result


async def synthesize_topical_wikis_cron(ctx: dict) -> dict:
    """Cron job: synthesize shared memories into topical wiki pages.

    Gated by ``WIKI_SYNTHESIZER_ENABLED``; the synthesizer itself returns
    an empty result when disabled, so this wrapper just forwards.
    """
    from extensions.wiki_synthesizer.config import synthesizer_settings
    from extensions.wiki_synthesizer.synthesizer import synthesize_all

    if not synthesizer_settings.enabled:
        return {"skipped": True, "reason": "WIKI_SYNTHESIZER_ENABLED=false"}
    service: MemoryService = ctx["service"]
    result = await synthesize_all(service=service, settings=synthesizer_settings)
    return {
        "pages_created": result.pages_created,
        "pages_updated": result.pages_updated,
        "memories_processed": result.memories_processed,
        "errors": result.errors,
    }


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"raw-{h}"


def _synthesizer_cron_hours() -> list[int]:
    """Resolve the hours the wiki-synthesizer cron fires.

    ``WIKI_SYNTHESIZER_CRON_HOURS`` is an interval in hours (default 6).
    We translate it into the discrete hour-of-day set that arq's `cron`
    accepts. Starts at 03:45 + offset to stagger with the dedup
    (hour=dedup_cron_hours) and the expire (hour={3}) crons.
    """
    from extensions.wiki_synthesizer.config import synthesizer_settings

    interval = max(1, min(24, synthesizer_settings.cron_hours))
    return list(range(0, 24, interval))


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


async def process_connector_sync(ctx: dict, connector_id: str) -> dict:
    """Background task: pull a connector's resources and ingest changed ones.

    Incremental: resources whose source-side revision matches the last-seen
    revision are skipped; content-hash dedup in ``store_raw`` is the backstop
    for sources without a revision marker. Sync state (cursor + per-resource
    revisions + timestamp) is persisted back to the vault.
    """
    from datetime import timezone

    from connectors.registry import build_adapter
    from ingest.pipeline import IngestDoc, ingest_document

    vault = ctx.get("vault")
    if vault is None:
        return {"skipped": True, "reason": "connectors disabled (no vault)"}
    record = await vault.get(connector_id)
    if not record:
        return {"skipped": True, "reason": "connector not found"}
    if not record.get("enabled", True):
        return {"skipped": True, "reason": "connector disabled"}

    service: MemoryService = ctx["service"]
    owner = record.get("owner_user_id") or settings.default_user_id
    adapter = build_adapter(record)
    revisions = dict(record.get("last_revision_by_id") or {})
    cursor = record.get("cursor")
    next_cursor = cursor
    synced = skipped = 0

    try:
        resources, next_cursor = await adapter.list_resources(cursor)
        for r in resources:
            if r.revision and revisions.get(r.external_id) == r.revision:
                skipped += 1
                continue
            try:
                content = await adapter.fetch(r)
            except Exception as e:
                logger.warning(f"Connector {connector_id}: fetch failed for {r.external_id}: {e}")
                continue
            if not content or not content.strip():
                continue
            doc = IngestDoc(
                content=content,
                source=adapter.source_descriptor(r),
                user_id=owner,
                category=record.get("default_category", "domain_knowledge"),
                scope=record.get("default_scope", "global"),
                project_id=record.get("default_project_id"),
                visibility=record.get("default_visibility"),
            )
            await asyncio.to_thread(ingest_document, service, doc)
            if r.revision:
                revisions[r.external_id] = r.revision
            synced += 1
    finally:
        await adapter.aclose()

    await vault.update_sync_state(
        connector_id,
        cursor=next_cursor,
        last_synced_at=datetime.now(timezone.utc).isoformat(),
        revisions=revisions,
    )
    logger.info(f"Connector sync [{connector_id}]: {synced} ingested, {skipped} unchanged")
    return {"connector_id": connector_id, "synced": synced, "skipped": skipped}


async def connector_sync_cron(ctx: dict) -> dict:
    """Cron job: enqueue a sync for every enabled connector."""
    if not settings.connectors_enabled:
        return {"skipped": True, "reason": "CONNECTORS_ENABLED=false"}
    vault = ctx.get("vault")
    if vault is None:
        return {"skipped": True, "reason": "no vault"}
    redis = ctx["redis"]  # ARQ injects the ArqRedis pool into the job context
    enqueued = 0
    for rec in await vault.list():
        if not rec.get("enabled", True):
            continue
        await redis.enqueue_job(
            "process_connector_sync",
            rec["connector_id"],
            _job_id=f"sync-{rec['connector_id']}",
        )
        enqueued += 1
    logger.info(f"Connector sync cron: enqueued {enqueued} connector(s)")
    return {"enqueued": enqueued}


def _connector_sync_cron_hours() -> list[int]:
    """Translate the connector sync interval (hours) into a set of hours-of-day."""
    interval = max(1, min(24, settings.connector_sync_cron_hours))
    return list(range(0, 24, interval))


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

    # Connector vault (only when connectors are enabled + a vault key is set).
    if settings.connectors_enabled:
        try:
            from connectors.vault import ConnectorVault

            ctx["vault"] = ConnectorVault.from_settings(settings)
            logger.info("Connector vault initialized")
        except Exception:
            logger.warning(
                "Connector vault init failed; connector sync disabled this run",
                exc_info=True,
            )

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
        process_ingest_document,
        process_connector_sync,
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
        cron(
            synthesize_topical_wikis_cron,
            # Imported lazily so importing worker.py doesn't load the
            # synthesizer's whole settings tree before WorkerSettings is
            # actually instantiated. Cadence is read directly from the env
            # var here because cron() needs a concrete value at class
            # body evaluation time.
            hour=set(_synthesizer_cron_hours()),
            minute=45,
            timeout=1800,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            connector_sync_cron,
            hour=set(_connector_sync_cron_hours()),
            minute=50,
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
