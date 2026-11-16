"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import third_party_telemetry  # noqa: F401 — must precede any mem0/Graphiti import (defaults their telemetry off)

import asyncio
import hashlib
import logging
from datetime import datetime

import structlog
from arq.cron import cron
from arq.jobs import Job
from arq.worker import Retry

import adapters  # noqa: F401 — registers knowledge-adapter taxonomies at import (deterministic MEMORY_CATEGORIES across all workers)
from config import parse_redis_settings, settings
from memory_service import MemoryService
from task_manager import connector_sync_job_id, enqueue_connector_sync_job

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


async def _note_session_messages(
    ctx: dict, user_id: str, session_id: str | None, messages: list[dict]
) -> None:
    """E3: buffer a conversation write's messages for its session and enqueue
    summary-slot refreshes when the message count crosses a threshold.

    The trigger runs inline with the write, but the refresh job itself is
    enqueued onto the GRAPH queue (audit 27 #24): a summary refresh is a
    multi-second Gemini call with no latency SLO, and running it on the fast
    queue let bursts of summaries compete with latency-sensitive writes for
    the 10 fast-worker slots. The refresh job id is deterministic per
    (session, slot, threshold bucket) so a burst of writes crossing the same
    threshold coalesces onto one job. Best-effort — session bookkeeping never
    fails a store.
    """
    if not session_id or not settings.session_summary_enabled:
        return
    import session_summarizer as ss

    try:
        count, due = await asyncio.to_thread(
            ss.record_messages, user_id, session_id, messages
        )
        for slot in due:
            bucket = count // ss.slot_interval(slot)
            await ctx["redis"].enqueue_job(
                "process_session_summary",
                user_id,
                session_id,
                slot,
                _job_id=f"sess-{ss._safe(user_id)}-{ss._safe(session_id)}-{slot}-{bucket}",
                _queue_name=settings.graph_queue_name,
            )
    except Exception:  # noqa: BLE001 — summarization is best-effort
        logger.warning("session summary trigger failed (non-fatal)", exc_info=True)


async def process_session_summary(
    ctx: dict, user_id: str, session_id: str, slot: str
) -> dict:
    """Background task (GRAPH queue — audit 27 #24): refresh one session summary slot.

    Recursive compression — prior slot text + only the messages since that
    slot's last refresh. The slot is REPLACED in Redis (TTL'd); it is never
    stored as a memory row (see session_summarizer.py for the decision).
    """
    from extensions.conversation_compiler.compile import _async_call_gemini
    from session_summarizer import refresh_slot

    return await refresh_slot(user_id, session_id, slot, llm_call=_async_call_gemini)


async def process_memory_store(
    ctx: dict,
    messages: list[dict],
    user_id: str,
    project_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    occurred_at: str | None = None,
) -> dict:
    """Background task: LLM extraction + store each fact.

    ``occurred_at`` (event time, ISO 8601) rides along for historical
    ingestion — when the conversation actually happened, stamped on every
    extracted fact. None (the default, and every pre-existing queued job)
    means event time unknown.
    """
    service: MemoryService = ctx["service"]
    # E3: conversation writes feed the per-session summary slots (run_id is
    # the session identifier on this path).
    await _note_session_messages(ctx, user_id, run_id, messages)
    # Offload the synchronous, network-bound work (LLM extraction + embedding +
    # Qdrant insert) to a thread so it doesn't block the ARQ event loop and
    # serialize other concurrent jobs. (process_memory_raw_batch already does this.)
    # return_stats (audit 27 #22): long conversations extract in windows, and a
    # partially-failed extraction (some windows down, others stored) must be
    # visible in the task result rather than masquerading as a full success.
    memories, extraction_stats = await asyncio.to_thread(
        service.extract_and_store,
        messages=messages,
        user_id=user_id,
        project_id=project_id,
        agent_id=agent_id,
        run_id=run_id,
        occurred_at=occurred_at,
        return_stats=True,
    )
    created_ids = set(extraction_stats.get("created_ids") or [])

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
                # Conversation-extracted rows: the vault mirror decides by the
                # category's tier (not the private stamp) unless the sensitivity
                # gate forced them private. See extensions/conversation_compiler.
                # Only rows THIS extraction created count as extracted; a
                # dedup survivor is a pre-existing row (possibly an explicit
                # private write) and keeps its own stored visibility.
                "extracted": getattr(mem, "id", "") in created_ids,
                "sensitivity": getattr(mem, "sensitivity", None),
                "source": "worker",
            })
        # Rebuild category index once after all vault writes (not per-fact)
        _rebuild_category_index(registry)

    result: dict = {"memories": [m.model_dump(exclude_none=True) for m in memories]}
    if extraction_stats.get("windows_failed"):
        # Partial extraction: some windows' LLM calls failed but the rest
        # were stored. Total failure (ALL windows) raises inside
        # extract_and_store and fails the job — this branch is honest
        # partial-success reporting for pollers of the task status.
        result["partial_extraction"] = True
        result["windows_total"] = extraction_stats.get("windows_total")
        result["windows_failed"] = extraction_stats.get("windows_failed")
        result["window_errors"] = extraction_stats.get("window_errors")
    return result


_STALE_CHECK_COALESCE_TTL_S = 300


async def _enqueue_stale_graph_check(
    ctx: dict, mem, content: str, user_id: str, project_id: str | None
) -> None:
    """Dedup hit on an existing row: queue a cheap graph-staleness check.

    A content-hash/semantic dedup hit used to return without ever touching the
    graph, so a fact Graphiti had since invalidated (row live, edges dead)
    stayed contradicted even though the user just re-asserted it. The check
    runs on the GRAPH queue (never the fast path): one Cypher probe, and an
    LLM enrichment only when the row has no episode ref or all its episode's
    edges are dead. Deterministic job id per (memory_id, content hash) so a
    burst of repeated hits coalesces onto one job. Best-effort.
    """
    from memory.hashing import content_hash

    # Part of the flagged supersession feature: off => dedup hits enqueue
    # nothing (pre-existing behaviour).
    if not settings.graph_supersession_enabled:
        return
    # Vector-only rows never reach the graph by design: nothing to check.
    if getattr(mem, "memory_kind", None) == "passage" or getattr(mem, "workspace", None) not in (None, "memory"):
        return
    redis = ctx["redis"]
    # Coalescing: a short-lived NX marker per (memory, content hash) so a burst
    # of repeated hits queues ONE check even after arq has dropped the finished
    # job's result (the _job_id below only dedups while the result is retained).
    # Fail-open: any Redis error just means we may queue a duplicate (the job
    # itself is idempotent and cheap).
    try:
        marker = f"graph-stale-check-lock:{mem.id}:{content_hash(content)}"
        if not await redis.set(marker, "1", nx=True, ex=_STALE_CHECK_COALESCE_TTL_S):
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        await ctx["redis"].enqueue_job(
            "process_graph_enrichment",
            mem.id,
            content,
            user_id,
            project_id,
            getattr(mem, "visibility", None),
            None,
            None,
            True,
            _queue_name=settings.graph_queue_name,
            _job_id=f"graph-stale-check:{mem.id}:{content_hash(content)}",
        )
    except Exception as e:  # noqa: BLE001 — the check is best-effort
        logger.warning(f"Stale-graph check enqueue failed for {getattr(mem, 'id', '?')}: {e}")


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

    # Resolve the visibility this write will ACTUALLY land at, so the
    # idempotency check below is visibility-aware. Without this, a dictator
    # promoting existing private/shared text to a `standard` would match the
    # old copy and be dropped before ever reaching store_raw (which is the
    # tier-aware create path).
    #
    # Must use the RESOLVED (post sensitivity-gate) visibility, not the
    # caller-requested one (F7): store_raw applies the sensitivity gate
    # internally and can force visibility=private for a `shared`-requested
    # write. If this pre-check compared against the requested visibility
    # instead, a gated write with no override could match a pre-existing
    # SHARED row with the same content and get treated as a dedup hit —
    # returning that shared row and skipping the write store_raw's gate
    # would have forced private, silently leaving the content shared.
    # resolve_gated_visibility (memory/sensitivity.py) is the single source
    # of truth store_raw itself uses, so this check can never drift from
    # what store_raw actually decides.
    from memory.sensitivity import resolve_gated_visibility
    _eff_req_vis, _, _, _ = resolve_gated_visibility(
        content, category, v2_extras.get("visibility"),
        v2_extras.get("sensitivity_override", False),
    )

    # Idempotency: skip only when identical content already exists AT THE SAME
    # visibility tier (a different-tier write is a distinct memory).
    # vector_only (audit 27 #12): this is an internal near-dupe probe — it
    # must not pay the graph pass + edge enrichment of a full hybrid recall,
    # and must not pollute the dreaming recall traces.
    # The probe searches the memory workspace only, so a write into a reference
    # workspace skips it (store_raw's own dedup key includes the workspace);
    # otherwise identical text in the memory workspace would swallow the write.
    _workspace = v2_extras.get("workspace")
    try:
        # Off the loop: retry_transient may time.sleep for tens of seconds.
        existing = [] if _workspace not in (None, "memory") else await asyncio.to_thread(
            service.search,
            query=content,
            user_id=user_id,
            project_id=project_id,
            limit=3,
            vector_only=True,
        )
        for mem in existing:
            if (
                mem.memory.strip().lower() == content.strip().lower()
                and getattr(mem, "visibility", None) == _eff_req_vis
            ):
                logger.info(f"Skipping duplicate memory for user {user_id}: {content[:50]}...")
                await _enqueue_stale_graph_check(ctx, mem, content, user_id, project_id)
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

    # Fast path: vector write only (sub-second). The slow Graphiti graph.add
    # is deferred to a separate process_graph_enrichment job (see below) so it
    # can't make this job block for minutes and starve other writes. Run it in a
    # thread: the embedding + Qdrant insert + history write are synchronous and
    # network-bound, so calling them directly would block the ARQ event loop and
    # serialize concurrent jobs (process_memory_raw_batch already offloads).
    memories, created = await asyncio.to_thread(
        service.store_raw,
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
        occurred_at=v2_extras.get("occurred_at"),
        derived_from=v2_extras.get("derived_from"),
        epistemic_level=v2_extras.get("epistemic_level"),
        visibility=v2_extras.get("visibility"),
        sensitivity_override=v2_extras.get("sensitivity_override", False),
        memory_kind=v2_extras.get("memory_kind"),
        source_ref=v2_extras.get("source_ref"),
        workspace=v2_extras.get("workspace"),
        add_to_graph=False,
        return_created=True,
    )

    # Defer graph enrichment for newly-created rows onto the graph queue. Skip
    # for content-hash dedup hits (created=False) — they're already in the graph.
    # Exception (GRAPH_SUPERSESSION_ENABLED): a dedup hit that REVIVED a
    # tombstoned row is a re-assertion of that fact, so it re-enters the graph
    # (and Graphiti's contradiction engine) like a fresh write.
    revived = bool(
        settings.graph_supersession_enabled
        and memories
        and getattr(memories[0], "revived", None)
    )
    if memories and not created and not revived:
        # Plain dedup hit: not re-enriched, but check whether the graph has
        # since gone stale for this row (flagged: GRAPH_SUPERSESSION_ENABLED).
        await _enqueue_stale_graph_check(ctx, memories[0], content, user_id, project_id)
    # Reference-workspace rows are vector-only everywhere else (no graph
    # episode, no extraction LLM), so a raw write into one is not graph-enriched.
    if (created or revived) and memories and _workspace in (None, "memory"):
        mem = memories[0]
        try:
            await ctx["redis"].enqueue_job(
                "process_graph_enrichment",
                mem.id,
                content,
                user_id,
                project_id,
                getattr(mem, "visibility", None),
                _queue_name=settings.graph_queue_name,
            )
        except Exception as e:
            # The enqueue is this memory's ONLY shot at graph enrichment: a
            # later retry/dedup returns created=False and skips this block, so
            # a lost enqueue means the graph is never updated. Fall back to
            # enriching inline (in a thread) so graph state still lands when the
            # graph queue is down/misconfigured — slower, but not silently lost.
            logger.warning(f"Graph enqueue failed for {mem.id}; enriching inline as fallback: {e}")
            try:
                await asyncio.to_thread(
                    service.enrich_graph,
                    content=content,
                    user_id=user_id,
                    project_id=project_id,
                    visibility=getattr(mem, "visibility", None) or "private",
                    memory_id=mem.id,
                )
            except Exception as e2:  # noqa: BLE001 — enrichment is best-effort
                logger.warning(f"Inline graph-enrichment fallback also failed for {mem.id}: {e2}")

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

    dedup_ids: set[str] = set()
    memories = await asyncio.to_thread(service.store_raw_batch, items, dedup_ids=dedup_ids)

    # Dedup survivors weren't re-enriched: queue the cheap stale-graph check
    # for each (same semantics as process_memory_raw's dedup hits).
    if dedup_ids:
        item_by_content = {i.get("content"): i for i in items if i.get("content")}
        for mem in memories:
            if mem.id in dedup_ids:
                src = item_by_content.get(mem.memory) or {}
                await _enqueue_stale_graph_check(
                    ctx, mem, mem.memory, src.get("user_id", ""), src.get("project_id")
                )

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


async def _enqueue_graph_jobs(ctx: dict, jobs: list[dict], adapter: str | None) -> int:
    """Enqueue deferred graph-enrichment jobs produced by ``ingest_document``.

    One ``process_graph_enrichment`` job per newly-created fact, onto the graph
    queue, carrying the knowledge-adapter *name* (the worker re-resolves the
    ontology). Mirrors process_memory_raw's fallback: if an enqueue fails, the
    fact is enriched inline (in a thread) rather than silently never reaching
    the graph — a dedup'd re-ingest returns created=False and would never
    re-produce the job. Returns the number successfully enqueued.
    """
    service: MemoryService = ctx["service"]
    enqueued = 0
    for job in jobs:
        try:
            await ctx["redis"].enqueue_job(
                "process_graph_enrichment",
                job["memory_id"],
                job["content"],
                job["user_id"],
                job.get("project_id"),
                job.get("visibility"),
                job.get("source_ref"),
                adapter,
                _queue_name=settings.graph_queue_name,
            )
            enqueued += 1
        except Exception as e:
            logger.warning(
                f"Graph enqueue failed for ingested fact {job['memory_id']}; "
                f"enriching inline as fallback: {e}"
            )
            try:
                graph_ontology = None
                if adapter:
                    # Strict (audit 27 #36): an unregistered adapter fails this
                    # fallback loudly instead of enriching under no ontology.
                    from adapters import require_adapter

                    graph_ontology = require_adapter(adapter).graph_ontology_kwargs()
                await asyncio.to_thread(
                    service.enrich_graph,
                    content=job["content"],
                    user_id=job["user_id"],
                    project_id=job.get("project_id"),
                    visibility=job.get("visibility") or "private",
                    memory_id=job["memory_id"],
                    source_ref=job.get("source_ref"),
                    graph_ontology=graph_ontology,
                )
            except Exception as e2:  # noqa: BLE001 — enrichment is best-effort
                logger.warning(
                    f"Inline graph-enrichment fallback also failed for {job['memory_id']}: {e2}"
                )
    return enqueued


async def process_memory_retag(
    ctx: dict,
    caller_user_id: str,
    filters: dict,
    ops: dict,
) -> dict:
    """Background task: bulk-retag memories matching a filter set.

    Runs on the fast queue — the retag itself is a Qdrant scroll +
    per-point set_payload (milliseconds each). When a project change moves
    memories between graph group_id partitions, the service returns one
    graph job per moved memory; those fan out onto the graph queue via
    ``_enqueue_graph_jobs`` (with its inline-enrichment fallback) so the
    slow Graphiti re-ingest never runs here.
    """
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(
        service.retag_memories, caller_user_id, filters, ops
    )
    graph_jobs = result.pop("graph_jobs", [])
    result["graph_jobs_enqueued"] = (
        await _enqueue_graph_jobs(ctx, graph_jobs, None) if graph_jobs else 0
    )
    logger.info(
        "Retag complete: matched=%s updated=%s forbidden=%s invalid=%s graph_jobs=%s",
        result.get("matched"), result.get("updated"),
        result.get("skipped_forbidden"), result.get("skipped_invalid"),
        result["graph_jobs_enqueued"],
    )
    return result


async def process_ingest_document(ctx: dict, doc: dict) -> dict:
    """Background task: ingest a document into passages + distilled facts.

    ``doc`` is the serialized :class:`ingest.pipeline.IngestDoc` field set
    (content, source descriptor, user_id, category, scope, options). Re-ingest
    is idempotent via content-hash dedup in ``store_raw``. Fact graph enrichment
    is deferred onto the graph queue (see ``_enqueue_graph_jobs``).
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
        sensitivity_override=doc.get("sensitivity_override", False),
        tags=doc.get("tags"),
        agent_id=doc.get("agent_id"),
        run_id=doc.get("run_id"),
        extract_facts=doc.get("extract_facts", True),
        index_passages=doc.get("index_passages", True),
        max_chars=doc.get("max_chars"),
        overlap=doc.get("overlap"),
        adapter=doc.get("adapter", "default"),
        occurred_at=doc.get("occurred_at"),
    )
    result = await asyncio.to_thread(ingest_document, service, ingest_doc)

    # Defer fact graph enrichment onto the graph queue (adapter name rides along
    # so the graph worker re-resolves the ontology). The raw job payloads are
    # popped — clients polling the task result get a count, not full contents.
    graph_jobs = result.pop("graph_jobs", [])
    result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(
        ctx, graph_jobs, adapter=result.get("adapter")
    )

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


async def _ingest_exemplars(
    ctx: dict, service, images: list[dict], filename: str, options: dict, *, user_id: str
) -> list[dict]:
    """Index a trading-book file's chart images as visual exemplars.

    ``images`` are the figures already harvested by the caller's (single)
    Docling conversion — ``{"bytes", "ext", "page_ref"}`` each. Every image is
    stored, vision-described (multimodal, via the gateway), and written as a
    setup/visual_exemplar memory + VisualExemplar graph node (enrichment
    deferred onto the graph queue, same as ingested facts). Re-ingested images
    are skipped via the image-hash dedup lookup — the vision description is
    nondeterministic, so the memory body's content hash can't be relied on for
    idempotency. Best-effort.
    """
    from adapters.trading.exemplars import find_existing_exemplar, ingest_exemplar

    if not images:
        return []

    # Strategy name for grouping comes from a `strategy:<name>` upload tag.
    strategy_name = None
    for tag in options.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("strategy:"):
            strategy_name = tag[len("strategy:"):].strip() or None
            break

    summaries: list[dict] = []
    graph_jobs: list[dict] = []
    for img in images:
        try:
            existing = await asyncio.to_thread(
                find_existing_exemplar, service, image_bytes=img["bytes"], user_id=user_id
            )
            if existing:
                summaries.append({"memory_ids": [existing], "deduped": True})
                continue
            summary = await asyncio.to_thread(
                ingest_exemplar,
                service,
                image_bytes=img["bytes"],
                ext=img["ext"],
                settings=settings,
                strategy_name=strategy_name,
                page_ref=img.get("page_ref"),
                user_id=user_id,
                project_id=options.get("project_id"),
                scope=options.get("scope", "global"),
                visibility=options.get("visibility"),
                add_to_graph=False,
            )
            job = summary.pop("graph_job", None)
            if job:
                graph_jobs.append(job)
            summaries.append(summary)
        except Exception:  # noqa: BLE001
            logger.warning("Exemplar ingest failed for one image in %s", filename, exc_info=True)
    if graph_jobs:
        await _enqueue_graph_jobs(ctx, graph_jobs, adapter="trading_strategy")
    return summaries


async def process_ingest_file(ctx: dict, payload: dict) -> dict:
    """Background task: parse an uploaded file's bytes → text → passages + facts.

    ``payload`` is ``{filename, source_ref, options, user_id, ...}`` plus EITHER
    ``stored_path`` (the artifact's path on the shared volume — preferred) or
    ``data_b64`` (inline bytes, used only when artifact storage is disabled).
    Parsing (Docling / MarkItDown) and chunking both run here on the dedicated
    ingest worker, so a large PDF or a whole folder never blocks the fast queue.
    A file that no parser can handle is reported as skipped rather than failing
    the whole job.
    """
    import base64

    from ingest.extract import UnsupportedFile, extract_text, extract_text_and_images
    from ingest.pipeline import IngestDoc, ingest_document
    from ingest.storage import read_artifact

    service: MemoryService = ctx["service"]
    filename = payload.get("filename", "upload")
    if payload.get("stored_path"):
        data = await asyncio.to_thread(read_artifact, payload["stored_path"], settings)
    else:
        data = base64.b64decode(payload["data_b64"])
    options = payload.get("options", {})

    # ── Graphify bundle members (graph.json / GRAPH_REPORT.md) ──
    # A graphify-out/ upload (zip-expanded or as loose files) is detected by
    # its canonical filenames. graph.json is distilled into its STABLE
    # semantic layer only (never mirrored raw — it churns per commit and stays
    # queryable live via the query_code_graph tools); GRAPH_REPORT.md is
    # upgraded onto the code_graph adapter's section chunker + report
    # extractor and then flows through the normal text pipeline below.
    from adapters.code_graph import code_graph_available
    from ingest.code_graph import detect_graphify_member, ingest_code_graph_json

    graphify_kind = detect_graphify_member(filename, data)
    if graphify_kind == "graph":
        if not code_graph_available():
            # Graceful degradation (no crash): the optional graphifyy library
            # isn't installed, and ingesting a graph.json as prose would only
            # produce megabytes of JSON noise — skip with an honest reason.
            reason = (
                "graph.json detected but the code-graph extra is not installed "
                "(uv sync --extra code-graph); skipping instead of ingesting raw JSON"
            )
            logger.warning("Ingest file skipped (%s): %s", filename, reason)
            return {"filename": filename, "skipped": True, "reason": reason}
        result = await asyncio.to_thread(ingest_code_graph_json, service, data, payload)
        graph_jobs = result.pop("graph_jobs", [])
        result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(
            ctx, graph_jobs, adapter=result.get("adapter")
        )
        registry = ctx.get("extension_registry")
        if registry and result.get("memory_ids"):
            await registry.emit_event("memory_stored", {
                "user_id": payload["user_id"],
                "memory_id": result["memory_ids"][0],
                "content": f"[ingest:code_graph] {filename} → {result['facts']} semantic facts",
                "category": options.get("category", "domain_knowledge"),
                "scope": options.get("scope", "global"),
                "project_id": options.get("project_id"),
                "source": "ingest",
            })
        return {"filename": filename, "doc_type": "code_graph", **result}
    if graphify_kind == "report" and code_graph_available():
        # Only upgrade from the default adapter — an explicit adapter choice
        # (e.g. a deliberate re-ingest under another taxonomy) wins.
        if options.get("adapter", "default") == "default":
            options = {**options, "adapter": "code_graph"}

    # When visual exemplars apply, one Docling conversion yields BOTH the text
    # and the embedded figures — a book PDF takes minutes per parse, so the
    # text and image paths must not each convert separately.
    want_exemplars = (
        options.get("adapter") == "trading_strategy"
        and settings.exemplar_store_enabled
        and settings.docling_extract_images
    )
    images: list[dict] = []
    try:
        if want_exemplars:
            text, doc_type, images = await asyncio.to_thread(
                extract_text_and_images,
                filename,
                data,
                settings,
                options.get("page_offset") or 0,
            )
        else:
            text, doc_type = await asyncio.to_thread(extract_text, filename, data, settings)
    except UnsupportedFile as e:
        logger.warning("Ingest file skipped (%s): %s", filename, e)
        return {"filename": filename, "skipped": True, "reason": str(e)}

    ingest_doc = IngestDoc(
        content=text,
        source=payload["source_ref"],
        user_id=payload["user_id"],
        category=options.get("category", "domain_knowledge"),
        scope=options.get("scope", "global"),
        project_id=options.get("project_id"),
        visibility=options.get("visibility"),
        sensitivity_override=options.get("sensitivity_override", False),
        tags=options.get("tags"),
        agent_id=options.get("agent_id"),
        run_id=options.get("run_id"),
        extract_facts=options.get("extract_facts", True),
        index_passages=options.get("index_passages", True),
        adapter=options.get("adapter", "default"),
    )
    result = await asyncio.to_thread(ingest_document, service, ingest_doc)

    # Defer fact graph enrichment onto the graph queue (see _enqueue_graph_jobs).
    graph_jobs = result.pop("graph_jobs", [])
    result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(
        ctx, graph_jobs, adapter=result.get("adapter")
    )

    # ── Visual setup exemplars (trading adapter only) ──
    # Vision-describe the figures harvested above and index each as a
    # setup/visual_exemplar memory + VisualExemplar graph node. Best-effort —
    # never fails the text ingest.
    if want_exemplars and images:
        try:
            exemplars = await _ingest_exemplars(
                ctx, service, images, filename, options, user_id=payload["user_id"]
            )
            if exemplars:
                result["exemplars"] = exemplars
        except Exception:  # noqa: BLE001
            logger.warning("Exemplar ingestion failed for %s (non-fatal)", filename, exc_info=True)

    registry = ctx.get("extension_registry")
    if registry and result.get("memory_ids"):
        await registry.emit_event("memory_stored", {
            "user_id": payload["user_id"],
            "memory_id": result["memory_ids"][0],
            "content": f"[ingest:{doc_type}] {filename} → "
                       f"{result['passages']} passages + {result['facts']} facts",
            "category": options.get("category", "domain_knowledge"),
            "scope": options.get("scope", "global"),
            "project_id": options.get("project_id"),
            "source": "ingest",
        })

    return {"filename": filename, "doc_type": doc_type, **result}


async def process_ingest_okf_bundle(ctx: dict, payload: dict) -> dict:
    """Background task: walk + ingest an uploaded OKF bundle zip.

    ``payload`` mirrors ``process_ingest_file`` ({filename, source_ref,
    options, user_id} plus ``stored_path`` or ``data_b64``), but the zip is
    treated as ONE knowledge bundle: concepts are parsed with their
    frontmatter, types map to categories (heuristics + a single batched
    LLM fallback), cross-links become graph relationship hints, and every
    memory's source_ref carries {bundle URI/path, concept ID}.
    """
    import base64

    from ingest.okf_bundle import (
        default_type_llm,
        ingest_okf_bundle,
        load_bundle_zip,
    )
    from ingest.storage import read_artifact

    service: MemoryService = ctx["service"]
    filename = payload.get("filename", "bundle.zip")
    if payload.get("stored_path"):
        data = await asyncio.to_thread(read_artifact, payload["stored_path"], settings)
    else:
        data = base64.b64decode(payload["data_b64"])
    options = payload.get("options", {})

    files = await asyncio.to_thread(
        load_bundle_zip,
        data,
        max_file_bytes=settings.ingest_max_file_mb * 1024 * 1024,
        max_files=settings.ingest_max_files,
        max_total_uncompressed_bytes=settings.ingest_max_archive_uncompressed_mb * 1024 * 1024,
    )
    if not files:
        return {"filename": filename, "skipped": True, "reason": "no markdown members in bundle"}

    source_ref = payload.get("source_ref") or {}
    bundle_uri = source_ref.get("url") or payload.get("stored_path") or filename
    result = await asyncio.to_thread(
        ingest_okf_bundle,
        service,
        files=files,
        bundle_uri=bundle_uri,
        # The artifact download endpoint (API-relative) — preserved on every
        # concept's source_ref.url so the bundle stays re-fetchable.
        bundle_url=source_ref.get("url"),
        user_id=payload["user_id"],
        scope=options.get("scope", "global"),
        project_id=options.get("project_id"),
        visibility=options.get("visibility"),
        tags=options.get("tags"),
        extract_facts=options.get("extract_facts", True),
        index_passages=options.get("index_passages", True),
        llm_call=default_type_llm(service),
    )

    graph_jobs = result.pop("graph_jobs", [])
    result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(ctx, graph_jobs, adapter=None)

    if result.get("status") == "failed":
        # Every concept with text failed: that is a failed import, not a
        # success with zero counts. Fail the job so the task status says so.
        first = (result.get("failed_concepts") or [{}])[0]
        raise RuntimeError(
            f"OKF bundle ingest failed: all {result.get('failed', 0)} concepts failed "
            f"(first: {first.get('concept_id')}: {first.get('error')})"
        )
    if result.get("status") == "partial":
        logger.warning(
            "OKF bundle %s imported partially: %d ingested, %d failed (%s)",
            filename, result.get("concepts", 0), result.get("failed", 0),
            [c["concept_id"] for c in result.get("failed_concepts", [])][:10],
        )

    registry = ctx.get("extension_registry")
    if registry and result.get("memory_ids"):
        await registry.emit_event("memory_stored", {
            "user_id": payload["user_id"],
            "memory_id": result["memory_ids"][0],
            "content": f"[ingest:okf] {filename} → {result['concepts']} concepts, "
                       f"{result['passages']} passages + {result['facts']} facts",
            "category": options.get("category", "domain_knowledge"),
            "scope": options.get("scope", "global"),
            "project_id": options.get("project_id"),
            "source": "ingest",
        })

    return {"filename": filename, **result}


async def process_code_index(ctx: dict, payload: dict) -> dict:
    """Phase G: index a repo INTO a code knowledge system through NS.

    ``payload`` = ``{system, repo_source, code_space, project_id, user_id}``.

    Runs on the ingest queue (PLAN §5: uniform observability + 202-task
    semantics; CBM's ~1.3s doesn't need the queue but the rail/observability do).
    Steps:
      1. Resolve the real per-code_space engine for ``system`` and call
         ``engine.index(repo_source)``.
      2. Record ``{repo_sha, indexed_at, engine_version, symbols, edges}`` per
         code_space and set the project routing config (``code_systems`` /
         ``default_engine``) so later recalls route to the indexed engine.
      3. Run the external-engine liveness diff → the existing dreaming consumer
         (``detect_inventory_diff_liveness``): real events for engines that can
         enumerate symbols (graphify/native), honest graceful N/A for CBM.

    Idempotent: re-indexing the same code_space overwrites metadata + reindexes
    the backend. The through-NS index always runs a FULL build
    (``engine.index(incremental=False)``) for every engine, so a re-index is a
    clean deterministic rebuild rather than a partial incremental.
    """
    import time

    from adapters.code_graph import code_graph_available
    from knowledge.code_dispatch import resolve_code_engine

    service: MemoryService = ctx["service"]
    system = payload["system"]
    repo_source = payload["repo_source"]
    code_space = payload["code_space"]
    project_id = payload.get("project_id")
    user_id = payload.get("user_id") or settings.default_user_id

    if not code_graph_available():
        return {"ok": False, "error": "code-graph extra not installed", "code_space": code_space}

    # Operator-side only: the source must sit inside a CODE_REPOS directory, for
    # every engine, and the caller must own the target code_space (or be admin).
    from knowledge.code_dispatch import (
        code_space_access_error,
        code_space_shape_error,
        repo_source_error,
    )

    denied = code_space_shape_error(code_space) or code_space_access_error(code_space, user_id)
    if denied:
        return {"ok": False, "error": denied, "code_space": code_space, "system": system}
    refused = repo_source_error(repo_source, settings)
    if refused:
        return {"ok": False, "error": refused, "code_space": code_space, "system": system}
    # Index the path that was checked (resolved once), not the caller's string:
    # closes the window where a symlink is swapped after the check.
    import os as _os

    repo_source = _os.path.realpath(_os.path.expanduser(str(repo_source)))

    engine = resolve_code_engine(system, code_space, user_id, settings, for_index=True)
    if engine is None:
        from knowledge.code_dispatch import code_repos_error

        # Operator-side by design: name CODE_REPOS when the repo is not on the server.
        reason = code_repos_error(code_space, settings) or "the engine's backend (e.g. the CBM bridge) is unreachable"
        return {
            "ok": False,
            "error": f"could not resolve engine for system={system!r} code_space={code_space!r}: {reason}",
            "code_space": code_space,
            "system": system,
        }

    # 1. Index.
    report = await asyncio.to_thread(engine.index, repo_source, incremental=False)
    symbols = getattr(report, "symbols_indexed", 0)
    edges = getattr(report, "edges_indexed", 0)
    files = getattr(report, "files_indexed", 0)
    duration_s = getattr(report, "duration_s", 0.0)
    engine_version = getattr(report, "system_version", None)

    # 2. Record metadata + set routing config.
    repo_sha = await asyncio.to_thread(_git_head_sha, repo_source)
    indexed_at = time.time()
    from knowledge import index_store
    meta = {
        "system": system,
        "repo_source": repo_source,
        "repo_sha": repo_sha,
        "indexed_at": indexed_at,
        "engine_version": engine_version,
        "symbols": symbols,
        "edges": edges,
        # CBM binds its project slug from the bridge on read; stash it if known.
        "cbm_project": getattr(engine, "project", None),
    }
    await asyncio.to_thread(index_store.record_index, code_space, meta)
    config_updated, config_reason = False, "no project_id: no project config to update"
    if project_id:
        config_updated, config_reason = await asyncio.to_thread(
            _merge_index_into_project_config, project_id, system, code_space
        )

    # 3. External-engine liveness diff → dreaming consumer (best-effort).
    liveness = {"events": [], "flagged": 0, "summary": "skipped"}
    try:
        from extensions.dreaming.liveness import detect_inventory_diff_liveness

        liveness = await asyncio.to_thread(
            detect_inventory_diff_liveness, service, code_space, engine
        )
    except Exception as e:  # noqa: BLE001 — liveness is best-effort, never fails an index
        logger.warning("Inventory-diff liveness failed for %s (non-fatal): %s", code_space, e)
        liveness = {"events": [], "flagged": 0, "summary": f"liveness error: {e}"}

    logger.info(
        "Through-NS index complete: system=%s code_space=%s symbols=%d edges=%d in %.2fs",
        system, code_space, symbols, edges, duration_s,
    )
    return {
        "ok": True,
        "code_space": code_space,
        "system": system,
        "symbols_indexed": symbols,
        "edges_indexed": edges,
        "files_indexed": files,
        "duration_s": duration_s,
        "engine_version": engine_version,
        "repo_sha": repo_sha,
        "indexed_at": indexed_at,
        "liveness": {"flagged": liveness.get("flagged", 0), "summary": liveness.get("summary", "")},
        "config_updated": config_updated,
        "config_reason": config_reason,
    }


def _merge_index_config_dict(raw: dict | None, project_id: str, system: str, code_space: str) -> dict:
    """The project config after an index of ``system`` into ``code_space`` (pure).

    Adds ``system`` to ``code_systems`` and keeps every admin-set field. A pinned
    ``default_engine`` is kept; ``code_space`` follows the effective default engine
    (the pinned one, else the system just indexed), so indexing a native space for a
    project pinned to another engine does not point that engine at the native space.
    """
    raw = dict(raw or {})
    systems = list(raw.get("code_systems") or [])
    if system not in systems:
        systems.append(system)
    effective_default = raw.get("default_engine") or system
    code_space_out = code_space if system == effective_default else raw.get("code_space")
    return {
        "project_id": project_id,
        "code_systems": systems,
        "fuse_code_into_recall": bool(raw.get("fuse_code_into_recall", True)),
        "default_engine": effective_default,
        "code_space": code_space_out,
        "op_preference": raw.get("op_preference"),
    }


def _merge_index_into_project_config(project_id: str, system: str, code_space: str) -> tuple[bool, str]:
    """Record an index in the project's knowledge config WITHOUT replacing the config.

    Read-modify-write under Redis WATCH/MULTI on the config key, so a concurrent
    admin edit is retried over, never lost. The stored copy is read from Redis, never
    from the in-process cache: with Redis unavailable the write is skipped (and
    logged) rather than saving a possibly stale cached copy over the real one.
    Returns ``(updated, reason)``: ``reason`` is empty on success, else why the config
    was left alone.
    """
    import json as _json
    import time as _time

    from knowledge import index_store
    from knowledge.router import _config_from_raw, _mirror_put

    client = index_store._redis()
    if client is None:
        reason = "Redis unavailable: not writing from a possibly stale cache"
        logger.warning("project config for %s not updated after indexing %s: %s", project_id, code_space, reason)
        return False, reason
    key = index_store._CFG_PREFIX + project_id
    try:
        import redis as _redis_mod

        watch_error = _redis_mod.WatchError
    except Exception:  # noqa: BLE001
        watch_error = ()
    for _ in range(10):
        try:
            with client.pipeline() as pipe:
                pipe.watch(key)
                blob = pipe.get(key)
                raw = _json.loads(blob) if blob else None
                merged = _merge_index_config_dict(raw, project_id, system, code_space)
                pipe.multi()
                pipe.set(key, _json.dumps(merged, default=str))
                pipe.execute()
        except watch_error:
            continue  # someone changed the config between read and write: redo on the new value
        except Exception:  # noqa: BLE001
            logger.warning("project config for %s not updated (Redis error)", project_id, exc_info=True)
            return False, "Redis error while writing the project config"
        # Refresh this process's caches from what was just stored.
        index_store._mem_cfg[project_id] = dict(merged)
        index_store._mem_cfg_at[project_id] = _time.monotonic()
        _mirror_put(_config_from_raw(project_id, merged))
        return True, ""
    reason = "too many concurrent edits to the project config"
    logger.warning("project config for %s not updated: %s", project_id, reason)
    return False, reason


def _git_head_sha(repo_path: str) -> str | None:
    """Best-effort ``git rev-parse HEAD`` for staleness stamping (None if git-less)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _row_episode_uuid(service, memory_id: str) -> str | None:
    """The graph episode uuid stamped on the row's payload (None when unknown / row gone / not a real store)."""
    try:
        point = service._memory.vector_store.get(vector_id=memory_id)
        meta = ((getattr(point, "payload", None) or {}).get("metadata")) or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        uuid = meta.get("graph_episode_uuid")
        return uuid if isinstance(uuid, str) and uuid else None
    except Exception:  # noqa: BLE001 - best effort; the content match is the fallback
        return None


async def process_graph_enrichment(
    ctx: dict,
    memory_id: str,
    content: str,
    user_id: str,
    project_id: str | None = None,
    visibility: str | None = None,
    source_ref: dict | None = None,
    adapter: str | None = None,
    only_if_stale: bool = False,
) -> dict:
    """Background task: add a stored memory's content to the knowledge graph.

    ``only_if_stale=True`` (dedup-hit re-assertions): enrich only when the row
    has no graph episode ref or every edge of its episode is dead; otherwise
    return ``skipped: graph_current`` with no LLM call.

    This is the slow half of a write (Graphiti entity extraction, ~minutes,
    Gemini-gated) split out of the synchronous vector store so it runs on the
    dedicated graph queue/worker and never blocks fast writes or reads.
    Best-effort — MemoryService.enrich_graph swallows its own errors and
    returns whether the graph write actually succeeded. We surface that honestly
    in the result (``enriched``) and log dropped enrichments so a silent run of
    transient Gemini 503s (which leave memories vector-only) is observable rather
    than masquerading as success.

    ``source_ref`` (connector provenance) rides along so the ``(:Source)`` node /
    ``DERIVED_FROM`` back-reference is attached even on the deferred path.
    ``adapter`` is a knowledge-adapter *name* — the ontology itself isn't
    queue-serializable (Pydantic classes, tuple-keyed maps), so the worker
    re-resolves it here via ``require_adapter``; an unregistered name FAILS the
    job with a clear error (audit 27 #36) rather than silently enriching
    without the adapter's ontology, so the worker image must register the same
    adapter set as the API/ingest processes that enqueue these jobs.
    """
    service: MemoryService = ctx["service"]
    # Guard against a delete/expiry that happened while this job sat in the
    # queue: graph enrichment can run minutes after the write, and adding now
    # would resurrect the deleted memory's content in Neo4j. If it's gone from
    # the store, skip rather than re-create graph state for it.
    mem = await asyncio.to_thread(service.get_memory, memory_id)
    if mem is None:
        logger.info(
            "Skipping graph enrichment for memory %s — no longer in the store "
            "(deleted/expired while the job was queued).",
            memory_id,
        )
        return {"memory_id": memory_id, "enriched": False, "skipped": "memory_missing"}
    # Real event time for Graphiti bi-temporal dating: get_memory() surfaces
    # occurred_at as a top-level MemoryResponse field (mapped from the payload
    # metadata in _mem_to_response), so read it directly.
    occurred_at = getattr(mem, "occurred_at", None)

    # The job carries the visibility/project/content the memory had WHEN IT WAS
    # ENQUEUED. A narrowing edit (shared -> private), a project move or a content
    # rewrite can land while the job sits in the queue; publishing the queued
    # values would put the memory into a graph group its owner has since
    # withdrawn it from. The row just re-read above is authoritative: derive the
    # group (visibility + project + workspace) from it, and skip rather than
    # enrich text the memory no longer holds (the edit that changed the content
    # enqueues its own re-ingest of the new text).
    current_content = getattr(mem, "memory", None)
    if isinstance(current_content, str) and " ".join(current_content.split()) != " ".join(content.split()):
        logger.info(
            "Skipping graph enrichment for memory %s — its content changed while the job "
            "was queued (the edit enqueues its own re-ingest).", memory_id,
        )
        return {"memory_id": memory_id, "enriched": False, "skipped": "content_changed"}
    current_visibility = getattr(mem, "visibility", None) or "private"  # legacy null == private
    current_project = getattr(mem, "project_id", project_id) or None
    current_workspace = getattr(mem, "workspace", None) or None
    if (current_visibility, current_project) != ((visibility or "private"), (project_id or None)):
        logger.info(
            "Graph enrichment for memory %s: row changed while queued "
            "(visibility %r -> %r, project %r -> %r); using the current values.",
            memory_id, visibility, current_visibility, project_id, current_project,
        )
    if only_if_stale:
        # Re-enrichment must reproduce the original write: carry the row's
        # connector provenance. Reference-workspace / adapter rows (their
        # ontology is not recoverable from the row) and passages are
        # vector-only by design and never re-enriched here.
        if getattr(mem, "memory_kind", None) == "passage" or getattr(mem, "workspace", None) not in (None, "memory"):
            return {"memory_id": memory_id, "enriched": False, "skipped": "vector_only"}
        source_ref = source_ref or getattr(mem, "source_ref", None)
        stale = await asyncio.to_thread(
            service.graph_row_is_stale,
            memory_id,
            user_id,
            current_project,
            current_visibility,
            getattr(mem, "workspace", None),
        )
        if not stale:
            return {"memory_id": memory_id, "enriched": False, "skipped": "graph_current"}
    graph_ontology = None
    if adapter:
        # Strict resolution (audit 27 #36): a queued job carrying an adapter
        # name that isn't registered in this worker FAILS with a clear error
        # instead of silently enriching without the adapter's ontology.
        from adapters import require_adapter

        graph_ontology = require_adapter(adapter).graph_ontology_kwargs()
    episode_before = await asyncio.to_thread(_row_episode_uuid, service, memory_id)
    enrich_kwargs: dict = {}
    race_outcome: dict | None = None
    if getattr(settings, "graph_delete_race_cleanup_enabled", False) is True:
        # enrich_graph removes the episode itself when the row was deleted mid-write and says so here.
        race_outcome = {}
        enrich_kwargs["outcome"] = race_outcome
    # enrich_graph reports the episode it wrote through this out-parameter so
    # stale-job expiry can pin it directly (no before/after diff of the row ref).
    graph_written: dict = {}
    enriched = await asyncio.to_thread(
        service.enrich_graph,
        content=content,
        user_id=user_id,
        project_id=current_project,
        visibility=current_visibility,
        memory_id=memory_id,
        source_ref=source_ref,
        graph_ontology=graph_ontology,
        occurred_at=occurred_at,
        workspace=current_workspace,
        result_out=graph_written,
        **enrich_kwargs,
    )
    if enriched and race_outcome and race_outcome.get("cleaned_up"):
        logger.warning(
            "Memory %s was deleted while its graph enrichment ran; the episode just written was "
            "already removed by the enrichment.", memory_id,
        )
        return {"memory_id": memory_id, "enriched": False, "skipped": "changed_during_enrichment"}
    episode_after = await asyncio.to_thread(_row_episode_uuid, service, memory_id) if enriched else None
    if enriched:
        # The first check above only covers the time the job sat in the queue.
        # Extraction itself takes minutes: re-read the row now and, if it was
        # deleted or moved (visibility/project/workspace) or rewritten while we
        # were writing, expire the episode we just published — the edit/delete
        # cascade ran BEFORE it existed, so nothing else would ever expire it.
        expire = getattr(service, "_cascade_or_fallback_expire", None)
        after = await asyncio.to_thread(service.get_memory, memory_id)
        if after is None:
            changed = True
        else:
            text_after = getattr(after, "memory", None)
            changed = (
                (getattr(after, "visibility", None) or "private") != current_visibility
                or (getattr(after, "project_id", current_project) or None) != current_project
                or (getattr(after, "workspace", None) or None) != current_workspace
                or (isinstance(text_after, str) and " ".join(text_after.split()) != " ".join(content.split()))
            )
        if changed and callable(expire):
            from memory.groups import _build_group_id

            written_group = _build_group_id(current_visibility, user_id, current_project, current_workspace)
            logger.warning(
                "Memory %s changed or was deleted while its graph enrichment ran; "
                "expiring the episode just written to group %r.", memory_id, written_group,
            )
            # Pin the episode THIS job wrote when it can be told apart (the row's
            # episode ref changed across our enrich_graph call): the cascade then
            # resolves by uuid first, so a newer re-ingested episode of the same
            # memory is never the one expired. Otherwise exact-content match.
            written_uuid = graph_written.get("episode_uuid")
            ours = (
                written_uuid if isinstance(written_uuid, str) and written_uuid
                else episode_after if episode_after and episode_after != episode_before
                else None
            )
            resolved = False
            for attempt, pause in enumerate((0, 1.0, 3.0), start=1):
                if pause:
                    await asyncio.sleep(pause)
                result = await asyncio.to_thread(
                    expire,
                    {"memory": content, "metadata": {"graph_episode_uuid": ours} if ours else {}, "user_id": user_id},
                    group_id=written_group,
                    memory_id=memory_id,
                )
                resolved = bool(isinstance(result, dict) and result.get("resolved"))
                if resolved:
                    break
                logger.warning(
                    "Cleanup of the stale episode for memory %s (group %r) unresolved on attempt %d",
                    memory_id, written_group, attempt,
                )
            if not resolved:
                # Never swallow it: a shared episode of a now-private/deleted memory must not stay readable
                # unnoticed. Fail the job visibly with the identity needed to retry the cleanup.
                raise RuntimeError(
                    f"graph cleanup unresolved for memory {memory_id}: group_id={written_group!r} "
                    f"episode_uuid={ours!r}; re-run the episode cascade for this group/episode"
                )
            return {"memory_id": memory_id, "enriched": False, "skipped": "changed_during_enrichment"}
    if not enriched:
        logger.warning(
            "Graph enrichment did not complete for memory %s — it remains "
            "vector-only (graph unconfigured or a transient LLM/Graphiti error "
            "was swallowed). See enrich_graph warnings above.",
            memory_id,
        )
    return {"memory_id": memory_id, "enriched": enriched}


async def process_graph_reenrich(ctx: dict, memory_id: str) -> dict:
    """Background task: re-enrich one memory into a fresh episode after its shared episode was removed.

    Enqueued by the support-aware removal (``GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED``) for every memory that
    recorded the removed episode, when no ledger producer is installed.  The job carries only the memory id and
    re-reads the owner, project and visibility from the row; a memory deleted meanwhile is skipped, and an
    enrichment that does not complete raises so ARQ retries the job (the row keeps ``graph_reenrich_pending``
    until it has run).
    """
    service: MemoryService = ctx["service"]
    try:
        return await asyncio.to_thread(service.reenrich_memory_graph, memory_id)
    except Exception as exc:
        # ARQ re-runs a job only when it raises ``Retry``: any other exception marks it failed for good and the
        # row would stay pending until the sweep (``graph_reenrich_sweep_cron``) or a later write found it.
        tries = int(ctx.get("job_try") or 1)
        logger.warning(
            "Graph re-enrichment of memory %s failed (try %d); retrying in %ds: %s",
            memory_id, tries, _REENRICH_RETRY_BASE_S * tries, exc,
        )
        raise Retry(defer=_REENRICH_RETRY_BASE_S * tries) from exc


_REENRICH_RETRY_BASE_S = 30  # seconds of back-off per try for a failed survivor re-enrichment


async def graph_reenrich_sweep_cron(ctx: dict) -> dict:
    """Cron job: re-run the re-enrichment of rows left marked ``graph_reenrich_pending``.

    The marker survives a lost dispatch (queue or ledger down) and a job that used up its tries; this is the
    retry.  Bounded per run; the enrichment clears the marker itself.  A no-op unless
    ``GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED`` is on.
    """
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.sweep_pending_reenrichments)
    if result.get("reenriched") or result.get("failed"):
        logger.info(f"Re-enrichment sweep cron: {result}")
    return result


_EXPIRY_LOG_ID_CAP = 20


def _log_expiry_graph_problems(result: dict) -> None:
    """Warn about expired rows whose graph cleanup or sibling repair did not complete.

    The expired rows are already gone, so a re-run never retries them: the log
    line (counts + a capped id list) is the operator's trace besides the job result.
    """
    unresolved = result.get("graph_cascade_unresolved_ids") or []
    unrepaired = result.get("graph_sibling_unrepaired_ids") or []
    if unresolved:
        logger.warning(
            f"Expiry cron: graph cleanup unresolved for {len(unresolved)} expired memories "
            f"(first {_EXPIRY_LOG_ID_CAP}: {unresolved[:_EXPIRY_LOG_ID_CAP]})"
        )
    if unrepaired:
        logger.warning(
            f"Expiry cron: {len(unrepaired)} sibling memories could not be repaired "
            f"(first {_EXPIRY_LOG_ID_CAP}: {unrepaired[:_EXPIRY_LOG_ID_CAP]})"
        )
    if result.get("graph_sibling_repair_failed"):
        logger.warning("Expiry cron: sibling lookup failed; siblings of expired memories may have stale graph refs")


async def expire_old_memories_cron(ctx: dict) -> dict:
    """Cron job: purge memories whose expires_at is in the past (memory-model v2)."""
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.expire_old_memories)
    # Sibling facts that shared an expired row's conversation episode need a
    # single-fact graph re-ingest (graph queue), like a bulk delete.
    sibling_jobs = result.pop("graph_jobs", [])
    if sibling_jobs:
        result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(ctx, sibling_jobs, None)
    _log_expiry_graph_problems(result)
    if result.get("deleted_count"):
        logger.info(f"Expiry cron: purged {result['deleted_count']} memories")
    return result


async def repair_stale_vectors_cron(ctx: dict) -> dict:
    """Cron job: re-embed rows whose vector write did not finish after an edit (``vector_pending``).

    An edit commits text + revision in one compare-and-set and then writes the vector; a crash or a
    stalled writer leaves the marker, and this converges the vector to the stored text.
    """
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.repair_stale_vectors)
    if result.get("scanned") or result.get("meta_repaired"):
        logger.info(f"Edit repair cron: {result}")
    return result


async def run_dream_sweep(
    ctx: dict,
    pool: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Background task: one dreaming sweep, enqueued by the admin endpoint.

    The API endpoint returns 202 and enqueues here instead of sweeping
    in-process: a sweep is minutes of LLM + store work, and running it on
    the API's event loop starves /health (autoheal then restarts the
    container mid-sweep — observed live 2026-07-03).
    """
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all

    service: MemoryService = ctx["service"]
    run = await dream_all(
        service=service,
        settings=dreaming_settings,
        dry_run=dry_run,
        only_pool=pool,
        force=force,
    )
    return run.to_dict()


async def dream_sweep_cron(ctx: dict) -> dict:
    """Cron job: run one dreaming sweep (light → deep → REM per pool).

    Gated by ``DREAMING_ENABLED``; per-pool time/volume gates inside the
    sweep keep quiet pools cheap (one Redis read + at most one scroll).
    Replaces the wiki_synthesizer cron — see docs/DREAMING_MODE_SPEC.md.
    """
    from extensions.dreaming.config import dreaming_settings
    from extensions.dreaming.sweep import dream_all

    if not dreaming_settings.enabled:
        return {"skipped": True, "reason": "DREAMING_ENABLED=false"}
    service: MemoryService = ctx["service"]
    run = await dream_all(service=service, settings=dreaming_settings)
    return run.to_dict()["totals"] | {"run_id": run.run_id}


async def synthesize_strategy_playbooks_cron(ctx: dict) -> dict:
    """Cron job: merge trading rule memories into canonical strategy playbooks.

    Gated by ``STRATEGY_SYNTHESIZER_ENABLED``; the synthesizer returns an empty
    result when disabled, so this wrapper just forwards. Runs on the graph
    worker (slow queue), staggered against the wiki-synth cron.
    """
    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings
    from extensions.strategy_synthesizer.synthesizer import synthesize_all

    if not strategy_synthesizer_settings.enabled:
        return {"skipped": True, "reason": "STRATEGY_SYNTHESIZER_ENABLED=false"}
    service: MemoryService = ctx["service"]
    result = await synthesize_all(service=service, settings=strategy_synthesizer_settings)
    return {
        "playbooks_created": result.playbooks_created,
        "playbooks_updated": result.playbooks_updated,
        "playbooks_skipped_unchanged": result.playbooks_skipped_unchanged,
        "memories_processed": result.memories_processed,
        "errors": result.errors,
    }


def _generate_job_id(content: str, user_id: str) -> str:
    """Generate a deterministic job ID from content + user_id."""
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"raw-{h}"


def _dreaming_cron_hours() -> list[int]:
    """Resolve the hours the dreaming sweep cron fires.

    ``DREAMING_CRON_HOURS`` is an interval in hours (default 24 = nightly
    at :35 past the anchor hour). Translated into the discrete hour-of-day
    set arq's `cron` accepts, anchored at ``DREAMING_CRON_ANCHOR_HOUR``
    (default 3 — after the dedup (:00) and expiry (:15) crons; the worker
    clock is normally UTC, so non-UTC operators set the anchor to land the
    sweep in their local quiet hours).
    """
    from extensions.dreaming.config import dreaming_settings

    anchor = dreaming_settings.cron_anchor_hour % 24
    interval = max(1, min(24, dreaming_settings.cron_hours))
    return [(anchor + h) % 24 for h in range(0, 24, interval)]


def _dedup_cron_jobs() -> list:
    """Dedup cron entry, or nothing when ``DEDUP_CRON_HOURS`` is empty.

    An empty hour set means "cron disabled" and must never reach arq: arq's
    next-fire search (``cron.py _get_next_dt``) iterates candidate datetimes
    until one matches — an empty hour set never matches, so it spins forever
    INSIDE the event loop, pegging a core and starving every queued job
    (observed on the bench stack, 2026-07-06).
    """
    if not settings.dedup_cron_hours:
        return []
    return [
        cron(
            dedup_all_memories,
            hour=settings.dedup_cron_hours,
            minute=0,
            # dedup_all_memories offloads blocking work to threads; thread-pool
            # calls are not cooperatively cancellable, so an ARQ timeout would
            # mark the cron failed while the underlying dedup keeps running.
            timeout=None,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
    ]


def _strategy_synthesizer_cron_hours() -> list[int]:
    """Resolve the hours the strategy-playbook synthesizer cron fires.

    ``STRATEGY_SYNTHESIZER_CRON_HOURS`` is an interval in hours (default 6),
    translated to arq's discrete hour-of-day set. Fires at :55 to stagger after
    the wiki-synth (:45) and dedup (:00) crons on the same graph worker.
    """
    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings

    interval = max(1, min(24, strategy_synthesizer_settings.cron_hours))
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

    # E3: the flush path's session_id feeds the same summary slots as run_id
    # does on the extraction path (one session concept, two entry points).
    await _note_session_messages(ctx, user_id, session_id, [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ])

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
    if ctx.get("ledger") is not None:
        # This cron compiles for the *default user*; under the ledger no cron may act as one.
        return {"skipped": True, "reason": "default-user cron is disabled while the ledger is on"}

    now = datetime.now()
    if now.hour < compiler_settings.compile_after_hour:
        return None

    service: MemoryService = ctx["service"]
    writer: ObsidianWriter = ctx.get("compiler_writer") or ObsidianWriter()
    ctx.setdefault("compiler_writer", writer)

    # The vault is one shared tree and every daily-log entry records its owner, so
    # post-compile dedup runs for each user whose entries were compiled;
    # ``default_user_id`` only covers entries written before owners were recorded.
    results = await compile_all_pending(service, writer, user_id=settings.default_user_id)
    if results:
        writer.append_log(f"Auto-compiled {len(results)} date(s) via cron")
        logger.info(f"Auto-compile cron: compiled {len(results)} dates")
        return {"dates_compiled": len(results)}
    return None


async def dedup_all_memories(ctx: dict) -> dict:
    """Cron job: deduplicate Qdrant memories for every user.

    When dreaming is enabled the lossy semantic phase is skipped — the
    dream sweep's MERGE action supersedes it (folds unique details into
    the survivor instead of hard-deleting the older near-duplicate). The
    lossless exact-hash phase always runs.
    """
    from extensions.dreaming.config import dreaming_settings

    service: MemoryService = ctx["service"]
    semantic = not dreaming_settings.enabled
    user_ids = await asyncio.to_thread(service.get_all_user_ids, batch_size=settings.dedup_batch_size)
    logger.info(
        f"Dedup cron: found {len(user_ids)} users (semantic phase "
        f"{'on' if semantic else 'off — dreaming owns near-dup merges'})"
    )

    results = []
    for uid in user_ids:
        try:
            # Offload blocking dedup work to a thread pool so the event loop
            # stays responsive (health checks, ARQ timeout, concurrent jobs).
            # dedup_memories does sync Qdrant scroll pagination + embedding/LLM
            # calls that would otherwise freeze the loop for minutes at scale.
            result = await asyncio.to_thread(service.dedup_memories, uid, semantic=semantic)
            # Sibling facts that shared a deleted duplicate's conversation
            # episode need a single-fact graph re-ingest (graph queue).
            sibling_jobs = result.pop("graph_jobs", [])
            if sibling_jobs:
                result["graph_jobs_enqueued"] = await _enqueue_graph_jobs(ctx, sibling_jobs, None)
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


async def process_connector_sync(
    ctx: dict, connector_id: str, generation: str | None = None
) -> dict:
    """Background task: pull a connector's resources and ingest changed ones.

    Incremental: resources whose source-side revision matches the last-seen
    revision are skipped; content-hash dedup in ``store_raw`` is the backstop
    for sources without a revision marker. Sync state (cursor + per-resource
    revisions + timestamp) is persisted back to the vault.

    ``generation`` is the connector instance the job was enqueued for (``None``
    for a record from before generations, and for jobs queued before this
    argument existed). When the stored connector is a different instance
    (deleted and re-registered, or re-registered by another owner, since the job
    was queued) the job skips: the new instance has its own job id and syncs
    under it.
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
    # Literal equality: a job queued before generations existed carries ``None`` and runs only
    # while the stored record is still generationless; once the connector was replaced (and so
    # given a generation) that job skips instead of syncing the replacement under its old grant.
    if record.get("generation") != generation:
        return {"skipped": True, "reason": "connector replaced since this sync was enqueued"}
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
            _ing = await asyncio.to_thread(ingest_document, service, doc)
            # Same deferral as file/document ingest: fact graph writes go to
            # the graph queue rather than running inline in the sync loop.
            await _enqueue_graph_jobs(
                ctx, _ing.pop("graph_jobs", []), adapter=_ing.get("adapter")
            )
            if r.revision:
                revisions[r.external_id] = r.revision
            synced += 1
    finally:
        await adapter.aclose()

    # Atomic, field-scoped and bound to the instance read above: a connector
    # deleted, re-created or re-owned meanwhile keeps its own state.
    written = await vault.update_sync_state(
        connector_id,
        cursor=next_cursor,
        last_synced_at=datetime.now(timezone.utc).isoformat(),
        revisions=revisions,
        generation=record.get("generation"),
        # Only on top of the progress this sync started from: an overlapping sync of the
        # same connector that already saved newer progress is never rolled back.
        expected_cursor=cursor,
        expected_revisions=dict(record.get("last_revision_by_id") or {}),
    )
    if not written:
        logger.info(
            f"Connector sync [{connector_id}]: connector replaced or removed, or newer progress saved "
            "by an overlapping sync; state not saved"
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
        # Same job id as an API-triggered sync of this connector instance
        # (task_manager.connector_sync_job_id), so the two coalesce while one is
        # in flight; a finished run's cached result is cleared (not skipped).
        job = await enqueue_connector_sync_job(
            redis,
            rec["connector_id"],
            rec.get("generation"),
            connector_sync_job_id(
                rec["connector_id"], rec.get("owner_user_id"), rec.get("generation")
            ),
        )
        if job is not None:
            enqueued += 1
    logger.info(f"Connector sync cron: enqueued {enqueued} connector(s)")
    return {"enqueued": enqueued}


def _artifact_reference_counter(service):
    """``(owner_dir, file_id) -> number of memory rows referencing the artifact``.

    A row references an artifact through its ``source_ref``: ``parent_id`` /
    ``external_id`` equal the artifact id (file uploads) or ``url`` is the
    artifact's download URL (bundle concepts). The count is over ALL rows
    (any owner, any visibility, tombstoned included): scoping it to an owner
    would need a lossy directory-name-to-user mapping, and an owner directory
    that maps to nobody must never read as "unreferenced". The cost is an
    unindexed payload filter per candidate artifact, bounded by the sweep.
    ``owner_dir`` is accepted for the sweep's callback signature only.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = service._get_memory().vector_store.client
    collection = settings.qdrant_collection

    def count(owner_dir: str, file_id: str) -> int:
        should = []
        for prefix in ("metadata.source_ref.", "metadata.metadata.source_ref."):
            for key in ("parent_id", "external_id"):
                should.append(FieldCondition(key=prefix + key, match=MatchValue(value=file_id)))
            should.append(
                FieldCondition(key=prefix + "url", match=MatchValue(value=f"/v1/ingest/artifacts/{file_id}"))
            )
        return client.count(
            collection_name=collection,
            count_filter=Filter(should=should),
            exact=True,
        ).count

    return count


async def artifact_gc_cron(ctx: dict) -> dict:
    """Cron job: remove orphaned ingest artifacts (flag ``ARTIFACT_GC_ENABLED``, off by default).

    Reports only unless ``ARTIFACT_GC_DRY_RUN=false``. See
    ``ingest.storage.sweep_orphan_artifacts`` for the guards.
    """
    if not settings.artifact_gc_enabled:
        return {"skipped": True, "reason": "ARTIFACT_GC_ENABLED=false"}
    from ingest.storage import sweep_orphan_artifacts

    service: MemoryService = ctx["service"]
    counter = _artifact_reference_counter(service)
    return await asyncio.to_thread(
        sweep_orphan_artifacts,
        settings,
        counter,
        grace_days=settings.artifact_gc_grace_days,
        dry_run=settings.artifact_gc_dry_run,
    )


def _connector_sync_cron_hours() -> list[int]:
    """Translate the connector sync interval (hours) into a set of hours-of-day."""
    interval = max(1, min(24, settings.connector_sync_cron_hours))
    return list(range(0, 24, interval))


def _make_on_job_start(queue_name: str):
    """Build the per-queue ``on_job_start`` hook: bind job context to the logs.

    ARQ runs every job in its own asyncio task, so the contextvars bound here
    reach every log line the job emits (structlog ``merge_contextvars``) and
    never leak into another job. The function name is not in ARQ's job
    context, so it is read from the queued job record; failing to read it
    only leaves ``function`` off the log lines.
    """
    async def on_job_start(ctx: dict) -> None:
        info = None
        try:
            bound = {"job_id": ctx.get("job_id"), "job_try": ctx.get("job_try"), "queue": queue_name}
            structlog.contextvars.bind_contextvars(**{k: v for k, v in bound.items() if v is not None})
            redis = ctx.get("redis")
            if redis is not None and ctx.get("job_id"):
                info = await Job(ctx["job_id"], redis, _queue_name=queue_name).info()
                if info is not None and info.function:
                    structlog.contextvars.bind_contextvars(function=info.function)
        except Exception:  # noqa: BLE001 — observability must never fail a job
            logger.debug("binding job log context failed (non-fatal)", exc_info=True)
        await _bind_usage_attempt(ctx, info)

    return on_job_start


def _multi_writer_batch(info) -> bool:
    """Whether the job is a raw batch holding items of more than one user."""
    try:
        if getattr(info, "function", None) != "process_memory_raw_batch":
            return False
        items = (getattr(info, "args", None) or (None,))[0]
        if not isinstance(items, list):
            return False
        return len({i.get("user_id") for i in items if isinstance(i, dict) and i.get("user_id")}) > 1
    except Exception:  # noqa: BLE001
        return False


async def _bind_usage_attempt(ctx: dict, info=None) -> None:
    """Attribute the provider calls this job makes to the person it serves (only when usage capture is on).

    The person is read from the enqueue-time ``task -> user`` record; a job without one (a cron, a job
    enqueued by another job) is recorded as unattributed rather than dropped.  The context lives as long as
    the job's own asyncio task, and threads the job starts inherit it.  Never fails a job.
    """
    try:
        import usage_capture

        job_id = ctx.get("job_id")
        if usage_capture.get_usage_recorder() is None or not job_id:
            return
        from task_manager import _task_user_key

        tenant = None
        redis = ctx.get("redis")
        if redis is not None:
            raw = await redis.get(_task_user_key(job_id))
            if raw:
                tenant = raw.decode() if isinstance(raw, bytes) else str(raw)
        if tenant is not None and _multi_writer_batch(info):
            # The enqueue-time record keeps only the last writer of a mixed-user batch: billing the whole
            # batch to that one person would be wrong, so it is recorded as unattributed.
            tenant = None
        usage_capture.bind_job_attempt(tenant, job_id, ctx.get("job_try"))
    except Exception:  # noqa: BLE001 — capture must never fail a job
        logger.debug("binding the usage attempt failed (non-fatal)", exc_info=True)


def _make_after_job_end(queue_name: str):
    """Build the per-queue ``after_job_end`` hook: queue.empty webhook (C4).

    After ARQ records a job's result (so the queue sorted set no longer
    contains it), check whether this worker's queue is now empty; if so —
    and ``WEBHOOK_QUEUE_EMPTY_URL`` is configured — POST a small
    ``queue.empty`` JSON event so ingest-then-query flows can stop polling
    per task. The event carries the finishing job's owner (via the
    ``ns:task-user:`` reverse map task_manager writes at enqueue), so a
    consumer can match it to their own work. Delivery is SSRF-guarded and
    fire-and-forget (see webhooks.py); the hook itself swallows every
    error — observability must never fail a job.
    """
    async def after_job_end(ctx: dict) -> None:
        try:
            await _queue_empty_webhook(ctx)
        finally:
            structlog.contextvars.unbind_contextvars("job_id", "job_try", "queue", "function")
            try:
                import usage_capture

                usage_capture.unbind_job_attempt()
            except Exception:  # noqa: BLE001
                pass

    async def _queue_empty_webhook(ctx: dict) -> None:
        url = settings.webhook_queue_empty_url
        if not url:
            return
        try:
            redis = ctx.get("redis")
            if redis is None:
                return
            depth = await redis.zcard(queue_name)
            if depth:
                return
            user_id = None
            job_id = ctx.get("job_id")
            if job_id:
                try:
                    raw = await redis.get(f"ns:task-user:{job_id}")
                    if raw:
                        user_id = raw.decode() if isinstance(raw, bytes) else str(raw)
                except Exception:  # noqa: BLE001 — attribution is best-effort
                    pass
            from datetime import timezone

            from webhooks import fire_queue_empty

            fire_queue_empty(url, {
                "event": "queue.empty",
                "queue": queue_name,
                "job_id": job_id,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001
            logger.debug("queue.empty webhook check failed (non-fatal)", exc_info=True)

    return after_job_end


async def startup(ctx: dict) -> None:
    """Worker startup: initialize MemoryService + extension registry."""
    from logging_config import configure_logging

    configure_logging(worker=True)
    logger.info("ARQ worker starting up...")
    # First, so the healthcheck sees a beat as early as possible; the (sync)
    # warm-up below runs in a thread, so the beat keeps going through it.
    from worker_liveness import start_heartbeat

    start_heartbeat(ctx)
    logger.info(third_party_telemetry.startup_log_line())
    # Durable usage capture (off unless USAGE_STORE_ENABLED).  The recorder must
    # exist before the service builds its provider clients so they are wrapped.
    if settings.usage_store_enabled:
        from usage_store import install_usage_capture

        ctx["usage_store"] = install_usage_capture(settings)
        logger.info("Durable usage store opened (backend=%s)", settings.usage_store_backend)
    from provider_routing import apply_gemini_base_url

    apply_gemini_base_url(settings.ns_gemini_base_url)  # must precede the first client
    service = MemoryService()
    # Off the loop so the liveness heartbeat keeps beating through a slow cold start.
    await asyncio.to_thread(service._get_memory)  # warm up connections
    from hybrid_inference import warmup_extraction
    await asyncio.to_thread(warmup_extraction)
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
    from worker_liveness import stop_heartbeat

    await stop_heartbeat(ctx)
    registry = ctx.get("extension_registry")
    if registry:
        await registry.shutdown_all()
    service: MemoryService = ctx.get("service")
    if service:
        service.close()
    usage_store = ctx.get("usage_store")
    if usage_store is not None:
        from usage_store import uninstall_usage_capture

        uninstall_usage_capture(usage_store)
    logger.info("ARQ worker stopped.")


class WorkerSettings:
    """Light/fast worker: vector writes, reads, conversation tasks, light crons.

    The slow Graphiti work (per-write graph enrichment + the heavy dedup /
    wiki-synth crons) lives on GraphWorkerSettings, and bulk document/file
    ingestion + connector sync live on IngestWorkerSettings — so neither a burst
    of graph writes, a 12-minute wiki-synth run, nor a folder ingest can starve
    the fast writes/reads here. Run with ``arq worker.WorkerSettings``.
    """
    functions = [
        process_memory_store,
        process_memory_raw,
        process_memory_raw_batch,
        process_memory_retag,
        process_conversation_flush,
        process_conversation_compile,
        # process_session_summary lives on GraphWorkerSettings (audit 27 #24):
        # summary refreshes are multi-second Gemini calls with no latency SLO
        # and must not compete for the 10 fast slots here.
    ]
    cron_jobs = [
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
    on_job_start = _make_on_job_start(settings.arq_queue_name)
    after_job_end = _make_after_job_end(settings.arq_queue_name)
    max_jobs = settings.arq_max_jobs
    job_timeout = settings.arq_job_timeout
    max_tries = settings.arq_max_retries  # keep light-queue retry behavior unchanged


class GraphWorkerSettings:
    """Heavy/slow worker: knowledge-graph enrichment + the expensive crons.

    Consumes the dedicated graph queue so slow Graphiti entity extraction
    (~minutes per write) and the dedup / wiki-synth crons run here, isolated
    from the latency-sensitive vector writes/reads on WorkerSettings. Run with
    ``arq worker.GraphWorkerSettings``. Lower max_jobs caps concurrent Gemini
    extraction (avoids AI-Studio throttling); higher job_timeout fits the
    multi-minute graph writes.
    """
    functions = [
        process_graph_enrichment,
        process_graph_reenrich,
        run_dream_sweep,
        # Session summary refreshes (audit 27 #24): Gemini-bound, no latency
        # SLO — enqueued by _note_session_messages onto the graph queue.
        process_session_summary,
    ]
    cron_jobs = _dedup_cron_jobs() + [
        cron(
            repair_stale_vectors_cron,
            minute=20,          # hourly; a no-op scroll when no row is pending
            timeout=900,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            graph_reenrich_sweep_cron,
            minute=25,          # hourly; a no-op scroll when no row awaits re-enrichment
            timeout=3600,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            dream_sweep_cron,
            # Imported lazily so importing worker.py doesn't load the
            # dreaming settings tree before the worker is actually
            # instantiated. :35 staggers after dedup (:00) and expiry
            # (:15), before the strategy-playbook synth (:55).
            hour=set(_dreaming_cron_hours()),
            minute=35,
            timeout=3600,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            synthesize_strategy_playbooks_cron,
            # Staggered to :55 so it runs after the wiki-synth (:45) and dedup
            # (:00) crons rather than contending for the same Gemini budget.
            hour=set(_strategy_synthesizer_cron_hours()),
            minute=55,
            timeout=1800,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_settings()
    queue_name = settings.graph_queue_name
    on_job_start = _make_on_job_start(settings.graph_queue_name)
    after_job_end = _make_after_job_end(settings.graph_queue_name)
    max_jobs = settings.graph_max_jobs
    job_timeout = 900  # graph.add + entity extraction can run several minutes
    max_tries = settings.arq_max_retries


class IngestWorkerSettings:
    """Ingestion worker: bulk document/file ingest + connector sync.

    Consumes the dedicated ingest queue so a folder/zip upload (chunking +
    Docling parse + LLM fact extraction) or a connector re-sync runs here,
    isolated from the latency-sensitive vector writes/reads on WorkerSettings.
    Run with ``arq worker.IngestWorkerSettings``. ``max_jobs`` is low to bound
    concurrent Docling/Gemini calls; ``job_timeout`` is generous for large files.
    """
    functions = [
        process_ingest_document,
        process_ingest_file,
        process_ingest_okf_bundle,
        process_connector_sync,
        process_code_index,
    ]
    cron_jobs = [
        cron(
            connector_sync_cron,
            hour=set(_connector_sync_cron_hours()),
            minute=50,
            timeout=600,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
        cron(
            artifact_gc_cron,
            hour={4},
            minute=20,
            timeout=1800,
            unique=True,
            max_tries=1,
            run_at_startup=False,
        ),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_settings()
    queue_name = settings.ingest_queue_name
    on_job_start = _make_on_job_start(settings.ingest_queue_name)
    after_job_end = _make_after_job_end(settings.ingest_queue_name)
    max_jobs = settings.ingest_max_jobs
    job_timeout = 900  # a large PDF parse + fact extraction can run minutes
    max_tries = settings.arq_max_retries
