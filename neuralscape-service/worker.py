"""ARQ worker for async memory processing.

Run with: arq worker.WorkerSettings
"""

import asyncio
import hashlib
import logging
from datetime import datetime

from arq.cron import cron

import adapters  # noqa: F401 — registers knowledge-adapter taxonomies at import (deterministic MEMORY_CATEGORIES across all workers)
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

    # Resolve the requested effective visibility so the idempotency check below is
    # visibility-aware. Without this, a dictator promoting existing private/shared
    # text to a `standard` would match the old copy and be dropped before ever
    # reaching store_raw (which is the tier-aware create path). Mirrors store_raw.
    from schemas import default_visibility_for_category, normalize_visibility
    _req_vis = v2_extras.get("visibility")
    _eff_req_vis = (
        normalize_visibility(_req_vis) if _req_vis
        else default_visibility_for_category(category).value
    )

    # Idempotency: skip only when identical content already exists AT THE SAME
    # visibility tier (a different-tier write is a distinct memory).
    # vector_only (audit 27 #12): this is an internal near-dupe probe — it
    # must not pay the graph pass + edge enrichment of a full hybrid recall,
    # and must not pollute the dreaming recall traces.
    try:
        existing = service.search(
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
        memory_kind=v2_extras.get("memory_kind"),
        source_ref=v2_extras.get("source_ref"),
        add_to_graph=False,
        return_created=True,
    )

    # Defer graph enrichment for newly-created rows onto the graph queue. Skip
    # for content-hash dedup hits (created=False) — they're already in the graph.
    if created and memories:
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


async def process_graph_enrichment(
    ctx: dict,
    memory_id: str,
    content: str,
    user_id: str,
    project_id: str | None = None,
    visibility: str | None = None,
    source_ref: dict | None = None,
    adapter: str | None = None,
) -> dict:
    """Background task: add a stored memory's content to the knowledge graph.

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
    graph_ontology = None
    if adapter:
        # Strict resolution (audit 27 #36): a queued job carrying an adapter
        # name that isn't registered in this worker FAILS with a clear error
        # instead of silently enriching without the adapter's ontology.
        from adapters import require_adapter

        graph_ontology = require_adapter(adapter).graph_ontology_kwargs()
    enriched = await asyncio.to_thread(
        service.enrich_graph,
        content=content,
        user_id=user_id,
        project_id=project_id,
        visibility=visibility or "private",
        memory_id=memory_id,
        source_ref=source_ref,
        graph_ontology=graph_ontology,
        occurred_at=occurred_at,
    )
    if not enriched:
        logger.warning(
            "Graph enrichment did not complete for memory %s — it remains "
            "vector-only (graph unconfigured or a transient LLM/Graphiti error "
            "was swallowed). See enrich_graph warnings above.",
            memory_id,
        )
    return {"memory_id": memory_id, "enriched": enriched}


async def expire_old_memories_cron(ctx: dict) -> dict:
    """Cron job: purge memories whose expires_at is in the past (memory-model v2)."""
    service: MemoryService = ctx["service"]
    result = await asyncio.to_thread(service.expire_old_memories)
    if result.get("deleted_count"):
        logger.info(f"Expiry cron: purged {result['deleted_count']} memories")
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
            _queue_name=settings.ingest_queue_name,
        )
        enqueued += 1
    logger.info(f"Connector sync cron: enqueued {enqueued} connector(s)")
    return {"enqueued": enqueued}


def _connector_sync_cron_hours() -> list[int]:
    """Translate the connector sync interval (hours) into a set of hours-of-day."""
    interval = max(1, min(24, settings.connector_sync_cron_hours))
    return list(range(0, 24, interval))


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
    after_job_end = _make_after_job_end(settings.arq_queue_name)
    max_jobs = 10
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
        run_dream_sweep,
        # Session summary refreshes (audit 27 #24): Gemini-bound, no latency
        # SLO — enqueued by _note_session_messages onto the graph queue.
        process_session_summary,
    ]
    cron_jobs = _dedup_cron_jobs() + [
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
    after_job_end = _make_after_job_end(settings.graph_queue_name)
    max_jobs = 4
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
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = parse_redis_settings()
    queue_name = settings.ingest_queue_name
    after_job_end = _make_after_job_end(settings.ingest_queue_name)
    max_jobs = 3
    job_timeout = 900  # a large PDF parse + fact extraction can run minutes
    max_tries = settings.arq_max_retries
