"""Ingest a document into memory: verbatim passages + distilled facts.

Every memory produced carries the document's ``source_ref`` provenance
descriptor:
- **Passages** (``memory_kind="passage"``) — verbatim chunks, vector-only
  (not added to the graph: a long doc would otherwise spawn an entity-
  extraction graph episode per chunk). Each carries its ``chunk_index`` and
  ``span`` so it backlinks to its exact position in the parent document.
- **Facts** (``memory_kind="fact"``) — LLM-distilled atomic facts, stored with
  full graph linkage so they feed the knowledge graph + a ``(:Source)`` node.

Re-ingesting the same content is idempotent: ``store_raw`` dedupes by
content hash, so a connector re-sync that re-fetches unchanged content
produces no duplicates.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from adapters import require_adapter
from ingest.chunking_strategies import get_chunking_strategy
from ingest.extractors import get_extractor
from schemas import (
    GLOBAL_CATEGORIES,
    PROJECT_CATEGORIES,
    MemoryScope,
    default_scope_for_category,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestDoc:
    """A document to ingest, with its base source descriptor + storage options.

    ``source`` is the parent-level :class:`~schemas.SourceDescriptor` as a dict
    (``connector_id``/``connector_type`` required). Passage memories extend it
    with per-chunk ``chunk_index``/``span``; facts use it as-is.
    """
    content: str
    source: dict
    user_id: str
    category: str = "domain_knowledge"
    scope: str = "global"
    project_id: str | None = None
    visibility: str | None = None
    tags: list[str] | None = None
    agent_id: str | None = None
    run_id: str | None = None
    extract_facts: bool = True
    index_passages: bool = True
    # Chunking knobs (fall back to the adapter's defaults when None)
    max_chars: int | None = None
    overlap: int | None = None
    # Knowledge adapter selecting the taxonomy / chunker / extractor / graph
    # ontology. "default" reproduces today's behavior exactly.
    adapter: str = "default"
    # Envelope-level chunking-strategy override (registry id). Set by callers
    # whose *format* dictates the chunker independent of any domain adapter —
    # e.g. the OKF bundle walker's frontmatter-aware strategy. None defers to
    # the adapter's strategy (today's behavior).
    chunking_strategy: str | None = None
    # Event time (ISO 8601): when the ingested content actually happened /
    # was written, for historical ingestion (imported journals, old notes).
    # Stamped on every produced passage + fact. None ⇒ omitted (event time
    # unknown; readers fall back to created_at).
    occurred_at: str | None = None
    # Workspace partition (WT6): absent/"memory" = memory type; any other value
    # = reference workspace (fenced out of card/reflection/default recall). None
    # defers to adapter-based defaulting in ingest_document.
    workspace: str | None = None


def _served_tokens(stored) -> int:
    """Token cost of stored rows via their write-time stamp (no tokenizer)."""
    from savings_meter import hit_tokens

    return sum(hit_tokens(m) for m in stored)


def _fact_scope(category: str, project_id: str | None) -> tuple[str, str | None]:
    """Resolve (scope, project_id) for an extracted fact, mirroring _batch_store_facts."""
    scope = default_scope_for_category(category)
    if project_id and category not in GLOBAL_CATEGORIES:
        scope = MemoryScope.PROJECT
    if category in PROJECT_CATEGORIES and not project_id:
        scope = MemoryScope.GLOBAL
    scope_val = scope.value if isinstance(scope, MemoryScope) else scope
    # A global-scope fact is cross-project, so it must not carry a project_id.
    return scope_val, (project_id if scope_val == "project" else None)


def ingest_document(service, doc: IngestDoc) -> dict:
    """Chunk + ingest ``doc`` into memory. Returns a summary dict.

    Args:
        service: A ``MemoryService`` instance.
        doc: The document + options.

    Returns:
        ``{"passages": int, "facts": int, "memory_ids": [...], "parent_id": str}``.
    """
    # Resolve the knowledge adapter (taxonomy / chunker / extractor / graph
    # ontology). STRICT (audit 27 #36): a queued job whose adapter isn't
    # registered in this worker (failed import / missing optional extra /
    # removed adapter) fails the task with a clear error — silently falling
    # back to the default taxonomy would ingest under the wrong
    # taxonomy/ontology. Request-time typos are already rejected with a 422
    # by ``schemas.validate_adapter_name`` before anything is enqueued.
    adapter = require_adapter(doc.adapter)

    # WT6: workspace defaulting. Non-default adapters (e.g. trading_strategy, code_graph)
    # DEFAULT to a reference workspace derived from adapter name + optional title, unless
    # the caller explicitly overrode workspace. This makes the trigger case (trading books
    # poisoning the identity card) impossible by default — ingested reference content lands
    # in its own partition, fenced out of card/reflection/default recall.
    effective_workspace = doc.workspace
    if effective_workspace is None and adapter.name != "default":
        # Auto-derive a reference workspace name: adapter-name[--title-slug]
        base_name = f"ref-{adapter.name}"
        title_slug = None
        if doc.source.get("title"):
            # Slugify title: lowercase, alphanumeric + dashes only
            import re
            title_slug = re.sub(r'[^a-z0-9\-]+', '-', doc.source["title"].lower()).strip('-')[:30]
        effective_workspace = f"{base_name}--{title_slug}" if title_slug else base_name
        logger.info(
            f"Auto-derived workspace '{effective_workspace}' for adapter '{adapter.name}' "
            f"(WT6: reference content fenced from memory pools by default)"
        )

    base = dict(doc.source)  # shallow copy — we never mutate the caller's dict
    # Stamp sync time once for the whole document.
    base.setdefault("last_synced_at", datetime.now(timezone.utc).isoformat())
    # parent_id identifies the document; default to external_id when unset so
    # passages of one doc share a stable parent backlink.
    parent_id = base.get("parent_id") or base.get("external_id") or base.get("connector_id")
    base["parent_id"] = parent_id

    memory_ids: list[str] = []
    passage_count = 0
    fact_count = 0
    # M1 (ingest lifecycle): sum of the token cost of everything we STORE
    # (the compressed form future recalls serve instead of re-reading the
    # source). Reads the write-time stamp on each stored row — but a dedup hit
    # can return a legacy row with token_estimate=None, whose hit_tokens would
    # fall back to tiktoken; so accumulate ONLY when the meter is on, keeping
    # the kill-switch's "zero tokenizer work when off" guarantee.
    served_tok = 0
    try:
        import savings_meter as _sm

        meter_on = _sm._meter_enabled()
    except Exception:
        meter_on = False

    # ── Passages (verbatim, vector-only) ──
    if doc.index_passages:
        chunker = get_chunking_strategy(doc.chunking_strategy or adapter.chunking_strategy)
        max_chars = doc.max_chars if doc.max_chars is not None else adapter.default_max_chars
        overlap = doc.overlap if doc.overlap is not None else adapter.default_overlap
        for chunk in chunker.chunk(doc.content, max_chars=max_chars, overlap=overlap):
            chunk_source = {
                **base,
                "chunk_index": chunk.index,
                "span": chunk.span,
                "content_hash": hashlib.md5(chunk.text.encode()).hexdigest(),
            }
            try:
                stored = service.store_raw(
                    content=chunk.text,
                    user_id=doc.user_id,
                    category=doc.category,
                    scope=doc.scope,
                    project_id=doc.project_id,
                    tags=doc.tags,
                    agent_id=doc.agent_id,
                    run_id=doc.run_id,
                    source_type="imported",
                    visibility=doc.visibility,
                    memory_kind="passage",
                    source_ref=chunk_source,
                    occurred_at=doc.occurred_at,
                    workspace=effective_workspace,
                    add_to_graph=False,
                )
                memory_ids.extend(m.id for m in stored)
                passage_count += len(stored)
                if meter_on:
                    served_tok += _served_tokens(stored)
            except Exception as e:
                logger.warning(f"Passage store failed (chunk {chunk.index}): {e}")

    # ── Facts (LLM-distilled, graph-linked) ──
    # Facts are stored vector-only here (fast), with graph enrichment DEFERRED:
    # each newly-created fact yields a ``graph_jobs`` entry the caller enqueues
    # onto the graph queue (worker re-resolves the adapter's ontology by name).
    # Inline graph.add at book scale (hundreds of facts × ~minutes of Graphiti
    # extraction each) would blow the ingest job timeout and re-run the whole
    # parse on every ARQ retry. Dedup hits (created=False) produce no job —
    # they're already in the graph.
    graph_jobs: list[dict] = []
    if doc.extract_facts:
        # E4: user/project ride along so operator extraction instructions
        # compose with the adapter's own prompt (adapter first, addendum after).
        facts = service.extract_facts_only(
            doc.content,
            extractor=get_extractor(adapter.extractor),
            user_id=doc.user_id,
            project_id=doc.project_id,
        )
        for category, content in facts:
            scope_val, fact_pid = _fact_scope(category, doc.project_id)
            try:
                stored, created = service.store_raw(
                    content=content,
                    user_id=doc.user_id,
                    category=category,
                    scope=scope_val,
                    project_id=fact_pid,
                    tags=doc.tags,
                    agent_id=doc.agent_id,
                    run_id=doc.run_id,
                    source_type="imported",
                    # Distilled facts restate what the document says directly
                    # → epistemically "explicit" (A1). Verbatim passages carry
                    # no level (they're source text, not extracted claims).
                    epistemic_level="explicit",
                    visibility=doc.visibility,
                    memory_kind="fact",
                    source_ref=base,
                    occurred_at=doc.occurred_at,
                    workspace=effective_workspace,
                    add_to_graph=False,
                    return_created=True,
                )
                memory_ids.extend(m.id for m in stored)
                fact_count += len(stored)
                if meter_on:
                    served_tok += _served_tokens(stored)
                if created:
                    for m in stored:
                        graph_jobs.append({
                            "memory_id": m.id,
                            "content": content,
                            "user_id": doc.user_id,
                            "project_id": fact_pid,
                            "visibility": getattr(m, "visibility", None),
                            "source_ref": base,
                        })
            except Exception as e:
                logger.warning(f"Fact store failed ({category}): {e}")

    logger.info(
        f"Ingested doc parent_id={parent_id} → {passage_count} passages + {fact_count} facts "
        f"({len(graph_jobs)} graph jobs deferred, "
        f"connector={base.get('connector_type')}/{base.get('connector_id')})"
    )

    # M1 — ingest lifecycle: baseline = the full source document a client
    # would otherwise re-read to extract these facts; served = the token cost
    # of everything we stored. Best-effort, off any latency-sensitive path
    # (this runs in the ingest worker); a meter failure never fails ingest.
    try:
        if meter_on and doc.user_id:
            baseline_tok = _sm.count_tokens(doc.content or "")
            event = _sm.measure_ingest(
                baseline_tok, served_tok, item_id=parent_id, corr_id=doc.run_id
            )
            if event is not None:
                _sm.record_event(doc.user_id, event)
    except Exception:
        logger.debug("ingest savings metering failed (non-fatal)", exc_info=True)

    return {
        "passages": passage_count,
        "facts": fact_count,
        "memory_ids": memory_ids,
        "parent_id": parent_id,
        # Deferred graph-enrichment payloads + the adapter whose ontology applies.
        # Callers (the ingest worker) enqueue these onto the graph queue and then
        # drop the key from any client-facing result.
        "graph_jobs": graph_jobs,
        "adapter": adapter.name,
    }
