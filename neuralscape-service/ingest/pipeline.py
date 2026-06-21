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

from ingest.chunking import chunk_text
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
    # Chunking knobs (fall back to chunker defaults when None)
    max_chars: int | None = None
    overlap: int | None = None


def _fact_scope(category: str, project_id: str | None) -> tuple[str, str | None]:
    """Resolve (scope, project_id) for an extracted fact, mirroring _batch_store_facts."""
    scope = default_scope_for_category(category)
    if project_id and category not in GLOBAL_CATEGORIES:
        scope = MemoryScope.PROJECT
    if category in PROJECT_CATEGORIES and not project_id:
        scope = MemoryScope.GLOBAL
    scope_val = scope.value if isinstance(scope, MemoryScope) else scope
    return scope_val, (project_id if scope_val == "project" else project_id)


def ingest_document(service, doc: IngestDoc) -> dict:
    """Chunk + ingest ``doc`` into memory. Returns a summary dict.

    Args:
        service: A ``MemoryService`` instance.
        doc: The document + options.

    Returns:
        ``{"passages": int, "facts": int, "memory_ids": [...], "parent_id": str}``.
    """
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

    # ── Passages (verbatim, vector-only) ──
    if doc.index_passages:
        kwargs = {}
        if doc.max_chars is not None:
            kwargs["max_chars"] = doc.max_chars
        if doc.overlap is not None:
            kwargs["overlap"] = doc.overlap
        for chunk in chunk_text(doc.content, **kwargs):
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
                    add_to_graph=False,
                )
                memory_ids.extend(m.id for m in stored)
                passage_count += len(stored)
            except Exception as e:
                logger.warning(f"Passage store failed (chunk {chunk.index}): {e}")

    # ── Facts (LLM-distilled, graph-linked) ──
    if doc.extract_facts:
        facts = service.extract_facts_only(doc.content)
        for category, content in facts:
            scope_val, fact_pid = _fact_scope(category, doc.project_id)
            try:
                stored = service.store_raw(
                    content=content,
                    user_id=doc.user_id,
                    category=category,
                    scope=scope_val,
                    project_id=fact_pid,
                    tags=doc.tags,
                    agent_id=doc.agent_id,
                    run_id=doc.run_id,
                    source_type="imported",
                    visibility=doc.visibility,
                    memory_kind="fact",
                    source_ref=base,
                    add_to_graph=True,
                )
                memory_ids.extend(m.id for m in stored)
                fact_count += len(stored)
            except Exception as e:
                logger.warning(f"Fact store failed ({category}): {e}")

    logger.info(
        f"Ingested doc parent_id={parent_id} → {passage_count} passages + {fact_count} facts "
        f"(connector={base.get('connector_type')}/{base.get('connector_id')})"
    )
    return {
        "passages": passage_count,
        "facts": fact_count,
        "memory_ids": memory_ids,
        "parent_id": parent_id,
    }
