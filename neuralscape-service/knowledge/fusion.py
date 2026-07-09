"""Section composer for fused knowledge-system answers (Phase E).

Per PLAN §6: fusion COMPOSES SECTIONS, never interleaves scores. CBM BM25,
graphify label matches, and NS RRF scores are mutually incomparable; any blended
ranking would be fake precision.

A fused answer structure:
  [structure]   from the routed code system(s): symbol/file:line/edges — engine-attributed
  [semantics]   anchor-joined NS memories: decisions, gotchas, bugfix history (E4 format)
  [memory]      base recall results as today (when the query was a generic recall)

The flagship flow ("who calls X and why is it like this?"):
  1. code system answers structurally (e.g. CBM trace_path/search_graph, 1.5–5 ms)
  2. driver normalizes result FQNs → canonical (§2)
  3. ONE batched anchor lookup for all returned FQNs (generalizing _get_anchor_memories)
  4. attach memories under each hit exactly as E4 formats today

TRANSPORT INVARIANT (DECISIONS.md cross-cutting): fusion NEVER branches on `transport`
— operates only on KnowledgeSystem answers, agnostic to whether the backend is
in-process, MCP-bridge, or HTTP.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.base import SystemAnswer

logger = logging.getLogger(__name__)


def compose_fusion_answer(
    *,
    code_answer: SystemAnswer | None = None,
    anchor_memories: dict[str, list[dict]] | None = None,
    base_answer: SystemAnswer | None = None,
) -> str:
    """Compose a section-structured fusion answer.

    Args:
        code_answer: Structural answer from a code system (optional).
        anchor_memories: Batched anchor memories keyed by canonical FQN (optional).
            Each value is a list of memory dicts from the batched anchor join.
        base_answer: Base NS recall answer (optional, only for generic recall).

    Returns:
        Composed answer text with [structure], [semantics], [memory] sections.

    Example output:
        [structure] (code-cbm)
        src/click/core.py:42 CommandCollection.get_command(name)
          --> CALLS BaseCommand.__init__

        [semantics] (NS anchored memories)
        click.core.CommandCollection:
          - [decision] Switched to lazy command loading for faster CLI startup
          - [gotcha] get_command returns None for unknown commands (not an error)

        [memory] (NS base recall)
        1. [tech_stack] Project uses Click 8.1 for CLI framework
        2. [convention] All CLI commands inherit from BaseCommand
    """
    sections = []

    # [structure] section: code system's structural answer
    if code_answer and code_answer.content:
        sections.append(f"[structure] ({code_answer.system_name})")
        sections.append(code_answer.content)
        sections.append("")  # blank line separator

    # [semantics] section: anchor-joined NS memories
    if anchor_memories:
        sections.append("[semantics] (NS anchored memories)")
        for canonical_fqn, memories in anchor_memories.items():
            if memories:
                sections.append(f"{canonical_fqn}:")
                for mem in memories:
                    category = mem.get("category", "unknown")
                    content = (mem.get("content") or "")[:150]  # clip for display
                    sections.append(f"  - [{category}] {content}")
        sections.append("")

    # [memory] section: base recall results (only for generic recall)
    if base_answer and base_answer.content:
        sections.append("[memory] (NS base recall)")
        sections.append(base_answer.content)
        sections.append("")

    # If no sections, return a fallback message
    if not sections:
        return "No results found across knowledge systems."

    return "\n".join(sections)


def extract_fqns_from_code_answer(code_answer: SystemAnswer) -> list[str]:
    """Extract FQN list from a code system's structured hits.

    Args:
        code_answer: SystemAnswer from a code system with .hits (structured).

    Returns:
        List of FQNs found in the answer's hits.
    """
    fqns = []
    if code_answer.hits:
        for hit in code_answer.hits:
            if isinstance(hit, dict) and "fqn" in hit:
                fqns.append(hit["fqn"])
    return fqns


def batched_anchor_lookup(
    fqns: list[str],
    *,
    repo: str,
    to_canonical_fn,
    user_id: str | None = None,
    limit_per_anchor: int = 3,
) -> dict[str, list[dict]]:
    """Batched anchor join: ONE Qdrant query for all FQNs.

    Generalizes NativeEngine._get_anchor_memories from per-symbol to batched.
    Uses Qdrant MatchAny filter over source_ref.external_id, then groups results
    by anchor key and applies the SAME visibility post-filter VERBATIM.

    Args:
        fqns: List of FQNs to look up (from code system answer).
        repo: Repository name (for anchor key construction).
        to_canonical_fn: Engine's to_canonical(raw_fqn) -> canonical_fqn.
        user_id: Caller user ID for visibility scoping.
        limit_per_anchor: Max memories to return per anchor (default 3).

    Returns:
        Dict mapping canonical FQN -> list of memory dicts.
    """
    from memory_service import get_shared_service
    from qdrant_client.models import FieldCondition, Filter, MatchAny
    import numpy as np

    if not fqns:
        return {}

    # Canonicalize all FQNs and build anchor keys
    canonical_fqns = [to_canonical_fn(fqn) for fqn in fqns]
    anchor_keys = [f"{repo}::{cfqn}" for cfqn in canonical_fqns]

    # Build ONE Qdrant filter: source_ref.external_id IN anchor_keys
    try:
        service = get_shared_service()
        m = service._get_memory()
        client = m.vector_store.client

        must = [
            FieldCondition(
                key="metadata.source_ref.external_id",
                match=MatchAny(any=anchor_keys),
            )
        ]
        query_filter = Filter(must=must)

        # Use a dummy embedding for the query (we're filtering by source_ref, not semantic)
        # Handle mocks/tests where embedding_model may not be set up
        try:
            vector_size = len(m.embedding_model.embed("test", memory_action="search"))
        except (AttributeError, Exception):
            # Fallback for tests: use a default vector size (768 for Gemini)
            vector_size = 768
        dummy_vector = np.zeros(vector_size).tolist()

        result = client.query_points(
            collection_name=m.vector_store.collection_name,
            query=dummy_vector,
            query_filter=query_filter,
            limit=len(anchor_keys) * limit_per_anchor * 2,  # over-fetch for post-filter
            with_payload=True,
        )
        hits = list(getattr(result, "points", result) or [])

        # Group results by anchor key and apply visibility post-filter
        memories_by_anchor: dict[str, list[dict]] = {key: [] for key in anchor_keys}

        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            metadata = payload.get("metadata", {})
            source_ref = metadata.get("source_ref", {})
            anchor_key = source_ref.get("external_id", "")
            visibility = metadata.get("visibility", "private")
            owner = metadata.get("user_id")

            if anchor_key not in memories_by_anchor:
                continue  # not one of our requested anchors

            # Check readability (VERBATIM from native_engine._get_anchor_memories)
            readable = False
            if visibility in ("shared", "standard"):
                readable = True
            elif visibility == "private" and user_id and owner == user_id:
                readable = True

            if readable:
                memory_dict = {
                    "id": payload.get("id"),
                    "content": payload.get("data"),
                    "category": metadata.get("category"),
                    "visibility": visibility,
                    "created_at": metadata.get("created_at"),
                }
                memories_by_anchor[anchor_key].append(memory_dict)

        # Limit memories per anchor
        for key in memories_by_anchor:
            memories_by_anchor[key] = memories_by_anchor[key][:limit_per_anchor]

        # Map back from anchor keys to canonical FQNs for the return dict
        result_by_fqn = {}
        for cfqn, akey in zip(canonical_fqns, anchor_keys):
            if memories_by_anchor[akey]:
                result_by_fqn[cfqn] = memories_by_anchor[akey]

        logger.debug(
            "Batched anchor lookup: %d FQNs -> %d with memories",
            len(fqns), len(result_by_fqn),
        )
        return result_by_fqn

    except Exception:
        logger.warning("Batched anchor lookup failed (non-fatal)", exc_info=True)
        return {}
