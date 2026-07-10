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
import re

from knowledge.base import SystemAnswer

logger = logging.getLogger(__name__)

# A dotted FQN-shaped token: >=2 identifier segments joined by dots
# (e.g. ``click.core.Command``, ``src.click.core.Command``,
# ``tests.test_options.test_x``). Used only as a content fallback for
# extract_fqns_from_code_answer.
_FQN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
# Bound the content-derived FQN list so a large traversal answer can't build an
# unbounded MatchAny filter for the batched anchor join.
_MAX_CONTENT_FQNS = 60
# Echo/status line prefixes (lowercased) that echo the QUESTION rather than a hit.
# The content fallback must NOT harvest FQNs from these, or a query token would
# anchor-join even when the engine resolved nothing (the "echo-join" defect).
_ECHO_PREFIXES = (
    "code graph search results for:",  # native query header echoes the question
    "no symbols matching",             # native/cbm miss echoes the question
    "symbols matching ",               # cbm hit header echoes the question (FQNs follow)
    "neighbors of",                    # neighbors header echoes the label
    "no neighbors",
    "no path",
    "no node",
    "traversal:",                      # graphify BFS header echoes Start seeds
    "start:",
)


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


def extract_fqns_from_code_answer(code_answer: SystemAnswer, query: str | None = None) -> list[str]:
    """Extract the FQN list to drive the batched anchor join.

    Preference order:
      1. Structured ``.hits`` with an ``fqn`` field (locate/neighbors/impact ops).
      2. **Content fallback** (Phase G-final GF2): the fusion code leg runs the
         ``query`` op, whose ``CodeKnowledgeSystem`` rendering is text-only
         (``hits=None``). Without this the flagship "who calls X and why is it
         like this" flow could never populate the ``[semantics]`` section live —
         the anchor join had no FQNs to join on. We parse dotted FQN-shaped
         tokens out of the rendered content, but ONLY from genuine hit/traversal
         lines: echo/status header lines (``_ECHO_PREFIXES``) are skipped, and
         (when ``query`` is provided) tokens whose lowercase appears verbatim in
         the query are excluded. Together these kill the "echo-join" — a query
         token like ``Command.invoke`` echoed in a *miss* header must NOT anchor
         when the engine resolved nothing. This is a format-only reader, NOT
         engine intelligence; bounded to ``_MAX_CONTENT_FQNS``.

    Args:
        code_answer: SystemAnswer from a code system.
        query: The originating query, for echo exclusion (recommended). When
            None, only header-line skipping guards against the echo.

    Returns:
        De-duplicated FQN list (structured hits first; else content-derived).
    """
    fqns: list[str] = []
    seen: set[str] = set()
    if code_answer.hits:
        for hit in code_answer.hits:
            if isinstance(hit, dict) and "fqn" in hit:
                f = hit["fqn"]
                if f and f not in seen:
                    seen.add(f)
                    fqns.append(f)

    if not fqns and getattr(code_answer, "content", None):
        q_low = (query or "").lower()
        for line in code_answer.content.split("\n"):
            stripped = line.strip()
            low = stripped.lower()
            # Skip echo/status headers — they carry the question, not a hit.
            if any(low.startswith(p) for p in _ECHO_PREFIXES):
                continue
            for m in _FQN_RE.findall(stripped):
                # File paths (``foo.py``) are not symbols — skip them.
                if m.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java")):
                    continue
                # Echo guard: a token that appears verbatim in the query is the
                # user's own phrasing echoed back, not a resolved code hit.
                if q_low and m.lower() in q_low:
                    continue
                if m not in seen:
                    seen.add(m)
                    fqns.append(m)
                if len(fqns) >= _MAX_CONTENT_FQNS:
                    logger.debug(
                        "extract_fqns: content FQN cap (%d) reached; truncating",
                        _MAX_CONTENT_FQNS,
                    )
                    return fqns
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

        # Phase G-final (GF2): use scroll (filter-only, NO query vector) rather
        # than query_points with a throwaway zero vector. The anchor join filters
        # purely by source_ref.external_id — similarity ordering is meaningless
        # here — so scroll is the right primitive. This also removes a live
        # embedding_model.embed("test") call that was being made on EVERY fused
        # recall solely to learn the vector dimension (~400 ms per call, the
        # dominant fused-leg overhead the GF2 probe measured; the plan predicted
        # ~20-40 ms for the join itself). scroll returns (points, next_offset).
        scroll_result = client.scroll(
            collection_name=m.vector_store.collection_name,
            scroll_filter=query_filter,
            limit=len(anchor_keys) * limit_per_anchor * 2,  # over-fetch for post-filter
            with_payload=True,
        )
        if isinstance(scroll_result, tuple):
            hits = list(scroll_result[0] or [])
        else:  # tolerate mocks that return a bare iterable
            hits = list(getattr(scroll_result, "points", scroll_result) or [])

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


# ── Cross-engine dedup (Phase F) ─────────────────────────────────────

# Per-op precision preferences (PLAN §6 + rootcause §1):
# - neighbors: CBM is higher-precision (structured call-graph edges)
# - path: graphify is higher-precision (measured best path accuracy)
# - query/locate: no strong preference, first wins

_OP_ENGINE_PREFERENCE = {
    "neighbors": ["code-cbm", "code-graphify-lib", "code-graphify-json", "code-native"],
    "path": ["code-graphify-lib", "code-graphify-json", "code-cbm", "code-native"],
    # For query/locate, no strong preference — first-seen wins (order by input)
}


def dedup_code_answers(
    answers: list[SystemAnswer],
    *,
    operation: str = "query",
) -> SystemAnswer:
    """Dedup results from multiple code systems on canonical FQN.

    When >1 code system answers the same op (rare; only on explicit "ask both"),
    dedup on canonical FQN, prefer the higher-precision engine per op class
    (CBM for neighbors, graphify for path — measured, rootcause §1), attribute BOTH.

    Args:
        answers: List of SystemAnswer from different code systems.
        operation: The operation type (neighbors/path/query/locate) for preference.

    Returns:
        Single SystemAnswer with deduped hits, attributed to all contributing engines.
    """
    if len(answers) <= 1:
        return answers[0] if answers else SystemAnswer(
            system_name="none",
            system_version=None,
            content="",
            hits=None,
            metadata={},
        )

    # Get precision-ordered engine list for this op
    preferred_order = _OP_ENGINE_PREFERENCE.get(operation, [])

    # Sort answers by preference (preferred engines first, then input order)
    def engine_priority(ans: SystemAnswer) -> int:
        try:
            return preferred_order.index(ans.system_name)
        except ValueError:
            return len(preferred_order)  # unknown engines go last

    sorted_answers = sorted(answers, key=engine_priority)

    # Dedup by canonical FQN (if hits are structured)
    seen_fqns = set()
    deduped_hits = []
    contributing_systems = set()

    for answer in sorted_answers:
        contributing_systems.add(answer.system_name)
        if answer.hits:
            for hit in answer.hits:
                if isinstance(hit, dict) and "fqn" in hit:
                    fqn = hit["fqn"]
                    if fqn not in seen_fqns:
                        seen_fqns.add(fqn)
                        # COPY the hit before annotating so the caller's original
                        # hit dicts are never mutated (no caller-visible side effects).
                        annotated = dict(hit)
                        annotated["_source_system"] = answer.system_name
                        deduped_hits.append(annotated)

    # Compose content from the preferred (first) answer, note others contributed.
    # Sort the "also searched" list so output is deterministic (a set is not).
    primary = sorted_answers[0]
    content_lines = [primary.content] if primary.content else []
    if len(contributing_systems) > 1:
        others = sorted(s for s in contributing_systems if s != primary.system_name)
        content_lines.append(
            f"\n(Also searched: {', '.join(others)}; results deduped on canonical FQN)"
        )

    return SystemAnswer(
        system_name=f"{primary.system_name}+{len(contributing_systems)-1}",
        system_version=primary.system_version,
        content="\n".join(content_lines),
        hits=deduped_hits if deduped_hits else None,
        metadata={
            "operation": operation,
            "deduped": True,
            "contributing_systems": sorted(contributing_systems),
            "preferred_system": primary.system_name,
        },
    )
