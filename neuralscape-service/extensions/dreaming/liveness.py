"""Code-intel liveness consumer for dreaming (E5).

When a code reindex detects deleted/modified symbols, this module flags anchored
memories for temporal_reframe via the dreaming sweep. The integration respects
all WT5 dreaming-safety invariants:

- **Reversible**: liveness-driven reframes are reversible tombstones/soft-flags,
  never hard deletes of user memories.
- **Scoped invalidation**: only memories anchored to symbols in the blast radius
  are flagged — never a broad pool sweep.
- **Secret gate + silent-sweep guards**: the liveness pass runs under the same
  enable/secret conditions as the rest of the sweep.
- **Boundary**: consumes anchors and flags memories via the source_ref bridge —
  does NOT weld the code graph into the memory graph.

The "ambiguous" extraction tag plumbing already exists (via source_ref.extraction
in E4) but is unconsumed until this slice. This is its consumer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LivenessEvent:
    """One liveness event flagging a memory for reframe/invalidation.

    Attributes:
        memory_id: The memory to flag.
        anchor_key: The code anchor that triggered this event (repo::fqn).
        reason: Why this memory is flagged (e.g., "symbol deleted", "symbol modified").
        confidence: Confidence score for the reframe action (0.0-1.0).
    """

    memory_id: str
    anchor_key: str
    reason: str
    confidence: float = 0.9  # E5: high confidence for code-grounded invalidation


def detect_affected_memories(
    service,
    change_report,
    *,
    code_space: str,
) -> list[LivenessEvent]:
    """Detect memories affected by code changes via the anchor bridge.

    Args:
        service: The MemoryService instance.
        change_report: ChangeReport from NativeEngine.detect_changes().
        code_space: The code_space partition key.

    Returns:
        List of LivenessEvent instances flagging affected memories.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
    )

    if not change_report.affected_anchors:
        return []

    # Search for memories with source_ref.external_id in affected_anchors
    # (E4 bridge: memories anchored to code use source_ref.external_id = "<repo>::<fqn>")
    m = service._get_memory()
    client = m.vector_store.client
    collection_name = m.vector_store.collection_name

    # Build filter: source_ref.external_id IN affected_anchors
    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source_ref.external_id",
                match=MatchAny(any=change_report.affected_anchors),
            )
        ]
    )

    # Use scroll for metadata-only lookup (no embedder call, no NumPy)
    hits = []
    try:
        offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            # scroll returns (records, next_offset)
            records, next_offset = scroll_result
            hits.extend(records)
            if next_offset is None or len(records) == 0:
                break
            offset = next_offset
            # Safety cap
            if len(hits) >= 1000:
                logger.warning(
                    "Code liveness: hit 1000-memory cap for code_space=%s (truncated)",
                    code_space,
                )
                break
    except Exception:
        logger.warning(
            "Failed to fetch affected memories for code changes (non-fatal)",
            exc_info=True,
        )
        return []

    # Build liveness events
    events = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        memory_id = payload.get("id")
        metadata = payload.get("metadata", {})
        source_ref = metadata.get("source_ref", {})
        anchor_key = source_ref.get("external_id", "")

        if not memory_id or not anchor_key:
            continue

        # Determine reason: was the symbol deleted or modified?
        reason = "symbol modified"
        for fqn in change_report.deleted_symbols:
            repo = code_space.split("--")[-1] if "--" in code_space else "unknown"
            if anchor_key == f"{repo}::{fqn}":
                reason = "symbol deleted"
                break

        # E5: high confidence for code-grounded invalidation
        # (the code change is objective ground truth, not LLM speculation)
        events.append(
            LivenessEvent(
                memory_id=memory_id,
                anchor_key=anchor_key,
                reason=reason,
                confidence=0.95,
            )
        )

    logger.info(
        "Code liveness: detected %d affected memories for code_space=%s",
        len(events), code_space,
    )
    return events


def apply_liveness_events(
    service,
    events: list[LivenessEvent],
    *,
    dry_run: bool = False,
) -> int:
    """Apply liveness events by flagging affected memories for temporal reframe.

    This is the E5 liveness consumer: when code changes are detected, flag the
    anchored memories by setting a metadata marker that the next dreaming sweep
    will consume and reframe via temporal_reframe.

    Args:
        service: The MemoryService instance.
        events: LivenessEvent instances from detect_affected_memories().
        dry_run: If True, log what would be flagged without writing.

    Returns:
        Number of memories flagged.

    Safety:
        - Reversible: the flag is a metadata marker, not a delete.
        - Scoped: only memory_ids in events are touched.
        - Gated: caller must check dreaming.enabled before calling.
    """
    if not events:
        return 0

    from datetime import datetime, timezone

    flagged = 0
    for event in events:
        if dry_run:
            logger.info(
                "DRY RUN: would flag memory %s (anchor=%s, reason=%s)",
                event.memory_id, event.anchor_key, event.reason,
            )
            flagged += 1
            continue

        # Flag the memory by setting metadata.code_liveness_stale = True
        # The dreaming sweep will consume this flag and apply temporal_reframe
        try:
            m = service._get_memory()
            client = m.vector_store.client
            collection_name = m.vector_store.collection_name
            patch = {
                "code_liveness_stale": True,
                "code_liveness_anchor": event.anchor_key,
                "code_liveness_reason": event.reason,
                "code_liveness_flagged_at": datetime.now(timezone.utc).isoformat(),
            }
            client.set_payload(
                collection_name=collection_name,
                payload=patch,
                points=[event.memory_id],
                key="metadata",
            )
            flagged += 1
        except Exception:
            logger.warning(
                "Failed to flag memory %s for liveness (non-fatal)",
                event.memory_id, exc_info=True,
            )

    logger.info("Flagged %d memories for code-liveness reframe", flagged)
    return flagged


def process_code_changes_for_liveness(
    service,
    code_space: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Detect code changes and flag affected memories for dreaming reframe.

    This is the main E5 liveness entry point: call after NativeEngine.index()
    to detect changes and flag affected memories.

    Args:
        service: The MemoryService instance.
        code_space: The code_space partition key.
        dry_run: If True, report what would be flagged without writing.

    Returns:
        Dict with liveness report: {events, flagged, summary}.

    Safety:
        - Respects dreaming.enabled (no-op if disabled).
        - All write operations are reversible metadata patches.
    """
    from .config import dreaming_settings

    if not dreaming_settings.enabled and not dry_run:
        logger.info("Dreaming disabled — skipping code-liveness pass")
        return {"events": [], "flagged": 0, "summary": "dreaming disabled"}

    # Get the code-graph engine for this code_space
    # Parse code_space to extract repo_name: "code--{user_id}--{repo_name}"
    try:
        from adapters.code_graph.query import get_engine
        from config import settings as core_settings

        parts = code_space.split("--")
        if len(parts) < 3:
            raise ValueError(f"Invalid code_space format: {code_space}")
        repo_name = parts[2]
        user_id = parts[1]

        # Use repo:<name> graph_id format to get NativeEngine
        graph_id = f"repo:{repo_name}"
        engine = get_engine(graph_id, user_id, core_settings)
    except Exception:
        logger.warning(
            "Failed to get code-graph engine for %s (non-fatal)", code_space,
            exc_info=True,
        )
        return {"events": [], "flagged": 0, "summary": "engine unavailable"}

    # Detect changes
    try:
        change_report = engine.detect_changes()
    except Exception:
        logger.warning(
            "detect_changes failed for %s (non-fatal)", code_space, exc_info=True,
        )
        return {"events": [], "flagged": 0, "summary": "detect_changes failed"}

    # Detect affected memories
    events = detect_affected_memories(service, change_report, code_space=code_space)

    # Flag the memories
    flagged = apply_liveness_events(service, events, dry_run=dry_run)

    summary = (
        f"Code liveness for {code_space}: {len(events)} events, "
        f"{flagged} memories flagged. {change_report.summary}"
    )
    logger.info(summary)
    return {"events": events, "flagged": flagged, "summary": summary}


def detect_inventory_diff_liveness(
    service,
    code_space: str,
    engine,
    *,
    previous_inventory: set[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Phase E: Generalized liveness from external reindex (any engine).

    Diff the engine's current symbol inventory against a previous snapshot
    (or anchors in Neo4j) to detect deleted/changed symbols, then flag affected
    memories for dreaming reframe.

    This upgrades E5 to work regardless of which engine (native/CBM/graphify)
    indexed. The consumer (temporal_reframe) is unchanged; only the per-driver
    diff PRODUCER is new.

    Args:
        service: The MemoryService instance.
        code_space: The code_space partition key.
        engine: The CodeIntelEngine instance (any driver).
        previous_inventory: Optional previous symbol inventory (canonical FQNs).
            If None, fetches current anchors from Neo4j.
        dry_run: If True, report what would be flagged without writing.

    Returns:
        Dict with liveness report: {events, flagged, summary}.

    Safety:
        - Respects dreaming.enabled (no-op if disabled).
        - All write operations are reversible metadata patches.
    """
    from .config import dreaming_settings

    if not dreaming_settings.enabled and not dry_run:
        logger.info("Dreaming disabled — skipping inventory-diff liveness")
        return {"events": [], "flagged": 0, "summary": "dreaming disabled"}

    try:
        # Get current symbol inventory from the engine
        if not hasattr(engine, "get_symbol_inventory"):
            logger.warning(
                "Engine %s doesn't support get_symbol_inventory (liveness unavailable)",
                type(engine).__name__,
            )
            return {"events": [], "flagged": 0, "summary": "inventory method unavailable"}

        current_inventory = engine.get_symbol_inventory()

        # Get previous inventory: either passed in or fetch from anchors
        if previous_inventory is None:
            previous_inventory = _fetch_anchor_inventory(code_space)

        # Diff: detect deleted symbols (in previous but not current)
        deleted_fqns = previous_inventory - current_inventory

        if not deleted_fqns:
            logger.debug("No deleted symbols detected in inventory diff for %s", code_space)
            return {"events": [], "flagged": 0, "summary": "no deleted symbols"}

        # Build anchor keys for deleted symbols
        parts = code_space.split("--")
        repo = parts[-1] if len(parts) >= 3 else "unknown"
        deleted_anchors = [f"{repo}::{fqn}" for fqn in deleted_fqns]

        # Search for memories with these anchor keys
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        m = service._get_memory()
        client = m.vector_store.client
        collection_name = m.vector_store.collection_name

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source_ref.external_id",
                    match=MatchAny(any=deleted_anchors),
                )
            ]
        )

        # Scroll to find all affected memories
        hits = []
        offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records, next_offset = scroll_result
            hits.extend(records)
            if next_offset is None or len(records) == 0:
                break
            offset = next_offset
            if len(hits) >= 1000:
                logger.warning(
                    "Inventory-diff liveness: hit 1000-memory cap for code_space=%s (truncated)",
                    code_space,
                )
                break

        # Build liveness events
        events = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            memory_id = payload.get("id")
            metadata = payload.get("metadata", {})
            source_ref = metadata.get("source_ref", {})
            anchor_key = source_ref.get("external_id", "")

            if not memory_id or not anchor_key:
                continue

            events.append(
                LivenessEvent(
                    memory_id=memory_id,
                    anchor_key=anchor_key,
                    reason="symbol deleted (inventory diff)",
                    confidence=0.95,
                )
            )

        # Flag the memories
        flagged = apply_liveness_events(service, events, dry_run=dry_run)

        summary = (
            f"Inventory-diff liveness for {code_space}: {len(deleted_fqns)} deleted symbols, "
            f"{len(events)} events, {flagged} memories flagged."
        )
        logger.info(summary)
        return {"events": events, "flagged": flagged, "summary": summary}

    except Exception:
        logger.warning(
            "Inventory-diff liveness failed for %s (non-fatal)",
            code_space,
            exc_info=True,
        )
        return {"events": [], "flagged": 0, "summary": "inventory diff failed"}


def _fetch_anchor_inventory(code_space: str) -> set[str]:
    """Fetch current anchor inventory (canonical FQNs) from Neo4j.

    Args:
        code_space: The code_space partition key.

    Returns:
        Set of canonical FQNs that have CodeAnchor nodes.
    """
    from adapters.code_graph.query import get_engine
    from config import settings as core_settings

    try:
        parts = code_space.split("--")
        if len(parts) < 3:
            return set()
        repo_name = parts[2]
        user_id = parts[1]

        # Get the engine to access Neo4j (native engine has _code_neo4j)
        graph_id = f"repo:{repo_name}"
        engine = get_engine(graph_id, user_id, core_settings)

        # Fetch all CodeAnchor fqns (canonical)
        cypher = """
        MATCH (a:CodeAnchor {code_space: $code_space})
        RETURN a.fqn AS fqn
        """
        results = engine._run_cypher(cypher, code_space=code_space)
        return {r["fqn"] for r in results if r["fqn"]}

    except Exception:
        logger.warning(
            "Failed to fetch anchor inventory for %s (non-fatal)",
            code_space,
            exc_info=True,
        )
        return set()
