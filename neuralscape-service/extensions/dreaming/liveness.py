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

    # Build filter: source_ref.external_id IN affected_anchors
    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source_ref.external_id",
                match=MatchAny(any=change_report.affected_anchors),
            )
        ]
    )

    # Scroll the collection to find all affected memories
    # (use a dummy vector since we're filtering by metadata, not semantic search)
    import numpy as np
    vector_size = len(m.embedding_model.embed("test", memory_action="search"))
    dummy_vector = np.zeros(vector_size).tolist()

    try:
        result = client.query_points(
            collection_name=m.vector_store.collection_name,
            query=dummy_vector,
            query_filter=query_filter,
            limit=1000,  # cap to avoid runaway queries
            with_payload=True,
        )
        hits = list(getattr(result, "points", result) or [])
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

    from config import settings as core_settings
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
            patch = {
                "code_liveness_stale": True,
                "code_liveness_anchor": event.anchor_key,
                "code_liveness_reason": event.reason,
                "code_liveness_flagged_at": datetime.now(timezone.utc).isoformat(),
            }
            client.set_payload(
                collection_name=core_settings.qdrant_collection,
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
    try:
        from adapters.code_graph import get_engine

        engine = get_engine(service, code_space=code_space)
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
