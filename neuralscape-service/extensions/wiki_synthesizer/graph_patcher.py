"""Neo4j Cypher helpers for attaching back-references between memories,
graph nodes, and synthesized wiki pages.

Two responsibilities:

1. **At write time** — after a memory write produces Graphiti entity
   nodes, mark those nodes with the originating ``memory_id`` and the
   memory's ``visibility`` / ``owner_user_id``. This is the link the
   wiki synthesizer later uses to walk from a memory back to the graph
   nodes it contributed.

2. **After synthesis** — once a wiki page has been written, set
   ``wiki_path`` and ``wiki_synthesized_at`` on every node that
   contributed. Search results in the graph then carry a pointer to
   the human-readable wiki page.

Both helpers swallow driver-level errors (logging instead of raising)
because they are best-effort enrichments — the underlying memory
write/synthesis must not fail just because a Cypher hop didn't land.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Window for "freshly written by this task" — we identify newly-created
# nodes by matching their `created_at` against this many seconds before
# the write started. Small enough to avoid attaching memory_id to nodes
# from unrelated concurrent writes; large enough to cover graph-write
# latency under load.
WRITE_WINDOW_SECONDS = 120


async def attach_memory_id(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    visibility: str | None,
    owner_user_id: str | None,
    write_started_at: datetime,
    window_seconds: int | None = None,
) -> int:
    """Mark recently-created Graphiti nodes with their originating memory.

    Returns the number of nodes patched. Best-effort — never raises.

    The match window is ``[write_started_at - window_seconds, now]``
    intersected with the given ``group_id``. ``window_seconds`` defaults
    to :data:`WRITE_WINDOW_SECONDS` (120s). Operators can override per
    call (or globally via ``WIKI_SYNTHESIZER_ATTACH_WINDOW_SECONDS``) —
    longer windows help on slow Gemini days where entity extraction
    overruns the default 2-minute envelope.

    Nodes that already carry a ``memory_id`` are left alone via
    ``coalesce`` so we never overwrite an earlier attach.
    """
    if not memory_id or not group_id:
        return 0
    window = window_seconds if window_seconds is not None else WRITE_WINDOW_SECONDS
    cypher = """
    MATCH (n)
    WHERE n.group_id = $group_id
      AND n.created_at >= datetime($lower_bound)
    SET n.memory_id = coalesce(n.memory_id, $memory_id),
        n.ns_visibility = coalesce(n.ns_visibility, $visibility),
        n.ns_owner = coalesce(n.ns_owner, $owner)
    RETURN count(n) AS patched
    """
    lower_bound = (
        write_started_at.astimezone(timezone.utc) - _delta_seconds(window)
    ).isoformat()
    try:
        async with driver.session() as session:
            result = await session.run(
                cypher,
                group_id=group_id,
                lower_bound=lower_bound,
                memory_id=memory_id,
                visibility=visibility,
                owner=owner_user_id,
            )
            record = await result.single()
            return int(record["patched"]) if record else 0
    except Exception:
        logger.warning(
            "attach_memory_id failed for memory_id=%s group_id=%s (non-fatal)",
            memory_id,
            group_id,
            exc_info=True,
        )
        return 0


async def patch_wiki_path(
    service: Any,
    *,
    node_uuids: Iterable[str],
    wiki_path: str,
    group_id: str | None = None,
    synthesized_at: datetime | None = None,
) -> int:
    """Stamp ``wiki_path`` onto every node whose UUID is in ``node_uuids``.

    Returns the number of nodes updated. Best-effort.

    Takes a ``MemoryService`` (not the raw driver) and dispatches the
    Cypher via ``service._run_on_bridge_async`` so it runs on the loop
    Graphiti's async driver was created on.

    When ``group_id`` is supplied the patch is scoped to nodes in that
    group_id, guarding against (highly improbable) UUID collisions
    across groups and matching the data-isolation model the rest of
    the service relies on. Pass ``None`` to skip the group_id check.
    """
    uuids = [u for u in node_uuids if u]
    if not uuids or not wiki_path:
        return 0
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None:
        return 0
    ts = (synthesized_at or datetime.now(timezone.utc)).isoformat()
    if group_id:
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids AND n.group_id = $group_id
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = datetime($synthesized_at)
        RETURN count(n) AS patched
        """
        params: dict[str, Any] = {
            "uuids": uuids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
            "group_id": group_id,
        }
    else:
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = datetime($synthesized_at)
        RETURN count(n) AS patched
        """
        params = {
            "uuids": uuids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
        }

    async def _inner() -> int:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            return int(record["patched"]) if record else 0

    try:
        return await service._run_on_bridge_async(_inner(), timeout=30.0)
    except Exception:
        logger.warning(
            "patch_wiki_path failed for %d uuids → %s (non-fatal)",
            len(uuids),
            wiki_path,
            exc_info=True,
        )
        return 0


async def patch_wiki_path_by_memory_ids(
    service: Any,
    *,
    memory_ids: Iterable[str],
    wiki_path: str,
    group_id: str | None = None,
    synthesized_at: datetime | None = None,
) -> int:
    """Stamp ``wiki_path`` onto every node whose ``memory_id`` is in the input.

    The category-based synthesizer doesn't go through a community walk to
    collect node UUIDs — it owns the set of source memory IDs directly.
    This helper closes the loop by matching nodes whose ``memory_id``
    property (set at write time by :func:`attach_memory_id`) is in the
    page's source set.

    Same loop-bridge + best-effort semantics as :func:`patch_wiki_path`.
    """
    mids = [m for m in memory_ids if m]
    if not mids or not wiki_path:
        return 0
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None:
        return 0
    ts = (synthesized_at or datetime.now(timezone.utc)).isoformat()
    if group_id:
        cypher = """
        MATCH (n)
        WHERE n.memory_id IN $mids AND n.group_id = $group_id
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = datetime($synthesized_at)
        RETURN count(n) AS patched
        """
        params: dict[str, Any] = {
            "mids": mids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
            "group_id": group_id,
        }
    else:
        cypher = """
        MATCH (n)
        WHERE n.memory_id IN $mids
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = datetime($synthesized_at)
        RETURN count(n) AS patched
        """
        params = {
            "mids": mids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
        }

    async def _inner() -> int:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            return int(record["patched"]) if record else 0

    try:
        return await service._run_on_bridge_async(_inner(), timeout=30.0)
    except Exception:
        logger.warning(
            "patch_wiki_path_by_memory_ids failed for %d memory_ids → %s (non-fatal)",
            len(mids),
            wiki_path,
            exc_info=True,
        )
        return 0


def _delta_seconds(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=seconds)
