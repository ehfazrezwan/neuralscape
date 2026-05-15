"""Neo4j Cypher helpers for attaching back-references between memories,
graph nodes, and synthesized wiki pages.

Two responsibilities:

1. **At write time** — after a memory write produces Graphiti entity
   nodes, mark those nodes with the originating ``memory_id`` and the
   memory's ``visibility`` / ``owner_user_id``. This is the link the
   wiki synthesizer later uses to walk from a community back to its
   source memories.

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
) -> int:
    """Mark recently-created Graphiti nodes with their originating memory.

    Returns the number of nodes patched. Best-effort — never raises.

    The match window is ``[write_started_at - WRITE_WINDOW_SECONDS, now]``
    intersected with the given ``group_id``. Nodes that already carry a
    ``memory_id`` are left alone (``coalesce``).
    """
    if not memory_id or not group_id:
        return 0
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
        write_started_at.astimezone(timezone.utc)
        - _delta_seconds(WRITE_WINDOW_SECONDS)
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
    driver: Any,
    *,
    node_uuids: Iterable[str],
    wiki_path: str,
    synthesized_at: datetime | None = None,
) -> int:
    """Stamp ``wiki_path`` onto every node whose UUID is in ``node_uuids``.

    Returns the number of nodes updated. Best-effort.
    """
    uuids = [u for u in node_uuids if u]
    if not uuids or not wiki_path:
        return 0
    ts = (synthesized_at or datetime.now(timezone.utc)).isoformat()
    cypher = """
    MATCH (n)
    WHERE n.uuid IN $uuids
    SET n.wiki_path = $wiki_path,
        n.wiki_synthesized_at = datetime($synthesized_at)
    RETURN count(n) AS patched
    """
    try:
        async with driver.session() as session:
            result = await session.run(
                cypher,
                uuids=uuids,
                wiki_path=wiki_path,
                synthesized_at=ts,
            )
            record = await result.single()
            return int(record["patched"]) if record else 0
    except Exception:
        logger.warning(
            "patch_wiki_path failed for %d uuids → %s (non-fatal)",
            len(uuids),
            wiki_path,
            exc_info=True,
        )
        return 0


def _delta_seconds(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=seconds)
