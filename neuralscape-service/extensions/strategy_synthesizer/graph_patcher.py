"""Neo4j back-reference patch: stamp ``strategy_playbook_path`` on the graph
nodes that contributed to a synthesized playbook.

Mirrors ``wiki_synthesizer.graph_patcher.patch_wiki_path_by_memory_ids`` — match
nodes by the ``memory_id`` set the playbook was built from (set at write time by
``attach_memory_id``) and stamp the playbook back-reference. Best-effort: logs
and swallows errors, never raises (a failed hop must not fail synthesis).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


async def patch_playbook_path_by_memory_ids(
    service: Any,
    *,
    memory_ids: Iterable[str],
    playbook_path: str,
    synthesized_at: datetime | None = None,
) -> int:
    """Stamp ``strategy_playbook_path`` onto every node whose ``memory_id`` is in the input.

    Runs the Cypher on Graphiti's bridge loop (the async driver is bound to it).
    Returns the number of nodes patched.
    """
    mids = [m for m in memory_ids if m]
    if not mids or not playbook_path:
        return 0
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None:
        return 0
    ts = (synthesized_at or datetime.now(timezone.utc)).isoformat()
    cypher = """
    MATCH (n)
    WHERE n.memory_id IN $mids
    SET n.strategy_playbook_path = $playbook_path,
        n.strategy_synthesized_at = $synthesized_at
    RETURN count(n) AS patched
    """

    async def _inner() -> int:
        async with driver.session() as session:
            result = await session.run(
                cypher, mids=mids, playbook_path=playbook_path, synthesized_at=ts
            )
            record = await result.single()
            return int(record["patched"]) if record else 0

    try:
        return await service._run_on_bridge_async(_inner(), timeout=30.0)
    except Exception:
        logger.warning(
            "patch_playbook_path_by_memory_ids failed for %d memory_ids → %s (non-fatal)",
            len(mids),
            playbook_path,
            exc_info=True,
        )
        return 0
