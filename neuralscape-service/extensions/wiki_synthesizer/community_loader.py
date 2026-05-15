"""Graphiti community queries for the wiki synthesizer.

A *community* in Graphiti is a cluster of entity nodes connected by a
``HAS_MEMBER`` edge from a ``Community`` node. The synthesizer treats
each community as a "topic" — one wiki page per (category, community)
pair.

This module is intentionally Cypher-direct rather than going through
Graphiti's Python API: we only need read-only enumeration of communities
and their members, and a direct Cypher query is both faster and keeps
the synthesizer decoupled from the Graphiti subtree's internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Community:
    """A topical cluster found inside a single ``group_id``."""

    uuid: str
    name: str
    summary: str = ""
    member_node_uuids: list[str] = field(default_factory=list)
    member_memory_ids: list[str] = field(default_factory=list)


async def load_communities(driver: Any, *, group_id: str) -> list[Community]:
    """Return every community within ``group_id`` with its member memory IDs.

    Members come from two hops:

    1. ``(Community)-[:HAS_MEMBER]->(EntityNode)`` — the community's
       direct entity members.
    2. Each entity's ``memory_id`` attribute, which the
       :mod:`graph_patcher` attached at memory-write time.

    Entities without a ``memory_id`` (older writes, or writes that
    landed before the patcher was wired) are still listed under
    ``member_node_uuids`` — the synthesizer can decide whether to load
    their content via a fallback path or skip them.
    """
    if not group_id:
        return []

    cypher = """
    MATCH (c:Community {group_id: $group_id})
    OPTIONAL MATCH (c)-[:HAS_MEMBER]->(m)
    RETURN c.uuid AS uuid,
           c.name AS name,
           c.summary AS summary,
           collect({uuid: m.uuid, memory_id: m.memory_id}) AS members
    """
    try:
        async with driver.session() as session:
            result = await session.run(cypher, group_id=group_id)
            records = await result.data()
    except Exception:
        logger.warning(
            "load_communities failed for group_id=%s (non-fatal — synthesis will skip)",
            group_id,
            exc_info=True,
        )
        return []

    communities: list[Community] = []
    for r in records:
        members = r.get("members") or []
        node_uuids: list[str] = []
        memory_ids: list[str] = []
        for m in members:
            if not m:
                continue
            uuid = m.get("uuid")
            mem_id = m.get("memory_id")
            if uuid:
                node_uuids.append(uuid)
            if mem_id:
                memory_ids.append(mem_id)
        communities.append(
            Community(
                uuid=r.get("uuid") or "",
                name=r.get("name") or "(unnamed)",
                summary=r.get("summary") or "",
                member_node_uuids=node_uuids,
                # de-dupe while preserving order (mem_ids appear once per node)
                member_memory_ids=list(dict.fromkeys(memory_ids)),
            )
        )
    return communities
