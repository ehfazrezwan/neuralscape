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

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Deadlock retries for the shared-node MERGE in attach_source_ref: concurrent
# graph jobs from the same file all MERGE the same (:Source) node, and Neo4j
# resolves the lock cycle by aborting one transaction with a TransientError.
# Retrying after a short jittered backoff is the documented client remedy.
_SOURCE_ATTACH_RETRIES = 3

# Indirection so tests can stub the backoff (no real sleeps, no jitter
# nondeterminism) without patching the global asyncio module.
_backoff_sleep = asyncio.sleep


# Window for "freshly written by this task" — we identify newly-created
# nodes by matching their `created_at` against this many seconds before
# the write started. Small enough to avoid attaching memory_id to nodes
# from unrelated concurrent writes; large enough to cover graph-write
# latency under load.
WRITE_WINDOW_SECONDS = 120


# Node kinds Graphiti stores that carry ``group_id`` and that our stamps reach.
# A label-less ``MATCH (n)`` cannot use Graphiti's label-scoped indexes and
# scans every node in the database; these queries name the kinds explicitly.
_NODE_KINDS = ("Entity", "Episodic", "Community")


def _any_kind(where: str) -> str:
    """Cypher ``CALL`` block yielding ``n`` for every node kind matching ``where``.

    One label-scoped ``MATCH`` per kind (so each can use its label index),
    UNION-ed. Replaces the label-less ``MATCH (n) WHERE ...`` scan without
    dropping any kind the old query could reach.
    """
    branches = "\n        UNION\n".join(
        f"        MATCH (n:{kind}) WHERE {where} RETURN n" for kind in _NODE_KINDS
    )
    return f"CALL {{\n{branches}\n    }}"


# The episode's own node plus every entity it mentions: exactly the nodes one
# write produced or touched, whatever else is writing to the group at once.
_EPISODE_NODES_BY_UUID = """
    MATCH (ep:Episodic {group_id: $group_id}) WHERE ep.uuid = $episode_uuid
    OPTIONAL MATCH (ep)-[:MENTIONS]->(e:Entity) WHERE e.group_id = $group_id
    WITH collect(DISTINCT ep) + collect(DISTINCT e) AS nodes
    UNWIND nodes AS n
"""
_EPISODE_NODES_BY_NAME = _EPISODE_NODES_BY_UUID.replace(
    "ep.uuid = $episode_uuid", "ep.name = $episode_name"
)


def _episode_selector(episode_uuid: str | None, episode_name: str | None) -> tuple[str, dict] | None:
    """Cypher prefix + params selecting a write's own nodes, or None (no ref)."""
    if episode_uuid:
        return _EPISODE_NODES_BY_UUID, {"episode_uuid": episode_uuid}
    if episode_name:
        return _EPISODE_NODES_BY_NAME, {"episode_name": episode_name}
    return None


async def attach_memory_id(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    visibility: str | None,
    owner_user_id: str | None,
    write_started_at: datetime,
    window_seconds: int | None = None,
    episode_uuid: str | None = None,
    episode_name: str | None = None,
) -> int:
    """Mark the Graphiti nodes a write produced with their originating memory.

    Returns the number of nodes patched. Best-effort — never raises.

    When the write's episode is known (``episode_uuid`` or the deterministic
    ``episode_name``) the nodes are selected by provenance: the ``Episodic``
    node itself and every ``Entity`` it ``MENTIONS``. Concurrent graph jobs in
    one group therefore never stamp each other's nodes. Only when no episode
    reference is available does this fall back (with a warning) to the legacy
    time window ``[write_started_at - window_seconds, now]`` over the group;
    ``window_seconds`` defaults to :data:`WRITE_WINDOW_SECONDS` and is only
    meaningful on that fallback.

    Properties written on each node:

    * ``memory_id`` — the *first* memory to claim the node (``coalesce``, never
      overwritten). Kept for existing readers (wiki/dream/playbook patchers).
    * ``memory_ids`` — additive list of EVERY memory id whose episode mentions
      the node. One conversation episode backs N stored facts but ``memory_id``
      holds only the first; ``memory_ids`` records them all. Idempotent
      (a memory id is appended once).
    * ``ns_visibility`` / ``ns_owner`` — coalesced, as before.
    """
    if not memory_id or not group_id:
        return 0
    # SET of a list property is read-modify-write: take the node's write lock
    # first (a throwaway property) so two jobs appending to one shared entity
    # serialise instead of losing an update, and drop it before commit.
    set_clause = """
    SET n.ns_attach_lock = true
    WITH n
    SET n.memory_id = coalesce(n.memory_id, $memory_id),
        n.memory_ids = CASE
            WHEN $memory_id IN coalesce(n.memory_ids, []) THEN n.memory_ids
            ELSE coalesce(n.memory_ids, []) + $memory_id
        END,
        n.ns_visibility = coalesce(n.ns_visibility, $visibility),
        n.ns_owner = coalesce(n.ns_owner, $owner)
    REMOVE n.ns_attach_lock
    RETURN count(n) AS patched
    """
    params: dict[str, Any] = {
        "group_id": group_id,
        "memory_id": memory_id,
        "visibility": visibility,
        "owner": owner_user_id,
    }
    selector = _episode_selector(episode_uuid, episode_name)
    if selector is not None:
        prefix, extra = selector
        cypher = prefix + set_clause
        params.update(extra)
    else:
        logger.warning(
            "attach_memory_id: no episode reference for memory_id=%s group_id=%s; "
            "falling back to the creation-time window (may stamp concurrent writes' nodes)",
            memory_id,
            group_id,
        )
        window = window_seconds if window_seconds is not None else WRITE_WINDOW_SECONDS
        params["lower_bound"] = (
            write_started_at.astimezone(timezone.utc) - _delta_seconds(window)
        ).isoformat()
        cypher = (
            _any_kind("n.group_id = $group_id AND n.created_at >= datetime($lower_bound)")
            + set_clause
        )
    try:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
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


async def attach_source_ref(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    source_ref: dict,
    write_started_at: datetime,
    window_seconds: int | None = None,
    episode_uuid: str | None = None,
    episode_name: str | None = None,
) -> int:
    """Link a write's graph nodes to the data-layer source they came from.

    For memories ingested from an external connector (Google Drive, Notion,
    an MCP server, …), this MERGEs a ``(:Source)`` node keyed by
    ``connector_id`` + a stable source key (external_id → parent_id →
    connector_id) and connects every node the write produced to it via
    ``(:Entity)-[:DERIVED_FROM]->(:Source)``. It also stamps connector
    properties directly on those nodes so a graph search can answer "which
    connector produced this / what else came from the same source" without a
    hop. Node selection follows :func:`attach_memory_id` (the write's own
    episode + mentioned entities; the time window only as a logged fallback
    when no episode reference is known). Best-effort — never raises.
    """
    if not source_ref or not group_id:
        return 0
    connector_id = source_ref.get("connector_id")
    if not connector_id:
        return 0
    # MERGE keys must be non-null; pick the most specific stable id available.
    source_key = (
        source_ref.get("external_id")
        or source_ref.get("parent_id")
        or connector_id
    )
    merge_source = """
    MERGE (s:Source {connector_id: $connector_id, source_key: $source_key})
    SET s.connector_type = $connector_type,
        s.url = coalesce($url, s.url),
        s.title = coalesce($title, s.title),
        s.external_id = coalesce($external_id, s.external_id),
        s.last_synced_at = $last_synced_at
    WITH s
    """
    stamp = """
    SET n.ns_connector_id = coalesce(n.ns_connector_id, $connector_id),
        n.ns_connector_type = coalesce(n.ns_connector_type, $connector_type),
        n.ns_source_url = coalesce(n.ns_source_url, $url)
    MERGE (n)-[:DERIVED_FROM]->(s)
    RETURN count(n) AS patched
    """
    params: dict[str, Any] = {
        "connector_id": connector_id,
        "source_key": source_key,
        "connector_type": source_ref.get("connector_type"),
        "url": source_ref.get("url"),
        "title": source_ref.get("title"),
        "external_id": source_ref.get("external_id"),
        "last_synced_at": source_ref.get("last_synced_at"),
        "group_id": group_id,
    }
    selector = _episode_selector(episode_uuid, episode_name)
    if selector is not None:
        prefix, extra = selector
        # ``s`` must survive the WITH in the node selector.
        cypher = merge_source + prefix.replace(
            "WITH collect(DISTINCT ep) + collect(DISTINCT e) AS nodes",
            "WITH s, collect(DISTINCT ep) + collect(DISTINCT e) AS nodes",
        ) + stamp
        params.update(extra)
    else:
        logger.warning(
            "attach_source_ref: no episode reference for memory_id=%s group_id=%s; "
            "falling back to the creation-time window (may stamp concurrent writes' nodes)",
            memory_id,
            group_id,
        )
        window = window_seconds if window_seconds is not None else WRITE_WINDOW_SECONDS
        params["lower_bound"] = (
            write_started_at.astimezone(timezone.utc) - _delta_seconds(window)
        ).isoformat()
        cypher = (
            merge_source
            + _any_kind("n.group_id = $group_id AND n.created_at >= datetime($lower_bound)")
            + stamp
        )
    from neo4j.exceptions import TransientError

    try:
        for attempt in range(_SOURCE_ATTACH_RETRIES + 1):
            try:
                async with driver.session() as session:
                    result = await session.run(cypher, **params)
                    record = await result.single()
                    return int(record["patched"]) if record else 0
            except TransientError:
                if attempt == _SOURCE_ATTACH_RETRIES:
                    raise
                await _backoff_sleep(0.2 * (2**attempt) + random.uniform(0, 0.2))
    except Exception:
        logger.warning(
            "attach_source_ref failed for connector_id=%s group_id=%s (non-fatal)",
            connector_id,
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
        cypher = _any_kind("n.uuid IN $uuids AND n.group_id = $group_id") + """
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = $synthesized_at
        RETURN count(n) AS patched
        """
        params: dict[str, Any] = {
            "uuids": uuids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
            "group_id": group_id,
        }
    else:
        cypher = _any_kind("n.uuid IN $uuids") + """
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = $synthesized_at
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
        cypher = _any_kind("n.memory_id IN $mids AND n.group_id = $group_id") + """
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = $synthesized_at
        RETURN count(n) AS patched
        """
        params: dict[str, Any] = {
            "mids": mids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
            "group_id": group_id,
        }
    else:
        cypher = _any_kind("n.memory_id IN $mids") + """
        SET n.wiki_path = $wiki_path,
            n.wiki_synthesized_at = $synthesized_at
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


# ── Dreaming additions ──────────────────────────────────────────────


async def invalidate_memory_graph(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    superseded_by: str | None = None,
    now: datetime | None = None,
) -> int:
    """Bi-temporally invalidate a memory's graph facts (never delete).

    Fact-scoped (audit 27 #28): each memory write is one Graphiti episode,
    and every RELATES_TO edge records the episode uuids that asserted it in
    ``r.episodes``. Invalidation anchors on the memory's own Episodic
    node(s) — stamped ``memory_id`` at write time by ``attach_memory_id``
    — and stamps ``invalid_at``/``expired_at`` only on edges whose ENTIRE
    episode provenance lies inside that set. Edges co-asserted by any
    other (live) memory's episode survive untouched — the pre-fix
    entity-node adjacency sweep invalidated those too, destroying facts
    asserted by live memories that merely shared an entity.

    Known residual imprecision (NS-layer best effort, no Graphiti change):

    - If the write-time ``memory_id`` stamp missed the Episodic node (a
      graph write slower than the attach window), zero edges are
      invalidated — deliberately fail-safe (a stale-but-filtered edge
      beats destroying live ones).
    - An edge asserted by episodes of N > 1 memories is only invalidated
      when ALL those memories' episodes belong to the tombstoned memory —
      tombstoning each co-asserter in separate sweeps leaves the edge
      live (each pass sees the others' episodes as external evidence).

    Node marking is UNCONDITIONAL (Copilot, PR #125): entity/episodic
    nodes stamped with this ``memory_id`` always get the
    ``dream_superseded_by`` hop marker — a walker convenience mirroring
    the vector-side tombstone, not an invalidation — via its own
    statement issued BEFORE the edge pass, so it can never be skipped by
    edge-scoping outcomes (zero exclusively-derived edges, or the empty
    episode set of the fail-safe case above) or an edge-query failure.

    Returns the number of edges invalidated (0 on any failure — best-effort
    like every helper in this module).
    """
    ts = (now or datetime.now(timezone.utc)).isoformat()
    node_cypher = _any_kind("n.group_id = $group_id AND n.memory_id = $memory_id") + """
    SET n.dream_superseded_by = $superseded_by, n.dream_invalidated_at = $ts
    RETURN count(n) AS nodes
    """
    edge_cypher = """
    OPTIONAL MATCH (ep:Episodic {group_id: $group_id, memory_id: $memory_id})
    WITH collect(ep.uuid) AS eps
    OPTIONAL MATCH ()-[r:RELATES_TO {group_id: $group_id}]->()
    WHERE r.invalid_at IS NULL
      AND size(eps) > 0
      AND size(coalesce(r.episodes, [])) > 0
      AND all(x IN coalesce(r.episodes, []) WHERE x IN eps)
    SET r.invalid_at = $ts, r.expired_at = $ts
    RETURN count(r) AS edges
    """
    try:
        async with driver.session() as session:
            await session.run(
                node_cypher,
                group_id=group_id,
                memory_id=memory_id,
                ts=ts,
                superseded_by=superseded_by or "",
            )
            cursor = await session.run(
                edge_cypher,
                group_id=group_id,
                memory_id=memory_id,
                ts=ts,
            )
            records = await cursor.data()
            return int(records[0]["edges"]) if records else 0
    except Exception:
        logger.warning(
            "invalidate_memory_graph failed for %s/%s (non-fatal)",
            group_id,
            memory_id,
            exc_info=True,
        )
        return 0


async def patch_dream_path_by_memory_ids(
    driver: Any,
    *,
    memory_ids: Iterable[str],
    dream_path: str,
    group_id: str,
) -> int:
    """Stamp ``dream_path`` on every node whose memory contributed to a
    dream-diary page — the dreaming analog of ``patch_wiki_path_by_memory_ids``."""
    ids = [m for m in memory_ids if m]
    if not ids:
        return 0
    ts = datetime.now(timezone.utc).isoformat()
    cypher = _any_kind("n.group_id = $group_id AND n.memory_id IN $memory_ids") + """
    SET n.dream_path = $dream_path, n.dreamt_at = $ts
    RETURN count(n) AS patched
    """
    try:
        async with driver.session() as session:
            cursor = await session.run(
                cypher,
                group_id=group_id,
                memory_ids=ids,
                dream_path=dream_path,
                ts=ts,
            )
            records = await cursor.data()
            return int(records[0]["patched"]) if records else 0
    except Exception:
        logger.warning(
            "patch_dream_path_by_memory_ids failed for %s (non-fatal)",
            group_id,
            exc_info=True,
        )
        return 0


async def find_contradicted_edges(
    driver: Any,
    *,
    group_id: str,
    write_started_at: datetime,
    new_episode_uuid: str,
) -> list[dict]:
    """RELATES_TO edges a graph write just invalidated (supersession probe).

    Graphiti's ``add_episode`` contradiction pass sets BOTH ``invalid_at``
    (the new edge's ``valid_at``) and ``expired_at`` (now) on an older edge.
    Requiring ``invalid_at`` excludes the episode-delete cascade, which only
    sets ``expired_at``. ``expired_at >= write_started_at`` bounds the probe
    to this write's window. The edge must cite SOME episode other than the new
    one: an edge born expired by the new episode cites only it (drop), but when
    Graphiti resolves the new episode's extracted edge as a duplicate of an
    existing one it APPENDS the new episode to that edge's ``episodes`` and
    then invalidates it, so the invalidated old edge cites ``[old, new]``
    (the old ``NOT new IN episodes`` predicate wrongly dropped exactly that
    case). ``episodes`` in the result excludes the new episode.

    Returns ``[{"uuid", "fact", "episodes": [...]}]``. Best-effort — logs and
    returns ``[]`` on any driver error (never raises).
    """
    if not group_id or not new_episode_uuid:
        return []
    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.group_id = $group_id
      AND r.invalid_at IS NOT NULL
      AND r.expired_at >= datetime($since)
      AND any(ep IN coalesce(r.episodes, []) WHERE ep <> $new_episode_uuid)
    RETURN r.uuid AS uuid, r.fact AS fact, coalesce(r.episodes, []) AS episodes
    """
    try:
        async with driver.session() as session:
            result = await session.run(
                cypher,
                group_id=group_id,
                since=write_started_at.astimezone(timezone.utc).isoformat(),
                new_episode_uuid=new_episode_uuid,
            )
            rows = list(await result.data())
            for row in rows:
                row["episodes"] = [
                    ep for ep in (row.get("episodes") or []) if ep != new_episode_uuid
                ]
            return rows
    except Exception:
        logger.warning(
            "find_contradicted_edges failed for group_id=%s (non-fatal)",
            group_id,
            exc_info=True,
        )
        return []


async def count_live_edges_by_episode(
    driver: Any,
    *,
    group_id: str,
    episode_uuids: list[str],
) -> dict[str, int]:
    """Live (neither invalid nor expired) RELATES_TO edge count per episode.

    An episode that appears in ``episode_uuids`` but has no live edge maps to
    ``0``. Raises on driver errors — the caller decides how to degrade (the
    supersession hook treats a failure as "leave rows alone").
    """
    counts = {ep: 0 for ep in episode_uuids}
    if not group_id or not episode_uuids:
        return counts
    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.group_id = $group_id
      AND r.invalid_at IS NULL
      AND r.expired_at IS NULL
    UNWIND [ep IN coalesce(r.episodes, []) WHERE ep IN $episode_uuids] AS ep
    RETURN ep AS episode_uuid, count(r) AS live
    """
    async with driver.session() as session:
        result = await session.run(
            cypher, group_id=group_id, episode_uuids=list(episode_uuids)
        )
        for row in await result.data():
            counts[row["episode_uuid"]] = int(row["live"])
    return counts


async def episode_edge_counts(
    driver: Any,
    *,
    group_id: str,
    episode_uuid: str,
) -> dict[str, int]:
    """``{"total", "live"}`` RELATES_TO edge counts citing ``episode_uuid``.

    ``live`` = neither ``invalid_at`` nor ``expired_at`` set. Used by the
    stale-graph check on dedup hits: an episode whose edges ALL died
    (``total > 0 and live == 0``) means Graphiti has since contradicted what
    the row says. Raises on driver errors (the caller fails safe).
    """
    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.group_id = $group_id
      AND $episode_uuid IN coalesce(r.episodes, [])
    RETURN count(r) AS total,
           sum(CASE WHEN r.invalid_at IS NULL AND r.expired_at IS NULL
                    THEN 1 ELSE 0 END) AS live
    """
    async with driver.session() as session:
        result = await session.run(cypher, group_id=group_id, episode_uuid=episode_uuid)
        rows = await result.data()
    row = rows[0] if rows else {}
    return {"total": int(row.get("total") or 0), "live": int(row.get("live") or 0)}
