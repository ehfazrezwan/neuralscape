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


# ── Kuzu provider support (solo engine) ─────────────────────────────
#
# Kuzu has no label-less property MATCH (columns are per-table), its
# session.run() returns None (reads go through execute_query), and its
# RELATES_TO facts are reified as RelatesToNode_ nodes. Every helper below
# keeps the Neo4j Cypher byte-identical and adds a Kuzu branch; parity notes
# live in docs/neuralscape/29-kuzu-port-inventory.md.

# Node tables Neo4j's label-less `MATCH (n)` would reach. RelatesToNode_ is
# deliberately absent: on Neo4j it is an EDGE, invisible to `MATCH (n)`.
_KUZU_NODE_TABLES = ("Entity", "Episodic", "Community", "Saga")


def _is_kuzu(driver: Any) -> bool:
    try:
        from graphiti_core.driver.driver import GraphProvider

        return getattr(driver, "provider", None) == GraphProvider.KUZU
    except ImportError:
        return False


async def _kuzu_stamp(
    driver: Any,
    *,
    where: str,
    set_clause: str,
    params: dict,
    tables: tuple[str, ...] = _KUZU_NODE_TABLES,
) -> int:
    """Typed per-table stamp loop summing patched counts.

    Callers must not pass None-valued params: KuzuDriver.execute_query strips
    them, which turns a bound `$param` into a binder error.
    """
    total = 0
    for table in tables:
        rows, _, _ = await driver.execute_query(
            f"MATCH (n:{table}) WHERE {where} {set_clause} RETURN count(n) AS patched",
            **params,
        )
        if rows:
            total += int(rows[0]["patched"])
    return total


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
    lower_bound_dt = write_started_at.astimezone(timezone.utc) - _delta_seconds(window)
    if _is_kuzu(driver):
        # Per-table typed stamp; created_at is a Kuzu TIMESTAMP so the bound
        # is passed as a native datetime (Kuzu has no datetime() function).
        # None-valued optional stamps are omitted — identical outcome to
        # coalesce(n.x, null), and Kuzu strips None params anyway.
        sets = ["n.memory_id = coalesce(n.memory_id, $memory_id)"]
        params: dict[str, Any] = {
            "group_id": group_id,
            "lower_bound": lower_bound_dt,
            "memory_id": memory_id,
        }
        if visibility is not None:
            sets.append("n.ns_visibility = coalesce(n.ns_visibility, $visibility)")
            params["visibility"] = visibility
        if owner_user_id is not None:
            sets.append("n.ns_owner = coalesce(n.ns_owner, $owner)")
            params["owner"] = owner_user_id
        try:
            return await _kuzu_stamp(
                driver,
                where="n.group_id = $group_id AND n.created_at >= $lower_bound",
                set_clause="SET " + ", ".join(sets),
                params=params,
            )
        except Exception:
            logger.warning(
                "attach_memory_id failed for memory_id=%s group_id=%s (non-fatal)",
                memory_id,
                group_id,
                exc_info=True,
            )
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
    lower_bound = lower_bound_dt.isoformat()
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


async def attach_source_ref(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    source_ref: dict,
    write_started_at: datetime,
    window_seconds: int | None = None,
) -> int:
    """Link freshly-created graph nodes to the data-layer source they came from.

    For memories ingested from an external connector (Google Drive, Notion,
    an MCP server, …), this MERGEs a ``(:Source)`` node keyed by
    ``connector_id`` + a stable source key (external_id → parent_id →
    connector_id) and connects every node written in this group/window to it
    via ``(:Entity)-[:DERIVED_FROM]->(:Source)``. It also stamps connector
    properties directly on those nodes so a graph search can answer "which
    connector produced this / what else came from the same source" without a
    hop. Best-effort — never raises (mirrors :func:`attach_memory_id`).
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
    window = window_seconds if window_seconds is not None else WRITE_WINDOW_SECONDS
    lower_bound_dt = write_started_at.astimezone(timezone.utc) - _delta_seconds(window)
    if _is_kuzu(driver):
        return await _attach_source_ref_kuzu(
            driver,
            group_id=group_id,
            source_ref=source_ref,
            connector_id=connector_id,
            source_key=source_key,
            lower_bound=lower_bound_dt,
        )
    lower_bound = lower_bound_dt.isoformat()
    cypher = """
    MERGE (s:Source {connector_id: $connector_id, source_key: $source_key})
    SET s.connector_type = $connector_type,
        s.url = coalesce($url, s.url),
        s.title = coalesce($title, s.title),
        s.external_id = coalesce($external_id, s.external_id),
        s.last_synced_at = $last_synced_at
    WITH s
    MATCH (n)
    WHERE n.group_id = $group_id
      AND n.created_at >= datetime($lower_bound)
    SET n.ns_connector_id = coalesce(n.ns_connector_id, $connector_id),
        n.ns_connector_type = coalesce(n.ns_connector_type, $connector_type),
        n.ns_source_url = coalesce(n.ns_source_url, $url)
    MERGE (n)-[:DERIVED_FROM]->(s)
    RETURN count(n) AS patched
    """
    from neo4j.exceptions import TransientError

    try:
        for attempt in range(_SOURCE_ATTACH_RETRIES + 1):
            try:
                async with driver.session() as session:
                    result = await session.run(
                        cypher,
                        connector_id=connector_id,
                        source_key=source_key,
                        connector_type=source_ref.get("connector_type"),
                        url=source_ref.get("url"),
                        title=source_ref.get("title"),
                        external_id=source_ref.get("external_id"),
                        last_synced_at=source_ref.get("last_synced_at"),
                        group_id=group_id,
                        lower_bound=lower_bound,
                    )
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


async def _attach_source_ref_kuzu(
    driver: Any,
    *,
    group_id: str,
    source_ref: dict,
    connector_id: str,
    source_key: str,
    lower_bound: datetime,
) -> int:
    """Kuzu arm of :func:`attach_source_ref`.

    Source rows are keyed on the synthetic single-column PK
    ``key`` = ``<connector_id>::<source_key>`` (Kuzu PKs are single-column;
    Neo4j MERGEs on the pair). No TransientError retry: Kuzu is embedded
    single-writer — there are no lock-cycle aborts to retry. None-valued
    optional props are omitted, matching Neo4j's coalesce($x, s.x)
    keep-existing semantics.
    """
    key = f"{connector_id}::{source_key}"
    src_sets = ["s.connector_id = $connector_id", "s.source_key = $source_key"]
    src_params: dict[str, Any] = {
        "key": key,
        "connector_id": connector_id,
        "source_key": source_key,
    }
    for col in ("connector_type", "url", "title", "external_id", "last_synced_at"):
        val = source_ref.get(col)
        if val is not None:
            src_sets.append(f"s.{col} = ${col}")
            src_params[col] = val
    node_sets = ["n.ns_connector_id = coalesce(n.ns_connector_id, $connector_id)"]
    node_params: dict[str, Any] = {
        "key": key,
        "connector_id": connector_id,
        "group_id": group_id,
        "lower_bound": lower_bound,
    }
    if source_ref.get("connector_type") is not None:
        node_sets.append(
            "n.ns_connector_type = coalesce(n.ns_connector_type, $connector_type)"
        )
        node_params["connector_type"] = source_ref["connector_type"]
    if source_ref.get("url") is not None:
        node_sets.append("n.ns_source_url = coalesce(n.ns_source_url, $url)")
        node_params["url"] = source_ref["url"]
    try:
        await driver.execute_query(
            "MERGE (s:Source {key: $key}) SET " + ", ".join(src_sets), **src_params
        )
        total = 0
        for table in _KUZU_NODE_TABLES:
            rows, _, _ = await driver.execute_query(
                f"MATCH (s:Source {{key: $key}}) "
                f"WITH s MATCH (n:{table}) "
                f"WHERE n.group_id = $group_id AND n.created_at >= $lower_bound "
                f"SET {', '.join(node_sets)} "
                f"MERGE (n)-[:DERIVED_FROM]->(s) "
                f"RETURN count(n) AS patched",
                **node_params,
            )
            if rows:
                total += int(rows[0]["patched"])
        return total
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
    if _is_kuzu(driver):
        where = "n.uuid IN $uuids" + (" AND n.group_id = $group_id" if group_id else "")
        kuzu_params: dict[str, Any] = {
            "uuids": uuids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
        }
        if group_id:
            kuzu_params["group_id"] = group_id
        try:
            return await service._run_on_bridge_async(
                _kuzu_stamp(
                    driver,
                    where=where,
                    set_clause=(
                        "SET n.wiki_path = $wiki_path, "
                        "n.wiki_synthesized_at = $synthesized_at"
                    ),
                    params=kuzu_params,
                ),
                timeout=30.0,
            )
        except Exception:
            logger.warning(
                "patch_wiki_path failed for %d uuids → %s (non-fatal)",
                len(uuids),
                wiki_path,
                exc_info=True,
            )
            return 0
    if group_id:
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids AND n.group_id = $group_id
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
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids
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
    if _is_kuzu(driver):
        where = "n.memory_id IN $mids" + (
            " AND n.group_id = $group_id" if group_id else ""
        )
        kuzu_params: dict[str, Any] = {
            "mids": mids,
            "wiki_path": wiki_path,
            "synthesized_at": ts,
        }
        if group_id:
            kuzu_params["group_id"] = group_id
        try:
            return await service._run_on_bridge_async(
                _kuzu_stamp(
                    driver,
                    where=where,
                    set_clause=(
                        "SET n.wiki_path = $wiki_path, "
                        "n.wiki_synthesized_at = $synthesized_at"
                    ),
                    params=kuzu_params,
                ),
                timeout=30.0,
            )
        except Exception:
            logger.warning(
                "patch_wiki_path_by_memory_ids failed for %d memory_ids → %s (non-fatal)",
                len(mids),
                wiki_path,
                exc_info=True,
            )
            return 0
    if group_id:
        cypher = """
        MATCH (n)
        WHERE n.memory_id IN $mids AND n.group_id = $group_id
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
        cypher = """
        MATCH (n)
        WHERE n.memory_id IN $mids
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
    ts_dt = now or datetime.now(timezone.utc)
    if _is_kuzu(driver):
        return await _invalidate_memory_graph_kuzu(
            driver,
            group_id=group_id,
            memory_id=memory_id,
            superseded_by=superseded_by or "",
            ts_dt=ts_dt,
        )
    ts = ts_dt.isoformat()
    node_cypher = """
    MATCH (n {group_id: $group_id, memory_id: $memory_id})
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


async def _invalidate_memory_graph_kuzu(
    driver: Any,
    *,
    group_id: str,
    memory_id: str,
    superseded_by: str,
    ts_dt: datetime,
) -> int:
    """Kuzu arm of :func:`invalidate_memory_graph`.

    RELATES_TO facts are reified ``RelatesToNode_`` nodes on Kuzu, and the
    exclusively-derived filter (every asserting episode belongs to the
    tombstoned memory) runs in Python rather than relying on unverified
    list-predicate dialect. Semantics mirror the Neo4j statements exactly:
    node marking is its own statement, first and unconditional (PR #125);
    zero stamped episodes → zero edge invalidations (fail-safe); co-asserted
    edges survive.
    """
    ts_iso = ts_dt.isoformat()
    try:
        await _kuzu_stamp(
            driver,
            where="n.group_id = $group_id AND n.memory_id = $memory_id",
            set_clause=(
                "SET n.dream_superseded_by = $superseded_by, "
                "n.dream_invalidated_at = $ts"
            ),
            params={
                "group_id": group_id,
                "memory_id": memory_id,
                "superseded_by": superseded_by,
                "ts": ts_iso,
            },
        )
        eps_rows, _, _ = await driver.execute_query(
            "MATCH (ep:Episodic {group_id: $group_id, memory_id: $memory_id}) "
            "RETURN ep.uuid AS uuid",
            group_id=group_id,
            memory_id=memory_id,
        )
        eps = {r["uuid"] for r in eps_rows}
        if not eps:
            return 0  # fail-safe: no stamped episodes → invalidate nothing
        cand, _, _ = await driver.execute_query(
            "MATCH (r:RelatesToNode_) "
            "WHERE r.group_id = $group_id AND r.invalid_at IS NULL "
            "RETURN r.uuid AS uuid, r.episodes AS episodes",
            group_id=group_id,
        )
        doomed = [
            c["uuid"]
            for c in cand
            if c.get("episodes") and set(c["episodes"]).issubset(eps)
        ]
        if not doomed:
            return 0
        rows, _, _ = await driver.execute_query(
            "MATCH (r:RelatesToNode_) WHERE r.uuid IN $uuids "
            "SET r.invalid_at = $ts, r.expired_at = $ts "
            "RETURN count(r) AS edges",
            uuids=doomed,
            ts=ts_dt,  # graphiti-owned TIMESTAMP columns → native datetime
        )
        return int(rows[0]["edges"]) if rows else 0
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
    if _is_kuzu(driver):
        try:
            return await _kuzu_stamp(
                driver,
                where="n.group_id = $group_id AND n.memory_id IN $memory_ids",
                set_clause="SET n.dream_path = $dream_path, n.dreamt_at = $ts",
                params={
                    "group_id": group_id,
                    "memory_ids": ids,
                    "dream_path": dream_path,
                    "ts": ts,
                },
            )
        except Exception:
            logger.warning(
                "patch_dream_path_by_memory_ids failed for %s (non-fatal)",
                group_id,
                exc_info=True,
            )
            return 0
    cypher = """
    MATCH (n {group_id: $group_id})
    WHERE n.memory_id IN $memory_ids
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
