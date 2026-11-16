"""Graph-contradiction supersession: tombstone the Qdrant rows whose graph
edges Graphiti just invalidated.

Graphiti's ``add_episode`` contradiction pass invalidates an older
``RELATES_TO`` edge (``invalid_at`` + ``expired_at``) when a newer fact
contradicts it — but the Qdrant row that edge was extracted from stayed live,
so recall kept serving the contradicted fact. This module closes that gap with
a reversible tombstone, reusing the ``metadata.dream_tombstoned`` key every
read path already excludes (no read-side change).

Flow (all best-effort, driven from ``memory/write.py`` behind
``GRAPH_SUPERSESSION_ENABLED``):

1. Probe Neo4j for edges invalidated inside this write's window that the new
   episode did not itself create (``find_contradicted_edges``).
2. Map each edge's source episode(s) to Qdrant rows via
   ``metadata.graph_episode_uuid`` — same owner user only, never the rows just
   written.
3. Decide per row:
   - episode is a single-fact episode (``graph_episode_kind="fact"``, stamped
     by ``enrich_graph``) with ONE live row: supersede if every edge citing
     the episode is dead; if some are still live (one fact yields several
     edges), only when the contradicted edge matches the row at
     ``graph_supersession_partial_min_similarity`` AND beats every live edge
     by ``graph_supersession_margin`` (a compound row whose live edge is as
     close as the dead one is left alone);
   - any other episode (conversation / unstamped, one or several rows): supersede the row
     whose stored dense vector best matches the embedded edge ``fact``, only
     if it clears ``graph_supersession_min_similarity`` and beats the
     runner-up by ``graph_supersession_margin``; otherwise leave all alone.
4. Tombstone via an atomic nested-key ``set_payload(key="metadata")`` (same
   pattern as ``_revive_if_tombstoned`` / the dreaming ``_tombstone``).
   Never deletes; ``_revive_if_tombstoned`` undoes it on re-derivation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from config import settings

logger = logging.getLogger(__name__)

SUPERSEDED_REASON = "graph_contradiction"

# Metadata keys this module stamps on top of ``dream_tombstoned``. Exported so
# the revive path can clear exactly this set.
SUPERSESSION_KEYS = (
    "superseded_by",
    "superseded_by_episode",
    "superseded_reason",
    "superseded_edge_uuids",
)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _row_meta(payload: dict) -> dict:
    meta = payload.get("metadata") or {}
    # mem0 sometimes nests {metadata: {metadata: {...}}}.
    inner = meta.get("metadata")
    if isinstance(inner, dict) and "graph_episode_uuid" not in meta:
        return inner
    return meta


def _rows_by_episode(
    service, owner_user_id: str, episode_uuids: list[str], exclude_ids: set[str]
) -> dict[str, list[tuple[str, dict]]]:
    """Live Qdrant rows (owner only) keyed by ``metadata.graph_episode_uuid``."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    client = service._memory.vector_store.client
    flt = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=owner_user_id)),
            FieldCondition(
                key="metadata.graph_episode_uuid", match=MatchAny(any=episode_uuids)
            ),
        ]
    )
    out: dict[str, list[tuple[str, dict]]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points or []:
            pid = str(pt.id)
            payload = pt.payload or {}
            # Defence in depth: the filter already pins user_id.
            if payload.get("user_id") != owner_user_id or pid in exclude_ids:
                continue
            meta = _row_meta(payload)
            if meta.get("dream_tombstoned"):
                continue  # already hidden
            ep = meta.get("graph_episode_uuid")
            if ep in episode_uuids:
                out.setdefault(ep, []).append((pid, payload))
        if offset is None:
            break
    return out


def _best_row_for_fact(
    service, fact: str, rows: list[tuple[str, dict]]
) -> str | None:
    """Clear-winner row id for ``fact`` among a multi-row episode, else None."""
    min_sim = settings.graph_supersession_min_similarity
    margin = settings.graph_supersession_margin
    try:
        embedder = service._get_memory().embedding_model
        fact_vec = embedder.embed(fact, memory_action="add")
        stored = service._fetch_dense_vectors([pid for pid, _ in rows])
        scored: list[tuple[float, str]] = []
        for pid, payload in rows:
            vec = stored.get(pid)
            if vec is None:
                vec = embedder.embed(payload.get("data") or "", memory_action="add")
            scored.append((_cosine(list(fact_vec), list(vec)), pid))
    except Exception as e:  # noqa: BLE001 — ambiguity ⇒ leave rows alone
        logger.warning(f"Supersession similarity match failed (leaving rows): {e}")
        return None
    scored.sort(reverse=True)
    best, runner_up = scored[0], scored[1] if len(scored) > 1 else (0.0, "")
    if best[0] >= min_sim and (best[0] - runner_up[0]) >= margin:
        return best[1]
    logger.info(
        "Supersession ambiguous for edge fact (best=%.3f runner_up=%.3f); "
        "leaving %d rows untouched",
        best[0], runner_up[0], len(rows),
    )
    return None


def _dead_edge_dominates(
    service,
    row_id: str,
    row_payload: dict,
    dead_fact: str,
    live_facts: list[str],
    cache: dict[str, list[float]] | None = None,
) -> tuple[bool, dict]:
    """Is the contradicted edge the row's MAIN claim, not an incidental one?

    One stored fact routinely yields several edges (e.g. "Alice lives in
    Paris, France" -> "Alice resides in Paris" + "Paris is a city in
    France"). Requiring every edge to be dead would never supersede such a
    row. Instead the dead edge must (1) match the row at
    ``graph_supersession_partial_min_similarity`` and (2) beat every
    still-live edge of the episode by ``graph_supersession_margin``
    (cosine on the stored dense vector). Any embedding failure, missing or
    mismatched vector leaves the row alone. ``cache`` memoises embeddings of
    fact strings for the duration of one supersession call. Returns
    ``(decision, evidence)``.
    """
    margin = settings.graph_supersession_margin
    floor = settings.graph_supersession_partial_min_similarity
    evidence: dict = {"dead_sim": None, "best_live_sim": None, "margin": margin, "floor": floor}
    cache = cache if cache is not None else {}
    try:
        embedder = service._get_memory().embedding_model

        def _embed(text: str) -> list[float]:
            if text not in cache:
                cache[text] = list(embedder.embed(text, memory_action="add"))
            return cache[text]

        row_vec = service._fetch_dense_vectors([row_id]).get(row_id)
        if row_vec is None:
            row_vec = _embed(row_payload.get("data") or "")
        row_vec = list(row_vec)
        dead_vec = _embed(dead_fact)
        if len(dead_vec) != len(row_vec):
            evidence["error"] = f"vector length mismatch ({len(dead_vec)} vs {len(row_vec)})"
            return False, evidence
        dead = _cosine(row_vec, dead_vec)
        live = [
            _cosine(row_vec, _embed(f)) for f in live_facts if f and f.strip()
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Supersession dominance check failed (leaving row): {e}")
        evidence["error"] = str(e)
        return False, evidence
    best_live = max(live) if live else 0.0
    evidence.update(dead_sim=round(dead, 4), best_live_sim=round(best_live, 4))
    return (dead >= floor and (dead - best_live) >= margin), evidence


def supersede_contradicted_rows(
    service,
    *,
    group_id: str,
    owner_user_id: str,
    write_started_at: datetime,
    new_episode_uuid: str | None,
    new_memory_ids: set[str],
    superseded_by: str | None = None,
    trace: list[dict] | None = None,
) -> list[str]:
    """Tombstone rows whose graph edges the write just invalidated.

    ``trace`` (optional, caller-owned list) receives the decision trace:
    probe result, row lookup, per-row evidence. It carries fact text, is
    never logged, and is only surfaced by tests' failure messages.

    Returns the ids tombstoned. Raises on infrastructure errors; the caller
    (``WriteMixin._maybe_supersede_contradicted``) swallows them so a failed
    probe never fails the write.
    """
    from extensions.dreaming.graph_patcher import (
        find_contradicted_edges,
        live_edge_facts_by_episode,
    )

    if trace is None:
        trace = []
    embed_cache: dict[str, list[float]] = {}
    trace.append({
        "stage": "start", "group_id": group_id, "owner": owner_user_id,
        "new_episode": new_episode_uuid, "write_started_at": write_started_at.isoformat(),
    })

    if not new_episode_uuid:
        logger.debug("Supersession skipped: no episode uuid for this write")
        return []

    def _bridge(coro):
        try:
            return service._run_on_bridge(coro, timeout=15.0)
        except Exception:
            coro.close()
            raise

    driver = service._graphiti.driver
    edges = _bridge(
        find_contradicted_edges(
            driver,
            group_id=group_id,
            write_started_at=write_started_at,
            new_episode_uuid=new_episode_uuid,
        )
    ) or []
    trace.append({"stage": "probe", "edges": edges})
    if not edges:
        return []

    episode_uuids = sorted({ep for e in edges for ep in (e.get("episodes") or [])})
    if not episode_uuids:
        return []
    rows = _rows_by_episode(service, owner_user_id, episode_uuids, set(new_memory_ids))
    trace.append({
        "stage": "rows", "lookup_key": "metadata.graph_episode_uuid",
        "episode_uuids": episode_uuids,
        "matched": {ep: [pid for pid, _ in rs] for ep, rs in rows.items()},
    })
    if not rows:
        return []

    live_facts: dict[str, list[str]] | None = None
    # row id -> edge uuids that justify superseding it
    chosen: dict[str, list[str]] = {}
    for edge in edges:
        fact = edge.get("fact") or ""
        for ep in edge.get("episodes") or []:
            ep_rows = rows.get(ep)
            if not ep_rows:
                continue
            # Single-fact episodes (stamped by enrich_graph with
            # graph_episode_kind="fact") ARE their one row, but one fact can
            # yield several edges, so "every edge dead" is too strict: the
            # contradicted edge must dominate the row's still-live edges
            # (_dead_edge_dominates). Anything else (conversation episodes,
            # pre-flag rows) may hold unrelated facts in a lone stamped row
            # (dedup hits are not stamped), so the similarity gate applies.
            single_fact = len(ep_rows) == 1 and (
                _row_meta(ep_rows[0][1]).get("graph_episode_kind") == "fact"
            )
            if single_fact:
                if live_facts is None:
                    live_facts = _bridge(
                        live_edge_facts_by_episode(
                            driver, group_id=group_id, episode_uuids=list(rows)
                        )
                    )
                still_live = live_facts.get(ep, [])
                pid = ep_rows[0][0]
                if still_live:
                    ok, evidence = _dead_edge_dominates(
                        service, pid, ep_rows[0][1], fact, still_live, embed_cache
                    )
                    trace.append({"stage": "decide", "row": pid, "episode": ep,
                                  "mode": "single_fact_partial", "dead_fact": fact,
                                  "live_facts": still_live, "supersede": ok, **evidence})
                    if not ok:
                        continue
                else:
                    trace.append({"stage": "decide", "row": pid, "episode": ep,
                                  "mode": "single_fact_all_dead", "supersede": True})
            else:
                pid = _best_row_for_fact(service, fact, ep_rows)
                trace.append({"stage": "decide", "row": pid, "episode": ep,
                              "mode": "multi_or_unstamped", "rows": [r for r, _ in ep_rows],
                              "supersede": pid is not None})
                if pid is None:
                    continue
            chosen.setdefault(pid, [])
            if edge.get("uuid") and edge["uuid"] not in chosen[pid]:
                chosen[pid].append(edge["uuid"])

    if not chosen:
        return []

    client = service._memory.vector_store.client
    now = datetime.now(timezone.utc).isoformat()
    done: list[str] = []
    failed: list[tuple[str, list[str]]] = []
    for pid, edge_uuids in chosen.items():
        try:
            client.set_payload(
                collection_name=settings.qdrant_collection,
                payload={
                    "dream_tombstoned": True,
                    "superseded_by": superseded_by,
                    "superseded_by_episode": new_episode_uuid,
                    "superseded_reason": SUPERSEDED_REASON,
                    "superseded_at": now,
                    "superseded_edge_uuids": edge_uuids,
                },
                points=[pid],
                key="metadata",
            )
        except Exception:
            failed.append((pid, edge_uuids))
            logger.error(
                "Supersession tombstone FAILED for memory %s (edges %s invalidated "
                "by episode %s); graph and vector store are now inconsistent for "
                "this row until it is re-derived or tombstoned by hand",
                pid, edge_uuids, new_episode_uuid, exc_info=True,
            )
            continue
        done.append(pid)
        logger.info(
            "Superseded memory %s: graph edges %s invalidated by episode %s",
            pid, edge_uuids, new_episode_uuid,
        )
    if failed:
        logger.error("Supersession: %d/%d tombstones failed", len(failed), len(chosen))
    return done
