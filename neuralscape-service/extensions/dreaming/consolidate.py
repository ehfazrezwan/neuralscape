"""LIGHT staging + DEEP consolidation for the dreaming sweep.

LIGHT (`stage_pool`) scrolls one pool's memories out of Qdrant, filters to
new/changed-since-last-dream (plus low-strength/old rows offered as prune/
reframe candidates), attaches trace aggregates + scores, and returns the
staged batch. No writes.

DEEP (`decide` + `apply_actions`) runs ONE consolidation LLM call over the
staged batch, parses the strict-JSON action list, splits it by the hybrid
adoption posture (reversible → apply; destructive below the confidence
gate → report only), and applies:

- merge            → survivor rewritten (fold-in), losers tombstoned +
                     graph-invalidated (bi-temporal, reversible)
- invalidate       → tombstone + graph invalid_at (reversible)
- prune            → gate-cleared secrets: hard-delete (the one
                     irreversible case, confidence-gated like every other
                     destructive action); everything else — including
                     below-gate secret claims — tombstone + invalidate
- rewrite          → content updated in place (re-embedded)
- temporal_reframe → content updated in place (re-embedded)

Tombstoned rows stay in Qdrant with ``metadata.dream_tombstoned=true`` and
``metadata.superseded_by``; the search path excludes them (must_not) but a
tombstone can always be lifted by clearing the flag. The hot path never
does any of this — consolidation authority lives here alone (§4.1).

Pool keys: ``shared``, ``shared--project--<pid>``, ``user--<uid>``,
``user--<uid>--project--<pid>`` — matching the graph group_id convention,
so gate/lock keys and Cypher group_id line up 1:1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .prompts import CONSOLIDATION_PROMPT, parse_json_response, render_memories_block
from .scoring import score_memory
from .traces import read_aggregates, read_dynamics

logger = logging.getLogger(__name__)

_DESTRUCTIVE = {"invalidate", "prune"}
_REVERSIBLE = {"merge", "rewrite", "temporal_reframe"}


@dataclass(slots=True)
class PoolBatch:
    pool: str
    group_id: str
    visibility: str          # "shared" | "private"
    owner_user_id: str | None
    project_id: str | None
    memories: list[dict] = field(default_factory=list)
    new_count: int = 0       # new/changed since last dream (volume gate input)


@dataclass(slots=True)
class ActionResult:
    applied: list[dict] = field(default_factory=list)
    reported: list[dict] = field(default_factory=list)   # shadow trial: not applied
    errors: list[str] = field(default_factory=list)


# ── Pool enumeration + LIGHT staging ────────────────────────────────


def pool_key(*, visibility: str, owner_user_id: str | None, project_id: str | None) -> str:
    """Derive the pool key for a memory row (mirrors graph group_id)."""
    if visibility == "shared":
        return f"shared--project--{project_id}" if project_id else "shared"
    uid = owner_user_id or "unknown"
    return f"user--{uid}--project--{project_id}" if project_id else f"user--{uid}"


def enumerate_pools(service, *, batch_size: int = 500) -> dict[str, PoolBatch]:
    """One Qdrant scroll over the collection, grouped into pools.

    ``standard``-tier memories are excluded entirely — authoritative
    dictator content is read-only to dreaming (§4.2). Rows already
    tombstoned by a prior sweep are skipped (they are historical record).
    """
    from qdrant_client.models import Filter

    from config import settings as core_settings

    client = service._memory.vector_store.client
    pools: dict[str, PoolBatch] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=core_settings.qdrant_collection,
            scroll_filter=Filter(must=[]),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            meta = payload.get("metadata", {}) or {}
            if isinstance(meta.get("metadata"), dict):  # mem0 double-wrap
                meta = meta["metadata"]
            visibility = meta.get("visibility") or "private"
            if visibility == "standard":
                continue  # authoritative tier: read-only to dreaming
            if meta.get("dream_tombstoned"):
                continue  # already consolidated away
            owner = meta.get("owner_user_id") or payload.get("user_id")
            project_id = meta.get("project_id")
            is_shared = visibility == "shared"
            key = pool_key(
                visibility="shared" if is_shared else "private",
                owner_user_id=None if is_shared else owner,
                project_id=project_id,
            )
            batch = pools.get(key)
            if batch is None:
                batch = PoolBatch(
                    pool=key,
                    group_id=key,
                    visibility="shared" if visibility == "shared" else "private",
                    owner_user_id=None if visibility == "shared" else owner,
                    project_id=project_id,
                )
                pools[key] = batch
            batch.memories.append(
                {
                    "memory_id": str(getattr(point, "id", "") or ""),
                    "content": payload.get("data", "") or "",
                    "category": meta.get("category"),
                    "visibility": visibility,
                    "owner_user_id": owner,
                    "project_id": project_id,
                    "scope": meta.get("scope"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "confidence": meta.get("confidence"),
                    "concepts": meta.get("concepts"),
                    "related_memory_ids": meta.get("related_memory_ids"),
                    "source_type": meta.get("source_type"),
                    "observation_type": meta.get("observation_type"),
                    "hash": payload.get("hash"),
                    # Reinforcement counter (A2): how many times this fact was
                    # re-derived/deduped onto this row. Feeds promotion scoring.
                    "times_derived": meta.get("times_derived"),
                }
            )
        if offset is None:
            break
    return pools


def _parse_ts(value) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def stage_pool(
    batch: PoolBatch,
    redis,
    *,
    last_dreamt_at: float,
    max_memories: int,
    strength_half_life_days: float,
    prune_strength_threshold: float,
    dynamics_enabled: bool = True,
) -> PoolBatch:
    """LIGHT: filter + score one pool's staged batch in place.

    Keeps (a) everything new/changed since the last dream, and (b) older
    rows whose retention strength fell below the prune threshold or that
    carry a passed future-date (reframe candidates surface via the LLM
    reading their content — we just make sure old rows are visible to it).
    Dream-authored memories get two guards: *fresh* ones are excluded from
    staging and from the volume count entirely (feedback-loop guard, spec
    §9), while *older* ones may re-enter via the weak-retention path so
    stale insights can decay out (PRUNE/INVALIDATE) — but they are never
    merge material (``decide`` enforces that).

    When ``dynamics_enabled``, retention strength is salience-dynamics-
    driven for memories with a recorded state (A4) — so a frequently
    co-recalled pair resists the weak-retention path while an untouched
    sibling decays into it. Low strength only NOMINATES for prune here;
    the LLM pass and the confidence gate still stand (§A4 guardrail 2).
    """
    now = time.time()
    ids = [m["memory_id"] for m in batch.memories]
    traces = read_aggregates(redis, ids)
    dynamics_states = read_dynamics(redis, ids) if dynamics_enabled else {}

    staged: list[dict] = []
    new_count = 0
    for mem in batch.memories:
        scores = score_memory(
            mem, traces, now=now, strength_half_life_days=strength_half_life_days,
            dynamics_states=dynamics_states,
        )
        mem.update(scores)
        changed_at = max(_parse_ts(mem.get("created_at")), _parse_ts(mem.get("updated_at")))
        is_new = changed_at > last_dreamt_at
        if mem.get("source_type") == "dream" and is_new:
            # Freshly dreamt insights neither re-enter the cycle nor count
            # toward the volume gate — otherwise a sweep's own writes would
            # trigger the next sweep (self-sustaining feedback loop).
            continue
        if is_new:
            new_count += 1
        if is_new or mem["retention_strength"] < prune_strength_threshold:
            staged.append(mem)

    # newest first; cap the batch so huge pools consolidate incrementally
    staged.sort(key=lambda m: _parse_ts(m.get("created_at")), reverse=True)
    batch.memories = staged[:max_memories]
    batch.new_count = new_count
    return batch


# ── DEEP: decide + apply ────────────────────────────────────────────


async def decide(batch: PoolBatch, llm_call) -> list[dict]:
    """Run the consolidation decision pass. Returns validated actions."""
    if not batch.memories:
        return []
    now = datetime.now(timezone.utc)
    prompt = CONSOLIDATION_PROMPT.format(
        today=now.date().isoformat(),
        year=now.year,
        memories_block=render_memories_block(batch.memories),
    )
    raw = await llm_call(prompt)
    actions = parse_json_response(raw, key="actions")
    known_ids = {m["memory_id"] for m in batch.memories}
    dream_ids = {
        m["memory_id"] for m in batch.memories if m.get("source_type") == "dream"
    }
    valid: list[dict] = []
    for act in actions:
        a_type = act.get("type")
        mids = [m for m in (act.get("memory_ids") or []) if m in known_ids]
        if a_type == "merge":
            # Enforce the prompt contract in code: dream-authored insights
            # are never merge material (re-consolidating consolidations is
            # the feedback loop the staging guard exists to prevent). They
            # remain valid PRUNE/INVALIDATE targets.
            mids = [m for m in mids if m not in dream_ids]
        if a_type not in (_DESTRUCTIVE | _REVERSIBLE) or not mids:
            continue
        act["memory_ids"] = mids
        if a_type == "merge" and (
            len(mids) < 2 or act.get("survivor_id") not in mids
        ):
            continue
        if a_type in ("rewrite", "temporal_reframe", "merge") and not (
            act.get("content") or ""
        ).strip():
            continue
        try:
            act["confidence"] = max(0.0, min(1.0, float(act.get("confidence", 0.0))))
        except (TypeError, ValueError):
            act["confidence"] = 0.0
        valid.append(act)
    return valid


def split_by_posture(actions: list[dict], *, auto_apply_confidence: float) -> tuple[list[dict], list[dict]]:
    """Hybrid adoption: (to_apply, to_report).

    Reversible actions always apply. Destructive actions apply only at or
    above the confidence gate — below it they are reported (shadow trial).

    Secret claims (audit 27 #25): ``contains_secret`` is an unvalidated LLM
    field, and the secret prune is the ONE irreversible primitive — so it
    must pass the same confidence gate as every other destructive action.
    A below-gate secret claim is still acted on (a possible credential is
    never left fully live) but *downgraded* to the reversible path:
    ``apply_actions`` tombstones instead of hard-deleting and records the
    claim in metadata so a later sweep can re-examine it.
    """
    to_apply: list[dict] = []
    to_report: list[dict] = []
    for act in actions:
        if act["type"] in _REVERSIBLE:
            to_apply.append(act)
        elif act["confidence"] >= auto_apply_confidence:
            to_apply.append(act)
        elif act.get("contains_secret"):
            act["secret_claim_downgraded"] = True
            to_apply.append(act)
        else:
            to_report.append(act)
    return to_apply, to_report


def reconcile_batch(batch: PoolBatch, applied: list[dict]) -> int:
    """Fold applied actions back into the staged batch's in-memory dicts.

    ``apply_actions`` writes to Qdrant/the graph but the batch dicts a
    later pass reads (the librarian's topic pages and essential-story
    lines) would otherwise still show the pre-consolidation view of the
    same sweep: merge losers, invalidated and pruned rows as live, merge
    survivors with their unfolded text. Marks consumed rows
    ``dream_tombstoned`` and updates rewritten content in place; returns
    the number of rows tombstoned.
    """
    tombstoned: set[str] = set()
    rewritten: dict[str, str] = {}
    for act in applied:
        a_type = act.get("type")
        ids = act.get("memory_ids") or []
        if a_type == "merge":
            survivor = act.get("survivor_id")
            tombstoned.update(m for m in ids if m != survivor)
            if survivor and (act.get("content") or "").strip():
                rewritten[survivor] = act["content"]
        elif a_type in ("invalidate", "prune"):
            tombstoned.update(ids)
        elif a_type in ("rewrite", "temporal_reframe"):
            if (act.get("content") or "").strip():
                rewritten.update({m: act["content"] for m in ids})
    count = 0
    for mem in batch.memories:
        mid = mem.get("memory_id")
        if mid in tombstoned:
            mem["dream_tombstoned"] = True
            count += 1
        elif mid in rewritten:
            mem["content"] = rewritten[mid]
    return count


async def apply_actions(
    service,
    batch: PoolBatch,
    actions: list[dict],
    *,
    dry_run: bool,
) -> ActionResult:
    """Apply consolidation actions to Qdrant + the graph.

    Every sync primitive (Qdrant round-trips, embedding calls, deletes)
    runs via ``asyncio.to_thread`` — the sweep must never starve the host
    event loop (the API's /health endpoint shares it, and autoheal
    restarts the container when health stalls).
    """
    import asyncio

    result = ActionResult()
    for act in actions:
        try:
            if dry_run:
                result.applied.append({**act, "dry_run": True})
                continue
            a_type = act["type"]
            if a_type == "merge":
                survivor = act["survivor_id"]
                losers = [m for m in act["memory_ids"] if m != survivor]
                # Reinforcement-aware merge (A2): the survivor's counter
                # becomes the SUM of every merged row's times_derived —
                # folding N observations into one row must not forget that
                # the fact was observed N times. Read live from Qdrant (not
                # the staged batch) so write-path bumps since staging count.
                merged_times_derived = await asyncio.to_thread(
                    _sum_times_derived, service, act["memory_ids"]
                )
                # A1 provenance: the survivor's derived_from records the folded-in
                # loser ids (unioned with any prior premises, same upsert as the
                # content rewrite so there's no read-modify-write race). The
                # summed times_derived rides along in the same upsert.
                await asyncio.to_thread(
                    _rewrite_content, service, survivor, act["content"],
                    derived_from_add=losers,
                    extra_meta={"times_derived": merged_times_derived},
                )
                for mid in losers:
                    await asyncio.to_thread(_tombstone, service, mid, superseded_by=survivor)
                    await _graph_invalidate(service, batch.group_id, mid, superseded_by=survivor)
            elif a_type == "invalidate":
                superseded_by = act.get("superseded_by_id")
                for mid in act["memory_ids"]:
                    await asyncio.to_thread(_tombstone, service, mid, superseded_by=superseded_by)
                    await _graph_invalidate(
                        service, batch.group_id, mid, superseded_by=superseded_by
                    )
            elif a_type == "prune":
                for mid in act["memory_ids"]:
                    if act.get("contains_secret") and not act.get("secret_claim_downgraded"):
                        # The one irreversible case: gate-cleared secrets must
                        # not persist, not even as tombstones.
                        await asyncio.to_thread(service.delete_memory, mid)
                    else:
                        # Below-gate secret claims ride the reversible path
                        # (audit 27 #25): tombstone + record the claim for a
                        # later sweep to re-examine at higher confidence.
                        secret_claim = None
                        if act.get("contains_secret"):
                            secret_claim = {
                                "contains_secret": True,
                                "confidence": act.get("confidence"),
                                "claimed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        await asyncio.to_thread(
                            _tombstone, service, mid, superseded_by=None,
                            pruned=True, secret_claim=secret_claim,
                        )
                        await _graph_invalidate(service, batch.group_id, mid, superseded_by=None)
            elif a_type in ("rewrite", "temporal_reframe"):
                for mid in act["memory_ids"]:
                    await asyncio.to_thread(
                        _rewrite_content, service, mid, act["content"],
                        reframed=(a_type == "temporal_reframe"),
                    )
            result.applied.append(act)
        except Exception as exc:
            logger.exception("dream action failed: %s", act.get("type"))
            result.errors.append(f"{act.get('type')}:{act.get('memory_ids')}: {exc.__class__.__name__}")
    return result


# ── Apply primitives (Qdrant + graph) ───────────────────────────────


def _rewrite_content(
    service,
    memory_id: str,
    new_content: str,
    *,
    reframed: bool = False,
    derived_from_add: list[str] | None = None,
    extra_meta: dict | None = None,
) -> None:
    """Update a memory's text in place: re-embed + overwrite payload data.

    Atomic from the reader's perspective (single upsert). ``updated_at``
    and the dreaming provenance markers ride along in metadata.
    ``derived_from_add`` (MERGE) unions premise ids into the survivor's
    ``metadata.derived_from`` — read-merge-write like ``_tombstone``, but
    folded into this upsert so provenance and content land atomically.
    Any ``extra_meta`` patch (e.g. the merge branch's summed
    ``times_derived``) lands in the same upsert too.
    """
    from config import settings as core_settings

    m = service._get_memory()
    client = m.vector_store.client
    points = client.retrieve(
        collection_name=core_settings.qdrant_collection,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        raise ValueError(f"memory {memory_id} not found for rewrite")
    payload = dict(points[0].payload or {})
    meta = dict(payload.get("metadata") or {})
    meta["dream_rewritten_at"] = datetime.now(timezone.utc).isoformat()
    if reframed:
        meta["dream_temporal_reframed"] = True
    if derived_from_add:
        existing = meta.get("derived_from") or []
        # union, order-preserving (existing premises first), dedup'd
        meta["derived_from"] = list(dict.fromkeys([*existing, *derived_from_add]))
    if extra_meta:
        meta.update(extra_meta)
    payload["metadata"] = meta
    payload["data"] = new_content
    # Audit 27 #6: the content hash MUST follow the content. A stale hash
    # corrupts write-path dedup (a re-store of the new text isn't caught;
    # a re-store of the OLD text wrongly short-circuits onto this row) and
    # the exact-dedup cron, which groups by hash and hard-deletes
    # "duplicates". Same helper as store_raw — never a re-implementation.
    from memory_service import content_hash

    payload["hash"] = content_hash(new_content)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    vector = m.embedding_model.embed(new_content, memory_action="update")
    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name=core_settings.qdrant_collection,
        points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
    )


def _tombstone(
    service,
    memory_id: str,
    *,
    superseded_by: str | None,
    pruned: bool = False,
    secret_claim: dict | None = None,
) -> None:
    """Mark a Qdrant row consolidated-away without deleting it.

    The search path excludes ``metadata.dream_tombstoned=true`` rows; the
    row (and its graph history) remains for audit/reversal.

    ``secret_claim`` (audit 27 #25) records a below-confidence-gate
    ``contains_secret`` claim on the tombstone so a later sweep can
    re-examine it instead of the claim being silently dropped.
    """
    from config import settings as core_settings

    client = service._memory.vector_store.client
    now = datetime.now(timezone.utc).isoformat()
    patch = {
        "dream_tombstoned": True,
        "superseded_at": now,
    }
    if superseded_by:
        patch["superseded_by"] = superseded_by
    if pruned:
        patch["dream_pruned"] = True
    if secret_claim:
        patch["dream_secret_claim"] = secret_claim
    client.set_payload(
        collection_name=core_settings.qdrant_collection,
        payload={"metadata": _merged_metadata(client, memory_id, patch)},
        points=[memory_id],
    )


def _times_derived_of(meta: dict | None) -> int:
    """``times_derived`` of a metadata dict — absent/invalid ⇒ 1 (seen once)."""
    try:
        return max(1, int((meta or {}).get("times_derived") or 1))
    except (TypeError, ValueError):
        return 1


def _sum_times_derived(service, memory_ids: list[str]) -> int:
    """Sum the reinforcement counters of a merge set (A2).

    Rows that can't be retrieved still count as 1 — a merge decision proves
    each id was a live observation moments ago, and undercounting would
    silently shrink the survivor's reinforcement.
    """
    from config import settings as core_settings

    client = service._memory.vector_store.client
    found: dict[str, int] = {}
    try:
        points = client.retrieve(
            collection_name=core_settings.qdrant_collection,
            ids=list(memory_ids),
            with_payload=True,
            with_vectors=False,
        )
        for pt in points or []:
            found[str(getattr(pt, "id", ""))] = _times_derived_of(
                (getattr(pt, "payload", None) or {}).get("metadata")
            )
    except Exception:
        logger.warning("times_derived retrieve failed; defaulting to 1 per row", exc_info=True)
    return sum(found.get(str(mid), 1) for mid in memory_ids)


def _merged_metadata(client, memory_id: str, patch: dict) -> dict:
    """Read-modify-write the nested metadata dict (set_payload replaces keys wholesale)."""
    from config import settings as core_settings

    points = client.retrieve(
        collection_name=core_settings.qdrant_collection,
        ids=[memory_id],
        with_payload=True,
        with_vectors=False,
    )
    meta = {}
    if points:
        meta = dict((points[0].payload or {}).get("metadata") or {})
    meta.update(patch)
    return meta


async def _graph_invalidate(service, group_id: str, memory_id: str, *, superseded_by: str | None) -> int:
    """Bi-temporal graph invalidation via the bridge loop (best-effort)."""
    graphiti = getattr(service, "_graphiti", None)
    driver = getattr(graphiti, "driver", None) if graphiti else None
    if driver is None or not getattr(service, "_bridge", None):
        return 0
    from .graph_patcher import invalidate_memory_graph

    coro = invalidate_memory_graph(
        driver,
        group_id=group_id,
        memory_id=memory_id,
        superseded_by=superseded_by,
    )
    try:
        return await service._run_on_bridge_async(coro, timeout=30.0)
    except Exception:
        logger.warning("graph invalidation failed for %s (non-fatal)", memory_id, exc_info=True)
        return 0
