"""Delete path: single/bulk deletes, graph-edge expiry, and the dedup cron.

BULK-DELETE INVARIANT (``delete_memories``, every sweep; tests in
``tests/test_bulk_delete_honesty.py``): a sweep either deletes NOTHING and raises
``MemoryListingFailed`` (REST 503), or reports exactly what it deleted, what failed
(``failed_ids``) and whether it is incomplete (``truncated`` / ``listing_failed``,
REST 207); and graph cleanup runs only for rows whose vector delete succeeded.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import uuid

from datetime import datetime, timezone
from config import settings
from schemas import MemoryVisibility, normalize_visibility
from memory.groups import _build_group_id
from memory.junk import _deleted_msg
from memory.ranking import _mem_is_tombstoned, _times_derived_from_metadata

logger = logging.getLogger(__name__)


def semantic_dedup_phase_enabled() -> bool:
    """Whether dedup's semantic phase (phase 2) may run for the dedup cron and post-compile dedup.

    With dreaming on, the dream sweep's MERGE action owns near-duplicate merges,
    so the LOSSY phase (hard delete) is skipped. With
    ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` phase 2 is not lossy (it only tombstones:
    reversible, nothing deleted, nothing cascaded), so an operator who opted in
    gets it whether or not dreaming is on. With dreaming on, two guards keep it off
    the sweep's rows: ``dedup_memories`` holds a pool's dream lock for as long as
    it works on that pool's rows (and skips the pool if it cannot take it), and
    the sweep's ``apply_actions`` skips any action whose rows were tombstoned
    after staging.
    """
    from extensions.dreaming.config import dreaming_settings

    return (not dreaming_settings.enabled) or bool(settings.dedup_semantic_tombstone_enabled)


# Sentinel: no dream-pool lock is held (dreaming off), so no neighbour pool check.
_NO_POOL_GUARD: str = "\0no-pool-guard"
# Expiry of the dream lock dedup holds while it works on one page of a pool's rows:
# long enough for a page of vector searches, short enough that a crashed run frees
# the pool for the sweep quickly.
_DEDUP_LOCK_TTL_MS = 10 * 60 * 1000


def _dream_pool_of(payload: dict) -> str | None:
    """The dream-sweep pool a raw row belongs to (None for tiers the sweep never touches).

    Same derivation as ``consolidate.enumerate_pools``, so the key matches the
    one the sweep locks.
    """
    from extensions.dreaming.consolidate import pool_key

    meta, _ = _dedup_meta(payload)
    visibility = meta.get("visibility") or "private"
    if visibility == "standard":
        return None  # authoritative tier: read-only to dreaming, never swept
    shared = visibility == "shared"
    owner = meta.get("owner_user_id") or payload.get("user_id")
    return pool_key(
        visibility="shared" if shared else "private",
        owner_user_id=None if shared else owner,
        project_id=meta.get("project_id"),
        workspace=meta.get("workspace"),
    )


# Bulk-delete listing page and the cap on listing rounds: a filtered sweep keeps
# re-listing (deleted rows drop out of the next listing) until nothing new is
# left; if rows remain unseen when the cap is hit the result says ``truncated``.
_BULK_PAGE = 500
_BULK_MAX_ROUNDS = 40


def _tombstones_note(n: int) -> str:
    """Message suffix saying how many of the erased rows were hidden tombstones."""
    return f" (including {n} tombstoned)" if n else ""


def _bulk_partial_fields(failed_ids: list[str], truncated: bool) -> dict:
    """Honest-status keys every bulk delete returns (``failed_ids`` empty / ``truncated`` False when clean)."""
    return {"failed_ids": list(failed_ids), "failed": len(failed_ids), "truncated": bool(truncated)}


class MemoryListingFailed(RuntimeError):
    """A bulk delete could not list the caller's memories, so it deleted NOTHING.

    Raised (instead of returning a message) so a caller can never mistake a
    failed listing for a completed delete; the REST routes answer 503.
    """


def _is_valid_point_id(memory_id: str) -> bool:
    """True for a well-formed Qdrant point id (UUID string or unsigned int)."""
    try:
        uuid.UUID(str(memory_id))
        return True
    except (ValueError, AttributeError, TypeError):
        return str(memory_id).isdigit()


def _dedup_meta(payload: dict) -> tuple[dict, bool]:
    """Unwrapped metadata of a raw Qdrant payload + whether it was double-wrapped."""
    meta = payload.get("metadata", {}) or {}
    if isinstance(meta.get("metadata"), dict):
        return meta["metadata"], True
    return meta, False


def _dedup_visibility(meta: dict) -> str | None:
    """Visibility as the dedup partition sees it: an absent stamp is private.

    Conversation-extracted rows used to carry no ``metadata.visibility`` and
    are now stamped ``private``; every reader already treats an absent value as
    de-facto private, so the two must share one partition or dedup would stop
    merging a new row with its legacy twin. An unrecognised value stays its own
    (raw) partition — stricter, never a cross-partition merge.
    """
    vis = meta.get("visibility")
    if vis is None or vis == "":
        return MemoryVisibility.PRIVATE.value
    try:
        return normalize_visibility(vis)
    except (ValueError, TypeError):
        return vis


def _dedup_partition(payload: dict) -> tuple:
    """Partition identity a semantic near-duplicate must share to be merged.

    (user_id, scope, project_id, visibility, workspace). Mirrors the write
    path's partitioning (``memory/write.py`` content-hash identity and
    ``memory/groups.py::_build_group_id``): the same text in two projects,
    at two visibility tiers, or in a reference workspace vs the memory
    workspace is two distinct memories. Missing/"memory" workspace is the
    default memory workspace; an empty project_id is no project. Visibility
    and scope compare raw (an absent value is its own partition), matching
    the exact phase's ``(hash, visibility)`` key — stricter means fewer
    merges, never a cross-partition delete.
    """
    meta, _ = _dedup_meta(payload)
    workspace = meta.get("workspace")
    if not workspace or workspace == "memory":
        workspace = None
    return (
        payload.get("user_id"),
        meta.get("scope"),
        meta.get("project_id") or None,
        _dedup_visibility(meta),
        workspace,
    )


def _dedup_partition_filters(user_id: str, payload: dict) -> dict:
    """Qdrant filter (fork dict syntax) restricting a search to ``payload``'s partition.

    The fork's ``Qdrant._create_filter`` only expresses equality on present
    values — it has no is-empty/is-null operator — so a partition field that
    is absent on the source row (no project, no visibility stamp, default
    workspace) cannot be pushed down; those are enforced by the strict
    ``_dedup_partition`` equality check on every hit.
    """
    meta, wrapped = _dedup_meta(payload)
    prefix = "metadata.metadata." if wrapped else "metadata."
    filters: dict = {"user_id": user_id}
    for field in ("scope", "project_id", "visibility", "workspace"):
        value = meta.get(field)
        if field == "workspace" and value == "memory":
            continue  # default workspace: absent OR "memory" — post-filtered
        if isinstance(value, str) and value and value != "*":
            filters[prefix + field] = value
    return filters


def _is_tombstoned(payload: dict) -> bool:
    """Whether a raw Qdrant payload is a dream/dedup tombstone (hidden from recall)."""
    # A double-wrapped row (``metadata.metadata``) can carry the flag at
    # either level: recall filters read the outer one, dedup/dreaming patches
    # may have written the inner one. Either hides the row.
    outer = payload.get("metadata", {}) or {}
    inner, _ = _dedup_meta(payload)
    return bool(outer.get("dream_tombstoned") or inner.get("dream_tombstoned"))


# Semantic-dedup neighbour search: ask for a small page first and widen it
# only while the page is full and its lowest hit still clears the threshold,
# so copies from other partitions (which the fork's equality-only filter
# cannot always exclude) cannot crowd an in-partition duplicate out of the
# result. The last size is the hard cap per row.
_DEDUP_TOP_K_STEPS = (10, 40, 160)


def _exact_category(payload: dict):
    """Category component of the exact-phase group key (same hash, other category: not a duplicate)."""
    return _dedup_meta(payload)[0].get("category")


def _differs_only_by_category(a: dict, b: dict) -> bool:
    """True for two rows with the SAME text hash but different categories.

    Such a pair can only exist because someone wrote the same text on purpose
    under two categories (the explicit write path keys on the category), so
    no dedup phase may merge them. Near-duplicates with different hashes are
    unaffected and still merge across categories.
    """
    ha, hb = a.get("hash"), b.get("hash")
    if not ha or ha != hb:
        return False
    ca = _dedup_meta(a)[0].get("category")
    cb = _dedup_meta(b)[0].get("category")
    return ca != cb


def _dense_vector(vector) -> list[float] | None:
    """The unnamed dense vector of a retrieved point (``None`` if unusable).

    Collections with the ``bm25`` sparse slot return ``{"": dense, "bm25": sparse}``;
    pre-v3 collections return the dense list directly.
    """
    if isinstance(vector, dict):
        vector = vector.get("")
    if isinstance(vector, (list, tuple)) and vector and all(
        isinstance(x, (int, float)) for x in vector
    ):
        return list(vector)
    return None


class DeleteMixin:
    """DeleteMixin for MemoryService (mechanical split — see memory_service.py)."""

    def delete_memory(self, memory_id: str, caller_user_id: str | None = None) -> dict:
        """Delete a single memory by ID from both vector store and graph.

        ``caller_user_id`` enforces the same ownership/visibility rules as
        bulk delete: private memories are owner-only; shared memories allow
        deletion by the author or a dictator; standard-tier memories are
        dictator-only. This is the only delete path with no user namespacing
        (bulk deletes are already scoped by ``user_id``), so without this
        check any caller could remove anyone's private memory or a binding
        standard by ID.
        """
        m = self._get_memory()

        # First, get the memory content to find related graph edges. A
        # malformed id cannot name a stored point (Qdrant ids are UUIDs or
        # unsigned ints) and makes the store raise a 400-class error — that
        # is "not found", not a server fault (the route maps LookupError to
        # 404). A well-formed id whose lookup fails still propagates.
        try:
            mem = m.get(memory_id)
        except Exception:
            if not _is_valid_point_id(memory_id):
                raise LookupError(f"Memory {memory_id} not found") from None
            raise
        if mem is None:
            # A well-formed id that names no stored point: not found, not a
            # server fault (the route maps LookupError to 404; a genuine
            # store failure raised above and still propagates).
            raise LookupError(f"Memory {memory_id} not found")

        # Permission gate: enforce ownership/visibility rules when caller_user_id
        # is provided (backward-compatible: None skips the gate, existing tests).
        if mem is not None and caller_user_id is not None:
            metadata = mem.get("metadata", {}) or {}
            # Unwrap mem0's potential double-wrap
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            owner = metadata.get("owner_user_id") or mem.get("user_id", "")
            vis = metadata.get("visibility") or MemoryVisibility.PRIVATE.value

            # Dictators may delete anything
            if not settings.is_dictator(caller_user_id):
                # Standard tier: dictator-only
                if vis == MemoryVisibility.STANDARD.value:
                    raise PermissionError(
                        "Only a dictator may delete 'standard'-tier memories."
                    )
                # Shared tier: author or dictator
                elif vis == MemoryVisibility.SHARED.value:
                    if caller_user_id != owner:
                        raise PermissionError(
                            f"Only the memory's owner may delete it (owner: {owner!r})."
                        )
                # Private tier (or legacy null visibility): owner-only
                else:
                    if caller_user_id != owner:
                        raise PermissionError(
                            f"Only the memory's owner may delete it (owner: {owner!r})."
                        )

        result = m.delete(memory_id)

        # Expire related graph edges (soft-delete, non-critical). Tries the
        # episode-precise cascade first (memory/provenance.py); falls back to
        # the substring heuristic below only when no episode can be resolved.
        #
        # Zero-effect detection (audit): surface whether cleanup was actually
        # VERIFIED, not just attempted — a pre-fix incident here was a silent
        # no-op reported as a clean success. `graph_cascade` is additive on
        # mem0's own delete result and is only set when the graph is
        # configured at all (so a graph-disabled deployment's response shape
        # is unchanged).
        if mem and self._graphiti and self._bridge:
            try:
                batch = self._cascade_rows_once(
                    [{"id": memory_id, "mem": mem, "group_id": None}],
                    deleted_ids={memory_id},
                )
                cascade_result = batch["results"].get(memory_id) or {}
                if isinstance(result, dict):
                    result["graph_cascade"] = "resolved" if cascade_result.get("resolved") else "unresolved"
                    # Sibling facts that shared this memory's graph episode lost
                    # it (the episode body holds the deleted fact verbatim, so
                    # it must go); the caller enqueues these single-fact
                    # re-ingests so the survivors regain graph evidence.
                    result.update(self._graph_repair_summary(batch))
                if not cascade_result.get("resolved"):
                    logger.warning(
                        "delete_memory(%s): graph episode could not be "
                        "resolved/verified — cleanup relied on the lossy "
                        "substring fallback",
                        memory_id,
                    )
            except Exception as e:
                logger.warning(f"Graph edge expiration failed for {memory_id} (non-critical): {e}")
                if isinstance(result, dict):
                    result["graph_cascade"] = "error"

        return result

    @staticmethod
    def _graph_repair_summary(batch: dict) -> dict:
        """Additive response keys describing the sibling re-enrichment work a
        cascade produced (jobs for the caller to enqueue + what could not be
        repaired). Always the same four keys so clients can rely on them."""
        jobs = batch.get("graph_jobs") or []
        return {
            "graph_jobs": jobs,
            "graph_jobs_count": len(jobs),
            "graph_sibling_unrepaired_ids": list(batch.get("sibling_unrepaired_ids") or []),
            "graph_sibling_repair_failed": bool(batch.get("sibling_repair_failed")),
        }

    def delete_memories(
        self,
        user_id: str,
        scope: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
        filter_null_category: bool = False,
        include_shared: bool = False,
    ) -> dict:
        """Bulk delete memories with filters from both vector store and graph.

        By default this only removes the caller's PRIVATE writes. Shared
        memories the caller authored survive even an unfiltered bulk
        delete — they're team artifacts and one user shouldn't be able
        to wipe team knowledge with a sweep call (which an LLM client
        can trigger via the MCP tool). Pass ``include_shared=True`` to
        also delete the caller's shared writes (admin-style nuke).

        Every sweep erases the caller's tombstoned rows (dreaming / semantic-dedup
        / supersession tombstones) like any other row, under the same
        shared/standard rules and graph cascade; the count includes them.

        Single-memory delete by ID is unaffected — that path is always
        an intentional action against a specific memory.
        """
        m = self._get_memory()

        has_any_filter = scope or category or project_id or filter_null_category
        if not has_any_filter:
            if include_shared:
                # Caller explicitly asked to remove everything they wrote,
                # including shared. Use mem0's bulk delete + the full
                # graph cleanup that touches per-memory edges in shared
                # groups too.
                logger.warning(
                    f"Deleting ALL memories for user={user_id} including "
                    f"shared writes (include_shared=True)"
                )
                unresolved_ids: list[str] = []
                if self._graphiti and self._bridge:
                    # strict: if the user's rows cannot be listed the graph
                    # cleanup cannot run — refuse BEFORE delete_all rather than
                    # delete the vectors and silently leave their graph facts live.
                    unresolved_ids = self._expire_user_graph_writes(user_id, strict=True)
                m.delete_all(user_id=user_id)
                return {
                    "message": "All memories deleted (including shared)",
                    "graph_cascade_unresolved_ids": unresolved_ids,
                }

            # Default: remove only private writes. Shared memories stay.
            logger.warning(
                f"Deleting all PRIVATE memories for user={user_id} "
                f"(shared writes preserved; pass include_shared=True to override)"
            )
            return self._delete_private_only(user_id)

        deleted_count = 0
        tombstones_erased = 0
        skipped_shared = 0
        skipped_standard = 0
        # Rows whose vector delete succeeded; the graph cascade runs once
        # per distinct episode over all of them afterwards.
        cascade_rows: list[dict] = []
        failed_ids: list[str] = []
        seen: set[str] = set()

        listing_failed = False
        # Widest raw window a sweep may ask for; beyond it the result says ``truncated``.
        cap = _BULK_PAGE * _BULK_MAX_ROUNDS

        def _sweep(fetch, row_id, handle) -> bool:
            """Page a sweep: returns True when it stopped early (truncated or a mid-sweep listing failure).

            ``fetch(limit)`` returns the RAW rows (tombstones included); exhaustion is
            judged from that raw size, never from a post-filter length. Rows already
            handled (deleted, failed, or preserved shared/standard) stay in the next
            listing, so when a round makes no progress the window is DOUBLED to reach
            rows behind them, up to ``cap``.
            """
            nonlocal listing_failed
            limit = _BULK_PAGE
            for round_no in range(_BULK_MAX_ROUNDS):
                try:
                    batch = fetch(limit)
                except Exception as e:
                    if deleted_count == 0:   # nothing deleted at all (preserved rows and failed rows delete nothing)
                        raise MemoryListingFailed(
                            "Memories could not be listed; nothing was deleted. Retry."
                        ) from e
                    logger.warning(f"Bulk delete listing failed after partial progress: {e}")
                    listing_failed = True
                    return True
                fresh = [x for x in batch if row_id(x) not in seen]
                for row in fresh:
                    seen.add(row_id(row))
                    handle(row)
                if len(batch) < limit:
                    return False  # a short RAW page: everything that matched has been seen
                if not fresh:
                    if limit >= cap:
                        return True
                    limit = min(limit * 2, cap)
            return True

        if filter_null_category:
            def _handle_null(mem_info):
                nonlocal deleted_count, tombstones_erased, skipped_shared, skipped_standard
                meta = mem_info.get("metadata", {}) or {}
                if isinstance(meta.get("metadata"), dict):
                    meta = meta["metadata"]
                if not include_shared and meta.get("visibility") == MemoryVisibility.SHARED.value:
                    skipped_shared += 1
                    return
                if meta.get("visibility") == MemoryVisibility.STANDARD.value and not settings.is_dictator(user_id):
                    skipped_standard += 1
                    return
                mid = mem_info["id"]
                try:
                    m.vector_store.delete(mid)
                    cascade_rows.append({
                        "id": mid,
                        "mem": {"memory": mem_info.get("data", ""), "metadata": meta},
                        "user_id": user_id,
                        "group_id": None,
                    })
                    deleted_count += 1
                    if _mem_is_tombstoned(mem_info):
                        tombstones_erased += 1
                except Exception as e:
                    failed_ids.append(mid)
                    logger.warning(f"Failed to delete null-category memory {mid}: {e}")

            truncated = _sweep(
                lambda lim: self._list_null_category_memories(
                    user_id=user_id, scope=scope, project_id=project_id, limit=lim,
                ),
                lambda x: x["id"],
                _handle_null,
            )
            # Cleanup runs for what WAS deleted even if the sweep stopped early.
            graph = self._bulk_graph_cleanup(cascade_rows, "null-category")
            return {
                "message": _deleted_msg("null-category memories", deleted_count, skipped_shared, skipped_standard)
                + _tombstones_note(tombstones_erased)
                + (f" ({len(failed_ids)} failed)" if failed_ids else ""),
                **graph,
                **_bulk_partial_fields(failed_ids, truncated),
                **({"listing_failed": True} if listing_failed else {}),
            }

        # For filtered deletes, we need to list then delete individually
        def _fetch_raw(limit):
            """The filtered listing as RAW mem0 dicts (tombstones included) so the page size is the raw size."""
            self._ensure_filter_indexes()
            filters: dict = {"user_id": user_id}
            if scope:
                filters["metadata.scope"] = scope
            if category:
                filters["metadata.category"] = category
            if project_id:
                filters["metadata.project_id"] = project_id
            return self._extract_memory_list(m.get_all(filters=filters, top_k=limit))

        def _handle_row(mem):
            nonlocal deleted_count, tombstones_erased, skipped_shared, skipped_standard
            # Tombstoned rows (dreaming / semantic-dedup / supersession) are erased
            # like any other row of the caller, under the same shared/standard rules.
            meta = mem.get("metadata") or {}
            if isinstance(meta.get("metadata"), dict):
                meta = meta["metadata"]
            visibility = meta.get("visibility")
            if not include_shared and visibility == MemoryVisibility.SHARED.value:
                skipped_shared += 1
                return
            if visibility == MemoryVisibility.STANDARD.value and not settings.is_dictator(user_id):
                skipped_standard += 1
                return
            mid = mem.get("id")
            try:
                # Get full memory for graph cleanup before deleting
                full_mem = m.get(mid)
                m.delete(mid)
                if full_mem:
                    cascade_rows.append({"id": mid, "mem": full_mem, "group_id": None})
                deleted_count += 1
                if _mem_is_tombstoned(mem):
                    tombstones_erased += 1
            except Exception as e:
                failed_ids.append(mid)
                logger.warning(f"Failed to delete memory {mid}: {e}")

        truncated = _sweep(_fetch_raw, lambda x: x.get("id"), _handle_row)
        graph = self._bulk_graph_cleanup(cascade_rows, "filtered")
        return {
            "message": _deleted_msg("memories", deleted_count, skipped_shared, skipped_standard)
            + _tombstones_note(tombstones_erased)
            + (f" ({len(failed_ids)} failed)" if failed_ids else ""),
            **graph,
            **_bulk_partial_fields(failed_ids, truncated),
            **({"listing_failed": True} if listing_failed else {}),
        }

    def _bulk_graph_cleanup(
        self,
        rows: list[dict],
        label: str,
        *,
        deleted_ids=None,
        repair_siblings: bool = True,
    ) -> dict:
        """Graph half of a bulk delete over rows whose vectors are gone.

        One cascade per distinct episode (``_cascade_rows_once``); siblings
        that survive the delete get their stale episode ref cleared and a
        re-enrichment job. Returns the bulk response keys
        (``graph_cascade_unresolved_ids`` + the sibling-repair summary). A
        row with an unresolved episode is reported, never silently absorbed.
        """
        empty = {
            "graph_cascade_unresolved_ids": [],
            "graph_jobs": [],
            "graph_jobs_count": 0,
            "graph_sibling_unrepaired_ids": [],
            "graph_sibling_repair_failed": False,
        }
        if not rows or not (self._graphiti and self._bridge):
            return empty
        done = {r["id"] for r in rows} if deleted_ids is None else set(deleted_ids)
        batch = self._cascade_rows_once(rows, deleted_ids=done)
        if not repair_siblings:
            batch = dict(batch, graph_jobs=[], sibling_unrepaired_ids=[])
        for mid in batch["unresolved_ids"]:
            logger.warning(
                "Bulk delete (%s) for memory=%s could not resolve/verify graph cleanup",
                label, mid,
            )
        return {
            "graph_cascade_unresolved_ids": list(batch["unresolved_ids"]),
            **self._graph_repair_summary(batch),
        }

    def _delete_private_only(self, user_id: str) -> dict:
        """Delete every PRIVATE memory the user owns; leave shared writes alone.

        Used by the default (non-include_shared) unfiltered bulk-delete
        path. Scrolls the user's full set, partitions by visibility,
        deletes the private rows one by one via Qdrant (mem0's
        ``delete_all`` can't be filtered).

        Graph cleanup is two-layered (F4): a bulk group-edge expiry is a
        cheap backstop, but edges alone leave episodes and entity-node
        summaries live — so every private row also gets the same
        provenance-exact per-row cascade (``_cascade_or_fallback_expire``)
        the shared branch (``_expire_user_graph_writes``) already used.
        Group ids are built via ``_build_group_id`` (workspace-aware),
        not a manual ``user--{user_id}`` string, so a workspace-partitioned
        row's group actually gets swept.
        """
        try:
            all_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
            logger.warning(f"Failed to scroll memories for private-only delete: {e}")
            raise MemoryListingFailed(
                "Memories could not be listed; nothing was deleted. Retry."
            ) from e

        # (memory_id, payload, metadata) for every non-shared row.
        private_rows: list[tuple[str, dict, dict]] = []
        private_groups: set[str] = set()
        row_groups: dict[str, str] = {}
        shared_preserved = 0
        standard_preserved = 0
        for mem in all_memories:
            payload = mem.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            if visibility == MemoryVisibility.SHARED.value:
                shared_preserved += 1
                continue
            if visibility == MemoryVisibility.STANDARD.value:
                # Authoritative tier is never collateral of a private-only
                # wipe — standards are removed only through the explicit,
                # dictator-gated delete paths.
                standard_preserved += 1
                continue
            private_rows.append((mem["id"], payload, metadata))
            row_group = _build_group_id(
                visibility, user_id, metadata.get("project_id"), metadata.get("workspace")
            )
            private_groups.add(row_group)
            row_groups[mem["id"]] = row_group

        # Vectors FIRST (NS issue #176): the store must never be left
        # partially deleted by a slow/failed graph step — an interrupted or
        # crashing graph cascade used to strand the remaining rows as
        # half-deleted points. Once every vector is gone the graph work is
        # best-effort and fully reportable.
        deleted = 0
        tombstones_erased = 0
        deleted_ids: set[str] = set()
        failed_ids: list[str] = []
        for mid, payload, _ in private_rows:
            try:
                self._memory.vector_store.delete(mid)
                deleted += 1
                deleted_ids.add(mid)
                if _is_tombstoned(payload):
                    tombstones_erased += 1
            except Exception as e:
                failed_ids.append(mid)
                logger.warning(f"Failed to delete private memory {mid}: {e}")

        # Cheap backstop: bulk-expire every edge in a private group (ONE Cypher
        # statement per call, not a per-edge write loop) — only AFTER the vector
        # deletes and only for groups whose rows ALL went: a row that failed to
        # delete is still live in Qdrant and must keep its graph facts.
        failed_groups = {row_groups[f] for f in failed_ids}
        clean_groups = sorted(private_groups - failed_groups)
        if self._graphiti and self._bridge and clean_groups:
            try:
                self._expire_graph_edges_for_groups(clean_groups)
            except Exception as e:
                logger.warning(f"Graph cleanup for private groups failed: {e}")

        # Provenance-exact cascade (episodes + node summaries), ONCE per
        # distinct episode and bounded to the rows just scrolled — never an
        # unbounded search (NS issue #176). A row whose vector delete failed
        # is still live: it is repaired as a sibling, not treated as deleted.
        graph = self._bulk_graph_cleanup(
            [
                {
                    "id": mid,
                    "mem": {"memory": payload.get("data", ""), "metadata": metadata},
                    "user_id": user_id,
                    "group_id": None,
                }
                for mid, payload, metadata in private_rows
                if mid in deleted_ids      # a row that failed to delete is still live: no graph state is touched for it
            ],
            "private-only",
            deleted_ids=deleted_ids,
        )

        msg = f"Deleted {deleted} private memories" + _tombstones_note(tombstones_erased)
        if shared_preserved:
            msg += f" (preserved {shared_preserved} shared)"
        if standard_preserved:
            msg += f" (preserved {standard_preserved} standard)"
        if failed_ids:
            msg += f" ({len(failed_ids)} failed)"
        return {"message": msg, **graph, **_bulk_partial_fields(failed_ids, False)}

    def _expire_graph_edges_for_memory(self, mem: dict) -> None:
        """Soft-delete graph edges related to a memory by setting expired_at."""
        content = mem.get("memory", "")
        if not content:
            return
        try:
            from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

            # Audit 27 #10: deep-copy — mutating the shared singleton here
            # clamped every concurrent search's graph fan-out to 5.
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 5

            metadata = mem.get("metadata", {}) or {}
            # Unwrap mem0's potential double-wrap
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            # Scope edge expiration to the memory's exact namespace.
            # `_get_group_ids` would return the owner's whole readable
            # universe (their private + the shared pool), which means
            # deleting a private memory could expire similarly-worded
            # edges from the shared pool — wrong pool.
            owner = metadata.get("owner_user_id") or mem.get("user_id", "")
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            group_id = _build_group_id(
                visibility, owner, metadata.get("project_id"), metadata.get("workspace")
            )
            group_ids = [group_id]

            results = self._run_on_bridge(
                self._graphiti.search_(
                    query=content,
                    config=config,
                    group_ids=group_ids,
                )
            )
            now = datetime.now(timezone.utc)
            for edge in results.edges:
                if edge.fact and content.lower() in edge.fact.lower():
                    edge.expired_at = now
                    self._run_on_bridge(edge.save(self._graphiti.driver))
        except Exception as e:
            logger.warning(f"Graph edge expiration failed (non-critical): {e}")

    def _expire_user_graph_writes(self, user_id: str, *, strict: bool = False) -> list[str]:
        """Expire graph edges/episodes/summaries across every group_id this
        user authored.

        Used by the unfiltered (include_shared=True) bulk-delete path.
        Private groups (`user--{user_id}` and `user--{user_id}--project--*`,
        workspace-suffixed when set — via ``_build_group_id``) get a bulk
        edge-expiry backstop PLUS a provenance-exact per-row cascade (F4:
        edges-only left episodes and entity-node summaries live). Shared
        groups (`shared`, `shared--project--*`) hold team knowledge from
        many writers, so we only ever expire the specific edges/episodes
        this user authored via the same per-memory cascade.

        Returns the memory ids whose episode could not be resolved/verified
        (zero-effect detection, private or shared) — never silently
        dropped, so the caller can report them.

        ``strict=True`` (the bulk-delete caller) raises ``MemoryListingFailed``
        when the rows cannot be listed instead of returning ``[]`` — an empty
        list reads as "nothing to clean up", which is wrong when the listing
        itself failed.
        """
        try:
            user_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
            if strict:
                logger.warning(f"Failed to scroll memories for graph cleanup: {e}")
                raise MemoryListingFailed(
                    "Memories could not be listed; nothing was deleted. Retry."
                ) from e
            logger.warning(f"Failed to scroll memories for graph cleanup (non-critical): {e}")
            return []

        private_groups: set[str] = set()
        private_memories: list[dict] = []
        shared_memories: list[dict] = []
        for mem in user_memories:
            payload = mem.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            # mem0 sometimes double-wraps metadata
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            if visibility == MemoryVisibility.SHARED.value:
                # Don't touch the shared group_id — other users' edges live
                # there too. Per-memory edge expiration narrows to just
                # this user's specific facts.
                shared_memories.append({
                    "memory": payload.get("data", ""),
                    "metadata": metadata,
                    "id": str(mem.get("id", "")),
                })
            else:
                private_memories.append({
                    "memory": payload.get("data", ""),
                    "metadata": metadata,
                    "id": str(mem.get("id", "")),
                })
                private_groups.add(
                    _build_group_id(
                        visibility, user_id, metadata.get("project_id"), metadata.get("workspace")
                    )
                )

        if private_groups:
            self._expire_graph_edges_for_groups(sorted(private_groups))
        # Every row of this user is being deleted (delete_all follows), so
        # there are no surviving siblings to repair — only dedupe the cascade
        # to once per distinct episode.
        rows = [
            {"id": mem.get("id", ""), "mem": mem, "user_id": user_id, "group_id": None}
            for mem in private_memories + shared_memories
        ]
        batch = self._cascade_rows_once(
            rows, deleted_ids={r["id"] for r in rows}
        )
        for mid in batch["unresolved_ids"]:
            logger.warning(
                "Nuke-all delete for user=%s memory=%s could not "
                "resolve/verify graph cleanup", user_id, mid,
            )
        return list(batch["unresolved_ids"])

    def _expire_graph_edges_for_groups(self, group_ids: list[str]) -> int:
        """Expire every live edge in the given groups (bulk soft-delete).

        ONE Cypher statement (one round trip) regardless of how many edges
        the groups hold. The previous implementation fetched at most 1000
        edges and issued one serial ``edge.save()`` write per edge — O(N)
        blocking round trips inside the synchronous delete request for a
        large user (NS issue #176), and it silently left every edge past the
        1000th live. Returns the number of edges expired (0 on failure —
        non-critical, logged).
        """
        if not group_ids:
            return 0
        cypher = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.group_id IN $group_ids AND r.expired_at IS NULL
        SET r.expired_at = datetime($now)
        RETURN count(r) AS edges_expired
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, group_ids=list(group_ids), now=now_iso)
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=60.0) or []
        except Exception as e:
            coro.close()
            logger.warning(f"Bulk graph edge expiration failed (non-critical): {e}")
            return 0
        return int(records[0]["edges_expired"]) if records else 0

    # ──────────────────────────────────────────────
    # Dedup operations
    # ──────────────────────────────────────────────

    def _iter_user_memory_pages(self, user_id: str, batch_size: int = 100):
        """Yield one page (a list of ``{"id", "payload"}``) at a time.

        Streams a user's pool through Qdrant ``scroll`` so a caller that can
        work page by page (dedup) never holds the whole pool in memory.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._memory.vector_store.client
        collection = settings.qdrant_collection
        scroll_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                yield [{"id": str(pt.id), "payload": pt.payload or {}} for pt in points]
            if next_offset is None:
                break
            offset = next_offset

    def _scroll_all_user_memories(self, user_id: str, batch_size: int = 100) -> list[dict]:
        """Paginate through Qdrant scroll() to collect all points for a user.

        Bypasses mem0's wrapper which doesn't support pagination. Callers that
        can stream should use ``_iter_user_memory_pages``.

        Returns:
            List of {"id": str, "payload": dict} for every point matching user_id.
        """
        return [pt for page in self._iter_user_memory_pages(user_id, batch_size) for pt in page]

    def _fetch_dense_vectors(self, ids: list[str]) -> dict[str, list[float]]:
        """Stored dense vectors for ``ids`` (one Qdrant retrieve; ``{}`` on failure).

        Callers fall back to the embedder for any id missing from the result.
        """
        if not ids:
            return {}
        try:
            points = self._memory.vector_store.client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=ids,
                with_payload=False,
                with_vectors=True,
            )
        except Exception as e:
            logger.warning(f"Dedup vector fetch failed for {len(ids)} ids (embedding fallback): {e}")
            return {}
        out: dict[str, list[float]] = {}
        for pt in points or []:
            vec = _dense_vector(getattr(pt, "vector", None))
            if vec is not None:
                out[str(pt.id)] = vec
        return out

    def _delete_qdrant_memory_with_graph_cleanup(
        self, memory_id: str, payload: dict, *, collector: dict | None = None
    ) -> None:
        """Delete a single memory from Qdrant and expire related graph edges.

        Graph cleanup is non-critical — failures are logged but don't propagate.

        With a ``collector`` (``dedup_memories`` passes one) the cascade also
        repairs sibling facts that shared the deleted row's conversation
        episode: their stale refs are cleared and the single-fact re-ingest
        jobs accumulate in ``collector["graph_jobs"]`` for the caller to
        enqueue (``collector["unrepaired_ids"]`` lists the ones that could
        not be repaired). Without a collector (the TTL-expiry sweep in
        memory/write.py has no way to return jobs) the cascade is unchanged
        and siblings are not touched.
        """
        self._memory.vector_store.delete(memory_id)

        if self._graphiti and self._bridge:
            try:
                mem = {
                    "memory": payload.get("data", ""),
                    "metadata": payload.get("metadata", {}),
                    "user_id": payload.get("user_id"),
                }
                if collector is None:
                    self._cascade_or_fallback_expire(mem, memory_id=memory_id)
                    return
                collector.setdefault("deleted", set()).add(memory_id)
                batch = self._cascade_rows_once(
                    [{"id": memory_id, "mem": mem, "user_id": payload.get("user_id"),
                      "group_id": None}],
                    deleted_ids=collector["deleted"],
                )
                collector.setdefault("graph_jobs", []).extend(batch["graph_jobs"])
                collector.setdefault("unrepaired_ids", []).extend(batch["sibling_unrepaired_ids"])
                if batch.get("sibling_repair_failed"):
                    collector["repair_failed"] = True
            except Exception as e:
                logger.warning(f"Graph cleanup failed for {memory_id} (non-critical): {e}")

    @staticmethod
    def _dreaming_enabled() -> bool:
        from extensions.dreaming.config import dreaming_settings

        return bool(dreaming_settings.enabled)

    @staticmethod
    def _dream_lock_client():
        """A Redis client that answers, or None (dedup then skips the semantic phase)."""
        try:
            from extensions.dreaming.sweep import _get_redis

            client = _get_redis()
            client.ping()
            return client
        except Exception:
            logger.warning(
                "dream lock unavailable (Redis down): the semantic dedup phase is skipped "
                "this run, nothing is tombstoned without the pool's dream lock",
                exc_info=True,
            )
            return None

    def _live_rows(self, rows: list[dict]) -> list[dict]:
        """``rows`` minus any tombstoned since the page was read (fresh Qdrant read)."""
        points = self._memory.vector_store.client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[r["id"] for r in rows],
            with_payload=True,
            with_vectors=False,
        )
        hidden = {
            str(pt.id) for pt in points or []
            if _is_tombstoned(getattr(pt, "payload", None) or {})
        }
        return [r for r in rows if r["id"] not in hidden]

    def dedup_memories(self, user_id: str, *, semantic: bool = True) -> dict:
        """Remove duplicate memories for a user in two phases.

        Phase 1 — Exact: group by (payload hash, partition), keep newest,
                  delete rest.
        Phase 2 — Semantic: for each remaining memory, search for near-duplicates
                  in the same partition above the cosine threshold and
                  delete the older one (default) or, with
                  ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED``, tombstone it.

        Both phases share one partition rule (``_dedup_partition``): a row is
        only ever deleted as a duplicate of a row with the same user, scope,
        project_id, visibility and workspace.

        ``semantic=False`` skips phase 2. The dreaming sweep's MERGE action
        supersedes it when ``DREAMING_ENABLED=true``: where this phase
        hard-deletes the older near-duplicate (losing any unique details it
        held), the dream merge folds those details into the survivor and
        tombstones reversibly. The lossless exact-hash phase always runs.

        With ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` (off by default) phase 2
        does not hard-delete: the older near-duplicate gets
        the same atomic tombstone dreaming uses (``dream_tombstoned``,
        ``superseded_by``, ``superseded_reason="semantic_dedup"``,
        ``superseded_at``), so recall hides it, its text stays on disk, and
        re-deriving it revives it (``_revive_if_tombstoned``). Its
        ``times_derived`` moves to the survivor as before. A tombstone
        cascades nothing, so phase 2 produces no ``graph_jobs``; only the
        exact phase's hard deletes do.

        Returns:
            Dict with user_id, exact_duplicates_removed, semantic_duplicates_removed,
            total_checked.
        """
        m = self._get_memory()
        threshold = settings.dedup_similarity_threshold
        batch_size = settings.dedup_batch_size

        deleted_ids: set[str] = set()
        collector: dict = {}

        # The pool is streamed page by page; nothing below holds all of it.
        # Tombstoned rows (dreaming or an earlier semantic merge) are already
        # hidden from recall and are never candidates or survivors.

        # ── Phase 1: Exact dedup by hash ──
        # Key on (hash, partition) — the same partition rule phase 2 uses
        # (user, scope, project_id, visibility, workspace). The hash is an md5
        # of the text alone, so identical text in two projects, at two tiers
        # (a standard is binding, a shared note is not) or in a reference
        # workspace vs a memory must stay separate rows.
        # Pass 1 counts keys (small tuples); pass 2 keeps full rows only for
        # keys seen more than once, so memory is bounded by the duplicates.
        total_checked = 0
        key_counts: dict[tuple, int] = {}
        for page in self._iter_user_memory_pages(user_id, batch_size):
            total_checked += len(page)
            for mem in page:
                h = mem["payload"].get("hash")
                if h and not _is_tombstoned(mem["payload"]):
                    key = (h, _dedup_partition(mem["payload"]), _exact_category(mem["payload"]))
                    key_counts[key] = key_counts.get(key, 0) + 1

        exact_removed = 0
        hash_groups: dict[tuple, list[dict]] = {}
        if any(c > 1 for c in key_counts.values()):
            for page in self._iter_user_memory_pages(user_id, batch_size):
                for mem in page:
                    h = mem["payload"].get("hash")
                    if not h or _is_tombstoned(mem["payload"]):
                        continue
                    key = (h, _dedup_partition(mem["payload"]), _exact_category(mem["payload"]))
                    if key_counts.get(key, 0) > 1:
                        hash_groups.setdefault(key, []).append(mem)
        del key_counts

        for h, group in hash_groups.items():
            if len(group) < 2:
                continue
            # Sort by created_at descending — keep the first (newest)
            group.sort(
                key=lambda x: x["payload"].get("created_at", ""),
                reverse=True,
            )
            survivor = group[0]
            # Reinforcement-aware dedup: each dropped duplicate is a repeated
            # observation of the same fact. The survivor absorbs the counters
            # of every row it replaces (sum semantics — a dropped row may
            # itself have accumulated write-path reinforcements).
            reinforcement = 0
            for dup in group[1:]:
                mid = dup["id"]
                if mid in deleted_ids:
                    continue
                try:
                    self._delete_qdrant_memory_with_graph_cleanup(mid, dup["payload"], collector=collector)
                    deleted_ids.add(mid)
                    exact_removed += 1
                    reinforcement += _times_derived_from_metadata(
                        dup["payload"].get("metadata")
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete exact dup {mid}: {e}")
            if reinforcement:
                self._bump_times_derived(survivor["id"], reinforcement)
        del hash_groups

        # ── Phase 2: Semantic dedup ──
        semantic_removed = 0
        if not semantic:
            logger.info(
                "dedup run user=%s scanned=%d exact_removed=%d semantic=skipped "
                "(caller skipped the semantic phase)",
                user_id, total_checked, exact_removed,
            )
            return {
                "user_id": user_id,
                "exact_duplicates_removed": exact_removed,
                "semantic_duplicates_removed": 0,
                "semantic_skipped": "superseded by dreaming MERGE",
                "total_checked": total_checked,
                **self._dedup_graph_summary(collector, deleted_ids),
            }

        # One page at a time, so stored vectors are held for at most
        # ``batch_size`` rows (a full user pool of 768-d vectors would not fit).
        # With dreaming on, phase 2 HOLDS the dream lock of each pool for the
        # time it works on that pool's rows (the sweep's own key, non-blocking):
        # a pool whose lock it cannot take is mid-sweep and is skipped this run
        # (reported, the next run picks it up). With Redis down it cannot take
        # any lock, so the whole phase is skipped, never run unlocked.
        locked_pools: set[str] = set()
        dreaming_on = self._dreaming_enabled()
        lock_redis = None
        if dreaming_on:
            lock_redis = self._dream_lock_client()
            if lock_redis is None:
                logger.info(
                    "dedup run user=%s scanned=%d exact_removed=%d semantic=skipped "
                    "(dream lock unavailable)", user_id, total_checked, exact_removed,
                )
                return {
                    "user_id": user_id,
                    "exact_duplicates_removed": exact_removed,
                    "semantic_duplicates_removed": 0,
                    "semantic_skipped": "dream lock unavailable (Redis)",
                    "total_checked": total_checked,
                    **self._dedup_graph_summary(collector, deleted_ids),
                }
        for page in self._iter_user_memory_pages(user_id, batch_size):
            chunk = [
                mem for mem in page
                if mem["id"] not in deleted_ids and not _is_tombstoned(mem["payload"])
            ]
            if not chunk:
                continue
            groups: dict[str | None, list[dict]] = {}
            for mem in chunk:
                pool = _dream_pool_of(mem["payload"]) if dreaming_on else None
                groups.setdefault(pool, []).append(mem)
            for pool, rows in groups.items():
                token = None
                if pool is not None:
                    from extensions.dreaming import gate

                    token = gate.acquire_lock(lock_redis, pool, ttl_ms=_DEDUP_LOCK_TTL_MS)
                    if token is None:
                        locked_pools.add(pool)
                        continue
                try:
                    if pool is not None:
                        # Rows may have been tombstoned by a sweep that ran
                        # between reading the page and taking the lock.
                        rows = self._live_rows(rows)
                        if not rows:
                            continue
                    stored_vectors = self._fetch_dense_vectors([mem["id"] for mem in rows])
                    for mem in rows:
                        semantic_removed += self._semantic_dedup_one(
                            m, user_id, mem, stored_vectors.get(mem["id"]), threshold,
                            deleted_ids, collector=collector,
                            held_pool=pool if dreaming_on else _NO_POOL_GUARD,
                        )
                finally:
                    if token is not None:
                        gate.release_lock(lock_redis, pool, token)

        logger.info(
            "dedup run user=%s scanned=%d exact_removed=%d semantic_pairs_merged=%d "
            "semantic_mode=%s threshold=%s semantic_skipped_locked_pools=%d",
            user_id, total_checked, exact_removed, semantic_removed,
            "tombstone" if settings.dedup_semantic_tombstone_enabled else "delete", threshold,
            len(locked_pools),
        )
        return {
            "user_id": user_id,
            "exact_duplicates_removed": exact_removed,
            "semantic_duplicates_removed": semantic_removed,
            "semantic_mode": "tombstone" if settings.dedup_semantic_tombstone_enabled else "delete",
            "semantic_skipped_locked_pools": len(locked_pools),
            "total_checked": total_checked,
            **self._dedup_graph_summary(collector, deleted_ids),
        }

    @staticmethod
    def _dedup_graph_summary(collector: dict, deleted_ids: set[str]) -> dict:
        """Sibling re-enrichment work a dedup run produced (caller enqueues
        ``graph_jobs``). Jobs for rows a later dedup step also deleted are
        dropped — there is nothing left to re-ingest for them."""
        jobs = [j for j in collector.get("graph_jobs", []) if j["memory_id"] not in deleted_ids]
        return {
            "graph_jobs": jobs,
            "graph_jobs_count": len(jobs),
            "graph_sibling_unrepaired_ids": [
                i for i in collector.get("unrepaired_ids", []) if i not in deleted_ids
            ],
            "graph_sibling_repair_failed": bool(collector.get("repair_failed")),
        }

    def _semantic_dedup_one(
        self,
        m,
        user_id: str,
        mem: dict,
        vector: list[float] | None,
        threshold: float,
        deleted_ids: set[str],
        collector: dict | None = None,
        held_pool: str | None = _NO_POOL_GUARD,
    ) -> int:
        """Phase 2 for one memory: merge near-duplicates in its own partition.

        ``held_pool`` is the dream pool whose lock the caller holds (None: a tier
        the sweep never touches). A neighbour in another pool is left alone: the
        lock covers one pool, and the sweep may be working on the other. The
        dedup partition normally lies inside one pool; the two differ only for
        rows whose ``owner_user_id`` is not their ``user_id``. Left at the
        default no pool check is made (dreaming off: no lock is held).

        Searches with the row's STORED dense vector (the embedder is only
        called when the row has none) and restricts candidates to the same
        (user, scope, project_id, visibility, workspace) partition — pushed
        down as a Qdrant filter where the fork's filter syntax can express
        it, and enforced strictly on every hit. Returns the number merged (tombstoned).
        """
        removed = 0
        mid = mem["id"]
        if mid in deleted_ids or _is_tombstoned(mem["payload"]):
            return 0

        text = mem["payload"].get("data", "")
        if not text:
            return 0
        partition = _dedup_partition(mem["payload"])

        try:
            if vector is None:
                # Same task type the write path stores vectors with, so the
                # query lives in the same space as the rows it is compared to.
                vector = m.embedding_model.embed(text, memory_action="add")
            hits = self._partition_neighbours(
                m, user_id, mem["payload"], text, vector, threshold, partition
            )
        except Exception as e:
            logger.warning(f"Semantic search failed for {mid}: {e}")
            return 0

        for hit in hits:
            hit_id = str(hit["id"]) if isinstance(hit, dict) else str(hit.id)
            hit_payload = hit.get("payload", {}) if isinstance(hit, dict) else (hit.payload or {})

            if hit_id == mid or hit_id in deleted_ids:
                continue
            if held_pool is not _NO_POOL_GUARD and _dream_pool_of(hit_payload) != held_pool:
                continue

            # Merge the older one into the newer
            mem_created = mem["payload"].get("created_at", "")
            hit_created = hit_payload.get("created_at", "")
            older_id, older_payload, newer_id = (
                (hit_id, hit_payload, mid)
                if hit_created <= mem_created
                else (mid, mem["payload"], hit_id)
            )

            if older_id in deleted_ids:
                continue
            try:
                if settings.dedup_semantic_tombstone_enabled:
                    self._tombstone_semantic_duplicate(older_id, newer_id, older_payload)
                else:
                    self._delete_qdrant_memory_with_graph_cleanup(
                        older_id, older_payload, collector=collector
                    )
                deleted_ids.add(older_id)
                removed += 1
                # Same reinforcement transfer as the exact phase: the
                # near-duplicate we just merged away was a repeated
                # observation — its counter moves to the survivor.
                self._bump_times_derived(
                    newer_id, _times_derived_from_metadata(older_payload.get("metadata"))
                )
            except Exception as e:
                logger.warning(f"Failed to merge semantic dup {older_id}: {e}")

            # If we merged ourselves away, stop checking this memory
            if older_id == mid:
                break

        return removed

    def _partition_neighbours(
        self, m, user_id: str, payload: dict, text: str, vector, threshold: float, partition: tuple
    ) -> list:
        """Live same-partition rows scoring at/above ``threshold``, best first.

        The partition is pushed down as a Qdrant filter wherever the fork's
        equality-only filter can express it; fields absent on the source row
        (no project, default workspace, no visibility stamp) cannot be, so
        other-partition copies can still come back. The page therefore widens
        (``_DEDUP_TOP_K_STEPS``) while it is full and its lowest hit still
        clears the threshold, so they cannot crowd an in-partition duplicate
        out. Every returned hit passes the strict ``_dedup_partition`` check
        and is not a tombstone.
        """
        # mem0 v2.0.2 renamed the search kwarg ``limit`` → ``top_k`` on its
        # Qdrant wrapper; calling with ``limit`` raises and silently disables
        # dedup for the whole pool.
        filters = _dedup_partition_filters(user_id, payload)
        hits: list = []
        for top_k in _DEDUP_TOP_K_STEPS:
            hits = m.vector_store.search(
                query=text, vectors=vector, top_k=top_k, filters=filters
            ) or []
            if len(hits) < top_k:
                break
            lowest = hits[-1]["score"] if isinstance(hits[-1], dict) else hits[-1].score
            if lowest < threshold:
                break

        out = []
        for hit in hits:
            score = hit["score"] if isinstance(hit, dict) else hit.score
            hit_payload = hit.get("payload", {}) if isinstance(hit, dict) else (hit.payload or {})
            if score < threshold:
                continue
            # Never dedup across partitions — the same text in another
            # project, scope, visibility tier (a standard is binding, a
            # shared note is not) or workspace (a reference passage vs a
            # memory) is a distinct memory, not a duplicate.
            if _dedup_partition(hit_payload) != partition:
                continue
            if _is_tombstoned(hit_payload):
                continue
            if _differs_only_by_category(payload, hit_payload):
                continue
            out.append(hit)
        return out

    def _tombstone_semantic_duplicate(
        self, older_id: str, survivor_id: str, payload: dict | None = None
    ) -> None:
        """Hide ``older_id`` as a near-duplicate of ``survivor_id`` without deleting it.

        Same atomic nested-key merge dreaming's ``_tombstone`` uses
        (``set_payload`` with ``key="metadata"``): only these keys are
        written server-side, so a concurrent ``times_derived`` bump is not
        clobbered. A double-wrapped row (``metadata.metadata``, see
        ``_dedup_meta``) is patched at BOTH levels: the outer one is what the
        recall filters read, the inner one is where its other fields live.
        Raises on failure so the caller does not count a merge that did not
        happen.
        """
        _, wrapped = _dedup_meta(payload or {})
        patch = {
            "dream_tombstoned": True,
            "superseded_by": survivor_id,
            "superseded_reason": "semantic_dedup",
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        client = self._memory.vector_store.client
        for key in ("metadata", "metadata.metadata") if wrapped else ("metadata",):
            client.set_payload(
                collection_name=settings.qdrant_collection,
                payload=patch,
                points=[older_id],
                key=key,
            )
