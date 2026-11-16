"""Delete path: single/bulk deletes, graph-edge expiry, and the dedup cron.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import uuid

from datetime import datetime, timezone
from config import settings
from schemas import MemoryVisibility
from memory.groups import _build_group_id
from memory.junk import _deleted_msg
from memory.ranking import _times_derived_from_metadata

logger = logging.getLogger(__name__)


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
        meta.get("visibility"),
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
                    unresolved_ids = self._expire_user_graph_writes(user_id)
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

        if filter_null_category:
            memories_to_delete = self._list_null_category_memories(
                user_id=user_id, scope=scope, project_id=project_id,
            )
            deleted_count = 0
            skipped_shared = 0
            skipped_standard = 0
            # Rows whose vector delete succeeded; the graph cascade runs once
            # per distinct episode over all of them afterwards.
            cascade_rows: list[dict] = []
            for mem_info in memories_to_delete:
                meta = mem_info.get("metadata", {}) or {}
                if isinstance(meta.get("metadata"), dict):
                    meta = meta["metadata"]
                if not include_shared and meta.get("visibility") == MemoryVisibility.SHARED.value:
                    skipped_shared += 1
                    continue
                if meta.get("visibility") == MemoryVisibility.STANDARD.value and not settings.is_dictator(user_id):
                    skipped_standard += 1
                    continue
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
                except Exception as e:
                    logger.warning(f"Failed to delete null-category memory {mid}: {e}")
            graph = self._bulk_graph_cleanup(cascade_rows, "null-category")
            return {
                "message": _deleted_msg("null-category memories", deleted_count, skipped_shared, skipped_standard),
                **graph,
            }

        # For filtered deletes, we need to list then delete individually
        memories = self.list_memories(
            user_id=user_id,
            scope=scope,
            category=category,
            project_id=project_id,
        )

        deleted_count = 0
        skipped_shared = 0
        skipped_standard = 0
        cascade_rows = []
        for mem in memories:
            if not include_shared and getattr(mem, "visibility", None) == MemoryVisibility.SHARED.value:
                skipped_shared += 1
                continue
            if getattr(mem, "visibility", None) == MemoryVisibility.STANDARD.value and not settings.is_dictator(user_id):
                skipped_standard += 1
                continue
            try:
                # Get full memory for graph cleanup before deleting
                full_mem = m.get(mem.id)
                m.delete(mem.id)
                if full_mem:
                    cascade_rows.append({"id": mem.id, "mem": full_mem, "group_id": None})
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {mem.id}: {e}")

        graph = self._bulk_graph_cleanup(cascade_rows, "filtered")
        return {
            "message": _deleted_msg("memories", deleted_count, skipped_shared, skipped_standard),
            **graph,
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
            return {"message": "No memories deleted (scroll failed)"}

        # (memory_id, payload, metadata) for every non-shared row.
        private_rows: list[tuple[str, dict, dict]] = []
        private_groups: set[str] = set()
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
            private_groups.add(
                _build_group_id(
                    visibility, user_id, metadata.get("project_id"), metadata.get("workspace")
                )
            )

        # Cheap backstop: bulk-expire every edge in the private groups
        # (ONE Cypher statement per call, not a per-edge write loop).
        if self._graphiti and self._bridge and private_groups:
            try:
                self._expire_graph_edges_for_groups(sorted(private_groups))
            except Exception as e:
                logger.warning(f"Graph cleanup for private groups failed: {e}")

        # Vectors FIRST (NS issue #176): the store must never be left
        # partially deleted by a slow/failed graph step — an interrupted or
        # crashing graph cascade used to strand the remaining rows as
        # half-deleted points. Once every vector is gone the graph work is
        # best-effort and fully reportable.
        deleted = 0
        deleted_ids: set[str] = set()
        for mid, payload, _ in private_rows:
            try:
                self._memory.vector_store.delete(mid)
                deleted += 1
                deleted_ids.add(mid)
            except Exception as e:
                logger.warning(f"Failed to delete private memory {mid}: {e}")

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
            ],
            "private-only",
            deleted_ids=deleted_ids,
        )

        msg = f"Deleted {deleted} private memories"
        if shared_preserved:
            msg += f" (preserved {shared_preserved} shared)"
        if standard_preserved:
            msg += f" (preserved {standard_preserved} standard)"
        return {"message": msg, **graph}

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

    def _expire_user_graph_writes(self, user_id: str) -> list[str]:
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
        """
        try:
            user_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
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
                    key = (h, _dedup_partition(mem["payload"]))
                    key_counts[key] = key_counts.get(key, 0) + 1

        exact_removed = 0
        hash_groups: dict[tuple, list[dict]] = {}
        if any(c > 1 for c in key_counts.values()):
            for page in self._iter_user_memory_pages(user_id, batch_size):
                for mem in page:
                    h = mem["payload"].get("hash")
                    if not h or _is_tombstoned(mem["payload"]):
                        continue
                    key = (h, _dedup_partition(mem["payload"]))
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
        for page in self._iter_user_memory_pages(user_id, batch_size):
            chunk = [
                mem for mem in page
                if mem["id"] not in deleted_ids and not _is_tombstoned(mem["payload"])
            ]
            if not chunk:
                continue
            stored_vectors = self._fetch_dense_vectors([mem["id"] for mem in chunk])
            for mem in chunk:
                semantic_removed += self._semantic_dedup_one(
                    m, user_id, mem, stored_vectors.get(mem["id"]), threshold, deleted_ids,
                    collector=collector,
                )

        return {
            "user_id": user_id,
            "exact_duplicates_removed": exact_removed,
            "semantic_duplicates_removed": semantic_removed,
            "semantic_mode": "tombstone" if settings.dedup_semantic_tombstone_enabled else "delete",
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
    ) -> int:
        """Phase 2 for one memory: merge near-duplicates in its own partition.

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
