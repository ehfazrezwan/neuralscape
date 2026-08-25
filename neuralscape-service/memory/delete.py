"""Delete path: single/bulk deletes, graph-edge expiry, and the dedup cron.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging

from datetime import datetime, timezone
from config import settings
from schemas import MemoryVisibility
from memory.groups import _build_group_id
from memory.junk import _deleted_msg
from memory.ranking import _times_derived_from_metadata

logger = logging.getLogger(__name__)

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

        # First, get the memory content to find related graph edges
        mem = m.get(memory_id)

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
                cascade_result = self._cascade_or_fallback_expire(mem, memory_id=memory_id)
                if isinstance(result, dict):
                    result["graph_cascade"] = "resolved" if cascade_result.get("resolved") else "unresolved"
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
            # Zero-effect detection: never silently absorb a row whose graph
            # episode couldn't be resolved/verified — count and report it.
            unresolved_ids: list[str] = []
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
                    if self._graphiti and self._bridge:
                        cascade_result = self._cascade_or_fallback_expire(
                            {"memory": mem_info.get("data", ""), "metadata": meta},
                            memory_id=mid,
                        )
                        if not cascade_result.get("resolved"):
                            logger.warning(
                                "Bulk delete (null-category) for memory=%s could "
                                "not resolve/verify graph cleanup", mid,
                            )
                            unresolved_ids.append(mid)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete null-category memory {mid}: {e}")
            return {
                "message": _deleted_msg("null-category memories", deleted_count, skipped_shared, skipped_standard),
                "graph_cascade_unresolved_ids": unresolved_ids,
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
        unresolved_ids: list[str] = []
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
                if full_mem and self._graphiti and self._bridge:
                    cascade_result = self._cascade_or_fallback_expire(full_mem, memory_id=mem.id)
                    if not cascade_result.get("resolved"):
                        logger.warning(
                            "Bulk delete for memory=%s could not resolve/verify "
                            "graph cleanup", mem.id,
                        )
                        unresolved_ids.append(mem.id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {mem.id}: {e}")

        return {
            "message": _deleted_msg("memories", deleted_count, skipped_shared, skipped_standard),
            "graph_cascade_unresolved_ids": unresolved_ids,
        }

    def _delete_private_only(self, user_id: str) -> dict:
        """Delete every PRIVATE memory the user owns; leave shared writes alone.

        Used by the default (non-include_shared) unfiltered bulk-delete
        path. Scrolls the user's full set, partitions by visibility,
        deletes the private rows one by one via Qdrant (mem0's
        ``delete_all`` can't be filtered), then expires the per-user
        private graph groups in bulk.
        """
        try:
            all_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
            logger.warning(f"Failed to scroll memories for private-only delete: {e}")
            return {"message": "No memories deleted (scroll failed)"}

        private_ids: list[tuple[str, dict]] = []
        private_groups: set[str] = set()
        shared_preserved = 0
        for mem in all_memories:
            payload = mem.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            if visibility == MemoryVisibility.SHARED.value:
                shared_preserved += 1
                continue
            private_ids.append((mem["id"], payload))
            pid = metadata.get("project_id")
            if pid:
                private_groups.add(f"user--{user_id}--project--{pid}")
            else:
                private_groups.add(f"user--{user_id}")

        if self._graphiti and self._bridge and private_groups:
            try:
                self._expire_graph_edges_for_groups(sorted(private_groups))
            except Exception as e:
                logger.warning(f"Graph cleanup for private groups failed: {e}")

        deleted = 0
        for mid, payload in private_ids:
            try:
                self._memory.vector_store.delete(mid)
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete private memory {mid}: {e}")

        msg = f"Deleted {deleted} private memories"
        if shared_preserved:
            msg += f" (preserved {shared_preserved} shared)"
        return {"message": msg}

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
            group_id = _build_group_id(visibility, owner, metadata.get("project_id"))
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
        """Expire graph edges across every group_id this user authored.

        Used by the unfiltered bulk-delete path. Private groups
        (`user--{user_id}` and `user--{user_id}--project--*`) are
        expired wholesale — they only contain this user's writes.
        Shared groups (`shared`, `shared--project--*`) hold team
        knowledge from many writers, so we only expire the specific
        edges this user authored via per-memory cleanup.

        Returns the memory ids whose shared-group episode could not be
        resolved/verified (zero-effect detection) — never silently
        dropped, so the caller can report them.
        """
        try:
            user_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
            logger.warning(f"Failed to scroll memories for graph cleanup (non-critical): {e}")
            return []

        private_groups: set[str] = set()
        shared_memories: list[dict] = []
        for mem in user_memories:
            payload = mem.get("payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            # mem0 sometimes double-wraps metadata
            if isinstance(metadata.get("metadata"), dict):
                metadata = metadata["metadata"]
            visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
            pid = metadata.get("project_id")
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
                if pid:
                    private_groups.add(f"user--{user_id}--project--{pid}")
                else:
                    private_groups.add(f"user--{user_id}")

        if private_groups:
            self._expire_graph_edges_for_groups(sorted(private_groups))
        unresolved_ids: list[str] = []
        for mem in shared_memories:
            mid = mem.get("id", "")
            try:
                cascade_result = self._cascade_or_fallback_expire(mem, memory_id=mid)
                if not cascade_result.get("resolved"):
                    logger.warning(
                        "Nuke-all delete for user=%s shared memory=%s could not "
                        "resolve/verify graph cleanup", user_id, mid,
                    )
                    unresolved_ids.append(mid)
            except Exception as e:
                logger.warning(f"Per-shared-memory edge expiration failed (non-critical): {e}")
                unresolved_ids.append(mid)
        return unresolved_ids

    def _expire_graph_edges_for_groups(self, group_ids: list[str]) -> None:
        """Expire all graph edges in the given groups (bulk soft-delete)."""
        try:
            from graphiti_core.edges import EntityEdge

            edges = self._run_on_bridge(
                EntityEdge.get_by_group_ids(
                    self._graphiti.driver, group_ids=group_ids, limit=1000
                )
            )
            now = datetime.now(timezone.utc)
            for edge in edges:
                edge.expired_at = now
                self._run_on_bridge(edge.save(self._graphiti.driver))
        except Exception as e:
            logger.warning(f"Bulk graph edge expiration failed (non-critical): {e}")

    # ──────────────────────────────────────────────
    # Dedup operations
    # ──────────────────────────────────────────────

    def _scroll_all_user_memories(self, user_id: str, batch_size: int = 100) -> list[dict]:
        """Paginate through Qdrant scroll() to collect all points for a user.

        Bypasses mem0's wrapper which doesn't support pagination.

        Returns:
            List of {"id": str, "payload": dict} for every point matching user_id.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._memory.vector_store.client
        collection = settings.qdrant_collection
        scroll_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

        all_points: list[dict] = []
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
            for pt in points:
                all_points.append({"id": str(pt.id), "payload": pt.payload or {}})
            if next_offset is None:
                break
            offset = next_offset

        return all_points

    def _delete_qdrant_memory_with_graph_cleanup(self, memory_id: str, payload: dict) -> None:
        """Delete a single memory from Qdrant and expire related graph edges.

        Graph cleanup is non-critical — failures are logged but don't propagate.
        """
        self._memory.vector_store.delete(memory_id)

        if self._graphiti and self._bridge:
            try:
                mem = {
                    "memory": payload.get("data", ""),
                    "metadata": payload.get("metadata", {}),
                }
                self._cascade_or_fallback_expire(mem, memory_id=memory_id)
            except Exception as e:
                logger.warning(f"Graph cleanup failed for {memory_id} (non-critical): {e}")

    def dedup_memories(self, user_id: str, *, semantic: bool = True) -> dict:
        """Remove duplicate memories for a user in two phases.

        Phase 1 — Exact: group by payload hash, keep newest, delete rest.
        Phase 2 — Semantic: for each remaining memory, search for near-duplicates
                  above the cosine threshold, delete the older one.

        ``semantic=False`` skips phase 2. The dreaming sweep's MERGE action
        supersedes it when ``DREAMING_ENABLED=true``: where this phase
        hard-deletes the older near-duplicate (losing any unique details it
        held), the dream merge folds those details into the survivor and
        tombstones reversibly. The lossless exact-hash phase always runs.

        Returns:
            Dict with user_id, exact_duplicates_removed, semantic_duplicates_removed,
            total_checked.
        """
        m = self._get_memory()
        threshold = settings.dedup_similarity_threshold
        batch_size = settings.dedup_batch_size

        memories = self._scroll_all_user_memories(user_id, batch_size=batch_size)
        deleted_ids: set[str] = set()

        def _pvis(payload: dict) -> str | None:
            """Visibility of a raw Qdrant payload (handles mem0 double-wrap)."""
            meta = payload.get("metadata", {}) or {}
            if isinstance(meta.get("metadata"), dict):
                meta = meta["metadata"]
            return meta.get("visibility")

        # ── Phase 1: Exact dedup by hash ──
        # Key on (hash, visibility) so a `standard` memory is never collapsed
        # into an identically-worded `shared`/`private` one (different tiers are
        # semantically distinct — a standard is binding, a shared note is not).
        exact_removed = 0
        hash_groups: dict[tuple, list[dict]] = {}
        for mem in memories:
            h = mem["payload"].get("hash")
            if h:
                hash_groups.setdefault((h, _pvis(mem["payload"])), []).append(mem)

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
                    self._delete_qdrant_memory_with_graph_cleanup(mid, dup["payload"])
                    deleted_ids.add(mid)
                    exact_removed += 1
                    reinforcement += _times_derived_from_metadata(
                        dup["payload"].get("metadata")
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete exact dup {mid}: {e}")
            if reinforcement:
                self._bump_times_derived(survivor["id"], reinforcement)

        # ── Phase 2: Semantic dedup ──
        semantic_removed = 0
        if not semantic:
            return {
                "user_id": user_id,
                "exact_duplicates_removed": exact_removed,
                "semantic_duplicates_removed": 0,
                "semantic_skipped": "superseded by dreaming MERGE",
                "total_checked": len(memories),
            }
        remaining = [mem for mem in memories if mem["id"] not in deleted_ids]

        for mem in remaining:
            mid = mem["id"]
            if mid in deleted_ids:
                continue

            text = mem["payload"].get("data", "")
            if not text:
                continue

            try:
                embedding = m.embedding_model.embed(text)
                # mem0 v2.0.2 renamed the search kwarg ``limit`` → ``top_k``
                # on its Qdrant wrapper (``mem0/mem0/vector_stores/qdrant.py``).
                # Calling with ``limit`` raises ``Qdrant.search() got an
                # unexpected keyword argument 'limit'`` and dedup silently
                # fails for every memory in the user's pool.
                hits = m.vector_store.search(
                    query=text,
                    vectors=embedding,
                    top_k=5,
                    filters={"user_id": user_id},
                )
            except Exception as e:
                logger.warning(f"Semantic search failed for {mid}: {e}")
                continue

            for hit in hits:
                hit_id = str(hit["id"]) if isinstance(hit, dict) else str(hit.id)
                hit_score = hit["score"] if isinstance(hit, dict) else hit.score
                hit_payload = hit.get("payload", {}) if isinstance(hit, dict) else (hit.payload or {})

                if hit_id == mid or hit_id in deleted_ids:
                    continue
                if hit_score < threshold:
                    continue
                # Never dedup across visibility tiers — a standard must not be
                # merged into a shared/private near-duplicate (or vice-versa).
                if _pvis(hit_payload) != _pvis(mem["payload"]):
                    continue

                # Delete the older one
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
                    self._delete_qdrant_memory_with_graph_cleanup(older_id, older_payload)
                    deleted_ids.add(older_id)
                    semantic_removed += 1
                    # Same reinforcement transfer as the exact phase: the
                    # near-duplicate we just dropped was a repeated
                    # observation — its counter moves to the survivor.
                    self._bump_times_derived(
                        newer_id, _times_derived_from_metadata(older_payload.get("metadata"))
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete semantic dup {older_id}: {e}")

                # If we deleted ourselves, stop checking this memory
                if older_id == mid:
                    break

        return {
            "user_id": user_id,
            "exact_duplicates_removed": exact_removed,
            "semantic_duplicates_removed": semantic_removed,
            "total_checked": len(memories),
        }
