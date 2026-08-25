"""Graph admin/inspection: nodes, edges, episodes, communities, junk cleanup.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import re

from datetime import datetime, timezone
from config import settings
from memory.groups import _edge_is_invalidated, _get_group_ids
from memory.junk import _JUNK_RE

logger = logging.getLogger(__name__)

class GraphAdminMixin:
    """GraphAdminMixin for MemoryService (mechanical split — see memory_service.py)."""

    # ──────────────────────────────────────────────
    # Graph introspection
    # ──────────────────────────────────────────────

    def get_graph_nodes(
        self,
        user_id: str,
        project_id: str | None = None,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[dict]:
        """List entity nodes from Graphiti.

        By default excludes nodes whose connecting graph edges are ALL
        expired/invalidated. ``EntityNode.summary`` is Graphiti's "regional
        summary of surrounding edges" — once every RELATES_TO edge touching
        a node has been soft-expired (memory/provenance.py's cascade on a
        visibility flip/delete, or the dreaming sweep's INVALIDATE/PRUNE/
        MERGE) that summary describes content that must not be surfaced
        through this listing endpoint. A node with NO edges at all (never
        connected to anything) is left alone — that's "not yet enriched",
        not "expired". Pass ``include_expired=True`` for an operator/debug
        view that includes everything (never the default).
        """
        g = self._get_graphiti()
        if g is None:
            return []

        from graphiti_core.nodes import EntityNode

        group_ids = _get_group_ids(user_id, project_id)

        try:
            nodes = self._run_on_bridge(
                EntityNode.get_by_group_ids(g.driver, group_ids=group_ids, limit=limit)
            )
            if not include_expired:
                nodes = self._filter_live_nodes(g, group_ids, nodes)
            return [
                {
                    "uuid": n.uuid,
                    "name": n.name,
                    "summary": n.summary,
                    "labels": n.labels,
                    "group_id": n.group_id,
                    "created_at": n.created_at.isoformat(),
                }
                for n in nodes
            ]
        except Exception as e:
            logger.warning("get_graph_nodes failed: %s", e)
            return []

    def _filter_live_nodes(self, g, group_ids: list[str], nodes: list) -> list:
        """Drop entity nodes whose connecting RELATES_TO edges are ALL
        expired/invalidated (see ``get_graph_nodes`` docstring). Reuses the
        SAME edge-liveness definition as the search path
        (memory/groups.py::_edge_is_invalidated / _live_edges_filter) rather
        than inventing a second one. Fails open (returns ``nodes``
        unfiltered) on a lookup error — this is a defense-in-depth listing
        filter, not the primary expiry mechanism.
        """
        if not nodes:
            return nodes
        from graphiti_core.edges import EntityEdge

        try:
            edges = self._run_on_bridge(
                EntityEdge.get_by_group_ids(g.driver, group_ids=group_ids, limit=5000)
            )
        except Exception as e:
            logger.warning("Live-node edge lookup failed (fail-open to unfiltered): %s", e)
            return nodes

        touched: set[str] = set()
        live: set[str] = set()
        for e in edges:
            is_live = not _edge_is_invalidated(e)
            for node_uuid in (e.source_node_uuid, e.target_node_uuid):
                touched.add(node_uuid)
                if is_live:
                    live.add(node_uuid)
        return [n for n in nodes if n.uuid not in touched or n.uuid in live]

    def get_graph_edges(
        self,
        user_id: str,
        project_id: str | None = None,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[dict]:
        """List entity edges (facts) from Graphiti.

        By default excludes soft-expired/invalidated edges — same
        bi-temporal liveness definition the search path enforces
        (memory/groups.py::_edge_is_invalidated): an edge with a non-null
        ``expired_at`` (memory/provenance.py's cascade, or the dreaming
        sweep) or ``invalid_at`` is dropped. Without this, a listing
        endpoint would hand back exactly the facts a visibility flip/delete
        just expired — defeating that cascade entirely. Pass
        ``include_expired=True`` for an operator/debug view that includes
        everything (never the default).
        """
        g = self._get_graphiti()
        if g is None:
            return []

        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        group_ids = _get_group_ids(user_id, project_id)

        try:
            edges = self._run_on_bridge(
                EntityEdge.get_by_group_ids(g.driver, group_ids=group_ids, limit=limit)
            )
            if not include_expired:
                edges = [e for e in edges if not _edge_is_invalidated(e)]
            return [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "fact": e.fact,
                    "source_node_uuid": e.source_node_uuid,
                    "target_node_uuid": e.target_node_uuid,
                    "group_id": e.group_id,
                    "created_at": e.created_at.isoformat(),
                    "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                    "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
                    "expired_at": e.expired_at.isoformat() if e.expired_at else None,
                }
                for e in edges
            ]
        except Exception as e:
            logger.warning("get_graph_edges failed: %s", e)
            return []

    def search_episodes_fulltext(
        self, query: str, user_id: str, project_id: str | None = None, limit: int = 3
    ) -> list[dict]:
        """Relevance-ranked verbatim episode excerpts (group-scoped).

        Read-side evidence leg for ask: distillation drops one-off
        micro-details (colors, gifts, stated reasons); the raw session text
        still has them. Uses Graphiti's `episode_content` fulltext index —
        one Cypher call, no embeddings.
        """
        g = self._get_graphiti()
        if g is None:
            return []
        # Lowercase + drop Lucene boolean operators so caller tokens can't
        # change query semantics or trip the fulltext parser.
        terms = [
            t.lower()
            for t in re.findall(r"[A-Za-z0-9']+", query)
            if len(t) >= 3 and t.upper() not in ("AND", "OR", "NOT")
        ]
        if not terms:
            return []
        lucene = " OR ".join(terms[:12])
        group_ids = _get_group_ids(user_id, project_id)
        cypher = """
        CALL db.index.fulltext.queryNodes('episode_content', $q) YIELD node, score
        WHERE node.group_id IN $group_ids
        RETURN node.uuid AS uuid, node.content AS content,
               node.created_at AS created_at, score
        ORDER BY score DESC LIMIT $limit
        """

        async def _run():
            async with g.driver.session() as session:
                result = await session.run(
                    cypher, q=lucene, group_ids=group_ids, limit=limit
                )
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=10.0) or []
        except Exception as e:
            logger.warning(f"episode fulltext search failed: {e}")
            return []
        finally:
            # No-op for an awaited coroutine; prevents "never awaited"
            # leaks when the bridge is mocked or fails before scheduling.
            coro.close()
        return [
            {
                "uuid": rec.get("uuid"),
                "content": rec.get("content") or "",
                "created_at": str(rec.get("created_at") or ""),
                "score": rec.get("score"),
            }
            for rec in records
            if rec.get("content")
        ]

    def get_graph_episodes(
        self, user_id: str, project_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List episodic nodes from Graphiti."""
        g = self._get_graphiti()
        if g is None:
            return []

        group_ids = _get_group_ids(user_id, project_id)
        now = datetime.now(timezone.utc)

        try:
            episodes = self._run_on_bridge(
                g.retrieve_episodes(
                    reference_time=now,
                    last_n=limit,
                    group_ids=group_ids,
                )
            )
            return [
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content,
                    "source_description": ep.source_description,
                    "group_id": ep.group_id,
                    "created_at": ep.created_at.isoformat(),
                    "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in episodes
            ]
        except Exception as e:
            logger.warning("get_graph_episodes failed: %s", e)
            return []

    def get_graph_communities(
        self, user_id: str, project_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List community nodes from Graphiti."""
        g = self._get_graphiti()
        if g is None:
            return []

        from graphiti_core.nodes import CommunityNode

        group_ids = _get_group_ids(user_id, project_id)

        try:
            communities = self._run_on_bridge(
                CommunityNode.get_by_group_ids(g.driver, group_ids=group_ids, limit=limit)
            )
            return [
                {
                    "uuid": c.uuid,
                    "name": c.name,
                    "summary": c.summary if hasattr(c, "summary") else "",
                    "group_id": c.group_id,
                    "created_at": c.created_at.isoformat(),
                }
                for c in communities
            ]
        except Exception as e:
            logger.warning("get_graph_communities failed: %s", e)
            return []

    def delete_episode(self, episode_uuid: str, user_id: str | None = None, project_id: str | None = None) -> dict:
        """Delete a single episodic node from the graph by UUID.

        Args:
            episode_uuid: The episode UUID to delete.
            user_id: The caller's user ID for authorization. When None, skips
                the authorization check (backward-compatible with tests).
            project_id: Optional project scope for authorization.

        Returns:
            Dict with message or error. Raises PermissionError if the episode
            is not in the caller's readable group_ids (same pool rules as reads).
        """
        g = self._get_graphiti()
        if g is None:
            return {"error": "Graphiti not initialized"}

        # Authorization: only delete episodes in the caller's readable group_ids
        # (same pool union as search: private + shared + standard-when-enabled).
        # When user_id is None, skip the check (backward-compatible with tests).
        if user_id is None:
            # No authz — delete directly (legacy behavior for tests)
            try:
                self._run_on_bridge(
                    g.driver.execute_query(
                        "MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e",
                        uuid=episode_uuid,
                    )
                )
                return {"message": f"Episode {episode_uuid} deleted"}
            except Exception as e:
                logger.error(f"Failed to delete episode {episode_uuid}: {e}")
                return {"error": str(e)}

        group_ids = _get_group_ids(user_id, project_id)
        try:
            # First verify the episode exists and is in an authorized group
            async def _check_and_delete():
                async with g.driver.session() as session:
                    # Check episode exists and belongs to an authorized group
                    check = await session.run(
                        "MATCH (e:Episodic {uuid: $uuid}) RETURN e.group_id AS group_id",
                        uuid=episode_uuid,
                    )
                    records = await check.data()
                    if not records:
                        return {"error": f"Episode {episode_uuid} not found"}
                    ep_group = records[0].get("group_id")
                    if ep_group not in group_ids:
                        raise PermissionError(
                            f"Episode {episode_uuid} belongs to group {ep_group!r}, "
                            f"which is not in the caller's readable groups."
                        )
                    # Authorized — delete it
                    await session.run(
                        "MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e",
                        uuid=episode_uuid,
                    )
                    return {"message": f"Episode {episode_uuid} deleted"}

            return self._run_on_bridge(_check_and_delete())
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete episode {episode_uuid}: {e}")
            return {"error": str(e)}

    def _find_junk_episodes(self, episodes: list[dict]) -> list[dict]:
        """Filter a list of episodes to only those matching junk patterns."""
        junk = []
        for ep in episodes:
            content = ep.get("content", "")
            is_assistant_log = content.strip().startswith("assistant:")
            is_junk_pattern = bool(_JUNK_RE.search(content))
            if is_assistant_log or is_junk_pattern:
                junk.append(ep)
        return junk

    def delete_junk_episodes(
        self,
        user_id: str,
        project_id: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Find and delete junk episodic nodes whose content matches raw event patterns.

        Junk episodes are those with content starting with 'assistant:' (raw conversation
        logs) or matching _JUNK_PATTERNS.

        When project_id is provided, only that group is cleaned.
        When omitted, ALL known groups are cleaned (global + all projects).

        Args:
            user_id: User identifier (maps to group_id for filtering).
            project_id: Optional project scope. If None, cleans all known groups.
            dry_run: If True, list junk episodes without deleting.

        Returns:
            Dict with per-group breakdown of junk counts / deletions.
        """
        # Determine which project_ids to scan
        if project_id is not None:
            project_ids_to_scan = [project_id]
        else:
            # None = global group, then all known projects
            # (deployment-specific, supplied via KNOWN_PROJECT_SLUGS)
            project_ids_to_scan = [None] + settings.known_projects

        breakdown = {}
        total_junk = 0
        total_deleted = 0
        all_samples = []

        for pid in project_ids_to_scan:
            group_label = pid if pid else "global"
            episodes = self.get_graph_episodes(user_id=user_id, project_id=pid, limit=500)
            junk_episodes = self._find_junk_episodes(episodes)
            total_junk += len(junk_episodes)

            if dry_run:
                breakdown[group_label] = {"junk_count": len(junk_episodes)}
                all_samples.extend(
                    {"uuid": ep["uuid"], "group": group_label, "content": ep["content"][:120]}
                    for ep in junk_episodes[:5]
                )
            else:
                deleted_uuids = []
                for ep in junk_episodes:
                    result = self.delete_episode(ep["uuid"], user_id=user_id, project_id=pid)
                    if "error" not in result:
                        deleted_uuids.append(ep["uuid"])
                breakdown[group_label] = {
                    "deleted_count": len(deleted_uuids),
                    "deleted_uuids": deleted_uuids,
                }
                total_deleted += len(deleted_uuids)
                all_samples.extend(
                    {"uuid": ep["uuid"], "group": group_label, "content": ep["content"][:120]}
                    for ep in junk_episodes[:3]
                )

        if dry_run:
            return {
                "dry_run": True,
                "junk_count": total_junk,
                "breakdown": breakdown,
                "samples": all_samples[:15],
            }

        return {
            "deleted_count": total_deleted,
            "breakdown": breakdown,
            "samples": all_samples[:15],
        }

    def rebuild_graph_from_vectors(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        wipe_first: bool = False,
        batch_size: int = 256,
        max_rows: int | None = None,
    ) -> dict:
        """Rebuild the knowledge graph from EXISTING Qdrant vector rows WITHOUT re-extracting or re-embedding.

        This is the efficiency win for future re-ingests where only graph-side
        logic changed (e.g. R1 reference_time, R2 message-type, R4 bitemporal
        fields): it re-enriches the graph per stored vector row, bypassing the
        write-path content-hash dedup graph-skip (audit R6).

        The dedup skip (worker.py ~line 251-253 `if created and memories:`)
        means re-asserting an existing fact returns created=False and skips
        graph enrichment, starving Graphiti's contradiction/dup engine. This
        batch rebuild path sidesteps that skip by enriching every row directly.

        CRITICAL SEMANTIC SCOPE:
        - Single-fact path (store_raw/remember/ingest_document): 1 Qdrant
          vector row ⇄ 1 graph episode. A vector→graph rebuild reproduces this
          graph EXACTLY.
        - Conversation path (extract_and_store, used by accuracy bench): the
          graph episode is the RAW CONVERSATION text (one episode per session),
          while Qdrant holds EXTRACTED FACTS. A vector-only rebuild produces a
          PER-FACT graph, NOT the original per-conversation graph — and the raw
          conversation is not in Qdrant. This method is the SINGLE-FACT-equivalent
          rebuild; conversation-extraction stores need the raw conversations,
          not just vectors.

        Args:
            user_id: Scope to user's memories (owner filter). None = all users.
            project_id: Scope to project. None = all projects.
            wipe_first: Clear graph for the scope before rebuilding. Requires
                user_id for safety (refuses global wipe without explicit scope).
            batch_size: Qdrant scroll page size (1-500, default 256).
            max_rows: Optional cap on rows processed (for testing / partial rebuilds).

        Returns:
            Dict with counts: {"scanned": N, "enriched": M, "failed": F, "skipped": S}
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from schemas import MemoryVisibility

        # Safety: refuse global graph wipe without explicit scope
        if wipe_first and not user_id:
            return {
                "error": "wipe_first requires user_id (refusing global graph wipe without explicit scope)"
            }

        # Trigger lazy mem0/Graphiti init up front (cold instances have
        # _graphiti/_bridge unset until _get_memory runs) so wipe_first below
        # actually runs instead of silently no-oping (Copilot review).
        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection

        # Wipe graph for the scope if requested — PRIVATE groups ONLY.
        if wipe_first:
            logger.info(f"Wiping PRIVATE graph groups for user_id={user_id}, project_id={project_id}")
            try:
                from memory.groups import _build_group_id

                # ONLY the caller's PRIVATE group(s). The SHARED group_id
                # (`shared` / `shared--project--{pid}`) is CROSS-USER — it holds
                # every user's shared knowledge — so a per-user rebuild must
                # never expire it (Copilot review: that would wipe shared graph
                # state for everyone). Shared-pool rebuilds are out of scope here.
                groups_to_wipe = [
                    _build_group_id(MemoryVisibility.PRIVATE.value, user_id, project_id or None)
                ]

                # Expire all edges in these groups
                if self._graphiti and self._bridge:
                    from graphiti_core.edges import EntityEdge

                    def _make_expire(gid):
                        async def _expire_group():
                            edges = await EntityEdge.get_by_group_ids(
                                self._graphiti.driver, group_ids=[gid], limit=10000
                            )
                            now = datetime.now(timezone.utc)
                            for edge in edges:
                                edge.expired_at = now
                            await EntityEdge.save_bulk(self._graphiti.driver, edges)
                        return _expire_group

                    for group_id in groups_to_wipe:
                        try:
                            self._run_on_bridge(_make_expire(group_id)())
                        except Exception as e:
                            logger.warning(f"Failed to wipe group {group_id} (non-critical): {e}")
            except Exception as e:
                logger.warning(f"Graph wipe failed (non-critical): {e}")

        # Build Qdrant filter for the scope
        must = []
        if project_id:
            must.append(
                FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id))
            )

        # Owner scoping: ownership lives EITHER in metadata.owner_user_id
        # (shared/newer rows) OR the top-level `user_id` (legacy/private rows)
        # — mirror the retag sweep's should-OR so neither shape is missed
        # (Copilot review). Qdrant: match = all(must) AND none(must_not) AND
        # at least one(should) when should is non-empty.
        should = []
        if user_id:
            should.append(
                FieldCondition(key="metadata.owner_user_id", match=MatchValue(value=user_id))
            )
            should.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        # Exclude verbatim passage chunks (mirror _scroll_standard convention)
        must_not = [
            FieldCondition(key="metadata.memory_kind", match=MatchValue(value="passage"))
        ]

        scroll_filter = (
            Filter(
                must=must or None,
                must_not=must_not,
                should=should or None,
            )
            if (must or must_not or should)
            else None
        )
        page_size = max(1, min(batch_size, 500))

        scanned = enriched = failed = skipped = 0
        offset = None

        logger.info(
            f"Starting graph rebuild from vectors: user_id={user_id}, project_id={project_id}, "
            f"max_rows={max_rows}, batch_size={page_size}"
        )

        while True:
            # Scroll Qdrant for existing rows
            try:
                points, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.error(f"Qdrant scroll failed: {e}")
                break

            if not points:
                break

            for point in points:
                scanned += 1
                if max_rows and scanned > max_rows:
                    logger.info(f"Hit max_rows cap ({max_rows}), stopping")
                    break

                try:
                    # Extract fields from the Qdrant point payload
                    payload = getattr(point, "payload", None) or {}
                    content = payload.get("data", "")
                    if not content:
                        skipped += 1
                        continue

                    metadata = payload.get("metadata", {}) or {}
                    # Unwrap double-nested metadata if present
                    if isinstance(metadata.get("metadata"), dict):
                        metadata = metadata["metadata"]

                    memory_id = str(getattr(point, "id", ""))
                    owner_user_id = metadata.get("owner_user_id") or payload.get("user_id")
                    if not owner_user_id:
                        logger.warning(f"Row {memory_id} has no owner_user_id, skipping")
                        skipped += 1
                        continue

                    row_project_id = metadata.get("project_id")
                    visibility = metadata.get("visibility") or MemoryVisibility.PRIVATE.value
                    occurred_at = metadata.get("occurred_at")
                    source_ref = metadata.get("source_ref")

                    # Call enrich_graph with the stored fields (NO re-extraction, NO re-embedding)
                    success = self.enrich_graph(
                        content=content,
                        user_id=owner_user_id,
                        project_id=row_project_id,
                        visibility=visibility,
                        memory_id=memory_id,
                        source_ref=source_ref,
                        occurred_at=occurred_at,
                    )

                    if success:
                        enriched += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.warning(f"Failed to enrich row {getattr(point, 'id', '?')}: {e}")
                    failed += 1

            if max_rows and scanned >= max_rows:
                break
            if offset is None:
                break

        logger.info(
            f"Graph rebuild complete: scanned={scanned}, enriched={enriched}, "
            f"failed={failed}, skipped={skipped}"
        )

        return {
            "scanned": scanned,
            "enriched": enriched,
            "failed": failed,
            "skipped": skipped,
        }
