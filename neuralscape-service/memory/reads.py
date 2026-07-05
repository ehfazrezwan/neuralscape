"""Simple reads: contexts, gets, listings, provenance chains, and the timeline.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import uuid

from datetime import datetime, timezone
from config import settings
from schemas import ContextResponse, MemoryResponse, MemoryVisibility
from memory.audit import _audit_log
from memory.hashing import _created_at_key
from memory.ranking import _mem_is_tombstoned

logger = logging.getLogger(__name__)

# Retrieval economics (C1/C2) bounds: max ids per batch-get and max window
# half-width for the timeline tool. Shared by REST + MCP boundaries.
GET_MEMORIES_MAX_IDS = 50
TIMELINE_MAX_DEPTH = 50

class ReadsMixin:
    """ReadsMixin for MemoryService (mechanical split — see memory_service.py)."""

    # ──────────────────────────────────────────────
    # Context operations
    # ──────────────────────────────────────────────

    def get_project_context(
        self,
        user_id: str,
        project_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> ContextResponse:
        """Get project + global context organized by category, with paging.

        Retrieves user preferences (global) plus project-specific memories,
        organized into category buckets for easy consumption by agents. The
        combined set is sorted newest-first and paged by ``offset``/``limit``
        so a large project doesn't return one oversized payload.

        Args:
            user_id: User identifier
            project_id: Project identifier
            limit: Max memories across all categories for this page. ``None``
                returns everything from ``offset`` on (legacy behavior).
            offset: Number of (newest-first) memories to skip.

        Returns:
            ContextResponse with the page bucketed by category plus pagination
            metadata (``total``, ``returned``, ``offset``, ``limit``,
            ``has_more``).
        """
        m = self._get_memory()

        # Get global memories
        # mem0 v2.0.2: ``user_id`` must live inside ``filters`` (top-level
        # rejected) and ``limit`` was renamed to ``top_k``.
        global_result = m.get_all(
            filters={"user_id": user_id, "metadata.scope": "global"},
            top_k=200,
        )

        # Get project memories
        project_result = m.get_all(
            filters={"user_id": user_id, "metadata.project_id": project_id},
            top_k=200,
        )

        # Flatten to a deterministically ordered list (newest first) so paging
        # is stable across calls. created_at may be absent → fall back to id.
        flat: list[tuple[str, MemoryResponse]] = []
        for result_set in [global_result, project_result]:
            for mem in self._extract_memory_list(result_set):
                response = self._mem_to_response(mem)
                # Bucket by the response's resolved category — `_mem_to_response`
                # unwraps mem0's nested `{metadata: {metadata: {...}}}` shape, so
                # reading raw `mem["metadata"]` here would mis-bucket as the
                # default whenever the category lives one level deeper.
                cat = getattr(response, "category", None) or "personal_fact"
                flat.append((cat, response))

        flat.sort(
            key=lambda cr: (
                str(getattr(cr[1], "created_at", "") or ""),
                str(getattr(cr[1], "id", "") or ""),
            ),
            reverse=True,
        )

        total = len(flat)
        offset = max(0, offset)
        # Normalize a non-positive limit to 1: otherwise the page is empty while
        # has_more stays True, and a client advancing by `offset += returned`
        # never progresses (infinite pagination loop).
        if limit is not None and limit < 1:
            limit = 1
        page = flat[offset:] if limit is None else flat[offset : offset + limit]

        categories: dict[str, list[MemoryResponse]] = {}
        for cat, response in page:
            categories.setdefault(cat, []).append(response)

        standards = self._get_standards(project_id=project_id)
        if standards:
            _audit_log.info(
                "standards_served",
                user_id=user_id,
                project_id=project_id,
                count=len(standards),
            )

        return ContextResponse(
            user_id=user_id,
            project_id=project_id,
            categories=categories,
            standards=standards,
            total=total,
            returned=len(page),
            offset=offset,
            limit=limit,
            has_more=(offset + len(page)) < total,
        )

    def get_global_context(self, user_id: str) -> ContextResponse:
        """Get only global user context (preferences, skills, etc.).

        Args:
            user_id: User identifier

        Returns:
            ContextResponse with global memories organized by category.
        """
        m = self._get_memory()

        # mem0 v2.0.2 kwarg drift — see list_memories below.
        result = m.get_all(
            filters={"user_id": user_id, "metadata.scope": "global"},
            top_k=200,
        )

        categories: dict[str, list[MemoryResponse]] = {}
        memories = self._extract_memory_list(result)
        for mem in memories:
            response = self._mem_to_response(mem)
            cat = getattr(response, "category", None) or "personal_fact"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(response)

        standards = self._get_standards(project_id=None)
        if standards:
            _audit_log.info(
                "standards_served", user_id=user_id, project_id=None, count=len(standards)
            )

        # Global context isn't paged — report the full set so the pagination
        # metadata isn't misleading (total=0 with non-empty categories).
        count = len(memories)
        return ContextResponse(
            user_id=user_id,
            categories=categories,
            standards=standards,
            total=count,
            returned=count,
            offset=0,
            limit=None,
            has_more=False,
        )

    # ──────────────────────────────────────────────
    # CRUD operations
    # ──────────────────────────────────────────────

    def get_memory(self, memory_id: str) -> MemoryResponse | None:
        """Get a single memory by ID."""
        m = self._get_memory()
        result = m.get(memory_id)
        if not result:
            return None
        return self._mem_to_response(result)

    def get_reasoning_chain(
        self,
        memory_id: str,
        max_depth: int = 3,
        node_cap: int = 50,
    ) -> dict | None:
        """Walk a memory's ``derived_from`` provenance into a reasoning tree.

        A derived memory (dream MERGE survivor, REM insight, or any write that
        supplied ``derived_from``) records its premise memory ids; this
        resolves each premise via the vector store (mem0 ``get`` by id) and
        recurses, so an agent can audit *why* the system believes something —
        Honcho's "a derived memory that can't show its premises is a
        liability" made walkable.

        Each node is ``{memory_id, content (snippet), epistemic_level,
        children}``. A node where the walk stops is marked instead of
        expanded: ``missing`` (premise no longer resolvable — e.g.
        hard-deleted), ``cycle`` (id already on the current path), or
        ``truncated`` ("max_depth" | "node_cap" — unexpanded premises
        remain, re-query with a higher max_depth or start deeper). The
        node budget bounds the TOTAL emitted node count, so the response
        size stays bounded even though internal ``derived_from`` lists are
        uncapped (a wide MERGE fan-in). Tombstoned premises still resolve —
        a merge survivor's provenance must outlive its losers' recall
        visibility.

        Returns None when the root memory itself doesn't exist (callers map
        that to 404).
        """
        max_depth = max(1, int(max_depth))
        budget = {"nodes": 0}
        _SNIPPET = 200

        def _walk(mid: str, depth: int, path: frozenset[str]) -> dict:
            budget["nodes"] += 1
            if mid in path:
                return {"memory_id": mid, "cycle": True, "children": []}
            try:
                mem = self.get_memory(mid)
            except Exception as e:
                logger.debug(f"Reasoning-chain lookup failed for {mid}: {e}")
                mem = None
            if mem is None:
                return {"memory_id": mid, "missing": True, "children": []}
            node: dict = {
                "memory_id": mid,
                "content": (mem.memory or "")[:_SNIPPET],
                "epistemic_level": mem.epistemic_level,
                "children": [],
            }
            premises = mem.derived_from or []
            if premises and depth >= max_depth:
                node["truncated"] = "max_depth"
                return node
            child_path = path | {mid}
            for pid in premises:
                if budget["nodes"] >= node_cap:
                    # Stop emitting entirely — appending one stub per remaining
                    # premise would let a wide fan-in inflate the response
                    # unboundedly despite the budget.
                    node["truncated"] = "node_cap"
                    break
                node["children"].append(_walk(pid, depth + 1, child_path))
            return node

        root = _walk(memory_id, 0, frozenset())
        if root.get("missing"):
            return None
        return root

    def get_memories_by_ids(
        self, ids: list[str], caller_user_id: str | None
    ) -> dict:
        """Batch-fetch full memory payloads by id (C1, layer 3 of the contract).

        One Qdrant ``retrieve`` round-trip for up to ``GET_MEMORIES_MAX_IDS``
        ids. Per-id visibility is enforced with the same rules as search's
        pools; ids the caller may not read are reported in ``missing``
        exactly like nonexistent ids, so this can't be used as an existence
        oracle for other users' private memories. Input order is preserved
        in ``results`` (minus misses); duplicate ids are collapsed.

        Returns ``{"results": [MemoryResponse...], "missing": [id...]}``.
        Raises ValueError on an empty or oversized id list.
        """
        if not ids:
            raise ValueError("ids must be a non-empty list")
        if len(ids) > GET_MEMORIES_MAX_IDS:
            raise ValueError(
                f"At most {GET_MEMORIES_MAX_IDS} ids per call (got {len(ids)})"
            )
        ordered = list(dict.fromkeys(str(i) for i in ids))

        # Qdrant point ids are UUIDs (or ints) — a malformed id would 400 the
        # whole retrieve, so route those straight to `missing` instead.
        valid: list[str] = []
        malformed: list[str] = []
        for mid in ordered:
            try:
                uuid.UUID(mid)
                valid.append(mid)
            except (ValueError, AttributeError, TypeError):
                malformed.append(mid)

        m = self._get_memory()
        points = []
        if valid:
            points = m.vector_store.client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=valid,
                with_payload=True,
                with_vectors=False,
            )
        by_id = {str(getattr(p, "id", "")): p for p in points or []}

        results: list[MemoryResponse] = []
        missing: list[str] = list(malformed)
        for mid in ordered:
            if mid in malformed:
                continue
            point = by_id.get(mid)
            if point is None:
                missing.append(mid)
                continue
            payload = getattr(point, "payload", None) or {}
            if not self._payload_readable_by(payload, caller_user_id):
                # Deliberately indistinguishable from not-found.
                missing.append(mid)
                continue
            results.append(self._point_to_response(point))
        return {"results": results, "missing": missing}

    def _timeline_filter(
        self,
        user_id: str,
        project_id: str | None,
        exclude_id: str | None,
        boundary: datetime,
        direction: str,
    ):
        """Build the Qdrant filter for one side of a timeline window.

        Visibility mirrors search's pool union: caller's own rows (any
        visibility) + shared + standard (when enabled). Other users' private
        and legacy no-visibility rows never match. Tombstoned (dream-
        consolidated) rows are excluded. ``direction`` selects the created_at
        range: strictly-before (``lt``) or at-or-after (``gte``) the boundary
        — ties land on the *after* side, and the anchor itself is excluded
        by id so it never duplicates.
        """
        from qdrant_client.models import (
            DatetimeRange,
            FieldCondition,
            Filter,
            HasIdCondition,
            MatchValue,
        )

        visibility_should = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=MemoryVisibility.SHARED.value),
            ),
        ]
        if settings.standards_enabled:
            visibility_should.append(
                FieldCondition(
                    key="metadata.visibility",
                    match=MatchValue(value=MemoryVisibility.STANDARD.value),
                )
            )
        must: list = [Filter(should=visibility_should)]
        if project_id:
            # Dual-scope, like search(): this project's rows + global rows.
            must.append(
                Filter(
                    should=[
                        FieldCondition(
                            key="metadata.project_id",
                            match=MatchValue(value=project_id),
                        ),
                        FieldCondition(
                            key="metadata.scope", match=MatchValue(value="global")
                        ),
                    ]
                )
            )
        if direction == "before":
            must.append(
                FieldCondition(key="created_at", range=DatetimeRange(lt=boundary))
            )
        else:
            must.append(
                FieldCondition(key="created_at", range=DatetimeRange(gte=boundary))
            )
        must_not: list = [
            FieldCondition(
                key="metadata.dream_tombstoned", match=MatchValue(value=True)
            )
        ]
        if exclude_id:
            must_not.append(HasIdCondition(has_id=[exclude_id]))
        return Filter(must=must, must_not=must_not)

    def _timeline_window(
        self,
        user_id: str,
        project_id: str | None,
        exclude_id: str | None,
        boundary: datetime,
        direction: str,
        limit: int,
    ) -> list[MemoryResponse]:
        """One side of the timeline: the ``limit`` memories nearest the boundary.

        Prefers a single bounded ``order_by`` scroll (qdrant-client ≥1.13 and
        a DATETIME index on created_at — ensured lazily). If the server
        rejects order_by (e.g. index creation raced or is unsupported), falls
        back to scrolling the filtered set (bounded pages) and sorting in
        Python. Returned rows are always sorted ascending by created_at.
        """
        from qdrant_client.models import Direction, OrderBy

        client = self._get_memory().vector_store.client
        flt = self._timeline_filter(user_id, project_id, exclude_id, boundary, direction)
        order = OrderBy(
            key="created_at",
            direction=Direction.DESC if direction == "before" else Direction.ASC,
        )

        points: list = []
        self._ensure_created_at_index()
        try:
            points, _ = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=limit,
                order_by=order,
                with_payload=True,
                with_vectors=False,
            )
            points = list(points or [])
        except Exception as e:
            logger.warning(
                f"Timeline order_by scroll failed ({e}) — falling back to Python sort"
            )
            points = self._scroll_all_sorted(client, flt, direction, limit)

        responses = [self._point_to_response(p) for p in points]
        responses.sort(key=lambda r: _created_at_key(r.created_at))
        return responses

    # Fallback page size / total cap when order_by isn't available. The range
    # filter already bounds the set to one side of the anchor; the cap only
    # guards a pathological pool from unbounded scrolling.
    _TIMELINE_FALLBACK_PAGE = 500
    _TIMELINE_FALLBACK_CAP = 5000

    def _scroll_all_sorted(self, client, flt, direction: str, limit: int) -> list:
        """order_by-less fallback: scroll the filtered set and sort in Python."""
        collected: list = []
        offset = None
        while len(collected) < self._TIMELINE_FALLBACK_CAP:
            page, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=self._TIMELINE_FALLBACK_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(page or [])
            # `is None`, not truthiness: Qdrant's next-page offset is a point
            # id, and integer point ids can legitimately be 0 (falsy) — a
            # truthiness check would end pagination early on such stores.
            if offset is None:
                break

        def _created(p):
            return _created_at_key((getattr(p, "payload", None) or {}).get("created_at"))

        # Nearest-to-boundary first: before → newest first; after → oldest first.
        collected.sort(key=_created, reverse=(direction == "before"))
        return collected[:limit]

    def timeline(
        self,
        anchor: str,
        user_id: str,
        depth: int = 10,
        project_id: str | None = None,
    ) -> dict | None:
        """Chronological window of ±``depth`` memories around an anchor (C2).

        ``anchor`` is a memory id (UUID) or a natural-language query:

        - **id**: fetched directly, with the same per-id visibility rule as
          ``get_memories_by_ids`` — an unreadable or unknown id resolves to
          None (callers map that to 404; no existence oracle).
        - **query**: resolved to the best *vector* search hit (graph edges
          carry no created_at, so they can't anchor a timeline). Search
          already enforces pool visibility and tombstone exclusion.

        The window unions the caller-visible pools (own + shared + standard
        when enabled), excludes dream-tombstoned rows, and — when
        ``project_id`` is given — mirrors search's project+global dual scope.
        Dream insights and session-context rows interleave naturally: they
        are just memories with created_at.

        Returns ``{"anchor_id": str, "memories": [MemoryResponse...]}`` in
        ascending created_at order (anchor included), or None when the
        anchor can't be resolved.
        """
        anchor = (anchor or "").strip()
        if not anchor:
            raise ValueError("anchor must be a memory id or a search query")
        depth = min(max(int(depth), 1), TIMELINE_MAX_DEPTH)

        # ── Resolve the anchor ──
        anchor_mem: MemoryResponse | None = None
        looks_like_id = False
        try:
            uuid.UUID(anchor)
            looks_like_id = True
        except (ValueError, AttributeError, TypeError):
            pass

        if looks_like_id:
            got = self.get_memories_by_ids([anchor], user_id)
            anchor_mem = got["results"][0] if got["results"] else None
        else:
            hits = self.search(
                query=anchor, user_id=user_id, project_id=project_id, limit=5
            )
            anchor_mem = next(
                (
                    h
                    for h in hits
                    if h.source == "vector" and h.id and h.created_at
                ),
                None,
            )
        if anchor_mem is None:
            return None

        # ── Parse the anchor timestamp ──
        boundary: datetime | None = None
        if anchor_mem.created_at:
            try:
                boundary = datetime.fromisoformat(
                    str(anchor_mem.created_at).replace("Z", "+00:00")
                )
                if boundary.tzinfo is None:
                    boundary = boundary.replace(tzinfo=timezone.utc)
            except ValueError:
                boundary = None
        if boundary is None:
            # Un-windowable anchor (no/broken created_at) — degrade to the
            # anchor alone rather than failing the read.
            return {"anchor_id": anchor_mem.id, "memories": [anchor_mem]}

        before = self._timeline_window(
            user_id, project_id, anchor_mem.id, boundary, "before", depth
        )
        after = self._timeline_window(
            user_id, project_id, anchor_mem.id, boundary, "after", depth
        )
        return {
            "anchor_id": anchor_mem.id,
            "memories": [*before, anchor_mem, *after],
        }

    def list_memories(
        self,
        user_id: str,
        scope: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
        include_tombstoned: bool = False,
    ) -> list[MemoryResponse]:
        """List memories with optional filters.

        Audit 27 #14: rows the dreaming sweep consolidated away
        (``metadata.dream_tombstoned=true``) are excluded from listings by
        default, matching every recall path. ``include_tombstoned=True`` is
        the audit escape hatch — it returns the raw set, tombstones included.
        """
        m = self._get_memory()
        self._ensure_filter_indexes()

        # mem0 v2.0.2 ``get_all`` rejects top-level entity kwargs and
        # renamed ``limit`` -> ``top_k`` on its Qdrant wrapper. Same drift
        # pattern as ``Memory.search`` (#46) and
        # ``Memory.vector_store.search`` (#48).
        filters: dict = {"user_id": user_id}
        if scope:
            filters["metadata.scope"] = scope
        if category:
            filters["metadata.category"] = category
        if project_id:
            filters["metadata.project_id"] = project_id

        result = m.get_all(
            filters=filters,
            top_k=limit,
        )

        memories = self._extract_memory_list(result)
        if not include_tombstoned:
            # mem0's filter dict can't express a must_not, so the tombstone
            # exclusion happens here. Tombstones are rare (reversible dream
            # consolidations), so under-filling `limit` slightly beats a
            # second round trip.
            memories = [mem for mem in memories if not _mem_is_tombstoned(mem)]
        return [self._mem_to_response(mem) for mem in memories]

    def list_projects(self, user_id: str) -> list[str]:
        """Return the distinct project_ids the caller can scope memory to.

        Projects in Neuralscape are *implicit*: a project "exists" exactly
        when at least one memory has been stored under its ``project_id``.
        There is no separate project entity to create, update, or delete —
        ``remember(..., project_id="x")`` brings project ``x`` into being and
        ``delete_memories(scope="project", project_id="x")`` removes it.

        Rather than scanning every memory, this derives the list from Neo4j
        ``group_id`` values with an index-backed ``DISTINCT`` query. Each
        project is encoded in the group_id (``user--{uid}--project--{pid}`` for
        the caller's private projects, ``shared--project--{pid}`` for the
        team-wide pool), and Graphiti maintains range indexes on ``group_id``
        per node label (``entity_group_id`` / ``episode_group_id`` /
        ``community_group_id``), so the ``STARTS WITH`` prefix seeks are cheap
        and the database returns only the distinct group_ids (tens), never the
        underlying memories (potentially many thousands).

        Returns the caller's private projects **plus all team-shared
        projects** — the picker can scope to a shared project even before the
        caller has contributed to it. Powers the plugin's `project` selection
        skill (notably in Claude Cowork, which has no working directory to
        derive a ``project_id`` from).
        """
        g = self._get_graphiti()
        if g is None:
            return []

        user_prefix = f"user--{user_id}--project--"
        shared_prefix = "shared--project--"
        # One MATCH per indexed label so the per-label group_id range index is
        # used; UNION dedupes across labels (and is itself DISTINCT).
        cypher = """
        MATCH (n:Entity)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        UNION
        MATCH (n:Episodic)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        UNION
        MATCH (n:Community)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        """

        async def _run():
            async with g.driver.session() as session:
                result = await session.run(
                    cypher, user_prefix=user_prefix, shared_prefix=shared_prefix
                )
                return await result.data()

        coro = _run()
        try:
            records = self._run_on_bridge(coro, timeout=10.0) or []
        except Exception as e:
            # If _run_on_bridge raised before awaiting the coroutine, close it
            # so it doesn't leak / emit "coroutine was never awaited".
            coro.close()
            logger.warning(f"list_projects graph query failed: {e}")
            return []

        projects: set[str] = set()
        for rec in records:
            gid = rec.get("group_id") or ""
            # Both namespaces end with '--project--{pid}'; everything after the
            # separator is the project id. Global groups ('user--{uid}',
            # 'shared') have no separator and are skipped.
            _, sep, pid = gid.partition("--project--")
            if sep and pid.strip():
                projects.add(pid)
        return sorted(projects)

    def _list_null_category_memories(
        self,
        user_id: str,
        scope: str | None = None,
        project_id: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """List memories where metadata.category is null/missing using Qdrant's IsNullCondition."""
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsNullCondition,
            MatchValue,
            PayloadField,
        )

        client = self._memory.vector_store.client
        collection = settings.qdrant_collection

        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            IsNullCondition(is_null=PayloadField(key="metadata.category")),
        ]
        if scope:
            must_conditions.append(
                FieldCondition(key="metadata.scope", match=MatchValue(value=scope))
            )
        if project_id:
            must_conditions.append(
                FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id))
            )

        scroll_filter = Filter(must=must_conditions)
        all_points: list[dict] = []
        offset = None
        while len(all_points) < limit:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=min(100, limit - len(all_points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                payload = pt.payload or {}
                all_points.append({
                    "id": str(pt.id),
                    "data": payload.get("data", ""),
                    "metadata": payload.get("metadata", {}),
                })
            if next_offset is None:
                break
            offset = next_offset

        return all_points

    def get_all_user_ids(self, batch_size: int = 100) -> list[str]:
        """Return every distinct user_id that has at least one memory.

        Qdrant is the authoritative source here (not Neo4j): a memory's author
        lives on the Qdrant point payload, including for SHARED writes — whereas
        the graph's shared group_ids (``shared`` / ``shared--project--{pid}``)
        don't encode the author, so a user who only ever wrote shared memories
        would be invisible to a group_id scan. Qdrant's facet API isn't an
        option either: it requires a keyword payload index on ``user_id``, which
        the collection doesn't maintain.

        So we still scroll the collection (the dedup cron and backfill genuinely
        need every author), but project the payload to ONLY ``user_id`` — turning
        this from "transfer every memory" into "transfer one short string per
        point". The win is in the payload size, not the iteration.
        """
        client = self._memory.vector_store.client
        collection = settings.qdrant_collection

        user_ids: set[str] = set()
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=["user_id"],  # project: only the field we need
                with_vectors=False,
            )
            for pt in points:
                uid = (pt.payload or {}).get("user_id")
                if uid:
                    user_ids.add(uid)
            if next_offset is None:
                break
            offset = next_offset

        return list(user_ids)
