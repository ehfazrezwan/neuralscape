"""Business logic layer for neuralscape memory service.

Both REST endpoints and MCP tools call into this same MemoryService.
"""

import hashlib
import json
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timezone

from google import genai

from config import settings


# HTTP status codes / error substrings that indicate transient failures
_TRANSIENT_PATTERNS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "rate limit", "overloaded", "capacity", "timed out", "timeout")


def _is_transient(exc: Exception) -> bool:
    """Check if an exception looks like a transient/retryable API error."""
    msg = str(exc)
    return any(p.lower() in msg.lower() for p in _TRANSIENT_PATTERNS)


def retry_transient(
    fn,
    *args,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    operation: str = "operation",
    fallback_model: str | None = None,
    model_kwarg: str = "model",
    **kwargs,
):
    """Call fn(*args, **kwargs) with exponential backoff on transient errors.

    Non-transient exceptions are raised immediately.

    If fallback_model is provided and the primary model exhausts all retries
    on transient errors, the function is retried once more with the model kwarg
    swapped to the fallback model.
    """
    if max_retries is None:
        max_retries = settings.llm_max_retries
    if base_delay is None:
        base_delay = settings.llm_retry_base_delay
    if max_delay is None:
        max_delay = settings.llm_retry_max_delay

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not _is_transient(e):
                raise
            if attempt == max_retries:
                break  # exhausted retries — try fallback below
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                f"Transient error in {operation} (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {delay:.1f}s: {e}"
            )
            time.sleep(delay)

    # Primary model exhausted retries — try fallback model if configured
    if fallback_model and model_kwarg in kwargs:
        primary_model = kwargs[model_kwarg]
        if primary_model != fallback_model:
            logger.warning(
                f"Primary model {primary_model} exhausted retries for {operation}, "
                f"falling back to {fallback_model}"
            )
            kwargs[model_kwarg] = fallback_model
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Fallback model {fallback_model} also failed for {operation}: {e}"
                )
                raise

    raise last_exc  # type: ignore[misc]
from prompts import (
    build_extraction_messages,
    parse_extraction_response,
)
from schemas import (
    GLOBAL_CATEGORIES,
    MEMORY_CATEGORIES,
    PROJECT_CATEGORIES,
    ContextResponse,
    MemoryResponse,
    MemoryScope,
    default_scope_for_category,
)

logger = logging.getLogger(__name__)


def _build_group_id(scope: str, project_id: str | None = None) -> str:
    """Build a Graphiti group_id from scope and project_id."""
    if scope == MemoryScope.PROJECT and project_id:
        return f"project--{project_id}"
    return "global"


def _get_group_ids(project_id: str | None = None) -> list[str]:
    """Get group_ids to search across (always includes global)."""
    group_ids = ["global"]
    if project_id:
        group_ids.append(f"project--{project_id}")
    return group_ids


class MemoryService:
    """Encapsulates all memory business logic.

    Provides a unified interface used by both REST endpoints and MCP tools.
    """

    def __init__(self):
        self._memory = None
        self._graphiti = None
        self._bridge = None
        self._genai_model = None
        self._init_lock = threading.Lock()

    def _get_memory(self):
        """Lazy-initialize mem0 Memory with Graphiti backend.

        Thread-safe: uses a lock to prevent double-initialization on
        concurrent cold-start requests.
        """
        if self._memory is None:
            with self._init_lock:
                if self._memory is None:
                    from mem0 import Memory

                    config = settings.get_mem0_config()
                    self._memory = Memory.from_config(config)

                    if hasattr(self._memory, "graph") and hasattr(self._memory.graph, "graphiti"):
                        self._graphiti = self._memory.graph.graphiti
                        self._bridge = self._memory.graph._bridge

        return self._memory

    def _get_graphiti(self):
        """Get the underlying Graphiti instance."""
        self._get_memory()
        return self._graphiti

    def _run_on_bridge(self, coro, timeout: float = 30.0):
        """Run an async coroutine on the Graphiti adapter's event loop.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait for the result (default 30s).

        Raises:
            RuntimeError: If the bridge is not initialized.
            TimeoutError: If the operation exceeds the timeout.
        """
        if self._bridge is None:
            raise RuntimeError("Graphiti bridge not initialized")
        import asyncio as _asyncio
        import concurrent.futures

        future = _asyncio.run_coroutine_threadsafe(coro, self._bridge._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.error(f"Bridge call timed out after {timeout}s")
            raise TimeoutError(f"Graph operation timed out after {timeout}s")

    def _get_genai_client(self):
        """Get a Gemini client for fact extraction.

        Thread-safe: reuses _init_lock to prevent double-initialization.
        """
        if self._genai_model is None:
            with self._init_lock:
                if self._genai_model is None:
                    self._genai_model = genai.Client(api_key=settings.google_api_key)
        return self._genai_model

    def close(self):
        """Clean up resources."""
        if self._graphiti and self._bridge:
            self._bridge.run(self._graphiti.close())

    # ──────────────────────────────────────────────
    # Store operations
    # ──────────────────────────────────────────────

    def extract_and_store(
        self,
        messages: list[dict],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[MemoryResponse]:
        """Extract facts from conversation via LLM, then store each with category metadata.

        Args:
            messages: Conversation messages [{role, content}, ...]
            user_id: User identifier
            project_id: Optional project identifier
            agent_id: Optional agent identifier for provenance
            run_id: Optional session identifier

        Returns:
            List of stored memory responses.
        """
        m = self._get_memory()

        # Step 1: Call Gemini for fact extraction
        extraction_messages = build_extraction_messages(messages)
        client = self._get_genai_client()

        try:
            from google.genai.types import GenerateContentConfig, HttpOptions

            response = retry_transient(
                client.models.generate_content,
                model=settings.gemini_llm_model,
                contents=extraction_messages[0]["content"],
                config=GenerateContentConfig(
                    http_options=HttpOptions(timeout=60_000),  # milliseconds
                ),
                operation="LLM extraction",
                fallback_model=settings.gemini_llm_fallback_model,
            )
            parsed_facts = parse_extraction_response(response.text)
        except Exception as e:
            logger.error(f"LLM extraction failed after retries: {e}")
            # Fallback: store the raw conversation through mem0's pipeline
            result = retry_transient(
                m.add,
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                operation="mem0 fallback add",
            )
            return self._result_to_responses(result)

        if not parsed_facts:
            logger.info("No facts extracted from conversation")
            return []

        # Step 2: Batch-store all facts (single embed + single Qdrant upsert)
        try:
            stored = self._batch_store_facts(
                facts=parsed_facts,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                source="conversation",
            )
        except Exception as e:
            logger.error(f"Batch store failed: {e}")
            stored = []

        # Step 3: Add raw conversation text to knowledge graph
        group_id = _build_group_id(
            MemoryScope.PROJECT.value if project_id else MemoryScope.GLOBAL.value,
            project_id,
        )
        raw_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        )
        try:
            if self._graphiti and self._bridge:
                retry_transient(
                    self._memory.graph.add,
                    data=raw_text,
                    filters={"user_id": user_id, "group_id": group_id},
                    operation="graph storage (extract_and_store)",
                )
        except Exception as e:
            logger.warning(f"Graph storage failed (non-critical): {e}")

        return stored

    def store_raw(
        self,
        content: str,
        user_id: str,
        category: str,
        scope: str = "global",
        project_id: str | None = None,
        tags: list[str] | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[MemoryResponse]:
        """Store a single pre-categorized fact directly (no LLM extraction).

        Args:
            content: The fact to store
            user_id: User identifier
            category: Memory category (validated against MEMORY_CATEGORIES)
            scope: "global" or "project"
            project_id: Required when scope="project"
            tags: Optional free-form tags
            agent_id: Optional agent identifier
            run_id: Optional session identifier

        Returns:
            List of stored memory responses.
        """
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of: {list(MEMORY_CATEGORIES.keys())}")

        if scope == "project" and not project_id:
            raise ValueError("project_id is required when scope='project'")

        m = self._get_memory()
        now_iso = datetime.now(timezone.utc).isoformat()

        # ── Direct embed + Qdrant insert (bypass m.add) ──
        mid = str(uuid.uuid4())
        embedding = m.embedding_model.embed(content, memory_action="add")

        payload = {
            "data": content,
            "hash": hashlib.md5(content.encode()).hexdigest(),
            "created_at": now_iso,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": {
                "scope": scope,
                "category": category,
                "project_id": project_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "source": "explicit",
            },
        }
        if tags:
            payload["metadata"]["tags"] = tags

        m.vector_store.insert(
            vectors=[embedding],
            ids=[mid],
            payloads=[payload],
        )

        try:
            m.db.add_history(mid, None, content, "ADD", created_at=now_iso)
        except Exception as e:
            logger.warning(f"History record failed for {mid}: {e}")

        # ── Add to knowledge graph ──
        group_id = _build_group_id(scope, project_id)
        try:
            if self._graphiti and self._bridge:
                retry_transient(
                    self._memory.graph.add,
                    data=content,
                    filters={"user_id": user_id, "group_id": group_id},
                    operation="store_raw graph add",
                )
        except Exception as e:
            logger.warning(f"Graph storage failed (non-critical): {e}")

        return [
            MemoryResponse(
                id=mid,
                memory=content,
                category=category,
                scope=scope,
                project_id=project_id,
                source="vector",
                created_at=now_iso,
            )
        ]

    def _batch_store_facts(
        self,
        facts: list[tuple[str, str]],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        source: str = "conversation",
    ) -> list[MemoryResponse]:
        """Store multiple categorized facts via a single batch embed + single Qdrant upsert.

        Bypasses mem0's per-fact m.add() pipeline which triggers per-fact graph
        ingestion.  The caller is responsible for a separate graph.add() call
        with the full conversation text.

        Args:
            facts: List of (category, content) tuples.
            user_id: User identifier.
            project_id: Optional project identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run/session identifier.
            source: Provenance tag for metadata.

        Returns:
            List of MemoryResponse objects for stored facts.
        """
        if not facts:
            return []

        m = self._get_memory()
        now_iso = datetime.now(timezone.utc).isoformat()

        # ── Build per-fact metadata, IDs, and texts ──
        texts: list[str] = []
        memory_ids: list[str] = []
        payloads: list[dict] = []
        fact_meta: list[tuple[str, str]] = []  # (category, scope) per fact

        for category, content in facts:
            scope = default_scope_for_category(category)

            if project_id and category not in GLOBAL_CATEGORIES:
                scope = MemoryScope.PROJECT

            if category in PROJECT_CATEGORIES and not project_id:
                scope = MemoryScope.GLOBAL
                logger.warning(
                    f"Category '{category}' typically requires project_id, storing as global"
                )

            mid = str(uuid.uuid4())
            payload = {
                "data": content,
                "hash": hashlib.md5(content.encode()).hexdigest(),
                "created_at": now_iso,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": {
                    "scope": scope.value if isinstance(scope, MemoryScope) else scope,
                    "category": category,
                    "project_id": project_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "source": source,
                },
            }

            texts.append(content)
            memory_ids.append(mid)
            payloads.append(payload)
            fact_meta.append((category, scope.value if isinstance(scope, MemoryScope) else scope))

        # ── Single batch embed ──
        embeddings = m.embedding_model.embed_batch(texts, memory_action="add")

        # ── Single Qdrant upsert ──
        m.vector_store.insert(
            vectors=embeddings,
            ids=memory_ids,
            payloads=payloads,
        )

        # ── Record history entries ──
        for mid, content in zip(memory_ids, texts):
            try:
                m.db.add_history(mid, None, content, "ADD", created_at=now_iso)
            except Exception as e:
                logger.warning(f"History record failed for {mid}: {e}")

        # ── Build responses ──
        responses: list[MemoryResponse] = []
        for mid, content, (category, scope_val) in zip(memory_ids, texts, fact_meta):
            responses.append(
                MemoryResponse(
                    id=mid,
                    memory=content,
                    category=category,
                    scope=scope_val,
                    project_id=project_id,
                    source="vector",
                    created_at=now_iso,
                )
            )

        logger.info(
            f"Batch-stored {len(responses)} facts for user={user_id} "
            f"(1 embed call, 1 Qdrant upsert)"
        )
        return responses

    # ──────────────────────────────────────────────
    # Search operations
    # ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        project_id: str | None = None,
        categories: list[str] | None = None,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[MemoryResponse]:
        """Semantic search across memories with scope/category filters.

        When project_id is provided, searches both global and project memories.

        Args:
            query: Search query
            user_id: User identifier
            project_id: Optional project identifier (searches global + project)
            categories: Optional category filter
            scope: Optional scope filter ("global" or "project")
            limit: Maximum results

        Returns:
            List of matching memory responses sorted by score.
        """
        m = self._get_memory()

        # Build metadata filters
        filters = {}
        if categories:
            filters["category"] = {"in": categories}
        if scope:
            filters["scope"] = scope

        # If project_id is given and no explicit scope, search both scopes
        if project_id and not scope:
            # Search project-scoped memories
            project_filters = {**filters, "project_id": project_id}
            project_results = m.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=project_filters,
            )

            # Search global memories
            global_filters = {**filters, "scope": "global"}
            global_results = m.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=global_filters,
            )

            # Merge by score, deduplicate by ID
            all_results = self._merge_results(project_results, global_results)
            vector_responses = self._results_to_responses(all_results[:limit])
        else:
            if project_id:
                filters["project_id"] = project_id

            results = m.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters if filters else None,
            )
            vector_responses = self._results_to_responses(results)

        # Also query the knowledge graph and merge edge facts
        graph_responses: list[MemoryResponse] = []
        try:
            graph_results = self.search_graph(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=limit,
            )
            for edge in graph_results.get("edges", []):
                graph_responses.append(
                    MemoryResponse(
                        id=edge.get("uuid", ""),
                        memory=edge.get("fact", edge.get("name", "")),
                        source="graph",
                    )
                )
        except Exception as e:
            logger.warning(f"Graph search failed during recall (non-critical): {e}")

        # Deduplicate and enforce caller's limit
        combined = self._deduplicate_responses(vector_responses, graph_responses)
        return combined[:limit]

    def search_graph(
        self,
        query: str,
        user_id: str,
        project_id: str | None = None,
        limit: int = 10,
        search_config: dict | None = None,
    ) -> dict:
        """Knowledge graph search via Graphiti.

        Args:
            query: Search query
            user_id: User identifier
            project_id: Optional project to include in search scope
            limit: Maximum results
            search_config: Optional SearchConfig dict override

        Returns:
            Dict with edges, nodes, episodes, communities.
        """
        g = self._get_graphiti()
        if g is None:
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

        from graphiti_core.search.search_config import SearchConfig
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        group_ids = _get_group_ids(project_id)

        if search_config:
            try:
                config = SearchConfig(**search_config)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid search_config, falling back to default: {e}")
                config = EDGE_HYBRID_SEARCH_RRF
        else:
            config = EDGE_HYBRID_SEARCH_RRF

        config.limit = limit

        try:
            results = self._run_on_bridge(
                g.search_(
                    query=query,
                    config=config,
                    group_ids=group_ids,
                )
            )

            return {
                "edges": [
                    {"uuid": e.uuid, "name": e.name, "fact": e.fact}
                    for e in results.edges
                ],
                "nodes": [
                    {"uuid": n.uuid, "name": n.name, "summary": n.summary}
                    for n in results.nodes
                ],
                "episodes": [
                    {"uuid": ep.uuid, "name": ep.name, "content": ep.content}
                    for ep in results.episodes
                ],
                "communities": [
                    {"uuid": c.uuid, "name": c.name}
                    for c in results.communities
                ],
            }
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    # ──────────────────────────────────────────────
    # Context operations
    # ──────────────────────────────────────────────

    def get_project_context(self, user_id: str, project_id: str) -> ContextResponse:
        """Get full project + global context organized by category.

        Retrieves all user preferences (global) plus project-specific memories,
        organized into category buckets for easy consumption by agents.

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            ContextResponse with memories organized by category.
        """
        m = self._get_memory()

        # Get global memories
        global_result = m.get_all(
            user_id=user_id,
            filters={"scope": "global"},
            limit=200,
        )

        # Get project memories
        project_result = m.get_all(
            user_id=user_id,
            filters={"project_id": project_id},
            limit=200,
        )

        # Organize by category
        categories: dict[str, list[MemoryResponse]] = {}

        for result_set in [global_result, project_result]:
            memories = self._extract_memory_list(result_set)
            for mem in memories:
                metadata = mem.get("metadata", {}) or {}
                cat = metadata.get("category", "personal_fact")
                response = self._mem_to_response(mem)
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(response)

        return ContextResponse(
            user_id=user_id,
            project_id=project_id,
            categories=categories,
        )

    def get_global_context(self, user_id: str) -> ContextResponse:
        """Get only global user context (preferences, skills, etc.).

        Args:
            user_id: User identifier

        Returns:
            ContextResponse with global memories organized by category.
        """
        m = self._get_memory()

        result = m.get_all(
            user_id=user_id,
            filters={"scope": "global"},
            limit=200,
        )

        categories: dict[str, list[MemoryResponse]] = {}
        memories = self._extract_memory_list(result)
        for mem in memories:
            metadata = mem.get("metadata", {}) or {}
            cat = metadata.get("category", "personal_fact")
            response = self._mem_to_response(mem)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(response)

        return ContextResponse(
            user_id=user_id,
            categories=categories,
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

    def list_memories(
        self,
        user_id: str,
        scope: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryResponse]:
        """List memories with optional filters."""
        m = self._get_memory()

        filters = {}
        if scope:
            filters["scope"] = scope
        if category:
            filters["category"] = category
        if project_id:
            filters["project_id"] = project_id

        result = m.get_all(
            user_id=user_id,
            filters=filters if filters else None,
            limit=limit,
        )

        memories = self._extract_memory_list(result)
        return [self._mem_to_response(mem) for mem in memories]

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Update a memory's content or metadata."""
        m = self._get_memory()

        if category and category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        # Fetch existing memory for user_id and metadata (needed for graph group_id)
        existing = m.get(memory_id) if content else None

        if content:
            m.update(memory_id, content)

        # Re-ingest updated content into the knowledge graph so Graphiti's
        # contradiction detection can expire stale edges and create new ones.
        if content and existing and self._graphiti and self._bridge:
            try:
                metadata = existing.get("metadata", {}) or {}
                scope = metadata.get("scope", "global")
                project_id = metadata.get("project_id")
                user_id = existing.get("user_id", "")
                group_id = _build_group_id(scope, project_id)
                retry_transient(
                    self._memory.graph.add,
                    data=content,
                    filters={"user_id": user_id, "group_id": group_id},
                    operation=f"graph re-ingestion for {memory_id}",
                )
            except Exception as e:
                logger.warning(f"Graph re-ingestion failed for {memory_id} (non-critical): {e}")

        return {"message": "Memory updated successfully"}

    def delete_memory(self, memory_id: str) -> dict:
        """Delete a single memory by ID from both vector store and graph."""
        m = self._get_memory()

        # First, get the memory content to find related graph edges
        mem = m.get(memory_id)
        result = m.delete(memory_id)

        # Expire related graph edges (soft-delete, non-critical)
        if mem and self._graphiti and self._bridge:
            try:
                self._expire_graph_edges_for_memory(mem)
            except Exception as e:
                logger.warning(f"Graph edge expiration failed for {memory_id} (non-critical): {e}")

        return result

    def delete_memories(
        self,
        user_id: str,
        scope: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Bulk delete memories with filters from both vector store and graph."""
        m = self._get_memory()

        if not scope and not category and not project_id:
            # Delete all for user — also expire all graph edges in user's groups
            if self._graphiti and self._bridge:
                group_ids = _get_group_ids(project_id)
                self._expire_graph_edges_for_groups(group_ids)
            m.delete_all(user_id=user_id)
            return {"message": "All memories deleted"}

        # For filtered deletes, we need to list then delete individually
        memories = self.list_memories(
            user_id=user_id,
            scope=scope,
            category=category,
            project_id=project_id,
        )

        deleted_count = 0
        for mem in memories:
            try:
                # Get full memory for graph cleanup before deleting
                full_mem = m.get(mem.id)
                m.delete(mem.id)
                if full_mem and self._graphiti and self._bridge:
                    self._expire_graph_edges_for_memory(full_mem)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {mem.id}: {e}")

        return {"message": f"Deleted {deleted_count} memories"}

    # ──────────────────────────────────────────────
    # Graph introspection
    # ──────────────────────────────────────────────

    def get_graph_nodes(
        self, user_id: str, project_id: str | None = None, limit: int = 50
    ) -> list[dict]:
        """List entity nodes from Graphiti."""
        g = self._get_graphiti()
        if g is None:
            return []

        from graphiti_core.nodes import EntityNode

        group_ids = _get_group_ids(project_id)

        try:
            nodes = self._run_on_bridge(
                EntityNode.get_by_group_ids(g.driver, group_ids=group_ids, limit=limit)
            )
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

    def get_graph_edges(
        self, user_id: str, project_id: str | None = None, limit: int = 50
    ) -> list[dict]:
        """List entity edges (facts) from Graphiti."""
        g = self._get_graphiti()
        if g is None:
            return []

        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        group_ids = _get_group_ids(project_id)

        try:
            edges = self._run_on_bridge(
                EntityEdge.get_by_group_ids(g.driver, group_ids=group_ids, limit=limit)
            )
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

    def get_graph_episodes(
        self, user_id: str, project_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List episodic nodes from Graphiti."""
        g = self._get_graphiti()
        if g is None:
            return []

        group_ids = _get_group_ids(project_id)
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

        group_ids = _get_group_ids(project_id)

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

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _extract_memory_list(self, result) -> list[dict]:
        """Extract the list of memories from a mem0 result (handles both v1.0 and v1.1 formats)."""
        if isinstance(result, dict):
            return result.get("results", [])
        if isinstance(result, list):
            return result
        return []

    def _mem_to_response(self, mem: dict) -> MemoryResponse:
        """Convert a mem0 memory dict to a MemoryResponse."""
        metadata = mem.get("metadata", {}) or {}
        return MemoryResponse(
            id=mem.get("id", ""),
            memory=mem.get("memory", ""),
            category=metadata.get("category"),
            scope=metadata.get("scope"),
            project_id=metadata.get("project_id"),
            tags=metadata.get("tags"),
            score=mem.get("score"),
            created_at=mem.get("created_at"),
            updated_at=mem.get("updated_at"),
            source="vector",
        )

    def _result_to_responses(
        self, result, category: str | None = None, scope: str | None = None
    ) -> list[MemoryResponse]:
        """Convert a mem0 add() result to MemoryResponse list."""
        memories = self._extract_memory_list(result)
        responses = []
        for mem in memories:
            resp = self._mem_to_response(mem)
            if category and not resp.category:
                resp.category = category
            if scope and not resp.scope:
                resp.scope = scope
            responses.append(resp)
        return responses

    def _results_to_responses(self, results) -> list[MemoryResponse]:
        """Convert mem0 search/get_all results to MemoryResponse list."""
        memories = self._extract_memory_list(results)
        return [self._mem_to_response(mem) for mem in memories]

    def _deduplicate_responses(
        self,
        vector_responses: list[MemoryResponse],
        graph_responses: list[MemoryResponse],
    ) -> list[MemoryResponse]:
        """Deduplicate and interleave vector and graph results.

        Removes graph results whose content closely matches a vector result,
        then interleaves the remaining results (vector-1, graph-1, vector-2, ...).
        """
        seen_content: set[str] = set()
        unique_graph: list[MemoryResponse] = []

        # Index vector content for fuzzy matching
        for vr in vector_responses:
            seen_content.add(vr.memory.strip().lower())

        for gr in graph_responses:
            normalized = gr.memory.strip().lower()
            # Skip if exact or substring match with any vector result
            is_duplicate = False
            for vc in seen_content:
                if normalized == vc or normalized in vc or vc in normalized:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_graph.append(gr)
                seen_content.add(normalized)

        # Interleave: vector-1, graph-1, vector-2, graph-2, ...
        interleaved: list[MemoryResponse] = []
        vi, gi = 0, 0
        while vi < len(vector_responses) or gi < len(unique_graph):
            if vi < len(vector_responses):
                interleaved.append(vector_responses[vi])
                vi += 1
            if gi < len(unique_graph):
                interleaved.append(unique_graph[gi])
                gi += 1

        return interleaved

    def _expire_graph_edges_for_memory(self, mem: dict) -> None:
        """Soft-delete graph edges related to a memory by setting expired_at."""
        content = mem.get("memory", "")
        if not content:
            return
        try:
            from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

            config = EDGE_HYBRID_SEARCH_RRF
            config.limit = 5

            metadata = mem.get("metadata", {}) or {}
            group_ids = _get_group_ids(metadata.get("project_id"))

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
                self._expire_graph_edges_for_memory(mem)
            except Exception as e:
                logger.warning(f"Graph cleanup failed for {memory_id} (non-critical): {e}")

    def get_all_user_ids(self, batch_size: int = 100) -> list[str]:
        """Scroll the entire Qdrant collection and return unique user_ids."""
        client = self._memory.vector_store.client
        collection = settings.qdrant_collection

        user_ids: set[str] = set()
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
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

    def dedup_memories(self, user_id: str) -> dict:
        """Remove duplicate memories for a user in two phases.

        Phase 1 — Exact: group by payload hash, keep newest, delete rest.
        Phase 2 — Semantic: for each remaining memory, search for near-duplicates
                  above the cosine threshold, delete the older one.

        Returns:
            Dict with user_id, exact_duplicates_removed, semantic_duplicates_removed,
            total_checked.
        """
        m = self._get_memory()
        threshold = settings.dedup_similarity_threshold
        batch_size = settings.dedup_batch_size

        memories = self._scroll_all_user_memories(user_id, batch_size=batch_size)
        deleted_ids: set[str] = set()

        # ── Phase 1: Exact dedup by hash ──
        exact_removed = 0
        hash_groups: dict[str, list[dict]] = {}
        for mem in memories:
            h = mem["payload"].get("hash")
            if h:
                hash_groups.setdefault(h, []).append(mem)

        for h, group in hash_groups.items():
            if len(group) < 2:
                continue
            # Sort by created_at descending — keep the first (newest)
            group.sort(
                key=lambda x: x["payload"].get("created_at", ""),
                reverse=True,
            )
            for dup in group[1:]:
                mid = dup["id"]
                if mid in deleted_ids:
                    continue
                try:
                    self._delete_qdrant_memory_with_graph_cleanup(mid, dup["payload"])
                    deleted_ids.add(mid)
                    exact_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to delete exact dup {mid}: {e}")

        # ── Phase 2: Semantic dedup ──
        semantic_removed = 0
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
                hits = m.vector_store.search(
                    query=text,
                    vectors=embedding,
                    limit=5,
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

                # Delete the older one
                mem_created = mem["payload"].get("created_at", "")
                hit_created = hit_payload.get("created_at", "")
                older_id, older_payload = (
                    (hit_id, hit_payload) if hit_created <= mem_created else (mid, mem["payload"])
                )

                if older_id in deleted_ids:
                    continue
                try:
                    self._delete_qdrant_memory_with_graph_cleanup(older_id, older_payload)
                    deleted_ids.add(older_id)
                    semantic_removed += 1
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

    def _merge_results(self, *result_sets) -> list[dict]:
        """Merge multiple result sets, deduplicate by ID, sort by score descending."""
        seen_ids = set()
        merged = []

        for result_set in result_sets:
            memories = self._extract_memory_list(result_set)
            for mem in memories:
                mem_id = mem.get("id")
                if mem_id and mem_id not in seen_ids:
                    seen_ids.add(mem_id)
                    merged.append(mem)

        # Sort by score (descending) if available
        merged.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
        return merged
