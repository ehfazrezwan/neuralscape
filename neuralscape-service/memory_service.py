"""Business logic layer for neuralscape memory service.

Both REST endpoints and MCP tools call into this same MemoryService.
"""

import json
import logging
from datetime import datetime, timezone

from google import genai

from config import settings
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
        return f"project:{project_id}"
    return "global"


def _get_group_ids(project_id: str | None = None) -> list[str]:
    """Get group_ids to search across (always includes global)."""
    group_ids = ["global"]
    if project_id:
        group_ids.append(f"project:{project_id}")
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

    def _get_memory(self):
        """Lazy-initialize mem0 Memory with Graphiti backend."""
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

    def _run_on_bridge(self, coro):
        """Run an async coroutine on the Graphiti adapter's event loop."""
        if self._bridge is None:
            raise RuntimeError("Graphiti bridge not initialized")
        return self._bridge.run(coro)

    def _get_genai_client(self):
        """Get a Gemini client for fact extraction."""
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
            response = client.models.generate_content(
                model=settings.gemini_llm_model,
                contents=extraction_messages[0]["content"],
            )
            parsed_facts = parse_extraction_response(response.text)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fallback: store the raw conversation through mem0's pipeline
            result = m.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
            return self._result_to_responses(result)

        if not parsed_facts:
            logger.info("No facts extracted from conversation")
            return []

        # Step 2: Store each fact with category metadata
        stored = []
        for category, fact_content in parsed_facts:
            scope = default_scope_for_category(category)

            # Override scope if project_id provided and category allows it
            if project_id and category not in GLOBAL_CATEGORIES:
                scope = MemoryScope.PROJECT

            # Validate: project categories require project_id
            if category in PROJECT_CATEGORIES and not project_id:
                scope = MemoryScope.GLOBAL
                logger.warning(
                    f"Category '{category}' typically requires project_id, "
                    f"storing as global"
                )

            metadata = {
                "scope": scope.value,
                "category": category,
                "project_id": project_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "source": "conversation",
            }

            try:
                result = m.add(
                    messages=[{"role": "user", "content": fact_content}],
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    metadata=metadata,
                    infer=False,
                )
                responses = self._result_to_responses(result, category=category, scope=scope.value)
                stored.extend(responses)
            except Exception as e:
                logger.error(f"Failed to store fact '{fact_content[:50]}...': {e}")

        # Step 3: In parallel, add raw conversation text to knowledge graph
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
                self._memory.graph.add(
                    data=raw_text,
                    filters={"user_id": user_id, "group_id": group_id},
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

        metadata = {
            "scope": scope,
            "category": category,
            "project_id": project_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "source": "explicit",
        }
        if tags:
            metadata["tags"] = tags

        result = m.add(
            messages=[{"role": "user", "content": content}],
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            infer=False,
        )

        # Also add to knowledge graph
        group_id = _build_group_id(scope, project_id)
        try:
            if self._graphiti and self._bridge:
                self._memory.graph.add(
                    data=content,
                    filters={"user_id": user_id, "group_id": group_id},
                )
        except Exception as e:
            logger.warning(f"Graph storage failed (non-critical): {e}")

        return self._result_to_responses(result, category=category, scope=scope)

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

        return vector_responses + graph_responses

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
            config = SearchConfig(**search_config)
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

        if content:
            m.update(memory_id, content)

        # For metadata updates (category, tags), we'd need to update the
        # vector store metadata directly. mem0's update only handles content.
        # For now, content updates go through mem0, metadata updates are
        # noted but require direct vector store access.
        if category and category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        return {"message": "Memory updated successfully"}

    def delete_memory(self, memory_id: str) -> dict:
        """Delete a single memory by ID."""
        m = self._get_memory()
        return m.delete(memory_id)

    def delete_memories(
        self,
        user_id: str,
        scope: str | None = None,
        category: str | None = None,
        project_id: str | None = None,
    ) -> dict:
        """Bulk delete memories with filters."""
        m = self._get_memory()

        if not scope and not category and not project_id:
            # Delete all for user
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
                m.delete(mem.id)
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
