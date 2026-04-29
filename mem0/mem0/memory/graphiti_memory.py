import asyncio
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.errors import GroupsEdgesNotFoundError
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.nodes import EntityNode, EpisodeType
    from graphiti_core.search.search_config import SearchConfig
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    from graphiti_core.search.search_filters import SearchFilters
except ImportError:
    raise ImportError(
        "graphiti-core is required for the Graphiti graph memory provider. "
        "Install it with: pip install 'mem0ai[graphiti]'"
    )


def _create_llm_client(
    provider: str,
    model: str | None,
    api_key: str | None,
    fallback_model: str | None = None,
):
    """Create a Graphiti LLM client based on provider name."""
    if provider == "gemini":
        from graphiti_core.llm_client.gemini_client import GeminiClient

        config = LLMConfig(api_key=api_key, model=model, fallback_model=fallback_model)
        return GeminiClient(config=config)
    elif provider == "openai":
        from graphiti_core.llm_client.openai_client import OpenAIClient

        config = LLMConfig(api_key=api_key, model=model, fallback_model=fallback_model)
        return OpenAIClient(config=config)
    elif provider == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        config = LLMConfig(api_key=api_key, model=model, fallback_model=fallback_model)
        return AnthropicClient(config=config)
    elif provider == "groq":
        from graphiti_core.llm_client.groq_client import GroqClient

        config = LLMConfig(api_key=api_key, model=model, fallback_model=fallback_model)
        return GroqClient(config=config)
    else:
        raise ValueError(f"Unsupported Graphiti LLM provider: {provider}")


def _create_embedder(provider: str, model: str | None, api_key: str | None):
    """Create a Graphiti embedder client based on provider name."""
    if provider == "gemini":
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        config = GeminiEmbedderConfig(api_key=api_key)
        if model:
            config.embedding_model = model
        return GeminiEmbedder(config=config)
    elif provider == "openai":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        config = OpenAIEmbedderConfig(api_key=api_key)
        if model:
            config.embedding_model = model
        return OpenAIEmbedder(config=config)
    elif provider == "voyage":
        from graphiti_core.embedder.voyage import VoyageEmbedder, VoyageEmbedderConfig

        config = VoyageEmbedderConfig(api_key=api_key)
        if model:
            config.embedding_model = model
        return VoyageEmbedder(config=config)
    else:
        raise ValueError(f"Unsupported Graphiti embedder provider: {provider}")


def _create_cross_encoder(provider: str | None, api_key: str | None):
    """Create a Graphiti cross-encoder/reranker client."""
    if provider is None or provider == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        return OpenAIRerankerClient(api_key=api_key)
    elif provider == "gemini":
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

        config = LLMConfig(api_key=api_key)
        return GeminiRerankerClient(config=config)
    elif provider == "bge":
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()
    else:
        raise ValueError(f"Unsupported Graphiti reranker provider: {provider}")


class _AsyncBridge:
    """Runs a dedicated event loop in a background thread.

    All Graphiti async operations (including the Neo4j driver) are executed
    on this single loop, avoiding the 'Future attached to a different loop'
    error that occurs when mixing event loops.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """Submit a coroutine to the background loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


class MemoryGraph:
    """Graphiti-backed MemoryGraph adapter for mem0.

    Implements the same interface as mem0's default MemoryGraph (graph_memory.py)
    but delegates all graph operations to Graphiti's temporal knowledge graph engine.

    Graphiti handles entity extraction, deduplication, edge invalidation, and
    community detection — bypassing mem0's built-in LLM extraction pipeline.

    The Graphiti instance is exposed as `self.graphiti` for advanced usage
    (communities, sagas, search recipes, etc.).
    """

    def __init__(self, config):
        self.config = config
        graph_config = self.config.graph_store.config

        # Create a dedicated async bridge (background thread + event loop)
        self._bridge = _AsyncBridge()

        # Build Graphiti clients from config
        llm_client = _create_llm_client(
            provider=graph_config.graphiti_llm_provider,
            model=graph_config.graphiti_llm_model,
            api_key=graph_config.graphiti_llm_api_key,
            fallback_model=getattr(graph_config, "graphiti_llm_fallback_model", None),
        )

        embedder = _create_embedder(
            provider=graph_config.graphiti_embedder_provider,
            model=graph_config.graphiti_embedder_model,
            api_key=graph_config.graphiti_embedder_api_key,
        )

        cross_encoder = _create_cross_encoder(
            provider=graph_config.graphiti_reranker_provider,
            api_key=graph_config.graphiti_llm_api_key,
        )

        # Create Neo4j driver and Graphiti instance ON the bridge loop
        # so the async neo4j driver is bound to the correct event loop
        def _init_graphiti():
            driver = Neo4jDriver(
                uri=graph_config.url,
                user=graph_config.username,
                password=graph_config.password,
                database=graph_config.database,
            )

            return Graphiti(
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
                graph_driver=driver,
                store_raw_episode_content=graph_config.store_raw_episode_content,
            )

        # Run the sync init parts on the bridge loop so the neo4j driver
        # attaches to that loop
        async def _async_init():
            return _init_graphiti()

        self.graphiti = self._bridge.run(_async_init())

        self._update_communities = graph_config.update_communities
        self._indices_built = False

    def _ensure_indices(self):
        """Build Neo4j indices/constraints on first use."""
        if not self._indices_built:
            try:
                self._bridge.run(self.graphiti.build_indices_and_constraints())
                self._indices_built = True
            except Exception as e:
                logger.warning(f"Failed to build Graphiti indices (may already exist): {e}")
                self._indices_built = True

    def _get_group_id(self, filters: dict) -> str:
        """Build composite group_id from mem0 filters.

        Supports namespace scoping:
        - If filters contain an explicit 'group_id', use it directly.
        - If filters contain 'project_id', returns 'project--{project_id}'.
        - Otherwise returns 'global' (overarching/cross-project memories).

        Falls back to user_id for backward compatibility with old callers
        that don't use the new scoping model.
        """
        # Explicit group_id takes precedence (used by MemoryService)
        if "group_id" in filters:
            return filters["group_id"]

        # New scoping model — must match memory_service._build_group_id format
        project_id = filters.get("project_id")
        if project_id:
            return f"project--{project_id}"

        # Default to global for new callers, user_id for legacy callers
        scope = filters.get("scope")
        if scope is not None:
            return "global"

        # Legacy: use user_id as group_id for backward compatibility
        return filters.get("user_id", "default")

    def _get_group_ids(self, filters: dict) -> list[str]:
        """Get list of group_ids for multi-scope search.

        When project_id is provided, searches both global and project scope.
        Otherwise returns single group_id.
        """
        # Explicit group_ids takes precedence
        if "group_ids" in filters:
            return filters["group_ids"]

        project_id = filters.get("project_id")
        if project_id:
            return ["global", f"project--{project_id}"]

        group_id = self._get_group_id(filters)
        return [group_id]

    def _build_source_description(self, filters: dict) -> str:
        """Build a source description string from mem0 filters for episode metadata."""
        parts = []
        if filters.get("user_id"):
            parts.append(f"user: {filters['user_id']}")
        if filters.get("agent_id"):
            parts.append(f"agent: {filters['agent_id']}")
        if filters.get("run_id"):
            parts.append(f"run: {filters['run_id']}")
        return ", ".join(parts) if parts else "mem0"

    async def _resolve_edge_names(self, edges: list[EntityEdge]) -> list[dict]:
        """Convert EntityEdge objects to triple dicts with resolved node names."""
        node_uuids = set()
        for edge in edges:
            node_uuids.add(edge.source_node_uuid)
            node_uuids.add(edge.target_node_uuid)

        uuid_to_name = {}
        if node_uuids:
            try:
                nodes = await EntityNode.get_by_uuids(
                    self.graphiti.driver, list(node_uuids)
                )
                for node in nodes:
                    uuid_to_name[node.uuid] = node.name
            except Exception as e:
                logger.warning(f"Failed to resolve node names: {e}")

        results = []
        for edge in edges:
            results.append({
                "source": uuid_to_name.get(edge.source_node_uuid, edge.source_node_uuid),
                "relationship": edge.name,
                "destination": uuid_to_name.get(edge.target_node_uuid, edge.target_node_uuid),
                "fact": edge.fact,
            })
        return results

    def add(self, data, filters):
        """Add data to the graph via Graphiti's add_episode.

        Args:
            data (str): The data to add to the graph.
            filters (dict): Filters with user_id, agent_id, run_id.

        Returns:
            dict: {"deleted_entities": [...], "added_entities": [...]}
        """
        self._ensure_indices()
        group_id = self._get_group_id(filters)
        source_description = self._build_source_description(filters)
        now = datetime.now(timezone.utc)

        async def _add():
            result = await self.graphiti.add_episode(
                name=f"mem0_episode_{now.isoformat()}",
                episode_body=data,
                source_description=source_description,
                reference_time=now,
                source=EpisodeType.text,
                group_id=group_id,
                update_communities=self._update_communities,
            )

            added = []
            for edge in result.edges:
                source_name = edge.source_node_uuid
                target_name = edge.target_node_uuid

                for node in result.nodes:
                    if node.uuid == edge.source_node_uuid:
                        source_name = node.name
                    if node.uuid == edge.target_node_uuid:
                        target_name = node.name

                added.append({
                    "source": source_name,
                    "relationship": edge.name,
                    "destination": target_name,
                })

            return {"deleted_entities": [], "added_entities": added}

        try:
            return self._bridge.run(_add())
        except Exception as e:
            logger.error(f"Graphiti add_episode failed: {e}")
            raise

    def search(self, query, filters, limit=100):
        """Search the graph via Graphiti's hybrid search.

        Args:
            query (str): Query to search for.
            filters (dict): Filters with user_id, agent_id, run_id, project_id.
            limit (int): Maximum results.

        Returns:
            list[dict]: List of {"source", "relationship", "destination"} triples.
        """
        self._ensure_indices()
        group_ids = self._get_group_ids(filters)

        async def _search():
            edges = await self.graphiti.search(
                query=query,
                group_ids=group_ids,
                num_results=limit,
            )
            return await self._resolve_edge_names(edges)

        try:
            results = self._bridge.run(_search())
            logger.info(f"Graphiti search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Graphiti search failed: {e}")
            return []

    def delete_all(self, filters):
        """Delete all graph data for a user/group via Graphiti.

        Args:
            filters (dict): Filters with user_id (used as group_id).
        """
        group_id = self._get_group_id(filters)

        async def _delete():
            from graphiti_core.nodes import Node

            await Node.delete_by_group_id(self.graphiti.driver, group_id)

        try:
            self._bridge.run(_delete())
            logger.info(f"Deleted all Graphiti data for group_id={group_id}")
        except Exception as e:
            logger.error(f"Graphiti delete_all failed: {e}")
            raise

    def get_all(self, filters, limit=100):
        """Retrieve all edges/facts from the graph for a group.

        Args:
            filters (dict): Filters with user_id, project_id (used for group_ids).
            limit (int): Maximum results.

        Returns:
            list[dict]: List of {"source", "relationship", "target"} triples.
        """
        self._ensure_indices()
        group_ids = self._get_group_ids(filters)

        async def _get_all():
            try:
                edges = await EntityEdge.get_by_group_ids(
                    self.graphiti.driver,
                    group_ids=group_ids,
                    limit=limit,
                )
            except GroupsEdgesNotFoundError:
                return []

            resolved = await self._resolve_edge_names(edges)
            return [
                {
                    "source": r["source"],
                    "relationship": r["relationship"],
                    "target": r["destination"],
                }
                for r in resolved
            ]

        try:
            results = self._bridge.run(_get_all())
            logger.info(f"Graphiti get_all returned {len(results)} relationships")
            return results
        except Exception as e:
            logger.error(f"Graphiti get_all failed: {e}")
            return []
