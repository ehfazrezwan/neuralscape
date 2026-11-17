import asyncio
import concurrent.futures
import logging
import os
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
    small_model: str | None = None,
):
    """Create a Graphiti LLM client based on provider name."""
    if provider == "gemini":
        from graphiti_core.llm_client.gemini_client import GeminiClient

        # NEURALSCAPE PATCH: default small_model to the main model. Graphiti's
        # GeminiClient uses small_model for cheaper sub-tasks (e.g.
        # dedupe_edges.resolve_edge); if left unset it uses its own hardcoded
        # DEFAULT_SMALL_MODEL. Threading the configured main model here keeps the
        # ENTIRE enrichment path (extraction + dedup) on one reliable model
        # instead of silently splitting onto a different default.
        config = LLMConfig(
            api_key=api_key,
            model=model,
            fallback_model=fallback_model,
            small_model=small_model or model,
        )
        return GeminiClient(config=config)
    elif provider == "openai":
        from graphiti_core.llm_client.openai_client import OpenAIClient

        # NEURALSCAPE PATCH: default small_model to the main model. Graphiti's
        # OpenAIBaseClient uses small_model for cheaper sub-tasks and otherwise
        # falls back to DEFAULT_SMALL_MODEL ("gpt-4.1-nano") — an OpenAI model an
        # OpenAI-compatible gateway that only fronts Google/Anthropic-on-Vertex
        # doesn't provision. base_url is left to OPENAI_BASE_URL.
        config = LLMConfig(
            api_key=api_key,
            model=model,
            fallback_model=fallback_model,
            small_model=small_model or model,
        )
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


def _create_embedder(
    provider: str,
    model: str | None,
    api_key: str | None,
    embedding_dim: int | None = None,
):
    """Create a Graphiti embedder client based on provider name.

    NEURALSCAPE PATCH: ``embedding_dim`` is passed explicitly. Without it the
    config inherits Graphiti's ambient ``EMBEDDING_DIM`` env default (1024),
    which nothing in the deployment sets, so the stored graph vector size was
    an accident of the environment. ``None`` keeps that legacy behaviour.
    """
    dim_kwargs = {"embedding_dim": int(embedding_dim)} if embedding_dim else {}
    if provider == "gemini":
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        config = GeminiEmbedderConfig(api_key=api_key, **dim_kwargs)
        if model:
            config.embedding_model = model
        return GeminiEmbedder(config=config)
    elif provider == "openai":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        config = OpenAIEmbedderConfig(api_key=api_key, **dim_kwargs)
        if model:
            config.embedding_model = model
        return OpenAIEmbedder(config=config)
    elif provider == "voyage":
        from graphiti_core.embedder.voyage import VoyageEmbedder, VoyageEmbedderConfig

        config = VoyageEmbedderConfig(api_key=api_key, **dim_kwargs)
        if model:
            config.embedding_model = model
        return VoyageEmbedder(config=config)
    else:
        raise ValueError(f"Unsupported Graphiti embedder provider: {provider}")


def _create_cross_encoder(provider: str | None, api_key: str | None, model: str | None = None):
    """Create a Graphiti cross-encoder/reranker client."""
    if provider is None or provider == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        # NEURALSCAPE PATCH: OpenAIRerankerClient takes a `config` (LLMConfig),
        # not `api_key=` — passing api_key raises TypeError and fails graphiti
        # init. Build a config (model + key; base_url via OPENAI_BASE_URL) so the
        # reranker can route through an OpenAI-compatible gateway.
        config = LLMConfig(api_key=api_key, model=model)
        return OpenAIRerankerClient(config=config)
    elif provider == "gemini":
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

        config = LLMConfig(api_key=api_key)
        return GeminiRerankerClient(config=config)
    elif provider == "bge":
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()
    else:
        raise ValueError(f"Unsupported Graphiti reranker provider: {provider}")


# NEURALSCAPE PATCH: upper bound on one bridge call. Graph writes legitimately
# take minutes (LLM extraction), so this is a hang guard, not a latency target.
# It must never undercut the longest worker job that runs graph work: the graph
# and ingest workers use ``job_timeout = 900`` (neuralscape-service/worker.py),
# so the default is 900 s plus a 60 s margin. Before this patch the wait was
# unbounded. Override per process with GRAPHITI_BRIDGE_TIMEOUT_S, or per call
# with ``_AsyncBridge.run(timeout=...)``.
DEFAULT_BRIDGE_TIMEOUT_S = 960.0


def _bridge_timeout_from_env() -> float:
    raw = os.environ.get("GRAPHITI_BRIDGE_TIMEOUT_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_BRIDGE_TIMEOUT_S
    except ValueError:
        return DEFAULT_BRIDGE_TIMEOUT_S
    return value if value > 0 else DEFAULT_BRIDGE_TIMEOUT_S


class _AsyncBridge:
    """Runs a dedicated event loop in a background thread.

    All Graphiti async operations (including the Neo4j driver) are executed
    on this single loop, avoiding the 'Future attached to a different loop'
    error that occurs when mixing event loops.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._stopped = False
        # Futures handed out by run() that have not finished. shutdown() cancels
        # whatever is left in here, so a caller blocked on a loop that was stopped,
        # wedged or missed by the task snapshot is released instead of waiting out
        # the bridge limit. ``_state_lock`` orders "stopped" against "submitted".
        self._state_lock = threading.Lock()
        self._inflight: set = set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float | None = None):
        """Submit a coroutine to the background loop and wait for the result.

        NEURALSCAPE PATCH: the wait is bounded. Upstream waited forever, so a
        stalled Neo4j or LLM call parked the calling worker slot for good.
        ``timeout`` (seconds) wins; otherwise ``GRAPHITI_BRIDGE_TIMEOUT_S``;
        otherwise ``DEFAULT_BRIDGE_TIMEOUT_S``. On expiry the coroutine is
        cancelled on the loop and ``TimeoutError`` is raised, which the
        service's graph callers already log and treat as non-critical.
        """
        limit = timeout if timeout is not None and timeout > 0 else _bridge_timeout_from_env()
        with self._state_lock:
            # Checked and submitted under the lock shutdown() takes to set the flag:
            # a call either fails here, or is tracked before shutdown's final sweep.
            if self._stopped or self._loop.is_closed():
                stopped = True
            else:
                stopped = False
                try:
                    future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                except RuntimeError:
                    stopped = True
                else:
                    self._inflight.add(future)
        if stopped:
            if asyncio.iscoroutine(coro):
                coro.close()  # never leave an un-awaited coroutine behind
            raise RuntimeError("Graphiti bridge is shut down")
        future.add_done_callback(self._inflight.discard)
        try:
            return future.result(timeout=limit)
        except concurrent.futures.CancelledError:
            if self._stopped:
                raise RuntimeError(
                    "Graphiti bridge was shut down while this call was waiting"
                ) from None
            raise
        except concurrent.futures.TimeoutError:
            if future.done():
                # The coroutine itself raised TimeoutError (driver/LLM timeout,
                # or it finished in the same instant): propagate it unchanged
                # so its message and the service's transient retry still apply.
                return future.result()
            future.cancel()
            # No "timed out"/"timeout" wording: the service's transient-error
            # retry matches on those and would re-run a call that just hung.
            raise TimeoutError(
                f"Graphiti bridge call did not finish within {limit:g}s and was cancelled"
            ) from None

    def _release_waiters(self) -> None:
        """Cancel every future run() handed out that is still pending, so its waiter
        returns now (concurrent futures need no loop to cancel)."""
        with self._state_lock:
            leftover = list(self._inflight)
        for future in leftover:
            future.cancel()

    def shutdown(self, timeout: float = 10.0) -> None:
        """NEURALSCAPE PATCH: stop the bridge loop cleanly (not in upstream).

        Cancel outstanding tasks, shut down async generators, stop the loop and
        join its thread. Fire-and-forget work scheduled on the loop (the driver
        schedules its index build there at construction) must not outlive the
        loop: left running it surfaces later as 'Event loop is closed' / 'no
        running event loop' noise. Idempotent; a later ``run()`` raises
        RuntimeError.
        """
        with self._state_lock:
            self._stopped = True
        try:
            self._shutdown_loop(timeout)
        finally:
            # On every path (late submit missed by the task snapshot, a loop that never
            # answered, a stopped loop): nobody stays blocked in run().
            self._release_waiters()

    def _shutdown_loop(self, timeout: float) -> None:
        if self._loop.is_closed():
            return

        async def _cancel_all():
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self._loop.shutdown_asyncgens()

        if not self._loop.is_running():
            # The loop was already stopped (nothing can answer a cancel request):
            # skip the wait for it, just release the loop.
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                # The thread is gone, so nothing else drives the loop: run the
                # cancellation here, bounded, so tasks still pending on it are
                # cancelled (and never-started coroutines closed) before it closes.
                try:
                    self._loop.run_until_complete(
                        asyncio.wait_for(_cancel_all(), timeout=timeout)
                    )
                except Exception:
                    logger.warning(
                        "Graphiti bridge shutdown: cancelling tasks on a stopped loop failed",
                        exc_info=True,
                    )
                self._loop.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(_cancel_all(), self._loop).result(timeout=timeout)
        except Exception:
            logger.warning("Graphiti bridge shutdown: cancelling tasks failed", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._loop.close()
        else:
            logger.warning(
                "Graphiti bridge thread did not stop within %.0fs; abandoning it", timeout
            )


def _hybrid_egress_ok(text: str) -> bool:
    """NEURALSCAPE PATCH: text matching a sensitivity class never goes to the
    optional hybrid decision endpoint (the service's regex floor, no LLM)."""
    try:
        from memory.sensitivity import classify_sensitivity
    except ImportError:  # library used outside the service: no gate available
        return True
    return classify_sensitivity(text) is None


def _build_hybrid_decisions(graph_config):
    """NEURALSCAPE PATCH: the optional decision client, always built WITH the egress gate."""
    from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions

    return HybridDecisions(
        graph_config.graphiti_hybrid_api_key,
        endpoint=graph_config.graphiti_hybrid_endpoint,
        model=graph_config.graphiti_hybrid_model,
        threshold=graph_config.graphiti_hybrid_threshold,
        timeout=graph_config.graphiti_hybrid_timeout,
        egress_check=_hybrid_egress_ok,
    )


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
            small_model=getattr(graph_config, "graphiti_llm_small_model", None),
        )

        embedder = _create_embedder(
            provider=graph_config.graphiti_embedder_provider,
            model=graph_config.graphiti_embedder_model,
            api_key=graph_config.graphiti_embedder_api_key,
            embedding_dim=getattr(graph_config, "graphiti_embedding_dim", None),
        )

        # NEURALSCAPE PATCH: attach the optional decision client only when the
        # graph flag is on; flag off constructs nothing.
        if getattr(graph_config, "graphiti_hybrid_enabled", False):
            llm_client.decision_client = _build_hybrid_decisions(graph_config)

        cross_encoder = _create_cross_encoder(
            provider=graph_config.graphiti_reranker_provider,
            api_key=graph_config.graphiti_llm_api_key,
            model=graph_config.graphiti_llm_model,
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
        """Return the explicit ``group_id`` the caller names.

        NEURALSCAPE PATCH: the upstream-style fallbacks are gone. This used to
        derive ``project:{project_id}``, ``global`` or the bare ``user_id`` when
        no ``group_id`` was passed, and the service never reads those names: its
        groups come from ``memory.groups._build_group_id`` (pool + owner +
        project + workspace). A caller that omitted ``group_id`` would have
        written, searched or DELETED graph data in a namespace nothing else
        sees. No service caller does (every call site passes ``group_id`` or
        ``group_ids``), so the fallback is replaced by a hard error.
        """
        group_id = filters.get("group_id")
        if not group_id:
            raise ValueError(
                "MemoryGraph requires an explicit 'group_id' in filters (build it with "
                "memory.groups._build_group_id); the legacy project:/user_id fallback was removed"
            )
        return group_id

    def _get_group_ids(self, filters: dict) -> list[str]:
        """Return the explicit ``group_ids`` (or the single ``group_id``) the caller names.

        NEURALSCAPE PATCH: see ``_get_group_id``; no implicit ``["global", "project:{id}"]``.
        """
        group_ids = filters.get("group_ids")
        if group_ids:
            return list(group_ids)
        return [self._get_group_id(filters)]

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
                # R4: surface bi-temporal validity metadata so the MCP
                # search_knowledge_graph path carries it too (additive).
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
            })
        return results

    def add(
        self,
        data,
        filters,
        entity_types=None,
        edge_types=None,
        edge_type_map=None,
        excluded_entity_types=None,
        custom_extraction_instructions=None,
        episode_name=None,
        reference_time: datetime | None = None,
        episode_source: str = "text",
    ):
        """Add data to the graph via Graphiti's add_episode.

        Args:
            data (str): The data to add to the graph.
            filters (dict): Filters with user_id, agent_id, run_id.
            entity_types (dict[str, type[BaseModel]] | None): Optional custom
                Graphiti entity types (a knowledge adapter's ontology). None ⇒
                Graphiti's built-in generic extraction (current behavior).
            edge_types (dict[str, type[BaseModel]] | None): Optional custom edge
                types.
            edge_type_map (dict[tuple[str, str], list[str]] | None): Optional
                map of (source_type, target_type) → allowed edge type names.
            excluded_entity_types (list[str] | None): Optional entity type names
                to drop from the graph.
            custom_extraction_instructions (str | None): Optional extra guidance
                injected into Graphiti's extraction prompts.
            episode_name (str | None): NEURALSCAPE PATCH (audit 27 #21) —
                optional caller-supplied episode name, used as an idempotency
                carrier: the service derives it deterministically from the
                content + group so a re-run can find (and skip) an
                already-ingested episode. None ⇒ the legacy timestamp-based
                name (every call mints a distinct episode).
            reference_time (datetime | None): Real event time for Graphiti
                bi-temporal dating; None ⇒ ingestion wall-clock (legacy behavior).
            episode_source (str): Episode source type for Graphiti extraction.
                Conversation episodes use "message" to activate Graphiti's
                speaker-first extract_message prompt; single facts stay "text".
                Default "text" preserves existing behavior for all current callers.

        Returns:
            dict: {"deleted_entities": [...], "added_entities": [...]}
        """
        self._ensure_indices()
        group_id = self._get_group_id(filters)
        source_description = self._build_source_description(filters)
        now = datetime.now(timezone.utc)

        # Only forward custom-ontology kwargs when supplied so the default path
        # is byte-for-byte the pre-adapter add_episode call.
        episode_kwargs = {}
        if entity_types is not None:
            episode_kwargs["entity_types"] = entity_types
        if edge_types is not None:
            episode_kwargs["edge_types"] = edge_types
        if edge_type_map is not None:
            episode_kwargs["edge_type_map"] = edge_type_map
        if excluded_entity_types is not None:
            episode_kwargs["excluded_entity_types"] = excluded_entity_types
        if custom_extraction_instructions is not None:
            episode_kwargs["custom_extraction_instructions"] = custom_extraction_instructions

        async def _add():
            # Translate episode_source string to EpisodeType enum
            source_type = (
                EpisodeType.message if episode_source == "message"
                else EpisodeType.text
            )
            result = await self.graphiti.add_episode(
                name=episode_name or f"mem0_episode_{now.isoformat()}",
                episode_body=data,
                source_description=source_description,
                reference_time=reference_time or now,
                source=source_type,
                group_id=group_id,
                update_communities=self._update_communities,
                **episode_kwargs,
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

            if self._update_communities:
                await self._ensure_group_communities(group_id)

            # NEURALSCAPE PATCH: surface the resolved episode identity so
            # callers can persist a durable memory→episode back-reference
            # (see neuralscape-service/memory/provenance.py). Additive —
            # existing callers that only read deleted_entities/added_entities
            # are unaffected.
            return {
                "deleted_entities": [],
                "added_entities": added,
                "episode_uuid": result.episode.uuid,
                "episode_name": result.episode.name,
            }

        try:
            return self._bridge.run(_add())
        except Exception as e:
            logger.error(f"Graphiti add_episode failed: {e}")
            raise

    async def _ensure_group_communities(self, group_id: str) -> None:
        """NEURALSCAPE PATCH: build the first set of communities for a group.

        ``add_episode(update_communities=True)`` only attaches new entities to
        communities that already exist (``update_community`` returns nothing
        when there is none to join), and nothing else ever creates the first
        one, so with the flag on ``GET /v1/graph/communities`` stayed empty
        forever. When the flag is on and this group has no *populated*
        community yet, build them with Graphiti's clustering, scoped to THIS
        group only (``Graphiti.build_communities`` calls ``remove_communities``,
        which deletes every community of every group, so it is deliberately not
        used). Later episodes grow them through ``update_community``.

        Safe across processes: community and membership uuids are derived from
        the group and the cluster (smallest member uuid), and Graphiti saves
        with ``MERGE`` on uuid, so two workers that build at the same moment
        write the same nodes and edges instead of duplicates. A community
        without any ``HAS_MEMBER`` edge (a build that died between the node and
        edge saves) counts as "not built": it is removed and rebuilt by the
        next write. The in-process lock is per group.

        Best-effort: the episode is already stored, so a failure here is logged
        and never fails the write.
        """
        import uuid as _uuid

        from graphiti_core.helpers import semaphore_gather
        from graphiti_core.utils.maintenance.community_operations import build_communities

        locks = self.__dict__.setdefault("_communities_locks", {})
        lock = locks.get(group_id)
        if lock is None:
            lock = locks.setdefault(group_id, asyncio.Lock())
        async with lock:
            try:
                g = self.graphiti
                records, _, _ = await g.driver.execute_query(
                    "MATCH (c:Community {group_id: $g}) "
                    "RETURN c.uuid AS uuid, EXISTS { (c)-[:HAS_MEMBER]->() } AS populated",
                    g=group_id,
                )
                if any(r["populated"] for r in records):
                    return
                if records:
                    await g.driver.execute_query(
                        "MATCH (c:Community {group_id: $g}) WHERE NOT EXISTS { (c)-[:HAS_MEMBER]->() } "
                        "DETACH DELETE c",
                        g=group_id,
                    )
                nodes, edges = await build_communities(g.driver, g.llm_client, [group_id])
                if not nodes:
                    return
                for node in nodes:
                    members = sorted(e.target_node_uuid for e in edges if e.source_node_uuid == node.uuid)
                    old = node.uuid
                    node.uuid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"ns-community:{group_id}:{members[0] if members else old}"))
                    for e in edges:
                        if e.source_node_uuid == old:
                            e.source_node_uuid = node.uuid
                            e.uuid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"ns-community-member:{node.uuid}:{e.target_node_uuid}"))
                await semaphore_gather(
                    *[n.generate_name_embedding(g.embedder) for n in nodes],
                    max_coroutines=g.max_coroutines,
                )
                await semaphore_gather(*[n.save(g.driver) for n in nodes], max_coroutines=g.max_coroutines)
                await semaphore_gather(*[e.save(g.driver) for e in edges], max_coroutines=g.max_coroutines)
                logger.info(f"Built {len(nodes)} initial communities for group_id={group_id}")
            except Exception as e:
                logger.warning(f"Initial community build failed for group_id={group_id} (non-fatal): {e}")

    def search(self, query, filters, limit=100):
        """Search the graph via Graphiti's hybrid search.

        Args:
            query (str): Query to search for.
            filters (dict): Filters with user_id, agent_id, run_id, project_id.
            limit (int): Maximum results.

        Returns:
            list[dict]: List of {"source", "relationship", "destination"} triples.

        Raises:
            Exception: whatever the Graphiti bridge raised (never disguised as
            an empty result).
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
            # NEURALSCAPE PATCH: a broken bridge must not look like an empty
            # graph. Log with the traceback and let the caller decide whether
            # to degrade.
            logger.error(f"Graphiti search failed: {e}", exc_info=True)
            raise

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

        Raises:
            Exception: whatever the Graphiti bridge raised (see search()).
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
            # NEURALSCAPE PATCH: see search() — surface the failure.
            logger.error(f"Graphiti get_all failed: {e}", exc_info=True)
            raise
