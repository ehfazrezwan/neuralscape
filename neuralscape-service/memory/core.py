"""Core lazy-init plumbing for MemoryService: mem0/Graphiti/genai clients and Qdrant indexes.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import threading

from google import genai
from config import settings

logger = logging.getLogger(__name__)

class CoreMixin:
    """CoreMixin for MemoryService (mechanical split — see memory_service.py)."""

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

        mem0 v2.0.2 (upstream c9e8482a, merged in #38) removed auto-init
        of ``Memory.graph`` from ``mem0/mem0/memory/main.py``. The
        ``MemoryGraph`` adapter for Graphiti is still present in
        ``mem0.memory.graphiti_memory`` and still selectable via
        ``GraphStoreFactory``, but nothing wires it onto the ``Memory``
        instance anymore. We re-attach it manually below so downstream
        code (``store_raw``, ``extract_and_store``, the wiki synthesizer,
        etc.) keeps using ``self._memory.graph.add(...)`` unchanged.
        Without this re-attach every graph write is a silent no-op and
        the health check reports ``graph_store: not_initialized``.
        """
        if self._memory is None:
            with self._init_lock:
                if self._memory is None:
                    from mem0 import Memory

                    config = settings.get_mem0_config()
                    self._memory = Memory.from_config(config)

                    # Re-attach the Graphiti adapter mem0 v2.0.2 stopped
                    # auto-creating. The MemoryGraph constructor in
                    # ``mem0/mem0/memory/graphiti_memory.py`` reads
                    # ``config.graph_store.config.<field>`` via attribute
                    # access, but mem0 v2.0.2's ``MemoryConfig`` removed
                    # the ``graph_store`` attribute alongside the graph
                    # auto-init. We build a SimpleNamespace shim from
                    # our own dict so MemoryGraph still finds what it
                    # needs without us forking the mem0 subtree.
                    # Best-effort: a failure here logs and leaves the
                    # service running with vector-only memory.
                    graph_store_cfg = (
                        config.get("graph_store") if isinstance(config, dict) else None
                    )
                    if graph_store_cfg:
                        try:
                            from types import SimpleNamespace
                            from mem0.utils.factory import GraphStoreFactory

                            provider = graph_store_cfg.get("provider", "default")
                            inner = graph_store_cfg.get("config", {})
                            shim = SimpleNamespace(
                                graph_store=SimpleNamespace(
                                    provider=provider,
                                    config=SimpleNamespace(**inner),
                                )
                            )
                            self._memory.graph = GraphStoreFactory.create(provider, shim)
                            logger.info(
                                "Graphiti adapter attached manually "
                                "(provider=%s) — mem0 v2.0.2 no longer "
                                "auto-creates Memory.graph",
                                provider,
                            )
                            if inner.get("graph_provider") == "kuzu":
                                # Kuzu's schema is static: NS back-reference
                                # columns (memory_id, wiki_path, dream_*, ns_*)
                                # and the Source/DERIVED_FROM provenance tables
                                # must be declared before any patcher SETs
                                # them. Idempotent; neo4j is schema-free and
                                # never takes this branch.
                                from .kuzu_schema import apply_ns_kuzu_schema

                                g = self._memory.graph
                                g._bridge.run(
                                    apply_ns_kuzu_schema(g.graphiti.driver)
                                )
                        except Exception as e:
                            logger.warning(
                                f"Graphiti adapter init failed (non-critical): {e}"
                            )

                    if hasattr(self._memory, "graph") and hasattr(self._memory.graph, "graphiti"):
                        self._graphiti = self._memory.graph.graphiti
                        self._bridge = self._memory.graph._bridge

        return self._memory

    def _get_graphiti(self):
        """Get the underlying Graphiti instance."""
        self._get_memory()
        return self._graphiti

    async def _run_on_bridge_async(self, coro, timeout: float = 30.0):
        """Async wrapper around :meth:`_run_on_bridge`.

        Schedules ``coro`` on the Graphiti bridge's event loop without
        blocking the caller's loop. Use this from any ``async def`` that
        needs to make Graphiti / Neo4j calls — the synthesizer admin
        endpoint, the worker cron, and any future async caller. Without
        the ``asyncio.to_thread`` wrap, calling :meth:`_run_on_bridge`
        from an async function would block its event loop for the whole
        synthesis pass.
        """
        import asyncio as _asyncio

        return await _asyncio.to_thread(self._run_on_bridge, coro, timeout)

    def _run_on_bridge(self, coro, timeout: float = 30.0):
        """Run an async coroutine on the Graphiti adapter's event loop.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait for the result (default 30s).

        Raises:
            RuntimeError: If the bridge is not initialized, or its loop is
                not a real event loop (half-initialized adapter / mocked
                bridge) — failing fast instead of parking on
                ``future.result()`` for the full timeout.
            TimeoutError: If the operation exceeds the timeout.
        """
        if self._bridge is None:
            raise RuntimeError("Graphiti bridge not initialized")
        import asyncio as _asyncio
        import concurrent.futures

        # If the bridge's loop isn't a real event loop (e.g. a mocked bridge
        # in unit tests, or a half-initialized adapter), run_coroutine_threadsafe
        # would return a future that never completes and we'd park on
        # future.result() until the timeout. Fail fast instead — callers
        # already treat bridge failures as non-critical/fail-open. (Same
        # guard as the episode-idempotency probe, hoisted to the choke point.)
        loop = getattr(self._bridge, "_loop", None)
        if not isinstance(loop, _asyncio.AbstractEventLoop):
            raise RuntimeError(
                "Graphiti bridge loop is not an asyncio event loop "
                f"(got {type(loop).__name__}) — bridge mocked or half-initialized"
            )
        future = _asyncio.run_coroutine_threadsafe(coro, loop)
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

    def _ensure_created_at_index(self) -> None:
        """Best-effort: ensure a DATETIME payload index on ``created_at``.

        Qdrant's ``order_by`` scroll requires a range-capable payload index.
        Creation is idempotent server-side; attempted once per process. On
        failure the timeline falls back to a Python-side sort, so this never
        raises.
        """
        if getattr(self, "_created_at_index_ok", False):
            return
        try:
            from qdrant_client.models import PayloadSchemaType

            self._get_memory().vector_store.client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name="created_at",
                field_schema=PayloadSchemaType.DATETIME,
                wait=True,
            )
        except Exception as e:
            logger.debug(f"created_at payload index ensure failed (non-fatal): {e}")
        # Attempt once per process either way — the order_by scroll has its
        # own fallback if the index genuinely isn't there.
        self._created_at_index_ok = True

    # Hot-path payload filters that deserve keyword/bool indexes (audit 27
    # #14): every pool query carries a dream_tombstoned must_not, the shared/
    # standard pools filter on visibility, and dual-scope merges filter on
    # scope — all unindexed meant full payload scans on every search.
    _FILTER_INDEX_FIELDS = (
        ("metadata.dream_tombstoned", "bool"),
        ("metadata.visibility", "keyword"),
        ("metadata.scope", "keyword"),
    )

    def _ensure_filter_indexes(self) -> None:
        """Best-effort: ensure payload indexes on the hot-path filter fields.

        Same pattern as ``_ensure_created_at_index``: idempotent server-side,
        attempted once per service instance, ``wait=False`` so index builds
        never block the first read, and any failure degrades to Qdrant's
        unindexed filtering rather than raising.
        """
        if getattr(self, "_filter_indexes_ok", False):
            return
        # Mark attempted FIRST — a flaky Qdrant must not re-pay this on
        # every subsequent search.
        self._filter_indexes_ok = True
        try:
            from qdrant_client.models import PayloadSchemaType

            schema_by_name = {
                "bool": PayloadSchemaType.BOOL,
                "keyword": PayloadSchemaType.KEYWORD,
            }
            client = self._get_memory().vector_store.client
            for field_name, schema_name in self._FILTER_INDEX_FIELDS:
                try:
                    client.create_payload_index(
                        collection_name=settings.qdrant_collection,
                        field_name=field_name,
                        field_schema=schema_by_name[schema_name],
                        wait=False,
                    )
                except Exception as e:
                    logger.debug(
                        f"payload index ensure failed for {field_name} (non-fatal): {e}"
                    )
        except Exception as e:
            logger.debug(f"filter payload index ensure failed (non-fatal): {e}")
