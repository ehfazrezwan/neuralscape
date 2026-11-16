"""Core lazy-init plumbing for MemoryService: mem0/Graphiti/genai clients and Qdrant indexes.

``CoreMixin`` was mechanically extracted from memory_service.py (mixins-with-facade
split) with its behavior unchanged. The Qdrant cold-start helpers above it
(``_memory_from_config`` and the 409/500 detectors, PR #453) were written
afterwards and are not part of that extraction.
"""

import logging
import threading
import time

from google import genai
from config import settings

logger = logging.getLogger(__name__)

# One mem0 start touches two Qdrant collections (memories and migrations).
# Each can cost two failed attempts, a 409 from our create and then a 500 from
# reading it while another process finishes creating it, so the worst
# expected start uses four. A collection being created was readable within
# 0.1 to 0.16 s in live boots (0.155 s was the slowest observed), against 2.8 s
# of total backoff here (sleeps of 0.1 * attempt after attempts 1 to 7). A fault
# that persists still fails start-up after the last attempt.
_COLD_START_ATTEMPTS = 8


def _is_collection_exists_conflict(exc: BaseException) -> bool:
    """Qdrant's 409 ``Collection `x` already exists!``: another process created it first.

    Only that conflict counts. A 409 with any other message (or none) is a
    different failure and must not be retried as a concurrent creation.
    """
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:  # pragma: no cover — qdrant_client ships with mem0
        return False
    if not isinstance(exc, UnexpectedResponse) or exc.status_code != 409:
        return False
    return b"already exists" in (exc.content or b"")


def _is_collection_not_yet_readable(exc: BaseException) -> bool:
    """Qdrant's 500 ``0 of 0 read operations failed``: a collection another
    process is still creating.

    Qdrant lists a new collection before its shard can be read, and for a
    moment after the create call returns. Reading it then fails with this
    500. Only this exact body counts: any other 500, or ``N of N`` with
    N > 0 (shards exist but failed), is a real failure.
    """
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:  # pragma: no cover — qdrant_client ships with mem0
        return False
    if not isinstance(exc, UnexpectedResponse) or exc.status_code != 500:
        return False
    return b"0 of 0 read operations failed" in (exc.content or b"")


def _is_concurrent_creation(exc: BaseException) -> bool:
    return _is_collection_exists_conflict(exc) or _is_collection_not_yet_readable(exc)


def _memory_from_config(memory_cls, config):
    """``Memory.from_config``, tolerating a concurrent first start.

    mem0's Qdrant store lists the collections and creates its own when it is
    missing, or reads it when it is listed. When the API and the workers start
    together against an empty Qdrant, a process that loses the race either
    gets ``409 already exists`` from its create, or sees the collection listed
    while another process is still creating it and gets ``500 0 of 0 read
    operations failed`` when reading it. Either way the collection is there or
    about to be, so building ``Memory`` again a moment later goes on. Any other
    error is raised at once.
    """
    for attempt in range(1, _COLD_START_ATTEMPTS + 1):
        try:
            return memory_cls.from_config(config)
        except Exception as exc:
            if attempt == _COLD_START_ATTEMPTS or not _is_concurrent_creation(exc):
                raise
            logger.info(
                "Qdrant collection is being created concurrently by another process; "
                "retrying memory initialisation (attempt %d/%d)",
                attempt + 1,
                _COLD_START_ATTEMPTS,
            )
            time.sleep(0.1 * attempt)

_BM25_MISSING_DETAIL = (
    "collection predates the BM25 sparse slot; keyword leg disabled (search is dense-only). "
    "No in-place migration exists: recreate the collection and re-ingest to enable hybrid search."
)
# One warning per process, not per search.
_bm25_warned = False
_bm25_encoder_warned = False


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
                    self._memory = _memory_from_config(Memory, config)
                    # Capture provider usage only when a recorder is installed.
                    from usage_capture import get_usage_recorder

                    capture_usage = get_usage_recorder() is not None
                    if capture_usage:
                        from usage_wrappers import install_mem0_wrappers

                        install_mem0_wrappers(self._memory)

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
                        except Exception as e:
                            logger.warning(
                                f"Graphiti adapter init failed (non-critical): {e}"
                            )

                    if hasattr(self._memory, "graph") and hasattr(self._memory.graph, "graphiti"):
                        self._graphiti = self._memory.graph.graphiti
                        self._bridge = self._memory.graph._bridge
                        if capture_usage:
                            from usage_wrappers import install_graphiti_wrappers

                            install_graphiti_wrappers(self._graphiti)

                    self._ensure_indexes_at_startup()

        # A startup ensure that failed (Qdrant not ready yet) is retried from
        # here, so a write-only worker (which never searches) still recovers.
        # Cheap no-op unless an attempt failed and the cool-down has passed.
        if not self._filter_indexes_ok or not getattr(self, "_created_at_index_ok", False):
            self._retry_index_ensure()

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
                    endpoint = getattr(settings, "ns_gemini_base_url", "")
                    if isinstance(endpoint, str) and endpoint:
                        from inference_facade import build_gemini_client

                        self._genai_model = build_gemini_client(settings, genai)
                    else:
                        # Default (and any mocked setting): unchanged.
                        self._genai_model = genai.Client(api_key=settings.google_api_key)
        return self._genai_model

    def close(self):
        """Clean up resources."""
        if self._graphiti and self._bridge:
            self._bridge.run(self._graphiti.close())

    def _ensure_created_at_index(self) -> None:
        """Best-effort: ensure a DATETIME payload index on ``created_at``.

        Qdrant's ``order_by`` scroll requires a range-capable payload index.
        Creation is idempotent server-side. On failure the flag stays unset and
        the next call past the cool-down re-checks; meanwhile the timeline falls
        back to a Python-side sort, so this never raises.
        """
        if getattr(self, "_created_at_index_ok", False):
            return
        if time.monotonic() < getattr(self, "_created_at_retry_at", 0.0):
            return
        self._created_at_retry_at = time.monotonic() + self._FILTER_INDEX_RETRY_S
        try:
            from qdrant_client.models import PayloadSchemaType

            self._get_memory().vector_store.client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name="created_at",
                field_schema=PayloadSchemaType.DATETIME,
                wait=True,
            )
        except Exception as e:
            # Not marked done: re-checked after the cool-down. Warn once.
            logger.debug(f"created_at payload index ensure failed (non-fatal): {e}")
            if not getattr(self, "_created_at_warned", False):
                self._created_at_warned = True
                logger.warning(
                    "qdrant_payload_index_ensure_failed fields=created_at: timeline falls back "
                    "to a Python-side sort (will re-check in %ds)",
                    int(self._FILTER_INDEX_RETRY_S),
                )
            return
        self._created_at_index_ok = True

    # Hot-path payload filters that deserve keyword/bool indexes (audit 27
    # #14): every pool query carries a dream_tombstoned must_not, the shared/
    # standard pools filter on visibility, and dual-scope merges filter on
    # scope — all unindexed meant full payload scans on every search.
    #
    # Every payload key the service filters on (grep ``FieldCondition(key=``
    # in memory/, okf/, knowledge/, extensions/, adapters/trading/) is listed
    # here; tests/test_qdrant_indexes_bm25_health.py scans those dirs and fails
    # on a gap. The payload layout is mem0's: ``hash`` and ``user_id`` are top-level; everything NS stamps lives under
    # ``metadata.*``. ``user_id`` is indexed by mem0's own store, and
    # ``created_at`` by ``_ensure_created_at_index`` (DATETIME), so neither is
    # repeated here. Array fields (tags, concepts) take a keyword index too.
    _FILTER_INDEX_FIELDS = (
        ("metadata.dream_tombstoned", "bool"),
        ("metadata.visibility", "keyword"),
        ("metadata.scope", "keyword"),
        ("hash", "keyword"),
        ("metadata.project_id", "keyword"),
        ("metadata.category", "keyword"),
        ("metadata.domain", "keyword"),
        ("metadata.observation_type", "keyword"),
        ("metadata.concepts", "keyword"),
        ("metadata.tags", "keyword"),
        ("metadata.memory_kind", "keyword"),
        ("metadata.owner_user_id", "keyword"),
        ("metadata.speaker", "keyword"),
        ("metadata.workspace", "keyword"),
        ("metadata.source_ref.external_id", "keyword"),
        ("metadata.code_liveness_stale", "bool"),
        # Shared-episode sibling lookup (memory/provenance.py) and
        # graph-contradiction supersession (memory/supersession.py, edge
        # episode -> row): both scroll on these; unindexed they scan the
        # collection each time.
        ("metadata.graph_episode_uuid", "keyword"),
        ("metadata.graph_episode_name", "keyword"),
    )

    # After a failed attempt, do not retry on every search against a flaky
    # Qdrant; allow a re-check after this many seconds.
    _FILTER_INDEX_RETRY_S = 60.0
    _filter_indexes_ok = False
    _created_at_index_ok = False

    def _ensure_filter_indexes(self) -> None:
        """Best-effort: ensure payload indexes on the filter fields.

        Idempotent server-side, ``wait=False`` so index builds never block a
        read, and any failure degrades to Qdrant's unindexed filtering rather
        than raising. The done flag is set only when every field succeeded;
        after a failure the next call past ``_FILTER_INDEX_RETRY_S`` re-checks,
        and the failed field names are logged at warning (once per instance).
        """
        if getattr(self, "_filter_indexes_ok", False):
            return
        if time.monotonic() < getattr(self, "_filter_index_retry_at", 0.0):
            return
        # Surface a pre-v3 collection (no BM25 slot) once per process.
        self._warn_bm25_missing_once()
        self._filter_index_retry_at = time.monotonic() + self._FILTER_INDEX_RETRY_S
        failed: list[str] = []
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
                    failed.append(field_name)
                    logger.debug(f"payload index ensure failed for {field_name}: {e}")
        except Exception as e:
            failed.append("*")
            logger.debug(f"filter payload index ensure failed: {e}")
        if not failed:
            self._filter_indexes_ok = True
        elif not getattr(self, "_filter_index_warned", False):
            self._filter_index_warned = True
            logger.warning(
                "qdrant_payload_index_ensure_failed fields=%s: filters on these run unindexed "
                "(will re-check in %ds)",
                ",".join(failed),
                int(self._FILTER_INDEX_RETRY_S),
            )

    def _retry_index_ensure(self) -> None:
        """Re-run the background ensure after a failed attempt, once per cool-down."""
        now = time.monotonic()
        pending = []  # retry-at times of the ensures that have not succeeded
        if not self._filter_indexes_ok:
            pending.append(getattr(self, "_filter_index_retry_at", 0.0))
        if not getattr(self, "_created_at_index_ok", False):
            pending.append(getattr(self, "_created_at_retry_at", 0.0))
        # 0.0 = never attempted (init in progress): leave it to the startup hook.
        if not pending or any(t == 0.0 for t in pending) or now < min(pending):
            return
        thread = getattr(self, "_index_ensure_thread", None)
        if thread is not None and thread.is_alive():
            return
        self._ensure_indexes_at_startup()

    def _ensure_indexes_at_startup(self) -> None:
        """Ensure all payload indexes once memory is initialised, off the caller.

        Runs on a daemon thread so neither the event loop nor a cold-start
        request waits on Qdrant; never raises. Workers get the indexes before
        their first dedup probe, not on their first search.
        """

        def _run():
            try:
                self._ensure_filter_indexes()
                self._ensure_created_at_index()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"startup payload index ensure failed (non-fatal): {e}")

        thread = threading.Thread(target=_run, name="qdrant-index-ensure", daemon=True)
        self._index_ensure_thread = thread
        thread.start()

    def vector_store_capabilities(self) -> dict:
        """What the live vector store can do, for ``/health`` (public fields only).

        ``bm25_sparse``: the collection has the BM25 sparse slot (``False`` =
        predates v3, search is dense-only). ``bm25_encoder``: ``loaded`` /
        ``unavailable`` (fastembed missing or failed to load: dense-only even
        with the slot) / ``not_loaded`` (lazy, not used yet). ``None`` values
        mean not initialised or not reported. Free-text detail goes to the
        server log, not to this unauthenticated payload. Never initialises
        the store and never raises.
        """
        memory = self._memory
        store = getattr(memory, "vector_store", None) if memory is not None else None
        flag = getattr(store, "_has_bm25_slot", None) if store is not None else None
        if not isinstance(flag, bool):
            return {"bm25_sparse": None, "bm25_encoder": None}
        encoder = getattr(store, "_bm25_encoder", None)
        if encoder is False:
            state = "unavailable"
        elif encoder is None:
            state = "not_loaded"
        else:
            state = "loaded"
        if not flag:
            self._warn_bm25_missing_once()
        elif state == "unavailable":
            self._warn_bm25_encoder_once()
        return {"bm25_sparse": flag, "bm25_encoder": state}

    def _warn_bm25_encoder_once(self) -> None:
        global _bm25_encoder_warned
        if _bm25_encoder_warned:
            return
        _bm25_encoder_warned = True
        logger.warning(
            "vector_store_bm25_encoder_unavailable collection=%s: the BM25 encoder (fastembed) "
            "failed to load, so the keyword leg is off and search is dense-only.",
            settings.qdrant_collection,
        )

    def _warn_bm25_missing_once(self) -> None:
        """One structured warning per process when the BM25 slot is missing."""
        global _bm25_warned
        if _bm25_warned:
            return
        store = getattr(self._memory, "vector_store", None) if self._memory is not None else None
        if getattr(store, "_has_bm25_slot", None) is not False:
            return
        _bm25_warned = True
        logger.warning(
            "vector_store_bm25_slot_missing collection=%s: keyword leg disabled, search is dense-only. %s",
            settings.qdrant_collection,
            _BM25_MISSING_DETAIL,
        )
