"""Business logic layer for neuralscape memory service.

Both REST endpoints and MCP tools call into this same MemoryService.
"""

import hashlib
import json
import logging
import math
import random
import re
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


# Patterns matching raw tool/event log lines that should not be stored as memories
_JUNK_PATTERNS = [
    r"^Ran command:",
    r"^Edited file[:\s]",
    r"^Wrote file[:\s]",
    r"^Read file[:\s]",
    r"^Created file[:\s]",
    r"^Deleted file[:\s]",
    r"^Launched \w+ task:",
    r"^Tool result:",
    r"^Command output:",
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE)

# All known project group IDs for multi-group episode cleanup
ALL_KNOWN_PROJECTS = ["svc-utility-belt", "lightpath", "neuralscape", "openclaw"]

# Tags that mark a `standard` as ALWAYS-INJECT (surfaced in the session-start
# context regardless of relevance). Every other standard stays out of the
# always-on block and instead surfaces on demand, relevance-ranked, via recall.
_ALWAYS_INJECT_TAGS = ["critical", "always"]


def _parse_expires_at(value) -> datetime | None:
    """Parse an `expires_at` payload value to an aware UTC datetime.

    Accepts ISO-8601 strings (with or without a trailing `Z`), `datetime`
    instances (naive treated as UTC), or anything else returns None. Used by
    the expiry cron — comparing raw strings is unsafe across mixed offsets
    (`Z` vs `+00:00` vs `-05:00` won't sort lexicographically).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # datetime.fromisoformat doesn't accept the literal 'Z' suffix until
    # Python 3.11 fully; normalize defensively.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Reinforcement-aware dedup (times_derived) ──────────────────────────
# When a duplicate is detected — at write time (content-hash short-circuit),
# in the dedup cron, or by a dreaming MERGE — the survivor's
# `metadata.times_derived` counter absorbs the dropped duplicate(s) instead
# of silently discarding the reinforcement signal (inspired by Honcho's
# times_derived). The counter feeds the dreaming promotion score and a small
# deterministic recall re-rank boost below.

# Recall boost strength: boosted = score * (1 + K * log1p(times_derived - 1)).
# k=0.05 keeps the boost gentle — a memory reinforced 10× gets ~+12%, enough
# to outrank a one-off at comparable cosine similarity without letting
# repetition drown out relevance. `times_derived - 1` (the count of *extra*
# derivations) means unreinforced/legacy rows keep their raw score exactly.
#
# Audit 27 #4: the live value comes from `settings.reinforcement_boost_k`
# (env REINFORCEMENT_BOOST_K; 0 disables byte-identically) — this constant
# is only the historical default / test reference. `times_derived` is
# clamped at REINFORCEMENT_TIMES_DERIVED_CAP and the boosted score capped
# at 1.0, so an over-reinforced mediocre hit (e.g. 0.80 cosine, td=30) can
# no longer outrank a plainly better one (0.90, td=1), and returned scores
# stay in the cosine range.
REINFORCEMENT_BOOST_K = 0.05
REINFORCEMENT_TIMES_DERIVED_CAP = 10


# ── Hybrid recall: dense + BM25 lexical rank fusion (audit 27 #1) ──────
# Commit 7024a81 ("embed once") replaced mem0-v3 hybrid search with a
# dense-only exactly-k query per pool, silently amputating the BM25 leg —
# sparse vectors were still WRITTEN on every insert (mem0 fork qdrant.py)
# but never queried, so proper-noun recall collapsed. The helpers below
# restore a lexical pass per pool and fuse the two legs by reciprocal rank
# BEFORE the pool merge, keeping the single query embed.

# Standard RRF constant (Cormack et al.): a hit at 0-based rank r
# contributes 1 / (RRF_K + r + 1).
RRF_K = 60


def _rrf_fuse(dense_hits: list, lexical_hits: list, limit: int) -> list[dict]:
    """Rank-fuse one pool's dense and lexical (BM25) hit lists.

    Returns up to ``limit`` entries ``{"hit", "dense", "rrf"}`` ordered by
    summed reciprocal-rank score; a hit present in both legs accumulates
    both contributions, so leg agreement ranks it up. ``dense`` marks
    entries whose ``hit.score`` is a real cosine similarity — lexical-only
    entries carry a raw BM25 score that is NOT cosine-comparable, so the
    caller imputes a presentation score for those. With an empty lexical
    list this is an order-preserving passthrough of the dense ranking
    (1/(k+r) is strictly decreasing and the sort is stable).
    """
    fused: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits):
        hid = str(getattr(hit, "id", ""))
        fused[hid] = {"hit": hit, "dense": True, "rrf": 1.0 / (RRF_K + rank + 1)}
    for rank, hit in enumerate(lexical_hits):
        hid = str(getattr(hit, "id", ""))
        entry = fused.get(hid)
        if entry is None:
            fused[hid] = {"hit": hit, "dense": False, "rrf": 1.0 / (RRF_K + rank + 1)}
        else:
            entry["rrf"] += 1.0 / (RRF_K + rank + 1)
    ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return ordered[:limit]


def _dense_score_floor(dense_hits: list) -> float | None:
    """Weakest cosine score in a pool's dense leg (imputed score for
    lexical-only hits: presence via a strong keyword match is worth at
    least as much as the pool's weakest dense candidate — never more)."""
    scores = [
        s for s in (getattr(h, "score", None) for h in dense_hits) if s is not None
    ]
    return min(scores) if scores else None


def _times_derived_from_metadata(metadata: dict | None) -> int:
    """Read `times_derived` out of a memory's metadata dict (min 1).

    Handles the mem0 double-wrap (`{"metadata": {"metadata": {...}}}`) and
    treats absent/invalid values as 1 (a memory observed exactly once), so
    legacy rows never need a migration.
    """
    meta = metadata or {}
    if isinstance(meta.get("metadata"), dict):
        meta = meta["metadata"]
    try:
        return max(1, int(meta.get("times_derived") or 1))
    except (TypeError, ValueError):
        return 1


def _reinforcement_boost(score: float | None, metadata: dict | None) -> float | None:
    """Apply the deterministic reinforcement re-rank boost to a vector score.

    Feature-safe: None scores pass through, and times_derived <= 1 (including
    every legacy row without the field) returns the score unchanged.

    Audit 27 #4 guardrails:
    - k comes from ``settings.reinforcement_boost_k``; k <= 0 returns the
      raw score immediately (byte-identical disable path).
    - effective times_derived clamps at REINFORCEMENT_TIMES_DERIVED_CAP
      (=10 ⇒ max lift ≈ +12% at the default k), so unbounded reinforcement
      can never drown out relevance.
    - the boosted score caps at 1.0 — recall scores stay cosine-shaped.
    """
    if score is None:
        return None
    try:
        k = float(settings.reinforcement_boost_k)
    except (AttributeError, TypeError, ValueError):
        k = REINFORCEMENT_BOOST_K
    if k <= 0.0:
        return score
    times_derived = min(
        _times_derived_from_metadata(metadata), REINFORCEMENT_TIMES_DERIVED_CAP
    )
    if times_derived <= 1:
        return score
    return min(1.0, score * (1.0 + k * math.log1p(times_derived - 1)))


def _salience_tiebreak(responses: list) -> list:
    """A4 guardrail 1: bounded logarithmic salience tie-breaker on recall.

    OFF by default (``DREAMING_SALIENCE_RECALL_K=0.0``) — and when off (or
    on any failure) this returns immediately, without touching Redis or
    the responses, so the default search path is byte-identical to a
    world without salience dynamics. When enabled the boost is
    ``score * (1 + k*log1p(strength_signal))``: a small, bounded
    multiplicative lift (the config caps k at 0.1 ⇒ ≤ ~16% at the strength
    cap) so relevance dominates — a faded-but-relevant memory beats a
    hot-but-mediocre one at any permitted k. Salience never *gates*
    retrieval: nothing is dropped or filtered here, the boost only
    re-orders results whose relevance scores are already close.
    (Distinct from the ``times_derived`` boost above: that is
    reinforcement-by-dedup stored on the row; this is recall-frequency
    strength from the dreaming traces — the two signals never double-count
    one another.)
    """
    try:
        from extensions.dreaming.config import dreaming_settings

        k = float(dreaming_settings.salience_recall_k)
    except Exception:
        return responses
    if k <= 0.0 or not responses:
        return responses
    try:
        from extensions.dreaming.traces import get_strength_signals

        signals = get_strength_signals([r.id for r in responses if r.id])
        if not signals:
            return responses
        for r in responses:
            signal = signals.get(r.id or "", 0.0)
            if r.score is not None and signal > 0.0:
                r.score = r.score * (1.0 + k * math.log1p(signal))
    except Exception:
        logger.debug("salience tie-break skipped (non-fatal)", exc_info=True)
    return responses


def content_hash(content: str) -> str:
    """Canonical content hash for write-path dedup (audit 27 #6).

    The service-layer writers of ``payload["hash"]`` — store_raw, the
    conversation batch path, checkpoints, and the dreaming rewrite (which
    re-embeds new text in place) — all route through this helper; new
    writers should too. The exact-dedup cron groups rows by this value and
    hard-deletes "duplicates", so a stale or divergent hash silently
    corrupts dedup in both directions.
    """
    return hashlib.md5(content.encode()).hexdigest()


def _live_edges_filter():
    """Graphiti ``SearchFilters`` selecting only bi-temporally LIVE edges.

    Excludes any edge with a non-null ``invalid_at`` or ``expired_at`` —
    i.e. facts the dreaming sweep (or Graphiti's own contradiction
    handling) has invalidated/superseded (audit 27 #3). Built per call:
    SearchFilters is a mutable pydantic model and a shared singleton
    could be mutated by a concurrent caller.
    """
    from graphiti_core.search.search_filters import (
        ComparisonOperator,
        DateFilter,
        SearchFilters,
    )

    return SearchFilters(
        invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
        expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
    )


def _edge_is_invalidated(edge) -> bool:
    """True when a graph edge carries a bi-temporal invalidation stamp.

    Post-filter companion to ``_live_edges_filter`` for result paths where
    the Cypher-level filter may not have applied. Only real timestamp
    values (datetime or non-empty string) count — mock/partial edge
    objects without the attribute are treated as live.
    """
    for attr in ("invalid_at", "expired_at"):
        value = getattr(edge, attr, None)
        if isinstance(value, datetime) or (isinstance(value, str) and value.strip()):
            return True
    return False


def _is_junk_fact(content: str) -> bool:
    """Return True if an extracted fact is a raw event log rather than contextual knowledge."""
    stripped = content.strip()
    if len(stripped) < 10:
        return True
    return bool(_JUNK_RE.search(stripped))


def _deleted_msg(noun: str, deleted: int, skipped_shared: int, skipped_standard: int) -> str:
    """Human-readable summary for a filtered delete, naming each preserved tier.

    Standards and shared writes are preserved for different reasons (dictator-only
    vs. team-owned), so report them separately rather than lumping standards under
    "shared".
    """
    msg = f"Deleted {deleted} {noun}"
    preserved = []
    if skipped_shared:
        preserved.append(f"{skipped_shared} shared")
    if skipped_standard:
        preserved.append(f"{skipped_standard} standard")
    if preserved:
        msg += f" (preserved {', '.join(preserved)})"
    return msg


def _clean_conversation_for_graph(messages: list[dict]) -> list[dict]:
    """Filter junk lines from conversation messages before graph ingestion.

    Removes lines matching _JUNK_RE from each message's content.
    Messages that become empty after filtering are dropped entirely.
    """
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            cleaned.append(msg)
            continue
        clean_lines = [
            line for line in content.splitlines()
            if not _JUNK_RE.match(line.strip())
        ]
        clean_content = "\n".join(clean_lines).strip()
        if clean_content:
            cleaned.append({**msg, "content": clean_content})
    return cleaned


# Known project slugs for project_id inference
_KNOWN_PROJECT_SLUGS = [
    "svc-utility-belt",
    "lightpath",
    "neuralscape",
    "openclaw",
]


def _infer_project_id(content: str) -> str | None:
    """Try to infer a project_id from memory content by matching known project slugs."""
    content_lower = content.lower()
    for slug in _KNOWN_PROJECT_SLUGS:
        if slug in content_lower:
            return slug
    return None


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
from index_format import distill_title, estimate_tokens  # noqa: F401 — estimate_tokens re-exported for legacy callers/tests
# E2: write-time token stamping — a REAL tiktoken count when the savings
# meter is enabled, the len/4 heuristic when it's off (zero tokenizer calls).
# The stored field keeps its `token_estimate` name either way.
from savings_meter import stamp_tokens
from prompts import (
    build_extraction_messages,
    parse_extraction_response,
)
from schemas import (
    EPISTEMIC_LEVEL_VOCAB,
    GLOBAL_CATEGORIES,
    MEMORY_CATEGORIES,
    PROJECT_CATEGORIES,
    ContextResponse,
    MemoryResponse,
    MemoryScope,
    MemoryVisibility,
    default_scope_for_category,
    default_visibility_for_category,
    normalize_visibility,
)

logger = logging.getLogger(__name__)

# Retrieval economics (C1/C2) bounds: max ids per batch-get and max window
# half-width for the timeline tool. Shared by REST + MCP boundaries.
GET_MEMORIES_MAX_IDS = 50
TIMELINE_MAX_DEPTH = 50


def _created_at_key(value) -> datetime:
    """Chronological sort key for a created_at payload value.

    Parses ISO-8601 (any offset — mem0-written rows may carry non-UTC
    offsets, where lexicographic string comparison missorts). Unparseable /
    missing values sort first.
    """
    if value:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)

# Structured audit trail for authoritative-context serving (standards +
# processes). Rendered as JSON in prod via logging_config; a plain stdlib
# logger is used so this has no hard dependency on structlog being configured.
import structlog  # noqa: E402

_audit_log = structlog.get_logger("neuralscape.audit")

# Process slugs are constrained to a tag-safe charset so Qdrant tag filters
# match exactly and slugs round-trip cleanly through `process:<slug>` tags.
_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")


def _build_group_id(
    visibility: str,
    user_id: str,
    project_id: str | None = None,
) -> str:
    """Build a Graphiti group_id for the multi-user model.

    | visibility | project_id | group_id                          |
    |------------|------------|-----------------------------------|
    | private    | None       | user--{user_id}                   |
    | private    | set        | user--{user_id}--project--{pid}   |
    | shared     | None       | shared                            |
    | shared     | set        | shared--project--{project_id}     |

    The user namespace fixes the prior cross-user leak in Graphiti
    (previously all users shared `"global"` / `"project--..."`). The
    `shared` namespace is the team-wide knowledge pool readable by any
    authenticated user.
    """
    # Tolerate enum / str / legacy "MemoryVisibility.X" / None — see
    # normalize_visibility docstring for why this matters. Unrecognized
    # values fall back to PRIVATE so an unknown visibility never
    # accidentally lands a memory in the shared pool (safe default
    # preserves the pre-fix behavior of ``str(visibility or PRIVATE)``).
    try:
        vis = normalize_visibility(visibility) or MemoryVisibility.PRIVATE.value
    except (ValueError, TypeError):
        vis = MemoryVisibility.PRIVATE.value
    if vis == MemoryVisibility.STANDARD.value:
        # Authoritative org-wide pool: dictator-written, everyone-readable.
        if project_id:
            return f"standard--project--{project_id}"
        return "standard"
    if vis == MemoryVisibility.SHARED.value:
        if project_id:
            return f"shared--project--{project_id}"
        return "shared"
    # private
    if project_id:
        return f"user--{user_id}--project--{project_id}"
    return f"user--{user_id}"


def _get_group_ids(caller_user_id: str, project_id: str | None = None) -> list[str]:
    """Group ids the caller is permitted to read across.

    Returns the caller's private namespace + the shared pool, plus the
    project-scoped equivalents when `project_id` is given. A read
    against this set returns the union of the caller's private memories
    and all shared memories (no cross-user private leakage).

    When the `standard` tier is enabled, the authoritative pool is appended
    for EVERY caller (including anonymous/legacy-key readers) so binding org
    standards are always visible.
    """
    def _standard_groups() -> list[str]:
        if not settings.standards_enabled:
            return []
        return ["standard"] + ([f"standard--project--{project_id}"] if project_id else [])

    if not caller_user_id:
        # Anonymous / unauthenticated readers see the shared + standard pools.
        anon = ["shared"] + ([f"shared--project--{project_id}"] if project_id else [])
        return anon + _standard_groups()
    group_ids = [f"user--{caller_user_id}", "shared"]
    if project_id:
        group_ids.append(f"user--{caller_user_id}--project--{project_id}")
        group_ids.append(f"shared--project--{project_id}")
    return group_ids + _standard_groups()


def _check_edit_permission(
    meta: dict,
    payload_user_id: str,
    caller_user_id: str | None,
    *,
    edits_content: bool,
    edits_visibility: bool,
) -> None:
    """Gate an edit against the memory's visibility tier and ownership.

    The split model (locked with the team):
    - dictators may edit anything;
    - ``standard`` tier is dictator-only (mirrors the delete gate);
    - ``shared`` memories: organizational metadata (tags/category/project/v2
      fields) is team-editable housekeeping, but *content* and *visibility*
      changes rewrite or re-tier someone's words — owner only;
    - ``private`` (and legacy null-visibility) memories: owner only.

    Raises PermissionError; returns None when the edit is allowed.
    """
    if settings.is_dictator(caller_user_id):
        return
    owner = meta.get("owner_user_id") or payload_user_id or ""
    try:
        vis = normalize_visibility(meta.get("visibility")) or MemoryVisibility.PRIVATE.value
    except (ValueError, TypeError):
        vis = MemoryVisibility.PRIVATE.value
    if vis == MemoryVisibility.STANDARD.value:
        raise PermissionError("Only a dictator may edit 'standard'-tier memories.")
    if vis == MemoryVisibility.SHARED.value:
        if (edits_content or edits_visibility) and caller_user_id != owner:
            raise PermissionError(
                "Only the memory's owner may edit its content or visibility "
                f"(owner: {owner!r}). Metadata edits (tags/category/project) are open to the team."
            )
        return
    # private / legacy null visibility
    if caller_user_id != owner:
        raise PermissionError(f"Only the memory's owner may edit it (owner: {owner!r}).")


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

    def _attach_memory_id_to_graph_nodes(
        self,
        *,
        group_id: str,
        memory_id: str,
        visibility: str,
        owner_user_id: str,
        write_started_at: datetime,
        source_ref: dict | None = None,
    ) -> None:
        """Stamp ``memory_id``/visibility/owner onto recently-created Graphiti nodes.

        Called by every store path after the Graphiti write completes. The
        wiki synthesizer uses these properties to walk community → source
        memories. When ``source_ref`` is present (data-layer-connector
        ingestion), also links those nodes to a ``(:Source)`` node so the
        graph can walk memory → source → connector. Failures here never fail
        the underlying write — the helpers log and return 0.
        """
        if not (self._graphiti and self._bridge):
            return
        from extensions.dreaming.config import dreaming_settings as synthesizer_settings
        from extensions.dreaming.graph_patcher import (
            attach_memory_id,
            attach_source_ref,
        )

        # Coroutine built separately so we can ``.close()`` it if
        # ``_run_on_bridge`` raises before awaiting — otherwise tests
        # with mocked bridges leak ``RuntimeWarning: coroutine never
        # awaited`` and slow the suite down with traceback collection.
        coro = attach_memory_id(
            self._graphiti.driver,
            group_id=group_id,
            memory_id=memory_id,
            visibility=visibility,
            owner_user_id=owner_user_id,
            write_started_at=write_started_at,
            window_seconds=synthesizer_settings.attach_window_seconds,
        )
        try:
            self._run_on_bridge(coro, timeout=10.0)
        except Exception:
            coro.close()
            logger.warning(
                "attach_memory_id post-write hook failed (non-critical)",
                exc_info=True,
            )

        # Link to the originating data-layer source, when ingested via a connector.
        if source_ref:
            src_coro = attach_source_ref(
                self._graphiti.driver,
                group_id=group_id,
                memory_id=memory_id,
                source_ref=source_ref,
                write_started_at=write_started_at,
                window_seconds=synthesizer_settings.attach_window_seconds,
            )
            try:
                self._run_on_bridge(src_coro, timeout=10.0)
            except Exception:
                src_coro.close()
                logger.warning(
                    "attach_source_ref post-write hook failed (non-critical)",
                    exc_info=True,
                )

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

        # Step 1: Call Gemini for fact extraction. E4: operator-supplied
        # extraction instructions (per-project + per-user, composed) ride
        # along as a clearly-delimited addendum that can steer fact
        # selection/phrasing but never the JSON output contract.
        from extraction_settings import resolve_instructions

        operator_guidance = resolve_instructions(user_id, project_id)
        extraction_messages = build_extraction_messages(
            messages, operator_guidance=operator_guidance
        )
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
            # Loud failure: propagate so the ARQ job (and task status) reports
            # failed instead of completing with zero facts. A silent [] here
            # masked real outages (the caller can't tell "nothing worth
            # extracting" from "extraction is down").
            logger.error(
                f"LLM extraction failed for user {user_id}: {e}", exc_info=True
            )
            raise

        # Filter out junk facts from extraction
        pre_filter_count = len(parsed_facts)
        parsed_facts = [
            (cat, content) for cat, content in parsed_facts
            if not _is_junk_fact(content)
        ]
        if pre_filter_count != len(parsed_facts):
            logger.info(
                f"Filtered {pre_filter_count - len(parsed_facts)} junk facts from extraction"
            )

        if not parsed_facts:
            logger.info("No facts extracted from conversation")
            return []

        # Step 2: Batch-store all facts (single embed + single Qdrant upsert).
        # A failure here (embedding, Qdrant) must PROPAGATE: the previous
        # except-and-continue silently stored zero facts while the task
        # reported success — exactly what hid the gateway's single-input
        # embed rejection in production. Raising fails the ARQ job, so
        # /v1/memories/status/{task_id} reports status=failed with the error.
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
            logger.error(
                f"Batch store failed for user {user_id}: {e} — "
                f"{len(parsed_facts)} extracted facts were NOT stored",
                exc_info=True,
            )
            raise

        # Step 3: Add cleaned conversation text to knowledge graph.
        # Conversation extractions are personal (private) by default — the
        # caller's spoken context isn't team-shared automatically.
        group_id = _build_group_id(
            MemoryVisibility.PRIVATE.value,
            user_id,
            project_id,
        )
        cleaned_messages = _clean_conversation_for_graph(messages)
        raw_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in cleaned_messages
        )
        graph_write_started_at = datetime.now(timezone.utc)
        try:
            if self._graphiti and self._bridge and raw_text.strip():
                retry_transient(
                    self._memory.graph.add,
                    data=raw_text,
                    filters={"user_id": user_id, "group_id": group_id},
                    operation="graph storage (extract_and_store)",
                )
                # 1-episode → N-memories shape: a single graph.add produces
                # entities that could legitimately belong to any of the N
                # extracted memories. We call attach_memory_id once per
                # stored memory; the Cypher uses ``coalesce(memory_id,
                # $memory_id)``, so the first call wins and later calls are
                # no-ops on the same node. The result is each fresh entity
                # carries one representative memory_id from the batch — not
                # a perfect mapping, but enough for the wiki synthesizer
                # to walk community → source memory.
                for mem in stored:
                    self._attach_memory_id_to_graph_nodes(
                        group_id=group_id,
                        memory_id=getattr(mem, "id", "") or "",
                        visibility=MemoryVisibility.PRIVATE.value,
                        owner_user_id=user_id,
                        write_started_at=graph_write_started_at,
                    )
        except Exception as e:
            logger.warning(f"Graph storage failed (non-critical): {e}")

        return stored

    def extract_facts_only(
        self,
        text: str,
        extractor=None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Run LLM fact extraction over a block of text and return (category, content) tuples.

        Reuses the same Gemini extraction + junk filter as ``extract_and_store``
        but stores nothing — the caller decides how to persist (the ingest
        pipeline stores them with ``source_type="imported"`` and the document's
        ``source_ref``). Returns ``[]`` on extraction failure (logged), so a
        flaky LLM call degrades to passages-only rather than failing ingest.

        ``extractor`` is an optional :class:`ingest.extractors.FactExtractor`
        supplied by a knowledge adapter — it owns the extraction *prompt* and the
        *parse* of the response. When ``None`` (the default), behavior is
        byte-for-byte the coding-assistant extractor: the LLM client, retries,
        model selection, and junk filter below are shared across all adapters so
        only the taxonomy/prompt varies, never the envelope.

        ``user_id`` / ``project_id`` scope the E4 operator guidance lookup:
        when set, the composed custom extraction instructions are appended
        to the prompt as the OPERATOR GUIDANCE addendum — AFTER the adapter's
        own prompt, so instructions compose with (never replace) adapters.
        """
        if not text or not text.strip():
            return []
        if extractor is not None:
            extraction_messages = extractor.build_messages(text)
        else:
            extraction_messages = build_extraction_messages([{"role": "user", "content": text}])
        # E4: operator guidance rides after the (possibly adapter-owned)
        # prompt. Best-effort — resolve failure means no addendum.
        from extraction_settings import resolve_instructions
        from prompts import append_operator_guidance

        operator_guidance = resolve_instructions(user_id, project_id)
        if operator_guidance:
            extraction_messages[0]["content"] = append_operator_guidance(
                extraction_messages[0]["content"], operator_guidance
            )
        client = self._get_genai_client()
        try:
            from google.genai.types import GenerateContentConfig, HttpOptions

            response = retry_transient(
                client.models.generate_content,
                model=settings.gemini_llm_model,
                contents=extraction_messages[0]["content"],
                config=GenerateContentConfig(
                    http_options=HttpOptions(timeout=60_000),
                ),
                operation="LLM extraction (ingest)",
                fallback_model=settings.gemini_llm_fallback_model,
            )
            if extractor is not None:
                parsed_facts = extractor.parse(response.text)
            else:
                parsed_facts = parse_extraction_response(response.text)
        except Exception as e:
            logger.error(f"Ingest extraction failed: {e}")
            return []
        return [(cat, content) for cat, content in parsed_facts if not _is_junk_fact(content)]

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
        # Memory-model v2 fields (all optional, all additive)
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        source_type: str | None = None,
        related_memory_ids: list[str] | None = None,
        confidence: float | None = None,
        expires_at: datetime | None = None,
        # Provenance epistemics (A1, optional)
        derived_from: list[str] | None = None,
        epistemic_level: str | None = None,
        # Multi-user model (None → category default)
        visibility: str | None = None,
        # Data-layer connectors (None → omitted)
        memory_kind: str | None = None,
        source_ref: dict | None = None,
        # Write-path isolation: when False, skip the slow inline graph.add so
        # the fast vector write returns immediately — the caller is expected to
        # enqueue enrich_graph() onto the graph worker. When return_created is
        # True, returns (responses, created) so the caller only enqueues graph
        # enrichment for genuinely new rows (not content-hash dedup hits).
        add_to_graph: bool = True,
        return_created: bool = False,
        # Knowledge-adapter graph ontology (Graphiti custom types), resolved
        # per-ingest and forwarded to enrich_graph → add_episode. None ⇒
        # Graphiti's built-in generic extraction (every regular memory write).
        graph_ontology: dict | None = None,
    ) -> list[MemoryResponse] | tuple[list[MemoryResponse], bool]:
        """Store a single pre-categorized fact directly (no LLM extraction).

        Returns a ``list[MemoryResponse]`` by default, or ``(responses, created)``
        when ``return_created=True`` (``created`` is False for content-hash dedup
        hits) — callers must not assume the return is always a list.

        Performs content-hash dedup before insert: if a memory with the same
        md5(content) + user_id + scope already exists, the existing memory is
        returned instead of creating a duplicate. This makes hook-driven
        re-flushes idempotent.

        Args:
            content: The fact to store
            user_id: User identifier
            category: Memory category (validated against MEMORY_CATEGORIES)
            scope: "global" or "project"
            project_id: Required when scope="project"
            tags: Optional free-form tags
            agent_id: Optional agent identifier
            run_id: Optional session identifier
            domain: Memory-model v2 — life-context domain
            observation_type: Memory-model v2 — shape of observation
            concepts: Memory-model v2 — controlled-vocab cross-cutting tags
            source_type: Memory-model v2 — provenance
            related_memory_ids: Memory-model v2 — graph linkage
            confidence: Memory-model v2 — extractor's self-rated 0.0-1.0
            expires_at: Memory-model v2 — optional expiry timestamp
            derived_from: A1 provenance — premise memory IDs this memory was
                derived from (walked by get_reasoning_chain)
            epistemic_level: A1 provenance — how this memory is known
                (explicit | deductive | inductive | reflection)

        Returns:
            List of stored memory responses (always exactly one item). When the
            content-hash dedup short-circuits, the returned `id`/`created_at`
            will match the existing memory; otherwise they reflect the new row.
            Both paths use `source="vector"`; callers that need to distinguish
            should compare the returned `id` against their expected new UUID.
        """
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of: {list(MEMORY_CATEGORIES.keys())}")
        if epistemic_level is not None and epistemic_level not in EPISTEMIC_LEVEL_VOCAB:
            raise ValueError(
                f"Invalid epistemic_level: {epistemic_level}. "
                f"Must be one of: {sorted(EPISTEMIC_LEVEL_VOCAB)}"
            )
        # Normalize empty→None so the response echoes exactly what was
        # persisted (metadata only stores truthy lists; echoing [] while
        # storing nothing would make the write and later reads disagree).
        derived_from = derived_from or None

        # Resolve visibility: explicit caller value > per-category default.
        # ``normalize_visibility`` handles MemoryVisibility enum, plain
        # str, and the legacy "MemoryVisibility.X" stringified-enum
        # format from the pre-__str__-override bug (Python 3.11+ str(Enum)
        # regression). Without normalization, "MemoryVisibility.SHARED"
        # used to land in Qdrant metadata and break both the GET API
        # shape and the conversation_compiler event handler.
        # Resolved BEFORE the scope check because standards force scope below.
        effective_visibility = (
            normalize_visibility(visibility)
            if visibility is not None
            else default_visibility_for_category(category).value
        )

        # ── Authoritative "standard" tier: gate + force global scope ──
        # The REST route / MCP tool gate this earlier and synchronously, but
        # this covers every store path (worker, batch, sync-fallback) so a
        # standard memory can only ever be written by an authorized dictator.
        # Standards are org-wide by definition → always global scope, no
        # project_id (so a project-category standard like a global `convention`
        # doesn't inherit scope="project" and fail the project check below).
        if effective_visibility == MemoryVisibility.STANDARD.value:
            if not settings.standards_enabled:
                raise PermissionError(
                    "The 'standard' visibility tier is disabled (set STANDARDS_ENABLED=true)."
                )
            if not settings.is_dictator(user_id):
                raise PermissionError(
                    f"User {user_id!r} is not authorized to write 'standard'-tier memories."
                )
            scope = "global"
            project_id = None

        if scope == "project" and not project_id:
            raise ValueError("project_id is required when scope='project'")

        m = self._get_memory()
        now_iso = datetime.now(timezone.utc).isoformat()
        chash = content_hash(content)

        # ── Content-hash dedup ──
        # Skip insert if this exact (user_id, scope, hash) already exists.
        existing = self._find_by_content_hash(
            user_id=user_id, content_hash=chash, scope=scope,
            project_id=project_id, visibility=effective_visibility,
        )
        if existing is not None:
            logger.info(
                f"Dedup hit for user={user_id} hash={chash[:8]}... — returning existing id={existing.id}"
            )
            # Reinforcement-aware dedup: the caller just re-derived this exact
            # fact — count it on the survivor instead of discarding the signal.
            self._bump_times_derived(existing.id, 1)
            # Audit 27 #5: if the survivor was dream-tombstoned, re-derivation
            # resurrects it — otherwise this dedup short-circuit silently
            # swallows the write while the fact stays excluded from recall.
            # Runs AFTER the bump: the bump's read-merge-write could otherwise
            # re-persist the stale tombstone flag it read moments earlier.
            if self._revive_if_tombstoned(existing.id):
                existing.revived = True
            # created=False → caller must NOT re-enqueue graph enrichment.
            return ([existing], False) if return_created else [existing]

        # ── Direct embed + Qdrant insert (bypass m.add) ──
        mid = str(uuid.uuid4())
        embedding = m.embedding_model.embed(content, memory_action="add")

        # Retrieval economics (C1): distill a ~10-word title + token cost at
        # write time so index-only recall never has to fetch/parse content.
        # Pure heuristics — no LLM call on the hot write path.
        title = distill_title(content)
        token_estimate = stamp_tokens(content)

        metadata: dict = {
            "scope": scope,
            "category": category,
            "project_id": project_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "source": "explicit",
            # Multi-user model: who owns this memory + who can read it.
            "owner_user_id": user_id,
            "visibility": effective_visibility,
            # Retrieval economics (C1): index-row fields
            "title": title,
            "token_estimate": token_estimate,
        }
        if tags:
            metadata["tags"] = tags
        # Memory-model v2 metadata (only stored when set)
        if domain is not None:
            metadata["domain"] = domain
        if observation_type is not None:
            metadata["observation_type"] = observation_type
        if concepts:
            metadata["concepts"] = concepts
        if source_type is not None:
            metadata["source_type"] = source_type
        if related_memory_ids:
            metadata["related_memory_ids"] = related_memory_ids
        if confidence is not None:
            metadata["confidence"] = confidence
        # A1 provenance (only stored when set)
        if derived_from:
            metadata["derived_from"] = derived_from
        if epistemic_level is not None:
            metadata["epistemic_level"] = epistemic_level
        if expires_at is not None:
            metadata["expires_at"] = expires_at.isoformat() if hasattr(expires_at, "isoformat") else str(expires_at)
        # Data-layer connector provenance
        if memory_kind is not None:
            metadata["memory_kind"] = memory_kind
        if source_ref:
            metadata["source_ref"] = source_ref

        payload = {
            "data": content,
            "hash": chash,
            "created_at": now_iso,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": metadata,
        }

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
        # Graphiti entity extraction is slow (~minutes, gated by Gemini). When
        # add_to_graph is False the caller enqueues enrich_graph() onto the
        # graph worker instead, so this fast path returns right after the
        # vector insert. source_ref (connector provenance) rides along so the
        # back-reference attached to the graph nodes records where it came from.
        if add_to_graph:
            self.enrich_graph(
                content=content,
                user_id=user_id,
                project_id=project_id,
                visibility=effective_visibility,
                memory_id=mid,
                source_ref=source_ref,
                graph_ontology=graph_ontology,
            )

        responses = [
            MemoryResponse(
                id=mid,
                memory=content,
                category=category,
                scope=scope,
                project_id=project_id,
                tags=tags,
                source="vector",
                created_at=now_iso,
                domain=domain,
                observation_type=observation_type,
                concepts=concepts,
                source_type=source_type,
                related_memory_ids=related_memory_ids,
                confidence=confidence,
                expires_at=expires_at.isoformat() if expires_at and hasattr(expires_at, "isoformat") else None,
                derived_from=derived_from,
                epistemic_level=epistemic_level,
                memory_kind=memory_kind,
                source_ref=source_ref,
                visibility=effective_visibility,
                owner_user_id=user_id,
                title=title,
                token_estimate=token_estimate,
            )
        ]
        # created=True → a new row was written, so the caller should enqueue
        # graph enrichment (when it deferred it via add_to_graph=False).
        return (responses, True) if return_created else responses

    def enrich_graph(
        self,
        content: str,
        user_id: str,
        project_id: str | None,
        visibility: str,
        memory_id: str,
        source_ref: dict | None = None,
        graph_ontology: dict | None = None,
    ) -> bool:
        """Add content to the knowledge graph + attach memory_id back-refs.

        Extracted from store_raw so it can run either inline (add_to_graph=True)
        or — preferably — as a separate background job on the graph worker,
        since Graphiti entity extraction is slow (~minutes, Gemini-gated) and
        must not block the fast vector write or the API/MCP event loop.
        Best-effort: logs and swallows errors, never raises.

        ``graph_ontology`` is an optional dict of Graphiti custom-type kwargs
        (``entity_types``/``edge_types``/``edge_type_map``/
        ``excluded_entity_types``/``custom_extraction_instructions``) supplied by
        a knowledge adapter and resolved *per ingest*. When None (the default,
        and every regular memory write), the graph write is byte-for-byte the
        pre-adapter path.

        Returns True if the graph write actually succeeded, False if it was
        skipped (no graph configured) or swallowed an error. Callers use this
        to report honest enrichment status instead of assuming success — a
        transient Gemini 503 leaves the memory vector-only, and that must be
        observable rather than reported as ``enriched=True``.
        """
        if not (self._graphiti and self._bridge):
            return False
        # Group_id encodes visibility + user namespace so graph search can
        # scope by allowed groups without re-leaking cross-user facts.
        group_id = _build_group_id(visibility, user_id, project_id)
        graph_write_started_at = datetime.now(timezone.utc)
        try:
            retry_transient(
                self._memory.graph.add,
                data=content,
                filters={"user_id": user_id, "group_id": group_id},
                operation="enrich_graph add",
                **(graph_ontology or {}),
            )
            # Best-effort: attach memory_id back-reference onto the entity nodes
            # Graphiti just created in this group, so the wiki synthesizer can
            # walk community → source memories.
            self._attach_memory_id_to_graph_nodes(
                group_id=group_id,
                memory_id=memory_id,
                visibility=visibility,
                owner_user_id=user_id,
                write_started_at=graph_write_started_at,
                source_ref=source_ref,
            )
            return True
        except Exception as e:
            logger.warning(f"Graph enrichment failed (non-critical): {e}")
            return False

    def _search_shared_pool(
        self,
        m,
        query: str,
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
        visibility_value: str = MemoryVisibility.SHARED.value,
    ) -> list[MemoryResponse]:
        """Search Qdrant for cross-writer memories of a given visibility.

        Used by ``search()`` to deliver team-wide knowledge to all
        authenticated callers. Bypasses mem0's wrapper because that
        wrapper enforces user_id namespacing — for the shared/standard pools we
        explicitly want hits across writers, scoped by
        ``metadata.visibility=<visibility_value>`` plus any other supplied
        filters. ``visibility_value`` selects the pool: ``"shared"`` (default,
        team-wide) or ``"standard"`` (authoritative dictator-written).

        ``query_embedding`` lets the caller pass a precomputed query vector so a
        single ``search()`` doesn't re-embed the same query for every pool/scope
        (the embed round-trip dominates read latency); falls back to embedding
        ``query`` when not provided.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        client = m.vector_store.client
        embedding = (
            query_embedding
            if query_embedding is not None
            else m.embedding_model.embed(query, memory_action="search")
        )

        must: list = [
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=visibility_value),
            )
        ]
        # Dreaming: exclude reversible consolidation tombstones from recall.
        must_not = [
            FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True))
        ]
        if categories:
            must.append(FieldCondition(key="metadata.category", match=MatchAny(any=categories)))
        if scope:
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=scope)))
        if project_id:
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
        if domain:
            must.append(FieldCondition(key="metadata.domain", match=MatchValue(value=domain)))
        if observation_type:
            must.append(FieldCondition(key="metadata.observation_type", match=MatchValue(value=observation_type)))
        if concepts:
            must.append(FieldCondition(key="metadata.concepts", match=MatchAny(any=concepts)))

        # qdrant-client v1.13+ removed `.search()` in favor of `.query_points()`;
        # the response wraps hits in a `.points` attribute.
        qfilter = Filter(must=must, must_not=must_not)
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=embedding,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        dense_hits = list(getattr(result, "points", result) or [])

        # Hybrid recall (audit 27 #1): BM25 lexical leg under the SAME
        # filter, rank-fused with the dense leg before the pool merge.
        lexical_hits = self._lexical_pool_hits(m, query, qfilter, limit)
        fused = _rrf_fuse(dense_hits, lexical_hits, limit)
        dense_floor = _dense_score_floor(dense_hits)

        out: list[MemoryResponse] = []
        for entry in fused:
            hit = entry["hit"]
            payload = getattr(hit, "payload", None) or {}
            metadata = payload.get("metadata", {})
            # Lexical-only hits carry a raw BM25 score (not cosine-comparable);
            # impute the pool's dense floor so the cross-pool score sort stays
            # meaningful without ever letting a keyword match outrank a
            # stronger dense hit on score alone.
            raw_score = getattr(hit, "score", None) if entry["dense"] else dense_floor
            mem_dict = {
                "id": str(getattr(hit, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": metadata,
                # Reinforcement-aware recall: memories that survived N dedup
                # collapses rank slightly above one-offs at equal similarity.
                "score": _reinforcement_boost(raw_score, metadata),
                "created_at": payload.get("created_at"),
            }
            out.append(self._mem_to_response(mem_dict))
        return out

    def _lexical_pool_hits(self, m, query: str, query_filter, limit: int) -> list:
        """BM25 lexical leg for one pool (audit 27 #1).

        Runs a sparse-vector keyword query against the collection's
        ``bm25`` named-vector slot using the SAME payload filter as the
        pool's dense pass, so the lexical leg can never widen
        visibility/scope. Degrades to an empty list (dense-only search,
        no error) when:

        - the collection predates the v3 hybrid schema (no ``bm25`` sparse
          slot — the mem0 fork detects this at startup via
          ``_has_bm25_slot``),
        - the configured vector store isn't the NS mem0 Qdrant fork,
        - the query is empty or sparse encoding fails,
        - the Qdrant call errors — recall must never break on the lexical
          extra.
        """
        vs = m.vector_store
        # `is True` (not truthiness): only the fork sets a real bool; any
        # other store lacks the attribute and must skip the sparse leg.
        if not query or getattr(vs, "_has_bm25_slot", False) is not True:
            return []
        try:
            sparse = vs._encode_bm25(query)
            if sparse is None:
                return []
            result = vs.client.query_points(
                collection_name=settings.qdrant_collection,
                query=sparse,
                using="bm25",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result) or [])
        except Exception as e:
            logger.debug(f"BM25 lexical leg failed (non-fatal, dense-only): {e}")
            return []

    def _search_standard_pool(
        self,
        m,
        query: str,
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryResponse]:
        """Search the authoritative ``standard``-tier pool (dictator-written).

        Thin wrapper over ``_search_shared_pool`` that scopes to
        ``metadata.visibility=standard``. Returned to every caller so binding
        org standards surface in recall regardless of ``include_shared``.
        """
        return self._search_shared_pool(
            m=m,
            query=query,
            project_id=project_id,
            categories=categories,
            scope=scope,
            domain=domain,
            observation_type=observation_type,
            concepts=concepts,
            limit=limit,
            query_embedding=query_embedding,
            visibility_value=MemoryVisibility.STANDARD.value,
        )

    def _search_personal_pool(
        self,
        m,
        user_id: str,
        query_embedding: list[float],
        project_id: str | None,
        categories: list[str] | None,
        scope: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
        limit: int,
        query: str = "",
    ) -> list[MemoryResponse]:
        """Search Qdrant for the caller's own memories using a precomputed vector.

        Mirrors ``_search_shared_pool`` but scopes by the top-level ``user_id``
        payload field (mem0's namespace) instead of ``visibility=shared``, and
        returns the caller's memories regardless of visibility. We query Qdrant
        directly — like the shared pool — rather than via ``Memory.search`` so a
        single ``search()`` embeds the query ONCE and reuses ``query_embedding``
        across every pool/scope, instead of re-embedding per ``Memory.search``
        call (the embed round-trip dominates read latency).

        ``query`` (raw text) feeds the BM25 lexical leg only — sparse
        encoding is lexical, not an embed round-trip. Empty ⇒ dense-only.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        client = m.vector_store.client
        must: list = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        # Dreaming: rows consolidated away by a sweep stay in Qdrant as
        # reversible tombstones but are excluded from live recall.
        must_not = [
            FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True))
        ]
        if categories:
            must.append(FieldCondition(key="metadata.category", match=MatchAny(any=categories)))
        if scope:
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=scope)))
        if project_id:
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
        if domain:
            must.append(FieldCondition(key="metadata.domain", match=MatchValue(value=domain)))
        if observation_type:
            must.append(FieldCondition(key="metadata.observation_type", match=MatchValue(value=observation_type)))
        if concepts:
            must.append(FieldCondition(key="metadata.concepts", match=MatchAny(any=concepts)))

        qfilter = Filter(must=must, must_not=must_not)
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_embedding,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        dense_hits = list(getattr(result, "points", result) or [])

        # Hybrid recall (audit 27 #1): BM25 lexical leg under the SAME
        # filter, rank-fused with the dense leg before the pool merge.
        lexical_hits = self._lexical_pool_hits(m, query, qfilter, limit)
        fused = _rrf_fuse(dense_hits, lexical_hits, limit)
        dense_floor = _dense_score_floor(dense_hits)

        out: list[MemoryResponse] = []
        for entry in fused:
            hit = entry["hit"]
            payload = getattr(hit, "payload", None) or {}
            metadata = payload.get("metadata", {})
            # Lexical-only hits: impute the pool's dense floor (see
            # _search_shared_pool for the rationale).
            raw_score = getattr(hit, "score", None) if entry["dense"] else dense_floor
            mem_dict = {
                "id": str(getattr(hit, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": metadata,
                # Same reinforcement boost as the shared/standard pools — the
                # re-rank must be identical across pools or a reinforced
                # private memory would lose to its unreinforced shared twin.
                "score": _reinforcement_boost(raw_score, metadata),
                "created_at": payload.get("created_at"),
            }
            out.append(self._mem_to_response(mem_dict))
        return out

    def _find_by_content_hash(
        self,
        user_id: str,
        content_hash: str,
        scope: str,
        project_id: str | None = None,
        visibility: str | None = None,
    ) -> MemoryResponse | None:
        """Look up a memory by (user_id, hash, scope, visibility) for dedup.

        Returns the existing MemoryResponse on hit, or None if not found.
        Failures here are non-fatal — we'd rather risk a duplicate than
        block an insert.

        ``visibility`` is part of the key: the same text at two different tiers
        (e.g. a dictator's private note vs. an authoritative ``standard``) are
        distinct memories, so a ``standard`` write must not dedup onto a
        pre-existing ``private``/``shared`` row of the same content.
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            must = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="hash", match=MatchValue(value=content_hash)),
                FieldCondition(key="metadata.scope", match=MatchValue(value=scope)),
            ]
            if visibility is not None:
                must.append(FieldCondition(key="metadata.visibility", match=MatchValue(value=visibility)))
            if scope == "project" and project_id:
                must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id)))
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=must),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            pt = points[0]
            payload = pt.payload or {}
            metadata = payload.get("metadata", {}) or {}
            return MemoryResponse(
                id=str(pt.id),
                memory=payload.get("data", ""),
                category=metadata.get("category"),
                scope=metadata.get("scope"),
                project_id=metadata.get("project_id"),
                tags=metadata.get("tags"),
                source="vector",
                created_at=payload.get("created_at"),
                domain=metadata.get("domain"),
                observation_type=metadata.get("observation_type"),
                concepts=metadata.get("concepts"),
                source_type=metadata.get("source_type"),
                related_memory_ids=metadata.get("related_memory_ids"),
                confidence=metadata.get("confidence"),
                expires_at=metadata.get("expires_at"),
                derived_from=metadata.get("derived_from"),
                epistemic_level=metadata.get("epistemic_level"),
                memory_kind=metadata.get("memory_kind"),
                source_ref=metadata.get("source_ref"),
                visibility=metadata.get("visibility"),
                owner_user_id=metadata.get("owner_user_id"),
                title=metadata.get("title"),
                token_estimate=metadata.get("token_estimate"),
            )
        except Exception as e:
            logger.warning(f"Content-hash dedup lookup failed (non-fatal): {e}")
            return None

    def _revive_if_tombstoned(self, memory_id: str) -> bool:
        """Resurrect a dream-tombstoned dedup survivor (audit 27 #5).

        ``_find_by_content_hash`` matches tombstoned rows too — before this
        existed, re-storing a fact the dreaming sweep had tombstoned was
        silently swallowed by the dedup short-circuit and the fact stayed
        excluded from recall forever (search filters
        ``metadata.dream_tombstoned=true``). Revival semantics:
        re-derivation is fresh evidence the fact is live, so the tombstone
        flag is cleared and the row becomes recallable again.

        The clear is an atomic nested-key merge (``set_payload`` with
        ``key="metadata"``) rather than a read-merge-write of the whole
        metadata dict, so it can't race a concurrent metadata patch into
        resurrecting stale fields.

        Returns True if a tombstone was actually cleared. Best-effort:
        failures log and return False — a lost revival must never block
        the write path.
        """
        try:
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            points = client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return False
            meta = (points[0].payload or {}).get("metadata") or {}
            if not meta.get("dream_tombstoned"):
                return False
            client.set_payload(
                collection_name=collection,
                payload={
                    "dream_tombstoned": False,
                    "dream_revived_at": datetime.now(timezone.utc).isoformat(),
                },
                points=[memory_id],
                key="metadata",
            )
            logger.info(f"Revived dream-tombstoned memory {memory_id} on re-derivation")
            return True
        except Exception as e:
            logger.warning(f"Tombstone revival failed for {memory_id} (non-fatal): {e}")
            return False

    def _bump_times_derived(self, memory_id: str, add: int = 1) -> None:
        """Increment ``metadata.times_derived`` on a surviving memory.

        Called when a duplicate of ``memory_id`` was detected and dropped —
        the reinforcement is counted on the survivor instead of discarded.
        Read-merge-write of the nested metadata dict (Qdrant's ``set_payload``
        replaces top-level keys wholesale, so the merge is mandatory —
        mirrors ``extensions/dreaming/consolidate._merged_metadata``).

        Best-effort: losing a reinforcement tick must never block a write or
        a dedup pass, so every failure is swallowed with a warning.
        """
        if add <= 0:
            return
        try:
            client = self._memory.vector_store.client
            collection = settings.qdrant_collection
            points = client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return
            meta = dict((points[0].payload or {}).get("metadata") or {})
            meta["times_derived"] = _times_derived_from_metadata(meta) + add
            client.set_payload(
                collection_name=collection,
                payload={"metadata": meta},
                points=[memory_id],
            )
        except Exception as e:
            logger.warning(f"times_derived bump failed for {memory_id} (non-fatal): {e}")

    def store_raw_batch(
        self,
        items: list[dict],
    ) -> list[MemoryResponse]:
        """Store multiple pre-categorized facts (memory-model v2).

        Each item is a dict matching RawMemoryRequest's shape. Reuses
        ``store_raw`` per item so dedup, validation, and graph ingestion
        stay consistent. Single network round-trip from the caller.

        Args:
            items: list of dicts with at least content/user_id/category, plus
                any v2 optional fields. Each item is independent; one bad item
                won't block the others (we collect errors but continue).

        Returns:
            Flattened list of MemoryResponse objects across all successful items.
        """
        results: list[MemoryResponse] = []
        for idx, item in enumerate(items):
            try:
                # Convert ISO string -> datetime if needed (came in via JSON)
                expires_at = item.get("expires_at")
                if expires_at and isinstance(expires_at, str):
                    try:
                        from datetime import datetime as _dt
                        expires_at = _dt.fromisoformat(expires_at.replace("Z", "+00:00"))
                    except ValueError:
                        expires_at = None
                stored = self.store_raw(
                    content=item["content"],
                    user_id=item["user_id"],
                    category=item["category"],
                    scope=item.get("scope", "global"),
                    project_id=item.get("project_id"),
                    tags=item.get("tags"),
                    agent_id=item.get("agent_id"),
                    run_id=item.get("run_id"),
                    domain=item.get("domain"),
                    observation_type=item.get("observation_type"),
                    concepts=item.get("concepts"),
                    source_type=item.get("source_type"),
                    related_memory_ids=item.get("related_memory_ids"),
                    confidence=item.get("confidence"),
                    expires_at=expires_at,
                    derived_from=item.get("derived_from"),
                    epistemic_level=item.get("epistemic_level"),
                    visibility=item.get("visibility"),
                    memory_kind=item.get("memory_kind"),
                    source_ref=item.get("source_ref"),
                )
                results.extend(stored)
            except Exception as e:
                logger.warning(f"Batch item {idx} failed (continuing): {e}")
        logger.info(f"store_raw_batch: stored {len(results)} memories from {len(items)} items")
        return results

    def expire_old_memories(self, batch_size: int = 100) -> dict:
        """Delete memories whose expires_at is in the past (memory-model v2 cron).

        Qdrant doesn't have a direct datetime range filter on string fields,
        so we scroll the whole collection and filter in Python by parsing
        each memory's `expires_at` to an aware UTC datetime. For larger
        deployments we'd switch to a numeric epoch field.

        Returns:
            Dict with deleted_count and per-user breakdown.
        """
        # Ensure mem0/Qdrant client is initialized — the nightly cron can
        # fire on a cold-started worker before any request has touched
        # _memory, which would otherwise raise AttributeError.
        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection
        now_dt = datetime.now(timezone.utc)

        deleted_count = 0
        per_user: dict[str, int] = {}
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                payload = pt.payload or {}
                metadata = payload.get("metadata", {}) or {}
                expires_at = metadata.get("expires_at")
                if not expires_at:
                    continue
                expires_dt = _parse_expires_at(expires_at)
                if expires_dt is None:
                    # Malformed timestamp — log and skip rather than risk
                    # deleting on an unparseable value.
                    logger.warning(
                        f"Memory {pt.id} has unparseable expires_at={expires_at!r}; skipping"
                    )
                    continue
                if expires_dt >= now_dt:
                    continue  # not yet expired
                try:
                    self._delete_qdrant_memory_with_graph_cleanup(str(pt.id), payload)
                    uid = payload.get("user_id", "unknown")
                    per_user[uid] = per_user.get(uid, 0) + 1
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to expire memory {pt.id}: {e}")
            if next_offset is None:
                break
            offset = next_offset

        if deleted_count:
            logger.info(f"expire_old_memories: deleted {deleted_count} expired memories")
        return {"deleted_count": deleted_count, "per_user": per_user}

    def _batch_store_facts(
        self,
        facts: list[tuple[str, str]],
        user_id: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        source: str = "conversation",
        source_type: str | None = None,
        memory_kind: str | None = None,
        source_ref: dict | None = None,
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
        fact_meta: list[tuple[str, str, str | None]] = []  # (category, scope, project_id) per fact

        for category, content in facts:
            scope = default_scope_for_category(category)
            fact_project_id = project_id

            if fact_project_id and category not in GLOBAL_CATEGORIES:
                scope = MemoryScope.PROJECT

            if category in PROJECT_CATEGORIES and not fact_project_id:
                inferred = _infer_project_id(content)
                if inferred:
                    fact_project_id = inferred
                    scope = MemoryScope.PROJECT
                    logger.info(
                        f"Inferred project_id='{inferred}' from content for category '{category}'"
                    )
                else:
                    scope = MemoryScope.GLOBAL
                    logger.warning(
                        f"Category '{category}' requires project_id but none provided and could not be inferred. "
                        f"Content snippet: '{content[:80]}'. Storing as global scope."
                    )

            mid = str(uuid.uuid4())
            payload = {
                "data": content,
                "hash": content_hash(content),
                "created_at": now_iso,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": {
                    "scope": scope.value if isinstance(scope, MemoryScope) else scope,
                    "category": category,
                    "project_id": fact_project_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "source": source,
                    # Extraction stores directly-stated facts → epistemically
                    # "explicit" (A1). Derived levels are stamped by their
                    # authors (dreaming reflection/merge), never here.
                    "epistemic_level": "explicit",
                    # Retrieval economics (C1): index-row fields (heuristic, no LLM)
                    "title": distill_title(content),
                    "token_estimate": stamp_tokens(content),
                    **({"source_type": source_type} if source_type is not None else {}),
                    **({"memory_kind": memory_kind} if memory_kind is not None else {}),
                    **({"source_ref": source_ref} if source_ref else {}),
                },
            }

            texts.append(content)
            memory_ids.append(mid)
            payloads.append(payload)
            fact_meta.append((category, scope.value if isinstance(scope, MemoryScope) else scope, fact_project_id))

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
        for mid, content, (category, scope_val, fact_pid) in zip(memory_ids, texts, fact_meta):
            responses.append(
                MemoryResponse(
                    id=mid,
                    memory=content,
                    category=category,
                    scope=scope_val,
                    project_id=fact_pid,
                    source="vector",
                    created_at=now_iso,
                    source_type=source_type,
                    epistemic_level="explicit",
                    memory_kind=memory_kind,
                    source_ref=source_ref,
                    title=distill_title(content),
                    token_estimate=stamp_tokens(content),
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
        # Memory-model v2 filters
        domain: str | None = None,
        observation_type: str | None = None,
        concepts: list[str] | None = None,
        memory_kind: str | None = None,
        # Multi-user pool selection
        visibility: str | None = None,
        include_shared: bool = True,
    ) -> list[MemoryResponse]:
        """Semantic search across memories with scope/category filters.

        Multi-user model: returns the union of two pools, dedup'd by id:

        - **Personal pool**: memories owned by `user_id` (regardless of
          visibility — you can always read what you wrote).
        - **Shared pool**: memories with `metadata.visibility=shared`, written
          by anyone in this Neuralscape instance.

        Use ``visibility="private"`` to scope to the personal pool only;
        ``visibility="shared"`` to scope to the shared pool only;
        ``include_shared=False`` to skip the shared pool entirely.

        When project_id is provided, both pools search project+global memories
        for that project (existing dual-scope merge preserved per pool).

        Returns:
            List of matching memory responses sorted by score.
        """
        m = self._get_memory()

        # Embed the query ONCE and reuse the vector across every pool/scope
        # below. Both pools query Qdrant directly with this precomputed vector
        # (see _search_personal_pool / _search_shared_pool). Previously each
        # Memory.search + shared-pool call re-embedded the same query — 4-5
        # embeds per recall — and the embed round-trip dominates read latency.
        query_embedding = m.embedding_model.embed(query, memory_action="search")

        vector_responses: list[MemoryResponse] = []

        # ── Personal pool: the caller's own memories (any visibility) ──
        # Skip when caller restricted to shared-only. Dedup + sort + limit
        # happen once across both pools below.
        want_personal = visibility != MemoryVisibility.SHARED.value
        if want_personal and user_id:
            # Failure isolation: a transient Qdrant error in the personal
            # pool must not abort the whole recall — degrade to shared/graph
            # results instead, matching the shared-pool/graph paths below.
            try:
                if project_id and not scope:
                    # Dual-scope: this user's project-scoped + global memories.
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=project_id, categories=categories, scope=None,
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                        )
                    )
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=None, categories=categories, scope="global",
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                        )
                    )
                else:
                    vector_responses.extend(
                        self._search_personal_pool(
                            m=m, user_id=user_id, query_embedding=query_embedding,
                            project_id=project_id, categories=categories, scope=scope,
                            domain=domain, observation_type=observation_type,
                            concepts=concepts, limit=limit, query=query,
                        )
                    )
            except Exception as e:
                logger.warning(f"Personal-pool search failed (non-critical): {e}")

        # ── Shared pool: direct Qdrant, no user_id namespace ───────
        # Bypass mem0's wrapper because shared memories span multiple
        # writers; we need a search that returns hits regardless of
        # which user_id wrote them. Only memories with explicit
        # `metadata.visibility=shared` are returned (legacy memories
        # without that field stay de-facto private until migration).
        #
        # Dual-scope merge: when `project_id` is set and `scope` is
        # omitted, mirror the personal-pool's project+global merge —
        # otherwise a project-context search would miss global shared
        # memories that should still be visible (the graph read-set
        # already covers both via `_get_group_ids`). The downstream
        # dedup at line ~1094 collapses any overlap.
        # An explicit `visibility="shared"` selects the shared pool even when
        # `include_shared=False` — otherwise the vector path would suppress the
        # shared pool while the graph path (which keys off `visibility==shared`)
        # still returns it, yielding inconsistent/partial results.
        want_shared = visibility == MemoryVisibility.SHARED.value or (
            include_shared and visibility != MemoryVisibility.PRIVATE.value
        )
        if want_shared:
            try:
                if project_id and not scope:
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=project_id,
                            categories=categories,
                            scope=None,
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            query_embedding=query_embedding,
                        )
                    )
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=None,
                            categories=categories,
                            scope="global",
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            query_embedding=query_embedding,
                        )
                    )
                else:
                    vector_responses.extend(
                        self._search_shared_pool(
                            m=m,
                            query=query,
                            project_id=project_id,
                            categories=categories,
                            scope=scope,
                            domain=domain,
                            observation_type=observation_type,
                            concepts=concepts,
                            limit=limit,
                            query_embedding=query_embedding,
                        )
                    )
            except Exception as e:
                logger.warning(f"Shared-pool search failed (non-critical): {e}")

        # ── Standard pool: authoritative dictator-written memories ──────
        # Always included when the tier is enabled (independent of
        # include_shared), because org standards are binding and must surface
        # in recall for everyone. Suppressed only when the caller explicitly
        # narrowed to a different single pool (visibility=private/shared).
        want_standard = settings.standards_enabled and visibility in (
            None,
            MemoryVisibility.STANDARD.value,
        )
        if want_standard:
            try:
                # Standards are ALWAYS written global-scope with no project_id
                # (store_raw forces this), so the standard pool must NOT inherit
                # the caller's scope/project_id — doing so returns zero standards
                # for a project-scoped recall and breaks the everyone-reads-
                # standards guarantee. Query the pool unscoped; it's already
                # filtered to visibility=standard.
                vector_responses.extend(
                    self._search_standard_pool(
                        m=m,
                        query=query,
                        project_id=None,
                        categories=categories,
                        scope=None,
                        domain=domain,
                        observation_type=observation_type,
                        concepts=concepts,
                        limit=limit,
                        query_embedding=query_embedding,
                    )
                )
            except Exception as e:
                logger.warning(f"Standard-pool search failed (non-critical): {e}")

        # Dedup across the pools (caller's own shared writes match both).
        seen_ids: set[str] = set()
        deduped: list[MemoryResponse] = []
        for r in vector_responses:
            if r.id and r.id not in seen_ids:
                seen_ids.add(r.id)
                deduped.append(r)
        # A4 salience tie-breaker (opt-in; k=0 default returns untouched).
        deduped = _salience_tiebreak(deduped)
        deduped.sort(key=lambda r: r.score or 0.0, reverse=True)
        vector_responses = deduped[:limit]

        # Also query the knowledge graph and merge edge facts.
        # Multi-user: when caller restricted the visibility, restrict the
        # graph search's group_ids to match — otherwise the graph would
        # walk the full read-set (caller's private + shared) and we'd
        # have to retroactively filter, which is unreliable for graph
        # rows whose enriched visibility ends up as None.
        graph_responses: list[MemoryResponse] = []
        try:
            graph_results = self._search_graph_for_visibility(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=limit,
                visibility=visibility,
                include_shared=include_shared,
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

        # Memory-model v2: enrich graph results with v2 metadata from their
        # nearest source memory, then apply the same v2 filters to graph rows.
        # Graphiti's edge schema doesn't carry v2 fields natively — we recover
        # them by semantic match against the Qdrant store.
        v2_filter_active = bool(domain or observation_type or concepts)
        if graph_responses and v2_filter_active:
            graph_responses = self._enrich_and_filter_graph(
                graph_responses,
                user_id=user_id,
                project_id=project_id,
                domain=domain,
                observation_type=observation_type,
                concepts=concepts,
            )
        elif graph_responses and memory_kind == "passage":
            # The passage filter below reads memory_kind off each row, which
            # for graph rows only exists via source-memory enrichment.
            graph_responses = self._enrich_graph_with_v2(
                graph_responses, user_id=user_id, project_id=project_id
            )

        # Multi-user model: post-filter graph rows by enriched visibility.
        # The Graphiti search above already scopes by group_ids, so most
        # rows arrive in the right pool. This pass mops up the edge case
        # where enrichment couldn't find a source memory: when the caller
        # asked for `private`, an unenriched row (visibility=None) is
        # treated as private — it came from the private group_id range
        # we just scoped to. When they asked for `shared`, an unenriched
        # row could only have come from a shared group_id, so we keep it.
        if visibility and graph_responses:
            graph_responses = [
                r for r in graph_responses
                if r.visibility == visibility or r.visibility is None
            ]

        # Deduplicate and enforce caller's limit — ranked vector hits keep
        # priority; graph rows are appended after, capped within the limit.
        combined = self._deduplicate_responses(
            vector_responses, graph_responses, limit=limit
        )

        # memory_kind filter (data-layer connectors). Legacy memories have no
        # memory_kind, so a "fact" filter treats null as fact (back-compat);
        # "passage" matches only explicitly-tagged passages.
        if memory_kind == "fact":
            combined = [r for r in combined if (r.memory_kind or "fact") == "fact"]
        elif memory_kind == "passage":
            combined = [r for r in combined if r.memory_kind == "passage"]

        results = combined[:limit]

        # Dreaming: fire-and-forget recall trace (reinforcement signal for
        # the dream sweep's promotion/retention scoring). Runs on a daemon
        # thread inside log_recall — never blocks or fails the read.
        try:
            from extensions.dreaming.traces import log_recall

            log_recall([r.id for r in results if r.id], query)
        except Exception:
            pass

        return results

    # Minimum vector similarity for graph→source enrichment to be trusted.
    # Below this, the "source" is just the nearest unrelated memory and we
    # leave v2 fields as None rather than propagating wrong metadata.
    _GRAPH_ENRICH_THRESHOLD: float = 0.7

    def _enrich_graph_with_v2(
        self,
        graph_responses: list[MemoryResponse],
        user_id: str,
        project_id: str | None,
    ) -> list[MemoryResponse]:
        """For each graph edge, find its nearest Qdrant source memory and
        copy that source's v2 metadata onto the graph response — only when
        the similarity score clears _GRAPH_ENRICH_THRESHOLD.

        Memory-model v2 augmentation: Graphiti edges don't carry domain /
        observation_type / concepts natively, but they're derived from
        source memories that do. We do a top-1 vector match per edge to
        recover those fields.

        Audit 27 #7: this used to embed + query Qdrant once per edge,
        sequentially — 10 edges meant 10 Gemini round-trips + 10 Qdrant
        queries (3-12s of the measured hybrid-search latency). All edge
        texts now go through ONE ``embed_batch`` call and ONE Qdrant
        ``query_batch_points`` round trip. Any failure leaves the rows
        un-enriched (never breaks the read).

        Multi-user model: enrichment can use either the caller's private
        memories OR shared-pool memories (a graph edge for a shared fact
        should pick up the shared source's metadata). Restricts to the
        active project_id when supplied so v2 filter parity holds.
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
            QueryRequest,
        )

        enrichable = [(i, r.memory) for i, r in enumerate(graph_responses) if r.memory]
        if not enrichable:
            return graph_responses

        try:
            m = self._get_memory()
            client = m.vector_store.client

            texts = [text for _, text in enrichable]
            embed_batch = getattr(m.embedding_model, "embed_batch", None)
            if callable(embed_batch):
                embeddings = embed_batch(texts, memory_action="search")
            else:  # embedder without a batch API — degrade to per-text embeds
                embeddings = [
                    m.embedding_model.embed(t, memory_action="search") for t in texts
                ]

            # Enrichment source = OR of per-pool sub-filters (Qdrant `should`
            # accepts nested Filters). Personal + shared are constrained to the
            # active project; the authoritative STANDARD pool is always global
            # (no project_id), so it must NOT carry the project constraint —
            # otherwise standard-origin graph edges never match their source
            # and lose their v2 metadata / get dropped by v2 filters. The
            # filter is identical for every edge, so it is built ONCE.
            proj = (
                [FieldCondition(key="metadata.project_id", match=MatchValue(value=project_id))]
                if project_id else []
            )
            should_filters: list = []
            if user_id:
                should_filters.append(
                    Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))] + proj)
                )
            should_filters.append(
                Filter(must=[FieldCondition(
                    key="metadata.visibility",
                    match=MatchValue(value=MemoryVisibility.SHARED.value),
                )] + proj)
            )
            if settings.standards_enabled:
                should_filters.append(
                    Filter(must=[FieldCondition(
                        key="metadata.visibility",
                        match=MatchValue(value=MemoryVisibility.STANDARD.value),
                    )])
                )
            qf = Filter(should=should_filters)

            requests = [
                QueryRequest(query=emb, filter=qf, limit=1, with_payload=True)
                for emb in embeddings
            ]
            batch_results = client.query_batch_points(
                collection_name=settings.qdrant_collection,
                requests=requests,
            )
        except Exception as e:
            logger.debug(f"Graph enrichment skipped (batch failed): {e}")
            return graph_responses

        for (idx, _), result in zip(enrichable, batch_results):
            resp = graph_responses[idx]
            try:
                hits = getattr(result, "points", result) or []
                if not hits:
                    continue
                hit = hits[0]
                score = getattr(hit, "score", None)
                if score is None and isinstance(hit, dict):
                    score = hit.get("score")
                if score is not None and score < self._GRAPH_ENRICH_THRESHOLD:
                    continue  # too weak a match to trust the metadata link

                payload = getattr(hit, "payload", None)
                if payload is None and isinstance(hit, dict):
                    payload = hit.get("payload", {})
                payload = payload or {}
                src_metadata = payload.get("metadata", {}) or {}
                if isinstance(src_metadata.get("metadata"), dict):
                    src_metadata = src_metadata["metadata"]

                # Copy v2 fields when source has them and graph response doesn't
                if resp.domain is None:
                    resp.domain = src_metadata.get("domain")
                if resp.observation_type is None:
                    resp.observation_type = src_metadata.get("observation_type")
                if resp.concepts is None:
                    resp.concepts = src_metadata.get("concepts")
                if resp.source_type is None:
                    resp.source_type = src_metadata.get("source_type")
                if resp.epistemic_level is None:
                    resp.epistemic_level = src_metadata.get("epistemic_level")
                if resp.confidence is None:
                    resp.confidence = src_metadata.get("confidence")
                if resp.expires_at is None:
                    resp.expires_at = src_metadata.get("expires_at")
                if resp.memory_kind is None:
                    resp.memory_kind = src_metadata.get("memory_kind")
                if resp.source_ref is None:
                    resp.source_ref = src_metadata.get("source_ref")
                if resp.category is None:
                    resp.category = src_metadata.get("category")
                if resp.scope is None:
                    resp.scope = src_metadata.get("scope")
                if resp.project_id is None:
                    resp.project_id = src_metadata.get("project_id")
                if resp.visibility is None:
                    resp.visibility = src_metadata.get("visibility")
                if resp.owner_user_id is None:
                    resp.owner_user_id = src_metadata.get("owner_user_id")
            except Exception as e:
                logger.debug(f"Graph enrichment skipped for {resp.id}: {e}")
        return graph_responses

    def _enrich_and_filter_graph(
        self,
        graph_responses: list[MemoryResponse],
        user_id: str,
        project_id: str | None,
        domain: str | None,
        observation_type: str | None,
        concepts: list[str] | None,
    ) -> list[MemoryResponse]:
        """Enrich graph rows with v2 metadata, then drop rows that don't match
        the supplied filter. Used when the caller passes domain/observation_type/
        concepts in SearchMemoryRequest.
        """
        enriched = self._enrich_graph_with_v2(
            graph_responses, user_id=user_id, project_id=project_id
        )
        out: list[MemoryResponse] = []
        for resp in enriched:
            if domain and resp.domain != domain:
                continue
            if observation_type and resp.observation_type != observation_type:
                continue
            if concepts:
                resp_concepts = set(resp.concepts or [])
                if not (resp_concepts & set(concepts)):
                    continue
            out.append(resp)
        return out

    def _search_graph_for_visibility(
        self,
        query: str,
        user_id: str,
        project_id: str | None,
        limit: int,
        visibility: str | None,
        include_shared: bool,
    ) -> dict:
        """search_graph with multi-user visibility scoping.

        When the caller restricts visibility to one pool, narrow the
        Graphiti `group_ids` to that pool's namespace. This is
        load-bearing for cross-user isolation: if we walked the full
        group_id set and then filtered by enriched visibility, an
        unenriched row from the shared pool could slip into a
        private-only response.
        """
        def _standard_groups() -> list[str]:
            if not settings.standards_enabled:
                return []
            return ["standard"] + ([f"standard--project--{project_id}"] if project_id else [])

        if visibility == MemoryVisibility.PRIVATE.value:
            group_ids = [f"user--{user_id}"]
            if project_id:
                group_ids.append(f"user--{user_id}--project--{project_id}")
        elif visibility == MemoryVisibility.STANDARD.value:
            group_ids = _standard_groups()
        elif visibility == MemoryVisibility.SHARED.value:
            group_ids = ["shared"]
            if project_id:
                group_ids.append(f"shared--project--{project_id}")
        elif not include_shared:
            # No explicit visibility, but caller opted out of shared pool.
            # Standards remain in-scope — they are binding and independent of
            # the shared opt-out.
            group_ids = [f"user--{user_id}"]
            if project_id:
                group_ids.append(f"user--{user_id}--project--{project_id}")
            group_ids += _standard_groups()
        else:
            # Default: full read-set (caller's private + shared + standard).
            group_ids = _get_group_ids(user_id, project_id)

        return self._do_graph_search(query=query, group_ids=group_ids, limit=limit)

    def _do_graph_search(
        self,
        query: str,
        group_ids: list[str],
        limit: int,
        search_config: dict | None = None,
    ) -> dict:
        """Internal: run a graph search across the given group_ids."""
        g = self._get_graphiti()
        if g is None:
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

        from graphiti_core.search.search_config import SearchConfig
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        # Audit 27 #10: EDGE_HYBRID_SEARCH_RRF is a module-level singleton
        # shared across every thread — mutating its `.limit` in place let a
        # concurrent delete (limit=5) clamp a live search's graph fan-out.
        # Always work on a deep copy.
        if search_config:
            try:
                config = SearchConfig(**search_config)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid search_config, falling back to default: {e}")
                config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        else:
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = limit

        try:
            results = self._run_on_bridge(
                g.search_(
                    query=query,
                    config=config,
                    group_ids=group_ids,
                    # Audit 27 #3: bi-temporal liveness — edges the dreaming
                    # sweep stamped invalid_at/expired_at (INVALIDATE / PRUNE /
                    # MERGE, see extensions/dreaming/graph_patcher.py) must not
                    # surface as live facts and eat top-k slots.
                    search_filter=_live_edges_filter(),
                )
            )
            edges = [
                {"uuid": e.uuid, "name": e.name, "fact": e.fact}
                for e in results.edges
                # Belt-and-suspenders: some drivers/recipes skip the Cypher
                # filter constructor, so drop stamped edges here too.
                if not _edge_is_invalidated(e)
            ]
            nodes = [
                {"uuid": n.uuid, "name": n.name, "summary": n.summary}
                for n in results.nodes
            ]
            episodes = [
                {"uuid": ep.uuid, "name": ep.name, "content": ep.content}
                for ep in results.episodes
            ]
            communities = [
                {"uuid": c.uuid, "name": c.name} for c in results.communities
            ]
            # Enrich nodes/edges/communities with the back-references the
            # synthesizer set (memory_id, wiki_path). Best-effort; a
            # failed enrichment leaves the result as-is.
            self._enrich_graph_results(nodes, edges, communities)
            return {
                "edges": edges,
                "nodes": nodes,
                "episodes": episodes,
                "communities": communities,
            }
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"edges": [], "nodes": [], "episodes": [], "communities": []}

    def _enrich_graph_results(
        self,
        nodes: list[dict],
        edges: list[dict],
        communities: list[dict],
    ) -> None:
        """Annotate graph search results with ``memory_id`` + ``wiki_path``.

        Both fields are added by the wiki synthesizer's Cypher patchers
        (``attach_memory_id`` and ``patch_wiki_path``) as top-level Neo4j
        properties, but Graphiti's ORM doesn't rehydrate them. We do one
        extra Cypher round-trip per result set to fetch the values by
        UUID, then mutate the dicts in place. Failure logs and leaves
        the original dicts unchanged.
        """
        all_uuids: list[str] = []
        for collection in (nodes, edges, communities):
            for item in collection:
                u = item.get("uuid")
                if u:
                    all_uuids.append(u)
        if not all_uuids:
            return
        if self._graphiti is None or self._bridge is None:
            return
        cypher = """
        MATCH (n)
        WHERE n.uuid IN $uuids
        RETURN n.uuid AS uuid,
               n.memory_id AS memory_id,
               n.wiki_path AS wiki_path
        """

        async def _run():
            async with self._graphiti.driver.session() as session:
                result = await session.run(cypher, uuids=all_uuids)
                return await result.data()

        try:
            records = self._run_on_bridge(_run(), timeout=10.0) or []
        except Exception:
            logger.warning("graph result enrichment failed (non-critical)", exc_info=True)
            return
        by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}
        for collection in (nodes, edges, communities):
            for item in collection:
                rec = by_uuid.get(item.get("uuid"))
                if not rec:
                    continue
                if rec.get("memory_id"):
                    item["memory_id"] = rec["memory_id"]
                if rec.get("wiki_path"):
                    item["wiki_path"] = rec["wiki_path"]

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

        # Multi-user: search across the caller's private namespace + the
        # shared pool, plus project-scoped variants. Replaces the prior
        # cross-user `"global"`/`"project--..."` group_ids.
        group_ids = _get_group_ids(user_id, project_id)

        # Audit 27 #10: never mutate the shared module-level recipe singleton
        # — deep-copy before setting the per-call limit.
        if search_config:
            try:
                config = SearchConfig(**search_config)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid search_config, falling back to default: {e}")
                config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        else:
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)

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

    # ──────────────────────────────────────────────
    # Retrieval economics (C1 batch get + C2 timeline)
    # ──────────────────────────────────────────────

    @staticmethod
    def _payload_readable_by(payload: dict, caller_user_id: str | None) -> bool:
        """Whether ``caller_user_id`` may read a Qdrant point's payload.

        Mirrors the pool rules of ``search()``: you can always read what you
        own (top-level ``user_id`` namespace or ``metadata.owner_user_id``);
        everyone reads ``shared``; everyone reads ``standard`` when the tier
        is enabled. Legacy rows without a visibility field are de-facto
        private to their writer.
        """
        if caller_user_id and payload.get("user_id") == caller_user_id:
            return True
        meta = payload.get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        if caller_user_id and meta.get("owner_user_id") == caller_user_id:
            return True
        vis = meta.get("visibility")
        if vis == MemoryVisibility.SHARED.value:
            return True
        if vis == MemoryVisibility.STANDARD.value and settings.standards_enabled:
            return True
        return False

    def _point_to_response(self, point) -> MemoryResponse:
        """Convert a raw Qdrant point (retrieve/scroll result) to a MemoryResponse."""
        payload = getattr(point, "payload", None) or {}
        return self._mem_to_response(
            {
                "id": str(getattr(point, "id", "")),
                "memory": payload.get("data", ""),
                "metadata": payload.get("metadata", {}) or {},
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }
        )

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

    def keyword_search(
        self,
        user_id: str,
        terms: list[str],
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[MemoryResponse]:
        """Grep-style exact/keyword scan over caller-visible memories (C3).

        Case-insensitive substring match: a memory matches when its content
        contains ANY of ``terms``. This is the dialectic discipline for
        enumeration/counting questions — embeddings under-recall exhaustive
        "list every X" sets, so the ask path runs this exact pass *before*
        semantic search and dedups the union.

        Visibility mirrors search's pool union (caller's own rows at any
        visibility + shared + standard-when-enabled; ``project_id`` adds the
        project+global dual scope) and dream-tombstoned rows are excluded —
        the same rules as ``_timeline_filter``. Bounded scan: pages of
        ``_TIMELINE_FALLBACK_PAGE`` up to ``_TIMELINE_FALLBACK_CAP`` points,
        stopping early once ``limit`` matches are found. Results carry no
        score (exact matches aren't ranked).
        """
        lowered = [t.lower() for t in terms if t and t.strip()]
        if not lowered:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_memory().vector_store.client

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
        flt = Filter(
            must=must,
            must_not=[
                FieldCondition(
                    key="metadata.dream_tombstoned", match=MatchValue(value=True)
                )
            ],
        )

        matches: list[MemoryResponse] = []
        offset = None
        scanned = 0
        while scanned < self._TIMELINE_FALLBACK_CAP and len(matches) < limit:
            page, offset = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=flt,
                limit=self._TIMELINE_FALLBACK_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in page or []:
                content = ((getattr(pt, "payload", None) or {}).get("data") or "").lower()
                if any(t in content for t in lowered):
                    matches.append(self._point_to_response(pt))
                    if len(matches) >= limit:
                        break
            scanned += len(page or [])
            # `is None`, not truthiness — integer point-id offsets can be 0.
            if offset is None:
                break
        return matches

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

    # ──────────────────────────────────────────────
    # Authoritative standards + processes (dictator tier)
    # ──────────────────────────────────────────────

    # Hard safety ceiling on a full standards scroll — standards are authoritative
    # and must all be injected, but this bounds a pathological runaway.
    _STANDARD_SCROLL_MAX = 5000

    def _scroll_standard(self, must_extra: list, limit: int = 500) -> list:
        """Scroll ALL standard-tier points matching extra conditions (paginated).

        Returns raw Qdrant points (visibility=standard AND all `must_extra`).
        Pages through Qdrant so the authoritative set is returned in full (up to
        a safety ceiling), rather than silently truncating at a single page —
        binding directives must not be dropped. Empty on any error so callers
        degrade gracefully.

        Verbatim ``passage`` chunks are EXCLUDED: every scroll caller
        (session-start standards injection, process enumeration) wants distilled
        directives/definitions, not raw document chunks. When a dictator ingests
        a standards document its passages still live in the standard pool for
        semantic ``recall`` (via ``_search_standard_pool``) — they just don't
        flood the always-on standards block or a process bundle.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        m = self._get_memory()
        client = m.vector_store.client
        must = [
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=MemoryVisibility.STANDARD.value),
            )
        ] + must_extra
        must_not = [
            FieldCondition(key="metadata.memory_kind", match=MatchValue(value="passage"))
        ]
        try:
            scroll_filter = Filter(must=must, must_not=must_not)
            page_size = max(1, min(limit, 500))
            collected: list = []
            offset = None
            while len(collected) < self._STANDARD_SCROLL_MAX:
                points, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=scroll_filter,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                )
                collected.extend(points or [])
                if offset is None or not points:
                    break
            if len(collected) >= self._STANDARD_SCROLL_MAX:
                logger.warning(
                    "Standard scroll hit the %d-row safety ceiling; some standards may be omitted.",
                    self._STANDARD_SCROLL_MAX,
                )
            return collected
        except Exception as e:
            logger.warning(f"Standard-tier scroll failed (non-critical): {e}")
            return []

    def _get_standards(
        self, project_id: str | None = None, critical_only: bool = True
    ) -> list[MemoryResponse]:
        """Fetch authoritative standard-tier memories (dictator-written).

        Standards are org-wide by definition — always stored at global scope
        (see ``store_raw``) — so this returns the global standard pool,
        newest-first, regardless of ``project_id`` (kept for signature symmetry).
        NOT filtered by caller: standards are readable by everyone.

        ``critical_only`` (default True) returns ONLY the always-inject subset —
        standards tagged ``critical``/``always`` — so the always-on session-start
        block stays small and doesn't dump the whole corpus into every session.
        The rest of the standard pool still surfaces on demand, relevance-ranked,
        through ``recall``/``search``. Pass ``critical_only=False`` to retrieve the
        full set (admin/review). Empty when the tier is disabled.
        """
        if not settings.standards_enabled:
            return []
        from qdrant_client.models import FieldCondition, MatchAny, MatchValue

        must_extra = [FieldCondition(key="metadata.scope", match=MatchValue(value="global"))]
        if critical_only:
            must_extra.append(
                FieldCondition(
                    key="metadata.tags",
                    match=MatchAny(any=list(_ALWAYS_INJECT_TAGS)),
                )
            )
        raw = self._scroll_standard(must_extra)
        seen: set[str] = set()
        out: list[MemoryResponse] = []
        for hit in raw:
            hid = str(getattr(hit, "id", ""))
            if not hid or hid in seen:
                continue
            seen.add(hid)
            payload = getattr(hit, "payload", None) or {}
            out.append(
                self._mem_to_response(
                    {
                        "id": hid,
                        "memory": payload.get("data", ""),
                        "metadata": payload.get("metadata", {}),
                        "score": None,
                        "created_at": payload.get("created_at"),
                    }
                )
            )
        out.sort(key=lambda r: str(getattr(r, "created_at", "") or ""), reverse=True)
        return out

    @staticmethod
    def _tags_of(response: MemoryResponse) -> list[str]:
        return list(getattr(response, "tags", None) or [])

    @staticmethod
    def _slug_from_tags(tags: list[str]) -> str | None:
        for t in tags:
            if t.startswith("process:"):
                slug = t.split(":", 1)[1].strip()
                if slug:
                    return slug
        return None

    @staticmethod
    def _title_and_summary(content: str) -> tuple[str, str]:
        """First content line = title; the remainder (trimmed) = a short summary.

        The summary powers natural-language matching in the `/process` picker —
        an agent maps a user's free-text request to the right process by title +
        summary without needing the full bundle.
        """
        lines = [ln.strip() for ln in (content or "").strip().splitlines() if ln.strip()]
        if not lines:
            return "", ""
        title = lines[0][:200]
        summary = " ".join(lines[1:])[:400]
        return title, summary

    def list_processes(self, project_id: str | None = None) -> list[dict]:
        """Enumerate available dictator-authored processes for the picker.

        A process is a set of standard-tier memories sharing a
        ``process:<slug>`` tag; its definition memory also carries the
        ``process-def`` tag (title = first content line, the rest = summary).
        Returns ``[{"slug","title","description"}]`` sorted by slug so the
        `/process` skill can match a user's free-text request to a process.
        Empty when processes are disabled. Mirrors ``list_projects``.
        """
        if not settings.processes_enabled:
            return []
        from qdrant_client.models import FieldCondition, MatchValue

        # Standards are org-wide (global); project_id is accepted for API
        # symmetry but doesn't scope the process registry.
        must_extra = [FieldCondition(key="metadata.tags", match=MatchValue(value="process-def"))]
        raw = self._scroll_standard(must_extra)
        out: dict[str, dict] = {}
        for hit in raw:
            payload = getattr(hit, "payload", None) or {}
            meta = payload.get("metadata", {}) or {}
            slug = self._slug_from_tags(list(meta.get("tags") or []))
            if not slug:
                continue
            title, summary = self._title_and_summary(payload.get("data", "") or "")
            out.setdefault(slug, {"slug": slug, "title": title or slug, "description": summary})
        return [out[s] for s in sorted(out)]

    def get_process(self, slug: str, project_id: str | None = None) -> dict | None:
        """Return a full process bundle by slug, or None if unknown/disabled.

        Pulls EVERY standard-tier memory tagged ``process:<slug>`` and splits it:
          - ``definition``  — the ``process-def`` memory (title + overview),
          - ``steps``       — ``process-step:<NN>`` memories, ordered by index,
          - ``guidelines``  — all OTHER standards tagged for the process (rules,
            gates, tone/format constraints ingested for it).
        This is how a process "pulls in its standards" so the `/process` skill
        can inject them as an authoritative playbook. Emits a ``process_served``
        audit event.
        """
        if not settings.processes_enabled:
            return None
        slug = (slug or "").strip()
        if not slug or not _SLUG_RE.match(slug):
            return None
        from qdrant_client.models import FieldCondition, MatchValue

        must_extra = [FieldCondition(key="metadata.tags", match=MatchValue(value=f"process:{slug}"))]
        raw = self._scroll_standard(must_extra)
        definition = ""
        title = slug
        steps: list[tuple[str, str]] = []  # (step-tag, content)
        guidelines: list[str] = []
        for hit in raw:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("data", "") or ""
            tags = list((payload.get("metadata", {}) or {}).get("tags") or [])
            if "process-def" in tags:
                definition = content
                t, _ = self._title_and_summary(content)
                title = t or slug
            else:
                step_tag = next((t for t in tags if t.startswith("process-step:")), None)
                if step_tag:
                    steps.append((step_tag, content))
                elif content.strip():
                    # Any other standard tagged for this process is a guideline.
                    guidelines.append(content)
        if not definition and not steps and not guidelines:
            return None
        steps.sort(key=lambda st: st[0])
        _audit_log.info(
            "process_served",
            slug=slug,
            project_id=project_id,
            steps=len(steps),
            guidelines=len(guidelines),
        )
        return {
            "slug": slug,
            "title": title,
            "definition": definition,
            "steps": [c for _, c in steps],
            "guidelines": guidelines,
        }

    # ── Editing ─────────────────────────────────────────────
    #
    # Metadata keys a PATCH may touch inside payload["metadata"]. `scope` is
    # deliberately absent — it is always re-derived from category + project_id,
    # and `owner_user_id` is never editable.
    _PATCHABLE_META_KEYS = (
        "category", "project_id", "tags", "domain",
        "observation_type", "concepts", "confidence", "expires_at",
    )
    # Keys where an explicit null means "clear the field".
    _CLEARABLE_META_KEYS = frozenset(
        {"project_id", "tags", "domain", "observation_type", "concepts", "confidence", "expires_at"}
    )

    @staticmethod
    def _derive_scope(category: str | None, project_id: str | None) -> str:
        """Re-derive scope from the effective category + project_id (same rule as writes)."""
        if category in PROJECT_CATEGORIES:
            if not project_id:
                raise ValueError(
                    f"project_id is required for project category '{category}' "
                    "(set project_id in the same edit, or change the category)"
                )
            return "project"
        if category in GLOBAL_CATEGORIES:
            return "global"
        # flexible / adapter / null category: scope follows project_id
        return "project" if project_id else "global"

    def _apply_meta_changes(self, meta: dict, changes: dict) -> dict:
        """Merge PATCH changes into a copy of the nested metadata dict.

        Presence-keyed: only keys in ``changes`` are touched; a None value
        clears the key where legal. Category membership is checked at call
        time because adapters extend MEMORY_CATEGORIES at import.
        """
        new_meta = dict(meta)
        for key in self._PATCHABLE_META_KEYS:
            if key not in changes:
                continue
            value = changes[key]
            if value is None:
                if key not in self._CLEARABLE_META_KEYS:
                    raise ValueError(f"'{key}' cannot be cleared — provide a value or omit the field")
                new_meta.pop(key, None)
                continue
            if key == "category" and value not in MEMORY_CATEGORIES:
                raise ValueError(f"Invalid category: {value}")
            if key == "expires_at":
                value = value.isoformat() if hasattr(value, "isoformat") else str(value)
            new_meta[key] = value
        return new_meta

    def patch_memory(self, memory_id: str, caller_user_id: str | None, changes: dict) -> dict:
        """Partially update a memory across the vector store and knowledge graph.

        ``changes`` is presence-keyed (built from the request's
        ``model_fields_set``): an explicit None clears the field where legal,
        an absent key is untouched. Returns::

            {"memory": MemoryResponse, "graph_job": dict | None,
             "graph": "unchanged" | "reingest_pending" | "migration_pending"}

        The caller is responsible for enqueuing ``graph_job`` on the graph
        queue — Graphiti work is minutes-slow and must never run inline on a
        request thread. Graph impact:

        - tags/category/v2 fields: not stored in Neo4j → vector-only patch.
        - project_id/visibility: part of the graph group_id partition →
          old edges are soft-expired here (fast, no LLM) and the content is
          re-ingested into the new group by the returned graph_job. Node
          group_ids are never mutated in place — entity nodes are shared
          across memories.
        - content: re-embedded here; the graph_job re-ingests so Graphiti's
          contradiction detection expires stale facts.
        """
        m = self._get_memory()
        point = m.vector_store.get(vector_id=memory_id)
        if point is None:
            raise LookupError(f"Memory {memory_id} not found")
        payload = dict(getattr(point, "payload", None) or {})
        meta = payload.get("metadata") or {}
        # mem0 sometimes double-wraps metadata; unwrap before reading
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        meta = dict(meta)

        for key in ("content", "category", "visibility"):
            if key in changes and changes[key] is None:
                raise ValueError(f"'{key}' cannot be cleared — provide a value or omit the field")

        edits_content = "content" in changes and changes["content"] != payload.get("data", "")
        edits_visibility = "visibility" in changes
        _check_edit_permission(
            meta,
            payload.get("user_id", ""),
            caller_user_id,
            edits_content=edits_content,
            edits_visibility=edits_visibility,
        )

        # Passages are verbatim chunks of an ingested artifact — rewriting one
        # would silently diverge from the source document. Metadata edits are fine.
        if edits_content and meta.get("memory_kind") == "passage":
            raise ValueError(
                "Content edits are not allowed on 'passage' memories — they mirror "
                "an ingested artifact. Re-ingest the corrected source instead."
            )

        owner = meta.get("owner_user_id") or payload.get("user_id", "")
        old_visibility = meta.get("visibility") or MemoryVisibility.PRIVATE.value
        old_group = _build_group_id(old_visibility, owner, meta.get("project_id"))

        new_meta = self._apply_meta_changes(meta, changes)

        if edits_visibility:
            new_visibility = normalize_visibility(changes["visibility"])
            if new_visibility == MemoryVisibility.STANDARD.value:
                # Mirror store_raw's standard-tier gate: dictator-only, forced
                # global scope (standards are org-wide by definition).
                if not settings.standards_enabled:
                    raise PermissionError(
                        "The 'standard' visibility tier is disabled (set STANDARDS_ENABLED=true)."
                    )
                if not settings.is_dictator(caller_user_id):
                    raise PermissionError(
                        f"User {caller_user_id!r} is not authorized to write 'standard'-tier memories."
                    )
                new_meta.pop("project_id", None)
            new_meta["visibility"] = new_visibility
        else:
            new_visibility = normalize_visibility(old_visibility) or MemoryVisibility.PRIVATE.value

        new_meta["scope"] = self._derive_scope(new_meta.get("category"), new_meta.get("project_id"))
        # store_raw always writes the project_id key (None when global) — keep that shape.
        new_meta["project_id"] = new_meta.get("project_id")

        # Retrieval economics (C1): a content edit invalidates the write-time
        # title/token_estimate — refresh them from the new content.
        if edits_content:
            new_meta["title"] = distill_title(changes["content"])
            new_meta["token_estimate"] = stamp_tokens(changes["content"])

        now_iso = datetime.now(timezone.utc).isoformat()
        if edits_content:
            # Full mem0 update: re-embed + BM25 refresh + history. Passing the
            # merged nested metadata is load-bearing — mem0's _update_memory
            # rebuilds the ENTIRE payload from this kwarg, so omitting it (the
            # old update_memory bug) wipes category/scope/tags/visibility/owner.
            m.update(memory_id, changes["content"], metadata={"metadata": new_meta})
        else:
            # Metadata-only: direct payload patch. set_payload merges at top
            # level, so data/hash/created_at/user_id are preserved and the
            # dense + BM25 vectors stay valid (content unchanged).
            m.vector_store.update(memory_id, payload={"metadata": new_meta, "updated_at": now_iso})

        new_content = changes["content"] if edits_content else payload.get("data", "")
        new_group = _build_group_id(new_visibility, owner, new_meta.get("project_id"))

        graph_job = None
        graph_status = "unchanged"
        if new_group != old_group:
            # Partition migration: soft-expire the memory's edges in the old
            # group (fast — hybrid search + edge saves, no LLM), then the
            # caller re-ingests into the new group via the graph queue.
            try:
                self._expire_graph_edges_for_memory(
                    {"memory": payload.get("data", ""), "metadata": meta, "user_id": owner}
                )
            except Exception as e:
                logger.warning(f"Graph edge expiration failed for {memory_id} (non-critical): {e}")
            graph_status = "migration_pending"
        elif edits_content:
            graph_status = "reingest_pending"
        if graph_status != "unchanged":
            graph_job = {
                "memory_id": memory_id,
                "content": new_content,
                "user_id": owner,
                "project_id": new_meta.get("project_id"),
                "visibility": new_visibility,
                "source_ref": meta.get("source_ref"),
            }

        _audit_log.info(
            "memory_patched",
            memory_id=memory_id,
            caller=caller_user_id,
            fields=sorted(changes.keys()),
            graph=graph_status,
        )
        return {"memory": self.get_memory(memory_id), "graph_job": graph_job, "graph": graph_status}

    def retag_memories(
        self,
        caller_user_id: str | None,
        filters: dict,
        ops: dict,
        dry_run: bool = False,
    ) -> dict:
        """Bulk-apply metadata operations to memories matching a filter set.

        ``filters``: scope / category / project_id / visibility / tags_contains
        (AND semantics). ``ops`` is presence-keyed: add_tags, remove_tags,
        set_category, set_project_id (explicit None clears the project).

        Visibility and content are deliberately NOT bulk-editable. Other
        users' private memories never enter the candidate set (the scroll
        filter restricts to shared/standard pools + the caller's own rows),
        so counts can't leak their existence. Per-row permission and
        category-matrix violations are skipped and counted, not fatal.

        Returns ``{matched, updated, skipped_forbidden, skipped_invalid,
        graph_jobs, dry_run}`` — the caller enqueues ``graph_jobs`` (produced
        when a project change moves a memory between graph groups).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if ops.get("set_category") and ops["set_category"] not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category: {ops['set_category']}")

        m = self._get_memory()
        client = m.vector_store.client
        collection = settings.qdrant_collection

        must = []
        if filters.get("scope"):
            must.append(FieldCondition(key="metadata.scope", match=MatchValue(value=filters["scope"])))
        if filters.get("category"):
            must.append(FieldCondition(key="metadata.category", match=MatchValue(value=filters["category"])))
        if filters.get("project_id"):
            must.append(FieldCondition(key="metadata.project_id", match=MatchValue(value=filters["project_id"])))
        if filters.get("visibility"):
            must.append(FieldCondition(key="metadata.visibility", match=MatchValue(value=filters["visibility"])))
        for tag in filters.get("tags_contains") or []:
            must.append(FieldCondition(key="metadata.tags", match=MatchValue(value=tag)))
        # Service-side backstop for the request-boundary sweep guard: worker and
        # MCP paths hand this raw dicts, and falsey filter values ("" / []) are
        # skipped above — without this check they'd select every candidate row.
        if not must:
            raise ValueError(
                "At least one effective filter is required — refusing an unfiltered retag sweep"
            )
        # Candidate set = shared OR standard OR the caller's own rows. This keeps
        # other users' PRIVATE memories out entirely (no permission-skip count
        # leakage; legacy null-visibility rows are covered by the user_id clause).
        should = [
            FieldCondition(key="metadata.visibility", match=MatchValue(value=MemoryVisibility.SHARED.value)),
            FieldCondition(key="metadata.visibility", match=MatchValue(value=MemoryVisibility.STANDARD.value)),
            Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=caller_user_id or ""))]),
        ]
        scroll_filter = Filter(must=must, should=should)

        add_tags = list(ops.get("add_tags") or [])
        remove_tags = set(ops.get("remove_tags") or [])
        now_iso = datetime.now(timezone.utc).isoformat()

        matched = updated = skipped_forbidden = skipped_invalid = 0
        graph_jobs: list[dict] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                matched += 1
                payload = pt.payload or {}
                meta = payload.get("metadata") or {}
                if isinstance(meta.get("metadata"), dict):
                    meta = meta["metadata"]
                meta = dict(meta)
                try:
                    _check_edit_permission(
                        meta, payload.get("user_id", ""), caller_user_id,
                        edits_content=False, edits_visibility=False,
                    )
                except PermissionError:
                    skipped_forbidden += 1
                    continue

                new_meta = dict(meta)
                if add_tags or remove_tags:
                    tags = [t for t in (meta.get("tags") or []) if t not in remove_tags]
                    tags.extend(t for t in add_tags if t not in tags)
                    if tags:
                        new_meta["tags"] = tags
                    else:
                        new_meta.pop("tags", None)
                if ops.get("set_category"):
                    new_meta["category"] = ops["set_category"]
                if "set_project_id" in ops:
                    if ops["set_project_id"] is None:
                        new_meta.pop("project_id", None)
                    else:
                        new_meta["project_id"] = ops["set_project_id"]

                changed = (
                    (new_meta.get("tags") or None) != (meta.get("tags") or None)
                    or new_meta.get("category") != meta.get("category")
                    or new_meta.get("project_id") != meta.get("project_id")
                )
                if not changed:
                    continue  # matched, nothing to change
                try:
                    new_meta["scope"] = self._derive_scope(
                        new_meta.get("category"), new_meta.get("project_id")
                    )
                except ValueError:
                    skipped_invalid += 1
                    continue
                new_meta["project_id"] = new_meta.get("project_id")

                updated += 1
                if dry_run:
                    continue
                m.vector_store.update(
                    str(pt.id), payload={"metadata": new_meta, "updated_at": now_iso}
                )
                owner = meta.get("owner_user_id") or payload.get("user_id", "")
                visibility = meta.get("visibility") or MemoryVisibility.PRIVATE.value
                old_group = _build_group_id(visibility, owner, meta.get("project_id"))
                new_group = _build_group_id(visibility, owner, new_meta.get("project_id"))
                if new_group != old_group:
                    try:
                        self._expire_graph_edges_for_memory(
                            {"memory": payload.get("data", ""), "metadata": meta, "user_id": owner}
                        )
                    except Exception as e:
                        logger.warning(f"Graph edge expiration failed for {pt.id} (non-critical): {e}")
                    graph_jobs.append({
                        "memory_id": str(pt.id),
                        "content": payload.get("data", ""),
                        "user_id": owner,
                        "project_id": new_meta.get("project_id"),
                        "visibility": visibility,
                        "source_ref": meta.get("source_ref"),
                    })
            if offset is None:
                break

        _audit_log.info(
            "memories_retagged",
            caller=caller_user_id,
            filters={k: v for k, v in filters.items() if v is not None},
            ops={k: v for k, v in ops.items()},
            matched=matched,
            updated=updated,
            skipped_forbidden=skipped_forbidden,
            skipped_invalid=skipped_invalid,
            graph_migrations=len(graph_jobs),
            dry_run=dry_run,
        )
        return {
            "matched": matched,
            "updated": updated,
            "skipped_forbidden": skipped_forbidden,
            "skipped_invalid": skipped_invalid,
            "graph_jobs": graph_jobs,
            "dry_run": dry_run,
        }

    def delete_memory(self, memory_id: str, caller_user_id: str | None = None) -> dict:
        """Delete a single memory by ID from both vector store and graph.

        ``caller_user_id`` gates deletion of authoritative ``standard``-tier
        memories to dictators. This is the only delete path with no user
        namespacing (bulk deletes are already scoped by ``user_id``), so
        without this check any caller could remove a binding standard by ID.
        """
        m = self._get_memory()

        # First, get the memory content to find related graph edges
        mem = m.get(memory_id)

        # Standard-tier delete protection: only a dictator may remove standards.
        if mem is not None:
            try:
                vis = getattr(self._mem_to_response(mem), "visibility", None)
            except Exception:
                vis = None
            if vis == MemoryVisibility.STANDARD.value and not settings.is_dictator(caller_user_id):
                raise PermissionError(
                    "Only a dictator may delete 'standard'-tier memories."
                )

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
                if self._graphiti and self._bridge:
                    self._expire_user_graph_writes(user_id)
                m.delete_all(user_id=user_id)
                return {"message": "All memories deleted (including shared)"}

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
                        self._expire_graph_edges_for_memory(
                            {"memory": mem_info.get("data", ""), "metadata": meta}
                        )
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete null-category memory {mid}: {e}")
            return {"message": _deleted_msg("null-category memories", deleted_count, skipped_shared, skipped_standard)}

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
                    self._expire_graph_edges_for_memory(full_mem)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {mem.id}: {e}")

        return {"message": _deleted_msg("memories", deleted_count, skipped_shared, skipped_standard)}

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

        group_ids = _get_group_ids(user_id, project_id)

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

        group_ids = _get_group_ids(user_id, project_id)

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

    def delete_episode(self, episode_uuid: str) -> dict:
        """Delete a single episodic node from the graph by UUID."""
        g = self._get_graphiti()
        if g is None:
            return {"error": "Graphiti not initialized"}

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
            project_ids_to_scan = [None] + ALL_KNOWN_PROJECTS

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
                    result = self.delete_episode(ep["uuid"])
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
        """Convert a mem0 memory dict to a MemoryResponse.

        mem0's _search_vector_store / _get_all_from_vector_store helpers lift
        every non-promoted payload field into a top-level "metadata" dict.
        Because our Qdrant payload already nests our fields under a literal
        "metadata" key, the returned shape is {"metadata": {"metadata": {...}}}.
        Unwrap one level if we see that pattern so category/scope/project_id/
        tags/source resolve to their real values.

        Memory-model v2 fields (domain, observation_type, concepts, source_type,
        related_memory_ids, confidence, expires_at) surface as nulls for legacy
        memories that didn't store them — no migration needed.
        """
        metadata = mem.get("metadata", {}) or {}
        if isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]
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
            domain=metadata.get("domain"),
            observation_type=metadata.get("observation_type"),
            concepts=metadata.get("concepts"),
            source_type=metadata.get("source_type"),
            related_memory_ids=metadata.get("related_memory_ids"),
            confidence=metadata.get("confidence"),
            expires_at=metadata.get("expires_at"),
            derived_from=metadata.get("derived_from"),
            epistemic_level=metadata.get("epistemic_level"),
            memory_kind=metadata.get("memory_kind"),
            source_ref=metadata.get("source_ref"),
            visibility=metadata.get("visibility"),
            owner_user_id=metadata.get("owner_user_id"),
            title=metadata.get("title"),
            token_estimate=metadata.get("token_estimate"),
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
        limit: int | None = None,
    ) -> list[MemoryResponse]:
        """Deduplicate and weave vector and graph results (audit 27 #2).

        Removes graph results whose content closely matches a vector result,
        then appends the surviving graph rows AFTER the vector hits: scored
        vector hits keep their ranked order and take priority. The old 1:1
        positional interleave let unranked (score=None) relation strings
        evict up to half of the ranked vector top-k after the caller's
        ``[:limit]`` cut.

        When ``limit`` is given, graph rows are capped at
        ``max(1, limit // 4)`` slots within the limit — except they may
        additionally fill any shortfall when the vector legs returned fewer
        than ``limit`` hits. Vector hits are only ever displaced down to
        ``limit - cap`` slots (e.g. 8 of 10 at the default k=10), never
        below.
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

        if limit is None:
            # No budget context (legacy callers): vector first, graph after.
            return vector_responses + unique_graph

        # Graph reservation: max(1, limit // 4) slots, but never the vector
        # top hit's slot (at limit=1 the single ranked vector hit wins).
        graph_cap = max(1, limit // 4)
        if vector_responses:
            graph_cap = min(graph_cap, limit - 1)
        vector_target = min(len(vector_responses), max(limit - graph_cap, 0))
        # Graph rows fill their reservation plus any vector shortfall;
        # vector reclaims reserved slots the graph leg can't use.
        graph_take = min(len(unique_graph), limit - vector_target)
        vector_kept = vector_responses[: limit - graph_take]
        return vector_kept + unique_graph[:graph_take]

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

    def _expire_user_graph_writes(self, user_id: str) -> None:
        """Expire graph edges across every group_id this user authored.

        Used by the unfiltered bulk-delete path. Private groups
        (`user--{user_id}` and `user--{user_id}--project--*`) are
        expired wholesale — they only contain this user's writes.
        Shared groups (`shared`, `shared--project--*`) hold team
        knowledge from many writers, so we only expire the specific
        edges this user authored via per-memory cleanup.
        """
        try:
            user_memories = self._scroll_all_user_memories(user_id)
        except Exception as e:
            logger.warning(f"Failed to scroll memories for graph cleanup (non-critical): {e}")
            return

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
                shared_memories.append({"memory": payload.get("data", ""), "metadata": metadata})
            else:
                if pid:
                    private_groups.add(f"user--{user_id}--project--{pid}")
                else:
                    private_groups.add(f"user--{user_id}")

        if private_groups:
            self._expire_graph_edges_for_groups(sorted(private_groups))
        for mem in shared_memories:
            try:
                self._expire_graph_edges_for_memory(mem)
            except Exception as e:
                logger.warning(f"Per-shared-memory edge expiration failed (non-critical): {e}")

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
