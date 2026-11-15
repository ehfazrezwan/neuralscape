"""Recall ranking helpers: RRF fusion, reinforcement and salience boosts.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import math

from config import VECTOR_EMBEDDING_DIMS, settings

logger = logging.getLogger(__name__)

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


def _unit_cosine(a: list, b: list) -> float | None:
    """Cosine similarity between two vectors, clamped to [0, 1].

    Used to score graph edges with their STORED ``fact_embedding`` against
    the query vector ``search()`` already computed — pure local arithmetic,
    no embed/API calls. Both vectors live in the same Gemini embedding
    space as the Qdrant rows, so the result is directly comparable to the
    vector leg's cosine scores. Returns ``None`` on any malformed input
    (empty, length mismatch, zero norm, non-numeric) — an unscorable edge
    must degrade to the legacy unscored path, never break the read.
    """
    try:
        if not a or not b or len(a) != len(b):
            return None
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a <= 0.0 or norm_b <= 0.0:
            return None
        return max(0.0, min(1.0, dot / math.sqrt(norm_a * norm_b)))
    except (TypeError, ValueError):
        return None


# ── Cross-store (Qdrant 768-d vs Graphiti 1024-d) comparison ───────────
# Graphiti's stored vectors and the mem0/Qdrant vectors come from the same
# Matryoshka (MRL) model, gemini-embedding-001, at different
# ``output_dimensionality``. A lower-dimension MRL vector is the leading
# prefix of the full vector -- Google's embeddings guide, section "Controlling
# embedding size": gemini-embedding-001 "is trained using the Matryoshka
# Representation Learning (MRL) technique" and its output "can be truncated";
# non-3072 outputs of gemini-embedding-001 are not unit-norm
# (https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size).
# Two such vectors of different length are therefore compared on their common
# prefix (cosine renormalises). Only models listed
# here may be compared that way; anything else keeps the strict
# length-match contract of ``_unit_cosine``.
_MRL_EMBEDDING_MODELS = frozenset({"gemini-embedding-001"})


def _canonical_embedding_model(name: str | None) -> str:
    """``models/x``, ``google-vertex/x`` and ``x`` all name the same model."""
    return (name or "").strip().lower().rsplit("/", 1)[-1]


def _graph_vectors_mrl_comparable() -> bool:
    """True when the graph embedder and the vector embedder are the SAME MRL
    model, so prefix comparison across their dimensions is meaningful.

    The graph side always uses ``gemini_embedder_model`` (both graphiti model
    dicts in ``Settings``); the vector side uses the gateway embedder model
    when the gateway is enabled, else ``gemini_embedder_model``.
    """
    graph_model = _canonical_embedding_model(settings.gemini_embedder_model)
    vector_model = _canonical_embedding_model(
        settings.llm_gateway_embedder_model
        if settings.llm_gateway_enabled
        else settings.gemini_embedder_model
    )
    return graph_model == vector_model and graph_model in _MRL_EMBEDDING_MODELS


def _mrl_cosine(a: list, b: list) -> float | None:
    """``_unit_cosine`` over the common prefix when the lengths differ.

    Callers must have established that both vectors are leading-prefix
    truncations of one MRL model (``_graph_vectors_mrl_comparable``);
    this function does not check. Equal lengths behave exactly like
    ``_unit_cosine``. The prefix is not unit-norm, but cosine normalises.
    """
    try:
        if a and b and len(a) != len(b):
            n = min(len(a), len(b))
            return _unit_cosine(list(a[:n]), list(b[:n]))
    except TypeError:
        return None
    return _unit_cosine(a, b)


def _graph_edge_cosine(query_embedding: list, edge_embedding: list) -> float | None:
    """Score a graph edge's stored ``fact_embedding`` against the query vector.

    Same-length vectors: plain cosine. Different lengths: prefix cosine when
    both stores use the same MRL model, otherwise ``None`` (unscorable, as
    before).
    """
    if (
        query_embedding
        and edge_embedding
        and len(query_embedding) != len(edge_embedding)
        and not _graph_vectors_mrl_comparable()
    ):
        return None
    return _mrl_cosine(query_embedding, edge_embedding)


def _fit_vector_to_collection(vec: list | None) -> list | None:
    """Make a stored graph edge vector usable as a query against the Qdrant
    collection (``VECTOR_EMBEDDING_DIMS``).

    Longer MRL vectors are truncated to the collection dimension and
    re-normalised; longer non-MRL vectors cannot be searched with and yield
    ``None`` (caller treats the row as having no stored embedding). Vectors
    already at or below the collection dimension pass through unchanged.
    """
    if not vec:
        return vec
    if len(vec) <= VECTOR_EMBEDDING_DIMS:
        return vec
    if not _graph_vectors_mrl_comparable():
        return None
    try:
        head = [float(x) for x in vec[:VECTOR_EMBEDDING_DIMS]]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(x * x for x in head))
    if norm <= 0.0:
        return None
    return [x / norm for x in head]


def _dense_score_floor(dense_hits: list) -> float | None:
    """Weakest cosine score in a pool's dense leg (imputed score for
    lexical-only hits: presence via a strong keyword match is worth at
    least as much as the pool's weakest dense candidate — never more)."""
    scores = [
        s for s in (getattr(h, "score", None) for h in dense_hits) if s is not None
    ]
    return min(scores) if scores else None


def _fused_order_keys(fused: list[dict], dense_hits: list) -> list[float | None]:
    """Rank-transferred ordering keys for one pool's RRF-fused entries.

    ``_rrf_fuse`` orders a pool by leg agreement, but the rows it returns
    keep their NATIVE presentation score (real cosine for dense hits, the
    pool's dense floor for lexical-only hits). Sorting the cross-pool merge
    by that native score throws the fused order away: every lexical-only
    hit sits at the pool floor and is cut, and a dense hit that RRF
    promoted is re-ordered back by raw cosine.

    Fix: transfer the fused ORDER onto the cosine SCALE. The pool's dense
    cosines, sorted descending, are handed out by fused position — the
    entry at position ``i`` gets the ``i``-th best cosine; positions past
    the dense leg's length get the pool floor. The keys are monotone
    non-increasing in fused order (so the pool's RRF order survives any
    sort by key) and live on the cosine scale (so they stay comparable
    across pools, unlike raw RRF values, which would degenerate into a
    positional interleave). Keys only ORDER rows; displayed scores are
    untouched.

    Returns one entry per fused position; every entry is ``None`` when the
    pool's dense leg was empty (no cosine scale to transfer onto) — the
    caller resolves those against the other pools.
    """
    dense_scores = sorted(
        (s for s in (getattr(h, "score", None) for h in dense_hits) if s is not None),
        reverse=True,
    )
    if not dense_scores:
        return [None] * len(fused)
    floor = dense_scores[-1]
    return [
        dense_scores[i] if i < len(dense_scores) else floor
        for i in range(len(fused))
    ]


def _record_order_key(
    order_keys: dict | None, row_id: str, key: float | None
) -> None:
    """Record a pool row's ordering key; the best key wins across pools.

    ``None`` (empty-dense pool) only registers the id so ``search()`` can
    impute it later; it never overwrites a real key.
    """
    if order_keys is None or not row_id:
        return
    prior = order_keys.get(row_id)
    if prior is None or (key is not None and key > prior):
        order_keys[row_id] = key


def _resolve_order_keys(
    order_keys: dict[str, float | None],
) -> tuple[dict[str, float], set[str]]:
    """Impute keys for rows from empty-dense pools (key ``None``).

    A lexical-only hit in a pool whose dense leg returned nothing has no
    cosine to rank on. Same principle as ``_dense_score_floor`` (presence
    via a keyword match is worth at least the weakest dense candidate,
    never more): such rows get the LOWEST real key seen across all pools,
    or 0.0 when no pool had a dense leg at all. They tie with that weakest
    row; the returned ``imputed`` id set lets the caller sort them AHEAD of
    that tie, so they are not silently sunk below everything, yet never
    outrank a stronger dense hit. Returns ``(keys, imputed_ids)``.
    """
    real = [k for k in order_keys.values() if k is not None]
    fallback = min(real) if real else 0.0
    keys = {i: (k if k is not None else fallback) for i, k in order_keys.items()}
    imputed = {i for i, k in order_keys.items() if k is None}
    return keys, imputed


def _mem_is_tombstoned(mem: dict) -> bool:
    """Whether a mem0 memory dict carries the dreaming consolidation
    tombstone (``metadata.dream_tombstoned``), handling mem0's potential
    ``{"metadata": {"metadata": {...}}}`` double-wrap."""
    meta = mem.get("metadata") or {}
    if isinstance(meta.get("metadata"), dict):
        meta = meta["metadata"]
    return bool(meta.get("dream_tombstoned"))


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


def _salience_tiebreak(responses: list, order_keys: dict | None = None) -> list:
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

    ``order_keys`` (optional ``{id: ordering key}`` from the hybrid pool
    merge) receives the same multiplicative lift as the displayed score, so
    the sort — which runs on the keys — stays consistent with the boost.
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
                lift = 1.0 + k * math.log1p(signal)
                r.score = r.score * lift
                if order_keys is not None and order_keys.get(r.id) is not None:
                    order_keys[r.id] = order_keys[r.id] * lift
    except Exception:
        logger.debug("salience tie-break skipped (non-fatal)", exc_info=True)
    return responses
