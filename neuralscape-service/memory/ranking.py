"""Recall ranking helpers: RRF fusion, reinforcement and salience boosts.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import math

from config import settings

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


def _dense_score_floor(dense_hits: list) -> float | None:
    """Weakest cosine score in a pool's dense leg (imputed score for
    lexical-only hits: presence via a strong keyword match is worth at
    least as much as the pool's weakest dense candidate — never more)."""
    scores = [
        s for s in (getattr(h, "score", None) for h in dense_hits) if s is not None
    ]
    return min(scores) if scores else None


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
