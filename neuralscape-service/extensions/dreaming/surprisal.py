"""A5 — surprisal-targeted REM (lite): novelty as distance from the pool centroid.

The cheap version of Honcho's cover trees: one batched Qdrant retrieve pulls
the staged rows' embeddings, and each memory's **surprisal** is its cosine
distance from the pool's embedding centroid — an anomaly score in [0, 2]
(0 = dead center of what the pool already believes, →2 = pointing away from
everything else). The reflection substrate is then *biased* toward the top-K
anomalies: they are moved to the front of the memories block, the remainder
keeps its original order. Nothing is dropped — bias, not filter.

``DREAMING_SURPRISAL_TOP_K = 0`` disables the whole pass: no vector fetch, no
``surprisal`` keys, and :func:`bias_substrate` returns the input list object
unchanged, so the reflection prompt stays byte-identical to the uniform path.

Pure math on stdlib floats — no numpy dependency, no I/O outside
:func:`fetch_vectors`.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


# ── Qdrant vector fetch (the one I/O helper) ────────────────────────


def _as_vector(raw) -> list[float] | None:
    """Normalize a qdrant Record.vector to a dense float list.

    Handles the three shapes mem0's Qdrant layer produces: a plain list, a
    dict of named vectors — e.g. ``{"bm25": SparseVector, "": [dense...]}``
    when hybrid BM25 is enabled — and None. For named vectors the unnamed
    default dense vector is preferred; sparse vectors (non-list values)
    are skipped rather than mistaken for embeddings.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        candidates = ([raw[""]] if "" in raw else []) + [
            v for k, v in raw.items() if k != ""
        ]
        for candidate in candidates:
            vec = _as_vector(candidate)
            if vec is not None:
                return vec
        return None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]
    return None


def fetch_vectors(service, memory_ids: list[str]) -> dict[str, list[float]]:
    """One batched retrieve of the staged ids' embeddings (with_vectors=True)."""
    if not memory_ids:
        return {}
    from config import settings as core_settings

    client = service._memory.vector_store.client
    points = client.retrieve(
        collection_name=core_settings.qdrant_collection,
        ids=list(memory_ids),
        with_payload=False,
        with_vectors=True,
    )
    out: dict[str, list[float]] = {}
    for point in points or []:
        vec = _as_vector(getattr(point, "vector", None))
        if vec:
            out[str(getattr(point, "id", "") or "")] = vec
    return out


# ── Pure math ───────────────────────────────────────────────────────


def centroid(vectors: list[list[float]]) -> list[float] | None:
    """Component-wise mean. None when empty or dimensions disagree."""
    if not vectors:
        return None
    dim = len(vectors[0])
    if dim == 0 or any(len(v) != dim for v in vectors):
        return None
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 − cos(a, b) in [0, 2]; 0.0 when either vector has zero norm."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return 1.0 - dot / math.sqrt(na * nb)


def surprisal_scores(id_to_vector: dict[str, list[float]]) -> dict[str, float]:
    """Cosine distance of each vector from the pool centroid.

    Fewer than 3 vectors ⇒ no meaningful "pool center" — returns {} (a pair
    is always mutually anomalous; scoring it would just be noise).
    """
    if len(id_to_vector) < 3:
        return {}
    center = centroid(list(id_to_vector.values()))
    if center is None:
        return {}
    return {mid: cosine_distance(vec, center) for mid, vec in id_to_vector.items()}


def annotate(memories: list[dict], id_to_vector: dict[str, list[float]]) -> int:
    """Stamp ``surprisal`` onto staged dicts (in place). Returns count stamped.

    Memories whose vector could not be fetched are left untouched — they
    simply never rank as anomalies.
    """
    scores = surprisal_scores(id_to_vector)
    stamped = 0
    for mem in memories:
        score = scores.get(mem.get("memory_id") or "")
        if score is not None:
            mem["surprisal"] = round(score, 6)
            stamped += 1
    return stamped


def bias_substrate(memories: list[dict], top_k: int) -> list[dict]:
    """Reorder the reflection substrate: top-K anomalies first, rest stable.

    ``top_k <= 0`` returns the input list object unchanged (byte-identical
    uniform behavior). Otherwise the K highest-surprisal memories lead the
    list in descending surprisal order (memory_id as a deterministic
    tie-break) and every other row follows in its original order. Rows
    without a ``surprisal`` score can never be selected as anomalies.
    """
    if top_k <= 0:
        return memories
    scored = [m for m in memories if isinstance(m.get("surprisal"), (int, float))]
    if not scored:
        return memories
    top = sorted(
        scored,
        key=lambda m: (-float(m["surprisal"]), str(m.get("memory_id") or "")),
    )[:top_k]
    top_ids = {id(m) for m in top}
    rest = [m for m in memories if id(m) not in top_ids]
    return top + rest
