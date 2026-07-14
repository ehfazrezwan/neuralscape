"""C3 nl_locate retrieval helpers — token-free card-text BM25 + local dense.

The A/B (``reports/ICE_V2_NLLOCATE_EMBEDDINGS.md``) proved native ``locate``'s
0.16 h@1 was a *configuration artifact*, not a ceiling: the deterministic default
indexed no symbol-card text and ranked on ``fqn``/``file`` tokens alone, but an
nl_locate query is a natural-language docstring that shares almost no tokens with a
dotted FQN. The fix — proved token-free and quality-≥-cloud — is:

- **C1 (always-on, zero-dep):** BM25 over the symbol-card TEXT
  (name + signature + docstring + source). Alone: h@1 0.16 → 0.60, h@5 → 0.96.
- **C3 (default):** add a **local** code embedder dense leg fused with BM25 +
  graph-degree via RRF → h@1 ~0.76, *beating* cloud embeddings at **zero API
  tokens**. The local embedder is ``fastembed`` ONNX (already a dependency), so
  no torch and no container-gate landmine.

This module is the self-contained retrieval core: a dependency-free Okapi BM25
index over card text (module-cached per code_space), the canonical card-text
builder shared by index + query so the two legs match, a deterministic
symbol→point-id (so dense + lexical legs fuse on symbol identity), and a lazy
``fastembed`` wrapper for the local dense leg.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter

logger = logging.getLogger(__name__)

# Deterministic namespace for symbol point ids. A symbol's Qdrant point id and
# its lexical-leg hit id are both uuid5(code_space::fqn), so RRF fuses the two
# legs on symbol identity (leg agreement ranks a symbol up) and re-indexing a
# symbol is idempotent (same id → upsert overwrites, no duplicate points).
_POINT_NS = uuid.UUID("6f4d2c1a-9b3e-4f7a-8c2d-1e5a7b9c0d3f")


def symbol_point_id(code_space: str, fqn: str) -> str:
    """Stable point id for a symbol, shared by the dense + lexical legs."""
    return str(uuid.uuid5(_POINT_NS, f"{code_space}::{fqn}"))


# ── Card text ────────────────────────────────────────────────────────


def build_card_text(
    fqn: str,
    kind: str,
    signature: str | None,
    docstring: str | None,
    source: str | None,
) -> str:
    """Canonical symbol-card text (name + signature + docstring + source).

    Used identically at index time (what we embed / BM25-index) and to
    reconstruct the lexical corpus, so both retrieval legs see the same text.
    """
    parts = [f"{kind or ''} {fqn or ''}".strip()]
    if signature:
        parts.append(f"Signature: {signature}")
    if docstring:
        parts.append(f"Doc: {docstring}")
    if source:
        parts.append(f"Source:\n{source}")
    return "\n".join(p for p in parts if p)


# ── Tokenizer (shared by index + query) ──────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, splitting camelCase and snake_case so identifiers
    like ``setIndentation`` / ``set_indentation`` match a docstring's "set
    indentation". Deterministic and dependency-free."""
    if not text:
        return []
    # Insert a boundary between camelCase humps, then lowercase + extract runs.
    spaced = _CAMEL_RE.sub(" ", text)
    return _WORD_RE.findall(spaced.lower())


# ── Okapi BM25 (dependency-free, deterministic) ──────────────────────


class BM25Index:
    """Compact Okapi BM25 over a fixed document corpus.

    Deterministic, pure-Python, no network — the token-free lexical leg. For the
    small-py corpus (~1,500 cards) fit + per-query scoring is sub-millisecond.
    """

    __slots__ = ("k1", "b", "N", "avgdl", "idf", "tf", "doc_len")

    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        doc_tokens = [tokenize(d) for d in docs]
        self.N = len(doc_tokens)
        self.doc_len = [len(t) for t in doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: list[Counter] = []
        df: dict[str, int] = {}
        for toks in doc_tokens:
            c = Counter(toks)
            self.tf.append(c)
            for term in c:
                df[term] = df.get(term, 0) + 1
        # Okapi IDF with the +1 shift (keeps every idf > 0, so a common term
        # never contributes a negative score).
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(doc_index, score)`` pairs, best first."""
        q_terms = tokenize(query)
        if not q_terms or self.N == 0 or self.avgdl == 0:
            return []
        scores: list[tuple[int, float]] = []
        for i in range(self.N):
            tf = self.tf[i]
            dl = self.doc_len[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (f * (self.k1 + 1)) / denom
            if s > 0.0:
                scores.append((i, s))
        # Stable sort by score desc (ties keep corpus order → reproducible).
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ── Fusion hit shape (uniform with Qdrant point objects) ─────────────


class LexHit:
    """A lexical-leg hit shaped like a Qdrant point (``.id`` + ``.payload`` +
    ``.score``) so ``memory.ranking._rrf_fuse`` fuses it with dense hits."""

    __slots__ = ("id", "payload", "score")

    def __init__(self, id: str, payload: dict, score: float):
        self.id = id
        self.payload = payload
        self.score = score


# ── Module-level BM25 corpus cache (survives per-request engine instances) ──

_BM25_CACHE: dict[str, tuple[BM25Index, list[dict]]] = {}


def get_or_build_bm25(code_space: str, loader) -> tuple[BM25Index, list[dict]]:
    """Return a cached ``(BM25Index, payloads)`` for ``code_space``, building it
    via ``loader()`` (which returns ``list[payload_dict]`` each carrying a
    ``card`` text field) on a cache miss. Invalidated by :func:`invalidate_bm25`
    at (re)index time."""
    cached = _BM25_CACHE.get(code_space)
    if cached is not None:
        return cached
    payloads = loader() or []
    index = BM25Index([p.get("card") or "" for p in payloads])
    _BM25_CACHE[code_space] = (index, payloads)
    return index, payloads


def invalidate_bm25(code_space: str) -> None:
    """Drop the cached BM25 corpus for a code_space (call after (re)indexing)."""
    _BM25_CACHE.pop(code_space, None)


# ── Local code embedder (lazy fastembed ONNX — token-free, no torch) ──

_EMBEDDER_CACHE: dict[str, "CodeEmbedder"] = {}


class CodeEmbedder:
    """Lazy ``fastembed`` wrapper for the local dense leg.

    Default model ``jinaai/jina-embeddings-v2-base-code`` (Apache-2.0, 768-dim,
    ONNX — no torch dependency), validated token-free at h@1 0.747 in the A/B
    (§4), in the same band as cloud (0.753) and CodeRankEmbed (0.76). The model
    is downloaded/loaded on first use, so construction is cheap and unit tests
    never touch the network unless they explicitly embed.

    ``query_prefix`` supports asymmetric models (e.g. CodeRankEmbed wants
    ``"Represent this query for searching relevant code: "`` on queries only);
    jina is symmetric so it defaults to empty.
    """

    def __init__(self, model: str, query_prefix: str = ""):
        self.model = model
        self.query_prefix = query_prefix or ""
        self._backend = None

    def _get(self):
        if self._backend is None:
            from fastembed import TextEmbedding

            self._backend = TextEmbedding(model_name=self.model)
            logger.info("Loaded local code embedder: %s", self.model)
        return self._backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(v) for v in self._get().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        q = f"{self.query_prefix}{text}" if self.query_prefix else text
        return list(next(iter(self._get().query_embed([q]))))


def get_code_embedder(model: str, query_prefix: str = "") -> CodeEmbedder:
    """Process-cached :class:`CodeEmbedder` keyed by model (one ONNX load)."""
    emb = _EMBEDDER_CACHE.get(model)
    if emb is None:
        emb = CodeEmbedder(model, query_prefix=query_prefix)
        _EMBEDDER_CACHE[model] = emb
    return emb
