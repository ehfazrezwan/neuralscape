"""NativeEngine — tree-sitter indexer + Neo4j code label-space (E2).

E2 scope (Python only):
- Index with tree-sitter (heuristic FQN, no LSP) → Neo4j code_space partition
- Incremental by file content-hash
- Query/neighbors/path produce the SAME text output as GraphifyJsonEngine (parity)
- Locate/detect_changes/semantic_layer/export_snapshot raise EngineCapabilityError (E3+)

Label-space schema:
  (:CodeRepo {code_space, name, path})
  (:CodeFile {code_space, path, hash, language, span})
  (:CodeSymbol {code_space, fqn, kind, file, span, degree, community_id})
  Edges: CALLS | IMPORTS | DEFINES | INHERITS | REFERENCES
    each with {extraction: "extracted"|"inferred"|"ambiguous"}

Partition key: code_space = "code--{owner}--{repo}" on EVERY node.
Degree and community_id (Louvain, seed=42) persisted on symbols at index time.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.code_graph.engine import (
    ChangeReport,
    EngineCapabilityError,
    IndexReport,
    LocateHit,
    SemanticFact,
)

logger = logging.getLogger(__name__)


@dataclass
class _Symbol:
    """Internal symbol representation for indexing."""

    fqn: str
    kind: str  # function | class | method | module
    file: str
    line: int
    end_line: int
    docstring: str = ""


@dataclass
class _Edge:
    """Internal edge representation."""

    source_fqn: str
    target_fqn: str
    relation: str  # CALLS | IMPORTS | DEFINES | INHERITS | REFERENCES
    extraction: str  # extracted | inferred | ambiguous


class NativeEngine:
    """E2 native code-intel engine: tree-sitter → Neo4j code label-space.

    Attributes:
        repo_path: Filesystem path to the repository being indexed.
        code_space: Partition key (code--{owner}--{repo}).
        bridge: The Graphiti bridge (for Neo4j driver access).
        settings: Config (for confidence thresholds, etc.).
    """

    def __init__(
        self,
        repo_path: str,
        code_space: str,
        bridge: Any,
        settings: Any,
        driver: Any = None,
    ):
        """Initialize with repo path and Neo4j bridge.

        Args:
            repo_path: Absolute path to the repo root.
            code_space: Partition key (code--{owner}--{repo}).
            bridge: Graphiti async bridge (provides ._loop / .run). NOTE: the
                real mem0 ``_AsyncBridge`` does NOT expose ``.driver`` — the
                Neo4j driver is passed separately as ``driver``. A mock bridge
                carrying ``.driver`` still works via the fallback in _run_cypher.
            settings: Config object.
            driver: Graphiti Neo4j async driver (``service._graphiti.driver``).
                Required for live Cypher; None ⇒ _run_cypher falls back to
                ``bridge.driver`` (mock/unit-test path).
        """
        self.repo_path = Path(repo_path)
        self.code_space = code_space
        self.bridge = bridge
        self.settings = settings
        self.driver = driver

    # ── Canonical FQN normalization (Phase C) ───────────────────────────

    @staticmethod
    def to_canonical(raw_fqn: str) -> str:
        """Normalize native engine's FQN to canonical form.

        Native format: `<module_path>.<qualname>` where module_path includes
        src/lib roots (e.g. `src.click.core.CommandCollection`).

        Canonical format: `<module_path>.<qualname>` with src/lib roots stripped
        (e.g. `click.core.CommandCollection`).

        Per PLAN §2: canonical_fqn := <repo-relative module path, src/lib roots
        stripped, '/' → '.'> + '.' + <qualname dotted>.

        Args:
            raw_fqn: Native engine's FQN (may include src/lib prefix).

        Returns:
            Canonical FQN (src/lib stripped).
        """
        # Strip genuine source-root directories from the start ONLY.
        # Kept deliberately narrow: `core`/`main`/`app`/`internal`/`pkg` are
        # real module names (the click corpus has `click.core`), so stripping
        # them would corrupt canonical FQNs. Only `src`/`lib` are true source
        # roots that an indexer prepends. Driven by the ≥98% oracle conformance
        # measurement (test_canonical_fqn_conformance.py).
        root_markers = {"src", "lib"}
        parts = raw_fqn.split(".")

        # Remove leading root markers
        while parts and parts[0] in root_markers:
            parts.pop(0)

        canonical = ".".join(parts)
        logger.debug(f"Native to_canonical: {raw_fqn} → {canonical}")
        return canonical

    @staticmethod
    def from_canonical(canonical_fqn: str) -> str:
        """Convert canonical FQN back to native format (best-effort).

        Since we don't know which root (src/lib/etc) was stripped, we can't
        reconstruct it exactly. For queries, we use the canonical form directly
        (matches will work as long as the query is on the canonical FQN).

        Args:
            canonical_fqn: Canonical FQN (e.g. click.core.CommandCollection).

        Returns:
            FQN for native queries (same as canonical for search purposes).
        """
        # For search, canonical works as-is (our CONTAINS query tolerates it).
        return canonical_fqn

    def query(
        self,
        question: str,
        *,
        mode: str = "bfs",
        depth: int = 3,
        token_budget: int = 2000,
        user_id: str | None = None,
    ) -> str:
        """Search the code graph via BFS/DFS from scored seed nodes.

        E4: Enriches results with attached memories (decisions, gotchas, bugfix
        history) via anchors, respecting caller's read scope.
        """
        # Normalize params the same way GraphifyJsonEngine does
        mode = mode if mode in ("bfs", "dfs") else "bfs"
        depth = max(1, min(int(depth), 6))
        token_budget = max(100, min(int(token_budget), 20_000))

        # Retrieve symbols matching the question keywords
        keywords = question.lower().split()
        symbols = self._search_symbols(keywords, limit=5)
        if not symbols:
            return f"No symbols matching '{question}' found in {self.code_space}."

        # Fix 3: Include matched symbols in output (symbol_lookup fix)
        lines = [f"Code graph search results for: {question}", ""]

        # First, emit the matched seed symbols themselves
        seed_fqn = symbols[0]["fqn"]
        for sym in symbols:
            lines.append(
                f"{sym['fqn']} ({sym['kind']}) in {sym['file']}:{sym['line']}"
            )
            # E4: Append attached memories for matched symbols
            memories = self._get_anchor_memories(sym["fqn"], user_id=user_id, limit=2)
            if memories:
                lines.append("  Memories:")
                for mem in memories:
                    snippet = (mem["content"] or "")[:100]
                    lines.append(f"    - [{mem['category']}] {snippet}")

        # Then traverse from the top-scored symbol
        lines.append("")  # separator
        visited = self._traverse(seed_fqn, mode=mode, depth=depth, budget=token_budget)
        for item in visited:
            lines.append(
                f"{item['fqn']} ({item['kind']}) in {item['file']}:{item['line']}"
            )
            if item.get("edges"):
                for edge in item["edges"][:3]:  # limit outbound edges shown
                    lines.append(f"  --> {edge['relation']} {edge['target']}")

            # E4: Append attached memories
            memories = self._get_anchor_memories(item["fqn"], user_id=user_id, limit=2)
            if memories:
                lines.append("  Memories:")
                for mem in memories:
                    snippet = (mem["content"] or "")[:100]
                    lines.append(f"    - [{mem['category']}] {snippet}")

        return "\n".join(lines)

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
        user_id: str | None = None,
    ) -> str:
        """Direct in/out neighbors of one code-graph symbol.

        E4: Enriches results with attached memories.
        """
        # Find the symbol by FQN substring match
        matches = self._find_symbol(label)
        if not matches:
            return f"No symbol matching '{label}' found in {self.code_space}."

        symbol = matches[0]
        fqn = symbol["fqn"]
        rel_filter = (relation_filter or "").lower()

        # Fetch in/out edges
        edges = self._get_edges(fqn, rel_filter=rel_filter)
        lines = [f"Neighbors of {fqn}:"]

        # E4: Append attached memories for the queried symbol
        memories = self._get_anchor_memories(fqn, user_id=user_id, limit=2)
        if memories:
            lines.append("Attached memories:")
            for mem in memories:
                snippet = (mem["content"] or "")[:100]
                lines.append(f"  - [{mem['category']}] {snippet}")
            lines.append("")

        if not edges:
            lines.append("  (no neighbors matching the filter)")
        for edge in edges:
            direction = "-->" if edge["direction"] == "out" else "<--"
            lines.append(
                f"  {direction} {edge['neighbor']} [{edge['relation']}] [{edge['extraction']}]"
            )
        return "\n".join(lines)

    def path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 8,
    ) -> str:
        """Shortest path between two code-graph symbols."""
        max_hops = max(1, min(int(max_hops), 32))

        # Resolve source and target
        src_matches = self._find_symbol(source)
        tgt_matches = self._find_symbol(target)
        if not src_matches:
            return f"No symbol matching source '{source}' found."
        if not tgt_matches:
            return f"No symbol matching target '{target}' found."

        src_fqn = src_matches[0]["fqn"]
        tgt_fqn = tgt_matches[0]["fqn"]
        if src_fqn == tgt_fqn:
            return f"'{source}' and '{target}' both resolved to '{src_fqn}'."

        # Run shortest-path Cypher query
        path_nodes = self._shortest_path(src_fqn, tgt_fqn, max_hops)
        if not path_nodes:
            return f"No path found between '{src_fqn}' and '{tgt_fqn}'."

        hops = len(path_nodes) - 1
        if hops > max_hops:
            return f"Path exceeds max_hops={max_hops} ({hops} hops found)."

        # Format the path
        segments = [path_nodes[0]["fqn"]]
        for i in range(hops):
            edge = path_nodes[i]["edge"]
            next_node = path_nodes[i + 1]
            rel = edge["relation"]
            extr = edge["extraction"]
            segments.append(f"--{rel} [{extr}]--> {next_node['fqn']}")

        return f"Shortest path ({hops} hops):\n  " + " ".join(segments)

    # ── F2-future methods (E3+) ──────────────────────────────────────

    def _code_embedder_mode(self) -> str:
        """Resolve the active locate posture: "off" | "local" | "cloud" (C3)."""
        mode = getattr(self.settings, "code_embedder", "local")
        return mode if mode in ("off", "local", "cloud") else "local"

    def _neighbors_resolver_mode(self) -> str:
        """Resolve the neighbors call-graph resolver: "off" | "jedi" | "lsp".

        "jedi" (Wave 3) is in-process Jedi; "lsp" (resolver-svc mission) calls the
        external pyright resolver service and falls back to Jedi if it is down.
        """
        mode = getattr(self.settings, "code_neighbors_resolver", "jedi")
        return mode if mode in ("off", "jedi", "lsp") else "jedi"

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
        user_id: str | None = None,
    ) -> list[LocateHit]:
        """Hybrid code retrieval: card-text BM25 + local dense + graph degree (C3).

        The A/B (reports/ICE_V2_NLLOCATE_EMBEDDINGS.md) proved native locate's
        0.16 h@1 was a config artifact — the old default indexed no card text and
        ranked on fqn/file tokens, blind to NL docstring queries. The C3 default:

        - **Lexical leg (always on, token-free):** Okapi BM25 over the symbol-card
          TEXT (name + signature + docstring + source). Alone: h@1 0.16 → 0.60.
        - **Dense leg (default local, token-free):** a local fastembed ONNX code
          embedder over the same cards; cloud (Gemini) only when opted in.
        - Fuse both via RRF, apply the graph-degree boost → h@1 ~0.76.

        E4: enriches the top-k with attached memories (decisions/gotchas/bugfixes).
        """
        mode = self._code_embedder_mode()
        lexical_on = getattr(self.settings, "code_locate_lexical_cards", True)

        # Retrieve a wider candidate pool per leg than k so the degree boost can
        # legitimately promote a strong-but-lower-ranked hit before truncation
        # (matches the A/B sidecar: boost the whole fused pool, then take top-k).
        pool = max(k * 4, 40)
        lexical_hits = (
            self._locate_lexical_cards(query, pool) if lexical_on else []
        )
        dense_hits: list = []
        if mode in ("local", "cloud"):
            try:
                dense_hits = self._locate_dense(query, pool, mode)
            except Exception:
                logger.warning(
                    "dense locate leg failed (%s mode) — degrading to BM25", mode,
                    exc_info=True,
                )

        # Ultimate fallback: no card text indexed yet AND no dense vectors (e.g.
        # a graph built before card-text indexing) → the legacy fqn/file overlap
        # rank, so locate never returns empty for lack of a card corpus.
        if not lexical_hits and not dense_hits:
            return self._locate_deterministic(query, k=k, user_id=user_id)

        from memory.ranking import _rrf_fuse
        fused = _rrf_fuse(dense_hits, lexical_hits, pool)

        # Apply the degree boost to the WHOLE fused pool, then truncate — a
        # high-degree hit at fused rank > k can legitimately outrank an unboosted
        # earlier one. Anchor memories are then fetched for ONLY the final top-k
        # (bounded Neo4j round-trips).
        scored: list[tuple[float, dict]] = []
        for entry in fused:
            payload = getattr(entry["hit"], "payload", None) or {}
            degree = payload.get("degree", 0) or 0
            degree_boost = 1.0 + 0.01 * min(degree, 50)  # cap boost at ~1.5x
            scored.append((entry["rrf"] * degree_boost, payload))
        scored.sort(key=lambda x: x[0], reverse=True)

        hits: list[LocateHit] = []
        for final_score, payload in scored[:k]:
            fqn = payload.get("fqn", "")
            memories = self._get_anchor_memories(fqn, user_id=user_id, limit=3)
            hits.append(LocateHit(
                fqn=fqn,
                kind=payload.get("kind", ""),
                file=payload.get("file", ""),
                line=payload.get("line", 0),
                signature=payload.get("signature", ""),
                docstring=payload.get("docstring", ""),
                score=final_score,
                anchor_id=payload.get("anchor_id"),
                memories=memories if memories else None,
            ))
        return hits

    def _locate_dense(self, query: str, limit: int, mode: str) -> list:
        """Dense leg of locate: embed the query (local or cloud) and search the
        code_index collection. Returns Qdrant point hits (``.id``/``.payload``)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from memory_service import get_shared_service

        service = get_shared_service()
        m = service._get_memory()
        # Query path: the collection is created at index time, so this is the
        # fast existence check (no dim computation / embed("test") per query).
        self._ensure_code_index_collection(m)

        if mode == "local":
            query_embedding = self._get_local_code_embedder().embed_query(query)
        else:  # cloud
            query_embedding = m.embedding_model.embed(query, memory_action="search")

        # MF-2: filter on the EMBEDDER IDENTITY, not just code_space. Both the
        # local (jina) and cloud (Gemini) embedders are 768-dim, so a mode switch
        # without a reindex would otherwise search the other space's vectors and
        # fuse silent garbage. Tagging each point with its embedder and filtering
        # here means a stale-embedder point simply doesn't match → the dense leg
        # returns empty and locate degrades to BM25 (0.60) until reindex.
        dense_filter = Filter(must=[
            FieldCondition(key="code_space", match=MatchValue(value=self.code_space)),
            FieldCondition(
                key="embedder",
                match=MatchValue(value=self._code_embedder_identity(mode)),
            ),
        ])
        dense_result = m.vector_store.client.query_points(
            collection_name="code_index",
            query=query_embedding,
            query_filter=dense_filter,
            limit=limit,
            with_payload=True,
        )
        return list(getattr(dense_result, "points", dense_result) or [])

    def _code_embedder_identity(self, mode: str) -> str:
        """Stable identity of the active code embedder (``mode:model``), stamped on
        each code_index point and matched at query time so vectors from a
        different embedder are never fused into results (MF-2)."""
        if mode == "local":
            model = getattr(
                self.settings, "code_embedder_model",
                "jinaai/jina-embeddings-v2-base-code",
            )
            return f"local:{model}"
        model = getattr(self.settings, "gemini_embedder_model", "cloud")
        return f"cloud:{model}"

    def _locate_lexical_cards(self, query: str, limit: int) -> list:
        """Lexical leg of locate: Okapi BM25 over the symbol-card text (name +
        signature + docstring + source). Token-free, deterministic, always on.

        The corpus is built once per code_space from the ``card`` property that
        ``_index_symbol_cards`` writes onto each :CodeSymbol, then cached
        (invalidated at reindex). Returns hits shaped like Qdrant points so they
        fuse with the dense leg."""
        from adapters.code_graph.code_locate import (
            LexHit,
            get_or_build_bm25,
            symbol_point_id,
        )

        if not query:
            return []
        try:
            index, payloads = get_or_build_bm25(
                self.code_space, self._card_epoch(), self._load_card_corpus
            )
        except Exception:
            logger.debug("BM25 card corpus build failed (non-fatal)", exc_info=True)
            return []
        hits = []
        for doc_i, score in index.search(query, limit):
            payload = payloads[doc_i]
            hits.append(LexHit(
                id=symbol_point_id(self.code_space, payload.get("fqn", "")),
                payload=payload,
                score=score,
            ))
        return hits

    def _load_card_corpus(self) -> list[dict]:
        """Load all symbol cards (payload dicts carrying a ``card`` text field)
        for this code_space from Neo4j — the BM25 loader."""
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        WHERE s.card IS NOT NULL
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span,
               coalesce(s.degree, 0) AS degree, s.signature AS signature,
               s.docstring AS docstring, s.card AS card
        """
        rows = self._run_cypher(cypher, code_space=self.code_space)
        payloads: list[dict] = []
        for r in rows:
            span = r.get("span") or "1:1"
            try:
                line = int(str(span).split(":")[0])
            except (ValueError, IndexError):
                line = 0
            payloads.append({
                "fqn": r.get("fqn") or "",
                "kind": r.get("kind") or "",
                "file": r.get("file") or "",
                "line": line,
                "signature": r.get("signature") or "",
                "docstring": r.get("docstring") or "",
                "degree": r.get("degree") or 0,
                "anchor_id": None,
                "card": r.get("card") or "",
            })
        return payloads

    def _card_epoch(self) -> int:
        """The code_space's card epoch — a counter bumped on :CodeRepo at every
        reindex. Cheap Cypher; drives cross-process BM25 cache invalidation
        (MF-1: the API process can't see the worker's in-process invalidate)."""
        try:
            rows = self._run_cypher(
                "MATCH (r:CodeRepo {code_space: $cs}) "
                "RETURN coalesce(r.card_epoch, 0) AS epoch",
                cs=self.code_space,
            )
            return int(rows[0]["epoch"]) if rows else 0
        except Exception:
            return 0

    def _bump_card_epoch(self) -> None:
        """Advance the code_space's card epoch (invalidates every process's BM25
        cache on the next locate). Best-effort — never fails an index."""
        try:
            self._run_cypher_with_retry(
                "MERGE (r:CodeRepo {code_space: $cs}) "
                "SET r.card_epoch = coalesce(r.card_epoch, 0) + 1",
                cs=self.code_space,
            )
        except Exception:
            logger.debug("card_epoch bump failed (non-fatal)", exc_info=True)

    def _get_local_code_embedder(self):
        """Process-cached local code embedder (fastembed ONNX, token-free)."""
        from adapters.code_graph.code_locate import get_code_embedder

        return get_code_embedder(
            getattr(
                self.settings,
                "code_embedder_model",
                "jinaai/jina-embeddings-v2-base-code",
            ),
            getattr(self.settings, "code_embedder_query_prefix", ""),
        )

    def _code_vector_size(self, m, mode: str) -> int:
        """Vector dimension of the active code embedder (local vs cloud), so the
        code_index collection is sized to whatever is actually written."""
        if mode == "local":
            probe = self._get_local_code_embedder().embed_documents(["_probe_"])
            return len(probe[0]) if probe else 768
        return len(m.embedding_model.embed("test", memory_action="add"))

    def _locate_deterministic(
        self, query: str, *, k: int = 10, user_id: str | None = None
    ) -> list[LocateHit]:
        """Deterministic, local, no-network locate (default when embeddings off).

        Ranks :CodeSymbol nodes by lexical token overlap between the query and
        each symbol's fqn/file, boosted by graph degree. Pure Neo4j read + local
        scoring — reproducible and API-token-free.
        """
        import re as _re

        def _tok(s: str) -> set[str]:
            return {t for t in _re.split(r"[^a-zA-Z0-9]+", (s or "").lower()) if t}

        q_tokens = _tok(query)
        if not q_tokens:
            return []
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span,
               coalesce(s.degree, 0) AS degree
        """
        symbols = self._run_cypher(cypher, code_space=self.code_space)
        scored: list[tuple[float, dict]] = []
        for sym in symbols:
            fqn = sym.get("fqn") or ""
            file = sym.get("file")
            cand_tokens = _tok(fqn) | _tok(file or "")
            if not cand_tokens:
                continue
            overlap = len(q_tokens & cand_tokens)
            if overlap == 0:
                continue
            # Lexical score = overlap fraction of the query; degree gives a small
            # tie-break boost (capped) so hub symbols rank slightly higher.
            lex = overlap / len(q_tokens)
            degree = sym.get("degree", 0) or 0
            score = lex * (1.0 + 0.01 * min(degree, 50))
            scored.append((score, sym))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Build LocateHits (frozen) for ONLY the top-k, fetching attached
        # memories per hit here — a common query token can overlap hundreds of
        # symbols, so doing anchor lookups before truncation was O(matches)
        # Neo4j round-trips and made locate pathological.
        hits: list[LocateHit] = []
        for score, sym in scored[:k]:
            fqn = sym.get("fqn") or ""
            span = sym.get("span") or "1:1"
            try:
                line = int(str(span).split(":")[0])
            except (ValueError, IndexError):
                line = 0
            mems = self._get_anchor_memories(fqn, user_id=user_id, limit=3)
            hits.append(LocateHit(
                fqn=fqn,
                kind=sym.get("kind"),
                file=sym.get("file"),
                line=line,
                signature=None,
                docstring=None,
                score=score,
                anchor_id=None,
                memories=mems or None,
            ))
        return hits

    def _lexical_code_search(self, m, query: str, query_filter, limit: int) -> list:
        """BM25 lexical search for code_index collection (mirrors memory search).

        Returns empty list if BM25 not available or query fails.
        """
        vs = m.vector_store
        if not query or getattr(vs, "_has_bm25_slot", False) is not True:
            return []
        try:
            sparse = vs._encode_bm25(query)
            if sparse is None:
                return []
            result = vs.client.query_points(
                collection_name="code_index",
                query=sparse,
                using="bm25",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result) or [])
        except Exception:
            logger.debug("BM25 code search failed (non-fatal)", exc_info=True)
            return []

    def _ensure_code_index_collection(
        self, m, *, vector_size: int | None = None, recreate_on_mismatch: bool = False
    ):
        """Create the code_index Qdrant collection if missing (lazy init).

        ``vector_size`` sizes the collection to the ACTIVE code embedder (local
        jina 768 vs cloud Gemini dim); defaults to the memory embedder's dim for
        back-compat. When ``recreate_on_mismatch`` (index path only), an existing
        collection whose dimension no longer matches the active embedder — e.g.
        after switching ``code_embedder`` local↔cloud — is dropped and rebuilt,
        making a mode switch self-healing at reindex time.
        """
        from qdrant_client.models import Distance, VectorParams

        client = m.vector_store.client
        collection_name = "code_index"

        # Resolve the target dim LAZILY: the hot query path (collection exists,
        # no recreate) must not pay an embed("test") call just to confirm the
        # collection is there. Only the create / mismatch-check branches need it.
        _resolved: dict = {"size": vector_size}

        def _size() -> int:
            if _resolved["size"] is None:
                _resolved["size"] = len(
                    m.embedding_model.embed("test", memory_action="add")
                )
            return _resolved["size"]

        try:
            info = client.get_collection(collection_name)
            if not recreate_on_mismatch:
                return  # exists — fast path, no dim computation
            existing = self._collection_vector_size(info)
            want = _size()
            if existing is not None and existing != want:
                logger.info(
                    "code_index dim %s != active embedder dim %s — recreating",
                    existing, want,
                )
                client.delete_collection(collection_name)
            else:
                return  # exists and compatible
        except Exception:
            pass  # doesn't exist (or get failed) → create below

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=_size(),
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created code_index collection (size={_size()})")
        except Exception as e:
            logger.warning(f"Failed to create code_index collection: {e}")
            raise

    @staticmethod
    def _collection_vector_size(info) -> int | None:
        """Best-effort extraction of a Qdrant collection's dense vector size."""
        try:
            vectors = info.config.params.vectors
            size = getattr(vectors, "size", None)
            if size is not None:
                return int(size)
            # Named-vector schema: {name: VectorParams}
            if isinstance(vectors, dict):
                for vp in vectors.values():
                    s = getattr(vp, "size", None)
                    if s is not None:
                        return int(s)
        except Exception:
            return None
        return None

    def blast_radius(
        self,
        symbol: str,
        *,
        max_hops: int = 4,
    ) -> str:
        """Compute blast radius from a given symbol (E7).

        BFS over CALLS/IMPORTS edges to find all symbols affected by changes to
        the given symbol. Returns a text summary (file:line format, consistent
        with locate output).

        Args:
            symbol: FQN or partial match of the epicenter symbol.
            max_hops: Maximum BFS depth (1-16, default 4).

        Returns:
            Text summary of affected symbols (one per line, file:line format).

        Raises:
            ValueError: Symbol not found or max_hops out of range.
        """
        max_hops = max(1, min(int(max_hops), 16))  # clamp to [1, 16]

        # Resolve symbol to FQN via the SAME fuzzy resolver neighbors()/path()
        # use (FQN substring match). Do NOT pass a bare str to _search_symbols
        # — that expects a keyword list and would iterate characters.
        matches = self._find_symbol(symbol)
        if not matches:
            return f"No symbol matching '{symbol}' found in {self.code_space}."

        fqn = matches[0]["fqn"]

        # Run BFS blast radius
        affected_fqns = self._blast_radius_bfs([fqn], max_depth=max_hops)

        if not affected_fqns:
            return f"No blast radius for '{fqn}' (isolated symbol)."

        # Fetch details for all affected symbols
        affected_details = []
        for affected_fqn in sorted(affected_fqns):
            details = self._get_symbol_details(affected_fqn)
            if details:
                affected_details.append(details)

        if not affected_details:
            return f"Blast radius computed ({len(affected_fqns)} symbols) but details unavailable."

        # Format output (file:line, similar to locate)
        lines = [f"Blast radius from '{fqn}' (max_hops={max_hops}): {len(affected_details)} symbols"]
        for detail in affected_details:
            file = detail.get("file", "unknown")
            line = detail.get("line", 0)
            kind = detail.get("kind", "symbol")
            detail_fqn = detail.get("fqn", "")
            lines.append(f"  {file}:{line} [{kind}] {detail_fqn}")

        return "\n".join(lines)

    def _get_symbol_details(self, fqn: str) -> dict | None:
        """Fetch symbol details (file, line, kind) by FQN."""
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $fqn})
        RETURN s.fqn AS fqn, s.file AS file, s.kind AS kind, s.span AS span
        """
        results = self._run_cypher(cypher, code_space=self.code_space, fqn=fqn)
        if not results:
            return None
        row = results[0]
        # Native indexer stores spans as "<start>:<end>" (colon) — see _store_file.
        span = row.get("span") or ""
        line = int(span.split(":")[0]) if ":" in span else 0
        return {
            "fqn": row.get("fqn", ""),
            "file": row.get("file", ""),
            "kind": row.get("kind", ""),
            "line": line,
        }

    def get_symbol_inventory(self) -> set[str]:
        """Get current symbol inventory (canonical FQNs) for liveness tracking.

        Phase E: Used to detect deleted/changed symbols after reindex.
        Returns canonical FQNs (via to_canonical) so the liveness diff is
        engine-agnostic.

        Returns:
            Set of canonical FQNs currently indexed.
        """
        # Fetch all symbols from Neo4j (persisted index)
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        RETURN coalesce(s.canonical_fqn, s.fqn) AS fqn
        """
        results = self._run_cypher(cypher, code_space=self.code_space)

        # Filter out null/falsy fqns — a symbol written before canonical_fqn
        # existed with no raw fqn would otherwise produce a "repo::None" anchor key.
        canonical_fqns = {r["fqn"] for r in results if r.get("fqn")}
        logger.debug("Native symbol inventory: %d canonical FQNs", len(canonical_fqns))
        return canonical_fqns

    def detect_changes(
        self,
        since: str | bytes | None = None,
    ) -> ChangeReport:
        """Blast-radius detection: compare persisted/snapshot index vs fresh parse.

        E5: Compares persisted Neo4j index vs fresh working-tree parse.
        E6: Extends to support snapshot-based comparison — `since` can be snapshot bytes.

        Args:
            since: None (E5: persisted vs fresh), or bytes (E6: snapshot vs current).

        Returns:
            ChangeReport with deleted/modified/added symbols and affected anchors.
        """
        # Determine baseline: persisted or snapshot
        if since is None:
            # E5 path: persisted Neo4j index
            baseline = self._fetch_persisted_symbols()
        elif isinstance(since, bytes):
            # E6 path: snapshot artifact
            baseline = self._parse_snapshot_symbols(since)
        else:
            # Unknown type (git ref support deferred)
            raise ValueError(
                f"detect_changes(since): unsupported type {type(since)}. "
                "E6 supports bytes (snapshot) or None (persisted). Git ref deferred."
            )

        # 2. Parse fresh working tree
        fresh = self._parse_fresh_symbols()

        # 3. Classify changes
        baseline_fqns = {s["fqn"] for s in baseline}
        fresh_fqns = {s["fqn"] for s in fresh}

        deleted_symbols = sorted(baseline_fqns - fresh_fqns)
        added_symbols = sorted(fresh_fqns - baseline_fqns)

        # Modified: signature or body_hash changed
        baseline_map = {s["fqn"]: s for s in baseline}
        fresh_map = {s["fqn"]: s for s in fresh}
        modified_symbols = []
        for fqn in baseline_fqns & fresh_fqns:
            b = baseline_map[fqn]
            f = fresh_map[fqn]
            # Compare body_hash (cheap) or fall back to signature comparison
            if b.get("body_hash") != f.get("body_hash"):
                modified_symbols.append(fqn)

        modified_symbols.sort()

        # 4. Blast-radius BFS: walk CALLS/IMPORTS from deleted+modified symbols
        blast_roots = deleted_symbols + modified_symbols
        affected_fqns = self._blast_radius_bfs(blast_roots, max_depth=3)

        # 5. Collect anchors for affected symbols
        affected_anchors = self._collect_affected_anchors(affected_fqns)

        # 6. Build summary
        source = "snapshot" if isinstance(since, bytes) else "persisted"
        summary = (
            f"Detected {len(deleted_symbols)} deleted, "
            f"{len(modified_symbols)} modified, {len(added_symbols)} added symbols "
            f"(vs {source}). "
            f"Blast radius: {len(affected_fqns)} affected symbols, "
            f"{len(affected_anchors)} anchors flagged."
        )

        return ChangeReport(
            deleted_symbols=deleted_symbols,
            modified_symbols=modified_symbols,
            added_symbols=added_symbols,
            affected_anchors=affected_anchors,
            summary=summary,
        )

    def semantic_layer(self) -> list[SemanticFact]:
        """Semantic distillation from stored degree + community_id properties.

        Returns:
            - One 'module' fact per community (top members listed)
            - One 'hotspot' fact per high-degree symbol (god nodes)

        No NetworkX pass at query time — reads persisted properties only.
        """
        facts: list[SemanticFact] = []

        # Fetch all symbols with their stored properties
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        WHERE s.community_id IS NOT NULL AND s.degree IS NOT NULL
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file,
               s.degree AS degree, s.community_id AS community_id
        ORDER BY s.community_id, s.degree DESC
        """
        symbols = self._run_cypher(cypher, code_space=self.code_space)

        if not symbols:
            logger.info(f"No symbols with community_id for {self.code_space}")
            return facts

        # Group symbols by community
        from collections import defaultdict
        communities: dict[int, list[dict]] = defaultdict(list)
        for sym in symbols:
            cid = sym["community_id"]
            if cid >= 0:  # skip singleton community (-1)
                communities[cid].append(sym)

        # Generate community (module) facts
        # Limit to top N communities by size
        max_communities = getattr(self.settings, "code_graph_max_communities", 10)
        sorted_communities = sorted(
            communities.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:max_communities]

        for cid, members in sorted_communities:
            # Take top members by degree
            top_members = sorted(members, key=lambda x: x["degree"], reverse=True)[:8]
            member_labels = [f"{m['fqn']} ({m['kind']})" for m in top_members]
            key_members = ", ".join(member_labels)

            facts.append(
                SemanticFact(
                    category="module",
                    content=(
                        f"Code community {cid} groups {len(members)} symbols. "
                        f"Key members: {key_members}."
                    ),
                    epistemic_level="inductive",
                    confidence=getattr(self.settings, "code_graph_inferred_confidence", 0.7),
                    external_id=f"community:{cid}",
                    title=f"Community {cid}",
                    tags=["code-structure"],
                )
            )

        # Generate hotspot (god node) facts for high-degree symbols.
        # A "god node" is a highly connected symbol (degree >= 10).
        max_god_nodes = getattr(self.settings, "code_graph_max_god_nodes", 15)
        sorted_by_degree = sorted(symbols, key=lambda x: x["degree"], reverse=True)
        hotspots = [s for s in sorted_by_degree if s["degree"] >= 10][:max_god_nodes]

        for hs in hotspots:
            community_label = (
                f" in community {hs['community_id']}"
                if hs["community_id"] >= 0
                else ""
            )
            facts.append(
                SemanticFact(
                    category="hotspot",
                    content=(
                        f"'{hs['fqn']}' is a highly connected {hs['kind']} with "
                        f"degree {hs['degree']}{community_label}. "
                        f"Source: {hs['file']}."
                    ),
                    # Hotspot status is structure-DERIVED (degree count), not a
                    # directly stated fact — deductive at extracted confidence,
                    # matching semantic.py's _god_node_facts mapping.
                    epistemic_level="deductive",
                    confidence=getattr(self.settings, "code_graph_extracted_confidence", 0.9),
                    external_id=f"symbol:{hs['fqn']}",
                    title=f"Hotspot: {hs['fqn']}",
                    tags=["code-structure", "god-node"],
                )
            )

        logger.info(
            f"Generated {len(facts)} semantic facts for {self.code_space} "
            f"({len(sorted_communities)} communities, {len(hotspots)} hotspots)"
        )
        return facts

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Index a codebase (Python, TypeScript, Go, Rust, Java) into Neo4j code label-space."""
        start = time.time()
        repo_path = Path(source).resolve()
        if not repo_path.is_dir():
            raise ValueError(f"source must be a directory: {source}")

        # Collect source files across supported languages
        file_patterns = {
            "*.py": "python",
            "*.ts": "typescript",
            "*.tsx": "typescript",
            "*.js": "javascript",
            "*.jsx": "javascript",
            "*.go": "go",
            "*.rs": "rust",
            "*.java": "java",
        }
        source_files = []
        for pattern, lang in file_patterns.items():
            for f in repo_path.rglob(pattern):
                source_files.append((f, lang))

        if not source_files:
            logger.info("No source files found in %s", repo_path)
            return IndexReport(
                files_indexed=0,
                symbols_indexed=0,
                edges_indexed=0,
                incremental=incremental,
                duration_s=time.time() - start,
            )

        # Create or verify the CodeRepo node
        self._ensure_repo_node(str(repo_path))

        files_indexed = 0
        symbols_indexed = 0
        edges_indexed = 0

        # Wave 3: when the neighbors resolver is on, collect Python call sites
        # during the parse (into self._pending_call_sites) and the per-file symbol
        # spans, then resolve + store real CALLS edges AFTER all symbols exist
        # (the store MATCHes both endpoints, so cross-file targets must already be
        # in the graph).
        resolver_mode = self._neighbors_resolver_mode()
        self._resolver_collect = resolver_mode in ("jedi", "lsp")
        self._pending_call_sites: dict[str, list[dict]] = {}
        symbols_by_file: dict[str, list[tuple[int, int, str]]] = {}

        # try/finally so a parse/store/resolve failure can never leave
        # _resolver_collect stuck True — that would make later _parse_file callers
        # (e.g. detect_changes) silently collect call sites and suppress legacy
        # CALLS edges (nit-1).
        try:
            for source_file, lang in source_files:
                rel_path = str(source_file.relative_to(repo_path))
                file_hash = self._file_hash(source_file)

                # Incremental: skip unchanged files
                if incremental and self._file_unchanged(rel_path, file_hash):
                    continue

                # Parse and index
                symbols, edges = self._parse_file(source_file, repo_path, lang)
                if symbols or edges:
                    self._store_file(rel_path, file_hash, symbols, edges, lang)
                    files_indexed += 1
                    symbols_indexed += len(symbols)
                    edges_indexed += len(edges)
                if self._resolver_collect and lang == "python":
                    symbols_by_file[rel_path] = [
                        (s.line, s.end_line, s.fqn) for s in symbols
                    ]

            # Wave 3: resolve Python call sites to real symbols and store CALLS
            # edges BEFORE degree/community so both reflect the real call graph.
            if self._resolver_collect:
                edges_indexed += self._resolve_and_store_calls(
                    repo_path, symbols_by_file
                )
        finally:
            self._resolver_collect = False
            self._pending_call_sites = {}

        # Compute and persist degree on all symbols
        self._compute_degrees()

        # I2: Compute and persist Louvain communities
        self._compute_communities()

        # E4: Create CodeAnchor nodes and link symbols to them
        self._ensure_anchors()

        # E3/C3: build symbol cards (card text for BM25 + optional dense vectors).
        dense_degraded = self._index_symbol_cards(repo_path)

        duration = time.time() - start
        logger.info(
            "Indexed %d files, %d symbols, %d edges in %.2fs (code_space=%s)",
            files_indexed, symbols_indexed, edges_indexed, duration, self.code_space,
        )
        return IndexReport(
            files_indexed=files_indexed,
            symbols_indexed=symbols_indexed,
            edges_indexed=edges_indexed,
            incremental=incremental,
            duration_s=duration,
            dense_degraded=dense_degraded,
        )

    def teardown(self) -> dict:
        """R-C: drop this code_space's label-space for a true cold reset.

        Deletes the code GRAPH (CodeRepo/CodeFile/CodeSymbol + their edges) and
        the code_index symbol cards for this code_space, so a subsequent
        ``index()`` is a genuine cold build (not an incremental skip). Scoped
        strictly to ``self.code_space``.

        THE MOAT SURVIVES: this NEVER touches the memory Qdrant collection or any
        Memory/Entity node, and it PRESERVES CodeAnchor nodes (the code-side
        anchor points designed to outlive a symbol reindex — see
        ``_ensure_anchors``). Memory↔code recall joins on the memory's
        ``source_ref`` in Qdrant, which is left fully intact.

        Idempotent: tearing down a non-existent code_space removes 0 and returns
        cleanly.

        Returns:
            {"nodes_deleted": int, "cards_cleared": bool}
        """
        # Label-anchored deletes (one per code label) so Neo4j uses a label scan
        # + code_space predicate — NOT a full-DB node scan that would walk the
        # entire memory graph (Fable SHOULD-FIX). CodeAnchor is deliberately
        # excluded so the moat's code-side anchor points survive.
        nodes_deleted = 0
        for label in ("CodeRepo", "CodeFile", "CodeSymbol"):
            rows = self._run_cypher(
                f"MATCH (n:{label} {{code_space: $code_space}}) "
                f"DETACH DELETE n RETURN count(n) AS deleted",
                code_space=self.code_space,
            )
            nodes_deleted += int(rows[0]["deleted"]) if rows else 0
        cards_cleared = self._delete_code_index_cards()
        # Drop this process's cached BM25 corpus too (MF-1) — the graph is gone.
        from adapters.code_graph.code_locate import invalidate_bm25
        invalidate_bm25(self.code_space)
        logger.info(
            "Native teardown code_space=%s: %d graph nodes deleted, cards_cleared=%s "
            "(CodeAnchor + memory graph preserved)",
            self.code_space, nodes_deleted, cards_cleared,
        )
        return {"nodes_deleted": nodes_deleted, "cards_cleared": cards_cleared}

    def _delete_code_index_cards(self) -> bool:
        """Delete this code_space's symbol cards from the code_index collection.

        Best-effort and scoped by the ``code_space`` payload field (the same key
        locate() filters on). Returns True if the delete was issued, False if the
        collection/store was unavailable (nothing to clear).
        """
        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                FilterSelector,
                MatchValue,
            )

            from memory_service import get_shared_service

            service = get_shared_service()
            m = service._get_memory()
            client = m.vector_store.client
            flt = Filter(
                must=[FieldCondition(key="code_space", match=MatchValue(value=self.code_space))]
            )
            client.delete(
                collection_name="code_index",
                points_selector=FilterSelector(filter=flt),
            )
            return True
        except Exception:  # noqa: BLE001 — no code_index / store down ⇒ nothing to clear
            logger.debug(
                "code_index card delete skipped for %s", self.code_space, exc_info=True
            )
            return False

    def export_snapshot(self) -> bytes:
        """Export code graph snapshot as portable, content-addressed artifact.

        E6: Serializes all :CodeRepo/:CodeFile/:CodeSymbol nodes, all edges, and all
        :CodeAnchor nodes + ANCHORED links for the current code_space. Uses stdlib
        json+gzip for compression. Includes a header with format version, metadata,
        and content hash for verification.

        Returns:
            Compressed snapshot bytes (gzipped JSON).
        """
        import gzip
        import json

        # Extract repo name from code_space
        parts = self.code_space.split("--")
        repo = parts[-1] if len(parts) >= 3 else "unknown"

        # Fetch all nodes
        nodes_cypher = """
        MATCH (n)
        WHERE n.code_space = $code_space
          AND (n:CodeRepo OR n:CodeFile OR n:CodeSymbol OR n:CodeAnchor)
        RETURN labels(n) AS labels, properties(n) AS props
        """
        nodes = self._run_cypher(nodes_cypher, code_space=self.code_space)

        # Fetch all edges between code nodes
        edges_cypher = """
        MATCH (s)-[r]->(t)
        WHERE s.code_space = $code_space
          AND t.code_space = $code_space
          AND (
            (s:CodeSymbol AND t:CodeSymbol AND type(r) IN ['CALLS', 'IMPORTS', 'DEFINES', 'INHERITS', 'REFERENCES'])
            OR (s:CodeSymbol AND t:CodeAnchor AND type(r) = 'ANCHORED')
          )
        RETURN type(r) AS rel_type, properties(r) AS props,
               labels(s) AS source_labels, properties(s) AS source_props,
               labels(t) AS target_labels, properties(t) AS target_props
        """
        edges = self._run_cypher(edges_cypher, code_space=self.code_space)

        # Build snapshot payload
        snapshot = {
            "nodes": [
                {"labels": n["labels"], "properties": n["props"]}
                for n in nodes
            ],
            "edges": [
                {
                    "type": e["rel_type"],
                    "properties": e["props"],
                    "source": {"labels": e["source_labels"], "properties": e["source_props"]},
                    "target": {"labels": e["target_labels"], "properties": e["target_props"]},
                }
                for e in edges
            ],
        }

        # Compute content hash
        snapshot_json = json.dumps(snapshot, sort_keys=True)
        content_hash = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()

        # Build header
        header = {
            "format_version": "1.0",
            "code_space": self.code_space,
            "repo": repo,
            "symbol_count": sum(1 for n in nodes if "CodeSymbol" in n["labels"]),
            "edge_count": len(edges),
            "content_hash": content_hash,
        }

        # Combine header + payload
        envelope = {
            "header": header,
            "snapshot": snapshot,
        }

        # Serialize and compress
        envelope_json = json.dumps(envelope, sort_keys=True)
        compressed = gzip.compress(envelope_json.encode("utf-8"))
        logger.info(
            "Exported snapshot: %d nodes, %d edges, %d bytes (code_space=%s)",
            len(nodes), len(edges), len(compressed), self.code_space,
        )
        return compressed

    def import_snapshot(self, data: bytes):
        """Import a code graph snapshot into Neo4j (CI-built index → deployment).

        E6: Rebuilds the code label-space for the snapshot's code_space by MERGE-ing
        all nodes and edges. Idempotent (re-import produces no duplicates). Uses
        the graph_patcher-style deadlock retry pattern.

        Args:
            data: Compressed snapshot bytes (from export_snapshot).
        """
        import gzip
        import json

        # Decompress and parse
        decompressed = gzip.decompress(data)
        envelope = json.loads(decompressed.decode("utf-8"))

        header = envelope["header"]
        snapshot = envelope["snapshot"]

        # Verify format version
        if header["format_version"] != "1.0":
            raise ValueError(f"Unsupported snapshot format: {header['format_version']}")

        # Verify content hash
        snapshot_json = json.dumps(snapshot, sort_keys=True)
        computed_hash = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
        if computed_hash != header["content_hash"]:
            raise ValueError(
                f"Snapshot corrupted: content_hash mismatch "
                f"(expected {header['content_hash']}, got {computed_hash})"
            )

        logger.info(
            "Importing snapshot: %d nodes, %d edges (code_space=%s)",
            len(snapshot["nodes"]), len(snapshot["edges"]), header["code_space"]
        )

        # MERGE nodes (order: CodeRepo → CodeFile → CodeSymbol → CodeAnchor)
        # to respect foreign-key-like dependencies
        node_order = ["CodeRepo", "CodeFile", "CodeSymbol", "CodeAnchor"]
        for label_filter in node_order:
            nodes_to_merge = [
                n for n in snapshot["nodes"] if label_filter in n["labels"]
            ]
            for node in nodes_to_merge:
                self._merge_node(node["labels"], node["properties"])

        # MERGE edges
        for edge in snapshot["edges"]:
            self._merge_edge(
                edge["source"]["labels"],
                edge["source"]["properties"],
                edge["type"],
                edge["target"]["labels"],
                edge["target"]["properties"],
                edge["properties"],
            )

        logger.info("Snapshot import complete (code_space=%s)", header["code_space"])

    def _merge_node(self, labels: list[str], props: dict):
        """MERGE a node by its primary key (code_space + label-specific key).

        Uses deadlock retry pattern.
        """
        # Determine primary key based on label
        label = labels[0]  # First label is the primary type
        if label == "CodeRepo":
            match_key = "code_space"
        elif label == "CodeFile":
            match_key = "code_space, path"
        elif label == "CodeSymbol":
            match_key = "code_space, fqn"
        elif label == "CodeAnchor":
            match_key = "code_space, repo, fqn"
        else:
            logger.warning(f"Unknown label for merge: {label}")
            return

        # Build MERGE cypher (SET all properties)
        label_str = ":".join(labels)
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in props.keys())
        cypher = f"""
        MERGE (n:{label_str} {{{match_key.replace(", ", ": $")}: ${match_key.replace(", ", ", ")}$}})
        SET {set_clauses}
        """
        # Clean up the match clause to use actual keys
        if label == "CodeRepo":
            cypher = f"MERGE (n:{label_str} {{code_space: $code_space}}) SET {set_clauses}"
        elif label == "CodeFile":
            cypher = f"MERGE (n:{label_str} {{code_space: $code_space, path: $path}}) SET {set_clauses}"
        elif label == "CodeSymbol":
            cypher = f"MERGE (n:{label_str} {{code_space: $code_space, fqn: $fqn}}) SET {set_clauses}"
        elif label == "CodeAnchor":
            cypher = f"MERGE (n:{label_str} {{code_space: $code_space, repo: $repo, fqn: $fqn}}) SET {set_clauses}"

        self._run_cypher_with_retry(cypher, **props)

    def _merge_edge(
        self,
        source_labels: list[str],
        source_props: dict,
        rel_type: str,
        target_labels: list[str],
        target_props: dict,
        edge_props: dict,
    ):
        """MERGE an edge between two nodes (deadlock retry).

        Resolves source and target by their primary keys, then creates/updates the edge.
        """
        # Build match predicates for source and target
        src_label = source_labels[0]
        tgt_label = target_labels[0]

        # Determine match keys
        src_match = self._build_match_predicate(src_label, source_props)
        tgt_match = self._build_match_predicate(tgt_label, target_props)

        # Build edge SET clause
        set_clause = (
            ", ".join(f"r.{k} = ${k}" for k in edge_props.keys())
            if edge_props
            else ""
        )
        set_part = f"SET {set_clause}" if set_clause else ""

        cypher = f"""
        MATCH (s:{src_label} {src_match})
        MATCH (t:{tgt_label} {tgt_match})
        MERGE (s)-[r:{rel_type}]->(t)
        {set_part}
        """

        # Merge all props (source, target, edge)
        all_props = {**source_props, **target_props, **edge_props}
        # Prefix source/target props to avoid collisions
        params = {}
        for k, v in source_props.items():
            params[f"src_{k}"] = v
        for k, v in target_props.items():
            params[f"tgt_{k}"] = v
        params.update(edge_props)

        # Rebuild cypher with prefixed params
        src_match_prefixed = self._build_match_predicate(src_label, source_props, prefix="src_")
        tgt_match_prefixed = self._build_match_predicate(tgt_label, target_props, prefix="tgt_")
        cypher = f"""
        MATCH (s:{src_label} {src_match_prefixed})
        MATCH (t:{tgt_label} {tgt_match_prefixed})
        MERGE (s)-[r:{rel_type}]->(t)
        {set_part}
        """

        self._run_cypher_with_retry(cypher, **params)

    def _build_match_predicate(self, label: str, props: dict, prefix: str = "") -> str:
        """Build a Cypher match predicate for a node by its primary key."""
        if label == "CodeRepo":
            return f"{{code_space: ${prefix}code_space}}"
        elif label == "CodeFile":
            return f"{{code_space: ${prefix}code_space, path: ${prefix}path}}"
        elif label == "CodeSymbol":
            return f"{{code_space: ${prefix}code_space, fqn: ${prefix}fqn}}"
        elif label == "CodeAnchor":
            return f"{{code_space: ${prefix}code_space, repo: ${prefix}repo, fqn: ${prefix}fqn}}"
        else:
            # Fallback: use code_space only
            return f"{{code_space: ${prefix}code_space}}"

    # ── Internal indexing helpers ────────────────────────────────────

    def _file_hash(self, path: Path) -> str:
        """SHA256 content hash of a file."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _file_unchanged(self, rel_path: str, file_hash: str) -> bool:
        """Check if a file's hash matches the indexed version."""
        cypher = """
        MATCH (f:CodeFile {code_space: $code_space, path: $path})
        RETURN f.hash = $hash AS unchanged
        """
        result = self._run_cypher(cypher, code_space=self.code_space, path=rel_path, hash=file_hash)
        return result[0]["unchanged"] if result else False

    def _parse_file(self, source_file: Path, repo_root: Path, language: str) -> tuple[list[_Symbol], list[_Edge]]:
        """Parse a source file with tree-sitter and extract symbols/edges.

        Args:
            source_file: Path to the source file.
            repo_root: Repository root (for relative paths).
            language: Language name (python, typescript, javascript, go, rust, java).

        Returns:
            (symbols, edges) tuples.
        """
        try:
            from tree_sitter import Language, Parser
        except ImportError:
            raise RuntimeError(
                "tree-sitter dependencies not installed. "
                "Install with: uv sync --extra code-graph"
            ) from None

        # Get the language parser
        ts_lang = self._get_tree_sitter_language(language)
        if ts_lang is None:
            # Language not available, skip
            logger.debug(f"tree-sitter language not available for {language}")
            return ([], [])

        parser = Parser(ts_lang)
        source_bytes = source_file.read_bytes()
        tree = parser.parse(source_bytes)
        root = tree.root_node

        # Build module path from file path
        rel_path = str(source_file.relative_to(repo_root))
        module_path = self._build_module_path(rel_path, language)

        symbols: list[_Symbol] = []
        edges: list[_Edge] = []

        # Delegate to language-specific parser
        if language == "python":
            symbols, edges = self._parse_python(root, rel_path, module_path, source_bytes)
        else:
            # For E3: simple heuristic extraction for other languages
            # (tree-sitter query API would be more robust but not required for E3)
            symbols, edges = self._parse_generic(root, rel_path, module_path, language, source_bytes)

        return symbols, edges

    def _get_tree_sitter_language(self, language: str):
        """Get the tree-sitter Language object for a given language name."""
        try:
            from tree_sitter import Language
            if language == "python":
                import tree_sitter_python
                return Language(tree_sitter_python.language())
            elif language in ("typescript", "tsx"):
                import tree_sitter_typescript
                return Language(tree_sitter_typescript.language_typescript())
            elif language in ("javascript", "jsx"):
                import tree_sitter_javascript
                return Language(tree_sitter_javascript.language())
            elif language == "go":
                import tree_sitter_go
                return Language(tree_sitter_go.language())
            elif language == "rust":
                import tree_sitter_rust
                return Language(tree_sitter_rust.language())
            elif language == "java":
                import tree_sitter_java
                return Language(tree_sitter_java.language())
        except ImportError:
            logger.debug(f"tree-sitter parser for {language} not installed")
            return None
        return None

    def _build_module_path(self, rel_path: str, language: str) -> str:
        """Build a module/package path from a file path."""
        if language == "python":
            return rel_path.replace("/", ".").removesuffix(".py")
        elif language in ("typescript", "javascript"):
            return rel_path.replace("/", ".").removesuffix(".ts").removesuffix(".tsx").removesuffix(".js").removesuffix(".jsx")
        elif language == "go":
            # Go uses directory-based packages
            return str(Path(rel_path).parent).replace("/", ".")
        elif language == "rust":
            # Rust uses crate::module paths
            return rel_path.replace("/", "::").removesuffix(".rs")
        elif language == "java":
            # Java uses package.Class
            return rel_path.replace("/", ".").removesuffix(".java")
        return rel_path

    def _parse_python(self, root, rel_path: str, module_path: str, source_bytes: bytes) -> tuple[list[_Symbol], list[_Edge]]:
        """Parse Python AST.

        Wave 3: when the neighbors resolver is on (``_resolver_collect``), the walk
        tracks the ENCLOSING function/method of each call and records the call site
        (position + real source FQN) into ``self._pending_call_sites`` for
        Jedi resolution after all files are indexed, and does NOT mint the legacy
        phantom ``{module}.{rawtext}`` CALLS edge (which _store_file would drop
        anyway). Resolver off ⇒ the exact E2 heuristic behavior is preserved.
        """
        symbols: list[_Symbol] = []
        edges: list[_Edge] = []
        collect_calls = getattr(self, "_resolver_collect", False)
        call_sink = (
            self._pending_call_sites.setdefault(rel_path, [])
            if collect_calls else None
        )

        def _callee_name_node(func_node):
            # The token to resolve: the final identifier of the callee. For
            # ``a.b.c(...)`` that's ``c`` (the attribute); for ``foo(...)`` it's
            # ``foo`` itself.
            if func_node.type == "attribute":
                attr = func_node.child_by_field_name("attribute")
                return attr if attr is not None else func_node
            return func_node

        def walk(node, parent_class=None, enclosing_func=None):
            """Recursive tree walker."""
            node_type = node.type

            # Function definitions
            if node_type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = name_node.text.decode("utf8")
                    if parent_class:
                        # Method inside a class
                        fqn = f"{parent_class}.{func_name}"
                        kind = "method"
                        edges.append(_Edge(
                            source_fqn=parent_class,
                            target_fqn=fqn,
                            relation="DEFINES",
                            extraction="extracted",
                        ))
                    else:
                        # Top-level function
                        fqn = f"{module_path}.{func_name}"
                        kind = "function"

                    symbols.append(_Symbol(
                        fqn=fqn,
                        kind=kind,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
                    # Only the BODY (the `block` child) runs inside this function;
                    # the parameter list (default-arg expressions) and return
                    # annotation evaluate in the ENCLOSING scope at def time, so
                    # their calls attribute outward — not to this function
                    # (Copilot). Decorators are a sibling `decorated_definition`
                    # node, already outside. (tree-sitter wrapper nodes aren't
                    # identity-comparable, so match the body by type.)
                    for child in node.children:
                        scope = fqn if child.type == "block" else enclosing_func
                        walk(child, parent_class=parent_class, enclosing_func=scope)
                    return

            # Class definitions
            elif node_type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = name_node.text.decode("utf8")
                    fqn = f"{module_path}.{class_name}"
                    symbols.append(_Symbol(
                        fqn=fqn,
                        kind="class",
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
                    # Recursively walk the class body with this class as parent
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            walk(child, parent_class=fqn)
                    return  # Don't continue walking children (already walked body)

            # Import statements
            elif node_type == "import_statement":
                # Extract dotted_name from import statement
                for child in node.children:
                    if child.type == "dotted_name":
                        imported = child.text.decode("utf8")
                        edges.append(_Edge(
                            source_fqn=module_path,
                            target_fqn=imported,
                            relation="IMPORTS",
                            extraction="extracted",
                        ))

            elif node_type == "import_from_statement":
                # from X import Y
                module_node = node.child_by_field_name("module_name")
                if module_node:
                    imported = module_node.text.decode("utf8")
                    edges.append(_Edge(
                        source_fqn=module_path,
                        target_fqn=imported,
                        relation="IMPORTS",
                        extraction="extracted",
                    ))

            # Call expressions
            elif node_type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    if collect_calls:
                        # Record the call site for Jedi resolution — only when it
                        # sits inside a real function/method, so the CALLS edge has
                        # a real symbol source (module-level calls can't attach).
                        if enclosing_func is not None:
                            name_node = _callee_name_node(func_node)
                            sp = name_node.start_point
                            call_sink.append({
                                "line": sp[0] + 1,   # Jedi: 1-based line
                                "col": sp[1],        # Jedi: 0-based column
                                "src_fqn": enclosing_func,
                            })
                    else:
                        # Legacy heuristic (resolver off): best-effort phantom FQN.
                        target_name = func_node.text.decode("utf8")
                        edges.append(_Edge(
                            source_fqn=module_path,
                            target_fqn=f"{module_path}.{target_name}",
                            relation="CALLS",
                            extraction="inferred",
                        ))

            # Recurse to children (unless we already handled them above)
            for child in node.children:
                walk(child, parent_class=parent_class, enclosing_func=enclosing_func)

        walk(root)
        return symbols, edges

    def _parse_generic(self, root, rel_path: str, module_path: str, language: str, source_bytes: bytes) -> tuple[list[_Symbol], list[_Edge]]:
        """Generic heuristic parser for non-Python languages (E3).

        Simple tree-walk extraction: function/class/method definitions and call sites.
        Not LSP-grade but sufficient for E3's locate use case.
        """
        symbols: list[_Symbol] = []
        edges: list[_Edge] = []

        def walk(node, parent_class=None):
            node_type = node.type

            # Functions/methods (language-specific patterns)
            if node_type in ("function_declaration", "method_definition", "method_declaration"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf8", errors="ignore")
                    if parent_class:
                        fqn = f"{parent_class}.{name}"
                        kind = "method"
                    else:
                        fqn = f"{module_path}.{name}" if module_path else name
                        kind = "function"
                    symbols.append(_Symbol(
                        fqn=fqn,
                        kind=kind,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))

            # Classes/structs/interfaces
            elif node_type in ("class_declaration", "struct_item", "interface_declaration", "type_declaration"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf8", errors="ignore")
                    fqn = f"{module_path}.{name}" if module_path else name
                    kind = "class" if "class" in node_type else "type"
                    symbols.append(_Symbol(
                        fqn=fqn,
                        kind=kind,
                        file=rel_path,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
                    # Recurse into body to find methods
                    for child in node.children:
                        walk(child, parent_class=fqn)
                    return

            # Imports (best-effort)
            elif node_type in ("import_statement", "import_declaration", "use_declaration"):
                # Extract imported name (varies by language)
                imported = node.text.decode("utf8", errors="ignore")
                if len(imported) < 200:  # sanity check
                    edges.append(_Edge(
                        source_fqn=module_path or rel_path,
                        target_fqn=imported,
                        relation="IMPORTS",
                        extraction="extracted",
                    ))

            # Calls (inferred)
            elif node_type in ("call_expression", "method_invocation"):
                func_node = node.child_by_field_name("function")
                if func_node:
                    target = func_node.text.decode("utf8", errors="ignore")
                    if len(target) < 100:
                        edges.append(_Edge(
                            source_fqn=module_path or rel_path,
                            target_fqn=target,
                            relation="CALLS",
                            extraction="inferred",
                        ))

            # Recurse
            for child in node.children:
                walk(child, parent_class=parent_class)

        walk(root)
        return symbols, edges

    def _ensure_repo_node(self, path: str):
        """Create or update the (:CodeRepo) node."""
        cypher = """
        MERGE (r:CodeRepo {code_space: $code_space})
        SET r.name = $name, r.path = $path
        """
        repo_name = Path(path).name
        self._run_cypher(cypher, code_space=self.code_space, name=repo_name, path=path)

    def _store_file(
        self,
        rel_path: str,
        file_hash: str,
        symbols: list[_Symbol],
        edges: list[_Edge],
        language: str = "python",
    ):
        """Store a file + its symbols + edges in Neo4j (with deadlock retry)."""
        # Store the file node
        file_cypher = """
        MERGE (f:CodeFile {code_space: $code_space, path: $path})
        SET f.hash = $hash, f.language = $language
        """
        self._run_cypher_with_retry(
            file_cypher,
            code_space=self.code_space,
            path=rel_path,
            hash=file_hash,
            language=language,
        )

        # Store symbols in batches (E5: persist body_hash for change detection)
        # Fix 1: UNWIND-batched writes instead of per-symbol round-trips
        if symbols:
            _BATCH_SIZE = 500
            symbol_rows = []
            for sym in symbols:
                # Compute body hash from the symbol's source range
                body_hash = self._compute_symbol_body_hash(rel_path, sym)
                span = f"{sym.line}:{sym.end_line}"
                symbol_rows.append({
                    "fqn": sym.fqn,
                    # Phase C: persist the engine-agnostic canonical FQN alongside
                    # the raw fqn so anchors can be keyed on it end-to-end (create
                    # and lookup both use the SAME canonical key). See _ensure_anchors.
                    "canonical_fqn": self.to_canonical(sym.fqn),
                    "kind": sym.kind,
                    "file": sym.file,
                    "span": span,
                    "body_hash": body_hash,
                })

            # Batch symbols into chunks and write
            for i in range(0, len(symbol_rows), _BATCH_SIZE):
                batch = symbol_rows[i : i + _BATCH_SIZE]
                sym_cypher = """
                UNWIND $rows AS row
                MERGE (s:CodeSymbol {code_space: $code_space, fqn: row.fqn})
                SET s.kind = row.kind, s.file = row.file, s.span = row.span,
                    s.body_hash = row.body_hash, s.canonical_fqn = row.canonical_fqn
                """
                self._run_cypher_with_retry(
                    sym_cypher,
                    code_space=self.code_space,
                    rows=batch,
                )

        # Store edges in batches, grouped by relation type
        # Fix 2: MATCH both endpoints (only link edges where both symbols exist)
        if edges:
            _BATCH_SIZE = 500
            # Group edges by relation type (dynamic relation type requires grouping)
            from collections import defaultdict
            edges_by_relation = defaultdict(list)
            for edge in edges:
                epistemic = self._extraction_to_epistemic(edge.extraction)
                edges_by_relation[edge.relation].append({
                    "src_fqn": edge.source_fqn,
                    "tgt_fqn": edge.target_fqn,
                    "extraction": edge.extraction,
                    "epistemic": epistemic,
                })

            # Batch-write each relation type
            for relation, edge_list in edges_by_relation.items():
                for i in range(0, len(edge_list), _BATCH_SIZE):
                    batch = edge_list[i : i + _BATCH_SIZE]
                    edge_cypher = """
                    UNWIND $rows AS row
                    MATCH (src:CodeSymbol {code_space: $code_space, fqn: row.src_fqn})
                    MATCH (tgt:CodeSymbol {code_space: $code_space, fqn: row.tgt_fqn})
                    MERGE (src)-[r:%s]->(tgt)
                    SET r.extraction = row.extraction, r.epistemic_level = row.epistemic
                    """ % relation
                    self._run_cypher_with_retry(
                        edge_cypher,
                        code_space=self.code_space,
                        rows=batch,
                    )

    def _extraction_to_epistemic(self, extraction: str) -> str:
        """Map extraction confidence to epistemic level (mirrors semantic.py)."""
        if extraction == "extracted":
            return "explicit"
        elif extraction in ("inferred", "ambiguous"):
            return "deductive"
        return "deductive"  # fallback

    # ── Wave 3: neighbors call resolution (Jedi) ─────────────────────

    def _resolve_and_store_calls(
        self, repo_path: Path, symbols_by_file: dict[str, list[tuple[int, int, str]]]
    ) -> int:
        """Resolve collected Python call sites to real symbols and store CALLS
        edges. Returns the number of distinct resolved edges stored.

        Best-effort: if Jedi is unavailable the neighbors graph simply stays as it
        was (no phantom edges were minted), never failing the index.
        """
        import time as _time

        sites_by_file = getattr(self, "_pending_call_sites", {})
        # MF-1: every re-parsed Python file cleared its prior resolver='jedi' CALLS
        # so a call REMOVED by an edit doesn't linger — otherwise the resolved
        # graph could only grow and the neighbors count would monotonically
        # over-inflate under incremental reindex (a meter-honesty failure). Runs
        # BEFORE the early return: a file whose calls were all deleted still needs
        # clearing. sites_by_file keys every parsed Python file (setdefault in the
        # walk), so its keys are exactly the re-parse set.
        self._delete_stale_resolved_calls(list(sites_by_file.keys()))

        total_sites = sum(len(v) for v in sites_by_file.values())
        if not total_sites:
            return 0
        resolver, provenance = self._build_call_resolver(repo_path)
        if resolver is None:
            logger.warning(
                "no call resolver available — neighbors stays heuristic (~0 CALLS)",
            )
            return 0

        _t0 = _time.time()
        resolved: list[dict] = []
        resolved_count = 0
        for rel_path, sites in sites_by_file.items():
            abs_path = repo_path / rel_path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            defs = resolver.resolve_file(
                abs_path, source, [(s["line"], s["col"]) for s in sites]
            )
            for site, (def_path, def_line) in zip(sites, defs):
                if not def_path or def_line is None:
                    continue  # unresolved (stdlib/external/dynamic) → dropped
                tgt = self._map_def_to_fqn(
                    def_path, def_line, repo_path, symbols_by_file
                )
                if not tgt:
                    continue  # resolved outside the repo's indexed symbols
                src = site["src_fqn"]
                if src == tgt:
                    continue  # skip trivial self-recursion self-loops
                resolved_count += 1
                resolved.append({"src_fqn": src, "tgt_fqn": tgt})

        # Dedup (src, tgt) — many call sites share the same edge.
        seen: set[tuple[str, str]] = set()
        uniq: list[dict] = []
        for e in resolved:
            key = (e["src_fqn"], e["tgt_fqn"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)

        self._store_resolved_call_edges(uniq, provenance)
        logger.info(
            "%s neighbors resolver: %d/%d call sites resolved to in-repo "
            "symbols → %d distinct CALLS edges (%d files) in %.2fs",
            provenance, resolved_count, total_sites, len(uniq), len(sites_by_file),
            _time.time() - _t0,
        )
        return len(uniq)

    def _build_call_resolver(self, repo_path: Path):
        """Select the neighbors call resolver, returning ``(resolver, provenance)``.

        - ``lsp``: the external pyright resolver service (resolver-svc). Probed
          first; if unreachable we transparently fall back to in-process Jedi so a
          down service degrades gracefully rather than dropping every edge (the
          brief keeps Jedi as the fallback).
        - ``jedi``: in-process Jedi.

        Returns ``(None, None)`` only when neither resolver can be constructed.
        The ``provenance`` string tags stored CALLS edges (``r.resolver``) so the
        stale-edge cleanup and the meter reflect what actually resolved.
        """
        mode = self._neighbors_resolver_mode()
        if mode == "lsp":
            try:
                from adapters.code_graph.code_resolve_lsp import LspCallResolver

                url = getattr(
                    self.settings, "code_resolver_url", "http://resolver-svc:8201"
                )
                resolver = LspCallResolver(repo_path, url)
                resolver.health()  # raises if the service is down/unhealthy
                logger.info("neighbors resolver: using pyright service at %s", url)
                return resolver, "lsp"
            except Exception:
                logger.warning(
                    "LSP resolver unavailable — falling back to in-process Jedi",
                    exc_info=True,
                )
                # fall through to Jedi

        try:
            import jedi  # noqa: F401  — probe so a truly-missing dep warns here
            from adapters.code_graph.code_resolve import JediCallResolver

            return JediCallResolver(repo_path), "jedi"
        except Exception:
            logger.warning("jedi resolver unavailable", exc_info=True)
            return None, None

    def _delete_stale_resolved_calls(self, files: list[str]) -> None:
        """Delete any resolver-produced CALLS edge (r.resolver set — 'jedi' or
        'lsp') whose SOURCE symbol lives in one of the re-parsed ``files``, so a
        call removed by an edit doesn't survive an incremental reindex (MF-1: keeps
        the resolved graph — and the neighbors count on the meter — from
        monotonically over-inflating). Matching ANY resolver (not just the active
        one) also means switching resolvers, e.g. jedi→lsp, cleanly replaces the
        prior resolver's edges for the re-parsed files instead of double-counting."""
        if not files:
            return
        try:
            self._run_cypher_with_retry(
                "MATCH (src:CodeSymbol {code_space: $cs})-[r:CALLS]->() "
                "WHERE src.file IN $files AND r.resolver IS NOT NULL DELETE r",
                cs=self.code_space, files=files,
            )
        except Exception:
            logger.debug("stale resolved-CALLS cleanup failed (non-fatal)", exc_info=True)

    def _map_def_to_fqn(
        self,
        def_abs_path: str,
        def_line: int,
        repo_root: Path,
        symbols_by_file: dict[str, list[tuple[int, int, str]]],
    ) -> str | None:
        """Map a Jedi definition ``(abs_path, line)`` to the EXACT stored symbol
        FQN whose span contains that line — so a resolved target always matches an
        existing :CodeSymbol (never a new phantom). Prefers the innermost span
        (largest start). Falls back to a Neo4j lookup for files not in
        ``symbols_by_file`` (incremental reindex: unchanged files weren't parsed)."""
        try:
            rel = str(Path(def_abs_path).resolve().relative_to(Path(repo_root).resolve()))
        except Exception:
            return None  # definition lives outside the repo → external, drop
        spans = symbols_by_file.get(rel)
        if spans is None:
            # Incremental: target lives in an unchanged (unparsed) file. Load its
            # spans from Neo4j once and cache back into symbols_by_file so many
            # cross-file calls to the same file don't re-query (nit-3).
            spans = self._symbol_spans_for_file(rel)
            symbols_by_file[rel] = spans
        best: tuple[int, int, str] | None = None
        for start, end, fqn in spans:
            hi = end or start
            if start <= def_line <= hi and (best is None or start > best[0]):
                best = (start, hi, fqn)
        return best[2] if best else None

    def _symbol_spans_for_file(self, rel_path: str) -> list[tuple[int, int, str]]:
        """Load (start_line, end_line, fqn) for every symbol in a file from Neo4j
        (incremental fallback for _map_def_to_fqn)."""
        try:
            rows = self._run_cypher(
                "MATCH (s:CodeSymbol {code_space: $cs, file: $file}) "
                "RETURN s.fqn AS fqn, s.span AS span",
                cs=self.code_space, file=rel_path,
            )
        except Exception:
            return []
        spans: list[tuple[int, int, str]] = []
        for r in rows:
            span = str(r.get("span") or "0:0")
            parts = span.split(":")
            try:
                start = int(parts[0])
                end = int(parts[1]) if len(parts) > 1 else start
            except (ValueError, IndexError):
                continue
            spans.append((start, end, r.get("fqn") or ""))
        return spans

    def _store_resolved_call_edges(
        self, edges: list[dict], provenance: str = "jedi"
    ) -> None:
        """Store resolved CALLS edges, MATCHing both endpoints (both symbols exist
        by now). extraction='extracted' (statically resolved → epistemic explicit);
        r.resolver=<provenance> ('jedi' | 'lsp') marks which resolver produced it."""
        if not edges:
            return
        _BATCH = 500
        for i in range(0, len(edges), _BATCH):
            batch = edges[i : i + _BATCH]
            cypher = """
            UNWIND $rows AS row
            MATCH (src:CodeSymbol {code_space: $cs, fqn: row.src_fqn})
            MATCH (tgt:CodeSymbol {code_space: $cs, fqn: row.tgt_fqn})
            MERGE (src)-[r:CALLS]->(tgt)
            SET r.extraction = 'extracted', r.epistemic_level = 'explicit',
                r.resolver = $provenance
            """
            self._run_cypher_with_retry(
                cypher, cs=self.code_space, rows=batch, provenance=provenance
            )

    def _compute_degrees(self):
        """Compute in+out degree for all symbols and persist it."""
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        OPTIONAL MATCH (s)-[r]->()
        WITH s, count(r) AS out_deg
        OPTIONAL MATCH (s)<-[r]-()
        WITH s, out_deg, count(r) AS in_deg
        SET s.degree = out_deg + in_deg
        """
        self._run_cypher(cypher, code_space=self.code_space)

    def _compute_communities(self):
        """Compute Louvain communities at index time and persist community_id.

        Reads CALLS + IMPORTS edges, builds an undirected graph, runs Louvain
        with seed=42 for determinism, and persists stable community_id on each
        :CodeSymbol. Guards against graphs > 200k edges (logs warning and skips).

        Isolated symbols (no CALLS/IMPORTS edges) get community_id = -1 (singleton).
        """
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        # 1. Fetch CALLS + IMPORTS edges for this code_space
        edge_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})-[r]->(t:CodeSymbol)
        WHERE type(r) IN ['CALLS', 'IMPORTS']
        RETURN s.fqn AS source, t.fqn AS target
        """
        edges = self._run_cypher(edge_cypher, code_space=self.code_space)

        # 2. Size guard: if > 200k edges, skip with warning (leave community_id unset)
        if len(edges) > 200_000:
            logger.warning(
                f"Skipping community computation for {self.code_space}: "
                f"{len(edges)} edges exceeds 200k limit"
            )
            return

        # 3. Build undirected networkx graph
        G = nx.Graph()
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])

        # 4. No CALLS/IMPORTS graph: every symbol is a singleton. Still populate
        #    community_id = -1 on all symbols so the property is never unset
        #    (semantic_layer() filters on community_id IS NOT NULL).
        if G.number_of_nodes() == 0:
            logger.info(
                f"No CALLS/IMPORTS graph for {self.code_space}; "
                f"assigning singleton community_id=-1 to all symbols"
            )
            self._persist_singleton_communities()
            return

        # 5. Run Louvain with deterministic seed
        communities = louvain_communities(G, seed=42)

        # 6. Build stable community id mapping (sort communities by min fqn)
        # This ensures same input graph => same community_id assignment
        sorted_communities = sorted(communities, key=lambda c: min(c))
        fqn_to_community_id = {}
        for idx, community in enumerate(sorted_communities):
            for fqn in community:
                fqn_to_community_id[fqn] = idx

        # 7. Fetch all symbols to find isolated ones (not in the graph)
        all_symbols_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        RETURN s.fqn AS fqn
        """
        all_symbols = self._run_cypher(all_symbols_cypher, code_space=self.code_space)

        # 8. Assign community_id = -1 to isolated symbols
        for symbol in all_symbols:
            fqn = symbol["fqn"]
            if fqn not in fqn_to_community_id:
                fqn_to_community_id[fqn] = -1

        # 9. Persist community_id via batched UNWIND SET (retry for transient errors)
        if fqn_to_community_id:
            batch_data = [
                {"fqn": fqn, "community_id": cid}
                for fqn, cid in fqn_to_community_id.items()
            ]
            update_cypher = """
            UNWIND $batch AS row
            MATCH (s:CodeSymbol {code_space: $code_space, fqn: row.fqn})
            SET s.community_id = row.community_id
            """
            self._run_cypher_with_retry(
                update_cypher,
                code_space=self.code_space,
                batch=batch_data,
            )
            logger.info(
                f"Persisted {len(sorted_communities)} communities "
                f"({len(fqn_to_community_id)} symbols) for {self.code_space}"
            )

    def _persist_singleton_communities(self):
        """Assign community_id = -1 to every symbol in the code_space.

        Used when there are no CALLS/IMPORTS edges to cluster on — every symbol
        is its own singleton. Batched UNWIND SET (retry for transient errors).
        """
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        SET s.community_id = -1
        """
        self._run_cypher_with_retry(cypher, code_space=self.code_space)

    def _ensure_anchors(self):
        """E4: Create CodeAnchor nodes and link symbols to them.

        MERGE anchors keyed by (code_space, repo, CANONICAL FQN) and create
        (:CodeSymbol)-[:ANCHORED]->(:CodeAnchor) edges. Anchors survive symbol
        reindexes — symbols are deleted/recreated, anchors persist, and the
        ANCHORED edges are recreated pointing to the same anchor nodes.

        Phase C (anchor moat): the anchor's ``fqn`` is the CANONICAL FQN (src/lib
        stripped), computed at symbol-write time and persisted as
        ``s.canonical_fqn`` (see _store_file). This is END-TO-END consistent with
        _get_anchor_memories, which canonicalizes the incoming FQN before building
        its lookup key — so create and lookup produce the SAME key, and the
        cross-engine anchor join hits regardless of which engine (native/CBM/
        graphify) produced the answer.

        MIGRATION NOTE: pre-existing raw-keyed anchors (there are none on ice/v2
        — every index here is fresh) would need a one-time rekey to canonical when
        this reaches `dev`. That migration is a documented follow-up, not this PR.

        Uses deadlock retry pattern per graph_patcher.py.
        """
        # Extract repo name from code_space: "code--{owner}--{repo}"
        parts = self.code_space.split("--")
        repo = parts[-1] if len(parts) >= 3 else "unknown"

        # Key the anchor on the persisted canonical FQN. coalesce() guards the
        # (transient) case of a symbol written before this field existed.
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        MERGE (a:CodeAnchor {
            code_space: $code_space, repo: $repo,
            fqn: coalesce(s.canonical_fqn, s.fqn)
        })
        MERGE (s)-[:ANCHORED]->(a)
        RETURN count(a) AS anchored
        """
        result = self._run_cypher_with_retry(
            cypher,
            code_space=self.code_space,
            repo=repo,
        )
        count = result[0]["anchored"] if result else 0
        logger.info(f"Ensured {count} CodeAnchors (canonical FQN) for {self.code_space}")

    def _get_anchor_memories(
        self,
        fqn: str,
        user_id: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        """E4: Fetch memories attached to a code anchor by (repo, canonical FQN).

        Phase E: Refactored to call the batched lookup with a single FQN, keeping
        the single-symbol method signature working for backward compatibility.

        Phase C: Uses CANONICAL FQN (src/lib stripped) so memories anchored to
        a symbol are retrievable regardless of which engine indexed it.

        Respects visibility: returns only memories the caller may read
        (their own private memories + shared/standard pools).

        Args:
            fqn: The FQN from the engine (will be canonicalized).
            user_id: Caller user ID for visibility scoping.
            limit: Max memories to return.

        Returns list of dicts: [{id, content, category, visibility, ...}, ...]
        """
        # Extract repo from code_space
        parts = self.code_space.split("--")
        repo = parts[-1] if len(parts) >= 3 else "unknown"

        # Phase E: delegate to batched lookup with a single FQN
        from knowledge.fusion import batched_anchor_lookup

        result_by_fqn = batched_anchor_lookup(
            fqns=[fqn],
            repo=repo,
            to_canonical_fn=self.to_canonical,
            user_id=user_id,
            limit_per_anchor=limit,
        )

        # Return memories for this single FQN (empty list if no matches)
        canonical_fqn = self.to_canonical(fqn)
        return result_by_fqn.get(canonical_fqn, [])

    def _index_symbol_cards(self, repo_path: Path):
        """Build symbol cards and index them for locate (C3).

        For each source-backed symbol, builds a card (name + signature + docstring
        + first N lines of source) and:
        - **Always** writes the card text (+ signature/docstring) back onto the
          :CodeSymbol node, powering the token-free BM25 lexical leg (C1). This is
          the deterministic default — no cloud, no network.
        - When ``code_embedder`` is local/cloud, ALSO embeds the cards (local
          fastembed ONNX by default; Gemini only when opted in) and upserts them
          to the code_index Qdrant collection for the dense leg (C3).
        """
        from adapters.code_graph.code_locate import (
            build_card_text,
            invalidate_bm25,
            symbol_point_id,
        )

        mode = self._code_embedder_mode()

        # Fetch all symbols with degree
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span,
               coalesce(s.degree, 0) AS degree
        """
        symbols = self._run_cypher(cypher, code_space=self.code_space)

        if not symbols:
            logger.info("No symbols to index in code_index")
            invalidate_bm25(self.code_space)
            self._bump_card_epoch()
            return None

        # Build symbol cards (shared card text for BM25 + optional dense embed)
        cards: list[str] = []
        points_data: list[dict] = []
        node_updates: list[dict] = []
        for sym in symbols:
            fqn = sym["fqn"]
            kind = sym["kind"]
            file_path = sym["file"]
            # External / inferred symbols may have no file — they can't be
            # located or turned into a source-backed card, so skip them.
            if not file_path:
                continue
            span = sym["span"] or "1:1"
            try:
                line = int(str(span).split(":")[0])
            except (ValueError, IndexError):
                line = 0
            degree = sym["degree"]

            signature, docstring, first_lines = self._extract_symbol_details(
                repo_path / file_path, fqn, line
            )
            card_text = build_card_text(fqn, kind, signature, docstring, first_lines)

            cards.append(card_text)
            points_data.append({
                "fqn": fqn,
                "kind": kind,
                "file": file_path,
                "line": line,
                "signature": signature,
                "docstring": docstring,
                "degree": degree,
                "anchor_id": None,  # E4: anchors deferred
            })
            node_updates.append({
                "fqn": fqn,
                "signature": signature or "",
                "docstring": docstring or "",
                "card": card_text,
            })

        # Always persist card text onto the graph (powers the BM25 lexical leg),
        # bump the card epoch (cross-process BM25 cache invalidation, MF-1), and
        # invalidate this process's cache immediately.
        self._write_card_fields(node_updates)
        invalidate_bm25(self.code_space)
        self._bump_card_epoch()

        if mode == "off":
            logger.info(
                "code_embedder=off — %d symbol cards indexed for BM25 (token-free "
                "deterministic default); no dense vectors", len(node_updates)
            )
            return None  # dense leg not applicable

        # Dense leg: embed cards (local fastembed by default; cloud only opted in).
        # MF-5: an offline/air-gapped deployment (the audience for a token-free
        # default) can't fetch the local ONNX model on first use — so the ENTIRE
        # dense stage is best-effort. On any failure we keep the already-written
        # BM25 cards (locate degrades to ~0.60 h@1) instead of failing the whole
        # index job, and signal the degradation up to the IndexReport.
        from memory_service import get_shared_service
        from qdrant_client.models import PointStruct

        try:
            service = get_shared_service()
            m = service._get_memory()
            vector_size = self._code_vector_size(m, mode)
            # recreate_on_mismatch handles a true dim change (e.g. a non-768 cloud
            # model); the per-point embedder tag (below) handles same-dim switches.
            self._ensure_code_index_collection(
                m, vector_size=vector_size, recreate_on_mismatch=True
            )

            if mode == "local":
                logger.info("Embedding %d symbol cards (local code embedder)", len(cards))
                embeddings = self._get_local_code_embedder().embed_documents(cards)
            else:  # cloud — Gemini batchEmbedContents caps at 100 per call
                logger.info("Embedding %d symbol cards (cloud embedder)", len(cards))
                _EMBED_CHUNK = 100
                embeddings = []
                for i in range(0, len(cards), _EMBED_CHUNK):
                    embeddings.extend(
                        m.embedding_model.embed_batch(
                            cards[i : i + _EMBED_CHUNK], memory_action="add"
                        )
                    )

            if len(embeddings) != len(cards):
                logger.warning(
                    f"Embedding mismatch: {len(embeddings)} embeddings for "
                    f"{len(cards)} cards — dense leg skipped (BM25 still active)"
                )
                return True  # degraded: card text is indexed, dense is not

            # MF-2: clear this code_space's existing cards first so a reindex can't
            # leave stale points behind — legacy uuid4 points or points from a
            # different embedder (whose vectors live in another space) — which RRF
            # would otherwise fuse as garbage. Then upsert with a stable per-symbol
            # id (dense + lexical fuse on identity; reindex is idempotent) and the
            # embedder identity stamped on each point (matched at query time).
            self._delete_code_index_cards()
            identity = self._code_embedder_identity(mode)
            points = []
            for embed, data in zip(embeddings, points_data):
                points.append(PointStruct(
                    id=symbol_point_id(self.code_space, data["fqn"]),
                    vector=embed,
                    payload={
                        "code_space": self.code_space,
                        "embedder": identity,
                        "fqn": data["fqn"],
                        "kind": data["kind"],
                        "file": data["file"],
                        "line": data["line"],
                        "signature": data["signature"],
                        "docstring": data["docstring"],
                        "degree": data["degree"],
                        "anchor_id": data["anchor_id"],
                    },
                ))

            m.vector_store.client.upsert(collection_name="code_index", points=points)
            logger.info(f"Indexed {len(points)} symbols into code_index collection")
            return False  # dense leg fully built
        except Exception:
            logger.warning(
                "Dense code-index leg failed (%s mode); card-text BM25 still "
                "active so locate degrades to ~0.60 h@1 rather than failing the "
                "index. Common cause: local embedder model unavailable offline.",
                mode, exc_info=True,
            )
            return True  # degraded

    def _write_card_fields(self, updates: list[dict]) -> None:
        """Batch-write card text (+ signature/docstring) onto :CodeSymbol nodes."""
        if not updates:
            return
        cypher = """
        UNWIND $rows AS row
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: row.fqn})
        SET s.signature = row.signature,
            s.docstring = row.docstring,
            s.card = row.card
        """
        for i in range(0, len(updates), 500):
            self._run_cypher_with_retry(
                cypher, code_space=self.code_space, rows=updates[i : i + 500]
            )

    def _extract_symbol_details(
        self, file_path: Path, fqn: str, line: int
    ) -> tuple[str, str, str]:
        """Extract signature, docstring, and first N lines for a symbol.

        Returns: (signature, docstring, first_lines)
        """
        if not file_path.exists():
            return ("", "", "")

        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return ("", "", "")

        # Simple heuristic extraction (tree-sitter query would be more accurate but E3 doesn't require LSP-grade)
        signature = ""
        docstring = ""
        first_lines = ""

        if line > 0 and line <= len(lines):
            # Signature: the definition line
            def_line = lines[line - 1].strip()
            signature = def_line

            # Docstring: look for triple-quoted string in next few lines
            doc_lines = []
            in_doc = False
            for i in range(line, min(line + 10, len(lines))):
                l = lines[i].strip()
                if '"""' in l or "'''" in l:
                    if not in_doc:
                        in_doc = True
                        doc_lines.append(l.split('"""')[-1].split("'''")[-1])
                        if l.count('"""') >= 2 or l.count("'''") >= 2:
                            break
                    else:
                        doc_lines.append(l.split('"""')[0].split("'''")[0])
                        break
                elif in_doc:
                    doc_lines.append(l)
            docstring = " ".join(doc_lines).strip()

            # First N lines of source (skip docstring)
            start = line
            end = min(line + 5, len(lines))
            first_lines = "\n".join(lines[start - 1:end])

        return (signature, docstring, first_lines)

    # ── Internal query helpers ───────────────────────────────────────

    def _search_symbols(self, keywords: list[str], limit: int = 10) -> list[dict]:
        """Search symbols by FQN substring match (scored by degree)."""
        # Fix 5: Parameterize keywords to prevent Cypher injection
        if not keywords:
            return []

        # Build WHERE clause with parameterized keywords
        where_clauses = [f"toLower(s.fqn) CONTAINS toLower($kw{i})" for i in range(len(keywords))]
        where_clause = " AND ".join(where_clauses)

        cypher = f"""
        MATCH (s:CodeSymbol {{code_space: $code_space}})
        WHERE {where_clause}
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span, s.degree AS degree
        ORDER BY coalesce(s.degree, 0) DESC
        LIMIT $limit
        """

        # Build params dict with keyword parameters
        params = {"code_space": self.code_space, "limit": limit}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = kw

        results = self._run_cypher(cypher, **params)
        return [
            {
                "fqn": r["fqn"],
                "kind": r["kind"],
                "file": r["file"],
                "line": int(r["span"].split(":")[0]) if r.get("span") else 0,
            }
            for r in results
        ]

    def _find_symbol(self, label: str) -> list[dict]:
        """Find symbols by FQN substring match."""
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        WHERE toLower(s.fqn) CONTAINS toLower($label)
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span
        LIMIT 5
        """
        results = self._run_cypher(cypher, code_space=self.code_space, label=label)
        return [
            {
                "fqn": r["fqn"],
                "kind": r["kind"],
                "file": r["file"],
                "line": int(r["span"].split(":")[0]) if r.get("span") else 0,
            }
            for r in results
        ]

    def _get_edges(self, fqn: str, rel_filter: str = "") -> list[dict]:
        """Fetch in/out edges for a symbol."""
        # Out edges
        out_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $fqn})-[r]->(t:CodeSymbol)
        WHERE type(r) IN ['CALLS', 'IMPORTS', 'DEFINES', 'INHERITS', 'REFERENCES']
        RETURN 'out' AS direction, t.fqn AS neighbor, type(r) AS relation, r.extraction AS extraction
        """
        # In edges
        in_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $fqn})<-[r]-(t:CodeSymbol)
        WHERE type(r) IN ['CALLS', 'IMPORTS', 'DEFINES', 'INHERITS', 'REFERENCES']
        RETURN 'in' AS direction, t.fqn AS neighbor, type(r) AS relation, r.extraction AS extraction
        """
        out_edges = self._run_cypher(out_cypher, code_space=self.code_space, fqn=fqn)
        in_edges = self._run_cypher(in_cypher, code_space=self.code_space, fqn=fqn)

        edges = out_edges + in_edges
        if rel_filter:
            edges = [e for e in edges if rel_filter in e["relation"].lower()]
        return edges

    def _traverse(self, seed_fqn: str, mode: str, depth: int, budget: int) -> list[dict]:
        """BFS/DFS traversal from a seed symbol (respects token budget)."""
        # Simplified: just fetch neighbors up to depth hops
        cypher = f"""
        MATCH path = (s:CodeSymbol {{code_space: $code_space, fqn: $fqn}})-[*1..{depth}]->(t:CodeSymbol)
        RETURN DISTINCT t.fqn AS fqn, t.kind AS kind, t.file AS file, t.span AS span
        LIMIT 20
        """
        results = self._run_cypher(cypher, code_space=self.code_space, fqn=seed_fqn)
        visited = [
            {
                "fqn": r["fqn"],
                "kind": r["kind"],
                "file": r["file"],
                "line": int(r["span"].split(":")[0]) if r.get("span") else 0,
                "edges": [],  # would need a second query per node for full output
            }
            for r in results
        ]
        return visited

    def _shortest_path(self, src_fqn: str, tgt_fqn: str, max_hops: int) -> list[dict]:
        """Compute shortest path between two symbols."""
        cypher = f"""
        MATCH path = shortestPath(
          (s:CodeSymbol {{code_space: $code_space, fqn: $src_fqn}})-[*1..{max_hops}]-(t:CodeSymbol {{fqn: $tgt_fqn}})
        )
        RETURN [n IN nodes(path) | {{fqn: n.fqn, kind: n.kind}}] AS nodes,
               [r IN relationships(path) | {{relation: type(r), extraction: r.extraction}}] AS edges
        LIMIT 1
        """
        results = self._run_cypher(
            cypher,
            code_space=self.code_space,
            src_fqn=src_fqn,
            tgt_fqn=tgt_fqn,
        )
        if not results:
            return []

        # Reconstruct path with edges interleaved
        nodes = results[0]["nodes"]
        edges = results[0]["edges"]
        path = []
        for i, node in enumerate(nodes):
            path.append({
                "fqn": node["fqn"],
                "kind": node["kind"],
                "edge": edges[i] if i < len(edges) else None,
            })
        return path

    # ── E5 detect_changes helpers ───────────────────────────────────

    def _compute_symbol_body_hash(self, rel_path: str, sym: _Symbol) -> str:
        """Compute SHA256 hash of a symbol's body content (for change detection).

        Reads the source file and hashes the lines covering the symbol's span.
        Falls back to empty hash if file doesn't exist or read fails.
        """
        try:
            file_path = self.repo_path / rel_path
            if not file_path.exists():
                return ""
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            # Extract lines from sym.line to sym.end_line (1-indexed)
            start = max(0, sym.line - 1)
            end = min(len(lines), sym.end_line)
            body_lines = lines[start:end]
            body_text = "\n".join(body_lines)
            return hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        except Exception:
            logger.debug(f"Failed to compute body_hash for {sym.fqn}", exc_info=True)
            return ""

    def _fetch_persisted_symbols(self) -> list[dict]:
        """Fetch all persisted symbols from Neo4j with their body_hash."""
        cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space})
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span,
               s.body_hash AS body_hash
        """
        results = self._run_cypher(cypher, code_space=self.code_space)
        return [
            {
                "fqn": r["fqn"],
                "kind": r.get("kind", ""),
                "file": r.get("file", ""),
                "span": r.get("span", ""),
                "body_hash": r.get("body_hash", ""),
            }
            for r in results
        ]

    def _parse_fresh_symbols(self) -> list[dict]:
        """Parse the working tree and return symbol descriptors with body_hash.

        Re-parses all source files in the repo (same as index() but without writing).
        """
        file_patterns = {
            "*.py": "python",
            "*.ts": "typescript",
            "*.tsx": "typescript",
            "*.js": "javascript",
            "*.jsx": "javascript",
            "*.go": "go",
            "*.rs": "rust",
            "*.java": "java",
        }
        source_files = []
        for pattern, lang in file_patterns.items():
            for f in self.repo_path.rglob(pattern):
                source_files.append((f, lang))

        fresh_symbols = []
        for source_file, lang in source_files:
            rel_path = str(source_file.relative_to(self.repo_path))
            symbols, _ = self._parse_file(source_file, self.repo_path, lang)
            for sym in symbols:
                body_hash = self._compute_symbol_body_hash(rel_path, sym)
                fresh_symbols.append({
                    "fqn": sym.fqn,
                    "kind": sym.kind,
                    "file": sym.file,
                    "span": f"{sym.line}:{sym.end_line}",
                    "body_hash": body_hash,
                })
        return fresh_symbols

    def _parse_snapshot_symbols(self, snapshot_data: bytes) -> list[dict]:
        """Parse symbols from a snapshot artifact (E6).

        Extracts CodeSymbol nodes from the snapshot and returns them in the same
        format as _fetch_persisted_symbols() for change comparison.

        Args:
            snapshot_data: Compressed snapshot bytes.

        Returns:
            List of symbol dicts with fqn, kind, file, span, body_hash.
        """
        import gzip
        import json

        # Decompress and parse
        decompressed = gzip.decompress(snapshot_data)
        envelope = json.loads(decompressed.decode("utf-8"))
        snapshot = envelope["snapshot"]

        # Extract CodeSymbol nodes
        symbols = []
        for node in snapshot["nodes"]:
            if "CodeSymbol" in node["labels"]:
                props = node["properties"]
                symbols.append({
                    "fqn": props.get("fqn", ""),
                    "kind": props.get("kind", ""),
                    "file": props.get("file", ""),
                    "span": props.get("span", ""),
                    "body_hash": props.get("body_hash", ""),
                })
        return symbols

    def _blast_radius_bfs(self, roots: list[str], max_depth: int = 3) -> set[str]:
        """BFS over CALLS/IMPORTS edges from root symbols to find affected symbols.

        Args:
            roots: FQNs of deleted/modified symbols (blast epicenters).
            max_depth: Maximum BFS depth (default 3).

        Returns:
            Set of affected FQNs (includes roots).
        """
        if not roots:
            return set()

        # Fix 4: Single-Cypher blast radius using variable-length path query
        # Both incoming and outgoing edges (a deleted function affects callers AND callees)
        cypher = f"""
        UNWIND $roots AS root_fqn
        MATCH (root:CodeSymbol {{code_space: $code_space, fqn: root_fqn}})
        OPTIONAL MATCH (root)<-[r:CALLS|IMPORTS*1..{max_depth}]-(caller:CodeSymbol)
        WITH root, collect(DISTINCT caller.fqn) AS callers
        OPTIONAL MATCH (root)-[r:CALLS|IMPORTS*1..{max_depth}]->(callee:CodeSymbol)
        WITH root, callers, collect(DISTINCT callee.fqn) AS callees
        RETURN root.fqn AS root_fqn, callers, callees
        """
        results = self._run_cypher(cypher, code_space=self.code_space, roots=roots)

        # Collect all affected FQNs
        affected = set(roots)
        for row in results:
            affected.update(row.get("callers", []) or [])
            affected.update(row.get("callees", []) or [])

        return affected

    def _get_blast_neighbors(self, fqn: str) -> list[str]:
        """Get neighbors of a symbol via CALLS/IMPORTS edges (both directions).

        Returns list of neighbor FQNs.
        """
        # Out-edges: symbols this one calls/imports
        out_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $fqn})-[r]->(t:CodeSymbol)
        WHERE type(r) IN ['CALLS', 'IMPORTS']
        RETURN DISTINCT t.fqn AS neighbor
        """
        # In-edges: symbols that call/import this one
        in_cypher = """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $fqn})<-[r]-(t:CodeSymbol)
        WHERE type(r) IN ['CALLS', 'IMPORTS']
        RETURN DISTINCT t.fqn AS neighbor
        """
        out_results = self._run_cypher(out_cypher, code_space=self.code_space, fqn=fqn)
        in_results = self._run_cypher(in_cypher, code_space=self.code_space, fqn=fqn)

        neighbors = [r["neighbor"] for r in out_results + in_results]
        return list(set(neighbors))  # dedup

    def _collect_affected_anchors(self, affected_fqns: set[str]) -> list[str]:
        """Collect anchor keys for all affected symbols.

        Returns list of anchor keys in the format "<repo>::<fqn>".
        """
        if not affected_fqns:
            return []

        # Extract repo from code_space
        parts = self.code_space.split("--")
        repo = parts[-1] if len(parts) >= 3 else "unknown"

        # Build anchor keys
        anchor_keys = [f"{repo}::{fqn}" for fqn in affected_fqns]
        return sorted(anchor_keys)

    # ── Neo4j bridge helpers (follow graph_patcher.py pattern) ──────

    def _run_cypher(self, cypher: str, **params) -> list[dict]:
        """Run a Cypher query via the Graphiti bridge (synchronous wrapper)."""
        import asyncio
        import concurrent.futures

        if not self.bridge:
            raise RuntimeError("Neo4j bridge not initialized")

        loop = getattr(self.bridge, "_loop", None)
        if not isinstance(loop, asyncio.AbstractEventLoop):
            raise RuntimeError("Bridge loop is not an asyncio event loop")

        async def _inner():
            # The real mem0 _AsyncBridge has no .driver; use the injected
            # Graphiti driver, falling back to bridge.driver for mock bridges.
            driver = self.driver if self.driver is not None else getattr(self.bridge, "driver", None)
            if driver is None:
                raise RuntimeError("No Neo4j driver available (bridge has no .driver and none injected)")
            async with driver.session() as session:
                result = await session.run(cypher, **params)
                records = await result.data()
                return records

        future = asyncio.run_coroutine_threadsafe(_inner(), loop)
        try:
            return future.result(timeout=30.0)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("Cypher query timed out after 30s") from None

    def _run_cypher_with_retry(self, cypher: str, **params):
        """Run Cypher with deadlock retry (mirrors graph_patcher.py)."""
        from neo4j.exceptions import TransientError

        for attempt in range(4):  # 3 retries + 1 initial
            try:
                return self._run_cypher(cypher, **params)
            except TransientError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (2 ** attempt) + random.uniform(0, 0.2))
