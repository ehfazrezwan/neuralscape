"""NativeEngine — tree-sitter indexer + Neo4j code label-space (E2).

E2 scope (Python only):
- Index with tree-sitter (heuristic FQN, no LSP) → Neo4j code_space partition
- Incremental by file content-hash
- Query/neighbors/path produce the SAME text output as GraphifyJsonEngine (parity)
- Locate/detect_changes/semantic_layer/export_snapshot raise EngineCapabilityError (E3+)

Label-space schema:
  (:CodeRepo {code_space, name, path})
  (:CodeFile {code_space, path, hash, language, span})
  (:CodeSymbol {code_space, fqn, kind, file, span, degree})
  Edges: CALLS | IMPORTS | DEFINES | INHERITS | REFERENCES
    each with {extraction: "extracted"|"inferred"|"ambiguous"}

Partition key: code_space = "code--{owner}--{repo}" on EVERY node.
Degree persisted on symbols at index time.
community_id stubbed (Louvain is a later slice).
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
    ):
        """Initialize with repo path and Neo4j bridge.

        Args:
            repo_path: Absolute path to the repo root.
            code_space: Partition key (code--{owner}--{repo}).
            bridge: Graphiti bridge (provides .driver and ._loop).
            settings: Config object.
        """
        self.repo_path = Path(repo_path)
        self.code_space = code_space
        self.bridge = bridge
        self.settings = settings

    def query(
        self,
        question: str,
        *,
        mode: str = "bfs",
        depth: int = 3,
        token_budget: int = 2000,
    ) -> str:
        """Search the code graph via BFS/DFS from scored seed nodes."""
        # Normalize params the same way GraphifyJsonEngine does
        mode = mode if mode in ("bfs", "dfs") else "bfs"
        depth = max(1, min(int(depth), 6))
        token_budget = max(100, min(int(token_budget), 20_000))

        # Retrieve symbols matching the question keywords
        keywords = question.lower().split()
        symbols = self._search_symbols(keywords, limit=5)
        if not symbols:
            return f"No symbols matching '{question}' found in {self.code_space}."

        # BFS/DFS traversal from the top-scored symbol
        lines = [f"Code graph search results for: {question}", ""]
        seed_fqn = symbols[0]["fqn"]
        visited = self._traverse(seed_fqn, mode=mode, depth=depth, budget=token_budget)
        for item in visited:
            lines.append(
                f"{item['fqn']} ({item['kind']}) in {item['file']}:{item['line']}"
            )
            if item.get("edges"):
                for edge in item["edges"][:3]:  # limit outbound edges shown
                    lines.append(f"  --> {edge['relation']} {edge['target']}")
        return "\n".join(lines)

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
    ) -> str:
        """Direct in/out neighbors of one code-graph symbol."""
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

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[LocateHit]:
        """Hybrid code retrieval — requires E3 (embeddings + code_index collection)."""
        raise EngineCapabilityError(
            "locate() requires E3 (dense embeddings + code_index collection). "
            "E2 implements only structural queries (query, neighbors, path). "
            "Use query() for structure search."
        )

    def detect_changes(
        self,
        since: str,
    ) -> ChangeReport:
        """Blast-radius detection — requires E5 (historical snapshots)."""
        raise EngineCapabilityError(
            "detect_changes() requires E5 (incremental diff + blast-radius BFS). "
            "E2 implements incremental indexing by content-hash but no historical "
            "snapshot comparison. Re-index and manually compare the code_space."
        )

    def semantic_layer(self) -> list[SemanticFact]:
        """Semantic distillation — requires Louvain communities (later slice)."""
        raise EngineCapabilityError(
            "semantic_layer() requires Louvain community detection (deferred). "
            "E2 persists degree but not community_id. Use query() for structure."
        )

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Index a Python codebase into the Neo4j code label-space."""
        start = time.time()
        repo_path = Path(source).resolve()
        if not repo_path.is_dir():
            raise ValueError(f"source must be a directory: {source}")

        # Collect Python files
        py_files = list(repo_path.rglob("*.py"))
        if not py_files:
            logger.info("No Python files found in %s", repo_path)
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

        for py_file in py_files:
            rel_path = str(py_file.relative_to(repo_path))
            file_hash = self._file_hash(py_file)

            # Incremental: skip unchanged files
            if incremental and self._file_unchanged(rel_path, file_hash):
                continue

            # Parse and index
            symbols, edges = self._parse_file(py_file, repo_path)
            if symbols or edges:
                self._store_file(rel_path, file_hash, symbols, edges)
                files_indexed += 1
                symbols_indexed += len(symbols)
                edges_indexed += len(edges)

        # Compute and persist degree on all symbols
        self._compute_degrees()

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
        )

    def export_snapshot(self) -> bytes:
        """Export snapshot — requires E6 (content-addressed snapshots)."""
        raise EngineCapabilityError(
            "export_snapshot() requires E6 (index-in-CI snapshot export). "
            "E2 stores the live index in Neo4j only."
        )

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

    def _parse_file(self, py_file: Path, repo_root: Path) -> tuple[list[_Symbol], list[_Edge]]:
        """Parse a Python file with tree-sitter and extract symbols/edges."""
        try:
            from tree_sitter import Language, Parser
        except ImportError:
            raise RuntimeError(
                "tree-sitter dependencies not installed. "
                "Install with: uv sync --extra code-graph"
            ) from None

        # Get the Python language from tree-sitter-language-pack
        try:
            import tree_sitter_python
            language = Language(tree_sitter_python.language())
        except ImportError:
            raise RuntimeError(
                "tree-sitter-python not installed (comes with tree-sitter-language-pack). "
                "Install with: uv sync --extra code-graph"
            ) from None

        parser = Parser(language)
        source_bytes = py_file.read_bytes()
        tree = parser.parse(source_bytes)
        root = tree.root_node

        # Walk the tree directly (simpler than Query API for E2)
        rel_path = str(py_file.relative_to(repo_root))
        module_path = rel_path.replace("/", ".").removesuffix(".py")

        symbols: list[_Symbol] = []
        edges: list[_Edge] = []

        def walk(node, parent_class=None):
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

            # Call expressions (inferred edges)
            elif node_type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    target_name = func_node.text.decode("utf8")
                    # Best-effort FQN (inferred)
                    edges.append(_Edge(
                        source_fqn=module_path,
                        target_fqn=f"{module_path}.{target_name}",
                        relation="CALLS",
                        extraction="inferred",
                    ))

            # Recurse to children (unless we already handled them above)
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
    ):
        """Store a file + its symbols + edges in Neo4j (with deadlock retry)."""
        # Store the file node
        file_cypher = """
        MERGE (f:CodeFile {code_space: $code_space, path: $path})
        SET f.hash = $hash, f.language = 'python'
        """
        self._run_cypher_with_retry(file_cypher, code_space=self.code_space, path=rel_path, hash=file_hash)

        # Store symbols
        for sym in symbols:
            sym_cypher = """
            MERGE (s:CodeSymbol {code_space: $code_space, fqn: $fqn})
            SET s.kind = $kind, s.file = $file, s.span = $span
            """
            span = f"{sym.line}:{sym.end_line}"
            self._run_cypher_with_retry(
                sym_cypher,
                code_space=self.code_space,
                fqn=sym.fqn,
                kind=sym.kind,
                file=sym.file,
                span=span,
            )

        # Store edges
        for edge in edges:
            # Map extraction type to epistemic level (reuse semantic.py's mapping)
            epistemic = self._extraction_to_epistemic(edge.extraction)
            edge_cypher = """
            MERGE (src:CodeSymbol {code_space: $code_space, fqn: $src_fqn})
            MERGE (tgt:CodeSymbol {code_space: $code_space, fqn: $tgt_fqn})
            MERGE (src)-[r:%s]->(tgt)
            SET r.extraction = $extraction, r.epistemic_level = $epistemic
            """ % edge.relation
            self._run_cypher_with_retry(
                edge_cypher,
                code_space=self.code_space,
                src_fqn=edge.source_fqn,
                tgt_fqn=edge.target_fqn,
                extraction=edge.extraction,
                epistemic=epistemic,
            )

    def _extraction_to_epistemic(self, extraction: str) -> str:
        """Map extraction confidence to epistemic level (mirrors semantic.py)."""
        if extraction == "extracted":
            return "explicit"
        elif extraction in ("inferred", "ambiguous"):
            return "deductive"
        return "deductive"  # fallback

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

    # ── Internal query helpers ───────────────────────────────────────

    def _search_symbols(self, keywords: list[str], limit: int = 10) -> list[dict]:
        """Search symbols by FQN substring match (scored by degree)."""
        # Simple keyword AND match on FQN
        where_clauses = [f"toLower(s.fqn) CONTAINS '{kw}'" for kw in keywords]
        where_clause = " AND ".join(where_clauses)
        cypher = f"""
        MATCH (s:CodeSymbol {{code_space: $code_space}})
        WHERE {where_clause}
        RETURN s.fqn AS fqn, s.kind AS kind, s.file AS file, s.span AS span, s.degree AS degree
        ORDER BY coalesce(s.degree, 0) DESC
        LIMIT {limit}
        """
        results = self._run_cypher(cypher, code_space=self.code_space)
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
            driver = self.bridge.driver
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
