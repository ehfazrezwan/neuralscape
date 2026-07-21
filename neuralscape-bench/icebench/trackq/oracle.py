"""
Tree-sitter oracle for structural QA ground truth.

This is an INDEPENDENT oracle separate from NS's indexer. Both use tree-sitter,
which introduces a shared-oracle bias that is explicitly documented in scoring
metadata.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not available; oracle will be limited")


@dataclass
class Symbol:
    """A symbol definition."""
    name: str
    file: str  # Relative path from corpus root
    line: int
    kind: str  # "function", "class", "method", "variable", etc.


@dataclass
class Edge:
    """A structural edge between symbols."""
    from_symbol: str
    to_symbol: str
    kind: str  # "calls", "imports", "references"
    from_file: str
    to_file: str


class TreeSitterOracle:
    """
    Extract ground-truth structural edges from a corpus using tree-sitter.

    This oracle is INDEPENDENT of NS's indexer, but shares the tree-sitter
    parsing layer (shared-oracle bias).
    """

    def __init__(self, corpus_path: str, language: str):
        """
        Initialize oracle for a corpus.

        Args:
            corpus_path: Absolute path to corpus root.
            language: Language name ("python", "go", "typescript", etc.).
        """
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError("tree-sitter not available")

        self.corpus_path = Path(corpus_path)
        self.language = language

        # Set up parser based on language
        if language == "python":
            PY_LANGUAGE = Language(tree_sitter_python.language())
            self.parser = Parser(PY_LANGUAGE)
        else:
            raise ValueError(f"Language {language} not supported by oracle (only Python available)")

        # Ground truth: symbol definitions and edges
        self.symbols: dict[str, Symbol] = {}  # symbol_name -> Symbol
        self.edges: list[Edge] = []

        # Index state
        self._indexed = False

    def index(self) -> None:
        """Parse the corpus and extract ground-truth symbols and edges.

        TWO-PASS extraction so forward references resolve correctly:
          Pass 1 - parse every file and collect ALL symbol definitions (and
                   imports). This makes the full symbol table available before
                   any call edge is resolved.
          Pass 2 - walk the same (cached) parse trees and resolve call edges,
                   so a call to a function defined LATER in the file (or in a
                   file parsed later) is still recorded.
        """
        if self._indexed:
            return

        logger.info(f"Indexing corpus at {self.corpus_path} (language={self.language})")

        # Find source files
        exts = self._extensions_for_language(self.language)
        files = sorted(
            p for p in self.corpus_path.rglob("*")
            if p.is_file() and p.suffix in exts and ".git" not in p.parts
        )

        logger.info(f"Found {len(files)} source files")

        # Parse each file once; keep the tree + source alive for both passes.
        parsed: list[tuple[str, object, bytes]] = []
        for file_path in files:
            try:
                rel_path = str(file_path.relative_to(self.corpus_path))
                source = file_path.read_bytes()
                tree = self.parser.parse(source)
                parsed.append((rel_path, tree, source))
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")

        # Pass 1: symbol definitions + imports (populate the full symbol table).
        for rel_path, tree, source in parsed:
            if self.language == "python":
                self._extract_python_defs(tree.root_node, rel_path, source)
                self._extract_python_imports(tree.root_node, rel_path, source)

        # Pass 2: resolve call edges now that ALL symbols are known.
        for rel_path, tree, source in parsed:
            if self.language == "python":
                self._extract_python_call_edges(tree.root_node, rel_path, source)

        self._indexed = True
        logger.info(f"Indexed {len(self.symbols)} symbols, {len(self.edges)} edges")

    def _extensions_for_language(self, language: str) -> set[str]:
        """Map language to file extensions."""
        mapping = {
            "python": {".py"},
            "go": {".go"},
            "typescript": {".ts", ".tsx"},
            "javascript": {".js", ".jsx"},
            "rust": {".rs"},
            "java": {".java"},
        }
        return mapping.get(language, set())

    def _extract_python_defs(self, node, file: str, source: bytes) -> None:
        """PASS 1: extract Python symbol definitions (functions/classes/methods).

        No call edges are resolved here — that happens in pass 2 once the full
        symbol table exists, so forward references are not missed.
        """
        def walk(n, class_context: str | None = None):
            if n.type == "function_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    full_name = f"{class_context}.{name}" if class_context else name

                    self.symbols[full_name] = Symbol(
                        name=full_name,
                        file=file,
                        line=n.start_point[0] + 1,  # 1-indexed
                        kind="method" if class_context else "function",
                    )

            elif n.type == "class_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    full_name = f"{class_context}.{name}" if class_context else name

                    self.symbols[full_name] = Symbol(
                        name=full_name,
                        file=file,
                        line=n.start_point[0] + 1,
                        kind="class",
                    )

                    # Recurse into class body with class context
                    for child in n.children:
                        walk(child, full_name)
                    return  # Don't recurse again below

            # Recurse
            for child in n.children:
                walk(child, class_context)

        walk(node)

    def _extract_python_call_edges(self, node, file: str, source: bytes) -> None:
        """PASS 2: resolve call edges, tracking the enclosing symbol.

        Walks the tree carrying the current class + function context so each
        ``call`` node is attributed to the symbol it appears inside. Because the
        full symbol table is already populated, calls to functions defined later
        (forward references) resolve correctly.
        """
        def walk(n, class_context: str | None, enclosing: str | None):
            new_enclosing = enclosing

            if n.type == "class_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    class_context = f"{class_context}.{name}" if class_context else name

            elif n.type == "function_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    new_enclosing = f"{class_context}.{name}" if class_context else name

            elif n.type == "call" and enclosing is not None:
                func = n.child_by_field_name("function")
                if func:
                    called = source[func.start_byte:func.end_byte].decode("utf-8")
                    # Record when the callee is a known symbol (full table now)
                    # or an attribute-style reference (contains a dot).
                    if called in self.symbols or "." in called:
                        self.edges.append(Edge(
                            from_symbol=enclosing,
                            to_symbol=called,
                            kind="calls",
                            from_file=file,
                            to_file=self.symbols.get(called, Symbol("", "", 0, "")).file,
                        ))

            for child in n.children:
                walk(child, class_context, new_enclosing)

        walk(node, None, None)

    def _extract_python_imports(self, node, file: str, source: bytes) -> None:
        """Extract import statements."""
        def walk(n):
            if n.type in ("import_statement", "import_from_statement"):
                # Extract module name (simplified)
                for child in n.children:
                    if child.type == "dotted_name":
                        module = source[child.start_byte:child.end_byte].decode("utf-8")

                        # Record as an import edge (from file to module)
                        self.edges.append(Edge(
                            from_symbol=f"<module:{file}>",
                            to_symbol=module,
                            kind="imports",
                            from_file=file,
                            to_file="",  # External/unknown
                        ))

            for child in n.children:
                walk(child)

        walk(node)

    # Go extraction methods removed - only Python supported currently
    # To add Go support, install tree-sitter-go and implement _extract_go()

    def get_symbol_location(self, symbol: str) -> Symbol | None:
        """
        Look up a symbol's definition location.

        Args:
            symbol: Symbol name.

        Returns:
            Symbol object or None.
        """
        if not self._indexed:
            self.index()

        return self.symbols.get(symbol)

    def get_callers(self, symbol: str) -> list[str]:
        """
        Get all symbols that call the given symbol.

        Args:
            symbol: Target symbol name.

        Returns:
            List of caller symbol names.
        """
        if not self._indexed:
            self.index()

        return [
            edge.from_symbol
            for edge in self.edges
            if edge.to_symbol == symbol and edge.kind == "calls"
        ]

    def get_importers(self, module: str) -> list[str]:
        """
        Get all files that import the given module.

        Args:
            module: Module name.

        Returns:
            List of file paths.
        """
        if not self._indexed:
            self.index()

        return [
            edge.from_file
            for edge in self.edges
            if edge.to_symbol == module and edge.kind == "imports"
        ]

    def find_paths(self, from_symbol: str, to_symbol: str, max_depth: int = 4) -> list[list[str]]:
        """
        Find paths between two symbols (BFS up to max_depth).

        Args:
            from_symbol: Start symbol.
            to_symbol: End symbol.
            max_depth: Maximum path length.

        Returns:
            List of paths (each path is a list of symbol names).
        """
        if not self._indexed:
            self.index()

        # Build adjacency list
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            if edge.kind == "calls":
                adj[edge.from_symbol].append(edge.to_symbol)

        # BFS
        from collections import deque

        queue = deque([(from_symbol, [from_symbol])])
        paths = []
        visited = {from_symbol}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == to_symbol:
                paths.append(path)
                continue

            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return paths
