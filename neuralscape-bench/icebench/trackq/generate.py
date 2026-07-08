"""
Query generation for Track-Q.

Generates both:
1. Structural QA queries (symbol_lookup, neighbors_1hop, path_le4)
2. NL locate queries (docstring -> function)
"""

import logging
import random
import re
from pathlib import Path
from dataclasses import dataclass

from icebench.adapters.base import Corpus
from icebench.trackq.oracle import TreeSitterOracle, TREE_SITTER_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class QuerySpec:
    """A query with ground-truth answer."""
    op: str
    payload: dict  # Input to adapter.query()
    gold: dict  # Ground-truth answer for scoring


def generate_queries(op: str, corpus: Corpus, n: int = 200, seed: int = 42) -> list[QuerySpec]:
    """
    Generate n queries for the given operation and corpus.

    Args:
        op: Operation name ("symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate").
        corpus: Corpus to query.
        n: Number of queries to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of QuerySpec objects.
    """
    rng = random.Random(seed)

    if op == "symbol_lookup":
        return _generate_symbol_lookup(corpus, n, rng)
    elif op == "neighbors_1hop":
        return _generate_neighbors_1hop(corpus, n, rng)
    elif op == "path_le4":
        return _generate_path_le4(corpus, n, rng)
    elif op == "nl_locate":
        return _generate_nl_locate(corpus, n, rng)
    else:
        logger.warning(f"Unknown operation: {op}")
        return []


def _generate_symbol_lookup(corpus: Corpus, n: int, rng: random.Random) -> list[QuerySpec]:
    """Generate symbol_lookup queries: where is symbol X defined?"""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter not available; returning empty symbol_lookup queries")
        return []

    try:
        oracle = TreeSitterOracle(corpus.path, corpus.language)
        oracle.index()
    except Exception as e:
        logger.error(f"Oracle indexing failed: {e}")
        return []

    # Sample symbols
    symbols = list(oracle.symbols.values())
    if not symbols:
        logger.warning("No symbols found by oracle")
        return []

    sampled = rng.sample(symbols, min(n, len(symbols)))

    queries = []
    for sym in sampled:
        queries.append(QuerySpec(
            op="symbol_lookup",
            payload={"symbol": sym.name, "corpus": corpus},
            gold={"file": sym.file, "symbol": sym.name, "line": sym.line},
        ))

    return queries


def _generate_neighbors_1hop(corpus: Corpus, n: int, rng: random.Random) -> list[QuerySpec]:
    """Generate neighbors_1hop queries: who calls symbol X?"""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter not available; returning empty neighbors_1hop queries")
        return []

    try:
        oracle = TreeSitterOracle(corpus.path, corpus.language)
        oracle.index()
    except Exception as e:
        logger.error(f"Oracle indexing failed: {e}")
        return []

    # Find symbols that have callers
    symbols_with_callers = [
        (sym_name, oracle.get_callers(sym_name))
        for sym_name in oracle.symbols.keys()
    ]
    symbols_with_callers = [(s, c) for s, c in symbols_with_callers if c]

    if not symbols_with_callers:
        logger.warning("No symbols with callers found")
        return []

    sampled = rng.sample(symbols_with_callers, min(n, len(symbols_with_callers)))

    queries = []
    for sym_name, callers in sampled:
        queries.append(QuerySpec(
            op="neighbors_1hop",
            payload={"symbol": sym_name, "corpus": corpus},
            gold={"symbol": sym_name, "callers": callers},
        ))

    return queries


def _generate_path_le4(corpus: Corpus, n: int, rng: random.Random) -> list[QuerySpec]:
    """Generate path_le4 queries: find path from A to B (≤4 hops)."""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("tree-sitter not available; returning empty path_le4 queries")
        return []

    try:
        oracle = TreeSitterOracle(corpus.path, corpus.language)
        oracle.index()
    except Exception as e:
        logger.error(f"Oracle indexing failed: {e}")
        return []

    # Sample pairs of symbols and find paths
    symbol_names = list(oracle.symbols.keys())
    if len(symbol_names) < 2:
        logger.warning("Not enough symbols for path queries")
        return []

    queries = []
    attempts = 0
    max_attempts = n * 10  # Try harder to find valid paths

    while len(queries) < n and attempts < max_attempts:
        attempts += 1
        from_sym = rng.choice(symbol_names)
        to_sym = rng.choice(symbol_names)

        if from_sym == to_sym:
            continue

        paths = oracle.find_paths(from_sym, to_sym, max_depth=4)

        if paths:
            queries.append(QuerySpec(
                op="path_le4",
                payload={"from": from_sym, "to": to_sym, "corpus": corpus},
                gold={"from": from_sym, "to": to_sym, "paths": paths},
            ))

    if len(queries) < n:
        logger.warning(f"Only found {len(queries)}/{n} valid paths")

    return queries


def _generate_nl_locate(corpus: Corpus, n: int, rng: random.Random) -> list[QuerySpec]:
    """
    Generate nl_locate queries: use docstring as query to find function.

    CodeSearchNet-style: sample functions with docstrings, strip the docstring,
    use it as the query, and the target is (file, symbol).
    """
    corpus_path = Path(corpus.path)

    # Find source files
    exts = _extensions_for_language(corpus.language)
    files = sorted(
        p for p in corpus_path.rglob("*")
        if p.is_file() and p.suffix in exts and ".git" not in p.parts
    )

    if not files:
        logger.warning("No source files found")
        return []

    # Extract functions with docstrings
    candidates = []
    for file_path in files:
        try:
            funcs = _extract_functions_with_docstrings(file_path, corpus.language)
            for func_name, docstring in funcs:
                rel_path = str(file_path.relative_to(corpus_path))
                candidates.append((rel_path, func_name, docstring))
        except Exception as e:
            logger.debug(f"Failed to extract from {file_path}: {e}")

    if not candidates:
        logger.warning("No functions with docstrings found")
        return []

    # Ensure we have at least 150 samples (or as many as possible)
    target = max(n, 150)
    sampled = rng.sample(candidates, min(target, len(candidates)))

    queries = []
    for file, symbol, docstring in sampled:
        queries.append(QuerySpec(
            op="nl_locate",
            payload={"query": docstring, "corpus": corpus},
            gold={"file": file, "symbol": symbol},
        ))

    logger.info(f"Generated {len(queries)} nl_locate queries")
    return queries


def _extensions_for_language(language: str) -> set[str]:
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


def _extract_functions_with_docstrings(file_path: Path, language: str) -> list[tuple[str, str]]:
    """
    Extract (function_name, docstring) pairs from a file.

    Args:
        file_path: Path to source file.
        language: Language name.

    Returns:
        List of (function_name, docstring) tuples.
    """
    if language == "python":
        return _extract_python_docstrings(file_path)
    elif language == "go":
        return _extract_go_docstrings(file_path)
    # Add more languages as needed
    else:
        return []


def _extract_python_docstrings(file_path: Path) -> list[tuple[str, str]]:
    """Extract Python function docstrings."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    results = []

    # Regex to find function definitions followed by docstrings
    # This is a simplified approach; tree-sitter would be more robust but adds complexity
    pattern = re.compile(
        r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*'
        r'(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')',
        re.DOTALL | re.MULTILINE
    )

    for match in pattern.finditer(source):
        func_name = match.group(1)
        docstring = match.group(2) or match.group(3)

        if docstring and len(docstring.strip()) > 10:  # Must have meaningful content
            # Clean the docstring: first line or first sentence
            cleaned = _clean_docstring(docstring)
            if cleaned:
                results.append((func_name, cleaned))

    return results


def _extract_go_docstrings(file_path: Path) -> list[tuple[str, str]]:
    """Extract Go function doc comments."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    results = []

    # Go doc comments are // lines immediately before function
    # Pattern: // comment lines followed by func Name
    lines = source.split("\n")

    i = 0
    while i < len(lines):
        # Collect consecutive // comment lines
        comments = []
        while i < len(lines) and lines[i].strip().startswith("//"):
            comment = lines[i].strip()[2:].strip()
            if comment:
                comments.append(comment)
            i += 1

        # Check if next non-empty line is a function
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i < len(lines) and lines[i].strip().startswith("func "):
            # Extract function name
            func_line = lines[i].strip()
            match = re.match(r'func\s+(?:\([^)]+\)\s*)?([a-zA-Z_][a-zA-Z0-9_]*)', func_line)
            if match and comments:
                func_name = match.group(1)
                docstring = " ".join(comments)

                if len(docstring) > 10:
                    results.append((func_name, docstring))

        i += 1

    return results


def _clean_docstring(docstring: str) -> str:
    """
    Clean a docstring: take first line or first sentence.

    Args:
        docstring: Raw docstring.

    Returns:
        Cleaned query string.
    """
    # Strip leading/trailing whitespace
    doc = docstring.strip()

    # Take first line
    first_line = doc.split("\n")[0].strip()

    # Or take first sentence (ends with .)
    sentences = re.split(r'[.!?]', first_line)
    if sentences:
        first_sentence = sentences[0].strip()
        if first_sentence:
            return first_sentence

    # Fallback to first line
    if first_line:
        return first_line

    # Last resort: whole docstring up to 200 chars
    return doc[:200].strip()
