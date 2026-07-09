"""
Corpus filtering and preprocessing utilities for ICEBench.

Provides functions to filter/preprocess source files before indexing to ensure
fair benchmarking (e.g., stripping docstrings for nl_locate to avoid contamination).
"""

import re
from pathlib import Path
from typing import Iterator


# Tool output directories that should NEVER be indexed as source code.
# These are written BY the tools being benchmarked and would contaminate results.
TOOL_OUTPUT_DIRS = {
    "graphify-out",  # graphify CLI writes graph.json + reports here
    "stat-index.json",  # graphify cache file
    ".cbm",  # CBM cache (if it ever appears in corpus)
    ".codebase-memory",  # CBM portable artifact dir
    "__pycache__",  # Python bytecode
    ".pytest_cache",  # pytest cache
    "node_modules",  # npm packages
}


def is_tool_output(file_path: Path, corpus_root: Path) -> bool:
    """
    Check if a file is tool-generated output that should be excluded from indexing.

    Args:
        file_path: The file to check.
        corpus_root: The corpus root directory.

    Returns:
        True if the file is inside a tool output directory.
    """
    try:
        rel_path = file_path.relative_to(corpus_root)
    except ValueError:
        # Not under corpus_root
        return False

    # Check if any part of the path matches a tool output dir
    for part in rel_path.parts:
        if part in TOOL_OUTPUT_DIRS:
            return True

    return False


def strip_python_docstrings(source: str) -> str:
    """
    Strip docstrings from Python source code.

    Removes triple-quoted strings that appear immediately after:
    - module-level (start of file)
    - class definitions
    - function/method definitions

    This is used for nl_locate corpus preparation to ensure queries don't match
    verbatim against indexed docstring content (contamination).

    Args:
        source: Python source code.

    Returns:
        Source with docstrings replaced by empty strings or removed.
    """
    # This is a simplified regex-based approach; a full AST-based stripper would
    # be more robust but adds complexity. The regex handles the common cases.

    # Pattern to match docstrings that appear immediately after:
    # 1. Start of file (module docstring)
    # 2. def/class lines (function/method/class docstrings)

    # First pass: module-level docstring at the very beginning
    # Match optional shebang/encoding, then optional whitespace/comments, then the docstring
    module_pattern = re.compile(
        r'^((?:#![^\n]*\n)?(?:\s*#[^\n]*\n|\s*\n)*)'  # Optional shebang + comments/blank lines
        r'(\s*)("""(?:[^"]|"(?!""))*"""|\'\'\'(?:[^\']|\'(?!\'\'))*\'\'\')',  # The docstring
        re.MULTILINE | re.DOTALL
    )

    result = module_pattern.sub(r'\1', source, count=1)  # Remove first docstring, keep prefix

    # Second pass: function/class docstrings
    # Match def/class line, then whitespace, then docstring
    func_class_pattern = re.compile(
        r'((?:^|\n)([ \t]*(?:def|class)\s+[^\n]+:)[ \t]*\n)'  # def/class line with trailing newline
        r'([ \t]*)("""(?:[^"]|"(?!""))*"""|\'\'\'(?:[^\']|\'(?!\'\'))*\'\'\')',  # Indented docstring
        re.MULTILINE | re.DOTALL
    )

    def replace_func_docstring(match):
        # Keep the def/class line, remove the docstring
        return match.group(1)

    result = func_class_pattern.sub(replace_func_docstring, result)

    return result


def filter_corpus_files(
    corpus_root: Path,
    extensions: set[str],
    exclude_tool_output: bool = True,
    strip_docstrings: bool = False,
    language: str = "python",
) -> Iterator[Path]:
    """
    Iterate over source files in a corpus with optional filtering.

    Args:
        corpus_root: Root directory of the corpus.
        extensions: Set of file extensions to include (e.g., {".py"}).
        exclude_tool_output: If True, skip files in tool output directories.
        strip_docstrings: If True, process Python files to strip docstrings
            (NOTE: this is read-only; the files themselves are not modified).
        language: Language for docstring stripping ("python" or other).

    Yields:
        Paths to source files matching the criteria.
    """
    for file_path in corpus_root.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip .git
        if ".git" in file_path.parts:
            continue

        # Check extension
        if file_path.suffix not in extensions:
            continue

        # Check for tool output
        if exclude_tool_output and is_tool_output(file_path, corpus_root):
            continue

        yield file_path


def read_source_file(
    file_path: Path,
    strip_docstrings: bool = False,
    language: str = "python",
) -> str:
    """
    Read a source file with optional docstring stripping.

    Args:
        file_path: Path to the source file.
        strip_docstrings: If True, strip docstrings from the content.
        language: Language for docstring stripping.

    Returns:
        File content (possibly with docstrings stripped).
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    if strip_docstrings and language == "python":
        content = strip_python_docstrings(content)

    return content
