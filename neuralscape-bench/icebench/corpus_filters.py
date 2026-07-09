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
    # be more robust but adds complexity. The regex handles the common cases:
    # module / def / async def / class docstrings, optionally with a string
    # prefix (r/b/f/u and combinations, case-insensitive), and separated from
    # the def/class header by blank or comment lines.

    # A triple-quoted string literal with an OPTIONAL prefix (r, b, f, u, rb, ...).
    # Both """ and ''' variants; DOTALL is set globally so content may span lines.
    _prefix = r'(?:[rRbBuUfF]{0,2})'
    _dq = r'"""(?:[^"]|"(?!""))*"""'
    _sq = r"'''(?:[^']|'(?!''))*'''"
    _docstring = rf'{_prefix}(?:{_dq}|{_sq})'

    # First pass: module-level docstring at the very beginning.
    # Match optional shebang/encoding, then optional whitespace/comments, then
    # the (possibly prefixed) docstring.
    module_pattern = re.compile(
        r'^((?:#![^\n]*\n)?(?:[ \t]*#[^\n]*\n|[ \t]*\n)*)'  # shebang + comments/blank lines
        rf'([ \t]*)({_docstring})',  # the docstring (group 3)
        re.MULTILINE | re.DOTALL
    )
    result = module_pattern.sub(r'\1', source, count=1)  # keep prefix, drop docstring

    # Second pass: function / async function / class docstrings. The docstring may
    # be preceded by blank lines and/or comment lines after the header line.
    func_class_pattern = re.compile(
        r'((?:^|\n)[ \t]*(?:async[ \t]+)?(?:def|class)\s+[^\n]+:[ \t]*\n'  # header line (grp 1 start)
        r'(?:[ \t]*#[^\n]*\n|[ \t]*\n)*)'  # optional blank/comment lines (still grp 1)
        rf'([ \t]*)({_docstring})',  # indentation (grp 2) + docstring (grp 3)
        re.MULTILINE | re.DOTALL
    )

    def replace_func_docstring(match):
        # Keep the header (and any interleaving blank/comment lines), drop the docstring.
        return match.group(1)

    result = func_class_pattern.sub(replace_func_docstring, result)

    return result


def filter_corpus_files(
    corpus_root: Path,
    extensions: set[str],
    exclude_tool_output: bool = True,
) -> Iterator[Path]:
    """
    Iterate over source files in a corpus with optional filtering.

    This yields PATHS only. To read content (optionally with docstring
    stripping), pass each yielded path to ``read_source_file``.

    Args:
        corpus_root: Root directory of the corpus.
        extensions: Set of file extensions to include (e.g., {".py"}).
        exclude_tool_output: If True, skip files in tool output directories.

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
