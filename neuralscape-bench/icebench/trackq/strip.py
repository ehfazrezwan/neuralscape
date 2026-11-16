"""
Docstring stripping for the Track-Q ``nl_locate`` corpus.

``nl_locate`` is CodeSearchNet-style: a function's docstring is the query and
the function's (file, symbol) is the gold answer. That is only a retrieval test
if the docstring is NOT present in what the systems index. Without stripping,
the query text sits verbatim inside the indexed target (symbol cards embed the
docstring) and the score measures self-retrieval.

This module builds a docstring-stripped COPY of a corpus. Queries are still
generated from the ORIGINAL corpus (docstrings intact); every system indexes
and is queried against the stripped copy.

Invariants (verified per file, otherwise the file is left unchanged and
reported in ``unstripped_files``):

* line numbers are preserved: every definition keeps its ``lineno`` and
  ``end_lineno``, so the oracle's (file, symbol, line) ground truth remains
  valid on the copy;
* the stripped file still parses.

Languages covered: Python (AST based; module, class, function and async
function docstrings) and Go (``//`` doc-comment runs that precede a ``func``).
Other languages are returned unchanged with status ``not_applicable``;
``nl_locate`` generation itself only supports Python and Go.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from icebench.adapters.base import Corpus

logger = logging.getLogger(__name__)

# Bump when the stripping behaviour changes (invalidates cached copies).
STRIPPER_VERSION = 2

# Marker file written into a finished stripped copy.
MARKER_NAME = ".ice-docstrings-stripped.json"

# Languages the stripper handles (matches what nl_locate generation supports).
LANGUAGES_COVERED = ("python", "go")

# Never copied into the stripped tree (corpus ROOT only, so a nested
# pkg/snapshot-helpers.py is kept): per-system output written by adapters into
# the corpus directory, which must not leak between systems or runs, and .git.
_SKIP_NAMES = {"graphify-out", MARKER_NAME, ".git"}
_SKIP_PREFIXES = ("snapshot-",)

PROTOCOL_STRIPPED = "docstring-stripped corpus"
PROTOCOL_LEAKED = "upper bound, leaked (docstrings NOT stripped from the indexed corpus)"


def protocol_for(reports: list["StripReport"]) -> str:
    """Label for a stripped run: downgraded if any file or query still leaks."""
    files = sum(len(r.unstripped_files) for r in reports)
    leaks = sum(r.residual_query_leaks or 0 for r in reports)
    if files or leaks:
        return f"partially stripped ({files} files unstripped, {leaks} leaking queries)"
    return PROTOCOL_STRIPPED


PROTOCOL_UNKNOWN = (
    "unknown: run predates docstring stripping; treat nl_locate as upper bound, leaked"
)


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------


def _docstring_nodes(tree: ast.Module) -> list[tuple[ast.Expr, bool, bool]]:
    """Return (expr, is_module, is_sole_statement) for every docstring."""
    found: list[tuple[ast.Expr, bool, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.append((first, isinstance(node, ast.Module), len(body) == 1))
    return found


def _def_spans(tree: ast.Module) -> list[tuple[str, str, int, int | None]]:
    """(kind, name, lineno, end_lineno) of every def/class, in a stable order."""
    spans = [
        (type(n).__name__, n.name, n.lineno, getattr(n, "end_lineno", None))
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return sorted(spans, key=lambda s: (s[2], s[1], s[0]))


def strip_python_source(source: str) -> tuple[str, int]:
    """
    Remove every module/class/function docstring from Python source.

    Returns ``(new_source, n_stripped)``. Line numbers of all definitions are
    preserved. If the source does not parse, or the result would fail the
    invariants, the original source is returned with ``n_stripped == -1``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source, -1

    targets = _docstring_nodes(tree)
    if not targets:
        return source, 0

    # ast columns are UTF-8 byte offsets, so edit bytes.
    lines = source.encode("utf-8").split(b"\n")

    # Edit from the bottom up so earlier line/col positions stay valid.
    targets.sort(key=lambda t: (t[0].lineno, t[0].col_offset), reverse=True)
    for expr, is_module, is_sole in targets:
        l0, l1 = expr.lineno - 1, expr.end_lineno - 1  # type: ignore[operator]
        c0, c1 = expr.col_offset, expr.end_col_offset  # type: ignore[assignment]
        head = lines[l0][:c0]
        tail = lines[l1][c1:]
        n_nl = l1 - l0
        full_line = not head.strip() and (
            not tail.strip() or tail.strip().startswith(b"#")
        )
        if full_line and (is_module or not is_sole):
            repl = b""  # blank line(s); nothing else needs the statement
        elif full_line:
            # Sole body statement of a def/class: `pass` goes on the docstring's
            # LAST line (earlier lines blank) so the def's end_lineno is kept.
            new = b"\n" * n_nl + head + b"pass" + tail
            lines[l0 : l1 + 1] = new.split(b"\n")
            continue
        else:
            # Shares a line with other code: keep it a valid expression
            # statement spanning the same lines.
            repl = b"(None" + b"\n" * n_nl + b")"
            new = head + repl + tail
            lines[l0 : l1 + 1] = new.split(b"\n")
            continue
        new = head + repl + b"\n" * n_nl + tail
        lines[l0 : l1 + 1] = new.split(b"\n")

    out = b"\n".join(lines).decode("utf-8")

    # Verify the invariants; never emit a file that moved the oracle's spans.
    try:
        new_tree = ast.parse(out)
    except (SyntaxError, ValueError):
        return source, -1
    if out.count("\n") != source.count("\n") or _def_spans(new_tree) != _def_spans(tree):
        return source, -1
    return out, len(targets)


# --------------------------------------------------------------------------
# Go
# --------------------------------------------------------------------------

# Machine directives (//go:build, //go:generate, //nolint:..., //export x) are
# not documentation and some change compilation; keep them.
_GO_DIRECTIVE = re.compile(r"^//(?:[a-z0-9_]+:|export\b|line\b)")


def strip_go_source(source: str) -> tuple[str, int]:
    """
    Blank the ``//`` doc-comment runs that precede a ``func`` declaration.

    Mirrors the nl_locate Go extractor (a comment run followed, after optional
    blank lines, by a line starting with ``func ``). Comment lines become empty
    lines, so line numbers are preserved. Returns ``(new_source, n_runs)``.
    """
    lines = source.split("\n")
    n_runs = 0
    i = 0
    while i < len(lines):
        start = i
        while i < len(lines) and lines[i].strip().startswith("//"):
            i += 1
        run = list(range(start, i))
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if run and j < len(lines) and lines[j].strip().startswith("func "):
            blanked = False
            for k in run:
                text = lines[k].strip()
                if _GO_DIRECTIVE.match(text):
                    continue
                lines[k] = "\r" if lines[k].endswith("\r") else ""
                blanked = True
            n_runs += 1 if blanked else 0
        if i == start:
            i += 1
    return "\n".join(lines), n_runs


# --------------------------------------------------------------------------
# Tree copy
# --------------------------------------------------------------------------


@dataclass
class StripReport:
    """What was done to one corpus (serialised into the run sidecar)."""

    corpus: str
    language: str
    status: str  # "stripped" | "reused" | "not_applicable"
    source_path: str
    stripped_path: str
    files_scanned: int = 0
    files_changed: int = 0
    docstrings_stripped: int = 0
    unstripped_files: list[str] = field(default_factory=list)
    residual_query_leaks: int | None = None
    nl_locate_candidates: int | None = None
    stripper_version: int = STRIPPER_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def stripped_path_for(corpus: Corpus) -> Path:
    """Sibling directory ``<corpus>-nodoc`` (keeps the /<root>/<name>/ depth)."""
    p = Path(corpus.path)
    return p.with_name(p.name + "-nodoc")


def _ext_for(language: str) -> str | None:
    return {"python": ".py", "go": ".go"}.get(language)


def _skip(name: str) -> bool:
    return name in _SKIP_NAMES or name.startswith(_SKIP_PREFIXES)


def _manifest(root: Path, ext: str) -> dict[str, str]:
    """sha256 of every source file under root (relative path -> digest)."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob(f"*{ext}")):
        rel = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or ".git" in rel.parts:
            continue
        out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _load_marker(dest: Path) -> dict | None:
    try:
        return json.loads((dest / MARKER_NAME).read_text())
    except (OSError, ValueError):
        return None


def build_stripped_corpus(corpus: Corpus) -> tuple[Corpus, StripReport]:
    """
    Return ``(corpus_to_index, report)``.

    For a covered language, copies the tree to ``<corpus.path>-nodoc`` with all
    docstrings removed and returns a Corpus pointing at the copy (name, sha and
    everything else unchanged, so result rows and gold alignment are
    unaffected). An existing, marker-valid copy is REUSED untouched: adapters
    keep their on-disk indexes (e.g. graphify-out) inside the corpus dir, so it
    must survive between the index and query phases. For other languages the
    original corpus is returned with status ``not_applicable``.
    """
    src = Path(corpus.path)
    dest = stripped_path_for(corpus)
    ext = _ext_for(corpus.language)
    if ext is None:
        return corpus, StripReport(
            corpus=corpus.name, language=corpus.language, status="not_applicable",
            source_path=str(src), stripped_path=str(src),
        )

    marker = _load_marker(dest)
    if (
        marker
        and marker.get("stripper_version") == STRIPPER_VERSION
        and marker.get("repo_sha") == corpus.repo_sha
        and marker.get("manifest", {}).get("src") == _manifest(src, ext)
        and marker.get("manifest", {}).get("dest") == _manifest(dest, ext)
    ):
        report = StripReport(**{**marker["report"], "status": "reused"})
        return dataclasses.replace(corpus, path=str(dest)), report

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest, symlinks=True,
        ignore=lambda d, names: (
            [n for n in names if _skip(n)] if Path(d) == src else []
        ),
    )

    report = StripReport(
        corpus=corpus.name, language=corpus.language, status="stripped",
        source_path=str(src), stripped_path=str(dest),
    )
    strip_fn = strip_python_source if corpus.language == "python" else strip_go_source
    for path in sorted(dest.rglob(f"*{ext}")):
        if not path.is_file() or path.is_symlink() or ".git" in path.relative_to(dest).parts:
            continue
        report.files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            report.unstripped_files.append(str(path.relative_to(dest)))
            continue
        new, n = strip_fn(text)
        if n < 0:
            report.unstripped_files.append(str(path.relative_to(dest)))
            continue
        if new != text:
            path.write_text(new, encoding="utf-8")
            report.files_changed += 1
            report.docstrings_stripped += n

    stripped = dataclasses.replace(corpus, path=str(dest))
    report.residual_query_leaks, report.nl_locate_candidates = count_residual_leaks(
        corpus, stripped
    )
    if report.unstripped_files:
        logger.warning(
            "%s: %d file(s) could not be stripped safely: %s",
            corpus.name, len(report.unstripped_files), report.unstripped_files[:5],
        )
    if report.residual_query_leaks:
        logger.warning(
            "%s: %d nl_locate query text(s) still appear in their gold file after "
            "stripping", corpus.name, report.residual_query_leaks,
        )
    (dest / MARKER_NAME).write_text(
        json.dumps(
            {
                "stripper_version": STRIPPER_VERSION,
                "repo_sha": corpus.repo_sha,
                "report": report.to_dict(),
                "manifest": {"src": _manifest(src, ext), "dest": _manifest(dest, ext)},
            },
            indent=2,
        )
    )
    return stripped, report


def count_residual_leaks(original: Corpus, stripped: Corpus) -> tuple[int, int]:
    """
    Count nl_locate queries whose text still occurs in their gold file in the
    stripped copy. Returns ``(leaks, candidates)``. A non-zero count is
    reported in the run metadata rather than hidden.

    Limit: the Go extractor space-joins multi-line ``//`` comments into one
    query, so a substring check cannot see those queries leaking; for Go this
    count is a lower bound (the stripper itself blanks the whole run).
    """
    from icebench.trackq.generate import generate_specs

    # generate_specs logs an UNDER-SIZED warning for small corpora; the real
    # generation already does, so keep this internal pass quiet.
    gen_logger = logging.getLogger("icebench.trackq.generate")
    prior_level = gen_logger.level
    gen_logger.setLevel(logging.ERROR)
    try:
        specs = generate_specs("nl_locate", original, n=10**9, seed=0)
    finally:
        gen_logger.setLevel(prior_level)
    leaks = 0
    root = Path(stripped.path)
    for spec in specs:
        try:
            text = (root / spec.gold["file"]).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if spec.payload["query"] in text:
            leaks += 1
    return leaks, len(specs)
