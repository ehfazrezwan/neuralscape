"""Wave 3 neighbors resolver — Jedi-based static call resolution (Python).

The heuristic parser attributed every call to its MODULE and minted a phantom
``{module}.{rawtext}`` target, so ``_store_file``'s "both endpoints must exist"
MATCH dropped essentially all CALLS edges — native ``neighbors`` was ~0 by design.

This resolver runs at INDEX time (off the interactive path) and statically
resolves each Python call to the REAL definition it targets, so CALLS edges land
on real symbols and neighbors becomes a genuine, type-resolved call graph.

**Why Jedi, not stack-graphs.** The brief's stretch primary,
``tree-sitter-stack-graphs``, is a Rust crate not published to PyPI — vendoring it
means a Rust toolchain in the image (a container-gate landmine). Jedi is
pure-Python (only ``parso``), mature for Python, and resolves names to their
definition file+line — exactly what we need. The brief explicitly sanctions this
fallback for the Python corpus; other languages / SCIP / multilspy are later.

The resolver returns each call's definition as ``(def_abs_path, def_line)``; the
engine maps that to the exact stored symbol FQN (by file + span containment), so a
resolved target always matches an existing ``:CodeSymbol`` — never a new phantom.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Definition kinds we treat as real call targets. A call resolving to a variable,
# parameter, module, or keyword is not a function-call edge in the graph.
_CALLABLE_TYPES = frozenset({"function", "class"})


class JediCallResolver:
    """Resolve Python call sites to their definition (file, line) via Jedi.

    One :class:`jedi.Project` rooted at the repo drives cross-file import
    resolution; one :class:`jedi.Script` is built per file. Fully best-effort:
    any Jedi failure yields an unresolved ``(None, None)`` for that site, so a
    resolver hiccup degrades neighbors gracefully rather than failing the index.
    """

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root)
        self._project = None

    def _get_project(self):
        if self._project is None:
            import jedi

            self._project = jedi.Project(str(self.repo_root))
        return self._project

    def resolve_file(
        self, abs_path: str | Path, source_code: str, sites: list[tuple[int, int]]
    ) -> list[tuple[str | None, int | None]]:
        """Resolve every ``(line, column)`` call site in one file.

        Returns a list parallel to ``sites``: ``(def_abs_path, def_line)`` when the
        call resolves to a function/class definition, else ``(None, None)``.
        ``line`` is 1-based, ``column`` 0-based (tree-sitter / Jedi convention).
        """
        if not sites:
            return []
        try:
            import jedi

            script = jedi.Script(
                code=source_code, path=str(abs_path), project=self._get_project()
            )
        except Exception:
            logger.debug("jedi Script build failed for %s", abs_path, exc_info=True)
            return [(None, None)] * len(sites)

        out: list[tuple[str | None, int | None]] = []
        for line, col in sites:
            out.append(self._resolve_one(script, line, col))
        return out

    @staticmethod
    def _resolve_one(script, line: int, col: int) -> tuple[str | None, int | None]:
        try:
            defs = script.goto(
                line, col, follow_imports=True, follow_builtin_imports=False
            )
        except Exception:
            return (None, None)
        for d in defs or []:
            try:
                module_path = getattr(d, "module_path", None)
                if module_path is None:
                    continue  # built-in / compiled / dynamic — not in the repo
                if getattr(d, "type", None) not in _CALLABLE_TYPES:
                    continue  # variable/param/module/keyword — not a call edge
                return (str(module_path), int(d.line))
            except Exception:
                continue
        return (None, None)
