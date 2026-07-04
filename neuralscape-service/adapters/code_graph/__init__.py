"""The ``code_graph`` knowledge adapter — Graphify-backed code knowledge.

Maintainer decision (roadmap Phase F): **coding-domain knowledge defers to
Graphify for code *structure*; Neuralscape stores the knowledge *about* the
code.** Graphify (PyPI: ``graphifyy``) is used strictly as a library — NS never
rebuilds its extraction/query layer and never points clients at Graphify's own
MCP server. The interaction surface is ALWAYS Neuralscape (the
``query_code_graph`` / ``get_code_neighbors`` / ``code_path`` tools).

This package is import-safe without the optional ``code-graph`` extra: only
:func:`code_graph_available` and :func:`register` live here, and both degrade
gracefully (a clear log line, no crash) when ``graphify`` isn't importable.
The graphify-dependent modules (``profile``, ``semantic``, ``query``) import
the library at module level and are only imported behind the availability gate.
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)

ADAPTER_NAME = "code_graph"

_MISSING_EXTRA_MSG = (
    "code_graph adapter unavailable — the optional Graphify dependency is not "
    "installed. Install the 'code-graph' extra (uv sync --extra code-graph / "
    "pip install 'neuralscape-service[code-graph]') to enable code-graph "
    "ingestion and the query_code_graph / get_code_neighbors / code_path tools."
)


def code_graph_available() -> bool:
    """True when the optional ``graphifyy`` library is actually importable.

    ``find_spec`` first, as the cheap negative (probing availability on every
    MCP ``list_tools`` call must not pay an import attempt when the extra
    plainly isn't installed) — then a real ``import`` so a broken/partial
    install reports unavailable and degrades cleanly instead of surfacing
    ImportErrors from the gated tools/routes. The import is a no-op after the
    first success (``sys.modules`` cache). Tests simulate the missing-extra
    case by poisoning ``sys.modules["graphify"]``.
    """
    try:
        if importlib.util.find_spec("graphify") is None:
            return False
    except (ImportError, ValueError):
        return False
    try:
        import graphify  # noqa: F401 — availability == real importability
    except Exception:  # noqa: BLE001 — any broken install ⇒ unavailable
        return False
    return True


def register() -> bool:
    """Register the ``code_graph`` adapter if Graphify is installed.

    Called from ``adapters/__init__`` at import. Returns True when the adapter
    (taxonomy + chunker + extractor + ontology) registered; False — with a
    clear, non-exception log line — when the ``code-graph`` extra is absent.
    """
    if not code_graph_available():
        # Covers both "extra not installed" and "installed but broken" —
        # code_graph_available() performs a real import attempt.
        logger.info(_MISSING_EXTRA_MSG)
        return False
    from adapters.code_graph import profile

    profile.register()
    return True
