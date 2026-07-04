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
    """True when the optional ``graphifyy`` library is importable.

    Checked via ``find_spec`` so merely probing availability (e.g. on every
    MCP ``list_tools`` call) never pays graphify's import cost; the actual
    import happens lazily inside the gated modules. Tests simulate the
    missing-extra case by poisoning ``sys.modules["graphify"]`` /
    monkeypatching this function.
    """
    try:
        return importlib.util.find_spec("graphify") is not None
    except (ImportError, ValueError):
        return False


def register() -> bool:
    """Register the ``code_graph`` adapter if Graphify is installed.

    Called from ``adapters/__init__`` at import. Returns True when the adapter
    (taxonomy + chunker + extractor + ontology) registered; False — with a
    clear, non-exception log line — when the ``code-graph`` extra is absent.
    """
    if not code_graph_available():
        logger.info(_MISSING_EXTRA_MSG)
        return False
    try:
        import graphify  # noqa: F401 — confirm the import actually succeeds
    except Exception:  # noqa: BLE001 — a broken install must not crash adapters
        logger.warning(_MISSING_EXTRA_MSG, exc_info=True)
        return False
    from adapters.code_graph import profile

    profile.register()
    return True
