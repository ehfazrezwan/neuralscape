"""Pluggable knowledge adapters.

An adapter swaps the ingest pipeline's taxonomy, chunking strategy, fact
extractor, and Graphiti graph ontology while keeping the fixed metadata envelope
identical. See :mod:`adapters.base`.

Importing this package registers every built-in adapter as a side effect (each
adapter module calls :func:`~adapters.base.register_adapter` and, if it defines
new categories, :func:`schemas.register_categories` at import).
"""

from adapters.base import (
    ADAPTER_REGISTRY,
    DEFAULT_ADAPTER,
    DEFAULT_ADAPTER_NAME,
    KnowledgeAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)

# ── Register built-in adapters (import side effects) ──
# The trading adapter registers itself + its taxonomy on import. Guarded: a
# broken adapter module must degrade to "that adapter is unavailable" (requests
# naming it get a 422 from validate_adapter_name), NOT take down this package —
# ingest/pipeline imports `adapters` at module top, so an unguarded failure here
# would kill DEFAULT ingestion along with it.
import logging as _logging

try:
    from adapters import trading as _trading  # noqa: F401,E402
except Exception:  # noqa: BLE001 — adapter registration is best-effort
    _logging.getLogger(__name__).exception(
        "trading_strategy adapter failed to register — continuing without it"
    )

__all__ = [
    "ADAPTER_REGISTRY",
    "DEFAULT_ADAPTER",
    "DEFAULT_ADAPTER_NAME",
    "KnowledgeAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
