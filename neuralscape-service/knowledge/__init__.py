"""Knowledge system registry — pluggable knowledge backends for routing + fusion.

Importing this package registers every available knowledge system as a side effect:
  - NSMemorySystem (base): always registered (the default, always-eligible system).
  - Code systems: registered when the code-graph extra is available.

Phase B scope (pure refactor, ZERO behavior change):
  - Base system wraps existing MemoryService search facade.
  - Code system wraps existing GraphifyJsonEngine (artifact path).
  - Native engine (code-native) is EXCLUDED from default registry per LOCKED
    decision #1 (CODE_NATIVE_ENABLED opt-in only; Phase A applies fixes but
    the system stays frozen).

Phases C/F will add CBMEngine and GraphifyLibEngine.
"""

from __future__ import annotations

import logging

from knowledge.base import (
    HealthStatus,
    IndexReport,
    IndexRequest,
    KnowledgeSystem,
    KnowledgeSystemInfo,
    RecallRequest,
    SystemAnswer,
    TaskRef,
)
from knowledge.registry import (
    KNOWLEDGE_REGISTRY,
    eligible_systems,
    get_system,
    list_systems,
    register_system,
)

logger = logging.getLogger(__name__)

# ── Register base system (always) ────────────────────────────────────

from knowledge.ns_memory import NSMemorySystem

_base_system = NSMemorySystem()
register_system(_base_system)

# ── Register code systems (gated by code-graph extra availability) ────

# Phase B: wrap ONLY the existing GraphifyJsonEngine artifact path.
# NativeEngine wrapper exists but is NOT registered by default (decision #1:
# absent from default registry; explicit CODE_NATIVE_ENABLED opt-in only).
# CBMEngine and GraphifyLibEngine are Phases C and F.

try:
    # Check if code-graph extra is available (same gate as adapters/code_graph).
    from adapters.code_graph import code_graph_available

    if code_graph_available():
        # The code-graph extra is installed; we CAN import the engine types.
        # But Phase B only wires GraphifyJsonEngine via the existing artifact
        # path — there's no standalone registry entry for it yet because it's
        # already wired via query.py's get_engine() factory (graph_id resolution).
        #
        # The wrapper exists (CodeKnowledgeSystem can wrap any CodeIntelEngine),
        # but registration defers to Phase C/F when we add the new backends
        # (CBM, graphify-lib) that need their own registry entries.
        #
        # For now, log that code systems are available but not auto-registered.
        logger.info(
            "code-graph extra is available; code system wrappers ready "
            "(registration deferred to Phase C/F for new backends)"
        )
    else:
        logger.info(
            "code-graph extra not installed; code systems unavailable "
            "(install via `uv sync --extra code-graph`)"
        )
except Exception:  # noqa: BLE001 — code system registration is best-effort
    logger.exception(
        "Code system registration failed — continuing with base system only"
    )

__all__ = [
    # Base types
    "KnowledgeSystem",
    "KnowledgeSystemInfo",
    "HealthStatus",
    "RecallRequest",
    "SystemAnswer",
    "IndexRequest",
    "IndexReport",
    "TaskRef",
    # Registry accessors
    "KNOWLEDGE_REGISTRY",
    "register_system",
    "get_system",
    "list_systems",
    "eligible_systems",
]
