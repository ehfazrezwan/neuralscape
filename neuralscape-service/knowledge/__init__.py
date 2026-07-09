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

# Phase C: register CBMEngine when CBM bridge is available/enabled.
# Phase F: register GraphifyLibEngine.
# NativeEngine wrapper exists but is NOT registered by default (decision #1:
# absent from default registry; explicit CODE_NATIVE_ENABLED opt-in only).

try:
    # Check if code-graph extra is available (same gate as adapters/code_graph).
    from adapters.code_graph import code_graph_available

    if code_graph_available():
        logger.info("code-graph extra is available; registering code systems")

        # Phase C: register CBMEngine if CBM_ENABLED=true and bridge is reachable.
        import os
        cbm_enabled = os.getenv("CBM_ENABLED", "false").lower() == "true"
        cbm_bridge_url = os.getenv("CBM_BRIDGE_URL", "http://cbm-bridge:8200")

        if cbm_enabled:
            try:
                from adapters.code_graph.cbm_engine import CBMEngine
                from knowledge.code_system import CodeKnowledgeSystem

                # Create a CBMEngine instance (no project yet; set on index).
                # Health check will verify bridge reachability.
                cbm_engine = CBMEngine(bridge_url=cbm_bridge_url)

                # Wrap as CodeKnowledgeSystem
                cbm_system = CodeKnowledgeSystem(
                    name="code-cbm",
                    engine=cbm_engine,
                    capabilities=frozenset({
                        "query",
                        "neighbors",
                        "locate",
                        "index",
                    }),
                    transport="http",
                    version=None,  # Fetched lazily on first health check
                )
                register_system(cbm_system)
                logger.info("Registered code-cbm system (CBM bridge at %s)", cbm_bridge_url)
            except Exception as e:
                logger.warning(
                    "CBM system registration failed (CBM_ENABLED=true but bridge unreachable): %s",
                    e,
                )
        else:
            logger.info("CBM system not enabled (set CBM_ENABLED=true to enable)")

        # Phase F: register GraphifyLibEngine (in-process library, always available
        # when code-graph extra is installed).
        #
        # GraphifyLibEngine is per-code_space (like NativeEngine), so we DON'T register
        # a singleton engine instance here. Instead, we register a MARKER system that
        # declares code-graphify-lib is available as a routing target. The actual
        # GraphifyLibEngine instances are created on-demand by the query.py factory
        # (_get_graphify_lib_engine), cached per code_space in _ctx_cache.
        #
        # This registration makes "code-graphify-lib" ELIGIBLE for routing and
        # shows up in health checks / registry listings (transport="in-process").
        try:
            from knowledge.code_system import CodeKnowledgeSystem
            from adapters.code_graph.graphify_lib_engine import GraphifyLibEngine

            # Create a placeholder engine (no code_space yet; engine is created per-space
            # in query.py). The wrapper's health() will check that graphify imports work.
            # We use a dummy instance just for the registration; real instances are
            # created per code_space on-demand.
            placeholder_engine = GraphifyLibEngine(
                code_space="__registry_placeholder__",
                source_root="/dev/null",  # never used
            )

            graphify_lib_system = CodeKnowledgeSystem(
                name="code-graphify-lib",
                engine=placeholder_engine,
                capabilities=frozenset({
                    "query",
                    "neighbors",
                    "path",
                    "index",
                    "impact",  # detect_changes with seed symbol
                }),
                transport="in-process",
                version=None,  # Graphify lib version (fetched lazily if needed)
            )
            register_system(graphify_lib_system)
            logger.info("Registered code-graphify-lib system (in-process library, per-code_space)")
        except Exception as e:
            logger.warning("GraphifyLibEngine registration failed: %s", e)

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
