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
        # when the code-graph extra is installed).
        #
        # GraphifyLibEngine is per-code_space (like NativeEngine), so real engine
        # instances are created on-demand by the query.py factory
        # (_get_graphify_lib_engine), cached per code_space in _ctx_cache. The
        # REGISTRY ENTRY represents the CAPABILITY (graphify library importable /
        # per-code_space factory available) — NOT a specific loaded graph.
        #
        # The wrapped engine here is a capability placeholder: its health() probes
        # that graphify imports cleanly (independent of any resident graph), so
        # code-graphify-lib is ELIGIBLE for routing whenever the extra is present
        # (eligible_systems requires health=="ok"). transport="in-process".
        try:
            from knowledge.code_system import CodeKnowledgeSystem
            from adapters.code_graph.graphify_lib_engine import GraphifyLibEngine

            # Capability placeholder: health() checks graphify importability, not a
            # loaded graph, so G=None does NOT make the system ineligible. Real
            # per-code_space instances are built by the query.py factory on demand.
            capability_engine = GraphifyLibEngine(
                code_space="__registry_capability__",
                source_root="/dev/null",  # never used; factory builds real instances
            )

            graphify_lib_system = CodeKnowledgeSystem(
                name="code-graphify-lib",
                engine=capability_engine,
                capabilities=frozenset({
                    "query",
                    "neighbors",
                    "path",
                    "index",
                    "impact",  # detect_changes(seed) → blast radius (git-less repos)
                }),
                transport="in-process",
                version=None,  # Graphify lib version (fetched lazily if needed)
            )
            register_system(graphify_lib_system)
            logger.info("Registered code-graphify-lib system (in-process library, per-code_space)")
        except Exception as e:
            logger.warning("GraphifyLibEngine registration failed: %s", e)

        # Phase G-final (GF3): register the frozen NativeEngine as a first-class
        # code system when CODE_NATIVE_ENABLED=true. Default-OFF (LOCKED decision
        # #1 keeps native a frozen opt-in fallback in prod); the bench stack opts
        # in so native runs the SAME through-NS routing + POST /v1/code-graph/index
        # path as cbm/graphify-lib for a clean apples-to-apples comparison.
        #
        # Like graphify-lib, NativeEngine is per-code_space: this entry is a
        # CAPABILITY placeholder (code_space="__registry_capability__"; real
        # per-space instances are built by the query.py factory on demand). Its
        # __init__ is lazy (no Neo4j at construction), so the placeholder is safe
        # to build at import even when Neo4j is down.
        native_enabled = os.getenv("CODE_NATIVE_ENABLED", "false").lower() == "true"
        if native_enabled:
            try:
                from adapters.code_graph.native_engine import NativeEngine
                from knowledge.code_system import CodeKnowledgeSystem

                try:
                    from config import settings as _ns_settings
                except Exception:  # noqa: BLE001 — settings optional for placeholder
                    _ns_settings = None

                native_capability_engine = NativeEngine(
                    repo_path="",  # never used; real instances built by factory
                    code_space="__registry_capability__",
                    bridge=None,
                    settings=_ns_settings,
                    driver=None,
                )
                native_system = CodeKnowledgeSystem(
                    name="code-native",
                    engine=native_capability_engine,
                    # NativeEngine supports every op (symbol_lookup via locate,
                    # neighbors, path, blast_radius via impact, query, index).
                    capabilities=frozenset(
                        {"query", "neighbors", "path", "locate", "impact", "index"}
                    ),
                    transport="in-process",
                    version=None,
                )
                register_system(native_system)
                logger.info(
                    "Registered code-native system (CODE_NATIVE_ENABLED=true; frozen fallback)"
                )
            except Exception as e:
                logger.warning("NativeEngine registration failed: %s", e)
        else:
            logger.info(
                "code-native not enabled (set CODE_NATIVE_ENABLED=true to enable "
                "the frozen native engine as a first-class code system)"
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
