"""Three-layer deterministic knowledge system router (Phase D).

Resolution order (first hit wins), per PLAN §4:
  1. **Explicit override**: optional `knowledge_system` param (additive to all
     recall/code tools); existing `graph_id` ref-shape dispatch (repo:/code--/.json)
     subsumed as an explicit code signal.
  2. **Project config default**: per-project settings document
     `{code_systems: [...], fuse_code_into_recall: bool (DEFAULT TRUE per decision #3),
     default_engine: str}`. Editable via REST/MCP; set at index time.
  3. **Deterministic signals** (only for generic `recall_memories`/`ask_memory` when
     layer 2 permits code fusion):
       - project has ≥1 healthy indexed code_space (necessary condition — cheap
         registry lookup via `eligible_systems`).
       - query-shape gate for the EXTRA code leg: FQN-ish token (r`\w+\.\w+\(`, `::`),
         path-like token (r`\w+/\w+\.\w{1,4}`), backticked snake_case/CamelCase
         identifier, or structural keywords (who calls / where is / defined / imports /
         blast radius). Plain-prose recall on a code project does NOT fan out to code
         (recall precision + latency floor). Ambiguity → base-only (additive: code is
         an enrichment leg, never a replacement).

Implemented as reviewable table-driven logic. Operates on registry entries via
`eligible_systems`; **NEVER branches on `transport`** (per DECISIONS.md cross-cutting
rule). Router overhead budget: <1 ms.

CRITICAL: Phase D only RESOLVES the route; it does NOT compose a code leg into
recall_memories output yet (that's Phase E fusion). Generic recall responses must be
byte-identical to today unless `knowledge_system` is explicitly given. This preserves
the latency floor and keeps D/E separable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.base import KnowledgeSystem

logger = logging.getLogger(__name__)

# ── Per-project config (stored in a future project-config store; stubbed for Phase D) ──


@dataclass
class ProjectKnowledgeConfig:
    """Per-project knowledge routing settings (editable via REST/MCP).

    Set at index time; governs which systems are eligible for this project's queries.
    Layer 2 of the router (project config default).
    """

    project_id: str
    code_systems: list[str] = None  # ["code-cbm"] or ["code-graphify"] etc.
    fuse_code_into_recall: bool = True  # DEFAULT TRUE per decision #3
    default_engine: str | None = None  # "code-cbm" or "code-graphify"

    def __post_init__(self):
        if self.code_systems is None:
            self.code_systems = []


# Stub: in-memory project config store (Phase D; Phase E will persist to Redis/Neo4j)
_PROJECT_CONFIGS: dict[str, ProjectKnowledgeConfig] = {}


def get_project_config(project_id: str | None) -> ProjectKnowledgeConfig | None:
    """Get the project's knowledge routing config, or None if not set."""
    if not project_id:
        return None
    return _PROJECT_CONFIGS.get(project_id)


def set_project_config(config: ProjectKnowledgeConfig) -> None:
    """Set (or update) a project's knowledge routing config."""
    _PROJECT_CONFIGS[config.project_id] = config


# ── Query-shape coding-signal gate (layer 3) ──

# FQN-ish: foo.bar( or foo::bar (method calls, namespaced refs)
_FQN_PATTERN = re.compile(r"\w+\.\w+\(|\w+::\w+")

# Path-like: foo/bar.py or src/utils/helper.ts (file paths)
_PATH_PATTERN = re.compile(r"\w+/\w+\.\w{1,4}")

# Backticked identifiers: `some_function`, `ClassName`, `snake_case`, `CamelCase`
_BACKTICK_IDENT_PATTERN = re.compile(r"`[a-zA-Z_][a-zA-Z0-9_]*`")

# Structural keywords (who calls / where is / defined / imports / blast radius)
_STRUCTURAL_KEYWORDS = frozenset({
    "who calls",
    "what calls",
    "where is",
    "defined",
    "definition",
    "imports",
    "blast radius",
    "impact",
    "callers",
    "callees",
    "dependencies",
    "dependents",
})


def _has_coding_signal(query: str) -> bool:
    """Layer 3 gate: does the query show coding-shaped signals?

    Plain-prose recall on a code project stays base-only to preserve the +20%
    latency floor. Only queries that LOOK like code questions get the extra code leg.

    Returns:
        True if the query has FQN tokens, path tokens, backticked identifiers, or
        structural keywords. False for plain prose.
    """
    query_lower = query.lower()

    # Check structural keywords first (cheapest)
    for keyword in _STRUCTURAL_KEYWORDS:
        if keyword in query_lower:
            logger.debug("Coding signal: structural keyword '%s'", keyword)
            return True

    # Check regex patterns
    if _FQN_PATTERN.search(query):
        logger.debug("Coding signal: FQN-ish token")
        return True

    if _PATH_PATTERN.search(query):
        logger.debug("Coding signal: path-like token")
        return True

    if _BACKTICK_IDENT_PATTERN.search(query):
        logger.debug("Coding signal: backticked identifier")
        return True

    logger.debug("No coding signal detected (plain prose)")
    return False


# ── The router: resolve_systems ──


@dataclass
class RouteDecision:
    """What the router decided and why (for debug logging + tests).

    ``wants_code_fusion`` + ``code_system_names`` are the STRUCTURED fusion signal
    consumed by the Phase E recall wiring — never infer fusion from ``rationale``
    text (a wording tweak must not silently toggle behavior). When
    ``wants_code_fusion`` is True, ``code_system_names`` holds the resolved code
    system name(s) to compose into recall, honoring project config
    (``default_engine`` first, else ``code_systems``).
    """

    systems: list[KnowledgeSystem]
    rationale: str  # Human-readable explanation (layer hit, signal present/absent, etc.)
    layer: int  # 1=explicit, 2=project config, 3=signal-based
    wants_code_fusion: bool = False  # Structured fusion gate (consumed by recall wiring)
    code_system_names: list[str] = field(default_factory=list)  # Resolved code system(s) for fusion


def resolve_systems(
    query: str,
    project_id: str | None = None,
    knowledge_system: str | None = None,
    graph_id: str | None = None,
    operation: str | None = None,
    is_code_tool: bool = False,
) -> RouteDecision:
    """Three-layer deterministic resolver: which KnowledgeSystem(s) to query?

    Resolution order (first hit wins):
      1. Explicit override: `knowledge_system` param or `graph_id` ref-shape dispatch.
      2. Project config: per-project `code_systems` + `fuse_code_into_recall` settings.
      3. Deterministic signals: project has code + query has coding-signal.

    Args:
        query: The user's query string (used for signal detection in layer 3).
        project_id: Optional project scope.
        knowledge_system: Explicit system name (layer 1 override).
        graph_id: Existing ref-shape dispatch (repo:/code--/.json) — subsumed as
            an explicit code signal (layer 1).
        operation: Op-class hint (query/neighbors/path/locate/impact) for capability
            filtering.
        is_code_tool: True when called from one of the 5 code MCP tools (always routes
            to a code system; the only choice is WHICH backend).

    Returns:
        RouteDecision: which systems to query + why.

    CRITICAL: This is Phase D (RESOLVE only). Phase E will use the decision to
    compose fusion responses. For now, generic recall (no explicit knowledge_system)
    MUST return base-only to stay byte-identical to today.
    """
    from knowledge.registry import eligible_systems, get_system

    # ── Layer 1: Explicit override ──

    # Explicit knowledge_system param (new in Phase D, additive)
    if knowledge_system:
        sys = get_system(knowledge_system)
        if sys is None:
            logger.warning(
                "Explicit knowledge_system '%s' not registered; falling back to base",
                knowledge_system,
            )
            base = get_system("ns-memory")
            return RouteDecision(
                systems=[base] if base else [],
                rationale=f"Explicit system '{knowledge_system}' not found; fallback to base",
                layer=1,
            )

        # Check health + capability
        health = sys.health()
        if health.status != "ok":
            logger.warning(
                "Explicit knowledge_system '%s' unhealthy (%s); falling back to base",
                knowledge_system,
                health.status,
            )
            base = get_system("ns-memory")
            return RouteDecision(
                systems=[base] if base else [],
                rationale=f"Explicit system '{knowledge_system}' unhealthy; fallback to base",
                layer=1,
            )

        if operation and operation not in sys.info.capabilities:
            logger.warning(
                "Explicit knowledge_system '%s' doesn't support operation '%s'; falling back to base",
                knowledge_system,
                operation,
            )
            base = get_system("ns-memory")
            return RouteDecision(
                systems=[base] if base else [],
                rationale=f"Explicit system '{knowledge_system}' lacks '{operation}' capability; fallback to base",
                layer=1,
            )

        logger.debug("Layer 1: explicit knowledge_system='%s'", knowledge_system)
        return RouteDecision(
            systems=[sys],
            rationale=f"Explicit knowledge_system='{knowledge_system}'",
            layer=1,
        )

    # Subsume existing graph_id ref-shape dispatch as explicit code signal (layer 1)
    if graph_id:
        # graph_id ref-shapes (query.py:87-149):
        #   - repo:<name>  → NativeEngine (code-native system)
        #   - code--<owner>--<repo>  → NativeEngine by code_space
        #   - .json artifact path  → GraphifyJsonEngine (code-graphify-json system)
        #
        # The 5 code tools already pass graph_id; this makes them explicit layer-1
        # routing (they never fall through to signal-based). For now, we resolve
        # graph_id to the appropriate code system name (Phase C+ will have the real
        # systems registered; Phase D uses the stub).

        if graph_id.startswith("repo:") or graph_id.startswith("code--"):
            # NativeEngine — map to code-native system (if registered)
            sys = get_system("code-native")
            if sys:
                logger.debug("Layer 1: graph_id='%s' → code-native", graph_id)
                return RouteDecision(
                    systems=[sys],
                    rationale=f"graph_id='{graph_id}' (native ref) → code-native",
                    layer=1,
                )
        elif graph_id.endswith(".json"):
            # GraphifyJsonEngine — map to code-graphify-json system (Phase B stub)
            sys = get_system("code-graphify-json")
            if sys:
                logger.debug("Layer 1: graph_id='%s' → code-graphify-json", graph_id)
                return RouteDecision(
                    systems=[sys],
                    rationale=f"graph_id='{graph_id}' (artifact) → code-graphify-json",
                    layer=1,
                )

        # If the ref-shape matched but the system isn't registered, log and continue
        # (don't fail; just fall through to base).
        logger.debug(
            "graph_id='%s' ref-shape recognized but target system not registered; falling back",
            graph_id,
        )

    # ── Layer 2: Project config default ──

    if project_id:
        proj_cfg = get_project_config(project_id)
        if proj_cfg:
            # Project has code_systems configured
            if proj_cfg.code_systems:
                # For code tools (is_code_tool=True), always route to the project's
                # code system (the default_engine or first in the list).
                if is_code_tool:
                    engine_name = proj_cfg.default_engine or proj_cfg.code_systems[0]
                    sys = get_system(engine_name)
                    if sys and sys.health().status == "ok":
                        logger.debug(
                            "Layer 2 (code tool): project '%s' default_engine='%s'",
                            project_id,
                            engine_name,
                        )
                        return RouteDecision(
                            systems=[sys],
                            rationale=f"Project config: code tool → '{engine_name}'",
                            layer=2,
                        )

                # For generic recall: only add code if fuse_code_into_recall=True AND
                # query has coding signal (decision #3 + latency floor).
                # CRITICAL: Phase D does NOT compose fusion yet; this decision is
                # recorded but not acted on in recall output (that's Phase E).
                if proj_cfg.fuse_code_into_recall and _has_coding_signal(query):
                    # Phase E: base always answers; the recall wiring composes the
                    # resolved code system(s) on top. Honor default_engine first,
                    # else the full code_systems list (respect project config —
                    # the wiring must NOT re-pick a backend).
                    base = get_system("ns-memory")
                    resolved = (
                        [proj_cfg.default_engine]
                        if proj_cfg.default_engine
                        else list(proj_cfg.code_systems)
                    )
                    logger.debug(
                        "Layer 2: project '%s' has code systems + coding signal detected "
                        "(fusion ON) — composing %s",
                        project_id,
                        resolved,
                    )
                    return RouteDecision(
                        systems=[base] if base else [],
                        rationale=(
                            f"Project config: code fusion enabled + coding signal detected "
                            f"(compose {resolved})"
                        ),
                        layer=2,
                        wants_code_fusion=True,
                        code_system_names=resolved,
                    )

                # fuse_code_into_recall=False or no coding signal → base-only
                if not proj_cfg.fuse_code_into_recall:
                    base = get_system("ns-memory")
                    logger.debug(
                        "Layer 2: project '%s' has code but fuse_code_into_recall=False",
                        project_id,
                    )
                    return RouteDecision(
                        systems=[base] if base else [],
                        rationale="Project config: code fusion disabled (fuse_code_into_recall=False)",
                        layer=2,
                    )

    # ── Layer 3: Deterministic signals (only for generic recall) ──

    # Layer 3 is only reached when:
    #   - No explicit override (layer 1)
    #   - No project config (layer 2) OR project config didn't trigger code
    #   - NOT a code tool (is_code_tool=False)
    #
    # Necessary condition: project has ≥1 healthy indexed code_space.
    # Query-shape gate: query has coding signal.

    if project_id and not is_code_tool:
        # Check if project has any healthy code systems
        code_systems = eligible_systems(project_id=project_id, kind="code")
        if code_systems and _has_coding_signal(query):
            # Phase E: base always answers; the recall wiring composes the resolved
            # code system(s) on top. No project config here, so use the eligible
            # healthy code systems (by name) the registry resolved.
            base = get_system("ns-memory")
            resolved = [s.info.name for s in code_systems]
            logger.debug(
                "Layer 3: project '%s' has code systems + coding signal detected "
                "(composing %s)",
                project_id,
                resolved,
            )
            return RouteDecision(
                systems=[base] if base else [],
                rationale=(
                    f"Signal-based: project has {len(code_systems)} code system(s) + coding signal "
                    f"(compose {resolved})"
                ),
                layer=3,
                wants_code_fusion=True,
                code_system_names=resolved,
            )

    # ── Default: base-only (ns-memory) ──

    base = get_system("ns-memory")
    return RouteDecision(
        systems=[base] if base else [],
        rationale="Default: base system only (no explicit override, no project config, no coding signal)",
        layer=3,
    )
