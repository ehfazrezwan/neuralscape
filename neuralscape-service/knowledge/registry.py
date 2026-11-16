"""Knowledge system registry — discovery, registration, eligibility.

Mirrors the ADAPTER_REGISTRY idiom (``adapters/base.py``): module-level
instantiation, dict-based registry, register/get/list accessors. Config gates
which systems exist (CODE_NATIVE_ENABLED for code-native, always-on for base).
"""

from __future__ import annotations

import logging

from knowledge.base import KnowledgeSystem

logger = logging.getLogger(__name__)


KNOWLEDGE_REGISTRY: dict[str, KnowledgeSystem] = {}
"""The global registry of knowledge systems, keyed by name.

Populated at import via ``register_system`` calls from the system modules
(``ns_memory.py``, ``code_system.py``). Config gates which entries exist:
code-native requires CODE_NATIVE_ENABLED=true; base (NS memory) always registers.
"""


def register_system(system: KnowledgeSystem) -> None:
    """Register (or replace) a knowledge system under its ``info.name``.

    Idempotent — re-registering the same name overwrites, so a module that
    defines a system can be imported more than once without error. This mirrors
    the ``register_adapter`` idiom.

    Args:
        system: The KnowledgeSystem implementation to register.
    """
    name = system.info.name
    if name in KNOWLEDGE_REGISTRY:
        logger.debug("Re-registering knowledge system %s (overwrites previous)", name)
    KNOWLEDGE_REGISTRY[name] = system
    logger.info(
        "Registered knowledge system: %s (kind=%s, transport=%s, capabilities=%s)",
        name,
        system.info.kind,
        system.info.transport,
        sorted(system.info.capabilities),
    )


def get_system(name: str) -> KnowledgeSystem | None:
    """Resolve a knowledge system by name, returning None if not registered.

    Use this for explicit system selection (e.g. ``knowledge_system`` param
    on recall_memories). Returns None for unknown names so the caller can
    decide whether to fallback to base or raise.

    Args:
        name: The system's registry key (e.g. "ns-memory", "code-cbm").

    Returns:
        The KnowledgeSystem instance, or None if not registered.
    """
    return KNOWLEDGE_REGISTRY.get(name)


def list_systems() -> list[str]:
    """Return the registered system names, sorted.

    Useful for introspection, health reporting, and routing debug logs.
    """
    return sorted(KNOWLEDGE_REGISTRY.keys())


def code_space_matches_project(code_space: str | None, project_id: str) -> tuple[bool, str]:
    """Does ``code_space`` (``code--owner--repo``) belong to ``project_id``'s repo?

    The project's repo is the ``code_space`` recorded in its knowledge config at
    index time. A match is the same code_space, or two code_spaces that
    ``CODE_REPOS`` maps to the same directory. A project with no recorded
    code_space has no repo to compare against, so it is not constrained. Never
    raises: any lookup failure leaves the system eligible (the older behavior).
    Returns ``(ok, reason)``; ``reason`` explains a refusal.
    """
    try:
        from knowledge.router import get_project_config

        cfg = get_project_config(project_id)
    except Exception:  # noqa: BLE001 - config store trouble must not break routing
        logger.debug("project config lookup failed for %s", project_id, exc_info=True)
        return True, ""
    project_space = getattr(cfg, "code_space", None) if cfg is not None else None
    if not project_space:
        return True, ""
    if code_space == project_space:
        return True, ""
    try:
        from config import settings
        from knowledge.code_dispatch import repo_path_for_code_space

        engine_path = repo_path_for_code_space(code_space or "", settings)
        project_path = repo_path_for_code_space(project_space, settings)
        if engine_path and engine_path == project_path:
            return True, ""
    except Exception:  # noqa: BLE001
        logger.debug("repo path comparison failed", exc_info=True)
    return False, (
        f"code_space {code_space!r} does not match project repo {project_space!r}"
    )


def eligible_systems(
    project_id: str | None = None,
    operation: str | None = None,
    kind: str | None = None,
) -> list[KnowledgeSystem]:
    """Filter registered systems by eligibility: healthy + declares the capability.

    Eligibility criteria:
      1. System is registered (config-gated at import).
      2. System is healthy (``health().status == "ok"``).
      3. System declares the requested ``operation`` in its capabilities
         (if operation is specified).
      4. System matches the requested ``kind`` (if kind is specified).
      5. HARDENING FIX (M2): For code systems with project_id, system must have
         a REAL indexed code_space for the project (not a capability placeholder).

    **No routing logic** — this is a filtering primitive. The router (Phase D)
    will layer on project config, explicit overrides, and signal-based
    classification.

    Args:
        project_id: Optional project filter. For code systems, filters to only those
                   with indexed code_spaces for this project (not capability placeholders).
        operation: Op-class name (e.g. "recall", "neighbors", "index"). None = no capability filter.
        kind: System kind filter ("base" | "code"). None = no kind filter.

    Returns:
        List of eligible systems, sorted by name for deterministic ordering.
    """
    eligible = []
    for name in sorted(KNOWLEDGE_REGISTRY.keys()):
        system = KNOWLEDGE_REGISTRY[name]

        # Health check (blocks eligibility on failure, but doesn't raise —
        # one unhealthy system shouldn't crash routing).
        try:
            health = system.health()
            if health.status != "ok":
                logger.debug(
                    "System %s ineligible: health=%s",
                    name,
                    health.status,
                )
                continue
        except Exception:
            logger.exception("Health check failed for system %s; marking ineligible", name)
            continue

        # Kind filter
        if kind is not None and system.info.kind != kind:
            continue

        # Capability filter: if operation is specified, system must declare it.
        if operation is not None and operation not in system.info.capabilities:
            logger.debug(
                "System %s ineligible: missing capability %s (has: %s)",
                name,
                operation,
                sorted(system.info.capabilities),
            )
            continue

        # M2 FIX: Project-scope filter for code systems. A code system is eligible
        # for a project ONLY if it has an indexed code_space for that project.
        # A registered global system or capability placeholder (code_space =
        # "__registry_capability__") is NOT an indexed project code_space.
        if project_id is not None and system.info.kind == "code":
            engine = getattr(system, "_engine", None)
            if engine is not None:
                code_space = getattr(engine, "code_space", None)
                # Capability placeholder: not eligible for specific projects
                if code_space == "__registry_capability__":
                    logger.debug(
                        "System %s ineligible for project %s: capability placeholder",
                        name,
                        project_id,
                    )
                    continue
                # Empty/unindexed engine: not eligible
                if hasattr(engine, "G") and engine.G is None:
                    logger.debug(
                        "System %s ineligible for project %s: no indexed graph",
                        name,
                        project_id,
                    )
                    continue
                # A code_space bound to a project must be that project's repo.
                ok, reason = code_space_matches_project(code_space, project_id)
                if not ok:
                    logger.debug(
                        "System %s ineligible for project %s: %s",
                        name,
                        project_id,
                        reason,
                    )
                    continue

        eligible.append(system)

    return eligible
