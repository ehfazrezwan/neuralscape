"""AR1 — per-op engine-preference map (config-driven, health-gated).

Ehfaz's 2026-07-10 decision: NS should **automatically choose the best engine per
operation**. This module is the resolver for that choice — given a code op-class,
it returns the ordered list of engines to try, best→worst, and picks the first
that is registered+enabled, declares the capability, and is healthy.

**This is CONFIG, not hardcoded branches** (PLAN §6: "per-op engine preference is
config, not code"). The preference map is a plain dict, layered:

    project config `op_preference`  (per-project override; extends PLAN §4 layer 2)
      → settings `code_op_preference`  (deployment override, env-driven)
        → DEFAULT_CODE_OP_PREFERENCE  (the measured winners below)

The default map is **seeded from the MEASURED per-op winners** in
`ICE_V2_ENGINE_COMPARISON.md` (through-NS, small-py, post-residuals) — accuracy is
the PRIMARY key, latency the tiebreak — NOT invented:

    query (symbol_lookup)  native 0.983 > cbm 0.517 > graphify 0.00 (traversal)
    neighbors              graphify 0.68 R @ 3.5 ms > cbm 0.40 R > native ~0
    path                   graphify 1.00 > native 0 (cbm: no path capability)
    locate (nl_locate)     native 0.167 > cbm 0.00 (graphify: no locate)
    impact (blast_radius)  graphify @ 9.1 ms > native @ 22.7 ms (cbm: git-diff N/A)

Op-classes are the internal names (`query|neighbors|path|locate|impact`) the
router/`CodeKnowledgeSystem` operate on. Bench op names (`symbol_lookup`, …) are
aliased for robustness.

**NEVER branches on `transport`** (DECISIONS.md cross-cutting rule) — resolution
is over registry entries (name, declared capabilities, health), transport-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── The measured-winner default preference map (best → worst per op) ──
#
# Ordered by accuracy (primary), latency (tiebreak). Each list is intentionally
# the full ordered fallback chain, INCLUDING engines that would degrade rather
# than error (e.g. native `path` = 0 by design) — the capability + health gate
# filters non-declaring/unhealthy engines, and the caller falls through on a
# bind miss, so a last-resort degrade beats an honest-but-total N/A only when
# every better engine is unavailable.
DEFAULT_CODE_OP_PREFERENCE: dict[str, list[str]] = {
    "query": ["code-native", "code-cbm", "code-graphify-lib"],
    "neighbors": ["code-graphify-lib", "code-cbm", "code-native"],
    "path": ["code-graphify-lib", "code-native"],
    "locate": ["code-native", "code-cbm"],
    "impact": ["code-graphify-lib", "code-native"],
}

# Bench op names → internal op-class, so resolve_op_engine tolerates either.
_OP_ALIAS: dict[str, str] = {
    "symbol_lookup": "query",
    "neighbors_1hop": "neighbors",
    "path_le4": "path",
    "blast_radius": "impact",
    "nl_locate": "locate",
}


def normalize_op(op: str | None) -> str:
    """Map a bench op name to its internal op-class; pass internal names through."""
    if not op:
        return "query"
    return _OP_ALIAS.get(op, op)


def preference_for_op(
    op: str,
    *,
    project_id: str | None = None,
    settings=None,
) -> list[str]:
    """Resolve the ordered engine-preference list for one op-class.

    Layering (first source that DEFINES the op wins — replace, not merge):
      1. project config ``op_preference[op]`` (per-project override).
      2. settings ``code_op_preference[op]`` (deployment/env override).
      3. ``DEFAULT_CODE_OP_PREFERENCE[op]`` (the measured winners).

    Returns [] for an unknown op (caller degrades to honest N/A).
    """
    op = normalize_op(op)

    # Layer 1: per-project override.
    if project_id:
        try:
            from knowledge.router import get_project_config

            cfg = get_project_config(project_id)
            proj_map = getattr(cfg, "op_preference", None) if cfg else None
            if proj_map and op in proj_map and proj_map[op]:
                return list(proj_map[op])
        except Exception:  # noqa: BLE001 — config lookup is best-effort
            logger.debug("preference_for_op: project config lookup failed", exc_info=True)

    # Layer 2: settings override.
    if settings is None:
        try:
            from config import settings as _settings

            settings = _settings
        except Exception:  # noqa: BLE001
            settings = None
    if settings is not None:
        set_map = getattr(settings, "code_op_preference", None)
        if set_map and op in set_map and set_map[op]:
            return list(set_map[op])

    # Layer 3: measured-winner default.
    return list(DEFAULT_CODE_OP_PREFERENCE.get(op, []))


@dataclass
class AutoResolution:
    """The auto-router's per-op decision (for dispatch + AR3 attribution + tests)."""

    op: str
    best: str | None  # first registered+capable+healthy engine name, or None
    candidates: list[str] = field(default_factory=list)  # full filtered fallback chain
    reason: str = ""


def resolve_op_engine(
    op: str,
    *,
    project_id: str | None = None,
    settings=None,
) -> AutoResolution:
    """Resolve the best HEALTHY, CAPABLE, REGISTERED engine for an op-class.

    For each engine in the preference list, keep it only when it is (a)
    registered+enabled (in the registry), (b) declares the op capability
    (honest N/A), and (c) reports registry health ``ok`` (the cheap, bounded
    probe — CBM's is cached ~10 s; in-process placeholders answer importability).
    ``best`` is the first survivor; ``candidates`` is the full ordered survivor
    list so the caller can fall through on a bind miss. Both empty → no engine
    can serve the op → the caller returns honest N/A (never fabricates).

    A dict lookup + a handful of cached health probes — well within the <1 ms
    routing budget (PLAN §4 / AR2).
    """
    from knowledge.registry import get_system

    op = normalize_op(op)
    prefs = preference_for_op(op, project_id=project_id, settings=settings)
    candidates: list[str] = []
    skipped: list[str] = []

    for name in prefs:
        sys = get_system(name)
        if sys is None:
            skipped.append(f"{name}(not-registered)")
            continue
        if op not in sys.info.capabilities:
            skipped.append(f"{name}(no-cap)")
            continue
        try:
            if sys.health().status != "ok":
                skipped.append(f"{name}(unhealthy)")
                continue
        except Exception:  # noqa: BLE001 — a throwing probe is ineligible, never fatal
            logger.debug("resolve_op_engine: health probe raised for %s", name, exc_info=True)
            skipped.append(f"{name}(health-error)")
            continue
        candidates.append(name)

    best = candidates[0] if candidates else None
    if best:
        reason = f"auto: {op} → {best}" + (
            f" (fallbacks: {candidates[1:]})" if len(candidates) > 1 else ""
        )
    else:
        reason = f"auto: no healthy capable engine for {op} (skipped: {skipped})"
    return AutoResolution(op=op, best=best, candidates=candidates, reason=reason)
