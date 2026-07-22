"""Multi-user group-id construction, permission gates, and live-edge filters.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

from datetime import datetime
from config import settings
from schemas import MemoryVisibility, normalize_visibility

def _live_edges_filter():
    """Graphiti ``SearchFilters`` selecting only bi-temporally LIVE edges.

    Excludes any edge with a non-null ``invalid_at`` or ``expired_at`` —
    i.e. facts the dreaming sweep (or Graphiti's own contradiction
    handling) has invalidated/superseded (audit 27 #3). Built per call:
    SearchFilters is a mutable pydantic model and a shared singleton
    could be mutated by a concurrent caller.
    """
    from graphiti_core.search.search_filters import (
        ComparisonOperator,
        DateFilter,
        SearchFilters,
    )

    return SearchFilters(
        invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
        expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
    )


def _edge_is_invalidated(edge) -> bool:
    """True when a graph edge carries a bi-temporal invalidation stamp.

    Post-filter companion to ``_live_edges_filter`` for result paths where
    the Cypher-level filter may not have applied. Only real timestamp
    values (datetime or non-empty string) count — mock/partial edge
    objects without the attribute are treated as live.
    """
    for attr in ("invalid_at", "expired_at"):
        value = getattr(edge, attr, None)
        if isinstance(value, datetime) or (isinstance(value, str) and value.strip()):
            return True
    return False


def _build_group_id(
    visibility: str,
    user_id: str,
    project_id: str | None = None,
    workspace: str | None = None,
) -> str:
    """Build a Graphiti group_id for the multi-user model + workspace partition.

    | visibility | project_id | workspace    | group_id                                      |
    |------------|------------|--------------|-----------------------------------------------|
    | private    | None       | memory/None  | user--{user_id}                               |
    | private    | set        | memory/None  | user--{user_id}--project--{pid}               |
    | shared     | None       | memory/None  | shared                                        |
    | shared     | set        | memory/None  | shared--project--{project_id}                 |
    | *          | *          | <ref>        | <above>--ws--{workspace}                      |

    The user namespace fixes the prior cross-user leak in Graphiti
    (previously all users shared `"global"` / `"project--..."`). The
    `shared` namespace is the team-wide knowledge pool readable by any
    authenticated user.

    Workspace partition (WT6): absent or "memory" ⇒ NO suffix (byte-identical
    group_ids for existing rows, zero migration). Any other value appends
    `--ws--{workspace}`, isolating reference content from memory pools.
    """
    # Tolerate enum / str / legacy "MemoryVisibility.X" / None — see
    # normalize_visibility docstring for why this matters. Unrecognized
    # values fall back to PRIVATE so an unknown visibility never
    # accidentally lands a memory in the shared pool (safe default
    # preserves the pre-fix behavior of ``str(visibility or PRIVATE)``).
    try:
        vis = normalize_visibility(visibility) or MemoryVisibility.PRIVATE.value
    except (ValueError, TypeError):
        vis = MemoryVisibility.PRIVATE.value
    if vis == MemoryVisibility.STANDARD.value:
        # Authoritative org-wide pool: dictator-written, everyone-readable.
        base = f"standard--project--{project_id}" if project_id else "standard"
        if workspace and workspace != "memory":
            return f"{base}--ws--{workspace}"
        return base
    if vis == MemoryVisibility.SHARED.value:
        base = f"shared--project--{project_id}" if project_id else "shared"
        if workspace and workspace != "memory":
            return f"{base}--ws--{workspace}"
        return base
    # private
    base = f"user--{user_id}--project--{project_id}" if project_id else f"user--{user_id}"
    # Workspace suffix: absent or "memory" ⇒ no suffix (backward compatible)
    if workspace and workspace != "memory":
        return f"{base}--ws--{workspace}"
    return base


def _get_group_ids(caller_user_id: str, project_id: str | None = None) -> list[str]:
    """Group ids the caller is permitted to read across.

    Returns the caller's private namespace + the shared pool, plus the
    project-scoped equivalents when `project_id` is given. A read
    against this set returns the union of the caller's private memories
    and all shared memories (no cross-user private leakage).

    When the `standard` tier is enabled, the authoritative pool is appended
    for EVERY caller (including anonymous/legacy-key readers) so binding org
    standards are always visible.
    """
    def _standard_groups() -> list[str]:
        if not settings.standards_enabled:
            return []
        return ["standard"] + ([f"standard--project--{project_id}"] if project_id else [])

    if not caller_user_id:
        # Anonymous / unauthenticated readers see the shared + standard pools.
        anon = ["shared"] + ([f"shared--project--{project_id}"] if project_id else [])
        return anon + _standard_groups()
    group_ids = [f"user--{caller_user_id}", "shared"]
    if project_id:
        group_ids.append(f"user--{caller_user_id}--project--{project_id}")
        group_ids.append(f"shared--project--{project_id}")
    return group_ids + _standard_groups()


def _check_edit_permission(
    meta: dict,
    payload_user_id: str,
    caller_user_id: str | None,
    *,
    edits_content: bool,
    edits_visibility: bool,
) -> None:
    """Gate an edit against the memory's visibility tier and ownership.

    The split model (locked with the team):
    - dictators may edit anything;
    - ``standard`` tier is dictator-only (mirrors the delete gate);
    - ``shared`` memories: organizational metadata (tags/category/project/v2
      fields) is team-editable housekeeping, but *content* and *visibility*
      changes rewrite or re-tier someone's words — owner only;
    - ``private`` (and legacy null-visibility) memories: owner only.

    Raises PermissionError; returns None when the edit is allowed.
    """
    if settings.is_dictator(caller_user_id):
        return
    owner = meta.get("owner_user_id") or payload_user_id or ""
    try:
        vis = normalize_visibility(meta.get("visibility")) or MemoryVisibility.PRIVATE.value
    except (ValueError, TypeError):
        vis = MemoryVisibility.PRIVATE.value
    if vis == MemoryVisibility.STANDARD.value:
        raise PermissionError("Only a dictator may edit 'standard'-tier memories.")
    if vis == MemoryVisibility.SHARED.value:
        if (edits_content or edits_visibility) and caller_user_id != owner:
            raise PermissionError(
                "Only the memory's owner may edit its content or visibility "
                f"(owner: {owner!r}). Metadata edits (tags/category/project) are open to the team."
            )
        return
    # private / legacy null visibility
    if caller_user_id != owner:
        raise PermissionError(f"Only the memory's owner may edit it (owner: {owner!r}).")
