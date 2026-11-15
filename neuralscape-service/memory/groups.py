"""Multi-user group-id construction, permission gates, and live-edge filters.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import re

from datetime import datetime
from typing import NamedTuple

from config import settings
from schemas import MemoryVisibility, normalize_visibility

logger = logging.getLogger(__name__)

# ── Injective group ids ────────────────────────────────────────────────
# Legacy group ids join their parts with ``--``, but a legal id may itself
# contain ``--`` (the id alphabet is ``[A-Za-z0-9_.-]``), so user
# ``alice--project--payroll`` and user ``alice`` in project ``payroll`` shared
# one graph namespace (a cross-user read/write alias).
#
# A component is *plain* when it is non-empty, uses only Graphiti's group-id
# alphabet ``[A-Za-z0-9_-]`` (so NOT ``.``: Graphiti's validate_group_id rejects it,
# hence no data can exist under a dotted name) and has no ``--``. A plain component can still start or end with a single ``-``: the
# delimiter is then followed/preceded by one more ``-`` and the reading stays
# unique (checked exhaustively in tests/test_group_ids.py). In a tuple of plain
# components the name reads back to exactly one tuple, and plain tuples keep
# their legacy name BYTE FOR BYTE (zero migration for every id that does not hit
# the collision). Any tuple with a non-plain
# component is named ``nsg1--`` + the hex of a length-prefixed encoding of the
# whole tuple: the ``nsg1`` first segment is not a legacy pool name, the
# characters are all Graphiti-valid, and the encoding is canonical, so distinct
# tuples can never share a name. (Idea reused from the old graph-wave branch,
# minus its auth-profile flag: injectivity must not be opt-in.)
_PLAIN_COMPONENT = re.compile(r"(?!.*--)[A-Za-z0-9_\-]+")
ENCODED_GROUP_PREFIX = "nsg1--"
_POOL_KINDS = ("private", "shared", "standard")


class GroupKey(NamedTuple):
    """The placement a group id names; ``owner`` is set for (only) private pools."""

    pool: str
    owner: str | None
    project: str | None
    workspace: str | None


def _is_plain_component(value: str | None) -> bool:
    # ``""`` only reaches here as a private owner (anonymous legacy write); it
    # keeps its historical name ``user--``.
    return value is None or value == "" or _PLAIN_COMPONENT.fullmatch(value) is not None


def _encode_group_key(key: GroupKey) -> str:
    """``nsg1--`` + hex of one segment per field: ``~`` if absent else ``<utf8-length>:<value>``."""
    parts = [
        "~" if value is None else f"{len(value.encode('utf-8'))}:{value}"
        for value in key
    ]
    return ENCODED_GROUP_PREFIX + "".join(parts).encode("utf-8").hex()


def decode_group_id(group_id: str) -> GroupKey | None:
    """Exact placement named by an encoded (``nsg1--``) group id, else ``None``.

    Only canonical encodings of tuples that actually need one decode; a legacy
    name (including every plain tuple's name) returns ``None``.
    """
    if not isinstance(group_id, str) or not group_id.startswith(ENCODED_GROUP_PREFIX):
        return None
    try:
        data = bytes.fromhex(group_id[len(ENCODED_GROUP_PREFIX):])
    except ValueError:
        return None
    values: list[str | None] = []
    position = 0
    for _ in GroupKey._fields:
        if data[position:position + 1] == b"~":
            values.append(None)
            position += 1
            continue
        colon = data.find(b":", position)
        digits = data[position:colon] if colon != -1 else b""
        if not digits.isdigit():
            return None
        end = colon + 1 + int(digits)
        if end > len(data):
            return None
        try:
            values.append(data[colon + 1:end].decode("utf-8"))
        except UnicodeDecodeError:
            return None
        position = end
    key = GroupKey(*values)
    if (
        position != len(data)
        or key.pool not in _POOL_KINDS
        or (key.pool == "private") != (key.owner is not None)
        or _encode_group_key(key) != group_id
        or _key_is_plain(key)
    ):
        return None
    return key


def _key_is_plain(key: GroupKey) -> bool:
    return all(_is_plain_component(v) for v in (key.owner, key.project, key.workspace))

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


def _live_node_uuids(run_on_bridge, driver, uuids: list[str], group_ids: list[str]) -> set[str] | None:
    """Server-side liveness check for a candidate set of entity node uuids.

    A node is LIVE (kept by a listing endpoint's default expiry filter) when
    either (a) it has NO connecting ``RELATES_TO`` edges at all — "not yet
    enriched", not "expired" — or (b) at least one of its connecting edges
    is bi-temporally live (``expired_at IS NULL AND invalid_at IS NULL``).
    Same liveness definition as ``_edge_is_invalidated``, evaluated in one
    bounded Cypher statement instead of fetching every edge in the group
    into Python: the prior approach (``EntityEdge.get_by_group_ids(...,
    limit=5000)``) silently mis-classified nodes once a group held more
    edges than the cap, and failed OPEN (returned nodes unfiltered) on any
    lookup error — for listing endpoints that exist specifically to hide
    soft-expired PRIVATE content, fail-open is a leak, not a convenience.

    Returns the set of live node uuids (a subset of ``uuids``), or ``None``
    on any query failure — callers MUST fail CLOSED (treat ``None`` as "no
    nodes are provably live") rather than falling back to unfiltered.
    """
    if not uuids:
        return set()

    cypher = """
    MATCH (n:Entity) WHERE n.uuid IN $uuids
    OPTIONAL MATCH (n)-[e:RELATES_TO]-()
    WHERE e.group_id IN $group_ids
    WITH n.uuid AS uuid,
         count(e) AS edge_count,
         sum(CASE WHEN e.expired_at IS NULL AND e.invalid_at IS NULL THEN 1 ELSE 0 END) AS live_count
    RETURN uuid, edge_count, live_count
    """

    async def _run():
        async with driver.session() as session:
            result = await session.run(cypher, uuids=uuids, group_ids=group_ids)
            return await result.data()

    coro = _run()
    try:
        records = run_on_bridge(coro) or []
    except Exception:
        coro.close()
        logger.warning(
            "Node liveness query failed for group_ids=%r (failing CLOSED)",
            group_ids, exc_info=True,
        )
        return None
    return {
        rec["uuid"]
        for rec in records
        if not rec.get("edge_count") or rec.get("live_count")
    }


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

    Injective WHEN ``GRAPH_GROUP_ID_INJECTIVE_ENABLED`` is on (default off:
    the names above for every id, byte for byte): a tuple with a non-plain component (one containing ``--``, a ``.`` or
    any character outside ``[A-Za-z0-9_-]``) is named
    ``nsg1--<hex>`` (see ``_encode_group_key``) so it can never alias another
    tuple's namespace; every plain tuple keeps the legacy name shown above.
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
    pool = vis if vis in (MemoryVisibility.STANDARD.value, MemoryVisibility.SHARED.value) else "private"
    key = GroupKey(
        pool,
        str(user_id) if pool == "private" else None,
        str(project_id) if project_id else None,
        str(workspace) if workspace and workspace != "memory" else None,
    )
    if settings.graph_group_id_injective_enabled and not _key_is_plain(key):
        return _encode_group_key(key)
    # Workspace suffix: absent or "memory" ⇒ no suffix (backward compatible)
    if pool == MemoryVisibility.STANDARD.value:
        # Authoritative org-wide pool: dictator-written, everyone-readable.
        base = f"standard--project--{key.project}" if key.project else "standard"
    elif pool == MemoryVisibility.SHARED.value:
        base = f"shared--project--{key.project}" if key.project else "shared"
    else:
        base = f"user--{key.owner}--project--{key.project}" if key.project else f"user--{key.owner}"
    return f"{base}--ws--{key.workspace}" if key.workspace else base


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
        return ["standard"] + ([_build_group_id("standard", "", project_id)] if project_id else [])

    if not caller_user_id:
        # Anonymous / unauthenticated readers see the shared + standard pools.
        anon = ["shared"] + ([_build_group_id("shared", "", project_id)] if project_id else [])
        return anon + _standard_groups()
    group_ids = [_build_group_id("private", caller_user_id), "shared"]
    if project_id:
        group_ids.append(_build_group_id("private", caller_user_id, project_id))
        group_ids.append(_build_group_id("shared", caller_user_id, project_id))
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
