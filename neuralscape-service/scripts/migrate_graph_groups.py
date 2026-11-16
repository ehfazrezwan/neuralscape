#!/usr/bin/env python
"""Rewrite legacy Graphiti `group_id` values to the user-namespaced format.

The pre-multi-user model wrote graph entries under `group_id="global"`
or `group_id="project--{id}"`. The new format namespaces by user so
the graph search can't leak across users:

    Legacy              →   New (private, single-owner installs)
    ------------------------------------------------------------------
    global              →   user--{owner_user_id}
    project--{pid}      →   user--{owner_user_id}--project--{pid}

This script rewrites ALL legacy group_ids to belong to the supplied
``--owner`` (typically the single existing user on a fresh multi-user
deployment). If you want some legacy facts to land in the shared pool
instead, do that AFTER this migration by running
``bulk_promote_visibility.py`` against the corresponding Qdrant rows
and then re-running ``store_raw`` (which will write the matching
``shared``/``shared--project--...`` graph entries on next ingestion).

Usage:
    python scripts/migrate_graph_groups.py --owner ehfaz --target-format legacy            # dry-run (default)
    python scripts/migrate_graph_groups.py --owner ehfaz --target-format legacy --apply
    python scripts/migrate_graph_groups.py --audit-delimiter-ids            # read-only

``--audit-delimiter-ids`` (no Neo4j needed) lists the memories whose graph group
id WOULD CHANGE when ``GRAPH_GROUP_ID_INJECTIVE_ENABLED`` is turned on. The old builder joined
user / project / workspace with ``--``, so ids that themselves contain ``--``
(or a ``.`` / other character outside ``[A-Za-z0-9_-]``, which Graphiti rejects) aliased
other tuples' namespaces; those tuples are now named ``nsg1--<hex>``. Every other
memory keeps its group id byte for byte, so only the listed rows need attention:
their graph facts under the old (ambiguous) group should be re-enriched into the
new group (edit the memory, or re-ingest it). The audit never writes anything.

Safe to re-run: the rewrite is idempotent (a second run finds no legacy group and changes
nothing). If the owner's target group (for example ``user--<owner>``) already holds data,
the legacy rows are MERGED into it, not replaced. Each legacy group is renamed in ONE
transaction, so a very large group can exhaust memory or hit the transaction timeout;
batching is not implemented (a stated limit), so run it where such a group fits.

Rewrites node group ids (Episodic, Entity, Community, Saga) AND edge group ids
(``RELATES_TO`` / ``MENTIONS`` / ``HAS_MEMBER`` / ``HAS_EPISODE`` / ``NEXT_EPISODE``
relationships), naming each target with the service's own group-id builder in the
format you choose with ``--target-format`` (it refuses to run when that disagrees with
the service's current setting, unless ``--force``). Every legacy group becomes PRIVATE
to ``--owner``. Each legacy group is renamed in one transaction and verified afterwards. Run it once, on a deployment that
predates the multi-user model; on a current install it finds nothing and exits.

The script writes to Neo4j via direct Cypher (Graphiti's own driver),
so the backend service does NOT need to be running.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_DIR = _HERE.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))


def _legacy_group_id(visibility: str, owner: str, project: str | None, workspace: str | None) -> str:
    """The pre-fix (delimiter-joined, non-injective) group id, for the audit only."""
    if visibility == "standard":
        base = f"standard--project--{project}" if project else "standard"
    elif visibility == "shared":
        base = f"shared--project--{project}" if project else "shared"
    else:
        base = f"user--{owner}--project--{project}" if project else f"user--{owner}"
    return f"{base}--ws--{workspace}" if workspace and workspace != "memory" else base


def changed_group_rows(points) -> list[dict]:
    """Rows (Qdrant points) whose graph group id differs from the legacy name.

    ``points`` is any iterable of objects with ``.id`` and ``.payload`` (a Qdrant
    scroll result). Pure function: no I/O, so it is unit-testable with fakes.
    """
    from config import settings  # noqa: E402
    from memory.groups import _build_group_id  # noqa: E402

    rows: list[dict] = []
    for pt in points:
        payload = pt.payload or {}
        meta = payload.get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        visibility = meta.get("visibility") or "private"
        owner = meta.get("owner_user_id") or payload.get("user_id") or ""
        project = meta.get("project_id") or None
        workspace = meta.get("workspace") or None
        old = _legacy_group_id(visibility, owner, project, workspace)
        # What GRAPH_GROUP_ID_INJECTIVE_ENABLED=true would name it: the real
        # builder with the flag forced on (no parallel encoding rule here).
        flag_was = settings.graph_group_id_injective_enabled
        settings.graph_group_id_injective_enabled = True
        try:
            new = _build_group_id(visibility, owner, project, workspace)
        finally:
            settings.graph_group_id_injective_enabled = flag_was
        if old != new:
            rows.append({"id": str(pt.id), "owner": owner, "project_id": project,
                         "workspace": workspace, "visibility": visibility,
                         "old_group_id": old, "new_group_id": new})
    return rows


def audit_delimiter_ids(client, collection: str) -> list[dict]:
    """Scroll the whole collection and return ``changed_group_rows`` over it."""
    out: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=500, offset=offset,
            with_payload=True, with_vectors=False,
        )
        out.extend(changed_group_rows(points))
        if offset is None:
            return out


# Graphiti stores group_id on these node labels and on these relationship types.
# (Entity edges are ``RELATES_TO`` *relationships*; there is no ``EntityEdge`` node
# label, so a node-only rewrite silently leaves every fact's edge in the legacy group.)
NODE_LABELS = ("Episodic", "Entity", "Community", "Saga")
REL_TYPES = ("RELATES_TO", "MENTIONS", "HAS_MEMBER", "HAS_EPISODE", "NEXT_EPISODE")
TARGET_FORMATS = ("legacy", "injective")


def current_format(settings) -> str:
    """The group-id format the running service uses (``GRAPH_GROUP_ID_INJECTIVE_ENABLED``)."""
    return "injective" if settings.graph_group_id_injective_enabled else "legacy"


def rewrite_target(group_id: str, owner: str, fmt: str = "legacy") -> str | None:
    """The group id (in format ``fmt``) for a legacy one, or None when it is not legacy.

    ``fmt`` is explicit: the result never depends on the process's flag state. The service's
    own builder (``memory.groups._build_group_id``) names the target with the flag forced to
    ``fmt``, so the name cannot drift from what the service reads and writes in that format.
    """
    from config import settings  # noqa: E402
    from memory.groups import _build_group_id  # noqa: E402

    if fmt not in TARGET_FORMATS:
        raise ValueError(f"unknown target format {fmt!r}")
    if group_id == "global":
        args = ("private", owner, None, None)
    elif group_id.startswith("project--") and len(group_id) > len("project--"):
        args = ("private", owner, group_id[len("project--"):], None)
    else:
        return None
    was = settings.graph_group_id_injective_enabled
    settings.graph_group_id_injective_enabled = fmt == "injective"
    try:
        return _build_group_id(*args)
    finally:
        settings.graph_group_id_injective_enabled = was


def scan_legacy_groups(session, owner: str, fmt: str = "legacy") -> dict[str, list[tuple[str, str, int]]]:
    """Per node label / relationship type: ``[(legacy_group_id, new_group_id, row_count)]``."""
    where = "x.group_id = 'global' OR x.group_id STARTS WITH 'project--'"
    plan: dict[str, list[tuple[str, str, int]]] = {}
    queries = [(f"{label}", f"MATCH (x:{label}) WHERE {where} RETURN x.group_id AS g, count(x) AS c") for label in NODE_LABELS]
    queries += [
        (rel, f"MATCH ()-[x:{rel}]->() WHERE {where} RETURN x.group_id AS g, count(x) AS c")
        for rel in REL_TYPES
    ]
    for kind, cypher in queries:
        rows = []
        for record in session.run(cypher):
            new = rewrite_target(record["g"], owner, fmt)
            if new is not None:
                rows.append((record["g"], new, record["c"]))
        plan[kind] = sorted(rows)
    return plan


def _remaining(session, old: str) -> int:
    # Deliberately label-less: this is the verification step, so it counts ANY node or
    # relationship still carrying the old group id, including labels this script does not list.
    nodes = session.run("MATCH (x) WHERE x.group_id = $old RETURN count(x) AS c", old=old).single()["c"]
    rels = session.run("MATCH ()-[x]->() WHERE x.group_id = $old RETURN count(x) AS c", old=old).single()["c"]
    return nodes + rels


def apply_rewrites(session, plan: dict[str, list[tuple[str, str, int]]], print_fn=print) -> int:
    """Rewrite every planned legacy group id; returns the number of rows changed.

    One transaction per legacy group id covers every label and relationship type, so an
    interrupted run never leaves one group's facts split between old and new names; after
    the commit the old name must be gone, or the run stops with an error.
    """
    by_old: dict[str, str] = {}
    for per in plan.values():
        for old, new, _ in per:
            by_old[old] = new
    per_kind = {kind: 0 for kind in plan}
    updated = 0
    for old, new in sorted(by_old.items()):
        with session.begin_transaction() as tx:
            for kind, per in plan.items():
                if not any(o == old for o, _, _ in per):
                    continue
                if kind in NODE_LABELS:
                    cypher = f"MATCH (x:{kind} {{group_id: $old}}) SET x.group_id = $new RETURN count(x) AS c"
                else:
                    cypher = f"MATCH ()-[x:{kind} {{group_id: $old}}]->() SET x.group_id = $new RETURN count(x) AS c"
                n = tx.run(cypher, old=old, new=new).single()["c"]
                per_kind[kind] += n
                updated += n
            tx.commit()
        left = _remaining(session, old)
        if left:
            raise RuntimeError(f"verification failed: {left} rows still carry group_id {old!r} after the rewrite")
    for kind, n in per_kind.items():
        print_fn(f"  {kind:<11}  rewrote {n} rows")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--audit-delimiter-ids",
        action="store_true",
        help="Read-only: list memories whose graph group id changed with the injective "
             "group-id builder (ids containing '--', '.' or other characters outside [A-Za-z0-9_-]).",
    )
    parser.add_argument(
        "--owner",
        help="The user_id every legacy group_id should be rewritten to belong to.",
    )
    parser.add_argument(
        "--target-format",
        choices=TARGET_FORMATS,
        help="The group-id format to rewrite INTO ('legacy' = delimiter-joined names, 'injective' = "
             "nsg1-- names). Required, and it must match the service's GRAPH_GROUP_ID_INJECTIVE_ENABLED "
             "setting (use --force to override); a wrong choice renames every group to a name the service "
             "does not read.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even though --target-format disagrees with the service's current setting.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite group_ids. Without this flag, only previews the plan.",
    )
    args = parser.parse_args(argv)

    from config import settings  # noqa: E402

    if args.audit_delimiter_ids:
        from qdrant_client import QdrantClient  # noqa: E402

        client = QdrantClient(url=settings.qdrant_url) if settings.qdrant_url else QdrantClient(
            path=str(Path(settings.qdrant_path).expanduser())
        )
        rows = audit_delimiter_ids(client, settings.qdrant_collection)
        print(f"{len(rows)} memories have a changed graph group id.")
        for r in rows:
            print(f"  {r['id']}  {r['old_group_id']!r} -> {r['new_group_id']}")
        if rows:
            print("Re-enrich these memories so their graph facts live under the new group.")
        return 0
    if not args.owner:
        parser.error("--owner is required (unless --audit-delimiter-ids)")
    if not args.target_format:
        parser.error("--target-format legacy|injective is required (unless --audit-delimiter-ids)")
    live = current_format(settings)
    print(f"Target format: {args.target_format} (service is configured for: {live})")
    if args.target_format != live and not args.force:
        print(
            f"error: --target-format {args.target_format} disagrees with the service's current "
            f"GRAPH_GROUP_ID_INJECTIVE_ENABLED setting ({live}); refusing. Use --force only if you are "
            "migrating ahead of a deliberate flag change.",
            file=sys.stderr,
        )
        return 2

    from neo4j import GraphDatabase  # noqa: E402

    if not settings.neo4j_password:
        print("error: NEO4J_PASSWORD is not set.", file=sys.stderr)
        return 2

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    new_owner = args.owner
    with driver.session(database=settings.neo4j_database) as session:
        plan = scan_legacy_groups(session, new_owner, args.target_format)

    total_overall = sum(n for per in plan.values() for _, _, n in per)
    print(f"Legacy group_id audit (owner = {new_owner!r}):")
    for kind, per in plan.items():
        print(f"  {kind:<11}  {sum(n for _, _, n in per):>5} rows in {len(per)} legacy group(s)")
        for old, new, n in per:
            print(f"      {old!r} -> {new!r}: {n}")
    print(f"  {'TOTAL':<11}  {total_overall} rows to rewrite")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to actually rewrite.")
        driver.close()
        return 0

    print(
        f"\nWARNING: every legacy 'global' and 'project--*' group becomes PRIVATE to {new_owner!r}. "
        "Nothing is shared by this script."
    )
    print(f"Rewriting {total_overall} rows (one transaction per legacy group, verified afterwards)...")
    with driver.session(database=settings.neo4j_database) as session:
        updated_total = apply_rewrites(session, plan, print_fn=print)
    driver.close()
    print(f"Done: {updated_total} rows rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
