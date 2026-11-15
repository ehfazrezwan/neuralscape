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
    python scripts/migrate_graph_groups.py --owner ehfaz                    # dry-run
    python scripts/migrate_graph_groups.py --owner ehfaz --apply
    python scripts/migrate_graph_groups.py --audit-delimiter-ids            # read-only

``--audit-delimiter-ids`` (no Neo4j needed) lists the memories whose graph group
id CHANGED when ``_build_group_id`` became injective. The old builder joined
user / project / workspace with ``--``, so ids that themselves contain ``--``
(or start/end with ``-``, or use characters outside ``[A-Za-z0-9_.-]``) aliased
other tuples' namespaces; those tuples are now named ``nsg1--<hex>``. Every other
memory keeps its group id byte for byte, so only the listed rows need attention:
their graph facts under the old (ambiguous) group should be re-enriched into the
new group (edit the memory, or re-ingest it). The audit never writes anything.

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
        new = _build_group_id(visibility, owner, project, workspace)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--audit-delimiter-ids",
        action="store_true",
        help="Read-only: list memories whose graph group id changed with the injective "
             "group-id builder (ids containing '--', edge '-', or odd characters).",
    )
    parser.add_argument(
        "--owner",
        help="The user_id every legacy group_id should be rewritten to belong to.",
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

    from neo4j import GraphDatabase  # noqa: E402

    if not settings.neo4j_password:
        print("error: NEO4J_PASSWORD is not set.", file=sys.stderr)
        return 2

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    # Graphiti puts group_id on Episodic, Entity, EntityEdge, and Community nodes.
    # We rewrite all of them. The Cypher uses a CASE expression so each row
    # gets the right new value based on whether its old group_id had the
    # `project--` prefix or was the bare `"global"` namespace.
    targets = ["Episodic", "Entity", "EntityEdge", "Community"]
    new_owner = args.owner

    plan: dict[str, dict[str, int]] = {}
    with driver.session(database=settings.neo4j_database) as session:
        for label in targets:
            count_legacy_global = session.run(
                f"MATCH (n:{label}) WHERE n.group_id = 'global' RETURN count(n) AS c"
            ).single()["c"]
            count_legacy_project = session.run(
                f"MATCH (n:{label}) WHERE n.group_id STARTS WITH 'project--' "
                f"RETURN count(n) AS c"
            ).single()["c"]
            plan[label] = {
                "global": count_legacy_global,
                "project": count_legacy_project,
                "total": count_legacy_global + count_legacy_project,
            }

    total_overall = sum(p["total"] for p in plan.values())
    print(f"Legacy group_id audit (owner = {new_owner!r}):")
    for label, p in plan.items():
        print(
            f"  {label:<11}  global → user--{new_owner}: {p['global']:>5}"
            f"   |  project--<x> → user--{new_owner}--project--<x>: {p['project']:>5}"
        )
    print(f"  {'TOTAL':<11}  {total_overall} rows to rewrite")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to actually rewrite.")
        driver.close()
        return 0

    print(f"\nRewriting {total_overall} rows...")
    updated_total = 0
    with driver.session(database=settings.neo4j_database) as session:
        for label in targets:
            # global -> user--{owner}
            res = session.run(
                f"MATCH (n:{label}) WHERE n.group_id = 'global' "
                f"SET n.group_id = $new_id RETURN count(n) AS c",
                new_id=f"user--{new_owner}",
            )
            n1 = res.single()["c"]
            # project--xxx -> user--{owner}--project--xxx
            res = session.run(
                f"MATCH (n:{label}) WHERE n.group_id STARTS WITH 'project--' "
                f"SET n.group_id = 'user--' + $owner + '--' + n.group_id "
                f"RETURN count(n) AS c",
                owner=new_owner,
            )
            n2 = res.single()["c"]
            updated_total += n1 + n2
            print(f"  {label:<11}  rewrote {n1 + n2} rows")
    driver.close()
    print(f"Done: {updated_total} rows rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
