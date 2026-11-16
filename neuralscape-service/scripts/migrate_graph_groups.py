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
id WOULD CHANGE when ``GRAPH_GROUP_ID_INJECTIVE_ENABLED`` is turned on. The old builder joined
user / project / workspace with ``--``, so ids that themselves contain ``--``
(or a ``.`` / other character outside ``[A-Za-z0-9_-]``, which Graphiti rejects) aliased
other tuples' namespaces; those tuples are now named ``nsg1--<hex>``. Every other
memory keeps its group id byte for byte, so only the listed rows need attention:
their graph facts under the old (ambiguous) group should be re-enriched into the
new group (edit the memory, or re-ingest it). The audit never writes anything.

Rewrites node group ids (Episodic, Entity, Community) AND edge group ids
(``RELATES_TO`` / ``MENTIONS`` / ``HAS_MEMBER`` relationships), and names each
target with the service's own group-id builder. Run it once, on a deployment that
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
NODE_LABELS = ("Episodic", "Entity", "Community")
REL_TYPES = ("RELATES_TO", "MENTIONS", "HAS_MEMBER")


def rewrite_target(group_id: str, owner: str) -> str | None:
    """The current group id for a legacy one, or None when it is not legacy.

    Uses the service's own builder (``memory.groups._build_group_id``) so the
    target can never drift from what the service reads and writes.
    """
    from memory.groups import _build_group_id  # noqa: E402

    if group_id == "global":
        return _build_group_id("private", owner, None, None)
    if group_id.startswith("project--") and len(group_id) > len("project--"):
        return _build_group_id("private", owner, group_id[len("project--"):], None)
    return None


def scan_legacy_groups(session, owner: str) -> dict[str, list[tuple[str, str, int]]]:
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
            new = rewrite_target(record["g"], owner)
            if new is not None:
                rows.append((record["g"], new, record["c"]))
        plan[kind] = sorted(rows)
    return plan


def apply_rewrites(session, plan: dict[str, list[tuple[str, str, int]]], print_fn=print) -> int:
    """Rewrite every planned legacy group id; returns the number of rows changed."""
    updated = 0
    for kind, per in plan.items():
        done = 0
        for old, new, _ in per:
            if kind in NODE_LABELS:
                cypher = f"MATCH (x:{kind} {{group_id: $old}}) SET x.group_id = $new RETURN count(x) AS c"
            else:
                cypher = f"MATCH ()-[x:{kind} {{group_id: $old}}]->() SET x.group_id = $new RETURN count(x) AS c"
            done += session.run(cypher, old=old, new=new).single()["c"]
        updated += done
        print_fn(f"  {kind:<11}  rewrote {done} rows")
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

    new_owner = args.owner
    with driver.session(database=settings.neo4j_database) as session:
        plan = scan_legacy_groups(session, new_owner)

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

    print(f"\nRewriting {total_overall} rows...")
    with driver.session(database=settings.neo4j_database) as session:
        updated_total = apply_rewrites(session, plan, print_fn=print)
    driver.close()
    print(f"Done: {updated_total} rows rewritten.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
