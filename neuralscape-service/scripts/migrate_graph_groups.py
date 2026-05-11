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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--owner",
        required=True,
        help="The user_id every legacy group_id should be rewritten to belong to.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite group_ids. Without this flag, only previews the plan.",
    )
    args = parser.parse_args(argv)

    from config import settings  # noqa: E402
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
