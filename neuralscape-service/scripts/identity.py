#!/usr/bin/env python3
"""Admin CLI for Neuralscape identity reconciliation.

Commands
--------
  list                      Show durable links (Redis) + existing memory owners.
  link  <email> <user_id>   Map an email (and/or --sub) → user_id in the store.
  unlink <email>            Remove an email (and/or --sub) mapping.
  merge <from_id> <to_id>   Re-key all memories owned by <from_id> → <to_id>
                            across Qdrant + Neo4j. DRY-RUN by default; pass
                            --apply to actually mutate data.

Run from neuralscape-service/:
  uv run python scripts/identity.py list
  uv run python scripts/identity.py merge "siliconinjax@gmail.com" ehfazrezwan
  uv run python scripts/identity.py merge "siliconinjax@gmail.com" ehfazrezwan --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import identity_store
from config import settings

# ── Qdrant REST helpers ───────────────────────────────────────────────────


def _qdrant(path: str, body: dict) -> dict:
    url = f"{settings.qdrant_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


_COLLECTIONS = ["neuralscape_memories", "neuralscape_memories_entities"]


def _scroll_owned(collection: str, owner: str) -> list[dict]:
    """All points in a collection whose top-level user_id == owner."""
    points, offset = [], None
    while True:
        body = {
            "filter": {"must": [{"key": "user_id", "match": {"value": owner}}]},
            "limit": 256, "with_payload": True, "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        res = _qdrant(f"/collections/{collection}/points/scroll", body)["result"]
        points.extend(res.get("points", []))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return points


def _qdrant_rekey(collection: str, points: list[dict], to_id: str) -> None:
    """Set user_id + metadata.owner_user_id → to_id for each point."""
    for p in points:
        md = dict(p.get("payload", {}).get("metadata") or {})
        if "owner_user_id" in md:
            md["owner_user_id"] = to_id
        _qdrant(
            f"/collections/{collection}/points/payload",
            {"payload": {"user_id": to_id, "metadata": md}, "points": [p["id"]]},
        )


# ── Neo4j helpers ─────────────────────────────────────────────────────────


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def _neo4j_count(session, old_prefix: str) -> tuple[int, int]:
    nodes = session.run(
        "MATCH (n) WHERE n.group_id STARTS WITH $p RETURN count(n) AS c",
        p=old_prefix,
    ).single()["c"]
    rels = session.run(
        "MATCH ()-[r]->() WHERE r.group_id STARTS WITH $p RETURN count(r) AS c",
        p=old_prefix,
    ).single()["c"]
    return nodes, rels


def _neo4j_rekey(session, old_prefix: str, new_prefix: str) -> None:
    session.run(
        "MATCH (n) WHERE n.group_id STARTS WITH $old "
        "SET n.group_id = $new + substring(n.group_id, size($old))",
        old=old_prefix, new=new_prefix,
    )
    session.run(
        "MATCH ()-[r]->() WHERE r.group_id STARTS WITH $old "
        "SET r.group_id = $new + substring(r.group_id, size($old))",
        old=old_prefix, new=new_prefix,
    )


# ── commands ──────────────────────────────────────────────────────────────


def cmd_list(_args) -> int:
    links = asyncio.run(identity_store.all_links())
    print("== durable links (Redis) ==")
    print("  by sub  :", links["by_sub"] or "(none)")
    print("  by email:", links["by_email"] or "(none)")
    print("\n== existing memory owners (Qdrant user_id) ==")
    owners: dict[str, int] = {}
    for coll in _COLLECTIONS:
        try:
            res = _qdrant(
                f"/collections/{coll}/points/scroll",
                {"limit": 1000, "with_payload": ["user_id"], "with_vector": False},
            )["result"]
        except Exception as e:
            print(f"  ({coll}: {e})")
            continue
        for p in res.get("points", []):
            uid = (p.get("payload") or {}).get("user_id")
            if uid:
                owners[uid] = owners.get(uid, 0) + 1
    for uid, n in sorted(owners.items(), key=lambda x: -x[1]):
        print(f"  {n:5}  {uid}")
    return 0


def cmd_link(args) -> int:
    ok = asyncio.run(identity_store.link(args.user_id, sub=args.sub, email=args.email))
    print("linked" if ok else "FAILED (Redis unavailable?)")
    return 0 if ok else 1


def cmd_unlink(args) -> int:
    ok = asyncio.run(identity_store.unlink(sub=args.sub, email=args.email))
    print("unlinked" if ok else "FAILED")
    return 0 if ok else 1


def cmd_merge(args) -> int:
    from_id, to_id = args.from_id, args.to_id
    old_prefix = f"user--{from_id}"
    new_prefix = f"user--{to_id}"
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== merge {from_id!r} → {to_id!r}  [{mode}] ===\n")

    # Qdrant
    qdrant_points: dict[str, list[dict]] = {}
    for coll in _COLLECTIONS:
        try:
            pts = _scroll_owned(coll, from_id)
        except Exception as e:
            print(f"  Qdrant {coll}: ERROR {e}")
            pts = []
        qdrant_points[coll] = pts
        print(f"  Qdrant {coll}: {len(pts)} point(s) owned by {from_id!r}")

    # Neo4j
    try:
        driver = _neo4j_driver()
        with driver.session(database=settings.neo4j_database) as session:
            nodes, rels = _neo4j_count(session, old_prefix)
        print(f"  Neo4j: {nodes} node(s) + {rels} relationship(s) under "
              f"group_id prefix {old_prefix!r}")
        driver.close()
    except Exception as e:
        print(f"  Neo4j: ERROR {e}")
        nodes = rels = 0

    if not args.apply:
        print("\nDRY-RUN only — nothing changed. Re-run with --apply to execute.")
        return 0

    print("\nApplying…")
    for coll, pts in qdrant_points.items():
        if pts:
            _qdrant_rekey(coll, pts, to_id)
            print(f"  Qdrant {coll}: re-keyed {len(pts)} point(s)")
    if nodes or rels:
        driver = _neo4j_driver()
        with driver.session(database=settings.neo4j_database) as session:
            _neo4j_rekey(session, old_prefix, new_prefix)
        driver.close()
        print(f"  Neo4j: re-keyed group_id {old_prefix!r} → {new_prefix!r}")
    print("\nDone. Verify with:  uv run python scripts/identity.py list")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Neuralscape identity admin")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    p_link = sub.add_parser("link")
    p_link.add_argument("email")
    p_link.add_argument("user_id")
    p_link.add_argument("--sub", default=None)
    p_link.set_defaults(func=cmd_link)

    p_unlink = sub.add_parser("unlink")
    p_unlink.add_argument("email", nargs="?", default=None)
    p_unlink.add_argument("--sub", default=None)
    p_unlink.set_defaults(func=cmd_unlink)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("from_id")
    p_merge.add_argument("to_id")
    p_merge.add_argument("--apply", action="store_true", help="actually mutate data")
    p_merge.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
