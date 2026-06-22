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


# A user's private group_ids are exactly "user--{id}" (global) and
# "user--{id}--project--{pid}" (per project). We must NOT use a bare
# STARTS WITH "user--{id}" — that also matches a different user whose id has
# {id} as a prefix (e.g. from_id="a" matching "user--alice"). Match the exact
# global id OR the project-namespace prefix only.
def _exact(uid: str) -> str:
    return f"user--{uid}"


def _proj_prefix(uid: str) -> str:
    return f"user--{uid}--project--"


_OWN_WHERE = "(n.group_id = $exact OR n.group_id STARTS WITH $proj)"
_OWN_WHERE_REL = "(r.group_id = $exact OR r.group_id STARTS WITH $proj)"


def _neo4j_count(session, from_id: str) -> tuple[int, int]:
    p = {"exact": _exact(from_id), "proj": _proj_prefix(from_id)}
    nodes = session.run(
        f"MATCH (n) WHERE {_OWN_WHERE} RETURN count(n) AS c", **p
    ).single()["c"]
    rels = session.run(
        f"MATCH ()-[r]->() WHERE {_OWN_WHERE_REL} RETURN count(r) AS c", **p
    ).single()["c"]
    return nodes, rels


def _neo4j_rekey(session, from_id: str, to_id: str) -> None:
    # Rewrite by replacing the "user--{from_id}" prefix with "user--{to_id}",
    # preserving any "--project--{pid}" suffix via substring.
    p = {"exact": _exact(from_id), "proj": _proj_prefix(from_id), "new": _exact(to_id)}
    session.run(
        f"MATCH (n) WHERE {_OWN_WHERE} "
        "SET n.group_id = $new + substring(n.group_id, size($exact))",
        **p,
    )
    session.run(
        f"MATCH ()-[r]->() WHERE {_OWN_WHERE_REL} "
        "SET r.group_id = $new + substring(r.group_id, size($exact))",
        **p,
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
        offset = None
        try:
            while True:  # paginate — a single scroll page misses owners past the limit
                body = {"limit": 1000, "with_payload": ["user_id"], "with_vector": False}
                if offset is not None:
                    body["offset"] = offset
                res = _qdrant(f"/collections/{coll}/points/scroll", body)["result"]
                for p in res.get("points", []):
                    uid = (p.get("payload") or {}).get("user_id")
                    if uid:
                        owners[uid] = owners.get(uid, 0) + 1
                offset = res.get("next_page_offset")
                if offset is None:
                    break
        except Exception as e:
            print(f"  ({coll}: {e})")
            continue
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
            nodes, rels = _neo4j_count(session, from_id)
        print(f"  Neo4j: {nodes} node(s) + {rels} relationship(s) under "
              f"group_id 'user--{from_id}' (exact + --project--)")
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
            _neo4j_rekey(session, from_id, to_id)
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
