#!/usr/bin/env python
"""Inspect or change the dimension of the vectors Graphiti stores in Neo4j.

Graphiti keeps one embedding per entity name (``Entity.name_embedding``,
``Community.name_embedding``) and one per relationship fact
(``RELATES_TO.fact_embedding``). The service writes them at
``GRAPHITI_EMBEDDING_DIM`` (default 1024); the vector store is 768-d. This
script moves an existing store to another dimension. See
``docs/GRAPH_EMBEDDING_DIMENSION.md`` for the whole procedure.

Modes
-----
``dry-run`` (default)
    Read-only. Counts every embedding property by vector length, per group_id,
    and lists any other ``*embedding*`` property found in the schema.
``truncate``  (needs ``--target-dim``)
    No API calls. Keeps the leading ``target-dim`` values of each longer
    vector and renormalises them. Valid only when the graph embedder is a
    Matryoshka (MRL) model such as gemini-embedding-001; refused otherwise.
``reembed``   (needs ``--target-dim``)
    Re-embeds the entity name / edge fact text with the configured Graphiti
    embedder at ``--target-dim``, in batches. Costs embedding API calls.

Safety
------
* No hidden default store. The Neo4j target must be given explicitly:
  ``--uri`` (or ``REEMBED_NEO4J_URI``), ``--user`` (or ``REEMBED_NEO4J_USER``)
  and the password in ``REEMBED_NEO4J_PASSWORD`` (never on the command line).
  The service's own ``NEO4J_*`` settings are deliberately NOT used.
* Scope is explicit: one or more ``--group-id`` or ``--all-groups``.
* Write modes need ``--confirm-backup`` (you have a backup of the store).
* Idempotent and resumable: a vector already at the target length is skipped,
  so an interrupted run is simply re-run. ``--limit N`` bounds a trial run.

Examples
--------
    export REEMBED_NEO4J_PASSWORD=...        # not echoed, not in argv
    python scripts/reembed_graph.py --uri neo4j://localhost:7687 --user neo4j \\
        --all-groups                                   # dry-run (default)
    python scripts/reembed_graph.py --uri ... --user neo4j --group-id user--alice \\
        --mode truncate --target-dim 768 --limit 50 --confirm-backup
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_DIR = _HERE.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

import third_party_telemetry  # noqa: E402,F401 — must precede any mem0/Graphiti import (defaults their telemetry off)

# (kind, label/type, property, text property used when re-embedding)
TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("node", "Entity", "name_embedding", "name"),
    ("node", "Community", "name_embedding", "name"),
    ("rel", "RELATES_TO", "fact_embedding", "fact"),
)

DEFAULT_BATCH_SIZE = 100


def _match(kind: str, label: str) -> tuple[str, str]:
    """Return (MATCH pattern, variable) for a target."""
    if kind == "node":
        return f"(x:{label})", "x"
    return f"()-[x:{label}]->()", "x"


def _scope_clause(all_groups: bool) -> str:
    return "" if all_groups else "x.group_id IN $groups AND "


# ── Cypher (each query is tagged with a // marker comment) ─────────────


def q_histogram(kind: str, label: str, prop: str, all_groups: bool) -> str:
    pattern, _ = _match(kind, label)
    where = "" if all_groups else "WHERE x.group_id IN $groups"
    return (
        "// reembed:histogram\n"
        f"MATCH {pattern} {where}\n"
        f"RETURN x.group_id AS group_id, "
        f"CASE WHEN x.{prop} IS NULL THEN 0 ELSE size(x.{prop}) END AS dim, "
        f"count(*) AS n ORDER BY group_id, dim"
    )


def q_page(kind: str, label: str, prop: str, text_prop: str, all_groups: bool) -> str:
    """Next page of rows whose vector is not yet at ``$target``, keyset-paged
    on uuid so shorter/skipped rows can never make the scan loop forever."""
    pattern, _ = _match(kind, label)
    return (
        "// reembed:page\n"
        f"MATCH {pattern}\n"
        f"WHERE {_scope_clause(all_groups)}x.{prop} IS NOT NULL "
        f"AND size(x.{prop}) <> $target AND x.uuid > $after\n"
        f"RETURN x.uuid AS uuid, x.{prop} AS emb, x.{text_prop} AS text\n"
        "ORDER BY x.uuid LIMIT $batch"
    )


def q_write(kind: str, label: str, prop: str) -> str:
    # Graphiti stores these through db.create.set*VectorProperty; do the same.
    if kind == "node":
        return (
            "// reembed:write\n"
            "UNWIND $rows AS row\n"
            f"MATCH (x:{label} {{uuid: row.uuid}})\n"
            f'CALL db.create.setNodeVectorProperty(x, "{prop}", row.emb)\n'
            "RETURN count(x) AS n"
        )
    return (
        "// reembed:write\n"
        "UNWIND $rows AS row\n"
        f"MATCH ()-[x:{label} {{uuid: row.uuid}}]->()\n"
        f'CALL db.create.setRelationshipVectorProperty(x, "{prop}", row.emb)\n'
        "RETURN count(x) AS n"
    )


Q_SCHEMA_NODES = (
    "// reembed:schema-nodes\n"
    "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
    "RETURN nodeLabels AS owner, propertyName"
)
Q_SCHEMA_RELS = (
    "// reembed:schema-rels\n"
    "CALL db.schema.relTypeProperties() YIELD relType, propertyName "
    "RETURN relType AS owner, propertyName"
)


# ── helpers ────────────────────────────────────────────────────────────


def renormalised_prefix(vec, target_dim: int) -> list[float] | None:
    """Leading ``target_dim`` values of ``vec``, scaled to unit length.
    ``None`` if the vector is shorter than the target or has zero norm."""
    if vec is None or len(vec) < target_dim:
        return None
    head = [float(v) for v in vec[:target_dim]]
    norm = math.sqrt(sum(v * v for v in head))
    if norm <= 0.0:
        return None
    return [v / norm for v in head]


class Store:
    """Thin wrapper over a Neo4j driver: ``read`` and ``write`` both return
    a list of plain dicts."""

    def __init__(self, driver, database: str):
        self._driver = driver
        self._database = database

    def run(self, query: str, **params) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            return [dict(r) for r in session.run(query, **params)]


def _resolve_connection(args) -> tuple[str, str, str]:
    uri = args.uri or os.environ.get("REEMBED_NEO4J_URI")
    user = args.user or os.environ.get("REEMBED_NEO4J_USER")
    password = os.environ.get("REEMBED_NEO4J_PASSWORD")
    missing = [
        name
        for name, val in (
            ("--uri / REEMBED_NEO4J_URI", uri),
            ("--user / REEMBED_NEO4J_USER", user),
            ("REEMBED_NEO4J_PASSWORD", password),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            "refusing to run: no explicit Neo4j connection (missing "
            + ", ".join(missing)
            + "). This script has no default store; the service's NEO4J_* "
            "settings are intentionally not used."
        )
    return uri, user, password


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--uri", help="Neo4j URI (or REEMBED_NEO4J_URI)")
    p.add_argument("--user", help="Neo4j user (or REEMBED_NEO4J_USER)")
    p.add_argument("--database", default="neo4j", help="Neo4j database (default: neo4j)")
    scope = p.add_argument_group("scope (one required)")
    scope.add_argument(
        "--group-id", action="append", default=[], dest="group_ids",
        help="Graphiti group_id to process; repeatable",
    )
    scope.add_argument("--all-groups", action="store_true", help="process every group_id")
    p.add_argument(
        "--mode", choices=("dry-run", "truncate", "reembed"), default="dry-run",
        help="default: dry-run (read-only)",
    )
    p.add_argument("--dry-run", action="store_true", help="alias for --mode dry-run")
    p.add_argument("--target-dim", type=int, help="vector length to move to")
    p.add_argument("--limit", type=int, default=0, help="stop after N vectors per target (0 = all)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--embedder-model",
        help="Graphiti embedder model (default: the service's GEMINI_EMBEDDER_MODEL)",
    )
    p.add_argument(
        "--confirm-backup", action="store_true",
        help="required for truncate/reembed: you have a backup of this Neo4j store",
    )
    return p


def _default_embedder_model() -> str:
    from config import settings

    return settings.gemini_embedder_model


def _default_embedder_factory(model: str, target_dim: int):
    from config import settings
    from mem0.memory.graphiti_memory import _create_embedder

    return _create_embedder("gemini", model, settings.google_api_key, embedding_dim=target_dim)


# ── modes ──────────────────────────────────────────────────────────────


def run_dry_run(store: Store, group_ids: list[str], all_groups: bool, out) -> dict:
    """Read-only inventory. Returns {(label, prop): {group_id: {dim: n}}}."""
    report: dict = {}
    for kind, label, prop, _ in TARGETS:
        rows = store.run(q_histogram(kind, label, prop, all_groups), groups=group_ids)
        by_group: dict = defaultdict(dict)
        for r in rows:
            by_group[r["group_id"]][r["dim"]] = r["n"]
        report[(label, prop)] = dict(by_group)
        print(f"\n{label}.{prop}", file=out)
        if not by_group:
            print("  (no rows in scope)", file=out)
        for gid, dims in sorted(by_group.items(), key=lambda kv: str(kv[0])):
            cells = ", ".join(
                f"{'missing' if d == 0 else str(d) + '-d'}: {n}" for d, n in sorted(dims.items())
            )
            print(f"  group {gid}: {cells}", file=out)
    known = {(label, prop) for _, label, prop, _ in TARGETS}
    extras = set()
    for query in (Q_SCHEMA_NODES, Q_SCHEMA_RELS):
        try:
            for r in store.run(query):
                owner = r["owner"]
                owners = owner if isinstance(owner, list) else [owner]
                # relTypeProperties reports the type as ":`RELATES_TO`"
                owners = [str(o).strip(":`") for o in owners]
                prop = r["propertyName"]
                if prop and "embedding" in prop.lower() and not any(
                    (o, prop) in known for o in owners
                ):
                    extras.add((",".join(str(o) for o in owners), prop))
        except Exception as exc:  # schema procedures are best-effort
            print(f"\n(schema scan skipped: {exc})", file=out)
    if extras:
        print("\nOther embedding-like properties found (NOT handled by this script):", file=out)
        for owner, prop in sorted(extras):
            print(f"  {owner}.{prop}", file=out)
    print("\ndry-run: nothing was written.", file=out)
    return report


def run_convert(
    store: Store,
    mode: str,
    group_ids: list[str],
    all_groups: bool,
    target_dim: int,
    limit: int,
    batch_size: int,
    embedder,
    out,
) -> dict:
    """Truncate or re-embed. Returns {(label, prop): {"converted", "skipped"}}."""
    summary: dict = {}
    loop = asyncio.new_event_loop() if mode == "reembed" else None
    try:
        _convert_all(
            store, mode, group_ids, all_groups, target_dim, limit, batch_size,
            embedder, loop, summary, out,
        )
    finally:
        if loop is not None:
            loop.close()
    print(f"\nsummary ({mode} -> {target_dim}-d):", file=out)
    for (label, prop), s in summary.items():
        print(f"  {label}.{prop}: converted={s['converted']} skipped={s['skipped']}", file=out)
    return summary


def _convert_all(
    store, mode, group_ids, all_groups, target_dim, limit, batch_size,
    embedder, loop, summary, out,
) -> None:
    for kind, label, prop, text_prop in TARGETS:
        converted = skipped = 0
        after = ""
        while True:
            if limit and converted >= limit:
                break
            page = store.run(
                q_page(kind, label, prop, text_prop, all_groups),
                groups=group_ids,
                target=target_dim,
                after=after,
                batch=batch_size,
            )
            if not page:
                break
            after = page[-1]["uuid"]
            rows = []
            if mode == "truncate":
                for r in page:
                    vec = renormalised_prefix(r["emb"], target_dim)
                    if vec is None:  # shorter than target (or zero): leave alone
                        skipped += 1
                    else:
                        rows.append({"uuid": r["uuid"], "emb": vec})
            else:
                usable = [r for r in page if (r["text"] or "").strip()]
                skipped += len(page) - len(usable)  # empty name/fact: nothing to embed
                texts = [r["text"].replace("\n", " ") for r in usable]
                vecs = loop.run_until_complete(embedder.create_batch(texts)) if texts else []
                for r, vec in zip(usable, vecs):
                    if not vec or len(vec) != target_dim:
                        skipped += 1
                    else:
                        rows.append({"uuid": r["uuid"], "emb": list(vec)})
            if limit:
                rows = rows[: max(0, limit - converted)]
            if rows:
                store.run(q_write(kind, label, prop), rows=rows)
                converted += len(rows)
            print(f"  {label}.{prop}: {converted} converted, {skipped} skipped", file=out)
        summary[(label, prop)] = {"converted": converted, "skipped": skipped}


def main(argv=None, *, driver_factory=None, embedder_factory=None, out=None) -> int:
    out = out or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = "dry-run" if args.dry_run else args.mode
    if args.dry_run and args.mode not in ("dry-run",):
        parser.error("--dry-run conflicts with --mode " + args.mode)

    if not args.group_ids and not args.all_groups:
        parser.error("choose a scope: --group-id <id> (repeatable) or --all-groups")
    if args.group_ids and args.all_groups:
        parser.error("--group-id and --all-groups are mutually exclusive")
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be >= 1 and --limit >= 0")

    uri, user, password = _resolve_connection(args)

    if mode != "dry-run":
        if not args.target_dim or args.target_dim < 1:
            parser.error(f"--mode {mode} requires --target-dim")
        if not args.confirm_backup:
            parser.error(f"--mode {mode} writes to Neo4j: pass --confirm-backup once you have a backup")
        if mode == "truncate":
            from memory.ranking import _MRL_EMBEDDING_MODELS, _canonical_embedding_model

            model = args.embedder_model or _default_embedder_model()
            if _canonical_embedding_model(model) not in _MRL_EMBEDDING_MODELS:
                print(
                    f"refusing --mode truncate: embedder model {model!r} is not a known "
                    "Matryoshka (MRL) model, so a prefix is not a valid lower-dimension "
                    "embedding. Use --mode reembed.",
                    file=sys.stderr,
                )
                return 2

    embedder = None
    if mode == "reembed":
        model = args.embedder_model or _default_embedder_model()
        embedder = (embedder_factory or _default_embedder_factory)(model, args.target_dim)

    if driver_factory is None:
        from neo4j import GraphDatabase

        driver_factory = lambda u, usr, pw: GraphDatabase.driver(u, auth=(usr, pw))  # noqa: E731
    driver = driver_factory(uri, user, password)
    store = Store(driver, args.database)
    scope_txt = "all groups" if args.all_groups else ", ".join(args.group_ids)
    print(f"target store: {uri}  database: {args.database}  scope: {scope_txt}  mode: {mode}", file=out)
    try:
        if mode == "dry-run":
            run_dry_run(store, args.group_ids, args.all_groups, out)
        else:
            run_convert(
                store, mode, args.group_ids, args.all_groups, args.target_dim,
                args.limit, args.batch_size, embedder, out,
            )
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
