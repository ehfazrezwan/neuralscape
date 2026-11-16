"""scripts/reembed_graph.py against a real Neo4j.

Seeds a target group and a control group with 1024-d vectors on the three
properties Graphiti writes (``Entity.name_embedding``, ``Community.name_embedding``,
``RELATES_TO.fact_embedding``) using the same ``db.create.set*VectorProperty``
procedures Graphiti uses, then drives the script's ``main`` against it.
"""

from __future__ import annotations

import importlib.util
import io
import math
import random
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reembed_graph.py"
DIM_FROM, DIM_TO = 1024, 768

PROPS = (("Entity", "name_embedding"), ("Community", "name_embedding"), ("RELATES_TO", "fact_embedding"))


def _load_script():
    spec = importlib.util.spec_from_file_location("reembed_graph_it", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vec(seed: int, n: int = DIM_FROM) -> list[float]:
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(n)]


@pytest.fixture
def store(neo4j_conn, unique_ns, monkeypatch):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_conn.uri, auth=(neo4j_conn.user, neo4j_conn.password))
    target, control = f"{unique_ns.group_prefix}_target", f"{unique_ns.group_prefix}_control"

    def run(query, **params):
        with driver.session(database=neo4j_conn.database) as s:
            return [dict(r) for r in s.run(query, **params)]

    def cleanup():
        run("MATCH (n) WHERE n.group_id STARTS WITH $p DETACH DELETE n", p=unique_ns.group_prefix)
        driver.close()

    unique_ns.on_cleanup(cleanup)

    seeds: dict[str, list[float]] = {}
    for gi, group in enumerate((target, control)):
        for i in range(3):
            for label in ("Entity", "Community"):
                uid = f"{group}-{label.lower()}-{i}"
                seeds[uid] = _vec(hash((gi, label, i)) & 0xFFFF)
                run(
                    f"CREATE (n:{label} {{uuid: $u, group_id: $g, name: $name}}) "
                    'WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $v)',
                    u=uid, g=group, name=f"name {uid}", v=seeds[uid],
                )
            uid = f"{group}-edge-{i}"
            seeds[uid] = _vec(hash((gi, "edge", i)) & 0xFFFF)
            run(
                "CREATE (a:Entity {uuid: $a, group_id: $g})-[e:RELATES_TO {uuid: $u, group_id: $g, fact: $f}]->"
                "(b:Entity {uuid: $b, group_id: $g}) "
                'WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $v)',
                a=f"{uid}-a", b=f"{uid}-b", u=uid, g=group, f=f"fact {uid}", v=seeds[uid],
            )

    monkeypatch.setenv("REEMBED_NEO4J_URI", neo4j_conn.uri)
    monkeypatch.setenv("REEMBED_NEO4J_USER", neo4j_conn.user)
    monkeypatch.setenv("REEMBED_NEO4J_PASSWORD", neo4j_conn.password)

    class S:
        pass

    s = S()
    s.run, s.target, s.control, s.seeds, s.db = run, target, control, seeds, neo4j_conn.database

    def snapshot(group):
        out = {}
        out.update(
            {
                r["u"]: r["v"]
                for r in run(
                    "MATCH (n) WHERE n.group_id = $g AND n.name_embedding IS NOT NULL "
                    "RETURN n.uuid AS u, n.name_embedding AS v", g=group,
                )
            }
        )
        out.update(
            {
                r["u"]: r["v"]
                for r in run(
                    "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $g AND e.fact_embedding IS NOT NULL "
                    "RETURN e.uuid AS u, e.fact_embedding AS v", g=group,
                )
            }
        )
        return out

    s.snapshot = snapshot
    return s


def _main(store, *argv):
    mod = _load_script()
    out = io.StringIO()
    code = mod.main(
        ["--database", store.db, "--embedder-model", "gemini-embedding-001", *argv], out=out
    )
    return code, out.getvalue()


def test_dry_run_reports_lengths_and_changes_nothing(store):
    before = {g: store.snapshot(g) for g in (store.target, store.control)}
    code, out = _main(store, "--group-id", store.target, "--group-id", store.control)
    assert code == 0 and "dry-run: nothing was written" in out
    assert f"group {store.target}:" in out and f"group {store.control}:" in out
    assert "1024-d: 3" in out
    # the schema scan must recognise the three handled properties (no "other" noise for them)
    tail = out.split("Other embedding-like")[1].splitlines() if "Other embedding-like" in out else []
    assert not [l for l in tail if "fact_embedding" in l or "RELATES_TO" in l or "name_embedding" in l]
    after = {g: store.snapshot(g) for g in (store.target, store.control)}
    assert after == before  # untouched, bit for bit


def test_truncate_converts_only_the_target_group(store):
    control_before = store.snapshot(store.control)
    code, out = _main(
        store, "--group-id", store.target, "--mode", "truncate",
        "--target-dim", str(DIM_TO), "--confirm-backup",
    )
    assert code == 0, out
    target = store.snapshot(store.target)
    assert len(target) == 9  # 3 entities + 3 communities + 3 edges
    for uid, vec in target.items():
        seed = store.seeds[uid]
        head = seed[:DIM_TO]
        norm = math.sqrt(sum(x * x for x in head))
        assert len(vec) == DIM_TO
        assert math.sqrt(sum(x * x for x in vec)) == pytest.approx(1.0, abs=1e-5)
        assert vec == pytest.approx([x / norm for x in head], abs=1e-5)
    assert store.snapshot(store.control) == control_before  # control untouched


def test_second_truncate_run_is_a_no_op(store):
    args = ("--group-id", store.target, "--mode", "truncate", "--target-dim", str(DIM_TO), "--confirm-backup")
    _main(store, *args)
    after_first = store.snapshot(store.target)
    code, out = _main(store, *args)
    assert code == 0
    assert out.count("converted=0") == 3  # one summary line per target, nothing converted
    assert store.snapshot(store.target) == after_first


def test_limit_bounds_a_trial_run(store):
    _main(
        store, "--group-id", store.target, "--mode", "truncate",
        "--target-dim", str(DIM_TO), "--confirm-backup", "--limit", "1",
    )
    lens = [len(v) for v in store.snapshot(store.target).values()]
    assert lens.count(DIM_TO) == 3 and lens.count(DIM_FROM) == 6  # one per property
