"""scripts/reembed_graph.py against an in-memory fake Neo4j (unit tests only;
the real-store proof is a planned integration test, see the PR description)."""

import importlib.util
import io
import math
import re
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "reembed_graph", Path(__file__).resolve().parent.parent / "scripts" / "reembed_graph.py"
)
rg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rg)


class FakeNeo4j:
    """Interprets the script's tagged queries over dict-of-rows tables."""

    def __init__(self):
        # table key = (label, prop); rows = {uuid: {"group_id", "emb", "text"}}
        self.tables = {(label, prop): {} for _, label, prop, _ in rg.TARGETS}
        self.queries: list[str] = []
        self.writes: list[tuple] = []
        self.closed = False

    def add(self, label, prop, uuid, group_id, emb, text="t"):
        self.tables[(label, prop)][uuid] = {"group_id": group_id, "emb": emb, "text": text}

    # driver API
    def session(self, database=None):
        return _Session(self)

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **p):
        db = self.db
        db.queries.append(query)
        tag = re.match(r"// reembed:(\S+)", query).group(1)
        if tag in ("schema-nodes", "schema-rels"):
            if tag == "schema-nodes":
                return [
                    {"owner": ["Entity"], "propertyName": "name_embedding"},
                    {"owner": ["Entity"], "propertyName": "summary_embedding"},
                ]
            return [{"owner": ":`RELATES_TO`", "propertyName": "fact_embedding"}]
        key = _table_for(query, db)
        rows = db.tables[key]
        if tag == "histogram":
            out: dict = {}
            for r in rows.values():
                if "$groups" in query and r["group_id"] not in p["groups"]:
                    continue
                d = 0 if r["emb"] is None else len(r["emb"])
                out[(r["group_id"], d)] = out.get((r["group_id"], d), 0) + 1
            return [{"group_id": g, "dim": d, "n": n} for (g, d), n in sorted(out.items())]
        if tag == "page":
            sel = [
                (u, r)
                for u, r in sorted(rows.items())
                if (("$groups" not in query) or r["group_id"] in p["groups"])
                and r["emb"] is not None
                and len(r["emb"]) != p["target"]
                and u > p["after"]
            ][: p["batch"]]
            return [{"uuid": u, "emb": r["emb"], "text": r["text"]} for u, r in sel]
        if tag == "write":
            for row in p["rows"]:
                rows[row["uuid"]]["emb"] = row["emb"]
                db.writes.append((key, row["uuid"], len(row["emb"])))
            return [{"n": len(p["rows"])}]
        raise AssertionError(tag)


def _table_for(query, db):
    for _, label, prop, _ in rg.TARGETS:
        if re.search(rf"[:\[]{label}\b", query) and prop in query:
            return (label, prop)
    return None


ENV = {
    "REEMBED_NEO4J_URI": "neo4j://test-host:7687",
    "REEMBED_NEO4J_USER": "u",
    "REEMBED_NEO4J_PASSWORD": "p",
}


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _vec(n, scale=2.0):
    return [scale * (i + 1) for i in range(n)]


def _seed():
    db = FakeNeo4j()
    for i in range(5):
        db.add("Entity", "name_embedding", f"e{i}", "g1", _vec(1024))
    db.add("Entity", "name_embedding", "e-done", "g1", [1.0] + [0.0] * 767)  # already 768
    db.add("Entity", "name_embedding", "e-other", "g2", _vec(1024))
    for i in range(3):
        db.add("RELATES_TO", "fact_embedding", f"r{i}", "g1", _vec(1024))
    return db


def _run(db, *argv, embedder_factory=None):
    out = io.StringIO()
    code = rg.main(
        list(argv), driver_factory=lambda *a: db, embedder_factory=embedder_factory, out=out
    )
    return code, out.getvalue()


def _snapshot(db):
    return {k: {u: list(r["emb"]) for u, r in t.items()} for k, t in db.tables.items()}


class TestDryRun:
    def test_default_mode_is_dry_run_and_writes_nothing(self, env):
        db = _seed()
        before = _snapshot(db)
        code, out = _run(db, "--all-groups")
        assert code == 0
        assert db.writes == []
        assert _snapshot(db) == before
        assert not any(q.startswith("// reembed:write") for q in db.queries)
        assert "dry-run: nothing was written" in out
        assert "group g1: 768-d: 1, 1024-d: 5" in out
        assert "group g2: 1024-d: 1" in out
        assert "Entity.summary_embedding" in out  # unknown embedding property is reported
        tail = out.split("Other embedding-like")[1].splitlines()
        # Neo4j reports the rel type as :`RELATES_TO`; no line of the "other
        # properties" list may mention a handled property in any spelling.
        assert not [l for l in tail if "fact_embedding" in l or "RELATES_TO" in l or "name_embedding" in l]

    def test_scope_limits_groups(self, env):
        _, out = _run(_seed(), "--group-id", "g2", "--dry-run")
        assert "group g2" in out and "group g1" not in out

    def test_dry_run_with_confirm_flags_still_read_only(self, env):
        db = _seed()
        _run(db, "--all-groups", "--confirm-backup")
        assert db.writes == []


class TestTruncate:
    def test_writes_renormalised_prefixes_and_skips_done(self, env):
        db = _seed()
        original = _vec(1024)
        code, out = _run(
            db, "--group-id", "g1", "--mode", "truncate", "--target-dim", "768", "--confirm-backup"
        )
        assert code == 0
        for i in range(5):
            v = db.tables[("Entity", "name_embedding")][f"e{i}"]["emb"]
            assert len(v) == 768
            assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
            # direction of the original prefix preserved
            ratio = v[0] / original[0]
            assert all(math.isclose(v[j], original[j] * ratio, rel_tol=1e-9) for j in range(768))
        # untouched: already at target, and a group out of scope
        assert db.tables[("Entity", "name_embedding")]["e-done"]["emb"] == [1.0] + [0.0] * 767
        assert len(db.tables[("Entity", "name_embedding")]["e-other"]["emb"]) == 1024
        assert all(uuid != "e-done" for _, uuid, _ in db.writes)
        assert len(db.tables[("RELATES_TO", "fact_embedding")]["r0"]["emb"]) == 768

    def test_rerun_is_idempotent(self, env):
        db = _seed()
        args = ("--all-groups", "--mode", "truncate", "--target-dim", "768", "--confirm-backup")
        _run(db, *args)
        n = len(db.writes)
        snap = _snapshot(db)
        _run(db, *args)
        assert len(db.writes) == n and _snapshot(db) == snap

    def test_limit_bounds_a_trial_run(self, env):
        db = _seed()
        _run(
            db, "--all-groups", "--mode", "truncate", "--target-dim", "768",
            "--confirm-backup", "--limit", "2", "--batch-size", "10",
        )
        converted_entities = [w for w in db.writes if w[0] == ("Entity", "name_embedding")]
        converted_edges = [w for w in db.writes if w[0] == ("RELATES_TO", "fact_embedding")]
        assert len(converted_entities) == 2 and len(converted_edges) == 2

    def test_batches(self, env):
        db = _seed()
        _run(
            db, "--all-groups", "--mode", "truncate", "--target-dim", "768",
            "--confirm-backup", "--batch-size", "2",
        )
        assert sum(q.startswith("// reembed:write") for q in db.queries) >= 4
        assert all(len(r["emb"]) == 768 for t in db.tables.values() for r in t.values())

    def test_refuses_non_mrl_model(self, env, capsys):
        db = _seed()
        code, _ = _run(
            db, "--all-groups", "--mode", "truncate", "--target-dim", "768",
            "--confirm-backup", "--embedder-model", "text-embedding-3-small",
        )
        assert code == 2 and db.writes == []
        assert "not a known Matryoshka" in capsys.readouterr().err

    def test_vector_shorter_than_target_is_left_alone(self, env):
        db = FakeNeo4j()
        db.add("Entity", "name_embedding", "short", "g1", [1.0] * 100)
        _run(db, "--all-groups", "--mode", "truncate", "--target-dim", "768", "--confirm-backup")
        assert db.writes == [] and len(db.tables[("Entity", "name_embedding")]["short"]["emb"]) == 100


class TestReembed:
    class _Embedder:
        def __init__(self, dim):
            self.dim = dim
            self.calls = []

        async def create_batch(self, texts):
            self.calls.append(list(texts))
            return [[0.5] * self.dim for _ in texts]

    def test_reembeds_text_in_batches_at_target_dim(self, env):
        db = FakeNeo4j()
        for i in range(5):
            db.add("Entity", "name_embedding", f"e{i}", "g1", _vec(1024), text=f"name {i}\nx")
        emb = self._Embedder(768)
        seen = {}

        def factory(model, dim):
            seen.update(model=model, dim=dim)
            return emb

        code, _ = _run(
            db, "--group-id", "g1", "--mode", "reembed", "--target-dim", "768",
            "--confirm-backup", "--batch-size", "2", embedder_factory=factory,
        )
        assert code == 0 and seen["dim"] == 768
        assert [len(c) for c in emb.calls] == [2, 2, 1]
        assert emb.calls[0][0] == "name 0 x"  # newline flattened like Graphiti does
        assert all(len(r["emb"]) == 768 for r in db.tables[("Entity", "name_embedding")].values())

    def test_empty_text_rows_are_skipped_and_counted_not_embedded(self, env):
        db = FakeNeo4j()
        db.add("Entity", "name_embedding", "a", "g1", _vec(1024), text="")
        db.add("Entity", "name_embedding", "b", "g1", _vec(1024), text=None)
        db.add("Entity", "name_embedding", "c", "g1", _vec(1024), text="   ")
        db.add("Entity", "name_embedding", "d", "g1", _vec(1024), text="real name")
        emb = self._Embedder(768)
        _, out = _run(
            db, "--all-groups", "--mode", "reembed", "--target-dim", "768",
            "--confirm-backup", embedder_factory=lambda m, d: emb,
        )
        assert emb.calls == [["real name"]]
        assert [w[1] for w in db.writes] == ["d"]
        assert "Entity.name_embedding: converted=1 skipped=3" in out

    def test_reembed_skips_vectors_already_at_target(self, env):
        db = FakeNeo4j()
        db.add("Entity", "name_embedding", "a", "g1", [0.1] * 768)
        emb = self._Embedder(768)
        _run(
            db, "--all-groups", "--mode", "reembed", "--target-dim", "768",
            "--confirm-backup", embedder_factory=lambda m, d: emb,
        )
        assert emb.calls == [] and db.writes == []


class TestRefusals:
    def test_refuses_without_explicit_connection(self, monkeypatch):
        for k in ENV:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(SystemExit) as exc:
            rg.main(["--all-groups"], driver_factory=lambda *a: pytest.fail("connected"))
        assert "no explicit Neo4j connection" in str(exc.value)

    def test_service_neo4j_env_is_not_a_hidden_default(self, monkeypatch):
        for k in ENV:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("NEO4J_URI", "neo4j://prod:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        with pytest.raises(SystemExit):
            rg.main(["--all-groups"], driver_factory=lambda *a: pytest.fail("connected"))

    def test_requires_explicit_scope(self, env):
        with pytest.raises(SystemExit):
            _run(FakeNeo4j())

    def test_scope_flags_are_exclusive(self, env):
        with pytest.raises(SystemExit):
            _run(FakeNeo4j(), "--all-groups", "--group-id", "g1")

    def test_write_mode_needs_target_dim_and_backup_confirmation(self, env):
        with pytest.raises(SystemExit):
            _run(FakeNeo4j(), "--all-groups", "--mode", "truncate", "--confirm-backup")
        with pytest.raises(SystemExit):
            _run(FakeNeo4j(), "--all-groups", "--mode", "truncate", "--target-dim", "768")

    def test_password_is_not_a_command_line_option(self, env):
        with pytest.raises(SystemExit):
            _run(FakeNeo4j(), "--all-groups", "--password", "x")
