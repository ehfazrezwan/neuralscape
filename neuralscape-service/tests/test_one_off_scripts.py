"""Smoke tests for the one-off maintenance scripts, against current dev's APIs, using fakes.

Each script is imported (so its imports must resolve) and driven through its dry-run /
apply paths with fake stores; no Qdrant, Neo4j, LLM or vault service is needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from schemas import MemoryResponse


# ── migrate_graph_groups ──


KINDS = ("Episodic", "Entity", "Community", "Saga", "RELATES_TO", "MENTIONS", "HAS_MEMBER", "HAS_EPISODE", "NEXT_EPISODE")


class FakeNeo4j:
    """Just enough of a Neo4j session for the legacy-group scan, rewrite, transaction and verify."""

    def __init__(self, rows: dict[str, dict[str, int]], fail_on_kind: str | None = None, skip_rewrite_of: str | None = None):
        self.rows = {k: dict(v) for k, v in rows.items()}  # kind -> {group_id: count}
        self.writes: list[tuple[str, dict]] = []
        self.commits = self.rollbacks = 0
        self.fail_on_kind, self.skip_rewrite_of = fail_on_kind, skip_rewrite_of

    def _kind(self, cypher):
        return next((k for k in KINDS if f":{k} " in cypher or f":{k})" in cypher or f"[x:{k}" in cypher), None)

    def run(self, cypher, **params):
        kind = self._kind(cypher)
        if "SET x.group_id" in cypher:
            old, new = params["old"], params["new"]
            if kind == self.fail_on_kind:
                raise RuntimeError("boom")
            n = self.rows.get(kind, {}).get(old, 0)
            if old != self.skip_rewrite_of:
                self.rows.get(kind, {}).pop(old, None)
                self.rows.setdefault(kind, {})[new] = self.rows.get(kind, {}).get(new, 0) + n
            self.writes.append((kind, params))
            return SimpleNamespace(single=lambda: {"c": n})
        if kind is None and "WHERE x.group_id = $old" in cypher:
            kinds = [k for k in KINDS if (k in ("Episodic", "Entity", "Community", "Saga")) == ("MATCH (x)" in cypher)]
            c = sum(self.rows.get(k, {}).get(params["old"], 0) for k in kinds)
            return SimpleNamespace(single=lambda: {"c": c})
        return [{"g": g, "c": c} for g, c in self.rows.get(kind, {}).items()
                if g == "global" or g.startswith("project--")]

    def begin_transaction(self):
        import copy

        outer = self
        snapshot = copy.deepcopy(self.rows)

        class Tx:
            def __enter__(self_):
                return self_

            def __exit__(self_, exc_type, *a):
                if exc_type is not None:
                    outer.rows = snapshot
                    outer.rollbacks += 1
                return False

            def run(self_, cypher, **params):
                return outer.run(cypher, **params)

            def commit(self_):
                outer.commits += 1

        return Tx()


class TestMigrateGraphGroups:
    def test_rewrite_target_is_explicit_about_the_format(self):
        from config import settings
        from scripts.migrate_graph_groups import rewrite_target

        assert rewrite_target("global", "alice", "legacy") == "user--alice"
        assert rewrite_target("project--payroll", "alice", "legacy") == "user--alice--project--payroll"
        for current in ("user--alice", "shared", "shared--project--x", "standard", "project--", ""):
            assert rewrite_target(current, "alice", "legacy") is None
        # a project id the legacy format cannot name unambiguously: the two formats disagree
        legacy = rewrite_target("project--a.b", "alice", "legacy")
        injective = rewrite_target("project--a.b", "alice", "injective")
        assert legacy == "user--alice--project--a.b" and injective.startswith("nsg1--") and legacy != injective
        # the result never depends on, and never leaks, the process's flag state
        assert settings.graph_group_id_injective_enabled is False
        settings.graph_group_id_injective_enabled = True
        try:
            assert rewrite_target("project--a.b", "alice", "legacy") == legacy
            assert rewrite_target("project--a.b", "alice", "injective") == injective
            assert settings.graph_group_id_injective_enabled is True
        finally:
            settings.graph_group_id_injective_enabled = False
        with pytest.raises(ValueError):
            rewrite_target("global", "alice", "bogus")

    def test_scan_covers_nodes_and_every_edge_relationship_type(self):
        from scripts.migrate_graph_groups import scan_legacy_groups

        session = FakeNeo4j({"Entity": {"global": 2}, "RELATES_TO": {"global": 5, "project--p": 1}, "MENTIONS": {"global": 3},
                             "Saga": {"global": 1}, "HAS_EPISODE": {"global": 4}, "NEXT_EPISODE": {"project--p": 2},
                             "Episodic": {"user--alice": 9}})
        plan = scan_legacy_groups(session, "alice")
        assert plan["RELATES_TO"] == [("global", "user--alice", 5), ("project--p", "user--alice--project--p", 1)]
        assert plan["MENTIONS"] == [("global", "user--alice", 3)]
        assert plan["Entity"] == [("global", "user--alice", 2)]
        assert plan["Saga"] == [("global", "user--alice", 1)] and plan["HAS_EPISODE"] == [("global", "user--alice", 4)]
        assert plan["NEXT_EPISODE"] == [("project--p", "user--alice--project--p", 2)]
        assert plan["Episodic"] == [] and plan["Community"] == [] and plan["HAS_MEMBER"] == []

    def test_scan_names_targets_in_the_requested_format(self):
        from scripts.migrate_graph_groups import scan_legacy_groups

        session = FakeNeo4j({"Entity": {"project--a.b": 1}})
        assert scan_legacy_groups(session, "alice", "legacy")["Entity"] == [("project--a.b", "user--alice--project--a.b", 1)]
        assert scan_legacy_groups(session, "alice", "injective")["Entity"][0][1].startswith("nsg1--")

    def test_apply_rewrites_edges_not_only_nodes_and_is_idempotent(self):
        from scripts.migrate_graph_groups import apply_rewrites, scan_legacy_groups

        session = FakeNeo4j({"Entity": {"global": 2}, "RELATES_TO": {"global": 5}})
        plan = scan_legacy_groups(session, "alice")
        assert apply_rewrites(session, plan, print_fn=lambda *_: None) == 7
        assert session.rows["RELATES_TO"] == {"user--alice": 5}
        again = scan_legacy_groups(session, "alice")
        assert all(not per for per in again.values())

    def test_one_transaction_per_legacy_group_across_all_kinds(self):
        from scripts.migrate_graph_groups import apply_rewrites, scan_legacy_groups

        session = FakeNeo4j({"Entity": {"global": 2, "project--p": 1}, "RELATES_TO": {"global": 5}})
        apply_rewrites(session, scan_legacy_groups(session, "alice"), print_fn=lambda *_: None)
        assert session.commits == 2  # 'global' and 'project--p'

    def test_a_failure_rolls_the_group_back_and_stops(self):
        from scripts.migrate_graph_groups import apply_rewrites, scan_legacy_groups

        session = FakeNeo4j({"Entity": {"global": 2}, "RELATES_TO": {"global": 5}}, fail_on_kind="RELATES_TO")
        with pytest.raises(RuntimeError, match="boom"):
            apply_rewrites(session, scan_legacy_groups(session, "alice"), print_fn=lambda *_: None)
        assert session.rollbacks == 1 and session.rows["Entity"] == {"global": 2}   # nothing half-renamed

    def test_verification_fails_loudly_when_rows_keep_the_old_group(self):
        from scripts.migrate_graph_groups import apply_rewrites, scan_legacy_groups

        session = FakeNeo4j({"Entity": {"global": 2}}, skip_rewrite_of="global")
        with pytest.raises(RuntimeError, match="verification failed"):
            apply_rewrites(session, scan_legacy_groups(session, "alice"), print_fn=lambda *_: None)

    def test_cypher_targets_relationships_with_a_relationship_pattern(self):
        from scripts.migrate_graph_groups import apply_rewrites

        seen = []

        class Tx:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def run(s, cypher, **p):
                seen.append(cypher)
                return SimpleNamespace(single=lambda: {"c": 1})
            def commit(s): pass

        class S:
            def begin_transaction(self): return Tx()
            def run(self, cypher, **p): return SimpleNamespace(single=lambda: {"c": 0})

        apply_rewrites(S(), {"RELATES_TO": [("global", "user--a", 1)], "Entity": [("global", "user--a", 1)]}, print_fn=lambda *_: None)
        assert any("()-[x:RELATES_TO" in c for c in seen) and any("(x:Entity" in c for c in seen)
        assert not any("EntityEdge" in c for c in seen)

    def test_audit_changed_group_rows_runs_on_fake_points(self):
        from scripts.migrate_graph_groups import changed_group_rows

        pts = [SimpleNamespace(id="1", payload={"user_id": "alice", "metadata": {"visibility": "private"}}),
               SimpleNamespace(id="2", payload={"user_id": "a--b", "metadata": {"visibility": "private"}})]
        rows = changed_group_rows(pts)
        assert [r["id"] for r in rows] == ["2"] and rows[0]["new_group_id"].startswith("nsg1--")

    # ── main(): dry run by default, explicit format, refusal on a mismatch ──

    @pytest.fixture
    def neo(self, monkeypatch):
        import config
        import neo4j

        session = FakeNeo4j({"Entity": {"global": 2}, "RELATES_TO": {"global": 5}})

        class Ctx:
            def __enter__(s): return session
            def __exit__(s, *a): return False

        driver = SimpleNamespace(session=lambda database=None: Ctx(), close=lambda: None)
        monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: driver)
        monkeypatch.setattr(config.settings, "neo4j_password", "pw")
        return session

    def test_default_is_a_dry_run_that_changes_nothing(self, neo, capsys):
        from scripts.migrate_graph_groups import main

        assert main(["--owner", "alice", "--target-format", "legacy"]) == 0
        assert neo.writes == [] and neo.commits == 0
        out = capsys.readouterr().out
        assert "Dry-run only" in out and "Target format: legacy" in out and "7 rows to rewrite" in out

    def test_apply_rewrites_and_warns_that_everything_becomes_private(self, neo, capsys):
        from scripts.migrate_graph_groups import main

        assert main(["--owner", "alice", "--target-format", "legacy", "--apply"]) == 0
        assert neo.rows["RELATES_TO"] == {"user--alice": 5}
        assert "PRIVATE to 'alice'" in capsys.readouterr().out

    def test_target_format_is_required(self, neo):
        from scripts.migrate_graph_groups import main

        with pytest.raises(SystemExit) as exc:
            main(["--owner", "alice"])
        assert exc.value.code == 2 and neo.writes == []

    def test_a_format_that_disagrees_with_the_service_is_refused_even_with_apply(self, neo, capsys):
        from scripts.migrate_graph_groups import main

        assert main(["--owner", "alice", "--target-format", "injective", "--apply"]) == 2
        assert neo.writes == [] and "refusing" in capsys.readouterr().err

    def test_force_overrides_the_refusal(self, neo):
        from scripts.migrate_graph_groups import main

        assert main(["--owner", "alice", "--target-format", "injective", "--force"]) == 0


# ── scrub_qdrant_visibility ──


class FakeQdrant:
    def __init__(self, points):
        self.points = points  # list of dicts {id, payload}
        self.calls = []

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        cond = scroll_filter.must[0]
        field, value = cond.key, cond.match.value
        hits = []
        for p in self.points:
            cur = p["payload"]
            for part in field.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            if cur == value:
                hits.append(SimpleNamespace(id=p["id"]))
        return hits, None

    def set_payload(self, collection_name, payload, points, key, wait):
        self.calls.append((key, payload, list(points)))
        for p in self.points:
            if p["id"] in points:
                cur = p["payload"]
                for part in key.split("."):
                    cur = cur[part]
                cur.update(payload)


class TestScrubQdrantVisibility:
    def _points(self):
        return [
            {"id": 1, "payload": {"metadata": {"visibility": "MemoryVisibility.SHARED", "category": "x"}}},
            {"id": 2, "payload": {"metadata": {"visibility": "MemoryVisibility.PRIVATE"}}},
            {"id": 3, "payload": {"metadata": {"metadata": {"visibility": "MemoryVisibility.SHARED"}}}},   # double-wrapped
            {"id": 4, "payload": {"metadata": {"visibility": "shared"}}},
        ]

    def test_dry_run_counts_both_nestings_and_writes_nothing(self):
        from scripts.scrub_qdrant_visibility import _scrub

        client = FakeQdrant(self._points())
        assert _scrub(client, "c", "MemoryVisibility.SHARED", "shared", dry_run=True) == (2, 0)
        assert client.calls == []

    def test_apply_rewrites_both_nestings_and_keeps_other_metadata(self):
        from scripts.scrub_qdrant_visibility import _scrub

        client = FakeQdrant(self._points())
        assert _scrub(client, "c", "MemoryVisibility.SHARED", "shared", dry_run=False) == (2, 2)
        assert {c[0] for c in client.calls} == {"metadata", "metadata.metadata"}
        by_id = {p["id"]: p["payload"] for p in client.points}
        assert by_id[1]["metadata"] == {"visibility": "shared", "category": "x"}
        assert by_id[3]["metadata"]["metadata"]["visibility"] == "shared"
        assert by_id[2]["metadata"]["visibility"] == "MemoryVisibility.PRIVATE"  # other legacy value untouched here
        assert _scrub(client, "c", "MemoryVisibility.SHARED", "shared", dry_run=False) == (0, 0)  # idempotent

    def test_legacy_map_covers_exactly_the_known_values(self):
        from scripts.scrub_qdrant_visibility import _LEGACY_TO_CANONICAL

        assert _LEGACY_TO_CANONICAL == {"MemoryVisibility.SHARED": "shared", "MemoryVisibility.PRIVATE": "private"}


# ── relocate_vault_to_raw (guard added for the dreaming librarian's layout) ──


class TestRelocateGuard:
    def test_forward_leaves_projects_alone_when_the_librarian_owns_it(self, tmp_path):
        from scripts.relocate_vault_to_raw import plan_moves

        (tmp_path / "Projects" / "p").mkdir(parents=True)
        (tmp_path / "Daily").mkdir()
        (tmp_path / "Home.md").write_text("# Home")
        sources = {src.name for src, _ in plan_moves(tmp_path, reverse=False)}
        assert "Daily" in sources and "Projects" not in sources

    def test_forward_moves_projects_for_a_legacy_vault(self, tmp_path):
        from scripts.relocate_vault_to_raw import plan_moves

        (tmp_path / "Projects").mkdir()
        assert "Projects" in {src.name for src, _ in plan_moves(tmp_path, reverse=False)}

    def test_reverse_is_unaffected_by_home_md(self, tmp_path):
        from scripts.relocate_vault_to_raw import plan_moves

        (tmp_path / "_raw" / "Projects").mkdir(parents=True)
        (tmp_path / "Home.md").write_text("# Home")
        assert "Projects" in {src.name for src, _ in plan_moves(tmp_path, reverse=True)}


# ── bulk_promote_visibility ──


class PromoteQdrant:
    def __init__(self, points):
        self.points = points
        self.set_calls = []

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        return [SimpleNamespace(id=p["id"], payload=p["payload"]) for p in self.points], None

    def set_payload(self, collection_name, payload, points):
        self.set_calls.append((payload, list(points)))


class TestBulkPromoteVisibility:
    @pytest.fixture
    def client(self, monkeypatch):
        import config
        import qdrant_client

        fake = PromoteQdrant([
            {"id": "a", "payload": {"user_id": "alice", "metadata": {"category": "tech_stack"}}},
            {"id": "b", "payload": {"user_id": "alice", "metadata": {"metadata": {"category": "tech_stack", "visibility": "shared"}}}},
            {"id": "c", "payload": {"user_id": "alice", "metadata": {"category": "tech_stack", "visibility": "private", "tags": ["t"]}}},
        ])
        monkeypatch.setattr(config.settings, "qdrant_url", "http://qdrant.invalid:6333")
        monkeypatch.setattr(qdrant_client, "QdrantClient", lambda url=None, **kw: fake)
        return fake

    def test_dry_run_changes_nothing(self, client, capsys):
        from scripts.bulk_promote_visibility import main

        assert main(["--owner", "alice", "--category", "tech_stack"]) == 0
        assert client.set_calls == []
        out = capsys.readouterr().out
        assert "will update: 2" in out and "already at target visibility (skip): 1" in out and "Dry-run only" in out

    def test_apply_promotes_unwraps_and_keeps_metadata(self, client):
        from scripts.bulk_promote_visibility import main

        assert main(["--owner", "alice", "--apply"]) == 0
        updated = {pts[0]: payload["metadata"] for payload, pts in client.set_calls}
        assert set(updated) == {"a", "c"}
        assert updated["a"] == {"category": "tech_stack", "visibility": "shared", "owner_user_id": "alice"}
        assert updated["c"]["tags"] == ["t"] and updated["c"]["visibility"] == "shared"

    def test_demote_targets_private(self, client, capsys):
        from scripts.bulk_promote_visibility import main

        main(["--owner", "alice", "--to", "private"])
        assert "Target visibility: 'private'" in capsys.readouterr().out

    def test_missing_qdrant_url_is_an_error(self, monkeypatch, capsys):
        import config

        from scripts.bulk_promote_visibility import main

        monkeypatch.setattr(config.settings, "qdrant_url", "")
        assert main(["--owner", "alice"]) == 2


# ── backfill_vault_writes ──


class TestBackfillVaultWrites:
    def test_payload_matches_the_worker_event_shape(self):
        from scripts.backfill_vault_writes import _payload_from_memory

        mem = MemoryResponse(id="m1", memory="fact", category="decision", scope="project", project_id="p",
                             visibility="shared", owner_user_id="bob", created_at="c")
        payload = _payload_from_memory(mem, "alice")
        assert payload["user_id"] == "alice" and payload["owner_user_id"] == "bob"
        assert payload["source"] == "backfill"  # not the compiler's own source: its self-write guard must not fire
        assert {"memory_id", "content", "category", "scope", "visibility", "created_at", "project_id"} <= set(payload)
        assert _payload_from_memory(MemoryResponse(id="m", memory="x"), "u")["category"] == ""

    def _run(self, monkeypatch, args, memories):
        import scripts.backfill_vault_writes as mod

        service = MagicMock()
        service.get_all_user_ids.return_value = ["alice"]
        service.list_memories.return_value = memories
        registry = MagicMock()
        registry.discover = AsyncMock()
        registry.startup_all = AsyncMock()
        registry.shutdown_all = AsyncMock()
        registry.emit_event = AsyncMock(return_value=SimpleNamespace(responses=[{"ok": 1}]))
        monkeypatch.setattr(mod, "MemoryService", lambda: service)
        monkeypatch.setattr(mod, "ExtensionRegistry", lambda: registry)
        ns = mod.argparse.Namespace(**{"dry_run": False, "user": None, "limit": 10, "verbose": False, **args})
        return asyncio.run(mod._run(ns)), service, registry

    def test_dry_run_emits_nothing_and_still_shuts_down(self, monkeypatch):
        rc, service, registry = self._run(monkeypatch, {"dry_run": True}, [MemoryResponse(id="m1", memory="x", category="decision")])
        assert rc == 0
        registry.emit_event.assert_not_awaited()
        registry.shutdown_all.assert_awaited_once()
        service.close.assert_called_once()

    def test_apply_emits_one_event_per_memory(self, monkeypatch):
        mems = [MemoryResponse(id=f"m{i}", memory="x", category="decision", visibility="shared") for i in range(3)]
        rc, _, registry = self._run(monkeypatch, {}, mems)
        assert rc == 0 and registry.emit_event.await_count == 3
        assert registry.emit_event.await_args.args[0] == "memory_stored"

    def test_emit_failures_make_the_exit_code_nonzero(self, monkeypatch):
        import scripts.backfill_vault_writes as mod

        service = MagicMock()
        service.get_all_user_ids.return_value = ["alice"]
        service.list_memories.return_value = [MemoryResponse(id="m1", memory="x")]
        registry = MagicMock(discover=AsyncMock(), startup_all=AsyncMock(), shutdown_all=AsyncMock(),
                             emit_event=AsyncMock(side_effect=RuntimeError("boom")))
        monkeypatch.setattr(mod, "MemoryService", lambda: service)
        monkeypatch.setattr(mod, "ExtensionRegistry", lambda: registry)
        ns = mod.argparse.Namespace(dry_run=False, user="alice", limit=1, verbose=False)
        assert asyncio.run(mod._run(ns)) == 1


# ── migrate_wiki_to_topics ──


class TestMigrateWikiToTopics:
    def _vault(self, tmp_path):
        page = tmp_path / "Wiki" / "myproj" / "Procedural" / "Workflow.md"
        page.parent.mkdir(parents=True)
        page.write_text('---\ntitle: Workflow\nsource_memory_ids: [m1, m2]\n---\n\n# Workflow\n\nDeploy by pushing to main.\n')
        return tmp_path

    def test_every_import_the_script_needs_resolves_on_current_dev(self):
        import extensions.conversation_compiler.obsidian_writer as ow
        import extensions.dreaming.config as cfg
        import extensions.dreaming.librarian as lib
        import extensions.dreaming.prompts as prompts
        import extensions.dreaming.sweep as sweep

        for mod, names in ((lib, ("split_page", "_slug_title", "render_topic_page", "_write_hub", "_write_home")),
                           (sweep, ("_make_llm_call",)), (cfg, ("dreaming_settings",)),
                           (prompts, ("parse_json_response",)), (ow, ("_atomic_write",))):
            for name in names:
                assert hasattr(mod, name), f"{mod.__name__}.{name} is gone"

    def test_collect_scopes_groups_pages_by_top_level_folder(self, tmp_path):
        from scripts.migrate_wiki_to_topics import collect_scopes

        vault = self._vault(tmp_path)
        assert {k: [p.name for p in v] for k, v in collect_scopes(vault / "Wiki").items()} == {"myproj": ["Workflow.md"]}

    def test_dry_run_calls_the_llm_but_writes_nothing_and_apply_writes_topics(self, tmp_path, monkeypatch):
        import extensions.dreaming.sweep as sweep

        from scripts.migrate_wiki_to_topics import migrate_scope

        vault = self._vault(tmp_path)

        async def make(_settings):
            async def llm(prompt):
                assert "Deploy by pushing to main." in prompt and "SCOPE: myproj" in prompt
                return '{"topics": [{"title": "Release Flow", "summary": "How we ship", "body": "Deploy by pushing to main."}]}'
            return llm

        monkeypatch.setattr(sweep, "_make_llm_call", make)
        pages = sorted((vault / "Wiki" / "myproj").rglob("*.md"))
        dry = asyncio.run(migrate_scope("myproj", pages, vault, apply=False))
        assert dry["topics"] == 1 and not (vault / "Projects").exists()
        done = asyncio.run(migrate_scope("myproj", pages, vault, apply=True))
        written = vault / "Projects" / "myproj" / "Release-Flow.md"
        out = [p for p in (vault / "Projects" / "myproj").glob("*.md") if "Release" in p.name]
        assert done["topics"] == 1 and out, list((vault / "Projects").rglob("*"))
        text = out[0].read_text()
        assert "migrated: true" in text and "m1" in text and "m2" in text

    def test_no_topics_from_the_llm_is_reported_not_raised(self, tmp_path, monkeypatch):
        import extensions.dreaming.sweep as sweep

        from scripts.migrate_wiki_to_topics import migrate_scope

        vault = self._vault(tmp_path)

        async def make(_settings):
            async def llm(prompt):
                return '{"topics": []}'
            return llm

        monkeypatch.setattr(sweep, "_make_llm_call", make)
        res = asyncio.run(migrate_scope("myproj", sorted((vault / "Wiki").rglob("*.md")), vault, apply=True))
        assert res["topics"] == 0 and "no topics" in res["error"]

    def test_missing_wiki_tree_is_a_clean_no_op(self, tmp_path, monkeypatch):
        import scripts.migrate_wiki_to_topics as mod

        monkeypatch.setattr("sys.argv", ["x", "--vault", str(tmp_path)])
        assert asyncio.run(mod.main()) == 0
