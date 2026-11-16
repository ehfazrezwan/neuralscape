"""Direct tests for small extension/adapter leaf modules: conversation_compiler config/schemas,
dreaming prompts + manifest, strategy_synthesizer renderer/extension + manifest, trading adapter package."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from extensions.base import ExtensionManifest

EXT = Path(__file__).resolve().parent.parent / "extensions"


# ── manifests ──


@pytest.mark.parametrize("ext", ["dreaming", "strategy_synthesizer", "conversation_compiler"])
def test_manifest_loads_and_validates(ext):
    data = json.loads((EXT / ext / "manifest.json").read_text(encoding="utf-8"))
    m = ExtensionManifest(**data)
    assert m.name and m.version.count(".") == 2 and m.description
    assert isinstance(m.hooks, list)


def test_dreaming_manifest_exact_values():
    m = ExtensionManifest(**json.loads((EXT / "dreaming" / "manifest.json").read_text(encoding="utf-8")))
    assert (m.name, m.version, m.author) == ("dreaming", "0.2.0", "Neuralscape")
    assert m.description.startswith("Background memory consolidation:") and "humane vault" in m.description


def test_cron_driven_extensions_declare_no_event_hooks():
    for ext in ("dreaming", "strategy_synthesizer"):
        assert ExtensionManifest(**json.loads((EXT / ext / "manifest.json").read_text())).hooks == []


def test_manifest_name_rules_reject_bad_names():
    with pytest.raises(ValidationError):
        ExtensionManifest(name="Bad_Name", version="0.1.0", description="x")
    ExtensionManifest(name="ok-name-1", version="0.1.0", description="x")


def test_strategy_synthesizer_extension_exposes_its_manifest():
    from extensions.strategy_synthesizer import StrategySynthesizerExtension

    assert StrategySynthesizerExtension.manifest.name == "strategy-synthesizer"
    ext = StrategySynthesizerExtension()
    assert ext.get_routes() is None


@pytest.mark.asyncio
async def test_strategy_synthesizer_ignores_events_and_reports_a_synthesis_pass(monkeypatch):
    from types import SimpleNamespace

    import extensions.strategy_synthesizer as mod

    ext = mod.StrategySynthesizerExtension()
    assert await ext.on_event("memory_stored", {"a": 1}) is None
    seen = {}

    async def fake_synthesize_all(**kw):
        seen.update(kw)
        pb = SimpleNamespace(strategy_name="S", owner="alice", playbook_path="Playbooks/alice/s.md", created=True, source_memory_count=3)
        return SimpleNamespace(playbooks_created=1, playbooks_updated=0, playbooks_skipped_unchanged=0,
                               memories_processed=3, errors=[], playbooks=[pb])

    monkeypatch.setattr(mod, "synthesize_all", fake_synthesize_all)
    ext._service = object()
    out = await ext.run_synthesis(only_strategy="S", dry_run=True, force=True)
    assert (seen["only_strategy"], seen["dry_run"], seen["force"]) == ("S", True, True)
    assert out["playbooks"] == [{"strategy_name": "S", "owner": "alice", "playbook_path": "Playbooks/alice/s.md",
                                 "created": True, "source_memory_count": 3}]
    assert out["playbooks_created"] == 1 and out["memories_processed"] == 3


# ── conversation_compiler config ──


class TestCompilerSettings:
    def test_defaults(self):
        from extensions.conversation_compiler.config import CompilerSettings

        s = CompilerSettings(_env_file=None)
        assert (s.obsidian_vault_path, s.compile_after_hour, s.auto_compile) == ("/data/vault", 18, True)
        assert s.compiler_llm_model == ""
        assert s.get_llm_model("fallback-model") == "fallback-model"

    def test_env_overrides_and_configured_model_wins(self, monkeypatch):
        from extensions.conversation_compiler.config import CompilerSettings

        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/tmp/some-vault")
        monkeypatch.setenv("COMPILER_LLM_MODEL", "custom")
        monkeypatch.setenv("COMPILE_AFTER_HOUR", "6")
        s = CompilerSettings(_env_file=None)
        assert s.compile_after_hour == 6 and s.get_llm_model("fallback") == "custom"
        assert s.vault_path == Path("/tmp/some-vault").resolve()

    @pytest.mark.parametrize("hour", [-1, 24])
    def test_hour_bounds(self, monkeypatch, hour):
        from extensions.conversation_compiler.config import CompilerSettings

        monkeypatch.setenv("COMPILE_AFTER_HOUR", str(hour))
        with pytest.raises(ValidationError):
            CompilerSettings(_env_file=None)

    def test_vault_path_expands_the_home_directory(self, monkeypatch):
        from extensions.conversation_compiler.config import CompilerSettings

        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "~/vault-x")
        assert CompilerSettings(_env_file=None).vault_path == (Path.home() / "vault-x").resolve()

    def test_unknown_env_is_ignored(self, monkeypatch):
        from extensions.conversation_compiler.config import CompilerSettings

        monkeypatch.setenv("SOMETHING_ELSE", "1")
        settings = CompilerSettings(_env_file=None, not_a_field="x")      # an unknown field through the real input path
        assert not hasattr(settings, "not_a_field") and settings.auto_compile is True


# ── conversation_compiler schemas ──


class TestCompilerSchemas:
    def test_flush_request_limits(self):
        from extensions.conversation_compiler.schemas import FlushRequest

        ok = FlushRequest(user_message="hi", assistant_response="yo", session_id="s", user_id="alice")
        assert ok.channel == "api" and ok.project_id is None
        for bad in ({"user_message": ""}, {"assistant_response": ""}, {"session_id": ""}, {"user_id": ""},
                    {"user_message": "x" * 50001}, {"channel": "c" * 51}, {"user_id": "u" * 101}, {"session_id": "s" * 201}):
            with pytest.raises(ValidationError):
                FlushRequest(**{"user_message": "hi", "assistant_response": "yo", "session_id": "s", "user_id": "alice", **bad})

    def test_compile_query_lint_requests(self):
        from extensions.conversation_compiler.schemas import CompileRequest, LintRequest, QueryRequest

        assert CompileRequest(user_id="a").date is None
        with pytest.raises(ValidationError):
            CompileRequest(user_id="")
        q = QueryRequest(question="why?", user_id="a")
        assert q.file_back is False
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 5001, user_id="a")
        assert LintRequest().structural_only is False

    def test_result_models_default_to_empty(self):
        from extensions.conversation_compiler.schemas import (
            CompileResult, FlushResult, LintResult, QueryResult, StatusResponse, ExtractedFact, LintFinding, CompiledArticle,
        )

        f = FlushResult(session_id="s", timestamp="t")
        assert (f.facts_extracted, f.facts, f.category_paths, f.memories_stored, f.daily_log_path) == (0, [], [], 0, None)
        c = CompileResult(date="2026-01-01")
        assert (c.articles, c.entries_compiled, c.dedup_triggered, c.dedup_graph_jobs_enqueued) == ([], 0, False, 0)
        assert LintResult().findings == [] and QueryResult(answer="a").sources == []
        s = StatusResponse()
        assert (s.extension, s.status, s.article_count) == ("conversation-compiler", "ok", 0)
        assert ExtractedFact(category="decision", content="x").tags == []
        assert LintFinding(check="c", message="m").severity == "warning"
        assert CompiledArticle(path="p", title="t", article_type="session").created is False

    def test_list_defaults_are_not_shared_between_instances(self):
        from extensions.conversation_compiler.schemas import FlushResult

        a, b = FlushResult(session_id="a", timestamp="t"), FlushResult(session_id="b", timestamp="t")
        a.category_paths.append("x")
        assert b.category_paths == []


# ── dreaming prompts ──


class TestDreamingPrompts:
    def test_render_memories_block(self):
        from extensions.dreaming.prompts import render_memories_block

        mems = [
            {"memory_id": "m1", "created_at": "2026-05-06T10:00:00Z", "category": "decision", "visibility": "private",
             "retention_strength": 0.5, "content": "line one\nline two"},
            {"memory_id": "m2", "content": "   "},                       # blank: skipped
            {"memory_id": "m3", "content": "x"},                          # all optional fields missing
        ]
        rows = render_memories_block(mems).splitlines()
        assert rows[0] == "m1 | 2026-05-06 | decision | private | 0.50 | line one line two"
        assert rows[1] == "m3 | ? | ? | ? | ? | x" and len(rows) == 2
        assert render_memories_block(mems, include_strength=False).splitlines()[0] == "m1 | 2026-05-06 | decision | line one line two"
        assert render_memories_block([]) == "(no memories)"
        assert render_memories_block([{"memory_id": "m", "content": ""}]) == "(no memories)"

    def test_prompts_format_with_their_placeholders_and_keep_the_json_schema(self):
        from extensions.dreaming.prompts import CONSOLIDATION_PROMPT, REFLECTION_PROMPT

        c = CONSOLIDATION_PROMPT.format(today="2026-05-06", year=2026, memories_block="BLOCK")
        assert "2026-05-06" in c and "BLOCK" in c and '{"actions"' in c
        for action in ("merge", "invalidate", "prune", "rewrite", "temporal_reframe"):
            assert f'"{action}"' in c
        r = REFLECTION_PROMPT.format(today="2026-05-06", categories="decision, convention", max_insights=3, memories_block="BLOCK")
        assert "decision, convention" in r and "At most 3 insights" in r and '{"insights"' in r

    @pytest.mark.parametrize("raw", [
        '{"actions": [{"type": "merge"}]}',
        '```json\n{"actions": [{"type": "merge"}]}\n```',
        'Sure! Here you go:\n{"actions": [{"type": "merge"}]}\nHope that helps.',
        '```\n{"actions": [{"type": "merge"}]}\n```',
    ])
    def test_json_extraction_tolerates_fences_and_prose(self, raw):
        from extensions.dreaming.prompts import parse_json_response

        assert parse_json_response(raw, key="actions") == [{"type": "merge"}]

    @pytest.mark.parametrize("raw", [None, "", "no json here", "{not json}", "[1,2,3]", '{"actions": "oops"}', '{"other": []}'])
    def test_garbage_yields_empty_never_raises(self, raw):
        from extensions.dreaming.prompts import parse_json_object, parse_json_response

        assert parse_json_response(raw, key="actions") == []
        expected = {'{"actions": "oops"}': {"actions": "oops"}, '{"other": []}': {"other": []}}.get(raw, {})
        assert parse_json_object(raw) == expected

    def test_non_dict_items_are_dropped(self):
        from extensions.dreaming.prompts import parse_json_response

        assert parse_json_response('{"actions": [{"a": 1}, "x", 3, null, {"b": 2}]}', key="actions") == [{"a": 1}, {"b": 2}]

    def test_parse_json_object(self):
        from extensions.dreaming.prompts import parse_json_object

        assert parse_json_object('text {"k": 1} text') == {"k": 1}
        assert parse_json_object("[1]") == {}


# ── strategy_synthesizer playbook renderer ──


class TestPlaybookRenderer:
    def test_page_path_is_keyed_by_owner_and_strategy(self, tmp_path):
        from extensions.strategy_synthesizer.playbook_renderer import playbook_page_path

        assert playbook_page_path(tmp_path, "Alice", "Opening Range Breakout") == tmp_path / "alice" / "opening-range-breakout.md"
        assert playbook_page_path(tmp_path, None, "S1") == tmp_path / "shared" / "s1.md"
        assert playbook_page_path(tmp_path, "###", "S1") == tmp_path / "shared" / "s1.md"   # unusable owner slug
        assert playbook_page_path(tmp_path, "alice", "!!!") is None                        # unusable strategy slug

    def test_rel_path_falls_back_to_shared_for_an_empty_or_unusable_owner(self):
        from extensions.strategy_synthesizer.playbook_renderer import playbook_rel_path

        assert playbook_rel_path("", "S1") == "Playbooks/shared/s1.md"
        assert playbook_rel_path("###", "S1") == "Playbooks/shared/s1.md"

    def test_rel_path_matches_the_absolute_layout(self, tmp_path):
        from extensions.strategy_synthesizer.playbook_renderer import playbook_page_path, playbook_rel_path

        rel = playbook_rel_path("alice", "Gap Fill")
        assert rel == "Playbooks/alice/gap-fill.md"
        assert playbook_page_path(tmp_path / "Playbooks", "alice", "Gap Fill") == tmp_path / rel
        assert playbook_rel_path(None, "Gap Fill") == "Playbooks/shared/gap-fill.md"
        assert playbook_rel_path("alice", "???") is None

    def test_render_then_split_round_trips_the_frontmatter(self):
        from extensions.strategy_synthesizer.playbook_renderer import render_page, split_existing_page

        now = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
        page = render_page(strategy_name="Gap Fill", owner="alice", body="\n  Entry: gap > 2%.  \n", source_memory_ids=["m1", "m2"],
                           version_number=3, source_count=7, now=now)
        fm, body = split_existing_page(page)
        assert fm["kind"] == "strategy_playbook" and fm["strategy_name"] == "Gap Fill" and fm["owner"] == "alice"
        assert fm["version_number"] == "3" and fm["source_count"] == "7" and fm["source_memory_ids"] == "[m1, m2]"
        assert fm["title"] == "Gap Fill"
        assert fm["last_synthesized"] == now.isoformat()
        assert body.startswith("# Gap Fill — Playbook") and body.rstrip().endswith("Entry: gap > 2%.")
        # the body is stripped: no leading/trailing whitespace around it in the page
        assert "# Gap Fill — Playbook\n\nEntry: gap > 2%.\n" in page and "  Entry" not in page and "2%.  " not in page

    def test_owner_defaults_to_shared_and_empty_ids_render_as_an_empty_list(self):
        from extensions.strategy_synthesizer.playbook_renderer import render_page, split_existing_page

        fm, _ = split_existing_page(render_page(strategy_name="S", owner=None, body="b", source_memory_ids=[], version_number=1, source_count=0))
        assert fm["owner"] == "shared" and fm["source_memory_ids"] == "[]"

    def test_split_handles_missing_frontmatter_and_empty_input(self):
        from extensions.strategy_synthesizer.playbook_renderer import split_existing_page

        assert split_existing_page("") == ({}, "")
        assert split_existing_page("# just a body\n") == ({}, "# just a body\n")
        fm, body = split_existing_page("---\nbad line\nk: v: with colon\n---\n\nbody\n")
        assert fm == {"k": "v: with colon"} and body == "body\n"

    def test_write_page_is_atomic_and_overwrites(self, tmp_path):
        from extensions.strategy_synthesizer.playbook_renderer import write_page

        target = tmp_path / "Playbooks" / "alice" / "s.md"
        write_page(target, "one")
        write_page(target, "two")
        assert target.read_text() == "two" and not list(target.parent.glob("*.tmp*"))

    def test_write_page_goes_through_a_temp_file_and_a_rename(self, tmp_path, monkeypatch):
        import extensions.conversation_compiler.obsidian_writer as ow
        from extensions.strategy_synthesizer.playbook_renderer import write_page

        target = tmp_path / "Playbooks" / "alice" / "s.md"
        write_page(target, "old")
        renames = []
        real = ow.os.rename

        def spy(src, dst):
            renames.append((Path(src), Path(dst), target.read_text()))
            return real(src, dst)

        monkeypatch.setattr(ow.os, "rename", spy)
        write_page(target, "new")
        (src, dst, seen), = renames
        assert dst == target and src.parent == target.parent and src.suffix == ".tmp" and src != target
        assert seen == "old"                       # the target still held the old page until the rename

    def test_an_interrupted_write_leaves_the_old_page_and_no_temp_file(self, tmp_path, monkeypatch):
        import extensions.conversation_compiler.obsidian_writer as ow
        from extensions.strategy_synthesizer.playbook_renderer import write_page

        target = tmp_path / "Playbooks" / "alice" / "s.md"
        write_page(target, "old")

        def boom(src, dst):
            raise OSError("disk gone")

        monkeypatch.setattr(ow.os, "rename", boom)
        with pytest.raises(OSError):
            write_page(target, "new")
        assert target.read_text() == "old" and not list(target.parent.glob("*.tmp"))


# ── adapters.trading package ──


def test_trading_package_exports_and_registers_the_adapter():
    import adapters.trading as trading
    from adapters.base import ADAPTER_REGISTRY, get_adapter
    from schemas import MEMORY_CATEGORIES

    assert set(trading.__all__) == {"ADAPTER_NAME", "TRADING_ADAPTER", "TRADING_CATEGORIES"}
    assert trading.ADAPTER_NAME == "trading_strategy"
    assert ADAPTER_REGISTRY[trading.ADAPTER_NAME] is trading.TRADING_ADAPTER is get_adapter("trading_strategy")
    assert trading.TRADING_CATEGORIES and set(trading.TRADING_CATEGORIES) <= set(MEMORY_CATEGORIES)
