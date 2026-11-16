"""Direct unit tests for strategy_synthesizer/graph_patcher.py and prompts.py."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from extensions.strategy_synthesizer.graph_patcher import patch_playbook_path_by_memory_ids
from extensions.strategy_synthesizer.prompts import PLAYBOOK_MERGE_PROMPT, render_memories_block


# ── graph_patcher ───────────────────────────────────────────────────


class _Record(dict):
    pass


class _Result:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _Session:
    def __init__(self, log, record):
        self._log, self._record = log, record

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **params):
        self._log.append((cypher, params))
        return _Result(self._record)


def _service(record=None, log=None, raises=None):
    log = log if log is not None else []

    async def run_on_bridge(coro, timeout=30.0):
        if raises:
            coro.close()
            raise raises
        return await coro

    driver = SimpleNamespace(session=lambda: _Session(log, record))
    return SimpleNamespace(_graphiti=SimpleNamespace(driver=driver), _run_on_bridge_async=run_on_bridge), log


def _run(coro):
    return asyncio.run(coro)


class TestGraphPatcher:
    def test_stamps_every_node_for_the_memory_ids_and_returns_the_count(self):
        svc, log = _service(record={"patched": 3})
        when = datetime(2026, 7, 2, tzinfo=timezone.utc)
        n = _run(patch_playbook_path_by_memory_ids(
            svc, memory_ids=["m1", "", "m2"], playbook_path="Playbooks/alice/s.md", synthesized_at=when))
        assert n == 3
        cypher, params = log[0]
        assert "n.memory_id IN $mids" in cypher and "strategy_playbook_path" in cypher
        # both back-reference properties are SET (pins the clause, not just the parameters)
        assert "n.strategy_playbook_path = $playbook_path" in cypher
        assert "n.strategy_synthesized_at = $synthesized_at" in cypher
        assert params["mids"] == ["m1", "m2"]  # blanks dropped
        assert params["playbook_path"] == "Playbooks/alice/s.md"
        assert params["synthesized_at"] == when.isoformat()

    def test_the_timestamp_defaults_to_now_in_utc(self):
        svc, log = _service(record={"patched": 1})
        _run(patch_playbook_path_by_memory_ids(svc, memory_ids=["m1"], playbook_path="p.md"))
        stamp = datetime.fromisoformat(log[0][1]["synthesized_at"])
        assert stamp.tzinfo is not None and abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 60

    def test_no_ids_or_no_path_is_a_no_op(self):
        svc, log = _service(record={"patched": 1})
        assert _run(patch_playbook_path_by_memory_ids(svc, memory_ids=[], playbook_path="p.md")) == 0
        assert _run(patch_playbook_path_by_memory_ids(svc, memory_ids=["m1"], playbook_path="")) == 0
        assert log == []

    def test_a_service_without_a_graph_is_a_no_op(self):
        assert _run(patch_playbook_path_by_memory_ids(
            SimpleNamespace(_graphiti=None), memory_ids=["m1"], playbook_path="p.md")) == 0
        assert _run(patch_playbook_path_by_memory_ids(
            SimpleNamespace(_graphiti=SimpleNamespace(driver=None)), memory_ids=["m1"], playbook_path="p.md")) == 0

    def test_an_empty_result_is_zero(self):
        svc, _ = _service(record=None)
        assert _run(patch_playbook_path_by_memory_ids(svc, memory_ids=["m1"], playbook_path="p.md")) == 0

    def test_a_bridge_failure_is_swallowed_never_raised(self):
        svc, _ = _service(raises=RuntimeError("neo4j down"))
        assert _run(patch_playbook_path_by_memory_ids(svc, memory_ids=["m1"], playbook_path="p.md")) == 0

    def test_a_session_failure_is_swallowed_never_raised(self):
        svc, _ = _service(record={"patched": 1})

        class Boom:
            async def __aenter__(self):
                raise ConnectionError("no session")

            async def __aexit__(self, *a):
                return False

        svc._graphiti.driver.session = lambda: Boom()
        assert _run(patch_playbook_path_by_memory_ids(svc, memory_ids=["m1"], playbook_path="p.md")) == 0


# ── prompts ─────────────────────────────────────────────────────────


class TestPrompts:
    def test_block_numbers_memories_and_carries_metadata(self):
        block = render_memories_block([
            {"content": "Enter on the close above the zone", "category": "entry_rule", "confidence": 0.9,
             "created_at": "2026-07-01"},
            {"content": "Stop 5 pips below the tail", "category": "stop_rule"},
        ])
        lines = block.splitlines()
        assert lines[0] == "1. Enter on the close above the zone [category=entry_rule, confidence=0.9, created_at=2026-07-01]"
        assert lines[1] == "2. Stop 5 pips below the tail [category=stop_rule]"

    def test_blank_content_is_skipped_and_numbering_stays_dense(self):
        block = render_memories_block([{"content": "  "}, {"content": "real rule"}, {"content": ""}])
        assert block == "1. real rule"

    def test_empty_input_has_a_placeholder(self):
        assert render_memories_block([]) == "(no memories)"
        assert render_memories_block([{"content": ""}]) == "(no memories)"

    def test_missing_metadata_leaves_no_brackets(self):
        assert "[" not in render_memories_block([{"content": "bare rule", "confidence": None, "category": ""}])

    def test_merge_prompt_formats_with_all_three_fields_and_keeps_the_contract(self):
        text = PLAYBOOK_MERGE_PROMPT.format(
            strategy_name="naked-forex", existing_body="(empty)", memories_block="1. rule")
        assert "STRATEGY: naked-forex" in text and "1. rule" in text
        for heading in ("## Thesis", "## Entry", "## Stop", "## Version Updates"):
            assert heading in text
        assert "CITATION ALIGNMENT" in text and "Do NOT invent numbers" in text
        assert "{" not in text.replace("{strategy_name}", "")  # no unfilled placeholders
