"""Tests for the reasoning-tiered ask path (roadmap C3).

Covers the tier matrix (each level's retrieval breadth / iteration cap /
output cap / mechanical passes), the dialectic disciplines (grep-first for
enumeration, forced update-language pass, contradiction surfacing in the
prompt contract), strict abstention (no evidence → "don't know" with zero
LLM calls and no fabricated ids), citation validation, and the follow-up
search loop budget.
"""

import json
from unittest.mock import MagicMock

import pytest

import ask as ask_mod
from ask import (
    REASONING_TIERS,
    AskUnavailable,
    ask_memory,
    extract_keywords,
    is_enumeration_question,
)
from schemas import MemoryResponse


def _mem(mid: str, content: str, created_at: str = "2026-07-01T00:00:00+00:00") -> MemoryResponse:
    return MemoryResponse(
        id=mid, memory=content, category="decision", source="vector",
        created_at=created_at, score=0.9,
    )


def _graph_mem(mid: str, content: str) -> MemoryResponse:
    """A graph-sourced row: no timestamp, no score (Graphiti edge shape)."""
    return MemoryResponse(
        id=mid, memory=content, category="relation", source="graph",
        created_at=None, score=None,
    )


def _service(search_results=None, keyword_results=None, keyword_capped=False) -> MagicMock:
    svc = MagicMock(name="MemoryService")
    svc.search.return_value = search_results if search_results is not None else []
    svc.keyword_search.return_value = (
        keyword_results if keyword_results is not None else [],
        keyword_capped,
    )
    return svc


def _answer_llm(answer: str, citations=None, abstained=False):
    """Fake LLM that always returns a final answer, recording prompts."""
    prompts: list[str] = []

    async def call(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({
            "action": "answer", "answer": answer,
            "citations": citations or [], "abstained": abstained,
        })

    call.prompts = prompts
    return call


# ──────────────────────────────────────────────
# Tier matrix
# ──────────────────────────────────────────────


class TestTierMatrix:
    def test_all_levels_defined(self):
        assert set(REASONING_TIERS) == {"minimal", "low", "medium", "high"}

    def test_retrieval_breadth_monotonic(self):
        limits = [REASONING_TIERS[l].search_limit for l in ("minimal", "low", "medium", "high")]
        assert limits == sorted(limits) and limits[0] < limits[-1]

    def test_iteration_cap_monotonic_and_minimal_has_no_loop(self):
        extras = [REASONING_TIERS[l].extra_searches for l in ("minimal", "low", "medium", "high")]
        assert extras == sorted(extras)
        assert REASONING_TIERS["minimal"].extra_searches == 0
        assert REASONING_TIERS["high"].extra_searches >= 3

    def test_output_cap_monotonic(self):
        caps = [REASONING_TIERS[l].max_answer_words for l in ("minimal", "low", "medium", "high")]
        assert caps == sorted(caps) and caps[0] < caps[-1]

    def test_mechanical_passes_by_tier(self):
        assert not REASONING_TIERS["minimal"].keyword_pass
        assert not REASONING_TIERS["minimal"].update_pass
        assert REASONING_TIERS["low"].update_pass and not REASONING_TIERS["low"].keyword_pass
        assert REASONING_TIERS["medium"].keyword_pass
        assert REASONING_TIERS["high"].keyword_pass and REASONING_TIERS["high"].update_pass

    def test_timeouts_come_from_settings(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "ask_timeout_high_s", 7)
        assert REASONING_TIERS["high"].llm_timeout_s == 7

    @pytest.mark.asyncio
    async def test_unknown_level_rejected(self):
        with pytest.raises(ValueError, match="reasoning_level"):
            await ask_memory(_service(), question="q", user_id="u", reasoning_level="ultra")


# ──────────────────────────────────────────────
# Retrieval passes per tier
# ──────────────────────────────────────────────


class TestRetrievalPasses:
    @pytest.mark.asyncio
    async def test_minimal_is_single_search_no_loop(self):
        svc = _service([_mem("m1", "The sync is on Tuesday.")])
        llm = _answer_llm("Tuesday [m1]", ["m1"])
        out = await ask_memory(svc, question="When is the sync?", user_id="u",
                               reasoning_level="minimal", llm_call=llm)
        # Exactly one semantic search, no keyword pass, one LLM call.
        assert svc.search.call_count == 1
        svc.keyword_search.assert_not_called()
        assert len(llm.prompts) == 1
        assert out["searches"] == ["When is the sync?"]
        assert svc.search.call_args.kwargs["limit"] == REASONING_TIERS["minimal"].search_limit

    @pytest.mark.asyncio
    async def test_low_adds_update_language_pass(self):
        """Temporal question at low tier → the update-language recency pass
        runs (atemporal questions skip it — see TestUpdatePassGating)."""
        svc = _service([_mem("m1", "fact")])
        llm = _answer_llm("ans", ["m1"])
        out = await ask_memory(svc, question="Was the sync rescheduled?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert svc.search.call_count == 2
        update_query = svc.search.call_args_list[1].kwargs["query"]
        for term in ("changed", "rescheduled", "now"):
            assert term in update_query
        assert len(out["searches"]) == 2

    @pytest.mark.asyncio
    async def test_high_runs_keyword_pass_before_semantic(self):
        """Discipline 1: the grep-style exact pass runs (and is recorded)
        before the semantic passes at keyword-enabled tiers."""
        svc = _service([_mem("m1", "apple")], [_mem("k1", "banana list entry")])
        llm = _answer_llm("ans")
        out = await ask_memory(svc, question="List all fruits I mentioned",
                               user_id="u", reasoning_level="high", llm_call=llm)
        svc.keyword_search.assert_called_once()
        assert out["searches"][0].startswith("keyword:")
        assert out["memories_considered"] == 2  # union of both passes, dedup'd

    @pytest.mark.asyncio
    async def test_search_breadth_follows_tier(self):
        svc = _service([_mem("m1", "x")])
        await ask_memory(svc, question="q?", user_id="u",
                         reasoning_level="high", llm_call=_answer_llm("a"))
        assert svc.search.call_args.kwargs["limit"] == REASONING_TIERS["high"].search_limit

    @pytest.mark.asyncio
    async def test_project_id_forwarded(self):
        svc = _service([_mem("m1", "x")])
        await ask_memory(svc, question="q?", user_id="u", project_id="proj",
                         reasoning_level="minimal", llm_call=_answer_llm("a"))
        assert svc.search.call_args.kwargs["project_id"] == "proj"


# ──────────────────────────────────────────────
# Follow-up search loop (tool budget / iteration cap)
# ──────────────────────────────────────────────


class TestSearchLoop:
    @pytest.mark.asyncio
    async def test_llm_directed_searches_capped_at_tier_budget(self):
        """An LLM that keeps asking for more searches is cut off at the
        tier's iteration cap, then forced to answer."""
        svc = _service([_mem("m1", "x")])
        calls = {"n": 0}

        async def greedy_llm(prompt: str) -> str:
            calls["n"] += 1
            return json.dumps({"action": "search", "query": f"follow-up {calls['n']}"})

        # Temporal question so the (gated) update pass still runs.
        out = await ask_memory(svc, question="What is the latest plan?", user_id="u",
                               reasoning_level="high", llm_call=greedy_llm)
        tier = REASONING_TIERS["high"]
        # Semantic passes: initial + update + at most `extra_searches` follow-ups.
        assert svc.search.call_count == 2 + tier.extra_searches
        # LLM calls: one per follow-up + the final forced-answer pass.
        assert calls["n"] <= tier.extra_searches + 2
        # Greedy model never answered — fallback shape, but never a crash.
        assert out["status"] == "ok"

    @pytest.mark.asyncio
    async def test_minimal_never_loops_even_if_llm_asks(self):
        svc = _service([_mem("m1", "x")])
        prompts: list[str] = []
        responses = iter([
            json.dumps({"action": "search", "query": "more"}),
            json.dumps({"action": "answer", "answer": "done", "citations": []}),
        ])

        async def llm(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="minimal", llm_call=llm)
        # minimal's output contract never offers the search action.
        instructions = prompts[0].split("QUESTION:")[1]
        assert '"action": "search"' not in instructions
        assert '"action": "answer"' in instructions
        # The search request is not honored (budget 0) — one forced answer pass.
        assert svc.search.call_count == 1
        assert out["answer"] == "done"

    @pytest.mark.asyncio
    async def test_followup_search_results_merge_into_evidence(self):
        """A follow-up hit becomes citable evidence (low: initial + update
        + 1 LLM-directed follow-up)."""
        svc = _service()
        svc.search.side_effect = [
            [_mem("m1", "first")],                     # initial semantic
            [_mem("m1", "first")],                     # forced update pass
            [_mem("m1", "first"), _mem("m2", "second")],  # LLM follow-up
        ]
        responses = iter([
            json.dumps({"action": "search", "query": "narrower"}),
            json.dumps({"action": "answer", "answer": "both [m1] [m2]",
                        "citations": ["m1", "m2"]}),
        ])

        async def llm(prompt: str) -> str:
            return next(responses)

        # Temporal question so the (gated) update pass still runs.
        out = await ask_memory(svc, question="Where do we meet now?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert svc.search.call_count == 3
        assert svc.search.call_args_list[2].kwargs["query"] == "narrower"
        assert out["memories_considered"] == 2
        assert set(out["citations"]) == {"m1", "m2"}  # follow-up hit is citable


# ──────────────────────────────────────────────
# Evidence budget (audit 27 #15): over budget, keep by priority
# (keyword hits > score > newest), render survivors chronologically
# ──────────────────────────────────────────────


class TestEvidenceBudget:
    def _timestamped_rows(self, n: int):
        """n rows with strictly ascending timestamps (id order = age order)."""
        return [
            _mem(f"mem-{i:03d}", f"fact number {i:03d} stands alone",
                 f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00")
            for i in range(n)
        ]

    def test_budget_overflow_keeps_newest_rows(self):
        """130 ascending-timestamp rows → the NEWEST 120 survive the cut
        (pre-fix the ascending sort + head-truncation kept the OLDEST)."""
        rows = self._timestamped_rows(130)
        evidence = {m.id: m for m in rows}
        out = ask_mod._evidence_rows(evidence, [], False)
        ids = [m.id for m in out]
        assert len(ids) == ask_mod._EVIDENCE_MAX_ROWS
        assert "mem-129" in ids  # newest row survives
        assert "mem-000" not in ids  # oldest rows are the ones cut
        # Survivors re-sorted chronologically ascending for the prompt.
        assert ids == [f"mem-{i:03d}" for i in range(10, 130)]

    @pytest.mark.asyncio
    async def test_budget_overflow_newest_rows_reach_the_prompt(self):
        rows = self._timestamped_rows(130)
        svc = _service(rows)
        llm = _answer_llm("ans", ["mem-129"])
        await ask_memory(svc, question="q?", user_id="u",
                         reasoning_level="minimal", llm_call=llm)
        assert "[mem-129]" in llm.prompts[0]
        assert "[mem-000]" not in llm.prompts[0]

    def test_timestampless_graph_rows_do_not_eat_the_budget(self):
        """Over budget, unscored timestamp-less graph rows are lowest
        priority (pre-fix their '' sort key put them FIRST in the keep-set)."""
        rows = self._timestamped_rows(120)
        graph = [_graph_mem(f"g-{i}", f"relation {i}") for i in range(5)]
        evidence = {m.id: m for m in graph + rows}  # graph rows arrive first
        out = ask_mod._evidence_rows(evidence, [], False)
        ids = [m.id for m in out]
        assert len(ids) == ask_mod._EVIDENCE_MAX_ROWS
        assert not [i for i in ids if i.startswith("g-")]
        assert "mem-119" in ids

    def test_timestampless_rows_render_last_when_under_budget(self):
        rows = self._timestamped_rows(3)
        graph = [_graph_mem("g-1", "relation one")]
        evidence = {m.id: m for m in graph + rows}
        ids = [m.id for m in ask_mod._evidence_rows(evidence, [], False)]
        assert ids == ["mem-000", "mem-001", "mem-002", "g-1"]

    def test_keyword_hits_survive_the_cut_even_when_old(self):
        rows = self._timestamped_rows(130)
        evidence = {m.id: m for m in rows}
        out = ask_mod._evidence_rows(evidence, ["mem-000"], False)
        ids = [m.id for m in out]
        assert "mem-000" in ids  # keyword hit outranks recency
        assert "mem-010" not in ids  # the extra slot comes from the oldest non-kw row
        # Chronological rendering puts the surviving old kw hit first.
        assert ids[0] == "mem-000"


# ──────────────────────────────────────────────
# Evidence clipping (audit 27 #16): passage rows get a 1500-char budget,
# every row clips at a sentence boundary instead of mid-word
# ──────────────────────────────────────────────


def _passage(mid: str, content: str) -> MemoryResponse:
    return MemoryResponse(
        id=mid, memory=content, category="domain_knowledge", source="vector",
        created_at="2026-07-01T00:00:00+00:00", score=0.9, memory_kind="passage",
    )


class TestEvidenceClip:
    def test_passage_row_keeps_content_beyond_500_chars(self):
        """Ingest passages default to 1500 chars; a fact-sized 500-char clip
        hid two-thirds of every passage from the answerer."""
        filler = "This is a filler sentence to pad out the passage body. "  # 56 chars
        text = filler * 13 + "The vault access code is 9137. " + filler * 8
        assert 700 < text.index("9137") < 800  # the answer sits past the old clip
        rendered = ask_mod._render_evidence([_passage("p1", text)])
        assert "The vault access code is 9137." in rendered

    def test_fact_row_keeps_500_budget(self):
        filler = "This is a filler sentence to pad out the fact body zzz. "  # 57 chars
        text = filler * 13 + "The vault access code is 9137."
        row = _mem("f1", text)
        rendered = ask_mod._render_evidence([row])
        assert "9137" not in rendered  # fact rows keep the tight budget
        assert "…" in rendered

    def test_fact_clip_lands_on_sentence_boundary(self):
        # Sentences sized so char 500 falls mid-word: 57-char sentences →
        # boundary at 456; the old code sliced blindly at 500.
        filler = "This is a filler sentence to pad out the fact body zzz. "
        text = (filler * 12).strip()
        rendered = ask_mod._render_evidence([_mem("f1", text)])
        assert rendered.rstrip().endswith("zzz. …")

    def test_passage_over_budget_clips_at_sentence_boundary(self):
        filler = "This is a filler sentence to pad out the passage body. "
        text = (filler * 32).strip()  # ~1790 chars > 1500
        rendered = ask_mod._render_evidence([_passage("p1", text)])
        body = rendered.split(") ", 1)[1]
        assert len(body) <= ask_mod._EVIDENCE_PASSAGE_CLIP + 2  # "+ …"
        assert rendered.rstrip().endswith("body. …")

    def test_clip_falls_back_to_whitespace_when_no_sentence_boundary(self):
        text = "word " * 200  # no ". " anywhere, 1000 chars
        rendered = ask_mod._render_evidence([_mem("f1", text.strip())])
        assert "word  …" not in rendered  # no dangling partial token
        body = rendered.split(") ", 1)[1]
        clipped = body[: -len(" …")]
        assert clipped.endswith("word")  # cut on a whole word


# ──────────────────────────────────────────────
# Cross-source dedup (audit 27 #17): a Graphiti edge and its Qdrant twin
# arriving in different passes collapse to ONE evidence row (vector wins)
# ──────────────────────────────────────────────


class TestCrossSourceDedup:
    @pytest.mark.asyncio
    async def test_graph_then_vector_twin_collapses_to_vector_row(self):
        """Graph edge lands first, its vector twin in a later pass — the
        vector row (richer metadata) replaces the graph one."""
        graph_row = _graph_mem("graph-uuid-1", "alice moved to berlin")
        vector_row = _mem("vector-id-1", "Alice moved to Berlin.")
        svc = _service()
        svc.search.side_effect = [
            [graph_row],   # initial semantic pass
            [vector_row],  # update-language pass
        ]
        llm = _answer_llm("Berlin [vector-id-1]", ["vector-id-1"])
        out = await ask_memory(svc, question="Where does Alice live now?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert out["memories_considered"] == 1
        assert "[vector-id-1]" in llm.prompts[0]
        assert "[graph-uuid-1]" not in llm.prompts[0]
        assert out["citations"] == ["vector-id-1"]

    @pytest.mark.asyncio
    async def test_vector_then_graph_twin_drops_the_graph_row(self):
        vector_row = _mem("vector-id-1", "Alice moved to Berlin.")
        graph_row = _graph_mem("graph-uuid-1", "alice moved to berlin")
        svc = _service()
        svc.search.side_effect = [
            [vector_row],
            [graph_row],
        ]
        llm = _answer_llm("Berlin [vector-id-1]", ["vector-id-1"])
        out = await ask_memory(svc, question="Where does Alice live now?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert out["memories_considered"] == 1
        assert "[vector-id-1]" in llm.prompts[0]
        assert "[graph-uuid-1]" not in llm.prompts[0]

    @pytest.mark.asyncio
    async def test_distinct_rows_are_not_collapsed(self):
        svc = _service()
        svc.search.side_effect = [
            [_mem("m1", "Alice moved to Berlin.")],
            [_mem("m2", "Bob works at the bakery.")],
        ]
        llm = _answer_llm("ans", ["m1", "m2"])
        out = await ask_memory(svc, question="Where does Alice live now?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert out["memories_considered"] == 2


# ──────────────────────────────────────────────
# Keyword-pass prompt honesty (audit 27 #18): a capped (partial) lexical
# scan must not be endorsed as exact/exhaustive to the answering LLM
# ──────────────────────────────────────────────


class TestKeywordPromptHonesty:
    @pytest.mark.asyncio
    async def test_capped_scan_softens_prompt_language(self):
        svc = _service([_mem("m1", "apple")], [_mem("k1", "banana")],
                       keyword_capped=True)
        llm = _answer_llm("2")
        await ask_memory(svc, question="How many fruits did I mention?",
                         user_id="u", reasoning_level="high", llm_call=llm)
        prompt = llm.prompts[0]
        assert "partial" in prompt.lower()
        assert "Exact-keyword matches are listed first" not in prompt

    @pytest.mark.asyncio
    async def test_full_scan_keeps_exact_match_claim(self):
        svc = _service([_mem("m1", "apple")], [_mem("k1", "banana")],
                       keyword_capped=False)
        llm = _answer_llm("2")
        await ask_memory(svc, question="How many fruits did I mention?",
                         user_id="u", reasoning_level="high", llm_call=llm)
        prompt = llm.prompts[0]
        assert "Exact-keyword matches are listed first" in prompt
        assert "partial" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_legacy_list_return_still_accepted(self):
        """A test double (or older fork) returning a bare list instead of
        (rows, capped) degrades to capped=False, not a crash."""
        svc = _service([_mem("m1", "apple")])
        svc.keyword_search.return_value = [_mem("k1", "banana")]
        llm = _answer_llm("2")
        out = await ask_memory(svc, question="How many fruits did I mention?",
                               user_id="u", reasoning_level="high", llm_call=llm)
        assert out["memories_considered"] == 2
        assert "Exact-keyword matches are listed first" in llm.prompts[0]


# ──────────────────────────────────────────────
# Update-pass gating (audit 27 #19): the forced update-language search
# only runs when temporal cues appear in the question or first-pass evidence
# ──────────────────────────────────────────────


class TestUpdatePassGating:
    @pytest.mark.asyncio
    async def test_atemporal_question_skips_update_pass(self):
        svc = _service([_mem("m1", "Alice's favorite color is teal.")])
        llm = _answer_llm("teal [m1]", ["m1"])
        out = await ask_memory(svc, question="What is Alice's favorite color?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert svc.search.call_count == 1  # no update pass issued
        assert out["searches"] == ["What is Alice's favorite color?"]
        assert out["skipped_passes"] == ["update"]

    @pytest.mark.asyncio
    async def test_temporal_question_runs_update_pass(self):
        svc = _service([_mem("m1", "Alice lives in Berlin.")])
        llm = _answer_llm("Berlin [m1]", ["m1"])
        out = await ask_memory(svc, question="Where does Alice live now?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert svc.search.call_count == 2
        assert out["skipped_passes"] == []

    @pytest.mark.asyncio
    async def test_update_language_in_first_pass_evidence_triggers_pass(self):
        """Atemporal question, but the evidence itself mentions a change —
        the recency pass still runs."""
        svc = _service([_mem("m1", "The sync was rescheduled to Thursday.")])
        llm = _answer_llm("Thursday [m1]", ["m1"])
        out = await ask_memory(svc, question="What day is the sync?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        assert svc.search.call_count == 2
        assert out["skipped_passes"] == []

    def test_update_language_heuristic(self):
        yes = [
            "Where does Alice live now?",
            "When did we switch banks?",
            "Do I still go to that gym?",
            "What did the plan used to cost?",
            "Has the standup been rescheduled?",
            "What is the latest deploy target?",
            "Did the office move?",
            "Is she not at that job anymore?",
        ]
        no = [
            "What is Alice's favorite color?",
            "How many movies did I watch?",  # 'movies' must not match 'move'
            "Who owns the staging cluster?",
        ]
        for q in yes:
            assert ask_mod._has_update_language(q), q
        for q in no:
            assert not ask_mod._has_update_language(q), q


# ──────────────────────────────────────────────
# Abstention (dialectic discipline 4)
# ──────────────────────────────────────────────


class TestAbstention:
    @pytest.mark.asyncio
    async def test_no_evidence_abstains_without_llm_call(self):
        svc = _service([], [])
        called = {"n": 0}

        async def llm(prompt: str) -> str:
            called["n"] += 1
            return "should never run"

        out = await ask_memory(svc, question="What is my blood type?", user_id="u",
                               reasoning_level="high", llm_call=llm)
        assert out["abstained"] is True
        assert out["citations"] == []
        assert "don't know" in out["answer"].lower()
        assert out["memories_considered"] == 0
        assert called["n"] == 0  # no LLM call, no chance to fabricate

    @pytest.mark.asyncio
    async def test_model_abstention_flag_respected(self):
        svc = _service([_mem("m1", "irrelevant fact")])
        llm = _answer_llm("I don't know — nothing stored covers this.", [], abstained=True)
        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert out["abstained"] is True
        assert out["citations"] == []

    @pytest.mark.asyncio
    async def test_prompt_contains_abstention_discipline(self):
        svc = _service([_mem("m1", "x")])
        llm = _answer_llm("a")
        await ask_memory(svc, question="q?", user_id="u",
                         reasoning_level="high", llm_call=llm)
        prompt = llm.prompts[0]
        assert "I don't know" in prompt
        assert "NEVER fabricate" in prompt


# ──────────────────────────────────────────────
# Citations (no fabricated ids)
# ──────────────────────────────────────────────


class TestCitations:
    @pytest.mark.asyncio
    async def test_fabricated_citations_filtered(self):
        svc = _service([_mem("real-id", "the fact")])
        llm = _answer_llm("answer [real-id] [ghost-id]", ["real-id", "ghost-id"])
        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert out["citations"] == ["real-id"]

    @pytest.mark.asyncio
    async def test_unparseable_reply_recovers_citations_from_text(self):
        svc = _service([_mem("abc-123", "the fact")])

        async def llm(prompt: str) -> str:
            return "Plain prose answer citing [abc-123] without JSON."

        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="minimal", llm_call=llm)
        assert out["citations"] == ["abc-123"]
        assert "Plain prose answer" in out["answer"]
        assert out["abstained"] is False


# ──────────────────────────────────────────────
# Contradiction surfacing (dialectic discipline 3)
# ──────────────────────────────────────────────


class TestContradictions:
    @pytest.mark.asyncio
    async def test_evidence_carries_timestamps_and_discipline(self):
        """The prompt must give the model what discipline 3 needs: both
        conflicting rows WITH their timestamps + the surface-both rule."""
        old = _mem("m-old", "The sync is on Tuesday.", "2026-01-01T00:00:00+00:00")
        new = _mem("m-new", "The sync was rescheduled to Thursday.", "2026-06-01T00:00:00+00:00")
        svc = _service([new, old])
        llm = _answer_llm(
            "Thursday [m-new]; an older memory said Tuesday [m-old] — preferring the newer.",
            ["m-new", "m-old"],
        )
        out = await ask_memory(svc, question="When is the sync?", user_id="u",
                               reasoning_level="medium", llm_call=llm)
        prompt = llm.prompts[0]
        assert "2026-01-01" in prompt and "2026-06-01" in prompt
        assert "CONTRADICTIONS" in prompt and "BOTH" in prompt
        # Chronological order in the evidence block: older row first.
        assert prompt.index("m-old") < prompt.index("m-new")
        assert set(out["citations"]) == {"m-new", "m-old"}

    @pytest.mark.asyncio
    async def test_enumeration_dedup_instruction_present(self):
        svc = _service([_mem("m1", "apple")], [_mem("k1", "banana")])
        llm = _answer_llm("2 fruits")
        await ask_memory(svc, question="How many fruits did I mention?",
                         user_id="u", reasoning_level="high", llm_call=llm)
        prompt = llm.prompts[0]
        assert "dedup" in prompt.lower()
        assert "Never count raw rows" in prompt


# ──────────────────────────────────────────────
# Output cap + failure modes
# ──────────────────────────────────────────────


class TestOutputAndFailures:
    @pytest.mark.asyncio
    async def test_answer_clipped_to_tier_output_cap(self):
        svc = _service([_mem("m1", "x")])
        long_answer = "word " * 500
        llm = _answer_llm(long_answer.strip(), ["m1"])
        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="minimal", llm_call=llm)
        cap = REASONING_TIERS["minimal"].max_answer_words
        assert len(out["answer"].split()) <= cap + 1  # + ellipsis marker

    @pytest.mark.asyncio
    async def test_llm_unavailable_raises(self):
        svc = _service([_mem("m1", "x")])

        async def failing_call(prompt: str) -> str:
            raise AskUnavailable("Answering model unavailable")

        with pytest.raises(AskUnavailable):
            await ask_memory(svc, question="q?", user_id="u",
                             reasoning_level="minimal", llm_call=failing_call)

    @pytest.mark.asyncio
    async def test_empty_answer_becomes_abstention(self):
        svc = _service([_mem("m1", "x")])
        llm = _answer_llm("", ["m1"])
        out = await ask_memory(svc, question="q?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert out["abstained"] is True
        assert out["answer"]  # never empty


# ──────────────────────────────────────────────
# REST + MCP surfaces
# ──────────────────────────────────────────────


class TestAskSurfaces:
    def test_rest_route_returns_answer(self, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        async def fake_ask(service, **kwargs):
            assert kwargs["reasoning_level"] == "high"
            assert kwargs["user_id"] == "alice"
            return {
                "status": "ok", "reasoning_level": "high", "answer": "42 [m1]",
                "citations": ["m1"], "abstained": False,
                "searches": ["q"], "memories_considered": 3,
                "skipped_passes": ["update"],
            }

        monkeypatch.setattr(ask_mod, "ask_memory", fake_ask)
        client = TestClient(main.app, raise_server_exceptions=False)
        resp = client.post("/v1/ask", json={
            "question": "What is the answer?", "user_id": "alice",
            "reasoning_level": "high",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "42 [m1]" and body["citations"] == ["m1"]
        assert body["skipped_passes"] == ["update"]  # gating is visible in meta

    def test_rest_route_invalid_level_422(self):
        from fastapi.testclient import TestClient

        import main

        client = TestClient(main.app, raise_server_exceptions=False)
        resp = client.post("/v1/ask", json={
            "question": "q", "reasoning_level": "galaxy-brain",
        })
        assert resp.status_code == 422

    def test_rest_route_llm_unavailable_503(self, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        async def fake_ask(service, **kwargs):
            raise AskUnavailable("model down")

        monkeypatch.setattr(ask_mod, "ask_memory", fake_ask)
        client = TestClient(main.app, raise_server_exceptions=False)
        resp = client.post("/v1/ask", json={"question": "q"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_mcp_tool_dispatch(self, monkeypatch):
        import mcp_server

        async def fake_ask(service, **kwargs):
            return {
                "status": "ok", "reasoning_level": kwargs["reasoning_level"],
                "answer": "hi [m1]", "citations": ["m1"], "abstained": False,
                "searches": ["q"], "memories_considered": 1,
            }

        monkeypatch.setattr(ask_mod, "ask_memory", fake_ask)
        result = await mcp_server.call_tool("ask_memory", {
            "question": "q?", "user_id": "alice", "reasoning_level": "medium",
        })
        data = json.loads(result[0].text)
        assert data["answer"] == "hi [m1]" and data["reasoning_level"] == "medium"

    @pytest.mark.asyncio
    async def test_mcp_tool_rejects_bad_level_and_missing_question(self):
        import mcp_server

        bad_level = await mcp_server.call_tool("ask_memory", {
            "question": "q", "reasoning_level": "turbo",
        })
        assert "error" in json.loads(bad_level[0].text)
        no_question = await mcp_server.call_tool("ask_memory", {"reasoning_level": "low"})
        assert "error" in json.loads(no_question[0].text)


# ──────────────────────────────────────────────
# keyword_search (grep-style pass, service layer)
# ──────────────────────────────────────────────


class TestKeywordSearch:
    def _service_with_points(self, points):
        from memory_service import MemoryService

        svc = MemoryService()
        svc._memory = MagicMock(name="Memory")
        svc._memory.vector_store.client.scroll.return_value = (points, None)
        return svc

    def _point(self, mid, content, user_id="u", **meta):
        from types import SimpleNamespace

        return SimpleNamespace(id=mid, payload={
            "data": content, "created_at": "2026-07-01T00:00:00+00:00",
            "user_id": user_id,
            "metadata": {"category": "decision", "owner_user_id": user_id,
                         "visibility": "private", "scope": "global", **meta},
        })

    def test_case_insensitive_substring_match(self):
        svc = self._service_with_points([
            self._point("m1", "Deployed the Blue-Green pipeline"),
            self._point("m2", "Wrote docs about testing"),
        ])
        out, capped = svc.keyword_search("u", ["blue-green"])
        assert [m.id for m in out] == ["m1"]
        assert capped is False

    def test_two_terms_require_both(self):
        """Audit 27 #18: single-common-term rows are noise, not exact
        matches — with two terms a row must contain BOTH."""
        svc = self._service_with_points([
            self._point("m1", "apples are great"),
            self._point("m2", "bananas are fine"),
            self._point("m3", "apples and bananas smoothie"),
        ])
        out, _ = svc.keyword_search("u", ["apples", "bananas"])
        assert {m.id for m in out} == {"m3"}

    def test_three_plus_terms_require_at_least_two(self):
        svc = self._service_with_points([
            self._point("m1", "apples alone here"),
            self._point("m2", "apples and bananas together"),
            self._point("m3", "bananas with carrots on top"),
        ])
        out, _ = svc.keyword_search("u", ["apples", "bananas", "carrots"])
        assert {m.id for m in out} == {"m2", "m3"}

    def test_single_term_still_matches_alone(self):
        svc = self._service_with_points([
            self._point("m1", "apples are great"),
            self._point("m2", "carrots though"),
        ])
        out, _ = svc.keyword_search("u", ["apples"])
        assert [m.id for m in out] == ["m1"]

    def test_scan_capped_flag_set_when_point_cap_hit(self):
        """Ten 500-point pages with more remaining → the 5000-point cap
        terminated the scan; the caller must know it was partial."""
        from memory_service import MemoryService

        svc = self._service_with_points([])
        pages = [
            ([self._point(f"m{p}-{i}", "unrelated note") for i in range(500)], p + 1)
            for p in range(10)
        ]
        svc._memory.vector_store.client.scroll.side_effect = pages
        out, capped = svc.keyword_search("u", ["apples", "bananas"])
        assert out == []
        assert capped is True
        assert MemoryService._TIMELINE_FALLBACK_CAP == 5000  # test models the real cap

    def test_scan_not_capped_when_collection_exhausted(self):
        svc = self._service_with_points([
            self._point("m1", "apples and bananas"),
        ])  # helper returns (points, None): collection fully scanned
        _, capped = svc.keyword_search("u", ["apples", "bananas"])
        assert capped is False

    def test_limit_respected(self):
        svc = self._service_with_points([
            self._point(f"m{i}", "deploy note") for i in range(10)
        ])
        out, _ = svc.keyword_search("u", ["deploy"], limit=3)
        assert len(out) == 3

    def test_empty_terms_short_circuits(self):
        svc = self._service_with_points([])
        assert svc.keyword_search("u", []) == ([], False)
        assert svc.keyword_search("u", ["", "  "]) == ([], False)
        svc._memory.vector_store.client.scroll.assert_not_called()

    def test_filter_excludes_tombstones_and_scopes_visibility(self):
        """The Qdrant filter must carry the visibility pool union and the
        dream-tombstone exclusion (same contract as the timeline)."""
        svc = self._service_with_points([])
        svc.keyword_search("alice", ["x"], project_id="p1")
        flt = svc._memory.vector_store.client.scroll.call_args.kwargs["scroll_filter"]
        rendered = str(flt)
        assert "dream_tombstoned" in rendered
        assert "alice" in rendered  # own-rows condition
        assert "shared" in rendered  # shared pool condition
        assert "p1" in rendered  # project dual-scope


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


class TestHelpers:
    def test_enumeration_detection(self):
        assert is_enumeration_question("How many meetings did I have?")
        assert is_enumeration_question("List all the projects I work on")
        assert is_enumeration_question("Count the deploys this week")
        assert not is_enumeration_question("When is the standup?")

    def test_keyword_extraction_drops_stopwords(self):
        terms = extract_keywords("What are all the deploy pipelines for neuralscape?")
        assert "deploy" in terms and "pipelines" in terms and "neuralscape" in terms
        assert "the" not in terms and "what" not in terms and "all" not in terms

    def test_parse_llm_json_tolerates_fences(self):
        raw = '```json\n{"action": "answer", "answer": "x", "citations": []}\n```'
        parsed = ask_mod._parse_llm_json(raw)
        assert parsed and parsed["action"] == "answer"

    def test_parse_llm_json_extracts_embedded_object(self):
        raw = 'Sure! Here you go: {"action": "answer", "answer": "y"} hope that helps'
        parsed = ask_mod._parse_llm_json(raw)
        assert parsed and parsed["answer"] == "y"

    def test_parse_llm_json_garbage_is_none(self):
        assert ask_mod._parse_llm_json("no json here") is None
        assert ask_mod._parse_llm_json("") is None
