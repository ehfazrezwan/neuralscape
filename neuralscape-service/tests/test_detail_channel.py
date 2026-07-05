"""Detail memory channel — capped micro-detail capture without top-k dilution.

Empirical basis (DMR experiment ledger, 2026-07): distillation drops one-off
micro-details (possession attributes, stated reasons, gifts, fears,
family/pet specifics) — ~half of residual benchmark abstentions had the gold
fact ABSENT from the store. Naively densifying the fact extractor was
measured NET-NEGATIVE twice (66% vs 72%, 66% vs 70% on matched samples): the
extra facts dilute top-k retrieval. ADDITIVE CAPPED evidence channels win
(a 3-excerpt verbatim-episode leg added +3.8pp full-scale) because they never
compete with core facts for rank.

The channel, piece by piece (each covered here):

1. **Extraction contract** — the extraction JSON gains an optional
   ``"details"`` array alongside ``"facts"``; the facts instructions are
   byte-identical; flag off ⇒ prompt and parser byte-identical to today.
2. **Storage** — details are normal memories stamped
   ``metadata.memory_kind="detail"`` flowing through the same content-hash
   dedup / times_derived / tombstone-revival / two-pass-embed write path.
3. **Retrieval isolation (THE invariant)** — the main search excludes
   ``memory_kind="detail"`` rows at the Qdrant index (must_not, same
   mechanism as dream_tombstoned) so details can never dilute core top-k.
4. **Ask detail leg** — non-minimal tiers run ONE additional capped
   (limit-3) pass restricted to the detail channel; rows join evidence
   tagged ``source="detail"``; kill switch ``ASK_DETAIL_EVIDENCE``;
   failure non-fatal.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import settings
from prompts import (
    CODING_ASSISTANT_EXTRACTION_PROMPT,
    DETAIL_EXTRACTION_ADDENDUM,
    build_extraction_messages,
    parse_extraction_response,
    parse_extraction_response_with_details,
)
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Piece 1 — extraction contract (prompts.py)
# ──────────────────────────────────────────────


MSGS = [{"role": "user", "content": "My dog Biscuit is a corgi."}]


class TestExtractionPrompt:
    def test_flag_off_prompt_is_byte_identical_to_today(self):
        """The load-bearing back-compat assertion: include_details=False (the
        EXTRACT_DETAIL_MEMORIES=false kill switch) composes EXACTLY today's
        prompt — not a near-copy."""
        out = build_extraction_messages(MSGS, include_details=False)
        expected = CODING_ASSISTANT_EXTRACTION_PROMPT + "user: My dog Biscuit is a corgi.\n"
        assert out == [{"role": "user", "content": expected}]

    def test_default_is_flag_off_for_existing_callers(self):
        """Ingest extractors / extract_facts_only call without the kwarg —
        they must stay on the facts-only contract regardless of the setting."""
        assert build_extraction_messages(MSGS) == build_extraction_messages(
            MSGS, include_details=False
        )

    def test_flag_on_keeps_facts_instructions_byte_identical(self):
        """The 'capture micro-details too' prompt rule measured net-negative
        is exactly what this must NOT be: the facts rules text is unchanged,
        the detail channel is a purely additive addendum."""
        content = build_extraction_messages(MSGS, include_details=True)[0]["content"]
        base_body = CODING_ASSISTANT_EXTRACTION_PROMPT.rsplit("\nCONVERSATION:\n", 1)[0]
        assert base_body in content  # every facts instruction, byte-identical
        assert DETAIL_EXTRACTION_ADDENDUM in content
        assert '"details"' in content
        # Addendum sits between the facts contract and the conversation.
        assert content.index(DETAIL_EXTRACTION_ADDENDUM) < content.index("CONVERSATION:")

    def test_addendum_is_additive_not_reductive(self):
        assert "facts" in DETAIL_EXTRACTION_ADDENDUM
        # The addendum must explicitly instruct that facts extraction is
        # unchanged — never move facts into details.
        assert "exactly as instructed above" in DETAIL_EXTRACTION_ADDENDUM

    def test_base_prompt_constant_unchanged(self):
        """No detail contract leaks into the facts-only prompt constant."""
        assert '"details"' not in CODING_ASSISTANT_EXTRACTION_PROMPT
        assert "DETAIL CHANNEL" not in CODING_ASSISTANT_EXTRACTION_PROMPT
        assert CODING_ASSISTANT_EXTRACTION_PROMPT.startswith("You are a memory extraction engine")
        assert CODING_ASSISTANT_EXTRACTION_PROMPT.endswith("CONVERSATION:\n")


class TestDetailParsing:
    def test_facts_and_details_parsed_separately(self):
        raw = json.dumps({
            "facts": ["[preference] Prefers dark mode in every editor"],
            "details": ["Her dog Biscuit is a corgi", "[interaction] Got a red scarf as a gift"],
        })
        facts, details = parse_extraction_response_with_details(raw)
        assert facts == [("preference", "Prefers dark mode in every editor")]
        assert details == [
            ("personal_fact", "Her dog Biscuit is a corgi"),  # untagged → personal_fact
            ("interaction", "Got a red scarf as a gift"),     # tag respected
        ]

    def test_facts_parse_identical_to_legacy_parser(self):
        """The sibling's facts leg must agree byte-for-byte with
        parse_extraction_response on the same payload."""
        raw = json.dumps({
            "facts": ["[decision] Chose Qdrant over pgvector for recall latency"],
            "details": ["The demo laptop is the silver one"],
        })
        facts, _ = parse_extraction_response_with_details(raw)
        assert facts == parse_extraction_response(raw)

    def test_missing_details_key_degrades_to_empty(self):
        raw = json.dumps({"facts": ["[preference] Uses vim keybindings everywhere"]})
        facts, details = parse_extraction_response_with_details(raw)
        assert len(facts) == 1
        assert details == []

    def test_malformed_details_never_break_fact_parsing(self):
        for bad in ('"not-a-list"', "17", '{"nested": true}', "null"):
            raw = f'{{"facts": ["[preference] Likes tabs over spaces"], "details": {bad}}}'
            facts, details = parse_extraction_response_with_details(raw)
            assert facts == [("preference", "Likes tabs over spaces")], bad
            assert details == [], bad

    def test_non_string_and_blank_detail_entries_dropped(self):
        raw = json.dumps({
            "facts": [],
            "details": [42, None, "", "   ", {"x": 1}, "The cat is named Mochi"],
        })
        _, details = parse_extraction_response_with_details(raw)
        assert details == [("personal_fact", "The cat is named Mochi")]

    def test_fence_tolerant_like_the_rest(self):
        raw = "```json\n" + json.dumps({
            "facts": ["[preference] Prefers matcha over coffee"],
            "details": ["Their niece is named Lily"],
        }) + "\n```"
        facts, details = parse_extraction_response_with_details(raw)
        assert facts == [("preference", "Prefers matcha over coffee")]
        assert details == [("personal_fact", "Their niece is named Lily")]

    def test_unparseable_response_degrades_like_legacy(self):
        """Non-JSON: facts fall back to the line-scan (unchanged), details []."""
        raw = '[preference] Likes rainy mornings\ntotal garbage'
        facts, details = parse_extraction_response_with_details(raw)
        assert facts == parse_extraction_response(raw)
        assert ("preference", "Likes rainy mornings") in facts
        assert details == []

    def test_legacy_parser_ignores_details_key(self):
        """parse_extraction_response itself is untouched by the channel."""
        raw = json.dumps({
            "facts": ["[preference] Reads changelogs for fun"],
            "details": ["Owns a blue bike"],
        })
        assert parse_extraction_response(raw) == [("preference", "Reads changelogs for fun")]


class TestDetailConfig:
    def test_flags_exist_with_safe_defaults(self):
        assert settings.extract_detail_memories is True
        assert settings.ask_detail_evidence is True

    def test_env_names(self):
        """Both flags are plain BaseSettings fields → EXTRACT_DETAIL_MEMORIES /
        ASK_DETAIL_EVIDENCE env vars."""
        from config import Settings

        assert "extract_detail_memories" in Settings.model_fields
        assert "ask_detail_evidence" in Settings.model_fields


# ──────────────────────────────────────────────
# Piece 2 — storage: memory_kind="detail" stamping + shared write path
# ──────────────────────────────────────────────


from memory_service import MemoryService  # noqa: E402


@pytest.fixture
def service():
    """MemoryService with mocked internals (mirrors test_write_path.py)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.embedding_model.embed_batch.side_effect = (
        lambda texts, **kw: [[0.1] * 768 for _ in texts]
    )
    svc._attach_memory_id_to_graph_nodes = MagicMock(name="attach")
    svc._graph_episode_exists = MagicMock(return_value=True)  # skip graph adds
    svc._find_by_content_hash = MagicMock(return_value=None)
    return svc


@pytest.fixture(autouse=True)
def _no_operator_guidance(monkeypatch):
    """Hermetic: operator guidance resolution must not touch Redis/env."""
    import extraction_settings

    monkeypatch.setattr(extraction_settings, "resolve_instructions", lambda *a, **k: None)


def _mock_extraction(svc, payload: dict) -> MagicMock:
    client = MagicMock()
    svc._genai_model = client
    client.models.generate_content.return_value = MagicMock(text=json.dumps(payload))
    return client


CONVO = [{"role": "user", "content": "I got my sister a red scarf for her birthday."}]


class TestDetailStorage:
    def test_details_stored_with_memory_kind_detail(self, service):
        _mock_extraction(service, {
            "facts": ["[preference] Prefers handwritten thank-you notes"],
            "details": ["Gave her sister a red scarf for her birthday"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")

        payloads = []
        for call in service._memory.vector_store.insert.call_args_list:
            payloads.extend(call.kwargs["payloads"])
        by_kind = {p["metadata"].get("memory_kind"): p for p in payloads}
        assert set(by_kind) == {None, "detail"}
        detail_payload = by_kind["detail"]
        assert detail_payload["data"] == "Gave her sister a red scarf for her birthday"
        assert detail_payload["metadata"]["category"] == "personal_fact"
        # Details are returned alongside facts (they were stored).
        assert {m.memory for m in stored} == {
            "Prefers handwritten thank-you notes",
            "Gave her sister a red scarf for her birthday",
        }
        detail_resp = next(m for m in stored if m.memory_kind == "detail")
        assert detail_resp.category == "personal_fact"

    def test_detail_with_category_tag_keeps_it(self, service):
        _mock_extraction(service, {
            "facts": [],
            "details": ["[interaction] Mentioned being afraid of deep water at the lake trip"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")
        assert len(stored) == 1
        assert stored[0].category == "interaction"
        assert stored[0].memory_kind == "detail"

    def test_flag_off_is_byte_identical_facts_only_path(self, service, monkeypatch):
        """EXTRACT_DETAIL_MEMORIES=false ⇒ the prompt sent to Gemini is
        byte-identical to today's AND a details-bearing reply is ignored."""
        monkeypatch.setattr(settings, "extract_detail_memories", False)
        client = _mock_extraction(service, {
            "facts": ["[preference] Prefers handwritten thank-you notes"],
            "details": ["Gave her sister a red scarf for her birthday"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")

        sent = client.models.generate_content.call_args.kwargs["contents"]
        assert sent == CODING_ASSISTANT_EXTRACTION_PROMPT + (
            "user: I got my sister a red scarf for her birthday.\n"
        )
        assert [m.memory for m in stored] == ["Prefers handwritten thank-you notes"]
        assert all(m.memory_kind is None for m in stored)

    def test_flag_on_prompt_carries_detail_contract(self, service):
        client = _mock_extraction(service, {"facts": [], "details": []})
        service.extract_and_store(messages=CONVO, user_id="ehfaz")
        sent = client.models.generate_content.call_args.kwargs["contents"]
        assert DETAIL_EXTRACTION_ADDENDUM in sent

    def test_details_only_conversation_still_stores(self, service):
        """No core facts extracted but details present — the channel write
        must not be dropped by the legacy no-facts early return."""
        _mock_extraction(service, {
            "facts": [],
            "details": ["Their neighbor's parrot is named Pesto"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")
        assert len(stored) == 1
        assert stored[0].memory_kind == "detail"

    def test_details_pass_the_junk_filter(self, service):
        _mock_extraction(service, {
            "facts": [],
            "details": ["short", "Ran command: git push origin main", "Their neighbor's parrot is named Pesto"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")
        assert [m.memory for m in stored] == ["Their neighbor's parrot is named Pesto"]

    def test_details_share_the_dedup_write_path(self, service):
        """Re-deriving a stored detail bumps times_derived on the survivor and
        revives a dream tombstone — the PR #120/#124 invariants hold because
        details flow through the SAME _batch_store_facts path as facts."""
        survivor = MemoryResponse(
            id="d-1", memory="Gave her sister a red scarf for her birthday",
            category="personal_fact", memory_kind="detail",
        )
        service._find_by_content_hash = MagicMock(return_value=survivor)
        service._bump_times_derived = MagicMock(name="bump")
        service._revive_if_tombstoned = MagicMock(name="revive", return_value=True)
        _mock_extraction(service, {
            "facts": [],
            "details": ["Gave her sister a red scarf for her birthday"],
        })
        stored = service.extract_and_store(messages=CONVO, user_id="ehfaz")
        service._bump_times_derived.assert_called_once_with("d-1", 1)
        service._revive_if_tombstoned.assert_called_once_with("d-1")
        assert stored[0].id == "d-1" and stored[0].revived is True
        # Dedup hit ⇒ zero new points inserted.
        service._memory.vector_store.insert.assert_not_called()

    def test_detail_is_a_valid_memory_kind(self):
        from schemas import MEMORY_KIND_VOCAB, RawMemoryRequest

        assert "detail" in MEMORY_KIND_VOCAB
        req = RawMemoryRequest(
            content="Her cat is named Mochi", user_id="u",
            category="personal_fact", memory_kind="detail",
        )
        assert req.memory_kind == "detail"


# ──────────────────────────────────────────────
# Piece 3 — retrieval isolation: THE load-bearing invariant.
# Main search excludes memory_kind="detail" at the Qdrant index
# (must_not, same mechanism as dream_tombstoned).
# ──────────────────────────────────────────────


def _qresult(hits):
    r = MagicMock()
    r.points = hits
    return r


def _kind_conditions(conditions):
    return [
        c for c in (conditions or [])
        if getattr(c, "key", None) == "metadata.memory_kind"
        and getattr(getattr(c, "match", None), "value", None) == "detail"
    ]


@pytest.fixture
def search_service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = None  # vector pools only
    svc._bridge = None
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    svc._memory.vector_store._has_bm25_slot = False  # skip the lexical leg
    return svc


class TestMainSearchExcludesDetails:
    def _filters(self, svc):
        return [
            call.kwargs["query_filter"]
            for call in svc._memory.vector_store.client.query_points.call_args_list
        ]

    def test_personal_pool_excludes_detail_rows_at_the_index(self, search_service):
        search_service.search(query="red scarf", user_id="u1", limit=10)
        filters = self._filters(search_service)
        assert filters, "personal pool never queried"
        for flt in filters:
            assert _kind_conditions(flt.must_not), (
                "main search filter is missing the metadata.memory_kind=detail "
                f"must_not exclusion: {flt}"
            )
            assert not _kind_conditions(flt.must)

    def test_shared_and_standard_pools_exclude_detail_rows(self, search_service, monkeypatch):
        monkeypatch.setattr(settings, "standards_enabled", True)
        search_service.search(query="red scarf", user_id="u1", limit=10)
        filters = self._filters(search_service)
        assert len(filters) >= 3  # personal + shared + standard
        for flt in filters:
            assert _kind_conditions(flt.must_not)

    def test_project_dual_scope_passes_exclude_detail_rows(self, search_service):
        search_service.search(query="red scarf", user_id="u1", project_id="p1", limit=10)
        for flt in self._filters(search_service):
            assert _kind_conditions(flt.must_not)

    def test_explicit_detail_search_flips_the_filter(self, search_service):
        """memory_kind='detail' is the ONLY way detail rows are recallable:
        the same condition moves from must_not to must."""
        search_service.search(query="red scarf", user_id="u1", limit=3, memory_kind="detail")
        filters = self._filters(search_service)
        assert filters
        for flt in filters:
            assert _kind_conditions(flt.must)
            assert not _kind_conditions(flt.must_not)

    def test_detail_rows_returned_on_explicit_detail_search(self, search_service):
        hit = MagicMock()
        hit.id = "d1"
        hit.score = 0.9
        hit.payload = {
            "data": "Gave her sister a red scarf",
            "created_at": "2026-07-01T00:00:00+00:00",
            "metadata": {"category": "personal_fact", "memory_kind": "detail"},
        }
        search_service._memory.vector_store.client.query_points.return_value = _qresult([hit])
        out = search_service.search(query="red scarf", user_id="u1", limit=3,
                                    memory_kind="detail")
        assert [r.id for r in out] == ["d1"]
        assert out[0].memory_kind == "detail"

    def test_detail_search_skips_the_graph_pass(self, search_service):
        """Graph edges can never be detail rows — a detail-restricted search
        must not pay (or wait on) the graph leg."""
        search_service._graphiti = MagicMock(name="Graphiti")
        search_service._bridge = MagicMock(name="AsyncBridge")
        search_service._search_graph_for_visibility = MagicMock(
            return_value={"edges": [], "nodes": [], "episodes": [], "communities": []}
        )
        search_service.search(query="q", user_id="u1", limit=3, memory_kind="detail")
        search_service._search_graph_for_visibility.assert_not_called()
        search_service.search(query="q", user_id="u1", limit=3)
        search_service._search_graph_for_visibility.assert_called_once()

    def test_keyword_search_excludes_detail_rows(self, search_service):
        """ask's grep pass uses keyword_search with a ~20-row limit — details
        must not enter evidence through it (the capped leg is the only door)."""
        search_service._memory.vector_store.client.scroll.return_value = ([], None)
        search_service.keyword_search("u1", ["scarf"])
        flt = search_service._memory.vector_store.client.scroll.call_args.kwargs["scroll_filter"]
        assert _kind_conditions(flt.must_not)

    def test_passage_and_fact_filters_unchanged(self, search_service):
        """Regression guard: the existing post-hoc fact/passage semantics keep
        working, and both exclude detail rows at the index."""
        for kind in ("fact", "passage"):
            search_service._memory.vector_store.client.query_points.reset_mock()
            search_service.search(query="q", user_id="u1", limit=5, memory_kind=kind)
            for flt in self._filters(search_service):
                assert _kind_conditions(flt.must_not)


# ──────────────────────────────────────────────
# Piece 4 — ask detail evidence leg (capped, additive, non-fatal)
# ──────────────────────────────────────────────


import ask as ask_mod  # noqa: E402
from ask import _DETAIL_EVIDENCE_LIMIT, ask_memory, extract_keywords  # noqa: E402


def _mem(mid, content, **kw):
    return MemoryResponse(
        id=mid, memory=content, category="personal_fact", source="vector",
        created_at="2026-07-01T00:00:00+00:00", score=0.9, **kw,
    )


def _ask_service(main_rows=None, detail_rows=None):
    """Fake service whose search distinguishes the detail-leg call by its
    memory_kind kwarg (the main passes never send one)."""
    svc = MagicMock(name="MemoryService")

    def search_fn(**kwargs):
        if kwargs.get("memory_kind") == "detail":
            return list(detail_rows or [])
        return list(main_rows or [])

    svc.search.side_effect = search_fn
    svc.keyword_search.return_value = ([], False)
    return svc


def _detail_calls(svc):
    return [c for c in svc.search.call_args_list
            if c.kwargs.get("memory_kind") == "detail"]


def _answer_llm(answer, citations=None):
    prompts = []

    async def call(prompt):
        prompts.append(prompt)
        return json.dumps({"action": "answer", "answer": answer,
                           "citations": citations or [], "abstained": False})

    call.prompts = prompts
    return call


class TestAskDetailLeg:
    @pytest.mark.asyncio
    async def test_detail_leg_runs_capped_and_filtered(self):
        detail = _mem("d1", "Gave her sister a red scarf", memory_kind="detail")
        svc = _ask_service([_mem("m1", "Has one sister")], [detail])
        llm = _answer_llm("a red scarf [d1]", ["d1"])
        out = await ask_memory(svc, question="What gift did she give her sister?",
                               user_id="u", reasoning_level="low", llm_call=llm)
        calls = _detail_calls(svc)
        assert len(calls) == 1
        assert calls[0].kwargs["limit"] == _DETAIL_EVIDENCE_LIMIT == 3
        assert calls[0].kwargs["user_id"] == "u"
        # Detail row joined the evidence: rendered + citable.
        assert "[d1]" in llm.prompts[0]
        assert out["citations"] == ["d1"]
        assert out["memories_considered"] == 2

    @pytest.mark.asyncio
    async def test_detail_query_is_stopword_filtered_keywords(self):
        svc = _ask_service([_mem("m1", "x")])
        question = "What are all the gifts for the wedding?"
        await ask_memory(svc, question=question, user_id="u",
                         reasoning_level="medium", llm_call=_answer_llm("a"))
        call = _detail_calls(svc)[0]
        expected = " ".join(extract_keywords(question))
        assert call.kwargs["query"] == expected
        assert "the" not in expected.split()

    @pytest.mark.asyncio
    async def test_detail_pass_recorded_in_searches(self):
        svc = _ask_service([_mem("m1", "x")])
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="low", llm_call=_answer_llm("a"))
        assert any(s.startswith("detail:") for s in out["searches"])

    @pytest.mark.asyncio
    async def test_rows_join_evidence_tagged_source_detail(self):
        detail = _mem("d1", "Her bike is blue", memory_kind="detail")
        svc = _ask_service([_mem("m1", "Owns a bike")], [detail])
        await ask_memory(svc, question="What color is her bike?", user_id="u",
                         reasoning_level="low", llm_call=_answer_llm("blue [d1]", ["d1"]))
        assert detail.source == "detail"

    @pytest.mark.asyncio
    async def test_minimal_tier_skips_the_leg(self):
        svc = _ask_service([_mem("m1", "x")])
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="minimal", llm_call=_answer_llm("a"))
        assert _detail_calls(svc) == []
        assert svc.search.call_count == 1
        assert not any(s.startswith("detail:") for s in out["searches"])

    @pytest.mark.asyncio
    async def test_all_non_minimal_tiers_run_the_leg(self):
        for level in ("low", "medium", "high"):
            svc = _ask_service([_mem("m1", "x")])
            await ask_memory(svc, question="What color is her bike?", user_id="u",
                             reasoning_level=level, llm_call=_answer_llm("a"))
            assert len(_detail_calls(svc)) == 1, level

    @pytest.mark.asyncio
    async def test_kill_switch_disables_the_leg(self, monkeypatch):
        monkeypatch.setattr(settings, "ask_detail_evidence", False)
        svc = _ask_service([_mem("m1", "x")])
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="high", llm_call=_answer_llm("a"))
        assert _detail_calls(svc) == []
        assert not any(s.startswith("detail:") for s in out["searches"])

    @pytest.mark.asyncio
    async def test_leg_failure_is_non_fatal(self):
        main = _mem("m1", "Owns a bike")
        svc = MagicMock(name="MemoryService")

        def search_fn(**kwargs):
            if kwargs.get("memory_kind") == "detail":
                raise RuntimeError("qdrant hiccup")
            return [main]

        svc.search.side_effect = search_fn
        svc.keyword_search.return_value = ([], False)
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="low", llm_call=_answer_llm("a [m1]", ["m1"]))
        assert out["status"] == "ok"
        assert out["citations"] == ["m1"]

    @pytest.mark.asyncio
    async def test_leg_tolerates_non_list_result(self):
        svc = MagicMock(name="MemoryService")

        def search_fn(**kwargs):
            if kwargs.get("memory_kind") == "detail":
                return {"weird": "shape"}
            return [_mem("m1", "Owns a bike")]

        svc.search.side_effect = search_fn
        svc.keyword_search.return_value = ([], False)
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="low", llm_call=_answer_llm("a [m1]", ["m1"]))
        assert out["status"] == "ok"
        assert out["memories_considered"] == 1

    @pytest.mark.asyncio
    async def test_detail_rows_never_displace_core_evidence(self):
        """The additive-capped property: detail rows join evidence ON TOP of
        the main rows — every main-pass row is still rendered."""
        main = [_mem(f"m{i}", f"core fact number {i}") for i in range(5)]
        details = [_mem(f"d{i}", f"micro detail {i}", memory_kind="detail")
                   for i in range(3)]
        svc = _ask_service(main, details)
        llm = _answer_llm("a")
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="low", llm_call=llm)
        assert out["memories_considered"] == 8
        for m in main:
            assert f"[{m.id}]" in llm.prompts[0]

    @pytest.mark.asyncio
    async def test_no_evidence_at_all_still_abstains_cleanly(self):
        svc = _ask_service([], [])
        out = await ask_memory(svc, question="What color is her bike?", user_id="u",
                               reasoning_level="low")
        assert out["abstained"] is True
        assert out["memories_considered"] == 0
