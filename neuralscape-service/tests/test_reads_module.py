"""Direct unit tests for ``memory/reads.py`` (ReadsMixin).

A real ``MemoryService`` is built with the store boundary mocked (mem0
``Memory``, its Qdrant client and the Graphiti bridge). The existing suites
cover the happy paths through REST/MCP; these tests target the helpers and
edge cases no other file reaches: id canonicalisation, the ordered context
fetch internals (tie-group dedup, undated rows, tombstone belt-and-braces,
fallback cut), the count fallback, the scroll fallback, timeline anchor
resolution and windows, batch-get validation, list filters and the
user-id / null-category scans.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    HasIdCondition,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
)

import memory.reads as reads
from memory_service import MemoryService
from schemas import MemoryResponse

UID = "11111111-1111-4111-8111-111111111111"
UID2 = "22222222-2222-4222-8222-222222222222"
UID3 = "33333333-3333-4333-8333-333333333333"


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    # Indexes are "already ensured": reads must not call create_payload_index.
    svc._filter_indexes_ok = True
    svc._created_at_index_ok = True
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    return svc


def _client(svc):
    return svc._memory.vector_store.client


def _pt(pid, created_at="2026-07-01T00:00:00Z", metadata=None, user_id="u", data=None):
    return SimpleNamespace(
        id=pid,
        score=None,
        payload={
            "user_id": user_id,
            "data": data if data is not None else f"fact {pid}",
            "metadata": metadata if metadata is not None else {"category": "preference"},
            "created_at": created_at,
        },
    )


def _keys(flt):
    out = []
    for bucket in (flt.must, flt.should, flt.must_not):
        for c in bucket or []:
            out.extend([c.key] if isinstance(c, FieldCondition) else _keys(c) if isinstance(c, Filter) else [])
    return out


# ── _canonical_point_id ────────────────────────────────────────────────


class TestCanonicalPointId:
    canon = staticmethod(MemoryService._canonical_point_id)

    def test_lowercase_uuid_unchanged(self):
        assert self.canon(UID) == UID

    def test_uppercase_uuid_lowercased(self):
        assert self.canon(UID.upper()) == UID

    def test_braced_uuid(self):
        assert self.canon("{" + UID + "}") == UID

    def test_dashless_uuid(self):
        letters = "abcdef12-3456-4789-8abc-def012345678"
        assert self.canon(letters.replace("-", "")) == letters

    def test_all_digit_dashless_uuid_is_still_a_uuid(self):
        # 32 decimal digits >= 2**64: must fall through to the UUID parse
        assert self.canon(UID.replace("-", "")) == UID

    def test_digits_become_int(self):
        assert self.canon("12345") == 12345
        assert self.canon("0") == 0

    def test_int_input(self):
        assert self.canon(42) == 42

    def test_uint64_boundary(self):
        assert self.canon(str(2**64 - 1)) == 2**64 - 1
        assert self.canon(str(2**64)) is None

    @pytest.mark.parametrize(
        "bad", ["", "   ", "not-a-uuid", "12-34", "-5", "1.5", "abc123", "٣٤"]
    )
    def test_malformed_returns_none(self, bad):
        # includes non-ASCII digits (Arabic-Indic), which str.isdigit() accepts
        assert self.canon(bad) is None

    def test_none_input_is_string_none(self):
        assert self.canon(None) is None


# ── get_memory ─────────────────────────────────────────────────────────


class TestGetMemory:
    def test_malformed_id_never_touches_store(self, service):
        assert service.get_memory("../../etc/passwd") is None
        service._memory.get.assert_not_called()

    def test_numeric_id_passed_to_store_as_int(self, service):
        service._memory.get.return_value = {"id": "7", "memory": "x", "metadata": {}}
        service.get_memory("7")
        service._memory.get.assert_called_once_with(7)

    def test_uuid_canonicalised_before_lookup(self, service):
        service._memory.get.return_value = None
        service.get_memory(UID.upper())
        service._memory.get.assert_called_once_with(UID)

    def test_unknown_returns_none(self, service):
        service._memory.get.return_value = None
        assert service.get_memory(UID) is None

    def test_empty_result_dict_returns_none(self, service):
        service._memory.get.return_value = {}
        assert service.get_memory(UID) is None

    def test_found_without_caller_skips_gate(self, service):
        service._memory.get.return_value = {
            "id": UID, "memory": "m", "user_id": "someone-else", "metadata": {"category": "preference"},
        }
        got = service.get_memory(UID)
        assert isinstance(got, MemoryResponse)
        assert got.id == UID and got.category == "preference"

    def test_caller_owner_allowed(self, service):
        service._memory.get.return_value = {"id": UID, "memory": "m", "user_id": "alice", "metadata": {}}
        assert service.get_memory(UID, caller_user_id="alice") is not None

    def test_other_users_private_row_hidden(self, service):
        service._memory.get.return_value = {"id": UID, "memory": "m", "user_id": "bob", "metadata": {}}
        assert service.get_memory(UID, caller_user_id="alice") is None

    def test_shared_row_visible_to_other_user(self, service):
        service._memory.get.return_value = {
            "id": UID, "memory": "m", "user_id": "bob", "metadata": {"visibility": "shared"},
        }
        assert service.get_memory(UID, caller_user_id="alice") is not None

    def test_owner_user_id_in_metadata_grants_access(self, service):
        service._memory.get.return_value = {
            "id": UID, "memory": "m", "user_id": "bob", "metadata": {"owner_user_id": "alice"},
        }
        assert service.get_memory(UID, caller_user_id="alice") is not None

    def test_unwraps_double_wrapped_metadata(self, service):
        service._memory.get.return_value = {
            "id": UID, "memory": "m", "metadata": {"metadata": {"category": "tech_stack", "scope": "project"}},
        }
        got = service.get_memory(UID)
        assert got.category == "tech_stack" and got.scope == "project"


# ── get_memories_by_ids ────────────────────────────────────────────────


class TestGetMemoriesByIds:
    def test_empty_list_raises(self, service):
        with pytest.raises(ValueError, match="non-empty"):
            service.get_memories_by_ids([], "u")

    def test_oversized_list_raises(self, service):
        ids = [str(uuid.uuid4()) for _ in range(reads.GET_MEMORIES_MAX_IDS + 1)]
        with pytest.raises(ValueError, match="At most"):
            service.get_memories_by_ids(ids, "u")

    def test_exactly_max_ids_allowed(self, service):
        ids = [str(uuid.uuid4()) for _ in range(reads.GET_MEMORIES_MAX_IDS)]
        _client(service).retrieve.return_value = []
        out = service.get_memories_by_ids(ids, "u")
        assert out["results"] == [] and len(out["missing"]) == reads.GET_MEMORIES_MAX_IDS

    def test_all_malformed_skips_retrieve(self, service):
        out = service.get_memories_by_ids(["nope", "also-bad"], "u")
        _client(service).retrieve.assert_not_called()
        assert out == {"results": [], "missing": ["nope", "also-bad"]}

    def test_malformed_filtered_out_of_retrieve_call(self, service):
        _client(service).retrieve.return_value = [_pt(UID, user_id="u")]
        out = service.get_memories_by_ids([UID, "bad"], "u")
        assert _client(service).retrieve.call_args.kwargs["ids"] == [UID]
        assert [r.id for r in out["results"]] == [UID]
        assert out["missing"] == ["bad"]

    def test_duplicates_collapsed_input_order_kept(self, service):
        _client(service).retrieve.return_value = [
            _pt(UID3, user_id="u"), _pt(UID, user_id="u"),
        ]
        out = service.get_memories_by_ids([UID, UID3, UID], "u")
        assert _client(service).retrieve.call_args.kwargs["ids"] == [UID, UID3]
        assert [r.id for r in out["results"]] == [UID, UID3]

    def test_unreadable_row_reported_missing_like_unknown(self, service):
        _client(service).retrieve.return_value = [_pt(UID, user_id="bob")]
        out = service.get_memories_by_ids([UID, UID2], "alice")
        assert out["results"] == []
        assert out["missing"] == [UID, UID2]

    def test_shared_row_readable(self, service):
        _client(service).retrieve.return_value = [
            _pt(UID, user_id="bob", metadata={"visibility": "shared"}),
        ]
        out = service.get_memories_by_ids([UID], "alice")
        assert [r.id for r in out["results"]] == [UID]

    def test_none_caller_only_sees_shared(self, service):
        _client(service).retrieve.return_value = [
            _pt(UID, user_id="bob", metadata={"visibility": "shared"}),
            _pt(UID2, user_id="bob"),
        ]
        out = service.get_memories_by_ids([UID, UID2], None)
        assert [r.id for r in out["results"]] == [UID]
        assert out["missing"] == [UID2]

    def test_retrieve_returning_none_treated_as_empty(self, service):
        _client(service).retrieve.return_value = None
        out = service.get_memories_by_ids([UID], "u")
        assert out == {"results": [], "missing": [UID]}

    def test_non_string_ids_coerced(self, service):
        _client(service).retrieve.return_value = [_pt(UID, user_id="u")]
        out = service.get_memories_by_ids([uuid.UUID(UID)], "u")
        assert [r.id for r in out["results"]] == [UID]

    def test_integer_ids_are_reported_missing_not_retrieved(self, service):
        # The batch path only accepts UUID spellings (contrast with get_memory).
        out = service.get_memories_by_ids(["123"], "u")
        _client(service).retrieve.assert_not_called()
        assert out["missing"] == ["123"]


# ── _context_total ─────────────────────────────────────────────────────


class TestContextTotal:
    def test_exact_count_used(self, service):
        _client(service).count.return_value = MagicMock(count=40)
        assert service._context_total(MagicMock(), 5, 10) == 40

    def test_count_never_below_fetched(self, service):
        _client(service).count.return_value = MagicMock(count=3)
        assert service._context_total(MagicMock(), 8, 10) == 8

    def test_count_error_falls_back_to_fetched(self, service):
        _client(service).count.side_effect = RuntimeError("down")
        assert service._context_total(MagicMock(), 8, 10) == 8

    def test_count_requested_exact_with_filter(self, service):
        flt = MagicMock()
        _client(service).count.return_value = MagicMock(count=1)
        service._context_total(flt, 1, 1)
        kw = _client(service).count.call_args.kwargs
        assert kw["count_filter"] is flt and kw["exact"] is True

    def test_count_missing_attr_defaults_to_fetched(self, service):
        _client(service).count.return_value = object()
        assert service._context_total(MagicMock(), 4, 10) == 4


# ── _ordered_context_rows ──────────────────────────────────────────────


def _ordered_only(svc, points):
    """order_by scroll returns ``points``; every other scroll (undated) is empty."""
    _client(svc).scroll.side_effect = lambda **kw: (
        (list(points), None) if kw.get("order_by") is not None else ([], None)
    )


class TestOrderedContextRows:
    def test_need_clamped_to_one(self, service):
        service._ordered_context_rows(Filter(), 0)
        first = _client(service).scroll.call_args_list[0].kwargs
        assert first["limit"] == 1

    def test_need_clamped_to_fetch_cap(self, service):
        service._ordered_context_rows(Filter(), 10**9)
        first = _client(service).scroll.call_args_list[0].kwargs
        assert first["limit"] == reads.CONTEXT_FETCH_CAP

    def test_under_need_fetches_undated_rows_last(self, service):
        dated = _pt("d1", "2026-07-02T00:00:00Z")
        undated = _pt("u1", None)
        calls = []

        def scroll(**kw):
            calls.append(kw)
            if kw.get("order_by") is not None:
                return [dated], None
            return [undated], None

        _client(service).scroll.side_effect = scroll
        rows = service._ordered_context_rows(Filter(), 5)
        assert [r.id for r in rows] == ["d1", "u1"]
        assert len(calls) == 2
        # undated lookup asks for only the remaining slots
        assert calls[1]["limit"] == 4

    def test_at_need_pulls_tie_group_without_duplicates(self, service):
        same = "2026-07-01T00:00:00Z"
        first_page = [_pt("a", "2026-07-02T00:00:00Z"), _pt("b", same)]
        tie = [_pt("b", same), _pt("c", same), _pt("d", same)]

        def scroll(**kw):
            if kw.get("order_by") is not None:
                return first_page, None
            return tie, None

        _client(service).scroll.side_effect = scroll
        rows = service._ordered_context_rows(Filter(), 2)
        assert sorted(r.id for r in rows) == ["a", "b", "c", "d"]
        assert len({r.id for r in rows}) == 4

    def test_tie_scroll_filter_pins_boundary_timestamp(self, service):
        same = "2026-07-01T00:00:00Z"
        calls = []

        def scroll(**kw):
            calls.append(kw)
            return [_pt("a", same)], None

        _client(service).scroll.side_effect = scroll
        service._ordered_context_rows(Filter(), 1)
        tie_filter = calls[1]["scroll_filter"]
        cond = tie_filter.must[1]
        assert cond.key == "created_at"
        boundary = datetime(2026, 7, 1, tzinfo=timezone.utc)
        assert cond.range.gte == boundary and cond.range.lte == boundary

    def test_rows_sorted_newest_first_then_id_desc(self, service):
        _ordered_only(service, [
            _pt("a", "2026-07-01T00:00:00Z"), _pt("c", "2026-07-03T00:00:00Z"),
            _pt("b", "2026-07-03T00:00:00Z"),
        ])
        rows = service._ordered_context_rows(Filter(), 10)
        assert [r.id for r in rows] == ["c", "b", "a"]

    def test_tombstoned_rows_dropped_even_if_server_returned_them(self, service):
        live = _pt("live", "2026-07-02T00:00:00Z")
        dead = _pt("dead", "2026-07-03T00:00:00Z", metadata={"category": "preference", "dream_tombstoned": True})
        _ordered_only(service, [dead, live])
        rows = service._ordered_context_rows(Filter(), 10)
        assert [r.id for r in rows] == ["live"]

    def test_double_wrapped_tombstone_dropped(self, service):
        dead = _pt("dead", metadata={"metadata": {"dream_tombstoned": True}})
        _ordered_only(service, [dead])
        assert service._ordered_context_rows(Filter(), 10) == []

    def test_none_scroll_results_tolerated(self, service):
        _client(service).scroll.return_value = (None, None)
        assert service._ordered_context_rows(Filter(), 3) == []

    def test_order_by_failure_falls_back_and_cuts_at_need_keeping_ties(self, service):
        pts = [
            _pt("p1", "2026-07-05T00:00:00Z"),
            _pt("p2", "2026-07-04T00:00:00Z"),
            _pt("p3", "2026-07-03T00:00:00Z"),
            _pt("p4", "2026-07-03T00:00:00Z"),  # tied with the cut row
            _pt("p5", "2026-07-01T00:00:00Z"),
        ]

        def scroll(**kw):
            if kw.get("order_by") is not None:
                raise RuntimeError("no datetime index")
            return pts, None

        _client(service).scroll.side_effect = scroll
        rows = service._ordered_context_rows(Filter(), 3)
        # need=3 cuts at p3's timestamp; p4 shares it and must be kept, p5 dropped
        assert {r.id for r in rows} == {"p1", "p2", "p3", "p4"}

    def test_fallback_not_cut_when_fewer_than_need(self, service):
        def scroll(**kw):
            if kw.get("order_by") is not None:
                raise RuntimeError("boom")
            return [_pt("only")], None

        _client(service).scroll.side_effect = scroll
        assert [r.id for r in service._ordered_context_rows(Filter(), 10)] == ["only"]

    def test_ensures_created_at_index_first(self, service, monkeypatch):
        called = []
        monkeypatch.setattr(service, "_ensure_created_at_index", lambda: called.append(1))
        service._ordered_context_rows(Filter(), 1)
        assert called == [1]


# ── context filters ────────────────────────────────────────────────────


class TestContextFilters:
    def test_global_filter_is_caller_global_minus_tombstones(self, service):
        flt = service._global_context_filter("alice")
        assert FieldCondition(key="user_id", match=MatchValue(value="alice")) in flt.must
        assert FieldCondition(key="metadata.scope", match=MatchValue(value="global")) in flt.must
        assert [c.key for c in flt.must_not] == ["metadata.dream_tombstoned"]

    def test_project_filter_shape(self, service):
        flt = service._project_context_filter("alice", "proj")
        own, shared = flt.should
        assert "user_id" in _keys(own)
        assert "metadata.project_id" in _keys(own) and "metadata.scope" in _keys(own)
        assert "metadata.visibility" in _keys(shared)
        # workspace fence: only the "memory" workspace is served
        ws = [c for c in shared.must_not if c.key == "metadata.workspace"][0]
        assert ws.match.except_ == ["memory"]
        assert [c.key for c in flt.must_not] == ["metadata.dream_tombstoned"]
        # exact predicate values, not just keys
        assert FieldCondition(key="user_id", match=MatchValue(value="alice")) in own.must
        assert FieldCondition(key="metadata.project_id", match=MatchValue(value="proj")) in shared.must
        assert FieldCondition(key="metadata.visibility", match=MatchValue(value="shared")) in shared.must
        # own rows: the caller's rows in the default ("memory") workspace only
        # (reference-workspace rows stay out of session-start context), plus
        # a scope=global-or-this-project branch. Order: user, workspace, scope.
        assert len(own.must) == 3
        ws_fence, inner = own.must[1], own.must[2]
        assert isinstance(ws_fence, Filter) and isinstance(inner, Filter)
        assert IsEmptyCondition(is_empty=PayloadField(key="metadata.workspace")) in ws_fence.should
        assert FieldCondition(key="metadata.workspace", match=MatchValue(value="memory")) in ws_fence.should
        assert FieldCondition(key="metadata.workspace", match=MatchValue(value="")) in ws_fence.should
        assert not any(
            isinstance(c, FieldCondition) and isinstance(c.match, MatchAny) for c in ws_fence.should
        ), "no reference workspace may be admitted"
        assert FieldCondition(key="metadata.project_id", match=MatchValue(value="proj")) in inner.should
        assert FieldCondition(key="metadata.scope", match=MatchValue(value="global")) in inner.should

    def test_tombstone_condition_matches_true(self):
        cond = MemoryService._tombstone_condition()
        assert cond.key == "metadata.dream_tombstoned" and cond.match.value is True


# ── get_global_context / get_project_context edges ─────────────────────


class TestContextEdges:
    def _store(self, svc, rows, total=None):
        client = _client(svc)

        def scroll(**kw):
            if kw.get("order_by") is not None:
                return [r for r in rows if r.payload.get("created_at")][: kw["limit"]], None
            return [], None

        client.scroll.side_effect = scroll
        client.count.return_value = MagicMock(count=len(rows) if total is None else total)

    def test_global_context_empty_store(self, service):
        self._store(service, [])
        ctx = service.get_global_context("u")
        assert ctx.total == 0 and ctx.returned == 0 and ctx.has_more is False
        assert ctx.categories == {}

    def test_global_has_more_when_total_exceeds_served(self, service):
        rows = [_pt(f"r{i}", f"2026-07-01T00:{i // 60:02d}:{i % 60:02d}Z") for i in range(5)]
        self._store(service, rows, total=reads.CONTEXT_GLOBAL_LIMIT + 50)
        ctx = service.get_global_context("u")
        assert ctx.returned == 5 and ctx.has_more is True
        assert ctx.total == reads.CONTEXT_GLOBAL_LIMIT + 50
        assert ctx.offset == 0 and ctx.limit is None

    def test_global_total_never_below_returned(self, service):
        rows = [_pt("a"), _pt("b")]
        self._store(service, rows, total=0)
        ctx = service.get_global_context("u")
        assert ctx.total >= ctx.returned == 2

    def test_row_without_category_bucketed_as_personal_fact(self, service):
        self._store(service, [_pt("x", metadata={})])
        ctx = service.get_global_context("u")
        assert list(ctx.categories) == ["personal_fact"]

    def test_global_context_requests_exactly_the_global_limit(self, service):
        self._store(service, [])
        service.get_global_context("u")
        first = _client(service).scroll.call_args_list[0].kwargs
        assert first["limit"] == reads.CONTEXT_GLOBAL_LIMIT

    def test_project_unbounded_page_is_capped(self, service):
        self._store(service, [])
        service.get_project_context("u", "p")
        first = _client(service).scroll.call_args_list[0].kwargs
        assert first["limit"] == reads.CONTEXT_UNBOUNDED_PAGE

    def test_negative_offset_clamped_to_zero(self, service):
        self._store(service, [_pt("a")])
        ctx = service.get_project_context("u", "p", limit=5, offset=-9)
        assert ctx.offset == 0 and ctx.returned == 1

    def test_offset_past_end_returns_empty_page_without_has_more(self, service):
        self._store(service, [_pt("a"), _pt("b")])
        ctx = service.get_project_context("u", "p", limit=5, offset=10)
        assert ctx.returned == 0 and ctx.has_more is False and ctx.total == 2

    def test_project_response_echoes_ids_and_limit(self, service):
        self._store(service, [_pt("a")])
        ctx = service.get_project_context("alice", "proj", limit=3)
        assert (ctx.user_id, ctx.project_id, ctx.limit) == ("alice", "proj", 3)

    def test_zero_limit_normalised_to_one(self, service):
        self._store(service, [_pt("a"), _pt("b")])
        ctx = service.get_project_context("u", "p", limit=0)
        assert ctx.limit == 1 and ctx.returned == 1 and ctx.has_more is True

    def test_exact_fit_page_has_no_more(self, service):
        self._store(service, [_pt("a"), _pt("b"), _pt("c")])
        ctx = service.get_project_context("u", "p", limit=3)
        assert ctx.returned == 3 and ctx.total == 3 and ctx.has_more is False
        ctx = service.get_project_context("u", "p", limit=2)
        assert ctx.returned == 2 and ctx.has_more is True

    def test_page_order_newest_first_across_categories(self, service):
        rows = [
            _pt("old", "2026-07-01T00:00:00Z", metadata={"category": "tech_stack"}),
            _pt("new", "2026-07-03T00:00:00Z", metadata={"category": "preference"}),
            _pt("mid", "2026-07-02T00:00:00Z", metadata={"category": "tech_stack"}),
        ]
        self._store(service, rows)
        ctx = service.get_project_context("u", "p", limit=3)
        assert [m.id for m in ctx.categories["tech_stack"]] == ["mid", "old"]
        assert [m.id for m in ctx.categories["preference"]] == ["new"]
        first = _client(service).scroll.call_args_list[0].kwargs
        assert first["order_by"].key == "created_at" and first["order_by"].direction.value == "desc"

    def test_standards_returned_and_audited(self, service, monkeypatch):
        self._store(service, [])
        std = [MemoryResponse(id="s1", memory="rule")]
        monkeypatch.setattr(service, "_get_standards", lambda project_id=None: std)
        audit = MagicMock()
        monkeypatch.setattr(reads, "_audit_log", audit)
        ctx = service.get_project_context("u", "p")
        assert ctx.standards == std
        audit.info.assert_called_once()
        assert audit.info.call_args.args[0] == "standards_served"
        g = service.get_global_context("u")
        assert g.standards == std and audit.info.call_count == 2

    def test_no_standards_no_audit(self, service, monkeypatch):
        self._store(service, [])
        monkeypatch.setattr(service, "_get_standards", lambda project_id=None: [])
        audit = MagicMock()
        monkeypatch.setattr(reads, "_audit_log", audit)
        service.get_global_context("u")
        audit.info.assert_not_called()

    def test_project_standards_lookup_passes_project_id(self, service, monkeypatch):
        self._store(service, [])
        seen = {}
        monkeypatch.setattr(
            service, "_get_standards", lambda project_id=None: seen.setdefault("pid", project_id) and []
        )
        service.get_project_context("u", "projX")
        assert seen["pid"] == "projX"


# ── _scroll_all_sorted ─────────────────────────────────────────────────


class TestScrollAllSorted:
    def test_paginates_until_offset_none(self, service):
        pages = iter([([_pt("a", "2026-07-01T00:00:00Z")], "tok"), ([_pt("b", "2026-07-02T00:00:00Z")], None)])
        _client(service).scroll.side_effect = lambda **kw: next(pages)
        out = service._scroll_all_sorted(_client(service), MagicMock(), "before", 10)
        assert [p.id for p in out] == ["b", "a"]

    def test_zero_offset_does_not_end_pagination(self, service):
        # Integer point ids may legitimately be 0 (falsy): only None ends the scroll.
        pages = iter([([_pt("a")], 0), ([_pt("b")], None)])
        _client(service).scroll.side_effect = lambda **kw: next(pages)
        out = service._scroll_all_sorted(_client(service), MagicMock(), "after", 10)
        assert {p.id for p in out} == {"a", "b"}

    def test_before_is_newest_first_after_is_oldest_first(self, service):
        pts = [_pt("old", "2026-07-01T00:00:00Z"), _pt("new", "2026-07-09T00:00:00Z")]
        _client(service).scroll.side_effect = lambda **kw: (list(pts), None)
        before = service._scroll_all_sorted(_client(service), MagicMock(), "before", 10)
        after = service._scroll_all_sorted(_client(service), MagicMock(), "after", 10)
        assert [p.id for p in before] == ["new", "old"]
        assert [p.id for p in after] == ["old", "new"]

    def test_limit_applied(self, service):
        pts = [_pt(str(i), f"2026-07-0{i + 1}T00:00:00Z") for i in range(5)]
        _client(service).scroll.side_effect = lambda **kw: (list(pts), None)
        out = service._scroll_all_sorted(_client(service), MagicMock(), "after", 2)
        assert [p.id for p in out] == ["0", "1"]

    def test_cap_stops_runaway_scroll(self, service, monkeypatch):
        monkeypatch.setattr(MemoryService, "_TIMELINE_FALLBACK_PAGE", 2)
        monkeypatch.setattr(MemoryService, "_TIMELINE_FALLBACK_CAP", 4)
        n = {"calls": 0}

        def scroll(**kw):
            n["calls"] += 1
            return [_pt(f"{n['calls']}a"), _pt(f"{n['calls']}b")], "more"

        _client(service).scroll.side_effect = scroll
        service._scroll_all_sorted(_client(service), MagicMock(), "before", 100)
        assert n["calls"] == 2

    def test_none_page_tolerated(self, service):
        _client(service).scroll.side_effect = lambda **kw: (None, None)
        assert service._scroll_all_sorted(_client(service), MagicMock(), "before", 5) == []

    def test_undated_rows_sort_without_error(self, service):
        pts = [_pt("u", None), _pt("d", "2026-07-01T00:00:00Z")]
        _client(service).scroll.side_effect = lambda **kw: (list(pts), None)
        out = service._scroll_all_sorted(_client(service), MagicMock(), "before", 5)
        assert {p.id for p in out} == {"u", "d"}


# ── _timeline_filter / _timeline_window ────────────────────────────────

B = datetime(2026, 7, 1, tzinfo=timezone.utc)


class TestTimelineFilter:
    def test_before_uses_lt_after_uses_gte(self, service):
        before = service._timeline_filter("u", None, None, B, "before")
        after = service._timeline_filter("u", None, None, B, "after")

        def rng(flt):
            c = [c for c in flt.must if isinstance(c, FieldCondition) and c.key == "created_at"][0]
            return c.range

        assert rng(before) == DatetimeRange(lt=B)
        assert rng(after) == DatetimeRange(gte=B)

    def test_exclude_id_added_to_must_not(self, service):
        flt = service._timeline_filter("u", None, UID, B, "before")
        ids = [c for c in flt.must_not if isinstance(c, HasIdCondition)]
        assert ids and ids[0].has_id == [UID]

    def test_no_exclude_id_means_only_tombstone_exclusion(self, service):
        flt = service._timeline_filter("u", None, None, B, "after")
        assert len(flt.must_not) == 1 and flt.must_not[0].key == "metadata.dream_tombstoned"

    def test_project_adds_dual_scope_clause(self, service):
        without = service._timeline_filter("u", None, None, B, "before")
        with_p = service._timeline_filter("u", "proj", None, B, "before")
        assert len(with_p.must) == len(without.must) + 1
        keys = _keys(with_p)
        assert "metadata.project_id" in keys and "metadata.scope" in keys

    def test_standards_visibility_only_when_enabled(self, service, monkeypatch):
        monkeypatch.setattr(reads.settings, "standards_enabled", False)
        off = service._timeline_filter("u", None, None, B, "before")
        monkeypatch.setattr(reads.settings, "standards_enabled", True)
        on = service._timeline_filter("u", None, None, B, "before")
        assert len(on.must[0].should) == len(off.must[0].should) + 1
        assert on.must[0].should[-1].match.value == "standard"

    def test_visibility_always_has_owner_and_shared(self, service, monkeypatch):
        monkeypatch.setattr(reads.settings, "standards_enabled", False)
        flt = service._timeline_filter("alice", None, None, B, "before")
        should = flt.must[0].should
        assert FieldCondition(key="user_id", match=MatchValue(value="alice")) in should
        assert any(c.key == "metadata.visibility" and c.match.value == "shared" for c in should)


class TestTimelineWindow:
    def test_orders_ascending_even_for_before_side(self, service):
        # order_by DESC on the before side returns newest-first; window sorts asc
        _client(service).scroll.return_value = (
            [_pt("n", "2026-06-30T00:00:00Z"), _pt("o", "2026-06-01T00:00:00Z")], None,
        )
        out = service._timeline_window("u", None, None, B, "before", 5)
        assert [r.id for r in out] == ["o", "n"]

    def test_before_direction_requests_desc_after_requests_asc(self, service):
        service._timeline_window("u", None, None, B, "before", 3)
        service._timeline_window("u", None, None, B, "after", 3)
        calls = _client(service).scroll.call_args_list
        assert calls[0].kwargs["order_by"].direction.value == "desc"
        assert calls[1].kwargs["order_by"].direction.value == "asc"
        assert calls[0].kwargs["limit"] == 3

    def test_order_by_failure_uses_python_fallback(self, service):
        pts = [_pt("a", "2026-06-29T00:00:00Z"), _pt("b", "2026-06-30T00:00:00Z")]

        def scroll(**kw):
            if kw.get("order_by") is not None:
                raise RuntimeError("no index")
            return pts, None

        _client(service).scroll.side_effect = scroll
        out = service._timeline_window("u", None, None, B, "before", 1)
        assert [r.id for r in out] == ["b"]

    def test_none_points_tolerated(self, service):
        _client(service).scroll.return_value = (None, None)
        assert service._timeline_window("u", None, None, B, "after", 3) == []


# ── timeline ───────────────────────────────────────────────────────────


def _resp(rid, created_at="2026-07-01T00:00:00Z", source="vector"):
    r = MemoryResponse(id=rid, memory="m", created_at=created_at)
    r.source = source
    return r


class TestTimeline:
    @pytest.mark.parametrize("anchor", ["", "   ", None])
    def test_blank_anchor_raises(self, service, anchor):
        with pytest.raises(ValueError, match="anchor"):
            service.timeline(anchor, "u")

    def test_id_anchor_resolves_through_batch_get(self, service, monkeypatch):
        seen = {}

        def fake(ids, caller):
            seen["args"] = (ids, caller)
            return {"results": [_resp(UID)], "missing": []}

        monkeypatch.setattr(service, "get_memories_by_ids", fake)
        monkeypatch.setattr(service, "_timeline_window", lambda *a: [])
        out = service.timeline(f"  {UID} ", "alice")
        assert seen["args"] == ([UID], "alice")
        assert out["anchor_id"] == UID and [m.id for m in out["memories"]] == [UID]

    def test_unreadable_id_anchor_returns_none(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "get_memories_by_ids", lambda ids, caller: {"results": [], "missing": ids}
        )
        assert service.timeline(UID, "alice") is None

    def test_query_anchor_picks_first_vector_hit_with_created_at(self, service, monkeypatch):
        hits = [
            _resp("g1", source="graph"),
            _resp("nodate", created_at=None),
            _resp("v1"),
            _resp("v2"),
        ]
        called = {}

        def fake_search(**kw):
            called.update(kw)
            return hits

        monkeypatch.setattr(service, "search", fake_search)
        monkeypatch.setattr(service, "_timeline_window", lambda *a: [])
        out = service.timeline("what database", "alice", project_id="p")
        assert out["anchor_id"] == "v1"
        assert called["query"] == "what database" and called["project_id"] == "p"
        assert called["limit"] == 5

    def test_query_with_no_usable_hit_returns_none(self, service, monkeypatch):
        monkeypatch.setattr(service, "search", lambda **kw: [_resp("g", source="graph")])
        assert service.timeline("anything", "u") is None

    def test_window_arguments_and_order(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "get_memories_by_ids",
            lambda ids, caller: {"results": [_resp(UID, "2026-07-01T12:00:00Z")], "missing": []},
        )
        sides = []

        def window(user_id, project_id, exclude_id, boundary, direction, limit):
            sides.append((user_id, project_id, exclude_id, boundary, direction, limit))
            return [_resp(f"{direction}-1")]

        monkeypatch.setattr(service, "_timeline_window", window)
        out = service.timeline(UID, "alice", depth=4, project_id="proj")
        assert [m.id for m in out["memories"]] == ["before-1", UID, "after-1"]
        assert sides[0][4] == "before" and sides[1][4] == "after"
        assert sides[0][:3] == ("alice", "proj", UID) and sides[0][5] == 4
        assert sides[0][3] == datetime(2026, 7, 1, 12, tzinfo=timezone.utc)

    @pytest.mark.parametrize(
        "depth,expected", [(0, 1), (-5, 1), (1, 1), (7, 7), (999, reads.TIMELINE_MAX_DEPTH)]
    )
    def test_depth_clamped(self, service, monkeypatch, depth, expected):
        monkeypatch.setattr(
            service, "get_memories_by_ids",
            lambda ids, caller: {"results": [_resp(UID)], "missing": []},
        )
        limits = []
        monkeypatch.setattr(service, "_timeline_window", lambda *a: limits.append(a[5]) or [])
        service.timeline(UID, "u", depth=depth)
        assert limits == [expected, expected]

    def test_naive_timestamp_assumed_utc(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "get_memories_by_ids",
            lambda ids, caller: {"results": [_resp(UID, "2026-07-01T00:00:00")], "missing": []},
        )
        bounds = []
        monkeypatch.setattr(service, "_timeline_window", lambda *a: bounds.append(a[3]) or [])
        service.timeline(UID, "u")
        assert bounds[0].tzinfo == timezone.utc

    def test_unparseable_timestamp_degrades_to_anchor_alone(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "get_memories_by_ids",
            lambda ids, caller: {"results": [_resp(UID, "not a date")], "missing": []},
        )
        called = []
        monkeypatch.setattr(service, "_timeline_window", lambda *a: called.append(1) or [])
        out = service.timeline(UID, "u")
        assert called == []
        assert [m.id for m in out["memories"]] == [UID]

    def test_missing_timestamp_on_id_anchor_degrades_to_anchor_alone(self, service, monkeypatch):
        monkeypatch.setattr(
            service, "get_memories_by_ids",
            lambda ids, caller: {"results": [_resp(UID, None)], "missing": []},
        )
        called = []
        monkeypatch.setattr(service, "_timeline_window", lambda *a: called.append(1) or [])
        out = service.timeline(UID, "u")
        assert called == [] and out["anchor_id"] == UID

    def test_non_numeric_depth_raises(self, service):
        with pytest.raises((ValueError, TypeError)):
            service.timeline(UID, "u", depth="deep")


# ── list_memories ──────────────────────────────────────────────────────


class TestListMemories:
    def test_default_filters_only_user(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories("alice")
        kw = service._memory.get_all.call_args.kwargs
        assert kw["filters"] == {"user_id": "alice"} and kw["top_k"] == 100

    def test_all_filters_applied(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories(
            "alice", scope="project", category="tech_stack", project_id="p",
            limit=7, agent_id="agent-1", run_id="run-9",
        )
        kw = service._memory.get_all.call_args.kwargs
        assert kw["filters"] == {
            "user_id": "alice",
            "metadata.scope": "project",
            "metadata.category": "tech_stack",
            "metadata.project_id": "p",
            "agent_id": "agent-1",
            "run_id": "run-9",
        }
        assert kw["top_k"] == 7

    def test_agent_only_narrowing(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories("alice", agent_id="a")
        assert service._memory.get_all.call_args.kwargs["filters"] == {"user_id": "alice", "agent_id": "a"}

    def test_run_only_narrowing(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories("alice", run_id="r")
        assert service._memory.get_all.call_args.kwargs["filters"] == {"user_id": "alice", "run_id": "r"}

    def test_empty_string_filters_ignored(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories("alice", scope="", category="", project_id="", agent_id="", run_id="")
        assert service._memory.get_all.call_args.kwargs["filters"] == {"user_id": "alice"}

    def test_tombstones_hidden_by_default(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "a", "memory": "live", "metadata": {}},
                {"id": "b", "memory": "dead", "metadata": {"dream_tombstoned": True}},
                {"id": "c", "memory": "dead2", "metadata": {"metadata": {"dream_tombstoned": True}}},
            ]
        }
        assert [r.id for r in service.list_memories("u")] == ["a"]

    def test_include_tombstoned_returns_raw_set(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "a", "memory": "live", "metadata": {}},
                {"id": "b", "memory": "dead", "metadata": {"dream_tombstoned": True}},
            ]
        }
        assert [r.id for r in service.list_memories("u", include_tombstoned=True)] == ["a", "b"]

    def test_accepts_bare_list_result(self, service):
        service._memory.get_all.return_value = [{"id": "a", "memory": "x", "metadata": {}}]
        assert [r.id for r in service.list_memories("u")] == ["a"]

    def test_unknown_result_shape_is_empty(self, service):
        service._memory.get_all.return_value = None
        assert service.list_memories("u") == []

    def test_ensures_filter_indexes(self, service, monkeypatch):
        called = []
        monkeypatch.setattr(service, "_ensure_filter_indexes", lambda: called.append(1))
        service._memory.get_all.return_value = {"results": []}
        service.list_memories("u")
        assert called == [1]


# ── list_projects ──────────────────────────────────────────────────────


class TestListProjects:
    def _wire(self, service, records):
        service._run_on_bridge = MagicMock(side_effect=lambda coro, timeout=30.0: (coro.close(), records)[1])

    def test_no_graph_returns_empty(self, service):
        service._graphiti = None
        service._memory.graph = None
        assert service.list_projects("u") == []

    def test_distinct_sorted_and_namespaces_merged(self, service):
        self._wire(service, [
            {"group_id": "user--u--project--zeta"},
            {"group_id": "shared--project--alpha"},
            {"group_id": "user--u--project--alpha"},
        ])
        assert service.list_projects("u") == ["alpha", "zeta"]

    def test_global_groups_and_blank_ids_skipped(self, service):
        self._wire(service, [
            {"group_id": "user--u"},
            {"group_id": "shared"},
            {"group_id": "user--u--project--   "},
            {"group_id": ""},
            {"group_id": None},
            {},
        ])
        assert service.list_projects("u") == []

    def test_project_ids_containing_separator_preserved(self, service):
        self._wire(service, [{"group_id": "user--u--project--a--project--b"}])
        assert service.list_projects("u") == ["a--project--b"]

    def test_none_records_treated_as_empty(self, service):
        self._wire(service, None)
        assert service.list_projects("u") == []

    def test_bridge_error_returns_empty_and_closes_coroutine(self, service):
        captured = {}

        def boom(coro, timeout=30.0):
            captured["coro"] = coro
            raise TimeoutError("neo4j slow")

        service._run_on_bridge = boom
        assert service.list_projects("u") == []
        # closed: awaiting-state is gone, so close() is a no-op and cr_frame is None
        assert captured["coro"].cr_frame is None

    def test_query_parameters_carry_prefixes_and_timeout(self, service):
        seen = {}

        def capture(coro, timeout=30.0):
            seen["timeout"] = timeout
            coro.close()
            return []

        service._run_on_bridge = capture
        service.list_projects("alice")
        assert seen["timeout"] == 10.0


# ── _list_null_category_memories ───────────────────────────────────────


class TestListNullCategory:
    def _bind(self, service):
        # method reads self._memory directly (already set by the fixture)
        return _client(service)

    def test_filter_has_user_and_null_category(self, service):
        client = self._bind(service)
        service._list_null_category_memories("alice")
        flt = client.scroll.call_args.kwargs["scroll_filter"]
        assert FieldCondition(key="user_id", match=MatchValue(value="alice")) in flt.must
        assert len(flt.must) == 2

    def test_scope_and_project_conditions_added(self, service):
        client = self._bind(service)
        service._list_null_category_memories("alice", scope="project", project_id="p")
        flt = client.scroll.call_args.kwargs["scroll_filter"]
        assert len(flt.must) == 4
        assert FieldCondition(key="metadata.scope", match=MatchValue(value="project")) in flt.must
        assert FieldCondition(key="metadata.project_id", match=MatchValue(value="p")) in flt.must

    def test_pages_until_exhausted(self, service):
        client = self._bind(service)
        pages = iter([([_pt("a")], "tok"), ([_pt("b")], None)])
        client.scroll.side_effect = lambda **kw: next(pages)
        out = service._list_null_category_memories("u")
        assert [r["id"] for r in out] == ["a", "b"]

    def test_row_shape(self, service):
        client = self._bind(service)
        client.scroll.return_value = ([_pt("a", data="hello", metadata={"scope": "global"})], None)
        assert service._list_null_category_memories("u") == [
            {"id": "a", "data": "hello", "metadata": {"scope": "global"}}
        ]

    def test_missing_payload_fields_defaulted(self, service):
        client = self._bind(service)
        client.scroll.return_value = ([SimpleNamespace(id=5, payload=None)], None)
        assert service._list_null_category_memories("u") == [{"id": "5", "data": "", "metadata": {}}]

    def test_limit_bounds_page_size_and_total(self, service):
        client = self._bind(service)
        sizes = []

        def scroll(**kw):
            sizes.append(kw["limit"])
            return [_pt(f"p{len(sizes)}-{i}") for i in range(kw["limit"])], "more"

        client.scroll.side_effect = scroll
        out = service._list_null_category_memories("u", limit=130)
        assert len(out) == 130
        assert sizes == [100, 30]

    def test_limit_zero_never_scrolls(self, service):
        client = self._bind(service)
        assert service._list_null_category_memories("u", limit=0) == []
        client.scroll.assert_not_called()


# ── get_all_user_ids ───────────────────────────────────────────────────


class TestGetAllUserIds:
    def test_union_across_pages_deduped(self, service):
        client = _client(service)
        pages = iter([
            ([_pt("1", user_id="a"), _pt("2", user_id="b")], "t"),
            ([_pt("3", user_id="a"), _pt("4", user_id="c")], None),
        ])
        client.scroll.side_effect = lambda **kw: next(pages)
        assert sorted(service.get_all_user_ids()) == ["a", "b", "c"]

    def test_blank_and_missing_user_ids_skipped(self, service):
        client = _client(service)
        client.scroll.return_value = (
            [_pt("1", user_id=""), _pt("2", user_id=None), SimpleNamespace(id="3", payload=None)],
            None,
        )
        assert service.get_all_user_ids() == []

    def test_payload_projection_and_batch_size(self, service):
        client = _client(service)
        client.scroll.return_value = ([], None)
        service.get_all_user_ids(batch_size=17)
        kw = client.scroll.call_args.kwargs
        assert kw["with_payload"] == ["user_id"] and kw["with_vectors"] is False
        assert kw["limit"] == 17

    def test_offset_threaded_between_pages(self, service):
        client = _client(service)
        offsets = []
        pages = iter([([_pt("1", user_id="a")], "tok"), ([], None)])

        def scroll(**kw):
            offsets.append(kw["offset"])
            return next(pages)

        client.scroll.side_effect = scroll
        service.get_all_user_ids()
        assert offsets == [None, "tok"]

    def test_zero_offset_continues_pagination(self, service):
        client = _client(service)
        pages = iter([([_pt("1", user_id="a")], 0), ([_pt("2", user_id="b")], None)])
        client.scroll.side_effect = lambda **kw: next(pages)
        assert sorted(service.get_all_user_ids()) == ["a", "b"]
