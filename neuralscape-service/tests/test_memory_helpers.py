"""Direct unit tests for the small ``memory/*`` leaf modules: hashing, junk, audit, retry, convert."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import settings
from memory import hashing, junk, retry
from memory.convert import ConvertMixin
from schemas import MemoryResponse

UTC = timezone.utc


# ── hashing ──


class TestParseExpiresAt:
    @pytest.mark.parametrize("value", [None, "", "   ", "not-a-date", 12345, ["2026-01-01"], {"a": 1}])
    def test_unusable_values_are_none(self, value):
        assert hashing._parse_expires_at(value) is None

    def test_z_suffix_is_utc(self):
        assert hashing._parse_expires_at("2026-12-01T10:00:00Z") == datetime(2026, 12, 1, 10, tzinfo=UTC)

    def test_offsets_compare_correctly(self):
        a = hashing._parse_expires_at("2026-12-01T10:00:00+00:00")
        b = hashing._parse_expires_at("2026-12-01T05:00:00-05:00")
        assert a == b

    def test_naive_string_and_naive_datetime_are_treated_as_utc(self):
        assert hashing._parse_expires_at("2026-12-01T10:00:00").tzinfo is not None
        naive = datetime(2026, 12, 1, 10)
        assert hashing._parse_expires_at(naive) == naive.replace(tzinfo=UTC)

    def test_aware_datetime_is_returned_unchanged(self):
        aware = datetime(2026, 12, 1, 10, tzinfo=timezone(timedelta(hours=2)))
        assert hashing._parse_expires_at(aware) is aware

    def test_surrounding_whitespace_is_ignored(self):
        assert hashing._parse_expires_at("  2026-12-01T10:00:00Z  ") == datetime(2026, 12, 1, 10, tzinfo=UTC)


def test_content_hash_is_the_md5_hex_of_the_utf8_text():
    assert hashing.content_hash("hello") == hashlib.md5(b"hello").hexdigest()
    assert hashing.content_hash("héllo ✓") == hashlib.md5("héllo ✓".encode()).hexdigest()
    # No normalisation at all: case and whitespace are part of the identity.
    assert hashing.content_hash("a") != hashing.content_hash("a ")
    assert hashing.content_hash("Hello") != hashing.content_hash("hello")
    assert hashing.content_hash("a b") != hashing.content_hash("a  b")
    assert hashing.content_hash("a\n") != hashing.content_hash("a")


def test_infer_project_id_matches_known_slugs_case_insensitively(monkeypatch):
    monkeypatch.setattr(type(settings), "known_projects", property(lambda self: ["alpha-proj", "beta"]))
    assert hashing._infer_project_id("Working on ALPHA-PROJ today") == "alpha-proj"
    assert hashing._infer_project_id("nothing relevant") is None


def test_infer_project_id_first_configured_slug_wins(monkeypatch):
    monkeypatch.setattr(type(settings), "known_projects", property(lambda self: ["beta", "alpha"]))
    assert hashing._infer_project_id("alpha and beta") == "beta"


class TestCreatedAtKey:
    def test_orders_across_offsets_chronologically(self):
        earlier = "2026-01-01T12:00:00+05:00"   # 07:00Z
        later = "2026-01-01T09:00:00+00:00"     # 09:00Z, but "smaller" as a string than "12:..."
        assert hashing._created_at_key(earlier) < hashing._created_at_key(later)
        assert sorted([later, earlier], key=hashing._created_at_key) == [earlier, later]

    @pytest.mark.parametrize("value", [None, "", "garbage"])
    def test_missing_or_unparseable_sorts_first(self, value):
        assert hashing._created_at_key(value) < hashing._created_at_key("1999-01-01T00:00:00Z")

    def test_z_suffix_and_naive_are_utc(self):
        assert hashing._created_at_key("2026-01-01T00:00:00Z") == hashing._created_at_key("2026-01-01T00:00:00")


# ── junk ──


class TestIsJunkFact:
    @pytest.mark.parametrize("text", [
        "Ran command: pytest -q", "Edited file: main.py", "Wrote file main.py", "Read file: x.py",
        "Created file x", "Deleted file: y", "Launched explorer task: scan", "Tool result: ok",
        "Command output: 3 passed", "ran command: lower-case also junk", "  Ran command: leading space",
    ])
    def test_event_log_lines_are_junk(self, text):
        assert junk._is_junk_fact(text)

    @pytest.mark.parametrize("text", ["", "short", "         ", "123456789", "  123456789  "])
    def test_too_short_is_junk(self, text):
        assert junk._is_junk_fact(text)

    def test_the_length_boundary_is_ten_stripped_characters(self):
        assert junk._is_junk_fact("a" * 9) and not junk._is_junk_fact("a" * 10)
        assert not junk._is_junk_fact("  " + "a" * 10 + "  ")

    @pytest.mark.parametrize("text", [
        "Uses Python 3.12 with FastAPI for backend services",
        "The team ran command line tools yesterday and liked them",  # not at line start
        "Prefers dark mode in every editor",
    ])
    def test_real_knowledge_is_kept(self, text):
        assert not junk._is_junk_fact(text)

    def test_a_junk_line_anywhere_in_a_multiline_fact_is_junk(self):
        assert junk._is_junk_fact("Useful sentence about the project.\nRan command: ls")


class TestDeletedMsg:
    def test_plain(self):
        assert junk._deleted_msg("memories", 3, 0, 0) == "Deleted 3 memories"

    def test_names_each_preserved_tier_separately(self):
        assert junk._deleted_msg("memories", 3, 2, 1) == "Deleted 3 memories (preserved 2 shared, 1 standard)"
        assert junk._deleted_msg("memories", 0, 0, 4) == "Deleted 0 memories (preserved 4 standard)"
        assert junk._deleted_msg("memories", 1, 5, 0) == "Deleted 1 memories (preserved 5 shared)"


class TestCleanConversationForGraph:
    def test_junk_lines_are_removed_and_the_rest_kept(self):
        msgs = [{"role": "user", "content": "Ran command: ls\nI prefer tabs.\nTool result: ok"}]
        assert junk._clean_conversation_for_graph(msgs) == [{"role": "user", "content": "I prefer tabs."}]

    def test_a_message_that_becomes_empty_is_dropped(self):
        msgs = [{"role": "assistant", "content": "Ran command: ls\nCommand output: x"}, {"role": "user", "content": "hi there"}]
        assert junk._clean_conversation_for_graph(msgs) == [{"role": "user", "content": "hi there"}]

    def test_messages_without_content_pass_through_untouched(self):
        msgs = [{"role": "user", "content": ""}, {"role": "user"}]
        assert junk._clean_conversation_for_graph(msgs) == msgs

    def test_indented_junk_lines_and_surrounding_blank_lines_are_handled(self):
        msgs = [{"role": "user", "content": "\n\n  Ran command: x\nkeep this line\n   \n"}]
        assert junk._clean_conversation_for_graph(msgs) == [{"role": "user", "content": "keep this line"}]

    def test_other_keys_are_preserved_and_input_not_mutated(self):
        msg = {"role": "user", "content": "Read file: a\nkeep me", "name": "n"}
        out = junk._clean_conversation_for_graph([msg])
        assert out == [{"role": "user", "content": "keep me", "name": "n"}]
        assert msg["content"].startswith("Read file")


# ── audit ──


def test_audit_logger_is_a_usable_structlog_logger():
    from memory import audit

    import structlog

    with structlog.testing.capture_logs() as logs:
        audit._audit_log.info("authoritative_context_served", user_id="u")
        audit._audit_log.bind(tier="standard").info("served")
    assert logs[0]["event"] == "authoritative_context_served" and logs[0]["user_id"] == "u"
    assert logs[1]["tier"] == "standard"


# ── retry ──


class TestIsTransient:
    @pytest.mark.parametrize("msg", [
        "503 Service Unavailable", "HTTP 429", "UNAVAILABLE: try later", "RESOURCE_EXHAUSTED",
        "Rate Limit hit", "model is overloaded", "no capacity", "request timed out", "Timeout reading",
    ])
    def test_transient(self, msg):
        assert retry._is_transient(Exception(msg))

    @pytest.mark.parametrize("msg", ["400 bad request", "KeyError: 'x'", "invalid api key", ""])
    def test_permanent(self, msg):
        assert not retry._is_transient(Exception(msg))


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    monkeypatch.setattr(retry.random, "uniform", lambda a, b: 0.0)
    return sleeps


class TestRetryTransient:
    def test_returns_first_success_without_sleeping(self, no_sleep):
        fn = MagicMock(return_value=7)
        assert retry.retry_transient(fn, 1, k=2, max_retries=3) == 7
        fn.assert_called_once_with(1, k=2)
        assert no_sleep == []

    def test_retries_transient_errors_with_exponential_capped_backoff(self, no_sleep):
        fn = MagicMock(side_effect=[Exception("503"), Exception("503"), Exception("503"), "ok"])
        out = retry.retry_transient(fn, max_retries=5, base_delay=1.0, max_delay=3.0)
        assert out == "ok" and fn.call_count == 4
        assert no_sleep == [1.0, 2.0, 3.0]  # 1, 2, then 4 capped at 3

    def test_non_transient_error_is_raised_immediately(self, no_sleep):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError):
            retry.retry_transient(fn, max_retries=5)
        fn.assert_called_once()
        assert no_sleep == []

    def test_exhausted_retries_raise_the_last_error(self, no_sleep):
        fn = MagicMock(side_effect=[Exception("503 a"), Exception("503 b")])
        with pytest.raises(Exception, match="503 b"):
            retry.retry_transient(fn, max_retries=1, base_delay=0.1, max_delay=1)
        assert fn.call_count == 2 and len(no_sleep) == 1

    def test_falls_back_to_the_fallback_model_after_exhaustion(self, no_sleep):
        calls = []

        def fn(model=None):
            calls.append(model)
            if model == "primary":
                raise Exception("429 rate limit")
            return f"ok:{model}"

        out = retry.retry_transient(fn, max_retries=1, base_delay=0.1, max_delay=1, model="primary", fallback_model="backup")
        assert out == "ok:backup" and calls == ["primary", "primary", "backup"]

    def test_a_failing_fallback_raises_its_error(self, no_sleep):
        def fn(model=None):
            raise Exception("503 down" if model == "primary" else "503 still down")

        with pytest.raises(Exception, match="still down"):
            retry.retry_transient(fn, max_retries=0, model="primary", fallback_model="backup")

    def test_no_fallback_when_models_are_equal_or_kwarg_absent(self, no_sleep):
        fn = MagicMock(side_effect=Exception("503 primary"))
        with pytest.raises(Exception, match="503 primary"):
            retry.retry_transient(fn, max_retries=0, model="m", fallback_model="m")
        assert fn.call_count == 1
        # kwarg absent: the ORIGINAL error surfaces (not a KeyError from a blind fallback lookup)
        fn2 = MagicMock(side_effect=Exception("503 original"))
        with pytest.raises(Exception, match="503 original") as exc:
            retry.retry_transient(fn2, max_retries=0, fallback_model="backup")
        assert exc.type is Exception and fn2.call_count == 1

    def test_fallback_forwards_the_model_kwarg_and_other_arguments(self, no_sleep):
        seen = []

        def fn(a, model=None, extra=None):
            seen.append((a, model, extra))
            if model == "primary":
                raise Exception("503")
            return "ok"

        assert retry.retry_transient(fn, 1, max_retries=0, model="primary", extra="x", fallback_model="backup") == "ok"
        assert seen == [(1, "primary", "x"), (1, "backup", "x")]

    def test_a_custom_model_kwarg_name_is_honoured(self, no_sleep):
        seen = []

        def fn(engine=None):
            seen.append(engine)
            if engine == "p":
                raise Exception("503")
            return "ok"

        assert retry.retry_transient(fn, max_retries=0, engine="p", fallback_model="b", model_kwarg="engine") == "ok"
        assert seen == ["p", "b"]

    def test_jitter_is_added_below_the_cap(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(retry.time, "sleep", sleeps.append)
        monkeypatch.setattr(retry.random, "uniform", lambda a, b: 0.5)
        fn = MagicMock(side_effect=[Exception("503"), "ok"])
        retry.retry_transient(fn, max_retries=2, base_delay=1.0, max_delay=10.0)
        assert sleeps == [1.5]

    def test_defaults_come_from_settings(self, no_sleep, monkeypatch):
        monkeypatch.setattr(settings, "llm_max_retries", 2)
        monkeypatch.setattr(settings, "llm_retry_base_delay", 5.0)
        monkeypatch.setattr(settings, "llm_retry_max_delay", 60.0)
        fn = MagicMock(side_effect=[Exception("503"), Exception("503"), "ok"])
        assert retry.retry_transient(fn) == "ok"
        assert no_sleep == [5.0, 10.0]


# ── convert ──


class _Conv(ConvertMixin):
    """The mixin has no state; a bare subclass is all it needs."""


@pytest.fixture
def conv():
    return _Conv()


class TestPayloadReadableBy:
    def test_owner_by_top_level_user_id(self):
        assert ConvertMixin._payload_readable_by({"user_id": "alice"}, "alice")
        assert not ConvertMixin._payload_readable_by({"user_id": "alice"}, "bob")

    def test_owner_by_metadata_owner_user_id_including_double_nested(self):
        assert ConvertMixin._payload_readable_by({"metadata": {"owner_user_id": "alice"}}, "alice")
        assert ConvertMixin._payload_readable_by({"metadata": {"metadata": {"owner_user_id": "alice"}}}, "alice")

    def test_shared_is_readable_by_everyone_even_anonymous(self):
        assert ConvertMixin._payload_readable_by({"metadata": {"visibility": "shared"}}, "bob")
        assert ConvertMixin._payload_readable_by({"metadata": {"visibility": "shared"}}, None)

    def test_standard_requires_the_tier_to_be_enabled(self, monkeypatch):
        payload = {"metadata": {"visibility": "standard"}}
        monkeypatch.setattr(settings, "standards_enabled", True)
        assert ConvertMixin._payload_readable_by(payload, "bob")
        monkeypatch.setattr(settings, "standards_enabled", False)
        assert not ConvertMixin._payload_readable_by(payload, "bob")

    def test_private_and_legacy_rows_are_not_readable_by_others(self):
        assert not ConvertMixin._payload_readable_by({"metadata": {"visibility": "private"}, "user_id": "alice"}, "bob")
        assert not ConvertMixin._payload_readable_by({}, "bob")
        assert not ConvertMixin._payload_readable_by({"user_id": "alice"}, None)

    @pytest.mark.parametrize("caller", [None, ""])
    @pytest.mark.parametrize("payload", [
        {}, {"metadata": {}}, {"metadata": {"visibility": "private"}},
        {"metadata": {"metadata": {}}}, {"user_id": None, "metadata": {"owner_user_id": None}},
        {"user_id": "", "metadata": {"owner_user_id": ""}},
    ])
    def test_an_anonymous_caller_never_matches_a_row_without_an_owner(self, payload, caller):
        assert not ConvertMixin._payload_readable_by(payload, caller)

    def test_an_anonymous_caller_reads_only_shared_and_enabled_standard(self, monkeypatch):
        monkeypatch.setattr(settings, "standards_enabled", True)
        assert ConvertMixin._payload_readable_by({"metadata": {"visibility": "standard"}}, None)
        monkeypatch.setattr(settings, "standards_enabled", False)
        assert not ConvertMixin._payload_readable_by({"metadata": {"visibility": "standard"}}, None)
        assert not ConvertMixin._payload_readable_by({"user_id": "alice", "metadata": {"visibility": "private"}}, None)


class TestMemToResponse:
    def test_unwraps_the_double_nested_mem0_metadata(self, conv):
        mem = {"id": "m1", "memory": "text", "score": 0.5, "created_at": "c", "updated_at": "u",
               "metadata": {"metadata": {"category": "preference", "scope": "global", "tags": ["a"], "workspace": "ref-x",
                                          "visibility": "shared", "owner_user_id": "alice", "revision": 3}}}
        r = conv._mem_to_response(mem)
        assert (r.id, r.memory, r.category, r.scope, r.tags, r.workspace) == ("m1", "text", "preference", "global", ["a"], "ref-x")
        assert (r.visibility, r.owner_user_id, r.revision, r.score, r.source) == ("shared", "alice", "3", 0.5, "vector")

    def test_every_mapped_field_comes_through(self, conv):
        meta = {
            "category": "decision", "scope": "project", "project_id": "p1", "tags": ["a", "b"], "occurred_at": "2026-01-02T00:00:00Z",
            "domain": "coding", "observation_type": "decision", "concepts": ["c1"], "source_type": "tool_extraction",
            "related_memory_ids": ["r1"], "confidence": 0.8, "expires_at": "2027-01-01T00:00:00Z", "derived_from": ["d1"],
            "epistemic_level": "explicit", "memory_kind": "fact", "source_ref": {"connector_id": "c"}, "visibility": "shared",
            "owner_user_id": "alice", "revision": 4, "workspace": "ref-x", "sensitivity": "high", "sensitivity_source": "auto",
            "title": "T", "token_estimate": 42, "speaker": "Mina",
        }
        r = conv._mem_to_response({"id": "m1", "memory": "text", "score": 0.7, "created_at": "c", "updated_at": "u", "metadata": meta})
        assert (r.id, r.memory, r.score, r.created_at, r.updated_at, r.source) == ("m1", "text", 0.7, "c", "u", "vector")
        for field in ("category", "scope", "project_id", "tags", "occurred_at", "domain", "observation_type", "concepts",
                      "source_type", "related_memory_ids", "confidence", "expires_at", "derived_from", "epistemic_level",
                      "memory_kind", "source_ref", "visibility", "owner_user_id", "workspace", "sensitivity",
                      "sensitivity_source", "title", "token_estimate", "speaker"):
            assert getattr(r, field) == meta[field], field
        assert r.revision == "4"

    def test_flat_metadata_also_works_and_missing_fields_are_none(self, conv):
        r = conv._mem_to_response({"id": "m", "memory": "t", "metadata": {"category": "decision"}})
        assert r.category == "decision" and r.scope is None and r.workspace is None and r.revision == "0"

    def test_legacy_owner_falls_back_to_the_payload_user_id(self, conv):
        assert conv._mem_to_response({"id": "m", "memory": "t", "user_id": "legacy"}).owner_user_id == "legacy"
        assert conv._mem_to_response({"id": "m", "memory": "t", "user_id": ""}).owner_user_id is None
        assert conv._mem_to_response({"id": "m", "memory": "t", "user_id": "u", "metadata": {"owner_user_id": "o"}}).owner_user_id == "o"

    def test_empty_input_yields_an_empty_response(self, conv):
        r = conv._mem_to_response({})
        assert r.id == "" and r.memory == ""

    def test_malformed_category_evidence_is_dropped(self, conv):
        r = conv._mem_to_response({"id": "m", "memory": "t", "metadata": {"category": "decision", "category_evidence": {"x": 1}}})
        assert r.category_evidence is None


class TestPointToResponse:
    def test_maps_a_qdrant_point(self, conv):
        point = SimpleNamespace(id=42, payload={"data": "fact", "user_id": "alice", "created_at": "c", "updated_at": "u",
                                                "metadata": {"category": "convention", "memory_kind": "passage"}})
        r = conv._point_to_response(point)
        assert (r.id, r.memory, r.category, r.memory_kind, r.owner_user_id, r.created_at) == ("42", "fact", "convention", "passage", "alice", "c")

    def test_point_without_payload(self, conv):
        r = conv._point_to_response(SimpleNamespace(id="p", payload=None))
        assert r.id == "p" and r.memory == ""


class TestResultConversion:
    def test_extract_memory_list_handles_dict_list_and_garbage(self, conv):
        assert conv._extract_memory_list({"results": [{"id": "a"}]}) == [{"id": "a"}]
        assert conv._extract_memory_list({"other": 1}) == []
        assert conv._extract_memory_list([{"id": "b"}]) == [{"id": "b"}]
        assert conv._extract_memory_list(None) == []
        assert conv._extract_memory_list("x") == []

    def test_result_to_responses_fills_missing_category_and_scope_only(self, conv):
        result = {"results": [
            {"id": "a", "memory": "x", "metadata": {}},
            {"id": "b", "memory": "y", "metadata": {"category": "decision", "scope": "project"}},
        ]}
        a, b = conv._result_to_responses(result, category="preference", scope="global")
        assert (a.category, a.scope) == ("preference", "global")
        assert (b.category, b.scope) == ("decision", "project")

    def test_results_to_responses_does_not_default(self, conv):
        (a,) = conv._results_to_responses([{"id": "a", "memory": "x"}])
        assert isinstance(a, MemoryResponse) and a.category is None


class TestDuplicateContent:
    def test_normalize(self):
        assert ConvertMixin.normalize_memory_content("  Hello World \n") == "hello world"
        assert ConvertMixin.normalize_memory_content(None) == ""

    def test_find_duplicate_exact_and_substring_either_way(self):
        find = ConvertMixin.find_duplicate_content
        assert find("a b c", ["x", "a b c"]) == "a b c"
        assert find("a b", ["xx a b yy"]) == "xx a b yy"        # candidate contains it
        assert find("xx a b yy", ["a b"]) == "a b"              # it contains the candidate
        assert find("alpha", ["beta", "gamma"]) is None
        assert find("alpha", []) is None

    def test_first_matching_candidate_wins(self):
        assert ConvertMixin.find_duplicate_content("a", ["zz", "ab", "a"]) == "ab"


# ── remediation (pure helpers not covered by tests/test_remediation.py) ──


class TestHeuristicHit:
    def test_reports_kind_uuid_first_matching_token_and_a_200_char_snippet(self):
        from memory.remediation import RemediationMixin

        hit = RemediationMixin._heuristic_hit(
            "edge", {"uuid": "u1", "text": "$1,234.56 " + "x" * 250}, ["$9", "$1,234.56"]
        )
        assert (hit["kind"], hit["uuid"], hit["matched_token"]) == ("edge", "u1", "$1,234.56")
        assert len(hit["snippet"]) == 200

    def test_matched_token_is_the_first_token_present_in_the_text(self):
        from memory.remediation import RemediationMixin

        hit = RemediationMixin._heuristic_hit("entity", {"uuid": "u2", "text": "acct 1234567 and $5.00"}, ["$5.00", "1234567"])
        assert hit["matched_token"] == "$5.00"

    def test_missing_text_is_empty_not_an_error(self):
        from memory.remediation import RemediationMixin

        assert RemediationMixin._heuristic_hit("episode", {"uuid": "u3"}, ["x"]) == {
            "kind": "episode", "uuid": "u3", "matched_token": None, "snippet": "",
        }
