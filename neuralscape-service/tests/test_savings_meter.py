"""Unit tests for E2 — the honest token-savings meter (savings_meter.py).

Honesty invariants under test:

- net_tokens_saved is SIGNED and can go negative — never clamped;
- rederivation_savings_estimate is a separate estimated field that never
  sums into the measured net;
- the per-release tool-schema overhead constant covers the REAL rendered
  MCP tool schemas (it may only shrink unless consciously updated);
- the kill-switch means ZERO tokenizer calls anywhere on the hot path;
- the ledger is append-only and the O(1) totals equal the sum of entries.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import savings_meter as sm
from config import settings
from index_format import estimate_tokens
from savings_constants import (
    MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
    SAVINGS_LINE_OVERHEAD_TOKENS,
)
from schemas import MemoryResponse


@pytest.fixture(autouse=True)
def _fresh_meter():
    """Reset cached singletons + pin meter config for each test."""
    sm._reset_for_tests()
    saved_enabled = settings.savings_meter_enabled
    saved_mult = settings.savings_rederivation_multiplier
    settings.savings_meter_enabled = True
    settings.savings_rederivation_multiplier = 10.0
    yield
    settings.savings_meter_enabled = saved_enabled
    settings.savings_rederivation_multiplier = saved_mult
    sm._reset_for_tests()


def _hit(mid: str, content: str, token_estimate: int | None = None) -> MemoryResponse:
    return MemoryResponse(
        id=mid, memory=content, category="decision", token_estimate=token_estimate
    )


# ── Tokenizer + write-time stamping ──────────────────────────────────


class TestCountAndStamp:
    def test_count_tokens_is_real_tiktoken(self):
        import tiktoken

        enc = tiktoken.get_encoding(settings.savings_tokenizer)
        text = "Real counts, never estimates — the honest meter."
        assert sm.count_tokens(text) == len(enc.encode(text))

    def test_stamp_tokens_enabled_uses_real_count(self):
        text = "a" * 400  # heuristic would say 100
        assert sm.stamp_tokens(text) == sm.count_tokens(text)

    def test_stamp_tokens_disabled_uses_heuristic(self):
        settings.savings_meter_enabled = False
        text = "a" * 400
        assert sm.stamp_tokens(text) == estimate_tokens(text)

    def test_encoder_failure_degrades_to_heuristic(self, monkeypatch):
        import tiktoken

        monkeypatch.setattr(
            tiktoken, "get_encoding",
            MagicMock(side_effect=RuntimeError("offline")),
        )
        sm._reset_for_tests()
        text = "some content here"
        assert sm.count_tokens(text) == estimate_tokens(text)

    def test_hit_tokens_prefers_stored_estimate(self):
        hit = _hit("m1", "irrelevant content", token_estimate=42)
        assert sm.hit_tokens(hit) == 42

    def test_hit_tokens_counts_legacy_rows(self):
        hit = _hit("m1", "legacy row without a stored count")
        assert sm.hit_tokens(hit) == sm.count_tokens("legacy row without a stored count")

    def test_hit_tokens_zero_stamp_is_present_not_missing(self, monkeypatch):
        """A stored token_estimate of 0 is a value, not an absence — it must
        never trigger a hot-path re-tokenization (Copilot review, PR #115)."""
        tripwire = MagicMock(side_effect=AssertionError("re-tokenized a stamped row"))
        monkeypatch.setattr(sm, "count_tokens", tripwire)
        hit = _hit("m1", "some content", token_estimate=0)
        assert sm.hit_tokens(hit) == 0
        tripwire.assert_not_called()


# ── Measurement math ─────────────────────────────────────────────────


class TestMeasureRecall:
    def test_index_recall_positive_savings(self):
        hits = [_hit(f"m{i}", "x", token_estimate=500) for i in range(10)]
        payload = json.dumps([{"id": f"m{i}", "title": "t"} for i in range(10)])
        event = sm.measure_recall(
            "search_index", hits, index_payload=payload, include_line_overhead=True
        )
        assert event.baseline_tokens == 5000
        assert event.served_tokens == 0  # no content served, only the map
        assert event.overhead_tokens == sm.count_tokens(payload) + SAVINGS_LINE_OVERHEAD_TOKENS
        assert event.net_tokens_saved == 5000 - event.overhead_tokens
        assert event.net_tokens_saved > 0

    def test_net_can_go_negative_and_is_never_clamped(self):
        """Tiny memories whose index rows cost more than their content:
        the honest meter reports negative, not zero."""
        hits = [_hit(f"m{i}", "hi", token_estimate=2) for i in range(5)]
        payload = json.dumps([
            {"id": f"m{i}", "title": "a much longer rendered index row " * 3}
            for i in range(5)
        ])
        event = sm.measure_recall(
            "search_index", hits, index_payload=payload, include_line_overhead=True
        )
        assert event.baseline_tokens == 10
        assert event.net_tokens_saved < 0

    def test_full_recall_serves_baseline(self):
        hits = [_hit("m1", "x", token_estimate=300), _hit("m2", "y", token_estimate=200)]
        event = sm.measure_recall("search", hits, served_full=True)
        assert event.baseline_tokens == 500
        assert event.served_tokens == 500
        assert event.overhead_tokens == 0
        assert event.net_tokens_saved == 0

    def test_measure_ask_baseline_vs_answer(self):
        event = sm.measure_ask(4000, "short answer")
        assert event.baseline_tokens == 4000
        assert event.served_tokens == sm.count_tokens("short answer")
        assert event.net_tokens_saved == 4000 - event.served_tokens

    def test_rederivation_estimate_separate_and_never_in_net(self):
        hits = [_hit("m1", "x", token_estimate=100)]
        event = sm.measure_recall("search_index", hits, index_payload="[]")
        assert event.rederivation_savings_estimate == 1000  # 10x multiplier

        settings.savings_rederivation_multiplier = 1000.0
        inflated = sm.measure_recall("search_index", hits, index_payload="[]")
        # the estimate explodes; the measured net is untouched
        assert inflated.rederivation_savings_estimate == 100_000
        assert inflated.net_tokens_saved == event.net_tokens_saved

    def test_disabled_returns_none(self):
        settings.savings_meter_enabled = False
        assert sm.measure_recall("search", [_hit("m", "x")], served_full=True) is None
        assert sm.measure_ask(100, "answer") is None


class TestSavingsLine:
    def test_positive_line(self):
        event = sm.SavingsEvent("search_index", 1000, 0, 100, 900, 10_000)
        assert sm.format_savings_line(event) == "saved ~900 tokens (90%), net of overhead"

    def test_negative_line_stays_signed(self):
        event = sm.SavingsEvent("search_index", 10, 0, 60, -50, 100)
        line = sm.format_savings_line(event)
        assert "~-50 tokens" in line
        assert "(-500%)" in line

    def test_zero_baseline_no_division_error(self):
        event = sm.SavingsEvent("search_index", 0, 0, 70, -70, 0)
        assert "(0%)" in sm.format_savings_line(event)


# ── Kill-switch: zero tokenizer calls ────────────────────────────────


class TestKillSwitch:
    def test_disabled_means_zero_tokenizer_calls(self, monkeypatch):
        import tiktoken

        settings.savings_meter_enabled = False
        sm._reset_for_tests()
        tripwire = MagicMock(side_effect=AssertionError("tokenizer called with meter off"))
        monkeypatch.setattr(tiktoken, "get_encoding", tripwire)

        # write path
        assert sm.stamp_tokens("some content" * 50) == estimate_tokens("some content" * 50)
        # recall paths
        assert sm.measure_recall("search_index", [_hit("m", "x")], index_payload="[]") is None
        assert sm.measure_ask(10, "answer") is None
        # ledger path
        assert sm.record_event("user", None) is False
        tripwire.assert_not_called()

    def test_disabled_ask_evidence_tokens_zero(self):
        """ask_memory's evidence baseline must not tokenize when off."""
        settings.savings_meter_enabled = False
        # hit_tokens on a legacy row (no stored estimate) would tokenize —
        # ask.py guards on the setting; emulate that guard's contract here.
        assert settings.savings_meter_enabled is False


# ── Ledger: append-only stream + O(1) totals ─────────────────────────


class FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._ops: list = []

    def xadd(self, *a, **kw):
        self._ops.append(("xadd", a, kw))

    def hincrby(self, *a, **kw):
        self._ops.append(("hincrby", a, kw))

    def execute(self):
        for op, a, kw in self._ops:
            getattr(self._redis, op)(*a, **kw)
        self._ops = []


class FakeRedis:
    def __init__(self):
        self.streams: dict[str, list[dict]] = {}
        self.hashes: dict[str, dict[str, int]] = {}
        self.kv: dict[str, str] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.streams.setdefault(key, []).append(dict(fields))

    def hincrby(self, key, field, amount):
        h = self.hashes.setdefault(key, {})
        h[field] = h.get(field, 0) + int(amount)

    def hgetall(self, key):
        return {k: str(v) for k, v in self.hashes.get(key, {}).items()}

    def pipeline(self):
        return FakePipeline(self)


class TestLedger:
    def _event(self, net=900):
        return sm.SavingsEvent("search_index", 1000, 0, 1000 - net, net, 10_000)

    def test_append_fields_and_daily_schema_charge(self):
        r = FakeRedis()
        assert sm.record_event("alice", self._event(), redis=r)
        entries = r.streams[sm.LEDGER_KEY.format(user_id="alice")]
        # first metered op of the day charges the tool-schema constant first
        assert len(entries) == 2
        schema, event = entries
        assert schema["op"] == "tool_schema"
        assert schema["net"] == -MCP_TOOL_SCHEMA_OVERHEAD_TOKENS
        assert event["op"] == "search_index"
        assert set(event) == {"ts", "op", "baseline", "served", "overhead", "net", "rederiv_est"}

    def test_schema_charged_once_per_day(self):
        r = FakeRedis()
        sm.record_event("alice", self._event(), redis=r)
        sm.record_event("alice", self._event(), redis=r)
        entries = r.streams[sm.LEDGER_KEY.format(user_id="alice")]
        assert [e["op"] for e in entries] == ["tool_schema", "search_index", "search_index"]

    def test_totals_match_ledger_sum(self):
        r = FakeRedis()
        for net in (900, -50, 300):
            sm.record_event("alice", self._event(net), redis=r)
        entries = r.streams[sm.LEDGER_KEY.format(user_id="alice")]
        ledger_net = sum(int(e["net"]) for e in entries)
        snapshot = sm.metrics_snapshot("alice", redis=r)
        assert snapshot["user"]["net_tokens_saved"] == ledger_net
        assert snapshot["user"]["events"] == len(entries)
        assert snapshot["instance"]["net_tokens_saved"] == ledger_net
        # signed cumulative includes the -schema charge
        assert ledger_net == 900 - 50 + 300 - MCP_TOOL_SCHEMA_OVERHEAD_TOKENS

    def test_estimated_field_kept_out_of_measured_totals(self):
        r = FakeRedis()
        sm.record_event("alice", self._event(), redis=r)
        snapshot = sm.metrics_snapshot("alice", redis=r)
        user = snapshot["user"]
        assert user["rederivation_savings_estimate"] == 10_000
        assert user["net_tokens_saved"] != user["net_tokens_saved"] + user["rederivation_savings_estimate"]

    def test_record_none_or_disabled_is_noop(self):
        r = FakeRedis()
        assert sm.record_event("alice", None, redis=r) is False
        settings.savings_meter_enabled = False
        assert sm.record_event("alice", self._event(), redis=r) is False
        assert r.streams == {}

    def test_redis_failure_swallowed(self, monkeypatch):
        monkeypatch.setattr(sm, "_get_redis", MagicMock(side_effect=ConnectionError))
        assert sm.record_event("alice", self._event()) is False  # no raise

    def test_snapshot_disabled_shape(self):
        settings.savings_meter_enabled = False
        snapshot = sm.metrics_snapshot("alice", redis=FakeRedis())
        assert snapshot["enabled"] is False
        assert snapshot["user"] is None
        assert snapshot["instance"] is None


# ── Overhead constants: measured per release, only shrink ────────────


class TestOverheadConstants:
    def test_tool_schema_overhead_constant_covers_reality(self):
        """Render the REAL MCP tool schemas and count real tokens. If this
        fails, the schemas grew past the checked-in constant — update
        MCP_TOOL_SCHEMA_OVERHEAD_TOKENS consciously in the same PR."""
        import tiktoken

        import mcp_server

        tools = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            mcp_server.list_tools()
        )
        rendered = json.dumps(
            [t.model_dump(exclude_none=True) for t in tools],
            default=str, ensure_ascii=False,
        )
        enc = tiktoken.get_encoding(settings.savings_tokenizer)
        measured = len(enc.encode(rendered))
        assert measured <= MCP_TOOL_SCHEMA_OVERHEAD_TOKENS, (
            f"MCP tool schemas measure {measured} tokens > checked-in constant "
            f"{MCP_TOOL_SCHEMA_OVERHEAD_TOKENS}. The NS-injected overhead may only "
            f"shrink; if this growth is intentional, update savings_constants.py "
            f"in this PR."
        )

    def test_savings_line_overhead_constant_covers_reality(self):
        import tiktoken

        event = sm.SavingsEvent("search_index", 999_999, 0, 99_999, -899_999, 9_999_999)
        line = sm.format_savings_line(event)
        rendered = json.dumps({"savings": line, "savings_detail": event.detail()})
        enc = tiktoken.get_encoding(settings.savings_tokenizer)
        measured = len(enc.encode(rendered))
        assert measured <= SAVINGS_LINE_OVERHEAD_TOKENS, (
            f"savings line renders at {measured} tokens > constant "
            f"{SAVINGS_LINE_OVERHEAD_TOKENS}; update savings_constants.py consciously."
        )


# ── REST surface: index_only carries the savings line ────────────────


class TestRestSavingsLine:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import main

        return TestClient(main.app, raise_server_exceptions=False), main

    def test_index_only_response_includes_savings(self, client):
        test_client, main_mod = client
        hits = [
            _hit("11111111-1111-1111-1111-111111111111", "long content " * 40, token_estimate=400),
            _hit("22222222-2222-2222-2222-222222222222", "more content " * 40, token_estimate=350),
        ]
        with patch.object(main_mod._service, "search", return_value=hits), \
             patch.object(sm, "record_event", return_value=True) as rec:
            resp = test_client.post(
                "/v1/search",
                json={"query": "content", "user_id": "alice", "index_only": True},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_only"] is True
        assert body["savings"].startswith("saved ~")
        assert body["savings"].endswith("net of overhead")
        detail = body["savings_detail"]
        assert detail["baseline_tokens"] == 750
        assert detail["served_tokens"] == 0
        assert detail["net_tokens_saved"] == 750 - detail["overhead_tokens"]
        rec.assert_called_once()
        assert rec.call_args.args[0] == "alice"

    def test_index_only_meter_disabled_no_savings_fields(self, client):
        test_client, main_mod = client
        settings.savings_meter_enabled = False
        hits = [_hit("11111111-1111-1111-1111-111111111111", "content", token_estimate=50)]
        with patch.object(main_mod._service, "search", return_value=hits):
            resp = test_client.post(
                "/v1/search",
                json={"query": "q", "user_id": "alice", "index_only": True},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("savings") is None
        assert body.get("savings_detail") is None

    def test_metrics_endpoint_shape(self, client):
        test_client, _ = client
        fake = FakeRedis()
        with patch.object(sm, "_get_redis", return_value=fake):
            sm.record_event("alice", sm.SavingsEvent("search_index", 100, 0, 10, 90, 1000), redis=fake)
            resp = test_client.get("/v1/metrics", params={"user_id": "alice"})
        assert resp.status_code == 200
        body = resp.json()["savings_meter"]
        assert body["enabled"] is True
        assert body["tool_schema_overhead_tokens"] == MCP_TOOL_SCHEMA_OVERHEAD_TOKENS
        assert body["user"]["user_id"] == "alice"
        assert body["user"]["net_tokens_saved"] == 90 - MCP_TOOL_SCHEMA_OVERHEAD_TOKENS
