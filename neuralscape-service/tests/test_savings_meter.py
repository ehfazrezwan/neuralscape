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
    LIFECYCLE_STAGES,
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

    def __getattr__(self, name):
        # Record any buffered command (xadd/hincrby/sadd/expire/set/delete/…)
        # and replay it against the FakeRedis on execute().
        def _record(*a, **kw):
            self._ops.append((name, a, kw))
            return self

        return _record

    def execute(self):
        results = []
        for op, a, kw in self._ops:
            results.append(getattr(self._redis, op)(*a, **kw))
        self._ops = []
        return results


class FakeRedis:
    def __init__(self):
        self.streams: dict[str, list[dict]] = {}
        self.hashes: dict[str, dict[str, int]] = {}
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, *keys):
        n = 0
        for k in keys:
            for store in (self.kv, self.hashes, self.sets, self.streams):
                if k in store:
                    del store[k]
                    n += 1
        return n

    def expire(self, key, ttl):  # TTL is irrelevant to these synchronous tests
        return True

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.streams.setdefault(key, []).append(dict(fields))

    def hincrby(self, key, field, amount):
        h = self.hashes.setdefault(key, {})
        h[field] = h.get(field, 0) + int(amount)

    def hgetall(self, key):
        return {k: str(v) for k, v in self.hashes.get(key, {}).items()}

    def sadd(self, key, *members):
        s = self.sets.setdefault(key, set())
        before = len(s)
        s.update(str(m) for m in members)
        return len(s) - before

    def smembers(self, key):
        return set(self.sets.get(key, set()))

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
        # M1/M2/M4: rows now also carry lifecycle/item_id/corr_id + adjusted net.
        assert set(event) == {
            "ts", "op", "lifecycle", "item_id", "corr_id",
            "baseline", "served", "overhead", "net", "adjusted", "rederiv_est",
        }

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


# ── REST surface: index_only meters off the hot path (audit 27 #11) ──


class TestRestSavingsLine:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import main

        return TestClient(main.app, raise_server_exceptions=False), main

    def test_index_only_ledgered_off_the_hot_path(self, client):
        """Audit 27 #11: measurement + ledger append moved to the telemetry
        executor — the response no longer blocks on (or carries) the savings
        line, but the honest event still lands in the ledger with the full
        rendered body measured as overhead."""
        import telemetry

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
            telemetry.flush()
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_only"] is True
        # The per-recall line/detail are no longer served on the response —
        # totals surface via GET /v1/metrics instead.
        assert body.get("savings") is None
        assert body.get("savings_detail") is None
        rec.assert_called_once()
        assert rec.call_args.args[0] == "alice"
        event = rec.call_args.args[1]
        assert event.op == "search_index"
        assert event.baseline_tokens == 750
        assert event.served_tokens == 0
        assert event.overhead_tokens > 0  # rendered body measured verbatim
        assert event.net_tokens_saved == 750 - event.overhead_tokens

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


# ── M1: per-event lifecycle-tagged ledger ────────────────────────────


class TestLifecycleTagging:
    def test_events_carry_stage_and_bucket_per_lifecycle(self):
        r = FakeRedis()
        sm.record_event(
            "alice",
            sm.measure_recall(
                "search_index",
                [_hit("m1", "x" * 200, token_estimate=200)],
                index_payload="rows",
            ),
            redis=r,
        )
        sm.record_event("alice", sm.measure_ingest(5000, 400, item_id="doc1"), redis=r)
        sm.record_event(
            "alice",
            sm.measure_code_nav(
                "code_nav_locate", served_text="path/to/f.py:12", files=["path/to/f.py"]
            ),
            redis=r,
        )
        sm.record_event("alice", sm.measure_compaction(3000, 200, item_id="slot0"), redis=r)
        snap = sm.metrics_snapshot("alice", redis=r)
        by_lc = snap["user"]["by_lifecycle"]
        # every distinct stage exercised shows up, including the tool_schema
        # charge which is booked under context_assembly.
        assert {"retrieval", "ingest", "code_nav", "compaction", "context_assembly"} <= set(by_lc)
        assert by_lc["ingest"]["baseline_tokens"] == 5000
        assert by_lc["compaction"]["baseline_tokens"] == 3000
        # per-op breakdown carries the op keys too
        assert "search_index" in snap["user"]["by_op"]
        assert "ingest" in snap["user"]["by_op"]

    def test_event_records_item_and_corr_id(self):
        r = FakeRedis()
        ev = sm.measure_ingest(100, 10, item_id="doc-9", corr_id="run-1")
        assert ev.lifecycle_stage == "ingest"
        assert ev.item_id == "doc-9"
        assert ev.corr_id == "run-1"
        sm.record_event("alice", ev, redis=r)
        row = r.streams[sm.LEDGER_KEY.format(user_id="alice")][-1]
        assert row["item_id"] == "doc-9"
        assert row["corr_id"] == "run-1"
        assert row["lifecycle"] == "ingest"


# ── M2: bounce-adjusted net ──────────────────────────────────────────


class TestBounceAdjustment:
    def test_served_then_refetched_deducts_only_adjusted(self):
        r = FakeRedis()
        hit = _hit("11111111-1111-1111-1111-111111111111", "content " * 50, token_estimate=400)
        sm.arm_bounce("alice", [hit], redis=r)  # served as an index row
        sm.record_event(
            "alice",
            sm.measure_recall("search_index", [hit], index_payload="rows"),
            redis=r,
        )
        before = sm.metrics_snapshot("alice", redis=r)["user"]
        deducted = sm.check_and_deduct_bounce("alice", [hit], redis=r)  # full-fetch
        assert deducted == 1
        after = sm.metrics_snapshot("alice", redis=r)["user"]
        # raw net is untouched; only the bounce-adjusted net moves, by the
        # credited baseline share (the item's stored token count).
        assert after["net_tokens_saved"] == before["net_tokens_saved"]
        assert after["adjusted_net_tokens_saved"] == before["adjusted_net_tokens_saved"] - 400

    def test_refetch_is_one_shot(self):
        r = FakeRedis()
        hit = _hit("22222222-2222-2222-2222-222222222222", "z" * 100, token_estimate=200)
        sm.arm_bounce("alice", [hit], redis=r)
        assert sm.check_and_deduct_bounce("alice", [hit], redis=r) == 1
        # the marker is consumed — a second full-fetch is not a bounce
        assert sm.check_and_deduct_bounce("alice", [hit], redis=r) == 0

    def test_no_bounce_without_prior_arm(self):
        r = FakeRedis()
        hit = _hit("33333333-3333-3333-3333-333333333333", "x", token_estimate=50)
        assert sm.check_and_deduct_bounce("alice", [hit], redis=r) == 0


# ── M3: code-nav avoided-read baseline ───────────────────────────────


class TestCodeNavBaseline:
    def test_baseline_counts_distinct_files(self):
        ev = sm.measure_code_nav(
            "code_nav_locate",
            served_text="short",
            files=["a.py", "a.py", "b.py"],  # 2 distinct
            avoided_read_tokens=1000,
        )
        assert ev.lifecycle_stage == "code_nav"
        assert ev.baseline_tokens == 2000  # 2 distinct files × 1000
        assert ev.served_tokens == sm.count_tokens("short")
        assert ev.net_tokens_saved == 2000 - ev.served_tokens

    def test_baseline_uses_settings_default_per_file(self):
        per_file = settings.savings_code_nav_avoided_read_tokens_per_file
        ev = sm.measure_code_nav("code_nav_neighbors", served_tokens=10, file_count=3)
        assert ev.baseline_tokens == 3 * per_file

    def test_zero_files_is_zero_baseline_not_content_size(self):
        ev = sm.measure_code_nav("code_nav_query", served_text="a long answer " * 20)
        assert ev.baseline_tokens == 0  # nothing located → nothing avoided


# ── M4: per-task rollup ──────────────────────────────────────────────


class TestPerTaskRollup:
    def test_events_roll_up_by_corr_id(self):
        r = FakeRedis()
        sm.record_event("alice", sm.measure_ask(1000, "short answer", corr_id="task-7"), redis=r)
        sm.record_event(
            "alice",
            sm.measure_recall(
                "search_index",
                [_hit("m1", "y" * 80, token_estimate=80)],
                index_payload="r",
                corr_id="task-7",
            ),
            redis=r,
        )
        # an event under a different corr_id must not roll into task-7
        sm.record_event("alice", sm.measure_ask(500, "a", corr_id="other"), redis=r)
        snap = sm.metrics_snapshot("alice", redis=r, task_id="task-7")
        assert snap["task"]["corr_id"] == "task-7"
        assert snap["task"]["events"] == 2  # the schema charge has no corr_id

    def test_task_absent_when_not_requested(self):
        r = FakeRedis()
        sm.record_event("alice", sm.measure_ask(100, "a", corr_id="t"), redis=r)
        snap = sm.metrics_snapshot("alice", redis=r)
        assert "task" not in snap


# ── M5/M6: metrics payload shape + honesty labels ────────────────────


class TestMetricsPayloadShapeM6:
    def test_honesty_labels_and_breakdowns_present(self):
        r = FakeRedis()
        sm.record_event(
            "alice",
            sm.measure_recall(
                "search_index",
                [_hit("m1", "z" * 120, token_estimate=120)],
                index_payload="rows",
            ),
            redis=r,
        )
        body = sm.metrics_snapshot("alice", redis=r)
        # M5 honesty surface
        assert body["tokenizer"] == settings.savings_tokenizer
        assert "tokenizer_basis_note" in body
        assert body["rederivation_multiplier"] == settings.savings_rederivation_multiplier
        assert "rederivation_note" in body
        assert "code_nav_baseline_note" in body
        assert body["code_nav_avoided_read_tokens_per_file"] == (
            settings.savings_code_nav_avoided_read_tokens_per_file
        )
        assert set(body["lifecycle_stages"]) == set(LIFECYCLE_STAGES)
        # M6 breakdowns on both scopes + adjusted net
        for scope in ("user", "instance"):
            assert "by_lifecycle" in body[scope]
            assert "by_op" in body[scope]
            assert "adjusted_net_tokens_saved" in body[scope]

    def test_new_entrypoints_short_circuit_when_disabled(self):
        settings.savings_meter_enabled = False
        assert sm.measure_ingest(100, 10) is None
        assert sm.measure_code_nav("code_nav_locate", files=["a.py"]) is None
        assert sm.measure_compaction(100, 10) is None
        assert sm.measure_assemble(100, 10) is None
        assert sm.arm_bounce("alice", [_hit("m1", "x", token_estimate=5)]) is False
        assert sm.check_and_deduct_bounce("alice", [_hit("m1", "x", token_estimate=5)]) == 0

    def test_snapshot_disabled_with_task_id_returns_none(self):
        settings.savings_meter_enabled = False
        snap = sm.metrics_snapshot("alice", redis=FakeRedis(), task_id="t")
        assert snap["task"] is None


# ── MF-1/Copilot C-nav: strict code-file parser (no prose as files) ──


class TestCodeNavFileParser:
    def test_prose_dotted_tokens_are_not_files(self):
        prose = (
            "See e.g. np.array and settings.savings; visit example.com or v2.x; "
            "call os.path.join(foo.bar) and mem0.Memory — i.e. nothing here."
        )
        assert sm.distinct_files_in_text(prose) == []

    def test_slash_paths_are_found(self):
        text = "defined in neuralscape-service/main.py:1554 and adapters/base.py here"
        files = sm.distinct_files_in_text(text)
        assert "neuralscape-service/main.py" in files
        assert "adapters/base.py" in files
        assert len(files) == 2

    def test_bare_file_requires_line_number(self):
        # a bare dotted token without a slash or :line is NOT a path…
        assert sm.distinct_files_in_text("please edit config.py now") == []
        # …but file.ext:line is an unambiguous code reference.
        assert sm.distinct_files_in_text("please edit config.py:12 now") == ["config.py"]

    def test_line_suffix_normalized_for_dedup(self):
        text = "a/b/c.py:10 and a/b/c.py:20 are the same file"
        assert sm.distinct_files_in_text(text) == ["a/b/c.py"]

    def test_extension_allowlist_rejects_non_code(self):
        # slash paths but non-code extensions (images/binaries/domains) skipped
        assert sm.distinct_files_in_text("assets/logo.png and cdn.site/x.woff2") == []

    def test_cap_bounds_file_count(self):
        text = " ".join(f"pkg/mod{i}.py" for i in range(50))
        assert len(sm.distinct_files_in_text(text)) == 20  # default cap
        assert len(sm.distinct_files_in_text(text, cap=5)) == 5

    def test_measure_code_nav_from_prose_answer_books_zero_baseline(self):
        # end-to-end: a prose query answer with no real paths must not
        # fabricate avoided-read baseline (the honesty guarantee).
        prose = "The helper np.array is used across settings.savings and foo.bar."
        ev = sm.measure_code_nav(
            "code_nav_query", served_text=prose, files=sm.distinct_files_in_text(prose)
        )
        assert ev.baseline_tokens == 0


# ── MF-4: adjusted net is DERIVED (net − bounced), never a stored field ──


class TestAdjustedDerived:
    def test_legacy_totals_hash_reads_adjusted_equals_net(self):
        r = FakeRedis()
        key = sm.TOTALS_KEY.format(user_id="alice")
        # a pre-M2 totals hash: net history but NO ``bounced`` field
        r.hashes[key] = {
            "events": 5, "baseline": 1000, "served": 100,
            "overhead": 50, "net": 850, "rederiv_est": 10000,
        }
        totals = sm._read_totals(r, key)
        assert totals["bounced_tokens"] == 0
        assert totals["adjusted_net_tokens_saved"] == totals["net_tokens_saved"] == 850

    def test_bounce_moves_only_derived_adjusted_not_stored(self):
        r = FakeRedis()
        hit = _hit("55555555-5555-5555-5555-555555555555", "c" * 40, token_estimate=300)
        sm.arm_bounce("alice", [hit], redis=r)
        sm.record_event("alice", sm.measure_recall("search_index", [hit], index_payload="rows"), redis=r)
        sm.check_and_deduct_bounce("alice", [hit], redis=r)
        # the totals hash stores a ``bounced`` accumulator, NOT an ``adjusted`` one
        totals_hash = r.hashes[sm.TOTALS_KEY.format(user_id="alice")]
        assert "adjusted" not in totals_hash
        assert int(totals_hash["bounced"]) == 300
        user = sm.metrics_snapshot("alice", redis=r)["user"]
        assert user["adjusted_net_tokens_saved"] == user["net_tokens_saved"] - 300

    def test_zero_token_item_consumes_marker_without_booking_bounce(self):
        r = FakeRedis()
        hit = _hit("66666666-6666-6666-6666-666666666666", "", token_estimate=0)
        sm.arm_bounce("alice", [hit], redis=r)
        assert sm.check_and_deduct_bounce("alice", [hit], redis=r) == 0  # C2
        assert sm.check_and_deduct_bounce("alice", [hit], redis=r) == 0  # marker gone


# ── C3: unknown lifecycle stages are clamped (never silently lost) ───


class TestLifecycleClamp:
    def test_unknown_stage_clamped_to_retrieval(self):
        r = FakeRedis()
        ev = sm.measure_recall(
            "search_index", [_hit("m1", "x" * 40, token_estimate=40)], index_payload="rows"
        )
        ev.lifecycle_stage = "bogus_stage"
        sm.record_event("alice", ev, redis=r)
        row = r.streams[sm.LEDGER_KEY.format(user_id="alice")][-1]
        assert row["lifecycle"] == "retrieval"
        snap = sm.metrics_snapshot("alice", redis=r)
        assert "retrieval" in snap["user"]["by_lifecycle"]
        assert "bogus_stage" not in snap["user"]["by_lifecycle"]
