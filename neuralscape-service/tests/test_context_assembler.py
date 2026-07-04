"""Unit tests for the E3 token-budgeted context assembler.

No running services. Covers: the hard budget cap at multiple budgets
(rendered plain bundle measured with the same counter), the 60/40
messages/summary split of the post-card/index remainder, card inclusion +
its reserved share, query index rows, the three provider formatter shapes,
savings ledgering (baseline = full transcript + full hit content, served =
bundle), and the REST route (identity resolution + format validation).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import context_assembler as ca
import savings_meter as sm
import session_summarizer as ss
from config import settings
from schemas import MemoryResponse
from tests.fake_sync_redis import FakeSyncRedis

USER = "asm-user"
SESSION = "asm-session"


@pytest.fixture()
def r():
    return FakeSyncRedis()


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch):
    """Capture ledger events instead of touching Redis streams."""
    events = []
    monkeypatch.setattr(sm, "record_event", lambda uid, ev, redis=None: events.append((uid, ev)) or True)
    yield events


def _seed_session(r, n_messages: int = 40, msg_words: int = 30):
    body = " ".join(["tokenword"] * msg_words)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} {body}"}
            for i in range(n_messages)]
    ss.record_messages(USER, SESSION, msgs, redis=r)


def _seed_slot(r, slot: str, words: int):
    text = " ".join(["summary"] * words)
    record = {"text": text, "tokens": ss.text_tokens(text), "through_count": 40,
              "updated_at": "2026-07-04T00:00:00+00:00", "slot": slot}
    r.set(ss.slot_key(USER, SESSION, slot), json.dumps(record))


def _seed_card(r, pool: str = f"user--{USER}", lines: int = 6):
    card_lines = [f"ATTRIBUTE: stable fact number {i}" for i in range(lines)]
    card_lines[0] = "IDENTITY: A test user who assembles context."
    r.set(f"dreaming:card:{pool}", json.dumps({"lines": card_lines, "updated_at": "x"}))


def _hit(mid: str, content: str) -> MemoryResponse:
    return MemoryResponse(
        id=mid, memory=content, category="decision", created_at="2026-07-01T00:00:00+00:00",
        token_estimate=ss.text_tokens(content),
    )


def _service(hits=None):
    svc = MagicMock()
    svc.search.return_value = hits or []
    return svc


def _assemble(r, service=None, **kw):
    kw.setdefault("user_id", USER)
    kw.setdefault("budget_tokens", 2000)
    kw.setdefault("session_id", SESSION)
    return ca.assemble_context(service or _service(), redis=r, **kw)


# ──────────────────────────────────────────────
# Budget discipline
# ──────────────────────────────────────────────


class TestBudget:
    @pytest.mark.parametrize("budget", [300, 1000, 2000, 8000])
    def test_never_exceeds_budget(self, r, budget):
        _seed_session(r, 80, 60)
        _seed_slot(r, "short", 900)
        _seed_slot(r, "long", 3500)
        _seed_card(r)
        out = _assemble(r, budget_tokens=budget, fmt="plain")
        assert out["used_tokens"] <= budget
        # The REAL rendered payload (headers included) fits too.
        assert ss.text_tokens(out["bundle"]["text"]) <= budget

    @pytest.mark.parametrize("budget", [1000, 2000, 8000])
    def test_60_40_split_of_remainder(self, r, budget):
        # Oversupply both sides so each section fills its share.
        _seed_session(r, 200, 60)
        _seed_slot(r, "short", 5000)
        _seed_slot(r, "long", 8000)
        out = _assemble(r, budget_tokens=budget)
        s = out["sections"]
        remaining = budget - s["card_tokens"] - s["index_tokens"] - ca.FRAMING_OVERHEAD_TOKENS
        messages_share = int(remaining * ca.MESSAGES_SHARE)
        summary_share = remaining - messages_share
        assert s["summary_tokens"] <= summary_share
        # Summary fills its share (oversized slots get truncated INTO it), so
        # nothing rolls over: messages stay within their 60%.
        assert s["messages_tokens"] <= messages_share
        # And the split is actually exercised, not vacuous: each side uses
        # most of its share (± one message / truncation granularity).
        assert s["summary_tokens"] >= summary_share - 20
        assert s["messages_tokens"] >= messages_share - (ss.text_tokens("m0 " + "tokenword " * 60) + 2)

    def test_unspent_summary_share_rolls_into_messages(self, r):
        _seed_session(r, 200, 60)
        _seed_slot(r, "short", 30)  # tiny summary, big leftover
        out = _assemble(r, budget_tokens=2000)
        s = out["sections"]
        remaining = 2000 - s["card_tokens"] - s["index_tokens"] - ca.FRAMING_OVERHEAD_TOKENS
        assert s["messages_tokens"] > int(remaining * ca.MESSAGES_SHARE)  # got the rollover
        assert out["used_tokens"] <= 2000

    def test_long_slot_preferred_when_it_fits(self, r):
        _seed_session(r, 10, 10)
        _seed_slot(r, "short", 100)
        _seed_slot(r, "long", 200)
        out = _assemble(r, budget_tokens=8000)
        assert out["sections"]["summary_slot"] == "long"

    def test_short_slot_fallback_when_long_overflows_share(self, r):
        _seed_session(r, 10, 10)
        _seed_slot(r, "short", 100)
        _seed_slot(r, "long", 8000)
        out = _assemble(r, budget_tokens=1000)
        assert out["sections"]["summary_slot"] == "short"

    def test_empty_session_and_no_card_still_ok(self, r):
        out = _assemble(r, session_id=None)
        assert out["status"] == "ok"
        assert out["used_tokens"] == 0
        assert out["bundle"] == {"text": ""}


# ──────────────────────────────────────────────
# Card + index sections
# ──────────────────────────────────────────────


class TestSections:
    def test_card_included_when_present(self, r):
        _seed_card(r)
        out = _assemble(r)
        assert out["sections"]["card_lines"] > 0
        assert "IDENTITY: A test user" in out["bundle"]["text"]

    def test_project_card_joins_user_card(self, r):
        _seed_card(r)
        r.set(
            "dreaming:card:shared--project--proj1",
            json.dumps({"lines": ["IDENTITY: The proj1 project.",
                                  "ATTRIBUTE: proj1 ships weekly."]}),
        )
        out = _assemble(r, project_id="proj1")
        assert out["sections"]["card_lines"] == 8
        assert "proj1 ships weekly" in out["bundle"]["text"]

    def test_card_capped_at_its_share(self, r):
        # A pathological 40-line card of long lines can't eat the budget.
        lines = ["ATTRIBUTE: " + "detail " * 60 for _ in range(40)]
        lines[0] = "IDENTITY: big card"
        r.set(f"dreaming:card:user--{USER}", json.dumps({"lines": lines}))
        out = _assemble(r, budget_tokens=1000)
        assert out["sections"]["card_tokens"] <= int(1000 * ca.CARD_BUDGET_SHARE)

    def test_query_adds_index_rows_and_hits_feed_baseline(self, r, _no_ledger):
        hits = [_hit(f"id-{i}", "a decision about the deploy pipeline " * 20) for i in range(3)]
        svc = _service(hits)
        out = _assemble(r, service=svc, query="deploy pipeline")
        svc.search.assert_called_once()
        assert out["sections"]["index_rows"] > 0
        assert "id-0" in out["bundle"]["text"]
        # Baseline includes the hits' full content, served does not.
        (_, event), = _no_ledger[-1:]
        assert event.baseline_tokens >= sum(ss.text_tokens(h.memory) for h in hits)

    def test_no_query_no_search(self, r):
        svc = _service()
        _assemble(r, service=svc)
        svc.search.assert_not_called()

    def test_search_failure_degrades(self, r):
        svc = MagicMock()
        svc.search.side_effect = RuntimeError("qdrant down")
        out = _assemble(r, service=svc, query="anything")
        assert out["status"] == "ok"
        assert out["sections"]["index_rows"] == 0


# ──────────────────────────────────────────────
# Provider formatters
# ──────────────────────────────────────────────


class TestFormatters:
    def test_plain_is_single_text(self, r):
        _seed_session(r, 4, 5)
        out = _assemble(r, fmt="plain")
        assert set(out["bundle"]) == {"text"}
        assert "## Recent messages" in out["bundle"]["text"]

    def test_anthropic_shape(self, r):
        _seed_session(r, 4, 5)
        _seed_card(r)
        out = _assemble(r, fmt="anthropic")
        bundle = out["bundle"]
        assert set(bundle) == {"system", "messages"}
        assert "IDENTITY:" in bundle["system"]
        assert all(m["role"] in ("user", "assistant") for m in bundle["messages"])
        assert len(bundle["messages"]) == 4

    def test_openai_shape(self, r):
        _seed_session(r, 4, 5)
        _seed_card(r)
        out = _assemble(r, fmt="openai")
        msgs = out["bundle"]["messages"]
        assert msgs[0]["role"] == "system"
        assert all(m["role"] in ("system", "user", "assistant") for m in msgs)

    def test_weird_roles_normalized(self, r):
        ss.record_messages(USER, SESSION, [{"role": "tool", "content": "x"}], redis=r)
        out = _assemble(r, fmt="anthropic", budget_tokens=8000)
        assert out["bundle"]["messages"][0]["role"] == "user"

    def test_unknown_format_rejected(self, r):
        with pytest.raises(ValueError, match="Invalid format"):
            _assemble(r, fmt="gemini")


# ──────────────────────────────────────────────
# Savings ledgering
# ──────────────────────────────────────────────


class TestSavings:
    def test_every_response_ledgers(self, r, _no_ledger):
        _seed_session(r, 100, 60)
        out = _assemble(r, budget_tokens=1000)
        assert len(_no_ledger) == 1
        uid, event = _no_ledger[0]
        assert uid == USER
        assert event.op == "context_assemble"
        assert event.served_tokens == out["used_tokens"]
        assert event.baseline_tokens > event.served_tokens  # 100 msgs vs 1k budget
        assert out["savings"] is not None
        assert out["savings_detail"]["net_tokens_saved"] == event.net_tokens_saved

    def test_meter_off_no_ledger_no_line(self, r, _no_ledger, monkeypatch):
        monkeypatch.setattr(settings, "savings_meter_enabled", False)
        _seed_session(r, 10, 10)
        out = _assemble(r)
        assert _no_ledger == []
        assert out["savings"] is None and out["savings_detail"] is None

    def test_measure_assemble_signed(self, monkeypatch):
        monkeypatch.setattr(settings, "savings_meter_enabled", True)
        event = sm.measure_assemble(10, 50)
        assert event.op == "context_assemble"
        assert event.net_tokens_saved == -40  # signed, never clamped

    def test_measure_assemble_disabled_none(self, monkeypatch):
        monkeypatch.setattr(settings, "savings_meter_enabled", False)
        assert sm.measure_assemble(100, 10) is None


# ──────────────────────────────────────────────
# REST route
# ──────────────────────────────────────────────


class TestRoute:
    @pytest.fixture()
    def client(self, r, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        monkeypatch.setattr(ss, "_redis", r)
        monkeypatch.setattr(main, "_service", _service())
        return TestClient(main.app, raise_server_exceptions=False)

    def test_route_assembles(self, client, r):
        _seed_session(r, 6, 5)
        resp = client.post(
            "/v1/context/assemble",
            json={"budget_tokens": 2000, "user_id": USER, "session_id": SESSION,
                  "format": "anthropic"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["used_tokens"] <= 2000
        assert "messages" in body["bundle"]

    def test_route_validates_format(self, client):
        resp = client.post(
            "/v1/context/assemble",
            json={"budget_tokens": 2000, "user_id": USER, "format": "nope"},
        )
        assert resp.status_code == 422

    def test_route_validates_budget(self, client):
        resp = client.post(
            "/v1/context/assemble", json={"budget_tokens": 5, "user_id": USER}
        )
        assert resp.status_code == 422

    def test_existing_context_surface_untouched(self, client, monkeypatch):
        """D1's plugin consumes GET /v1/context/inject — must keep working."""
        import main

        from schemas import ContextResponse

        main._service.get_global_context = MagicMock(
            return_value=ContextResponse(user_id=USER, categories={}, standards=[])
        )
        resp = client.get("/v1/context/inject", params={"user_id": USER})
        assert resp.status_code == 200
        assert "additionalContext" in resp.json()
