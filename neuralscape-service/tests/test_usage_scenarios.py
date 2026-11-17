"""Per-person scenario reports: declared mixes, reconciled inputs, explicit unknowns, no double debit."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

import usage_scenario_pilot as pilot
import usage_scenarios as us
from contracts_usage import UsageEvent
from contracts_usage_reconcile import reconcile_usage_events

TENANT = "tenant-s"
AS_OF = date(2026, 10, 16)


def qty(value=None, reason=None):
    return {"value": value, "missing_reason": reason}


def usage(inp=1000, out=200, cache=0, reasoning=0, *, cache_inclusion="included_in_input", reasoning_inclusion="additional_to_output",
          cache_known=True, reasoning_known=True):
    return dict(
        measurement="actual",
        input_tokens=qty(inp), output_tokens=qty(out),
        cache_read_tokens=qty(cache) if cache_known else qty(None, "provider_did_not_report"),
        cache_write_tokens=qty(None, "not_applicable"),
        reasoning_tokens=qty(reasoning) if reasoning_known else qty(None, "provider_did_not_report"),
        cache_read_inclusion=cache_inclusion, cache_write_inclusion="not_applicable", reasoning_inclusion=reasoning_inclusion,
    )


_DEFAULT = object()


def ev(event_id, *, operation="extract", attempt="a1", status="final", outcome="completed", u=_DEFAULT, model="m-1", provider="google",
       predecessor=None, late=False, task="t1"):
    values = dict(
        schema_version="candidate-v1", event_id=event_id, tenant_id=TENANT, task_id=task, session_id=None, intent_id=None,
        attempt_id=attempt, ledger="service", operation=operation, predecessor_event_id=predecessor, status=status,
        attempt_outcome=outcome, late_after_cancellation=late,
        attribution=dict(producer=operation, bill_owner=None, bill_owner_missing_reason="dependency_unavailable",
                         execution_location="provider_hosted", provider=provider, provider_missing_reason=None,
                         model=model, model_missing_reason=None, recipients=None, recipients_missing_reason="dependency_unavailable"),
        usage=(usage() if u is _DEFAULT else u) if status != "pending" and outcome != "in_progress" else None,
        usage_missing_reason=None,
    )
    if values["usage"] is None and status != "pending":
        values["usage_missing_reason"] = "provider_did_not_report"
    return UsageEvent.model_validate_json(json.dumps(values))


def ledger(*events):
    return reconcile_usage_events(tuple(events))


def card(model="m-1", *, frm=date(2026, 1, 1), inp="1.00", out="4.00", cache="0.10", reasoning="4.00", currency="USD"):
    d = lambda v: None if v is None else Decimal(v)  # noqa: E731
    return us.RateCard(provider="google", model=model, effective_from=frm, currency=currency, input_per_mtok=Decimal(inp),
                       output_per_mtok=Decimal(out), cache_read_per_mtok=d(cache), cache_write_per_mtok=None, reasoning_per_mtok=d(reasoning))


def mix(foreground=(("extract", 10),), *, name="regular", days=20, background=(), bg_days=0):
    return us.ActivityMix(
        name=name, basis="a declared test mix", agents=2, projects=3, active_days_per_month=days,
        foreground=tuple(us.ActivityLine(operation=o, per_day=n) for o, n in foreground),
        background_days_per_month=bg_days, background=tuple(us.ActivityLine(operation=o, per_day=n) for o, n in background),
        missing_categories=("consumer agent usage",),
    )


def report(m, led, cards=None, **kw):
    return us.scenario_report(m, led, [card()] if cards is None else cards, as_of=AS_OF, **kw)


# ----------------------------------------------------------------- declared inputs
def test_a_mix_needs_a_basis_and_real_activity_and_there_is_no_default_mix() -> None:
    with pytest.raises(ValidationError):
        us.ActivityMix(name="light", basis="", agents=1, projects=1, active_days_per_month=1, foreground=(us.ActivityLine(operation="x", per_day=1),),
                       background_days_per_month=0, background=(), missing_categories=())
    with pytest.raises(ValidationError):
        mix(foreground=())
    with pytest.raises(ValidationError):
        us.ActivityLine(operation="x", per_day=0)
    with pytest.raises(ValidationError):
        mix(background=(("extract", 1),), bg_days=0)  # background operations without background days
    with pytest.raises(ValidationError):
        card(inp="-1")
    with pytest.raises(ValidationError):
        card(inp="NaN")
    assert not [n for n, v in vars(us).items() if isinstance(v, us.ActivityMix)]  # nothing ships a default mix


# ----------------------------------------------------------------- arithmetic, checked by hand
def test_cost_follows_the_inclusion_rules_and_equals_a_hand_calculation() -> None:
    # 1 completed call: input 1,000 (400 of it cached), output 200, reasoning 50 additional to output.
    led = ledger(ev("e1", u=usage(1000, 200, cache=400, reasoning=50)))
    r = report(mix(foreground=(("extract", 10),), days=20), led)
    line = r["lines"][0]
    assert line["logical_operations_per_month"] == 200
    # per operation: 600 input at 1.00, 400 cached at 0.10, 200 output at 4.00, 50 reasoning at 4.00 (per million tokens)
    per_op = (Decimal(600) * 1 + Decimal(400) * Decimal("0.10") + Decimal(200) * 4 + Decimal(50) * 4) / 1_000_000
    assert Decimal(line["cost_per_month"]) == (per_op * 200).quantize(Decimal("0.000001"))
    # tokens: input + output + reasoning (cached tokens are inside input and are not added again)
    assert Decimal(line["tokens_per_month"]) == Decimal(1000 + 200 + 50) * 200
    service = r["per_person_per_month"]["service"]
    assert service["cost_total"] == line["cost_per_month"] and service["tokens_complete"] and service["cost_complete"]


def test_reasoning_already_inside_output_is_not_billed_again() -> None:
    led = ledger(ev("e1", u=usage(1000, 200, reasoning=50, reasoning_inclusion="included_in_output")))
    line = report(mix(), led)["lines"][0]
    assert Decimal(line["tokens_per_month"]) == Decimal(1200) * 200
    expected = (Decimal(1000) * 1 + Decimal(200) * 4) / 1_000_000 * 200
    assert Decimal(line["cost_per_month"]) == expected.quantize(Decimal("0.000001"))


def test_the_failed_and_retried_attempts_that_reported_tokens_belong_to_the_operation() -> None:
    led = ledger(
        ev("f1", attempt="a1", outcome="failed", u=usage(300, 0)),  # a failed attempt that reported its input
        ev("c1", attempt="a2", u=usage(1000, 200)),
    )
    line = report(mix(foreground=(("extract", 1),), days=1), led)["lines"][0]
    assert Decimal(line["tokens_per_operation"]["input_tokens"]) == Decimal(1300)  # per completed logical call
    assert line["tokens_complete"] is True


# ----------------------------------------------------------------- unknown stays unknown
def test_an_operation_without_a_completed_measurement_is_unknown_and_stays_out_of_the_total() -> None:
    led = ledger(ev("e1"))
    r = report(mix(foreground=(("extract", 10), ("graph", 5))), led)
    graph = next(x for x in r["lines"] if x["operation"] == "graph")
    assert graph["measured"] is False and graph["tokens_per_month"] is None and graph["cost_per_month"] is None
    assert graph["unknown_reasons"] == ["no_completed_measurement"]
    service = r["per_person_per_month"]["service"]
    assert service["tokens_total"] is None and service["cost_total"] is None  # never a total with a hole in it
    assert service["incomplete_operations"] == ["graph"]
    assert Decimal(service["cost_known_subtotal"]) > 0  # the known part is still reported, as a subtotal


def test_an_attempt_whose_usage_was_never_reported_makes_the_line_a_lower_bound() -> None:
    led = ledger(ev("f1", attempt="a1", outcome="failed", status="unavailable", u=None), ev("c1", attempt="a2"))
    line = report(mix(), led)["lines"][0]
    assert line["tokens_per_month"] is None and line["cost_per_month"] is None
    assert Decimal(line["tokens_known_per_month"]) > 0 and "1_attempts_with_unknown_usage" in line["unknown_reasons"]


def test_a_category_the_provider_did_not_report_is_unknown_not_zero() -> None:
    led = ledger(ev("c1", u=usage(1000, 200, cache_known=False)))
    line = report(mix(), led)["lines"][0]
    assert line["tokens_per_month"] is None and line["cost_per_month"] is None
    assert any(reason.startswith("category_not_reported:cache_read_tokens") for reason in line["unknown_reasons"])


def test_a_missing_rate_leaves_tokens_known_and_cost_unknown() -> None:
    led = ledger(ev("c1", u=usage(1000, 200, cache=400)))
    no_cache_rate = [card(cache=None)]
    line = report(mix(), led, no_cache_rate)["lines"][0]
    assert line["tokens_per_month"] is not None  # tokens are known
    assert line["cost_per_month"] is None and any(r.startswith("rate_missing") for r in line["unknown_reasons"])
    none_at_all = report(mix(), led, [])["lines"][0]
    assert none_at_all["cost_per_month"] is None and any(r.startswith("no_rate:") for r in none_at_all["unknown_reasons"])
    assert none_at_all["tokens_per_month"] is not None


def test_rates_are_chosen_by_date_and_a_future_rate_is_not_used() -> None:
    led = ledger(ev("c1"))
    old, new, future = card(inp="1.00", frm=date(2026, 1, 1)), card(inp="2.00", frm=date(2026, 6, 1)), card(inp="9.00", frm=date(2027, 1, 1))
    chosen = report(mix(), led, [old, new, future])["lines"][0]["cost_per_month"]
    assert chosen == report(mix(), led, [new])["lines"][0]["cost_per_month"] != report(mix(), led, [old])["lines"][0]["cost_per_month"]
    assert report(mix(), led, [future])["lines"][0]["cost_per_month"] is None


def test_mixed_currencies_do_not_add_up() -> None:
    led = ledger(ev("a", model="m-1"), ev("b", attempt="a2", model="m-2"))
    r = report(mix(), led, [card("m-1"), card("m-2", currency="EUR")])
    service = r["per_person_per_month"]["service"]
    assert service["cost_total"] is None and "mixed_currencies" in service["incomplete_operations"]


def test_infrastructure_and_other_ledgers_are_explicit() -> None:
    led = ledger(ev("c1"))
    assert report(mix(), led)["infrastructure"]["status"] == "unknown"
    half = us.InfraAllocation(monthly_cost=Decimal("100"), active_people=None, basis="declared")
    assert report(mix(), led, infrastructure=half)["infrastructure"]["per_person_per_month"] is None
    full = us.InfraAllocation(monthly_cost=Decimal("100"), active_people=40, basis="declared")
    assert report(mix(), led, infrastructure=full)["infrastructure"]["per_person_per_month"] == "2.500000"
    others = report(mix(), led)["other_ledgers"]
    assert others["consuming_agent"]["coverage"] == "not_collected" and others["evaluation"]["coverage"] == "excluded"


def test_only_the_service_ledger_is_itemised() -> None:
    consumer = ev("c9", attempt="a9")
    values = json.loads(consumer.model_dump_json())
    values["ledger"] = "consuming_agent"
    other = UsageEvent.model_validate_json(json.dumps(values))
    alone, mixed = report(mix(), ledger(ev("c1"))), report(mix(), ledger(ev("c1"), other))
    assert alone["lines"] == mixed["lines"]


def test_markdown_never_prints_an_unknown_as_zero() -> None:
    led = ledger(ev("e1"))
    text = us.render_markdown(report(mix(foreground=(("extract", 10), ("graph", 5))), led))
    graph_row = next(row for row in text.splitlines() if row.startswith("| graph"))
    assert "| unknown | unknown |" in graph_row and "no_completed_measurement" in graph_row
    assert "total unknown" in text and "Consumer-agent usage: not collected" in text and "Shared infrastructure per person per month: unknown" in text


# ----------------------------------------------------------------- duplicates, late events, receipts
def test_duplicates_and_a_late_final_event_are_counted_once() -> None:
    base = [ev("p1", attempt="a1", status="pending", outcome="in_progress")]
    cancelled = ev("x1", attempt="a1", status="unavailable", outcome="cancelled", u=None, predecessor="p1")
    late = ev("l1", attempt="a1", outcome="cancelled", late=True, predecessor="x1", u=usage(500, 50))
    good = ev("c1", attempt="a2", u=usage(1000, 200))
    once = ledger(*base, cancelled, late, good)
    replayed = ledger(*base, cancelled, late, good, late, good, cancelled, *base)  # replayed and re-delivered
    assert once == replayed
    assert report(mix(), once) == report(mix(), replayed)
    receipts = [us.ProviderReceipt(provider="google", model="m-1", input_tokens=1500, output_tokens=250, scope="run")]
    assert us.reconcile_to_receipts(once, receipts)[0]["status"] == "matched"  # the late 500 + 50 were debited once, not twice


def test_receipt_reconciliation_flags_a_missing_a_doubled_and_an_unexplained_count() -> None:
    led = ledger(ev("c1", u=usage(1000, 200)))
    def check(i, o): return us.reconcile_to_receipts(led, [us.ProviderReceipt(provider="google", model="m-1", input_tokens=i, output_tokens=o, scope="r")])[0]["status"]
    assert check(1000, 200) == "matched"
    assert check(1200, 200) == "ledger_below_receipt_unexplained"
    assert check(900, 200) == "ledger_above_receipt"  # would mean a double debit or an invented count
    with_unknown = ledger(ev("c1", u=usage(1000, 200)), ev("f1", attempt="a2", outcome="failed", status="unavailable", u=None))
    got = us.reconcile_to_receipts(with_unknown, [us.ProviderReceipt(provider="google", model="m-1", input_tokens=1300, output_tokens=200, scope="r")])[0]
    assert got["status"] == "ledger_below_receipt_with_unknown_attempts" and got["unknown_attempts"] == 1
    assert us.reconcile_to_receipts(led, [us.ProviderReceipt(provider="google", model="other", input_tokens=5, output_tokens=5, scope="r")])[0]["status"] == "ledger_below_receipt_unexplained"


def test_the_report_is_deterministic_and_json_serialisable() -> None:
    led = ledger(ev("c1"))
    a, b = report(mix(), led), report(mix(), led)
    assert us.report_json(a) == us.report_json(b)
    assert json.loads(us.report_json(a))["schema"] == us.REPORT_SCHEMA


# ----------------------------------------------------------------- the pilot, through the real capture seam
def test_the_pilot_reconciles_to_receipts_and_injection_changes_nothing_but_the_late_usage(tmp_path) -> None:
    result = pilot.run_pilot(tasks=48, workdir=tmp_path)
    assert result["synthetic"] is True
    assert result["duplicates_replayed"] == result["events_stored"] and result["conflicting_duplicate_rejected"] is True
    assert all(r["status"] == "matched" for r in result["receipts_before"])
    assert all(r["status"] == "matched" for r in result["receipts_after_late_event"])  # late usage debited exactly once
    assert result["clean_ledger"]["replay_changed_the_reconciliation"] is False
    assert all(r["status"] == "matched" for r in result["clean_ledger"]["receipts"])
    m = result["measurements"]
    assert m["provider_calls"] > 0 and m["report_bytes"]["regular"] > 0 and set(m["report_bytes"]) == {"light", "regular", "heavy"}


def test_the_pilot_keeps_unknowns_unknown_and_shows_a_complete_total_only_for_the_clean_ledger(tmp_path) -> None:
    result = pilot.run_pilot(tasks=48, workdir=tmp_path)
    measured = result["reports"]["measured"]["regular"]["per_person_per_month"]["service"]
    assert measured["tokens_total"] is None and measured["cost_total"] is None and Decimal(measured["cost_known_subtotal"]) > 0
    heavy = {line["operation"]: line for line in result["reports"]["measured"]["heavy"]["lines"]}
    assert heavy["graph_enrichment"]["unknown_reasons"] == ["no_completed_measurement"] and heavy["graph_enrichment"]["cost_per_month"] is None
    clean = result["reports"]["clean"]["regular"]["per_person_per_month"]["service"]
    assert clean["tokens_complete"] and clean["cost_complete"] and clean["cost_total"] is not None


def test_the_clean_pilot_cost_equals_a_hand_calculation_from_the_scripted_counters(tmp_path) -> None:
    """Independent of the ledger: recompute the extraction line from the counters the scripted provider returns."""

    tasks = 48
    result = pilot.run_pilot(tasks=tasks, workdir=tmp_path)
    ops = [i for i in range(tasks) if i % len(pilot.OPERATIONS) == 0]  # extraction tasks
    total = Decimal(0)
    rate = {"in": Decimal("0.30"), "cache": Decimal("0.03"), "out": Decimal("2.50"), "reason": Decimal("2.50")}
    for i in ops:
        prompt, output = 800 + 10 * (i % 7), 120 + 5 * (i % 5)
        cached, reasoning = (300 if i % 2 else 0), (40 if i % 3 == 0 else 0)
        total += ((prompt - cached) * rate["in"] + cached * rate["cache"] + output * rate["out"] + reasoning * rate["reason"]) / 1_000_000
    per_call = total / len(ops)
    line = next(x for x in result["reports"]["clean"]["regular"]["lines"] if x["operation"] == "llm_extraction" and x["kind"] == "foreground")
    assert Decimal(line["cost_per_month"]) == (per_call * 800).quantize(Decimal("0.000001"))


# ----------------------------------------------------------------- hardening
def test_conflicting_rate_cards_for_one_date_are_refused_not_chosen_by_order() -> None:
    led = ledger(ev("e1"))
    with pytest.raises(ValueError, match="conflicting rate cards"):
        report(mix(), led, [card(inp="1.00"), card(inp="9.00")])
    # an exact repeat is harmless
    assert report(mix(), led, [card(), card()])["per_person_per_month"]["service"]["cost_total"] is not None


def test_an_absurd_rate_is_refused_as_input() -> None:
    with pytest.raises(ValidationError):
        card(inp="1e30")


def test_a_very_large_ledger_never_crashes_the_arithmetic() -> None:
    big = 9_007_199_254_740_991
    led = ledger(ev("e1", u=usage(inp=big // 2, out=big // 2)))
    result = report(mix([("extract", 1_000_000)], days=31), led, [card(inp="1000000", out="1000000")])
    assert result["per_person_per_month"]["service"]["cost_total"] is not None


def test_markdown_cannot_be_broken_by_a_provider_model_name_or_declared_text() -> None:
    hostile = "m|x\n# injected heading\x00\x1b[31m"
    led = ledger(ev("e1", model=hostile))
    m = us.ActivityMix(
        name="regular", basis="a | basis\n## sneaky", agents=1, projects=1, active_days_per_month=5,
        foreground=(us.ActivityLine(operation="ex|tract\nrow", per_day=1),),
        background_days_per_month=0, background=(), missing_categories=("left\nout | x",),
    )
    text = us.render_markdown(report(m, led, []))
    assert "\x00" not in text and "\x1b" not in text
    assert "\n# injected" not in text and "\n## sneaky" not in text
    table = [row for row in text.splitlines() if row.startswith("| ") and "extract" in row.replace("\\|", "")]
    assert len(table) == 1 and table[0].count(" | ") + table[0].count("\\|") >= 5
    assert len(table[0].replace("\\|", "").split("|")) == 8  # the declared columns and nothing more
