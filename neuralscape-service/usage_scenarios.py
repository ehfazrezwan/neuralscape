"""Per-person scenario reports from reconciled usage ledgers.

A scenario report answers one question: *given what this person is declared to do in
a month, and what the service ledger measured for each operation, what would the
service's token use and cost be?*  It is deliberately modest:

* The activity mix is **declared** by the caller, with the basis it comes from.  Nothing
  here ships a default or an "average user"; the light, regular and heavy names only
  label declarations.
* Measurements come only from a reconciled service ledger
  (:func:`contracts_usage_reconcile.reconcile_usage_events`), so duplicates, corrections
  and late events are counted once.  Failed and retried attempts that reported tokens are
  part of the cost of the operation they belong to.
* Unknown stays unknown.  An operation with no completed measurement, an attempt whose
  usage was never reported, a category without a rate, or a missing allocation yields
  ``None`` and a reason, never zero.  Totals separate a *known subtotal* from a total
  that exists only when nothing is unknown.
* Tokens and money are kept apart: money needs a dated rate card, applied with the
  provider's own inclusion rules (cached tokens inside input, reasoning inside or
  additional to output) so no category is billed twice.
* Only the service ledger is itemised.  Consumer-agent usage is not collected (it needs the
  owner's opt-in) and evaluation usage is research expense; both are reported as such and
  are never added to service figures.

The module is pure: no clock, no I/O, no content.  Money is :class:`~decimal.Decimal`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, localcontext
from typing import Annotated, Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts_common import OpaqueId, SafeCounter
from contracts_usage_reconcile import UsageReconciliation

REPORT_SCHEMA = "usage-scenario-report-v1"
CATEGORIES = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_PER_MILLION = Decimal(1_000_000)
_Money = Annotated[Decimal, Field(ge=0, le=Decimal("1e12"), allow_inf_nan=False)]
_Basis = Annotated[str, Field(min_length=1, max_length=512)]


# ---------------------------------------------------------------- declared inputs
class _Declared(BaseModel):
    """Closed, strict and immutable input models.  They are report inputs, not product wire contracts."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ActivityLine(_Declared):
    """One declared operation and how many times a day it happens."""

    operation: OpaqueId
    per_day: Annotated[int, Field(ge=1, le=1_000_000)]


class ActivityMix(_Declared):
    """A declared month of one person's activity (across all their agents and projects)."""

    name: Literal["light", "regular", "heavy"]
    basis: _Basis  # where the declaration comes from; required, never implied
    agents: Annotated[int, Field(ge=1, le=1000)]
    projects: Annotated[int, Field(ge=1, le=1000)]
    active_days_per_month: Annotated[int, Field(ge=1, le=31)]
    foreground: tuple[ActivityLine, ...]
    background_days_per_month: Annotated[int, Field(ge=0, le=31)]
    background: tuple[ActivityLine, ...]
    missing_categories: tuple[_Basis, ...]  # what the author knows this mix leaves out

    @model_validator(mode="after")
    def _check(self) -> "ActivityMix":
        if not self.foreground:
            raise ValueError("a mix needs at least one foreground operation")
        if bool(self.background) != bool(self.background_days_per_month):
            raise ValueError("background operations and background days must be declared together")
        return self


class RateCard(_Declared):
    """A dated provider rate, in ``currency`` per million tokens.  An absent rate is unknown."""

    provider: OpaqueId
    model: OpaqueId
    effective_from: date
    currency: Annotated[str, Field(min_length=3, max_length=3)]
    input_per_mtok: _Money
    output_per_mtok: _Money
    cache_read_per_mtok: _Money | None
    cache_write_per_mtok: _Money | None
    reasoning_per_mtok: _Money | None


class InfraAllocation(_Declared):
    """Shared infrastructure for the active population, as declared; both numbers are optional."""

    monthly_cost: _Money | None
    active_people: Annotated[int, Field(ge=1)] | None
    basis: _Basis


class ProviderReceipt(_Declared):
    """What a provider (or client) reported for a measured run, to reconcile the ledger against."""

    provider: OpaqueId
    model: OpaqueId
    input_tokens: SafeCounter
    output_tokens: SafeCounter
    scope: _Basis


# ---------------------------------------------------------------- measurement
@dataclass
class _Sums:
    """Known token sums for one provider, model and inclusion convention."""

    cache_read_inclusion: str
    reasoning_inclusion: str
    tokens: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    unknown: set[str] = field(default_factory=set)  # categories some attempt did not report


@dataclass
class OperationProfile:
    """What the service ledger measured for one operation (all attempts, retries included)."""

    operation: str
    attempts: int = 0
    completed: int = 0  # completed logical calls
    unknown_attempts: int = 0  # attempts whose total usage is not known
    sums: dict[tuple[str, str, str, str], _Sums] = field(default_factory=dict)


def operation_profiles(reconciliation: UsageReconciliation) -> dict[str, OperationProfile]:
    """Per-operation measurements from the reconciled **service** streams only."""

    profiles: dict[str, OperationProfile] = {}
    for stream in reconciliation.streams:
        if stream.ledger != "service":
            continue
        profile = profiles.setdefault(stream.operation, OperationProfile(stream.operation))
        profile.attempts += 1
        if stream.status == "final" and stream.attempt_outcome == "completed":
            profile.completed += 1
        if stream.total_tokens is None:
            profile.unknown_attempts += 1
        usage = stream.usage
        if usage is None:
            continue
        key = (
            stream.attribution.provider or "unknown",
            stream.attribution.model or "unknown",
            usage.cache_read_inclusion,
            usage.reasoning_inclusion,
        )
        sums = profile.sums.setdefault(key, _Sums(usage.cache_read_inclusion, usage.reasoning_inclusion))
        for category in CATEGORIES:
            quantity = getattr(usage, category)
            if quantity.value is not None:
                sums.tokens[category] += quantity.value
            elif quantity.missing_reason != "not_applicable":
                sums.unknown.add(category)
    return profiles


def _bill(sums: _Sums, scale: Decimal, card: RateCard | None) -> tuple[Decimal, bool]:
    """Money for ``sums`` times ``scale`` and whether it is complete.

    Cached tokens inside input are billed at the cache rate and removed from input;
    reasoning tokens additional to output are billed at the reasoning rate and never
    at the output rate.  A category with tokens but no rate makes the figure incomplete.
    """

    if card is None:
        return Decimal(0), False
    t = sums.tokens
    complete = not sums.unknown
    total = Decimal(0)
    input_tokens = t["input_tokens"]
    if sums.cache_read_inclusion == "included_in_input":
        input_tokens = max(0, input_tokens - t["cache_read_tokens"])
    total += Decimal(input_tokens) * scale * card.input_per_mtok / _PER_MILLION
    total += Decimal(t["output_tokens"]) * scale * card.output_per_mtok / _PER_MILLION
    extras = [("cache_read_tokens", card.cache_read_per_mtok), ("cache_write_tokens", card.cache_write_per_mtok)]
    if sums.reasoning_inclusion == "additional_to_output":
        extras.append(("reasoning_tokens", card.reasoning_per_mtok))
    for category, rate in extras:
        if t[category] == 0:
            continue
        if rate is None:
            complete = False
        else:
            total += Decimal(t[category]) * scale * rate / _PER_MILLION
    return total, complete


def _card_for(provider: str, model: str, as_of: date, cards: Iterable[RateCard]) -> RateCard | None:
    eligible = [c for c in cards if c.provider == provider and c.model == model and c.effective_from <= as_of]
    return max(eligible, key=lambda c: c.effective_from) if eligible else None


def _q(value: Decimal, places: int = 6) -> str:
    return format(value.quantize(Decimal(1).scaleb(-places)), "f")


# ---------------------------------------------------------------- the report
def scenario_report(
    mix: ActivityMix,
    reconciliation: UsageReconciliation,
    rate_cards: Iterable[RateCard],
    *,
    as_of: date,
    infrastructure: InfraAllocation | None = None,
) -> dict[str, Any]:
    """The month for one declared mix, from the service ledger (see ``_scenario_report``)."""

    # Wide enough that no accepted ledger and rate overflow or round the money arithmetic.
    with localcontext() as context:
        context.prec = 100
        return _scenario_report(mix, reconciliation, rate_cards, as_of=as_of, infrastructure=infrastructure)


def _scenario_report(
    mix: ActivityMix,
    reconciliation: UsageReconciliation,
    rate_cards: Iterable[RateCard],
    *,
    as_of: date,
    infrastructure: InfraAllocation | None = None,
) -> dict[str, Any]:
    """The month for one declared mix, from the service ledger.  JSON-serialisable.

    ``as_of`` selects the rate card (the latest one effective on that date); it is stated in the
    report.  Nothing is defaulted: a line without a measurement, a rate or a complete category
    set reports ``None`` with the reasons, and a total is present only when nothing was unknown.
    """

    cards = tuple(rate_cards)
    seen: dict[tuple[str, str, date], RateCard] = {}
    for rate in cards:
        if seen.setdefault((rate.provider, rate.model, rate.effective_from), rate) != rate:
            raise ValueError("conflicting rate cards for one provider, model and date")
    profiles = operation_profiles(reconciliation)
    lines: list[dict[str, Any]] = []
    currencies: set[str] = set()
    for kind, declared, days in (
        ("foreground", mix.foreground, mix.active_days_per_month),
        ("background", mix.background, mix.background_days_per_month),
    ):
        for item in declared:
            lines.append(_line(kind, item, days, profiles.get(item.operation), cards, as_of, currencies))

    tokens_known = sum((Decimal(line["tokens_known_per_month"]) for line in lines), Decimal(0))
    cost_known = sum((Decimal(line["cost_known_per_month"]) for line in lines), Decimal(0))
    tokens_complete = all(line["tokens_complete"] for line in lines)
    mixed = len(currencies) > 1
    cost_complete = all(line["cost_complete"] for line in lines) and not mixed
    service = {
        "tokens_complete": tokens_complete,
        "cost_complete": cost_complete,
        "tokens_known_subtotal": _q(tokens_known, 3),
        "tokens_total": _q(tokens_known, 3) if tokens_complete else None,
        "cost_known_subtotal": _q(cost_known),
        "cost_total": _q(cost_known) if cost_complete else None,
        "currency": next(iter(currencies)) if len(currencies) == 1 else None,
        "incomplete_operations": [line["operation"] for line in lines if not (line["tokens_complete"] and line["cost_complete"])]
        + (["mixed_currencies"] if mixed else []),
    }
    return {
        "schema": REPORT_SCHEMA,
        "mix": mix.model_dump(mode="json"),
        "as_of": as_of.isoformat(),
        "per_person_per_month": {"agents": mix.agents, "projects": mix.projects, "service": service},
        "lines": lines,
        "infrastructure": _infrastructure(infrastructure),
        "other_ledgers": {
            "consuming_agent": {
                "coverage": "not_collected",
                "reason": "consumer metering is opt-in and consent is pending; unknown, never zero",
            },
            "evaluation": {"coverage": "excluded", "reason": "research expense, never part of service figures"},
        },
        "measurement_basis": {
            "ledger": "service",
            "operations_measured": {
                op: {"attempts": p.attempts, "completed": p.completed, "unknown_attempts": p.unknown_attempts}
                for op, p in sorted(profiles.items())
            },
        },
        "caveats": [
            "The mix is a declaration with the stated basis, not an observed or average population.",
            "Costs use the rate cards effective on as_of; unknown rates and unknown tokens stay unknown.",
            "Hypothetical avoided work is not counted; no savings are claimed.",
        ],
    }


def _line(
    kind: str,
    item: ActivityLine,
    days: int,
    profile: OperationProfile | None,
    cards: tuple[RateCard, ...],
    as_of: date,
    currencies: set[str],
) -> dict[str, Any]:
    per_month = item.per_day * days
    out: dict[str, Any] = {
        "kind": kind,
        "operation": item.operation,
        "logical_operations_per_month": per_month,
        "measured": False,
        "tokens_per_operation": None,
        "tokens_per_month": None,
        "tokens_known_per_month": "0.000",
        "tokens_complete": False,
        "cost_per_month": None,
        "cost_known_per_month": "0.000000",
        "cost_complete": False,
        "unknown_reasons": [],
    }
    reasons: list[str] = out["unknown_reasons"]
    if profile is None or profile.completed == 0:
        reasons.append("no_completed_measurement")
        return out
    out["measured"] = True
    scale = Decimal(per_month) / Decimal(profile.completed)  # every attempt of the operation, retries included
    tokens_ok = True
    if profile.unknown_attempts:
        reasons.append(f"{profile.unknown_attempts}_attempts_with_unknown_usage")
        tokens_ok = False
    per_op: dict[str, Decimal] = dict.fromkeys(CATEGORIES, Decimal(0))
    known_tokens = Decimal(0)
    cost = Decimal(0)
    cost_ok = True
    for (provider, model, _cache, _reasoning), sums in sorted(profile.sums.items()):
        for category in CATEGORIES:
            per_op[category] += Decimal(sums.tokens[category]) / Decimal(profile.completed)
        for category in sorted(sums.unknown):
            reasons.append(f"category_not_reported:{category}:{provider}/{model}")
            tokens_ok = False
        # Totalling rule of the reconciler: a category already inside a base is not added again.
        total = sums.tokens["input_tokens"] + sums.tokens["output_tokens"]
        if sums.cache_read_inclusion == "additional_to_input":
            total += sums.tokens["cache_read_tokens"] + sums.tokens["cache_write_tokens"]
        if sums.reasoning_inclusion == "additional_to_output":
            total += sums.tokens["reasoning_tokens"]
        known_tokens += Decimal(total)
        card = _card_for(provider, model, as_of, cards)
        if card is None:
            reasons.append(f"no_rate:{provider}/{model}")
            cost_ok = False
        else:
            currencies.add(card.currency)
        money, billed_ok = _bill(sums, scale, card)
        cost += money
        if card is not None and not billed_ok and not sums.unknown:
            reasons.append(f"rate_missing_for_a_reported_category:{provider}/{model}")
        cost_ok = cost_ok and billed_ok
    out["tokens_per_operation"] = {c: _q(v, 3) for c, v in per_op.items()}
    out["tokens_known_per_month"] = _q(known_tokens * scale, 3)
    out["tokens_complete"] = tokens_ok
    if tokens_ok:
        out["tokens_per_month"] = out["tokens_known_per_month"]
    out["cost_known_per_month"] = _q(cost)
    out["cost_complete"] = cost_ok and tokens_ok  # unknown tokens make the money a lower bound
    if out["cost_complete"]:
        out["cost_per_month"] = _q(cost)
    return out


def _infrastructure(allocation: InfraAllocation | None) -> dict[str, Any]:
    if allocation is None or allocation.monthly_cost is None or allocation.active_people is None:
        return {
            "status": "unknown",
            "per_person_per_month": None,
            "basis": None if allocation is None else allocation.basis,
            "reason": "shared infrastructure cost and active population are not both declared",
        }
    return {
        "status": "declared",
        "per_person_per_month": _q(allocation.monthly_cost / Decimal(allocation.active_people)),
        "monthly_cost": _q(allocation.monthly_cost),
        "active_people": allocation.active_people,
        "basis": allocation.basis,
    }


# ---------------------------------------------------------------- receipts
def reconcile_to_receipts(
    reconciliation: UsageReconciliation, receipts: Iterable[ProviderReceipt]
) -> list[dict[str, Any]]:
    """Compare the ledger's known input/output tokens with provider or client receipts.

    ``matched`` means equal with no unknown attempt.  A ledger below a receipt is explained only
    by attempts whose usage the ledger knows it does not know; otherwise it is flagged.  A ledger
    above a receipt is always flagged (it would mean a double debit or an invented count).
    """

    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "unknown": 0})
    for stream in reconciliation.streams:
        if stream.ledger != "service":
            continue
        key = (stream.attribution.provider or "unknown", stream.attribution.model or "unknown")
        if stream.total_tokens is None:
            totals[key]["unknown"] += 1
        if stream.usage is not None:
            totals[key]["input"] += stream.usage.input_tokens.value or 0
            totals[key]["output"] += stream.usage.output_tokens.value or 0
    results = []
    for receipt in receipts:
        ledger = totals.get((receipt.provider, receipt.model), {"input": 0, "output": 0, "unknown": 0})
        below = ledger["input"] < receipt.input_tokens or ledger["output"] < receipt.output_tokens
        above = ledger["input"] > receipt.input_tokens or ledger["output"] > receipt.output_tokens
        if above:
            status = "ledger_above_receipt"
        elif below:
            status = "ledger_below_receipt_with_unknown_attempts" if ledger["unknown"] else "ledger_below_receipt_unexplained"
        else:
            status = "matched" if not ledger["unknown"] else "matched_on_known_with_unknown_attempts"
        results.append(
            {
                "provider": receipt.provider,
                "model": receipt.model,
                "scope": receipt.scope,
                "receipt": {"input_tokens": receipt.input_tokens, "output_tokens": receipt.output_tokens},
                "ledger_known": {"input_tokens": ledger["input"], "output_tokens": ledger["output"]},
                "unknown_attempts": ledger["unknown"],
                "status": status,
            }
        )
    return results


# ---------------------------------------------------------------- rendering
def _text(value: Any) -> str:
    """One printable line with table pipes escaped: ledger, provider and declared strings are data."""

    flat = "".join(c if c.isprintable() else " " for c in str(value))
    return " ".join(flat.split()).replace("\\", "\\\\").replace("|", "\\|")


def render_markdown(report: dict[str, Any]) -> str:
    """A readable rendering; unknown values print as ``unknown`` with their reasons, never as 0."""

    mix, person = report["mix"], report["per_person_per_month"]
    service = person["service"]

    def show(value: str | None, unit: str = "") -> str:
        return "unknown" if value is None else f"{value}{unit}"

    rows = [
        f"# Scenario: {mix['name']} (declared)",
        "",
        f"Basis of the declaration: {_text(mix['basis'])}",
        f"Rates as of {report['as_of']}. One person, {mix['agents']} agent(s), {mix['projects']} project(s), "
        f"{mix['active_days_per_month']} active day(s) a month.",
        "",
        "| operation | kind | ops per month | tokens per month | cost per month | why unknown |",
        "|---|---|---|---|---|---|",
    ]
    for line in report["lines"]:
        rows.append(
            f"| {_text(line['operation'])} | {line['kind']} | {line['logical_operations_per_month']} | "
            f"{show(line['tokens_per_month'])} | {show(line['cost_per_month'], ' ' + _text(service['currency'] or ''))} | "
            f"{_text(', '.join(line['unknown_reasons'])) or '-'} |"
        )
    rows += [
        "",
        f"Service tokens per person per month: known subtotal {service['tokens_known_subtotal']}, "
        f"total {show(service['tokens_total'])}.",
        f"Service cost per person per month: known subtotal {service['cost_known_subtotal']}, "
        f"total {show(service['cost_total'])} {_text(service['currency'] or '')}".rstrip() + ".",
        f"Shared infrastructure per person per month: {show(report['infrastructure']['per_person_per_month'])}.",
        "Consumer-agent usage: not collected (opt-in pending). Evaluation usage: excluded.",
    ]
    if mix["missing_categories"]:
        rows.append("Left out by the author: " + "; ".join(_text(c) for c in mix["missing_categories"]) + ".")
    rows.append("")
    rows += [f"- {caveat}" for caveat in report["caveats"]]
    return "\n".join(rows) + "\n"


def report_json(report: dict[str, Any]) -> str:
    """Canonical JSON (sorted keys), the form whose size the pilot measures."""

    return json.dumps(report, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ActivityLine",
    "ActivityMix",
    "InfraAllocation",
    "OperationProfile",
    "ProviderReceipt",
    "RateCard",
    "REPORT_SCHEMA",
    "operation_profiles",
    "reconcile_to_receipts",
    "render_markdown",
    "report_json",
    "scenario_report",
]
