"""Functional and measurement pilot for per-person scenario reports.

The pilot drives the real capture seam (the retry helper with a scripted provider that returns
Gemini-shaped usage metadata) into a real SQLite usage store, injects duplicate, conflicting and
late events, reconciles the stored events, reconciles the known counters to the provider's own
receipt totals, and generates the report for three *declared* mixes.  It reports bytes, tokens and
latency of the report path.  Everything is synthetic: the counters are invented by the scripted
provider, the rates are illustrative, and the mixes are declarations with a stated basis, not
observations of any population.  Nothing here is a cost or savings claim.

Run: ``python usage_scenario_pilot.py [--tasks N]`` prints the measurements as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import usage_capture
from contracts_usage import UsageEvent
from contracts_usage_reconcile import reconcile_usage_events
from usage_scenarios import (
    ActivityLine,
    ActivityMix,
    InfraAllocation,
    ProviderReceipt,
    RateCard,
    reconcile_to_receipts,
    render_markdown,
    report_json,
    scenario_report,
)
from usage_store import SqliteUsageStore, UsageEventConflict

TENANT = "pilot-tenant"
PRIMARY, FALLBACK = "model-primary", "model-fallback"
OPERATIONS = ("LLM extraction", "LLM summary")


class ScriptedProvider:
    """Returns Gemini-shaped usage metadata and keeps its own receipt totals."""

    def __init__(self) -> None:
        self.receipt: dict[str, dict[str, int]] = {}
        self.calls = 0

    def respond(self, model: str, prompt: int, output: int, cached: int, reasoning: int) -> Any:
        totals = self.receipt.setdefault(model, {"input": 0, "output": 0})
        totals["input"] += prompt
        totals["output"] += output
        self.calls += 1
        return SimpleNamespace(
            text="ok",
            model_version=model,
            usage_metadata=SimpleNamespace(
                prompt_token_count=prompt,
                candidates_token_count=output,
                cached_content_token_count=cached,
                thoughts_token_count=reasoning,
            ),
        )


def _run_task(index: int, operation: str, provider: ScriptedProvider, failures: bool = True) -> None:
    """One logical call whose shape is chosen by the task index: clean, retried, fallback or failed."""

    from memory.retry import retry_transient

    shape = index % 8
    if not failures:
        shape = 0  # a clean ledger: every call succeeds on the first attempt
    prompt, output = 800 + 10 * (index % 7), 120 + 5 * (index % 5)
    cached, reasoning = (300 if index % 2 else 0), (40 if index % 3 == 0 else 0)

    def generate_content(model: str | None = None, contents: Any = None) -> Any:
        if shape == 1 and not state["retried"]:
            state["retried"] = True
            raise RuntimeError("503 UNAVAILABLE")
        if shape == 2 and model == PRIMARY:
            raise RuntimeError("503 UNAVAILABLE")
        if shape == 3:
            raise RuntimeError("503 UNAVAILABLE")  # every attempt fails: the usage stays unknown
        return provider.respond(model or PRIMARY, prompt, output, cached, reasoning)

    state = {"retried": False}
    context = usage_capture.AttemptContext(tenant_id=TENANT, task_id=f"task-{index}", attempt_id=f"task-{index}:a")
    with usage_capture.attempt_context(context):
        try:
            retry_transient(
                generate_content, model=PRIMARY, contents="x", max_retries=1, base_delay=0, max_delay=0,
                operation=operation, fallback_model=FALLBACK,
            )
        except RuntimeError:
            pass


def _late_chain(base: UsageEvent) -> list[UsageEvent]:
    """A cancelled attempt followed by its late final event, built from a stored pending event."""

    def make(**changes: Any) -> UsageEvent:
        values = base.model_dump(mode="json")
        values.update(changes)
        return UsageEvent.model_validate_json(json.dumps(values))

    usage = {
        "measurement": "actual",
        "input_tokens": {"value": 500, "missing_reason": None},
        "output_tokens": {"value": 50, "missing_reason": None},
        "cache_read_tokens": {"value": None, "missing_reason": "not_applicable"},
        "cache_write_tokens": {"value": None, "missing_reason": "not_applicable"},
        "reasoning_tokens": {"value": None, "missing_reason": "not_applicable"},
        "cache_read_inclusion": "not_applicable",
        "cache_write_inclusion": "not_applicable",
        "reasoning_inclusion": "not_applicable",
    }
    attribution = {**base.model_dump(mode="json")["attribution"], "provider": "google", "provider_missing_reason": None}
    cancelled = make(
        event_id="late-cancelled", predecessor_event_id=base.event_id, status="unavailable", attribution=attribution,
        attempt_outcome="cancelled", usage=None, usage_missing_reason="not_yet_reported",
    )
    late = make(
        event_id="late-final", predecessor_event_id="late-cancelled", status="final", attribution=attribution,
        attempt_outcome="cancelled", late_after_cancellation=True, usage=usage, usage_missing_reason=None,
    )
    return [cancelled, late]


def declared_mixes() -> list[ActivityMix]:
    """Three illustrative *declarations*; the basis says they are not observed or representative."""

    basis = "illustrative pilot declaration; not an observed or representative population"
    common = {"basis": basis, "missing_categories": ("consumer agent usage", "storage and egress", "dreaming and maintenance crons")}
    return [
        ActivityMix(name="light", agents=1, projects=1, active_days_per_month=8, background_days_per_month=0, background=(),
                    foreground=(ActivityLine(operation="llm_extraction", per_day=10),), **common),
        ActivityMix(name="regular", agents=2, projects=2, active_days_per_month=20, background_days_per_month=30,
                    background=(ActivityLine(operation="llm_summary", per_day=2),),
                    foreground=(ActivityLine(operation="llm_extraction", per_day=40), ActivityLine(operation="llm_summary", per_day=5)),
                    **common),
        ActivityMix(name="heavy", agents=4, projects=3, active_days_per_month=26, background_days_per_month=30,
                    background=(ActivityLine(operation="llm_summary", per_day=6), ActivityLine(operation="graph_enrichment", per_day=20)),
                    foreground=(ActivityLine(operation="llm_extraction", per_day=150), ActivityLine(operation="llm_summary", per_day=20)),
                    **common),
    ]


def illustrative_rates(complete: bool = False) -> list[RateCard]:
    """Invented rates, effective from a fixed date: they exist so the arithmetic can be exercised."""

    base = {"provider": "google", "effective_from": date(2026, 1, 1), "currency": "USD"}
    return [
        RateCard(model=PRIMARY, input_per_mtok=Decimal("0.30"), output_per_mtok=Decimal("2.50"),
                 cache_read_per_mtok=Decimal("0.03"), cache_write_per_mtok=None, reasoning_per_mtok=Decimal("2.50"), **base),
        # The fallback model has no cache or reasoning rate: costs that need them stay unknown.
        RateCard(model=FALLBACK, input_per_mtok=Decimal("0.10"), output_per_mtok=Decimal("0.40"),
                 cache_read_per_mtok=Decimal("0.01") if complete else None, cache_write_per_mtok=None,
                 reasoning_per_mtok=Decimal("0.40") if complete else None, **base),
    ]


def _collect(directory: Path, tasks: int, failures: bool) -> tuple[SqliteUsageStore, ScriptedProvider, float]:
    store = SqliteUsageStore(str(directory / "usage.db"))
    provider = ScriptedProvider()
    previous = usage_capture.set_usage_recorder(usage_capture.DurableUsageRecorder(store, retry_delay=0.0))
    started = time.perf_counter()
    try:
        for index in range(tasks):
            _run_task(index, OPERATIONS[index % len(OPERATIONS)], provider, failures)
    finally:
        usage_capture.set_usage_recorder(previous)
    return store, provider, time.perf_counter() - started


def _receipts(provider: ScriptedProvider, extra_primary: tuple[int, int] = (0, 0)) -> list[ProviderReceipt]:
    return [
        ProviderReceipt(
            provider="google", model=model,
            input_tokens=t["input"] + (extra_primary[0] if model == PRIMARY else 0),
            output_tokens=t["output"] + (extra_primary[1] if model == PRIMARY else 0),
            scope="pilot run",
        )
        for model, t in sorted(provider.receipt.items())
    ]


def run_pilot(tasks: int = 64, workdir: Path | None = None) -> dict[str, Any]:
    """Run the whole pilot and return its measurements and reports.

    Two synthetic ledgers are built through the real capture seam: one *as measured* (retried, fallback and
    failed calls leave attempts whose usage stays unknown, and the fallback model has no cache or
    reasoning rate) and one *clean* (every call succeeds first time, complete rates) to show a complete total.
    """

    logging.getLogger("memory.retry").setLevel(logging.CRITICAL)  # the scripted failures are expected
    directory = workdir or Path(tempfile.mkdtemp(prefix="usage-pilot-"))
    clock = time.perf_counter
    infra = InfraAllocation(monthly_cost=None, active_people=None, basis="no allocation declared in the pilot")
    mixes, as_of = declared_mixes(), date(2026, 10, 16)

    # -- as measured, with duplicate, conflicting and late events injected
    measured_dir = directory / "measured"
    measured_dir.mkdir(parents=True, exist_ok=True)
    store, provider, build_s = _collect(measured_dir, tasks, failures=True)
    events = store.events(TENANT)
    before = reconcile_usage_events(events)
    receipt_check_before = reconcile_to_receipts(before, _receipts(provider))
    duplicates = sum(1 for event in events if store.append(event) is False)
    conflict_rejected = False
    try:
        store.append(events[0].model_copy(update={"operation": "changed"}))
    except (UsageEventConflict, ValueError):
        conflict_rejected = True
    pending = next(e for e in events if e.status == "pending")
    base = pending.model_copy(update={"event_id": "late-base", "attempt_id": "late-attempt", "task_id": "task-late"})
    store.append(base)
    chain = _late_chain(base)
    for _delivery in range(2):  # the same cancelled and late events delivered twice
        for event in chain:
            store.append(event)
    after_events = store.events(TENANT)
    t0 = clock()
    after = reconcile_usage_events(after_events)
    reconcile_s = clock() - t0
    receipt_check_after = reconcile_to_receipts(after, _receipts(provider, extra_primary=(500, 50)))
    rates = illustrative_rates()
    t0 = clock()
    measured = {m.name: scenario_report(m, after, rates, as_of=as_of, infrastructure=infra) for m in mixes}
    report_s = clock() - t0
    store.close()

    # -- clean ledger, complete rates
    clean_dir = directory / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_store, clean_provider, _ = _collect(clean_dir, tasks, failures=False)
    clean_before = reconcile_usage_events(clean_store.events(TENANT))
    for event in clean_store.events(TENANT):
        clean_store.append(event)  # replay: no double debit
    clean_after = reconcile_usage_events(clean_store.events(TENANT))
    clean = {
        m.name: scenario_report(m, clean_after, illustrative_rates(complete=True), as_of=as_of, infrastructure=infra)
        for m in mixes if m.name != "heavy"  # the heavy mix declares an operation this ledger never measured
    }
    clean_receipts = reconcile_to_receipts(clean_after, _receipts(clean_provider))
    clean_store.close()

    markdown = {f"{kind}:{name}": render_markdown(report)
                for kind, group in (("measured", measured), ("clean", clean)) for name, report in group.items()}
    sizes = {name: len(report_json(report).encode()) for name, report in measured.items()}
    return {
        "synthetic": True,
        "tasks": tasks,
        "events_stored": len(events),
        "events_after_injection": len(after_events),
        "duplicates_replayed": duplicates,
        "conflicting_duplicate_rejected": conflict_rejected,
        "receipts_before": receipt_check_before,
        "receipts_after_late_event": receipt_check_after,
        "clean_ledger": {
            "replay_changed_the_reconciliation": clean_before != clean_after,
            "receipts": clean_receipts,
        },
        "reports": {"measured": measured, "clean": clean},
        "markdown": markdown,
        "measurements": {
            "build_ledger_seconds": round(build_s, 4),
            "reconcile_seconds": round(reconcile_s, 4),
            "report_seconds_three_mixes": round(report_s, 4),
            "report_bytes": sizes,
            "markdown_bytes": {name: len(text.encode()) for name, text in markdown.items()},
            "provider_calls": provider.calls,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=int, default=64)
    args = parser.parse_args()
    result = run_pilot(args.tasks)
    slim = {k: v for k, v in result.items() if k not in ("reports", "markdown")}
    print(json.dumps(slim, indent=2, sort_keys=True, default=str))
    for text in result["markdown"].values():
        print(text)


if __name__ == "__main__":
    main()
