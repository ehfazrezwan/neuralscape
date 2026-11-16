# Per-person scenario reports and the usage pilot

`neuralscape-service/usage_scenarios.py` turns a **declared** month of one person's activity and the service's
**reconciled usage ledger** into a report of tokens and cost. `usage_scenario_pilot.py` runs a synthetic pilot of it
end to end. The report is a calculator over measurements; it is not a price list and it makes no savings claim.

## What goes in

| Input | Where it comes from | Notes |
|---|---|---|
| Activity mix (`ActivityMix`) | the caller | Named `light`, `regular` or `heavy`, with the **basis** it was declared from (required), the logical operations per active day, active days per month, any background operations, how many agents and projects the one person spans, and the categories the author knows it leaves out. There is no default and no "average user". |
| Measurements | the **service** ledger, reconciled with `reconcile_usage_events` | Duplicates, corrections and late events are counted once. Every attempt that reported tokens is part of its operation's cost, so retries and fallbacks are included. |
| Rate cards (`RateCard`) | the caller | Dated, per million tokens, per provider and model. The card effective on the report's `as_of` date is used; a later card is ignored. An absent rate is unknown. |
| Infrastructure allocation (`InfraAllocation`) | the caller, optional | Both a monthly cost and an active population are needed for a per-person share; otherwise it is reported unknown. |
| Provider or client receipts (`ProviderReceipt`) | the caller, optional | Used by `reconcile_to_receipts` to check the ledger against run totals. |

## Rules the report follows

* **Unknown stays unknown.** An operation with no completed measurement, an attempt whose usage was never reported, a
  category the provider did not report, and a missing rate each make the affected figure `null` with a reason. A total
  exists only when nothing in it is unknown; otherwise the report gives a *known subtotal* next to a `null` total.
  The Markdown rendering prints `unknown`, never `0`.
* **Tokens and money are separate.** Cost applies the provider's inclusion rules: cached tokens inside input are billed
  once, at the cache rate, and removed from the input charge; reasoning tokens additional to output are billed at the
  reasoning rate and never at the output rate; reasoning inside output is not billed again.
* **Only the service ledger is itemised.** Consumer-agent usage is not collected (it is opt-in) and is reported as
  `not_collected`; evaluation usage is research expense and is excluded. Neither is ever added to service figures.
* **No double debit.** Replaying events, re-delivering a late event, and a conflicting duplicate under an existing event
  id do not change the figures; the pilot checks this, and checks that late usage is debited exactly once against a
  receipt.
* **Content-free.** Reports contain operation names, counts, token categories and money; no prompts or memory content.

## Reading a report

`scenario_report(mix, reconciliation, rate_cards, as_of=..., infrastructure=...)` returns a JSON-serialisable dict:
`lines` (one per declared operation: operations per month, per-operation and per-month tokens, cost, and the reasons
anything is unknown), `per_person_per_month.service` (known subtotal and total for tokens and cost), `infrastructure`,
`other_ledgers`, `measurement_basis` (attempts, completed calls and unknown attempts per operation) and `caveats`.
`render_markdown` and `report_json` give a readable and a canonical form.

## The pilot

`python usage_scenario_pilot.py [--tasks N]` (from `neuralscape-service/`) drives the real capture seam with a scripted
provider that returns provider-shaped usage metadata, stores the events in a SQLite usage store, injects duplicate,
conflicting and late events, reconciles the stored events, reconciles the known counters to the provider's own receipt
totals, and generates reports for three declared mixes from two synthetic ledgers (one *as measured*, with retries,
fallback and failed calls, and one *clean*, to show a complete total). All counters, rates and mixes are invented for the
exercise.

Measured on one machine (synthetic ledgers, single process):

| Tasks | Events stored | Build ledger | Reconcile | Three reports | Report size |
|---|---|---|---|---|---|
| 64 | 208 | 0.5 s | 0.10 s | 0.9 ms | 2.1 to 3.6 KB each |
| 2,000 | 6,500 | 6.6 s | 3.0 s | 18 ms | 2.2 to 3.6 KB each |

The report size does not grow with the ledger; reconciliation is linear in the number of events and dominates the cost.
These figures describe the instrumentation, not provider cost or any product saving.

## Where events come from in the running service

With `USAGE_STORE_ENABLED=true` the API and every worker queue open the store at start-up and install the durable
recorder before the memory service builds its provider clients (`usage_store.install_usage_capture`). A provider call
records an event only when it runs under an attempt context, which names the person (the tenant) it is billed to:

* **ARQ jobs** (every queue): the worker's `on_job_start` hook binds the job to the user who enqueued it, read from the
  enqueue-time task-owner record; a job with no owner record (a cron, a job another job enqueued) is recorded under the
  tenant `unattributed`, not dropped. This covers extraction, graph enrichment and the other provider calls a job makes.
* **Search and Ask** on the API and MCP side: `MemoryService.search`, `search_graph` and `ask.ask_memory` bind the caller.
  Ask's answering model call is recorded as `llm_ask`; the embeddings of its searches are recorded as `mem0_embedding`.
* A caller that already bound a context (a ledger stage, with its ledger attempt) keeps it.

Calls made outside these entry points (a script, a one-off CLI) are not recorded. The SQLite backend is a file per
process host; with the API and the workers in separate containers use the PostgreSQL backend so one ledger holds
everything. No route reads the ledger in this slice.

## Limits

* Mixes are declarations. Representative activity distributions and the allocation of shared infrastructure and
  maintenance are open decisions; the report asks for them as explicit inputs rather than assuming them.
* Operations the service ledger has not measured (for example graph enrichment in the pilot) are unknown, not estimated.
* Storage, egress and background maintenance are outside the service token ledger and appear only as declared
  allocation or as "left out by the author".
* A task-quality or savings comparison needs controlled trials; nothing here is evidence for one.
