# 24 — Competitive Roadmap: Memory Parity and Beyond

**Date:** 2026-07-04
**Status:** Living document — each phase item is intended to ship as its own PR against `dev`.
**Inputs:** deep source-level research of three reference systems (2026-07-04):

| System | What it is | Scale signal |
|---|---|---|
| **MemPalace** (`github.com/MemPalace/mempalace`, MIT) | Local-first verbatim memory palace for coding agents; spatial hierarchy (wings/rooms/halls/drawers), Zettelkasten index cards, no LLM extraction by philosophy | ~57k ★ |
| **Honcho** (`github.com/plastic-labs/honcho`, AGPL) | "Memory that reasons" — Postgres-only server; deriver → summarizer → dreamer → dialectic pipeline; theory-of-mind (observer, observed) collections | hosted platform + very active OSS |
| **claude-mem** (`github.com/thedotmack/claude-mem`, Apache) | Dominant Claude Code memory plugin; live observer compression, progressive-disclosure timeline injection, File Read Gate | ~86k ★ |

Full research reports live outside the repo (session artifacts); this document is the distilled
comparison and the build plan.

---

## 1. Where Neuralscape already wins

These are the moats. Roadmap items must not regress them.

1. **Memory lifecycle.** None of the three has real consolidation. MemPalace forbids it by
   philosophy, claude-mem is append-only forever (30-second dedup window only), and Honcho's
   dreamer *hard-deletes* superseded conclusions. Neuralscape's dreaming sweep
   (MERGE / INVALIDATE / PRUNE / REWRITE / TEMPORAL-REFRAME) on top of Graphiti's
   **bi-temporal invalidation** answers "what was true when" — nobody else can.
2. **A real knowledge graph.** Entities, typed edges, communities, temporal validity windows.
   Honcho's only structure is a premise DAG; MemPalace has a manually-populated SQLite triple
   store; claude-mem has nothing queryable.
3. **Memory taxonomy + knowledge adapters.** 13 categories with auto-scoping plus pluggable
   per-domain taxonomy/ontology/chunkers/extractors. claude-mem's "modes" independently
   converged on the same envelope (its concept vocabulary is literally ours) — but modes are
   prompt-only; adapters swap the whole pipeline.
4. **Document ingestion.** Docling/MarkItDown parsing, archive-bomb guards, artifact storage,
   `source_ref` provenance with re-fetch handles, connector sync. The others ingest transcripts
   (claude-mem), messages (Honcho), or local files verbatim (MemPalace).
5. **Human-readable memory.** The Obsidian vault (subject-first topic pages, hubs, wikilinks,
   Home.md, dream diary) has no equivalent anywhere: Honcho's memory is fully opaque,
   MemPalace's export is a flat uncross-linked dump, claude-mem's viewer is a feed, not a garden.
6. **Team semantics.** private/shared/standard visibility, dictator standards and processes,
   per-user token identity. claude-mem's team runtime is beta; Honcho isolates by workspace only.
7. **Server-grade architecture.** Async 202 writes, three isolated ARQ queues, REST + MCP for
   any client — not just coding CLIs with hook systems.

## 2. Gap analysis — what they do better

| Gap | Who proved it | Severity |
|---|---|---|
| No provenance DAG on derived memories (insight → premise links) | Honcho `source_ids` + `get_reasoning_chain` | High |
| No epistemic status on memories (explicit vs deduced vs induced) | Honcho `level` | High |
| Dedup discards reinforcement signal instead of counting it | Honcho `times_derived` | High |
| Dreaming can consolidate a pool mid-write-burst (no settling guard) | Honcho threshold + cooldown + idle-cancel | Medium |
| Recall returns full content per hit — no index-first retrieval economics | claude-mem progressive disclosure, MemPalace closets | High |
| No timeline retrieval dimension ("what happened around X?") | claude-mem `timeline` | High |
| Vault lacks a ranked entry-point narrative and predictable page skeleton | MemPalace L0/L1 + halls | High (user-priority) |
| No pinned, always-injected identity artifact per user/project | Honcho peer cards, MemPalace identity.txt | Medium |
| No connection-level salience math (strengthen on co-recall, dim with disuse) | MemPalace `dynamics.py` | Medium |
| No File Read Gate / live observer compression in the plugin | claude-mem | Medium |
| No structured Stop summaries + "Previously…" continuity | claude-mem | Medium |
| No surprisal-guided consolidation targeting | Honcho | Low |
| No SSE live feed / token-economics telemetry | claude-mem | Low |
| No public benchmark harness (LongMemEval / LoCoMo) | all three publish numbers | Low |

## 3. Design principles adopted from the research

- **"Dim, don't delete."** (MemPalace) Salience decays to a floor; content survives. Dreaming
  PRUNE already tombstones rather than destroys — extend the same semantics to the vault
  (a collapsed *Faded* section, never a missing page).
- **"The map, not the path."** (claude-mem) Inject an index that shows *what exists and what it
  costs to read*; let the agent choose. Never smart-prefetch on the system's guess.
- **"Consolidate settled data."** (Honcho) Dream when volume + cooldown + idleness align, not
  merely when the clock ticks.
- **"Operational clarity beats clever retrieval."** (MemPalace's own retraction) Fixed,
  predictable structure — same page skeleton everywhere — is worth more than novel search.
- **Provenance is trust.** (Honcho) A derived memory that can't show its premises is a liability.

---

## 4. Phased roadmap

Each item = one feature branch off `dev`, own PR, own tests (E2E-first where possible using an
isolated compose stack — never against a developer's live data). Ordered so that earlier items
unlock later ones.

### Phase A — Dreaming to parity-plus (the reasoning core)

**A1. Provenance + epistemics on derived memories.**
Add `derived_from: [memory_id]` (the premise list) and `epistemic_level:
explicit | deductive | inductive | reflection` to the memory envelope. Dreaming MERGE stamps
the merged sources; REM insights already carry `related_memory_ids` — promote that to the
first-class field. New MCP tool `get_reasoning_chain(memory_id)` walks the DAG (vector metadata
+ Graphiti edges). Vault topic pages render a "Derived from" footnote per insight.
*Tests:* unit (envelope, DAG walk), E2E (sweep produces insights whose chains resolve).

**A2. Reinforcement-aware dedup (`times_derived`).**
When extraction or dreaming hits a near-duplicate, increment a `times_derived` counter on the
survivor instead of silently dropping. Feed it into the existing promotion score (richness →
reinforcement term) and recall ranking.
*Tests:* unit (counter merge), E2E (re-storing the same fact N times ranks it above a one-off).

**A3 (slimmed). Settling guard + manual trigger.**
Full activity-aware scheduling (event-driven threshold triggers, idle timers with
cancel-on-write) is deferred: the time + volume gates already exist, the nightly cron is
anchored to the operator's quiet hours, and queue isolation covers resource contention — the
big machinery only pays off for always-on multi-agent deployments (revisit alongside E6).
What ships now is the correctness kernel:
- **Settling guard** (~10 lines in the sweep): a pool written to within the last N minutes is
  deferred to the next pass with status `"settling"` — never consolidate mid-conversation.
- **`schedule_dream(pool)` MCP tool** exposing the existing admin trigger to agents.
*Tests:* unit (settling window with fake clock), E2E (fresh write defers the pool; quiet pool dreams).

**A4. Salience dynamics (Hebbian / Ebbinghaus / spacing).**
Port MemPalace's ~100-line `dynamics.py` math onto the existing recall traces: strength +δ per
co-recall (capped), stability grows only on reinforcements ≥1h apart, exponential decay,
**floor 0.05 — never zero**. Replaces the current single half-life knob as the input to
promotion/prune scoring.

*Recall-safety guardrails (contractual — the implementation MUST honor all three):*
1. **Salience never gates retrieval.** It shapes consolidation and vault presentation only.
   Any influence on search ranking is a bounded, logarithmic tie-breaker
   (`score * (1 + k*log1p(strength_signal))`, `k` config-exposed, conservative default,
   zero disables it) — relevance always dominates; a faded-but-relevant memory must beat a
   hot-but-mediocre one.
2. **Low strength only nominates for PRUNE — never executes it.** The existing gates stay:
   the consolidation LLM must independently concur, destructive actions below the confidence
   threshold go to shadow-report, and pruning tombstones (reversible), never deletes.
3. **No rich-get-richer runaway.** Strength increments saturate (hard cap), and the
   query-diversity term keeps single-query hammering from compounding. No
   retrieval-induced-inhibition: memories dim only from their own disuse, never because a
   sibling was recalled.
*Tests:* pure-function unit suite (bounds, spacing effect, floor), guardrail tests (faded
memory still returned & ranked above weaker matches; k=0 is a no-op), plus a sweep E2E
asserting a frequently-co-recalled pair resists pruning.

**A5. Surprisal-targeted REM (lite).**
Score staged memories for novelty (distance to pool centroid — cheap version of Honcho's cover
trees) and bias the reflection prompt's substrate toward the top anomalies instead of uniform
sampling.
*Tests:* unit (scoring), E2E (planted anomaly appears in reflection substrate).

### Phase B — The humane vault v2 (the user-facing layer)

**B1. Home.md = L0 identity + L1 Essential Story.**
Top of Home.md: a tiny identity block (who the operator is, active projects — sourced from the
identity card, see B4), then a budget-bounded (~3,200 chars) *Essential Story*: top ~15 memories
by salience (A4), grouped by topic, one line each with a wikilink to the topic page. The MOC
table below gains counts: `| Hub | Pages | Memories | Last dreamt |`.

**B2. Fixed page skeleton (halls) + index-card header.**
Every topic page opens with a compact pointer table (`what | entities | → source`), then fixed
sections in fixed order: **Decisions & Facts / Events / Discoveries / Preferences / Advice**
(the 13 categories map onto these five). Same skeleton on every page — predictability is the
feature.

**B3. Bridges (cross-hub tunnels) + Faded section.**
When a subject appears under multiple projects (or a person and a project), both pages get a
**Bridges** section with labeled wikilinks — sourced from Graphiti shared-entity edges. Memories
whose salience fell below the prune threshold but survived (dim-don't-delete) render in a
collapsed *Faded* section instead of disappearing.

**B4. Identity card (peer-card equivalent).**
A grammar-constrained artifact per user and per project — max 40 lines of
`IDENTITY: / ATTRIBUTE: / RELATIONSHIP: / INSTRUCTION:` — maintained by the dreaming sweep
(deduction-style pass), stored as a pinned memory + rendered at `Me/Card.md` and
`Projects/<pid>/Card.md`, and exposed via a `get_card` MCP tool for guaranteed session-start
injection. Never searchable noise; always-available grounding.

*Phase B tests:* librarian unit tests per renderer; E2E sweep against a seeded pool asserting the
full layout (Home budget respected, skeleton order, bridge reciprocity, faded collapse).

### Phase C — Retrieval economics (the agent-facing layer)

**C1. Index-first recall + batch get.**
`recall_memories(index_only=true)` returns `id | title | category glyph | age | ~token cost`
rows (50–100 tokens/hit); new `get_memories(ids=[...])` batch tool returns full payloads.
Distill a ~10-word `title` at write time; store a `token_estimate`. Plugin skill text teaches the
3-layer workflow: index → filter → batch-get.

**C2. `timeline` MCP tool.**
Anchor (memory id or query) ± depth, chronologically interleaving memories, session summaries,
and dream insights. Powers "what was happening around X?", standups, weekly digests — all of
which become cheap prompt-work skills afterwards.

**C3. Recall reasoning-tier knob + dialectic disciplines.**
`reasoning_level: minimal | low | medium | high` on the ask/answer path, jointly selecting
model, tool budget, and iteration cap. Adopt Honcho's battle-tested prompt disciplines in the
answering agent: grep-first for enumeration questions, forced update-language searches so newer
facts supersede, surface-both-and-ask on contradictions, strict abstention.

**C4. `checkpoint` batch save + queue visibility.**
One MCP call that dedups + stores a list of memories + writes a session note (single tool card
in the host UI). Plus a workspace-level `queue_status` aggregate and a `queue.empty` webhook so
ingest-then-query flows stop polling per task.

### Phase D — Plugin capture parity (the Claude Code layer)

**D1. Progressive-disclosure session-start injection** (uses C1/C2): day-grouped index table
with token costs and a savings header, replacing full-content injection.
**D2. Structured Stop summaries** (`request / investigated / learned / completed / next_steps`)
+ "Previously…" injection at next session start.
**D3. File Read Gate**: PreToolUse(Read) — when memories reference the file, substitute a ranked
per-file timeline with an escalation menu (<1,500-byte bypass, user-overridable).
**D4. Capture hygiene**: `<private>` tag redaction, excluded-project globs, and the never-block
hook contract (transport failure → exit 0; client bug → exit 2; fail-loud after N consecutive
failures).

### Phase E — Platform & proof

**E1. SSE live stream** (`/v1/stream`) + minimal single-page feed — the "it's alive" demo.
**E2. Token-economics telemetry**: `discovery_tokens` vs recall tokens; savings surfaced in the
injected header and `/status`.
**E3. Session summarizer slots**: recursive short/long summaries per conversation with a
token-budgeted `get_context` assembler.
**E4. Custom extraction instructions** per project/user (bounded token budget) — the lightweight
80% of knowledge adapters that users will ask for first.
**E5. Benchmark harness**: LongMemEval + LoCoMo runners against a compose stack, results
committed. All three competitors publish numbers; we should too — with the lifecycle features
turned on, since that is the differentiator the benchmarks under-measure.
**E6. (v2, design-first) Observer/observed perspective scoping** — directional memories keyed by
(observer, observed). Unlocks multi-agent products; needs a spec pass before code.

### Phase F — Code-graph knowledge adapter (Graphify)

[Graphify](https://github.com/safishamsi/graphify) (MIT, ~77k ★) turns a folder of code, docs,
schemas, and media into a queryable knowledge graph — tree-sitter AST parsing for 36 languages
(offline, no LLM for code), community detection, an MCP server (`query_graph`, `get_node`,
`get_neighbors`, `shortest_path`), and exports including `graph.json`, Cypher for Neo4j, and an
Obsidian vault. Every inferred edge carries a confidence tag: `EXTRACTED | INFERRED | AMBIGUOUS`.

The maintainer decision: **coding-domain knowledge defers to Graphify for code *structure*;
Neuralscape stores the knowledge *about* the code.** Concretely, two items:

**F1. `graphify` ingest extractor + `code_graph` knowledge adapter.**
A new adapter under `adapters/code_graph/` (the `trading_strategy` seam), plus an extractor in
`ingest/extractors.py` that accepts Graphify output files (`graph.json`, `GRAPH_REPORT.md`):
- **Do NOT mirror the raw code graph into Graphiti** — it is huge, churns with every commit,
  and Graphify already serves it better over MCP. Ingest the *stable semantic layer* only:
  LLM-labeled communities (module purposes), god nodes, surprising cross-module connections,
  extracted rationale/comment nodes (`# NOTE:` / `# HACK:` become memories), and the
  GRAPH_REPORT insights — each stamped with a `source_ref` whose retrieval handle points at
  the Graphify MCP server / `graph.json` path so agents can re-fetch live structure.
- **Confidence-tag mapping onto A1's epistemic levels**: `EXTRACTED → explicit`,
  `INFERRED → deductive` (with reduced confidence), `AMBIGUOUS → stored only above a
  configurable floor, flagged for the dreaming sweep's contradiction pass`.
- Adapter taxonomy/ontology: code-native categories (module, boundary, invariant, rationale,
  hotspot) registered via `register_categories`; Graphiti ontology gets `Module`, `Symbol`
  (sparingly — hubs only), `depends_on`-style relations for the ingested summary layer.

**F2. Coding-domain deferral policy.**
When Graphify is present for a project (`graphify-out/` or its MCP server configured), NS
components treat it as the source of truth for code structure: the plugin's skills/hooks answer
"how does X connect to Y" via Graphify's MCP tools instead of storing structural facts as
memories; extraction demotes purely-structural observations (they'd rot) in favor of decisions,
gotchas, and rationale; the dreaming librarian's project hub links out to Graphify's Obsidian/
HTML exports rather than duplicating them. `sync`-style liveness: a dreaming sweep step flags
memories whose `source_ref` points at Graphify nodes that no longer exist in the current
`graph.json` (the code moved on) as INVALIDATE candidates.

*Tests:* F1 unit (extractor parses fixture graph.json/report; epistemic mapping; node-liveness
diff), E2E (ingest a small fixture repo's Graphify output → memories + source_refs resolve);
F2 is mostly plugin/prompt policy — assert extraction skip-rules and hub link rendering.

---

## 5. Build order for the ship loop

Priority interleaves "dreaming to parity" (Phase A) with the user-priority vault work (Phase B).
A3 was slimmed to a settling guard + manual trigger and folded into the salience PR, which
pulls the vault work two slots earlier:

| # | Item | Branch |
|---|---|---|
| 1 | A1 provenance + epistemic level | `feat/memory-provenance` |
| 2 | A2 reinforcement dedup | `feat/reinforcement-dedup` |
| 3 | B1+B2 Home story + page skeleton | `feat/vault-essential-story` |
| 4 | B3+B4 bridges, faded, identity card | `feat/vault-bridges-card` |
| 5 | A4 salience dynamics + A3-lite settling guard | `feat/salience-dynamics` |
| 6 | F1 graphify extractor + code_graph adapter | `feat/code-graph-adapter` |
| 7 | C1+C2 index recall + timeline | `feat/retrieval-economics` |
| 8 | C3+C4 tiers + checkpoint | `feat/recall-tiers` |
| 9 | D1+D2 plugin injection + summaries + F2 deferral policy | `feat/plugin-disclosure` |
| 10 | D3+D4 read gate + hygiene | `feat/plugin-read-gate` |
| 11 | A5 + E1+E2 surprisal, stream, economics | `feat/observability` |
| 12 | E3–E5 context assembler, custom instructions, benchmarks | `feat/platform-proof` |

Loop per item: worktree off `dev` → build → unit + isolated-compose E2E (one Neuralscape
deployment at a time) → PR to `dev` → CodeRabbit/Copilot review → address → merge → next.
