# Spec: Dreaming Mode for Neuralscape

**Audience:** a Claude Code session working in the `neuralscape` repo.
**Status:** design + implementation plan. Nothing here is built yet.
**Author:** distilled from a study of three shipped "dreaming" systems + the memory-consolidation
literature, mapped onto Neuralscape's existing seams.

> Self-contained. Cites the exact files/seams in `neuralscape-service` you'll touch, the data model,
> and a phased implementation with acceptance criteria. Verify every file:line against the current
> tree before editing — line numbers drift.

---

## 0. The ask

Give Neuralscape a **"dreaming mode"**: a background process that periodically reflects on what the
system has learned and *improves the memory store itself* — merging duplicates, resolving
contradictions, pruning stale/noisy entries, and **surfacing new higher-order insights** — the way a
brain consolidates during sleep.

We already have a first cut of this idea (`wiki_synthesizer`), but it under-delivers (§2). Dreaming
mode is its principled successor.

**Design north star:** dreaming must be **(a) reflective** (generate *new* insight, not just
re-summarize), **(b) consolidating** (merge/resolve/prune, not merely accrete), **(c)
non-destructive** (reversible via the graph's bi-temporal invalidation — never blind hard-delete),
and **(d) recallable** (dream output feeds back into the store as first-class memories, retrievable
through the normal recall path).

### 0.1 Decisions assumed in this spec (confirm / redirect)

These three forks were flagged for the product owner; this spec is written to the recommended
default of each. Changing any of them changes the shape below.

1. **Adoption posture = hybrid.** Auto-apply *reversible* ops (dedup-merge, insight-add) directly;
   route *destructive* ops (contradiction-invalidation, prune) through a confidence gate + a
   report-only shadow trial before applying. (Alternatives: fully-autonomous-non-destructive;
   propose-then-review only; separate swap-in store.)
2. **`wiki_synthesizer` = superseded.** Dreaming is the successor; the wiki renderer is reused as
   dreaming's REM/diary output; the standalone `synthesize_*` cron is retired.
3. **v1 scope = Consolidation + Reflection + Recall-reinforcement ranking.** Hierarchical topic
   synthesis (RAPTOR/communities) is deferred to v2 — it is exactly what `wiki_synthesizer` already
   choked on for performance.

---

## 1. What "dreaming" means across the field (study notes)

Three shipped systems + the research, and what each contributes to our design.

### 1.1 Anthropic Managed-Agents "Dreams"
An **async job** takes an existing **memory store** + **1–100 past session transcripts** and produces
a **new, separate** store with duplicates merged, stale/contradicted entries replaced with the latest
value, and new insights surfaced. **The input is never mutated** — you review the output and adopt or
discard. Steerable via freeform `instructions`; poll-based lifecycle (`pending → running →
completed/failed/canceled`); billed at token rates.
**Take-aways for us:** the *non-destructive + reviewable* contract; *transcripts are an input, not
just the store*; *steerable synthesis*; *async job with a lifecycle*.

### 1.2 The vendored TS reference implementation (`mem0/openclaw/`)
A working background consolidation plugin already in our tree. Two load-bearing pieces:

- **`mem0/openclaw/dream-gate.ts`** — a **gate economy + lock**. Cheap gates first
  (`minHours` since last run, `minSessions` since last run — pure local file reads), then the
  expensive gate (`minMemories`). A stale-reclaimable lock (`LOCK_STALE_MS = 1h`) prevents concurrent
  runs. State (`lastConsolidatedAt`, `sessionsSince`, `lastSessionId`) persists across restarts.
  Defaults: `minHours=24, minSessions=5, minMemories=20`.
- **`mem0/openclaw/skills/memory-dream/SKILL.md`** — the **consolidation protocol**: four phases
  (Orient → Gather targets → Consolidate → Report) with three actions — **DELETE**
  (credentials/secrets/noise/raw-tool-output/TTL-expired), **MERGE** (same fact in different words →
  keep most complete, fold in details, drop redundant), **REWRITE** (vague / first-person / missing
  temporal anchor / wrong category / >50 words). Quality targets: zero secrets, zero dupes, third
  person, temporal anchors, one atomic fact per memory, 15–50 words.

Its docs describe a **three-phase sleep cycle** and a **reinforcement-ranking** economy:

| Phase | Purpose | Durable write? |
|---|---|---|
| **Light** | Ingest recent signals (daily memory, recall traces, redacted transcripts); dedup; stage candidates; record reinforcement | No |
| **Deep** | Rank candidates, apply threshold gates (`minScore`, `minRecallCount`, `minUniqueQueries`), promote | **Yes** |
| **REM** | Build theme/reflection summaries; record reinforcement for deep ranking | No |

Promotion score = weighted signals: **relevance 0.30, frequency 0.24, query-diversity 0.15,
recency 0.15, consolidation 0.10, conceptual-richness 0.06**, plus a recency-decayed boost from
light/REM hits. A **shadow trial** layers a report-only review verdict (helpful→boost/promote,
neutral→defer, harmful→reject) *before* any durable write. A **dream diary** (`DREAMS.md`) is a
human-readable narrative, explicitly *not* a promotion source.
**Take-aways for us:** the light→deep→REM structure; the *cheap-then-expensive gate economy + lock*;
*recall traces as the reinforcement signal*; *report-only shadow trial* as the safety layer; *diary
≠ store*.

### 1.3 Letta sleep-time agents (the productized version of sleep-time compute)
Letta (≥0.7.0) ships the paper's idea as a **two-agent architecture**: creating a sleep-time-enabled
agent spins up a **primary agent** (converses, calls tools, *has no memory-editing tools*) and a
**sleep-time agent** (holds ALL the memory-editing tools; asynchronously rewrites the primary's
in-context memory blocks from "raw context" into "learned context"). Updates are **anytime** — the
primary reads memory whenever, never waiting on the dreamer. A **frequency** knob trades tokens for
revision opportunities. Recommended config: a *stronger, slower model* for the sleep-time agent,
since it isn't latency-constrained.
**Take-aways for us:** the **authority split** — the hot path never consolidates; only the dreamer
holds destructive/rewrite authority (our API read/write path vs the dreaming worker maps 1:1);
anytime semantics (recall never blocks on a dream); a dedicated, stronger `DREAMING_MODEL`.

### 1.4 OpenAI ChatGPT memory dreaming (V0 April 2025 → V3 June 2026)
OpenAI's own name for exactly this pattern ("Dreaming: Better memory for a more helpful ChatGPT",
June 2026). Their timeline: **saved memories** (2024, explicit "remember X" cues, went stale) →
**Dreaming V0** (2025, a background process that curates memory by referencing chat history — no
explicit cues needed) → **Dreaming V3** (2026, a full memory architecture *built on* dreaming,
"optimized for freshness, continuity and relevance"). Three evaluation objectives: **carry forward
context**, **follow preferences/constraints** (explicit instructions, personal constraints, *and
implicit* preferences), and **stay current over time**. The distinctive mechanism in the third:
**time-passage rewriting** — dreaming revises "user is planning a Singapore trip in July" into
"user went to Singapore in July 2026" *when the trip ends*, without any new conversation
triggering it. Memories synthesized by dreaming are **user-reviewable via a memory summary page**
(view highlights, correct facts, steer what comes up). A recent revision cut serving compute ~5×,
which is what made dreaming viable at free-tier scale.
**Take-aways for us:** (a) **temporal reframing is a first-class dream action** — future-dated /
event-anchored memories should be re-perspectived once their date passes, not merely decayed or
contradiction-invalidated (and Graphiti's `valid_at` gives us the anchor for free); (b) the diary
doubles as the **user-facing "what I know about you" summary** — reviewable and steerable, not just
a log; (c) their three eval objectives are a ready-made taxonomy for our golden-query set (§4.3);
(d) compute-efficiency is a launch gate, not a nice-to-have — hence our gate economy.

### 1.5 The research
First wave (core mechanics):
- **Generative Agents** (arXiv 2304.03442) — the canonical precedent: a memory stream is periodically
  synthesized into higher-level **reflections** (inferences — "the recurring pattern is X"), stored
  back and retrieved. *This is the reflection step our current system lacks entirely.*
- **Sleep-time compute** (Letta, arXiv 2504.13171) — "think" offline about context *before* queries
  arrive; pre-compute; amortize across future queries → ~5× cheaper test-time + higher accuracy.
  *The economic argument: move work off the hot recall path.*
- **mem0** (arXiv 2504.19413, vendored) — explicit **ADD / UPDATE / DELETE** consolidation decisions
  over memories.
- **RAPTOR** (arXiv 2401.18059) — recursive embed→cluster→summarize into a tree of abstraction
  levels (our deferred v2 hierarchy).
- **MemGPT** (arXiv 2310.08560) — tiered memory, self-editing.

Second wave (gap-fill pass — forgetting, evolution, failure-learning, retrieval):
- **MemoryBank** (arXiv 2305.10250) — memory updating driven by the **Ebbinghaus forgetting curve**:
  each memory carries a retention strength that *decays over time* and is *reinforced on recall*.
  *Gives PRUNE a principled, graded basis (strength below threshold) instead of a binary
  stale/not-stale judgment — and our recall traces (§4) are exactly the reinforcement signal it
  needs.*
- **A-MEM** (arXiv 2502.12110) — Zettelkasten-style agentic memory: each new memory is dynamically
  **linked** into the network and can trigger **evolution of its neighbors** (existing notes updated
  in light of the new one). *A new fact arriving is a consolidation trigger, not just a row insert —
  adopted as DEEP's link-enrichment sub-action (§3.3).*
- **Reflexion** (arXiv 2303.11366) — agents verbally reflect on **failure signals** and store the
  lessons in an episodic buffer that improves later attempts. *REM should mine the daily logs for
  error/retry/correction patterns and emit `procedure`-category "lesson" memories, not only
  preference/pattern insights.*
- **HippoRAG** (arXiv 2405.14831) — hippocampal-indexing-inspired retrieval: KG + Personalized
  PageRank to mimic neocortex/hippocampus roles. *Parked for v2 alongside RAPTOR — a retrieval-side
  upgrade, not a dreaming-cycle requirement.*
- **Zep/Graphiti** (arXiv 2501.13956) — our own graph substrate's paper; the bi-temporal model §2.1
  leans on.

---

## 2. What already exists in Neuralscape (grounded) + why `wiki_synthesizer` under-delivers

### 2.1 The seams we build on
- **Extension system** — `extensions/base.py` (`NeuralscapeExtension`: manifest/startup/shutdown/
  on_event/get_routes), discovered by `ExtensionRegistry`, driven by cron on the slow
  `GraphWorkerSettings` queue or by events (`extensions/events.py`). Dreaming is a new extension here.
- **Graphiti bi-temporal graph** (Neo4j) — contradiction-as-invalidation (`valid_at`/`invalid_at`,
  `created_at`/`expired_at`), never deletion; entity dedup. **This is our non-destructive engine —
  the thing the reference implementations approximate with `memory_delete` we get for free.**
- **mem0 / Qdrant** — vector store + `ADD/UPDATE/DELETE` primitives; `MemoryService` wraps both.
- **`conversation_compiler`** (`extensions/conversation_compiler/`) — flushes conversation turns into
  `vault/Daily/YYYY-MM-DD.md` raw logs and has an LLM-powered **contradiction detector** in its
  `lint` step. *These daily logs are our "redacted session transcripts" input.*
- **`wiki_synthesizer`** (`extensions/wiki_synthesizer/`) — the current synthesis extension (see 2.2).
- **Memory-model v2 fields** (`schemas.py`) — `category`, `observation_type`, `confidence`,
  `concepts`, `source_type`, plus the fixed envelope (`source_ref`, `memory_kind`, scope,
  visibility). Dreaming reads and writes these.

### 2.2 Why `wiki_synthesizer` is a filing cabinet, not a dreamer
Read `extensions/wiki_synthesizer/synthesizer.py` + `prompts.py`. Diagnosis:

1. **Rigid grouping.** Buckets are `(group_id × category)`. It *explicitly abandoned* real
   topic/entity discovery (Graphiti community detection) for being too slow (`synthesizer.py:14-19`).
   Unrelated facts pile onto one category page; a decision + its convention + its gotcha about the
   *same subsystem* scatter across three pages.
2. **Additive-only, never critical.** `INCREMENTAL_MERGE_PROMPT` says "integrate every distinct fact,
   don't duplicate." It **never resolves contradictions, marks stale, or prunes**. It only accretes —
   the opposite of consolidation.
3. **No reflection.** It re-renders facts into prose; it never *infers* a higher-order insight.
4. **Dead-end output.** It writes Obsidian markdown + a `wiki_path` back-ref. The synthesized
   knowledge is **never itself a recallable memory** — `recall_memories` can't return it.
5. **Ignores our best tool.** It never drives Graphiti invalidation.
6. **Ignores episodic material.** It scans only Qdrant rows, never the `conversation_compiler` daily
   logs.

Dreaming keeps what worked (idempotent skip on unchanged source-id sets; the vault renderer; the
graph back-patch) and fixes 1–6.

### 2.3 Overlap with the existing maintenance crons (must be resolved, not ignored)
Two crons on `GraphWorkerSettings` already do fragments of dreaming's DEEP phase — naively adding
dreaming would triple-process the same rows and race them:

- **`dedup_all_memories`** (`worker.py:768` → `memory_service.dedup_memories`, ~`:3483`) — two-phase
  dedup: exact (hash, keep newest) + semantic (cosine above `dedup_similarity_threshold`,
  **hard-delete the older one** via `_delete_qdrant_memory_with_graph_cleanup`). The semantic phase
  is **information-lossy**: when two near-duplicates each carry unique details, the older one is
  destroyed wholesale. It also keys exact dedup on `(hash, visibility)` — a nuance dreaming must
  preserve (a `standard` memory is never collapsed into an identically-worded `shared`/`private`
  one).
- **`expire_old_memories_cron`** (`worker.py:604`) — mechanical TTL purge of `expires_at`-past rows.

**Resolution:**
- Dreaming's **MERGE supersedes the semantic dedup phase**. Where the cron deletes the older
  near-duplicate (losing its unique details), MERGE folds details into the survivor and
  *invalidates* the rest — strictly better. When `DREAMING_ENABLED=true` for a scope, the dedup
  cron's semantic phase is skipped for that scope (flag-gated); the cheap exact-hash phase stays
  (it's lossless and O(scroll)).
- **`expire_old_memories_cron` stays** — mechanical TTL purge is cheap and deterministic; dreaming's
  PRUNE handles *judgment-based* staleness (decayed strength, contradicted, noise), not explicit
  `expires_at`.
- **Scheduling**: the graph worker already staggers dedup (:00), expiry (hour=3), wiki-synth (:45 +
  offset), and the strategy-playbook synthesizer. The dreaming sweep joins this schedule (nightly
  03:xx default) and its per-scope Redis lock (§3.1) must be *shared with* (or checked by) the dedup
  cron so the two never consolidate the same scope concurrently.

---

## 3. Architecture — the dream cycle

One new extension, `extensions/dreaming/`, running a **light → deep → REM** sweep on the slow queue,
gated by a cheap-then-expensive gate economy, writing through Graphiti invalidation + Qdrant, and
emitting a human-readable diary.

```
                          ┌─────────────────────── dreaming cron (slow queue) ───────────────────────┐
 recall traces ─┐         │  gate economy (§3.1)                                                       │
 daily logs ────┼──▶ LIGHT (stage) ──▶ DEEP (consolidate + promote) ──▶ REM (reflect) ──▶ diary       │
 memory store ──┘         │      │                    │  ▲ reinforcement           │                   │
                          │   candidates          Graphiti invalidation +      new insight memories    │
                          │   (.dreams state)     Qdrant merge/update          (source_type=dream)     │
                          └────────────────────────────────────────────────────────────────────────┘
```

### 3.1 The gate economy (port `dream-gate.ts` to Python)
`extensions/dreaming/gate.py` — a Python port of the vendored gate logic, keyed per **scope**
(per `group_id`, matching `wiki_synthesizer`'s shared-group walk). State in Redis (not a local file —
we're multi-process; the API and workers are separate) under `dreaming:gate:<group_id>`:
`last_dreamt_at`, `sessions_since`, `writes_since`.

- **Cheap gates** (Redis reads only): `min_hours` since last dream, `min_sessions` since last dream.
- **Expensive gate**: `min_new_memories` in this scope since `last_dreamt_at` (a Qdrant
  count/scroll). Only checked if cheap gates pass.
- **Lock**: a Redis `SET NX PX` lock (`dreaming:lock:<group_id>`, stale-reclaim after 1h) — the
  distributed analog of the file lock. Prevents overlapping sweeps and collision with the dedup cron.

Defaults (config §6): `min_hours=24`, `min_sessions=5`, `min_new_memories=20`.

### 3.2 LIGHT — stage (no durable write)
Gather this sweep's raw material for one scope and stage candidates in `.dreams` state (Redis):
1. **New/changed memories** since `last_dreamt_at` (Qdrant scroll on `created_at`/`updated_at`).
2. **Recall traces** for the scope (§4) — which memories were retrieved, by which queries.
3. **Daily logs** from `conversation_compiler` for the window (the transcript analog) — mined for
   patterns the structured memories may have missed.

Dedup within the batch (content-hash + embedding near-dup), and record **reinforcement signals**
(recall frequency, distinct-query count) onto each candidate. Light never writes to the store.

### 3.3 DEEP — consolidate + promote (the only durable write)
For the staged candidates, decide an action per memory (mirrors the `memory-dream` SKILL, but
Graphiti-aware). Each decision carries a **confidence**; the **adoption posture** (§3.6) decides
whether it applies now or goes to shadow review.

| Action | Reversible? | Mechanism |
|---|---|---|
| **MERGE** (duplicates) | yes | `memory_update` the most-complete survivor to fold in details; **invalidate** (not delete) the redundant ones in Graphiti; tombstone the Qdrant rows (`superseded_by` in metadata). |
| **INVALIDATE** (contradiction) | yes | Set `invalid_at` on the superseded Graphiti fact; keep the row, mark `superseded_by`. Never hard-delete. |
| **PRUNE** (stale/TTL/noise/secret) | mostly | Secrets → hard-delete (safety exception). TTL/stale/noise → invalidate + tombstone. |
| **REWRITE** (clarity/voice/anchor/category) | yes | `memory_update` in place (atomic, preserves history). |
| **PROMOTE** (short-term → durable) | yes | Only if the promotion score clears the threshold gates (§4). |
| **TEMPORAL-REFRAME** (date passed) | yes | From OpenAI's dreaming: a future-dated / event-anchored memory whose date has passed is *rewritten in past perspective* ("planning trip in July" → "went on the trip in July 2026") via `memory_update`, with the original preserved through Graphiti's bi-temporal record. Candidates found cheaply: memories whose extracted dates / `valid_at` are now behind the sweep time. Distinct from PRUNE — the fact is still valuable, its *tense* is stale. |

Additional DEEP sub-action (from A-MEM, cheap because Graphiti already does entity resolution):
**LINK-ENRICH** — when a staged new memory strongly relates to existing neighbors (shared entities /
high similarity), record `related_memory_ids` links and, where the new fact *changes the reading* of
an old one, queue the old one for REWRITE. A new fact is a consolidation trigger, not just an
insert.

**Promotion gates** (from the reference): `min_score`, `min_recall_count`, `min_unique_queries`.
Score = the weighted signal blend (§4).

### 3.4 REM — reflect (no durable write to source; writes NEW insight memories)
The step `wiki_synthesizer` never had. For each scope, cluster the (post-consolidation) memories by
topic/entity (cheap: reuse `category` + Graphiti entity grouping; no expensive community detection in
v1) and run a **reflection prompt** that asks for *inferences*, not summaries: recurring patterns,
implied preferences, cross-memory conclusions. Two reflection lenses (both in the prompt):
- **Pattern lens** (Generative Agents): recurring behaviors, implied preferences, cross-memory
  conclusions.
- **Failure lens** (Reflexion): mine the daily logs for error → retry → correction sequences and
  emit the *lesson* as a `procedure`/`gotcha` memory ("X fails when Y; do Z instead") — these are
  the highest-value insights an agent memory can hold.

Each reflection becomes:
- a **new first-class memory** — `source_type="dream"`, `observation_type="reflection"`, a
  `DERIVED_FROM` link (via `related_memory_ids` / graph edge) to every source memory, and a
  `confidence` — so it is **recallable through the normal path** and auditable.
- a section in the **dream diary** (reuse `wiki_renderer`) — human-readable narrative, *not* itself a
  promotion source (excluded from the next light phase to prevent feedback loops). Per OpenAI's
  memory-summary-page pattern (§1.4), the diary doubles as the user-facing **"what the system
  knows"** summary per pool — the review-and-steer surface, not just an audit log.

REM also records reinforcement (a reflection referencing a memory reinforces that memory's deep score
next sweep).

### 3.5 Reuse from `wiki_synthesizer` (supersede, don't rewrite)
- **Idempotent skip** — carry over the `source_memory_ids` unchanged-set short-circuit
  (`synthesizer.py:266-287`) so an unchanged scope is a no-op.
- **Renderer** — `wiki_renderer.py` becomes the diary/reflection renderer.
- **Graph back-patch** — `graph_patcher.py` (`patch_wiki_path_by_memory_ids`) stamps
  `dream_path`/`reflection_id` back-references.
- Retire `synthesize_all` cron; keep the module until the diary renderer is extracted, then delete.

### 3.6 Adoption posture (hybrid — the assumed default)
- **Auto-apply** reversible, high-confidence ops: MERGE, REWRITE, INSIGHT-ADD, PROMOTE. Safe because
  Graphiti invalidation is reversible and insights are additive + provenance-linked.
- **Shadow-trial** (report-only) destructive/low-confidence ops: INVALIDATE (contradiction), PRUNE.
  A candidate below `auto_apply_confidence` is written to a **dream report** (`DREAMS.md` + an admin
  endpoint) with verdict/reason/evidence and **not applied** until adopted. This is the vendored
  shadow-trial + Anthropic's reviewable contract, expressed on our store.
- **Everything is reversible** except secret hard-deletes: invalidations can be un-set; a rejected
  proposal is dropped.

---

## 4. Recall-reinforcement ranking + cross-cutting mechanics

Promotion/consolidation is only "smart" if we know **what actually gets used**. Today recall is not
traced.

- **Log recall traces.** In the read path (`memory_service.search`/recall), after results are
  returned, enqueue a lightweight async trace to Redis: `(memory_id, query_hash, scope, ts)`. Keep it
  off the hot path (fire-and-forget, matching the async-write ethos). A rolling window (e.g. 30 days)
  is enough.
- **Aggregate per memory**: `recall_count`, `unique_query_count` (distinct `query_hash`),
  `last_recalled_at`.
- **Score** (weights from the reference, tune later):
  `score = 0.30·relevance + 0.24·frequency + 0.15·query_diversity + 0.15·recency +
  0.10·consolidation + 0.06·conceptual_richness`, where `relevance` = mean recall similarity,
  `frequency`/`query_diversity` from the aggregates, `recency` = decay on `last_recalled_at`,
  `consolidation` = how many sources merged into it, `conceptual_richness` = `len(concepts)` /
  content signal.
- The score powers DEEP promotion gates and orders the shadow-trial report.
- **Retention strength (Ebbinghaus, from MemoryBank).** Derive a graded
  `strength = f(base_confidence, recall reinforcement, time decay)` per memory: strength decays on
  a forgetting curve from `last_recalled_at` (or `created_at` if never recalled) and is bumped on
  every recall. PRUNE's "stale" input is then *strength below `prune_strength_threshold`* — graded
  and reinforcement-aware — rather than a binary LLM staleness judgment. Strength is computed at
  dream time from the trace aggregates (no per-recall writes needed).

> If recall-reinforcement is dropped from v1, DEEP falls back to recency + confidence + dedup-count
> only, strength degrades to pure time-decay, and promotion is coarser — but the rest of the cycle
> stands.

### 4.1 The authority split (Letta's two-agent principle, mapped to our processes)
**The hot path never consolidates.** The API/MCP read/write path only ever *adds* memories and
*logs traces*; all MERGE/INVALIDATE/PRUNE/REWRITE authority lives exclusively in the dreaming
worker. Recall is **anytime** — it never blocks on, or waits for, a dream; a sweep mid-run just
means recall sees pre-consolidation rows a little longer. Corollary: the dreamer may use a
**stronger, slower model** (`DREAMING_MODEL`) than the extraction path, since it is not
latency-constrained.

---

### 4.2 Visibility, privacy, and redaction (blind spot in v0 of this spec)

`wiki_synthesizer` ducked this by being shared-only. Dreaming cannot — most consolidation value
(dupes, contradictions, noise) lives in the **private** per-user pools.

- **Pool isolation.** A sweep runs *per pool* (`shared`, `shared--project--<pid>`,
  `user--<uid>[--project--<pid>]`), and every dream output — merged survivors, insight memories,
  diary/report artifacts — **inherits that pool's visibility and owner**. Cross-pool reads inside a
  sweep are forbidden: a private memory must never inform a shared reflection, and vice-versa a
  shared sweep never touches private rows. (Mirror of the dedup cron's `(hash, visibility)` keying,
  §2.3, and of PR #90's visibility-blind-dedup lessons.)
- **`standard` tier is read-only to dreaming.** Dictator-authored standards are authoritative; the
  dreamer may *cite* them in reflections but never merges, rewrites, invalidates, or prunes them.
- **Redaction at intake, not just at prune.** LIGHT scrubs credentials/secrets/token patterns from
  daily-log material *before* staging (the reference implementation feeds dreaming *redacted*
  transcripts). DEEP's secret hard-delete then only catches what slipped through historical writes.
- **Diary respects visibility.** Private-pool diaries render under the user's vault namespace;
  only shared-pool diaries land in the team wiki tree.

### 4.3 Evaluation — how we know dreaming helped

"Review the output" (Anthropic) needs a measurable proxy, and we already have the tooling precedent
(the perf-bench + dashboard from PR #77):

- **Golden-query set per scope.** A small fixed set of recall queries with expected-memory
  annotations. Run before/after each sweep; the `DreamRun` report records the recall-precision
  delta alongside the mechanical deltas (store size, duplicate rate, contradiction count,
  mean-memories-per-recall). Structure the queries along OpenAI's three memory objectives (§1.4):
  **carry-forward** (does recall surface the right prior context?), **preference-following** (are
  explicit *and implicit* constraints retrieved when relevant?), and **staying-current** (does a
  time-sensitive query get the post-reframe answer, not the stale tense?).
- **Regression tripwire.** If golden-query recall *drops* after a sweep, flag the run in the report
  (and, in a later iteration, auto-revert its invalidations — they're reversible by construction).
- **Dry-run first.** `dry_run=true` produces the full report with zero writes — the tuning loop for
  thresholds/weights before enabling auto-apply.

## 5. Data model additions (`schemas.py`)

Fixed envelope unchanged. Additions:
- `source_type`: add `"dream"` (reflections/consolidations authored by dreaming).
- `observation_type`: add `"reflection"`.
- Memory metadata (Qdrant `metadata` + Graphiti attrs, all optional/back-compat):
  `superseded_by: str|None`, `superseded_at: iso|None`, `dream_id: str|None`,
  `recall_count/unique_query_count/last_recalled_at`, `dream_score: float|None`.
- A **`DreamRun`** record (Redis + optional Neo4j node): `id`, `group_id`, `started/ended_at`,
  `status`, per-action counts (merged/invalidated/pruned/rewritten/promoted/reflected), `errors`,
  `report_path`. Mirrors `wiki_synthesizer`'s `LastRunSnapshot` but persisted (cross-process) and
  poll-able — the Anthropic dream-lifecycle analog.

---

## 6. Config (`extensions/dreaming/config.py`, `DREAMING_*` prefix)

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `false` | Master switch (dark until opted in, like `wiki_synthesizer`). |
| `cron` / `cron_hours` | `0 3 * * *` (nightly) | Sweep cadence (stagger vs dedup + wiki crons). |
| `min_hours` / `min_sessions` / `min_new_memories` | `24 / 5 / 20` | Gate economy. |
| `auto_apply_confidence` | `0.85` | Above → auto-apply destructive ops; below → shadow report. |
| `min_score` / `min_recall_count` / `min_unique_queries` | tune | Promotion gates. |
| `reflection_enabled` | `true` | Toggle REM. |
| `model` | inherit | LLM for consolidation/reflection. Per §4.1, prefer a *stronger* model than the extraction path — the dreamer isn't latency-constrained. |
| `prune_strength_threshold` | tune | Ebbinghaus retention-strength floor below which PRUNE considers a memory (§4). |
| `disable_semantic_dedup_cron` | `true` when enabled | Skip the lossy semantic phase of `dedup_all_memories` for dreaming-enabled scopes (§2.3). |
| `dry_run` | `false` | Do everything, write nothing (report only). |
| `scopes` | `shared* + user pools` | Which pools to sweep. Unlike wiki_synth, private per-user pools are in scope — with strict pool isolation (§4.2). |

Add `synthesize_dreams_cron` to `GraphWorkerSettings` in `worker.py` (retire the wiki cron).

---

## 7. Implementation phases (ordered, each shippable + testable)

**Phase 0 — Extension skeleton + gate economy.** `extensions/dreaming/` (manifest, config, `gate.py`
Redis port of `dream-gate.ts`, `__init__.py`), cron registered but no-op behind `enabled=false`.
**Acceptance:** cron runs, gates read/write Redis, lock acquire/release + stale-reclaim covered by
unit tests; disabled = no-op.

**Phase 1 — Recall-reinforcement traces.** Log + aggregate recall traces; expose `dream_score`.
**Acceptance:** a recall increments `recall_count`/`unique_query_count`; scoring is a pure, unit-
tested function; the trace write is off the hot path (fire-and-forget) and never blocks a read.

**Phase 2 — DEEP consolidation (Graphiti-aware).** MERGE/INVALIDATE/PRUNE/REWRITE/LINK-ENRICH/
TEMPORAL-REFRAME with the hybrid adoption posture; secret-scrub hard-delete exception; gate the semantic phase of
`dedup_all_memories` off for dreaming-enabled scopes; enforce pool isolation + `standard`-tier
read-only. **Acceptance:** a seeded scope with a known duplicate + a known contradiction + a stale
entry → duplicate merged (survivor keeps details, others `superseded_by`), contradiction gets
`invalid_at` (row retained), stale invalidated; a low-confidence contradiction lands in the report
**unapplied**; a near-duplicate pair where the *older* memory holds unique details survives with
those details folded in (the case the semantic dedup cron gets wrong); a `standard` memory is
untouched; a private-pool sweep writes nothing shared (and vice-versa); re-run with no changes =
`skipped_unchanged`.

**Phase 3 — REM reflection + diary.** Reflection prompt → new `source_type="dream"` insight memories
with `DERIVED_FROM` provenance; diary via `wiki_renderer`. **Acceptance:** a scope with a recurring
pattern across ≥3 memories yields a reflection memory that (a) is returned by `recall_memories`, (b)
links to its sources, (c) appears in the diary; diary entries are excluded from the next light phase.

**Phase 4 — Supersede `wiki_synthesizer` + admin surface + eval.** Extract the renderer, retire the
wiki cron, add `POST /v1/extensions/dreaming/run` (manual trigger, `dry_run`) + `GET /status`
(`DreamRun` poll), wire the golden-query before/after eval into the `DreamRun` report (§4.3).
**Acceptance:** wiki cron removed with no loss of vault output; manual dry-run reports planned
actions without writing; status endpoint returns the last `DreamRun` including recall-precision +
store-size deltas.

**Phase 5 (later) — Hierarchical synthesis (v2).** RAPTOR/community abstraction tree. Deferred.

---

## 8. Seam checklist (files to touch)

| Phase | File | Change |
|---|---|---|
| 0 | `extensions/dreaming/{__init__,config,manifest.json,gate}.py` (new) | extension + gate economy |
| 0 | `worker.py` | register `synthesize_dreams_cron` on `GraphWorkerSettings` |
| 1 | `memory_service.py` (read path) | fire-and-forget recall-trace enqueue |
| 1 | `extensions/dreaming/scoring.py` (new) | trace aggregation + weighted score (pure fn) |
| 2 | `extensions/dreaming/consolidate.py` (new) | MERGE/INVALIDATE/PRUNE/REWRITE/LINK-ENRICH/TEMPORAL-REFRAME via Graphiti + Qdrant |
| 2 | `schemas.py` | `source_type="dream"`, `observation_type="reflection"`, supersede/score metadata |
| 2 | `memory_service.py` | invalidate/supersede helpers (drive Graphiti `invalid_at`) |
| 2 | `worker.py` (`dedup_all_memories`) | skip semantic-dedup phase for dreaming-enabled scopes (§2.3) |
| 3 | `extensions/dreaming/reflect.py` (new) + `prompts.py` | reflection → insight memories + diary |
| 3 | reuse `extensions/wiki_synthesizer/wiki_renderer.py` | diary/reflection renderer |
| 4 | `main.py` / extension routes | `run` + `status` endpoints; retire wiki cron |
| 4 | `extensions/dreaming/eval.py` (new) | golden-query before/after recall eval (§4.3) |

---

## 9. Risks

- **Feedback loops.** Dream-authored memories re-entering LIGHT and being re-reflected. Mitigation:
  exclude `source_type="dream"` (and diary artifacts) from the light-phase intake; only source
  `source_type in {conversation, tool_extraction, explicit, imported}`.
- **Bad auto-merge.** A wrong merge is reversible (invalidation), but noisy. Mitigation: high
  `auto_apply_confidence`, embedding+content-hash agreement required for auto-merge, everything logged
  in the `DreamRun` report.
- **Cost.** Reflection is the expensive step. Mitigation: gate economy (only dream when enough
  changed), idempotent skip, nightly cadence on the slow queue, `dry_run` for tuning.
- **Cross-process state.** `wiki_synthesizer`'s status is process-local; dreaming's `DreamRun` +
  gate state live in Redis so the API and workers agree.
- **Pool fan-out.** Sweeping private per-user pools multiplies sweep count by user count. The gate
  economy is the throttle — a pool with `< min_new_memories` since its last dream costs one Qdrant
  count and nothing else; in practice only actively-written pools dream on any given night.
- **Graphiti edge-attribute gap** (known upstream issue) — keep supersede markers on **entity/row
  metadata**, not edge attributes, until verified on the pinned subtree.
- **Codename hygiene.** Do not introduce the vendored plugin's brand name into tracked files
  (CLAUDE.md forbidden-codename rule); reference it only by path (`mem0/openclaw/…`) where a seam
  citation requires it.

---

## Appendix A — Case-study source map
- **Anthropic Dreams** — `platform.claude.com/docs/en/managed-agents/dreams` (async job; store +
  transcripts → new store; non-destructive/reviewable; steerable; lifecycle).
- **Vendored reference** — `mem0/openclaw/dream-gate.ts` (gate economy + lock),
  `mem0/openclaw/skills/memory-dream/SKILL.md` (DELETE/MERGE/REWRITE protocol + quality targets);
  its docs (light/deep/REM phases, weighted reinforcement ranking, shadow trial, dream diary).
- **Letta sleep-time agents** — `letta.com/blog/sleep-time-compute` (two-agent authority split;
  anytime memory; stronger-model-for-the-dreamer; frequency knob; released in Letta 0.7.0).
- **Generative Agents** arXiv 2304.03442 (reflection); **Sleep-time compute** arXiv 2504.13171
  (offline amortization); **mem0** arXiv 2504.19413 (ADD/UPDATE/DELETE); **RAPTOR** arXiv 2401.18059
  (hierarchy, v2); **MemGPT** arXiv 2310.08560 (tiers); **MemoryBank** arXiv 2305.10250 (Ebbinghaus
  retention strength → graded PRUNE); **A-MEM** arXiv 2502.12110 (Zettelkasten link-enrichment /
  memory evolution); **Reflexion** arXiv 2303.11366 (failure-lesson reflections); **HippoRAG**
  arXiv 2405.14831 (PPR retrieval, v2); **Zep/Graphiti** arXiv 2501.13956 (our substrate).
- **OpenAI ChatGPT dreaming** — `openai.com/index/chatgpt-memory-dreaming/` (June 4, 2026; Dreaming
  V0→V3 timeline; background synthesis for freshness/continuity/relevance; time-passage rewriting;
  user-reviewable memory summary; ~5× serving-compute reduction as the scale gate). *Note: the page
  sits behind an aggressive bot wall that serves a fake-404 challenge page to plain HTTP clients —
  it is only reachable with a real (headed) browser.*
