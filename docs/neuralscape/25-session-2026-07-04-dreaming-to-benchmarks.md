# 25 — Session Chronicle: Dreaming Engine → Feature Freeze → Benchmarks (2026-07-03/04)

A record of the single sustained session that took Neuralscape from "wiki_synthesizer isn't
doing a great job" to a feature-frozen system with 22 merged PRs, a competitive roadmap fully
built out, and a six-suite benchmark battery mid-flight. Written as the durable account —
what was built, in what order, what broke, and what was learned.

---

## 1. The Dreaming engine (PR #97)

The session began by replacing the wiki_synthesizer with **dreaming mode**
(`extensions/dreaming/`), designed after a study of three shipped "dreaming" systems and the
memory-consolidation literature (spec: `docs/DREAMING_MODE_SPEC.md`):

- **Sweep shape**: cron-driven light → deep → REM per memory pool. LIGHT stages and scores;
  DEEP runs one consolidation LLM pass emitting MERGE / INVALIDATE / PRUNE / REWRITE /
  TEMPORAL-REFRAME actions under a hybrid adoption posture (reversible actions auto-apply;
  destructive ones below a confidence gate go to a shadow report; secrets always hard-delete);
  REM reflects over the staged pool and writes insight memories back as first-class,
  recallable rows (`source_type="dream"`).
- **Non-destructive core**: Graphiti bi-temporal invalidation (`invalid_at`, never delete) +
  Qdrant tombstones excluded from search but always liftable.
- **Gate economy**: cheap time gate → owner-safe distributed lock → expensive volume gate,
  with an idempotent skip on unchanged staged-id sets.
- **The humane vault**: subject-first topic pages, per-project hubs, Home.md map of content,
  dream diaries — replacing the old taxonomy-bucketed wiki (23 legacy scopes migrated).
- **Found live during deployment**: an in-process sweep starved `/health` until autoheal
  killed the API — the admin endpoint now enqueues sweeps onto the graph worker (202 + poll).

Review hardening (CodeRabbit 6 + nitpicks, Copilot 3) added: token-based compare-and-delete
lock ownership, 409 on disabled `/run`, per-scope migration archiving, and in-code enforcement
that dream-authored rows are never merge material (the prompt contract alone was not enough).
A UTC-container gotcha was fixed with `DREAMING_CRON_ANCHOR_HOUR` so "nightly" means the
operator's night.

## 2. Research and the competitive roadmap (PRs #98, #102, #104, #109)

Deep source-level research of the strongest reference systems — **MemPalace** (~57k ★,
spatial palace + Zettelkasten + L0/L1 wake-up stack), **Honcho** (Plastic Labs — provenance
DAG, epistemic levels, activity-aware dreaming, dialectic tiers), **claude-mem** (~86k ★ —
progressive disclosure, File Read Gate, live observer compression) — was distilled into
`24-competitive-roadmap.md`. The core finding: **none of the three has real memory
consolidation**; the dreaming lifecycle + bi-temporal graph is the moat, and the roadmap
steals their UX/retrieval-economics ideas while keeping that edge.

Amendments during the day: A3 slimmed to a settling guard + manual trigger; A4 given three
contractual recall-safety guardrails; Phase F added (Graphify code-graph adapter — library
imported, never rebuilt; NS remains the sole interaction surface); Phase G added (Google's
Open Knowledge Format — export first, ingest second, explicitly *not* an adapter); E2 amended
with the LeanCTX honest-savings methodology (build the meter, don't integrate the binary).

## 3. The ship loop: 22 PRs in one day

Every item ran the same loop: worktree off `dev` → build with logic-bucketed commits → unit +
isolated E2E → PR → Copilot review requested explicitly (it never fires on its own) → valid
findings fixed with replies → central merge in dependency order, with dedicated rebase agents
for the recurring `mcp_server.py` registry collisions (census 14 → 25 tools across the day).
Up to five builder agents ran in parallel, partitioned by file ownership; E2E stayed
serialized via a lock with per-agent collections and dream-pool users.

| Wave | PRs | Delivered |
|---|---|---|
| Core | #97 | dreaming engine + humane vault |
| Docs | #98 #102 #104 #109 | roadmap + graphify/OKF/LeanCTX amendments |
| A | #100 #99 #106 | provenance DAG + epistemic levels; times_derived reinforcement; salience dynamics + settling guard + schedule_dream |
| B | #101 #105 | Home L0/L1 + fixed page skeletons; Bridges + Faded + identity cards |
| C | #108 #110 | index-first recall, batch get, timeline; ask_memory tiers + checkpoint + queue_status + webhook |
| D | #113 #116 | plugin progressive disclosure + session notes + redaction; File Read Gate + hygiene (plugin 2.8→2.9) |
| E | #115 #117 | surprisal REM + SSE stream + honest token meter; session summarizers + context/assemble + custom extraction instructions |
| F/G | #107 #111 | Graphify code_graph adapter; OKF export/ingest round-trip |
| Hygiene | #103 #112 #118 | ELv2 license + notices + Redis pin; Dockerfile okf COPY hotfix; gateway batch-embed fix |
| Bench | #114 | six-suite accuracy harness |

## 4. Production bugs found and fixed live

1. **Dockerfile selective-COPY gap (#112)**: PR #111's new top-level `okf/` package was
   missing from the COPY lists — containers raised `ModuleNotFoundError`, the dreaming
   extension silently dropped out while `/health` stayed green. Lesson: extension import
   failures are non-fatal by design; grep startup logs after every deploy, and a container
   smoke test belongs in CI.
2. **Silent gateway embed data loss (#118)**: with `LLM_GATEWAY_ENABLED=true` (the live
   config), the gateway's embedding endpoint 400s mem0's batched input and
   `extract_and_store` swallowed the failure — conversation extraction reported success while
   storing **zero facts**. Fixed with a configurable embed batch size (gateway path → 1), a
   single-input-rejection runtime fallback, and failure propagation to task status. Lesson:
   an except-and-continue around a store pipeline converts infra breakage into silent data
   loss.

## 5. Licensing

The repo had **no license** (all-rights-reserved by default). Given the business model
(self-hosting free, hosted seats later), **Elastic License 2.0** was adopted (#103) with
`THIRD_PARTY_NOTICES.md` covering the Apache-2.0 mem0/graphiti subtrees (§4(b) modification
notices for the restored graph layer), the Neo4j-GPLv3-as-a-process posture, and a Redis tag
pin (the floating `:7` tag had silently crossed the RSALv2/SSPLv1 boundary at 7.4).

## 6. Feature freeze and the benchmark battery

After #118, the livestack was redeployed from frozen dev (verified: dreaming + bridges +
cards enabled, zero import failures, savings meter live at `/v1/metrics`) and the six-suite
battery launched on an isolated stack — the union of every suite the competitor set
publishes: LoCoMo, LongMemEval_S/_M, DMR, BEAM, ConvoMem, MemBench (~$143 estimated, all six
datasets verified fetchable). Evaluation findings live in
`26-benchmark-evaluation-findings.md`; the headline: the first scores were a floor, not a
measurement — the maintainer's stop-and-verify call recovered +11.2pp on DMR by proving the
answer phase had raced the async store.

## 7. Session learnings worth keeping

- **Enforce prompt contracts in code.** Every "the LLM must never X" needs a validator on the
  response path.
- **In-process E2Es don't see container breakage.** Selective Dockerfile COPY + non-fatal
  extension loading = invisible production regressions.
- **Async stores need drain barriers in any measurement loop.** 202-and-poll semantics are
  fine for products and fatal for benchmarks that answer immediately after enqueueing.
- **Parallelism policies differ by activity**: file-ownership partitioning made 5-wide
  building safe; scoring requires one suite at a time on an otherwise idle machine.
- **Registry files are the collision magnets.** Any shared tail-append surface (MCP tool
  registry, dispatcher chains, census tests, doc inventories) will conflict on every parallel
  PR — state the expected merged census in every rebase brief.
- **Honest meters build trust.** The savings meter's first live number was *negative* — and
  that is precisely why its positive numbers mean something.
