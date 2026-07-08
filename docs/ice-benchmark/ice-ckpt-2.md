# ICE Checkpoint 2 — engine runtime validated; benchmark harness complete

**Date:** 2026-07-08 · **Branch:** `ice/benchmark` (+ PR #186 pending) · **Executor:** Opus

Follows ice-ckpt-1 (Phase I integration). This checkpoint covers Phase II
(harness) completion + the Phase-III smoke discovery that the native engine had
never run live, and its hardening.

## Merged since ckpt-1
| PR | What |
|----|------|
| #181 | H1 harness core: SystemAdapter protocol + results schema + runner + enforced safety rail (mem cap/timeout→DNF) + real peak-RSS/CPU metrics |
| #183 | H3 Track-Q: independent tree-sitter oracle + structural-QA/NL-locate generators + scorer (hits@k/MRR, N/A honesty) |
| #184 | H2 competitor adapters: graphify 0.9.10 + CBM 0.9.0 ACTUALLY installed + smoke-verified live (pinned SHAs); rail-wrapped; real CBM graph.db fixture produced |
| #185 | H4 report generator: MD + self-contained HTML; nearest-rank percentiles; N/A≠0, DNF first-class; HTML-escaped |
| #186 (pending) | **E8/smoke: native engine runtime hardening** — see below |

## THE headline finding: the native engine had never run live
E2–E6 (merged from dev) shipped with **only mock-based unit tests**. Its
live-Neo4j runtime path had never executed — it had no invocation surface until
E7 added the index CLI. Bringing up the `-p ice` stack and indexing real code
crashed immediately. Fixed in #186:
1. `_run_cypher` used `self.bridge.driver`, but the real mem0 `_AsyncBridge` has
   no `.driver` (only `_loop`) — inject `service._graphiti.driver`; mock-bridge
   fallback retained. Threaded through every construction site.
2. `_index_symbol_cards` crashed on `repo_path / None` for file-less
   (external/inferred) symbols — skip them.
3. Symbol-card embeds exceeded Gemini's 100/batch cap — chunked to ≤100.
Plus regression tests for all three + ICE-stack compose fixes (docling removed,
repo-root build context, API 8599 to avoid the running retrcost stack, neo4j
default DB `memory`, CODE_REPOS `{}`).

## Live validation (in-container, pallets/click subset)
`native_index_cli` → **13 files / 124 symbols / 1153 edges / 10.5s**; 10 Louvain
communities persisted (I2); 625 CodeAnchors (E4); 123 symbol cards embedded to
the `code_index` Qdrant collection (Gemini); post-index liveness pass ran.
Reads: `neighbors`, `blast_radius` (65 symbols), `semantic_layer` (20
community/hotspot facts) — all correct. **The ICE engine now works end-to-end.**

## Suite / gate
Full fast suite: **2091 pass / 1 skip**. Container gate: green (I1..E7 at ckpt-1
= 2089 in-container; E8 re-gated). Functionality floor holds.

## State of the mission
- DONE + merged: Phase I (I1–I4) + E7 + Phase II (H1–H4) + E8 engine hardening.
- VALIDATED: engine indexes + queries real code end-to-end; competitors installed;
  harness + rail + report all built and unit-green.
- REMAINING (measured benchmark) — see `/data/ice/reports/phase-III-runbook.md`:
  ns-ice adapter docker-exec + corpora-mount wiring (# ICE-INTEGRATE); pin
  medium/large/multi-lang corpora; **QUIESCENCE** (3 nsbench stacks were up —
  hard gate for measured Track-P); smoke → full matrix → report → Fable → FINAL
  PR `ice/benchmark → nsbench`.

## Honest status
The FINAL PR is **not** opened — it must carry real, quiesced benchmark numbers,
which require the nsbench factory stacks wound down + the measured run completed.
That is the documented remaining work. Nothing has been fabricated; no numbers
are reported that were not measured live.
