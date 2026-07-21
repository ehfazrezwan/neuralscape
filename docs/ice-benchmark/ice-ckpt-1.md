# ICE Checkpoint 1 — Phase I complete (integration + engine completion)

**Date:** 2026-07-08 · **Branch:** `ice/benchmark` @ `25d9fe4` · **Executor:** Opus

## Scope of Phase I
Integrate dev's code-intel engine into the nsbench bench lineage and complete
the Intelligent Code Engine, per ICE_MASTERPLAN.md Phase I + the executor-added
E7 engine-surface completion.

## PRs merged (into `ice/benchmark`, serial merge commits, never squash)
| PR | Task | What |
|----|------|------|
| #177 | I1 | Merge dev `3c4e613` (E1–E6 code-intel) into bench lineage; resolve the sole conflict (memory/write.py, 9 hunks, both-sides-win: speaker+workspace+occurred_at); fix standards-pool NameError regression + add `code_repos` config seam (Copilot #9/#5); commit F2 spec under docs/ |
| #179 | I2 | Louvain communities persisted at index time (deterministic seed=42, stable ids by min-fqn, isolated→-1, >200k-edge guard); `semantic_layer()` rewritten to read stored props (no query-time NetworkX) |
| #178 | I4 | Close the code-liveness → dreaming consumer loop (sweep consumes `code_liveness_stale` via direct Qdrant scroll → deterministic reversible `temporal_reframe`, post-apply flag clear, dry-run safe); fix 3 liveness.py bugs (get_engine import, collection mismatch, embed→scroll) |
| #180 | I3 | CBM `graph.db.zst` migration importer (zstandard in code-graph extra only; archive-bomb guards; relation whitelist vs Cypher injection; colon span format); fix snapshot_cli.py AttributeError (wrong service API) |
| #182 | E7 | Complete native engine invocation surface: native index CLI + `code_impact` MCP tool + REST `/v1/code-graph/impact` + post-index liveness trigger; blast_radius via fuzzy resolver; graphify→501 |

## Suite floor (functionality floor — hold at/above this)
- nsbench ck-2 baseline: 1986 pass / 1 skip.
- After I1 (dev merge + fixes): **2043 pass / 1 skip** (in-container).
- After I2+I3+I4 merged: 2074 pass / 1 skip (host).
- After E7 merged: 2089 pass / 1 skip (host).
- **In-container floor at ice-ckpt-1 (this tree): 2089 pass / 1 skip** (this is the mission functionality floor — hold at or above it).

## Container gate (HARD rule — run on the exact merged tree, ice/benchmark @ 25d9fe4)
`docker build -f neuralscape-service/Dockerfile -t ice-gate . && docker run --rm ice-gate`
+ `--target runtime` build:
- **PASS.** test-stage build EXIT 0; in-container suite EXIT 0 → **2089 passed, 1 skipped**; runtime-stage build EXIT 0.
- (First attempt hit a transient Docker containerd tag-commit quirk — `AlreadyExists: ice-gate:latest` — not a code or disk failure; resolved by removing the stale tag and re-running. Noted for reproducibility.)

Dockerfile COPY lists: the only new dir dev added (`adapters/code_graph/queries/`)
is covered by the wholesale `COPY neuralscape-service/adapters/` in all 3 stages;
dev's `--extra code-graph` on builder+test stages carried through. No COPY change
needed. I3 added `zstandard` to the code-graph extra (in-image via the extra).

## Functionality preservation (MISSION format)
- **Nothing removed.** All changes additive. nsbench Wave-1/2/G behavioral wins
  survive byte-for-byte (I1 both-sides-win conflict resolution).
- **Regression fixed, not introduced:** the dev merge carried in a dev bug
  (standards-pool `_search_standard_pool` NameError → silent loss of standards
  recall); I1 fixed it + added a regression test so it can't recur.
- **MCP tool surface:** 22 core tools unchanged. Code-graph tools grow from 4
  (query/neighbors/path/locate) to 5 (+code_impact), gated behind the
  `code-graph` extra exactly like `locate`; absent-extra path still 501/self-
  disables. `MCP_TOOL_SCHEMA_OVERHEAD_TOKENS` updated accordingly.
- **REST schemas:** additive (`/v1/code-graph/impact` new; others unchanged).
- **Feature flags default to prior behavior:** `code_repos` defaults `{}`
  (native engine self-reports "not configured" — pre-E2 behavior); liveness
  consumer runs under existing dreaming gates (secret/silent-sweep/reversible),
  reversible + idempotent.

## Plan-premise gaps found & resolved (transparency for Fable audit)
1. **Native engine had no invocation surface** (index/blast_radius/detect_changes
   only as methods; masterplan's "native index via ingest queue" was false — the
   queue only ingests graphify graph.json). Resolved by E7 (spec-aligned: F2 §1
   calls for index-in-CI + code_impact tool). Full writeup:
   `engine-surface-gap.md`.
2. **`code_repos` config missing** → native engine unreachable. Fixed in I1.
3. **Liveness consumer half-wired** (producer flagged, no sweep consumer / no
   post-index trigger). Closed in I4 (consumer) + E7 (post-index trigger).
   Premise details: `premise-verification.md`.

## Review discipline
Every PR: executor review + Copilot review requested; all Copilot findings
triaged (fixed / routed / false-positive) with new tests for every real defect.
Notable catches: I4 consumer was non-functional in prod (wrong staged-row keys) —
caught + reworked to Qdrant scroll; I3 Cypher-injection whitelisted; E7
blast_radius char-by-char resolver fixed.

## Coexistence / quiescence
Code + host/in-container unit tests only this phase (no Gemini, no heavy index).
nsbench-accuracy stack up but idle (load ~2/8); shared status stale since 09:22
(ck-2 approved, executor idle). True quiescence to be re-verified before the
Phase-III measured full matrix; stated in the benchmark report.

## Next
Phase II harness: H1 (merged #181 — SystemAdapter protocol + results schema +
runner + enforced rail + real resource metrics). In flight: H2 (competitor
adapters), H3 (Track-Q). Then H4 (report gen). Then Phase III smoke → quiesced
full matrix → report → Fable review → final PR `ice/benchmark → nsbench`.
