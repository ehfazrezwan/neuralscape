# Phase III runbook — measured benchmark (resumable)

State as of 2026-07-08 ~16:55. Written so the measured full matrix + report +
Fable + final PR can be completed in a resumed session / verified-quiet window.

## What is DONE + VALIDATED
- Phase I (I1–I4) + E7 + Phase II (H1–H4) all merged into `ice/benchmark`.
  Container gate green (2089 in-container at ice-ckpt-1).
- **The native ICE engine now RUNS end-to-end** (was never executed live before —
  E2–E6 had only mock unit tests). Fixed in PR #186 (E8/smoke):
  driver injection, file-less-symbol guard, Gemini 100-batch chunking, +
  ICE-stack compose fixes. Live-validated on pallets/click subset:
  index (13 files/124 sym/1153 edges/10.5s), Louvain communities, anchors,
  card embeddings, neighbors, blast_radius, semantic_layer — all correct.
- `-p ice` stack comes up: neo4j (DB `memory`, ports 7574/7787), qdrant
  (6433/6434), redis (6479), API+workers (**8599**), bind-mounts under
  /data/ice/stack. Creds in /data/ice/.env (GOOGLE_API_KEY + NEO4J_PASSWORD,
  outside repo). Bring up:
  `cd neuralscape-bench/icebench && export $(grep -v '^#' /data/ice/.env|xargs) && docker compose -p ice -f docker-compose.ice.yml up -d`
  (docling removed; not needed).
- Competitors (H2) actually installed + smoke-verified: graphify 0.9.10, CBM
  0.9.0 (pinned SHAs in /data/ice/tools/systems.lock.json). Real CBM graph.db
  fixture at /data/ice/corpora/cbm_fixture/.

## REMAINING for the measured benchmark (in order)
1. **Merge PR #186** (E8/smoke) once its container gate + Copilot are clear.
2. **ns-ice adapter integration (# ICE-INTEGRATE)** — the last wiring:
   - The adapter (`icebench/adapters/ns_ice.py`) runs the index CLI via host
     `python`; it MUST run INSIDE the ice API container (the CLI needs the
     container's Neo4j `memory` DB + service code). Change index_* to
     `docker compose -p ice exec -T neuralscape python -m adapters.code_graph.native_index_cli ...`
     (still wrapped by run_with_rail for RSS/CPU/timeout).
   - Mount corpora into the container: add `- /data/ice/corpora:/corpora:ro` to
     the neuralscape service in docker-compose.ice.yml, and pass
     `--repo-path /corpora/<name>@<sha>` to the in-container CLI.
   - code_space coupling: index CLI writes `--code-space code--ice-bench--<corpus>`;
     the REST query path resolves `repo:<name>` → `_get_native_engine` →
     `code--{user_id}--{name}`. Make them match: EITHER pass graph_id that maps
     to the same code_space (set user_id=ice-bench + CODE_REPOS={"<corpus>":
     "/corpora/<name>@<sha>"} so query resolves the SAME space), OR add a
     graph_id form that carries the code_space directly. Verify a query returns
     hits against a freshly-indexed corpus before trusting Track P/Q numbers.
3. **Corpora**: pin real medium (~100–300k LOC) + large (~0.5–1M, sized to VM) +
   one small repo each TS/JS/Go/Rust/Java, with exact SHAs, in corpora.py +
   corpora.lock.json. `python -m icebench.run corpora` to fetch. (small-py=click
   already pinned.) NOTE H3 oracle is Python-only → Track-Q structural QA is
   Python-only; multi-lang repos get Track-P + capability-matrix coverage.
4. **QUIESCENCE (hard gate)**: 3 nsbench stacks (retrcost, retr-cost-waveg,
   nsbench-accuracy) were up. Measured Track-P requires them wound down (or a
   confirmed quiet window: load low + no active Gemini/battery). State
   quiescence per-run in the report.
5. **Smoke**: `python -m icebench.run index --system ns-ice --corpus small-py`
   then `... query ...` then `... score ...`; fix any remaining harness bugs.
6. **Full matrix** (resumable, background): all corpora × 4 systems (ns-ice,
   ns-graphify, graphify, cbm) × Track P + Q, 3 reps. Results to
   /data/ice/results/raw/. CBM under the 12G rail cap (blowups = DNF).
7. **Report**: `python -m icebench.run report` (H4) → ICE_BENCH_REPORT.md + HTML.
8. **Fable review** (MISSION §Fable): review all ice/* PR diffs + methodology +
   report → /data/ice/reports/ice-fable-review.md; address verdicts.
9. **FINAL PR** `ice/benchmark → nsbench` (open, do NOT merge). Body = mission
   summary + PR list + ice-ckpt-1 + headline tables + report links + Fable link.
   Telegram the URL; status phase=mission-complete.

## Known caveats to state in the report
- Native engine had no live-runtime coverage pre-E8 (fixed); FQN resolution is
  heuristic v1 (some noisy FQNs like module-qualified names).
- Shared-oracle bias (tree-sitter both sides); LSP spot-check skipped (pyright/
  gopls not installable here) — H3.
- Track-Q structural QA Python-only (H3 oracle).
- I3 cbm_import assumed a simpler CBM schema than the real one (H2 documented
  the real schema) — task #11 I3-reconcile (not a benchmark blocker; benchmark
  uses CBM standalone).
