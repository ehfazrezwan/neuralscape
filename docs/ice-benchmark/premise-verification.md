# ICE Phase-I premise verification (executor, static analysis on merged tree)

Verified against ice/integrate-dev (dev 3c4e613 merged) before spending builders.

## I2 — Louvain communities: CONFIRMED STUBBED (fix needed)
- `adapters/code_graph/native_engine.py:17-18` header: "Degree persisted on
  symbols at index time. community_id stubbed (Louvain is a later slice)."
- `semantic_layer()` (line 456) raises EngineCapabilityError: "requires Louvain
  community detection (deferred). E2 persists degree but not community_id."
- `index()` calls `_compute_degrees()` (line 525) at index time — the seam where
  `_compute_communities()` should be added. degree IS persisted; community_id is NOT.
- Fix (I2): compute Louvain at index time on CALLS/IMPORTS projection via
  networkx.community.louvain_communities (deterministic seed), persist community_id
  on :CodeSymbol, rewrite semantic_layer() to read stored props. Guard >200k edges.

## I3 — CBM graph.db.zst reader: CONFIRMED ABSENT (build needed)
- No zstandard dep, no graph.db.zst handling in tree. Matches masterplan.

## I4 — Liveness consumer: HALF-WIRED (fix needed, not just verification)
- Producer EXISTS: `extensions/dreaming/liveness.py` — detect_affected_memories()
  builds events from ChangeReport; apply_liveness_events() flags memories with
  metadata.code_liveness_stale=True (+anchor/reason/flagged_at); reversible.
- GAP 1 (trigger): `process_code_changes_for_liveness` is called ONLY from tests
  (test_code_liveness.py) — no production caller (no post-index hook in the
  ingest worker / index path).
- GAP 2 (consumer): NOTHING reads code_liveness_stale outside liveness.py+tests.
  sweep.py has zero liveness/temporal_reframe references. The flag is written but
  never consumed → temporal_reframe never proposed for code-stale memories.
- temporal_reframe itself is a real dreaming action handled in consolidate.py
  (reversible set, line 47). So the action machinery exists; only the
  code-liveness→sweep bridge is missing.
- Fix (I4): (a) call process_code_changes_for_liveness after native index;
  (b) consume code_liveness_stale in the sweep — deterministically apply
  temporal_reframe for code-ground-truth-stale memories (reversible), OR surface
  the flag to decide()'s candidate set. Live-stack e2e trace during smoke phase.

## Dockerfile / COPY
- Only new dir is adapters/code_graph/queries/ — covered by wholesale
  `COPY neuralscape-service/adapters/` in all 3 stages. No COPY change needed.
- dev already added `--extra code-graph` to builder + test stages.

## Disk-space constraint (executor discovery, 2026-07-08 ~14:48)
- `/` (root, 58G) hosts docker images/containers/volumes; was at 98% and a
  container build FAILED "no space left on device". Freed ~7.7GB via
  `docker image prune -f` + `docker builder prune -f` (safe: dangling + unused
  cache only; nsbench-accuracy stack untouched). Now 77% / 14G free.
- `/data` (98G) has 82G free — all heavy ICE state (corpora, tools, results,
  per-system stores) MUST live here.
- **H1 REQUIREMENT**: the `-p ice` compose override must put neo4j/qdrant/redis
  data + the ingest artifact volume on `/data/ice/...` (bind mounts), NOT
  default docker named volumes on `/`. Otherwise the benchmark will fill root
  and crash (and risk the coexisting nsbench stack). Docker data-root is on `/`.
- nsbench-accuracy stack images consume much of `/`'s docker space and are
  ACTIVE (claude-waveg may need them) — do NOT prune tagged nsbench images.
