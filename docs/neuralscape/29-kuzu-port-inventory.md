# 29 — Kuzu Port Inventory (solo engine, unit 3 workplan)

**Status:** working document for the `solo/kuzu-test-matrix` unit. Updated as sites are ported.
**Baseline:** `solo-engine @ 9375d6b` (post unit 2 — `_build_graph_driver` seam live, real-Kuzu smoke green).

Two cross-cutting findings dominate the port:

1. **`KuzuDriverSession.run()` returns `None`** (`graphiti_core/driver/kuzu_driver.py:277-283`). Every NS raw-Cypher *read* shaped as `result = await session.run(...); await result.data()/.single()` breaks on Kuzu with `AttributeError`. The portable read path both drivers implement identically is `rows, _, _ = await driver.execute_query(cypher, **params)` — Neo4j returns `Record`s and Kuzu returns dicts, and both support `record["key"]`/`.get()`. **Every read below must migrate to `execute_query` (a no-op behavior change on Neo4j).**
2. **Kuzu's schema is static** (`kuzu_driver.py:54-132`: node tables `Episodic, Entity, Community, RelatesToNode_, Saga`; rel tables `RELATES_TO, MENTIONS, HAS_MEMBER, HAS_EPISODE, NEXT_EPISODE`). NS writes custom properties that exist on **no** table — `memory_id`, `ns_visibility`, `ns_owner`, `ns_connector_id/type`, `ns_source_url`, `wiki_path`, `wiki_synthesized_at`, `strategy_playbook_path`, `strategy_synthesized_at`, `dream_superseded_by`, `dream_invalidated_at`, `dream_path`, `dreamt_at` — plus a net-new `Source` node label and `DERIVED_FROM` rel. Kuzu rejects `SET n.<undeclared>` and unknown labels at bind time. **Tier 0 = an NS schema-extension hook (ALTER TABLE ADD + new tables) applied after `KuzuDriver.setup_schema()`.**

Also: `RELATES_TO` edges are **reified** on Kuzu — `(n)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m)`, with all edge columns (`fact`, `fact_embedding`, `episodes`, `valid_at`, `invalid_at`, `expired_at`, `uuid`, `group_id`) living on the `RelatesToNode_` **node**. Any NS pattern `()-[r:RELATES_TO]->()` with property access must be re-modeled as `MATCH (rtn:RelatesToNode_)`.

## Site inventory and porting tiers

Status legend: ☐ pending · ◐ in progress · ☑ ported+tested (both providers).

### Tier 0 — prerequisite (blocks all of extensions/)

| Status | Work | Notes |
|---|---|---|
| ☑ | NS Kuzu schema extension hook | `memory/kuzu_schema.py`, applied post-attach in `memory/core.py` (kuzu only). `Source` keyed on synthetic `key` = `<connector_id>::<source_key>` (Kuzu PKs are single-column) — the Kuzu branch of `attach_source_ref` must MERGE on that |
| ☑ | FTS bootstrap | Discovered during port: `build_indices_and_constraints` is a Kuzu **no-op** and `setup_schema` only makes tables — graphiti's own BM25 leg had no indices on Kuzu. Bootstrap now runs `INSTALL FTS; LOAD EXTENSION FTS` + graphiti's `get_fulltext_indices(KUZU)`. Empirically verified (probe + tests): indices are **maintained on insert**, not snapshots |

### Tier 1 — trivial (typed labels already; read-seam swap only)

| Status | # | Site | Purpose | Port |
|---|---|---|---|---|
| ☑ | 4 | `memory/graph_admin.py` `delete_episode` | `MATCH (e:Episodic {uuid}) DETACH DELETE` | verified as-is on real Kuzu (`test_kuzu_port.py`) |
| ☑ | 5 | `memory/write.py` `_graph_episode_exists` | episode idempotency probe | ported to `execute_query`; probe query verified on real Kuzu |

### Tier 2 — mechanical provider branches

| Status | # | Site | Purpose | Kuzu divergence |
|---|---|---|---|---|
| ☑ | 3 | `memory/graph_admin.py` fulltext episodes | episode excerpt search | ported: provider branch reusing `get_nodes_query`, unified `execute_query` read; group-scope verified on real Kuzu. **Nuance for parity bench:** Kuzu applies `TOP := $limit` *before* the group filter (same as graphiti's own episode search) — multi-group stores can under-fill vs Neo4j; fix with over-fetch only if DMR shows it |
| ☑ | 6 | `extensions/dreaming/graph_patcher.py:78-86` `attach_memory_id` | stamp memory_id/visibility/owner | label-less `MATCH (n)` → per-label loop; `datetime($s)` → native datetime param; `coalesce()` → verify or `CASE WHEN`; needs Tier 0 |
| ☑ | 7 | `graph_patcher.py:147-163` `attach_source_ref` | Source node + DERIVED_FROM link | same as #6 + `Source`/`DERIVED_FROM` tables (Tier 0); `TransientError` retry gated to neo4j branch |
| ☑ | 8/9 | `graph_patcher.py:228-248, 300-320` `patch_wiki_path*` | wiki back-refs | label-less → per-label loop; needs Tier 0 |
| ☑ | 12 | `graph_patcher.py:453-457` `patch_dream_path_by_memory_ids` | dream-diary back-ref | same shape as #8/9 |
| ☑ | 10 | `graph_patcher.py:397-401` invalidate (node arm) | tombstone marking | label-less w/ property map → per-label loop |
| ☑ | 14 | `extensions/strategy_synthesizer/graph_patcher.py:39-45` | playbook back-ref | same shape as #8/9 |

### Tier 3 — redesign (reified edges / label-less scans / Cypher UNION)

| Status | # | Site | Purpose | Redesign |
|---|---|---|---|---|
| ☑ | 2 | `memory/search.py` graph-result enricher | fetch NS props + edge embedding by uuid | ported: explicit per-label node arm (bare MATCH would double-report reified edges) + RelatesToNode_ edge arm, app-side union; embedding round-trip verified |
| ☑ | 11 | `graph_patcher.py` invalidate (edge arm) | bi-temporal edge invalidation | ported: RelatesToNode_ + Python-side exclusively-derived filter (no unverified list predicates); semantics matrix green on real Kuzu (exclusive dies, co-asserted/empty survive, fail-safe, unconditional node marking) |
| ☑ | 1 | `memory/reads.py` `list_projects` | distinct project group_ids | ported: per-label queries + app-side union; `STARTS WITH` verified working on Kuzu |
| ☑ | 13 | `extensions/dreaming/bridges.py` `SHARED_ENTITY_CYPHER` | hub entities across pools | ported: raw per-label fetch + full aggregation pipeline in Python (key/casefold, head-of-collect, DISTINCT, >=2/>=2 filter, ordering, limit) |
| ☑ | 15-18 | `scripts/identity.py`, `scripts/migrate_graph_groups.py` | offline admin | direct bolt driver + untyped rel patterns + `EntityEdge` label — RESOLVED as documented-Neo4j-only: solo has one identity and no legacy groups; scripts print a clear error under kuzu (no code change needed — they require NEO4J_URI explicitly) |

### Dialect verification checklist (no in-subtree Kuzu precedent — test each)

- `STARTS WITH` (#1, scripts) · `UNION`/`UNION ALL` in Kuzu Cypher (#1, #2 — graphiti avoids it, unions in Python) · `coalesce()` (#6, #7, #11) · `size()/substring()/toLower()/trim()/head()/collect(DISTINCT)/all()` (#11, #13, #16) · `MERGE` on rel tables (#7).

### Test strategy

The existing graph tests are MagicMock-seam tests (`svc._graphiti = MagicMock()`, `_run_on_bridge` stubs) — they pass against any driver and give **zero** Kuzu coverage. Real coverage follows the `test_graph_provider_seam.py::TestKuzuSmoke` template: construct the real embedded driver on a tmp path, run the ported query via `execute_query`, assert rows. Each ported site gets (a) a real-Kuzu test and (b) where behavior is subtle (#11), a same-fixture Neo4j-vs-Kuzu parity assertion in the integration suite.

### Exception handling

Only NS product-code `neo4j.exceptions` import: `extensions/dreaming/graph_patcher.py:164` (`TransientError` deadlock retry in `attach_source_ref`). Kuzu is single-writer embedded — no transient deadlocks; gate the import + retry to the neo4j provider branch.

## E2E findings (solo/e2e-verify — real daemon, embedded stores, live Gemini)

The first true end-to-end boot (NS_MODE=solo, zero containers) surfaced three defects no unit test had caught:

1. **Startup crash without Redis** — `TaskManager.connect()` was unguarded, and a `None` pool raised `AttributeError` from enqueues, which the write paths' sync fallbacks (`except (ConnectionError, OSError)`) do NOT catch. Fixed: `connect()` skips Redis when `task_backend != "redis"`, and the pool rests as a falsy `_DisabledPool` sentinel whose any-use raises `ConnectionError` — routing every write into the sync fallbacks. (Interim until unit 4's real inline backend.)
2. **`KuzuDriver` missing `_database`** — `graphiti.add_episode` compares `driver._database` to the group_id before its no-op base-class `clone()`; Neo4j sets the attribute, Kuzu didn't → every group-scoped episode write crashed. Subtree-patched in the driver `__init__`.
3. **Kuzu DDL drift** — the edge model's `reference_time` and four Saga columns (`summary`, `first/last_episode_uuid`, `last_summarized_at`) are SET by the Kuzu save queries but were never declared in `SCHEMA_QUERIES` (static schema → binder errors). Fixed in the DDL for fresh stores + `_GRAPHITI_DRIFT_ALTERS` in the NS bootstrap for existing ones, and guarded forever by `TestKuzuSchemaDrift` — a static cross-check of every Kuzu save query's SET columns against the DDL.

Lesson recorded: the MagicMock seam tests and even the real-driver query tests couldn't find these — only booting the actual daemon and pushing a conversation through Gemini → Graphiti → Kuzu did. The parity bench (unit 7) must run against the daemon, not the library.
