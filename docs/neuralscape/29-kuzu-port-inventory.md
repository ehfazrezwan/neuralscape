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
| ☐ | NS Kuzu schema extension hook | `ALTER TABLE … ADD` the NS columns on `Entity`/`Episodic`/`Community`/`RelatesToNode_`; `CREATE NODE TABLE Source` + `CREATE REL TABLE DERIVED_FROM`; tolerant of already-exists; runs post-`setup_schema()` for `graph_provider=kuzu` only |

### Tier 1 — trivial (typed labels already; read-seam swap only)

| Status | # | Site | Purpose | Port |
|---|---|---|---|---|
| ☐ | 4 | `memory/graph_admin.py:223-227` `delete_episode` | `MATCH (e:Episodic {uuid}) DETACH DELETE` | already `execute_query` + typed + `DETACH DELETE` supported → verify test only |
| ☐ | 5 | `memory/write.py:327-335` `_graph_episode_exists` | episode idempotency probe | `session.run` → `execute_query` |

### Tier 2 — mechanical provider branches

| Status | # | Site | Purpose | Kuzu divergence |
|---|---|---|---|---|
| ☐ | 3 | `memory/graph_admin.py:115-128` fulltext episodes | episode excerpt search | `CALL db.index.fulltext.queryNodes` → `CALL QUERY_FTS_INDEX('Episodic','episode_content',$q,TOP:=$limit)`; reuse `graph_queries.get_nodes_query` |
| ☐ | 6 | `extensions/dreaming/graph_patcher.py:78-86` `attach_memory_id` | stamp memory_id/visibility/owner | label-less `MATCH (n)` → per-label loop; `datetime($s)` → native datetime param; `coalesce()` → verify or `CASE WHEN`; needs Tier 0 |
| ☐ | 7 | `graph_patcher.py:147-163` `attach_source_ref` | Source node + DERIVED_FROM link | same as #6 + `Source`/`DERIVED_FROM` tables (Tier 0); `TransientError` retry gated to neo4j branch |
| ☐ | 8/9 | `graph_patcher.py:228-248, 300-320` `patch_wiki_path*` | wiki back-refs | label-less → per-label loop; needs Tier 0 |
| ☐ | 12 | `graph_patcher.py:453-457` `patch_dream_path_by_memory_ids` | dream-diary back-ref | same shape as #8/9 |
| ☐ | 10 | `graph_patcher.py:397-401` invalidate (node arm) | tombstone marking | label-less w/ property map → per-label loop |
| ☐ | 14 | `extensions/strategy_synthesizer/graph_patcher.py:39-45` | playbook back-ref | same shape as #8/9 |

### Tier 3 — redesign (reified edges / label-less scans / Cypher UNION)

| Status | # | Site | Purpose | Redesign |
|---|---|---|---|---|
| ☐ | 2 | `memory/search.py:1104-1123` graph-result enricher | fetch NS props + edge embedding by uuid | edge arm must read `RelatesToNode_` on Kuzu; `UNION ALL` → app-side union |
| ☐ | 11 | `graph_patcher.py:402-411` invalidate (edge arm) | bi-temporal edge invalidation | `SET r.invalid_at` on reified node; highest logic risk — parity test against Neo4j behavior required |
| ☐ | 1 | `memory/reads.py:638-657` `list_projects` | distinct project group_ids | per-label queries + app-side union; verify `STARTS WITH` |
| ☐ | 13 | `extensions/dreaming/bridges.py:59-70` `SHARED_ENTITY_CYPHER` | hub entities across pools | per-label rewrite; verify `toLower/trim/head/collect(DISTINCT)/size` |
| ☐ | 15-18 | `scripts/identity.py`, `scripts/migrate_graph_groups.py` | offline admin | direct bolt driver + untyped rel patterns + `EntityEdge` label — keep Neo4j-only, document skip on Kuzu (solo has one identity and no legacy groups) |

### Dialect verification checklist (no in-subtree Kuzu precedent — test each)

- `STARTS WITH` (#1, scripts) · `UNION`/`UNION ALL` in Kuzu Cypher (#1, #2 — graphiti avoids it, unions in Python) · `coalesce()` (#6, #7, #11) · `size()/substring()/toLower()/trim()/head()/collect(DISTINCT)/all()` (#11, #13, #16) · `MERGE` on rel tables (#7).

### Test strategy

The existing graph tests are MagicMock-seam tests (`svc._graphiti = MagicMock()`, `_run_on_bridge` stubs) — they pass against any driver and give **zero** Kuzu coverage. Real coverage follows the `test_graph_provider_seam.py::TestKuzuSmoke` template: construct the real embedded driver on a tmp path, run the ported query via `execute_query`, assert rows. Each ported site gets (a) a real-Kuzu test and (b) where behavior is subtle (#11), a same-fixture Neo4j-vs-Kuzu parity assertion in the integration suite.

### Exception handling

Only NS product-code `neo4j.exceptions` import: `extensions/dreaming/graph_patcher.py:164` (`TransientError` deadlock retry in `attach_source_ref`). Kuzu is single-writer embedded — no transient deadlocks; gate the import + retry to the neo4j provider branch.
