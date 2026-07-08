# System Capabilities Matrix

This document declares which operations each ICEBench system supports and which are N/A (not applicable), with justifications.

**Both competitor tools were actually installed and smoke-run on the benchmark VM**, and the adapters were verified end-to-end against them (see the `test_live_*` tests in each system's `test_adapter.py`):

- **graphify** `0.9.10` — source `safishamsi/graphify` @ `20bfdf60`, installed via `uv` venv.
- **cbm** (codebase-memory-mcp) binary `0.9.0` — source `DeusData/codebase-memory-mcp` @ `ee68144`, prebuilt portable Linux binary.

Every command shown below is the **real, verified** CLI surface (not an assumed interface). Exact pins are in `/data/ice/tools/systems.lock.json`.

**Rail safety:** every competitor subprocess — index AND query AND list AND delete — is routed through `icebench.rail.run_with_rail`, so the hard memory cap + wall-timeout + peak-RSS/CPU capture apply and any OOM/timeout/SIGABRT becomes a DNF row instead of a shared-VM outage.

## Operation Classes

ICEBench defines 5 operation classes for code intelligence systems:

1. **symbol_lookup**: Find a symbol by name and return its definition/location
2. **neighbors_1hop**: Find all symbols directly connected to a given symbol (1 hop in the graph)
3. **path_le4**: Find paths between two symbols with length ≤ 4 hops
4. **nl_locate**: Natural language query to locate symbols/concepts (semantic search)
5. **blast_radius**: Impact analysis — what would be affected by changing a symbol

---

## Graphify Standalone (`graphify`)

**Capabilities:** `symbol_lookup`, `neighbors_1hop`, `path_le4`

**N/A Operations:** `nl_locate`, `blast_radius`

### Supported Operations

Index (verified): `graphify extract <path> --code-only --no-cluster` → `<path>/graphify-out/graph.json`.

| Operation | Implementation (verified) | Justification |
|-----------|---------------|---------------|
| `symbol_lookup` | `graphify explain "<label-or-id>" --graph <graph.json>` | `explain` returns node details (source location, degree, connections). Accepts either the node label (`add()`) or id (`calc_add`). |
| `neighbors_1hop` | `graphify explain "<label-or-id>" --graph <graph.json>` | Same as symbol_lookup; `explain` lists all direct connections (neighbors). |
| `path_le4` | `graphify path "<from>" "<to>" --graph <graph.json>` | `path` finds the shortest path between two nodes. |

### N/A Operations

| Operation | Reason |
|-----------|--------|
| `nl_locate` | Graphify has no natural language query capability. Its `query` command is graph traversal (BFS/DFS), not NL→symbol resolution. Graphify is built on NetworkX + tree-sitter AST (deterministic, no LLM), so there's no semantic layer for NL queries. |
| `blast_radius` | Graphify has no impact analysis tool. It has `affected` (reverse traversal to find nodes impacted by X) but that's a generic graph query, not a true blast radius with change propagation semantics. The brief specifies blast radius as impact analysis, which requires reasoning about call chains, data flow, and change propagation — graphify lacks this. |

---

## CBM Standalone (`cbm`)

**Capabilities:** `symbol_lookup`, `neighbors_1hop`, `path_le4`

**N/A Operations:** `nl_locate`, `blast_radius`

### Supported Operations

Index (verified): `cbm cli index_repository '{"repo_path": "<path>"}'`. Every query tool **requires a `project` argument** (the slug from `list_projects`); the adapter resolves it by matching a project's `root_path` to the corpus path (no slug guessing).

| Operation | Implementation (verified) | Justification |
|-----------|---------------|---------------|
| `symbol_lookup` | `search_graph {project, name_pattern}` | Regex name matching; returns qualified names, labels, file locations, degrees. |
| `neighbors_1hop` | `trace_path {project, function_name, direction:"both", depth:1}` | BFS over the call graph; depth=1 returns direct callers + callees. |
| `path_le4` | `query_graph {project, query}` with `MATCH (a)-[*1..4]-(b) WHERE a.name='...' AND b.name='...' RETURN b.name LIMIT 1` | Variable-length path `[*1..4]`. **Verified quirk:** CBM's Cypher lexer rejects the `p=(...)` path-assignment form and `$params`, so the adapter uses a bare match and single-quote-escapes interpolated names. |

### N/A Operations

| Operation | Reason |
|-----------|--------|
| `nl_locate` | CBM has `semantic_query` (vector search via Nomic embeddings), but this is NOT natural language → symbol locate. `semantic_query` is vector similarity search over code chunks, returning snippets. The brief's `nl_locate` implies "find the symbol that implements X functionality" (e.g., "where is HTTP routing handled") — a higher-level semantic mapping. CBM's vector search doesn't resolve to specific symbols, and the MCP layer (the agent) would be doing the interpretation, which violates the no-added-intelligence rule. |
| `blast_radius` | CBM has `detect_changes` (git diff → affected symbols + risk classification), but this is git-diff based, not a general blast radius tool. It only works on uncommitted changes, not arbitrary "what if I change this symbol" queries. A true blast radius would require transitive call-graph + data-flow impact analysis independent of git state, which CBM's `detect_changes` doesn't provide. |

---

## Fairness & Honesty

Both systems are evaluated **as-shipped**, with no emulation or workarounds:

- **Graphify** is a deterministic AST-based graph builder with CLI query commands. It excels at structural queries but has no semantic layer.
- **CBM** is a tree-sitter + Hybrid LSP graph builder with MCP tools. It has vector search but not full NL→symbol resolution.

Operations marked N/A will raise `UnsupportedOp` in the adapters. The benchmark runner records these as N/A, never as failures or zeros, preserving fairness across systems with different capabilities.

---

## Notes on Index Operations

Both systems support:

- **index_cold**: Full index from scratch (rail-run).
- **index_second**: Second full index (stability probe; CBM's known SIGABRT-on-second-index becomes a DNF).
- **store_size_bytes**:
  - graphify → total size of the `graphify-out/` dir for this corpus.
  - CBM → this corpus's `size_bytes` from `list_projects` (matched by `root_path`), i.e. **scoped to the one corpus**, not a sum of every DB in the cache dir.
- **export_snapshot / import_snapshot**:
  - graphify → byte-copy of `graph.json` (its portable store).
  - CBM → byte-copy of `<cache>/<slug>.db` (CBM's persistent SQLite store). CBM's documented `.codebase-memory/graph.db.zst` portable artifact is only emitted for **git** repos, so the adapter snapshots the real store file directly. `import_snapshot` requires the corpus to have been indexed (so a slug exists) — otherwise it returns N/A rather than guessing CBM's path→slug transform.

Neither system has **index_incremental** in the traditional sense:

- **Graphify**: No incremental mode → returns N/A (DNF with `dnf_reason="incremental_na"`)
- **CBM**: Has an auto-sync background watcher but no explicit incremental index command in CLI mode → N/A

This is honest reporting: the systems don't support it, so the benchmark doesn't fake it.

## Real CBM `graph.db` fixture (for I3)

A **real** CBM `graph.db` was produced (not synthesized) at
`/data/ice/corpora/cbm_fixture/graph.db` (+ `README.md` documenting provenance
and schema) so I3's importer can be validated against actual CBM output. It is a
plain SQLite DB with tables `nodes`, `edges`, `projects`, `file_hashes`,
`node_vectors`, `token_vectors`, `project_summaries`, and an FTS5 `nodes_fts`
virtual table. Note CBM's real store file is `<cache>/<slug>.db` — there is no
file literally named `graph.db` in its cache; the fixture was renamed for I3's
convenience.
