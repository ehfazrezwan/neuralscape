# System Capabilities Matrix

This document declares which operations each ICEBench system supports and which are N/A (not applicable), with justifications.

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

| Operation | Implementation | Justification |
|-----------|---------------|---------------|
| `symbol_lookup` | `graphify explain <symbol>` | Graphify's `explain` command returns node details including source location, community, degree, and connections |
| `neighbors_1hop` | `graphify explain <symbol>` | Same as symbol_lookup; `explain` returns all direct connections (neighbors) |
| `path_le4` | `graphify path <from> <to>` | Graphify's `path` command finds shortest paths between nodes (typically ≤ 4 hops for reasonable graphs) |

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

| Operation | Implementation | Justification |
|-----------|---------------|---------------|
| `symbol_lookup` | `search_graph` with name pattern | CBM's `search_graph` tool allows regex name matching to find symbols. Returns qualified names, labels, file locations. |
| `neighbors_1hop` | `trace_path` with depth=1, direction=both | CBM's `trace_path` performs BFS traversal of the call graph. Setting depth=1 returns all direct neighbors (callers and callees). |
| `path_le4` | `query_graph` with Cypher variable-length path | CBM supports Cypher queries with variable-length paths `[*1..4]`. A path query between two symbols with max 4 hops maps directly. |

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

- **index_cold**: Full index from scratch
- **index_second**: Second full index (stability probe for CBM's known SIGABRT issue)
- **store_size_bytes**: On-disk footprint measurement
- **export_snapshot**: graphify → graph.json copy; CBM → graph.db.zst artifact copy
- **import_snapshot**: Restore from snapshot

Neither system has **index_incremental** in the traditional sense:

- **Graphify**: No incremental mode → returns N/A (DNF with `dnf_reason="incremental_na"`)
- **CBM**: Has auto-sync background watcher but no explicit incremental index command → N/A

This is honest reporting: the systems don't support it, so the benchmark doesn't fake it.
