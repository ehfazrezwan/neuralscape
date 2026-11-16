# System Capabilities Matrix

This document records which operations each standalone ICEBench system adapter implements, and what each adapter actually calls.

> **Correction notice.** An earlier version of this file declared graphify `blast_radius` and incremental indexing, and CBM `nl_locate`, as N/A. `docs/ice-benchmark/ice-fable-review.md` section 3.3 found the graphify claims contradicted by the installed graphify CLI, and the adapter code has since changed. The tables below now follow the adapter code (`graphify_standalone/adapter.py`, `cbm_standalone/adapter.py`, `capabilities()` and `index_incremental()`). Whether each wired command **works as intended against the installed tool** has not been re-verified for this revision: nothing here was run, and no measurement is implied. Results produced before the adapters were changed (including the `ice-final` report) predate these capabilities and treat them as N/A; treat those N/A cells as unverified or contradicted by review, not as established tool limits.

The adapter modules' own top-of-file docstrings still describe the older, smaller capability sets; `capabilities()` in each file is the authority.

Tool versions the adapters were written against (from their docstrings): graphify `0.9.10` and codebase-memory-mcp (cbm) `0.9.0`. Exact pins for any past run were kept outside this repository.

**Rail safety:** every competitor subprocess is routed through `icebench.rail.run_with_rail`, so the memory cap, wall-timeout and peak-RSS/CPU capture apply and an OOM, timeout or SIGABRT becomes a DNF row.

## Operation Classes

ICEBench defines 5 operation classes for code intelligence systems:

1. **symbol_lookup**: Find a symbol by name and return its definition/location
2. **neighbors_1hop**: Find all symbols directly connected to a given symbol (1 hop in the graph)
3. **path_le4**: Find paths between two symbols with length ≤ 4 hops
4. **nl_locate**: Natural language query to locate symbols/concepts (semantic search)
5. **blast_radius**: Impact analysis — what would be affected by changing a symbol

---

## Graphify Standalone (`graphify`)

**Implemented query ops (`capabilities()`):** `symbol_lookup`, `neighbors_1hop`, `path_le4`, `blast_radius`

**Not implemented:** `nl_locate` (the adapter treats graphify's `query` as graph traversal, not natural-language-to-symbol resolution).

Index: `graphify extract <path> --code-only --no-cluster` writes `<path>/graphify-out/graph.json`.

| Operation | What the adapter calls | Status |
|-----------|---------------|---------------|
| `symbol_lookup` | `graphify explain "<symbol>" --graph <graph.json>` | implemented |
| `neighbors_1hop` | `graphify explain "<symbol>" --graph <graph.json>` | implemented (same command as symbol_lookup) |
| `path_le4` | `graphify path "<from>" "<to>" --graph <graph.json>` | implemented |
| `blast_radius` | `graphify affected "<symbol>" --graph <graph.json> --depth <max_hops>` | implemented; earlier N/A claim contradicted by review (ice-fable-review.md section 3.3) |
| index_incremental | `graphify update <path> --no-cluster` | implemented; earlier N/A claim contradicted by review (ice-fable-review.md section 3.3) |

Caveat on `blast_radius`: `affected` is a reverse graph traversal. It is not data-flow analysis, and neither is the Neuralscape implementation it would be compared with (a breadth-first search over call and import edges). Compare them as like for like only with that stated.

---

## CBM Standalone (`cbm`)

**Implemented query ops (`capabilities()`):** `symbol_lookup`, `neighbors_1hop`, `path_le4`, `nl_locate`

**Not implemented:** `blast_radius` (the adapter treats `detect_changes` as git-diff based, so it does not answer "what if I change this symbol").

Index: `cbm cli index_repository '{"repo_path": "<path>"}'`. Every query tool requires a `project` argument (the slug from `list_projects`); the adapter resolves it by matching a project's `root_path` to the corpus path rather than guessing.

| Operation | What the adapter calls | Status |
|-----------|---------------|---------------|
| `symbol_lookup` | `search_graph {project, name_pattern}` | implemented |
| `neighbors_1hop` | `trace_path {project, function_name, direction:"both", depth:1}` | implemented |
| `path_le4` | `query_graph {project, query}` with `MATCH (a)-[*1..4]-(b) WHERE a.name='...' AND b.name='...' RETURN b.name LIMIT 1` | implemented; the adapter's notes say cbm's Cypher lexer rejected the `p=(...)` form and `$params`, so it uses a bare match and single-quote-escapes the names |
| `nl_locate` | `search_code {project, pattern}` (semantic search over code chunks) | implemented; the adapter's own docstring still lists it as N/A |
| index_incremental | none | N/A: cbm has a background watcher but no explicit incremental command in CLI mode; reported as DNF with `dnf_reason="incremental_na"` |

Caveat on `nl_locate`: `search_code` returns code chunks, so whether a result counts as resolving to the right symbol depends on how the scorer maps it. Read the scorer before drawing conclusions.

---

## Fairness & Honesty

The adapters run each tool as shipped, with no emulation or added intelligence: every answer is the tool's own output, parsed as-is.

- **Graphify** is a deterministic AST-based graph builder with CLI query commands and no semantic layer.
- **CBM** is a tree-sitter graph builder with MCP/CLI tools, including vector search over code chunks.

An operation an adapter does not implement raises `UnsupportedOp`, and the runner records it as N/A rather than as a failure or a zero. An N/A therefore means "the adapter does not call anything for this", which is only as sound as the adapter author's reading of the tool. That reading was wrong for graphify once (see the correction notice), so treat N/A cells with that in mind.

---

## Notes on Index Operations

Both adapters implement:

- **index_cold**: full index from scratch (rail-run).
- **index_second**: second full index (stability probe; cbm's known SIGABRT-on-second-index becomes a DNF).
- **store_size_bytes**:
  - graphify: total size of the `graphify-out/` dir for this corpus.
  - cbm: this corpus's `size_bytes` from `list_projects` (matched by `root_path`), scoped to the one corpus, not a sum of every DB in the cache dir.
- **export_snapshot / import_snapshot**:
  - graphify: byte-copy of `graph.json` (its portable store).
  - cbm: byte-copy of `<cache>/<slug>.db` (its persistent SQLite store). The documented `.codebase-memory/graph.db.zst` artifact is only emitted for git repos, so the adapter snapshots the store file directly. `import_snapshot` requires the corpus to have been indexed (so a slug exists); otherwise it returns N/A rather than guessing the path-to-slug transform.

**index_incremental:**

- graphify: implemented via `graphify update` (see the table above).
- cbm: N/A (`dnf_reason="incremental_na"`).

## CBM `graph.db` fixture

An earlier revision of this file described a real cbm `graph.db` fixture kept outside this repository. It is not part of the repo and is not needed to read the adapters. For reference, cbm's real store file is `<cache>/<slug>.db`; there is no file literally named `graph.db` in its cache.
