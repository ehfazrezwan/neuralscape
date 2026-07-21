# Finding: native engine has no index/impact trigger surface (plan-premise gap)

**Discovered:** 2026-07-08, executor, during H1 (harness) design, static analysis
of ice/benchmark @ 51b176d (post-I1).

## The gap
`NativeEngine` (adapters/code_graph/native_engine.py) implements the full F2
protocol: `query, neighbors, path, locate, detect_changes, semantic_layer,
index, export_snapshot, import_snapshot`. But only FOUR have any invocation
surface:

| Capability | MCP tool | REST route | query.py delegation | Trigger for benchmark? |
|---|---|---|---|---|
| query      | ✅ query_code_graph | ✅ /code-graph/query | ✅ | symbol_lookup ✅ |
| neighbors  | ✅ get_code_neighbors | ✅ /code-graph/neighbors | ✅ | neighbors_1hop ✅ |
| path       | ✅ code_path | ✅ /code-graph/path | ✅ | path≤4 ✅ |
| locate     | ✅ locate | ✅ /code-graph/locate | ✅ | nl_locate ✅ |
| **index**  | ❌ | ❌ | ❌ | index_cold/incremental/second ❌ |
| **detect_changes / blast_radius** | ❌ | ❌ | ❌ | blast_radius ❌ |
| export/import_snapshot | ❌ | ❌ | (snapshot_cli.py CLI) | snapshot ✅ (via CLI, fixed in I3) |

The masterplan H1 premise "ns-ice: native index via ingest queue" does NOT hold:
`ingest/code_graph.py` + worker.py only ingest a pre-built **graphify
graph.json / GRAPH_REPORT.md** (the GraphifyJsonEngine semantic-distillation
path). They never run the NativeEngine tree-sitter indexer. There is no CLI, no
REST, no MCP, and no queue path that calls `NativeEngine.index()`. blast_radius
(`_blast_radius_bfs`) is internal-only; `detect_changes` is a reindex-diff that
needs a working-tree change, not a symbol→impact query.

## Why this matters
Two of the benchmark's op classes (index_*, blast_radius) and the ability to
create an ns-ice index AT ALL have no drivable surface. Without a fix, NS
ICE-mode cannot be benchmarked on index performance or blast_radius, and the
harness would have to reach into private engine internals (unfair + not
"native surface").

## Resolution (spec-aligned engine completion, NOT a redesign) — task E7
The F2 spec §1 explicitly says "locate and code_impact become new MCP tools +
REST twins" and "index is pull/CI-driven." dev shipped locate but not
code_impact, and no index entrypoint. Completing these IS mission goal #1
("Complete the Intelligent Code Engine"). E7 adds:
1. **Native index CLI** (`adapters/code_graph/native_index_cli.py`, argparse,
   mirrors snapshot_cli): index a repo path into `code--{owner}--{repo}`,
   `--incremental` flag. This is ALSO the fairest index_* measurement — it
   matches how competitors index (run a command), and is the spec's CI-driven
   index entrypoint. Requires `code_repos` config (added in I1) OR a direct
   `--repo-path`.
2. **code_impact**: MCP tool `code_impact` + REST `/v1/code-graph/impact` +
   `query.py` `code_impact(symbol, *, max_hops)` delegation → blast-radius from
   a given symbol (thin public wrapper over the existing `_blast_radius_bfs`).
   Under the same availability gate; additive + backward compatible; MCP core
   tool count unchanged (code_impact is a code-graph tool, gated like locate).

**Sequencing:** E7's code_impact may add a small public method on
native_engine.py, which I2 currently owns — so **E7 runs after I2 merges** to
avoid a conflict. H1's ns-ice adapter targets E7's surface (index CLI +
/impact); interface contract is fixed here so H1 can build in parallel:
- index_cold/incremental/second → `python -m adapters.code_graph.native_index_cli --repo-path <p> --code-space <cs> [--incremental]`
- blast_radius → GET /v1/code-graph/impact?symbol=<fqn>&max_hops=4
- symbol_lookup/neighbors/path/nl_locate → existing /code-graph/{query,neighbors,path,locate}
- snapshot export/import → snapshot_cli.py (I3-fixed)

## Status
NOT blocked — clear spec-aligned path, all Phase-I lanes proceed. Flagged for
Fable's methodology audit. Recorded in status.jsonl.
