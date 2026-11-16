# Graph embedding dimension

## The problem

NeuralScape keeps two embedding stores built from the same model
(`gemini-embedding-001`):

| Store | Holds | Dimension |
|---|---|---|
| Qdrant (mem0) | memories | 768, fixed (`config.VECTOR_EMBEDDING_DIMS`) |
| Neo4j (Graphiti) | `Entity.name_embedding`, `Community.name_embedding`, `RELATES_TO.fact_embedding` | whatever the Graphiti embedder was configured with |

The Graphiti embedder used to be built with only an API key, so its dimension
came from the ambient `EMBEDDING_DIM` environment variable (Graphiti default
1024), which nothing in this repo sets. Recall scores every graph edge by
comparing its stored `fact_embedding` with the 768-d query embedding. The
comparison returns "unscorable" when lengths differ, so on the default path
every graph edge went unscored and ranked last. The twin-decoration step also
sent 1024-d edge vectors to the 768-d Qdrant collection.

## What changed

* `GRAPHITI_EMBEDDING_DIM` (setting `graphiti_embedding_dim`) is now an
  explicit setting. It is passed through the graph config into the Graphiti
  embedder; the ambient `EMBEDDING_DIM` is no longer relied on.
* **The default stays 1024.** Existing graph data is stored at 1024, so by
  default no stored vector changes and nothing needs migrating.
* Recall works across the two dimensions without a migration (next section).
* `scripts/reembed_graph.py` moves an existing store to another dimension. It
  is opt-in and owner-run.

## Before deploying this change

The explicit setting replaces the ambient `EMBEDDING_DIM` that Graphiti used to
read. An environment that already sets `EMBEDDING_DIM` would now write new graph
vectors at 1024 unless told otherwise (the service logs a warning at startup
when `EMBEDDING_DIM` is set and differs from `GRAPHITI_EMBEDDING_DIM`). So, on
the target store, before the first deploy:

1. Run the dry-run histogram (command below, read-only) and read the lengths.
2. Stored vectors are 1024: nothing to do. Anything else: set
   `GRAPHITI_EMBEDDING_DIM` to the stored length before deploying.

## How scoring works across dimensions

`gemini-embedding-001` is trained with Matryoshka Representation Learning
(MRL). A vector requested with a smaller `output_dimensionality` is the leading
prefix of the full vector, and its values are not unit-normalised for any size
other than 3072. Source: Google's embeddings guide, section "Controlling
embedding size",
<https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size>.

So a 768-d query vector and a 1024-d stored vector can be compared on their
common 768-value prefix; cosine normalises both. `memory/ranking.py`:

* `_graph_vectors_mrl_comparable()` is true only when the graph embedder model
  and the vector embedder model are the same MRL model (`gemini-embedding-001`,
  also under a provider prefix such as the gateway tag).
* `_graph_edge_cosine()` uses `_mrl_cosine()` (prefix cosine) when lengths
  differ and the models are comparable; otherwise it returns `None` as before.
  Equal lengths use the plain cosine. `_unit_cosine()` is unchanged.
* The twin-decoration lookup truncates a longer edge vector to 768 values and
  renormalises it. For a non-MRL pairing it drops the vector (the row is
  looked up by an embed call if a filter needs it, otherwise left undecorated).

Effect on ranking: graph edges now carry a real cosine score in unified
recall, so they compete with vector hits on merit as the fusion was designed.

The prefix relation is checked against the real service by
`integration_tests/test_graph_embedding_mrl_gemini.py` (768-d output versus the
first 768 values of the 1024-d output, cosine >= 0.99). A vector shorter than
the Qdrant collection dimension cannot be used as a query against it, so such
edges are not decorated from their twin.

Graphiti's BM25-only search branch builds a zero query vector of the ambient
`EMBEDDING_DIM` length (`graphiti_core/search/search.py`). It is never compared
against stored vectors, so it needs no change, but it does not follow
`GRAPHITI_EMBEDDING_DIM`.

## Neo4j and vector indexes

Graphiti's Neo4j store has **no vector index**. It creates range and fulltext
indexes only (`graph_queries.py`). Embeddings are plain properties written
with `db.create.setNodeVectorProperty` / `db.create.setRelationshipVectorProperty`
and compared at query time with `vector.similarity.cosine`. No index has to be
dropped or rebuilt when the dimension changes. The query vector and the stored
vectors must have the same length, though, which is why the owner-run procedure
below keeps the window between steps short.

## Owner-run: move a store to 768-d (optional)

Only worth doing for storage (about a quarter less per vector) or if the graph
must share one dimension with Qdrant. The default setup already works at 1024.
This touches your data; run it yourself, never from CI.

1. **Back up the Neo4j store** (volume snapshot or `neo4j-admin database dump`).
2. **Dry-run** to see what exists (read-only):

   ```bash
   export REEMBED_NEO4J_PASSWORD=...        # not on the command line
   python scripts/reembed_graph.py --uri neo4j://<host>:7687 --user neo4j --all-groups
   ```

   It prints, per group, counts by vector length for each embedding property,
   and lists any other embedding-like property it finds but does not handle.
3. **Trial run** on one group with a limit:

   ```bash
   python scripts/reembed_graph.py --uri ... --user neo4j --group-id <group> \
       --mode truncate --target-dim 768 --limit 50 --confirm-backup
   ```
4. **Stop the writers and readers of the graph** (API, worker, graph worker)
   so no search runs with mixed lengths.
5. **Full truncate run** (no API calls; prefix + renormalise; refused unless the
   embedder is an MRL model). Re-running is safe: vectors already at 768 are
   skipped.

   ```bash
   python scripts/reembed_graph.py --uri ... --user neo4j --all-groups \
       --mode truncate --target-dim 768 --confirm-backup
   ```

   `--mode reembed` instead recomputes the vectors from the stored name / fact
   text through the configured Graphiti embedder (costs embedding API calls;
   works for non-MRL models; same flags).
6. **Set `GRAPHITI_EMBEDDING_DIM=768`** in the service environment.
7. **Restart** the API, worker and graph worker.
8. **Verify**: run the dry-run again (everything 768-d); `POST /v1/search` with
   a query that matches a stored fact and confirm a `source: graph` row with a
   numeric `score`.

Rollback: restore the backup and unset `GRAPHITI_EMBEDDING_DIM` (back to 1024).
Truncation is lossy, so the backup is the only way back to the original 1024
values.

Connection safety: the script has no default store. It needs `--uri`/`--user`
(or `REEMBED_NEO4J_URI` / `REEMBED_NEO4J_USER`) and `REEMBED_NEO4J_PASSWORD`,
and ignores the service's `NEO4J_*` settings.
