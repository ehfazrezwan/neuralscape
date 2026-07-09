# Third-Party Notices

Neuralscape itself is licensed under the Elastic License 2.0 (see `LICENSE`).
The following third-party components are vendored in this repository as git
subtrees and **remain under their original licenses**. Their license files
travel with them and must not be removed.

## mem0 (`mem0/`)

- Upstream: https://github.com/mem0ai/mem0
- License: Apache License 2.0 (`mem0/LICENSE`)
- **Modified by Neuralscape.** Per Apache License 2.0 §4(b), notice is hereby
  given that this copy differs from upstream. The principal modifications:
  - The self-hostable OSS graph memory layer (`mem0/graphs/`,
    `graph_memory.py`, `memgraph_memory.py`), removed upstream in mem0
    PR #4805, is restored and maintained here.
  - `graphiti_memory.py` and the `GraphStoreFactory` are Neuralscape
    additions that never existed upstream (they present Graphiti as a mem0
    graph provider).
  - The full delta is documented in
    `docs/neuralscape/14-upstream-delta-report.md`.

## Graphiti (`graphiti/`)

- Upstream: https://github.com/getzep/graphiti
- License: Apache License 2.0 (`graphiti/LICENSE`)
- Modified by Neuralscape where noted in
  `docs/neuralscape/14-upstream-delta-report.md`.

## CBM (codebase-memory-mcp)

- Upstream: https://github.com/DeusData/codebase-memory-mcp
- License: MIT License
- **Used by the CBM bridge service** (`cbm-bridge/Dockerfile`). The Dockerfile
  DOWNLOADS the prebuilt MIT-licensed release binary (v0.9.0 portable
  linux-amd64 tarball) from the project's GitHub releases at image-build time and
  runs it unmodified behind our REST shim. The bridge exposes structured JSON
  tools only (no raw Cypher). No modifications to CBM; no redistribution of
  modified sources (the binary is fetched at build time, not vendored into
  this repo).

## Backing services (not distributed, referenced by `docker-compose.yml`)

These run as separate, unmodified processes pulled as official images at
deploy time; they are not part of this source distribution.

| Service | Image | License notes |
|---|---|---|
| Neo4j | `neo4j:5-community` | GPLv3. Consumed unmodified over the Bolt protocol as a separate process; not linked or redistributed. |
| Redis | `redis:7.4-alpine` (pinned) | Redis ≥ 7.4 is RSALv2/SSPLv1 (not OSI open source). Fine for self-hosting and for running inside a hosted Neuralscape offering; the restriction applies to offering *Redis itself* as a managed service. Valkey (BSD-3) is the drop-in fallback if this ever tightens. |
| Qdrant | `qdrant/qdrant` | Apache License 2.0. |

## Trademarks

"mem0", "Zep"/"Graphiti", "Neo4j", "Redis", and "Qdrant" are trademarks of
their respective owners. References here are factual and do not imply
endorsement.
