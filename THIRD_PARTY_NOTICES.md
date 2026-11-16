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

### The vendored READMEs are upstream's

`mem0/README.md` and `graphiti/README.md` are upstream's own text, kept
verbatim so the subtrees stay mergeable with upstream. They describe upstream
features, some of which Neuralscape does not use or does not ship: for example
mem0's hosted platform and its managed graph memory, and Graphiti's standalone
server and MCP server (which are pruned from this repository). They are not
Neuralscape documentation. What Neuralscape actually uses, and how its copies
differ from upstream, is in `docs/neuralscape/14-upstream-delta-report.md`.

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

## Other bundled or build-time components

These are not vendored source trees (apart from the one marked as committed).
They are either committed as a single file, downloaded when an image is built,
or installed from a package index, and each keeps its own license. Licenses are
the upstream projects' own; check the pinned version when you redistribute an
image.

| Component | Where it is used | How it arrives | License notes |
|---|---|---|---|
| Chart.js v4.4.1 | `neuralscape-bench/static/chart.umd.min.js`, the benchmark dashboard | committed, minified file (its license header is retained in the file) | MIT |
| pyright (npm) | `resolver-svc/`, driven over LSP at index time | `npm install` in the resolver image build, version pinned in its Dockerfile | MIT. See `resolver-svc/THIRD_PARTY.md`. |
| Jedi | `neuralscape-service/adapters/code_graph/code_resolve.py`, in-process symbol resolution | Python dependency (`jedi` in `neuralscape-service/pyproject.toml`) | MIT |
| spaCy `en_core_web_sm` model | lemmatization for mem0's keyword layer | downloaded at image build, or an optional pre-downloaded wheel in `neuralscape-service/vendor/` (git-ignored; see its README) | MIT (spaCy and the model) |
| Docling (`docling-serve`) | rich-format document parsing for ingest | pinned container image in `docker-compose.yml`, run as a separate service | MIT |
| tree-sitter and `tree-sitter-language-pack` | native code indexer | Python dependencies of the optional `code-graph` extra | see each package's metadata |
| graphify (`graphifyy`) | optional code-graph engine | Python dependency of the optional `code-graph` extra | see its package metadata |
| cloudflared, autoheal, docker-socket-proxy | optional tunnel and container-restart services in `docker-compose.yml` | pinned container images, run as separate processes | their own upstream licenses; not distributed with this repository |

The plugin (`neuralscape-plugin/`) is built with esbuild from its own sources;
its runtime bundles contain no third-party packages (it has no runtime
dependencies, only dev tooling).

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
