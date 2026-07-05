# 21 — Document & File Ingestion

Neuralscape can ingest **files** (Markdown, HTML, PDF, MS Office — docx/xlsx/pptx),
a **folder** or a **`.zip`**, and **manually-provided context** (pasted text).
Everything is chunked into verbatim **passages** and distilled into graph
**facts**, and every produced memory references a **stored source artifact** so it
stays traceable and re-fetchable.

Ingestion runs on a **dedicated worker/queue**, isolated from latency-sensitive
memory reads/writes.

## Entry points

| Interface | What it takes |
|---|---|
| `POST /v1/ingest/text` | A block of context (`content`, `title?`, `category?`, `scope?`, `project_id?`, …). Persisted as a Markdown artifact. |
| `POST /v1/ingest/files` | Multipart upload: one or more files, or a `.zip` (expanded server-side). Form fields set `category`/`scope`/`project_id`/`tags`/`extract_facts`/`index_passages`. Returns one `task_id` per file. |
| `GET /v1/ingest/artifacts/{file_id}` | Download a stored artifact (owner-scoped). This is the `url`/retrieval handle stamped on each ingested memory. |
| MCP `ingest_text` | Manual context from Claude Code **and** Cowork. |
| MCP `ingest_document` | A fetched document with an explicit connector `source_ref`. |
| `/neuralscape:ingest` skill | Client-side: uploads files/folders (CLI) or ingests pasted text (both platforms). |

These write endpoints normally return **202** and enqueue onto the ingest queue;
poll `/v1/memories/status/{task_id}` for `{passages, facts}` counts. (If Redis is
unavailable, `/v1/ingest/text` and `/v1/ingest/document` fall back to running the
ingest inline and return **200** with the result.)

## Pipeline

```
upload → persist artifact (volume) → enqueue (ingest queue) → Ingest Worker:
   extract_text (Docling → Markdown; MarkItDown fallback)
   → chunk_text (paragraph-aware passages)
   → passages (verbatim, vector-only) + facts (LLM-distilled, graph-linked)
   → each memory carries source_ref → the stored artifact
```

Modules (`neuralscape-service/ingest/`):

- **`extract.py`** — tiered parsing. Plain text/Markdown is read as-is; rich
  formats go to the **Docling** container (`DOCLING_URL`, `to_formats=md`), with
  an in-process **MarkItDown** fallback when Docling is disabled/unreachable.
- **`archive.py`** — `iter_archive` expands a `.zip` with stdlib `zipfile`,
  skipping dirs / `__MACOSX/` / dotfiles / nested archives and enforcing
  per-file, member-count, and total-uncompressed caps (zip-bomb guards).
- **`storage.py`** — persists artifacts to `INGEST_STORAGE_DIR` under
  `{user}/{project}/{category}/{content_hash}{ext}`, and builds the
  `source_ref` (path + `/v1/ingest/artifacts/{id}` handle). Content-hash naming
  makes re-uploads idempotent.
- **`chunking.py`** / **`pipeline.py`** — deterministic chunker + the
  passages-plus-facts pipeline (unchanged core).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `INGEST_QUEUE_NAME` | `neuralscape:ingest` | Dedicated ARQ queue for ingest + connector sync. |
| `INGEST_STORAGE_ENABLED` | `true` | Persist uploads/context as artifacts. |
| `INGEST_STORAGE_DIR` | `~/.neuralscape/ingest` | Artifact volume (shared by API + ingest worker). |
| `DOCLING_ENABLED` | `true` | Prefer the Docling container for rich formats. |
| `DOCLING_URL` | `http://docling:5001` | Docling-serve base URL (empty → MarkItDown only). |
| `DOCLING_TIMEOUT_S` | `120` | Per-file convert timeout. |
| `INGEST_MAX_FILE_MB` | `25` | Reject a single file larger than this. |
| `INGEST_MAX_FILES` | `200` | Max files per request (post zip-expansion). |
| `INGEST_MAX_ARCHIVE_UNCOMPRESSED_MB` | `200` | Total unzipped-size cap (per zip). |
| `INGEST_MAX_REQUEST_MB` | `500` | Total bytes processed per upload request. |
| `INGEST_STORAGE_ENABLED` | `true` | Persist uploads/context as artifacts (else ship bytes through Redis). |

## Ingesting authoritative standards

When the `standard` tier is enabled (`STANDARDS_ENABLED=true`), a **dictator**
(`DICTATOR_USER_IDS`) can bulk-ingest org standards through this same pipeline by
passing `visibility=standard` (the `visibility` form field on `/v1/ingest/files`,
or the arg on `ingest_text` / `ingest_document`). Enforcement:

- **Authorship is gated at the API/MCP boundary** — a non-dictator (or a request
  when the tier is disabled) is rejected **synchronously** (`403` / MCP error),
  so no ingest jobs are enqueued only to fail later in the worker.
- Standards are **always global-scope** (`store_raw` forces `scope=global`,
  `project_id=None`) regardless of the scope field.
- Verbatim **passages** stay in the standard pool for semantic `recall` but are
  excluded from the always-on session block and from process bundles.

### How standards surface (hybrid)

To keep the always-on session-start context small as the standard set grows,
standards surface in two ways:

- **Always-injected (critical):** standards tagged **`critical`** or **`always`**
  are injected into every session's binding-directive block regardless of
  relevance. Tag the truly non-negotiable rules this way (or set `tags` when
  ingesting a standards doc).
- **On-demand (the rest):** every other standard surfaces **relevance-ranked**
  through `recall`/`search` (which always searches the standard pool), not the
  always-on block. Retrieve the full set explicitly for review via
  `recall` with `visibility="standard"`.

For a single directive prefer `remember` with `visibility=standard`
(add `tags:["critical"]` to always-inject it); use this pipeline for standards
**documents**.

## Deployment notes

- Run the ingest worker: `uv run arq worker.IngestWorkerSettings` (compose:
  `neuralscape-ingest-worker`; Helm: `ingest-worker-deployment.yaml`).
- Docling is a separate container (`ghcr.io/docling-project/docling-serve`, port
  5001) — CPU/RAM-heavy (ML models); size it accordingly. Toggle off with
  `DOCLING_ENABLED=false` to rely on the in-process MarkItDown parser.
- **The API and ingest worker must share the artifact volume.** The API writes
  the uploaded file and hands the worker a *path* (not the bytes), so the worker
  reads it back from the same volume. On Kubernetes this needs a **ReadWriteMany**
  PVC (e.g. Filestore on GKE) — see `ingestStorage` in the Helm values.
- Object storage (GCS/S3) is a later swap behind the small `ingest/storage.py`
  interface.

## Known limitations / future work

- **File upload from Claude Cowork is not supported.** Uploading files/folders
  goes through the `POST /v1/ingest/files` multipart endpoint, which the
  `/neuralscape:ingest` skill drives with `curl` in **Claude Code** (where a
  service URL + bearer token are available). In **Cowork** the only
  authenticated channel is the OAuth MCP connector, whose token isn't exposed to
  a skill's shell — and MCP tool calls carry JSON, not binary — so neither raw
  multipart nor a tool call can carry an uploaded file there. Cowork can still
  ingest **pasted text** via the `ingest_text` MCP tool; local **text/markdown**
  files could be read and passed to `ingest_text` too.
- **To close this later, the options are:** an `ingest_file` MCP tool that takes
  a base64 payload over the authenticated MCP channel (works for small–medium
  files; base64 + MCP payload limits rule out large binaries like a multi-MB
  PPTX), or a presigned-upload-URL flow (handles large files, needs object
  storage). Deferred until there's a concrete need.
- **Artifact re-fetch is provenance-only.** Each memory's `source_ref.url`
  (`/v1/ingest/artifacts/{file_id}`) serves the original file (authenticated,
  owner-scoped). A Claude Code agent can `curl` it; there is no MCP tool that
  streams artifact bytes, so it's not a first-class fetch flow in Cowork. The
  primary reuse path is recalling the stored passages/facts, not re-downloading.
