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
| `INGEST_MAX_ARCHIVE_UNCOMPRESSED_MB` | `200` | Total unzipped-size cap. |

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
