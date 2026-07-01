---
name: ingest
description: Ingest files, folders, or a block of pasted context into Neuralscape memory. Use when the user says "ingest this file/folder", "add these docs to memory", "index this PDF/Word doc", "load this documentation into Neuralscape", or pastes a long passage they want remembered. Handles Markdown, HTML, PDF, and MS Office (docx/xlsx/pptx); files are stored as artifacts and referenced back. Works in both Claude Code and Claude Cowork.
---

# Neuralscape — Ingest

Bring larger content into memory: uploaded **files** (a folder or a `.zip`, or individual Markdown/HTML/PDF/Office files) or a **manually-pasted block of context**. Ingested content is chunked into verbatim passages **and** distilled into graph facts, and every produced memory references a stored source artifact you can fetch back.

Two mechanisms, chosen by what's actually available (probe the artifact, not the platform):

- **Files/folder + Claude Code** (a service URL and `curl` are present) → upload to the `POST /v1/ingest/files` endpoint. The server parses rich formats (PDF/Office/HTML via Docling), so binaries work — you don't parse them yourself.
- **Pasted context, or Claude Cowork (no local filesystem)** → the MCP `ingest_text` tool. Works in both platforms.

For a single short fact, use `/neuralscape:remember` instead — this skill is for documents and longer passages.

## What to do

1. **Determine the input:**
   - The user pasted / dictated a block of text → **manual context path** (step 3).
   - The user named a file or folder path → **file-upload path** (step 4).
2. **Resolve `user_id`** (Identity block below) and **`project_id`** — an active project selected this session → else (Claude Code) the plugin's project-id resolution, in order (`PROJECT_ID` override → nearest `.neuralscape-project` marker walking up from cwd → git-repo-root basename → cwd basename) → else omit (global). Pick a **`category`** for the produced memories (default `domain_knowledge`; use `tech_stack`/`convention`/`architecture`/etc. when the docs are project-specific — this is also how artifacts are filed into subfolders on the server).

3. **Manual context path — MCP `ingest_text`:**
   - Call `ingest_text(content=<the text>, title=<short label>, user_id=<resolved>, category=<cat>, project_id=<id or omit>)`.
   - The server persists the text as a Markdown artifact (filed under user/project/category) and the memories reference it. Async by default; report that ingestion was **queued** (passages + facts land shortly).

4. **File-upload path — `POST /v1/ingest/files` (Claude Code only):**
   - Requires a service URL (`CLAUDE_PLUGIN_OPTION_URL` / `NEURALSCAPE_URL`) and `curl`. If neither is present (e.g. Claude Cowork), you **cannot** upload binaries — tell the user file upload needs the Claude Code CLI with a configured URL, and offer to ingest pasted text via the manual path instead. Never error.
   - **A folder:** either zip it and upload the single `.zip` (the server expands it), or upload each file. A `.zip` is expanded server-side; `__MACOSX/`, dotfiles, and nested archives are skipped automatically.
   - Build a multipart request, one `-F files=@<path>` per file (or the zip), plus form fields:
     ```bash
     curl -sS -X POST "<URL>/v1/ingest/files" \
       -H "Authorization: Bearer <API_KEY>" \  # only if an API key/token is set
       -F "files=@/path/to/folder.zip" \
       -F "user_id=<resolved>" \
       -F "category=<cat>" \
       -F "project_id=<id>"                     # omit for global scope
     ```
   - The response is `{"files": [{"filename", "task_id", "file_id"}], "count": N}` (HTTP 202). Each file is parsed + chunked on a dedicated ingest worker, so this won't interrupt the active session.
   - Optionally poll `GET <URL>/v1/memories/status/{task_id}` for per-file `{passages, facts}` counts. The original file is retrievable at `<URL>/v1/ingest/artifacts/{file_id}`.

5. **Confirm** to the user: how many files/how much context was accepted, the category/scope, and that parsing is running in the background (queued). List any files the server skipped.

## Identity block (how to resolve `user_id`)

The MCP `ingest_text` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Privacy

Don't ingest secrets. Skip files that are obviously credential stores (`.env`, key files, `*_rsa`) and redact API-key-shaped strings from pasted context. When content is sensitive, tell the user rather than silently ingesting it.

## Notes

- Supported types: Markdown/`.txt`/`.rst` (read as-is), and PDF, MS Office (docx/xlsx/pptx), HTML (converted to Markdown server-side by Docling, with an in-process MarkItDown fallback). Per-file and archive size caps apply (default 25 MB/file, 200 files/request).
- Files and pasted context are stored as artifacts on the server volume, organized into `{user}/{project}/{category}/` subfolders, and each memory's `source_ref` points back to the artifact (`/v1/ingest/artifacts/{file_id}`).
- Ingest runs on a dedicated worker/queue, isolated from fast memory reads/writes.
- Endpoints are documented at `docs/neuralscape/` in the Neuralscape repo.
