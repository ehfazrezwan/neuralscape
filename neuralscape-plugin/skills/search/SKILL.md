---
name: search
description: Search the user's Neuralscape memory store for relevant facts. Use when the user asks "what do I know about X", "have we discussed Y", "what did I decide about Z", or any retrieval-style question. Returns hybrid vector + graph results. Works in both Claude Code and Claude Cowork (MCP-driven).
---

# Neuralscape — Search

Run this skill when the user wants to recall stored memories on a topic.

This skill prefers the **MCP `recall_memories` tool**, so it works identically in Claude Code and Claude Cowork. A raw-HTTP path to `POST /v1/search` is kept only as a Claude Code fast path for when MCP is unavailable.

## What to do

1. **Resolve the query** — use the user's exact phrasing as the search query. Do not rewrite or summarize unless the query is empty (then ask what they want to look up).
2. **Resolve `user_id`** — see the Identity block below.
3. **Detect project scope** — an active project selected this session → else (Claude Code) the repo's pinned id (first line of a `.neuralscape-project` marker at the repo root, else the git-repo-root/working-directory basename) → else omit. If the user says "across all projects" or "globally", omit `project_id`.
4. **Primary path — MCP (both platforms):** call `recall_memories(query=<user's question>, user_id=<resolved>, project_id=<id or omit>, limit=10)`. This is the default and the only path that works in Cowork.
5. **Optional Claude Code fast path:** only if the `recall_memories` MCP tool is unavailable AND a service URL (`CLAUDE_PLUGIN_OPTION_URL` / `NEURALSCAPE_URL`) plus `curl` are present, fall back to `POST <URL>/v1/search` with body `{"query", "user_id", "project_id", "limit": 10}` and `Authorization: Bearer <API_KEY>` if set (8s timeout). Never require env; never error in Cowork — if neither path is available, say the memory store isn't reachable and stop.
6. **Render results** as a compact markdown list, ordered by score:

   ```
   ## Memories matching "<query>"

   1. **[preference]** TypeScript over JavaScript for new projects (score 0.91, source: vector)
   2. **[architecture]** FastAPI service mounts MCP HTTP at `/mcp/` (score 0.84, source: graph)
   3. **[decision]** Use Qdrant over pgvector (project: neuralscape, source: vector)
   ```

   Show `category` in brackets, then the memory text, then `(score, source)`. Include `project: <id>` when scope is project. When results carry a `source` field, prefer `graph`-sourced results as authoritative when they conflict with `vector` results. If no results, say "No matches found" and suggest broadening the query.

7. **Brief synthesis** (optional, 1-2 sentences) after the list if the results have a clear collective answer.

## Identity block (how to resolve `user_id`)

The MCP `recall_memories` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Notes

- Read-only. Never store or modify memories from this skill.
- The Neuralscape `/v1/search` endpoint is documented at `docs/neuralscape/04-memory-service-core.md#hybrid-search` in the Neuralscape repo.
- Categories and scopes (13 total) are documented at `docs/neuralscape/03-memory-model.md` in the Neuralscape repo.
