---
name: search
description: Search the user's Neuralscape memory store for relevant facts. Use when the user asks "what do I know about X", "have we discussed Y", "what did I decide about Z", or any retrieval-style question. Returns hybrid vector + graph results.
---

# Neuralscape — Search

Run this skill when the user wants to recall stored memories on a topic.

## What to do

1. **Resolve the query** — use the user's exact phrasing as the search query. Do not rewrite or summarize unless the query is empty (in that case, ask what they want to look up).
2. **Read config** from `process.env.CLAUDE_PLUGIN_OPTION_URL` and `CLAUDE_PLUGIN_OPTION_USER_ID` (legacy fallbacks: `NEURALSCAPE_URL`, `NEURALSCAPE_USER_ID`). If `USER_ID` is missing, abort and tell the user to run `/plugin config neuralscape@neuralscape-plugins`.
3. **Detect project scope** — call `process.cwd()` and use the basename as `project_id`. If the user explicitly says "across all projects" or "globally", omit `project_id` so the search hits global scope only.
4. **POST `<URL>/v1/search`** with body:

   ```json
   {
     "query": "<the user's question>",
     "user_id": "<USER_ID>",
     "project_id": "<basename of cwd, or null>",
     "limit": 10
   }
   ```

   Set `Authorization: Bearer <CLAUDE_PLUGIN_OPTION_API_KEY>` if the API key is set. Timeout: 8 seconds.

5. **Render results** as a compact markdown list, ordered by score:

   ```
   ## Memories matching "<query>"

   1. **[preference]** TypeScript over JavaScript for new projects (score 0.91, source: vector)
   2. **[architecture]** FastAPI service mounts MCP HTTP at `/mcp/` (score 0.84, source: graph)
   3. **[decision]** Use Qdrant over pgvector (project: neuralscape, source: vector)
   ```

   Show `category` in brackets, then the memory text, then `(score, source)`. Include `project: <id>` when scope is project. If no results, say "No matches found" and suggest broadening the query.

6. **Brief synthesis** (optional, 1-2 sentences) after the list if the results have a clear collective answer.

## Notes

- Read-only. Never store or modify memories from this skill.
- The Neuralscape `/v1/search` endpoint is documented at `docs/neuralscape/04-memory-service-core.md#hybrid-search` in the Neuralscape repo.
- Categories and scopes (13 total) are documented at `docs/neuralscape/03-memory-model.md` in the Neuralscape repo.
