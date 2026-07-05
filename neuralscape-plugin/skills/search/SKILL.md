---
name: search
description: Search the user's Neuralscape memory store for relevant facts. Use when the user asks "what do I know about X", "have we discussed Y", "what did I decide about Z", or any retrieval-style question. Returns hybrid vector + graph results. Works in both Claude Code and Claude Cowork (MCP-driven).
---

# Neuralscape — Search

Run this skill when the user wants to recall stored memories on a topic.

**MCP only — never curl.** All searches go through the MCP `recall_memories` tool; that surface is the whole point of the plugin, and it works identically in Claude Code and Claude Cowork. Do **not** fall back to `curl`/REST (`/v1/search` etc.) even if a service URL is configured — if the MCP tool is unavailable, say the memory connector isn't reachable and stop.

**Pick the cheapest tool that answers** (measured tiers on the production stack):

- The user wants **matching memories** (this skill's job) → `recall_memories` (~1.6 s, one embedding round-trip). For a broad scan, pass `index_only: true` to get ~50-token index rows, then `get_memories(ids=[...])` (~0.1 s) for only the rows worth reading in full — ~10× cheaper in context tokens at the same speed.
- The user wants **a specific memory they can already identify** → `get_memories` / `list_memories` (instant, no embedding).
- The user asked a **question and wants an answer, not a list** → consider `ask_memory` instead, but know its cost: it runs an LLM synthesis pass (~3 s at `reasoning_level: "minimal"`, ~5 s at `"low"`, more at higher tiers). Never use it for a lookup this skill's list format already answers.

## What to do

1. **Resolve the query** — use the user's exact phrasing as the search query. Do not rewrite or summarize unless the query is empty (then ask what they want to look up).
2. **Resolve `user_id`** — see the Identity block below.
3. **Detect project scope** — an active project selected this session → else (Claude Code) the plugin's project-id resolution, in order (`PROJECT_ID` override → nearest `.neuralscape-project` marker walking up from cwd → git-repo-root basename → cwd basename) → else omit. If the user says "across all projects" or "globally", omit `project_id`.
4. **Call `recall_memories`** — `recall_memories(query=<user's question>, user_id=<resolved>, project_id=<id or omit>, limit=10)`. This is the only path, on both platforms. For broad scans, use the two-step index-first flow instead: `recall_memories(..., index_only=true, limit=20)` → filter the index rows → `get_memories(ids=[...])` for the few full payloads you need. If the MCP tool errors or is missing, report that the memory connector isn't reachable and stop — do not curl.
5. **Render results** as a compact markdown list, ordered by score:

   ```
   ## Memories matching "<query>"

   1. **[preference]** TypeScript over JavaScript for new projects (score 0.91, source: vector)
   2. **[architecture]** FastAPI service mounts MCP HTTP at `/mcp/` (score 0.84, source: graph)
   3. **[decision]** Use Qdrant over pgvector (project: neuralscape, source: vector)
   ```

   Show `category` in brackets, then the memory text, then `(score, source)`. Include `project: <id>` when scope is project. When results carry a `source` field, prefer `graph`-sourced results as authoritative when they conflict with `vector` results. If no results, say "No matches found" and suggest broadening the query.

6. **Brief synthesis** (optional, 1-2 sentences) after the list if the results have a clear collective answer. If the user's request was really a question needing cross-memory reasoning ("so what did we end up deciding?"), offer `ask_memory` as the follow-up rather than stretching this skill.

## Identity block (how to resolve `user_id`)

The MCP `recall_memories` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Notes

- Read-only. Never store or modify memories from this skill.
- Related tools for adjacent jobs: `timeline(anchor=...)` for "what was happening around X?" (chronological, instant), `search_knowledge_graph` for entity/relationship queries, `ask_memory` for synthesized answers with citations.
- Categories and scopes (13 total) are documented at `docs/neuralscape/03-memory-model.md` in the Neuralscape repo.
