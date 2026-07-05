---
name: search
description: Search the user's Neuralscape memory store for relevant facts. Use when the user asks "what do I know about X", "have we discussed Y", "what did I decide about Z", or any retrieval-style question. Returns hybrid vector + graph results. Works in both Claude Code and Claude Cowork (MCP-driven).
---

# Neuralscape — Search

Run this skill when the user wants to recall stored memories on a topic.

**MCP first — the MCP tool is the primary surface.** All searches go through the MCP `recall_memories` tool, which works identically in Claude Code and Claude Cowork. The REST API deliberately mirrors the MCP surface, so `POST /v1/search` exists as a **fallback only** — use it solely when the MCP tool is genuinely absent from the session (step 5), never as a substitute while the tool is available. Loading this skill is not a reason to switch to `curl`.

**`recall_memories` is the default for anything relevance-shaped — never downgrade it.** The other tools are for *different jobs*, not cheaper versions of this one:

- You already hold **specific memory IDs** → `get_memories` (~0.1 s point fetch — a follow-up to a search, not a search).
- The user wants an **inventory or audit**, not relevance → `list_memories`.
- The user wants **chronology** ("what was happening around X?") → `timeline`.
- The user asked a **question and wants a reasoned answer, not a list** → `ask_memory` (an LLM synthesis pass, ~3 s at `reasoning_level: "minimal"` to ~5 s+ at higher tiers — worth it exactly when synthesis is the deliverable).

For an explicitly **broad scan** (survey-style, `limit` > 10), `index_only: true` is available: the server runs the *identical* hybrid search and ranking — only the return payload shrinks to ~50-token index rows (~10× cheaper in context). Titles are lossy, so expand anything even plausibly relevant via `get_memories(ids=[...])` rather than ruling it out by title. When in doubt, use plain full-payload recall — it is the accuracy-measured configuration.

## What to do

1. **Resolve the query** — use the user's exact phrasing as the search query. Do not rewrite or summarize unless the query is empty (then ask what they want to look up).
2. **Resolve `user_id`** — see the Identity block below.
3. **Detect project scope** — an active project selected this session → else (Claude Code) the plugin's project-id resolution, in order (`PROJECT_ID` override → nearest `.neuralscape-project` marker walking up from cwd → git-repo-root basename → cwd basename) → else omit. If the user says "across all projects" or "globally", omit `project_id`.
4. **Primary path — MCP (both platforms):** `recall_memories(query=<user's question>, user_id=<resolved>, project_id=<id or omit>, limit=10)`. For broad scans, the two-step index-first flow: `recall_memories(..., index_only=true, limit=20)` → filter the index rows → `get_memories(ids=[...])` for the payloads worth reading in full.
5. **REST fallback (Claude Code only, and only when MCP is truly unavailable):** if — and only if — the `recall_memories` MCP tool is not present in the session AND a service URL (`CLAUDE_PLUGIN_OPTION_URL` / `NEURALSCAPE_URL`) plus `curl` are available, fall back to `POST <URL>/v1/search` with body `{"query", "user_id", "project_id", "limit": 10}` and `Authorization: Bearer <API_KEY>` if set (8s timeout). The REST API mirrors the MCP surface, so results render identically. Never take this path while the MCP tool exists; never require env; never error in Cowork — if neither path is available, say the memory store isn't reachable and stop.
6. **Render results** as a compact markdown list, ordered by score:

   ```
   ## Memories matching "<query>"

   1. **[preference]** TypeScript over JavaScript for new projects (score 0.91, source: vector)
   2. **[architecture]** FastAPI service mounts MCP HTTP at `/mcp/` (score 0.84, source: graph)
   3. **[decision]** Use Qdrant over pgvector (project: neuralscape, source: vector)
   ```

   Show `category` in brackets, then the memory text, then `(score, source)`. Include `project: <id>` when scope is project. When results carry a `source` field, prefer `graph`-sourced results as authoritative when they conflict with `vector` results. If no results, say "No matches found" and suggest broadening the query.

7. **Escalate when results look thin.** If the top results are off-topic or sparse: broaden the query / raise `limit` and re-run; if the request was really a question needing cross-memory reasoning ("so what did we end up deciding?"), escalate to `ask_memory` — it runs its own multi-pass search internally. Never settle for a weak answer when one escalation step remains untried.
8. **Brief synthesis** (optional, 1-2 sentences) after the list if the results have a clear collective answer.

## Identity block (how to resolve `user_id`)

The MCP `recall_memories` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value.
- If neither is set (likely Claude Cowork) → pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Notes

- Read-only. Never store or modify memories from this skill.
- Related tools for adjacent jobs: `timeline(anchor=...)` for "what was happening around X?" (chronological, instant), `search_knowledge_graph` for entity/relationship queries, `ask_memory` for synthesized answers with citations.
- Categories and scopes (13 total) are documented at `docs/neuralscape/03-memory-model.md` in the Neuralscape repo.
