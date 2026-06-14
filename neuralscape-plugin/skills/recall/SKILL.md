---
name: recall
description: Load the user's relevant Neuralscape memories before planning or acting — their preferences, conventions, tech stack, and past decisions. Use at the start of a task, when the user says "neuralscape on" / "load my context" / "what do you know about me or this project", or whenever you need prior context. Works in both Claude Code and Claude Cowork (MCP-driven).
---

# Neuralscape — Recall

Load stored context so you act on what the user already knows and decided, instead of starting cold. In Claude Code the SessionStart hook does this automatically; this skill is the manual/on-demand equivalent, and it's the **primary** way context gets loaded in Claude Cowork (which runs no hooks).

This skill uses the **MCP tools** (`recall_memories`, `get_project_context`), so it works identically on both platforms — no local service URL or `curl` required.

## What to do

1. **Resolve `user_id`** — see the Identity block below.
2. **Resolve the active `project_id`** (in priority order):
   - An active project already selected this session (via the `project` skill, or one the user named) — reuse it.
   - Claude Code: the repo's pinned id — the first line of a `.neuralscape-project` marker file at the repo root if present, otherwise the git-repo-root (or working-directory) basename. This is the same id the hooks use, so manual and automatic memory stay in one scope.
   - Otherwise: leave it unset (global recall). If the user clearly means a project but none is active, offer the `project` skill to pick one.
3. **Load context:**
   - If a concrete `project_id` is known → call `get_project_context(project_id=<id>, user_id=<resolved>)`. This returns global + project memories organized by category.
   - Otherwise → call `recall_memories(query=<the user's intent, or "preferences conventions tech stack decisions">, user_id=<resolved>, project_id=<id or omit>, limit=15)`.
4. **Render** a compact summary grouped by category (preferences, conventions, tech stack, decisions, …). When `recall_memories` returns a `source` field, prefer `graph`-sourced results as authoritative when they conflict with `vector` results.
5. **Treat what you recalled as authoritative context** for the rest of the turn — don't re-ask the user for things you just loaded.

## Identity block (how to resolve `user_id`)

The MCP `recall_memories` / `get_project_context` schemas mark `user_id` as required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity. Resolve it like this:

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value. (Claude Code / local-no-token: this value is authoritative — get it right.)
- If neither is set → you are almost certainly on a token-authenticated connector (Claude Cowork). Pass a placeholder like `"cowork"` purely to satisfy the schema; the OAuth token determines the real identity, so the placeholder is harmless.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Notes

- Read-only. This skill never stores or modifies memories.
- To save what you learn, use `/neuralscape:remember` (one fact) or `/neuralscape:save-session` (extract from the conversation).
- To switch which project you're scoped to, use `/neuralscape:project`.
