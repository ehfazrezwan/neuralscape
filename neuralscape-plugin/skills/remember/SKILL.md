---
name: remember
description: Save one durable fact to Neuralscape memory now — a preference, decision, convention, gotcha, or architectural choice. Use when the user says "remember this", "save that", "note that I prefer X", or when you learn something worth keeping for future sessions. Works in both Claude Code and Claude Cowork (MCP-driven).
---

# Neuralscape — Remember

Store a single, durable fact via the MCP `remember` tool. Works identically in Claude Code and Claude Cowork (no local service URL or `curl` required).

Use this for one clear fact at a time. To capture many facts from a whole working session, use `/neuralscape:save-session` instead.

## What to do

1. **Compose ONE wiki-style sentence.** Lead with the substance, not the tooling. Apply the same tone the `compile-observations` skill uses (don't restate the rubric here — that skill is the source of truth):
   - ❌ "User said they like TypeScript"
   - ✓ "Prefers TypeScript over JavaScript for all new projects — wants strict mode on by default."
2. **Pick a `category`** (required) from the 13-value enum:
   - **Semantic**: `preference`, `personal_fact`, `technical_skill`, `domain_knowledge`
   - **Project**: `tech_stack`, `convention`, `architecture`, `dependency`
   - **Episodic**: `decision`, `interaction`
   - **Procedural**: `workflow`, `procedure`
   - **Working**: `task_context`
3. **Resolve `project_id`**: an active project selected this session → else Claude Code working-directory basename → else omit (global). Project categories (`tech_stack`, `convention`, `architecture`, `dependency`) generally want a `project_id`.
4. **Resolve `user_id`** — see the Identity block below.
5. **Visibility**: **omit** the `visibility` field to take the per-category default (semantic/personal categories default `private`; team categories like `tech_stack`/`convention`/`architecture`/`dependency`/`decision`/`interaction`/`workflow`/`procedure` default `shared`). Only set `visibility="private"` explicitly when a normally-shared fact is sensitive (internal politics, a personal note, a draft).
6. **Call `remember`** with `content`, `category`, `user_id`, `project_id?`, `visibility?`. Set `wait: true` only if the user wants confirmation that it landed (default is fire-and-forget, async).
7. **Confirm** to the user what you stored, under which category, and its visibility (private vs shared).

## Identity block (how to resolve `user_id`)

The MCP `remember` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value. (Claude Code / local-no-token: authoritative — get it right.)
- If neither is set → you're almost certainly on a token-authenticated connector (Claude Cowork). Pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Privacy

Don't store secrets: skip `<private>` content, API-key-shaped strings (`sk-…`, `gsk_…`, `ghp_…`), passwords, and env-var values. If a normally-shared fact is sensitive, force `visibility="private"`. When in doubt about sharing, default toward private.
