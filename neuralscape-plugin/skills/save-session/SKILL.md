---
name: save-session
description: Extract and save the durable facts from the current conversation to Neuralscape at the end of a substantive working session. Use when the user says "save this session", "save this conversation", "compile this session", or when wrapping up meaningful work. Works in both Claude Code and Claude Cowork (MCP-driven) — and is the Cowork stand-in for the Claude Code Stop hook.
---

# Neuralscape — Save Session

Hand the conversation to Neuralscape's extractor so it identifies, categorizes, and stores all the important facts at once via the MCP `remember_conversation` tool. Works identically in Claude Code and Claude Cowork.

This is the conversation-level counterpart to `/neuralscape:remember` (one fact). In Claude Code the `Stop` hook does this automatically at session end; in Claude Cowork (no hooks) this skill is how end-of-session capture happens. The `/neuralscape:sync` skill delegates here when no local service URL is configured.

## What to do

1. **Gather the relevant recent messages** as a list of `{role, content}` objects from the conversation. Include the substantive user/assistant exchanges; the server's LLM does the extraction, so you don't pre-summarize.
2. **Apply the noise filter** — drop turns that carry no signal: assistant responses shorter than ~20 characters, and anything matching `NO_REPLY` / `[heartbeat]` / `[system]`. Skip routine acknowledgements and trivia.
3. **Resolve `project_id`**: an active project selected this session → else Claude Code working-directory basename → else omit (global).
4. **Resolve `user_id`** — see the Identity block below.
5. **Call `remember_conversation(messages=<filtered list>, user_id=<resolved>, project_id=<id or omit>)`.** Extraction is async by default — pass `wait: true` only if the user wants to block until storage completes.
6. **Report** that extraction was queued, and roughly how many messages you sent. The extracted facts become retrievable via `/neuralscape:recall` shortly after.

## Identity block (how to resolve `user_id`)

The MCP `remember_conversation` schema marks `user_id` required, but under token auth the server **ignores** the value you pass and scopes by the authenticated token identity.

- If `CLAUDE_PLUGIN_OPTION_USER_ID` (or `NEURALSCAPE_USER_ID`) is set → pass that value. (Claude Code / local-no-token: authoritative.)
- If neither is set → you're almost certainly on a token-authenticated connector (Claude Cowork). Pass a placeholder like `"cowork"` to satisfy the schema; the OAuth token determines the real identity.
- **Never** block, prompt, or error solely because `user_id` is unknown.

## Notes

- `remember_conversation` runs server-side LLM extraction (this is the conversation path, distinct from the tool-observation path that `capture`/`compile-observations` use in Claude Code).
- Don't include secrets in the messages you pass — the same privacy rules as `/neuralscape:remember` apply.
