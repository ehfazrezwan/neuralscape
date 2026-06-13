# Changelog

All notable changes to the `neuralscape` Claude plugin are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.1] - 2026-06-13

### Fixed

- **Skill name collision with built-in slash commands.** The `status` and
  `config` skills declared bare frontmatter names identical to Claude
  Code's built-in `/status` and `/config`, so typing `/status` resolved
  to the plugin skill instead of the built-in. Renamed them to `ns-status`
  and `ns-config` (invoked as `/neuralscape:ns-status` and
  `/neuralscape:ns-config`); the built-ins are no longer shadowed.

## [2.2.0] - 2026-05-11

### Added

- **Multi-user support (single-team model).** Each authenticated user has
  a personal memory pool plus access to a shared team knowledge pool.
  Private memories (preferences, personal facts, task context) stay
  visible only to their writer; shared memories (tech_stack, convention,
  architecture, decision, etc.) are visible to anyone authenticated to
  the Neuralscape instance.
- **HMAC-signed per-user tokens.** The API_KEY field now accepts either
  a per-user token (issued via `scripts/issue_user_token.py --user
  <name>` on the backend) or the legacy single-shared key. Per-user
  tokens carry `user_id` in their signed claims, so the server can
  enforce identity without trusting the request body.
- **Default visibility per category.** The compile-observations skill
  now defaults each new memory to private or shared based on its
  category (personal categories private, team categories shared),
  with an explicit override mechanism documented in `SKILL.md`.
- **`visibility` field** on the `remember` MCP tool input schema.
- **`include_shared` + `visibility` filters** on `recall_memories` to
  scope retrieval to one pool or the other.

### Changed

- API_KEY userConfig description rewritten to describe both per-user
  token and legacy shared-key formats.
- USER_ID userConfig description notes it must match the token's
  user_id when using per-user tokens.

### Compatibility

- All existing v2.1.x memories continue to work unchanged. They have no
  `metadata.visibility` field, so the server treats them as **private**
  to their original writer — no cross-user leakage. To bulk-promote a
  category to the shared pool retroactively, run
  `scripts/bulk_promote_visibility.py --owner <user_id> --category
  <category> --to shared --apply`.
- Graphiti graph entries written before v2.2.0 sit under the legacy
  `group_id="global"` / `"project--..."` namespaces; the new search
  doesn't see them until you run
  `scripts/migrate_graph_groups.py --owner <user_id> --apply` once
  per environment.

## [2.1.0] - 2026-05-09

### Added
- **Incremental memory capture (PostToolUse hook).** A new `post-tool-use.js`
  hook fires after every tool invocation, filters read-only tools (Read, Glob,
  Grep, NotebookRead, etc.) and trivial Bash commands at the gate, truncates
  large `tool_output` (head 800 + tail 200), and appends one JSONL row to a
  per-session buffer at `${CLAUDE_PLUGIN_DATA}/observations/{session_id}.jsonl`.
  Pure recorder — zero LLM calls in the hook itself, sub-50ms typical.
- **Client-side extraction (compile-observations skill).** A new
  `skills/compile-observations/SKILL.md` instructs Claude to read the buffer,
  group consecutive ops into work units, apply a quality rubric (decision /
  discovery / gotcha / pattern / bugfix / convention / architecture / outcome),
  and submit each significant work unit as a single memory via the
  `mcp__plugin_neuralscape_neuralscape__remember` MCP tool. Runs on the user's
  Claude Code subscription tokens — backend Gemini is bypassed entirely for
  capture.
- **UserPromptSubmit trigger.** A new `user-prompt-submit.js` hook checks the
  per-session buffer on every user message; when it reaches the configured
  threshold (default 25 observations) or age (default 30 minutes), it injects
  `additionalContext` asking Claude to invoke the compile-observations skill
  before responding. Long-running sessions get fresh memories without waiting
  for a session boundary.
- **SessionStart fallback.** The session-start hook now scans for unprocessed
  buffers from prior sessions and prepends a compile prompt to its
  `additionalContext` so short sessions that didn't hit the threshold still
  flush on the next opening.
- **Stop-time stale marker.** session-end now writes a `.stale` sentinel next
  to any non-empty buffer so the next SessionStart picks it up even if the
  user closes Claude Code without sending another prompt.
- **`/neuralscape:capture` slash command** for on-demand compilation.
- **userConfig prompts**: `COMPILE_THRESHOLD` and `COMPILE_AGE_MIN` for
  tuning the in-session compile cadence (defaults 25 / 30).

### Changed
- **Memory model v2** (backend, additive). The plugin now passes 7 new
  optional fields through to Neuralscape on every `remember` call: `domain`,
  `observation_type`, `concepts`, `source_type`, `related_memory_ids`,
  `confidence`, `expires_at`. Existing v1 memories continue to render
  unchanged with these fields as null. See
  `docs/neuralscape/03-memory-model.md` for the full vocabulary tables.
- **Domain-neutral category descriptions.** All 13 category descriptions
  rewritten to apply equally to coding, research, meetings, writing, and
  ops work — the team uses Claude Code for more than just code.
- `readConfig` is now exported from `utils.ts` so the new hooks can read
  user-configurable thresholds.

### Notes
- No migration required. v1 memories continue to work as-is; v2 fields
  populate only on new writes that supply them.
- The 13 memory categories are unchanged — only their descriptions and
  surrounding metadata gained domain-neutrality.
- Backend additions to support the new flow: `POST /v1/memories/raw/batch`,
  content-hash dedup on insert (idempotent re-flushes), and an
  `expire_old_memories` cron.

## [2.0.2] - 2026-05-08

### Fixed
- `.mcp.json` now uses `${user_config.URL}` and `${user_config.API_KEY}`
  for substitution instead of `${CLAUDE_PLUGIN_OPTION_URL}` and
  `${CLAUDE_PLUGIN_OPTION_API_KEY}`. The `CLAUDE_PLUGIN_OPTION_*` form
  is an environment-variable reference resolved when Claude Code spawns
  a subprocess (stdio MCP servers, hook scripts). HTTP MCP servers are
  invoked directly by Claude Code with no subprocess, so that form
  never resolved — the validator reported "Missing environment
  variables" even though the userConfig values were present in
  `~/.claude/settings.json` and `~/.claude/.credentials.json`. The
  `${user_config.<KEY>}` form is the documented substitution for
  declarative manifest fields and resolves correctly from pluginConfigs.

## [2.0.1] - 2026-05-08

### Fixed
- Drop the explicit `"hooks": "./hooks/hooks.json"` reference from
  `plugin.json`. Claude Code 2.1+ auto-loads the standard
  `hooks/hooks.json` from the plugin root; declaring it explicitly causes
  a "Duplicate hooks file detected" load error in `/doctor`. Hooks still
  fire identically — this is a manifest-only correction.

## [2.0.0] - 2026-05-07

### Breaking
- Configuration now flows through Claude Code/Cowork's `userConfig` prompts at
  install time. The plugin reads `CLAUDE_PLUGIN_OPTION_URL`,
  `CLAUDE_PLUGIN_OPTION_API_KEY`, and `CLAUDE_PLUGIN_OPTION_USER_ID`. Legacy
  `NEURALSCAPE_URL` / `NEURALSCAPE_API_KEY` / `NEURALSCAPE_USER_ID` env vars
  remain as a fallback for one release; they will be dropped in 3.0.
- The `DEFAULT_USER_ID` no longer falls back to a hardcoded `"ehfaz"` value.
  If neither the manifest prompt nor an env var supplies a user ID, the plugin
  uses the OS user (`$USER` / `%USERNAME%`) and logs an error if that's also
  unset, instead of silently writing memories under the wrong identity.

### Added
- `marketplace.json` with full Anthropic schema so users can install via
  `/plugin marketplace add ehfazrezwan/neuralscape` followed by
  `/plugin install neuralscape@neuralscape-plugins`.
- Slash command skills under `skills/`: `status`, `search`, `sync`, `config`.
- Bundled remote MCP via `.mcp.json` pointing at the configured service's
  Streamable HTTP `/mcp/` endpoint — installs the 7 Neuralscape MCP tools
  alongside the plugin without separate setup.
- `LICENSE` (MIT).
- This `CHANGELOG.md`.

### Fixed
- `extractClaudeCodeTurns` no longer advances the transcript offset before
  the flush completes. Offsets are staged in memory and persisted by a new
  `commitClaudeCodeFlush()` after `flushTurns` returns, so a crash mid-flush
  leaves the cursor at its prior position and the next session re-flushes.
- `getProjectId` now uses `path.parse(cwd).name` instead of a `/`-only split,
  so Windows backslash paths resolve correctly.
- All three hooks now write a clear stderr line when `user_id` is missing
  rather than silently no-op'ing or writing under a wrong identity.

### Notes
- `hooks/openclaw-hooks.json` continues to use `${PLUGIN_ROOT}` rather than
  `${CLAUDE_PLUGIN_ROOT}` because OpenClaw's hook runner expansion convention
  has not been confirmed. If OpenClaw mirrors Claude Code, switch this in 2.1.

## [1.0.0]

Initial release. SessionStart context injection, Stop-time conversation
flush + compile, Claude Code + OpenClaw + generic adapters.
