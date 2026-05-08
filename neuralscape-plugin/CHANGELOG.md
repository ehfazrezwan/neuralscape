# Changelog

All notable changes to the `neuralscape` Claude plugin are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project adheres to [Semantic Versioning](https://semver.org/).

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
