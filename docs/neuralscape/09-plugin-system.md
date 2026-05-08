---
title: Plugin & Adapter System
date: 2026-05-06
tags: [reference, neuralscape, plugin, claude-code, openclaw, vault]
source: handwritten
---

# Plugin & Adapter System

## Overview

Neuralscape's plugin system is the bridge that lets multiple AI clients — Claude Code, **Claude Cowork**, OpenClaw, future agents like Cursor — write into the **same** memory layer without each client needing to understand Qdrant, Neo4j, or the [13-category taxonomy](./03-memory-model.md). The design is two-layered:

1. **Client-side TypeScript plugin** (`neuralscape-plugin/`) — a single Node bundle exposing three hook entry points. It auto-detects the host client from the shape of the JSON payload arriving on stdin, runs a per-client *adapter* that normalizes the payload into a canonical `ConversationTurn`, and then makes ordinary HTTP calls to the Neuralscape service.
2. **Server-side `conversation_compiler` extension** — a [FastAPI extension](./02-service-architecture.md) that owns `/flush` and `/compile`, runs Gemini extraction, dual-writes to the Obsidian vault, and synthesizes daily knowledge articles.

The crucial property is that *everything past the adapter is client-agnostic*. Adapters are the only place where Claude Code transcript files, OpenClaw `message:sent` events, or any future client's wire format are touched. After normalization, a turn from Claude Code and a turn from OpenClaw look identical, follow the same [extraction pipeline](./07-async-pipeline.md), and end up in the same Qdrant collection, the same Neo4j graph, and the same vault folder.

**Cowork uses Claude Code's plugin model verbatim** — same `.claude-plugin/plugin.json` manifest, same hook events, same marketplace catalog. The plugin ships exactly one set of files; Cowork users install via the Cowork UI (Customize → Browse plugins), Claude Code users install via `/plugin install`, and both end up with the same on-disk plugin cache and identical runtime behavior. There is no `.cowork-plugin/` or separate manifest. (Cowork *does* restrict MCP server source types — it blocks `npm` and `pip` — but Neuralscape's MCP runs over HTTP at the configured service URL, which Cowork accepts.)

This page walks the entire surface: plugin layout and build, the v2 manifest with userConfig prompts and bundled MCP, the detection function and three adapters, the lifecycle of both supported hook surfaces, the server-side compiler extension, the dual-write mechanic that fans memories out to Obsidian, and finally the contributor checklist for adding a new adapter.

## Components map

The plugin and the compiler extension live in different parts of the repo and ship on different cadences:

| Layer | Location | Language | Trigger |
|-------|----------|----------|---------|
| Hook entry points | `neuralscape-plugin/scripts/*.js` | TS → ESM bundle | Host client invokes via `node` |
| Adapters | `neuralscape-plugin/src/adapters/` | TypeScript | Called from hook entry points |
| Core helpers | `neuralscape-plugin/src/core/` | TypeScript | `flush()`, `compile()` HTTP calls |
| `/flush` and `/compile` routes | `neuralscape-service/extensions/conversation_compiler/routes.py` | Python | HTTP POST from plugin |
| Gemini extraction + vault writes | `neuralscape-service/extensions/conversation_compiler/flush.py` | Python | Inside ARQ worker or sync request |
| Daily synthesis | `neuralscape-service/extensions/conversation_compiler/compile.py` | Python | `/compile` endpoint |
| Vault writer | `neuralscape-service/extensions/conversation_compiler/obsidian_writer.py` | Python | Both flush and compile |

The TS plugin only ever speaks to the service over HTTP — it never imports or links anything Python. Likewise the compiler extension never knows whether a turn came from Claude Code, OpenClaw, or a curl test script.

## Plugin layout & build

```
neuralscape-plugin/
├── .claude-plugin/plugin.json        v2 manifest (full schema + userConfig)
├── .mcp.json                         remote HTTP MCP at <URL>/mcp/
├── hooks/
│   ├── hooks.json                    Claude Code: SessionStart, Stop
│   └── openclaw-hooks.json           OpenClaw: message:sent, session:end
├── skills/
│   ├── status/SKILL.md
│   ├── search/SKILL.md
│   ├── sync/SKILL.md
│   └── config/SKILL.md
├── src/
│   ├── adapters/{detect,claude-code,openclaw,generic}.ts
│   ├── core/{types,compile,flush}.ts
│   ├── hooks/{session-start,conversation-turn,session-end}.ts
│   ├── types.ts                      13-category taxonomy mirror
│   └── utils.ts                      HTTP client, identity, stdin parser
├── esbuild.config.js
├── package.json                      v2.0.0; postinstall builds bundles
├── LICENSE                           MIT
└── CHANGELOG.md
```

The repository root also carries `.claude-plugin/marketplace.json` so the repo *is* the marketplace; users install with `/plugin marketplace add ehfazrezwan/neuralscape` followed by `/plugin install neuralscape@neuralscape-plugins` (Claude Code) or the equivalent Cowork UI flow.

`npm run build` runs `esbuild.config.js`, which bundles the three hook source files in `src/hooks/` into three matching `scripts/*.js` outputs. Each is an ESM Node 18+ bundle with a `#!/usr/bin/env node` shebang and `minify: true`, so it can be invoked directly as a command. There are exactly three entry points because there are exactly three lifecycle moments: session start (sync, blocks the user briefly while context loads), per-turn capture (async, fast), and session end (async, may take a minute while compile runs).

Built scripts are gitignored and regenerated by the `postinstall` script in `package.json`, so a marketplace install does not require the user to run `npm run build` themselves.

The Claude Code hook manifest at `neuralscape-plugin/hooks/hooks.json:1-30` is what Claude Code reads when it discovers the plugin:

```json
{
  "hooks": {
    "SessionStart": [{"matcher": ".*", "hooks": [{
      "type": "command",
      "command": "node \"${CLAUDE_PLUGIN_ROOT}/scripts/session-start.js\"",
      "timeout": 30,
      "statusMessage": "Loading memory context..."
    }]}],
    "Stop": [{"hooks": [{
      "type": "command",
      "command": "node \"${CLAUDE_PLUGIN_ROOT}/scripts/session-end.js\"",
      "timeout": 60,
      "async": true
    }]}]
  }
}
```

The OpenClaw manifest at `neuralscape-plugin/hooks/openclaw-hooks.json:1-31` follows the same shape but binds to OpenClaw's event names — `message:sent` (15s, async) for `conversation-turn.js` and `session:end` (15s, async) for `session-end.js`. It uses `${PLUGIN_ROOT}` rather than `${CLAUDE_PLUGIN_ROOT}` because OpenClaw's expansion convention has not been confirmed against the Claude Code spec; if your OpenClaw build doesn't expose that variable, swap to whichever it does export. Note that `session-end.js` is **shared**: the same script handles Claude Code's `Stop` hook and OpenClaw's `session:end`. The script uses the adapter system to figure out what kind of payload it received.

## v2 manifest, marketplace, and userConfig

The v2 plugin manifest (`neuralscape-plugin/.claude-plugin/plugin.json`) is no longer minimal — it carries the full Anthropic plugin schema plus three `userConfig` prompts that Claude Code/Cowork shows the user at install time:

```json
{
  "name": "neuralscape",
  "version": "2.0.0",
  "skills": "./skills/",
  "hooks": "./hooks/hooks.json",
  "mcpServers": "./.mcp.json",
  "userConfig": {
    "URL":     { "type": "string", "title": "Neuralscape service URL", "required": true },
    "API_KEY": { "type": "string", "title": "API key (optional)", "sensitive": true },
    "USER_ID": { "type": "string", "title": "Your user ID", "required": true }
  }
}
```

At runtime the plugin reads these as `process.env.CLAUDE_PLUGIN_OPTION_URL`, `CLAUDE_PLUGIN_OPTION_API_KEY`, and `CLAUDE_PLUGIN_OPTION_USER_ID`. The legacy `NEURALSCAPE_*` env vars from v1 still work as a fallback for one release of overlap; they will be dropped in v3. The `sensitive: true` flag on `API_KEY` routes the value to the OS keychain rather than `settings.json`.

The repo-root `marketplace.json` (`name: neuralscape-plugins`) lists exactly one plugin and points `source: "./neuralscape-plugin"` at the subdirectory. That naming is why the install string is `neuralscape@neuralscape-plugins` rather than `neuralscape@neuralscape`.

Bundled MCP lives in `neuralscape-plugin/.mcp.json`, a tiny config that points at the Streamable HTTP MCP transport mounted on the service:

```json
{
  "mcpServers": {
    "neuralscape": {
      "type": "http",
      "url": "${CLAUDE_PLUGIN_OPTION_URL}/mcp/",
      "headers": { "Authorization": "Bearer ${CLAUDE_PLUGIN_OPTION_API_KEY}" }
    }
  }
}
```

When the plugin is enabled, the user's MCP-capable clients (Claude Code, Cowork) automatically pick up the seven Neuralscape tools — `recall_memories`, `remember`, `remember_conversation`, `get_project_context`, `search_knowledge_graph`, `list_memories`, `delete_memories` — without separate Claude Desktop config. See [08-mcp-server](./08-mcp-server.md) for the full tool surface.

## Slash commands (skills)

The `skills/` directory holds four pure-prompt skills that surface as discoverable commands:

| Skill | Trigger | What it does |
|---|---|---|
| `/neuralscape:status` | "is neuralscape working?", health questions | Calls `GET /health`, reports per-backend status, displays resolved URL and user_id |
| `/neuralscape:search` | "what do I know about X?", recall questions | POSTs to `/v1/search`, renders ranked vector + graph results |
| `/neuralscape:sync` | "save this conversation now" | Manually flushes recent turns to `/flush`, then triggers `/compile` |
| `/neuralscape:config` | "what's my neuralscape config?" | Shows URL, user_id, API-key state with provenance (userConfig vs env vs default) |

Each `SKILL.md` carries YAML frontmatter (`name`, `description`) and Markdown instructions describing what to do. There are no companion scripts — Claude executes the HTTP calls with its existing tools. This keeps the skill surface lean and lets Claude pick up improvements (better prompt fallback handling, retries) automatically rather than rebuilding the plugin.

## Client detection & adapters

Detection is a single, terse function at `neuralscape-plugin/src/adapters/detect.ts:15-21`:

```typescript
export function detectClient(raw: Record<string, unknown>): ClientType {
  // Claude Code hooks include transcript_path or hook_event_name
  if (raw.transcript_path || raw.hook_event_name) return "claude-code";
  // OpenClaw events have type + action fields
  if (raw.type && raw.action) return "openclaw";
  return "generic";
}
```

That's the entire identification mechanism: shape-based duck typing on the stdin JSON. Claude Code always passes either `transcript_path` (for `Stop`) or `hook_event_name` (for `SessionStart`), OpenClaw always wraps events as `{type, action, ...}`, and anything else falls through to the `generic` adapter, which is meant for testing and pre-normalized payloads.

The same file (`detect.ts:26-48`) holds the registries. `getTurnExtractor()` and `getSessionEndExtractor()` are switch-statement dispatch over `ClientType`, returning the matching `extractClaudeCodeTurns`, `extractOpenClawTurns`, or `extractGenericTurns` (and corresponding session-end extractors). Adding a new client is a matter of adding a string to the union and a case to each switch — see the checklist at the bottom of this page.

The contract every adapter must satisfy is in `neuralscape-plugin/src/core/types.ts:8-37`:

```typescript
export interface ConversationTurn {
  userMessage: string;
  assistantResponse: string;
  sessionId: string;
  channel: string;
  timestamp: string;
  projectId?: string;
  userId: string;
}

export interface SessionEndInput {
  date: string;
  userId: string;
  shouldCompile: boolean;
}
```

The three current adapters differ chiefly in how they recover `userMessage` / `assistantResponse`:

- **Claude Code** (`neuralscape-plugin/src/adapters/claude-code.ts:123-190`) is the most involved. The Stop hook receives a `transcript_path` pointing at a JSONL file the host maintains during the session. The adapter reads a sibling `.neuralscape-offset` file via `readFlushOffset(transcriptPath)` so already-processed messages are skipped, then parses the transcript with `parseTranscript(content, offset)`, keeps only `user` and `assistant` roles, pairs them into turns, and writes a fresh offset back when done. There is also a fallback path in `extractClaudeCodeTurns(raw)` that uses `raw.prompt` and `raw.last_assistant_message` directly when the transcript path is unavailable.
- **OpenClaw** (`neuralscape-plugin/src/adapters/openclaw.ts:26-64`) is comparatively trivial: each `message:sent` event already contains exactly one user/assistant pair, so the adapter returns a single-element array.
- **Generic** (`neuralscape-plugin/src/adapters/generic.ts:17-47`) just type-checks fields and passes through; useful for `curl` testing the pipeline without a real client.

The asymmetry — Claude Code returning many turns from one event, OpenClaw returning one turn per event — is exactly what `TurnExtractor` accommodates by returning `ConversationTurn[]`.

## Hook lifecycle: Claude Code

A full Claude Code session runs through the plugin in two synchronous phases at the boundaries:

**1. SessionStart fires (sync, 30s budget).** Claude Code spawns `node scripts/session-start.js` and pipes a JSON payload — `session_id`, `cwd`, `hook_event_name`, etc. — to its stdin. The plugin parses stdin, derives `userId` from identity helpers in `utils.ts`, derives `projectId` from `cwd`, and issues `GET /v1/context/{project_id}` (or `/v1/context/global` if no project is detected). The response is grouped into the [category taxonomy](./03-memory-model.md); the plugin formats it according to a fixed `CATEGORY_ORDER` and truncates to roughly 8000 characters. Output goes back to Claude Code as a JSON object, `{continue: true, additionalContext: "# Neuralscape Memory ..."}`. Claude Code injects that block into the system context and the model now sees, before its first user turn, the user's preferences, the project's tech stack, prior decisions, and so on.

**2. The user works.** Claude Code maintains its own transcript file at `transcript_path`. Neuralscape doesn't touch this file during the session.

**3. Stop fires (async, 60s budget).** When the user stops the session, `node scripts/session-end.js` is launched in the background. The first thing the script does is print `{continue: true, suppressOutput: true}` to stdout so Claude Code is unblocked immediately — the rest happens asynchronously. The script then:

- Calls `detectClient(raw)`, which returns `"claude-code"` because `transcript_path` is present.
- Reads the transcript file and the `.neuralscape-offset` sibling.
- Parses out only the *new* user/assistant turns since the last run.
- For each turn, POSTs to `/v1/extensions/conversation-compiler/flush`. This kicks the [async pipeline](./07-async-pipeline.md): Gemini extraction → Qdrant + Neo4j write → vault dual-write → `memory_stored` event.
- Calls `commitClaudeCodeFlush(raw)` *after* `flushTurns` returns. The new offset that `extractClaudeCodeTurns` staged in memory is only persisted to `.neuralscape-offset` here, so a crash between extract and commit leaves the cursor at its prior position and the next session re-flushes from there.
- POSTs to `/v1/extensions/conversation-compiler/compile`, which groups the day's daily-log entries and asks Gemini to synthesize Sessions, Projects, Decisions, and Research articles.

The `.neuralscape-offset` file is the trick that makes the Stop hook idempotent across multiple stops in a single day. Splitting the offset write out of `extractClaudeCodeTurns` (v2 fix) is what makes that idempotency robust to mid-flush crashes.

## Hook lifecycle: OpenClaw

OpenClaw's lifecycle is finer-grained but simpler per event:

**1. `message:sent` fires (async, 15s budget) on every assistant turn.** OpenClaw runs `node scripts/conversation-turn.js`, which detects `"openclaw"` from the `{type, action}` shape, runs the OpenClaw adapter to produce a single `ConversationTurn`, and POSTs that one turn to `/flush`. Because there is no transcript file, there is no offset bookkeeping — every event is its own self-contained chunk.

**2. `session:end` fires (async, 15s budget).** This invokes the same `session-end.js` script Claude Code uses, but detection now returns `"openclaw"`, so the OpenClaw `extractOpenClawSessionEnd` runs instead. It produces a `SessionEndInput` with the date and `shouldCompile=true`, and the script POSTs to `/compile`.

The fact that `session-end.js` is a single shared script with adapter-driven branching is the cleanest evidence of how the plugin's normalization pays off: the host-specific divergence is contained inside `detect.ts` and the adapter files; the hook entry points themselves are nearly identical for both clients.

## Conversation Compiler extension

On the service side, everything is a [FastAPI extension](./02-service-architecture.md) mounted under `/v1/extensions/conversation-compiler/`. The relevant files:

| File | Purpose |
|------|---------|
| `routes.py:52-93` | `/flush` — enqueues to ARQ for async, or runs sync if fast-path enabled |
| `routes.py:95-122` | `/compile` — daily compilation request |
| `flush.py:130-267` | Gemini extraction; stores three ways (memory service, vault dual-write, daily log) |
| `compile.py:194-352` | Groups daily entries; Gemini synthesizes sessions/projects/decisions/research |
| `obsidian_writer.py:446-547` | `append_category_entry()`, `update_category_index()` |

`/flush` is where a normalized `ConversationTurn` becomes structured memory. `flush.py` runs Gemini against the user/assistant pair using prompts from [the extraction layer](./05-llm-extraction.md), yielding zero or more facts, each tagged with one of the 13 categories. Each fact is then:

1. **Stored in memory** via `MemoryService` — written to Qdrant (with category metadata) and ingested into the Graphiti temporal graph in Neo4j. See [storage backends](./06-storage-backends.md).
2. **Dual-written to the vault** via `ObsidianWriter.append_category_entry()` — the fact appears in `Semantic/Preferences/entries.md`, `Project/Tech-Stack/{project}.md`, etc., depending on category.
3. **Logged to the daily journal** at `Daily/{YYYY-MM-DD}.md` with a `compiled: false` marker.

`/compile` is the synthesis half. `compile.py` reads the daily log, groups uncompiled entries by topic and project, and asks Gemini to write four kinds of long-form artifacts:

- `Sessions/{YYYY-MM-DD}.md` — one summary per day
- `Projects/{project}/README.md` — the rolling compiled knowledge of each project
- `Decisions/{slug}.md` — one ADR-style record per significant decision
- `Research/{slug}.md` — articles for substantive research threads

After compilation, the daily log entries are flipped to `compiled: true` so the next compile pass skips them. This is where the [episodic-to-semantic distillation](./03-memory-model.md) actually happens at the file level: ephemeral conversation turns become permanent, structured Obsidian articles.

## Vault dual-write & the `memory_stored` event

The dual-write mechanism is what makes the Obsidian vault consistent with Qdrant + Neo4j regardless of how a memory entered the system. There are three ways memory gets stored — the REST API, the [MCP server](./08-mcp-server.md), or the plugin's `/flush` endpoint — and the vault must reflect all three.

The implementation is event-driven. Whenever `MemoryService` finishes a write, it emits a `memory_stored` event into the extension bus. The compiler extension subscribes via `_handle_memory_stored()` at `neuralscape-service/extensions/conversation_compiler/__init__.py:149-192`. The handler is short and load-bearing:

```python
async def _handle_memory_stored(self, payload: dict) -> Optional[dict]:
    # Skip if this memory was stored by the flush path (it already wrote to vault)
    if payload.get("source") == "conversation-compiler":
        return None
    content = payload.get("content", "")
    category = payload.get("category", "")
    ...
    cat_path = self.writer.append_category_entry(
        category=category, content=content,
        project_id=project_id, session_id=session_id, timestamp=ts,
    )
    self.writer.append_daily_log(date, [...])
```

The `source == "conversation-compiler"` guard is critical. The `/flush` path *already* writes to the vault before emitting the event, so without the guard each plugin-originated memory would be written twice — once by `flush.py` directly, once by the event handler. By tagging its own writes with `source="conversation-compiler"`, the flush path opts out of the secondary handler. API and MCP writes don't carry that source tag, so the handler picks them up and ensures they appear in the vault.

The category-to-folder mapping is canonical: `neuralscape-service/schemas.py:83-97` defines `CATEGORY_VAULT_PATHS`. The resulting vault layout:

```
vault/
├── Daily/{YYYY-MM-DD}.md                     timestamped entries
├── Sessions/{YYYY-MM-DD}.md                  synthesized session summary
├── Projects/{project}/README.md              compiled project knowledge
├── Decisions/{slug}.md                       synthesized decision records
├── Research/{slug}.md                        synthesized research articles
├── Semantic/Preferences/entries.md
├── Semantic/Personal-Facts/entries.md
├── Project/Tech-Stack/{project}.md           project-scoped per-file
├── Episodic/Decisions/entries.md
├── category-index.md                         rebuilt after each flush
└── index.md
```

The recent commit `f591781` added rebuilding `category-index.md` after each `memory_stored`-driven vault write so the index stays current as memories arrive.

The 13 categories are mirrored in two places that must stay in sync: `neuralscape-plugin/src/types.ts:25-147` on the TypeScript side, `neuralscape-service/schemas.py:47-97` on the Python side. The plugin needs the taxonomy because the SessionStart context formatter groups injected memories by category before sending them to Claude Code.

## Adding a new adapter

The contributor checklist for, say, a Cursor adapter is short because the contract is small:

1. **Create `src/adapters/cursor.ts`.** Implement `extractCursorTurns: TurnExtractor` and `extractCursorSessionEnd: SessionEndExtractor` matching the `ConversationTurn` and `SessionEndInput` shapes in `core/types.ts`.
2. **Update `src/adapters/detect.ts`.** Add `"cursor"` to the `ClientType` union, add a detection branch in `detectClient()` keyed on whatever Cursor-specific fields appear in stdin, and register the new extractors in the two switch statements at lines 26-48.
3. **Add a hook manifest** if Cursor's hook event names differ — for example `hooks/cursor-hooks.json` mirroring the OpenClaw one.
4. **Run `npm run build`** to bundle the updated `scripts/*.js`.
5. **Update `README.md`** with installation steps for the new client.

That is the entire surface. There is **no** server change required because the `/flush` and `/compile` endpoints already accept the normalized `ConversationTurn` shape; once `cursor.ts` produces that shape, every downstream component — Gemini extraction, Qdrant write, Neo4j ingestion, vault dual-write, daily log, compilation — works without modification. This is the core architectural payoff of the adapter pattern: the radius of change for a new client is bounded inside `neuralscape-plugin/src/adapters/`.

## Related

- [01-getting-started](./01-getting-started.md) — Step 8 walks through the marketplace install
- [03-memory-model](./03-memory-model.md) — the 13-category taxonomy that adapters serve and the vault organizes
- [02-service-architecture](./02-service-architecture.md) — how the conversation_compiler extension mounts into the FastAPI app
- [07-async-pipeline](./07-async-pipeline.md) — what happens after `/flush` returns 202
- [08-mcp-server](./08-mcp-server.md) — the seven MCP tools the plugin auto-wires via `.mcp.json`
- [05-llm-extraction](./05-llm-extraction.md) — the Gemini prompts used by `flush.py` to derive facts from turns
- [06-storage-backends](./06-storage-backends.md) — Qdrant and Neo4j, the shared sinks for all client adapters
