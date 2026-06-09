# Neuralscape Plugin

Persistent agentic memory for **Claude Code** and **Claude Cowork**. The plugin auto-captures your conversations, recalls relevant context on every session start, and exposes the 7 Neuralscape MCP tools — all backed by your own Neuralscape service (FastAPI + mem0 + Graphiti).

- **What you get:** memory injection on `SessionStart`, conversation flush + compile on `Stop`, **incremental tool-observation capture on `PostToolUse` + threshold-driven compile on `UserPromptSubmit`** (no extra API cost — runs on your subscription), five slash command skills (`status`, `search`, `sync`, `config`, `capture`), and the Neuralscape MCP toolkit auto-wired via `.mcp.json`.
- **Where it stores:** in your own Neuralscape deployment. The plugin never sends data anywhere else.
- **Cost:** zero additional. The plugin is a thin client over your service.

## 60-second install (Claude Code)

```text
/plugin marketplace add ehfazrezwan/neuralscape
/plugin install neuralscape@neuralscape-plugins
```

Claude Code prompts you for three values at install time:

| Prompt | Notes |
|---|---|
| Neuralscape service URL | e.g. `https://neuralscape.example.com` or `http://localhost:8199` |
| API key (optional) | bearer token if your deployment is authenticated; leave empty for local dev |
| Your user ID | a stable identifier so memories are scoped to you (e.g. your username) |

That's it. Open a new session — the SessionStart hook will pull your prior context and inject it as `additionalContext`.

> **First time on Neuralscape?** Set up the service first. The full 8-step walkthrough is at [`docs/neuralscape/01-getting-started.md`](../docs/neuralscape/01-getting-started.md) in this repo.

## Claude Cowork install

Cowork is **not** the same experience as Claude Code: it does not run plugin
hooks ([#27398](https://github.com/anthropics/claude-code/issues/27398)) and
its connector UI can't take a static Bearer token
([#112](https://github.com/anthropics/claude-ai-mcp/issues/112)). The supported
Cowork path is a **remote MCP OAuth connector** plus a standing-context memory
protocol — not the hook-driven marketplace flow.

➡️ **Full Cowork runbook: [`../COWORK.md`](../COWORK.md).** In short:

1. Admin: expose the service on a public HTTPS URL and set
   `NEURALSCAPE_PUBLIC_URL` + `NEURALSCAPE_USER_TOKEN_SECRET` (turns on the
   built-in OAuth server).
2. User: **Settings → Connectors → Add custom connector** → URL
   `https://<your-host>/mcp/` → **Connect** → paste your per-user token once.
3. Paste the [standing-context block](./cowork/STANDING_CONTEXT.md) into your
   Cowork workspace instructions so Claude recalls at task start and saves at
   task end via the MCP tools.

Installing the marketplace plugin in Cowork still loads the **skills** (useful),
but hooks won't fire — rely on the connector + standing context above.

## What gets installed

```
.claude/plugins/cache/neuralscape-plugins/neuralscape/2.1.0/
├── .claude-plugin/plugin.json    manifest with userConfig prompts
├── .mcp.json                      remote HTTP MCP at <URL>/mcp/
├── hooks/hooks.json               SessionStart, PostToolUse, UserPromptSubmit, Stop
├── skills/{status,search,sync,config,capture,compile-observations}/SKILL.md
├── scripts/                       built hook bundles
└── LICENSE / CHANGELOG.md
```

The plugin reaches your service via these calls:

| Trigger | Endpoint | Purpose |
|---|---|---|
| SessionStart | `GET /v1/context/{project_id}` or `/v1/context/global` | Fetch stored memories, format by category, inject as `additionalContext`. Also flags any pending observation buffers from prior sessions. |
| PostToolUse | (none — local file write) | Append `{tool, input, output, ts, project_id}` to per-session JSONL buffer. Filters read-only tools and trivial Bash commands; truncates large outputs. |
| UserPromptSubmit | (none — local check) | When the buffer crosses the threshold (default 25 obs or 30 min old), prepends an `additionalContext` instruction asking Claude to compile the buffer using the `compile-observations` skill before responding. |
| compile-observations skill | `POST /v1/memories/raw` (via `mcp__plugin_neuralscape_neuralscape__remember`) | Claude reads the buffer, applies the quality rubric, and submits one wiki-quality memory per significant work unit. Backend embeds and stores — **no Gemini call**. |
| Stop (per turn) | `POST /v1/extensions/conversation-compiler/flush` | Stream each user/assistant pair to Gemini extraction (legacy conversation-driven memory). |
| Stop (after flush) | `POST /v1/extensions/conversation-compiler/compile` | Synthesize the day's facts into Sessions/Decisions/Research articles. |
| Stop (after compile) | (none — local marker) | Drop a `.stale` marker next to any non-empty observation buffer so the next SessionStart compiles it even if no further user prompts were sent. |

### Where extraction happens

The PostToolUse path is **client-LLM-extracted**: the hook records raw observations to disk (no LLM, sub-50ms), and Claude Code's own LLM compiles them on the next user turn using your existing subscription tokens. The backend never runs Gemini for tool-driven capture — it just receives pre-categorized memories and stores them. See [`docs/neuralscape/05-llm-extraction.md`](../docs/neuralscape/05-llm-extraction.md) for the full diagram.

## Slash commands

Once installed, ask Claude any of:

- "Is neuralscape working?" → `/neuralscape:status`
- "What do I know about X?" → `/neuralscape:search`
- "Save this conversation to memory now" → `/neuralscape:sync`
- "What's my neuralscape config?" → `/neuralscape:config`
- "Compile my tool observations now" → `/neuralscape:capture`

Claude can also invoke them automatically when it judges them relevant.

## MCP tools (auto-wired)

Installing the plugin enables seven MCP tools backed by your Neuralscape service:

| Tool | Purpose |
|---|---|
| `recall_memories` | Hybrid vector + graph search across global + project scopes |
| `remember` | Store a single categorized fact (async write, returns task_id) |
| `remember_conversation` | Extract facts from a list of `{role, content}` messages |
| `get_project_context` | Load all memories for a project, organized by category |
| `search_knowledge_graph` | Graph-only structured search (entities, relationships, episodes) |
| `list_memories` | List with filters (scope, category, project_id) |
| `delete_memories` | Bulk delete by ID or filters |

These are reachable from any MCP client that picks up your plugin's `.mcp.json`. The transport is Streamable HTTP at `<URL>/mcp/`; the API key (if set) is sent as a Bearer token.

## OpenClaw integration (manual)

OpenClaw doesn't share Claude Code's marketplace. Install the OpenClaw hook manifest at `~/.openclaw/hooks/neuralscape/`:

1. Build the plugin from source (see **Development** below).
2. Copy `hooks/openclaw-hooks.json` and the `scripts/` directory into `~/.openclaw/hooks/neuralscape/`.
3. Set `NEURALSCAPE_URL`, `NEURALSCAPE_USER_ID`, and (optional) `NEURALSCAPE_API_KEY` in OpenClaw's hook env.
4. Restart OpenClaw — `message:sent` and `session:end` will start firing.

> **Note:** the OpenClaw hook manifest still uses `${PLUGIN_ROOT}` rather than `${CLAUDE_PLUGIN_ROOT}` because OpenClaw's expansion convention hasn't been confirmed against the Claude Code spec. If your OpenClaw build complains, swap to whichever variable it does export — the rest of the script is platform-agnostic.

## Configuration

The plugin reads from `userConfig` prompts (modern) or env vars (legacy fallback for one release):

| Setting | Modern (manifest) | Legacy fallback |
|---|---|---|
| Service URL | `CLAUDE_PLUGIN_OPTION_URL` | `NEURALSCAPE_URL` |
| API key | `CLAUDE_PLUGIN_OPTION_API_KEY` (sensitive — keychain-stored) | `NEURALSCAPE_API_KEY` |
| User ID | `CLAUDE_PLUGIN_OPTION_USER_ID` | `NEURALSCAPE_USER_ID` |
| Compile threshold (obs) | `CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD` (default `25`) | `NEURALSCAPE_COMPILE_THRESHOLD` |
| Compile age (minutes) | `CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN` (default `30`) | `NEURALSCAPE_COMPILE_AGE_MIN` |

To change settings after install:

```text
/plugin config neuralscape@neuralscape-plugins
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| SessionStart silently skips context | `USER_ID` not set | Run `/plugin config neuralscape@neuralscape-plugins` and fill in the `Your user ID` prompt |
| `/neuralscape:status` returns 503 | Vector store (Qdrant) unreachable | See `01-getting-started.md` Step 4 in the service docs |
| 202s become 200s on writes | Redis disconnected — plugin falls back to sync | Check `docker compose logs redis`; `/neuralscape:status` will report `redis: degraded` |
| `429` from Gemini in compile | Free-tier quota | Service auto-retries with `gemini-2.5-flash` fallback; tune `LLM_RETRY_MAX_DELAY` if it exhausts |
| Project memories not found | Old `group_id` format | Run `cypher-shell -u neo4j -p $NEO4J_PASSWORD < neuralscape-service/scripts/migrate-group-ids.cypher` once |
| Plugin not updating after `/plugin update` | Plugin cache stale | `/reload-plugins` or remove `~/.claude/plugins/cache/neuralscape-plugins/neuralscape/<old-version>/` |

For verbose diagnostics: ask Claude to run `/neuralscape:status` — it returns the resolved URL, user_id, API-key state, and a live `/health` probe of all three backends.

## Development

```bash
cd neuralscape-plugin
npm install        # also runs the postinstall build
npm run build      # rebuild bundles
npm run watch      # rebuild on save
```

The TypeScript source lives in `src/`. esbuild bundles five entry points (`session-start.ts`, `conversation-turn.ts`, `session-end.ts`, `post-tool-use.ts`, `user-prompt-submit.ts`) into `scripts/*.js` (ESM, Node 18+, minified). Built scripts are gitignored — `npm install` regenerates them.

To run a hook locally for testing:

```bash
echo '{"user_message":"hello","assistant_response":"Hi there. How can I help today?","session_id":"test","channel":"cli"}' | node scripts/conversation-turn.js
```

Architecture, adapter contracts, and "adding a new client" walkthrough live in [`docs/neuralscape/09-plugin-system.md`](../docs/neuralscape/09-plugin-system.md).

## License

MIT — see [`LICENSE`](./LICENSE).
