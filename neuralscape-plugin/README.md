# Neuralscape Plugin

Persistent agentic memory for **Claude Code** and **Claude Cowork**, backed by your own Neuralscape service (FastAPI + mem0 + Graphiti). In Claude Code, plugin hooks auto-capture your conversations and recall relevant context on every session start; in Claude Cowork (which doesn't run plugin hooks), the cross-platform skills drive the same recall/capture loop on demand through the MCP connector. Exposes the 8 Neuralscape MCP tools.

- **What you get:** **progressive-disclosure memory injection** on `SessionStart` (a budget-bounded, day-grouped index of your memories — the map, not the payloads — plus your identity card and a "Previously…" block from the last session), a **File Read Gate** on `PreToolUse(Read)` (large files that already have memories get a ranked per-file memory timeline injected *alongside* the read — the read always proceeds), conversation flush + compile on `Stop`, a **structured session summary** stored via one `checkpoint` call on `SessionEnd`, **incremental tool-observation capture on `PostToolUse` + threshold-driven compile on `UserPromptSubmit`** (no extra API cost — runs on your subscription), cross-platform MCP-driven skills (`recall`, `remember`, `save-session`, `project`, `search`, `ns-status`, `ns-config`) plus Claude-Code capture skills (`sync`, `capture`), and the Neuralscape MCP toolkit auto-wired via `.mcp.json`.
- **Where it stores:** in your own Neuralscape deployment. The plugin never sends data anywhere else.
- **Cost:** zero additional. The plugin is a thin client over your service.

## 60-second install (Claude Code)

```text
/plugin marketplace add ehfazrezwan/neuralscape
/plugin install neuralscape@neuralscape-plugins
```

The MCP connector URL is **baked into the plugin** per distribution channel
(see **Distribution & self-hosting** below), and auth is handled by **OAuth** —
so you don't enter a URL or API key. On first connect, Claude Code runs the
OAuth flow; paste your per-user token once on the consent page.

`userConfig` (URL / USER_ID) now only configures the Claude Code **hooks**, and
can be left at defaults — the hooks derive the same baked URL from `.mcp.json`.
Set `USER_ID` if you want hook-captured memories scoped to a specific id; set
`URL` only to point the hooks at a different host than the baked connector.

Open a new session — the SessionStart hook pulls your prior context and injects
it as `additionalContext`.

> **First time on Neuralscape?** Set up the service first. The full 8-step walkthrough is at [`docs/neuralscape/01-getting-started.md`](../docs/neuralscape/01-getting-started.md) in this repo.

## Claude Cowork install

Cowork is **not** the same experience as Claude Code: it does not run plugin
hooks ([#27398](https://github.com/anthropics/claude-code/issues/27398)). The
memory loop is driven by the **cross-platform skills + the bundled OAuth MCP
connector** instead. Installing the marketplace plugin in Cowork is enough —
the connector ships **pre-configured** with the channel's baked URL (Cowork
bundles `.mcp.json` read-only), so there is **no manual URL entry**.

➡️ **Full Cowork runbook: [`../COWORK.md`](../COWORK.md).** In short:

1. Admin (self-host): expose the service on a public HTTPS URL and set
   `NEURALSCAPE_PUBLIC_URL` + `NEURALSCAPE_USER_TOKEN_SECRET` (turns on the
   built-in OAuth server). See **Distribution & self-hosting** below to bake
   your URL into the plugin.
2. User: install the plugin from your channel's marketplace → open the bundled
   **Neuralscape** connector → **Connect** → paste your per-user token once on
   the OAuth consent page. (No key, no URL to type.)
3. Optional: paste the [standing-context block](./cowork/STANDING_CONTEXT.md)
   into your Cowork workspace instructions so Claude recalls at task start and
   saves at task end via the MCP tools.

Hooks won't fire in Cowork — it doesn't run plugin hooks
([#27398](https://github.com/anthropics/claude-code/issues/27398)), so the Hooks
panel lists them but they're inert. The skills + connector do that work instead.

## Distribution & self-hosting (baked URL)

Neuralscape is **self-hosted** — every deployment has its own URL. Cowork bundles
the MCP connector from `.mcp.json` **read-only** and can't interpolate
`${user_config.*}`, so a per-user runtime URL is impossible for a bundled
connector. The model instead is: **the connector URL is a literal baked into
`.mcp.json` per distribution channel.** A channel = a repo/marketplace whose
plugin has its URL baked in. Three ways to distribute:

| Channel | Who | How |
|---|---|---|
| **Official** | Hosted Neuralscape | Install from `ehfazrezwan/neuralscape`; URL is the official endpoint. |
| **Self-host fork** | You run the service | Fork this repo, bake your URL, publish the fork as your marketplace. |
| **Vendor marketplace** | You already run a Claude marketplace | Vendor the `neuralscape-plugin/` dir into your marketplace repo with your URL baked in. |

**Bake your channel** (rewrites the literal URL in `.mcp.json`; the hooks derive
their base from the same file, so it's the single source of truth):

```bash
cd neuralscape-plugin
npm run bake -- --url https://your-host          # → https://your-host/mcp/
npm run bake -- --url https://your-host --dry-run # preview only
# publishing your own marketplace? also stamp its identity:
npm run bake -- --url https://your-host --marketplace-name acme-plugins --owner "Acme"
```

Then commit the result to your channel's repo (marketplace installs pull raw
from git — there's no build step, so the baked value must be committed).

**Enable OAuth on your service** (required; the connector is OAuth-only, no API
key). It's a thin wrapper over the token you already issue — no user database,
no OAuth app registration:

1. Set `NEURALSCAPE_PUBLIC_URL` (your public https base URL — turns OAuth on).
2. Set `NEURALSCAPE_USER_TOKEN_SECRET` (the HMAC signing secret you already use).
3. Expose the service over public HTTPS.

Users then connect once and paste their admin-issued token on the consent page;
dynamic client registration and PKCE are automatic.

## What gets installed

```
.claude/plugins/cache/neuralscape-plugins/neuralscape/2.4.0/
├── .claude-plugin/plugin.json    manifest with userConfig (hooks) prompts
├── .mcp.json                      bundled OAuth MCP connector at the baked <URL>/mcp/
├── hooks/hooks.json               SessionStart, PreToolUse (Read), PostToolUse, UserPromptSubmit, Stop, SessionEnd
├── skills/{recall,remember,save-session,project,search,ns-status,ns-config,sync,capture,compile-observations}/SKILL.md
├── scripts/                       built hook bundles
└── LICENSE / CHANGELOG.md
```

The plugin reaches your service via these calls:

| Trigger | Endpoint | Purpose |
|---|---|---|
| SessionStart | `GET /v1/context/{project_id}` or `/v1/context/global` (+ `GET /v1/extensions/dreaming/card`, `GET /v1/code-graph/query` probe) | **Index mode (default):** inject a day-grouped, budget-bounded memory *index* (`#id \| time \| type \| title \| ~tokens`) with a savings header, the identity card(s) when the dreaming sweep has built them, a "Previously…" block from the last session note, and an escalation footer teaching `recall_memories(index_only=true)` → `get_memories(ids=[...])` → `timeline`. `CONTEXT_MODE=full` restores legacy full-content injection. Also flags any pending observation buffers from prior sessions. |
| PreToolUse (Read) | `GET /v1/memories?fields=index` (newest-first, project-scoped, once per session) | **File Read Gate (steering):** when the target file is larger than `READ_GATE_MIN_BYTES` and stored memories reference it (path-tail match, at least `dir/basename`), the hook injects `additionalContext` with a ranked per-file memory timeline (`#id \| when \| title \| ~tokens`) plus an escalation menu (`get_memories` → `timeline`) — **the Read always proceeds**. The index fetch happens at most once per session (cached), under a hard time budget; steering fires at most once per file per session. |
| PostToolUse | (none — local file write) | Append `{tool, input, output, ts, project_id}` to per-session JSONL buffer. Filters read-only tools, harness plumbing (`SKIP_TOOLS`), and trivial Bash commands; truncates large outputs. |
| UserPromptSubmit | (none — local check) | When the buffer crosses the threshold (default 25 obs or 30 min old), prepends an `additionalContext` instruction asking Claude to compile the buffer using the `compile-observations` skill before responding. |
| compile-observations skill | `POST /v1/memories/raw` (via `mcp__plugin_neuralscape_neuralscape__remember`) | Claude reads the buffer, applies the quality rubric, and submits one wiki-quality memory per significant work unit. Backend embeds and stores — **no Gemini call**. |
| Stop (per turn) | `POST /v1/extensions/conversation-compiler/flush` | Stream each user/assistant pair to Gemini extraction (legacy conversation-driven memory). |
| Stop (after flush) | `POST /v1/extensions/conversation-compiler/compile` | Synthesize the day's facts into Sessions/Decisions/Research articles. |
| Stop (after compile) | (none — local marker) | Drop a `.stale` marker next to any non-empty observation buffer so the next SessionStart compiles it even if no further user prompts were sent. |
| SessionEnd | `POST /v1/checkpoint` | Distill the whole session (transcript + observation buffer, deterministic — no LLM) into a structured `{request, investigated, learned, completed, next_steps}` note and store it as ONE checkpoint `session_note`. The next SessionStart renders it as the "Previously…" block, next steps first. Fires once per session (clear/logout/exit), not per turn. |

### Where extraction happens

The PostToolUse path is **client-LLM-extracted**: the hook records raw observations to disk (no LLM, sub-50ms), and Claude Code's own LLM compiles them on the next user turn using your existing subscription tokens. The backend never runs Gemini for tool-driven capture — it just receives pre-categorized memories and stores them. See [`docs/neuralscape/05-llm-extraction.md`](../docs/neuralscape/05-llm-extraction.md) for the full diagram.

## Slash commands

Once installed, ask Claude any of:

- "Load my context / neuralscape on" → `/neuralscape:recall` *(both platforms)*
- "Remember that I prefer X" → `/neuralscape:remember` *(both platforms)*
- "Save this session to memory" → `/neuralscape:save-session` *(both platforms)*
- "Switch project / what projects do I have?" → `/neuralscape:project` *(both platforms)*
- "What do I know about X?" → `/neuralscape:search` *(both platforms)*
- "Is neuralscape working?" → `/neuralscape:ns-status`
- "What's my neuralscape config?" → `/neuralscape:ns-config`
- "Save this conversation to memory now" → `/neuralscape:sync` *(delegates to save-session in Cowork)*
- "Compile my tool observations now" → `/neuralscape:capture` *(Claude Code only)*

Claude can also invoke them automatically when it judges them relevant.

### Skills in Cowork

Claude Cowork loads plugin **skills** but runs no hooks. The MCP-driven skills above (`recall`, `remember`, `save-session`, `project`, `search`, `ns-status`, `ns-config`) work in Cowork via the MCP connector; the hook-fed skills (`capture`, and `sync`'s HTTP path) detect the missing buffer/URL and redirect you to the MCP equivalents instead of erroring. See [`../COWORK.md`](../COWORK.md) for connector setup.

## MCP tools (auto-wired)

Installing the plugin enables eight MCP tools backed by your Neuralscape service:

| Tool | Purpose |
|---|---|
| `recall_memories` | Hybrid vector + graph search across global + project scopes |
| `remember` | Store a single categorized fact (async write, returns task_id) |
| `remember_conversation` | Extract facts from a list of `{role, content}` messages |
| `get_project_context` | Load all memories for a project, organized by category |
| `search_knowledge_graph` | Graph-only structured search (entities, relationships, episodes) |
| `list_memories` | List with filters (scope, category, project_id) |
| `list_projects` | List the caller's distinct project_ids (powers the `project` skill) |
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
|  | *Default when unset:* the hooks fall back to the **OS username** (`$USER` / `$USERNAME`) so zero-config local installs keep a stable identity. Set it explicitly on shared machines or if your OS username isn't the id you want memories filed under. Servers with token auth derive identity from the Bearer token and ignore client-claimed ids. | |
| Compile threshold (obs) | `CLAUDE_PLUGIN_OPTION_COMPILE_THRESHOLD` (default `25`) | `NEURALSCAPE_COMPILE_THRESHOLD` |
| Compile age (minutes) | `CLAUDE_PLUGIN_OPTION_COMPILE_AGE_MIN` (default `30`) | `NEURALSCAPE_COMPILE_AGE_MIN` |
| SessionStart context mode | `CLAUDE_PLUGIN_OPTION_CONTEXT_MODE` (`index` default \| `full` legacy) | `NEURALSCAPE_CONTEXT_MODE` |
| Index budget (tokens) | `CLAUDE_PLUGIN_OPTION_INDEX_BUDGET_TOKENS` (default `1500`) | `NEURALSCAPE_INDEX_BUDGET_TOKENS` |
| Code-graph deferral | `CLAUDE_PLUGIN_OPTION_CODE_GRAPH` (`auto` default \| `on` \| `off`) | `NEURALSCAPE_CODE_GRAPH` |
| File read gate | `CLAUDE_PLUGIN_OPTION_READ_GATE_ENABLED` (default `true`) | `NEURALSCAPE_READ_GATE_ENABLED` |
| Read gate min size (bytes) | `CLAUDE_PLUGIN_OPTION_READ_GATE_MIN_BYTES` (default `1500`) | `NEURALSCAPE_READ_GATE_MIN_BYTES` |
| Read gate fetch budget (ms) | `CLAUDE_PLUGIN_OPTION_READ_GATE_TIME_BUDGET_MS` (default `2000`) | `NEURALSCAPE_READ_GATE_TIME_BUDGET_MS` |
| Excluded projects (globs) | `CLAUDE_PLUGIN_OPTION_EXCLUDED_PROJECTS` (comma-separated, `*`/`?` wildcards) | `NEURALSCAPE_EXCLUDED_PROJECTS` (or `NS_EXCLUDED_PROJECTS`) |
| Extra skip-tools (capture) | `CLAUDE_PLUGIN_OPTION_SKIP_TOOLS` (comma-separated, additive) | `NEURALSCAPE_SKIP_TOOLS` |
| Fail-loud threshold | `CLAUDE_PLUGIN_OPTION_FAIL_LOUD_THRESHOLD` (default `3`) | `NEURALSCAPE_FAIL_LOUD_THRESHOLD` |

To change settings after install:

```text
/plugin config neuralscape@neuralscape-plugins
```

### Pinning the project id (monorepos)

Memories are scoped to a `project_id`. By default the plugin resolves it
deterministically so the **same repo always reports the same id** no matter
which subdirectory a command runs in (important for monorepos whose service
and plugin live in sibling folders — otherwise the id flips between folder
basenames and memories fragment). Resolution precedence:

1. **`PROJECT_ID` env override** — `CLAUDE_PLUGIN_OPTION_PROJECT_ID` /
   `NEURALSCAPE_PROJECT_ID`. Pins one id everywhere this shell runs. Use
   sparingly — it applies globally, so it's the wrong tool if you work across
   multiple repos.
2. **A `.neuralscape-project` marker file** at the repo root (walked up from
   the cwd). Its first line is the id; an empty marker falls back to the
   marker directory's basename. **This is the recommended mechanism** — commit
   it so every contributor and every subdirectory agree:
   ```text
   echo neuralscape > .neuralscape-project
   ```
3. **The git repo root basename** (walk up for `.git`).
4. **The working-directory basename** (legacy fallback).

## File Read Gate (PreToolUse)

When Claude is about to `Read` a file **larger than `READ_GATE_MIN_BYTES`
(default 1500)** that Neuralscape already holds memories about, the gate
**steers**: it injects `additionalContext` with a ranked per-file memory
timeline alongside the read — **the Read itself always proceeds** (it is
never denied, and memories are never substituted for real file contents):

```text
[Neuralscape] 3 stored memories reference `src/worker.ts` — prior context that may complement the file you are reading:

`#id | when | title | ~tokens`
#a1b2… | 2d | 🐛 Fixed queue-starvation race in worker.ts | ~120
#c3d4… | 5d | 🔍 worker.ts drains the retry queue before shutdown | ~90

Details: get_memories with ids for full payloads; timeline for surrounding history; …
```

Behavior details:

- **Ranking:** memories that *modified* the file (`observation_type` in
  bugfix/feature/refactor, or modification verbs in the content) outrank ones
  that merely read it; memories mentioning fewer files rank higher
  (specificity), deeper path-tail matches higher still; capped at 10 rows.
- **File-reference signal:** NS memories carry no structured
  `files_read`/`files_modified` metadata — paths survive only in memory
  content/title/tags. The gate matches **path tails** (at least
  `dir/basename`, e.g. `src/worker.ts`; a bare basename is only used for
  single-segment paths) so same-named files elsewhere in the repo don't
  produce false references.
- **Hot-path cost:** the index fetch (`GET /v1/memories?fields=index`,
  newest 150 project-scoped index rows — no content payloads) happens **at
  most once per session** and is cached; later Reads match in-process. The
  one fetch runs under a hard time budget (`READ_GATE_TIME_BUDGET_MS`,
  default 2000 ms) — on timeout or error the hook exits 0 and stays quiet
  for the rest of the session.
- **Never in your way:** steering fires at most **once per file per
  session**. Small files, binary/media extensions, excluded projects, and an
  unreachable service all bypass the gate entirely (allow, exit 0).
- Disable with `READ_GATE_ENABLED=false`; tune the size floor with
  `READ_GATE_MIN_BYTES` and the fetch budget with `READ_GATE_TIME_BUDGET_MS`.

## Excluded projects

Set `EXCLUDED_PROJECTS` to comma-separated project-id globs (`*` and `?`
wildcards, case-insensitive — e.g. `scratch-*,client-secret`) and every
capturing hook honors it: PostToolUse observation capture, Stop conversation
flush, the SessionEnd session note, SessionStart context injection, and the
read gate all skip matching projects entirely. `NS_EXCLUDED_PROJECTS` works
as an env-var alias.

## Privacy: `<private>` tags

Wrap anything in `<private>…</private>` and the plugin will never store or
transmit it: the spans are replaced with `[redacted]` **before** observation
rows hit disk, before conversation turns are flushed to the compiler, and
before session-note fields go into a checkpoint. Redaction is case-insensitive
and fail-closed — an unclosed `<private>` redacts to the end of the text. The
compile-observations skill additionally skips any work unit containing
`<private>` content entirely.

## Hook failure taxonomy (never-block contract)

The hooks distinguish *whose* fault a failure is, and encode it in the exit
code:

| Failure | Behavior | Exit code |
|---|---|---|
| Neuralscape unreachable / transport error | Continue without memory. SessionStart injects a one-line `[neuralscape] memory service unreachable` notice; other hooks log to stderr; the read gate allows the read. The session is NEVER blocked. | `0` |
| Neuralscape unreachable for `FAIL_LOUD_THRESHOLD` (default 3) **consecutive** events | Fail loud instead of staying quietly degraded: the next SessionStart notice names the streak — `unreachable for N consecutive events — check docker compose ps`. Any successful call resets the counter. | `0` |
| Malformed hook stdin (not valid JSON — a client bug) | Fail loud so the bug is visible instead of silently swallowed. | `2` on hooks where exit 2 cannot block (`SessionStart`, `SessionEnd`, `PostToolUse`) |
| Malformed stdin on `Stop` / `UserPromptSubmit` / `PreToolUse` | Exit 2 has *blocking* semantics on these events (it would force Claude to continue, erase the user's prompt, or deny the tool call), so the never-block principle wins: log to stderr, exit `0`. | `0` |
| Hook-internal error (our bug) | Logged to stderr, session continues. | `0` |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| SessionStart silently skips context | `USER_ID` not set | Run `/plugin config neuralscape@neuralscape-plugins` and fill in the `Your user ID` prompt |
| `/neuralscape:ns-status` returns 503 | Vector store (Qdrant) unreachable | See `01-getting-started.md` Step 4 in the service docs |
| 202s become 200s on writes | Redis disconnected — plugin falls back to sync | Check `docker compose logs redis`; `/neuralscape:ns-status` will report `redis: degraded` |
| `429` from Gemini in compile | Free-tier quota | Service auto-retries with `gemini-2.5-flash` fallback; tune `LLM_RETRY_MAX_DELAY` if it exhausts |
| Project memories not found | Old `group_id` format | Run `cypher-shell -u neo4j -p $NEO4J_PASSWORD < neuralscape-service/scripts/migrate-group-ids.cypher` once |
| Plugin not updating after `/plugin update` | Plugin cache stale | `/reload-plugins` or remove `~/.claude/plugins/cache/neuralscape-plugins/neuralscape/<old-version>/` |

For verbose diagnostics: ask Claude to run `/neuralscape:ns-status` — it returns the resolved URL, user_id, API-key state, and a live `/health` probe of all three backends.

## Development

```bash
cd neuralscape-plugin
npm install        # also runs the postinstall build
npm run build      # rebuild bundles
npm run watch      # rebuild on save
npm run package    # build a clean installable zip under dist/
```

The TypeScript source lives in `src/`. esbuild bundles seven entry points (`pre-tool-use.ts`, `session-start.ts`, `conversation-turn.ts`, `session-end.ts`, `session-summary.ts`, `post-tool-use.ts`, `user-prompt-submit.ts`) into `scripts/*.js` (ESM, Node 18+, minified). Built scripts are **committed** (marketplace installs pull raw from git) — rebuild with `npm run build` whenever `src/` changes.

Unit tests run with vitest: `npm test` (builds first, then runs the pure-logic suites plus subprocess tests that assert the exit-code taxonomy against the built bundles). Coverage: `npm run test:coverage`.

### Packaging for local install (Cowork / manual)

`npm run package` (alias `npm run dist`, source at `tools/package.mjs`) produces `dist/neuralscape-plugin-<version>.zip` — an installable artifact containing only the runtime files (manifest, `.mcp.json`, `hooks/`, built `scripts/`, `skills/`, `cowork/`, docs). It deliberately **excludes `node_modules/`, `src/`, and `tests/`** so the archive has zero symlinks; Cowork rejects any zip that contains one ("Zip file contains a symbolic link"). The script rebuilds the bundle first and fails loudly if a symlink ever slips into the archive. `dist/` is gitignored.

To run a hook locally for testing:

```bash
echo '{"user_message":"hello","assistant_response":"Hi there. How can I help today?","session_id":"test","channel":"cli"}' | node scripts/conversation-turn.js
```

Architecture, adapter contracts, and "adding a new client" walkthrough live in [`docs/neuralscape/09-plugin-system.md`](../docs/neuralscape/09-plugin-system.md).

## License

MIT — see [`LICENSE`](./LICENSE).
