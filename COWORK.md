# Neuralscape in Claude Cowork

Neuralscape was built for Claude Code CLI, where four plugin lifecycle hooks
(`SessionStart`, `PostToolUse`, `UserPromptSubmit`, `Stop`) make memory
fully automatic. **Claude Cowork does not run plugin hooks**, so that
automatic loop does not fire there. This document is how you get a
first-class memory experience in Cowork anyway.

> **TL;DR**
> 1. Expose the service on a public HTTPS URL and set `NEURALSCAPE_PUBLIC_URL`
>    + `NEURALSCAPE_USER_TOKEN_SECRET`. This turns on a built-in OAuth server.
> 2. In Cowork, add Neuralscape as a **custom connector** → **Connect** →
>    paste your per-user token **once**. Done — Anthropic refreshes silently.
> 3. Paste the [standing-context block](#5-standing-context-the-no-hooks-memory-loop)
>    into your Cowork workspace instructions. It tells Claude to recall at the
>    start of a task and save at the end, via the MCP tools — the job the
>    hooks would otherwise do.

---

## Why Cowork is different (the three real limitations)

These are Anthropic-side limitations, confirmed by open issues — not bugs in
this plugin. Re-check them before assuming they still hold.

| # | What breaks | Issue | Consequence |
|---|---|---|---|
| 1 | Plugin hooks never fire in Cowork (the VM launches the CLI with `--setting-sources user`, which excludes plugin-scoped hooks). | [#27398](https://github.com/anthropics/claude-code/issues/27398) | No auto-injection, no auto-capture, no PostToolUse observation buffering. |
| 2 | The custom-connector UI only supports OAuth — there is **no field for a static Bearer token or custom headers**. | [#112](https://github.com/anthropics/claude-ai-mcp/issues/112) | You can't paste a token into the connector. OAuth is the only first-class auth. |
| 3 | Plugin `userConfig` values aren't reliably prompted on install in the desktop UI. | [#39455](https://github.com/anthropics/claude-code/issues/39455) / [#39827](https://github.com/anthropics/claude-code/issues/39827) | The marketplace-plugin config path (URL/token) is unreliable in Cowork. |

The takeaway that drove this design: **the one thing Cowork reliably does is
connect out to a public MCP server over HTTP and speak OAuth.** So we lean on
exactly that and nothing else. We do **not** rely on hooks firing, on
`userConfig` prompting, on managed settings, or on a shared-folder bridge.

> **Note on the earlier "bridge folder + host daemon" proposal.** That design
> assumed the Cowork VM could not do its own network I/O and needed a host
> daemon to forward memory over a shared folder. It doesn't: the Cowork agent
> reaches this service directly through the HTTP MCP connector. The bridge
> would re-implement, less reliably, what one MCP call already does — so it
> was dropped. If Anthropic ever network-isolates connector traffic from the
> VM, revisit it; today it's unnecessary.

---

## What works in Cowork, and what degrades

| Capability | Claude Code CLI | Claude Cowork |
|---|---|---|
| 8 MCP tools (recall/remember/search/projects/…) | ✅ hooks + MCP | ✅ MCP connector |
| Recall context at task start | ✅ automatic (`SessionStart`) | ✅ **instructed / on-demand** (standing context or `/neuralscape:recall` → `get_project_context`/`recall_memories`) |
| Save memories at task end | ✅ automatic (`Stop`) | ✅ **instructed / on-demand** (standing context or `/neuralscape:save-session` → `remember`/`remember_conversation`) |
| Pick a project (no working dir in Cowork) | ✅ from `cwd` | ✅ `/neuralscape:project` → `list_projects` |
| Passive per-tool observation buffering | ✅ automatic (`PostToolUse`) | ⚠️ **not available** — no hook can observe tool use in Cowork. Degrades to explicit "checkpoint" saves. *(No design can do this in Cowork; the hook is the only thing that can passively observe tool calls.)* |
| Onboarding | `userConfig` prompts / env vars | **OAuth connector** (this doc) |

"Instructed" means it rides Claude following the standing-context system
prompt rather than a deterministic hook. The MCP tool descriptions already
nudge it (`recall_memories` says *"ALWAYS call this tool before starting
work"*), and the standing-context block reinforces it.

---

## 1. Server prerequisites (admin, one-time)

The OAuth connector path requires two things on the service:

1. **A public HTTPS URL.** The simplest route is the bundled cloudflared
   tunnel profile:
   ```bash
   # in .env
   CLOUDFLARE_TUNNEL_TOKEN=<token from Cloudflare Zero Trust → Tunnels>
   docker compose --profile tunnel up -d cloudflared
   ```
   Point the tunnel's public hostname at `neuralscape:8199`. Anthropic
   connects to your MCP server **from Anthropic's cloud**, so the URL must be
   reachable from the public internet (not just your LAN).

2. **OAuth turned on**, by setting both:
   ```bash
   # in .env
   NEURALSCAPE_PUBLIC_URL=https://neuralscape.example.com   # your tunnel hostname, no trailing slash
   NEURALSCAPE_USER_TOKEN_SECRET=<your 32-byte base64 signing secret>
   ```
   With both set, the service serves OAuth discovery + authorization
   endpoints. With either missing, those endpoints return 404 and the service
   behaves exactly as before (safe for local dev / Claude Code CLI).

3. **Issue a per-user token** for each team member (this is what they paste
   once during Connect):
   ```bash
   docker exec neuralscape-neuralscape-1 \
     python scripts/issue_user_token.py -u alice --days 365
   ```

Verify OAuth is live:
```bash
curl https://neuralscape.example.com/.well-known/oauth-protected-resource
# → {"resource": ".../mcp", "authorization_servers": ["https://neuralscape.example.com"], ...}
```

---

## 2. Add the connector (each user, one-time)

In Claude (Cowork / claude.ai):

1. **Settings → Connectors → Add custom connector.**
2. **Remote MCP server URL:** `https://neuralscape.example.com/mcp/`
3. Save, then click **Connect**.
4. Claude discovers the Authorization Server and opens the Neuralscape consent
   page. **Paste the per-user token your admin issued you** and click
   **Authorize**.
5. That's it. Anthropic now holds short-lived OAuth tokens and refreshes them
   silently — you never paste anything again.

After connecting, the 8 Neuralscape MCP tools appear in Cowork. The server
reads your identity from the OAuth token, so the `user_id` you pass is
ignored over the connector. A few tools mark `user_id` as required in their
schema (`recall_memories`, `remember`, `remember_conversation`) — pass a
harmless placeholder like `"cowork"` when one does; the token still scopes
memory to you. (A real `user_id` argument is honored for local/stdio use.)

### How the OAuth flow works (for the curious / for debugging)

The service is its own OAuth 2.1 Authorization Server (`oauth.py`), built on
the existing HMAC token machinery — there is **no separate user database**.

```
Cowork ──GET /mcp/ (no token)──▶ 401 + WWW-Authenticate: resource_metadata=…
       ──fetch .well-known/oauth-protected-resource──▶ points at the AS
       ──fetch .well-known/oauth-authorization-server──▶ endpoints + PKCE S256
       ──POST /oauth/register (DCR)──▶ signed client_id (stateless)
       ──GET  /oauth/authorize──▶ consent page  ← user pastes their token ONCE
       ──(token verified → user_id)──▶ 303 redirect with authorization code
       ──POST /oauth/token (code + PKCE)──▶ access_token (= a per-user HMAC
                                            token, typ="access") + refresh_token
       ──GET /mcp/ (Bearer access_token)──▶ identity flows to every tool call
```

The OAuth **access token is the same HMAC token format** the service already
validates, distinguished by a `typ` claim so that authorization codes and
refresh tokens can never be replayed as a Bearer credential.

---

## 3. Org / Team rollout (admin)

For a whole team, do the per-user steps above once and standardize the rest:

- **Connect the marketplace** under *Organization settings → Plugins* (GitHub-
  synced to this repo). Force-enabling the plugin reliably loads the **skills**
  in Cowork even though hooks won't fire. The cross-platform skills do the work
  the hooks would: `/neuralscape:recall`, `/neuralscape:remember`,
  `/neuralscape:save-session`, `/neuralscape:project`, plus
  `/neuralscape:search`, `/neuralscape:ns-status`, `/neuralscape:ns-config`.
  With those installed, the standing-context paste-block becomes optional (it
  remains the fallback for workspaces without the plugin).
- **Distribute the connector URL** and each member's per-user token (e.g. via
  your secrets manager). Each member adds the connector + Connects once.
- **Ship the standing context** (below) as the default workspace instruction
  for shared Cowork workspaces, so capture/recall behavior is uniform.

---

## 4. Bootup word

Give the team a single memorable trigger. Recommended: **"neuralscape on"** at
the start of a session, and **"remember this"** whenever something is worth
saving. The standing context wires both to MCP calls. (With the standing
context in place, recall also happens automatically at the start of any task —
the bootup word is just an explicit nudge for when you want to be sure.)

---

## 5. Standing context: the no-hooks memory loop

Paste this block into your Cowork **workspace instructions** (or project
instructions). It is the system-prompt equivalent of the Claude Code hooks —
the determinism here rides on Claude following these instructions, so keep
them short and imperative. A ready-to-copy copy lives at
[`neuralscape-plugin/cowork/STANDING_CONTEXT.md`](./neuralscape-plugin/cowork/STANDING_CONTEXT.md).

```markdown
## Neuralscape memory protocol (Cowork)

You have a persistent memory layer via the Neuralscape MCP connector. There are
no automatic hooks here, so YOU must drive memory explicitly:

- **At the start of any task** (and whenever the user says "neuralscape on"):
  call `recall_memories` (or `get_project_context` for a known project) to load
  the user's preferences, conventions, tech stack, and past decisions BEFORE
  you plan or act. Treat what you recall as authoritative context.

- **When you learn something durable** — a preference, a decision, a project
  convention, a gotcha, an architectural choice — call `remember` with a clear
  one-sentence fact and the right category. Do this as it happens; don't wait.

- **When the user says "remember this" / "save this"**, or at the end of a
  substantive working session, call `remember_conversation` with the relevant
  messages so the service can extract and store the facts.

- **Identity is from the connector token, not `user_id`.** The server scopes
  memory to whoever authenticated the connector and ignores any `user_id` you
  pass. Some tools (`recall_memories`, `remember`, `remember_conversation`)
  mark `user_id` as required — when one does, pass a harmless placeholder like
  `"cowork"` to satisfy the schema; the token still determines your real
  identity. Never block on not knowing the `user_id`.

- Skip trivia (routine file reads, searches, acknowledgements). Save signal,
  not noise.
```

---

## 6. Config precedence & the legacy fallback (keep it)

Outside Cowork, config still flows through the plugin's `readConfig()` with
this precedence (`neuralscape-plugin/src/utils.ts`):

1. `CLAUDE_PLUGIN_OPTION_*` (modern `userConfig` — Claude Code CLI)
2. `NEURALSCAPE_*` env vars (`NEURALSCAPE_URL`, `NEURALSCAPE_USER_ID`, `NEURALSCAPE_API_KEY`)
3. defaults

**The `NEURALSCAPE_*` legacy fallback is load-bearing and stays permanently.**
Every non-Cowork config path depends on it, and `userConfig` prompting is
unreliable in Cowork (#39455). Claude Code CLI behavior is unchanged by
anything in this document — both platforms work: CLI via hooks, Cowork via the
OAuth connector + standing context.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Connector "Connect" button does nothing / no login page | OAuth not enabled or URL not public | Confirm `NEURALSCAPE_PUBLIC_URL` **and** `NEURALSCAPE_USER_TOKEN_SECRET` are set; `curl …/.well-known/oauth-protected-resource` should return JSON, not 404. |
| `401` after connecting | Pasted token invalid/expired, or wrong signing secret | Re-issue the user token; confirm the service's `NEURALSCAPE_USER_TOKEN_SECRET` matches the one tokens were signed with. |
| Tools appear but never get called | No standing context | Paste the [standing-context block](#5-standing-context-the-no-hooks-memory-loop) into workspace instructions. |
| `WWW-Authenticate` header missing on 401 | `NEURALSCAPE_PUBLIC_URL` unset | Set it; that header is what triggers Cowork's OAuth discovery. |
| Memories saved under the wrong user | A `user_id` was passed and an old build trusted it | Update to this build — token identity is now authoritative over arguments. |

---

## References

- #27398 — Plugin hooks don't fire in Cowork: https://github.com/anthropics/claude-code/issues/27398
- #40495 — Settings sources silently ignored in the Cowork sandbox: https://github.com/anthropics/claude-code/issues/40495
- #39455 / #39827 — `userConfig` not prompted on install: https://github.com/anthropics/claude-code/issues/39455
- #112 — Custom connector UI has no Bearer/header field, OAuth only: https://github.com/anthropics/claude-ai-mcp/issues/112
- Custom connectors via remote MCP: https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp
- Manage Cowork plugins for your org: https://support.claude.com/en/articles/13837433-manage-claude-cowork-plugins-for-your-organization
