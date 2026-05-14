# Neuralscape Web UI — PRD

> Brief for a frontend design agent. Lists *what* to build; design agent owns *how* it looks. Beautiful, modern, minimal, functional.

## Context

Neuralscape is a multi-user persistent memory layer for AI agents (mem0 + Graphiti). Today it's API + MCP + CLI only — no human UI. Now that v2.2 multi-user is shipped, humans need a way to:

- Audit what they and the team have remembered.
- Search the hybrid vector + knowledge-graph index visually.
- Manage their private pool and contribute deliberately to a shared team pool.
- Administer users, tokens, health, and migrations without dropping to CLI.

The UI does **not** replace the Claude Code plugin — agents continue to use MCP. This is for the *human* in the loop.

## Audience

- **End users** (every team member): browse, search, edit, share memories.
- **Admins** (role-gated): user management, token issuance, health, bulk migrations, secret rotation.

Single Neuralscape instance = single team. No multi-team isolation in v1.

## Constraints the design must respect

- **Writes are async.** `remember` and `remember_conversation` return a `task_id`; processing happens in an ARQ worker. The UI must show task state (queued / processing / completed / failed) and never pretend a write is finished when it isn't.
- **Reads are sync.** Search, list, graph queries return immediately.
- **Auth is per-user HMAC tokens.** Users paste a token issued by an admin via `scripts/issue_user_token.py`. No password flow, no SSO in v1. The token carries `user_id`; the server enforces it.
- **Visibility model is load-bearing.** Every memory is `private` (owner only) or `shared` (whole team). Defaults are per-category. The UI must make visibility obvious on every memory and offer an explicit toggle.
- **Owner identity matters on shared memories.** Search results carry `owner_user_id`; show it.
- **Destructive ops on shared content are dangerous.** Bulk delete defaults to private-only; opt-in to remove shared writes. Token revocation invalidates all tokens. Secret rotation is a one-way door. Every such action needs strong confirmation.

## Features the UI must support

### Memory operations
- Browse all memories with filters: visibility (private / shared / both), category (13 values), scope (global / project), project, owner, date range, expires-within.
- Filter by memory-model v2 fields: domain, observation_type, concepts, source_type, confidence range.
- View a single memory's full state including all v2 fields, provenance, related memories, and the knowledge-graph edges it produced.
- Create a memory manually (content + category + optional v2 fields + visibility override).
- Edit content, category, tags, visibility, expires_at.
- Multi-select for bulk: delete, change visibility, change category.
- Single delete and bulk delete (with the private-only default behavior surfaced).

### Search
- Semantic search across vector + graph, merged by relevance.
- Toggle: vector-only / graph-only / both.
- Filters: visibility, include_shared, project, categories, limit.
- Each result shows score, source (`vector` | `graph`), visibility, owner, project, content preview.
- Click result → open detail view inline without losing the search context.

### Knowledge graph view
- Visualize Graphiti entities, edges, episodes, communities.
- Filter by `group_id` (private / shared / project-scoped).
- Show edge temporal state (current / invalidated / expired).
- Click node → side panel with summary, incident edges, source memories.
- Two surfaces: a "graph view" of the current explorer's filter state, AND a standalone full-graph view at `/graph`.

### Team / shared pool
- Dedicated view of just `visibility=shared` memories, with owner always visible.
- Contributor leaderboard (who's shared what, recently).
- Same filter and edit affordances as the main explorer.

### Projects
- List of all projects with memory counts and recent activity.
- Per-project landing that scopes every feature above to that `project_id`.

### Activity / task queue
- Live list of recent async writes with their lifecycle: queued → processing → completed / failed.
- Poll `GET /v1/memories/status/{task_id}` while tasks are active; back off when idle.
- For failed tasks, show the full error payload and offer retry (for batches: retry only the failed items).
- Show in-flight worker count and queue depth.

### User settings
- Theme, density, time zone, default project.
- View own token: fingerprint and expiry only (never display the full token after initial issuance).
- Request token reissuance from admin (sends a signal; admin acts).

### Admin: user management
- Roster of all users with memory counts (private/shared split), last seen, token TTL chip.
- Per-user inspection: pool composition, token claims, recent activity.
- Issue new token: pick `user_id`, TTL (preset 7d / 30d / 90d / 365d / never, or custom). Token is displayed once with a copy button and never persisted client-side.
- Reissue token (regenerates with new TTL, keeps user_id).
- Bulk operations on a user's memories (with the same private-only-default safety).

### Admin: tokens audit
- Active tokens table: subject, issued at, expires at, last successful auth, fingerprint.
- Append-only issuance log: every issue / reissue / revoke event with who, when, what.
- Per-token revoke action explains it requires secret rotation (and offers the same).

### Admin: health
- Per-backend status (Neo4j, Redis, Qdrant, Gemini) with last-check latency.
- Worker status (which workers are healthy, how many tasks in flight, last cron run).
- Gemini quota usage prominently — that's the single biggest production-fragility vector.
- Recent log tail with level filter.

### Admin: migrations
- Run the existing CLI migration scripts safely from the UI:
  - **Bulk visibility promotion** (private → shared or back) with a category / owner filter.
  - **Graph group_id backfill** (legacy `"global"` / `"project--…"` → user-namespaced format).
  - **Expire old memories** (mark past-expires_at memories as expired).
- Every migration is dry-run first. The Apply button requires a type-to-confirm dialog.
- History of past runs with summaries; rollback where applicable.

### Admin: system settings
- Default visibility per category (override built-in defaults).
- Default token TTL.
- Dedup cadence.
- Retention policy (auto-expire after N days for selected categories).
- Legacy shared-API-key toggle.
- Signing-secret rotation (fingerprint only; rotation invalidates all tokens — type-to-confirm).

### Login
- Token paste. Service URL + token field. Client-side parse the token payload to decode `user_id` + `exp` for instant feedback before submission. Server still verifies the signature.

## States and behaviors the design must define everywhere

- **Loading**: skeletons matching the final shape.
- **Empty**: every list teaches the next action — never a bare "no data".
- **Error**: show the raw API response with a copy button. Never swallow it.
- **Unauthorized**: bounce to login with a banner and return-to URL.
- **Forbidden**: inline message (e.g. "this shared memory belongs to alice — only she can edit").
- **Rate-limit (429 from Gemini)**: amber banner with retry-after countdown; auto-retry once on dismissal. Don't pretend the search returned nothing.

## Accessibility

- Visibility (private / shared) and status (queued / processing / completed / failed) must encode in BOTH color AND a glyph/label. Color alone is never the only signal.
- Focus rings on every interactive, never removed.
- Reduced-motion respected throughout.
- Body contrast 4.5:1, large text 3:1 minimum.

## Out of scope (v1)

- Real-time collaboration / WebSocket-driven live updates.
- Memory threading / conversation reconstruction.
- Multi-team isolation.
- Mobile (responsive down to tablet only).
- External SSO / OIDC.
- Native desktop app.
- Public/marketing pages.

## Critical references

When in doubt about exact data shapes, consult:

- `neuralscape-service/schemas.py` — `MemoryResponse`, `MemoryVisibility`, `BulkDeleteRequest`, the 13 categories with scope/visibility defaults.
- `neuralscape-service/main.py` — every `/v1/*` route the UI will call.
- `.claude/skills/neuralscape-memory/SKILL.md` — agent-facing description that mirrors what the UI shows to humans.
- `docs/neuralscape/03-memory-model.md` — deep memory-model reference.
- `docs/neuralscape/09-plugin-system.md` — plugin / hook integration the Activity screen visualizes.

## Deliverable

A working frontend project (Next.js App Router or React + Vite, design agent's call) that:

1. Implements every feature listed above against **mock data** shaped like the real API (no live calls).
2. Has a coherent design system — colors, type, spacing, components — internally consistent.
3. Supports dark + light themes.
4. Has working keyboard shortcuts where they help.
5. Includes a short README listing each screen with a screenshot.

Hand back the design agent's chosen visual direction in the README so we can review the *look* separately from the feature set.
