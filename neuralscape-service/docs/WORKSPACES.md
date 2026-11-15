# Workspaces

A workspace is a partition of the memory store. Reference material (a book, a
manual, an imported corpus) goes into its own workspace so it never reads as a
fact about the user. This page describes what exists on `dev`.

There are two kinds of workspace:

- the **memory** workspace: the default. Rows with no `metadata.workspace`, a
  null one, or the value `memory` belong to it. Nothing about existing rows
  changed when workspaces were added.
- **reference** workspaces: any other name matching `[A-Za-z0-9_.-]{1,100}`.

## Write side

- Every row can carry `metadata.workspace`.
- `POST /v1/memories/raw` and the document ingest route `POST /v1/ingest` accept an
  optional `workspace` field (same name pattern), but the API routes do not forward it today: the
  value is validated and then ignored, so rows written through them carry no
  workspace unless the ingest pipeline derives one (next bullet). `POST /v1/ingest/text` and
  `POST /v1/ingest/files` have no `workspace` field.
- The ingest pipeline gives every document ingested with a non-default adapter
  (for example `trading_strategy`) a workspace named `ref-<adapter>`, or
  `ref-<adapter>--<title-slug>` when the document has a title. The slug is the
  lower-cased title with every run of characters outside `a-z0-9-` replaced by
  `-`, trimmed of leading and trailing `-`, cut to 30 characters. The title is
  the artifact file name when ingest storage is on (`<title>.md` for
  `POST /v1/ingest/text`).
- The dedup key includes the workspace, so identical text in two workspaces is
  stored twice.
- The `retag_memories` MCP tool has a `set_workspace` operation that moves
  existing rows into a workspace (an explicit null clears it).

## Graph group ids

Graph episodes and edges live in Graphiti groups named after the pool:
`user--<id>`, `user--<id>--project--<pid>`, `shared`, `shared--project--<pid>`,
`standard`, `standard--project--<pid>`. A reference workspace appends
`--ws--<workspace>` to the pool name, for example `user--u1--ws--ref-x`. The
memory workspace appends nothing.

## Dreaming fences

Dreaming (the consolidation sweep) keeps workspaces apart:

- pools are keyed per workspace, so rows from different workspaces are never
  merged together;
- a reference workspace only runs `merge` and `prune` actions; reflection,
  temporal reframing and invalidation are skipped;
- reference rows are never eligible for the identity card.

## Read side

Recall reads the memory workspace only, unless the caller names workspaces.

- MCP `recall_memories`: optional `workspaces`, an array of up to 10 names.
- REST `POST /v1/search`: optional `workspaces`, same shape. A malformed list is
  rejected (422 on REST, an `error` result on MCP).
- `MemoryService.search(workspaces=[...])`.

Behaviour:

- Absent or `[]`: the memory workspace only, as before.
- `["ref-x"]`: only rows in `ref-x`.
- `["memory", "ref-x"]`: both. Names match exactly: `ref-trading_strategy` does
  not match `ref-trading_strategy--my-book`.
- The vector pools apply the workspace condition inside the Qdrant filter, on
  both the dense and the BM25 leg, so a reference row is found even when
  stronger rows from another workspace would fill `limit`. A per-row check after
  the query stays as a second guard.
- The graph leg searches the caller's own groups (private, shared, standard,
  plus the project variants) with `--ws--<workspace>` appended for each
  requested reference workspace. A workspace name never adds a group the caller
  could not already read. With `workspaces` absent the groups are unchanged.
- Visibility, project, category and the other recall filters apply as usual.

Not built: there is no tool or endpoint that lists workspaces, and no way to
bind a session or project to a workspace. Callers learn the name from the
ingest (`ref-<adapter>[--<title-slug>]`).
