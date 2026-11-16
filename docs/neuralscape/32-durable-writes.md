# Durable Writes (the ledger write path)

With `LEDGER_ENABLED=true` the asynchronous write paths are accepted by the durable ledger instead of being handed to
the queue: a write is committed as an *intent* before the API answers `202`, a wake-up on the ledger queue lets a worker
run each stage, and the client polls a *receipt* built from the ledger. With the flag off (the default) every path
behaves as before. The ledger package itself is described in the ledger code (`neuralscape-service/ledger/`); this page is
the contract of the write paths on top of it.

## What goes through the ledger

`POST /v1/memories`, `/v1/memories/raw`, `/v1/memories/raw/batch`, `PATCH`/`PUT /v1/memories/{id}`, `DELETE`
`/v1/memories/{id}` and `DELETE /v1/memories` (the v1 bulk delete), `POST /v1/memories/retag`, the ingest routes (text,
document, files, OKF bundle), connector sync, code indexing, `POST /memories` and `DELETE /memories` (legacy), the MCP
`delete_memories` tool, the conversation compiler (flush and compile), the dreaming sweep, the session-summary
refresh, and the scheduled maintenance crons (dream sweep, strategy playbooks, dedup, expiry, connector sync).

Not routed: the MCP `edit_memory` tool (it stays a synchronous edit that does not advance the ledger's revision; the next `If-Match` edit of that memory is baselined again from the stored row) and the synchronous read routes. A dry-run of `rescope-private-derivatives` stays
synchronous.

## Clients

* `Idempotency-Key` (1-200 characters) is optional on every write route. The same key and the same request returns the
  original intent; the same key with a different request is `409`.
* Without a key **every request is its own intent.** The ledger cannot tell a retry from a second save, and a
  content-derived key would let a re-save of deleted content be skipped as a replay of the old receipt. Content-hash
  deduplication in the memory service still absorbs identical content applied one intent after another. Two identical
  keyless writes applied at the same moment can both pass that check: a client that retries must send a key.
* `PATCH` requires `If-Match: <revision>` (the revision the ledger registered for the memory; `1` for a memory written
  before the ledger existed). A stale revision is `409 stale_revision` and nothing is written.
* `DELETE` is refused with `409 hold_active` while a retention hold covers the memory (or, for a bulk delete, while the
  tenant has any hold).
* Poll `GET /v1/memories/status/{task_id}` (or the legacy `/memories/status/{task_id}`): the body is the same
  `queued` / `processing` / `completed` / `failed` (and `fenced`) vocabulary as before, with the receipt in `result`. A
  receipt lists at most 256 outputs; `GET /v1/memories/status/{task_id}/outputs?offset=&limit=` pages the rest.

## Workers

Run `arq worker.LedgerWorkerSettings` against the same ledger as the API. It has its own queue (`<ARQ_QUEUE_NAME>:ledger`),
`max_tries = 1` (a retry is a ledger attempt), a recovery-poll cron that claims pending obligations and steals expired
leases, and the retention cron when `LEDGER_RETENTION_ENABLED` is on. The vector stage runs in this process, so with
`ENGINE_ROUTING_ENABLED` it builds its own engine switch at start-up; the API builds one for the admin routes.
The scheduled crons are accepted once per tenant under an explicit subject (`LEDGER_CRON_SUBJECTS`, a JSON object
`{tenant_id: subject_id}`); a tenant with no entry gets no cron job, never a default user.

## Online cutover (`LEDGER_CUTOVER_ENABLED`)

`GET /v1/admin/ledger-cutover` and `POST /v1/admin/ledger-cutover/{start,backfill,advance,complete,rollback}` (dictators
only; `404` while the feature is off) move the legacy writers onto the ledger in four phases: `legacy`, `dual_accept`
(new writes go to the ledger, legacy jobs already queued still run), `ledger_only` (the legacy queue is drained and
fenced: a legacy job admitted after the fence reports `fenced` and runs nothing) and `complete` (after a clean final
reconciliation; rollback is refused). `backfill` is read-only and resumable: it registers a baseline reference for each
existing row and never writes to the legacy stores.

## Retention (`LEDGER_RETENTION_ENABLED`)

Idempotency records are freed after `LEDGER_RETENTION_IDEMPOTENCY_DAYS` (7), content is deleted and intents and receipts
are compacted after `LEDGER_RETENTION_CONTENT_DAYS` (30), stale processing work (24 hours) is reviewed rather than
failed, and tombstones and the audit record are kept unless configured. **A hold keeps every intent that touched the held
record**: the records the intent names, and the records its stages created or changed (an intent that created a memory
names the memory only in its receipts). An intent whose output list is too long for a receipt is kept while the tenant
has any hold.

## Policy

Under `NS_POLICY_MODE=enforce` a bulk delete or retag (a filter, not a row) is judged as if it may reach the `standard`
tier, so it asks for `publish_standard` as well as `delete` / `update`. The memory service still refuses a
non-dictator's standard rows one row at a time; relaxing the policy check is the owner's decision.

## Not yet wired

The owner-override erasure (`ledger.erasure.erase_owner_override`) and hold placement are library functions: no route or
tool calls them. Offboarding and the legacy-policy migration are wired by the onboarding routes (a later slice). The
compose files and the Helm chart have no ledger worker yet.
