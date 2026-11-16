# Phase 2 story scenarios

Black-box stories for the Phase 2 slices that are on `dev` (the durable write ledger and the authority
store, scoped credentials and route policy), plus the steps that need slices that are not yet there. Same
harness, same rules as `README.md`; this file only says what the stories are and how a gate runs them.

Files: `profiles/phase2.env`, `scenarios/test_phase2_durable_write_story.py`, `test_phase2_two_principal.py`,
`test_phase2_member_and_tenant.py` (all profile `phase2`), `test_phase2_shared_pool_gate.py` (profile `default`),
`scenarios/phase2_kit.py` (helpers), `nsaccept/redmark.py` (`@expected_red`).

## What is real on dev today (read from the code, not assumed)

* The ledger (`LEDGER_ENABLED`, `LEDGER_BACKEND` sqlite|postgres) is a library: nothing routes a write
  through it yet, so with the flag on a write still returns a legacy queue task and `Idempotency-Key` is
  ignored. No receipt exists on any route until the write path is cut over onto the ledger.
* The authority store, route policy (`NS_POLICY_MODE` off|observe|enforce) and scoped credentials
  (`CREDENTIALS_ENABLED`) are live. In `enforce` an identity the authority store does not know is denied on
  every route, a named denial is 403 and an existence-safe one is 404.
* The one offline way to make identities known is `python -m policy_migration apply`. It grants each user
  their own pools (project-less pools always; project pools only the ones observed at migration time) and
  never grants `manage_credentials` or `manage_membership`.
* `GET /v1/export/okf` exports the caller's pool; an OKF bundle can be re-imported through
  `POST /v1/ingest/files`. The verified portable export, membership routes and tenant cells are not on dev.

## The stories

| Story | Scenario file | Steps | Surfaces |
|---|---|---|---|
| Durable write | `test_phase2_durable_write_story` | save, accepted receipt, current recall, correction, privacy narrowing, stale replay denied, export, restore, restart keeps the accepted work | `POST /v1/memories/raw`, `GET /v1/memories/status/{id}`, `POST /v1/search`, `PATCH /v1/memories/{id}`, `GET /v1/export/okf`, `POST /v1/ingest/files`, `POST /v1/portability/exports` |
| Two principals | `test_phase2_two_principal` | outsider, user B and scoped credentials denied at request, job-status and export boundaries; a revoked credential stops working | `GET/PATCH/DELETE /v1/memories/{id}`, `POST /v1/search`, `GET /v1/memories/status/{id}`, `GET /v1/export/okf`, `POST /v1/credentials` |
| Member journey (gap O-18) | `test_phase2_member_and_tenant` | invite, join, list, the member's default capture stays private | `POST /v1/members/invites`, `POST /v1/members/join`, `GET /v1/members`, `POST /v1/memories/raw` |
| Two real tenants (gap E-06) | `test_phase2_member_and_tenant` | another tenant's person sees nothing of tenant one over REST and MCP | the same read, status and export routes, MCP `recall_memories` |
| Project-less shared facts (gap E-05) | `test_phase2_shared_pool_gate` | a project-less shared fact is recalled from any project; a project's shared fact only from that project | `POST /v1/memories/raw`, `POST /v1/ingest/text`, `POST /v1/search` |

Each scenario lists the atlas nodes it proves with `@proves`; a node is listed only where the assertion
would fail if that node were broken.

## Expected red

`@expected_red("needs Phase 2 <slice>")` labels a scenario written against behaviour a later slice delivers.
The reason is one of a closed set of slice names (`nsaccept/redmark.py`). It is not a skip and not an xfail:
the scenario runs, fails honestly, and a pass is a pass. The label is the `ns_expected_red` pytest marker
(same marker and one-argument form the coverage-closure PR uses) and an `EXPECTED-RED[...]` prefix on the
failure message, which the results JSON already carries in `detail`.

| Reason | Scenarios that stay red until it lands |
|---|---|
| durable write cutover | accepted receipt; correction names and returns a revision; replay of the original write; stale edit refused; the receipt survives an API restart |
| portable export | tenant export is a verified bundle only its owner downloads |
| membership and onboarding | scoped credentials (issuing needs `manage_credentials`), including revoke; invite, join and list |
| onboarding policy | a member's default capture is private |
| tenant cells | two real tenants (needs `NS_TOKEN_T2` and `NS_USER_T2`) |

## Evidence rule

* A negative control proves something only if the same check PASSES for the right principal first. A
  control whose check cannot pass yet (the route or receipt does not exist) is itself `@expected_red`; it is
  never a green control that passes because the thing is missing.
* A denial scenario carries the policy atlas ids only if it asserts something only enforcement does. The
  outsider scenario does (an unknown identity is denied everywhere). Denials of user B are also produced by
  the visibility and owner rules with policy off, so they carry the visibility and edit ids only.

## What the gate does for `phase2` (its own restart, not part of the flags-on union)

`NS_POLICY_MODE=enforce` denies every identity the authority store does not know, which would break every
other profile's scenarios on the same stack, so this profile needs its own restart.

1. Run the `default` profile first with a fixed `NS_RUN_ID`, so the acceptance users A and B and their pools
   exist. Never write data as `acc-<runid>-outsider` (it must stay unknown).
2. In the API container, before the restart:
   `python -m policy_migration apply --tenant acc --authority-sqlite /data/ingest/phase2-authority.db --qdrant-url http://qdrant:6333 --collection <qdrant collection> --no-legacy-dictator-grants`
3. Restart the API and every worker with the lines of `profiles/phase2.env`, then run
   `run.sh <url> --compose-project <gate project> --profile phase2` with the SAME `NS_RUN_ID`.

The SQLite files sit on the API's `ingest_data` volume. The ledger is a single-host SQLite file: when the
write path starts using it, the worker that applies intents has to mount the same volume.

Optional, for the scenarios that need more than dev can give: bind user A as the tenant's first
administrator (membership slice), and export `NS_TOKEN_T2` / `NS_USER_T2` for a member of a second tenant
(tenant-cell slice). Without them those scenarios fail with a message that says so.

## Not covered here (and why)

* Restoring a portable bundle into a fresh tenant: the identity-mapping document needs the source tenant
  and subject ids, which no public surface on dev returns. The restore step is the OKF round trip today.
* Driving the plugin's own hooks as a joined member: the capture step uses the REST write the plugin makes
  for a team-default category. The hook-level capture scenarios belong to the plugin family.
* Adversarial and qualification scenarios, and persona agents: out of scope for this PR.

The durable write story restarts the API service once, as its last fixture step (`ctx.compose("restart", ...)`,
allowed by the harness), to show the accepted work survives it; later modules wait for `/health` before they
start.
