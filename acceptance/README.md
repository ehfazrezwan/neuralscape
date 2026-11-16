# Acceptance harness

Black-box proof that features work **in action**: scenarios run against a
live, production-like compose stack (API, real Neo4j / Qdrant / Redis, all
three ARQ workers, real LLM) and talk to it only through its real surfaces:

- REST (`/v1/*`, `/health`)
- MCP Streamable HTTP at `/mcp/`
- the plugin's **built** hook scripts (`neuralscape-plugin/scripts/*.js`, run with `node`)

Scenarios import nothing from `neuralscape-service` or the plugin source and
use no mocks. Python + pytest + httpx (pinned in `requirements.txt`).

## Run

```bash
acceptance/run.sh http://127.0.0.1:18199 --compose-project nstest-acceptance
acceptance/run.sh http://127.0.0.1:18199 --profile default --out /tmp/default.json
acceptance/run.sh http://127.0.0.1:18199 -k remember          # subset
acceptance/run.sh http://127.0.0.1:1 --list                  # inventory JSON, no stack contact
acceptance/run.sh --selftest                                  # harness self-tests, no stack
```

Works from any cwd. Environment: `uv` if it runs, otherwise a venv under
`${XDG_CACHE_HOME:-$HOME/.cache}/ns-acceptance-venv/<hash of requirements.txt>`
(python3 >= 3.10); neither available -> exit 2.

Exit codes: `0` every selected scenario passed or skipped, `1` any fail or
error, `2` harness misconfiguration (no token source, unknown profile,
nothing selected).

### Tokens

Each run creates fresh users `acc-<runid>-a` / `acc-<runid>-b` and a project
`acc-<runid>-proj`, so runs never collide and the two-user isolation tests are
real. Token sources, first match wins:

1. `--compose-project P` (or `NS_COMPOSE_PROJECT`): runs
   `docker compose -p P exec -T <service> python scripts/issue_user_token.py --user <id> --days 1`
   (the AUTH.md way). Service defaults to `neuralscape`; override with `NS_API_SERVICE`.
2. `NS_TOKEN_SECRET`: sign locally with the stack's own secret (two-segment
   HMAC-SHA256 format from AUTH.md, re-implemented in `nsaccept/tokens.py`).
3. `NS_TOKEN_A` + `NS_TOKEN_B` with `NS_USER_A` + `NS_USER_B` (two different users).

A token signed with a random, wrong secret (`ctx.bad_token`, fixture `bad`)
is always minted locally, for negative auth tests. Token-shaped strings are
scrubbed from the results file.

Other env: `NS_RUN_ID` (fix the run id), `NS_COMMIT` (commit recorded in the
results; default `git rev-parse HEAD` of the repo holding `acceptance/`),
`NS_REPO_ROOT`, `NS_HAVE_<CAP>=1` (declare a capability for `@needs`).

## Gate command

Run on the host where the throwaway stack runs, once per file in `profiles/`:

```bash
NS_COMMIT=<sha> acceptance/run.sh http://127.0.0.1:18199 --compose-project <disposable project> --profile <name> --out <file>
```

`NS_COMMIT` is the commit the stack image was built from; it is recorded as
`commit` in the results. If the stack ever exposes its own commit (a `commit`
or `git_sha` field in `/health`, or `NS_STACK_COMMIT`), it is recorded as
`stack_commit` and the run exits 2 when the two disagree (prefix match). Today
`/health` exposes none, so `stack_commit` is `null` unless `NS_STACK_COMMIT`
is set. Without `NS_COMMIT` the harness checkout's HEAD is recorded, which only
describes the harness, not the stack.

For a profile whose `profiles/<name>.env` starts with a `# services: a, b`
header, the gate brings those extra compose services up next to the stack and
exports `NS_HAVE_<SERVICE>=1` (e.g. `NS_HAVE_DOCLING=1`) so `@needs` scenarios
run instead of skipping. The header is documentation for the gate; this harness
only reads `KEY=value` lines (recorded as `profile_flags`, plus the file's
`profile_sha256`, in the results).

## Safety and cleanup

- **Production guard.** The run writes real data, so it refuses (exit 2) the
  compose project name `neuralscape` (no override), port `8199` (override only
  with `NS_ALLOW_PROD_PORT=1`) and any non-loopback host (override only with
  `NS_ALLOW_REMOTE=1`). Never point it at production.
- **No cleanup.** `acc-<runid>-*` users, projects and memories are never
  removed; run against throwaway volumes (the gate's own disposable compose project).
- **Compose token minting.** `docker compose -p <project> exec` without `-f`
  finds the project's existing containers by their compose labels, so it must run
  on the host where that project is running.
- **One negative control needs no stack**
  (`test_health_check_rejects_a_dead_backend`); against a dead URL it still
  passes, which is expected.

## Profiles

`--profile <name>` runs the scenarios whose `@profiles(...)` include it
(default profile: `("default",)`); others are deselected, not reported as
skipped. The gate does **one run per profile file** in `profiles/`.

A flags-on profile is `profiles/<name>.env`: plain `KEY=value` lines the gate
adds to the API **and all workers'** environment before bringing the stack up.
An optional header line `# services: docling` lists extra compose services the
profile needs; the gate then also exports `NS_HAVE_DOCLING=1`, which is what
`@needs("docling")` checks. Add a profile file in the same PR as the scenario
that needs it. `profiles/default.env` has no flags.

## Flags-on profiles (what the gate must do for each)

A feature behind a flag is not green until it is proven working **with the flag on**. Each profile file
turns flags on for the throwaway stack only (product defaults are never flipped). One run per profile
file, each preceded by a stack restart with that file's lines added to the API and every worker's env:

| profile | flags / services | scenario files |
|---|---|---|
| `flags-on` | dreaming, standards, processes, dictator, connectors (+stdio allowlist), queue-empty webhook, native code graph, OAuth server (token login), speaker-aware extraction, strategy synthesizer | `test_dreaming`, `test_standards`, `test_connectors`, `test_webhook`, `test_code_graph`, `test_oauth`, `test_speaker` |
| `oauth-google` | `AUTH_PROVIDER=google` (placeholder client id) | `test_oauth` (provider part) |
| `oauth-supabase` | `AUTH_PROVIDER=supabase` (placeholder project URL) | `test_oauth` (provider part) |
| `cbm` | `CBM_ENABLED`, service `cbm-bridge` | `test_code_graph_cbm` |
| `docling` | Docling + `EXEMPLAR_STORE_ENABLED` + `DOCLING_EXTRACT_IMAGES` | `test_ingest` (Docling), `test_exemplars` |

**Placeholders.** No secret is written in a profile. A line like `VAULT_KEY=<generate:fernet>` (or
`<generate:hex32>`) is replaced by the gate with a freshly generated value before the stack starts: one
value per KEY, identical for the API and every worker, never written back to the repo or to the results
(the harness records the placeholder in `profile_flags`, and redacts any secret-looking key whose value is
not a placeholder). `fernet` = `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`,
`hex32` = 32 random bytes as hex. Single-quoted values (`CODE_REPOS='{"svc":"..."}'`) are literal JSON.

**Header lines.** `# services: a, b` names extra compose services to bring up (and exports `NS_HAVE_<NAME>=1`).
`# sidecars: nsc-acc-sink` and `# mounts: ...` are documentation for the gate: the harness starts and removes
the sink itself, and the `cbm` profile needs the service's `neuralscape-service/knowledge/` directory mounted
read-only at `/corpus/knowledge` in the `cbm-bridge` container (the profile's `CODE_REPOS` names that path).

**Dictator.** A profile with `DICTATOR_USER_IDS=acc-dictator` gives the run a third identity: fixtures
`dictator` (REST) and `mcp_dictator` (MCP), `ctx.dictator_user` / `ctx.token_dictator`. The token is minted for
the FIRST id in the list the same way as the A/B tokens (compose issuer, else `NS_TOKEN_SECRET`); a run given two
pre-made tokens has none and `@needs("dictator")` scenarios skip. The harness exports `NS_HAVE_DICTATOR=1` when it
has one. In `flags-on` the acceptance users A and B are NOT dictators/admins.

**Sink sidecar.** `nsaccept/sidecar.py`, fixture `sink`, `@needs("compose")`: one
`docker run -d --rm --name nsc-acc-sink --network <project>_default --memory 64m --cpus 0.25 python:3.12-alpine`
serving the run's fixture JSON and logging every request (read back with `docker logs`). It refuses any name
not starting `nsc-acc-`, the `neuralscape` project, more than two live sidecars, publishes no port and mounts
nothing, issues only `docker run|logs|rm|network ls`, and is removed in the session finalizer (and at exit).
The gate should `docker pull python:3.12-alpine` once beforehand. The stack's own `WEBHOOK_QUEUE_EMPTY_URL`
and the connector's `list_url` use the container name `nsc-acc-sink`, which only resolves while the sink runs.

**Time.** The flags-on run is dominated by the dreaming sweep (real LLM, queued behind graph enrichment of
the seeds on the single graph worker) and the native index; budget about 15 minutes on 2 CPUs.

### Reading a flags-on result honestly

- **Skips are not evidence.** `flags-on` declares `# requires: compose`, and `run.sh` exits 2 without
  `--compose-project` rather than skipping the compose-only scenarios. A skipped scenario never counts as proof.
- **Compose projects.** Every entry point that touches Docker (`ctx.compose`, token minting, the sink) goes through
  one function, `guard.check_compose_project`: case-insensitive after strip, refuses `neuralscape`, and refuses any name
  outside the disposable prefix `nstest-` (add prefixes only with `NS_COMPOSE_ALLOWLIST`).
  Only two read-only `exec` forms exist besides the Qdrant one: vault `grep -rl` under `/data/vault` and
  `redis-cli GET ns:connector:<id>`.
- **LLM-dependent assertions** (an LLM decision, not wording, but a decision nonetheless): the contradicting fact is
  invalidated (not merged), a reflection insight is stored, the identity card keeps the user's name token, speaker
  attribution on two real extractions, the Docling figure detection. A red here is a real finding for the owning lane
  or a model regression, not a harness bug; none was loosened to pass.
- **Order dependence.** Within a module, tests share one fixture (the seeds, the sweeps, the index, the sync) and
  read what it recorded, so they do not depend on each other's order; the exceptions are the last test of
  `test_code_graph` / `test_code_graph_cbm` (they delete the code space) and `test_connectors`' delete test, which
  use only objects they create. Do not run a single test of those modules with `-k` after a previous run on the same stack.
- **`@expected_red("#<pr>")`** labels a scenario that is known to fail until that PR lands. It is not an xfail: the scenario runs, a failure is reported as `fail`, and the results JSON carries `expected_red` (the reference, else null) so known reds can be told from new ones. Nodes deliberately not covered by any scenario are listed with a reason in `COVERAGE.md`.
- **Expected red until PR 495 (sec-03):** the consent-page framing headers and refresh-token rotation in `test_oauth`.
- **Coverage of the old blind e2e scripts is partial** (see the PR body's disposition table): they stay in the repo
  for the parts not ported.
- **Hard-coded port.** The OAuth profiles carry `NEURALSCAPE_PUBLIC_URL=http://127.0.0.1:18199`; the scenarios assert
  it equals the URL under test, so a gate on another port must edit that one line.

## Results JSON

Written to `--out` (default `acceptance/results/<profile>-<commit12>.json`,
git-ignored):

```json
{
  "commit": "<40-hex or NS_COMMIT>", "profile": "default",
  "base_url": "http://127.0.0.1:18199",
  "started": "2026-11-16T12:00:00Z", "finished": "2026-11-16T12:03:10Z",
  "scenarios": [
    {"id": "test_auth::test_no_token_is_401", "proves": ["auth.bearer-middleware"],
     "result": "pass", "seconds": 0.04, "detail": "", "negative_control": false}
  ]
}
```

Also at the top level: `stack_commit` (or null), `profile_flags` (the `KEY=value` lines of the profile file) and `profile_sha256`.

`result`: `pass`, `fail` (assertion), `skip` (`detail` = reason, e.g. "node not
available"), `error` (anything else: connection refused, fixture failure,
harness bug). `detail` is a short, token-scrubbed message.

## Adding a scenario (any lane)

Drop a `test_<area>.py` in `scenarios/`. Template:

```python
from nsaccept import eventually, expect_check_fails, negative_control, proves, profiles

# one REAL check, reused by the positive scenario and the negative control
def check_recalls(client, marker):
    hits = client.search(f"remember {marker}", project_id=PROJECT)
    assert any(marker in h.get("memory", "") for h in hits), "marker not recalled"

@proves("write.remember-raw", "read.hybrid-search-core")   # atlas node ids
# @profiles("flag-x")        # only when a flags-on profile is needed
def test_my_feature(ctx, a):                    # a, b = user A / B clients
    marker = f"zq-{ctx.run_id}-myfeature"       # unique per run, no LLM wording
    task = a.store_raw(f"{marker} some fact", "tech_stack", scope="project",
                       project_id=ctx.project_id, visibility="private")
    assert a.poll_task(task, timeout=120)["status"] == "completed"      # async write
    eventually(lambda: check_recalls(a, marker), timeout=90)             # async effect

@proves("write.remember-raw", "read.hybrid-search-core")
@negative_control                                # shows the check CAN fail
def test_my_feature_negative_control(ctx, a, b):
    expect_check_fails(check_recalls, b, f"zq-{ctx.run_id}-myfeature")   # wrong user
```

Helpers: fixtures `ctx`, `anon`, `a`, `b`, `bad` (REST clients), `mcp_a`,
`mcp_b` (MCP clients with `initialize`, `list_tools`, `call_tool`),
`nsaccept.hooks.run_hook(name, payload, base_url=..., token=..., user_id=...,
project_id=...)`, `@needs("node")` / `@needs("docling")` / `@needs("compose")`,
and the shared session writes in `scenarios/conftest.py`: `a_fact`,
`shared_fact` (visibility=shared, in `shared_project(ctx)`), `graph_seed`
(facts enqueued early so the slow graph worker overlaps the rest of the run) and
`conv_seed` (the suite's one LLM extraction write, a 22-message conversation).

More helpers in `nsaccept/`:

- `helpers.py`: `write_raw` (raw write + poll, records instead of raising),
  `Written.require()`, `list_rows`, `rows_with`, `mcp_json`, `graph_listing`,
  `blob_has`, `memory_ids_in`.
- `ctx.compose(*args)`: `docker compose -p <project> <args>`, only when run
  with `--compose-project` (which also sets the `compose` capability, so
  `@needs("compose")` scenarios run). Refuses the production project name and
  any verb outside pause/unpause/stop/start/restart/ps/logs/top/exec/up (`up` only with `-d --no-build --no-recreate`, i.e. starting existing containers).
  Expired-token scenarios mint inside the API container (`tokens.mint_via_compose_python`) when `NS_TOKEN_SECRET` is not set, so they also run on the gate.
- `sse.SSEReader`: background reader for `GET /v1/stream`
  (`wait_connected`, `wait_for`, `events`).
- `upload.upload_files` / `upload.build_zip`: multipart upload to
  `/v1/ingest/files`; `pdfgen.make_pdf`: dependency-free text PDF.
- `probe.probe_route`: sends a harmless request and says whether the route is
  mounted (not 405, not FastAPI's own "Not Found", not a 5xx).

Scenario files (one per family; default profile unless noted):
`test_auth_isolation` (tokens, identity, task ownership), `test_write_paths`,
`test_read_paths`, `test_graph`, `test_edit_delete`, `test_ingest`
(Docling cases only under the `docling` profile), `test_extensions` (registry,
SSE, knowledge router, admin gates, session summaries), `test_fault_injection`
(needs `--compose-project`; Qdrant outage guarantees and a 3-cycle cold start
that deletes the collections and starts the API + workers together; always
restores the stack, or fails with `ps` output). Run order is alphabetical except `test_edit_delete` and
`test_fault_injection`, which run last (see `RUN_LAST` in `scenarios/conftest.py`).
Flags-on files (`test_dreaming`, `test_standards`, `test_connectors`, `test_webhook`, `test_code_graph`,
`test_code_graph_cbm`, `test_oauth`, `test_speaker`, `test_exemplars`) each declare their profile once with a
module-level `pytestmark = profiles(...)`; `scenarios/cg_helpers.py` is shared by the two code-graph files.
`scenarios/route_table.py` lists every REST route with a harmless probe; a
drift scenario compares it with the stack's `/openapi.json`.

Rules:

- **A scenario must fail when the feature is broken.** Assert on an observable
  effect through the real surface, never on "no exception". Every scenario file
  ships at least one negative control (the file-level self-test enforces this,
  and that every scenario has `@proves`).
- Keep LLM calls minimal (prefer `/v1/memories/raw`); assert on unique markers,
  never on model wording.
- Use the run's unique ids (`ctx.run_id`, `ctx.user_a/b`, `ctx.project_id`);
  never reuse fixed ids, never touch other users' data.
- Async effects go through `poll_task` then `eventually`, with bounded timeouts.
- A flagged feature needs a flags-on profile **and** a scenario under it; it is
  not green because the flag is off.
- No hostnames, org names or secrets in any file here (public repo).

## Self-tests

`acceptance/selftest/` needs no stack: token format vs a hard-coded vector,
profile selection, results writer, `expect_check_fails`, MCP client against a
local stub (JSON, SSE, session id), REST client polling, hook runner, and a lint
over the real scenarios. `run.sh` never collects it; run `run.sh --selftest`.


## Destructive scenarios

`test_fault_injection::test_api_and_workers_started_together_on_empty_collections_stay_up`
**wipes the Qdrant collections** (every memory on the stack) three times. It runs only
when `NS_ALLOW_DESTRUCTIVE=1` is set and a compose project is given that passes the
allowlist below; otherwise it skips (or is refused) with that reason. The gate sets
`NS_ALLOW_DESTRUCTIVE=1` on its throwaway stack only; never set it elsewhere.


## Compose project allowlist, and `uv`-only boxes

Every compose project name the harness touches (`run.sh`, `guard.check`, `ctx.compose`,
token minting) goes through `guard.check_compose_project`: stripped and lower-cased,
`neuralscape` is always refused, and only disposable prefixes pass (default `nstest-`;
`NS_COMPOSE_ALLOWLIST=prefix1,prefix2` is the only way to add one, e.g. the gate exports
its own project's prefix). Container
listing is always `docker compose -p <project> ps`, never global `docker ps`.

`run.sh` needs only `uv`: it runs `uv run --no-project --python 3.12 --with-requirements ...`
(uv may download a managed CPython; downloads are allowed). Set `NS_UV=/path/to/uv` for a
standalone binary that is not on `PATH`. Without uv it falls back to python3 >= 3.10 + venv.

**Evidence rule.** A results row is node evidence only when `evidence` is true: rows from `@negative_control` or `@expected_red` scenarios are never evidence (`evidence: false`). An expected-red scenario that passes is flagged `unexpected_pass: true` and named in the run summary, so the marker can be dropped.
