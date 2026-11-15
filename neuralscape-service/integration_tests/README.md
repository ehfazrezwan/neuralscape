# Integration tests

Tests here run against **real** Neo4j, Qdrant and Redis (and Gemini where the
feature uses it). They are written on a laptop but only *run* by the VPS gate
(and CI jobs that name a file explicitly). A feature behind a flag is not done
until a test here shows it working **with the flag on**.

## Why not under `tests/`

CI runs `pytest tests/ --ignore=tests/test_async_pipeline.py` with no stores,
then `tests/test_ci_skip_gate.py` rejects every skip that is not declared. An
integration test under `tests/` would either fail (no stores) or skip (gate
failure). `integration_tests/` is never collected by `pytest tests/`, is copied
into the test and runtime images, and `conftest.py` here marks everything in
it `integration`. `tests/test_integration_convention.py` fails CI if
`pytest.mark.integration` appears under `tests/` (except the excluded
`test_async_pipeline.py`) or if the auto-marking is removed.

## Writing one

Copy `test_flag_on_exemplar.py`. The fixtures (`conftest.py`):

| fixture | gives you |
|---|---|
| `stores` | `neo4j_uri/user/password/database`, `qdrant_url`, `redis_url` from env, each probed once |
| `neo4j_conn` | Neo4j only (for tests that need no other store; the snapshot job uses it) |
| `llm_key` | the `GOOGLE_API_KEY` value (ask for it only if the test spends LLM calls; never print it) |
| `pg_dsn` | the PostgreSQL+pgvector DSN (`NS_TEST_PG_DSN`), probed |
| `api_user` | `.user_id` and `.headers` (Bearer token) of the per-run API identity |
| `live_api` | `.url`, `.user_id` and `.headers` of the running API + workers |
| `flags(**fields)` | turns Settings fields on for this one test, e.g. `flags(standards_enabled=True)` |
| `unique_ns` | unique `user_id`, `project_id`, `qdrant_collection`, `group_prefix` (all carry `nsit`) and `unique_ns.on_cleanup(fn)` |

**Turning a flag on.** `flags` does `monkeypatch.setenv(FIELD_NAME, ...)` (for any
new `Settings()` or subprocess) and `monkeypatch.setattr(config.settings, field,
value)` (the same singleton every `from config import settings` module holds, the
mechanism the unit tests use). Unknown field names raise. Caveat: a flag copied
at import time (module constant, default argument, decorator argument) is not
changed by setattr; import or `importlib.reload` that module after calling
`flags`, inside the test.

**Isolation.** Write only under the `unique_ns` names and register deletion with
`on_cleanup` (drop the Qdrant collection, `DETACH DELETE` the group, delete the
Redis keys). Leftovers from a crashed run are findable by the `nsit` prefix.

**Missing infrastructure.** With `NS_INTEGRATION_REQUIRED=1` (the VPS gate sets
it) an unreachable store or missing key **fails** the test; without it the test
**skips** with the reason. Never silently pass.

**Zero skips on the gate.** The gate sets `NS_INTEGRATION_REQUIRED=1`, so
missing infrastructure is an error, never a skip. Any SKIPPED integration test
in a gate run must be justified in the PR body, and a test may skip only on a
declared condition (for example an optional extra that is not installed), never
for missing stores, keys or API. A cleanup that fails in `unique_ns` fails the
test's teardown. Store URLs in messages and reprs have their passwords redacted.

**The per-PR "flag-on proof".** A PR that adds or changes a flagged feature adds
one test here that (1) calls `flags(...)`, (2) exercises the feature through the
service code against the real stores, (3) asserts the observable result, and
ideally (4) shows the flag-off contrast. State the test id in the PR body.

## Environment contract

A test may rely on exactly these variables and nothing else. The gate sets them.

| Variable | Meaning | Fixture |
|---|---|---|
| `NS_INTEGRATION_REQUIRED` | `1` on the gate: missing infrastructure fails instead of skipping | all |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Neo4j | `stores`, `neo4j_conn` |
| `QDRANT_URL` | Qdrant | `stores` |
| `REDIS_URL` | Redis | `stores` |
| `GOOGLE_API_KEY` | a real Gemini key | `llm_key` |
| `NS_TEST_PG_DSN` | PostgreSQL with pgvector | `pg_dsn` |
| `NS_API_URL` | the live API (with the three workers) | `live_api` |
| `NEURALSCAPE_USER_TOKEN_SECRET` | the stack's own user-token secret; a per-run token is minted from it | `api_user`, `live_api` |
| `NS_API_TOKEN`, `NS_API_USER` | optional override of the minted identity | `api_user`, `live_api` |

The gate also sets `LEDGER_TEST_PG_DSN`, `ENGINE_TEST_PG_DSN`,
`USAGE_TEST_PG_DSN` and `AUTHORITY_TEST_PG_DSN` to the same DSN as
`NS_TEST_PG_DSN`, for Phase 2 slices whose tests read those names. Phase 2
slice 1 shipped its PostgreSQL and Redis tests under `tests/` with `skipif`
guards (declared in the skip manifest); the expected home is
`integration_tests/` using `pg_dsn` / `stores`, and they are being moved there.

**Fail versus skip.** Missing infrastructure fails the test (a fixture error,
exit status 1) when `NS_INTEGRATION_REQUIRED=1` **or** `GITHUB_ACTIONS=true`
(so the snapshot CI job can never go green by skipping); otherwise it skips
with the reason.

**Live API identity.** `api_user` / `live_api` mint a short-lived token with the
service's own `tokens.issue_user_token` for a unique user id and send it as
`Authorization: Bearer`. The identity comes from the token, so request bodies
must carry `api_user.user_id` (the API answers 400 on a mismatch).
`tests/test_async_pipeline.py` does the same through
`integration_tests/api_identity.py` and reads `NS_API_URL` (default
`http://localhost:8199`); it fails instead of skipping when the API is
unreachable and `NS_INTEGRATION_REQUIRED=1`.

## The VPS gate command

One command covers both, run inside the **test image** (WORKDIR
`/app/neuralscape-service`) on the stores' network, with the variables above
passed as environment (never echoed):

```
NS_INTEGRATION_REQUIRED=1 pytest integration_tests/ tests/test_async_pipeline.py -m integration -v -p no:cacheprovider
```

`tests/test_async_pipeline.py` lives under `tests/` but is named explicitly, so
`excluded_suites` (which only keeps it out of `pytest tests/`) does not stop it.

The native snapshot job in `service-ci.yml` still runs
`integration_tests/test_native_snapshot_neo4j.py` by file name; the marker does
not change it. On GitHub Actions (`GITHUB_ACTIONS=true`) the fail-instead-of-skip
rule is on automatically, so that job can never go green by skipping.

## Skip manifest (`.github/ci/expected_skips.json`)

The CI skip gate compares each run's skips with the declared inventory. The
manifest file keeps `version`, `excluded_suites` and `entries_dir`; every
declared skip is **one file** in `.github/ci/expected_skips.d/`, named after the
test id with `::` replaced by `--`:

```json
{
  "test": "tests.test_baseline_reproductions::test_x",
  "reason_contains": "start of the skip reason",
  "profiles": ["container", "host"],
  "why": "pinned bug: ..."
}
```

**When a lane fixes a pinned bug:** in the same PR that removes the strict
`xfail`, `git rm` that bug's entry file. No other file changes, so parallel PRs
never conflict; CI's skip gate proves it (a declared skip that no longer skips,
or a fixed xfail that is still declared, fails the run).

**Adding many skips at once** (a slice with a generated test matrix): run the
suite with `--junitxml`, then

```
python3 -I neuralscape-service/tests/test_ci_skip_gate.py --junit junit.xml \
  --profile container --write-entries .github/ci/expected_skips.d --why "<one line>"
```

Repeat per profile (profiles merge into existing files). The generated files are
a draft: review every test id and reason before committing.

## Black-box evidence

"It works in action" evidence (REST/MCP/hooks against the running compose stack)
lives in the separate top-level `acceptance/` suite, not here. Integration tests
here call service code against real stores; acceptance scenarios go through the
real surfaces.
