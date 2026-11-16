# Nodes with no acceptance scenario

Each atlas node below is deliberately not covered by a scenario, with the reason. Everything else is proven by a scenario carrying its id in `@proves`. Reasons that name an open PR go stale when it merges.

| node | reason |
|---|---|
| `auth.legacy-shared-key` | needs NEURALSCAPE_API_KEY set on the stack AND known to the scenario; profiles hold no literal secrets and a generated placeholder value is not visible to scenarios (a literal key would also be redacted from the run record), and setting a key changes the auth surface every other scenario runs on; covered by tests/test_auth.py and the (#480) identity binding; becomes a scenario if the gate exports its generated key (own profile) |
| `ext.trading-adapter` | scenario ships in open PR #504 (acceptance/scenarios/test_workspace_recall.py: a trading_strategy ingest recalled through the workspaces parameter); on dev there is no recall path to assert |
| `fork.mem0.entity-store` | unreachable on a live path: since PR #479 no route calls mem0's add/search, so the entity side-collection is never touched (writes go through the service's own store_raw / extract_and_store); the lock and cold-start retry are covered by (#481) real-store integration tests |
| `auth.email-identity-slug` | the email-to-user mapping runs only inside a federated login callback that needs the identity provider's code exchange; the oauth-google / oauth-supabase profiles stop at the redirect to the provider and /oauth/link needs a ticket from a completed callback; covered by tests/test_auth_providers.py |
| `auth.identity-store` | durable identity links runs only inside a federated login callback that needs the identity provider's code exchange; the oauth-google / oauth-supabase profiles stop at the redirect to the provider and /oauth/link needs a ticket from a completed callback; covered by tests/test_auth_providers.py |
| `auth.email-allowlist` | the domain/address allowlist runs only inside a federated login callback that needs the identity provider's code exchange; the oauth-google / oauth-supabase profiles stop at the redirect to the provider and /oauth/link needs a ticket from a completed callback; covered by tests/test_auth_providers.py |
| `ops.autoheal` | the autoheal sidecar and its docker-socket proxy are not part of the gate's stack (app services and backing stores only), and proving it means making a container unhealthy and waiting out its 90 s start delay plus poll interval; wiring reviewed with (#488) |
| `ext.dreaming-graph-patcher` | its effects are Neo4j node properties (memory_id, ns_visibility/ns_owner stamps, dream_superseded_by) that no REST or MCP read exposes; the one observable effect (edge invalidation of a superseded fact) rides on the graph-supersession and flags-on dreaming scenarios, which claim their own nodes; the attach-window mislabel bug is owned by (#501) |
| `fork.mem0.add-pipeline` | mem0's native add is on no live path: since PR #479 legacy POST /memories runs the service's own extract_and_store and the service bypasses m.add everywhere; that legacy write is exercised by the delete.legacy-passthrough scenarios |
| `fork.mem0.telemetry` | mem0 PostHog telemetry: third-party telemetry is switched off in-process (third_party_telemetry.py sets the vendored libraries' env defaults before they import); from outside the process the only evidence would be the absence of outbound connections, which a one-instant socket snapshot cannot show, and the harness's read-only compose exec allowlist has no form that reads container sockets or environments (a new form means editing runctx.py, owned by open PR #526); (PR #451) merged with unit tests |
| `fork.graphiti.custom-ontology` | ontology-typed nodes are written into the reference-workspace graph group of an adapter ingest; no REST or MCP listing reads those groups on dev (open PR #504 adds the graph leg for them); revisit when #504 merges |
| `fork.graphiti.telemetry` | Graphiti PostHog telemetry: third-party telemetry is switched off in-process (third_party_telemetry.py sets the vendored libraries' env defaults before they import); from outside the process the only evidence would be the absence of outbound connections, which a one-instant socket snapshot cannot show, and the harness's read-only compose exec allowlist has no form that reads container sockets or environments (a new form means editing runctx.py, owned by open PR #526); (PR #451) merged with unit tests |
| `knowledge.native-snapshot` | CLI only (python -m adapters.code_graph.snapshot_cli export/import), no REST or MCP surface; a scenario needs a new exec form on the API container, which the harness's compose exec allowlist does not have (runctx.py is under open PR #526); covered by the real-Neo4j integration tests in integration_tests/test_native_snapshot_neo4j.py (one red, fix in #534) |
| `read.keyword-scan` | reached only inside /v1/ask medium and high: the response shows that the pass ran ('keyword:' in searches, asserted under read.ask-tiers) but not what it matched, and the answering LLM decides what it cites, so no black-box assertion separates keyword hits from semantic ones; the scan's matching and visibility rules are covered by the keyword_search tests in tests/test_ask_memory.py |
| `read.ask-evidence-budget` | the row cap and clip lengths shape the prompt text only; no response field exposes the rendered rows (memories_considered counts retrieved evidence, not rendered); covered by tests/test_ask_memory.py |
| `read.ask-second-chance` | fires only when the answering LLM abstains although keyword-overlapping evidence exists; a black-box run cannot force an abstention deterministically, so a scenario would test an LLM decision, not the pass; it shows in searches as 'second-chance ...' when it fires |
| `vec.mem0-posthog-telemetry` | mem0 PostHog telemetry: third-party telemetry is switched off in-process (third_party_telemetry.py sets the vendored libraries' env defaults before they import); from outside the process the only evidence would be the absence of outbound connections, which a one-instant socket snapshot cannot show, and the harness's read-only compose exec allowlist has no form that reads container sockets or environments (a new form means editing runctx.py, owned by open PR #526); (PR #451) merged with unit tests |
| `ops.settings-surface` | internal pydantic settings: each flag is proven by the scenario for the behaviour it switches (the flags-on, graph-supersession, dedup-tombstone and other profiles), the env templates are checked against the settings by open PR #498, and (planned) owns the env-name bug and the dead settings |
| `ops.vendored-posthog-telemetry` | vendored PostHog phone-home: third-party telemetry is switched off in-process (third_party_telemetry.py sets the vendored libraries' env defaults before they import); from outside the process the only evidence would be the absence of outbound connections, which a one-instant socket snapshot cannot show, and the harness's read-only compose exec allowlist has no form that reads container sockets or environments (a new form means editing runctx.py, owned by open PR #526); (PR #451) merged with unit tests |
| `auth.contracts-tenancy-readiness` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.common-primitives` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.policy` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.lifecycle` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.engines-inference` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.tenancy-readiness` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.portability` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.usage` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.context-freshness-processing` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.pack` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.evidence-records` | Phase 2 slice code: to be proven by scenarios as slices land |
| `contracts.persona-probes` | Phase 2 slice code: to be proven by scenarios as slices land |
| `ops.compose-profiles` | only the cbm and docling profile services have a stack scenario (test_code_graph_cbm, test_ingest/test_exemplars under their profiles, which claim their own nodes); the resolver, test and ingest-heavy profiles have none |
| `ext.dreaming-recall-traces` | traces are Redis keys written in-process on recall (log_recall) and no API reads them; their only visible effect is the salience tie-break; real-store proof is 's flag-on integration tests (open PR #483) |
| `ext.dreaming-code-liveness` | needs a repo whose symbols change between two index runs, but the only mapped repo is the image's own immutable source, and the reframe runs inside the dreaming sweep; covered by tests/test_code_liveness.py and tests/test_liveness_phase_e.py; flag-on integration proof in open PR #483 |
| `read.salience-tiebreak` | needs two memories whose scores are nearly tied, which a black-box run cannot construct; proven against the real stores by (open PR #529: integration_tests/test_salience_tiebreak_real_stores.py) |
| `ext.workspaces` | scenario ships in open PR #504 : acceptance/scenarios/test_workspace_recall.py covers the workspaces parameter over REST and MCP |
| `knowledge.cbm-import` | CLI only (adapters.code_graph.cbm_import) and partial: the importer targets an assumed schema, the reconcile with the real CBM schema was never done (#491 notes); no REST or MCP surface and no exec form |
| `read.cross-encoder-reranker` | nothing is implemented on dev (dropped): no module, flag or test; the flag-gated HYBRID_RERANK is a different feature |
| `archive.phase2-led-a` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-pol` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-eng` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-usg` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-led-b` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-led-c` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-onb` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-ten` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-e2ee` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-port` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-ctx-plg` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.phase2-evd` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.lightrag-c110` | Phase 2 slice code: to be proven by scenarios as slices land |
| `archive.single-copy-fixes` | Phase 2 slice code: to be proven by scenarios as slices land (#487 open, planned) |
| `archive.pr451` | merged as (PR #451); folded into the telemetry decision: third-party telemetry is switched off in-process (third_party_telemetry.py sets the vendored libraries' env defaults before they import); from outside the process the only evidence would be the absence of outbound connections, which a one-instant socket snapshot cannot show, and the harness's read-only compose exec allowlist has no form that reads container sockets or environments (a new form means editing runctx.py, owned by open PR #526); (PR #451) merged with unit tests |
| `archive.pr240-jev` | hybrid inference is flagged off and a real-model run needs an owner-supplied endpoint and key; the flag-on proof uses a local fake endpoint in integration_tests/test_hybrid_integration.py (#471, fix-forward #518) and the acceptance stack has no fake endpoint |
| `archive.pr148-graph-fallback` | held PR (owner hold, needs bench validation), not merged: no code on dev to run |
| `archive.pr226-access-model` | design document only (open docs PR #226): nothing to run |
