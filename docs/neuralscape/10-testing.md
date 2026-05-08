---
title: Testing Strategy
date: 2026-05-06
tags: [reference, neuralscape, testing, pytest]
source: handwritten
---

# Testing Strategy

## Overview

Neuralscape ships ~4,852 LOC of pytest tests across 11 files in `neuralscape-service/tests/`, covering ~1.6k LOC of service code. The strategy is bimodal: most tests are **unit tests** that mock every external dependency (Gemini, Qdrant, Neo4j, Redis) and run in milliseconds, while a single file (`test_async_pipeline.py`) drives **integration tests** against a fully-running stack (API on `:8199`, ARQ worker, Redis, Qdrant, Neo4j). `conftest.py` is intentionally minimal — only seven lines that prepend the parent directory to `sys.path` — so all fixtures live inline with the tests that use them. There is no coverage tooling and no CI threshold; correctness is enforced by review and by exercising the async pipeline end-to-end before release.

## Test inventory

| File | LOC | Purpose |
|------|-----|---------|
| `conftest.py` | 7 | Adds parent dir to `sys.path` |
| `test_service.py` | 496 | FastAPI endpoint tests (legacy + v1) via `TestClient` with mocks |
| `test_memory_service.py` | 876 | Core [04-memory-service-core](./04-memory-service-core.md) unit tests — helpers, CRUD, batch extraction, dedup, graph re-ingest, junk cleanup |
| `test_async_pipeline.py` | 284 | Integration tests; needs Redis, Qdrant, Neo4j, API, worker |
| `test_dedup.py` | 335 | Qdrant dedup — pagination, hash detection, semantic re-rank, `IsNullCondition` |
| `test_mcp_tools.py` | 282 | [08-mcp-server](./08-mcp-server.md)'s 7 tools; mocked `MemoryService` / `TaskManager` |
| `test_auth.py` | 166 | `BearerAuthMiddleware` on/off; health endpoints stay public |
| `test_production_readiness.py` | 990 | Config validation, Redis URL parser, Redis fallback, exception handler, thread safety, retry backoff |
| `test_conversation_compiler.py` | 823 | `ObsidianWriter` atomicity, flush extraction, compile grouping, lint |
| `test_extension_registry.py` | 593 | Extension lifecycle, dispatch, route mounting, `/v1/extensions` endpoint |

`__init__.py` is empty.

## Fixtures

All fixtures are inline in their test files (function-scoped — there are no session fixtures). Two files use `autouse=True` so every test in the module is patched without explicit injection.

| Fixture | Scope | File:Line | Purpose |
|---------|-------|-----------|---------|
| `mock_memory` | function | `neuralscape-service/tests/test_service.py:17-38` | Patches `main._memory`, `_graphiti`, `_bridge`, `_async_memory` |
| `mock_service` | function | `neuralscape-service/tests/test_service.py:42-48` | Patches `main._service` |
| `mock_task_manager` | function | `neuralscape-service/tests/test_service.py:52-68` | `AsyncMock` for `enqueue_*` and `get_status` |
| `client` | function | `neuralscape-service/tests/test_service.py:72-73` | `TestClient(app, raise_server_exceptions=False)` |
| `service` | function | `neuralscape-service/tests/test_memory_service.py:50-61` | `MemoryService` with mocked `_memory` / `_graphiti` / `_bridge` |
| `mock_mcp_service` | function (autouse) | `neuralscape-service/tests/test_mcp_tools.py:13-19` | Patches `mcp_server._service` |
| `mock_task_manager` | function (autouse) | `neuralscape-service/tests/test_mcp_tools.py:23-37` | Patches `mcp_server._task_manager` |
| `tmp_vault` | function | `neuralscape-service/tests/test_conversation_compiler.py:46-50` | Temp Obsidian vault via `tmp_path` |
| `writer` | function | `neuralscape-service/tests/test_conversation_compiler.py:54-56` | `ObsidianWriter` rooted at `tmp_vault` |

## Mocking strategy

- **Gemini** — `MagicMock()` is assigned to `service._genai_model`. `models.generate_content()` returns a `MagicMock(text='{"facts": [...]}')` with hardcoded JSON, so [05-llm-extraction](./05-llm-extraction.md) is exercised without API calls.
- **Qdrant** — mocks `service._memory.vector_store.{insert, search, delete, client.scroll}`; `embedding_model` returns `[0.1] * 768` to satisfy dimensionality checks.
- **Neo4j / Graphiti** — mocks `service._graphiti`, `_bridge`, `_memory.graph.graphiti`; `driver.execute_query` is mocked at the [06-storage-backends](./06-storage-backends.md) boundary.
- **Redis / ARQ** — `mock_task_manager` is an `AsyncMock`; integration tests use real Redis instead.
- **HTTP** — unit tests use FastAPI's `TestClient`; integration tests use `httpx.Client` against `http://localhost:8199`.
- **Test data** — embedded inline; there are no golden files and no `fixtures/` directory.

## Integration tests

`test_async_pipeline.py` is the only file gated behind running services. It requires the full stack (Neo4j, Redis, Qdrant, the FastAPI app on `:8199`, and an ARQ worker). The module-level marker auto-skips when the API is unreachable (`neuralscape-service/tests/test_async_pipeline.py:37-43`):

```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not server_is_reachable(), reason="API server not running"),
]
```

`server_is_reachable()` pings `/health`. Async writes return `202` with a `task_id`, so the helper at `neuralscape-service/tests/test_async_pipeline.py:46-57` polls until terminal state:

```python
def poll_until_done(task_id: str) -> dict:
    url = f"{BASE_URL}/v1/memories/status/{task_id}"
    deadline = time.time() + 300
    while time.time() < deadline:
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(5)
    pytest.fail(f"Task {task_id} did not complete within 300s")
```

Test classes:

- `TestHealth` (65-69) — `/health` returns 200 with `status == "ok"`.
- `TestAsyncRawMemoryStore` (77-142) — POST → 202 → `task_id` → completed.
- `TestAsyncConversationExtraction` (150-187) — POST → 202 → extraction completes.
- `TestTaskStatusPolling` (195-227) — observes `queued` → `processing` → `completed` transitions.
- `TestSyncSearch` (234-248) — sync search continues to work alongside async writes.
- `TestLegacyAsyncEndpoint` (256-284) — `/memories/async` backward compatibility.

See [07-async-pipeline](./07-async-pipeline.md) for how the worker drains these tasks.

## Markers

Defined in `neuralscape-service/pyproject.toml:44-47`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: integration tests requiring running services (Redis, Qdrant, Neo4j, API, Worker)",
]
```

Implicit markers: `@pytest.mark.asyncio` on async test methods (provided by `pytest-asyncio` in dev deps) and `@pytest.mark.skipif(...)` on the integration module.

## Run commands

| Goal | Command |
|------|---------|
| Unit tests (no services) | `uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v` |
| Single file | `uv run pytest tests/test_dedup.py -v` |
| Single test by name | `uv run pytest tests/test_service.py -k "test_search_memories" -v` |
| Integration (needs stack) | `uv run pytest tests/test_async_pipeline.py -v -s` |
| Skip integration explicitly | `uv run pytest tests/ -m "not integration" -v` |

All commands run from `neuralscape-service/`.

## Gaps

- **No coverage enforcement** — `pytest-cov` is not installed, there is no `.coveragerc`, and no CI threshold gates merges. The ~3:1 test-to-source LOC ratio is reassuring but unverified.
- **Plugin layer untested** — `neuralscape-plugin/` (TypeScript) has **no automated tests**. Per its README (lines 142-153), validation is manual stdin piping:

```bash
echo '{"user_message":"hello",...}' | node scripts/conversation-turn.js
```

The plugin's flows are covered transitively through the service's integration tests rather than directly. See [09-plugin-system](./09-plugin-system.md) for the surface area that this leaves uncovered.

## Related

- [07-async-pipeline](./07-async-pipeline.md)
- [04-memory-service-core](./04-memory-service-core.md)
- [02-service-architecture](./02-service-architecture.md)
- [11-deployment](./11-deployment.md)
- [08-mcp-server](./08-mcp-server.md)
