# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mem0 is a memory layer for AI applications. It provides APIs to store, search, and manage memories across user sessions, agents, and runs. The core library supports 25+ vector stores, 15+ LLM providers, multiple graph databases, and offers both sync and async Python interfaces plus a TypeScript SDK.

## Development Commands

### Python Package (run from repo root)

```bash
make format           # Format with ruff
make sort             # Sort imports with isort
make lint             # Lint with ruff
make test             # Run all tests (pytest)
make build            # Build package (hatch build)
```

Individual test commands:
```bash
pytest tests/test_main.py                    # Specific file
pytest tests/test_main.py::test_method       # Specific test
pytest tests/ -k "not integration"           # Skip integration tests
```

### REST API Server (`server/`)

```bash
cd server
make build            # Docker build
make run_local        # Docker run with volume mount
```

### OpenMemory Full-Stack App (`openmemory/`)

```bash
cd openmemory
make env              # Copy .env.example files
make build            # Docker compose build
make up               # Start all containers
make down             # Stop containers and clean volumes
make upgrade          # Run alembic migrations
make ui-dev           # Run Next.js frontend in dev mode
```

### TypeScript SDK (`mem0-ts/`)

```bash
cd mem0-ts
pnpm install
pnpm build            # Build with tsup
pnpm test             # Run Jest tests
pnpm format           # Format with prettier
```

## Code Architecture

### Core Library (`mem0/`)

The library uses a **factory pattern** throughout. Provider selection happens at runtime via config:

- `mem0/utils/factory.py` — `LlmFactory`, `EmbedderFactory`, `VectorStoreFactory`, `GraphStoreFactory`, `RerankerFactory` instantiate the correct provider based on config strings.
- `mem0/configs/base.py` — `MemoryConfig` is the top-level Pydantic config that composes `LlmConfig`, `EmbedderConfig`, `VectorStoreConfig`, `GraphStoreConfig`, `RerankerConfig`.

**Main entry points** (`mem0/__init__.py` exports):
- `Memory` / `AsyncMemory` — Local memory management (`mem0/memory/main.py`, ~2300 lines). Creates vector store, LLM, embedder, and optionally graph store instances via factories.
- `MemoryClient` / `AsyncMemoryClient` — API client for the hosted Mem0 platform (`mem0/client/main.py`).

**Memory operations flow**: `add()` takes messages, extracts facts via LLM, generates embeddings, stores in vector DB, and optionally builds graph relationships. `search()` does embedding-based retrieval with optional reranking.

**Storage layers**:
- `mem0/memory/storage.py` — `SQLiteManager` for local history persistence (`~/.mem0/history.db`)
- `mem0/vector_stores/` — 25+ vector store backends (each in its own file)
- `mem0/graphs/` — Graph database integrations (Neo4j, Memgraph, Kuzu, Neptune)

**Provider directories** (each has a base class + provider implementations):
- `mem0/llms/` — LLM providers (openai, anthropic, gemini, groq, ollama, etc.)
- `mem0/embeddings/` — Embedding providers
- `mem0/reranker/` — Reranking providers
- `mem0/configs/` — Pydantic config classes organized by provider type

### Multi-Level Memory

Memories are scoped by `user_id`, `agent_id`, and `run_id` (session). These act as filters across all operations (add, search, get_all, delete, reset).

### Other Components

- `server/main.py` — FastAPI REST API wrapping the `Memory` class
- `openmemory/` — Full-stack app with FastAPI backend (`api/`) and Next.js frontend (`ui/`)
- `mem0-ts/` — TypeScript SDK (pnpm, tsup, Jest)
- `embedchain/` — Separate RAG framework (has its own pyproject.toml and poetry config)

## Build System

- **Python**: Hatch (configured in `pyproject.toml`). Environments for dev, test, and multi-Python-version testing.
- **TypeScript**: pnpm + tsup
- **Linting**: ruff (line length 120, excludes `embedchain/` and `openmemory/`)
- **Import sorting**: isort (profile: black)

## Testing

Tests are in `tests/` using pytest. Extensive use of `unittest.mock`. Key test files:
- `test_main.py` — Core Memory class tests (~12K lines)
- `test_memory.py` — Memory operations (~10K lines)
- `test_memory_integration.py` — Integration tests (~7K lines)
- Subdirectories for `llms/`, `embeddings/`, `vector_stores/`, `configs/`

CI runs tests on Python 3.10, 3.11, 3.12.
