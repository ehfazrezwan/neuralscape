# NeuralScape Graphiti - Conversation Summary

## Project Overview

This repository (`neuralscape-graphiti`) is a fork/working copy of [Graphiti](https://github.com/getzep/graphiti), a Python framework for building temporally-aware knowledge graphs for AI agents. We've customized it to use **Google Gemini** (via AI Studio API key) for LLM/embeddings and **Neo4j Desktop** with a custom database named `memory`.

## Repository Structure

```
neuralscape-graphiti/
  .git/
  .gitignore              # Root-level gitignore (covers .env, .venv, etc.)
  .env                    # Root-level env (NEO4J + GOOGLE_API_KEY credentials)
  docs/
    conversation-summary.md   # This file
  graphiti/                   # All upstream graphiti source lives here
    graphiti_core/            # Core library (the Python package)
    server/                   # REST API server (FastAPI)
    mcp_server/               # MCP server (Model Context Protocol)
    examples/                 # Example scripts
    tests/                    # Test suite
    pyproject.toml            # Root package config for graphiti-core
    Makefile, README.md, etc.
```

The entire upstream graphiti codebase was moved into the `graphiti/` subdirectory to allow for a monorepo structure.

## Branch

All work is on the `bootstrap/setting-up` branch (forked from `main`).

## What Was Done

### 1. MCP Server: Neo4j Database Passthrough Fix

**Problem**: `DatabaseDriverFactory.create_config()` in `graphiti/mcp_server/src/services/factories.py` returned `{uri, user, password}` for Neo4j but omitted the `database` field, so it always defaulted to `neo4j`.

**Fix**: Added `'database': neo4j_config.database` to the returned dict.

**Problem**: The Neo4j branch in `graphiti/mcp_server/src/graphiti_mcp_server.py` passed `uri/user/password` directly to `Graphiti()`, which internally creates a `Neo4jDriver` with default `database='neo4j'`. The FalkorDB branch correctly created a `FalkorDriver` instance first.

**Fix**: Changed to create a `Neo4jDriver` instance directly (matching the FalkorDB pattern) and pass it via `graph_driver=`.

### 2. Cross-Encoder Support for Non-OpenAI Providers

**Problem**: `Graphiti.__init__()` (in `graphiti/graphiti_core/graphiti.py:221-224`) defaults ALL None clients to OpenAI variants:
- `llm_client=None` -> `OpenAIClient()`
- `embedder=None` -> `OpenAIEmbedder()`
- `cross_encoder=None` -> `OpenAIRerankerClient()`

This means if you pass `llm_client` and `embedder` but forget `cross_encoder`, it crashes with "OPENAI_API_KEY not set".

**Fix**: Both servers now explicitly create a `GeminiRerankerClient` when using Gemini as the LLM provider, using `graphiti_core.cross_encoder.gemini_reranker_client.GeminiRerankerClient`.

### 3. REST Server: Multi-Provider Support

**Files changed**: `graphiti/server/graph_service/config.py`, `graphiti/server/graph_service/zep_graphiti.py`

Previously the REST server was hardcoded to OpenAI. Now:

- `Settings` class has `llm_provider`, `embedding_provider`, `google_api_key`, `small_model_name`, `neo4j_database` fields. `openai_api_key` is now optional.
- `ZepGraphiti.__init__` accepts `graph_driver=`, `embedder=`, `cross_encoder=` params (forwarded to `Graphiti.__init__`).
- Helper functions `_create_llm_client()`, `_create_embedder()`, `_create_cross_encoder()` dispatch on provider setting.
- Both `get_graphiti()` and `initialize_graphiti()` create a `Neo4jDriver` with `database=settings.neo4j_database`.
- No router files needed changes - they use `ZepGraphitiDep` which is unaffected.

### 4. MCP Server Config Switched to Gemini + Neo4j

**File**: `graphiti/mcp_server/config/config.yaml`

- `llm.provider`: `openai` -> `gemini`
- `llm.model`: `gpt-4o-mini` -> `gemini-3-pro-preview`
- `embedder.provider`: `openai` -> `gemini`
- `embedder.model`: `text-embedding-3-small` -> `gemini-embedding-001`
- `embedder.dimensions`: `1536` -> `1024`
- `database.provider`: `falkordb` -> `neo4j`

### 5. Local graphiti-core via Editable Path

Both `graphiti/server/pyproject.toml` and `graphiti/mcp_server/pyproject.toml` have:

```toml
[tool.uv.sources]
graphiti-core = { path = "..", editable = true }
```

This points to the local `graphiti/` directory (which contains `graphiti_core/` and `pyproject.toml`). This is necessary because the PyPI version of `graphiti-core` (0.28.0) does NOT have the `database` parameter on `Neo4jDriver.__init__` - the local repo version does.

The `google-genai` extra was also added to both dependency specs:
- Server: `graphiti-core[google-genai]`
- MCP Server: `graphiti-core[falkordb,google-genai]`

### 6. Repository Restructuring

All files moved from repo root into `graphiti/` subdirectory via `git mv`. A copy of `.gitignore` remains at the repo root.

## Commit History

```
e0cd131 refactor: move all repository contents into graphiti/ subdirectory
232a4a5 chore: switch to Gemini + Neo4j and use local graphiti-core
174e166 feat: add multi-provider LLM, embedder, and cross-encoder support to REST server
696771d fix: pass Neo4j database param through MCP server and add cross-encoder support
ab7f316 fix: route execute_query to correct database when using custom db name  (pre-existing)
```

## Environment Setup

### Prerequisites

- **Python 3.12** (managed by uv)
- **uv** package manager (`~/.local/bin/uv`)
- **Neo4j Desktop** with a database named `memory` running on `neo4j://127.0.0.1:7687`
- **Google API Key** for Gemini (AI Studio)

### .env Files

These are gitignored. Three locations:

**Root `.env`** (used by `graphiti/examples/quickstart/quickstart_memory.py`):
```env
GOOGLE_API_KEY=<key>
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

**`graphiti/mcp_server/.env`**:
```env
GOOGLE_API_KEY=<key>
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

**`graphiti/server/.env`**:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<key>
MODEL_NAME=gemini-3-pro-preview
SMALL_MODEL_NAME=gemini-3-flash-preview
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL_NAME=gemini-embedding-001
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

### Setting Up Virtual Environments

Each server has its own `.venv`. From the repo root:

```bash
# REST server
cd graphiti/server
uv sync --extra dev

# MCP server
cd graphiti/mcp_server
uv sync
```

**Important**: Always use each server's own `.venv` when running commands:
- REST server: `graphiti/server/.venv/bin/python`, `graphiti/server/.venv/bin/uvicorn`
- MCP server: `graphiti/mcp_server/.venv/bin/python`

### Running the Servers

Both default to port 8000, so run one at a time unless you override the port.

**REST Server**:
```bash
cd graphiti/server
.venv/bin/uvicorn graph_service.main:app --host 127.0.0.1 --port 8000 --reload
# Healthcheck: curl http://127.0.0.1:8000/healthcheck
# Returns: {"status":"healthy"}
```

**MCP Server**:
```bash
cd graphiti/mcp_server
.venv/bin/python main.py
# Healthcheck: curl http://localhost:8000/health
# Returns: {"status":"healthy","service":"graphiti-mcp"}
# MCP endpoint: http://localhost:8000/mcp/
```

## Key Learnings / Gotchas

1. **Graphiti defaults everything to OpenAI**: If you pass `None` for `llm_client`, `embedder`, or `cross_encoder`, `Graphiti.__init__` creates OpenAI defaults. You must pass ALL three explicitly when using a non-OpenAI provider.

2. **Neo4jDriver `database` param**: Only available in the local/latest version of graphiti-core, not in the PyPI 0.28.0 release. This is why we use `[tool.uv.sources]` to point at the local repo.

3. **Each server has its own .venv**: Don't use the root `.venv` or cross-use venvs between servers. The root `.venv` is for the graphiti-core library itself.

4. **Config flow**:
   - REST server: `.env` -> `pydantic_settings` `Settings` class -> helper functions create clients -> `Neo4jDriver` + `ZepGraphiti`
   - MCP server: `.env` + `config/config.yaml` -> `GraphitiConfig` -> factory classes create clients -> `Neo4jDriver` + `Graphiti`

5. **MCP server config.yaml supports env var expansion**: Values like `${GOOGLE_API_KEY}` in the YAML are expanded from environment variables (loaded from `.env`).

6. **Available Gemini cross-encoder**: `graphiti_core.cross_encoder.gemini_reranker_client.GeminiRerankerClient` - scores passages on a 0-100 scale (Gemini doesn't support logprobs like OpenAI).

7. **Quickstart reference**: `graphiti/examples/quickstart/quickstart_memory.py` demonstrates the correct Gemini + Neo4j pattern with all components wired up properly.

## Files Modified (from upstream)

| File | What Changed |
|------|-------------|
| `graphiti/mcp_server/src/services/factories.py` | Added `database` to Neo4j config dict |
| `graphiti/mcp_server/src/graphiti_mcp_server.py` | Neo4jDriver pattern + cross-encoder creation |
| `graphiti/mcp_server/config/config.yaml` | Switched to gemini/neo4j providers |
| `graphiti/mcp_server/pyproject.toml` | Added google-genai extra + uv source override |
| `graphiti/server/graph_service/config.py` | Added provider fields, made openai_api_key optional |
| `graphiti/server/graph_service/zep_graphiti.py` | Multi-provider support, graph_driver param, Neo4jDriver |
| `graphiti/server/pyproject.toml` | Added google-genai extra + uv source override |
