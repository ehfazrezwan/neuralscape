# Graphiti-backed mem0 Integration -- Implementation Plan

## Overview

This document covers the full implementation of Graphiti as a graph memory backend for mem0, including a lightweight FastAPI+MCP service for AI client integration.

**Key insight**: Both mem0 and Graphiti perform LLM-based entity extraction. Graphiti's pipeline is far more sophisticated (temporal tracking, deduplication, community detection, edge invalidation). The adapter delegates extraction entirely to Graphiti and bypasses mem0's built-in extraction.

**group_id safety**: `Neo4jDriver.clone()` returns `self` (no-op), so passing `user_id` as `group_id` does NOT create new Neo4j databases. Multi-tenant partitioning happens within a single database.

---

## 1. Prerequisites

### Required Software
- Python 3.10+
- Neo4j Desktop 5.26+ (or Docker)
- Google AI Studio API key (for Gemini)

### Neo4j Setup

**Neo4j Desktop:**
1. Download and install Neo4j Desktop
2. Create a new project and database
3. Name the database `memory`
4. Start the database
5. Note credentials (default: `neo4j` / your-password)

**Docker:**
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5.26
```

Then create the database:
```cypher
CREATE DATABASE memory
```

### Environment Variables
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export NEO4J_PASSWORD="your-neo4j-password"
```

---

## 2. Code Changes

### 2a. GraphitiConfig (`mem0/mem0/graphs/configs.py`)

Added `GraphitiConfig` class with all configuration fields:
- Neo4j connection: `url`, `username`, `password`, `database`
- LLM config: `graphiti_llm_provider`, `graphiti_llm_model`, `graphiti_llm_api_key`
- Embedder config: `graphiti_embedder_provider`, `graphiti_embedder_model`, `graphiti_embedder_api_key`
- Reranker: `graphiti_reranker_provider`
- Options: `store_raw_episode_content`, `update_communities`

Updated `GraphStoreConfig`:
- Added `"graphiti"` to provider description
- Added `GraphitiConfig` to config Union type (first in Union order for correct matching)
- Added validation branch for `provider == "graphiti"`

### 2b. Factory Registration (`mem0/mem0/utils/factory.py`)

Added to `GraphStoreFactory.provider_to_class`:
```python
"graphiti": "mem0.memory.graphiti_memory.MemoryGraph",
```

### 2c. Core Adapter (`mem0/mem0/memory/graphiti_memory.py`)

**Class: `MemoryGraph`** (~280 lines)

Architecture:
1. **Client creation**: Helper functions create Graphiti LLM, embedder, and cross-encoder clients from config strings
2. **Async bridge**: `_EventLoopThread` class manages per-thread event loops for sync-to-async
3. **Method mapping**:
   - `add()` -> `Graphiti.add_episode()` -> returns `{"deleted_entities": [], "added_entities": [...]}`
   - `search()` -> `Graphiti.search()` -> resolves node UUIDs to names -> returns triple dicts
   - `delete_all()` -> `Node.delete_by_group_id()` -> removes all data for a group
   - `get_all()` -> `EntityEdge.get_by_group_ids()` -> resolves names -> returns triple dicts
4. **group_id mapping**: `user_id` -> `group_id` for multi-tenant partitioning
5. **Exposed Graphiti**: `self.graphiti` is public for advanced usage

### 2d. Optional Dependency (`mem0/pyproject.toml`)

Added:
```toml
graphiti = [
    "graphiti-core[google-genai]>=0.28.0",
]

[tool.uv.sources]
graphiti-core = { path = "../graphiti", editable = true }
```

---

## 3. Lightweight Service (`neuralscape-service/`)

### Architecture

```
neuralscape-service/
  main.py          # FastAPI REST endpoints + advanced graph endpoints
  mcp_server.py    # MCP tools for AI clients (stdio transport)
  config.py        # Pydantic Settings from .env
  .env             # Environment configuration
  pyproject.toml   # Dependencies (local graphiti-core + mem0)
```

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/memories` | Add memory (vector + graph) |
| POST | `/search` | Search memories |
| GET | `/memories` | List all memories |
| DELETE | `/memories` | Delete all memories |
| GET | `/graph/nodes` | List entity nodes |
| GET | `/graph/edges` | List entity edges/facts |
| GET | `/graph/episodes` | List episodes |
| GET | `/graph/communities` | List communities |
| POST | `/graph/search` | Advanced search with SearchConfig |

### MCP Tools

| Tool | Description |
|------|-------------|
| `add_memories` | Store new memory |
| `search_memory` | Find relevant memories |
| `list_memories` | View all stored memories |
| `delete_memories` | Remove memories |
| `search_graph` | Advanced Graphiti graph search |

### Running

```bash
cd neuralscape-service
uv sync
python main.py       # REST at http://localhost:8199
python mcp_server.py # MCP via stdio
```

---

## 4. Configuration Examples

### Minimal (Gemini + Neo4j)

```python
from mem0 import Memory

config = {
    "graph_store": {
        "provider": "graphiti",
        "config": {
            "url": "neo4j://127.0.0.1:7687",
            "username": "neo4j",
            "password": "your-password",
            "database": "memory",
            "graphiti_llm_provider": "gemini",
            "graphiti_llm_api_key": "your-gemini-key",
            "graphiti_embedder_provider": "gemini",
            "graphiti_embedder_api_key": "your-gemini-key",
        },
    },
    "version": "v1.1",
}

m = Memory.from_config(config)
```

---

## 5. Testing Workflow

### Step 1: Import Validation
```python
from mem0.memory.graphiti_memory import MemoryGraph
from mem0.graphs.configs import GraphitiConfig
print("Imports OK")
```

### Step 2: Integration Test (Neo4j + Gemini)
```python
import os
from mem0 import Memory

config = {
    "graph_store": {
        "provider": "graphiti",
        "config": {
            "url": "neo4j://127.0.0.1:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": "memory",
            "graphiti_llm_provider": "gemini",
            "graphiti_llm_api_key": os.getenv("GOOGLE_API_KEY"),
            "graphiti_embedder_provider": "gemini",
            "graphiti_embedder_api_key": os.getenv("GOOGLE_API_KEY"),
        },
    },
    "version": "v1.1",
}

m = Memory.from_config(config)
test_user = "integration_test"

result = m.add("Alice is a research engineer at Anthropic.", user_id=test_user)
results = m.search("Where does Alice work?", user_id=test_user)
all_facts = m.get_all(user_id=test_user)
m.delete_all(user_id=test_user)
```

### Step 3: Service Test
```bash
curl http://localhost:8199/health
curl -X POST http://localhost:8199/memories -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I love hiking"}], "user_id": "test"}'
curl -X POST http://localhost:8199/search -H "Content-Type: application/json" \
  -d '{"query": "hobbies", "user_id": "test"}'
```

### Step 4: Verify No mem0 Main Changes Needed

The existing `_add_to_graph()` / `_search_graph()` paths in `Memory` work unchanged.

---

## 6. Advanced Graphiti Features

All accessible via `graphiti = m.graph.graphiti`:

- **Temporal queries**: `SearchFilters` with `valid_at` / `invalid_at`
- **Communities**: `graphiti.build_communities(group_ids=[...])`
- **Sagas**: `graphiti.add_episode(..., saga="name")`
- **16+ search recipes**: `EDGE_HYBRID_SEARCH_RRF`, `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`, etc.
- **Custom entity types**: Pydantic models via `entity_types` param
- **Manual triplets**: `graphiti.add_triplet(source_node, edge, target_node)`
- **Full SearchConfig**: `graphiti.search_(query, config=SearchConfig(...))`

---

## 7. MCP Client Configuration

### Claude Desktop
```json
{
  "mcpServers": {
    "neuralscape-memory": {
      "command": "python",
      "args": ["/path/to/neuralscape-service/mcp_server.py"],
      "env": { "GOOGLE_API_KEY": "key", "NEO4J_PASSWORD": "pw" }
    }
  }
}
```

---

## Files Summary

| File | Action |
|------|--------|
| `mem0/mem0/memory/graphiti_memory.py` | CREATE |
| `mem0/mem0/graphs/configs.py` | MODIFY |
| `mem0/mem0/utils/factory.py` | MODIFY |
| `mem0/pyproject.toml` | MODIFY |
| `neuralscape-service/main.py` | CREATE |
| `neuralscape-service/mcp_server.py` | CREATE |
| `neuralscape-service/config.py` | CREATE |
| `neuralscape-service/.env` | CREATE |
| `neuralscape-service/pyproject.toml` | CREATE |
| `docs/graphiti-mem0-implementation.md` | CREATE |
| `~/.claude/skills/mem0-graphiti/SKILL.md` | CREATE |
| `~/.claude/skills/mem0-graphiti/references/*.md` | CREATE (3 files) |
| `~/.claude/skills/mem0-graphiti/examples/*.md` | CREATE (2 files) |
