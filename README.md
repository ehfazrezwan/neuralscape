# Neuralscape - Agentic Memory Layer

A production-grade memory system for AI coding assistants and personal agents. Neuralscape gives any LLM-powered agent persistent, structured memory across sessions and projects — remembering user preferences, project conventions, technical decisions, and learned facts.

Built on [mem0](https://github.com/mem0ai/mem0) (vector storage + LLM deduplication) and [Graphiti](https://github.com/getzep/graphiti) (temporal knowledge graph), exposed via REST API and MCP server.

## How It Works

```
                    ┌──────────────────────────────────────────────┐
                    │           neuralscape-service                │
                    │                                              │
  Claude Code ────► │  MCP Server (7 tools)   REST API (/v1)      │
  Any Agent ──────► │       stdio / HTTP         FastAPI           │
                    │              │                │              │
                    │              └──────┬─────────┘              │
                    │                    │                         │
                    │             MemoryService                    │
                    │          (business logic layer)              │
                    └──────────────┬──────────────┬────────────────┘
                                   │              │
                        ┌──────────▼──┐    ┌──────▼───────┐
                        │   Qdrant    │    │    Neo4j     │
                        │  (vectors)  │    │  (Graphiti   │
                        │  on-disk    │    │   graph)     │
                        └─────────────┘    └──────────────┘
```

Every memory is stored **twice**: as a vector embedding in Qdrant (for semantic search) and as entities/relationships in a Neo4j knowledge graph via Graphiti (for structured reasoning). Both paths are queried on every search and results are merged.

## Core Concepts

### Two-Scope Namespace

Memories live in one of two scopes:

| Scope | Graphiti group_id | Purpose |
|---|---|---|
| **Global** | `"global"` | Cross-project facts: user preferences, skills, personal details |
| **Project** | `"project:{slug}"` | Project-specific: tech stack, conventions, architecture decisions |

When you search with a `project_id`, Neuralscape searches **both** scopes and merges results by relevance score. An agent working on `neuralscape-graphiti` sees your global "prefers 4-space indentation" preference alongside the project-specific "uses FastAPI with Graphiti backend" fact.

### 13 Memory Categories

Since self-hosted mem0 has no native category system, every memory gets a `category` metadata field that controls scope defaults and enables filtered retrieval:

| Group | Categories | Default Scope |
|---|---|---|
| **Semantic** | `preference`, `personal_fact`, `technical_skill`, `domain_knowledge` | Global |
| **Project** | `tech_stack`, `convention`, `architecture`, `dependency` | Project |
| **Episodic** | `decision`, `interaction` | Flexible |
| **Procedural** | `workflow`, `procedure` | Flexible |
| **Working** | `task_context` | Flexible |

### Custom LLM Extraction

When an agent sends a conversation to `POST /v1/memories`, Neuralscape doesn't just pass it through to mem0. Instead:

1. It calls Gemini with a specialized extraction prompt
2. The LLM returns facts tagged with categories: `[preference] Prefers tabs over spaces`
3. Each fact is parsed and stored with proper scope/category metadata via `mem0.add(infer=False)`
4. The raw conversation is also fed to Graphiti's knowledge graph for entity/relationship extraction

This gives you categorized vector memories **and** a rich knowledge graph from the same input.

### Agent Isolation

`agent_id` is metadata for provenance tracking, not a scope boundary. All agents (Claude Code, a Cursor plugin, a custom bot) share the same memory space for a given user. Conflicts are handled by Graphiti's temporal edge invalidation (old facts get `invalid_at` timestamps) and mem0's LLM-based deduplication.

## REST API

All new endpoints live under `/v1`. Legacy endpoints at root are preserved for backward compatibility.

### Remember

```bash
# Extract and store from conversation (LLM-powered)
curl -X POST http://localhost:8199/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "I use Python 3.12 with FastAPI"}],
    "user_id": "ehfaz",
    "project_id": "neuralscape-graphiti"
  }'

# Store a single pre-categorized fact (no LLM)
curl -X POST http://localhost:8199/v1/memories/raw \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Prefers 4-space indentation",
    "user_id": "ehfaz",
    "category": "preference"
  }'

# Non-blocking version (returns task_id)
curl -X POST http://localhost:8199/v1/memories/async \
  -H "Content-Type: application/json" \
  -d '{"messages": [...], "user_id": "ehfaz"}'

# Poll async status
curl http://localhost:8199/v1/memories/status/{task_id}
```

### Recall

```bash
# Semantic search (searches global + project when project_id given)
curl -X POST http://localhost:8199/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "indentation style",
    "user_id": "ehfaz",
    "project_id": "neuralscape-graphiti",
    "categories": ["preference", "convention"],
    "limit": 10
  }'

# Knowledge graph search
curl -X POST http://localhost:8199/v1/graph/search \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI", "user_id": "ehfaz"}'
```

### Context Loading

```bash
# Full project context (global prefs + project facts, organized by category)
curl "http://localhost:8199/v1/context/neuralscape-graphiti?user_id=ehfaz"

# Global-only context
curl "http://localhost:8199/v1/context/global?user_id=ehfaz"
```

### Manage

```bash
# List with filters
curl "http://localhost:8199/v1/memories?user_id=ehfaz&scope=global&category=preference"

# Get/update/delete single memory
curl http://localhost:8199/v1/memories/{id}
curl -X PUT http://localhost:8199/v1/memories/{id} -d '{"content": "..."}'
curl -X DELETE http://localhost:8199/v1/memories/{id}

# List available categories
curl http://localhost:8199/v1/categories

# Graph introspection
curl "http://localhost:8199/v1/graph/nodes?user_id=ehfaz&project_id=neuralscape-graphiti"
curl "http://localhost:8199/v1/graph/edges?user_id=ehfaz"
curl "http://localhost:8199/v1/graph/episodes?user_id=ehfaz"
curl "http://localhost:8199/v1/graph/communities?user_id=ehfaz"
```

## MCP Tools

7 tools exposed via MCP for direct use by AI agents:

| Tool | Purpose |
|---|---|
| `recall_memories` | Semantic search across global + project memories. Agents should call this before starting work. |
| `remember` | Store a single categorized fact (agent provides content + category). |
| `remember_conversation` | Bulk extract from conversation messages via LLM. |
| `get_project_context` | Bootstrap: load all user prefs + project context organized by category. |
| `search_knowledge_graph` | Graph-based entity/relationship search. |
| `list_memories` | List/inspect stored memories with filters. |
| `delete_memories` | Delete by ID or by filters. |

### Transport

- **stdio** (default): For local Claude Code. Configure in Claude Code's MCP settings.
- **Streamable HTTP**: Set `MCP_TRANSPORT=http` to mount at `/mcp/` on port 8199 for remote agents.

### Claude Code Configuration

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "neuralscape-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/neuralscape-service", "python", "mcp_server.py"]
    }
  }
}
```

## Project Structure

```
neuralscape-graphiti/
├── neuralscape-service/          # The service (what you deploy)
│   ├── main.py                   # FastAPI app: legacy + v1 endpoints
│   ├── memory_service.py         # Business logic layer (MemoryService class)
│   ├── mcp_server.py             # MCP server: 7 tools, stdio + HTTP
│   ├── schemas.py                # Enums, category taxonomy, Pydantic models
│   ├── prompts.py                # LLM extraction prompt, category parser
│   ├── config.py                 # Pydantic settings (env-driven)
│   ├── .env                      # Environment variables
│   ├── pyproject.toml            # Dependencies
│   └── tests/
│       ├── test_service.py       # REST endpoint tests (legacy + v1)
│       ├── test_memory_service.py # Business logic tests
│       └── test_mcp_tools.py     # MCP tool tests
├── mem0/                         # mem0 fork (local editable)
│   └── mem0/memory/
│       └── graphiti_memory.py    # Graphiti adapter (modified for scoping)
└── graphiti/                     # graphiti-core fork (local editable)
```

## Prerequisites

- **Python 3.10+**
- **Neo4j** running at `neo4j://127.0.0.1:7687` (for Graphiti knowledge graph)
- **Google API key** with Gemini access (for LLM extraction + embeddings)

Qdrant runs embedded (on-disk at `~/.neuralscape/qdrant`) — no separate server needed.

## Setup

```bash
cd neuralscape-service

# Create .env file
cat > .env << 'EOF'
GOOGLE_API_KEY=your-gemini-api-key
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=memory
EOF

# Install dependencies
uv sync

# Run the service
uv run python main.py
# → Listening on http://0.0.0.0:8199

# Run tests
uv run python -m pytest tests/ -v
```

## Configuration

All settings are environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | | Gemini API key |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | Model for LLM extraction |
| `GEMINI_EMBEDDER_MODEL` | `text-embedding-004` | Model for embeddings |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | | Neo4j password |
| `NEO4J_DATABASE` | `memory` | Neo4j database name |
| `QDRANT_ON_DISK` | `true` | Persist Qdrant to disk |
| `QDRANT_PATH` | `~/.neuralscape/qdrant` | Qdrant storage path |
| `QDRANT_COLLECTION` | `neuralscape_memories` | Qdrant collection name |
| `PORT` | `8199` | Service port |
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` or `http` |

## Architecture Decisions

**Why custom extraction instead of mem0's built-in?** Self-hosted mem0 doesn't support categories. By doing extraction in our service layer, we tag each fact with a category *before* storage, enabling filtered retrieval and organized context loading.

**Why two storage backends?** Vector search (Qdrant) is for "find memories similar to this query." Knowledge graph (Graphiti/Neo4j) is for "what entities are related to X?" and handles temporal fact invalidation (when facts change over time). Together they provide comprehensive recall.

**Why group_id-based scoping instead of separate databases?** Graphiti partitions data by `group_id` within a single Neo4j database. Using composite IDs (`"global"`, `"project:my-app"`) keeps the infrastructure simple while providing proper namespace isolation. Multi-scope search just queries multiple group_ids.

**Why not use agent_id as a scope boundary?** Multiple agents (Claude Code, a Slack bot, a CI pipeline) should all benefit from the same memory. Agent isolation would fragment knowledge. Instead, `agent_id` is provenance metadata — you can see *who* learned a fact but everyone can use it.
