# Graphiti System Guide

A comprehensive reference for the Graphiti temporal knowledge graph framework — covering architecture, setup, the core Python library, REST API server, MCP server, search system, and integration patterns.

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Model](#3-data-model)
4. [Setup Guide](#4-setup-guide)
5. [Core Library Usage](#5-core-library-usage)
6. [REST API Server](#6-rest-api-server)
7. [MCP Server](#7-mcp-server)
8. [Search System](#8-search-system)
9. [Integration Patterns](#9-integration-patterns)
10. [Development Guide](#10-development-guide)

---

## 1. Overview

Graphiti is a Python framework (package: `graphiti-core`, version 0.28.0) for building **temporally-aware knowledge graphs** designed as memory backends for AI agents. It provides:

- **Real-time incremental updates** — add information one episode at a time without batch recomputation
- **Bi-temporal data model** — explicit tracking of when facts became true (`valid_at`) and when they stopped being true (`invalid_at`), separate from when they were recorded (`created_at`)
- **Hybrid retrieval** — combines semantic embeddings (cosine similarity), keyword search (BM25 fulltext), and graph traversal (BFS) with configurable reranking
- **Custom entity definitions** — define domain-specific entity types via Pydantic models
- **Multi-tenant partitioning** — `group_id` isolates data across users, sessions, or domains
- **Multiple backends** — Neo4j (primary), FalkorDB, Kuzu, Amazon Neptune

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Episode** | A unit of information ingested into the graph (text, JSON, or message) |
| **Entity (Node)** | An extracted person, place, concept, or object |
| **Fact (Edge)** | A relationship between two entities, expressed as natural language |
| **Community** | A cluster of related entities with a summarized description |
| **Saga** | An ordered sequence of related episodes (e.g., a conversation thread) |
| **group_id** | A partition key for multi-tenant graph isolation |

### Ingestion Flow

When `add_episode()` is called:
1. Previous episodes are retrieved for context
2. LLM extracts entities from the episode content
3. Extracted entities are deduplicated against existing graph nodes
4. LLM extracts relationships (facts) between entities
5. New facts are checked against existing facts — contradictions invalidate old facts
6. Node attributes and summaries are extracted/updated
7. Embeddings are generated for nodes and edges
8. Everything is saved to the graph database

---

## 2. Architecture

### Directory Structure

```
graphiti/
├── graphiti_core/           # Core Python library
│   ├── graphiti.py          # Main Graphiti class (entry point)
│   ├── nodes.py             # Node data models (EntityNode, EpisodicNode, etc.)
│   ├── edges.py             # Edge data models (EntityEdge, EpisodicEdge, etc.)
│   ├── namespaces.py        # Namespace API (graphiti.nodes.*, graphiti.edges.*)
│   ├── driver/              # Database drivers
│   │   ├── driver.py        # GraphDriver ABC, GraphProvider enum
│   │   ├── neo4j_driver.py  # Neo4j implementation
│   │   ├── falkordb_driver.py
│   │   ├── kuzu_driver.py
│   │   └── neptune_driver.py
│   ├── llm_client/          # LLM provider clients
│   │   ├── client.py        # LLMClient ABC
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   ├── gemini_client.py
│   │   └── groq_client.py
│   ├── embedder/            # Embedding providers
│   │   ├── client.py        # EmbedderClient ABC
│   │   ├── openai.py
│   │   └── voyageai.py
│   ├── cross_encoder/       # Reranking providers
│   │   └── openai_reranker_client.py
│   ├── search/              # Search system
│   │   ├── search.py        # Main search function
│   │   ├── search_config.py # SearchConfig, SearchResults models
│   │   ├── search_config_recipes.py  # Pre-built configurations
│   │   ├── search_filters.py         # SearchFilters model
│   │   └── search_utils.py           # Search utilities
│   ├── prompts/             # LLM prompts for extraction
│   └── utils/               # Maintenance, bulk operations
├── server/                  # FastAPI REST server
│   └── graph_service/
│       ├── main.py          # FastAPI app
│       ├── config.py        # Settings
│       ├── zep_graphiti.py  # ZepGraphiti wrapper
│       ├── routers/
│       │   ├── ingest.py    # Ingest endpoints
│       │   └── retrieve.py  # Retrieve endpoints
│       └── dto/             # Data transfer objects
├── mcp_server/              # MCP server for AI assistants
│   ├── src/
│   │   └── graphiti_mcp_server.py  # MCP tools
│   ├── config/
│   │   └── config.yaml      # Configuration schema
│   └── docker/              # Docker compose files
├── examples/                # Usage examples
│   └── quickstart/
│       └── quickstart_neo4j.py
├── tests/                   # Test suite
├── docker-compose.yml       # Root Docker setup
├── Makefile                 # Development commands
└── pyproject.toml           # Package configuration
```

### Component Dependencies

```
Graphiti Class
├── GraphDriver (Neo4j/FalkorDB/Kuzu/Neptune)
├── LLMClient (OpenAI/Anthropic/Gemini/Groq)
├── EmbedderClient (OpenAI/Voyage/Gemini)
├── CrossEncoderClient (OpenAI Reranker)
├── NodeNamespace (graphiti.nodes.*)
├── EdgeNamespace (graphiti.edges.*)
└── Tracer (OpenTelemetry, optional)
```

### Driver Architecture

The `GraphDriver` ABC (`graphiti_core/driver/driver.py`) defines the interface for all database operations. Each driver implementation provides:

- `execute_query()` — Run Cypher queries
- `session()` — Get a database session (context manager)
- `build_indices_and_constraints()` — Create fulltext and vector indices
- `transaction()` — Transactional context manager
- Operations interfaces — Pluggable operation handlers for nodes, edges, search, maintenance

The `GraphProvider` enum: `NEO4J`, `FALKORDB`, `KUZU`, `NEPTUNE`.

---

## 3. Data Model

### Node Types

#### EntityNode (`graphiti_core/nodes.py:484`)

Represents an extracted entity (person, place, organization, concept).

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Unique identifier |
| `name` | str | Entity name |
| `group_id` | str | Graph partition |
| `labels` | list[str] | Type labels (e.g., `['Person', 'Entity']`) |
| `summary` | str | LLM-generated summary of the entity |
| `attributes` | dict[str, Any] | Custom attributes from entity type definitions |
| `name_embedding` | list[float] \| None | Semantic embedding of the name |
| `created_at` | datetime | When the node was stored |

#### EpisodicNode (`graphiti_core/nodes.py:307`)

Represents a raw episode (a unit of ingested information).

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Unique identifier |
| `name` | str | Episode name |
| `group_id` | str | Graph partition |
| `source` | EpisodeType | `message`, `json`, or `text` |
| `source_description` | str | Description of the data source |
| `content` | str | Raw episode content |
| `valid_at` | datetime | When the event occurred |
| `entity_edges` | list[str] | UUIDs of related entity edges |
| `created_at` | datetime | When the episode was stored |

#### CommunityNode (`graphiti_core/nodes.py:666`)

Represents a cluster of related entities.

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Unique identifier |
| `name` | str | Community name |
| `group_id` | str | Graph partition |
| `summary` | str | LLM-generated summary of the community |
| `name_embedding` | list[float] \| None | Semantic embedding |
| `created_at` | datetime | When created |

#### SagaNode (`graphiti_core/nodes.py:844`)

Groups related episodes into an ordered sequence.

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Unique identifier |
| `name` | str | Saga name |
| `group_id` | str | Graph partition |
| `created_at` | datetime | When created |

### Edge Types

#### EntityEdge (`graphiti_core/edges.py:263`) — `RELATES_TO`

A fact connecting two entities.

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | str | Unique identifier |
| `name` | str | Relation name (e.g., `WORKS_AT`) |
| `fact` | str | Natural language fact description |
| `fact_embedding` | list[float] \| None | Semantic embedding of the fact |
| `source_node_uuid` | str | Source entity UUID |
| `target_node_uuid` | str | Target entity UUID |
| `group_id` | str | Graph partition |
| `episodes` | list[str] | Episode UUIDs that reference this edge |
| `created_at` | datetime | When stored |
| `valid_at` | datetime \| None | When the fact became true |
| `invalid_at` | datetime \| None | When the fact stopped being true |
| `expired_at` | datetime \| None | When superseded by new information |
| `attributes` | dict[str, Any] | Custom edge attributes |

#### EpisodicEdge — `MENTIONS`

Links an `EpisodicNode` to an `EntityNode` it references.

#### CommunityEdge — `HAS_MEMBER`

Links a `CommunityNode` to its member `EntityNode`.

#### HasEpisodeEdge — `HAS_EPISODE`

Links a `SagaNode` to its episodes.

#### NextEpisodeEdge — `NEXT_EPISODE`

Links consecutive episodes within a saga.

### Bi-Temporal Model

Graphiti tracks two temporal dimensions:

1. **Transaction time** — When information was recorded (`created_at`, `expired_at`)
2. **Valid time** — When the fact was true in the real world (`valid_at`, `invalid_at`)

When new information contradicts existing facts, the old edge's `expired_at` is set and the new edge takes its place. The `valid_at`/`invalid_at` fields represent the real-world temporal bounds.

### EpisodeType Enum (`graphiti_core/nodes.py:54`)

```python
class EpisodeType(Enum):
    message = 'message'  # Format: "actor: content" (e.g., "user: Hello")
    json = 'json'        # JSON string of structured data
    text = 'text'        # Plain text
```

---

## 4. Setup Guide

### Prerequisites

- **Python** 3.10+
- **Neo4j** 5.26+ (Docker, Neo4j Desktop, or cloud)
- **OpenAI API key** (or alternative LLM provider key)

### Install Graphiti

```bash
# From PyPI
pip install graphiti-core

# With optional backends
pip install graphiti-core[falkordb]
pip install graphiti-core[anthropic]
pip install graphiti-core[google-genai]
pip install graphiti-core[voyageai]
pip install graphiti-core[tracing]

# From source (development)
git clone https://github.com/getzep/graphiti
cd graphiti
uv sync --extra dev
```

### Start Neo4j

**Docker (quickest)**:
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:5.26.2
```

**Docker Compose (from project root)**:
```bash
docker compose up neo4j
```

The `docker-compose.yml` at the project root defines:
- `neo4j` service on ports 7474 (HTTP) and 7687 (Bolt)
- `graph` service (REST API server) on port 8000
- Optional `falkordb` profile for FalkorDB backend

**Neo4j Desktop**:
1. Download from https://neo4j.com/download/
2. Create a new project → Add Local DBMS
3. Set a password, start the DBMS
4. Default connection: `bolt://localhost:7687`

### Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-...
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Optional
export SEMAPHORE_LIMIT=10          # Max concurrent LLM operations
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=...
export VOYAGE_API_KEY=...
export USE_PARALLEL_RUNTIME=false  # Neo4j enterprise only
```

### Verify Connection

```python
import asyncio
from graphiti_core import Graphiti

async def verify():
    g = Graphiti('bolt://localhost:7687', 'neo4j', 'password')
    await g.build_indices_and_constraints()
    print('Connected successfully')
    await g.close()

asyncio.run(verify())
```

---

## 5. Core Library Usage

### Initialization

```python
from graphiti_core import Graphiti

# Basic (uses OpenAI defaults)
graphiti = Graphiti('bolt://localhost:7687', 'neo4j', 'password')

# With custom providers
from graphiti_core.llm_client import AnthropicClient
from graphiti_core.embedder import VoyageEmbedder

graphiti = Graphiti(
    uri='bolt://localhost:7687',
    user='neo4j',
    password='password',
    llm_client=AnthropicClient(model='claude-sonnet-4-5-latest'),
    embedder=VoyageEmbedder(model='voyage-3'),
    max_coroutines=15,
)

# With custom driver
from graphiti_core.driver.neo4j_driver import Neo4jDriver

driver = Neo4jDriver('bolt://localhost:7687', 'neo4j', 'password', database='my_db')
graphiti = Graphiti(graph_driver=driver)

# Always build indices after initialization
await graphiti.build_indices_and_constraints()
```

### Adding Episodes

```python
from datetime import datetime, timezone
from graphiti_core.nodes import EpisodeType

# Plain text
result = await graphiti.add_episode(
    name='meeting_notes_1',
    episode_body='Alice joined Acme Corp as VP of Engineering in January 2025.',
    source=EpisodeType.text,
    source_description='meeting notes',
    reference_time=datetime.now(timezone.utc),
    group_id='team_alpha',
)

# Conversation message
result = await graphiti.add_episode(
    name='chat_msg_42',
    episode_body='user(human): I prefer dark mode and use VSCode daily.',
    source=EpisodeType.message,
    source_description='chat conversation',
    reference_time=datetime.now(timezone.utc),
    group_id='user_123',
)

# Structured JSON
import json
data = {'customer': 'Acme Corp', 'plan': 'Enterprise', 'seats': 50}
result = await graphiti.add_episode(
    name='crm_update_1',
    episode_body=json.dumps(data),
    source=EpisodeType.json,
    source_description='CRM system',
    reference_time=datetime.now(timezone.utc),
    group_id='crm_data',
)

# With custom entity types
from pydantic import BaseModel

class Person(BaseModel):
    """A person entity"""
    pass

class Organization(BaseModel):
    """An organization entity"""
    pass

result = await graphiti.add_episode(
    ...,
    entity_types={'Person': Person, 'Organization': Organization},
    excluded_entity_types=['Topic'],  # Skip generic types
)

# With saga (conversation thread)
result = await graphiti.add_episode(
    ...,
    saga='conversation_thread_1',
)
```

**AddEpisodeResults** contains:
- `result.episode` — The saved `EpisodicNode`
- `result.nodes` — Extracted `EntityNode` list
- `result.edges` — Extracted `EntityEdge` list
- `result.episodic_edges` — `EpisodicEdge` list
- `result.communities` — Updated `CommunityNode` list
- `result.community_edges` — `CommunityEdge` list

### Searching

```python
# Basic search — returns list[EntityEdge]
edges = await graphiti.search(
    query='Who works at Acme Corp?',
    group_ids=['team_alpha'],
    num_results=10,
)
for edge in edges:
    print(f'{edge.fact}')
    print(f'  Valid: {edge.valid_at} → {edge.invalid_at}')

# With center node reranking
edges = await graphiti.search(
    query='project updates',
    group_ids=['team_alpha'],
    center_node_uuid='node-uuid-here',
)

# Advanced search — returns SearchResults
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)

results = await graphiti.search_(
    query='California politics',
    config=NODE_HYBRID_SEARCH_RRF,
    group_ids=['public_data'],
)
for node in results.nodes:
    print(f'{node.name}: {node.summary}')

# Search with filters
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator

results = await graphiti.search_(
    query='team members',
    search_filter=SearchFilters(
        node_labels=['Person'],
        invalid_at=[[DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)]],
    ),
)
```

### Graph Management

```python
# Build communities
communities, community_edges = await graphiti.build_communities(group_ids=['team_alpha'])

# Remove an episode
await graphiti.remove_episode(episode_uuid='...')

# Delete all data for a group
from graphiti_core.nodes import Node
await Node.delete_by_group_id(graphiti.driver, group_id='team_alpha')

# Manual triplet insertion
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now

source = EntityNode(name='Alice', group_id='team_alpha', summary='Engineer')
target = EntityNode(name='Acme Corp', group_id='team_alpha', summary='Tech company')
edge = EntityEdge(
    name='WORKS_AT',
    fact='Alice works at Acme Corp',
    source_node_uuid=source.uuid,
    target_node_uuid=target.uuid,
    group_id='team_alpha',
    created_at=utc_now(),
)
result = await graphiti.add_triplet(source, edge, target)

# Close connection
await graphiti.close()
```

### Namespace API

Fine-grained CRUD operations on individual graph elements:

```python
# Save/retrieve nodes
await graphiti.nodes.entity.save(node)
node = await graphiti.nodes.entity.get_by_uuid(uuid)
nodes = await graphiti.nodes.entity.get_by_uuids([uuid1, uuid2])
await graphiti.nodes.entity.delete(node)

# Save/retrieve edges
await graphiti.edges.entity.save(edge)
edge = await graphiti.edges.entity.get_by_uuid(uuid)

# Episode, community, saga operations follow the same pattern
await graphiti.nodes.episode.save(episode)
await graphiti.nodes.community.save(community)
await graphiti.nodes.saga.save(saga)
```

---

## 6. REST API Server

### Overview

The REST server (`server/graph_service/`) is a FastAPI application that wraps Graphiti for HTTP access.

**Published Docker image**: `zepai/graphiti`

### Configuration

The server uses environment variables loaded via Pydantic `BaseSettings` (`server/graph_service/config.py`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_BASE_URL` | No | Custom OpenAI base URL |
| `MODEL_NAME` | No | Override LLM model |
| `EMBEDDING_MODEL_NAME` | No | Override embedding model |
| `NEO4J_URI` | Yes | Neo4j bolt URI |
| `NEO4J_USER` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | Yes | Neo4j password |

### Running

```bash
# Development
cd server/
uv sync --extra dev
uvicorn graph_service.main:app --reload

# Docker Compose
docker compose up  # Starts Neo4j + server

# Access
# API: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Ingest Endpoints

**Source**: `server/graph_service/routers/ingest.py`

#### `POST /ingest/messages` → 202 Accepted

Add messages to a background processing queue. Messages are processed sequentially per group.

```json
{
  "group_id": "session_123",
  "messages": [
    {
      "uuid": "optional-uuid",
      "name": "msg_1",
      "role": "User",
      "role_type": "human",
      "content": "I prefer dark mode.",
      "timestamp": "2025-01-01T00:00:00Z",
      "source_description": "chat"
    }
  ]
}
```

#### `POST /ingest/entity-node` → 201 Created

Create an entity node directly.

```json
{
  "uuid": "optional-uuid",
  "group_id": "partition_id",
  "name": "Acme Corp",
  "summary": "A technology company"
}
```

#### `DELETE /ingest/entity-edge/{uuid}` → 200

Delete an entity edge by UUID.

#### `DELETE /ingest/group/{group_id}` → 200

Delete all data for a group.

#### `DELETE /ingest/episode/{uuid}` → 200

Delete an episode by UUID.

#### `POST /ingest/clear` → 200

Clear entire graph and rebuild indices.

### Retrieve Endpoints

**Source**: `server/graph_service/routers/retrieve.py`

#### `POST /retrieve/search` → 200

Search for facts (entity edges).

```json
// Request
{
  "query": "team members",
  "group_ids": ["org_123"],
  "max_facts": 10
}

// Response
{
  "facts": [
    {
      "uuid": "edge-uuid",
      "name": "WORKS_AT",
      "fact": "Alice works at Acme Corp",
      "valid_at": "2025-01-01T00:00:00Z",
      "invalid_at": null,
      "source_node_uuid": "...",
      "target_node_uuid": "...",
      "source_node_name": "Alice",
      "target_node_name": "Acme Corp"
    }
  ]
}
```

#### `GET /retrieve/entity-edge/{uuid}` → 200

Get a single entity edge by UUID.

#### `GET /retrieve/episodes/{group_id}?last_n=N` → 200

Get the N most recent episodes for a group.

#### `POST /retrieve/get-memory` → 200

Search from a message conversation context. Combines messages into a query.

```json
{
  "group_id": "session_123",
  "messages": [
    {"role": "User", "role_type": "human", "content": "What are my preferences?"}
  ],
  "max_facts": 10
}
```

---

## 7. MCP Server

### Overview

The MCP (Model Context Protocol) server (`mcp_server/`) exposes Graphiti as tools for AI assistants like Claude Desktop and Cursor.

**Published Docker image**: `zepai/knowledge-graph-mcp`
**Framework**: FastMCP

### Configuration

**Config file**: `mcp_server/config/config.yaml`

The config supports environment variable expansion (`${VAR_NAME:default}`):

```yaml
server:
  transport: "http"     # Options: stdio, sse, http
  host: "0.0.0.0"
  port: 8000

llm:
  provider: "openai"    # Options: openai, azure_openai, anthropic, gemini, groq
  model: "gpt-4o-mini"
  max_tokens: 4096

embedder:
  provider: "openai"    # Options: openai, azure_openai, gemini, voyage
  model: "text-embedding-3-small"
  dimensions: 1536

database:
  provider: "neo4j"     # Options: neo4j, falkordb
  providers:
    neo4j:
      uri: ${NEO4J_URI:bolt://localhost:7687}
      username: ${NEO4J_USER:neo4j}
      password: ${NEO4J_PASSWORD}
      database: ${NEO4J_DATABASE:neo4j}

graphiti:
  group_id: ${GRAPHITI_GROUP_ID:main}
  entity_types:
    - name: "Preference"
      description: "User preferences, choices, opinions"
    - name: "Requirement"
      description: "Specific needs or functionality"
    - name: "Procedure"
      description: "Standard operating procedures"
    - name: "Location"
      description: "Physical or virtual places"
    - name: "Event"
      description: "Time-bound activities"
    - name: "Organization"
      description: "Companies, institutions"
```

### Running

```bash
# Local development
cd mcp_server/
uv sync
export OPENAI_API_KEY=sk-...
uv run main.py

# Docker with Neo4j
cd mcp_server/docker/
docker compose -f docker-compose-neo4j.yml up

# MCP endpoint: http://localhost:8000/mcp/
```

### MCP Tools

**Source**: `mcp_server/src/graphiti_mcp_server.py`

#### `add_memory`

Add an episode to memory. Returns immediately; processing happens asynchronously.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Episode name |
| `episode_body` | str | Yes | Content to store |
| `group_id` | str | No | Graph partition (defaults to config) |
| `source` | str | No | `"text"`, `"json"`, or `"message"` |
| `source_description` | str | No | Source description |
| `uuid` | str | No | Custom UUID |

#### `search_nodes`

Search for entity nodes using hybrid search.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | Search query |
| `group_ids` | list[str] | No | Filter by groups |
| `max_nodes` | int | No | Max results (default: 10) |
| `entity_types` | list[str] | No | Filter by type labels |

#### `search_memory_facts`

Search for facts (relationships) in the graph.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str | Yes | Search query |
| `group_ids` | list[str] | No | Filter by groups |
| `max_facts` | int | No | Max results (default: 10) |
| `center_node_uuid` | str | No | Rerank by proximity |

#### `get_entity_edge`

Get an entity edge by UUID. Parameter: `uuid` (str, required).

#### `get_episodes`

Get recent episodes. Parameters: `group_id` (str), `last_n` (int, default: 10).

#### `delete_entity_edge`

Delete an entity edge by UUID.

#### `delete_episode`

Delete an episode by UUID.

#### `clear_graph`

Clear all data for a group and rebuild indices. Parameter: `group_id` (str, optional).

### Connecting AI Clients

**Claude Desktop** — add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "graphiti": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

**Cursor** — add to MCP settings with the same format.

---

## 8. Search System

### Architecture

The search system (`graphiti_core/search/`) operates across four independent layers, each with configurable search methods and rerankers:

| Layer | Target | Methods | Embedding Field |
|-------|--------|---------|-----------------|
| Edge | EntityEdge (facts) | cosine_similarity, bm25, bfs | `fact_embedding` |
| Node | EntityNode (entities) | cosine_similarity, bm25, bfs | `name_embedding` |
| Episode | EpisodicNode (raw content) | bm25 | — |
| Community | CommunityNode (clusters) | cosine_similarity, bm25 | `name_embedding` |

### Search Methods

- **`cosine_similarity`** — Semantic vector search using embeddings. Requires the query to be embedded first.
- **`bm25`** — BM25 fulltext search using Neo4j's fulltext indices.
- **`bfs`** (edges and nodes only) — Breadth-first graph traversal from origin nodes. Discovers connected subgraphs.

### Rerankers

| Reranker | Available For | Description |
|----------|---------------|-------------|
| `rrf` | All | Reciprocal Rank Fusion — merges rankings from multiple methods |
| `node_distance` | Edges, Nodes | Reranks by graph distance from `center_node_uuid` |
| `episode_mentions` | Edges, Nodes | Sorts by number of episode references |
| `mmr` | Edges, Nodes, Communities | Maximal Marginal Relevance — balances relevance and diversity |
| `cross_encoder` | All | Uses a cross-encoder model for high-quality semantic reranking |

### SearchConfig

```python
from graphiti_core.search.search_config import (
    SearchConfig, EdgeSearchConfig, NodeSearchConfig,
    EdgeSearchMethod, NodeSearchMethod,
    EdgeReranker, NodeReranker,
)

config = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
        sim_min_score=0.0,    # Min cosine similarity threshold
        mmr_lambda=0.5,       # MMR balance (0=diverse, 1=relevant)
        bfs_max_depth=2,      # Max BFS traversal depth
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
    ),
    limit=10,                 # Max results per layer
    reranker_min_score=0.0,   # Global minimum reranker score
)
```

Set a layer config to `None` to skip searching that layer.

### Pre-Built Recipes

**Source**: `graphiti_core/search/search_config_recipes.py`

| Recipe | Layers | Methods | Reranker |
|--------|--------|---------|----------|
| `EDGE_HYBRID_SEARCH_RRF` | Edges | BM25 + cosine | RRF |
| `EDGE_HYBRID_SEARCH_NODE_DISTANCE` | Edges | BM25 + cosine | Node distance |
| `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS` | Edges | BM25 + cosine | Episode mentions |
| `EDGE_HYBRID_SEARCH_MMR` | Edges | BM25 + cosine | MMR |
| `EDGE_HYBRID_SEARCH_CROSS_ENCODER` | Edges | BM25 + cosine + BFS | Cross-encoder |
| `NODE_HYBRID_SEARCH_RRF` | Nodes | BM25 + cosine | RRF |
| `NODE_HYBRID_SEARCH_NODE_DISTANCE` | Nodes | BM25 + cosine | Node distance |
| `NODE_HYBRID_SEARCH_EPISODE_MENTIONS` | Nodes | BM25 + cosine | Episode mentions |
| `NODE_HYBRID_SEARCH_MMR` | Nodes | BM25 + cosine | MMR |
| `NODE_HYBRID_SEARCH_CROSS_ENCODER` | Nodes | BM25 + cosine + BFS | Cross-encoder |
| `COMBINED_HYBRID_SEARCH_RRF` | All | BM25 + cosine | RRF |
| `COMBINED_HYBRID_SEARCH_MMR` | All | BM25 + cosine | MMR |
| `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` | All | BM25 + cosine + BFS | Cross-encoder |
| `COMMUNITY_HYBRID_SEARCH_RRF` | Communities | BM25 + cosine | RRF |
| `COMMUNITY_HYBRID_SEARCH_MMR` | Communities | BM25 + cosine | MMR |
| `COMMUNITY_HYBRID_SEARCH_CROSS_ENCODER` | Communities | BM25 + cosine | Cross-encoder |

### SearchFilters

**Source**: `graphiti_core/search/search_filters.py`

```python
from graphiti_core.search.search_filters import SearchFilters, DateFilter, ComparisonOperator

# Filter by entity type
filters = SearchFilters(node_labels=['Person', 'Organization'])

# Filter by edge type
filters = SearchFilters(edge_types=['WORKS_AT', 'MANAGES'])

# Temporal filter — only currently valid facts
filters = SearchFilters(
    invalid_at=[[DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)]]
)

# Temporal filter — facts valid after a date
from datetime import datetime, timezone
filters = SearchFilters(
    valid_at=[[DateFilter(
        date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        comparison_operator=ComparisonOperator.greater_than,
    )]]
)
```

`ComparisonOperator` values: `equals`, `not_equals`, `greater_than`, `less_than`, `greater_than_equal`, `less_than_equal`, `is_null`, `is_not_null`.

### Search Flow

1. **Query embedding** — If any layer uses `cosine_similarity` or `mmr`, the query is embedded via the `EmbedderClient`
2. **Parallel retrieval** — Each configured layer executes its search methods concurrently
3. **Per-layer reranking** — Each layer independently applies its configured reranker
4. **Limit enforcement** — Results are truncated to `config.limit` per layer
5. **Result assembly** — Combined into `SearchResults` with scores

---

## 9. Integration Patterns

### Direct Python Integration

```python
class MemoryBackend:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    async def initialize(self):
        await self.graphiti.build_indices_and_constraints()

    async def store(self, content: str, group_id: str, source_type='text'):
        return await self.graphiti.add_episode(
            name=f'memory_{datetime.now().timestamp()}',
            episode_body=content,
            source=EpisodeType[source_type],
            source_description='memory backend',
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
        )

    async def recall(self, query: str, group_id: str, max_results=10):
        return await self.graphiti.search(
            query=query, group_ids=[group_id], num_results=max_results
        )

    async def close(self):
        await self.graphiti.close()
```

### REST API Integration

For language-agnostic access, use the FastAPI server:

```python
import httpx

async def store_via_api(content: str, group_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:8000/ingest/messages', json={
            'group_id': group_id,
            'messages': [{
                'name': 'msg',
                'role': 'user',
                'role_type': 'human',
                'content': content,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source_description': 'api',
            }],
        })
        return response.json()

async def search_via_api(query: str, group_ids: list[str]):
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:8000/retrieve/search', json={
            'query': query,
            'group_ids': group_ids,
            'max_facts': 10,
        })
        return response.json()
```

### MCP Integration for AI Agents

Connect the MCP server to Claude Desktop or Cursor for tool-based access:

```json
{
  "mcpServers": {
    "memory": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

### Multi-Tenant Memory

Use `group_id` to partition the graph:

```python
# Per-user memory
await graphiti.add_episode(..., group_id='user_alice')

# Per-session memory
await graphiti.add_episode(..., group_id='session_xyz')

# Cross-partition search
results = await graphiti.search(
    query='preferences',
    group_ids=['user_alice', 'user_bob', 'shared'],
)

# Delete a partition
await Node.delete_by_group_id(graphiti.driver, 'user_alice')
```

### Queue-Based Processing

Both the REST server and MCP server use background queues to process episodes. This is the recommended pattern for production:

```python
# REST server uses AsyncWorker (server/graph_service/routers/ingest.py)
# MCP server uses QueueService (mcp_server/src/services/queue_service.py)

# Both process episodes sequentially per group_id to avoid race conditions
# while allowing concurrent processing across different groups
```

### Saga-Based Conversation Threading

```python
prev_uuid = None
for msg in conversation_messages:
    result = await graphiti.add_episode(
        name=f'msg_{msg["id"]}',
        episode_body=f'{msg["role"]}({msg["role_type"]}): {msg["content"]}',
        source=EpisodeType.message,
        source_description='conversation',
        reference_time=msg['timestamp'],
        group_id='session_abc',
        saga='thread_1',
        saga_previous_episode_uuid=prev_uuid,
    )
    prev_uuid = result.episode.uuid
```

---

## 10. Development Guide

### Commands

All from the project root:

```bash
# Install dependencies
make install        # or: uv sync --extra dev

# Format code
make format         # ruff check --fix --select I && ruff format

# Lint code
make lint           # ruff check && pyright

# Run tests (unit only)
make test           # pytest tests/ -k "not _int"

# Run all checks
make check          # format + lint + test

# Integration tests (requires database)
pytest tests/ -k "_int"

# Specific test file
pytest tests/test_graphiti_mock.py

# Specific test method
pytest tests/test_graphiti_mock.py::test_method_name
```

Server-specific commands:
```bash
cd server/
make format
make lint
make test
```

### Code Style

- **Formatter/Linter**: Ruff (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **Quote style**: Single quotes
- **Type checking**: Pyright (`basic` mode for core, `standard` for server)
- **Python version**: 3.10+

### Testing

- **Unit tests**: Use mocks, no database required. Files: `tests/test_graphiti_mock.py`, `tests/test_add_triplet.py`, etc.
- **Integration tests**: Require a running database. Files suffixed with `_int` (e.g., `tests/test_graphiti_int.py`)
- **Fixtures**: `tests/conftest.py` and `tests/helpers_test.py`
- **Runner**: pytest with pytest-xdist for parallel execution, pytest-asyncio for async tests

### Key Source Files

| Component | File | Description |
|-----------|------|-------------|
| Main class | `graphiti_core/graphiti.py` | Graphiti entry point |
| Nodes | `graphiti_core/nodes.py` | Node data models |
| Edges | `graphiti_core/edges.py` | Edge data models |
| Search | `graphiti_core/search/search.py` | Core search function |
| Search config | `graphiti_core/search/search_config.py` | SearchConfig model |
| Search recipes | `graphiti_core/search/search_config_recipes.py` | Pre-built configs |
| Search filters | `graphiti_core/search/search_filters.py` | Filter models |
| Driver base | `graphiti_core/driver/driver.py` | GraphDriver ABC |
| Neo4j driver | `graphiti_core/driver/neo4j_driver.py` | Neo4j implementation |
| Namespaces | `graphiti_core/namespaces.py` | Namespace CRUD API |
| REST server | `server/graph_service/main.py` | FastAPI app |
| REST ingest | `server/graph_service/routers/ingest.py` | Ingest endpoints |
| REST retrieve | `server/graph_service/routers/retrieve.py` | Retrieve endpoints |
| REST config | `server/graph_service/config.py` | Server settings |
| MCP server | `mcp_server/src/graphiti_mcp_server.py` | MCP tools |
| MCP config | `mcp_server/config/config.yaml` | MCP config schema |
| Docker | `docker-compose.yml` | Root docker setup |
| MCP Docker | `mcp_server/docker/docker-compose-neo4j.yml` | MCP + Neo4j |
| Quickstart | `examples/quickstart/quickstart_neo4j.py` | Usage example |

### Concurrency Tuning

The `SEMAPHORE_LIMIT` environment variable controls max concurrent LLM operations. Guidelines:

| LLM Provider Tier | Recommended Limit |
|-------------------|-------------------|
| OpenAI Free/Tier 1 | 1-2 |
| OpenAI Tier 2 | 5-8 |
| OpenAI Tier 3 | 10-15 |
| OpenAI Tier 4 | 20-50 |
| Anthropic default | 5-8 |
| Anthropic high tier | 15-30 |

**Symptoms of too-high limit**: 429 rate limit errors
**Symptoms of too-low limit**: Slow throughput
