# NeuralScape Extensions

Extensions are self-contained Python packages that add API routes, hook into events, and extend NeuralScape's capabilities — all without modifying core code.

## Why Extensions?

NeuralScape core handles memory storage, retrieval, and knowledge graph management. Extensions let you build on top of that:

- **React to events** — trigger actions when memories are stored, sessions start/end, etc.
- **Add API routes** — expose new endpoints under `/v1/extensions/<name>/`
- **Keep it modular** — each extension is isolated; failures don't affect core or other extensions

## The NeuralscapeExtension Protocol

Every extension implements this protocol (defined in `extensions/base.py`):

```python
from typing import Optional, Protocol, runtime_checkable
from fastapi import APIRouter
from pydantic import BaseModel, Field

class ExtensionManifest(BaseModel):
    name: str          # unique identifier
    version: str       # semantic version
    description: str   # human-readable description
    author: str | None = None
    hooks: list[str] = []  # event types to listen to

@runtime_checkable
class NeuralscapeExtension(Protocol):
    manifest: ExtensionManifest

    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]: ...
    def get_routes(self) -> Optional[APIRouter]: ...
```

## Creating an Extension

### Step 1: Create the directory

```
neuralscape-service/extensions/my_extension/
    __init__.py
```

### Step 2: Implement the protocol

```python
# extensions/my_extension/__init__.py

from typing import Optional
from fastapi import APIRouter
from extensions.base import ExtensionManifest, NeuralscapeExtension


class MyExtension:
    """Example NeuralScape extension."""

    manifest = ExtensionManifest(
        name="my-extension",
        version="0.1.0",
        description="Does something useful",
        author="Your Name",
        hooks=["memory_stored"],  # events to listen to
    )

    async def startup(self) -> None:
        # Initialize resources (DB connections, clients, etc.)
        pass

    async def shutdown(self) -> None:
        # Clean up resources
        pass

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        if event_type == "memory_stored":
            # React to a new memory being stored
            memory_id = payload.get("memory_id")
            # ... do something with it
            return {"processed": memory_id}
        return None

    def get_routes(self) -> Optional[APIRouter]:
        router = APIRouter()

        @router.get("/status")
        async def status():
            return {"status": "ok", "extension": self.manifest.name}

        return router
```

That's it. NeuralScape will auto-discover it on startup.

### Step 3: Test it

Start NeuralScape and verify:

```bash
# Check it's registered
curl http://localhost:8199/v1/extensions

# Hit your custom route
curl http://localhost:8199/v1/extensions/my-extension/status
```

## Extension Lifecycle

```
Discovery → Registration → Startup → Events → Shutdown
```

1. **Discovery** — on app startup, the registry scans `extensions/` subdirectories and the `NEURALSCAPE_EXTENSIONS` env var
2. **Registration** — each discovered extension is instantiated and its manifest recorded
3. **Startup** — `startup()` is called on every registered extension (failures are logged, not fatal)
4. **Events** — as NeuralScape operates, events are dispatched to extensions whose `manifest.hooks` includes the event type
5. **Shutdown** — `shutdown()` is called on all started extensions during app teardown

## Adding Routes

Return an `APIRouter` from `get_routes()`. Routes are mounted at `/v1/extensions/<name>/`.

```python
def get_routes(self) -> Optional[APIRouter]:
    router = APIRouter()

    @router.get("/items")
    async def list_items():
        return {"items": [...]}

    @router.post("/items")
    async def create_item(data: dict):
        # ...
        return {"created": True}

    return router
```

Return `None` if your extension doesn't need routes (event-only extensions).

## Listening to Events

Declare the event types in `manifest.hooks`, then handle them in `on_event()`:

```python
manifest = ExtensionManifest(
    name="my-listener",
    version="0.1.0",
    description="Listens to memory events",
    hooks=["memory_stored", "session_start"],
)

async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
    if event_type == "memory_stored":
        # payload has: user_id, memory_id, content, category, scope, project_id
        ...
    elif event_type == "session_start":
        # payload has: user_id, session_id, project_id, agent_id, metadata
        ...
    return None
```

### Standard Event Types

| Event Type | Description | Key Payload Fields |
|---|---|---|
| `conversation_turn` | Conversation turn to process | `user_id`, `messages`, `project_id` |
| `session_start` | New session began | `user_id`, `session_id`, `project_id` |
| `session_end` | Session ended | `user_id`, `session_id`, `duration_seconds` |
| `memory_stored` | Memory was stored | `user_id`, `memory_id`, `content`, `category` |
| `compile_requested` | Daily compilation requested | `user_id`, `project_id`, `requested_by` |

### Posting Events Externally

External systems (like OpenClaw hooks) can post events via the API:

```bash
curl -X POST http://localhost:8199/v1/extensions/events \
  -H "Content-Type: application/json" \
  -d '{"event_type": "session_start", "payload": {"user_id": "alice", "session_id": "abc123"}}'
```

## Configuration

### Auto-discovery

Place your extension in `neuralscape-service/extensions/<name>/` with an `__init__.py` exporting a class that implements `NeuralscapeExtension`. It will be discovered automatically.

### Explicit registration via env var

For extensions installed as separate packages or located elsewhere:

```bash
export NEURALSCAPE_EXTENSIONS="mypackage.ext_module,another_package.ext"
```

Comma-separated Python import paths. Each module should export a class implementing `NeuralscapeExtension`.

## Error Handling

Extensions are isolated from core and from each other:

- If `startup()` fails, the extension is marked `failed` and skipped for events
- If `on_event()` throws, the error is logged and other extensions still receive the event
- If `shutdown()` fails, the error is logged and other extensions still shut down
- If `get_routes()` fails, the error is logged and the extension runs without routes

## Example: Skeleton Extension

```python
# extensions/skeleton/__init__.py
"""Minimal NeuralScape extension skeleton."""

from typing import Optional
from fastapi import APIRouter
from extensions.base import ExtensionManifest


class SkeletonExtension:
    manifest = ExtensionManifest(
        name="skeleton",
        version="0.1.0",
        description="A minimal skeleton extension",
    )

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        return None

    def get_routes(self) -> Optional[APIRouter]:
        return None
```
