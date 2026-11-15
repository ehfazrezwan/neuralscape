"""memory package: MemoryService mixins and shared helpers (split from memory_service.py)."""

# Package-level guard: every mem0/Graphiti import in the service lives under
# memory/ (or main.py), so loading the package first keeps their telemetry off.
import third_party_telemetry  # noqa: F401 — must precede any mem0/Graphiti import (defaults their telemetry off)
