"""
CBM (codebase-memory-mcp) standalone adapter.

Drives CBM through its native MCP tools over stdio (or CLI mode).
"""

from .adapter import CBMStandaloneAdapter

__all__ = ["CBMStandaloneAdapter"]
