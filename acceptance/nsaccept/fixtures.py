"""Session fixtures shared by every scenario."""

from __future__ import annotations

import pytest

from .client import NSClient
from .mcp import MCPClient
from .runctx import Ctx


@pytest.fixture(scope="session")
def ctx(request: pytest.FixtureRequest) -> Ctx:
    built = getattr(request.config, "_ns_ctx", None)
    if built is None:  # pragma: no cover - plugin builds it at collection finish
        raise RuntimeError("acceptance context was not built (plugin not loaded?)")
    return built


@pytest.fixture(scope="session")
def anon(ctx: Ctx):
    c = NSClient(ctx.base_url)
    yield c
    c.close()


@pytest.fixture(scope="session")
def a(ctx: Ctx):
    c = NSClient(ctx.base_url, ctx.token_a, ctx.user_a)
    yield c
    c.close()


@pytest.fixture(scope="session")
def b(ctx: Ctx):
    c = NSClient(ctx.base_url, ctx.token_b, ctx.user_b)
    yield c
    c.close()


@pytest.fixture(scope="session")
def bad(ctx: Ctx):
    """Client whose token is well-formed but signed with a secret the stack never saw."""
    c = NSClient(ctx.base_url, ctx.bad_token, ctx.user_a)
    yield c
    c.close()


@pytest.fixture(scope="session")
def mcp_a(ctx: Ctx):
    c = MCPClient(ctx.base_url, ctx.token_a)
    yield c
    c.close()


@pytest.fixture(scope="session")
def mcp_b(ctx: Ctx):
    c = MCPClient(ctx.base_url, ctx.token_b)
    yield c
    c.close()
