"""Session fixtures shared by every scenario."""

from __future__ import annotations

import pytest

from . import sidecar

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


@pytest.fixture(scope="session")
def dictator(ctx: Ctx):
    """Client for the profile's dictator identity (first DICTATOR_USER_IDS entry)."""
    if not ctx.token_dictator:
        pytest.skip("dictator not available (profile names no DICTATOR_USER_IDS, or tokens cannot be minted)")
    c = NSClient(ctx.base_url, ctx.token_dictator, ctx.dictator_user)
    yield c
    c.close()


@pytest.fixture(scope="session")
def mcp_dictator(ctx: Ctx):
    if not ctx.token_dictator:
        pytest.skip("dictator not available (profile names no DICTATOR_USER_IDS, or tokens cannot be minted)")
    c = MCPClient(ctx.base_url, ctx.token_dictator)
    yield c
    c.close()


@pytest.fixture(scope="session")
def sink(ctx: Ctx):
    """The ``nsc-acc-sink`` sidecar on the stack's network; always removed at session end."""
    if not ctx.compose_project:
        pytest.skip("the sink sidecar needs --compose-project")
    s = sidecar.Sink(ctx.compose_project, sidecar.sink_fixtures(ctx.run_id))
    s.start()
    try:
        yield s
    finally:
        s.stop()


def _legacy_ready(ctx: Ctx) -> None:
    """The `legacy-key` profile's scenarios exist only for that stack: without the key it is a failure, not a skip."""
    if not ctx.legacy_key or not ctx.legacy_user:
        pytest.fail(
            "PRECONDITION FAILED: this scenario needs the `legacy-key` profile's stack. The gate must start the stack "
            "with that profile's lines (NEURALSCAPE_API_KEY=<generate:hex32>, NEURALSCAPE_API_KEY_USER_ID) and export the "
            "generated key as NS_LEGACY_KEY (profile header `# harness-env: NS_LEGACY_KEY=NEURALSCAPE_API_KEY`); "
            f"NS_LEGACY_KEY {'is set' if ctx.legacy_key else 'is missing'}, the bound user id is "
            f"{'known' if ctx.legacy_user else 'missing from the profile flags'}.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def legacy(ctx: Ctx):
    """REST client authenticating with the stack's legacy shared key (`legacy-key` profile only)."""
    _legacy_ready(ctx)
    c = NSClient(ctx.base_url, ctx.legacy_key, ctx.legacy_user)
    yield c
    c.close()


@pytest.fixture(scope="session")
def mcp_legacy(ctx: Ctx):
    _legacy_ready(ctx)
    c = MCPClient(ctx.base_url, ctx.legacy_key)
    yield c
    c.close()
