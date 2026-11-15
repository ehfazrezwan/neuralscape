"""Who the hooks are and which project they file things under.

Identity: the URL, API_KEY and USER_ID options are what the hooks actually put on the wire; a wrong
key is reported as a rejected key (not as a dead service) and does not advance the unreachable
counter; a USER_ID that disagrees with the token is refused by the service and diagnosed.
Project: PROJECT_ID option > nearest `.neuralscape-project` marker > git root > the directory's own
name, the same id for every hook and every subdirectory. Everything is LLM-free and writes nothing.
"""

from __future__ import annotations

import pytest

from nsaccept import expect_check_fails, needs, negative_control, proves
from nsaccept.proxy import RecordingProxy
from plugin_kit import Session, assert_continues, context_of

pytestmark = needs("node")

DEAD_URL = "http://127.0.0.1:1"


# ── identity ─────────────────────────────────────────────────────


def check_identity_on_the_wire(proxy: RecordingProxy, *, project: str, user: str, token: str) -> None:
    asked = proxy.calls(f"/v1/context/{project}", "GET")
    assert len(asked) == 1, f"{len(asked)} context requests reached the configured URL (want 1)"
    call = asked[0]
    assert call.query.get("user_id") == [user], f"user_id sent: {call.query.get('user_id')}"
    assert call.bearer == token, "the API_KEY option was not sent as the bearer token"
    assert call.status == 200, f"the service answered {call.status} to the hook's own identity"


@proves("plugin.config-identity")
def test_url_api_key_and_user_id_options_are_what_the_hook_sends(ctx):
    with RecordingProxy(ctx.base_url) as proxy:
        s = Session(ctx, project_id=f"{ctx.project_id}-pident", proxy=proxy)
        try:
            assert_continues(s.session_start())
            check_identity_on_the_wire(proxy, project=s.project_id, user=ctx.user_a, token=ctx.token_a)
        finally:
            s.close()


@proves("plugin.config-identity")
@negative_control
def test_identity_check_fails_for_another_user_and_for_a_missing_key(ctx):
    with RecordingProxy(ctx.base_url) as proxy:
        s = Session(ctx, project_id=f"{ctx.project_id}-pident", proxy=proxy)
        try:
            s.session_start()
            expect_check_fails(check_identity_on_the_wire, proxy, project=s.project_id, user=ctx.user_b, token=ctx.token_a)
            expect_check_fails(check_identity_on_the_wire, proxy, project=s.project_id, user=ctx.user_a, token=ctx.token_b)
        finally:
            s.close()


@proves("plugin.config-identity", "plugin.unreachable-counter")
def test_a_rejected_key_is_reported_as_a_rejected_key_and_is_not_counted_as_an_outage(ctx):
    with RecordingProxy(ctx.base_url) as proxy:
        s = Session(ctx, project_id=f"{ctx.project_id}-pident", proxy=proxy, token=ctx.bad_token)
        try:
            for _ in range(4):  # one more than FAIL_LOUD_THRESHOLD's default of 3
                res = s.session_start()
                assert_continues(res)
                text = context_of(res)
                assert "rejected the API key" in text, f"no auth notice: {text[:160]!r}"
                assert "unreachable" not in text, "a rejected key was reported as an unreachable service"
            assert [c.status for c in proxy.calls(f"/v1/context/{s.project_id}")] == [401] * 4, "the service did not answer 401 to the bad key"
            counter = s.sandbox.observations / "unreachable.count"
            assert not counter.exists(), "rejected keys advanced the unreachable counter"
        finally:
            s.close()


@proves("plugin.config-identity")
def test_a_user_id_option_that_disagrees_with_the_token_is_refused_and_diagnosed(ctx):
    with RecordingProxy(ctx.base_url) as proxy:
        s = Session(ctx, project_id=f"{ctx.project_id}-pident", proxy=proxy, user_id=f"{ctx.user_a}-someone-else")
        try:
            res = s.session_start()
            assert_continues(res)
            call = proxy.calls(f"/v1/context/{s.project_id}", "GET")[0]
            assert call.query.get("user_id") == [f"{ctx.user_a}-someone-else"], "the USER_ID option was not the id the hook sent"
            assert call.status == 400, f"the service answered {call.status} (want 400) to an id that disagrees with the token"
            text = context_of(res)
            assert "rejected the context request" in text, f"no client-error notice: {text[:160]!r}"
            assert "unreachable" not in text, "a refused request was reported as an unreachable service"
        finally:
            s.close()


# ── project id ───────────────────────────────────────────────────


@pytest.fixture
def tree(ctx):
    """A scratch directory tree inside one hook sandbox (its own `work` is the root)."""
    s = Session(ctx, project_id=None, proxy=None, url=DEAD_URL)
    root = s.sandbox.work
    run = ctx.run_id
    marked = root / "mono"
    (marked / "svc" / "deep").mkdir(parents=True)
    (marked / ".neuralscape-project").write_text(f"pinned-{run}\n")
    empty_marker = root / "emptymark"
    (empty_marker / "sub").mkdir(parents=True)
    (empty_marker / ".neuralscape-project").write_text("")
    repo = root / f"repo-{run}"
    (repo / ".git").mkdir(parents=True)
    (repo / "pkg" / "inner").mkdir(parents=True)
    plain = root / f"plain-{run}"
    plain.mkdir()
    yield s, {
        "marker": (marked / "svc" / "deep", f"pinned-{run}"),
        "empty marker": (empty_marker / "sub", "emptymark"),
        "git root": (repo / "pkg" / "inner", f"repo-{run}"),
        "folder name": (plain, f"plain-{run}"),
    }
    s.close()


def project_of(ctx, owner: Session, cwd, *, option_project: str | None = None) -> str | None:
    """The project id the hook files a tool observation under, for a session working in ``cwd``."""
    s = Session(ctx, project_id=option_project, proxy=None, url=DEAD_URL, sandbox=owner.sandbox, cwd=cwd)
    s.bash("make build", "ok")
    rows = s.buffer_rows()
    assert rows, "the hook buffered nothing"
    return rows[-1].get("project_id")


def check_project_is(ctx, owner, cwd, expected: str, **kw) -> None:
    got = project_of(ctx, owner, cwd, **kw)
    assert got == expected, f"project id {got!r}, want {expected!r}"


@proves("plugin.project-id-resolution")
def test_the_project_id_comes_from_marker_then_git_root_then_folder_name(ctx, tree):
    owner, cases = tree
    for what, (cwd, expected) in cases.items():
        check_project_is(ctx, owner, cwd, expected)  # the same id from a deep subdirectory as from the root


@proves("plugin.project-id-resolution")
def test_the_project_id_option_beats_every_marker(ctx, tree):
    owner, cases = tree
    for what, (cwd, _) in cases.items():
        check_project_is(ctx, owner, cwd, f"opt-{ctx.run_id}", option_project=f"opt-{ctx.run_id}")


@proves("plugin.project-id-resolution")
def test_the_server_calls_use_the_same_resolved_id(ctx, tree):
    owner, cases = tree
    cwd, expected = cases["marker"]
    with RecordingProxy(ctx.base_url) as proxy:
        s = Session(ctx, project_id=None, proxy=proxy, cwd=cwd)
        try:
            s.session_start()
            assert len(proxy.calls(f"/v1/context/{expected}", "GET")) == 1, (
                f"the SessionStart did not ask for /v1/context/{expected}: {[c.path for c in proxy.requests]}"
            )
        finally:
            s.close()


@proves("plugin.project-id-resolution")
@negative_control
def test_project_check_fails_for_the_wrong_resolution_rule(ctx, tree):
    owner, cases = tree
    marker_cwd, marker_id = cases["marker"]
    git_cwd, git_id = cases["git root"]
    expect_check_fails(check_project_is, ctx, owner, marker_cwd, git_id)  # a marker is not a git root
    expect_check_fails(check_project_is, ctx, owner, git_cwd, marker_id)
