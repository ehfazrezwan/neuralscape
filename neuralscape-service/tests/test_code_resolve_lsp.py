"""Resolver-service driver tests — pyright over REST (CODE_NEIGHBORS_RESOLVER=lsp).

Covers the NS-side driver: the LspCallResolver REST client (interface parity with
JediCallResolver, degrade-to-unresolved on transport failure), and the engine's
resolver selection — lsp when the service is healthy, transparent fallback to
in-process Jedi when it is down, and provenance tagging on stored CALLS edges.

The real pyright integration lives in the resolver-svc image and is exercised at
benchmark time; here the service is mocked so the driver logic is tested in
isolation.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.code_graph.code_resolve_lsp import LspCallResolver, ResolverServiceError
from adapters.code_graph.native_engine import NativeEngine


# ── LspCallResolver REST client ──────────────────────────────────────


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    g()\n")
    (tmp_path / "b.py").write_text("def g():\n    pass\n")
    return tmp_path


def test_resolve_file_happy_path(repo):
    r = LspCallResolver(repo, "http://resolver-svc:8201")
    payload = {
        "repo_path": str(repo),
        "files": [{"path": "a.py", "defs": [[str(repo / "b.py"), 1]]}],
    }
    with patch.object(r._client, "post", return_value=_FakeResp(payload)) as post:
        out = r.resolve_file(repo / "a.py", "src", [(2, 4)])
    assert out == [(str(repo / "b.py"), 1)]
    # The request carries repo-relative path + [line, col] sites.
    body = post.call_args.kwargs["json"]
    assert body["repo_path"] == str(repo)
    assert body["files"][0]["path"] == "a.py"
    assert body["files"][0]["sites"] == [[2, 4]]


def test_resolve_file_empty_sites_no_call(repo):
    r = LspCallResolver(repo, "http://x:8201")
    with patch.object(r._client, "post") as post:
        assert r.resolve_file(repo / "a.py", "src", []) == []
    post.assert_not_called()


def test_resolve_file_outside_repo_all_unresolved(repo):
    r = LspCallResolver(repo, "http://x:8201")
    out = r.resolve_file("/etc/passwd", "src", [(1, 0), (2, 0)])
    assert out == [(None, None), (None, None)]


def test_resolve_file_transport_failure_degrades(repo):
    r = LspCallResolver(repo, "http://x:8201")
    with patch.object(r._client, "post", side_effect=RuntimeError("boom")):
        out = r.resolve_file(repo / "a.py", "src", [(2, 4)])
    assert out == [(None, None)]


def test_resolve_file_length_mismatch_degrades(repo):
    r = LspCallResolver(repo, "http://x:8201")
    # Server returns 1 def for 2 sites → treat as unresolved (defensive).
    payload = {"files": [{"path": "a.py", "defs": [[str(repo / "b.py"), 1]]}]}
    with patch.object(r._client, "post", return_value=_FakeResp(payload)):
        out = r.resolve_file(repo / "a.py", "src", [(2, 4), (3, 4)])
    assert out == [(None, None), (None, None)]


def test_health_ok(repo):
    r = LspCallResolver(repo, "http://x:8201")
    with patch("httpx.Client") as C:
        C.return_value.__enter__.return_value.get.return_value = _FakeResp({"status": "ok"})
        r.health()  # no raise


def test_health_unhealthy_raises(repo):
    r = LspCallResolver(repo, "http://x:8201")
    with patch("httpx.Client") as C:
        C.return_value.__enter__.return_value.get.return_value = _FakeResp({"status": "bad"})
        with pytest.raises(ResolverServiceError):
            r.health()


def test_health_unreachable_raises(repo):
    r = LspCallResolver(repo, "http://x:8201")
    with patch("httpx.Client") as C:
        C.return_value.__enter__.return_value.get.side_effect = RuntimeError("conn refused")
        with pytest.raises(ResolverServiceError):
            r.health()


# ── Engine resolver selection ────────────────────────────────────────


def _engine(repo, mode, url="http://resolver-svc:8201"):
    s = Mock()
    s.code_neighbors_resolver = mode
    s.code_resolver_url = url
    s.code_embedder = "off"
    return NativeEngine(
        repo_path=str(repo), code_space="code--u--mini",
        bridge=Mock(), settings=s, driver=MagicMock(),
    )


def test_build_resolver_jedi_mode(repo):
    eng = _engine(repo, "jedi")
    resolver, provenance = eng._build_call_resolver(repo)
    assert provenance == "jedi"
    from adapters.code_graph.code_resolve import JediCallResolver

    assert isinstance(resolver, JediCallResolver)


def test_build_resolver_lsp_mode_healthy(repo):
    eng = _engine(repo, "lsp")
    fake = MagicMock()
    fake.health.return_value = None
    with patch(
        "adapters.code_graph.code_resolve_lsp.LspCallResolver", return_value=fake
    ) as ctor:
        resolver, provenance = eng._build_call_resolver(repo)
    assert provenance == "lsp"
    assert resolver is fake
    ctor.assert_called_once()
    fake.health.assert_called_once()


def test_build_resolver_lsp_down_falls_back_to_jedi(repo):
    eng = _engine(repo, "lsp")
    fake = MagicMock()
    fake.health.side_effect = ResolverServiceError("down")
    with patch(
        "adapters.code_graph.code_resolve_lsp.LspCallResolver", return_value=fake
    ):
        resolver, provenance = eng._build_call_resolver(repo)
    # Service down → transparent fallback to in-process Jedi.
    assert provenance == "jedi"
    from adapters.code_graph.code_resolve import JediCallResolver

    assert isinstance(resolver, JediCallResolver)


def test_store_resolved_edges_tags_provenance(repo):
    eng = _engine(repo, "lsp")
    captured = {}
    with patch.object(
        eng, "_run_cypher_with_retry",
        side_effect=lambda c, **k: captured.update(k) or [],
    ):
        eng._store_resolved_call_edges([{"src_fqn": "a", "tgt_fqn": "b"}], "lsp")
    assert captured["provenance"] == "lsp"


def test_delete_stale_matches_any_resolver(repo):
    eng = _engine(repo, "lsp")
    seen = {}
    with patch.object(
        eng, "_run_cypher_with_retry",
        side_effect=lambda c, **k: seen.update({"cypher": c, "kw": k}) or [],
    ):
        eng._delete_stale_resolved_calls(["a.py"])
    # Broadened: deletes ANY resolver-produced edge (jedi OR lsp), not just jedi.
    assert "r.resolver IS NOT NULL" in seen["cypher"]
    assert "{resolver: 'jedi'}" not in seen["cypher"]


def test_lsp_end_to_end_stores_lsp_provenance(repo):
    """Engine resolve+store with the LSP resolver mocked to return a cross-file
    def resolves to the real FQN and tags the edge resolver='lsp'."""
    eng = _engine(repo, "lsp")
    eng._resolver_collect = True
    eng._pending_call_sites = {}
    symbols_by_file = {}
    for rel in ("a.py", "b.py"):
        syms, _ = eng._parse_file(repo / rel, repo, "python")
        symbols_by_file[rel] = [(s.line, s.end_line, s.fqn) for s in syms]

    # Mock the LSP resolver: a.py's g() call → b.py def line 1 (g).
    fake = MagicMock()
    fake.health.return_value = None

    def _resolve_file(abs_path, source, sites):
        if str(abs_path).endswith("a.py"):
            return [(str(repo / "b.py"), 1) for _ in sites]
        return [(None, None) for _ in sites]

    fake.resolve_file.side_effect = _resolve_file

    stored = []
    with patch(
        "adapters.code_graph.code_resolve_lsp.LspCallResolver", return_value=fake
    ), patch.object(
        eng, "_run_cypher_with_retry",
        side_effect=lambda c, **k: (
            stored.extend(k["rows"]) if "MERGE (src)-[r:CALLS]" in c else None
        ) or [],
    ):
        n = eng._resolve_and_store_calls(repo, symbols_by_file)

    pairs = {(e["src_fqn"], e["tgt_fqn"]) for e in stored}
    assert ("a.f", "b.g") in pairs
    assert n >= 1
