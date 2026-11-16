import json
import os
from pathlib import Path

import pytest

from nsaccept import guard, runctx
from nsaccept import pytest_plugin as plug

ACCEPTANCE = Path(__file__).resolve().parents[1]


def test_prod_compose_project_refused():
    with pytest.raises(guard.GuardError, match="production"):
        guard.check("http://127.0.0.1:18199", "neuralscape", env={})
    guard.check("http://127.0.0.1:18199", "nstest-acceptance", env={})


def test_prod_compose_project_not_overridable():
    with pytest.raises(guard.GuardError):
        guard.check("http://127.0.0.1:18199", "neuralscape", env={"NS_ALLOW_REMOTE": "1", "NS_ALLOW_PROD_PORT": "1"})


def test_port_8199_refused_unless_explicit_env():
    with pytest.raises(guard.GuardError, match="8199"):
        guard.check("http://127.0.0.1:8199", None, env={})
    with pytest.raises(guard.GuardError, match="8199"):
        guard.check("http://127.0.0.1:8199", None, env={"NS_ALLOW_REMOTE": "1"})
    guard.check("http://127.0.0.1:8199", None, env={"NS_ALLOW_PROD_PORT": "1"})


@pytest.mark.parametrize("url", ["http://10.0.0.5:18199", "https://example.com", "http://192.0.2.1:9000"])
def test_non_loopback_refused_unless_allowed(url):
    with pytest.raises(guard.GuardError, match="non-loopback"):
        guard.check(url, None, env={})
    guard.check(url, None, env={"NS_ALLOW_REMOTE": "1"})


@pytest.mark.parametrize("url", ["http://127.0.0.1:18199", "http://localhost:18199", "http://[::1]:18199", "http://127.5.5.5:1"])
def test_loopback_allowed(url):
    guard.check(url, None, env={})


def test_run_sh_refuses_prod_project_and_port(tmp_path):
    import subprocess
    env = {**os.environ, "NS_TOKEN_SECRET": "s", "PYTHONDONTWRITEBYTECODE": "1"}
    for args in (["http://127.0.0.1:18199", "--compose-project", "neuralscape"], ["http://127.0.0.1:8199"],
                 ["http://203.0.113.9:18199"]):
        r = subprocess.run([str(ACCEPTANCE / "run.sh"), *args, "--out", str(tmp_path / "o.json")],
                           env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 2, (args, r.returncode, r.stdout[-300:], r.stderr[-300:])
        assert not (tmp_path / "o.json").exists()


def test_ctx_repr_contains_no_token():
    c = runctx.Ctx("http://x", "default", "r", "ua", "ub", "TOKA.sigsigsigsigsig", "TOKB.sigsigsigsigsig",
                   "BAD.sigsigsigsigsig", "p", "src")
    text = repr(c)
    for secret in ("TOKA", "TOKB", "BAD.", "sigsig"):
        assert secret not in text


def test_profile_info_parses_flags_and_hash(tmp_path, monkeypatch):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "p.env").write_text("# services: docling\n\nFLAG_A=1\nB = two=2\n")
    monkeypatch.setattr(plug, "ACCEPTANCE_DIR", tmp_path)
    flags, sha = plug._profile_info("p")
    assert flags == {"FLAG_A": "1", "B": "two=2"} and len(sha) == 64
    assert plug._profile_info("missing") == ({}, None)


def test_same_commit_prefix_match():
    assert plug._same_commit("abcdef012345", "abcdef0123456789")
    assert not plug._same_commit("abcdef012345", "123456abcdef")


def test_probe_stack_commit(stub):
    _, url = stub
    assert plug._probe_stack_commit(url) is None  # stub /health has no commit field


STACK_SCEN = '''
from nsaccept import proves
@proves("x.s")
def test_s(ctx):
    pass
'''


def _run(pytester, monkeypatch, *args, env=None):
    monkeypatch.setenv("PYTHONPATH", str(ACCEPTANCE))
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    pytester.makepyfile(test_s=STACK_SCEN)
    out = pytester.path / "r.json"
    r = pytester.runpytest_subprocess("-p", "nsaccept.pytest_plugin", "--ns-out", str(out), "-p", "no:cacheprovider", *args)
    return r, out


def test_commit_mismatch_exits_2(pytester, monkeypatch, stub):
    # Stub that reports a different commit than NS_COMMIT: patch via env override NS_STACK_COMMIT.
    _, url = stub
    r, out = _run(pytester, monkeypatch, "--ns-base-url", url, env={"NS_COMMIT": "aaaaaaaaaaaa", "NS_STACK_COMMIT": "bbbbbbbbbbbb"})
    assert r.ret == 2 and not out.exists()


def test_results_record_commit_stack_commit_and_profile(pytester, monkeypatch, stub):
    _, url = stub
    r, out = _run(pytester, monkeypatch, "--ns-base-url", url,
                  env={"NS_COMMIT": "aaaaaaaaaaaa1111", "NS_STACK_COMMIT": "aaaaaaaaaaaa"})
    assert r.ret == 0
    doc = json.loads(out.read_text())
    assert doc["commit"] == "aaaaaaaaaaaa1111" and doc["stack_commit"] == "aaaaaaaaaaaa"
    import hashlib
    assert doc["profile_flags"] == {}  # default.env carries no flags
    assert doc["profile_sha256"] == hashlib.sha256((ACCEPTANCE / "profiles" / "default.env").read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["neuralscape", " neuralscape ", "NeuralScape", "NEURALSCAPE\t"])
def test_production_project_refused_in_every_spelling(name):
    with pytest.raises(guard.GuardError, match="production"):
        guard.check_compose_project(name, env={})


@pytest.mark.parametrize("name", ["neuralscape-prod", "prod", "myproject", "nstes", "xnstest-a", "nstest", "acceptance", ""])
def test_non_disposable_projects_refused(name):
    with pytest.raises(guard.GuardError):
        guard.check_compose_project(name, env={})


@pytest.mark.parametrize("name", ["nstest-acceptance", "nstest-a2", " NSTest-X ", "nstest-a", "nstest-"])
def test_disposable_projects_pass_and_are_normalised(name):
    assert guard.check_compose_project(name, env={}) == name.strip()


def test_allowlist_is_overridable_only_by_env():
    env = {"NS_COMPOSE_ALLOWLIST": "lab-, Scratch"}
    assert guard.check_compose_project("lab-1", env=env) == "lab-1"
    assert guard.check_compose_project("scratch9", env=env) == "scratch9"
    with pytest.raises(guard.GuardError):
        guard.check_compose_project("nstest-acceptance", env=env)
    with pytest.raises(guard.GuardError, match="production"):
        guard.check_compose_project("neuralscape", env={"NS_COMPOSE_ALLOWLIST": "neural"})


def test_every_compose_entry_point_uses_the_shared_rule():
    import subprocess
    from nsaccept import tokens

    ok = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "aaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbb\n", "")  # noqa: E731
    for name in (" neuralscape ", "prodish"):
        with pytest.raises(guard.GuardError):
            runctx.run_compose(name, "ps", run=ok)
        with pytest.raises(tokens.TokenError):
            tokens.mint_via_compose(name, "u", run=ok)
        with pytest.raises(tokens.TokenError):
            tokens.mint_via_compose_python(name, "u", 60, run=ok)
        with pytest.raises(guard.GuardError):
            guard.check("http://127.0.0.1:18199", name, env={})
    assert tokens.mint_via_compose(" nstest-a ", "u", run=ok)


@pytest.mark.parametrize("name", ["neuralscape", " NeuralScape ", "prodish", "nscamp"])
def test_run_sh_refuses_non_disposable_projects(name, tmp_path):
    import subprocess
    env = {**os.environ, "NS_TOKEN_SECRET": "s"}
    r = subprocess.run([str(ACCEPTANCE / "run.sh"), "http://127.0.0.1:18199", "--compose-project", name],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "refusing compose project" in r.stderr, (name, r.returncode, r.stderr[-200:])


def test_run_sh_uses_uv_when_it_is_the_only_python_tooling(tmp_path):
    import subprocess
    shim = tmp_path / "uv"
    shim.write_text("#!/bin/sh\necho 'uv 0.0-shim'\n")
    shim.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    for tool in ("bash", "dirname", "tr", "sed", "cat", "env", "sh"):
        for d in ("/usr/bin", "/bin"):
            if (Path(d) / tool).exists():
                (bins / tool).symlink_to(Path(d) / tool)
                break
    for var, path in (("PATH", f"{tmp_path}:{bins}"),):
        env = {"PATH": path, "HOME": str(tmp_path), "NS_ENV_SELECT_ONLY": "1"}
        r = subprocess.run([str(bins / "bash"), str(ACCEPTANCE / "run.sh"), "http://127.0.0.1:1", "--list"],
                           env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        lines = r.stdout.split()
        assert lines[0] == "uv" and "--python" in lines and "3.12" in lines and "--no-python-downloads" not in lines, r.stdout
        assert not (tmp_path / ".cache").exists(), "the venv branch was taken"
    # $NS_UV selects a standalone binary that is not on PATH
    env = {"PATH": str(bins), "HOME": str(tmp_path), "NS_ENV_SELECT_ONLY": "1", "NS_UV": str(shim)}
    r = subprocess.run([str(bins / "bash"), str(ACCEPTANCE / "run.sh"), "http://127.0.0.1:1", "--list"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.stdout.split()[0] == str(shim), r.stdout
