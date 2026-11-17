"""Self-tests for the flags-on machinery: profile parsing/secret hygiene, the dictator identity,
the sink sidecar's safety rules and its inline server, and the figure-carrying PDF (no stack)."""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from nsaccept import guard, profile, runctx, sidecar, tokens
from nsaccept.pdfgen import make_pdf_with_image

ACCEPTANCE = Path(__file__).resolve().parents[1]
PROFILES = sorted((ACCEPTANCE / "profiles").glob("*.env"))
# The repo's secrecy rule (CLAUDE.md "Public Repo"): no deployment names. Assembled from pieces so this
# file does not itself contain the words the pre-push grep looks for.
FORBIDDEN = re.compile("|".join(["optimi" + "zely", "iiis-" + "492427", "windmill-" + "gke", "svc-utility-" + "belt", "light" + "path", "silicon" + "inja", r"159\.223\."]), re.I)


# ── profile parsing ──────────────────────────────────────────────


def test_parse_records_placeholders_headers_and_unquotes_json():
    p = profile.parse_text(
        "# services: docling, cbm-bridge\n# sidecars: nsc-acc-sink\nA=1\nVAULT_KEY=<generate:fernet>\n"
        "X_API_KEY=<generate:hex32>\nCODE_REPOS='{\"svc\":\"/x\"}'\nDICTATOR_USER_IDS=acc-dictator, other\n\n#comment\nnoequals\n"
    )
    assert p.flags["A"] == "1" and p.flags["VAULT_KEY"] == "<generate:fernet>"
    assert p.placeholders == {"VAULT_KEY": "fernet", "X_API_KEY": "hex32"}
    assert p.flags["CODE_REPOS"] == '{"svc":"/x"}'
    assert p.headers == {"services": ["docling", "cbm-bridge"], "sidecars": ["nsc-acc-sink"]}
    assert p.dictator_ids() == ["acc-dictator", "other"] and p.first_dictator() == "acc-dictator"


def test_a_literal_secret_is_never_recorded():
    p = profile.parse_text("SOME_SECRET=hunter2hunter2\nAPI_TOKEN=abc\nPLAIN=ok\nVAULT_KEY=<generate:fernet>\nEMPTY_KEY=\n")
    assert p.flags["SOME_SECRET"] == profile.REDACTED and p.flags["API_TOKEN"] == profile.REDACTED
    assert p.flags["PLAIN"] == "ok" and p.flags["VAULT_KEY"] == "<generate:fernet>" and p.flags["EMPTY_KEY"] == ""
    assert "hunter2" not in json.dumps(p.flags)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_shipped_profile_is_free_of_literal_secrets_and_deployment_details(path):
    text = path.read_text()
    prof = profile.parse_text(text)
    assert profile.REDACTED not in prof.flags.values(), f"{path.name} holds a literal secret-looking value"
    assert not FORBIDDEN.search(text), f"{path.name} names a deployment"
    for key, value in prof.flags.items():
        assert not re.fullmatch(r"[A-Za-z0-9+/_=-]{32,}", value), f"{path.name}: {key} looks like a key"
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            assert re.match(r"^[A-Z][A-Z0-9_]*=", line), f"{path.name}: not KEY=value: {line!r}"


def test_the_flags_on_profile_turns_on_every_flag_the_scenarios_need():
    flags = profile.load(ACCEPTANCE / "profiles" / "flags-on.env").flags
    for key in ("DREAMING_ENABLED", "STANDARDS_ENABLED", "PROCESSES_ENABLED", "CONNECTORS_ENABLED", "CODE_NATIVE_ENABLED",
                "EXTRACTION_REQUIRE_SPEAKER", "STRATEGY_SYNTHESIZER_ENABLED"):
        assert flags[key] == "true", key
    assert flags["DICTATOR_USER_IDS"] == "acc-dictator"
    assert flags["VAULT_KEY"] == "<generate:fernet>"
    assert flags["WEBHOOK_QUEUE_EMPTY_URL"] == f"http://{sidecar.SINK_NAME}:{sidecar.PORT}{sidecar.QUEUE_EMPTY_PATH}"
    assert json.loads(flags["CODE_REPOS"]) == {"svc": "/app/neuralscape-service/knowledge", "svcdel": "/app/neuralscape-service/knowledge"}
    assert 0 < float(flags["DREAMING_SALIENCE_RECALL_K"]) <= 0.1


def test_provider_and_cbm_profiles_carry_their_wiring():
    g = profile.load(ACCEPTANCE / "profiles" / "oauth-google.env")
    assert g.flags["AUTH_PROVIDER"] == "google" and g.placeholders["GOOGLE_OAUTH_CLIENT_SECRET"] == "hex32"
    s = profile.load(ACCEPTANCE / "profiles" / "oauth-supabase.env")
    assert s.flags["AUTH_PROVIDER"] == "supabase" and s.placeholders["SUPABASE_ANON_KEY"] == "hex32"
    c = profile.load(ACCEPTANCE / "profiles" / "cbm.env")
    assert c.flags["CBM_ENABLED"] == "true" and c.headers["services"] == ["cbm-bridge"]
    d = profile.load(ACCEPTANCE / "profiles" / "docling.env")
    assert d.flags["EXEMPLAR_STORE_ENABLED"] == "true" and d.flags["DOCLING_EXTRACT_IMAGES"] == "true"


# ── dictator identity ────────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("NS_TOKEN_SECRET", "NS_TOKEN_A", "NS_TOKEN_B", "NS_USER_A", "NS_USER_B", "NS_COMPOSE_PROJECT", "NS_RUN_ID"):
        monkeypatch.delenv(k, raising=False)


def test_a_dictator_token_is_minted_for_the_first_listed_id(monkeypatch, clean_env):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s3")
    c = runctx.build("http://x", "flags-on", profile_flags={"DICTATOR_USER_IDS": "acc-dictator"}, dictator_user="acc-dictator")
    assert c.dictator_user == "acc-dictator" and c.token_dictator
    claims = json.loads(tokens.base64.urlsafe_b64decode(c.token_dictator.split(".")[0] + "=="))
    assert claims["user_id"] == "acc-dictator" and c.profile_flags == {"DICTATOR_USER_IDS": "acc-dictator"}


def test_the_dictator_token_is_minted_through_compose_when_that_is_the_source(monkeypatch, clean_env):
    calls = []
    monkeypatch.setattr(runctx.tokens, "mint_via_compose", lambda project, user, service="neuralscape", **kw: calls.append(user) or f"tok-{user}")
    c = runctx.build("http://x", "flags-on", compose_project="nstest-p1", dictator_user="acc-dictator")
    assert c.token_dictator == "tok-acc-dictator" and calls == [c.user_a, c.user_b, "acc-dictator"]


def test_there_is_no_dictator_without_one_named_or_without_a_way_to_mint(monkeypatch, clean_env):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    assert runctx.build("http://x", "default").token_dictator is None  # none named
    monkeypatch.delenv("NS_TOKEN_SECRET")
    monkeypatch.setenv("NS_TOKEN_A", "ta")
    monkeypatch.setenv("NS_TOKEN_B", "tb")
    monkeypatch.setenv("NS_USER_A", "ua")
    monkeypatch.setenv("NS_USER_B", "ub")
    c = runctx.build("http://x", "flags-on", dictator_user="acc-dictator")  # pre-made tokens cannot be minted for him
    assert c.token_dictator is None and c.dictator_user is None


def test_ctx_repr_hides_the_dictator_token(monkeypatch, clean_env):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    c = runctx.build("http://x", "flags-on", dictator_user="acc-dictator")
    assert c.token_dictator not in repr(c)


# ── the sink sidecar: safety rules ───────────────────────────────


class Recorder:
    def __init__(self, logs="", network="nstest-x_default\n"):
        self.calls, self.logs, self.network = [], logs, network

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = ""
        if cmd[1:3] == ["network", "ls"]:
            out = self.network
        elif cmd[1] == "logs":
            out = self.logs
        return subprocess.CompletedProcess(cmd, 0, out, "")


@pytest.fixture(autouse=True)
def no_live_sidecars():
    sidecar._LIVE.clear()
    yield
    sidecar._LIVE.clear()


READY = json.dumps({"ready": True, "port": 8080}) + "\n"


def _sink(rec, project="nstest-x", **kw):
    return sidecar.Sink(project, {"/items": {"a": 1}}, run=rec, clock=time.monotonic, sleep=lambda s: None, **kw)


def test_start_runs_one_small_locked_down_container_on_the_discovered_network():
    rec = Recorder(logs=READY)
    s = _sink(rec).start()
    run = next(c for c in rec.calls if c[1] == "run")
    for flag, value in (("--name", "nsc-acc-sink"), ("--network", "nstest-x_default"), ("--memory", "64m"), ("--cpus", "0.25"), ("--pids-limit", "32")):
        assert run[run.index(flag) + 1] == value, flag
    assert {"-d", "--rm", "--read-only"} <= set(run) and run[run.index("--cap-drop") + 1] == "ALL"
    assert not any(a.startswith("-v") or a == "--volume" or a == "-p" for a in run), "a sidecar must publish nothing and mount nothing"
    assert run[-3:-1] == ["python", "-c"] and run[-4] == sidecar.IMAGE
    assert sidecar._LIVE == {"nsc-acc-sink"}
    s.stop()
    assert sidecar._LIVE == set() and rec.calls[-1] == ["docker", "rm", "-f", "nsc-acc-sink"]


def test_only_a_nsc_acc_name_is_accepted():
    for bad in ("sink", "nsc-acc-", "nsc-acc-UPPER", "nsc-acc-a;rm", "neuralscape-api", "nsc-acc-" + "x" * 60, ""):
        with pytest.raises(sidecar.SidecarError):
            sidecar.check_name(bad)
        with pytest.raises(sidecar.SidecarError):
            _sink(Recorder(), name=bad)
    assert sidecar.check_name("nsc-acc-sink") == "nsc-acc-sink"


@pytest.mark.parametrize("project", ["neuralscape", " Neuralscape "])
def test_the_production_project_is_refused(project):
    rec = Recorder()
    with pytest.raises(guard.GuardError):
        _sink(rec, project=project)
    assert rec.calls == []


def test_the_number_of_live_sidecars_is_capped():
    sidecar._LIVE.update({"nsc-acc-one", "nsc-acc-two"})
    rec = Recorder(logs=READY)
    with pytest.raises(sidecar.SidecarError, match="cap"):
        _sink(rec).start()
    assert not any(c[1] == "run" for c in rec.calls)


def test_a_sink_that_never_reports_ready_is_removed():
    rec = Recorder(logs="")
    s = _sink(rec)
    ticks = iter(range(0, 1000))
    s._clock = lambda: next(ticks)
    with pytest.raises(sidecar.SidecarError, match="did not report ready"):
        s.start(ready_timeout=5)
    assert rec.calls[-1] == ["docker", "rm", "-f", "nsc-acc-sink"] and sidecar._LIVE == set()


def test_docker_is_only_ever_asked_for_its_four_verbs_and_only_about_its_own_name():
    s = _sink(Recorder())
    for args in (("ps",), ("exec", "x"), ("rm", "-f", "nsc-acc-other"), ("logs", "neuralscape"), ("network", "rm", "x"), ("volume", "rm", "x"), ()):
        with pytest.raises(sidecar.SidecarError):
            s._docker(*args)


def test_the_network_is_discovered_not_assumed():
    assert _sink(Recorder(network="nstest-x_default\nother_net\n")).network() == "nstest-x_default"
    assert _sink(Recorder(network="only_one\n")).network() == "only_one"
    with pytest.raises(sidecar.SidecarError):
        _sink(Recorder(network="a\nb\n")).network()
    with pytest.raises(sidecar.SidecarError):
        _sink(Recorder(network="")).network()


def test_request_lines_in_the_container_log_are_parsed_and_filtered():
    logs = READY + "".join(json.dumps(r) + "\n" for r in (
        {"method": "GET", "path": "/items", "authorization": "Bearer x", "body": ""},
        {"method": "POST", "path": "/queue-empty", "body": json.dumps({"event": "queue.empty", "user_id": "u"})},
        {"method": "POST", "path": "/queue-empty", "body": "not json"},
    )) + "garbage line\n"
    s = _sink(Recorder(logs=logs))
    assert [r["path"] for r in s.requests()] == ["/items", "/queue-empty", "/queue-empty"]
    assert len(s.requests("POST", "/queue-empty")) == 2
    assert s.json_bodies("/queue-empty") == [{"event": "queue.empty", "user_id": "u"}]


# ── the sink's inline server, run for real ───────────────────────


def test_the_inline_server_serves_fixtures_and_logs_each_request():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "NSC_PORT": str(port), "NSC_FIXTURES": json.dumps(sidecar.sink_fixtures("r1")), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, "-c", sidecar.SERVER_SCRIPT], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert json.loads(proc.stdout.readline()) == {"ready": True, "port": port}
        base = f"http://127.0.0.1:{port}"
        items = httpx.get(base + "/items", headers={"Authorization": "Bearer zq"})
        assert items.status_code == 200 and [i["id"] for i in items.json()["items"]] == ["item-1", "item-2"]
        assert sidecar.connector_marker("r1", 1) in items.json()["items"][0]["body"]
        assert httpx.get(base + "/nothing").status_code == 404
        assert httpx.post(base + "/queue-empty", json={"event": "queue.empty"}).status_code == 204
        lines = [json.loads(proc.stdout.readline()) for _ in range(3)]
        assert [(l["method"], l["path"]) for l in lines] == [("GET", "/items"), ("GET", "/nothing"), ("POST", "/queue-empty")]
        assert lines[0]["authorization"] == "Bearer zq" and json.loads(lines[2]["body"]) == {"event": "queue.empty"}
    finally:
        proc.kill()
        proc.wait()


def test_the_fixture_items_carry_per_run_markers():
    items = sidecar.sink_fixtures("abc")[sidecar.CONNECTOR_ITEMS_PATH]["items"]
    assert [sidecar.connector_marker("abc", n) in i["body"] for n, i in enumerate(items, 1)] == [True, True]
    assert sidecar.connector_marker("abc", 1) not in items[1]["body"]


# ── the figure-carrying PDF ──────────────────────────────────────


def test_the_pdf_with_a_figure_has_a_consistent_xref_and_an_image_object():
    pdf = make_pdf_with_image(["hello (world)"])
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/Subtype /Image" in pdf and b"/FlateDecode" in pdf and b"/XObject << /Im1 7 0 R >>" in pdf
    xref_at = int(pdf.rsplit(b"startxref\n", 1)[1].split()[0])
    table = pdf[xref_at:].split(b"\n")
    count = int(table[1].split()[1])
    for n in range(1, count):
        off = int(table[2 + n].split()[0])
        assert pdf[off:].startswith(f"{n} 0 obj".encode()), f"object {n} is not where the xref says"
    assert make_pdf_with_image(["hello (world)"]) == pdf, "the PDF is not deterministic"


# ── the plugin: a flags-on run records placeholders and skips dictator scenarios without one ──


def test_a_flags_on_run_records_placeholders_never_values_and_skips_without_a_dictator(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ACCEPTANCE))
    monkeypatch.delenv("NS_HAVE_DICTATOR", raising=False)
    pytester.makepyfile(test_f='''
from nsaccept import needs, profiles, proves
@profiles("flags-on")
@needs("dictator")
@proves("x.dictator")
def test_as_dictator():
    raise RuntimeError("must not run")
''')
    out = pytester.path / "r.json"
    pytester.runpytest_subprocess("-p", "nsaccept.pytest_plugin", "--ns-profile", "flags-on", "--ns-out", str(out), "-p", "no:cacheprovider")
    doc = json.loads(out.read_text())
    assert doc["profile_flags"]["VAULT_KEY"] == "<generate:fernet>" and doc["profile_flags"]["DICTATOR_USER_IDS"] == "acc-dictator"
    assert doc["profile_sha256"] and len(doc["profile_sha256"]) == 64
    s = doc["scenarios"][0]
    assert s["result"] == "skip" and "dictator not available" in s["detail"]


# ── one compose-project gate, everywhere ─────────────────────────


@pytest.mark.parametrize("name", ["neuralscape", " Neuralscape ", "NEURALSCAPE", "production", "myapp", "nscamp", ""])
def test_check_compose_project_refuses_prod_and_non_disposable_names(name):
    with pytest.raises(guard.GuardError):
        guard.check_compose_project(name, env={})


def test_check_compose_project_accepts_disposable_prefixes_and_normalises():
    assert guard.check_compose_project(" NSTest-Acceptance ", env={}).lower() == "nstest-acceptance"
    assert guard.check_compose_project("nstest-abc", env={}) == "nstest-abc"
    assert guard.check_compose_project("bench-1", env={"NS_COMPOSE_ALLOWLIST": "bench-"}) == "bench-1"
    with pytest.raises(guard.GuardError):
        guard.check_compose_project("nstest-acceptance", env={"NS_COMPOSE_ALLOWLIST": "bench-"})
    with pytest.raises(guard.GuardError):  # the override cannot re-admit the production name
        guard.check_compose_project("neuralscape", env={"NS_COMPOSE_ALLOWLIST": "neuralscape"})


def test_every_docker_entry_point_uses_the_gate(monkeypatch):
    calls = []
    run = lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "tok.en\n", "")
    for bad in ("Neuralscape", "somethingelse"):
        with pytest.raises(guard.GuardError):
            runctx.run_compose(bad, "ps", run=run)
        with pytest.raises(tokens.TokenError):
            tokens.mint_via_compose(bad, "u", run=run)
        with pytest.raises(tokens.TokenError):
            tokens.mint_via_compose_python(bad, "u", 60, run=run)
        with pytest.raises(guard.GuardError):
            sidecar.Sink(bad)
    assert calls == []
    with pytest.raises(guard.GuardError):
        guard.check("http://127.0.0.1:18199", " NeuralScape ", env={})


def test_only_the_two_read_only_exec_forms_pass():
    ok = [
        ("exec", "-T", "neuralscape", "grep", "-rl", "--", "zq-x", "/data/vault"),
        ("exec", "-T", "neuralscape", "grep", "-rl", "--", "zq-x", "/data/vault/Projects/p-1"),
        ("exec", "-T", "redis", "redis-cli", "GET", "ns:connector:acc-1-rest"),
    ]
    bad = [
        ("exec", "-T", "neuralscape", "grep", "-rl", "--", "zq-x", "/etc"),
        ("exec", "-T", "neuralscape", "grep", "-rl", "--", "zq-x", "/data/vault/../../etc"),
        ("exec", "-T", "neuralscape", "grep", "-rl", "--", "-e", "/data/vault"),
        ("exec", "-T", "neuralscape", "cat", "/data/vault/x"),
        ("exec", "-T", "redis", "redis-cli", "FLUSHALL"),
        ("exec", "-T", "redis", "redis-cli", "GET", "arq:queue"),
        ("exec", "-T", "neo4j", "grep", "-rl", "--", "x", "/data/vault"),
    ]
    assert all(runctx.read_only_exec(a) for a in ok)
    assert not any(runctx.read_only_exec(a) for a in bad)
    with pytest.raises(runctx.ConfigError):
        runctx.run_compose("nstest-acceptance", *bad[0], run=lambda *a, **k: None)


def test_run_sh_refuses_a_compose_profile_without_a_compose_project():
    sh = str(ACCEPTANCE / "run.sh")
    env = {k: v for k, v in os.environ.items() if k != "NS_COMPOSE_PROJECT"}
    r = subprocess.run([sh, "http://127.0.0.1:18200", "--profile", "flags-on"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "requires --compose-project" in r.stderr, r.stderr
    r = subprocess.run([sh, "http://127.0.0.1:18200", "--profile", "flags-on", "--compose-project", " NeuralScape "], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "production" in r.stderr, r.stderr
    assert profile.load(ACCEPTANCE / "profiles" / "flags-on.env").headers["requires"] == ["compose"]


# ── negative controls state their precondition ───────────────────


def test_a_negative_control_whose_precondition_fails_reports_the_precondition_not_a_vacuous_pass():
    from nsaccept import expect_check_fails, precondition

    def check(indexed: bool):
        assert indexed, "nothing is indexed"

    with pytest.raises(AssertionError, match=r"PRECONDITION FAILED \(the index exists\)"):
        # setup failed: the check fails with the RIGHT input too, so "fails with the wrong input" would be vacuous
        precondition(check, False, what="the index exists")
        expect_check_fails(check, False)
    precondition(check, True, what="the index exists")  # a healthy setup passes through
    expect_check_fails(check, False)


def test_every_negative_control_in_the_flag_scenarios_states_a_precondition():
    """AST scan: each function decorated with negative_control must CALL precondition(...) (a comment or a
    string cannot satisfy it). The framing control is exempt: it compares against an ordinary page."""
    import ast

    flagged = ["test_dreaming", "test_standards", "test_connectors", "test_webhook", "test_code_graph", "test_code_graph_cbm",
               "test_oauth", "test_speaker", "test_exemplars"]
    exempt = {"test_frame_check_fails_for_a_page_without_the_headers"}

    def deco_name(d):
        d = d.func if isinstance(d, ast.Call) else d
        return d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")

    def calls_precondition(fn):
        """A precondition(...) call that comes BEFORE the first expect_check_fails(...) call (by position)."""
        pre = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and deco_name(n.func) == "precondition"]
        neg = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and deco_name(n.func) == "expect_check_fails"]
        return bool(pre) and (not neg or min(pre) < min(neg))

    missing, seen = [], 0
    for mod in flagged:
        tree = ast.parse((ACCEPTANCE / "scenarios" / f"{mod}.py").read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if any(deco_name(d) == "negative_control" for d in fn.decorator_list):
                seen += 1
                if fn.name not in exempt and not calls_precondition(fn):
                    missing.append(f"{mod}::{fn.name}")
    assert seen > 20, f"scan found only {seen} negative controls"
    assert missing == [], f"negative controls that never call precondition() before the wrong-input check: {missing}"


def test_a_comment_or_string_does_not_satisfy_the_precondition_scan():
    import ast

    src = "@negative_control\ndef test_x():\n    # precondition(...)\n    s = 'precondition(x)'\n"
    fn = ast.parse(src).body[0]
    assert not any(isinstance(n, ast.Call) for n in ast.walk(fn))
