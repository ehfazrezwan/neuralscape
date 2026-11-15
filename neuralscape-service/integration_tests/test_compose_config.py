"""Every compose combination and profile must render with the docker CLI.

Run this where a ``docker`` CLI with the compose plugin and the repository
checkout are available (the VPS gate host, or a laptop: ``docker compose config``
only reads files, it needs no daemon and starts nothing). It is NOT meaningful
inside the packaged test image, which has neither: that is a failure with a
clear message, never a skip.

    cd neuralscape-service && pytest integration_tests/test_compose_config.py -q

It proves, against the real compose parser, what ``tests/test_compose_files.py``
only approximates: the base file, the VPS overlay, the CD overlay and the prod
overlay merge and validate in every supported stacking; each profile brings its
service in and nothing else; the required variables really are required; no
published port leaves loopback except Caddy's 80/443 in the prod stacking.

What it cannot prove: the Cloudflare tunnel itself (needs a real tunnel token and
a Cloudflare account: only the ``tunnel`` profile's rendering is checked) and
the live behaviour of the Docker socket proxy (checked on a running stack).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = (
    "docker-compose.yml",
    "docker-compose.vps.yml",
    "docker-compose.deploy.yml",
    "docker-compose.prod.yml",
)
_RUN = uuid.uuid4().hex[:12]  # throwaway values, generated per run
DUMMY = {
    "NEO4J_PASSWORD": f"neo-{_RUN}",
    "REDIS_PASSWORD": f"red-{_RUN}",
    "DOMAIN": "ns.example.test",
    "CLOUDFLARE_TUNNEL_TOKEN": f"tun-{_RUN}",
}
APP = {"neuralscape", "neuralscape-worker", "neuralscape-graph-worker", "neuralscape-ingest-worker"}

STACKS = {
    "base": ("docker-compose.yml",),
    "base+vps": ("docker-compose.yml", "docker-compose.vps.yml"),
    "base+vps+deploy": ("docker-compose.yml", "docker-compose.vps.yml", "docker-compose.deploy.yml"),
    "base+prod": ("docker-compose.yml", "docker-compose.prod.yml"),
    "base+prod+deploy": ("docker-compose.yml", "docker-compose.prod.yml", "docker-compose.deploy.yml"),
}
# profile -> (stack that defines it, services it must add, services that exist only with it)
PROFILES = {
    "cbm": ("base", {"cbm-bridge"}),
    "resolver": ("base", {"resolver-svc"}),
    "test": ("base", {"neuralscape-test"}),
    "tunnel": ("base+vps", {"cloudflared"}),
    "ingest-heavy": ("base+vps", {"docling"}),
}


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """A scratch project dir: the repo's compose files plus the `.env` they reference."""
    if shutil.which("docker") is None:
        pytest.fail("the docker CLI is not installed here; run this test on a host that has it (see the module docstring)")
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        pytest.fail(f"docker compose plugin unavailable: {probe.stderr.strip()}")
    missing = [name for name in (*COMPOSE_FILES, "Caddyfile") if not (REPO_ROOT / name).is_file()]
    if missing:
        pytest.fail(f"repository files missing next to this test ({missing}): run it from a full checkout")
    directory = tmp_path_factory.mktemp("compose-config")
    for name in (*COMPOSE_FILES, "Caddyfile"):
        shutil.copy(REPO_ROOT / name, directory / name)
    (directory / ".env").write_text("GOOGLE_API_KEY=dummy\nNEURALSCAPE_AUTH_DISABLED=true\n", encoding="utf-8")
    return directory


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in DUMMY and not k.startswith(("COMPOSE_", "NS_IMAGE"))}
    env.update({**DUMMY, **extra})
    return env


def _config(project: Path, stack: str, profiles: tuple[str, ...] = (), env: dict | None = None):
    cmd = ["docker", "compose", "--project-name", "nscfgtest"]
    for name in STACKS[stack]:
        cmd += ["-f", name]
    for profile in profiles:
        cmd += ["--profile", profile]
    cmd += ["config", "--format", "json"]
    return subprocess.run(cmd, cwd=project, capture_output=True, text=True, env=env or _env(), check=False)


def _services(project: Path, stack: str, profiles: tuple[str, ...] = ()) -> dict:
    result = _config(project, stack, profiles)
    assert result.returncode == 0, f"{stack} {profiles}: {result.stderr}"
    return json.loads(result.stdout)["services"]


@pytest.mark.parametrize("stack", sorted(STACKS))
def test_every_stacking_renders(project: Path, stack: str) -> None:
    services = _services(project, stack)
    assert APP <= set(services), f"{stack} lost an app service: {APP - set(services)}"
    assert {"neo4j", "redis", "qdrant", "autoheal", "docker-socket-proxy"} <= set(services)
    # profile services are absent unless asked for
    assert not {"cbm-bridge", "resolver-svc", "neuralscape-test", "cloudflared"} & set(services)


def test_all_profiles_together_render_on_every_stack(project: Path) -> None:
    everything = tuple(PROFILES)
    for stack in STACKS:
        result = _config(project, stack, everything)
        assert result.returncode == 0, f"{stack} with all profiles: {result.stderr}"


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_each_profile_brings_exactly_its_service(project: Path, profile: str) -> None:
    stack, added = PROFILES[profile]
    without = set(_services(project, stack))
    with_profile = set(_services(project, stack, (profile,)))
    assert with_profile - without == added
    assert without <= with_profile


def test_cbm_and_resolver_profiles_are_wired_into_the_services_that_use_them(project: Path) -> None:
    services = _services(project, "base", ("cbm", "resolver"))
    for name in ("neuralscape", "neuralscape-ingest-worker"):
        env = services[name]["environment"]
        assert env["CBM_ENABLED"] == "false" and env["CBM_BRIDGE_URL"] == "http://cbm-bridge:8200"
        assert env["CODE_NEIGHBORS_RESOLVER"] == "jedi" and env["CODE_RESOLVER_URL"] == "http://resolver-svc:8201"
    # the switch is honoured from the environment, so the profile can actually be used
    on = json.loads(
        _config(project, "base", ("cbm", "resolver"), _env(CBM_ENABLED="true", CODE_NEIGHBORS_RESOLVER="lsp")).stdout
    )["services"]
    assert on["neuralscape"]["environment"]["CBM_ENABLED"] == "true"
    assert on["neuralscape-ingest-worker"]["environment"]["CODE_NEIGHBORS_RESOLVER"] == "lsp"
    assert services["resolver-svc"]["environment"]["RESOLVER_REQUEST_TIMEOUT"] == "30"
    assert "RESOLVER_TIMEOUT" not in services["resolver-svc"]["environment"]


def test_tunnel_profile_renders_with_the_token_it_is_given(project: Path) -> None:
    services = _services(project, "base+vps", ("tunnel",))
    assert services["cloudflared"]["environment"]["TUNNEL_TOKEN"] == DUMMY["CLOUDFLARE_TUNNEL_TOKEN"]
    assert "@sha256:" in services["cloudflared"]["image"]


def test_vps_docling_flag_follows_the_profile_default(project: Path) -> None:
    worker = _services(project, "base+vps")["neuralscape-ingest-worker"]
    assert worker["environment"]["DOCLING_ENABLED"] == "false"
    on = json.loads(_config(project, "base+vps", ("ingest-heavy",), _env(DOCLING_ENABLED="true")).stdout)["services"]
    assert on["neuralscape-ingest-worker"]["environment"]["DOCLING_ENABLED"] == "true"
    assert _services(project, "base")["neuralscape-ingest-worker"]["environment"]["DOCLING_ENABLED"] == "true"


def test_vps_never_pulls_the_deployed_image(project: Path) -> None:
    services = _services(project, "base+vps+deploy")
    for name in APP:
        assert services[name]["pull_policy"] == "never"


@pytest.mark.parametrize("stack", sorted(STACKS))
def test_published_ports_stay_on_loopback(project: Path, stack: str) -> None:
    for name, service in _services(project, stack, tuple(PROFILES)).items():
        for port in service.get("ports", []):
            if name == "caddy" and port["published"] in {"80", "443"}:
                assert "prod" in stack
                continue
            assert port.get("host_ip") in {"127.0.0.1", "::1"}, f"{stack}: {name} publishes {port}"
    if "vps" in stack:
        published = {(n, p["published"]) for n, s in _services(project, stack).items() for p in s.get("ports", [])}
        assert published == {("neuralscape", "8199")}


def test_neo4j_password_is_required(project: Path) -> None:
    env = {k: v for k, v in _env().items() if k != "NEO4J_PASSWORD"}
    result = _config(project, "base", env=env)
    assert result.returncode != 0
    assert "NEO4J_PASSWORD" in result.stderr


@pytest.mark.parametrize("missing", ["REDIS_PASSWORD", "DOMAIN"])
def test_prod_requires_its_own_secrets(project: Path, missing: str) -> None:
    env = {k: v for k, v in _env().items() if k != missing}
    result = _config(project, "base+prod", env=env)
    assert result.returncode != 0 and missing in result.stderr


def test_prod_redis_clients_authenticate(project: Path) -> None:
    services = _services(project, "base+prod")
    for name in APP:
        assert services[name]["environment"]["REDIS_URL"] == f"redis://:{DUMMY['REDIS_PASSWORD']}@redis:6379"
    assert services["redis"]["environment"]["REDISCLI_AUTH"] == DUMMY["REDIS_PASSWORD"]
    assert "--requirepass" in " ".join(services["redis"]["command"])


def test_only_the_socket_proxy_has_the_docker_socket(project: Path) -> None:
    for stack in STACKS:
        for name, service in _services(project, stack, tuple(PROFILES)).items():
            sources = [v.get("source") for v in service.get("volumes", [])]
            assert ("/var/run/docker.sock" in sources) == (name == "docker-socket-proxy"), (stack, name)
