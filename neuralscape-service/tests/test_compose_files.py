"""Static guards for the docker-compose files and the Dockerfile base images.

No Docker needed: PyYAML parses every ``docker-compose*.yml`` and a small
re-implementation of compose's merge rules builds the same effective service
maps that ``docker compose -f base -f overlay config`` does (the VPS gate checks
the real thing in ``integration_tests/test_compose_config.py``).

In the packaged test image only the service tree exists (no compose files), so
each test returns early when the repo-level files are absent, the same pattern as
``test_baseline_reproductions._compose_volumes``. The host CI job runs them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_DIR.parent
BASE = "docker-compose.yml"
OVERLAYS = ("docker-compose.vps.yml", "docker-compose.deploy.yml", "docker-compose.prod.yml")
SNAPSHOT = "docker-compose.snapshot-test.yml"
ALL_FILES = (BASE, *OVERLAYS, SNAPSHOT)

APP_SERVICES = (
    "neuralscape",
    "neuralscape-worker",
    "neuralscape-graph-worker",
    "neuralscape-ingest-worker",
)
STORES = ("neo4j", "redis", "qdrant")
# Third-party images that used to float (:latest / a bare major) and must now be
# pinned to an exact release AND a digest.
DIGEST_PINNED = ("docling", "cloudflared", "caddy", "autoheal", "docker-socket-proxy")
LOOPBACK = {"127.0.0.1", "::1"}
# The one deliberate public listener: Caddy in the production overlay.
PUBLIC_EXCEPTIONS = {("docker-compose.prod.yml", "caddy"): {"80", "443"}}


# ── loading + merging ───────────────────────────────────────────────


class _Override:
    def __init__(self, value):
        self.value = value


class _Reset:
    pass


class _Loader(yaml.SafeLoader):
    pass




def _tagged_value(loader: yaml.SafeLoader, node: yaml.Node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_constructor("!override", lambda ld, node: _Override(_tagged_value(ld, node)))
_Loader.add_constructor("!reset", lambda ld, node: _Reset())

# Sequence keys compose appends across files; every other sequence is replaced.
_APPENDED = {"ports", "volumes", "expose", "networks", "profiles", "cap_add", "dns", "secrets", "configs"}


def _have_repo() -> bool:
    return all((REPO_ROOT / name).is_file() for name in ALL_FILES)


def _load(name: str) -> dict:
    with open(REPO_ROOT / name, encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_Loader)  # noqa: S506 - SafeLoader subclass


def _env_map(value) -> dict:
    if isinstance(value, _Override):
        return _env_map(value.value)
    if value is None:
        return {}
    if isinstance(value, list):
        out = {}
        for item in value:
            key, _, val = str(item).partition("=")
            out[key] = val
        return out
    return dict(value)


def _merge(base, over, key=""):
    if isinstance(over, _Override):
        return over.value
    if isinstance(over, _Reset):
        return None
    if isinstance(base, dict) and isinstance(over, dict):
        merged = dict(base)
        for k, v in over.items():
            if k == "environment":
                merged[k] = {**_env_map(base.get(k)), **_env_map(v)}
            else:
                merged[k] = _merge(base.get(k), v, k)
        return merged
    if isinstance(base, list) and isinstance(over, list) and key in _APPENDED:
        return base + [item for item in over if item not in base]
    return over


def _merged(*names: str) -> dict:
    """The effective top-level mapping after merging ``names`` left to right."""
    result: dict = {}
    for name in names:
        result = _merge(result, _load(name))
    return result


def _services(*names: str) -> dict:
    return _merged(*names)["services"]


def _host_ip(port) -> tuple[str | None, str]:
    """(host ip or None when unbound, published host port) of one ``ports`` entry."""
    if isinstance(port, dict):
        return port.get("host_ip"), str(port.get("published", ""))
    text = str(port).split("/")[0]
    if text.startswith("["):
        ip, _, rest = text[1:].partition("]")
        return ip, rest.lstrip(":").split(":")[0]
    parts = text.split(":")
    if len(parts) == 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return None, parts[0]
    return None, ""


def _plain(value):
    return value.value if isinstance(value, _Override) else value


def _published(services: dict) -> list[tuple[str, str | None, str]]:
    return [
        (name, *_host_ip(p))
        for name, svc in services.items()
        for p in (_plain(svc.get("ports")) or [])
    ]


# ── ports ───────────────────────────────────────────────────────────


def test_no_file_publishes_a_port_on_every_interface() -> None:
    if not _have_repo():
        return
    for name in ALL_FILES:
        for service, ip, host_port in _published(_load(name).get("services", {})):
            if host_port in PUBLIC_EXCEPTIONS.get((name, service), set()):
                continue
            assert ip in LOOPBACK, (
                f"{name}: {service} publishes host port {host_port!r} on "
                f"{ip or 'every interface'}; bind it to 127.0.0.1"
            )


def test_every_overlay_combination_keeps_ports_on_loopback() -> None:
    if not _have_repo():
        return
    combos = [
        (BASE,),
        (BASE, "docker-compose.vps.yml"),
        (BASE, "docker-compose.vps.yml", "docker-compose.deploy.yml"),
        (BASE, "docker-compose.prod.yml"),
        (BASE, "docker-compose.prod.yml", "docker-compose.deploy.yml"),
    ]
    for combo in combos:
        for service, ip, host_port in _published(_services(*combo)):
            if service == "caddy" and host_port in {"80", "443"} and "docker-compose.prod.yml" in combo:
                continue
            assert ip in LOOPBACK, f"{'+'.join(combo)}: {service}:{host_port} bound to {ip or 'all interfaces'}"


def test_base_publishes_the_dev_ports_on_loopback_only() -> None:
    if not _have_repo():
        return
    published = {(s, p) for s, ip, p in _published(_services(BASE)) if ip == "127.0.0.1"}
    for expected in [
        ("neo4j", "7474"), ("neo4j", "7687"), ("redis", "6379"), ("qdrant", "6333"),
        ("qdrant", "6334"), ("neuralscape", "8199"), ("docling", "5001"),
        ("cbm-bridge", "8200"), ("resolver-svc", "8201"),
    ]:
        assert expected in published, f"dev stack lost loopback port {expected}"


def test_vps_overlay_publishes_nothing_but_the_loopback_api() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.vps.yml")
    assert _published(services) == [("neuralscape", "127.0.0.1", "8199")]


def test_prod_overlay_publishes_only_caddy_and_no_database_port() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.prod.yml")
    assert sorted(_published(services)) == [("caddy", None, "443"), ("caddy", None, "80")]
    for store in (*STORES, "docling", "cbm-bridge", "resolver-svc", "neuralscape"):
        assert not services[store].get("ports"), f"{store} must not publish a port in prod"


# ── secrets and images ──────────────────────────────────────────────


def test_neo4j_password_has_no_default_anywhere() -> None:
    if not _have_repo():
        return
    for name in ALL_FILES:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert ":-neuralscape" not in text, f"{name} still falls back to the 'neuralscape' password"
        for match in re.finditer(r"\$\{NEO4J_PASSWORD([^}]*)\}", text):
            assert match.group(1).startswith(":?"), (
                f"{name}: {match.group(0)} must be a required variable (${{NEO4J_PASSWORD:?...}})"
            )
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^NEO4J_PASSWORD=\s*$", example, re.M), ".env.example must ship NEO4J_PASSWORD blank"
    assert not re.search(r"^NEO4J_PASSWORD=\S", example, re.M)


def test_every_required_compose_variable_is_documented_in_an_env_template() -> None:
    if not _have_repo():
        return
    templates = (REPO_ROOT / ".env.example").read_text(encoding="utf-8") + (
        REPO_ROOT / ".env.production"
    ).read_text(encoding="utf-8")
    required = set()
    for name in ALL_FILES:
        required |= set(re.findall(r"\$\{([A-Z0-9_]+):\?", (REPO_ROOT / name).read_text(encoding="utf-8")))
    assert {"NEO4J_PASSWORD", "REDIS_PASSWORD", "DOMAIN"} <= required
    for var in required:
        assert re.search(rf"^#?\s*{var}=", templates, re.M), f"{var} is required by compose but in no .env template"


def _all_images() -> list[tuple[str, str, str]]:
    out = []
    for name in ALL_FILES:
        for service, svc in _load(name).get("services", {}).items():
            if "image" in svc:
                out.append((name, service, str(svc["image"])))
    return out


def test_no_compose_image_floats_on_latest_or_has_no_tag() -> None:
    if not _have_repo():
        return
    for name, service, image in _all_images():
        match = re.fullmatch(r"\$\{NS_IMAGE:-([^}]+)\}", image)
        if match and name in {"docker-compose.vps.yml", "docker-compose.deploy.yml"}:
            # The project's own image: deploy.yml loads the sha tag and tags it :latest
            # locally on purpose; NS_IMAGE names the exact build. See the vps overlay.
            assert service in APP_SERVICES
            continue
        if "@sha256:" in image:
            continue  # a digest is immutable, whatever the tag says
        last = image.rsplit("/", 1)[-1]
        assert ":" in last, f"{name}: {service} image {image!r} has no tag"
        assert not last.endswith(":latest"), f"{name}: {service} image {image!r} is :latest"


def test_formerly_floating_third_party_images_are_pinned_to_a_digest() -> None:
    if not _have_repo():
        return
    seen = set()
    for _, service, image in _all_images():
        if service in DIGEST_PINNED:
            seen.add(service)
            assert re.search(r"@sha256:[0-9a-f]{64}$", image), f"{service}: {image!r} needs a digest"
            assert re.search(r":(v?\d[\w.]*)(-[\w.]+)?@", image) or service == "autoheal", (
                f"{service}: {image!r} needs an exact version tag next to the digest"
            )
    assert seen == set(DIGEST_PINNED)


def test_app_services_in_vps_and_deploy_overlays_use_the_ns_image_variable() -> None:
    if not _have_repo():
        return
    for name in ("docker-compose.vps.yml", "docker-compose.deploy.yml"):
        services = _load(name)["services"]
        for service in APP_SERVICES:
            assert str(services[service]["image"]).startswith("${NS_IMAGE:-"), (name, service)


# ── production overlay is a thin overlay ────────────────────────────


def test_prod_overlay_defines_only_caddy_besides_overrides_of_base_services() -> None:
    if not _have_repo():
        return
    base, prod = _load(BASE)["services"], _load("docker-compose.prod.yml")["services"]
    assert set(prod) - set(base) == {"caddy"}
    for service in APP_SERVICES:
        # Anything beyond environment/ports here would be a second copy that can drift.
        extra = set(prod.get(service, {})) - {"environment", "ports"}
        assert not extra, f"prod overlay re-declares {sorted(extra)} of {service}: change the base file instead"


def test_prod_merge_keeps_all_four_app_services_and_the_stores() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.prod.yml")
    for service in (*APP_SERVICES, *STORES, "autoheal", "docker-socket-proxy", "caddy"):
        assert service in services, f"prod merge lost {service}"
    assert services["neuralscape-graph-worker"]["command"] == ["arq", "worker.GraphWorkerSettings"]
    assert services["neuralscape-ingest-worker"]["command"] == ["arq", "worker.IngestWorkerSettings"]
    for service in APP_SERVICES:
        assert services[service]["build"]["target"] == "runtime", f"{service} must build the runtime stage"


def test_prod_redis_requires_a_password_and_every_client_uses_it() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.prod.yml")
    redis = services["redis"]
    assert "--requirepass" in " ".join(redis["command"])
    assert "${REDIS_PASSWORD:?" in redis["environment"]["REDISCLI_AUTH"]
    assert "-a" not in redis["healthcheck"]["test"], "keep the password out of argv"
    for name, svc in services.items():
        url = _env_map(svc.get("environment")).get("REDIS_URL")
        if url:
            assert re.fullmatch(r"redis://:\$\{REDIS_PASSWORD:\?[^}]*\}@redis:6379", url), f"{name}: {url}"


def test_prod_caddy_needs_a_domain_and_waits_for_the_api() -> None:
    if not _have_repo():
        return
    caddy = _services(BASE, "docker-compose.prod.yml")["caddy"]
    assert "${DOMAIN:?" in caddy["environment"]["DOMAIN"]
    assert caddy["depends_on"]["neuralscape"]["condition"] == "service_healthy"
    assert "{$DOMAIN}" in (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")
    assert "reverse_proxy neuralscape:8199" in (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")


# ── VPS overlay ─────────────────────────────────────────────────────


def test_vps_app_images_are_never_pulled() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.vps.yml")
    for service in APP_SERVICES:
        assert services[service]["pull_policy"] == "never", (
            f"{service}: a manual `docker compose pull` would replace the deployed build with registry :latest"
        )


def test_vps_docling_flag_matches_the_profile_that_provides_it() -> None:
    if not _have_repo():
        return
    services = _services(BASE, "docker-compose.vps.yml")
    assert services["docling"]["profiles"] == ["ingest-heavy"]
    env = _env_map(services["neuralscape-ingest-worker"]["environment"])
    assert env["DOCLING_ENABLED"] == "${DOCLING_ENABLED:-false}", (
        "docling is off by default on the VPS, so the worker must not default to using it"
    )
    # ...while the base file (dev) keeps docling on with the container present.
    base_env = _env_map(_services(BASE)["neuralscape-ingest-worker"]["environment"])
    assert base_env["DOCLING_ENABLED"] == "${DOCLING_ENABLED:-true}"
    assert "profiles" not in _services(BASE)["docling"]


# ── profiles ────────────────────────────────────────────────────────


def test_each_profile_service_is_opt_in_and_fully_wired() -> None:
    if not _have_repo():
        return
    services = _services(BASE)
    for service, profile in (
        ("cbm-bridge", "cbm"), ("resolver-svc", "resolver"), ("cloudflared", "tunnel"), ("neuralscape-test", "test"),
    ):
        assert services[service]["profiles"] == [profile], service

    consumers = [_env_map(services[s]["environment"]) for s in ("neuralscape", "neuralscape-ingest-worker")]
    for env in consumers:
        # cbm: the bridge only does something when the service is told to register it.
        assert env["CBM_ENABLED"] == "${CBM_ENABLED:-false}"
        assert env["CBM_BRIDGE_URL"] == "${CBM_BRIDGE_URL:-http://cbm-bridge:8200}"
        # resolver: lsp is the only mode that talks to resolver-svc; default stays jedi.
        assert env["CODE_NEIGHBORS_RESOLVER"] == "${CODE_NEIGHBORS_RESOLVER:-jedi}"
        assert env["CODE_RESOLVER_URL"] == "${CODE_RESOLVER_URL:-http://resolver-svc:8201}"
    assert _env_map(services["cbm-bridge"]["environment"])["PORT"] == 8200
    assert _env_map(services["resolver-svc"]["environment"])["PORT"] == 8201

    cloudflared = services["cloudflared"]
    assert cloudflared["environment"]["TUNNEL_TOKEN"] == "${CLOUDFLARE_TUNNEL_TOKEN}"
    assert cloudflared["depends_on"]["neuralscape"]["condition"] == "service_healthy"
    assert services["neuralscape-test"]["build"]["target"] == "test"


def test_resolver_timeout_uses_the_name_the_resolver_reads() -> None:
    if not _have_repo():
        return
    env = _env_map(_services(BASE)["resolver-svc"]["environment"])
    assert "RESOLVER_TIMEOUT" not in env, "nothing reads RESOLVER_TIMEOUT"
    assert env["RESOLVER_REQUEST_TIMEOUT"] == "${RESOLVER_REQUEST_TIMEOUT:-30}"
    main = REPO_ROOT / "resolver-svc" / "main.py"
    if main.is_file():
        source = main.read_text(encoding="utf-8")
        assert 'os.getenv("RESOLVER_REQUEST_TIMEOUT", "30")' in source, "resolver-svc default moved: update compose"


# ── autoheal and the Docker socket ──────────────────────────────────

_SOCKET = "/var/run/docker.sock"
def _mounts_socket(svc: dict) -> bool:
    return any(_SOCKET in str(v) for v in svc.get("volumes", []))


def test_only_the_socket_proxy_mounts_the_docker_socket() -> None:
    if not _have_repo():
        return
    for name in ALL_FILES:
        for service, svc in _load(name).get("services", {}).items():
            if _mounts_socket(svc):
                assert service == "docker-socket-proxy", f"{name}: {service} mounts the Docker socket"
    proxy = _services(BASE)["docker-socket-proxy"]
    assert _mounts_socket(proxy)
    assert all(str(v).endswith(":ro") for v in proxy["volumes"])
    assert not proxy.get("ports"), "the proxy must not be published"


def test_autoheal_talks_to_the_proxy_over_an_internal_network() -> None:
    if not _have_repo():
        return
    merged = _merged(BASE)
    autoheal = merged["services"]["autoheal"]
    proxy = merged["services"]["docker-socket-proxy"]
    assert autoheal["environment"]["DOCKER_SOCK"] == "tcp://docker-socket-proxy:2375"
    assert autoheal.get("network_mode") != "host"
    assert autoheal["networks"] == ["autoheal-net"] and proxy["networks"] == ["autoheal-net"]
    assert merged["networks"]["autoheal-net"]["internal"] is True
    # Only autoheal and the proxy are on that network.
    members = [s for s, v in merged["services"].items() if "autoheal-net" in (v.get("networks") or [])]
    assert sorted(members) == ["autoheal", "docker-socket-proxy"]


def test_socket_proxy_config_allows_only_container_list_and_restart() -> None:
    if not _have_repo():
        return
    proxy = _services(BASE)["docker-socket-proxy"]
    mount = "./ops/docker-socket-proxy.haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg.template:ro"
    assert mount in proxy["volumes"]
    # No env switch may re-open API sections: the custom config ignores them, but a
    # reader must not be misled into thinking CONTAINERS/POST/EXEC are in play.
    assert set(proxy["environment"]) == {"DISABLE_IPV6"}
    cfg = (REPO_ROOT / "ops" / "docker-socket-proxy.haproxy.cfg").read_text(encoding="utf-8")
    rules = [line.strip() for line in cfg.splitlines() if line.strip().startswith("http-request")]
    assert len(rules) == 3, rules
    assert rules[0].startswith("http-request allow if METH_GET") and rules[0].endswith("/containers/json$ }")
    assert rules[1].startswith("http-request allow if METH_POST") and rules[1].endswith("/restart$ }")
    assert rules[2] == "http-request deny", "the last rule must deny everything not allowed above"
    assert "env(" not in cfg.split("frontend dockerfrontend", 1)[1]
    assert "${BIND_CONFIG}" in cfg and "bind" in cfg, "the image seds BIND_CONFIG into the bind line"
    for forbidden in ("exec", "create", "/start", "images", "volumes", "events"):
        assert not any(forbidden in rule for rule in rules[:2]), forbidden


# ── Dockerfile ──────────────────────────────────────────────────────


def test_dockerfile_pins_uv_and_never_floats_a_base_image() -> None:
    dockerfile = SERVICE_DIR / "Dockerfile"
    if not dockerfile.is_file():
        return
    text = dockerfile.read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert ":latest" not in code
    uv_refs = re.findall(r"--from=(ghcr\.io/astral-sh/uv:[^\s]+)", text)
    assert len(uv_refs) == 2, "builder and test stages both install uv"
    for ref in uv_refs:
        assert re.fullmatch(r"ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", ref), ref
    assert len(set(uv_refs)) == 1, "builder and test stages must use the same uv"


def test_overlay_pinning_the_project_image_on_latest_never_pulls_it() -> None:
    """A fallback to the project's registry ``:latest`` must come with ``pull_policy: never``.

    deploy.yml pushes only immutable sha tags to GHCR and tags ``:latest`` locally on
    the VPS, so a pull of that name would fetch a stale or missing image.
    """
    if not _have_repo():
        return
    checked = 0
    for name in OVERLAYS:
        for service, spec in (_load(name).get("services") or {}).items():
            image = str((spec or {}).get("image", ""))
            if "/neuralscape:latest" in image:
                checked += 1
                assert (spec or {}).get("pull_policy") == "never", (name, service)
    assert checked >= 8  # four services in each of the vps and deploy overlays
