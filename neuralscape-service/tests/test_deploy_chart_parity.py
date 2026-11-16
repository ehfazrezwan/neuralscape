"""Static parity guards for the GKE deployment tree (``deploy/``).

Nothing here needs a cluster, Helm or Terraform. PyYAML parses the chart's
``values.yaml``; the chart templates are Go-templated and so not YAML, so env
entries are read with regexes on ``- name: UPPER_CASE`` lines; Terraform files are
read as text. What is asserted:

* every env var name the chart or Terraform sets is a real ``config.Settings``
  field (case-insensitive) or a documented extension/vendored key;
* ``values.yaml`` ``config:`` holds only a small allowlist of cluster-specific
  overrides and never restates a ``config.py`` default (no second set of defaults);
* no floating ``latest`` image tag (app image, Docling image, Terraform default);
* no reference to settings that no longer exist (the deleted wiki synthesizer);
* the four app Deployments + Docling exist with entrypoints that are real
  ``worker.py`` settings classes and probes that hit the service's real routes.

In the packaged test image only the service tree exists (no ``deploy/``), so every
test returns early when it is absent, the same pattern as ``test_compose_files``.
The host CI job runs them.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic_core import PydanticUndefined

import config

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_DIR.parent
DEPLOY = REPO_ROOT / "deploy"
CHART = DEPLOY / "charts" / "neuralscape"
TEMPLATES = CHART / "templates"
TERRAFORM = DEPLOY / "terraform"

# The only keys values.yaml ``config:`` may carry. Each is a cluster-specific value
# that has no useful config.py default (backend addresses), a mount path the chart
# itself defines, or a deliberate chart-level override of config.py's default. Adding
# a key is a conscious edit here; everything else must fall through to config.py.
CHART_CONFIG_ALLOWLIST = frozenset({
    "NEO4J_URI",  # in-cluster Service address
    "QDRANT_URL",
    "REDIS_URL",
    "MCP_TRANSPORT",  # container serves HTTP; config.py defaults to stdio for the CLI
    "INGEST_STORAGE_DIR",  # mount path of the shared artifact volume
    "OBSIDIAN_VAULT_PATH",  # mount path of the vault scratch volume (extension key)
    "MEM0_TELEMETRY",  # privacy opt-out (config.py leaves it unset)
    "GRAPHITI_TELEMETRY_ENABLED",
    "NEURALSCAPE_PUBLIC_URL",  # only if a private overlay sets it; Terraform does
    "AUTH_PROVIDER",  # overlay override only
    "AUTH_ALLOWED_DOMAINS",
})

REMOVED_SETTINGS_PATTERN = re.compile(r"wiki[_-]?synth", re.IGNORECASE)

APP_DEPLOYMENTS = {
    "api-deployment.yaml": ("neuralscape-api", None),
    "worker-deployment.yaml": ("neuralscape-worker", "WorkerSettings"),
    "graph-worker-deployment.yaml": ("neuralscape-graph-worker", "GraphWorkerSettings"),
    "ingest-worker-deployment.yaml": ("neuralscape-ingest-worker", "IngestWorkerSettings"),
}

_ENV_NAME_RE = re.compile(r"^\s*-\s*name:\s*([A-Z][A-Z0-9_]+)\s*$")
_QUOTED_UPPER_RE = re.compile(r'"([A-Z][A-Z0-9_]{2,})"')
_TF_KEY_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE)


def _present() -> bool:
    return CHART.is_dir() and TERRAFORM.is_dir()


def _setting_names() -> set[str]:
    """Env names Settings reads, plus keys owned by extension/vendored readers."""
    names = {name.upper() for name in config.Settings.model_fields}
    return names | set(config.EXTENSION_ENV_KEYS) | set(config.VENDORED_ENV_KEYS)


def _is_setting(name: str) -> bool:
    return name in _setting_names() or name.startswith(config.EXTENSION_ENV_PREFIXES)


def _code_lines(text: str):
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            yield line


def _template_env_names() -> dict[str, set[str]]:
    """Env var names set per template file (literal ``- name: UPPER`` entries)."""
    out: dict[str, set[str]] = {}
    for path in sorted(TEMPLATES.glob("*")):
        names: set[str] = set()
        for line in _code_lines(path.read_text()):
            match = _ENV_NAME_RE.match(line)
            if match:
                names.add(match.group(1))
        if path.name == "configmap.yaml":
            for line in _code_lines(path.read_text()):
                names.update(_QUOTED_UPPER_RE.findall(line))
        if names:
            out[path.name] = names
    return out


def _values() -> dict:
    return yaml.safe_load((CHART / "values.yaml").read_text())


def _terraform_env_names() -> set[str]:
    text = (TERRAFORM / "helm.tf").read_text()
    return set(_TF_KEY_RE.findall("\n".join(_code_lines(text))))


# ── env parity ──────────────────────────────────────────────────────


def test_every_chart_and_terraform_env_name_is_a_real_setting():
    if not _present():
        return
    bad: list[str] = []
    for fname, names in _template_env_names().items():
        bad += [f"templates/{fname}: {n}" for n in sorted(names) if not _is_setting(n)]
    bad += [f"values.yaml config: {k}" for k in _values()["config"] if not _is_setting(k.upper())]
    for entry in _values().get("extraEnv") or []:
        if not _is_setting(entry["name"].upper()):
            bad.append(f"values.yaml extraEnv: {entry['name']}")
    bad += [f"terraform/helm.tf: {n}" for n in sorted(_terraform_env_names()) if not _is_setting(n)]
    assert not bad, f"env names that are not config.Settings fields: {bad}"


def test_chart_config_is_a_small_allowlist_and_never_restates_a_default():
    if not _present():
        return
    cfg = _values()["config"]
    extra = sorted(set(cfg) - CHART_CONFIG_ALLOWLIST)
    assert not extra, (
        f"values.yaml config keys outside the documented allowlist: {extra}; "
        "leave them out so config.py's default applies"
    )
    fields = {name.upper(): f for name, f in config.Settings.model_fields.items()}
    restated: list[str] = []
    for key, value in cfg.items():
        field = fields.get(key.upper())
        if field is None or field.default is PydanticUndefined or field.default is None:
            continue
        if str(field.default).lower() == str(value).lower():
            restated.append(f"{key}={value!r}")
    assert not restated, f"values.yaml restates config.py defaults (remove them): {restated}"


def test_configmap_derives_docling_and_ingest_switches_from_values():
    if not _present():
        return
    text = (TEMPLATES / "configmap.yaml").read_text()
    for needle in ("DOCLING_ENABLED", "DOCLING_URL", "INGEST_STORAGE_ENABLED"):
        assert needle in text, f"configmap.yaml no longer derives {needle}"
    cfg = _values()["config"]
    for derived in ("DOCLING_ENABLED", "DOCLING_URL", "INGEST_STORAGE_ENABLED"):
        assert derived not in cfg, f"{derived} is derived in the ConfigMap; do not also default it in values.yaml"


def test_secret_keys_line_up_between_terraform_externalsecret_and_chart():
    if not _present():
        return
    es = yaml.safe_load((DEPLOY / "k8s" / "externalsecret.yaml").read_text())
    data = es["spec"]["data"]
    remote_keys = {d["remoteRef"]["key"] for d in data}
    secret_keys = {d["secretKey"] for d in data}
    ids_block = re.search(r"neuralscape_secret_ids\s*=\s*\[(.*?)\]", (TERRAFORM / "secrets.tf").read_text(), re.DOTALL)
    assert ids_block, "secrets.tf neuralscape_secret_ids list missing"
    tf_ids = set(re.findall(r'"([a-z0-9-]+)"', ids_block.group(1)))
    assert tf_ids == remote_keys, f"terraform secret ids {tf_ids ^ remote_keys} differ from ExternalSecret remoteRefs"
    helper = (TEMPLATES / "_helpers.tpl").read_text()
    used = set(re.findall(r"key:\s*([a-z0-9-]+)\s*\}", helper))
    assert used <= secret_keys, f"chart reads Secret keys the ExternalSecret never syncs: {used - secret_keys}"


# ── image tags ──────────────────────────────────────────────────────


def test_no_floating_latest_image_tag():
    if not _present():
        return
    values = _values()
    assert values["image"]["tag"] != "latest"
    docling_tag = str(values["docling"]["image"]["tag"])
    assert docling_tag and docling_tag != "latest", "docling image must be pinned"
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text())
    assert str(chart["appVersion"]) not in ("", "latest"), "appVersion is the fallback image tag; it must be pinned"
    helper = (TEMPLATES / "_helpers.tpl").read_text()
    assert 'eq $tag "latest"' in helper and "fail" in helper, "neuralscape.image must reject an empty or latest tag"
    variables = (TERRAFORM / "variables.tf").read_text()
    block = re.search(r'variable "image_tag" \{(.*?)\n\}', variables, re.DOTALL)
    assert block, "terraform image_tag variable missing"
    assert not re.search(r"^\s*default\s*=", block.group(1), re.MULTILINE), "terraform image_tag must be required (no default, never latest)"
    for path in sorted(TEMPLATES.glob("*.yaml")):
        for line in _code_lines(path.read_text()):
            assert not re.search(r"image:.*:latest\b", line), f"{path.name}: {line.strip()}"
    for path in sorted((DEPLOY / "k8s").glob("*.yaml")):
        for line in _code_lines(path.read_text()):
            assert not re.search(r"image:.*:latest\b", line), f"k8s/{path.name}: {line.strip()}"


# ── removed settings / stale text ───────────────────────────────────


def test_deploy_tree_does_not_reference_removed_settings():
    if not _present():
        return
    hits: list[str] = []
    for path in sorted(DEPLOY.rglob("*")):
        if not path.is_file() or ".terraform" in path.parts:
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if REMOVED_SETTINGS_PATTERN.search(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not hits, f"references to the deleted wiki synthesizer: {hits}"


def test_readme_lists_every_shipped_workload():
    if not _present():
        return
    readme = (DEPLOY / "README.md").read_text()
    assert "three entrypoints" not in readme
    for name in ("neuralscape-api", "neuralscape-worker", "neuralscape-graph-worker", "neuralscape-ingest-worker", "docling"):
        assert name in readme, f"deploy/README.md does not mention {name}"


# ── workloads and probes ────────────────────────────────────────────


def test_each_app_deployment_exists_with_a_real_entrypoint():
    if not _present():
        return
    worker_classes = set(re.findall(r"^class (\w+)", (SERVICE_DIR / "worker.py").read_text(), re.MULTILINE))
    for fname, (deployment, settings_class) in APP_DEPLOYMENTS.items():
        path = TEMPLATES / fname
        assert path.is_file(), f"missing template {fname}"
        text = path.read_text()
        assert f"name: {deployment}\n" in text, f"{fname} does not define Deployment {deployment}"
        if settings_class is not None:
            assert settings_class in worker_classes, f"worker.py has no {settings_class}"
            assert f'"worker.{settings_class}"' in text, f"{fname} does not run worker.{settings_class}"
    assert (TEMPLATES / "docling-deployment.yaml").is_file()


def test_api_probes_hit_the_services_real_endpoints():
    if not _present():
        return
    main_text = (SERVICE_DIR / "main.py").read_text()
    for route in ("/health", "/health/live"):
        assert f'@app.get("{route}")' in main_text, f"main.py has no {route} route"
    api = (TEMPLATES / "api-deployment.yaml").read_text()
    readiness = re.search(r"readinessProbe:\s*\n\s*httpGet:\s*\{\s*path:\s*(\S+),", api)
    liveness = re.search(r"livenessProbe:\s*\n\s*httpGet:\s*\{\s*path:\s*(\S+),", api)
    assert readiness and readiness.group(1) == "/health"
    assert liveness and liveness.group(1) == "/health/live"
    assert str(_values()["api"]["containerPort"]) == "8199"
    assert "8199" in (SERVICE_DIR / "Dockerfile").read_text(), "api.containerPort no longer matches the image's port"
