"""Static guards for ``.github/workflows/deploy.yml`` (audit gap ci-deploy-no-tests).

No network, no Docker: PyYAML parses the workflow and the assertions pin the
properties that must not regress:

* the SSH host key is pinned from a secret (no trust-on-first-use),
* GHCR only ever receives the immutable ``sha-<12>`` tag, never ``:latest``,
* the deploy runs only on a push to ``main`` (plus manual dispatch),
* the remote health-polling loops still exist.

In the packaged test image only the service tree exists (no ``.github``), so each
test returns early (no skipif) when the repo-level file is absent; the host CI job runs them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

KNOWN_HOSTS_SECRET = "VPS_SSH_KNOWN_HOSTS"


def _have_repo() -> bool:
    return WORKFLOW.is_file()


@pytest.fixture(scope="module")
def raw() -> str:
    if not _have_repo():
        return ""
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def workflow() -> dict:
    if not _have_repo():
        return {}
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def steps(workflow) -> list[dict]:
    if not workflow:
        return []
    return workflow["jobs"]["deploy"]["steps"]


def _step(steps: list[dict], needle: str) -> dict:
    for s in steps:
        if needle in (s.get("name") or ""):
            return s
    raise AssertionError(f"no step named like {needle!r}")


def test_no_trust_on_first_use(raw):
    if not _have_repo():
        return
    assert "accept-new" not in raw
    assert "StrictHostKeyChecking=no" not in raw


def test_ssh_pins_host_key_from_known_hosts_file(steps):
    if not _have_repo():
        return
    ship = _step(steps, "Ship image")["run"]
    assert "StrictHostKeyChecking=yes" in ship
    assert "UserKnownHostsFile=" in ship


def test_known_hosts_secret_is_used_and_fails_early_when_empty(raw, steps):
    if not _have_repo():
        return
    pin = _step(steps, "Pin the VPS host key")
    assert f"secrets.{KNOWN_HOSTS_SECRET}" in pin["env"]["KNOWN_HOSTS"]
    run = pin["run"]
    assert "::error::" in run and "exit 1" in run  # clear message, hard stop
    # The pin step runs before anything is built, pushed or shipped.
    names = [s.get("name") or s.get("uses") or "" for s in steps]
    assert names.index(pin["name"]) < names.index(_step(steps, "Build runtime image")["name"])
    # The file written by the pin step is the one ssh reads.
    path = "known_hosts_deploy"
    assert path in run and path in _step(steps, "Ship image")["run"]


def test_ghcr_gets_only_the_immutable_tag(raw, steps):
    if not _have_repo():
        return
    push = _step(steps, "Push to GHCR")["run"]
    pushes = [ln.strip() for ln in push.splitlines() if "docker push" in ln]
    assert pushes, "expected a docker push"
    for ln in pushes:
        assert ":latest" not in ln
        assert "steps.meta.outputs.tag" in ln
    assert "docker tag" not in push  # no :latest retag before the push
    # No remote `docker push` anywhere in the file targets :latest either.
    for ln in raw.splitlines():
        if "docker push" in ln:
            assert "latest" not in ln.lower()


def test_vps_keeps_its_local_latest_tag(steps):
    if not _have_repo():
        return
    # Intentional: docker-compose.vps.yml pins pull_policy: never and relies on a
    # locally tagged :latest for plain `docker compose up -d` recovery.
    ship = _step(steps, "Ship image")["run"]
    assert 'docker tag ${IMG} ${LATEST}' in ship


def test_deploys_only_on_push_to_main(workflow):
    if not _have_repo():
        return
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) <= {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" not in triggers and "pull_request_target" not in triggers
    assert "tags" not in triggers["push"]


def test_health_polling_loops_remain(steps):
    if not _have_repo():
        return
    ship = _step(steps, "Ship image")["run"]
    assert "/health/live" in ship and "liveness never came up" in ship
    assert "http://127.0.0.1:8199/health " in ship or "8199/health &&" in ship
    assert "deep health never went green" in ship
    assert ship.count("seq 1") >= 2
