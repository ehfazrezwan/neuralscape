"""Persona-derived probes: observe facts about one source tree; judge nothing.

Run as ``python persona_probes.py --service-dir DIR``.  The probes import the *target tree's* modules
(DIR is put first on ``sys.path`` in this fresh process), so the same definitions can observe a
baseline checkout and a candidate checkout.  A probe returns short, content-free facts or reports that
a capability is absent; it never decides a verdict.  Expected results live in reviewer-owned oracle
files (``tests/fixtures/persona_oracles/``), not here.  Personas are synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


class CapabilityAbsent(Exception):
    """The tree has no module or API a scenario needs."""


def _need(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name or (error.name or "").split(".")[0] == name.split(".")[0]:
            raise CapabilityAbsent(name) from None
        raise


def lifecycle_state() -> dict[str, str]:
    """Alex saves a note, edits it, then deletes it: which intent state moves are legal."""

    lifecycle = _need("contracts_lifecycle")
    status = lifecycle.IntentStatus
    legal = lifecycle.is_legal_intent_transition
    return {
        "accepted_to_processing": str(legal(status.ACCEPTED, status.PROCESSING)).lower(),
        "applied_to_accepted": str(legal(status.APPLIED, status.ACCEPTED)).lower(),
        "failed_to_applied": str(legal(status.FAILED, status.APPLIED)).lower(),
        "cancelled_to_processing": str(legal(status.CANCELLED, status.PROCESSING)).lower(),
    }


def project_preference() -> dict[str, str]:
    """Bea states a preference and a project convention: each keeps its category's scope."""

    schemas = _need("schemas")
    scope = schemas.default_scope_for_category
    return {
        "preference_scope": str(getattr(scope("preference"), "value", scope("preference"))),
        "convention_scope": str(getattr(scope("convention"), "value", scope("convention"))),
    }


def citation_resolution() -> dict[str, str]:
    """Cam follows a citation: a handle names its tenant, and a foreign handle can be told apart."""

    references = _need("contracts_references")
    handle = references.ReferenceHandle
    try:
        handle(kind="memory", id="memory-1", resolver="resolve_memory")
        without_tenant = "accepted"
    except Exception:  # noqa: BLE001 - any validation failure is a refusal
        without_tenant = "refused"
    return {
        "handle_without_tenant": without_tenant,
        "foreign_marker_field": "present" if "cross_tenant" in handle.model_fields else "absent",
    }


def card_freshness() -> dict[str, str]:
    """Dev's card cites two sources: a projection is current only when every version matches."""

    refs = _need("contracts_references")
    freshness = _need("contracts_freshness")

    def version(record: str, revision: int):
        return refs.SourceVersion(record_id=record, content_revision=revision, policy_epoch=1)

    required = [version("s-1", 2), version("s-2", 1)]
    return {
        "matching_versions": str(getattr(freshness.projection_status_for([version("s-1", 2), version("s-2", 1)], required), "value", "?")),
        "older_revision": str(getattr(freshness.projection_status_for([version("s-1", 1), version("s-2", 1)], required), "value", "?")),
        "missing_source": str(getattr(freshness.projection_status_for([version("s-1", 2)], required), "value", "?")),
    }


def retained_source_hold() -> dict[str, str]:
    """Eli's note is under a legal hold: erasure is refused while it holds and allowed after release."""

    _need("ledger")
    sqlite_ledger = _need("ledger.sqlite")
    port = _need("ledger.port")
    with tempfile.TemporaryDirectory() as root:
        ledger = sqlite_ledger.SqliteLedger(str(Path(root) / "ledger.db"))
        try:
            ledger.put_object_ref(port.ObjectRef("tenant-p", "record-p", 1, 1, (), "sha256:p"))
            ledger.place_hold("tenant-p", "hold-p", "record-p", "legal", "holder-p", "cred-p")
            try:
                ledger.tombstone("tenant-p", "record-p", 1, "user_delete")
                while_held = "allowed"
            except port.HoldActive:
                while_held = "refused"
            ledger.release_hold("tenant-p", "hold-p", "releaser-p", "review_done")
            try:
                ledger.tombstone("tenant-p", "record-p", 1, "user_delete")
                after_release = "allowed"
            except port.HoldActive:
                after_release = "refused"
        finally:
            ledger.close()
    return {"erase_while_held": while_held, "erase_after_release": after_release}


SCENARIOS = {
    "persona-lifecycle-state": ("persona-alex", lifecycle_state),
    "persona-project-preference": ("persona-bea", project_preference),
    "persona-citation-resolution": ("persona-cam", citation_resolution),
    "persona-card-freshness": ("persona-dev", card_freshness),
    "persona-retained-source-hold": ("persona-eli", retained_source_hold),
}


def observe_all() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for scenario_id, (persona, probe) in SCENARIOS.items():
        try:
            out[scenario_id] = {"persona": persona, "facts": probe()}
        except CapabilityAbsent as error:
            out[scenario_id] = {"persona": persona, "absent": str(error)}
        except Exception as error:  # noqa: BLE001 - recorded as an invalid attempt by the caller
            out[scenario_id] = {"persona": persona, "error": f"{type(error).__name__}: {str(error)[:120]}"}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.service_dir).resolve()))
    print(json.dumps(observe_all(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
