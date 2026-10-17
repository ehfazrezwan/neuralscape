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


_NEEDED: list[str] = []
_TARGET: Path | None = None
_HARNESS: Path | None = None


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _foreign_modules() -> list[str]:
    """Modules loaded from the harness's own tree while observing another tree (never allowed)."""

    if _TARGET is None or _HARNESS is None or _HARNESS.resolve() == _TARGET.resolve():
        return []
    found = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or name in ("__main__", "persona_probes"):
            continue
        path = Path(file)
        parts = set(path.parts)
        if ".venv" in parts or "site-packages" in parts:
            continue
        if _under(path, _HARNESS) and not _under(path, _TARGET):
            found.append(name)
    return sorted(found)


def _sources() -> dict[str, str]:
    out = {}
    for name in _NEEDED:
        file = getattr(sys.modules.get(name), "__file__", None)
        if file and _TARGET is not None:
            out[name] = str(Path(file).resolve().relative_to(_TARGET.resolve())) if _under(Path(file), _TARGET) else "OUTSIDE-TARGET"
    return out


def _need(name: str):
    import importlib

    _NEEDED.append(name)
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


def _is_missing_field(error: Exception, field: str) -> bool:
    """True for a validation error whose only complaint is that ``field`` was not supplied."""

    errors = getattr(error, "errors", None)
    if not callable(errors):
        return False
    try:
        details = errors()
        return bool(details) and all(item.get("type") == "missing" and tuple(item.get("loc", ())) == (field,) for item in details)
    except Exception:  # noqa: BLE001 - an unreadable error is not a recognised refusal
        return False


def citation_resolution() -> dict[str, str]:
    """Cam follows a citation: a handle names its tenant, and a foreign handle can be told apart."""

    references = _need("contracts_references")
    handle = references.ReferenceHandle
    # A refusal only means something next to a success: a tenant-scoped handle must be accepted first.
    # A failure here (or any failure that is not the missing tenant below) propagates and is recorded
    # as an invalid attempt, never as a refusal.
    handle(kind="memory", id="memory-2", tenant_id="tenant-1", resolver="resolve_memory")
    try:
        handle(kind="memory", id="memory-1", resolver="resolve_memory")
        without_tenant = "accepted"
    except Exception as error:  # noqa: BLE001 - narrowed below: only a missing-tenant error is a refusal
        if not _is_missing_field(error, "tenant_id"):
            raise
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
        del _NEEDED[:]
        try:
            result = {"persona": persona, "facts": probe(), "sources": _sources()}
            foreign = _foreign_modules()
            if foreign or "OUTSIDE-TARGET" in result["sources"].values():
                result = {"persona": persona, "foreign": foreign or ["OUTSIDE-TARGET"]}
            out[scenario_id] = result
        except CapabilityAbsent as error:
            foreign = _foreign_modules()
            out[scenario_id] = {"persona": persona, "foreign": foreign} if foreign else {"persona": persona, "absent": str(error)}
        except Exception as error:  # noqa: BLE001 - recorded as an invalid attempt by the caller
            out[scenario_id] = {"persona": persona, "error": type(error).__name__}  # the class only: message text can hold content
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--harness-dir", default=None)
    args = parser.parse_args()
    global _TARGET, _HARNESS
    _TARGET = Path(args.service_dir).resolve()
    _HARNESS = Path(args.harness_dir).resolve() if args.harness_dir else Path(__file__).resolve().parent
    # The target tree is the only tree on the path: drop the harness's directory (and any editable-install
    # entry for it) so a module the target lacks cannot be found there.
    if _HARNESS != _TARGET:
        sys.path[:] = [entry for entry in sys.path if not (entry and _under(Path(entry), _HARNESS) and ".venv" not in Path(entry).parts and "site-packages" not in Path(entry).parts)]
    sys.path.insert(0, str(_TARGET))
    print(json.dumps(observe_all(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
