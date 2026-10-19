"""Context and freshness rebuild stored contract graphs with the shared snapshot, not a private copy.

Three storage shapes that an older, duplicated rebuild routine accepted while the native
view rejects them.  Each is checked through the context helper and the freshness helper."""

from __future__ import annotations

import pytest

import contracts_context
import contracts_freshness
from contracts_common import snapshot_contract_graph
from contracts_references import SourceVersion

BASE = {"record_id": "rec-1", "content_revision": 1, "policy_epoch": 1}


def make() -> SourceVersion:
    return SourceVersion(**BASE)


def native(model) -> dict:
    from pydantic import BaseModel

    return BaseModel.__dict__["__dict__"].__get__(model, BaseModel)


class HidingDict(dict):
    """A ``__dict__`` replacement whose public ``items()`` omits an injected entry."""

    def items(self):  # noqa: D102
        return [(k, v) for k, v in dict.items(self) if k != "injected"]


def shape_hidden_injected_entry() -> SourceVersion:
    value = make()
    stored = HidingDict(native(value))
    dict.__setitem__(stored, "injected", "smuggled")
    object.__setattr__(value, "__dict__", stored)
    return value


class DisagreeingExtras(dict):
    """Extra storage whose ``keys()`` is harmless while ``items()`` carries a declared field."""

    def keys(self):  # noqa: D102
        return ["harmless"]

    def items(self):  # noqa: D102
        return [("content_revision", 7)]


def shape_declared_field_in_extras() -> SourceVersion:
    value = make()
    dict.__delitem__(native(value), "content_revision")
    object.__setattr__(value, "__pydantic_extra__", DisagreeingExtras())
    return value


def shape_decoy_dict_descriptor() -> SourceVersion:
    from pydantic import BaseModel

    descriptor = BaseModel.__dict__["__dict__"]

    class Decoy(SourceVersion):
        # A data descriptor: construction still writes the real storage, but a public read
        # of ``__dict__`` returns a decoy that looks valid.
        __dict__ = property(lambda self: dict(BASE), lambda self, value: descriptor.__set__(self, value))  # type: ignore[assignment]

    real = Decoy(**BASE)
    dict.__setitem__(descriptor.__get__(real, BaseModel), "content_revision", -1)  # the real state is invalid
    return real


SHAPES = [shape_hidden_injected_entry, shape_declared_field_in_extras, shape_decoy_dict_descriptor]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda f: f.__name__)
def test_the_shared_snapshot_rejects_the_shape(shape) -> None:
    value = shape()
    with pytest.raises(Exception):
        SourceVersion.model_validate(snapshot_contract_graph(value), strict=True)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda f: f.__name__)
def test_the_context_helper_rejects_what_the_native_view_rejects(shape) -> None:
    assert contracts_context._revalidated_snapshot(SourceVersion, shape()) is None


@pytest.mark.parametrize("shape", SHAPES, ids=lambda f: f.__name__)
def test_the_freshness_helper_rejects_what_the_native_view_rejects(shape) -> None:
    with pytest.raises(Exception):
        contracts_freshness.projection_status_for((shape(),), (make(),))


def test_an_honest_model_still_round_trips_through_both_helpers() -> None:
    honest = make()
    assert contracts_context._revalidated_snapshot(SourceVersion, honest) == honest
    assert contracts_freshness.projection_status_for((honest,), (make(),)) is contracts_freshness.ProjectionStatus.CURRENT


def test_neither_module_keeps_a_private_graph_rebuild_routine() -> None:
    assert not hasattr(contracts_context, "_native_contract_graph")
    assert not hasattr(contracts_freshness, "_native_contract_graph")


# ── an object that only reports a model class never reaches Pydantic ──────────


class _Proxy:
    """Reports ``SourceVersion`` through ``__class__`` and delegates attributes to a real one."""

    def __init__(self, target: object, events: list[str]) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_events", events)

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        object.__getattribute__(self, "_events").append("__class__")
        return SourceVersion

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_target"), name)


@pytest.mark.parametrize("revision", [1, -5], ids=["valid", "invalid"])
def test_a_delegating_proxy_is_refused_without_running_its_class_callback(revision) -> None:
    events: list[str] = []
    target = make().model_copy(update={"content_revision": revision})
    assert contracts_context._revalidated_snapshot(SourceVersion, _Proxy(target, events)) is None
    with pytest.raises(Exception):
        contracts_freshness.projection_status_for((_Proxy(target, events),), (make(),))
    assert events == []


@pytest.mark.parametrize("leaf", [object(), {1, 2}, 1 + 2j, type, len])
def test_an_unsupported_leaf_is_a_stable_error_before_validation(leaf) -> None:
    with pytest.raises(ValueError, match="unsupported value type"):
        contracts_freshness.require_native_snapshot({"a": [leaf]})


def test_genuine_scalars_enums_datetimes_and_containers_pass_the_native_check() -> None:
    from datetime import datetime, timezone

    snapshot = {"a": (1, 2.5, "x", None, True, datetime(2026, 1, 1, tzinfo=timezone.utc), contracts_freshness.ProjectionStatus.CURRENT), "b": [[]]}
    assert contracts_freshness.require_native_snapshot(snapshot) is snapshot


def test_a_shared_cycle_free_structure_is_walked_once() -> None:
    shared: tuple = (1,)
    for _ in range(60):
        shared = (shared, shared)  # 2**60 paths over 61 tuples
    contracts_freshness.require_native_snapshot(shared)  # returns at once: identities are visited once


# ── the shared snapshot itself: type checks by native MRO, bounded work ──────


def _hooked(base_meta, events: list[str]):
    class Meta(base_meta):
        def __hash__(cls):
            events.append("hash")
            return id(cls)

        def __eq__(cls, other):
            events.append("eq")
            return False

        def __ne__(cls, other):
            events.append("ne")
            return True

        def __subclasscheck__(cls, subclass):
            events.append("subclasscheck")
            return False

        def __instancecheck__(cls, instance):
            events.append("instancecheck")
            return False

    return Meta


@pytest.mark.parametrize("base", [tuple, list, dict])
def test_the_shared_snapshot_runs_no_metaclass_hook_on_a_hostile_container_class(base) -> None:
    events: list[str] = []
    hostile = _hooked(type, events)("Hostile", (base,), {})
    value = hostile()
    events.clear()  # building the class is not under test
    try:
        snapshot_contract_graph(value)
    except Exception:  # an unsupported subclass may be refused; running its hooks is what is not allowed
        pass
    assert events == []


def test_ordinary_graphs_are_far_inside_the_budget() -> None:
    snapshot = snapshot_contract_graph([make() for _ in range(500)])
    assert len(snapshot) == 500


# ── an enum member is a leaf only when its class is one of the contracts' own enums ──

from enum import Enum  # noqa: E402

import contracts_context as _context  # noqa: E402
from contracts_references import ReferenceHandle  # noqa: E402


class _EnumProxyBase:
    """Reports ``ReferenceHandle`` through ``__class__`` and delegates attributes to a real handle."""

    events: list = []

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        _EnumProxyBase.events.append("__class__")
        return ReferenceHandle

    def __getattr__(self, name: str):
        return getattr(ReferenceHandle(kind="source", id="s-1", tenant_id="t-1", resolver="ns:resolver"), name)


class _ForeignEnum(_EnumProxyBase, Enum):
    MEMBER = 1


def test_a_member_of_a_foreign_enum_is_refused_before_validation_and_its_class_is_never_read() -> None:
    _EnumProxyBase.events = []
    member = _ForeignEnum.MEMBER
    with pytest.raises(ValueError, match="unsupported value type"):
        contracts_freshness.require_native_snapshot({"source": member})
    assert _EnumProxyBase.events == []


def test_a_foreign_enum_member_standing_in_for_a_model_is_refused_by_the_context_helper() -> None:
    from contracts_freshness import SourceCheckpoint

    _EnumProxyBase.events = []
    value = {
        "schema_version": "candidate-v1", "source": _ForeignEnum.MEMBER, "verification_status": "unverified",
        "verification_method": "observation_only", "observed_at": "2026-09-28T12:00:00Z",
        "last_successful_verification_at": None, "provider_revision": None, "opaque_cursor": None,
        "processed_through": None, "reconciliation_state": "complete", "freshness_policy": "p",
    }
    assert _context._revalidated_snapshot(SourceCheckpoint, value) is None
    assert _EnumProxyBase.events == []


def test_a_same_named_foreign_enum_is_refused() -> None:
    class ProjectionStatus(str, Enum):  # same name and values as the contract's own enum
        CURRENT = "current"

    with pytest.raises(ValueError, match="unsupported value type"):
        contracts_freshness.require_native_snapshot([ProjectionStatus.CURRENT])


def test_every_enum_the_contracts_define_is_a_supported_leaf() -> None:
    members = []
    for module in (contracts_freshness, contracts_context):
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, Enum) and value is not Enum and value.__module__ == module.__name__:
                members.extend(value)
    assert len(members) > 20
    contracts_freshness.require_native_snapshot(members)
