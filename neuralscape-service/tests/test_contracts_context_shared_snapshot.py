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
