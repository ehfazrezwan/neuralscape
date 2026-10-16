"""Processing policy and body contracts: no caller code runs before the input is captured.

Caller objects (proxies that report a model class, string subclasses, classes only registered
as a base) must be classified by their real type, never through ``__class__``, hashing or an
ABC registry, and unsupported objects must be refused before Pydantic sees them."""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel, ValidationError

from contracts_bodies import OpaqueEnvelopeBody, PlaintextBody, parse_memory_body
from contracts_processing import PlaintextDispatchRejected, ProcessingPolicy, validate_plaintext_dispatch

BASE = dict(
    schema_version="candidate-v1", mode="operator_trusted", allowed_execution_locations=("external_provider",),
    approved_recipient_ids=("r-ok",), fallback_policy="deny", policy_epoch=3,
)


def make_policy(**over) -> ProcessingPolicy:
    return ProcessingPolicy.model_validate({**BASE, **over})


def dispatch(policy=None, **over):
    arguments = dict(
        policy=policy if policy is not None else make_policy(), execution_location="external_provider",
        recipient_id="r-ok", current_policy_epoch=3, currently_authorized_recipient_ids={"r-ok"}, is_fallback=False,
    )
    arguments.update(over)
    return validate_plaintext_dispatch(**arguments)


class Spy(str):
    """A string subclass whose hash and equality record a call (and lie about equality)."""

    events: list[str] = []

    def __hash__(self):
        Spy.events.append("hash")
        return str.__hash__(self)

    def __eq__(self, other):
        Spy.events.append("eq")
        return True

    def __ne__(self, other):
        Spy.events.append("ne")
        return False


def test_a_string_subclass_location_is_normalized_without_running_its_hooks() -> None:
    Spy.events.clear()
    dispatch(execution_location=Spy("external_provider"))  # a declared value is still accepted
    assert Spy.events == []
    with pytest.raises(PlaintextDispatchRejected):
        dispatch(execution_location=Spy("operator_worker"))  # equal to everything, but not declared
    assert Spy.events == []


def test_string_subclass_recipients_and_authority_ids_run_no_hooks() -> None:
    Spy.events.clear()
    with pytest.raises(PlaintextDispatchRejected):
        dispatch(recipient_id=Spy("r-evil"))
    dispatch(recipient_id=Spy("r-ok"), currently_authorized_recipient_ids=[Spy("r-ok")])
    with pytest.raises(PlaintextDispatchRejected):
        dispatch(recipient_id="r-ok", currently_authorized_recipient_ids=[Spy("r-other")])
    assert Spy.events == []


class PolicyProxy:
    """Reports ``ProcessingPolicy`` through ``__class__`` and delegates attributes to a target."""

    def __init__(self, target, events):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_events", events)

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        object.__getattribute__(self, "_events").append("__class__")
        return ProcessingPolicy

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)


@pytest.mark.parametrize("valid", [True, False], ids=["valid", "unchecked-invalid"])
def test_a_delegating_proxy_policy_is_refused_without_running_its_class_callback(valid) -> None:
    events: list[str] = []
    target = make_policy() if valid else make_policy().model_copy(update={"mode": "strict_local"})
    with pytest.raises(PlaintextDispatchRejected):
        dispatch(policy=PolicyProxy(target, events))
    assert events == []


class BodyProxy:
    def __init__(self, target, events):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_events", events)

    @property
    def __class__(self):  # noqa: D401 - the callback under test
        object.__getattribute__(self, "_events").append("__class__")
        return PlaintextBody

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)


@pytest.mark.parametrize("text", ["fine", ""], ids=["valid", "unchecked-invalid"])
def test_a_delegating_proxy_body_is_refused_without_running_its_class_callback(text) -> None:
    events: list[str] = []
    target = PlaintextBody(kind="plaintext", text="x").model_copy(update={"text": text})
    with pytest.raises(ValidationError):
        parse_memory_body(BodyProxy(target, events))
    assert events == []


def test_a_class_only_registered_as_a_model_base_is_refused_cleanly() -> None:
    class Fake:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        BaseModel.register(Fake)
    with pytest.raises(PlaintextDispatchRejected):
        dispatch(policy=Fake())
    with pytest.raises(ValidationError):
        parse_memory_body(Fake())


def test_genuine_inputs_still_pass_everywhere() -> None:
    dispatch()
    dispatch(policy=dict(BASE))  # a plain dict is validated like the model
    assert parse_memory_body({"kind": "plaintext", "text": "x"}) == PlaintextBody(kind="plaintext", text="x")
    assert parse_memory_body(OpaqueEnvelopeBody(kind="opaque_envelope", envelope_id="env-1")).kind == "opaque_envelope"
    assert parse_memory_body(PlaintextBody(kind="plaintext", text="x").model_copy()).text == "x"
