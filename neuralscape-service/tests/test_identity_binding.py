"""Binding resolver, namespace codec and legacy adjudication (no ledger)."""

from __future__ import annotations

import itertools
import random
import threading

import pytest

from contracts_policy import DelegationConstraints, PrincipalContext
from contracts_references import ReferenceHandle
from identity import slugify_email
from identity_binding import (
    InMemoryBindingStore,
    LegacyObservation,
    NamespaceKey,
    bind_identity,
    classify_legacy_identities,
    decode_namespace,
    encode_namespace,
    new_subject_id,
    resolve_binding,
    validate_service_account_controller,
)
from memory.groups import _build_group_id


# ── binding ──────────────────────────────────────────────────────────────


def test_binding_is_verbatim_and_stable() -> None:
    store = InMemoryBindingStore()
    first = bind_identity(store, "https://issuer.example", "sub-1")
    again = bind_identity(store, "https://issuer.example", "sub-1")

    assert first.created and not again.created
    assert first.subject_id == again.subject_id
    assert first.subject_id.startswith("sub_")
    assert resolve_binding(store, "https://issuer.example", "sub-1") == first.subject_id


@pytest.mark.parametrize(
    "left,right",
    [
        (("i", "Sub"), ("i", "sub")),
        (("i", "sub"), ("i", "sub ")),
        (("I", "sub"), ("i", "sub")),
        (("a", "b:c"), ("a:b", "c")),
        (("a", "b c"), ("a", "b\u00a0c")),
        (("i", "\u00e9"), ("i", "e\u0301")),
    ],
)
def test_distinct_pairs_never_merge(left, right) -> None:
    store = InMemoryBindingStore()
    assert bind_identity(store, *left).subject_id != bind_identity(store, *right).subject_id


def test_unbound_pair_resolves_to_none_without_creating() -> None:
    store = InMemoryBindingStore()
    assert resolve_binding(store, "i", "nobody") is None
    assert store.get("i", "nobody") is None


def test_subject_id_is_not_derived_from_identity_data() -> None:
    first = bind_identity(InMemoryBindingStore(), "i", "same").subject_id
    second = bind_identity(InMemoryBindingStore(), "i", "same").subject_id
    assert first != second
    assert len({new_subject_id() for _ in range(100)}) == 100


@pytest.mark.parametrize("bad", ["", None, 1, b"x", "x" * 257, "\ud800"])
def test_invalid_components_are_rejected(bad) -> None:
    store = InMemoryBindingStore()
    with pytest.raises(ValueError):
        bind_identity(store, bad, "sub")
    with pytest.raises(ValueError):
        bind_identity(store, "issuer", bad)
    with pytest.raises(ValueError):
        resolve_binding(store, bad, "sub")


def test_concurrent_first_binding_converges_on_one_subject() -> None:
    store = InMemoryBindingStore()
    results: list[str] = []
    barrier = threading.Barrier(16)

    def worker() -> None:
        barrier.wait()
        results.append(bind_identity(store, "i", "racer").subject_id)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(results)) == 1
    assert store.get("i", "racer") == results[0]


def test_losing_creator_reports_not_created() -> None:
    class Loser(InMemoryBindingStore):
        def get(self, issuer, provider_subject):  # simulate the race window
            return None

    store = Loser()
    store.put_if_absent("i", "s", "sub_winner")
    result = bind_identity(store, "i", "s", generate_id=lambda: "sub_loser")
    assert result == ("sub_winner", False)


# ── legacy functions collide; the codec does not ─────────────────────────


def test_legacy_email_slug_collisions_exist() -> None:
    assert slugify_email("a@b-c.com") == slugify_email("a-b@c.com")
    assert slugify_email("a+b@x.com") == slugify_email("a-b@x.com")


def test_legacy_group_id_delimiter_collision_exists() -> None:
    assert _build_group_id("private", "x--project--y") == _build_group_id(
        "private", "x", project_id="y"
    )


def test_codec_separates_the_legacy_group_id_collision() -> None:
    left = encode_namespace(NamespaceKey("t", "private", "x--project--y", None, None))
    right = encode_namespace(NamespaceKey("t", "private", "x", "y", None))
    assert left != right
    assert decode_namespace(left).owner == "x--project--y"
    assert decode_namespace(right) == ("t", "private", "x", "y", None)


def test_bindings_keep_slug_colliding_logins_apart() -> None:
    store = InMemoryBindingStore()
    one = bind_identity(store, "idp", "a@b-c.com").subject_id
    two = bind_identity(store, "idp", "a-b@c.com").subject_id
    assert one != two


# ── codec ────────────────────────────────────────────────────────────────

_ATOMS = ["a", "b", "-", "--", "--project--", ":", "1", "12:", "~", "ns1", "\u00e9", "e\u0301", " "]


def _components(limit: int) -> list[str]:
    values = set(_ATOMS)
    for size in (2, 3):
        values.update("".join(p) for p in itertools.product(_ATOMS[:6], repeat=size))
    return sorted(values)[:limit]


def _keys(limit: int = 60):
    comps = _components(limit)
    optional = [None, *comps[:6]]
    for tenant in comps[:8]:
        for owner in comps[:10]:
            for project in optional:
                for workspace in optional[:4]:
                    yield NamespaceKey(tenant, "private", owner, project, workspace)
        for kind in ("shared", "standard"):
            for project in optional:
                for workspace in optional[:4]:
                    yield NamespaceKey(tenant, kind, None, project, workspace)


def test_round_trip_and_injectivity_over_adversarial_tuples() -> None:
    seen: dict[str, NamespaceKey] = {}
    for key in _keys():
        encoded = encode_namespace(key)
        assert decode_namespace(encoded) == key
        assert seen.setdefault(encoded, key) == key
    assert len(seen) > 1000


def test_round_trip_random_unicode() -> None:
    rng = random.Random(20261016)
    alphabet = "ab-_.:~0123456789\u00e9e\u0301\u4e2d\U0001f600 \t"
    for _ in range(2000):
        def pick() -> str:
            return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))

        kind = rng.choice(["private", "shared", "standard"])
        key = NamespaceKey(
            pick(),
            kind,
            pick() if kind == "private" else None,
            pick() if rng.random() < 0.5 else None,
            pick() if rng.random() < 0.5 else None,
        )
        assert decode_namespace(encode_namespace(key)) == key


def test_codec_does_not_normalize_or_fold() -> None:
    nfc = encode_namespace(NamespaceKey("t", "shared", None, "\u00e9", None))
    nfd = encode_namespace(NamespaceKey("t", "shared", None, "e\u0301", None))
    upper = encode_namespace(NamespaceKey("T", "shared", None, None, None))
    lower = encode_namespace(NamespaceKey("t", "shared", None, None, None))
    assert len({nfc, nfd, upper, lower}) == 4


def test_encoding_is_versioned_and_length_prefixed() -> None:
    assert encode_namespace(NamespaceKey("t", "private", "o", None, None)) == "ns11:t7:private1:o~~"
    assert encode_namespace(NamespaceKey("\u00e9", "shared", None, None, None)).startswith("ns12:\u00e9")


@pytest.mark.parametrize(
    "key",
    [
        NamespaceKey("", "shared", None, None, None),
        NamespaceKey("t", "private", None, None, None),
        NamespaceKey("t", "shared", "o", None, None),
        NamespaceKey("t", "other", None, None, None),
        NamespaceKey("t", "shared", None, "", None),
        NamespaceKey("t" * 257, "shared", None, None, None),
    ],
)
def test_invalid_keys_are_not_encodable(key) -> None:
    with pytest.raises(ValueError):
        encode_namespace(key)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "ns21:t6:shared~~~",  # unknown version
        "ns11:t6:shared~~",  # truncated
        "ns11:t6:shared~~~x",  # trailing data
        "ns101:t6:shared~~~",  # leading-zero length
        "ns11:t7:private~~~",  # private pool without owner
        "ns11:t6:shared1:o~~",  # shared pool with owner
        "ns19:t6:shared~~~",  # length past end
        "ns10::6:shared~~~",  # zero length
        "ns1+1:t6:shared~~~",  # signed length
    ],
)
def test_malformed_or_non_canonical_text_is_rejected(text) -> None:
    with pytest.raises(ValueError):
        decode_namespace(text)


def test_decode_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        decode_namespace(None)  # type: ignore[arg-type]


# ── service accounts ─────────────────────────────────────────────────────


def _principal(kind: str, subject: str = "subject-a", **extra) -> PrincipalContext:
    return PrincipalContext(
        schema_version="candidate-v1",
        tenant_id="tenant-a",
        subject_id=subject,
        subject_kind=kind,
        credential_id="credential-a",
        membership_version=1,
        device_id=None,
        delegation=extra.get("delegation"),
    )


def test_only_a_human_may_control_a_service_account() -> None:
    validate_service_account_controller(
        _principal("human"), service_account_subject_id="sa-1"
    )
    for kind in ("service_account", "worker"):
        with pytest.raises(ValueError):
            validate_service_account_controller(
                _principal(kind), service_account_subject_id="sa-1"
            )
    ceiling = DelegationConstraints(
        delegated_by_subject_id="subject-h",
        delegated_by_credential_id="c",
        allowed_actions=("read",),
        allowed_resources=(),
    )
    with pytest.raises(ValueError):
        validate_service_account_controller(
            _principal("delegated_agent", delegation=ceiling),
            service_account_subject_id="sa-1",
        )


def test_service_account_cannot_control_itself() -> None:
    with pytest.raises(ValueError):
        validate_service_account_controller(
            _principal("human", "same"), service_account_subject_id="same"
        )


# ── legacy adjudication ──────────────────────────────────────────────────


def _obs(legacy, issuer=None, subject=None, email=None) -> LegacyObservation:
    return LegacyObservation(legacy, issuer, subject, email)


def test_unique_ambiguous_and_unbound_are_separated() -> None:
    result = classify_legacy_identities(
        [
            _obs("alice", "idp", "s-alice", "alice@example.com"),
            _obs("alice", "idp", "s-alice", "alice@example.com"),
            _obs("bob", "idp", "s-bob1"),
            _obs("bob", "idp", "s-bob2"),
            _obs("carol"),
            _obs("dave", "idp", "s-shared"),
            _obs("erin", "idp", "s-shared"),
            _obs("x--project--y", "idp", "s-x"),
        ]
    )

    assert result["alice"].status == "unique"
    assert result["alice"].identities == (("idp", "s-alice"),)
    assert result["bob"] == ("ambiguous", ("multiple_identities",), (("idp", "s-bob1"), ("idp", "s-bob2")))
    assert result["carol"].status == "unbound"
    assert result["dave"].reasons == result["erin"].reasons == ("identity_under_multiple_legacy_ids",)
    assert result["x--project--y"].reasons == ("namespace_delimiter",)


def test_slug_colliding_emails_are_quarantined_not_merged() -> None:
    result = classify_legacy_identities(
        [
            _obs("a-b-c.com", "idp", "s1", "a@b-c.com"),
            _obs("a-b-c.com", "idp", "s2", "a-b@c.com"),
        ]
    )
    reasons = result["a-b-c.com"].reasons
    assert result["a-b-c.com"].status == "ambiguous"
    assert "email_slug_collision" in reasons and "multiple_identities" in reasons


def test_classification_is_pure_and_order_independent() -> None:
    observations = [
        _obs("u1", "i", "s1", "u1@x.com"),
        _obs("u2", "i", "s1", "u2@x.com"),
        _obs("u3"),
    ]
    forward = classify_legacy_identities(observations)
    backward = classify_legacy_identities(list(reversed(observations)))
    assert forward == backward
    assert classify_legacy_identities([]) == {}
