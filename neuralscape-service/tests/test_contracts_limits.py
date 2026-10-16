"""Tests for the shared native-discovery budget."""

import pytest

import contracts_limits
from contracts_limits import (
    DISCOVERY_BUDGET_MESSAGE,
    PORTABILITY,
    STANDARD,
    DiscoveryBudget,
    DiscoveryTier,
    is_exact_type,
)


def small(nodes=4, edges=6, depth=3):
    return DiscoveryBudget(DiscoveryTier("test", nodes, edges, depth))


def assert_stable_rejection(excinfo):
    assert type(excinfo.value) is ValueError
    assert str(excinfo.value) == "contract graph exceeds the native discovery budget"
    assert str(excinfo.value) == DISCOVERY_BUDGET_MESSAGE


class Hostile:
    """Counts every Python data-model hook the accounting must never call."""

    calls = []

    def _hit(self, name):
        Hostile.calls.append(name)

    def __iter__(self):
        self._hit("iter")
        return iter(())

    def __eq__(self, other):
        self._hit("eq")
        return True

    def __ne__(self, other):
        self._hit("ne")
        return False

    def __hash__(self):
        self._hit("hash")
        return 0

    def __len__(self):
        self._hit("len")
        return 0

    def __bool__(self):
        self._hit("bool")
        return False

    def __getattr__(self, name):
        self._hit("getattr:" + name)
        raise AttributeError(name)

    def __repr__(self):
        self._hit("repr")
        return "hostile"

    @property
    def __class__(self):  # spoofs isinstance(); type() is unaffected
        self._hit("class")
        return dict


@pytest.fixture(autouse=True)
def reset_hostile_calls():
    Hostile.calls = []
    yield
    Hostile.calls = []


def test_tier_constants_follow_the_recorded_envelope():
    assert (STANDARD.max_nodes, STANDARD.max_edges, STANDARD.max_depth) == (16_384, 131_072, 256)
    assert (PORTABILITY.max_nodes, PORTABILITY.max_edges, PORTABILITY.max_depth) == (131_072, 786_432, 256)
    assert DiscoveryBudget().tier is STANDARD
    assert DiscoveryBudget(PORTABILITY).tier is PORTABILITY


def test_error_message_is_exactly_the_stable_value_free_text():
    assert DISCOVERY_BUDGET_MESSAGE == "contract graph exceeds the native discovery budget"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3", None])
def test_tier_rejects_non_positive_or_non_integer_limits(bad):
    for field in ("max_nodes", "max_edges", "max_depth"):
        kwargs = dict(name="t", max_nodes=1, max_edges=1, max_depth=1)
        kwargs[field] = bad
        with pytest.raises(ValueError):
            DiscoveryTier(**kwargs)


def test_tier_rejects_empty_or_non_string_name_and_budget_rejects_non_tier():
    for name in ("", None, 3):
        with pytest.raises(ValueError):
            DiscoveryTier(name, 1, 1, 1)
    with pytest.raises(TypeError):
        DiscoveryBudget(object())
    with pytest.raises(TypeError):
        DiscoveryBudget((16, 16, 16))


@pytest.mark.parametrize("limit", [1, 4, 17])
def test_node_limit_minus_one_limit_and_limit_plus_one(limit):
    budget = DiscoveryBudget(DiscoveryTier("t", limit, 10**6, 10))
    objs = [object() for _ in range(limit + 1)]
    for obj in objs[: limit - 1]:
        assert budget.charge_node(obj) is True
    assert budget.nodes == limit - 1
    assert budget.charge_node(objs[limit - 1]) is True  # exactly at the limit
    assert budget.nodes == limit
    with pytest.raises(ValueError) as excinfo:  # limit + 1
        budget.charge_node(objs[limit])
    assert_stable_rejection(excinfo)


@pytest.mark.parametrize("limit", [1, 5, 40])
def test_edge_limit_minus_one_limit_and_limit_plus_one(limit):
    budget = DiscoveryBudget(DiscoveryTier("t", 10, limit, 10))
    budget.charge_edge(limit - 1)
    assert budget.edges == limit - 1
    budget.charge_edge()  # exactly at the limit
    assert budget.edges == limit
    with pytest.raises(ValueError) as excinfo:
        budget.charge_edge()
    assert_stable_rejection(excinfo)


def test_a_single_bulk_edge_charge_over_the_limit_is_rejected_atomically():
    budget = DiscoveryBudget(DiscoveryTier("t", 10, 10, 10))
    budget.charge_edge(4)
    with pytest.raises(ValueError):
        budget.charge_edge(7)  # 4 + 7 > 10
    with pytest.raises(ValueError):  # exhausted, so even a zero-sized charge refuses
        budget.charge_edge(0)


@pytest.mark.parametrize("limit", [1, 3, 9])
def test_depth_limit_minus_one_limit_and_limit_plus_one(limit):
    budget = DiscoveryBudget(DiscoveryTier("t", 1000, 10**6, limit))
    assert budget.visit(object(), limit - 1) is True
    assert budget.visit(object(), limit) is True  # exactly at the limit
    with pytest.raises(ValueError) as excinfo:
        budget.visit(object(), limit + 1)
    assert_stable_rejection(excinfo)


def test_depth_bound_applies_to_non_authoritative_paths_too():
    budget = small(depth=2)
    with pytest.raises(ValueError):
        budget.visit(object(), 3, authoritative=False)


def test_real_tier_constants_are_enforced_at_their_boundaries():
    for tier in (STANDARD, PORTABILITY):
        budget = DiscoveryBudget(tier)
        for _ in range(tier.max_nodes):
            budget.charge_node(object())
        with pytest.raises(ValueError):
            budget.charge_node(object())
        budget = DiscoveryBudget(tier)
        budget.charge_edge(tier.max_edges)
        with pytest.raises(ValueError):
            budget.charge_edge()
        budget = DiscoveryBudget(tier)
        assert budget.visit(object(), tier.max_depth) is True
        with pytest.raises(ValueError):
            budget.visit(object(), tier.max_depth + 1)


def test_nodes_are_deduplicated_by_identity_and_repeats_are_free():
    budget = small(nodes=2)
    first, second = object(), object()
    assert budget.charge_node(first) is True
    assert budget.charge_node(first) is False
    assert budget.charge_node(first) is False
    assert budget.nodes == 1
    assert budget.charge_node(second) is True
    assert budget.nodes == 2


def test_equal_but_distinct_objects_are_distinct_nodes():
    budget = small(nodes=3)
    a, b = [1, 2], [1, 2]
    assert a == b
    assert budget.charge_node(a) is True
    assert budget.charge_node(b) is True
    assert budget.nodes == 2


def test_identity_cannot_be_recycled_during_a_call():
    budget = small(nodes=1000)
    for _ in range(500):
        budget.charge_node(object())  # each temporary would free its id if not retained
    assert budget.nodes == 500


def test_edges_are_charged_for_repeats_of_an_already_seen_node():
    budget = DiscoveryBudget(DiscoveryTier("t", 5, 6, 5))
    shared = object()
    budget.charge_node(shared)
    for _ in range(6):
        budget.charge_edge()  # six references to one node
    assert budget.nodes == 1
    with pytest.raises(ValueError):
        budget.charge_edge()


def test_first_visit_expands_and_non_authoritative_revisit_does_not():
    budget = small()
    obj = object()
    assert budget.visit(obj, 2, authoritative=False) is True
    assert budget.authoritative_depth(obj) is None
    assert budget.visit(obj, 1, authoritative=False) is False
    assert budget.visit(obj, 3, authoritative=False) is False
    assert budget.authoritative_depth(obj) is None


def test_minimum_authoritative_depth_is_retained_independent_of_visit_order():
    for order in ((3, 1, 2), (1, 2, 3), (2, 3, 1)):
        budget = small(nodes=2, depth=5)
        obj = object()
        expansions = [budget.visit(obj, depth) for depth in order]
        assert budget.authoritative_depth(obj) == 1
        assert budget.nodes == 1
        # expansion happens exactly when a new minimum is reached
        minimum, expected = None, []
        for depth in order:
            expected.append(minimum is None or depth < minimum)
            minimum = depth if minimum is None else min(minimum, depth)
        assert expansions == expected


def test_authoritative_visit_after_non_authoritative_one_expands_once_at_its_depth():
    budget = small(depth=5)
    obj = object()
    assert budget.visit(obj, 4, authoritative=False) is True
    assert budget.visit(obj, 4, authoritative=True) is True
    assert budget.authoritative_depth(obj) == 4
    assert budget.visit(obj, 4, authoritative=True) is False
    assert budget.visit(obj, 2, authoritative=True) is True
    assert budget.authoritative_depth(obj) == 2


def test_re_expansion_edges_are_charged_so_the_walk_stays_bounded():
    budget = DiscoveryBudget(DiscoveryTier("t", 10, 8, 20))
    obj = object()
    expansions = 0
    with pytest.raises(ValueError) as excinfo:
        for depth in range(10, 0, -1):  # ever shallower authoritative paths
            if budget.visit(obj, depth):
                expansions += 1
                budget.charge_edge(2)  # the walk charges the edges it traverses
    assert_stable_rejection(excinfo)
    assert expansions == 5  # the fifth expansion's edges are the ones over the limit
    assert budget.edges == 8


def test_one_envelope_is_shared_across_roots():
    budget = small(nodes=4, edges=5, depth=3)
    root_a, root_b = object(), object()
    shared_child = object()
    for root in (root_a, root_b):
        assert budget.visit(root, 0) is True
        budget.charge_edge()
        budget.visit(shared_child, 1)
        budget.charge_edge()
    assert budget.nodes == 3  # the child is counted once across both roots
    assert budget.edges == 4
    budget.charge_edge()  # fifth edge: at the limit across roots
    with pytest.raises(ValueError):
        budget.charge_edge()


def test_second_root_cannot_start_afresh_after_the_first_filled_the_envelope():
    budget = small(nodes=3)
    for _ in range(3):
        budget.charge_node(object())
    with pytest.raises(ValueError):
        budget.visit(object(), 0)  # a second root's first node


def test_rejection_is_sticky_and_value_free():
    budget = small(nodes=1)
    secret = type("Secret", (), {"__repr__": lambda self: "TOP-SECRET"})()
    budget.charge_node(object())
    with pytest.raises(ValueError) as excinfo:
        budget.charge_node(secret)
    assert "TOP-SECRET" not in str(excinfo.value)
    assert_stable_rejection(excinfo)
    for action in (
        lambda: budget.charge_node(object()),
        lambda: budget.charge_node(secret),  # even a node that was never accepted
        lambda: budget.charge_edge(),
        lambda: budget.visit(object(), 0),
    ):
        with pytest.raises(ValueError) as again:
            action()
        assert_stable_rejection(again)


def test_a_budget_that_was_not_exhausted_keeps_accepting():
    budget = small(nodes=3, edges=3, depth=3)
    budget.charge_node(object())
    budget.charge_edge(3)
    with pytest.raises(ValueError):
        budget.charge_edge()
    other = small(nodes=3, edges=3, depth=3)  # envelopes are independent
    other.charge_edge(3)


@pytest.mark.parametrize("bad", [-1, 1.0, True, "1", None])
def test_charge_edge_rejects_invalid_counts(bad):
    with pytest.raises(ValueError):
        small().charge_edge(bad)


@pytest.mark.parametrize("bad", [-1, 1.0, True, "1", None])
def test_visit_rejects_invalid_depth(bad):
    with pytest.raises(ValueError):
        small().visit(object(), bad)


def test_visit_requires_a_boolean_authority_flag():
    for bad in (1, 0, None, "yes"):
        with pytest.raises(TypeError):
            small().visit(object(), 0, authoritative=bad)


def test_accounting_never_calls_hooks_on_hostile_objects():
    budget = small(nodes=5, edges=10, depth=4)
    hostile = [Hostile() for _ in range(3)]
    Hostile.calls = []
    for obj in hostile:
        budget.charge_node(obj)
        budget.charge_node(obj)  # repeat
        budget.visit(obj, 1)
        budget.visit(obj, 0)  # shallower authoritative revisit
        budget.visit(obj, 2, authoritative=False)
        budget.authoritative_depth(obj)
        budget.charge_edge()
        is_exact_type(obj, dict, list, tuple)
    assert budget.nodes == 3
    assert Hostile.calls == []


def test_rejection_paths_never_call_hooks_either():
    budget = small(nodes=1)
    first, second = Hostile(), Hostile()
    Hostile.calls = []
    budget.charge_node(first)
    with pytest.raises(ValueError):
        budget.charge_node(second)
    with pytest.raises(ValueError):
        budget.visit(second, 0)
    assert Hostile.calls == []


def test_exact_type_check_ignores_spoofed_class_and_subclasses():
    class MyDict(dict):
        pass

    assert is_exact_type({}, dict) is True
    assert is_exact_type([], dict, list) is True
    assert is_exact_type(MyDict(), dict) is False
    assert is_exact_type(True, int) is False
    hostile = Hostile()
    Hostile.calls = []
    assert hostile.__class__ is dict  # the spoof is active (this call is the property)
    Hostile.calls = []
    assert is_exact_type(hostile, dict) is False
    assert is_exact_type(hostile, Hostile) is True
    assert Hostile.calls == []


def test_module_is_pure_standard_library():
    import ast
    import sys
    from pathlib import Path

    tree = ast.parse(Path(contracts_limits.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {"dataclasses"}
    assert imported == {"dataclasses"}
    assert imported <= stdlib
    assert "pydantic" not in imported
