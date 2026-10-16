"""Shared accounting for native contract-graph discovery.

Receiving boundaries that walk the stored (native) state of a contract graph
use one ``DiscoveryBudget`` per validation call, shared by every root they
visit, so the cost of discovery is bounded by the graph the caller actually
stored rather than by what the caller chooses to expose.

The budget is deliberately callback-free.  It reads nothing from the objects it
is given: nodes are identified by ``id()`` (the object is retained so the
identity cannot be recycled during the call) and ``is_exact_type`` compares
``type(obj)`` directly.  It never iterates, hashes, compares, measures,
truth-tests or attribute-reads a caller object, so hostile ``__iter__``,
``__eq__``, ``__hash__``, ``__len__``, ``__bool__``, ``__getattr__`` or
``__class__`` overrides are never invoked by the accounting itself.

Accounting vocabulary:

* a *node* is a unique object, deduplicated by identity and charged once;
* an *edge* is one captured container entry or stored field and is charged
  every time it is traversed, including repeats of an already seen node, so
  aliasing cannot hide work;
* the *depth frontier* tracks ``(object, depth, authoritative)``.  Depth is
  bounded for every path.  An object first reached through a deep path is
  expanded again when a shallower *authoritative* path reaches it, so a depth
  outcome never depends on traversal order.  Re-expansion is bounded because
  the retained depth strictly decreases, and its edges are charged.

Exceeding any limit raises ``ValueError`` with the stable, value-free message
``DISCOVERY_BUDGET_MESSAGE`` and leaves the budget exhausted: later charges
raise again.  Callers should apply their strict top-level type checks first,
then charge the budget, then run their ordered snapshot and diagnostics.

The limits are internal named constants measured against legitimate inputs
with generous headroom.  They are not an advertised capacity.  This module is
pure standard library so every lineage can merge it without conflicts.
"""

from dataclasses import dataclass

DISCOVERY_BUDGET_MESSAGE = "contract graph exceeds the native discovery budget"

STANDARD_MAX_NODES = 16_384
STANDARD_MAX_EDGES = 131_072
STANDARD_MAX_DEPTH = 256

PORTABILITY_MAX_NODES = 131_072
PORTABILITY_MAX_EDGES = 786_432
PORTABILITY_MAX_DEPTH = 256


def _positive_int(value):
    return type(value) is int and value > 0


@dataclass(frozen=True)
class DiscoveryTier:
    """The three limits applied to one validation call."""

    name: str
    max_nodes: int
    max_edges: int
    max_depth: int

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise ValueError("tier name must be a non-empty string")
        for field in ("max_nodes", "max_edges", "max_depth"):
            if not _positive_int(getattr(self, field)):
                raise ValueError(f"{field} must be a positive integer")


STANDARD = DiscoveryTier(
    "standard", STANDARD_MAX_NODES, STANDARD_MAX_EDGES, STANDARD_MAX_DEPTH
)
"""Contracts with small declared inventories (engines, tenancy, readiness)."""

PORTABILITY = DiscoveryTier(
    "portability", PORTABILITY_MAX_NODES, PORTABILITY_MAX_EDGES, PORTABILITY_MAX_DEPTH
)
"""In-process ``PortableManifest`` instances; larger exports use the wire path."""


def is_exact_type(obj, *types):
    """Return whether ``type(obj)`` is exactly one of ``types`` (no hooks run)."""

    return type(obj) in types


def _exceeded():
    return ValueError(DISCOVERY_BUDGET_MESSAGE)


class DiscoveryBudget:
    """One shared envelope for a validation call across all of its roots."""

    __slots__ = ("_tier", "_nodes", "_edges", "_seen", "_auth_depth", "_exhausted")

    def __init__(self, tier=STANDARD):
        if type(tier) is not DiscoveryTier:
            raise TypeError("tier must be a DiscoveryTier")
        self._tier = tier
        self._nodes = 0
        self._edges = 0
        self._seen = {}
        self._auth_depth = {}
        self._exhausted = False

    @property
    def tier(self):
        return self._tier

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def _fail(self):
        self._exhausted = True
        raise _exceeded()

    def charge_node(self, obj):
        """Charge ``obj`` once by identity; return True when it is new."""

        if self._exhausted:
            raise _exceeded()
        key = id(obj)
        if key in self._seen:
            return False
        if self._nodes >= self._tier.max_nodes:
            self._fail()
        self._nodes += 1
        self._seen[key] = obj  # retained so id() cannot be recycled mid-call
        return True

    def charge_edge(self, count=1):
        """Charge ``count`` captured edges; repeats of seen nodes are charged."""

        if type(count) is not int or count < 0:
            raise ValueError("count must be a non-negative integer")
        if self._exhausted:
            raise _exceeded()
        if count > self._tier.max_edges - self._edges:
            self._fail()
        self._edges += count

    def visit(self, obj, depth, authoritative=True):
        """Record a visit at ``depth``; return whether to (re)expand ``obj``.

        Raises when ``depth`` exceeds the tier limit or a new node would exceed
        the node limit.  A first visit always expands.  A later visit expands
        again only when it is authoritative and shallower than every earlier
        authoritative visit, so the retained depth is the minimum
        authoritative depth.
        """

        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if type(authoritative) is not bool:
            raise TypeError("authoritative must be a bool")
        if self._exhausted:
            raise _exceeded()
        if depth > self._tier.max_depth:
            self._fail()
        if self.charge_node(obj):
            if authoritative:
                self._auth_depth[id(obj)] = depth
            return True
        if not authoritative:
            return False
        best = self._auth_depth.get(id(obj))
        if best is None or depth < best:
            self._auth_depth[id(obj)] = depth
            return True
        return False

    def authoritative_depth(self, obj):
        """Return the minimum authoritative depth seen for ``obj`` or ``None``."""

        return self._auth_depth.get(id(obj))


__all__ = [
    "DISCOVERY_BUDGET_MESSAGE",
    "DiscoveryBudget",
    "DiscoveryTier",
    "PORTABILITY",
    "PORTABILITY_MAX_DEPTH",
    "PORTABILITY_MAX_EDGES",
    "PORTABILITY_MAX_NODES",
    "STANDARD",
    "STANDARD_MAX_DEPTH",
    "STANDARD_MAX_EDGES",
    "STANDARD_MAX_NODES",
    "is_exact_type",
]
