"""Product contracts for upstream verification and local projection freshness.

An upstream checkpoint reports what was observed at a source authority.  It is
deliberately independent of projection freshness: a local index can be healthy
and internally current while its last successful upstream verification is old.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Sequence

from pydantic import AfterValidator, AwareDatetime, Field, model_validator

from contracts_common import (
    ContractModel,
    OpaqueId,
    VersionedContract,
    is_canonical_enum_member,
    snapshot_contract_graph,
)
from contracts_references import ReferenceHandle, SourceVersion


class UpstreamVerificationStatus(str, Enum):
    """Knowledge available about the source authority, not an index status."""

    CURRENT = "current"
    STALE = "stale"
    UNVERIFIED = "unverified"
    UNAVAILABLE = "unavailable"


class VerificationMethod(str, Enum):
    """Recognized basis used to check an upstream source.

    Recognition does not imply that this candidate can support a known
    freshness claim for every method.  Methods without reviewed evidence and
    resolution semantics remain available for truthful weak-status reports.
    """

    PROVIDER_REVISION = "provider_revision"
    OPAQUE_CURSOR = "opaque_cursor"
    CONDITIONAL_READ = "conditional_read"
    CONTENT_DIGEST = "content_digest"
    OBSERVATION_ONLY = "observation_only"


class ReconciliationState(str, Enum):
    """Whether the connector knows of unprocessed or uncertain source ranges."""

    COMPLETE = "complete"
    KNOWN_GAP = "known_gap"
    RECONCILING = "reconciling"
    UNKNOWN = "unknown"


class ProjectionStatus(str, Enum):
    """Relationship between a projection and its declared canonical inputs."""

    CURRENT = "current"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


_DATETIME_PARTS = (
    datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute, datetime.second,
    datetime.microsecond,
)
_TIMEDELTA_PARTS = (timedelta.days, timedelta.seconds, timedelta.microseconds)


def pin_datetime(value: object) -> object:
    """A caller-owned aware ``datetime`` as a plain one with a fixed offset, its zone asked exactly once.

    The offset is read through the built-in accessor (a ``datetime`` subclass's own ``utcoffset`` is never
    called) and rebuilt as an exact ``timedelta`` from its base fields (a ``timedelta`` subclass's arithmetic is
    never run); the fields are read through the built-in descriptors.  Every later comparison is then between
    plain values, so a zone that answers differently on a later read, or fails on it, cannot split validation
    from the check that follows.  A zone that raises, or answers with something that is no offset, is a value
    error.  Anything that is not a ``datetime``, and a naive one, is returned unchanged for validation to refuse.

    A zone that is present but answers ``None`` also comes back as the caller's own object, so a caller that keeps
    the result must check it (``_is_pinned``) and never cache it: a later read of that zone could accept what was
    unusable when it was captured.
    """

    if not any(base is datetime for base in vars(type)["__mro__"].__get__(type(value))):
        return value
    try:
        offset = datetime.utcoffset(value)  # type: ignore[arg-type]
        if offset is None:
            return value
        exact = timedelta(*(part.__get__(offset, timedelta) for part in _TIMEDELTA_PARTS))
        fields = [part.__get__(value, datetime) for part in _DATETIME_PARTS]
        return datetime(*fields, tzinfo=timezone(exact))
    except Exception:
        raise ValueError("a datetime must have a usable UTC offset") from None


def _is_pinned(value: object) -> bool:
    """Whether ``value`` is what a successful pin produces: a plain datetime on an exact fixed-offset zone.

    ``pin_datetime`` hands back the caller's own object when the zone reports no offset at the pin, and that
    object still has a (caller-owned) zone, so "aware" is not enough to tell the two apart.
    """

    return type(value) is datetime and type(value.tzinfo) is timezone


def _pin_validated(value: datetime) -> datetime:
    """Pin a datetime pydantic has just accepted as aware (a zone that now reports no offset is refused)."""

    pinned = pin_datetime(value)
    if not _is_pinned(pinned):
        raise ValueError("a datetime must have a usable UTC offset")
    return pinned


def _pin_at_capture(value: object) -> object:
    """The pin a snapshot applies to a datetime leaf as its entry is captured.

    Aware datetimes become exact fixed-offset copies; a zone that is present but reports no usable offset is
    refused here (never kept to be read again later); a naive datetime is left for validation to refuse.
    """

    pinned = pin_datetime(value)
    if datetime.tzinfo.__get__(value) is not None and not _is_pinned(pinned):  # type: ignore[arg-type]
        raise ValueError("a datetime must have a usable UTC offset")
    return pinned


def snapshot_pinned(value: object, contract: object = None) -> object:
    """``snapshot_contract_graph`` with every datetime leaf pinned at capture, before any iterator advances.

    ``contract`` is the class (or, for a tuple of two roots, the classes) the caller validates the snapshot
    against: public-view entries are validated early against that declared contract.
    """

    return snapshot_contract_graph(value, pin_leaf=_pin_at_capture, entry_guard=require_native_snapshot, contract=contract)  # type: ignore[arg-type]


PinnedAwareDatetime = Annotated[AwareDatetime, AfterValidator(_pin_validated)]
"""An aware datetime stored as a plain fixed-offset one, so no later check can see a different instant."""


def _instant_after(left: datetime, right: datetime) -> bool:
    """Whether ``left`` is a later instant than ``right``, read without leaving the datetime range.

    Two datetimes that share one zone object compare by wall time and ignore ``fold``, which is wrong
    across a clock rollback, so the instants are ordered by wall time minus offset.  Converting to
    UTC raises ``OverflowError`` at the ends of the range; the difference of two naive datetimes and
    the difference of two offsets cannot.  Each offset is read once.
    """

    pinned = [pin_datetime(value) for value in (left, right)]
    if not all(_is_pinned(value) for value in pinned):
        raise ValueError("a datetime must have a usable UTC offset")
    return pinned[0] > pinned[1]  # plain aware datetimes with fixed offsets compare by instant


class SourceCheckpoint(VersionedContract):
    """A versioned observation of one upstream source authority.

    Provider revisions and cursors are opaque within a source.  They must not be
    ordered or compared across sources.  ``observation_only`` is the explicit
    weaker basis for a source that exposes neither kind of version marker.
    Current and known-stale evidence both require a recorded successful check.
    ``content_digest`` and ``conditional_read`` are recognized methods, but
    their known-status evidence protocols are deferred in this candidate; they
    can currently report only ``unverified`` or ``unavailable``.
    """

    source: ReferenceHandle
    verification_status: UpstreamVerificationStatus
    verification_method: VerificationMethod
    observed_at: PinnedAwareDatetime
    last_successful_verification_at: PinnedAwareDatetime | None
    provider_revision: OpaqueId | None
    opaque_cursor: OpaqueId | None
    processed_through: OpaqueId | None
    reconciliation_state: ReconciliationState
    freshness_policy: OpaqueId

    @model_validator(mode="after")
    def validate_verification_basis(self) -> "SourceCheckpoint":
        if self.source.kind != "source":
            raise ValueError("a source checkpoint requires a source reference")

        if self.last_successful_verification_at is not None:
            if _instant_after(self.last_successful_verification_at, self.observed_at):
                raise ValueError(
                    "last_successful_verification_at cannot be after observed_at"
                )

        marker_by_method = {
            VerificationMethod.PROVIDER_REVISION: self.provider_revision,
            VerificationMethod.OPAQUE_CURSOR: self.opaque_cursor,
        }
        required_marker = marker_by_method.get(self.verification_method)
        if self.verification_method in marker_by_method and required_marker is None:
            raise ValueError(
                f"{self.verification_method.value} requires its corresponding marker"
            )

        if self.verification_method is VerificationMethod.OBSERVATION_ONLY:
            if self.provider_revision is not None or self.opaque_cursor is not None:
                raise ValueError(
                    "observation_only cannot claim a provider revision or cursor"
                )

        if self.verification_status in {
            UpstreamVerificationStatus.CURRENT,
            UpstreamVerificationStatus.STALE,
        } and self.last_successful_verification_at is None:
            raise ValueError(
                f"{self.verification_status.value} verification requires "
                "a successful check time"
            )

        if self.verification_status in {
            UpstreamVerificationStatus.CURRENT,
            UpstreamVerificationStatus.STALE,
        } and self.verification_method in {
            VerificationMethod.CONTENT_DIGEST,
            VerificationMethod.CONDITIONAL_READ,
        }:
            raise ValueError(
                f"{self.verification_method.value} cannot claim current or stale "
                "until its evidence protocol is supported"
            )

        if self.verification_status is UpstreamVerificationStatus.CURRENT:
            if self.reconciliation_state is not ReconciliationState.COMPLETE:
                raise ValueError(
                    "current verification cannot contain a known source gap"
                )

        return self


class ProjectionFreshness(ContractModel):
    """Local projection state and the complete source-version set it consumed."""

    status: ProjectionStatus
    applied_sources: tuple[SourceVersion, ...] = Field(min_length=1)
    verified_at: PinnedAwareDatetime

    @model_validator(mode="after")
    def require_unique_sources(self) -> "ProjectionFreshness":
        _version_map(self.applied_sources)
        return self


class FreshnessWitness(ContractModel):
    """Independent upstream and local-projection freshness evidence.

    A checkpoint's source ID maps directly to ``SourceVersion.record_id``.
    Current and known-stale upstream statuses require exact checkpoint coverage
    of every applied source.  Unverified and unavailable statuses may have
    partial coverage but never unrelated sources.
    """

    upstream_status: UpstreamVerificationStatus
    source_checkpoints: tuple[SourceCheckpoint, ...]
    projection: ProjectionFreshness

    @model_validator(mode="after")
    def require_unique_checkpoints(self) -> "FreshnessWitness":
        source_keys = [
            (checkpoint.source.kind, checkpoint.source.id)
            for checkpoint in self.source_checkpoints
        ]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source_checkpoints must identify unique sources")
        checkpoint_tenants = {
            str(checkpoint.source.tenant_id)
            for checkpoint in self.source_checkpoints
        }
        if len(checkpoint_tenants) > 1:
            raise ValueError("source checkpoints must share one tenant scope")

        checkpoint_ids = {
            str(checkpoint.source.id) for checkpoint in self.source_checkpoints
        }
        applied_ids = {
            str(version.record_id) for version in self.projection.applied_sources
        }
        if not checkpoint_ids.issubset(applied_ids):
            raise ValueError(
                "checkpoint source IDs must map to applied source record IDs"
            )
        has_complete_coverage = checkpoint_ids == applied_ids
        checkpoint_statuses = {
            checkpoint.verification_status
            for checkpoint in self.source_checkpoints
        }
        if self.upstream_status is UpstreamVerificationStatus.CURRENT:
            if not has_complete_coverage:
                raise ValueError(
                    "current upstream status requires every applied source checkpoint"
                )
            if any(
                checkpoint.verification_status
                is not UpstreamVerificationStatus.CURRENT
                for checkpoint in self.source_checkpoints
            ):
                raise ValueError("current upstream status requires current checkpoints")
        elif self.upstream_status is UpstreamVerificationStatus.STALE:
            if not has_complete_coverage:
                raise ValueError(
                    "stale upstream status requires every applied source checkpoint"
                )
            if not checkpoint_statuses.issubset(
                {
                    UpstreamVerificationStatus.CURRENT,
                    UpstreamVerificationStatus.STALE,
                }
            ):
                raise ValueError(
                    "stale upstream status requires only current or stale checkpoints"
                )
            if UpstreamVerificationStatus.STALE not in checkpoint_statuses:
                raise ValueError(
                    "stale upstream status requires at least one stale checkpoint"
                )
        elif self.upstream_status is UpstreamVerificationStatus.UNVERIFIED:
            if UpstreamVerificationStatus.UNAVAILABLE in checkpoint_statuses:
                raise ValueError(
                    "unverified upstream status cannot hide unavailable checkpoints"
                )
            if (
                has_complete_coverage
                and UpstreamVerificationStatus.UNVERIFIED
                not in checkpoint_statuses
            ):
                raise ValueError(
                    "unverified upstream status requires unverified evidence "
                    "or incomplete checkpoint coverage"
                )
        elif (
            has_complete_coverage
            and UpstreamVerificationStatus.UNAVAILABLE not in checkpoint_statuses
        ):
            raise ValueError(
                "unavailable upstream status requires unavailable evidence "
                "or incomplete checkpoint coverage"
            )
        return self


_NATIVE_SCALARS = (type(None), bool, int, float, str, datetime)
_NATIVE_CONTAINERS = (dict, list, tuple)
_TYPE_MRO = vars(type)["__mro__"]
# The scalar kinds a mapping key may have (a datetime is a value only).
_NATIVE_KEY_KINDS = (type(None), bool, int, float, str)

_ALLOWED_ENUMS: tuple[type, ...] = (
    UpstreamVerificationStatus,
    VerificationMethod,
    ReconciliationState,
    ProjectionStatus,
)


def allow_native_enums(*enums: type) -> None:
    """Add the contract modules' own enum classes to the supported leaves (fail closed otherwise).

    A member is a leaf only when its class is one of these, by identity: a foreign enum, even one
    with the same name and values, could report a model through ``__class__`` or delegate to one.
    """

    global _ALLOWED_ENUMS
    for enum in enums:
        if not (isinstance(enum, type) and any(base is Enum for base in _TYPE_MRO.__get__(enum))):
            raise TypeError("only an Enum class can be allowed as a snapshot leaf")
    _ALLOWED_ENUMS = _ALLOWED_ENUMS + tuple(enum for enum in enums if not any(enum is known for known in _ALLOWED_ENUMS))


_PATH_LIMIT = 200
_KEY_SHOWN = 40


_INT_LABEL_BITS = 128


def _label_text(label: object) -> str:
    """A short label for a key or position, built with the base types' own methods only.

    A key is an exact native scalar, but its text can still run caller code (a ``datetime``'s zone has its own
    ``repr``) or be very large (an ``int`` with thousands of digits), and a message is built while the value is
    being refused.  Text is cut and quoted by ``str``, integers and floats use their base ``repr`` (a huge integer
    is named by its size), and a datetime or anything else is a fixed word.
    """

    kind = type(label)
    if kind is str:
        text = str.__getitem__(label, slice(0, _KEY_SHOWN))
        return str.__repr__(text) + ("..." if str.__len__(label) > _KEY_SHOWN else "")
    if kind is bool:
        return "True" if label is True else "False"
    if kind is type(None):
        return "None"
    if kind is int:
        bits = int.bit_length(label)  # type: ignore[arg-type]
        return f"<int of {bits} bits>" if bits > _INT_LABEL_BITS else int.__repr__(label)  # type: ignore[arg-type]
    if kind is float:
        return float.__repr__(label)  # type: ignore[arg-type]
    if kind is datetime:
        return "<datetime key>"
    return "<key>"


def _trail_path(trail: list[tuple[int, object]], index: int, tail: str = "") -> str:
    """The path to a record in ``trail`` (exact text keys and integer positions only), shortened for a message."""

    parts: list[str] = []
    while index >= 0:
        parent, label = trail[index]
        if parent < 0:
            parts.append("$")  # the root of the snapshot
        else:
            parts.append(f"[{_label_text(label)}]")
        index = parent
    text = "".join(reversed(parts)) + tail
    return text if len(text) <= _PATH_LIMIT else text[:_PATH_LIMIT] + "..."


def require_native_snapshot(snapshot: object) -> object:
    """Reject, before Pydantic sees it, any value that is not a supported native kind.

    ``snapshot_contract_graph`` rebuilds stored models as exact dicts and returns every other
    value as it is.  An object that merely reports a model class through ``__class__`` (or
    delegates to one) would reach validation, where Pydantic reads ``__class__`` and may run
    caller code.  Only real types are consulted: exact scalars and containers by identity,
    enum members only of the contracts' own enum classes (``allow_native_enums``), also by identity.
    The walk is iterative.
    """

    # Each container visited gets one small record, (index of its parent's record, the key or position that
    # leads to it); a path string is built from that chain only when a value is refused, so a long key is held
    # once however many descendants lie below it.
    trail: list[tuple[int, object]] = []
    pending: list[tuple[object, int, object]] = [(snapshot, -1, None)]
    seen: set[int] = set()
    while pending:
        value, parent, label = pending.pop()
        kind = type(value)
        if any(kind is container for container in _NATIVE_CONTAINERS):
            if id(value) in seen:
                continue
            seen.add(id(value))
            trail.append((parent, label))
            here = len(trail) - 1
            if kind is dict:
                for key, item in dict.items(value):
                    if not any(type(key) is scalar for scalar in _NATIVE_KEY_KINDS):
                        # A key is judged like any other leaf: exact native kinds only (strict validation then
                        # refuses a non-text key itself), never an object whose hooks or repr could run.  A
                        # datetime is a value, never a key: no contract has a datetime-keyed field, and strict
                        # validation would format the key (and its caller-owned zone) in its error.
                        raise ValueError(f"unsupported key type at {_trail_path(trail, here, '<key>')}")
                    pending.append((item, here, key))
            else:
                pending.extend((item, here, index) for index, item in enumerate(value))
        elif not (
            any(kind is scalar for scalar in _NATIVE_SCALARS)
            or any(kind is allowed for allowed in _ALLOWED_ENUMS)
        ):
            trail.append((parent, label))
            raise ValueError(f"unsupported value type at {_trail_path(trail, len(trail) - 1)}")
        elif any(kind is allowed for allowed in _ALLOWED_ENUMS) and not is_canonical_enum_member(value):
            # ``str.__new__(Enum, "value")`` has the right class but is no member: every identity test on
            # it fails and a decision falls through to its default branch.  Members only, by identity.
            trail.append((parent, label))
            raise ValueError(f"noncanonical enum value at {_trail_path(trail, len(trail) - 1)}")
    return snapshot


def pin_native_snapshot(snapshot: object) -> object:
    """A copy of a checked native snapshot with every ``datetime`` leaf pinned (its zone asked once, now).

    ``snapshot_contract_graph`` hands a stored datetime back as the caller's own object, zone included.
    Replacing each with a plain fixed-offset value here, once and before anything is validated or compared,
    means no later step can read a different offset.  Iterative; the snapshot is acyclic and its containers
    are exact ``dict``, ``list`` and ``tuple`` (``require_native_snapshot`` has checked that).
    """

    rebuilt: dict[int, object] = {}
    pinned: dict[int, tuple[object, object]] = {}  # one datetime object, however often it appears, is read once

    def is_container(kind: type) -> bool:
        return kind is dict or kind is list or kind is tuple  # identity: no class's __eq__ is consulted

    def leaf(value: object) -> object:
        kind = type(value)
        if kind is datetime:
            known = pinned.get(id(value))
            if known is None:
                result = pin_datetime(value)
                if datetime.tzinfo.__get__(value) is not None and not _is_pinned(result):
                    # a zone that is there but reported no offset at the pin: never kept as the result, or a later
                    # read of the caller's zone could accept what was unusable when captured
                    raise ValueError("a datetime must have a usable UTC offset")
                known = pinned[id(value)] = (value, result)  # the original stays alive: no id reuse
            return known[1]
        return rebuilt.get(id(value), value) if is_container(kind) else value

    # A true post-order: a container is rebuilt only after every container beneath it, whichever parent
    # reaches it first, so each reference to a shared container gets the one rebuilt copy.
    stack: list[tuple[object, bool]] = [(snapshot, False)]
    entered: set[int] = set()
    while stack:
        node, children_done = stack.pop()
        kind = type(node)
        if not is_container(kind) or id(node) in rebuilt:
            continue
        if not children_done:
            if id(node) in entered:
                continue  # already being expanded (a shared node, or a cycle: never expanded twice)
            entered.add(id(node))
            stack.append((node, True))
            children = dict.values(node) if kind is dict else node  # type: ignore[arg-type]
            stack.extend((child, False) for child in children if is_container(type(child)))
        elif kind is dict:
            rebuilt[id(node)] = {key: leaf(item) for key, item in dict.items(node)}
        elif kind is list:
            rebuilt[id(node)] = [leaf(item) for item in node]  # type: ignore[attr-defined]
        else:
            rebuilt[id(node)] = tuple(leaf(item) for item in node)  # type: ignore[attr-defined]
    return leaf(snapshot)


def projection_status_for(
    applied_sources: Sequence[SourceVersion],
    required_sources: Sequence[SourceVersion],
) -> ProjectionStatus:
    """Compare complete dependency sets without using a global max revision.

    A source's content revision and policy epoch are separate requirements.  A
    missing, extra, or changed source therefore makes the applied projection
    stale.  Duplicate source identities are invalid and raise ``ValueError``.
    """

    applied, required = _version_maps(applied_sources, required_sources)
    return (
        ProjectionStatus.CURRENT
        if applied == required
        else ProjectionStatus.STALE
    )


def _version_map(versions: Sequence[SourceVersion]) -> dict[str, tuple[int, int]]:
    return _version_maps(versions)[0]


_MAX_SEQUENCE_ENTRIES = 131_072
"""The most entries read from the two dependency sequences together (the snapshot's visit budget)."""
_SEQUENCE_BUDGET_MESSAGE = "dependency sequences exceed the snapshot budget"


def _version_maps(*sequences: Sequence[SourceVersion]) -> tuple[dict[str, tuple[int, int]], ...]:
    """One map per sequence, each entry captured at first observation, before any callback can edit it.

    1. Every argument that is an exact ``tuple`` or ``list`` is read through its native storage and
       all of them are captured together in one snapshot *before any argument's public iterator
       runs*, so the first argument's iterator cannot repair an invalid model held natively in the
       second (nor the reverse).
    2. Any other sequence is a public view: it is iterated once, in argument order, and each entry
       is captured the moment it is yielded, before the iterator advances, so an entry exposed and
       repaired later is judged as it was exposed.  Every entry read is charged to a budget, so an
       iterator that never ends is cut off.
    3. A model reachable twice is captured once, at its first exposure, and that capture is reused.
    """

    # The native entries count against the same budget as the iterated ones, and are counted from their built-in
    # lengths before anything is copied, so an oversized native sequence is refused without being read.
    native_entries = sum(
        tuple.__len__(sequence) if type(sequence) is tuple else list.__len__(sequence)  # type: ignore[arg-type]
        for sequence in sequences
        if type(sequence) is tuple or type(sequence) is list
    )
    if native_entries > _MAX_SEQUENCE_ENTRIES:
        raise ValueError(_SEQUENCE_BUDGET_MESSAGE)
    native: list[tuple[object, ...] | None] = []
    for sequence in sequences:
        kind = type(sequence)
        if kind is tuple:
            native.append(tuple(tuple.__iter__(sequence)))  # type: ignore[arg-type]
        elif kind is list:
            native.append(tuple(list.__iter__(sequence)))  # type: ignore[arg-type]
        else:
            native.append(None)

    first_capture: dict[int, tuple[object, object]] = {}  # id -> (entry, snapshot); the entry stays alive
    exact_groups = [group for group in native if group is not None]
    exact_snapshots = iter(())
    if exact_groups:
        captured = require_native_snapshot(snapshot_contract_graph(tuple(exact_groups), entry_guard=require_native_snapshot))
        exact_snapshots = iter(captured)  # type: ignore[call-overload]
        for group, snapshots in zip(exact_groups, captured):  # type: ignore[call-overload]
            for entry, snapshot in zip(group, snapshots):
                first_capture.setdefault(id(entry), (entry, snapshot))

    def admit(result: dict[str, tuple[int, int]], snapshot: object) -> None:
        version = SourceVersion.model_validate(snapshot, strict=True)
        record_id = str(version.record_id)
        if record_id in result:
            raise ValueError(f"duplicate source version for record_id {record_id!r}")
        result[record_id] = (version.content_revision, version.policy_epoch)

    read = native_entries  # carried into the iterated entries: one combined limit
    maps = []
    for sequence, group in zip(sequences, native):
        result: dict[str, tuple[int, int]] = {}
        if group is not None:
            for snapshot in next(exact_snapshots):  # type: ignore[call-overload]
                admit(result, snapshot)
        else:
            iterator = iter(sequence)
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                read += 1
                if read > _MAX_SEQUENCE_ENTRIES:
                    raise ValueError(_SEQUENCE_BUDGET_MESSAGE)
                seen = first_capture.get(id(entry))
                if seen is not None and seen[0] is entry:
                    admit(result, seen[1])
                    continue
                snapshot = require_native_snapshot(snapshot_contract_graph(entry, entry_guard=require_native_snapshot))
                first_capture[id(entry)] = (entry, snapshot)
                # Validate now: an invalid entry ends the call before the iterator is asked for the next
                # one, so the work done on invalid input is bounded by the first bad entry (each entry
                # is snapshotted under its own budget, so reading on would cost one full copy apiece).
                admit(result, snapshot)
        maps.append(result)
    return tuple(maps)


__all__ = [
    "FreshnessWitness",
    "PinnedAwareDatetime",
    "pin_datetime",
    "snapshot_pinned",
    "pin_native_snapshot",
    "ProjectionFreshness",
    "ProjectionStatus",
    "allow_native_enums",
    "require_native_snapshot",
    "ReconciliationState",
    "SourceCheckpoint",
    "UpstreamVerificationStatus",
    "VerificationMethod",
    "projection_status_for",
]
