"""Truthful observations over the existing asynchronous response models.

These helpers describe what the legacy queue API has actually observed.  They
do not strengthen its guarantees and are deliberately not used by the routes:
the existing response bodies and status codes remain unchanged.
"""

from dataclasses import dataclass, field
from enum import Enum

from schemas import StoreMemoryResponse, TaskAcceptedResponse, TaskStatusResponse


class LegacyAsyncObservationKind(str, Enum):
    """Evidence available from one legacy response."""

    QUEUE_ACKNOWLEDGED = "queue_acknowledged"
    QUEUED = "queued"
    WORKER_PROCESSING = "worker_processing"
    WORKER_RETURNED = "worker_returned"
    WORKER_FAILED = "worker_failed"
    NOT_FOUND_IN_REDIS = "not_found_in_redis"
    UNKNOWN_STATUS = "unknown_status"
    SYNC_HANDLER_RETURNED = "sync_handler_returned"


@dataclass(frozen=True, slots=True)
class LegacyAsyncObservation:
    """A non-authoritative interpretation of an unchanged legacy response.

    ``durable_intent_committed`` and ``all_projections_applied`` are always
    false because neither fact is established by these response shapes.
    """

    kind: LegacyAsyncObservationKind
    task_id: str | None
    durable_intent_committed: bool = field(default=False, init=False)
    all_projections_applied: bool = field(default=False, init=False)


def observe_task_accepted(response: TaskAcceptedResponse) -> LegacyAsyncObservation:
    """Interpret a legacy 202 response as queue acknowledgement only.

    A task identifier and poll URL show that the queue accepted or already
    knew the ARQ job identifier.  They do not prove an atomic canonical
    intent/outbox commit.
    """

    if response.status != "accepted":
        raise ValueError("task accepted response status must be 'accepted'")
    return LegacyAsyncObservation(
        kind=LegacyAsyncObservationKind.QUEUE_ACKNOWLEDGED,
        task_id=response.task_id,
    )


def observe_task_status(response: TaskStatusResponse) -> LegacyAsyncObservation:
    """Interpret an ARQ/Redis polling response without inferring projections.

    In particular, legacy ``completed`` means that ARQ recorded a successful
    worker return.  It does not mean that vector, graph, Markdown, or any other
    projection has committed.
    """

    kinds = {
        "queued": LegacyAsyncObservationKind.QUEUED,
        "processing": LegacyAsyncObservationKind.WORKER_PROCESSING,
        "completed": LegacyAsyncObservationKind.WORKER_RETURNED,
        "failed": LegacyAsyncObservationKind.WORKER_FAILED,
        "not_found": LegacyAsyncObservationKind.NOT_FOUND_IN_REDIS,
    }
    return LegacyAsyncObservation(
        kind=kinds.get(response.status, LegacyAsyncObservationKind.UNKNOWN_STATUS),
        task_id=response.task_id,
    )


def observe_redis_unavailable_sync_200(
    response: StoreMemoryResponse,
) -> LegacyAsyncObservation:
    """Describe the REST route's Redis-unavailable synchronous 200 branch.

    The unchanged body does not identify Redis failure or prove projection
    convergence.  Calling this helper is therefore a control-flow assertion by
    the caller, not a conclusion that can be derived from the response alone.
    """

    return LegacyAsyncObservation(
        kind=LegacyAsyncObservationKind.SYNC_HANDLER_RETURNED,
        task_id=response.task_id,
    )


__all__ = [
    "LegacyAsyncObservation",
    "LegacyAsyncObservationKind",
    "observe_redis_unavailable_sync_200",
    "observe_task_accepted",
    "observe_task_status",
]
