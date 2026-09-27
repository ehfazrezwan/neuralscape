"""Characterization tests for the unchanged legacy async wire contract."""

from contracts_legacy_async import (
    LegacyAsyncObservationKind,
    observe_redis_unavailable_sync_200,
    observe_task_accepted,
    observe_task_status,
)
from schemas import MemoryResponse, StoreMemoryResponse, TaskAcceptedResponse, TaskStatusResponse


def test_task_accepted_wire_shape_is_unchanged_and_only_acknowledges_queue() -> None:
    response = TaskAcceptedResponse(task_id="ns-123", poll_url="/v1/memories/status/ns-123")

    assert response.model_dump() == {
        "status": "accepted",
        "task_id": "ns-123",
        "poll_url": "/v1/memories/status/ns-123",
    }
    observation = observe_task_accepted(response)
    assert observation.kind is LegacyAsyncObservationKind.QUEUE_ACKNOWLEDGED
    assert observation.durable_intent_committed is False
    assert observation.all_projections_applied is False


def test_completed_status_means_worker_return_not_projection_convergence() -> None:
    response = TaskStatusResponse(
        task_id="ns-123",
        status="completed",
        result={"memories": [{"id": "memory-1"}]},
        error=None,
    )

    assert response.model_dump() == {
        "task_id": "ns-123",
        "status": "completed",
        "result": {"memories": [{"id": "memory-1"}]},
        "error": None,
    }
    observation = observe_task_status(response)
    assert observation.kind is LegacyAsyncObservationKind.WORKER_RETURNED
    assert observation.all_projections_applied is False


def test_each_documented_legacy_status_is_characterized_without_new_wire_enum() -> None:
    expected = {
        "queued": LegacyAsyncObservationKind.QUEUED,
        "processing": LegacyAsyncObservationKind.WORKER_PROCESSING,
        "failed": LegacyAsyncObservationKind.WORKER_FAILED,
        "not_found": LegacyAsyncObservationKind.NOT_FOUND_IN_REDIS,
    }
    for status, kind in expected.items():
        response = TaskStatusResponse(task_id="ns-123", status=status)
        assert observe_task_status(response).kind is kind

    # The current response model accepts arbitrary strings.  Characterization
    # records that fact rather than tightening the existing public wire shape.
    unknown = TaskStatusResponse(task_id="ns-123", status="provider_paused")
    assert observe_task_status(unknown).kind is LegacyAsyncObservationKind.UNKNOWN_STATUS


def test_sync_200_body_does_not_claim_queue_or_projection_completion() -> None:
    response = StoreMemoryResponse(
        memories=[MemoryResponse(id="memory-1", memory="Prefers concise answers")]
    )

    assert response.model_dump(exclude_none=True) == {
        "status": "ok",
        "memories": [{"id": "memory-1", "memory": "Prefers concise answers"}],
    }
    observation = observe_redis_unavailable_sync_200(response)
    assert observation.kind is LegacyAsyncObservationKind.SYNC_HANDLER_RETURNED
    assert observation.task_id is None
    assert observation.durable_intent_committed is False
    assert observation.all_projections_applied is False
