"""Unit tests for `occurred_at` — the event-time envelope field.

NS stamps `created_at` at write time; `occurred_at` is the optional
event-time override for historical ingestion (imported journals, old chat
exports). Absence means "event time unknown — fall back to created_at";
it is NEVER defaulted to the storage time.

Covers: validation (ISO 8601, future-date rejection with clock-skew
allowance, naive→UTC), request-model exposure, write-path payload
stamping (raw / batch / conversation), absent-means-absent, response
surfacing, ask evidence rendering + recency-discipline text, and MCP
tool-schema exposure. All external services mocked (unit-test convention).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from memory_service import MemoryService
from schemas import (
    IngestTextRequest,
    MemoryResponse,
    RawMemoryRequest,
    StoreMemoryRequest,
    validate_occurred_at,
)


# ──────────────────────────────────────────────
# Piece 1: validation + schema exposure
# ──────────────────────────────────────────────


class TestValidateOccurredAt:
    def test_none_passes_through(self):
        assert validate_occurred_at(None) is None

    def test_aware_iso_accepted(self):
        out = validate_occurred_at("2023-05-01T10:00:00+02:00")
        assert out == "2023-05-01T10:00:00+02:00"

    def test_z_suffix_accepted(self):
        out = validate_occurred_at("2023-05-01T10:00:00Z")
        assert out == "2023-05-01T10:00:00+00:00"

    def test_naive_assumes_utc(self):
        out = validate_occurred_at("2023-05-01T10:00:00")
        assert out == "2023-05-01T10:00:00+00:00"

    def test_date_only_accepted(self):
        # fromisoformat accepts a bare date; it normalizes to midnight UTC.
        out = validate_occurred_at("2023-05-01")
        assert out == "2023-05-01T00:00:00+00:00"

    def test_garbage_rejected(self):
        with pytest.raises(ValueError, match="occurred_at"):
            validate_occurred_at("last tuesday")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="occurred_at"):
            validate_occurred_at("")

    def test_far_future_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        with pytest.raises(ValueError, match="future"):
            validate_occurred_at(future)

    def test_future_within_clock_skew_accepted(self):
        near = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert validate_occurred_at(near) == near

    def test_datetime_input_accepted(self):
        dt = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert validate_occurred_at(dt) == dt.isoformat()


class TestRequestModels:
    def test_raw_request_accepts_and_normalizes(self):
        req = RawMemoryRequest(
            content="x", category="preference",
            occurred_at="2023-05-01T10:00:00Z",
        )
        assert req.occurred_at == "2023-05-01T10:00:00+00:00"

    def test_raw_request_defaults_none(self):
        req = RawMemoryRequest(content="x", category="preference")
        assert req.occurred_at is None

    def test_raw_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            RawMemoryRequest(content="x", category="preference", occurred_at="nope")

    def test_raw_request_rejects_far_future(self):
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        with pytest.raises(ValidationError):
            RawMemoryRequest(content="x", category="preference", occurred_at=future)

    def test_conversation_request_accepts(self):
        req = StoreMemoryRequest(
            messages=[{"role": "user", "content": "hi"}],
            occurred_at="2022-01-01T00:00:00+00:00",
        )
        assert req.occurred_at == "2022-01-01T00:00:00+00:00"

    def test_conversation_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            StoreMemoryRequest(messages=[{"role": "user", "content": "hi"}],
                               occurred_at="whenever")

    def test_ingest_text_request_accepts(self):
        req = IngestTextRequest(content="ctx", occurred_at="2021-06-01T12:00:00Z")
        assert req.occurred_at == "2021-06-01T12:00:00+00:00"

    def test_ingest_text_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            IngestTextRequest(content="ctx", occurred_at="not a time")

    def test_memory_response_has_field_defaulting_none(self):
        resp = MemoryResponse(id="m1", memory="x")
        assert resp.occurred_at is None
