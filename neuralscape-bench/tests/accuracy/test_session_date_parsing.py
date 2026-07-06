"""Tests for session date parsing and occurred_at pass-through (T1.3 part B)."""

from __future__ import annotations

import pytest
from httpx import MockTransport, Request, Response

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.ingest import parse_session_date


class TestParseSessionDate:
    """Test the session date parser for common dataset formats."""

    def test_locomo_format_full_month_name(self):
        """LoCoMo: '1:00 pm on 5 May, 2023' → '2023-05-05'"""
        assert parse_session_date("1:00 pm on 5 May, 2023") == "2023-05-05"

    def test_locomo_format_abbreviated_month(self):
        """LoCoMo with abbreviated month: '2:30 pm on 12 Jan, 2022'"""
        assert parse_session_date("2:30 pm on 12 Jan, 2022") == "2022-01-12"

    def test_locomo_format_no_comma(self):
        """LoCoMo without comma: '10:00 am on 3 March 2021'"""
        assert parse_session_date("10:00 am on 3 March 2021") == "2021-03-03"

    def test_longmemeval_format(self):
        """LongMemEval: '2023/05/05 some other text' → '2023-05-05'"""
        assert parse_session_date("2023/05/05 conversation details") == "2023-05-05"

    def test_longmemeval_format_just_date(self):
        """LongMemEval: just the date '2023/12/25'"""
        assert parse_session_date("2023/12/25") == "2023-12-25"

    def test_iso_format(self):
        """Already ISO: '2023-05-05' → '2023-05-05'"""
        assert parse_session_date("2023-05-05") == "2023-05-05"

    def test_iso_format_with_time(self):
        """ISO with time: '2023-05-05T10:00:00' → '2023-05-05'"""
        assert parse_session_date("2023-05-05T10:00:00") == "2023-05-05"

    def test_none_returns_none(self):
        """None input → None output"""
        assert parse_session_date(None) is None

    def test_empty_string_returns_none(self):
        """Empty string → None"""
        assert parse_session_date("") is None
        assert parse_session_date("   ") is None

    def test_unparseable_returns_none(self):
        """Garbage input → None (never raises)"""
        assert parse_session_date("not a date") is None
        assert parse_session_date("sometime last week") is None
        assert parse_session_date("random text") is None

    def test_partial_match_returns_none(self):
        """Malformed LoCoMo-like string that doesn't fully match"""
        assert parse_session_date("on May 2023") is None  # missing day

    def test_zero_padded_day(self):
        """Single-digit day gets zero-padded: '5 May, 2023' → '2023-05-05'"""
        result = parse_session_date("3:00 pm on 5 May, 2023")
        assert result == "2023-05-05"


class TestExtractWriteOccurredAt:
    """Test client.extract_write sends occurred_at when set."""

    def test_extract_write_includes_occurred_at_in_body(self):
        """When occurred_at is provided, it's included in the request body."""
        import asyncio
        import httpx
        import json

        request_captured = None

        def handler(request: Request) -> Response:
            nonlocal request_captured
            request_captured = request
            return Response(200, json={"task_id": "task-123"})

        # Create AsyncClient with MockTransport
        http_client = httpx.AsyncClient(transport=MockTransport(handler), base_url="http://test")
        client = NeuralscapeClient(base_url="http://test", http=http_client)

        asyncio.run(client.extract_write(
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
            occurred_at="2023-05-05",
        ))

        body = json.loads(request_captured.content.decode())
        assert body["occurred_at"] == "2023-05-05"

    def test_extract_write_omits_occurred_at_when_none(self):
        """When occurred_at is None, it's not included in the request body."""
        import asyncio
        import httpx
        import json

        request_captured = None

        def handler(request: Request) -> Response:
            nonlocal request_captured
            request_captured = request
            return Response(200, json={"task_id": "task-456"})

        http_client = httpx.AsyncClient(transport=MockTransport(handler), base_url="http://test")
        client = NeuralscapeClient(base_url="http://test", http=http_client)

        asyncio.run(client.extract_write(
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
            occurred_at=None,
        ))

        body = json.loads(request_captured.content.decode())
        assert "occurred_at" not in body

    def test_extract_write_omits_occurred_at_when_not_passed(self):
        """When occurred_at parameter is omitted, it's not in the body."""
        import asyncio
        import httpx
        import json

        request_captured = None

        def handler(request: Request) -> Response:
            nonlocal request_captured
            request_captured = request
            return Response(200, json={"task_id": "task-789"})

        http_client = httpx.AsyncClient(transport=MockTransport(handler), base_url="http://test")
        client = NeuralscapeClient(base_url="http://test", http=http_client)

        asyncio.run(client.extract_write(
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
        ))

        body = json.loads(request_captured.content.decode())
        assert "occurred_at" not in body
