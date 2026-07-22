"""Tests for NS-CBM adapter."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from icebench.adapters.base import Corpus, UnsupportedOp
from icebench.adapters.ns_cbm import NSCbmAdapter


@pytest.fixture
def corpus():
    """Test corpus fixture."""
    return Corpus(
        name="test-repo",
        path="/tmp/test-repo",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )


@pytest.fixture
def adapter():
    """NS-CBM adapter fixture."""
    return NSCbmAdapter(api_url="http://localhost:8699")


def test_capabilities(adapter):
    """Test that ns-cbm declares correct capabilities."""
    caps = adapter.capabilities()
    assert caps == {"symbol_lookup", "neighbors_1hop", "nl_locate"}

    # Ensure excluded ops are not present
    assert "path_le4" not in caps
    assert "blast_radius" not in caps


def test_make_code_space(adapter):
    """Test code_space generation."""
    code_space = adapter._make_code_space("test-repo")
    assert code_space == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_index_cold_success(mock_client_class, adapter, corpus):
    """Test successful cold index."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    # Mock POST response (202 with task_id)
    post_resp = Mock()
    post_resp.status_code = 202
    post_resp.json.return_value = {"task_id": "task-123", "code_space": "code--ice-bench--test-repo"}
    mock_client.post.return_value = post_resp

    # Mock poll response (200 completed)
    poll_resp = Mock()
    poll_resp.status_code = 200
    poll_resp.json.return_value = {
        "status": "completed",
        "result": {
            "symbols_indexed": 100,
            "edges_indexed": 50,
            "files_indexed": 10,
        }
    }
    mock_client.get.return_value = poll_resp

    result = adapter.index_cold(corpus)

    assert result.ok
    assert result.symbols == 100
    assert result.edges == 50
    assert result.files == 10
    assert result.peak_rss_mb == 0  # Server-side
    assert result.cpu_s == 0  # Server-side
    assert not result.dnf

    # Verify POST was called with correct params
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/v1/code-graph/index" in call_args[0][0]
    posted_json = call_args[1]["json"]
    assert posted_json["repo_source"] == corpus.path
    assert posted_json["system"] == "code-cbm"
    assert posted_json["code_space"] == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_index_cold_timeout(mock_client_class, adapter, corpus):
    """Test index poll timeout (DNF)."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client
    adapter.index_timeout_s = 0.1  # Very short timeout
    adapter.poll_sleep_s = 0.05

    # Mock POST response
    post_resp = Mock()
    post_resp.status_code = 202
    post_resp.json.return_value = {"task_id": "task-123"}
    mock_client.post.return_value = post_resp

    # Mock poll response (always processing)
    poll_resp = Mock()
    poll_resp.status_code = 200
    poll_resp.json.return_value = {"status": "processing"}
    mock_client.get.return_value = poll_resp

    result = adapter.index_cold(corpus)

    assert not result.ok
    assert result.dnf
    assert "timeout" in result.dnf_reason.lower()
    assert result.symbols == 0


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_query_symbol_lookup(mock_client_class, adapter, corpus):
    """Test symbol_lookup query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    # Mock GET response
    resp = Mock()
    resp.status_code = 200
    resp.text = "symbol found"
    mock_client.get.return_value = resp

    result = adapter.query("symbol_lookup", {"symbol": "TestClass", "corpus": corpus})

    assert result.ok
    assert result.answer["status"] == "ok"
    assert result.answer["text"] == "symbol found"
    assert result.latency_ms > 0

    # Verify GET was called with correct params
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert "/v1/code-graph/query" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["question"] == "TestClass"
    assert params["knowledge_system"] == "code-cbm"
    assert params["graph_id"] == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_query_neighbors_1hop(mock_client_class, adapter, corpus):
    """Test neighbors_1hop query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 200
    resp.text = "neighbors data"
    mock_client.get.return_value = resp

    result = adapter.query("neighbors_1hop", {"symbol": "TestClass", "corpus": corpus})

    assert result.ok
    assert result.answer["status"] == "ok"

    call_args = mock_client.get.call_args
    assert "/v1/code-graph/neighbors" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["label"] == "TestClass"
    assert params["knowledge_system"] == "code-cbm"


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_query_nl_locate(mock_client_class, adapter, corpus):
    """Test nl_locate query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 200
    resp.text = "located symbols"
    mock_client.get.return_value = resp

    result = adapter.query("nl_locate", {"query": "find authentication", "corpus": corpus})

    assert result.ok

    call_args = mock_client.get.call_args
    assert "/v1/code-graph/locate" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["query"] == "find authentication"
    assert params["knowledge_system"] == "code-cbm"


def test_query_unsupported_op(adapter, corpus):
    """Test that unsupported ops raise UnsupportedOp."""
    with pytest.raises(UnsupportedOp):
        adapter.query("path_le4", {"from": "A", "to": "B", "corpus": corpus})

    with pytest.raises(UnsupportedOp):
        adapter.query("blast_radius", {"symbol": "TestClass", "corpus": corpus})


def test_store_size_bytes(adapter, corpus):
    """Test store_size_bytes returns 0 (N/A for through-REST)."""
    size = adapter.store_size_bytes(corpus)
    assert size == 0


def test_export_snapshot_na(adapter, corpus):
    """Test snapshot export returns None (N/A)."""
    result = adapter.export_snapshot(corpus)
    assert result is None


def test_import_snapshot_na(adapter, corpus):
    """Test snapshot import returns None (N/A)."""
    result = adapter.import_snapshot(corpus, "/tmp/snapshot.bin")
    assert result is None


@patch("icebench.adapters.ns_cbm.httpx.Client")
def test_teardown(mock_client_class, adapter, corpus):
    """Test teardown deletes the code_space."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    adapter.teardown(corpus)

    # Verify DELETE was called
    mock_client.delete.assert_called_once()
    call_args = mock_client.delete.call_args
    assert "/v1/code-graph/graph" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["graph_id"] == "code--ice-bench--test-repo"
