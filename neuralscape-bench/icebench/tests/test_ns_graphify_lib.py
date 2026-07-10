"""Tests for NS-Graphify-Lib adapter."""

import json
import pytest
from unittest.mock import Mock, patch

from icebench.adapters.base import Corpus, UnsupportedOp
from icebench.adapters.ns_graphify_lib import NSGraphifyLibAdapter


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
    """NS-Graphify-Lib adapter fixture."""
    return NSGraphifyLibAdapter(api_url="http://localhost:8699")


def test_capabilities(adapter):
    """Test that ns-graphify-lib declares correct capabilities."""
    caps = adapter.capabilities()
    assert caps == {"symbol_lookup", "neighbors_1hop", "path_le4", "blast_radius"}

    # Ensure excluded op is not present
    assert "nl_locate" not in caps


def test_make_code_space(adapter):
    """Test code_space generation."""
    code_space = adapter._make_code_space("test-repo")
    assert code_space == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_index_cold_success(mock_client_class, adapter, corpus):
    """Test successful cold index."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    # Mock POST response (202 with task_id)
    post_resp = Mock()
    post_resp.status_code = 202
    post_resp.json.return_value = {"task_id": "task-456", "code_space": "code--ice-bench--test-repo"}
    mock_client.post.return_value = post_resp

    # Mock poll response (200 completed)
    poll_resp = Mock()
    poll_resp.status_code = 200
    poll_resp.json.return_value = {
        "status": "completed",
        "result": {
            "symbols_indexed": 200,
            "edges_indexed": 150,
            "files_indexed": 20,
        }
    }
    mock_client.get.return_value = poll_resp

    result = adapter.index_cold(corpus)

    assert result.ok
    assert result.symbols == 200
    assert result.edges == 150
    assert result.files == 20
    assert result.peak_rss_mb == 0  # Server-side
    assert result.cpu_s == 0  # Server-side
    assert not result.dnf

    # Verify POST was called with correct params
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/v1/code-graph/index" in call_args[0][0]
    posted_json = call_args[1]["json"]
    assert posted_json["repo_source"] == corpus.path
    assert posted_json["system"] == "code-graphify-lib"
    assert posted_json["code_space"] == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_index_cold_failed(mock_client_class, adapter, corpus):
    """Test index that fails during processing."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    # Mock POST response
    post_resp = Mock()
    post_resp.status_code = 202
    post_resp.json.return_value = {"task_id": "task-fail"}
    mock_client.post.return_value = post_resp

    # Mock poll response (failed status)
    poll_resp = Mock()
    poll_resp.status_code = 200
    poll_resp.json.return_value = {"status": "failed"}
    mock_client.get.return_value = poll_resp

    result = adapter.index_cold(corpus)

    assert not result.ok
    assert result.dnf
    assert result.symbols == 0


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_query_symbol_lookup(mock_client_class, adapter, corpus):
    """Test symbol_lookup query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    # Mock GET response
    resp = Mock()
    resp.status_code = 200
    resp.text = "symbol result"
    mock_client.get.return_value = resp

    result = adapter.query("symbol_lookup", {"symbol": "MyClass", "corpus": corpus})

    assert result.ok
    assert result.answer["status"] == "ok"
    assert result.answer["text"] == "symbol result"

    # Verify GET was called with correct params
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    assert "/v1/code-graph/query" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["question"] == "MyClass"
    assert params["knowledge_system"] == "code-graphify-lib"
    assert params["graph_id"] == "code--ice-bench--test-repo"


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_query_neighbors_1hop(mock_client_class, adapter, corpus):
    """Test neighbors_1hop query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 200
    resp.text = "neighbors result"
    mock_client.get.return_value = resp

    result = adapter.query("neighbors_1hop", {"symbol": "MyClass", "corpus": corpus})

    assert result.ok

    call_args = mock_client.get.call_args
    assert "/v1/code-graph/neighbors" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["label"] == "MyClass"
    assert params["knowledge_system"] == "code-graphify-lib"


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_query_path_le4(mock_client_class, adapter, corpus):
    """Test path_le4 query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 200
    resp.text = "path result"
    mock_client.get.return_value = resp

    result = adapter.query("path_le4", {"from": "A", "to": "B", "corpus": corpus})

    assert result.ok

    call_args = mock_client.get.call_args
    assert "/v1/code-graph/path" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["source"] == "A"
    assert params["target"] == "B"
    assert params["max_hops"] == 4
    assert params["knowledge_system"] == "code-graphify-lib"


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_query_blast_radius(mock_client_class, adapter, corpus):
    """Test blast_radius query."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 200
    resp.text = "impact result"
    mock_client.get.return_value = resp

    result = adapter.query("blast_radius", {"symbol": "MyClass", "max_hops": 3, "corpus": corpus})

    assert result.ok

    call_args = mock_client.get.call_args
    assert "/v1/code-graph/impact" in call_args[0][0]
    params = call_args[1]["params"]
    assert params["symbol"] == "MyClass"
    assert params["max_hops"] == 3
    assert params["knowledge_system"] == "code-graphify-lib"


def test_query_unsupported_op(adapter, corpus):
    """Test that nl_locate raises UnsupportedOp."""
    with pytest.raises(UnsupportedOp):
        adapter.query("nl_locate", {"query": "find auth", "corpus": corpus})


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_query_error_response(mock_client_class, adapter, corpus):
    """Test query with non-200 response."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    resp = Mock()
    resp.status_code = 500
    resp.text = "Internal Server Error"
    mock_client.get.return_value = resp

    result = adapter.query("symbol_lookup", {"symbol": "Test", "corpus": corpus})

    assert not result.ok
    assert result.answer["status"] == "error"
    assert "Internal Server Error" in result.answer["error"]


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


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
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


@patch("icebench.adapters.ns_graphify_lib.httpx.Client")
def test_index_incremental(mock_client_class, adapter, corpus):
    """Test incremental index falls back to cold index."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    adapter.client = mock_client

    post_resp = Mock()
    post_resp.status_code = 202
    post_resp.json.return_value = {"task_id": "task-inc"}
    mock_client.post.return_value = post_resp

    poll_resp = Mock()
    poll_resp.status_code = 200
    poll_resp.json.return_value = {
        "status": "completed",
        "result": {"symbols_indexed": 100, "edges_indexed": 50, "files_indexed": 5}
    }
    mock_client.get.return_value = poll_resp

    result = adapter.index_incremental(corpus, ["file.py"])

    assert result.ok
    # Should still call the full index endpoint
    mock_client.post.assert_called_once()
