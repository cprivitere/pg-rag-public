"""Tests the pgrag.rag.reranker_client.rerank_documents contract: truncation of
long queries/documents, document-count capping, payload shape, response parsing
and sorting, and RerankError on server errors or malformed payloads."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from pgrag.rag import reranker_client


class _FakeResponse:
    def __init__(self, ok=True, payload=None):
        self._ok = ok
        self._payload = payload or {
            "results": [{"index": 0, "relevance_score": 1.0}],
        }

    def raise_for_status(self):
        if not self._ok:
            raise requests.exceptions.HTTPError("500")

    def json(self):
        return self._payload


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_truncates_long_documents(mock_post):
    mock_post.return_value = _FakeResponse()
    long_doc = "x" * (reranker_client.MAX_RERANK_DOC_CHARS + 5000)
    reranker_client.rerank_documents("q", [long_doc], 1)

    body = mock_post.call_args[1]["json"]
    assert len(body["documents"][0]) == reranker_client.MAX_RERANK_DOC_CHARS


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_truncates_long_query(mock_post):
    """Pathological queries are truncated too — the doc cap alone must not
    let a giant query exceed the server's batch ceiling."""
    mock_post.return_value = _FakeResponse()
    long_query = "q" * (reranker_client.MAX_RERANK_QUERY_CHARS + 5000)
    reranker_client.rerank_documents(long_query, ["doc"], 1)

    body = mock_post.call_args[1]["json"]
    assert len(body["query"]) == reranker_client.MAX_RERANK_QUERY_CHARS


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_oversized_doc_stays_under_server_batch(mock_post):
    """Regression: llama.cpp 500s with 'input is too large to process' when a
    query+doc pair exceeds the server batch. The client must truncate oversized
    chunks so a pathological doc can never trigger that error again."""
    mock_post.return_value = _FakeResponse()
    oversized = "x" * (reranker_client.MAX_RERANK_DOC_CHARS + 10000)
    assert len(oversized) > reranker_client.MAX_RERANK_DOC_CHARS
    reranker_client.rerank_documents("q", [oversized], 1)

    sent = mock_post.call_args[1]["json"]["documents"][0]
    assert len(sent) == reranker_client.MAX_RERANK_DOC_CHARS
    # Worst-case chars->tokens (~4 chars/token) stays under the 8192-token
    # batch the rerank server is started with (see .mise/tasks/rerank-start.ps1).
    assert len(sent) // 4 < 8192


def test_rerank_truncation_bound_keeps_query_plus_doc_under_batch():
    long_query = "q" * 200
    worst = reranker_client.MAX_RERANK_DOC_CHARS // 4 + len(long_query) // 4
    assert worst < 8192


def test_rerank_start_script_raises_server_batch():
    """The server fix lives in the start script: -b/-ub must exceed the 512
    default that caused the 'input too large' 500."""
    path = Path(__file__).parents[1] / ".mise" / "tasks" / "rerank-start.ps1"
    text = path.read_text(encoding="utf-8")
    assert "'-b'" in text or '"-b"' in text
    assert "'-ub'" in text or '"-ub"' in text
    assert "8192" in text


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_caps_document_count(mock_post):
    mock_post.return_value = _FakeResponse()
    docs = ["doc"] * (reranker_client.MAX_RERANK_DOCS + 50)
    reranker_client.rerank_documents("q", docs, reranker_client.MAX_RERANK_DOCS)

    body = mock_post.call_args[1]["json"]
    assert len(body["documents"]) == reranker_client.MAX_RERANK_DOCS


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_payload_shape(mock_post):
    mock_post.return_value = _FakeResponse()
    reranker_client.rerank_documents("cheese query", ["cheddar", "gouda"], 2)

    body = mock_post.call_args[1]["json"]
    assert body["query"] == "cheese query"
    assert body["documents"] == ["cheddar", "gouda"]
    assert body["top_n"] == 2


def test_rerank_empty_documents_no_request():
    assert reranker_client.rerank_documents("q", [], 5) == []


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_parses_sorted_indices(mock_post):
    mock_post.return_value = _FakeResponse(
        payload={
            "results": [
                {"index": 2, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
    )
    indices = reranker_client.rerank_documents("q", ["a", "b", "c"], 3)
    assert indices == [0, 1, 2]


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_server_error_raises(mock_post):
    mock_post.return_value = _FakeResponse(ok=False)
    with pytest.raises(reranker_client.RerankError):
        reranker_client.rerank_documents("q", ["a"], 1)


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_bad_shape_raises(mock_post):
    mock_post.return_value = _FakeResponse(payload={"results": []})
    with pytest.raises(reranker_client.RerankError):
        reranker_client.rerank_documents("q", ["a"], 1)


@patch("pgrag.rag.reranker_client.requests.post")
def test_rerank_magic_mock_requests_are_ignored(mock_post):
    """Ensure the requests.post mock path doesn't short-circuit parsing."""
    mock_post.return_value = MagicMock()
    mock_post.return_value.raise_for_status.return_value = None
    mock_post.return_value.json.return_value = {"results": [{"index": 0, "relevance_score": 1.0}]}
    indices = reranker_client.rerank_documents("q", ["a"], 1)
    assert indices == [0]
