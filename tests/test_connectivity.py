"""Tests server-error handling for pgrag.embeddings.llama_embeddings.embed_batch
and pgrag.rag.llm.generate: unreachable/timeout servers raise EmbeddingServerError
/LLMServerError with the endpoint in the message, and HTTP errors propagate."""

from unittest.mock import patch

import pytest
import requests

from pgrag.embeddings.llama_embeddings import EMBEDDING_URL, EmbeddingServerError, embed_batch
from pgrag.rag.llm import LLM_URL, LLMServerError, generate


def test_embedding_server_unreachable():
    with patch("pgrag.embeddings.llama_embeddings.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(EmbeddingServerError) as exc:
            embed_batch(["test"])
        msg = str(exc.value)
        assert "8081" in msg
        assert EMBEDDING_URL in msg


def test_embedding_server_timeout():
    with patch("pgrag.embeddings.llama_embeddings.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(EmbeddingServerError) as exc:
            embed_batch(["test"])
        msg = str(exc.value)
        assert "timed out" in msg.lower()
        assert EMBEDDING_URL in msg


def test_embedding_server_http_error_still_raised():
    with patch("pgrag.embeddings.llama_embeddings.requests.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with pytest.raises(requests.exceptions.HTTPError):
            embed_batch(["test"])


def test_llm_server_unreachable():
    with patch("pgrag.rag.llm.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(LLMServerError) as exc:
            generate("test prompt")
        msg = str(exc.value)
        assert "8080" in msg
        assert LLM_URL in msg


def test_llm_server_timeout():
    with patch("pgrag.rag.llm.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout()
        with pytest.raises(LLMServerError) as exc:
            generate("test prompt")
        msg = str(exc.value)
        assert "timed out" in msg.lower()
        assert LLM_URL in msg


def test_llm_server_http_error_still_raised():
    with patch("pgrag.rag.llm.requests.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with pytest.raises(requests.exceptions.HTTPError):
            generate("test prompt")
