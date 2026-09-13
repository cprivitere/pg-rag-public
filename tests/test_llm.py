"""Tests for the LLM client's streaming (SSE) parsing."""

import pytest

from pgrag.rag import llm


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self._chunks


def test_stream_generate_parses_sse_deltas(monkeypatch):
    seen = {}

    def fake_post(prompt, stream, **kwargs):
        seen["stream"] = stream
        chunks = [
            'data: {"choices": [{"delta": {"content": "Hel"}}]}',
            "event: keepalive",
            'data: {"choices": [{"delta": {"content": "lo"}}]}',
            'data: {"choices": [{"delta": {}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "IGNORED"}}]}',
        ]
        return _FakeResponse(chunks)

    monkeypatch.setattr(llm, "_post", fake_post)

    assert list(llm.stream_generate("prompt")) == ["Hel", "lo"]
    assert seen["stream"] is True


def test_stream_generate_raises_server_error(monkeypatch):
    import requests

    def boom(prompt, stream, **kwargs):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(llm, "_post", boom)

    with pytest.raises(llm.LLMServerError):
        list(llm.stream_generate("prompt"))


def test_post_emits_temperature_and_optional_seed(monkeypatch):
    """_post sends temperature; seed only when provided (None -> omitted)."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        return _FakeResponse([])

    monkeypatch.setattr(llm.requests, "post", fake_post)

    llm._post("q", stream=False, temperature=0, seed=42)
    assert captured["json"]["temperature"] == 0
    assert captured["json"]["seed"] == 42

    llm._post("q", stream=False)
    assert captured["json"]["temperature"] == 0.2
    assert "seed" not in captured["json"]
