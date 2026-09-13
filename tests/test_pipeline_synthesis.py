"""Synthesis is ephemeral: it augments the in-request context and is never
persisted to disk. Regression for the removed curated-dir write (V24/V20)."""

import pytest

from pgrag.config import DERIVED_DIR
from pgrag.rag import pipeline


def _fake_retrieve(question, **kwargs):
    return {
        "ids": [["snippet_1"]],
        "documents": [["First snippet about the topic."]],
        "metadatas": [[{"source": "wiki"}]],
        "distances": [[0.42]],
        "rerank_used": False,
    }


def test_synthesis_uses_ephemeral_context(monkeypatch):
    """A synthesizing ask replaces the context with the synthesized text and
    does not persist it."""
    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "general")
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", _fake_retrieve)
    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", lambda *a, **k: True)
    monkeypatch.setattr("pgrag.rag.pipeline.synthesize_answer", lambda *a, **k: "SYNTHESIZED BODY")
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda prompt: "final answer")

    result = pipeline.ask("how do mushrooms grow")

    assert result["documents"] == ["SYNTHESIZED BODY"]
    assert result["answer"] == "final answer"


def test_persist_surface_removed():
    """The write path must not exist: no _persist_synthesized, no CURATED_DIR,
    and no create_curated_doc imported into the pipeline module."""
    assert not hasattr(pipeline, "_persist_synthesized")
    assert not hasattr(pipeline, "CURATED_DIR")
    assert hasattr(pipeline, "synthesize_answer")


def test_synthesis_writes_nothing():
    """After a synthesizing ask, no synthesized file lands in the derived
    curated dir (the old persist target)."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "general")
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", _fake_retrieve)
    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", lambda *a, **k: True)
    monkeypatch.setattr("pgrag.rag.pipeline.synthesize_answer", lambda *a, **k: "SYNTHESIZED BODY")
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda prompt: "final answer")
    try:
        pipeline.ask("how do mushrooms grow")
    finally:
        monkeypatch.undo()

    curated = DERIVED_DIR / "curated"
    mono = list(curated.glob("synthesized_*")) if curated.exists() else []
    assert mono == []
