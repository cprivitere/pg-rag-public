"""Tests the lookup query path of pgrag.rag.pipeline.ask.

Covers a lookup-classified query passing count=20 and hybrid=True through to
retrieve and returning the generated answer.
"""

from pgrag.rag import pipeline


def _lookup_result():
    return {
        "ids": [["skill_Pooping_chunk_0"]],
        "documents": [["Skill: Dungcrafting level requirements"]],
        "metadatas": [[{"name": "Dungcrafting", "table": "skills"}]],
        "distances": [[0.3]],
    }


def test_lookup_query_passes_valid_count(monkeypatch):
    captured = {}

    def fake_retrieve(
        question,
        metadata_filter=None,
        token_filter=None,
        query_type=None,
        count=None,
        hybrid=None,
        trace=None,
    ):
        captured["count"] = count
        captured["hybrid"] = hybrid
        return _lookup_result()

    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "lookup")
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", lambda *a, **k: False)
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda p: "Level 5.")

    result = pipeline.ask("what level is Dungcrafting")

    assert result["query_type"] == "lookup"
    assert result["answer"] == "Level 5."
    assert captured["count"] == 40
    assert captured["hybrid"] is True
