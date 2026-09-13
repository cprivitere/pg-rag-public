"""Comparison routing: a question naming 2+ entities must reach the
multi-entity context, so the answer sees both entities' own facts."""

from pgrag.rag import pipeline


def test_two_entity_comparison_uses_multi_entity_context(monkeypatch):
    """ask() routes comparison with >=2 entities through build_multi_entity_context
    (local-imported from entity_retrieval), so both entities' blocks reach the
    generation prompt."""
    captured = {}

    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "comparison")
    monkeypatch.setattr(
        "pgrag.rag.pipeline.find_entities",
        lambda q: [
            ("Punch", "ability_punch", "ability"),
            ("Front Kick", "ability_front_kick", "ability"),
        ],
    )

    def fake_multi(question, entities, trace=None):
        return {
            "ids": [["ability_punch", "ability_front_kick", "recipe_shared"]],
            "documents": [
                [
                    "=== Punch (ability) ===",
                    "Punch does 6 damage.",
                    "=== Front Kick (ability) ===",
                    "Front Kick does 11 damage.",
                ]
            ],
            "metadatas": [[]],
            "distances": [[0.0, 0.0, 0.0]],
            "rerank_used": False,
        }

    monkeypatch.setattr("pgrag.rag.entity_retrieval.build_multi_entity_context", fake_multi)

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Punch deals 6, Front Kick deals 11."

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)

    result = pipeline.ask("which deals more damage, Punch or Front Kick")

    assert result["query_type"] == "comparison"
    assert result["answer"] == "Punch deals 6, Front Kick deals 11."
    assert "=== Punch (ability) ===" in captured["prompt"]
    assert "=== Front Kick (ability) ===" in captured["prompt"]
    assert "Punch does 6 damage." in captured["prompt"]
    assert "Front Kick does 11 damage." in captured["prompt"]


def test_single_entity_comparison_stays_general_path(monkeypatch):
    """A comparison naming only one entity must NOT use the multi-entity path."""
    calls = []

    monkeypatch.setattr("pgrag.rag.pipeline.classify_query", lambda q: "comparison")
    monkeypatch.setattr(
        "pgrag.rag.pipeline.find_entities",
        lambda q: [("Punch", "ability_punch", "ability")],
    )
    monkeypatch.setattr(
        "pgrag.rag.entity_retrieval.build_multi_entity_context",
        lambda *a, **k: calls.append("called"),
    )

    def fake_retrieve(question, **kwargs):
        return {
            "ids": [["s1"]],
            "documents": [["Snippet."]],
            "metadatas": [[{"name": "N"}]],
            "distances": [[0.3]],
            "rerank_used": False,
        }

    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("pgrag.rag.pipeline._find_matching_summary", lambda *a, **k: None)
    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", lambda *a, **k: False)
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda prompt, **k: "answer")

    pipeline.ask("how does Punch compare to the best damage ability?")
    assert calls == []
