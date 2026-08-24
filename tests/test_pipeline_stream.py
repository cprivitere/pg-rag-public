"""Tests for the streaming pipeline variant (ask_stream)."""

from pgrag.rag import pipeline

HUB_CTX = {
    "ids": [["skillprofile_Pooping_chunk_0", "skillprofile_Pooping_chunk_1"]],
    "documents": [
        [
            "Skill Profile: Dungcrafting\n\nDescription: creates fertilizer-grade poop",
            "XP Table: 10 to 500\n\nQuests: Graffiti Mastering Poop",
        ]
    ],
    "metadatas": [
        [
            {"name": "Dungcrafting", "table": "skills", "type": "skillprofile"},
            {"name": "Dungcrafting", "table": "skills", "type": "skillprofile"},
        ]
    ],
    "distances": [[0.0, 0.0]],
}


def _set_entity(monkeypatch, ctx=HUB_CTX):
    monkeypatch.setattr(
        "pgrag.rag.pipeline.classify_query", lambda q: "entity"
    )
    monkeypatch.setattr(
        "pgrag.rag.pipeline.find_entity",
        lambda q: ("skillprofile_Pooping", "skill"),
    )
    monkeypatch.setattr(
        "pgrag.rag.pipeline.build_entity_context", lambda q, h, include_leveling=False: ctx
    )


def test_ask_stream_entity_streams_tokens(monkeypatch):
    _set_entity(monkeypatch)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.stream_generate",
        lambda p: iter(["Dung", "crafting", " is", " fine."]),
    )
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", lambda *a, **k: None)

    events = list(pipeline.ask_stream("what is Dungcrafting"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    final = events[-1]

    assert tokens == "Dungcrafting is fine."
    assert final["type"] == "final"
    assert final["result"]["query_type"] == "entity"
    assert final["result"]["answer"] == "Dungcrafting is fine."
    assert final["result"]["sources"][0]["id"] == "skillprofile_Pooping_chunk_0"


def test_ask_stream_resets_on_gap_fill(monkeypatch):
    _set_entity(monkeypatch)
    answers = iter(
        [
            "I do not know how to learn this skill.",
            "Complete the Graffiti quest to learn it.",
        ]
    )

    def fake_stream(prompt):
        return [next(answers)]

    monkeypatch.setattr("pgrag.rag.pipeline.stream_generate", fake_stream)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda *a, **k: {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["Quest: Graffiti Mastering Poop rewards the ability"]],
            "metadatas": [[{"name": "Graffiti: Mastering 'Poop'", "table": "quests"}]],
            "distances": [[0.2]],
        },
    )

    events = list(pipeline.ask_stream("what is Dungcrafting", allow_gap_fill=True))
    types = [e["type"] for e in events]

    assert types.count("reset") == 1
    assert types[-1] == "final"
    final = events[-1]["result"]
    assert final["answer"] == "Complete the Graffiti quest to learn it."
    assert final["sources"][-1]["id"] == "quest_quest_197_chunk_1"