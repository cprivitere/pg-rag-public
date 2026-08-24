"""Tests the entity-specific path of pgrag.rag.pipeline.ask.

Covers the skill-profile hub overriding the general corpus, bypassing
synthesis, fallback to the general corpus on a hub miss, and the gap-fill
re-retrieval loop firing at most once with an empty-subject/answer fallback.
"""

import pytest

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
        "pgrag.rag.pipeline.find_entity", lambda q: ("skillprofile_Pooping", "skill")
    )
    monkeypatch.setattr(
        "pgrag.rag.pipeline.build_entity_context", lambda q, h, include_leveling=False: ctx
    )


def test_entity_path_used(monkeypatch):
    _set_entity(monkeypatch)
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return "Dungcrafting is an animal-specific skill for fertilizer-grade poop."

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", lambda *a, **k: None)

    result = pipeline.ask("what is the Dungcrafting skill")

    assert result["query_type"] == "entity"
    assert "Skill Profile: Dungcrafting" in prompts[0]
    assert "XP Table: 10 to 500" in prompts[0]
    assert result["sources"][0]["id"] == "skillprofile_Pooping_chunk_0"


def test_entity_bypasses_synthesis(monkeypatch):
    _set_entity(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("synthesis must not run on entity path")

    monkeypatch.setattr("pgrag.rag.pipeline.should_synthesize", boom)
    monkeypatch.setattr("pgrag.rag.pipeline.generate", lambda p: "answer")
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", lambda *a, **k: None)

    result = pipeline.ask("what is Dungcrafting")
    assert result["answer"] == "answer"


def test_hub_miss_falls_back_general(monkeypatch):
    _set_entity(monkeypatch, ctx=None)
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return "general answer"

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda question, metadata_filter=None, token_filter=None, query_type="general", count=20, hybrid=True, trace=None: {
            "ids": [["skill_Pooping"]],
            "documents": [["Skill: Dungcrafting description"]],
            "metadatas": [[{"name": "Dungcrafting", "table": "skills"}]],
            "distances": [[0.5]],
        },
    )

    result = pipeline.ask("what is the Dungcrafting skill")
    assert result["query_type"] == "general"
    assert "Skill: Dungcrafting" in prompts[0]


def test_gap_fill_fires_once(monkeypatch):
    _set_entity(monkeypatch)
    answers = iter(
        [
            "I do not know how to learn this skill.",
            "Complete Graffiti: Mastering 'Poop' to learn it.",
        ]
    )
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return next(answers)

    def fake_retrieve(question, count=5, hybrid=True, rerank=True, trace=None, **kwargs):
        assert "learn" in question or "how to learn" in question
        return {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["Quest: Graffiti Mastering Poop rewards the ability"]],
            "metadatas": [[{"name": "Graffiti: Mastering 'Poop'", "table": "quests"}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)

    result = pipeline.ask("what is the Dungcrafting skill", allow_gap_fill=True)

    assert len(prompts) == 2
    assert "Quest: Graffiti Mastering Poop" in prompts[1]
    assert result["answer"] == "Complete Graffiti: Mastering 'Poop' to learn it."
    assert result["sources"][-1]["id"] == "quest_quest_197_chunk_1"


def test_gap_fill_empty_subject_falls_back_to_question(monkeypatch):
    _set_entity(monkeypatch)
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "I do not know anything about that."

    retrieved = []

    def fake_retrieve(question, count=5, hybrid=True, rerank=True, trace=None, **kwargs):
        retrieved.append(question)
        return {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["Quest: Graffiti Mastering Poop"]],
            "metadatas": [[{"name": "Graffiti: Mastering 'Poop'", "table": "quests"}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)

    result = pipeline.ask("what is Dungcrafting", allow_gap_fill=True)
    assert len(retrieved) == 1
    assert "what is dungcrafting" in retrieved[0].strip().lower()
    assert len(calls) == 2
    assert result["answer"] == "I do not know anything about that."


def test_gap_fill_max_one_loop(monkeypatch):
    _set_entity(monkeypatch)
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "I do not know how to learn this skill."

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda *a, **k: {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["quest text"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        },
    )

    pipeline.ask("what is Dungcrafting", allow_gap_fill=True)
    assert len(calls) == 2


def test_gap_fill_empty_answer_retries_without_retrieve(monkeypatch):
    _set_entity(monkeypatch)
    queue = ["", "Dungcrafting is learned via the Graffiti quest."]
    calls = []
    retrieved = []

    def fake_generate(prompt):
        calls.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda *a, **k: retrieved.append(a) or {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["quest text"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        },
    )

    result = pipeline.ask("what is Dungcrafting", allow_gap_fill=True)

    assert len(calls) == 2
    assert len(retrieved) == 0
    assert result["answer"] == "Dungcrafting is learned via the Graffiti quest."


def test_gap_fill_empty_answer_then_retrieve(monkeypatch):
    _set_entity(monkeypatch)
    queue = ["", "", "Learned from the Graffiti quest."]
    calls = []
    retrieved = []

    def fake_generate(prompt):
        calls.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr("pgrag.rag.pipeline.generate", fake_generate)
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda *a, **k: retrieved.append(a) or {
            "ids": [["quest_quest_197_chunk_1"]],
            "documents": [["quest text"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        },
    )

    result = pipeline.ask("what is Dungcrafting", allow_gap_fill=True)

    assert len(calls) == 3
    assert len(retrieved) == 1
    assert result["answer"] == "Learned from the Graffiti quest."
