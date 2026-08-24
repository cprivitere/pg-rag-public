"""Regression tests for the ask() general/comparison path source handling.

ask() unpacks _gap_fill()'s (answer, ids, docs, metas, dists, flag) return
into (documents, metadatas, distances). A swapped arg order once made
gap-filled entries append dicts to the distances list and floats to the
metadatas list, crashing sources with "'float' object has no attribute 'get'".
"""

from pgrag.rag import pipeline


def _retrieve_result(extra_docs=()):
    return {
        "ids": [["skill_Cheesemaking_chunk_0", *extra_docs]],
        "documents": [
            [
                "Skill: Cheesemaking\\n\\nMake cheese by combining milk and rennet.",
                *["Quest: first cheese reward"],
            ]
        ],
        "metadatas": [
            [
                {"name": "Cheesemaking", "table": "skills"},
                {"name": "Quest: First Cheese", "table": "quests"},
            ]
        ],
        "distances": [[0.3, 0.2]],
    }


def _setup_general(monkeypatch, generate):
    monkeypatch.setattr(
        "pgrag.rag.pipeline.classify_query", lambda q: "general"
    )
    monkeypatch.setattr(
        "pgrag.rag.pipeline.should_synthesize", lambda *a, **k: False
    )
    monkeypatch.setattr("pgrag.rag.pipeline.generate", generate)


def test_general_path_sources_well_formed(monkeypatch):
    """No gap-fill: sources must pair each id with a dict metadata and a
    float distance. Catches a broken metas/dists arg-order swap."""
    _setup_general(
        monkeypatch,
        lambda p: "Make cheese by combining milk and rennet to level up.",
    )
    monkeypatch.setattr(
        "pgrag.rag.pipeline.retrieve",
        lambda *a, **k: _retrieve_result(),
    )

    result = pipeline.ask("how do I level Cheesemaking")

    assert result["query_type"] == "general"
    sources = result["sources"]
    assert len(sources) == 1
    assert sources[0]["metadata"]["name"] == "Cheesemaking"
    assert isinstance(sources[0]["distance"], float)
    assert "skills" in sources[0]["citation"]


def test_general_gap_fill_keeps_sources_well_formed(monkeypatch):
    """Gap-fill appends a re-retrieved doc: its metadata/distance must land
    in the right lists. This is the case that crashed with the swapped
    arg order ('float' object has no attribute 'get')."""
    answers = iter(
        [
            "I do not know how to level Cheesemaking.",
            "Do the Quest: First Cheese to learn cheese making.",
        ]
    )

    def fake_generate(prompt):
        return next(answers)

    calls = []

    def fake_retrieve(question, *a, **k):
        calls.append(question)
        if len(calls) == 1:
            return _retrieve_result()
        return {
            "ids": [["quest_quest_1_chunk_1"]],
            "documents": [["Quest: First Cheese rewards you with milk knowledge"]],
            "metadatas": [[{"name": "Quest: First Cheese", "table": "quests"}]],
            "distances": [[0.15]],
        }

    _setup_general(monkeypatch, fake_generate)
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)

    result = pipeline.ask("how do I level Cheesemaking", allow_gap_fill=True)

    sources = result["sources"]
    # original doc + one gap-filled doc, both well typed
    assert len(sources) == 2
    for s in sources:
        assert isinstance(s["metadata"], dict), s
        assert isinstance(s["distance"], float), s
    assert sources[-1]["id"] == "quest_quest_1_chunk_1"
    assert sources[-1]["metadata"]["table"] == "quests"


def test_general_plan_propagates_native_and_token_filters(monkeypatch):
    """A high-confidence plan splits into Chroma-where (native) + post-fusion
    (token) filters and is recorded in the trace."""
    seen = {}

    def fake_retrieve(question, metadata_filter=None, token_filter=None,
                      **kwargs):
        seen["mf"] = metadata_filter
        seen["tf"] = token_filter
        return _retrieve_result()

    _setup_general(
        monkeypatch,
        lambda p: "You mix milk with mushrooms into a cheese.",
    )
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)

    trace = {}
    pipeline.ask("recipes using mushrooms", trace=trace)

    assert seen["mf"] == {"type": "recipe"}
    assert seen["tf"] == {"ingredients": "Mushroom"}
    assert trace["plan"]["label"] == "recipe ingredient=Mushroom"


def test_general_user_filter_beats_plan(monkeypatch):
    """A caller-supplied metadata_filter must not be overridden by a plan."""
    seen = {}

    def fake_retrieve(question, metadata_filter=None, token_filter=None,
                      **kwargs):
        seen["mf"] = metadata_filter
        seen["tf"] = token_filter
        return _retrieve_result()

    _setup_general(
        monkeypatch,
        lambda p: "Alchemy answer",
    )
    monkeypatch.setattr("pgrag.rag.pipeline.retrieve", fake_retrieve)

    pipeline.ask("alchemy recipes", metadata_filter={"type": "recipe"})

    assert seen["mf"] == {"type": "recipe"}
    assert seen["tf"] is None