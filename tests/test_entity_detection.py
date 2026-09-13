"""Tests pgrag.rag.query_classifier entity detection: classify_query routing
to entity/general/comparison, find_entity (longest-name wins, word boundary,
plural append, typo fallback, real alias/lorebook index), and find_entities
greedy multi-entity resolution in query order."""

from pathlib import Path

import pytest

from pgrag.rag.query_classifier import classify_query, find_entities, find_entity

SYNTHETIC_INDEX = [
    ("Dungcrafting", "skill_Pooping", "skill"),
    ("Gardening", "skill_Gardening", "skill"),
    ("Cheese", "item_42", "item"),
    ("Pig Poop", "item_1495", "item"),
    ("Poop", "ability_ability_9488", "ability"),
    ("Graffiti: Mastering 'Poop'", "quest_quest_197", "quest"),
    ("Mushroom", "item_900", "item"),
]


@pytest.fixture(autouse=True)
def fake_index(monkeypatch):
    monkeypatch.setattr(
        "pgrag.rag.query_classifier._load_entity_index",
        lambda: sorted(SYNTHETIC_INDEX, key=lambda t: len(t[0]), reverse=True),
    )


def test_what_is_skill_entity():
    assert classify_query("what is the Dungcrafting skill") == "entity"


def test_find_entity_skill_hub():
    hub, dtype = find_entity("tell me about Dungcrafting")
    assert hub == "skillprofile_Pooping"
    assert dtype == "skill"


def test_find_entity_item_hub():
    hub, dtype = find_entity("what is cheese")
    assert hub == "item_42"
    assert dtype == "item"


def test_bare_name_entity():
    assert classify_query("Dungcrafting") == "entity"


def test_entity_hub_miss_fallback(monkeypatch):
    monkeypatch.setattr("pgrag.rag.query_classifier._load_entity_index", lambda: [])
    assert classify_query("what is the Dungcrafting skill") == "general"


def test_entity_pattern_no_name_general():
    assert classify_query("what is the proper way to farm") == "general"


def test_lookup_precedence():
    assert classify_query("how much does cheese cost") == "lookup"


def test_comparison_precedence():
    assert classify_query("what is the strongest cheese") == "comparison"


def test_word_boundary_match():
    hub, _ = find_entity("what is poop")
    assert hub == "ability_ability_9488"


def test_longest_name_wins():
    hub, _ = find_entity("what is pig poop")
    assert hub == "item_1495"


def test_unknown_query_general():
    assert classify_query("how does crafting work in this game") == "general"


def test_find_entity_typo_fallback(monkeypatch):
    monkeypatch.setattr("pgrag.rag.query_classifier.correct_query", lambda q: "what is mushroom")
    hub, dtype = find_entity("what is msurhoom")
    assert hub == "item_900"
    assert dtype == "item"


# --- find_entities (multi-entity greedy resolver) ---


def _identity(q):
    return q


def test_find_entities_two_hubs_in_query_order(monkeypatch):
    index = [
        *SYNTHETIC_INDEX,
        ("Punch", "ability_1001", "ability"),
        ("Front Kick", "ability_1002", "ability"),
    ]
    monkeypatch.setattr(
        "pgrag.rag.query_classifier._load_entity_index",
        lambda: sorted(index, key=lambda t: len(t[0]), reverse=True),
    )
    monkeypatch.setattr("pgrag.rag.query_classifier.correct_query", _identity)

    got = find_entities("which deals more damage, Punch or Front Kick")
    assert got == [
        ("Punch", "ability_1001", "ability"),
        ("Front Kick", "ability_1002", "ability"),
    ]


def test_find_entities_longest_non_overlap(monkeypatch):
    monkeypatch.setattr("pgrag.rag.query_classifier.correct_query", _identity)
    # Index (longest-first) has both "Pig Poop" and "Poop"; the long one claims
    # the span and the nested "Poop" must be skipped (non-overlapping).
    got = find_entities("what is pig poop")
    assert got == [("Pig Poop", "item_1495", "item")]


def test_find_entities_empty_when_none(monkeypatch):
    monkeypatch.setattr("pgrag.rag.query_classifier._load_entity_index", lambda: [])
    assert find_entities("no entities here") == []


def test_find_entities_corrected_fallback(monkeypatch):
    monkeypatch.setattr(
        "pgrag.rag.query_classifier.correct_query",
        lambda q: "tell me about mushroom",
    )
    # original yields nothing; corrected "mushroooms" -> item index (typo).
    got = find_entities("tell me about mushroooms")
    assert got == [("Mushroom", "item_900", "item")]


def test_plural_append_resolves_entity(monkeypatch):
    """'{entity}s' / '{entity}es' (append to the full name) resolves the
    entity: 'Field Mushrooms' -> 'Field Mushroom'. NOT a stem match: 'gardens'
    is not 'Gardening'+'s' and must stay unmatched."""
    index = [*SYNTHETIC_INDEX, ("Field Mushroom", "item_11004", "item")]
    monkeypatch.setattr(
        "pgrag.rag.query_classifier._load_entity_index",
        lambda: sorted(index, key=lambda t: len(t[0]), reverse=True),
    )
    assert find_entity("what is a Field Mushrooms")[0] == "item_11004"
    assert find_entity("gardens") == (None, None)
    assert find_entity("garden") == (None, None)


def test_plural_append_does_not_resolve_stem_variant():
    """'Field' is not 'Field Mushroom'+'s', so a bare 'Fields' must not
    resolve the compound entity."""
    assert find_entity("Fields") == (None, None)


def test_aliases_populate_entity_index(monkeypatch):
    """Query-facing aliases resolve to their real docs via the injected index;
    'Chalice Saga' -> lorebook, 'Animal Feces' -> poop items, 'Ranalon Den' ->
    quest cluster. Real index (undo fake_index autouse patch)."""
    if not Path("data/documents.json").exists():
        pytest.skip("corpus documents.json not present")
    monkeypatch.undo()
    assert find_entity("the Chalice Saga about")[0] == "lorebook_Book_103"
    assert find_entity("recipes use Animal Feces")[0] == "item_1501"
    assert find_entity("quests in Ranalon Den")[0] == "quest_quest_25401"


def test_lorebook_type_indexed(monkeypatch):
    """'lorebook' is an entity type; the Wasted Wishes lore book resolves.
    Real index (undo fake_index autouse patch)."""
    if not Path("data/documents.json").exists():
        pytest.skip("corpus documents.json not present")
    monkeypatch.undo()
    assert find_entity("the lore book The Wasted Wishes")[0] == "lorebook_Book_101"


def test_classify_two_entities_is_comparison(monkeypatch):
    index = [
        *SYNTHETIC_INDEX,
        ("Punch", "ability_1001", "ability"),
        ("Front Kick", "ability_1002", "ability"),
    ]
    monkeypatch.setattr(
        "pgrag.rag.query_classifier._load_entity_index",
        lambda: sorted(index, key=lambda t: len(t[0]), reverse=True),
    )
    monkeypatch.setattr("pgrag.rag.query_classifier.correct_query", _identity)

    assert classify_query("which deals more damage, Punch or Front Kick") == "comparison"
