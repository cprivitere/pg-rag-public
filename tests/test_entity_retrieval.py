"""Tests pgrag.rag.entity_retrieval.build_entity_context.

Covers dossier assembly order (coverage, rows, narrative), the leveling doc
joining skill hubs gated on intent, facet type filters, budget caps, and wiki
page linkage keyed on the entity_id, plus multi-entity dedup and tracing.
"""

import pytest

from pgrag.rag import entity_retrieval as er


def _mk(doc_id, text, chunk_index=None, dtype="skill", name="X"):
    meta = {"type": dtype, "name": name, "table": "skills"}
    if chunk_index is not None:
        meta["chunk_index"] = chunk_index
        meta["chunk_count"] = 3
    return {"id": doc_id, "text": text, "metadata": meta}


def _mkdocs():
    return [
        _mk("skillprofile_Pooping_chunk_1", "Reuse 1800s XP Table Quests", 1),
        _mk("skillprofile_Pooping_chunk_0", "Skill Profile: Dungcrafting Abilities", 0),
    ]


def _empty_retrieve(**kwargs):
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


@pytest.fixture(autouse=True)
def fake_docs(monkeypatch):
    monkeypatch.setattr(er, "_load_docs", _mkdocs)
    monkeypatch.setattr(er, "retrieve", _empty_retrieve)


def test_wiki_links_ordered_coverage_row_narrative():
    # Wiki records of the same entity: coverage, then rows, then narrative.
    hub = _mk("skill_Alchemy_chunk_0", "Alchemy overview", 0, dtype="skill")
    cov = {"id": "wiki_Alchemy_table_0_coverage", "text": "covers A, B",
           "metadata": {"type": "wiki", "table": "wiki", "entity_id": "skill_Alchemy",
                        "entity_type": "skill", "table_record": "coverage"}}
    row = {"id": "wiki_Alchemy_table_0_row_1", "text": "A row",
           "metadata": {"type": "wiki", "table": "wiki", "entity_id": "skill_Alchemy",
                        "entity_type": "skill", "table_record": "row"}}
    narr = {"id": "wiki_Alchemy_Overview", "text": "Alchemy is a skill",
            "metadata": {"type": "wiki", "table": "wiki", "entity_id": "skill_Alchemy",
                         "entity_type": "skill"}}
    er._load_docs = lambda: [hub, row, narr, cov]
    r = er.build_entity_context("what is Alchemy", "skill_Alchemy")
    ids = r["ids"][0]
    assert ids == [
        "skill_Alchemy_chunk_0",
        "wiki_Alchemy_table_0_coverage",
        "wiki_Alchemy_table_0_row_1",
        "wiki_Alchemy_Overview",
    ]


def test_other_entity_wiki_excluded():
    # A wiki page owned by another entity is left out of this dossier.
    hub = _mk("skill_Alchemy_chunk_0", "Alchemy overview", 0, dtype="skill")
    other = {"id": "wiki_Mycology_table_0_coverage", "text": "covers P",
             "metadata": {"type": "wiki", "table": "wiki", "entity_id": "skill_Mycology",
                          "entity_type": "skill", "table_record": "coverage"}}
    er._load_docs = lambda: [hub, other]
    r = er.build_entity_context("what is Alchemy", "skill_Alchemy")
    assert all("Mycology" not in i for i in r["ids"][0])


def test_hub_whole_loaded_in_order():
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    assert r["ids"][0] == [
        "skillprofile_Pooping_chunk_0",
        "skillprofile_Pooping_chunk_1",
    ]
    assert r["documents"][0][0].startswith("Skill Profile")
    assert "XP Table" in r["documents"][0][1]


def test_unchunked_hub_included():
    docs = [_mk("item_42", "Item text")]
    er._load_docs = lambda: docs
    r = er.build_entity_context("what is cheese", "item_42")
    assert r["ids"][0] == ["item_42"]


def test_leveling_doc_joined_for_skill_hub():
    # The computed per-skill leveling dossier (leveling_<Skill>) joins the skill
    # hub right after it, in the non-truncated heads, so leveling questions see
    # the full cumulative XP ladder.
    hub = _mk("skill_Cheesemaking", "Cheesemaking skill")
    lvl = {"id": "leveling_Cheesemaking",
           "text": "Level 25: 990 XP (cumulative 11710)",
           "metadata": {"type": "computed", "name": "Cheesemaking"}}
    er._load_docs = lambda: [hub, lvl]
    r = er.build_entity_context("how to level Cheesemaking from 17 to 25",
                                "skill_Cheesemaking", include_leveling=True)
    ids = r["ids"][0]
    assert "leveling_Cheesemaking" in ids
    assert ids.index("leveling_Cheesemaking") == 1


def test_leveling_doc_gated_on_intent():
    # Leveling doc is NOT shoved into every skill dossier — only leveling
    # questions (pipeline passes include_leveling). Unrelated skill questions
    # keep their wiki/table rows un-crowded.
    hub = _mk("skill_Cheesemaking", "Cheesemaking skill")
    lvl = {"id": "leveling_Cheesemaking",
           "text": "Level 25: 990 XP (cumulative 11710)",
           "metadata": {"type": "computed", "name": "Cheesemaking"}}
    er._load_docs = lambda: [hub, lvl]
    r = er.build_entity_context("where are field mushrooms",
                                "skill_Cheesemaking", include_leveling=False)
    assert "leveling_Cheesemaking" not in r["ids"][0]


def test_leveling_doc_absent_builds_dossier():
    # Corpora without computed docs (no leveling_X): dossier builds with just
    # the hub, no crash.
    hub = _mk("skill_Cheesemaking", "Cheesemaking skill")
    er._load_docs = lambda: [hub]
    r = er.build_entity_context("level Cheesemaking", "skill_Cheesemaking")
    assert r["ids"][0] == ["skill_Cheesemaking"]


def test_facet_type_filters(monkeypatch):
    calls = []

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True, trace=None):
        calls.append((question, metadata_filter, count))
        return _empty_retrieve()

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    filters = [c[1] for c in calls]
    assert {"type": "recipe"} in filters
    # Leaner per-facet caps are the stated contract (test_facet_type_filters):
    # recipe pulls 8, every non-recipe facet 6 — a revert to the noisy 20/10
    # full-set dossier (dungcrafting regression) must fail here.
    for _q, filt, count in calls:
        if not filt:
            continue
        if filt.get("type") == "recipe":
            assert count == 8, f"recipe facet count should be 8, got {count}"
        else:
            assert count == 6, f"non-recipe facet count should be 6, got {count}"
    assert {"type": "quest"} in filters
    assert {"type": "npc"} in filters
    assert {"type": "advancementtable"} in filters


def test_facet_uses_entity_name_not_full_question(monkeypatch):
    calls = []

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True, trace=None):
        calls.append(question)
        return _empty_retrieve()

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    er.build_entity_context(
        "Tell me what recipes I will use while leveling saddlery from 0 to 15",
        "skillprofile_Saddlery",
    )
    for q in calls:
        assert "Tell me" not in q
        assert "leveling" not in q
        assert "Saddlery" in q


def test_facet_dedupe(monkeypatch):
    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True, trace=None):
        return {
            "ids": [["skillprofile_Pooping_chunk_1", "recipe_906"]],
            "documents": [["dup text", "Recipe text"]],
            "metadatas": [[{"type": "skill"}, {"type": "recipe"}]],
            "distances": [[0.5, 0.3]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    ids = r["ids"][0]
    assert ids.count("skillprofile_Pooping_chunk_1") == 1
    assert "recipe_906" in ids


def test_budget_cap(monkeypatch):
    monkeypatch.setattr(er, "CONTEXT_BUDGET", 100)

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True, trace=None):
        return {
            "ids": [["recipe_906"]],
            "documents": [["R" * 300]],
            "metadatas": [[{"type": "recipe"}]],
            "distances": [[0.3]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("what is Dungcrafting", "skillprofile_Pooping")
    texts = r["documents"][0]
    assert sum(len(t) for t in texts) <= 100
    assert texts[0].startswith("Skill Profile")


def test_hub_miss_returns_none():
    er._load_docs = lambda: []
    assert er.build_entity_context("what is missing", "skillprofile_Nope") is None


def test_non_entity_prefix_no_facets():
    docs = [_mk("summary_cheese", "Cheese summary", dtype="summary", name="Cheese Summary")]
    er._load_docs = lambda: docs
    r = er.build_entity_context("what is summary", "summary_cheese")
    assert r["ids"][0] == ["summary_cheese"]


def test_skill_recipes_sorted_by_required_level(monkeypatch):
    """Skill dossier lists recipes lowest-required-level first so the LLM can
    pick what is usable at the player's target level."""
    import re

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True, rerank=True, trace=None):
        if metadata_filter == {"type": "recipe"}:
            return {
                "ids": [["recipe_50", "recipe_5", "recipe_25"]],
                "documents": [
                    [
                        "Recipe: High\n\nRequired Skill Level:\n50",
                        "Recipe: Low\n\nRequired Skill Level:\n5",
                        "Recipe: Mid\n\nRequired Skill Level:\n25",
                    ]
                ],
                "metadatas": [[{"type": "recipe"}, {"type": "recipe"}, {"type": "recipe"}]],
                "distances": [[0.5, 0.4, 0.3]],
            }
        return _empty_retrieve()

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("level Pooping", "skillprofile_Pooping")

    ids = r["ids"][0]
    # hub chunks first
    assert ids[:2] == ["skillprofile_Pooping_chunk_0", "skillprofile_Pooping_chunk_1"]
    # then recipes ascending by required level
    recipe_ids = ids[2:]
    assert recipe_ids == ["recipe_5", "recipe_25", "recipe_50"]


# --- wiki page linkage (Step 5) ---


def _mkwiki(doc_id, text, entity_id):
    return {
        "id": doc_id,
        "text": text,
        "metadata": {"type": "wiki", "table": "wiki", "entity_id": entity_id},
    }


def test_wiki_docs_linked_to_skillprofile_hub():
    """skillprofile hubs must pull wiki pages whose entity_id is the CDN
    skill doc (e.g. Mushroom Farming mechanics live in the wiki, not the
    skill profile)."""
    docs = [
        _mk("skillprofile_MushroomFarming_chunk_0", "Skill Profile", 0),
        _mkwiki(
            "wiki_Mushroom Farming_Mechanics_chunk_3",
            "Field Mushroom 15 05 hrs Bone Organs",
            "skill_MushroomFarming",
        ),
    ]
    er._load_docs = lambda: docs
    r = er.build_entity_context(
        "Which mushrooms can I grow?", "skillprofile_MushroomFarming"
    )
    assert "wiki_Mushroom Farming_Mechanics_chunk_3" in r["ids"][0]


def test_wiki_docs_linked_by_item_entity_id():
    docs = [
        _mk("item_11004", "Item: Field Mushroom", None, "item", "Field Mushroom"),
        _mkwiki("wiki_Field Mushroom_How_to_Obtain", "15 Mycology required",
                "item_11004"),
    ]
    er._load_docs = lambda: docs
    r = er.build_entity_context("Where can I find Field Mushrooms?", "item_11004")
    assert "wiki_Field Mushroom_How_to_Obtain" in r["ids"][0]


def test_wiki_linkage_respects_budget(monkeypatch):
    monkeypatch.setattr(er, "CONTEXT_BUDGET", 20)
    docs = [
        _mk("skillprofile_MushroomFarming_chunk_0", "Skill Profile", 0),
        _mkwiki(
            "wiki_Mushroom Farming_Mechanics_chunk_3",
            "Field Mushroom 15 05 hrs Bone Organs",
            "skill_MushroomFarming",
        ),
    ]
    er._load_docs = lambda: docs
    r = er.build_entity_context(
        "Which mushrooms can I grow?", "skillprofile_MushroomFarming"
    )
    assert "wiki_Mushroom Farming_Mechanics_chunk_3" not in r["ids"][0]


def test_wiki_linkage_skips_unlinked_wiki():
    docs = [
        _mk("skillprofile_Pooping_chunk_0", "Skill Profile", 0),
        _mkwiki("wiki_Serbule_Overview", "A town", "none"),
    ]
    er._load_docs = lambda: docs
    r = er.build_entity_context("what is Pooping", "skillprofile_Pooping")
    assert "wiki_Serbule_Overview" not in r["ids"][0]


def test_budget_param_truncates(monkeypatch):
    """An explicit budget caps the dossier while keeping the first doc."""

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True,
                      rerank=True, trace=None):
        return {
            "ids": [["recipe_extra"]],
            "documents": [["R" * 300]],
            "metadatas": [[{"type": "recipe"}]],
            "distances": [[0.3]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)
    r = er.build_entity_context("what is Pooping", "skillprofile_Pooping", budget=50)
    texts = r["documents"][0]
    assert sum(len(t) for t in texts) <= 50
    assert texts[0].startswith("Skill Profile")


def test_build_multi_entity_context_keeps_both_hubs_dedupes(monkeypatch):
    """Multi-entity context labels each block and dedupes a shared doc."""
    ability_docs = [
        _mk("ability_punch_chunk_0", "Punch does 6 damage.", 0, "ability", "Punch"),
        _mk("ability_front_kick_chunk_0", "Front Kick does 11 damage.", 0, "ability", "Front Kick"),
    ]
    monkeypatch.setattr(er, "_load_docs", lambda: ability_docs)

    def fake_retrieve(question, count=3, metadata_filter=None, hybrid=True,
                      rerank=True, trace=None):
        return {
            "ids": [["recipe_shared"]],
            "documents": [["Shared recipe text"]],
            "metadatas": [[{"type": "recipe", "name": "Shared"}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr(er, "retrieve", fake_retrieve)

    r = er.build_multi_entity_context(
        "which deals more damage",
        [("Punch", "ability_punch", "ability"),
         ("Front Kick", "ability_front_kick", "ability")],
    )
    docs = r["documents"][0]
    assert "=== Punch (ability) ===" in docs
    assert "=== Front Kick (ability) ===" in docs
    assert "Punch does 6 damage." in docs
    assert "Front Kick does 11 damage." in docs
    # the shared recipe was returned by both hubs' facets but kept once
    assert docs.count("Shared recipe text") == 1


def test_build_multi_entity_context_unresolved_recorded_in_trace(monkeypatch):
    mono = {"unresolved": []}
    monkeypatch.setattr(er, "_load_docs", lambda: [])
    r = er.build_multi_entity_context(
        "Punch or Ghost", [("Punch", "ability_punch", "ability"),
                           ("Ghost", "ability_ghost", "ability")],
        trace=mono,
    )
    assert r is None
    assert "Ghost" in mono["unresolved"]

def test_skill_dossier_pulls_own_abilities_deterministically():
    # Contract: a skill/skillprofile dossier attaches ALL of its own
    # standalone ability docs by metadata.skill == hub skill code (the CDN
    # ability `Skill` field), and excludes off-skill ability docs, rather
    # than relying on the fuzzy top-N facet query that mis-ranks
    # combat-flavored but legitimate skills abilities (e.g. Gardening's
    # Spade Assault 1-6) below the cut. Regression guard for Spade Assault.
    hub = _mk("skillprofile_Gardening_chunk_0", "Gardening Abilities XP", 0,
              name="Gardening")
    own1 = {"id": "ability_ability_9301", "text": "Spade Assault",
            "metadata": {"type": "ability", "skill": "Gardening",
                         "table": "abilities"}}
    own2 = {"id": "ability_ability_9316", "text": "Pumpkin Bomb",
            "metadata": {"type": "ability", "skill": "Gardening",
                         "table": "abilities"}}
    off = {"id": "ability_ability_9485", "text": "Weed Plants",
           "metadata": {"type": "ability", "skill": "CivilEngineering",
                        "table": "abilities"}}
    er._load_docs = lambda: [hub, own1, own2, off]
    r = er.build_entity_context("Tell me about Gardening", "skillprofile_Gardening")
    ids = r["ids"][0]
    assert "ability_ability_9301" in ids
    assert "ability_ability_9316" in ids
    assert "ability_ability_9485" not in ids

def test_skill_dossier_bounds_and_level_orders_own_abilities(monkeypatch):
    # Contract: for high-ability skills (Archery ~257 own ability docs) the
    # deterministic pull is bounded to _MAX_SKILL_ABILITIES and keeps the
    # lowest-level (most commonly used) abilities, so the budget cut can't
    # flood the dossier or drop the useful abilities in favor of an id-order
    # subset. Exact-ability questions route to the `ability` hub elsewhere.
    monkeypatch.setattr(er, "_MAX_SKILL_ABILITIES", 2)
    hub = _mk("skillprofile_Archery_chunk_0", "Archery Abilities", 0,
              name="Archery")
    low = {"id": "ability_ability_2501", "text": "Quick Shot",
           "metadata": {"type": "ability", "skill": "Archery",
                        "table": "abilities", "level": 1}}
    mid = {"id": "ability_ability_2601", "text": "Long Shot",
           "metadata": {"type": "ability", "skill": "Archery",
                        "table": "abilities", "level": 5}}
    hi = {"id": "ability_ability_2701", "text": "Multishot 8",
          "metadata": {"type": "ability", "skill": "Archery",
                       "table": "abilities", "level": 60}}
    er._load_docs = lambda: [hub, hi, low, mid]  # deliberately unordered
    r = er.build_entity_context("Tell me about Archery", "skillprofile_Archery")
    abilities = [i for i in r["ids"][0] if i.startswith("ability_ability_")]
    assert len(abilities) == 2                     # bounded by the cap
    assert "ability_ability_2501" in abilities     # lowest level kept
    assert "ability_ability_2701" not in abilities  # highest level dropped
