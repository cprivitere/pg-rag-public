"""Query planning (Phase 5): high-confidence extraction vs no-filter fallback.

A plan is only emitted when the constraint is unambiguous; ambiguous or
unsupported questions must return None so retrieval stays broad (a false
negative is worse than a false positive).
"""
from pgrag.rag.query_plan import plan_query


# --- recipe + skill ---

def test_recipe_skill_and_max_level():
    p = plan_query("which alchemy recipes can I make at level 30?")
    assert p is not None
    assert p["native"] == {
        "$and": [
            {"type": "recipe"},
            {"skill": "Alchemy"},
            {"skill_level_req": {"$lte": 30}},
        ]
    }
    assert p["token"] == {}


def test_recipe_skill_level_before_word():
    p = plan_query("recipes that require alchemy below level 15")
    assert p["native"] == {
        "$and": [
            {"type": "recipe"},
            {"skill": "Alchemy"},
            {"skill_level_req": {"$lte": 15}},
        ]
    }


def test_recipe_skill_only_no_level():
    p = plan_query("alchemy recipe")
    assert p["native"] == {"$and": [{"type": "recipe"}, {"skill": "Alchemy"}]}
    assert "skill_level_req" not in p["native"]


def test_recipe_skill_canonicalizes_collapsed_name():
    p = plan_query("recipes using flower arrangement")
    assert p["native"] == {"$and": [{"type": "recipe"}, {"skill": "FlowerArrangement"}]}


# --- recipe ingredient ---

def test_recipe_ingredient_token():
    p = plan_query("recipes crafted with spider silk")
    assert p is not None
    assert p["native"] == {"type": "recipe"}
    assert p["token"] == {"ingredients": "Spider Silk"}


def test_recipe_ingredient_singularized():
    # "mushrooms" -> item "Mushroom" (post-fusion token membership is exact)
    p = plan_query("what recipes use mushrooms?")
    assert p["token"] == {"ingredients": "Mushroom"}


def test_ingredient_stopword_phrase_is_unplanned():
    # "bottle of milk" cannot match the delimited list -> no-filter is safer
    assert plan_query("recipes using a bottle of milk") is None


def test_mass_noun_ingredient_not_singularized():
    # "feces" is not "fece"; only consonant-final plurals get singularized.
    p = plan_query("what recipes use animal feces?")
    assert p is not None
    assert p["token"] == {"ingredients": "Animal Feces"}


# --- ability + damage type ---

def test_ability_damage_type():
    p = plan_query("which abilities deal fire damage?")
    assert p["native"] == {"$and": [{"type": "ability"}, {"damage_type": "Fire"}]}
    assert p["token"] == {}


def test_ability_damage_alias_cold():
    p = plan_query("best ice damage ability")
    assert p["native"] == {"$and": [{"type": "ability"}, {"damage_type": "Cold"}]}


def test_native_where_is_chroma_valid():
    # A bare multi-field dict is not a valid Chroma `where` (it needs a single
    # operator cell); every emitted `native` must be a single-field clause or
    # an `$and` list of single-field clauses.
    def check(native):
        assert isinstance(native, dict)
        if "$and" in native:
            for clause in native["$and"]:
                assert len(clause) == 1, clause
        else:
            assert len(native) == 1, native

    for q in [
        "which alchemy recipes can I make at level 30?",
        "alchemy recipe",
        "recipes crafted with spider silk",
        "which abilities deal fire damage?",
    ]:
        p = plan_query(q)
        assert p is not None
        check(p["native"])


# --- no-filter fallbacks ---

def test_location_gathering_question_unplanned():
    # Needs the entity index to know item vs npc/area; ambiguity -> no filter.
    assert plan_query("where can I find the mushroom trainer?") is None


def test_bare_verb_unplanned():
    assert plan_query("crafting") is None


def test_generic_unrelated_unplanned():
    assert plan_query("what is the weather today") is None


def test_empty_unplanned():
    assert plan_query("") is None
    assert plan_query(None) is None


def test_creature_location_listing_plans_to_creatures_table():
    """'List the locations with deer, sheep, goats, cows, or oxen' — the
    authoritative answer table is `creatures`; the Chroma where narrows to it
    so rarer spawns (Infernal Buck at dense-rank ~106) are not starved out."""
    p = plan_query(
        "List all the locations with deer, sheep, goats, cows, or oxen in the game."
    )
    assert p == {"native": {"table": "creatures"}, "token": {}, "label": "creature locations"}


def test_where_do_animals_live_plans_to_creatures():
    p = plan_query("where do deer live?")
    assert p is not None
    assert p["native"] == {"table": "creatures"}


def test_non_creature_location_listing_unplanned():
    """A location question without an animal/creature token must NOT trap
    itself in the creatures table (e.g. blacksmithing trainers)."""
    assert plan_query("the locations of blacksmithing trainers") is None
    assert plan_query("where can I find the mushroom trainer?") is None