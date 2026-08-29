"""Whitebox suite for the per-level combat-XP comparison docs (computed family)."""
import re

from pgrag.documents.combat_xp import build_combat_xp_documents


class _FakeDB:
    def __init__(self, tables):
        self.tables = tables
        self.wiki = {}


def _xp_tables():
    return {
        "810_EpicBoss3": {"Level_40": {"MONSTER_COMBAT_XP_VALUE": 184, "MAX_HEALTH": 500}},
        "811_EpicBoss4": {"Level_40": {"MONSTER_COMBAT_XP_VALUE": 230}},
        "962_ExtraXp3": {"Level_40": {"MONSTER_COMBAT_XP_VALUE": 23}},
        "989_Evasion4": {"Level_40": {"MONSTER_COMBAT_XP_VALUE": 19}},
    }


def test_combat_xp_doc_sweeps_all_archetypes_sorted_desc():
    db = _FakeDB({"advancementtables": _xp_tables()})
    docs = build_combat_xp_documents(db)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "combatxp_40"
    assert doc["type"] == "combatxp"
    assert doc["metadata"]["source"] == "computed"
    assert doc["metadata"]["table"] == "advancementtables"
    assert doc["metadata"]["name"].startswith("Combat XP by Monster Archetype,")
    assert doc["metadata"]["name"].endswith("Level 40")
    assert doc["metadata"]["level"] == 40
    lines = [line for line in doc["text"].splitlines() if line.startswith("- ")]
    assert [line.split(":", 1)[0] for line in lines] == [
        
        "- 811_EpicBoss4", "- 810_EpicBoss3", "- 962_ExtraXp3", "- 989_Evasion4",
    ]
    assert "Level 39" not in doc["text"]


def test_skips_levels_with_single_contributor():
    tables = {
        "a": {"Level_1": {"MONSTER_COMBAT_XP_VALUE": 5}, "Level_2": {"MONSTER_COMBAT_XP_VALUE": 10}},
        "b": {"Level_1": {"MONSTER_COMBAT_XP_VALUE": 7}},
        "c": {"Level_2": {"MONSTER_COMBAT_XP_VALUE": 8}},
    }
    db = _FakeDB({"advancementtables": tables})
    docs = build_combat_xp_documents(db)
    doc_map = {d["id"]: d for d in docs}
    assert set(doc_map) == {"combatxp_1", "combatxp_2"}
    assert re.search(r"- a:\s*5", doc_map["combatxp_1"]["text"])
    assert re.search(r"- b:\s*7", doc_map["combatxp_1"]["text"])
    assert re.search(r"- c:\s*8", doc_map["combatxp_2"]["text"])


def test_handles_float_values_and_ties():
    tables = {
        "x": {"Level_1": {"MONSTER_COMBAT_XP_VALUE": 230.0}},
        "y": {"Level_1": {"MONSTER_COMBAT_XP_VALUE": 230}},
    }
    db = _FakeDB({"advancementtables": tables})
    docs = build_combat_xp_documents(db)
    assert len(docs) == 1
    assert re.search(r"- x:\s*230", docs[0]["text"])
    assert re.search(r"- y:\s*230", docs[0]["text"])
