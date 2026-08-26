"""Contract tests for build_creature_zones_documents.

Contract: each wiki MOB page with a known spawn zone emits exactly one
creature_<Name> doc (metadata.table "creatures", source "wiki"). Zones come
from {{MOB Location| area = ... }} blocks and [[Category:<Zone> Creatures]]
markers (deduped, in document order); creature_type comes from
{{MOB infobox| type = ... }}. Text carries no wikicode residue.
"""

import re

from pgrag.documents.creature_zones import build_creature_zones_documents


class FakeDB:
    def __init__(self, wiki):
        self.wiki = wiki


INFERNAL_BUCK = """__NOTOC__
{{MOB infobox
| title = Infernal Buck
| type      = Ruminant
| health    = 18347
}}
[[Infernal Buck]] are a type of deer infected with demonic energy.
== Locations ==
{{MOB Location
| area = Vidaria
| lootlevel = 92
}}
{{MOB Location
| area = Statehelm Sewers
| lootlevel = 102
}}
"""

VIDARIAN_DOE = """__NOTOC__
{{MOB infobox
| title = Vidarian Doe
| type      = Ruminant
}}
[[Vidarian Doe]] are seemingly magical female deer.
== Locations ==
{{MOB Location
| area = [[Vidaria]]
| location = Across the farmlands, usually areas with more trees.
}}
"""

# Zone via category only, linked area value, duplicate zone to force dedup.
CATEGORY_PAGE = """{{MOB infobox
| type = Arthropod
}}
[[Category:Serbule Hills Spider Cave Creatures]]
{{MOB Location
| area = [[Serbule Hills]]
}}
{{MOB Location
| area = Serbule Hills
}}
"""

# A creature with no spawn zone should not produce a doc.
NO_ZONE = """{{MOB infobox
| type = Ruminant
}}
{{MOB Location
| area =
}}
"""


def test_mob_location_and_type():
    db = FakeDB({"Infernal Buck": INFERNAL_BUCK})
    docs = build_creature_zones_documents(db)
    assert len(docs) == 1
    d = docs[0]
    assert d["id"] == "creature_Infernal_Buck"
    assert d["type"] == "wiki"
    assert d["metadata"]["source"] == "wiki"
    assert d["metadata"]["table"] == "creatures"
    assert d["metadata"]["name"] == "Infernal Buck"
    assert d["metadata"]["creature_type"] == "Ruminant"
    assert d["metadata"]["locations"] == "Vidaria | Statehelm Sewers"
    assert "Vidaria" in d["text"] and "Statehelm Sewers" in d["text"]
    assert "Ruminant" in d["text"]
    # Lead description must carry the species so "deer" queries recall the doc.
    assert "type of deer" in d["text"]


def test_wikilinked_area_and_location_subfield():
    db = FakeDB({"Vidarian Doe": VIDARIAN_DOE})
    d = build_creature_zones_documents(db)[0]
    # area value is a wikilink; "location" sub-field must not leak as a zone.
    assert d["metadata"]["locations"] == "Vidaria"
    assert "farmlands" not in d["text"]


def test_category_zones_deduped_with_areas():
    db = FakeDB({"Spider": CATEGORY_PAGE})
    d = build_creature_zones_documents(db)[0]
    # MOB areas first (deduped) then category zone appended.
    assert d["metadata"]["locations"] == "Serbule Hills | Serbule Hills Spider Cave"
    assert d["metadata"]["creature_type"] == "Arthropod"


def test_no_zone_page_emits_nothing():
    db = FakeDB({"Nameless": NO_ZONE, "Ordinary": "no templates here"})
    assert build_creature_zones_documents(db) == []


def test_empty_wiki_emits_nothing():
    assert build_creature_zones_documents(FakeDB({})) == []
    assert build_creature_zones_documents(FakeDB(None)) == []


def test_text_is_clean_and_deterministic():
    db = FakeDB({"Infernal Buck": INFERNAL_BUCK})
    d = build_creature_zones_documents(db)[0]
    assert not re.search(r"\{\{|\}\}|\[\[|]]", d["text"])
    again = build_creature_zones_documents(db)[0]
    assert again["text"] == d["text"]