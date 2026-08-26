"""Tests pgrag.documents.wiki_builder.

Covers _preserve_template_names keeping Item/NPC/Quest/Skill/Area/Recipe/
LoreBook/Ability template names while other wikicode is stripped, and
build_wiki_documents emitting row, coverage, and narrative records.
"""

from pgrag.documents.wiki_builder import build_wiki_documents, _preserve_template_names


class FakeDB:
    def __init__(self, wiki):
        self.wiki = wiki
        self.tables = {}


# --- _preserve_template_names unit tests ---


def test_item_template_preserved():
    raw = "{{Item|Mortaferus Mushroom}} || 95 || [[Vidaria]]"
    result = _preserve_template_names(raw)
    assert "Mortaferus Mushroom" in result
    assert "{{Item|" not in result


def test_npc_template_preserved():
    raw = "Talk to {{NPC|Rita}} in [[Serbule]]."
    result = _preserve_template_names(raw)
    assert "Rita" in result
    assert "{{NPC|" not in result


def test_quest_template_preserved():
    raw = "Complete {{Quest|Graffiti_Glyph27}} for XP."
    result = _preserve_template_names(raw)
    assert "Graffiti_Glyph27" in result


def test_skill_template_preserved():
    raw = "Requires {{Skill|Alchemy}} level 50."
    result = _preserve_template_names(raw)
    assert "Alchemy" in result


def test_area_template_preserved():
    raw = "Found in {{Area|AreaSunVale}}."
    result = _preserve_template_names(raw)
    assert "AreaSunVale" in result


def test_recipe_template_preserved():
    raw = "Uses {{Recipe|Healing Potion}} recipe."
    result = _preserve_template_names(raw)
    assert "Healing Potion" in result


def test_lorebook_template_preserved():
    raw = "Read {{LoreBook|The Wasted Wishes}}."
    result = _preserve_template_names(raw)
    assert "The Wasted Wishes" in result


def test_ability_template_preserved():
    raw = "Cast {{Ability|Fireball}} on enemies."
    result = _preserve_template_names(raw)
    assert "Fireball" in result


def test_ignores_unknown_templates():
    raw = "{{RandomTemplate|KeepMe}} and {{Item|Mushroom}}"
    result = _preserve_template_names(raw)
    assert "{{RandomTemplate|KeepMe}}" in result
    assert "Mushroom" in result


def test_template_with_extra_args():
    raw = "{{Item|Boletus Mushroom|link=no|icon=small}}"
    result = _preserve_template_names(raw)
    assert "Boletus Mushroom" in result
    assert "{{Item|" not in result


def test_no_templates_unchanged():
    raw = "Just plain text with no templates."
    assert _preserve_template_names(raw) == raw


def test_multiple_templates():
    raw = "Get {{Skill|Mycology}} 60, then farm {{Item|Ghostshroom}} in {{Area|Vidaria}}."
    result = _preserve_template_names(raw)
    assert "Mycology" in result
    assert "Ghostshroom" in result
    assert "Vidaria" in result
    assert "{{" not in result


# --- build_wiki_documents integration tests ---


def test_wiki_page_preserves_item_names():
    raw = """__NOTOC__
== Harvestables ==
{| class="wikitable"
!Name
!Level
|-
| {{Item|Parasol Mushroom}} || 00
|-
| {{Item|Mycena Mushroom}} || 05
|-
| {{Item|Mortaferus Mushroom}} || 95
|}
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Parasol Mushroom" in all_text
    assert "Mycena Mushroom" in all_text
    assert "Mortaferus Mushroom" in all_text


def test_wiki_page_still_strips_other_wikicode():
    raw = """__NOTOC__
== Overview ==
This is [[Mycology]], a skill for gathering mushrooms.
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Mycology" in all_text
    assert "[[" not in all_text
    assert "]]" not in all_text


def test_wiki_table_emits_row_and_coverage_records():
    raw = """__NOTOC__
== Grow ==
{| class="sortable" style="width:100%"
! Mushroom
! Mycology Req
|-
| {{Item|Parasol Mushroom}} || N/A
|-
| {{Item|Mycena Mushroom}} || 05
|}
"""
    db = FakeDB({"Mushroom Farming": raw})
    docs = build_wiki_documents(db)
    by_id = {d["id"]: d for d in docs}

    cov = by_id.get("wiki_Mushroom_Farming_table_0_coverage")
    row0 = by_id.get("wiki_Mushroom_Farming_table_0_row_0")
    row1 = by_id.get("wiki_Mushroom_Farming_table_0_row_1")

    assert cov is not None
    assert row0 is not None and row1 is not None

    # coverage: first-column cells, compact
    assert cov["metadata"]["table_record"] == "coverage"
    assert cov["metadata"]["table_id"] == "wiki_Mushroom_Farming_table_0"
    assert "Parasol Mushroom" in cov["text"]
    assert "Mycena Mushroom" in cov["text"]

    # rows: granular, cell-cleaned
    assert row0["metadata"]["table_record"] == "row"
    assert row0["metadata"]["row_key"] == "Parasol Mushroom"
    assert "Parasol Mushroom" in row0["text"]
    assert "N/A" in row0["text"]
    assert "{{Item|" not in row0["text"]
    assert row1["metadata"]["row_key"] == "Mycena Mushroom"


def test_wiki_table_cells_clean_links_and_markup():
    raw = """__NOTOC__
== Grow ==
{| class="wikitable"
! Item
! Where
|-
| {{Item|Field Mushroom}} || [[Serbule]]''Grass''<br/>&nbsp;
|}
"""
    db = FakeDB({"Field Mushroom": raw})
    docs = build_wiki_documents(db)
    row = next(d for d in docs if d["metadata"].get("table_record") == "row")
    assert "Field Mushroom" in row["text"]
    assert "[[Serbule" not in row["text"]
    assert "<br/>" not in row["text"]
    assert "''" not in row["text"]


def test_wiki_table_truncated_to_950():
    long_cell = "x" * 2000
    opener = "{|" + ' class="wikitable"'
    raw = f"""__NOTOC__
== Grow ==
{opener}
! Item
|-
| {long_cell} || tail
|}}
"""
    db = FakeDB({"Tabby": raw})
    docs = [d for d in build_wiki_documents(db)
            if d.get("type") == "wiki" and d["metadata"].get("table_record")]
    assert docs
    for d in docs:
        assert len(d["text"]) <= 950


def test_unclosed_table_falls_back_to_narrative():
    # No closing |} -> no partial row records; page keeps inline narrative.
    raw = """__NOTOC__
== Grow ==
{| class="wikitable"
| {{Item|Parasol Mushroom}} || N/A
-
| {{Item|Mycena Mushroom}} || 05
"""
    db = FakeDB({"Mushroom Farming": raw})
    docs = build_wiki_documents(db)
    table_recs = [d for d in docs
                  if d.get("type") == "wiki" and d["metadata"].get("table_record")]
    assert table_recs == []
    all_text = " ".join(d["text"] for d in docs)
    assert "Parasol Mushroom" in all_text
    assert "Mycena Mushroom" in all_text


def test_wiki_table_removed_from_narrative_section():
    raw = """__NOTOC__
== Grow ==
Intro line describing the growing mechanics in enough detail to survive the minimum section length threshold for this page.
{| class="wikitable"
| {{Item|Parasol Mushroom}} || N/A
|}
"""
    db = FakeDB({"Mushroom Farming": raw})
    docs = build_wiki_documents(db)
    narr = [d for d in docs if not d["metadata"].get("table_record")]
    assert any("Intro line" in d["text"] for d in narr)
    # the raw table markup is gone from every narrative doc
    for d in narr:
        assert "{|" not in d["text"]
        assert "{{Item|" not in d["text"]
    raw = """__NOTOC__
== Trainers ==
Talk to {{NPC|Mushroom Jack}} in [[Serbule]] to learn.
"""
    db = FakeDB({"Mycology": raw})
    docs = build_wiki_documents(db)
    all_text = " ".join(d["text"] for d in docs)
    assert "Mushroom Jack" in all_text


def test_empty_wiki_produces_no_docs():
    db = FakeDB({})
    docs = build_wiki_documents(db)
    assert docs == []


def test_short_section_skipped():
    raw = """__NOTOC__
== Tiny ==
Hi.
"""
    db = FakeDB({"TestPage": raw})
    docs = build_wiki_documents(db)
    assert len(docs) == 0


def test_wiki_page_strips_stray_midline_double_braces_keeps_single():
    """A page whose section carries a stray mid-line `}}`/`{{` (e.g. from a
    half-stripped {{Icon|...}} shell that strip_code leaves as real text) must
    emit docs with NO `{{`/`}}` residue, while preserving legitimate single
    `{`/`}` prose — the hygiene-guard contract (_WIKI_RESIDUE) is only about
    double-brace markup residue."""
    raw = """__NOTOC__
== Summary ==
Irkima asks you to gather a phoenix feather from the far mountain valley, and
the trek is long but the phoenix feather}} is worth it, so use a knife
{carefully} when you lift it free. There is more than enough text to clear the
minimum section length without any trouble at all.
"""
    db = FakeDB({"Broken Wings": raw})
    docs = build_wiki_documents(db)
    texts = [d["text"] for d in docs]
    assert texts, "section should emit a doc"
    for t in texts:
        assert "{{" not in t and "}}" not in t
    assert any("{carefully}" in t for t in texts)
