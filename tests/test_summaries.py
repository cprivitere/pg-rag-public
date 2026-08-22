"""Contracts for build_summary_documents (CDN/computed per-skill summaries).

Owns summary doc shape {id, type, text, metadata} and grouping/ranking for
the computed-summary builder. The related gathering variant is covered in
test_gathering_summaries.py; test_documents.py covers end-to-end
build_documents() summary *supplementation* (CDN vs wiki precedence).
"""

from pgrag.documents.summaries import build_summary_documents


def _make_recipes(recipes_dict):
    return [
        {
            "id": rid,
            "type": "recipe",
            "text": f"Recipe: {r['Name']}\n\nSkill:\n{r['Skill']}\n\nRequired Skill Level:\n{r['SkillLevelReq']}",
            "metadata": {"source": "cdn", "table": "recipes", "name": r["Name"]}
        }
        for rid, r in recipes_dict.items()
    ]


def test_groups_by_skill():
    recipes = _make_recipes({
        "r1": {"Name": "Cheese A", "Skill": "Cheesemaking", "SkillLevelReq": 10},
        "r2": {"Name": "Cheese B", "Skill": "Cheesemaking", "SkillLevelReq": 20},
        "r3": {"Name": "Bread", "Skill": "Baking", "SkillLevelReq": 5},
    })
    summaries = build_summary_documents(recipes)
    skill_names = [s["metadata"]["name"] for s in summaries]
    assert "Cheesemaking Summary" in skill_names
    assert "Baking Summary" in skill_names


def test_ranks_by_level_descending():
    recipes = _make_recipes({
        "r1": {"Name": "Cheese A", "Skill": "Cheesemaking", "SkillLevelReq": 10},
        "r2": {"Name": "Cheese B", "Skill": "Cheesemaking", "SkillLevelReq": 50},
        "r3": {"Name": "Cheese C", "Skill": "Cheesemaking", "SkillLevelReq": 30},
    })
    summaries = build_summary_documents(recipes)
    cheesemaking = next(s for s in summaries if s["metadata"]["name"] == "Cheesemaking Summary")
    lines = cheesemaking["text"].strip().splitlines()
    assert "Cheese B (50)" in lines[1]
    assert "Cheese C (30)" in lines[2]
    assert "Cheese A (10)" in lines[3]


def test_cdn_summary_shape():
    recipes = _make_recipes({
        "r1": {"Name": "Cheese A", "Skill": "Cheesemaking", "SkillLevelReq": 10},
    })
    summaries = build_summary_documents(recipes)
    assert len(summaries) == 1
    doc = summaries[0]
    assert set(doc.keys()) == {"id", "type", "text", "metadata"}
    assert doc["type"] == "summary"
    assert doc["metadata"]["source"] == "computed"
    assert doc["metadata"]["table"] == "summaries"


def test_empty_recipes():
    summaries = build_summary_documents([])
    assert summaries == []


def test_single_recipe_per_skill():
    recipes = _make_recipes({
        "r1": {"Name": "Cheese A", "Skill": "Cheesemaking", "SkillLevelReq": 10},
    })
    summaries = build_summary_documents(recipes)
    assert len(summaries) == 1
    assert "1. Cheese A (10)" in summaries[0]["text"]
