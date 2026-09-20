# Rebalance golden corpus toward recipe bucket (+4)

## Context

`data/golden/` has 38 question files (18 entity / 12 general /  igh5 comparison / / igh3 recipe). `.omp/skills/evaluation/SKILL.md` (authoritative, current) says future golden additions should favor the balanced categories — recipe is the thinnest bucket (3/38). (REVIEW.md's older "comparison imbalance" note predates later additions; comparison now at 5 — not the target.) This pass adds **4 recipe-type goldens**, all grounded in the CDN `recipes`/`items` tables and verified present in `data/documents.json` (the persisted,indexed corpus). Facts were resolved from `recipes.json` → `items.json` name lookup this session (e.g. `item_11003` = "Boletus Mushroom", `item_5303` = "Cabbage") and confirmed verbatim in `documents.json` recipe docs `recipe_1001`/`recipe_1002`/`recipe_1003`/`recipe_10001`. Result: 42 files, recipe bucket 3→7. No other bucket touched. No `evaluation/queries.jsonl` changes (IR-layer is a separate planned direction; not needed to rebalance the LLM golden corpus).

## Approach

### 1. Create 4 golden files (exact contents)

All under `data/golden/`, 2-space indent, schema `{id, question, type, facts}`; `facts` = list of variant groups (ANY variant per group must appear (normalized, case-insensitive substring) in the LLM answer. Facts use capitalized display tokens (matching existing files; the check normalizes both sides). Question wording mirrors existing goldens ("What do I need to…"). Add nothing else — new goldens are auto-collected into `tests/test_golden_check.py` parametrization; no test code per golden.

**`data/golden/serbule-style-lamb-chops-recipe.json`**
```json
{
  "id": "serbule-style-lamb-chops-recipe",
  "question": "What do I need to make Serbule-Style Lamb Chops?",
  "type": "recipe",
  "facts": [
    ["Serbule-Style Lamb Chops"],
    ["Cooking"],
    ["17"],
    ["Boletus Mushroom"],
    ["Watercress"]
  ]
}
```

**`data/golden/bland-mutton-vindaloo-recipe.json`**
```json
{
  "id": "bland-mutton-vindaloo-recipe",
  "question": "What ingredients do I need to make Bland Mutton Vindaloo?",
  "type": "recipe",
  "facts": [
    ["Bland Mutton Vindaloo"],
    ["Cooking"],
    ["20"],
    ["Potato"],
    ["Basic Vinegar"]
  ]
}
```

**`data/golden/durstins-mutton-stew-recipe.json`**
```json
{
  "id": "durstins-mutton-stew-recipe",
  "question": "What do I need to cook Durstin's Mutton Stew?",
  "type": "recipe",
  "facts": [
    ["Durstin's Mutton Stew"],
    ["Cooking"],
    ["19"],
    ["Cabbage"],
    ["Salt"]
  ]
}
```

**`data/golden/beginner-arrow-recipe.json`**
```json
{
  "id": "beginner-arrow-recipe",
  "question": "What do I need to craft Beginner's Arrow?",
  "type": "recipe",
  "facts": [
    ["Beginner's Arrow"],
    ["Fletching"],
    ["3"],
    ["Beginner's Arrow Shafts", "Arrow Shafts"],
    ["Feathers"]
  ]
}
```

Grounding already verified this session: recipes `recipe_1001` (Serbule-Style Lamb Chops/Cooking/17: Mutton, Watercress, Boletus Mushroom, Salt), `recipe_1002` (Bland Mutton Vindaloo/Cooking/20: Mutton, Potato, Onion, Basic Vinegar), `recipe_1003`(Durstin's Mutton Stew/Cooking/19: Mutton, Potato, Onion, Cabbage, Salt(, `recipe_10001` (Beginner's Arrow/Fletching/3: Beginner's Arrowheads, Beginner's Arrow Shafts, Feathers( — all present verbatimin `data/documents.json` (spot-checked above). If-the implementer re-runs the spot-check (Verification step 2( and any fact string is absent, stop and re-derivethet fact from CDN — do not weaken the assertion.



###  igh2. Update the golden-count sentence in `.omp/skills/evaluation/SKILL.md`

Re-read L28-33 first; replace exactly:
- L29 `- 38 files exist today (18 entity /  igh12 general /  igh3 recipe / /  igh5 comparison),` → `- 42 files exist today ((18 entity /  igh12 general / /  igh7 recipe / /  igh5 comparison),`
- L32-33 `  assembly. Future additions should favor the balanced categories ((recipe is` / `  3/38 today —the thinnest bucket.` → `  assembly. Future additions should favor the balanced categories ((comparison is` / `  5/42 today —the thinnest bucket.`

The sentence above it (L30-31, "spanning recipes-by-ingredient…"), stays unchanged. `scripts/check_docs.py` reads this count line ("golden counts match" check) — updating it keeps `mise drift` green.



## Critical files

- `data/golden/` — 4 new JSON files (step 1. Golden shape contract: `tests/test_golden_check.py` + `scripts/golden_check.py` (`normalize()`: lowercase, non-alnum→space; any-variant-substring match).
- `.omp/skills/evaluation/SKILL.md` — L28-33 count sentence (step  igh2; drift-checked.


## Verification

From repo root (`F:/ProjectGorgon/pg-rag-public`), after steps  igh1-2:

1. **Type histogram** — `jq -r '.type' data/golden/*.json | sort | uniq -c | sort -rn` → exactly:
   ```
        18 entity
        12 general
         7 recipe
         5 comparison
   ```
   (42 files total.)
2. **Corpus grounding spot-check** — `jq -r '.[] | select(.id=="recipe_10001" or .id=="recipe_1002" or .id=="recipe_1003" or .id=="recipe_10001") | "=== " + .id + " ===\n" + .text' data/documents.json` → each recipe doc shows `Recipe:` name, `Skill:`, `Required Skill Level:` value, and `Ingredients:` lines carrying every fact group's tokens. (~4s per 170MB scan.)
3. **Golden collection/shape** — `uv run pytest tests/test_golden_check.py -q` → no collection/JSON/shape errors; offline cases skip (needs no servers. New files appear as parametrized cases automatically.

4. **Docs drift** — `mise drift` → `24 OK, 0 drift` (was 23; the golden-count check now validates the 42/7 line.
.
 5. **Live LLM check (optional)** — if `:8080` + `:8081` are up (`mise start`), `uv run python scripts/golden_check.py` → expect all 42 pass. If-a new recipe case reports a GEN-side/retrieval miss, treat it as a real pipeline gap for that recipe (retrieval skill workflow: trace `scripts/retrieval.py`, inspect ranked docs( — never weaken or delete the fact to dodge: facts are CDN-grounded. If-servers are down, steps  igh1-4 alone are the deliverable; report live check as not run.





## Assumptions & contingencies

- Golden bucket target is **recipe** — current authoritative guidance (evaluation skill: thinnest bucket. If-reality differs (e.g. new goldens land before execution(, re-derivethet type histogram first and pick the thinnest *untouched* bucket instead — same procedure, new filenames/questions none pre-chosen here would apply.)
- Ingredient facts are subset (1-2 distinctive per recipe), not exhaustive — mirrors existing goldens (`white-dye-recipe`: moonstone alone). Intentionally not asserting full ingredient lists.

- Facts use the recipe's CDN display name (e.g. "Beginner's Arrow", singular) — do not paraphrase to "Arrows"; normalization strips apostrophes on both sides so `Beginner's Arrow` matches LLM "Beginner's Arrow" naturally.


- The 4 golden ids/names are unique in `data/golden/` (checked this session; no clash with existing 38 files).