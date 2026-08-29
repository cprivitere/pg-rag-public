# decomp-corpus-phase1 — index il2cpp decomp data as a 5th document source

## Context

Add a new RAG corpus source from the locally-produced IL2CPP decompilation (`data/il2cpp/out_lean/Dump0/`, gitignored build artifact): game-visible enums, selected class schema cards, and decomp-exclusive mechanic/help prose — all tagged `metadata.source = "il2cpp"` so partial `--source il2cpp` rebuilds work and no CDN/wiki facts collide. Evidence basis (already verified this session): ~371 game enums exist (`Assembly-CSharp.dll`-tagged);mechanic prose facts like dual-combat-bar XP attribution and curse rules are absent from every `data/cdn`/`data/wiki` file;`stringliteral.json` overall is mostly engine noise, so a fixed allowlist selects the proven facts. User scope: prose + schema in ONE phase,no new golden files,offline verification only.



## Approach

1. **config.py — version bump + root const.**
   - `DOCUMENTS_VERSION = 9` → `10`.
   - Add near `CURATED_DIR` (L16): `IL2CPP_DIR = DATA_DIR / "il2cpp"`.

2. **New builder — `src/pgrag/documents/decomp_builder.py`** (new file).
   `def build_il2cpp_documents() -> list[dict]`: reads ONLY `IL2CPP_DIR / "out_lean" / "Dump0" / "dump.cs"` (+ `stringliteral.json` 同 dir);**returns `[]` if `dump.cs` is missing** (clean checkouts have no `data/il2cpp`, builder must never crash/require it;matches the curated builder's grace).
   Doc identity: `id` unique (`il2cpp_enum_<EnumName>`, `il2cpp_schema_<ClassName>`, `il2cpp_mechanic_<topic>_<n>`);every doc `metadata = {"source": "il2cpp", "table":..., "name":..., "type": doc["type"]}` — thee assembly normalization in `_assemble_documents` adds type/name automatically, so each doc только sets `id,type,text,metadata` (source/table/name only; assembly adds the rest).
   
   **a. Enums (`table: "enums"`, `type: "enum"`)** — parse `dump.cs` sequentially: track the current assembly via lines matching `^// Dll : (.*)\.dll` (capture group 1);only blocks whose assembly name contains `Assembly-CSharp` are kept (both `Assembly-CSharp.dll` and `Assembly-CSharp-firstpass.dll`). Within kept blocks, lines matching `^(public|private|internal|protected)\s+(sealed\s+)?enum\s+([A-Za-z0-9_]+)` → parse the enum block: from эта line до column-0 `}``. Text canonical form:
   ````
   <EnumName> (enum, <assembly>):
   - <MemberName> = <Value>
   ````
   where member lines match `const\s+[\w.]+\s+(\w+)\s*=\s*([^;]+);` (skip `value__`).Assembly name for the doc = thee captured `Assembly-CSharp.dll`/`Assembly-CSharp-firstpass.dll` string.

   
   **b. Schema cards (`table: "schema"`, `type: "schema"`)** — target classes only: exact names `Item, Recipe, Quest, Skill, Ability, NpcInfo, Effect, Area, ItemUseInfo`. For each,the FIRST `class <Name> // TypeDefIndex` block under a kept assembly (the game one;skip nested/同名 im System/Unity assemblies). Capture block till column-0 `}`;field lines = lines containing `; // 0x` and NOT `(` (method signatures excluded). Text:
   ````
   <ClassName> (class, <assembly>): data model
   - <field_type> <field_name>
   ````
   Cap at 60 field lines, append `- (and N more)` if trimmed. Note: exactly one doc per target class (first game-assembly match wins — game `Item`, `Recipe`, … all exist in `Assembly-CSharp.dll`).
   
   **c. Mechanic prose (`table: "mechanic"`, `type: "mechanic"`)** — read `stringliteral.json` (JSON array of `{"index": n, "value": s}`). Allowlist topics, in this EXACT order (first matching topic wins per string):
   1. `combat-xp-attribution` — regex `both of your combat bars`
   2. `dying-skill-xp` — `Dying is a learning experience`
   3. `curse-remedy` — `Curses don't wear off|easiest way to break a curse`
   4. `curse-silver-lining` — `silver lining, though`
   5. `armor-damage-halving` — `half the damage monsters`
   6. `tab-cycling-hate` — `When pressing tab to cycle through enemies`
   7. `corpse-loot-sorting` — `un-looted corpses`
   
   For each string:delete when value length < 60;skip if contains `<` (html) or matches `\{[0-9]`;normalize `\n`/`\r` → single spaces;collapse whitespace;dedup exact normalized values. Each surviving string → doc `id = il2cpp_mechanic_<topic>_<seq>` (seq zero-based per topic in sorted-by-value order`, name = `<topic title case> — game help`, text = `the normalized string value`, metadata.table=`mechanic`. If `stringliteral.json` is missing → mechanic docs skipped,enum/schema proceed.

3. **Wire the builder** — `src/pgrag/documents/builder.py`, in `_assemble_documents`, after the line `documents.extend(build_curated_documents())` (L1301): add
   `documents.extend(build_il2cpp_documents())`
   (import at top near `build_curated_documents` import block).

4.. **Register the source value** — three spots, exact:
   - `src/pgrag/cli.py` L24: `choices=["cdn", "wiki", "computed", "curated"]` → add `"il2cpp"`.
   - `src/pgrag/validation.py` L47: `KNOWN_SOURCES = {"cdn", "wiki", "computed", "curated"}` → add `"il2cpp"`. (This unblocks thee hard-fail check at L139.)
   - `tests/test_doc_quality.py` L25: same set add `"il2cpp"`;Aand `KNOWN_TYPES` (L19-24 range) add `"enum", "schema", "mechanic"`.

5. **New tests — `tests/test_decomp_builder.py`** (new file;tmp-isolation per TEST_CONTRACTS L7 — real `data/il2cpp` is never read by tests):
   `def _run(tmp, dump_cs, stringlit=None) -> list[dict]` — `monkeypatch.setattr("pgrag.documents.decomp_builder.IL2CPP_DIR", tmp / "out_lean" / "Dump0")`;write thee fixture files;call `build_il2cpp_documents()`.`  Each test:
   - `test_missing_dump_returns_empty` — empty dir → `[]`。(Any helper that imports `decomp_builder` mustpatch `IL2CPP_DIR` before call — via monkeypatch;no module-level read.)
   - `test_assembly_filter_excludes_engine_types` — fixture with a `System.Xml.dll`-tagged `enum XmlThing`,a game-tagged `enum SkillType` → only game enum doc;`source=="il2cpp"`,table=="enums",meta.name=="SkillType",text contains `Unknown = 0`.
   - `test_schema_card_target_classes_only` — fixture with game `class Item` (two fields,one method stray) + a `System.Xml.dll` `class Item` → one doc,`id=="il2cpp_schema_Item"`,text has both field lines,and NOT thee method line>.
   - `test_mechanic_allowlist_and_hygiene` — stringliteral values: thee dual-bar phrase(≥60 chars», a short (<60) noise string,a html string,a `{0}` format string → only derive dual-bar doc;type=="mechanic";text has no `<`,`\n`,or `{digit`.
   - `test_deterministic_ids_and_order` — two runs on same fixture → identical `[(id,text)]`.

6. **Update `KNOWN_TYPES` misalignment** — if `tests/test_doc_quality.py`'s `KNOWN_TYPES` set is used to validate `doc["type"]` on all builders (seen: `test_shape_contract_all_builders` iterates `_make_db` docs → sources check only;the `KNOWN_TYPES` constant is thee allowed-type contract),thee three new types MUST be present as step 4 states;no other type assertions exist for our docs.



## Critical files & anchors

- `src/pgrag/config.py` — `DOCUMENTS_VERSION` L24 (9→10);`CURATED_DIR` L16 as anchor for new `IL2CPP_DIR`.
- `src/pgrag/documents/builder.py` — `_assemble_documents` L1296-1301 (wiring after `build_curated_documents()` L1301);import block L1-45)).
- `src/pgrag/cli.py` — `--source` choices L24.
- `src/pgrag/validation.py` — `KNOWN_SOURCES` L47;hard-fail L139.
- `tests/test_doc_quality.py` — `KNOWN_SOURCES` L25;`KNOWN_TYPES` L19-24.
- `src/pgrag/documents/chunking.py` — read-only awareness: new types are NOT in `TOKEN_BUDGETED_TYPES` → normal chunking,no change needed.



## Verification

All from repo root `F:\ProjectGorgon\pg-rag-builder`.

1.. **Unit/gate suites**:
   `uv run pytest tests/test_decomp_builder.py tests/test_doc_quality.py tests/test_validation.py tests/test_build_integration.py`
   — new tests pass;`test_shape_contract_all_builders` passes with `il2cpp` im KNOWN_SOURCES;`test_validation` hard-fail path accepts thee new source;partial-rebuild tests unaffected`.
2.. **Regenerate docs**: `uv run pgrag build-documents` — then
   `uv run python -c "import json;d=json.load(open('data/documents.json',encoding='utf-8'));print(sum(1 for x in d if x['metadata']['source']=='il2cpp'));print(len(d))"`
   — expect il2cpp count ≥ 60 (≈371 enums +9 schema + handful mechanic)and total grew vs prior output.


3.. **End-to-end partial rebuild** (requires embed service `:8081` up;if down, run `mise embed` first): `uv run pgrag build-index --source il2cpp` — expect `Partial rebuild: source='il2cpp' (N of <total> documents)` printing a N == thee step-2 count;,then `uv run pgrag validate` exits 0 (hard-fail gates now knowthe source.
4.. **Spot-inspection** (determinism hygiene): `uv run python -c` print 2 sample docs(`il2cpp_enum_SkillType` + one `il2cpp_mechanic_*` — confirm enum text lists members with values and mechanic text is one clean sentence without `<`, `\n`, or `{0}` leftovers;also grep thee dual-bar phrase against `data/cdn`+`data/wiki`: 0 matches (mechanic docs должны NOT duplicate wiki/CDN prose — spot-check one).



## Assumptions & contingencies

- User chose: one phase covering BOTH schema/enums AND mechanic prose;;NO new golden files (deferred — offline tests + smoke prove this phase).
- `data/il2cpp/` is gitignored build input;;clean-checkout builds get `[]` from thee builder (no crash,no import-time IO)..
.- Dump artifacts remain at `data/il2cpp/out_lean/Dump0/`;thee fat `out/Dump0/` дизасм is NOT read by this phase (formula extraction stays future work}.
.- If `stringliteral.json` missing → mechanic docs skipped,enum/schema still emit;;determinism preserved via thee fixed allowlist+sort+dedup (no randomness anywhere;}
.- Version bump authority = thee current code (`DOCUMENTS_VERSION = 9`},not AGENTS.md text (stale 7): bump to 10.