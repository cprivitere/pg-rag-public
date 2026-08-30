#!/usr/bin/env python3
"""Analyze GorgonCore classes not in SCHEMA_CLASSES for RAG-useful data models.

Parses dump.cs, extracts every class in GorgonCore, reports:
- Class name, field names and types
- String fields containing Desc/Tooltip/Label/Name/Text/Preface/Hint/Help
- Has JSONData constructor (indicates loaded from data file)
- Cross-reference against CDN tables

Output: section appended to docs/discovered-schemas.md
"""

import json
import re
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
DUMP_PATH = REPO_DIR / "data" / "il2cpp" / "out_lean" / "Dump0" / "dump.cs"
CDN_DIR = REPO_DIR / "data" / "cdn"
DOC_PATH = REPO_DIR / "docs" / "discovered-schemas.md"

# ── current schema classes (skip these) ────────────────────────────────────
EXISTING_SCHEMA_CLASSES = {
    "Item", "Recipe", "Quest", "Skill", "Ability",
    "NpcInfo", "Effect", "AreaInfo", "ItemUseInfo",
}

# ── player-facing field patterns ──────────────────────────────────────────
PLAYER_FACING_FIELD = re.compile(
    r"(Desc|Tooltip|Label|Name|Text|Preface|Hint|Help|Title|FriendlyName|"
    r"Suffix|Prefix|SubTitle|SortTitle|Comment|Keyword|"
    r"RequirementFriendlyDescription)", re.IGNORECASE
)

# ── pattern for JSONData-like constructor ──────────────────────────────────
JSON_CTOR = re.compile(r"JSONData|FromData|LoadFromJson|Deserialize")


def parse_dump_classes(path: Path) -> list[dict]:
    """Parse GorgonCore classes from dump.cs."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    assembly_re = re.compile(r"^//\s*Dll\s*:\s*(.+)\.dll$")
    class_re = re.compile(
        r"^(?:public|private|internal|protected)\s+(?:sealed\s+)?"
        r"(?:static\s+)?class\s+([A-Za-z0-9_]+)\s*//\s*TypeDefIndex"
    )
    field_re = re.compile(r";\s*//\s*0x")
    # Detect constructors (method with class name)
    ctor_re = re.compile(
        r"^\s*(?:public|private|internal|protected)\s+([A-Za-z0-9_]+)\s*\("
    )

    classes = []
    assembly = ""
    in_class = False
    current = None
    class_lines = []
    brace_depth = 0
    in_body = False

    for line in lines:
        m = assembly_re.match(line)
        if m:
            assembly = m.group(1).strip()
            in_class = False
            continue

        if assembly != "GorgonCore":
            continue

        if in_class:
            class_lines.append(line)

            # Check for constructor
            cm = ctor_re.match(line)
            if cm and cm.group(1) == current["name"] and "{" not in line.split("//")[0]:
                current["has_ctor"] = True
                # Check if constructor references JSON data
                rest = line[line.index("("):]
                current["has_json_ctor"] = bool(JSON_CTOR.search(rest))

            opens = line.count("{")
            brace_depth += opens - line.count("}")
            if opens:
                in_body = True
            if brace_depth <= 0 and in_body:
                # End of class
                classes.append(current)
                in_class = False
                current = None
                class_lines = []
                brace_depth = 0
                in_body = False
            continue

        cm = class_re.match(line)
        if cm:
            name = cm.group(1)
            current = {
                "name": name,
                "fields": [],
                "string_fields": [],
                "player_facing_fields": [],
                "has_ctor": False,
                "has_json_ctor": False,
                "field_count": 0,
                "is_static": "static" in line,
            }
            in_class = True
            class_lines = [line]
            brace_depth = line.count("{") - line.count("}")
            in_body = brace_depth > 0

    return classes


def extract_fields(current: dict, class_lines: list[str]):
    """Extract fields from class lines."""
    field_re = re.compile(r";\s*//\s*0x")
    for line in class_lines:
        if "(" in line and "class " not in line:
            continue
        if not field_re.search(line):
            continue
        current["field_count"] += 1
        field = line.strip()
        field = re.sub(r";\s*//\s*0x.*$", "", field)
        field = field.rstrip(";").strip()
        current["fields"].append(field)

        # Check if it's a string field
        if re.search(r"\bstring\b", field.split("//")[0]):
            current["string_fields"].append(field)
            # Check if it's player-facing
            if PLAYER_FACING_FIELD.search(field):
                current["player_facing_fields"].append(field)


def find_cdn_reference(class_name: str) -> str | None:
    """Find matching CDN file for a class."""
    # Map class names to CDN filenames
    cdn_map = {
        "PlayerTitle": "playertitles.json",
        "LoreBook": "lorebooks.json",
        "LoreBookCategory": "lorebookinfo.json",
        "Landmark": "landmarks.json",
        "DirectedGoal": "directedgoals.json",
        "AdvancementTable": "advancementtables.json",
        "XPTable": "xptables.json",
        "AttributeDisplayInfo": "attributes.json",
        "AbilityDoTDisplayInfo": "abilitydynamicdots.json",
        "AbilitySpecialValue": "abilitydynamicspecialvalues.json",
        "AbilityKeywordInfo": "abilitykeywords.json",
        "AbilityConditionalKeyword": "abilitykeywords.json",
        "StorageVault": "storagevaults.json",
        "AIConfig": "ai.json",
        "NpcService": "npcs.json",
        "TSysProfile": "tsysprofiles.json",
        "TSysPower": "tsysclientinfo.json",
        "GolfCourse": None,  # No obvious CDN file
        "AreaInfo": "areas.json",
        "Item": "items.json",
        "Recipe": "recipes.json",
        "Quest": "quests.json",
        "Skill": "skills.json",
        "Ability": "abilities.json",
        "Effect": "effects.json",
        "ItemUseInfo": "itemuses.json",
        "NpcInfo": "npcs.json",
    }
    fname = cdn_map.get(class_name)
    if fname is None:
        return None
    path = CDN_DIR / fname
    if path.exists():
        return fname
    return None


def check_cdn_fields(class_name: str, player_fields: list[str]) -> tuple[list[str], list[str]]:
    """Check which player-facing fields already exist in CDN data.
    Returns (present_in_cdn, missing_from_cdn).
    """
    cdn_map = {
        "PlayerTitle": "playertitles.json",
        "LoreBook": "lorebooks.json",
        "Landmark": "landmarks.json",
        "DirectedGoal": "directedgoals.json",
        "StorageVault": "storagevaults.json",
        "AIConfig": "ai.json",
    }
    fname = cdn_map.get(class_name)
    if fname is None:
        return [], []

    path = CDN_DIR / fname
    if not path.exists():
        return [], []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Get keys from first entry
    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        cdn_keys = set(first_val.keys()) if isinstance(first_val, dict) else set()
    elif isinstance(data, list) and data:
        cdn_keys = set(data[0].keys())
    else:
        return [], []

    # Extract field names from player-facing field lines
    field_names = set()
    for field in player_fields:
        parts = field.split()
        if len(parts) >= 3:
            name = parts[2].rstrip(";{}")
            field_names.add(name)

    missing = [f for f in field_names if f not in cdn_keys]
    present = [f for f in field_names if f in cdn_keys]
    return present, missing


def main():
    print(f"Parsing {DUMP_PATH}...")
    classes = parse_dump_classes(DUMP_PATH)

    # Remove existing schema classes
    skipped = [c for c in classes if c["name"] in EXISTING_SCHEMA_CLASSES]
    candidates = [c for c in classes if c["name"] not in EXISTING_SCHEMA_CLASSES]

    # Extract fields for each class
    # We need to re-parse to get class_lines - let's do a simpler approach
    # by reading the file again
    with open(DUMP_PATH, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    assembly_re = re.compile(r"^//\s*Dll\s*:\s*(.+)\.dll$")
    class_re = re.compile(
        r"^(?:public|private|internal|protected)\s+(?:sealed\s+)?"
        r"(?:static\s+)?class\s+([A-Za-z0-9_]+)\s*//\s*TypeDefIndex"
    )

    assembly = ""
    in_class = False
    current = None
    class_lines = []
    brace_depth = 0
    in_body = False
    candidate_map = {c["name"]: c for c in candidates}

    for line in raw_lines:
        m = assembly_re.match(line)
        if m:
            assembly = m.group(1).strip()
            in_class = False
            continue

        if assembly != "GorgonCore":
            continue

        if in_class:
            class_lines.append(line)
            opens = line.count("{")
            brace_depth += opens - line.count("}")
            if opens:
                in_body = True
            if brace_depth <= 0 and in_body:
                if current and current["name"] in candidate_map:
                    extract_fields(current, class_lines)
                in_class = False
                current = None
                class_lines = []
                brace_depth = 0
                in_body = False
            continue

        cm = class_re.match(line)
        if cm:
            name = cm.group(1)
            if name in candidate_map:
                current = candidate_map[name]
                in_class = True
                class_lines = [line]
                brace_depth = line.count("{") - line.count("}")
                in_body = brace_depth > 0

    # ── output ─────────────────────────────────────────────────────────
    # Terminal report
    print(f"\n{'=' * 70}")
    print(f"GorgonCore CLASS ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total: {len(classes)} classes")
    print(f"Already in SCHEMA_CLASSES: {len(skipped)} ({', '.join(c['name'] for c in skipped)})")
    print(f"Candidates (not yet extracted): {len(candidates)}\n")

    # Sort: classes with player-facing fields first
    candidates.sort(key=lambda c: -len(c["player_facing_fields"]))

    print(f"{'─' * 70}")
    print(f"CLASSES WITH PLAYER-FACING TEXT FIELDS")
    print(f"{'─' * 70}")
    for c in candidates:
        if not c["player_facing_fields"]:
            continue
        cdns = find_cdn_reference(c["name"])
        cdn_str = f"CDN: {cdns}" if cdns else "CDN: ?"

        # Check CDN coverage
        present, missing = [], []
        if cdns:
            present, missing = check_cdn_fields(c["name"], c["player_facing_fields"])

        print(f"\n  {c['name']} ({len(c['fields'])} fields, {cdn_str})")
        print(f"    Static: {c['is_static']}, Has ctor: {c['has_ctor']}, JSON ctor: {c['has_json_ctor']}")
        print(f"    String fields: {len(c['string_fields'])}")
        print(f"    Player-facing fields ({len(c['player_facing_fields'])}):")
        for f in c["player_facing_fields"]:
            field_name = f.split()[-1].rstrip(";{}") if len(f.split()) >= 3 else f
            in_cdn = "✓ CDN" if field_name in present else ("✗ MISSING" if field_name in missing else "?")
            print(f"      - {f.split('//')[0].strip():55s} {in_cdn}")

    print(f"\n{'─' * 70}")
    print(f"CLASSES WITHOUT PLAYER-FACING TEXT FIELDS ({len([c for c in candidates if not c['player_facing_fields']])})")
    print(f"{'─' * 70}")
    for c in candidates:
        if c["player_facing_fields"]:
            continue
        cdns = find_cdn_reference(c["name"])
        cdn_str = f"CDN: {cdns}" if cdns else "CDN: ?"
        print(f"  {c['name']:35s} ({c['field_count']:2d} fields, {cdn_str})")

    # ── write docs ────────────────────────────────────────────────────
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing doc or create
    existing = ""
    if DOC_PATH.exists():
        existing = DOC_PATH.read_text(encoding="utf-8")

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("# Discovered Schemas — GorgonCore Classes\n\n")
        f.write(f"Generated by `scripts/analyze_schemas.py`\n")
        f.write(f"GorgonCore total: {len(classes)} classes\n")
        f.write(f"Already extracted (SCHEMA_CLASSES): {len(skipped)}\n")
        f.write(f"Candidates surveyed: {len(candidates)}\n\n")

        f.write("## Classes with Player-Facing Text\n\n")
        f.write("| Class | Fields | Player-Facing Fields | CDN | CDN Covers? |\n")
        f.write("|-------|--------|---------------------|-----|-------------|\n")
        for c in candidates:
            if not c["player_facing_fields"]:
                continue
            cdns = find_cdn_reference(c["name"])
            cdn_str = cdns or "None found"
            present, missing = [], []
            if cdns:
                present, missing = check_cdn_fields(c["name"], c["player_facing_fields"])
            cdn_covers = "✓" if not missing and present else f"Missing: {', '.join(missing)}" if missing else "?"
            field_names = ", ".join(f.split()[-1].rstrip(";{}") for f in c["player_facing_fields"])
            f.write(f"| {c['name']} | {c['field_count']} | {field_names} | {cdn_str} | {cdn_covers} |\n")

        f.write("\n## Classes Without Player-Facing Text\n\n")
        f.write("| Class | Fields | CDN | Notes |\n")
        f.write("|-------|--------|-----|-------|\n")
        for c in candidates:
            if c["player_facing_fields"]:
                continue
            cdns = find_cdn_reference(c["name"])
            cdn_str = cdns or "None found"
            notes = []
            if c["is_static"]: notes.append("static")
            if c["has_json_ctor"]: notes.append("JSON ctor")
            f.write(f"| {c['name']} | {c['field_count']} | {cdn_str} | {', '.join(notes)} |\n")

        f.write("\n## Recommended SCHEMA_CLASSES Additions\n\n")
        f.write("<!-- Manually reviewed -- fill after reviewing output -->\n")
        f.write("| Class | Reason | CDN Coverage | Add? |\n")
        f.write("|-------|--------|-------------|------|\n")

        # Auto-recommend based on analysis
        recommendations = []
        for c in candidates:
            if not c["player_facing_fields"]:
                continue
            cdns = find_cdn_reference(c["name"])
            present, missing = [], []
            if cdns:
                present, missing = check_cdn_fields(c["name"], c["player_facing_fields"])
            # Recommend if there are MISSING fields (not in CDN)
            if missing or not cdns:
                recommendations.append((c, missing, cdns))

        for c, missing, cdns in recommendations:
            cdn_status = f"CDN missing: {', '.join(missing)}" if missing else f"No CDN reference found"
            f.write(f"| {c['name']} | {len(c['player_facing_fields'])} player-facing fields | {cdn_status} | \n")

        f.write("\n## Assembly-CSharp Survey\n")
        f.write("\n<!-- Filled manually in Step 3 -->\n")

    print(f"\nReport written to {DOC_PATH}")
    print(f"Candidate classes with player-facing fields: {sum(1 for c in candidates if c['player_facing_fields'])}")
    print(f"Recommended for addition: {len(recommendations)}")
    print("\nNext step: review recommendations and update SCHEMA_CLASSES in decomp_builder.py")


if __name__ == "__main__":
    main()