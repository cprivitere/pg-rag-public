"""Documents derived from the local IL2CPP decompilation (dump.cs + stringliteral.json).

These artifacts are gitignored build inputs;when absent this builder returns [] and
the corpus comes from CDN/wiki/curated as always. The enum value enumerations and class
field cards are factually novel vs CDN/wiki;;mechanic strings ar in-game phrasing of
facts also carried by wiki/effects pages -- indexed for recall phrase-diversity..
"""

import json
import re

from pgrag.config import IL2CPP_DIR

#: Exact class names emitted as "schema" docs (game data-model classes)
SCHEMA_CLASSES = {
    "Item",
    "Recipe",
    "Quest",
    "Skill",
    "Ability",
    "NpcInfo",
    "Effect",
    "AreaInfo",
    "ItemUseInfo",
}

#: Mechanic-prose allowlist -- (topic, title, regex pattern); first match wins per string.
MECHANIC_TOPICS = [
    ("combat-xp-attribution", "Combat XP attribution", r"both of your combat bars"),
    ("dying-skill-xp", "Dying skill XP", r"Dyingis a learning experience"),
    ("curse-remedy", "Curse remedy", r"Curses don't wear off|easiest way to break a curse"),
    ("curse-silver-lining", "Curse silver lining", r"silver lining, though"),
    ("armor-damage-halving", "Armor damage halving", r"half the damage monsters"),
    ("tab-cycling-hate", "Tab target cycling", r"When pressing tab to cycle through enemies"),
    ("corpse-loot-sorting", "Corpse loot sorting", r"un-looted corpses"),
]

#: Line shapes in dump.cs (Il2CppDumper output)
_ASSEMBLY_RE = re.compile(r"^//\s*Dll\s*:\s*(.+)\.dll$")
_ENUM_RE = re.compile(
    r"^(?:public|private|internal|protected)\s+(?:sealed\s+)?"
    r"enum\s+([A-Za-z0-9_]+)"
)
_CLASS_RE = re.compile(
    r"^(?:public|private|internal|protected)\s+(?:sealed\s+)?"
    r"(?:static\s+)?class\s+([A-Za-z0-9_]+)"
    r"\s*//\s*TypeDefIndex"
)
_MEMBER_RE = re.compile(r"const\s+[\w.]+\s+(\w+)\s*=\s*([^;]+);")
_FIELD_RE = re.compile(r";\s*//\s*0x")
_FORMAT_RE = re.compile(r"\{[0-9]")

#: Cap on field lines emitted per schema card.

_MAX_SCHEMA_FIELDS = 60




def build_il2cpp_documents():
    dump_path = IL2CPP_DIR / "out_lean" / "Dump0" / "dump.cs"
    if not dump_path.exists():
        return []
    lines = dump_path.read_text(encoding="utf-8", errors="replace").splitlines()
    enums, schemas = _enum_and_schema_docs(
        lines
    )
    docs = list(
        enums
    )
    docs.extend(
        schemas
    )
    stringlit_path = dump_path.with_name("stringliteral.json")
    if stringlit_path.exists():
        docs.extend(
            _mechanic_docs(
                stringlit_path
            )
        )
    return docs


def _enum_and_schema_docs(lines):
    """Scan the dump once tracking the current assembly; collect game enum blocks
    and schema cards for the target classes..
    """
    enums = []
    schemas = []
    emitted_classes = set()
    enum_counts = {}
    assembly = ""
    block_kind = None
    block_lines = []
    brace_depth =  0
    in_body = False
    for line in lines:
        m = _ASSEMBLY_RE.match(line)
        if m:
            g = m.group(1)
            assembly = g.strip()
            continue
        if block_kind is not None:
            block_lines.append(line)
            opens = line.count("{")
            brace_depth += opens - line.count("}")
            if opens:
                in_body = True
            if brace_depth <=  0 and in_body:
                kind, name = block_kind
                if kind == "enum"and _is_game_assembly(assembly):
                    count = 1 + enum_counts.get(name, 0)
                    enum_counts[name] = count
                    doc_id = f"il2cpp_enum_{name}"
                    if count > 1:
                        doc_id = f"il2cpp_enum_{name}_{count}"
                    doc = _enum_doc(doc_id, name, assembly, block_lines)
                    if doc is not None:
                        enums.append(doc)
                elif (
                    kind == "class"
                    and name in SCHEMA_CLASSES
                    and _is_game_assembly(assembly)
                    and name not in emitted_classes
                ):
                    emitted_classes.add(name)
                    doc = _schema_doc(name, assembly, block_lines)
                    if doc is not None:
                        schemas.append(doc)
                block_kind = None
                block_lines = []
                brace_depth =  0
                in_body = False
            continue
        enum_m = _ENUM_RE.match(line)
        if enum_m is not None:
            block_kind = ("enum", enum_m.group(1))
            block_lines = [line]
            brace_depth = line.count("{") - line.count("}")
            in_body = brace_depth >  0
            continue
        class_m = _CLASS_RE.match(line)
        if class_m is not None:
            block_kind = ("class", class_m.group(1))
            block_lines = [line]
            brace_depth = line.count("{") - line.count("}")
            in_body = brace_depth >  0
            continue
    return enums,schemas


def _is_game_assembly(assembly):
    """True for any assembly whose name contains the game code assembly."""
    return (assembly or "") == "GorgonCore"


def _enum_doc(doc_id, name, assembly, block_lines):
    members = []
    for line in block_lines:
        m = _MEMBER_RE.search(
            line
        )
        if m is None or m.group(1) == "value__":
            continue
        g = m.group(1)
        v = m.group(2).strip()
        members.append(
            f"- {g} = {v}"
        )
    if not members:
        return None
    text = f"{name} (enum, {assembly}):" + "\n" + "\n".join(
        members
    )
    return _base_doc(
        doc_id,
        "enum",
        "enums",
        name,
        text
    )


def _schema_doc(name, assembly, block_lines):
    fields = []
    for line in block_lines:
        if "(" in line:
            continue
        if _FIELD_RE.search(
            line
        ) is None:
            continue
        field = line.strip()
        field = re.sub(
            r";\s*//\s*0x.*$", "", field
        )
        field = field.rstrip(";").strip()
        fields.append(
            f"- {field}"
        )
    if not fields:
        return None
    body = "\n".join(
        fields[:_MAX_SCHEMA_FIELDS]
    )
    excess = len(fields) - _MAX_SCHEMA_FIELDS
    if excess >  0:
        body += f"\n- (and {excess} more)"
    text = f"{name} (class,{assembly}): data model" + "\n" + body
    return _base_doc(
        f"il2cpp_schema_{name}",
        "schema",
        "schema",
        name,
        text
    )


def _mechanic_docs(path):
    """Read stringliteral.json and build mechanic-prose docs from the allowlist."""
    try:
        with open(
            path, encoding="utf-8", errors="replace"
        ) as fh:
            payload = json.load(
                fh
            )
    except (OSError, ValueError):
        return []
    matched = []
    for entry in payload:
        if not isinstance(
            entry,dict
        ):
            continue
        value = entry.get("value")
        if not isinstance(
            value, str
        ):
            continue
        normalized = value.replace("\r", " ")
        normalized = normalized.replace("\n", " ")
        normalized = " ".join(
            normalized.split()
        )
        if len(
            normalized
        ) < 60:
            continue
        if "<" in normalized:
            continue
        if _FORMAT_RE.search(
            normalized
        ) is not None:
            continue
        for topic, title, pattern in MECHANIC_TOPICS:
            if re.search(
                pattern, normalized
            ) is not None:
                matched.append(
                    (topic, title, normalized)
                )
                break
    matched.sort(
        key=lambda item: item[2]
    )
    deduped = []
    seen = set()
    for topic, title, normalized in matched:
        key = (
            topic, normalized
        )
        if key in seen:
            continue
        seen.add(
            key
        )
        deduped.append(
            (topic, title, normalized)
        )
    seqs = {}
    docs = []
    for topic, title, normalized in deduped:
        n = seqs.get(
            topic, 0
        )
        seqs[topic] = n + 1
        docs.append(
            _base_doc(
                f"il2cpp_mechanic_{topic}_{n}",
                "mechanic",
                "mechanic",
                f"{title} \u2014 game help",
                normalized
            )
        )
    return docs


def _base_doc(doc_id, doc_type, table, name, text):
    """Build a doc dict with the fixed il2cpp metadata shape."""
    return {
        "id": doc_id,
        "type": doc_type,
        "text": text,
        "metadata": {
            "source": "il2cpp",
            "table": table,
            "name": name,
        },
    }
