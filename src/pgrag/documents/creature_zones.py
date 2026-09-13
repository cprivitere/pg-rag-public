"""Build a per-creature "where to find it" document from wiki MOB pages.

Creature pages on the wiki are structured: they carry a ``{{MOB infobox| type =
... }}`` (creature class) and zero-or-more ``{{MOB Location| area = ... }}``
blocks (spawn zones), and/or ``[[Category:<Zone> Creatures]]`` markers. These
give an authoritative creature -> spawn-zone mapping the corpus otherwise
lacks (spawn facts were previously only scattered across loot-table rows and
named-NPC Location fields).

Each creature with at least one known zone emits a ``creature_<Name>`` doc
(metadata.table "creatures", source "wiki") so a query like "where do deer
live" retrieves the zone list directly instead of drowning in Deer-skill
treasure noise.
"""

import re

__all__ = ["build_creature_zones_documents"]

# {{MOB infobox| type = Ruminant | ... }} — creature class.
_MOB_INFOBOX = re.compile(r"\{\{MOB[ \t]+infobox(.*?)\}\}", re.S)
_TYPE = re.compile(r"(?:^|\n)[ \t]*\|[ \t]*type[ \t]*=[ \t]*([^|\n]+)")

# Repeatable {{MOB Location| area = Vidaria | ... }} — spawn zones.
_MOB_LOCATION = re.compile(r"\{\{MOB[ \t]+Location(.*?)\}\}", re.S)
_AREA = re.compile(r"(?:^|\n)[ \t]*\|[ \t]*area[ \t]*=[ \t]*([^|\n]+)")

# Fallback zone source: [[Category:<Zone> Creatures]]. "Wide-Ranging Monsters"
# and other non-zone categories never match the trailing " Creatures".
_ZONE_CATEGORY = re.compile(r"\[\[Category:([^]\[\n]+) Creatures\]\]")
_WIKILINK = re.compile(r"\[\[(?:[^]|]*\|)?([^]|]+)\]\]")

# First prose paragraph (after the infobox, before any heading / MOB Location).
# The original page's lead sentence names the species ("{{Name}} are a type of
# deer ...", "... female deer", "... a breed of cattle ..."), which the
# synthetic doc would otherwise drop — and which is exactly what makes a
# "which animals, where" query recall the doc.
_TEMPLATE = re.compile(r"\{\{.*?\}\}", re.S)
_SECTION = re.compile(r"=+.*?=+", re.S)
_DESCRIPTION_MAX = 220


def _clean(value: str) -> str:
    """Resolve a raw template value to display text (drop wiki links/markup)."""
    value = _WIKILINK.sub(r"\1", value)  # [[X|Y]] -> Y ; [[X]] -> X ; [[X (Z)]] -> X (Z)
    return re.sub(r"\s+", " ", value).strip().strip("|").strip()


def _lead_paragraph(raw: str) -> str:
    """First descriptive sentence after the infobox, wikicode stripped.

    The infobox and every {{MOB Location}} template are dropped up front
    (they span multiple lines and can carry `| area =` on a later line, so
    splitting on the first `}}` or trusting one non-greedy template regex can
    leave a `{{MOB Location | area` prefix or a stray `}}` in the lead). Any
    remaining braces/pipes/bold markup are then swept — including once more
    AFTER the length truncation, so a cut mid-template cannot reintroduce
    `{{`/`}}` residue (wiki hygiene guard).
    """
    body = _MOB_INFOBOX.sub(" ", raw)
    body = _MOB_LOCATION.sub(" ", body)
    body = _TEMPLATE.sub(" ", body)  # drop remaining {{...}}
    body = _SECTION.split(body, 1)[0]  # stop at the first heading
    body = _WIKILINK.sub(r"\1", body)  # [[X|Y]] -> Y ; [[X]] -> X
    body = body.replace("'''", "")  # '''bold''' markup
    body = re.sub(r"__[A-Z]+__(?:\s*|$)", " ", body)  # __NOTOC__/__FORCETOC__
    body = re.sub(r"[{}|]+", " ", body)  # sweep stray braces/pipes
    body = re.sub(r"\s+", " ", body).strip()
    body = body[:_DESCRIPTION_MAX].strip()
    body = re.sub(r"[{}|]+", " ", body).strip()
    # A "lead" that is only category markers/booleans (e.g. a MOB page whose
    # only prose is a template row) is noise — drop it so the doc omits the
    # Description line instead of carrying "Category:Bosses".
    if not re.search(r"[A-Za-z]{2,}", body):
        return ""
    return body


def _indef_article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _extract_zones(raw: str) -> list[str]:
    """Union of MOB Location `area=` values and zone categories, deduped."""
    zones: list[str] = []
    for m in _MOB_LOCATION.finditer(raw):
        for am in _AREA.finditer(m.group(1)):
            z = _clean(am.group(1))
            if z and z not in zones:
                zones.append(z)
    for cm in _ZONE_CATEGORY.finditer(raw):
        z = _clean(cm.group(1))
        if z and z not in zones:
            zones.append(z)
    return zones


def build_creature_zones_documents(db) -> list[dict]:
    """Build one ``creature_<Name>`` doc per wiki MOB page with known zones."""
    documents = []
    pages = getattr(db, "wiki", None) or {}
    for title, raw in pages.items():
        zones = _extract_zones(raw)
        if not zones:
            continue

        ctype = ""
        box = _MOB_INFOBOX.search(raw)
        if box:
            tm = _TYPE.search(box.group(1))
            if tm:
                ctype = _clean(tm.group(1))

        join = ", ".join(zones)
        lines = [f"Creature Locations: {title}", f"Locations: {join}"]
        if ctype:
            lines.insert(1, f"Type: {ctype}")
        description = _lead_paragraph(raw)
        if description:
            lines.append(f"Description: {description}")
        if ctype:
            lines.append(
                f"{title} is {_indef_article(ctype)} {ctype} creature. It can be found in {join}."
            )
        else:
            lines.append(f"{title} can be found in {join}.")

        documents.append(
            {
                "id": "creature_" + title.replace(" ", "_"),
                "type": "wiki",
                "text": "\n".join(lines),
                "metadata": {
                    "source": "wiki",
                    "table": "creatures",
                    "name": title,
                    "creature_type": ctype,
                    "locations": " | ".join(zones),
                },
            }
        )
    return documents
