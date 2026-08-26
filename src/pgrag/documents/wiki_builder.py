import json
import re

import mwparserfromhell

from pgrag.config import WIKI_PARSED_CACHE

MIN_SECTION_CHARS = 50

CACHE_FILE = WIKI_PARSED_CACHE

# Bump when the cached doc shape OR generated text changes, so stale cache
# entries are rebuilt instead of served with old content (a mtime-equal page
# with a changed section-cleanup step would otherwise keep residue forever).
CACHE_VERSION = 4

# CDN tables whose entity names wiki pages can link to.
_ENTITY_TABLES = {
    "items": "item",
    "recipes": "recipe",
    "abilities": "ability",
    "skills": "skill",
    "quests": "quest",
    "npcs": "npc",
    "areas": "area",
    "effects": "effect",
}


def _norm_name(value):
    return " ".join(str(value).lower().split())


def _build_entity_index(db):
    """name -> (entity doc id, entity type) over CDN entity tables.

    Matches wiki page names against entity Names (underscores count as
    spaces). Misses are fine — not every page is an entity page.
    """
    index = {}
    for table, etype in _ENTITY_TABLES.items():
        for key, record in db.tables.get(table, {}).items():
            if not isinstance(record, dict):
                continue
            name = record.get("Name")
            if not name:
                continue
            if table in ("items", "recipes"):
                entity_id = key
            else:
                entity_id = f"{etype}_{key}"
            for variant in {
                _norm_name(name),
                _norm_name(name.replace(" ", "_")),
            }:
                index.setdefault(variant, (entity_id, etype))
    return index

# Templates whose first argument is a meaningful name that gets
# destroyed by strip_code().  We rewrite them to plain text first.
_TEMPLATE_PATTERN = re.compile(
    r"\{\{(Item|NPC|Quest|Skill|Area|Recipe|LoreBook|Ability)"
    r"\|([^}|]+)(?:\|[^}]*)?\}\}"
)


def _preserve_template_names(wikicode_text):
    """Replace {{Template|Name|...}} with just Name before stripping."""
    return _TEMPLATE_PATTERN.sub(r"\2", wikicode_text)


# Cell-level markup cleanup for MediaWiki table cells. Template name
# preservation runs first so {{Item|Parasol Mushroom}} -> "Parasol Mushroom",
# then wiki links, HTML tags and leftover template shells are dropped.
def _clean_cell(raw):
    c = _preserve_template_names((raw or "").strip())
    c = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", c)
    c = re.sub(r"<[^>]+>", "", c)
    c = c.replace("'''", "").replace("''", "")
    c = c.replace("&nbsp;", " ")
    c = re.sub(r"\{\{[^{}]*\}\}", "", c)
    return re.sub(r"\s+", " ", c).strip()


def _parse_table(tab, idx, display, safe, base_meta):
    """A parsed MediaWiki table node -> coverage + one row record per row.

    Rows come from the real <tr> structure (templates stay intact inside
    cells), so a single-cell layout wrapper holding an {{... infobox}} is
    NOT mistaken for data rows. All-<th> header rows are consumed but not
    emitted.
    """
    data_rows = []
    for tr in tab.contents.nodes:
        if (getattr(tr, "__class__", None).__name__ != "Tag"
                or tr.tag != "tr"):
            continue
        cells = []
        th_only = True
        for cell in tr.contents.nodes:
            cls = getattr(cell, "__class__", None).__name__
            if cls != "Tag" or cell.tag not in ("th", "td"):
                continue
            if cell.tag != "th":
                th_only = False
            cells.append(_clean_cell(str(cell.contents)))
        if cells and not th_only:
            # A single cell that is still an unresolved template is a layout
            # wrapper (e.g. a `{| width=100%` holding `{{Ability infobox}}`),
            # not a data row. Meaningful {{Item|X}} cells were already
            # cleaned by _clean_cell, so a leftover `{{` means an infobox.
            if len(cells) == 1 and "{{" in cells[0]:
                continue
            data_rows.append(cells)

    if not data_rows:
        return []

    table_id = f"{safe}_table_{idx}"
    section = base_meta.get("section")
    records = []

    first_cells = [cells[0] for cells in data_rows if cells[0]]
    cov = f"{display} table covers: " + ", ".join(first_cells)
    if len(cov) > 950:
        cov = cov[:950]
    records.append({
        "id": f"{table_id}_coverage",
        "type": "wiki",
        "text": cov,
        "metadata": dict(
            base_meta, table_id=table_id, table_record="coverage",
            section=section,
        ),
    })

    for r, cells in enumerate(data_rows):
        row_key = cells[0]
        text = f"{display} table row: " + " | ".join(cells)
        if len(text) > 950:
            text = text[:950]
        records.append({
            "id": f"{table_id}_row_{r}",
            "type": "wiki",
            "text": text,
            "metadata": dict(
                base_meta, table_id=table_id, table_record="row",
                row_key=row_key, section=section,
            ),
        })

    return records


def _is_table_node(node):
    return (getattr(node, "__class__", None).__name__ == "Tag"
            and node.tag == "table")


def _parse_page(page_name, raw_text, entity_info=None):
    documents = []
    seen_ids = set()

    metadata = {
        "source": "wiki",
        "table": "wiki",
        "name": page_name.replace("_", " "),
        # Links every section chunk of the page to the page's lead doc.
        "parent_id": f"wiki_{page_name}",
    }
    if entity_info is not None:
        metadata["entity_id"], metadata["entity_type"] = entity_info

    wikicode = mwparserfromhell.parse(raw_text)
    sections = wikicode.get_sections(
        levels=[2], include_lead=True
    )

    table_offset = 0
    display = page_name.replace("_", " ")
    safe = "wiki_" + "".join(
        c if c.isalnum() else "_" for c in page_name
    )
    for section in sections:
        heading = ""
        for h in section.filter_headings():
            h_clean = mwparserfromhell.parse(
                str(h.title)
            ).strip_code(
                normalize=False, collapse=True
            ).strip()
            heading = h_clean
            break

        # Pull this section's top-level tables out before the narrative path.
        # A table that yields no records (e.g. an all-template layout wrapper)
        # is left in place so its infobox content survives as before.
        tables = [
            node for node in section.nodes if _is_table_node(node)
        ]
        table_records = []
        removed = 0
        for tab in tables:
            recs = _parse_table(
                tab, table_offset + removed, display, safe, metadata
            )
            if not recs:
                continue
            table_records.extend(recs)
            section.remove(tab)
            removed += 1
        table_offset += removed

        for rec in table_records:
            rec_id = rec["id"]
            if rec_id in seen_ids:
                continue
            seen_ids.add(rec_id)
            documents.append(rec)

        text = _preserve_template_names(str(section))
        text = mwparserfromhell.parse(text).strip_code(
            normalize=False, collapse=True
        ).strip()

        # mwparserfromhell glitch: unclosed ''' before a == heading leaves
        # preceding template shells intact (B16). A half-stripped template
        # (e.g. a nested/{{Icon}} shell) can also leave a stray trailing
        # `}}` mid-line that the open/close-only sweeps below miss. After
        # strip_code any surviving brace is markup residue, so drop them all
        # outright — no wiki doc may carry `{{`/`}}` (hygiene guard).
        text = re.sub(r"[{}]+", " ", text).strip()

        if not text or len(text) < MIN_SECTION_CHARS:
            continue

        if text.startswith("__NOTOC__"):
            continue

        doc_id = f"wiki_{page_name}"
        if heading:
            safe_heading = (
                heading
                .replace(" ", "_")
                .replace("/", "_")
                .replace("&", "and")[:80]
            )
            doc_id = f"wiki_{page_name}_{safe_heading}"

        if doc_id in seen_ids:
            counter = 2
            while f"{doc_id}_{counter}" in seen_ids:
                counter += 1
            doc_id = f"{doc_id}_{counter}"
        seen_ids.add(doc_id)

        documents.append({
            "id": doc_id,
            "type": "wiki",
            "text": text,
            "metadata": dict(metadata, section=heading if heading else None),
        })

    return documents


def _load_cache():
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        if cache.get("__version") != CACHE_VERSION:
            return {}
        return cache
    return {}


def _save_cache(cache):
    cache["__version"] = CACHE_VERSION
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )


def build_wiki_documents(db):
    entity_index = _build_entity_index(db)

    def _parse_with_entity(page_name, raw_text):
        entity_info = entity_index.get(_norm_name(page_name))
        return _parse_page(page_name, raw_text, entity_info)

    if not hasattr(db, "wiki_mtimes"):
        documents = []
        for page_name, raw_text in db.wiki.items():
            documents.extend(_parse_with_entity(page_name, raw_text))
        return documents

    cache = _load_cache()
    changed = set()

    documents = []
    for page_name, raw_text in db.wiki.items():
        mtime = db.wiki_mtimes.get(page_name)
        cached = cache.get(page_name)

        if cached is not None and cached.get("mtime") == mtime:
            documents.extend(cached["docs"])
            continue

        docs = _parse_with_entity(page_name, raw_text)
        cache[page_name] = {"mtime": mtime, "docs": docs}
        changed.add(page_name)
        documents.extend(docs)

    for page_name in list(cache):
        if page_name not in db.wiki:
            del cache[page_name]
            changed.add(page_name)

    if changed:
        _save_cache(cache)

    return documents