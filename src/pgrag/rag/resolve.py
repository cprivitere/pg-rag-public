"""Bounded wiki-parent (sibling chunk) expansion for the gap-fill path.

The deferred "parent-child / request more" retrieval: when an answer comes
back "missing", the already-retrieved wiki chunks often put the target fact
in a *sibling* chunk of the same page, not in a freshly re-retrieved subject.
`expand_parents` pulls those siblings so the model can re-answer from the
full page. It is deliberately bounded (one round, a few pages, a char cap)
and deterministic — no LLM tool-calling, no corpus rebuild.
"""
import re

from pgrag.rag.bm25 import load_bm25_index

# Wiki page expansion bounds: at most these pages and chars of appended text.
EXPAND_MAX_PAGES = 2
EXPAND_MAX_CHARS = 48000
EXPAND_PLACEHOLDER_DIST = 1.0


def _is_changelog_page(meta):
    """True for anti-answer wiki pages (patch-note changelogs, ...).

    One such page ranking #1 (its text often matches the query's exact
    wording) previously dragged ALL of its sibling chunks into the context —
    a low-signal flood that crowded out real sources. Changelogs are still
    directly retrievable; they're just never sibling-*expanded*.
    """
    name = (meta or {}).get("name")
    return isinstance(name, str) and (
        name.startswith("Game updates/")
        # Meta-file-missing fallback: filename stem "Game updates2026-06-25"
        # (underscore→space, no slash) must also be caught.
        or bool(re.match(r"^Game updates\d", name))
    )

_doc_store = None
_parent_index = None


def load_parent_index():
    """Return (id->doc, parent_id->[docs]), lazily from the persisted BM25
    doc store (data/bm25_index.pkl). Built once per process."""
    global _doc_store, _parent_index
    if _doc_store is not None:
        return _doc_store, _parent_index
    _, all_docs = load_bm25_index()
    _doc_store = {d["id"]: d for d in all_docs}
    index = {}
    for d in all_docs:
        pid = d.get("metadata", {}).get("parent_id")
        if pid:
            index.setdefault(pid, []).append(d)
    _parent_index = index
    return _doc_store, _parent_index


def expand_parents(ids, texts, metas, dists, max_chars=EXPAND_MAX_CHARS,
                   max_pages=EXPAND_MAX_PAGES):
    """Return `(ids, texts, metas, dists)` with sibling wiki chunks of the
    already-retrieved docs appended (bounded, id-deduped, index-aligned), or
    the inputs unchanged when no `parent_id`'d wiki doc yields new siblings.

    Appended docs get distance `EXPAND_PLACEHOLDER_DIST`; each page's
    siblings are spliced right after its first retrieved chunk so the answer
    is not buried at the end of the context. The inputs are not mutated.
    """
    _, parent_index = load_parent_index()

    parent_ids = []
    seen = set()
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        if _is_changelog_page(meta):
            # Patch-note changelogs are never sibling-expanded (see helper);
            # the directly-retrieved chunk(s) still rank normally.
            continue
        parent = meta.get("parent_id")
        if parent and parent not in seen:
            seen.add(parent)
            parent_ids.append(parent)
        if len(parent_ids) >= max_pages:
            break

    if not parent_ids:
        return ids, texts, metas, dists

    existing = set(ids)
    per_parent = {}
    used_chars = 0

    for pid in parent_ids:
        added_ids, added_texts, added_metas, added_dists = [], [], [], []
        for doc in parent_index.get(pid, []):
            if doc["id"] in existing:
                continue
            if used_chars + len(doc["text"]) > max_chars:
                continue
            existing.add(doc["id"])
            added_ids.append(doc["id"])
            added_texts.append(doc["text"])
            added_metas.append(doc["metadata"])
            added_dists.append(EXPAND_PLACEHOLDER_DIST)
            used_chars += len(doc["text"])
        if added_ids:
            per_parent[pid] = (
                added_ids, added_texts, added_metas, added_dists,
            )

    if not per_parent:
        return ids, texts, metas, dists

    result_ids, result_texts, result_metas, result_dists = [], [], [], []
    spliced = set()
    for i in range(len(ids)):
        result_ids.append(ids[i])
        result_texts.append(texts[i])
        result_metas.append(metas[i])
        result_dists.append(dists[i])
        pid = (
            metas[i].get("parent_id")
            if isinstance(metas[i], dict) else None
        )
        if pid in per_parent and pid not in spliced:
            spliced.add(pid)
            a_ids, a_texts, a_metas, a_dists = per_parent[pid]
            result_ids.extend(a_ids)
            result_texts.extend(a_texts)
            result_metas.extend(a_metas)
            result_dists.extend(a_dists)

    # A parent not present in the retrieved list is impossible (parents come
    # from retrieved metadata), but append defensively.
    for pid, (a_ids, a_texts, a_metas, a_dists) in per_parent.items():
        if pid not in spliced:
            result_ids.extend(a_ids)
            result_texts.extend(a_texts)
            result_metas.extend(a_metas)
            result_dists.extend(a_dists)

    return result_ids, result_texts, result_metas, result_dists