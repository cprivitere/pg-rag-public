import logging
import re

import chromadb

from pgrag.config import CONTEXT_BUDGET
from pgrag.embeddings.llama_embeddings import embed_text
from pgrag.rag.query_classifier import classify_query, find_entity, find_entities, is_leveling_intent
from pgrag.rag.entity_retrieval import build_entity_context
from pgrag.rag.retriever import retrieve
from pgrag.rag.query_plan import plan_query
from pgrag.rag.resolve import expand_parents
from pgrag.rag.prompts import build_prompt
from pgrag.rag.llm import generate, stream_generate
from pgrag.rag.synthesis_detector import should_synthesize
from pgrag.rag.synthesis_generator import synthesize_answer

logger = logging.getLogger(__name__)

MISSING_REGEX = re.compile(
    r"\b(i do not know|not found|no information|unable to find)\b",
    re.IGNORECASE,
)
SUBJECT_REGEX = re.compile(
    r"i do not know (?:how to|about) ([^.,!?\n]+)",
    re.IGNORECASE,
)


_SUMMARY_CANDIDATES = 10

# The deferred "request more" is bounded: at most ONE wiki-parent expansion
# round per missing answer (an unbounded loop risks runaway context growth).
_AGENTIC_MAX_ROUNDS = 1

# Query terms hinting at a gathering/harvesting question
_GATHERING_TERMS = {
    "mushroom", "mushrooms", "gather", "gathering", "harvest", "harvestable",
    "pick", "pickable", "forage", "foraging", "mine", "mining", "fish",
    "fishing", "mycology", "tanning", "collect", "collecting",
}

# Query terms hinting at a crafting/recipe question — these must NOT route to
# gathering summaries, even when they share a skill word (e.g. "Mycology
# recipe" is about recipes, not harvesting).
_CRAFTING_TERMS = {
    "recipe", "recipes", "craft", "crafting", "cook", "cooking", "make",
    "makeable", "learn", "train", "ability", "abilities", "craftable",
}


def _summary_score(question, doc, meta, dist):
    """Score a summary candidate: lexical overlap + domain preference - distance."""
    query_terms = {t for t in re.findall(r"[a-z]+", question.lower()) if len(t) > 2}
    text_lower = doc.lower()
    overlap = sum(1 for t in query_terms if t in text_lower)

    name_lower = str(meta.get("name", "")).lower()
    score = overlap - dist

    has_gathering = bool(query_terms & _GATHERING_TERMS)
    has_crafting = bool(query_terms & _CRAFTING_TERMS)
    gathering_named = "gathering" in name_lower

    if has_gathering and not has_crafting and gathering_named:
        score += 3.0
    elif has_crafting and gathering_named:
        # Crafting questions want recipe summaries, not harvest tables
        score -= 2.0
    # Wiki-derived summaries carry curated, complete tables
    if "wiki" in name_lower:
        score += 1.0

    return score


def _find_matching_summary(question):
    """Find the most relevant summary document for a comparison query."""
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection(name="project_gorgon")

    embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=_SUMMARY_CANDIDATES,
        where={"type": "summary"},
    )

    if not results["ids"][0]:
        return None

    best, best_score = None, None
    best_id = best_meta = best_dist = None
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = _summary_score(question, doc, meta, dist)
        if best_score is None or score > best_score:
            best, best_score = doc, score
            best_id, best_meta, best_dist = doc_id, meta, dist
    if best is None:
        return None
    # Rechunked summaries split into `_chunk_` docs — pull the whole artifact
    # so "all recipes for X" isn't answered from a single fragment.
    _, best_texts, _, _ = expand_parents(
        [best_id], [best], [best_meta], [best_dist],
    )
    return "\n\n".join(best_texts)


def _fit_context(documents, max_chars=CONTEXT_BUDGET):
    """Trim the retrieved/expanded docs to at most max_chars total (keeping
    whole docs from the head, dropping only the overflow tail) so retrieval +
    sibling expansion can never silently push the prompt past the LLM window.
    The retrieval-ranked head and its spliced siblings are preserved; only the
    tail is dropped. Returns a (possibly unchanged) new list."""
    used, kept = 0, []
    for doc in documents:
        if used and used + len(doc) > max_chars:
            break
        kept.append(doc)
        used += len(doc)
    return kept


def _generate_with(question, documents, query_type, generation=None):
    context = "\n\n---\n\n".join(_fit_context(documents))
    prompt = build_prompt(question, context, query_type=query_type)
    return generate(prompt, **(generation or {}))


def _build_sources(ids, distances, metadatas):
    return [
        {
            "id": doc_id,
            "distance": distance,
            "metadata": metadata,
            "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
        }
        for doc_id, distance, metadata in zip(ids, distances, metadatas)
    ]


def _resolve_expansion(ids, docs, metas, dists):
    """Attempt one bounded wiki-parent/sibling expansion of the current
    context (the deferred "request more": the missing answer may live in a
    sibling chunk of an already-retrieved page). Returns the (possibly
    unchanged) lists plus the number of appended docs (0 = nothing to
    expand, e.g. no wiki parent among the retrieved docs)."""
    if not any(isinstance(m, dict) and m.get("parent_id") for m in metas):
        return ids, docs, metas, dists, 0
    before = len(docs)
    ids, docs, metas, dists = expand_parents(ids, docs, metas, dists)
    return ids, docs, metas, dists, len(docs) - before


def _gap_fill(question, answer, ids, docs, metas, dists, query_type, generation=None, trace=None, allow_gap_fill=False):
    """One-shot targeted re-retrieval on the missing subject (V36).
    Bare "I do not know." carries no subject -> fall back to the question itself.
    Empty answer also counts as missing.
    """
    if not allow_gap_fill:
        return answer, ids, docs, metas, dists, False
    if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
        return answer, ids, docs, metas, dists, False
    if not (answer or "").strip():
        answer = _generate_with(question, docs, query_type, generation=generation)
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            return answer, ids, docs, metas, dists, False

    # Deferred "request more": re-answer from the expanded wiki page before
    # falling through to the subject re-retrieval. One bounded round.
    expanded = 0
    ids, docs, metas, dists, expanded = _resolve_expansion(ids, docs, metas, dists)
    if expanded:
        answer = _generate_with(question, docs, query_type, generation=generation)
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            if trace is not None:
                trace["resolve"] = {"rounds": 1, "expanded": expanded}
            return answer, ids, docs, metas, dists, False
    if trace is not None:
        trace["resolve"] = {"rounds": 1 if expanded else 0, "expanded": expanded}

    m = SUBJECT_REGEX.search(answer)
    subject = m.group(1).strip() if m else question
    extra = retrieve(
        f"{subject} {question}",
        count=5,
        hybrid=True,
        rerank=True,
        trace=trace,
    )
    if trace is not None:
        trace["gap_fill"] = {"triggered": True, "subject": subject}
    seen = set(ids)
    for i in range(len(extra["ids"][0])):
        did = extra["ids"][0][i]
        if did in seen:
            continue
        seen.add(did)
        ids.append(did)
        docs.append(extra["documents"][0][i])
        metas.append(extra["metadatas"][0][i])
        dists.append(extra["distances"][0][i])
    answer = _generate_with(question, docs, query_type, generation=generation)
    return answer, ids, docs, metas, dists, extra.get("rerank_used", False)


def _expand_synonyms(query: str) -> str:
    """Replace user-facing synonyms with game terminology before classification.

    Runs before classify_query so the classifier, spelling correction, and
    retrieval all see the canonical game term rather than the synonym.
    """
    _SYNONYM_MAP = {
        r"\blycanthropes\b": "werewolf",
        r"\blycanthrope\b": "werewolf",
    }
    for pattern, replacement in _SYNONYM_MAP.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    return query


def _prepare_entity(question):
    """Retrieve an entity dossier without generating. Returns
    (ids, docs, metas, dists, rerank_used) or None if no hub found."""
    hub_id, _ = find_entity(question)
    ctx = build_entity_context(
        question, hub_id, include_leveling=is_leveling_intent(question),
    )
    if ctx is None:
        return None
    return (
        list(ctx["ids"][0]),
        list(ctx["documents"][0]),
        list(ctx["metadatas"][0]),
        list(ctx["distances"][0]),
        ctx.get("rerank_used", False),
    )


def _ask_entity(question, generation=None, trace=None, allow_gap_fill=False):
    prepared = _prepare_entity(question)
    if prepared is None:
        return None
    ids, docs, metas, dists, rerank_used = prepared

    answer = _generate_with(question, docs, "entity", generation=generation)
    answer, ids, docs, metas, dist, gap_used = _gap_fill(
        question, answer, ids, docs, metas, dists, "entity",
        generation=generation, trace=trace, allow_gap_fill=allow_gap_fill,
    )

    return {
        "answer": answer,
        "documents": docs,
        "query_type": "entity",
        "rerank_used": rerank_used or gap_used,
        "sources": _build_sources(ids, dist, metas),
    }


def _prepare_multi_entity(question, entities, generation=None, trace=None):
    """Build one labeled context from several entity dossiers and answer.

    Used for comparison queries naming 2+ entities (e.g. "Punch vs Front
    Kick"), so the answer sees both entities' own damage/levels rather than
    only the first-resolved hub.
    """
    from pgrag.rag.entity_retrieval import build_multi_entity_context

    ctx = build_multi_entity_context(question, entities, trace=trace)
    if ctx is None:
        return None
    docs = ctx["documents"][0]
    if trace is not None:
        trace["entities"] = [
            {"name": name, "id": hub, "dtype": dtype}
            for name, hub, dtype in entities
        ]
    answer = _generate_with(question, docs, "comparison", generation=generation)
    return {
        "answer": answer,
        "documents": docs,
        "query_type": "comparison",
        "rerank_used": ctx.get("rerank_used", False),
        "sources": _build_sources(ctx["ids"][0], ctx["distances"][0], ctx["metadatas"][0]),
    }


def _apply_plan(question, metadata_filter, trace=None):
    """Turn a high-confidence metadata plan into native+token filters for the
    general path. A caller-supplied filter wins (no auto-plan override)."""
    if metadata_filter is not None:
        return metadata_filter, None
    plan = plan_query(question)
    if plan is None:
        return None, None
    if trace is not None:
        trace["plan"] = {
            "label": plan["label"],
            "native": plan["native"],
            "token": plan["token"],
        }
    return plan["native"], plan["token"] or None


def _prepare_general(question, query_type, metadata_filter=None, token_filter=None, trace=None, generation=None):
    """Retrieve (and possibly synthesize) context for a general/comparison
    query. Returns (query_type, ids, documents, metadatas, distances,
    rerank_used). Does not call the LLM."""
    if trace is not None:
        trace["query"] = question
        trace["classifier"] = query_type
    # "lookup" queries ("Where can I find X?") need the same wide hybrid
    # recall + wiki expansion as "general" — an answer is a fact, not a
    # comparison, so a 3-doc dense-only window starves it.
    is_wide = query_type in ("general", "lookup")
    # A plan that narrows Chroma to one table needs a wider surfaced count —
    # every candidate is on-topic, and the family's full set (creature spawn
    # table, per-NPC gift summaries) is what enumeration questions need.
    # "List all the locations with <animals>" / "who can I gift X to?" must
    # surface their complete member list, not a tight generic top-k.
    mf_tables = {
        c.get("table") for c in (metadata_filter or {}).get("$and", [])
    } | {(metadata_filter or {}).get("table")}
    plan_count = 80 if mf_tables & {"creatures", "summaries"} else (40 if is_wide else 3)
    results = retrieve(
        question,
        metadata_filter=metadata_filter,
        token_filter=token_filter,
        query_type=query_type,
        count=plan_count,
        hybrid=is_wide,
        trace=trace,
    )

    documents = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    _expanded = False
    if is_wide and any(
        isinstance(m, dict) and m.get("parent_id") for m in metadatas
    ):
        # Wiki page expansion: pull sibling chunks via parent_id so
        # "how-to" answers aren't lost in a non-retrieved chunk.
        before = len(ids)
        ids, documents, metadatas, distances = expand_parents(
            ids, documents, metadatas, distances,
        )
        _expanded = len(ids) > before
        if trace is not None and _expanded:
            parent_ids = sorted({
                m.get("parent_id")
                for m in metadatas
                if isinstance(m, dict) and m.get("parent_id")
            })
            trace["expansion"] = {
                "parent_ids": parent_ids,
                "chars": sum(len(d) for d in documents[before:]),
                "added_count": len(ids) - before,
            }

    if query_type == "comparison":
        summary = _find_matching_summary(question)
        if summary:
            documents = [summary] + documents

    # Check if synthesis should be triggered. When wiki expansion spliced
    # sibling chunks into the context, the assembled page IS the coherent
    # answer — synthesis over the top-3 retrieved docs (often stale curated
    # summaries) would throw the specific row away.
    result_dicts = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    if should_synthesize(result_dicts, query_type) and not _expanded:
        # Synthesize scattered results
        try:
            synthesized = synthesize_answer(question, result_dicts[:3], generation=generation)
            # Use synthesized as context instead of raw docs (ephemeral, in-request
            # only — never persisted to disk).
            documents = [synthesized]
            if trace is not None:
                trace["synthesis"] = {
                    "triggered": True,
                    "source_ids": list(ids[:3]),
                }
        except Exception as exc:
            logger.warning(
                "synthesis failed for %r (query_type=%s): %s",
                question,
                query_type,
                exc,
            )

    return query_type, ids, documents, metadatas, distances, results.get("rerank_used", False)


def _stream_generation(question, documents, query_type, generation=None):
    context = "\n\n---\n\n".join(_fit_context(documents))
    prompt = build_prompt(question, context, query_type=query_type)
    return stream_generate(prompt, **(generation or {}))


def _stream_answer(question, ids, docs, metas, dists, query_type, generation=None, trace=None, allow_gap_fill=False):
    """Stream the answer (and any gap-fill re-answer) for a prepared context.

    Yields {"type": "token", "text"} deltas and {"type": "reset"} before a
    gap-fill regeneration. Returns (answer, ids, docs, metas, dists,
    gap_used) once the final generation completes.
    """
    answer = ""
    for delta in _stream_generation(question, docs, query_type, generation=generation):
        answer += delta
        yield {"type": "token", "text": delta}

    if not allow_gap_fill:
        return answer, ids, docs, metas, dists, False

    if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
        return answer, ids, docs, metas, dists, False

    if not (answer or "").strip():
        answer = ""
        yield {"type": "reset"}
        for delta in _stream_generation(question, docs, query_type, generation=generation):
            answer += delta
            yield {"type": "token", "text": delta}
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            return answer, ids, docs, metas, dists, False

    # Deferred "request more": re-answer from the expanded wiki page before
    # falling through to the subject re-retrieval. One bounded round.
    expanded = 0
    ids, docs, metas, dists, expanded = _resolve_expansion(ids, docs, metas, dists)
    if expanded:
        answer = ""
        yield {"type": "reset"}
        for delta in _stream_generation(question, docs, query_type, generation=generation):
            answer += delta
            yield {"type": "token", "text": delta}
        if (answer or "").strip() and not MISSING_REGEX.search(answer or ""):
            if trace is not None:
                trace["resolve"] = {"rounds": 1, "expanded": expanded}
            return answer, ids, docs, metas, dists, False
    if trace is not None:
        trace["resolve"] = {"rounds": 1 if expanded else 0, "expanded": expanded}

    m = SUBJECT_REGEX.search(answer)
    subject = m.group(1).strip() if m else question
    extra = retrieve(
        f"{subject} {question}",
        count=5,
        hybrid=True,
        rerank=True,
        trace=trace,
    )
    if trace is not None:
        trace["gap_fill"] = {"triggered": True, "subject": subject}
    seen = set(ids)
    for i in range(len(extra["ids"][0])):
        did = extra["ids"][0][i]
        if did in seen:
            continue
        seen.add(did)
        ids.append(did)
        docs.append(extra["documents"][0][i])
        metas.append(extra["metadatas"][0][i])
        dists.append(extra["distances"][0][i])

    answer = ""
    yield {"type": "reset"}
    for delta in _stream_generation(question, docs, query_type, generation=generation):
        answer += delta
        yield {"type": "token", "text": delta}
    return answer, ids, docs, metas, dists, extra.get("rerank_used", False)


def ask_stream(question, metadata_filter=None, generation=None, trace=None, allow_gap_fill=False):
    """Streaming variant of ask().

    Yields {"type": "token", "text"} deltas (with {"type": "reset"} between
    gap-fill re-answers) and finally {"type": "final", "result": {...}}
    where result mirrors ask()'s return value.
    """
    question = _expand_synonyms(question)
    query_type = classify_query(question)
    if trace is not None:
        trace.setdefault("query", question)
        trace.setdefault("classifier", query_type)

    if query_type == "entity":
        prepared = _prepare_entity(question)
        if prepared is not None:
            ids, docs, metas, dists, rerank_used = prepared
            answer, ids, docs, metas, dists, gap_used = yield from _stream_answer(
                question, ids, docs, metas, dists, "entity",
                generation=generation, trace=trace, allow_gap_fill=allow_gap_fill,
            )
            if trace is not None:
                trace["generation"] = dict(generation or {})
                trace["corrected_query"] = (trace.get("retrieval_calls") or [{}])[0].get("query", question)
            yield {"type": "final", "result": {
                "answer": answer,
                "documents": docs,
                "query_type": "entity",
                "rerank_used": rerank_used or gap_used,
                "sources": _build_sources(ids, dists, metas),
            }}
            return
        query_type = "general"

    if query_type == "comparison":
        ents = find_entities(question)
        if len(ents) >= 2:
            multi = _prepare_multi_entity(
                question, ents, generation=generation, trace=trace,
            )
            if multi is not None:
                if trace is not None:
                    trace["generation"] = dict(generation or {})
                    trace["corrected_query"] = (trace.get("retrieval_calls") or [{}])[0].get("query", question)
                yield {"type": "final", "result": multi}
                return
        # No entities found → widen comparison to general retrieval for
        # hybrid (BM25 + dense) recall, so a synonym-expanded query like
        # "best armor for werewolf" matches werewolf armor items lexically.
        _comparison_fallback = True
        query_type = "general"
    else:
        _comparison_fallback = False

    mf, tf = _apply_plan(question, metadata_filter, trace=trace)
    qt, ids, documents, metadatas, distances, rerank_used = _prepare_general(
        question, query_type, mf, token_filter=tf, trace=trace, generation=generation
    )

    if _comparison_fallback:
        qt = "comparison"

    answer, ids, documents, metadatas, distances, gap_used = yield from _stream_answer(
        question, ids, documents, metadatas, distances, qt,
        generation=generation, trace=trace, allow_gap_fill=allow_gap_fill,
    )
    if trace is not None:
        trace["generation"] = dict(generation or {})
        trace["corrected_query"] = (trace.get("retrieval_calls") or [{}])[0].get("query", question)
    yield {"type": "final", "result": {
        "answer": answer,
        "documents": documents,
        "query_type": qt,
        "rerank_used": rerank_used or gap_used,
        "sources": [
            {
                "id": doc_id,
                "distance": distance,
                "metadata": metadata,
                "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
            }
            for doc_id, distance, metadata in zip(
                ids,
                distances,
                metadatas
            )
        ],
    }}


def ask(question, metadata_filter=None, generation=None, trace=None, allow_gap_fill=False):
    question = _expand_synonyms(question)
    query_type = classify_query(question)
    if trace is not None:
        trace.setdefault("query", question)
        trace.setdefault("classifier", query_type)

    if query_type == "entity":
        prepared = _prepare_entity(question)
        if prepared is not None:
            return _ask_entity(question, generation=generation, trace=trace, allow_gap_fill=allow_gap_fill)
        query_type = "general"

    if query_type == "comparison":
        ents = find_entities(question)
        if len(ents) >= 2:
            multi = _prepare_multi_entity(
                question, ents, generation=generation, trace=trace,
            )
            if multi is not None:
                return multi
        # No entities found → widen comparison to general retrieval for
        # hybrid (BM25 + dense) recall.
        _comparison_fallback = True
        query_type = "general"
    else:
        _comparison_fallback = False

    mf, tf = _apply_plan(question, metadata_filter, trace=trace)
    qt, ids, documents, metadatas, distances, rerank_used = _prepare_general(
        question, query_type, mf, token_filter=tf, trace=trace, generation=generation
    )

    if _comparison_fallback:
        qt = "comparison"

    context = "\n\n---\n\n".join(_fit_context(documents))

    prompt = build_prompt(
        question,
        context,
        query_type=qt
    )

    answer = generate(prompt, **(generation or {}))
    answer, ids, documents, metadatas, distances, gap_used = _gap_fill(
        question, answer, ids, documents, metadatas, distances, qt,
        generation=generation, trace=trace, allow_gap_fill=allow_gap_fill,
    )

    if trace is not None:
        trace["generation"] = dict(generation or {})
        trace["corrected_query"] = (trace.get("retrieval_calls") or [{}])[0].get("query", question)

    return {
        "answer": answer,
        "documents": documents,
        "query_type": qt,
        "rerank_used": rerank_used or gap_used,
        "sources": [
            {
                "id": doc_id,
                "distance": distance,
                "metadata": metadata,
                "citation": f"{metadata.get('name', doc_id)} ({metadata.get('table', 'unknown')})"
            }
            for doc_id, distance, metadata in zip(
                ids,
                distances,
                metadatas
            )
        ]
    }
