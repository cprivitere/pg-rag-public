import logging
import re

import chromadb

from pgrag.embeddings.llama_embeddings import embed_text
from pgrag.rag.spelling import correct_query

logger = logging.getLogger(__name__)

RERANK_MULTIPLIER = 3
RRF_K = 60
HYBRID_MULTIPLIER = 3

# Near-duplicate cluster members that are fragments of one retrievable base
# unit. For most corpora (wiki leveling tables, skill profile chunks) each
# member is distinctly relevant, so collapsing them visibly regresses
# multi-row results. The one debris class that does NOT carry per-member
# relevance is the `tsys_power_*` mechanics-doc chunk splits: a handful of
# `tsys_power_NNNN_chunk_M` fragments of the same mechanics page crowd a
# single unrelated target out of the rerank window. Cap those per base.
_TSYS_CHUNK_RE = re.compile(r"^(tsys_power_\d+)_chunk_\d+$")
_TSYS_ORIGIN_RE = re.compile(r"^(tsys_power_\d+)(?:_chunk_\d+)?$")

# Unique tsys power-mechanics chunks kept per base doc in the fused window.
MAX_TSYS_CHUNK_MEMBERS = 2

# Distinct `tsys_power_*` bases (treasure-suffix powers) allowed in the fused
# window. These are 7.6k docs that matching a shared term ("Sword", "damage")
# swarms ability/skill/comparison queries; a genuine treasure-suffix query
# still surfaces a handful, so cap rather than exclude.
MAX_TSYS_BASES = 3


def _tsys_base_id(doc_id: str) -> str | None:
    """Base doc id if doc_id is a tsys_power_* chunk fragment, else None."""
    m = _TSYS_CHUNK_RE.match(doc_id)
    return m.group(1) if m else None


def _tsys_origin_id(doc_id: str) -> str | None:
    """`tsys_power_NNNN` origin for any tsys doc (bare or chunked), else None.

    Used for the distinct-base cap: treasure powers are 7.6k distinct docs that
    swarm shared-term queries, so the cap counts distinct origins whether a
    doc is un-chunked or a `_chunk_` fragment of the same base.
    """
    m = _TSYS_ORIGIN_RE.match(doc_id)
    return m.group(1) if m else None


# --- Exact entity-name recall / promotion ---
#
# Lookup-style questions ("What level is Fireball 3 unlocked at?", "What
# level should I be for Gazluk Keep?") name a specific entity whose cdn
# `metadata.name` is an exact, unambiguous multi-token phrase in the query.
# The dense/BM25 window can still miss it: a terse name-only stub gets a weak
# embedding (surfaced far outside the window), and verbose query tokens dilute
# its BM25 score. And when it does reach the rerank pool, the cross-encoder
# can under-rank its sparse text. So for docs whose exact name is a contiguous
# multi-token n-gram of the query, (a) inject them into the rerank pool when
# the window missed them, and (b) float them to the top of the reranked
# output. Bounded to multi-token exact names so single common words
# ("fire", "magic", "level") never mass-promote.
_NAME_INDEX = None
MAX_NAME_INJECT = 4

# Fragment ids inherit their parent page's name (wiki table rows/coverage and
# chunk splits all carry the page title), so matching on `name` alone would
# mass-promote an entire row family. Skip them: promote the substantive docs.
_FRAGMENT_SUFFIX_RE = re.compile(r"_(?:row|chunk)_\d+$|_coverage$")


def _is_fragment_id(doc_id: str) -> bool:
    return bool(_FRAGMENT_SUFFIX_RE.search(doc_id))


def _tokenize(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _entity_name_match(query: str, doc_name: str | None) -> bool:
    """True iff doc_name is a contiguous 2+ token n-gram of the query.

    e.g. name "Fireball 3" matches query "what level is fireball 3 unlocked
    at"; name "fire" (single token) does not.
    """
    if not doc_name:
        return False
    name = _tokenize(doc_name)
    q = _tokenize(query)
    if len(name) < 2 or len(name) > len(q):
        return False
    for i in range(len(q) - len(name) + 1):
        if q[i:i + len(name)] == name:
            return True
    return False


def _load_name_index(docs=None) -> dict:
    """Build (once, in-process) a lowercased exact-name -> [doc ids] map."""
    global _NAME_INDEX
    if _NAME_INDEX is None:
        if docs is None:
            from pgrag.rag.bm25 import load_bm25_index
            docs = load_bm25_index()[1]
        idx = {}
        for d in docs:
            n = (d.get("metadata") or {}).get("name")
            if not n:
                continue
            idx.setdefault(" ".join(n.lower().split()), []).append(d["id"])
        _NAME_INDEX = idx
    return _NAME_INDEX


def _name_injection_ids(query: str, docs=None) -> list[str]:
    """Doc ids whose exact name matches the longest multi-token query span.

    Only docs whose full name equals the longest contiguous 2+ token n-gram of
    the query are returned (the most specific entity, e.g. "Healing Potion
    Omega" beats the shorter "Healing Potion"). Capped at MAX_NAME_INJECT.
    """
    idx = _load_name_index(docs)
    q = _tokenize(query)
    for size in range(min(len(q), 4), 1, -1):
        cur = []
        seen = set()
        for i in range(len(q) - size + 1):
            key = " ".join(q[i:i + size])
            for did in idx.get(key, ()):
                if did not in seen:
                    seen.add(did)
                    cur.append(did)
        if cur:
            return [d for d in cur if not _is_fragment_id(d)][:MAX_NAME_INJECT]
    return []


def _apply_name_promotion(query, pool_ids, pool_docs, pool_metas, pool_dists,
                          ranked_ids, ranked_docs, ranked_metas, ranked_dists, count):
    """Float the exact-entity docs to the front of the reranked top-N.

    A pool doc whose full name equals the longest multi-token query span (the
    most specific entity name in the query, e.g. "Gazluk Keep", "Fireball 3")
    is moved ahead of the cross-encoder's picks — an exact name is a strong
    signal the encoder can underweight for terse docs. Shorter/generic name
    matches (e.g. "Fireball", "Healing Potion") are NOT promoted, so a broad
    family match never displaces the specific gold doc. A matched doc absent
    from the ranked top-N is spliced back in. Order is otherwise preserved;
    output trimmed to ``count``.
    """
    spans = []
    for i, m in enumerate(pool_metas):
        if _is_fragment_id(pool_ids[i]):
            continue
        name = (m or {}).get("name")
        nm = _tokenize(name)
        if _entity_name_match(query, name):
            spans.append((i, len(nm)))
    if not spans:
        return ranked_ids, ranked_docs, ranked_metas, ranked_dists

    longest = max(nl for _, nl in spans)
    matched_pool = {i for i, nl in spans if nl == longest}

    result = list(ranked_ids)
    present = set(result)
    # Re-include any matched pool doc the encoder dropped from its top-N.
    for i in sorted(matched_pool):
        if pool_ids[i] not in present:
            result.insert(0, pool_ids[i])
            present.add(pool_ids[i])

    matched_set = {pool_ids[i] for i in matched_pool}
    ordered = (
        [x for x in result if x in matched_set]
        + [x for x in result if x not in matched_set]
    )[:count]

    by_id = {
        pool_ids[i]: (pool_docs[i], pool_metas[i], pool_dists[i])
        for i in range(len(pool_ids))
    }
    new_docs, new_metas, new_dists = [], [], []
    for x in ordered:
        d, m, dst = by_id[x]
        new_docs.append(d)
        new_metas.append(m)
        new_dists.append(dst)
    return ordered, new_docs, new_metas, new_dists


def _term_overlap(query, document):
    """Fraction of query terms (lowercased, whitespace-split) that appear as
    substrings in the document text; 0.0 for an empty query."""
    query_terms = set(query.lower().split())
    if not query_terms:
        return 0.0
    doc_text = document.lower()
    matches = sum(1 for t in query_terms if t in doc_text)
    return matches / len(query_terms)


def _rerank(query, ids, documents, metadatas, distances, count):
    """Pure lexical fallback re-ranker (no server): score each doc as
    1/(rank+1) + _term_overlap(query, doc), sort descending, return the top
    ``count`` quads."""
    scored = []
    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        orig_score = 1.0 / (rank + 1)
        term_score = _term_overlap(query, doc)
        scored.append((orig_score + term_score, doc_id, doc, meta, dist))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:count]
    return (
        [t[1] for t in top],
        [t[2] for t in top],
        [t[3] for t in top],
        [t[4] for t in top],
    )


def _scalar_eq(value, target):
    """Equality that understands Chroma scalar metadata quirks:
    delimited `" | "`-joined strings match by token membership, and
    numeric metadata matches numeric strings (filters arrive as strings)."""
    if value is None:
        return False
    if isinstance(value, str) and " | " in value:
        return str(target) in value.split(" | ")
    if isinstance(value, (int, float)) and isinstance(target, str):
        try:
            return value == float(target)
        except ValueError:
            return False
    return value == target


_WHERE_OPS = {
    "$eq": lambda v, t: _scalar_eq(v, t),
    "$ne": lambda v, t: not _scalar_eq(v, t),
    "$gt": lambda v, t: v is not None and v > t,
    "$gte": lambda v, t: v is not None and v >= t,
    "$lt": lambda v, t: v is not None and v < t,
    "$lte": lambda v, t: v is not None and v <= t,
}


def _where_matches(metadata, clause):
    """Evaluate a Chroma-style `where` clause against one doc's metadata.

    Used for the post-fusion filter: BM25-fused hits never passed through
    Chroma's `where`, so retrieve() re-checks them here. Supports the
    operators Chroma supports ($eq/$ne/$gt/$gte/$lt/$lte), $and/$or, plain
    equality, and token membership on " | "-delimited fields.
    """
    for key, condition in clause.items():
        if key == "$and":
            if not all(_where_matches(metadata, c) for c in condition):
                return False
            continue
        if key == "$or":
            if not any(_where_matches(metadata, c) for c in condition):
                return False
            continue
        value = metadata.get(key)
        if isinstance(condition, dict):
            for op, target in condition.items():
                fn = _WHERE_OPS.get(op)
                if fn is None or not fn(value, target):
                    return False
        elif not _scalar_eq(value, condition):
            return False
    return True


def _hybrid_fuse(dense_ids, dense_texts, dense_metadatas, dense_distances,
                  bm25_ids, all_docs, count):
    """RRF-fuse the dense and BM25 rank lists (RRF_K=60), then apply the
    tsys_power_* caps (per-base chunk members, distinct-base origins) so a
    treasure-suffix mechanics swarm cannot crowd unrelated targets out of the
    rerank window. Returns the top ``count`` (ids, texts, metas, dists);
    BM25-only docs get dist 0.0."""
    dense_info = {}
    for rank, (doc_id, text, meta, dist) in enumerate(
        zip(dense_ids, dense_texts, dense_metadatas, dense_distances)
    ):
        dense_info[doc_id] = (rank, text, meta, dist)

    bm25_ranks = {doc_id: rank for rank, doc_id in enumerate(bm25_ids)}
    doc_lookup = {d["id"]: d for d in all_docs}

    all_ids = set(dense_ids) | set(bm25_ranks.keys())
    fallback_rank = max(len(dense_ids), len(bm25_ids)) * 2

    scored = []
    for doc_id in all_ids:
        dr = dense_info[doc_id][0] if doc_id in dense_info else fallback_rank
        br = bm25_ranks.get(doc_id, fallback_rank)
        rrf = 1.0 / (RRF_K + dr + 1) + 1.0 / (RRF_K + br + 1)
        scored.append((rrf, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Two distinct tsys bounds, one window:
    #   - Chunk-collapse (per base, `_tsys_base_id`): a cluster of
    #     `tsys_power_NNNN_chunk_M` fragments of the SAME base crowds an
    #     unrelated single target out of the rerank window — keep at most
    #     MAX_TSYS_CHUNK_MEMBERS per base.
    #   - Distinct-base cap (`_tsys_origin_id`, counts bare AND chunked):
    #     the 7.6k treasure-suffix powers swarm ability/skill/comparison
    #     queries on a shared term ("Sword", "Slashing", "damage"); a genuine
    #     treasure-suffix query still surfaces a few, so cap ORIGINS, not
    #     exclude the type entirely.
    # Wiki table row / skill chunks are distinctly relevant and stay.
    top_ids = []
    per_base: dict[str, int] = {}
    origins_seen = 0
    for _, doc_id in scored:
        base = _tsys_base_id(doc_id)
        if base is not None:
            if per_base.get(base, 0) >= MAX_TSYS_CHUNK_MEMBERS:
                continue
            per_base[base] = per_base.get(base, 0) + 1
        origin = _tsys_origin_id(doc_id)
        if origin is not None:
            if origins_seen >= MAX_TSYS_BASES:
                continue
            origins_seen += 1
        top_ids.append(doc_id)
        if len(top_ids) >= count:
            break

    del per_base

    ids, texts, metas, dists = [], [], [], []
    for doc_id in top_ids:
        if doc_id in dense_info:
            _, text, meta, dist = dense_info[doc_id]
        else:
            doc = doc_lookup.get(doc_id, {})
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            dist = 0.0
        ids.append(doc_id)
        texts.append(text)
        metas.append(meta)
        dists.append(dist)

    return ids, texts, metas, dists


def retrieve(question, count=3, metadata_filter=None, token_filter=None, rerank=True, hybrid=False, query_type="general", trace=None):
    """Run the retrieval stage: spell-correct, embed + query Chroma (count
    effective = 20 for comparison, else ``count or 3``), optionally RRF-fuse BM25
    (_hybrid_fuse) and inject/promote exact-name entities, apply post-fusion
    metadata/token filters (BM25 bypasses Chroma's where), then cross-encode
    via :8082 with lexical fallback. Returns the Chroma results dict
    (ids/documents/metadatas/distances + rerank_used); appends a per-stage
    id list to ``trace`` when given."""
    raw_question = question
    question = correct_query(question)
    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_collection(
        name="project_gorgon"
    )

    embedding = embed_text(question)

    effective_count = 20 if query_type == "comparison" else (count or 3)

    dense_count = effective_count
    if hybrid:
        dense_count *= HYBRID_MULTIPLIER
    if rerank and not hybrid:
        dense_count *= RERANK_MULTIPLIER

    query_kwargs = dict(
        query_embeddings=[embedding],
        n_results=dense_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if metadata_filter is not None:
        query_kwargs["where"] = metadata_filter

    results = collection.query(**query_kwargs)
    results["rerank_used"] = False

    trace_rec = None
    if trace is not None:
        trace_rec = {
            "query": question,
            "hybrid": hybrid,
            "metadata_filter": metadata_filter,
            "token_filter": token_filter,
            "dense_ids": list(results["ids"][0]),
            "dense_dists": list(results["distances"][0]),
        }

    if hybrid:
        from pgrag.rag.bm25 import load_bm25_index
        fuse_target = effective_count * RERANK_MULTIPLIER if rerank else effective_count
        bm25_model, all_docs = load_bm25_index()
        bm25_indices, _ = bm25_model.search(question, k=fuse_target)
        bm25_ids = [all_docs[i]["id"] for i in bm25_indices]
        if trace_rec is not None:
            trace_rec["bm25_ids"] = bm25_ids

        fused_ids, fused_texts, fused_metas, fused_dists = _hybrid_fuse(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            bm25_ids,
            all_docs,
            fuse_target,
        )
        results["ids"] = [fused_ids]
        results["documents"] = [fused_texts]
        results["metadatas"] = [fused_metas]
        results["distances"] = [fused_dists]
        if trace_rec is not None:
            trace_rec["rrf_ids"] = list(fused_ids)

        # Exact entity-name recall: a doc whose name is the query's named
        # entity (e.g. "Fireball 3", "Gazluk Keep") can be missed entirely by
        # the dense/BM25 window (weak embedding for a terse stub; verbose
        # query dilutes its BM25 score). Surface it into the rerank pool so
        # the name-promotion step can float it up.
        name_ids = _name_injection_ids(raw_question, all_docs)
        if name_ids:
            in_pool = set(fused_ids)
            doc_lookup = {d["id"]: d for d in all_docs}
            for nid in name_ids:
                if nid in in_pool or nid not in doc_lookup:
                    continue
                doc = doc_lookup[nid]
                fused_ids.append(nid)
                fused_texts.append(doc.get("text", ""))
                fused_metas.append(doc.get("metadata", {}))
                fused_dists.append(0.0)
            results["ids"] = [fused_ids]
            results["documents"] = [fused_texts]
            results["metadatas"] = [fused_metas]
            results["distances"] = [fused_dists]

    # Post-fusion metadata filter (BM25 ignores where clause). The token
    # filter (delimited `" | "` fields Chroma can't $contains) only runs
    # here; metadata_filter already narrowed the Chroma query.
    post_filter = None
    if metadata_filter or token_filter:
        post_filter = {**(metadata_filter or {}), **(token_filter or {})}
    if post_filter and results["ids"][0]:
        filtered = [
            i for i in range(len(results["ids"][0]))
            if _where_matches(results["metadatas"][0][i], post_filter)
        ]
        results["ids"] = [[results["ids"][0][i] for i in filtered]]
        results["documents"] = [[results["documents"][0][i] for i in filtered]]
        results["metadatas"] = [[results["metadatas"][0][i] for i in filtered]]
        results["distances"] = [[results["distances"][0][i] for i in filtered]]
        if trace_rec is not None:
            trace_rec["post_filter_ids"] = list(results["ids"][0])

    if rerank and len(results["ids"][0]) > effective_count:
        ids, docs, metas, dists, used = _rerank_or_cross_encoder(
            question,
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            effective_count,
            name_query=raw_question,
        )
        results["rerank_used"] = used
        results["ids"] = [ids]
        results["documents"] = [docs]
        results["metadatas"] = [metas]
        results["distances"] = [dists]
        if trace_rec is not None:
            trace_rec["reranked_ids"] = list(ids)
            trace_rec["rerank_used"] = used

    if trace_rec is not None:
        trace_rec["rerank_used"] = results.get("rerank_used", False)
        trace.setdefault("retrieval_calls", []).append(trace_rec)

    return results


def _rerank_or_cross_encoder(query, ids, documents, metadatas, distances, count, name_query=None):
    """Cross-encoder rerank via :8082; lexical fallback on failure (V60, V61).

    Returns reordered quads + used flag. ``name_query`` (defaults to
    ``query``) is the text used for exact-entity-name promotion — pass the
    uncorrected question so digits/names the spell-corrector rewrites
    (e.g. "Fireball 3" -> "fireball") still match.
    """
    if name_query is None:
        name_query = query
    try:
        from pgrag.rag.reranker_client import rerank_documents, record_failure, record_success

        indices = rerank_documents(query, documents, count)
        reranked = (
            [ids[i] for i in indices],
            [documents[i] for i in indices],
            [metadatas[i] for i in indices],
            [distances[i] for i in indices],
        )
        reranked = _apply_name_promotion(
            name_query, ids, documents, metadatas, distances,
            *reranked, count,
        )
        record_success()
        return reranked + (True,)
    except Exception as exc:
        logger.warning("[WARN] reranker (:8082) unavailable/failed: %s", exc)
        try:  # stats update must not break retrieval (V64)
            record_failure()
        except Exception:
            pass
        fallback = _rerank(query, ids, documents, metadatas, distances, count)
        fallback = _apply_name_promotion(
            name_query, ids, documents, metadatas, distances,
            *fallback, count,
        )
        return fallback + (False,)
