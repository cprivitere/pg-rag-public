"""Pre-LLM Retrieval Evaluation Suite & IR Metrics Engine.

Provides deterministic IR metrics (Recall@k, MRR, NDCG@k, Hit@k, Reranker Uplift,
Entity Accuracy), stage-by-stage evaluation, dataset loader, and benchmark aggregator.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pgrag.rag.entity_retrieval import build_entity_context_offline
from pgrag.rag.query_classifier import classify_query, find_entities, find_entity

logger = logging.getLogger(__name__)

# --- Core Information Retrieval Metrics (Binary Relevance) ---


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """Compute Recall@k: |relevant ∩ top-k| / |relevant|.

    Returns 0.0 when relevant is empty or k <= 0.
    """
    if not relevant or k <= 0:
        return 0.0
    rel_set = set(relevant)
    top_k = set(ranked[:k])
    return len(top_k & rel_set) / len(rel_set)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str] | set[str]) -> float:
    """Compute Reciprocal Rank: 1 / rank of first relevant doc (1-based), or 0.0."""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in rel_set:
            return 1.0 / i
    return 0.0


def mrr(queries: Sequence[tuple[Sequence[str], Sequence[str] | set[str]]]) -> float:
    """Compute Mean Reciprocal Rank across (ranked, relevant) query pairs."""
    if not queries:
        return 0.0
    return sum(reciprocal_rank(ranked, relevant) for ranked, relevant in queries) / len(queries)


def dcg_at_k(ranked: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k with binary relevance.

    DCG@k = Σ_{i=1..min(k, len(ranked))} rel_i / log2(i + 1)
    where rel_i is 1.0 if ranked[i-1] in relevant else 0.0.
    """
    if not relevant or k <= 0:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in rel_set:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k with binary relevance.

    NDCG@k = DCG@k / IDCG@k
    Ideal ranking puts min(k, |relevant|) relevant documents at the top ranks.
    """
    if not relevant or k <= 0:
        return 0.0
    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def hit_at_k(ranked: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """Compute Hit@k: 1.0 if any relevant doc is in top-k, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    rel_set = set(relevant)
    return 1.0 if any(doc_id in rel_set for doc_id in ranked[:k]) else 0.0


def entity_jaccard(
    predicted: Sequence[str] | set[str], expected: Sequence[str] | set[str]
) -> float:
    """Compute Jaccard similarity between predicted and expected entity sets."""
    set_p = {p.lower() for p in predicted} if predicted else set()
    set_e = {e.lower() for e in expected} if expected else set()
    if not set_p and not set_e:
        return 1.0
    union = set_p | set_e
    if not union:
        return 1.0
    return len(set_p & set_e) / len(union)


def entity_accuracy(
    resolved_hubs: Sequence[str] | Sequence[Sequence[str]],
    expected_hubs: Sequence[str] | Sequence[Sequence[str]],
) -> float:
    """Compute fraction of matching hub resolutions across a list of query expectations."""
    if not resolved_hubs and not expected_hubs:
        return 1.0
    if not resolved_hubs or not expected_hubs:
        return 0.0
    matches = sum(1 for a, b in zip(resolved_hubs, expected_hubs, strict=False) if a == b)
    return matches / len(resolved_hubs)


# --- Relevance Resolution ---


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive lookup."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def resolve_relevant_ids(
    case: dict[str, Any],
    doc_store: Sequence[dict[str, Any]] | None = None,
    doc_name_map: dict[str, set[str]] | None = None,
    doc_chunk_map: dict[str, set[str]] | None = None,
) -> set[str]:
    """Resolve relevant document IDs for a test case.

    Combines explicit `relevant_ids` and doc IDs resolved by name matching
    against `relevant_names`. Also resolves chunk suffixes (_chunk_0, etc.).

    Exact normalized-name matches are always kept. The only substring fallback
    is the broadening direction (a doc name inside the query name — plurals,
    dropped modifiers); the reverse (query name inside doc name) is excluded
    because a short generic term like "Sword"/"Slash" substring-matches
    hundreds of unrelated docs and inflates the relevant set toward recall@k ≈ 0.
    """
    relevant: set[str] = set()

    for rid in case.get("relevant_ids", []):
        relevant.add(rid)

    relevant_names = [normalize_text(n) for n in case.get("relevant_names", []) if n]
    if relevant_names and doc_name_map:
        for name in relevant_names:
            if name in doc_name_map:
                relevant.update(doc_name_map[name])
            else:
                # Broadening catch ONLY: a doc name sitting inside the query
                # name (e.g. doc "Field Mushroom" inside query "Field
                # Mushrooms") — handles plurals and dropped modifiers. The
                # reverse direction (query name inside doc name) is the
                # fan-out trap: "Slash"/"Sword" substring-match hundreds of
                # unrelated docs and inflate |relevant| toward recall@k ≈ 0,
                # and it never supplies a target the explicit `relevant_ids`
                # don't already name — so it is deliberately excluded.
                for map_name, map_ids in doc_name_map.items():
                    if map_name in name:
                        relevant.update(map_ids)

    if doc_chunk_map and relevant:
        base_ids = {re.sub(r"_chunk_\d+$", "", rid) for rid in relevant}
        for base in base_ids:
            if base in doc_chunk_map:
                relevant.update(doc_chunk_map[base])

    if relevant:
        relevant = {canonical_doc_id(rid) for rid in relevant}

    return relevant


def build_doc_name_map(
    doc_store: Sequence[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build name_map (normalized name -> set of doc IDs) and chunk_map (base ID -> chunk IDs)."""
    name_map: dict[str, set[str]] = {}
    chunk_map: dict[str, set[str]] = {}

    for doc in doc_store:
        doc_id = doc.get("id", "")
        meta = doc.get("metadata", {})

        base_id = re.sub(r"_chunk_\d+$", "", doc_id)
        chunk_map.setdefault(base_id, set()).add(doc_id)

        names = []
        if meta.get("name"):
            names.append(meta["name"])
        if meta.get("title"):
            names.append(meta["title"])

        for name in names:
            norm = normalize_text(name)
            if norm:
                name_map.setdefault(norm, set()).add(doc_id)

        # Also map bare doc_id & base_id
        name_map.setdefault(normalize_text(doc_id), set()).add(doc_id)
        name_map.setdefault(normalize_text(base_id), set()).add(doc_id)

    return name_map, chunk_map


# --- Query Evaluation ---


_SUFFIX_RE = re.compile(r"_(?:chunk|row)_\d+$|_coverage$")


def canonical_doc_id(doc_id: str) -> str:
    """Reduce a doc id to its retrievable unit (page/section base).

    A single wiki table materializes as many index docs: a `_coverage` doc and
    one `_row_<n>` doc per row, plus `_chunk_<n>` splits on longer docs. IR
    metrics should count that cluster once, or recall@k is deflated by the row
    explosion and chunked ranked docs never match base relevant ids. Apply to
    BOTH relevant and ranked ids so exact-membership semantics are preserved
    (identical transformation on both sides).
    """
    return _SUFFIX_RE.sub("", doc_id)


# --- Gold-Set Integrity Validation ---


def validate_gold(
    cases: Sequence[dict[str, Any]],
    doc_store: Sequence[dict[str, Any]] | None = None,
    max_relevant: int = 40,
) -> dict[str, Any]:
    """Check that each case's gold `relevant_ids` resolve against the live corpus.

    Returns counts of (a) `relevant_ids` with no matching doc (stale/nonexistent)
    and (b) cases whose resolved relevant set exceeds `max_relevant` (fan-out
    that structurally caps recall@k near zero). Intentional read-only check that
    reports (via the returned dict and logger) rather than failing the run, so
    it stays usable as a hygiene signal during development.
    """
    if doc_store is None:
        docs_file = Path("data/documents.json")
        if not docs_file.exists():
            return {"missing_count": 0, "fanout_count": 0, "issues": []}
        try:
            doc_store = json.loads(docs_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("validate_gold: could not read documents.json: %s", exc)
            return {"missing_count": 0, "fanout_count": 0, "issues": []}

    all_ids = {canonical_doc_id(d.get("id", "")) for d in doc_store}
    name_map, _chunk_map = build_doc_name_map(doc_store)

    issues: list[dict[str, Any]] = []
    missing_count = 0
    fanout_count = 0

    for case in cases:
        cid = case.get("id", "unknown")
        missing = [r for r in case.get("relevant_ids", []) if canonical_doc_id(r) not in all_ids]
        if missing:
            missing_count += 1
            issues.append({"id": cid, "kind": "missing", "relevant_ids": missing})
            logger.warning(
                "gold-validation: %s has relevant_ids absent from the corpus: %s",
                cid,
                missing,
            )

        resolved = resolve_relevant_ids(case, doc_store=doc_store, doc_name_map=name_map)
        if len(resolved) > max_relevant:
            fanout_count += 1
            issues.append({"id": cid, "kind": "fanout", "relevant_size": len(resolved)})
            logger.warning(
                "gold-validation: %s relevant set is %d (fan-out; recall@k capped near zero)",
                cid,
                len(resolved),
            )

    return {"missing_count": missing_count, "fanout_count": fanout_count, "issues": issues}


def _dossier_coverage(dossier_ids: Sequence[str], relevant_ids: Sequence[str] | set[str]) -> float:
    """Fraction of canonical relevant units present anywhere in a context dossier.

    A comparison dossier is a context blob production feeds to the LLM in
    full — never a ranked top-k the pipeline truncates. Its "hit" question is
    whether each relevant unit is present at all, not whether it happens to
    sort into a top-5 of a non-production ordering (which overweights
    whichever entity's facet docs landed first). Canonicalized on both sides.
    """
    rel = {canonical_doc_id(r) for r in relevant_ids}
    if not rel:
        return 0.0
    present = {canonical_doc_id(d) for d in dossier_ids} & rel
    return len(present) / len(rel)


def compute_stage_metrics(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str] | set[str]
) -> dict[str, float]:
    """Compute all standard IR metrics for a single ranked list.

    Ids are canonicalized (row/coverage/chunk clusters count as one unit)
    and deduped order-preserving, so several chunks of the same doc are not
    counted as several hits and recall@k measures distinct retrieved units.
    """
    rel_set = {canonical_doc_id(r) for r in relevant_ids}
    seen: set[str] = set()
    ranked_ids = [
        t for r in ranked_ids if (t := canonical_doc_id(r)) not in seen and not seen.add(t)
    ]
    return {
        "recall@1": recall_at_k(ranked_ids, rel_set, 1),
        "recall@3": recall_at_k(ranked_ids, rel_set, 3),
        "recall@5": recall_at_k(ranked_ids, rel_set, 5),
        "recall@10": recall_at_k(ranked_ids, rel_set, 10),
        "recall@20": recall_at_k(ranked_ids, rel_set, 20),
        "mrr": reciprocal_rank(ranked_ids, rel_set),
        "ndcg@3": ndcg_at_k(ranked_ids, rel_set, 3),
        "ndcg@5": ndcg_at_k(ranked_ids, rel_set, 5),
        "ndcg@10": ndcg_at_k(ranked_ids, rel_set, 10),
        "hit@1": hit_at_k(ranked_ids, rel_set, 1),
        "hit@3": hit_at_k(ranked_ids, rel_set, 3),
        "hit@5": hit_at_k(ranked_ids, rel_set, 5),
    }


def evaluate_query(
    case: dict[str, Any],
    stages: Sequence[str] = ("dense", "bm25", "hybrid", "rerank", "entity", "comparison"),
    doc_store: Sequence[dict[str, Any]] | None = None,
    doc_name_map: dict[str, set[str]] | None = None,
    doc_chunk_map: dict[str, set[str]] | None = None,
    offline: bool = False,
    bm25_model: Any | None = None,
    bm25_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval stages on a single test case.

    Supports offline evaluation (BM25 + classifier + entity hubs) and
    full live evaluation (Chroma dense + hybrid RRF + cross-encoder rerank).
    """
    query = case["query"]
    relevant_ids = resolve_relevant_ids(
        case,
        doc_store=doc_store,
        doc_name_map=doc_name_map,
        doc_chunk_map=doc_chunk_map,
    )

    # 1. Query classification (mirrors pipeline.ask: classify_query drives routing,
    # including whether a query is a multi-entity comparison). The same label is
    # forwarded to retrieve() so the eval measures the production query path.
    query_type = classify_query(query)
    predicted_classifier = query_type
    expected_classifier = case.get("expected_classifier") or case.get("classifier")
    classifier_match = predicted_classifier == expected_classifier if expected_classifier else True

    # 2. Entity identification
    found_entities_list = find_entities(query)
    found_entity_names = [e[0] for e in found_entities_list]
    found_entity_hubs = [e[1] for e in found_entities_list]

    single_hub, _ = find_entity(query)
    if single_hub and single_hub not in found_entity_hubs:
        found_entity_hubs.append(single_hub)

    target_entities = case.get("target_entities") or case.get("entities", [])
    entity_score = entity_jaccard(
        found_entity_names or found_entity_hubs,
        target_entities,
    )

    # 3. Stage evaluation
    stage_results: dict[str, dict[str, Any]] = {}
    stage_latencies: dict[str, float] = {}

    # Check for live retrieval vs offline
    if not offline:
        try:
            from pgrag.rag.retriever import retrieve

            trace: dict[str, Any] = {}
            t0 = time.perf_counter()
            retrieve(
                query,
                count=20,
                hybrid=True,
                rerank=True,
                query_type=query_type,
                trace=trace,
            )
            total_lat = (time.perf_counter() - t0) * 1000.0

            calls = trace.get("retrieval_calls", [])
            if calls:
                rec = calls[0]
                if "dense" in stages and "dense_ids" in rec:
                    dense_ids = rec["dense_ids"]
                    stage_results["dense"] = {
                        "ranked_ids": dense_ids,
                        "metrics": compute_stage_metrics(dense_ids, relevant_ids),
                    }
                if "bm25" in stages and "bm25_ids" in rec:
                    bm25_ids = rec["bm25_ids"]
                    stage_results["bm25"] = {
                        "ranked_ids": bm25_ids,
                        "metrics": compute_stage_metrics(bm25_ids, relevant_ids),
                    }
                if "hybrid" in stages:
                    hybrid_ids = rec.get("post_filter_ids") or rec.get("rrf_ids") or []
                    stage_results["hybrid"] = {
                        "ranked_ids": hybrid_ids,
                        "metrics": compute_stage_metrics(hybrid_ids, relevant_ids),
                    }
                if "rerank" in stages and "reranked_ids" in rec:
                    rerank_ids = rec["reranked_ids"]
                    stage_results["rerank"] = {
                        "ranked_ids": rerank_ids,
                        "metrics": compute_stage_metrics(rerank_ids, relevant_ids),
                        "rerank_used": rec.get("rerank_used", False),
                    }
            stage_latencies["total"] = total_lat
        except Exception as exc:
            logger.warning(
                "Live retrieval failed for %r (%s); falling back to offline mode",
                query,
                exc,
            )
            offline = True

    if offline:
        # Offline BM25 evaluation
        if "bm25" in stages:
            t0 = time.perf_counter()
            if bm25_model is None or bm25_docs is None:
                from pgrag.rag.bm25 import load_bm25_index

                try:
                    bm25_model, bm25_docs = load_bm25_index()
                except Exception as exc:
                    logger.debug("BM25 index not loadable offline: %s", exc)
                    bm25_model, bm25_docs = None, None

            if bm25_model is not None and bm25_docs is not None:
                indices, _ = bm25_model.search(query, k=20)
                bm25_ids = [bm25_docs[i]["id"] for i in indices if i < len(bm25_docs)]
                stage_latencies["bm25"] = (time.perf_counter() - t0) * 1000.0
                stage_results["bm25"] = {
                    "ranked_ids": bm25_ids,
                    "metrics": compute_stage_metrics(bm25_ids, relevant_ids),
                }

        # Offline Entity hub evaluation (direct doc extraction without network calls)
        # Offline Entity hub evaluation using proper dossier builder (wiki-linked)
        if "entity" in stages and single_hub:
            t0 = time.perf_counter()
            # Load docs once for the offline builder
            if doc_store:
                docs = doc_store
            else:
                from pgrag.rag.entity_retrieval import _load_docs

                docs = _load_docs()
            dossier = build_entity_context_offline(single_hub, docs=docs)
            if dossier:
                stage_latencies["entity"] = (time.perf_counter() - t0) * 1000.0
                stage_results["entity"] = {
                    "ranked_ids": dossier["ids"][0],
                    "metrics": compute_stage_metrics(dossier["ids"][0], relevant_ids),
                }
    # 4. Comparison queries: score the multi-entity dossier production feeds
    # the LLM (pipeline.ask -> _prepare_multi_entity -> build_multi_entity_context),
    # not the single-hub rerank window (find_entity keeps only one subject).
    # Mirrors ask()'s `len(find_entities) >= 2` routing gate. Built whenever a
    # query classifies as a comparison, independent of which base retrieval
    # stages a caller requested (the dossier is what production consumes).
    if query_type == "comparison" and len(found_entities_list) >= 2:
        from pgrag.rag.entity_retrieval import build_multi_entity_context

        t0 = time.perf_counter()
        ctx = build_multi_entity_context(query, found_entities_list)
        if ctx is not None:
            comparison_ids = list(ctx["ids"][0])
            stage_results["comparison"] = {
                "ranked_ids": comparison_ids,
                "metrics": compute_stage_metrics(comparison_ids, relevant_ids),
                "coverage": _dossier_coverage(comparison_ids, relevant_ids),
                "entities": found_entity_names,
            }
            stage_latencies["comparison"] = (time.perf_counter() - t0) * 1000.0

    # Compute reranker uplift (NDCG@5_rerank - NDCG@5_hybrid)
    reranker_uplift = 0.0
    if "rerank" in stage_results and "hybrid" in stage_results:
        ndcg_rerank = stage_results["rerank"]["metrics"]["ndcg@5"]
        ndcg_hybrid = stage_results["hybrid"]["metrics"]["ndcg@5"]
        reranker_uplift = ndcg_rerank - ndcg_hybrid

    return {
        "id": case.get("id", "unknown"),
        "query": query,
        "category": case.get("category", "general"),
        "expected_classifier": expected_classifier,
        "predicted_classifier": predicted_classifier,
        "classifier_match": classifier_match,
        "target_entities": target_entities,
        "predicted_entities": found_entity_names or found_entity_hubs,
        "entity_accuracy": entity_score,
        "relevant_ids": sorted(relevant_ids),
        "stages": stage_results,
        "latencies_ms": stage_latencies,
        "reranker_uplift": reranker_uplift,
    }


# --- Benchmark Suite Aggregation ---


def load_benchmark_cases(path: Path | str) -> list[dict[str, Any]]:
    """Load benchmark cases from a .jsonl or .json file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark cases not found at: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl" or ("\n" in text.strip() and not text.strip().startswith("{")):
        cases = []
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
        return cases
    else:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "cases" in data:
            return data["cases"]
        return [data]


def aggregate_metrics(
    eval_list: Sequence[dict[str, Any]], stages: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Aggregate per-stage average metrics over a list of query evaluations."""
    out: dict[str, dict[str, float]] = {}
    metric_keys = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "recall@20",
        "mrr",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "hit@1",
        "hit@3",
        "hit@5",
    ]

    for stage in stages:
        present_evals = [e["stages"][stage]["metrics"] for e in eval_list if stage in e["stages"]]
        if not present_evals:
            continue
        n = len(present_evals)
        stage_avg: dict[str, float] = {}
        for k in metric_keys:
            stage_avg[k] = sum(m[k] for m in present_evals) / n
        stage_avg["cases_evaluated"] = float(n)
        out[stage] = stage_avg

    return out


def run_benchmark(
    cases_path: Path | str | Sequence[dict[str, Any]] = "evaluation/queries.jsonl",
    stages: Sequence[str] = ("dense", "bm25", "hybrid", "rerank", "entity", "comparison"),
    out_path: Path | str | None = None,
    offline: bool = False,
    doc_store: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run evaluation benchmark across all test cases.

    Returns structured summary including overall stage metrics, category breakdowns,
    classifier/entity resolution accuracy, reranker uplift, and per-query diagnostics.
    """
    if isinstance(cases_path, (str, Path)):
        cases = load_benchmark_cases(cases_path)
    else:
        cases = list(cases_path)

    if not cases:
        return {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "total_cases": 0,
            "offline": offline,
            "summary": {},
            "categories": {},
            "regressions": [],
            "queries": [],
        }

    # Preload doc store if needed for name resolution
    if doc_store is None:
        docs_file = Path("data/documents.json")
        if docs_file.exists():
            try:
                doc_store = json.loads(docs_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Failed to read documents.json: %s", exc)
                doc_store = None

    if doc_store:
        doc_name_map, doc_chunk_map = build_doc_name_map(doc_store)
    else:
        doc_name_map, doc_chunk_map = None, None

    # Gold-set integrity check (stale ids / fan-out) — hygiene signal, not a gate.
    gold_report = validate_gold(cases, doc_store=doc_store)
    # Preload BM25 model if running offline
    bm25_model, bm25_docs = None, None
    if offline:
        try:
            from pgrag.rag.bm25 import load_bm25_index

            bm25_model, bm25_docs = load_bm25_index()
        except Exception:
            pass

    evaluated_queries: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for case in cases:
        res = evaluate_query(
            case,
            stages=stages,
            doc_store=doc_store,
            doc_name_map=doc_name_map,
            doc_chunk_map=doc_chunk_map,
            offline=offline,
            bm25_model=bm25_model,
            bm25_docs=bm25_docs,
        )
        evaluated_queries.append(res)

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    # Aggregate overall metrics
    active_stages = sorted({stage for e in evaluated_queries for stage in e["stages"]})
    overall_stage_metrics = aggregate_metrics(evaluated_queries, active_stages)

    classifier_acc = (
        sum(1 for e in evaluated_queries if e["classifier_match"]) / len(evaluated_queries)
        if evaluated_queries
        else 0.0
    )
    entity_acc = (
        sum(e["entity_accuracy"] for e in evaluated_queries) / len(evaluated_queries)
        if evaluated_queries
        else 0.0
    )

    uplifts = [e["reranker_uplift"] for e in evaluated_queries if "rerank" in e["stages"]]
    avg_uplift = sum(uplifts) / len(uplifts) if uplifts else 0.0

    # Category breakdown
    categories: dict[str, dict[str, Any]] = {}
    cat_groups: dict[str, list[dict[str, Any]]] = {}
    for e in evaluated_queries:
        cat = e.get("category", "general")
        cat_groups.setdefault(cat, []).append(e)

    for cat, items in sorted(cat_groups.items()):
        cat_stages = aggregate_metrics(items, active_stages)
        cat_uplifts = [e["reranker_uplift"] for e in items if "rerank" in e["stages"]]
        categories[cat] = {
            "count": len(items),
            "stages": cat_stages,
            "reranker_uplift": sum(cat_uplifts) / len(cat_uplifts) if cat_uplifts else 0.0,
        }

    # Find missed queries / candidate regressions
    regressions = []
    for e in evaluated_queries:
        primary_stage = (
            "comparison"
            if "comparison" in e["stages"]
            else "rerank"
            if "rerank" in e["stages"]
            else "hybrid"
            if "hybrid" in e["stages"]
            else "bm25"
            if "bm25" in e["stages"]
            else "dense"
            if "dense" in e["stages"]
            else None
        )
        if primary_stage:
            # Comparison queries: the dossier is a full LLM context, so the
            # zero-flag is dossier coverage (relevant units present anywhere),
            # not recall@5 over its non-production ordering.
            zero = (
                e["stages"][primary_stage].get("coverage", 0.0) == 0.0
                if primary_stage == "comparison"
                else e["stages"][primary_stage]["metrics"]["recall@5"] == 0.0
            )
            if zero:
                regressions.append(
                    {
                        "id": e["id"],
                        "query": e["query"],
                        "category": e["category"],
                        "stage": primary_stage,
                        "recall@5": e["stages"][primary_stage].get("coverage", 0.0)
                        if primary_stage == "comparison"
                        else 0.0,
                        "mrr": e["stages"][primary_stage]["metrics"]["mrr"],
                        "relevant_ids": e["relevant_ids"],
                    }
                )

    benchmark_result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_cases": len(evaluated_queries),
        "total_time_ms": total_time_ms,
        "offline": offline,
        "gold_report": gold_report,
        "summary": {
            "classifier_accuracy": classifier_acc,
            "entity_accuracy": entity_acc,
            "reranker_uplift": avg_uplift,
            "stages": overall_stage_metrics,
        },
        "categories": categories,
        "regressions": regressions,
        "queries": evaluated_queries,
    }

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(benchmark_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return benchmark_result


# --- Comparison / Regression Diff Engine ---


def compare_benchmarks(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = 0.001
) -> dict[str, Any]:
    """Compare a current benchmark run against a baseline run.

    Computes deltas for overall metrics, categories, and identifies per-query
    regressions and improvements.
    """
    curr_summary = current.get("summary", {})
    base_summary = baseline.get("summary", {})

    stage_deltas: dict[str, dict[str, float]] = {}
    curr_stages = curr_summary.get("stages", {})
    base_stages = base_summary.get("stages", {})

    all_stages = sorted(set(curr_stages.keys()) | set(base_stages.keys()))
    for stg in all_stages:
        c_m = curr_stages.get(stg, {})
        b_m = base_stages.get(stg, {})
        keys = set(c_m.keys()) | set(b_m.keys())
        stg_delta: dict[str, float] = {}
        for k in keys:
            if k == "cases_evaluated":
                stg_delta[k] = c_m.get(k, 0.0) - b_m.get(k, 0.0)
            else:
                stg_delta[k] = c_m.get(k, 0.0) - b_m.get(k, 0.0)
        stage_deltas[stg] = stg_delta

    summary_delta = {
        "classifier_accuracy": curr_summary.get("classifier_accuracy", 0.0)
        - base_summary.get("classifier_accuracy", 0.0),
        "entity_accuracy": curr_summary.get("entity_accuracy", 0.0)
        - base_summary.get("entity_accuracy", 0.0),
        "reranker_uplift": curr_summary.get("reranker_uplift", 0.0)
        - base_summary.get("reranker_uplift", 0.0),
        "stages": stage_deltas,
    }

    # Query level diffs
    curr_q_map = {q["id"]: q for q in current.get("queries", [])}
    base_q_map = {q["id"]: q for q in baseline.get("queries", [])}

    regressions = []
    improvements = []
    unchanged_count = 0

    for qid, curr_q in curr_q_map.items():
        if qid not in base_q_map:
            continue
        base_q = base_q_map[qid]

        # Compare primary metric (NDCG@5 on highest available stage)
        curr_primary_stg = next(
            (
                s
                for s in ("comparison", "rerank", "hybrid", "bm25", "dense")
                if s in curr_q["stages"]
            ),
            None,
        )
        base_primary_stg = next(
            (
                s
                for s in ("comparison", "rerank", "hybrid", "bm25", "dense")
                if s in base_q["stages"]
            ),
            None,
        )

        if not curr_primary_stg or not base_primary_stg:
            unchanged_count += 1
            continue

        curr_ndcg = curr_q["stages"][curr_primary_stg]["metrics"]["ndcg@5"]
        base_ndcg = base_q["stages"][base_primary_stg]["metrics"]["ndcg@5"]
        curr_recall = curr_q["stages"][curr_primary_stg]["metrics"]["recall@5"]
        base_recall = base_q["stages"][base_primary_stg]["metrics"]["recall@5"]

        delta_ndcg = curr_ndcg - base_ndcg
        delta_recall = curr_recall - base_recall

        if delta_ndcg < -threshold or delta_recall < -threshold:
            regressions.append(
                {
                    "id": qid,
                    "query": curr_q["query"],
                    "stage": curr_primary_stg,
                    "delta_ndcg@5": delta_ndcg,
                    "delta_recall@5": delta_recall,
                    "curr_ndcg@5": curr_ndcg,
                    "base_ndcg@5": base_ndcg,
                }
            )
        elif delta_ndcg > threshold or delta_recall > threshold:
            improvements.append(
                {
                    "id": qid,
                    "query": curr_q["query"],
                    "stage": curr_primary_stg,
                    "delta_ndcg@5": delta_ndcg,
                    "delta_recall@5": delta_recall,
                    "curr_ndcg@5": curr_ndcg,
                    "base_ndcg@5": base_ndcg,
                }
            )
        else:
            unchanged_count += 1

    return {
        "summary_delta": summary_delta,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged_count": unchanged_count,
        "current_total": current.get("total_cases", 0),
        "baseline_total": baseline.get("total_cases", 0),
    }
