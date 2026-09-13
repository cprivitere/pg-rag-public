"""Offline retrieval metrics and benchmark evaluation suite tests.

Tests pure deterministic IR metric functions (Recall@k, MRR, NDCG@k, Hit@k,
Entity Jaccard, Entity Accuracy), stage metrics computation, relevance ID
resolution, comparison diff engine, and offline benchmark runner.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from pgrag.rag.retrieval_eval import (
    build_doc_name_map,
    canonical_doc_id,
    compare_benchmarks,
    compute_stage_metrics,
    dcg_at_k,
    entity_accuracy,
    entity_jaccard,
    evaluate_query,
    hit_at_k,
    load_benchmark_cases,
    mrr,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    resolve_relevant_ids,
    run_benchmark,
)

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_cases.json"
BENCHMARK_DATASET = Path("evaluation/queries.jsonl")


# --- Core IR Metric Tests ---


def test_exact_metric_values():
    ranked = [["d1", "d3", "d2"], ["d2", "d5", "d4"], ["d1", "d2", "d5"]]
    relevant = [["d1"], ["d2", "d4"], ["d5"]]
    queries = list(zip(ranked, relevant, strict=False))

    # q1: d1 at rank 1
    assert recall_at_k(ranked[0], relevant[0], 3) == 1.0
    # q2: d2 in top-2, d4 at rank 3 -> Recall@2 = 1/2
    assert recall_at_k(ranked[1], relevant[1], 2) == 0.5
    # q3: d5 at rank 3 -> Recall@3 = 1/1
    assert recall_at_k(ranked[2], relevant[2], 3) == 1.0

    # MRR: (1 + 1 + 1/3) / 3 = 7/9
    assert mrr(queries) == pytest.approx(7 / 9)

    # NDCG@2, q2: DCG = 1/log2(2)=1.0 (d2); IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(ranked[1], relevant[1], 2) == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))
    # perfect ranking -> NDCG 1
    assert ndcg_at_k(["d2", "d4", "d5"], ["d2", "d4"], 2) == 1.0


def test_recall_mrr_detect_rank_regression():
    """Swapping a relevant doc out of top-1 must change Recall@1 and MRR."""
    good = ["d1", "d2"]
    bad = ["d2", "d1"]  # d1 (the relevant one) dropped from top-1
    relevant = ["d1"]

    assert recall_at_k(good, relevant, 1) == 1.0
    assert recall_at_k(bad, relevant, 1) == 0.0
    assert reciprocal_rank(good, relevant) == 1.0
    assert reciprocal_rank(bad, relevant) == 0.5


def test_metric_edge_cases_empty_inputs():
    """Empty ranked lists, empty relevant lists, and k <= 0 must return 0.0 safely."""
    # Empty relevant
    assert recall_at_k(["d1", "d2"], [], 5) == 0.0
    assert reciprocal_rank(["d1", "d2"], []) == 0.0
    assert dcg_at_k(["d1", "d2"], [], 5) == 0.0
    assert ndcg_at_k(["d1", "d2"], [], 5) == 0.0
    assert hit_at_k(["d1", "d2"], [], 5) == 0.0

    # Empty ranked
    assert recall_at_k([], ["d1"], 5) == 0.0
    assert reciprocal_rank([], ["d1"]) == 0.0
    assert dcg_at_k([], ["d1"], 5) == 0.0
    assert ndcg_at_k([], ["d1"], 5) == 0.0
    assert hit_at_k([], ["d1"], 5) == 0.0

    # k <= 0
    assert recall_at_k(["d1"], ["d1"], 0) == 0.0
    assert recall_at_k(["d1"], ["d1"], -1) == 0.0
    assert dcg_at_k(["d1"], ["d1"], 0) == 0.0
    assert dcg_at_k(["d1"], ["d1"], -2) == 0.0
    assert ndcg_at_k(["d1"], ["d1"], 0) == 0.0
    assert hit_at_k(["d1"], ["d1"], 0) == 0.0

    # Empty queries list for MRR
    assert mrr([]) == 0.0


def test_canonical_doc_id_strips_suffixes():
    """Row, coverage, and chunk suffixes collapse to the retrievable unit."""
    assert canonical_doc_id("wiki_Spider_Silk_table_0_row_3") == "wiki_Spider_Silk_table_0"
    assert canonical_doc_id("wiki_Spider_Silk_table_0_coverage") == "wiki_Spider_Silk_table_0"
    assert canonical_doc_id("wiki_Spider_Silk_Uses") == "wiki_Spider_Silk_Uses"
    assert canonical_doc_id("recipe_8533_chunk_0") == "recipe_8533"
    assert canonical_doc_id("recipe_8533") == "recipe_8533"
    # Multiple row indices beyond 9
    assert canonical_doc_id("wiki_X_table_1_row_12") == "wiki_X_table_1"


def test_compute_stage_metrics_collapses_row_cluster():
    """A wiki table's rows count once: ranking one row hits the whole cluster."""
    rows = [f"wiki_Spider_Silk_table_0_row_{i}" for i in range(25)]
    # ranked surfaces a single row; relevant is the full 25-row cluster
    metrics = compute_stage_metrics([rows[0], *["unrelated"] * 10], [rows[0], *rows[1:]])
    assert metrics["hit@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    # Without canonicalization this would be recall@1 == 0.0 (24/25 missed)
    assert metrics["recall@5"] == 1.0


def test_compute_stage_metrics_matches_chunked_ranked_to_base_relevant():
    """Ranked chunk docs reconcile against base (uncannonical) relevant ids."""
    ranked = ["recipe_8533_chunk_0", "recipe_8533_chunk_1", "other"]
    metrics = compute_stage_metrics(ranked, ["recipe_8533"])
    assert metrics["hit@3"] == 1.0
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_compute_stage_metrics_dedupes_chunk_cluster():
    """Multiple chunks of the same doc count once: recall measures distinct units."""
    # 5 chunks of one doc + a missed second doc -> only X recovered
    ranked = [f"recipe_8533_chunk_{i}" for i in range(5)]
    metrics = compute_stage_metrics(ranked, ["recipe_8533", "recipe_9999"])
    assert metrics["hit@5"] == 1.0
    assert metrics["recall@5"] == 0.5  # 1 distinct unit of 2 relevant recovered
    assert metrics["mrr"] == pytest.approx(1.0)  # dedupe keeps rank 1


def test_metric_boundary_large_k():
    """k larger than ranked list length must compute gracefully."""
    ranked = ["d1", "d2"]
    relevant = ["d1", "d3"]

    # Recall at k=10 with 2 items in ranked
    assert recall_at_k(ranked, relevant, 10) == 0.5
    assert hit_at_k(ranked, relevant, 10) == 1.0
    # DCG at k=10 should stop after available items
    assert dcg_at_k(ranked, relevant, 10) == 1.0 / math.log2(2)
    # NDCG with k=10
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0 / idcg)


def test_hit_at_k_behavior():
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = ["d4"]

    assert hit_at_k(ranked, relevant, 1) == 0.0
    assert hit_at_k(ranked, relevant, 3) == 0.0
    assert hit_at_k(ranked, relevant, 4) == 1.0
    assert hit_at_k(ranked, relevant, 5) == 1.0
    assert hit_at_k(ranked, ["d99"], 5) == 0.0


def test_entity_jaccard_and_accuracy():
    # Exact match
    assert entity_jaccard(["Punch", "Front Kick"], ["punch", "front kick"]) == 1.0
    # Disjoint
    assert entity_jaccard(["Punch"], ["Fireball"]) == 0.0
    # Partial overlap: union 3, intersection 1 -> 1/3
    assert entity_jaccard(["Punch", "Fireball"], ["Punch", "Front Kick"]) == pytest.approx(1 / 3)
    # Both empty
    assert entity_jaccard([], []) == 1.0
    # One empty
    assert entity_jaccard(["Punch"], []) == 0.0
    assert entity_jaccard([], ["Punch"]) == 0.0

    # List accuracy
    assert entity_accuracy(["Punch", "Front Kick"], ["Punch", "Front Kick"]) == 1.0
    assert entity_accuracy(["Punch", "Punch"], ["Punch", "Front Kick"]) == 0.5
    assert entity_accuracy([], ["Punch"]) == 0.0
    assert entity_accuracy([], []) == 1.0


def test_compute_stage_metrics():
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = ["d2", "d4"]
    metrics = compute_stage_metrics(ranked, relevant)

    assert set(metrics.keys()) == {
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
    }
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.5  # d2 in top 3
    assert metrics["recall@5"] == 1.0  # d2 and d4 in top 5
    assert metrics["mrr"] == 0.5  # d2 at rank 2 -> 1/2
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0


# --- Relevance Resolution Tests ---


def test_resolve_relevant_ids_with_names_and_chunks():
    sample_docs = [
        {
            "id": "recipe_1001_chunk_0",
            "metadata": {"name": "Spider Silk Tunic", "type": "recipe"},
            "text": "Spider silk recipe details",
        },
        {
            "id": "recipe_1001_chunk_1",
            "metadata": {"name": "Spider Silk Tunic", "type": "recipe"},
            "text": "More details",
        },
        {
            "id": "item_5001",
            "metadata": {"name": "Spider Silk", "type": "item"},
            "text": "Raw spider silk",
        },
    ]
    name_map, chunk_map = build_doc_name_map(sample_docs)

    case = {
        "id": "test-case",
        "query": "spider silk recipe",
        "relevant_ids": ["recipe_1001"],
        "relevant_names": ["Spider Silk Tunic"],
    }

    resolved = resolve_relevant_ids(
        case,
        doc_store=sample_docs,
        doc_name_map=name_map,
        doc_chunk_map=chunk_map,
    )

    assert "recipe_1001" in resolved
    # chunk fans collapse to the canonical base (recipe_1001)
    assert "recipe_1001_chunk_0" not in resolved
    assert "recipe_1001_chunk_1" not in resolved


def test_resolve_relevant_ids_collapses_wiki_row_coverage_fans():
    """Wiki table row/coverage fans collapse to their page/table base in |relevant|."""
    docs_by_id = {
        # One wiki page table -> coverage + many rows
        "wiki_Field_Mushroom_table_0_coverage": {"name": "Field Mushroom"},
        **{f"wiki_Field_Mushroom_table_0_row_{i}": {"name": "Field Mushroom"} for i in range(12)},
        # Distinct legitimate targets must survive canonicalization
        "item_11004": {"name": "Field Mushroom"},
        "wiki_Field Mushroom_Uses": {"name": "Field Mushroom"},
        "skill_Mycology": {"name": "Mycology"},
    }
    sample_docs = [
        {"id": did, "metadata": meta, "text": "content"} for did, meta in docs_by_id.items()
    ]
    name_map, chunk_map = build_doc_name_map(sample_docs)

    case = {
        "id": "grow-field",
        "query": "how to grow field mushrooms",
        "relevant_names": ["Field Mushroom", "Mycology"],
    }
    resolved = resolve_relevant_ids(
        case,
        doc_store=sample_docs,
        doc_name_map=name_map,
        doc_chunk_map=chunk_map,
    )

    # The 12-row + coverage fan of one table collapses to a single canonical base
    assert "wiki_Field_Mushroom_table_0" in resolved
    assert not any("_row_" in r for r in resolved)
    assert not any("_coverage" in r for r in resolved)
    # Distinct targets stay: item, wiki narrative page, skill
    assert "item_11004" in resolved
    assert "wiki_Field Mushroom_Uses" in resolved
    assert "skill_Mycology" in resolved
    # 14 raw Field-Mushroom members + 1 Mycology -> 4 canonical units (not 13+)
    assert len(resolved) == 4


def test_evaluate_query_passes_query_type_to_retrieve(monkeypatch):
    """evaluate_query forwards classify_query's label to retrieve (production parity)."""
    captured: dict[str, Any] = {}

    def fake_retrieve(
        question,
        count=3,
        metadata_filter=None,
        token_filter=None,
        rerank=True,
        hybrid=False,
        query_type="general",
        trace=None,
    ):
        captured["query_type"] = query_type
        captured["count"] = count
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "rerank_used": False,
        }

    monkeypatch.setattr("pgrag.rag.retriever.retrieve", fake_retrieve)
    monkeypatch.setattr("pgrag.rag.retrieval_eval.classify_query", lambda q: "comparison")
    monkeypatch.setattr(
        "pgrag.rag.retrieval_eval.find_entities",
        lambda q: [
            ("Fireball", "ability_ability_3502", "ability"),
            ("Fire Breath", "ability_ability_3607", "ability"),
        ],
    )
    monkeypatch.setattr(
        "pgrag.rag.retrieval_eval.find_entity", lambda q: ("ability_ability_3607", None)
    )

    case = {
        "id": "fireball-vs-firebreath",
        "query": "What is the difference between Fireball and Fire Breath?",
        "expected_classifier": "comparison",
        "target_entities": ["Fireball", "Fire Breath"],
        "relevant_ids": ["ability_ability_3502", "ability_ability_3607"],
    }
    evaluate_query(case, stages=("dense", "comparison"))

    assert captured.get("query_type") == "comparison"
    assert captured.get("count") == 20


def test_comparison_routes_multientity_dossier_both_subjects(monkeypatch):
    """Comparison queries are scored on the multi-entity dossier (both subjects)."""
    captured: dict[str, Any] = {}

    def fake_retrieve(
        question,
        count=3,
        metadata_filter=None,
        token_filter=None,
        rerank=True,
        hybrid=False,
        query_type="general",
        trace=None,
    ):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "rerank_used": False,
        }

    # The multi-entity dossier production feeds the LLM: both subjects present
    def fake_multi(question, entities, trace=None):
        captured["entities"] = entities
        return {
            "ids": [["ability_ability_3502_chunk_0", "ability_ability_3607_chunk_0"]],
            "documents": [["a", "b"]],
            "metadatas": [[]],
            "distances": [[]],
            "rerank_used": False,
        }

    monkeypatch.setattr("pgrag.rag.retriever.retrieve", fake_retrieve)
    monkeypatch.setattr("pgrag.rag.retrieval_eval.classify_query", lambda q: "comparison")
    monkeypatch.setattr(
        "pgrag.rag.retrieval_eval.find_entities",
        lambda q: [
            ("Fireball", "ability_ability_3502", "ability"),
            ("Fire Breath", "ability_ability_3607", "ability"),
        ],
    )
    monkeypatch.setattr(
        "pgrag.rag.retrieval_eval.find_entity", lambda q: ("ability_ability_3607", None)
    )
    monkeypatch.setattr("pgrag.rag.entity_retrieval.build_multi_entity_context", fake_multi)

    case = {
        "id": "fireball-vs-firebreath",
        "query": "What is the difference between Fireball and Fire Breath?",
        "expected_classifier": "comparison",
        "target_entities": ["Fireball", "Fire Breath"],
        "relevant_ids": ["ability_ability_3502", "ability_ability_3607"],
    }
    # "comparison" not listed in stages: the dossier is scored automatically
    # for comparison queries (production parity).
    res = evaluate_query(case, stages=("dense",))

    assert res["stages"]["comparison"]["entities"] == ["Fireball", "Fire Breath"]
    # Both subject hubs forwarded to the dossier builder
    hubs = [e[1] for e in captured.get("entities", [])]
    assert "ability_ability_3502" in hubs and "ability_ability_3607" in hubs
    # Both subjects' docs are in the dossier (metrics count each canonical unit)
    assert res["stages"]["comparison"]["metrics"]["recall@1"] == 0.5
    assert res["stages"]["comparison"]["metrics"]["recall@5"] == 1.0
    # Production feeds the whole dossier, so coverage measures both subjects present
    assert res["stages"]["comparison"]["coverage"] == 1.0


# --- Comparison Diff Engine Tests ---


def test_compare_benchmarks_deltas_and_regressions():
    current_run = {
        "total_cases": 2,
        "summary": {
            "classifier_accuracy": 0.90,
            "entity_accuracy": 0.85,
            "reranker_uplift": 0.10,
            "stages": {
                "hybrid": {
                    "recall@1": 0.50,
                    "recall@5": 0.80,
                    "mrr": 0.65,
                    "ndcg@5": 0.70,
                    "hit@5": 0.85,
                },
                "rerank": {
                    "recall@1": 0.60,
                    "recall@5": 0.90,
                    "mrr": 0.75,
                    "ndcg@5": 0.80,
                    "hit@5": 0.95,
                },
            },
        },
        "queries": [
            {
                "id": "q1",
                "query": "query 1",
                "stages": {
                    "rerank": {"metrics": {"ndcg@5": 0.90, "recall@5": 1.0}},
                },
            },
            {
                "id": "q2",
                "query": "query 2",
                "stages": {
                    "rerank": {"metrics": {"ndcg@5": 0.40, "recall@5": 0.50}},
                },
            },
        ],
    }

    baseline_run = {
        "total_cases": 2,
        "summary": {
            "classifier_accuracy": 0.80,
            "entity_accuracy": 0.80,
            "reranker_uplift": 0.05,
            "stages": {
                "hybrid": {
                    "recall@1": 0.40,
                    "recall@5": 0.70,
                    "mrr": 0.55,
                    "ndcg@5": 0.60,
                    "hit@5": 0.75,
                },
                "rerank": {
                    "recall@1": 0.50,
                    "recall@5": 0.80,
                    "mrr": 0.65,
                    "ndcg@5": 0.70,
                    "hit@5": 0.85,
                },
            },
        },
        "queries": [
            {
                "id": "q1",
                "query": "query 1",
                "stages": {
                    "rerank": {"metrics": {"ndcg@5": 0.70, "recall@5": 0.80}},
                },
            },
            {
                "id": "q2",
                "query": "query 2",
                "stages": {
                    "rerank": {"metrics": {"ndcg@5": 0.80, "recall@5": 1.0}},
                },
            },
        ],
    }

    diff = compare_benchmarks(current_run, baseline_run)

    # Summary deltas
    assert diff["summary_delta"]["classifier_accuracy"] == pytest.approx(0.10)
    assert diff["summary_delta"]["reranker_uplift"] == pytest.approx(0.05)
    assert diff["summary_delta"]["stages"]["rerank"]["ndcg@5"] == pytest.approx(0.10)

    # Query level regressions & improvements
    assert len(diff["improvements"]) == 1
    assert diff["improvements"][0]["id"] == "q1"
    assert diff["improvements"][0]["delta_ndcg@5"] == pytest.approx(0.20)

    assert len(diff["regressions"]) == 1
    assert diff["regressions"][0]["id"] == "q2"
    assert diff["regressions"][0]["delta_ndcg@5"] == pytest.approx(-0.40)


# --- Dataset Loading and Benchmark Runner Integration ---


def test_load_benchmark_cases_jsonl():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"id": "q1", "query": "what is punch"}) + "\n")
        f.write(json.dumps({"id": "q2", "query": "what is fireball"}) + "\n")
        tmp_path = Path(f.name)

    try:
        cases = load_benchmark_cases(tmp_path)
        assert len(cases) == 2
        assert cases[0]["id"] == "q1"
        assert cases[1]["id"] == "q2"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_run_benchmark_offline_synthetic():
    """Test run_benchmark runs completely offline against mock test cases."""
    cases = [
        {
            "id": "test-punch",
            "query": "what is the Punch ability?",
            "expected_classifier": "entity",
            "target_entities": ["Punch"],
            "relevant_ids": ["ability_punch_1"],
            "category": "abilities",
        },
        {
            "id": "test-fireball",
            "query": "what is Fireball?",
            "expected_classifier": "entity",
            "target_entities": ["Fireball"],
            "relevant_ids": ["ability_fireball_1"],
            "category": "abilities",
        },
    ]

    mock_doc_store = [
        {
            "id": "ability_punch_1",
            "metadata": {"name": "Punch", "type": "ability"},
            "text": "Punch info",
        },
        {
            "id": "ability_fireball_1",
            "metadata": {"name": "Fireball", "type": "ability"},
            "text": "Fireball info",
        },
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        tmp_out = Path(f.name)

    try:
        res = run_benchmark(
            cases_path=cases,
            stages=("entity",),
            out_path=tmp_out,
            offline=True,
            doc_store=mock_doc_store,
        )

        assert res["total_cases"] == 2
        assert "summary" in res
        assert "classifier_accuracy" in res["summary"]
        assert "entity_accuracy" in res["summary"]
        assert "categories" in res
        assert "abilities" in res["categories"]
        assert tmp_out.exists()
        saved = json.loads(tmp_out.read_text(encoding="utf-8"))
        assert saved["total_cases"] == 2
    finally:
        tmp_out.unlink(missing_ok=True)


def test_benchmark_dataset_fixture_present_and_valid():
    """Verify evaluation/queries.jsonl exists and contains valid test cases."""
    assert BENCHMARK_DATASET.exists(), "evaluation/queries.jsonl must exist"
    cases = load_benchmark_cases(BENCHMARK_DATASET)
    assert len(cases) >= 30, f"Expected at least 30 benchmark queries, found {len(cases)}"

    for case in cases:
        assert isinstance(case.get("id"), str) and case["id"]
        assert isinstance(case.get("query"), str) and case["query"]
        assert isinstance(case.get("category"), str) and case["category"]
        assert "relevant_ids" in case or "relevant_names" in case


# --- Canonical Fixture Conformance (Preserved from Existing Suite) ---


def test_fixture_has_all_canonical_cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = {c["id"] for c in data["cases"]}
    assert {
        "punch-vs-front-kick",
        "grow-field-mushrooms",
        "field-mushroom-locations",
        "mushroom-farming-gathering",
        "spider-silk-recipe",
    } <= ids


@pytest.mark.parametrize(
    "case_id",
    [
        "punch-vs-front-kick",
        "grow-field-mushrooms",
        "field-mushroom-locations",
        "mushroom-farming-gathering",
        "spider-silk-recipe",
    ],
)
def test_fixture_case_well_formed(case_id: str):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    case = next(c for c in data["cases"] if c["id"] == case_id)
    assert isinstance(case["query"], str) and case["query"]
    assert case["classifier"] in {"general", "lookup", "comparison"}
    assert isinstance(case["expected_ranked_ids"], list)
    assert isinstance(case["relevant_ids"], list) and case["relevant_ids"]


# --- CLI main() Integration Tests ---


def test_cli_main_compare_different_baseline(monkeypatch, capsys):
    """Verify scripts.retrieval_eval.main() correctly prints delta diff when comparing against baseline."""
    import sys

    from scripts.retrieval_eval import main as cli_main

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f_dataset:
        f_dataset.write(
            json.dumps(
                {
                    "id": "test-q1",
                    "query": "what is punch",
                    "expected_classifier": "entity",
                    "target_entities": ["Punch"],
                    "relevant_ids": ["ability_punch_1"],
                    "category": "abilities",
                }
            )
            + "\n"
        )
        dataset_path = Path(f_dataset.name)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_base:
        # Baseline with artificially lower scores and different q1 metric to create a non-zero diff
        baseline_content = {
            "total_cases": 1,
            "summary": {
                "classifier_accuracy": 0.50,
                "entity_accuracy": 0.50,
                "reranker_uplift": 0.0,
                "stages": {
                    "entity": {
                        "recall@1": 0.10,
                        "recall@5": 0.20,
                        "mrr": 0.20,
                        "ndcg@5": 0.15,
                        "hit@5": 0.20,
                    }
                },
            },
            "queries": [
                {
                    "id": "test-q1",
                    "query": "what is punch",
                    "stages": {
                        "entity": {
                            "metrics": {
                                "ndcg@5": 0.15,
                                "recall@5": 0.20,
                            }
                        }
                    },
                }
            ],
        }
        f_base.write(json.dumps(baseline_content, indent=2))
        baseline_path = Path(f_base.name)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f_out:
        out_path = Path(f_out.name)

    try:
        test_argv = [
            "retrieval_eval.py",
            "--dataset",
            str(dataset_path),
            "--stages",
            "entity",
            "--offline",
            "--compare",
            str(baseline_path),
            "--out",
            str(out_path),
        ]
        monkeypatch.setattr(sys, "argv", test_argv)

        exit_code = cli_main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "RETRIEVAL BENCHMARK DELTA (Current vs Baseline)" in captured.out
        assert "Classifier Accuracy Delta:" in captured.out
        assert "Stage Metric Deltas" in captured.out
        assert out_path.exists()
    finally:
        dataset_path.unlink(missing_ok=True)
        baseline_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def test_cli_main_compare_same_out_path_preserves_comparison(monkeypatch, capsys):
    """Verify comparison baseline is loaded before --out overwrites it when both point to the same path."""
    import sys

    from scripts.retrieval_eval import main as cli_main

    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f_dataset:
        f_dataset.write(
            json.dumps(
                {
                    "id": "test-q1",
                    "query": "what is punch",
                    "expected_classifier": "entity",
                    "target_entities": ["Punch"],
                    "relevant_ids": ["ability_punch_1"],
                    "category": "abilities",
                }
            )
            + "\n"
        )
        dataset_path = Path(f_dataset.name)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f_shared:
        initial_baseline = {
            "total_cases": 1,
            "summary": {
                "classifier_accuracy": 0.20,
                "entity_accuracy": 0.20,
                "reranker_uplift": 0.0,
                "stages": {
                    "entity": {
                        "recall@1": 0.0,
                        "recall@5": 0.0,
                        "mrr": 0.0,
                        "ndcg@5": 0.0,
                        "hit@5": 0.0,
                    }
                },
            },
            "queries": [
                {
                    "id": "test-q1",
                    "query": "what is punch",
                    "stages": {
                        "entity": {
                            "metrics": {
                                "ndcg@5": 0.0,
                                "recall@5": 0.0,
                            }
                        }
                    },
                }
            ],
        }
        f_shared.write(json.dumps(initial_baseline, indent=2))
        shared_path = Path(f_shared.name)

    try:
        test_argv = [
            "retrieval_eval.py",
            "--dataset",
            str(dataset_path),
            "--stages",
            "entity",
            "--offline",
            "--compare",
            str(shared_path),
            "--out",
            str(shared_path),
        ]
        monkeypatch.setattr(sys, "argv", test_argv)

        exit_code = cli_main()
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "RETRIEVAL BENCHMARK DELTA (Current vs Baseline)" in captured.out
        assert "Classifier Accuracy Delta:" in captured.out
        # The file should have been overwritten with the new results
        saved = json.loads(shared_path.read_text(encoding="utf-8"))
        assert saved["summary"]["classifier_accuracy"] != 0.20
    finally:
        dataset_path.unlink(missing_ok=True)
        shared_path.unlink(missing_ok=True)


def test_cli_main_compare_missing_file_fails_fast(monkeypatch, capsys):
    """Verify scripts.retrieval_eval.main() exits 1 if comparison baseline does not exist."""
    import sys

    from scripts.retrieval_eval import main as cli_main

    test_argv = [
        "retrieval_eval.py",
        "--compare",
        "non_existent_baseline_file_12345.json",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)

    exit_code = cli_main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: comparison baseline not found" in captured.err
