#!/usr/bin/env python3
"""CLI Benchmark Runner for Pre-LLM Retrieval Evaluation Suite.

Evaluates information retrieval metrics (Recall@k, MRR, NDCG@k, Hit@k, Reranker Uplift,
Entity Resolution Accuracy) per retrieval stage across an extensible query dataset.
Supports offline evaluation, diff/regression comparisons, and structured JSON output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pgrag.rag.retrieval_eval import (
    compare_benchmarks,
    run_benchmark,
)


def format_table(
    headers: list[str], rows: list[list[str]], alignments: list[str] | None = None
) -> str:
    """Format a clean ASCII table with header borders and aligned columns."""
    if not rows:
        return ""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    if alignments is None:
        alignments = ["<"] + [">"] * (len(headers) - 1)

    def format_row(cells: list[str]) -> str:
        formatted = []
        for i, cell in enumerate(cells):
            w = col_widths[i]
            align = alignments[i] if i < len(alignments) else "<"
            if align == ">":
                formatted.append(f"{cell!s:>{w}}")
            else:
                formatted.append(f"{cell!s:<{w}}")
        return "| " + " | ".join(formatted) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines = [
        sep,
        format_row(headers),
        sep,
    ]
    for row in rows:
        lines.append(format_row(row))
    lines.append(sep)
    return "\n".join(lines)


def print_stage_table(stages_data: dict[str, dict[str, float]]) -> None:
    """Print the stage comparison metric table."""
    headers = [
        "Stage",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR",
        "NDCG@5",
        "Hit@5",
        "Cases",
    ]
    rows = []
    stage_order = ["dense", "bm25", "hybrid", "rerank", "entity"]
    sorted_stages = sorted(
        stages_data.keys(),
        key=lambda s: stage_order.index(s) if s in stage_order else 99,
    )
    for stg in sorted_stages:
        m = stages_data[stg]
        rows.append(
            [
                stg,
                f"{m.get('recall@1', 0.0):.4f}",
                f"{m.get('recall@5', 0.0):.4f}",
                f"{m.get('recall@10', 0.0):.4f}",
                f"{m.get('mrr', 0.0):.4f}",
                f"{m.get('ndcg@5', 0.0):.4f}",
                f"{m.get('hit@5', 0.0):.4f}",
                f"{int(m.get('cases_evaluated', 0))}",
            ]
        )
    print(format_table(headers, rows))


def print_category_table(categories: dict[str, dict[str, Any]]) -> None:
    """Print the category breakdown table."""
    if not categories:
        return
    headers = [
        "Category",
        "Cases",
        "BM25 NDCG@5",
        "Hybrid NDCG@5",
        "Rerank NDCG@5",
        "Uplift",
    ]
    rows = []
    for cat, data in sorted(categories.items()):
        stgs = data.get("stages", {})
        bm25_n5 = f"{stgs['bm25']['ndcg@5']:.4f}" if "bm25" in stgs else "-"
        hyb_n5 = f"{stgs['hybrid']['ndcg@5']:.4f}" if "hybrid" in stgs else "-"
        rr_n5 = f"{stgs['rerank']['ndcg@5']:.4f}" if "rerank" in stgs else "-"
        uplift = data.get("reranker_uplift", 0.0)
        uplift_str = f"{uplift:+.4f}" if "rerank" in stgs else "-"
        rows.append(
            [
                cat,
                str(data.get("count", 0)),
                bm25_n5,
                hyb_n5,
                rr_n5,
                uplift_str,
            ]
        )
    print("\n--- Category Breakdown ---")
    print(format_table(headers, rows))


def print_comparison_diff(comparison: dict[str, Any]) -> None:
    """Print the delta comparison against a baseline run."""
    print("\n" + "=" * 60)
    print("  RETRIEVAL BENCHMARK DELTA (Current vs Baseline)")
    print("=" * 60)

    summary_delta = comparison.get("summary_delta", {})
    cls_d = summary_delta.get("classifier_accuracy", 0.0)
    ent_d = summary_delta.get("entity_accuracy", 0.0)
    upl_d = summary_delta.get("reranker_uplift", 0.0)

    print(f"Classifier Accuracy Delta: {cls_d:+.2%}")
    print(f"Entity Accuracy Delta:     {ent_d:+.2%}")
    print(f"Reranker Uplift Delta:     {upl_d:+.4f}")

    stage_deltas = summary_delta.get("stages", {})
    if stage_deltas:
        headers = [
            "Stage",
            "Δ Recall@1",
            "Δ Recall@5",
            "Δ MRR",
            "Δ NDCG@5",
            "Δ Hit@5",
        ]
        rows = []
        for stg, deltas in sorted(stage_deltas.items()):
            rows.append(
                [
                    stg,
                    f"{deltas.get('recall@1', 0.0):+.4f}",
                    f"{deltas.get('recall@5', 0.0):+.4f}",
                    f"{deltas.get('mrr', 0.0):+.4f}",
                    f"{deltas.get('ndcg@5', 0.0):+.4f}",
                    f"{deltas.get('hit@5', 0.0):+.4f}",
                ]
            )
        print("\n--- Stage Metric Deltas ---")
        print(format_table(headers, rows))

    regressions = comparison.get("regressions", [])
    improvements = comparison.get("improvements", [])

    if regressions:
        print(f"\n[!] REGRESSIONS DETECTED ({len(regressions)}):")
        for r in regressions:
            print(
                f"  - [{r['id']}] {r['query']}\n"
                f"    Stage: {r['stage']} | ΔNDCG@5: {r['delta_ndcg@5']:+.4f} ({r['base_ndcg@5']:.4f} -> {r['curr_ndcg@5']:.4f}) | ΔRecall@5: {r['delta_recall@5']:+.4f}"
            )
    else:
        print("\n[OK] No query regressions detected.")

    if improvements:
        print(f"\n[*] IMPROVEMENTS OBSERVED ({len(improvements)}):")
        for imp in improvements:
            print(
                f"  + [{imp['id']}] {imp['query']}\n"
                f"    Stage: {imp['stage']} | ΔNDCG@5: {imp['delta_ndcg@5']:+.4f} ({imp['base_ndcg@5']:.4f} -> {imp['curr_ndcg@5']:.4f})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Pre-LLM Information Retrieval Evaluation Suite"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/queries.jsonl",
        help="Path to evaluation queries JSONL file (default: evaluation/queries.jsonl)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated retrieval stages to evaluate (dense,bm25,hybrid,rerank,entity,all) (default: all)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline stages only (BM25, entity resolution, query classifier) without server dependencies",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to previous benchmark JSON run for delta/regression comparison",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation/results/latest.json",
        help="Path to save benchmark JSON results (default: evaluation/results/latest.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-query diagnostics and missing documents",
    )

    args = parser.parse_args()

    # Pre-load comparison baseline BEFORE running benchmark or overwriting out_path
    baseline_data: dict[str, Any] | None = None
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(
                f"Error: comparison baseline not found at '{args.compare}'",
                file=sys.stderr,
            )
            return 1
        try:
            baseline_data = json.loads(compare_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"Error loading comparison file '{args.compare}': {exc}",
                file=sys.stderr,
            )
            return 1

    # Determine stages
    if args.stages == "all":
        stages = ("dense", "bm25", "hybrid", "rerank", "entity")
    else:
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())

    print("=" * 60)
    print("  PRE-LLM RETRIEVAL EVALUATION SUITE")
    print("=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Mode:     {'Offline' if args.offline else 'Live (Dense+Hybrid+Rerank+Entity)'}")
    print(f"Stages:   {', '.join(stages)}")
    print(f"Output:   {args.out}")
    if baseline_data:
        print(
            f"Compare:  {args.compare} (loaded baseline with {baseline_data.get('total_cases', 0)} cases)"
        )
    print("Running benchmark...\n")

    try:
        results = run_benchmark(
            cases_path=args.dataset,
            stages=stages,
            out_path=args.out,
            offline=args.offline,
        )
    except Exception as exc:
        print(f"Error running benchmark: {exc}", file=sys.stderr)
        return 1
    summary = results.get("summary", {})
    stages_data = summary.get("stages", {})

    print(f"Total Cases Evaluated: {results.get('total_cases', 0)}")
    print(f"Total Execution Time:  {results.get('total_time_ms', 0.0):.1f} ms")
    print(f"Query Classifier Accuracy: {summary.get('classifier_accuracy', 0.0) * 100:.1f}%")
    print(f"Entity Resolution Accuracy: {summary.get('entity_accuracy', 0.0) * 100:.1f}%")
    if "rerank" in stages_data and "hybrid" in stages_data:
        print(f"Reranker Uplift (NDCG@5):  {summary.get('reranker_uplift', 0.0):+.4f}")

    gold = results.get("gold_report", {})
    if gold:
        print(
            f"Gold-Set Hygiene: {gold.get('missing_count', 0)} cases with missing ids, "
            f"{gold.get('fanout_count', 0)} fan-out cases"
        )
        for issue in gold.get("issues", []):
            if issue["kind"] == "missing":
                print(
                    f"  [!] {issue['id']} relevant_ids absent from corpus: {issue['relevant_ids']}"
                )
            else:
                print(f"  [!] {issue['id']} relevant set size {issue['relevant_size']} (fan-out)")

    print("\n--- Retrieval Stage Performance ---")
    print_stage_table(stages_data)

    print_category_table(results.get("categories", {}))

    # Regressions / Zero-recall queries
    regressions = results.get("regressions", [])
    if regressions:
        print(f"\n[!] Zero-Recall@5 Queries ({len(regressions)}):")
        for r in regressions[:10]:
            print(f"  - [{r['id']}] ({r['category']}) {r['query']}")
        if len(regressions) > 10:
            print(f"  ... and {len(regressions) - 10} more.")

    # Verbose diagnostics
    if args.verbose:
        print("\n--- Detailed Query Diagnostics ---")
        for q in results.get("queries", []):
            stg_info = ", ".join(
                f"{stg}: R@5={data['metrics']['recall@5']:.2f}, N@5={data['metrics']['ndcg@5']:.2f}"
                for stg, data in q.get("stages", {}).items()
            )
            match_sym = "✓" if q.get("classifier_match") else "✗"
            print(f"[{q['id']}] {q['query']}")
            print(
                f"  Classifier: {match_sym} {q.get('predicted_classifier')} (expected: {q.get('expected_classifier')})"
            )
            print(
                f"  Entities:   {q.get('predicted_entities')} (score: {q.get('entity_accuracy', 0.0):.2f})"
            )
            print(f"  Stages:     {stg_info}")
            print(f"  Relevant:   {q.get('relevant_ids')[:5]}")
            print()

    # Comparison against baseline if requested
    if baseline_data is not None:
        diff = compare_benchmarks(results, baseline_data)
        print_comparison_diff(diff)
    print(f"\nResults saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
