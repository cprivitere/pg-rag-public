import argparse
import json
import os
import re
import sys
from pathlib import Path

from pgrag.rag.pipeline import ask, _fit_context

GOLDEN_DIR = Path("data/golden")
TRACE_DIR = Path("data/retrieval_traces")

# Deterministic generation for reproducible golden runs: temperature 0 with a
# fixed seed makes generation best-effort repeatable (llama.cpp may still vary).
GENERATION = {"temperature": 0, "seed": 0}


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_golden(golden, trace=None):
    """Run one golden case. Returns (result, misses, context) where misses is a
    list of (variants, first_variant) tuples and context is the normalized text
    of the exact documents the LLM received."""
    result = ask(golden["question"], generation=GENERATION, trace=trace)
    answer = normalize(result["answer"])
    # Mirror the pipeline's _fit_context trim so a truncated (never-sent) doc
    # is not mislabeled GEN-side; entity dossiers are already ≤ budget, so
    # this is a no-op there.
    fitted = _fit_context(result.get("documents") or [])
    context = normalize("\n\n---\n\n".join(fitted))
    misses = []
    for variants in golden["facts"]:
        if not any(normalize(v) in answer for v in variants):
            misses.append((variants, variants[0]))
    return result, misses, context


def main():
    parser = argparse.ArgumentParser(description="Run the fact-presence golden eval.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="for each FAIL, report whether the missing fact is present in the "
             "retrieved context (GEN-side extraction gap) or absent (RET-side "
             "retrieval gap), using the exact context the LLM received.",
    )
    args = parser.parse_args()

    total_miss = 0
    capture_trace = os.environ.get("PGRAG_TRACE") == "1"
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        golden = json.loads(path.read_text(encoding="utf-8"))
        trace = {} if capture_trace else None
        result, misses, context = check_golden(golden, trace=trace)
        if capture_trace:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                trace_path = TRACE_DIR / f"{golden['id']}.json"
                trace_path.write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"    (trace write failed: {exc})")
        status = "PASS" if not misses else "FAIL"
        print("[%s] %s (%s, %s)" % (
            status,
            golden["id"],
            golden["type"],
            result["query_type"],
        ))
        for variants, first in misses:
            if args.diagnose:
                in_ctx = any(normalize(v) in context for v in variants)
                tag = (
                    "GEN-SIDE (fact in context but not extracted)"
                    if in_ctx else
                    "RET-SIDE (fact not in retrieved context)"
                )
                print("    MISSING: %s  [%s]" % (first, tag))
            else:
                print("    MISSING: %s" % first)
        total_miss += len(misses)

    print()
    if total_miss:
        print("FAIL: %d fact(s) missing" % total_miss)
        sys.exit(1)
    print("OK: all golden facts present")


if __name__ == "__main__":
    main()