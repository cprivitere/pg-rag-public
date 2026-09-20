---
name: retrieval
description: Search-quality troubleshooting manual for pg-rag. Use when investigating why a query returns wrong/missing/refusing results, or before changing any retrieval stage.
---

# retrieval — search troubleshooting

## The pipeline (in order)

```
query understanding (spelling → query_classifier)
  → entity detection (leveling-intent → skill entity)
  → document generation (is the fact even in the corpus?)
  → dense retrieval (Chroma :8081 embeddings)
  → BM25 retrieval (lexical arm; re-parses documents.json per hybrid query)
  → RRF fusion (_hybrid_fuse, retriever.py)
  → reranker (:8082 cross-encoder; lexical fallback if down)
  → answer generation (LLM :8080 prompts.py)
```

## Debugging discipline

**First localize the failure. Never modify multiple stages at once.**

1. **Query understanding** — what does `query_classifier` say? Is spelling
   corrected? `tests/test_query_classifier*.py`, `tests/test_spelling.py`.
2. **Entity detection** — did `find_entity` / `build_entity_context` resolve a
   hub? Wrong-entity answers point here. `tests/test_entity_detection.py`,
   `tests/test_entity_retrieval*.py`.
3. **Document generation** — is the fact present in `data/documents.json` at
   all? `grep` the JSON for the expected name/level. If absent, the problem is
   upstream (loader/builder), not retrieval.
4. **Dense vs BM25 vs fusion** — run `scripts/retrieval.py` and inspect the
   per-stage lists: does dense find it? does BM25? does RRF surface it?
   `tests/test_retrieval_unit.py`, `tests/test_bm25.py`.
5. **Reranker** — is `rerank_used` true? Reranker drops/buries the doc?
   `tests/test_rerank*.py`, `tests/test_rerank_fallback.py`; check
   `logs/rerank.log` and `data/rerank_stats.json`.
6. **Answer generation** — context is right but the LLM refuses? Check
   `_gap_fill` fired (MISSING_REGEX), and the refusal wording in
   `prompts.py`. `tests/test_pipeline_*.py`, `tests/test_prompts.py`, golden set.

## Rule of thumb

- One-stage-at-a-time changes. Re-run retrieval regression tests after each.
- Wiki "how-to" questions need page expansion: `_prepare_general` pulls
  sibling chunks of a wiki page via `parent_id` (bounded, id-deduped) — if a
  fact sits in another chunk of the same page, that expansion is the fix, not
  a new retrieval system.
- Hybrid is enabled for `general` queries only; `comparison` boosts count to
  20; entity queries build a dossier instead.

## Evaluation loop (see `evaluation` skill)

For every retrieval change: create representative queries → record expected
facts → compare before/after top-k → inspect ranking changes → only then decide.
`data/golden/*.json` + `scripts/golden_check.py` is the automated form; the
`evaluation` skill documents the harness.