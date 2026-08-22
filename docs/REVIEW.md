# REVIEW — pipeline, tests, and documentation audit (verified v2)

Date: 2026-08-21 · Revised after line-level verification (see "Correction").

Companion artifact: `docs/TEST_CONTRACTS.md` (contract map + triage protocol).
Deliverable that drove this pass: the regression-triage protocol and
shared-function facet map in `docs/TEST_CONTRACTS.md`, plus `AGENTS.md`,
`.omp/RULES.md`, `.omp/WATCHDOG.md` updates.

## Correction (v1 → v2) — reading beats summarizing

The first version of this review shipped two headline findings sourced from
agent summaries that were not verified against the test files:

1. "`test_bm25.py` and `test_retrieval_unit.py` both own five identical BM25
   tests" — **false.** A full raw read shows the five BM25 tests live only in
   `test_bm25.py`; `test_retrieval_unit.py` contains `retrieve()` filter tests
   and `ask()` citation tests, no BM25 duplicates. An exact-name scan of all
   `def test_*` across the 54 files finds **exactly one** name collision in
   the whole suite (`test_summary_shape`, in `test_summaries.py` and
   `test_gathering_summaries.py`).
2. "The `DOCUMENTS_VERSION` refusal has no direct test" — **false.**
   `test_build_index.py::test_documents_version_refuses_stale` and
   `::test_documents_version_refuses_missing_marker` directly assert
   `load_documents()` raises `ValueError` on stale/missing markers.

Both errors came in because the finding matched the expected narrative (the
user reported the model makes regression errors → a review "expects" to find
duplication and uncovered guardrails → the summaries were accepted
uncritically). That is the same failure mode the engagement exists to stop,
and it validated the core recommendation before any code was touched: **facts
about this repo must be read from source lines, not inferred from summaries
that fit a prior.** No test was deleted or altered on a false premise, and the
brittle "assert DOCUMENTS_VERSION == 4" test proposed in v1 was dropped (it
would break on the next legitimate version bump).

## Findings (verified)

### Tests — genuinely clean; the redundancy concern was over-stated

The suite is better than the original audit claimed:
- **No duplicate BM25 tests, and the freshness guard is tested.** Both
  P0 candidates from v1 dissolve on verification.
- **One true name collision**: `test_summary_shape` appears in
  `test_summaries.py` (asserts `build_summary_documents` shape) and
  `test_gathering_summaries.py` (asserts `build_gathering_summaries` shape).
  Same contract name, two different builders. Fixed in this pass by renaming
  to `test_cdn_summary_shape` / `test_gathering_summary_shape` and adding
  per-file module docstrings naming each file's builder.
- **Real structural pattern: shared-function facet coverage.** One function is
  asserted across several files, each a distinct facet — this is normal and
  healthy layered coverage, but it is the load-bearing discoverability gap:
  - `retrieve()` — `test_bm25.py` (hybrid routing, comparison/general
    `n_results`), `test_retrieval_unit.py` (where-filters, default, citation),
    `test_where_filter.py` (operator + token post-fusion), `test_retrieval_trace.py`
    (trace fields), `test_retrieval_eval.py` (query-type pass-through).
  - `_hybrid_fuse` — `test_bm25.py` (RRF intersection, multiplier, bm25-only
    doc), `test_rerank.py` (tsys chunk/base caps, origin forms).
  A model editing `retrieve()` or `_hybrid_fuse` cannot learn the full
  constraint set from one file; nothing previously said so. This is now
  documented in `docs/TEST_CONTRACTS.md` L3.
- **Isolation and coverage remain strong**: session immutability guard
  (`conftest.py`), tmp-dir patches for wiki/derived paths, mocked servers,
  keys-only shape asserts, deterministic data. The rerank trio
  (`test_rerank.py` / `_client` / `_fallback`), L6 eval, and L7 loader suites
  are each non-redundant internally.

### Documentation & comments

- `AGENTS.md`, `.omp/RULES.md`, `.omp/WATCHDOG.md`, four skills: substantive,
  consistent, and unusual for the repo type. The advisor (second model) is
  enabled in `.omp/config.yml`.
- Real gap this pass closed: watchlist/rule text referenced unverified
  specifics ("duplicate BM25 tests", "unguarded version refusal"). Those are
  corrected to the verified `retrieve()`/`_hybrid_fuse` facet map and the
  tested version-refusal path.
- **`scripts/` is operational documentation** (verified line-level this pass):
  `golden_check.py` parses `{id, question, type, facts}` with
  any-variant-substring pass and `evaluation/queries.jsonl` fields exactly
  match their documented contracts; `rag_chat.py` (port 7860, embed:8081 +
  LLM:8080, one-shot) matches. Drift fixed: `pg_rag.py`'s `PG_ROOT` is
  `PG_RAG_ROOT` env-or-default, not "hardcoded" (AGENTS.md corrected).
  Completeness debt: `scripts/curator.py::create_curated_from_fragments` is
  self-labeled "simplified version - in production, would use LLM" — a stub
  generator shipping as curation; AGENTS.md does not disclose it (backlog).
- Comment quality on core symbols is mixed: `retriever.py`/`resolve.py` have
  strong inline WHY comments, but several load-bearing functions
  (`retriever._hybrid_fuse`, `retriever._rerank`, `build.py.generate_documents`,
  `build_index.build_index`, `spelling.correct_query`) lack docstrings, so a
  fresh model must grep to find the rationale. No TODO/FIXME debt markers exist
  anywhere in `src/`/`scripts/`.

## Backlog (re-scoped to verified reality)

### P0 — already landed this pass
1. **Corrected the review's false findings** in `docs/TEST_CONTRACTS.md`,
   `docs/REVIEW.md`, `AGENTS.md`, `.omp/RULES.md`, `.omp/WATCHDOG.md` — the
   guardrails now encode the verified test layout, not a plausible-sounding one.
2. **Disambiguated the one true test-name collision** (`test_summary_shape` →
   `test_cdn_summary_shape` / `test_gathering_summary_shape`) and added
   builder-owning module docstrings to `test_summaries.py`,
   `test_gathering_summaries.py`, `test_documents.py`.
3. **Added the ground-truth rule** to `TEST_CONTRACTS.md` (verify by reading
   source lines before encoding a finding) — the meta-guardrail this
   engagement produced.

### P1 — document, don't delete (LANDED 2026-08-21)
4. **Docstring pass on core symbols** — done: added to `retriever._term_overlap`,
   `_rerank`, `_hybrid_fuse`, `retrieve`; `build.generate_documents`;
   `build_index.build_index`; all 23 `builder.py` build_* docstrings (each
   metadata.table claim grep-verified against source). `spelling.correct_query`
   / `_where_matches` / `_name_injection_ids` were *already* documented and left
   untouched.
5. **Module docstrings for the remaining test files** — done: 34 added across
   the suite (files already having one skipped), each derived from the file it
   heads. Verified by AST-parse + full suite: **547 passed, 14 deselected,
   0 failures**.

### P2 — coverage depth (LANDED 2026-08-21, premise-corrected)
6. **Golden-set expansion** — premise stale: the set is **already 34** (not 5;
   docs said 5/"expand to 20-30"). Left as-is (well-formed, passes collection);
   corrected the count in `evaluation/SKILL.md` + `pg-rag/SKILL.md`. Imbalance
   noted: comparison = 1/34 vs entity = 18/34 — future goldens should favor
   comparison. Grounding spot-checked against `documents.json` (facts present,
   no fabrication).
7. **Pipeline-set redundancy** — landed via P1.5: every `test_pipeline_*.py`
   and `test_prompts.py` now carries a docstring splitting routing
   (`test_pipeline_*`) from prompt-content (`test_prompts`), and
   `TEST_CONTRACTS.md` L5 states the seam. Rejected "factor a shared fixture"
   — refactor risk on a passing suite for no coverage gain.
8. **Curator stub** — chosen to *document, not build LLM curation*: `curator.py`
   now states its heuristic-template nature (deterministic, offline, stable
   anchors) and removes the misleading "in production, would use LLM" line;
   AGENTS.md updated. LLM curation is available but deliberately rejected —
   it would break the suite's determinism and offline-test ethos.

## Addendum (2026-08-22) — bakeoff prefix verification

The Aug 22 "prefix fairness" pass (Tier 0) set `mxbai-xsmall` to nomic's
`search_query:`/`search_document:` on the assumption its embed is
instruction-tuned. Card verification refuted that: mxbai-embed-**xsmall** is a
bare, pooled (mean) model — `Represent this sentence for searching relevant
passages: ` belongs to mxbai-embed-**large** only. Measured on the 40-query
corpus, applying nomic's two prefixes (0.7744) cost mxbai ~0.13 MRR over true
bare (0.9042). A full-list review then caught two half-applied prefixes in
`BAKEOFF_CANDIDATES`: mxbai's `doc_prefix` still carried `search_document:` —
query-bare / doc-prefixed, an intermediate state measuring 0.9250, not the
bare score — and nomic's `doc_prefix` was missing (0.9437 → restored to
0.9563 with `search_document:`).

**Fix / rule:** prefixes are per-model card facts — read them from the model's
HF card, never borrow across families, and re-audit the whole `BAKEOFF_CANDIDATES`
list when touching one (a half-applied fix is as confounded as a wrong one).
## Addendum (2026-08-22) — production embed switched to bge-small-f16

Wired the bakeoff winner into the pipeline. Live-validation before wiring
found bge-small's token ceiling is **hard at 512**: llama-server rejects
inputs >512 tokens with HTTP 400, and sweeping `-c` 512..4096 / `-b`
512..16384 / `-ub` 512..8192 changed nothing (unlike mxbai, no server flag
expands the window). The live corpus has 528 docs (0.2%) over that limit
(computed skill profiles up to 2071 tokens) — embedding them raw would have
crashed `build-index` (embed_batch only caught Connection/Timeout, not the
400). Fix: `llama_embeddings.embed_batch` clips inputs to
`MAX_EMBED_CHARS=2000` (≈400 tokens at the measured ~4.9 chars/token) with a
shrink-and-retry on residual 400s; the server runs `--pooling cls` (mxbai's
re-embed was required because `embedding_hash` is content-based (a model
switch is invisible to the incremental indexer). The index path embeds docs
bare (`build_index` → `embed_batch`); the **query** path applies bge-small's
`Represent this sentence for searching relevant passages: ` prompt via
`embed_text` — the same query_prefix/doc_prefix split the bakeoff validated
(0.9028 bare → 0.9875 with the query prompt). Tradeoff: the longest 0.2% of
docs are indexed by their first ~400 tokens.

## Verdict

The test suite is in better shape than the original review claimed: no mass
duplication, and the version-freshness refusal — the trap that keeps biting —
is already directly tested. The real, verified problems are **discoverability**
(shared functions constrained by many files, with no map) and **process**
(findings were being encoded without line-level verification). Both are fixed
in `docs/TEST_CONTRACTS.md` and the rules files. No rearchitecting needed.