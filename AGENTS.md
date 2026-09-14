# Repository Guidelines

Oh My Pi agent guide for **pg-rag-builder** — a RAG pipeline and retrieval harness for the *Project Gorgon* game wiki. Vectorizes CDN game data + wiki text into Chroma, retrieves with hybrid BM25+dense fusion, and answers questions through a local LLM.

---

## Project Overview

Build a searchable, fact-grounded knowledge base from two sources:
- **CDN**: structured `data/cdn/*.json` tables (items, recipes, abilities, skills, quests, npcs, areas, effects).
- **Wiki**: raw `data/wiki/*.txt` page dumps (hashed filenames, titles via `data/wiki/.meta.json`).

Generate typed documents → embed → index in Chroma → retrieve (dense + BM25 → RRF fuse → rerank) → answer via LLM. Ships a CLI (`pgrag`), OpenWebUI pipe, Gradio chat, curation tooling, and golden/embed eval suites.

## Architecture & Data Flow

```
cdn/*.json ─┐
            ├─ loaders → GameDatabase(tables + wiki) ─┐
wiki/*.txt ─┘                                          ├─ documents/ (builder + wiki_builder + skill_profiles + summaries)
data/wiki/curated/*.json ──────────────────────────────┘
data/il2cpp/out_lean/Dump0/ ───────────────────────────┘  (decomp: enums/schema/mechanics, refreshed by mise sync-il2cpp)
                        │  build.py: generate_documents() → documents.json (+ documents_version.json, stamps DOCUMENTS_VERSION)
                        ▼
              build_index.py → Chroma collection "project_gorgon" (incremental, hash-based)
                        │  EMBED_BATCH_SIZE=10000, validates EMBEDDING_DIM=384
                        ▼
Query → query_classifier → retriever (dense + BM25 → RRF fuse → reranker :8082)
                        │
                        ├─ entity queries → entity_retrieval (skill dossiers sorted by required level) + gap-fill
                        └─ pipeline.ask / ask_stream → prompt → LLM :8080 → answer
```

- **Freshness contract (avoid stale-document trap)**: `build-documents` stamps `data/derived/documents_version.json` with `DOCUMENTS_VERSION` (config.py; authority is the constant, not this doc). `build-index` only reads the persisted `documents.json` and refuses to embed if the stored version differs — it never regenerates. To converge a source in one command, use `mise sync-*` tasks (they run `build-documents` first). Bump `DOCUMENTS_VERSION` whenever document shape changes.

## Key Directories

- `src/pgrag/` — importable package (`uv` installs editable).
  - `cli.py` — CLI: `download-cdn`, `download-wiki`, `build-documents`, `build-index`, `validate`.
  - `config.py` — paths, `EMBEDDING_DIM=384`, `CONTEXT_BUDGET=80000`, `DOCUMENTS_VERSION`, `TARGET_CATEGORIES`/`RECURSIVE_CATEGORIES`.
  - `build.py` — `generate_documents()` orchestration.
  - `loaders/` — `cdn_loader` (tables from CDN json), `wiki_loader` (wiki text + `.meta.json` title mapping, orphan cleanup), `database.GameDatabase` (in-memory `tables` + `wiki` bag).
  - `documents/` — `builder.py`, `wiki_builder.py` (mwparserfromhell → sections/chunks, `parent_id` links), `chunking.py`, `resolver.py` (internal code → display name), `skill_profiles.py`, `summaries.py` (gathering skill maps).
  - `embeddings/llama_embeddings.py` → :8081.
  - `vectorstore/` — `build_index.py`, `hashes.py`, `health_check.py`.
  - `rag/` — `retriever.py`, `reranker_client.py`, `bm25.py`, `query_classifier.py`, `query_plan.py`, `spelling.py`, `entity_retrieval.py`, `resolve.py`, `synthesis_detector.py`+`synthesis_generator.py`, `pipeline.py` (+ `ask_stream`), `prompts.py`, `llm.py`.
- `scripts/` — eval + service tooling (see Important Files).
- `tests/` — pytest suite, imports the installed `pgrag` package.
- `notebooks/` — training notebooks NOT part of the RAG pipeline: `gen_synthetic.training.py` (distill/generation notebook) + `molab-mirror/` (snapshot of the molab training sandbox: bf16-final teacher config, pipeline scripts, decision trail incl. fp8 retirement), with `notebooks/molab-mirror/README.md` as the restore path.
- `data/` (gitignored) — `cdn/`, `wiki/` (+`curated/`, `.meta.json`), `derived/` (`documents_version.json`), `documents.json`, `chroma/`, `golden/`, `retrieval_traces/`, eval records (`embed_eval_*.log`, `embed_vram.json`, `bakeoff_*.json`), `il2cpp/` (decomp artifacts: client binaries `GameAssembly.dll` + `global-metadata.dat` quoted verbatim from the Steam client, dumper output `out_lean/Dump0/{dump.cs,stringliteral.json}` the document builder reads, survey tools + the cargo-built dumper under `tools/il2cpp-dumper-rs` — refreshed by `mise sync-il2cpp`). Service logs live at project-root `logs/` (`embed.log`, `llm.log`, `rerank.log`, `chat.log`, and `webui.log` when run).
- `.omp/` — oh-my-pi config: `RULES.md`, `config.yml`, `WATCHDOG.md`, `skills/` (`pg-rag`, `pg-data`, `retrieval`, `evaluation`, `testing`).

## Development Commands

Python ≥3.14, managed with `uv`. `mise tasks` lists everything (`uv run pgrag …`).

```sh
uv run pgrag download-wiki          # fetch wiki dumps
uv run pgrag download-cdn           # fetch CDN json
uv run pgrag build-documents        # regenerate documents.json (stamps version)
uv run pgrag build-index            # embed + index into Chroma
uv run pgrag validate              # full offline pipeline integrity check (sources, documents+freshness, wiki meta, index)
uv run pgrag build-index --source cdn|wiki|computed|curated|il2cpp   # partial rebuild of one source
mise sync-wiki / sync-cdn / sync   # build-documents + build-index in one shot (aliases syw/syc/sy)
mise sync-il2cpp                   # IL2CPP decomp: stage fresh Steam-client binaries + re-run the dumper + partial re-embed (alias syi)
mise generate-docs                 # bare idempotent documents rebuild (alias docs)
mise golden                        # golden eval (needs :8080 + :8081)
mise golden-short                  # quick tier (~3-5 min; alias gds)
mise golden-one -- fireball-ability   # rerun named golden case(s) fast (alias go; comma-separate ids)
mise golden-flaky / golden-flaky-list # focus on historically-troublesome golden cases / audit stats
mise chat                          # Gradio chat (primary UI; also started by `mise start` = start-all; needs embed + LLM up)
mise lint                        # ruff check src scripts tests (alias li) — run after editing code
mise fmt                         # ruff format + safe autofix (alias fo)
uv run pytest                      # offline test suite
uv run pytest tests/test_retrieval_unit.py tests/test_bm25.py tests/test_rerank*.py  # retrieval regression
mise drift                        # check docs/skills against the repo (aliases: dr)
```

**Build/refresh needs no servers**; only Q&A/eval (`golden`, `chat`, `scripts/retrieval.py`) do.

## Code Conventions & Common Patterns

- **Python/uv**: py≥3.14, type-annotated, package-relative imports (`from pgrag.config import …`). Edit the package in `src/pgrag/`, never hand-edit build artifacts (`documents.json`, `data/derived/wiki_parsed.json`, `bm25_index.pkl`).
- **Doc identity contract**: every doc carries `id` + `metadata.source` + `metadata.table`. Chroma `update()` merges metadata (add-only) — renames/removals require a full rebuild. Preserve these fields.
- **Retrieval architecture is fixed**: dense + BM25 → RRF fusion (`_hybrid_fuse`, `HYBRID_MULTIPLIER`) → reranker (`bge-reranker-v2-m3` :8082, lexical fallback if down). Don't replace a component without identifying the existing implementation first; verify with the retrieval regression tests.
- **Incremental index**: `build_index.py` computes embedding hashes to avoid re-embedding unchanged docs, batches at `EMBED_BATCH_SIZE=10000`, validates dims against `EMBEDDING_DIM`. **Never change embedding models silently** — embeddings are a fixed-dim contract with the Chroma collection.
- **Error handling**: server clients raise domain errors (e.g. `EmbeddingServerError`, LLM/rerank errors) with the URL in the message; offline tests assert these. Server reachability is checked but servers down → graceful lexical/fallback paths.
- **Wiki categories**: `TARGET_CATEGORIES` flat + `RECURSIVE_CATEGORIES` (Creatures d2, Items d1) — monsters/items only via recursion. Subcats bare (no `Category:` prefix). Wiki filenames `{safe_title}_<sha256-8>.txt`; display names come only from `.meta.json` (never filenames). Sync ends with `remove_orphan_files`.
- **Test isolation**: temp dirs for anything touching `data/` (real meta/documents are guarded — a past bug silently destroyed `data/wiki/.meta.json`). `tests/conftest.py` snapshots `data/cdn`/`data/wiki` and asserts immutability.

## Important Files

- `src/pgrag/cli.py` — entry point; `config.py` — constants/paths; `build.py` — document orchestration; `rag/pipeline.py` — query path (deterministic temp=0/seed=0).
- `scripts/pg_rag.py` — OpenWebUI pipe, `PG_ROOT = os.environ.get("PG_RAG_ROOT", r"F:\ProjectGorgon\pg-rag-builder")` (env override, Windows default) + `os.chdir()`, adds `PG_ROOT/src` to `sys.path` — the default path is what moves if the repo relocates. Valves: `TOP_K=40`, `USE_HYBRID=True`, `USE_RERANK=True`.
- `scripts/curator.py` + `curator_scheduler.py` — heuristic (non-LLM) curation: regex-detect fragmented knowledge (area_levels, skill_trainers, crafting_progressions), write template docs to `data/wiki/curated/`, scheduler persists state to `data/curator_state.json` and rebuilds doc/index on change. Deterministic by design — no LLM, so curated docs are stable anchors.
- `scripts/golden_check.py` — fact-presence golden eval → `data/golden/`; `scripts/golden_rerun.py` — quick named-case rerun + flaky focus (`mise golden-one`/`golden-flaky`; appends miss history to `data/golden/history.jsonl`); `scripts/embed_eval.py` (+`bakeoff_corpus.py`; VRAM helpers in `embed_vram_probe.py`) — embedding bake-offs.
- `docs/TEST_CONTRACTS.md` — layer→tests→contract map + regression-triage protocol (read before changing behavior/tests); `docs/REVIEW.md` — audit findings + improvement backlog.
- `scripts/check_services.py` — [OK]/[DOWN] probes for all services.
- `mise.toml` `[env]`: `WEBUI_DIR`, `LOGS_DIR` — update if paths move.

## Runtime/Tooling Preferences

- **Runtime**: Python ≥3.14 via `uv`; model binaries fetched with `hf` global CLI (cache `F:\AI\models\hub\`; GGUF via `-hf org/repo:quant`).
- **Local services** (running on Windows host):
  | svc | port | model / note |
  |-----|------|--------------|
  | Embeddings | 8081 | `EMBED_MODEL` (`[env]`) — bge-small f16, cls pooling, hard 512-token server cap (input chars clipped to `llama_embeddings.MAX_EMBED_CHARS` = 2000); needed for Q&A/eval |
  | LLM | 8080 | `LLM_MODEL` (`[env]`; `LLM_FLAGS` carries launch tuning, e.g. `--jinja`) — RAG Q&A; a running instance OOMs before start → `mise down` (alias of `stop-all`) first |
  | Reranker | 8082 | `RERANK_MODEL` (`[env]`) — bge-reranker cross-encoder; optional, lexical fallback |
  | Chat (Gradio) | 7860 | primary UI — part of `mise start` (alias of `start-all`); `mise chat` foreground; history in browser localStorage |
  | OpenWebUI | 3000 | optional (legacy) — `mise webui-start`/`webui-stop`; no longer in `start-all`/`stop-all` |
- **Single-source models**: refs + tuning flags live once in `mise.toml [env]` (`*_MODEL`/`*_FLAGS`); consumed by `.mise/tasks/*-start.ps1` (`$env:`), `mise debug-*` (`{{ env.* }}`), `scripts/vram_sweep.py` and `scripts/embed_eval.py` (tomllib). `mise drift` fails if any consumer hardcodes a model literal.
- `scripts/rag_chat.py` and `pg_rag.py` assume these services up.
- Tests import the installed `pgrag` package — after changing `src/pgrag/`, no reinstall needed (editable install).

## Testing & QA

- **Framework**: pytest via `uv run pytest`; 597 tests pass, 42 skipped (655 collected — skips are offline server-guards: golden tests), 16 slow deselected. All offline; temp-dir integration.
- **Golden eval** (`tests/test_golden_check.py`): parametrized over `data/golden/*.json`; `require_servers` fixture skips unless LLM :8080 + embed :8081 are up; retries once (2 attempts) to damp LLM nondeterminism; both attempts fail = regression. `GENERATION = {"temperature":0,"seed":0}` is wired into every `ask()`; observed 7/8/10 single-run variance was **LLM-side** (reasoning-ON thinking trajectory is not greedy-pinned even at temp 0 + seed), not a harness miss. For a byte-reproducible comparison, launch the server with `--reasoning off` (gemma AND qwen proven reproducible); reasoning-ON production runs remain single-sample noisy. See `docs/LLM_MODEL_WIRING.md` → "Card flags & determinism".
- **Retrieval regression set** (run when changing retrieval): `tests/test_bm25.py`, `tests/test_retrieval_unit.py`, `tests/test_rerank*.py`, `tests/test_retriever_spelling.py`. **`docs/TEST_CONTRACTS.md` is the authoritative layer→tests→contract map** — read L3 before any retrieval change: it flags shared-function facet coverage (`retrieve()` asserted across 5 files; `_hybrid_fuse` across `test_bm25.py`/`test_rerank.py`) and notes the stale-`DOCUMENTS_VERSION` refusal is directly tested in `test_build_index.py`.
- **Test edits are contract changes.** Editing what a test *asserts* is not a workaround — state the new contract in the test, run the layer's sibling suite (`docs/TEST_CONTRACTS.md`), and never hand-edit a build artifact (`documents.json`, `bm25_index.pkl`, `wiki_parsed.json`) to satisfy a test. If the behavior didn't legitimately change, the failure is a source regression: fix the source, not the test.
- **Lint gate** (`tests/test_lint.py`, layer L9 in `docs/TEST_CONTRACTS.md`): `uv run ruff check src scripts tests` must be clean — a failing check fails `uv run pytest`. Run `mise lint` after editing code (`mise fmt` applies the safe autofixes: import sort, unused imports, f-strings, formatting). Rules live solely in `ruff.toml`; the per-file-ignores there (scripts: broad excepts, unchecked subprocess; tests: broad catches) are deliberate. New rule ⇒ fix all findings in the same commit.
- **Key suites** (from `tests/`): `test_documents.py` (doc shape), `test_chunking.py`, `test_health_check.py` (index integrity incl. no-SQLite-crash on large collections), `test_llm.py` (SSE streaming parse), `test_download_wiki.py` (batching, redirects, orphan cleanup — patches `META_FILE`+`WIKI_DIR` to tmp), `test_query_classifier.py`/`test_query_plan.py`, `test_bm25_persist.py`, `test_rerank_fallback.py`, `test_hashes.py`, `test_embed_validation.py`.
- **Coverage expectation**: one test defends each observable contract; integration tests use temp dirs and mocked servers rather than live ones.