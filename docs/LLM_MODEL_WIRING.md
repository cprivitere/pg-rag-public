# LLM Model Wiring Reference

Source of truth for how to launch each LLM for the pg-rag RAG pipeline. The
`LLM_MODEL` / `LLM_FLAGS` pair below is pasted into `mise.toml [env]` and
consumed by `.mise/tasks/llm-start.ps1` (`llama-server -hf {{ env.LLM_MODEL }}
{{ env.LLM_FLAGS }} --host 0.0.0.0 --port 8080`). `mise drift` fails if any
consumer hardcodes a model literal.

A `mise.toml` entry always sets the **current** model; swap by editing the two
vars there, then `mise llm-stop` → `mise llm-start` (stop tree-kills, verified
to actually free :8080).

Golden = `mise golden` (fact-presence; lower facts-missing is better) at
`CONTEXT_BUDGET=80000` (~19k tokens fed). Baselines:
3B=34 missing · 8B granite=20 · Qwen3.5-9B-MTP=8.
These are noisy single-draw reasoning-ON scores; deterministic reasoning-OFF
re-baseline (gemma 8, qwen 12) is in "Card flags & determinism" below.

## Common flags (all models)

- `-ngl 999 -fa on` — full offload, flash attention.
- `-c 32768` (32k) on the 12B winner — real fed context is 5-19k tokens; with
  reasoning@1024 + ≤8k answer the worst case ≈ 28.5k < 32k, so 32k is safe and
  halves the 64k KV. Only go back to 64k if a model idles at a larger
  reasoning budget (gemma-4 @4096 would need it).
- `-ctk q8_0 -ctv q8_0` — Q8 KV. **Never Q4 on the V axis** (fact-extraction
  axis golden measures). Keep unless VRAM forbids.
- `--no-mmproj` (double-dash; build rejects `-no-mmproj`) — drop the vision
  projector for text-only RAG (~0.16 GiB). Add to any model bundled with an
  `mmproj` whose RAG path never sends images.

## Ornith-1.5-9B (trial 2026-08-28)

Dense ~9B reasoning model (built on Qwen3.5 + Gemma4; Qwen chat
template adjusted — GGUF's embedded template renders correctly, NO `--jinja`)
with a vision projector bundled in repo (dropped via `--no-mmproj`). NO MTP`
head (dense — the qwen35-9B-MTP's spec-decode does NOT apply); native thinking
via `--reasoning-budget 4096` (qwen-style economical — verified content clean,
unlike gemma-4's verbose thinking).

```toml
LLM_MODEL = "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M"
LLM_FLAGS = "--no-mmproj -ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 --reasoning-budget 4096"
```

- Weights 5.78 GiB (Q4_K_M). KV Q8 @64k; measured **6,580 MiB** (≈6.6 GB —
  lightest of all contenders).). 64k over the winner's 32k because reasoning@4096 +
  <=19k ctx + <=8k ans ≈ <=31.5k,and the densest-content edge (~2.4 chars/
  token) can approach 32k — 64k keeps the headroom and stays the lightest).
- Golden (current 38-case set,: **38/38 PASS —0 facts missing** — first clean
  sweep, INCLUDING the shared-hard cases (`recipes_using_animal_feces` (the
  qwen 0/5 stalemate, `dungcrafting`, `gardening`, `field-mushroom-locations`,
  `pig_poop`). Single noisy reasoning-ON draw; presence-only fact check
  (verbosity gamesthe metric; hallucination unp penalized.. Deterministic reasoning-
  OFF re-baseline + a same-set gemma-12B control are both stillpending.



## Card flags & determinism (empirically validated 2026-08)

Unsloth's llama-server launch guide recommends `--chat-template-kwargs
{"enable_thinking":…}` + card sampling `--temp 1.0 --top-p 0.95 --top-k 64`.
Validated against this llama.cpp (10549):

- **`enable_thinking` is deprecated in our build** — the server logs
  `Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use
  --reasoning on / --reasoning off instead`. Both still work; the modern,
  non-deprecated synonym is `--reasoning on/off` (same semantics). Reasoning
  defaults **ON** even with no flag (matches the card's intent), so production
  `LLM_FLAGS` needs no explicit `--reasoning` — the default already IS
  reasoning-on.
- **Card sampling is non-reproducible.** `--temp 1.0 --top-p 0.95 --top-k 64`
  diverges run-to-run on any real multi-clause answer even with a fixed
  `seed` — sampling diversity defeats the seed in this build. It is for open
  chat, NOT a benchmark. Golden therefore uses greedy **temp 0 + fixed
  seed** (already in `scripts/golden_check.py`: `GENERATION =
  {"temperature": 0, "seed": 0}` since inception).
- **Determinism rule of thumb**: reasoning ON + greedy is NOT reproducible
  (the `<|think|>` trajectory isn't greedy-pinned — observed 7/8/10 single-draw
  golden variance was LLM-side, not a harness miss). reasoning OFF
  (`--reasoning off`) + greedy + seed IS byte-reproducible for gemma AND qwen.

### Deterministic re-baseline (identical conditions: reasoning OFF, temp 0,
seed 0)

| model | facts missing | FAIL cases |
|---|---|---|
| **gemma-4-12B-it-qat** | **8** | dungcrafting, gardening, grow-field-mushrooms, recipes_using_animal_feces |
| **Qwen3.5-9B-MTP** | **12** | + field-mushroom-locations, pig_poop |

Former noisy "7 vs 8" was a single lucky-draw for qwen; deterministically gemma
leads by a real 4-fact margin. Production reasoning-ON spot-check on the FAIL
set: gemma recovers `recipes_using_animal_feces` **5/5** (all ingredients +
level); qwen **0/5** even with reasoning. Only `gardening/spade assault`
persists for both (retrieval-coverage gap, reasoning cannot fix). Winner
confirmed: **gemma-4-12B-it-qat**.

## Qwen3.5-9B-MTP

Mamba2-hybrid, 4 KV heads → tiny cache; MTP head bundled in the main GGUF
(`nextn_predict_layers=1`), so `--spec-type draft-mtp` works single-file.

```toml
LLM_MODEL = "unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL"
LLM_FLAGS = "--spec-type draft-mtp --spec-draft-n-max 6 -ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 --reasoning-budget 4096"
```

- Native **thinking** (`--reasoning-budget 4096`), MTP spec-decode n6.
- VRAM: **8187 MiB @32k** (lighter than the 8B granite). KV @32k Q8 = 1.03 GiB.
- Golden: **31 PASS / 3 FAIL, 8 facts missing**.

## Granite-4.1-3B / 8B (superseded)

```toml
# 8B (UD-Q4_K_XL):
LLM_MODEL = "unsloth/granite-4.1-8b-GGUF:UD-Q4_K_XL"
LLM_FLAGS = "--jinja -ngl 999 -fa on -c 32768 -ctk q8_0 -ctv q8_0"
# 3B (Q8_0):
LLM_MODEL = "unsloth/granite-4.1-3b-GGUF:Q8_0"
LLM_FLAGS = "--jinja -ngl 999 -fa on -c 32768 -ctk q8_0 -ctv q8_0"
```

- `--jinja` **required** (unsloth card): loads the GGUF's embedded chat-template
  fix. **No** thinking mode and **no** MTP head → no `--reasoning-budget` /
  `--spec-type`.
- KV: 8B = 40 layers × 8 KV heads × 128 = 2.6 GiB @32k.
- Golden: 3B = 17/17, **34 missing** · 8B = 25/9, **20 missing**.

## Gemma-4 family

Native **thinking** (enable via `--reasoning-budget`); hybrid sliding-window
attention (tiny KV). No `--jinja` needed (built-in template). MTP head ships as
a **separate** file auto-discovered by recent llama.cpp from `-hf` — do **not**
pass `--model-draft` (unsloth card). Context 256K.

### gemma-4-12B-it-qat (48L Dense, QAT-lossless UD-Q4)

```toml
LLM_MODEL = "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
LLM_FLAGS = "--spec-type draft-mtp --spec-draft-n-max 4 --no-mmproj -ngl 999 -fa on -c 32768 -ctk q8_0 -ctv q8_0 --reasoning-budget 1024"
```

- MTP n4 (draft `mtp-gemma-4-12B-it.gguf` auto-discovered), thinking budget 1024
  (budget A/B/C 2026-08, reasoning-ON, 6 cases × 2 runs: **1024** → 0/12 empty +
  34/35 facts; **4096** → 0/12 empty + 33/35; **default -1 unrestricted** → 1/12
  empty + 33/35. Empty-content is a reasoning-ON failure that surfaces at the
  unrestricted default (-1), not at 4096; 4096 gives no factual gain — it and -1
  both lose `mycology level of 15` that 1024 keeps. 1024 stays).
- **Shaved for VRAM (text-only RAG):** `--no-mmproj` drops the 0.16 GiB vision
  projector (image input only — RAG never sends images; must be **double-dash**,
  this llama.cpp build rejects `-no-mmproj`); `-c 32768` halves the KV (safe:
  `_fit_context` caps context at CONTEXT_BUDGET (80000 chars) so expansion
  can't overflow the window; extraction-first (80k golden-clean, see note).
  Measured **8,019 MiB** (was 8,741) — ~0.7 GiB saved, zero quality cost.
- Weights 6.26 GiB.
- **No smaller weight quant exists** — unsloth ships only the QAT-lossless
  UD-Q4_K_XL for this model; the 6.26 GiB floor is immovable (below Q4 breaks
  QAT-losslessness — the Q2 collapse measured on 27B). Remaining trim = drop
  MTP (−0.24 GiB, loses spec-decode generation speed).
- Golden: **31 PASS / 3 FAIL, 7 facts missing** (≈ qwen 9B's 8; deterministic
  re-baseline (reasoning off, temp 0 + seed): 8 missing vs qwen 12; production
  reasoning-on spot-check recovers the shared hard case
  `recipes_using_animal_feces` 5/5 (qwen 0/5) — see "Card flags & determinism".

### gemma-4-26B-A4B-it-qat (30L MoE, 3.8B active, QAT-lossless UD-Q4)

```toml
LLM_MODEL = "unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL"
LLM_FLAGS = "--spec-type draft-mtp --spec-draft-n-max 4 -ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 --reasoning-budget 4096"
```

- MoE runs ~like a 4B for speed (0.9s/trivial case). Weights 13.27 GiB, measured
  **17,224 MiB** — needs Project Gorgon **closed** (22.6 GiB free before load,
  ~5.4 left).
- Golden: **27 PASS / 7 FAIL, 12 facts missing** — WORSE than the 12B (7) and
  qwen (8). Sparse MoE routing + terse output (142 tok vs 12B's 1031) drops
  exhaustive facts. Dense 12B beats the MoE 26B on this extraction task.

### gemma-4-26B-A4B-it (non-QAT, UD-Q2_K_XL)

```toml
LLM_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q2_K_XL"
LLM_FLAGS = "-ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 --reasoning-budget 4096"
```

## Qwen3.8-27B (UD-Q2_K_XL) — ruled out

```toml
LLM_MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-Q2_K_XL"
LLM_FLAGS = "--spec-type draft-mtp --spec-draft-n-max 6 -ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 --reasoning-budget 4096"
```

- **Dense** (qwen35 Mamba2-hybrid, 65L, 4 KV heads → tiny cache). MTP head
  bundled (`nextn_predict_layers=1`) → single-file `draft-mtp`. Native thinking.
- Measured **15,339 MiB** even at Q2 (dense + 64k KV + runtime) — heavy.
- Golden: **29 PASS / 5 FAIL, 10 facts missing** — capacity at Q2 does NOT
  beat lossless-QAT density; Q2 damaged extraction (dropped "tavilak",
  gazluk "5 players" that smaller models caught). Dense-12B-QAT is the sweet
  spot; big dense quants lose on BOTH VRAM and extraction here.

## Final leaderboard (facts missing, lower = better)

| rank | model | missing | FAIL | VRAM |
|---|---|---|---|---|
| 1 | **gemma-4-12B-it-qat** | **7** | 3 | 8.0 |
| 2 | Qwen3.5-9B-MTP | 8 | 3 | 8.2 |
| 2 | gemma-26B-A4B Q2 | 8 | 4 | — |
| 4 | Qwen3.8-27B Q2 | 10 | 5 | 15.3 |
| 5 | gemma-26B-A4B-it-qat | 12 | 7 | 17.2 |
| 6 | granite-4.1-8b | 20 | 9 | 8.4 |
| 7 | granite-4.1-3b | 34 | 17 | — |
*Leaderboard `missing` = noisy single-draw reasoning-ON scores (gemma 7, qwen 8
are within one draw's variance). Deterministic reasoning-OFF re-baseline
(temp 0 + seed) re-measures the top two: **gemma 8, qwen 12** — same ordering,
a real 4-fact margin — see "Card flags & determinism". Rows 3–7 were not
deterministically re-measured.*


Winner: **Ornith-1.5-9B** — current production selection (LLM_MODEL/LLM_FLAGS shipped in mise.toml;wiring + flags documented in its section above(. Its record golden run reported 0-missing on  38 cases (~6.6 GiB),sthe cleanest on file — determinism/same-set control re-baseline was still pending as of this doc's note.

The prior winner,**gemma-4-12B-it-qat**, stays the documented dense+QAT benchmark (7-facts-missing on the older  34-case set();its leaderboard row above is historical,superseded only in the production slot。 lhe swap decision lives in mise.toml [env] — treat that file as the source of truth for what ships。
