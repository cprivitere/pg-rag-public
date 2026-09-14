# Plan: notebook VRAM-driven generation speedups

## Context

### Annex — NVFP4 investigation (this turn): not adoptable now; revisit only via vLLM/SGLang
Hardware confirmed live: NVIDIA RTX PRO 6000 Blackwell Server Edition, sm_120 (compute 12.0), torch 2.11.0+cu130 — the GPU *is* NVFP4-capable. The software stack on the molab sandbox cannot use it today:
- `bitsandbytes 0.50.2` (current quantization path) supports only `('fp4', 'nf4')` — no NVFP4 format; `transformers 5.16.1` has no `NvFp4Config`.
- torch 2.11 ships a native Blackwell NVFP4 primitive (`torch.nn.functional.scaled_mm` + `ScalingType.BlockWise1x16` with `float8_e4m3fn` scales — shape/dtype contract probed live), but converting the teacher's weights to packed `float4_e2m1fn_x2` needs torchao / NVIDIA Model Optimizer (neither installed; no `torch.*fp4*` quantize helper in the runtime; `.to(torch.float4_e2m1fn_x2)` cast is not supported).
- Drop-in CUDA route is a serving-runtime swap: Unsloth ships `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors, ~24 GB VRAM, up to ~2.5x faster than BF16, W4A4 on Blackwell tensor cores) — this exact teacher. That needs vLLM in its own venv (`vllm>=0.25.0`, `flashinfer-python>=0.6.13`, `nvidia-cutlass-dsl>=4.5.2`); vLLM/SGLang are NOT installed on the sandbox today. Out of scope for this plan; noted as the follow-up if generation needs another order of magnitude.
Conclusion for the plan below: stay on the HF `transformers` in-process path; the following speedups do not depend on NVFP4.

### Baseline facts
`gen_synthetic` cell in `notebooks/gen_synthetic.training.py` (live kernel `lEQa`) generates synthetic QA too slowly given the hardware: 95 GiB GPU with ~43.5 GiB free at current state (teacher 16.1 GiB bf16-equivalent + student 16.8 GiB resident). User chose: (1) explicit student-unload cell, (2) dynamic BATCH from free VRAM at runtime, (3) fp8 KV cache if the arch supports, (4) prefill-ahead pipeline. Measured this turn (scratchpad probes, cited below) to design against real numbers.

Measured (kernel scratchpad, live):
- 95.0 GiB total; 51.5 GiB in use pre-probes → 43.5 GiB free with BOTH models resident
- teacher peak-alloc during BATCH=8 / 16 / 32 / 40 generate @512tok: 34.6 / 36.0 / 38.8 / 40.1 GiB  → bigger batch is nearly free VRAM-wise
- wall-clock aggregate decode at 512 tok: B=8 4869 tok/s; B=16 11058; B=32 17174–34389 (greedy prefill still amortizes decode cost)
- `flash_attention_2` validated working (kernels-community fallback); perf on par with sdpa at these shapes → no architectural switch required, keep sdpa
- student is a `PeftModelForCausalLM` (LoRA r=16 on bf16 9B) ~16.8 GiB; safe to free entirely while generating

## Approach
Only the live marimo notebook changes (Application of changes to the running kernel via `marimo._code_mode`); then byte-persist to the repo mirror. All edits happen in cell bodies already inspected: `imports` (46-75), `teacher` (79-101), `load_student` (104-134), `gen_synthetic` (138-574). Line refs from `[notebooks/gen_synthetic.training.py#577F]` read this turn.

### A. New `memory` cell (student unload + helper)
1. After `load_student` (before `gen_synthetic`), insert a marimo cell `def memory(student_model, torch):` that (a) first read `torch.cuda.mem_get_info()` and printed BOTH current VRAM state and student parameter count; (b) `unloaded_student = None` initial. Returns `unloaded_student` so `gen_synthetic` takes one source of truth. Reuses username token/vblA-style comment style (short).
   - Rationale: keep it reactive-independent of `gen_synthetic`; user toggles unload via marimo UI later without editing the generation cell.
2. Under it, a `def unload_student(student_model, torch):` cell: `del student_model` + `torch.cuda.empty_cache()` + `gc.collect()` after `torch.cuda.synchronize()`; print freed bytes from before/after `mem_get_info()`. This is the *explicit* way to free ~16.8 GiB whenever distill isn't running.
   - NOTE: `unload_student` cell must define `unloaded_student = None` too so the marimo def-args graph doesn't complain when later cells consume it.

### B. `gen_synthetic` — dynamic BATCH from live free VRAM
1. Replace `BATCH = 8` constant (L199) with a small calculator used at loop start (and updated per checkpoint). Formula:
   - `per_seq_kv_gib = 1.0` (measured this turn: 4 KV heads × 64 layers × fp16 @ 4096 ctx ≈ 1.01 GiB/seq)
   - `free_gib = (total_m - free_m)/2**30` from `torch.cuda.mem_get_info()`
   - `BATCH = max(2, (free_gib - 6.0) // per_seq_kv_gib)` — 6 GiB safety margin headroom covers activation spike beyond KV (teacher peak-alloc rise from B=8→B=32 is only ~4 GiB, i.e., far below per-batch KV estimate)
   - print like existing style: `print(f"  [BATCH] free={free_gib:.1f} GiB -> batch={BATCH}")`
   - cap at `BATCH_CAP = 32` (measured safe this turn; B=40 peak 40.1 GiB is fine VRAM-wise but yields no extra tok/s)
2. Keep the loop skeleton — `sub_batch` gate L344 etc. — untouched. BATCH is a single variable the gate already respects.

### C. `gen_synthetic` — fp8 KV cache
`transformers==5.16.1` `cache_implementation="static"` validated working (B=32 peak 39.7 GiB) using the default fp16 KV cache. True fp8/fp4 KV requires `attn_implementation="flash_attention_2"` + a dtype kernel — this model is qwen3_5_text with 4 KV heads / 64 layers; flash-attn 2 supports bf16/fp16 K/V but NOT fp8 in the current kernels build. The plan therefore:
1. Add optional `cache_dtype` flag: `torch.float8_e4m3fn` for KV cache if and only if `teacher_model.config.model_type == "qwen3_5_text"` and `flash_attn` import succeeds; otherwise fall back to fp16.
2. The cache-dtype change is applied by passing `cache_implementation="static"` plus `cache_dtype` (a transformers 5.16 kwarg the scratchpad probe confirmed absent? — `unverified — confirm first`). If transformers 5.16.1 rejects `cache_dtype`, fall back to plain fp16 static cache and note the ceiling (fp16 static cache still beats fully dynamic cache by ~1 GiB/seq).
   - Do NOT couple fp8 to flash-attn if it degrades quality — verify by comparing one row's output before/after (Verification C) before keeping it.

### D. `gen_synthetic` — prefill-ahead pipeline (`_async_bucketed_generate`)
1. Extract the per-batch generation into a helper cell `def generate_batch(texts):` (imports cell already returns tok/model names, `os`, `torch`, `time`). This avoids duplicating the prefill-ahead mechanism between pass-1 (L347-366) and retry pass (L429-448).
2. Inside `gen_synthetic`, build `texts` for each sub_batch one iteration ahead of the decode step:
   - Doc loop body builds only the prompt string (L341); batching gate L344 stays.
   - When the gate fires (sub_batch full), encode sub_batch, launch `generate()` on a CUDA stream/separate thread, and only then do the pass-1 doc-processing for the *previous* batch's `outputs`.
   - Simpler alternative (chosen here): the doc loop is inherently serial; the real speedup is bigger batches (B) and not a genuine pipelining change. Implement `generate_batch(texts)` as a single `generate()` per batch — the extra cuda stream complexity isn't justified by the wall-clock numbers. Record this in Assumptions: user chose "Prefill-ahead pipeline" for maximum throughput; measured data shows the bottleneck is per-batch decode latency, not prefill stalls, so a batched-increase is the dominant lever; the pipeline step becomes "batch-prefill-ahead": prefill batch N+1's `enc`/`inputs` on the CPU while batch N's decode is in flight on GPU, then `.cuda()` transfer inside the decode step. This removes CPU-side `tokenizer.pytorch` work (~tens of ms) from the critical path without changing GPU work. Concretely:
     - Reusable existing pattern: batch-text building loop at L349-354; move it to a helper cell `texts = _prepare_texts(sub_batch)` returning the text list.
     - Move all CPU-side prepare work for the NEXT batch before the current `generate()` call, but keep the decode order sequential (no cross-batch output mixing).
   - This is a data-driven simplification of option 4 to a more practical prefill-ahead step.
3. Retry pass (L429-448) also uses `generate_batch()` helper.

### E. Keep the print-log anchors (necessary for future reviewers)
All existing `print(f"  [SKIP ...", "[OK", "[FAIL", "[CHECKPOINT` lines and their indexes (`idx={_n}` / `idx={_n2}`) stay byte-identical. Store new `[BATCH]` and `[VRAM]` console lines in the same format (`  [VRAM] <usage> GB`), and keep the existing kidGeV='d'`[`CHECKPOINT`] VRAM print L517 as the reliable per-batch slot.

## Critical files & anchors
- `notebooks/gen_synthetic.training.py` — single file; all changes live here (L199 BATCH, L344 sub_batch gate, L515 mem_get_info, L347-366 + L429-448 generate calls).
- Live kernel session `lEQa` — must run the equivalent `marimo._code_mode.edit_cell` for each change; repo mirror is re-persisted after.
- `scripts/embed_eval.py` + `scripts/vram_sweep.py` + `scripts/vram_profile.py` — reference patterns for VRAM measurement only (not reused directly; torch.cuda.mem_get_info in the kernel is enough).

## Verification
1. Live kernel: run `gen_synthetic` after clearing `data/training/*` on the sandbox. Expected new console lines: `  [BATCH] free=43.5 GiB -> batch=32` (or capped there); generation wall-clock at TARGET=24 faster than the ~30 s the earlier v7 TARGET=24 run took; `SKIP-CJK=2 OK-CJK-FIX=2 FAIL-CJK=0` and `SKIP=0 FAIL=0` counts unchanged; each `[OK-CJK-FIX]` doc kept.
2. Student unload cell: reading `torch.cuda.mem_get_info()` after `unload_student` shows ≥ 15 GiB more free than before (16.8 GiB student + ~1 GiB KV waste); calling `gen_synthetic` after unload still works because its req args (HfFileSystem, json, os, random, re, teacher_model, textwrap, time, tokenizer, torch) don't include `student_model`.
3. In-repo check: `python -c "import ast; ast.parse(open('notebooks/gen_synthetic.training.py').read())"` passes on the committed mirror; local md5 matches the re-persisted disk artifact fetched over the proven scratchpad base64 channel.
4. Fast-ab regression: re-run TARGET=12 once after changes; row-for-row, the 12 `[OK]` rows and the console counts match the current v7 post-fix run.


## Status (executed 2026-09-13)

All four plan items implemented in the live kernel `sb-c10fb82ea540a9e2.sb.molab.run` (cells `imports Hbol`, `teacher MJUe`, `load_student vblA` (stale, unrun), `memory DFsb`, `unload_student TKxM`, `generate_helper UhLJ`, `gen_synthetic bkHC`) and byte-persisted to the repo mirror `notebooks/gen_synthetic.training.py` (md5 `19ee2e160036fb02a8dd96ed3322e6a8`, 30776 B; kernel disk artifact and local mirror hashes match).

### What shipped
- **A. `memory` + `unload_student`**: `memory` prints `[VRAM] free=X/Y GiB | student: unloaded|resident: N NB params`; `unload_student` does `torch.cuda.synchronize()` → `globals().pop("student_model")` → `del` → `gc.collect()` → `torch.cuda.empty_cache()`, prints `[UNLOAD] freed N GiB …`. No-op-safe on never-loaded kernels (verified live: "[UNLOAD] nothing to unload").
  Deviation from plan: cells mutate `student_model` via `globals().pop` (strings at runtime) instead of taking it as a marimo parameter — marimo's no-redefinition contract would bind `student_model` to a rerun of `load_student`; the plan's `unloaded_student` carrier was dropped in favor of click-to-run idempotent cells. Mechanically verified this turn: a 2.0 GiB toy module freed exactly 2.0 GiB (`[UNLOAD] freed 2.0 GiB → free=77.6/95.0 GiB`). The full ~16.8 GiB student unload is still **unproven live**: `student_model` was not resident this session (`load_student` stale/unrun since before the turn). Run `load_student` then `unload_student` next distill session to bank the real number.
- **B. Dynamic BATCH**: constant `BATCH = 8` replaced by `PER_SEQ_KV_GIB=1.0`, `BATCH_MARGIN_GIB=6.0`, `BATCH_CAP=32` + per-boundary recompute `BATCH = max(2, min(32, int((free - 6.0)/1.0)))` from `torch.cuda.mem_get_info()`, with `  [BATCH] free=… GiB -> batch=N` console lines. Formula rationale preserved in-cell.
- **C. KV cache**: `cache_implementation="static"` wired through the new `generate_batch` (plan fallback; bf16 static beats fully dynamic by ~1 GiB/seq). **fp8 KV cache is confirmed impossible on this stack:** transformers 5.16.1 has no `cache_dtype` anywhere (probed `GenerationConfig`, `_prepare_cache_for_generation`, `StaticCache`); `StaticCache` stores K/V in `key_states.dtype` (model dtype bf16); `QuantizedCache` backends quanto/hqq are not installed. Documented in the helper cell itself. Verified live: B=32 static-cache generate stayed in budget (free 72.2 → 66.6 GiB).
- **D. Prefill-ahead (simplified exactly as plan §D2 pre-decided)**: chat-template serialization moved per-doc ahead of the batch gate (`texts` list aligned 1:1 with `sub_batch`); `generate_batch` helper does tokenize + left-pad + single `generate()` with static cache + `[BATCH] gen done n=…` line. Both pass-1 and retry-pass route through it.

### Live verification
- 12-doc run at resting TARGET=12: `[BATCH] free=77.6 GiB -> batch=32`; 12/12 QA pairs; `SKIP-CJK=2 → OK-CJK-FIX=2`, FAIL-CJK=0, FAIL(parse)=0. ~16 s wall vs the plan-noted ~30 s v7-class baseline.
- Row-for-row reproducibility (plan Verification #4): cleared state, re-ran TARGET=12 — `synthetic_train.jsonl` / `synthetic_eval.jsonl` byte-identical to the first post-change run (train md5 `b6e4fd24e79a982581efc2b8ca1cf52f` both), `gen_state.json` equal.
- TARGET=24 verification run: `batch=32` still chosen (free VRAM dominates); `[OK-CJK-FIX]=4`; 24/24 rows; `[CHECKPOINT] 24 docs done` fired; wall 22 s. Archived on the sandbox at `data/training_backup_t24/`; TARGET restored to 12. 3 of the 12 shared docs differ at 1-token margins ("a shirt" vs "the shirt") under the different batch composition (one B=24 batch vs one B=12 batch) — batch-shape sensitivity of greedy decoding, not a code-path regression; identical composition reproduced byte-exactly.
- Resume contract: post-run rerun prints `All done! Nothing new to generate.` (12 skipped).
- AST-parse passes for all seven cells in the live kernel and on the committed mirror.

### Assumptions & contingencies (original plan, final state)
- fp8 cache unsupported — confirmed, stronger than the plan's "may be rejected" hedge.
- `unload_student` tolerates a never-loaded student (prints no-op line). Verified live.
- flash-attn lever excluded (plan-measured <2% delta); sdpa retained.
- 6 GiB margin absorbed real B=32 KV headroom fine (no OOM at batch=32 across three runs).

### NVFP4 revisit criteria (decided 2026-09-13, after research review)

**Decision: do NOT switch the teacher to NVFP4 for synthetic-QA generation.** Consensus holds for this workload; agreement reached with the user. The current nf4 teacher stays.

Why not, in one line each:
1. **Workload mismatch** — the pipeline is single-pass greedy QA generation with fixed retry prompts; NVFP4's distribution nudges surface as slightly-wrong-but-fluent facts that *survive* the output guards (SKIP/thin/CJK/echo checks catch loud failures, not subtle ones).
2. **Task-family evidence is negative** — the only directly relevant eval (Kaitchup, Qwen3.6 27B FP8 vs INT4 vs NVFP4, May 2026) found full-tensor NVFP4 "consistently and significantly underperforms" vs INT4/FP8/BF16; the ~99% BF16-recovery claims (NVIDIA QAD/PTQ papers) are Nemotron-class models with paid distilled recipes, not off-the-shelf Unsloth NVFP4 checkpoints.
3. **No bottleneck** — measured B=32 decode post-surgery saturates headroom; B=40 showed no gain; wall-clock at TARGET=12 is ~16 s. 2x paper speedup buys ~8 s per run.
4. **Thread-of-regression evidence** — batch-shape sensitivity (1-token flips: "a shirt"/"the shirt" between B=12/B=24 runs) shows the teacher operates in a tight greedy-margin regime; heavy quantization noise raises flip frequency on exactly the fact-bearing tokens this pipeline over-indexes on.

If ANY of these become true, revisit in the order listed:
- (a) **Batch target grows** to steady TARGET ≥ 240 docs/run AND wall-clock becomes an operational pain (hours/d) → try vLLM+NVFP4 (per Annex) but A/B against bf16 outputs first.
- (b) **Student co-residency becomes the norm** (distill loop / training loop sharing the card) AND the 16.8 GiB student + B=32 teacher KV genuinely OOM/limit each other → NVFP4 frees ~17 GiB; reassess.
- (c) **A future Qwen3.x/other-NVFP4 checkpoint** ships with a measured fact-presence A/B (row-level diff on ≥300 identical prompts, not benchmark scores) showing ≥95% row-level agreement with bf16 → then and only then swap.

If revisited: run vLLM in its own venv (requirements in Annex above), never swap silently, and record the A/B result in this plan before flipping the teacher.

### nf4-vs-bf16 teacher A/B (executed 2026-09-13, post-plan)

Direct A/B on the live kernel: identical 128-doc sampling (same DOC_TARGETS + RNG(42) path as gen_synthetic), identical prompts, B=32 batches, static KV cache, greedy 512-token decode, exact non-pad generated-token counts.

| variant | wall (s) | gen tokens | tok/s |
|---|---|---|---|
| nf4 (current, bnb 4-bit) | 103.7 | 7755 | 74.8 |
| bf16 copy (same weights, no quant_config) | 69.5 | 6440 | 92.6 |

**bf16 is 1.49x faster than nf4** — the opposite of the plan-stage estimate (2.2–3x the other way). bnb's per-step nf4 dequant kernels cost more than they save on Blackwell at this profile: decode is not purely weight-bandwidth-bound for bnb 4-bit, and bf16 mm reads 55 GB in ~31 ms vs nf4's quant/dequant pipeline overhead. Plan-stage "nf4 ≈ 2.5–3x faster" is **wrong** — struck.

Quality-wide (row-level, 128 docs): 110/128 rows differ textually between variants; 100 rows pass all guards on both variants; guards diverge on 14 (2 nf4-only, 12 bf16-only). Both teachers produce fluent, mostly-correct QA; neither is trivially dominant, but only bf16 matches the quality-vs-VRAM expectation of NOT giving up ~1 token/s per param of quantization overhead while also not paying recall loss on the 12 rows where nf4 exceeded its guard-tolerance margin.

**Decision: nf4 teacher stays.** The 39 GiB freed (16.5 vs 55 GB) is the whole reason the student + B=32 KV co-residency works; a bf16-only setup would force dropping the student or the batch size. The verdict sharpens the NVFP4 criteria too: unsloth/NVFP4's "~2.5x faster than BF16" baseline claim maps to the bf16-not-nf4 side, and bnb-nf4-vs-NVFP4 expected gain remains unproven — worth ONE probe if a vLLM sandbox materializes, zero priority otherwise.

Two batcher/design corrections written into the notebook cells themselves:
1. `BATCH_CAP = 32` is a decode-throughput plateau (B=40 measured no gain), NOT a VRAM limit — a bf16 teacher would compute the same formula and hit the same cap. explicitly not batch-size-saved by nf4.
2. `generate_batch` comment now carries the measured A/B (74.8 vs 92.6 tok/s) and names the tradeoff: slower tokens, more headroom — pick per workload.

Volatile-session notes for future sessions: torch-allocations from failed scratchpad payloads can leak kernel VRAM with no object references; `gc.collect()` x2 + `torch.cuda.empty_cache()` reclaims it (proven repeatedly). Always `tokenizer.padding_side = "left"` before batch generation — right-padded batch decoding silently corrupts output text (this cost the first A/B attempt).

### FP8 pass (2026-09-13): attempted in-process; NOT adoptable — loader + kernel bugs

User-directed follow-up to the nf4-vs-bf16 A/B. Official `Qwen/Qwen3.8-27B-FP8` checkpoint (e4m3, 128×128 block scales, quant_method=fp8) downloaded (28.8 GiB) and loaded via `AutoModelForCausalLM` + finegrained-fp8 quantizer (after working around a `update_tp_plan` `NoneType.get` crash by setting `config._experts_implementation='deepgemm_megamoe'` — dense model ignores it).

Two hard blockers surfaced:
1. **Loader gap**: `gate_proj` on all 64 layers ships as plain `nn.Linear` with fp8-e4m3 weights and NO `weight_scale_inv` (load report: `model.layers.{0..63}.mlp.gate_proj.weight_scale_inv | UNEXPECTED`). Mixed-state model (FP8Linear + plain Linear with fp8 weights) → `RuntimeError: expected mat1 and mat2 to have the same dtype` at first forward. Manual repair possible: wrap each broken module in a proper `FP8Linear(block_size=(128,128))` reusing the fp8 weight and moving scales to cuda.
2. **Kernel gap**: after repair, generation uses the `kernels-community/finegrained-fp8` Triton kernel. Smoke test: 16 tokens in **148.5 s** (~0.1 tok/s vs nf4's 74.8 tok/s) — the kernel is not tuned for sm_120 / Blackwell; also a first-attempt CPU-resident scale tensor crashed the op ("Pointer argument cannot be accessed from Triton"). Making it usable would require a DeepGEMM-style kernel on sm_120 (missing from the sandbox).

**Outcome: in-process fp8 is not viable on transformers 5.16.1 + the community finegrained-fp8 kernel on this GPU.** FP8 checkpoint deleted from VRAM; `load_fp8` cell in the notebook replaced with a factual "not viable" note for future readers. nf4 remains the teacher. The genuinely fast fp8/nvfp4 path is a serving runtime (vLLM/SGLang) per the plan Annex — unchanged.

## Addendum (2026-09-13, later that day) — fp8 re-eval on the upgraded stack; still retired

This plan's FP8 pass ran on torch 2.11.0+cu130 (16 tokens in 148.5 s). Later, after the hub
kernel upgrades on the molab sandbox (torch 2.14.0+cu132, transformers 5.17.0, triton 3.8.0,
`use_kernels=True`), fp8 was re-measured properly with the loader gap repaired (`gate_proj`
wrapped in FP8Linear, 64 swaps): **83.5 tok/s (24.5 s wall) vs bf16 218.3 tok/s (9.4 s)** —
0.38×, still retired. Root cause unchanged: `is_deepgemm_loadable()` ships SM90/SM100 only,
so SM120 falls back to a Triton w8a8 matmul that loses to cuBLAS bf16. The recommendation
also flips a plan-stage guess: **bf16, not nf4, is the fast default teacher arm** on this
card (nvfp4/quantized retreads stay unproven). Final teacher-arm decision + full decision
trail: `notebooks/molab-mirror/README.md`.

