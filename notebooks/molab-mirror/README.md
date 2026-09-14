# molab notebook mirror — Qwen3.8 distillation pipeline (bf16 final config)

Mirror of everything relevant on the molab instance (sb-dda4216fe38ee51a), verified by sha256.

## Files
- `notebook.py` — live marimo notebook, FINAL config (fp8 retired 2026-09-14):
  - imports cell: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (before torch import)
  - `kernel_map` cell: intentional EMPTY overrides (Liger + GDN mappings tested and
    rejected, see NOTES); transformers wires causal-conv1d/GPL/mamba fn kernels itself
  - `teacher_bf16` (DEFAULT, only teacher): bf16 + use_kernels=True, ~54 GiB
    (docstring is fixed title; no fp8 arm anymore)
  - `bench_bf16` cell + rendered `bench_results` markdown: A/B verdict showing
    bf16 218.3 tok/s (default) vs retired fp8 83.5 tok/s 0.38x
    (fp8 optional only if 27 GiB headroom ever matters; DeepGEMM SM90/SM100 gate)
- `pyproject.toml` / `lock.txt` — remote env (torch 2.14.0+cu132, transformers 5.17.0,
  triton 3.8.0, kernels 0.16.1, marimo uv sandbox deps)

## Pipeline roles of scripts (these run CLI-side, not from the notebook)
1. `generate_training_data.py` — CoreWeave/W&B DeepSeek-V4-Flash: golden-eval
   augmentation + curated synthetic QA (needs omp agent.db credential; reads
   `data/golden/*.json` + `data/documents.json` from pg-rag-builder).
2. `run_generate_data.py` — molab side local-teacher synthetic Qwen3.8-27B generator
   (documents.json -> expanded_train/eval, merges into unsloth_*).

   NOTE: notebook cell `SFPL` does the SAME job in-notebook (resume-safe via
   `data/training/gen_state.json`); scripts remain the CLI path.
3. `run_sft.py` — TRL SFTTrainer loop over unsloth_train/eval.jsonl; LoRA r=16 on student
   `ornith-ai/Ornith-1.5-9B`; pushes to HF `Nubula/ornith-pgrag-training1`.
4. `run_distill.py` — KL+CE temperature-2 distillation: teacher logits into the student
   (LoRA), checkpoints in `distill_output/checkpoints`, adapter in `distill_output/lora`,
   merged model in `distill_output/final`.
5. `benchmark_opts.py` — 5-step TRL micro-bench for student optimization flags
   (superseded by the notebook A/B for teacher-side fp8-vs-bf16).

## Student vs goldens (RAG path)
`mise.toml` pins the PRODUCTION model as `LLM_MODEL=ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`
(RAG answers path). To validate a remote-trained student:
  (a) convert the merged LoRA to GGUF and point `LLM_MODEL` at it, then `mise golden-short`;
  (b) or serve remotely with transformers and adapt `scripts/golden_check.py` to target the
      remote endpoint.

## Decision trail (keep documented)
- transformers shim removed (upstream 5.17 fixed `update_tp_plan`).
- `Qwen3_5GatedDeltaNet` -> mamba-ssm override REMOVED: kernel hub repo exports FUNCTIONS
  (causal_conv1d_*/selective_state_update/etc.); the whole layer mapping is invalid shape.
- `RMSNormZeroCentered` -> LigerRMSNorm override REMOVED: LigerRMSNorm.forward wants
  `variance_epsilon` but Qwen3_5RMSNorm only has `eps` (API mismatch). Normal layers stay on
  the reference wrapper cost (~128 norms/fwd) until transformers ships a compatible mapping.
- fp8 arm retired: HF Triton w8a8 fallback on SM120 measured 83.5 tok/s vs bf16 218.3.
- allocator: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at import time.

## Final cleanup pass (2026-09-14)
- Notebook trimmed: `update_tp_plan` shim comment removed; fp8-KV note removed from
  `generate_batch`; kernel_map cell slimmed to outcome-only docstring; teacher_bf16 and
  bench docstrings references to `A/B` retirement trimmed; bench_results markdown tightened.
- Final remote sha256: `9875a4994c16f013984a1c1095aa7efe3a4eb312934f467d81b8f225f410fcb5`.
- Weight-VRAM after full bf16-bench cycle: 43.8 GiB driver-free, 0.03 GiB torch-alloc
  (everything else is reusable reserved pool).
