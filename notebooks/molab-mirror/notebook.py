# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "bitsandbytes==0.50.2",
#     "cuda-bindings==13.3.1",
#     "cuda-pathfinder==1.8.1",
#     "datasets==5.0.1",
#     "evaluate==0.4.6",
#     "huggingface-hub==1.31.0",
#     "kernels==0.16.1",
#     "kernels-data==0.16.1",
#     "marimo[mcp]>=0.24.0",
#     "mcp>=1",
#     "openai==3.8.0",
#     "peft==0.20.0",
#     "pydantic>=2",
#     "python-lsp-ruff==2.3.3",
#     "python-lsp-server==1.15.0",
#     "ruff==0.16.6",
#     "safetensors==0.8.0",
#     "sigstore==4.5.0",
#     "sigstore-models==0.0.6",
#     "sigstore-rekor-types==0.0.18",
#     "sse-starlette==3.4.11",
#     "starlette==1.6.0",
#     "timm==1.0.29",
#     "tokenizers==0.23.2",
#     "torch==2.14.0",
#     "tqdm==4.70.0",
#     "transformers==5.17.0",
#     "trl==1.12.0",
#     "urllib3==2.7.0",
#     "websockets==17.1",
#     "xxhash==4.0.1",
#     "yarl==1.24.5",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    return


@app.cell
def _():
    # Allocator: expandable segments avoid permanent fragmentation when swapping teacher
    # models (54 GiB bf16) in and out of VRAM.

    import os
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    import subprocess, json, random, re, time, os, gc, math, torch
    import marimo as mo
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import notebook_login, HfFileSystem
    from pathlib import Path
    import torch.nn.functional as F
    import textwrap
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    #uv pip install -U peft marimo datasets huggingface-hub transformers --torch-backend=auto
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        HfFileSystem,
        LoraConfig,
        TaskType,
        get_peft_model,
        json,
        mo,
        os,
        random,
        re,
        textwrap,
        time,
        torch,
    )


@app.function
def generate_batch(texts, teacher_model, tokenizer, torch, max_new_tokens=512):
    """Batched teacher generate helper.

    Pads from the LEFT (decoder-only batch generation): padded batch rows keep the last
    real token flush with the row end, which is what generate() needs to continue from.
    Set per call so no other notebook cell inherits the side-flip.

    Static KV cache (cache_implementation="static"): bf16, ~1 GiB/seq at 4k ctx (KV stores K/V in the model dtype).
    """
    tokenizer.padding_side = 'left'
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
    gen_inputs = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        gen_outputs = teacher_model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cache_implementation='static',
        )
    _free, _tot = torch.cuda.mem_get_info()
    print(f'  [BATCH] gen done n={len(texts)} free={_free / 2**30:.1f} GiB')
    return gen_outputs, gen_inputs


@app.cell
def _():
    return


@app.cell
def _(AutoModelForCausalLM, LoraConfig, TaskType, get_peft_model, torch):
    STUDENT_MODEL = "ornith-ai/Ornith-1.5-9B"
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    return


@app.cell
def _(torch):
    _free, _tot = torch.cuda.mem_get_info()
    _student = globals().get("student_model")
    if _student is None:
        print(f"  [VRAM] free={_free / 2**30:.1f}/{_tot / 2**30:.1f} GiB | student: unloaded")
    else:
        _n_params = sum((p.numel() for p in _student.parameters()))
        print(
            f"  [VRAM] free={_free / 2**30:.1f}/{_tot / 2**30:.1f} GiB | student resident: {_n_params / 1000000000.0:.2f}B params"
        )
    return


@app.cell
def _(torch):
    import gc as _gc

    _student = globals().get("student_model")
    if _student is None:
        print("  [UNLOAD] nothing to unload (student already free or never loaded)")
    else:
        _free0, _tot = torch.cuda.mem_get_info()
        torch.cuda.synchronize()
        globals().pop("student_model", None)
        del _student
        _gc.collect()
        torch.cuda.empty_cache()
        _free1, _ = torch.cuda.mem_get_info()
        print(
            f"  [UNLOAD] freed {(_free1 - _free0) / 2**30:.1f} GiB -> free={_free1 / 2**30:.1f}/{_tot / 2**30:.1f} GiB"
        )
    return


@app.cell
def _(
    HfFileSystem,
    inputs,
    inputs2,
    json,
    os,
    outputs,
    outputs2,
    random,
    re,
    teacher_model,
    textwrap,
    time,
    tokenizer,
    torch,
):
    "\n    Generate synthetic QA data from HF bucket documents.json\n    using the already-loaded teacher model.\n\n    Resume-safe: re-run the cell to continue where it left off.\n    State saved to data/training/gen_state.json after every checkpoint.\n\n    Run after cell order: imports -> teacher -> (load_student) -> (memory/unload_student) -> gen_synthetic.\n    All imports provided by the shared imports cell; batched generate lives in generate_helper (dynamic BATCH + static KV cache).\n    Hardware: molab-class GPU (CUDA cuda:0) -- unload the student first to free ~16.8 GiB, then BATCH scales to live free VRAM (capped at 32).\n"

    BASE_PROMPT = textwrap.dedent(
        "    You are a Project Gorgon game assistant.\n\n    Answer the user's question using the provided context.\n\n    Rules:\n    - Figure out what the user is really asking, even when the question is informal, and assemble the answer from the context — the exact answer need not be stated word-for-word in the documents.\n    - Reason from the context: connect information across documents, compare and rank options, and draw conclusions that follow from the stated facts. Planning a path or choosing the most efficient option from stated values is expected and helpful.\n    - Include relevant names, skills, levels, ingredients, and quantities when available.\n    - If the user asks about recipes, include the recipe name, ingredients with quantities, required skill level, and the recipe's description text (e.g. dose counts, effects, results).\n    - If multiple answers exist, list them.\n    - If the context contains PARTIAL information for the question, answer with exactly what is present and explicitly state what is missing — do not refuse the whole question because one detail is absent.\n    - Only say you do not know when the context contains nothing relevant: no facts, names, levels, or values that bear on the question.\n    - NEVER fabricate: do not invent facts, names, values, recipes, XP numbers, formulas, or mechanics that are not present in the context. Arithmetic directly derived from stated context values (such as calculating the difference between two stated cumulative XP totals: Target Level cumulative XP minus Start Level cumulative XP) is grounded analysis, not fabrication. If a specific number is not stated or derivable from the context, say it is not stated rather than guessing.\n    - NEVER cite sources that are not listed in the provided context.\n    - When listing sources, only reference documents that actually contributed to your answer.\n    "
    )
    BUCKET_PATH = "hf://buckets/Nubula/paddock/documents.json"
    TARGET = 12
    MAX_DOC_CHARS = 2048
    BATCH_LOG = 8
    CHECKPOINT_EVERY = 8
    GEN_MAX_TOKENS = 512
    RNG = random.Random(42)
    OUT_DIR = "data/training"
    STATE_FILE = f"{OUT_DIR}/gen_state.json"
    OUT_TRAIN = f"{OUT_DIR}/synthetic_train.jsonl"
    OUT_EVAL = f"{OUT_DIR}/synthetic_eval.jsonl"
    PER_SEQ_KV_GIB = 1.0
    BATCH_MARGIN_GIB = 6.0
    BATCH_CAP = 32
    DOC_TARGETS = [
        ("item", 2000),
        ("recipe", 2000),
        ("ability", 1500),
        ("quest", 1000),
        ("npc", 500),
        ("wiki", 1000),
        ("skill", 300),
        ("summary", 200),
        ("curated", 200),
        ("effect", 200),
        ("leveling", 100),
        ("lorebook", 100),
        ("combatxp", 50),
        ("xptable", 50),
    ]
    QUESTION_HINTS = [
        "Focus on what the item or ability IS USED FOR (results, effects, purpose) — not just skill requirements or ingredients.",
        "Create a COMPARISON question the context can uniquely answer (e.g. which value is highest, which recipe yields more). Give a rich, complete answer: state both compared values fully with quantities/units, identify the winner, and explain why — include what the comparison is and the answer.",
        "Create a HOW-TO/STRATEGY question about leveling or completing, answerable from step-by-step info in the context (XP needed, activities, trainers, unlock order).",
        "Create a question about quantities, cadence, cost, or specific numeric values in the context (stack size, value, doses, durations, uses), and give the answer with surrounding context — the number alone is not an answer.",
        "Ask about identity/naming (internal names, keywords, categories, shell names).",
    ]
    PROMPT = "Based on this data, in English, write one question a player might ask and its answer, using only the information given. Do not number the question; give exactly one question with no enumerations.\n\nData:\n{document_text}\n\n{q_hint}\nAnswer:"
    print(f"Loading documents from {BUCKET_PATH} ...")
    fs = HfFileSystem()
    with fs.open(BUCKET_PATH, "rb") as f:
        all_docs = json.loads(f.read())
    print(f"Loaded {len(all_docs)} documents")
    by_type = {}
    for d in all_docs:
        t = d.get("type", "unknown")
        by_type.setdefault(t, []).append(d)
    sampled = []
    seen_ids = set()
    for t, n in DOC_TARGETS:
        pool = by_type.get(t, [])
        pool.sort(key=lambda d: len(d.get("text", "")), reverse=True)
        RNG.shuffle(pool)
        count = 0
        for d in pool:
            _pid = d.get("id") if isinstance(d, dict) else d
            if _pid is not None and _pid not in seen_ids:
                sampled.append(d)
                seen_ids.add(d["id"])
                count += 1
                if count >= n:
                    break
        print(f"  {t}: {min(n, count)} sampled ({len(pool)} available)")
    if len(sampled) > TARGET:
        RNG.shuffle(sampled)
        sampled = sampled[:TARGET]
    print(f"Total docs to process: {len(sampled)} (target={TARGET})")
    processed_ids = set()
    _legacy_keys = set()
    existing_rows = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            processed_ids = set(state["processed_ids"])
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(
                f"  [WARN] {STATE_FILE} unreadable ({type(e).__name__}: {e}); rebuilding from scanned rows"
            )
    if os.path.exists(OUT_TRAIN):
        with open(OUT_TRAIN) as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [WARN] {OUT_TRAIN}: skipping malformed row {line_no + 1}")
                    continue
                md = r.get("metadata", {})
                row_doc_id = md.get("doc_id")
                if row_doc_id is not None:
                    processed_ids.add(row_doc_id)
                else:
                    _legacy_keys.add((md.get("doc_type"), md.get("doc_name")))
                existing_rows += 1
    print(f"Resuming: {len(processed_ids)} docs already processed, {existing_rows} existing rows")


    def _doc_skipped(d):
        if not isinstance(d, dict) or "id" not in d:
            return False
        _name = d.get("metadata", {}).get("name", d["id"])
        return d["id"] in processed_ids or (d.get("type", "unknown"), _name) in _legacy_keys


    to_process = [d for d in sampled if not _doc_skipped(d)]
    skipped = len(sampled) - len(to_process)
    print(f"To process now: {len(to_process)} docs ({skipped} skipped)")
    if not to_process:
        print("All done! Nothing new to generate.")
    else:
        teacher_model.eval()
        print(f"Using tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")
        rows = []
        sub_batch = []
        texts = []
        _flushed = 0
        new_processed = set(processed_ids)
        retry_pending = []
        start = time.time()
        for i, doc in enumerate(to_process):
            doc_id = doc["id"] if isinstance(doc, dict) else doc
            doc_text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            doc_name = doc.get("metadata", {}).get("name", doc_id) if isinstance(doc, dict) else doc
            doc_type = doc.get("type", "unknown") if isinstance(doc, dict) else "unknown"
            if i == 0:
                print(f"  Sampled {len(to_process)} docs. Generating ...")
            if len(doc_text) < 50:
                new_processed.add(doc_id)
                continue
            if len(doc_text) > MAX_DOC_CHARS:
                doc_text = doc_text[:MAX_DOC_CHARS] + "\n[... truncated]"
            q_hint_i = i % 5
            q_hint = QUESTION_HINTS[q_hint_i]
            prompt = PROMPT.format(document_text=doc_text, q_hint=q_hint)
            messages = [{"role": "user", "content": prompt}]
            _text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _text = _text + "Question: "
            texts.append(_text)
            sub_batch.append((doc, doc_id, doc_name, doc_type, doc_text, prompt))
            _free_gib = torch.cuda.mem_get_info()[0] / 2**30
            BATCH = max(2, min(BATCH_CAP, int((_free_gib - BATCH_MARGIN_GIB) / PER_SEQ_KV_GIB)))
            if len(sub_batch) < BATCH and i + 1 < len(to_process):
                continue
            print(f"  [BATCH] free={_free_gib:.1f} GiB -> batch={BATCH}")
            try:
                gen_out, gen_in = generate_batch(
                    texts, teacher_model, tokenizer, torch, max_new_tokens=GEN_MAX_TOKENS
                )
                for _n, _sb in enumerate(sub_batch):
                    _doc, doc_id, doc_name, doc_type, doc_text, _prompt = _sb
                    response = tokenizer.decode(
                        outputs[_n][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                    ).strip()
                    parts = re.split("\\n?\\s*Answer:\\s*", response, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        question = parts[0].strip().lstrip("Question: ").strip()
                        question = re.sub("^\\s*\\d+\\.\\s+", "", question)
                        answer = parts[1].strip()
                        answer = re.sub("\\s*<\\|im_end\\|>.*", "", answer, flags=re.DOTALL).strip()
                        answer = re.sub("^\\s*response\\b[:\\s]*", "", answer).strip()
                        if question.lower().startswith(("question", "data", "answer", "here is")):
                            print(
                                f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} meta-question: {question[:80]!r}"
                            )
                            retry_pending.append(_sb)
                            continue
                        if len(answer) < 20 or len(question) < 15:
                            print(
                                f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r}  ans_len={len(answer)} q_len={len(question)} — too thin for SFT"
                            )
                            retry_pending.append(_sb)
                            continue
                        if any((ord(ch) > 11903 for ch in question + answer)):
                            print(
                                f"  [SKIP-CJK] idx={_n} doc={doc_name[:30]!r} — non-English output; will retry with English-only prompt"
                            )
                            retry_pending.append(_sb)
                            continue
                        if "?" in answer or "Question:" in answer:
                            print(
                                f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} — answer echoes a question"
                            )
                            retry_pending.append(_sb)
                            continue
                        truncated = doc_text
                        user_content = f"{BASE_PROMPT}\n\nContext:\n{truncated}\n\nQuestion: {question}"
                        row = {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": answer},
                            ],
                            "metadata": {
                                "source": "synthetic",
                                "doc_id": doc_id,
                                "doc_type": doc_type,
                                "doc_name": doc_name,
                            },
                        }
                        rows.append(row)
                        new_processed.add(doc_id)
                        if len(rows) <= 3:
                            print(f"  [OK] idx={_n} Q={question[:60]}  A={answer[:60]}")
                    else:
                        print(
                            f"  [FAIL] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} split={len(parts)} raw={response[:250]}"
                        )
                        retry_pending.append(_sb)
                if retry_pending:
                    texts2 = []
                    for _sb in retry_pending:
                        _en = "IMPORTANT: Write the question and answer in English only.\n\n"
                        messages2 = [
                            {
                                "role": "user",
                                "content": _en
                                + PROMPT.format(document_text=_sb[4], q_hint=QUESTION_HINTS[0]),
                            }
                        ]
                        text2 = tokenizer.apply_chat_template(
                            messages2, tokenize=False, add_generation_prompt=True, enable_thinking=False
                        )
                        text2 = text2 + "Question: "
                        texts2.append(text2)
                    gen_out2, gen_in2 = generate_batch(
                        texts2, teacher_model, tokenizer, torch, max_new_tokens=GEN_MAX_TOKENS
                    )
                    for _n2, _sb in enumerate(retry_pending):
                        _doc, doc_id, doc_name, doc_type, doc_text, _prompt = _sb
                        response2 = tokenizer.decode(
                            outputs2[_n2][inputs2["input_ids"].shape[1] :], skip_special_tokens=True
                        ).strip()
                        parts2 = re.split(
                            "\\n?\\s*Answer:\\s*", response2, maxsplit=1, flags=re.IGNORECASE
                        )
                        if len(parts2) == 2:
                            question = parts2[0].strip().lstrip("Question: ").strip()
                            question = re.sub("^\\s*\\d+\\.\\s+", "", question)
                            answer = parts2[1].strip()
                            answer = re.sub("\\s*<\\|im_end\\|>.*", "", answer, flags=re.DOTALL).strip()
                            answer = re.sub("^\\s*response\\b[:\\s]*", "", answer).strip()
                            if any((ord(ch) > 11903 for ch in question + answer)):
                                print(
                                    f"  [FAIL-CJK] idx={_n2} doc={doc_name[:30]!r} — still non-English after retry; doc skipped permanently"
                                )
                                new_processed.add(doc_id)
                                continue
                            if (
                                "?" in answer
                                or "Question:" in answer
                                or len(answer) < 20
                                or (len(question) < 15)
                            ):
                                print(
                                    f"  [SKIP-R] idx={_n2} doc={doc_name[:30]!r} — guard hit after retry"
                                )
                                new_processed.add(doc_id)
                                continue
                            truncated = doc_text
                            user_content = (
                                f"{BASE_PROMPT}\n\nContext:\n{truncated}\n\nQuestion: {question}"
                            )
                            row = {
                                "messages": [
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": answer},
                                ],
                                "metadata": {
                                    "source": "synthetic",
                                    "doc_id": doc_id,
                                    "doc_type": doc_type,
                                    "doc_name": doc_name,
                                },
                            }
                            rows.append(row)
                            print(
                                f"  [OK-CJK-FIX] idx={_n2} doc={doc_name[:30]!r} Q={question[:50]}  A={answer[:50]}"
                            )
                        else:
                            print(
                                f"  [FAIL] idx={_n2} doc={doc_name[:30]!r} split={len(parts2)} raw={response2[:150]}"
                            )
                        new_processed.add(doc_id)
                    retry_pending = []
            except Exception as e:
                print(f"  [ERR] batch at #{i + 1} {type(e).__name__}: {str(e)[:150]}")
            sub_batch = []
            texts = []
            if (i + 1) % BATCH_LOG == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                avg = elapsed / (i + 1) if elapsed > 0 else 0
                print(
                    f"  [{i + 1}/{len(to_process)}] doc {doc_name[:30]} {len(rows)} rows ({rate * 60:.1f} docs/min, {avg:.1f}s/doc)"
                )
            if (i + 1) % CHECKPOINT_EVERY == 0:
                new_rows = rows[_flushed:]
                if new_rows:
                    os.makedirs(OUT_DIR, exist_ok=True)
                    with open(OUT_TRAIN, "a") as f:
                        for r in new_rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    _flushed = len(rows)
                    with open(STATE_FILE, "w") as f:
                        json.dump({"processed_ids": sorted(new_processed)}, f)
                    free_m, total_m = torch.cuda.mem_get_info()
                    print(
                        f"  [CHECKPOINT] {len(new_processed)} docs done, +{len(new_rows)} rows appended, VRAM: {(total_m - free_m) / 1000000000.0:.1f}/{total_m / 1000000000.0:.1f} GB"
                    )
        if rows:
            os.makedirs(OUT_DIR, exist_ok=True)
            remaining = rows[_flushed:]
            if remaining:
                with open(OUT_TRAIN, "a") as f:
                    for r in remaining:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            all_rows = []
            if os.path.exists(OUT_TRAIN):
                with open(OUT_TRAIN) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"  [WARN] {OUT_TRAIN}: dropping malformed row during eval split")
            eval_rows_final = []
            train_rows_final = []
            groups = {}
            for _r in all_rows:
                groups.setdefault(_r.get("metadata", {}).get("doc_type", "unknown"), []).append(_r)
            for _grp in groups.values():
                RNG.shuffle(_grp)
                _n = max(1, len(_grp) // 20)
                eval_rows_final.extend(_grp[:_n])
                train_rows_final.extend(_grp[_n:])
            with open(OUT_TRAIN, "w") as f:
                for r in train_rows_final:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(OUT_EVAL, "w") as f:
                for r in eval_rows_final:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(STATE_FILE, "w") as f:
                json.dump({"processed_ids": sorted(new_processed)}, f)
            elapsed = time.time() - start
            print(f"\n{'=' * 50}")
            print(f"Generated {len(rows)} new QA pairs this session")
            print(f"Total dataset: {len(train_rows_final)} train + {len(eval_rows_final)} eval")
            print(f"Time: {elapsed:.0f}s  ({len(rows) / elapsed:.1f} rows/s)")
            print(f"To resume later: re-run cell (skips already done docs)")
        else:
            print("No rows generated (docs may be too short or parsing failed)")
    return


@app.cell
def kernel_map():
    """Hub-kernel mapping overrides: none needed on transformers 5.17.0 + kernels 0.16.1.

    Tested and rejected during the bf16/fp8 teacher A/B (2026-09-14, see bench_results):
      - RMSNormZeroCentered/Qwen3_5RMSNorm -> LigerRMSNorm v3: LigerRMSNorm.forward
        expects variance_epsilon which Qwen3_5RMSNorm lacks (eps only) - API-incompatible.
      - Qwen3_5GatedDeltaNet -> mamba-ssm: kernel repo exports functions only
        (causal_conv1d_*, selective_state_update, ...), not the layer class, so the
        class-name mapping raises ValueError in kernelize.

    causal_conv1d / mamba / fla function kernels are wired natively by transformers itself.
    """
    # (kernels-community overrides intentionally absent)

    return


@app.cell
def teacher_bf16(AutoModelForCausalLM, AutoTokenizer, torch):
    """DEFAULT TEACHER: bf16 + use_kernels=True. ~54 GiB resident weights before kernels hub load."""
    TEACHER_MODEL_BF16 = "Qwen/Qwen3.8-27B"

    _free0, _tot = torch.cuda.mem_get_info()
    print(f"[BF16 LOAD] free before: {_free0 / 2**30:.1f} GiB")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_BF16,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        use_kernels=True,
    )
    teacher_model.eval()
    print(
        f"Teacher (bf16): {teacher_model.config.model_type}, "
        f"{teacher_model.num_parameters():,} params | "
        f"free after: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB"
    )

    bf16_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_BF16)
    if bf16_tokenizer.pad_token is None:
        bf16_tokenizer.pad_token = bf16_tokenizer.eos_token
    print(f"Tokenizer loaded: {type(bf16_tokenizer).__name__}, vocab={bf16_tokenizer.vocab_size}")
    return (teacher_model,)


@app.cell
def ab_bench(HfFileSystem, json, random, torch):
    """Teacher throughput bench (bf16 default arm).

    Fixed batch, identical seeded prompts from the notebook's own HF corpus, greedy.
    Uses the resident teacher_model + bf16_tokenizer pair (the only teacher arm; the
    fp8 arm was measured 0.38x slower and retired — full A/B in bench_results).
    """
    import time as _time

    BENCH_BATCH = 8
    BENCH_MAX_NEW = 256

    # --- identical prompts for both runs, sampled from the notebook's own corpus ---
    _fs = HfFileSystem()
    with _fs.open("hf://buckets/Nubula/paddock/documents.json", "rb") as _f:
        _all_docs = json.loads(_f.read())
    _rng = random.Random(42)
    _docs_sample = _rng.sample([_d for _d in _all_docs if len(_d.get("text", "")) > 800], BENCH_BATCH)
    _prompts = []
    for _d in _docs_sample:
        _text = _d["text"][:1800]
        _msgs = [
            {
                "role": "user",
                "content": (
                    "You are a Project Gorgon game assistant.\n\n"
                    "Using ONLY the context, answer the question in 2-4 sentences.\n\n"
                    "Context:\n"
                    + _text
                    + "\n\nQuestion: "
                    + _d.get("metadata", {}).get("name", "What is this?")
                ),
            }
        ]
        _serialized = globals()["bf16_tokenizer"].apply_chat_template(
            _msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        _prompts.append(_serialized)

    print(f"[BENCH] prompts built: {len(_prompts)} (avg chars={sum(len(p) for p in _prompts) // len(_prompts)})")


    def bench_bf16(model, tok, prompts, batch, max_new):
        tok.padding_side = "left"
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.cuda() for k, v in enc.items()}
        torch.cuda.synchronize()
        _t0 = _time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        elapsed = _time.time() - _t0
        _free, _tot = torch.cuda.mem_get_info()
        return {"tok_per_s": batch * max_new / elapsed, "total_s": elapsed, "free_gib": _free / 2**30}


    results = {}

    # --- bf16 benchmark ---
    teacher_model_bench = globals().get("teacher_model")
    if teacher_model_bench is None:
        print("[BENCH] bf16 arm skipped - teacher_model not resident (run the bf16 loader first)")
        print("(no arms ran - load a teacher and rerun)")
    else:
        results["bf16"] = bench_bf16(
            teacher_model_bench, globals()["bf16_tokenizer"], _prompts, BENCH_BATCH, BENCH_MAX_NEW
        )

    print("\n=== BF16 BENCH (fixed batch=8, identical prompts, greedy) ===")
    if "bf16" in results:
        _b = results["bf16"]
        print(f"bf16: tok/s={_b['tok_per_s']:8.1f} wall={_b['total_s']:6.1f}s free-after={_b['free_gib']:.1f} GiB")
    else:
        print("(bf16 arm not run yet - rerun after loading the bf16 teacher)")
    return


@app.cell
def bench_results(mo):
    mo.md(r"""
    # FP8 vs BF16 teacher smoke test — Qwen3.8-27B on RTX PRO 6000 Blackwell (SM 12.0)

    Fixed batch=8, identical seeded prompts (Project Gorgon corpus), greedy, 256 new tokens.
    Stack: torch 2.14.0+cu132, transformers 5.17.0, triton 3.8.0, `use_kernels=True`.

    | Arm | tok/s | wall |
    |---|---|---|
    | bf16 (sdpa + hub kernels) — **DEFAULT, kept** | **218.3** (warm-cache high: 265.8) | 9.4 s |
    | fp8 (FineGrainedFP8 + gate repair) — **retired, loader deleted** | 83.5 | 24.5 s |

    Why fp8 lost: `is_deepgemm_loadable()` only ships SM90/SM100 kernels, so SM120 falls back
    to a Triton w8a8 matmul that is slower than cuBLAS bf16 on this card. fp8 saves 27 GiB
    VRAM, but on a 95 GiB card that trade is never worth ~2.6x throughput.
    """)
    return


if __name__ == "__main__":
    app.run()
