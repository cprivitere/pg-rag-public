# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "bitsandbytes==0.50.2",
#     "cuda-bindings==13.3.1",
#     "cuda-pathfinder==1.8.1",
#     "datasets==5.0.1",
#     "evaluate==0.4.6",
#     "huggingface-hub==1.30.0",
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
#     "sigstore==4.5.0",
#     "sigstore-models==0.0.6",
#     "sigstore-rekor-types==0.0.18",
#     "sse-starlette==3.4.11",
#     "starlette==1.6.0",
#     "timm==1.0.29",
#     "tokenizers==0.23.2",
#     "torch==2.14.0",
#     "tqdm==4.70.0",
#     "transformers==5.16.1",
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
def imports():
    import os, subprocess, json, random, re, time, os, gc, math, torch
    import marimo as mo
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import notebook_login,HfFileSystem
    from pathlib import Path
    import torch.nn.functional as F
    import textwrap
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    #uv pip install -U transformers trl torch torchvision accelerate bitsandbytes peft timm triton certifi marimo[recommended] mcp huggingface_hub --torch-backend=auto
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfFileSystem,
        LoraConfig,
        TaskType,
        get_peft_model,
        json,
        os,
        random,
        re,
        textwrap,
        time,
        torch,
    )


@app.function
def generate_batch(texts, teacher_model, tokenizer, torch, time, max_new_tokens=512):
    # Single generate() per batch. Static KV cache: cache_implementation="static".
    # NOTE: no fp8 KV path on transformers 5.16.1 -- StaticCache stores K/V in
    # key_states.dtype (model dtype -> bf16 = 1.0 GiB/seq @4096); QuantizedCache
    # backends (quanto/hqq) are not installed. Static-vs-dynamic measured: fp16/bf16
    # static cache beats fully dynamic by ~1 GiB/seq @4096 ctx.
    # Measured 128-doc A/B (2026-09-13, B=32 static cache, greedy 512-token): nf4
    # teacher 74.8 tok/s vs bf16 copy 92.6 tok/s -- bnb 4-bit dequant overhead makes
    # nf4 SLOWER than plain bf16 here; nf4 kept for VRAM headroom (16.5 vs 55 GB).
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cache_implementation="static",
        )
    _free, _tot = torch.cuda.mem_get_info()
    print(f"  [BATCH] gen done n={len(texts)} free={_free / 2**30:.1f} GiB")
    return outputs, inputs


@app.cell
def teacher(AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, torch):
    TEACHER_MODEL = "Qwen/Qwen3.8-27B"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=quant_config,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    print(f"Teacher model: {teacher_model.config.model_type}, {teacher_model.num_parameters():,} params")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")
    return teacher_model, tokenizer


@app.cell
def load_fp8():
    # FP8 teacher via in-process transformers 5.16.1: NOT viable (measured 2026-09-13).
    # Official Qwen/Qwen3.8-27B-FP8 loads, but the finegrained-fp8 path hits two hard problems:
    #   1) loader gap: gate_proj ships as plain nn.Linear with fp8 weights, no weight_scale_inv
    #      (manual FP8Linear repair works: 64 modules rebuilt)
    #   2) kernel gap: after repair, community finegrained-fp8 Triton kernel is ~1000x slower
    #      than nf4 (smoke: 148.5s for 16 tokens vs nf4 74.8 tok/s) on sm_120 -- unusable
    # The FP8 speed pass is therefore blocked in-process; NF4 remains the teacher.
    # Real fp8/nvfp4 speed requires a serving runtime (vLLM/SGLang) per plan Annex.

    return


@app.cell
def load_student(
    AutoModelForCausalLM,
    LoraConfig,
    TaskType,
    get_peft_model,
    torch,
):
    STUDENT_MODEL = "ornith-ai/Ornith-1.5-9B"
    # tokenizer loaded by vblA (same Qwen2Tokenizer)

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
def memory(torch):
    # Resident VRAM report (click-to-run; safe whether or not the student is loaded)
    _free, _tot = torch.cuda.mem_get_info()
    _student = globals().get("student_model")
    if _student is None:
        print(f"  [VRAM] free={_free / 2**30:.1f}/{_tot / 2**30:.1f} GiB | student: unloaded")
    else:
        _n_params = sum(p.numel() for p in _student.parameters())
        print(f"  [VRAM] free={_free / 2**30:.1f}/{_tot / 2**30:.1f} GiB | student resident: {_n_params / 1e9:.2f}B params")
    return


@app.cell
def unload_student(torch):
    # Explicit ~16.8 GiB reclaim when distill is not running (click-to-run; no-op safe).
    # Runtime deletion of another cell's global: globals().pop avoids the static
    # multiply-defined-name rule and matches the object at the moment you click Run.
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
        print(f"  [UNLOAD] freed {(_free1 - _free0) / 2**30:.1f} GiB -> free={_free1 / 2**30:.1f}/{_tot / 2**30:.1f} GiB")
    return


@app.cell
def gen_synthetic(
    HfFileSystem,
    json,
    os,
    random,
    re,
    teacher_model,
    textwrap,
    time,
    tokenizer,
    torch,
):
    """
    Generate synthetic QA data from HF bucket documents.json
    using the already-loaded teacher model.

    Resume-safe: re-run the cell to continue where it left off.
    State saved to data/training/gen_state.json after every checkpoint.

    Run after cell order: imports -> teacher -> (load_student) -> (memory/unload_student) -> gen_synthetic.
    All imports provided by the shared imports cell; batched generate lives in generate_helper (dynamic BATCH + static KV cache).
    Hardware: molab-class GPU (CUDA cuda:0) -- unload the student first to free ~16.8 GiB, then BATCH scales to live free VRAM (capped at 32).
    """

    # -- Config --
    # Production base preamble: located by the dedent call (no local import here; textwrap comes from the shared imports cell),he cell-body 4-space indent.
    BASE_PROMPT = textwrap.dedent("""\
    You are a Project Gorgon game assistant.

    Answer the user's question using the provided context.

    Rules:
    - Figure out what the user is really asking, even when the question is informal, and assemble the answer from the context \u2014 the exact answer need not be stated word-for-word in the documents.
    - Reason from the context: connect information across documents, compare and rank options, and draw conclusions that follow from the stated facts. Planning a path or choosing the most efficient option from stated values is expected and helpful.
    - Include relevant names, skills, levels, ingredients, and quantities when available.
    - If the user asks about recipes, include the recipe name, ingredients with quantities, required skill level, and the recipe's description text (e.g. dose counts, effects, results).
    - If multiple answers exist, list them.
    - If the context contains PARTIAL information for the question, answer with exactly what is present and explicitly state what is missing \u2014 do not refuse the whole question because one detail is absent.
    - Only say you do not know when the context contains nothing relevant: no facts, names, levels, or values that bear on the question.
    - NEVER fabricate: do not invent facts, names, values, recipes, XP numbers, formulas, or mechanics that are not present in the context. Arithmetic directly derived from stated context values (such as calculating the difference between two stated cumulative XP totals: Target Level cumulative XP minus Start Level cumulative XP) is grounded analysis, not fabrication. If a specific number is not stated or derivable from the context, say it is not stated rather than guessing.
    - NEVER cite sources that are not listed in the provided context.
    - When listing sources, only reference documents that actually contributed to your answer.
    """)

    # -- Config --
    BUCKET_PATH = "hf://buckets/Nubula/paddock/documents.json"
    TARGET = 12  # resting value; verification runs may temporarily raise it (e.g. 24) and restore after
    MAX_DOC_CHARS = 2048
    # Cadence == BATCH: mid-run progress logs + checkpoints + state saves actually fire
    # for every TARGET small enough to finish inside one session.
    BATCH_LOG = 8
    CHECKPOINT_EVERY = 8
    GEN_MAX_TOKENS = 512
    RNG = random.Random(42)
    OUT_DIR = "data/training"
    STATE_FILE = f"{OUT_DIR}/gen_state.json"
    OUT_TRAIN = f"{OUT_DIR}/synthetic_train.jsonl"
    OUT_EVAL = f"{OUT_DIR}/synthetic_eval.jsonl"

    # Lever 4 (VRAM-driven dynamic batch): BATCH recomputed from live free VRAM at every
    # batch boundary -- bigger batches ride the headroom freed by unloading the student
    # (~16.8 GiB). Anchors measured this turn:
    #   KV GiB/seq@4096 = 1.0 (64 layers x 4 KV heads x 256 head_dim, bf16 = 262144 B/token)
    #   6.0 GiB margin covers the activation spike beyond KV (peak-alloc rise B=8 -> B=32
    #   is only ~4 GiB of activations vs per-batch KV dominating at B>=8).
    # Cap 32: B=40 fit VRAM but measured no tok/s gain over B=32 -- the cap is a
    # decode-bandwidth/throughput plateau, NOT a VRAM limit (bf16 teacher at 54.7 GB
    # would compute the same formula -> same cap; nf4 does NOT buy batch size).
    # Why nf4 then: (1) student co-residency needs 16.5-vs-55 GB fit, (2) bnb 4-bit
    # measured SLOWER than bf16 (74.8 vs 92.6 tok/s @B=32/greedy-512tok, 128-doc A/B this
    # turn), so if we only ever ran generation a bf16 teacher would be the faster choice.
    PER_SEQ_KV_GIB = 1.0
    BATCH_MARGIN_GIB = 6.0
    BATCH_CAP = 32

    # Sampling targets (matched to document types)
    DOC_TARGETS = [
        ("item", 2000), ("recipe", 2000), ("ability", 1500),
        ("quest", 1000), ("npc", 500), ("wiki", 1000),
        ("skill", 300), ("summary", 200), ("curated", 200),
        ("effect", 200), ("leveling", 100), ("lorebook", 100),
        ("combatxp", 50), ("xptable", 50),
    ]

    # Lever 3: rotate question-type hints for diversity. Cycle by global doc index.
    QUESTION_HINTS = [
        "Focus on what the item or ability IS USED FOR (results, effects, purpose) \u2014 not just skill requirements or ingredients.",
        "Create a COMPARISON question the context can uniquely answer (e.g. which value is highest, which recipe yields more). Give a rich, complete answer: state both compared values fully with quantities/units, identify the winner, and explain why \u2014 include what the comparison is and the answer.",
        "Create a HOW-TO/STRATEGY question about leveling or completing, answerable from step-by-step info in the context (XP needed, activities, trainers, unlock order).",
        "Create a question about quantities, cadence, cost, or specific numeric values in the context (stack size, value, doses, durations, uses), and give the answer with surrounding context \u2014 the number alone is not an answer.",
        "Ask about identity/naming (internal names, keywords, categories, shell names).",
    ]

    PROMPT = (
        "Based on this data, in English, write one question a player might ask "
        "and its answer, using only the information given. Do not number the "
        "question; give exactly one question with no enumerations.\n\n"
        "Data:\n{document_text}\n\n"
        "{q_hint}\nAnswer:"
    )

    # -- Load docs from HF bucket --
    print(f"Loading documents from {BUCKET_PATH} ...")
    fs = HfFileSystem()
    with fs.open(BUCKET_PATH, "rb") as f:
        all_docs = json.loads(f.read())
    print(f"Loaded {len(all_docs)} documents")

    # Index by type
    by_type = {}
    for d in all_docs:
        t = d.get("type", "unknown")
        by_type.setdefault(t, []).append(d)

    # Sample target docs
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

    # Cap to TARGET
    if len(sampled) > TARGET:
        RNG.shuffle(sampled)
        sampled = sampled[:TARGET]

    print(f"Total docs to process: {len(sampled)} (target={TARGET})")

    # -- Resume support --
    # Identity sources: gen_state.json (doc ids) + rows already on disk.
    # Rows written before doc_id existed in metadata are matched by (doc_type, doc_name).
    processed_ids = set()
    _legacy_keys = set()
    existing_rows = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            processed_ids = set(state["processed_ids"])
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [WARN] {STATE_FILE} unreadable ({type(e).__name__}: {e}); rebuilding from scanned rows")
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
            return False  # non-dict / id-less entries fall through to the tolerant processing loop
        _name = d.get("metadata", {}).get("name", d["id"])
        return d["id"] in processed_ids or (d.get("type", "unknown"), _name) in _legacy_keys

    # Filter out processed docs
    to_process = [d for d in sampled if not _doc_skipped(d)]
    skipped = len(sampled) - len(to_process)
    print(f"To process now: {len(to_process)} docs ({skipped} skipped)")

    if not to_process:
        print("All done! Nothing new to generate.")
    else:
        # -- Generate --
        teacher_model.eval()
        print(f"Using tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")

        rows = []
        sub_batch = []  # (doc, doc_id, doc_name, doc_type, doc_text, prompt) tuples awaiting a batch generate
        texts = []  # (prefill-ahead) serialized chat prompts aligned 1:1 with sub_batch
        _flushed = 0  # len prefix of rows already appended to OUT_TRAIN this session
        new_processed = set(processed_ids)
        retry_pending = []  # docs this batch whose first-pass output was non-English; retried once below, cleared after each retry pass
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
            # (prefill-ahead) serialize the chat prompt NOW (CPU work), before the batch gate,
            # so the flush path only pays tokenization + generate.
            messages = [{"role": "user", "content": prompt}]
            _text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # Pre-fill "Question: " in the assistant turn so the model continues from there
            _text = _text + "Question: "  # model generates: <question text>\nAnswer: <answer text>
            texts.append(_text)
            sub_batch.append((doc, doc_id, doc_name, doc_type, doc_text, prompt))

            # -- Dynamic BATCH: recompute from live free VRAM at each batch boundary --
            # (student unload moves this; formula + constants anchored in -- Config -- above)
            _free_gib = torch.cuda.mem_get_info()[0] / 2**30
            BATCH = max(2, min(BATCH_CAP, int((_free_gib - BATCH_MARGIN_GIB) / PER_SEQ_KV_GIB)))
            if len(sub_batch) < BATCH and i + 1 < len(to_process):
                continue
            print(f"  [BATCH] free={_free_gib:.1f} GiB -> batch={BATCH}")
            # -- Batch generate --
            try:
                # (prefill-ahead) chat-template serialization happened per-doc above the gate;
                # the flush path only pays tokenization + generate here.
                outputs, inputs = generate_batch(
                    texts,
                    teacher_model,
                    tokenizer,
                    torch,
                    time,
                    max_new_tokens=GEN_MAX_TOKENS,
                )

                for _n, _sb in enumerate(sub_batch):
                    _doc, doc_id, doc_name, doc_type, doc_text, _prompt = _sb
                    response = tokenizer.decode(
                        outputs[_n][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()

                    # Output should be: "player question text\nAnswer: detailed answer"
                    parts = re.split(r"\n?\s*Answer:\s*", response, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        question = parts[0].strip().lstrip("Question: ").strip()
                        question = re.sub(r"^\s*\d+\.\s+", "", question)
                        answer = parts[1].strip()
                        # Clean up extraneous template tokens
                        answer = re.sub(r"\s*<\|im_end\|>.*", "", answer, flags=re.DOTALL).strip()
                        answer = re.sub(r"^\s*response\b[:\s]*", "", answer).strip()
                        if question.lower().startswith(("question", "data", "answer", "here is")):
                            print(f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} "
                          f"meta-question: {question[:80]!r}")
                            retry_pending.append(_sb)  # retried once below; permanent burn only after retry
                            continue
                        if len(answer) < 20 or len(question) < 15:
                            print(f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} "
                          f" ans_len={len(answer)} q_len={len(question)} \u2014 too thin for SFT")
                            retry_pending.append(_sb)  # hint #0 may draw a richer answer; burned only if retry also fails
                            continue
                        if any(ord(ch) > 0x2E7F for ch in question + answer):
                            print(f"  [SKIP-CJK] idx={_n} doc={doc_name[:30]!r} \u2014 non-English output; will retry with English-only prompt")
                            retry_pending.append(_sb)
                            continue
                        if "?" in answer or "Question:" in answer:
                            print(f"  [SKIP] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} \u2014 answer echoes a question")
                            retry_pending.append(_sb)  # retried once below; permanent burn only after retry
                            continue

                        # Row context = exactly what the teacher saw (doc_text is already clipped to MAX_DOC_CHARS at the top of the loop); no separate 1500-char slice.
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
                        print(f"  [FAIL] idx={_n} doc={_doc.get('metadata', {}).get('name', '')[:30]!r} split={len(parts)} raw={response[:250]}")
                        retry_pending.append(_sb)  # transient parse failure; retried once below
                # Retry pass: one more shot for transient first-pass failures. Greedy decoding makes
                # unchanged prompts deterministic, so a retry needs a different prompt shape:
                # the English directive is PREPENDED and the hint swaps to the use-focused index 0.
                if retry_pending:
                    texts2 = []
                    for _sb in retry_pending:
                        # English directive must PRECEDE the hint: appended suffixes do not
                        # perturb the greedy path (verified: suffix retry reproduced identical CJK output).
                        _en = "IMPORTANT: Write the question and answer in English only.\n\n"
                        messages2 = [{"role": "user", "content": _en + PROMPT.format(document_text=_sb[4], q_hint=QUESTION_HINTS[0])}]
                        text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        text2 = text2 + "Question: "
                        texts2.append(text2)
                    outputs2, inputs2 = generate_batch(
                        texts2,
                        teacher_model,
                        tokenizer,
                        torch,
                        time,
                        max_new_tokens=GEN_MAX_TOKENS,
                    )
                    for _n2, _sb in enumerate(retry_pending):
                        _doc, doc_id, doc_name, doc_type, doc_text, _prompt = _sb
                        response2 = tokenizer.decode(
                            outputs2[_n2][inputs2["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                        parts2 = re.split(r"\n?\s*Answer:\s*", response2, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts2) == 2:
                            question = parts2[0].strip().lstrip("Question: ").strip()
                            question = re.sub(r"^\s*\d+\.\s+", "", question)
                            answer = parts2[1].strip()
                            answer = re.sub(r"\s*<\|im_end\|>.*", "", answer, flags=re.DOTALL).strip()
                            answer = re.sub(r"^\s*response\b[:\s]*", "", answer).strip()
                            if any(ord(ch) > 0x2E7F for ch in question + answer):
                                print(f"  [FAIL-CJK] idx={_n2} doc={doc_name[:30]!r} \u2014 still non-English after retry; doc skipped permanently")
                                new_processed.add(doc_id)
                                continue
                            if "?" in answer or "Question:" in answer or len(answer) < 20 or len(question) < 15:
                                print(f"  [SKIP-R] idx={_n2} doc={doc_name[:30]!r} \u2014 guard hit after retry")
                                new_processed.add(doc_id)
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
                            print(f"  [OK-CJK-FIX] idx={_n2} doc={doc_name[:30]!r} Q={question[:50]}  A={answer[:50]}")
                        else:
                            print(f"  [FAIL] idx={_n2} doc={doc_name[:30]!r} split={len(parts2)} raw={response2[:150]}")
                        new_processed.add(doc_id)
                    retry_pending = []  # retry pass consumed the list; never carry-over between doc-loop batches
            except Exception as e:
                print(f"  [ERR] batch at #{i+1} {type(e).__name__}: {str(e)[:150]}")

            sub_batch = []
            texts = []

            # Log progress
            if (i + 1) % BATCH_LOG == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                avg = elapsed / (i + 1) if elapsed > 0 else 0
                print(f"  [{i+1}/{len(to_process)}] doc {doc_name[:30]} {len(rows)} rows ({rate*60:.1f} docs/min, {avg:.1f}s/doc)")

            # Checkpoint: flush only rows not yet on disk + save state
            if (i + 1) % CHECKPOINT_EVERY == 0:
                new_rows = rows[_flushed:]
                if new_rows:
                    os.makedirs(OUT_DIR, exist_ok=True)
                    # Append only unflushed rows (old code re-appended rows[-CHECKPOINT_EVERY:])
                    with open(OUT_TRAIN, "a") as f:
                        for r in new_rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    _flushed = len(rows)
                    # Save state
                    with open(STATE_FILE, "w") as f:
                        json.dump({"processed_ids": sorted(new_processed)}, f)
                    free_m, total_m = torch.cuda.mem_get_info()
                    print(f"  [CHECKPOINT] {len(new_processed)} docs done, +{len(new_rows)} rows appended, "
                          f"VRAM: {(total_m-free_m)/1e9:.1f}/{total_m/1e9:.1f} GB")

        # -- Final save --
        if rows:
            os.makedirs(OUT_DIR, exist_ok=True)

            # Flush rows not yet appended (_flushed is authoritative)
            remaining = rows[_flushed:]
            if remaining:
                with open(OUT_TRAIN, "a") as f:
                    for r in remaining:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Move a portion to eval set
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

            # contract: eval set = stratified 5% per doc_type, seeded by RNG; never a total head-slice
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

            # Save final state
            with open(STATE_FILE, "w") as f:
                json.dump({"processed_ids": sorted(new_processed)}, f)

            elapsed = time.time() - start
            print(f"\n{'='*50}")
            print(f"Generated {len(rows)} new QA pairs this session")
            print(f"Total dataset: {len(train_rows_final)} train + {len(eval_rows_final)} eval")
            print(f"Time: {elapsed:.0f}s  ({len(rows)/elapsed:.1f} rows/s)")
            print(f"To resume later: re-run cell (skips already done docs)")
        else:
            print("No rows generated (docs may be too short or parsing failed)")
    return


if __name__ == "__main__":
    app.run()
