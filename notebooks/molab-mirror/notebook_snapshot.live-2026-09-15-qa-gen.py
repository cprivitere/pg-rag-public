# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "huggingface-hub==1.31.0",
#     "kernels==0.16.1",
#     "kernels-data==0.16.1",
#     "marimo[mcp]>=0.24.0",
#     "mcp>=1",
#     "pydantic>=2",
#     "python-lsp-ruff==2.3.3",
#     "python-lsp-server==1.15.0",
#     "ruff==0.16.7",
#     "safetensors==0.8.0",
#     "sigstore==4.5.0",
#     "sigstore-models==0.0.6",
#     "sigstore-rekor-types==0.0.18",
#     "sse-starlette==3.4.11",
#     "starlette==1.6.0",
#     "tokenizers==0.23.2",
#     "torch==2.14.0",
#     "tqdm==4.70.0",
#     "transformers==5.17.0",
#     "urllib3==2.7.0",
#     "websockets==17.1",
#     "xxhash==4.0.1",
#     "yarl==1.24.5",
# ]
# ///


__generated_with = "0.24.0"

# %%
import subprocess as _subprocess

# ---- Environment repair (setup runs FIRST, before any torch-importing cell) ----
# molab sandboxes rotate; the image torch may be stale vs the driver
# (driver CUDA 13.2 here). Policy A: repair whenever the venv torch is not the
# driver-matched build; --torch-backend=auto on driver 13.2 resolves
# torch==2.14.0+cu132 (verified 2026-09-16 via uv dry-run on this fleet).
# Probe in a SUBPROCESS so the kernel never imports a broken torch (which
# would poison sys.modules for the whole session). kernels left UNBOUNDED:
# 0.17.x exists and hub builds verified (mamba-ssm torch214-cu130 and
# torch212-cu132 variants); einops added because the 0.17 mamba-ssm kernel
# imports it at load time.
_PROBE_CODE = "import torch; assert torch.cuda.is_available(); print(torch.__version__ + '_' + torch.version.cuda.replace('.', ''))"
_r = _subprocess.run(
    ["/tmp/uv-venv/bin/python", "-c", _PROBE_CODE],
    capture_output=True, text=True, timeout=120,
)
_out = (_r.stdout or "").strip()
if _r.returncode == 0 and (_out.startswith("2.14.0+cu132_") or _out == "2.14.0+cu132_130"):
    print("[ENV] torch 2.14.0+cu132 OK (alloc + reduce) - no repair needed")
elif _r.returncode == 0 and not _out:
    _why = "torch importable but probe printed nothing"
    print("[ENV] env mismatch (%s) -> repairing" % _why)
    _rr = _subprocess.run(_REPAIR_CMD, shell=True, capture_output=True, text=True, timeout=900)
    print("[ENV] repair rc=%s" % _rr.returncode)
else:
    _why = "torch not importable (rc=%s)" % _r.returncode if _r.returncode != 0 else "torch=%r" % _out
    print("[ENV] env mismatch (%s) -> repairing" % _why)
    _REPAIR_CMD = (
        "uv pip install -U "
        "transformers kernels marimo[recommended] huggingface-hub transformers torch einops "
        "--torch-backend=auto -p /tmp/uv-venv/bin/python"
    )
    _rr = _subprocess.run(_REPAIR_CMD, shell=True, capture_output=True, text=True, timeout=900)
    print("[ENV] repair rc=%s" % _rr.returncode)
    if _rr.returncode != 0:
        print("[ENV] repair stderr tail:", (_rr.stderr or "")[-300:])
    print("[ENV] NOTE: after repair, restart the session (UI) so no stale torch "
          "sits in this kernel's sys.modules")
print("[ENV] env-check complete")


# %%
import json, random, re, time, os, gc, torch, textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFileSystem

# Please leave this comment
#uv pip install -U "transformers kernels<0.17.0" marimo[recommended] huggingface-hub transformers torch --torch-backend=auto

# %%
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

# %%
_free, _tot = torch.cuda.mem_get_info()
print(f"  [VRAM] free={_free / 2**30:.1f}/{_tot / 2**30:.1f} GiB")


# %%
# Deliberate: unload touches teacher_model only via globals() so this cell does
# NOT become a reactive descendant of the teacher cell — unload must not fire
# implicit reloads of the model it is trying to clear.
unload = globals().get("teacher_model")
_free0, _tot = torch.cuda.mem_get_info()
torch.cuda.synchronize()
globals().pop("teacher_model", None)
del unload
gc.collect()
torch.cuda.empty_cache()
_free1, _ = torch.cuda.mem_get_info()
print(
    f"  [UNLOAD] freed {(_free1 - _free0) / 2**30:.1f} GiB -> free={_free1 / 2**30:.1f}/{_tot / 2**30:.1f} GiB"
)


# %%
import torch as _torch
from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTok

_teacher_name = "Qwen/Qwen3.8-27B"

_free0, _tot = _torch.cuda.mem_get_info()
print(f"[Teacher Load] free before: {_free0 / 2**30:.1f} GiB")

teacher_model = _AutoModel.from_pretrained(
    _teacher_name,
    dtype=_torch.bfloat16,
    attn_implementation="sdpa",
    device_map="cuda:0",
    use_kernels=True,
)
teacher_model.eval()
print(
    f"Teacher: {teacher_model.config.model_type}, "
    f"{teacher_model.num_parameters():,} params | "
    f"free after: {_torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB"
)

tokenizer = _AutoTok.from_pretrained(_teacher_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")


# %%
"\n    Generate synthetic QA data from HF bucket documents.json\n    using the already-loaded teacher model.\n\n    Resume-safe: re-run the cell to continue where it left off.\n    State saved to data/training/gen_state.json after every checkpoint.\n\n    Run after cell order: imports -> teacher -> (load_student) -> (memory/unload_student) -> gen_synthetic.\n    All imports provided by the shared imports cell; batched generate lives in generate_helper (dynamic BATCH + static KV cache).\n    Hardware: molab-class GPU (CUDA cuda:0) -- unload the student first to free ~16.8 GiB, then BATCH scales to live free VRAM (capped at 32).\n"

BASE_PROMPT = textwrap.dedent(
    "    You are a Project Gorgon game assistant.\n\n    Answer the user's question using the provided context.\n\n    Rules:\n    - Figure out what the user is really asking, even when the question is informal, and assemble the answer from the context — the exact answer need not be stated word-for-word in the documents.\n    - Reason from the context: connect information across documents, compare and rank options, and draw conclusions that follow from the stated facts. Planning a path or choosing the most efficient option from stated values is expected and helpful.\n    - Include relevant names, skills, levels, ingredients, and quantities when available.\n    - If the user asks about recipes, include the recipe name, ingredients with quantities, required skill level, and the recipe's description text (e.g. dose counts, effects, results).\n    - If multiple answers exist, list them.\n    - If the context contains PARTIAL information for the question, answer with exactly what is present and explicitly state what is missing — do not refuse the whole question because one detail is absent.\n    - Only say you do not know when the context contains nothing relevant: no facts, names, levels, or values that bear on the question.\n    - NEVER fabricate: do not invent facts, names, values, recipes, XP numbers, formulas, or mechanics that are not present in the context. Arithmetic directly derived from stated context values (such as calculating the difference between two stated cumulative XP totals: Target Level cumulative XP minus Start Level cumulative XP) is grounded analysis, not fabrication. If a specific number is not stated or derivable from the context, say it is not stated rather than guessing.\n    - NEVER cite sources that are not listed in the provided context.\n    - When listing sources, only reference documents that actually contributed to your answer.\n    "
)
BUCKET_PATH = "hf://buckets/Nubula/paddock/documents.json"
TARGET = 200
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
                    gen_out[_n][gen_in["input_ids"].shape[1] :], skip_special_tokens=True
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
                        gen_out2[_n2][gen_in2["input_ids"].shape[1] :], skip_special_tokens=True
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
        # Durable artifacts: push final split to the HF bucket so sandbox death
        # cannot erase the run (bucket write-verified earlier).
        try:
            fs2 = HfFileSystem()
            for src_path, dst_path in (
                (OUT_TRAIN, "unsloth_train.jsonl"),
                (OUT_EVAL, "unsloth_eval.jsonl"),
                (STATE_FILE, "gen_state.json"),
            ):
                if os.path.exists(src_path):
                    with (
                        open(src_path, "rb") as fsrc,
                        fs2.open(
                            "hf://buckets/Nubula/paddock/training/" + dst_path,
                            "wb",
                        ) as fdst,
                    ):
                        fdst.write(fsrc.read())
                    print(f"  [PUSH] {dst_path} -> hf://buckets/Nubula/paddock/training/")
                else:
                    print(f"  [PUSH SKIP] {src_path} missing")
        except Exception as push_err:
            print(f"  [PUSH FAIL] {type(push_err).__name__}: {str(push_err)[:150]} (local files intact)")

    else:
        print("No rows generated (docs may be too short or parsing failed)")