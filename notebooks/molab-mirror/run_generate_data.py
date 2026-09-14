#!/usr/bin/env python
"""
Generate synthetic training data using local Qwen3.8-27B teacher model.

Loads the document corpus, samples documents by type, generates QA pairs
using the teacher model (Qwen3.8-27B, 4-bit), and merges with existing
training data. Outputs expanded JSONL files to data/training/.

Usage:
  uv run python run_generate_data.py [--target N] [--resume] [--dry-run]
"""

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_generate_data")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DOCUMENTS_PATH = ROOT / "data" / "documents.json"
OUTPUT_DIR = ROOT / "data" / "training"
TRAIN_PATH = OUTPUT_DIR / "expanded_train.jsonl"
EVAL_PATH = OUTPUT_DIR / "expanded_eval.jsonl"
EXISTING_TRAIN = OUTPUT_DIR / "unsloth_train.jsonl"
EXISTING_EVAL = OUTPUT_DIR / "unsloth_eval.jsonl"
STATE_PATH = OUTPUT_DIR / "generation_state.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MAX_TOKENS = 512
BATCH_SIZE = 1  # Process one doc at a time for VRAM stability

# Document sampling targets — match existing data types
DOC_SAMPLE_TARGETS: list[dict[str, Any]] = [
    {"type": "item", "count": 2000, "focus": "Uses, sources, where to find"},
    {"type": "recipe", "count": 2000, "focus": "Ingredients, skill level, quantities"},
    {"type": "ability", "count": 1500, "focus": "Damage, skill req, effects"},
    {"type": "quest", "count": 1000, "focus": "NPCs, reqs, rewards"},
    {"type": "npc", "count": 500, "focus": "Location, teaches what"},
    {"type": "wiki", "count": 1000, "focus": "General wiki content"},
    {"type": "skill", "count": 300, "focus": "Skill descriptions, leveling"},
    {"type": "summary", "count": 200, "focus": "Cross-document summaries"},
    {"type": "curated", "count": 200, "focus": "Entity descriptions, guides"},
    {"type": "effect", "count": 200, "focus": "Status effects and modifiers"},
    {"type": "leveling", "count": 100, "focus": "Skill leveling paths, XP"},
    {"type": "lorebook", "count": 100, "focus": "Lore and story content"},
    {"type": "combatxp", "count": 100, "focus": "Combat XP data"},
    {"type": "xptable", "count": 100, "focus": "XP tables"},
]

# ---------------------------------------------------------------------------
# Model globals (lazy-loaded, cached)
# ---------------------------------------------------------------------------
_teacher_model: Any = None
_teacher_tokenizer: Any = None
_teacher_pipe: Any = None


def _get_teacher() -> tuple[Any, Any]:
    """Load teacher (Qwen3.8-27B, 4-bit) and tokenizer, cached."""
    global _teacher_model, _teacher_tokenizer
    if _teacher_model is not None and _teacher_tokenizer is not None:
        return _teacher_model, _teacher_tokenizer

    log.info("Loading teacher: Qwen/Qwen3.8-27B (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.8-27B",
        quantization_config=quant_config,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    _teacher_model = model
    _teacher_tokenizer = tokenizer

    free, total = torch.cuda.mem_get_info()
    log.info(f"Teacher loaded. VRAM: {(total - free) / 1e9:.1f} GB used / {total / 1e9:.1f} GB total")
    return model, tokenizer


def _generate_with_teacher(
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = MAX_TOKENS,
) -> str:
    """Generate text using the teacher model.

    Uses tokenizer.apply_chat_template with a user-only message, then
    generates with the standard causal LM forward pass.
    """
    model, tokenizer = _get_teacher()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result


# ---------------------------------------------------------------------------
# Document cache
# ---------------------------------------------------------------------------
_docs_by_id: dict[str, dict] = {}
_docs_by_type: dict[str, list[dict]] = {}
_all_docs: list[dict] = []


def _load_documents() -> None:
    """Load and index data/documents.json into memory."""
    global _all_docs, _docs_by_id, _docs_by_type
    if _all_docs:
        return
    log.info("Loading documents from %s ...", DOCUMENTS_PATH)
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        _all_docs = json.load(f)
    log.info("Loaded %d documents", len(_all_docs))

    _docs_by_id = {d["id"]: d for d in _all_docs}
    _docs_by_type = {}
    for d in _all_docs:
        dtype = d.get("type", "unknown")
        _docs_by_type.setdefault(dtype, []).append(d)
    log.info("Indexed %d types", len(_docs_by_type))


def _get_docs_by_type(dtype: str) -> list[dict]:
    _load_documents()
    return _docs_by_type.get(dtype, [])


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
GENERATION_PROMPT = """You are generating training data for a Project Gorgon game assistant.

Given the following information, create one question that a player might ask,
followed by a detailed accurate answer based ONLY on the information provided.

Information:
{document_text}

Generate in this format:
Question: <question>
Answer: <answer>"""


def _sample_docs(dtype: str, count: int) -> list[dict]:
    """Sample documents of a given type, prioritizing substantive text."""
    pool = _get_docs_by_type(dtype)
    if not pool:
        return []
    pool.sort(key=lambda d: len(d.get("text", "")), reverse=True)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)
    # Take top docs but ensure diversity with shuffling
    return pool[:count * 2]


def _generate_synthetic_qa(
    doc: dict,
    dry_run: bool = False,
) -> Optional[dict]:
    """Generate one QA pair from a document using the teacher."""
    if dry_run:
        return None

    doc_text = doc.get("text", "")
    doc_name = doc.get("metadata", {}).get("name", doc["id"])
    doc_type = doc.get("type", "unknown")

    if len(doc_text) < 50:
        return None

    # Truncate if too long
    max_chars = 2048
    if len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars] + "\n[... truncated]"

    prompt = GENERATION_PROMPT.format(document_text=doc_text)

    try:
        response = _generate_with_teacher(prompt, temperature=0.7, max_new_tokens=MAX_TOKENS)
    except Exception as e:
        log.warning("  Generation failed for %s: %s", doc_name, e)
        return None

    # Parse Question: ... / Answer: ... from response
    q_match = re.search(r"Question:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    a_match = re.search(r"Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)

    if not q_match or not a_match:
        log.warning("  Could not parse QA from response for %s", doc_name)
        log.debug("  Raw response: %s", response[:200])
        return None

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()

    # Wrap in chat format matching existing training data
    user_content = f"You are a Project Gorgon game assistant.\n\nAnswer the user's question using the provided context.\n\nRules:\n- Figure out what the user is really asking, even when the question is informal, and assemble the answer from the context — the exact answer need not be stated word-for-word in the documents.\n- Reason from the context: connect information across documents, compare and rank options, and draw conclusions that follow from the stated facts. Planning a path or choosing the most efficient option from stated values is expected and helpful.\n- Include relevant names, skills, levels, ingredients, and quantities when available.\n- If the user asks about recipes, include the recipe name, ingredients with quantities, required skill levels, and any other useful details.\n- Be concise but thorough. Focus on actionable information.\n- If the context doesn't contain the answer, say so clearly.\n\nContext:\n{doc_text}\n\nQuestion: {question}"

    row = {
        "messages": [
            {"role": "user", "content": user_content.format(doc_text=doc_text[:1500], question=question)},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": "synthetic",
            "doc_type": doc_type,
            "doc_name": doc_name,
        },
    }
    return row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    target: int = 9000,
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict]:
    """Generate synthetic training rows from document corpus.

    Args:
        target: Number of rows to generate.
        dry_run: Plan only, no model calls.
        resume: Skip rows already generated (load from state file).

    Returns list of row dicts.
    """
    log.info("=== Synthetic Data Generation (target: %d rows) ===", target)
    _load_documents()
    all_rows: list[dict] = []
    processed_ids: set = set()

    # Resume support
    if resume and STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        processed_ids = set(state.get("processed_ids", []))
        train_path = TRAIN_PATH
        eval_path = EVAL_PATH
        if train_path.exists() and eval_path.exists():
            log.info("Resuming: %d already processed, loading existing output", len(processed_ids))
            with open(train_path) as f:
                for line in f:
                    all_rows.append(json.loads(line))
            with open(eval_path) as f:
                for line in f:
                    all_rows.append(json.loads(line))
            log.info("Loaded %d existing rows", len(all_rows))

    # --- Step 1: Sample documents by category ---
    sampled_docs: list[dict] = []
    for spec in DOC_SAMPLE_TARGETS:
        dtype = spec["type"]
        count = spec["count"]
        docs = _sample_docs(dtype, count)
        # Filter already processed
        if resume:
            docs = [d for d in docs if d["id"] not in processed_ids]
        sampled_docs.extend(docs[:count])
        log.info("  Sampled %d/%d '%s' docs (non-resume: %d)",
                 min(count, len(docs)), count, dtype,
                 len(docs))

    log.info("Total documents to process: %d", len(sampled_docs))

    if dry_run:
        log.info("[DRY RUN] Would process %d documents", len(sampled_docs))
        return all_rows

    if not sampled_docs:
        log.info("No new documents to process.")
        return all_rows

    # --- Step 2: Generate QA pairs ---
    # Load teacher lazily
    _get_teacher()

    start_time = time.time()
    for i, doc in enumerate(sampled_docs):
        doc_id = doc["id"]
        doc_name = doc.get("metadata", {}).get("name", doc_id)

        row = _generate_synthetic_qa(doc, dry_run=False)
        if row:
            all_rows.append(row)

        # Save progress periodically
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info("  [%d/%d] %s — %d rows (%.1f docs/min)",
                     i + 1, len(sampled_docs), doc_name, len(all_rows), rate * 60)

        # Save state every 100 docs
        if (i + 1) % 100 == 0:
            state = {
                "processed_ids": list(processed_ids.union({d["id"] for d in sampled_docs[:i + 1]})),
                "total_rows": len(all_rows),
                "last_doc_idx": i,
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATE_PATH, "w") as f:
                json.dump(state, f)

            # Also save intermediate output
            train, eval_set = _split_train_eval(all_rows)
            _write_jsonl(train, TRAIN_PATH)
            _write_jsonl(eval_set, EVAL_PATH)
            log.info("  [Checkpoint] Saved %d train + %d eval", len(train), len(eval_set))

    log.info("Generation complete: %d total rows (%.1f docs/min)",
             len(all_rows), (len(sampled_docs) / (time.time() - start_time) * 60))

    return all_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _split_train_eval(rows: list[dict], eval_ratio: float = 0.1, seed: int = RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    """Split rows into train/eval sets."""
    rng = random.Random(seed)
    synthetic_rows = rows[:]
    rng.shuffle(synthetic_rows)
    n_eval = max(1, int(len(synthetic_rows) * eval_ratio))
    eval_set = synthetic_rows[:n_eval]
    train = synthetic_rows[n_eval:]
    return train, eval_set


def _write_jsonl(rows: list[dict], path: Path, mode: str = "w") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows to %s", len(rows), path)


def _merge_with_existing(existing_train: Path, existing_eval: Path,
                         new_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge newly generated rows with existing training data."""
    existing: list[dict] = []
    if existing_train.exists():
        with open(existing_train) as f:
            for line in f:
                existing.append(json.loads(line))
        log.info("Loaded %d existing train rows", len(existing))

    existing_eval_rows: list[dict] = []
    if existing_eval.exists():
        with open(existing_eval) as f:
            for line in f:
                existing_eval_rows.append(json.loads(line))
        log.info("Loaded %d existing eval rows", len(existing_eval_rows))

    # Combine: existing + new rows, split train/eval
    all_rows = existing + new_rows + existing_eval_rows
    train, eval_set = _split_train_eval(all_rows)
    return train, eval_set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using local Qwen3.8-27B teacher"
    )
    parser.add_argument(
        "--target", type=int, default=9000,
        help="Target number of synthetic rows (default: 9000)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without generating"
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Don't merge with existing training data (output separate files)"
    )
    args = parser.parse_args()

    if args.dry_run:
        log.info("=== DRY RUN ===")

    # Check document corpus exists
    if not DOCUMENTS_PATH.exists():
        log.error("Document corpus not found at %s", DOCUMENTS_PATH)
        log.error("Run 'uv run pgrag build-documents' first")
        sys.exit(1)

    rows = generate_synthetic_data(
        target=args.target,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    if args.dry_run:
        log.info("=== DRY RUN complete ===")
        return

    if not rows:
        log.warning("No rows generated!")
        return

    if args.no_merge:
        # Output standalone expanded files
        train, eval_set = _split_train_eval(rows)
        _write_jsonl(train, TRAIN_PATH)
        _write_jsonl(eval_set, EVAL_PATH)
    else:
        # Merge with existing data
        train, eval_set = _merge_with_existing(EXISTING_TRAIN, EXISTING_EVAL, rows)
        _write_jsonl(train, ROOT / "data" / "training" / "merged_train.jsonl")
        _write_jsonl(eval_set, ROOT / "data" / "training" / "merged_eval.jsonl")

    log.info("=== Generation complete ===")
    log.info("Total rows in output: %d", len(rows))
    log.info("  Train: %d", len(train))
    log.info("  Eval:  %d", len(eval_set))

    # Sample
    if rows:
        log.info("Sample row:")
        log.info("  User: %s ...", rows[0]["messages"][0]["content"][:150])
        log.info("  Assistant: %s ...", rows[0]["messages"][1]["content"][:150])


if __name__ == "__main__":
    main()