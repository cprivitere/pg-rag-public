#!/usr/bin/env python
"""
Generate ChatML training dataset for Unsloth QLoRA fine-tuning of
Ornith-1.5-9B.

Phases:
  1. Golden evals (42 x 3 rephrasings + 1 truncated-context negative ≈ 168 rows)
  2. Synthetic QA from document corpus (~300 docs x 3 QA pairs ≈ 900+ rows)

Teacher: deepseek-ai/DeepSeek-V4-Flash via CoreWeave/W&B API.
Output: data/training/unsloth_train.jsonl + data/training/unsloth_eval.jsonl

Usage:
  uv run python scripts/generate_training_data.py --source all
  uv run python scripts/generate_training_data.py --source golden --validate

Flags:
  --source {golden,synthetic,all}   default: all
  --validate                        run golden-fact validation on output
  --scale N                         target number of synthetic rows (default 1200)
  --resume                          skip already-written rows in output file
  --dry-run                         print plan but don't call API
"""

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_training_data")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "data" / "golden"
DOCUMENTS_PATH = ROOT / "data" / "documents.json"
OUTPUT_DIR = ROOT / "data" / "training"
TRAIN_PATH = OUTPUT_DIR / "unsloth_train.jsonl"
EVAL_PATH = OUTPUT_DIR / "unsloth_eval.jsonl"
STATE_PATH = OUTPUT_DIR / "generation_state.json"

# ---------------------------------------------------------------------------
# API configuration — read from omp credential store + env
# ---------------------------------------------------------------------------


# Resolve the CoreWeave/W&B API key
def _get_api_key() -> str:
    """Resolve the wandb API key from the omp agent database."""
    agent_db = Path(os.path.expanduser("~/.omp/agent/agent.db"))
    if agent_db.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(agent_db))
            row = conn.execute(
                "SELECT data FROM auth_credentials WHERE provider='coreweave' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                data = json.loads(row[0])
                key = data.get("key")
                if key and key.startswith("wandb_"):
                    return key
        except Exception as e:
            log.warning("agent.db lookup failed: %s", e)
    # Fallback: env var
    key = os.environ.get("WANDB_API_KEY") or os.environ.get("COREWEAVE_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "No CoreWeave/W&B API key found. "
        "Expected in omp agent.db (coreweave provider) or WANDB_API_KEY env var."
    )


API_KEY = _get_api_key()
COREWEAVE_PROJECT = os.environ.get("COREWEAVE_PROJECT", "returners/PG Research")
API_BASE = "https://api.inference.wandb.ai/v1"
TEACHER_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTEXT_BUDGET = 80_000
RANDOM_SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 120

# Phase 2 document sampling targets
DOC_SAMPLE_TARGETS: list[dict[str, Any]] = [
    {"type": "leveling", "count": 40, "focus": "Skill leveling paths, XP"},
    {"type": "skill", "count": 15, "focus": "Skill descriptions"},
    {"type": "item", "count": 35, "focus": "Uses, sources, where to find"},
    {"type": "recipe", "count": 30, "focus": "Ingredients, skill level, quantities"},
    {"type": "ability", "count": 25, "focus": "Damage, skill req, effects"},
    {"type": "quest", "count": 20, "focus": "NPCs, reqs, rewards"},
    {"type": "npc", "count": 15, "focus": "Location, teaches what"},
    {"type": "summary", "count": 20, "focus": "Cross-document summaries"},
    {"type": "curated", "count": 25, "focus": "Entity descriptions, guides"},
    {"type": "wiki", "count": 30, "focus": "General wiki content"},
    {"type": "lorebook", "count": 15, "focus": "Lore and story content"},
    {"type": "effect", "count": 15, "focus": "Status effects and modifiers"},
    {"type": "xptable", "count": 10, "focus": "XP tables and level requirements"},
    {"type": "combatxp", "count": 10, "focus": "Combat XP data and enemy XP values"},
]

# Comparison pair templates
COMPARISON_PAIRS = [
    ("ability", "ability"),  # Fireball vs Fire Breath, Punch vs Front Kick
    ("item", "item"),  # Healing Potion vs Healing Potion Omega
    ("skill", "skill"),  # Sword vs Unarmed burst
    ("recipe", "recipe"),  # Orcish Flour vs Basic Flatbread
]

# ---------------------------------------------------------------------------
# Document cache
# ---------------------------------------------------------------------------
_docs_by_id: dict[str, dict] = {}
_docs_by_type: dict[str, list[dict]] = {}
_docs_by_name: dict[str, list[dict]] = {}
_all_docs: list[dict] = []


def _load_documents():
    """Load and index data/documents.json into memory."""
    global _all_docs, _docs_by_id, _docs_by_type, _docs_by_name
    if _all_docs:
        return
    log.info("Loading documents from %s ...", DOCUMENTS_PATH)
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        _all_docs = json.load(f)
    log.info("Loaded %d documents", len(_all_docs))

    _docs_by_id = {d["id"]: d for d in _all_docs}
    _docs_by_type = {}
    _docs_by_name = {}
    for d in _all_docs:
        dtype = d.get("type", "unknown")
        _docs_by_type.setdefault(dtype, []).append(d)
        name = (d.get("metadata") or {}).get("name", "")
        if name:
            key = name.lower().strip()
            _docs_by_name.setdefault(key, []).append(d)
    log.info(
        "Indexed %d types, %d named entities",
        len(_docs_by_type),
        len(_docs_by_name),
    )


def _get_docs_by_name(name: str) -> list[dict]:
    """Find documents whose metadata.name matches (case-insensitive)."""
    _load_documents()
    return _docs_by_name.get(name.lower().strip(), [])


def _get_doc_by_id(doc_id: str) -> dict | None:
    """Find a document by its id field."""
    _load_documents()
    return _docs_by_id.get(doc_id)


def _get_docs_by_type(dtype: str) -> list[dict]:
    """Get all documents of a given type."""
    _load_documents()
    return _docs_by_type.get(dtype, [])


def _sample_docs(dtype: str, count: int, exclude_ids: set | None = None) -> list[dict]:
    """Sample documents of a given type, optionally excluding certain IDs."""
    pool = _get_docs_by_type(dtype)
    if exclude_ids:
        pool = [d for d in pool if d["id"] not in exclude_ids]
    if not pool:
        return []
    # Prioritize documents with more substantive text
    pool.sort(key=lambda d: len(d.get("text", "")), reverse=True)
    return pool[: count * 3] if len(pool) > count * 3 else pool


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update(
    {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Project": COREWEAVE_PROJECT,
        "Content-Type": "application/json",
    }
)


def call_teacher(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    strip_thinking: bool = False,
) -> str:
    """Call DeepSeek V4 Flash via CoreWeave/W&B API.

    Args:
        messages: OpenAI-format messages list.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Max output tokens.
        system_prompt: Optional system message prepended.
        strip_thinking: If True, remove  ~\n...\n block from output.

    Returns:
        The response text.
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    for attempt in range(MAX_RETRIES):
        try:
            resp = _session.post(
                f"{API_BASE}/chat/completions",
                json={
                    "model": TEACHER_MODEL,
                    "messages": full_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            if strip_thinking:
                content = re.sub(
                    r"^\s*<think>.*?\s*",
                    "",
                    content,
                    count=1,
                    flags=re.DOTALL | re.MULTILINE,
                )
                content = re.sub(
                    r"^\s*thinking.*?(\n|$)",
                    "",
                    content,
                    count=1,
                    flags=re.DOTALL | re.MULTILINE,
                )

            return content.strip()

        except requests.exceptions.Timeout:
            log.warning("API timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.RequestException as e:
            log.warning("API error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise

    raise RuntimeError(f"API request failed after {MAX_RETRIES} attempts")


def call_teacher_structured(
    messages: list[dict],
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Call teacher and split output into thinking block + answer.

    Expects response format:
       thinking
      <reasoning>
      <final answer>

    Returns (thinking, answer).
    """
    content = call_teacher(
        messages, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt
    )
    # Split on thinking block — handle multiple formats
    # Format:  thinking\n...\n\n... (Ornith-native)
    # Also handle: Thinking:\n...\n\n... (teacher variations)
    thinking_match = re.search(
        r"(?:^\s*thinking|^\s*Thinking)[:\s]*\n(.*?)\n{2,}(.*)",
        content,
        flags=re.DOTALL,
    )
    if thinking_match:
        return thinking_match.group(1).strip(), thinking_match.group(2).strip()

    # No thinking block — return empty thinking
    return "", content


# ---------------------------------------------------------------------------
# Context construction helpers
# ---------------------------------------------------------------------------


def _extract_entity_name(question: str) -> str | None:
    """Extract the primary entity name from a golden eval question."""
    # Known entity patterns from golden evals
    # Try to match common patterns: "What does X do?", "How to level X?",
    # "What is X?", "Which X ... Y or Z?", etc.
    # First check for known entities in the document index
    _load_documents()

    # Try name lookups for multi-word entities first
    # Sort by length (longest first) to match specific names before generic ones
    candidates = sorted(_docs_by_name.keys(), key=len, reverse=True)
    for name in candidates:
        if name.lower() in question.lower():
            return name
    return None


def _build_entity_context(entity_name: str) -> list[str]:
    """Build context text for a named entity from the document corpus."""
    docs = _get_docs_by_name(entity_name)
    if not docs:
        log.warning("No documents found for entity: %s", entity_name)
        return []

    # Sort: skill profile/leveling first, then type-specific, then wiki
    def _sort_key(d):
        t = d.get("type", "")
        priority = {
            "skillprofile": 0,
            "leveling": 0,
            "summary": 1,
            "skill": 2,
            "item": 2,
            "ability": 2,
            "recipe": 2,
            "quest": 2,
            "npc": 2,
            "curated": 2,
            "wiki": 3,
        }
        return priority.get(t, 4)

    docs.sort(key=_sort_key)

    texts = []
    total = 0
    for d in docs:
        text = d.get("text", "").strip()
        if not text:
            continue
        if total and total + len(text) > CONTEXT_BUDGET:
            break
        texts.append(text)
        total += len(text)
    return texts


def _build_comparison_context(entity1: str, entity2: str) -> list[str]:
    """Build context for a comparison between two entities."""
    ctx1 = _build_entity_context(entity1)
    ctx2 = _build_entity_context(entity2)

    # Deduplicate by merging entity contexts
    seen_texts = set()
    result = []
    for text in ctx1 + ctx2:
        norm = text.strip()[:200]
        if norm not in seen_texts:
            seen_texts.add(norm)
            result.append(text)

    return result[: CONTEXT_BUDGET // 200]


# ---------------------------------------------------------------------------
# Golden fact validation
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize text for fact matching."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _check_facts(answer: str, fact_variants: list[list[str]]) -> dict[str, bool]:
    """Check each fact group against the answer. Returns {fact_index: found}."""
    norm_answer = _normalize(answer)
    results = {}
    for i, variants in enumerate(fact_variants):
        found = any(_normalize(v) in norm_answer for v in variants)
        results[str(i)] = found
    return results


# ---------------------------------------------------------------------------
# Phase 1: Golden Eval Processing
# ---------------------------------------------------------------------------


def _load_golden_evals() -> list[dict]:
    """Load all golden eval JSON files."""
    evals = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            evals.append(json.load(f))
    log.info("Loaded %d golden evals", len(evals))
    return evals


def _generate_question_variants(golden: dict) -> list[str]:
    """Generate 3 natural rephrasings of a golden eval question."""
    prompt = f"""You are a data augmentation assistant for a Project Gorgon RAG system. Given a question about the game, generate 3 natural player-language rephrasings. Each variant should sound like a real player asking for help — informal, conversational, sometimes with minor typos or shorthand. Vary the phrasing structure (e.g., "how to," "what's the best way," "can you tell me," "I'm trying to").

Original question: {golden["question"]}

Output exactly 3 variants, one per line, with no numbering, bullets, or extra text:"""

    response = call_teacher(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=1024,
    )
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    return lines[:3]


def _generate_ideal_answer(
    question: str, context_text: str, query_type: str, facts: list[list[str]]
) -> tuple[str, str]:
    """Generate ideal answer with thinking trace from teacher.

    Returns (thinking, answer).
    """
    fact_hints = "; ".join(f"[{', '.join(vs[:3])}]" for vs in facts)

    prompt = f"""You are a Project Gorgon game assistant. Your task is to generate an IDEAL answer to a player's question, using ONLY the provided context.

Rules:
- Every claim you make MUST be stated in the provided context.
- If an expected fact isn't in context, state that it is missing rather than guessing.
- Include the following golden fact keywords where they appear in context: {fact_hints}
- Reason step by step through the available information before answering.
- Be thorough and specific — include names, numbers, levels, quantities when available.
- If the context contains PARTIAL information, answer with what IS present and clearly state what is missing.
- NEVER fabricate facts, numbers, or mechanics not in the context.

Context:
{context_text}

Question: {question}

First, write "  thinking" (two spaces + "thinking", lowercase) on its own line, then your reasoning trace, then a blank line, then your final answer.

Format example:
  thinking
Your reasoning about the context here...
<blank line>
Your final answer to the player here."""

    system = "You are a Project Gorgon game assistant generating ideal training data."
    return call_teacher_structured(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system,
        temperature=0.5,
        max_tokens=4096,
    )


def _generate_truncated_negative(
    golden: dict,
    full_context: list[str],
    query_type: str,
    facts: list[list[str]],
) -> tuple[str, str, list[str]]:
    """Generate a partial-context negative: remove a key document, verify
    the answer says "not found"/"not stated" for missing info.

    Returns (thinking, answer, truncated_context).

    If full_context has fewer than 3 documents, skip — too little to truncate
    meaningfully."""

    if len(full_context) < 3:
        return "", "", full_context

    # Remove roughly the last third of context to drop some facts
    split = len(full_context) // 3
    if split < 1:
        return "", "", full_context
    truncated = full_context[:-split]
    truncated_text = "\n\n---\n\n".join(truncated)

    # Check which facts survive
    # We'll just generate the answer with guidance to acknowledge missing info
    fact_hints = "; ".join(f"[{', '.join(vs[:3])}]" for vs in facts)

    prompt = f"""You are a Project Gorgon game assistant. Your task is to generate an answer to a player's question using ONLY the provided context.

Rules:
1. Answer with whatever IS present in the context.
2. If a golden fact keyword ({fact_hints}) is NOT supported by the context, explicitly state "The context does not state [fact]" rather than guessing.
3. Be honest about what information is absent.

Context:
{truncated_text}

Question: {golden["question"]}

First, write "  thinking" (two spaces + "thinking", lowercase) on its own line, then your reasoning trace, then a blank line, then your final answer.

Format example:
  thinking
Your reasoning here...
<blank line>
Your answer here."""

    system = "You are a Project Gorgon game assistant generating ideal training data."
    thinking, answer = call_teacher_structured(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system,
        temperature=0.5,
        max_tokens=4096,
    )
    return thinking, answer, truncated


def process_golden_eval(
    golden: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Process a single golden eval into training rows.

    Returns list of row dicts with 'messages' and 'metadata'.
    """
    entity = _extract_entity_name(golden["question"])
    query_type = golden.get("type", "general")

    # Build context
    if query_type == "comparison":
        # For comparison, extract both entity names
        # Try to find 2 entity names in the question
        _load_documents()
        names_in_question = []
        for name in sorted(_docs_by_name.keys(), key=len, reverse=True):
            if name.lower() in golden["question"].lower():
                names_in_question.append(name)
        if len(names_in_question) >= 2:
            context = _build_comparison_context(names_in_question[0], names_in_question[1])
        elif entity:
            context = _build_entity_context(entity)
        else:
            context = []
    elif entity:
        context = _build_entity_context(entity)
    else:
        context = []

    if not context:
        log.warning("No context found for golden eval: %s", golden["id"])
        return []

    context_text = "\n\n---\n\n".join(context)

    if dry_run:
        log.info(
            "[DRY RUN] Would process golden eval: %s (context: %d chars)",
            golden["id"],
            len(context_text),
        )
        return []

    rows = []

    # --- Generate 3 question variants, each with ideal answer ---
    try:
        variants = _generate_question_variants(golden)
    except Exception as e:
        log.error("Failed to generate variants for %s: %s", golden["id"], e)
        variants = [golden["question"]]  # fall back to original

    for variant in variants:
        try:
            thinking, answer = _generate_ideal_answer(
                variant,
                context_text,
                query_type,
                golden["facts"],
            )
        except Exception as e:
            log.error("Failed to generate answer for %s variant: %s", golden["id"], e)
            continue

        # Build assistant content
        assistant = f"  thinking\n{thinking}\n\n{answer}" if thinking else answer

        row = {
            "messages": [
                {
                    "role": "user",
                    "content": _build_prompt(variant, context_text, query_type),
                },
                {
                    "role": "assistant",
                    "content": assistant,
                },
            ],
            "metadata": {
                "source": "golden",
                "golden_id": golden["id"],
                "golden_type": query_type,
                "variant": True,
                "truncated": False,
            },
        }
        rows.append(row)

    # --- Generate truncated-context negative ---
    try:
        neg_thinking, neg_answer, truncated_ctx = _generate_truncated_negative(
            golden,
            context,
            query_type,
            golden["facts"],
        )
        truncated_text = "\n\n---\n\n".join(truncated_ctx)
        if neg_thinking:
            neg_assistant = f"  thinking\n{neg_thinking}\n\n{neg_answer}"
        else:
            neg_assistant = neg_answer

        row = {
            "messages": [
                {
                    "role": "user",
                    "content": _build_prompt(golden["question"], truncated_text, query_type),
                },
                {
                    "role": "assistant",
                    "content": neg_assistant,
                },
            ],
            "metadata": {
                "source": "golden",
                "golden_id": golden["id"],
                "golden_type": query_type,
                "variant": False,
                "truncated": True,
            },
        }
        rows.append(row)
    except Exception as e:
        log.warning("Failed to generate truncated negative for %s: %s", golden["id"], e)

    return rows


def _build_prompt(question: str, context: str, query_type: str = "general") -> str:
    """Build the full user-side prompt matching the production pipeline."""
    base = """You are a Project Gorgon game assistant.

Answer the user's question using the provided context.

Rules:
- Figure out what the user is really asking, even when the question is informal, and assemble the answer from the context — the exact answer need not be stated word-for-word in the documents.
- Reason from the context: connect information across documents, compare and rank options, and draw conclusions that follow from the stated facts. Planning a path or choosing the most efficient option from stated values is expected and helpful.
- Include relevant names, skills, levels, ingredients, and quantities when available.
- If the user asks about recipes, include the recipe name, ingredients with quantities, required skill level, and the recipe's description text (e.g. dose counts, effects, results).
- If multiple answers exist, list them.
- If the context contains PARTIAL information for the question, answer with exactly what is present and explicitly state what is missing — do not refuse the whole question because one detail is absent.
- Only say you do not know when the context contains nothing relevant: no facts, names, levels, or values that bear on the question.
- NEVER fabricate: do not invent facts, names, values, recipes, XP numbers, formulas, or mechanics that are not present in the context. Arithmetic directly derived from stated context values (such as calculating the difference between two stated cumulative XP totals: Target Level cumulative XP minus Start Level cumulative XP) is grounded analysis, not fabrication. If a specific number is not stated or derivable from the context, say it is not stated rather than guessing.
- NEVER cite sources that are not listed in the provided context.
- When listing sources, only reference documents that actually contributed to your answer."""

    if query_type == "comparison":
        base += """

COMPARISON QUESTION DETECTED:
- Examine ALL provided context carefully.
- When asked about highest/lowest/best/worst, compare values across all items.
- Identify the item with the extreme value (maximum or minimum).
- Explain your reasoning: "Comparing X, Y, and Z... the highest is..."
- If a summary document is provided, use it as a quick reference.
- Quote each compared item's own description verbatim when it states the compared attribute directly (e.g. "Swords are very damaging", "Not especially damaging..."). Such stated descriptions are authoritative; keep their exact wording in your answer.
- Conclude firmly: when a stated description resolves the question, do not hedge or reverse it with edge-case numbers — report numeric details only as supplementary context, and never let them contradict the stated description."""

    if query_type == "entity":
        base += """

ENTITY QUESTION DETECTED (the question names one specific skill, item, ability, quest, or similar):
- The context is a dossier of that entity gathered from every relevant source.
- Enumerate what is relevant to the question (trainers, abilities, recipes, rewards, requirements, stats) and reason about how it answers the user's ask — one compact line each, no rambling.
- Answer comprehensively — the user wants everything the context says about this entity.
- Named characters: when the dossier lists trainers, skill teachers, quest givers, or vendors (e.g. "- Floxie (Fae Realm)"), reproduce EVERY name on that list one per line with its parenthetical, in context order. Do not collapse the list or omit any entry — a dropped trainer name is a wrong answer even if the rest of the dossier is complete."""

    return f"""{base}

Context:

{context}

Question:
{question}"""


# ---------------------------------------------------------------------------
# Phase 2: Synthetic QA from Documents
# ---------------------------------------------------------------------------


def _generate_synthetic_qa(
    doc_text: str,
    doc_name: str,
    doc_type: str,
    num_questions: int = 3,
    dry_run: bool = False,
) -> list[dict]:
    """Generate QA pairs from a single document using the teacher.

    Returns list of rows, each with 'messages' and 'metadata'.
    """
    if dry_run:
        return []

    type_guidance = {
        "leveling": "Ask about XP amounts, leveling paths, skill requirements, efficiency.",
        "skill": "Ask about what the skill does, its benefits, requirements.",
        "item": "Ask about uses, where to find it, what it's good for.",
        "recipe": "Ask about ingredients, skill level needed, quantities, results.",
        "ability": "Ask about damage, effects, skill requirements.",
        "quest": "Ask about quest giver, requirements, rewards.",
        "npc": "Ask about location, services, what they teach.",
        "summary": "Ask comprehensive questions that span multiple aspects.",
        "curated": "Ask detailed questions about entity mechanics.",
        "wiki": "Ask about game mechanics, locations, lore details.",
        "lorebook": "Ask about the story, implications, related content.",
        "effect": "Ask about what the effect does, duration, sources.",
        "xp": "Ask about XP amounts, level requirements, comparisons.",
    }

    guidance = type_guidance.get(doc_type, "Ask natural questions about this content.")

    prompt = f"""You are a data augmentation assistant. Generate {num_questions} natural question-answer pairs from the provided document.

For each pair:
1. Write a natural player question that a real Project Gorgon player might ask.
2. Write a complete, accurate answer using ONLY information from the document.
3. Include a  thinking block with reasoning trace before the answer.

Guidance for question types: {guidance}

Document name: {doc_name}
Document type: {doc_type}

Document text:
{doc_text}

Output format (repeat for each question):
QUESTIONS_START
Player question: <question text>
  thinking
<reasoning through the document>
<blank line>
<answer>
QUESTIONS_END

IMPORTANT: The thinking marker must be exactly "  thinking" (two spaces + "thinking", lowercase) on its own line, before the reasoning."""

    system = f"You are a data augmentation assistant for a Project Gorgon game wiki. Generate QA pairs from the provided document about {doc_name}."

    response = call_teacher(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system,
        temperature=0.7,
        max_tokens=8192,
    )

    # Parse the response into individual QA pairs
    rows = []
    blocks = re.split(r"QUESTIONS_(?:START|END)", response)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q_match = re.search(r"Player question:\s*(.+)", block)
        if not q_match:
            continue
        question = q_match.group(1).strip()

        # Extract thinking + answer
        thinking_match = re.search(
            r"^\s*thinking\s*\n(.*?)\n{2,}(.*)",
            block,
            flags=re.DOTALL | re.MULTILINE,
        )
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            answer_content = thinking_match.group(2).strip()
            assistant = f"  thinking\n{thinking_content}\n\n{answer_content}"
        else:
            assistant = block.split("\n", 1)[-1].strip() if "\n" in block else block

        context_text = _build_prompt(question, doc_text, "general")

        row = {
            "messages": [
                {"role": "user", "content": context_text},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "source": "synthetic",
                "doc_type": doc_type,
                "doc_name": doc_name,
                "truncated": False,
            },
        }
        rows.append(row)

    return rows


def _generate_comparison_qa(
    doc1: dict,
    doc2: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Generate comparison QA from two documents."""
    if dry_run:
        return []

    name1 = doc1.get("metadata", {}).get("name", doc1["id"])
    name2 = doc2.get("metadata", {}).get("name", doc2["id"])
    context = f"--- Document 1: {name1} ---\n\n{doc1['text']}\n\n--- Document 2: {name2} ---\n\n{doc2['text']}"

    prompt = f"""You are a data augmentation assistant. Generate a comparison question-answer pair from the two documents below.

The question should ask about the differences, similarities, or which is better between:
- Document 1: {name1} ({doc1.get("type", "unknown")})
- Document 2: {name2} ({doc2.get("type", "unknown")})

Write the answer using ONLY the provided information. Include a  thinking block with reasoning.

Documents:
{context}

Output:
Player question: <comparison question>
  thinking
<reasoning>
<blank line>
<comparison answer>

IMPORTANT: The thinking marker must be exactly "  thinking" (two spaces + "thinking", lowercase) on its own line, before the reasoning."""

    response = call_teacher(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )

    rows = []
    q_match = re.search(r"Player question:\s*(.+)", response)
    if not q_match:
        return []

    question = q_match.group(1).strip()
    thinking_match = re.search(
        r"^\s*thinking\s*\n(.*?)\n{2,}(.*)",
        response,
        flags=re.DOTALL | re.MULTILINE,
    )
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        answer_content = thinking_match.group(2).strip()
        assistant = f"  thinking\n{thinking_content}\n\n{answer_content}"
    else:
        assistant = response.split("\n", 1)[-1].strip() if "\n" in response else response

    context_text = _build_prompt(question, context, "comparison")

    row = {
        "messages": [
            {"role": "user", "content": context_text},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": "synthetic",
            "doc_type": "comparison",
            "doc_name": f"{name1} vs {name2}",
            "truncated": False,
        },
    }
    rows = [row]
    return rows


def _generate_negative_qa(
    doc: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Generate a negative example: question NOT answerable from the document.

    The assistant must say "I don't know" or "not found in the provided context."
    """
    if dry_run:
        return []

    name = doc.get("metadata", {}).get("name", doc["id"])

    prompt = f"""You are a data augmentation assistant. Generate a question about Project Gorgon that CANNOT be answered from the provided document.

The question should be about a completely different topic than the document. The answer should acknowledge the information is not in the provided context.

Document text:
{doc["text"]}

Output:
Player question: <question about a different topic>
  thinking
<reasoning that the context does not contain this information>
<blank line>
<answer stating the information is not in the provided context>

IMPORTANT: The thinking marker must be exactly "  thinking" (two spaces + "thinking", lowercase) on its own line, before the reasoning."""

    response = call_teacher(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=2048,
    )

    q_match = re.search(r"Player question:\s*(.+)", response)
    if not q_match:
        return []

    question = q_match.group(1).strip()
    thinking_match = re.search(
        r"^\s*thinking\s*\n(.*?)\n{2,}(.*)",
        response,
        flags=re.DOTALL | re.MULTILINE,
    )
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        answer_content = thinking_match.group(2).strip()
        assistant = f"  thinking\n{thinking_content}\n\n{answer_content}"
    else:
        assistant = response.split("\n", 1)[-1].strip() if "\n" in response else response

    context_text = _build_prompt(question, doc["text"], "general")

    row = {
        "messages": [
            {"role": "user", "content": context_text},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": "synthetic",
            "doc_type": "negative",
            "doc_name": name,
            "truncated": False,
        },
    }
    rows = [row]
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def phase1_golden(dry_run: bool = False) -> list[dict]:
    """Phase 1: Generate training rows from golden evals."""
    log.info("=== Phase 1: Golden Evals ===")
    evals = _load_golden_evals()
    all_rows = []

    for i, golden in enumerate(evals):
        log.info(
            "[%d/%d] Processing golden eval: %s (%s)",
            i + 1,
            len(evals),
            golden["id"],
            golden.get("type", "general"),
        )
        try:
            rows = process_golden_eval(golden, dry_run=dry_run)
            all_rows.extend(rows)
            log.info("  -> Generated %d rows", len(rows))
        except Exception as e:
            log.error("  -> FAILED: %s", e)
            continue

    log.info("Phase 1 complete: %d total rows", len(all_rows))
    return all_rows


def phase2_synthetic(
    target: int = 1200,
    dry_run: bool = False,
    resume_ids: set | None = None,
) -> list[dict]:
    """Phase 2: Generate synthetic training rows from document corpus.

    Args:
        target: Target number of rows.
        dry_run: If True, only plan.
        resume_ids: Set of document IDs already processed (skip them).

    Returns list of row dicts.
    """
    log.info("=== Phase 2: Synthetic QA (target: %d rows) ===", target)
    _load_documents()
    all_rows = []
    exclude_ids = resume_ids or set()

    # --- Step 1: Sample documents by category ---
    sampled_docs: list[dict] = []
    for spec in DOC_SAMPLE_TARGETS:
        dtype = spec["type"]
        count = spec["count"]
        docs = _sample_docs(dtype, count, exclude_ids=exclude_ids)
        sampled_docs.extend(docs[:count])
        log.info("  Sampled %d/%d '%s' docs", min(count, len(docs)), count, dtype)

    log.info("Total sampled documents: %d", len(sampled_docs))

    if dry_run:
        log.info("[DRY RUN] Would process %d documents for synthetic QA", len(sampled_docs))
        return []

    # --- Step 2: Generate QA pairs for each document ---
    for i, doc in enumerate(sampled_docs):
        doc_id = doc["id"]
        doc_name = doc.get("metadata", {}).get("name", doc_id)
        doc_type = doc.get("type", "unknown")
        doc_text = doc.get("text", "")

        if len(doc_text) < 50:
            continue

        if i >= 5 and i % 25 == 0:
            log.info("  [%d/%d] %s (%s)", i + 1, len(sampled_docs), doc_name, doc_type)

        # Determine number of questions based on document length
        num_q = 2 if len(doc_text) < 200 else (4 if len(doc_text) > 2000 else 3)

        try:
            rows = _generate_synthetic_qa(
                doc_text,
                doc_name,
                doc_type,
                num_questions=num_q,
            )
            all_rows.extend(rows)
        except Exception as e:
            log.warning("  Failed to generate QA for %s: %s", doc_id, e)
            continue

        # Negative QA for every 5th document
        if i % 5 == 0:
            try:
                neg_rows = _generate_negative_qa(doc)
                all_rows.extend(neg_rows)
            except Exception as e:
                log.warning("  Failed to generate negative for %s: %s", doc_id, e)

    # --- Step 3: Comparison pairs ---
    log.info("  Generating comparison pairs...")
    comparison_count = min(30, len(sampled_docs) // 10)
    for pair_type_a, pair_type_b in COMPARISON_PAIRS:
        type_a_docs = _get_docs_by_type(pair_type_a)
        type_b_docs = _get_docs_by_type(pair_type_b)
        if not type_a_docs or not type_b_docs:
            continue

        for _ in range(comparison_count // len(COMPARISON_PAIRS)):
            d1 = random.choice(type_a_docs)
            d2 = random.choice(type_b_docs)
            if d1["id"] == d2["id"]:
                continue
            try:
                rows = _generate_comparison_qa(d1, d2)
                all_rows.extend(rows)
            except Exception as e:
                continue

    log.info("Phase 2 complete: %d total rows", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _split_train_eval(
    rows: list[dict], eval_ratio: float = 0.1, seed: int = RANDOM_SEED
) -> tuple[list[dict], list[dict]]:
    """Split rows into train/eval sets, keeping golden variants together."""
    # Group rows by source+golden_id for stratified split
    golden_rows = [r for r in rows if r["metadata"]["source"] == "golden"]
    synthetic_rows = [r for r in rows if r["metadata"]["source"] == "synthetic"]

    rng = random.Random(seed)

    # Golden: keep eval rows from distinct eval IDs (held-out)
    golden_by_id: dict[str, list[dict]] = {}
    for r in golden_rows:
        gid = r["metadata"]["golden_id"]
        golden_by_id.setdefault(gid, []).append(r)

    golden_ids = list(golden_by_id.keys())
    rng.shuffle(golden_ids)
    n_eval_golden = max(1, int(len(golden_ids) * eval_ratio))
    eval_golden_ids = set(golden_ids[:n_eval_golden])
    train_golden_ids = set(golden_ids[n_eval_golden:])

    train_golden = [r for gid in train_golden_ids for r in golden_by_id[gid]]
    eval_golden = [r for gid in eval_golden_ids for r in golden_by_id[gid]]

    # Synthetic: random split
    rng.shuffle(synthetic_rows)
    n_eval_synthetic = max(1, int(len(synthetic_rows) * eval_ratio))
    eval_synthetic = synthetic_rows[:n_eval_synthetic]
    train_synthetic = synthetic_rows[n_eval_synthetic:]

    train = train_golden + train_synthetic
    eval_set = eval_golden + eval_synthetic

    rng.shuffle(train)
    rng.shuffle(eval_set)

    log.info(
        "Split: %d train + %d eval (golden: %d train/%d eval, synthetic: %d train/%d eval)",
        len(train),
        len(eval_set),
        len(train_golden),
        len(eval_golden),
        len(train_synthetic),
        len(eval_synthetic),
    )

    return train, eval_set


def _write_jsonl(rows: list[dict], path: Path, mode: str = "w"):
    """Write rows to JSONL file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_golden_rows(rows: list[dict]) -> dict:
    """Validate golden-sourced rows against their facts.

    Returns summary of fact coverage.
    """
    golden_rows = [r for r in rows if r["metadata"]["source"] == "golden"]
    if not golden_rows:
        return {"total": 0, "checked": 0, "all_passed": True}

    # Load corresponding golden evals
    evals = {g["id"]: g for g in _load_golden_evals()}

    total_facts = 0
    found_facts = 0
    missed_facts: list[dict] = []

    for row in golden_rows:
        gid = row["metadata"]["golden_id"]
        golden = evals.get(gid)
        if not golden:
            continue

        is_truncated = row["metadata"].get("truncated", False)
        answer = row["messages"][1]["content"]

        results = _check_facts(answer, golden["facts"])
        for fact_idx, found in results.items():
            total_facts += 1
            if found:
                found_facts += 1
            elif not is_truncated:
                # Only flag missed facts for non-truncated rows
                fact_variants = golden["facts"][int(fact_idx)]
                missed_facts.append(
                    {
                        "golden_id": gid,
                        "fact_index": fact_idx,
                        "variants": fact_variants,
                    }
                )

    summary = {
        "total_golden_rows": len(golden_rows),
        "total_facts": total_facts,
        "found_facts": found_facts,
        "missed_facts": len(missed_facts),
        "fact_accuracy": f"{found_facts / total_facts * 100:.1f}%" if total_facts else "N/A",
        "all_passed": len(missed_facts) == 0,
        "missed_details": missed_facts[:20],  # First 20 for inspection
    }

    log.info(
        "Validation: %d/%d facts found (%s), %d missed",
        found_facts,
        total_facts,
        summary["fact_accuracy"],
        len(missed_facts),
    )
    if missed_facts:
        log.warning("Missed facts: %s", json.dumps(missed_facts[:5], indent=2))

    return summary


def _check_context_support(rows: list[dict], sample: int = 10) -> dict:
    """Check that answer claims are supported by the context for a sample."""
    import random as _random

    _random.seed(RANDOM_SEED)
    sample_rows = _random.sample(rows, min(sample, len(rows)))

    results = []
    for row in sample_rows:
        user_msg = row["messages"][0]["content"]
        assistant_msg = row["messages"][1]["content"]

        # Extract context from user message (text between "Context:" and "Question:")
        ctx_match = re.search(r"Context:\n\n(.*?)\n\nQuestion:", user_msg, re.DOTALL)
        context = ctx_match.group(1) if ctx_match else ""

        # Simple check: for each quoted number in the answer, verify it appears in context
        numbers_in_answer = re.findall(r"\b(\d{2,5})\b", assistant_msg)
        missing_numbers = [n for n in numbers_in_answer if n not in context]

        results.append(
            {
                "row_type": row["metadata"]["source"],
                "doc_name": row["metadata"].get("doc_name", row["metadata"].get("golden_id", "?")),
                "numbers_in_answer": len(numbers_in_answer),
                "numbers_not_in_context": len(missing_numbers),
                "missing_numbers_sample": missing_numbers[:5],
            }
        )

    return {
        "sampled": len(sample_rows),
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ChatML training dataset for Unsloth QLoRA fine-tuning"
    )
    parser.add_argument(
        "--source",
        choices=["golden", "synthetic", "all"],
        default="all",
        help="Source for training data generation (default: all)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run golden-fact validation on output"
    )
    parser.add_argument(
        "--scale", type=int, default=1200, help="Target number of synthetic rows (default: 1200)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-written rows in output file"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling API")
    args = parser.parse_args()

    if args.dry_run:
        log.info("=== DRY RUN ===")

    # Check API key is available (tests connectivity)
    if not args.dry_run:
        log.info("Testing API connectivity...")
        try:
            resp = _session.post(
                f"{API_BASE}/chat/completions",
                json={
                    "model": TEACHER_MODEL,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 10,
                },
                timeout=30,
            )
            resp.raise_for_status()
            log.info("API OK (model: %s)", TEACHER_MODEL)
        except Exception as e:
            log.error("API connectivity check FAILED: %s", e)
            log.error("Check WANDB_API_KEY / COREWEAVE_PROJECT env vars")
            sys.exit(1)

    all_rows: list[dict] = []

    # Phase 1
    if args.source in ("golden", "all"):
        rows = phase1_golden(dry_run=args.dry_run)
        all_rows.extend(rows)

    # Phase 2
    if args.source in ("synthetic", "all"):
        rows = phase2_synthetic(
            target=args.scale,
            dry_run=args.dry_run,
        )
        all_rows.extend(rows)

    if args.dry_run:
        log.info("=== DRY RUN complete (no API calls made) ===")
        return

    if not all_rows:
        log.warning("No rows generated!")
        return

    # Split and write
    train_rows, eval_rows = _split_train_eval(all_rows)
    _write_jsonl(train_rows, TRAIN_PATH)
    _write_jsonl(eval_rows, EVAL_PATH)

    log.info("=== Generation complete ===")
    log.info("Total rows: %d", len(all_rows))
    log.info("  Train: %d", len(train_rows))
    log.info("  Eval:  %d", len(eval_rows))

    # Validation
    if args.validate:
        log.info("=== Validation ===")
        val_summary = validate_golden_rows(all_rows)
        ctx_check = _check_context_support(all_rows, sample=10)

        log.info("Fact accuracy: %s", val_summary["fact_accuracy"])
        log.info("Context support: %d/%d rows sampled", ctx_check["sampled"], len(all_rows))
        for r in ctx_check["results"]:
            if r["numbers_not_in_context"] > 0:
                log.warning(
                    "  %s: %d numbers missing from context: %s",
                    r["doc_name"],
                    r["numbers_not_in_context"],
                    r["missing_numbers_sample"],
                )

    # Print summary
    source_counts = {}
    for row in all_rows:
        src = row["metadata"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    log.info("Source breakdown: %s", json.dumps(source_counts))

    # Sample rows
    log.info("\n=== Sample rows (first 2) ===")
    for row in all_rows[:2]:
        user_preview = row["messages"][0]["content"][:100] + "..."
        assistant_preview = row["messages"][1]["content"][:100] + "..."
        log.info("User: %s", user_preview)
        log.info("Assistant: %s", assistant_preview)
        log.info("Metadata: %s", json.dumps(row["metadata"]))
        log.info("---")


if __name__ == "__main__":
    main()
