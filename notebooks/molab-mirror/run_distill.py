#!/usr/bin/env python
"""
Distillation: Qwen3.8-27B (teacher, 4-bit) -> Ornith-1.5-9B (student, bf16 + LoRA).

Custom training loop with KL divergence + cross-entropy loss.
Temperature scaling (T=2.0), alpha=0.5. Only response tokens contribute.

Usage:
  uv run python run_distill.py
"""

import json, torch, os, time, gc, math
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
STUDENT_MODEL = "ornith-ai/Ornith-1.5-9B"
DATA_DIR = Path("data/training")

TRAIN_FILE = DATA_DIR / "expanded_train.jsonl"
FALLBACK_TRAIN = DATA_DIR / "unsloth_train.jsonl"
EVAL_FILE = DATA_DIR / "expanded_eval.jsonl"
FALLBACK_EVAL = DATA_DIR / "unsloth_eval.jsonl"

OUTPUT_DIR = Path("distill_output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_LORA_DIR = OUTPUT_DIR / "lora"
FINAL_MERGED_DIR = OUTPUT_DIR / "final"

# Hyperparameters
MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-4
EPOCHS = 3
WARMUP_STEPS = 20
LOGGING_STEPS = 5
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# Distillation
T = 2.0
ALPHA = 0.5

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_assistant_marker(text: str) -> int:
    """Find the start of the assistant response."""
    for marker in ["<|im_start|>assistant\n", "\nassistant\n", "<|start_header_id|>assistant<|end_header_id|>"]:
        idx = text.rfind(marker)
        if idx >= 0:
            return idx + len(marker)
    return -1


def tokenize_with_labels(messages: list[dict], tokenizer, max_length: int = MAX_LENGTH):
    """Apply chat template, tokenize, create labels masking prompt tokens."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]

    # Create labels: mask prompt tokens (-100), keep response tokens
    labels = input_ids.clone()
    marker_idx = find_assistant_marker(text)
    if marker_idx >= 0:
        prefix_tokens = tokenizer(text[:marker_idx], truncation=False)["input_ids"]
        prompt_len = len(prefix_tokens)
        labels[:prompt_len] = -100
    # Mask padding
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(tokenizer):
    """Load training data, preferring expanded files."""
    train_path = TRAIN_FILE if TRAIN_FILE.exists() else FALLBACK_TRAIN
    eval_path = EVAL_FILE if EVAL_FILE.exists() else FALLBACK_EVAL

    train_raw = load_jsonl(train_path)
    eval_raw = load_jsonl(eval_path)

    print(f"Train: {len(train_raw)}, Eval: {len(eval_raw)}")

    train_processed = [tokenize_with_labels(x["messages"], tokenizer) for x in train_raw]
    eval_processed = [tokenize_with_labels(x["messages"], tokenizer) for x in eval_raw]

    train_ds = Dataset.from_list(train_processed)
    eval_ds = Dataset.from_list(eval_processed)
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_teacher():
    """Load teacher in 4-bit, inference-only."""
    print(f"Loading teacher: {TEACHER_MODEL} (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=quant_config,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    # Disable gradients for teacher
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_student():
    """Load student in bf16 + LoRA."""
    print(f"Loading student: {STUDENT_MODEL} (bf16 + LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute distillation loss.

    loss = alpha * T^2 * KL(teacher || student) + (1-alpha) * CE(student, labels)

    Only over response tokens (labels != -100).
    """
    vocab_size = student_logits.size(-1)

    # Temperature-scaled logits
    s_logits = student_logits / T
    t_logits = teacher_logits / T

    # KL divergence: sum over vocab of teacher_probs * log(teacher_probs / student_probs)
    student_log_probs = F.log_softmax(s_logits, dim=-1)
    teacher_probs = F.softmax(t_logits, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",  # shape: [batch, seq_len, vocab]
    )

    # Mask for response tokens
    response_mask = (labels != -100).unsqueeze(-1).expand_as(kl)
    kl_masked = kl[response_mask].sum() / response_mask.sum().clamp(min=1)

    # Cross-entropy on student logits vs labels
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    loss = alpha * T * T * kl_masked + (1 - alpha) * ce_loss
    return loss, kl_masked, ce_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    """Main training function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_ds, eval_ds = load_data(tokenizer)
    def collate_fn(batch):
        """Collate a list of dicts into a batch dict of tensors."""
        def to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v
            return torch.tensor(v, dtype=torch.long)
        return {
            "input_ids": torch.stack([to_tensor(b["input_ids"]) for b in batch]),
            "attention_mask": torch.stack([to_tensor(b["attention_mask"]) for b in batch]),
            "labels": torch.stack([to_tensor(b["labels"]) for b in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Load teacher (no grads)
    teacher = load_teacher()

    # Load student with LoRA
    student = load_student()
    student.train()
    student.gradient_checkpointing_enable()

    # Optimizer (LoRA params only)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR)

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = min(WARMUP_STEPS, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # VRAM check
    free, total = torch.cuda.mem_get_info()
    print(f"\nVRAM: {(total - free) / 1e9:.1f} GB used / {total / 1e9:.1f} GB total")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}\n")

    # Training
    global_step = 0
    best_eval_loss = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")

        student.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_ce = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            # Student forward
            student_logits = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            # Compute loss
            loss, kl_loss, ce_loss = distill_loss(
                student_logits, teacher_logits, labels,
                T=T, alpha=ALPHA,
            )

            loss = loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM
            epoch_kl += kl_loss.item()
            epoch_ce += ce_loss.item()
            epoch_steps += 1

            # Gradient accumulation step
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % LOGGING_STEPS == 0:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    lr_current = scheduler.get_last_lr()[0]
                    print(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {epoch_loss / max(epoch_steps, 1):.4f} | "
                        f"KL: {epoch_kl / max(epoch_steps, 1):.4f} | "
                        f"CE: {epoch_ce / max(epoch_steps, 1):.4f} | "
                        f"LR: {lr_current:.2e} | "
                        f"VRAM: {(total_mem - free_mem) / 1e9:.1f} GB | "
                        f"Time: {time.time() - start_time:.0f}s"
                    )

                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    ckpt_dir = CHECKPOINT_DIR / f"step_{global_step}"
                    student.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"  Checkpoint saved: {ckpt_dir}")

                    # Clean old checkpoints
                    all_ckpts = sorted(
                        CHECKPOINT_DIR.glob("step_*"),
                        key=lambda p: int(p.name.split("_")[1]),
                    )
                    import shutil
                    while len(all_ckpts) > SAVE_TOTAL_LIMIT:
                        old = all_ckpts.pop(0)
                        shutil.rmtree(old)
                        print(f"  Removed old checkpoint: {old}")

        # End of epoch summary
        print(f"\n  Train | Loss={epoch_loss / max(epoch_steps, 1):.4f} "
              f"KL={epoch_kl / max(epoch_steps, 1):.4f} "
              f"CE={epoch_ce / max(epoch_steps, 1):.4f}")

        # Eval
        eval_loss = evaluate(student, teacher, eval_loader)
        print(f"  Eval  | Loss={eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            student.save_pretrained(str(FINAL_LORA_DIR))
            tokenizer.save_pretrained(str(FINAL_LORA_DIR))
            print(f"  New best! Saved to {FINAL_LORA_DIR}")

    # Save final
    print(f"\n{'='*60}")
    print("Training complete. Saving...")
    student.save_pretrained(str(FINAL_LORA_DIR))
    tokenizer.save_pretrained(str(FINAL_LORA_DIR))

    # Merge
    print("Merging LoRA into base model...")
    merged = student.merge_and_unload()
    merged.save_pretrained(str(FINAL_MERGED_DIR))
    tokenizer.save_pretrained(str(FINAL_MERGED_DIR))
    print(f"Merged model saved: {FINAL_MERGED_DIR}")

    # Cleanup
    del teacher, student, merged
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print("Done!")


def evaluate(student, teacher, eval_loader):
    """Evaluate student on eval set."""
    student.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")

            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            loss, _, _ = distill_loss(
                student_logits, teacher_logits, labels,
                T=T, alpha=ALPHA,
            )
            total_loss += loss.item()
            n += 1

    student.train()
    return total_loss / max(n, 1)


if __name__ == "__main__":
    train()