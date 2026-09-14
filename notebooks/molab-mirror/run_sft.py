import json, torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train_raw = load_jsonl("data/training/unsloth_train.jsonl")
eval_raw = load_jsonl("data/training/unsloth_eval.jsonl")
train_ds = Dataset.from_list([{"messages": x["messages"]} for x in train_raw])
eval_ds = Dataset.from_list([{"messages": x["messages"]} for x in eval_raw])
dataset = DatasetDict({"train": train_ds, "eval": eval_ds})

STUDENT_MODEL = "ornith-ai/Ornith-1.5-9B"
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
    attn_implementation="sdpa",
    trust_remote_code=True,
)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)
student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()

def format_messages(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_dataset = dataset["train"].map(format_messages, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["eval"].map(format_messages, remove_columns=dataset["eval"].column_names)

training_config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=100,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=50,
    report_to="none",
    seed=42,
    max_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=student_model, args=training_config,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
print("Starting FULL training (2 epochs, 1046 examples)...")
trainer.train()
trainer.save_model("./sft_output/final")
print("Training complete. Saving done.")

# Push to HuggingFace Hub
from huggingface_hub import HfApi
print("Pushing to Hub...")
HfApi().upload_folder(
    folder_path="./sft_output/final",
    repo_id="Nubula/ornith-pgrag-training1",
    repo_type="model",
)
print("DONE: Model pushed to Hub!")