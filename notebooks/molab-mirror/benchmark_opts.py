import json, torch, time
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

with open("data/training/unsloth_train.jsonl") as _f:
    raw = [json.loads(line) for line in _f][:40]
ds = Dataset.from_list([{"messages": x["messages"]} for x in raw])
STUDENT_MODEL = "ornith-ai/Ornith-1.5-9B"

def setup(attn=None, compile_m=False, liger=False):
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    if attn: kw["attn_implementation"] = attn
    m = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, **kw)
    m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none"))
    if compile_m: m = torch.compile(m, mode="reduce-overhead")
    def fmt(ex): return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    return m, tok, ds.map(fmt, remove_columns=ds.column_names)

def bench(label, attn=None, compile_m=False, liger=False):
    print(f"\n--- {label} ---")
    m, tok, tds = setup(attn, compile_m, liger)
    cfg = SFTConfig(output_dir=f"./b_{label.replace(chr(32),chr(95))}", per_device_train_batch_size=2, gradient_accumulation_steps=2, learning_rate=2e-5, max_steps=5, bf16=True, gradient_checkpointing=True, logging_steps=1, report_to="none", max_length=2048, dataset_text_field="text", use_liger_kernel=liger, save_strategy="no")
    tr = SFTTrainer(model=m, args=cfg, train_dataset=tds, processing_class=tok)
    t0 = time.time()
    tr.train()
    elapsed = time.time() - t0
    del tr, m
    torch.cuda.empty_cache()
    import time as _t; _t.sleep(2)
    print(f"Time: {elapsed:.1f}s ({elapsed/5:.1f}s/step)")
    return elapsed

print("GPU:", torch.cuda.get_device_name(0))
R = {}
R["baseline sdpa"] = bench("baseline", attn="sdpa")
R["sdpa + liger"] = bench("liger", attn="sdpa", liger=True)
R["sdpa + compile"] = bench("compile", attn="sdpa", compile_m=True)
print("\n=== RESULTS ===")
for k, v in R.items(): print(f"  {k}: {v:.1f}s ({v/5:.1f}s/step)")