# 🧮 GRPO Math Training (Colab Notebook)

This repository contains a **Colab-ready notebook** for training and evaluating a math-specialized language model using **Grouped Reinforcement Policy Optimization (GRPO)** with verifiable rewards.

⚠️ **Current Status**: Training has been completed **up through GRPO**.
Stages such as **PRM training**, **PRM-guided self-evolution**, and **Length-aware RL** are **planned but not yet implemented**.

---

## 🚀 QuickStart (5 minutes)

1. **Open in Colab**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10jD5uv-6-JednadO5qekNa_F7yKr0-Za)

2. **Download and unzip the GRPO checkpoint**
   From Google Drive: [grpo\_ckpt\_runs.zip](https://drive.google.com/file/d/1TnfDGtB_ZX1tdiPOWKnQ-_Fx6gnt6Zm8/view?usp=sharing)

   ```bash
   !unzip grpo_ckpt_runs.zip -d /content/outputs/
   ```

3. **Run inference**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   path = "/content/outputs/default-GRPO/checkpoint-2048"
   model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto")
   tok = AutoTokenizer.from_pretrained(path, use_fast=True)
   tok.pad_token, tok.padding_side = tok.eos_token, "left"

   q = "If a car travels at 60 miles per hour for 2.5 hours, how many miles does it travel?"
   inputs = tok(q, return_tensors="pt").to(model.device)
   out = model.generate(**inputs, max_new_tokens=128)
   print(tok.decode(out[0], skip_special_tokens=True))
   ```

---

## 📑 Workflow

The notebook follows the multi-stage workflow:

```mermaid
flowchart TB
  subgraph TRAINING["Training (Colab • A100-40GB)"]
    A0[Public math sets GSM8K/MATH/AIME]
    A1[Teacher synthesis OpenAI → PoT traces + tests JSON]
    A2[Local verification → Python/Sympy filter]
    A3[SFT → TRL SFTTrainer]
    A4[RL → GRPO (verifiable rewards)]
    A5[PRM training (OmegaPRM/PRM800K)]
    A6[PRM-guided self-evolution]
    A7[Length-aware finishing RL]
  end
  subgraph ARTS["Artifacts"]
    M[Checkpoints: sft-poT/, grpo/final/]
    P[Verified PoT JSONL]
    R[PRM weights optional]
    T[Tokenizer/configs]
  end
  A0-->A1-->A2-->A3-->A4-->A7
  A2-->A5-->A6-->A3
  A6-->A4
  A3-->M
  A4-->M
  A7-->M
  A2-->P
  A5-->R
  T-.->M
```

---

## 📂 Artifacts

* **Checkpoint Archive:** [grpo\_ckpt\_runs.zip](https://drive.google.com/file/d/1TnfDGtB_ZX1tdiPOWKnQ-_Fx6gnt6Zm8/view?usp=sharing)

  * `checkpoint-2048/` → GRPO model weights, tokenizer, configs
  * `runs/` → Training logs (optional for monitoring)

👉 For inference, only `checkpoint-2048/` is required.

---

## ⚙️ Requirements

If you’re not using Colab, install manually:

```bash
pip install torch==2.8.0
pip install transformers==4.56.1
pip install trl==0.22.2
pip install accelerate==1.10.1
pip install peft==0.17.1
pip install sympy datasets
```

---

## 📊 Next Steps (Planned)

* **A5. PRM Training**

  * Train a Process Reward Model (PRM) with PRM800K or heuristic labels.
  * Score step correctness probabilities.

* **A6. PRM-Guided Self-Evolution**

  * Generate N solutions per question.
  * Score with PRM + verify.
  * Keep only high-quality traces for further training.

* **A7. Length-Aware RL**

  * Final RL stage with brevity penalty.
  * Encourages short, precise outputs.

---

## 🧩 PRM Training Templates

### A5-A. Use an existing PRM

```python
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PRM_ID = "Qwen/Qwen2.5-Math-PRM-7B"
tok = AutoTokenizer.from_pretrained(PRM_ID, use_fast=True)
prm = AutoModelForSequenceClassification.from_pretrained(
    PRM_ID, torch_dtype="bfloat16", device_map="auto"
).eval()

def prm_step_score(q, step, idx=1):
    text = f"{q}\n\n# step {idx}\n{step}"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(prm.device)
    with torch.no_grad():
        logits = prm(**inputs).logits
        return float(F.softmax(logits, dim=-1)[0,1])
```

### A5-B. Train your own PRM on verified traces

```python
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

VERIFIED_JSONL = "/content/drive/MyDrive/ncu_green_ai/data/teacher_verified.jsonl"

rows = []
for line in open(VERIFIED_JSONL):
    js = json.loads(line)
    q, prog = js["question"], js["cot_program"]
    for i, ln in enumerate(prog.splitlines(),1):
        if ln.strip():
            rows.append({"text": f"{q}\n\n# step {i}\n{ln}", "label": 1})

ds = Dataset.from_list(rows)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", use_fast=True)

def tokenize_fn(ex): 
    out = tok(ex["text"], truncation=True, max_length=1024)
    out["labels"] = ex["label"]
    return out

ds_tok = ds.map(tokenize_fn, remove_columns=ds.column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B", num_labels=2, torch_dtype="bfloat16", device_map="auto"
)

args = TrainingArguments(
    output_dir="/content/prm_out", per_device_train_batch_size=4,
    gradient_accumulation_steps=2, learning_rate=1e-5,
    num_train_epochs=1, bf16=True, logging_steps=20, save_steps=200
)

trainer = Trainer(model=model, args=args, train_dataset=ds_tok, tokenizer=tok)
trainer.train()
trainer.save_model("/content/prm_out/final")
tok.save_pretrained("/content/prm_out/final")
```

---
