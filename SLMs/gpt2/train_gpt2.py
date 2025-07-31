import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

import logging
import wandb

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# --- Paths ---
data_path = "test_dataset.json"   # <-- Change to your dataset
model_output_dir = "./gpt2_125m_lora_output"
os.makedirs(model_output_dir, exist_ok=True)

# --- Load dataset ---
df = pd.read_json(data_path)

# --- Create prompts ---
df["prompt"] = (
    "Scenario: " + df["scenario"].astype(str) +
    "\nQuestion: " + df["question"].astype(str) +
    "\nAnswer:"
)
df["text"] = df["prompt"] + " " + df["answer"].astype(str)

# --- Train/eval split ---
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# --- Model and tokenizer ---
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

# --- LoRA Block (optional, commented out) ---

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # Rank (tunable)
    lora_alpha=16,           # Scaling (tunable)
    lora_dropout=0.05,       # Dropout (tunable)
 )
model = get_peft_model(model, lora_config)

# --- Tokenize ---
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# --- WANDB Integration ---
wandb.init(
    project="carqa-gpt2-125m",
    name="gpt2-125m-lora-pi4b-finetune"
)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=model_output_dir,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=3000,
    save_total_limit=1,
    per_device_train_batch_size=3,     # Adjust for RAM if needed
    per_device_eval_batch_size=3,      # Adjust for RAM if needed
    num_train_epochs=30,
    logging_steps=10,
    learning_rate=1e-5,
    fp16=False,
    push_to_hub=False,
    report_to=["wandb"],
    run_name="gpt2-125m-lora-pi4b-finetune",
    disable_tqdm=False,
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

logger.info("Training GPT2-125M on Raspberry Pi 4B")
trainer.train()
logger.info("Done! Saving final model.")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)