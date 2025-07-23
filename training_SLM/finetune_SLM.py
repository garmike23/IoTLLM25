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
import logging

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
data_path = "./final_unique_complex_driving_dataset_1000.json"   # <-- Your JSON file   
model_output_dir = "./distilgpt2_carqa_output"
os.makedirs(model_output_dir, exist_ok=True)

# ---- Load dataset ----
df = pd.read_json(data_path)

# ---- Create prompts ----
df["prompt"] = (
    "Scenario: " + df["scenario"].astype(str) +
    "\nQuestion: " + df["question"].astype(str) +
    "\nAnswer:"
)
df["text"] = df["prompt"] + " " + df["answer"].astype(str)

# ---- Train/eval split ----
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# ---- Model and tokenizer ----
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---- Tokenize ----
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

# ---- Data collator ----
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ---- Model ----
model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

# ---- WANDB Integration ----
import wandb
wandb.init(
    project="carqa-distilgpt2",         # Change to your project name
    name="distilgpt2-pi4b-finetune"     # This run's name
)

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=model_output_dir,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    push_to_hub=False,
    report_to=[],                 # <--- KEY LINE
    run_name="distilgpt2-pi4b-finetune", # <--- Easier to find your run
    disable_tqdm=False,
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

logger.info("Training DistilGPT-2 on Raspberry Pi 4B. This may take a while!")
trainer.train()
logger.info("Done! Saving final model.")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
