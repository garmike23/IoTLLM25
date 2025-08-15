import os
import pandas as pd
import matplotlib.pyplot as plt
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
data_path = "~/Desktop/IoTLLM25/SLMs/datasets/dataset.json"  # full data path
model_output_dir = "./gpt2_output"

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

# ---- Load local tokenizer and model ----
tokenizer = AutoTokenizer.from_pretrained(expand("./gpt2_local"))
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(expand("./gpt2_local"))
model.resize_token_embeddings(len(tokenizer))

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

# ---- Loss tracking ----
train_losses = []
eval_losses = []

from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    """Custom callback to log training and evaluation loss for plotting"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            eval_losses.append(logs["eval_loss"])

    def on_init_end(self, args, state, control, **kwargs):
        pass  # fixes AttributeError

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=40,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=1,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=9,
    logging_steps=10,
    learning_rate=1e-5,
    fp16=False,  # Pi doesn't support fp16 training
    push_to_hub=False,
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
    callbacks=[LossLoggerCallback()],
)

logger.info("Training gpt2_local on Raspberry Pi 5. This may take a while!")
trainer.train()
logger.info("Done! Saving final model.")
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# ---- Plot and save loss curves ----
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(eval_losses, label="Evaluation Loss")
plt.xlabel("Logging Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss (gpt2_local)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gpt2_local_training_eval_loss.png")
plt.close()

logger.info("Loss plot saved to gpt2_local_training_eval_loss.png")
