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
# from peft import get_peft_model, LoraConfig, TaskType
import logging

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
data_path = "/home/"  #full data path

# ---- Choose your model ----
# DistilGPT-2 (~82M)
model_id = "distilgpt2"
model_output_dir = "./distilgpt2_output"

# GPT-2 (~124M)
# model_id = "gpt2"
# model_output_dir = "./gpt2_output"

# OPT (~125M)
# model_id = "facebook/opt-125m"
# model_output_dir = "./opt125m_output"

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
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

# ---- Add LoRA (commented out) ---
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
# )
# model = get_peft_model(model, lora_config)

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

class LossLoggerCallback:
    """Custom callback to log training and evaluation loss for plotting"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            eval_losses.append(logs["eval_loss"])

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

logger.info(f"Training {model_id} on Raspberry Pi 4B. This may take a while!")
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
plt.title(f"Training and Evaluation Loss ({model_id})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{model_id.replace('/', '_')}_training_eval_loss.png")
plt.close()

logger.info(f"Loss plot saved to {model_id.replace('/', '_')}_training_eval_loss.png")
