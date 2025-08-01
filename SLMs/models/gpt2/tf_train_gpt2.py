import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForCausalLM,
)
import tensorflow as tf
import logging

tf.config.optimizer.set_jit(False)  # Disable XLA
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
data_path = "/home/carlos/Desktop/dataset.json"  # <-- Your JSON file
model_output_dir = "./gpt2_tf_output"
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

# ---- Tokenizer ----
model_id = "gpt2"  # Native TF weights available
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---- Tokenize function ----
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,  # Reduce for speed on Pi
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

# Convert to tf.data.Dataset
def hf_to_tf_dataset(hf_dataset, batch_size=1):
    features = dict(hf_dataset.remove_columns(["text", "prompt", "scenario", "question", "answer"]))
    labels = hf_dataset["input_ids"]
    return (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(len(hf_dataset))
        .batch(batch_size)
    )

tf_train_dataset = hf_to_tf_dataset(train_dataset, batch_size=1)  # Small batch for Pi
tf_eval_dataset = hf_to_tf_dataset(eval_dataset, batch_size=1)

# ---- Load TensorFlow model (native TF weights, no from_pt) ----
logger.info("Loading GPT-2 TensorFlow model...")
model = TFAutoModelForCausalLM.from_pretrained(model_id)
model.resize_token_embeddings(len(tokenizer))

# ---- Optimizer & Loss ----
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_losses = []
eval_losses = []

# ---- Training loop ----
epochs = 1  # Keep low for Pi testing
for epoch in range(epochs):
    logger.info(f"Epoch {epoch+1}/{epochs}")

    # Training
    train_loss = 0
    steps = 0
    for batch in tf_train_dataset:
        inputs, labels = batch
        with tf.GradientTape() as tape:
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss += loss.numpy()
        steps += 1
    avg_train_loss = train_loss / steps
    train_losses.append(avg_train_loss)

    # Evaluation
    eval_loss = 0
    steps = 0
    for batch in tf_eval_dataset:
        inputs, labels = batch
        outputs = model(inputs, labels=labels, training=False)
        eval_loss += outputs.loss.numpy()
        steps += 1
    avg_eval_loss = eval_loss / steps
    eval_losses.append(avg_eval_loss)

    logger.info(f"Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")

# ---- Save model ----
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
logger.info("Model saved to %s", model_output_dir)

# ---- Plot loss curves ----
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(eval_losses, label="Evaluation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss (GPT-2 TF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_eval_loss.png")
plt.close()

logger.info("Loss plot saved to training_eval_loss.png")
