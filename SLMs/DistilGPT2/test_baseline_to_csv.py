import os
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import csv

def expand(path):
    return os.path.expanduser(os.path.expandvars(path))

# === Paths ===
model_path = expand("./distilgpt2_output")
dataset_path = expand("~/Desktop/IoTLLM25/SLMs/datasets/mc_dataset.json")
csv_path = expand("~/Desktop/IoTLLM25/SLMs/DistilGPT2/csv_files/results_baseline_distilgpt2.csv")

# === Helpers ===
def format_input(scenario, question):
    return f"Scenario: {scenario}\nQuestion: {question}\nAnswer:"

def clean_answer(decoded):
    answer = decoded.strip()
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    return answer

def greedy_generate_torch(model, tokenizer, prompt, max_new_tokens=50, eos_token_id=None):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    generated = input_ids.clone()
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    cpu_usages = []
    start_time = time.time()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=torch.ones_like(generated))
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token_id], dim=1)
            cpu_usages.append(psutil.cpu_percent(interval=None))
            if eos_token_id is not None and next_token_id[0].item() == eos_token_id:
                break

    end_time = time.time()
    ram_after = process.memory_info().rss / (1024 * 1024)
    cpu_usage = np.mean(cpu_usages)
    new_tokens = generated[0, input_ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    answer = clean_answer(decoded)
    latency = round(end_time - start_time, 3)

    return answer, latency, ram_before, ram_after, cpu_usage

# === Load dataset ===
with open(dataset_path, "r") as f:
    dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()
eos_token_id = tokenizer.eos_token_id

results = []

# === Run ===
for idx, item in enumerate(dataset):
    prompt = format_input(item["scenario"], item["question"])
    answer, latency, ram_before, ram_after, cpu_usage = greedy_generate_torch(
        model, tokenizer, prompt, max_new_tokens=50, eos_token_id=eos_token_id
    )
    results.append([answer, ram_before, ram_after, cpu_usage, latency])
    print(f"Run {idx+1}: {answer}")
    print(f"   Time: {latency}s | RAM: {ram_before:.2f}->{ram_after:.2f} MB | CPU: {cpu_usage:.2f}%\n")

# === Save CSV ===
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["answer", "ram_before", "ram_after", "cpu_usage", "inference_time"])
    writer.writerows(results)

print(f"\nResults saved to {csv_path}")
