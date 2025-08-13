import os
import time
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import json
import csv

# ==== Set these per model ====
model_type = "onnx_int8"  # Change to "onnx_fp32", "baseline", etc
onnx_path = "./distilgpt2_output_int8.onnx"  # Change for each model
tokenizer_path = "./distilgpt2_output"
dataset_path = "~/Desktop/IoTLLM25/SLMs/datasets/mc_dataset.json"
output_csv = "results_onnx_int8_distilgpt2.csv"

def format_input(scenario, question):
    return f"Scenario: {scenario}\nQuestion: {question}\nAnswer:"

def clean_answer(decoded):
    answer = decoded.strip()
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()
    return answer

def greedy_generate_onnx(session, tokenizer, prompt, max_new_tokens=50, eos_token_id=None):
    enc = tokenizer(prompt, return_tensors="np")
    generated = enc["input_ids"].copy()
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    cpu_usages = []
    start_time = time.time()

    for _ in range(max_new_tokens):
        attention_mask = np.ones_like(generated, dtype=np.int64)
        outputs = session.run(
            ["logits"],
            {
                "input_ids": generated,
                "attention_mask": attention_mask
            }
        )
        next_token_logits = outputs[0][:, -1, :]
        next_token_id = next_token_logits.argmax(axis=-1)
        generated = np.concatenate([generated, next_token_id[:, None]], axis=1)
        cpu_usages.append(psutil.cpu_percent(interval=None))
        if eos_token_id is not None and next_token_id[0] == eos_token_id:
            break

    end_time = time.time()
    ram_after = process.memory_info().rss / (1024 * 1024)
    cpu_usage = np.mean(cpu_usages)
    new_tokens = generated[0, enc["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    answer = clean_answer(decoded)
    latency = round(end_time - start_time, 3)
    return answer, latency, ram_before, ram_after, cpu_usage

with open(dataset_path, "r") as f:
    dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token_id = tokenizer.eos_token_id
session = ort.InferenceSession(onnx_path)

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["answer", "ram_before", "ram_after", "cpu_usage", "inference_time"])
    for idx, item in enumerate(dataset):
        prompt = format_input(item["scenario"], item["question"])
        answer, latency, ram_before, ram_after, cpu_usage = greedy_generate_onnx(
            session, tokenizer, prompt, max_new_tokens=50, eos_token_id=eos_token_id
        )
        writer.writerow([answer, ram_before, ram_after, cpu_usage, latency])
        print(f"Run {idx+1}: {answer}")
        print(f"   Time: {latency}s | RAM: {ram_before:.2f}->{ram_after:.2f} MB | CPU: {cpu_usage:.2f}%\n")
