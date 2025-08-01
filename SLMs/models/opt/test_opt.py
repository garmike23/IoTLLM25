import os
import time
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import json
import matplotlib.pyplot as plt

onnx_path = "/aul/homes/ccarr192/Desktop/DistilGPT2/distilgpt2_int8.onnx"
tokenizer_path = "/aul/homes/ccarr192/Desktop/DistilGPT2/distilgpt2_82m_output"
dataset_path = "/aul/homes/ccarr192/Desktop/DistilGPT2/mc_scenario.json"

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
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    generated = input_ids.copy()
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    cpu_usages = []
    input_name = "input_ids"
    start_time = time.time()
    for _ in range(max_new_tokens):
        outputs = session.run(["logits"], {input_name: generated})
        next_token_logits = outputs[0][:, -1, :]
        next_token_id = next_token_logits.argmax(axis=-1)
        generated = np.concatenate([generated, next_token_id[:, None]], axis=1)
        cpu_usages.append(psutil.cpu_percent(interval=None))
        if eos_token_id is not None and next_token_id[0] == eos_token_id:
            break
    end_time = time.time()
    ram_after = process.memory_info().rss / (1024 * 1024)
    cpu_usage = np.mean(cpu_usages)
    new_tokens = generated[0, input_ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    answer = clean_answer(decoded)
    latency = round(end_time - start_time, 3)
    return answer, latency, ram_before, ram_after, cpu_usage

# Load dataset
with open(dataset_path, "r") as f:
    dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token_id = tokenizer.eos_token_id
session = ort.InferenceSession(onnx_path)

all_times, all_ram_before, all_ram_after, all_cpu = [], [], [], []

for idx, item in enumerate(dataset):
    prompt = format_input(item["scenario"], item["question"])
    answer, latency, ram_before, ram_after, cpu_usage = greedy_generate_onnx(
        session, tokenizer, prompt, max_new_tokens=50, eos_token_id=eos_token_id)
    all_times.append(latency)
    all_ram_before.append(ram_before)
    all_ram_after.append(ram_after)
    all_cpu.append(cpu_usage)
    print(f"Run {idx+1}: Inference Time: {latency}s, RAM: {ram_before:.2f} -> {ram_after:.2f} MB, CPU: {cpu_usage:.2f}%")

# Save plots to Desktop
desktop_path = os.path.expanduser("/aul/homes/ccarr192/Desktop/DistilGPT2/")

# Inference time
plt.figure(figsize=(10,5))
plt.bar(range(1, len(all_times)+1), all_times, color='teal')
plt.title("Inference Time per Run (DistilGPT2_ONNX_INT8)")
plt.xlabel("Run")
plt.ylabel("Inference Time (s)")
plt.tight_layout()
time_plot_path = os.path.join(desktop_path, "inference_time_per_run.png")
plt.savefig(time_plot_path, dpi=200)
plt.close()
print(f"Inference time plot saved to {time_plot_path}")

# RAM usage
plt.figure(figsize=(10,5))
plt.bar(range(1, len(all_ram_before)+1), all_ram_before, label="RAM Before", color='orange', alpha=0.6)
plt.bar(range(1, len(all_ram_after)+1), all_ram_after, label="RAM After", color='blue', alpha=0.6)
plt.title("RAM Usage per Run (DistilGPT2_ONNX_INT8)")
plt.xlabel("Run")
plt.ylabel("RAM (MB)")
plt.legend()
plt.tight_layout()
ram_plot_path = os.path.join(desktop_path, "ram_usage_per_run.png")
plt.savefig(ram_plot_path, dpi=200)
plt.close()
print(f"RAM usage plot saved to {ram_plot_path}")

# CPU usage
plt.figure(figsize=(10,5))
plt.bar(range(1, len(all_cpu)+1), all_cpu, color='purple')
plt.title("CPU Usage per Run (DistilGPT2_ONNX_INT8)")
plt.xlabel("Run")
plt.ylabel("Average CPU Usage (%)")
plt.tight_layout()
cpu_plot_path = os.path.join(desktop_path, "cpu_usage_per_run.png")
plt.savefig(cpu_plot_path, dpi=200)
plt.close()
print(f"CPU usage plot saved to {cpu_plot_path}")

print("\n==== Monte Carlo Summary (DistilGPT2_ONNX_INT8) ====")
print(f"Average Inference Time: {np.mean(all_times):.3f} s")
print(f"Average RAM Before: {np.mean(all_ram_before):.2f} MB")
print(f"Average RAM After: {np.mean(all_ram_after):.2f} MB")
print(f"Average CPU Usage: {np.mean(all_cpu):.2f} %")
