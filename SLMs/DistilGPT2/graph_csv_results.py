import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

# Folder to save all graphs
graphs_directory = "~/Desktop/IoTLLM25/SLMs/DistilGPT2/Graphs"

# CSV file paths
files = {
    "Baseline": "~/Desktop/IoTLLM25/SLMs/DistilGPT2/csv_files/results_baseline_distilgpt2.csv",
    "ONNX": "~/Desktop/IoTLLM25/SLMs/DistilGPT2/csv_files/results_onnx_fp32_distilgpt2.csv",
    "ONNX_INT8": "~/Desktop/IoTLLM25/SLMs/DistilGPT2/csv_files/results_onnx_int8_distilgpt2.csv"
}

# Colors for each method
colors = {
    "Baseline": "#4682B4",   # blue
    "ONNX": "#FFA500",       # orange
    "ONNX_INT8": "#32CD32"   # green
}

def expand(path):
    return os.path.expanduser(os.path.expandvars(path))

def save_path(name):
    """Build the full path for a file inside graphs_directory."""
    return os.path.join(expand(graphs_directory), name)

# Load data

data = {}
for label, file in files.items():
    df = pd.read_csv(expand(file))     # expand '~' so reads work
    data[label] = df                   # store DataFrame

# Plotting

def plot_cdf_single(column, xlabel, save_name):
    """Plot one CDF comparing all methods for a chosen column."""
    plt.figure(figsize=(8, 6))
    for label, df in data.items():
        vals = df[column].dropna().values             # clean values
        vals = np.sort(vals)                          # sort values
        cdf = np.arange(1, len(vals) + 1) / len(vals) # simple CDF
        plt.plot(vals, cdf, label=label, color=colors[label], linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True, linestyle='--', alpha=0.6)
    # Put legend on top, centered, across one row
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    # Add space at top for the legend
    plt.subplots_adjust(top=0.83)
    plt.tight_layout()
    # Save into the graphs folder
    plt.savefig(save_path(save_name), dpi=200)
    plt.show()

# CDF plots
plot_cdf_single(column="ram_after",      xlabel="RAM Usage (MB)",       save_name="distilgpt2_cdf_ram_after.png")
plot_cdf_single(column="cpu_usage",      xlabel="CPU Usage (%)",        save_name="distilgpt2_cdf_cpu_usage.png")
plot_cdf_single(column="inference_time", xlabel="Inference Time (s)",   save_name="distilgpt2_cdf_latency.png")

# Grouped Mean Bar Plots

mean_ram = []
mean_cpu = []
mean_inf = []
labels = []

# Compute means for each method
for label, df in data.items():
    labels.append(label)
    mean_ram.append(df["ram_after"].mean())
    mean_cpu.append(df["cpu_usage"].mean())
    mean_inf.append(df["inference_time"].mean())

x_pos = np.arange(len(labels))  # bar positions

# --- Mean RAM Usage ---
plt.figure(figsize=(8, 6))
plt.bar(x_pos, mean_ram, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean RAM Usage (MB)")
plt.tight_layout()
plt.savefig(save_path("distilgpt2_mean_ram_usage.png"), dpi=200)
plt.show()

# --- Mean CPU Usage ---
plt.figure(figsize=(8, 6))
plt.bar(x_pos, mean_cpu, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean CPU Usage (%)")
plt.tight_layout()
plt.savefig(save_path("distilgpt2_mean_cpu_usage.png"), dpi=200)
plt.show()

# --- Mean Inference Time ---
plt.figure(figsize=(8, 6))
plt.bar(x_pos, mean_inf, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean Inference Time (s)")
plt.tight_layout()
plt.savefig(save_path("distilgpt2_mean_inference_time.png"), dpi=200)
plt.show()
