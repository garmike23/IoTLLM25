import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

# --- File paths ---
files = {
    "Baseline": "~/Desktop/IoTLLM25/SLMs/OPT/results_baseline_opt.csv",
    "ONNX": "~/Desktop/IoTLLM25/SLMs/OPT/results_onnx_fp32_opt.csv",
    "ONNX_INT8": "~/Desktop/IoTLLM25/SLMs/OPT/results_onnx_int8_opt.csv"
}
colors = {
    "Baseline": "#4682B4",      # blue
    "ONNX": "#FFA500",          # orange
    "ONNX_INT8": "#32CD32"      # green
}

# --- Load data ---
data = {}
for label, file in files.items():
    df = pd.read_csv(file)
    data[label] = df

# --- Plotting CDF helper ---
def plot_cdf_single(column, xlabel, save_name):
    plt.figure(figsize=(8, 6))
    for label, df in data.items():
        vals = df[column].dropna().values
        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, cdf, label=label, color=colors[label], linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    plt.subplots_adjust(top=0.83)  # Add space at the top for legend
    plt.tight_layout()
    plt.savefig(save_name, dpi=200)
    plt.show()

# --- Plot each CDF as a separate figure with top-center legend ---
plot_cdf_single(column="ram_after", xlabel="RAM Usage (MB)", save_name="cdf_ram_after.png")
plot_cdf_single(column="cpu_usage", xlabel="CPU Usage (%)", save_name="cdf_cpu_usage.png")
plot_cdf_single(column="inference_time", xlabel="Inference Time (s)", save_name="cdf_latency.png")

# ========================
# --- Grouped Mean Bar Plots ---
# ========================

mean_ram = []
mean_cpu = []
mean_inf = []
labels = []

for label, df in data.items():
    labels.append(label)
    mean_ram.append(df["ram_after"].mean())
    mean_cpu.append(df["cpu_usage"].mean())
    mean_inf.append(df["inference_time"].mean())

x_pos = np.arange(len(labels))

# --- Plot Mean RAM Usage ---
plt.figure(figsize=(5,4))
plt.bar(x_pos, mean_ram, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean RAM Usage (MB)")
plt.tight_layout()
plt.savefig("opt_mean_ram_usage.png", dpi=200)
plt.show()

# --- Plot Mean CPU Usage ---
plt.figure(figsize=(5,4))
plt.bar(x_pos, mean_cpu, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean CPU Usage (%)")
plt.tight_layout()
plt.savefig("opt_mean_cpu_usage.png", dpi=200)
plt.show()

# --- Plot Mean Inference Time ---
plt.figure(figsize=(5,4))
plt.bar(x_pos, mean_inf, color=[colors[l] for l in labels])
plt.xticks(x_pos, labels)
plt.ylabel("Mean Inference Time (s)")
plt.tight_layout()
plt.savefig("opt_mean_inference_time.png", dpi=200)
plt.show()
