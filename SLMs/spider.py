import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- EDIT THESE PATHS ----------
files = {
    "DistilGPT2": "/home/carlos/Desktop/final_graphs_distilgpt2/distilgpt2_results_onnx_int8.csv",
    "GPT2": "/home/carlos/Desktop/final_graphs_gpt2/results_onnx_int8_gpt2.csv",
    "OPT": "/home/carlos/Desktop/final_graphs_opt/results_onnx_int8_opt.csv",
}

# Colors for consistency
colors = {
    "DistilGPT2": "#1f77b4",   # blue
    "GPT2": "#ff7f0e",         # orange
    "OPT": "#2ca02c",          # green
}

# Font Size
plt.rcParams.update({"font.size": 22})

# ---------- LOAD + AVERAGE ----------
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_time_column(df: pd.DataFrame) -> str:
    return "inference_time" if "inference_time" in df.columns else "latency"

averages = {}
for label, path in files.items():
    df = read_csv(path)
    time_col = get_time_column(df)
    averages[label] = {
        "avg_time": float(df[time_col].dropna().mean()),
        "avg_ram_after": float(df["ram_after"].dropna().mean()),
        "avg_cpu": float(df["cpu_usage"].dropna().mean()),
    }

# ---------- SCORE ----------
def to_scores(averages_dict):
    metrics = ["avg_time", "avg_ram_after", "avg_cpu"]
    mins = {m: min(averages_dict[k][m] for k in averages_dict) for m in metrics}
    scores = {}
    for label in averages_dict:
        scores[label] = {
            m: float(mins[m] / averages_dict[label][m] * 100.0) if averages_dict[label][m] > 0 else 0.0
            for m in metrics
        }
    return scores

scores = to_scores(averages)

# ---------- RADAR PLOT ----------
labels = ["Inference Time Score", "RAM Usage Score", "CPU Usage Score"]
metrics_order = ["avg_time", "avg_ram_after", "avg_cpu"]
N = len(labels)
angles = np.linspace(0, 2 * math.pi, N, endpoint=False)

def close_loop(values: list[float]) -> list[float]:
    return values + values[:1]

angles_closed = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Put first axis at the top
ax.set_theta_offset(math.pi / 2.0)
ax.set_theta_direction(-1)

# Draw one axis per variable
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.tick_params(axis="x", pad=20)  # Move vertex labels outward

# Radial labels
ax.set_rlabel_position(0)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"])
ax.set_ylim(0, 100)

# Plot each model
for label in files.keys():
    vals = [scores[label][m] for m in metrics_order]
    vals_closed = close_loop(vals)
    ax.plot(angles_closed, vals_closed, linewidth=2.5, label=label, color=colors[label])
    ax.fill(angles_closed, vals_closed, alpha=0.15, color=colors[label])

ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
    ncol=len(files)  # makes it horizontal
)


plt.tight_layout()
plt.savefig("spider_scores.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Print raw averages and scores ----------
print("\n===> Averages (lower is better)")
print(f"{'Model':<15} {'Avg Time (s)':<15} {'Avg RAM (MB)':<15} {'Avg CPU (%)':<15}")
for k, v in averages.items():
    print(f"{k:<15} {v['avg_time']:<15.4f} {v['avg_ram_after']:<15.2f} {v['avg_cpu']:<15.2f}")

print("\n===> Scores (higher is better, min gets 100)")
print(f"{'Model':<15} {'Time Score':<15} {'RAM Score':<15} {'CPU Score':<15}")
for k, v in scores.items():
    print(f"{k:<15} {v['avg_time']:<15.1f} {v['avg_ram_after']:<15.1f} {v['avg_cpu']:<15.1f}")
