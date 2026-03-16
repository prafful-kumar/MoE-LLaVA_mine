import os
import json
import matplotlib.pyplot as plt
import numpy as np

experiments = {
    "Qwen — Random (Author)": "checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/trainer_state.json",
    "Qwen — Student-Only": "checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/trainer_state.json",
    "Qwen — Teacher-Student": "/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/trainer_state.json",
    "Phi2 — Student-Only (partial)": "checkpoints_phi_student/llavaphi-2.7b-finetune-moe/checkpoint-1000/trainer_state.json",
}

colors = {
    "Qwen — Random (Author)": "#1f77b4",
    "Qwen — Student-Only": "#ff7f0e",
    "Qwen — Teacher-Student": "#2ca02c",
    "Phi2 — Student-Only (partial)": "#d62728",
}

def smooth_ema(values, alpha=0.05):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed

# --- Plot 1: Full training loss (all variants) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for name, path in experiments.items():
    with open(path) as f:
        data = json.load(f)
    log = data["log_history"]
    steps = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    smoothed = smooth_ema(losses, alpha=0.05)
    ax1.plot(steps, smoothed, label=name, color=colors[name], linewidth=1.5)

ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax1.set_title("Training Loss — All Variants", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Qwen-only, zoomed comparison ---
ax2 = axes[1]
qwen_exps = {k: v for k, v in experiments.items() if "Qwen" in k}
for name, path in qwen_exps.items():
    with open(path) as f:
        data = json.load(f)
    log = data["log_history"]
    steps = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    smoothed = smooth_ema(losses, alpha=0.05)
    ax2.plot(steps, smoothed, label=name, color=colors[name], linewidth=1.5)

ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax2.set_title("Training Loss — Qwen Variants (Router Init Comparison)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_comparison.png")

# --- Plot 3: Early convergence (first 1000 steps, all 4) ---
fig, ax = plt.subplots(figsize=(10, 6))
for name, path in experiments.items():
    with open(path) as f:
        data = json.load(f)
    log = data["log_history"]
    steps = [e["step"] for e in log if "loss" in e and e["step"] <= 1000]
    losses = [e["loss"] for e in log if "loss" in e and e["step"] <= 1000]
    smoothed = smooth_ema(losses, alpha=0.1)
    ax.plot(steps, smoothed, label=name, color=colors[name], linewidth=1.5)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax.set_title("Early Convergence (First 1000 Steps)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_early_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_early_convergence.png")

# --- Plot 4: Learning rate schedule ---
fig, ax = plt.subplots(figsize=(10, 6))
for name, path in experiments.items():
    with open(path) as f:
        data = json.load(f)
    log = data["log_history"]
    steps = [e["step"] for e in log if "learning_rate" in e]
    lrs = [e["learning_rate"] for e in log if "learning_rate" in e]
    ax.plot(steps, lrs, label=name, color=colors[name], linewidth=1.5)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("Learning Rate Schedule", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lr_schedule.png")
