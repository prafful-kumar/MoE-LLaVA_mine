import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PROPERLY RENAMED EXPERIMENTS
# ==========================================================

experiments = {
    "Student Only (Fisher Init, Norm)":
    "random_init_student_only_training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.0_0.0-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",

    "Author Baseline":
    "random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe",
}

# ==========================================================
# EMA Smoothing
# ==========================================================

def smooth_ema(values, alpha=0.05):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed

# ==========================================================
# Load + Normalize Steps
# ==========================================================

processed = {}

for name, folder in experiments.items():
    path = os.path.join(folder, "trainer_state.json")

    with open(path) as f:
        data = json.load(f)

    steps = []
    losses = []

    for entry in data["log_history"]:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    losses = smooth_ema(losses)

    total_steps = max(steps)
    norm_steps = [s / total_steps for s in steps]

    processed[name] = (norm_steps, losses)

    print(name)
    print("  Total Steps:", total_steps)
    print("  Logged Points:", len(steps))
    print()

# ==========================================================
# Plot Normalized Convergence
# ==========================================================

plt.figure(figsize=(8,6))

for name, (steps, losses) in processed.items():
    plt.plot(steps, losses, label=name)

plt.xlabel("Training Progress (0 → 1)")
plt.ylabel("Training Loss (EMA)")
plt.title("Student vs Author Baseline (Normalized Progress)")
plt.legend()
plt.grid(True)

plt.savefig("student_vs_author_normalized.png", dpi=300, bbox_inches="tight")
# plt.show()
plt.savefig("author.png")

# ==========================================================
# Plot ONLY First 10% of Training
# ==========================================================

plt.figure(figsize=(8,6))

for name, (steps, losses) in processed.items():

    early_steps = []
    early_losses = []

    for s, l in zip(steps, losses):
        if s <= 0.1:   # first 10% of normalized training
            early_steps.append(s)
            early_losses.append(l)

    plt.plot(early_steps, early_losses, label=name)

plt.xlabel("Training Progress (0 → 0.1)")
plt.ylabel("Training Loss (EMA)")
plt.title("Student vs Author Baseline (Early 10% Training)")
plt.legend()
plt.grid(True)

plt.xlim(0, 0.1)  # Force zoom to 0–10%
plt.savefig("student_vs_author_early_10_percent.png", dpi=300, bbox_inches="tight")
plt.close()