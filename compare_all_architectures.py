import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# DEFINE ALL ARCHITECTURES HERE (RENAMED CLEANLY)
# ==========================================================

experiments = {

    # Student Implementation
    "Student Only (Fisher Init, Norm)":
    "random_init_student_only_training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.0_0.0-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",

    # Author Baseline
    "Author Baseline":
    "random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe",

    # Fisher Variants
    # "Fisher (No Norm)":
    # "fisher_no_norm_input_norm_weight_during_training_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",

    "Fisher (TS + Norm)":
    "fisher_TS_norm_training_norm_input_norm_weight./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",
}

# ==========================================================
# EMA SMOOTHING
# ==========================================================

def smooth_ema(values, alpha=0.05):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed

# ==========================================================
# LOAD + NORMALIZE
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

    print(f"{name}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Logged Points: {len(steps)}")
    print()

# ==========================================================
# 1️⃣ FULL NORMALIZED CONVERGENCE
# ==========================================================

plt.figure(figsize=(8,6))

for name, (steps, losses) in processed.items():
    plt.plot(steps, losses, label=name)

plt.xlabel("Training Progress (0 → 1)")
plt.ylabel("Training Loss (EMA)")
plt.title("Full Normalized Convergence")
plt.legend()
plt.grid(True)

plt.savefig("full_normalized_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 2️⃣ EARLY NORMALIZED (FIRST 10%)
# ==========================================================

plt.figure(figsize=(8,6))

for name, (steps, losses) in processed.items():
    early_steps = []
    early_losses = []

    for s, l in zip(steps, losses):
        if s <= 0.1:  # first 10%
            early_steps.append(s)
            early_losses.append(l)

    plt.plot(early_steps, early_losses, label=name)

plt.xlabel("Training Progress (0 → 0.1)")
plt.ylabel("Training Loss (EMA)")
plt.title("Early Normalized Convergence (First 10%)")
plt.legend()
plt.grid(True)

plt.savefig("early_normalized_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 3️⃣ STABILITY (ROLLING VARIANCE)
# ==========================================================

plt.figure(figsize=(8,6))

window = 50

for name, (steps, losses) in processed.items():
    var = []
    var_steps = []

    for i in range(window, len(losses)):
        var.append(np.var(losses[i-window:i]))
        var_steps.append(steps[i])

    plt.plot(var_steps, var, label=name)

plt.xlabel("Training Progress")
plt.ylabel("Rolling Variance (window=50)")
plt.title("Training Stability Comparison")
plt.legend()
plt.grid(True)

plt.savefig("stability_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plots generated:")
print(" - full_normalized_convergence.png")
print(" - early_normalized_convergence.png")
print(" - stability_comparison.png")