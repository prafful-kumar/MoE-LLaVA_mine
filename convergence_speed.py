import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ---- Define your experiment folders ----
experiments = {
    "Random Init": "random_init_student_only_training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.0_0.0-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",
    
    "Fisher (No Norm)": "fisher_no_norm_input_norm_weight_during_training_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",
    
    "Fisher (TS + Norm)": "fisher_TS_norm_training_norm_input_norm_weight./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998/MoE-LLaVA-StableLM-Stage2-moe",
}

def smooth_ema(values, alpha=0.1):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed

plt.figure(figsize=(8,6))

for name, folder in experiments.items():
    path = os.path.join(folder, "trainer_state.json")
    
    with open(path) as f:
        data = json.load(f)
    
    log_history = data["log_history"]
    
    steps = []
    losses = []
    
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    
    losses = smooth_ema(losses, alpha=0.05)
    
    plt.plot(steps, losses, label=name)

plt.xlabel("Step")
plt.ylabel("Training Loss (EMA)")
plt.title("Convergence Comparison")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("loss_plot_trial.png")
plt.close()
print("completed")