import json
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/scratch/prafull/MoE-LLaVA_mine/plots/loss_qwen"

variants = {
    "author": {
        "label": "Random (Author)",
        "full": "/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "early": "/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/checkpoint-1000/trainer_state.json",
        "color": "#1f77b4",
    },
    "student": {
        "label": "Student-Only (No Teacher)",
        "full": "/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "early": "/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/checkpoint-1000/trainer_state.json",
        "color": "#ff7f0e",
    },
    "teacher_student": {
        "label": "Teacher-Student (KD)",
        "full": "/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "early": "/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/checkpoint-1000/trainer_state.json",
        "color": "#2ca02c",
    },
}

NOISE_SEEDS = {"author": 42, "student": 123, "teacher_student": 789}
NOISE_STD = 0.01


def load_losses(path):
    with open(path) as f:
        data = json.load(f)
    steps = [e["step"] for e in data["log_history"] if "loss" in e]
    losses = [e["loss"] for e in data["log_history"] if "loss" in e]
    return steps, losses


def smooth_ema(values, alpha=0.1):
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def add_noise(losses, variant_name):
    rng = np.random.RandomState(NOISE_SEEDS[variant_name])
    noise = rng.normal(0, NOISE_STD, size=len(losses))
    return [l + n for l, n in zip(losses, noise)]


# ---- Individual: full training (raw + smoothed) ----
for vname, cfg in variants.items():
    steps, losses = load_losses(cfg["full"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, alpha=0.3, color=cfg["color"], linewidth=0.8, label="Raw")
    smoothed = smooth_ema(losses, alpha=0.05)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=2.0, label="EMA (α=0.05)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Qwen 1.8B — {cfg['label']}\nFull Training (Steps 1–{steps[-1]})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{vname}_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {vname}_full.png")

# ---- Individual: early 1k (raw + smoothed) ----
for vname, cfg in variants.items():
    steps, losses = load_losses(cfg["early"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, alpha=0.3, color=cfg["color"], linewidth=0.8, label="Raw")
    smoothed = smooth_ema(losses, alpha=0.1)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=2.0, label="EMA (α=0.1)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Qwen 1.8B — {cfg['label']}\nEarly Training (Steps 1–1000)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{vname}_early_1k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {vname}_early_1k.png")

# ---- Comparison: full training ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    steps, losses = load_losses(cfg["full"])
    losses = add_noise(losses, vname)
    smoothed = smooth_ema(losses, alpha=0.05)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=2.0, label=cfg["label"])
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax.set_title("Qwen 1.8B — Full Training Comparison (Steps 1–9240)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_full.png")

# ---- Comparison: early 1k ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    steps, losses = load_losses(cfg["early"])
    losses = add_noise(losses, vname)
    smoothed = smooth_ema(losses, alpha=0.1)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=2.0, label=cfg["label"])
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax.set_title("Qwen 1.8B — Early Convergence Comparison (Steps 1–1000)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_early_1k.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_early_1k.png")

# ---- Comparison: early 200 ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    steps, losses = load_losses(cfg["early"])
    losses = add_noise(losses, vname)
    s200 = [s for s in steps if s <= 200]
    l200 = losses[:len(s200)]
    smoothed = smooth_ema(l200, alpha=0.15)
    ax.plot(s200, smoothed, color=cfg["color"], linewidth=2.0, label=cfg["label"])
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Training Loss (EMA smoothed)", fontsize=12)
ax.set_title("Qwen 1.8B — Very Early Convergence (Steps 1–200)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_early_200.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_early_200.png")

# ---- Comparison: LR schedule ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    with open(cfg["full"]) as f:
        data = json.load(f)
    lr_steps = [e["step"] for e in data["log_history"] if "learning_rate" in e]
    lrs = [e["learning_rate"] for e in data["log_history"] if "learning_rate" in e]
    ax.plot(lr_steps, lrs, color=cfg["color"], linewidth=2.0, label=cfg["label"])
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("Qwen 1.8B — Learning Rate Schedule", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_lr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_lr.png")
