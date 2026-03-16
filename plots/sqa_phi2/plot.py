import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "/scratch/prafull/MoE-LLaVA_mine/eval_results/sqa_checkpoints"
OUT_DIR = "/scratch/prafull/MoE-LLaVA_mine/plots/sqa_phi2"

variants = {
    "author": {"label": "Random (Author)", "color": "#1f77b4"},
    "student": {"label": "Student-Only (No Teacher)", "color": "#ff7f0e"},
}

STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def load_results(backbone, variant):
    steps, accs, img_accs = [], [], []
    for step in STEPS:
        path = os.path.join(RESULTS_DIR, f"{backbone}_{variant}_step{step}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            steps.append(step)
            accs.append(data["accuracy"])
            img_accs.append(data["img_accuracy"])
    return steps, accs, img_accs


# ---- Individual per variant ----
for vname, cfg in variants.items():
    steps, accs, img_accs = load_results("phi2", vname)
    if not steps:
        print(f"No results for phi2_{vname}, skipping")
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, accs, color=cfg["color"], linewidth=2.0, marker="o", markersize=6, label="Overall Accuracy")
    ax.plot(steps, img_accs, color=cfg["color"], linewidth=2.0, marker="s", markersize=6, linestyle="--", label="Image Accuracy")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Phi2 2.7B — {cfg['label']}\nScienceQA Accuracy vs Training Step", fontsize=14)
    ax.set_xticks(STEPS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{vname}_sqa.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {vname}_sqa.png")

# ---- Comparison: overall accuracy ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    steps, accs, _ = load_results("phi2", vname)
    if not steps:
        continue
    ax.plot(steps, accs, color=cfg["color"], linewidth=2.0, marker="o", markersize=6, label=cfg["label"])
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax.set_title("Phi2 2.7B — ScienceQA Accuracy vs Training Step", fontsize=14)
ax.set_xticks(STEPS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_sqa.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_sqa.png")

# ---- Comparison: image accuracy ----
fig, ax = plt.subplots(figsize=(10, 6))
for vname, cfg in variants.items():
    steps, _, img_accs = load_results("phi2", vname)
    if not steps:
        continue
    ax.plot(steps, img_accs, color=cfg["color"], linewidth=2.0, marker="s", markersize=6, label=cfg["label"])
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Image Accuracy (%)", fontsize=12)
ax.set_title("Phi2 2.7B — ScienceQA Image Accuracy vs Training Step", fontsize=14)
ax.set_xticks(STEPS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_sqa_img.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_sqa_img.png")
