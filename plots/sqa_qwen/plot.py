import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXCEL_FILE = "excel_results/checkpoint_sqa_qwen.xlsx"
OUT_DIR = "plots/sqa_qwen"

variants = {
    "author":          {"label": "Random (Author)",          "color": "#1f77b4"},
    "student":         {"label": "Student-Only (No Teacher)", "color": "#ff7f0e"},
    "teacher_student": {"label": "Teacher-Student (KD)",      "color": "#2ca02c"},
}

STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Load from Excel
df_all = pd.read_excel(EXCEL_FILE, sheet_name="All")

def load_results(variant):
    df = df_all[df_all["variant"] == variant].sort_values("step")
    return df["step"].tolist(), df["accuracy"].tolist(), df["img_accuracy"].tolist()


# ---- Individual per variant ----
for vname, cfg in variants.items():
    steps, accs, img_accs = load_results(vname)
    if not steps:
        print(f"No results for qwen_{vname}, skipping")
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, accs, color=cfg["color"], linewidth=2.0, marker="o", markersize=6, label="Overall Accuracy")
    ax.plot(steps, img_accs, color=cfg["color"], linewidth=2.0, marker="s", markersize=6, linestyle="--", label="Image Accuracy")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Qwen 1.8B — {cfg['label']}\nScienceQA Accuracy vs Training Step", fontsize=14)
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
    steps, accs, _ = load_results(vname)
    if not steps:
        continue
    ax.plot(steps, accs, color=cfg["color"], linewidth=2.0, marker="o", markersize=6, label=cfg["label"])
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax.set_title("Qwen 1.8B — ScienceQA Accuracy vs Training Step", fontsize=14)
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
    steps, _, img_accs = load_results(vname)
    if not steps:
        continue
    ax.plot(steps, img_accs, color=cfg["color"], linewidth=2.0, marker="s", markersize=6, label=cfg["label"])
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Image Accuracy (%)", fontsize=12)
ax.set_title("Qwen 1.8B — ScienceQA Image Accuracy vs Training Step", fontsize=14)
ax.set_xticks(STEPS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_sqa_img.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved all_sqa_img.png")
