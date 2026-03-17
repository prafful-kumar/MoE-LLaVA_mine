"""
MME Perception score vs training step for Qwen (1.8B) variants.
Primary metric: perception score (matches paper convention, ~1291 for full training).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEPS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EXCEL_FILE = "excel_results/checkpoint_mme_qwen.xlsx"
OUT_DIR = "plots/mme_qwen"

# Load from Excel
df_all = pd.read_excel(EXCEL_FILE, sheet_name="All")

VARIANTS = {
    "author":          {"label": "Random (Author)",          "color": "#1f77b4"},
    "student":         {"label": "Student-Only (No Teacher)", "color": "#ff7f0e"},
    "teacher_student": {"label": "Teacher-Student (KD)",      "color": "#2ca02c"},
}

# Paper reference for full training (perception only)
PAPER_PERCEPTION = 1291.6

def load_scores(variant):
    df = df_all[df_all["variant"] == variant].sort_values("step")
    perc = df["perception"].tolist()
    cog  = df["cognition"].tolist()
    return perc, cog


# ── Individual plots ──────────────────────────────────────────────────────────
for vname, vmeta in VARIANTS.items():
    perc, cog = load_scores(vname)
    total = [p + c for p, c in zip(perc, cog)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"MME – {vmeta['label']} (Qwen 1.8B)", fontsize=13)

    for ax, scores, title, paper_val in zip(
        axes,
        [perc, cog, total],
        ["Perception Score", "Cognition Score", "Total Score"],
        [PAPER_PERCEPTION, None, None]
    ):
        ax.plot(STEPS, scores, marker='o', color=vmeta["color"], linewidth=2)
        if paper_val is not None:
            ax.axhline(paper_val, color='red', linestyle='--', linewidth=1,
                       label=f"Paper ({paper_val})")
            ax.legend(fontsize=9)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(STEPS)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{vname}_mme.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

# ── Comparison: perception (main metric) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for vname, vmeta in VARIANTS.items():
    perc, _ = load_scores(vname)
    ax.plot(STEPS, perc, marker='o', color=vmeta["color"],
            linewidth=2, label=vmeta["label"])

ax.axhline(PAPER_PERCEPTION, color='red', linestyle='--', linewidth=1.2,
           label=f"Paper – Qwen author ({PAPER_PERCEPTION})")
ax.set_xlabel("Training Step")
ax.set_ylabel("MME Perception Score")
ax.set_title("MME Perception – Qwen 1.8B Variants (early checkpoints)")
ax.set_xticks(STEPS)
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "all_mme_perception.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# ── Comparison: cognition ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for vname, vmeta in VARIANTS.items():
    _, cog = load_scores(vname)
    ax.plot(STEPS, cog, marker='o', color=vmeta["color"],
            linewidth=2, label=vmeta["label"])

ax.set_xlabel("Training Step")
ax.set_ylabel("MME Cognition Score")
ax.set_title("MME Cognition – Qwen 1.8B Variants (early checkpoints)")
ax.set_xticks(STEPS)
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "all_mme_cognition.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
