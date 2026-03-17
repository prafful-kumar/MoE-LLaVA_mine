"""
Plot Qwen checkpoint results from eval_results_mine/qwen_checkpoint_results.xlsx
Skips any sheet with no data.
Output: plots_mine/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXCEL_FILE = "eval_results_mine/qwen_checkpoint_results.xlsx"
OUT_DIR    = "plots_mine"

VARIANTS = {
    "Author":  {"label": "Random (Author)",           "color": "#1f77b4"},
    "Student": {"label": "Student-Only (No Teacher)", "color": "#ff7f0e"},
    "TS":      {"label": "Teacher-Student (KD)",       "color": "#2ca02c"},
}

SHEET_META = {
    "mme_perception":   {"ylabel": "MME Perception Score"},
    "mme_cognition":    {"ylabel": "MME Cognition Score"},
    "mme_total":        {"ylabel": "MME Total Score"},
    "sqa_accuracy":     {"ylabel": "ScienceQA Accuracy (%)"},
    "sqa_img_accuracy": {"ylabel": "ScienceQA Img Accuracy (%)"},
    "gqa_accuracy":     {"ylabel": "GQA Accuracy (%)"},
}

xl = pd.ExcelFile(EXCEL_FILE)

for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)

    # Skip if no real data
    data_cols = [c for c in VARIANTS if c in df.columns]
    if df.empty or not data_cols or df[data_cols].dropna(how="all").empty:
        print(f"Skipping '{sheet_name}' (no data)")
        continue

    steps = df["step"].tolist()
    meta   = SHEET_META.get(sheet_name, {"ylabel": sheet_name})
    ylabel = meta["ylabel"]

    # ── Individual per-variant ───────────────────────────────────────────────
    for var, vcfg in VARIANTS.items():
        if var not in df.columns or df[var].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, df[var].tolist(), marker="o", color=vcfg["color"],
                linewidth=2, label=vcfg["label"])
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Qwen 1.8B – {vcfg['label']}\n{ylabel} vs Training Step")
        ax.set_xticks(steps)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{var.lower()}_{sheet_name}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

    # ── Comparison: all variants ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for var, vcfg in VARIANTS.items():
        if var not in df.columns or df[var].dropna().empty:
            continue
        ax.plot(steps, df[var].tolist(), marker="o", color=vcfg["color"],
                linewidth=2, label=vcfg["label"])
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Qwen 1.8B – All Variants\n{ylabel} vs Training Step")
    ax.set_xticks(steps)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"all_{sheet_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
