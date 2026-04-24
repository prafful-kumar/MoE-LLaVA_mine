"""
Panel C: Training loss curves across initialization variants.

Reads trainer_state.json from each checkpoint.
Output: diagnostics/figures/training_loss.pdf + .png

Usage:
    python diagnostics/plot_loss.py [--output_dir diagnostics]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diagnostics.utils import VARIANT_COLORS

HPC = "/home/prafull/scratch/hpc"

VARIANTS = [
    {
        "label":  "Random init (author)",
        "path":   f"{HPC}/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "color":  VARIANT_COLORS["author"],
        "ls":     "--",
    },
    {
        "label":  "K-means init (student)",
        "path":   f"{HPC}/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "color":  VARIANT_COLORS["student"],
        "ls":     "-",
    },
    {
        "label":  "Teacher-Student (TS)",
        "path":   f"{HPC}/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "color":  VARIANT_COLORS["TS"],
        "ls":     "-",
    },
    {
        "label":  "New topk-entropy loss (ours)",
        "path":   f"{HPC}/checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe/trainer_state.json",
        "color":  VARIANT_COLORS["new_entropy"],
        "ls":     "-",
    },
]


def smooth_ema(values, alpha=0.05):
    ema = values[0]
    out = []
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def load_loss(path):
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None, None
    with open(path) as f:
        state = json.load(f)
    log = state["log_history"]
    steps  = [e["step"] for e in log if "loss" in e]
    losses = [e["loss"] for e in log if "loss" in e]
    return steps, losses


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_dir   = os.path.join(repo_root, args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Full training loss ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for v in VARIANTS:
        steps, losses = load_loss(v["path"])
        if steps is None:
            continue
        smoothed = smooth_ema(losses, alpha=0.05)
        for ax in axes:
            ax.plot(steps, smoothed, color=v["color"], linestyle=v["ls"],
                    linewidth=1.8, label=v["label"])

    axes[0].set_title("Full training (all steps)", fontsize=11)
    axes[0].set_xlabel("Step", fontsize=10)
    axes[0].set_ylabel("Training loss (EMA smoothed)", fontsize=10)
    axes[0].legend(fontsize=9, framealpha=0.9)

    axes[1].set_title("Early training (step 1–200)", fontsize=11)
    axes[1].set_xlabel("Step", fontsize=10)
    axes[1].set_xlim(0, 200)
    axes[1].legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"training_loss.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)

    # ── Key numbers ────────────────────────────────────────────────────────────
    print()
    for v in VARIANTS:
        steps, losses = load_loss(v["path"])
        if steps is None:
            continue
        step1_loss = losses[0] if losses else float("nan")
        final_loss = losses[-1] if losses else float("nan")
        print(f"{v['label']}: step-1 loss={step1_loss:.4f}, final loss={final_loss:.4f}, "
              f"total steps={steps[-1] if steps else 0}")


if __name__ == "__main__":
    main()
