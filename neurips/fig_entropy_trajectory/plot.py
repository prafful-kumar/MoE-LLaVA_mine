"""
neurips/fig_entropy_trajectory/plot.py
=======================================
Publication-quality figures: routing entropy over training steps.

Shows that Fisher initialization (student) drives routing entropy
down from step 1, while random initialization (author) stays
near-maximum entropy throughout.

Reference lines:
  ln(4) ≈ 1.386  — uniform over all 4 experts (maximum entropy)
  ln(2) ≈ 0.693  — ideal top-2: equal weight to exactly 2 experts

Usage:
    python neurips/fig_entropy_trajectory/plot.py \\
        --files neurips/fig_entropy_trajectory/data/qwen_author.json \\
                neurips/fig_entropy_trajectory/data/qwen_student.json \\
        --labels "Author (random init)" "Student (Fisher init)" \\
        --max_step 1000
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_ENT  = math.log(4)   # 1.386
IDEAL_K2 = math.log(2)   # 0.693
REF_COL  = "#888888"

# Baseline → dashed, proposed → solid
COLORS  = ["#E07A0F", "#1D9E75", "#2563EB", "#9333EA"]
LSTYLES = ["--", "-", "-.", ":"]


def load(path):
    with open(path) as f:
        return json.load(f)


def filter_steps(d, max_step):
    """Keep only steps <= max_step."""
    pairs = [(s, h) for s, h in zip(d["steps"], d["mean_H"]) if s <= max_step]
    if not pairs:
        return d
    steps, mean_H = zip(*pairs)
    # filter layer_H too
    layer_H = {}
    for k, vals in d["layer_H"].items():
        filtered = [v for s, v in zip(d["steps"], vals) if s <= max_step]
        if filtered:
            layer_H[k] = filtered
    return {**d, "steps": list(steps), "mean_H": list(mean_H), "layer_H": layer_H}


def setup():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.family":  "sans-serif",
        "font.size":    11,
        "figure.dpi":   300,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — Model-level mean entropy over steps
# ─────────────────────────────────────────────────────────────────────────────

def plot_v1_mean(all_data, labels, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Subtle reference lines
    ax.axhline(MAX_ENT,  color=REF_COL, ls="--", lw=1.2, alpha=0.7,
               label=f"Max entropy  ln(4) = {MAX_ENT:.2f}")
    ax.axhline(IDEAL_K2, color=REF_COL, ls=":",  lw=1.2, alpha=0.7,
               label=f"Ideal top-2  ln(2) = {IDEAL_K2:.2f}")

    # Light shading for zones
    ax.axhspan(IDEAL_K2, MAX_ENT, color="#FEF3C7", alpha=0.10, zorder=0)
    ax.axhspan(0,        IDEAL_K2, color="#D1FAE5", alpha=0.10, zorder=0)

    for d, label, color, ls in zip(all_data, labels, COLORS, LSTYLES):
        ax.plot(d["steps"], d["mean_H"],
                color=color, lw=2.5, ls=ls, marker="o", ms=4,
                label=label, zorder=3)

    sns.despine(ax=ax)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Mean Routing Entropy H (nats)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-0.05, MAX_ENT + 0.18)
    ax.set_title("Routing Entropy vs Training Step", fontsize=15, pad=10)
    ax.legend(fontsize=11, frameon=False,
              bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v1_mean_entropy.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — Per-layer heatmap: first vs last checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def plot_v2_first_last_heatmap(all_data, labels, out_dir):
    n_models = len(all_data)
    fig, axes = plt.subplots(n_models, 1,
                             figsize=(10, 2.8 * n_models),
                             squeeze=False)

    im = None
    for row, (d, label) in enumerate(zip(all_data, labels)):
        ax = axes[row, 0]
        layer_keys = sorted(d["layer_H"].keys(), key=int)
        first_H = [d["layer_H"][k][0]  for k in layer_keys]
        last_H  = [d["layer_H"][k][-1] for k in layer_keys]
        mat = np.array([first_H, last_H])  # [2, n_layers]

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=MAX_ENT, interpolation="nearest")
        fig.colorbar(im, cax=cax, label="H (nats)")

        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Step {d['steps'][0]}", f"Step {d['steps'][-1]}"],
                           fontsize=11)
        ax.set_xticks(range(len(layer_keys)))
        ax.set_xticklabels([str(k) for k in layer_keys], fontsize=11)
        ax.set_xlabel("MoE Layer", fontsize=13)
        ax.set_title(label, fontsize=13, pad=6)

        for col_idx, (h0, h1) in enumerate(zip(first_H, last_H)):
            ax.text(col_idx, 0, f"{h0:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
            ax.text(col_idx, 1, f"{h1:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.suptitle("Routing Entropy: First vs Last Checkpoint per Layer",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v2_first_last_heatmap.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3 — Per-layer line trajectories (small multiples)
# ─────────────────────────────────────────────────────────────────────────────

def plot_v3_per_layer(all_data, labels, out_dir):
    layer_keys = sorted(all_data[0]["layer_H"].keys(), key=int)
    n_layers = len(layer_keys)
    ncols = 4
    nrows = math.ceil(n_layers / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 2.4),
                             sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for li, lk in enumerate(layer_keys):
        ax = axes_flat[li]
        for d, label, color, ls in zip(all_data, labels, COLORS, LSTYLES):
            if lk in d["layer_H"]:
                ax.plot(d["steps"], d["layer_H"][lk],
                        color=color, lw=2.0, ls=ls, marker="o", ms=2.5,
                        label=label if li == 0 else None)
        ax.axhline(MAX_ENT,  color=REF_COL, ls="--", lw=0.8, alpha=0.6)
        ax.axhline(IDEAL_K2, color=REF_COL, ls=":",  lw=0.8, alpha=0.6)
        ax.set_title(f"Layer {lk}", fontsize=11)
        ax.set_ylim(-0.05, MAX_ENT + 0.12)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    for li in range(n_layers, len(axes_flat)):
        axes_flat[li].set_visible(False)

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, fontsize=11, frameon=False,
               loc="lower center", ncol=len(all_data),
               bbox_to_anchor=(0.5, -0.03))

    fig.supxlabel("Training Step", fontsize=13, y=-0.02)
    fig.supylabel("Routing Entropy H (nats)", fontsize=13, x=-0.01)
    fig.suptitle("Per-Layer Routing Entropy Trajectory", fontsize=15, y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v3_per_layer.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",    nargs="+", required=True)
    parser.add_argument("--labels",   nargs="+", required=True)
    parser.add_argument("--max_step", type=int, default=1000,
                        help="Exclude checkpoints beyond this step (default: 1000)")
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        print("ERROR: --files and --labels must have the same length")
        return

    all_data, used_labels = [], []
    for path, label in zip(args.files, args.labels):
        fpath = path if os.path.isabs(path) else os.path.join(REPO, path)
        if not os.path.exists(fpath):
            print(f"WARNING: not found — {fpath}")
            continue
        d = filter_steps(load(fpath), args.max_step)
        all_data.append(d)
        used_labels.append(label)
        print(f"{label}: steps={d['steps']}  mean_H={[round(h, 3) for h in d['mean_H']]}")

    if not all_data:
        print("No data loaded.")
        return

    setup()
    plot_v1_mean(all_data, used_labels, OUT_DIR)
    plot_v2_first_last_heatmap(all_data, used_labels, OUT_DIR)
    plot_v3_per_layer(all_data, used_labels, OUT_DIR)
    print(f"\nAll figures saved to neurips/fig_entropy_trajectory/figures/")


if __name__ == "__main__":
    main()
