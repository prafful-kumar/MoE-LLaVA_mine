"""
Panel D: Expert utilization heatmap.

Input:
    diagnostics/data/utilization_{label_a}.npz
    diagnostics/data/utilization_{label_b}.npz

Output:
    diagnostics/figures/expert_utilization_heatmap.pdf + .png

Two-row heatmap:
  - Row 1: baseline (e.g. author / dot-product)
  - Row 2: new variant (e.g. new_entropy / cosine + topk-entropy)
  - x-axis: MoE layer index
  - y-axis: expert index (0 to E-1)
  - Color: white = 0%, reference grey = 1/E (uniform), red = ≥ 40% (overloaded)
  - White dashed line marks the 1/E uniform reference

Usage:
    python diagnostics/plot_utilization.py \
        --label_a author \
        --label_b new_entropy \
        --data_dir diagnostics/data \
        --output_dir diagnostics
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DISPLAY_NAMES = {
    "author":            "Random init (dot-product)",
    "student":           "K-means init (student)",
    "entropy_old":       "Old entropy (raw H, w=0.01)",
    "new_entropy":       "New topk-entropy loss",
    "entropy_topk_var":  "Topk-entropy + L_var (ours)",
    "TS":                "Teacher-Student",
}


def load_utilization(path):
    if not os.path.exists(path):
        warnings.warn(f"Missing: {path}")
        return None, None
    d = np.load(path)
    return d["layer_indices"], d["expert_fractions"]   # [N_layers], [N_layers, N_experts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_a",    default="author",
                        help="Baseline label (matches utilization_{label}.npz)")
    parser.add_argument("--label_b",    default="new_entropy",
                        help="New variant label")
    parser.add_argument("--data_dir",   default="diagnostics/data")
    parser.add_argument("--output_dir", default="diagnostics")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir  = os.path.join(repo_root, args.data_dir)
    fig_dir   = os.path.join(repo_root, args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    path_a = os.path.join(data_dir, f"utilization_{args.label_a}.npz")
    path_b = os.path.join(data_dir, f"utilization_{args.label_b}.npz")

    layer_indices_a, fractions_a = load_utilization(path_a)
    layer_indices_b, fractions_b = load_utilization(path_b)

    if fractions_a is None and fractions_b is None:
        print("No utilization data found. Run collect_utilization.py first.")
        return

    # Use whichever is available; both should be present for meaningful comparison
    n_experts = (fractions_a if fractions_a is not None else fractions_b).shape[1]
    uniform   = 1.0 / n_experts

    # Colormap: white (0%) → light yellow → orange → red (≥ 2× uniform)
    # Clamp at 2× uniform so the colormap is informative
    vmax = max(2.0 * uniform, 0.40)   # at least 40% as ceiling
    cmap = plt.cm.YlOrRd

    name_a = DISPLAY_NAMES.get(args.label_a, args.label_a)
    name_b = DISPLAY_NAMES.get(args.label_b, args.label_b)

    n_rows = (1 if fractions_a is None else 1) + (1 if fractions_b is None else 1)
    n_rows = 2  # always 2 rows; skip if data missing
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4), sharex=True)
    if n_rows == 1:
        axes = [axes]

    plt.style.use("seaborn-v0_8-whitegrid")

    def draw_heatmap(ax, fractions, layer_indices, title):
        if fractions is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(title, fontsize=10)
            return

        # fractions: [N_layers, N_experts] — plot as heatmap
        # x = layer index, y = expert index (0 at bottom)
        data = fractions.T * 100.0   # [N_experts, N_layers], in percent

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax * 100.0,
            interpolation="nearest",
        )

        # Mark uniform reference line
        uniform_pct = uniform * 100.0
        ax.axhline(y=-0.5 + n_experts * uniform_pct / (vmax * 100.0) * n_experts,
                   color="white", linestyle="--", linewidth=1.2, alpha=0.8)

        # x-ticks = layer indices, y-ticks = expert indices
        ax.set_yticks(range(n_experts))
        ax.set_yticklabels([f"E{e}" for e in range(n_experts)], fontsize=9)
        ax.set_xticks(range(len(layer_indices)))
        ax.set_xticklabels(layer_indices, fontsize=8, rotation=45)
        ax.set_title(title, fontsize=10, pad=4)

        return im

    im_a = draw_heatmap(axes[0], fractions_a, layer_indices_a,
                        f"{name_a}  (baseline)")
    im_b = draw_heatmap(axes[1], fractions_b, layer_indices_b,
                        f"{name_b}  (ours)")

    # Shared x-label
    axes[-1].set_xlabel("MoE layer index", fontsize=10)

    # Shared y-label on left
    fig.text(0.01, 0.5, "Expert", va="center", rotation="vertical", fontsize=10)

    # Shared colorbar
    ref_im = im_b if im_b is not None else im_a
    if ref_im is not None:
        cbar = fig.colorbar(ref_im, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("% of token assignments", fontsize=9)
        cbar.ax.axhline(y=uniform * 100.0, color="white", linestyle="--",
                        linewidth=1.5, label=f"Uniform ({uniform*100:.0f}%)")

    fig.suptitle("Expert utilization per layer  (uniform = "
                 f"{uniform*100:.0f}%, red ≥ {vmax*100:.0f}%)",
                 fontsize=11, y=1.01)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"expert_utilization_heatmap.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)

    # Key numbers
    for fractions, label in [(fractions_a, args.label_a), (fractions_b, args.label_b)]:
        if fractions is None:
            continue
        print(f"\n── Utilization summary for '{label}' ────────────────────────────")
        print(f"  Uniform reference: {100*uniform:.1f}% per expert")
        print(f"  Max expert fraction: {100*fractions.max():.1f}%")
        print(f"  Min expert fraction: {100*fractions.min():.1f}%")
        overloaded_layers = (fractions.max(axis=1) > 2 * uniform).sum()
        print(f"  Layers with max expert > 2× uniform: {overloaded_layers} / {fractions.shape[0]}")
        print(f"  Mean std (across experts, per layer): {fractions.std(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
