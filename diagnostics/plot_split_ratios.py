"""
Experiment 3: Plot routing split ratio histograms + CDF of secondary weight p̃(2).

Input:
    diagnostics/data/split_ratios_{label}.npz  for each label

Output:
    diagnostics/figures/split_ratio_histogram.pdf + .png
    diagnostics/figures/secondary_weight_cdf.pdf + .png

Usage:
    python diagnostics/plot_split_ratios.py \
        --labels author entropy_old new_entropy \
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
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diagnostics.utils import VARIANT_COLORS

# Display names for legend
DISPLAY_NAMES = {
    "author":           "Random init (author)",
    "student":          "K-means init (student)",
    "entropy_old":      "Old entropy (raw H, w=0.01)",
    "entropy":          "Old entropy (raw H, w=0.01)",
    "entropy_w01":      "Old entropy (raw H, w=0.1)",
    "new_entropy":      "Topk-entropy loss (no L_var)",
    "entropy_topk_var": "Topk-entropy + L_var (ours)",
    "TS":               "Teacher-Student",
}

# Colors: fall back to author color if label not in VARIANT_COLORS
def get_color(label):
    for key in [label, label.replace("_old", ""), label.replace("entropy_old", "entropy"),
                label.split("_")[0]]:
        if key in VARIANT_COLORS:
            return VARIANT_COLORS[key]
    return "#888888"


def plot_histogram(args, data_dir, fig_dir, out_tag=""):
    """Plot histogram of split_ratio = p_top1 / (p_top1 + p_top2)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    bins = np.arange(0.5, 1.025, 0.025)

    found_any = False
    data_tag = getattr(args, "_data_tag", "")
    for label in args.labels:
        tag_str = f"_{data_tag}" if data_tag else ""
        path = os.path.join(data_dir, f"split_ratios_{label}{tag_str}.npz")
        if not os.path.exists(path):
            warnings.warn(f"Missing: {path}")
            continue
        found_any = True

        data = np.load(path)
        ratios = data["split_ratios_flat"]
        color  = get_color(label)
        name   = DISPLAY_NAMES.get(label, label)

        counts, _ = np.histogram(ratios, bins=bins)
        density = counts / counts.sum()
        ax.bar(
            bins[:-1] + 0.0125,   # centre of each bin
            density,
            width=0.022,
            color=color,
            alpha=0.55,
            label=name,
            edgecolor="white",
            linewidth=0.4,
        )

        print(f"\n── Split ratio stats for '{label}' ({name}) ──────────────────")
        print(f"  % near-collapse (>0.9):  {100*(ratios > 0.9).mean():.1f}%")
        print(f"  % near-balanced (<0.6):  {100*(ratios < 0.6).mean():.1f}%")
        print(f"  Median split ratio:      {float(np.median(ratios)):.3f}")
        print(f"  Mean split ratio:        {float(ratios.mean()):.3f}")

    if not found_any:
        print("No data found. Run collect_split_ratios.py first.")
        plt.close(fig)
        return

    ax.axvline(0.5, color="green",  linestyle="--", linewidth=1.2,
               label="Ideal (equal split)")
    ax.axvline(1.0, color="red",    linestyle="--", linewidth=1.2,
               label="Collapsed (one expert)")

    ax.set_xlabel("Split ratio  (max weight / sum of top-2 weights)", fontsize=11)
    ax.set_ylabel("Fraction of routing decisions", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0.48, 1.02)
    ax.set_title("Top-2 routing split ratio distribution", fontsize=11)
    ax.legend(loc="upper center", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    tag_str = f"_{out_tag}" if out_tag else ""
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"split_ratio_histogram{tag_str}.{ext}")
        fig.savefig(p, dpi=300)
        print(f"\nSaved: {p}")
    plt.close(fig)


def plot_secondary_weight_cdf(args, data_dir, fig_dir, out_tag=""):
    """
    CDF of p̃(2) — the renormalized weight assigned to the secondary (2nd) expert.

    p̃(2) = p_top2 / (p_top1 + p_top2) = 1 − split_ratio
    (since split_ratio = p_top1 / (p_top1 + p_top2) with top1 ≥ top2)

    Values near 0 → dead second expert (within-k collapse).
    Values near 0.5 → balanced top-2 usage.

    Output: diagnostics/figures/secondary_weight_cdf.pdf + .png
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    found_any = False
    xs = np.linspace(0.0, 0.5, 500)
    data_tag = getattr(args, "_data_tag", "")

    for label in args.labels:
        tag_str = f"_{data_tag}" if data_tag else ""
        path = os.path.join(data_dir, f"split_ratios_{label}{tag_str}.npz")
        if not os.path.exists(path):
            warnings.warn(f"Missing: {path}")
            continue
        found_any = True

        data = np.load(path)
        split_ratios = data["split_ratios_flat"]
        p_tilde_2    = 1.0 - split_ratios          # secondary weight ∈ [0, 0.5]

        # CDF
        cdf = np.array([(p_tilde_2 <= x).mean() for x in xs])

        color     = get_color(label)
        name      = DISPLAY_NAMES.get(label, label)
        linestyle = "--" if label == "author" else "-"
        ax.plot(xs, cdf, color=color, linestyle=linestyle, linewidth=1.8, label=name)

        dead_frac = (p_tilde_2 < 0.05).mean()
        print(f"\n── p̃(2) stats for '{label}' ──────────────────────────────────")
        print(f"  % dead second expert (p̃(2) < 0.05): {100*dead_frac:.1f}%")
        print(f"  % near-balanced (p̃(2) > 0.4):       {100*(p_tilde_2 > 0.4).mean():.1f}%")
        print(f"  Median p̃(2): {float(np.median(p_tilde_2)):.3f}")
        print(f"  Mean   p̃(2): {float(p_tilde_2.mean()):.3f}")

        # Annotate dead-expert fraction on plot at x=0.05
        ax.annotate(
            f"{100*dead_frac:.0f}%",
            xy=(0.05, dead_frac),
            xytext=(0.09, dead_frac - 0.05),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    if not found_any:
        print("No data found. Run collect_split_ratios.py first.")
        plt.close(fig)
        return

    ax.axvline(0.05, color="red",   linestyle=":", linewidth=1.0,
               label="Dead threshold (p̃(2) < 0.05)")
    ax.axvline(0.5,  color="green", linestyle=":", linewidth=1.0,
               label="Ideal (equal split)")

    ax.set_xlabel("Secondary routing weight  p̃(2) = p_top2 / (p_top1 + p_top2)", fontsize=10)
    ax.set_ylabel("Cumulative fraction of routing decisions", fontsize=10)
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(labelsize=9)
    ax.set_title("CDF of secondary expert weight  (lower = more within-k collapse)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    tag_str = f"_{out_tag}" if out_tag else ""
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"secondary_weight_cdf{tag_str}.{ext}")
        fig.savefig(p, dpi=300)
        print(f"\nSaved: {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+",
                        default=["author", "entropy_topk_var"],
                        help="Labels to plot (must match split_ratios_{label}_{data_tag}.npz files)")
    parser.add_argument("--data_dir",   default="diagnostics/data")
    parser.add_argument("--output_dir", default="diagnostics")
    parser.add_argument("--data_tag", default="",
                        help="Tag in input filenames, e.g. 'sqa1000' → split_ratios_{label}_sqa1000.npz")
    parser.add_argument("--out_tag", default="",
                        help="Tag appended to output figure filenames, e.g. 'sqa1000'")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir  = os.path.join(repo_root, args.data_dir)
    fig_dir   = os.path.join(repo_root, args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Rewrite labels to include data_tag so load paths resolve correctly
    if args.data_tag:
        # Store original labels; patch the path search inside plot functions via args
        args._data_tag = args.data_tag
    else:
        args._data_tag = ""

    plot_histogram(args, data_dir, fig_dir, out_tag=args.out_tag)
    plot_secondary_weight_cdf(args, data_dir, fig_dir, out_tag=args.out_tag)


if __name__ == "__main__":
    main()
