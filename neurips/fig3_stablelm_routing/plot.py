"""
neurips/fig3_stablelm_routing/plot.py
========================================
Figure 2: Renormalized top-2 max probability histogram.

Metric: split_ratio = max(p_top1, p_top2) / (p_top1 + p_top2)  ∈ [0.5, 1.0]
  0.5 → both selected experts get equal weight  (ambiguous / sharing token)
  1.0 → one expert gets all weight             (confident / committed token)

Expected shapes:
  Author (aux loss, H≈1.38):    spike near 0.5 — router is uncertain, hedges on every token
  entropy_topk_var (collapsed): spike near 1.0 — router over-commits, second expert dead
  Adaptive entropy (ours):      BIMODAL — peak near 0.5 (sharing) AND near 1.0 (confident)
    → proves the router learned to treat tokens differently based on complexity

Data sources:
  Existing diagnostics data works directly:
    diagnostics/data/split_ratios_author_sqa1000.npz
    diagnostics/data/split_ratios_entropy_topk_var_sqa1000.npz
  New checkpoint data from collect.py:
    neurips/fig3_stablelm_routing/data/{label}.npz

Usage:
    # With existing diagnostics data
    python neurips/fig3_stablelm_routing/plot.py \\
        --files diagnostics/data/split_ratios_author_sqa1000.npz \\
                diagnostics/data/split_ratios_entropy_topk_var_sqa1000.npz \\
        --labels "Author (aux loss)" "Topk-entropy + L_var"

    # Once adaptive checkpoint is collected
    python neurips/fig3_stablelm_routing/plot.py \\
        --files diagnostics/data/split_ratios_author_sqa1000.npz \\
                neurips/fig3_stablelm_routing/data/adaptive_entropy.npz \\
        --labels "Author (aux loss)" "Adaptive entropy (ours)"
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Project-consistent colors
COLORS = ["#E07A0F", "#2563EB", "#059669", "#9333EA", "#DB2777"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (3,1,1,1))]


def setup():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.3,
        "grid.linewidth":    0.6,
    })


def load_split_ratios(path):
    if not os.path.exists(path):
        warnings.warn(f"Not found: {path}")
        return None
    d = np.load(path)
    return d["split_ratios_flat"]


def stats_text(sr):
    return (
        f"n = {len(sr)/1e6:.1f}M\n"
        f"median = {np.median(sr):.3f}\n"
        f"≤0.55 (share): {100*(sr<0.55).mean():.0f}%\n"
        f"≥0.90 (commit): {100*(sr>=0.9).mean():.0f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — Side-by-side histograms, one panel per model
# ─────────────────────────────────────────────────────────────────────────────

def plot_v1_side_by_side(all_ratios, labels, out_dir):
    n = len(all_ratios)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharey=False)
    if n == 1:
        axes = [axes]

    bins = np.linspace(0.5, 1.0, 55)

    for ax, sr, label, color in zip(axes, all_ratios, labels, COLORS):
        ax.hist(sr, bins=bins, density=True, color=color, alpha=0.78,
                edgecolor="white", linewidth=0.25)

        med = float(np.median(sr))
        ax.axvline(med, color=color, ls="--", lw=1.2, label=f"Median = {med:.3f}")

        ax.text(0.97, 0.97, stats_text(sr), transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB", lw=0.7))

        ax.set_xlabel("Renormalized top-2 max prob\n"
                      r"$\tilde{p}_1 = \max(p_{(1)}, p_{(2)}) / (p_{(1)} + p_{(2)})$",
                      fontsize=9)
        ax.set_xlim(0.48, 1.02)
        ax.set_title(label, fontsize=9, pad=4)
        ax.legend(fontsize=8, framealpha=0.9, loc="upper left")

    axes[0].set_ylabel("Density", fontsize=9.5)

    fig.suptitle(
        "Routing confidence: renormalized top-2 probability per token  "
        "(0.5 = sharing, 1.0 = committed)",
        fontsize=9.5, y=1.02,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v1_side_by_side.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — Overlay on single axis (best for 2–3 models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_v2_overlay(all_ratios, labels, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bins = np.linspace(0.5, 1.0, 55)

    for i, (sr, label, color) in enumerate(zip(all_ratios, labels, COLORS)):
        ax.hist(sr, bins=bins, density=True, color=color, alpha=0.55,
                edgecolor=color, linewidth=0.5, label=label)
        med = float(np.median(sr))
        ax.axvline(med, color=color, ls="--", lw=1.0)

    # Zone annotations
    ax.axvspan(0.5, 0.6,  color="#D1FAE5", alpha=0.3, zorder=0)
    ax.axvspan(0.9, 1.01, color="#FEE2E2", alpha=0.3, zorder=0)
    ax.text(0.507, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1,
            "Sharing\nzone", fontsize=7.5, color="#065F46", style="italic", va="top")

    ax.set_xlabel(r"Renormalized top-2 max prob  $\tilde{p}_1$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(0.48, 1.02)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.92)
    ax.set_title(
        "Router confidence distribution — spike at 0.5 = uncertain, 1.0 = collapsed",
        fontsize=9.5, pad=6,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v2_overlay.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3 — Annotated overlay with bimodality callouts (paper-ready)
# ─────────────────────────────────────────────────────────────────────────────

def plot_v3_annotated(all_ratios, labels, out_dir):
    """
    Clean version intended for the paper. Shades sharing and committed zones.
    For any model that appears bimodal, annotates both peaks.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    bins = np.linspace(0.5, 1.005, 60)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for i, (sr, label, color, ls) in enumerate(zip(all_ratios, labels, COLORS, LINE_STYLES)):
        counts, _ = np.histogram(sr, bins=bins, density=True)
        ax.plot(bin_centers, counts, color=color, lw=1.8, ls=ls, label=label)
        ax.fill_between(bin_centers, counts, alpha=0.12, color=color)

        med = float(np.median(sr))
        # Find density at median bin for annotation placement
        med_bin = np.searchsorted(bin_centers, med)
        med_y = counts[min(med_bin, len(counts)-1)]
        ax.annotate(f"median={med:.2f}",
                    xy=(med, med_y), xytext=(med + 0.04, med_y + counts.max()*0.12),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    # Zone shading
    ax.axvspan(0.495, 0.565, color="#DCFCE7", alpha=0.5, zorder=0)
    ax.axvspan(0.88,  1.005, color="#FEF2F2", alpha=0.5, zorder=0)
    ax.axvline(0.5,  color="#15803D", ls=":", lw=0.9, alpha=0.7)
    ax.axvline(1.0,  color="#B91C1C", ls=":", lw=0.9, alpha=0.7)

    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10
    ax.text(0.50, ymax * 0.07, "← both experts\n    equal weight",
            fontsize=7.5, color="#15803D", ha="left", style="italic")
    ax.text(0.99, ymax * 0.07, "hard\ncollapse →",
            fontsize=7.5, color="#B91C1C", ha="right", style="italic")

    ax.set_xlabel(r"Renormalized top-2 max prob  "
                  r"$\tilde{p}_1 = p_{(1)} / (p_{(1)} + p_{(2)})$",
                  fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(0.48, 1.02)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.95, loc="upper center")
    ax.set_title(
        "Bimodal routing confidence: tokens either share or commit",
        fontsize=10, pad=6,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v3_annotated.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",  nargs="+", required=True,
                        help="Paths to .npz files (supports diagnostics/data/ format)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Display name for each file (same order as --files)")
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        print("ERROR: --files and --labels must have the same number of entries.")
        return

    all_ratios, used_labels = [], []
    for path, label in zip(args.files, args.labels):
        fpath = path if os.path.isabs(path) else os.path.join(REPO, path)
        sr = load_split_ratios(fpath)
        if sr is None:
            print(f"WARNING: Skipping '{label}'")
            continue
        all_ratios.append(sr)
        used_labels.append(label)
        print(f"{label}: {len(sr)/1e6:.1f}M tokens  "
              f"median={np.median(sr):.3f}  "
              f"<0.55: {100*(sr<0.55).mean():.1f}%  "
              f">0.9: {100*(sr>0.9).mean():.1f}%")

    if not all_ratios:
        print("No data loaded.")
        return

    setup()
    plot_v1_side_by_side(all_ratios, used_labels, OUT_DIR)
    plot_v2_overlay(all_ratios, used_labels, OUT_DIR)
    plot_v3_annotated(all_ratios, used_labels, OUT_DIR)
    print(f"\nAll variants saved to neurips/fig3_stablelm_routing/figures/")


if __name__ == "__main__":
    main()
