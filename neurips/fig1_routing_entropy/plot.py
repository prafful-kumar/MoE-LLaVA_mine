"""
neurips/fig1_routing_entropy/plot.py
====================================
Figure 1: Layer-wise routing entropy comparing author (aux-loss baseline) vs
entropy_topk_var (topk-entropy + L_var).

Narrative: Aux load-balancing loss does NOT enforce routing confidence — the author
baseline router maintains near-maximum entropy (H ≈ 1.38 nats ≈ ln 4) across ALL 12
layers, meaning each token's routing distribution is nearly uniform over all four
experts. Attempts to fix this with explicit entropy penalties produce the opposite
pathology: routing collapses to H ≈ 0 in 11/12 layers (one expert dominates entirely).

Data used:
  diagnostics/data/layer_stats_A_sqa1000.json   (author, random init, aux only)
  diagnostics/data/layer_stats_B_sqa1000.json   (entropy_topk_var)

Generates THREE figure variants so you can pick the best one:
  v1_line.pdf/png         — classic line + shaded std band (existing style, polished)
  v2_grouped_bar.pdf/png  — grouped bar chart, one group per layer (most readable)
  v3_annotated_bar.pdf/png— v2 + region annotations + collapse count callouts

Run:
  python neurips/fig1_routing_entropy/plot.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_A = os.path.join(REPO, "diagnostics/data/layer_stats_A_sqa1000.json")
DATA_B = os.path.join(REPO, "diagnostics/data/layer_stats_B_sqa1000.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colours (match project-wide VARIANT_COLORS) ──────────────────────────────
COLOR_A = "#E07A0F"   # author — warm amber
COLOR_B = "#2563EB"   # entropy_topk_var — royal blue
REF_COLOR = "#6B7280" # gray for reference lines / zones

# ── reference entropy values ──────────────────────────────────────────────────
MAX_ENT   = math.log(4)   # ln 4 ≈ 1.386  (uniform over all experts)
IDEAL_K2  = math.log(2)   # ln 2 ≈ 0.693  (ideal top-2: equal weight to 2 experts)
COLLAPSE  = 0.5           # threshold below which we call a layer "collapsed"


# ── load data ─────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)


def arrays(d):
    """Return (xs, means, stds) — xs are MoE layer indices."""
    xs    = np.array(d["layer_indices"])
    means = np.array(d["routing_entropy_mean"])
    stds  = np.array(d["routing_entropy_std"])
    return xs, means, stds


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — Line + shaded band  (traditional, shows std bands clearly)
# ─────────────────────────────────────────────────────────────────────────────
def plot_v1_line(da, db, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    xs_a, ms_a, ss_a = arrays(da)
    xs_b, ms_b, ss_b = arrays(db)

    # Reference lines
    ax.axhline(MAX_ENT, color=REF_COLOR, ls="--", lw=0.9,
               label=f"Max entropy  (ln 4 ≈ {MAX_ENT:.2f})", zorder=1)
    ax.axhline(IDEAL_K2, color=REF_COLOR, ls=":",  lw=0.9,
               label=f"Ideal top-2  (ln 2 ≈ {IDEAL_K2:.2f})", zorder=1)
    ax.axhspan(0, COLLAPSE, color="#FEE2E2", alpha=0.45, zorder=0,
               label="Collapse zone  (H < 0.5)")

    # Model A
    ax.plot(xs_a, ms_a, color=COLOR_A, lw=1.8, marker="o", ms=4,
            ls="--", label=da["label"], zorder=3)
    ax.fill_between(xs_a, ms_a - ss_a, ms_a + ss_a,
                    color=COLOR_A, alpha=0.15, zorder=2)

    # Model B
    ax.plot(xs_b, ms_b, color=COLOR_B, lw=1.8, marker="o", ms=4,
            ls="-",  label=db["label"], zorder=3)
    ax.fill_between(xs_b, ms_b - ss_b, ms_b + ss_b,
                    color=COLOR_B, alpha=0.15, zorder=2)

    # Callout: collapsed layer count for B
    n_collapsed = int((ms_b < COLLAPSE).sum())
    ax.annotate(
        f"{n_collapsed}/12 layers\ncollapsed (H < 0.5)",
        xy=(xs_b[4], ms_b[4]), xytext=(xs_b[6] + 1, 0.65),
        fontsize=8, color=COLOR_B,
        arrowprops=dict(arrowstyle="->", color=COLOR_B, lw=0.9),
    )

    ax.set_xlabel("MoE Layer Index", fontsize=10)
    ax.set_ylabel("Mean Routing Entropy  H (nats)", fontsize=10)
    ax.set_ylim(-0.05, MAX_ENT + 0.12)
    ax.set_xticks(xs_a)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper right", ncol=1)
    ax.set_title(
        "Aux loss leaves routing uncertain; entropy penalty causes collapse",
        fontsize=9.5, pad=6,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v1_line.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — Grouped bar chart  (x = layer, grouped by model; most readable)
# ─────────────────────────────────────────────────────────────────────────────
def plot_v2_grouped_bar(da, db, out_dir):
    xs_a, ms_a, ss_a = arrays(da)
    xs_b, ms_b, ss_b = arrays(db)

    n_layers = len(xs_a)
    x = np.arange(n_layers)
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    bars_a = ax.bar(x - width/2, ms_a, width, color=COLOR_A, alpha=0.85,
                    yerr=ss_a, error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8),
                    label=da["label"], zorder=3)
    bars_b = ax.bar(x + width/2, ms_b, width, color=COLOR_B, alpha=0.85,
                    yerr=ss_b, error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8),
                    label=db["label"], zorder=3)

    # Reference lines
    ax.axhline(MAX_ENT,  color=REF_COLOR, ls="--", lw=1.0, zorder=2,
               label=f"Max entropy (ln 4 ≈ {MAX_ENT:.2f})")
    ax.axhline(IDEAL_K2, color=REF_COLOR, ls=":",  lw=1.0, zorder=2,
               label=f"Ideal top-2 (ln 2 ≈ {IDEAL_K2:.2f})")
    ax.axhspan(0, COLLAPSE, color="#FEE2E2", alpha=0.35, zorder=0,
               label="Collapse zone  (H < 0.5)")

    ax.set_xlabel("MoE Layer Index", fontsize=10)
    ax.set_ylabel("Mean Routing Entropy  H (nats)", fontsize=10)
    ax.set_ylim(0, MAX_ENT + 0.18)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in xs_a], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper right", ncol=2)
    ax.set_title(
        "Per-layer routing entropy: lower H = more confident router",
        fontsize=9.5, pad=6,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v2_grouped_bar.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3 — Annotated grouped bar  (v2 + zone shading + callout box)
# ─────────────────────────────────────────────────────────────────────────────
def plot_v3_annotated_bar(da, db, out_dir):
    xs_a, ms_a, ss_a = arrays(da)
    xs_b, ms_b, ss_b = arrays(db)

    n_layers = len(xs_a)
    x = np.arange(n_layers)
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    # Zone shading behind bars
    ax.axhspan(MAX_ENT - 0.08, MAX_ENT + 0.12, color="#FEF3C7", alpha=0.6, zorder=0)  # uncertain zone
    ax.axhspan(0, COLLAPSE, color="#FEE2E2", alpha=0.5, zorder=0)                       # collapse zone
    ax.axhspan(COLLAPSE, IDEAL_K2, color="#D1FAE5", alpha=0.45, zorder=0)               # target zone

    # Bars
    ax.bar(x - width/2, ms_a, width, color=COLOR_A, alpha=0.88,
           yerr=ss_a, error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8),
           label=da["label"], zorder=3)
    ax.bar(x + width/2, ms_b, width, color=COLOR_B, alpha=0.88,
           yerr=ss_b, error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8),
           label=db["label"], zorder=3)

    # Reference lines
    ax.axhline(MAX_ENT,  color=REF_COLOR, ls="--", lw=0.9, zorder=4)
    ax.axhline(IDEAL_K2, color=REF_COLOR, ls=":",  lw=0.9, zorder=4)
    ax.axhline(COLLAPSE, color="#EF4444",  ls="-",  lw=0.7, zorder=4, alpha=0.6)

    # Zone text labels (right side)
    ax.text(n_layers - 0.1, MAX_ENT + 0.04, "Uncertain\n(aux loss)", ha="right",
            va="bottom", fontsize=7.5, color="#92400E", style="italic")
    ax.text(n_layers - 0.1, IDEAL_K2 - 0.04, "Target\nzone", ha="right",
            va="top", fontsize=7.5, color="#065F46", style="italic")
    ax.text(n_layers - 0.1, COLLAPSE - 0.02, "Collapse", ha="right",
            va="top", fontsize=7.5, color="#991B1B", style="italic")

    # Callout stat box
    n_collapsed = int((ms_b < COLLAPSE).sum())
    mean_a = ms_a.mean()
    mean_b = ms_b.mean()
    stats_text = (
        f"{da['label'].split('(')[0].strip()}\n"
        f"  mean H = {mean_a:.2f}  •  0/12 collapsed\n\n"
        f"{db['label'].split('(')[0].strip()}\n"
        f"  mean H = {mean_b:.2f}  •  {n_collapsed}/12 collapsed"
    )
    ax.text(0.01, 0.97, stats_text, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D1D5DB", lw=0.8))

    ax.set_xlabel("MoE Layer Index", fontsize=10)
    ax.set_ylabel("Mean Routing Entropy  H (nats)", fontsize=10)
    ax.set_ylim(-0.02, MAX_ENT + 0.18)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in xs_a], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.95, loc="upper right", ncol=1)
    ax.set_title(
        "Routing entropy by layer  —  lower H = more confident router",
        fontsize=9.5, pad=6,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"v3_annotated_bar.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    da = load(DATA_A)
    db = load(DATA_B)

    print(f"Model A: {da['label']}")
    print(f"  entropy: {[round(x,3) for x in da['routing_entropy_mean']]}")
    print(f"Model B: {db['label']}")
    print(f"  entropy: {[round(x,3) for x in db['routing_entropy_mean']]}")
    print()

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.35,
        "grid.linewidth":    0.6,
    })

    plot_v1_line(da, db, OUT_DIR)
    plot_v2_grouped_bar(da, db, OUT_DIR)
    plot_v3_annotated_bar(da, db, OUT_DIR)

    print("\nAll 3 variants saved to neurips/fig1_routing_entropy/figures/")


if __name__ == "__main__":
    main()
