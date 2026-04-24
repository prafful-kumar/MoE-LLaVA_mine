"""
Experiment 2: Plot feature norm growth and routing entropy by layer.

Input:
    diagnostics/data/layer_stats_A.json
    diagnostics/data/layer_stats_B.json

Output:
    diagnostics/figures/feature_norms.pdf + .png
    diagnostics/figures/routing_entropy_by_layer.pdf + .png

Usage:
    python diagnostics/plot_layer_stats.py --data_dir diagnostics/data --output_dir diagnostics
"""

import argparse
import json
import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diagnostics.utils import VARIANT_COLORS

COLOR_A = VARIANT_COLORS["author"]
COLOR_B = VARIANT_COLORS["entropy_topk_var"]


def load_stats(path):
    if not os.path.exists(path):
        warnings.warn(f"Missing data file: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _arrays(d, key_mean, key_std):
    """Return (xs, means, stds) ignoring None entries."""
    indices = d["layer_indices"]
    means   = d[key_mean]
    stds    = d[key_std]
    valid   = [(i, m, s) for i, m, s in zip(indices, means, stds)
               if m is not None and s is not None]
    if not valid:
        return [], [], []
    xs, ms, ss = zip(*valid)
    return list(xs), list(ms), list(ss)


def plot_with_band(ax, xs, means, stds, color, label, linestyle="-"):
    xs    = np.array(xs)
    means = np.array(means)
    stds  = np.array(stds)
    ax.plot(xs, means, color=color, linestyle=linestyle, linewidth=1.8,
            marker="o", markersize=4, label=label)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="diagnostics/data")
    parser.add_argument("--output_dir", default="diagnostics")
    parser.add_argument("--data_suffix", default="",
                        help="Suffix on input filenames, e.g. '_sqa1000' → layer_stats_A_sqa1000.json")
    parser.add_argument("--out_tag", default="",
                        help="Tag appended to output figure filenames, e.g. 'sqa1000'")
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir   = os.path.join(repo_root, args.data_dir)
    fig_dir    = os.path.join(repo_root, args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sfx = args.data_suffix  # e.g. "_sqa1000"
    da = load_stats(os.path.join(data_dir, f"layer_stats_A{sfx}.json"))
    db = load_stats(os.path.join(data_dir, f"layer_stats_B{sfx}.json"))

    if da is None and db is None:
        print("No data found. Run collect_layer_stats.py first.")
        return

    label_a = da["label"] if da else "Model A"
    label_b = db["label"] if db else "Model B"

    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Figure 1: Feature norms ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    if da:
        xs, ms, ss = _arrays(da, "feature_norms_mean", "feature_norms_std")
        if xs:
            plot_with_band(ax, xs, ms, ss, COLOR_A, label_a, linestyle="--")
    if db:
        xs, ms, ss = _arrays(db, "feature_norms_mean", "feature_norms_std")
        if xs:
            plot_with_band(ax, xs, ms, ss, COLOR_B, label_b, linestyle="-")

    ax.set_xlabel("MoE layer index", fontsize=11)
    ax.set_ylabel("Mean L2 norm of hidden states", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title("Hidden state norms grow with depth", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    tag_str = f"_{args.out_tag}" if args.out_tag else ""
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"feature_norms{tag_str}.{ext}")
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)

    # ── Figure 2: Routing entropy by layer ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    max_ent  = math.log(4)   # log(E) for E=4
    ideal_k2 = math.log(2)   # equal split over top-2

    ax.axhline(max_ent,  color="gray", linestyle="--", linewidth=1.0,
               label=f"Max entropy / uniform (ln 4 ≈ {max_ent:.3f})")
    ax.axhline(ideal_k2, color="gray", linestyle=":",  linewidth=1.0,
               label=f"Ideal top-2 entropy (ln 2 ≈ {ideal_k2:.3f})")

    if da:
        xs, ms, ss = _arrays(da, "routing_entropy_mean", "routing_entropy_std")
        if xs:
            plot_with_band(ax, xs, ms, ss, COLOR_A, label_a, linestyle="--")
    if db:
        xs, ms, ss = _arrays(db, "routing_entropy_mean", "routing_entropy_std")
        if xs:
            plot_with_band(ax, xs, ms, ss, COLOR_B, label_b, linestyle="-")

    ax.set_xlabel("MoE layer index", fontsize=11)
    ax.set_ylabel("Routing entropy H (nats)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title(
        "Routing entropy: dot-product collapses vs cosine stays stable",
        fontsize=10,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"routing_entropy_by_layer{tag_str}.{ext}")
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)

    # ── Figure 3: Combined dual-axis (entropy left, norm right) ───────────────
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.axhline(max_ent,  color="gray", linestyle="--", linewidth=0.9,
                label=f"Max H (ln 4≈{max_ent:.2f})")
    ax1.axhline(ideal_k2, color="gray", linestyle=":",  linewidth=0.9,
                label=f"Ideal top-2 H (ln 2≈{ideal_k2:.2f})")

    lines = []
    if da:
        xs_e, ms_e, ss_e = _arrays(da, "routing_entropy_mean", "routing_entropy_std")
        xs_n, ms_n, ss_n = _arrays(da, "feature_norms_mean",   "feature_norms_std")
        if xs_e:
            l, = ax1.plot(xs_e, ms_e, color=COLOR_A, linestyle="--", linewidth=1.8,
                          marker="o", markersize=4, label=f"{label_a} — entropy (left)")
            ax1.fill_between(xs_e, np.array(ms_e)-np.array(ss_e),
                             np.array(ms_e)+np.array(ss_e), color=COLOR_A, alpha=0.12)
            lines.append(l)
        if xs_n:
            l, = ax2.plot(xs_n, ms_n, color=COLOR_A, linestyle=(0, (3,1,1,1)), linewidth=1.5,
                          marker="s", markersize=3, label=f"{label_a} — norm (right)")
            ax2.fill_between(xs_n, np.array(ms_n)-np.array(ss_n),
                             np.array(ms_n)+np.array(ss_n), color=COLOR_A, alpha=0.07)
            lines.append(l)
    if db:
        xs_e, ms_e, ss_e = _arrays(db, "routing_entropy_mean", "routing_entropy_std")
        xs_n, ms_n, ss_n = _arrays(db, "feature_norms_mean",   "feature_norms_std")
        if xs_e:
            l, = ax1.plot(xs_e, ms_e, color=COLOR_B, linestyle="-", linewidth=1.8,
                          marker="o", markersize=4, label=f"{label_b} — entropy (left)")
            ax1.fill_between(xs_e, np.array(ms_e)-np.array(ss_e),
                             np.array(ms_e)+np.array(ss_e), color=COLOR_B, alpha=0.12)
            lines.append(l)
        if xs_n:
            l, = ax2.plot(xs_n, ms_n, color=COLOR_B, linestyle=(0, (3,1,1,1)), linewidth=1.5,
                          marker="s", markersize=3, label=f"{label_b} — norm (right)")
            ax2.fill_between(xs_n, np.array(ms_n)-np.array(ss_n),
                             np.array(ms_n)+np.array(ss_n), color=COLOR_B, alpha=0.07)
            lines.append(l)

    ax1.set_xlabel("MoE layer index", fontsize=11)
    ax1.set_ylabel("Routing entropy H (nats)", fontsize=11, color="black")
    ax2.set_ylabel("Hidden state L2 norm", fontsize=11, color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax1.tick_params(labelsize=9)
    ax1.set_title(
        "Routing entropy per layer: author baseline vs topk-entropy + L_var",
        fontsize=9,
    )
    # Combine legend: lines from both axes + reference lines from ax1
    ref_handles, ref_labels = ax1.get_legend_handles_labels()
    all_handles = ref_handles + lines
    all_labels  = ref_labels  + [l.get_label() for l in lines]
    ax1.legend(all_handles, all_labels, fontsize=8, framealpha=0.9, loc="upper right",
               ncol=1)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"layer_entropy_and_norm{tag_str}.{ext}")
        fig.savefig(p, dpi=300)
        print(f"Saved: {p}")
    plt.close(fig)

    # ── Key numbers ────────────────────────────────────────────────────────────
    for d, tag in [(da, "A"), (db, "B")]:
        if d is None:
            continue
        norms = [v for v in d["feature_norms_mean"] if v is not None]
        ents  = [v for v in d["routing_entropy_mean"] if v is not None]
        idxs  = d["layer_indices"]
        if norms:
            print(f"Feature norm at layer {idxs[0]}: {tag}={norms[0]:.2f}")
            print(f"Feature norm at final layer {idxs[-1]}: {tag}={norms[-1]:.2f}")
            print(f"Norm growth ratio (final/first): {tag}={norms[-1]/norms[0]:.2f}x")
        if ents:
            print(f"Routing entropy variance across layers: {tag}={float(np.var(ents)):.4f}")
            collapsed = [idxs[i] for i, e in enumerate(
                d["routing_entropy_mean"]) if e is not None and e < 0.5]
            print(f"Layers where H < 0.5 (collapsed): {tag}={collapsed}")


if __name__ == "__main__":
    main()
