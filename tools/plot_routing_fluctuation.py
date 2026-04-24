"""
Plot routing fluctuation analysis (equivalent to StableMoE Figure 7).

For each token in a fixed probe set, finds the LAST training step where its expert
assignment differed from its FINAL assignment. Plots the cumulative distribution of
these "last fluctuation steps" (x-axis = % of training, y-axis = % of tokens settled).

A good router (Fisher-init) shows most tokens settling EARLY.
A bad router (random init) keeps changing token assignments throughout training.

Usage:
    python tools/plot_routing_fluctuation.py \\
        --inputs routing_data/stablelm_fisher.json routing_data/stablelm_random.json \\
        --output figures/routing_fluctuation_stablelm \\
        --backbone "StableLM-1.6B×4-Top2"
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Color / style config ─────────────────────────────────────────────────────────

CURVE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
CURVE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

ANNOTATION_X_VALUES = [20, 50, 80]   # percent of training


# ── Fluctuation statistics ───────────────────────────────────────────────────────

def compute_fluctuation_fractions(data, layer_indices_filter=None):
    """
    For each (layer, token_position) pair, find the last training step where the
    expert assignment differed from the final assignment.

    Returns a 1-D numpy array of "last_fluct_fraction" values in [0, 1].
    0.0 means the token ALWAYS matched the final assignment (never fluctuated).
    1.0 means the token was still changing at the very last recorded step.

    Args:
        data: parsed JSON dict from collect_routing_assignments.py
        layer_indices_filter: set of int layer indices to include (None = all)
    """
    records = data["records"]
    total_steps = data["total_steps"]

    if not records:
        return np.array([])

    # Sort by step
    records = sorted(records, key=lambda r: r["step"])

    # Final assignments: last record
    final_record = records[-1]["layer_assignments"]

    # Determine which layers to process
    all_layers = sorted(final_record.keys(), key=int)
    if layer_indices_filter is not None:
        all_layers = [l for l in all_layers if int(l) in layer_indices_filter]

    fractions = []

    for layer_key in all_layers:
        if layer_key not in final_record:
            continue
        final_assignments = final_record[layer_key]
        n_tokens = len(final_assignments)

        for t in range(n_tokens):
            final_expert = final_assignments[t]
            last_fluct_step = 0  # default: never deviated from final

            # Walk records in REVERSE order, excluding the last record
            for rec in reversed(records[:-1]):
                layer_data = rec["layer_assignments"]
                if layer_key not in layer_data:
                    continue
                assignments = layer_data[layer_key]
                if t >= len(assignments):
                    continue
                if assignments[t] != final_expert:
                    last_fluct_step = rec["step"]
                    break  # found the latest step where it differed

            fraction = last_fluct_step / total_steps if total_steps > 0 else 0.0
            fractions.append(fraction)

    return np.array(fractions)


def compute_cdf(fractions):
    """
    Compute CDF of last fluctuation fractions.
    Returns (x_axis_percent, y_axis_percent) suitable for plotting.
    x: percent of training (0–100), y: percent of tokens settled by that point (0–100).
    """
    x_axis = np.linspace(0, 100, 500)
    y_axis = np.array([(fractions <= x / 100).mean() * 100 for x in x_axis])
    return x_axis, y_axis


# ── Plotting ─────────────────────────────────────────────────────────────────────

def setup_style():
    for style_name in ["seaborn-v0_8-paper", "seaborn-paper"]:
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue
    # Default matplotlib style — no error


def annotate_curve(ax, x_axis, y_axis, color, x_thresh, curve_idx, total_curves):
    """Add a small annotation at x_thresh showing the y-value."""
    # Find y at x_thresh
    idx = np.searchsorted(x_axis, x_thresh)
    if idx >= len(y_axis):
        return
    y_val = y_axis[idx]

    # Offset text vertically to avoid overlap between curves
    offset = -6 if curve_idx % 2 == 0 else 6
    text_y = max(2, min(98, y_val + offset))

    ax.annotate(
        f"({x_thresh}%, {y_val:.1f}%)",
        xy=(x_thresh, y_val),
        xytext=(x_thresh + 1.5, text_y),
        fontsize=7.5,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
        ha="left",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot routing fluctuation CDF (StableMoE Figure 7 equivalent)."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more JSON files from collect_routing_assignments.py"
    )
    parser.add_argument(
        "--output", default="figures/routing_fluctuation",
        help="Output path (without extension); both .pdf and .png are saved"
    )
    parser.add_argument(
        "--title", default="Routing Stability Analysis",
        help="Figure title"
    )
    parser.add_argument(
        "--backbone", default="StableLM-1.6B",
        help="Backbone name for subtitle"
    )
    parser.add_argument(
        "--layers", default=None,
        help='Comma-separated layer indices to include (default: all). E.g. "0,4,8,12"'
    )
    args = parser.parse_args()

    # Parse layer filter
    layer_filter = None
    if args.layers:
        layer_filter = set(int(x.strip()) for x in args.layers.split(","))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Print table header
    print(f"\n{'─'*70}")
    print(f"{'Label':<35} {'@20%':>8} {'@50%':>8} {'@80%':>8}  still-fluct")
    print(f"{'─'*70}")

    all_curve_data = []  # for annotation pass

    for i, input_path in enumerate(args.inputs):
        if not os.path.exists(input_path):
            warnings.warn(f"Input file not found: {input_path}")
            continue

        with open(input_path) as f:
            data = json.load(f)

        label = data.get("label", os.path.splitext(os.path.basename(input_path))[0])
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        ls = CURVE_STYLES[i % len(CURVE_STYLES)]

        fractions = compute_fluctuation_fractions(data, layer_filter)
        if len(fractions) == 0:
            warnings.warn(f"No fluctuation data computed for {input_path}")
            continue

        x_axis, y_axis = compute_cdf(fractions)

        ax.plot(x_axis, y_axis, color=color, linestyle=ls, linewidth=2.0, label=label)
        all_curve_data.append((x_axis, y_axis, color, i))

        # Text summary
        vals = {}
        for thresh in ANNOTATION_X_VALUES:
            idx = np.searchsorted(x_axis, thresh)
            vals[thresh] = y_axis[idx] if idx < len(y_axis) else 100.0
        fluct_pct = 100.0 - vals[80]
        print(
            f"{label:<35} {vals[20]:>7.1f}% {vals[50]:>7.1f}% {vals[80]:>7.1f}%  "
            f"({fluct_pct:.1f}% still fluct. after 80%)"
        )

    print(f"{'─'*70}\n")

    # Annotations on plot
    for (x_axis, y_axis, color, curve_idx) in all_curve_data:
        for x_thresh in ANNOTATION_X_VALUES:
            annotate_curve(ax, x_axis, y_axis, color, x_thresh, curve_idx, len(all_curve_data))

    # Reference vertical lines
    for xv in ANNOTATION_X_VALUES:
        ax.axvline(xv, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Last Fluctuation Step (% of Total Training)", fontsize=12)
    ax.set_ylabel("Cumulative Token Percentage (%)", fontsize=12)
    ax.tick_params(labelsize=10)

    full_title = f"{args.title}\n({args.backbone})" if args.backbone else args.title
    ax.set_title(full_title, fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as both PDF and PNG
    for ext in ("pdf", "png"):
        out_path = f"{args.output}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
