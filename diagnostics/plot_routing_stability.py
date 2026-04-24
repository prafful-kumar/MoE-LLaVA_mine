"""
Experiment 4: Plot expert assignment stability during training.

Input:
    diagnostics/data/stability_student.json
    diagnostics/data/stability_TS.json

Output:
    diagnostics/figures/routing_stability.pdf + .png

Usage:
    python diagnostics/plot_routing_stability.py \
        --variants student TS \
        --data_dir diagnostics/data \
        --output_dir diagnostics
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

DISPLAY_NAMES = {
    "student":     "K-means init (student)",
    "TS":          "Teacher-Student",
    "TS_schedule": "TS + schedule",
    "author":      "Random init (author)",
}


def load_stability(path):
    if not os.path.exists(path):
        warnings.warn(f"Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def clean(xs, ys):
    """Remove NaN entries from paired lists."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None and not math.isnan(y)]
    if not pairs:
        return [], []
    return zip(*pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["student", "TS"])
    parser.add_argument("--data_dir",   default="diagnostics/data")
    parser.add_argument("--output_dir", default="diagnostics")
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir   = os.path.join(repo_root, args.data_dir)
    fig_dir    = os.path.join(repo_root, args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    datas = {}
    for v in args.variants:
        d = load_stability(os.path.join(data_dir, f"stability_{v}.json"))
        if d is not None:
            datas[v] = d

    if not datas:
        print("No stability data found. Run collect_routing_stability.py first.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    TARGET = 0.8

    for variant, d in datas.items():
        color = VARIANT_COLORS.get(variant, "#888888")
        label = DISPLAY_NAMES.get(variant, variant)
        steps = d["steps"]

        # Left: consecutive stability
        xs_c, ys_c = clean(steps, d["stability_consecutive"])
        if xs_c:
            axes[0].plot(list(xs_c), list(ys_c), color=color, linewidth=1.8,
                         marker="o", markersize=4, label=label)

        # Right: vs-final stability
        xs_f, ys_f = clean(steps, d["stability_vs_final"])
        if xs_f:
            axes[1].plot(list(xs_f), list(ys_f), color=color, linewidth=1.8,
                         marker="o", markersize=4, label=label)

        # Key numbers
        reach_consec = next(
            (s for s, sc in zip(steps, d["stability_consecutive"])
             if sc is not None and not math.isnan(sc) and sc >= TARGET),
            None,
        )
        reach_final = next(
            (s for s, sc in zip(steps, d["stability_vs_final"])
             if sc is not None and not math.isnan(sc) and sc >= TARGET),
            None,
        )
        print(f"\n── Stability: {variant} ──────────────────────────────────────")
        print(f"  Step reaching {TARGET} stability (consec): "
              f"{'step ' + str(reach_consec) if reach_consec else 'never'}")
        print(f"  Step reaching {TARGET} stability (vs final): "
              f"{'step ' + str(reach_final) if reach_final else 'never'}")

        # Final stability (second-to-last vs last)
        if len(steps) >= 2:
            sc_list = d["stability_consecutive"]
            last_valid = next(
                (sc for sc in reversed(sc_list)
                 if sc is not None and not math.isnan(sc)),
                None,
            )
            print(f"  Final stability (last consec pair): "
                  f"{f'{last_valid:.3f}' if last_valid is not None else 'N/A'}")

    for ax, title in zip(axes, [
        "Consecutive-step stability",
        "Stability vs final checkpoint",
    ]):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
                   label="Fully stable")
        ax.axhline(TARGET, color="gray", linestyle=":", linewidth=0.8,
                   label=f"Target ({TARGET})")
        ax.set_xscale("log")
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_ylabel("Stability score", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Expert assignment stability during training", fontsize=11, y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"routing_stability.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"\nSaved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
