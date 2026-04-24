"""
Experiment 1: Initialization quality — convergence curves.

Loads per-step ScienceQA accuracy from existing JSON files and plots
convergence curves for all variants on a log-scale x-axis.

Usage:
    python diagnostics/plot_convergence.py \
        --data_dir eval_results/sqa_checkpoints \
        --backbone qwen \
        --output_dir diagnostics/

JSON filename convention expected:
    {data_dir}/{backbone}_{variant}_step{step}.json
    or
    {data_dir}/{backbone}_{variant_alias}_step{step}.json

Variant aliases (file → canonical name):
    author           → author
    student          → student
    teacher_student  → TS
    TS               → TS
    entropy_w01      → entropy_w01
    new_entropy      → new_entropy
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

# ── repo root on path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diagnostics.utils import VARIANT_COLORS, VARIANT_LINESTYLES, STEP_CHECKPOINTS

# Map from filename slug → canonical variant name
FILE_ALIAS = {
    "author":          "author",
    "student":         "student",
    "teacher_student": "TS",
    "TS":              "TS",
    "TS_schedule":     "TS_schedule",
    "entropy":         "entropy",
    "entropy_w01":     "entropy_w01",
    "new_entropy":     "new_entropy",
}

# Which variants to plot (and their display labels)
VARIANTS_TO_PLOT = {
    "author":      "Random init (author)",
    "student":     "K-means init (student)",
    "TS":          "Teacher-Student",
    "entropy_w01": "Entropy w=0.1 (old)",
}


def load_sqa_results(data_dir, backbone, variant_slug, steps):
    """
    Load ScienceQA accuracy for a given variant across all steps.
    Returns dict {step: accuracy} — missing steps are omitted with a warning.
    """
    results = {}
    for step in steps:
        fname = f"{backbone}_{variant_slug}_step{step}.json"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            warnings.warn(f"Missing: {fpath}")
            continue
        with open(fpath) as f:
            data = json.load(f)
        acc = data.get("accuracy")
        if acc is None:
            warnings.warn(f"No 'accuracy' key in {fpath}")
            continue
        results[step] = acc
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot SQA convergence curves")
    parser.add_argument("--data_dir", default="eval_results/sqa_checkpoints",
                        help="Directory containing per-step SQA JSON files")
    parser.add_argument("--backbone", default="qwen",
                        help="Backbone prefix in filenames (e.g. qwen, phi2)")
    parser.add_argument("--output_dir", default="diagnostics",
                        help="Root output directory (figures/ and data/ created inside)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated step list (default: STEP_CHECKPOINTS + 10000)")
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir   = os.path.join(repo_root, args.data_dir)
    fig_dir    = os.path.join(repo_root, args.output_dir, "figures")
    csv_dir    = os.path.join(repo_root, args.output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = STEP_CHECKPOINTS + [10000]

    # ── Load data ──────────────────────────────────────────────────────────────
    all_data = {}   # canonical_variant → {step: acc}

    for file_slug, canonical in FILE_ALIAS.items():
        if canonical not in VARIANTS_TO_PLOT:
            continue
        results = load_sqa_results(data_dir, args.backbone, file_slug, steps)
        if results:
            # Merge if already have data from another alias
            if canonical not in all_data:
                all_data[canonical] = {}
            all_data[canonical].update(results)

    if not all_data:
        print("ERROR: No data found. Check --data_dir and --backbone.")
        sys.exit(1)

    # ── Print key numbers ──────────────────────────────────────────────────────
    print("\n── Step-1 accuracy ──────────────────────────────────────────")
    step1_accs = {}
    for variant in ["author", "student", "TS", "entropy_w01"]:
        acc = all_data.get(variant, {}).get(1, None)
        step1_accs[variant] = acc
        label = VARIANTS_TO_PLOT.get(variant, variant)
        print(f"  {label:35s}: {f'{acc:.2f}%' if acc is not None else 'N/A'}")

    if step1_accs.get("student") is not None and step1_accs.get("author") is not None:
        gap = step1_accs["student"] - step1_accs["author"]
        print(f"\n  Gap (student − author) at step 1: {gap:+.2f} pp")

    # How many steps for author to match student's step-100 accuracy
    student_step100 = all_data.get("student", {}).get(100, None)
    if student_step100 is not None and "author" in all_data:
        author_steps_sorted = sorted(all_data["author"].items())
        catch_up = next(
            (s for s, a in author_steps_sorted if a >= student_step100), None
        )
        print(f"  Steps for author to match student's step-100 acc "
              f"({student_step100:.2f}%): "
              f"{'step ' + str(catch_up) if catch_up else 'never within measured steps'}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted_variants = []
    for variant, label in VARIANTS_TO_PLOT.items():
        if variant not in all_data:
            continue
        d = all_data[variant]
        xs = sorted(d.keys())
        ys = [d[s] for s in xs]
        color = VARIANT_COLORS.get(variant, "gray")
        ls    = VARIANT_LINESTYLES.get(variant, "-")
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.8,
                marker="o", markersize=3, label=label)
        plotted_variants.append(variant)

        # Annotate step-1 value
        if 1 in d:
            ax.annotate(
                f"{d[1]:.1f}",
                xy=(1, d[1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
            )

    # Step-1 vertical marker
    ax.axvline(x=1, color="gray", linestyle=":", linewidth=1.0)
    ax.text(1.2, ax.get_ylim()[0] + 0.5, "step 1", fontsize=8, color="gray",
            va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("ScienceQA accuracy (%)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title(f"Convergence — {args.backbone.capitalize()} backbone", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    stem = f"convergence_{args.backbone}"
    pdf_path = os.path.join(fig_dir, stem + ".pdf")
    png_path = os.path.join(fig_dir, stem + ".png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    # ── CSV export ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(csv_dir, stem + ".csv")
    with open(csv_path, "w") as f:
        f.write("step,variant,sqa_acc\n")
        for variant in plotted_variants:
            for step in sorted(all_data[variant].keys()):
                f.write(f"{step},{variant},{all_data[variant][step]:.4f}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
