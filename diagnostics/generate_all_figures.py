"""
Unified entry point: runs all plot scripts in sequence and prints a summary table.

This script only runs the PLOT scripts (which read existing data files).
Run the collect_* scripts separately beforehand to generate the data.

Usage:
    python diagnostics/generate_all_figures.py --data_dir diagnostics/data

Optional per-experiment overrides:
    --backbone       (for plot_convergence, default: qwen)
    --sqa_data_dir   (for plot_convergence, default: eval_results/sqa_checkpoints)
    --variants_conv  (for convergence plot, space-separated)
    --variants_stab  (for stability plot, space-separated)
    --labels_split   (for split-ratio plot, space-separated)
"""

import argparse
import importlib
import json
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_plot_convergence(args):
    """Wraps plot_convergence.main() with appropriate sys.argv."""
    import diagnostics.plot_convergence as m
    import importlib
    importlib.reload(m)

    argv_backup = sys.argv[:]
    sys.argv = [
        "plot_convergence.py",
        "--data_dir",   args.sqa_data_dir,
        "--backbone",   args.backbone,
        "--output_dir", args.output_dir,
    ]
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup


def run_plot_layer_stats(args):
    import diagnostics.plot_layer_stats as m
    import importlib
    importlib.reload(m)

    argv_backup = sys.argv[:]
    sys.argv = [
        "plot_layer_stats.py",
        "--data_dir",   args.data_dir,
        "--output_dir", args.output_dir,
    ]
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup


def run_plot_split_ratios(args):
    import diagnostics.plot_split_ratios as m
    import importlib
    importlib.reload(m)

    argv_backup = sys.argv[:]
    sys.argv = [
        "plot_split_ratios.py",
        "--data_dir",   args.data_dir,
        "--output_dir", args.output_dir,
        "--labels",     *args.labels_split,
    ]
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup


def run_plot_routing_stability(args):
    import diagnostics.plot_routing_stability as m
    import importlib
    importlib.reload(m)

    argv_backup = sys.argv[:]
    sys.argv = [
        "plot_routing_stability.py",
        "--data_dir",   args.data_dir,
        "--output_dir", args.output_dir,
        "--variants",   *args.variants_stab,
    ]
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup


# ── Summary extraction helpers ─────────────────────────────────────────────────

def _get_conv_gap(data_dir, sqa_data_dir, backbone, repo_root):
    """Return (student_step1, author_step1, gap) from CSV."""
    csv = os.path.join(repo_root, "diagnostics", "data", f"convergence_{backbone}.csv")
    if not os.path.exists(csv):
        return None, None, None
    step1 = {}
    with open(csv) as f:
        next(f)  # header
        for line in f:
            step, variant, acc = line.strip().split(",")
            if int(step) == 1:
                step1[variant] = float(acc)
    s = step1.get("student")
    a = step1.get("author")
    gap = (s - a) if (s is not None and a is not None) else None
    return s, a, gap


def _get_norm_stats(data_dir, repo_root):
    da_path = os.path.join(repo_root, data_dir, "layer_stats_A.json")
    db_path = os.path.join(repo_root, data_dir, "layer_stats_B.json")
    results = {}
    for tag, path in [("A", da_path), ("B", db_path)]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        norms = [v for v in d["feature_norms_mean"] if v is not None]
        ents  = [v for v in d["routing_entropy_mean"] if v is not None]
        if norms:
            results[f"norm_ratio_{tag}"] = norms[-1] / norms[0]
        if ents:
            import numpy as np
            results[f"ent_var_{tag}"] = float(np.var(ents))
    return results


def _get_split_stats(data_dir, labels, repo_root):
    import numpy as np
    results = {}
    for label in labels:
        path = os.path.join(repo_root, data_dir, f"split_ratios_{label}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        ratios = data["split_ratios_flat"]
        results[label] = {
            "near_collapse": 100 * (ratios > 0.9).mean(),
            "near_balanced": 100 * (ratios < 0.6).mean(),
        }
    return results


def _get_stability_stats(data_dir, variants, repo_root, target=0.8):
    results = {}
    for v in variants:
        path = os.path.join(repo_root, data_dir, f"stability_{v}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        reach = next(
            (s for s, sc in zip(d["steps"], d["stability_consecutive"])
             if sc is not None and not math.isnan(sc) and sc >= target),
            None,
        )
        results[v] = reach
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="diagnostics/data")
    parser.add_argument("--output_dir",  default="diagnostics")
    parser.add_argument("--backbone",    default="qwen")
    parser.add_argument("--sqa_data_dir",default="eval_results/sqa_checkpoints")
    parser.add_argument("--variants_conv", nargs="+",
                        default=["author", "student", "TS", "entropy_w01"])
    parser.add_argument("--variants_stab", nargs="+", default=["student", "TS"])
    parser.add_argument("--labels_split",  nargs="+",
                        default=["author", "entropy_old", "new_entropy"])
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("\n" + "=" * 60)
    print("  Generating all diagnostic figures")
    print("=" * 60)

    print("\n[1/4] Convergence curves...")
    run_plot_convergence(args)

    print("\n[2/4] Layer stats (feature norms + routing entropy)...")
    run_plot_layer_stats(args)

    print("\n[3/4] Split ratio histograms...")
    run_plot_split_ratios(args)

    print("\n[4/4] Routing stability...")
    run_plot_routing_stability(args)

    # ── Summary table ──────────────────────────────────────────────────────────
    s1, a1, gap = _get_conv_gap(args.data_dir, args.sqa_data_dir, args.backbone, repo_root)
    norm_stats  = _get_norm_stats(args.data_dir, repo_root)
    split_stats = _get_split_stats(args.data_dir, args.labels_split, repo_root)
    stab_stats  = _get_stability_stats(args.data_dir, args.variants_stab, repo_root)

    print("\n")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ DIAGNOSTIC SUMMARY                                       │")
    print("├──────────────────────┬──────────────────────────────────┤")

    if gap is not None:
        print(f"│ Initialization gap   │ student − author at step 1: {gap:+.2f} pp    │")
    else:
        print("│ Initialization gap   │ N/A (missing convergence data)         │")

    nr_A = norm_stats.get("norm_ratio_A")
    nr_B = norm_stats.get("norm_ratio_B")
    norm_str = f"A={nr_A:.1f}x, B={nr_B:.1f}x" if nr_A and nr_B else "N/A"
    print(f"│ Norm growth ratio    │ {norm_str:<34s} │")

    ev_A = norm_stats.get("ent_var_A")
    ev_B = norm_stats.get("ent_var_B")
    ent_str = f"A={ev_A:.4f}, B={ev_B:.4f}" if ev_A is not None and ev_B is not None else "N/A"
    print(f"│ Entropy variance     │ {ent_str:<34s} │")

    # Split stats: show first two labels
    old_label = next((l for l in args.labels_split if "entropy" in l and "new" not in l), None)
    new_label = next((l for l in args.labels_split if "new" in l), None)
    if old_label and new_label and old_label in split_stats and new_label in split_stats:
        nc_old = split_stats[old_label]["near_collapse"]
        nc_new = split_stats[new_label]["near_collapse"]
        print(f"│ Near-collapse frac.  │ old={nc_old:.1f}%, new={nc_new:.1f}%{'':<19}│")
    else:
        print("│ Near-collapse frac.  │ N/A (missing split ratio data)         │")

    v1 = args.variants_stab[0] if args.variants_stab else None
    v2 = args.variants_stab[1] if len(args.variants_stab) > 1 else None
    s1_step = stab_stats.get(v1)
    s2_step = stab_stats.get(v2)
    if s1_step is not None or s2_step is not None:
        stab_str = f"{v1}@step{s1_step} vs {v2}@step{s2_step}"
        print(f"│ Stability advantage  │ {stab_str:<34s} │")
    else:
        print("│ Stability advantage  │ N/A (missing stability data)           │")

    print("└──────────────────────┴──────────────────────────────────┘")

    fig_dir = os.path.join(repo_root, args.output_dir, "figures")
    print(f"\nAll figures saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
