"""
vis_train_dist_routing.py — Multi-layer routing analysis on the actual training distribution.

Unlike vis_dual_routing_v2.py (which shows one layer at a time), this script shows
the full picture across ALL MoE layers for a single checkpoint. It answers:

  1. At which layers does the router differentiate by data source?
  2. Which experts specialize for which sources?
  3. How consistent is within-source routing?

Three output figures:
  A. specialization_heatmap.png
     [n_sources × n_moe_layers] heatmap.
     Each cell: dominant expert for that (source, layer) pair.
     Color = expert id, alpha = fraction of tokens going to that expert.
     Side-by-side: IMAGE tokens | TEXT tokens.

  B. specialization_score.png
     Per-layer specialization score = mean pairwise KL divergence between
     source routing distributions. Higher → sources are routed more differently.
     One line per token type (image/text). Peaks identify the most informative layers.

  C. best_layer_detail.png
     Full per-source routing distribution for the top-3 most specialized layers.
     Grouped bar chart: x = source, bars = expert fractions.
     Helps verify the heatmap pattern is real, not a visualisation artifact.

Usage:
    python moellava/vis/vis_train_dist_routing.py \\
        --input diagnostics/train_dist_stablelm_power_adaptive.pt \\
        --output diagnostics/train_dist_plots/stablelm_power_adaptive/ \\
        [--sources coco gqa ocr_vqa textvqa nlp]  # optional filter

Compare two checkpoints side by side by running on both .pt files and
viewing the specialization_score plots together.
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Expert colors (consistent with vis_dual_routing_v2.py)
# ---------------------------------------------------------------------------
EXPERT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
SOURCE_PALETTE = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71',
                  '#9B59B6', '#E67E22', '#1ABC9C', '#E91E63']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_routing_data(pt_path, source_filter=None):
    """Load .pt file. Returns dict keyed by sample_id."""
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location='cpu')
    if source_filter:
        data = {k: v for k, v in data.items() if v.get('category') in source_filter}
        print(f"Filtered to {len(data)} samples from sources: {source_filter}")
    return data


def extract_layer_info(data):
    """Return (layer_indices list, num_experts) from first sample."""
    sample = next(iter(data.values()))
    layer_indices = sample['layer_indices']
    num_experts = sample['gating_logit'][0].shape[-1]
    return layer_indices, num_experts


def get_sources(data):
    sources = sorted(set(v.get('category', 'unknown') for v in data.values()))
    return sources


# ---------------------------------------------------------------------------
# Core aggregation: token-level expert counts per (source, layer)
# ---------------------------------------------------------------------------

def aggregate_expert_counts(data, layer_indices, num_experts, top_k=2):
    """
    Returns two dicts: img_counts, txt_counts
      img_counts[source][layer_idx][expert] = int (number of image tokens routed there)
      txt_counts[source][layer_idx][expert] = int (number of text tokens routed there)
    """
    img_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(num_experts, dtype=np.float64)))
    txt_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(num_experts, dtype=np.float64)))

    for sample in tqdm(data.values(), desc="Aggregating counts"):
        source = sample.get('category', 'unknown')
        output_ids = sample['output_ids'].flatten().tolist()

        try:
            img_start = output_ids.index(-200)
            img_end = min(img_start + 576, len(output_ids))
        except ValueError:
            img_start = img_end = 0

        for list_idx, layer_idx in enumerate(layer_indices):
            if list_idx >= len(sample['gating_logit']):
                continue
            logits = sample['gating_logit'][list_idx].float()
            if logits.dim() == 3:
                logits = logits.squeeze(0)  # [seq_len, num_experts]

            k = min(top_k, num_experts)

            # Image tokens
            if img_end > img_start:
                img_part = logits[img_start:img_end]
                if img_part.shape[0] > 0:
                    topk_idx = torch.topk(img_part, k=k, dim=1).indices  # [n_img_tokens, k]
                    for exp_idx in topk_idx.flatten().tolist():
                        img_counts[source][layer_idx][int(exp_idx)] += 1

            # Text tokens (everything outside image span)
            txt_parts = []
            if img_start > 0:
                txt_parts.append(logits[:img_start])
            if img_end < logits.shape[0]:
                txt_parts.append(logits[img_end:])
            if txt_parts:
                txt_part = torch.cat(txt_parts, dim=0)
                if txt_part.shape[0] > 0:
                    topk_idx = torch.topk(txt_part, k=k, dim=1).indices
                    for exp_idx in topk_idx.flatten().tolist():
                        txt_counts[source][layer_idx][int(exp_idx)] += 1

    return img_counts, txt_counts


def normalize_counts(counts, sources, layer_indices):
    """Normalize raw counts to fractions. Returns prefs[source][layer_idx] = np.array[num_experts]."""
    prefs = {}
    for src in sources:
        prefs[src] = {}
        for layer_idx in layer_indices:
            c = counts[src][layer_idx]
            total = c.sum()
            prefs[src][layer_idx] = c / (total + 1e-8)
    return prefs


# ---------------------------------------------------------------------------
# Specialization score: mean pairwise KL divergence
# ---------------------------------------------------------------------------

def kl_divergence(p, q, eps=1e-8):
    """KL(p || q) with clipping."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def pairwise_kl(prefs, sources, layer_idx):
    """Mean pairwise symmetric KL across all source pairs for one layer."""
    dists = [prefs[src][layer_idx] for src in sources]
    if len(dists) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            total += 0.5 * (kl_divergence(dists[i], dists[j]) + kl_divergence(dists[j], dists[i]))
            n += 1
    return total / (n + 1e-8)


# ---------------------------------------------------------------------------
# Figure A: Specialization heatmap [sources × layers]
# ---------------------------------------------------------------------------

def plot_specialization_heatmap(img_prefs, txt_prefs, sources, layer_indices,
                                num_experts, out_dir, ckpt_name):
    n_layers = len(layer_indices)
    n_sources = len(sources)

    # Build matrices: dominant_expert[s, l] and dominance_frac[s, l]
    def build_matrices(prefs):
        dominant = np.zeros((n_sources, n_layers), dtype=int)
        dominance = np.zeros((n_sources, n_layers))
        for si, src in enumerate(sources):
            for li, layer_idx in enumerate(layer_indices):
                p = prefs[src][layer_idx]
                dominant[si, li] = int(np.argmax(p))
                dominance[si, li] = float(np.max(p))
        return dominant, dominance

    img_dom, img_frac = build_matrices(img_prefs)
    txt_dom, txt_frac = build_matrices(txt_prefs)

    fig, axes = plt.subplots(1, 2, figsize=(max(16, n_layers * 0.9), max(5, n_sources * 1.2 + 2)))
    fig.suptitle(f"Expert Specialization by Data Source — {ckpt_name}\n"
                 "Cell: dominant expert (color) × routing strength (opacity)",
                 fontsize=13, fontweight='bold')

    expert_cmap = matplotlib.colormaps['tab10'].resampled(num_experts)

    def draw_heatmap(ax, dominant, dominance, title):
        # Background: expert color, alpha: dominance fraction
        img_rgba = np.zeros((n_sources, n_layers, 4))
        for si in range(n_sources):
            for li in range(n_layers):
                color = expert_cmap(dominant[si, li] / max(num_experts - 1, 1))
                # Map dominance [1/K, 1] → alpha [0.2, 1.0] so min is still visible
                alpha = 0.2 + 0.8 * (dominance[si, li] - 1.0 / num_experts) / (1.0 - 1.0 / num_experts + 1e-8)
                img_rgba[si, li, :3] = color[:3]
                img_rgba[si, li, 3] = float(np.clip(alpha, 0.15, 1.0))

        ax.imshow(img_rgba, aspect='auto', origin='upper', interpolation='nearest')

        # Annotate each cell with expert id and fraction
        for si in range(n_sources):
            for li in range(n_layers):
                ax.text(li, si,
                        f"E{dominant[si, li]}\n{dominance[si, li]:.0%}",
                        ha='center', va='center', fontsize=7,
                        color='white' if dominance[si, li] > 0.45 else 'black',
                        fontweight='bold')

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(l) for l in layer_indices], fontsize=8, rotation=45)
        ax.set_yticks(range(n_sources))
        ax.set_yticklabels(sources, fontsize=9)
        ax.set_xlabel("MoE Layer Index")
        ax.set_title(title, fontweight='bold')

        # Expert color legend
        legend_patches = [
            matplotlib.patches.Patch(color=expert_cmap(e / max(num_experts - 1, 1)),
                                     label=f"Expert {e}")
            for e in range(num_experts)
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=7,
                  title="Expert", framealpha=0.8)

    draw_heatmap(axes[0], img_dom, img_frac, "IMAGE Tokens: Dominant Expert per Source × Layer")
    draw_heatmap(axes[1], txt_dom, txt_frac, "TEXT Tokens: Dominant Expert per Source × Layer")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "specialization_heatmap.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure B: Per-layer specialization score
# ---------------------------------------------------------------------------

def plot_specialization_score(img_prefs, txt_prefs, sources, layer_indices, out_dir, ckpt_name):
    img_scores = [pairwise_kl(img_prefs, sources, l) for l in layer_indices]
    txt_scores = [pairwise_kl(txt_prefs, sources, l) for l in layer_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(layer_indices) * 0.7), 5))
    x = np.arange(len(layer_indices))
    ax.plot(x, img_scores, 'o-', color='#E74C3C', label='Image tokens', linewidth=2, markersize=6)
    ax.plot(x, txt_scores, 's--', color='#3498DB', label='Text tokens', linewidth=2, markersize=6)
    ax.fill_between(x, img_scores, alpha=0.12, color='#E74C3C')
    ax.fill_between(x, txt_scores, alpha=0.12, color='#3498DB')

    # Mark top-3 layers by image specialization
    top3 = sorted(range(len(layer_indices)), key=lambda i: img_scores[i], reverse=True)[:3]
    for rank, i in enumerate(top3):
        ax.annotate(f"L{layer_indices[i]}",
                    xy=(i, img_scores[i]),
                    xytext=(i, img_scores[i] + 0.02 * max(img_scores + [0.001])),
                    ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layer_indices], fontsize=9, rotation=45)
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Mean pairwise symmetric KL divergence")
    ax.set_title(f"Per-Layer Routing Specialization Score — {ckpt_name}\n"
                 f"(Higher = sources routed more differently; {len(sources)} sources: {', '.join(sources)})",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "specialization_score.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")

    return img_scores, txt_scores


# ---------------------------------------------------------------------------
# Figure C: Best-layer detail
# ---------------------------------------------------------------------------

def plot_best_layer_detail(img_prefs, txt_prefs, sources, layer_indices,
                           img_scores, num_experts, out_dir, ckpt_name, top_n=3):
    # Pick top_n layers by image specialization
    top_layers = sorted(range(len(layer_indices)),
                        key=lambda i: img_scores[i], reverse=True)[:top_n]
    top_layer_indices = [layer_indices[i] for i in top_layers]

    fig, axes = plt.subplots(2, top_n,
                             figsize=(6 * top_n, 9),
                             squeeze=False)
    fig.suptitle(f"Routing Distribution by Source — Top-{top_n} Specialized Layers — {ckpt_name}",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(sources))
    bar_width = 0.8 / num_experts

    for col, (rank_idx, layer_idx) in enumerate(zip(top_layers, top_layer_indices)):
        for row, (prefs, token_type) in enumerate([(img_prefs, 'IMAGE'), (txt_prefs, 'TEXT')]):
            ax = axes[row][col]
            for e in range(num_experts):
                vals = [prefs[src][layer_idx][e] for src in sources]
                ax.bar(x + e * bar_width, vals, bar_width,
                       label=f'E{e}',
                       color=EXPERT_COLORS[e % len(EXPERT_COLORS)])
            ax.set_xticks(x + bar_width * (num_experts - 1) / 2)
            ax.set_xticklabels(sources, rotation=25, fontsize=8)
            ax.set_ylabel("Fraction of tokens" if col == 0 else "")
            ax.set_ylim(0, 1)
            ax.set_title(f"{token_type} — Layer {layer_idx}\n(rank #{col+1} by spec. score)",
                         fontsize=10, fontweight='bold')
            if row == 0 and col == top_n - 1:
                ax.legend(fontsize=7, loc='upper right', title='Expert')
            ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "best_layer_detail.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output, exist_ok=True)

    source_filter = set(args.sources) if args.sources else None
    data = load_routing_data(args.input, source_filter)

    if not data:
        print("ERROR: No samples loaded. Check --input path and --sources filter.")
        return

    layer_indices, num_experts = extract_layer_info(data)
    sources = get_sources(data)
    print(f"Sources: {sources}")
    print(f"MoE layers: {layer_indices}")
    print(f"Experts per layer: {num_experts}")

    ckpt_name = args.ckpt_name or os.path.splitext(os.path.basename(args.input))[0]

    img_counts, txt_counts = aggregate_expert_counts(data, layer_indices, num_experts)
    img_prefs = normalize_counts(img_counts, sources, layer_indices)
    txt_prefs = normalize_counts(txt_counts, sources, layer_indices)

    print("\nGenerating Figure A: specialization heatmap...")
    plot_specialization_heatmap(img_prefs, txt_prefs, sources, layer_indices,
                                num_experts, args.output, ckpt_name)

    print("Generating Figure B: specialization score...")
    img_scores, txt_scores = plot_specialization_score(
        img_prefs, txt_prefs, sources, layer_indices, args.output, ckpt_name
    )

    print("Generating Figure C: best-layer detail...")
    plot_best_layer_detail(img_prefs, txt_prefs, sources, layer_indices,
                           img_scores, num_experts, args.output, ckpt_name)

    # Print a quick text summary
    print("\n--- SPECIALIZATION SUMMARY ---")
    print(f"{'Layer':<8}", end="")
    print(f"{'ImgKL':>8}  {'TxtKL':>8}  ", end="")
    for src in sources:
        print(f"{'DomExp(' + src[:4] + ')':>12}", end="")
    print()
    for li, layer_idx in enumerate(layer_indices):
        print(f"L{layer_idx:<7}", end="")
        print(f"{img_scores[li]:>8.3f}  {txt_scores[li]:>8.3f}  ", end="")
        for src in sources:
            dom = int(np.argmax(img_prefs[src][layer_idx]))
            frac = img_prefs[src][layer_idx][dom]
            print(f"{'E' + str(dom) + '(' + f'{frac:.0%}' + ')':>12}", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help=".pt file from model_routing_probe_train_dist.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for plots")
    parser.add_argument("--sources", type=str, nargs='*', default=None,
                        help="Limit analysis to these sources (default: all)")
    parser.add_argument("--ckpt-name", type=str, default=None,
                        help="Checkpoint name for plot titles (default: derived from input filename)")
    args = parser.parse_args()

    main(args)
