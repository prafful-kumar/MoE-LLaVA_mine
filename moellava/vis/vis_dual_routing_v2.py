"""
vis_dual_routing_v2.py — improved routing analysis plot

Changes from v1:
1. t-SNE: still uses per-sample averaged logit vectors (correct for semantic clustering)
2. Bar charts & heatmap: now use TOKEN-LEVEL top-2 counts (correct for load analysis)
   - Each token contributes 2 expert assignments (top-2 routing)
   - Counts are aggregated across all tokens, per category/prompt_type group
3. --color_by: switch between 'category' and 'prompt_type' for t-SNE coloring
   - 'category': do image content cluster by visual category?
   - 'prompt_type': do questions of the same type cluster in routing space?

Layout (3×3):
  [0,0] IMAGE t-SNE (colors = color_by label)
  [0,1] TEXT  t-SNE (colors = color_by label)
  [0,2] Summary text box
  [1,0] IMAGE t-SNE (colors = expert argmax of averaged logit)
  [1,1] TEXT  t-SNE (colors = expert argmax of averaged logit)
  [1,2] Routing divergence heatmap (token-level top-2 counts)
  [2,0] IMAGE: expert preference bar (token-level top-2)
  [2,1] TEXT:  expert preference bar (token-level top-2)
  [2,2] Global utilization text box (token-level top-2)
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict
from tqdm import tqdm
import os


def visualize_dual_routing(args):
    print(f"⏳ Loading data from {args.input}...")
    try:
        data = torch.load(args.input, map_location='cpu')
    except Exception as e:
        print(f"❌ Failed to load {args.input}: {e}")
        return

    # --- LAYER INDEX MAPPING ---
    first_sample = list(data.values())[0]
    target_list_idx = args.layer_idx
    if 'layer_indices' in first_sample and args.layer_idx in first_sample['layer_indices']:
        target_list_idx = first_sample['layer_indices'].index(args.layer_idx)

    color_by = args.color_by  # 'category' or 'prompt_type'
    print(f"🔍 Analyzing Layer {args.layer_idx}, color_by='{color_by}'...")

    # -------------------------------------------------------------------
    # DATA EXTRACTION
    # Two parallel structures:
    #   *_samples: list of (avg_logit_vec, label)  ← for t-SNE only
    #   *_token_counts[label][expert] = int          ← token-level top-2 counts
    # -------------------------------------------------------------------
    img_samples = []   # (avg_logit_vec, label)
    txt_samples = []

    img_token_counts = defaultdict(lambda: defaultdict(int))  # label → expert → count
    txt_token_counts = defaultdict(lambda: defaultdict(int))

    num_experts = None

    for sample in tqdm(data.values()):
        if target_list_idx >= len(sample['gating_logit']):
            continue

        logits = sample['gating_logit'][target_list_idx].float()
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        if num_experts is None:
            num_experts = logits.shape[1]

        label = sample.get(color_by, 'unknown')
        output_ids = sample['output_ids'].flatten().tolist()

        try:
            if -200 not in output_ids:
                continue

            img_start = output_ids.index(-200)
            img_end = min(img_start + 576, logits.shape[0])

            img_part = logits[img_start:img_end]   # [num_img_tokens, num_experts]
            txt_part = torch.cat([logits[:img_start], logits[img_end:]], dim=0)  # [num_txt_tokens, num_experts]

            # --- t-SNE: per-sample average ---
            if img_part.shape[0] > 0:
                img_samples.append((img_part.mean(dim=0).numpy(), label))

            if txt_part.shape[0] > 0:
                txt_samples.append((txt_part.mean(dim=0).numpy(), label))

            # --- Load analysis: token-level top-2 counts ---
            # top-2 per token → each token contributes 2 expert assignments
            k = min(2, num_experts)

            if img_part.shape[0] > 0:
                top2_img = torch.topk(img_part, k=k, dim=1).indices  # [num_img_tokens, 2]
                for expert_idx in top2_img.flatten().tolist():
                    img_token_counts[label][expert_idx] += 1

            if txt_part.shape[0] > 0:
                top2_txt = torch.topk(txt_part, k=k, dim=1).indices  # [num_txt_tokens, 2]
                for expert_idx in top2_txt.flatten().tolist():
                    txt_token_counts[label][expert_idx] += 1

        except Exception:
            continue

    if not img_samples:
        print("❌ No valid samples found.")
        return

    # --- UNPACK t-SNE DATA ---
    def unpack(samples):
        X = np.array([s[0] for s in samples])
        labels = np.array([s[1] for s in samples])
        expert_ids = np.argmax(X, axis=1)  # argmax of avg logit → for t-SNE coloring only
        return X, labels, expert_ids

    X_img, lab_img, exp_img = unpack(img_samples)
    X_txt, lab_txt, exp_txt = unpack(txt_samples)

    all_labels = sorted(set(lab_img) | set(lab_txt))

    # --- TOKEN-LEVEL PREFERENCE MATRICES ---
    # prefs[label] = np.array of shape [num_experts], values = fraction of tokens routed to each expert
    def build_prefs(token_counts, labels):
        prefs = {}
        for lbl in labels:
            counts = np.array([token_counts[lbl].get(e, 0) for e in range(num_experts)], dtype=float)
            total = counts.sum()
            prefs[lbl] = counts / (total + 1e-8)
        return prefs

    img_prefs = build_prefs(img_token_counts, all_labels)
    txt_prefs = build_prefs(txt_token_counts, all_labels)

    # --- GLOBAL TOKEN UTILIZATION (top-2 counts across all labels) ---
    global_img = np.zeros(num_experts)
    global_txt = np.zeros(num_experts)
    for lbl in all_labels:
        for e in range(num_experts):
            global_img[e] += img_token_counts[lbl].get(e, 0)
            global_txt[e] += txt_token_counts[lbl].get(e, 0)
    global_img /= (global_img.sum() + 1e-8)
    global_txt /= (global_txt.sum() + 1e-8)

    # --- COLORS ---
    label_palette = [
        '#E74C3C', '#F39C12', '#3498DB', '#2ECC71',
        '#9B59B6', '#E67E22', '#1ABC9C', '#E91E63',
    ]
    label_color_map = {lbl: label_palette[i % len(label_palette)] for i, lbl in enumerate(all_labels)}
    expert_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # --- COMPUTE t-SNE ---
    print("📊 Computing t-SNE (Image)...")
    tsne_img = TSNE(n_components=2, perplexity=min(30, len(X_img) - 1), random_state=42)
    X_img_2d = tsne_img.fit_transform(X_img)

    print("📊 Computing t-SNE (Text)...")
    tsne_txt = TSNE(n_components=2, perplexity=min(30, len(X_txt) - 1), random_state=42)
    X_txt_2d = tsne_txt.fit_transform(X_txt)

    # --- PLOT ---
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    def plot_scatter(ax, X_2d, labels, use_label_colors, title):
        unique = sorted(set(labels))
        for lbl in unique:
            mask = labels == lbl
            if use_label_colors:
                c = label_color_map.get(lbl, 'gray')
                legend_text = str(lbl).upper()
            else:
                c = expert_colors[int(lbl) % len(expert_colors)]
                legend_text = f"Exp {lbl}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=legend_text,
                       alpha=0.7, s=60, c=[c], edgecolors='white', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', markerscale=1.2)
        ax.grid(True, alpha=0.2)

    color_by_label = color_by.replace('_', ' ').title()

    # Row 1: semantic space
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter(ax1, X_img_2d, lab_img, True,
                 f"IMAGE: Semantic Space\n(Colors = {color_by_label})")

    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter(ax2, X_txt_2d, lab_txt, True,
                 f"TEXT: Semantic Space\n(Colors = {color_by_label})")

    # Row 1 right: summary (token-level top-2)
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.axis('off')
    summary = f"ANALYSIS SUMMARY (Layer {args.layer_idx})\n{'='*32}\n"
    summary += f"Color by: {color_by_label}\n"
    summary += f"Routing: token-level top-2\n\n"
    summary += "DOMINANT EXPERT (IMAGE tokens):\n"
    for lbl in all_labels:
        best = np.argmax(img_prefs[lbl])
        frac = img_prefs[lbl][best]
        summary += f"  {str(lbl).upper():<12} → Exp {best} ({frac:.0%})\n"
    summary += "\nDOMINANT EXPERT (TEXT tokens):\n"
    for lbl in all_labels:
        best = np.argmax(txt_prefs[lbl])
        frac = txt_prefs[lbl][best]
        summary += f"  {str(lbl).upper():<12} → Exp {best} ({frac:.0%})\n"
    ax_summary.text(0.03, 0.97, summary, fontsize=9, family='monospace', va='top',
                    transform=ax_summary.transAxes)

    # Row 2: routing decisions (argmax of avg logit — for spatial layout only)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_scatter(ax3, X_img_2d, exp_img, False,
                 "IMAGE: Routing Decisions\n(argmax of avg logit)")

    ax4 = fig.add_subplot(gs[1, 1])
    plot_scatter(ax4, X_txt_2d, exp_txt, False,
                 "TEXT: Routing Decisions\n(argmax of avg logit)")

    # Row 2 right: divergence heatmap (token-level)
    ax6 = fig.add_subplot(gs[1, 2])
    diff_matrix = np.zeros((num_experts, len(all_labels)))
    for c_idx, lbl in enumerate(all_labels):
        for e in range(num_experts):
            diff_matrix[e, c_idx] = img_prefs[lbl][e] - txt_prefs[lbl][e]
    sns.heatmap(diff_matrix, ax=ax6, cmap='RdBu_r', center=0,
                annot=True, fmt='.2f',
                xticklabels=[str(l).upper()[:6] for l in all_labels],
                yticklabels=[f'E{e}' for e in range(num_experts)],
                cbar_kws={'label': 'Red=Img | Blue=Txt'})
    ax6.set_title("Routing Divergence\n(token-level top-2)", fontweight='bold')

    # Row 3: preference bar charts (token-level top-2)
    x = np.arange(len(all_labels))
    width = 0.8 / num_experts

    ax5 = fig.add_subplot(gs[2, 0])
    for e in range(num_experts):
        vals = [img_prefs[lbl][e] for lbl in all_labels]
        ax5.bar(x + e * width, vals, width, label=f'Exp {e}',
                color=expert_colors[e % len(expert_colors)])
    ax5.set_title("IMAGE: Expert Preference\n(token-level top-2)", fontweight='bold')
    ax5.set_xticks(x + width * (num_experts - 1) / 2)
    ax5.set_xticklabels([str(l).upper()[:6] for l in all_labels], rotation=25, fontsize=8)
    ax5.legend(fontsize=7)
    ax5.set_ylabel("Fraction of tokens")

    ax7 = fig.add_subplot(gs[2, 1])
    for e in range(num_experts):
        vals = [txt_prefs[lbl][e] for lbl in all_labels]
        ax7.bar(x + e * width, vals, width, label=f'Exp {e}',
                color=expert_colors[e % len(expert_colors)])
    ax7.set_title("TEXT: Expert Preference\n(token-level top-2)", fontweight='bold')
    ax7.set_xticks(x + width * (num_experts - 1) / 2)
    ax7.set_xticklabels([str(l).upper()[:6] for l in all_labels], rotation=25, fontsize=8)
    ax7.legend(fontsize=7)
    ax7.set_ylabel("Fraction of tokens")

    # Row 3 right: global utilization
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    stats = f"GLOBAL UTILIZATION\n{'='*28}\n"
    stats += "(token-level top-2 across all groups)\n\n"
    stats += "IMAGE tokens:\n"
    for e in range(num_experts):
        bar = '█' * int(global_img[e] * 20)
        stats += f"  Exp {e}: {global_img[e]:.1%}  {bar}\n"
    stats += "\nTEXT tokens:\n"
    for e in range(num_experts):
        bar = '█' * int(global_txt[e] * 20)
        stats += f"  Exp {e}: {global_txt[e]:.1%}  {bar}\n"

    # Routing entropy (measure of balance)
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / np.log(len(p) + 1e-8)  # normalized

    stats += f"\nRouting Entropy:\n"
    stats += f"  Image: {entropy(global_img):.3f}  (1=balanced)\n"
    stats += f"  Text:  {entropy(global_txt):.3f}  (1=balanced)\n"

    ax_stats.text(0.03, 0.97, stats, fontsize=9, family='monospace', va='top',
                  transform=ax_stats.transAxes)

    # --- SAVE ---
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"dual_analysis_layer_{args.layer_idx}_{color_by}.png")
    plt.suptitle(
        f"Routing Analysis — Layer {args.layer_idx}  |  color by: {color_by_label}",
        fontsize=15, fontweight='bold', y=0.99
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     type=str, required=True,
                        help="Path to .pt file from model_routing_probe_v2.py")
    parser.add_argument("--output",    type=str, default="analysis_out/",
                        help="Output directory for plots")
    parser.add_argument("--layer_idx", type=int, default=0,
                        help="Which MoE layer to analyze (real layer index)")
    parser.add_argument("--color_by",  type=str, default="category",
                        choices=["category", "prompt_type"],
                        help="Color t-SNE points by 'category' or 'prompt_type'")
    args = parser.parse_args()
    visualize_dual_routing(args)
