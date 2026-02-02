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

    # Layer Mapping
    first_sample = list(data.values())[0]
    target_list_idx = args.layer_idx
    if 'layer_indices' in first_sample and args.layer_idx in first_sample['layer_indices']:
        target_list_idx = first_sample['layer_indices'].index(args.layer_idx)

    # --- DATA EXTRACTION ---
    img_samples = []  # (logit_vector, category_label)
    text_samples = [] 
    
    print(f"🔍 Analyzing Layer {args.layer_idx}...")
    
    for sample in tqdm(data.values()):
        if target_list_idx >= len(sample['gating_logit']): continue

        logits = sample['gating_logit'][target_list_idx].float()
        if logits.dim() == 3: logits = logits.squeeze(0)
        
        category = sample.get('category', 'unknown')
        output_ids = sample['output_ids'].flatten().tolist()
        
        try:
            if -200 in output_ids:
                img_start = output_ids.index(-200)
                img_end = min(img_start + 576, logits.shape[0])
                
                # 1. IMAGE PART
                img_part = logits[img_start:img_end]
                if img_part.shape[0] > 0:
                    avg_vec = img_part.mean(dim=0).numpy()
                    img_samples.append((avg_vec, category))
                
                # 2. TEXT PART
                txt_part = torch.cat([logits[:img_start], logits[img_end:]], dim=0)
                if txt_part.shape[0] > 0:
                    avg_vec = txt_part.mean(dim=0).numpy()
                    text_samples.append((avg_vec, category))
                    
        except Exception:
            continue

    if not img_samples:
        print("❌ No valid samples found.")
        return

    # --- UNPACK DATA ---
    def unpack_data(samples):
        X = np.array([s[0] for s in samples])
        cats = np.array([s[1] for s in samples])
        expert_ids = np.argmax(X, axis=1) 
        return X, cats, expert_ids

    X_img, cat_img, exp_img = unpack_data(img_samples)
    X_txt, cat_txt, exp_txt = unpack_data(text_samples)
    
    categories = sorted(list(set(cat_img) | set(cat_txt)))
    num_experts = X_img.shape[1]

    # --- COMPUTE t-SNE ---
    print("📊 Computing t-SNE (Image)...")
    tsne_img = TSNE(n_components=2, perplexity=min(30, len(X_img)-1), random_state=42)
    X_img_2d = tsne_img.fit_transform(X_img)
    
    print("📊 Computing t-SNE (Text)...")
    tsne_txt = TSNE(n_components=2, perplexity=min(30, len(X_txt)-1), random_state=42)
    X_txt_2d = tsne_txt.fit_transform(X_txt)

    # --- PLOTTING CONFIGURATION ---
    # 3x3 Grid to fill all space
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)
    
    cat_colors = {
        'scene': '#E74C3C', 'food': '#F39C12', 'document': '#3498DB',
        'chart': '#2ECC71', 'code': '#9B59B6', 'text': '#E67E22'
    }
    expert_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Helper: Scatter Plot
    def plot_scatter(ax, X_2d, labels, label_type, title):
        unique_labels = sorted(list(set(labels)))
        for lbl in unique_labels:
            mask = labels == lbl
            if label_type == 'category':
                c = cat_colors.get(lbl, 'gray')
                label_text = lbl.upper()
            else: # expert
                c = expert_colors[int(lbl) % len(expert_colors)]
                label_text = f"Exp {lbl}"
            
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label_text,
                       alpha=0.7, s=60, c=[c], edgecolors='white', linewidth=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', markerscale=1.2)
        ax.grid(True, alpha=0.2)

    # === ROW 1: SEMANTIC CLUSTERING ===
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter(ax1, X_img_2d, cat_img, 'category', "IMAGE: Semantic Space\n(Colors = Categories)")
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter(ax2, X_txt_2d, cat_txt, 'category', "TEXT: Semantic Space\n(Colors = Categories)")

    # === ROW 2: DECISION BOUNDARIES & HEATMAP ===
    ax3 = fig.add_subplot(gs[1, 0])
    plot_scatter(ax3, X_img_2d, exp_img, 'expert', "IMAGE: Routing Decisions\n(Colors = Experts)")
    
    ax4 = fig.add_subplot(gs[1, 1])
    plot_scatter(ax4, X_txt_2d, exp_txt, 'expert', "TEXT: Routing Decisions\n(Colors = Experts)")

    # === PREFS CALCULATION ===
    def get_prefs(samples, labels):
        data_dict = defaultdict(list)
        for i, (vec, _) in enumerate(samples):
            data_dict[labels[i]].append(vec)
            
        prefs = {}
        for cat in categories:
            if cat in data_dict:
                logits = np.array(data_dict[cat])
                shift = logits - np.max(logits, axis=1, keepdims=True)
                exps = np.exp(shift)
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                prefs[cat] = probs.mean(axis=0)
            else:
                prefs[cat] = np.zeros(num_experts)
        return prefs

    img_prefs = get_prefs(img_samples, cat_img)
    text_prefs = get_prefs(text_samples, cat_txt)

    # === ROW 1 (Right): SUMMARY TEXT ===
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_summary.axis('off')
    summary_txt = f"ANALYSIS SUMMARY (Layer {args.layer_idx})\n{'='*30}\n\n"
    
    summary_txt += "VISUAL SEMANTIC ROUTING:\n"
    for cat in categories:
        best_exp = np.argmax(img_prefs[cat])
        prob = img_prefs[cat][best_exp]
        summary_txt += f"  {cat.upper():<10} -> Exp {best_exp} ({prob:.0%})\n"
    
    ax_summary.text(0.05, 0.95, summary_txt, fontsize=11, family='monospace', va='top')

    # === ROW 2 (Right): HEATMAP ===
    ax6 = fig.add_subplot(gs[1, 2])
    diff_matrix = np.zeros((num_experts, len(categories)))
    for c_idx, cat in enumerate(categories):
        for e in range(num_experts):
            diff_matrix[e, c_idx] = img_prefs[cat][e] - text_prefs[cat][e]
    sns.heatmap(diff_matrix, ax=ax6, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                xticklabels=[c.upper() for c in categories], yticklabels=[f'E{e}' for e in range(num_experts)],
                cbar_kws={'label': 'Red=Img Pref | Blue=Txt Pref'})
    ax6.set_title("Routing Divergence", fontweight='bold')

    # === ROW 3: BAR CHARTS & USAGE STATS ===
    
    # 1. Image Pref Bar
    ax5 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(categories))
    width = 0.8 / num_experts
    for i in range(num_experts):
        vals = [img_prefs[cat][i] for cat in categories]
        ax5.bar(x + i*width, vals, width, label=f'Exp {i}', color=expert_colors[i % len(expert_colors)])
    ax5.set_title("IMAGE: Expert Preference", fontweight='bold')
    ax5.set_xticks(x + width*(num_experts-1)/2)
    ax5.set_xticklabels([c.upper() for c in categories], rotation=25, fontsize=8)

    # 2. Text Pref Bar (RESTORED)
    ax_txt_bar = fig.add_subplot(gs[2, 1])
    for i in range(num_experts):
        vals = [text_prefs[cat][i] for cat in categories]
        ax_txt_bar.bar(x + i*width, vals, width, label=f'Exp {i}', color=expert_colors[i % len(expert_colors)])
    ax_txt_bar.set_title("TEXT: Expert Preference", fontweight='bold')
    ax_txt_bar.set_xticks(x + width*(num_experts-1)/2)
    ax_txt_bar.set_xticklabels([c.upper() for c in categories], rotation=25, fontsize=8)

    # 3. Global Stats (New Utilization Box)
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')
    
    # Calculate Global Usage
    total_img = len(exp_img)
    total_txt = len(exp_txt)
    usage_img = np.bincount(exp_img, minlength=num_experts)
    usage_txt = np.bincount(exp_txt, minlength=num_experts)

    stats_txt = f"GLOBAL EXPERT UTILIZATION\n{'='*30}\n\n"
    stats_txt += "IMAGE TOKENS (Visual Load):\n"
    for i in range(num_experts):
        stats_txt += f"  Expert {i}: {usage_img[i]/total_img:.1%}\n"
        
    stats_txt += "\nTEXT TOKENS (Language Load):\n"
    for i in range(num_experts):
        stats_txt += f"  Expert {i}: {usage_txt[i]/total_txt:.1%}\n"

    ax_stats.text(0.05, 0.95, stats_txt, fontsize=11, family='monospace', va='top')

    # Save
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"dual_analysis_layer_{args.layer_idx}.png")
    plt.suptitle(f"Routing Decision Boundaries - Layer {args.layer_idx}", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis_out/")
    parser.add_argument("--layer_idx", type=int, default=12)
    args = parser.parse_args()
    visualize_dual_routing(args)