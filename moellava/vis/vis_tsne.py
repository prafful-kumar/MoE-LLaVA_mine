import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm
import os

def visualize_category_routing(args):
    print(f"⏳ Loading data from {args.input}...")
    data = torch.load(args.input)
    
    # Get layer mapping
    first_sample = list(data.values())[0]
    if 'layer_indices' in first_sample:
        available_layers = first_sample['layer_indices']
        print(f"ℹ️  Available layers: {available_layers}")
        
        if args.layer_idx in available_layers:
            target_list_idx = available_layers.index(args.layer_idx)
            print(f"✅ Using Layer {args.layer_idx} (internal index {target_list_idx})")
        else:
            print(f"❌ Layer {args.layer_idx} not found. Available: {available_layers}")
            return
    else:
        target_list_idx = args.layer_idx
    
    # Collect data
    category_logits = defaultdict(list)
    category_expert_choices = defaultdict(list)
    
    print(f"🔍 Extracting routing patterns...")
    for sample in tqdm(data.values()):
        if target_list_idx >= len(sample['gating_logit']):
            continue
        
        logits = sample['gating_logit'][target_list_idx].float()
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        
        category = sample.get('category', 'unknown')
        output_ids = sample['output_ids'].flatten().tolist()
        
        try:
            img_start = output_ids.index(-200)
            img_end = min(img_start + 576, logits.shape[0])
            img_logits = logits[img_start:img_end]
            
            if img_logits.shape[0] > 0 and not torch.isnan(img_logits).any():
                # AVERAGE all 576 image tokens (not random sample!)
                avg_logits = img_logits.mean(dim=0)  # [num_experts]
                category_logits[category].append(avg_logits.numpy())
                
                # Also track expert choices per token
                expert_choices = torch.argmax(img_logits, dim=-1).numpy()
                category_expert_choices[category].extend(expert_choices)
        except:
            continue
    
    if len(category_logits) == 0:
        print("❌ No valid data found")
        return
    
    # Prepare data for visualization
    categories = sorted(category_logits.keys())
    X = []
    labels = []
    for cat in categories:
        X.extend(category_logits[cat])
        labels.extend([cat] * len(category_logits[cat]))
    
    X = np.array(X)
    labels = np.array(labels)
    
    # Compute t-SNE
    print(f"📊 Running t-SNE on {len(X)} samples...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Compute silhouette score (measures cluster quality)
    if len(categories) > 1:
        silhouette = silhouette_score(X_2d, labels)
        print(f"🎯 Silhouette Score: {silhouette:.3f} (higher is better, >0.5 is good)")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ============================================
    # Plot 1: t-SNE colored by category
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])
    category_colors = {
        'scene': '#E74C3C', 'food': '#F39C12', 'document': '#3498DB',
        'chart': '#2ECC71', 'code': '#9B59B6', 'text': '#E67E22'
    }
    
    for cat in categories:
        mask = labels == cat
        color = category_colors.get(cat, '#95A5A6')
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   label=cat.upper(), alpha=0.6, s=100, color=color, edgecolors='white')
    
    ax1.legend(loc='best', fontsize=10)
    ax1.set_title(f'Layer {args.layer_idx}: Category Clustering\n(Good = Distinct clusters)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ============================================
    # Plot 2: Expert preference by category (BAR CHART)
    # ============================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    num_experts = X.shape[1]
    expert_probs_by_category = {}
    
    for cat in categories:
        cat_logits = np.array(category_logits[cat])
        # Softmax to get probabilities
        cat_probs = np.exp(cat_logits) / np.exp(cat_logits).sum(axis=1, keepdims=True)
        expert_probs_by_category[cat] = cat_probs.mean(axis=0)
    
    x = np.arange(len(categories))
    width = 0.8 / num_experts
    
    expert_colors = ['#62A0CA', '#FFA556', '#6BBC6B', '#E26868']
    for expert_idx in range(num_experts):
        values = [expert_probs_by_category[cat][expert_idx] for cat in categories]
        ax2.bar(x + expert_idx * width, values, width, 
               label=f'Expert {expert_idx}', color=expert_colors[expert_idx % 4])
    
    ax2.set_xlabel('Category', fontsize=11)
    ax2.set_ylabel('Average Routing Probability', fontsize=11)
    ax2.set_title(f'Layer {args.layer_idx}: Expert Preference by Category\n(Good = Each category has distinct expert)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * (num_experts-1) / 2)
    ax2.set_xticklabels([c.upper() for c in categories], rotation=45)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # ============================================
    # Plot 3: Expert specialization (which expert handles which category most)
    # ============================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    expert_category_matrix = np.zeros((num_experts, len(categories)))
    for cat_idx, cat in enumerate(categories):
        for expert_idx in range(num_experts):
            expert_category_matrix[expert_idx, cat_idx] = expert_probs_by_category[cat][expert_idx]
    
    sns.heatmap(expert_category_matrix, ax=ax3, cmap='YlOrRd', annot=True, fmt='.2f',
               xticklabels=[c.upper() for c in categories], 
               yticklabels=[f'Expert {i}' for i in range(num_experts)],
               cbar_kws={'label': 'Routing Probability'})
    ax3.set_title(f'Layer {args.layer_idx}: Expert-Category Matrix\n(Each column should have one dominant expert)', 
                 fontsize=12, fontweight='bold')
    
    # ============================================
    # Plot 4: Token-level expert distribution per category
    # ============================================
    ax4 = fig.add_subplot(gs[1, :])
    
    category_positions = {}
    pos = 0
    for cat in categories:
        expert_counts = np.bincount(category_expert_choices[cat], minlength=num_experts)
        expert_probs = expert_counts / expert_counts.sum()
        
        x_pos = np.arange(num_experts) + pos
        bars = ax4.bar(x_pos, expert_probs, color=category_colors.get(cat, '#95A5A6'), 
                      alpha=0.7, label=cat.upper())
        
        # Add category label
        ax4.text(pos + num_experts/2, max(expert_probs) + 0.05, cat.upper(), 
                ha='center', fontweight='bold', fontsize=10)
        
        category_positions[cat] = (pos, pos + num_experts)
        pos += num_experts + 1
    
    ax4.set_xlabel('Expert ID (grouped by category)', fontsize=11)
    ax4.set_ylabel('Proportion of Tokens', fontsize=11)
    ax4.set_title(f'Layer {args.layer_idx}: Token-level Expert Selection by Category\n(Shows which expert most tokens choose for each category)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticks([])
    
    # Add vertical separators
    for pos_start, pos_end in category_positions.values():
        ax4.axvline(x=pos_end, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Comprehensive Routing Analysis - Layer {args.layer_idx}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"comprehensive_layer_{args.layer_idx}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {output_path}")
    
    # Print insights
    print("\n" + "="*60)
    print(f"📊 INSIGHTS FOR LAYER {args.layer_idx}:")
    print("="*60)
    for cat in categories:
        top_expert = np.argmax(expert_probs_by_category[cat])
        top_prob = expert_probs_by_category[cat][top_expert]
        print(f"{cat.upper():12} → Expert {top_expert} ({top_prob:.1%})")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="tsne_out/")
    parser.add_argument("--layer_idx", type=int, default=12)
    args = parser.parse_args()
    
    visualize_category_routing(args)
# ```

# ---

## 🎯 What This Improved Version Does:

# | Plot | What It Shows | Good Result |
# |------|---------------|-------------|
# | **1. t-SNE Clustering** | Do categories form distinct clusters? | 6 separate clusters |
# | **2. Expert Preference Bar** | Which expert each category prefers | Each category has 1 dominant expert |
# | **3. Heatmap Matrix** | Expert-category relationships | Each column has 1 high value |
# | **4. Token Distribution** | Actual token-level choices | Clear bars for each category |

# Plus it prints:
# ```
# SCENE        → Expert 2 (73.4%)
# FOOD         → Expert 1 (68.2%)
# DOCUMENT     → Expert 3 (81.5%)
# ...