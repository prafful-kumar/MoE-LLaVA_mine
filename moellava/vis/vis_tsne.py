import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

def visualize_tsne(args):
    print(f"⏳ Loading data from {args.input}...")
    try:
        data = torch.load(args.input)
    except Exception as e:
        print(f"❌ Failed to load .pt file: {e}")
        return

    all_logits = []
    all_categories = []
    all_expert_choices = []
    
    # --- NEW MAPPING LOGIC ---
    # Retrieve the first sample to check available layers
    first_sample = list(data.values())[0]
    
    target_list_idx = -1
    
    if 'layer_indices' in first_sample:
        # We have the mapping!
        available_layers = first_sample['layer_indices']
        print(f"ℹ️  Found Layer Mapping: {available_layers}")
        
        if args.layer_idx in available_layers:
            target_list_idx = available_layers.index(args.layer_idx)
            print(f"✅ Mapping Original Layer {args.layer_idx} -> Internal Index {target_list_idx}")
        else:
            print(f"❌ Error: Layer {args.layer_idx} was not captured in the .pt file.")
            print(f"   Available layers: {available_layers}")
            return
    else:
        # Fallback for old files
        print("⚠️ Warning: No layer mapping found in file. Using raw index.")
        target_list_idx = args.layer_idx

    print(f"🔍 Extracting logits...")
    
    samples = list(data.values())
    skipped = 0
    
    for i, sample in enumerate(tqdm(samples)):
        # Use the mapped index
        if target_list_idx >= len(sample['gating_logit']):
            skipped += 1
            continue
            
        logits = sample['gating_logit'][target_list_idx].float()
        if logits.dim() == 3: logits = logits.squeeze(0)
            
        out_ids_tensor = sample['output_ids']
        output_ids = out_ids_tensor.flatten().tolist() if isinstance(out_ids_tensor, torch.Tensor) else out_ids_tensor

        img_token_id = -200
        try:
            img_start_idx = output_ids.index(img_token_id)
            end_idx = min(img_start_idx + 576, logits.shape[0])
            
            if end_idx <= img_start_idx:
                skipped += 1
                continue

            img_logits = logits[img_start_idx:end_idx]
            
            if img_logits.shape[0] > 0:
                # Take 20 random tokens
                num_samples = min(20, img_logits.shape[0])
                indices = torch.randperm(img_logits.shape[0])[:num_samples]
                selected_logits = img_logits[indices]
                
                if not torch.isnan(selected_logits).any():
                    all_logits.append(selected_logits)
                    all_categories.extend([sample.get('category', 'unknown')] * len(selected_logits))
                    all_expert_choices.extend(torch.argmax(selected_logits, dim=-1).numpy())
                else:
                    skipped += 1
            else:
                skipped += 1
        except ValueError:
            skipped += 1
            continue

    if len(all_logits) == 0:
        print("❌ Error: No valid data found.")
        return

    # Concatenate & Plot
    X = torch.cat(all_logits, dim=0).numpy()
    categories = np.array(all_categories)
    experts = np.array(all_expert_choices)
    
    print(f"📊 Running t-SNE on {X.shape[0]} vectors...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), n_iter=1000, random_state=42, verbose=1)
    X_2d = tsne.fit_transform(X)
    
    print("🎨 Generating Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot A: Categories
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=categories, palette="tab10", s=15, alpha=0.6, ax=axes[0])
    axes[0].set_title(f"Layer {args.layer_idx} (Original): Semantic Clusters")
    
    # Plot B: Experts
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=experts, palette="bright", s=15, alpha=0.6, ax=axes[1])
    axes[1].set_title(f"Layer {args.layer_idx} (Original): Expert Selection")
    
    plt.tight_layout()
    output_path = args.output+f"{args.layer_idx}.png"
    os.makedirs(args.output,exist_ok=True)
    if args.output:
        plt.savefig(args.output+f"{args.layer_idx}.png", dpi=300)
        print(f"✅ Saved to {args.output+str(args.layer_idx)}.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="tsne_out/")
    parser.add_argument("--layer_idx", type=int, default=12, help="Original Model Layer Index")
    args = parser.parse_args()

    visualize_tsne(args)