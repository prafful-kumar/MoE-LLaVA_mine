import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import joblib
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from transformers import CLIPImageProcessor

# MoE-LLaVA imports
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates

def parse_args():
    parser = argparse.ArgumentParser(description="Compute K-Means centroids or Fisher directions for MoE initialization")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", nargs='+', required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="router_init.pkl")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=40960)
    parser.add_argument("--version", type=str, default="stablelm")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--init_method", type=str, default="fisher", 
                        choices=["kmeans", "fisher"],
                        help="kmeans: save centroids (Approach 1), fisher: save Fisher directions (Approach 2)")
    parser.add_argument("--max_tokens_for_fisher", type=int, default=1000000,
                        help="Maximum tokens to use for Fisher computation (subsamples if exceeded)")
    return parser.parse_args()


def compute_fisher_directions(features, labels, num_experts, device='cpu'):
    """
    Compute oriented Fisher Linear Discriminant directions.
    
    This implementation includes proper orientation to ensure each Fisher direction
    points TOWARD its corresponding cluster, not away from it.
    
    IMPORTANT: This function processes large datasets efficiently on CPU to avoid OOM.
    
    Args:
        features: torch.Tensor [num_tokens, feature_dim]
        labels: torch.Tensor [num_tokens] - cluster assignments from K-means
        num_experts: int - number of experts (directions to compute)
        device: str - 'cpu' or 'cuda' (will force CPU for large datasets)
        
    Returns:
        directions: torch.Tensor [num_experts, feature_dim] - Oriented & Normalized Fisher directions
    """
    # Force CPU for large datasets to avoid OOM
    if features.shape[0] > 500000:  # > 500K tokens
        print(f"  Large dataset detected ({features.shape[0]:,} tokens), forcing CPU computation...")
        device = 'cpu'
    
    features = features.to(device)
    labels = labels.to(device)
    
    D = features.shape[1]
    overall_mean = features.mean(dim=0)
    
    print(f"\nComputing Fisher directions for {num_experts} experts...")
    print(f"Feature dimension: {D}")
    print(f"Total tokens: {features.shape[0]:,}")
    print(f"Computing on: {device}")
    
    # Step 1: Compute Scatter Matrices (memory-efficient)
    S_W = torch.zeros(D, D, device=device, dtype=torch.float32)
    S_B = torch.zeros(D, D, device=device, dtype=torch.float32)
    
    # Store class means for orientation step
    class_means = []
    class_sizes = []
    
    print("\nComputing scatter matrices...")
    for expert_id in range(num_experts):
        mask = (labels == expert_id)
        X_expert = features[mask]
        
        if len(X_expert) == 0:
            print(f"  Warning: Expert {expert_id} has no assigned tokens!")
            class_means.append(None)
            class_sizes.append(0)
            continue
        
        class_sizes.append(len(X_expert))
        mean_expert = X_expert.mean(dim=0)
        class_means.append(mean_expert)
        
        # Within-class scatter: Process in chunks to save memory
        print(f"  Expert {expert_id}: {len(X_expert):,} tokens", end='')
        
        # For very large clusters, process in chunks
        chunk_size = 50000
        if len(X_expert) > chunk_size:
            print(f" (processing in chunks of {chunk_size:,})")
            for start_idx in range(0, len(X_expert), chunk_size):
                end_idx = min(start_idx + chunk_size, len(X_expert))
                chunk = X_expert[start_idx:end_idx]
                centered = chunk - mean_expert
                S_W += torch.matmul(centered.t(), centered)
                del centered, chunk  # Free memory
        else:
            print()
            centered = X_expert - mean_expert
            S_W += torch.matmul(centered.t(), centered)
            del centered
        
        # Between-class scatter
        diff = (mean_expert - overall_mean).unsqueeze(1)
        S_B += len(X_expert) * torch.matmul(diff, diff.t())
        
        del X_expert, diff  # Free memory
    
    print(f"Class sizes: {class_sizes}")
    
    # Step 2: Solve Fisher Eigenproblem
    # We want to find W that maximizes: (W^T S_B W) / (W^T S_W W)
    # This is equivalent to solving: (S_W)^-1 S_B W = λW
    
    eps = 1e-4
    S_W_reg = S_W + eps * torch.eye(D, device=device, dtype=torch.float32)
    
    print("Solving Fisher eigenproblem...")
    fisher_matrix = torch.linalg.solve(S_W_reg, S_B)
    
    # Get eigenvectors (these are the Fisher discriminant directions)
    eigvals, eigvecs = torch.linalg.eigh(fisher_matrix)
    
    # Sort by eigenvalue (descending) - larger eigenvalues = more discriminative
    idx = torch.argsort(eigvals, descending=True)
    sorted_eigvals = eigvals[idx]
    sorted_eigvecs = eigvecs[:, idx]
    
    # Step 3: Extract Top Fisher Directions
    # Fisher LDA can produce at most (num_classes - 1) meaningful directions
    # This is because the between-class scatter matrix has rank ≤ (C - 1)
    
    num_fisher = min(num_experts - 1, D)
    
    # Check how many eigenvalues are significant
    threshold = 1e-6
    num_significant = (sorted_eigvals > threshold).sum().item()
    num_fisher = min(num_fisher, num_significant)
    
    print(f"Fisher provides {num_fisher} meaningful directions (theoretical max: {num_experts - 1})")
    print(f"Top 5 eigenvalues: {sorted_eigvals[:5].cpu().numpy()}")
    
    # Extract Fisher directions
    fisher_dirs = sorted_eigvecs[:, :num_fisher].t()  # [num_fisher, D]
    
    # Step 4: DIRECTION ORIENTATION (Critical step!)
    # Fisher eigenvectors define a LINE (axis), not a direction
    # Both +v and -v are valid eigenvectors, but only one points toward the cluster
    # We must orient each direction to point TOWARD its corresponding cluster
    
    print("\nOrienting Fisher directions...")
    oriented_dirs = []
    
    for i in range(num_fisher):
        direction = fisher_dirs[i]
        
        # Project each class mean (relative to overall mean) onto this direction
        # This tells us how much each cluster "aligns" with this direction
        projections = torch.tensor([
            torch.dot(direction, m - overall_mean).item() 
            if m is not None else 0.0
            for m in class_means
        ], device=device)
        
        # Find which cluster this direction best represents
        # (the one with the largest absolute projection)
        best_expert_idx = torch.argmax(torch.abs(projections))
        
        # If the direction points AWAY from its primary cluster (negative projection),
        # flip it so it points TOWARD the cluster
        if projections[best_expert_idx] < 0:
            direction = -direction
            print(f"  Direction {i}: Flipped to align with expert {best_expert_idx} "
                  f"(projection: {projections[best_expert_idx].item():.4f} → {-projections[best_expert_idx].item():.4f})")
        else:
            print(f"  Direction {i}: Already aligned with expert {best_expert_idx} "
                  f"(projection: {projections[best_expert_idx].item():.4f})")
        
        oriented_dirs.append(direction)
    
    fisher_dirs = torch.stack(oriented_dirs)
    
    # Step 5: Orthogonal Completion (if needed)
    # If we need more directions than Fisher can provide, complete with orthogonal vectors
    
    if num_fisher < num_experts:
        remaining = num_experts - num_fisher
        print(f"\nCompleting with {remaining} orthogonal vectors...")
        
        # Generate random vectors
        extra = torch.randn(remaining, D, device=device, dtype=torch.float32)
        
        # Orthogonalize against existing Fisher directions (Gram-Schmidt)
        for fisher_dir in fisher_dirs:
            # Remove component in Fisher direction
            projection = (extra @ fisher_dir.unsqueeze(1)).squeeze(1)
            extra = extra - projection.unsqueeze(1) * fisher_dir.unsqueeze(0)
        
        # Orthogonalize the extra vectors among themselves using QR
        Q, _ = torch.linalg.qr(extra.t())
        orthogonal_completion = Q[:, :remaining].t()
        
        # Combine Fisher directions + orthogonal completion
        all_directions = torch.cat([fisher_dirs, orthogonal_completion], dim=0)
        
        print(f"Added {remaining} orthogonal directions")
    else:
        all_directions = fisher_dirs
    
    # Step 6: Final L2 Normalization
    # Ensure all directions are unit vectors
    all_directions = F.normalize(all_directions, p=2, dim=-1)
    
    print(f"\nFinal directions shape: {all_directions.shape}")
    print("Direction norms:", torch.norm(all_directions, dim=-1).cpu().numpy())
    
    return all_directions


class ActivationHook:
    def __init__(self, layer_idx, num_clusters, buffer_size=10240, init_method="fisher", max_tokens=1000000):
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.buffer = []
        self.fitted = False
        self.current_mask = None
        self.init_method = init_method
        self.max_tokens = max_tokens
        
        # For Fisher: store all features (need them for scatter matrices)
        self.all_features = []
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=42,
            batch_size=buffer_size, 
            n_init="auto"
        )

    def __call__(self, module, input, output):
        # output shape: [Batch, Seq_Len_Full, Dim]
        hidden_states = output.detach().cpu().float()
        
        if self.current_mask is not None:
            mask = self.current_mask.detach().cpu().bool()  # [Batch, Seq_Len_Text]
            
            # Handle VLM sequence length mismatch (Text Mask vs Image Tokens)
            if mask.shape[1] != hidden_states.shape[1]:
                batch_size = hidden_states.shape[0]
                seq_len_full = hidden_states.shape[1]
                diff = seq_len_full - mask.shape[1]
                
                # We assume image tokens are valid and should be clustered.
                # We append 'True' to the mask to cover the visual hidden states.
                extra_mask = torch.ones((batch_size, diff), dtype=torch.bool)
                mask = torch.cat([mask, extra_mask], dim=1)
            
            # Indexing with mask flattens the tensor to [Valid_Tokens, Dim]
            try:
                hidden_states = hidden_states[mask]
            except IndexError:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # Store features as numpy for K-means
        features_np = hidden_states.numpy()
        self.buffer.append(features_np)
        
        # For Fisher: also keep features as tensors for later computation
        if self.init_method == "fisher":
            self.all_features.append(hidden_states.clone())
        
        current_tokens = sum(x.shape[0] for x in self.buffer)
        if current_tokens >= self.buffer_size:
            self._fit_buffer()

    def _fit_buffer(self):
        if not self.buffer: 
            return
        data = np.concatenate(self.buffer, axis=0)
        if data.shape[0] < self.kmeans.n_clusters: 
            return 
        self.kmeans.partial_fit(data)
        self.buffer = []
        self.fitted = True

    def finalize(self):
        """Finalize K-means and optionally compute Fisher directions"""
        self._fit_buffer()
        
        # Compute Fisher directions if requested
        if self.init_method == "fisher" and self.fitted and self.all_features:
            print(f"\n{'='*60}")
            print(f"Computing Fisher directions for layer {self.layer_idx}")
            print(f"{'='*60}")
            
            # Concatenate all collected features
            all_feats = torch.cat(self.all_features, dim=0)
            print(f"Total features collected: {all_feats.shape[0]:,}")
            
            # Subsample if dataset is too large
            if all_feats.shape[0] > self.max_tokens:
                print(f"Subsampling to {self.max_tokens:,} tokens to save memory...")
                indices = torch.randperm(all_feats.shape[0])[:self.max_tokens]
                all_feats = all_feats[indices]
                print(f"Using {all_feats.shape[0]:,} tokens for Fisher computation")
            
            # Get cluster assignments from K-means (these are our pseudo-labels)
            print("Obtaining K-means labels...")
            labels = self.kmeans.predict(all_feats.numpy())
            labels = torch.from_numpy(labels)
            
            # Compute Fisher directions (force CPU to avoid OOM)
            # Fisher computation is memory-intensive, CPU is safer for large datasets
            self.fisher_directions = compute_fisher_directions(
                all_feats, 
                labels, 
                self.kmeans.n_clusters,
                device='cpu'  # Force CPU to avoid OOM
            )
            
            # Move back to numpy for saving
            self.fisher_directions = self.fisher_directions.numpy()
            
            print(f"\nFisher directions computed successfully for layer {self.layer_idx}")
            print(f"{'='*60}\n")
            
            # Clean up memory
            self.all_features = []
            del all_feats, labels
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"MoE Router Initialization - {args.init_method.upper()} Method")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Num Experts: {args.num_experts}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Init Method: {args.init_method}")
    print(f"Output: {args.output_file}")
    print(f"{'='*60}\n")
    
    # Load Dense Architecture
    model_name = "llava-stablelm"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if isinstance(image_processor, dict) or image_processor is None:
        print("Image processor is a dict or None. Loading directly from model path...")
        try:
            image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
        except Exception as e:
            print(f"Could not load from model_path: {e}. Falling back to default CLIP...")
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    if not hasattr(image_processor, 'image_mean'):
        image_processor.image_mean = getattr(image_processor, 'mean', [0.48145466, 0.4578275, 0.40821073])
        
    model.eval().cuda()
    model.config.image_aspect_ratio = args.image_aspect_ratio
    dtype = model.dtype 

    # Setup Hooks
    num_layers = model.config.num_hidden_layers
    moe_layers_idx = list(range(0, num_layers, 2)) 
    hooks = {}
    
    print(f"Setting up hooks for {len(moe_layers_idx)} MoE layers: {moe_layers_idx}")
    
    for idx in moe_layers_idx:
        target_layer = model.model.layers[idx].post_attention_layernorm
        hook_obj = ActivationHook(
            idx, 
            args.num_experts, 
            args.buffer_size, 
            args.init_method,
            args.max_tokens_for_fisher
        )
        handle = target_layer.register_forward_hook(hook_obj)
        hooks[idx] = (handle, hook_obj)

    # Load Data
    all_data = []
    for path in args.data_path:
        print(f"Loading data from: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    print(f"Total data samples: {len(all_data)}")
    
    # Validate data format
    print("\nValidating data format...")
    valid_samples = 0
    for i, item in enumerate(all_data[:100]):  # Check first 100 samples
        if 'conversations' in item and len(item['conversations']) > 0:
            valid_samples += 1
        elif i < 10:  # Show first few invalid samples
            print(f"  Warning: Sample {i} missing 'conversations' field")
    
    if valid_samples == 0:
        print("ERROR: No valid samples found! Check your data format.")
        return
    
    print(f"Validation: {valid_samples}/100 samples have correct format")
    
    np.random.shuffle(all_data)
    all_data = all_data[:args.num_samples]
    print(f"Using {len(all_data)} samples for initialization\n")

    # Processing Loop
    print("Processing batches...")
    error_count = 0
    success_count = 0
    
    for i in tqdm(range(0, len(all_data), args.batch_size), desc="Collecting features"):
        batch_items = all_data[i : i + args.batch_size]
        images, input_ids_list = [], []
        image_indices = []  # Track which items have images
        
        for idx, item in enumerate(batch_items):
            try:
                has_image = 'image' in item
                
                # Process image if present
                if has_image:
                    try:
                        img_path = os.path.join(args.image_folder, item['image'])
                        img = Image.open(img_path).convert('RGB')
                        processed_img = process_images([img], image_processor, model.config)[0]
                        images.append(processed_img)
                        image_indices.append(idx)
                    except Exception as e:
                        # Skip this item if image processing fails
                        continue
                
                # Get question text
                if 'conversations' not in item or len(item['conversations']) == 0:
                    continue
                
                qs = item['conversations'][0]['value']
                
                # Add image token if needed
                if has_image and DEFAULT_IMAGE_TOKEN not in qs:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
                # Create conversation
                conv = conv_templates[args.version].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                
                # Tokenize
                input_ids = tokenizer_image_token(
                    conv.get_prompt(), 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors='pt'
                )
                input_ids_list.append(input_ids)
                
            except Exception as e:
                # Skip problematic items silently
                continue

        # Skip batch if no valid items
        if not input_ids_list: 
            continue
        
        # Check image/text alignment
        if images and len(images) != len(input_ids_list):
            # Mismatch between images and text - skip this batch
            error_count += 1
            continue

        try:
            # Padding & Masking
            max_len = max(x.shape[0] for x in input_ids_list)
            input_ids = torch.stack([
                torch.cat([x, torch.full((max_len - x.shape[0],), tokenizer.pad_token_id, dtype=torch.long)]) 
                for x in input_ids_list
            ]).cuda()
            attention_mask = input_ids.ne(tokenizer.pad_token_id).cuda()
            
            # Update masks in hooks before forward pass
            for _, h in hooks.values():
                h.current_mask = attention_mask

            # Prepare inputs
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if images:
                inputs['images'] = torch.stack(images).to(dtype=dtype, device='cuda')

            # Forward pass
            with torch.no_grad():
                model(**inputs)
                success_count += 1
                
        except Exception as e:
            error_count += 1
            # Only print detailed errors occasionally
            if error_count % 100 == 1:
                print(f"\nError details (batch {i}): {type(e).__name__}: {str(e)}")
            torch.cuda.empty_cache()
            continue

    # Finalize and Save
    print("\n" + "="*60)
    print("Processing Summary:")
    print(f"  Successful batches: {success_count}")
    print(f"  Failed batches: {error_count}")
    print(f"  Success rate: {100*success_count/(success_count+error_count):.1f}%")
    print("="*60)
    print("\nFinalizing and saving results...")
    print("="*60 + "\n")
    
    layer_results = {}
    for idx, (handle, h) in hooks.items():
        h.finalize()
        handle.remove()
        
        if h.fitted:
            if args.init_method == "fisher":
                # Save Fisher directions
                if hasattr(h, 'fisher_directions'):
                    layer_results[idx] = h.fisher_directions
                    print(f"Layer {idx}: Fisher directions shape {h.fisher_directions.shape}")
                else:
                    print(f"Warning: Layer {idx} did not compute Fisher directions")
            else:
                # Save K-means centroids
                layer_results[idx] = h.kmeans.cluster_centers_
                print(f"Layer {idx}: K-means centroids shape {h.kmeans.cluster_centers_.shape}")

    joblib.dump(layer_results, args.output_file)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Saved {args.init_method} initialization to {args.output_file}")
    print(f"Layers processed: {list(layer_results.keys())}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
# Generate Fisher directions
# python compute_fisher_directions.py \
#     --model_path /path/to/model \
#     --data_path train_data.json \
#     --image_folder /path/to/images \
#     --output_file fisher_directions.pkl \
#     --num_experts 4 \
#     --init_method fisher