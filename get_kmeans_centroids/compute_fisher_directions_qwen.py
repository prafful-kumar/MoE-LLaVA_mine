import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import joblib
import traceback
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from transformers import CLIPImageProcessor

# MoE-LLaVA imports
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Fisher directions for Qwen MoE initialization")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", nargs='+', required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="fisher_directions_qwen/5000.pkl")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=40960)
    parser.add_argument("--version", type=str, default="qwen")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--init_method", type=str, default="fisher")
    parser.add_argument("--max_tokens_for_fisher", type=int, default=1000000)
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DeepSpeed launcher")
    return parser.parse_args()


def compute_fisher_directions(features, labels, num_experts, device='cpu'):
    """Compute oriented Fisher Linear Discriminant directions."""
    if features.shape[0] > 500000:  
        print(f"  Large dataset detected ({features.shape[0]:,} tokens), forcing CPU computation...")
        device = 'cpu'
    
    features = features.to(device)
    labels = labels.to(device)
    
    D = features.shape[1]
    overall_mean = features.mean(dim=0)
    
    S_W = torch.zeros(D, D, device=device, dtype=torch.float32)
    S_B = torch.zeros(D, D, device=device, dtype=torch.float32)
    class_means, class_sizes = [], []
    
    print("\nComputing scatter matrices...")
    for expert_id in range(num_experts):
        mask = (labels == expert_id)
        X_expert = features[mask]
        
        if len(X_expert) == 0:
            class_means.append(None)
            class_sizes.append(0)
            continue
        
        class_sizes.append(len(X_expert))
        mean_expert = X_expert.mean(dim=0)
        class_means.append(mean_expert)
        
        centered = X_expert - mean_expert
        S_W += torch.matmul(centered.t(), centered)
        
        diff = (mean_expert - overall_mean).unsqueeze(1)
        S_B += len(X_expert) * torch.matmul(diff, diff.t())
        del X_expert, centered, diff
    
    eps = 1e-4
    S_W_reg = S_W + eps * torch.eye(D, device=device, dtype=torch.float32)
    
    print("Solving Fisher eigenproblem...")
    fisher_matrix = torch.linalg.solve(S_W_reg, S_B)
    eigvals, eigvecs = torch.linalg.eigh(fisher_matrix)
    
    idx = torch.argsort(eigvals, descending=True)
    sorted_eigvecs = eigvecs[:, idx]
    
    num_fisher = min(num_experts - 1, D)
    fisher_dirs = sorted_eigvecs[:, :num_fisher].t()
    
    print("\nOrienting Fisher directions...")
    oriented_dirs = []
    for i in range(num_fisher):
        direction = fisher_dirs[i]
        projections = torch.tensor([
            torch.dot(direction, m - overall_mean).item() if m is not None else 0.0
            for m in class_means
        ], device=device)
        
        best_expert_idx = torch.argmax(torch.abs(projections))
        if projections[best_expert_idx] < 0:
            direction = -direction
        oriented_dirs.append(direction)
    
    fisher_dirs = torch.stack(oriented_dirs)
    
    if num_fisher < num_experts:
        remaining = num_experts - num_fisher
        extra = torch.randn(remaining, D, device=device, dtype=torch.float32)
        for fisher_dir in fisher_dirs:
            projection = (extra @ fisher_dir.unsqueeze(1)).squeeze(1)
            extra = extra - projection.unsqueeze(1) * fisher_dir.unsqueeze(0)
        Q, _ = torch.linalg.qr(extra.t())
        orthogonal_completion = Q[:, :remaining].t()
        all_directions = torch.cat([fisher_dirs, orthogonal_completion], dim=0)
    else:
        all_directions = fisher_dirs
    
    all_directions = F.normalize(all_directions, p=2, dim=-1)
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
        self.all_features = []
        
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=buffer_size, n_init="auto")

    def __call__(self, module, input, output):
        # QWEN FIX: Handle raw tensor output from ln_2 safely
        output_tensor = output[0] if isinstance(output, tuple) else output
        hidden_states = output_tensor.detach().cpu().float()
        
        if self.current_mask is not None:
            mask = self.current_mask.detach().cpu().bool()
            if mask.shape[1] != hidden_states.shape[1]:
                batch_size = hidden_states.shape[0]
                diff = hidden_states.shape[1] - mask.shape[1]
                extra_mask = torch.ones((batch_size, diff), dtype=torch.bool)
                mask = torch.cat([mask, extra_mask], dim=1)
            
            try:
                hidden_states = hidden_states[mask]
            except IndexError:
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        features_np = hidden_states.numpy()
        self.buffer.append(features_np)
        
        if self.init_method == "fisher":
            self.all_features.append(hidden_states.clone())
        
        current_tokens = sum(x.shape[0] for x in self.buffer)
        if current_tokens >= self.buffer_size:
            self._fit_buffer()

    def _fit_buffer(self):
        if not self.buffer: return
        data = np.concatenate(self.buffer, axis=0)
        if data.shape[0] < self.kmeans.n_clusters: return 
        self.kmeans.partial_fit(data)
        self.buffer = []
        self.fitted = True

    def finalize(self):
        self._fit_buffer()
        if self.init_method == "fisher" and self.fitted and self.all_features:
            print(f"\nComputing Fisher directions for layer {self.layer_idx}")
            all_feats = torch.cat(self.all_features, dim=0)
            
            if all_feats.shape[0] > self.max_tokens:
                indices = torch.randperm(all_feats.shape[0])[:self.max_tokens]
                all_feats = all_feats[indices]
            
            labels = torch.from_numpy(self.kmeans.predict(all_feats.numpy()))
            self.fisher_directions = compute_fisher_directions(all_feats, labels, self.kmeans.n_clusters, device='cpu').numpy()
            
            self.all_features = []
            del all_feats, labels
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    print(f"\n{'='*60}\nQWEN MoE Router Initialization - {args.init_method.upper()}\n{'='*60}")
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # QWEN FIX 1: Pad Token
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Set tokenizer.pad_token_id to eos_token_id for Qwen.")

    if isinstance(image_processor, dict) or image_processor is None:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    if not hasattr(image_processor, 'image_mean'):
        image_processor.image_mean = getattr(image_processor, 'mean', [0.48145466, 0.4578275, 0.40821073])
        
    model.eval().cuda()
    model.config.image_aspect_ratio = args.image_aspect_ratio
    dtype = model.dtype 

    # QWEN FIX 2: Target Qwen Transformer Blocks and ln_2
    num_layers = model.config.num_hidden_layers
    moe_layers_idx = list(range(0, num_layers, 2)) 
    hooks = {}
    
    print(f"Setting up hooks for {len(moe_layers_idx)} Qwen MoE layers: {moe_layers_idx}")
    
    if not (hasattr(model, 'transformer') and hasattr(model.transformer, 'h')):
        raise AttributeError("This script is strictly for Qwen. Could not find model.transformer.h")
        
    for idx in moe_layers_idx:
        layer_module = model.transformer.h[idx]
        if not hasattr(layer_module, 'ln_2'):
            raise AttributeError(f"Could not find ln_2 in Qwen block {idx}")
            
        target_layer = layer_module.ln_2
        hook_obj = ActivationHook(idx, args.num_experts, args.buffer_size, args.init_method, args.max_tokens_for_fisher)
        handle = target_layer.register_forward_hook(hook_obj)
        hooks[idx] = (handle, hook_obj)

    # Load Data
    all_data = []
    for path in args.data_path:
        with open(path, 'r') as f:
            all_data.extend(json.load(f))
    
    np.random.shuffle(all_data)
    all_data = all_data[:args.num_samples]
    print(f"Using {len(all_data)} samples for initialization\n")

    error_count, success_count = 0, 0
    
    for i in tqdm(range(0, len(all_data), args.batch_size), desc="Collecting Qwen features"):
        batch_items = all_data[i : i + args.batch_size]
        images, input_ids_list = [], []
        
        for idx, item in enumerate(batch_items):
            try:
                has_image = 'image' in item
                tmp_image = None
                
                if has_image:
                    img_path = os.path.join(args.image_folder, item['image'])
                    img = Image.open(img_path).convert('RGB')
                    tmp_image = process_images([img], image_processor, model.config)[0]
                
                if 'conversations' not in item or len(item['conversations']) == 0:
                    continue
                
                qs = item['conversations'][0]['value']
                if has_image and DEFAULT_IMAGE_TOKEN not in qs:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
                # QWEN FIX 3: Safe conversation template fallback
                try:
                    conv = conv_templates["qwen"].copy()
                except KeyError:
                    conv = conv_templates["qwen_1_5"].copy()

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                
                input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                
                # QWEN FIX 4: Only append if BOTH succeed
                if has_image:
                    images.append(tmp_image)
                input_ids_list.append(input_ids)
                
            except Exception as e:
                print(f"\n[Warning] Data Processing Error skipped: {type(e).__name__} - {str(e)}")
                continue

        if not input_ids_list or (images and len(images) != len(input_ids_list)): 
            error_count += 1
            continue

        try:
            max_len = max(x.shape[0] for x in input_ids_list)
            input_ids = torch.stack([
                torch.cat([x, torch.full((max_len - x.shape[0],), tokenizer.pad_token_id, dtype=torch.long)]) 
                for x in input_ids_list
            ]).cuda()
            attention_mask = input_ids.ne(tokenizer.pad_token_id).cuda()
            
            for _, h in hooks.values():
                h.current_mask = attention_mask

            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if images:
                inputs['images'] = torch.stack(images).to(dtype=dtype, device='cuda')

            with torch.no_grad():
                model(**inputs)
                success_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"\n[CRITICAL] Model Forward Error: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    print(f"\nSummary - Success: {success_count} | Failed: {error_count}")
    
    layer_results = {}
    for idx, (handle, h) in hooks.items():
        h.finalize()
        handle.remove()
        if h.fitted:
            layer_results[idx] = h.fisher_directions if args.init_method == "fisher" else h.kmeans.cluster_centers_

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    joblib.dump(layer_results, args.output_file)
    print(f"\nSUCCESS: Saved {args.init_method} initialization to {args.output_file}")

if __name__ == "__main__":
    main()