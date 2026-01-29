import os
import argparse
import torch
import numpy as np
import json
import joblib
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# MoE-LLaVA specific imports
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle

def parse_args():
    parser = argparse.ArgumentParser(description="Compute K-Means centroids with exact training data replication")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained dense VLM")
    parser.add_argument("--model_base", type=str, default=None, help="Base model if using adapters")
    
    # Matches training script behavior (multiple files)
    parser.add_argument("--data_path", nargs='+', required=True, help="Path(s) to the json training data file(s)")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output_file", type=str, default="centroids.pkl", help="Where to save the centroids")
    
    # Centroid Hyperparams
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts (Clusters)")
    parser.add_argument("--num_samples", type=int, default=20000, help="Number of samples to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--buffer_size", type=int, default=40960)
    
    # CRITICAL: Match Training Arguments
    parser.add_argument("--version", type=str, default="stablelm", help="Conversation version (e.g., stablelm, v1)")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad", help="Image processing mode")
    
    return parser.parse_args()

class ActivationHook:
    def __init__(self, layer_idx, num_clusters, buffer_size=10240):
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.buffer = []
        self.fitted = False
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=42,
            batch_size=buffer_size, 
            n_init="auto",
            reassignment_ratio=0.001 
        )

    def __call__(self, module, input, output):
        # Detach, move to CPU, float32, numpy
        hidden_states = output.detach().cpu().float().numpy()
        # Flatten to [Tokens, Dim]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        self.buffer.append(hidden_states)
        
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
# ... (Imports remain the same)
from transformers import CLIPImageProcessor # Ensure this is imported

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # Force 'llava-stablelm' to load the Dense architecture + Tokenizer correctly
    model_name = "llava-stablelm"
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )
    except OSError as e:
        print(f"Standard loading failed: {e}. Attempting manual tokenizer registration...")
        try:
            from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
            tokenizer = Arcade100kTokenizer.from_pretrained(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, args.model_base, model_name, tokenizer=tokenizer
            )
        except Exception as inner_e:
            print(f"Manual loading also failed: {inner_e}")
            raise e

    # ================= [FIX STARTS HERE] =================
    # The error 'dict object has no attribute image_mean' happens because 
    # image_processor is sometimes returned as a raw config dictionary.
    # We must convert it to a real CLIPImageProcessor object.
    
    if isinstance(image_processor, dict):
        print("Detected image_processor as dict. Converting to CLIPImageProcessor object...")
        try:
            # Try loading from the dict
            image_processor = CLIPImageProcessor.from_dict(image_processor)
        except Exception:
            # Fallback: Load directly from the model path or default to standard CLIP
            print("Conversion failed. Reloading processor from model path...")
            try:
                image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
            except:
                # Absolute fallback to the standard LLaVA visual tower
                print("Reloading failed. Using default openai/clip-vit-large-patch14-336...")
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # Verify it has the required attribute
    if not hasattr(image_processor, 'image_mean'):
        # Some processors use 'mean' instead of 'image_mean', or it might still be broken
        if hasattr(image_processor, 'mean'):
            image_processor.image_mean = image_processor.mean
        else:
            raise AttributeError("image_processor is missing 'image_mean' attribute. Please check the loaded processor.")
    # ================= [FIX ENDS HERE] =================

    # Configure Model
    model.eval()
    model.cuda()
    model.config.image_aspect_ratio = args.image_aspect_ratio
    dtype = model.dtype 

    # 2. Setup Hooks (Alternate Layers)
    num_layers = model.config.num_hidden_layers
    moe_layers_idx = list(range(0, num_layers, 2)) 
    print(f"Hooking layers: {moe_layers_idx}")

    hooks = {}
    for idx in moe_layers_idx:
        layer_module = model.model.layers[idx]
        if hasattr(layer_module, 'post_attention_layernorm'):
            target_layer = layer_module.post_attention_layernorm
        elif hasattr(layer_module, 'input_layernorm'): 
             target_layer = layer_module.input_layernorm
        else:
            raise AttributeError(f"Could not find post-attention norm in layer {idx}")

        hook_obj = ActivationHook(idx, args.num_experts, buffer_size=args.buffer_size)
        handle = target_layer.register_forward_hook(hook_obj)
        hooks[idx] = (handle, hook_obj)

    # 3. Load and Concatenate Data
    all_data = []
    if isinstance(args.data_path, str): args.data_path = [args.data_path]
    
    for path in args.data_path:
        print(f"Loading data from {path}...")
        with open(path, 'r') as f:
            all_data.extend(json.load(f))

    np.random.shuffle(all_data)
    all_data = all_data[:args.num_samples]
    
    print(f"Processing {len(all_data)} samples with mode '{args.version}'...")

    # 4. Processing Loop
    for i in tqdm(range(0, len(all_data), args.batch_size)):
        batch_items = all_data[i : i + args.batch_size]
        
        images = []
        input_ids_list = []
        
        for item in batch_items:
            # --- Image Handling ---
            image_tensor = None
            has_image = 'image' in item
            
            if has_image:
                image_file = item['image']
                try:
                    # Construct full path
                    image_path = os.path.join(args.image_folder, image_file)
                    image = Image.open(image_path).convert('RGB')
                    
                    # Process Image
                    # NOTE: process_images requires a LIST of images
                    image_tensor = process_images([image], image_processor, model.config)[0]
                    images.append(image_tensor)
                except Exception as e:
                    # Print error but don't crash the whole script for one bad image
                    # print(f"Failed to load image {image_file}: {e}") 
                    continue 

            # --- Text Handling ---
            qs = item['conversations'][0]['value']
            
            if has_image:
                if DEFAULT_IMAGE_TOKEN not in qs:
                    if getattr(model.config, 'mm_use_im_start_end', False):
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates[args.version].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_list.append(input_ids)

        if not input_ids_list:
            continue

        # Padding
        max_len = max(x.shape[0] for x in input_ids_list)
        padded_input_ids = torch.stack([
            torch.cat([x, torch.full((max_len - x.shape[0],), tokenizer.pad_token_id, dtype=torch.long)]) 
            for x in input_ids_list
        ])
        
        model_inputs = {
            'input_ids': padded_input_ids.cuda(),
            'attention_mask': padded_input_ids.ne(tokenizer.pad_token_id).cuda()
        }
        
        if images:
            model_inputs['images'] = torch.stack(images).to(dtype=dtype, device='cuda')

        with torch.no_grad():
            try:
                model(**model_inputs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

    # 5. Save
    print("Finalizing clusters...")
    layer_centroids = {}
    for idx in moe_layers_idx:
        handle, hook_obj = hooks[idx]
        hook_obj.finalize()
        handle.remove()
        
        if hook_obj.fitted:
            layer_centroids[idx] = hook_obj.kmeans.cluster_centers_
            print(f"Layer {idx}: {layer_centroids[idx].shape}")
        else:
            print(f"Warning: Layer {idx} empty.")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    joblib.dump(layer_centroids, args.output_file)
    print(f"Done. Saved to {args.output_file}")

if __name__ == "__main__":
    main()