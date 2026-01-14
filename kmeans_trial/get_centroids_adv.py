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
    parser = argparse.ArgumentParser(description="Compute K-Means centroids with Batch Masking and VLM support")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", nargs='+', required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="centroids.pkl")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=40960)
    parser.add_argument("--version", type=str, default="stablelm")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    return parser.parse_args()

class ActivationHook:
    def __init__(self, layer_idx, num_clusters, buffer_size=10240):
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.buffer = []
        self.fitted = False
        self.current_mask = None # Updated per batch in main loop
        
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
            mask = self.current_mask.detach().cpu().bool() # [Batch, Seq_Len_Text]
            
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
        
        self.buffer.append(hidden_states.numpy())
        
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

def main():
    args = parse_args()
    
    # [LOGIC]: Load Dense Architecture
    model_name = "llava-stablelm"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if isinstance(image_processor, dict) or image_processor is None:
        print("Image processor is a dict or None. Loading directly from model path...")
        try:
            # Try to load the official processor from the model folder
            image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
        except Exception as e:
            print(f"Could not load from model_path: {e}. Falling back to default CLIP...")
            # If that fails, load the standard CLIP-L/14 processor
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # Double-check for the necessary attribute used in 'expand2square' (padding)
    if not hasattr(image_processor, 'image_mean'):
        # Map 'mean' to 'image_mean' if necessary
        image_processor.image_mean = getattr(image_processor, 'mean', [0.48145466, 0.4578275, 0.40821073])
        
    model.eval().cuda()
    model.config.image_aspect_ratio = args.image_aspect_ratio
    dtype = model.dtype 

    # Setup Hooks
    num_layers = model.config.num_hidden_layers
    moe_layers_idx = list(range(0, num_layers, 2)) 
    hooks = {}
    for idx in moe_layers_idx:
        target_layer = model.model.layers[idx].post_attention_layernorm
        hook_obj = ActivationHook(idx, args.num_experts, args.buffer_size)
        handle = target_layer.register_forward_hook(hook_obj)
        hooks[idx] = (handle, hook_obj)

    # Load Data
    all_data = []
    for path in args.data_path:
        with open(path, 'r') as f:
            all_data.extend(json.load(f))
    np.random.shuffle(all_data)
    all_data = all_data[:args.num_samples]

    # Processing Loop
    for i in tqdm(range(0, len(all_data), args.batch_size)):
        batch_items = all_data[i : i + args.batch_size]
        images, input_ids_list = [], []
        
        for item in batch_items:
            has_image = 'image' in item
            if has_image:
                try:
                    img = Image.open(os.path.join(args.image_folder, item['image'])).convert('RGB')
                    images.append(process_images([img], image_processor, model.config)[0])
                except: continue

            qs = item['conversations'][0]['value']
            if has_image and DEFAULT_IMAGE_TOKEN not in qs:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates[args.version].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            input_ids_list.append(tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))

        if not input_ids_list: continue

        # Padding & Masking
        max_len = max(x.shape[0] for x in input_ids_list)
        input_ids = torch.stack([torch.cat([x, torch.full((max_len - x.shape[0],), tokenizer.pad_token_id, dtype=torch.long)]) for x in input_ids_list]).cuda()
        attention_mask = input_ids.ne(tokenizer.pad_token_id).cuda()
        
        # Update masks in hooks before forward pass
        for _, h in hooks.values():
            h.current_mask = attention_mask

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if images:
            inputs['images'] = torch.stack(images).to(dtype=dtype, device='cuda')

        with torch.no_grad():
            try: model(**inputs)
            except: torch.cuda.empty_cache()

    # Finalize
    layer_centroids = {}
    for idx, (handle, h) in hooks.items():
        h.finalize()
        handle.remove()
        if h.fitted:
            layer_centroids[idx] = h.kmeans.cluster_centers_
            print(f"Layer {idx} clusters done.")

    joblib.dump(layer_centroids, args.output_file)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()