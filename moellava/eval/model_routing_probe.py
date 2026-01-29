import argparse
import torch
import torch.nn as nn
import os
import json
import math
import re  # <--- Added for parsing layer indices
from tqdm import tqdm
from PIL import Image

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moellava.conversation import conv_templates
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path

# ==========================================
# 🪝 HOOKING LOGIC (Updated to Capture Layer IDs)
# ==========================================
class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out.detach().cpu()

def get_gating_logit_by_hook(model):
    fea_hooks = []
    layer_indices = [] # <--- Store the real layer IDs here
    
    print(f"🔍 Hooking MoE gates...")
    for n, m in model.named_modules():
        if 'wg' in n and isinstance(m, nn.Linear):
            # Parse the layer index from the name (e.g. "model.layers.12.mlp...")
            # Regex finds the number immediately following 'layers.'
            match = re.search(r'layers\.(\d+)', n)
            if match:
                layer_idx = int(match.group(1))
                print(f"   ✅ Hooked Layer {layer_idx} ({n})")
                
                cur_hook = HookTool()
                m.register_forward_hook(cur_hook.hook_fun)
                
                fea_hooks.append(cur_hook)
                layer_indices.append(layer_idx)
            else:
                print(f"   ⚠️ Warning: Could not parse layer index from {n}")

    return fea_hooks, layer_indices # <--- Return both

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model_probe(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # Register Hooks & Get Indices
    fea_hooks, layer_indices = get_gating_logit_by_hook(model)
    all_gating_logits = {}

    image_processor = processor['image']
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    print(f"🚀 Running Routing Probe on {len(questions)} samples...")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        category = line.get("category", "unknown")
        
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        images = None
        if 'image' in line:
            image_file = line["image"]
            if os.path.exists(image_file):
                img_path = image_file
            else:
                img_path = os.path.join(args.image_folder, image_file)
            
            image = Image.open(img_path).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, images=images, output_hidden_states=True, return_dict=True)

        if len(fea_hooks) > 0 and fea_hooks[0].fea is not None:
            layer_logits = [h.fea.clone() for h in fea_hooks]
            
            all_gating_logits[idx] = dict(
                gating_logit=layer_logits,
                images=images.detach().cpu() if images is not None else None,
                input_ids=input_ids.detach().cpu(),
                output_ids=input_ids.detach().cpu(),
                category=category,
                layer_indices=layer_indices # <--- Saving the mapping map!
            )

    if args.return_gating_logit:
        save_path = f'{args.return_gating_logit}.pt'
        torch.save(all_gating_logits, save_path)
        print(f"✅ Saved analysis data (with layer mapping) to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="dummy.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--return_gating_logit", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    eval_model_probe(args)