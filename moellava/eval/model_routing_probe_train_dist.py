"""
model_routing_probe_train_dist.py

Runs the MoE routing probe on the ACTUAL TRAINING DATA distribution.

Source labels come from the data itself (image subfolder for visual samples,
'nlp' for text-only), not from any hand-crafted category set. This removes
the cherry-picking objection from paper reviewers: categories are the dataset
structure, not analyst choices.

Output .pt format is compatible with vis_dual_routing_v2.py:
    {sample_id: {gating_logit, input_ids, output_ids, category, layer_indices}}

Usage:
    python moellava/eval/model_routing_probe_train_dist.py \\
        --model-path checkpoints_stablelm_power_adaptive/llava-stablelm-1.6b-finetune-moe \\
        --data-paths ../MoE-LLaVA-main/train_json/llava_image_tune_.json \\
                     ../MoE-LLaVA-main/train_json/nlp_tune.json \\
        --image-folder ../MoE-LLaVA-main/IMAGE_FOLDER \\
        --samples-per-source 80 \\
        --conv-mode stablelm \\
        --return-gating-logit diagnostics/train_dist_stablelm_power_adaptive.pt \\
        --seed 42
"""

import argparse
import json
import math
import os
import random
import re

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from moellava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
from moellava.conversation import conv_templates
from moellava.mm_utils import get_model_name_from_path, tokenizer_image_token
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init


# ---------------------------------------------------------------------------
# Hooking logic (identical to model_routing_probe_v2.py)
# ---------------------------------------------------------------------------

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out.detach().cpu()


def get_gating_logit_by_hook(model):
    fea_hooks, layer_indices = [], []
    print("Hooking MoE gates...")
    for n, m in model.named_modules():
        if 'wg' in n and isinstance(m, nn.Linear):
            match = re.search(r'(?:layers|h)\.(\d+)', n)
            if match:
                layer_idx = int(match.group(1))
                print(f"  Hooked Layer {layer_idx} ({n})")
            else:
                layer_idx = len(fea_hooks)
                print(f"  Could not parse layer index from {n}, using position {layer_idx}")
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
            layer_indices.append(layer_idx)
    return fea_hooks, layer_indices


# ---------------------------------------------------------------------------
# Source label derivation
# ---------------------------------------------------------------------------

# Map image subfolder → clean display name
SUBFOLDER_TO_SOURCE = {
    'coco':    'coco',
    'vg':      'vg',
    'ocr_vqa': 'ocr_vqa',
    'gqa':     'gqa',
    'textvqa': 'textvqa',
}


def derive_source(item, json_basename):
    """Return a source label for one training sample.

    For image samples: second path component of the image field (e.g. 'coco').
    For text-only samples: 'nlp'.
    """
    if 'image' in item:
        img = item['image']
        parts = img.replace('\\', '/').split('/')
        for part in parts:
            if part in SUBFOLDER_TO_SOURCE:
                return SUBFOLDER_TO_SOURCE[part]
        # fallback: use json filename stem
        return json_basename
    return 'nlp'


def load_and_group_by_source(data_paths):
    """Load all training JSONs and group samples by derived source label."""
    by_source = {}
    for path in data_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        print(f"Loading {path}...")
        data = json.load(open(path))
        for item in data:
            src = derive_source(item, basename)
            by_source.setdefault(src, []).append(item)
    for src, items in by_source.items():
        print(f"  {src}: {len(items):,} samples")
    return by_source


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(args):
    random.seed(args.seed)
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    fea_hooks, layer_indices = get_gating_logit_by_hook(model)
    image_processor = processor['image']

    # --- Load & sample training data ---
    by_source = load_and_group_by_source(args.data_paths)

    # Apply source filter if requested
    if args.sources:
        requested = set(args.sources)
        by_source = {k: v for k, v in by_source.items() if k in requested}
        print(f"Filtered to sources: {list(by_source.keys())}")

    sampled_items = []  # list of (item, source_label)
    for src, items in sorted(by_source.items()):
        random.shuffle(items)
        n = min(args.samples_per_source, len(items))
        sampled_items.extend((item, src) for item in items[:n])
        print(f"  Sampling {n} from '{src}'")

    random.shuffle(sampled_items)
    print(f"\nRunning probe on {len(sampled_items)} total samples...")

    all_gating_logits = {}

    for idx, (line, source_label) in enumerate(tqdm(sampled_items)):
        # Build question text
        convs = line.get('conversations', [])
        if not convs:
            continue
        qs = convs[0]['value'].replace('<image>', '').strip()

        images = None
        has_image = 'image' in line

        if has_image:
            image_file = line['image']
            # Try direct path first, then under image_folder
            if os.path.exists(image_file):
                img_path = image_file
            else:
                img_path = os.path.join(args.image_folder, image_file)

            if not os.path.exists(img_path):
                continue  # skip missing images silently

            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).half().cuda()
            except Exception:
                continue

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                images=images,
                output_hidden_states=False,
                return_dict=True,
            )

        if fea_hooks and fea_hooks[0].fea is not None:
            layer_logits = [h.fea.clone() for h in fea_hooks]
            sample_id = f"train_{idx:06d}"
            all_gating_logits[sample_id] = dict(
                gating_logit=layer_logits,
                images=images.detach().cpu() if images is not None else None,
                input_ids=input_ids.detach().cpu(),
                output_ids=input_ids.detach().cpu(),
                category=source_label,       # ← dataset source as category
                prompt_type=source_label,    # ← same, so vis_v2 works with either field
                layer_indices=layer_indices,
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.return_gating_logit)), exist_ok=True)
    torch.save(all_gating_logits, args.return_gating_logit)
    print(f"\nSaved {len(all_gating_logits)} samples to {args.return_gating_logit}")

    # Print source breakdown
    from collections import Counter
    source_counts = Counter(v['category'] for v in all_gating_logits.values())
    print("Source breakdown in saved file:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-paths", type=str, nargs='+', required=True,
                        help="Training JSON files (e.g. llava_image_tune_.json nlp_tune.json)")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Root folder for images (prepended to image paths)")
    parser.add_argument("--samples-per-source", type=int, default=80,
                        help="How many samples to draw from each dataset source")
    parser.add_argument("--sources", type=str, nargs='*', default=None,
                        help="If set, only include these source labels (e.g. coco gqa nlp)")
    parser.add_argument("--conv-mode", type=str, default="stablelm")
    parser.add_argument("--return-gating-logit", type=str, required=True,
                        help="Output path for .pt routing data file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    run_probe(args)
