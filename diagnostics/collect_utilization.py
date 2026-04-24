"""
Panel D: Expert utilization collection.

For each MoE layer, records what fraction of tokens are routed to each expert
(counting both top-1 and top-2 selections). This measures load balance across
experts — distinct from split_ratios which measures balance *within* the top-2
pair.

Hooks gate.wg output (raw logits [T, E]), applies softmax, takes top-k indices,
accumulates counts per expert per layer.

Output: diagnostics/data/utilization_{label}.npz
    layer_indices:     [N_moe_layers]         int array
    expert_fractions:  [N_moe_layers, N_experts]  float32, each row sums to 1.0
                       (fraction of top-k selections going to expert e at layer l)

Usage:
    python diagnostics/collect_utilization.py \
        --model_path /path/to/checkpoint \
        --label author \
        --n_samples 500 \
        --output_dir diagnostics/data \
        --question_file moellava/eval/scienceqa/problems.json \
        --image_folder moellava/eval/scienceqa/images/val \
        --conv_mode qwen
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diagnostics.utils import (
    load_model_for_inference,
    get_moe_gates,
    register_logit_hook,
    remove_all_hooks,
)
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates
from moellava.mm_utils import tokenizer_image_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--label",         required=True,
                        help="Short name, e.g. 'author', 'new_entropy'")
    parser.add_argument("--n_samples",     type=int, default=500)
    parser.add_argument("--output_dir",    default="diagnostics/data")
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder",  default="")
    parser.add_argument("--conv_mode",     default="qwen")
    parser.add_argument("--top_k",         type=int, default=2)
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = questions[:args.n_samples]

    print(f"Loading model: {args.model_path}")
    tokenizer, model, processor, _ = load_model_for_inference(args.model_path)
    image_processor = processor["image"]

    gates = get_moe_gates(model)
    if not gates:
        raise RuntimeError("No MoE gates found.")
    n_layers = len(gates)
    print(f"Found {n_layers} MoE layers: {[i for i, _ in gates]}")

    # Register logit hooks
    handles, stores = [], []
    for _, gate in gates:
        h, s = register_logit_hook(gate)
        handles.append(h)
        stores.append(s)

    # Per-layer expert token counts: [N_moe_layers, N_experts]
    # Infer N_experts from first forward pass; initialise after.
    expert_counts = None

    print(f"Running inference on {len(questions)} samples...")
    for i, line in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(questions)}")

        qs = line["conversations"][0]["value"].replace("<image>", "").strip()
        images = None

        if "image" in line:
            from PIL import Image
            img_path = (os.path.join(args.image_folder, line["image"])
                        if args.image_folder else line["image"])
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                images = image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0].unsqueeze(0).half().cuda()
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            else:
                warnings.warn(f"Image not found: {img_path}")

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        for s in stores:
            s.clear()

        with torch.no_grad():
            model(input_ids=input_ids, images=images, return_dict=True)

        for li, s in enumerate(stores):
            if not s:
                continue
            logits = s[0].float()           # [T, E]
            n_experts = logits.shape[-1]

            if expert_counts is None:
                expert_counts = np.zeros((n_layers, n_experts), dtype=np.int64)

            probs = torch.softmax(logits, dim=-1)               # [T, E]
            topk_indices = torch.topk(probs, args.top_k, dim=-1).indices  # [T, k]
            # Count each expert occurrence across all tokens and top-k slots
            flat_indices = topk_indices.cpu().numpy().ravel()   # [T*k]
            for expert_idx in flat_indices:
                expert_counts[li, expert_idx] += 1

    remove_all_hooks(handles)
    del model
    torch.cuda.empty_cache()

    if expert_counts is None:
        print("No data collected — check question file and model path.")
        return

    # Normalise: fraction of (token × top-k slot) selections per expert per layer
    row_sums = expert_counts.sum(axis=1, keepdims=True).astype(np.float64)
    expert_fractions = (expert_counts / np.maximum(row_sums, 1)).astype(np.float32)

    layer_indices = np.array([idx for idx, _ in gates], dtype=np.int32)
    out_path = os.path.join(output_dir, f"utilization_{args.label}.npz")
    np.savez_compressed(
        out_path,
        layer_indices=layer_indices,
        expert_fractions=expert_fractions,   # [N_layers, N_experts]
        expert_counts=expert_counts,
    )
    print(f"Saved: {out_path}")

    # Key numbers
    n_experts = expert_fractions.shape[1]
    uniform   = 1.0 / n_experts
    print(f"\n── Utilization stats for '{args.label}' ──────────────────────────")
    print(f"  Uniform reference: {100*uniform:.1f}% per expert")
    overloaded = (expert_fractions > 2 * uniform).sum(axis=1)
    print(f"  Layers with ≥1 expert >2× uniform: {(overloaded > 0).sum()} / {n_layers}")
    print(f"  Max expert fraction across all layers: {100*expert_fractions.max():.1f}%")
    print(f"  Min expert fraction across all layers: {100*expert_fractions.min():.1f}%")
    print(f"  Mean std across layers: {expert_fractions.std(axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
