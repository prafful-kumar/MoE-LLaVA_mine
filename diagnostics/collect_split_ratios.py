"""
Experiment 3: Top-k entropy loss — routing split histogram.

For each token × MoE layer, records the split ratio of the top-2 routing
decision:
    split_ratio = max(p_top1, p_top2) / (p_top1 + p_top2)
Values near 0.5 = balanced; values near 1.0 = one expert takes all weight.

Hooks gate.wg output (raw logits [T, E]), applies softmax, takes top-2.
Does NOT modify the model.

Usage:
    python diagnostics/collect_split_ratios.py \
        --model_path /scratch/prafull/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe \
        --label author \
        --n_samples 500 \
        --output_dir diagnostics/data \
        --question_file /path/to/sqa_val.json \
        --image_folder /path/to/images \
        --conv_mode qwen
"""

import argparse
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
    load_question_file,
    get_question_text,
)
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates
from moellava.mm_utils import tokenizer_image_token


def compute_split_ratios(logits_list, k=2):
    """
    logits_list: list of [T, E] tensors, one per sample for this layer.
    Returns np.array of split ratios, shape [sum(T_i)].
    """
    all_ratios = []
    for logits in logits_list:
        probs = torch.softmax(logits.float(), dim=-1)          # [T, E]
        topk_probs, _ = torch.topk(probs, k, dim=-1)           # [T, k]
        top1 = topk_probs[:, 0]
        top2 = topk_probs[:, 1]
        total = top1 + top2
        ratio = torch.max(top1, top2) / (total + 1e-8)         # [T]
        all_ratios.append(ratio.cpu().numpy())
    if all_ratios:
        return np.concatenate(all_ratios).astype(np.float32)
    return np.array([], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label", required=True,
                        help="Short name, e.g. 'author', 'entropy_old', 'new_entropy'")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", default="diagnostics/data")
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--conv_mode", default="qwen")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--out_tag", default="",
                        help="Optional tag appended to output filename, e.g. 'sqa1000' "
                             "→ split_ratios_{label}_sqa1000.npz")
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    questions = load_question_file(args.question_file, n_samples=args.n_samples)

    print(f"Loading model: {args.model_path}")
    tokenizer, model, processor, _ = load_model_for_inference(args.model_path)
    image_processor = processor["image"]

    gates = get_moe_gates(model)
    if not gates:
        raise RuntimeError("No MoE gates found.")
    print(f"Found {len(gates)} MoE layers: {[i for i, _ in gates]}")

    # Register logit hooks on all gates
    handles = []
    stores  = []
    for _, gate in gates:
        h, s = register_logit_hook(gate)
        handles.append(h)
        stores.append(s)

    # Per-layer accumulator: list of [T, E] logit tensors
    layer_logits = [[] for _ in range(len(gates))]

    print(f"Running inference on {len(questions)} samples...")
    for i, line in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(questions)}")

        qs = get_question_text(line)
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
            if s:
                layer_logits[li].append(s[0].cpu())

    remove_all_hooks(handles)
    del model
    torch.cuda.empty_cache()

    # ── Compute split ratios per layer ─────────────────────────────────────────
    # Output shape: [N_tokens_total, N_moe_layers]
    # Pad to equal length per sample is complex — instead save per-layer arrays
    # and stack into a 2D array where rows = tokens (variable), cols = layers.
    # Since different samples have different sequence lengths, we flatten across
    # tokens and layers into one flat array per label (as spec says).
    # But we also want the 2D shape for thorough analysis, so save both.

    all_layer_ratios = []
    for li, logits_list in enumerate(layer_logits):
        ratios = compute_split_ratios(logits_list, k=args.top_k)
        all_layer_ratios.append(ratios)

    # Flatten across all layers for the histogram
    flat_ratios = np.concatenate(all_layer_ratios) if all_layer_ratios else np.array([])

    tag_str = f"_{args.out_tag}" if args.out_tag else ""
    out_path = os.path.join(output_dir, f"split_ratios_{args.label}{tag_str}.npz")
    np.savez_compressed(
        out_path,
        split_ratios_flat=flat_ratios,
        layer_indices=np.array([i for i, _ in gates]),
        **{f"layer_{i}": r for i, r in enumerate(all_layer_ratios)},
    )
    print(f"Saved: {out_path}")

    # ── Key numbers ────────────────────────────────────────────────────────────
    if len(flat_ratios) > 0:
        print(f"\n── Split ratio stats for '{args.label}' ──────────────────────")
        print(f"  % near-collapse (>0.9):  {100*(flat_ratios > 0.9).mean():.1f}%")
        print(f"  % near-balanced (<0.6):  {100*(flat_ratios < 0.6).mean():.1f}%")
        print(f"  Median split ratio:      {float(np.median(flat_ratios)):.3f}")
        print(f"  Mean split ratio:        {float(flat_ratios.mean()):.3f}")


if __name__ == "__main__":
    main()
