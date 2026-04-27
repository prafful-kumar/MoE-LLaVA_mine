"""
neurips/fig2_routing_confidence/collect.py
==========================================
Collect renormalized top-2 max probability for any checkpoint.

The metric: split_ratio = max(p_top1, p_top2) / (p_top1 + p_top2)
  - 0.5  → both selected experts get equal weight (ambiguous token, sharing)
  - 1.0  → one expert gets everything (confident assignment, within-k collapse)

The key prediction: a well-trained adaptive router should produce a BIMODAL distribution:
  - Peak near 0.5:  ambiguous tokens the router chose to share between two experts
  - Peak near 1.0:  confident tokens the router committed to a single expert
  Both peaks coexisting proves the router learned to treat tokens differently.

Output: {output_dir}/{label}.npz
  Keys:
    split_ratios_flat   [n_total_tokens]        all split ratios concatenated
    layer_indices       [n_layers]               MoE layer indices in model
    layer_{i}           [n_tokens_in_layer_i]    per-layer split ratios

Existing data (no collection needed):
    diagnostics/data/split_ratios_author_sqa1000.npz
    diagnostics/data/split_ratios_entropy_topk_var_sqa1000.npz

Usage (for new adaptive checkpoint):
    python neurips/fig2_routing_confidence/collect.py \\
        --model_path /home/prafull/scratch/hpc/checkpoints_qwen_adaptive/llavaqwen-1.8b-finetune-moe \\
        --label adaptive_entropy \\
        --n_samples 1000 \\
        --gpu 6
"""

import argparse
import os
import sys

# ── CRITICAL: set CUDA_VISIBLE_DEVICES before any torch import ────────────────
# Parse --gpu early so the env var is set before torch initializes.
# This forces the model onto a single logical cuda:0 device rather than
# letting device_map="auto" scatter it across all free GPUs.
def _parse_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg in ("--gpu", "-gpu") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

_gpu_id = _parse_gpu_early()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
# After this, torch sees only one GPU — addressed as cuda:0.

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)

from diagnostics.utils import (
    load_model_for_inference,
    load_question_file,
    get_question_text,
)


def get_moe_gates(model):
    import re
    gates = []
    for name, module in model.named_modules():
        if not name.endswith(".gate"):
            continue
        if "deepspeed_moe" not in name:
            continue
        if not hasattr(module, "wg"):
            continue
        m = re.search(r'\.layers\.(\d+)\.|\.h\.(\d+)\.', name)
        layer_idx = int((m.group(1) or m.group(2))) if m else len(gates)
        gates.append((len(gates), layer_idx, module))
    return gates


def register_logit_hook(gate):
    """Hook gate.wg; captures raw logits [T, E]."""
    storage = []
    def _hook(module, inp, out):
        storage.append(out.detach().float().cpu())
    handle = gate.wg.register_forward_hook(_hook)
    return handle, storage


def compute_split_ratio(logits):
    """
    logits: [T, E] raw gate logits.
    Returns split_ratio [T]: max(p_top1, p_top2) / (p_top1 + p_top2).
    """
    probs = F.softmax(logits, dim=-1)                   # [T, E]
    top2_vals, _top2_idx = probs.topk(2, dim=-1)        # [T, 2]
    p1 = top2_vals[:, 0]
    p2 = top2_vals[:, 1]
    denom = p1 + p2
    # Avoid division by zero (shouldn't happen with softmax, but be safe)
    split = torch.where(denom > 1e-8, p1 / denom, torch.full_like(p1, 0.5))
    return split.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--label",         required=True,
                        help="E.g. 'adaptive_entropy' or 'double_adaptive'")
    parser.add_argument("--gpu",           type=int,   default=6)
    parser.add_argument("--n_samples",     type=int,   default=1000)
    parser.add_argument("--question_file",
                        default="moellava/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--image_folder",
                        default="moellava/eval/scienceqa/images/test")
    parser.add_argument("--conv_mode",     default="qwen")
    parser.add_argument("--output_dir",
                        default="neurips/fig2_routing_confidence/data")
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES was already set to args.gpu at import time,
    # so physical GPU args.gpu is now logical cuda:0.
    device = "cuda:0"
    print(f"Physical GPU {args.gpu} → logical {device}  (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    out_dir = os.path.join(REPO, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.npz")

    print(f"Loading: {args.model_path}")
    tokenizer, model, _, _ = load_model_for_inference(args.model_path, device=device)
    model.eval()

    gates = get_moe_gates(model)
    print(f"Found {len(gates)} MoE gates")

    handles, stores = [], []
    for _, _, gate in gates:
        h, s = register_logit_hook(gate)
        handles.append(h)
        stores.append(s)

    questions = load_question_file(
        os.path.join(REPO, args.question_file), n_samples=args.n_samples
    )
    print(f"Processing {len(questions)} samples...")

    from moellava.mm_utils import tokenizer_image_token
    from moellava.constants import IMAGE_TOKEN_INDEX
    from moellava.conversation import conv_templates

    gate_ratios = [[] for _ in gates]   # per gate: list of [T] arrays

    for i, line in enumerate(questions):
        if i % 100 == 0:
            print(f"  [{i}/{len(questions)}]")
        qtext = get_question_text(line)
        if not qtext:
            continue
        for s in stores:
            s.clear()
        try:
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qtext)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                model(input_ids=input_ids)
            for gi, s in enumerate(stores):
                if s:
                    ratios = compute_split_ratio(s[0])   # [T]
                    gate_ratios[gi].append(ratios)
        except Exception as e:
            if i < 3:   # print first few errors so we catch systematic failures
                import traceback
                print(f"  [sample {i}] ERROR: {e}")
                traceback.print_exc()
            continue

    for h in handles:
        h.remove()

    # Build output
    layer_indices = [li for _, li, _ in gates]
    save_dict = {"layer_indices": np.array(layer_indices)}
    all_flat = []

    for gi, (_, layer_idx, _) in enumerate(gates):
        if gate_ratios[gi]:
            arr = np.concatenate(gate_ratios[gi])        # [total_tokens_for_layer]
            save_dict[f"layer_{gi}"] = arr
            all_flat.append(arr)
        else:
            save_dict[f"layer_{gi}"] = np.array([])

    save_dict["split_ratios_flat"] = np.concatenate(all_flat) if all_flat else np.array([])

    np.savez(out_path, **save_dict)
    print(f"\nSaved: {out_path}")

    sr = save_dict["split_ratios_flat"]
    print(f"\nSplit ratio summary for '{args.label}':")
    print(f"  N tokens total : {len(sr):,}")
    print(f"  Mean           : {sr.mean():.3f}")
    print(f"  Median         : {np.median(sr):.3f}")
    print(f"  % near 0.5  (<0.55) : {100*(sr<0.55).mean():.1f}%  ← sharing zone")
    print(f"  % collapsed  (>0.9) : {100*(sr>0.9).mean():.1f}%  ← committed zone")
    print(f"  % exactly 1.0       : {100*(sr==1.0).mean():.1f}%  ← hard collapse")


if __name__ == "__main__":
    main()
