"""
Experiment 2: Feature norm growth and routing stability across layers.

Runs inference on N=200 samples and for each MoE layer collects:
  - mean L2 norm of hidden states entering the router
  - full-distribution routing entropy (Shannon H over all E experts)
  - top-k routing entropy (H over renormalised top-k probs)

Compares two checkpoints: (a) author/dot-product, (b) ours/cosine-normalised.

Usage:
    python diagnostics/collect_layer_stats.py \
        --model_path_a /scratch/prafull/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe \
        --model_path_b ./checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe \
        --label_a "Random (dot-product)" \
        --label_b "Ours (cosine-normalized)" \
        --n_samples 200 \
        --output_dir diagnostics/data \
        --question_file /path/to/sqa_val.json \
        --image_folder /path/to/images \
        --conv_mode qwen
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diagnostics.utils import (
    load_model_for_inference,
    get_moe_gates,
    register_logit_hook,
    remove_all_hooks,
    routing_entropy_from_logits,
    topk_entropy_from_logits,
    load_question_file,
    get_question_text,
)
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path


# ── Inference helper ───────────────────────────────────────────────────────────

def collect_stats_for_model(model_path, n_samples, questions, image_folder,
                            conv_mode, top_k=2):
    """
    Load model, register hooks, run inference on n_samples questions.
    Returns dict with layer_indices, feature_norms_{mean,std},
    routing_entropy_{mean,std}, topk_entropy_{mean,std}.
    """
    tokenizer, model, processor, _ = load_model_for_inference(model_path)
    image_processor = processor["image"]

    gates = get_moe_gates(model)
    if not gates:
        raise RuntimeError(f"No MoE gates found in model at {model_path}. "
                           "Check that this is an MoE checkpoint.")
    print(f"  Found {len(gates)} MoE layers: {[idx for idx, _ in gates]}")

    # Register hooks on gate.wg (logit hooks) and on post_attention_layernorm (norm hooks)
    logit_handles = []
    logit_stores  = []
    for _, gate in gates:
        h, s = register_logit_hook(gate)
        logit_handles.append(h)
        logit_stores.append(s)

    # Norm hooks: hook post_attention_layernorm for each MoE layer
    norm_stores  = [[] for _ in gates]
    norm_handles = []

    gate_layer_indices = [idx for idx, _ in gates]

    def make_norm_hook(store):
        def _hook(module, inp, out):
            # out: [B, T, D] or [T, D]
            x = out.detach().float()
            norms = x.norm(dim=-1).view(-1)   # flatten to [B*T]
            store.append(norms.cpu())
        return _hook

    # Map layer_idx → norm_store index
    layer_to_norm_idx = {idx: i for i, idx in enumerate(gate_layer_indices)}

    # Qwen uses ln_2 (before MLP/MoE), others use post_attention_layernorm.
    # Match either so the script works for both architectures.
    NORM_NAMES = ("post_attention_layernorm", "ln_2")
    for name, module in model.named_modules():
        if not any(name.endswith(f".{n}") or f".{n}." in name for n in NORM_NAMES):
            continue
        # Skip ln_2 that are actually inside other sub-modules (e.g. cross-attention)
        m = re.search(r'\.layers\.(\d+)\.', name) or re.search(r'\.h\.(\d+)\.', name)
        if not m:
            continue
        lidx = int(m.group(1))
        if lidx not in layer_to_norm_idx:
            continue
        store = norm_stores[layer_to_norm_idx[lidx]]
        h = module.register_forward_hook(make_norm_hook(store))
        norm_handles.append(h)

    # Per-layer accumulators
    n_layers = len(gates)
    all_routing_entropy  = [[] for _ in range(n_layers)]
    all_topk_entropy     = [[] for _ in range(n_layers)]

    questions = questions[:n_samples]
    print(f"  Running inference on {len(questions)} samples...")

    for i, line in enumerate(questions):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(questions)}")

        # ── Build input ────────────────────────────────────────────────────────
        qs = get_question_text(line)
        images = None

        if "image" in line:
            from PIL import Image
            img_path = os.path.join(image_folder, line["image"]) \
                if image_folder else line["image"]
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                images = image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0].unsqueeze(0).half().cuda()
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Clear logit stores before each forward
        for s in logit_stores:
            s.clear()

        with torch.no_grad():
            model(input_ids=input_ids, images=images, return_dict=True)

        # Collect logit-based stats
        for li, s in enumerate(logit_stores):
            if not s:
                continue
            logits = s[0]   # [T, E]
            all_routing_entropy[li].append(
                routing_entropy_from_logits(logits).mean().item()
            )
            all_topk_entropy[li].append(
                topk_entropy_from_logits(logits, k=top_k).mean().item()
            )

    # Remove hooks
    remove_all_hooks(logit_handles + norm_handles)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    feature_norms_mean, feature_norms_std = [], []
    for store in norm_stores:
        if store:
            all_norms = torch.cat(store).numpy()
            feature_norms_mean.append(float(all_norms.mean()))
            feature_norms_std.append(float(all_norms.std()))
        else:
            feature_norms_mean.append(None)
            feature_norms_std.append(None)

    routing_entropy_mean, routing_entropy_std = [], []
    topk_entropy_mean,    topk_entropy_std    = [], []
    for li in range(n_layers):
        if all_routing_entropy[li]:
            routing_entropy_mean.append(float(np.mean(all_routing_entropy[li])))
            routing_entropy_std.append( float(np.std(all_routing_entropy[li])))
        else:
            routing_entropy_mean.append(None)
            routing_entropy_std.append(None)

        if all_topk_entropy[li]:
            topk_entropy_mean.append(float(np.mean(all_topk_entropy[li])))
            topk_entropy_std.append( float(np.std(all_topk_entropy[li])))
        else:
            topk_entropy_mean.append(None)
            topk_entropy_std.append(None)

    result = {
        "model_path":            model_path,
        "layer_indices":         gate_layer_indices,
        "feature_norms_mean":    feature_norms_mean,
        "feature_norms_std":     feature_norms_std,
        "routing_entropy_mean":  routing_entropy_mean,
        "routing_entropy_std":   routing_entropy_std,
        "topk_entropy_mean":     topk_entropy_mean,
        "topk_entropy_std":      topk_entropy_std,
    }

    del model
    torch.cuda.empty_cache()
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_a", required=True,
                        help="Checkpoint A (e.g. author/random init)")
    parser.add_argument("--model_path_b", required=True,
                        help="Checkpoint B (e.g. your cosine-normalised model)")
    parser.add_argument("--label_a", default="Random (dot-product)")
    parser.add_argument("--label_b", default="Ours (cosine-normalized)")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="diagnostics/data")
    parser.add_argument("--question_file", required=True,
                        help="JSON file with questions (ScienceQA val format)")
    parser.add_argument("--image_folder", default="",
                        help="Root folder for images")
    parser.add_argument("--conv_mode", default="qwen")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--out_name_a", default="layer_stats_A.json",
                        help="Output filename for model A stats (relative to output_dir)")
    parser.add_argument("--out_name_b", default="layer_stats_B.json",
                        help="Output filename for model B stats (relative to output_dir)")
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    questions = load_question_file(args.question_file)

    for label, model_path, out_name in [
        (args.label_a, args.model_path_a, args.out_name_a),
        (args.label_b, args.model_path_b, args.out_name_b),
    ]:
        print(f"\n── Collecting stats for: {label} ──────────────────────────")
        stats = collect_stats_for_model(
            model_path  = model_path,
            n_samples   = args.n_samples,
            questions   = questions,
            image_folder= args.image_folder,
            conv_mode   = args.conv_mode,
            top_k       = args.top_k,
        )
        stats["label"] = label

        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {out_path}")

    # ── Print key numbers ──────────────────────────────────────────────────────
    for out_name, tag in [("layer_stats_A.json", "A"), ("layer_stats_B.json", "B")]:
        path = os.path.join(output_dir, out_name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        norms = [v for v in d["feature_norms_mean"] if v is not None]
        ents  = [v for v in d["routing_entropy_mean"] if v is not None]
        if norms:
            print(f"\nModel {tag} ({d['label']}):")
            print(f"  Feature norm at layer {d['layer_indices'][0]}: {norms[0]:.2f}")
            print(f"  Feature norm at layer {d['layer_indices'][-1]}: {norms[-1]:.2f}")
            print(f"  Norm growth ratio (final/first): {norms[-1]/norms[0]:.2f}x")
        if ents:
            print(f"  Routing entropy variance across layers: {float(np.var(ents)):.4f}")
            collapsed = [d['layer_indices'][i]
                         for i, e in enumerate(ents) if e is not None and e < 0.5]
            print(f"  Layers where H < 0.5 (collapsed): {collapsed}")


if __name__ == "__main__":
    main()
