"""
neurips/fig_entropy_trajectory/collect.py
==========================================
Compute mean routing entropy at each intermediate checkpoint of a model.

For each checkpoint-N found under --model_path, loads the weights, runs
--n_samples questions through the MoE layers, records:
  - Per-layer mean Shannon entropy H = -Σ p_j log(p_j) over all 4 experts
  - Model-level mean H (average across all MoE layers)

Output: {output_dir}/{label}.json
  {
    "label": "...",
    "steps": [1, 100, 200, ...],
    "mean_H": [1.38, 1.35, ...],          # model-level mean across all layers
    "layer_H": {                          # per-layer trajectory
      "0": [h_step1, h_step2, ...],
      "4": [...],
      ...
    }
  }

Usage:
    python neurips/fig_entropy_trajectory/collect.py \\
        --model_path /home/prafull/scratch/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe \\
        --label qwen_author \\
        --n_samples 200 \\
        --conv_mode qwen \\
        --gpu 2
"""

import argparse
import json
import os
import sys

def _parse_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg in ("--gpu", "-gpu") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

_gpu_id = _parse_gpu_early()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)

# StableLM deepspeed bypass (needed for stablelm MoE checkpoints)
import deepspeed as _ds
_orig_init_inference = _ds.init_inference

class _DSEngineStub:
    def __init__(self, model):
        self.module = model

def _patched_init_inference(model, **kwargs):
    return _DSEngineStub(model)

_ds.init_inference = _patched_init_inference

from diagnostics.utils import load_question_file, get_question_text


def get_gate_modules(model):
    """Return list of (layer_idx, gate_module) for every MoE gate with a wg linear."""
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
        layer_idx = int(m.group(1) or m.group(2)) if m else len(gates)
        gates.append((layer_idx, module))
    return gates


def register_logit_hook(gate):
    """Hook gate.wg forward; captures raw logits [T, E]."""
    storage = []
    def _hook(module, inp, out):
        storage.append(out.detach().float().cpu())
    handle = gate.wg.register_forward_hook(_hook)
    return handle, storage


def compute_entropy(logits):
    """logits [T, E] → mean Shannon entropy H over tokens."""
    probs = F.softmax(logits, dim=-1)
    H = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)  # [T]
    return float(H.mean().item())


def load_checkpoint(ckpt_path, conv_mode, device):
    from moellava.model.builder import load_pretrained_model
    from moellava.utils import disable_torch_init
    from moellava.mm_utils import get_model_name_from_path

    # Unique port per GPU to avoid conflicts with parallel jobs
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_port = str(12400 + int(gpu_id)) if gpu_id.isdigit() else "12388"
    for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", gpu_port),
                 ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")]:
        os.environ[k] = v  # overwrite each time to ensure correct port

    disable_torch_init()
    model_name = get_model_name_from_path(ckpt_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        ckpt_path, None, model_name
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


def measure_entropy_at_checkpoint(ckpt_path, questions, conv_mode, device):
    """Load checkpoint, run questions, return (layer_indices, per_layer_mean_H)."""
    from moellava.mm_utils import tokenizer_image_token
    from moellava.constants import IMAGE_TOKEN_INDEX
    from moellava.conversation import conv_templates

    tokenizer, model = load_checkpoint(ckpt_path, conv_mode, device)
    gates = get_gate_modules(model)
    if not gates:
        print(f"  WARNING: no gates found in {ckpt_path}")
        return [], []

    handles, stores = [], []
    for _, gate in gates:
        h, s = register_logit_hook(gate)
        handles.append(h)
        stores.append(s)

    layer_entropy_sum = [0.0] * len(gates)
    layer_count       = [0]   * len(gates)

    for i, line in enumerate(questions):
        if i % 50 == 0:
            print(f"    [{i}/{len(questions)}]")
        qtext = get_question_text(line)
        if not qtext:
            continue
        for s in stores:
            s.clear()
        try:
            conv = conv_templates[conv_mode].copy()
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
                    layer_entropy_sum[gi] += compute_entropy(s[0])
                    layer_count[gi]       += 1
        except Exception as e:
            if i < 3:
                import traceback; traceback.print_exc()
            continue

    for h in handles:
        h.remove()

    layer_indices = [li for li, _ in gates]
    per_layer_H   = [
        (layer_entropy_sum[gi] / layer_count[gi]) if layer_count[gi] > 0 else float("nan")
        for gi in range(len(gates))
    ]

    # Free GPU memory before next checkpoint
    del model
    torch.cuda.empty_cache()

    return layer_indices, per_layer_H


def find_checkpoints(model_path):
    """
    Return sorted list of (step, path) for every checkpoint-N subdir found
    under model_path.  Also includes the final checkpoint (model_path itself)
    if it contains model weights (safetensors / pytorch_model.bin).
    """
    import re
    checkpoints = []
    for entry in os.listdir(model_path):
        m = re.match(r"checkpoint-(\d+)$", entry)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, os.path.join(model_path, entry)))
    checkpoints.sort(key=lambda x: x[0])

    # Include final checkpoint if weights exist
    has_final = any(
        f.endswith(".safetensors") or f == "pytorch_model.bin"
        for f in os.listdir(model_path)
    )
    if has_final:
        # Infer final step from trainer_state.json if available
        state_path = os.path.join(model_path, "trainer_state.json")
        final_step = None
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            final_step = int(state.get("global_step", 0)) or None
        if final_step and (not checkpoints or final_step > checkpoints[-1][0]):
            checkpoints.append((final_step, model_path))

    return checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--label",         required=True)
    parser.add_argument("--gpu",           type=int,  default=0)
    parser.add_argument("--n_samples",     type=int,  default=200)
    parser.add_argument("--conv_mode",     default="qwen")
    parser.add_argument("--question_file",
                        default="moellava/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--output_dir",
                        default="neurips/fig_entropy_trajectory/data")
    args = parser.parse_args()

    device = "cuda:0"
    print(f"Physical GPU {args.gpu} → {device}  (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    out_dir = os.path.join(REPO, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.json")

    questions = load_question_file(
        os.path.join(REPO, args.question_file), n_samples=args.n_samples
    )
    print(f"Using {len(questions)} samples")

    checkpoints = find_checkpoints(args.model_path)
    if not checkpoints:
        print(f"ERROR: no checkpoints found under {args.model_path}")
        return
    print(f"Found {len(checkpoints)} checkpoints: steps {[s for s,_ in checkpoints]}")

    results = {
        "label":   args.label,
        "steps":   [],
        "mean_H":  [],
        "layer_H": {},
    }

    for step, ckpt_path in checkpoints:
        print(f"\n── Step {step}: {ckpt_path}")
        layer_indices, per_layer_H = measure_entropy_at_checkpoint(
            ckpt_path, questions, args.conv_mode, device
        )
        if not layer_indices:
            continue

        mean_H = float(np.nanmean(per_layer_H))
        results["steps"].append(step)
        results["mean_H"].append(mean_H)
        for li, h in zip(layer_indices, per_layer_H):
            key = str(li)
            if key not in results["layer_H"]:
                results["layer_H"][key] = []
            results["layer_H"][key].append(h)

        print(f"  mean H = {mean_H:.4f}  |  per-layer: {[round(h,3) for h in per_layer_H]}")

        # Save incrementally so we don't lose data if it crashes mid-run
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Steps: {results['steps']}")
    print(f"Mean H: {[round(h,3) for h in results['mean_H']]}")


if __name__ == "__main__":
    main()
