"""
Collect MoE routing assignment data across training checkpoints.

Runs inference on already-trained checkpoints (NO retraining). For each checkpoint,
records which expert each probe token is assigned to in every MoE layer. The resulting
JSON is used by plot_routing_fluctuation.py to reproduce a StableMoE Figure 7 analysis.

Usage:
    # Fisher-init model
    python tools/collect_routing_assignments.py \\
        --checkpoint_dir /home/prafull/scratch/hpc/checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe \\
        --output_file routing_data/stablelm_fisher.json \\
        --label "Fisher-Init (Ours)" \\
        --gpu 0

    # Random-init baseline
    python tools/collect_routing_assignments.py \\
        --checkpoint_dir /home/prafull/scratch/hpc/checkpoints_stablelm_author/llava-stablelm-1.6b-finetune-moe \\
        --output_file routing_data/stablelm_random.json \\
        --label "Random-Init (Baseline)" \\
        --gpu 0
"""

import argparse
import gc
import json
import os
import re
import sys
import warnings

import torch

# Allow running from repo root without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fixed probe sentences (identical across all checkpoints) ────────────────────

PROBE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A mixture of experts model routes tokens to specialized subnetworks.",
    "What is the capital of France?",
    "The image shows a cat sitting on a red chair.",
    "Please describe what you see in this photograph.",
    "Mathematics is the language of the universe.",
    "How does photosynthesis work in plants?",
    "The weather today is sunny with a high of 25 degrees.",
    "Can you recommend a good recipe for chocolate cake?",
    "The router assigns each token to the most relevant expert.",
    "Large language models have transformed natural language processing.",
    "What time does the next train to London depart?",
    "Expert specialization emerges through gradient-based learning.",
    "The Eiffel Tower is located in Paris, France.",
    "Sparse activation reduces computational cost at inference time.",
    "How many planets are in our solar system?",
    "Vision-language models process both images and text jointly.",
    "The neural network learned to classify images with high accuracy.",
    "Fisher Linear Discriminant Analysis finds maximally separating directions.",
    "Routing fluctuation harms training efficiency in MoE models.",
]


# ── Checkpoint discovery ─────────────────────────────────────────────────────────

def discover_checkpoints(checkpoint_dir, max_checkpoints):
    """
    Scan checkpoint_dir for subdirs matching checkpoint-N pattern, plus the
    root directory itself (treated as the final step). Returns a sorted list of
    (step, path) tuples. If more than max_checkpoints exist (excluding root),
    samples evenly spaced ones (always including first and last intermediate).
    """
    pattern = re.compile(r'^checkpoint-(\d+)$')
    intermediate = []
    for name in os.listdir(checkpoint_dir):
        m = pattern.match(name)
        if m and os.path.isdir(os.path.join(checkpoint_dir, name)):
            step = int(m.group(1))
            intermediate.append((step, os.path.join(checkpoint_dir, name)))

    intermediate.sort(key=lambda x: x[0])

    if len(intermediate) > max_checkpoints:
        # Always keep first and last; sample the middle evenly
        if max_checkpoints <= 2:
            intermediate = [intermediate[0], intermediate[-1]]
        else:
            n_middle = max_checkpoints - 2
            middle = intermediate[1:-1]
            indices = [round(i * (len(middle) - 1) / (n_middle - 1))
                       for i in range(n_middle)] if n_middle > 1 else [len(middle) // 2]
            indices = sorted(set(indices))
            intermediate = [intermediate[0]] + [middle[i] for i in indices] + [intermediate[-1]]

    # Infer final step: use root dir, step = max intermediate step found (or 0)
    final_step = intermediate[-1][0] + 1 if intermediate else 0
    all_checkpoints = intermediate + [(final_step, checkpoint_dir)]

    return all_checkpoints


# ── Gate detection ───────────────────────────────────────────────────────────────

def get_moe_gates(model):
    """
    Find all MoE gate modules. Returns list of (sequential_idx, gate_module).
    Gates are identified by having a .wg attribute inside a deepspeed_moe block.
    Assigns sequential indices in model.named_modules() order (consistent across loads).
    """
    gates = []
    for name, module in model.named_modules():
        if not name.endswith(".gate"):
            continue
        if "deepspeed_moe" not in name:
            continue
        if not hasattr(module, "wg"):
            continue
        gates.append((len(gates), module))
    return gates


# ── Hook-based assignment capture ───────────────────────────────────────────────

def register_dispatch_hook(gate):
    """
    Register a forward hook on the gate module itself (not gate.wg).
    Gate forward() returns (loss, combine_weights, dispatch_mask, exp_counts).
    dispatch_mask shape: [tokens, experts] or [tokens, experts, capacity].

    Returns (handle, storage_list). Storage will contain one entry per forward call:
      a list of ints, one per token — the expert index assigned to each token.
    """
    storage = []

    def _hook(module, inp, out):
        try:
            # out is a tuple; dispatch_mask is index 2
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                dispatch_mask = out[2]
            else:
                # Fallback: out itself may be the dispatch mask
                dispatch_mask = out

            if not isinstance(dispatch_mask, torch.Tensor):
                return

            dm = dispatch_mask.detach().float()

            if dm.dim() == 3:
                # [tokens, experts, capacity] → argmax over (experts × capacity)
                # A token assigned to expert e has dm[t, e, :].any() == True
                assignments = dm.any(dim=-1).float().argmax(dim=-1)
            elif dm.dim() == 2:
                # [tokens, experts]
                assignments = dm.float().argmax(dim=-1)
            else:
                return

            storage.append(assignments.cpu().tolist())
        except Exception as e:
            warnings.warn(f"Hook error: {e}")

    handle = gate.register_forward_hook(_hook)
    return handle, storage


# ── Probe batch construction ──────────────────────────────────────────────────────

def build_probe_batch(tokenizer, num_probe_tokens, device):
    """
    Tokenize the fixed probe sentences and build a batch of exactly num_probe_tokens
    sentences. Repeats/truncates PROBE_SENTENCES list to reach the target count.
    Returns dict with input_ids and attention_mask on the given device.
    """
    sentences = []
    while len(sentences) < num_probe_tokens:
        sentences.extend(PROBE_SENTENCES)
    sentences = sentences[:num_probe_tokens]

    encoded = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return {k: v.to(device) for k, v in encoded.items()}


# ── Per-checkpoint assignment collection ─────────────────────────────────────────

def collect_assignments_for_checkpoint(checkpoint_path, probe_batch, device):
    """
    Load model from checkpoint_path, run probe_batch through it, collect gate
    assignments for every MoE layer. Returns dict {layer_idx: [expert_id, ...]}.
    Cleans up model from GPU before returning.
    """
    from moellava.model.builder import load_pretrained_model
    from moellava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(checkpoint_path)

    try:
        tokenizer, model, _processor, _ctx_len = load_pretrained_model(
            checkpoint_path,
            model_base=None,
            model_name=model_name,
            device_map=device,
        )
    except Exception as e:
        warnings.warn(f"Failed to load {checkpoint_path}: {e}")
        return None

    model.eval()
    gates = get_moe_gates(model)

    if not gates:
        warnings.warn(f"No MoE gates found in {checkpoint_path}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return None

    # Register hooks on all gates
    handles = []
    stores = []
    for _idx, gate in gates:
        h, s = register_dispatch_hook(gate)
        handles.append(h)
        stores.append(s)

    # Forward pass (text-only; MoE layers are in language backbone)
    try:
        with torch.no_grad():
            # Pass only input_ids and attention_mask — no images needed
            model(
                input_ids=probe_batch["input_ids"],
                attention_mask=probe_batch["attention_mask"],
            )
    except Exception as e:
        warnings.warn(f"Forward pass failed for {checkpoint_path}: {e}")
        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return None

    # Collect results
    layer_assignments = {}
    for (layer_idx, _gate), store in zip(gates, stores):
        if store:
            # If multiple forward calls accumulated (e.g. cached), use first
            layer_assignments[str(layer_idx)] = store[0]
        else:
            warnings.warn(f"No assignments captured for layer {layer_idx}")

    # Cleanup
    for h in handles:
        h.remove()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return layer_assignments


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect MoE routing assignments across training checkpoints."
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Root checkpoint directory (e.g. checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe)"
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Output JSON path (e.g. routing_data/stablelm_fisher.json)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU id to use (default: 0)"
    )
    parser.add_argument(
        "--max_checkpoints", type=int, default=20,
        help="Max number of intermediate checkpoints to process (default: 20)"
    )
    parser.add_argument(
        "--num_probe_tokens", type=int, default=128,
        help="Number of probe sentences to use (default: 128)"
    )
    parser.add_argument(
        "--label", default="Model",
        help='Short label for this run, e.g. "Fisher-Init" or "Random-Init"'
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # Discover checkpoints
    print(f"Scanning {args.checkpoint_dir} for checkpoints...")
    checkpoints = discover_checkpoints(args.checkpoint_dir, args.max_checkpoints)
    print(f"Found {len(checkpoints)} checkpoints (including final):")
    for step, path in checkpoints:
        print(f"  step={step}  {path}")

    # Build probe batch using tokenizer from FINAL checkpoint (root dir)
    # so the probe is consistent across all loads
    print("\nBuilding probe batch from final checkpoint tokenizer...")
    from moellava.model.builder import load_pretrained_model
    from moellava.mm_utils import get_model_name_from_path

    final_path = checkpoints[-1][1]
    model_name = get_model_name_from_path(final_path)
    tokenizer, _model, _proc, _ctx = load_pretrained_model(
        final_path, model_base=None, model_name=model_name, device_map="cpu"
    )
    probe_batch = build_probe_batch(tokenizer, args.num_probe_tokens, device)
    # Move probe to CPU for storage; we'll move to device during each checkpoint run
    probe_batch_cpu = {k: v.cpu() for k, v in probe_batch.items()}
    del _model, _proc, _ctx
    gc.collect()
    print(f"Probe batch: {args.num_probe_tokens} sentences, "
          f"seq_len={probe_batch['input_ids'].shape[1]}")

    # Collect across all checkpoints
    records = []
    total_steps = checkpoints[-1][0]

    for step, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Processing step {step}: {ckpt_path}")

        # Re-move probe to device for each checkpoint
        probe_on_device = {k: v.to(device) for k, v in probe_batch_cpu.items()}

        layer_assignments = collect_assignments_for_checkpoint(ckpt_path, probe_on_device, device)
        del probe_on_device

        if layer_assignments is None:
            print(f"  WARNING: Skipped step {step} due to error.")
            continue

        n_layers = len(layer_assignments)
        n_tokens = len(next(iter(layer_assignments.values()))) if layer_assignments else 0
        print(f"  Step {step}: recorded assignments for {n_layers} layers, {n_tokens} tokens each")
        records.append({"step": step, "layer_assignments": layer_assignments})

    if not records:
        print("ERROR: No records collected. Exiting.")
        sys.exit(1)

    # Infer metadata from collected records
    sample_record = records[-1]["layer_assignments"]
    num_layers = len(sample_record)
    num_tokens_per_layer = len(next(iter(sample_record.values()))) if sample_record else 0

    # num_experts: infer from max assignment value + 1
    all_expert_ids = [
        e
        for rec in records
        for assignments in rec["layer_assignments"].values()
        for e in assignments
    ]
    num_experts = max(all_expert_ids) + 1 if all_expert_ids else 4

    output = {
        "label": args.label,
        "checkpoint_dir": str(args.checkpoint_dir),
        "total_steps": total_steps,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_probe_tokens": num_tokens_per_layer,
        "records": records,
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f)

    print(f"\n{'='*60}")
    print(f"Done. Saved {len(records)} records to {args.output_file}")
    print(f"  total_steps={total_steps}, num_layers={num_layers}, "
          f"num_experts={num_experts}, num_probe_tokens={num_tokens_per_layer}")


if __name__ == "__main__":
    main()
