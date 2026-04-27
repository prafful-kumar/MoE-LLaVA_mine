"""
Check pairwise cosine similarity of MoE expert weights in a checkpoint.
Reports per-layer and mean similarity — high similarity (>0.99) indicates
expert collapse.

Usage:
    python scripts/check_expert_cosine_sim.py <checkpoint_dir>
"""
import sys
import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import glob

ckpt_dir = sys.argv[1]

# Load all safetensors shards
shards = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
if not shards:
    # Try single file
    single = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(single):
        shards = [single]
    else:
        print("No safetensors found, trying pytorch_model.bin")
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        shards = None

if shards:
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))

# Find all gate_proj expert weights (pattern: .experts.X.gate_proj.weight)
# DeepSpeed MoE stores experts as: model.layers.N.mlp.experts.local_experts.X.w1/w2/w3
# or similar depending on the version

# Detect key patterns
all_keys = list(state_dict.keys())
expert_keys = [k for k in all_keys if "experts" in k and "gate_proj" in k]
if not expert_keys:
    expert_keys = [k for k in all_keys if "experts" in k and "w1" in k]
if not expert_keys:
    # Try down_proj as fallback
    expert_keys = [k for k in all_keys if "experts" in k and "down_proj" in k]

print(f"Checkpoint: {ckpt_dir}")
print(f"Total keys: {len(all_keys)}")
print(f"Expert weight keys found: {len(expert_keys)}")
if expert_keys:
    print(f"Example key: {expert_keys[0]}")
print()

# Group by layer
import re
layer_experts = {}
for k in expert_keys:
    # Extract layer number
    m = re.search(r'layers\.(\d+)', k)
    if m:
        layer = int(m.group(1))
        if layer not in layer_experts:
            layer_experts[layer] = {}
        # Extract expert index
        m2 = re.search(r'local_experts\.(\d+)', k)
        if m2:
            exp_idx = int(m2.group(1))
        else:
            m2 = re.search(r'experts\.(\d+)', k)
            exp_idx = int(m2.group(1)) if m2 else 0
        layer_experts[layer][exp_idx] = state_dict[k].float()

if not layer_experts:
    print("Could not parse expert layer structure. Dumping matching keys:")
    for k in all_keys[:30]:
        if "expert" in k.lower() or "moe" in k.lower():
            print(" ", k)
    sys.exit(1)

all_mean_sims = []
print(f"{'Layer':>5}  {'Experts':>7}  {'Mean Sim':>9}  {'Max Sim':>8}  {'Min Sim':>8}")
print("-" * 50)
for layer in sorted(layer_experts.keys()):
    experts = layer_experts[layer]
    if len(experts) < 2:
        continue
    weights = [F.normalize(experts[i].flatten(), dim=0) for i in sorted(experts.keys())]
    n = len(weights)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append((weights[i] * weights[j]).sum().item())
    mean_sim = sum(sims) / len(sims)
    all_mean_sims.append(mean_sim)
    print(f"{layer:>5}  {n:>7}  {mean_sim:>9.4f}  {max(sims):>8.4f}  {min(sims):>8.4f}")

print("-" * 50)
if all_mean_sims:
    overall = sum(all_mean_sims) / len(all_mean_sims)
    print(f"{'MEAN':>5}  {'':>7}  {overall:>9.4f}")
    print()
    if overall > 0.99:
        print("WARNING: Expert collapse detected (mean sim > 0.99)")
    elif overall > 0.95:
        print("CAUTION: High expert similarity (mean sim > 0.95)")
    else:
        print(f"OK: Expert diversity looks reasonable (mean sim = {overall:.4f})")