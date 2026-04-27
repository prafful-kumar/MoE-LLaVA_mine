"""
Shared utilities for MoE-LLaVA diagnostic scripts.
All data-collection and plotting scripts import from here.
"""

import json
import os
import re
import sys
import torch
import torch.nn as nn

# ── Make sure repo root is importable ──────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import get_model_name_from_path

# ── Colour palette (consistent across all paper figures) ──────────────────────
VARIANT_COLORS = {
    "author":            "#D85A30",   # coral
    "student":           "#1D9E75",   # teal
    "TS":                "#7F77DD",   # purple
    "TS_schedule":       "#9B6FCC",   # purple darker
    "entropy":           "#BA7517",   # amber
    "entropy_w01":       "#EF9F27",   # amber lighter
    "new_entropy":       "#5DCAA5",   # teal lighter
    "entropy_topk_var":  "#2563EB",   # royal blue
}

VARIANT_LINESTYLES = {
    "author":        "--",
    "student":       "-",
    "TS":            "-",
    "TS_schedule":   "-",
    "entropy":       "-",
    "entropy_w01":   "-",
    "new_entropy":   "-",
}

STEP_CHECKPOINTS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_for_inference(model_path, device="cuda"):
    """
    Load a checkpoint for read-only inference.
    Returns (tokenizer, model, image_processor, context_len).

    image_processor is the dict returned by load_pretrained_model;
    callers that need the actual processor should use result['image'].
    """
    # DeepSpeed needs distributed env vars even for single-GPU inference.
    # Set them if not already present so deepspeed.init_distributed works
    # without requiring an MPI launcher.
    # Derive a unique port from the GPU id so parallel jobs don't clash.
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_port = str(12400 + int(gpu_id) if gpu_id.isdigit() else 12388)
    for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", gpu_port),
                 ("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")]:
        os.environ.setdefault(k, v)

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, processor, context_len


# ── Gate discovery ─────────────────────────────────────────────────────────────

def get_moe_gates(model):
    """
    Return a list of (layer_idx, gate) for every MoE layer in the model.

    Handles two architectures:
      - Qwen / StableLM:  model.model.layers[i].mlp.deepspeed_moe.gate
      - Phi2:             model.transformer.h[i].mlp.deepspeed_moe.gate
    """
    gates = []
    for name, module in model.named_modules():
        # gate is a child of deepspeed_moe named exactly "gate"
        if not name.endswith(".gate"):
            continue
        # Must be inside an MoE block (deepspeed_moe parent)
        if "deepspeed_moe" not in name:
            continue

        # Parse layer index from name
        match = re.search(r'\.layers\.(\d+)\.', name) or \
                re.search(r'\.h\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(1))
        else:
            # Fallback: use position in list
            layer_idx = len(gates)

        gates.append((layer_idx, module))

    gates.sort(key=lambda x: x[0])
    return gates


# ── Hook utilities ─────────────────────────────────────────────────────────────

def register_logit_hook(gate):
    """
    Register a forward hook on gate.wg (the Linear router weight matrix).

    The hook captures the *output* of gate.wg, i.e. raw logits of shape [T, E].
    Returns (hook_handle, storage_list).

    Usage pattern:
        handle, store = register_logit_hook(gate)
        store.clear()
        model(...)          # forward pass
        logits = store[0]   # shape [T, E]
        handle.remove()
    """
    storage = []

    def _hook(module, inp, out):
        storage.append(out.detach().float())

    handle = gate.wg.register_forward_hook(_hook)
    return handle, storage


def remove_all_hooks(hook_handles):
    """Remove a list of hook handles returned by register_logit_hook."""
    for h in hook_handles:
        h.remove()


# ── Entropy helpers ────────────────────────────────────────────────────────────

def shannon_entropy(probs, eps=1e-8):
    """
    Shannon entropy H(p) = -sum(p * log(p)) for a probability vector.
    probs: tensor [..., E], assumed to sum to 1 along last dim.
    Returns scalar tensor.
    """
    return -(probs * (probs + eps).log()).sum(dim=-1)


def routing_entropy_from_logits(logits):
    """
    Compute full-distribution Shannon entropy from raw logits [T, E].
    Returns per-token entropy, shape [T].
    """
    probs = torch.softmax(logits, dim=-1)
    return shannon_entropy(probs)


def topk_entropy_from_logits(logits, k=2):
    """
    Routing entropy over the renormalised top-k distribution only.
    Returns per-token entropy, shape [T].
    """
    probs = torch.softmax(logits, dim=-1)
    topk_probs, _ = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return shannon_entropy(topk_probs)


# ── Dataset loading helpers ────────────────────────────────────────────────────

def load_question_file(path, n_samples=None):
    """
    Load a question file in JSON (list or dict) or JSONL format.

    Supported question formats:
      - ScienceQA / LLaVA instruct: list of dicts with "conversations" key
      - GQA / POPE / TextVQA:       list of dicts with "text" key (JSONL lines)

    Returns a list of dicts.
    """
    path = os.path.expanduser(path)
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON list
            questions = json.load(f)
        elif first_char == "{":
            # JSON object — try as list-of-dicts (JSONL with single object) or dict-of-dicts
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    questions = obj
                else:
                    # dict-of-dicts (e.g. ScienceQA problems.json) — not supported for iteration
                    raise ValueError(
                        f"{path} is a JSON dict, not a list. "
                        "Pass a JSONL or JSON-list question file instead."
                    )
            except json.JSONDecodeError:
                f.seek(0)
                questions = [json.loads(line) for line in f if line.strip()]
        else:
            # JSONL
            questions = [json.loads(line) for line in f if line.strip()]

    if n_samples is not None:
        questions = questions[:n_samples]
    return questions


def get_question_text(line):
    """
    Extract the question text from a question dict, handling multiple dataset formats:
      - ScienceQA / LLaVA instruct: line["conversations"][0]["value"]  (strips <image>)
      - GQA / POPE / VQA:           line["text"]
      - Generic:                     line["question"]
    """
    if "conversations" in line:
        return line["conversations"][0]["value"].replace("<image>", "").strip()
    if "text" in line:
        return line["text"].strip()
    if "question" in line:
        return line["question"].strip()
    return ""
