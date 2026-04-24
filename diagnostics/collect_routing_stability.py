"""
Experiment 4: Router assignment stability (EMA teacher effect).

For a fixed set of M=100 test tokens from ScienceQA val, loads each
step checkpoint and records the top-1 expert assignment per (token, layer).
Computes stability between consecutive checkpoints and vs final checkpoint.

Checkpoints are loaded one at a time and deleted to avoid OOM.

Usage:
    python diagnostics/collect_routing_stability.py \
        --checkpoint_dir /scratch/prafull/hpc/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe \
        --variant student \
        --steps 1,100,200,300,400,500,600,700,800,900,1000 \
        --n_tokens 100 \
        --output_dir diagnostics/data \
        --question_file /path/to/sqa_val.json \
        --image_folder /path/to/images \
        --conv_mode qwen

Run once for 'student' and once for 'TS', using the same fixed token indices.
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
    STEP_CHECKPOINTS,
)
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates
from moellava.mm_utils import tokenizer_image_token


FIXED_INDICES_FILE = "diagnostics/data/fixed_token_indices.json"


def load_or_create_fixed_indices(question_file, n_tokens, repo_root):
    """
    Load fixed token indices from disk (so both variants use the same tokens),
    or create and save them if they don't exist yet.
    Returns list of integer indices.
    """
    path = os.path.join(repo_root, FIXED_INDICES_FILE)
    if os.path.exists(path):
        with open(path) as f:
            indices = json.load(f)
        print(f"Loaded fixed token indices from {path}")
        return indices

    # Create: pick first n_tokens single-image questions from SQA val
    with open(os.path.expanduser(question_file)) as f:
        questions = json.load(f)

    indices = []
    for i, q in enumerate(questions):
        if "image" in q:
            indices.append(i)
        if len(indices) >= n_tokens:
            break

    if len(indices) < n_tokens:
        warnings.warn(
            f"Only found {len(indices)} single-image questions (wanted {n_tokens}). "
            "Using all of them."
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(indices, f)
    print(f"Saved fixed token indices to {path}")
    return indices


def get_expert_assignments(model, questions, fixed_indices, image_processor,
                           image_folder, conv_mode, tokenizer):
    """
    For the fixed question set, run inference and record top-1 expert
    assignment per (question, MoE layer).

    Returns np.array of shape [N_questions, N_moe_layers] with expert indices.
    """
    gates = get_moe_gates(model)
    if not gates:
        raise RuntimeError("No MoE gates found.")

    handles = []
    stores  = []
    for _, gate in gates:
        h, s = register_logit_hook(gate)
        handles.append(h)
        stores.append(s)

    n_layers = len(gates)
    # For each question, store the top-1 expert averaged over token positions
    # (take the mode / most common expert across the sequence)
    assignments = np.full((len(fixed_indices), n_layers), -1, dtype=np.int32)

    for qi, qidx in enumerate(fixed_indices):
        line = questions[qidx]
        qs   = line["conversations"][0]["value"].replace("<image>", "").strip()
        images = None

        if "image" in line:
            from PIL import Image
            img_path = (os.path.join(image_folder, line["image"])
                        if image_folder else line["image"])
            if os.path.exists(img_path):
                image  = Image.open(img_path).convert("RGB")
                images = image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0].unsqueeze(0).half().cuda()
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt    = conv.get_prompt()
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
            top1   = logits.argmax(dim=-1)  # [T]
            # Mode across token positions as representative assignment
            counts  = torch.bincount(top1, minlength=logits.shape[-1])
            assignments[qi, li] = int(counts.argmax().item())

    remove_all_hooks(handles)
    return assignments


def stability_score(a, b):
    """Fraction of (token, layer) pairs where top-1 expert matches."""
    valid = (a >= 0) & (b >= 0)
    if valid.sum() == 0:
        return float("nan")
    return float((a[valid] == b[valid]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Base dir containing checkpoint-N subdirectories")
    parser.add_argument("--variant", required=True,
                        help="e.g. 'student' or 'TS'")
    parser.add_argument("--steps", default=",".join(str(s) for s in STEP_CHECKPOINTS),
                        help="Comma-separated list of steps")
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--output_dir", default="diagnostics/data")
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--conv_mode", default="qwen")
    args = parser.parse_args()

    repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    steps = [int(s) for s in args.steps.split(",")]

    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)

    fixed_indices = load_or_create_fixed_indices(
        args.question_file, args.n_tokens, repo_root
    )
    # Clamp indices to available questions
    fixed_indices = [i for i in fixed_indices if i < len(questions)]

    print(f"Variant: {args.variant} | Steps: {steps} | Fixed tokens: {len(fixed_indices)}")

    # ── Load each checkpoint, collect assignments, delete ─────────────────────
    step_assignments = {}  # step → np.array [N_q, N_layers]

    for step in steps:
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_path):
            warnings.warn(f"Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n── Loading checkpoint-{step} ──────────────────────────────")
        try:
            tokenizer, model, processor, _ = load_model_for_inference(ckpt_path)
            image_processor = processor["image"]

            assignments = get_expert_assignments(
                model        = model,
                questions    = questions,
                fixed_indices= fixed_indices,
                image_processor= image_processor,
                image_folder = args.image_folder,
                conv_mode    = args.conv_mode,
                tokenizer    = tokenizer,
            )
            step_assignments[step] = assignments
            print(f"  Collected assignments shape: {assignments.shape}")

        except Exception as e:
            warnings.warn(f"Error at step {step}: {e}")
        finally:
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

    if not step_assignments:
        print("No checkpoints could be loaded. Exiting.")
        return

    sorted_steps = sorted(step_assignments.keys())

    # ── Compute stability ──────────────────────────────────────────────────────
    stability_consecutive = []
    stability_vs_final    = []
    final_step = sorted_steps[-1]
    final_asgn = step_assignments[final_step]

    for i, step in enumerate(sorted_steps):
        # vs final
        svf = stability_score(step_assignments[step], final_asgn)
        stability_vs_final.append(svf)

        # vs next step
        if i + 1 < len(sorted_steps):
            next_step = sorted_steps[i + 1]
            sc = stability_score(step_assignments[step], step_assignments[next_step])
            stability_consecutive.append(sc)
        else:
            stability_consecutive.append(float("nan"))  # last step has no "next"

    # ── Save ───────────────────────────────────────────────────────────────────
    out = {
        "variant":               args.variant,
        "checkpoint_dir":        args.checkpoint_dir,
        "steps":                 sorted_steps,
        "stability_consecutive": stability_consecutive,
        "stability_vs_final":    stability_vs_final,
    }
    out_path = os.path.join(output_dir, f"stability_{args.variant}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Key numbers ────────────────────────────────────────────────────────────
    print(f"\n── Stability summary for '{args.variant}' ──────────────────────")
    target = 0.8
    reach_step = next(
        (s for s, sc in zip(sorted_steps, stability_consecutive)
         if sc is not None and not np.isnan(sc) and sc >= target),
        None,
    )
    print(f"  Step at which '{args.variant}' reaches {target} stability (consec): "
          f"{'step ' + str(reach_step) if reach_step else 'never'}")

    # Step 900 vs 1000 stability (second-to-last vs last)
    if len(sorted_steps) >= 2:
        s_last   = sorted_steps[-1]
        s_second = sorted_steps[-2]
        sc_final = stability_score(step_assignments.get(s_second),
                                   step_assignments.get(s_last)) \
            if s_second in step_assignments and s_last in step_assignments \
            else float("nan")
        print(f"  Final stability (step {s_second} vs {s_last}): {sc_final:.3f}")


if __name__ == "__main__":
    main()
