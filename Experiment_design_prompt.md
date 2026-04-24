# Experiment Design & Implementation Prompt
## MoE-LLaVA: Diagnostic Evidence for Paper Figures

---

## IMPORTANT: READ THIS BEFORE WRITING ANY CODE

This document instructs you to build **four diagnostic analysis scripts** and
**one unified plotting script**. Each script produces evidence for a specific
claim in the paper. The scripts are analysis tools — they load existing
checkpoints and extract measurements. They do NOT retrain anything.

Read the full spec for all four experiments before touching a single file.
The scripts share data-loading patterns, so understanding all four first
prevents redundant code.

---

## Codebase orientation

Before writing any code, read these files in full:

- `moellava/eval/model_routing_probe.py` — existing hook-based router analysis
- `moellava/eval/model_routing_probe_v2.py` — improved version of the above
- `moellava/model/language_model/normalized_router_flexible.py` — gate classes,
  especially `SimplifiedNormalizedGate` and its `last_leak_loss`, `last_imbal_loss`,
  `last_entropy_loss`, `last_moe_loss` attributes
- `convergence_speed.py` — existing convergence plotting code to understand
  what checkpoint data already exists

Key facts about the codebase you must not forget:
- The router gate is `deepspeed_moe.gate` inside each MoE layer's `mlp`
- Gate class is `SimplifiedNormalizedGate` (no-teacher mode) or `NormalizedKDTopKGate`
- The router weight matrix is `gate.wg.weight`, shape `[num_experts, hidden_dim]`
- `self.k` on the gate holds the number of selected experts (DeepSpeed attribute)
- Hooks into `post_attention_layernorm` are how the existing probe captures
  hidden states — follow this pattern
- Checkpoints are under `/scratch/prafull/checkpoints_<backbone>_<variant>/`
- Existing variants: `author` (random init), `student` (K-means init, no KD),
  `TS` (teacher-student), `entropy` (no_teacher + entropy w=0.01),
  `entropy_w01` (no_teacher + entropy w=0.1)

---

## Experiment 1: Initialization quality — convergence curves

### Claim to prove
Fisher-initialized router weights are closer to the final trained state than
random (author) initialization. This shows as: lower loss at step 1, faster
convergence in early steps.

### What data you need
For each variant × backbone combination:
- Training loss (or ScienceQA accuracy) at steps: 1, 100, 200, 300, 400,
  500, 600, 700, 800, 900, 1000, 10000 (final)
- This data already exists from `EarlyDenseCheckpointCallback` and the
  `sqa_checkpoints/` directory under `eval_results_mine/`

### Script to write
**File:** `diagnostics/plot_convergence.py`

```
Purpose:
    Load per-step ScienceQA accuracy and/or training loss from existing
    checkpoint evaluation results. Plot convergence curves for all variants
    on the same axes, with log-scale x-axis.

Input:
    eval_results_mine/sqa_checkpoints/qwen_{variant}_step{step}.json
    for variant in [author, student, TS, entropy_w01]
    for step in [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

Output files:
    diagnostics/figures/convergence_qwen.pdf
    diagnostics/figures/convergence_qwen.png
    diagnostics/data/convergence_qwen.csv  (step, variant, sqa_acc columns)

Plot specification:
    - x-axis: training step, LOG SCALE (steps 1 to 1000 then 10000)
    - y-axis: ScienceQA accuracy (%)
    - One line per variant, consistent color across all figures in this paper:
        author (random):  coral / dashed
        student (K-means): teal / solid
        TS:                purple / solid
        entropy_w01:       amber / solid
    - Mark step=1 with a vertical dotted line labeled "step 1"
    - Annotate the y-value at step=1 for each variant directly on the plot
    - Legend inside upper-left corner
    - Figure size: 6×4 inches, 300 DPI
    - Save both PDF (for LaTeX) and PNG (for slides)

Key number to extract and print to stdout:
    "At step 1: author={x:.2f}%, student={x:.2f}%, TS={x:.2f}%, entropy={x:.2f}%"
    "Gap (student - author) at step 1: {x:.2f} percentage points"
    "Steps for author to match student's step-100 accuracy: {x}"
```

**Implementation notes:**
- Look at `convergence_speed.py` for how it loads these JSON files — reuse
  the loading logic, do not reinvent it
- If a JSON file is missing for a step/variant, skip that point and warn
- Use `matplotlib` with `plt.xscale('log')` for the log axis
- Export both `.pdf` and `.png` to the same figures directory

---

## Experiment 2: Feature norm growth and routing stability across layers

### Claim to prove
Feature norms grow with transformer depth (justifying why dot-product routing
fails in deep layers). Your cosine normalization produces stable routing
entropy across all layers, while vanilla routing collapses in deep layers.

### What data you need
Two measurements, collected via forward hooks:
1. L2 norm of hidden states at `post_attention_layernorm` output for each layer
2. Routing entropy per layer (Shannon entropy of the softmax routing distribution)

Both measured on a fixed diagnostic dataset (use the existing
`diagnostic_dataset/` or a small subset of the validation data).

### Script to write
**File:** `diagnostics/collect_layer_stats.py`

```
Purpose:
    Run inference on N=200 samples (enough for stable statistics, not too slow).
    For each MoE layer, collect:
        - mean L2 norm of hidden states entering the router
        - routing entropy: H(softmax(router_logits)) per token, averaged
        - routing assignment entropy: same but only using top-k probs
          renormalized (i.e. H(p_tilde_{T_k}))

    Compare two model checkpoints:
        (a) author checkpoint (random init, standard dot-product routing)
        (b) your student/entropy checkpoint (cosine-normalized routing)

Arguments:
    --model_path_a   path to author checkpoint
    --model_path_b   path to your checkpoint
    --label_a        "Random (dot-product)"    [for plot legend]
    --label_b        "Ours (cosine-normalized)"
    --n_samples      200
    --output_dir     diagnostics/data/

Output files:
    diagnostics/data/layer_stats_A.json
    diagnostics/data/layer_stats_B.json

    Each JSON has structure:
    {
      "layer_indices": [0, 2, 4, ...],       // which layers are MoE layers
      "feature_norms_mean": [...],           // one float per layer
      "feature_norms_std":  [...],
      "routing_entropy_mean": [...],         // Shannon H over all E experts
      "routing_entropy_std":  [...],
      "topk_entropy_mean": [...],            // H over renorm top-k only
      "topk_entropy_std":  [...]
    }

Hook pattern to follow (from model_routing_probe.py):
    Register a forward hook on each layer's post_attention_layernorm.
    In the hook: record output.detach().float().norm(dim=-1).mean().item()
    for feature norms, and separately capture router logits.

    For router logits: register a forward hook on gate.wg (the Linear layer).
    The input to gate.wg is the router input; the output is the raw logits.
    From logits, compute softmax → entropy.

    DO NOT modify the model. Use hooks only. Remove all hooks after inference.
```

**Implementation notes:**
- For Qwen backbone, MoE layers are every other layer (alternating). Check
  `config.moe['num_experts']` or look at `moe_layers_idx` in the model file
  to get the exact layer indices
- Use `torch.no_grad()` for all inference
- To load a checkpoint without running Stage III training, use:
  ```python
  from moellava.model.builder import load_pretrained_model
  ```
  Follow the pattern in `model_routing_probe.py` exactly
- Cap at N=200 samples using the diagnostic dataset or ScienceQA validation
  split (small, already downloaded)

---

### Plot script for Experiment 2
**File:** `diagnostics/plot_layer_stats.py`

```
Input:
    diagnostics/data/layer_stats_A.json
    diagnostics/data/layer_stats_B.json

Output:
    diagnostics/figures/feature_norms.pdf + .png
    diagnostics/figures/routing_entropy_by_layer.pdf + .png

Figure 1 — Feature norm growth:
    x-axis: layer index (only MoE layers)
    y-axis: mean L2 norm of hidden states
    Two lines: model A (coral, dashed) and model B (teal, solid)
    Shaded band: ±1 std
    Title: "Hidden state norms grow with depth"

Figure 2 — Routing entropy by layer:
    x-axis: layer index
    y-axis: routing entropy (Shannon H over all E experts)
    Two lines: model A and model B
    Horizontal dashed line at log(4) ≈ 1.386 labeled "max entropy (uniform)"
    Horizontal dashed line at log(2) ≈ 0.693 labeled "ideal top-2 entropy"
    Shaded band: ±1 std
    Title: "Routing entropy collapses in deep layers (dot-product) vs stays stable (ours)"

Key numbers to print to stdout:
    "Feature norm at layer 0: A={x:.2f}, B={x:.2f}"
    "Feature norm at final layer: A={x:.2f}, B={x:.2f}"
    "Norm growth ratio (final/first): A={x:.2f}x, B={x:.2f}x"
    "Routing entropy variance across layers: A={x:.4f}, B={x:.4f}"
    "Layers where H < 0.5 (collapsed): A={list}, B={list}"
```

---

## Experiment 3: Top-k entropy loss — routing split histogram

### Claim to prove
The old entropy loss drove top-2 routing toward single-expert behavior
(imbalanced splits). The new top-k entropy loss maintains genuine two-expert
utilization.

### What data you need
For each token in N=500 samples, given top-2 routing:
- The routing probabilities assigned to the two selected experts
- The "split ratio": `max(p1, p2) / (p1 + p2)` where p1, p2 are the top-2 probs
  A value of 1.0 = completely collapsed (all mass on one expert).
  A value of 0.5 = perfectly balanced.

Compare three checkpoints:
- `author` (random init, no special entropy)
- `entropy` (old entropy w=0.01, raw H minimization)
- `entropy_w01` (old entropy w=0.1, stronger)
- `new_entropy` (your retrained checkpoint with the redesigned loss) — add
  this when available; leave a placeholder stub for now

### Script to write
**File:** `diagnostics/collect_split_ratios.py`

```
Purpose:
    For each model checkpoint, run inference on N=500 samples.
    For every token × MoE layer, record the split ratio of the top-2
    routing decision.

    split_ratio = max(p_top1, p_top2) / (p_top1 + p_top2)

    This requires hooking the gate's forward pass to capture softmax probs.

Arguments:
    --model_path     path to checkpoint
    --label          short name for this run (e.g. "author", "entropy_old")
    --n_samples      500
    --output_dir     diagnostics/data/

Output:
    diagnostics/data/split_ratios_{label}.npz
    Contains: split_ratios array of shape [N_tokens × N_moe_layers],
    dtype float32, values in [0.5, 1.0]

Hook pattern:
    Register a forward hook on each gate's forward method.
    Inside the hook, access `gate.wg.weight` and the input to compute logits,
    or hook the output of the softmax inside the gate.

    Simplest approach: add a temporary attribute to the gate:
        gate._capture_probs = True
    Then modify the gate's forward temporarily to store probs in
        gate._last_probs  (shape [T, E])
    After each forward pass, read gate._last_probs and compute split_ratios.
    Remove the attribute after collection.

    Alternative (cleaner, no model modification):
    Hook the output of gate.wg (Linear). The output is raw logits [T, E].
    Apply softmax, then topk(2) to get the two selected probs.
    Compute split_ratio from those.

    Use the ALTERNATIVE (hook on gate.wg output). Do not modify the model.
```

---

### Plot script for Experiment 3
**File:** `diagnostics/plot_split_ratios.py`

```
Input:
    diagnostics/data/split_ratios_{label}.npz  for each label

Output:
    diagnostics/figures/split_ratio_histogram.pdf + .png

Figure specification:
    Histogram with bins from 0.5 to 1.0 (step 0.025)
    One overlaid histogram per model variant, semi-transparent (alpha=0.5)
    Colors: author=coral, entropy_old=amber, entropy_new=teal
    x-axis: "Split ratio (max_weight / sum_weights in top-2)"
    y-axis: "Fraction of routing decisions"
    Vertical dashed line at x=0.5 labeled "ideal (equal split)"
    Vertical dashed line at x=1.0 labeled "collapsed (one expert)"
    Normalize histograms so they sum to 1 (density=True equivalent)
    Legend top-center

Key numbers to print to stdout:
    For each variant:
    "% of decisions with split_ratio > 0.9 (near-collapse): {x:.1f}%"
    "% of decisions with split_ratio < 0.6 (near-balanced):  {x:.1f}%"
    "Median split ratio: {x:.3f}"
    "Mean split ratio:   {x:.3f}"
```

---

## Experiment 4: Router assignment stability (EMA teacher effect)

### Claim to prove
The EMA teacher provides trajectory smoothing: routing assignments stabilize
earlier in training for the TS (teacher-student) variant compared to the
student variant (no teacher).

### What data you need
For a fixed set of M=100 test tokens (from ScienceQA validation), at each
saved checkpoint step (1, 100, 200, ..., 1000), record the top-1 expert
assignment per token per MoE layer.

Then compute "stability score" between consecutive checkpoint pairs:
    stability(t, t+Δ) = fraction of (token, layer) pairs where
    top-1 expert at step t == top-1 expert at step t+Δ

### Script to write
**File:** `diagnostics/collect_routing_stability.py`

```
Purpose:
    For a fixed set of tokens, load each early checkpoint and record
    which expert is selected (top-1) per token per MoE layer.
    Compute pairwise stability between consecutive checkpoints.

Arguments:
    --checkpoint_dir   base directory containing step-N subdirectories
    --variant          e.g. "student" or "TS"
    --steps            1,100,200,300,400,500,600,700,800,900,1000
    --n_tokens         100
    --output_dir       diagnostics/data/

Output:
    diagnostics/data/stability_{variant}.json
    {
      "steps": [1, 100, 200, ...],
      "stability_consecutive": [0.42, 0.61, ...],  // stability(step_i, step_{i+1})
      "stability_vs_final": [0.31, 0.55, ...]       // stability(step_i, step_1000)
    }

Checkpoint loading:
    Each step checkpoint is a full model checkpoint saved by HuggingFace Trainer.
    Load with load_pretrained_model(), run inference on the fixed token set,
    record argmax of router probs per layer per token.

    IMPORTANT: Load and release each checkpoint inside the step loop.
    Do NOT keep all checkpoints in memory simultaneously — that will OOM.
    Pattern:
        for step in steps:
            model, ... = load_pretrained_model(f"{checkpoint_dir}/checkpoint-{step}")
            assignments[step] = run_inference(model, fixed_tokens)
            del model
            torch.cuda.empty_cache()

Fixed token set:
    Use the first 100 samples from ScienceQA validation that are single-image
    questions. Fix this set once and use the same indices for all variants.
    Save the fixed indices to diagnostics/data/fixed_token_indices.json
    so both student and TS variants use identical tokens.
```

---

### Plot script for Experiment 4
**File:** `diagnostics/plot_routing_stability.py`

```
Input:
    diagnostics/data/stability_student.json
    diagnostics/data/stability_TS.json

Output:
    diagnostics/figures/routing_stability.pdf + .png

Figure specification:
    x-axis: training step (log scale)
    y-axis: stability score (0 to 1)
    Two subplots side by side OR two lines on one plot:
        Left/Line 1: stability between consecutive checkpoints
        Right/Line 2: stability vs final checkpoint (step 1000)
    Colors: student=teal, TS=purple
    Horizontal dashed line at y=1.0 labeled "fully stable"
    Figure title: "Expert assignment stability during training"

Key numbers to print to stdout:
    "Step at which TS reaches 0.8 stability (consec): step {x}"
    "Step at which student reaches 0.8 stability (consec): step {x}"
    "Final stability (step 900 vs 1000): TS={x:.3f}, student={x:.3f}"
```

---

## Unified figure generation

### Script to write
**File:** `diagnostics/generate_all_figures.py`

```
Purpose:
    Single entry point that runs all four plot scripts in sequence and
    produces a summary report.

Usage:
    python diagnostics/generate_all_figures.py --data_dir diagnostics/data/

Output:
    Calls each plot_*.py script
    Prints a summary table:

    ┌─────────────────────────────────────────────────────────┐
    │ DIAGNOSTIC SUMMARY                                       │
    ├──────────────────────┬──────────────────────────────────┤
    │ Initialization gap   │ student - author at step 1: X pp │
    │ Norm growth ratio    │ A=Xx, B=Xx                       │
    │ Entropy variance     │ A=X.XX, B=X.XX                   │
    │ Near-collapse frac.  │ old=XX%, new=XX%                  │
    │ Stability advantage  │ TS reaches 0.8 at step X vs Y    │
    └──────────────────────┴──────────────────────────────────┘
```

---

## Directory structure to create

Before writing any script, create this layout:

```
diagnostics/
├── collect_layer_stats.py       ← Experiment 2 data collection
├── collect_split_ratios.py      ← Experiment 3 data collection
├── collect_routing_stability.py ← Experiment 4 data collection
├── plot_convergence.py          ← Experiment 1 plot (uses existing data)
├── plot_layer_stats.py          ← Experiment 2 plot
├── plot_split_ratios.py         ← Experiment 3 plot
├── plot_routing_stability.py    ← Experiment 4 plot
├── generate_all_figures.py      ← Unified entry point
├── data/                        ← All .json, .npz outputs
│   └── fixed_token_indices.json ← Fixed once, reused across Exp 4 variants
└── figures/                     ← All .pdf and .png outputs
```

---

## Shared utilities

Write these helper functions in `diagnostics/utils.py` and import them
in all scripts. Do NOT duplicate this logic.

```python
def load_model_for_inference(model_path, device="cuda"):
    """
    Load a checkpoint for read-only inference.
    Returns (tokenizer, model, image_processor, context_len).
    Follows the exact pattern from moellava/eval/model_routing_probe.py.
    """

def get_moe_gates(model):
    """
    Return a list of (layer_idx, gate) tuples for all MoE layers.
    Gate is model.model.layers[i].mlp.deepspeed_moe.gate for Qwen/StableLM.
    Gate is model.transformer.h[i].mlp.deepspeed_moe.gate for Phi2.
    Handle both architectures.
    """

def register_logit_hook(gate):
    """
    Register a forward hook on gate.wg (the Linear router weight matrix).
    The hook captures the output of gate.wg (raw logits, shape [T, E]).
    Returns (hook_handle, storage_list).
    Call storage_list.clear() before each forward pass.
    After each forward pass, storage_list[0] contains the logits tensor.
    """

def remove_all_hooks(hook_handles):
    """Remove a list of hook handles."""

VARIANT_COLORS = {
    "author":       "#D85A30",  # coral
    "student":      "#1D9E75",  # teal
    "TS":           "#7F77DD",  # purple
    "entropy":      "#BA7517",  # amber
    "entropy_w01":  "#EF9F27",  # amber lighter
    "new_entropy":  "#5DCAA5",  # teal lighter
}

STEP_CHECKPOINTS = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
```

---

## Strict constraints for the agent

1. Do NOT modify any existing training code. These are read-only analysis
   scripts. The only files you create are under `diagnostics/`.

2. Do NOT retrain any model. All scripts load existing checkpoints.

3. Always use `torch.no_grad()` during inference. Never call `.train()`.

4. Always `del model` and call `torch.cuda.empty_cache()` after loading
   each checkpoint, especially in the stability script that loops over steps.

5. Save both `.pdf` and `.png` for every figure. PDF goes into LaTeX. PNG
   goes into slides.

6. Every script must accept `--output_dir` as a command-line argument with
   a sensible default.

7. Print key numbers to stdout in the format specified. These numbers feed
   directly into the paper text.

8. If a checkpoint file does not exist, print a warning and skip — do not
   crash. Missing data points should be omitted from plots, not cause errors.

9. Use `matplotlib` only. No seaborn, no plotly. Keep style clean:
   `plt.style.use('seaborn-v0_8-whitegrid')` or equivalent minimal style.

10. Figure fonts: use 11pt for axis labels, 9pt for tick labels, 10pt for
    legends. These sizes look correct when a 6×4 inch PDF is scaled into
    a two-column paper layout.

---

## Order of implementation

Implement in this order. Each step is testable before the next begins.

1. `diagnostics/utils.py` — shared helpers first
2. `diagnostics/plot_convergence.py` — uses only existing JSON files,
   no new inference needed, verifiable immediately
3. `diagnostics/collect_layer_stats.py` — requires one model load
4. `diagnostics/plot_layer_stats.py` — verify with the data from step 3
5. `diagnostics/collect_split_ratios.py` — requires one model load per variant
6. `diagnostics/plot_split_ratios.py` — verify
7. `diagnostics/collect_routing_stability.py` — most expensive, loops over steps
8. `diagnostics/plot_routing_stability.py` — verify
9. `diagnostics/generate_all_figures.py` — wire everything together

Test step 2 before writing step 3. Do not implement all eight files at once.