# Agent Prompt: Implement Margin-Aware Adaptive Entropy Loss

## What You Are Doing

You are replacing the current uniform entropy loss in `SimplifiedNormalizedGate` with
a margin-aware version that adapts per token. The core idea: if the router is already
confident about a token (high probability gap between top-1 and top-2 expert), suppress
the entropy penalty for that token. If it is uncertain, apply the full penalty.

This affects ONE primary file and cascades to SIX supporting files.
Read ALL files listed below before touching anything.

---

## Files to Read First (in this order)

1. `moellava/model/language_model/normalized_router_flexible.py`  ← PRIMARY CHANGE
2. `moellava/train/train.py`                                       ← add new args
3. `moellava/model/language_model/llava_phi_moe.py`               ← pass new args
4. `moellava/model/language_model/llava_qwen_moe.py`              ← pass new args
5. `moellava/model/language_model/llava_stablelm_moe.py`          ← pass new args
6. `get_kmeans_centroids/compute_fisher_directions_phi.py`         ← increase samples
7. `get_kmeans_centroids/compute_fisher_directions_qwen.py`        ← increase samples

---

## Background: Current Loss in SimplifiedNormalizedGate

The current `forward()` in `SimplifiedNormalizedGate` computes:

```python
probs = F.softmax(logits, dim=-1)
H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
entropy_loss = self.entropy_loss_weight * H
total_loss = aux_loss + entropy_loss
```

This is a single scalar. It applies the same pressure to every token regardless of
whether the router is already confident about that token or not.

The problem: tasks like POPE require focused routing (confident tokens). This uniform
penalty causes those tokens to be pushed toward balanced top-2 routing, causing POPE
regression (observed in experiments with entropy_loss_weight=0.1).

---

## New Loss: Margin-Aware Adaptive Entropy (L_adaptive)

### Mathematical Definition

```
probs         = softmax(logits)              shape: [T, E]
top2_probs    = top-2 values of probs        shape: [T, 2]  (sorted descending)
prob_margin   = top2_probs[:,0] - top2_probs[:,1]   shape: [T]   range: [0, 1]

alpha(t)      = exp(-gamma * prob_margin[t]).detach()   shape: [T]
                # .detach() is CRITICAL — see note below

p_tilde(t)    = top2_probs[t] / top2_probs[t].sum()    shape: [T, 2]
                # renormalize over top-2 only

L_adaptive    = mean over t of [ alpha(t) * KL(p_tilde(t) || uniform_2) ]
              = mean over t of [ alpha(t) * (log(2) - H(p_tilde(t))) ]

total_loss    = aux_loss + entropy_loss_weight * L_adaptive
```

### Why prob_margin (NOT logit margin)

DO NOT use `m = logit[:,0] - logit[:,1]` (logit margin).
USE `m = top2_probs[:,0] - top2_probs[:,1]` (probability margin).

Reason: our router uses logit_scale=10.0 which multiplies all cosine similarities by 10.
Logit margins are therefore also ×10, which makes the gamma hyperparameter extremely
sensitive and hard to tune. Probability margin is always in [0, 1] and is independent
of logit_scale, so gamma=2.0 has the same meaning regardless of architecture.

### Why .detach() on alpha

Without `.detach()`, gradients flow through `prob_margin → probs → logits → wg.weight`.
This means the router can learn to artificially inflate the confidence margin just to
suppress its own penalty — an adversarial hack of the loss. With `.detach()`, alpha
is treated as a fixed per-token weight computed from the current forward pass, not as
a differentiable quantity. Gradients only flow through the KL term.

### Behavior at boundary conditions

- Token with prob_margin → 1.0 (perfectly confident):
  alpha → exp(-gamma * 1.0) ≈ 0.135 (gamma=2), penalty almost zero ✓
- Token with prob_margin ≈ 0.0 (completely uncertain):
  alpha → exp(0) = 1.0, full KL penalty fires ✓
- gamma=2.0 means: alpha=0.5 when prob_margin=0.35 (moderate confidence threshold)

---

## CHANGE 1: normalized_router_flexible.py — SimplifiedNormalizedGate

### 1a. __init__: add new parameters

Find the `__init__` of `SimplifiedNormalizedGate`. It currently has:
```python
entropy_loss_weight=0.0,
```

Add TWO new parameters directly after `entropy_loss_weight`:
```python
adaptive_gamma=2.0,          # NEW: steepness of confidence gating
use_adaptive_entropy=False,  # NEW: if True, use margin-aware loss; if False, use old uniform loss
```

Store them as instance attributes:
```python
self.adaptive_gamma = adaptive_gamma
self.use_adaptive_entropy = use_adaptive_entropy
```

Also add TWO new logging attributes to __init__ (same pattern as `self.last_entropy_loss`):
```python
self.last_adaptive_loss = 0.0    # NEW: raw L_adaptive (unweighted), for logging
self.last_alpha_mean = 0.0       # NEW: mean alpha across tokens, for logging
```

### 1b. forward: replace entropy computation

Find the block in `forward()` that currently reads:
```python
if self.entropy_loss_weight > 0.0:
    probs = F.softmax(logits, dim=-1)
    H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
    entropy_loss = self.entropy_loss_weight * H
    self.last_entropy_loss = H.item()
    total_loss = aux_loss + entropy_loss
else:
    self.last_entropy_loss = 0.0
    total_loss = aux_loss
```

Replace the entire block with this:
```python
if self.entropy_loss_weight > 0.0:
    probs = F.softmax(logits.float(), dim=-1)   # [T, E], float32 for stability

    if self.use_adaptive_entropy:
        # --- Margin-Aware Adaptive Entropy ---
        # Step 1: get top-2 probs per token
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)   # [T, 2], sorted descending

        # Step 2: probability margin in [0, 1] — logit_scale-independent
        prob_margin = top2_probs[:, 0] - top2_probs[:, 1]  # [T]

        # Step 3: confidence gate alpha — DETACHED so it does not affect gradients
        # alpha ≈ 0 when confident (large margin), alpha ≈ 1 when uncertain (small margin)
        alpha = torch.exp(-self.adaptive_gamma * prob_margin).detach()  # [T]

        # Step 4: renormalize top-2 probs to form p_tilde
        p_tilde = top2_probs / (top2_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [T, 2]

        # Step 5: per-token KL( p_tilde || uniform_2 ) = log(2) - H(p_tilde)
        H_top2 = -(p_tilde * torch.log(p_tilde + 1e-8)).sum(dim=-1)   # [T]
        kl_per_token = math.log(2) - H_top2                           # [T], >= 0

        # Step 6: weighted average — confident tokens contribute little
        L_adaptive = (alpha * kl_per_token).mean()

        # Logging (unweighted)
        self.last_entropy_loss = L_adaptive.item()
        self.last_adaptive_loss = L_adaptive.item()
        self.last_alpha_mean = alpha.mean().item()

        total_loss = aux_loss + self.entropy_loss_weight * L_adaptive

    else:
        # --- Original Uniform Entropy Loss (unchanged) ---
        H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
        entropy_loss = self.entropy_loss_weight * H
        self.last_entropy_loss = H.item()
        self.last_adaptive_loss = 0.0
        self.last_alpha_mean = 1.0
        total_loss = aux_loss + entropy_loss
else:
    self.last_entropy_loss = 0.0
    self.last_adaptive_loss = 0.0
    self.last_alpha_mean = 1.0
    total_loss = aux_loss
```

Note: you need `import math` at the top of the file. Check if it is already imported.
If not, add `import math` alongside the other imports.

### 1c. get_loss_dict: add new fields

Find `get_loss_dict`. It currently returns:
```python
return {
    'moe_loss': self.last_moe_loss,
    'kd_loss': 0.0,
    'entropy_loss': self.last_entropy_loss,
    ...
}
```

Add two new keys:
```python
'adaptive_loss': self.last_adaptive_loss,   # NEW
'alpha_mean': self.last_alpha_mean,          # NEW
```

### 1d. update_hyperparameters: add new fields

Find `update_hyperparameters`. It currently accepts `entropy_loss_weight`. Add:
```python
def update_hyperparameters(self, ..., entropy_loss_weight=None,
                           adaptive_gamma=None, use_adaptive_entropy=None):
    ...
    if adaptive_gamma is not None:
        self.adaptive_gamma = adaptive_gamma
    if use_adaptive_entropy is not None:
        self.use_adaptive_entropy = use_adaptive_entropy
```

---

## CHANGE 2: train.py — ModelArguments

Find the `ModelArguments` dataclass. It currently has `entropy_loss_weight`. Add these
TWO new fields directly after it:

```python
adaptive_gamma: float = field(
    default=2.0,
    metadata={"help": "Steepness of confidence gating in margin-aware entropy loss. "
                       "alpha(m) = exp(-gamma * prob_margin). "
                       "gamma=2.0: alpha=0.5 at prob_margin=0.35 (moderate confidence). "
                       "gamma=5.0: more aggressive suppression for confident tokens. "
                       "Only used when use_adaptive_entropy=True."}
)
use_adaptive_entropy: bool = field(
    default=False,
    metadata={"help": "If True, use margin-aware adaptive entropy loss (L_adaptive). "
                       "If False (default), use original uniform entropy loss. "
                       "Requires entropy_loss_weight > 0 to have any effect."}
)
```

---

## CHANGE 3: llava_phi_moe.py, llava_qwen_moe.py, llava_stablelm_moe.py

In each of the three model files, find where `SimplifiedNormalizedGate` is instantiated.
It currently passes `entropy_loss_weight=getattr(model_args, 'entropy_loss_weight', 0.0)`.

Add the two new arguments directly after it using `getattr` with defaults
(this ensures backward compatibility if loading old checkpoints):

```python
entropy_loss_weight=getattr(model_args, 'entropy_loss_weight', 0.0),
adaptive_gamma=getattr(model_args, 'adaptive_gamma', 2.0),           # NEW
use_adaptive_entropy=getattr(model_args, 'use_adaptive_entropy', False),  # NEW
```

Do this for ALL three model files. The pattern is identical in each.

---

## CHANGE 4: Increase K-means/Fisher samples to 20k for Phi and Qwen scripts

### Background

The Phi and Qwen backbone Fisher scripts have `--num_samples` hardcoded to **5000**,
while the generic/StableLM script already uses 20000. For Fisher LDA specifically,
reliable scatter matrix estimation needs roughly 5×–10× the hidden dimension per class.
With 4 classes and hidden dim ~2048, 5000 total tokens is below the reliable threshold.
20000 is meaningfully better for Fisher (scatter matrix quality improves). For K-means
alone the gain is smaller, but consistent with the other scripts.

### Files to change

**File 1: `get_kmeans_centroids/compute_fisher_directions_phi.py`**

Find these two lines in `parse_args()`:
```python
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--output_file", type=str, default="fisher_directions_phi/5000.pkl")
```

Change them to:
```python
parser.add_argument("--num_samples", type=int, default=20000)
parser.add_argument("--output_file", type=str, default="fisher_directions_phi/20000.pkl")
```

**File 2: `get_kmeans_centroids/compute_fisher_directions_qwen.py`**

Find these two lines in `parse_args()`:
```python
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--output_file", type=str, default="fisher_directions_qwen/5000.pkl")
```

Change them to:
```python
parser.add_argument("--num_samples", type=int, default=20000)
parser.add_argument("--output_file", type=str, default="fisher_directions_qwen/20000.pkl")
```

### What NOT to change in these files

Do NOT change anything else in these two scripts — not the Fisher computation logic,
not `--max_tokens_for_fisher`, not `--buffer_size`. Only the two lines above per file.

### Shell script update

After changing the Python defaults, find any shell scripts under
`get_kmeans_centroids/` (e.g. `run.sh`, `run_adv.sh`) that pass
`--output_file` with a path containing `5000.pkl`. Update those paths to `20000.pkl`
to stay consistent. If the shell scripts do not pass `--output_file` explicitly,
no shell script change is needed (the new Python default will take effect).

Also check the training shell scripts under `scripts/` for any line that passes
`--router_centroids_path` or `--router_init_path` pointing to a `5000.pkl` file.
If found, update those paths to `20000.pkl` as well — otherwise the training job
will still load the old low-sample centroids even after you regenerate them.

---

## Verification Tests

After making all changes, write a standalone test script at:
`tests/test_adaptive_entropy.py`

The script must NOT require DeepSpeed or a GPU. Use a mock TopKGate parent if needed.
Test these 5 cases and print PASS/FAIL for each:

```
TEST 1 — Confident token: probs = [0.95, 0.04, 0.005, 0.005]
  Expected: alpha < 0.2 (penalty almost suppressed for this token)

TEST 2 — Uncertain token: probs = [0.26, 0.25, 0.25, 0.24]
  Expected: alpha > 0.9 (full penalty for this token)

TEST 3 — use_adaptive_entropy=False still works (old path unchanged)
  Call forward with use_adaptive_entropy=False and entropy_loss_weight=0.03
  Expected: last_adaptive_loss == 0.0, last_alpha_mean == 1.0

TEST 4 — .detach() is in place (critical correctness check)
  alpha must NOT appear in the computation graph of L_adaptive
  Expected: L_adaptive.grad_fn should not mention prob_margin

TEST 5 — Gradient flows through KL term (not blocked by detach)
  Set requires_grad=True on a dummy weight, run forward, call .backward()
  Expected: gradient is not None and not NaN
```

Run the test script and confirm all 5 pass before finishing.

---

## What NOT to Change

- Do NOT modify `NormalizedKDTopKGate` — it is used for teacher-student mode only
- Do NOT modify `EntropyWarmupCallback` — it reads `entropy_loss_weight` which still exists
- Do NOT change the `entropy_loss_weight` parameter name — existing shell scripts use it
- Do NOT add `use_adaptive_entropy` or `adaptive_gamma` to the shell scripts —
  the defaults (False and 2.0) are correct for backward compatibility. Only add them
  to a shell script when explicitly creating a new experiment run.
- Do NOT modify the return signature of `forward()` — it must remain a 4-tuple:
  `(total_loss, expert_weights, expert_mask, expert_counts)`

---

## Summary of New Parameters

| Parameter | Default | Where set | Meaning |
|---|---|---|---|
| `use_adaptive_entropy` | `False` | `--use_adaptive_entropy` in shell script | Switches between old and new loss |
| `adaptive_gamma` | `2.0` | `--adaptive_gamma 2.0` in shell script | How quickly penalty drops for confident tokens |

To run the new loss on Phi2, add to the training shell script:
```bash
--use_adaptive_entropy True \
--adaptive_gamma 2.0 \
--entropy_loss_weight 0.03 \
```
