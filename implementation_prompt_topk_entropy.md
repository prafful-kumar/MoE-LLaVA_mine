# Implementation Prompt: Redesigned Top-k Entropy Loss

## Context

You are working on a research codebase called **MoE-LLaVA** — a Mixture-of-Experts 
Vision-Language Model built on top of DeepSpeed's MoE framework. The router (gating 
function) selects top-k experts per token. Currently the codebase uses a broken entropy
loss that minimizes raw Shannon entropy H toward 0, which creates one-hot routing even
when k=2. We are replacing it with a mathematically correct k-adaptive confidence loss.

**Primary file to edit:**
`moellava/model/language_model/normalized_router_flexible.py`

**Secondary file to edit:**
`moellava/train/router_callback.py`

Read both files fully before making any change.

---

## What the current code does (and why it is wrong)

Inside `SimplifiedNormalizedGate.forward()`, the current entropy computation is:

```python
probs = F.softmax(logits, dim=-1)
H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
entropy_loss = self.entropy_loss_weight * H
```

This minimizes H toward 0, which is correct only for k=1. For k=2 (our architecture),
the minimum achievable entropy for a correct routing decision is log(2) ≈ 0.693, not 0.
The current loss therefore penalizes correct top-2 routing and drives the router toward 
single-expert behavior — wasting the second expert slot entirely.

There are also two independent failure modes that one scalar H cannot distinguish:
1. **Leakage**: probability mass assigned to non-selected experts (outside top-k)
2. **Within-k collapse**: all top-k mass concentrates on one expert (router acts top-1)

---

## What we want (the new loss)

### Mathematical definition

Let:
- `p` = softmax distribution over E experts, shape [T, E]
- `T_k` = indices of the top-k experts per token
- `p_tilde` = p renormalized over T_k, shape [T, k]

**Term 1 — Leakage (per-token):**
```
L_leak = E_t[ 1 - sum_{i in T_k} p_i ]
```
Penalizes probability mass outside the selected set. Enforces consistency between the
continuous softmax and the discrete top-k routing decision.

**Term 2 — Imbalance (per-token, then batch-averaged):**
```
L_imbal = E_t[ KL( p_tilde || u_k ) ]
        = E_t[ log(k) - H(p_tilde) ]
```
Prevents within-k collapse. When k experts are selected, penalizes the router for
effectively routing through fewer than k of them. u_k is uniform-over-k (each = 1/k).
Applied per-token (NOT batch-mean-then-KL) because torch.topk returns rank-ordered
values, so averaging over dim=0 would average ranks, not expert identities.

**Combined:**
```
L_ent = L_leak + lam * L_imbal
```
where `lam` corresponds to the existing `entropy_loss_weight` hyperparameter.

**Key properties:**
- k=1: L_imbal = 0 identically (KL of single-element distribution = 0). Reduces to 
  pure leakage, which is correct for top-1.
- k=2: target entropy is log(2) ≈ 0.693, not 0. [0.5, 0.5] routing costs 0.
- k=4: target entropy is log(4) ≈ 1.386. [0.25, 0.25, 0.25, 0.25] routing costs 0.
- KL ≥ 0 always (Gibbs inequality), so no one-sided clamping quirks — but keep a 
  numerical safety clamp(min=0.0) to guard against floating-point errors in p_tilde.

---

## Step-by-step changes

### CHANGE 1: Add the standalone loss function

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Add this function near the top of the file, after the imports, before any class 
definitions:

```python
def topk_entropy_loss(router_logits, k, lam=1.0, eps=1e-8):
    """
    K-adaptive routing confidence loss.

        L_ent = L_leak + lam * L_imbal

    L_leak  (per-token):
        Probability mass outside the top-k selected experts.
        Enforces consistency between the softmax distribution and
        the discrete top-k routing decision.

    L_imbal (per-token, batch-averaged):
        E_t[ KL( p_tilde_{T_k} || u_k ) ] = E_t[ log(k) - H(p_tilde_{T_k}) ]
        Prevents within-k collapse: when k experts are selected, the router
        should actually use all k slots, not concentrate on 1.

    Why per-token (NOT batch-mean-over-ranks):
        torch.topk returns values sorted by magnitude, so dim=1 is rank position,
        not expert identity. Averaging over dim=0 computes a rank distribution, not
        expert utilization. This does NOT decouple token-level shape from batch-level
        behavior. Per-token is therefore the correct scope.

    Note:
        lam here is always 1.0 — the external entropy_loss_weight scales the entire
        L_ent from outside, exactly as it scaled H before. No behavior change to the
        training script or callbacks is needed.

    Args:
        router_logits : [T, E]  raw logits before softmax (grad-connected)
        k             : int, number of selected experts (top_k)
        lam           : float, relative weight of imbalance vs leakage (default 1.0)
        eps           : float, numerical stability

    Returns:
        Scalar loss tensor (grad-connected through router_logits).
    """
    probs = F.softmax(router_logits, dim=-1)              # [T, E]
    topk_probs, _ = torch.topk(probs, k, dim=-1)          # [T, k]
    topk_mass     = topk_probs.sum(dim=-1)                 # [T]

    # ── L_leak: per-token ───────────────────────────────────────────
    L_leak = (1.0 - topk_mass).mean()

    # ── L_imbal: per-token KL to uniform-over-k, then batch-averaged
    if k == 1:
        L_imbal = router_logits.new_zeros(())
    else:
        p_tilde = topk_probs / (topk_mass.unsqueeze(-1) + eps)      # [T, k]
        H_topk  = -(p_tilde * torch.log(p_tilde + eps)).sum(dim=-1) # [T]
        log_k   = torch.log(torch.tensor(float(k), device=router_logits.device))
        L_imbal = (log_k - H_topk).clamp(min=0.0).mean()

    return L_leak + lam * L_imbal
```

---

### CHANGE 2: Replace the broken entropy computation in SimplifiedNormalizedGate.forward()

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Find the `forward` method of `SimplifiedNormalizedGate`. Look for this block:

```python
if self.entropy_loss_weight > 0.0:
    probs = F.softmax(logits, dim=-1)
    H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
    entropy_loss = self.entropy_loss_weight * H
    self.last_entropy_loss = H.item()
    gate_output = (gate_output[0] + entropy_loss,) + gate_output[1:]
else:
    self.last_entropy_loss = 0.0
```

Replace it with:

```python
if self.entropy_loss_weight > 0.0:
    # topk_entropy_loss returns L_leak + lam * L_imbal (already combined).
    # We scale the combined loss by entropy_loss_weight, exactly as we
    # previously scaled H. The callback (EntropyWarmupCallback) continues
    # to ramp entropy_loss_weight from 0 to target with no changes needed.
    k = self.top_k  # number of selected experts — already stored on gate
    L_ent = topk_entropy_loss(logits, k=k, lam=1.0)
    entropy_loss = self.entropy_loss_weight * L_ent
    self.last_entropy_loss = L_ent.item()   # log the unweighted combined loss
    gate_output = (gate_output[0] + entropy_loss,) + gate_output[1:]
else:
    self.last_entropy_loss = 0.0
```

**Important:** `logits` here must be the raw pre-softmax logits, NOT the output of
softmax. Confirm that `logits` in the forward method is the output of `self.wg(input)`
before any softmax — it should be, because softmax happens inside the parent 
`TopKGate.forward()` call. Pass `logits` (or whichever variable holds the pre-softmax
output) to `topk_entropy_loss`. Do NOT pass probabilities.

**Also confirm:** `self.top_k` exists on the gate object. If it is named differently
(e.g. `self.k`, `self.num_selects`, or accessed as `self.config.top_k`), use that.
Search the class `__init__` for how k is stored. In DeepSpeed's `TopKGate`, it is 
typically set as `self.k = k` or similar in the parent. Check and use the correct 
attribute name.

---

### CHANGE 3: Add separate logging for leak and imbalance (optional but recommended)

This makes ablation analysis easier. In `SimplifiedNormalizedGate.__init__`, add:

```python
self.last_leak_loss   = 0.0
self.last_imbal_loss  = 0.0
```

Then update the forward method to log them separately. To do this, temporarily 
split the computation (only for logging — do not double-compute gradients):

```python
if self.entropy_loss_weight > 0.0:
    k = self.top_k
    # Compute full loss (grad-connected)
    L_ent = topk_entropy_loss(logits, k=k, lam=1.0)
    entropy_loss = self.entropy_loss_weight * L_ent
    
    # Log components (no_grad, for monitoring only)
    with torch.no_grad():
        probs_ng    = F.softmax(logits, dim=-1)
        topk_p, _   = torch.topk(probs_ng, k, dim=-1)
        topk_mass_  = topk_p.sum(dim=-1)
        self.last_leak_loss  = (1.0 - topk_mass_).mean().item()
        if k > 1:
            p_t     = topk_p / (topk_mass_.unsqueeze(-1) + 1e-8)
            H_      = -(p_t * torch.log(p_t + 1e-8)).sum(dim=-1)
            log_k_  = torch.log(torch.tensor(float(k)))
            self.last_imbal_loss = (log_k_ - H_).clamp(min=0.0).mean().item()
        else:
            self.last_imbal_loss = 0.0
    
    self.last_entropy_loss = L_ent.item()
    gate_output = (gate_output[0] + entropy_loss,) + gate_output[1:]
else:
    self.last_entropy_loss = 0.0
    self.last_leak_loss    = 0.0
    self.last_imbal_loss   = 0.0
```

Update the `get_diagnostics` method (or wherever `last_entropy_loss` is returned) to
also return `last_leak_loss` and `last_imbal_loss`.

---

### CHANGE 4: No changes needed to these files

The following files do **not** need modification:

- `moellava/train/router_callback.py` — `EntropyWarmupCallback` ramps 
  `entropy_loss_weight` from 0 to target. This still works correctly because 
  `entropy_loss_weight` scales `L_ent` exactly as it previously scaled `H`.

- `moellava/train/train.py` — `ModelArguments.entropy_loss_weight` field is unchanged.
  `EntropyWarmupCallback` instantiation is unchanged.

- All model files (`llava_qwen_moe.py`, `llava_phi_moe.py`, `llava_stablelm_moe.py`) —
  they pass `entropy_loss_weight` to `SimplifiedNormalizedGate` and this interface is
  unchanged.

- All training shell scripts — `entropy_loss_weight` argument is unchanged.

---

### CHANGE 5: Update the existing entropy training scripts (documentation only)

In each of these scripts:
- `scripts/v1/qwen/finetune_moe_entropy.sh`
- `scripts/v1/qwen/finetune_moe_entropy_w01.sh`
- `scripts/v1/phi/finetune_moe_entropy.sh`
- `scripts/v1/stablelm/finetune_moe_entropy.sh`

Add a comment at the top:

```bash
# Entropy loss: redesigned top-k aware loss (L_leak + L_imbal).
# See normalized_router_flexible.py::topk_entropy_loss for formulation.
# entropy_loss_weight scales the combined L_ent = L_leak + L_imbal.
# Correct for k=2 (top-2 routing). Reduces to leakage-only for k=1.
```

---

## Verification checklist after implementing

Run this quick sanity check in Python to confirm the function behaves correctly 
before running a full training job:

```python
import torch
import torch.nn.functional as F

# Paste the topk_entropy_loss function here, then:

# Test 1: ideal top-2 routing should give loss ≈ 0
logits_ideal = torch.tensor([[2.0, 2.0, -10.0, -10.0]])  # softmax ≈ [0.5, 0.5, 0, 0]
loss_ideal = topk_entropy_loss(logits_ideal, k=2)
print(f"Ideal top-2 loss: {loss_ideal:.6f}")   # should be close to 0.0

# Test 2: one-hot routing should give positive loss for k=2
logits_onehot = torch.tensor([[10.0, -10.0, -10.0, -10.0]])  # softmax ≈ [1, 0, 0, 0]
loss_onehot = topk_entropy_loss(logits_onehot, k=2)
print(f"One-hot k=2 loss: {loss_onehot:.6f}")  # should be > 0

# Test 3: uniform routing should give positive leakage for k=2 (mass on non-top-2)
logits_uniform = torch.zeros(1, 4)  # softmax = [0.25, 0.25, 0.25, 0.25]
loss_uniform = topk_entropy_loss(logits_uniform, k=2)
print(f"Uniform k=2 loss: {loss_uniform:.6f}")  # positive leakage

# Test 4: k=1 with one-hot should give ≈ 0
logits_ok_k1 = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
loss_ok_k1 = topk_entropy_loss(logits_ok_k1, k=1)
print(f"One-hot k=1 loss: {loss_ok_k1:.6f}")   # should be close to 0.0

# Test 5: k=1 L_imbal must be exactly 0
logits_any = torch.randn(8, 4)
loss_k1 = topk_entropy_loss(logits_any, k=1)
# Confirm by manually computing leakage only
probs = F.softmax(logits_any, dim=-1)
topk_mass = probs.max(dim=-1).values
leak_only = (1.0 - topk_mass).mean()
print(f"k=1 difference from pure leakage: {abs(loss_k1.item() - leak_only.item()):.8f}")
# should be 0.0
```

Expected outputs:
```
Ideal top-2 loss:      ~0.000     (both selected experts equally used, no leakage)
One-hot k=2 loss:      ~0.693     (imbalance = log(2), no leakage since top-2 capture all mass)
Uniform k=2 loss:      ~0.500     (leakage from experts 3 and 4)
One-hot k=1 loss:      ~0.000     (correct for k=1)
k=1 difference:         0.0       (imbalance term is exactly 0 for k=1)
```

---

## Summary of what changed and why

| What | Before | After | Why |
|------|--------|-------|-----|
| Entropy target | H → 0 (one-hot) | L_leak → 0, L_imbal → 0 | H=0 is wrong target for k>1 |
| Scope of entropy | All E experts | Renormalized over top-k | KL must operate on correct sub-distribution |
| Failure modes | Mixed into one scalar | Separated into leak + imbal | Orthogonal problems need separate signals |
| k=1 behavior | Correct by coincidence | Correct by construction | L_imbal = 0 identically for k=1 |
| External interface | entropy_loss_weight scales H | entropy_loss_weight scales L_ent | No change to any other file |
| Callbacks | EntropyWarmupCallback ramps weight | Unchanged | Interface is weight → gate, unchanged |
