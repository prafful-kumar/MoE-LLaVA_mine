# Implementation Prompt: Complete Routing Loss Redesign (v2)

## What changed from v1

v1 introduced `L_leak + L_imbal` to replace raw entropy minimization. Since then,
three things changed based on experimental results and deeper analysis:

1. **L_var added.** Results showed POPE dropping 1.5–2% across all backbones with
   the new entropy loss. Root cause: `L_imbal` is a per-token loss that cannot detect
   batch-level expert starvation (E3 and E4 never selected even when per-token routing
   looks perfect). `L_var` is a new batch-level term that directly penalizes unequal
   expert utilization. It replaces the author's `L_aux` with an original formulation.

2. **`lam` (imbalance weight) exposed as a hyperparameter.** Results show `lam=1.0`
   hurts POPE (focused grounding tasks) while helping MME cognition (reasoning tasks).
   The right value is empirically between 0.05 and 0.2. It must now be configurable
   from the training script, not hardcoded.

3. **`balance_loss_weight` added as a new training argument.** `L_var` needs its own
   weight separate from `entropy_loss_weight`. These control different failure modes
   at different scopes and must be tunable independently.

---

## Context

You are working on a research codebase called **MoE-LLaVA** — a Mixture-of-Experts
Vision-Language Model built on top of DeepSpeed's MoE framework. The router (gating
function) selects top-k experts per token.

**Files to edit (in order):**
1. `moellava/model/language_model/normalized_router_flexible.py` — gate logic
2. `moellava/train/train.py` — add new `ModelArguments` fields
3. `moellava/model/language_model/llava_qwen_moe.py` — pass new args to gate
4. `moellava/model/language_model/llava_phi_moe.py` — pass new args to gate
5. `moellava/model/language_model/llava_stablelm_moe.py` — pass new args to gate
6. Shell scripts — add new arguments

Read all files fully before making any change. The gate initialization pattern is
identical across all three model files so changes 3–5 follow the same pattern.

---

## The three failure modes and their loss terms

There are exactly three independent ways MoE routing can fail. Each needs its own term.

```
Failure 1 — Leakage (per-token):
    The router assigns probability to experts it will never call.
    Example: [0.35, 0.30, 0.20, 0.15] with top-2 selecting E1, E2.
    35% of probability mass sits on E3 and E4 which are ignored.
    → L_leak penalizes this directly.

Failure 2 — Within-k collapse (per-token):
    The router selects k experts but only trusts one of them.
    Example: top-2 selects E1 and E2, but split is [0.97, 0.03].
    E2 is selected but contributes almost nothing. Effectively top-1.
    → L_imbal penalizes this using KL divergence to uniform-over-k.

Failure 3 — Expert starvation (batch-level):
    Across the whole batch, some experts are never selected.
    Example: every token routes to E1 and E2. E3 and E4 receive zero
    gradient from L_lm and slowly die. L_leak and L_imbal cannot see
    this — they only look inside each token's top-k decision.
    → L_var penalizes unequal mean utilization across all E experts.
```

**Why L_leak and L_imbal cannot fix Failure 3:**
`torch.topk` returns values sorted by magnitude — dim=1 is rank position, not expert
identity. A batch where every token routes [0.51, 0.47, 0.01, 0.01] to E1 and E2
gives L_leak ≈ 0.02 and L_imbal ≈ 0.00 — both nearly zero. The loss is happy.
But E3 and E4 received zero tokens. They are starving. Only a batch-level term
over expert identities (not ranks) can detect this.

**Why L_var is better than the author's L_aux for this:**
L_aux = E · Σ f_i · m_i uses a non-differentiable hard selection count f_i as a
weighting signal. L_var directly minimizes the squared deviation of each expert's
mean probability from the uniform target 1/E. Its gradient at dead expert (m_i=0)
is -2/E² — a bounded constant, safe for bf16/DeepSpeed. L_aux and batch entropy
maximization both involve log(m_i) terms whose gradients explode as m_i → 0,
causing NaN in mixed precision training.

---

## Complete mathematical definition

Let:
- `p` = softmax over E experts, shape [T, E]
- `T_k(t)` = indices of top-k experts for token t
- `p_tilde` = p renormalized over T_k, shape [T, k]
- `m_i` = mean softmax probability for expert i across the batch = mean(p[:, i])

**Term 1 — Leakage (per-token):**
```
L_leak = E_t[ 1 - Σ_{i ∈ T_k} p_i(t) ]
```

**Term 2 — Imbalance (per-token, batch-averaged):**
```
L_imbal = E_t[ KL( p_tilde_{T_k} || u_k ) ]
        = E_t[ log(k) - H(p_tilde_{T_k}) ]
```
where u_k = uniform over k elements (each = 1/k).

**Term 3 — Variance balance (batch-level):**
```
L_var = (1/E) Σ_{i=1}^{E} ( m_i - 1/E )²
```

**Combined objective:**
```
L_total = L_lm  +  entropy_loss_weight · (L_leak + lam · L_imbal)
                +  balance_loss_weight  · L_var
```

**Key properties:**
- k=1: L_imbal = 0 identically. Loss reduces to L_leak only. Correct by construction.
- k=2: L_imbal target entropy is log(2) ≈ 0.693. [0.5, 0.5] inner routing costs 0.
- L_var gradient at m_i=0 is -2/E² — bounded constant, never NaN.
- L_var gradient at m_i=1/E is exactly 0 — zero pressure at target.
- All three terms are zero simultaneously only when routing is perfectly confident,
  balanced within each token's top-k, and equalized across all experts in the batch.

---

## Step-by-step changes

---

### CHANGE 1: Add two standalone loss functions

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Add both functions near the top of the file, after imports, before any class
definitions. Add them together as a block.

```python
def topk_entropy_loss(router_logits, k, lam=0.1, eps=1e-8):
    """
    K-adaptive per-token routing confidence loss.

        L_ent = L_leak + lam * L_imbal

    Addresses two per-token failure modes:

    L_leak (per-token):
        Penalizes probability mass outside the top-k selected experts.
        Enforces consistency between the continuous softmax and the
        discrete top-k routing decision.
        Gradient is linear and bounded at all values.

    L_imbal (per-token, batch-averaged):
        KL( p_tilde_{T_k} || u_k ) = log(k) - H(p_tilde_{T_k})
        Prevents within-k collapse: when k experts are selected, the
        router should actually use all k slots, not concentrate on 1.
        Applied per-token because torch.topk returns rank-ordered values
        (not expert identities), so batch-mean over dim=0 would average
        ranks — not what we want.

    Args:
        router_logits : [T, E]  raw logits before softmax (grad-connected)
        k             : int, number of selected experts
        lam           : float, relative weight of L_imbal vs L_leak.
                        Default 0.1 — gentle imbalance pressure.
                        Use 0.0 to disable L_imbal entirely (L_leak only).
        eps           : float, numerical stability constant

    Returns:
        Scalar loss tensor, grad-connected through router_logits.
    """
    probs     = F.softmax(router_logits, dim=-1)          # [T, E]
    topk_probs, _ = torch.topk(probs, k, dim=-1)          # [T, k]
    topk_mass = topk_probs.sum(dim=-1)                     # [T]

    # L_leak: per-token
    L_leak = (1.0 - topk_mass).mean()

    # L_imbal: per-token KL to uniform-over-k, then batch-averaged
    if k == 1:
        L_imbal = router_logits.new_zeros(())
    else:
        p_tilde = topk_probs / (topk_mass.unsqueeze(-1) + eps)       # [T, k]
        H_topk  = -(p_tilde * torch.log(p_tilde + eps)).sum(dim=-1)  # [T]
        log_k   = torch.log(torch.tensor(float(k), device=router_logits.device))
        L_imbal = (log_k - H_topk).clamp(min=0.0).mean()

    return L_leak + lam * L_imbal


def variance_balance_loss(router_logits):
    """
    Batch-level expert utilization balance via variance minimization.

    Addresses the batch-level failure mode (expert starvation) that
    L_leak and L_imbal cannot detect:
        Even when every token's per-token routing is perfect (L_ent ≈ 0),
        all tokens could route exclusively to E1 and E2, leaving E3 and E4
        with zero gradient from L_lm and causing them to atrophy.

    This term penalizes the squared deviation of each expert's mean
    routing probability from the uniform target 1/E.

    Gradient w.r.t. m_i = (2/E)(m_i - 1/E).

    NUMERICAL SAFETY — why variance beats alternatives:
        Batch entropy maximization (Σ m_i log m_i) has gradient log(m_i) + 1,
        which diverges to -∞ as m_i → 0 (dead expert), causing NaN in bf16.
        KL-based alternatives share the same log singularity problem.
        Variance gradient at m_i=0 is exactly -2/E² — a bounded constant.
        A dead expert receives a gentle, stable rescue signal regardless of
        how close to zero its probability is. Safe for DeepSpeed + bf16.

    Args:
        router_logits : [T, E]  raw logits before softmax (grad-connected)

    Returns:
        Scalar loss tensor, grad-connected through router_logits.
    """
    E     = router_logits.shape[-1]
    probs = F.softmax(router_logits, dim=-1)    # [T, E]
    m     = probs.mean(dim=0)                    # [E] mean prob per expert
    target = 1.0 / E                             # uniform target
    return ((m - target) ** 2).mean()
```

---

### CHANGE 2: Update SimplifiedNormalizedGate.__init__

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Find `SimplifiedNormalizedGate.__init__`. It currently accepts `entropy_loss_weight`.
Add two new parameters: `imbal_lam` and `balance_loss_weight`.

```python
def __init__(self,
             model_dim,
             num_experts,
             k=1,
             aux_loss_weight=0.01,
             entropy_loss_weight=0.0,
             imbal_lam=0.1,            # NEW: weight of L_imbal inside L_ent
             balance_loss_weight=0.0,  # NEW: weight of L_var (batch-level)
             **kwargs):
    super().__init__(model_dim, num_experts, k, **kwargs)
    self.aux_loss_weight      = aux_loss_weight
    self.entropy_loss_weight  = entropy_loss_weight
    self.imbal_lam            = imbal_lam
    self.balance_loss_weight  = balance_loss_weight

    # Diagnostic logging attributes
    self.last_moe_loss        = 0.0
    self.last_entropy_loss    = 0.0
    self.last_leak_loss       = 0.0
    self.last_imbal_loss      = 0.0
    self.last_balance_loss    = 0.0   # NEW
```

---

### CHANGE 3: Update SimplifiedNormalizedGate.forward()

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Find the entropy block in `forward()`. Replace the entire entropy section with
the block below. This handles all three loss terms together.

**Find this (the old entropy block):**
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

**Replace with:**
```python
# ── Per-token routing confidence loss (L_leak + lam * L_imbal) ──────────
if self.entropy_loss_weight > 0.0:
    k = self.k  # DeepSpeed stores top-k as self.k, NOT self.top_k
    L_ent = topk_entropy_loss(logits, k=k, lam=self.imbal_lam)
    entropy_loss = self.entropy_loss_weight * L_ent
    self.last_entropy_loss = L_ent.item()
    gate_output = (gate_output[0] + entropy_loss,) + gate_output[1:]

    # Diagnostic logging (no_grad — monitoring only, no effect on training)
    with torch.no_grad():
        probs_ng       = F.softmax(logits, dim=-1)
        topk_p, _      = torch.topk(probs_ng, k, dim=-1)
        topk_mass_     = topk_p.sum(dim=-1)
        self.last_leak_loss = (1.0 - topk_mass_).mean().item()
        if k > 1:
            p_t        = topk_p / (topk_mass_.unsqueeze(-1) + 1e-8)
            H_         = -(p_t * torch.log(p_t + 1e-8)).sum(dim=-1)
            log_k_     = torch.log(torch.tensor(float(k)))
            self.last_imbal_loss = (log_k_ - H_).clamp(min=0.0).mean().item()
        else:
            self.last_imbal_loss = 0.0
else:
    self.last_entropy_loss = 0.0
    self.last_leak_loss    = 0.0
    self.last_imbal_loss   = 0.0

# ── Batch-level expert utilization balance (L_var) ───────────────────────
if self.balance_loss_weight > 0.0:
    L_var = variance_balance_loss(logits)
    balance_loss = self.balance_loss_weight * L_var
    self.last_balance_loss = L_var.item()
    gate_output = (gate_output[0] + balance_loss,) + gate_output[1:]
else:
    self.last_balance_loss = 0.0
```

**Critical note on `self.k`:**
DeepSpeed's `TopKGate` stores the number of selected experts as `self.k`, not
`self.top_k`. This was confirmed by running the verification tests — using
`self.top_k` raises AttributeError. Always use `self.k`.

---

### CHANGE 4: Update update_hyperparameters method

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Find `SimplifiedNormalizedGate.update_hyperparameters`. Add the two new parameters
so they can be updated dynamically by callbacks if needed.

```python
def update_hyperparameters(self, temperature=None, kd_loss_weight=None,
                            ema_decay=None, entropy_loss_weight=None,
                            imbal_lam=None, balance_loss_weight=None):
    if entropy_loss_weight is not None:
        self.entropy_loss_weight = entropy_loss_weight
    if imbal_lam is not None:
        self.imbal_lam = imbal_lam
    if balance_loss_weight is not None:
        self.balance_loss_weight = balance_loss_weight
```

---

### CHANGE 5: Update get_loss_dict / get_diagnostics

**File:** `moellava/model/language_model/normalized_router_flexible.py`

Find the method that returns diagnostic values (likely `get_loss_dict` or
`get_diagnostics`). Add the three new logged values:

```python
def get_loss_dict(self):
    return {
        'moe_loss':       self.last_moe_loss,
        'entropy_loss':   self.last_entropy_loss,
        'leak_loss':      self.last_leak_loss,      # component of entropy_loss
        'imbal_loss':     self.last_imbal_loss,     # component of entropy_loss
        'balance_loss':   self.last_balance_loss,   # NEW: L_var
        'total_aux_loss': (self.aux_loss_weight   * self.last_moe_loss
                         + self.entropy_loss_weight * self.last_entropy_loss
                         + self.balance_loss_weight * self.last_balance_loss),
    }
```

---

### CHANGE 6: Add new fields to ModelArguments

**File:** `moellava/train/train.py`

Find `ModelArguments` (the dataclass with `entropy_loss_weight`). Add two new fields
immediately after `entropy_loss_weight`:

```python
@dataclass
class ModelArguments:
    # ... existing fields ...
    entropy_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for L_leak + lam*L_imbal (per-token routing loss)"}
    )
    imbal_lam: float = field(              # NEW
        default=0.1,
        metadata={"help": "Weight of L_imbal inside L_ent. 0.0 = L_leak only. "
                           "0.1 recommended. 1.0 hurts POPE (focused grounding)."}
    )
    balance_loss_weight: float = field(    # NEW
        default=0.0,
        metadata={"help": "Weight for L_var (batch-level expert utilization balance). "
                           "Replaces L_aux as original load balancing formulation. "
                           "Recommended: 0.01. Gradient bounded at all values (safe for bf16)."}
    )
```

---

### CHANGE 7: Pass new arguments to gate initialization

**Files:**
- `moellava/model/language_model/llava_qwen_moe.py`
- `moellava/model/language_model/llava_phi_moe.py`
- `moellava/model/language_model/llava_stablelm_moe.py`

In each file, find where `SimplifiedNormalizedGate` is instantiated. It currently
passes `entropy_loss_weight`. Add `imbal_lam` and `balance_loss_weight`:

```python
kd_gate = SimplifiedNormalizedGate(
    model_dim=hidden_size,
    num_experts=num_experts,
    k=model_args.top_k_experts,
    aux_loss_weight=model_args.router_aux_loss_coef,
    entropy_loss_weight=getattr(model_args, 'entropy_loss_weight', 0.0),
    imbal_lam=getattr(model_args, 'imbal_lam', 0.1),                    # NEW
    balance_loss_weight=getattr(model_args, 'balance_loss_weight', 0.0), # NEW
)
```

Use `getattr(..., default)` for all three — this ensures backward compatibility
if the model is loaded from a checkpoint trained without these arguments.

---

### CHANGE 8: Update training shell scripts

For each entropy training script, update the argument list to include the new
parameters. The recommended starting configuration is:

```bash
# Per-token routing confidence loss
--entropy_loss_weight 0.03 \
--imbal_lam 0.1 \           # gentle imbalance pressure — recovers POPE regression

# Batch-level expert utilization balance (replaces L_aux)
--router_aux_loss_coef 0.0 \   # keep at 0 — L_var replaces this
--balance_loss_weight 0.01 \   # L_var: gentle, bounded gradients, safe for bf16
```

Create one new script per backbone for the full combined loss:
- `scripts/v1/qwen/finetune_moe_full_loss.sh`
- `scripts/v1/phi/finetune_moe_full_loss.sh`
- `scripts/v1/stablelm/finetune_moe_full_loss.sh`

Add this comment block at the top of each new script:

```bash
# Full routing loss: three orthogonal failure modes, three terms.
#
# L_leak  (entropy_loss_weight * L_leak):
#     Per-token. Penalizes probability mass on non-selected experts.
#
# L_imbal (entropy_loss_weight * imbal_lam * L_imbal):
#     Per-token. Prevents within-k collapse (one expert dominates top-k).
#     Use lam=0.1 not 1.0 — high lam hurts POPE (focused grounding tasks).
#
# L_var   (balance_loss_weight * L_var):
#     Batch-level. Penalizes expert starvation via variance minimization.
#     Gradient bounded at all m_i values — safe for bf16/DeepSpeed.
#     Replaces L_aux with an original, more principled formulation.
```

---

### CHANGE 9: Update EntropyWarmupCallback (minor)

**File:** `moellava/train/router_callback.py`

The existing `EntropyWarmupCallback` ramps `entropy_loss_weight` from 0 to target
over the first 10% of training. This still works correctly for `entropy_loss_weight`.

Add one small update: also ramp `balance_loss_weight` if it is set, using the same
warmup schedule. This prevents L_var from firing before the model has established
basic routing patterns.

Find `on_step_begin` in `EntropyWarmupCallback`. After the existing entropy weight
update, add:

```python
# Ramp balance_loss_weight with same schedule if configured
for gate in self.gate_cache:
    if hasattr(gate, 'balance_loss_weight') and gate.balance_loss_weight > 0.0:
        target_balance = self._target_balance_weights[id(gate)]
        gate.update_hyperparameters(
            balance_loss_weight=fraction * target_balance
        )
```

In `on_train_begin`, cache the target balance weights alongside the entropy weights:

```python
self._target_balance_weights = {
    id(g): g.balance_loss_weight for g in self.gate_cache
}
# Reset to 0 for warmup
for g in self.gate_cache:
    g.update_hyperparameters(balance_loss_weight=0.0)
```

If this is too complex to integrate cleanly with the existing callback, it is
acceptable to skip the balance warmup for now and let L_var fire from step 1.
L_var's gradient is bounded, so it will not cause instability even without warmup.

---

## Verification checklist

Run all tests before starting any training job.

```python
import torch
import torch.nn.functional as F

# ── paste topk_entropy_loss and variance_balance_loss here ──

print("=== topk_entropy_loss tests ===")

# Test 1: ideal top-2 → loss ≈ 0
logits = torch.tensor([[2.0, 2.0, -10.0, -10.0]])
loss = topk_entropy_loss(logits, k=2, lam=0.1)
print(f"T1 ideal top-2:         {loss:.6f}  (expect ~0.000)")

# Test 2: one-hot top-2 → L_imbal fires, L_leak ≈ 0
logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
loss = topk_entropy_loss(logits, k=2, lam=1.0)
print(f"T2 one-hot k=2 lam=1:   {loss:.6f}  (expect ~0.693)")

# Test 3: uniform → L_leak fires, L_imbal ≈ 0
logits = torch.zeros(1, 4)
loss = topk_entropy_loss(logits, k=2, lam=0.1)
print(f"T3 uniform k=2:         {loss:.6f}  (expect ~0.500)")

# Test 4: k=1 one-hot → loss ≈ 0
logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
loss = topk_entropy_loss(logits, k=1, lam=0.1)
print(f"T4 one-hot k=1:         {loss:.6f}  (expect ~0.000)")

# Test 5: k=1 L_imbal ≡ 0 (check against pure leakage)
logits = torch.randn(8, 4)
loss_k1 = topk_entropy_loss(logits, k=1, lam=0.1)
probs = F.softmax(logits, dim=-1)
leak_only = (1.0 - probs.max(dim=-1).values).mean()
print(f"T5 k=1 imbal term:      {abs(loss_k1.item() - leak_only.item()):.8f}  (expect 0.0)")

# Test 6: lam=0 → pure leakage only
logits = torch.randn(8, 4)
loss_lam0 = topk_entropy_loss(logits, k=2, lam=0.0)
probs = F.softmax(logits, dim=-1)
topk_p, _ = torch.topk(probs, 2, dim=-1)
leak_manual = (1.0 - topk_p.sum(dim=-1)).mean()
print(f"T6 lam=0 = pure leak:   {abs(loss_lam0.item() - leak_manual.item()):.8f}  (expect 0.0)")

print("\n=== variance_balance_loss tests ===")

# Test 7: uniform routing → L_var ≈ 0
logits = torch.zeros(100, 4)
loss = variance_balance_loss(logits)
print(f"T7 uniform routing:     {loss:.6f}  (expect ~0.000)")

# Test 8: collapsed routing (all to E1) → L_var large
logits = torch.zeros(100, 4)
logits[:, 0] = 10.0
loss = variance_balance_loss(logits)
print(f"T8 collapsed to E1:     {loss:.6f}  (expect >0.1)")

# Test 9: gradient is bounded at dead expert (m_i → 0)
logits = torch.zeros(10, 4, requires_grad=True)
with torch.no_grad():
    logits_val = torch.zeros(10, 4)
    logits_val[:, 0] = 100.0   # E1 gets all probability, E2/E3/E4 ~ dead
logits = logits_val.clone().requires_grad_(True)
loss = variance_balance_loss(logits)
loss.backward()
max_grad = logits.grad.abs().max().item()
print(f"T9 max gradient (dead experts): {max_grad:.6f}  (expect < 1.0, not NaN)")
assert not torch.isnan(logits.grad).any(), "FAIL: NaN gradient detected"
print("T9 NaN check: PASSED")

# Test 10: gradient direction (dead expert should receive positive gradient)
# m_i < 1/E → gradient should be negative (pulling m_i up toward 1/E)
# equivalently, logit gradient pushes toward higher probability for E2/E3/E4
grad_dead = logits.grad[0, 1].item()  # E2 is dead
print(f"T10 gradient for dead E2: {grad_dead:.6f}  (expect != 0, should rescue E2)")
```

**Expected outputs:**
```
T1  ideal top-2:         ~0.000000   both terms satisfied
T2  one-hot k=2 lam=1:   ~0.693147   = log(2), L_imbal fires
T3  uniform k=2:         ~0.500000   50% leakage from E3+E4
T4  one-hot k=1:         ~0.000000   correct for k=1
T5  k=1 imbal term:       0.00000000  exactly zero
T6  lam=0 = pure leak:    0.00000000  exactly zero
T7  uniform routing:     ~0.000000   no imbalance
T8  collapsed to E1:      >0.100000   imbalance detected
T9  max gradient:         <1.0, no NaN  bounded and safe
T10 gradient for dead E2: nonzero    rescue signal present
```

---

## Ablation recommendations

Run these experiments in order to understand each term's contribution:

| Experiment | `entropy_loss_weight` | `imbal_lam` | `balance_loss_weight` | What it tests |
|---|---|---|---|---|
| Baseline (student) | 0.0 | — | 0.0 | No routing loss |
| L_leak only | 0.03 | 0.0 | 0.0 | Pure leakage penalty |
| L_leak + L_imbal (gentle) | 0.03 | 0.1 | 0.0 | Per-token loss, gentle imbalance |
| L_leak + L_imbal (strong) | 0.03 | 1.0 | 0.0 | Replicates failed experiment |
| L_var only | 0.0 | — | 0.01 | Batch-level only |
| Full loss | 0.03 | 0.1 | 0.01 | All three terms |

The "L_imbal strong" row should reproduce the POPE regression seen in the failed
experiment. This confirms L_imbal at lam=1.0 is the cause and validates the fix.

---

## Summary of all changes

| What | v1 | v2 | Why |
|---|---|---|---|
| Loss terms | L_leak + L_imbal | L_leak + lam·L_imbal + L_var | Expert starvation not addressed by per-token terms |
| lam (imbal weight) | Hardcoded 1.0 | Configurable, default 0.1 | lam=1.0 hurts POPE; 0.1 recovers it |
| Batch-level balance | None (L_aux=0) | L_var (variance minimization) | Original formulation, bounded gradients, bf16-safe |
| L_aux | Zeroed out | Stays zeroed out | L_var replaces it with original design |
| New training args | — | `imbal_lam`, `balance_loss_weight` | Independent control of each failure mode |
| Gate init | entropy_loss_weight only | + imbal_lam, balance_loss_weight | Pass new args from model files |
| DeepSpeed k attribute | Noted as self.k | Confirmed as self.k | Verified by passing tests |
| Gradient safety | Not analyzed | L_var bounded at m_i=0 | Prevents NaN in bf16/DeepSpeed |
| Verification tests | 5 tests | 10 tests | Covers both functions and gradient safety |