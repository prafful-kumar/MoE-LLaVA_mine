# Experiment TODO & Checkpoint Status

**IMPORTANT: Refer to this file before running any experiment to avoid overwriting checkpoints.**
**See also: `CHECKPOINT_STATUS.md` for checkpoint inventory details.**

---

## Lost Checkpoints

These checkpoints were overwritten because new training used the same output_dir names:

| Lost Checkpoint | Loss Type | Weight | Overwritten By | Date Lost |
|---|---|---|---|---|
| `checkpoints_phi_entropy/` | Old broken raw H | w=0.01 | topk_entropy (w=0.03) | 2026-03-26 |
| `checkpoints_qwen_entropy/` | Old broken raw H | w=0.01 | topk_entropy (w=0.03) | 2026-03-27 |
| `checkpoints_stablelm_entropy/` | Old broken raw H | w=0.01 | topk_entropy (w=0.03) | 2026-03-30 |

**Note:** Old eval results preserved in `benchmark_results.xlsx` under "Phi2/Qwen/StableLM Entropy (w=0.01)". Only checkpoints are lost.

---

## Completed Experiments

### Phi2 (2.7B) — 4 GPUs, batch=1, grad_accum=12

| Experiment | HPC Directory | Router Init | Loss | Key Hyperparams | Eval Done? |
|---|---|---|---|---|---|
| Phi2 Author | `checkpoints_phi/llavaphi-2.7b-finetune-moe` | teacher_kd (author code) | aux=0.0 | Trained with original MoE-LLaVA codebase | Yes (5/5) — POPE rerun pending |
| Phi2 Student | `checkpoints_phi_student/llavaphi-2.7b-finetune-moe` | no_teacher | aux=0.0 | centroids=fisher_directions_phi/5000.pkl | Yes (5/5) — POPE rerun pending |
| Phi2 TS | `checkpoints_phi_TS/llavaphi-2.7b-finetune-moe` | teacher_kd | KD + aux=0.0 | centroids=fisher_directions_phi/5000.pkl | Yes (5/5) — POPE rerun pending |
| Phi2 Entropy-topk | `checkpoints_phi_entropy/llavaphi-2.7b-finetune-moe` | no_teacher | topk_entropy w=0.03, aux=0.0 | centroids=fisher_directions_phi/5000.pkl | Yes (5/5) — POPE rerun pending |
| Phi2 Adaptive Entropy | `checkpoints_phi_adaptive_entropy/llavaphi-2.7b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, aux=0.0 | centroids=fisher_directions_phi/5000.pkl ⚠️ retrain with 20k | 🔄 **Training GPU 0,1,2,3 (started 2026-04-20)** |
| Phi2 Adaptive Entropy + L_var | `checkpoints_phi_adaptive_entropy_var/llavaphi-2.7b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0 | centroids=fisher_directions_phi/5000.pkl | ❌ Not started |
| Phi2 Old Entropy | **LOST** | no_teacher | old broken H w=0.01, aux=0.0 | centroids=fisher_directions_phi/5000.pkl | Yes (results in xlsx) |

### Qwen (1.8B) — 3 GPUs, batch=2, grad_accum=12

| Experiment | HPC Directory | Router Init | Loss | Key Hyperparams | Eval Done? |
|---|---|---|---|---|---|
| Qwen Author | `checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe` | teacher_kd (author code) | aux=0.0 | Trained with original MoE-LLaVA codebase | Partial — POPE truncated (1960/9000 lines), GQA missing (only step-wise). POPE rerun pending |
| Qwen Student | `checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe` | student_warm | aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | Yes (5/5) — POPE rerun pending |
| Qwen TS | `checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe` | teacher_kd | KD + aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | Yes (5/5) — POPE rerun pending |
| Qwen TS Schedule | `checkpoints_qwen_TS_schedule/llavaqwen-1.8b-finetune-moe` | teacher_kd | KD + aux=0.0 (scheduled hypers) | centroids=fisher_directions_qwen/5000.pkl | **Eval RUNNING GPU 7 (pos 1/5)** |
| Qwen Entropy-topk | `checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe` | no_teacher | topk_entropy w=0.03, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | Yes (5/5) — POPE rerun pending |
| Qwen Entropy-topk-var | `checkpoints_qwen_entropy_topk_var/llavaqwen-1.8b-finetune-moe` | no_teacher | topk_entropy w=0.03, lam=0.1, w_bal=0.01, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | **Eval RUNNING GPU 7 (pos 2/5)** |
| Qwen Adaptive Entropy | `checkpoints_qwen_adaptive_entropy/llavaqwen-1.8b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl ⚠️ retrain with 20k | **✅ Done (2026-04-20). Eval pending.** |
| Qwen Adaptive Entropy + L_var | `checkpoints_qwen_adaptive_entropy_var/llavaqwen-1.8b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | ❌ Not started |
| Qwen Old Entropy w=0.01 | **LOST** | no_teacher | old broken H w=0.01, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | Yes (results in xlsx) |
| Qwen Old Entropy w=0.1 | `checkpoints_qwen_entropy_w01/llavaqwen-1.8b-finetune-moe` | no_teacher | old broken H w=0.1, aux=0.0 | centroids=fisher_directions_qwen/5000.pkl | Yes (5/5) |

### StableLM (1.6B) — 3 GPUs, batch=2, grad_accum=12

| Experiment | HPC Directory | Router Init | Loss | Key Hyperparams | Eval Done? |
|---|---|---|---|---|---|
| StableLM Author | `random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe` | random | aux=0.01 | No centroids (random init), no KD | Yes (5/5) — POPE rerun pending |
| StableLM Student | `checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe` | student_warm | aux=0.0 | centroids=fisher_directions/5000.pkl | Yes (5/5) — POPE rerun pending |
| StableLM TS | `checkpoints_stablelm_TS/llava-stablelm-1.6b-finetune-moe` | teacher_kd | KD + aux=0.0 | centroids=fisher_directions/5000.pkl | Yes (5/5) — POPE rerun pending |
| StableLM Entropy-topk | `checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe` | no_teacher | topk_entropy w=0.03, aux=0.0 | centroids=fisher_directions/5000.pkl | Yes (5/5) — POPE rerun pending |
| StableLM Entropy-topk-aux | `checkpoints_stablelm_entropy_topk_aux/llava-stablelm-1.6b-finetune-moe` | no_teacher | topk_entropy w=0.03 + aux=0.01 | centroids=fisher_directions/5000.pkl | **Eval RUNNING GPU 7 (pos 3/5)** |
| StableLM Entropy-topk-var | `checkpoints_stablelm_entropy_topk_var/llava-stablelm-1.6b-finetune-moe` | no_teacher | topk_entropy w=0.03, lam=0.1, w_bal=0.01, aux=0.0 | centroids=fisher_directions/5000.pkl | **Eval RUNNING GPU 7 (pos 4/5)** |
| StableLM Adaptive Entropy | `checkpoints_stablelm_adaptive_entropy/llava-stablelm-1.6b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, aux=0.0 | centroids=fisher_directions/5000.pkl ⚠️ retrain with 20k | **✅ Done (2026-04-18). Eval pending.** |
| StableLM Adaptive Entropy + L_var | `checkpoints_stablelm_adaptive_entropy_var/llava-stablelm-1.6b-finetune-moe` | no_teacher | adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0 | centroids=fisher_directions/5000.pkl | ❌ Not started |
| StableLM Old Entropy | **LOST** | no_teacher | old broken H w=0.01, aux=0.0 | centroids=fisher_directions/5000.pkl | Yes (results in xlsx) |

### Common Hyperparameters (all experiments unless noted)
- `num_experts=4`, `top_k_experts=2`, `moe_mode=sparse`, `capacity_factor=1.5`
- `lr=2e-5`, `warmup_ratio=0.03`, `lr_scheduler=cosine`, `num_epochs=1`
- `model_max_length=2048`, `bf16=True`, `deepspeed=zero2.json`
- `train_modules`: Phi2=`fc1 fc2 wg`, Qwen=`mlp.w1 mlp.w2 mlp.c_proj wg`, StableLM=`gate_proj up_proj down_proj wg`

---

## Pending TODO

### Evaluation — In Progress (2026-04-17)

**GPU 6** (PID 724795, log: `logs/eval/gpu6_eval.log`):
- [ ] **Qwen ablations** — all 10 checkpoints × 5 benchmarks. Script: `eval_gpu6.sh` → `qwen_ablations_eval.sh`. Per-run logs: `logs/eval/qwen_ablations/{name}.log`. Variant prefix: `qwen_abl_*`. ~20h total.

**GPU 7** (PID 724796, log: `logs/eval/gpu7_eval.log`) — sequential queue:
- [ ] **[1/5] Qwen TS Schedule** — 5 benchmarks. Script: `qwen_TS_schedule_all.sh 7`.
- [ ] **[2/5] Phi2 Entropy-topk-var** — 5 benchmarks. Script: `phi_entropy_topk_var_all.sh 7`.
- [ ] **[3/5] StableLM Entropy-topk-aux** — 5 benchmarks. Script: `stablelm_entropy_topk_aux_all.sh 7`.
- [ ] **[4/5] StableLM Entropy-topk-var** — 5 benchmarks. Script: `stablelm_entropy_topk_var_all.sh 7`.
- [ ] **[5/5] StableLM ablations** — all 10 checkpoints × 5 benchmarks. Script: `stablelm_ablations_eval.sh 7`. Will poll for training completion. ~20h after prior evals.

### Evaluation — Waiting
- [ ] **POPE rerun ALL variants** — 13 checkpoints. Script: `pope_rerun_all.sh`. **Status: STOPPED. Resume when GPUs 6 or 7 free up.**
- [ ] **Qwen Author GQA** — no final GQA eval exists (only step-wise). Run from `checkpoints_qwen_author/` on HPC.

### Training — Need to Retrain
- [ ] **Phi2 Entropy-topk (retrain)** — delete HPC checkpoint first, retrain with topk_entropy_loss w=0.03. Script: `scripts/v1/phi2/finetune_moe_entropy.sh`. Output dir: `checkpoints_phi_entropy/`
- [ ] **Qwen Entropy-topk (retrain)** — delete HPC checkpoint first, retrain with topk_entropy_loss w=0.03. Script: `scripts/v1/qwen/finetune_moe_entropy.sh`. Output dir: `checkpoints_qwen_entropy/`
- [ ] **StableLM Entropy-topk (retrain)** — delete HPC checkpoint first, retrain with topk_entropy_loss w=0.03, aux=0.0. Script: `scripts/v1/stablelm/finetune_moe_entropy.sh` (revert aux to 0.0 first). Output dir: `checkpoints_stablelm_entropy/`
- [ ] **StableLM Adaptive Entropy (retrain w/ 20k centroids)** — run `get_kmeans_centroids/compute_fisher_directions.py --num_samples 20000` first (default already updated), then retrain. New checkpoint dir: `checkpoints_stablelm_adaptive_entropy_20k/`. Script: update `router_centroids_path` in `finetune_moe_adaptive_entropy.sh` to `fisher_directions/20000.pkl`.
- [ ] **Qwen Adaptive Entropy (retrain w/ 20k centroids)** — run `get_kmeans_centroids/compute_fisher_directions_qwen.py --num_samples 20000` first (default already updated to 20000), then retrain. New checkpoint dir: `checkpoints_qwen_adaptive_entropy_20k/`. Script: update `router_centroids_path` in `finetune_moe_adaptive_entropy.sh` to `fisher_directions_qwen/20000.pkl`.

### Training — New Variants (scripts ready, not yet run)
- [x] **Qwen Entropy-topk-var** — topk_entropy_loss w=0.03, imbal_lam=0.1, w_bal=0.01, aux=0.0. Done 2026-04-02. **Eval: RUNNING GPU 7 (queue pos 2/5).**
- [x] **StableLM Entropy-topk + aux=0.01** — topk_entropy_loss w=0.03 + aux_loss_coef=0.01. Done. **Eval: RUNNING GPU 7 (queue pos 3/5).**
- [x] **StableLM Entropy-topk-var** — topk_entropy_loss w=0.03, imbal_lam=0.1, w_bal=0.01, aux=0.0. Done. **Eval: RUNNING GPU 7 (queue pos 4/5).**
- [x] **StableLM Adaptive Entropy** — adaptive_entropy w=0.03, gamma=2.0, aux=0.0. Done 2026-04-18. Script: `scripts/v1/stablelm/finetune_moe_adaptive_entropy.sh`. Output dir: `checkpoints_stablelm_adaptive_entropy/`. ⚠️ Used 5000-pt centroids — retrain with 20000-pt when compute budget allows.
- ✅ **Qwen Adaptive Entropy** — adaptive_entropy w=0.03, gamma=2.0, aux=0.0. Done 2026-04-20. Script: `scripts/v1/qwen/finetune_moe_adaptive_entropy.sh`. Output dir: `checkpoints_qwen_adaptive_entropy/`. ⚠️ Used 5000-pt centroids.
- 🔄 **Phi2 Adaptive Entropy** — adaptive_entropy w=0.03, gamma=2.0, aux=0.0. **Training GPU 0,1,2,3 (started 2026-04-20).** Script: `scripts/v1/phi2/finetune_moe_adaptive_entropy.sh`. Output dir: `checkpoints_phi_adaptive_entropy/`. ⚠️ Used 5000-pt centroids.
- [ ] **Phi2 Adaptive Entropy + L_var** — adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0. Script: TBD. Output dir: `checkpoints_phi_adaptive_entropy_var/`.
- [ ] **Qwen Adaptive Entropy + L_var** — adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0. Script: TBD. Output dir: `checkpoints_qwen_adaptive_entropy_var/`.
- [ ] **StableLM Adaptive Entropy + L_var** — adaptive_entropy w=0.03, gamma=2.0, w_bal=0.01, aux=0.0. Script: TBD. Output dir: `checkpoints_stablelm_adaptive_entropy_var/`.

### Training — Never Trained (no checkpoint exists)
- [ ] **Phi2 no_teacher** — Qwen and StableLM have `no_teacher` (SimplifiedNormalizedGate, no entropy, no KD) but Phi2 does not. `checkpoints_phi_student/` was trained with `student_warm` (vanilla DeepSpeed TopKGate + centroid warm-start) — a different gate class. Need new script with `ROUTER_INIT_MODE="no_teacher"`, `entropy_loss_weight=0.0`. Output dir: `checkpoints_phi_no_teacher/` **CANCELLED 2026-04-06 at 10% — stopped to run ablation experiments instead. Checkpoint deleted.**
- [ ] **StableLM TS Schedule** — script exists (`scripts/v1/stablelm/finetune_moe_TS_schedule.sh`) but never trained
- [ ] **Phi2 TS Schedule** — no script exists yet
- [ ] **Qwen Entropy w=0.1 warmup** — script exists (`scripts/v1/qwen/finetune_moe_entropy_w01_warmup.sh`) but never trained

### Post-Training
- [ ] Eval all retrained entropy checkpoints on 5 benchmarks after training completes
- [ ] Transfer retrained checkpoints to HPC with `rsync -av --no-g`
- [ ] Update `benchmark_results.xlsx` with new results

---

## Power-Law Alpha Variants (Future)

**Idea:** Replace the exponential confidence gate in `use_adaptive_entropy=True` with a power-law form:

| Formula | Current (exponential) | Proposed (power-law) |
|---|---|---|
| alpha | `exp(-gamma * margin)` | `(1 - margin)^gamma` |
| alpha at margin=0 | 1.0 | 1.0 |
| alpha at margin=1 | `exp(-gamma)` > 0 | 0.0 (hard zero) |
| Behavior | Soft decay, never reaches 0 | Hard zero at full confidence; focal-loss style |

**Why power-law is more aggressive:**
- At margin=1 (perfectly confident token), exponential gives `exp(-2) ≈ 0.135` — still applies small one-hot push.
- Power-law gives exactly 0 — completely removes pressure on confident tokens.
- For intermediate margins, `(1-m)^gamma` decays faster than `exp(-gamma*m)` for gamma≥2 near m=0.
- Analogy: focal loss `(1-p)^gamma` in object detection — same motivation (down-weight easy examples).

**Code change in `normalized_router_flexible.py`** (single line, `SimplifiedNormalizedGate.forward()`):
```python
# Current:
alpha = torch.exp(-self.adaptive_gamma * prob_margin).detach()
# Proposed:
alpha = (1.0 - prob_margin).pow(self.adaptive_gamma).detach()
```

**Experiments to run** (after double_adaptive variants are evaluated):

| Priority | Backbone | Checkpoint Dir | Params | Status |
|---|---|---|---|---|
| P1 | StableLM 1.6B | `checkpoints_stablelm_power_adaptive/` | `alpha_mode=power, w_ent=0.1, w_bal=0.1, gamma=2.0` | **🔄 Training GPUs 0,1,2 (started 2026-04-24)** |
| P2 | Qwen 1.8B | `checkpoints_qwen_power_adaptive/` | same | ❌ Not started |
| P3 | Phi2 2.7B | `checkpoints_phi_power_adaptive/` | same | ❌ Not started |

**Implementation notes:**
- The `adaptive_gamma` parameter reuses for both formulas; no new arg needed.
- Consider adding a new `--alpha_mode` flag (`exp` vs `power`) to `train.py` `RouterArguments` and `SimplifiedNormalizedGate.__init__` so both variants can be run without code change.
- Gate the formula with `if self.alpha_mode == 'power': alpha = (1-m).pow(gamma) else: alpha = exp(-gamma*m)`.
- All other loss terms (L_leak, L_adaptive two-sided, L_var) remain unchanged.

---

## Hyperparameter Ablation (topk_entropy_loss)

**Loss formula:** `w_ent * (L_leak + lambda * L_imbal) + w_bal * L_var`

**Script args:** `--entropy_loss_weight` (w_ent), `--imbal_lam` (lambda), `--balance_loss_weight` (w_bal)

**Baseline (already done):** `qwen_entropy_topk_var` — w_ent=0.03, lambda=0.1, w_bal=0.01

**Backbone:** Qwen 1.8B (fastest). All checkpoints under `checkpoints_qwen_ablations/{short_name}/llavaqwen-1.8b-finetune-moe`

**Storage policy:** Use `--save_strategy "epoch"` (saves only final checkpoint; no intermediate saves needed for ablation runs). Only retrain with intermediate checkpoints for the best-performing variant.

### Ablation grid (10 experiments, vary one param at a time; lam/wbal each run at both w_ent values)

w_ent ∈ {0.01, 0.1} only (0.03 never used). lam/wbal ablations run at both w_ent values.

| Short Name | w_ent | lambda | w_bal | Notes | Train Status | Eval Status |
|---|---|---|---|---|---|---|
| `went_001` | **0.01** | 0.1 | 0.01 | w_ent low | ✅ Done (HPC) | Eval running GPU 6 |
| `went_01` | **0.1** | 0.1 | 0.01 | w_ent high | ✅ Done (HPC) | Eval running GPU 6 |
| `lam_001_w001` | 0.01 | **0.01** | 0.01 | lambda low, w_ent low | ✅ Done (HPC) | Eval running GPU 6 |
| `lam_001_w01` | 0.1 | **0.01** | 0.01 | lambda low, w_ent high | ✅ Done (HPC) | Eval running GPU 6 |
| `lam_1_w001` | 0.01 | **1.0** | 0.01 | lambda high, w_ent low | ✅ Done (HPC) | Eval running GPU 6 |
| `lam_1_w01` | 0.1 | **1.0** | 0.01 | lambda high, w_ent high | ✅ Done (HPC) | Eval running GPU 6 |
| `wbal_0_w001` | 0.01 | 0.1 | **0.0** | ablate L_var, w_ent low | ✅ Done (HPC) | Eval running GPU 6 |
| `wbal_0_w01` | 0.1 | 0.1 | **0.0** | ablate L_var, w_ent high | ✅ Done (HPC) | Eval running GPU 6 |
| `wbal_01_w001` | 0.01 | 0.1 | **0.1** | w_bal high, w_ent low | ✅ Done (HPC) | Eval running GPU 6 |
| `wbal_01_w01` | 0.1 | 0.1 | **0.1** | w_bal high, w_ent high | ✅ Done (HPC) | Eval running GPU 6 |

Training scripts: GPUs 5,6,7 (`run_all.sh`) + GPUs 2,3,4 (`run_wbal_234.sh`). All 10 transferred to HPC 2026-04-17, local copies deleted.
Eval script: `eval_gpu6.sh` → `qwen_ablations_eval.sh`. Variant prefix: `qwen_abl_*`.

### Phi2 Hyperparameter Ablation (topk_entropy_loss)

**Backbone:** Phi2 2.7B. All checkpoints under `checkpoints_phi_ablations/{short_name}/llavaphi-2.7b-finetune-moe`

**Storage policy:** `--save_strategy "epoch"` (final checkpoint only).

**Motivation:** Based on Qwen and StableLM ablation results:
- StableLM best: `lam_1_w01` (w_ent=0.1, λ=1.0, w_bal=0.01) — MME 1644.5 (+41 over baseline)
- Qwen best: `wbal_01_w01` (w_ent=0.1, λ=0.1, w_bal=0.1) — MME 1583.2
- Common signal: w_ent=0.1 consistently beats w_ent=0.01 across both backbones

| Priority | Short Name | w_ent | lambda | w_bal | Notes | Train Status | Eval Status |
|---|---|---|---|---|---|---|---|
| P1 | `lam_1_w01` | 0.1 | **1.0** | 0.01 | StableLM winner — highest value single run | ❌ Not started | ❌ |
| P2 | `wbal_01_w01` | 0.1 | 0.1 | **0.1** | Qwen winner — cross-backbone comparison | ❌ Not started | ❌ |
| P3 | `lam_1_wbal_01` | 0.1 | **1.0** | **0.1** | High λ + high w_bal — what if Phi2 wants both? | ❌ Not started | ❌ |

Script args:
- P1: `--entropy_loss_weight 0.1 --imbal_lam 1.0 --balance_loss_weight 0.01`
- P2: `--entropy_loss_weight 0.1 --imbal_lam 0.1 --balance_loss_weight 0.1`
- P3: `--entropy_loss_weight 0.1 --imbal_lam 1.0 --balance_loss_weight 0.1`

---

### StableLM Hyperparameter Ablation (topk_entropy_loss)

**Backbone:** StableLM 1.6B. All checkpoints under `checkpoints_stablelm_ablations/{short_name}/llava-stablelm-1.6b-finetune-moe`

| Short Name | w_ent | lambda | w_bal | Train Status | Eval Status |
|---|---|---|---|---|---|
| `went_001` | 0.01 | 0.1 | 0.01 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `went_01` | 0.1 | 0.1 | 0.01 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `lam_001_w001` | 0.01 | 0.01 | 0.01 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `lam_001_w01` | 0.1 | 0.01 | 0.01 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `lam_1_w001` | 0.01 | 1.0 | 0.01 | 🔄 Training GPU 0,1,2 (~4h) | Queued GPU 7 (pos 5/5, waits) |
| `lam_1_w01` | 0.1 | 1.0 | 0.01 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `wbal_0_w001` | 0.01 | 0.1 | 0.0 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `wbal_0_w01` | 0.1 | 0.1 | 0.0 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `wbal_01_w001` | 0.01 | 0.1 | 0.1 | ✅ Done (local) | Queued GPU 7 (pos 5/5) |
| `wbal_01_w01` | 0.1 | 0.1 | 0.1 | 🔄 Training GPU 3,4,5 (~4h) | Queued GPU 7 (pos 5/5, waits) |

Training scripts: GPUs 0,1,2 (`run_half1.sh`) + GPUs 3,4,5 (`run_half2.sh`). Started 2026-04-13.
Eval script: `eval_gpu7.sh` → `stablelm_ablations_eval.sh`. Variant prefix: `stablelm_abl_*`. Polls for training completion.

---

## Rules for Future Experiments

1. **ALWAYS check this file AND `CHECKPOINT_STATUS.md`** and `ls /home/prafull/scratch/hpc/` before launching training to verify no name collision
2. **ALWAYS use unique output_dir names** — never reuse an existing checkpoint directory name for a different experiment
3. **Naming convention:**
   - `checkpoints_{backbone}_{variant}/` — e.g., `checkpoints_qwen_entropy_topk_w01/`
   - Include loss type and weight in name if it differs from default
   - For ablations: `checkpoints_qwen_ablations/{short_name}/` (all under one parent folder)
4. **Transfer checkpoints** with `rsync -av --no-g` (avoids chgrp errors on SSHFS)
5. **Update both this file AND `CHECKPOINT_STATUS.md`** after completing any experiment
6. **save_steps=24000** for new training runs (avoid intermediate checkpoint bloat)
7. **Ablation runs:** use `--save_strategy "epoch"` to save only the final checkpoint; only keep intermediate checkpoints for the best-performing variant
