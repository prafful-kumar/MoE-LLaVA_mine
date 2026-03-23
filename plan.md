# Plan — MoE-LLaVA Experiments

## Experiment Schemes

We are comparing router initialization and regularization strategies for MoE Stage III training:

1. **Random** (`router_init_mode=random`) — original paper's approach (referred to as "author")
2. **Student-only** (`router_init_mode=no_teacher`) — K-means centroid init, no teacher KD
3. **Teacher-Student (TS)** (`router_init_mode=teacher_kd`) — K-means centroid init + teacher KD
4. **Entropy** (`router_init_mode=no_teacher`, `entropy_loss_weight>0`) — K-means init + entropy regularization to force router specialization

## Fixed Hyperparameters (all experiments)

| Parameter | Value |
|---|---|
| `initial_kd_weight` | 0.01 |
| `router_temp_start` | 1.0 |
| `router_ema_start` | 0.999 |
| `num_experts` | 4 |
| `top_k_experts` | 2 |
| `moe_mode` | sparse |

## Trained Models

### Qwen (1.8B) — 9240 steps per run

| Scheme | Checkpoint Dir | Location | Status |
|---|---|---|---|
| Random (author) | `checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/` | HPC | Complete |
| Student-only | `checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/` | HPC | Complete |
| Teacher-Student | `checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/` | HPC | Complete |
| Entropy (w=0.01) | `checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe/` | HPC | Complete |
| Entropy (w=0.1) | `checkpoints_qwen_entropy_w01/llavaqwen-1.8b-finetune-moe/` | HPC | Complete |

### Phi2 (2.7B) — 13860 steps per run

| Scheme | Checkpoint Dir | Location | Status |
|---|---|---|---|
| Random (author) | `checkpoints_phi/llavaphi-2.7b-finetune-moe/` | HPC | Complete |
| Student-only | `checkpoints_phi_student/llavaphi-2.7b-finetune-moe/` | HPC | Complete |
| Teacher-Student | `checkpoints_phi_TS/llavaphi-2.7b-finetune-moe/` | HPC | Complete |
| Entropy (w=0.01) | `checkpoints_phi_entropy/llavaphi-2.7b-finetune-moe/` | HPC | Complete |

### StableLM (1.6B) — steps TBD

| Scheme | Checkpoint Dir | Location | Status |
|---|---|---|---|
| Student-only | `checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe/` | HPC | Complete |
| Teacher-Student | `checkpoints_stablelm_TS/llava-stablelm-1.6b-finetune-moe/` | HPC | Complete |
| Entropy (w=0.01) | `checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe/` | HPC | Complete |

**Note:** StableLM has no "author" (random) variant trained yet.

## Evaluation Results

### StableLM — Complete

| Variant | POPE-Pop | POPE-Adv | TextVQA | ScienceQA | MME-P | MME-C | MME-Total | GQA |
|---|---|---|---|---|---|---|---|---|
| Student | 74.3% | — | 49.94% | 60.55% | 1362.2 | 244.6 | 1606.9 | 62.12% |
| TS | 74.5% | — | 50.06% | 60.58% | 1358.4 | 230.0 | 1588.4 | 61.55% |
| Entropy (w=0.01) | 74.4% | — | 50.18% | 59.75% | 1369.4 | 256.4 | 1625.8 | 62.01% |

### Qwen — Partial (author/student/TS from earlier; entropy just completed)

| Variant | POPE-Pop | POPE-Adv | TextVQA | ScienceQA | MME-P | MME-C | MME-Total | GQA |
|---|---|---|---|---|---|---|---|---|
| Author | 88.6% | 86.1% | 48.51% | 63.52% | — | — | 1572.8 | 62.02% |
| Student | — | — | 48.08% | 61.80% | — | — | 1553.8 | 61.95% |
| TS | — | — | 48.89% | 62.72% | — | — | 1561.5 | 62.15% |
| Entropy (w=0.01) | 87.9% | 85.9% | 48.07% | 61.80% | 1301.3 | 233.2 | 1534.5 | 61.73% |
| Entropy (w=0.1) | 87.8% | 86.0% | 47.98% | 62.13% | — | — | — | 61.91% |

### Phi2 — Partial (author/student from earlier; entropy just completed)

| Variant | POPE-Pop | POPE-Adv | TextVQA | ScienceQA | MME-P | MME-C | MME-Total | GQA |
|---|---|---|---|---|---|---|---|---|
| Author | 87.5% | 85.9% | 52.11% | 71.75% | — | — | 1685.7 | 59.44% |
| Student | — | — | 51.69% | 70.76% | — | — | 1670.3 | 60.95% |
| TS | — | — | — | — | — | — | — | — |
| Entropy (w=0.01) | 87.0% | 85.7% | 51.50% | 70.08% | 1337.1 | 272.5 | 1609.6 | 60.49% |

## Code Changes (recent)

### Entropy regularization (committed)
- `normalized_router_flexible.py` → `SimplifiedNormalizedGate`: added `entropy_loss_weight` param, entropy loss `H = -(p * log p).sum(-1).mean()` in forward
- `train.py` → added `entropy_loss_weight` to `ModelArguments`
- `llava_phi_moe.py`, `llava_qwen_moe.py` → pass `entropy_loss_weight` when creating `SimplifiedNormalizedGate`

### Entropy warmup callback (latest)
- `router_callback.py` → new `EntropyWarmupCallback`: linear warmup of entropy_loss_weight from 0 to target over first 10% of steps
- `normalized_router_flexible.py` → `SimplifiedNormalizedGate.update_hyperparameters` now accepts `entropy_loss_weight`
- `train.py` → conditionally adds `EntropyWarmupCallback` when `router_init_mode=no_teacher` and `entropy_loss_weight > 0`
- **Note:** `RouterDistillationCallback` (T/KD/EMA scheduling) remains commented out — not active for any current experiment

### Naming convention
- `w001` = entropy_loss_weight 0.01
- `w01` = entropy_loss_weight 0.1

## TODO

### Analysis (2026-03-23)
- [x] Compiled entropy variants results (all 4: StableLM, Phi2, Qwen w=0.01, Qwen w=0.1)
- [x] Compared entropy vs paper baseline (author's random router)
- [ ] **KEY FINDING**: Entropy improves MME **+13-23%** but POPE Random drops **-38%** ⚠️
- [ ] Investigate POPE Random collapse: is it routing, entropy weighting, or eval artifact?
- [ ] Address open questions in PROJECT_UNDERSTANDING.md one by one

### Blocking Issue
- **POPE Random anomaly**: Our entropy variants score ~50.5% (random guessing) while paper baseline achieves ~88.7%
  - Paper reports POPE Adversarial, we eval POPE Random (different splits)
  - Needs investigation: is entropy harming generalization to POPE random distribution?

### Planned
- [ ] After Exp 1-3 complete: evaluate to see if schedules fix POPE random
- [ ] Compile final results table across all variants (3 backbones × 5 schemes)

### Active Experiments (Sequential, 3 GPUs each, 2026-03-23)

1. **Exp 1: Qwen TS with KD Weight Schedule** — RUNNING (GPUs 2,3,4)
   - Router params: KD weight 0.05→0.01 (stronger early KD), EMA 0.999→0.95
   - Expected: Better TS routing via stronger early teacher signal
   - Duration: ~14h (9240 steps ÷ 11.1 steps/min ~= 14h)
   - Output: `checkpoints_qwen_TS_schedule/llavaqwen-1.8b-finetune-moe`
   - Status: Running step 0/9240

2. **Exp 2: StableLM TS with Temperature + KD Weight Schedule** — PENDING
   - Router params: Temp 2.0→0.8 (softer→sharper), KD weight 0.05→0.01, EMA 0.9999→0.99
   - Expected: Better routing specialization via softer early logits, then sharpening
   - Duration: ~6h (6000 steps ~= 6h)
   - Output: `checkpoints_stablelm_TS_schedule/llava-stablelm-1.6b-finetune-moe`

3. **Exp 3: Qwen entropy w=0.1 with Extended Warmup** — PENDING
   - Entropy params: weight=0.1, warmup_ratio=0.2 (20% instead of 10%)
   - Expected: Longer entropy warmup prevents early routing collapse, allows better specialization
   - Duration: ~14h (9240 steps ~= 14h)
   - Output: `checkpoints_qwen_entropy_w01_warmup/llavaqwen-1.8b-finetune-moe`

### Code Changes (2026-03-23)

- `train.py`: Enabled `RouterDistillationCallback` conditionally for `router_init_mode=teacher_kd`
- `train.py`: Added `entropy_warmup_ratio` parameter to `RouterArguments` (default 0.1)
- `train.py`: Updated `EntropyWarmupCallback` instantiation to use configurable warmup_ratio

### Future Experiments
- [ ] Train StableLM author (random) baseline for complete 3-way comparison
- [ ] Consider EMA freeze for TS (teacher stays at K-means, no drift)

## Notes

### Evaluation Methodology
- Conv modes: Phi2=`phi`, Qwen=`qwen`, StableLM=`stablelm`
- Eval scripts use `deepspeed --include localhost:<gpu>` for single-GPU inference
- All checkpoints backed up to HPC via rsync (use `--no-group` to avoid chgrp errors)
- HPC mount: `/home/prafull/scratch/hpc/` (SSHFS, accessible as local path)
- Eval data: `moellava/eval/<benchmark>/` (NOT top-level `eval/`)

### MME Scoring (CRITICAL)
- **MME Total = MME-Perception + MME-Cognition** (must sum both components)
- Example: Qwen entropy has Perception=1301.3 + Cognition=233.2 = Total 1534.5
- **Do NOT report only one component as the final MME score**
