# Plan — MoE-LLaVA Experiments

## Experiment Schemes

We are comparing 3 router initialization schemes:
1. **Random** (`router_init_mode=random`) — original paper's approach (referred to as "author")
2. **Student-only** (`router_init_mode=no_teacher`) — K-means centroid init, no teacher KD
3. **Teacher-Student (TS)** (`router_init_mode=teacher_kd`) — K-means centroid init + teacher KD

If "norm" appears in a file/checkpoint name, it means the router input is normalized.

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

### Qwen (1.8B) — fully trained (9240/9240 steps)

| Scheme | Checkpoint Dir | Status |
|---|---|---|
| Random (author) | `checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/` | Complete |
| Student-only | `checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/` | Complete |
| Teacher-Student | `checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/` | Complete |

### Phi2 (2.7B) — training incomplete

| Scheme | Checkpoint Dir | Status |
|---|---|---|
| Student-only | `checkpoints_phi_student/llavaphi-2.7b-finetune-moe/checkpoint-1000` | Incomplete (1000/13860 steps) |

### StableLM (1.6B) — not started

## Evaluation Workflow

### Phase 1: POPE Benchmark (starting point)
- [x] Create eval script for Phi2 (`scripts/v1/eval/moe_llava/pope_phi_student.sh`)
- [ ] Run POPE eval for Phi2 checkpoint-1000 — **in progress**
- [ ] Create & run POPE eval for Qwen author
- [ ] Create & run POPE eval for Qwen student
- [ ] Create & run POPE eval for Qwen TS
- [ ] Collect and compare POPE results (F1, Acc, Precision, Recall across random/popular/adversarial)

### Phase 2: Remaining Benchmarks (for all complete models)
- [ ] TextVQA
- [ ] ScienceQA
- [ ] MME
- [ ] MMBench
- [ ] VQAv2 (multi-GPU, submit to eval server)
- [ ] GQA (multi-GPU)
- [ ] MM-Vet
- [ ] SEED-Bench (multi-GPU)

### Phase 3: Results Summary
- [ ] Compile all benchmark results into a comparison table (random vs student vs TS)
- [ ] Compare against MoE-LLaVA paper baselines

## Training TODO
- [ ] Complete Phi2 student-only training (resume from checkpoint-1000, needs ~12860 more steps)
- [ ] Train Phi2 random and TS schemes
- [ ] Train StableLM all 3 schemes

## Eval Data Locations

All eval data is under `moellava/eval/<benchmark>/` (not top-level `eval/`).

| Benchmark | Data Path | Ready? |
|---|---|---|
| POPE | `moellava/eval/pope/` (40504 val2014 images) | Yes |
| TextVQA | `moellava/eval/textvqa/` | Check |
| ScienceQA | `moellava/eval/scienceqa/` | Check |
| MME | `moellava/eval/MME/` | Check |
| MMBench | `moellava/eval/mmbench/` | Check |
| VQAv2 | `moellava/eval/vqav2/` | Check |
| GQA | `moellava/eval/gqa/` | Check |
| MM-Vet | `moellava/eval/mm-vet/` | Check |
| SEED-Bench | `moellava/eval/seed_bench/` | Check |

## Notes
- Conv modes: Phi2=`phi`, Qwen=`qwen`, StableLM=`stablelm`
- POPE runs at ~6.4 it/s on A100, ~23 min for 8910 questions
- Eval scripts use `deepspeed --include localhost:<gpu>` for single-GPU inference
- Qwen total training steps: 9240 (3 GPUs, batch=2, grad_accum=12)
- Phi2 total training steps: 13860 (4 GPUs, batch=1, grad_accum=12)
