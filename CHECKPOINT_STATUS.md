# Checkpoint Status (2026-03-24)

## Available Checkpoints

### On HPC (`/home/prafull/scratch/hpc/`)

| Checkpoint | Backbone | Init Mode | Status |
|---|---|---|---|
| `checkpoints_qwen_author` | Qwen 1.8B | `teacher_kd` (author code) | OK |
| `checkpoints_qwen_student` | Qwen 1.8B | `student_warm` | OK |
| `checkpoints_qwen_TS` | Qwen 1.8B | `teacher_kd` | OK |
| `checkpoints_qwen_entropy` | Qwen 1.8B | `no_teacher` + entropy w=0.01 | OK |
| `checkpoints_qwen_entropy_w01` | Qwen 1.8B | `no_teacher` + entropy w=0.1 | OK |
| `checkpoints_phi` | Phi2 2.7B | `teacher_kd` (author code) | OK — this is phi2_author |
| `checkpoints_phi_TS` | Phi2 2.7B | `teacher_kd` | OK |
| `checkpoints_phi_entropy` | Phi2 2.7B | `no_teacher` + entropy w=0.01 | OK |
| `checkpoints_stablelm_TS` | StableLM 1.6B | `teacher_kd` | OK |
| `checkpoints_stablelm_student` | StableLM 1.6B | `student_warm` | OK |
| `checkpoints_stablelm_entropy` | StableLM 1.6B | `no_teacher` + entropy w=0.01 | OK |

### Local only (`/scratch/prafull/MoE-LLaVA_mine/`)

| Checkpoint | Backbone | Init Mode | Status |
|---|---|---|---|
| `checkpoints_qwen_TS_schedule` | Qwen 1.8B | `teacher_kd` + schedule | OK — training complete, not yet backed up to HPC, not yet evaluated |

## Missing / Need to Train

| Variant | Backbone | What's needed | Notes |
|---|---|---|---|
| **phi2_student** | Phi2 2.7B | Full training run | No `student_warm` script for Phi2 exists. No checkpoint anywhere. Noted as missing in `fill_missing_evals.sh` |
| **stablelm_author** | StableLM 1.6B | Full training run | No author-code checkpoint for StableLM. Would need to run original MoE-LLaVA code for StableLM |
| **stablelm_TS_schedule** | StableLM 1.6B | Full training run | Script exists (`scripts/v1/stablelm/finetune_moe_TS_schedule.sh`) but not yet trained |
| **phi2_TS_schedule** | Phi2 2.7B | Full training run | No script exists yet |
| **qwen_entropy_w01_warmup** | Qwen 1.8B | Full training run | Script exists (`scripts/v1/qwen/finetune_moe_entropy_w01_warmup.sh`) but not yet trained |

## Missing Evaluations (checkpoint exists but evals incomplete)

| Variant | Missing Benchmarks | Notes |
|---|---|---|
| **qwen_TS_schedule** | ALL 5 (SQA, TextVQA, GQA, POPE, MME) | Just finished training |
| **qwen_author** | GQA (final) | Only step-snapshots exist (step1-step900); step1000 dir is empty. POPE file was truncated (killed at 22%). Old POPE file under `llavaqwen-1.8b-finetune-moe_author.jsonl` is complete |
| **phi_author** | GQA differs from phi2_author | `phi_author/` = 59.40%, `phi2_author/` = 59.44% — nearly same, likely same checkpoint evaluated twice |
