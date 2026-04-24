# MoE-LLaVA — Commands Reference

All commands run from `/scratch/prafull/MoE-LLaVA_mine`.
Conda env: `moellava_mine`.
Single GPU for eval (shared server). Multi-GPU for training.

---

## Before Running Anything

```bash
# Check GPU availability
nvidia-smi

# Check for checkpoint collisions
cat EXPERIMENT_TODO.md
cat CHECKPOINT_STATUS.md

# Check if output_dir already exists (replace <dir> with actual path)
ls checkpoints_<backbone>_<variant>/
```

---

## Training

### Phi2 (2.7B) — uses 4 GPUs, batch=1, grad_accum=12
| Variant | Script | Output Dir | GPUs (default) |
|---|---|---|---|
| TS (teacher_kd) | `scripts/v1/phi2/finetune_moe.sh` | `checkpoints_phi_TS/` | 0,1,2,3 |
| Entropy-topk (w=0.03) | `scripts/v1/phi2/finetune_moe_entropy.sh` | `checkpoints_phi_entropy/` | 4,5,6,7 |
| **Entropy-topk-var (w=0.03, imbal=0.1, var=0.01)** | `scripts/v1/phi2/finetune_moe_entropy_topk_var_loss.sh` | `checkpoints_phi_entropy_topk_var/` | 4,5,6,7 |

```bash
mkdir -p logs/train
bash scripts/v1/phi2/finetune_moe_entropy_topk_var_loss.sh
# log: logs/train/phi_entropy_topk_var.log
```

### Qwen (1.8B) — uses 3 GPUs, batch=2, grad_accum=12
| Variant | Script | Output Dir | GPUs (default) |
|---|---|---|---|
| TS (teacher_kd) | `scripts/v1/qwen/finetune_moe.sh` | `checkpoints_qwen_TS/` | 0,1,2 |
| TS Schedule | `scripts/v1/qwen/finetune_moe_TS_schedule.sh` | `checkpoints_qwen_TS_schedule/` | 0,1,2 |
| Entropy-topk (w=0.03) | `scripts/v1/qwen/finetune_moe_entropy.sh` | `checkpoints_qwen_entropy/` | 0,1,2 |
| **Entropy-topk-var (w=0.03, imbal=0.1, var=0.01)** | `scripts/v1/qwen/finetune_moe_entropy_topk_var_loss.sh` | `checkpoints_qwen_entropy_topk_var/` | 3,4,5 |

```bash
mkdir -p logs/train
bash scripts/v1/qwen/finetune_moe_entropy_topk_var_loss.sh
# log: logs/train/qwen_entropy_topk_var.log
```

### StableLM (1.6B) — uses 3 GPUs, batch=2, grad_accum=12
| Variant | Script | Output Dir | GPUs (default) |
|---|---|---|---|
| TS (teacher_kd) | `scripts/v1/stablelm/finetune_moe_TS.sh` | `checkpoints_stablelm_TS/` | 0,1,2 |
| Student (student_warm) | `scripts/v1/stablelm/finetune_moe_student.sh` | `checkpoints_stablelm_student/` | 0,1,2 |
| Entropy-topk (w=0.03) | `scripts/v1/stablelm/finetune_moe_entropy.sh` | `checkpoints_stablelm_entropy/` | 0,1,2 |
| **Entropy-topk + aux=0.01** | `scripts/v1/stablelm/finetune_moe_entropy_topk_aux.sh` | `checkpoints_stablelm_entropy_topk_aux/` | 1,2,3 |
| **Entropy-topk-var (w=0.03, imbal=0.1, var=0.01)** | `scripts/v1/stablelm/finetune_moe_entropy_topk_var_loss.sh` | `checkpoints_stablelm_entropy_topk_var/` | 0,1,2 |

```bash
mkdir -p logs/train
bash scripts/v1/stablelm/finetune_moe_entropy_topk_var_loss.sh
# log: logs/train/stablelm_entropy_topk_var.log
```

---

## Evaluation (all 5 benchmarks — single GPU)

Each `*_all.sh` accepts GPU as argument: `bash <script> <GPU_ID>`

### Phi2
```bash
# Entropy-topk (checkpoints_phi_entropy/)
bash scripts/v1/eval/moe_llava/phi_entropy_all.sh 3

# Entropy-topk-var (checkpoints_phi_entropy_topk_var/) — NEW, no script yet
# Use phi_entropy_all.sh as template, change CKPT and VARIANT
```

### Qwen
```bash
# Entropy-topk (checkpoints_qwen_entropy/)
bash scripts/v1/eval/moe_llava/qwen_entropy_all.sh 4

# Entropy-topk-var (checkpoints_qwen_entropy_topk_var/) — DONE training, needs eval
# Template: qwen_entropy_all.sh with CKPT=checkpoints_qwen_entropy_topk_var, VARIANT=qwen_entropy_topk_var
```

### StableLM
```bash
# Entropy-topk (checkpoints_stablelm_entropy/)
bash scripts/v1/eval/moe_llava/stablelm_entropy_all.sh 2

# Entropy-topk-var (checkpoints_stablelm_entropy_topk_var/) — DONE training, needs eval
# Template: stablelm_entropy_all.sh with CKPT=checkpoints_stablelm_entropy_topk_var, VARIANT=stablelm_entropy_topk_var

# Entropy-topk-aux (checkpoints_stablelm_entropy_topk_aux/) — DONE training, needs eval
# Template: stablelm_entropy_all.sh with CKPT=checkpoints_stablelm_entropy_topk_aux, VARIANT=stablelm_entropy_topk_aux
```

### Generic single-benchmark eval (template)
```bash
GPU=3
CKPT="./checkpoints_<backbone>_<variant>/<model_dir>"
CONV="phi"   # or: qwen, stablelm
VARIANT="<variant_name>"
EVAL="moellava/eval"

# POPE
deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
    --image-folder "${EVAL}/pope/val2014" \
    --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
python3 moellava/eval/eval_pope.py \
    --annotation-dir "${EVAL}/pope/coco" \
    --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
    --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"

# TextVQA
deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "${EVAL}/textvqa/train_images" \
    --answers-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
python3 -m moellava.eval.eval_textvqa \
    --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" \
    --result-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl"

# ScienceQA
deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) moellava/eval/model_vqa_science.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${EVAL}/scienceqa/images/test" \
    --answers-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --single-pred-prompt --temperature 0 --conv-mode "${CONV}"
python3 moellava/eval/eval_science_qa.py \
    --base-dir "${EVAL}/scienceqa" \
    --result-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --output-file "${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl" \
    --output-result "${EVAL}/scienceqa/answers/${VARIANT}_result.json"

# MME
deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/MME/llava_mme.jsonl" \
    --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
    --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
cd "${EVAL}/MME" && python3 convert_answer_to_mme.py --experiment "${VARIANT}"
cd eval_tool && python3 calculation.py --results_dir "answers/${VARIANT}"
cd /scratch/prafull/MoE-LLaVA_mine

# GQA
mkdir -p "${EVAL}/gqa/answers/${VARIANT}"
deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "${EVAL}/gqa/data/images" \
    --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
    --num-chunks 1 --chunk-idx 0 --temperature 0 --conv-mode "${CONV}"
cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"
python3 scripts/convert_gqa_for_eval.py \
    --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
    --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"
python3 "${EVAL}/eval_gqa.py" \
    --tier testdev_balanced \
    --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
    --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"
```

---

## Checkpoint Transfer to HPC

```bash
# Generic transfer (avoids chgrp errors on SSHFS)
rsync -av --no-g checkpoints_<backbone>_<variant>/ /home/prafull/scratch/hpc/checkpoints_<backbone>_<variant>/

# Example: transfer qwen_entropy_topk_var
rsync -av --no-g checkpoints_qwen_entropy_topk_var/ /home/prafull/scratch/hpc/checkpoints_qwen_entropy_topk_var/
```

---

## Monitoring Training

```bash
# Live log
tail -f logs/train/<variant>.log

# Check if training process is alive
pgrep -a python | grep <variant_keyword>

# TensorBoard
tensorboard --logdir checkpoints_<backbone>_<variant>/<model_dir>/runs --port 6006
```

---

## Pending Evals (as of 2026-04-02)

| Checkpoint | Backbone | Status |
|---|---|---|
| `checkpoints_qwen_entropy_topk_var` | Qwen 1.8B | Training done — NO evals yet |
| `checkpoints_stablelm_entropy_topk_aux` | StableLM 1.6B | Training done — NO evals yet |
| `checkpoints_stablelm_entropy_topk_var` | StableLM 1.6B | Training done — NO evals yet |
| `checkpoints_phi_entropy_topk_var` | Phi2 2.7B | **Training in progress** — NO evals yet |
| `checkpoints_qwen_TS_schedule` | Qwen 1.8B | Training done — NO evals yet |
