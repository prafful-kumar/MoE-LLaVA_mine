#!/bin/bash
# Rerun POPE evaluation on ALL checkpoints
# Saves logs per variant in logs/eval/pope/
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-0}
EVAL="moellava/eval"
HPC="/home/prafull/scratch/hpc"
mkdir -p logs/eval/pope

run_pope() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo "=============================================="
    echo "POPE: ${VARIANT}"
    echo "  Checkpoint: ${CKPT}"
    echo "  GPU: ${GPU}"
    echo "=============================================="

    PORT=$((RANDOM + 29503))
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
        --image-folder "${EVAL}/pope/val2014" \
        --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
        --temperature 0 \
        --conv-mode "${CONV}"

    echo "--- POPE Results for ${VARIANT} ---"
    python3 moellava/eval/eval_pope.py \
        --annotation-dir "${EVAL}/pope/coco" \
        --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
        --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"

    echo ""
}

# ---- Phi2 variants ----
run_pope "${HPC}/checkpoints_phi/llavaphi-2.7b-finetune-moe" "phi" "phi2_author" \
    2>&1 | tee logs/eval/pope/phi2_author.log

run_pope "${HPC}/checkpoints_phi_student/llavaphi-2.7b-finetune-moe" "phi" "phi2_student_final" \
    2>&1 | tee logs/eval/pope/phi2_student.log

run_pope "${HPC}/checkpoints_phi_TS/llavaphi-2.7b-finetune-moe" "phi" "phi_TS" \
    2>&1 | tee logs/eval/pope/phi_TS.log

run_pope "${HPC}/checkpoints_phi_entropy/llavaphi-2.7b-finetune-moe" "phi" "phi_entropy" \
    2>&1 | tee logs/eval/pope/phi_entropy.log

# ---- Qwen variants ----
run_pope "${HPC}/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe" "qwen" "qwen_author" \
    2>&1 | tee logs/eval/pope/qwen_author.log

run_pope "${HPC}/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe" "qwen" "qwen_student" \
    2>&1 | tee logs/eval/pope/qwen_student.log

run_pope "${HPC}/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe" "qwen" "qwen_TS" \
    2>&1 | tee logs/eval/pope/qwen_TS.log

run_pope "${HPC}/checkpoints_qwen_TS_schedule/llavaqwen-1.8b-finetune-moe" "qwen" "qwen_TS_schedule" \
    2>&1 | tee logs/eval/pope/qwen_TS_schedule.log

run_pope "${HPC}/checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe" "qwen" "qwen_entropy" \
    2>&1 | tee logs/eval/pope/qwen_entropy.log

# ---- StableLM variants ----
run_pope "${HPC}/random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe" "stablelm" "stablelm_author" \
    2>&1 | tee logs/eval/pope/stablelm_author.log

run_pope "${HPC}/checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe" "stablelm" "stablelm_student" \
    2>&1 | tee logs/eval/pope/stablelm_student.log

run_pope "${HPC}/checkpoints_stablelm_TS/llava-stablelm-1.6b-finetune-moe" "stablelm" "stablelm_TS" \
    2>&1 | tee logs/eval/pope/stablelm_TS.log

run_pope "${HPC}/checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe" "stablelm" "stablelm_entropy" \
    2>&1 | tee logs/eval/pope/stablelm_entropy.log

echo ""
echo "=============================================="
echo "ALL POPE EVALUATIONS COMPLETE"
echo "=============================================="
