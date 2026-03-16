#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}
SPLIT="mmbench_dev_20230712"

declare -A MODELS
MODELS=(
    ["qwen_student"]="checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe|qwen"
    ["qwen_author"]="checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe|qwen"
    ["qwen_TS"]="/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe|qwen"
    ["phi2_student"]="checkpoints_phi_student/llavaphi-2.7b-finetune-moe/checkpoint-1000|phi"
)

for VARIANT in "${!MODELS[@]}"; do
    IFS='|' read -r CKPT CONV <<< "${MODELS[$VARIANT]}"
    PORT=$((RANDOM + 29503))

    echo "=============================================="
    echo "MMBench: ${VARIANT}"
    echo "  Model: ${CKPT}"
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_mmbench.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/mmbench/${SPLIT}.tsv" \
        --answers-file "${EVAL}/mmbench/answers/${SPLIT}/${VARIANT}.jsonl" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode "${CONV}"

    mkdir -p "${EVAL}/mmbench/answers_upload/${SPLIT}"

    echo "--- Converting ${VARIANT} for submission ---"
    python3 scripts/convert_mmbench_for_submission.py \
        --annotation-file "${EVAL}/mmbench/${SPLIT}.tsv" \
        --result-dir "${EVAL}/mmbench/answers/${SPLIT}" \
        --upload-dir "${EVAL}/mmbench/answers_upload/${SPLIT}" \
        --experiment "${VARIANT}"

    echo ""
done
