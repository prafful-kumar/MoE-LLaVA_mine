#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}

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
    echo "VisWiz: ${VARIANT}"
    echo "  Model: ${CKPT}"
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/vizwiz/llava_test.jsonl" \
        --image-folder "${EVAL}/vizwiz/test" \
        --answers-file "${EVAL}/vizwiz/answers/${VARIANT}.jsonl" \
        --temperature 0 \
        --conv-mode "${CONV}"

    # Convert for submission
    mkdir -p "${EVAL}/vizwiz/answers_upload"
    echo "--- Converting ${VARIANT} for submission ---"
    python3 scripts/convert_vizwiz_for_submission.py \
        --annotation-file "${EVAL}/vizwiz/llava_test.jsonl" \
        --result-file "${EVAL}/vizwiz/answers/${VARIANT}.jsonl" \
        --result-upload-file "${EVAL}/vizwiz/answers_upload/${VARIANT}.json"

    echo ""
done
