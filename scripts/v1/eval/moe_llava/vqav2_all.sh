#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}
SPLIT="llava_vqav2_mscoco_test-dev2015"

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
    echo "VQAv2: ${VARIANT}"
    echo "  Model: ${CKPT}"
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    ANSWER_DIR="${EVAL}/vqav2/answers/${SPLIT}/${VARIANT}"
    mkdir -p "${ANSWER_DIR}"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/vqav2/${SPLIT}.jsonl" \
        --image-folder "${EVAL}/vqav2/test2015" \
        --answers-file "${ANSWER_DIR}/1_0.jsonl" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode "${CONV}"

    # Merge (single chunk)
    cp "${ANSWER_DIR}/1_0.jsonl" "${ANSWER_DIR}/merge.jsonl"

    # Convert for submission
    mkdir -p "${EVAL}/vqav2/answers_upload"
    echo "--- Converting ${VARIANT} for submission ---"
    python3 scripts/convert_vqav2_for_submission.py \
        --split "${SPLIT}" \
        --ckpt "${VARIANT}" \
        --dir "${EVAL}/vqav2"

    echo ""
done
