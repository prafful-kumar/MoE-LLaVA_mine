#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}

# Use image-only questions (video frames not available)
QUESTION_FILE="${EVAL}/seed_bench/llava-seed-bench-image-only.jsonl"

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
    echo "SEED-Bench: ${VARIANT}"
    echo "  Model: ${CKPT}"
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    ANSWER_DIR="${EVAL}/seed_bench/answers/${VARIANT}"
    mkdir -p "${ANSWER_DIR}"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${EVAL}/seed_bench" \
        --answers-file "${ANSWER_DIR}/1_0.jsonl" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode "${CONV}"

    # Merge (single chunk)
    cp "${ANSWER_DIR}/1_0.jsonl" "${ANSWER_DIR}/merge.jsonl"

    # Evaluate
    echo "--- Evaluating ${VARIANT} ---"
    python3 scripts/convert_seed_for_submission.py \
        --annotation-file "${EVAL}/seed_bench/SEED-Bench.json" \
        --result-file "${ANSWER_DIR}/merge.jsonl" \
        --result-upload-file "${EVAL}/seed_bench/answers_upload/${VARIANT}.jsonl"

    echo ""
done
