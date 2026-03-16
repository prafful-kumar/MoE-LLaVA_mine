#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${EVAL}/gqa/data"

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
    echo "GQA: ${VARIANT}"
    echo "  Model: ${CKPT}"
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    ANSWER_DIR="${EVAL}/gqa/answers/${SPLIT}/${VARIANT}"
    mkdir -p "${ANSWER_DIR}"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/gqa/${SPLIT}.jsonl" \
        --image-folder "${EVAL}/gqa/data/images" \
        --answers-file "${ANSWER_DIR}/1_0.jsonl" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode "${CONV}"

    # Merge (single chunk, just copy)
    cp "${ANSWER_DIR}/1_0.jsonl" "${ANSWER_DIR}/merge.jsonl"

    # Convert for GQA eval
    PRED_DIR="${GQADIR}/${SPLIT}/${VARIANT}"
    mkdir -p "${PRED_DIR}"
    python3 scripts/convert_gqa_for_eval.py \
        --src "${ANSWER_DIR}/merge.jsonl" \
        --dst "${PRED_DIR}/testdev_balanced_predictions.json"

    # Run official GQA eval
    echo "--- Evaluating ${VARIANT} ---"
    python3 moellava/eval/eval_gqa.py \
        --tier "${EVAL}/gqa/data/${SPLIT}/${VARIANT}/testdev_balanced" \
        --questions "${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json"

    echo ""
done
