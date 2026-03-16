#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}   # default to GPU 7

if [ -z "$GPU" ]; then
    echo "Usage: bash sqa_all.sh [gpu_id]"
    exit 1
fi

declare -A MODELS
MODELS=(
    ["qwen_student"]="checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe|qwen"
    ["qwen_author"]="checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe|qwen"
    ["qwen_TS"]="/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe|qwen"
    ["phi2_student"]="checkpoints_phi_student/llavaphi-2.7b-finetune-moe/checkpoint-1000|phi"
)

for VARIANT in "${!MODELS[@]}"; do
    IFS='|' read -r CKPT CONV <<< "${MODELS[$VARIANT]}"

    echo "=============================================="
    echo "ScienceQA: ${VARIANT}"
    echo "  Model: ${CKPT}"
    PORT=$((RANDOM + 29503))
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    ANSWER_FILE="${EVAL}/scienceqa/answers/${VARIANT}.jsonl"
    OUTPUT_FILE="${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl"
    RESULT_FILE="${EVAL}/scienceqa/answers/${VARIANT}_result.json"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
        --model-path "${CKPT}" \
        --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
        --image-folder ${EVAL}/scienceqa/images/test \
        --answers-file "${ANSWER_FILE}" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode ${CONV}

    echo "--- Evaluating ${VARIANT} ---"
    python3 moellava/eval/eval_science_qa.py \
        --base-dir ${EVAL}/scienceqa \
        --result-file "${ANSWER_FILE}" \
        --output-file "${OUTPUT_FILE}" \
        --output-result "${RESULT_FILE}"

    echo ""
done
