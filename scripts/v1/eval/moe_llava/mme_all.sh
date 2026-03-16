#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

REPO_ROOT="/scratch/prafull/MoE-LLaVA_mine"
EVAL="moellava/eval"
GPU=${1:-7}   # default to GPU 7

if [ -z "$GPU" ]; then
    echo "Usage: bash mme_all.sh [gpu_id]"
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
    echo "MME: ${VARIANT}"
    echo "  Model: ${CKPT}"
    PORT=$((RANDOM + 29503))
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    EXPERIMENT_NAME="${VARIANT}"
    ANSWER_FILE="${EVAL}/MME/answers/${EXPERIMENT_NAME}.jsonl"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/MME/llava_mme.jsonl" \
        --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
        --answers-file "${ANSWER_FILE}" \
        --temperature 0 \
        --conv-mode "${CONV}"

    echo "--- Converting answers for ${VARIANT} ---"
    cd "${EVAL}/MME"
    python convert_answer_to_mme.py --experiment "${EXPERIMENT_NAME}"

    echo "--- Evaluating ${VARIANT} ---"
    cd eval_tool
    python calculation.py --results_dir "answers/${EXPERIMENT_NAME}"
    cd "${REPO_ROOT}"

    echo ""
done
