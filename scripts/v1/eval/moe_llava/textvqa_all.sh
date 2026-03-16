#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-7}   # default to GPU 7

if [ -z "$GPU" ]; then
    echo "Usage: bash textvqa_all.sh [gpu_id]"
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
    echo "TextVQA: ${VARIANT}"
    echo "  Model: ${CKPT}"
    PORT=$((RANDOM + 29503))
    echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
    echo "=============================================="

    ANSWER_FILE="${EVAL}/textvqa/answers/${VARIANT}.jsonl"

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ${EVAL}/textvqa/train_images \
        --answers-file "${ANSWER_FILE}" \
        --temperature 0 \
        --conv-mode ${CONV}

    echo "--- Evaluating ${VARIANT} ---"
    python3 -m moellava.eval.eval_textvqa \
        --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
        --result-file "${ANSWER_FILE}"

    echo ""
done
