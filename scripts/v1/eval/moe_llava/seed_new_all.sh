#!/bin/bash
# SEED-Bench inference + conversion for all current variants (image-only questions)
# Generates answers_upload/*.jsonl for submission
# Usage: bash scripts/v1/eval/moe_llava/seed_new_all.sh [GPU]
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-3}
EVAL="moellava/eval"
QUESTION_FILE="${EVAL}/seed_bench/llava-seed-bench-image-only.jsonl"

declare -A MODELS
# Phi2 (2.7B)
MODELS["phi2_author"]="/scratch/prafull/hpc/checkpoints_phi/llavaphi-2.7b-finetune-moe|phi"
MODELS["phi2_student_final"]="/scratch/prafull/hpc/checkpoints_phi_student/llavaphi-2.7b-finetune-moe|phi"
MODELS["phi_TS"]="/scratch/prafull/hpc/checkpoints_phi_TS/llavaphi-2.7b-finetune-moe|phi"
MODELS["phi_entropy"]="/scratch/prafull/hpc/checkpoints_phi_entropy/llavaphi-2.7b-finetune-moe|phi"
MODELS["phi_entropy_topk_var"]="/scratch/prafull/hpc/checkpoints_phi_entropy_topk_var/llavaphi-2.7b-finetune-moe|phi"
# Qwen (1.8B)
MODELS["qwen_author"]="/scratch/prafull/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_student"]="/scratch/prafull/hpc/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_TS"]="/scratch/prafull/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_TS_schedule"]="/scratch/prafull/hpc/checkpoints_qwen_TS_schedule/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_entropy"]="/scratch/prafull/hpc/checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_entropy_w01"]="/scratch/prafull/hpc/checkpoints_qwen_entropy_w01/llavaqwen-1.8b-finetune-moe|qwen"
MODELS["qwen_entropy_topk_var"]="/scratch/prafull/hpc/checkpoints_qwen_entropy_topk_var/llavaqwen-1.8b-finetune-moe|qwen"
# StableLM (1.6B)
MODELS["stablelm_author"]="/scratch/prafull/hpc/random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe|stablelm"
MODELS["stablelm_student"]="/scratch/prafull/hpc/checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe|stablelm"
MODELS["stablelm_TS"]="/scratch/prafull/hpc/checkpoints_stablelm_TS/llava-stablelm-1.6b-finetune-moe|stablelm"
MODELS["stablelm_entropy"]="/scratch/prafull/hpc/checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe|stablelm"
MODELS["stablelm_entropy_topk_aux"]="/scratch/prafull/hpc/checkpoints_stablelm_entropy_topk_aux/llava-stablelm-1.6b-finetune-moe|stablelm"
MODELS["stablelm_entropy_topk_var"]="/scratch/prafull/hpc/checkpoints_stablelm_entropy_topk_var/llava-stablelm-1.6b-finetune-moe|stablelm"

mkdir -p "${EVAL}/seed_bench/answers_upload"

ORDERED=(
    phi2_author phi2_student_final phi_TS phi_entropy phi_entropy_topk_var
    qwen_author qwen_student qwen_TS qwen_TS_schedule qwen_entropy qwen_entropy_w01 qwen_entropy_topk_var
    stablelm_author stablelm_student stablelm_TS stablelm_entropy stablelm_entropy_topk_aux stablelm_entropy_topk_var
)

for VARIANT in "${ORDERED[@]}"; do
    IFS='|' read -r CKPT CONV <<< "${MODELS[$VARIANT]}"

    ANSWER_DIR="${EVAL}/seed_bench/answers/${VARIANT}"

    # Skip if already done
    if [ -f "${EVAL}/seed_bench/answers_upload/${VARIANT}.jsonl" ]; then
        echo "[SKIP] ${VARIANT} — upload file already exists"
        continue
    fi

    PORT=$((RANDOM + 29503))
    echo "=============================================="
    echo "SEED-Bench: ${VARIANT}  GPU:${GPU}  Port:${PORT}"
    echo "  ${CKPT}"
    echo "=============================================="

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

    cp "${ANSWER_DIR}/1_0.jsonl" "${ANSWER_DIR}/merge.jsonl"

    python3 scripts/convert_seed_for_submission.py \
        --annotation-file "${EVAL}/seed_bench/SEED-Bench.json" \
        --result-file "${ANSWER_DIR}/merge.jsonl" \
        --result-upload-file "${EVAL}/seed_bench/answers_upload/${VARIANT}.jsonl"

    echo "[DONE] ${VARIANT}"
    echo ""
done

echo "=============================================="
echo "SEED-BENCH COMPLETE FOR ALL VARIANTS"
echo "Upload files in: ${EVAL}/seed_bench/answers_upload/"
echo "=============================================="
