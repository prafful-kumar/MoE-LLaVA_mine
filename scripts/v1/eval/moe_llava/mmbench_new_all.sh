#!/bin/bash
# MMBench inference + xlsx conversion for all current variants
# Generates answers_upload/*.xlsx for submission to https://mmbench.opencompass.org.cn/mmbench-submission
# Usage: bash scripts/v1/eval/moe_llava/mmbench_new_all.sh [GPU]
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-3}
EVAL="moellava/eval"
SPLIT="mmbench_dev_20230712"

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

mkdir -p "${EVAL}/mmbench/answers/${SPLIT}"
mkdir -p "${EVAL}/mmbench/answers_upload/${SPLIT}"

ORDERED=(
    phi2_author phi2_student_final phi_TS phi_entropy phi_entropy_topk_var
    qwen_author qwen_student qwen_TS qwen_TS_schedule qwen_entropy qwen_entropy_w01 qwen_entropy_topk_var
    stablelm_author stablelm_student stablelm_TS stablelm_entropy stablelm_entropy_topk_aux stablelm_entropy_topk_var
)

for VARIANT in "${ORDERED[@]}"; do
    IFS='|' read -r CKPT CONV <<< "${MODELS[$VARIANT]}"

    # Skip if xlsx already generated
    if [ -f "${EVAL}/mmbench/answers_upload/${SPLIT}/${VARIANT}.xlsx" ]; then
        echo "[SKIP] ${VARIANT} — xlsx already exists"
        continue
    fi

    PORT=$((RANDOM + 29503))
    echo "=============================================="
    echo "MMBench: ${VARIANT}  GPU:${GPU}  Port:${PORT}"
    echo "  ${CKPT}"
    echo "=============================================="

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_mmbench.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/mmbench/${SPLIT}.tsv" \
        --answers-file "${EVAL}/mmbench/answers/${SPLIT}/${VARIANT}.jsonl" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode "${CONV}"

    python3 scripts/convert_mmbench_for_submission.py \
        --annotation-file "${EVAL}/mmbench/${SPLIT}.tsv" \
        --result-dir "${EVAL}/mmbench/answers/${SPLIT}" \
        --upload-dir "${EVAL}/mmbench/answers_upload/${SPLIT}" \
        --experiment "${VARIANT}"

    echo "[DONE] ${VARIANT} — xlsx saved to ${EVAL}/mmbench/answers_upload/${SPLIT}/${VARIANT}.xlsx"
    echo ""
done

echo "=============================================="
echo "MMBENCH COMPLETE FOR ALL VARIANTS"
echo "Upload xlsx files from: ${EVAL}/mmbench/answers_upload/${SPLIT}/"
echo "Submit to: https://mmbench.opencompass.org.cn/mmbench-submission"
echo "=============================================="
