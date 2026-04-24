#!/bin/bash
# MM-Vet inference + JSON conversion for all current variants
# Generates results/*.json for upload to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator
# Usage: bash scripts/v1/eval/moe_llava/mmvet_new_all.sh [GPU]
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-3}
EVAL="moellava/eval"

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

mkdir -p "${EVAL}/mm-vet/answers"
mkdir -p "${EVAL}/mm-vet/results"

ORDERED=(
    phi2_author phi2_student_final phi_TS phi_entropy phi_entropy_topk_var
    qwen_author qwen_student qwen_TS qwen_TS_schedule qwen_entropy qwen_entropy_w01 qwen_entropy_topk_var
    stablelm_author stablelm_student stablelm_TS stablelm_entropy stablelm_entropy_topk_aux stablelm_entropy_topk_var
)

for VARIANT in "${ORDERED[@]}"; do
    IFS='|' read -r CKPT CONV <<< "${MODELS[$VARIANT]}"

    # Skip if result JSON already exists
    if [ -f "${EVAL}/mm-vet/results/${VARIANT}.json" ]; then
        echo "[SKIP] ${VARIANT} — result JSON already exists"
        continue
    fi

    PORT=$((RANDOM + 29503))
    echo "=============================================="
    echo "MM-Vet: ${VARIANT}  GPU:${GPU}  Port:${PORT}"
    echo "  ${CKPT}"
    echo "=============================================="

    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/mm-vet/llava-mm-vet.jsonl" \
        --image-folder "${EVAL}/mm-vet/images" \
        --answers-file "${EVAL}/mm-vet/answers/${VARIANT}.jsonl" \
        --temperature 0 \
        --conv-mode "${CONV}"

    python3 scripts/convert_mmvet_for_eval.py \
        --src "${EVAL}/mm-vet/answers/${VARIANT}.jsonl" \
        --dst "${EVAL}/mm-vet/results/${VARIANT}.json"

    echo "[DONE] ${VARIANT} — JSON saved to ${EVAL}/mm-vet/results/${VARIANT}.json"
    echo ""
done

echo "=============================================="
echo "MM-VET COMPLETE FOR ALL VARIANTS"
echo "Upload JSON files from: ${EVAL}/mm-vet/results/"
echo "Evaluator: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"
echo "Recommended model: gpt-4-0613 or gpt-4-turbo"
echo "=============================================="
