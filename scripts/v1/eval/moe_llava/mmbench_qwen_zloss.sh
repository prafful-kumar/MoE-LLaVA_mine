#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-4}
CKPT="/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_zloss/llavaqwen-1.8b-finetune-moe"
CONV="qwen"
EVAL="moellava/eval"
VARIANT="qwen_zloss"
SPLIT="mmbench_dev_20230712"

echo "=============================================="
echo "Qwen ZLoss — MMBench"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_mmbench.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/mmbench/${SPLIT}.tsv" \
    --answers-file "${EVAL}/mmbench/answers/${SPLIT}/${VARIANT}.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV}"

mkdir -p "${EVAL}/mmbench/answers_upload/${SPLIT}"

echo "--- Converting for submission ---"
python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file "${EVAL}/mmbench/${SPLIT}.tsv" \
    --result-dir "${EVAL}/mmbench/answers/${SPLIT}" \
    --upload-dir "${EVAL}/mmbench/answers_upload/${SPLIT}" \
    --experiment "${VARIANT}"
