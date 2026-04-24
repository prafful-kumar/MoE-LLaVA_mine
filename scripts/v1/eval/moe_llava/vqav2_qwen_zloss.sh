#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-4}
CKPT="/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_zloss/llavaqwen-1.8b-finetune-moe"
CONV="qwen"
EVAL="moellava/eval"
VARIANT="qwen_zloss"
SPLIT="llava_vqav2_mscoco_test-dev2015"

echo "=============================================="
echo "Qwen ZLoss — VQAv2"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

ANSWER_DIR="${EVAL}/vqav2/answers/${SPLIT}/${VARIANT}"
mkdir -p "${ANSWER_DIR}"

PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/vqav2/${SPLIT}.jsonl" \
    --image-folder "${EVAL}/vqav2/test2015" \
    --answers-file "${ANSWER_DIR}/1_0.jsonl" \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode "${CONV}"

cp "${ANSWER_DIR}/1_0.jsonl" "${ANSWER_DIR}/merge.jsonl"

mkdir -p "${EVAL}/vqav2/answers_upload"
echo "--- Converting for submission ---"
python3 scripts/convert_vqav2_for_submission.py \
    --split "${SPLIT}" \
    --ckpt "${VARIANT}" \
    --dir "${EVAL}/vqav2"
