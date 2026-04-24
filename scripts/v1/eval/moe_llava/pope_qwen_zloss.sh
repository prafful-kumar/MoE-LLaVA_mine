#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-4}
CKPT="/scratch/prafull/MoE-LLaVA_mine/checkpoints_qwen_zloss/llavaqwen-1.8b-finetune-moe"
CONV="qwen"
EVAL="moellava/eval"
VARIANT="qwen_zloss"

echo "=============================================="
echo "Qwen ZLoss — POPE"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
    --image-folder "${EVAL}/pope/val2014" \
    --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV}"

echo "--- POPE Results ---"
python3 moellava/eval/eval_pope.py \
    --annotation-dir "${EVAL}/pope/coco" \
    --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
    --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"
