#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-2}
CKPT="/scratch/prafull/MoE-LLaVA_mine/checkpoints_stablelm_zloss/llava-stablelm-1.6b-finetune-moe"
CONV="stablelm"
EVAL="moellava/eval"
VARIANT="stablelm_zloss"

echo "=============================================="
echo "StableLM ZLoss — ScienceQA"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${EVAL}/scienceqa/images/test" \
    --answers-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV}"

echo "--- ScienceQA Results ---"
python3 moellava/eval/eval_science_qa.py \
    --base-dir "${EVAL}/scienceqa" \
    --result-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --output-file "${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl" \
    --output-result "${EVAL}/scienceqa/answers/${VARIANT}_result.json"
