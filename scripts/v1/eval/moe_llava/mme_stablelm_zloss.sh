#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-2}
CKPT="/scratch/prafull/MoE-LLaVA_mine/checkpoints_stablelm_zloss/llava-stablelm-1.6b-finetune-moe"
CONV="stablelm"
EVAL="moellava/eval"
VARIANT="stablelm_zloss"

echo "=============================================="
echo "StableLM ZLoss — MME"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/MME/llava_mme.jsonl" \
    --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
    --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV}"

echo "--- MME Results ---"
cd "${EVAL}/MME"
python3 convert_answer_to_mme.py --experiment "${VARIANT}"
cd eval_tool
python3 calculation.py --results_dir "answers/${VARIANT}"
cd /scratch/prafull/MoE-LLaVA_mine
