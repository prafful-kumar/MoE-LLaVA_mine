#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-0}
CKPT="/scratch/prafull/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe"
CONV="qwen"
EVAL="moellava/eval"
VARIANT="qwen_author"

echo "=============================================="
echo "Qwen Author — GQA"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

echo ">>> GQA"
PORT=$((RANDOM + 29503))
mkdir -p "${EVAL}/gqa/answers/${VARIANT}"
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "${EVAL}/gqa/data/images" --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
    --num-chunks 1 --chunk-idx 0 --temperature 0 --conv-mode "${CONV}"
cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"
python3 scripts/convert_gqa_for_eval.py \
    --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
    --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"
python3 "${EVAL}/eval_gqa.py" --tier testdev_balanced \
    --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
    --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"

echo "=============================================="
echo "GQA COMPLETE FOR ${VARIANT}"
echo "=============================================="
