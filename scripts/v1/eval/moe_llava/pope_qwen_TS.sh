#!/bin/bash

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune-moe"
CKPT="/home/prafull/scratch/hpc/checkpoints_qwen_TS/${CKPT_NAME}"
EVAL="moellava/eval/pope"

echo "=== Running POPE inference: Qwen Teacher-Student ==="
deepspeed --include localhost:5 --master_port 29503 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --image-folder ${EVAL}/val2014 \
    --answers-file ${EVAL}/answers/${CKPT_NAME}_TS.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

echo ""
echo "=== Evaluating POPE: Qwen Teacher-Student ==="
python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/coco \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --result-file ${EVAL}/answers/${CKPT_NAME}_TS.jsonl
