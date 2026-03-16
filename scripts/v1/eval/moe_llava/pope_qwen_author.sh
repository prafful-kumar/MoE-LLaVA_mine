#!/bin/bash

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune-moe"
CKPT="checkpoints_qwen_author/${CKPT_NAME}"
EVAL="moellava/eval/pope"

echo "=== Running POPE inference: Qwen Author (Random) ==="
deepspeed --include localhost:3 --master_port 29502 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --image-folder ${EVAL}/val2014 \
    --answers-file ${EVAL}/answers/${CKPT_NAME}_author.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

echo ""
echo "=== Evaluating POPE: Qwen Author (Random) ==="
python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/coco \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --result-file ${EVAL}/answers/${CKPT_NAME}_author.jsonl
