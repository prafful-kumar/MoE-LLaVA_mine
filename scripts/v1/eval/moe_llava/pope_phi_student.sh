#!/bin/bash

CONV="phi"
CKPT_NAME="llavaphi-2.7b-finetune-moe"
CKPT="checkpoints_phi_student/${CKPT_NAME}/checkpoint-1000"
EVAL="moellava/eval/pope"

echo "=== Running POPE inference ==="
echo "Model: ${CKPT}"
echo "Conv mode: ${CONV}"

deepspeed --include localhost:0 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --image-folder ${EVAL}/val2014 \
    --answers-file ${EVAL}/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

echo ""
echo "=== Evaluating POPE results ==="
python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/coco \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --result-file ${EVAL}/answers/${CKPT_NAME}.jsonl
