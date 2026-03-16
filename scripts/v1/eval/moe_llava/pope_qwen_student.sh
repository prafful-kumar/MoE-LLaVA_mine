#!/bin/bash

CONV="qwen"
CKPT_NAME="llavaqwen-1.8b-finetune-moe"
CKPT="checkpoints_qwen_student/${CKPT_NAME}"
EVAL="moellava/eval/pope"

echo "=== Running POPE inference: Qwen Student ==="
deepspeed --include localhost:0 --master_port 29501 moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --image-folder ${EVAL}/val2014 \
    --answers-file ${EVAL}/answers/${CKPT_NAME}_student.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

echo ""
echo "=== Evaluating POPE: Qwen Student ==="
python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/coco \
    --question-file ${EVAL}/llava_pope_test.jsonl \
    --result-file ${EVAL}/answers/${CKPT_NAME}_student.jsonl
