#!/bin/bash

gpu_list="6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
CKPT="checkpoints/${CKPT_NAME}"
SPLIT="llava_gqa_testdev_balanced"
EVAL="moellava/eval"
GQADIR="${EVAL}/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) moellava/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL}/gqa/data/images \
        --answers-file ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $GQADIR/$SPLIT/${CKPT_NAME}
python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${CKPT_NAME}/testdev_balanced_predictions.json

cd $GQADIR
python3 eval/eval_gqa.py --tier $SPLIT/${CKPT_NAME}/testdev_balanced \
                         --questions ${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json