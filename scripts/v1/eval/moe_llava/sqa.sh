#!/bin/bash

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
EVAL="moellava/eval"

CKPT_FOLDERS=(
#   "kmeans_6k"
#   "kmeans_40000_KD_first_time_NOaux"
#   "random_no_KD_0_aux"
#   "random_no_KD_0.1_aux"
#   "random_no_KD_0.5_aux"
"kmeans_5000"
"kmeans_5000_KD_0.1_EMA_0.999_NOaux"
"kmeans_5000_large_BS_KD_0.1_EMA_0.999_NOaux"
"kmeans_20000_KD_0.1_EMA_0.999_NOaux"
"kmeans_40000"
"kmeans_40000_KD_second_time_NOaux"
"random_no_KD_0.01_aux"
)

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    CKPT="${CKPT_FOLDER}/${CKPT_NAME}"

    echo "Evaluating checkpoint folder: ${CKPT_FOLDER}"

    deepspeed --include localhost:7 --master_port $((1 + 29501)) \
        moellava/eval/model_vqa_science.py \
        --model-path ${CKPT} \
        --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
        --image-folder ${EVAL}/scienceqa/images/test \
        --answers-file ${EVAL}/scienceqa/answers/${CKPT_FOLDER}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode ${CONV}

    python3 moellava/eval/eval_science_qa.py \
        --base-dir ${EVAL}/scienceqa \
        --result-file ${EVAL}/scienceqa/answers/${CKPT_FOLDER}.jsonl \
        --output-file ${EVAL}/scienceqa/answers/${CKPT_FOLDER}_output.jsonl \
        --output-result ${EVAL}/scienceqa/answers/${CKPT_FOLDER}_result.json
done
