# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CONV="stablelm"
# CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
# CKPT="checkpoints/${CKPT_NAME}"
# SPLIT="llava_vqav2_mscoco_test-dev2015"
# EVAL="moellava/eval"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) moellava/eval/model_vqa_loader.py \
#         --model-path ${CKPT} \
#         --question-file ${EVAL}/vqav2/$SPLIT.jsonl \
#         --image-folder ${EVAL}/vqav2/test2015 \
#         --answers-file ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode ${CONV} &
# done

# wait

# output_file=${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${EVAL}/vqav2/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python3 scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt ${CKPT_NAME} --dir ${EVAL}/vqav2

#!/bin/bash
set -e

############################
# CONFIG
############################

gpu_list="2,3,5"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
EVAL="moellava/eval"
SPLIT="llava_vqav2_mscoco_test-dev2015"

CKPT_FOLDERS=(
  # "kmeans_6k"
  # "kmeans_40000_KD_first_time_NOaux"
  # "random_no_KD_0_aux"
  # "random_no_KD_0.1_aux"
  # "random_no_KD_0.5_aux"
  # "kmeans_5000"
  # "kmeans_5000_KD_0.1_EMA_0.999_NOaux"
  # "kmeans_5000_large_BS_KD_0.1_EMA_0.999_NOaux"
  # "kmeans_20000_KD_0.1_EMA_0.999_NOaux"
  # "kmeans_40000"
  # "kmeans_40000_KD_second_time_NOaux"
  # "random_no_KD_0.01_aux"
  # "kmeans_40000_KD_0.1_EMA_0.7_NOaux"
  # "kmeans_40000_KD_0.1_EMA_0.8_NOaux"
  "kmeans_40000_dyn_hyp"
)

############################
# LOOP OVER CHECKPOINTS
############################

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    echo "=============================================="
    echo "Evaluating VQAv2 checkpoint: ${CKPT_FOLDER}"
    echo "=============================================="

    CKPT="${CKPT_FOLDER}/${CKPT_NAME}"
    CKPT_TAG="${CKPT_FOLDER}/${CKPT_NAME}"

    ANSWER_DIR="${EVAL}/vqav2/answers/${SPLIT}/${CKPT_TAG}"
    mkdir -p "${ANSWER_DIR}"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        deepspeed \
          --include localhost:${GPULIST[$IDX]} \
          --master_port $((${GPULIST[$IDX]} + 29507)) \
          moellava/eval/model_vqa_loader.py \
          --model-path "${CKPT}" \
          --question-file "${EVAL}/vqav2/${SPLIT}.jsonl" \
          --image-folder "${EVAL}/vqav2/test2015" \
          --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
          --num-chunks "${CHUNKS}" \
          --chunk-idx "${IDX}" \
          --temperature 0 \
          --conv-mode "${CONV}" &
    done

    wait

    MERGED_FILE="${ANSWER_DIR}/merge.jsonl"
    > "${MERGED_FILE}"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGED_FILE}"
    done

    python3 scripts/convert_vqav2_for_submission.py \
        --split "${SPLIT}" \
        --ckpt "${CKPT_TAG}" \
        --dir "${EVAL}/vqav2"

done