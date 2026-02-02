# #!/bin/bash

# gpu_list="4,5"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CONV="stablelm"
# CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
# CKPT="kmeans_6k/${CKPT_NAME}"
# SPLIT="llava_gqa_testdev_balanced"
# EVAL="moellava/eval"
# GQADIR="${EVAL}/gqa/data"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29504)) moellava/eval/model_vqa_loader.py \
#         --model-path ${CKPT} \
#         --question-file ${EVAL}/gqa/$SPLIT.jsonl \
#         --image-folder ${EVAL}/gqa/data/images \
#         --answers-file ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode ${CONV} &
# done

# wait

# output_file=${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${EVAL}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# mkdir -p $GQADIR/$SPLIT/${CKPT_NAME}
# python3 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/$SPLIT/${CKPT_NAME}/testdev_balanced_predictions.json

# cd $GQADIR
# python3 eval/eval_gqa.py --tier $SPLIT/${CKPT_NAME}/testdev_balanced \
#                          --questions ${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json
#!/bin/bash
set -e

############################
# CONFIG
############################

gpu_list="4,5"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"

EVAL="moellava/eval"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${EVAL}/gqa/data"

CKPT_FOLDERS=(
  "kmeans_6k"
  "kmeans_40000_KD_first_time_NOaux"
  "random_no_KD_0_aux"
  "random_no_KD_0.1_aux"
  "random_no_KD_0.5_aux"
  "kmeans_5000"
  "kmeans_5000_KD_0.1_EMA_0.999_NOaux"
  "kmeans_5000_large_BS_KD_0.1_EMA_0.999_NOaux"
  "kmeans_20000_KD_0.1_EMA_0.999_NOaux"
  "kmeans_40000"
  "kmeans_40000_KD_second_time_NOaux"
  "random_no_KD_0.01_aux"
  "kmeans_40000_KD_0.1_EMA_0.7_NOaux"
  "kmeans_40000_KD_0.1_EMA_0.8_NOaux"
)

############################
# LOOP OVER CHECKPOINTS
############################

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    echo "=============================================="
    echo "Evaluating GQA checkpoint: ${CKPT_FOLDER}"
    echo "=============================================="

    ############################
    # MODEL PATH (AS YOU STATED)
    ############################
    CKPT="${CKPT_FOLDER}/${CKPT_NAME}"

    ############################
    # ANSWERS (EXPERIMENT-SPECIFIC)
    ############################
    ANSWER_DIR="${EVAL}/gqa/answers/${SPLIT}/${CKPT_FOLDER}/${CKPT_NAME}"
    mkdir -p "${ANSWER_DIR}"

    ############################
    # RUN DEEPSPEED CHUNKS
    ############################
    for IDX in $(seq 0 $((CHUNKS-1))); do
        deepspeed \
          --include localhost:${GPULIST[$IDX]} \
          --master_port $((${GPULIST[$IDX]} + 29504)) \
          moellava/eval/model_vqa_loader.py \
          --model-path "${CKPT}" \
          --question-file "${EVAL}/gqa/${SPLIT}.jsonl" \
          --image-folder "${EVAL}/gqa/data/images" \
          --answers-file "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" \
          --num-chunks "${CHUNKS}" \
          --chunk-idx "${IDX}" \
          --temperature 0 \
          --conv-mode "${CONV}" &
    done

    wait

    ############################
    # MERGE CHUNKS
    ############################
    MERGED_FILE="${ANSWER_DIR}/merge.jsonl"
    > "${MERGED_FILE}"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat "${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl" >> "${MERGED_FILE}"
    done

    ############################
    # CONVERT FOR GQA EVAL
    # (MATCHES YOUR DIRECTORY TREE)
    ############################
    PRED_DIR="${GQADIR}/${SPLIT}/${CKPT_FOLDER}/${CKPT_NAME}"
    mkdir -p "${PRED_DIR}"

    python3 scripts/convert_gqa_for_eval.py \
      --src "${MERGED_FILE}" \
      --dst "${PRED_DIR}/testdev_balanced_predictions.json"

    ############################
    # RUN OFFICIAL GQA EVAL
    ############################
    (
     python3 moellava/eval/eval_gqa.py \
      --tier "moellava/eval/gqa/data/${SPLIT}/${CKPT_FOLDER}/${CKPT_NAME}/testdev_balanced" \
      --questions "${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json"
    )
done
