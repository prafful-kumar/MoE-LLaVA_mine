#!/bin/bash
set -e

############################
# FIXED WORKING DIRECTORY
############################

REPO_ROOT="/scratch/prafull/MoE-LLaVA_mine"
cd "${REPO_ROOT}"

############################
# CONFIG
############################

CONV="stablelm"
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
EVAL="moellava/eval"

CKPT_FOLDERS=(
#   "kmeans_6k"
#   "kmeans_40000_KD_first_time_NOaux"
#   "random_no_KD_0_aux"
#   "random_no_KD_0.1_aux"
#   "random_no_KD_0.5_aux"
#   "kmeans_5000"
#   "kmeans_5000_KD_0.1_EMA_0.999_NOaux"
#   "kmeans_5000_large_BS_KD_0.1_EMA_0.999_NOaux"
#   "kmeans_20000_KD_0.1_EMA_0.999_NOaux"
#   "kmeans_40000"
#   "kmeans_40000_KD_second_time_NOaux"
#   "random_no_KD_0.01_aux"
#   "kmeans_40000_KD_0.1_EMA_0.7_NOaux"
#   "kmeans_40000_KD_0.1_EMA_0.8_NOaux"
#   "kmeans_40000_dyn_hyp"
#   "DYN_HYP_KMeans40k-T1.0_0.6-W0.1_0.01-E0.999_0.5"
#   "DYN_HYP_KMeans40k-T1.0_0.6-W0.1_0.01-E0.999_0.9"
#   "DYN_HYP_KMeans40k-T2.0_0.6-W0.1_0.01-E0.999_0.9"
#   "DYN_HYP_KMeans40k-T2.0_1.0-W0.1_0.01-E0.999_0.9"
# "training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.7"
# "training_norm_./DYN_HYP_KMeans40k-T1.0_0.6-W0.1_0.01-E0.999_0.7"
"student_only_training_norm_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.7" #1360.37
"fisher_no_norm_input_norm_weight_during_training_./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998" #1345.11
"fisher_init_TS_no_training_norm_OnlyInitNorm./DYN_HYP_Fisher5k-T1.0_0.6-W0.1_0.01-E0.999_0.998" #1273.63
)

############################
# LOOP OVER CHECKPOINTS
############################

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    echo "=============================================="
    echo "Running MME for checkpoint: ${CKPT_FOLDER}"
    echo "=============================================="

    CKPT="${CKPT_FOLDER}/${CKPT_NAME}"
    ANSWER_FILE="${EVAL}/MME/answers/${CKPT_FOLDER}_${CKPT_NAME}.jsonl"

    deepspeed --include localhost:0 --master_port $((RANDOM % 10000 + 20000)) moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/MME/llava_mme.jsonl" \
        --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
        --answers-file "${ANSWER_FILE}" \
        --temperature 0 \
        --conv-mode "${CONV}"

    cd "${EVAL}/MME"

    python convert_answer_to_mme.py \
        --experiment "${CKPT_FOLDER}_${CKPT_NAME}"

    cd eval_tool

    python calculation.py \
        --results_dir "answers/${CKPT_FOLDER}_${CKPT_NAME}"

    cd "${REPO_ROOT}"

done


