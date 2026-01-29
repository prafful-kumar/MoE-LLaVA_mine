#!/bin/bash

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
# GPU Setup (Adjust as needed)
# Example: Running on GPUs 4,5,6,7 based on your snippet
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
PORT=$((2 + 29501))

# Paths
CENTROIDS_PATH="kmeans_trial/kmeans_trial/teacher_centroids_40000.pkl"
BASE_OUTPUT_DIR="./checkpoints/moe_priority_tests"
JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"

# Environment Variables for Offline Mode
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

# =========================================================
# 🔧 COMMON ARGUMENTS (Merged from your script)
# =========================================================
COMMON_ARGS="
    --moe_enable True
    --num_experts 4
    --top_k_experts 2
    --capacity_factor 1.5
    --moe_mode sparse
    --use_residual False
    --train_modules gate_proj up_proj down_proj wg
    --deepspeed ./scripts/zero2.json
    --model_name_or_path ../MoE-LLaVA-main/checkpoints/MoE-LLaVA-StableLM-Stage2
    --version stablelm
    --data_path ../MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json ../MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json
    --image_folder ../MoE-LLaVA-main/${IMAGE_FOLDER}
    --image_tower openai/clip-vit-large-patch14-336
    --image_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --num_train_epochs 1
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 6
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 24000
    --save_total_limit 1
    --learning_rate 2e-5
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to tensorboard
    --cache_dir ./cache_dir
"

# =========================================================
# 🧪 EXPERIMENT RUNNER FUNCTION
# =========================================================
run_experiment() {
    EXP_ID=$1          # Unique ID
    DESC=$2            # Description
    INIT_MODE=$3       # random | student_warm | teacher_kd
    KD_WEIGHT=$4       # 0.01 (only used if teacher_kd)
    AUX_COEF=$5        # 0.01 (Standard) or 0.0 (No Balancing)
    EMA_DECAY=$6       # 0.999

    # Prepend MoE-LLaVA for evaluation compatibility
    FINAL_FOLDER_NAME="MoE-LLaVA-${EXP_ID}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${FINAL_FOLDER_NAME}"

    echo "================================================================"
    echo "🚀 RUNNING: $FINAL_FOLDER_NAME"
    echo "📝 Goal: $DESC"
    echo "⚙️  Mode: $INIT_MODE | KD: $KD_WEIGHT | Aux: $AUX_COEF"
    echo "================================================================"

    mkdir -p $OUTPUT_DIR
    echo "$DESC" > "${OUTPUT_DIR}/description.txt"

    # Only pass centroids path if NOT random
    if [ "$INIT_MODE" == "random" ]; then
        CENT_ARG=""
    else
        CENT_ARG="--router_centroids_path $CENTROIDS_PATH"
    fi

    # Run DeepSpeed
    deepspeed --num_gpus $NUM_GPUS --master_port $PORT moellava/train/train_mem.py \
        $COMMON_ARGS \
        $CENT_ARG \
        --output_dir $OUTPUT_DIR \
        --router_init_mode $INIT_MODE \
        --kd_loss_weight $KD_WEIGHT \
        --router_aux_loss_coef $AUX_COEF \
        --ema_decay $EMA_DECAY \
        > "${OUTPUT_DIR}/training.log" 2>&1

    echo "✅ COMPLETED: $FINAL_FOLDER_NAME"
    echo ""
}

# =========================================================
# 🏆 PRIORITY 1: ESTABLISH BASELINE & PROPOSED
# =========================================================

# # 1. Baseline (The Standard)
# run_experiment "1_baseline_std" \
#     "BASELINE: Random Init + Standard Aux Loss" \
#     "random" 0.0 0.01 0.999

# # 2. Your Method (The Proposal)
# run_experiment "2_teacher_kd_std" \
#     "PROPOSED: Teacher KD + Standard Aux Loss" \
#     "teacher_kd" 0.1 0.01 0.999
#     # Note: I used 0.1 for KD weight based on your snippet, adjust if needed


# # =========================================================
# # 🥈 PRIORITY 2: TEST "NO AUX" (Does Teacher fix balancing?)
# # =========================================================

# # 3. Your Method + No Aux
# run_experiment "3_teacher_kd_noaux" \
#     "HYPOTHESIS: Teacher KD alone is sufficient (No Aux)" \
#     "teacher_kd" 0.1 0.0 0.999

# 4. Baseline + No Aux (Control for #3)
run_experiment "4_baseline_noaux" \
    "CONTROL: Random + No Aux (Should Collapse)" \
    "random" 0.0 0.0 0.0


# # =========================================================
# # 🥉 PRIORITY 3: ABLATION (Is it just initialization?)
# # =========================================================

# # 5. Warm Start (Just Init)
# run_experiment "5_student_warm_std" \
#     "ABLATION: Student Warm Start + Standard Aux" \
#     "student_warm" 0.0 0.01 0.999

# echo "🎉 ALL PRIORITY EXPERIMENTS FINISHED."