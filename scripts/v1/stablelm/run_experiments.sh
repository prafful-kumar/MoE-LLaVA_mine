#!/bin/bash

# =========================================================
# CONFIGURATION
# =========================================================
GPU_IDS="0,1,2,3,4,5,6,7" # Adjust based on your available GPUs
CENTROIDS_PATH="./kmeans_trial/teacher_centroids_40000.pkl"
BASE_OUTPUT_DIR="./checkpoints/moe_experiments"

# Define the common training arguments (adjust paths as needed)
COMMON_ARGS="
    --moe_enable True
    --num_experts 4
    --top_k_experts 2
    --moe_mode sparse
    --model_name_or_path ../MoE-LLaVA-main/checkpoints/MoE-LLaVA-StableLM-Stage2
    --data_path ./your_data.json
    --image_folder ./your_images
    --bf16 True
    --num_train_epochs 1
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 2
    --save_strategy "steps"
    --save_steps 2000
    --save_total_limit 1
"

# =========================================================
# EXPERIMENT LOOP
# =========================================================

run_experiment() {
    EXP_ID=$1
    DESC=$2
    EXTRA_ARGS=$3

    echo "----------------------------------------------------------------"
    echo "Running Experiment: $EXP_ID"
    echo "Description: $DESC"
    echo "----------------------------------------------------------------"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_ID}"
    mkdir -p $OUTPUT_DIR

    deepspeed --include localhost:$GPU_IDS moellava/train/train_mem.py \
        $COMMON_ARGS \
        --output_dir $OUTPUT_DIR \
        $EXTRA_ARGS \
        > "${OUTPUT_DIR}/training.log" 2>&1

    echo "✅ Finished Experiment: $EXP_ID"
}

# 1. Baseline (No Centroids, No KD)
# We pass no centroids path, ensuring random init
run_experiment "exp1_baseline" "Vanilla MoE (Random Init)" \
    "--kd_loss_weight 0.0"

# 2. Warm Start (Student Init, No KD)
# Requires the code tweak mentioned in Part 3
run_experiment "exp2_warm_start" "Student Initialized with Centroids" \
    "--router_centroids_path $CENTROIDS_PATH --router_init_mode student_warm --kd_loss_weight 0.0"

# 3. Your Method (Teacher Init, KD, EMA 0.999)
# This is the 'Star' experiment
run_experiment "exp3_teacher_kd_main" "KD Training (EMA 0.999, W=0.01)" \
    "--router_centroids_path $CENTROIDS_PATH --router_init_mode teacher_kd --kd_loss_weight 0.01 --ema_decay 0.999"

# 4. Static Teacher (EMA 1.0)
# Teacher never updates
run_experiment "exp4_static_teacher" "Static Teacher (Fixed Centroids)" \
    "--router_centroids_path $CENTROIDS_PATH --router_init_mode teacher_kd --kd_loss_weight 0.01 --ema_decay 1.0"

# 5. High KD Weight (Ablation)
# Stronger forcing
run_experiment "exp5_high_kd" "High KD Weight (0.1)" \
    "--router_centroids_path $CENTROIDS_PATH --router_init_mode teacher_kd --kd_loss_weight 0.1 --ema_decay 0.999"

echo "ALL EXPERIMENTS COMPLETED."