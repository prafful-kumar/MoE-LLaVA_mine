#!/bin/bash

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
GPU_IDS="0,1,2,3,4,5,6,7" 
CENTROIDS_PATH="./kmeans_trial/teacher_centroids_40000.pkl"
BASE_OUTPUT_DIR="./checkpoints/moe_ablation_study"

# Common args for all runs
# Note: We do NOT set --router_aux_loss_coef here, it will be passed dynamically
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
# EXPERIMENT FUNCTION
# =========================================================
run_experiment() {
    # Arguments passed to function
    EXP_ID=$1          # Folder Name
    DESC=$2            # Description for logs
    INIT_MODE=$3       # random | student_warm | teacher_kd
    KD_WEIGHT=$4       # 0.0 for baselines, 0.01+ for KD
    AUX_COEF=$5        # 0.01 (Standard) or 0.0 (No Load Balancing)
    EMA_DECAY=$6       # 0.999 usually

    echo "================================================================"
    echo "🚀 STARTING EXPERIMENT: $EXP_ID"
    echo "📝 Description: $DESC"
    echo "⚙️  Config: Init=$INIT_MODE | KD=$KD_WEIGHT | Aux=$AUX_COEF"
    echo "================================================================"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_ID}"
    mkdir -p $OUTPUT_DIR
    
    # Save the description to a text file for easy reference later
    echo "$DESC" > "${OUTPUT_DIR}/experiment_description.txt"

    # Construct Centroids Argument
    # Only pass path if mode is NOT random
    if [ "$INIT_MODE" == "random" ]; then
        CENTROIDS_ARG=""
    else
        CENTROIDS_ARG="--router_centroids_path $CENTROIDS_PATH"
    fi

    # RUN COMMAND
    deepspeed --include localhost:$GPU_IDS moellava/train/train_mem.py \
        $COMMON_ARGS \
        $CENTROIDS_ARG \
        --output_dir $OUTPUT_DIR \
        --router_init_mode $INIT_MODE \
        --kd_loss_weight $KD_WEIGHT \
        --router_aux_loss_coef $AUX_COEF \
        --ema_decay $EMA_DECAY \
        > "${OUTPUT_DIR}/training.log" 2>&1

    echo "✅ FINISHED: $EXP_ID"
    echo ""
}

# =========================================================
# GROUP 1: BASELINES (Random Init)
# =========================================================

# 1a. Standard Baseline (Random + Aux Loss)
# Reference for normal performance
run_experiment "1a_baseline_std" "Random Init, Standard Aux Loss (0.01)" \
    "random" 0.0 0.01 0.999

# 1b. No-Aux Baseline (Random + No Aux)
# EXPECTATION: Expert Collapse (Router chooses only 1 expert forever)
run_experiment "1b_baseline_noaux" "Random Init, NO Aux Loss (0.0)" \
    "random" 0.0 0.0 0.999


# =========================================================
# GROUP 2: WARM START (Student Centroids, No Teacher)
# =========================================================

# 2a. Warm Start + Aux
# Does "good initialization" help when we still force balancing?
run_experiment "2a_warm_std" "Student Centroids, Standard Aux Loss (0.01)" \
    "student_warm" 0.0 0.01 0.999

# 2b. Warm Start + No Aux
# Can the initialization ALONE keep experts diverse without balancing loss?
run_experiment "2b_warm_noaux" "Student Centroids, NO Aux Loss (0.0)" \
    "student_warm" 0.0 0.0 0.999


# =========================================================
# GROUP 3: YOUR METHOD (Teacher KD)
# =========================================================

# 3a. KD + Aux (The "Safe" Proposed Method)
# Teacher guides semantic clustering, Aux guarantees balancing
run_experiment "3a_kd_std" "Teacher KD, Standard Aux Loss (0.01)" \
    "teacher_kd" 0.01 0.01 0.999

# 3b. KD + No Aux (The "Pure" Proposed Method)
# HYPOTHESIS: Teacher provides enough signal to prevent collapse 
# without needing the artificial Aux loss.
run_experiment "3b_kd_noaux" "Teacher KD, NO Aux Loss (0.0)" \
    "teacher_kd" 0.01 0.0 0.999


# =========================================================
# GROUP 4: ABLATION (High Pressure) - Optional
# =========================================================

# 4. High KD Weight + No Aux
# If 3b fails, maybe we just need a stronger Teacher signal?
run_experiment "4_high_kd_noaux" "High KD (0.1), NO Aux Loss" \
    "teacher_kd" 0.1 0.0 0.999

echo "🎉 ALL ABLATION STUDIES COMPLETED."