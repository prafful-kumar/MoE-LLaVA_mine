#!/bin/bash
# =============================================================================
# StableLM Power-Adaptive v4: Heavy Entropy + Heavy L_var
# =============================================================================
#
# Experiment name: stablelm_power_adaptive_v4
# Based on:        v2 (register_parametrization gradient fix)
#
# What changed vs v2:
#   - entropy_loss_weight: 0.1 → 0.3   (3x heavier adaptive entropy)
#   - balance_loss_weight:  0.1 → 0.3   (3x heavier L_var)
#   - router_aux_loss_coef: 0.0 (unchanged — aux loss off, same as v2)
#
# Why: v1/v2 both showed expert collapse (cosine sim=0.9994 after 1 epoch).
#      This ablation tests whether stronger regularization pressure can break
#      symmetry and force expert divergence. Both terms increased together to
#      keep their ratio fixed (1:1) while boosting total balance signal.
#
# Loss (v4): L_total = 0.3*(L_leak + L_adaptive) + 0.3*L_var
#            where alpha = (1 - prob_margin)^2  [power, gamma=2.0]
#
# Loss (v2): L_total = 0.1*(L_leak + L_adaptive) + 0.1*L_var
#
# To restore v2 weights: set entropy_loss_weight=0.1, balance_loss_weight=0.1
#
# GPUs: 2, 3, 4
# Output: checkpoints_stablelm_power_adaptive_v4/
# =============================================================================

eval "$(conda shell.bash hook)"
conda activate moellava_mine

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.00
ENTROPY_LOSS_WEIGHT=0.3    # heavy (v2 was 0.1)
BALANCE_LOSS_WEIGHT=0.3    # heavy (v2 was 0.1)
ADAPTIVE_GAMMA=2.0
ALPHA_MODE="power"
JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
router_centroids_path="get_kmeans_centroids/fisher_directions/5000.pkl"
ROUTER_INIT_MODE="no_teacher"

OUTPUT_DIR="./checkpoints_stablelm_power_adaptive_v4/llava-stablelm-1.6b-finetune-moe"

# Safety check: never overwrite existing checkpoint
if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists: $OUTPUT_DIR"
    echo "Aborting to avoid overwriting checkpoint."
    exit 1
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:2,3,4 --master_port $((RANDOM % 10000 + 20000)) moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules gate_proj up_proj down_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ../MoE-LLaVA-main/checkpoints/MoE-LLaVA-StableLM-Stage2 \
    --version stablelm \
    --data_path ../MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json ../MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
    --image_folder ../MoE-LLaVA-main/${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --router_centroids_path ${router_centroids_path} \
    --router_init_mode ${ROUTER_INIT_MODE} \
    --entropy_loss_weight ${ENTROPY_LOSS_WEIGHT} \
    --balance_loss_weight ${BALANCE_LOSS_WEIGHT} \
    --use_adaptive_entropy True \
    --alpha_mode ${ALPHA_MODE} \
    --adaptive_gamma ${ADAPTIVE_GAMMA} 2>&1 | tee logs/train/stablelm_power_adaptive_v4.log
