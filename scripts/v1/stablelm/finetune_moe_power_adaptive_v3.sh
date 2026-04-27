#!/bin/bash
# =============================================================================
# StableLM Power-Adaptive v3: Replace L_var with DeepSpeed Aux Loss
# =============================================================================
#
# Experiment name: stablelm_power_adaptive_v3
# Based on:        v2 (register_parametrization gradient fix)
#
# What changed vs v2:
#   - REMOVED:  balance_loss_weight * L_var  (set balance_loss_weight=0.0)
#               L_var = (1/E)*sum_i(m_i - 1/E)^2  — soft variance penalty on
#               per-expert mean routing probabilities. Encourages uniform load.
#   - ADDED:    router_aux_loss_coef=0.1  (DeepSpeed standard aux loss)
#               aux = sum_i f_i * P_i  where f_i = fraction of tokens dispatched
#               to expert i, P_i = mean routing probability to expert i.
#               Weight = 0.1 (same as balance_loss_weight was in v2).
#
# Why: Both L_var and DeepSpeed aux loss encourage load balance, but via
#      different mechanisms. This ablation isolates whether the benefit of
#      the balance term in v1/v2 came from L_var specifically or whether the
#      standard aux loss (which also has a gradient signal to the router) does
#      the same job — or better, since aux loss is the de-facto standard.
#
# Loss (v3): L_total = 0.1*aux + 0.1*(L_leak + L_adaptive)
#            where alpha = (1 - prob_margin)^2  [power, gamma=2.0]
#
# Loss (v2): L_total = 0.0*aux + 0.1*(L_leak + L_adaptive) + 0.1*L_var
#
# To restore L_var: set balance_loss_weight > 0.0 and router_aux_loss_coef=0.0
# See: normalized_router_flexible.py  [L_var toggle] comments.
#
# GPUs: 5, 6, 7
# Output: checkpoints_stablelm_power_adaptive_v3/
# =============================================================================

eval "$(conda shell.bash hook)"
conda activate moellava_mine

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.10    # DeepSpeed standard aux loss — replaces L_var
balance_loss_weight=0.0      # L_var disabled (set > 0 to re-enable)
ENTROPY_LOSS_WEIGHT=0.1
ADAPTIVE_GAMMA=2.0
ALPHA_MODE="power"
JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
router_centroids_path="get_kmeans_centroids/fisher_directions/5000.pkl"
ROUTER_INIT_MODE="no_teacher"

OUTPUT_DIR="./checkpoints_stablelm_power_adaptive_v3/llava-stablelm-1.6b-finetune-moe"

# Safety check: never overwrite existing checkpoint
if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists: $OUTPUT_DIR"
    echo "Aborting to avoid overwriting checkpoint."
    exit 1
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:5,6,7 --master_port $((RANDOM % 10000 + 20000)) moellava/train/train_mem.py \
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
    --balance_loss_weight ${balance_loss_weight} \
    --use_adaptive_entropy True \
    --alpha_mode ${ALPHA_MODE} \
    --adaptive_gamma ${ADAPTIVE_GAMMA} 2>&1 | tee logs/train/stablelm_power_adaptive_v3.log
