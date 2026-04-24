#!/bin/bash
# Margin-Aware Adaptive Entropy Loss for Phi2 2.7B.
# Uses use_adaptive_entropy=True with gamma=2.0.
# Confident tokens (high prob_margin) get suppressed entropy penalty.
# Uncertain tokens get full KL(p_tilde || uniform_2) penalty.
# See normalized_router_flexible.py::SimplifiedNormalizedGate for formulation.
#
# NOTE: Uses 5000-point Fisher centroids. Re-train with 20000-point centroids
# once compute_fisher_directions_phi.py is run with --num_samples 20000.
# (20000-point default is already set in the script.)

eval "$(conda shell.bash hook)"
conda activate moellava_mine

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.00
JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
router_centroids_path="get_kmeans_centroids/fisher_directions_phi/5000.pkl"
ROUTER_INIT_MODE="no_teacher"
ENTROPY_LOSS_WEIGHT=0.03
ADAPTIVE_GAMMA=2.0

OUTPUT_DIR="./checkpoints_phi_adaptive_entropy/llavaphi-2.7b-finetune-moe"

# Safety check: never overwrite existing checkpoint
if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists: $OUTPUT_DIR"
    echo "Aborting to avoid overwriting checkpoint."
    exit 1
fi

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:0,1,2,3 --master_port $((RANDOM % 10000 + 20000)) moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ../MoE-LLaVA-main/checkpoints/MoE-LLaVA-Phi2-Stage2 \
    --version phi \
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
    --per_device_train_batch_size 1 \
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
    --use_adaptive_entropy True \
    --adaptive_gamma ${ADAPTIVE_GAMMA} 2>&1 | tee logs/train/phi_adaptive_entropy.log
