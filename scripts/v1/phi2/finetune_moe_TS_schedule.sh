#!/bin/bash
# Phi2 TS with KD Weight Schedule
# KD weight 0.05 → 0.01, temp flat 1.0 → 1.0, EMA 0.999 → 0.95

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
ROUTER_INIT_MODE="teacher_kd"

ROUTER_TEMP_START=1.0
ROUTER_TEMP_END=1.0
ROUTER_WEIGHT_START=0.05
ROUTER_WEIGHT_END=0.01
ROUTER_EMA_START=0.999
ROUTER_EMA_END=0.95

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:4,5,6,7 --master_port $((19 + 29503)) moellava/train/train_mem.py \
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
    --output_dir ./checkpoints_phi_TS_schedule/llavaphi-2.7b-finetune-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 20 \
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
    --router_temp_start ${ROUTER_TEMP_START} \
    --router_temp_end ${ROUTER_TEMP_END} \
    --router_weight_start ${ROUTER_WEIGHT_START} \
    --router_weight_end ${ROUTER_WEIGHT_END} \
    --router_ema_start ${ROUTER_EMA_START} \
    --router_ema_end ${ROUTER_EMA_END} 2>&1 | tee logs/train/phi_TS_schedule.log
