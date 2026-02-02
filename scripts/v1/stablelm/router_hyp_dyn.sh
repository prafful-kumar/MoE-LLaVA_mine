#!/bin/bash

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.0
JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
router_centroids_path="get_kmeans_centroids/kmeans_trial/teacher_centroids_40000.pkl"

# Define your dynamic hyperparameters here
TEMP_START=2.0
TEMP_END=1.0
WEIGHT_START=0.1
WEIGHT_END=0.01
EMA_START=0.999
EMA_END=0.9
TOTAL_STEPS=13860

EXP_ID="KMeans40k"
HYPERPARAMS="T${TEMP_START}_${TEMP_END}-W${WEIGHT_START}_${WEIGHT_END}-E${EMA_START}_${EMA_END}"
OUTPUT_DIR="./DYN_HYP_${EXP_ID}-${HYPERPARAMS}"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:2,3,4,5 --master_port $((2 + 29503)) moellava/train/train_mem.py \
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
    --output_dir ${OUTPUT_DIR}/MoE-LLaVA-StableLM-Stage2-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --router_init_mode teacher_kd \
    --router_temp_start $TEMP_START \
    --router_temp_end $TEMP_END \
    --router_weight_start $WEIGHT_START \
    --router_weight_end $WEIGHT_END \
    --router_ema_start $EMA_START \
    --router_ema_end $EMA_END \
    --router_total_steps $TOTAL_STEPS
