#!/bin/bash
# wbal ablation experiments on GPUs 2,3,4
# Runs: wbal_0_w001, wbal_0_w01, wbal_01_w001, wbal_01_w01
# Skip logic: skips if checkpoint dir already exists (safe to run alongside 5,6,7 orchestrator)
# Usage: bash scripts/v1/qwen/ablations/run_wbal_234.sh [GPU_LIST]
set -e
cd /scratch/prafull/MoE-LLaVA_mine

eval "$(conda shell.bash hook)"
conda activate moellava_mine

GPUS=${1:-2,3,4}
BASE_CKPT="checkpoints_qwen_ablations"
MODEL="../MoE-LLaVA-main/checkpoints/MoE-LLaVA-Qwen-Stage2"
CENTROIDS="get_kmeans_centroids/fisher_directions_qwen/5000.pkl"

mkdir -p logs/train

run_ablation() {
    local SHORT_NAME=$1
    local W_ENT=$2
    local IMBAL_LAM=$3
    local W_BAL=$4

    local OUTDIR="./${BASE_CKPT}/${SHORT_NAME}/llavaqwen-1.8b-finetune-moe"
    local LOG="logs/train/ablation_${SHORT_NAME}.log"
    local PORT=$((RANDOM + 29503))

    if [ -d "${OUTDIR}" ]; then
        echo "[SKIP] ${SHORT_NAME} — checkpoint already exists at ${OUTDIR}"
        return
    fi

    echo "=============================================="
    echo "Ablation: ${SHORT_NAME}  GPUs:${GPUS}  Port:${PORT}"
    echo "  w_ent=${W_ENT}  lam=${IMBAL_LAM}  w_bal=${W_BAL}"
    echo "  Output: ${OUTDIR}"
    echo "=============================================="

    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed \
        --include localhost:${GPUS} --master_port ${PORT} \
        moellava/train/train_mem.py \
        --moe_enable True --num_experts 4 --top_k_experts 2 --capacity_factor 1.5 \
        --moe_mode sparse --use_residual False --router_aux_loss_coef 0.00 \
        --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path ${MODEL} \
        --version qwen \
        --data_path ../MoE-LLaVA-main/train_json/llava_image_tune_.json ../MoE-LLaVA-main/train_json/nlp_tune.json \
        --image_folder ../MoE-LLaVA-main/IMAGE_FOLDER \
        --image_tower openai/clip-vit-large-patch14-336 \
        --image_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${OUTDIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 12 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
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
        --router_centroids_path ${CENTROIDS} \
        --router_init_mode no_teacher \
        --entropy_loss_weight ${W_ENT} \
        --imbal_lam ${IMBAL_LAM} \
        --balance_loss_weight ${W_BAL} 2>&1 | tee ${LOG}

    echo "[DONE] ${SHORT_NAME}"
    echo ""
}

# GPUs 2,3,4 handle: wbal_0_w001 and wbal_01_w01
# GPUs 5,6,7 (run_all.sh) handle: wbal_0_w01 and wbal_01_w001 after lam_1_w01 finishes
run_ablation wbal_0_w001   0.01  0.1   0.0
run_ablation wbal_01_w01   0.1   0.1   0.1

echo "=============================================="
echo "wbal ABLATIONS COMPLETE (GPUs 2,3,4 subset)"
echo "Checkpoints under: ${BASE_CKPT}/"
echo "=============================================="
