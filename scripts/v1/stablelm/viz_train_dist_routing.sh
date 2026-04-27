#!/bin/bash
# viz_train_dist_routing.sh
#
# Analyzes routing specialization by ACTUAL TRAINING DATA SOURCE,
# not a hand-crafted diagnostic set.
#
# For each checkpoint listed in CKPT_FOLDERS:
#   1. model_routing_probe_train_dist.py  — forward pass, saves .pt
#   2. vis_train_dist_routing.py          — generates three plots:
#        specialization_heatmap.png  (Fig A)
#        specialization_score.png    (Fig B)
#        best_layer_detail.png       (Fig C)
#
# Also runs the existing vis_dual_routing_v2.py on the same .pt
# (color_by=category) for individual-layer detail at any layer of interest.
#
# Output hierarchy:
#   train_dist_analysis/<CKPT_FOLDER>/
#     train_dist_<CKPT_FOLDER>.pt
#     specialization_heatmap.png
#     specialization_score.png
#     best_layer_detail.png
#     dual_detail/
#       dual_analysis_layer_<L>_category.png  (for selected layers)

eval "$(conda shell.bash hook)"
conda activate moellava_mine

# ── Configuration ────────────────────────────────────────────────────────────
GPU=3
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"

DATA_PATHS=(
  "../MoE-LLaVA-main/train_json/llava_image_tune_.json"
  "../MoE-LLaVA-main/train_json/nlp_tune.json"
)
IMAGE_FOLDER="../MoE-LLaVA-main/IMAGE_FOLDER"

SAMPLES_PER_SOURCE=80        # per source; total ≈ 80×6 = 480 samples → fast
CONV_MODE="stablelm"

# Layers to generate dual_detail plots for (the most specialized ones,
# check specialization_score.png first; update these after the first run)
DETAIL_LAYERS=(4 8 12 16 20)

CKPT_FOLDERS=(
  "checkpoints_stablelm_power_adaptive/llava-stablelm-1.6b-finetune-moe"
  # Add other checkpoints to compare, e.g. a baseline random-init run:
  # "hpc/random_no_KD_0.01_aux/MoE-LLaVA-StableLM-Stage2-moe"
)

# ── Main loop ─────────────────────────────────────────────────────────────────
for CKPT_PATH in "${CKPT_FOLDERS[@]}"; do
    CKPT_FOLDER=$(basename "$(dirname "${CKPT_PATH}")")
    OUT_DIR="train_dist_analysis/${CKPT_FOLDER}"
    ROUTING_PT="${OUT_DIR}/train_dist_${CKPT_FOLDER}.pt"
    DUAL_OUT="${OUT_DIR}/dual_detail"

    echo "=============================================="
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Output:     ${OUT_DIR}"
    echo "=============================================="

    mkdir -p "${OUT_DIR}" "${DUAL_OUT}"

    # ── Step 1: Probe ──────────────────────────────────────────────────────
    if [ -f "${ROUTING_PT}" ]; then
        echo "  Probe already done: ${ROUTING_PT}"
    else
        echo "  Running probe..."
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        deepspeed --include localhost:${GPU} --master_port $((RANDOM % 10000 + 20000)) \
            moellava/eval/model_routing_probe_train_dist.py \
            --model-path "${CKPT_PATH}" \
            --data-paths "${DATA_PATHS[@]}" \
            --image-folder "${IMAGE_FOLDER}" \
            --samples-per-source ${SAMPLES_PER_SOURCE} \
            --conv-mode ${CONV_MODE} \
            --return-gating-logit "${ROUTING_PT}" \
            --seed 42
        if [ $? -ne 0 ]; then
            echo "  ERROR: probe failed for ${CKPT_PATH}"
            continue
        fi
    fi

    # ── Step 2: Multi-layer summary plots ─────────────────────────────────
    if [ -f "${OUT_DIR}/specialization_score.png" ]; then
        echo "  Summary plots already done: ${OUT_DIR}/"
    else
        echo "  Generating summary plots..."
        python moellava/vis/vis_train_dist_routing.py \
            --input "${ROUTING_PT}" \
            --output "${OUT_DIR}" \
            --ckpt-name "${CKPT_FOLDER}"
    fi

    # ── Step 3: Per-layer dual detail (reuses vis_dual_routing_v2.py) ────
    echo "  Generating dual detail for layers: ${DETAIL_LAYERS[*]}..."
    for layer_idx in "${DETAIL_LAYERS[@]}"; do
        out_png="${DUAL_OUT}/dual_analysis_layer_${layer_idx}_category.png"
        if [ -f "${out_png}" ]; then
            echo "    Skipping layer ${layer_idx} (already done)"
            continue
        fi
        echo "    Layer ${layer_idx}..."
        python moellava/vis/vis_dual_routing_v2.py \
            --input "${ROUTING_PT}" \
            --output "${DUAL_OUT}" \
            --layer_idx ${layer_idx} \
            --color_by category
    done

    echo "  Done: ${CKPT_FOLDER}"
done

echo ""
echo "All done. Results in train_dist_analysis/"
