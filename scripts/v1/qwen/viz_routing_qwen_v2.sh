#!/bin/bash
# viz_routing_qwen_v2.sh
# Run routing probe + generate plots for all 3 Qwen variants (final checkpoints)
# GPU: 4

set -e

GPU=4
DIAG_DATA_DIR="diagnostic_dataset"
QUESTION_FILE="${DIAG_DATA_DIR}/diagnostic_data_v2.json"
CKPT_NAME="llavaqwen-1.8b-finetune-moe"

declare -A VARIANTS
VARIANTS["qwen_author"]="checkpoints_qwen_author/${CKPT_NAME}"
VARIANTS["qwen_student"]="checkpoints_qwen_student/${CKPT_NAME}"
VARIANTS["qwen_TS"]="checkpoints_qwen_TS/${CKPT_NAME}"

for VARIANT in qwen_author qwen_student qwen_TS; do
    CKPT_PATH="${VARIANTS[$VARIANT]}"
    ROUTING_PT="${DIAG_DATA_DIR}/${VARIANT}_v2.pt"
    OUT_DIR="dual_v2/${VARIANT}"

    echo "=============================================="
    echo "Variant: ${VARIANT}"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "=============================================="

    # Step 1: Run probe
    if [ -f "${ROUTING_PT}" ]; then
        echo "  ⏭ Probe already exists: ${ROUTING_PT}"
    else
        echo "  🔍 Running routing probe on GPU ${GPU}..."
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        deepspeed --include localhost:${GPU} --master_port $((RANDOM + 29503)) \
            moellava/eval/model_routing_probe_v2.py \
            --model-path "${CKPT_PATH}" \
            --question-file "${QUESTION_FILE}" \
            --image-folder "" \
            --answers-file "${DIAG_DATA_DIR}/answers_${VARIANT}_v2.jsonl" \
            --conv-mode qwen \
            --return_gating_logit "${ROUTING_PT}"
        echo "  ✅ Probe saved: ${ROUTING_PT}"
    fi

    # Step 2: Generate plots for all MoE layers
    # Qwen 1.8B: 24 layers, MoE on even layers (0,2,4,...,22) = 12 MoE layers
    echo "  📊 Generating plots..."
    for layer_idx in 0 2 4 6 8 10 12 14 16 18 20 22; do
        for color_by in category prompt_type; do
            out_png="${OUT_DIR}/dual_analysis_layer_${layer_idx}_${color_by}.png"
            if [ -f "${out_png}" ]; then
                echo "    ⏭ Skipping layer ${layer_idx} color_by=${color_by}"
                continue
            fi
            echo "    → Layer ${layer_idx}, color_by=${color_by}"
            python3 moellava/vis/vis_dual_routing_v2.py \
                --input "${ROUTING_PT}" \
                --output "${OUT_DIR}" \
                --layer_idx ${layer_idx} \
                --color_by ${color_by}
        done
    done

    echo "  ✅ Done: ${VARIANT}"
    echo ""
done

echo "All variants complete."
