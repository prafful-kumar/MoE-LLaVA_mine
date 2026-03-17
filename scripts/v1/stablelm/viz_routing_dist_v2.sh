#!/bin/bash
# viz_routing_dist_v2.sh
#
# Uses:
#   - diagnostic_data_v2.json  (5 prompt types per image)
#   - model_routing_probe_v2.py (saves prompt_type field)
#   - vis_dual_routing_v2.py   (token-level top-2, --color_by option)
#
# Generates two plot sets per checkpoint per layer:
#   dual_v2/{CKPT}/dual_analysis_layer_{L}_category.png
#   dual_v2/{CKPT}/dual_analysis_layer_{L}_prompt_type.png

CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
DIAG_DATA_DIR="diagnostic_dataset"
QUESTION_FILE="${DIAG_DATA_DIR}/diagnostic_data_v2.json"
GPU=3

CKPT_FOLDERS=(
  "kmeans_5000"
  "kmeans_40000_KD_0.1_EMA_0.7_NOaux"
  "kmeans_40000_KD_0.1_EMA_0.8_NOaux"
  "kmeans_40000_KD_0.1_EMA_0.9_NOaux"
  "kmeans_40000_KD_0.1_EMA_0.999_NOaux"
  "kmeans_40000_dyn_hyp"
)

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    echo "=============================================="
    echo "Processing: ${CKPT_FOLDER}"
    echo "=============================================="

    CKPT_PATH="${CKPT_FOLDER}/${CKPT_NAME}"
    ROUTING_PT="${DIAG_DATA_DIR}/${CKPT_FOLDER}_v2.pt"
    OUT_DIR="dual_v2/${CKPT_FOLDER}"

    # Skip probe if .pt already exists
    if [ -f "${ROUTING_PT}" ]; then
        echo "  ⏭ Probe already done: ${ROUTING_PT}"
    else
        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        deepspeed --include localhost:${GPU} --master_port $((RANDOM % 10000 + 20000)) \
            moellava/eval/model_routing_probe_v2.py \
            --model-path ${CKPT_PATH} \
            --question-file ${QUESTION_FILE} \
            --image-folder "" \
            --answers-file ${DIAG_DATA_DIR}/answers_diagnostic_v2.jsonl \
            --conv-mode stablelm \
            --return_gating_logit "${ROUTING_PT}"
    fi

    # Generate plots for all MoE layers (even-numbered 0..22)
    for layer_idx in {0..22..2}; do
        for color_by in category prompt_type; do
            out_png="${OUT_DIR}/dual_analysis_layer_${layer_idx}_${color_by}.png"
            if [ -f "${out_png}" ]; then
                echo "  ⏭ Skipping layer ${layer_idx} color_by=${color_by}"
                continue
            fi
            echo "  → Layer ${layer_idx}, color_by=${color_by}"
            python moellava/vis/vis_dual_routing_v2.py \
                --input "${ROUTING_PT}" \
                --output "${OUT_DIR}" \
                --layer_idx ${layer_idx} \
                --color_by ${color_by}
        done
    done

    echo "  ✅ Done: ${CKPT_FOLDER}"
done
