#!/bin/bash
# Run OLD routing probe + OLD vis_dual_routing.py on 3 Qwen variants
# Uses 3-category diagnostic dataset (animal, food, scene) for clarity
# Output: dual_old/{qwen_author,qwen_student,qwen_TS}/dual_analysis_layer_N.png

set -e
cd /scratch/prafull/MoE-LLaVA_mine

DIAG_DATA_DIR="diagnostic_dataset"
QUESTION_FILE="${DIAG_DATA_DIR}/diagnostic_data_3cls.json"
GPU=4

declare -A VARIANTS
VARIANTS["qwen_author"]="checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe"
VARIANTS["qwen_student"]="checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe"
VARIANTS["qwen_TS"]="checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe"

for VARIANT in qwen_author qwen_student qwen_TS; do
    CKPT_PATH="${VARIANTS[$VARIANT]}"
    ROUTING_PT="${DIAG_DATA_DIR}/${VARIANT}_old_3cls.pt"
    OUT_DIR="dual_old/${VARIANT}"

    echo "=============================================="
    echo "Variant: ${VARIANT}"
    echo "=============================================="

    # Step 1: Run routing probe
    echo "  Running probe..."
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    deepspeed --include localhost:${GPU} --master_port $((RANDOM % 10000 + 20000)) \
        moellava/eval/model_routing_probe.py \
        --model-path "${CKPT_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "" \
        --answers-file "${DIAG_DATA_DIR}/answers_${VARIANT}_old_3cls.jsonl" \
        --conv-mode qwen \
        --return_gating_logit "${ROUTING_PT}"

    echo "  Probe saved: ${ROUTING_PT}"

    # Step 2: Generate plots for all 12 MoE layers (even indices 0..22)
    echo "  Generating plots..."
    for layer_idx in 0 2 4 6 8 10 12 14 16 18 20 22; do
        echo "    -> Layer ${layer_idx}"
        python3 moellava/vis/vis_dual_routing.py \
            --input "${ROUTING_PT}" \
            --output "${OUT_DIR}" \
            --layer_idx ${layer_idx}
    done

    echo "  Done: ${VARIANT} -> ${OUT_DIR}/"
done

echo ""
echo "All done. Plots in dual_old/"
