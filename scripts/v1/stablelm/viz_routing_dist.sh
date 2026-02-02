# #!/bin/bash

# CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
# CKPT_FOLDER="kmeans_40000_KD_0.1_EMA_0.9_NOaux"
# CKPT_PATH="${CKPT_FOLDER}/${CKPT_NAME}"
# DIAG_DATA_DIR="diagnostic_dataset"
# ROUTING_PT="${DIAG_DATA_DIR}/${CKPT_FOLDER}.pt"

# # Run routing probe (unchanged)
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:1 --master_port $((12 + 29503)) moellava/eval/model_routing_probe.py \
#     --model-path ${CKPT_PATH} \
#     --question-file ${DIAG_DATA_DIR}/diagnostic_data.json \
#     --image-folder "" \
#     --answers-file ${DIAG_DATA_DIR}/answers_diagnostic.jsonl \
#     --conv-mode stablelm \
#     --return_gating_logit "${ROUTING_PT}"

# # Use improved visualization
# for layer_idx in {0..22..2}; do
#     echo "Analyzing layer ${layer_idx}"
#     python moellava/vis/vis_dual_routing.py \
#         --input ${ROUTING_PT} \
#         --output dual_${CKPT_FOLDER} \
#         --layer_idx ${layer_idx}
# done

#!/bin/bash

CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
DIAG_DATA_DIR="diagnostic_dataset"

CKPT_FOLDERS=(
  "kmeans_6k"
  "kmeans_40000_KD_first_time_NOaux"
  "random_no_KD_0_aux"
  "random_no_KD_0.1_aux"
  "random_no_KD_0.5_aux"
  "kmeans_5000"
  "kmeans_5000_KD_0.1_EMA_0.999_NOaux"
  "kmeans_5000_large_BS_KD_0.1_EMA_0.999_NOaux"
  "kmeans_20000_KD_0.1_EMA_0.999_NOaux"
  "kmeans_40000"
  "kmeans_40000_KD_second_time_NOaux"
  "random_no_KD_0.01_aux"
  "kmeans_40000_KD_0.1_EMA_0.7_NOaux"
  "kmeans_40000_KD_0.1_EMA_0.8_NOaux"
  "kmeans_40000_dyn_hyp"
# "MoE-LLaVA-StableLM-Stage2-moe_6k_0.1"
)

for CKPT_FOLDER in "${CKPT_FOLDERS[@]}"; do
    echo "=============================================="
    echo "Processing checkpoint folder: ${CKPT_FOLDER}"
    echo "=============================================="

    CKPT_PATH="${CKPT_FOLDER}/${CKPT_NAME}"
    ROUTING_PT="${DIAG_DATA_DIR}/${CKPT_FOLDER}.pt"

    # 1. Run routing probe (once per checkpoint)
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    deepspeed --include localhost:5 --master_port $((RANDOM % 10000 + 20000)) \
        moellava/eval/model_routing_probe.py \
        --model-path ${CKPT_PATH} \
        --question-file ${DIAG_DATA_DIR}/diagnostic_data.json \
        --image-folder "" \
        --answers-file ${DIAG_DATA_DIR}/answers_diagnostic.jsonl \
        --conv-mode stablelm \
        --return_gating_logit "${ROUTING_PT}"

    # 2. Visualization for all even layers
    for layer_idx in {0..22..2}; do
        echo "  → Analyzing layer ${layer_idx}"
        python moellava/vis/vis_dual_routing.py \
            --input ${ROUTING_PT} \
            --output dual/${CKPT_FOLDER} \
            --layer_idx ${layer_idx}
    done
done
