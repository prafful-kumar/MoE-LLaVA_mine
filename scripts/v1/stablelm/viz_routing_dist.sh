# 1. Set your paths (Adjust if your checkpoint name is different)
CKPT_NAME="MoE-LLaVA-StableLM-Stage2-moe"
CKPT_PATH="kmeans_5000/${CKPT_NAME}"
DIAG_DATA_DIR="diagnostic_dataset_claude"

# 2. Run the Probe
# We use deepspeed (single GPU) to handle the model loading efficiently.
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:4 moellava/eval/model_routing_probe.py \
    --model-path ${CKPT_PATH} \
    --question-file ${DIAG_DATA_DIR}/diagnostic_data.json \
    --image-folder "" \
    --answers-file ${DIAG_DATA_DIR}/answers_diagnostic.jsonl \
    --conv-mode stablelm \
    --return_gating_logit "${DIAG_DATA_DIR}/diagnostic_routing"