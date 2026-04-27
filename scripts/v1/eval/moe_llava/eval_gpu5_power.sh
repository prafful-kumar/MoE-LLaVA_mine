#!/bin/bash
# Continuation runner for GPU 5 — power_adaptive variants.
# Waits for the previous eval_gpu5.sh queue (PID 426755) to finish first.
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=5
LOG_DIR="logs/eval"
PREV_PID=426755

echo "======================================================"
echo "GPU ${GPU} power_adaptive queue — $(date)"
echo "Waiting for PID ${PREV_PID} (eval_gpu5.sh) to finish..."
echo "======================================================"

wait ${PREV_PID} 2>/dev/null || true
echo "PID ${PREV_PID} done (or already finished). Starting power_adaptive evals — $(date)"

echo ""
echo ">>> [1/2] qwen_power_adaptive"
bash scripts/v1/eval/moe_llava/qwen_power_adaptive_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/qwen_power_adaptive.log"
echo ">>> [1/2] DONE — $(date)"

echo ""
echo ">>> [2/2] stablelm_power_adaptive"
bash scripts/v1/eval/moe_llava/stablelm_power_adaptive_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/stablelm_power_adaptive.log"
echo ">>> [2/2] DONE — $(date)"

echo ""
echo "======================================================"
echo "ALL POWER ADAPTIVE EVALS COMPLETE — $(date)"
echo "======================================================"
