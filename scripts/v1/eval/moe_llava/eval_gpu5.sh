#!/bin/bash
# Sequential eval runner for GPU 5.
# Queue (in order):
#   1. stablelm_adaptive_entropy  — inference already done, scoring only
#   2. qwen_adaptive_entropy
#   3. qwen_double_adaptive
#   4. stablelm_double_adaptive
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=5
LOG_DIR="logs/eval"
mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "GPU ${GPU} eval queue — $(date)"
echo "======================================================"

echo ""
echo ">>> [1/4] stablelm_adaptive_entropy (scoring only)"
bash scripts/v1/eval/moe_llava/stablelm_adaptive_entropy_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/stablelm_adaptive_entropy.log"
echo ">>> [1/4] DONE — $(date)"

echo ""
echo ">>> [2/4] qwen_adaptive_entropy"
bash scripts/v1/eval/moe_llava/qwen_adaptive_entropy_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/qwen_adaptive_entropy.log"
echo ">>> [2/4] DONE — $(date)"

echo ""
echo ">>> [3/4] qwen_double_adaptive"
bash scripts/v1/eval/moe_llava/qwen_double_adaptive_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/qwen_double_adaptive.log"
echo ">>> [3/4] DONE — $(date)"

echo ""
echo ">>> [4/4] stablelm_double_adaptive"
bash scripts/v1/eval/moe_llava/stablelm_double_adaptive_all.sh ${GPU} \
    2>&1 | tee "${LOG_DIR}/stablelm_double_adaptive.log"
echo ">>> [4/4] DONE — $(date)"

echo ""
echo "======================================================"
echo "ALL EVALS COMPLETE — $(date)"
echo "======================================================"
