#!/bin/bash
# GPU 6 eval orchestrator — Qwen ablations (10 checkpoints)
# Usage: bash scripts/v1/eval/moe_llava/eval_gpu6.sh
set -e
cd /scratch/prafull/MoE-LLaVA_mine

eval "$(conda shell.bash hook)"
conda activate moellava_mine

echo "[GPU6] Starting Qwen ablation eval — $(date)"

bash scripts/v1/eval/moe_llava/qwen_ablations_eval.sh 6

echo "[GPU6] ALL DONE — $(date)"
