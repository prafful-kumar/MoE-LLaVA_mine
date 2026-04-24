#!/bin/bash
# GPU 7 eval orchestrator — pending model evals then StableLM ablations
# Order: Qwen TS Schedule → Phi2 entropy-topk-var → StableLM entropy-topk-aux → StableLM entropy-topk-var → StableLM ablations (10)
# Usage: bash scripts/v1/eval/moe_llava/eval_gpu7.sh
set -e
cd /scratch/prafull/MoE-LLaVA_mine

eval "$(conda shell.bash hook)"
conda activate moellava_mine

echo "[GPU7] Starting pending model evals — $(date)"

echo ">>> [1/5] Qwen TS Schedule"
bash scripts/v1/eval/moe_llava/qwen_TS_schedule_all.sh 7

echo ">>> [2/5] Phi2 Entropy-topk-var"
bash scripts/v1/eval/moe_llava/phi_entropy_topk_var_all.sh 7

echo ">>> [3/5] StableLM Entropy-topk-aux"
bash scripts/v1/eval/moe_llava/stablelm_entropy_topk_aux_all.sh 7

echo ">>> [4/5] StableLM Entropy-topk-var"
bash scripts/v1/eval/moe_llava/stablelm_entropy_topk_var_all.sh 7

echo ">>> [5/5] StableLM ablations (10 checkpoints — will wait for training if needed)"
bash scripts/v1/eval/moe_llava/stablelm_ablations_eval.sh 7

echo "[GPU7] ALL DONE — $(date)"
