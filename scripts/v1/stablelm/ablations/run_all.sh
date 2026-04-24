#!/bin/bash
# StableLM ablation orchestrator — launches both halves in parallel
# Half 1 (5 experiments) on GPUs 0,1,2
# Half 2 (5 experiments) on GPUs 3,4,5
# Usage: bash scripts/v1/stablelm/ablations/run_all.sh
set -e
cd /scratch/prafull/MoE-LLaVA_mine

mkdir -p logs/train

echo "Launching StableLM ablation Half 1 (GPUs 0,1,2) in background..."
bash scripts/v1/stablelm/ablations/run_half1.sh 0,1,2 &
PID1=$!

echo "Launching StableLM ablation Half 2 (GPUs 3,4,5) in background..."
bash scripts/v1/stablelm/ablations/run_half2.sh 3,4,5 &
PID2=$!

echo "Both halves running. PIDs: ${PID1} (half1), ${PID2} (half2)"
echo "Monitor with:"
echo "  tail -f logs/train/stablelm_ablation_went_001.log   # half1 first run"
echo "  tail -f logs/train/stablelm_ablation_lam_1_w01.log  # half2 first run"

wait ${PID1}
echo "Half 1 done."
wait ${PID2}
echo "Half 2 done."

echo "=============================================="
echo "ALL STABLELM ABLATIONS COMPLETE"
echo "Checkpoints under: checkpoints_stablelm_ablations/"
echo "=============================================="
