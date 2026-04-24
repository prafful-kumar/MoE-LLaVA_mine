#!/bin/bash
# Orchestration script for entropy training runs
# Monitors Phi2 and Qwen, then chains StableLM
set -e
cd /scratch/prafull/MoE-LLaVA_mine

LOG_PHI="logs/train/phi_entropy.log"
LOG_QWEN="logs/train/qwen_entropy.log"

echo "[$(date)] Orchestrator started"

# ---- Wait for Phi2 entropy to finish & transfer checkpoint ----
echo "[$(date)] Waiting for Phi2 entropy training to complete..."
while true; do
    if grep -q "ALL BENCHMARKS COMPLETE\|Training completed\|train_runtime" "$LOG_PHI" 2>/dev/null; then
        break
    fi
    # Check if deepspeed processes on GPUs 4-7 are still running
    if ! pgrep -f "localhost:4,5,6,7.*train_mem" > /dev/null 2>&1; then
        echo "[$(date)] Phi2 training process ended"
        break
    fi
    sleep 300
done
echo "[$(date)] Phi2 entropy training done. Transferring checkpoint to HPC..."
rsync -av --progress checkpoints_phi_entropy/ /home/prafull/scratch/hpc/checkpoints_phi_entropy/
echo "[$(date)] Phi2 checkpoint transfer complete."

# ---- Wait for Qwen entropy to finish, then launch StableLM ----
echo "[$(date)] Waiting for Qwen entropy training to complete..."
while true; do
    if grep -q "ALL BENCHMARKS COMPLETE\|Training completed\|train_runtime" "$LOG_QWEN" 2>/dev/null; then
        break
    fi
    if ! pgrep -f "localhost:0,1,2.*train_mem" > /dev/null 2>&1; then
        echo "[$(date)] Qwen training process ended"
        break
    fi
    sleep 300
done
echo "[$(date)] Qwen entropy training done. Launching StableLM entropy..."

# ---- Launch StableLM entropy on GPUs 0,1,2 ----
bash scripts/v1/stablelm/finetune_moe_entropy.sh
echo "[$(date)] StableLM entropy training complete."
echo "[$(date)] All entropy runs finished!"
