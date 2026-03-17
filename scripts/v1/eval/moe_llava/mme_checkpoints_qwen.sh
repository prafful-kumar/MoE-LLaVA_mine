#!/bin/bash
# Evaluate MME on early checkpoints (steps 1-1000) for all 3 Qwen variants
# Runs all 3 variants in PARALLEL on same GPU using different master ports
# Usage: bash mme_checkpoints_qwen.sh [GPU]

cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-4}
RESULTS_DIR="eval_results/mme_checkpoints"
LOG_DIR="logs/eval/mme_checkpoints"
STEPS=(1 100 200 300 400 500 600 700 800 900 1000)
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

run_variant() {
    local VARIANT=$1
    local BASE_CKPT=$2
    local LOG_FILE="${LOG_DIR}/qwen_${VARIANT}.log"

    {
    echo "=============================================="
    echo "MME Checkpoint Eval: qwen_${VARIANT}"
    echo "GPU: ${GPU}, Base: ${BASE_CKPT}"
    echo "Start: $(date)"
    echo "=============================================="
    echo "[${VARIANT}] Starting MME checkpoint eval..."

    for STEP in 1 100 200 300 400 500 600 700 800 900 1000; do
        CKPT="${BASE_CKPT}/checkpoint-${STEP}"
        TAG="qwen_${VARIANT}_step${STEP}"
        ANSWER_FILE="${EVAL}/MME/answers/${TAG}.jsonl"
        RESULT_FILE="${RESULTS_DIR}/qwen_${VARIANT}_step${STEP}.json"

        if [ -f "${RESULT_FILE}" ]; then
            echo "[${VARIANT}] Skipping step ${STEP} (result exists)"
            continue
        fi

        PORT=$((RANDOM % 10000 + 20000))
        echo "[${VARIANT}] Step ${STEP}, port ${PORT}"

        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        deepspeed --include localhost:${GPU} --master_port ${PORT} \
            moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" \
            --question-file "${EVAL}/MME/llava_mme.jsonl" \
            --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
            --answers-file "${ANSWER_FILE}" \
            --temperature 0 \
            --conv-mode qwen

        # Convert to MME format (correct invocation: only --experiment)
        (cd "${EVAL}/MME" && python3 convert_answer_to_mme.py --experiment "${TAG}")

        # Score and extract totals
        echo "--- MME Scores: ${TAG} ---"
        MME_OUTPUT=$(cd "${EVAL}/MME/eval_tool" && python3 calculation.py --results_dir "answers/${TAG}" 2>&1)
        echo "${MME_OUTPUT}"
        PERCEPTION=$(echo "$MME_OUTPUT" | grep "total score:" | awk 'NR==1{print $3}' | tr -d ' ')
        COGNITION=$(echo "$MME_OUTPUT"  | grep "total score:" | awk 'NR==2{print $3}' | tr -d ' ')
        TOTAL=$(python3 -c "print(round(${PERCEPTION:-0} + ${COGNITION:-0}, 4))" 2>/dev/null)

        python3 -c "
import json
out = {
    'model': 'Qwen (1.8B)', 'variant': '${VARIANT}', 'step': ${STEP},
    'benchmark': 'MME',
    'total': ${TOTAL:-0}, 'perception': ${PERCEPTION:-0}, 'cognition': ${COGNITION:-0}
}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(out, f, indent=4)
print(f'[${VARIANT}] step${STEP}: total={out[\"total\"]}')
"
    done
    echo ""
    echo "=============================================="
    echo "[${VARIANT}] DONE at $(date)"
    echo "=============================================="
    } 2>&1 | tee "${LOG_FILE}"
}

# Launch all 3 variants in parallel (each gets its own random ports per step)
run_variant "author"          "checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe" &
PID_AUTHOR=$!

run_variant "student"         "checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe" &
PID_STUDENT=$!

run_variant "teacher_student" "/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe" &
PID_TS=$!

echo "Launched in parallel on GPU ${GPU}:"
echo "  author          PID=$PID_AUTHOR"
echo "  student         PID=$PID_STUDENT"
echo "  teacher_student PID=$PID_TS"

wait $PID_AUTHOR;          echo "author: done"
wait $PID_STUDENT;         echo "student: done"
wait $PID_TS;              echo "teacher_student: done"

echo "ALL QWEN MME CHECKPOINT EVALS COMPLETE"
