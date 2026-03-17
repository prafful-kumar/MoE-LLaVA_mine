#!/bin/bash
# Evaluate GQA on early checkpoints (steps 1-1000) for all 3 Qwen variants
# Runs all 3 variants in PARALLEL on same GPU using different master ports
# Usage: bash gqa_checkpoints_qwen.sh [GPU]

cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-4}
RESULTS_DIR="eval_results/gqa_checkpoints"
LOG_DIR="logs/eval/gqa_checkpoints"
SPLIT="llava_gqa_testdev_balanced"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

run_variant() {
    local VARIANT=$1
    local BASE_CKPT=$2
    local LOG_FILE="${LOG_DIR}/qwen_${VARIANT}.log"

    {
    echo "=============================================="
    echo "GQA Checkpoint Eval: qwen_${VARIANT}"
    echo "GPU: ${GPU}, Base: ${BASE_CKPT}"
    echo "Start: $(date)"
    echo "=============================================="

    for STEP in 1 100 200 300 400 500 600 700 800 900 1000; do
        CKPT="${BASE_CKPT}/checkpoint-${STEP}"
        TAG="qwen_${VARIANT}_step${STEP}"
        ANS_DIR="${EVAL}/gqa/answers/${TAG}"
        RESULT_FILE="${RESULTS_DIR}/qwen_${VARIANT}_step${STEP}.json"

        if [ -f "${RESULT_FILE}" ]; then
            echo "[${VARIANT}] Skipping step ${STEP} (result exists)"
            continue
        fi

        PORT=$((RANDOM % 10000 + 20000))
        echo ""
        echo "----------------------------------------------"
        echo "[${VARIANT}] Step ${STEP}, port ${PORT}"
        echo "----------------------------------------------"
        mkdir -p "${ANS_DIR}"

        HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        deepspeed --include localhost:${GPU} --master_port ${PORT} \
            moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" \
            --question-file "${EVAL}/gqa/${SPLIT}.jsonl" \
            --image-folder "${EVAL}/gqa/data/images" \
            --answers-file "${ANS_DIR}/1_0.jsonl" \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode qwen

        cp "${ANS_DIR}/1_0.jsonl" "${ANS_DIR}/merge.jsonl"

        python3 scripts/convert_gqa_for_eval.py \
            --src "${ANS_DIR}/merge.jsonl" \
            --dst "${ANS_DIR}/testdev_balanced_predictions.json"

        echo "--- GQA Scores: ${TAG} ---"
        GQA_OUTPUT=$(python3 "${EVAL}/gqa/eval_gqa.py" \
            --tier testdev_balanced \
            --questions "${EVAL}/gqa/data/questions1.2/testdev_balanced_questions.json" \
            --predictions "${ANS_DIR}/testdev_balanced_predictions.json" 2>&1)
        echo "${GQA_OUTPUT}"

        ACC=$(echo "$GQA_OUTPUT" | grep "^Accuracy:" | head -1 | awk '{print $2}' | tr -d '%')

        python3 -c "
import json
out = {
    'model': 'Qwen (1.8B)', 'variant': '${VARIANT}', 'step': ${STEP},
    'benchmark': 'GQA', 'accuracy': ${ACC:-0}
}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(out, f, indent=4)
print(f'Saved: ${TAG} -> accuracy={out[\"accuracy\"]}%')
"
    done
    echo ""
    echo "=============================================="
    echo "[${VARIANT}] DONE at $(date)"
    echo "=============================================="
    } 2>&1 | tee "${LOG_FILE}"
}

# Launch all 3 variants in parallel
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

echo "ALL QWEN GQA CHECKPOINT EVALS COMPLETE"
