#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

EVAL="moellava/eval"
GPU=${1:-6}
RESULTS_DIR="eval_results/sqa_checkpoints"
STEPS=(1 100 200 300 400 500 600 700 800 900 1000)

declare -A VARIANTS
VARIANTS=(
    ["author"]="/scratch/prafull/hpc/checkpoints_phi/llavaphi-2.7b-finetune-moe|phi"
    ["student"]="checkpoints_phi_student/llavaphi-2.7b-finetune-moe|phi"
)

for VARIANT in "${!VARIANTS[@]}"; do
    IFS='|' read -r BASE_CKPT CONV <<< "${VARIANTS[$VARIANT]}"

    for STEP in "${STEPS[@]}"; do
        CKPT="${BASE_CKPT}/checkpoint-${STEP}"
        TAG="phi2_${VARIANT}_step${STEP}"
        ANSWER_FILE="${EVAL}/scienceqa/answers/${TAG}.jsonl"
        RESULT_FILE="${RESULTS_DIR}/${TAG}.json"

        # Skip if result already exists
        if [ -f "${RESULT_FILE}" ]; then
            echo "Skipping ${TAG} (result exists)"
            continue
        fi

        PORT=$((RANDOM + 29503))
        echo "=============================================="
        echo "SQA: ${TAG}"
        echo "  Model: ${CKPT}"
        echo "  Conv: ${CONV}, GPU: ${GPU}, Port: ${PORT}"
        echo "=============================================="

        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
            --model-path "${CKPT}" \
            --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
            --image-folder "${EVAL}/scienceqa/images/test" \
            --answers-file "${ANSWER_FILE}" \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode "${CONV}"

        echo "--- Evaluating ${TAG} ---"
        python3 moellava/eval/eval_science_qa.py \
            --base-dir "${EVAL}/scienceqa" \
            --result-file "${ANSWER_FILE}" \
            --output-file "${EVAL}/scienceqa/answers/${TAG}_output.jsonl" \
            --output-result "${EVAL}/scienceqa/answers/${TAG}_result.json"

        # Extract accuracy and save
        python3 -c "
import json, sys
with open('${EVAL}/scienceqa/answers/${TAG}_result.json') as f:
    res = json.load(f)
out = {
    'model': 'Phi2 (2.7B)',
    'variant': '${VARIANT}',
    'step': ${STEP},
    'benchmark': 'ScienceQA',
    'accuracy': res.get('accuracy', res.get('avg', {}).get('accuracy', None)),
    'img_accuracy': res.get('img_accuracy', res.get('img', {}).get('accuracy', None))
}
with open('${RESULT_FILE}', 'w') as f:
    json.dump(out, f, indent=4)
print(f'Saved: ${TAG} -> acc={out[\"accuracy\"]}, img_acc={out[\"img_accuracy\"]}')
"
        echo ""
    done
done

echo "=============================================="
echo "ALL PHI2 SQA CHECKPOINT EVALS COMPLETE"
echo "=============================================="
