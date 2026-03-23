#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-0}
VARIANT=${2:-"stablelm_student"}
CKPT=${3:-"checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe"}
CONV="stablelm"
EVAL="moellava/eval"

echo "=============================================="
echo "StableLM MME + GQA"
echo "  Variant:    ${VARIANT}"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

# ---- MME ----
echo ">>> MME"
PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/MME/llava_mme.jsonl" \
    --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
    --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV}"

echo "--- MME Results ---"
cd "${EVAL}/MME"
python3 convert_answer_to_mme.py --experiment "${VARIANT}"
cd eval_tool
python3 calculation.py --results_dir "answers/${VARIANT}"
cd /scratch/prafull/MoE-LLaVA_mine

# ---- GQA ----
echo ""
echo ">>> GQA"
PORT=$((RANDOM + 29503))
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${EVAL}/gqa/data"

mkdir -p "${EVAL}/gqa/answers/${VARIANT}"

deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" \
    --question-file "${EVAL}/gqa/${SPLIT}.jsonl" \
    --image-folder "${EVAL}/gqa/data/images" \
    --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode "${CONV}"

cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"

python3 scripts/convert_gqa_for_eval.py \
    --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
    --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"

echo "--- GQA Results ---"
python3 "${EVAL}/gqa/eval_gqa.py" \
    --tier testdev_balanced \
    --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
    --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"

echo ""
echo "=============================================="
echo "MME + GQA COMPLETE FOR ${VARIANT}"
echo "=============================================="
