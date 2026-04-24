#!/bin/bash
# Eval a subset of StableLM ablation checkpoints — for parallel workers
# Usage: bash scripts/v1/eval/moe_llava/stablelm_ablations_subset.sh <GPU> <NAME1> [NAME2 ...]
set -e
cd /scratch/prafull/MoE-LLaVA_mine

eval "$(conda shell.bash hook)"
conda activate moellava_mine

GPU=${1:-7}
shift
CONV="stablelm"
EVAL="moellava/eval"
LOCAL="checkpoints_stablelm_ablations"

run_eval() {
    local NAME=$1
    local CKPT="${LOCAL}/${NAME}/llava-stablelm-1.6b-finetune-moe"
    local VARIANT="stablelm_abl_${NAME}"
    local PORT

    echo "=============================================="
    echo "Eval: ${VARIANT}  GPU:${GPU}"
    echo "  Checkpoint: ${CKPT}"
    echo "=============================================="

    # Wait if checkpoint not yet available (training may still be running)
    local waited=0
    while [ ! -f "${CKPT}/config.json" ]; do
        if [ $waited -eq 0 ]; then
            echo "[WAIT] Checkpoint not ready yet: ${CKPT} — polling every 5 min..."
        fi
        sleep 300
        waited=1
    done

    # POPE
    if [ ! -f "${EVAL}/pope/answers/${VARIANT}.jsonl" ]; then
        echo ">>> POPE"
        PORT=$((RANDOM + 29503))
        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
            --image-folder "${EVAL}/pope/val2014" --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
            --temperature 0 --conv-mode "${CONV}"
    fi
    python3 moellava/eval/eval_pope.py --annotation-dir "${EVAL}/pope/coco" \
        --question-file "${EVAL}/pope/llava_pope_test.jsonl" --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"

    # TextVQA
    if [ ! -f "${EVAL}/textvqa/answers/${VARIANT}.jsonl" ]; then
        echo ">>> TextVQA"
        PORT=$((RANDOM + 29503))
        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" --question-file "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
            --image-folder "${EVAL}/textvqa/train_images" --answers-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl" \
            --temperature 0 --conv-mode "${CONV}"
    fi
    python3 -m moellava.eval.eval_textvqa \
        --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" --result-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl"

    # ScienceQA
    if [ ! -f "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" ]; then
        echo ">>> ScienceQA"
        PORT=$((RANDOM + 29503))
        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
            --model-path "${CKPT}" --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
            --image-folder "${EVAL}/scienceqa/images/test" --answers-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
            --single-pred-prompt --temperature 0 --conv-mode "${CONV}"
    fi
    python3 moellava/eval/eval_science_qa.py --base-dir "${EVAL}/scienceqa" \
        --result-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
        --output-file "${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl" \
        --output-result "${EVAL}/scienceqa/answers/${VARIANT}_result.json"

    # MME
    if [ ! -f "${EVAL}/MME/answers/${VARIANT}.jsonl" ]; then
        echo ">>> MME"
        PORT=$((RANDOM + 29503))
        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" --question-file "${EVAL}/MME/llava_mme.jsonl" \
            --image-folder "${EVAL}/MME/MME_Benchmark_release_version" --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
            --temperature 0 --conv-mode "${CONV}"
    fi
    cd "${EVAL}/MME" && python3 convert_answer_to_mme.py --experiment "${VARIANT}"
    cd eval_tool && python3 calculation.py --results_dir "answers/${VARIANT}"
    cd /scratch/prafull/MoE-LLaVA_mine

    # GQA
    if [ ! -f "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" ]; then
        echo ">>> GQA"
        PORT=$((RANDOM + 29503))
        mkdir -p "${EVAL}/gqa/answers/${VARIANT}"
        deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
            --model-path "${CKPT}" --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
            --image-folder "${EVAL}/gqa/data/images" --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
            --num-chunks 1 --chunk-idx 0 --temperature 0 --conv-mode "${CONV}"
        cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"
    fi
    python3 scripts/convert_gqa_for_eval.py \
        --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
        --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"
    python3 "${EVAL}/eval_gqa.py" --tier testdev_balanced \
        --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
        --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"

    echo "[DONE] ${VARIANT}"
    echo ""
}

for NAME in "$@"; do
    run_eval "${NAME}"
done

echo "=============================================="
echo "STABLELM SUBSET EVAL COMPLETE: $*"
echo "=============================================="
