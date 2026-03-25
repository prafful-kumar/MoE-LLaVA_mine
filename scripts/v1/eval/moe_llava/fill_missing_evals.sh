#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-5}
EVAL="moellava/eval"

run_pope() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo ">>> POPE — $VARIANT"
    PORT=$((RANDOM + 29503))
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
        --image-folder "${EVAL}/pope/val2014" \
        --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
        --temperature 0 --conv-mode "${CONV}"
    echo "--- POPE Results ---"
    python3 moellava/eval/eval_pope.py \
        --annotation-dir "${EVAL}/pope/coco" \
        --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
        --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"
}

run_gqa() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo ">>> GQA — $VARIANT"
    PORT=$((RANDOM + 29503))
    mkdir -p "${EVAL}/gqa/answers/${VARIANT}"
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
        --image-folder "${EVAL}/gqa/data/images" \
        --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
        --num-chunks 1 --chunk-idx 0 \
        --temperature 0 --conv-mode "${CONV}"
    cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"
    python3 scripts/convert_gqa_for_eval.py \
        --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
        --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"
    echo "--- GQA Results ---"
    python3 "${EVAL}/eval_gqa.py" \
        --tier testdev_balanced \
        --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
        --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"
}

run_textvqa() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo ">>> TextVQA — $VARIANT"
    PORT=$((RANDOM + 29503))
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
        --image-folder "${EVAL}/textvqa/train_images" \
        --answers-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl" \
        --temperature 0 --conv-mode "${CONV}"
    echo "--- TextVQA Results ---"
    python3 -m moellava.eval.eval_textvqa \
        --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl"
}

run_sciqa() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo ">>> ScienceQA — $VARIANT"
    PORT=$((RANDOM + 29503))
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
        --image-folder "${EVAL}/scienceqa/images/test" \
        --answers-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
        --single-pred-prompt --temperature 0 --conv-mode "${CONV}"
    echo "--- ScienceQA Results ---"
    python3 moellava/eval/eval_science_qa.py \
        --base-dir "${EVAL}/scienceqa" \
        --result-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
        --output-file "${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl" \
        --output-result "${EVAL}/scienceqa/answers/${VARIANT}_result.json"
}

run_mme() {
    local CKPT=$1 CONV=$2 VARIANT=$3
    echo ">>> MME — $VARIANT"
    PORT=$((RANDOM + 29503))
    deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
        --model-path "${CKPT}" \
        --question-file "${EVAL}/MME/llava_mme.jsonl" \
        --image-folder "${EVAL}/MME/MME_Benchmark_release_version" \
        --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
        --temperature 0 --conv-mode "${CONV}"
    echo "--- MME Results ---"
    cd "${EVAL}/MME"
    python3 convert_answer_to_mme.py --experiment "${VARIANT}"
    cd eval_tool
    python3 calculation.py --results_dir "answers/${VARIANT}"
    cd /scratch/prafull/MoE-LLaVA_mine
}

echo "=============================================="
echo "FILLING MISSING BENCHMARK EVALUATIONS"
echo "GPU: ${GPU}"
echo "=============================================="

# ---- 1. Qwen author: POPE + GQA ----
echo ""
echo "====== QWEN AUTHOR (POPE + GQA) ======"
CKPT="/home/prafull/scratch/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe"
run_pope "$CKPT" "qwen" "qwen_author"
run_gqa  "$CKPT" "qwen" "qwen_author"

# ---- 2. Qwen student: POPE + GQA ----
echo ""
echo "====== QWEN STUDENT (POPE + GQA) ======"
CKPT="/home/prafull/scratch/hpc/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe"
run_pope "$CKPT" "qwen" "qwen_student"
run_gqa  "$CKPT" "qwen" "qwen_student"

# ---- 3. Qwen TS: POPE + GQA ----
echo ""
echo "====== QWEN TS (POPE + GQA) ======"
CKPT="/home/prafull/scratch/hpc/checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe"
run_pope "$CKPT" "qwen" "qwen_TS"
run_gqa  "$CKPT" "qwen" "qwen_TS"

# ---- 4. Phi2 TS: ALL 5 benchmarks ----
echo ""
echo "====== PHI2 TS (ALL 5 BENCHMARKS) ======"
CKPT="/home/prafull/scratch/hpc/checkpoints_phi_TS/llavaphi-2.7b-finetune-moe"
run_pope    "$CKPT" "phi" "phi_TS"
run_textvqa "$CKPT" "phi" "phi_TS"
run_sciqa   "$CKPT" "phi" "phi_TS"
run_mme     "$CKPT" "phi" "phi_TS"
run_gqa     "$CKPT" "phi" "phi_TS"

# ---- 5. Phi2 student: POPE + GQA ----
# NOTE: phi_student checkpoint does not exist, skipping

# ---- 6. Phi2 author: POPE + GQA (already has POPE, need GQA) ----
echo ""
echo "====== PHI2 AUTHOR (GQA only) ======"
CKPT="/home/prafull/scratch/hpc/checkpoints_phi/llavaphi-2.7b-finetune-moe"
run_gqa "$CKPT" "phi" "phi_author"

echo ""
echo "=============================================="
echo "ALL MISSING EVALUATIONS COMPLETE"
echo "=============================================="
