#!/bin/bash
set -e
cd /scratch/prafull/MoE-LLaVA_mine

GPU=${1:-0}
CKPT="/scratch/prafull/hpc/checkpoints_phi_entropy_topk_var/llavaphi-2.7b-finetune-moe"
CONV="phi"
EVAL="moellava/eval"
VARIANT="phi_entropy_topk_var"

echo "=============================================="
echo "Phi2 Entropy-topk-var — All benchmarks"
echo "  Checkpoint: ${CKPT}"
echo "  GPU: ${GPU}"
echo "=============================================="

echo ">>> POPE"
PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
    --image-folder "${EVAL}/pope/val2014" --answers-file "${EVAL}/pope/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
python3 moellava/eval/eval_pope.py --annotation-dir "${EVAL}/pope/coco" \
    --question-file "${EVAL}/pope/llava_pope_test.jsonl" --result-file "${EVAL}/pope/answers/${VARIANT}.jsonl"

echo ">>> TextVQA"
PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" --question-file "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "${EVAL}/textvqa/train_images" --answers-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
python3 -m moellava.eval.eval_textvqa \
    --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" --result-file "${EVAL}/textvqa/answers/${VARIANT}.jsonl"

echo ">>> ScienceQA"
PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_science.py \
    --model-path "${CKPT}" --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
    --image-folder "${EVAL}/scienceqa/images/test" --answers-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --single-pred-prompt --temperature 0 --conv-mode "${CONV}"
python3 moellava/eval/eval_science_qa.py --base-dir "${EVAL}/scienceqa" \
    --result-file "${EVAL}/scienceqa/answers/${VARIANT}.jsonl" \
    --output-file "${EVAL}/scienceqa/answers/${VARIANT}_output.jsonl" \
    --output-result "${EVAL}/scienceqa/answers/${VARIANT}_result.json"

echo ">>> MME"
PORT=$((RANDOM + 29503))
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" --question-file "${EVAL}/MME/llava_mme.jsonl" \
    --image-folder "${EVAL}/MME/MME_Benchmark_release_version" --answers-file "${EVAL}/MME/answers/${VARIANT}.jsonl" \
    --temperature 0 --conv-mode "${CONV}"
cd "${EVAL}/MME" && python3 convert_answer_to_mme.py --experiment "${VARIANT}"
cd eval_tool && python3 calculation.py --results_dir "answers/${VARIANT}"
cd /scratch/prafull/MoE-LLaVA_mine

echo ">>> GQA"
PORT=$((RANDOM + 29503))
mkdir -p "${EVAL}/gqa/answers/${VARIANT}"
deepspeed --include localhost:${GPU} --master_port ${PORT} moellava/eval/model_vqa_loader.py \
    --model-path "${CKPT}" --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "${EVAL}/gqa/data/images" --answers-file "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" \
    --num-chunks 1 --chunk-idx 0 --temperature 0 --conv-mode "${CONV}"
cp "${EVAL}/gqa/answers/${VARIANT}/1_0.jsonl" "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl"
python3 scripts/convert_gqa_for_eval.py \
    --src "${EVAL}/gqa/answers/${VARIANT}/merge.jsonl" \
    --dst "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json"
python3 "${EVAL}/eval_gqa.py" --tier testdev_balanced \
    --predictions "${EVAL}/gqa/answers/${VARIANT}/testdev_balanced_predictions.json" \
    --questions "moellava/eval/gqa/data/questions1.2/testdev_balanced_questions.json"

echo "=============================================="
echo "ALL BENCHMARKS COMPLETE FOR ${VARIANT}"
echo "=============================================="
