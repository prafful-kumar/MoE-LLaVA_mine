#!/bin/bash

JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
cd get_kmeans_centroids

# CUDA_VISIBLE_DEVICES=1 taskset -c 20-29 python get_centroids_adv.py \
#     --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-StableLM-Stage2 \
#     --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
#     --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
#     --output_file kmeans_trial/teacher_centroids_5000_large_BS.pkl \
#     --num_experts 4 \
#     --num_samples 5000 \
#     --buffer_size 40000 \
#     --version stablelm \
#     --image_aspect_ratio pad \
#     --batch_size 20

# mkdir -p fisher_directions

# CUDA_VISIBLE_DEVICES=0,1 taskset -c 20-29 python compute_fisher_directions.py \
#     --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-Qwen-1.8B-4e \
#     --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
#     --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
#     --output_file fisher_directions_qwen/5000.pkl \
#     --num_experts 4 \
#     --init_method fisher \
#     --version qwen \
#     --num_samples 5000

# mkdir -p fisher_directions_qwen

# taskset -c 20-29 deepspeed --include localhost:0 compute_fisher_directions.py \
#     --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-Qwen-1.8B-4e \
#     --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
#     --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
#     --output_file fisher_directions_qwen/5000.pkl \
#     --num_experts 4 \
#     --init_method fisher \
#     --version qwen \
#     --num_samples 5000

# mkdir -p fisher_directions_qwen_new_code
# taskset -c 20-29 deepspeed --include localhost:7 --master_port $((RANDOM % 10000 + 20000)) compute_fisher_directions_qwen.py \
#     --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-Qwen-1.8B-4e \
#     --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
#     --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
#     --output_file fisher_directions_qwen_new_code/8000.pkl \
#     --num_experts 4 \
#     --init_method fisher \
#     --version qwen \
#     --num_samples 8000

# mkdir -p fisher_directions_phi
taskset -c 20-29 deepspeed --include localhost:1 --master_port $((RANDOM % 10000 + 20000)) compute_fisher_directions_phi.py \
    --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-Phi2-2.7B-4e \
    --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
    --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
    --output_file fisher_directions_phi/20000.pkl \
    --num_experts 4 \
    --init_method fisher \
    --version phi \
    --num_samples 20000