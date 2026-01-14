#!/bin/bash

JSON_FOLDER="train_json"
IMAGE_FOLDER="IMAGE_FOLDER"
cd kmeans_trial

CUDA_VISIBLE_DEVICES=1 taskset -c 10-20 python get_centroids.py \
    --model_path /home/prafull/scratch/MoE-LLaVA-main/checkpoints/MoE-LLaVA-StableLM-Stage2 \
    --data_path /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/llava_image_tune_.json /home/prafull/scratch/MoE-LLaVA-main/${JSON_FOLDER}/nlp_tune.json \
    --image_folder /home/prafull/scratch/MoE-LLaVA-main/${IMAGE_FOLDER} \
    --output_file kmeans_trial/teacher_centroids_5000.pkl \
    --num_experts 4 \
    --num_samples 5000 \
    --buffer_size 40000 \
    --version stablelm \
    --image_aspect_ratio pad \
