#!/bin/bash
# Generation script for federated learning models

# Get GPU devices from command line argument
cuda_devices=$1

# Run generation
CUDA_VISIBLE_DEVICES="${cuda_devices}" python generate.py \
    --exp_name 'homo-1B' \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --model_dir '/home/scratch/haoyungw/genai/' \
    --communication_rounds 20 \
    --test_file_path './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
    --prediction_dir './predictions' \
    --batch_size 4 \
    --is_global_model
