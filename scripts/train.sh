#!/bin/bash
# Training script for federated learning

# Get GPU devices from command line argument
cuda_devices=$1

# Run training
CUDA_VISIBLE_DEVICES="${cuda_devices}" python train.py \
    --exp_name 'hetero-1B' \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --data_path './data/dataset1' \
    --model_dir '/home/scratch/haoyungw/genai/' \
    --num_communication_rounds 20 \
    --num_clients 8 \
    --client_selection_frac 1 \
    --federation_mode 'hetero'