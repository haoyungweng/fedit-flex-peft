#!/bin/bash
# Generation script for federated learning models

# Get GPU devices from command line argument
cuda_devices=$1

# Loop through client IDs 0 to 7
for client_id in {0..7}
do
  echo "Running generation for client_id=${client_id}..."

  CUDA_VISIBLE_DEVICES="${cuda_devices}" python generate.py \
      --exp_name 'hetero-3B-optimalr' \
      --base_model 'meta-llama/Llama-3.2-3B' \
      --model_dir '/home/scratch/haoyungw/genai/' \
      --communication_rounds 20 \
      --test_file_path './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
      --prediction_dir './predictions' \
      --batch_size 32 \
      --client_id "${client_id}" \
      --hetlora
done
