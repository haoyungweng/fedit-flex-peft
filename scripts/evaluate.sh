#!/bin/bash
# Evaluation script for federated learning models

# Run evaluation
python evaluate.py \
    --exp_name 'fedavg-1B' \
    --target_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
    --target_key 'output' \
    --prediction_dir './predictions' \
    --prediction_key 'answer' \
    --evaluation_dir './evaluations' \
    --communication_rounds 20