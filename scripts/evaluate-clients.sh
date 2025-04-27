#!/bin/bash
# Evaluation script for federated learning models

# Loop through client IDs 0 to 7
for client_id in {0..7}
do
    echo "Running evaluation for client_id=${client_id}..."

    python metric.py \
        --exp_name 'hetero-3B-optimalr' \
        --target_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
        --target_key 'output' \
        --prediction_dir './predictions' \
        --prediction_key 'answer' \
        --evaluation_dir './evaluations_final' \
        --communication_rounds 20 \
        --client_id "${client_id}" \
        --hetlora
done
