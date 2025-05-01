# FedIT-flex-PEFT

A flexible framework for federated learning with parameter-efficient fine-tuning (PEFT) of language models using LoRA.

## Acknowledgements

This project builds upon the codebase of [FederatedGPT-Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd).
The data used is a subset of the FLAN collection, sourced from [FedDPA](https://github.com/Lydia-yang/FedDPA).

## Project Overview

This framework enables fine-tuning language models using Low-Rank Adaptation (LoRA) in various federated and isolated settings. It supports four distinct modes controlled by the `use_federation` and `use_hetero` flags:

1.  **Federated Homogeneous (FedAvg):**
    *   `--use_federation=True`
    *   `--use_hetero=False`
    *   Clients train with the same LoRA rank (`--lora_r`). Models are aggregated using FedAvg.
2.  **Federated Heterogeneous (HetLoRA):**
    *   `--use_federation=True`
    *   `--use_hetero=True`
    *   Clients train with potentially different LoRA ranks (hardcoded in `train.py`). Models are aggregated using the HetLoRA algorithm (sparsity-weighted).
3.  **Isolated Clients (Homogeneous Ranks):**
    *   `--use_federation=False`
    *   `--use_hetero=False`
    *   Clients train independently with the same LoRA rank (`--lora_r`). No aggregation occurs. Each client saves its own model history.
4.  **Isolated Clients (Heterogeneous Ranks):**
    *   `--use_federation=False`
    *   `--use_hetero=True`
    *   Clients train independently with potentially different LoRA ranks (hardcoded in `train.py`). No aggregation occurs. Each client saves its own model history.

## Workflow

The project follows a three-stage workflow:

1.  **Train**: Fine-tune models using `train.py`.
2.  **Generate**: Produce model outputs using `generate.py`.
3.  **Evaluate**: Measure performance using `evaluate.py`.

## Command Line Arguments

### Train (`train.py`)

```bash
python train.py \
    --exp_name NAME \               # Experiment name (e.g., 'fedavg_r4')
    --base_model MODEL_PATH \       # Base model ID (e.g., 'meta-llama/Llama-3.2-1B')
    --data_path DATA_PATH \         # Path to parent data directory (e.g., './data')
    --model_dir MODEL_DIR \         # Directory to save trained models (e.g., './models')
    --num_clients N \               # Total number of clients (e.g., 8)
    --num_communication_rounds R \  # Number of training rounds (e.g., 20)
    --client_selection_frac F \     # Fraction of clients selected per round (e.g., 0.5)
    --local_num_epochs E \          # Number of local epochs per client (e.g., 3)
    --lora_r RANK \                 # Base LoRA rank (used when use_hetero=False)
    # --- Mode Control ---
    --use_federation BOOL \         # True for FedAvg/HetLoRA, False for Isolated Clients
    --use_hetero BOOL \             # True for Heterogeneous Ranks, False for Homogeneous Ranks
    # --- Other options ---
    --local_batch_size BATCH \
    --local_micro_batch_size MICRO_BATCH \
    --local_learning_rate LR \
    ... # Other LoRA and training parameters
```

**Example Modes:**

*   **FedAvg:** `--use_federation=True --use_hetero=False --lora_r=4`
*   **HetLoRA:** `--use_federation=True --use_hetero=True` (Base `--lora_r` ignored, uses hardcoded ranks)
*   **Isolated Homo:** `--use_federation=False --use_hetero=False --lora_r=4`
*   **Isolated Hetero:** `--use_federation=False --use_hetero=True` (Base `--lora_r` ignored, uses hardcoded ranks)

### Generate (`generate.py`)

```bash
python generate.py \
    --exp_name NAME \               # Experiment name (must match training)
    --base_model MODEL_PATH \       # Base model ID
    --model_dir MODEL_DIR \         # Directory where trained models are saved
    --prediction_dir PRED_DIR \     # Directory to save predictions (e.g., './predictions')
    --communication_rounds R \      # Specify round to load model from (e.g., 19 for last round)
    --test_file_path TEST_DATA \    # Path to the test dataset (e.g., './data/test.jsonl')
    # --- Specify Model ---
    [--is_global_model] \           # Use the aggregated global model (if use_federation=True during training)
    # OR
    [--client_id ID] \              # Use a specific client's model (if use_federation=False during training)
    # OR
    [--hetlora]                     # Use the HetLoRA global model with task-specific truncation for combined output.
    ... # Other generation parameters
```

### Evaluate (`evaluate.py`)

```bash
python evaluate.py \
    --exp_name NAME \               # Experiment name
    --target_file TARGET_PATH \     # Path to reference outputs (e.g., './data/test.jsonl')
    --prediction_dir PRED_DIR \     # Directory where predictions are saved
    --evaluation_dir EVAL_DIR \     # Directory to save evaluation metrics (e.g., './evaluations')
    --communication_rounds N \      # Evaluate model from this round
    [--client_id CLIENT_ID]         # Optional: evaluate specific client's predictions
```

## Running the Project

The provided scripts (`train.sh`, `generate.sh`, `evaluate.sh`) execute a specific, hardcoded configuration (e.g., Homogeneous Federated Learning for `train.sh` and `generate.sh`).

**To run the hardcoded configuration using the scripts:**

```bash
# Step 1: Train using the configuration in train.sh (pass GPU ID)
./scripts/train.sh 0

# Step 2: Generate using the configuration in generate.sh (pass GPU ID)
./scripts/generate.sh 0

# Step 3: Evaluate using the configuration in evaluate.sh
./scripts/evaluate.sh
```

**To run a custom configuration (Recommended):**

Run the Python scripts directly, providing the desired arguments as described in the "Command Line Arguments" section.

```bash
# Example: Train Federated Heterogeneous (HetLoRA)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --exp_name 'FL-hetero-1B' \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --data_path './data/dataset1' \
    --model_dir './models' \
    --num_communication_rounds 20 \
    --num_clients 8 \
    --client_selection_frac 1 \
    --use_federation=True \
    --use_hetero=True \
    --local_num_epochs 3 \
    # ... other desired arguments

# Example: Generate using the HetLoRA approach for the last round (19)
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --exp_name 'FL-hetero-1B' \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --model_dir './models' \
    --prediction_dir './predictions' \
    --communication_rounds 19 \
    --test_file_path './data/test.jsonl' \
    --hetlora \
    # ... other desired arguments

# Example: Evaluate the HetLoRA predictions for the last round (19)
python evaluate.py \
    --exp_name 'FL-hetero-1B' \
    --target_file './data/test.jsonl' \
    --prediction_dir './predictions' \
    --evaluation_dir './evaluations' \
    --communication_rounds 19 \
    # ... other desired arguments
```