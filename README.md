# FedIT-flex-PEFT

A flexible framework for federated learning with parameter-efficient fine-tuning of language models.

## Project Overview

This framework enables federated learning for fine-tuning language models using Low-Rank Adaptation (LoRA). It supports:

- Homogeneous federated learning (`homo`): All clients use the same LoRA rank
- Heterogeneous federated learning (`hetero`): Clients use different LoRA ranks
- Non-federated training (`none`): Each client trains independently

## Workflow

The project follows a three-stage workflow:

1. **Train**: Fine-tune models using `train.py`
2. **Generate**: Produce outputs with `generate.py`
3. **Evaluate**: Measure performance with `evaluate.py`

## Command Line Arguments

### Train

```bash
python train.py \
    --exp_name NAME \              # Experiment name
    --base_model MODEL_PATH \      # Base model to fine-tune
    --data_path DATA_PATH \        # Path to training data
    --model_dir MODEL_DIR \        # Where to save trained models
    --federation_mode MODE         # 'homo', 'hetero', or 'none'
```

### Generate

```bash
python generate.py \
    --exp_name NAME \              # Experiment name
    --base_model MODEL_PATH \      # Base model
    --model_dir MODEL_DIR \        # Where trained models are saved
    --prediction_dir PRED_DIR \    # Where to save predictions
    --is_global_model              # Use global model (or specify client_id)
```

### Evaluate

```bash
python evaluate.py \
    --exp_name NAME \              # Experiment name
    --target_file TARGET_PATH \    # Reference outputs
    --prediction_dir PRED_DIR \    # Where predictions are saved
    --evaluation_dir EVAL_DIR \    # Where to save evaluation metrics
    --communication_rounds N \     # Number of communication rounds
    [--client_id CLIENT_ID]        # Optional: evaluate specific client
```

## Running the Project

Run the three scripts in sequence:

```bash
# Step 1: Train the models
./scripts/train.sh 0        # Pass GPU device as argument

# Step 2: Generate outputs
./scripts/generate.sh 0     # Pass GPU device as argument

# Step 3: Evaluate results
./scripts/evaluate.sh
```